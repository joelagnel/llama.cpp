#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

bool decode_tokens(llama_context *                  ctx,
                   const std::vector<llama_token> & tokens,
                   llama_seq_id                     seq_id,
                   llama_pos                        pos_start,
                   bool                             logits_last) {
    const int32_t capacity = std::min<int32_t>(256, llama_n_batch(ctx));
    llama_batch   batch    = llama_batch_init(capacity, 0, 1);

    for (size_t offset = 0; offset < tokens.size(); offset += capacity) {
        common_batch_clear(batch);
        const size_t count = std::min<size_t>(capacity, tokens.size() - offset);
        for (size_t i = 0; i < count; ++i) {
            const bool logits = logits_last && offset + i + 1 == tokens.size();
            common_batch_add(batch, tokens[offset + i], pos_start + offset + i, { seq_id }, logits);
        }

        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return false;
        }
    }

    llama_synchronize(ctx);
    llama_batch_free(batch);
    return true;
}

bool wait_for_state(llama_kv_swap * swap, llama_seq_id seq_id, llama_kv_swap_state wanted) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
        llama_kv_swap_poll(swap);
        if (llama_kv_swap_seq_get_status(swap, seq_id).state == wanted) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

std::vector<uint8_t> save_sequence(llama_context * ctx, llama_seq_id seq_id) {
    std::vector<uint8_t> data(llama_state_seq_get_size(ctx, seq_id));
    const size_t         written = llama_state_seq_get_data(ctx, data.data(), data.size(), seq_id);
    if (written != data.size()) {
        data.clear();
    }
    return data;
}

}  // namespace

int main(int argc, char ** argv) {
    common_params params;
    params.kv_unified   = true;
    params.n_parallel   = 2;
    params.n_ctx        = 1024;
    params.n_batch      = 256;
    params.n_ubatch     = 256;
    params.n_gpu_layers = -2;
    params.fit_params   = false;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();
    common_init_result_ptr init  = common_init_from_params(params);
    llama_model *          model = init ? init->model() : nullptr;
    llama_context *        ctx   = init ? init->context() : nullptr;
    if (!model || !ctx) {
        fprintf(stderr, "test-kv-swap: failed to initialize the model\n");
        return 1;
    }

    char reason[256] = {};
    if (!llama_kv_swap_supported(ctx, reason, sizeof(reason))) {
        fprintf(stderr, "test-kv-swap: SKIP: %s\n", reason);
        return 77;
    }

    std::unique_ptr<llama_kv_swap, decltype(&llama_kv_swap_free)> swap(llama_kv_swap_init(ctx, 512ull * 1024 * 1024),
                                                                       llama_kv_swap_free);
    if (!swap) {
        fprintf(stderr, "test-kv-swap: failed to initialize swap\n");
        return 1;
    }

    std::string text;
    for (int i = 0; i < 32; ++i) {
        text += "alpha beta gamma delta epsilon zeta eta theta; ";
    }
    std::vector<llama_token> prompt = common_tokenize(ctx, text, true);
    if (prompt.size() > 192) {
        prompt.resize(192);
    }
    if (prompt.size() < 128 || !decode_tokens(ctx, prompt, 0, 0, false)) {
        fprintf(stderr, "test-kv-swap: failed to decode the reference prompt\n");
        return 1;
    }

    const auto reference_state = save_sequence(ctx, 0);
    if (reference_state.empty()) {
        fprintf(stderr, "test-kv-swap: failed to serialize the reference state\n");
        return 1;
    }

    std::vector<llama_token> probe_tokens = common_tokenize(ctx, " probe", false);
    if (probe_tokens.empty()) {
        fprintf(stderr, "test-kv-swap: failed to tokenize the probe\n");
        return 1;
    }
    probe_tokens.resize(1);

    if (!decode_tokens(ctx, probe_tokens, 0, prompt.size(), true)) {
        fprintf(stderr, "test-kv-swap: failed to decode the reference probe\n");
        return 1;
    }
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> greedy(llama_sampler_init_greedy(),
                                                                         llama_sampler_free);
    const llama_token reference_token = llama_sampler_sample(greedy.get(), ctx, -1);

    const size_t restored = llama_state_seq_set_data(ctx, reference_state.data(), reference_state.size(), 0);
    if (restored != reference_state.size()) {
        fprintf(stderr, "test-kv-swap: failed to restore the pre-probe state\n");
        return 1;
    }

    if (llama_kv_swap_seq_out(swap.get(), 0, 1) != LLAMA_KV_SWAP_RESULT_OK) {
        fprintf(stderr, "test-kv-swap: D2H submission failed: %s\n", llama_kv_swap_get_error(swap.get()));
        return 1;
    }

    // Decode another sequence while D2H is pending to exercise compute/copy
    // overlap without touching the evicted sequence's cells.
    std::vector<llama_token> other(prompt.begin(), prompt.begin() + 16);
    if (!decode_tokens(ctx, other, 1, 0, false)) {
        fprintf(stderr, "test-kv-swap: concurrent decode during D2H failed\n");
        return 1;
    }
    if (!wait_for_state(swap.get(), 0, LLAMA_KV_SWAP_STATE_HOST)) {
        fprintf(stderr, "test-kv-swap: timed out waiting for D2H\n");
        return 1;
    }

    // The eviction moves the allocation head to the newly freed front of the
    // cache.  Occupy part of it so sequence 0 must restore at different,
    // fragmented arbitrary cell indices.
    std::vector<llama_token> other_tail(prompt.begin() + 16, prompt.begin() + 48);
    if (!decode_tokens(ctx, other_tail, 1, other.size(), false)) {
        fprintf(stderr, "test-kv-swap: failed to fragment the cache before H2D\n");
        return 1;
    }

    if (llama_kv_swap_seq_in(swap.get(), 0) != LLAMA_KV_SWAP_RESULT_OK ||
        !wait_for_state(swap.get(), 0, LLAMA_KV_SWAP_STATE_RESIDENT)) {
        fprintf(stderr, "test-kv-swap: H2D restoration failed: %s\n", llama_kv_swap_get_error(swap.get()));
        return 1;
    }

    const auto swapped_state = save_sequence(ctx, 0);
    if (swapped_state != reference_state) {
        size_t mismatch = 0;
        while (mismatch < swapped_state.size() && mismatch < reference_state.size() &&
               swapped_state[mismatch] == reference_state[mismatch]) {
            ++mismatch;
        }
        fprintf(stderr, "test-kv-swap: byte mismatch after restore at offset %zu (%zu vs %zu bytes)\n", mismatch,
                swapped_state.size(), reference_state.size());
        return 1;
    }

    if (!decode_tokens(ctx, probe_tokens, 0, prompt.size(), true)) {
        fprintf(stderr, "test-kv-swap: failed to decode the restored probe\n");
        return 1;
    }
    const llama_token swapped_token = llama_sampler_sample(greedy.get(), ctx, -1);
    if (swapped_token != reference_token) {
        fprintf(stderr, "test-kv-swap: continuation mismatch: %d != %d\n", swapped_token, reference_token);
        return 1;
    }

    const auto stats = llama_kv_swap_get_stats(swap.get());
    if (stats.pages_out == 0 || stats.pages_in == 0 || stats.d2h_bytes == 0 || stats.h2d_bytes == 0) {
        fprintf(stderr, "test-kv-swap: transfer counters were not updated\n");
        return 1;
    }

    fprintf(stderr,
            "test-kv-swap: SUCCESS: %zu state bytes and continuation token %d are identical; "
            "%llu D2H / %llu H2D bytes\n",
            reference_state.size(), reference_token, (unsigned long long) stats.d2h_bytes,
            (unsigned long long) stats.h2d_bytes);
    printf(
        "KV_SWAP_CORRECTNESS {\"byte_identical\":true,\"continuation_identical\":true,"
        "\"state_bytes\":%zu,\"continuation_token\":%d,\"d2h_bytes\":%llu,\"h2d_bytes\":%llu}\n",
        reference_state.size(), reference_token, (unsigned long long) stats.d2h_bytes,
        (unsigned long long) stats.h2d_bytes);
    return 0;
}
