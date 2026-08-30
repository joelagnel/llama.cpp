#include "../src/llama-ext.h"

#include "arg.h"
#include "common.h"
#include "llama-cpp.h"

#include <cstdio>
#include <cstring>

static bool decode_one(llama_context * ctx, llama_token token, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, token, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool expect_no_moe_routing(llama_context * ctx) {
    size_t count = 1;
    const llama_moe_routing_entry * entries = llama_get_moe_routing(ctx, &count);
    if (entries != nullptr || count != 0) {
        fprintf(stderr, "%s: expected no MoE routing entries\n", __func__);
        return false;
    }
    return true;
}

static bool expect_moe_routing(llama_context * ctx, int32_t n_expert) {
    size_t count = 0;
    const llama_moe_routing_entry * entries = llama_get_moe_routing(ctx, &count);
    if (entries == nullptr || count == 0) {
        fprintf(stderr, "%s: expected MoE routing entries\n", __func__);
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        if (entries[i].token_index != 0 || entries[i].expert_index < 0 || entries[i].expert_index >= n_expert) {
            fprintf(stderr, "%s: invalid MoE routing entry\n", __func__);
            return false;
        }
    }

    return true;
}

static bool equal_logits(llama_context * lhs, llama_context * rhs, int32_t n_vocab) {
    llama_synchronize(lhs);
    llama_synchronize(rhs);

    const float * lhs_logits = llama_get_logits(lhs);
    const float * rhs_logits = llama_get_logits(rhs);
    if (lhs_logits == nullptr || rhs_logits == nullptr) {
        fprintf(stderr, "%s: missing logits\n", __func__);
        return false;
    }
    if (memcmp(lhs_logits, rhs_logits, n_vocab*sizeof(float)) != 0) {
        fprintf(stderr, "%s: diagnostic path changed logits\n", __func__);
        return false;
    }
    return true;
}

static llama_context_ptr make_context(llama_model * model, const common_params & params) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx = 64;
    cparams.n_batch = 8;
    cparams.n_ubatch = 1;
    cparams.n_seq_max = 1;
    return llama_context_ptr { llama_init_from_model(model, cparams) };
}

static bool test_dense_model(llama_model * model, const common_params & params) {
    if (llama_model_n_expert(model) != 0) {
        fprintf(stderr, "%s: expected a dense model\n", __func__);
        return false;
    }

    auto ctx_plain = make_context(model, params);
    auto ctx_off   = make_context(model, params);
    auto ctx_dense = make_context(model, params);
    if (!ctx_plain || !ctx_off || !ctx_dense) {
        fprintf(stderr, "%s: failed to create contexts\n", __func__);
        return false;
    }

    llama_set_moe_routing(ctx_off.get(), false);
    llama_set_moe_routing(ctx_dense.get(), true);

    for (llama_pos pos = 0; pos < 3; ++pos) {
        const llama_token token = (llama_token) (pos + 1);
        if (!decode_one(ctx_plain.get(), token, pos) ||
                !decode_one(ctx_off.get(), token, pos) ||
                !decode_one(ctx_dense.get(), token, pos)) {
            fprintf(stderr, "%s: dense decode failed\n", __func__);
            return false;
        }
        if (!expect_no_moe_routing(ctx_off.get()) || !expect_no_moe_routing(ctx_dense.get())) {
            return false;
        }
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    return equal_logits(ctx_plain.get(), ctx_off.get(), n_vocab) &&
        equal_logits(ctx_plain.get(), ctx_dense.get(), n_vocab);
}

static bool test_moe_model(llama_model * model, const common_params & params) {
    const int32_t n_expert = llama_model_n_expert(model);
    if (n_expert <= 0) {
        fprintf(stderr, "%s: expected an MoE model\n", __func__);
        return false;
    }

    auto ctx = make_context(model, params);
    if (!ctx) {
        fprintf(stderr, "%s: failed to create context\n", __func__);
        return false;
    }

    llama_set_moe_routing(ctx.get(), false);
    if (!decode_one(ctx.get(), 1, 0) || !expect_no_moe_routing(ctx.get())) {
        return false;
    }

    llama_set_moe_routing(ctx.get(), true);
    if (!decode_one(ctx.get(), 2, 1) || !expect_moe_routing(ctx.get(), n_expert)) {
        return false;
    }

    llama_set_moe_routing(ctx.get(), false);
    if (!decode_one(ctx.get(), 3, 2) || !expect_no_moe_routing(ctx.get())) {
        return false;
    }

    llama_set_moe_routing(ctx.get(), true);
    if (!decode_one(ctx.get(), 4, 3) || !expect_moe_routing(ctx.get(), n_expert)) {
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_gpu_layers = 0;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();
    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init ? llama_init->model() : nullptr;
    if (model == nullptr) {
        fprintf(stderr, "%s: failed to load model\n", __func__);
        return 1;
    }

    if (llama_model_n_expert(model) == 0) {
        return test_dense_model(model, params) ? 0 : 1;
    }

    return test_moe_model(model, params) ? 0 : 1;
}
