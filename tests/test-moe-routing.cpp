#include "../src/llama-ext.h"

#include "arg.h"
#include "common.h"
#include "llama-cpp.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <map>
#include <set>

static bool decode_one(llama_context * ctx, llama_token token, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, token, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_many(llama_context * ctx, llama_token token, llama_pos pos, int32_t n_tokens) {
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        common_batch_add(batch, token + i, pos + i, { 0 }, true);
    }
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
    if (llama_get_moe_routing_readback(ctx) != nullptr) {
        fprintf(stderr, "%s: expected no MoE routing readback\n", __func__);
        return false;
    }
    return true;
}

static bool expect_moe_routing(llama_context * ctx, int32_t n_expert, int32_t n_expert_used, int32_t n_tokens) {
    const llama_moe_routing_readback * readback = llama_get_moe_routing_readback(ctx);
    if (readback == nullptr || readback->version != LLAMA_MOE_ROUTING_READBACK_VERSION ||
            readback->struct_size < sizeof(*readback) || readback->row_count == 0 || readback->rows == nullptr) {
        fprintf(stderr, "%s: missing MoE routing readback\n", __func__);
        return false;
    }

    std::map<int32_t, std::set<int32_t>> rows_by_layer;
    std::set<uint32_t> physical_ubatches;
    std::set<std::pair<int32_t, uint32_t>> shared_metadata;
    for (size_t i = 0; i < readback->shared_expert_count; ++i) {
        const auto & metadata = readback->shared_experts[i];
        shared_metadata.insert({ metadata.layer_index, metadata.graph_type });
    }

    for (size_t i = 0; i < readback->row_count; ++i) {
        const auto & row = readback->rows[i];
        if (row.layer_index < 0 || row.row_index < 0 || row.ubatch_token_index < 0 || row.token_index < 0 ||
                row.token_index >= n_tokens || row.row_identity_status != LLAMA_MOE_ROUTING_VALUE_STATUS_VALID ||
                row.selected_experts_status != LLAMA_MOE_ROUTING_VALUE_STATUS_VALID ||
                row.selected_expert_count != (size_t) n_expert_used || row.selected_experts == nullptr ||
                row.selected_score_status != LLAMA_MOE_ROUTING_VALUE_STATUS_VALID ||
                row.rejected_score_status != LLAMA_MOE_ROUTING_VALUE_STATUS_VALID ||
                !std::isfinite(row.selected_score) || !std::isfinite(row.rejected_score) ||
                row.selected_score < row.rejected_score ||
                shared_metadata.count({ row.layer_index, row.graph_type }) == 0) {
            fprintf(stderr, "%s: invalid MoE routing row layer=%d row=%d ubatch=%d token=%d identity=%d selected=%d count=%zu selected-score=%f/%d rejected-score=%f/%d\n",
                    __func__, row.layer_index, row.row_index, row.ubatch_token_index, row.token_index,
                    row.row_identity_status, row.selected_experts_status, row.selected_expert_count,
                    row.selected_score, row.selected_score_status, row.rejected_score, row.rejected_score_status);
            return false;
        }

        for (size_t expert_index = 0; expert_index < row.selected_expert_count; ++expert_index) {
            const auto & expert = row.selected_experts[expert_index];
            if (expert.expert_index < 0 || expert.expert_index >= n_expert ||
                    expert.expert_index_status != LLAMA_MOE_ROUTING_VALUE_STATUS_VALID ||
                    expert.effective_weight_status != LLAMA_MOE_ROUTING_VALUE_STATUS_VALID ||
                    !std::isfinite(expert.effective_weight)) {
                fprintf(stderr, "%s: invalid selected expert\n", __func__);
                return false;
            }
        }

        rows_by_layer[row.layer_index].insert(row.token_index);
        physical_ubatches.insert(row.physical_ubatch_index);
    }

    for (const auto & item : rows_by_layer) {
        if ((int32_t) item.second.size() != n_tokens) {
            fprintf(stderr, "%s: missing routed rows for layer %d\n", __func__, item.first);
            return false;
        }
    }
    if (n_tokens > 1 && physical_ubatches.size() < 2) {
        fprintf(stderr, "%s: missing physical microbatch routing rows\n", __func__);
        return false;
    }

    size_t count = 0;
    const llama_moe_routing_entry * entries = llama_get_moe_routing(ctx, &count);
    if (entries == nullptr || count == 0) {
        fprintf(stderr, "%s: expected MoE routing entries\n", __func__);
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        if (entries[i].token_index < 0 || entries[i].token_index >= n_tokens ||
                entries[i].expert_index < 0 || entries[i].expert_index >= n_expert ||
                !std::isfinite(entries[i].effective_weight)) {
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
    cparams.n_ubatch = 2;
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
    const int32_t n_expert_used = llama_model_n_expert_used(model);
    if (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert) {
        fprintf(stderr, "%s: expected an MoE model\n", __func__);
        return false;
    }

    auto ctx_plain = make_context(model, params);
    auto ctx = make_context(model, params);
    if (!ctx_plain || !ctx) {
        fprintf(stderr, "%s: failed to create context\n", __func__);
        return false;
    }

    llama_set_moe_routing(ctx.get(), false);
    if (!decode_one(ctx_plain.get(), 1, 0) || !decode_one(ctx.get(), 1, 0) ||
            !expect_no_moe_routing(ctx.get())) {
        return false;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (!equal_logits(ctx_plain.get(), ctx.get(), n_vocab)) {
        return false;
    }

    llama_set_moe_routing(ctx.get(), true);
    if (!decode_many(ctx.get(), 2, 1, 3) || !expect_moe_routing(ctx.get(), n_expert, n_expert_used, 3)) {
        return false;
    }

    llama_set_moe_routing(ctx.get(), false);
    if (!decode_one(ctx.get(), 5, 4) || !expect_no_moe_routing(ctx.get())) {
        return false;
    }

    llama_set_moe_routing(ctx.get(), true);
    if (!decode_one(ctx.get(), 6, 5) || !expect_moe_routing(ctx.get(), n_expert, n_expert_used, 1)) {
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    common_params params;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    params.n_gpu_layers = 0;

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
