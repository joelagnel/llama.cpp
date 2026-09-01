#include "../src/llama-ext.h"
#include "../src/llama-context.h"

#include "arg.h"
#include "common.h"
#include "llama-cpp.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <vector>

static llama_token normalize_test_token(llama_context * ctx, llama_token token) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    GGML_ASSERT(n_vocab > 0);
    return token % n_vocab;
}

static bool decode_one(llama_context * ctx, llama_token token, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, normalize_test_token(ctx, token), pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_many(llama_context * ctx, llama_token token, llama_pos pos, int32_t n_tokens) {
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        common_batch_add(batch, normalize_test_token(ctx, token + i), pos + i, { 0 }, true);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_unequal_sequences(llama_context * ctx) {
    llama_batch batch = llama_batch_init(5, 0, 1);
    common_batch_add(batch, 1, 0, { 0 }, true);
    common_batch_add(batch, 2, 0, { 1 }, true);
    common_batch_add(batch, 3, 1, { 0 }, true);
    common_batch_add(batch, 4, 2, { 0 }, true);
    common_batch_add(batch, 5, 1, { 1 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool encode_one(llama_context * ctx, llama_token token, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, token, pos, { 0 }, true);
    const bool ok = llama_encode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

struct abort_dispatch {
    bool requested = false;

    static bool callback(void * user_data) {
        return static_cast<abort_dispatch *>(user_data)->requested;
    }
};

struct dispatch_script {
    std::vector<llama_context_dispatch_decision> decisions;
    size_t next = 0;

    static llama_context_dispatch_decision pre(
            void * user_data,
            llama_context_dispatch_operation operation) {
        dispatch_script * script = static_cast<dispatch_script *>(user_data);
        if (operation != LLAMA_CONTEXT_DISPATCH_OPERATION_DECODE || script->next >= script->decisions.size()) {
            return {};
        }
        return script->decisions[script->next++];
    }
};

static bool has_explicit_gpu_layers(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        if (strcmp(arg, "-ngl") == 0 || strcmp(arg, "--gpu-layers") == 0 || strcmp(arg, "--n-gpu-layers") == 0 ||
                strncmp(arg, "-ngl=", 5) == 0 || strncmp(arg, "--gpu-layers=", 13) == 0 ||
                strncmp(arg, "--n-gpu-layers=", 15) == 0) {
            return true;
        }
    }
    return false;
}

static bool expect_empty_moe_routing_observer(llama_context * ctx, const char * phase) {
    const auto observer = llama_moe_routing_test_observer_get(ctx);
    if (observer.enabled || observer.reserve_pending || observer.graph_reserve_invalidations != 0 ||
            observer.graph_reserves != 0 || observer.graph_output_extractions != 0 ||
            observer.capture_slot_allocations != 0 || observer.readback_allocations != 0 || observer.readback_copies != 0 ||
            observer.synchronizations != 0 || observer.batch_peer_reads != 0) {
        fprintf(stderr, "%s: unexpected MoE work in %s\n", __func__, phase);
        return false;
    }
#ifdef LLAMA_MOE_ROUTING_TEST_CUDA
    if (observer.device_to_host_copies != 0) {
        fprintf(stderr, "%s: unexpected CUDA readback in %s\n", __func__, phase);
        return false;
    }
#endif
    return true;
}

static bool expect_enabled_moe_routing_observer(llama_context * ctx) {
    const auto observer = llama_moe_routing_test_observer_get(ctx);
    if (!observer.enabled || observer.reserve_pending || observer.graph_reserves == 0 ||
            observer.graph_output_extractions == 0 || observer.capture_slot_allocations == 0 || observer.readback_allocations == 0 ||
            observer.readback_copies == 0 || observer.synchronizations == 0 ||
            observer.batch_peer_reads == 0) {
        fprintf(stderr, "%s: missing enabled MoE work\n", __func__);
        return false;
    }
    return true;
}

static bool expect_cuda_moe_routing_readback(llama_context * ctx) {
    const auto observer = llama_moe_routing_test_observer_get(ctx);
    if (observer.device_to_host_copies == 0) {
        fprintf(stderr, "%s: expected a CUDA device-to-host readback\n", __func__);
        return false;
    }
    return true;
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

static bool expect_moe_routing(
        llama_context * ctx,
        llama_model * model,
        int32_t n_expert,
        int32_t n_expert_used,
        int32_t n_tokens) {
    const llama_moe_routing_readback * readback = llama_get_moe_routing_readback(ctx);
    if (readback == nullptr || readback->version != LLAMA_MOE_ROUTING_READBACK_VERSION ||
            readback->struct_size < sizeof(*readback) || readback->row_count == 0 || readback->rows == nullptr) {
        fprintf(stderr, "%s: missing MoE routing readback\n", __func__);
        return false;
    }

    std::map<int32_t, std::set<int32_t>> rows_by_layer;
    std::set<int32_t> moe_layers;
    for (int32_t index = 0; index < llama_model_n_moe_layer(model); ++index) {
        const int32_t layer = llama_model_moe_layer_index(model, index);
        if (layer < 0 || layer >= llama_model_n_layer(model) || !moe_layers.insert(layer).second) {
            fprintf(stderr, "%s: invalid MoE layer topology\n", __func__);
            return false;
        }
    }
    if (moe_layers.empty() || llama_model_moe_layer_index(model, (int32_t) moe_layers.size()) != -1) {
        fprintf(stderr, "%s: incomplete MoE layer topology\n", __func__);
        return false;
    }
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
        if (moe_layers.count(row.layer_index) == 0) {
            fprintf(stderr, "%s: routing row outside MoE topology\n", __func__);
            return false;
        }
        physical_ubatches.insert(row.physical_ubatch_index);
    }

    for (const auto & item : rows_by_layer) {
        if ((int32_t) item.second.size() != n_tokens) {
            fprintf(stderr, "%s: missing routed rows for layer %d\n", __func__, item.first);
            return false;
        }
    }
    if (rows_by_layer.size() != moe_layers.size()) {
        fprintf(stderr, "%s: incomplete routed MoE topology\n", __func__);
        return false;
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

static bool test_moe_routing_primary_position_mapping() {
    const llama_moe_routing_test_row_position_mapping result =
        llama_context::test_map_moe_routing_primary_positions();
    if (result.all_row_count != 40 || result.output_row_count != 20 ||
            !result.all_rows_use_primary_positions || !result.output_rows_use_primary_positions ||
            !result.all_rows_are_unique || !result.output_rows_are_unique) {
        fprintf(stderr, "%s: four-plane row-position mapping was not exact\n", __func__);
        return false;
    }
    return true;
}

static bool test_dispatch_observer(llama_model * model, const common_params & params) {
    if (llama_model_n_expert(model) == 0) {
        return true;
    }

    auto ctx = make_context(model, params);
    if (!ctx) {
        fprintf(stderr, "%s: failed to create context\n", __func__);
        return false;
    }

    dispatch_script script = {{
        { 1, 1, 1, 1, true,  true,  true },
        { 1, 2, 2, 0, false, true,  true },
        { 1, 3, 3, 1, true,  true,  true },
    }};
    ctx->set_dispatch_observer({ &script, dispatch_script::pre });
    llama_moe_routing_test_observer_reset(ctx.get());
    if (!decode_many(ctx.get(), 1, 0, 6)) {
        fprintf(stderr, "%s: scripted multi-ubatch decode failed\n", __func__);
        return false;
    }

    llama_context_dispatch_drain drained = ctx->dispatch_drain();
    if (script.next != 3 || drained.notices.size() != 3 || drained.moe_routing_spans.size() != 2 ||
            !drained.losses.empty() || drained.dropped_notices != 0 || drained.dropped_moe_routing_spans != 0) {
        fprintf(stderr, "%s: missing scripted dispatch records\n", __func__);
        return false;
    }
    for (size_t i = 0; i < drained.notices.size(); ++i) {
        const auto & notice = drained.notices[i];
        if (notice.operation != LLAMA_CONTEXT_DISPATCH_OPERATION_DECODE ||
                notice.physical_step != i + 1 || notice.physical_microbatch != i ||
                notice.decision.microbatch_generation != i + 1 || notice.dispatch_monotonic_us <= 0) {
            fprintf(stderr, "%s: invalid physical boundary coordinate\n", __func__);
            return false;
        }
    }
    if (drained.moe_routing_spans[0].decision.microbatch_generation != 1 ||
            drained.moe_routing_spans[0].first_physical_step != 1 ||
            drained.moe_routing_spans[0].last_physical_step != 1 ||
            drained.moe_routing_spans[1].decision.microbatch_generation != 3 ||
            drained.moe_routing_spans[1].first_physical_step != 3 ||
            drained.moe_routing_spans[1].last_physical_step != 3 ||
            drained.moe_routing_spans[0].physical_dispatches.size() != 1 ||
            drained.moe_routing_spans[1].physical_dispatches.size() != 1 ||
            drained.moe_routing_spans[0].physical_dispatches.front().dispatch_monotonic_us <= 0 ||
            drained.moe_routing_spans[1].physical_dispatches.front().dispatch_monotonic_us <= 0) {
        fprintf(stderr, "%s: transition did not preserve exact routing spans\n", __func__);
        return false;
    }
    if (drained.moe_routing_spans[0].rows.empty() || drained.moe_routing_spans[1].rows.empty()) {
        fprintf(stderr, "%s: enabled spans did not retain routing rows\n", __func__);
        return false;
    }

    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx = 64;
    cparams.n_batch = 8;
    cparams.n_ubatch = 2;
    cparams.n_seq_max = 2;
    llama_context_ptr ctx_multi { llama_init_from_model(model, cparams) };
    if (!ctx_multi) {
        fprintf(stderr, "%s: failed to create multi-sequence context\n", __func__);
        return false;
    }
    dispatch_script steady = {{
        { 1, 3, 3, 1, true, true, true },
        { 1, 3, 3, 1, true, true, true },
        { 1, 3, 3, 1, true, true, true },
    }};
    ctx_multi->set_dispatch_observer({ &steady, dispatch_script::pre });
    if (!decode_unequal_sequences(ctx_multi.get())) {
        fprintf(stderr, "%s: unequal-sequence decode failed\n", __func__);
        return false;
    }
    drained = ctx_multi->dispatch_drain();
    if (drained.notices.empty() || drained.notices.front().physical_step != 1 ||
            drained.moe_routing_spans.size() != 1 ||
            drained.moe_routing_spans.front().first_physical_step != 1 ||
            drained.moe_routing_spans.front().last_physical_step <
                drained.moe_routing_spans.front().first_physical_step) {
        fprintf(stderr, "%s: unequal-sequence routing span was not retained\n", __func__);
        return false;
    }

    cparams.n_ctx = 128;
    cparams.n_batch = 72;
    cparams.n_ubatch = 2;
    cparams.n_seq_max = 1;
    llama_context_ptr ctx_loss { llama_init_from_model(model, cparams) };
    if (!ctx_loss) {
        fprintf(stderr, "%s: failed to create queue-loss context\n", __func__);
        return false;
    }
    dispatch_script changing;
    for (uint64_t generation = 1; generation <= 36; ++generation) {
        changing.decisions.push_back({ 1, generation, generation, 1, true, true, true });
    }
    ctx_loss->set_dispatch_observer({ &changing, dispatch_script::pre });
    if (!decode_many(ctx_loss.get(), 1, 0, 72)) {
        fprintf(stderr, "%s: queue-loss decode failed\n", __func__);
        return false;
    }
    drained = ctx_loss->dispatch_drain();
    if (changing.next != 36 || drained.notices.size() != 1 ||
            drained.moe_routing_spans.size() != 32 || drained.dropped_moe_routing_spans != 4 ||
            drained.losses.size() != 4 || drained.dropped_loss_descriptors != 0) {
        fprintf(stderr, "%s: bounded queue did not report the expected loss (next=%zu notices=%zu spans=%zu dropped=%llu losses=%zu descriptors=%llu)\n",
                __func__, changing.next, drained.notices.size(), drained.moe_routing_spans.size(),
                (unsigned long long) drained.dropped_moe_routing_spans, drained.losses.size(),
                (unsigned long long) drained.dropped_loss_descriptors);
        return false;
    }
    for (size_t i = 0; i < drained.losses.size(); ++i) {
        const auto & loss = drained.losses[i];
        if (!loss.moe_routing_span || loss.operation != LLAMA_CONTEXT_DISPATCH_OPERATION_DECODE ||
                loss.first_physical_step != i + 1 || loss.next_physical_step != i + 2 ||
                loss.decision.microbatch_generation != i + 1 || loss.first_dispatch_monotonic_us <= 0 ||
                loss.last_dispatch_monotonic_us < loss.first_dispatch_monotonic_us) {
            fprintf(stderr, "%s: queue loss did not retain exact coordinates\n", __func__);
            return false;
        }
    }

    cparams.n_ctx = 1024;
    cparams.n_batch = 620;
    cparams.n_ubatch = 2;
    llama_context_ptr ctx_saturation { llama_init_from_model(model, cparams) };
    if (!ctx_saturation) {
        fprintf(stderr, "%s: failed to create saturation context\n", __func__);
        return false;
    }
    dispatch_script saturating;
    for (uint64_t generation = 1; generation <= 310; ++generation) {
        // Alternating a non-routing microbatch control produces both boundary
        // and routing-span pressure while keeping native routing enabled.
        saturating.decisions.push_back({ 1, generation, generation,
            (uint8_t) (generation % 2 ? 1 : 33), true, true, true });
    }
    ctx_saturation->set_dispatch_observer({ &saturating, dispatch_script::pre });
    if (!decode_many(ctx_saturation.get(), 1, 0, 620)) {
        fprintf(stderr, "%s: saturation decode failed\n", __func__);
        return false;
    }
    drained = ctx_saturation->dispatch_drain();
    if (saturating.next != 310 || drained.notices.size() != 256 || drained.dropped_notices != 16 ||
            drained.moe_routing_spans.size() != 32 || drained.dropped_moe_routing_spans != 240 ||
            drained.losses.size() != 256 || drained.dropped_loss_descriptors != 0) {
        fprintf(stderr, "%s: saturation did not preserve bounded exact loss evidence (next=%zu notices=%zu dropped-notices=%llu spans=%zu dropped-spans=%llu losses=%zu descriptors=%llu)\n",
                __func__, saturating.next, drained.notices.size(),
                (unsigned long long) drained.dropped_notices, drained.moe_routing_spans.size(),
                (unsigned long long) drained.dropped_moe_routing_spans, drained.losses.size(),
                (unsigned long long) drained.dropped_loss_descriptors);
        return false;
    }
    std::set<uint64_t> lost_span_steps;
    std::set<uint64_t> lost_notice_steps;
    for (size_t i = 0; i < 255; ++i) {
        const auto & loss = drained.losses[i];
        if (loss.saturation || loss.first_physical_step == 0 ||
                loss.next_physical_step != loss.first_physical_step + 1 ||
                loss.first_physical_microbatch + 1 != loss.first_physical_step ||
                loss.last_physical_microbatch != loss.first_physical_microbatch ||
                loss.physical_dispatch_count != 1 || loss.first_dispatch_monotonic_us <= 0 ||
                loss.last_dispatch_monotonic_us != loss.first_dispatch_monotonic_us) {
            fprintf(stderr, "%s: detailed loss was not exact before saturation (index=%zu kind=%d steps=%llu-%llu micro=%u-%u count=%llu time=%lld-%lld)\n",
                    __func__, i, loss.moe_routing_span, (unsigned long long) loss.first_physical_step,
                    (unsigned long long) loss.next_physical_step, loss.first_physical_microbatch,
                    loss.last_physical_microbatch, (unsigned long long) loss.physical_dispatch_count,
                    (long long) loss.first_dispatch_monotonic_us, (long long) loss.last_dispatch_monotonic_us);
            return false;
        }
        (loss.moe_routing_span ? lost_span_steps : lost_notice_steps).insert(loss.first_physical_step);
    }
    if (lost_span_steps.size() != 239 || lost_notice_steps.size() != 16 ||
            *lost_span_steps.begin() != 1 || *lost_span_steps.rbegin() != 239 ||
            *lost_notice_steps.begin() != 1 || *lost_notice_steps.rbegin() != 16) {
        fprintf(stderr, "%s: detailed prefix did not retain both loss populations\n", __func__);
        return false;
    }
    for (size_t i = 0; i < drained.moe_routing_spans.size(); ++i) {
        const auto & span = drained.moe_routing_spans[i];
        if (span.first_physical_step != 240 + i || span.last_physical_step != 240 + i ||
                span.physical_dispatches.size() != 1 ||
                span.physical_dispatches.front().physical_step != 240 + i) {
            fprintf(stderr, "%s: retained spans overlap or leave a fake saturation coordinate\n", __func__);
            return false;
        }
    }
    const auto & saturation = drained.losses.back();
    if (!saturation.saturation || !saturation.moe_routing_span ||
            saturation.operation != LLAMA_CONTEXT_DISPATCH_OPERATION_DECODE ||
            saturation.first_physical_step != 272 || saturation.next_physical_step != 311 ||
            saturation.first_physical_microbatch != 271 || saturation.last_physical_microbatch != 309 ||
            saturation.physical_dispatch_count != 39 || !saturation.generation_mixed ||
            saturation.decision.microbatch_generation != 272 ||
            saturation.last_microbatch_generation != 310 || saturation.first_dispatch_monotonic_us <= 0 ||
            saturation.last_dispatch_monotonic_us < saturation.first_dispatch_monotonic_us) {
        fprintf(stderr, "%s: saturation interval was not exact\n", __func__);
        return false;
    }

    llama_context_ptr ctx_interleaved { llama_init_from_model(model, cparams) };
    if (!ctx_interleaved) {
        fprintf(stderr, "%s: failed to create interleaved saturation context\n", __func__);
        return false;
    }
    dispatch_script interleaved;
    for (uint64_t generation = 1; generation <= 272; ++generation) {
        interleaved.decisions.push_back({ 1, generation, generation,
            (uint8_t) (generation % 2 ? 1 : 33), true, true, true });
    }
    ctx_interleaved->set_dispatch_observer({ &interleaved, dispatch_script::pre });
    if (!decode_many(ctx_interleaved.get(), 1, 0, 544)) {
        fprintf(stderr, "%s: interleaved saturation decode failed\n", __func__);
        return false;
    }

    abort_dispatch abort = { true };
    llama_set_abort_callback(ctx_interleaved.get(), abort_dispatch::callback, &abort);
    if (decode_one(ctx_interleaved.get(), 545, 544)) {
        fprintf(stderr, "%s: requested graph failure unexpectedly succeeded\n", __func__);
        return false;
    }
    abort.requested = false;
    llama_set_abort_callback(ctx_interleaved.get(), nullptr, nullptr);

    for (llama_token token = 1; token <= 8; ++token) {
        if (!decode_one(ctx_interleaved.get(), 544 + token, 544 + token)) {
            fprintf(stderr, "%s: post-failure decode recovery failed\n", __func__);
            return false;
        }
    }
    drained = ctx_interleaved->dispatch_drain();
    if (interleaved.next != 272 || drained.losses.size() != 256 ||
            drained.moe_routing_spans.size() != 32 || drained.dropped_loss_descriptors != 0) {
        fprintf(stderr, "%s: interleaved saturation drain did not preserve its detailed prefix\n", __func__);
        return false;
    }
    const auto & interleaved_saturation = drained.losses.back();
    if (!interleaved_saturation.saturation || !interleaved_saturation.moe_routing_span ||
            interleaved_saturation.first_physical_step != 272 ||
            interleaved_saturation.next_physical_step != 281 ||
            interleaved_saturation.first_physical_microbatch != 271 ||
            interleaved_saturation.last_physical_microbatch != 0 ||
            interleaved_saturation.physical_dispatch_count != 9 ||
            interleaved_saturation.operation != LLAMA_CONTEXT_DISPATCH_OPERATION_DECODE ||
            interleaved_saturation.last_operation != LLAMA_CONTEXT_DISPATCH_OPERATION_DECODE ||
            interleaved_saturation.operation_mixed ||
            interleaved_saturation.encode_physical_dispatch_count != 0 ||
            interleaved_saturation.decode_physical_dispatch_count != 9 ||
            interleaved_saturation.encode_physical_dispatch_count +
                interleaved_saturation.decode_physical_dispatch_count !=
                    interleaved_saturation.physical_dispatch_count ||
            interleaved_saturation.last_dispatch_monotonic_us <
                interleaved_saturation.first_dispatch_monotonic_us) {
        fprintf(stderr, "%s: post-failure saturation was not exact\n", __func__);
        return false;
    }
    for (const auto & loss : drained.losses) {
        if (&loss != &interleaved_saturation &&
                loss.next_physical_step > interleaved_saturation.first_physical_step) {
            fprintf(stderr, "%s: detailed loss overlaps the terminal saturation interval\n", __func__);
            return false;
        }
    }
    for (size_t i = 0; i < drained.moe_routing_spans.size(); ++i) {
        const auto & span = drained.moe_routing_spans[i];
        if (span.first_physical_step != 240 + i || span.last_physical_step != 240 + i ||
                span.last_physical_step >= interleaved_saturation.first_physical_step) {
            fprintf(stderr, "%s: retained evidence overlaps the terminal saturation interval\n", __func__);
            return false;
        }
    }

    return true;
}

static bool test_dense_model(llama_model * model, const common_params & params) {
    if (llama_model_n_expert(model) != 0 || llama_model_n_expert_shared(model) != 0 ||
            llama_model_n_moe_layer(model) != 0 || llama_model_moe_layer_index(model, 0) != -1) {
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
    llama_moe_routing_test_observer_reset(ctx_off.get());
    llama_moe_routing_test_observer_reset(ctx_dense.get());
    llama_set_moe_routing(ctx_dense.get(), true);

    if (!expect_empty_moe_routing_observer(ctx_off.get(), "dense disabled setup") ||
            !expect_empty_moe_routing_observer(ctx_dense.get(), "dense enabled setup")) {
        return false;
    }

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
        if (!expect_empty_moe_routing_observer(ctx_off.get(), "dense disabled decode") ||
                !expect_empty_moe_routing_observer(ctx_dense.get(), "dense enabled decode")) {
            return false;
        }
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    return equal_logits(ctx_plain.get(), ctx_off.get(), n_vocab) &&
        equal_logits(ctx_plain.get(), ctx_dense.get(), n_vocab);
}

static bool test_moe_model(llama_model * model, const common_params & params, bool require_cuda_readback) {
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
    llama_moe_routing_test_observer_reset(ctx.get());
    if (!decode_one(ctx_plain.get(), 1, 0) || !decode_one(ctx.get(), 1, 0) ||
            !expect_no_moe_routing(ctx.get()) ||
            !expect_empty_moe_routing_observer(ctx.get(), "MoE disabled decode")) {
        return false;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (!equal_logits(ctx_plain.get(), ctx.get(), n_vocab)) {
        return false;
    }

    llama_set_moe_routing(ctx.get(), true);
    {
        const auto observer = llama_moe_routing_test_observer_get(ctx.get());
        if (!observer.enabled || !observer.reserve_pending || observer.graph_reserve_invalidations != 1) {
            fprintf(stderr, "%s: enabling MoE routing did not invalidate the graph once\n", __func__);
            return false;
        }
    }
    llama_moe_routing_test_observer_reset(ctx.get());
    if (!decode_many(ctx.get(), 2, 1, 3) || !expect_moe_routing(ctx.get(), model, n_expert, n_expert_used, 3)) {
        return false;
    }
    if (!expect_enabled_moe_routing_observer(ctx.get()) ||
            (require_cuda_readback && !expect_cuda_moe_routing_readback(ctx.get()))) {
        return false;
    }

    llama_set_moe_routing(ctx.get(), false);
    if (!decode_one(ctx.get(), 5, 4) || !expect_no_moe_routing(ctx.get())) {
        return false;
    }

    llama_moe_routing_test_observer_reset(ctx.get());
    if (!decode_one(ctx.get(), 6, 5) || !expect_no_moe_routing(ctx.get()) ||
            !expect_empty_moe_routing_observer(ctx.get(), "MoE disabled steady state")) {
        return false;
    }

    llama_set_moe_routing(ctx.get(), true);
    if (!decode_one(ctx.get(), 7, 6) || !expect_moe_routing(ctx.get(), model, n_expert, n_expert_used, 1)) {
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
    const bool explicit_gpu_layers = has_explicit_gpu_layers(argc, argv);
    if (!explicit_gpu_layers) {
        params.n_gpu_layers = 0;
    }
    const bool require_cuda_readback = explicit_gpu_layers && params.n_gpu_layers != 0;

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

    return test_moe_routing_primary_position_mapping() &&
        test_moe_model(model, params, require_cuda_readback) &&
        test_dispatch_observer(model, params) ? 0 : 1;
}
