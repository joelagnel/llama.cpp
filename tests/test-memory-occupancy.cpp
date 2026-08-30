// SPDX-License-Identifier: GPL-2.0

#include "../src/llama-ext.h"

#include "arg.h"
#include "common.h"
#include "llama-cpp.h"

#include <cstdio>

static bool check_primary_occupancy(llama_context * ctx, uint64_t & used_entries) {
    const llama_memory_snapshot shallow = llama_get_memory_snapshot(ctx, false);
    const llama_memory_snapshot deep = llama_get_memory_snapshot(ctx, true);
    const llama_memory_primary_occupancy & occupancy = shallow.primary_occupancy;
    const llama_memory_diagnostics & diagnostics = deep.diagnostics;

    if (!occupancy.available || !deep.diagnostics_collected || diagnostics.state != "available") {
        fprintf(stderr, "%s: occupancy is unavailable\n", __func__);
        return false;
    }
    const llama_memory_component_diagnostics * primary = nullptr;
    for (const auto & component : diagnostics.components) {
        if (!component.logical_primary) {
            continue;
        }
        if (primary != nullptr) {
            fprintf(stderr, "%s: more than one primary memory component\n", __func__);
            return false;
        }
        primary = &component;
    }

    if (primary == nullptr) {
        fprintf(stderr, "%s: primary memory component is missing\n", __func__);
        return false;
    }
    if (deep.primary_occupancy.capacity_entries != primary->capacity_entries ||
            deep.primary_occupancy.used_entries != primary->used_entries) {
        fprintf(stderr, "%s: deep snapshot is internally inconsistent\n", __func__);
        return false;
    }
    if (occupancy.capacity_entries != primary->capacity_entries || occupancy.used_entries != primary->used_entries) {
        fprintf(stderr,
                "%s: lightweight occupancy (%llu/%llu) differs from diagnostics (%llu/%llu)\n",
                __func__,
                (unsigned long long) occupancy.used_entries,
                (unsigned long long) occupancy.capacity_entries,
                (unsigned long long) primary->used_entries,
                (unsigned long long) primary->capacity_entries);
        return false;
    }
    if (occupancy.capacity_entries == 0 || occupancy.used_entries > occupancy.capacity_entries) {
        fprintf(stderr, "%s: invalid occupancy bounds\n", __func__);
        return false;
    }

    used_entries = occupancy.used_entries;
    return true;
}

static bool check_snapshot_is_immutable(llama_context * ctx) {
    const llama_memory_snapshot before = llama_get_memory_snapshot(ctx, true);
    if (!before.diagnostics_collected || before.diagnostics.components.empty()) {
        fprintf(stderr, "%s: deep snapshot is unavailable\n", __func__);
        return false;
    }
    const uint64_t used_before = before.primary_occupancy.used_entries;
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1)) {
        fprintf(stderr, "%s: failed to mutate memory after snapshot\n", __func__);
        return false;
    }
    const llama_memory_snapshot after = llama_get_memory_snapshot(ctx, false);
    if (before.primary_occupancy.used_entries != used_before || after.primary_occupancy.used_entries >= used_before) {
        fprintf(stderr, "%s: snapshot changed after memory mutation\n", __func__);
        return false;
    }
    return true;
}

static bool decode_sequence(llama_context * ctx, llama_seq_id seq_id) {
    constexpr int32_t n_tokens = 8;
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        common_batch_add(batch, 1, i, { seq_id }, i == n_tokens - 1);
    }

    const int result = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (result != 0) {
        fprintf(stderr, "%s: decode failed for sequence %d: %d\n", __func__, seq_id, result);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx = 256;
    params.n_batch = 32;
    params.n_ubatch = 32;
    params.n_parallel = 2;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PARALLEL)) {
        return 1;
    }
    params.n_gpu_layers = 0;

    const llama_memory_primary_occupancy null_occupancy = llama_get_memory_primary_occupancy(nullptr);
    if (null_occupancy.available || null_occupancy.capacity_entries != 0 || null_occupancy.used_entries != 0) {
        fprintf(stderr, "%s: null context returned available occupancy\n", __func__);
        return 1;
    }

    ggml_backend_load_all();
    common_init_result_ptr llama_init = common_init_from_params(params, true);
    llama_model * model = llama_init ? llama_init->model() : nullptr;
    if (model == nullptr) {
        fprintf(stderr, "%s: failed to load model\n", __func__);
        return 1;
    }

    llama_context_params context_params = common_context_params_to_llama(params);
    context_params.n_ctx = 256;
    context_params.n_batch = 32;
    context_params.n_ubatch = 32;
    context_params.n_seq_max = 2;
    context_params.kv_unified = params.kv_unified;

    llama_context_ptr ctx { llama_init_from_model(model, context_params) };
    if (!ctx) {
        fprintf(stderr, "%s: failed to create context\n", __func__);
        return 1;
    }

    uint64_t empty_used = 0;
    if (!check_primary_occupancy(ctx.get(), empty_used) || empty_used != 0) {
        fprintf(stderr, "%s: new context is not empty\n", __func__);
        return 1;
    }

    if (!decode_sequence(ctx.get(), 0) || !decode_sequence(ctx.get(), 1)) {
        return 1;
    }

    uint64_t decoded_used = 0;
    if (!check_primary_occupancy(ctx.get(), decoded_used) || decoded_used <= empty_used) {
        fprintf(stderr, "%s: decode did not increase occupancy\n", __func__);
        return 1;
    }

    if (!check_snapshot_is_immutable(ctx.get())) {
        return 1;
    }

    uint64_t remaining_used = 0;
    if (!check_primary_occupancy(ctx.get(), remaining_used) || remaining_used >= decoded_used) {
        fprintf(stderr, "%s: sequence removal did not decrease occupancy\n", __func__);
        return 1;
    }

    return 0;
}
