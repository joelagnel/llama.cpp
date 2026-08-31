#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_expert_used(const struct llama_model * model);
LLAMA_API int32_t llama_model_n_expert_shared(const struct llama_model * model);

// Physical transformer-layer indexes with a routed MoE gate. The indexes are
// sorted and remain valid for the lifetime of the model.
LLAMA_API int32_t llama_model_n_moe_layer(const struct llama_model * model);
LLAMA_API int32_t llama_model_moe_layer_index(const struct llama_model * model, int32_t moe_layer_index);

LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Physical micro-batches actually submitted to the compute graph. These are
// distinct from the logical llama_decode() batch supplied by the caller.
// Histogram buckets are cumulative and use the bounds returned by
// llama_ubatch_histogram_bounds().
constexpr size_t LLAMA_UBATCH_HISTOGRAM_BUCKET_COUNT = 14;

struct llama_ubatch_stats {
    uint64_t attempted = 0;
    uint64_t successful = 0;
    uint64_t tokens = 0;
    uint64_t sequence_tokens = 0;
    uint64_t sequences = 0;
    uint64_t unique_sequences = 0;
    uint64_t max_tokens = 0;
    std::array<uint64_t, LLAMA_UBATCH_HISTOGRAM_BUCKET_COUNT> token_buckets {};
};

LLAMA_API const std::array<uint32_t, LLAMA_UBATCH_HISTOGRAM_BUCKET_COUNT> & llama_ubatch_histogram_bounds();
LLAMA_API llama_ubatch_stats llama_get_ubatch_stats(const struct llama_context * ctx);

// Optional diagnostic output containing the selected routed expert IDs for the
// most recent logical llama_decode() batch. Disabled collection changes no graph
// outputs and performs no host copy. Enabling it changes graph topology and is
// intended for explicitly opted-in, bounded diagnostics rather than normal
// inference.
struct llama_moe_routing_entry {
    int32_t layer_index = -1;
    int32_t token_index = -1; // zero-based position in the logical llama_decode() batch
    int32_t expert_index = -1;
    float effective_weight = std::numeric_limits<float>::quiet_NaN(); // exact coefficient applied to this selected expert
};

LLAMA_API void llama_set_moe_routing(struct llama_context * ctx, bool enabled);
LLAMA_API const llama_moe_routing_entry * llama_get_moe_routing(
        struct llama_context * ctx,
        size_t * count);

// Versioned readback for routed MoE rows. The result and every pointer it
// contains remain valid until the next llama_decode(), llama_set_moe_routing(),
// or llama_context destruction. The caller must copy the result before it
// starts another decode on the same context.
constexpr uint32_t LLAMA_MOE_ROUTING_READBACK_VERSION = 1;

enum llama_moe_routing_value_status : uint8_t {
    LLAMA_MOE_ROUTING_VALUE_STATUS_VALID = 0,
    LLAMA_MOE_ROUTING_VALUE_STATUS_SOURCE_UNAVAILABLE,
    LLAMA_MOE_ROUTING_VALUE_STATUS_INVALID,
    LLAMA_MOE_ROUTING_VALUE_STATUS_NONFINITE,
};

struct llama_moe_routing_expert {
    int32_t expert_index = -1;
    float effective_weight = std::numeric_limits<float>::quiet_NaN();
    llama_moe_routing_value_status expert_index_status = LLAMA_MOE_ROUTING_VALUE_STATUS_SOURCE_UNAVAILABLE;
    llama_moe_routing_value_status effective_weight_status = LLAMA_MOE_ROUTING_VALUE_STATUS_SOURCE_UNAVAILABLE;
};

struct llama_moe_routing_row {
    int32_t layer_index = -1;
    uint32_t graph_type = 0;
    uint32_t physical_ubatch_index = 0;
    int32_t row_index = -1;
    int32_t ubatch_token_index = -1;
    int32_t token_index = -1; // zero-based position in the logical llama_decode() batch
    llama_token token = -1;
    llama_pos position = -1;

    size_t selected_expert_count = 0;
    const llama_moe_routing_expert * selected_experts = nullptr;
    llama_moe_routing_value_status row_identity_status = LLAMA_MOE_ROUTING_VALUE_STATUS_SOURCE_UNAVAILABLE;
    llama_moe_routing_value_status selected_experts_status = LLAMA_MOE_ROUTING_VALUE_STATUS_SOURCE_UNAVAILABLE;

    // Scores are from the selection source after selection bias and group
    // masking. A missing K+1 candidate is source-unavailable, not truncation.
    float selected_score = std::numeric_limits<float>::quiet_NaN();
    float rejected_score = std::numeric_limits<float>::quiet_NaN();
    llama_moe_routing_value_status selected_score_status = LLAMA_MOE_ROUTING_VALUE_STATUS_SOURCE_UNAVAILABLE;
    llama_moe_routing_value_status rejected_score_status = LLAMA_MOE_ROUTING_VALUE_STATUS_SOURCE_UNAVAILABLE;
};

// Shared experts are not members of selected_experts. There is at most one
// record for each captured layer and graph type pair.
struct llama_moe_shared_expert_metadata {
    int32_t layer_index = -1;
    uint32_t graph_type = 0;
    bool present = false;
    uint32_t configured_count = 0;
    uint32_t ffn_size = 0;
};

struct llama_moe_routing_readback {
    uint32_t version = LLAMA_MOE_ROUTING_READBACK_VERSION;
    uint32_t struct_size = 0;
    uint64_t capture_generation = 0;
    size_t row_count = 0;
    const llama_moe_routing_row * rows = nullptr;
    size_t shared_expert_count = 0;
    const llama_moe_shared_expert_metadata * shared_experts = nullptr;
};

LLAMA_API const llama_moe_routing_readback * llama_get_moe_routing_readback(
        struct llama_context * ctx);

#ifdef LLAMA_MOE_ROUTING_TEST_HOOKS
// Test-only counters for proving that inactive routing diagnostics do not
// affect graph execution or readback work.
struct llama_moe_routing_test_observer {
    bool enabled = false;
    bool reserve_pending = false;
    uint64_t graph_reserve_invalidations = 0;
    uint64_t graph_reserves = 0;
    uint64_t graph_output_extractions = 0;
    uint64_t capture_slot_allocations = 0;
    uint64_t readback_allocations = 0;
    uint64_t readback_copies = 0;
    uint64_t device_to_host_copies = 0;
    uint64_t synchronizations = 0;
    uint64_t batch_peer_reads = 0;
};

LLAMA_API void llama_moe_routing_test_observer_reset(struct llama_context * ctx);
LLAMA_API llama_moe_routing_test_observer llama_moe_routing_test_observer_get(
        const struct llama_context * ctx);
#endif

struct llama_memory_churn_data {
    uint64_t entries_allocated = 0;
    uint64_t entries_released = 0;
    uint64_t entries_overwritten = 0;
    uint64_t memberships_added = 0;
    uint64_t memberships_removed = 0;
    uint64_t sequence_remove_operations = 0;
    uint64_t sequence_copy_operations = 0;
    uint64_t shared_copy_entries = 0;
    uint64_t copied_entries = 0;
    uint64_t reset_operations = 0;
    uint64_t context_shift_operations = 0;
    uint64_t shifted_entries = 0;
    uint64_t prepare_failures = 0;
    uint64_t optimize_attempts = 0;

    llama_memory_churn_data & operator+=(const llama_memory_churn_data & other) {
        entries_allocated          += other.entries_allocated;
        entries_released           += other.entries_released;
        entries_overwritten        += other.entries_overwritten;
        memberships_added          += other.memberships_added;
        memberships_removed        += other.memberships_removed;
        sequence_remove_operations += other.sequence_remove_operations;
        sequence_copy_operations   += other.sequence_copy_operations;
        shared_copy_entries        += other.shared_copy_entries;
        copied_entries             += other.copied_entries;
        reset_operations           += other.reset_operations;
        context_shift_operations   += other.context_shift_operations;
        shifted_entries            += other.shifted_entries;
        prepare_failures           += other.prepare_failures;
        optimize_attempts          += other.optimize_attempts;
        return *this;
    }
};

struct llama_memory_component_diagnostics {
    std::string name;
    std::string kind;
    std::string entry_semantics;
    std::string state = "available";
    bool logical_primary = true;
    bool resident_tokens_supported = false;
    bool physical_sharing_supported = false;
    bool occupied_bytes_is_estimate = true;
    uint64_t capacity_entries = 0;
    uint64_t used_entries = 0;
    uint64_t resident_tokens = 0;
    uint64_t sequences_represented = 0;
    uint64_t sequences_sharing = 0;
    uint64_t shared_entries = 0;
    uint64_t shared_memberships = 0;
    uint64_t shared_groups = 0;
    uint64_t max_fanout = 0;
    uint64_t allocated_bytes = 0;
    uint64_t occupied_bytes_estimate = 0;
    llama_memory_churn_data churn;

    // Deep snapshots retain these memberships for in-process prefix analysis.
    // Telemetry endpoints never expose sequence identifiers.
    struct shared_prefix_entry {
        llama_pos position = -1;
        std::vector<llama_seq_id> sequence_ids;
    };
    bool shared_prefix_entries_available = false;
    std::vector<shared_prefix_entry> shared_prefix_entries;
};

struct llama_memory_diagnostics {
    std::string state = "unsupported";
    std::vector<llama_memory_component_diagnostics> components;
    llama_memory_churn_data churn;
};

struct llama_memory_primary_occupancy {
    bool available = false;
    uint64_t capacity_entries = 0;
    uint64_t used_entries = 0;
};

// A copy of memory data collected at one context boundary. A shallow snapshot
// has only primary occupancy. Set include_diagnostics for a deep snapshot.
struct llama_memory_snapshot {
    llama_memory_primary_occupancy primary_occupancy;
    llama_memory_diagnostics diagnostics;
    bool diagnostics_collected = false;
};

LLAMA_API llama_memory_primary_occupancy llama_get_memory_primary_occupancy(const struct llama_context * ctx);
LLAMA_API llama_memory_diagnostics llama_get_memory_diagnostics(const struct llama_context * ctx);
LLAMA_API llama_memory_snapshot llama_get_memory_snapshot(
        const struct llama_context * ctx,
        bool include_diagnostics);

// Returns the number of leading token positions represented by one physical
// memory entry shared by every supplied sequence. Sequence identifiers are
// inputs only and are never exposed by telemetry endpoints.
LLAMA_API uint64_t llama_get_memory_shared_prefix_length(
        const struct llama_context * ctx,
        const llama_seq_id * sequence_ids,
        size_t sequence_count,
        uint64_t maximum_prefix_tokens);
LLAMA_API uint64_t llama_get_memory_shared_prefix_length(
        const llama_memory_diagnostics & diagnostics,
        const llama_seq_id * sequence_ids,
        size_t sequence_count,
        uint64_t maximum_prefix_tokens);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);
