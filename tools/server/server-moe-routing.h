#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

enum server_moe_routing_capture_result {
    SERVER_MOE_ROUTING_CAPTURED,
    SERVER_MOE_ROUTING_INVALID,
    SERVER_MOE_ROUTING_CAP_DROPPED,
};

struct server_moe_routing_capture_counts {
    uint64_t total = 0;
    uint64_t captured = 0;
    uint64_t invalid = 0;
    uint64_t cap_dropped = 0;
};

inline bool server_moe_routing_assignment_is_valid(
        int32_t layer_index,
        int32_t expert_index,
        int32_t model_layers,
        int32_t configured_experts) {
    return layer_index >= 0 && layer_index < model_layers
        && expert_index >= 0 && expert_index < configured_experts;
}

inline bool server_moe_routing_weight_is_usable(float effective_weight) {
    return std::isfinite(effective_weight) && effective_weight >= 0.0f;
}

inline server_moe_routing_capture_result server_moe_routing_capture(
        server_moe_routing_capture_counts & counts,
        bool valid,
        uint64_t activation_count,
        size_t activation_limit) {
    counts.total += activation_count;
    if (!valid) {
        counts.invalid += activation_count;
        return SERVER_MOE_ROUTING_INVALID;
    }
    if (counts.captured > activation_limit || activation_count > activation_limit - counts.captured) {
        counts.cap_dropped += activation_count;
        return SERVER_MOE_ROUTING_CAP_DROPPED;
    }
    counts.captured += activation_count;
    return SERVER_MOE_ROUTING_CAPTURED;
}

inline bool server_moe_routing_was_truncated(const server_moe_routing_capture_counts & counts) {
    return counts.cap_dropped > 0;
}

inline bool server_moe_routing_producer_coverage_is_partial(
        uint64_t invalid_rows,
        uint64_t unavailable_rows,
        uint64_t unlinked_rows,
        uint64_t unlocated_rows,
        bool interrupted,
        bool source_unavailable) {
    return invalid_rows > 0 || unavailable_rows > 0 || unlinked_rows > 0 || unlocated_rows > 0 || interrupted || source_unavailable;
}

inline const char * server_moe_routing_capture_state(
        const server_moe_routing_capture_counts & counts,
        bool has_data) {
    return server_moe_routing_was_truncated(counts)
        ? "truncated"
        : has_data ? "available" : "no_data";
}
