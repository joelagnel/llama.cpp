#pragma once

#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

using json = nlohmann::ordered_json;

enum server_gpu_operation_kind {
    SERVER_GPU_OPERATION_REQUEST,
    SERVER_GPU_OPERATION_PREFILL,
    SERVER_GPU_OPERATION_NORMAL_DECODE,
    SERVER_GPU_OPERATION_MTP_DRAFT,
    SERVER_GPU_OPERATION_MTP_VERIFY,
};

enum server_gpu_timing_semantics {
    SERVER_GPU_TIMING_SUBMISSION_WINDOW,
    SERVER_GPU_TIMING_SYNCHRONIZED_WINDOW,
    SERVER_GPU_TIMING_REQUEST_LIFECYCLE_WINDOW,
};

// Native-owned, optional NVML GPM collection. The collector is asynchronous and
// never calls NVML from the inference thread. The hot path only enters
// record_operation() when collection is active. It uses a non-blocking ring
// write and counts a drop if an endpoint snapshot is copying the ring.
class server_gpu_telemetry {
public:
    server_gpu_telemetry();
    ~server_gpu_telemetry();

    server_gpu_telemetry(const server_gpu_telemetry &) = delete;
    server_gpu_telemetry & operator=(const server_gpu_telemetry &) = delete;

    void start();
    void stop();

    bool is_collecting() const;

    void record_operation(
            server_gpu_operation_kind kind,
            const std::string & trace_id,
            int32_t slot_id,
            int64_t started_monotonic_us,
            int64_t completed_monotonic_us,
            int64_t token_position,
            int64_t output_ordinal,
            int64_t logical_step,
            int64_t actual_target_pass,
            int64_t proposal_position,
            server_gpu_timing_semantics timing_semantics);

    json capability_json() const;
    json snapshot_json(
            const std::string & server_instance_id,
            uint64_t interval_cursor,
            size_t interval_limit,
            const std::string & trace_id) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
