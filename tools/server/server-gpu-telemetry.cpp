#include "server-gpu-telemetry.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace {

using nvmlReturn_t = int;
using nvmlDevice_t = void *;
using nvmlGpmSample_t = void *;

static constexpr nvmlReturn_t NVML_SUCCESS = 0;
static constexpr nvmlReturn_t NVML_ERROR_NOT_SUPPORTED = 3;
static constexpr unsigned int NVML_GPM_METRIC_SM_UTIL = 2;
static constexpr unsigned int NVML_GPM_METRIC_ANY_TENSOR_UTIL = 5;
static constexpr unsigned int NVML_GPM_METRIC_DRAM_BW_UTIL = 10;
static constexpr size_t NVML_GPM_METRIC_MAX = 333;

struct nvmlGpmMetric_t {
    unsigned int metricId;
    nvmlReturn_t nvmlReturn;
    double value;
    struct {
        char * shortName;
        char * longName;
        char * unit;
    } metricInfo;
};

struct nvmlGpmMetricsGet_t {
    unsigned int version;
    unsigned int numMetrics;
    nvmlGpmSample_t sample1;
    nvmlGpmSample_t sample2;
    nvmlGpmMetric_t metrics[NVML_GPM_METRIC_MAX];
};

struct nvmlGpmSupport_t {
    unsigned int version;
    unsigned int isSupportedDevice;
};

using nvmlInit_v2_fn = nvmlReturn_t (*)();
using nvmlShutdown_fn = nvmlReturn_t (*)();
using nvmlDeviceGetCount_v2_fn = nvmlReturn_t (*)(unsigned int *);
using nvmlDeviceGetHandleByIndex_v2_fn = nvmlReturn_t (*)(unsigned int, nvmlDevice_t *);
using nvmlDeviceGetUUID_fn = nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int);
using nvmlErrorString_fn = const char * (*)(nvmlReturn_t);
using nvmlGpmQueryDeviceSupport_fn = nvmlReturn_t (*)(nvmlDevice_t, nvmlGpmSupport_t *);
using nvmlGpmSampleAlloc_fn = nvmlReturn_t (*)(nvmlGpmSample_t *);
using nvmlGpmSampleFree_fn = nvmlReturn_t (*)(nvmlGpmSample_t);
using nvmlGpmSampleGet_fn = nvmlReturn_t (*)(nvmlDevice_t, nvmlGpmSample_t);
using nvmlGpmMetricsGet_fn = nvmlReturn_t (*)(nvmlGpmMetricsGet_t *);

static int64_t monotonic_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int64_t unix_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static bool disabled_by_environment() {
    const char * value = getenv("LLAMA_TELEMETRY_GPU_GPM");
    if (!value) {
        return false;
    }
    return strcmp(value, "0") == 0 || strcmp(value, "off") == 0 || strcmp(value, "false") == 0;
}

static size_t environment_size(const char * name, size_t fallback, size_t minimum, size_t maximum) {
    const char * value = getenv(name);
    if (!value) {
        return fallback;
    }
    try {
        return std::clamp<size_t>(std::stoull(value), minimum, maximum);
    } catch (...) {
        return fallback;
    }
}

static const char * operation_name(server_gpu_operation_kind kind) {
    switch (kind) {
        case SERVER_GPU_OPERATION_REQUEST:       return "request";
        case SERVER_GPU_OPERATION_PREFILL:       return "prefill";
        case SERVER_GPU_OPERATION_NORMAL_DECODE: return "normal_decode";
        case SERVER_GPU_OPERATION_MTP_DRAFT:     return "mtp_draft";
        case SERVER_GPU_OPERATION_MTP_VERIFY:    return "mtp_verify";
    }
    return "unknown";
}

static const char * timing_semantics_name(server_gpu_timing_semantics semantics) {
    switch (semantics) {
        case SERVER_GPU_TIMING_SUBMISSION_WINDOW:          return "submission_window";
        case SERVER_GPU_TIMING_SYNCHRONIZED_WINDOW:        return "synchronized_gpu_window";
        case SERVER_GPU_TIMING_REQUEST_LIFECYCLE_WINDOW:   return "request_lifecycle_window";
    }
    return "unknown";
}

struct dynamic_library {
#if defined(_WIN32)
    HMODULE handle = nullptr;
#else
    void * handle = nullptr;
#endif

    bool open() {
#if defined(_WIN32)
        std::array<wchar_t, MAX_PATH> system_directory {};
        const UINT length = GetSystemDirectoryW(system_directory.data(), (UINT) system_directory.size());
        if (length == 0 || length >= system_directory.size()) {
            return false;
        }
        std::wstring path(system_directory.data(), length);
        path += L"\\nvml.dll";
        handle = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
        handle = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
        return handle != nullptr;
    }

    void close() {
        if (!handle) {
            return;
        }
#if defined(_WIN32)
        FreeLibrary(handle);
#else
        dlclose(handle);
#endif
        handle = nullptr;
    }

    template <typename T>
    T symbol(const char * name) const {
#if defined(_WIN32)
        return reinterpret_cast<T>(GetProcAddress(handle, name));
#else
        return reinterpret_cast<T>(dlsym(handle, name));
#endif
    }

    ~dynamic_library() {
        close();
    }
};

struct gpu_reading {
    unsigned int metric_id = 0;
    nvmlReturn_t status = -1;
    double value = 0.0;
};

struct gpu_interval {
    uint64_t sequence = 0;
    unsigned int device_index = 0;
    std::string device_uuid;
    int64_t started_monotonic_us = 0;
    int64_t completed_monotonic_us = 0;
    int64_t started_unix_ms = 0;
    int64_t completed_unix_ms = 0;
    std::array<gpu_reading, 3> readings;
};

struct gpu_operation {
    uint64_t sequence = 0;
    server_gpu_operation_kind kind = SERVER_GPU_OPERATION_PREFILL;
    std::array<char, 72> trace_id {};
    int32_t slot_id = -1;
    int64_t started_monotonic_us = 0;
    int64_t completed_monotonic_us = 0;
    int64_t token_position = -1;
    int64_t output_ordinal = -1;
    int64_t logical_step = -1;
    int64_t actual_target_pass = -1;
    int64_t proposal_position = -1;
    server_gpu_timing_semantics timing_semantics = SERVER_GPU_TIMING_SUBMISSION_WINDOW;
};

struct gpu_device {
    unsigned int index = 0;
    nvmlDevice_t handle = nullptr;
    std::string uuid;
    nvmlGpmSample_t first = nullptr;
    nvmlGpmSample_t second = nullptr;
    int64_t first_monotonic_us = 0;
    int64_t first_unix_ms = 0;
};

} // namespace

struct server_gpu_telemetry::impl {
    dynamic_library library;
    nvmlInit_v2_fn init = nullptr;
    nvmlShutdown_fn shutdown = nullptr;
    nvmlDeviceGetCount_v2_fn device_get_count = nullptr;
    nvmlDeviceGetHandleByIndex_v2_fn device_get_handle = nullptr;
    nvmlDeviceGetUUID_fn device_get_uuid = nullptr;
    nvmlErrorString_fn error_string = nullptr;
    nvmlGpmQueryDeviceSupport_fn query_support = nullptr;
    nvmlGpmSampleAlloc_fn sample_alloc = nullptr;
    nvmlGpmSampleFree_fn sample_free = nullptr;
    nvmlGpmSampleGet_fn sample_get = nullptr;
    nvmlGpmMetricsGet_fn metrics_get = nullptr;

    mutable std::mutex mutex;
    std::condition_variable wake;
    std::thread worker;
    std::atomic<bool> stop_requested {false};
    std::atomic<bool> collecting {false};
    bool nvml_initialized = false;
    std::string state = "initializing";
    std::string reason = "Collector has not started.";
    size_t sampling_interval_ms = 200;
    size_t interval_capacity = 1500;
    uint64_t next_interval_sequence = 1;
    uint64_t dropped_intervals = 0;
    std::deque<gpu_interval> intervals;
    std::vector<gpu_device> devices;

    static constexpr size_t operation_capacity = 65536;
    mutable std::mutex operation_mutex;
    std::array<gpu_operation, operation_capacity> operations {};
    uint64_t next_operation_sequence = 1;
    std::atomic<uint64_t> dropped_operations {0};

    std::string nvml_error(nvmlReturn_t result) const {
        const char * text = error_string ? error_string(result) : nullptr;
        return text ? text : "NVML error " + std::to_string(result);
    }

    bool resolve_symbols() {
        init = library.symbol<nvmlInit_v2_fn>("nvmlInit_v2");
        shutdown = library.symbol<nvmlShutdown_fn>("nvmlShutdown");
        device_get_count = library.symbol<nvmlDeviceGetCount_v2_fn>("nvmlDeviceGetCount_v2");
        device_get_handle = library.symbol<nvmlDeviceGetHandleByIndex_v2_fn>("nvmlDeviceGetHandleByIndex_v2");
        device_get_uuid = library.symbol<nvmlDeviceGetUUID_fn>("nvmlDeviceGetUUID");
        error_string = library.symbol<nvmlErrorString_fn>("nvmlErrorString");
        query_support = library.symbol<nvmlGpmQueryDeviceSupport_fn>("nvmlGpmQueryDeviceSupport");
        sample_alloc = library.symbol<nvmlGpmSampleAlloc_fn>("nvmlGpmSampleAlloc");
        sample_free = library.symbol<nvmlGpmSampleFree_fn>("nvmlGpmSampleFree");
        sample_get = library.symbol<nvmlGpmSampleGet_fn>("nvmlGpmSampleGet");
        metrics_get = library.symbol<nvmlGpmMetricsGet_fn>("nvmlGpmMetricsGet");
        return init && shutdown && device_get_count && device_get_handle && device_get_uuid &&
            query_support && sample_alloc && sample_free && sample_get && metrics_get;
    }

    bool initialize() {
        sampling_interval_ms = environment_size("LLAMA_TELEMETRY_GPU_GPM_INTERVAL_MS", 200, 100, 5000);
        interval_capacity = environment_size("LLAMA_TELEMETRY_GPU_GPM_INTERVAL_CAPACITY", 1500, 64, 100000);
        if (disabled_by_environment()) {
            state = "disabled";
            reason = "Disabled by LLAMA_TELEMETRY_GPU_GPM.";
            return false;
        }
        if (!library.open()) {
            state = "unsupported";
            reason = "The NVIDIA Management Library is not installed.";
            return false;
        }
        if (!resolve_symbols()) {
            state = "unsupported";
            reason = "The installed NVML does not export the required GPM APIs.";
            return false;
        }
        nvmlReturn_t result = init();
        if (result != NVML_SUCCESS) {
            state = "unavailable";
            reason = "NVML initialization failed: " + nvml_error(result);
            return false;
        }
        nvml_initialized = true;

        unsigned int count = 0;
        result = device_get_count(&count);
        if (result != NVML_SUCCESS) {
            state = "unavailable";
            reason = "NVML could not enumerate devices: " + nvml_error(result);
            return false;
        }
        for (unsigned int index = 0; index < count; ++index) {
            gpu_device device;
            device.index = index;
            if (device_get_handle(index, &device.handle) != NVML_SUCCESS) {
                continue;
            }
            nvmlGpmSupport_t support {1, 0};
            if (query_support(device.handle, &support) != NVML_SUCCESS || !support.isSupportedDevice) {
                continue;
            }
            std::array<char, 96> uuid {};
            if (device_get_uuid(device.handle, uuid.data(), (unsigned int) uuid.size()) == NVML_SUCCESS) {
                device.uuid = uuid.data();
            } else {
                device.uuid = "device-" + std::to_string(index);
            }
            if (sample_alloc(&device.first) != NVML_SUCCESS || sample_alloc(&device.second) != NVML_SUCCESS) {
                if (device.first) {
                    sample_free(device.first);
                }
                if (device.second) {
                    sample_free(device.second);
                }
                continue;
            }
            if (sample_get(device.handle, device.first) != NVML_SUCCESS) {
                sample_free(device.first);
                sample_free(device.second);
                continue;
            }
            device.first_monotonic_us = monotonic_us();
            device.first_unix_ms = unix_ms();
            devices.push_back(std::move(device));
        }
        if (devices.empty()) {
            state = "unsupported";
            reason = "No NVML device reports GPM support. GPM requires a supported Hopper-or-newer GPU and is unavailable on Windows WDDM.";
            return false;
        }
        state = "available";
        reason = "Asynchronous llama-server-owned NVML GPM collection is active.";
        return true;
    }

    void cleanup() {
        for (gpu_device & device : devices) {
            if (device.first && sample_free) {
                sample_free(device.first);
            }
            if (device.second && sample_free) {
                sample_free(device.second);
            }
        }
        devices.clear();
        if (nvml_initialized && shutdown) {
            shutdown();
        }
        nvml_initialized = false;
        library.close();
    }

    void collect_loop() {
        collecting.store(true, std::memory_order_release);
        while (!stop_requested.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lock(mutex);
            if (wake.wait_for(lock, std::chrono::milliseconds(sampling_interval_ms), [this]() {
                    return stop_requested.load(std::memory_order_acquire);
                })) {
                break;
            }
            lock.unlock();

            for (gpu_device & device : devices) {
                const nvmlReturn_t sample_result = sample_get(device.handle, device.second);
                const int64_t completed_monotonic_us = monotonic_us();
                const int64_t completed_unix_ms = unix_ms();
                if (sample_result != NVML_SUCCESS) {
                    continue;
                }

                nvmlGpmMetricsGet_t request {};
                request.version = 1;
                request.numMetrics = 3;
                request.sample1 = device.first;
                request.sample2 = device.second;
                request.metrics[0].metricId = NVML_GPM_METRIC_SM_UTIL;
                request.metrics[1].metricId = NVML_GPM_METRIC_ANY_TENSOR_UTIL;
                request.metrics[2].metricId = NVML_GPM_METRIC_DRAM_BW_UTIL;
                const nvmlReturn_t metrics_result = metrics_get(&request);

                gpu_interval interval;
                interval.device_index = device.index;
                interval.device_uuid = device.uuid;
                interval.started_monotonic_us = device.first_monotonic_us;
                interval.completed_monotonic_us = completed_monotonic_us;
                interval.started_unix_ms = device.first_unix_ms;
                interval.completed_unix_ms = completed_unix_ms;
                for (size_t index = 0; index < interval.readings.size(); ++index) {
                    interval.readings[index].metric_id = request.metrics[index].metricId;
                    interval.readings[index].status = metrics_result == NVML_SUCCESS ? request.metrics[index].nvmlReturn : metrics_result;
                    interval.readings[index].value = request.metrics[index].value;
                }

                lock.lock();
                interval.sequence = next_interval_sequence++;
                intervals.push_back(std::move(interval));
                while (intervals.size() > interval_capacity) {
                    intervals.pop_front();
                    dropped_intervals++;
                }
                lock.unlock();

                std::swap(device.first, device.second);
                device.first_monotonic_us = completed_monotonic_us;
                device.first_unix_ms = completed_unix_ms;
            }
        }
        collecting.store(false, std::memory_order_release);
    }
};

server_gpu_telemetry::server_gpu_telemetry() : pimpl(std::make_unique<impl>()) {}

server_gpu_telemetry::~server_gpu_telemetry() {
    stop();
}

void server_gpu_telemetry::start() {
    if (pimpl->worker.joinable() || pimpl->collecting.load(std::memory_order_acquire)) {
        return;
    }
    pimpl->stop_requested.store(false, std::memory_order_release);
    if (!pimpl->initialize()) {
        // Keep the typed capability state/reason, but do not retain an initialized
        // NVML session or loaded module when this host cannot collect GPM data.
        pimpl->cleanup();
        return;
    }
    pimpl->worker = std::thread([this]() { pimpl->collect_loop(); });
}

void server_gpu_telemetry::stop() {
    pimpl->stop_requested.store(true, std::memory_order_release);
    pimpl->wake.notify_all();
    if (pimpl->worker.joinable()) {
        pimpl->worker.join();
    }
    pimpl->cleanup();
}

bool server_gpu_telemetry::is_collecting() const {
    return pimpl->collecting.load(std::memory_order_acquire);
}

void server_gpu_telemetry::record_operation(
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
        server_gpu_timing_semantics timing_semantics) {
    if (!is_collecting()) {
        return;
    }
    std::unique_lock<std::mutex> lock(pimpl->operation_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        pimpl->dropped_operations.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const uint64_t sequence = pimpl->next_operation_sequence++;
    gpu_operation & operation = pimpl->operations[(sequence - 1) % impl::operation_capacity];
    operation = {};
    operation.sequence = sequence;
    operation.kind = kind;
    const size_t trace_length = std::min(trace_id.size(), operation.trace_id.size() - 1);
    memcpy(operation.trace_id.data(), trace_id.data(), trace_length);
    operation.trace_id[trace_length] = '\0';
    operation.slot_id = slot_id;
    operation.started_monotonic_us = started_monotonic_us;
    operation.completed_monotonic_us = completed_monotonic_us;
    operation.token_position = token_position;
    operation.output_ordinal = output_ordinal;
    operation.logical_step = logical_step;
    operation.actual_target_pass = actual_target_pass;
    operation.proposal_position = proposal_position;
    operation.timing_semantics = timing_semantics;
    if (sequence > impl::operation_capacity) {
        pimpl->dropped_operations.fetch_add(1, std::memory_order_relaxed);
    }
}

json server_gpu_telemetry::capability_json() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return {
        {"state", pimpl->state},
        {"reason", pimpl->reason},
        {"owner", "llama.cpp/llama-server"},
        {"collector", "asynchronous NVML GPM"},
        {"llamascope_queries_nvml", false},
        {"sampling_interval_ms", pimpl->sampling_interval_ms},
        {"metrics", json::array({
            {{"id", NVML_GPM_METRIC_SM_UTIL}, {"name", "sm_utilization"}, {"unit", "percent"}, {"state", pimpl->state}, {"reason", pimpl->reason}},
            {{"id", NVML_GPM_METRIC_ANY_TENSOR_UTIL}, {"name", "tensor_core_utilization"}, {"unit", "percent"}, {"state", pimpl->state}, {"reason", pimpl->reason}},
            {{"id", NVML_GPM_METRIC_DRAM_BW_UTIL}, {"name", "dram_bandwidth_utilization"}, {"unit", "percent"}, {"state", pimpl->state}, {"reason", pimpl->reason}},
        })},
        {"operation_phases", json::array({"request", "prefill", "normal_decode", "mtp_draft", "mtp_verify"})},
        {"measurement_semantics", "one shared interval-derived GPM measurement may be associated with multiple token operations"},
    };
}

json server_gpu_telemetry::snapshot_json(
        const std::string & server_instance_id,
        uint64_t interval_cursor,
        size_t interval_limit,
        const std::string & trace_id) const {
    json source = capability_json();
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    interval_limit = std::clamp<size_t>(interval_limit, 1, 4096);

    uint64_t operation_begin = 1;
    uint64_t operation_end = 1;
    int64_t trace_first_time = INT64_MAX;
    int64_t trace_last_time = 0;
    {
        std::lock_guard<std::mutex> operation_lock(pimpl->operation_mutex);
        operation_begin = pimpl->next_operation_sequence > impl::operation_capacity
            ? pimpl->next_operation_sequence - impl::operation_capacity
            : 1;
        operation_end = pimpl->next_operation_sequence;
        if (!trace_id.empty()) {
            for (uint64_t sequence = operation_begin; sequence < operation_end; ++sequence) {
                const gpu_operation & operation = pimpl->operations[(sequence - 1) % impl::operation_capacity];
                if (operation.sequence == sequence && trace_id == operation.trace_id.data()) {
                    trace_first_time = std::min(trace_first_time, operation.started_monotonic_us);
                    trace_last_time = std::max(trace_last_time, operation.completed_monotonic_us);
                }
            }
        }
    }

    json interval_json = json::array();
    uint64_t cursor = interval_cursor;
    int64_t first_time = INT64_MAX;
    int64_t last_time = 0;
    for (const gpu_interval & interval : pimpl->intervals) {
        if (interval.sequence <= interval_cursor || interval_json.size() >= interval_limit) {
            continue;
        }
        if (!trace_id.empty() &&
                (trace_first_time == INT64_MAX ||
                 interval.completed_monotonic_us < trace_first_time ||
                 interval.started_monotonic_us > trace_last_time)) {
            continue;
        }
        json readings = json::object();
        for (const gpu_reading & reading : interval.readings) {
            const char * name = reading.metric_id == NVML_GPM_METRIC_SM_UTIL ? "sm_utilization" :
                reading.metric_id == NVML_GPM_METRIC_ANY_TENSOR_UTIL ? "tensor_core_utilization" :
                "dram_bandwidth_utilization";
            const char * state = reading.status == NVML_SUCCESS ? "available" :
                reading.status == NVML_ERROR_NOT_SUPPORTED ? "unsupported" :
                "unavailable";
            readings[name] = {
                {"metric_id", reading.metric_id},
                {"state", state},
                {"value_percent", reading.status == NVML_SUCCESS ? json(reading.value) : json(nullptr)},
                {"nvml_return", reading.status},
                {"reason", reading.status == NVML_SUCCESS
                    ? "Measured by the asynchronous llama.cpp/llama-server NVML GPM collector."
                    : pimpl->nvml_error(reading.status)},
            };
        }
        interval_json.push_back({
            {"sequence", interval.sequence},
            {"device_index", interval.device_index},
            {"device_uuid", interval.device_uuid},
            {"start_monotonic_us", interval.started_monotonic_us},
            {"end_monotonic_us", interval.completed_monotonic_us},
            {"start_unix_ms", interval.started_unix_ms},
            {"end_unix_ms", interval.completed_unix_ms},
            {"readings", std::move(readings)},
        });
        cursor = interval.sequence;
        first_time = std::min(first_time, interval.started_monotonic_us);
        last_time = std::max(last_time, interval.completed_monotonic_us);
    }

    std::vector<gpu_operation> operation_snapshot;
    bool operations_truncated = false;
    if (first_time != INT64_MAX) {
        std::lock_guard<std::mutex> operation_lock(pimpl->operation_mutex);
        operation_snapshot.reserve((size_t) std::min<uint64_t>(operation_end - operation_begin, 16384));
        for (uint64_t sequence = operation_begin; sequence < operation_end; ++sequence) {
            const gpu_operation & operation = pimpl->operations[(sequence - 1) % impl::operation_capacity];
            if (operation.sequence != sequence ||
                    operation.completed_monotonic_us < first_time || operation.started_monotonic_us > last_time) {
                continue;
            }
            if (operation_snapshot.size() >= 16384) {
                operations_truncated = true;
                break;
            }
            operation_snapshot.push_back(operation);
        }
    }

    json operation_json = json::array();
    for (const gpu_operation & operation : operation_snapshot) {
        operation_json.push_back({
            {"sequence", operation.sequence},
            {"trace_id", operation.trace_id.data()},
            {"is_requested_trace", trace_id.empty() ? json(nullptr) : json(trace_id == operation.trace_id.data())},
            {"slot_id", operation.slot_id},
            {"phase", operation_name(operation.kind)},
            {"start_monotonic_us", operation.started_monotonic_us},
            {"end_monotonic_us", operation.completed_monotonic_us},
            {"token_position", operation.token_position >= 0 ? json(operation.token_position) : json(nullptr)},
            {"output_ordinal", operation.output_ordinal >= 0 ? json(operation.output_ordinal) : json(nullptr)},
            {"logical_step", operation.logical_step >= 0 ? json(operation.logical_step) : json(nullptr)},
            {"actual_target_pass", operation.actual_target_pass >= 0 ? json(operation.actual_target_pass) : json(nullptr)},
            {"proposal_position", operation.proposal_position >= 0 ? json(operation.proposal_position) : json(nullptr)},
            {"timing_semantics", timing_semantics_name(operation.timing_semantics)},
        });
    }

    const uint64_t oldest = pimpl->intervals.empty() ? pimpl->next_interval_sequence : pimpl->intervals.front().sequence;
    return {
        {"schema_version", 1},
        {"server_instance_id", server_instance_id},
        {"state", pimpl->state},
        {"reason", pimpl->reason},
        {"source", std::move(source)},
        {"cursor", cursor},
        {"oldest_sequence", oldest},
        {"next_sequence", pimpl->next_interval_sequence},
        {"gap", interval_cursor != 0 && interval_cursor < oldest - 1},
        {"dropped_intervals", pimpl->dropped_intervals},
        {"dropped_operations", pimpl->dropped_operations.load(std::memory_order_relaxed)},
        {"operations_truncated", operations_truncated},
        {"trace_filter", trace_id.empty() ? json(nullptr) : json(trace_id)},
        {"intervals", std::move(interval_json)},
        {"operations", std::move(operation_json)},
    };
}
