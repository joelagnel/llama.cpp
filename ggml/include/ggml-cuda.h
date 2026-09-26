#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// Optional CUDA graph telemetry: one event per graph compute call on a backend.
// Resolved at runtime through ggml_backend_reg_get_proc_address() under the
// names "ggml_backend_cuda_graph_events_enable" and
// "ggml_backend_cuda_graph_events_drain". Recording is off until enabled.
enum ggml_cuda_graph_event_kind {
    GGML_CUDA_GRAPH_EVENT_EAGER_DISABLED       = 0, // CUDA graphs disabled (build, env or GPU arch)
    GGML_CUDA_GRAPH_EVENT_EAGER_INCOMPATIBLE   = 1, // graph contains ops that cannot be captured
    GGML_CUDA_GRAPH_EVENT_EAGER_WARMUP         = 2, // new or changed graph, run eagerly before capture
    GGML_CUDA_GRAPH_EVENT_EAGER_WARMUP_RESET   = 3, // captured graph changed, warmup restarts
    GGML_CUDA_GRAPH_EVENT_CAPTURE_INSTANTIATE  = 4, // captured and instantiated a new executable
    GGML_CUDA_GRAPH_EVENT_CAPTURE_UPDATE       = 5, // captured and updated the existing executable
    GGML_CUDA_GRAPH_EVENT_CAPTURE_REINSTANTIATE = 6, // executable update failed, re-instantiated
    GGML_CUDA_GRAPH_EVENT_LAUNCH               = 7, // launched the cached executable unchanged
};

struct ggml_cuda_graph_event {
    int64_t  begin_us;   // ggml_time_us() at graph compute entry
    int64_t  end_us;     // ggml_time_us() when host submission returned
    int64_t  build_us;   // host time in end-capture, instantiate and update
    uint64_t key;        // graph cache key (address of the first node)
    uint64_t uid;        // ggml cgraph uid, 0 when unset
    int32_t  kind;       // enum ggml_cuda_graph_event_kind
    int32_t  n_nodes;    // nodes in the ggml graph
    int32_t  n_cached;   // CUDA graphs cached on this backend after the call
    int32_t  n_evicted;  // CUDA graphs evicted as unused during the call
};

typedef void   (*ggml_backend_cuda_graph_events_enable_t)(ggml_backend_t backend, bool enable);
// Copies up to capacity pending events in order and removes them. *dropped
// receives the number of events lost to ring overflow since the last drain.
typedef size_t (*ggml_backend_cuda_graph_events_drain_t)(ggml_backend_t backend, struct ggml_cuda_graph_event * events, size_t capacity, uint64_t * dropped);

// Optional step profiling: the GPU time of every graph compute call, measured
// with CUDA events, and on request the GPU time of every kernel launch of a
// call, which then runs eagerly without a CUDA graph. Resolved through
// ggml_backend_reg_get_proc_address() under the names
// "ggml_backend_cuda_profile_enable", "ggml_backend_cuda_profile_next",
// "ggml_backend_cuda_profile_drain", "ggml_backend_cuda_nvtx_push" and
// "ggml_backend_cuda_nvtx_pop". While enabled, eager calls also emit one NVTX
// range per kernel launch named after its ggml node.
struct ggml_cuda_profile_op {
    char        name[64];    // ggml node name, e.g. "ffn_down-12"
    const char * kernel;     // static kernel path name, e.g. "mmvq", "mmq_stream_k", "fattn_mma_f16"; NULL when not recorded
    int32_t     op;          // enum ggml_op of the first node of the launch
    int32_t     n_fused;     // further nodes fused into this launch
    int32_t     src0_type;   // enum ggml_type of src[0], -1 when absent
    int64_t     ne[4];       // node shape
    int64_t     src0_ne[4];  // src[0] shape, zero when absent
    int64_t     src1_ne[4];  // src[1] shape, zero when absent
    float       gpu_us;      // GPU time from the first to the last kernel of the launch
};

struct ggml_cuda_profile_compute {
    uint64_t tag;            // tag set by the last ggml_backend_cuda_profile_next call
    int64_t  begin_us;       // ggml_time_us() at graph compute entry
    int64_t  end_us;         // ggml_time_us() when host submission returned
    float    gpu_us;         // GPU time of the call on its stream, -1 when unavailable
    int32_t  n_nodes;        // nodes in the ggml graph
    bool     cuda_graph;     // ran as a CUDA graph launch, false when eager
    int32_t  n_ops;          // entries in ops, 0 unless op timing was requested
    const struct ggml_cuda_profile_op * ops; // valid until the next drain call
};

typedef void   (*ggml_backend_cuda_profile_enable_t)(ggml_backend_t backend, bool enable);
// Tags the following graph compute calls and whether they time every kernel launch.
typedef void   (*ggml_backend_cuda_profile_next_t)(ggml_backend_t backend, uint64_t tag, bool ops);
// Waits for the GPU work of pending calls, then copies up to capacity of them in
// order and removes them. *dropped receives calls lost to overflow since the last drain.
typedef size_t (*ggml_backend_cuda_profile_drain_t)(ggml_backend_t backend, struct ggml_cuda_profile_compute * computes, size_t capacity, uint64_t * dropped);
typedef void   (*ggml_backend_cuda_nvtx_push_t)(const char * name);
typedef void   (*ggml_backend_cuda_nvtx_pop_t)(void);

#ifdef  __cplusplus
}
#endif
