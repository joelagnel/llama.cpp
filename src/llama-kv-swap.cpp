#include "ggml-cpp.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t KV_SWAP_PAGE_CELLS      = 128;
constexpr uint32_t KV_SWAP_PAGES_PER_GROUP = 8;

enum page_state {
    PAGE_STATE_RESIDENT,
    PAGE_STATE_D2H,
    PAGE_STATE_HOST,
    PAGE_STATE_H2D,
};

struct swap_cell {
    uint32_t          idx = 0;
    llama_pos         pos = -1;
    llama_kv_cell_ext ext;
};

struct swap_page {
    std::vector<swap_cell> cells;
    size_t                 host_offset = 0;
    page_state             state       = PAGE_STATE_RESIDENT;
    int64_t                last_use_us = 0;
};

enum transfer_direction {
    TRANSFER_DIRECTION_D2H,
    TRANSFER_DIRECTION_H2D,
};

struct swap_group {
    std::vector<swap_page>              pages;
    ggml_backend_buffer_ptr             host_buffer;
    std::vector<ggml_backend_event_ptr> events;
    std::atomic<bool>                   complete{ false };
    bool                                applied      = false;
    transfer_direction                  direction    = TRANSFER_DIRECTION_D2H;
    int64_t                             submitted_us = 0;
    std::atomic<int64_t>                completed_us{ 0 };
    size_t                              data_bytes   = 0;
    size_t                              buffer_bytes = 0;
};

struct swap_sequence {
    llama_seq_id                             seq_id = -1;
    llama_kv_swap_state                      state  = LLAMA_KV_SWAP_STATE_RESIDENT;
    std::vector<std::unique_ptr<swap_group>> groups;
    std::vector<ggml_backend_event_ptr>      compute_events;
    size_t                                   data_bytes        = 0;
    size_t                                   buffer_bytes      = 0;
    uint32_t                                 cell_count        = 0;
    int64_t                                  last_use_us       = 0;
    int64_t                                  transfer_start_us = 0;
    bool                                     discard           = false;
};

static ggml_backend_buffer_t tensor_buffer(const ggml_tensor * tensor) {
    return tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
}

static ggml_backend_dev_t tensor_device(const ggml_tensor * tensor) {
    ggml_backend_buffer_t buffer = tensor_buffer(tensor);
    if (!buffer) {
        return nullptr;
    }
    return ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer));
}

static void copy_reason(char * dst, size_t size, const std::string & value) {
    if (!dst || size == 0) {
        return;
    }

    const size_t n = std::min(size - 1, value.size());
    memcpy(dst, value.data(), n);
    dst[n] = '\0';
}

}  // namespace

struct llama_kv_swap {
    struct channel {
        ggml_backend_dev_t         device          = nullptr;
        ggml_backend_t             compute_backend = nullptr;
        ggml_backend_ptr           copy_backend;
        ggml_backend_buffer_type_t host_buft = nullptr;
    };

    llama_context *  ctx;
    llama_kv_cache * kv;
    const size_t     max_bytes;

    std::vector<channel>                                   channels;
    std::map<llama_seq_id, std::unique_ptr<swap_sequence>> sequences;

    llama_kv_swap_stats stats = {};
    std::string         error;

    std::mutex               worker_mutex;
    std::condition_variable  worker_cv;
    std::condition_variable  completion_cv;
    std::deque<swap_group *> worker_queue;
    bool                     worker_stop = false;
    std::thread              worker;

    llama_kv_swap(llama_context * ctx, llama_kv_cache * kv, size_t max_bytes) : ctx(ctx), kv(kv), max_bytes(max_bytes) {
        stats.page_cells   = KV_SWAP_PAGE_CELLS;
        stats.device_cells = kv->get_size();

        std::set<ggml_backend_dev_t> devices;
        for (const auto & layer : kv->layers) {
            devices.insert(tensor_device(layer.k_stream[0]));
            if (layer.v_stream[0]) {
                devices.insert(tensor_device(layer.v_stream[0]));
            }
        }

        for (ggml_backend_dev_t device : devices) {
            channel cur;
            cur.device          = device;
            cur.compute_backend = ctx->get_backend(device);
            cur.copy_backend.reset(ggml_backend_dev_init(device, nullptr));
            cur.host_buft = ggml_backend_dev_host_buffer_type(device);

            if (!cur.compute_backend || !cur.copy_backend || !cur.host_buft) {
                throw std::runtime_error("failed to initialize the KV swap copy backend");
            }

            channels.push_back(std::move(cur));
        }

        worker = std::thread([this]() { worker_loop(); });

        LLAMA_LOG_INFO("%s: page cells = %u, group pages = %u, host budget = %.2f MiB, host buffer = %s\n", __func__,
                       KV_SWAP_PAGE_CELLS, KV_SWAP_PAGES_PER_GROUP, max_bytes / 1024.0 / 1024.0,
                       ggml_backend_buft_name(channels[0].host_buft));
    }

    ~llama_kv_swap() {
        {
            std::lock_guard<std::mutex> lock(worker_mutex);
            worker_stop = true;
        }
        worker_cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }

    static llama_kv_cache * get_cache(llama_context * ctx, std::string & reason) {
        if (!ctx) {
            reason = "context is null";
            return nullptr;
        }

        if (!ctx->get_cparams().kv_unified) {
            reason = "KV swap requires a unified KV cache";
            return nullptr;
        }

        llama_memory_i * memory = ctx->get_memory();
        if (auto * cache = dynamic_cast<llama_kv_cache *>(memory)) {
            return cache;
        }

        if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory)) {
            return hybrid->get_mem_attn();
        }

        reason = "KV swap supports plain or hybrid attention memory only";
        return nullptr;
    }

    static bool supported(llama_context * ctx, std::string & reason) {
        llama_kv_cache * kv = get_cache(ctx, reason);
        if (!kv) {
            return false;
        }

        if (kv->n_stream != 1) {
            reason = "KV swap requires a unified attention cache";
            return false;
        }
        if (kv->other) {
            reason = "KV swap does not support a shared target/draft KV cache";
            return false;
        }
        if (kv->layers.empty()) {
            reason = "the model has no growing attention KV cache";
            return false;
        }

        std::set<ggml_backend_dev_t> devices;
        for (const auto & layer : kv->layers) {
            std::vector<ggml_tensor *> tensors = { layer.k_stream[0] };
            if (layer.v_stream[0]) {
                tensors.push_back(layer.v_stream[0]);
            }

            for (const ggml_tensor * tensor : tensors) {
                ggml_backend_dev_t device = tensor_device(tensor);
                if (!device) {
                    reason = "KV swap requires the attention KV cache to be offloaded";
                    return false;
                }

                ggml_backend_reg_t reg      = ggml_backend_dev_backend_reg(device);
                const char *       reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
                if (!reg_name || std::string(reg_name).find("CUDA") == std::string::npos) {
                    reason = "the KV swap MVP currently supports CUDA only";
                    return false;
                }

                if (!ctx->get_backend(device)) {
                    reason = "the context has no compute backend for an attention KV device";
                    return false;
                }
                if (!ggml_backend_dev_host_buffer_type(device)) {
                    reason = "the CUDA device has no pinned host buffer type";
                    return false;
                }

                devices.insert(device);
            }

            if (kv->v_trans && layer.v_stream[0] && ggml_blck_size(layer.v_stream[0]->type) != 1) {
                reason = "transposed quantized V tensors are not supported by KV swap";
                return false;
            }
        }

        for (ggml_backend_dev_t device : devices) {
            ggml_backend_event_t event = ggml_backend_event_new(device);
            if (!event) {
                reason = "the CUDA device does not support completion events";
                return false;
            }
            ggml_backend_event_free(event);
        }

        reason.clear();
        return true;
    }

    channel * get_channel(ggml_backend_dev_t device) {
        for (auto & cur : channels) {
            if (cur.device == device) {
                return &cur;
            }
        }
        return nullptr;
    }

    const channel * get_channel(ggml_backend_dev_t device) const {
        for (const auto & cur : channels) {
            if (cur.device == device) {
                return &cur;
            }
        }
        return nullptr;
    }

    size_t cell_bytes() const {
        size_t result = 0;
        for (const auto & layer : kv->layers) {
            const uint32_t il = layer.il;
            result += ggml_row_size(layer.k->type, kv->hparams.n_embd_k_gqa(il));

            if (!layer.v) {
                continue;
            }

            if (kv->v_trans) {
                result += ggml_type_size(layer.v->type) * kv->hparams.n_embd_v_gqa(il);
            } else {
                result += ggml_row_size(layer.v->type, kv->hparams.n_embd_v_gqa(il));
            }
        }
        return result;
    }

    uint32_t sequence_cells(llama_seq_id seq_id) const {
        if (seq_id < 0 || (size_t) seq_id >= kv->seq_to_stream.size()) {
            return 0;
        }

        const auto & cells  = kv->v_cells[kv->seq_to_stream[seq_id]];
        uint32_t     result = 0;
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.is_empty(i) && cells.seq_has(i, seq_id)) {
                ++result;
            }
        }
        return result;
    }

    uint32_t device_cells_used() const { return kv->v_cells[0].get_used(); }

    void worker_loop() {
        while (true) {
            swap_group * group = nullptr;
            {
                std::unique_lock<std::mutex> lock(worker_mutex);
                worker_cv.wait(lock, [&]() { return worker_stop || !worker_queue.empty(); });
                if (worker_queue.empty()) {
                    if (worker_stop) {
                        return;
                    }
                    continue;
                }
                group = worker_queue.front();
                worker_queue.pop_front();
            }

            for (const auto & event : group->events) {
                ggml_backend_event_synchronize(event.get());
            }

            group->completed_us.store(ggml_time_us(), std::memory_order_relaxed);
            group->complete.store(true, std::memory_order_release);
            completion_cv.notify_all();
        }
    }

    void enqueue(swap_group * group) {
        {
            std::lock_guard<std::mutex> lock(worker_mutex);
            worker_queue.push_back(group);
        }
        worker_cv.notify_one();
    }

    template <typename F> void for_each_run(const swap_page & page, F && fn) const {
        size_t i = 0;
        while (i < page.cells.size()) {
            const size_t begin = i;
            while (i + 1 < page.cells.size() && page.cells[i + 1].idx == page.cells[i].idx + 1) {
                ++i;
            }
            fn(begin, i - begin + 1, page.cells[begin].idx);
            ++i;
        }
    }

    void copy_page_d2h(swap_group & group, swap_page & page, std::vector<bool> & used_channels) {
        char * host = static_cast<char *>(ggml_backend_buffer_get_base(group.host_buffer.get())) + page.host_offset;
        size_t host_offset = 0;

        for (const auto & layer : kv->layers) {
            const uint32_t il        = layer.il;
            ggml_tensor *  k         = layer.k_stream[0];
            channel *      k_channel = get_channel(tensor_device(k));
            const size_t   k_row     = ggml_row_size(k->type, kv->hparams.n_embd_k_gqa(il));

            for_each_run(page, [&](size_t begin, size_t count, uint32_t idx) {
                ggml_backend_tensor_get_async(k_channel->copy_backend.get(), k, host + host_offset + begin * k_row,
                                              idx * k_row, count * k_row);
            });
            used_channels[k_channel - channels.data()] = true;
            host_offset += page.cells.size() * k_row;

            ggml_tensor * v = layer.v_stream[0];
            if (!v) {
                continue;
            }

            channel * v_channel = get_channel(tensor_device(v));
            if (!kv->v_trans) {
                const size_t v_row = ggml_row_size(v->type, kv->hparams.n_embd_v_gqa(il));
                for_each_run(page, [&](size_t begin, size_t count, uint32_t idx) {
                    ggml_backend_tensor_get_async(v_channel->copy_backend.get(), v, host + host_offset + begin * v_row,
                                                  idx * v_row, count * v_row);
                });
                host_offset += page.cells.size() * v_row;
            } else {
                const size_t v_size     = ggml_type_size(v->type);
                const size_t n_embd     = kv->hparams.n_embd_v_gqa(il);
                const size_t row_host   = page.cells.size() * v_size;
                const size_t row_device = kv->get_size() * v_size;

                for_each_run(page, [&](size_t begin, size_t count, uint32_t idx) {
                    ggml_backend_tensor_get_2d_async(v_channel->copy_backend.get(), v,
                                                     host + host_offset + begin * v_size, idx * v_size, count * v_size,
                                                     n_embd, row_device, row_host);
                });
                host_offset += n_embd * row_host;
            }
            used_channels[v_channel - channels.data()] = true;
        }

        GGML_ASSERT(host_offset == page.cells.size() * cell_bytes());
    }

    void copy_page_h2d(swap_group & group, swap_page & page, std::vector<bool> & used_channels) {
        const char * host =
            static_cast<const char *>(ggml_backend_buffer_get_base(group.host_buffer.get())) + page.host_offset;
        size_t host_offset = 0;

        for (const auto & layer : kv->layers) {
            const uint32_t il        = layer.il;
            ggml_tensor *  k         = layer.k_stream[0];
            channel *      k_channel = get_channel(tensor_device(k));
            const size_t   k_row     = ggml_row_size(k->type, kv->hparams.n_embd_k_gqa(il));

            for_each_run(page, [&](size_t begin, size_t count, uint32_t idx) {
                ggml_backend_tensor_set_async(k_channel->copy_backend.get(), k, host + host_offset + begin * k_row,
                                              idx * k_row, count * k_row);
            });
            used_channels[k_channel - channels.data()] = true;
            host_offset += page.cells.size() * k_row;

            ggml_tensor * v = layer.v_stream[0];
            if (!v) {
                continue;
            }

            channel * v_channel = get_channel(tensor_device(v));
            if (!kv->v_trans) {
                const size_t v_row = ggml_row_size(v->type, kv->hparams.n_embd_v_gqa(il));
                for_each_run(page, [&](size_t begin, size_t count, uint32_t idx) {
                    ggml_backend_tensor_set_async(v_channel->copy_backend.get(), v, host + host_offset + begin * v_row,
                                                  idx * v_row, count * v_row);
                });
                host_offset += page.cells.size() * v_row;
            } else {
                const size_t v_size     = ggml_type_size(v->type);
                const size_t n_embd     = kv->hparams.n_embd_v_gqa(il);
                const size_t row_host   = page.cells.size() * v_size;
                const size_t row_device = kv->get_size() * v_size;

                for_each_run(page, [&](size_t begin, size_t count, uint32_t idx) {
                    ggml_backend_tensor_set_2d_async(v_channel->copy_backend.get(), v,
                                                     host + host_offset + begin * v_size, idx * v_size, count * v_size,
                                                     n_embd, row_device, row_host);
                });
                host_offset += n_embd * row_host;
            }
            used_channels[v_channel - channels.data()] = true;
        }

        GGML_ASSERT(host_offset == page.cells.size() * cell_bytes());
    }

    void submit_group(swap_group & group, transfer_direction direction) {
        group.direction = direction;
        group.events.clear();
        group.applied = false;
        group.complete.store(false, std::memory_order_relaxed);
        group.completed_us.store(0, std::memory_order_relaxed);
        group.submitted_us = ggml_time_us();

        std::vector<bool> used_channels(channels.size(), false);
        for (auto & page : group.pages) {
            page.state = direction == TRANSFER_DIRECTION_D2H ? PAGE_STATE_D2H : PAGE_STATE_H2D;
            if (direction == TRANSFER_DIRECTION_D2H) {
                copy_page_d2h(group, page, used_channels);
            } else {
                copy_page_h2d(group, page, used_channels);
            }
        }

        for (size_t i = 0; i < channels.size(); ++i) {
            if (!used_channels[i]) {
                continue;
            }
            ggml_backend_event_ptr event(ggml_backend_event_new(channels[i].device));
            GGML_ASSERT(event);
            ggml_backend_event_record(event.get(), channels[i].copy_backend.get());
            group.events.push_back(std::move(event));
        }

        enqueue(&group);
    }

    bool allocate_groups(swap_sequence & sequence, std::vector<swap_cell> cells) {
        const size_t bytes_per_cell = cell_bytes();
        const size_t estimate       = cells.size() * bytes_per_cell;
        if (estimate > max_bytes - stats.host_bytes) {
            error = "the KV swap host arena is full";
            return false;
        }

        const size_t group_cells = KV_SWAP_PAGE_CELLS * KV_SWAP_PAGES_PER_GROUP;
        for (size_t group_begin = 0; group_begin < cells.size(); group_begin += group_cells) {
            auto         group     = std::make_unique<swap_group>();
            const size_t group_end = std::min(cells.size(), group_begin + group_cells);

            size_t host_offset = 0;
            for (size_t page_begin = group_begin; page_begin < group_end; page_begin += KV_SWAP_PAGE_CELLS) {
                swap_page    page;
                const size_t page_end = std::min(group_end, page_begin + KV_SWAP_PAGE_CELLS);
                page.cells.insert(page.cells.end(), cells.begin() + page_begin, cells.begin() + page_end);
                page.host_offset = host_offset;
                page.last_use_us = sequence.last_use_us;
                host_offset += page.cells.size() * bytes_per_cell;
                group->pages.push_back(std::move(page));
            }

            ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(channels[0].host_buft, host_offset));
            if (!buffer) {
                error = "failed to allocate pinned memory for the KV swap arena";
                return false;
            }

            group->data_bytes   = host_offset;
            group->buffer_bytes = ggml_backend_buffer_get_size(buffer.get());
            if (group->buffer_bytes > max_bytes - stats.host_bytes - sequence.buffer_bytes) {
                error = "the KV swap host arena is full after buffer alignment";
                return false;
            }
            group->host_buffer = std::move(buffer);

            sequence.data_bytes += group->data_bytes;
            sequence.buffer_bytes += group->buffer_bytes;
            sequence.groups.push_back(std::move(group));
        }

        return true;
    }

    llama_kv_swap_result seq_out(llama_seq_id seq_id, int64_t last_use_us) {
        poll();

        if (sequences.find(seq_id) != sequences.end()) {
            error = "the sequence already has a KV swap operation";
            return LLAMA_KV_SWAP_RESULT_BUSY;
        }
        if (seq_id < 0 || (size_t) seq_id >= kv->seq_to_stream.size()) {
            error = "the sequence id is out of range";
            return LLAMA_KV_SWAP_RESULT_ERROR;
        }

        auto &                 cells = kv->v_cells[kv->seq_to_stream[seq_id]];
        std::vector<swap_cell> saved_cells;
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells.is_empty(i) || !cells.seq_has(i, seq_id)) {
                continue;
            }
            if (cells.seq_count(i) != 1) {
                error = "KV swap pages require cells owned by a single sequence";
                return LLAMA_KV_SWAP_RESULT_BUSY;
            }
            if (cells.get_shift(i) != 0) {
                error = "the sequence has a pending KV shift";
                return LLAMA_KV_SWAP_RESULT_BUSY;
            }
            saved_cells.push_back({ i, cells.pos_get(i), cells.ext_get(i) });
        }

        if (saved_cells.empty()) {
            error.clear();
            return LLAMA_KV_SWAP_RESULT_NO_SEQUENCE;
        }

        std::sort(saved_cells.begin(), saved_cells.end(), [](const swap_cell & a, const swap_cell & b) {
            if (a.pos != b.pos) {
                return a.pos < b.pos;
            }
            return a.idx < b.idx;
        });

        auto sequence         = std::make_unique<swap_sequence>();
        sequence->seq_id      = seq_id;
        sequence->state       = LLAMA_KV_SWAP_STATE_SWAPPING_OUT;
        sequence->cell_count  = saved_cells.size();
        sequence->last_use_us = last_use_us;

        if (!allocate_groups(*sequence, std::move(saved_cells))) {
            return LLAMA_KV_SWAP_RESULT_NO_HOST_CAPACITY;
        }

        stats.host_bytes += sequence->buffer_bytes;
        stats.host_bytes_peak       = std::max(stats.host_bytes_peak, stats.host_bytes);
        sequence->transfer_start_us = ggml_time_us();

        for (auto & cur : channels) {
            ggml_backend_event_ptr event(ggml_backend_event_new(cur.device));
            GGML_ASSERT(event);
            ggml_backend_event_record(event.get(), cur.compute_backend);
            ggml_backend_event_wait(cur.copy_backend.get(), event.get());
            sequence->compute_events.push_back(std::move(event));
        }

        swap_sequence * raw = sequence.get();
        sequences.emplace(seq_id, std::move(sequence));
        for (auto & group : raw->groups) {
            submit_group(*group, TRANSFER_DIRECTION_D2H);
        }

        error.clear();
        return LLAMA_KV_SWAP_RESULT_OK;
    }

    bool reserve_device_cells(swap_sequence & sequence) {
        auto & cells = kv->v_cells[kv->seq_to_stream[sequence.seq_id]];
        if (sequence.cell_count > cells.size() - cells.get_used()) {
            return false;
        }

        std::vector<uint32_t> free_cells;
        free_cells.reserve(sequence.cell_count);
        for (uint32_t i = 0; i < cells.size() && free_cells.size() < sequence.cell_count; ++i) {
            if (cells.is_empty(i)) {
                free_cells.push_back(i);
            }
        }
        GGML_ASSERT(free_cells.size() == sequence.cell_count);

        size_t i_cell = 0;
        for (auto & group : sequence.groups) {
            for (auto & page : group->pages) {
                for (auto & saved : page.cells) {
                    saved.idx = free_cells[i_cell++];
                    cells.pos_set(saved.idx, saved.pos);
                    cells.ext_set(saved.idx, saved.ext);
                    cells.seq_add(saved.idx, sequence.seq_id);
                }
            }
        }

        kv->v_heads[kv->seq_to_stream[sequence.seq_id]] = free_cells.back() + 1;
        return true;
    }

    llama_kv_swap_result seq_in(llama_seq_id seq_id) {
        poll();

        auto it = sequences.find(seq_id);
        if (it == sequences.end()) {
            error = "the sequence has no host KV snapshot";
            return LLAMA_KV_SWAP_RESULT_NO_SEQUENCE;
        }

        swap_sequence & sequence = *it->second;
        if (sequence.state != LLAMA_KV_SWAP_STATE_HOST) {
            error = "the sequence KV snapshot is still transferring";
            return LLAMA_KV_SWAP_RESULT_BUSY;
        }

        if (!reserve_device_cells(sequence)) {
            error = "there are not enough free device KV cells to restore the sequence";
            return LLAMA_KV_SWAP_RESULT_NO_DEVICE_CAPACITY;
        }

        sequence.state             = LLAMA_KV_SWAP_STATE_SWAPPING_IN;
        sequence.transfer_start_us = ggml_time_us();
        for (auto & group : sequence.groups) {
            submit_group(*group, TRANSFER_DIRECTION_H2D);
        }

        error.clear();
        return LLAMA_KV_SWAP_RESULT_OK;
    }

    void release_group_cells(swap_sequence & sequence, swap_group & group) {
        auto &   cells    = kv->v_cells[kv->seq_to_stream[sequence.seq_id]];
        auto &   head     = kv->v_heads[kv->seq_to_stream[sequence.seq_id]];
        uint32_t head_new = cells.size();

        for (const auto & page : group.pages) {
            for (const auto & saved : page.cells) {
                if (cells.is_empty(saved.idx) || !cells.seq_has(saved.idx, sequence.seq_id)) {
                    continue;
                }
                if (cells.seq_rm(saved.idx, sequence.seq_id)) {
                    head_new = std::min(head_new, saved.idx);
                }
            }
        }

        if (head_new < head) {
            head = head_new;
        }
    }

    bool all_groups_applied(const swap_sequence & sequence) const {
        for (const auto & group : sequence.groups) {
            if (!group->applied) {
                return false;
            }
        }
        return true;
    }

    void erase_sequence(std::map<llama_seq_id, std::unique_ptr<swap_sequence>>::iterator it, bool remove_device) {
        swap_sequence & sequence = *it->second;
        if (remove_device) {
            kv->seq_rm(sequence.seq_id, -1, -1);
        }
        GGML_ASSERT(stats.host_bytes >= sequence.buffer_bytes);
        stats.host_bytes -= sequence.buffer_bytes;
        sequences.erase(it);
    }

    void poll() {
        std::vector<llama_seq_id> erase;

        for (auto & item : sequences) {
            swap_sequence & sequence = *item.second;

            for (auto & group_ptr : sequence.groups) {
                swap_group & group = *group_ptr;
                if (group.applied || !group.complete.load(std::memory_order_acquire)) {
                    continue;
                }

                if (group.direction == TRANSFER_DIRECTION_D2H) {
                    release_group_cells(sequence, group);
                    for (auto & page : group.pages) {
                        page.state = PAGE_STATE_HOST;
                    }
                    stats.pages_out += group.pages.size();
                } else {
                    for (auto & page : group.pages) {
                        page.state = PAGE_STATE_RESIDENT;
                    }
                    stats.pages_in += group.pages.size();
                }
                group.applied = true;
            }

            if (!all_groups_applied(sequence)) {
                continue;
            }

            int64_t completed_us = sequence.transfer_start_us;
            for (const auto & group : sequence.groups) {
                completed_us = std::max(completed_us, group->completed_us.load(std::memory_order_relaxed));
            }
            const uint64_t transfer_us =
                completed_us > sequence.transfer_start_us ? completed_us - sequence.transfer_start_us : 0;

            if (sequence.state == LLAMA_KV_SWAP_STATE_SWAPPING_OUT) {
                sequence.state = LLAMA_KV_SWAP_STATE_HOST;
                sequence.compute_events.clear();
                stats.d2h_bytes += sequence.data_bytes;
                stats.d2h_time_us += transfer_us;
                stats.sequences_out++;

                if (sequence.discard) {
                    erase.push_back(sequence.seq_id);
                }
            } else if (sequence.state == LLAMA_KV_SWAP_STATE_SWAPPING_IN) {
                stats.h2d_bytes += sequence.data_bytes;
                stats.h2d_time_us += transfer_us;
                stats.sequences_in++;

                if (sequence.discard) {
                    kv->seq_rm(sequence.seq_id, -1, -1);
                }
                erase.push_back(sequence.seq_id);
            } else if (sequence.state == LLAMA_KV_SWAP_STATE_HOST && sequence.discard) {
                erase.push_back(sequence.seq_id);
            }
        }

        for (llama_seq_id seq_id : erase) {
            auto it = sequences.find(seq_id);
            if (it != sequences.end()) {
                erase_sequence(it, false);
            }
        }
    }

    void seq_discard(llama_seq_id seq_id) {
        auto it = sequences.find(seq_id);
        if (it == sequences.end()) {
            return;
        }

        it->second->discard = true;
        {
            std::unique_lock<std::mutex> lock(worker_mutex);
            completion_cv.wait(lock, [&]() {
                auto cur = sequences.find(seq_id);
                if (cur == sequences.end()) {
                    return true;
                }
                for (const auto & group : cur->second->groups) {
                    if (!group->complete.load(std::memory_order_acquire)) {
                        return false;
                    }
                }
                return true;
            });
        }
        poll();
    }

    uint64_t seq_size(llama_seq_id seq_id) const {
        auto it = sequences.find(seq_id);
        if (it != sequences.end()) {
            return it->second->data_bytes;
        }
        return (uint64_t) sequence_cells(seq_id) * cell_bytes();
    }

    llama_kv_swap_seq_status seq_status(llama_seq_id seq_id) const {
        llama_kv_swap_seq_status result = {};
        result.state                    = LLAMA_KV_SWAP_STATE_RESIDENT;
        result.device_cells             = sequence_cells(seq_id);
        result.total_cells              = result.device_cells;

        auto it = sequences.find(seq_id);
        if (it == sequences.end()) {
            return result;
        }

        const swap_sequence & sequence = *it->second;
        result.state                   = sequence.state;
        result.total_cells             = sequence.cell_count;
        result.host_bytes              = sequence.buffer_bytes;
        result.last_use_us             = sequence.last_use_us;
        result.device_cells            = 0;

        for (const auto & group : sequence.groups) {
            for (const auto & page : group->pages) {
                if (page.state == PAGE_STATE_RESIDENT || page.state == PAGE_STATE_D2H) {
                    result.device_cells += page.cells.size();
                }
                if (page.state == PAGE_STATE_HOST) {
                    result.host_pages++;
                } else if (page.state == PAGE_STATE_D2H || page.state == PAGE_STATE_H2D) {
                    result.transfer_pages++;
                }
            }
        }

        return result;
    }

    llama_kv_swap_stats get_stats() const {
        llama_kv_swap_stats result = stats;
        result.device_cells_used   = device_cells_used();
        result.transfers_pending   = 0;
        for (const auto & item : sequences) {
            for (const auto & group : item.second->groups) {
                if (!group->complete.load(std::memory_order_acquire)) {
                    ++result.transfers_pending;
                }
            }
        }
        return result;
    }
};

bool llama_kv_swap_supported(llama_context * ctx, char * reason, size_t reason_size) {
    std::string message;
    const bool  result = llama_kv_swap::supported(ctx, message);
    copy_reason(reason, reason_size, message);
    return result;
}

llama_kv_swap * llama_kv_swap_init(llama_context * ctx, size_t max_bytes) {
    std::string      reason;
    llama_kv_cache * kv = llama_kv_swap::get_cache(ctx, reason);
    if (!kv || max_bytes == 0 || !llama_kv_swap::supported(ctx, reason)) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, reason.empty() ? "invalid host budget" : reason.c_str());
        return nullptr;
    }

    try {
        return new llama_kv_swap(ctx, kv, max_bytes);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return nullptr;
    }
}

void llama_kv_swap_free(llama_kv_swap * swap) {
    delete swap;
}

const char * llama_kv_swap_get_error(const llama_kv_swap * swap) {
    return swap ? swap->error.c_str() : "KV swap is not initialized";
}

llama_kv_swap_result llama_kv_swap_seq_out(llama_kv_swap * swap, llama_seq_id seq_id, int64_t last_use_us) {
    if (!swap) {
        return LLAMA_KV_SWAP_RESULT_UNSUPPORTED;
    }
    try {
        return swap->seq_out(seq_id, last_use_us);
    } catch (const std::exception & e) {
        swap->error = e.what();
        return LLAMA_KV_SWAP_RESULT_ERROR;
    }
}

llama_kv_swap_result llama_kv_swap_seq_in(llama_kv_swap * swap, llama_seq_id seq_id) {
    if (!swap) {
        return LLAMA_KV_SWAP_RESULT_UNSUPPORTED;
    }
    try {
        return swap->seq_in(seq_id);
    } catch (const std::exception & e) {
        swap->error = e.what();
        return LLAMA_KV_SWAP_RESULT_ERROR;
    }
}

void llama_kv_swap_poll(llama_kv_swap * swap) {
    if (swap) {
        swap->poll();
    }
}

void llama_kv_swap_seq_discard(llama_kv_swap * swap, llama_seq_id seq_id) {
    if (swap) {
        swap->seq_discard(seq_id);
    }
}

uint64_t llama_kv_swap_seq_size(const llama_kv_swap * swap, llama_seq_id seq_id) {
    return swap ? swap->seq_size(seq_id) : 0;
}

llama_kv_swap_seq_status llama_kv_swap_seq_get_status(const llama_kv_swap * swap, llama_seq_id seq_id) {
    if (!swap) {
        llama_kv_swap_seq_status result = {};
        result.state                    = LLAMA_KV_SWAP_STATE_RESIDENT;
        return result;
    }
    return swap->seq_status(seq_id);
}

llama_kv_swap_stats llama_kv_swap_get_stats(const llama_kv_swap * swap) {
    if (!swap) {
        return {};
    }
    return swap->get_stats();
}

void llama_kv_swap_add_overlap(llama_kv_swap * swap, uint64_t time_us) {
    if (swap) {
        swap->stats.overlap_time_us += time_us;
    }
}
