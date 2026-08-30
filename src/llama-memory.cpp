#include "llama-memory.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-dsa.h"
#include "llama-kv-cache-dsa-iswa.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-msa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <bitset>

llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? LLAMA_MEMORY_STATUS_SUCCESS : LLAMA_MEMORY_STATUS_NO_UPDATE;
}

bool llama_memory_status_is_fail(llama_memory_status status) {
    switch (status) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}

static uint64_t memory_breakdown_total(const std::map<ggml_backend_buffer_type_t, size_t> & breakdown) {
    uint64_t total = 0;
    for (const auto & item : breakdown) {
        total += item.second;
    }
    return total;
}

llama_memory_primary_occupancy llama_memory_primary_occupancy_collect(const llama_memory_i * memory) {
    if (!memory) {
        return {};
    }
    if (const auto * kv = dynamic_cast<const llama_kv_cache *>(memory)) {
        llama_memory_primary_occupancy result;
        result.available = true;
        result.capacity_entries = (uint64_t) kv->get_size() * kv->get_n_stream();
        for (uint32_t stream = 0; stream < kv->get_n_stream(); ++stream) {
            result.used_entries += kv->get_cells_by_stream(stream).get_used();
        }
        return result;
    }
    if (const auto * recurrent = dynamic_cast<const llama_memory_recurrent *>(memory)) {
        return { true, recurrent->size, recurrent->used };
    }
    if (const auto * iswa = dynamic_cast<const llama_kv_cache_iswa *>(memory)) {
        return llama_memory_primary_occupancy_collect(iswa->get_base());
    }
    if (const auto * hybrid = dynamic_cast<const llama_memory_hybrid *>(memory)) {
        return llama_memory_primary_occupancy_collect(hybrid->get_mem_attn());
    }
    if (const auto * hybrid_iswa = dynamic_cast<const llama_memory_hybrid_iswa *>(memory)) {
        return llama_memory_primary_occupancy_collect(hybrid_iswa->get_mem_attn());
    }
    if (const auto * dsa = dynamic_cast<const llama_kv_cache_dsa *>(memory)) {
        return llama_memory_primary_occupancy_collect(dsa->get_mla());
    }
    if (const auto * dsa_iswa = dynamic_cast<const llama_kv_cache_dsa_iswa *>(memory)) {
        return llama_memory_primary_occupancy_collect(dsa_iswa->get_dsa());
    }
    if (const auto * msa = dynamic_cast<const llama_kv_cache_msa *>(memory)) {
        return llama_memory_primary_occupancy_collect(msa->get_base());
    }
    if (const auto * dsv4 = dynamic_cast<const llama_kv_cache_dsv4 *>(memory)) {
        return llama_memory_primary_occupancy_collect(dsv4->get_raw());
    }
    return {};
}

static llama_memory_component_diagnostics kv_component_diagnostics(
        const llama_kv_cache & kv,
        std::string name,
        bool logical_primary) {
    llama_memory_component_diagnostics result;
    result.name = std::move(name);
    result.kind = "kv_cache";
    result.entry_semantics = "token_kv_cell";
    result.logical_primary = logical_primary;
    result.resident_tokens_supported = true;
    result.physical_sharing_supported = true;
    result.capacity_entries = (uint64_t) kv.get_size() * kv.get_n_stream();
    result.allocated_bytes = memory_breakdown_total(kv.memory_breakdown());
    result.churn = kv.get_churn();

    std::bitset<LLAMA_MAX_SEQ> represented;
    std::bitset<LLAMA_MAX_SEQ> sharing;
    for (uint32_t stream = 0; stream < kv.get_n_stream(); ++stream) {
        const auto & cells = kv.get_cells_by_stream(stream);
        bool previous_shared = false;
        uint32_t previous_index = 0;
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells.is_empty(i)) {
                previous_shared = false;
                continue;
            }

            result.used_entries++;
            const uint64_t fanout = cells.seq_count(i);
            result.max_fanout = std::max(result.max_fanout, fanout);
            for (llama_seq_id seq_id = 0; seq_id < LLAMA_MAX_SEQ; ++seq_id) {
                if (cells.seq_has(i, seq_id)) {
                    represented.set(seq_id);
                    if (fanout > 1) {
                        sharing.set(seq_id);
                    }
                }
            }

            if (fanout > 1) {
                result.shared_entries++;
                result.shared_memberships += fanout;
                const bool continues_group = previous_shared &&
                    cells.pos_get(i) == cells.pos_get(previous_index) + 1 &&
                    cells.seq_is_same(i, previous_index);
                if (!continues_group) {
                    result.shared_groups++;
                }
                previous_shared = true;
                previous_index = i;
            } else {
                previous_shared = false;
            }
        }
    }

    result.resident_tokens = result.used_entries;
    result.sequences_represented = represented.count();
    result.sequences_sharing = sharing.count();
    if (result.capacity_entries > 0) {
        result.occupied_bytes_estimate = (uint64_t) (
            (long double) result.allocated_bytes * result.used_entries / result.capacity_entries);
    }
    return result;
}

static llama_memory_component_diagnostics recurrent_component_diagnostics(
        const llama_memory_recurrent & memory,
        std::string name,
        bool logical_primary) {
    llama_memory_component_diagnostics result;
    result.name = std::move(name);
    result.kind = "recurrent_state";
    result.entry_semantics = "recurrent_state_cell";
    result.logical_primary = logical_primary;
    result.resident_tokens_supported = false;
    result.physical_sharing_supported = true;
    result.capacity_entries = memory.size;
    result.used_entries = memory.used;
    result.allocated_bytes = memory_breakdown_total(memory.memory_breakdown());
    result.churn = memory.get_churn();

    std::bitset<LLAMA_MAX_SEQ> represented;
    std::bitset<LLAMA_MAX_SEQ> sharing;
    for (const auto & cell : memory.cells) {
        if (cell.is_empty()) {
            continue;
        }
        const uint64_t fanout = cell.seq_id.size();
        result.max_fanout = std::max(result.max_fanout, fanout);
        for (const llama_seq_id seq_id : cell.seq_id) {
            if (seq_id >= 0 && seq_id < LLAMA_MAX_SEQ) {
                represented.set(seq_id);
                if (fanout > 1) {
                    sharing.set(seq_id);
                }
            }
        }
        if (fanout > 1) {
            result.shared_entries++;
            result.shared_memberships += fanout;
        }
    }
    result.sequences_represented = represented.count();
    result.sequences_sharing = sharing.count();
    if (result.capacity_entries > 0) {
        result.occupied_bytes_estimate = (uint64_t) (
            (long double) result.allocated_bytes * result.used_entries / result.capacity_entries);
    }
    return result;
}

static void collect_memory_components(
        const llama_memory_i * memory,
        const std::string & prefix,
        bool logical_primary,
        std::vector<llama_memory_component_diagnostics> & output) {
    if (!memory) {
        return;
    }
    if (const auto * kv = dynamic_cast<const llama_kv_cache *>(memory)) {
        output.push_back(kv_component_diagnostics(*kv, prefix.empty() ? "kv" : prefix, logical_primary));
    } else if (const auto * recurrent = dynamic_cast<const llama_memory_recurrent *>(memory)) {
        output.push_back(recurrent_component_diagnostics(*recurrent, prefix.empty() ? "recurrent" : prefix, logical_primary));
    } else if (const auto * iswa = dynamic_cast<const llama_kv_cache_iswa *>(memory)) {
        collect_memory_components(iswa->get_base(), prefix + "base", logical_primary, output);
        collect_memory_components(iswa->get_swa(),  prefix + "swa",  false, output);
    } else if (const auto * hybrid_idx = dynamic_cast<const llama_memory_hybrid_idx *>(memory)) {
        collect_memory_components(hybrid_idx->get_mem_attn(), prefix + "attention", logical_primary, output);
        collect_memory_components(hybrid_idx->get_mem_recr(), prefix + "recurrent", false, output);
        collect_memory_components(hybrid_idx->get_mem_idx(),  prefix + "indexer",   false, output);
    } else if (const auto * hybrid = dynamic_cast<const llama_memory_hybrid *>(memory)) {
        collect_memory_components(hybrid->get_mem_attn(), prefix + "attention", logical_primary, output);
        collect_memory_components(hybrid->get_mem_recr(), prefix + "recurrent", false, output);
    } else if (const auto * hybrid_iswa = dynamic_cast<const llama_memory_hybrid_iswa *>(memory)) {
        collect_memory_components(hybrid_iswa->get_mem_attn(), prefix + "attention_", logical_primary, output);
        collect_memory_components(hybrid_iswa->get_mem_recr(), prefix + "recurrent", false, output);
    } else if (const auto * dsa = dynamic_cast<const llama_kv_cache_dsa *>(memory)) {
        collect_memory_components(dsa->get_mla(), prefix + "mla", logical_primary, output);
        collect_memory_components(dsa->get_lid(), prefix + "lightning_index", false, output);
    } else if (const auto * dsa_iswa = dynamic_cast<const llama_kv_cache_dsa_iswa *>(memory)) {
        collect_memory_components(dsa_iswa->get_dsa(), prefix + "dsa_", logical_primary, output);
        collect_memory_components(dsa_iswa->get_swa(), prefix + "swa", false, output);
    } else if (const auto * msa = dynamic_cast<const llama_kv_cache_msa *>(memory)) {
        collect_memory_components(msa->get_base(), prefix + "base", logical_primary, output);
        collect_memory_components(msa->get_idx(), prefix + "index", false, output);
    } else if (const auto * dsv4 = dynamic_cast<const llama_kv_cache_dsv4 *>(memory)) {
        collect_memory_components(dsv4->get_raw(), prefix + "raw_", logical_primary, output);
        collect_memory_components(dsv4->get_csa(), prefix + "compressed_csa", false, output);
        collect_memory_components(dsv4->get_hca(), prefix + "compressed_hca", false, output);
        collect_memory_components(dsv4->get_lid(), prefix + "lightning_index", false, output);
    }
}

llama_memory_diagnostics llama_memory_diagnostics_collect(const llama_memory_i * memory) {
    llama_memory_diagnostics result;
    collect_memory_components(memory, "", true, result.components);
    if (!result.components.empty()) {
        result.state = "available";
        for (const auto & component : result.components) {
            result.churn += component.churn;
        }
    }
    return result;
}

uint64_t llama_memory_shared_prefix_length(
        const llama_memory_i * memory,
        const llama_seq_id * sequence_ids,
        size_t sequence_count,
        uint64_t maximum_prefix_tokens) {
    if (!memory || !sequence_ids || sequence_count < 2 || maximum_prefix_tokens == 0) {
        return 0;
    }

    if (const auto * kv = dynamic_cast<const llama_kv_cache *>(memory)) {
        if (kv->get_n_stream() != 1) {
            return 0;
        }
        const auto & cells = kv->get_cells_by_stream(0);
        std::vector<bool> shared(maximum_prefix_tokens, false);
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells.is_empty(i)) {
                continue;
            }
            const llama_pos position = cells.pos_get(i);
            if (position < 0 || (uint64_t) position >= maximum_prefix_tokens) {
                continue;
            }
            bool has_all = true;
            for (size_t s = 0; s < sequence_count; ++s) {
                if (sequence_ids[s] < 0 || sequence_ids[s] >= LLAMA_MAX_SEQ || !cells.seq_has(i, sequence_ids[s])) {
                    has_all = false;
                    break;
                }
            }
            shared[position] = has_all;
        }
        uint64_t length = 0;
        while (length < maximum_prefix_tokens && shared[length]) {
            ++length;
        }
        return length;
    }
    if (const auto * iswa = dynamic_cast<const llama_kv_cache_iswa *>(memory)) {
        return llama_memory_shared_prefix_length(iswa->get_base(), sequence_ids, sequence_count, maximum_prefix_tokens);
    }
    if (const auto * hybrid = dynamic_cast<const llama_memory_hybrid *>(memory)) {
        return llama_memory_shared_prefix_length(hybrid->get_mem_attn(), sequence_ids, sequence_count, maximum_prefix_tokens);
    }
    if (const auto * hybrid_iswa = dynamic_cast<const llama_memory_hybrid_iswa *>(memory)) {
        return llama_memory_shared_prefix_length(hybrid_iswa->get_mem_attn(), sequence_ids, sequence_count, maximum_prefix_tokens);
    }
    if (const auto * dsa = dynamic_cast<const llama_kv_cache_dsa *>(memory)) {
        return llama_memory_shared_prefix_length(dsa->get_mla(), sequence_ids, sequence_count, maximum_prefix_tokens);
    }
    if (const auto * dsa_iswa = dynamic_cast<const llama_kv_cache_dsa_iswa *>(memory)) {
        return llama_memory_shared_prefix_length(dsa_iswa->get_dsa(), sequence_ids, sequence_count, maximum_prefix_tokens);
    }
    if (const auto * msa = dynamic_cast<const llama_kv_cache_msa *>(memory)) {
        return llama_memory_shared_prefix_length(msa->get_base(), sequence_ids, sequence_count, maximum_prefix_tokens);
    }
    if (const auto * dsv4 = dynamic_cast<const llama_kv_cache_dsv4 *>(memory)) {
        return llama_memory_shared_prefix_length(dsv4->get_raw(), sequence_ids, sequence_count, maximum_prefix_tokens);
    }
    return 0;
}
