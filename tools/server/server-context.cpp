#include "server-context.h"
#include "server-gpu-telemetry.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-moe-routing.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-stream.h"

#include "build-info.h"
#include "base64.hpp"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "src/llama-ext.h"
#include "src/llama-graph.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cinttypes>
#include <cctype>
#include <exception>
#include <memory>
#include <filesystem>
#include <chrono>
#include <ctime>
#include <random>
#include <tuple>
#include <utility>
#include <fstream>
#include <deque>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <sstream>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

constexpr int HTTP_POLLING_SECONDS = 1;

static common_speculative_output_limits server_output_limits(const common_params & params) {
    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return { params.n_batch, 1 };
    }

    auto result = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));

    result.total   = std::max<int32_t>(1, result.total);
    result.per_seq = std::max<int32_t>(1, result.per_seq);
    return result;
}

// synthetic draft verification for benchmarking - accept draft tokens at random instead of by match with the target
// on replay the draft was already accepted before a context checkpoint restore, so repeat the same decisions
static std::vector<llama_token> server_sample_and_accept_synth(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft,
        const std::vector<double> & synth_probs,
        std::mt19937 & rng,
        bool is_replay) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);
    GGML_ASSERT(synth_probs.size() >= draft.size());

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < draft.size(); ++i) {
        const llama_token id = common_sampler_sample(smpl, ctx, idxs[i]);
        const bool accept = is_replay || dist(rng) < synth_probs[i];
        // do not accept a drafted EOG token - it would end the generation early
        // on replay the last token is from the target and can be EOG, so skip this check
        if (accept && (is_replay || !llama_vocab_is_eog(vocab, draft[i]))) {
            // synthetic draft tokens do not advance grammar or reasoning state
            // the last replay token is from the target and must advance both
            const bool is_replay_target = is_replay && i + 1 == draft.size();
            common_sampler_accept(smpl, draft[i], is_replay_target);
            result.push_back(draft[i]);
            continue;
        }

        common_sampler_accept(smpl, id, true);
        result.push_back(id);
        return result;
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct server_slot; // forward declaration

struct telemetry_probability_accumulator {
    uint64_t count = 0;
    double nll_sum = 0.0;
    double min_logprob = std::numeric_limits<double>::infinity();
    double max_logprob = -std::numeric_limits<double>::infinity();
    std::string unavailable_reason;

    void reset() {
        count = 0;
        nll_sum = 0.0;
        min_logprob = std::numeric_limits<double>::infinity();
        max_logprob = -std::numeric_limits<double>::infinity();
        unavailable_reason.clear();
    }

    void observe(double logprob) {
        if (!std::isfinite(logprob)) {
            unavailable_reason = "selected_token_log_probability_not_finite";
            return;
        }
        count++;
        nll_sum -= logprob;
        min_logprob = std::min(min_logprob, logprob);
        max_logprob = std::max(max_logprob, logprob);
    }
};

enum telemetry_output_token_origin {
    TELEMETRY_OUTPUT_TOKEN_ORIGIN_NORMAL_DECODE,
    TELEMETRY_OUTPUT_TOKEN_ORIGIN_MTP_ACCEPTED,
    TELEMETRY_OUTPUT_TOKEN_ORIGIN_TARGET_AFTER_MISS,
    TELEMETRY_OUTPUT_TOKEN_ORIGIN_TARGET_BONUS,
};

struct telemetry_output_token_record {
    uint64_t ordinal = 0;
    int64_t model_ready_offset_us = 0;
    int64_t model_ready_monotonic_us = 0;
    llama_pos model_position = -1;
    llama_token token_id = LLAMA_TOKEN_NULL;
    std::string token_piece;
    double selected_log_probability_ln = std::numeric_limits<double>::quiet_NaN();
    telemetry_output_token_origin origin = TELEMETRY_OUTPUT_TOKEN_ORIGIN_NORMAL_DECODE;
    int64_t logical_step = -1;
    int64_t actual_target_pass = -1;
    int64_t proposal_position = -1;
    int64_t accepted_depth = -1;
    int64_t proposed_count = -1;
    bool replay_pass = false;
};

enum telemetry_mtp_pass_outcome {
    TELEMETRY_MTP_PASS_ZERO_ACCEPTANCE,
    TELEMETRY_MTP_PASS_PARTIAL_ACCEPTANCE,
    TELEMETRY_MTP_PASS_FULL_ACCEPTANCE,
};

enum telemetry_mtp_proposal_disposition {
    TELEMETRY_MTP_PROPOSAL_ACCEPTED,
    TELEMETRY_MTP_PROPOSAL_FIRST_MISMATCH,
    TELEMETRY_MTP_PROPOSAL_INVALIDATED_TAIL,
};

struct telemetry_mtp_proposal_record {
    uint64_t position = 0;
    uint64_t evaluated_actual_target_pass = 0;
    llama_token draft_token_id = LLAMA_TOKEN_NULL;
    llama_token target_selected_token_id = LLAMA_TOKEN_NULL;
    double target_selected_log_probability_ln = std::numeric_limits<double>::quiet_NaN();
    int64_t committed_output_ordinal = -1;
    telemetry_mtp_proposal_disposition disposition = TELEMETRY_MTP_PROPOSAL_INVALIDATED_TAIL;
};

struct telemetry_mtp_pass_record {
    uint64_t logical_step = 0;
    uint64_t actual_target_pass = 0;
    int64_t replay_of_actual_target_pass = -1;
    uint64_t proposed_count = 0;
    uint64_t accepted_depth = 0;
    uint64_t target_rows_evaluated = 0;
    uint64_t committed_output_start_ordinal = 0;
    uint64_t committed_token_count = 0;
    uint64_t reached_rejected_token_count = 0;
    uint64_t invalidated_token_count = 0;
    telemetry_mtp_pass_outcome outcome = TELEMETRY_MTP_PASS_ZERO_ACCEPTANCE;
    bool replay_pass = false;
    bool discarded = false;
    bool counts_as_logical_step = false;
    std::vector<telemetry_mtp_proposal_record> proposals;
};

enum telemetry_token_candidate_decision_kind {
    TELEMETRY_TOKEN_CANDIDATE_ACCEPTED,
    TELEMETRY_TOKEN_CANDIDATE_FIRST_MISMATCH,
    TELEMETRY_TOKEN_CANDIDATE_TARGET_BONUS,
};

struct telemetry_token_candidate_value {
    llama_token token_id = LLAMA_TOKEN_NULL;
    double log_probability_ln = std::numeric_limits<double>::quiet_NaN();
    bool target_selected = false;
    bool draft_proposed = false;
};

struct telemetry_token_candidate_decision {
    uint64_t logical_step = 0;
    uint64_t actual_target_pass = 0;
    uint64_t proposal_position = 0;
    int64_t related_output_ordinal = -1;
    llama_token draft_token_id = LLAMA_TOKEN_NULL;
    llama_token target_selected_token_id = LLAMA_TOKEN_NULL;
    telemetry_token_candidate_decision_kind kind = TELEMETRY_TOKEN_CANDIDATE_FIRST_MISMATCH;
    std::string probability_state = "available";
    std::string probability_reason = "raw_target_model_pre_sampler_top_k";
    std::vector<telemetry_token_candidate_value> target_candidates;
};

struct telemetry_token_candidate_block_entry {
    std::string trace_id;
    json data;
    size_t bytes = 0;
};

static std::string telemetry_response_with_serialized_events(json response, const std::string & events) {
    const std::string marker = "\"events\":[]";
    std::string serialized = safe_json_to_str(response);
    const size_t offset = serialized.find(marker);
    GGML_ASSERT(offset != std::string::npos);
    serialized.replace(offset + marker.size() - 2, 2, events);
    return serialized;
}

struct telemetry_control_state {
    bool moe_routing = false;
    bool output_token_detail = false;
    bool token_candidates = false;
    bool prompt_perplexity = false;
    bool request_content = false;
    bool kv_pressure_detail = false;
    bool native_gpu_gpm = false;
    uint64_t generation = 0;
};

struct telemetry_control_application {
    telemetry_control_state effective;
    const char * effective_from = "next_request";
};

enum telemetry_moe_token_phase {
    TELEMETRY_MOE_TOKEN_PHASE_PREFILL_OUTPUT,
    TELEMETRY_MOE_TOKEN_PHASE_NORMAL_DECODE,
    TELEMETRY_MOE_TOKEN_PHASE_MTP_VERIFY,
};

struct telemetry_moe_token_activation_record {
    llama_pos model_position = -1;
    int32_t layer_index = -1;
    int32_t expert_index = -1;
    float effective_weight = std::numeric_limits<float>::quiet_NaN();
    telemetry_moe_token_phase phase = TELEMETRY_MOE_TOKEN_PHASE_NORMAL_DECODE;
    int64_t logical_step = -1;
    int64_t actual_target_pass = -1;
    int64_t proposal_position = -1;
    bool replay_pass = false;
};

// The native readback is invalidated by the next decode or routing toggle. Keep
// an owning copy while the decode still owns the native context.
struct telemetry_moe_routing_expert_capture {
    int32_t expert_index = -1;
    float effective_weight = std::numeric_limits<float>::quiet_NaN();
    llama_moe_routing_value_status expert_index_status = LLAMA_MOE_ROUTING_VALUE_STATUS_SOURCE_UNAVAILABLE;
    llama_moe_routing_value_status effective_weight_status = LLAMA_MOE_ROUTING_VALUE_STATUS_SOURCE_UNAVAILABLE;
};

struct telemetry_moe_routing_row_capture {
    int32_t layer_index = -1;
    uint32_t graph_type = 0;
    uint32_t physical_ubatch_index = 0;
    int32_t row_index = -1;
    int32_t ubatch_token_index = -1;
    int32_t token_index = -1;
    llama_token token = LLAMA_TOKEN_NULL;
    llama_pos position = -1;
    std::vector<telemetry_moe_routing_expert_capture> selected_experts;
    llama_moe_routing_value_status row_identity_status = LLAMA_MOE_ROUTING_VALUE_STATUS_SOURCE_UNAVAILABLE;
    llama_moe_routing_value_status selected_experts_status = LLAMA_MOE_ROUTING_VALUE_STATUS_SOURCE_UNAVAILABLE;
    float selected_score = std::numeric_limits<float>::quiet_NaN();
    float rejected_score = std::numeric_limits<float>::quiet_NaN();
    llama_moe_routing_value_status selected_score_status = LLAMA_MOE_ROUTING_VALUE_STATUS_SOURCE_UNAVAILABLE;
    llama_moe_routing_value_status rejected_score_status = LLAMA_MOE_ROUTING_VALUE_STATUS_SOURCE_UNAVAILABLE;
};

struct telemetry_moe_shared_expert_capture {
    int32_t layer_index = -1;
    uint32_t graph_type = 0;
    bool present = false;
    uint32_t configured_count = 0;
    uint32_t ffn_size = 0;
};

struct telemetry_moe_routing_readback_capture {
    uint32_t version = 0;
    uint64_t capture_generation = 0;
    std::vector<telemetry_moe_routing_row_capture> rows;
    std::vector<telemetry_moe_shared_expert_capture> shared_experts;
};

struct server_batch {
    llama_batch batch;
    bool batch_rendered = false;

    struct token {
        int32_t id_slot;
        llama_token token;
        llama_pos pos;
        bool output;
        bool is_prompt; // for stats tracking
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // in embd mode, we temporarily swap out the tokens arr and restore it on clear()
    bool has_embd = false;
    llama_token * tokens_ptr = nullptr;
    std::vector<float> embd;

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    server_batch() {
        batch.pos = nullptr; // sentinel: uninitialized batch
    }

    ~server_batch() {
        if (batch.pos != nullptr) {
            clear();
            llama_batch_free(batch);
        }
    }

    void init(int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        batch = llama_batch_init(n_tokens_alloc, 0, 1);
        tokens_ptr = batch.token;
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(!has_embd); // cannot mix tokens + embd in same batch
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, pos, output, is_prompt });
        return true;
    }

    bool add(int32_t id_slot, const std::vector<float> & embd_in, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, pos, output, is_prompt });
        has_embd = true;
        embd.insert(embd.end(), embd_in.begin(), embd_in.end());
        return true;
    }

    void clear() {
        tokens.clear();
        embd.clear();
        common_batch_clear(batch);
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
        batch_rendered    = false;
        has_embd          = false;
        if (batch.token == nullptr) {
            batch.token = tokens_ptr;
            batch.embd  = nullptr;
        }
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void render() {
        GGML_ASSERT(!batch_rendered);
        GGML_ASSERT(batch.pos != nullptr);
        common_batch_clear(batch);
        for (int32_t i = 0; i < size(); i++) {
            const auto & t = tokens[i];
            common_batch_add(batch, t.token, t.pos, { t.id_slot }, t.output);
        }
        if (has_embd) {
            batch.token = nullptr; // will be restored on clear()
            batch.embd  = embd.data();
        }
        batch_rendered = true;
    }

    llama_batch get_view(int32_t off, int32_t n_tokens) const {
        GGML_ASSERT(batch.pos != nullptr);
        GGML_ASSERT(batch_rendered);
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        auto * token = batch.token ? batch.token + off          : nullptr;
        auto * embd  = batch.embd  ? batch.embd  + off * n_embd : nullptr;

        llama_batch view = {
            n_tokens,
            token,
            embd,
            batch.pos      + off,
            batch.n_seq_id + off,
            batch.seq_id   + off,
            batch.logits   + off,
        };

        return view;
    }
};

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    common_memory mem;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;
    bool spec_is_replay = false;
    std::mt19937 spec_synth_rng;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx   = 0;  // context size per slot
    int32_t n_keep  = 0;
    int32_t i_batch = -1;

    // effective generation limit for the current task, -1 means unlimited
    int32_t n_predict_max = -1;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;
    size_t n_sent_text = 0; // number of sent text character (i.e. handle partial UTF-8 on streaming)

    std::vector<completion_token_output> generated_token_probs;
    telemetry_probability_accumulator response_probability;
    telemetry_probability_accumulator prompt_probability;
    std::vector<telemetry_output_token_record> telemetry_output_tokens;
    std::vector<telemetry_mtp_pass_record> telemetry_mtp_passes;
    std::vector<telemetry_mtp_proposal_record> telemetry_mtp_proposals_pending;
    std::vector<telemetry_token_candidate_decision> telemetry_token_candidate_decisions;
    const char * telemetry_token_candidate_state = "not_captured";
    const char * telemetry_token_candidate_reason = "candidate_detail_not_finalized";
    size_t telemetry_token_candidate_stored_bytes = 0;
    size_t telemetry_token_candidate_eligible_count = 0;
    size_t telemetry_token_candidate_dropped_count = 0;
    uint64_t telemetry_mtp_proposals_captured = 0;
    std::vector<uint64_t> telemetry_moe_expert_activations;
    std::vector<telemetry_moe_token_activation_record> telemetry_moe_token_activations;
    std::set<int32_t> telemetry_moe_layers;
    uint64_t telemetry_moe_routed_tokens = 0;
    uint64_t telemetry_moe_routed_token_layers = 0;
    server_moe_routing_capture_counts telemetry_moe_histogram_counts;
    uint64_t telemetry_moe_token_decisions_total = 0;
    uint64_t telemetry_moe_token_decisions_captured = 0;
    uint64_t telemetry_moe_token_decisions_invalid = 0;
    uint64_t telemetry_moe_token_decisions_cap_dropped = 0;
    server_moe_routing_capture_counts telemetry_moe_token_detail_counts;
    uint64_t telemetry_moe_chunk_sequence = 0;
    uint64_t telemetry_moe_chunk_decision_sequence = 0;
    uint64_t telemetry_moe_chunk_rows = 0;
    uint64_t telemetry_moe_chunk_trace_rows = 0;
    uint64_t telemetry_moe_chunk_invalid_rows = 0;
    uint64_t telemetry_moe_chunk_unavailable_rows = 0;
    uint64_t telemetry_moe_chunk_unlinked_rows = 0;
    uint64_t telemetry_moe_chunk_unlocated_rows = 0;
    uint64_t telemetry_moe_chunk_unlocated_pending = 0;
    bool telemetry_moe_chunk_capture_started = false;
    bool telemetry_moe_chunk_capture_interrupted = false;
    bool telemetry_moe_chunk_source_unavailable = false;
    bool telemetry_moe_chunk_attribution_ambiguous = false;
    json telemetry_moe_pending_chunk;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }

        return true;
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear() {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        mem.seq_rm(id, -1, -1);

        prompt.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    std::vector<float> inp_embd;

    server_slot_stats stats;
    bool telemetry_finalized = false;
    uint64_t telemetry_assignment_ordinal = 0;
    json telemetry_pending_completion_event;

    // accepted tokens per draft position
    // not in server_slot_stats to avoid copying to every task result
    std::vector<uint64_t> n_accepted_per_pos;
    uint64_t n_draft_hit_steps = 0;
    uint64_t n_draft_full_steps = 0;
    uint64_t n_spec_target_tokens = 0;
    uint64_t n_spec_useful_tokens = 0;
    uint64_t n_spec_target_passes = 0;
    uint64_t telemetry_spec_logical_step = 0;
    uint64_t telemetry_spec_proposed_count = 0;
    uint64_t telemetry_spec_accepted_depth = 0;

    std::function<void(int /* id_slot */)>   callback_on_release;
    std::function<void(server_slot &)>       callback_on_idle;  // called after the slot becomes idle
    std::function<void(const server_slot &)> callback_on_reset; // called before reset()

    // this is for printing timings with slot progress, not part of metrics
    int64_t t_print_last = 0;
    int32_t n_gen_last = 0;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        spec_is_replay = false;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        response_probability.reset();
        prompt_probability.reset();
        std::vector<telemetry_output_token_record>().swap(telemetry_output_tokens);
        std::vector<telemetry_mtp_pass_record>().swap(telemetry_mtp_passes);
        std::vector<telemetry_mtp_proposal_record>().swap(telemetry_mtp_proposals_pending);
        std::vector<telemetry_token_candidate_decision>().swap(telemetry_token_candidate_decisions);
        telemetry_token_candidate_state = "not_captured";
        telemetry_token_candidate_reason = "candidate_detail_not_finalized";
        telemetry_token_candidate_stored_bytes = 0;
        telemetry_token_candidate_eligible_count = 0;
        telemetry_token_candidate_dropped_count = 0;
        telemetry_mtp_proposals_captured = 0;
        telemetry_moe_expert_activations.clear();
        std::vector<telemetry_moe_token_activation_record>().swap(telemetry_moe_token_activations);
        telemetry_moe_layers.clear();
        telemetry_moe_routed_tokens = 0;
        telemetry_moe_routed_token_layers = 0;
        telemetry_moe_histogram_counts = {};
        telemetry_moe_token_decisions_total = 0;
        telemetry_moe_token_decisions_captured = 0;
        telemetry_moe_token_decisions_invalid = 0;
        telemetry_moe_token_decisions_cap_dropped = 0;
        telemetry_moe_token_detail_counts = {};
        telemetry_moe_chunk_sequence = 0;
        telemetry_moe_chunk_decision_sequence = 0;
        telemetry_moe_chunk_rows = 0;
        telemetry_moe_chunk_trace_rows = 0;
        telemetry_moe_chunk_invalid_rows = 0;
        telemetry_moe_chunk_unavailable_rows = 0;
        telemetry_moe_chunk_unlinked_rows = 0;
        telemetry_moe_chunk_unlocated_rows = 0;
        telemetry_moe_chunk_unlocated_pending = 0;
        telemetry_moe_chunk_capture_started = false;
        telemetry_moe_chunk_capture_interrupted = false;
        telemetry_moe_chunk_source_unavailable = false;
        telemetry_moe_chunk_attribution_ambiguous = false;
        telemetry_moe_pending_chunk = json();
        json_schema = json();

        task_prev = std::move(task);
        task.reset();

        // note: callback_on_reset() must have run before this, see release()
        stats = {};
        telemetry_finalized = false;
        telemetry_assignment_ordinal = 0;
        telemetry_pending_completion_event = json();
        n_accepted_per_pos.clear();
        n_draft_hit_steps = 0;
        n_draft_full_steps = 0;
        n_spec_target_tokens = 0;
        n_spec_useful_tokens = 0;
        n_spec_target_passes = 0;
        telemetry_spec_logical_step = 0;
        telemetry_spec_proposed_count = 0;
        telemetry_spec_accepted_depth = 0;

        n_predict_max = -1;

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd();
    }

    bool should_score_prompt() const {
        GGML_ASSERT(task);
        return task->params.prompt_perplexity && prompt_probability.unavailable_reason.empty();
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type
            && inp_embd.size() == other_slot.inp_embd.size()
            && are_lora_equal(lora, other_slot.lora);
    }

    // returns -1 if the generation is limitless
    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining() > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining() - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            if (!inp_embd.empty()) {
                add_ok &= batch.add(id, inp_embd, prompt.tokens.pos_next(), true, false);
            } else {
                add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true, false);
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(id, sampled, pos0++, true, false);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true, false);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used = ggml_time_us();
            stats.t_release = t_last_used;

            state = SLOT_STATE_IDLE;

            callback_on_idle(*this);

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear();
            }

            callback_on_reset(*this);

            reset();

            callback_on_release(id);
        }
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (stats.n_gen < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = stats.n_gen_tps();
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (stats.n_gen - n_gen_last);

        t_print_last = t_now;
        n_gen_last = stats.n_gen;

        SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) stats.n_gen, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double t_prompt_total = stats.t_prompt_ms();

        if (t_prompt_total < 3000.0) {
            return;
        }

        const double n_prompt_second = stats.n_prompt_tps();
        const double f_progress = task->n_tokens() > 0 ? (double) prompt.n_tokens() / task->n_tokens() : 0.0;

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                (int) stats.n_prompt_processed, f_progress, t_prompt_total / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt_total = stats.t_prompt_ms();
        const double t_gen_total    = stats.t_gen_ms();

        const double t_prompt        = stats.t_prompt_per_token_ms();
        const double n_prompt_second = stats.n_prompt_tps();

        const double t_gen        = stats.t_gen_per_token_ms();
        const double n_gen_second = stats.n_gen_tps();

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_total, (int) stats.n_prompt_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_gen_total, (int) stats.n_gen, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_total + t_gen_total, (int) (stats.n_prompt_processed + stats.n_gen));

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        const int32_t n_draft_total       = stats.n_draft_tokens;
        const int32_t n_draft_accepted    = stats.n_draft_accepted;
        const int32_t n_draft_verif_steps = stats.n_draft_verif_steps;

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = stats.n_prompt_processed;
            res["n_prompt_tokens_cache"]     = stats.n_prompt_cached;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = json::array({
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining()},
                    {"n_decoded",      stats.n_gen},
                }
            });

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.i_batch = i_batch;

        other.stats = stats;
        other.prompt_probability = prompt_probability;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};

// returns 0 on success
// caller need to update prompt.tokens after a successful call to keep track of the processing progress
// note: this is not a member of server_slot because we want to run it inside yield_to_queue
//       slot is passed as const to avoid accidental modification of the slot state
//       some pointers are allowed to be used, they are not used by to_json()
static int process_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx, size_t & n_tokens_out) {
    GGML_ASSERT(slot.mctx);
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);
    int32_t res = 0;

    auto try_decode = [&]() -> int32_t {
        if (mbatch) {
            float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
            if (embd) {
                void * cb_data = slot.spec;
                static auto cb = [](llama_batch batch, void * user_data) {
                    common_speculative * spec = static_cast<common_speculative *>(user_data);
                    if (!common_speculative_process(spec, batch)) {
                        return 1;
                    }
                    return 0;
                };

                llama_pos new_n_past; // unused for now
                res = mtmd_helper_decode_image_chunk(
                    mctx,
                    slot.ctx_tgt,
                    chunk.get(),
                    embd,
                    slot.prompt.tokens.pos_next(),
                    slot.id,
                    llama_n_batch(slot.ctx_tgt),
                    &new_n_past,
                    cb,
                    cb_data
                );
                if (res != 0) {
                    SLT_ERR(slot, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                    return -1;
                }
                n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                return 0; // success
            }
        }
        return 1; // (non-error) need to create & encode batch
    };

    // if the batch is already exist, try searching & encode
    res = try_decode();
    if (res == 0) {
        return 0;
    }
    if (res < 0) {
        // fatal error
        return res;
    }

    // otherwise, the batch is either uninitialized or is used up
    // we need to create & encode a new batch
    mbatch.reset(mtmd_batch_init(mctx));
    res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
    GGML_ASSERT(res == 0); // we should never have an empty batch

    // try batching as much as possible
    int n_added = 1;
    size_t idx_cur = idx;
    while (res == 0) {
        auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
        if (next_chunk == nullptr) {
            break;
        }
        res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
        n_added += (res == 0 ? 1 : 0);
        idx_cur = next_idx;
        SLT_DBG(slot, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
        // if res != 0, batch is full or chunk is not compatible -> this loop breaks
    }

    // TODO @ngxson : move this log line to debug when it become more stable
    SLT_TRC(slot, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

    res = mtmd_batch_encode(mbatch.get());
    if (res != 0) {
        SLT_ERR(slot, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
        return -1;
    }

    return try_decode();
}

//
// server_context_impl (private implementation)
//

static std::string server_moe_routing_partial_reason(
        const server_moe_routing_chunk_coverage & coverage,
        const char * prefix) {
    std::string result = prefix;
    bool first = true;
    const auto append = [&](const char * cause) {
        result += first ? " " : "; ";
        result += cause;
        first = false;
    };
    if (coverage.invalid_rows > 0) {
        append("invalid router rows");
    }
    if (coverage.unavailable_rows > 0) {
        append("unavailable native routing values");
    }
    if (coverage.unlinked_rows > 0) {
        append("unlinked routing rows");
    }
    if (coverage.unlocated_rows > 0) {
        append("rows lost before complete routing coordinates were retained");
    }
    if (coverage.interrupted) {
        append("routing capture was interrupted");
    }
    if (coverage.source_unavailable) {
        append("the native routing source was unavailable");
    }
    if (coverage.attribution_ambiguous) {
        append("unmappable routing rows could not be attributed to this request");
    }
    if (coverage.serialization_gaps) {
        append("serialized routing records could not be retained");
    }
    return result + ".";
}

void server_moe_routing_apply_canonical_event_coverage(
        json & event,
        const server_moe_routing_chunk_coverage & coverage,
        bool has_routable_records,
        const char * partial_reason_prefix,
        const char * unlocated_loss_reason,
        const char * no_records_reason) {
    const bool partial = server_moe_routing_chunk_is_partial(coverage);
    event["availability"] = server_moe_routing_chunk_availability(coverage, has_routable_records);
    if (partial) {
        event["reason"] = server_moe_routing_partial_reason(coverage, partial_reason_prefix);
    } else if (no_records_reason != nullptr) {
        event["reason"] = no_records_reason;
    }
    if (coverage.unlocated_rows > 0) {
        event["unlocated_coverage_loss"] = {
            {"count", coverage.unlocated_rows},
            {"reason", unlocated_loss_reason},
        };
    }
}

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    // note: video_params.ffmpeg_bin_dir points into params_base, which outlives this struct
    mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

    static void copy_ubatch_stats(
            server_metrics::physical_ubatch_metrics & dst,
            const llama_ubatch_stats & src) {
        dst.attempted        = src.attempted;
        dst.successful       = src.successful;
        dst.tokens           = src.tokens;
        dst.sequence_tokens  = src.sequence_tokens;
        dst.sequences        = src.sequences;
        dst.unique_sequences = src.unique_sequences;
        dst.max_tokens       = src.max_tokens;
        dst.token_histogram.sum = src.tokens;
        dst.token_histogram.count = src.successful;
        GGML_ASSERT(dst.token_histogram.buckets.size() == src.token_buckets.size());
        std::copy(src.token_buckets.begin(), src.token_buckets.end(), dst.token_histogram.buckets.begin());
    }

    server_metrics get_metrics() const {
        server_metrics result = metrics;
        if (ctx_tgt) {
            copy_ubatch_stats(result.physical_ubatch_target, llama_get_ubatch_stats(ctx_tgt));
        }
        if (ctx_dft) {
            copy_ubatch_stats(result.physical_ubatch_draft, llama_get_ubatch_stats(ctx_dft));
        }
        return result;
    }

    telemetry_control_state telemetry_control_current() const {
        std::lock_guard<std::mutex> lock(mutex_telemetry_control);
        return telemetry_control;
    }

    telemetry_control_application telemetry_control_apply(telemetry_control_state next) {
        std::lock_guard<std::mutex> lock(mutex_telemetry_control);
        const bool microbatch_changed =
                telemetry_control.moe_routing != next.moe_routing ||
                telemetry_control.kv_pressure_detail != next.kv_pressure_detail ||
                telemetry_control.native_gpu_gpm != next.native_gpu_gpm;
        next.generation = telemetry_control.generation + 1;
        telemetry_control = next;
        return {
            telemetry_control,
            microbatch_changed ? "next_microbatch" : "next_request",
        };
    }

    static json telemetry_control_effective_json(const telemetry_control_state & control) {
        return {
            {"moe_routing", control.moe_routing},
            {"output_token_detail", control.output_token_detail},
            {"token_candidates", control.token_candidates},
            {"prompt_perplexity", control.prompt_perplexity},
            {"request_content", control.request_content},
            {"kv_pressure_detail", control.kv_pressure_detail},
            {"native_gpu_gpm", control.native_gpu_gpm},
        };
    }

    json telemetry_control_applicability_json() const {
        const json gpu = gpu_telemetry.capability_json();
        const bool gpu_applicable = gpu.value("state", "unavailable") == "available";
        return {
            {"moe_routing", llama_model_n_expert(model_tgt) > 0
                ? json { {"applicable", true} }
                : json { {"applicable", false}, {"reason", "The loaded target model has no routed MoE experts."} }},
            {"output_token_detail", {{"applicable", true}}},
            {"token_candidates", {{"applicable", true}}},
            {"prompt_perplexity", {{"applicable", true}}},
            {"request_content", {{"applicable", true}}},
            {"kv_pressure_detail", {{"applicable", true}}},
            {"native_gpu_gpm", gpu_applicable
                ? json { {"applicable", true} }
                : json {
                    {"applicable", false},
                    {"reason", gpu.value("reason", "Native NVML GPM is not available for the active backend.")},
                }},
        };
    }

    static json telemetry_control_state_json(const telemetry_control_state & control) {
        return {
            {"effective", telemetry_control_effective_json(control)},
            {"generation", control.generation},
        };
    }

    json telemetry_control_snapshot_json() const {
        return telemetry_control_state_json(telemetry_control_current());
    }

    json telemetry_control_capability_json() const {
        return {
            {"supported", true},
            {"route", "/props"},
            {"method", "POST"},
            {"requires_props", true},
            {"requires_authentication", true},
            {"requires_loopback", true},
            {"replacement_semantics", "full"},
            {"features", {
                {"moe_routing", {
                    {"supported", true},
                    {"effective_from", "next_microbatch"},
                    {"dependencies", json::array({"moe_model"})},
                    {"privacy_sensitive", false},
                }},
                {"output_token_detail", {
                    {"supported", true},
                    {"effective_from", "next_request"},
                    {"dependencies", json::array()},
                    {"privacy_sensitive", true},
                }},
                {"token_candidates", {
                    {"supported", true},
                    {"effective_from", "next_request"},
                    {"dependencies", json::array({"output_token_detail"})},
                    {"privacy_sensitive", true},
                }},
                {"prompt_perplexity", {
                    {"supported", true},
                    {"effective_from", "next_request"},
                    {"dependencies", json::array()},
                    {"privacy_sensitive", false},
                }},
                {"request_content", {
                    {"supported", true},
                    {"effective_from", "next_request"},
                    {"dependencies", json::array()},
                    {"privacy_sensitive", true},
                }},
                {"kv_pressure_detail", {
                    {"supported", true},
                    {"effective_from", "next_microbatch"},
                    {"dependencies", json::array()},
                    {"privacy_sensitive", false},
                }},
                {"native_gpu_gpm", {
                    {"supported", true},
                    {"effective_from", "next_microbatch"},
                    {"dependencies", json::array()},
                    {"privacy_sensitive", false},
                }},
            }},
        };
    }

    size_t telemetry_token_candidate_max_bytes() const {
        return telemetry_token_candidate_max_block_bytes;
    }

    size_t telemetry_output_tokens_limit() const {
        return telemetry_output_token_limit;
    }

    size_t telemetry_mtp_passes_limit() const {
        return telemetry_mtp_pass_limit;
    }

    size_t telemetry_mtp_proposals_limit() const {
        return telemetry_mtp_proposal_limit;
    }

    size_t telemetry_token_candidate_decisions_limit() const {
        return telemetry_token_candidate_decision_limit;
    }

    size_t telemetry_moe_activation_limit_value() const {
        return telemetry_moe_activation_limit;
    }

    size_t telemetry_event_capacity_bytes() const {
        return telemetry_event_max_bytes;
    }

    const std::string & telemetry_instance_id() const {
        return telemetry_server_instance_id;
    }

    json telemetry_gpu_capability_json() const {
        return gpu_telemetry.capability_json();
    }

    json telemetry_kv_pressure_capability_json() const {
        const llama_memory_primary_occupancy & occupancy = telemetry_kv_boundary.memory.primary_occupancy;
        const bool primary_occupancy_available = telemetry_kv_boundary.available && occupancy.available &&
            occupancy.capacity_entries > 0 && occupancy.used_entries <= occupancy.capacity_entries;
        return {
            {"state", primary_occupancy_available ? "available" : "partial"},
            {"reason", primary_occupancy_available
                ? "Bounded causal KV-pressure events and lightweight primary occupancy are available."
                : "Causal KV-pressure events are available, but primary occupancy is unavailable."},
            {"owner", "llama.cpp/llama-server"},
            {"endpoint", "/telemetry/v1/kv-pressure"},
            {"schema_version", 1},
            {"sampling_interval_ms", telemetry_kv_pressure_sampling_interval_us / 1000},
            {"event_capacity", TELEMETRY_KV_PRESSURE_EVENT_CAPACITY},
            {"serialized_event_capacity_bytes", telemetry_kv_pressure_event_max_bytes},
            {"request_window_capacity", telemetry_kv_request_window_capacity},
            {"event_kinds", json::array({
                "utilization_sample",
                "decode_wait_started",
                "decode_retry",
                "decode_wait_finished",
                "idle_slot_evicted",
                "context_shift",
            })},
            {"decode_wait_semantics", "llama_decode_returned_1_no_kv_slot_available"},
            {"retry_size_semantics", "effective_server_decode_batch_limit"},
        };
    }

    void reset_metrics_bucket() {
        metrics.reset_bucket();
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    server_batch batch;

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;

    common_speculative_init_result_ptr spec_init;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;        // env: LLAMA_TRACE
    int slots_debug = 0;  // env: LLAMA_SERVER_SLOTS_DEBUG
    int slots_n_diff = 0; // env: LLAMA_SERVER_SLOTS_N_DIFF

    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;
    server_gpu_telemetry gpu_telemetry;
    std::vector<int64_t> gpu_verify_proposal_positions;

    struct telemetry_event_entry {
        uint64_t sequence = 0;
        std::string serialized;
        size_t bytes = 0;
    };
    struct telemetry_kv_pressure_event_entry {
        uint64_t sequence = 0;
        std::string serialized;
        size_t bytes = 0;
        std::string kind;
        std::string trace_id;
        std::string victim_trace_id;
        int64_t monotonic_us = 0;
    };
    struct telemetry_kv_request_window {
        std::string trace_id;
        int64_t start_monotonic_us = 0;
        int64_t end_monotonic_us = 0;
    };
    struct telemetry_kv_wait_identity {
        std::string trace_id;
        int32_t task_id = -1;
        int32_t slot_id = -1;
        int64_t started_monotonic_us = 0;
        std::set<int32_t> pending_batch_indices;
    };
    struct telemetry_kv_wait_episode {
        std::string id;
        int64_t started_monotonic_us = 0;
        int32_t retry_count = 0;
        std::vector<telemetry_kv_wait_identity> identities;

        bool active() const {
            return started_monotonic_us != 0;
        }

        void clear() {
            id.clear();
            started_monotonic_us = 0;
            retry_count = 0;
            identities.clear();
        }
    };
    struct telemetry_kv_eviction_result {
        bool cleared = false;
        int32_t victim_slot_id = -1;
        std::string victim_trace_id;
        uint64_t victim_prompt_tokens = 0;
        bool released_entries_available = false;
        uint64_t released_entries = 0;
        uint64_t memberships_removed = 0;
    };
    struct telemetry_kv_slot_snapshot {
        int32_t id = -1;
        std::vector<llama_token> tokens;
        std::string compatibility;
    };
    struct telemetry_kv_boundary_snapshot {
        llama_memory_snapshot memory;
        llama_memory_breakdown breakdown;
        int64_t monotonic_us = 0;
        uint64_t resident_slot_tokens = 0;
        size_t represented_slots = 0;
        size_t multimodal_sequences_skipped = 0;
        std::vector<telemetry_kv_slot_snapshot> slots;
        bool available = false;
    };
    std::deque<telemetry_event_entry> telemetry_events;
    std::deque<telemetry_kv_pressure_event_entry> telemetry_kv_pressure_events;
    std::deque<telemetry_kv_request_window> telemetry_kv_request_windows;
    std::deque<telemetry_token_candidate_block_entry> telemetry_token_candidate_blocks;
    std::deque<std::string> telemetry_token_candidate_expired_trace_ids;
    std::string telemetry_server_instance_id = "server-" + random_string();
    uint64_t telemetry_next_sequence = 1;
    uint64_t telemetry_next_assignment_ordinal = 1;
    uint64_t telemetry_dropped_events = 0;
    uint64_t telemetry_last_dropped_sequence = 0;
    size_t telemetry_event_bytes = 0;
    size_t telemetry_event_max_bytes = 64 * 1024 * 1024;
    size_t telemetry_moe_chunk_max_bytes = 1024 * 1024;
    uint64_t telemetry_kv_pressure_next_sequence = 1;
    uint64_t telemetry_kv_pressure_dropped_events = 0;
    uint64_t telemetry_kv_pressure_last_dropped_sequence = 0;
    uint64_t telemetry_kv_pressure_next_episode = 1;
    size_t telemetry_kv_pressure_event_bytes = 0;
    size_t telemetry_kv_pressure_event_max_bytes = 16 * 1024 * 1024;
    size_t telemetry_kv_request_window_capacity = 4096;
    int64_t telemetry_kv_pressure_sampling_interval_us = 100 * 1000;
    int64_t telemetry_kv_pressure_last_sample_us = 0;
    std::string telemetry_kv_primary_component;
    std::string telemetry_kv_primary_memory_kind;
    std::string telemetry_kv_primary_entry_semantics;
    telemetry_kv_boundary_snapshot telemetry_kv_boundary;
    telemetry_kv_wait_episode telemetry_kv_wait;
    std::vector<uint64_t> telemetry_slot_marks;
    uint64_t telemetry_slot_epoch = 0;
    mutable std::mutex mutex_telemetry_control;
    telemetry_control_state telemetry_control;
    bool telemetry_gpu_gpm_active = false;
    bool telemetry_kv_pressure_active = false;
    size_t telemetry_output_token_limit = 512;
    size_t telemetry_mtp_pass_limit = 512;
    size_t telemetry_mtp_proposal_limit = 512;
    size_t telemetry_token_candidate_decision_limit = 512;
    size_t telemetry_token_candidate_max_block_bytes = 1024 * 1024;
    size_t telemetry_token_candidate_retained_bytes = 0;
    static constexpr size_t TELEMETRY_TOKEN_CANDIDATE_BLOCK_CAPACITY = 256;
    static constexpr size_t TELEMETRY_TOKEN_CANDIDATE_RETAINED_MAX_BYTES = 64 * 1024 * 1024;
    size_t telemetry_moe_activation_limit = 65536;
    static constexpr size_t TELEMETRY_EVENT_CAPACITY = 2048;
    static constexpr size_t TELEMETRY_KV_PRESSURE_EVENT_CAPACITY = 32768;
    static constexpr size_t TELEMETRY_KV_REQUEST_WINDOW_MAX_CAPACITY = 32768;

    // queued prompt stats - llama_decode() is async, so the timing is only valid after a sync
    // note: kept out of server_metrics, which is copied as-is into the task result
    int64_t  t_decode_start  = 0; // start of the last submitted decode
    int64_t  t_prompt_start  = 0; // start of the oldest queued prompt decode
    uint64_t n_prompt_queued = 0;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    int64_t t_last_load_progress_ms = 0;

    void destroy() {
        gpu_telemetry.stop();
        telemetry_gpu_gpm_active = false;
        spec.reset();
        spec_init.reset();

        ctx_dft   = nullptr;
        model_dft = nullptr;

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
    }

    void telemetry_apply_micro_controls() {
        const telemetry_control_state control = telemetry_control_current();
        if (control.kv_pressure_detail != telemetry_kv_pressure_active) {
            telemetry_kv_pressure_active = control.kv_pressure_detail;
            telemetry_kv_wait.clear();
            if (telemetry_kv_pressure_active) {
                telemetry_kv_pressure_initialize();
                telemetry_kv_pressure_sample(0, true);
                for (const auto & slot : slots) {
                    if (slot.is_processing() && slot.task) {
                        telemetry_kv_request_started(slot);
                    }
                }
            }
        }
        if (control.native_gpu_gpm == telemetry_gpu_gpm_active) {
            return;
        }
        if (control.native_gpu_gpm) {
            gpu_telemetry.start();
        } else {
            gpu_telemetry.stop();
        }
        telemetry_gpu_gpm_active = control.native_gpu_gpm;
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            if (callback_state) {
                callback_state(SERVER_STATE_SLEEPING, {});
                // note: for sleeping == false, event is emitted by load_model()
            }
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;

        params_base = params;
        const auto output_limits = server_output_limits(params_base);
        params_base.n_outputs_max = output_limits.total;
        params_base.n_outputs_max_per_seq = output_limits.per_seq;

        const bool has_mmproj = !params.mmproj.path.empty();
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        const bool has_spec = has_draft || spec_mtp;

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        std::string & mmproj_path = params_base.mmproj.path;
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.device           = params_base.mmproj_device;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
            mparams.media_marker     = get_media_marker();
            // progress callback
            mparams.progress_callback           = load_progress_callback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // optionally get the memory usage of mmproj
        if (has_mmproj && params_base.fit_params) {
            int64_t t_start = ggml_time_us();
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            int64_t t_elapsed = ggml_time_us() - t_start;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // note: the draft / MTP context is fitted together with the target model, see common_fit_extra_model

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create_context with model '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (has_spec) {
            // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
            load_progress_callback(0.0f, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0;  // reset so internal cbs aren't delayed

            {
                common_params params_dft = common_base_params_to_speculative(params_base);

                // progress callback
                params_dft.load_progress_callback           = load_progress_callback;
                params_dft.load_progress_callback_user_data = &load_progress_spec;

                spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
                model_dft = spec_init->model();
                ctx_dft   = spec_init->context();

                if (has_draft && model_dft == nullptr) {
                    SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }

                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create MTP context\n");
                    return false;
                }

                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft;
            }

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", "mmproj_model"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            init_opt.video_params.fps_target = params_base.video_fps;
            init_opt.video_params.timestamp_interval_ms = params_base.video_timestamp_interval_ms;
            init_opt.video_params.ffmpeg_bin_dir = params_base.video_ffmpeg_bin_dir.empty()
                                ? nullptr : params_base.video_ffmpeg_bin_dir.c_str();

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        {
            // note: the capping itself is done in n_ctx_slot(), here we only report it
            const int n_ctx_seq = llama_n_ctx_seq(ctx_tgt);

            if (params_base.kv_unified_per_slot > 0) {
                if (n_ctx_seq > params_base.kv_unified_per_slot) {
                    SRV_INF("capping per-slot context (%d) to --kv-unified-per-slot (%d)\n",
                            n_ctx_seq, params_base.kv_unified_per_slot);
                } else if (params_base.kv_unified_per_slot > n_ctx_seq) {
                    // cap is above the per-slot pool capacity, so it can never bind
                    SRV_WRN(
                        "--kv-unified-per-slot (%d) exceeds the per-slot pool capacity (%d) - cap has no effect, "
                        "slots are limited to %d (raise the KV pool with -c, or unset -c to size it to "
                        "n_parallel * kv_unified_per_slot)\n",
                        params_base.kv_unified_per_slot, n_ctx_seq, n_ctx_seq);
                }
            }

            const int n_ctx_capped = params_base.kv_unified_per_slot > 0 ?
                std::min(n_ctx_seq, params_base.kv_unified_per_slot) : n_ctx_seq;

            if (n_ctx_capped > n_ctx_train) {
                SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n",
                        n_ctx_capped, n_ctx_train);
            }
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n",
                params_base.n_parallel, n_ctx_slot(), params_base.kv_unified ? "true" : "false");

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }
        telemetry_slot_marks.assign(slots.size(), 0);
        gpu_verify_proposal_positions.assign(slots.size(), 0);

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
                if (params_base.speculative.has_synth()) {
                    return false;
                }
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            spec_init.reset();
            ctx_dft   = nullptr;
            model_dft = nullptr;
        }

        if (!spec && params_base.speculative.has_synth()) {
            SRV_ERR("%s", "synthetic acceptance requires an initialized speculative decoding context\n");
            return false;
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft;
            slot.mem.init(ctx_tgt, ctx_dft);
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot();

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.callback_on_idle = [this](server_slot & slot) {
                telemetry_on_release(slot);
            };

            slot.callback_on_reset = [this](const server_slot & slot) {
                // flush the generated token stats before reset()
                if (slot.stats.n_gen > 0) {
                    metrics_on_prediction(slot);
                }
            };

            slot.reset();
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_N_DIFF = getenv("LLAMA_SERVER_SLOTS_N_DIFF");
            slots_n_diff = LLAMA_SERVER_SLOTS_N_DIFF ? atoi(LLAMA_SERVER_SLOTS_N_DIFF) : 0;

            if (slots_n_diff) {
                SRV_WRN("LLAMA_SERVER_SLOTS_N_DIFF = %d\n", slots_n_diff);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch.init(std::max(n_batch, params_base.n_parallel), n_embd);
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task, bool is_yielding) {
            return process_single_task(std::move(task), is_yielding);
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        const char * telemetry_output_token_limit_env = getenv("LLAMA_TELEMETRY_OUTPUT_TOKEN_LIMIT");
        if (telemetry_output_token_limit_env) {
            telemetry_output_token_limit = (size_t) std::max(32, std::min(8192, atoi(telemetry_output_token_limit_env)));
        }
        const char * telemetry_mtp_pass_limit_env = getenv("LLAMA_TELEMETRY_MTP_PASS_LIMIT");
        if (telemetry_mtp_pass_limit_env) {
            telemetry_mtp_pass_limit = (size_t) std::max(32, std::min(4096, atoi(telemetry_mtp_pass_limit_env)));
        }
        const char * telemetry_mtp_proposal_limit_env = getenv("LLAMA_TELEMETRY_MTP_PROPOSAL_LIMIT");
        if (telemetry_mtp_proposal_limit_env) {
            telemetry_mtp_proposal_limit = (size_t) std::max(32, std::min(4096, atoi(telemetry_mtp_proposal_limit_env)));
        }
        const char * telemetry_candidate_decision_limit_env = getenv("LLAMA_TELEMETRY_TOKEN_CANDIDATE_DECISION_LIMIT");
        if (telemetry_candidate_decision_limit_env) {
            telemetry_token_candidate_decision_limit = (size_t) std::max(8, std::min(4096, atoi(telemetry_candidate_decision_limit_env)));
        }
        const char * telemetry_token_candidate_max_bytes_env = getenv("LLAMA_TELEMETRY_TOKEN_CANDIDATE_MAX_BYTES");
        if (telemetry_token_candidate_max_bytes_env) {
            telemetry_token_candidate_max_block_bytes = (size_t) std::max(
                8192,
                std::min(1024 * 1024, atoi(telemetry_token_candidate_max_bytes_env)));
        }
        const char * telemetry_moe_activation_limit_env = getenv("LLAMA_TELEMETRY_MOE_ACTIVATION_LIMIT");
        if (telemetry_moe_activation_limit_env) {
            telemetry_moe_activation_limit = (size_t) std::max(1024, std::min(1048576, atoi(telemetry_moe_activation_limit_env)));
        }
        const char * telemetry_moe_chunk_max_bytes_env = getenv("LLAMA_TELEMETRY_MOE_CHUNK_MAX_BYTES");
        if (telemetry_moe_chunk_max_bytes_env) {
            telemetry_moe_chunk_max_bytes = (size_t) std::max(1024, std::min(1024 * 1024, atoi(telemetry_moe_chunk_max_bytes_env)));
        }
        const char * telemetry_buffer_env = getenv("LLAMA_TELEMETRY_EVENT_BUFFER_MIB");
        if (telemetry_buffer_env) {
            const int mib = std::max(1, std::min(4096, atoi(telemetry_buffer_env)));
            telemetry_event_max_bytes = (size_t) mib * 1024 * 1024;
        }
        const char * telemetry_kv_pressure_buffer_env = getenv("LLAMA_TELEMETRY_KV_PRESSURE_BUFFER_MIB");
        if (telemetry_kv_pressure_buffer_env) {
            const int mib = std::max(1, std::min(1024, atoi(telemetry_kv_pressure_buffer_env)));
            telemetry_kv_pressure_event_max_bytes = (size_t) mib * 1024 * 1024;
        }
        const char * telemetry_kv_pressure_interval_env = getenv("LLAMA_TELEMETRY_KV_PRESSURE_INTERVAL_MS");
        if (telemetry_kv_pressure_interval_env) {
            const int interval_ms = std::max(10, std::min(5000, atoi(telemetry_kv_pressure_interval_env)));
            telemetry_kv_pressure_sampling_interval_us = (int64_t) interval_ms * 1000;
        }
        const char * telemetry_kv_request_window_env = getenv("LLAMA_TELEMETRY_KV_PRESSURE_REQUEST_WINDOW_LIMIT");
        if (telemetry_kv_request_window_env) {
            telemetry_kv_request_window_capacity = (size_t) std::max(
                1,
                std::min((int) TELEMETRY_KV_REQUEST_WINDOW_MAX_CAPACITY, atoi(telemetry_kv_request_window_env)));
        }
        telemetry_kv_snapshot_capture(false);
        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* allow_video           */ mctx ? mtmd_helper_support_video(mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning");
                bool enabled = it != params_base.default_template_kwargs.end();
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
            }
        }

        // find the slot that has at least n% prompt similarity
        if (slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;

            for (server_slot & slot : slots) {
                if (task.id_slot != -1 && slot.id != task.id_slot) {
                    continue;
                }

                // skip the slot if it is not available
                if (slot.is_processing()) {
                    SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing());
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const size_t lcp_len = tokens.get_common_prefix(task.tokens);
                const float f_sim_cur = float(lcp_len) / task.tokens.size();

                SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

                // select the current slot if the criteria match
                if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                    f_sim_best = f_sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                if (task.id_slot == -1) {
                    SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                            f_sim_best, slot_prompt_similarity, f_keep);
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);

                update_cache = true;
            }
        }

        if (ret && !ret->is_processing()) {
            task.t_slot_start = ggml_time_us();
            task.t_cache_start = task.t_slot_start;
        }

        if (ret) {
            update_cache = update_cache && prompt_cache;

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache);

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear();
                }

                prompt_cache->update();

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    telemetry_kv_eviction_result clear_idle_slot(server_slot & slot) {
        telemetry_kv_eviction_result result;
        result.cleared = true;
        if (!telemetry_kv_pressure_active) {
            slot.prompt_clear();
            return result;
        }
        result.victim_slot_id = slot.id;
        result.victim_prompt_tokens = slot.prompt.n_tokens();
        if (slot.task_prev) {
            result.victim_trace_id = slot.task_prev->trace_id;
        }
        const llama_memory_primary_occupancy occupancy_before = llama_get_memory_primary_occupancy(ctx_tgt);
        const llama_memory_diagnostics diagnostics_before = llama_get_memory_diagnostics(ctx_tgt);
        slot.prompt_clear();
        const llama_memory_primary_occupancy occupancy_after = llama_get_memory_primary_occupancy(ctx_tgt);
        const llama_memory_diagnostics diagnostics_after = llama_get_memory_diagnostics(ctx_tgt);
        result.released_entries_available = occupancy_before.available && occupancy_after.available &&
            occupancy_before.used_entries >= occupancy_after.used_entries;
        if (result.released_entries_available) {
            result.released_entries = occupancy_before.used_entries - occupancy_after.used_entries;
        }
        if (diagnostics_after.churn.memberships_removed >= diagnostics_before.churn.memberships_removed) {
            result.memberships_removed = diagnostics_after.churn.memberships_removed -
                diagnostics_before.churn.memberships_removed;
        }
        return result;
    }

    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    telemetry_kv_eviction_result try_clear_idle_slots() {
        if (!params_base.kv_unified) {
            return {};
        }

        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());
                return clear_idle_slot(slot);
            }
        }

        return {};
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool use_backend_sampling = task.params.sampling.backend_sampling;

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            use_backend_sampling &= !need_pre_sample_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (use_backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());

            if (spec && !common_speculative_get_synth_probs(spec.get()).empty()) {
                const uint32_t seed = task.params.sampling.seed == LLAMA_DEFAULT_SEED
                    ? std::random_device{}()
                    : task.params.sampling.seed;
                slot.spec_synth_rng.seed(seed);
            }
        } else {
            slot.smpl.reset();
        }

        // the per-request limit takes priority over the global one
        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;

        slot.task = std::make_unique<const server_task>(std::move(task));
        slot.telemetry_assignment_ordinal = telemetry_next_assignment_ordinal++;

        slot.stats.trace_id = slot.task->trace_id;
        slot.stats.t_arrival = slot.task->t_arrival;
        slot.stats.t_enqueue = slot.task->t_enqueue;
        slot.stats.t_slot_start = slot.task->t_slot_start;
        slot.stats.t_cache_start = slot.task->t_cache_start;
        slot.stats.t_arrival_unix_ms = slot.task->t_arrival_unix_ms;

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        telemetry_on_start(slot);

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_gen = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), (int) slot.stats.n_gen, slot.n_ctx);
        }

        // check the limits
        if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget()) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_gen = %d, n_predict = %d\n", (int) slot.stats.n_gen, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", (int) slot.stats.n_gen, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && slot.stats.t_gen_ms() > slot.task->params.t_max_predict_ms) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", (int) slot.stats.n_gen, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", (int) slot.stats.n_gen, slot.n_remaining(), result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    bool populate_token_probs(
            const server_slot & slot,
            completion_token_output & result,
            bool post_sampling,
            bool special,
            int idx,
            double & raw_selected_logprob,
            std::string & raw_unavailable_reason) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;
        raw_selected_logprob = std::numeric_limits<double>::quiet_NaN();
        raw_unavailable_reason.clear();

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            std::vector<llama_token_data> cur;
            const bool raw_available = raw_target_token_probabilities(
                ctx_tgt,
                idx,
                result.tok,
                0,
                raw_selected_logprob,
                nullptr,
                &cur,
                n_probs_request,
                raw_unavailable_reason);
            if (!raw_available) {
                cur = get_token_probabilities(ctx_tgt, idx, n_probs_request);
            }
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            if (raw_available) {
                result.prob = (float) std::exp(raw_selected_logprob);
            } else {
                for (size_t i = 0; i < max_probs; i++) {
                    if (cur[i].id == result.tok) {
                        result.prob = cur[i].p;
                        break;
                    }
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }

            return raw_available;
        }

        return false;
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type, 0, 0, task.trace_id);
    }

    void send_error(server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        const char * category = "server";
        switch (type) {
            case ERROR_TYPE_INVALID_REQUEST: category = "invalid_request"; break;
            case ERROR_TYPE_AUTHENTICATION: category = "authentication"; break;
            case ERROR_TYPE_NOT_FOUND: category = "not_found"; break;
            case ERROR_TYPE_PERMISSION: category = "permission"; break;
            case ERROR_TYPE_NOT_SUPPORTED: category = "not_supported"; break;
            case ERROR_TYPE_UNAVAILABLE: category = "unavailable"; break;
            case ERROR_TYPE_EXCEED_CONTEXT_SIZE: category = "context_size"; break;
            case ERROR_TYPE_SERVER: break;
        }
        telemetry_finalize(slot, "error", error, category);
        const int64_t t_handoff = send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx, slot.task->trace_id);
        telemetry_on_response_handoff(slot, t_handoff);
    }

    int64_t send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0, const std::string & trace_id = {}) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->trace_id        = trace_id;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        return queue_results.send(std::move(res), true);
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->trace_id = slot.task->trace_id;
        res->index = slot.task->index;

        if (is_progress) {
            const int64_t monotonic_us = ggml_time_us();
            GGML_ASSERT(slot.stats.t_start > 0);
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.stats.n_prompt_cached;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.prompt_start_monotonic_us = slot.stats.t_start;
            res->progress.monotonic_us = monotonic_us;
            res->progress.time_ms   = (monotonic_us - slot.stats.t_start) / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->stats = slot.stats;
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;
        res->trace_id = slot.task->trace_id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        res->stats           = slot.stats;
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = slot.task->params.response_fields;

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        telemetry_finalize(slot, "success");

        // in stream mode, content and tokens are already in last partial chunk
        if (!slot.task->params.stream) {
            res->content = std::move(slot.generated_text);
            res->tokens  = std::move(slot.generated_tokens);
        }

        const int64_t t_handoff = queue_results.send(std::move(res), true);
        telemetry_on_response_handoff(slot, t_handoff);
    }

    void send_embedding(server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        telemetry_finalize(slot, "success");
        const int64_t t_handoff = queue_results.send(std::move(res), true);
        telemetry_on_response_handoff(slot, t_handoff);
    }

    void send_rerank(server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        telemetry_finalize(slot, "success");
        const int64_t t_handoff = queue_results.send(std::move(res), true);
        telemetry_on_response_handoff(slot, t_handoff);
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files, init_opt);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true, init_opt)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    telemetry_finalize(slot, "error", "failed to launch shared request", "server");
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            parent_task.child_tasks[idx].t_slot_start = parent_task.t_slot_start;
            parent_task.child_tasks[idx].t_cache_start = parent_task.t_cache_start;
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        const int id_task = slot.task->id;

        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task
        int64_t last = -1;
        for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
            if (it->id_task != id_task && last >= 0 && it->n_tokens <= last + params_base.checkpoint_min_step) {
                SLT_TRC(slot, "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        it->pos_min, it->pos_max, it->n_tokens, (float) it->size() / 1024 / 1024);

                it = slot.prompt.checkpoints.erase(it);
                continue;
            }

            last = it->n_tokens;
            ++it;
        }

        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        auto & cur = slot.prompt.checkpoints.emplace_back();

        cur.id_task = id_task;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        cur.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        cur.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        // stash the draft's speculative state with the checkpoint
        common_speculative_get_state(spec.get(), slot.id, cur.data_spec);

        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
    }

    // returns false to decline the task, it is offered again after the decode is done
    bool process_single_task(server_task && task, bool is_yielding) {
        // while yielding, an encode / decode is running and only reading the server state is safe
        if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET) {
            SRV_DBG("decoding, decline task, id_task = %d\n", task.id);
            return false;
        }

        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_task = task.id;

                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        task.t_cache_start = 0;
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        task.t_cache_start = 0;
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            task.t_cache_start = 0;
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                if (slot.prompt_save(*prompt_cache)) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                if (params_base.kv_unified) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    if (slot.prompt.n_tokens() > 0) {
                                        const telemetry_kv_eviction_result eviction = clear_idle_slot(slot);
                                        telemetry_kv_record_eviction(
                                            eviction,
                                            "idle_cache_policy",
                                            "idle cached state moved to the RAM prompt cache",
                                            false);
                                    } else {
                                        slot.prompt_clear();
                                    }
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            telemetry_finalize(slot, "cancelled");
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %s on unknown completion id=%s, no live slot\n",
                                task.params.control_action.c_str(), task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        if (slot.is_processing()) {
                            n_processing_slots++;
                        }
                    }
                    SRV_DBG("n_processing_slots = %d\n", n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    // Include the authoritative core physical-ubatch counters. They live on
                    // llama_context and are merged into the server counters on read.
                    res->metrics             = get_metrics();

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_TELEMETRY_SNAPSHOT:
            case SERVER_TASK_TYPE_TELEMETRY_EVENTS:
            case SERVER_TASK_TYPE_TELEMETRY_KV:
            case SERVER_TASK_TYPE_TELEMETRY_KV_PRESSURE:
            case SERVER_TASK_TYPE_TELEMETRY_GPU:
            case SERVER_TASK_TYPE_TELEMETRY_TOKEN_CANDIDATES:
                {
                    auto res = std::make_unique<server_task_result_telemetry>();
                    res->id = task.id;
                    if (task.type == SERVER_TASK_TYPE_TELEMETRY_SNAPSHOT) {
                        res->data = telemetry_snapshot_json();
                    } else if (task.type == SERVER_TASK_TYPE_TELEMETRY_EVENTS) {
                        res->serialized_data = telemetry_events_json(task.telemetry_cursor, task.telemetry_limit);
                        res->has_serialized_data = true;
                    } else if (task.type == SERVER_TASK_TYPE_TELEMETRY_KV) {
                        res->data = telemetry_kv_json(task.telemetry_deep_detail);
                    } else if (task.type == SERVER_TASK_TYPE_TELEMETRY_KV_PRESSURE) {
                        res->serialized_data = telemetry_kv_pressure_events_json(
                            task.telemetry_cursor,
                            task.telemetry_limit,
                            task.telemetry_trace_id);
                        res->has_serialized_data = true;
                    } else if (task.type == SERVER_TASK_TYPE_TELEMETRY_TOKEN_CANDIDATES) {
                        res->data = telemetry_token_candidates_json(task.telemetry_trace_id);
                    } else {
                        res->data = gpu_telemetry.snapshot_json(
                            telemetry_server_instance_id,
                            task.telemetry_cursor,
                            task.telemetry_limit,
                            task.telemetry_trace_id);
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_GET:
                {
                    json slots_data = json::array();

                    int n_idle_slots = 0;

                    for (server_slot & slot : slots) {
                        if (!slot.is_processing()) {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot.to_json(slots_debug == 0));
                    }
                    SRV_DBG("n_idle_slots = %d\n", n_idle_slots);

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id           = task.id;
                    res->slots_data   = std::move(slots_data);
                    res->n_idle_slots = n_idle_slots;

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    std::vector<char> packed;
                    try {
                        packed = slot->prompt.tokens.serialize();
                    } catch (const std::exception & err) {
                        send_error(task, err.what(), ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);
                    const size_t nwrite = llama_state_seq_save_file(
                        ctx_tgt, filepath.c_str(), slot->id,
                        reinterpret_cast<const llama_token *>(packed.data()), packed.size() / sizeof(llama_token));
                    if (nwrite == 0) {
                        send_error(task, "Unable to save slot", ERROR_TYPE_SERVER);
                        break;
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    size_t nread = 0;
                    try {
                        size_t n_packed = 0;
                        llama_tokens packed;
                        nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, nullptr, 0, &n_packed);
                        if (nread != 0) {
                            packed.resize(std::max<size_t>(1, n_packed));
                            nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, packed.data(), packed.size(), &n_packed);
                        }
                        if (nread == 0) {
                            throw std::runtime_error("No available space in KV cache or invalid slot save file");
                        }
                        packed.resize(n_packed);

                        server_tokens restored = server_tokens::deserialize(packed, mctx != nullptr);

                        if (restored.size() > (size_t) slot->n_ctx) {
                            throw std::runtime_error("Restored prompt does not fit in the slot context");
                        }

                        if (!restored.validate(ctx_tgt)) {
                            throw std::runtime_error("Invalid tokens in slot save file");
                        }

                        slot->prompt.clear();
                        slot->prompt.tokens = std::move(restored);
                    } catch (const std::exception & err) {
                        slot->prompt_clear();
                        send_error(task, std::string("Unable to restore slot: ") + err.what(), ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear();

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }

        return true;
    }

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    void update_slots() {
        telemetry_apply_micro_controls();
#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");

                metrics_flush_idle();

                return; // skip further processing

            } else {
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode();
            batch.render();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));

            // the batch is half-built and not rendered, skip now to avoid UB
            return;
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        llama_batch batch_view;
        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch_view = batch.get_view(off, n_tokens);
                bool ok = decode(n_batch, off, batch_view);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off, batch_view);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }
    }

    void pre_decode() {
        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                if (telemetry_kv_pressure_active) {
                    const llama_memory_diagnostics diagnostics_before = llama_get_memory_diagnostics(ctx_tgt);
                    slot.mem.seq_rm (slot.id, n_keep            , n_keep + n_discard);
                    slot.mem.seq_add(slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);
                    const llama_memory_diagnostics diagnostics_after = llama_get_memory_diagnostics(ctx_tgt);
                    const uint64_t shifted_entries = diagnostics_after.churn.shifted_entries >= diagnostics_before.churn.shifted_entries
                        ? diagnostics_after.churn.shifted_entries - diagnostics_before.churn.shifted_entries
                        : 0;
                    telemetry_kv_pressure_append({
                        {"kind", "context_shift"},
                        {"trace_id", slot.task->trace_id},
                        {"task_id", slot.task->id},
                        {"slot_id", slot.id},
                        {"role", "target"},
                        {"discarded_tokens", n_discard},
                        {"position_delta", -n_discard},
                        {"shifted_entries", shifted_entries},
                    });
                    telemetry_kv_pressure_sample(ggml_time_us(), true);
                } else {
                    slot.mem.seq_rm (slot.id, n_keep            , n_keep + n_discard);
                    slot.mem.seq_add(slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);
                }

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        });

        // start populating the batch for this iteration
        batch.clear();

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // determine which slots are generating and drafting
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .n_past   = */ slot.prompt.n_tokens(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        });

        // generate the actual drafts (if any)
        if (!drafting.empty()) {
            const int64_t draft_started_us = ggml_time_us();
            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
            const int64_t draft_completed_us = ggml_time_us();
            for (const server_slot * slot : drafting) {
                for (size_t proposal = 0; proposal < slot->spec_draft.size(); ++proposal) {
                    gpu_telemetry.record_operation(
                        SERVER_GPU_OPERATION_MTP_DRAFT,
                        slot->task->trace_id,
                        slot->id,
                        draft_started_us,
                        draft_completed_us,
                        slot->prompt.n_tokens() + (int64_t) proposal,
                        -1,
                        slot->stats.n_draft_verif_steps,
                        slot->n_spec_target_passes,
                        proposal,
                        SERVER_GPU_TIMING_SYNCHRONIZED_WINDOW);
                }
            }
        }

        // make checkpoints if needed
        iterate(drafting, [&](server_slot & slot) {
            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.stats.n_draft_tokens += draft.size();

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }

                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1)) {
                    GGML_ABORT("failed to remove sequence %d\n", slot.id);
                }
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        });

        // update the batch with the sampled/drafted tokens
        iterate(generating, [&](server_slot & slot) {
            slot.handle_last_sampled_token(batch);
        });

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            iterate(slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }

                if (!slot.is_processing()) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    return;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.stats.update_prompt_start();

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        slot.prompt_probability.reset();
                        if (slot.task->params.prompt_perplexity) {
                            bool has_multimodal_token = false;
                            for (size_t i = 0; i < input_tokens.size(); ++i) {
                                has_multimodal_token |= input_tokens[i] == LLAMA_TOKEN_NULL;
                            }
                            if (has_multimodal_token) {
                                slot.prompt_probability.unavailable_reason = "multimodal_prompt_scoring_not_supported";
                            } else if (input_tokens.size() < 2) {
                                slot.prompt_probability.unavailable_reason = "prompt_requires_at_least_two_text_tokens";
                            }
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            if (slot.task->params.cache_prompt && !slot.should_score_prompt()) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            slot.mem.seq_rm (slot.id, head_p, head_c);
                                            slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - slots_n_diff, 0);
                                    const int np1 = std::min<int>(n_past + slots_n_diff + 2, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
                                            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                            if (cur.pos_max > pos_next) {
                                                return false;
                                            }
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        it->load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        it->load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        // restore the draft's speculative state
                                        common_speculative_set_state(spec.get(), slot.id, it->data_spec);

                                        pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                        SLT_TRC(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) it->size() / 1024 / 1024);
                                    }

                                    if (do_reset) {
                                        SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        slot.stats.n_prompt_matched = n_past;
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.stats.n_prompt_cached    = n_past;
                        slot.stats.n_prompt_processed = 0;
                        slot.stats.t_cache_last = ggml_time_us();

                        metrics.add_prompt_cached(n_past);

                        slot.prompt.tokens.keep_first(n_past);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    // note: the prompt timing is advanced in post_decode(), so it does not cover
                    //       the tokens added to the batch below
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    slot.mem.seq_rm(slot.id, p0, -1);

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    bool has_mtmd = false;

                    // check if we should process the mtmd chunk
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // process the mtmd chunk
                        // note: it submits its own decode, potentially be async
                        //       so the timing is queued and flushed on the next sync
                        if (slot.stats.t_prefill_start == 0) {
                            slot.stats.t_prefill_start = ggml_time_us();
                        }
                        metrics_pre_decode();

                        // encode on the worker thread, so we can still handle metrics tasks
                        size_t n_tokens_out = 0;
                        int32_t res = 0;
                        queue_tasks.yield_to_queue([&]() {
                            res = process_mtmd_chunk(slot, slot.mbatch, cur_token_idx, n_tokens_out);
                        });

                        if (res != 0) {
                            SLT_ERR(slot, "failed to process mtmd chunk, res = %d\n", res);
                            send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                            slot.release();
                            return; // the slot is done, skip it entirely
                        }

                        metrics_queue_prompt(n_tokens_out);
                        slot.stats.n_prompt_processed += n_tokens_out;
                        slot.stats.update_prompt_last();
                        slot.stats.t_prefill_last = slot.stats.t_prompt_last;

                        // add the mtmd chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            // the chunk is already in the KV cache at this point, so we don't need to keep its data around
                            slot.prompt.tokens.push_back_placeholder(chunk.get());
                        }

                        has_mtmd = true;
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {
                        if (slot.stats.t_prefill_start == 0) {
                            slot.stats.t_prefill_start = ggml_time_us();
                        }
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        add_ok &= batch.add(slot.id,
                            cur_tok,
                            /* pos       = */ slot.prompt.tokens.pos_next(),
                            /* output    = */ slot.need_embd() || slot.should_score_prompt(),
                            /* is_prompt = */ true);
                        slot.prompt.tokens.push_back(cur_tok);

                        // Prompt perplexity is an explicitly expensive diagnostic. Keep at most one
                        // scored row per sequence in a logical decode so it stays within the context's
                        // normal n_outputs_max_per_seq allocation instead of reserving n_batch*vocab
                        // logits for every server request, including requests which do not enable it.
                        if (slot.should_score_prompt()) {
                            break;
                        }

                        // break at the last user message, or at user messages at least min step past the last checkpoint
                        if (do_checkpoint && spans.is_user_start(slot.prompt.n_tokens())) {
                            const auto pos = slot.prompt.n_tokens();
                            const auto & checkpoints = slot.prompt.checkpoints;

                            if (pos == last_user_pos || checkpoints.empty() || pos > checkpoints.back().n_tokens + params_base.checkpoint_min_step) {
                                break;
                            }
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    const bool is_user_start = spans.is_user_start(n_tokens_start);
                    const bool is_last_user_message = n_tokens_start == last_user_pos;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.stats.n_gen = 0;
                        slot.i_batch     = batch.size() - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                        // message or we are near the end of the prompt
                        if (!is_user_start && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end ||
                            n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    bool decode(int32_t & n_batch, int32_t off, llama_batch & batch_view) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        metrics_pre_decode();

        if (batch.size() == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        if (spec && batch.has_embd) {
            if (llama_model_n_embd_inp(model_dft) != llama_model_n_embd_inp(model_tgt)) {
                SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
                throw std::runtime_error("unsupported batch.has_embd + spec case");
            }
        }

        const telemetry_control_state control = telemetry_control_current();
        bool has_output = false;
        bool collect_moe_routing = false;
        for (int i = off; i < off + batch_view.n_tokens; ++i) {
            has_output |= batch.tokens[i].output;
            server_slot & slot = slots[batch.tokens[i].id_slot];
            if (llama_model_n_expert(model_tgt) > 0 && slot.task &&
                    slot.task->params.moe_routing_telemetry_permitted &&
                    !control.moe_routing && slot.telemetry_moe_chunk_capture_started) {
                if (!slot.telemetry_moe_chunk_capture_interrupted) {
                    slot.telemetry_moe_chunk_capture_interrupted = true;
                    ++slot.telemetry_moe_chunk_unlocated_pending;
                }
            }
            collect_moe_routing |= control.moe_routing
                && llama_model_n_expert(model_tgt) > 0
                && slot.task
                && slot.task->params.moe_routing_telemetry_permitted;
        }

        // yield to the queue, so we can still handle metrics tasks while decoding
        // note: the sync is done here too, so that the wait is also covered by the yield
        int ret = 0;
        telemetry_moe_routing_readback_capture moe_routing_readback;
        bool moe_routing_readback_copied = false;
        queue_tasks.yield_to_queue([&]() {
            llama_set_moe_routing(ctx_tgt, collect_moe_routing);
            ret = llama_decode(ctx_tgt, batch_view);
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
            if (ret == 0 && collect_moe_routing) {
                moe_routing_readback_copied = telemetry_copy_moe_routing_readback(moe_routing_readback);
            }
        });
        const int64_t decode_completed_us = ggml_time_us();

        if (ret == 0 && collect_moe_routing) {
            const bool readback_has_rows = moe_routing_readback_copied && !moe_routing_readback.rows.empty();
            for (int i = off; i < off + batch_view.n_tokens; ++i) {
                server_slot & slot = slots[batch.tokens[i].id_slot];
                if (!telemetry_moe_request_enabled(slot)) {
                    continue;
                }
                slot.telemetry_moe_chunk_capture_started = true;
                if (!readback_has_rows) {
                    slot.telemetry_moe_chunk_source_unavailable = true;
                    ++slot.telemetry_moe_chunk_unlocated_pending;
                }
            }
            if (moe_routing_readback_copied) {
                telemetry_record_moe_routing_chunks(
                    moe_routing_readback,
                    off,
                    batch_view.n_tokens,
                    control.generation);
                telemetry_record_moe_routing(moe_routing_readback, off, batch_view.n_tokens);
            }
        }

        if (ret == 0 && gpu_telemetry.is_collecting()) {
            std::fill(gpu_verify_proposal_positions.begin(), gpu_verify_proposal_positions.end(), 0);
            for (int i = off; i < off + batch_view.n_tokens; ++i) {
                const auto & token = batch.tokens[i];
                const server_slot & slot = slots[token.id_slot];
                const bool mtp_verify = !token.is_prompt && slot.can_speculate() && !slot.spec_draft.empty();
                const server_gpu_operation_kind kind = token.is_prompt
                    ? SERVER_GPU_OPERATION_PREFILL
                    : mtp_verify ? SERVER_GPU_OPERATION_MTP_VERIFY : SERVER_GPU_OPERATION_NORMAL_DECODE;
                gpu_telemetry.record_operation(
                    kind,
                    slot.task ? slot.task->trace_id : std::string(),
                    slot.id,
                    t_decode_start,
                    decode_completed_us,
                    token.pos,
                    token.is_prompt || mtp_verify ? -1 : slot.stats.n_gen,
                    mtp_verify ? (int64_t) slot.stats.n_draft_verif_steps : -1,
                    mtp_verify ? (int64_t) slot.n_spec_target_passes : -1,
                    mtp_verify ? gpu_verify_proposal_positions[token.id_slot]++ : -1,
                    has_output ? SERVER_GPU_TIMING_SYNCHRONIZED_WINDOW : SERVER_GPU_TIMING_SUBMISSION_WINDOW);
            }
        }

        if (ret == 1 && telemetry_kv_pressure_active) {
            telemetry_kv_wait_begin(off, batch_view.n_tokens, n_batch);
            telemetry_kv_pressure_sample(decode_completed_us, true);
        }

        if (ret != 0) {
            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (!err.empty()) {
                    if (telemetry_kv_pressure_active && telemetry_kv_wait.active()) {
                        if (ret == 1) {
                            telemetry_kv_wait_retry("terminal", n_batch, 0, decode_completed_us);
                            telemetry_kv_wait_finish("context_exhausted", decode_completed_us);
                        } else {
                            telemetry_kv_wait_finish("failed", decode_completed_us);
                        }
                    }
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.prompt_clear();
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            if (ret == 1) {
                const int32_t attempted_batch_size = n_batch;
                const telemetry_kv_eviction_result eviction = try_clear_idle_slots();
                if (eviction.cleared) {
                    if (telemetry_kv_pressure_active) {
                        telemetry_kv_wait_retry(
                            "emergency_idle_slot_eviction",
                            attempted_batch_size,
                            attempted_batch_size,
                            decode_completed_us);
                        telemetry_kv_record_eviction(
                            eviction,
                            "decode_pressure_recovery",
                            "idle cached state cleared after llama_decode returned KV-slot unavailable",
                            true);
                    }
                } else {
                    n_batch /= 2;
                    if (telemetry_kv_pressure_active) {
                        telemetry_kv_wait_retry("batch_halved", attempted_batch_size, n_batch, decode_completed_us);
                    }
                }
            } else {
                if (telemetry_kv_pressure_active && telemetry_kv_wait.active()) {
                    telemetry_kv_wait_finish("failed", decode_completed_us);
                }
                n_batch /= 2;
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        } else {
            if (telemetry_kv_pressure_active) {
                telemetry_kv_wait_finish_resumed(off, batch_view.n_tokens, decode_completed_us);
                telemetry_kv_pressure_sample(decode_completed_us, false);
            }

            // success, apply batch metrics
            metrics_post_decode(off, batch_view.n_tokens, has_output);
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (spec) {
            bool ok = true;
            queue_tasks.yield_to_queue([&]() {
                ok = common_speculative_process(spec.get(), batch_view);
            });

            if (!ok) {
                SRV_ERR("%s", "failed to process speculative batch\n");

                // TODO: handle error
                throw std::runtime_error("failed to process speculative batch");
            }
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                    slot.copy_state_to(*child);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    void post_decode(int32_t n_batch_tokens, int32_t off, llama_batch & batch_view) {
        // for checking if a given batch index is inside batch_view
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
                }
            }
        });

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        iterate(slots, [&](server_slot & slot) {
            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;

                if (slot.can_speculate()) {
                    common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                }
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;
            const llama_pos model_position = batch.tokens[slot.i_batch].pos + 1;

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
            }

            slot.i_batch = -1;

            common_sampler_accept(slot.smpl.get(), id, true);

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.stats.n_gen += 1;
            metrics.n_server_output_tokens++;

            if (slot.stats.n_gen == 1) {
                slot.stats.t_first_token = t_now;
                telemetry_on_first_token(slot);
                slot.stats.update_prompt_last();
                slot.t_print_last = t_now;
                slot.n_gen_last = 0;
            }

            completion_token_output result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            double selected_log_probability_ln = std::numeric_limits<double>::quiet_NaN();

            if (slot.task->params.sampling.n_probs > 0) {
                std::string reason;
                const bool raw_distribution_available = populate_token_probs(
                    slot,
                    result,
                    slot.task->params.post_sampling_probs,
                    params_base.special,
                    tok_idx,
                    selected_log_probability_ln,
                    reason);

                if (slot.response_probability.unavailable_reason.empty()) {
                    if (raw_distribution_available) {
                        slot.response_probability.observe(selected_log_probability_ln);
                    } else if (!slot.task->params.post_sampling_probs) {
                        slot.response_probability.unavailable_reason = std::move(reason);
                    } else {
                        double logprob = 0.0;
                        if (selected_token_log_probability(ctx_tgt, tok_idx, id, logprob, reason)) {
                            slot.response_probability.observe(logprob);
                            selected_log_probability_ln = logprob;
                        } else {
                            slot.response_probability.unavailable_reason = std::move(reason);
                        }
                    }
                }
            }

            telemetry_record_output_token(
                slot,
                id,
                result.text_to_send,
                t_now,
                model_position,
                selected_log_probability_ln,
                TELEMETRY_OUTPUT_TOKEN_ORIGIN_NORMAL_DECODE);

            slot.stats.update_gen_last();

            if (!process_token(result, slot)) {
                // release slot because of stop condition
                slot.print_timings();
                send_final_response(slot);
                slot.release();

                return;
            }

            slot.print_timings_tg();
        });

        // speculative decoding - main model sample and accept
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() ||
                    slot.spec_draft.empty() || slot.spec_i_batch.empty()) {
                return;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();
            const bool replay_pass = slot.spec_is_replay;

            GGML_ASSERT(n_draft > 0);

            // verify and try to accept the draft
            int64_t t_spec_sample = 0;
            std::vector<double> spec_selected_logprobs;
            const auto spec_rows = slot.spec_i_batch;
            {
                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());
                auto accepted = synth_probs.empty()
                    ? common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, spec_rows, slot.spec_draft)
                    : server_sample_and_accept_synth(
                            slot.smpl.get(), slot.ctx_tgt, spec_rows, slot.spec_draft,
                            synth_probs, slot.spec_synth_rng, slot.spec_is_replay);
                slot.spec_i_batch.clear();

                GGML_ASSERT(accepted.size() >= 1);
                t_spec_sample = ggml_time_us();

                // Every visit here follows a successful target-model
                // verification graph. Count replay passes as real forward
                // passes even when their output is discarded.
                metrics.spec_target_tokens_per_pass.observe(n_draft + 1);
                metrics.n_spec_target_tokens += n_draft + 1;
                metrics.n_spec_target_passes++;
                slot.n_spec_target_tokens += n_draft + 1;
                slot.n_spec_target_passes++;

                // Acceptance depth is a logical verification-decision metric.
                // A checkpoint replay completes the same decision and must not
                // create a second proposed/accepted-depth observation.
                if (!slot.spec_is_replay) {
                    const size_t accepted_depth = accepted.size() - 1;
                    if (telemetry_output_token_request_enabled(slot)) {
                        slot.telemetry_spec_logical_step = slot.stats.n_draft_verif_steps;
                        slot.telemetry_spec_proposed_count = n_draft;
                        slot.telemetry_spec_accepted_depth = accepted_depth;
                        telemetry_prepare_mtp_proposals(slot, slot.spec_draft, accepted);
                    }
                    metrics.spec_draft_depth.observe(n_draft);
                    metrics.spec_accepted_depth.observe(accepted_depth);
                    if (accepted_depth > 0) {
                        slot.n_draft_hit_steps++;
                        metrics.n_draft_hit_steps++;
                    }
                    if (accepted_depth == n_draft) {
                        slot.n_draft_full_steps++;
                        metrics.n_draft_full_steps++;
                    }
                }

                const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));

                // check for partial draft acceptance
                if (n_rollback > 0) {
                    if (use_ckpt_tgt) {
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                        }

                        // partial acceptance is not supported by the context -> truncate the draft and restore the state
                        slot.spec_is_replay = true;
                        slot.spec_draft = std::move(accepted);

                        const auto & ckpt = slot.spec_ckpt;

                        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                        ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                        if (slot.ctx_dft) {
                            ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.mem.seq_rm(slot.id, ckpt.pos_max + 1, -1);

                        slot.prompt.tokens.keep_first(ckpt.n_tokens);
                        common_sampler_copy(smpl_save.get(), slot.smpl.get());

                        // This expensive target pass committed no output. The
                        // replay pass will receive its own target/useful entry.
                        metrics.spec_useful_tokens_per_pass.observe(0);
                        telemetry_record_mtp_pass(
                            slot,
                            replay_pass,
                            true,
                            n_draft + 1,
                            slot.stats.n_gen,
                            0);
                        if (telemetry_output_token_request_enabled(slot)) {
                            slot.stats.update_gen_last();
                        } else {
                            slot.stats.t_gen_last = std::max(slot.stats.t_prompt_last, t_spec_sample);
                        }

                        return;
                    }
                }

                if (trace > 0) {
                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                }

                common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

                slot.spec_draft = std::move(accepted);

                const bool response_probability_enabled = slot.task->params.sampling.n_probs > 0
                    && slot.response_probability.unavailable_reason.empty();
                spec_selected_logprobs.clear();
                if (response_probability_enabled) {
                    spec_selected_logprobs.assign(slot.spec_draft.size(), std::numeric_limits<double>::quiet_NaN());
                }
                for (size_t i = 0; i < slot.spec_draft.size(); ++i) {
                    const bool candidate_enabled = telemetry_token_candidate_position_enabled(
                        slot,
                        i,
                        slot.telemetry_spec_accepted_depth,
                        slot.telemetry_spec_proposed_count);
                    if (!response_probability_enabled && !candidate_enabled) {
                        continue;
                    }

                    double logprob = 0.0;
                    std::string reason;
                    std::vector<telemetry_token_candidate_value> candidates;
                    const size_t top_k = candidate_enabled
                        ? (size_t) slot.task->params.output_token_candidate_top_k
                        : 0;
                    const bool available = raw_target_token_probabilities(
                        ctx_tgt,
                        spec_rows[i],
                        slot.spec_draft[i],
                        top_k,
                        logprob,
                        candidate_enabled ? &candidates : nullptr,
                        nullptr,
                        0,
                        reason);
                    if (available) {
                        if (response_probability_enabled) {
                            slot.response_probability.observe(logprob);
                            spec_selected_logprobs[i] = logprob;
                            if (i < slot.telemetry_mtp_proposals_pending.size() &&
                                    slot.telemetry_mtp_proposals_pending[i].disposition != TELEMETRY_MTP_PROPOSAL_INVALIDATED_TAIL) {
                                slot.telemetry_mtp_proposals_pending[i].target_selected_log_probability_ln = logprob;
                            }
                        }
                        if (candidate_enabled) {
                            telemetry_record_token_candidates(
                                slot,
                                i,
                                slot.telemetry_spec_accepted_depth,
                                slot.telemetry_spec_proposed_count,
                                slot.spec_draft[i],
                                slot.stats.n_gen + i,
                                std::move(candidates),
                                "available",
                                "raw_target_model_pre_sampler_top_k_from_one_full_vocabulary_normalization");
                        }
                    } else {
                        if (response_probability_enabled && slot.response_probability.unavailable_reason.empty()) {
                            slot.response_probability.unavailable_reason = reason;
                        }
                        if (candidate_enabled) {
                            telemetry_record_token_candidates(
                                slot,
                                i,
                                slot.telemetry_spec_accepted_depth,
                                slot.telemetry_spec_proposed_count,
                                slot.spec_draft[i],
                                slot.stats.n_gen + i,
                                {},
                                "unavailable",
                                reason);
                        }
                    }
                }
            }

            const auto ids = std::move(slot.spec_draft);

            size_t n_accepted = ids.size() - 1;
            if (slot.spec_is_replay && n_accepted > 0) {
                n_accepted--;
            }
            slot.spec_is_replay = false;

            if (slot.stats.n_gen == 0) {
                slot.stats.t_first_token = t_spec_sample;
                telemetry_on_first_token(slot);
                slot.stats.set_prompt_last(t_spec_sample);
                slot.t_print_last = t_spec_sample;
                slot.n_gen_last = 0;
            }
            // update how many tokens out of those tested were accepted
            slot.stats.n_draft_accepted += n_accepted;
            slot.stats.n_draft_verif_steps += 1;

            auto & n_accepted_per_pos = slot.n_accepted_per_pos;
            if (n_accepted_per_pos.empty()) {
                n_accepted_per_pos.resize(common_speculative_n_max(spec.get()), 0);
            }
            for (size_t i = 0; i < n_accepted && i < n_accepted_per_pos.size(); ++i) {
                n_accepted_per_pos[i]++;
            }

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);

            const uint64_t committed_output_start_ordinal = slot.stats.n_gen;
            uint64_t useful_tokens_this_pass = 0;
            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                const double selected_log_probability_ln = i < spec_selected_logprobs.size()
                    ? spec_selected_logprobs[i]
                    : std::numeric_limits<double>::quiet_NaN();
                result.prob         = std::isfinite(selected_log_probability_ln)
                    ? (float) std::exp(selected_log_probability_ln)
                    : 1.0f;
                result.logprob      = selected_log_probability_ln;

                // TODO: set result.probs

                slot.stats.n_gen += 1;
                metrics.n_server_output_tokens++;
                slot.n_spec_useful_tokens++;
                metrics.n_spec_useful_tokens++;
                useful_tokens_this_pass++;

                const auto origin = i < slot.telemetry_spec_accepted_depth
                    ? TELEMETRY_OUTPUT_TOKEN_ORIGIN_MTP_ACCEPTED
                    : i == slot.telemetry_spec_accepted_depth &&
                            slot.telemetry_spec_accepted_depth < slot.telemetry_spec_proposed_count
                        ? TELEMETRY_OUTPUT_TOKEN_ORIGIN_TARGET_AFTER_MISS
                        : TELEMETRY_OUTPUT_TOKEN_ORIGIN_TARGET_BONUS;
                telemetry_record_output_token(
                    slot,
                    result.tok,
                    result.text_to_send,
                    t_spec_sample,
                    batch.tokens[spec_rows[i]].pos + 1,
                    selected_log_probability_ln,
                    origin,
                    (int64_t) slot.telemetry_spec_logical_step,
                    slot.n_spec_target_passes > 0 ? (int64_t) slot.n_spec_target_passes - 1 : -1,
                    (int64_t) i,
                    (int64_t) slot.telemetry_spec_accepted_depth,
                    (int64_t) slot.telemetry_spec_proposed_count,
                    replay_pass);

                if (telemetry_output_token_request_enabled(slot) || slot.task->params.sampling.n_probs > 0) {
                    slot.stats.update_gen_last();
                } else {
                    slot.stats.t_gen_last = std::max(slot.stats.t_prompt_last, t_spec_sample);
                }

                if (!process_token(result, slot)) {
                    metrics.spec_useful_tokens_per_pass.observe(useful_tokens_this_pass);
                    telemetry_record_mtp_pass(
                        slot,
                        replay_pass,
                        false,
                        n_draft + 1,
                        committed_output_start_ordinal,
                        useful_tokens_this_pass);
                    slot.print_timings();
                    send_final_response(slot);
                    slot.release();

                    return;
                }
            }

            metrics.spec_useful_tokens_per_pass.observe(useful_tokens_this_pass);
            telemetry_record_mtp_pass(
                slot,
                replay_pass,
                false,
                n_draft + 1,
                committed_output_start_ordinal,
                useful_tokens_this_pass);

            slot.print_timings_tg();

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) n_accepted, (int) n_draft, slot.prompt.n_tokens());
        });
    }

    // context size of a single slot, capped by --kv-unified-per-slot and by the training context of the model
    int n_ctx_slot() const {
        int res = llama_n_ctx_seq(ctx_tgt);

        if (params_base.kv_unified_per_slot > 0) {
            res = std::min(res, params_base.kv_unified_per_slot);
        }

        return std::min(res, llama_model_n_ctx_train(model_tgt));
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }

    void telemetry_append_serialized(uint64_t sequence, std::string serialized) {
        telemetry_event_entry entry;
        entry.sequence = sequence;
        entry.serialized = std::move(serialized);
        entry.bytes = entry.serialized.size();
        const size_t bytes = entry.bytes;
        if (bytes > telemetry_event_max_bytes) {
            telemetry_dropped_events++;
            telemetry_last_dropped_sequence = entry.sequence;
            return;
        }
        telemetry_event_bytes += bytes;
        telemetry_events.push_back(std::move(entry));
        while (telemetry_events.size() > TELEMETRY_EVENT_CAPACITY || telemetry_event_bytes > telemetry_event_max_bytes) {
            telemetry_event_bytes -= telemetry_events.front().bytes;
            telemetry_last_dropped_sequence = telemetry_events.front().sequence;
            telemetry_events.pop_front();
            telemetry_dropped_events++;
        }
    }

    void telemetry_append(json event) {
        event["schema_version"] = 1;
        event["server_instance_id"] = telemetry_server_instance_id;
        const uint64_t sequence = telemetry_next_sequence++;
        event["sequence"] = sequence;
        telemetry_append_serialized(sequence, event.dump());
    }

    std::string telemetry_serialize_with_serialized_bytes(json event, uint64_t sequence) const {
        event["schema_version"] = 1;
        event["server_instance_id"] = telemetry_server_instance_id;
        event["sequence"] = sequence;
        event["serialized_bytes"] = 0;

        std::string serialized = event.dump();
        const std::string marker = "\"serialized_bytes\":0";
        const size_t marker_offset = serialized.find(marker);
        GGML_ASSERT(marker_offset != std::string::npos);

        size_t final_bytes = serialized.size();
        while (true) {
            const size_t next_bytes = serialized.size() - 1 + std::to_string(final_bytes).size();
            if (next_bytes == final_bytes) {
                break;
            }
            final_bytes = next_bytes;
        }
        serialized.replace(marker_offset + marker.size() - 1, 1, std::to_string(final_bytes));
        GGML_ASSERT(serialized.size() == final_bytes);
        return serialized;
    }

    void telemetry_append_with_serialized_bytes(json event) {
        const uint64_t sequence = telemetry_next_sequence++;
        telemetry_append_serialized(sequence, telemetry_serialize_with_serialized_bytes(std::move(event), sequence));
    }

    static int64_t telemetry_wall_unix_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    void telemetry_kv_snapshot_capture(bool include_diagnostics) {
        telemetry_kv_boundary_snapshot snapshot;
        snapshot.memory = llama_get_memory_snapshot(ctx_tgt, include_diagnostics);
        snapshot.breakdown = llama_get_memory_breakdown(ctx_tgt);
        snapshot.monotonic_us = ggml_time_us();
        snapshot.available = true;

        for (const auto & slot : slots) {
            snapshot.resident_slot_tokens += slot.prompt.n_tokens();
            snapshot.represented_slots += slot.prompt.n_tokens() > 0 ? 1 : 0;
            if (!include_diagnostics) {
                continue;
            }

            const auto & tokens = slot.prompt.tokens.get_tokens();
            if (tokens.empty()) {
                continue;
            }
            if (std::find(tokens.begin(), tokens.end(), LLAMA_TOKEN_NULL) != tokens.end()) {
                snapshot.multimodal_sequences_skipped++;
                continue;
            }
            std::ostringstream compatibility;
            compatibility << slot.n_ctx;
            for (const auto & adapter : slot.lora) {
                uint32_t scale_bits = 0;
                static_assert(sizeof(scale_bits) == sizeof(adapter.scale));
                std::memcpy(&scale_bits, &adapter.scale, sizeof(scale_bits));
                compatibility << ':' << reinterpret_cast<uintptr_t>(adapter.ptr) << ':' << scale_bits;
            }
            snapshot.slots.push_back({slot.id, tokens, compatibility.str()});
        }

        telemetry_kv_boundary = std::move(snapshot);
    }

    void telemetry_kv_pressure_initialize() {
        const llama_memory_diagnostics diagnostics = llama_get_memory_diagnostics(ctx_tgt);
        telemetry_kv_primary_component.clear();
        telemetry_kv_primary_memory_kind.clear();
        telemetry_kv_primary_entry_semantics.clear();
        for (const auto & component : diagnostics.components) {
            if (!component.logical_primary) {
                continue;
            }
            telemetry_kv_primary_component = component.name;
            telemetry_kv_primary_memory_kind = component.kind;
            telemetry_kv_primary_entry_semantics = component.entry_semantics;
            break;
        }
    }

    void telemetry_kv_pressure_append(json event) {
        if (!telemetry_kv_pressure_active) {
            return;
        }
        if (!event.contains("timestamp_unix_ms")) {
            event["timestamp_unix_ms"] = telemetry_wall_unix_ms();
        }
        if (!event.contains("monotonic_us")) {
            event["monotonic_us"] = ggml_time_us();
        }
        event["schema_version"] = 1;
        event["server_instance_id"] = telemetry_server_instance_id;
        event["sequence"] = telemetry_kv_pressure_next_sequence++;
        telemetry_kv_pressure_event_entry entry;
        entry.sequence = event.at("sequence").get<uint64_t>();
        entry.kind = event.at("kind").get<std::string>();
        const auto string_field = [&](const char * name) {
            return event.contains(name) && event.at(name).is_string()
                ? event.at(name).get<std::string>() : std::string();
        };
        entry.trace_id = string_field("trace_id");
        entry.victim_trace_id = string_field("victim_trace_id");
        entry.monotonic_us = event.at("monotonic_us").get<int64_t>();
        entry.serialized = event.dump();
        entry.bytes = entry.serialized.size();
        const size_t bytes = entry.bytes;
        if (bytes > telemetry_kv_pressure_event_max_bytes) {
            telemetry_kv_pressure_dropped_events++;
            telemetry_kv_pressure_last_dropped_sequence = entry.sequence;
            return;
        }
        telemetry_kv_pressure_event_bytes += bytes;
        telemetry_kv_pressure_events.push_back(std::move(entry));
        while (telemetry_kv_pressure_events.size() > TELEMETRY_KV_PRESSURE_EVENT_CAPACITY ||
                telemetry_kv_pressure_event_bytes > telemetry_kv_pressure_event_max_bytes) {
            telemetry_kv_pressure_event_bytes -= telemetry_kv_pressure_events.front().bytes;
            telemetry_kv_pressure_last_dropped_sequence = telemetry_kv_pressure_events.front().sequence;
            telemetry_kv_pressure_events.pop_front();
            telemetry_kv_pressure_dropped_events++;
        }
    }

    void telemetry_kv_pressure_sample(int64_t monotonic_us, bool force) {
        if (!telemetry_kv_pressure_active) {
            return;
        }
        if (monotonic_us <= 0) {
            monotonic_us = ggml_time_us();
        }
        if (!force && telemetry_kv_pressure_last_sample_us > 0 &&
                monotonic_us - telemetry_kv_pressure_last_sample_us < telemetry_kv_pressure_sampling_interval_us) {
            return;
        }
        telemetry_kv_pressure_last_sample_us = monotonic_us;

        const llama_memory_primary_occupancy occupancy = llama_get_memory_primary_occupancy(ctx_tgt);
        const bool valid = occupancy.available && occupancy.capacity_entries > 0 &&
            occupancy.used_entries <= occupancy.capacity_entries;
        const char * state = valid ? "available" : occupancy.available ? "no_data" : "unsupported";
        const char * reason = valid
            ? "authoritative lightweight primary memory occupancy"
            : occupancy.available
                ? "the primary memory component has no usable capacity"
                : "the active memory backend does not expose primary occupancy";
        telemetry_kv_pressure_append({
            {"kind", "utilization_sample"},
            {"monotonic_us", monotonic_us},
            {"component", telemetry_kv_primary_component.empty() ? json(nullptr) : json(telemetry_kv_primary_component)},
            {"memory_kind", telemetry_kv_primary_memory_kind.empty() ? json(nullptr) : json(telemetry_kv_primary_memory_kind)},
            {"entry_semantics", telemetry_kv_primary_entry_semantics.empty() ? json(nullptr) : json(telemetry_kv_primary_entry_semantics)},
            {"utilization_state", state},
            {"utilization_reason", reason},
            {"capacity_entries", valid ? json(occupancy.capacity_entries) : json(nullptr)},
            {"used_entries", valid ? json(occupancy.used_entries) : json(nullptr)},
            {"free_entries", valid ? json(occupancy.capacity_entries - occupancy.used_entries) : json(nullptr)},
            {"utilization", valid ? json((double) occupancy.used_entries / occupancy.capacity_entries) : json(nullptr)},
        });
    }

    void telemetry_kv_request_started(const server_slot & slot) {
        if (!telemetry_kv_pressure_active) {
            return;
        }
        GGML_ASSERT(slot.task);
        const std::string & trace_id = slot.task->trace_id;
        for (auto it = telemetry_kv_request_windows.begin(); it != telemetry_kv_request_windows.end();) {
            if (it->trace_id == trace_id) {
                it = telemetry_kv_request_windows.erase(it);
            } else {
                ++it;
            }
        }
        telemetry_kv_request_windows.push_back({
            trace_id,
            slot.stats.t_arrival > 0 ? slot.stats.t_arrival : ggml_time_us(),
            0,
        });
        while (telemetry_kv_request_windows.size() > telemetry_kv_request_window_capacity) {
            telemetry_kv_request_windows.pop_front();
        }
        telemetry_kv_pressure_sample(ggml_time_us(), true);
    }

    void telemetry_kv_request_finished(const server_slot & slot) {
        if (!telemetry_kv_pressure_active) {
            return;
        }
        GGML_ASSERT(slot.task);
        for (auto it = telemetry_kv_request_windows.rbegin(); it != telemetry_kv_request_windows.rend(); ++it) {
            if (it->trace_id == slot.task->trace_id) {
                it->end_monotonic_us = slot.stats.t_finalization_start;
                break;
            }
        }
        telemetry_kv_pressure_sample(slot.stats.t_finalization_start, true);
    }

    std::vector<telemetry_kv_wait_identity> telemetry_kv_wait_identities(int32_t off, int32_t n_tokens) const {
        std::vector<telemetry_kv_wait_identity> result;
        for (int32_t i = off; i < off + n_tokens; ++i) {
            const int32_t slot_id = batch.tokens[i].id_slot;
            const server_slot & slot = slots[slot_id];
            if (!slot.task) {
                continue;
            }
            auto identity = std::find_if(result.begin(), result.end(), [&](const auto & candidate) {
                return candidate.trace_id == slot.task->trace_id &&
                    candidate.task_id == slot.task->id && candidate.slot_id == slot.id;
            });
            if (identity == result.end()) {
                result.push_back({slot.task->trace_id, slot.task->id, slot.id, 0, {}});
                identity = std::prev(result.end());
            }
            identity->pending_batch_indices.insert(i);
        }
        return result;
    }

    static void telemetry_kv_add_identity(json & event, const telemetry_kv_wait_identity & identity) {
        event["trace_id"] = identity.trace_id;
        event["task_id"] = identity.task_id;
        event["slot_id"] = identity.slot_id;
        event["role"] = "target";
    }

    static bool telemetry_kv_same_identity(
            const telemetry_kv_wait_identity & lhs,
            const telemetry_kv_wait_identity & rhs) {
        return lhs.trace_id == rhs.trace_id && lhs.task_id == rhs.task_id && lhs.slot_id == rhs.slot_id;
    }

    void telemetry_kv_wait_finish_identity(
            const telemetry_kv_wait_identity & identity,
            const char * outcome,
            int64_t completed_monotonic_us) {
        const int64_t started_monotonic_us = identity.started_monotonic_us > 0
            ? identity.started_monotonic_us
            : telemetry_kv_wait.started_monotonic_us;
        const int64_t duration_us = std::max<int64_t>(
            0, completed_monotonic_us - started_monotonic_us);
        json event = {
            {"kind", "decode_wait_finished"},
            {"monotonic_us", completed_monotonic_us},
            {"episode_id", telemetry_kv_wait.id},
            {"wait_duration_us", duration_us},
            {"outcome", outcome},
        };
        telemetry_kv_add_identity(event, identity);
        telemetry_kv_pressure_append(std::move(event));
    }

    void telemetry_kv_wait_begin(int32_t off, int32_t n_tokens, int32_t attempted_batch_size) {
        if (!telemetry_kv_pressure_active) {
            telemetry_kv_wait.clear();
            return;
        }
        const int64_t started_monotonic_us = t_decode_start > 0 ? t_decode_start : ggml_time_us();
        if (!telemetry_kv_wait.active()) {
            telemetry_kv_wait.id = "kv-wait-" + std::to_string(telemetry_kv_pressure_next_episode++);
            telemetry_kv_wait.started_monotonic_us = started_monotonic_us;
        }
        std::vector<telemetry_kv_wait_identity> failed = telemetry_kv_wait_identities(off, n_tokens);
        for (auto & identity : failed) {
            auto active = std::find_if(
                telemetry_kv_wait.identities.begin(),
                telemetry_kv_wait.identities.end(),
                [&](const auto & candidate) { return telemetry_kv_same_identity(candidate, identity); });
            if (active != telemetry_kv_wait.identities.end()) {
                active->pending_batch_indices.insert(
                    identity.pending_batch_indices.begin(),
                    identity.pending_batch_indices.end());
                continue;
            }
            identity.started_monotonic_us = started_monotonic_us;
            telemetry_kv_wait.identities.push_back(std::move(identity));
            const auto & added = telemetry_kv_wait.identities.back();
            json event = {
                {"kind", "decode_wait_started"},
                {"monotonic_us", added.started_monotonic_us},
                {"episode_id", telemetry_kv_wait.id},
                {"attempted_batch_size", attempted_batch_size},
            };
            telemetry_kv_add_identity(event, added);
            telemetry_kv_pressure_append(std::move(event));
        }
        if (telemetry_kv_wait.identities.empty()) {
            telemetry_kv_wait.clear();
        }
    }

    void telemetry_kv_wait_retry(
            const char * action,
            int32_t attempted_batch_size,
            int32_t next_batch_size,
            int64_t monotonic_us) {
        if (!telemetry_kv_pressure_active) {
            telemetry_kv_wait.clear();
            return;
        }
        if (!telemetry_kv_wait.active()) {
            return;
        }
        telemetry_kv_wait.retry_count++;
        for (const auto & identity : telemetry_kv_wait.identities) {
            json event = {
                {"kind", "decode_retry"},
                {"monotonic_us", monotonic_us},
                {"episode_id", telemetry_kv_wait.id},
                {"retry_count", telemetry_kv_wait.retry_count},
                {"action", action},
                {"attempted_batch_size", attempted_batch_size},
                {"next_batch_size", next_batch_size},
            };
            telemetry_kv_add_identity(event, identity);
            telemetry_kv_pressure_append(std::move(event));
        }
    }

    void telemetry_kv_wait_finish(const char * outcome, int64_t completed_monotonic_us) {
        if (!telemetry_kv_pressure_active) {
            telemetry_kv_wait.clear();
            return;
        }
        if (!telemetry_kv_wait.active()) {
            return;
        }
        for (const auto & identity : telemetry_kv_wait.identities) {
            telemetry_kv_wait_finish_identity(identity, outcome, completed_monotonic_us);
        }
        telemetry_kv_wait.clear();
    }

    void telemetry_kv_wait_finish_resumed(
            int32_t off,
            int32_t n_tokens,
            int64_t completed_monotonic_us) {
        if (!telemetry_kv_pressure_active) {
            telemetry_kv_wait.clear();
            return;
        }
        if (!telemetry_kv_wait.active()) {
            return;
        }
        const std::vector<telemetry_kv_wait_identity> completed = telemetry_kv_wait_identities(off, n_tokens);
        for (auto it = telemetry_kv_wait.identities.begin(); it != telemetry_kv_wait.identities.end();) {
            const auto matched = std::find_if(completed.begin(), completed.end(), [&](const auto & identity) {
                return telemetry_kv_same_identity(*it, identity);
            });
            if (matched == completed.end()) {
                ++it;
                continue;
            }
            for (const int32_t index : matched->pending_batch_indices) {
                it->pending_batch_indices.erase(index);
            }
            if (!it->pending_batch_indices.empty()) {
                ++it;
                continue;
            }
            telemetry_kv_wait_finish_identity(*it, "resumed", completed_monotonic_us);
            it = telemetry_kv_wait.identities.erase(it);
        }
        if (telemetry_kv_wait.identities.empty()) {
            telemetry_kv_wait.clear();
        }
    }

    void telemetry_kv_wait_finish_request(
            const server_slot & slot,
            const char * outcome,
            int64_t completed_monotonic_us) {
        if (!telemetry_kv_pressure_active) {
            telemetry_kv_wait.clear();
            return;
        }
        if (!telemetry_kv_wait.active() || !slot.task) {
            return;
        }
        const telemetry_kv_wait_identity request_identity = {
            slot.task->trace_id,
            slot.task->id,
            slot.id,
            0,
            {},
        };
        for (auto it = telemetry_kv_wait.identities.begin(); it != telemetry_kv_wait.identities.end();) {
            if (!telemetry_kv_same_identity(*it, request_identity)) {
                ++it;
                continue;
            }
            telemetry_kv_wait_finish_identity(*it, outcome, completed_monotonic_us);
            it = telemetry_kv_wait.identities.erase(it);
        }
        if (telemetry_kv_wait.identities.empty()) {
            telemetry_kv_wait.clear();
        }
    }

    void telemetry_kv_record_eviction(
            const telemetry_kv_eviction_result & eviction,
            const char * cause,
            const char * reason,
            bool attach_wait) {
        if (!telemetry_kv_pressure_active || !eviction.cleared) {
            return;
        }
        json event = {
            {"kind", "idle_slot_evicted"},
            {"cause", cause},
            {"eviction_reason", reason},
            {"victim_slot_id", eviction.victim_slot_id},
            {"victim_trace_id", eviction.victim_trace_id.empty() ? json(nullptr) : json(eviction.victim_trace_id)},
            {"victim_prompt_tokens", eviction.victim_prompt_tokens},
            {"released_entries_state", eviction.released_entries_available ? "available" : "unavailable"},
            {"released_entries_reason", eviction.released_entries_available
                ? "before/after primary physical occupancy delta"
                : "primary physical occupancy was unavailable"},
            {"released_entries", eviction.released_entries_available ? json(eviction.released_entries) : json(nullptr)},
            {"memberships_removed", eviction.memberships_removed},
        };
        if (attach_wait && telemetry_kv_wait.active()) {
            event["episode_id"] = telemetry_kv_wait.id;
            if (!telemetry_kv_wait.identities.empty()) {
                telemetry_kv_add_identity(event, telemetry_kv_wait.identities.front());
            }
        }
        telemetry_kv_pressure_append(std::move(event));
        telemetry_kv_pressure_sample(ggml_time_us(), true);
    }

    bool raw_target_token_probabilities(
            llama_context * ctx,
            int idx,
            llama_token selected,
            size_t top_k,
            double & logprob,
            std::vector<telemetry_token_candidate_value> * candidates,
            std::vector<llama_token_data> * top_probabilities,
            size_t n_top_probabilities,
            std::string & unavailable_reason) const {
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        if (selected < 0 || selected >= n_vocab) {
            unavailable_reason = "selected_token_out_of_vocabulary";
            return false;
        }

        const float * logits = llama_get_logits_ith(ctx, idx);
        if (logits == nullptr) {
            unavailable_reason = "target_logits_unavailable";
            return false;
        }

        // Backend-sampled subsets are not the full raw-model distribution.
        if (llama_get_sampled_candidates_count_ith(ctx, idx) > 0 ||
                llama_get_sampled_logits_count_ith(ctx, idx) != n_vocab) {
            unavailable_reason = "full_raw_target_logits_unavailable_with_backend_sampling";
            return false;
        }

        double max_logit = -std::numeric_limits<double>::infinity();
        std::vector<std::pair<double, llama_token>> top_logits;
        top_logits.reserve(std::min<size_t>(top_k, (size_t) n_vocab));
        if (top_probabilities != nullptr) {
            top_probabilities->clear();
            top_probabilities->reserve(n_vocab);
        }
        for (int32_t i = 0; i < n_vocab; ++i) {
            if (std::isnan(logits[i])) {
                unavailable_reason = "target_logits_contain_nan";
                return false;
            }
            max_logit = std::max(max_logit, (double) logits[i]);
            if (top_k > 0) {
                const auto value = std::make_pair((double) logits[i], (llama_token) i);
                auto insertion = std::lower_bound(
                    top_logits.begin(),
                    top_logits.end(),
                    value,
                    [](const auto & lhs, const auto & rhs) {
                        return lhs.first > rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second);
                    });
                if (insertion != top_logits.end() || top_logits.size() < top_k) {
                    top_logits.insert(insertion, value);
                    if (top_logits.size() > top_k) {
                        top_logits.pop_back();
                    }
                }
            }
            if (top_probabilities != nullptr) {
                top_probabilities->push_back({(llama_token) i, logits[i], 0.0f});
            }
        }
        if (!std::isfinite(max_logit) || !std::isfinite(logits[selected])) {
            unavailable_reason = "selected_or_max_target_logit_not_finite";
            return false;
        }

        double sum_exp = 0.0;
        for (int32_t i = 0; i < n_vocab; ++i) {
            sum_exp += std::exp((double) logits[i] - max_logit);
        }
        if (!(sum_exp > 0.0) || !std::isfinite(sum_exp)) {
            unavailable_reason = "target_logsumexp_not_finite";
            return false;
        }

        const double log_sum_exp = std::log(sum_exp);
        logprob = (double) logits[selected] - max_logit - log_sum_exp;
        if (!std::isfinite(logprob)) {
            unavailable_reason = "selected_token_log_probability_not_finite";
            return false;
        }
        if (top_probabilities != nullptr) {
            n_top_probabilities = std::min(n_top_probabilities, top_probabilities->size());
            if (n_top_probabilities > 0) {
                std::partial_sort(
                    top_probabilities->begin(),
                    top_probabilities->begin() + n_top_probabilities,
                    top_probabilities->end(),
                    [](const llama_token_data & lhs, const llama_token_data & rhs) {
                        return lhs.logit > rhs.logit;
                    });
            }
            top_probabilities->resize(n_top_probabilities);
            for (auto & probability : *top_probabilities) {
                probability.p = (float) std::exp((double) probability.logit - max_logit - log_sum_exp);
            }
        }
        if (candidates != nullptr) {
            candidates->clear();
            candidates->reserve(top_logits.size());
            for (const auto & item : top_logits) {
                telemetry_token_candidate_value candidate;
                candidate.token_id = item.second;
                candidate.log_probability_ln = item.first - max_logit - log_sum_exp;
                candidate.target_selected = item.second == selected;
                candidates->push_back(candidate);
            }
        }
        return true;
    }

    bool selected_token_log_probability(
            llama_context * ctx,
            int idx,
            llama_token selected,
            double & logprob,
            std::string & unavailable_reason) const {
        return raw_target_token_probabilities(
            ctx,
            idx,
            selected,
            0,
            logprob,
            nullptr,
            nullptr,
            0,
            unavailable_reason);
    }

    int64_t telemetry_unix_ms(const server_slot & slot, int64_t t_us) const {
        if (slot.stats.t_arrival_unix_ms == 0 || slot.stats.t_arrival == 0 || t_us == 0) {
            return 0;
        }
        return slot.stats.t_arrival_unix_ms + (t_us - slot.stats.t_arrival) / 1000;
    }

    json telemetry_server_configuration() const {
        return {
            {"parallel_slots", params_base.n_parallel},
            {"logical_batch_size", params_base.n_batch},
            {"physical_ubatch_size", params_base.n_ubatch},
            {"context_size", params_base.n_ctx},
            {"continuous_batching", params_base.cont_batching},
            {"unified_kv", params_base.kv_unified},
        };
    }

    bool telemetry_copy_moe_routing_readback(telemetry_moe_routing_readback_capture & destination) {
        destination = {};
        const llama_moe_routing_readback * source = llama_get_moe_routing_readback(ctx_tgt);
        if (source == nullptr || source->version != LLAMA_MOE_ROUTING_READBACK_VERSION ||
                source->struct_size < sizeof(llama_moe_routing_readback)) {
            return false;
        }

        destination.version = source->version;
        destination.capture_generation = source->capture_generation;
        if (source->row_count > 0 && source->rows == nullptr) {
            return false;
        }
        destination.rows.reserve(source->row_count);
        for (size_t index = 0; index < source->row_count; ++index) {
            const llama_moe_routing_row & row = source->rows[index];
            telemetry_moe_routing_row_capture copied;
            copied.layer_index = row.layer_index;
            copied.graph_type = row.graph_type;
            copied.physical_ubatch_index = row.physical_ubatch_index;
            copied.row_index = row.row_index;
            copied.ubatch_token_index = row.ubatch_token_index;
            copied.token_index = row.token_index;
            copied.token = row.token;
            copied.position = row.position;
            copied.row_identity_status = row.row_identity_status;
            copied.selected_experts_status = row.selected_experts_status;
            copied.selected_score = row.selected_score;
            copied.rejected_score = row.rejected_score;
            copied.selected_score_status = row.selected_score_status;
            copied.rejected_score_status = row.rejected_score_status;
            if (row.selected_expert_count > 0 && row.selected_experts == nullptr) {
                copied.selected_experts_status = LLAMA_MOE_ROUTING_VALUE_STATUS_INVALID;
            } else {
                copied.selected_experts.reserve(row.selected_expert_count);
                for (size_t expert_index = 0; expert_index < row.selected_expert_count; ++expert_index) {
                    const llama_moe_routing_expert & expert = row.selected_experts[expert_index];
                    copied.selected_experts.push_back({
                        expert.expert_index,
                        expert.effective_weight,
                        expert.expert_index_status,
                        expert.effective_weight_status,
                    });
                }
            }
            destination.rows.push_back(std::move(copied));
        }

        if (source->shared_expert_count > 0 && source->shared_experts == nullptr) {
            return false;
        }
        destination.shared_experts.reserve(source->shared_expert_count);
        for (size_t index = 0; index < source->shared_expert_count; ++index) {
            const llama_moe_shared_expert_metadata & metadata = source->shared_experts[index];
            destination.shared_experts.push_back({
                metadata.layer_index,
                metadata.graph_type,
                metadata.present,
                metadata.configured_count,
                metadata.ffn_size,
            });
        }
        return true;
    }

    static uint32_t telemetry_moe_value_status_number(llama_moe_routing_value_status status) {
        return (uint32_t) status;
    }

    static const char * telemetry_moe_value_status_reason(llama_moe_routing_value_status status) {
        switch (status) {
            case LLAMA_MOE_ROUTING_VALUE_STATUS_SOURCE_UNAVAILABLE: return "source_unavailable";
            case LLAMA_MOE_ROUTING_VALUE_STATUS_INVALID: return "invalid";
            case LLAMA_MOE_ROUTING_VALUE_STATUS_NONFINITE: return "nonfinite";
            case LLAMA_MOE_ROUTING_VALUE_STATUS_VALID: return nullptr;
        }
        return "invalid";
    }

    static bool telemetry_moe_value_is_valid(
            llama_moe_routing_value_status status,
            float value) {
        return status == LLAMA_MOE_ROUTING_VALUE_STATUS_VALID && std::isfinite(value);
    }

    static bool telemetry_moe_status_is_invalid(llama_moe_routing_value_status status) {
        return status == LLAMA_MOE_ROUTING_VALUE_STATUS_INVALID ||
            status == LLAMA_MOE_ROUTING_VALUE_STATUS_NONFINITE;
    }

    static bool telemetry_moe_status_is_unavailable(llama_moe_routing_value_status status) {
        return status == LLAMA_MOE_ROUTING_VALUE_STATUS_SOURCE_UNAVAILABLE;
    }

    static uint32_t telemetry_moe_phase_number(
            uint32_t graph_type,
            const server_batch::token & token,
            const server_slot & slot) {
        if (graph_type == LLM_GRAPH_TYPE_DECODER_MTP) {
            return 3; // MtpVerify
        }
        if (graph_type == LLM_GRAPH_TYPE_ENCODER) {
            return 1; // PrefillOutput
        }
        if (token.is_prompt) {
            return 1; // PrefillOutput
        }
        return slot.can_speculate() && !slot.spec_draft.empty()
            ? 3 // MtpVerify
            : 2; // NormalDecode
    }

    static const char * telemetry_moe_graph_type_name(uint32_t graph_type) {
        switch (graph_type) {
            case LLM_GRAPH_TYPE_DEFAULT: return "default";
            case LLM_GRAPH_TYPE_ENCODER: return "encoder";
            case LLM_GRAPH_TYPE_DECODER: return "decoder";
            case LLM_GRAPH_TYPE_DECODER_MTP: return "decoder_mtp";
        }
        return "unknown";
    }

    bool telemetry_moe_row_is_invalid(const telemetry_moe_routing_row_capture & row) const {
        if (row.layer_index < 0 || telemetry_moe_status_is_invalid(row.row_identity_status) ||
                telemetry_moe_status_is_invalid(row.selected_experts_status) ||
                telemetry_moe_status_is_invalid(row.selected_score_status) ||
                telemetry_moe_status_is_invalid(row.rejected_score_status)) {
            return true;
        }
        for (const telemetry_moe_routing_expert_capture & expert : row.selected_experts) {
            if (telemetry_moe_status_is_invalid(expert.expert_index_status) ||
                    telemetry_moe_status_is_invalid(expert.effective_weight_status)) {
                return true;
            }
        }
        return false;
    }

    bool telemetry_moe_row_is_unavailable(const telemetry_moe_routing_row_capture & row) const {
        if (telemetry_moe_status_is_unavailable(row.row_identity_status) ||
                telemetry_moe_status_is_unavailable(row.selected_experts_status) ||
                telemetry_moe_status_is_unavailable(row.selected_score_status) ||
                telemetry_moe_status_is_unavailable(row.rejected_score_status)) {
            return true;
        }
        for (const telemetry_moe_routing_expert_capture & expert : row.selected_experts) {
            if (telemetry_moe_status_is_unavailable(expert.expert_index_status) ||
                    telemetry_moe_status_is_unavailable(expert.effective_weight_status)) {
                return true;
            }
        }
        return false;
    }

    size_t telemetry_moe_chunk_limit_bytes() const {
        return std::min(telemetry_moe_chunk_max_bytes, telemetry_event_max_bytes);
    }

    static std::string telemetry_moe_created_at() {
        const auto now = std::chrono::system_clock::now();
        const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
        const time_t seconds = (time_t) (milliseconds.count() / 1000);
        std::tm utc = {};
#if defined(_WIN32)
        gmtime_s(&utc, &seconds);
#else
        gmtime_r(&seconds, &utc);
#endif
        char timestamp[32] = {};
        std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", &utc);
        return string_format("%s.%03" PRId64 "Z", timestamp, milliseconds.count() % 1000);
    }

    std::string telemetry_serialize_moe_routing_chunk(json event, uint64_t sequence) const {
        event["schema_version"] = 2;
        event["event"] = "moe_routing_chunk";
        event["server_instance_id"] = telemetry_server_instance_id;
        event["sequence"] = sequence;
        event["serialized_bytes"] = 0;

        std::string serialized = event.dump();
        const std::string marker = "\"serialized_bytes\":0";
        const size_t marker_offset = serialized.find(marker);
        GGML_ASSERT(marker_offset != std::string::npos);

        size_t final_bytes = serialized.size();
        while (true) {
            const size_t next_bytes = serialized.size() - 1 + std::to_string(final_bytes).size();
            if (next_bytes == final_bytes) {
                break;
            }
            final_bytes = next_bytes;
        }
        serialized.replace(marker_offset + marker.size() - 1, 1, std::to_string(final_bytes));
        GGML_ASSERT(serialized.size() == final_bytes);
        return serialized;
    }

    bool telemetry_moe_chunk_fits_limit(const json & event) const {
        // Pack against the widest possible event-ring sequence so a later
        // sequence-digit boundary cannot make an already-buffered chunk exceed
        // its exact serialized limit.
        return telemetry_serialize_moe_routing_chunk(event, std::numeric_limits<uint64_t>::max()).size()
            <= telemetry_moe_chunk_limit_bytes();
    }

    json telemetry_moe_routing_descriptor() const {
        const int32_t routed_expert_count = llama_model_n_expert(model_tgt);
        const int32_t experts_per_token = llama_model_n_expert_used(model_tgt);
        const int32_t model_layer_count = llama_model_n_layer(model_tgt);
        const int32_t moe_layer_count = llama_model_n_moe_layer(model_tgt);
        const int32_t shared_expert_count = llama_model_n_expert_shared(model_tgt);
        GGML_ASSERT(routed_expert_count > 0 && experts_per_token > 0 && model_layer_count > 0 && moe_layer_count > 0);

        json moe_layer_indices = json::array();
        int32_t previous_layer_index = -1;
        for (int32_t index = 0; index < moe_layer_count; ++index) {
            const int32_t layer_index = llama_model_moe_layer_index(model_tgt, index);
            GGML_ASSERT(layer_index > previous_layer_index && layer_index < model_layer_count);
            moe_layer_indices.push_back(layer_index);
            previous_layer_index = layer_index;
        }

        return {
            {"schema_version", 2},
            {"server_instance_id", telemetry_server_instance_id},
            {"server_build", std::string(llama_build_info())},
            {"model_id", model_name},
            {"model_fingerprint", nullptr},
            {"model_fingerprint_availability", 10},
            {"model_fingerprint_reason", "llama-server does not expose a stable model digest."},
            {"routed_expert_count", routed_expert_count},
            {"experts_per_token", experts_per_token},
            {"moe_layer_count", moe_layer_count},
            {"model_layer_count", model_layer_count},
            {"moe_layer_indices", std::move(moe_layer_indices)},
            {"shared_expert_count", shared_expert_count},
            {"weight_semantics", "exact effective routed coefficient"},
            {"score_semantics", "router score after selection bias and group masking"},
            {"clock_domain", "utc_wall_clock"},
        };
    }

    size_t telemetry_moe_chunk_finalized_size(json event) const {
        // The one pending chunk becomes the substantive final chunk. Reserve
        // the bounded, coordinate-free interruption evidence it may need at
        // completion; never replace an oversized substantive chunk with an
        // empty "final" marker.
        json final_event = event;
        final_event["is_final_for_trace"] = true;
        final_event["availability"] = 1;
        final_event["reason"] = server_moe_routing_partial_reason(
            { 1, 1, 1, 1, true, true, true, true }, "The request ended with partial routing coverage:");
        final_event["unlocated_coverage_loss"] = {
            {"count", std::numeric_limits<uint64_t>::max()},
            {"reason", "Routing capture ended after an unavailable native or control-boundary interval."},
        };
        return telemetry_serialize_moe_routing_chunk(
            std::move(final_event), std::numeric_limits<uint64_t>::max()).size();
    }

    bool telemetry_moe_chunk_can_be_finalized(const json & event) const {
        return telemetry_moe_chunk_finalized_size(event) <= telemetry_moe_chunk_limit_bytes();
    }

    bool telemetry_append_moe_routing_chunk(json event) {
        const uint64_t sequence = telemetry_next_sequence;
        std::string serialized = telemetry_serialize_moe_routing_chunk(std::move(event), sequence);
        if (serialized.size() > telemetry_moe_chunk_limit_bytes() || serialized.size() > telemetry_event_max_bytes) {
            return false;
        }
        ++telemetry_next_sequence;
        telemetry_append_serialized(sequence, std::move(serialized));
        return true;
    }

    bool telemetry_flush_moe_pending_chunk(server_slot & slot) {
        if (slot.telemetry_moe_pending_chunk.is_null()) {
            return true;
        }
        if (!telemetry_append_moe_routing_chunk(slot.telemetry_moe_pending_chunk)) {
            return false;
        }
        slot.telemetry_moe_pending_chunk = json();
        return true;
    }

    static uint64_t telemetry_moe_chunk_unlocated_loss_count(const json & event) {
        if (!event.contains("unlocated_coverage_loss")) {
            return 0;
        }
        const json & loss = event.at("unlocated_coverage_loss");
        GGML_ASSERT(loss.is_object() && loss.contains("count") && loss.at("count").is_number_integer());
        return loss.at("count").get<uint64_t>();
    }

    static uint64_t telemetry_moe_chunk_population_count(const json & event) {
        uint64_t result = 0;
        const auto add = [&](uint64_t count) {
            GGML_ASSERT(count <= std::numeric_limits<uint64_t>::max() - result);
            result += count;
        };
        const auto add_array_size = [&](const char * name) {
            if (event.contains(name) && event.at(name).is_array()) {
                add((uint64_t) event.at(name).size());
            }
        };

        add_array_size("decisions");
        add_array_size("invalid_records");
        if (event.contains("gaps") && event.at("gaps").is_array()) {
            for (const json & gap : event.at("gaps")) {
                if (!gap.contains("first_sequence") || !gap.contains("next_sequence")) {
                    continue;
                }
                const uint64_t first = gap.at("first_sequence").get<uint64_t>();
                const uint64_t next = gap.at("next_sequence").get<uint64_t>();
                if (next > first) {
                    add(next - first);
                }
            }
        }
        add(telemetry_moe_chunk_unlocated_loss_count(event));
        return result;
    }

    void telemetry_queue_moe_routing_chunk(server_slot & slot, json event) {
        GGML_ASSERT(telemetry_moe_chunk_can_be_finalized(event));
        if (!telemetry_flush_moe_pending_chunk(slot)) {
            const uint64_t pending_population = telemetry_moe_chunk_population_count(slot.telemetry_moe_pending_chunk);
            const uint64_t incoming_population = telemetry_moe_chunk_population_count(event);
            uint64_t lost_population = pending_population;
            GGML_ASSERT(server_moe_routing_add_lost_population(incoming_population, lost_population));
            GGML_ASSERT(server_moe_routing_add_lost_population(
                lost_population, slot.telemetry_moe_chunk_unlocated_pending));
            GGML_ASSERT(server_moe_routing_add_lost_population(
                lost_population, slot.telemetry_moe_chunk_unlocated_rows));
            slot.telemetry_moe_pending_chunk = json();
            return;
        }
        slot.telemetry_moe_pending_chunk = std::move(event);
    }

    void telemetry_record_moe_routing_chunks(
            const telemetry_moe_routing_readback_capture & readback,
            int32_t batch_offset,
            int32_t batch_token_count,
            uint64_t control_generation) {
        if (readback.rows.empty()) {
            return;
        }

        struct physical_event_id {
            uint32_t microbatch = 0;
            int32_t layer = -1;
            uint32_t phase = 0;

            bool operator < (const physical_event_id & other) const {
                return std::tie(microbatch, layer, phase) < std::tie(other.microbatch, other.layer, other.phase);
            }
        };
        struct physical_event_coverage {
            std::set<std::string> expected_trace_ids;
            std::set<std::string> captured_trace_ids;
            size_t expected_decisions = 0;
            size_t captured_valid_decisions = 0;
        };
        struct trace_record {
            const telemetry_moe_routing_row_capture * row = nullptr;
            const server_batch::token * token = nullptr;
            physical_event_id event;
            uint64_t sequence = 0;
            bool invalid = false;
            bool gap = false;
            int64_t proposal_position = -1;
        };
        struct trace_capture {
            server_slot * slot = nullptr;
            std::vector<trace_record> records;
            size_t unlocated_rows = 0;
            std::string created_at;
            uint64_t pending_unlocated_rows = 0;
            std::vector<std::vector<const trace_record *>> chunks;
        };

        std::map<std::string, trace_capture> captures_by_trace;
        std::map<physical_event_id, physical_event_coverage> coverage_by_event;
        std::vector<int64_t> proposal_positions((size_t) batch_token_count, -1);
        std::vector<int64_t> next_proposal_position(slots.size(), 0);

        for (int32_t index = 0; index < batch_token_count; ++index) {
            const server_batch::token & token = batch.tokens[batch_offset + index];
            if (token.id_slot < 0 || token.id_slot >= (int32_t) slots.size()) {
                continue;
            }
            const server_slot & slot = slots[token.id_slot];
            if (!token.is_prompt && slot.can_speculate() && !slot.spec_draft.empty()) {
                proposal_positions[(size_t) index] = next_proposal_position[(size_t) token.id_slot]++;
            }
        }

        const auto record_unmappable_row = [&]() {
            std::set<int32_t> candidate_slots;
            for (int32_t index = 0; index < batch_token_count; ++index) {
                const server_batch::token & candidate = batch.tokens[batch_offset + index];
                if (candidate.id_slot < 0 || candidate.id_slot >= (int32_t) slots.size()) {
                    continue;
                }
                server_slot & slot = slots[candidate.id_slot];
                if (!telemetry_moe_request_enabled(slot)) {
                    continue;
                }
                candidate_slots.insert(candidate.id_slot);
            }
            if (candidate_slots.size() == 1) {
                server_slot & slot = slots[*candidate_slots.begin()];
                slot.telemetry_moe_chunk_capture_started = true;
                ++slot.telemetry_moe_chunk_unlocated_pending;
                ++slot.telemetry_moe_chunk_unlocated_rows;
                ++slot.telemetry_moe_chunk_unlinked_rows;
                return;
            }
            for (const int32_t slot_id : candidate_slots) {
                server_slot & slot = slots[slot_id];
                slot.telemetry_moe_chunk_capture_started = true;
                slot.telemetry_moe_chunk_attribution_ambiguous = true;
            }
        };

        for (size_t row_index = 0; row_index < readback.rows.size(); ++row_index) {
            const telemetry_moe_routing_row_capture & row = readback.rows[row_index];
            if (row.token_index < 0 || row.token_index >= batch_token_count) {
                record_unmappable_row();
                continue;
            }
            const server_batch::token & token = batch.tokens[batch_offset + row.token_index];
            if (token.id_slot < 0 || token.id_slot >= (int32_t) slots.size()) {
                record_unmappable_row();
                continue;
            }
            server_slot & slot = slots[token.id_slot];
            if (!slot.task) {
                continue;
            }
            const uint32_t phase = telemetry_moe_phase_number(row.graph_type, token, slot);
            const physical_event_id event_id = { row.physical_ubatch_index, row.layer_index, phase };
            const bool has_physical_event = row.layer_index >= 0;
            const bool locatable = row.layer_index >= 0
                && row.position >= 0
                && row.row_index >= 0
                && row.ubatch_token_index >= 0;
            const bool enabled = telemetry_moe_request_enabled(slot);
            if (has_physical_event) {
                physical_event_coverage & coverage = coverage_by_event[event_id];
                coverage.expected_trace_ids.insert(slot.task->trace_id);
                ++coverage.expected_decisions;
            }
            if (!enabled) {
                continue;
            }
            if (!locatable) {
                trace_capture & capture = captures_by_trace[slot.task->trace_id];
                capture.slot = &slot;
                if (capture.created_at.empty()) {
                    capture.created_at = telemetry_moe_created_at();
                }
                ++capture.unlocated_rows;
                ++slot.telemetry_moe_chunk_unlocated_rows;
                ++slot.telemetry_moe_chunk_unlinked_rows;
                continue;
            }

            physical_event_coverage & coverage = coverage_by_event.at(event_id);

            const int32_t expected_experts = llama_model_n_expert_used(model_tgt);
            const bool invalid = telemetry_moe_row_is_invalid(row)
                || expected_experts <= 0
                || row.selected_experts.size() != (size_t) expected_experts;
            trace_capture & capture = captures_by_trace[slot.task->trace_id];
            capture.slot = &slot;
            if (capture.created_at.empty()) {
                capture.created_at = telemetry_moe_created_at();
            }
            capture.records.push_back({
                &row,
                &token,
                event_id,
                slot.telemetry_moe_chunk_decision_sequence++,
                invalid,
                false,
                proposal_positions[(size_t) row.token_index],
            });
            coverage.captured_trace_ids.insert(slot.task->trace_id);
            coverage.captured_valid_decisions += invalid ? 0 : 1;
            ++slot.telemetry_moe_chunk_trace_rows;
            ++slot.telemetry_moe_chunk_rows;
            slot.telemetry_moe_chunk_invalid_rows += invalid;
            slot.telemetry_moe_chunk_unavailable_rows += telemetry_moe_row_is_unavailable(row);
        }

        const json descriptor = telemetry_moe_routing_descriptor();

        const auto shared_metadata = [&](const telemetry_moe_routing_row_capture & row) {
            for (const telemetry_moe_shared_expert_capture & shared : readback.shared_experts) {
                if (shared.layer_index == row.layer_index && shared.graph_type == row.graph_type) {
                    return json {
                        {"configured_count", shared.configured_count},
                        {"present", shared.present},
                        {"ffn_size", shared.ffn_size},
                        {"activated_expert_ids", json::array()},
                        {"execution_semantics", "metadata_only"},
                    };
                }
            }
            return json {
                {"configured_count", 0},
                {"present", false},
                {"ffn_size", nullptr},
                {"reason", "No shared-expert metadata was retained for this router row."},
                {"activated_expert_ids", json::array()},
                {"execution_semantics", "metadata_only"},
            };
        };

        const auto event_key = [&](const trace_record & record) {
            return json {
                {"server_instance_id", telemetry_server_instance_id},
                {"physical_microbatch", record.event.microbatch},
                {"physical_step", readback.capture_generation},
                {"phase", record.event.phase},
                {"layer_index", record.event.layer},
                {"control_generation", control_generation},
            };
        };

        const auto speculative_pass = [&](const trace_record & record, const server_slot & slot) {
            if (record.event.phase != 3) {
                return json(nullptr);
            }
            return json {
                {"logical_verification_step", (int64_t) slot.stats.n_draft_verif_steps},
                {"actual_target_pass", (int64_t) slot.n_spec_target_passes},
                {"proposal_position", record.proposal_position},
                {"is_replay_pass", slot.spec_is_replay},
            };
        };

        const auto selected_experts = [&](const telemetry_moe_routing_row_capture & row) {
            json result = json::array();
            for (const telemetry_moe_routing_expert_capture & expert : row.selected_experts) {
                const bool weight_valid = telemetry_moe_value_is_valid(expert.effective_weight_status, expert.effective_weight);
                json selected = {
                    {"expert_id", expert.expert_index},
                    {"effective_weight", weight_valid ? json(expert.effective_weight) : json(nullptr)},
                    {"expert_id_status", telemetry_moe_value_status_number(expert.expert_index_status)},
                    {"effective_weight_status", telemetry_moe_value_status_number(expert.effective_weight_status)},
                };
                if (const char * reason = telemetry_moe_value_status_reason(expert.expert_index_status)) {
                    selected["expert_id_reason"] = reason;
                }
                if (const char * reason = telemetry_moe_value_status_reason(expert.effective_weight_status)) {
                    selected["effective_weight_reason"] = reason;
                }
                result.push_back(std::move(selected));
            }
            return result;
        };

        const auto make_event = [&](const std::string & trace_id, const trace_capture & capture,
                                    const std::vector<const trace_record *> & records,
                                    uint64_t unlocated_rows, const std::string & chunk_id) {
            json decisions = json::array();
            json invalid_records = json::array();
            json intervals = json::array();
            json gaps = json::array();
            std::set<physical_event_id> physical_events_in_chunk;
            for (const trace_record * record : records) {
                const telemetry_moe_routing_row_capture & row = *record->row;
                if (record->gap) {
                    gaps.push_back({
                        {"first_sequence", record->sequence},
                        {"next_sequence", record->sequence + 1},
                        {"first_physical_step", readback.capture_generation},
                        {"last_physical_step", readback.capture_generation},
                        {"first_physical_microbatch", record->event.microbatch},
                        {"last_physical_microbatch", record->event.microbatch},
                        {"first_model_position", row.position},
                        {"last_model_position", row.position},
                        {"phase", record->event.phase},
                        {"layer_index", record->event.layer},
                        {"control_generation", control_generation},
                        {"cause", 8},
                        {"reason", "The producer could not retain this exact routing record within the serialized chunk limit."},
                    });
                    continue;
                }
                physical_events_in_chunk.insert(record->event);
                if (record->invalid) {
                    invalid_records.push_back({
                        {"sequence", record->sequence},
                        {"request_trace_id", trace_id},
                        {"event_key", event_key(*record)},
                        {"model_position", row.position},
                        {"speculative_pass", speculative_pass(*record, *capture.slot)},
                        {"code", "invalid_router_row"},
                        {"reason", "The native router row contained invalid or non-finite routing evidence."},
                    });
                    continue;
                }

                const bool kth_valid = telemetry_moe_value_is_valid(row.selected_score_status, row.selected_score);
                const bool rejected_valid = telemetry_moe_value_is_valid(row.rejected_score_status, row.rejected_score);
                json decision = {
                    {"sequence", record->sequence},
                    {"request_trace_id", trace_id},
                    {"event_key", event_key(*record)},
                    {"model_position", row.position},
                    {"speculative_pass", speculative_pass(*record, *capture.slot)},
                    {"selected_experts", selected_experts(row)},
                    {"kth_selected_score", kth_valid ? json(row.selected_score) : json(nullptr)},
                    {"kth_selected_score_status", telemetry_moe_value_status_number(row.selected_score_status)},
                    {"highest_rejected_score", rejected_valid ? json(row.rejected_score) : json(nullptr)},
                    {"highest_rejected_score_status", telemetry_moe_value_status_number(row.rejected_score_status)},
                    {"native_row", {
                        {"graph_type", telemetry_moe_graph_type_name(row.graph_type)},
                        {"physical_ubatch_index", row.physical_ubatch_index},
                        {"row_index", row.row_index},
                        {"ubatch_token_index", row.ubatch_token_index},
                        {"logical_token_index", row.token_index},
                        {"row_identity_status", telemetry_moe_value_status_number(row.row_identity_status)},
                    }},
                    {"shared_experts", shared_metadata(row)},
                };
                if (const char * reason = telemetry_moe_value_status_reason(row.selected_score_status)) {
                    decision["kth_selected_score_reason"] = reason;
                }
                if (const char * reason = telemetry_moe_value_status_reason(row.rejected_score_status)) {
                    decision["highest_rejected_score_reason"] = reason;
                }
                decisions.push_back(std::move(decision));
                intervals.push_back({
                    {"first_sequence", record->sequence},
                    {"next_sequence", record->sequence + 1},
                    {"first_physical_step", readback.capture_generation},
                    {"last_physical_step", readback.capture_generation},
                    {"first_physical_microbatch", record->event.microbatch},
                    {"last_physical_microbatch", record->event.microbatch},
                    {"first_model_position", row.position},
                    {"last_model_position", row.position},
                    {"phase", record->event.phase},
                    {"layer_index", record->event.layer},
                    {"control_generation", control_generation},
                });
            }

            json physical_events = json::array();
            for (const physical_event_id & id : physical_events_in_chunk) {
                const physical_event_coverage & coverage = coverage_by_event.at(id);
                const bool complete = coverage.captured_valid_decisions == coverage.expected_decisions
                    && coverage.captured_trace_ids.size() == coverage.expected_trace_ids.size();
                json trace_ids = json::array();
                for (const std::string & captured_trace_id : coverage.captured_trace_ids) {
                    trace_ids.push_back(captured_trace_id);
                }
                json physical = {
                    {"event_key", {
                        {"server_instance_id", telemetry_server_instance_id},
                        {"physical_microbatch", id.microbatch},
                        {"physical_step", readback.capture_generation},
                        {"phase", id.phase},
                        {"layer_index", id.layer},
                        {"control_generation", control_generation},
                    }},
                    {"expected_request_trace_count", coverage.expected_trace_ids.size()},
                    {"expected_decision_count", coverage.expected_decisions},
                    {"captured_valid_decision_count", coverage.captured_valid_decisions},
                    {"captured_request_trace_ids", trace_ids},
                    {"is_complete", complete},
                };
                if (!complete) {
                    physical["reason"] = "One or more request peers or router rows were not retained as valid routing decisions.";
                }
                physical_events.push_back(std::move(physical));
            }

            const server_moe_routing_chunk_coverage routing_coverage = {
                capture.slot->telemetry_moe_chunk_invalid_rows,
                capture.slot->telemetry_moe_chunk_unavailable_rows,
                capture.slot->telemetry_moe_chunk_unlinked_rows,
                unlocated_rows,
                capture.slot->telemetry_moe_chunk_capture_interrupted,
                capture.slot->telemetry_moe_chunk_source_unavailable,
                capture.slot->telemetry_moe_chunk_attribution_ambiguous,
                !gaps.empty(),
            };
            json event = {
                {"chunk_id", chunk_id},
                {"trace_id", trace_id},
                {"created_at", capture.created_at},
                {"first_sequence", records.empty() ? uint64_t(0) : records.front()->sequence},
                {"next_sequence", records.empty() ? uint64_t(0) : records.back()->sequence + 1},
                {"is_final_for_trace", false},
                {"descriptor", descriptor},
                {"coverage_intervals", intervals},
                {"gaps", gaps},
                {"invalid_records", invalid_records},
                {"physical_events", physical_events},
                {"decisions", decisions},
            };
            server_moe_routing_apply_canonical_event_coverage(
                event, routing_coverage, !records.empty(),
                "The producer retained a partial routing population:",
                "Routing rows were lost before complete routing coordinates were retained.");
            return event;
        };

        for (auto & trace_pair : captures_by_trace) {
            const std::string & trace_id = trace_pair.first;
            trace_capture & capture = trace_pair.second;
            if (capture.slot == nullptr) {
                continue;
            }

            capture.pending_unlocated_rows = capture.slot->telemetry_moe_chunk_unlocated_pending;
            capture.slot->telemetry_moe_chunk_unlocated_pending = 0;
            uint64_t unlocated_rows = capture.unlocated_rows;
            GGML_ASSERT(server_moe_routing_add_lost_population(
                capture.pending_unlocated_rows, unlocated_rows));
            std::vector<std::vector<const trace_record *>> & chunks = capture.chunks;
            std::vector<const trace_record *> current;
            const std::vector<const trace_record *> no_records;
            const size_t chunk_base_upper_bound = telemetry_moe_chunk_finalized_size(make_event(
                trace_id, capture, no_records, std::numeric_limits<uint64_t>::max(),
                "moe-pending-18446744073709551615"));
            size_t current_upper_bound = chunk_base_upper_bound;
            constexpr size_t completeness_reserve = 128;
            for (trace_record & record : capture.records) {
                const std::vector<const trace_record *> one_record = { &record };
                size_t one_record_upper_bound = telemetry_moe_chunk_finalized_size(make_event(
                    trace_id, capture, one_record, std::numeric_limits<uint64_t>::max(),
                    "moe-pending-18446744073709551615"));
                if (one_record_upper_bound > telemetry_moe_chunk_limit_bytes() && !record.invalid) {
                    record.gap = true;
                    physical_event_coverage & coverage = coverage_by_event.at(record.event);
                    if (coverage.captured_valid_decisions > 0) {
                        --coverage.captured_valid_decisions;
                    }
                    one_record_upper_bound = telemetry_moe_chunk_finalized_size(make_event(
                        trace_id, capture, one_record, std::numeric_limits<uint64_t>::max(),
                        "moe-pending-18446744073709551615"));
                }
                GGML_ASSERT(one_record_upper_bound >= chunk_base_upper_bound);
                size_t record_upper_bound = one_record_upper_bound - chunk_base_upper_bound;
                if (record_upper_bound <= std::numeric_limits<size_t>::max() - completeness_reserve) {
                    record_upper_bound += completeness_reserve;
                }
                if (chunk_base_upper_bound > telemetry_moe_chunk_limit_bytes()
                        || record_upper_bound > telemetry_moe_chunk_limit_bytes() - chunk_base_upper_bound) {
                    ++capture.slot->telemetry_moe_chunk_unlocated_pending;
                    ++capture.slot->telemetry_moe_chunk_unlocated_rows;
                    continue;
                }
                if (!current.empty() && record_upper_bound > telemetry_moe_chunk_limit_bytes() - current_upper_bound) {
                    chunks.push_back(std::move(current));
                    current.clear();
                    current_upper_bound = chunk_base_upper_bound;
                }
                current.push_back(&record);
                current_upper_bound += record_upper_bound;
            }
            if (!current.empty()) {
                chunks.push_back(std::move(current));
            }
            if (chunks.empty()) {
                GGML_ASSERT(server_moe_routing_add_lost_population(
                    unlocated_rows, capture.slot->telemetry_moe_chunk_unlocated_pending));
                capture.slot->telemetry_moe_chunk_capture_started = true;
                continue;
            }

            capture.slot->telemetry_moe_chunk_capture_started = true;
        }

        for (auto & trace_pair : captures_by_trace) {
            const std::string & trace_id = trace_pair.first;
            trace_capture & capture = trace_pair.second;
            if (capture.slot == nullptr || capture.chunks.empty()) {
                continue;
            }

            uint64_t unlocated_rows = capture.unlocated_rows;
            GGML_ASSERT(server_moe_routing_add_lost_population(
                capture.pending_unlocated_rows, unlocated_rows));
            for (size_t index = 0; index < capture.chunks.size(); ++index) {
                const std::string chunk_id = string_format("moe-%d-%" PRIu64,
                    capture.slot->id, ++capture.slot->telemetry_moe_chunk_sequence);
                json event = make_event(trace_id, capture, capture.chunks[index], index == 0 ? unlocated_rows : 0, chunk_id);
                GGML_ASSERT(telemetry_moe_chunk_fits_limit(event));
                GGML_ASSERT(telemetry_moe_chunk_can_be_finalized(event));
                telemetry_queue_moe_routing_chunk(*capture.slot, std::move(event));
            }
        }
    }

    void telemetry_record_moe_routing_final_marker(server_slot & slot, const char * outcome) {
        if (!slot.task || (!slot.telemetry_moe_chunk_capture_started && !slot.telemetry_moe_chunk_source_unavailable)) {
            return;
        }
        const uint64_t pending_unlocated_rows = slot.telemetry_moe_chunk_unlocated_pending;
        if (!slot.telemetry_moe_pending_chunk.is_null()) {
            slot.telemetry_moe_pending_chunk["is_final_for_trace"] = true;
            const uint64_t existing_unlocated_rows = telemetry_moe_chunk_unlocated_loss_count(slot.telemetry_moe_pending_chunk);
            uint64_t unlocated_rows = 0;
            GGML_ASSERT(server_moe_routing_combine_lost_population(
                existing_unlocated_rows, pending_unlocated_rows, unlocated_rows));
            const server_moe_routing_chunk_coverage routing_coverage = {
                slot.telemetry_moe_chunk_invalid_rows,
                slot.telemetry_moe_chunk_unavailable_rows,
                slot.telemetry_moe_chunk_unlinked_rows,
                unlocated_rows,
                slot.telemetry_moe_chunk_capture_interrupted,
                slot.telemetry_moe_chunk_source_unavailable,
                slot.telemetry_moe_chunk_attribution_ambiguous,
                false,
            };
            server_moe_routing_apply_canonical_event_coverage(
                slot.telemetry_moe_pending_chunk, routing_coverage, true,
                "The request ended with partial routing coverage:",
                "Routing capture lost rows before complete routing coordinates were retained.");
            slot.telemetry_moe_chunk_unlocated_pending = 0;
            GGML_ASSERT(telemetry_moe_chunk_can_be_finalized(slot.telemetry_moe_pending_chunk));
            GGML_ASSERT(telemetry_flush_moe_pending_chunk(slot));
            return;
        }

        if (llama_model_n_expert(model_tgt) <= 0 || llama_model_n_expert_used(model_tgt) <= 0 ||
                llama_model_n_moe_layer(model_tgt) <= 0) {
            return;
        }
        const uint64_t unlocated_rows = pending_unlocated_rows;
        const server_moe_routing_chunk_coverage routing_coverage = {
            slot.telemetry_moe_chunk_invalid_rows,
            slot.telemetry_moe_chunk_unavailable_rows,
            slot.telemetry_moe_chunk_unlinked_rows,
            unlocated_rows,
            slot.telemetry_moe_chunk_capture_interrupted,
            slot.telemetry_moe_chunk_source_unavailable,
            slot.telemetry_moe_chunk_attribution_ambiguous,
            false,
        };
        json marker = {
            {"chunk_id", string_format("moe-%d-%" PRIu64, slot.id, ++slot.telemetry_moe_chunk_sequence)},
            {"trace_id", slot.task->trace_id},
            {"created_at", telemetry_moe_created_at()},
            {"first_sequence", slot.telemetry_moe_chunk_decision_sequence},
            {"next_sequence", slot.telemetry_moe_chunk_decision_sequence},
            {"is_final_for_trace", true},
            {"descriptor", telemetry_moe_routing_descriptor()},
        };
        server_moe_routing_apply_canonical_event_coverage(
            marker, routing_coverage, false,
            "The request ended with partial routing coverage:",
            "Routing capture ended after an unavailable native or control-boundary interval.",
            "No routable MoE records were retained for this request.");
        GGML_ASSERT(telemetry_moe_chunk_can_be_finalized(marker));
        GGML_ASSERT(telemetry_append_moe_routing_chunk(std::move(marker)));
    }

    bool telemetry_moe_request_enabled(const server_slot & slot) const {
        return llama_model_n_expert(model_tgt) > 0
            && slot.task
            && slot.task->params.moe_routing_telemetry_permitted;
    }

    void telemetry_prepare_moe_storage(server_slot & slot) {
        const size_t histogram_size = (size_t) llama_model_n_layer(model_tgt)*(size_t) llama_model_n_expert(model_tgt);
        if (slot.telemetry_moe_expert_activations.size() != histogram_size) {
            slot.telemetry_moe_expert_activations.assign(histogram_size, 0);
        }
        if (slot.telemetry_moe_token_activations.capacity() == 0) {
            slot.telemetry_moe_token_activations.reserve(telemetry_moe_activation_limit);
        }
    }

    void telemetry_record_moe_routing(
            const telemetry_moe_routing_readback_capture & readback,
            int32_t batch_offset,
            int32_t batch_token_count) {
        std::vector<llama_moe_routing_entry> entries;
        for (const telemetry_moe_routing_row_capture & row : readback.rows) {
            for (const telemetry_moe_routing_expert_capture & expert : row.selected_experts) {
                entries.push_back({
                    row.layer_index,
                    row.token_index,
                    expert.expert_index,
                    expert.effective_weight,
                });
            }
        }
        if (entries.empty()) {
            return;
        }

        const int32_t configured_experts = llama_model_n_expert(model_tgt);
        const int32_t model_layers = llama_model_n_layer(model_tgt);
        std::vector<bool> routed_tokens((size_t) batch_token_count, false);
        int32_t previous_layer = -1;
        int32_t previous_token = -1;

        for (const llama_moe_routing_entry & entry : entries) {
            if (entry.token_index < 0 || entry.token_index >= batch_token_count) {
                continue;
            }

            const auto & token = batch.tokens[batch_offset + entry.token_index];
            server_slot & slot = slots[token.id_slot];
            if (!telemetry_moe_request_enabled(slot)) {
                continue;
            }

            const bool valid_assignment = server_moe_routing_assignment_is_valid(
                entry.layer_index, entry.expert_index, model_layers, configured_experts);
            if (valid_assignment && (entry.layer_index != previous_layer || entry.token_index != previous_token)) {
                slot.telemetry_moe_routed_token_layers++;
                slot.telemetry_moe_layers.insert(entry.layer_index);
                if (!routed_tokens[(size_t) entry.token_index]) {
                    routed_tokens[(size_t) entry.token_index] = true;
                    slot.telemetry_moe_routed_tokens++;
                }
                previous_layer = entry.layer_index;
                previous_token = entry.token_index;
            }

            const server_moe_routing_capture_result capture = server_moe_routing_capture(
                slot.telemetry_moe_histogram_counts,
                valid_assignment,
                1,
                telemetry_moe_activation_limit);
            if (capture != SERVER_MOE_ROUTING_CAPTURED) {
                continue;
            }

            telemetry_prepare_moe_storage(slot);
            const size_t histogram_index = (size_t) entry.layer_index*(size_t) configured_experts
                + (size_t) entry.expert_index;
            slot.telemetry_moe_expert_activations[histogram_index]++;
        }

        bool has_mtp_verify = false;
        for (int32_t i = 0; i < batch_token_count; ++i) {
            const auto & token = batch.tokens[batch_offset + i];
            const server_slot & slot = slots[token.id_slot];
            if (!token.is_prompt && slot.can_speculate() && !slot.spec_draft.empty()) {
                has_mtp_verify = true;
                break;
            }
        }

        std::vector<int64_t> proposal_positions;
        if (has_mtp_verify) {
            proposal_positions.assign((size_t) batch_token_count, -1);
            std::vector<int64_t> next_proposal_position(slots.size(), 0);
            for (int32_t i = 0; i < batch_token_count; ++i) {
                const auto & token = batch.tokens[batch_offset + i];
                const server_slot & slot = slots[token.id_slot];
                const bool mtp_verify = !token.is_prompt && slot.can_speculate() && !slot.spec_draft.empty();
                if (mtp_verify) {
                    proposal_positions[(size_t) i] = next_proposal_position[(size_t) token.id_slot]++;
                }
            }
        }

        size_t group_start = 0;
        while (group_start < entries.size()) {
            const int32_t token_index = entries[group_start].token_index;
            const int32_t layer_index = entries[group_start].layer_index;
            size_t group_end = group_start + 1;
            while (group_end < entries.size()
                    && entries[group_end].token_index == token_index
                    && entries[group_end].layer_index == layer_index) {
                group_end++;
            }

            if (token_index < 0 || token_index >= batch_token_count) {
                group_start = group_end;
                continue;
            }

            const auto & token = batch.tokens[batch_offset + token_index];
            server_slot & slot = slots[token.id_slot];
            if (!telemetry_moe_request_enabled(slot) || !token.output) {
                group_start = group_end;
                continue;
            }

            const size_t activation_count = group_end - group_start;
            slot.telemetry_moe_token_decisions_total++;

            bool valid = layer_index >= 0 && layer_index < model_layers
                && activation_count == (size_t) llama_model_n_expert_used(model_tgt);
            for (size_t i = group_start; valid && i < group_end; ++i) {
                valid = server_moe_routing_assignment_is_valid(
                    layer_index, entries[i].expert_index, model_layers, configured_experts);
            }
            const server_moe_routing_capture_result capture = server_moe_routing_capture(
                slot.telemetry_moe_token_detail_counts,
                valid,
                activation_count,
                telemetry_moe_activation_limit);
            if (capture == SERVER_MOE_ROUTING_INVALID) {
                slot.telemetry_moe_token_decisions_invalid++;
                group_start = group_end;
                continue;
            }
            if (capture == SERVER_MOE_ROUTING_CAP_DROPPED) {
                slot.telemetry_moe_token_decisions_cap_dropped++;
                group_start = group_end;
                continue;
            }

            const bool mtp_verify = !token.is_prompt && slot.can_speculate() && !slot.spec_draft.empty();
            const telemetry_moe_token_phase phase = token.is_prompt
                ? TELEMETRY_MOE_TOKEN_PHASE_PREFILL_OUTPUT
                : mtp_verify ? TELEMETRY_MOE_TOKEN_PHASE_MTP_VERIFY : TELEMETRY_MOE_TOKEN_PHASE_NORMAL_DECODE;
            for (size_t i = group_start; i < group_end; ++i) {
                telemetry_moe_token_activation_record record;
                record.model_position = token.pos + 1;
                record.layer_index = layer_index;
                record.expert_index = entries[i].expert_index;
                record.effective_weight = entries[i].effective_weight;
                record.phase = phase;
                record.logical_step = mtp_verify ? (int64_t) slot.stats.n_draft_verif_steps : -1;
                record.actual_target_pass = mtp_verify ? (int64_t) slot.n_spec_target_passes : -1;
                record.proposal_position = mtp_verify ? proposal_positions[(size_t) token_index] : -1;
                record.replay_pass = mtp_verify && slot.spec_is_replay;
                slot.telemetry_moe_token_activations.push_back(record);
            }
            slot.telemetry_moe_token_decisions_captured++;
            group_start = group_end;
        }
    }

    bool telemetry_output_token_request_enabled(const server_slot & slot) const {
        return slot.task && slot.task->params.output_token_telemetry;
    }

    bool telemetry_token_candidate_request_enabled(const server_slot & slot) const {
        return telemetry_output_token_request_enabled(slot)
            && slot.task->params.output_token_candidate_telemetry;
    }

    bool telemetry_request_content_enabled(const server_slot & slot) const {
        return slot.task && slot.task->telemetry_content;
    }

    size_t telemetry_token_candidate_request_cap(const server_slot & slot) const {
        return (size_t) std::max(
            8192,
            std::min(
                (int) telemetry_token_candidate_max_block_bytes,
                slot.task->params.output_token_candidate_byte_cap));
    }

    size_t telemetry_token_candidate_capture_limit(const server_slot & slot) const {
        const size_t top_k = (size_t) std::max(1, slot.task->params.output_token_candidate_top_k);
        const size_t estimated_decision_bytes = 768 + top_k * 384;
        return std::max<size_t>(
            1,
            std::min(telemetry_token_candidate_decision_limit, telemetry_token_candidate_request_cap(slot) / estimated_decision_bytes));
    }

    bool telemetry_token_candidate_position_enabled(
            server_slot & slot,
            size_t position,
            size_t accepted_depth,
            size_t proposed_count) const {
        if (!telemetry_token_candidate_request_enabled(slot)) {
            return false;
        }
        bool eligible = false;
        if (position < accepted_depth) {
            eligible = slot.task->params.output_token_candidate_include_accepted;
        } else {
            // The first mismatch/replacement or target bonus is the default high-value population.
            eligible = position == accepted_depth || position == proposed_count;
        }
        if (!eligible) {
            return false;
        }
        slot.telemetry_token_candidate_eligible_count++;
        if (slot.telemetry_token_candidate_decisions.size() >= telemetry_token_candidate_capture_limit(slot)) {
            slot.telemetry_token_candidate_dropped_count++;
            return false;
        }
        return true;
    }

    void telemetry_record_token_candidates(
            server_slot & slot,
            size_t position,
            size_t accepted_depth,
            size_t proposed_count,
            llama_token selected,
            uint64_t related_output_ordinal,
            std::vector<telemetry_token_candidate_value> candidates,
            const std::string & probability_state,
            const std::string & probability_reason) {
        telemetry_token_candidate_decision record;
        record.logical_step = slot.telemetry_spec_logical_step;
        record.actual_target_pass = slot.n_spec_target_passes > 0 ? slot.n_spec_target_passes - 1 : 0;
        record.proposal_position = position;
        record.related_output_ordinal = (int64_t) related_output_ordinal;
        record.target_selected_token_id = selected;
        record.probability_state = probability_state;
        record.probability_reason = probability_reason;
        record.kind = position < accepted_depth
            ? TELEMETRY_TOKEN_CANDIDATE_ACCEPTED
            : accepted_depth < proposed_count
                ? TELEMETRY_TOKEN_CANDIDATE_FIRST_MISMATCH
                : TELEMETRY_TOKEN_CANDIDATE_TARGET_BONUS;
        if (position < slot.telemetry_mtp_proposals_pending.size()) {
            record.draft_token_id = slot.telemetry_mtp_proposals_pending[position].draft_token_id;
        }
        for (auto & candidate : candidates) {
            candidate.target_selected = candidate.token_id == selected;
            candidate.draft_proposed = candidate.token_id == record.draft_token_id;
        }
        record.target_candidates = std::move(candidates);
        slot.telemetry_token_candidate_decisions.push_back(std::move(record));
    }

    void telemetry_record_output_token(
            server_slot & slot,
            llama_token token_id,
            const std::string & token_piece,
            int64_t model_ready_us,
            llama_pos model_position,
            double selected_log_probability_ln,
            telemetry_output_token_origin origin,
            int64_t logical_step = -1,
            int64_t actual_target_pass = -1,
            int64_t proposal_position = -1,
            int64_t accepted_depth = -1,
            int64_t proposed_count = -1,
            bool replay_pass = false) {
        if (!telemetry_output_token_request_enabled(slot) || slot.telemetry_output_tokens.size() >= telemetry_output_token_limit) {
            return;
        }

        telemetry_output_token_record record;
        record.ordinal = slot.stats.n_gen > 0 ? slot.stats.n_gen - 1 : 0;
        record.model_ready_offset_us = std::max<int64_t>(0, model_ready_us - slot.stats.t_arrival);
        record.model_ready_monotonic_us = model_ready_us;
        record.model_position = model_position;
        record.token_id = telemetry_request_content_enabled(slot) ? token_id : LLAMA_TOKEN_NULL;
        if (telemetry_request_content_enabled(slot)) {
            record.token_piece = token_piece;
        }
        record.selected_log_probability_ln = selected_log_probability_ln;
        record.origin = origin;
        record.logical_step = logical_step;
        record.actual_target_pass = actual_target_pass;
        record.proposal_position = proposal_position;
        record.accepted_depth = accepted_depth;
        record.proposed_count = proposed_count;
        record.replay_pass = replay_pass;
        slot.telemetry_output_tokens.push_back(std::move(record));
    }

    void telemetry_prepare_mtp_proposals(
            server_slot & slot,
            const llama_tokens & draft,
            const llama_tokens & accepted) {
        slot.telemetry_mtp_proposals_pending.clear();
        const size_t remaining = telemetry_mtp_proposal_limit > slot.telemetry_mtp_proposals_captured
            ? telemetry_mtp_proposal_limit - slot.telemetry_mtp_proposals_captured
            : 0;
        const size_t captured = std::min(draft.size(), remaining);
        const size_t accepted_depth = accepted.empty() ? 0 : accepted.size() - 1;
        const uint64_t actual_target_pass = slot.n_spec_target_passes > 0 ? slot.n_spec_target_passes - 1 : 0;
        slot.telemetry_mtp_proposals_pending.reserve(captured);
        for (size_t position = 0; position < captured; ++position) {
            telemetry_mtp_proposal_record record;
            record.position = position;
            record.evaluated_actual_target_pass = actual_target_pass;
            record.draft_token_id = telemetry_request_content_enabled(slot) ? draft[position] : LLAMA_TOKEN_NULL;
            if (position < accepted_depth) {
                record.disposition = TELEMETRY_MTP_PROPOSAL_ACCEPTED;
            } else if (position == accepted_depth && accepted_depth < draft.size()) {
                record.disposition = TELEMETRY_MTP_PROPOSAL_FIRST_MISMATCH;
            } else {
                record.disposition = TELEMETRY_MTP_PROPOSAL_INVALIDATED_TAIL;
            }
            if (record.disposition != TELEMETRY_MTP_PROPOSAL_INVALIDATED_TAIL && position < accepted.size()) {
                record.target_selected_token_id = telemetry_request_content_enabled(slot) ? accepted[position] : LLAMA_TOKEN_NULL;
            }
            slot.telemetry_mtp_proposals_pending.push_back(record);
        }
    }

    void telemetry_record_mtp_pass(
            server_slot & slot,
            bool replay_pass,
            bool discarded,
            uint64_t target_rows_evaluated,
            uint64_t committed_output_start_ordinal,
            uint64_t committed_token_count) {
        if (!telemetry_output_token_request_enabled(slot)) {
            slot.telemetry_mtp_proposals_pending.clear();
            return;
        }
        if (slot.telemetry_mtp_passes.size() >= telemetry_mtp_pass_limit) {
            if (!discarded) {
                slot.telemetry_mtp_proposals_pending.clear();
            }
            return;
        }

        telemetry_mtp_pass_record record;
        record.logical_step = slot.telemetry_spec_logical_step;
        record.actual_target_pass = slot.n_spec_target_passes > 0 ? slot.n_spec_target_passes - 1 : 0;
        record.replay_of_actual_target_pass = replay_pass && record.actual_target_pass > 0
            ? (int64_t) record.actual_target_pass - 1
            : -1;
        record.proposed_count = slot.telemetry_spec_proposed_count;
        record.accepted_depth = slot.telemetry_spec_accepted_depth;
        record.target_rows_evaluated = target_rows_evaluated;
        record.committed_output_start_ordinal = committed_output_start_ordinal;
        record.committed_token_count = committed_token_count;
        record.reached_rejected_token_count = record.accepted_depth < record.proposed_count ? 1 : 0;
        record.invalidated_token_count = record.proposed_count
            - record.accepted_depth
            - record.reached_rejected_token_count;
        record.outcome = record.accepted_depth == 0
            ? TELEMETRY_MTP_PASS_ZERO_ACCEPTANCE
            : record.accepted_depth == record.proposed_count
                ? TELEMETRY_MTP_PASS_FULL_ACCEPTANCE
                : TELEMETRY_MTP_PASS_PARTIAL_ACCEPTANCE;
        record.replay_pass = replay_pass;
        record.discarded = discarded;
        record.counts_as_logical_step = !discarded;
        if (!discarded) {
            for (telemetry_mtp_proposal_record & proposal : slot.telemetry_mtp_proposals_pending) {
                if (proposal.disposition != TELEMETRY_MTP_PROPOSAL_INVALIDATED_TAIL &&
                        proposal.position < committed_token_count) {
                    proposal.committed_output_ordinal = committed_output_start_ordinal + proposal.position;
                }
            }
            record.proposals = std::move(slot.telemetry_mtp_proposals_pending);
            slot.telemetry_mtp_proposals_captured += record.proposals.size();
        }
        slot.telemetry_mtp_passes.push_back(std::move(record));
    }

    static size_t telemetry_json_size(const json & value) {
        return value.dump().size();
    }

    static std::string telemetry_piece_base64(const std::string & piece) {
        return base64::encode(piece.data(), piece.size());
    }

    std::string telemetry_token_piece_base64(const server_slot & slot, llama_token token) const {
        if (!telemetry_request_content_enabled(slot) || token == LLAMA_TOKEN_NULL) {
            return {};
        }
        const std::string piece = common_token_to_piece(slot.ctx_tgt, token, true);
        return telemetry_piece_base64(piece);
    }

    void telemetry_store_token_candidate_detail(server_slot & slot) {
        if (!slot.task->params.output_token_telemetry) {
            return;
        }
        if (!slot.task->params.output_token_candidate_telemetry) {
            return;
        }

        const size_t request_cap = telemetry_token_candidate_request_cap(slot);
        json decisions = json::array();
        for (const auto & record : slot.telemetry_token_candidate_decisions) {
            json candidates = json::array();
            size_t rank = 1;
            for (const auto & candidate : record.target_candidates) {
                const bool identity_available = telemetry_request_content_enabled(slot) && candidate.token_id != LLAMA_TOKEN_NULL;
                const std::string token_piece_base64 = identity_available
                    ? telemetry_token_piece_base64(slot, candidate.token_id)
                    : std::string();
                const bool probability_available = std::isfinite(candidate.log_probability_ln);
                candidates.push_back({
                    {"rank", rank++},
                    {"token_id", identity_available ? json(candidate.token_id) : json(nullptr)},
                    {"token_identity_state", identity_available ? "available" : "not_captured"},
                    {"token_identity_reason", identity_available
                        ? "content_capture_enabled"
                        : "content_capture_disabled; candidate token IDs are treated as response content"},
                    {"token_piece_base64", identity_available ? json(token_piece_base64) : json(nullptr)},
                    {"token_piece_state", identity_available ? "available" : "not_captured"},
                    {"token_piece_reason", identity_available
                        ? "authoritative_target_tokenizer_piece_bytes"
                        : "content_capture_disabled; tokenizer pieces are treated as response content"},
                    {"log_probability_ln", probability_available ? json(candidate.log_probability_ln) : json(nullptr)},
                    {"probability_state", probability_available ? "available" : "unavailable"},
                    {"probability_reason", probability_available
                        ? "raw_target_model_pre_sampler_probability"
                        : "raw_target_probability_not_finite"},
                    {"target_selected", candidate.target_selected},
                    {"draft_proposed", candidate.draft_proposed},
                });
            }
            const char * kind = record.kind == TELEMETRY_TOKEN_CANDIDATE_ACCEPTED
                ? "accepted_draft_position"
                : record.kind == TELEMETRY_TOKEN_CANDIDATE_TARGET_BONUS
                    ? "target_bonus"
                    : "first_mismatch";
            const bool draft_identity_available = telemetry_request_content_enabled(slot) && record.draft_token_id != LLAMA_TOKEN_NULL;
            const bool selected_identity_available = telemetry_request_content_enabled(slot) && record.target_selected_token_id != LLAMA_TOKEN_NULL;
            const std::string draft_piece_base64 = draft_identity_available
                ? telemetry_token_piece_base64(slot, record.draft_token_id)
                : std::string();
            const std::string selected_piece_base64 = selected_identity_available
                ? telemetry_token_piece_base64(slot, record.target_selected_token_id)
                : std::string();
            decisions.push_back({
                {"logical_step", record.logical_step},
                {"actual_target_pass", record.actual_target_pass},
                {"proposal_position", record.proposal_position},
                {"related_output_ordinal", record.related_output_ordinal >= 0 ? json(record.related_output_ordinal) : json(nullptr)},
                {"decision_kind", kind},
                {"draft_token_id", draft_identity_available ? json(record.draft_token_id) : json(nullptr)},
                {"draft_token_identity_state", draft_identity_available ? "available" : record.kind == TELEMETRY_TOKEN_CANDIDATE_TARGET_BONUS ? "not_applicable" : "not_captured"},
                {"draft_token_identity_reason", draft_identity_available
                    ? "content_capture_enabled"
                    : record.kind == TELEMETRY_TOKEN_CANDIDATE_TARGET_BONUS
                        ? "A target bonus has no draft-proposed token at this position."
                        : "content_capture_disabled; candidate token IDs are treated as response content"},
                {"draft_token_piece_base64", draft_identity_available ? json(draft_piece_base64) : json(nullptr)},
                {"draft_token_piece_state", draft_identity_available ? "available" : record.kind == TELEMETRY_TOKEN_CANDIDATE_TARGET_BONUS ? "not_applicable" : "not_captured"},
                {"draft_token_piece_reason", draft_identity_available
                    ? "authoritative_target_tokenizer_piece_bytes"
                    : record.kind == TELEMETRY_TOKEN_CANDIDATE_TARGET_BONUS
                        ? "A target bonus has no draft-proposed tokenizer piece at this position."
                        : "content_capture_disabled; tokenizer pieces are treated as response content"},
                {"target_selected_token_id", selected_identity_available ? json(record.target_selected_token_id) : json(nullptr)},
                {"target_selected_token_identity_state", selected_identity_available ? "available" : "not_captured"},
                {"target_selected_token_identity_reason", selected_identity_available
                    ? "content_capture_enabled"
                    : "content_capture_disabled; candidate token IDs are treated as response content"},
                {"target_selected_token_piece_base64", selected_identity_available ? json(selected_piece_base64) : json(nullptr)},
                {"target_selected_token_piece_state", selected_identity_available ? "available" : "not_captured"},
                {"target_selected_token_piece_reason", selected_identity_available
                    ? "authoritative_target_tokenizer_piece_bytes"
                    : "content_capture_disabled; tokenizer pieces are treated as response content"},
                {"target_probability_state", record.probability_state},
                {"target_probability_reason", record.probability_reason},
                {"target_candidates", std::move(candidates)},
                {"draft_candidates_state", "unsupported"},
                {"draft_candidates_reason", "The current producer tier retains the draft proposal identity but does not retain a draft-model top-K distribution."},
                {"draft_candidates", json::array()},
            });
        }

        const size_t eligible = slot.telemetry_token_candidate_eligible_count;
        auto build_block = [&](const json & retained, size_t stored_bytes) {
            const size_t dropped = eligible - retained.size();
            const char * state = eligible == 0 ? "no_data" : dropped > 0 ? "truncated" : "available";
            const char * reason = eligible == 0
                ? "request_produced_no_eligible_mtp_candidate_positions"
                : dropped > 0
                    ? "per_request_candidate_detail_byte_cap_reached"
                    : "all_eligible_candidate_decisions_retained";
            return json {
                {"schema_version", 2},
                {"server_instance_id", telemetry_server_instance_id},
                {"trace_id", slot.task->trace_id},
                {"state", state},
                {"reason", reason},
                {"storage_kind", "external_bounded_detail_block"},
                {"population", "mtp_mismatch_replacement_and_bonus_positions_plus_opted_in_accepted_positions"},
                {"target_probability_semantics", "raw_target_model_pre_sampler_top_k_from_one_full_vocabulary_normalization"},
                {"target_top_k", slot.task->params.output_token_candidate_top_k},
                {"include_accepted_positions", slot.task->params.output_token_candidate_include_accepted},
                {"draft_candidates_state", "unsupported"},
                {"draft_candidates_reason", "Draft-model top-K capture is not implemented in this producer tier."},
                {"token_identity_state", telemetry_request_content_enabled(slot) ? "available" : "not_captured"},
                {"token_identity_reason", telemetry_request_content_enabled(slot) ? "content_capture_enabled" : "content_capture_disabled; token IDs are treated as response content"},
                {"token_piece_state", telemetry_request_content_enabled(slot) ? "available" : "not_captured"},
                {"token_piece_reason", telemetry_request_content_enabled(slot) ? "authoritative_target_tokenizer_piece_bytes" : "content_capture_disabled; tokenizer pieces are treated as response content"},
                {"eligible_decisions", eligible},
                {"captured_decisions", retained.size()},
                {"dropped_decisions", dropped},
                {"byte_cap", request_cap},
                {"stored_bytes", stored_bytes},
                {"decisions", retained},
            };
        };

        json block = build_block(decisions, 0);
        while (!decisions.empty() && telemetry_json_size(block) > request_cap) {
            decisions.erase(decisions.size() - 1);
            block = build_block(decisions, 0);
        }
        for (size_t iteration = 0; iteration < 16; ++iteration) {
            const size_t bytes = telemetry_json_size(block);
            if (block.at("stored_bytes").get<size_t>() == bytes) {
                break;
            }
            block["stored_bytes"] = bytes;
        }
        const size_t bytes = telemetry_json_size(block);
        if (bytes > request_cap) {
            slot.telemetry_token_candidate_state = "dropped";
            slot.telemetry_token_candidate_reason = "candidate_detail_envelope_exceeded_normalized_byte_cap";
            slot.telemetry_token_candidate_stored_bytes = 0;
            slot.telemetry_token_candidate_dropped_count = eligible;
            return;
        }

        for (auto it = telemetry_token_candidate_blocks.begin(); it != telemetry_token_candidate_blocks.end(); ++it) {
            if (it->trace_id == slot.task->trace_id) {
                telemetry_token_candidate_retained_bytes -= it->bytes;
                telemetry_token_candidate_blocks.erase(it);
                break;
            }
        }
        telemetry_token_candidate_retained_bytes += bytes;
        telemetry_token_candidate_blocks.push_back({slot.task->trace_id, block, bytes});
        while (telemetry_token_candidate_blocks.size() > TELEMETRY_TOKEN_CANDIDATE_BLOCK_CAPACITY ||
                telemetry_token_candidate_retained_bytes > TELEMETRY_TOKEN_CANDIDATE_RETAINED_MAX_BYTES) {
            telemetry_token_candidate_retained_bytes -= telemetry_token_candidate_blocks.front().bytes;
            telemetry_token_candidate_expired_trace_ids.push_back(telemetry_token_candidate_blocks.front().trace_id);
            telemetry_token_candidate_blocks.pop_front();
            while (telemetry_token_candidate_expired_trace_ids.size() > TELEMETRY_TOKEN_CANDIDATE_BLOCK_CAPACITY * 4) {
                telemetry_token_candidate_expired_trace_ids.pop_front();
            }
        }

        slot.telemetry_token_candidate_state = eligible == 0
            ? "no_data"
            : decisions.size() < eligible ? "truncated" : "available";
        slot.telemetry_token_candidate_reason = eligible == 0
            ? "request_produced_no_eligible_mtp_candidate_positions"
            : decisions.size() < eligible
                ? "per_request_candidate_detail_byte_cap_reached"
                : "all_eligible_candidate_decisions_retained";
        slot.telemetry_token_candidate_stored_bytes = bytes;
        slot.telemetry_token_candidate_eligible_count = eligible;
        slot.telemetry_token_candidate_dropped_count = eligible - decisions.size();
    }

    json telemetry_token_candidates_json(const std::string & trace_id) const {
        for (auto it = telemetry_token_candidate_blocks.rbegin(); it != telemetry_token_candidate_blocks.rend(); ++it) {
            if (it->trace_id == trace_id) {
                return it->data;
            }
        }
        if (std::find(
                telemetry_token_candidate_expired_trace_ids.begin(),
                telemetry_token_candidate_expired_trace_ids.end(),
                trace_id) != telemetry_token_candidate_expired_trace_ids.end()) {
            return {
                {"schema_version", 2},
                {"server_instance_id", telemetry_server_instance_id},
                {"trace_id", trace_id},
                {"state", "expired"},
                {"reason", "The bounded in-memory candidate-detail retention ring evicted this trace."},
                {"token_identity_state", "expired"},
                {"token_identity_reason", "The content-gated candidate identities expired with the detail block."},
                {"token_piece_state", "expired"},
                {"token_piece_reason", "The content-gated tokenizer pieces expired with the detail block."},
                {"decisions", json::array()},
            };
        }
        return {
            {"schema_version", 2},
            {"server_instance_id", telemetry_server_instance_id},
            {"trace_id", trace_id},
            {"state", "not_captured"},
            {"reason", "No external candidate-detail block is retained for this exact trace ID."},
            {"token_identity_state", "not_captured"},
            {"token_identity_reason", "No retained candidate-detail block contains token identities for this exact trace ID."},
            {"token_piece_state", "not_captured"},
            {"token_piece_reason", "No retained candidate-detail block contains tokenizer pieces for this exact trace ID."},
            {"decisions", json::array()},
        };
    }

    json telemetry_output_token_json(const server_slot & slot) const {
        if (!slot.task->params.output_token_telemetry) {
            return {
                {"schema_version", 3},
                {"mtp_pass_record_schema_version", 2},
                {"state", "not_enabled_for_request"},
                {"reason", "request_did_not_set_output_token_telemetry=true"},
                {"population", "committed_generation_tokens"},
                {"total_committed_tokens", slot.stats.n_gen},
                {"captured_tokens", 0},
                {"scored_tokens", 0},
                {"maximum_captured_tokens", telemetry_output_token_limit},
                {"dropped_tokens", slot.stats.n_gen},
                {"probability_state", "not_enabled_for_request"},
                {"probability_reason", "output_token_telemetry_not_enabled_for_request"},
                {"token_identity_state", telemetry_request_content_enabled(slot) ? "available" : "not_captured"},
                {"token_identity_reason", telemetry_request_content_enabled(slot) ? "content_capture_enabled" : "content_capture_disabled"},
                {"token_piece_state", "not_enabled_for_request"},
                {"token_piece_reason", "request_did_not_set_output_token_telemetry=true"},
                {"mtp_pass_state", "not_enabled_for_request"},
                {"mtp_pass_reason", "request_did_not_set_output_token_telemetry=true"},
                {"mtp_passes_total", slot.n_spec_target_passes},
                {"mtp_passes_captured", 0},
                {"mtp_passes_dropped", 0},
                {"mtp_proposal_state", "not_enabled_for_request"},
                {"mtp_proposal_reason", "request_did_not_set_output_token_telemetry=true"},
                {"mtp_proposals_total", slot.stats.n_draft_tokens},
                {"mtp_proposals_captured", 0},
                {"maximum_captured_mtp_passes", telemetry_mtp_pass_limit},
                {"maximum_captured_mtp_proposals", telemetry_mtp_proposal_limit},
                {"mtp_proposals_dropped", slot.stats.n_draft_tokens},
                {"candidate_detail_state", "not_enabled_for_request"},
                {"candidate_detail_reason", "candidate detail requires output_token_telemetry=true"},
                {"candidate_detail_key", slot.task->trace_id},
                {"candidate_detail_eligible_decisions", 0},
                {"candidate_detail_dropped_decisions", 0},
                {"candidate_detail_stored_bytes", 0},
                {"mtp_pass_records", json::array()},
                {"records", json::array()},
            };
        }

        const char * candidate_state = !slot.task->params.output_token_candidate_telemetry
            ? "not_enabled_for_request"
            : slot.telemetry_token_candidate_state;
        const char * candidate_reason = !slot.task->params.output_token_candidate_telemetry
            ? "request_did_not_set_output_token_candidate_telemetry=true"
            : slot.telemetry_token_candidate_reason;

        const uint64_t total = slot.stats.n_gen;
        const uint64_t captured = slot.telemetry_output_tokens.size();
        const uint64_t scored = std::count_if(
            slot.telemetry_output_tokens.begin(),
            slot.telemetry_output_tokens.end(),
            [](const telemetry_output_token_record & record) {
                return std::isfinite(record.selected_log_probability_ln);
            });
        const bool truncated = captured < total;
        const char * state = total == 0 ? "no_data" : truncated ? "truncated" : "available";
        const char * reason = total == 0
            ? "request_committed_no_generation_tokens"
            : truncated
                ? "per_request_output_token_limit_reached"
                : "all_committed_generation_tokens_captured";
        const char * probability_state = slot.task->params.sampling.n_probs == 0
            ? "not_enabled_for_request"
            : scored == captured
                ? "available"
                : scored > 0
                    ? "partial"
                    : "unavailable";
        const char * probability_reason = slot.task->params.sampling.n_probs == 0
            ? "enable n_probs > 0 to retain raw target selected-token log probability"
            : scored == captured
                ? "raw target selected-token log probability retained for every captured token"
                : "raw target selected-token log probability was unavailable for one or more captured tokens";

        json records = json::array();
        for (const telemetry_output_token_record & record : slot.telemetry_output_tokens) {
            const bool probability_available = std::isfinite(record.selected_log_probability_ln);
            const bool identity_available = record.token_id != LLAMA_TOKEN_NULL;
            const bool piece_available = telemetry_request_content_enabled(slot) && identity_available;
            const std::string piece_base64 = piece_available
                ? telemetry_piece_base64(record.token_piece)
                : std::string();
            const bool mtp_linkage_available = record.origin != TELEMETRY_OUTPUT_TOKEN_ORIGIN_NORMAL_DECODE;
            const char * origin = record.origin == TELEMETRY_OUTPUT_TOKEN_ORIGIN_MTP_ACCEPTED
                ? "mtp_accepted"
                : record.origin == TELEMETRY_OUTPUT_TOKEN_ORIGIN_TARGET_AFTER_MISS
                    ? "target_after_miss"
                    : record.origin == TELEMETRY_OUTPUT_TOKEN_ORIGIN_TARGET_BONUS
                        ? "target_bonus"
                        : "normal_decode";
            const std::string probability_reason = probability_available
                ? "raw_target_model_pre_sampler_selected_token_probability"
                : slot.task->params.sampling.n_probs == 0
                    ? "request_did_not_enable_logprobs"
                    : slot.response_probability.unavailable_reason.empty()
                        ? "raw_target_selected_probability_not_retained"
                        : slot.response_probability.unavailable_reason;
            records.push_back({
                {"ordinal", record.ordinal},
                {"model_ready_offset_us", record.model_ready_offset_us},
                {"model_ready_monotonic_us", record.model_ready_monotonic_us},
                {"model_position", record.model_position >= 0 ? json(record.model_position) : json(nullptr)},
                {"model_position_state", record.model_position >= 0 ? "available" : "unavailable"},
                {"model_position_reason", record.model_position >= 0
                    ? "Position of the committed token in the target-model context, derived directly from the evaluated logits-row position plus one."
                    : "The target-model context position was unavailable at commit time."},
                {"selected_log_probability_ln", probability_available ? json(record.selected_log_probability_ln) : json(nullptr)},
                {"probability_state", probability_available ? "available" : slot.task->params.sampling.n_probs == 0 ? "not_enabled_for_request" : "unavailable"},
                {"probability_reason", probability_reason},
                {"token_id", identity_available ? json(record.token_id) : json(nullptr)},
                {"token_identity_state", identity_available ? "available" : "not_captured"},
                {"token_identity_reason", identity_available ? "content_capture_enabled" : "content_capture_disabled; token IDs are treated as response content"},
                {"token_piece_base64", piece_available ? json(piece_base64) : json(nullptr)},
                {"token_piece_state", piece_available ? "available" : "not_captured"},
                {"token_piece_reason", piece_available
                    ? "authoritative_tokenizer_piece_bytes_reused_from_committed_server_output"
                    : "content_capture_disabled; tokenizer pieces are treated as response content"},
                {"origin", origin},
                {"origin_state", "available"},
                {"origin_reason", mtp_linkage_available
                    ? "Classified by llama.cpp/llama-server while committing the speculative verification result."
                    : "The token was committed by ordinary target-model decoding."},
                {"logical_step", mtp_linkage_available ? json(record.logical_step) : json(nullptr)},
                {"actual_target_pass", mtp_linkage_available ? json(record.actual_target_pass) : json(nullptr)},
                {"proposal_position", mtp_linkage_available ? json(record.proposal_position) : json(nullptr)},
                {"accepted_depth", mtp_linkage_available ? json(record.accepted_depth) : json(nullptr)},
                {"proposed_count", mtp_linkage_available ? json(record.proposed_count) : json(nullptr)},
                {"replay_pass", mtp_linkage_available ? json(record.replay_pass) : json(nullptr)},
                {"mtp_linkage_state", mtp_linkage_available ? "available" : "not_applicable"},
                {"mtp_linkage_reason", mtp_linkage_available
                    ? "Zero-based logical-step, actual-target-pass, and proposal-position linkage retained at commit time."
                    : "MTP linkage does not apply to an ordinary target-model decode token."},
            });
        }

        const uint64_t mtp_pass_total = slot.n_spec_target_passes;
        const uint64_t mtp_pass_captured = slot.telemetry_mtp_passes.size();
        const bool mtp_pass_truncated = mtp_pass_captured < mtp_pass_total;
        const char * mtp_pass_state = !slot.can_speculate()
            ? "not_applicable"
            : mtp_pass_total == 0
                ? "no_data"
                : mtp_pass_truncated
                    ? "truncated"
                    : "available";
        const char * mtp_pass_reason = !slot.can_speculate()
            ? "The request used ordinary target-model decoding, so MTP pass detail does not apply."
            : mtp_pass_total == 0
                ? "Speculative decoding was configured, but this request completed without a target verification pass."
                : mtp_pass_truncated
                    ? "The per-request MTP pass-detail cap was reached; later actual target passes were not retained."
                    : "Every successful target verification pass, including discarded replay work, was retained.";
        const uint64_t mtp_proposal_total = slot.stats.n_draft_tokens;
        const uint64_t mtp_proposal_captured = slot.telemetry_mtp_proposals_captured;
        const bool mtp_proposal_truncated = mtp_proposal_captured < mtp_proposal_total;
        const char * mtp_proposal_state = !slot.can_speculate()
            ? "not_applicable"
            : mtp_proposal_total == 0
                ? "no_data"
                : mtp_proposal_truncated
                    ? "truncated"
                    : "available";
        const char * mtp_proposal_reason = !slot.can_speculate()
            ? "The request used ordinary target-model decoding, so an MTP proposal ledger does not apply."
            : mtp_proposal_total == 0
                ? "Speculative decoding was configured, but this request produced no draft proposals."
                : mtp_proposal_truncated
                    ? "The per-request MTP proposal cap was reached; later draft positions were not retained."
                    : "Every logical MTP draft position was retained with its producer-observed disposition.";
        json mtp_pass_records = json::array();
        for (const telemetry_mtp_pass_record & record : slot.telemetry_mtp_passes) {
            const char * outcome = record.outcome == TELEMETRY_MTP_PASS_FULL_ACCEPTANCE
                ? "full_acceptance"
                : record.outcome == TELEMETRY_MTP_PASS_PARTIAL_ACCEPTANCE
                    ? "partial_acceptance"
                    : "zero_acceptance";
            json proposals = json::array();
            for (const telemetry_mtp_proposal_record & proposal : record.proposals) {
                const bool draft_identity_available = proposal.draft_token_id != LLAMA_TOKEN_NULL;
                const bool target_identity_available = proposal.target_selected_token_id != LLAMA_TOKEN_NULL;
                const bool target_selected = proposal.disposition != TELEMETRY_MTP_PROPOSAL_INVALIDATED_TAIL;
                const std::string draft_piece_base64 = draft_identity_available
                    ? telemetry_token_piece_base64(slot, proposal.draft_token_id)
                    : std::string();
                const std::string target_piece_base64 = target_identity_available
                    ? telemetry_token_piece_base64(slot, proposal.target_selected_token_id)
                    : std::string();
                const bool probability_available = std::isfinite(proposal.target_selected_log_probability_ln);
                const char * disposition = proposal.disposition == TELEMETRY_MTP_PROPOSAL_ACCEPTED
                    ? "accepted"
                    : proposal.disposition == TELEMETRY_MTP_PROPOSAL_FIRST_MISMATCH
                        ? "first_mismatch"
                        : "invalidated_tail";
                const std::string probability_reason = probability_available
                    ? "raw_target_model_pre_sampler_selected_token_probability"
                    : !target_selected
                        ? "position_was_not_evaluated_after_first_mismatch"
                        : slot.task->params.sampling.n_probs == 0
                            ? "request_did_not_enable_logprobs"
                            : slot.response_probability.unavailable_reason.empty()
                                ? "raw_target_selected_probability_not_retained"
                                : slot.response_probability.unavailable_reason;
                proposals.push_back({
                    {"position", proposal.position},
                    {"evaluated_actual_target_pass", proposal.evaluated_actual_target_pass},
                    {"disposition", disposition},
                    {"draft_token_id", draft_identity_available ? json(proposal.draft_token_id) : json(nullptr)},
                    {"draft_token_identity_state", draft_identity_available ? "available" : "not_captured"},
                    {"draft_token_identity_reason", draft_identity_available ? "content_capture_enabled" : "content_capture_disabled; token IDs are treated as response content"},
                    {"draft_token_piece_base64", draft_identity_available ? json(draft_piece_base64) : json(nullptr)},
                    {"draft_token_piece_state", draft_identity_available ? "available" : "not_captured"},
                    {"draft_token_piece_reason", draft_identity_available ? "authoritative_target_tokenizer_piece_bytes" : "content_capture_disabled; tokenizer pieces are treated as response content"},
                    {"target_selected_token_id", target_identity_available ? json(proposal.target_selected_token_id) : json(nullptr)},
                    {"target_selected_token_identity_state", !target_selected ? "not_applicable" : target_identity_available ? "available" : "not_captured"},
                    {"target_selected_token_identity_reason", !target_selected
                        ? "position_was_not_evaluated_after_first_mismatch"
                        : target_identity_available ? "content_capture_enabled" : "content_capture_disabled; token IDs are treated as response content"},
                    {"target_selected_token_piece_base64", target_identity_available ? json(target_piece_base64) : json(nullptr)},
                    {"target_selected_token_piece_state", !target_selected ? "not_applicable" : target_identity_available ? "available" : "not_captured"},
                    {"target_selected_token_piece_reason", !target_selected
                        ? "position_was_not_evaluated_after_first_mismatch"
                        : target_identity_available ? "authoritative_target_tokenizer_piece_bytes" : "content_capture_disabled; tokenizer pieces are treated as response content"},
                    {"target_selected_log_probability_ln", probability_available ? json(proposal.target_selected_log_probability_ln) : json(nullptr)},
                    {"target_selected_probability_state", probability_available ? "available" : !target_selected ? "not_applicable" : slot.task->params.sampling.n_probs == 0 ? "not_enabled_for_request" : "unavailable"},
                    {"target_selected_probability_reason", probability_reason},
                    {"committed_output_ordinal", proposal.committed_output_ordinal >= 0 ? json(proposal.committed_output_ordinal) : json(nullptr)},
                });
            }
            const bool proposal_complete = record.proposals.size() == record.proposed_count;
            const char * proposal_state = record.discarded
                ? "not_applicable"
                : proposal_complete
                    ? "available"
                    : "truncated";
            const char * proposal_reason = record.discarded
                ? "The logical proposal ledger is attached to the committing replay pass to avoid duplicate rows."
                : proposal_complete
                    ? "Every draft position in this logical decision was retained."
                    : "The per-request MTP proposal cap truncated this logical decision.";
            mtp_pass_records.push_back({
                {"logical_step", record.logical_step},
                {"actual_target_pass", record.actual_target_pass},
                {"replay_of_actual_target_pass", record.replay_of_actual_target_pass >= 0
                    ? json(record.replay_of_actual_target_pass)
                    : json(nullptr)},
                {"proposed_count", record.proposed_count},
                {"accepted_depth", record.accepted_depth},
                {"reached_rejected_tokens", record.reached_rejected_token_count},
                {"invalidated_tokens", record.invalidated_token_count},
                {"target_rows_evaluated", record.target_rows_evaluated},
                {"committed_output_start_ordinal", record.committed_token_count > 0
                    ? json(record.committed_output_start_ordinal)
                    : json(nullptr)},
                {"committed_token_count", record.committed_token_count},
                {"outcome", outcome},
                {"replay_pass", record.replay_pass},
                {"discarded", record.discarded},
                {"counts_as_logical_step", record.counts_as_logical_step},
                {"proposal_state", proposal_state},
                {"proposal_reason", proposal_reason},
                {"proposals", std::move(proposals)},
            });
        }

        return {
            {"schema_version", 3},
            {"mtp_pass_record_schema_version", 2},
            {"state", state},
            {"reason", reason},
            {"population", "committed_generation_tokens"},
            {"total_committed_tokens", total},
            {"captured_tokens", captured},
            {"scored_tokens", scored},
            {"maximum_captured_tokens", telemetry_output_token_limit},
            {"dropped_tokens", total - captured},
            {"probability_state", probability_state},
            {"probability_reason", probability_reason},
            {"token_identity_state", telemetry_request_content_enabled(slot) ? "available" : "not_captured"},
            {"token_identity_reason", telemetry_request_content_enabled(slot) ? "content_capture_enabled" : "content_capture_disabled; token IDs are treated as response content"},
            {"token_piece_state", telemetry_request_content_enabled(slot) ? "available" : "not_captured"},
            {"token_piece_reason", telemetry_request_content_enabled(slot) ? "authoritative_tokenizer_piece_bytes" : "content_capture_disabled; tokenizer pieces are treated as response content"},
            {"mtp_pass_state", mtp_pass_state},
            {"mtp_pass_reason", mtp_pass_reason},
            {"mtp_passes_total", mtp_pass_total},
            {"mtp_passes_captured", mtp_pass_captured},
            {"maximum_captured_mtp_passes", telemetry_mtp_pass_limit},
            {"mtp_passes_dropped", mtp_pass_total - mtp_pass_captured},
            {"mtp_proposal_state", mtp_proposal_state},
            {"mtp_proposal_reason", mtp_proposal_reason},
            {"mtp_proposals_total", mtp_proposal_total},
            {"mtp_proposals_captured", mtp_proposal_captured},
            {"maximum_captured_mtp_proposals", telemetry_mtp_proposal_limit},
            {"mtp_proposals_dropped", mtp_proposal_total - mtp_proposal_captured},
            {"candidate_detail_state", candidate_state},
            {"candidate_detail_reason", candidate_reason},
            {"candidate_detail_key", slot.task->trace_id},
            {"candidate_detail_eligible_decisions", slot.telemetry_token_candidate_eligible_count},
            {"candidate_detail_dropped_decisions", slot.telemetry_token_candidate_dropped_count},
            {"candidate_detail_stored_bytes", slot.telemetry_token_candidate_stored_bytes},
            {"mtp_pass_records", std::move(mtp_pass_records)},
            {"records", std::move(records)},
        };
    }

    json telemetry_moe_routing_json(const server_slot & slot) const {
        const int32_t configured_experts = llama_model_n_expert(model_tgt);
        const int32_t experts_per_token = llama_model_n_expert_used(model_tgt);
        const int32_t model_layer_count = llama_model_n_layer(model_tgt);
        const int32_t configured_moe_layer_count = llama_model_n_moe_layer(model_tgt);
        const bool configured_topology_available = configured_experts > 0
            && model_layer_count > 0 && configured_moe_layer_count > 0;
        const auto base = [&]() {
            const bool routed_model = configured_experts > 0;
            json moe_layer_indices = nullptr;
            if (configured_topology_available) {
                moe_layer_indices = json::array();
                int32_t previous_layer_index = -1;
                for (int32_t index = 0; index < configured_moe_layer_count; ++index) {
                    const int32_t layer_index = llama_model_moe_layer_index(model_tgt, index);
                    GGML_ASSERT(layer_index > previous_layer_index && layer_index < model_layer_count);
                    moe_layer_indices.push_back(layer_index);
                    previous_layer_index = layer_index;
                }
            }
            return json {
                {"schema_version", 2},
                {"token_detail_schema_version", 2},
                {"configuration_state", routed_model ? "available" : "not_applicable"},
                {"configuration_reason", routed_model
                    ? "The loaded target model exposes routed-expert configuration metadata."
                    : "The loaded target model is dense and has no routed-expert configuration."},
                {"configured_experts", routed_model ? json(configured_experts) : json(nullptr)},
                {"experts_per_token", experts_per_token > 0 ? json(experts_per_token) : json(nullptr)},
                {"model_layer_count", configured_topology_available ? json(model_layer_count) : json(nullptr)},
                {"moe_layer_indices", std::move(moe_layer_indices)},
                {"moe_layers", nullptr},
                {"routed_tokens", nullptr},
                {"routed_token_layer_decisions", nullptr},
                {"expert_activations_total", nullptr},
                {"expert_activations_captured", 0},
                {"maximum_captured_activations", telemetry_moe_activation_limit},
                {"dropped_activations", nullptr},
                {"invalid_activations", 0},
                {"cap_dropped_activations", 0},
                {"population", "target_model_routed_token_layer_decisions"},
                {"expert_activations", json::array()},
                {"token_detail_state", routed_model ? "unavailable" : "not_applicable"},
                {"token_detail_reason", routed_model
                    ? "No exact target-model output-row routing decision has been evaluated for this request yet."
                    : "Exact-token expert routing does not apply to a dense target model."},
                {"token_detail_population", "target_model_output_logit_rows_by_layer"},
                {"token_detail_decisions_total", nullptr},
                {"token_detail_decisions_captured", 0},
                {"token_detail_decisions_dropped", nullptr},
                {"token_detail_invalid_decisions", 0},
                {"token_detail_cap_dropped_decisions", 0},
                {"token_detail_activations_total", nullptr},
                {"token_detail_activations_captured", 0},
                {"token_detail_activations_dropped", nullptr},
                {"token_detail_invalid_activations", 0},
                {"token_detail_cap_dropped_activations", 0},
                {"maximum_captured_token_detail_activations", telemetry_moe_activation_limit},
                {"selected_expert_ids_state", routed_model ? "unavailable" : "not_applicable"},
                {"selected_expert_ids_reason", routed_model
                    ? "No exact target-model output-row routing decision has been evaluated for this request yet."
                    : "Selected routed-expert IDs do not apply to a dense target model."},
                {"routing_weights_state", routed_model ? "unavailable" : "not_applicable"},
                {"routing_weights_reason", routed_model
                    ? "No exact target-model output-row routing weights have been evaluated for this request yet."
                    : "Router weights do not apply to a dense target model."},
                {"router_margin_state", routed_model && experts_per_token >= 2 ? "unavailable" : "not_applicable"},
                {"router_margin_reason", routed_model && experts_per_token >= 2
                    ? "No exact target-model output-row routing margin has been evaluated for this request yet."
                    : routed_model
                        ? "A top-two router margin does not apply when the model selects fewer than two experts per token-layer decision."
                        : "Router margin does not apply to a dense target model."},
                {"token_decisions", json::array()},
            };
        };
        const auto set_token_detail_state = [](
                json & result,
                const char * state,
                const std::string & reason) {
            result["token_detail_state"] = state;
            result["token_detail_reason"] = reason;
            result["selected_expert_ids_state"] = state;
            result["selected_expert_ids_reason"] = reason;
        };
        const auto set_weight_state = [experts_per_token](
                json & result,
                const char * state,
                const std::string & reason) {
            result["routing_weights_state"] = state;
            result["routing_weights_reason"] = reason;
            result["router_margin_state"] = experts_per_token >= 2 ? state : "not_applicable";
            result["router_margin_reason"] = experts_per_token >= 2
                ? reason
                : "A top-two router margin does not apply when the model selects fewer than two experts per token-layer decision.";
        };

        if (configured_experts <= 0) {
            json result = base();
            result["state"] = "not_applicable";
            result["reason"] = "The loaded target model has no routed MoE experts.";
            return result;
        }
        if (!slot.task || (!slot.task->params.moe_routing_telemetry && !slot.telemetry_moe_chunk_capture_started)) {
            json result = base();
            result["state"] = "not_enabled_for_request";
            result["reason"] = "request_did_not_set_moe_routing_telemetry=true";
            set_token_detail_state(result, "not_enabled_for_request", "request_did_not_set_moe_routing_telemetry=true");
            set_weight_state(result, "not_enabled_for_request", "request_did_not_set_moe_routing_telemetry=true");
            return result;
        }
        if (slot.telemetry_moe_histogram_counts.total == 0) {
            json result = base();
            result["state"] = "no_data";
            result["reason"] = "The opted-in request completed without a retained routed-expert decision.";
            result["moe_layers"] = 0;
            result["routed_tokens"] = 0;
            result["routed_token_layer_decisions"] = 0;
            result["expert_activations_total"] = 0;
            result["dropped_activations"] = 0;
            set_token_detail_state(result, "no_data", "The opted-in request completed without a target-model output-row routed-expert decision.");
            result["token_detail_decisions_total"] = 0;
            result["token_detail_decisions_dropped"] = 0;
            result["token_detail_activations_total"] = 0;
            result["token_detail_activations_dropped"] = 0;
            set_weight_state(result, "no_data", "The opted-in request completed without an exact target-model output-row routing decision.");
            return result;
        }

        const bool truncated = server_moe_routing_was_truncated(slot.telemetry_moe_histogram_counts);
        const bool invalid = slot.telemetry_moe_histogram_counts.invalid > 0;
        json activations = json::array();
        std::vector<uint64_t> layer_totals((size_t) llama_model_n_layer(model_tgt), 0);
        for (size_t index = 0; index < slot.telemetry_moe_expert_activations.size(); ++index) {
            layer_totals[index/(size_t) configured_experts] += slot.telemetry_moe_expert_activations[index];
        }
        for (size_t index = 0; index < slot.telemetry_moe_expert_activations.size(); ++index) {
            const uint64_t count = slot.telemetry_moe_expert_activations[index];
            if (count == 0) {
                continue;
            }
            const size_t layer = index/(size_t) configured_experts;
            const size_t expert = index%(size_t) configured_experts;
            activations.push_back({
                {"layer_index", layer},
                {"expert_index", expert},
                {"activation_count", count},
                {"share_percent", slot.telemetry_moe_histogram_counts.captured > 0
                    ? json(100.0*count/slot.telemetry_moe_histogram_counts.captured)
                    : json(nullptr)},
                {"layer_share_percent", layer_totals[layer] > 0
                    ? json(100.0*count/layer_totals[layer])
                    : json(nullptr)},
            });
        }

        json result = base();
        result["state"] = server_moe_routing_capture_state(slot.telemetry_moe_histogram_counts, true);
        result["reason"] = truncated && invalid
            ? "The per-request routed-expert activation cap was reached, and malformed records were excluded separately; the histogram covers only valid retained-prefix records."
            : truncated
            ? "The per-request routed-expert activation cap was reached; the histogram covers only the retained prefix."
            : invalid
                ? "One or more routed-expert activation records were malformed and were excluded from the histogram; no valid activation was omitted because of the cap."
            : "All selected routed-expert IDs produced for this request were retained.";
        result["moe_layers"] = slot.telemetry_moe_layers.size();
        result["routed_tokens"] = slot.telemetry_moe_routed_tokens;
        result["routed_token_layer_decisions"] = slot.telemetry_moe_routed_token_layers;
        result["expert_activations_total"] = slot.telemetry_moe_histogram_counts.total;
        result["expert_activations_captured"] = slot.telemetry_moe_histogram_counts.captured;
        result["dropped_activations"] = slot.telemetry_moe_histogram_counts.total - slot.telemetry_moe_histogram_counts.captured;
        result["invalid_activations"] = slot.telemetry_moe_histogram_counts.invalid;
        result["cap_dropped_activations"] = slot.telemetry_moe_histogram_counts.cap_dropped;
        result["expert_activations"] = std::move(activations);

        const uint64_t token_activations_dropped = slot.telemetry_moe_token_detail_counts.total
            - slot.telemetry_moe_token_detail_counts.captured;
        const bool token_detail_truncated = server_moe_routing_was_truncated(slot.telemetry_moe_token_detail_counts);
        const bool token_detail_invalid = slot.telemetry_moe_token_detail_counts.invalid > 0;
        const char * token_detail_state = server_moe_routing_capture_state(
            slot.telemetry_moe_token_detail_counts,
            slot.telemetry_moe_token_decisions_total > 0);
        const std::string token_detail_reason = token_detail_truncated && token_detail_invalid
            ? "The exact-token routed-expert activation cap was reached, and malformed output rows were excluded separately; only complete valid retained-prefix decisions are present."
            : token_detail_truncated
            ? "The exact-token routed-expert activation cap was reached; only complete retained-prefix decisions are present."
            : token_detail_invalid
                ? "One or more routed-expert output rows were malformed and were excluded from exact-token detail."
                : slot.telemetry_moe_token_decisions_total > 0
                    ? "All selected expert IDs for retained target-model output rows were captured."
                    : "The request produced routed prompt rows but no target-model output-row routing decision.";
        set_token_detail_state(result, token_detail_state, token_detail_reason);
        result["token_detail_decisions_total"] = slot.telemetry_moe_token_decisions_total;
        result["token_detail_decisions_captured"] = slot.telemetry_moe_token_decisions_captured;
        result["token_detail_decisions_dropped"] = slot.telemetry_moe_token_decisions_total
            - slot.telemetry_moe_token_decisions_captured;
        result["token_detail_invalid_decisions"] = slot.telemetry_moe_token_decisions_invalid;
        result["token_detail_cap_dropped_decisions"] = slot.telemetry_moe_token_decisions_cap_dropped;
        result["token_detail_activations_total"] = slot.telemetry_moe_token_detail_counts.total;
        result["token_detail_activations_captured"] = slot.telemetry_moe_token_detail_counts.captured;
        result["token_detail_activations_dropped"] = token_activations_dropped;
        result["token_detail_invalid_activations"] = slot.telemetry_moe_token_detail_counts.invalid;
        result["token_detail_cap_dropped_activations"] = slot.telemetry_moe_token_detail_counts.cap_dropped;

        json token_decisions = json::array();
        uint64_t decisions_with_weights = 0;
        uint64_t decisions_with_margin = 0;
        size_t decision_start = 0;
        while (decision_start < slot.telemetry_moe_token_activations.size()) {
            const auto & first = slot.telemetry_moe_token_activations[decision_start];
            size_t decision_end = decision_start + 1;
            while (decision_end < slot.telemetry_moe_token_activations.size()) {
                const auto & next = slot.telemetry_moe_token_activations[decision_end];
                if (next.model_position != first.model_position
                        || next.layer_index != first.layer_index
                        || next.phase != first.phase
                        || next.logical_step != first.logical_step
                        || next.actual_target_pass != first.actual_target_pass
                        || next.proposal_position != first.proposal_position
                        || next.replay_pass != first.replay_pass) {
                    break;
                }
                decision_end++;
            }
            json selected_expert_ids = json::array();
            json effective_expert_weights = json::array();
            json normalized_expert_weight_shares = json::array();
            bool weights_available = true;
            double effective_weight_sum = 0.0;
            for (size_t i = decision_start; i < decision_end; ++i) {
                const auto & activation = slot.telemetry_moe_token_activations[i];
                selected_expert_ids.push_back(activation.expert_index);
                if (server_moe_routing_weight_is_usable(activation.effective_weight)) {
                    effective_expert_weights.push_back(activation.effective_weight);
                    effective_weight_sum += activation.effective_weight;
                } else {
                    weights_available = false;
                }
            }
            weights_available = weights_available
                && effective_expert_weights.size() == selected_expert_ids.size()
                && effective_weight_sum > 0.0;
            json normalized_router_margin = nullptr;
            if (weights_available) {
                double largest_share = -1.0;
                double second_largest_share = -1.0;
                for (const auto & weight : effective_expert_weights) {
                    const double share = weight.get<double>()/effective_weight_sum;
                    normalized_expert_weight_shares.push_back(share);
                    if (share > largest_share) {
                        second_largest_share = largest_share;
                        largest_share = share;
                    } else if (share > second_largest_share) {
                        second_largest_share = share;
                    }
                }
                decisions_with_weights++;
                if (second_largest_share >= 0.0) {
                    normalized_router_margin = largest_share - second_largest_share;
                    decisions_with_margin++;
                }
            } else {
                effective_expert_weights = json::array();
            }
            const char * phase = first.phase == TELEMETRY_MOE_TOKEN_PHASE_PREFILL_OUTPUT
                ? "prefill_output"
                : first.phase == TELEMETRY_MOE_TOKEN_PHASE_MTP_VERIFY ? "mtp_verify" : "normal_decode";
            token_decisions.push_back({
                {"model_position", first.model_position},
                {"layer_index", first.layer_index},
                {"phase", phase},
                {"logical_step", first.logical_step >= 0 ? json(first.logical_step) : json(nullptr)},
                {"actual_target_pass", first.actual_target_pass >= 0 ? json(first.actual_target_pass) : json(nullptr)},
                {"proposal_position", first.proposal_position >= 0 ? json(first.proposal_position) : json(nullptr)},
                {"replay_pass", first.replay_pass},
                {"selected_expert_ids", std::move(selected_expert_ids)},
                {"effective_expert_weights", std::move(effective_expert_weights)},
                {"normalized_expert_weight_shares", std::move(normalized_expert_weight_shares)},
                {"normalized_router_margin", std::move(normalized_router_margin)},
            });
            decision_start = decision_end;
        }

        const uint64_t captured_decisions = slot.telemetry_moe_token_decisions_captured;
        const char * weights_state = decisions_with_weights == captured_decisions && captured_decisions > 0
            ? token_detail_state
            : decisions_with_weights > 0 ? "partial" : captured_decisions > 0 ? "unavailable" : "no_data";
        const std::string weights_reason = decisions_with_weights == captured_decisions && captured_decisions > 0
            ? "llama.cpp/llama-server retained the exact effective expert coefficients and normalized selected-expert shares for every retained token-layer decision."
            : decisions_with_weights > 0
                ? "Effective expert coefficients were unavailable for one or more retained token-layer decisions; no missing value was replaced with zero."
                : "The backend did not expose usable effective expert coefficients for the retained token-layer decisions.";
        result["routing_weights_state"] = weights_state;
        result["routing_weights_reason"] = weights_reason;

        if (experts_per_token < 2) {
            result["router_margin_state"] = "not_applicable";
            result["router_margin_reason"] = "A top-two router margin does not apply when the model selects fewer than two experts per token-layer decision.";
        } else {
            const char * margin_state = decisions_with_margin == captured_decisions && captured_decisions > 0
                ? token_detail_state
                : decisions_with_margin > 0 ? "partial" : captured_decisions > 0 ? "unavailable" : "no_data";
            result["router_margin_state"] = margin_state;
            result["router_margin_reason"] = decisions_with_margin == captured_decisions && captured_decisions > 0
                ? "llama-server derived the normalized top-two selected-expert share difference from every retained decision's exact effective coefficients."
                : decisions_with_margin > 0
                    ? "A normalized top-two margin could be derived for only part of the retained token-layer population."
                    : "No retained token-layer decision contained two usable effective expert coefficients for a normalized margin.";
        }
        result["token_decisions"] = std::move(token_decisions);
        return result;
    }

    json telemetry_lifecycle_clock_json(const server_slot & slot) const {
        auto boundary = [](int64_t value) {
            return value > 0 ? json(value) : json(nullptr);
        };
        return {
            {"schema_version", 1},
            {"clock_domain", "server_process_monotonic_microseconds"},
            {"arrival_monotonic_us", boundary(slot.stats.t_arrival)},
            {"enqueue_monotonic_us", boundary(slot.stats.t_enqueue)},
            {"slot_start_monotonic_us", boundary(slot.stats.t_slot_start)},
            {"cache_start_monotonic_us", boundary(slot.stats.t_cache_start)},
            {"cache_end_monotonic_us", boundary(slot.stats.t_cache_last)},
            {"prefill_start_monotonic_us", boundary(slot.stats.t_prefill_start)},
            {"prefill_end_monotonic_us", boundary(slot.stats.t_prefill_last)},
            {"first_token_monotonic_us", boundary(slot.stats.t_first_token)},
            {"last_generation_work_monotonic_us", boundary(slot.stats.t_gen_last)},
            {"finalization_start_monotonic_us", boundary(slot.stats.t_finalization_start)},
            {"response_handoff_monotonic_us", boundary(slot.stats.t_complete)},
            {"slot_release_monotonic_us", boundary(slot.stats.t_release)},
        };
    }

    void telemetry_on_start(server_slot & slot) {
        if (telemetry_kv_pressure_active) {
            telemetry_kv_request_started(slot);
        }
        if (telemetry_output_token_request_enabled(slot)) {
            slot.telemetry_output_tokens.reserve(telemetry_output_token_limit);
            slot.telemetry_mtp_passes.reserve(telemetry_mtp_pass_limit);
        }
        if (telemetry_token_candidate_request_enabled(slot)) {
            slot.telemetry_token_candidate_decisions.reserve(
                std::min<size_t>(telemetry_token_candidate_decision_limit, 64));
        }
        telemetry_append({
            {"event", "request_started"},
            {"trace_id", slot.task->trace_id},
            {"w3c_trace_id", slot.task->correlation_trace_id.empty() ? json(nullptr) : json(slot.task->correlation_trace_id)},
            {"w3c_traceparent", slot.task->traceparent.empty() ? json(nullptr) : json(slot.task->traceparent)},
            {"task_id", slot.task->id},
            {"slot_id", slot.id},
            {"slot_assignment_ordinal", slot.telemetry_assignment_ordinal},
            {"completion_id", slot.task->params.oaicompat_cmpl_id},
            {"model", slot.task->params.oaicompat_model},
            {"server_build", std::string(llama_build_info())},
            {"timestamp_unix_ms", telemetry_unix_ms(slot, slot.stats.t_slot_start)},
            {"arrival_unix_ms", slot.stats.t_arrival_unix_ms},
            {"queue_ms", slot.stats.t_queue_ms()},
            {"prompt_tokens", slot.task->n_tokens()},
            {"server_configuration", telemetry_server_configuration()},
            {"lifecycle_clock", telemetry_lifecycle_clock_json(slot)},
        });
    }

    void telemetry_on_first_token(const server_slot & slot) {
        telemetry_append({
            {"event", "first_token"},
            {"trace_id", slot.task->trace_id},
            {"w3c_trace_id", slot.task->correlation_trace_id.empty() ? json(nullptr) : json(slot.task->correlation_trace_id)},
            {"w3c_traceparent", slot.task->traceparent.empty() ? json(nullptr) : json(slot.task->traceparent)},
            {"task_id", slot.task->id},
            {"slot_id", slot.id},
            {"slot_assignment_ordinal", slot.telemetry_assignment_ordinal},
            {"timestamp_unix_ms", telemetry_unix_ms(slot, slot.stats.t_first_token)},
            {"ttft_ms", slot.stats.t_ttft_ms()},
            {"queue_ms", slot.stats.t_queue_ms()},
            {"prompt_tokens", slot.task->n_tokens()},
            {"matched_prefix_tokens", slot.stats.n_prompt_matched},
            {"reused_prompt_tokens", slot.stats.n_prompt_cached},
            {"evaluated_prompt_tokens", slot.stats.n_prompt_processed},
            {"cache_status", slot.stats.cache_status()},
            {"cache_reuse_ratio", slot.stats.cache_reuse_ratio()},
            {"prefill_meaningful", std::strcmp(slot.stats.cache_status(), "full") != 0},
            {"cache_lookup_ms", slot.stats.t_cache_ms()},
            {"actual_prefill_ms", slot.stats.t_prefill_actual_ms()},
            {"server_configuration", telemetry_server_configuration()},
            {"lifecycle_clock", telemetry_lifecycle_clock_json(slot)},
        });
    }

    void telemetry_finalize(server_slot & slot, const char * outcome, const std::string & error = {}, const char * error_category = nullptr) {
        if (slot.telemetry_finalized) {
            return;
        }
        slot.telemetry_finalized = true;
        slot.stats.t_finalization_start = ggml_time_us();
        const char * wait_outcome = std::strcmp(outcome, "cancelled") == 0
            ? "cancelled"
            : std::strcmp(outcome, "error") == 0 ? "failed" : "resumed";
        if (telemetry_kv_pressure_active) {
            telemetry_kv_wait_finish_request(slot, wait_outcome, slot.stats.t_finalization_start);
            telemetry_kv_request_finished(slot);
        }
        telemetry_record_moe_routing_final_marker(slot, outcome);
        gpu_telemetry.record_operation(
            SERVER_GPU_OPERATION_REQUEST,
            slot.task->trace_id,
            slot.id,
            slot.stats.t_arrival,
            slot.stats.t_finalization_start,
            -1,
            -1,
            -1,
            -1,
            -1,
            SERVER_GPU_TIMING_REQUEST_LIFECYCLE_WINDOW);
        telemetry_store_token_candidate_detail(slot);

        metrics.n_requests_total++;
        if (std::strcmp(outcome, "success") == 0) {
            metrics.n_requests_success++;
        } else if (std::strcmp(outcome, "cancelled") == 0) {
            metrics.n_requests_cancelled++;
        } else {
            metrics.n_requests_error++;
        }

        if (slot.stats.t_slot_start > 0 && slot.stats.t_enqueue > 0) {
            metrics.request_queue_seconds.observe(slot.stats.t_queue_ms() / 1000.0);
        }
        if (slot.stats.t_first_token > 0) {
            metrics.request_ttft_seconds.observe(slot.stats.t_ttft_ms() / 1000.0);
        }
        if (slot.stats.t_prefill_start > 0 && slot.stats.t_prefill_last > 0) {
            metrics.request_prefill_seconds.observe(slot.stats.t_prefill_actual_ms() / 1000.0);
        }
        if (slot.stats.n_gen_steps() > 0) {
            metrics.request_tpot_seconds.observe(slot.stats.t_gen_per_token_ms() / 1000.0);
        }
        metrics.request_prompt_tokens.observe(slot.task->n_tokens());
        metrics.request_output_tokens.observe(slot.stats.n_gen);
        metrics.request_cache_reuse_ratio.observe(slot.stats.cache_reuse_ratio());
        const std::string cache_status = slot.stats.cache_status();
        if (cache_status == "full") {
            metrics.n_cache_full++;
        } else if (cache_status == "partial") {
            metrics.n_cache_partial++;
        } else {
            metrics.n_cache_miss++;
        }

        const char * finish_reason = "completed";
        if (std::strcmp(outcome, "cancelled") == 0) {
            finish_reason = "cancelled";
        } else if (std::strcmp(outcome, "error") == 0) {
            finish_reason = "error";
        } else if (slot.stop == STOP_TYPE_EOS || slot.stop == STOP_TYPE_WORD) {
            finish_reason = "stop";
        } else if (slot.stop == STOP_TYPE_LIMIT) {
            finish_reason = "length";
        }

        json sampling = slot.task->params.to_json(false);
        sampling["requested_temperature"] = slot.task->telemetry_requested_temperature;
        sampling["effective_temperature"] = slot.task->params.sampling.temp;
        sampling["effective_seed"] = slot.smpl
            ? json(common_sampler_get_seed(slot.smpl.get()))
            : json(nullptr);

        json event = {
            {"event", std::strcmp(outcome, "success") == 0 ? "request_completed" : "request_ended"},
            {"trace_id", slot.task->trace_id},
            {"w3c_trace_id", slot.task->correlation_trace_id.empty() ? json(nullptr) : json(slot.task->correlation_trace_id)},
            {"w3c_traceparent", slot.task->traceparent.empty() ? json(nullptr) : json(slot.task->traceparent)},
            {"task_id", slot.task->id},
            {"slot_id", slot.id},
            {"slot_assignment_ordinal", slot.telemetry_assignment_ordinal},
            {"timestamp_unix_ms", telemetry_unix_ms(slot, slot.stats.t_finalization_start)},
            {"completion_id", slot.task->params.oaicompat_cmpl_id},
            {"model", slot.task->params.oaicompat_model},
            {"server_build", std::string(llama_build_info())},
            {"outcome", outcome},
            {"finish_reason", finish_reason},
            {"partial_response", std::strcmp(outcome, "cancelled") == 0},
            {"error", error.empty() ? json(nullptr) : json(error)},
            {"error_category", error_category ? json(error_category) : json(nullptr)},
            {"prompt_tokens", slot.task->n_tokens()},
            {"matched_prefix_tokens", slot.stats.n_prompt_matched},
            {"reused_prompt_tokens", slot.stats.n_prompt_cached},
            {"evaluated_prompt_tokens", slot.stats.n_prompt_processed},
            {"cache_status", slot.stats.cache_status()},
            {"cache_reuse_ratio", slot.stats.cache_reuse_ratio()},
            {"prefill_meaningful", cache_status != "full"},
            {"output_tokens", slot.stats.n_gen},
            {"decode_steps", slot.stats.n_gen_steps()},
            {"timings", slot.stats.to_json()},
            {"sampling", std::move(sampling)},
            {"server_configuration", telemetry_server_configuration()},
            {"lifecycle_clock", telemetry_lifecycle_clock_json(slot)},
            {"speculative", {
                {"draft_tokens", slot.stats.n_draft_tokens},
                {"accepted_tokens", slot.stats.n_draft_accepted},
                {"rejected_tokens", slot.stats.n_draft_tokens - slot.stats.n_draft_accepted},
                {"verification_steps", slot.stats.n_draft_verif_steps},
                {"logical_target_passes", slot.stats.n_draft_verif_steps},
                {"actual_target_passes", slot.n_spec_target_passes},
                {"hit_steps", slot.n_draft_hit_steps},
                {"miss_steps", slot.stats.n_draft_verif_steps >= slot.n_draft_hit_steps ? slot.stats.n_draft_verif_steps - slot.n_draft_hit_steps : 0},
                {"full_chain_steps", slot.n_draft_full_steps},
                {"token_hit_rate", slot.stats.n_draft_tokens > 0 ? json((double) slot.stats.n_draft_accepted / slot.stats.n_draft_tokens) : json(nullptr)},
                {"step_hit_rate", slot.stats.n_draft_verif_steps > 0 ? json((double) slot.n_draft_hit_steps / slot.stats.n_draft_verif_steps) : json(nullptr)},
                {"full_chain_rate", slot.stats.n_draft_verif_steps > 0 ? json((double) slot.n_draft_full_steps / slot.stats.n_draft_verif_steps) : json(nullptr)},
                {"target_tokens", slot.n_spec_target_tokens},
                {"useful_output_tokens", slot.n_spec_useful_tokens},
                {"target_tokens_per_pass", slot.n_spec_target_passes > 0 ? json((double) slot.n_spec_target_tokens / slot.n_spec_target_passes) : json(nullptr)},
                {"useful_output_tokens_per_target_pass", slot.n_spec_target_passes > 0 ? json((double) slot.n_spec_useful_tokens / slot.n_spec_target_passes) : json(nullptr)},
                {"configuration", {
                    {"types", common_speculative_type_name_str(slot.task->params.speculative.types)},
                    {"draft_n_max", slot.task->params.speculative.draft.n_max},
                    {"draft_n_min", slot.task->params.speculative.draft.n_min},
                    {"draft_p_split", slot.task->params.speculative.draft.p_split},
                    {"draft_p_min", slot.task->params.speculative.draft.p_min},
                    {"ngram_mod_n_match", slot.task->params.speculative.ngram_mod.n_match},
                    {"ngram_mod_n_max", slot.task->params.speculative.ngram_mod.n_max},
                    {"ngram_mod_n_min", slot.task->params.speculative.ngram_mod.n_min},
                    {"ngram_simple_size_n", slot.task->params.speculative.ngram_simple.size_n},
                    {"ngram_simple_size_m", slot.task->params.speculative.ngram_simple.size_m},
                    {"ngram_simple_min_hits", slot.task->params.speculative.ngram_simple.min_hits},
                }},
            }},
        };

        event["output_token_telemetry"] = telemetry_output_token_json(slot);

        event["moe_routing"] = telemetry_moe_routing_json(slot);

        auto probability_json = [](const telemetry_probability_accumulator & probability, const char * semantics) {
            const double mean_nll = probability.nll_sum / probability.count;
            const double perplexity = std::exp(mean_nll);
            return json {
                {"state", "available"},
                {"available", true},
                {"semantics", semantics},
                {"scored_tokens", probability.count},
                {"mean_nll", mean_nll},
                {"mean_selected_token_log_probability", -mean_nll},
                {"min_selected_token_log_probability", probability.min_logprob},
                {"max_selected_token_log_probability", probability.max_logprob},
                {"perplexity", std::isfinite(perplexity) ? json(perplexity) : json(nullptr)},
                {"perplexity_state", std::isfinite(perplexity) ? "available" : "overflow"},
            };
        };

        if (slot.task->params.sampling.n_probs == 0) {
            event["response_probability"] = {
                {"state", "disabled"},
                {"available", false},
                {"reason", "request_did_not_enable_logprobs"},
            };
        } else if (!slot.response_probability.unavailable_reason.empty()) {
            event["response_probability"] = {
                {"state", "unavailable"},
                {"available", false},
                {"reason", slot.response_probability.unavailable_reason},
            };
        } else if (slot.response_probability.count == 0) {
            event["response_probability"] = {
                {"state", "unavailable"},
                {"available", false},
                {"reason", "no_selected_token_probabilities_recorded"},
            };
        } else {
            event["response_probability"] = probability_json(
                slot.response_probability, "raw_target_model_pre_sampler_selected_token_probability");
        }

        if (!slot.task->params.prompt_perplexity) {
            event["prompt_probability"] = {
                {"state", "disabled"},
                {"available", false},
                {"reason", "request_did_not_enable_prompt_perplexity"},
            };
        } else if (!slot.prompt_probability.unavailable_reason.empty()) {
            event["prompt_probability"] = {
                {"state", "unavailable"},
                {"available", false},
                {"reason", slot.prompt_probability.unavailable_reason},
            };
        } else if (slot.prompt_probability.count == 0) {
            event["prompt_probability"] = {
                {"state", "unavailable"},
                {"available", false},
                {"reason", "no_prompt_token_probabilities_recorded"},
            };
        } else {
            event["prompt_probability"] = probability_json(
                slot.prompt_probability, "raw_target_model_next_token_probability_cold_text_prompt");
            event["prompt_probability"]["conditioning_tokens"] = 1;
            event["prompt_probability"]["cache_reuse_disabled"] = true;
        }

        if (telemetry_request_content_enabled(slot)) {
            event["request"] = slot.task->telemetry_request;
            event["response"] = slot.generated_text;
        } else {
            event["request"] = {
                {"content_omitted", true},
                {"reason", "content_logging_disabled"},
            };
        }
        slot.telemetry_pending_completion_event = std::move(event);
    }

    void telemetry_on_response_handoff(server_slot & slot, int64_t t_handoff) {
        if (!slot.telemetry_finalized || t_handoff == 0) {
            return;
        }
        GGML_ASSERT(slot.stats.t_complete == 0);
        GGML_ASSERT(t_handoff >= slot.stats.t_finalization_start);
        slot.stats.t_complete = t_handoff;
        metrics.request_e2e_seconds.observe(slot.stats.t_e2e_ms() / 1000.0);

        if (slot.telemetry_pending_completion_event.is_null()) {
            return;
        }

        slot.telemetry_pending_completion_event["timestamp_unix_ms"] = telemetry_unix_ms(slot, slot.stats.t_complete);
        slot.telemetry_pending_completion_event["timings"] = slot.stats.to_json();
        slot.telemetry_pending_completion_event["lifecycle_clock"] = telemetry_lifecycle_clock_json(slot);
    }

    void telemetry_on_release(server_slot & slot) {
        if (!slot.telemetry_finalized || slot.telemetry_pending_completion_event.is_null()) {
            return;
        }

        slot.telemetry_pending_completion_event["lifecycle_clock"] = telemetry_lifecycle_clock_json(slot);
        slot.telemetry_pending_completion_event["slot_release_unix_ms"] = telemetry_unix_ms(slot, slot.stats.t_release);
        telemetry_append(std::move(slot.telemetry_pending_completion_event));
        slot.telemetry_pending_completion_event = json();
    }

    std::string telemetry_events_json(uint64_t cursor, size_t limit) const {
        const uint64_t latest = telemetry_next_sequence - 1;
        const bool future_cursor = cursor > latest;
        const uint64_t effective_cursor = future_cursor ? latest : cursor;
        std::string events = "[";
        const uint64_t oldest = telemetry_events.empty() ? telemetry_next_sequence : telemetry_events.front().sequence;
        json gap_ranges = json::array();
        uint64_t previous_sequence = effective_cursor;
        for (const auto & entry : telemetry_events) {
            if (entry.sequence <= effective_cursor) {
                continue;
            }
            if (previous_sequence + 1 < entry.sequence) {
                gap_ranges.push_back({
                    {"first_sequence", previous_sequence + 1},
                    {"last_sequence", entry.sequence - 1},
                });
            }
            previous_sequence = entry.sequence;
        }
        if (previous_sequence < latest) {
            gap_ranges.push_back({
                {"first_sequence", previous_sequence + 1},
                {"last_sequence", latest},
            });
        }
        size_t event_count = 0;
        uint64_t next_cursor = effective_cursor;
        for (const auto & entry : telemetry_events) {
            const uint64_t sequence = entry.sequence;
            if (sequence <= effective_cursor) {
                continue;
            }
            if (event_count++ > 0) {
                events += ',';
            }
            events += entry.serialized;
            next_cursor = sequence;
            if (event_count >= limit) {
                break;
            }
        }
        events += ']';
        return telemetry_response_with_serialized_events({
            {"schema_version", 1},
            {"server_instance_id", telemetry_server_instance_id},
            {"events", json::array()},
            {"cursor", next_cursor},
            {"oldest_sequence", oldest},
            {"next_sequence", telemetry_next_sequence},
            {"gap", future_cursor || !gap_ranges.empty()},
            {"gap_ranges", std::move(gap_ranges)},
            {"dropped_events", telemetry_dropped_events},
            {"last_dropped_sequence", telemetry_last_dropped_sequence},
            {"retained_serialized_bytes", telemetry_event_bytes},
            {"content_logging", telemetry_control_current().request_content},
        }, events);
    }

    std::string telemetry_kv_pressure_events_json(
            uint64_t cursor,
            size_t limit,
            const std::string & trace_id) const {
        const uint64_t oldest = telemetry_kv_pressure_events.empty()
            ? telemetry_kv_pressure_next_sequence
            : telemetry_kv_pressure_events.front().sequence;
        const uint64_t latest = telemetry_kv_pressure_next_sequence - 1;
        const bool future_cursor = cursor > latest;
        const uint64_t effective_cursor = future_cursor ? latest : cursor;
        const bool gap = future_cursor || effective_cursor < telemetry_kv_pressure_last_dropped_sequence ||
            (effective_cursor == 0 && oldest > 1) ||
            (effective_cursor != 0 && effective_cursor < oldest && oldest - effective_cursor > 1);

        const telemetry_kv_request_window * request_window = nullptr;
        if (!trace_id.empty()) {
            for (auto it = telemetry_kv_request_windows.rbegin(); it != telemetry_kv_request_windows.rend(); ++it) {
                if (it->trace_id == trace_id) {
                    request_window = &*it;
                    break;
                }
            }
        }

        auto matches_explicit_trace = [&](const telemetry_kv_pressure_event_entry & entry) {
            return entry.trace_id == trace_id || entry.victim_trace_id == trace_id;
        };
        const bool retained_explicit_trace = !trace_id.empty() && std::any_of(
            telemetry_kv_pressure_events.begin(),
            telemetry_kv_pressure_events.end(),
            matches_explicit_trace);

        auto matches_trace = [&](const telemetry_kv_pressure_event_entry & entry) {
            if (trace_id.empty() || matches_explicit_trace(entry)) {
                return true;
            }
            if (request_window && entry.kind == "utilization_sample") {
                return entry.monotonic_us >= request_window->start_monotonic_us &&
                    (request_window->end_monotonic_us == 0 || entry.monotonic_us <= request_window->end_monotonic_us);
            }
            return false;
        };

        std::string events = "[";
        size_t event_count = 0;
        uint64_t next_cursor = effective_cursor;
        for (const auto & entry : telemetry_kv_pressure_events) {
            const uint64_t sequence = entry.sequence;
            if (sequence <= effective_cursor) {
                continue;
            }
            next_cursor = sequence;
            if (!matches_trace(entry)) {
                continue;
            }
            if (event_count++ > 0) {
                events += ',';
            }
            events += entry.serialized;
            if (event_count >= limit) {
                break;
            }
        }
        events += ']';

        const llama_memory_primary_occupancy & occupancy = telemetry_kv_boundary.memory.primary_occupancy;
        const bool occupancy_available = telemetry_kv_boundary.available && occupancy.available &&
            occupancy.capacity_entries > 0 && occupancy.used_entries <= occupancy.capacity_entries;
        const char * state = "available";
        const char * reason = "bounded native KV-pressure event stream";
        if (future_cursor) {
            state = "partial";
            reason = "the supplied cursor was ahead of this server instance and was reset to its high-water mark";
        } else if (!trace_id.empty() && !request_window) {
            if (retained_explicit_trace) {
                state = "partial";
                reason = "matching trace events are retained, but the request window metadata has expired";
            } else {
                state = "no_data";
                reason = "the requested trace is not retained by this server instance";
            }
        } else if (gap) {
            state = "partial";
            reason = "older KV-pressure events were evicted from the bounded native buffer";
        } else if (!occupancy_available) {
            state = "partial";
            reason = "causal events are available, but primary occupancy is unavailable";
        }

        return telemetry_response_with_serialized_events({
            {"schema_version", 1},
            {"state", state},
            {"reason", reason},
            {"server_instance_id", telemetry_server_instance_id},
            {"trace_filter", trace_id.empty() ? json(nullptr) : json(trace_id)},
            {"request_start_monotonic_us", request_window
                ? json(request_window->start_monotonic_us) : json(nullptr)},
            {"request_end_monotonic_us", request_window && request_window->end_monotonic_us > 0
                ? json(request_window->end_monotonic_us) : json(nullptr)},
            {"events", json::array()},
            {"cursor", next_cursor},
            {"oldest_sequence", oldest},
            {"next_sequence", telemetry_kv_pressure_next_sequence},
            {"gap", gap},
            {"dropped_events", telemetry_kv_pressure_dropped_events},
            {"last_dropped_sequence", telemetry_kv_pressure_last_dropped_sequence},
            {"retained_serialized_bytes", telemetry_kv_pressure_event_bytes},
        }, events);
    }

    static json telemetry_churn_json(const llama_memory_churn_data & churn) {
        return {
            {"state", "available"},
            {"reason", "authoritative monotonic memory-backend churn counters"},
            {"entries_allocated", churn.entries_allocated},
            {"entries_released", churn.entries_released},
            {"entries_overwritten", churn.entries_overwritten},
            {"memberships_added", churn.memberships_added},
            {"memberships_removed", churn.memberships_removed},
            {"sequence_remove_operations", churn.sequence_remove_operations},
            {"sequence_copy_operations", churn.sequence_copy_operations},
            {"shared_copy_entries", churn.shared_copy_entries},
            {"copied_entries", churn.copied_entries},
            {"reset_operations", churn.reset_operations},
            {"context_shift_operations", churn.context_shift_operations},
            {"shifted_entries", churn.shifted_entries},
            {"prepare_failures", churn.prepare_failures},
            {"optimize_attempts", churn.optimize_attempts},
            {"defragmentation", {
                {"state", "not_applicable"},
                {"reason", "active KV defragmentation is not implemented by the current memory backends"},
            }},
        };
    }

    static json telemetry_memory_diagnostics_json(const llama_memory_diagnostics & diagnostics) {
        if (diagnostics.state != "available") {
            return {
                {"state", diagnostics.state},
                {"reason", "the active memory backend did not expose structured memory diagnostics"},
                {"components", json::array()},
            };
        }

        json components = json::array();
        const llama_memory_component_diagnostics * primary = nullptr;
        for (const auto & component : diagnostics.components) {
            if (!primary && component.logical_primary) {
                primary = &component;
            }
            components.push_back({
                {"name", component.name},
                {"kind", component.kind},
                {"entry_semantics", component.entry_semantics},
                {"state", component.state},
                {"logical_primary", component.logical_primary},
                {"capacity_entries", component.capacity_entries},
                {"used_entries", component.used_entries},
                {"free_entries", component.capacity_entries - component.used_entries},
                {"utilization", component.capacity_entries > 0 ? json((double) component.used_entries / component.capacity_entries) : json(nullptr)},
                {"resident_tokens", component.resident_tokens_supported ? json(component.resident_tokens) : json(nullptr)},
                {"resident_tokens_state", component.resident_tokens_supported ? "available" : "not_applicable"},
                {"resident_tokens_reason", component.resident_tokens_supported
                    ? "the memory component exposes authoritative resident-token membership"
                    : "this memory component does not represent token-addressable KV residency"},
                {"sequences_represented", component.sequences_represented},
                {"physical_sharing", {
                    {"state", component.physical_sharing_supported ? "available" : "not_applicable"},
                    {"shared_entries", component.shared_entries},
                    {"shared_tokens", component.resident_tokens_supported ? json(component.shared_entries) : json(nullptr)},
                    {"shared_memberships", component.shared_memberships},
                    {"sequences_benefiting", component.sequences_sharing},
                    {"shared_groups", component.shared_groups},
                    {"average_fanout", component.shared_entries > 0 ? json((double) component.shared_memberships / component.shared_entries) : json(nullptr)},
                    {"maximum_fanout", component.max_fanout},
                }},
                {"allocated_bytes", component.allocated_bytes},
                {"occupied_bytes_estimate", component.occupied_bytes_estimate},
                {"occupied_bytes_is_estimate", component.occupied_bytes_is_estimate},
                {"churn", telemetry_churn_json(component.churn)},
            });
        }

        json live = {
            {"state", primary ? "available" : "unsupported"},
            {"reason", primary ? "authoritative primary memory-component counters" : "no logical primary memory component exposed live occupancy"},
        };
        json sharing = {
            {"state", primary && primary->physical_sharing_supported ? "available" : "not_applicable"},
            {"reason", primary && primary->physical_sharing_supported ? "authoritative physical sequence membership" : "the primary memory component does not expose physical prefix sharing"},
        };
        if (primary) {
            live.set({"memory_kind", primary->kind});
            live.set({"entry_semantics", primary->entry_semantics});
            live.set({"capacity_entries", primary->capacity_entries});
            live.set({"used_entries", primary->used_entries});
            live.set({"free_entries", primary->capacity_entries - primary->used_entries});
            live.set({"utilization", primary->capacity_entries > 0 ? json((double) primary->used_entries / primary->capacity_entries) : json(nullptr)});
            live.set({"resident_tokens", primary->resident_tokens_supported ? json(primary->resident_tokens) : json(nullptr)});
            live.set({"resident_tokens_state", primary->resident_tokens_supported ? "available" : "not_applicable"});
            live.set({"resident_tokens_reason", primary->resident_tokens_supported
                ? "the primary memory component exposes authoritative resident-token membership"
                : "the primary memory component does not expose token-addressable residency"});
            live.set({"sequences_represented", primary->sequences_represented});
            live.set({"allocated_bytes", primary->allocated_bytes});
            live.set({"occupied_bytes_estimate", primary->occupied_bytes_estimate});
            live.set({"occupied_bytes_is_estimate", primary->occupied_bytes_is_estimate});

            sharing.set({"entry_semantics", primary->entry_semantics});
            sharing.set({"shared_entries", primary->shared_entries});
            sharing.set({"shared_tokens", primary->resident_tokens_supported ? json(primary->shared_entries) : json(nullptr)});
            sharing.set({"shared_memberships", primary->shared_memberships});
            sharing.set({"sequences_benefiting", primary->sequences_sharing});
            sharing.set({"shared_prefix_groups", primary->shared_groups});
            sharing.set({"average_fanout", primary->shared_entries > 0 ? json((double) primary->shared_memberships / primary->shared_entries) : json(nullptr)});
            sharing.set({"maximum_fanout", primary->max_fanout});
        }

        return {
            {"state", "available"},
            {"live_occupancy", std::move(live)},
            {"physical_prefix_sharing", std::move(sharing)},
            {"components", std::move(components)},
            {"churn", telemetry_churn_json(diagnostics.churn)},
        };
    }

    static json telemetry_memory_snapshot_json(const llama_memory_snapshot & snapshot) {
        if (snapshot.diagnostics_collected) {
            return telemetry_memory_diagnostics_json(snapshot.diagnostics);
        }

        const llama_memory_primary_occupancy & occupancy = snapshot.primary_occupancy;
        const bool valid = occupancy.available && occupancy.capacity_entries > 0 &&
            occupancy.used_entries <= occupancy.capacity_entries;
        return {
            {"state", valid ? "available" : occupancy.available ? "no_data" : "unsupported"},
            {"live_occupancy", {
                {"state", valid ? "available" : occupancy.available ? "no_data" : "unsupported"},
                {"reason", valid
                    ? "immutable primary memory occupancy captured at the latest decode boundary"
                    : "the latest decode boundary did not expose usable primary occupancy"},
                {"capacity_entries", valid ? json(occupancy.capacity_entries) : json(nullptr)},
                {"used_entries", valid ? json(occupancy.used_entries) : json(nullptr)},
                {"free_entries", valid ? json(occupancy.capacity_entries - occupancy.used_entries) : json(nullptr)},
                {"utilization", valid ? json((double) occupancy.used_entries / occupancy.capacity_entries) : json(nullptr)},
                {"resident_tokens", nullptr},
                {"resident_tokens_state", "not_collected"},
                {"resident_tokens_reason", "deep diagnostics were not requested"},
            }},
            {"physical_prefix_sharing", {
                {"state", "not_collected"},
                {"reason", "deep diagnostics were not requested"},
            }},
            {"components", json::array()},
            {"churn", {
                {"state", "not_collected"},
                {"reason", "deep diagnostics were not requested"},
            }},
        };
    }

    json telemetry_snapshot_json() {
        int active_slots = 0;
        uint64_t resident_slot_tokens = 0;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                active_slots++;
            }
            resident_slot_tokens += slot.prompt.n_tokens();
        }
        const json token_hit_rate = metrics.n_draft_tokens > 0 ? json((double) metrics.n_draft_accepted / metrics.n_draft_tokens) : json(nullptr);
        const json step_hit_rate = metrics.n_draft_verif_steps > 0 ? json((double) metrics.n_draft_hit_steps / metrics.n_draft_verif_steps) : json(nullptr);
        const llama_ubatch_stats ubatch_target = llama_get_ubatch_stats(ctx_tgt);
        const llama_ubatch_stats ubatch_draft = ctx_dft ? llama_get_ubatch_stats(ctx_dft) : llama_ubatch_stats {};
        const json memory_diagnostics = telemetry_kv_boundary.available
            ? telemetry_memory_snapshot_json(telemetry_kv_boundary.memory)
            : telemetry_memory_snapshot_json({});
        const auto histogram_summary = [](const server_histogram & histogram) {
            auto quantile_upper_bound = [&](double q) -> json {
                if (histogram.count == 0) {
                    return nullptr;
                }
                const uint64_t rank = std::max<uint64_t>(1, (uint64_t) std::ceil(q * histogram.count));
                for (size_t i = 0; i < histogram.bounds.size(); ++i) {
                    if (histogram.buckets[i] >= rank) {
                        return histogram.bounds[i];
                    }
                }
                return "+Inf";
            };
            json buckets = json::array();
            for (size_t i = 0; i < histogram.bounds.size(); ++i) {
                buckets.push_back({{"le", histogram.bounds[i]}, {"count", histogram.buckets[i]}});
            }
            buckets.push_back({{"le", "+Inf"}, {"count", histogram.count}});
            return json {
                {"count", histogram.count},
                {"sum", histogram.sum},
                {"mean", histogram.count > 0 ? json(histogram.sum / histogram.count) : json(nullptr)},
                {"maximum", histogram.count > 0 ? json(histogram.maximum) : json(nullptr)},
                {"p50_upper_bound", quantile_upper_bound(0.50)},
                {"p95_upper_bound", quantile_upper_bound(0.95)},
                {"buckets", std::move(buckets)},
            };
        };
        const auto ubatch_to_json = [](const llama_ubatch_stats & stats, bool supported) {
            if (!supported) {
                return json {{"state", "not_applicable"}};
            }
            const auto bounds = llama_ubatch_histogram_bounds();
            auto quantile_upper_bound = [&](double q) -> json {
                if (stats.successful == 0) {
                    return nullptr;
                }
                const uint64_t rank = std::max<uint64_t>(1, (uint64_t) std::ceil(q * stats.successful));
                for (size_t i = 0; i < bounds.size(); ++i) {
                    if (stats.token_buckets[i] >= rank) {
                        return bounds[i];
                    }
                }
                return "+Inf";
            };
            json buckets = json::array();
            for (size_t i = 0; i < bounds.size(); ++i) {
                buckets.push_back({{"le", bounds[i]}, {"count", stats.token_buckets[i]}});
            }
            buckets.push_back({{"le", "+Inf"}, {"count", stats.successful}});
            return json {
                {"state", stats.successful > 0 ? "available" : "no_data"},
                {"attempted", stats.attempted},
                {"successful", stats.successful},
                {"failed", stats.attempted - stats.successful},
                {"tokens", stats.tokens},
                {"sequence_tokens", stats.sequence_tokens},
                {"sequences", stats.sequences},
                {"unique_sequences", stats.unique_sequences},
                {"mean_tokens", stats.successful > 0 ? json((double) stats.tokens / stats.successful) : json(nullptr)},
                {"max_tokens", stats.successful > 0 ? json(stats.max_tokens) : json(nullptr)},
                {"p50_tokens_upper_bound", quantile_upper_bound(0.50)},
                {"p95_tokens_upper_bound", quantile_upper_bound(0.95)},
                {"token_buckets", std::move(buckets)},
            };
        };
        return {
            {"schema_version", 1},
            {"server_instance_id", telemetry_server_instance_id},
            {"telemetry_control", telemetry_control_snapshot_json()},
            {"timestamp_unix_ms", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()},
            {"clock", {
                {"monotonic_us", ggml_time_us()},
                {"process_start_monotonic_us", metrics.t_start},
                {"process_start_unix_s", metrics.t_start_unix},
            }},
            {"requests", {
                {"total", metrics.n_requests_total},
                {"success", metrics.n_requests_success},
                {"cancelled", metrics.n_requests_cancelled},
                {"error", metrics.n_requests_error},
                {"active", active_slots},
                {"queued", queue_tasks.queue_tasks_pending_size()},
                {"deferred", queue_tasks.queue_tasks_deferred_size()},
            }},
            {"throughput", {
                {"prefill_tps_window", metrics.prompt_bucket.n_per_second()},
                {"decode_tps_window", metrics.predict_bucket.n_per_second()},
                {"evaluated_prompt_tokens_total", metrics.prompt.count},
                {"server_output_tokens_total", metrics.n_server_output_tokens},
                {"generation_steps_total", metrics.predict.steps},
                {"prefill_state", metrics.prompt_bucket.time > 0 ? "available" : "no_data"},
                {"decode_state", metrics.predict_bucket.time > 0 ? "available" : "no_data"},
                {"server_output_tps", nullptr},
                {"server_output_tps_state", "derive_from_counter_delta"},
            }},
            {"forward_pass", {
                {"decode_calls", metrics.n_decode},
                {"logical_tokens", metrics.n_logical_tokens},
                {"successful_calls", metrics.n_logical_decode_success},
                {"mean_logical_tokens_per_call", metrics.n_logical_decode_success > 0 ? (double) metrics.n_logical_tokens / metrics.n_logical_decode_success : 0.0},
                {"logical_tokens_per_call_distribution", histogram_summary(metrics.logical_batch_tokens)},
                {"participating_slots_per_call_distribution", histogram_summary(metrics.logical_batch_slots)},
                {"physical_ubatch", {
                    {"target", ubatch_to_json(ubatch_target, true)},
                    {"draft", ubatch_to_json(ubatch_draft, ctx_dft != nullptr)},
                }},
            }},
            {"cache", {
                {"reused_prompt_tokens", metrics.n_prompt_cached},
                {"misses", metrics.n_cache_miss},
                {"partial_hits", metrics.n_cache_partial},
                {"full_hits", metrics.n_cache_full},
            }},
            {"speculative", {
                {"draft_tokens", metrics.n_draft_tokens},
                {"accepted_tokens", metrics.n_draft_accepted},
                {"rejected_tokens", metrics.n_draft_tokens - metrics.n_draft_accepted},
                {"verification_steps", metrics.n_draft_verif_steps},
                {"logical_target_passes", metrics.n_draft_verif_steps},
                {"hit_steps", metrics.n_draft_hit_steps},
                {"miss_steps", metrics.n_draft_verif_steps >= metrics.n_draft_hit_steps ? metrics.n_draft_verif_steps - metrics.n_draft_hit_steps : 0},
                {"full_chain_steps", metrics.n_draft_full_steps},
                {"target_passes", metrics.n_spec_target_passes},
                {"target_tokens", metrics.n_spec_target_tokens},
                {"useful_output_tokens", metrics.n_spec_useful_tokens},
                {"target_tokens_per_pass", metrics.n_spec_target_passes > 0 ? json((double) metrics.n_spec_target_tokens / metrics.n_spec_target_passes) : json(nullptr)},
                {"useful_output_tokens_per_target_pass", metrics.n_spec_target_passes > 0 ? json((double) metrics.n_spec_useful_tokens / metrics.n_spec_target_passes) : json(nullptr)},
                {"token_hit_rate", token_hit_rate},
                {"token_miss_rate", metrics.n_draft_tokens > 0 ? json(1.0 - token_hit_rate.get<double>()) : json(nullptr)},
                {"step_hit_rate", step_hit_rate},
                {"step_miss_rate", metrics.n_draft_verif_steps > 0 ? json(1.0 - step_hit_rate.get<double>()) : json(nullptr)},
                {"full_chain_rate", metrics.n_draft_verif_steps > 0 ? json((double) metrics.n_draft_full_steps / metrics.n_draft_verif_steps) : json(nullptr)},
                {"draft_depth_distribution", histogram_summary(metrics.spec_draft_depth)},
                {"accepted_depth_distribution", histogram_summary(metrics.spec_accepted_depth)},
                {"target_tokens_per_pass_distribution", histogram_summary(metrics.spec_target_tokens_per_pass)},
                {"useful_output_tokens_per_target_pass_distribution", histogram_summary(metrics.spec_useful_tokens_per_pass)},
                {"rate_state", metrics.n_draft_verif_steps > 0 ? "available" : "no_data"},
            }},
            {"kv", {
                {"resident_slot_tokens_upper_bound", telemetry_kv_boundary.available
                    ? telemetry_kv_boundary.resident_slot_tokens
                    : resident_slot_tokens},
                {"represented_slots", telemetry_kv_boundary.available
                    ? telemetry_kv_boundary.represented_slots
                    : std::count_if(slots.begin(), slots.end(), [](const server_slot & slot) { return slot.prompt.n_tokens() > 0; })},
                {"live_occupancy", memory_diagnostics.value("live_occupancy", json {
                    {"state", memory_diagnostics.value("state", "unsupported")},
                    {"reason", memory_diagnostics.value("reason", "the active memory backend did not expose live occupancy")},
                })},
                {"physical_prefix_sharing", memory_diagnostics.value("physical_prefix_sharing", json {
                    {"state", memory_diagnostics.value("state", "unsupported")},
                    {"reason", memory_diagnostics.value("reason", "the active memory backend did not expose physical sequence membership")},
                })},
                {"churn", memory_diagnostics.value("churn", json {
                    {"state", memory_diagnostics.value("state", "unsupported")},
                    {"reason", memory_diagnostics.value("reason", "the active memory backend did not expose churn counters")},
                })},
            }},
        };
    }

    json telemetry_duplicate_prefix_json(
            const llama_memory_diagnostics & diagnostics,
            const telemetry_kv_boundary_snapshot & snapshot) const {
        struct candidate {
            const telemetry_kv_slot_snapshot * slot;
            const std::vector<llama_token> * tokens;
            std::string compatibility;
        };

        std::vector<candidate> candidates;
        candidates.reserve(snapshot.slots.size());
        for (const auto & slot : snapshot.slots) {
            candidates.push_back({&slot, &slot.tokens, slot.compatibility});
        }

        std::sort(candidates.begin(), candidates.end(), [](const candidate & left, const candidate & right) {
            if (left.compatibility != right.compatibility) {
                return left.compatibility < right.compatibility;
            }
            return std::lexicographical_compare(
                left.tokens->begin(), left.tokens->end(), right.tokens->begin(), right.tokens->end());
        });

        const auto common_prefix = [](const std::vector<llama_token> & left, const std::vector<llama_token> & right) {
            const size_t limit = std::min(left.size(), right.size());
            size_t length = 0;
            while (length < limit && left[length] == right[length]) {
                ++length;
            }
            return length;
        };

        const llama_memory_component_diagnostics * primary = nullptr;
        for (const auto & component : diagnostics.components) {
            if (component.logical_primary && component.entry_semantics == "token_kv_cell") {
                primary = &component;
                break;
            }
        }
        const long double bytes_per_entry = primary && primary->capacity_entries > 0
            ? (long double) primary->allocated_bytes / primary->capacity_entries
            : 0.0L;

        std::vector<size_t> adjacent_lcp(candidates.size(), 0);
        for (size_t i = 0; i + 1 < candidates.size(); ++i) {
            if (candidates[i].compatibility == candidates[i + 1].compatibility) {
                adjacent_lcp[i] = common_prefix(*candidates[i].tokens, *candidates[i + 1].tokens);
            }
        }

        json groups = json::array();
        std::set<int> affected_slots;
        uint64_t redundant_tokens = 0;
        uint64_t longest_prefix = 0;
        uint64_t group_ordinal = 0;

        std::function<void(size_t, size_t, size_t)> analyze_range;
        analyze_range = [&](size_t begin, size_t end, size_t parent_depth) {
            if (end - begin < 2) {
                return;
            }
            size_t depth = std::numeric_limits<size_t>::max();
            for (size_t i = begin; i + 1 < end; ++i) {
                depth = std::min(depth, adjacent_lcp[i]);
            }

            if (depth > parent_depth) {
                std::vector<llama_seq_id> sequence_ids;
                sequence_ids.reserve(end - begin);
                for (size_t i = begin; i < end; ++i) {
                    sequence_ids.push_back(candidates[i].slot->id);
                }
                const uint64_t physically_shared = llama_get_memory_shared_prefix_length(
                    diagnostics, sequence_ids.data(), sequence_ids.size(), depth);
                const size_t unshared_start = std::max(parent_depth, (size_t) std::min<uint64_t>(physically_shared, depth));
                const uint64_t redundant_increment = (depth - unshared_start) * (end - begin - 1);
                if (redundant_increment > 0) {
                    for (size_t i = begin; i < end; ++i) {
                        affected_slots.insert(candidates[i].slot->id);
                    }
                    redundant_tokens += redundant_increment;
                    longest_prefix = std::max<uint64_t>(longest_prefix, depth);
                    groups.push_back({
                        {"ordinal", ++group_ordinal},
                        {"fanout", end - begin},
                        {"equivalent_prefix_tokens", depth},
                        {"physically_shared_prefix_tokens", physically_shared},
                        {"incremental_redundant_tokens", redundant_increment},
                        {"estimated_redundant_bytes", (uint64_t) (redundant_increment * bytes_per_entry)},
                    });
                }
                parent_depth = depth;
            }

            size_t child_begin = begin;
            for (size_t i = begin; i + 1 < end; ++i) {
                if (adjacent_lcp[i] <= parent_depth) {
                    analyze_range(child_begin, i + 1, parent_depth);
                    child_begin = i + 1;
                }
            }

            analyze_range(child_begin, end, parent_depth);
        };

        size_t compatibility_begin = 0;
        while (compatibility_begin < candidates.size()) {
            size_t compatibility_end = compatibility_begin + 1;
            while (compatibility_end < candidates.size() &&
                   candidates[compatibility_end].compatibility == candidates[compatibility_begin].compatibility) {
                ++compatibility_end;
            }
            analyze_range(compatibility_begin, compatibility_end, 0);
            compatibility_begin = compatibility_end;
        }

        const uint64_t redundant_bytes = (uint64_t) (redundant_tokens * bytes_per_entry);
        const json reclaim_ratio = primary && primary->occupied_bytes_estimate > 0
            ? json((double) redundant_bytes / primary->occupied_bytes_estimate)
            : json(nullptr);
        return {
            {"state", primary ? "available" : "not_applicable"},
            {"reason", primary ? "bounded compatible-prefix analysis over server slot metadata" : "the active memory backend has no token-KV primary component"},
            {"semantics", "optimization_estimate_from_compatible_token_metadata_validated_against_physical_sequence_membership"},
            {"compatible_sequences_examined", candidates.size()},
            {"duplicate_prefix_groups", groups.size()},
            {"affected_sequences", affected_slots.size()},
            {"redundant_prefix_tokens", redundant_tokens},
            {"estimated_redundant_kv_bytes", redundant_bytes},
            {"longest_duplicate_prefix_tokens", longest_prefix},
            {"estimated_potential_reclaim_ratio", reclaim_ratio},
            {"groups", std::move(groups)},
            {"multimodal_sequences_skipped", snapshot.multimodal_sequences_skipped},
        };
    }

    json telemetry_kv_json(bool include_diagnostics) {
        if (include_diagnostics) {
            telemetry_kv_snapshot_capture(true);
        }
        const telemetry_kv_boundary_snapshot & snapshot = telemetry_kv_boundary;
        llama_memory_breakdown_data total;
        json devices = json::array();
        const auto & breakdown = snapshot.breakdown;
        for (const auto & item : breakdown) {
            total.model += item.second.model;
            total.context += item.second.context;
            total.compute += item.second.compute;
            json allocation = {
                {"buffer_type", ggml_backend_buft_name(item.first)},
                {"model_bytes", item.second.model},
                {"context_bytes", item.second.context},
                {"compute_bytes", item.second.compute},
            };
            if (auto * device = ggml_backend_buft_get_device(item.first)) {
                size_t free_bytes = 0;
                size_t total_bytes = 0;
                ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
                allocation["device"] = ggml_backend_dev_name(device);
                allocation["device_free_bytes"] = free_bytes;
                allocation["device_total_bytes"] = total_bytes;
            } else {
                allocation["device"] = "host";
                allocation["device_free_bytes"] = nullptr;
                allocation["device_total_bytes"] = nullptr;
            }
            devices.push_back(std::move(allocation));
        }
        const json diagnostics = telemetry_memory_snapshot_json(snapshot.memory);
        const json duplicate_prefixes = snapshot.memory.diagnostics_collected
            ? telemetry_duplicate_prefix_json(snapshot.memory.diagnostics, snapshot)
            : json {
                {"state", "not_collected"},
                {"reason", "request detail=deep to collect bounded duplicate-prefix diagnostics"},
            };
        return {
            {"schema_version", 1},
            {"server_instance_id", telemetry_server_instance_id},
            {"allocated", {
                {"state", "available"},
                {"reason", "authoritative model/context/compute allocation totals grouped by backend buffer type"},
                {"model_bytes", total.model},
                {"context_bytes", total.context},
                {"compute_bytes", total.compute},
                {"total_bytes", total.total()},
                {"by_buffer_type", std::move(devices)},
            }},
            {"slot_metadata", {
                {"state", "available"},
                {"reason", "bounded server-slot metadata; resident token count is explicitly an upper bound"},
                {"resident_tokens_upper_bound", snapshot.resident_slot_tokens},
                {"represented_slots", snapshot.represented_slots},
            }},
            {"live_occupancy", diagnostics.value("live_occupancy", json {
                {"state", diagnostics.value("state", "unsupported")},
                {"reason", diagnostics.value("reason", "the active memory backend did not expose live occupancy")},
            })},
            {"physical_prefix_sharing", diagnostics.value("physical_prefix_sharing", json {
                {"state", diagnostics.value("state", "unsupported")},
                {"reason", diagnostics.value("reason", "the active memory backend did not expose physical sequence membership")},
            })},
            {"components", diagnostics.value("components", json::array())},
            {"churn", diagnostics.value("churn", json {
                {"state", diagnostics.value("state", "unsupported")},
                {"reason", diagnostics.value("reason", "the active memory backend did not expose churn counters")},
            })},
            {"duplicate_prefix_opportunities", std::move(duplicate_prefixes)},
        };
    }

    //
    // metrics helpers
    //

    // call before submitting a decode, so that the queued prompt stats can be timed
    void metrics_pre_decode() {
        t_decode_start = ggml_time_us();
    }

    // the batch is submitted, but its compute may not be done yet
    void metrics_queue_prompt(uint64_t n_tokens) {
        if (n_tokens == 0) {
            return;
        }
        if (n_prompt_queued == 0) {
            t_prompt_start = t_decode_start;
        }
        n_prompt_queued += n_tokens;
    }

    // call only after the context is synchronized, otherwise the time is meaningless
    void metrics_flush_prompt() {
        if (n_prompt_queued == 0) {
            return;
        }
        metrics.add_prompt(n_prompt_queued, ggml_time_us() - t_prompt_start);
        n_prompt_queued = 0;
    }

    // has_output is computed by the caller, which also already synchronized the context if it is set
    void metrics_post_decode(int32_t off, int32_t n_tokens, bool has_output) {
        metrics.n_decode++;
        metrics.n_logical_decode_success++;
        metrics.n_logical_tokens += n_tokens;
        metrics.logical_batch_tokens.observe(n_tokens);
        if (++telemetry_slot_epoch == 0) {
            std::fill(telemetry_slot_marks.begin(), telemetry_slot_marks.end(), 0);
            telemetry_slot_epoch = 1;
        }
        size_t participating_slots = 0;
        for (int i = off; i < off + n_tokens; ++i) {
            const size_t id_slot = batch.tokens[i].id_slot;
            if (telemetry_slot_marks[id_slot] != telemetry_slot_epoch) {
                telemetry_slot_marks[id_slot] = telemetry_slot_epoch;
                participating_slots++;
            }
        }
        metrics.logical_batch_slots.observe(participating_slots);
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                metrics.n_busy_slots++;
            }
            metrics.n_tokens_max = std::max(metrics.n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }

        // apply enqueued prompt tokens stats
        // note: a slot can be released before we get here, which clears its stats
        //       the tokens were still computed, counted in the global metrics, not in slot
        uint64_t n_prompt_tokens = 0;

        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];

            if (!t.is_prompt) {
                continue; // generated tokens are handled after sampling
            }

            n_prompt_tokens++;

            auto & slot = slots[t.id_slot];
            if (slot.stats.is_set()) {
                slot.stats.n_prompt_processed++;
            }

            if (t.output && slot.should_score_prompt() && slot.prompt_probability.unavailable_reason.empty()) {
                // A prompt row at position p predicts the text token at p + 1.
                // The diagnostic forces a cold, text-only, contiguous prompt, so
                // token indices and positions are intentionally equivalent here.
                const int64_t next_index = (int64_t) t.pos + 1;
                if (next_index >= 0 && next_index < slot.task->n_tokens()) {
                    const llama_token selected = slot.task->tokens[(size_t) next_index];
                    double logprob = 0.0;
                    std::string reason;
                    if (selected_token_log_probability(ctx_tgt, i - off, selected, logprob, reason)) {
                        slot.prompt_probability.observe(logprob);
                    } else {
                        slot.prompt_probability.unavailable_reason = std::move(reason);
                    }
                }
            }
        }

        metrics_queue_prompt(n_prompt_tokens);

        if (has_output) {
            // the context is already synchronized, so the timings are correct
            metrics_flush_prompt();
        }

        // advance the prompt timing of the slots that had tokens in this batch
        // note: a second pass, it must run after the sync to reflect the compute
        const int64_t t_now = ggml_time_us();
        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];
            auto & slot = slots[t.id_slot];
            if (t.is_prompt && slot.stats.is_set()) {
                slot.stats.set_prompt_last(t_now);
                slot.stats.t_prefill_last = t_now;
            }
        }
        telemetry_kv_snapshot_capture(false);
    }

    // flush any queued prompt metrics if all slots are now idle
    void metrics_flush_idle() {
        if (n_prompt_queued == 0) {
            return;
        }

        llama_synchronize(ctx_tgt);
        metrics_flush_prompt();
    }

    void metrics_on_prediction(const server_slot & slot) {
        const uint64_t t_us    = slot.stats.t_gen_us();
        const uint64_t n       = slot.stats.n_gen;
        const uint64_t n_steps = slot.stats.n_gen_steps();

        metrics.predict       .add(n, n_steps, t_us);
        metrics.predict_bucket.add(n, n_steps, t_us);

        metrics.n_draft_tokens      += slot.stats.n_draft_tokens;
        metrics.n_draft_accepted    += slot.stats.n_draft_accepted;
        metrics.n_draft_verif_steps += slot.stats.n_draft_verif_steps;

        auto & dst = metrics.n_accepted_per_pos;
        const auto & src = slot.n_accepted_per_pos;

        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0);
        }
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] += src[i];
        }
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* has_inp_video          */ impl->chat_params.allow_video,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* slot_n_ctx             */ impl->n_ctx_slot(),
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
        /* model_ftype            */ ftype_name,
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void ok_serialized(std::string response_data) {
        status = 200;
        data = std::move(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
}

//
// server_routes
//

static bool parse_w3c_traceparent(const std::string & value, std::string & trace_id) {
    if (value.size() != 55 || value[2] != '-' || value[35] != '-' || value[52] != '-') {
        return false;
    }
    auto is_hex_range = [&](size_t offset, size_t length) {
        return std::all_of(value.begin() + offset, value.begin() + offset + length, [](unsigned char c) {
            return std::isxdigit(c) != 0;
        });
    };
    if (!is_hex_range(0, 2) || !is_hex_range(3, 32) || !is_hex_range(36, 16) || !is_hex_range(53, 2)) {
        return false;
    }
    std::string version = value.substr(0, 2);
    std::string candidate = value.substr(3, 32);
    std::string parent_id = value.substr(36, 16);
    std::transform(version.begin(), version.end(), version.begin(), [](unsigned char c) { return std::tolower(c); });
    std::transform(candidate.begin(), candidate.end(), candidate.begin(), [](unsigned char c) { return std::tolower(c); });
    std::transform(parent_id.begin(), parent_id.end(), parent_id.begin(), [](unsigned char c) { return std::tolower(c); });
    if (version == "ff" || candidate == std::string(32, '0') || parent_id == std::string(16, '0')) {
        return false;
    }
    trace_id = std::move(candidate);
    return true;
}

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    const std::string request_trace_id = "trace-" + random_string();
    std::string correlation_trace_id;
    std::string request_traceparent;
    for (const auto & header : req.headers) {
        std::string name = header.first;
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
        if (name == "traceparent" && parse_w3c_traceparent(header.second, correlation_trace_id)) {
            request_traceparent = header.second;
            break;
        }
    }
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;
        const telemetry_control_state telemetry_control = ctx_server.telemetry_control_current();
        const auto request_feature_enabled = [&](const char * name, bool globally_enabled) {
            if (!data.contains(name)) {
                return globally_enabled;
            }
            if (!data.at(name).is_boolean()) {
                throw std::invalid_argument(std::string(name) + " must be a boolean");
            }
            return globally_enabled && data.at(name).get<bool>();
        };
        const auto request_feature_permitted = [&](const char * name) {
            if (!data.contains(name)) {
                return true;
            }
            if (!data.at(name).is_boolean()) {
                throw std::invalid_argument(std::string(name) + " must be a boolean");
            }
            return data.at(name).get<bool>();
        };
        const bool request_content = request_feature_enabled(
            "request_content", telemetry_control.request_content);

        json telemetry_request;
        if (request_content) {
            constexpr size_t MAX_TELEMETRY_REQUEST_BYTES = 4 * 1024 * 1024;
            if (req.body.size() <= MAX_TELEMETRY_REQUEST_BYTES) {
                telemetry_request["original_request"] = json::parse(req.body);
                if (data.contains("prompt")) {
                    telemetry_request["rendered_prompt"] = data.at("prompt");
                }
            } else {
                telemetry_request = {
                    {"content_omitted", true},
                    {"reason", "request_exceeds_4_mib_event_limit"},
                };
            }
        }

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        // message delimiters for checkpointing
        json delims = json_value(data, "message_delimiters", json::array());
        auto delimiters = common_chat_msg_delimiters_parse(delims);
        delimiters.tokenize(ctx_server.vocab);

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.trace_id = inputs.size() == 1 ? request_trace_id : request_trace_id + "-i" + std::to_string(i);
            task.traceparent = request_traceparent;
            task.correlation_trace_id = correlation_trace_id;
            task.t_arrival = req.t_arrival;
            task.t_arrival_unix_ms = req.t_arrival_unix_ms;
            task.telemetry_request = telemetry_request;
            task.telemetry_content = request_content;
            if (data.contains("temperature") && data.at("temperature").is_number()) {
                task.telemetry_requested_temperature = data.at("temperature");
            } else if (data.contains("temp") && data.at("temp").is_number()) {
                task.telemetry_requested_temperature = data.at("temp");
            }

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);

            task.params.prompt_perplexity = request_feature_enabled(
                "prompt_perplexity", telemetry_control.prompt_perplexity);
            task.params.output_token_telemetry = request_feature_enabled(
                "output_token_telemetry", telemetry_control.output_token_detail);
            task.params.output_token_candidate_telemetry = request_feature_enabled(
                "output_token_candidate_telemetry", telemetry_control.token_candidates);
            task.params.moe_routing_telemetry_permitted = request_feature_permitted(
                "moe_routing_telemetry");
            task.params.moe_routing_telemetry = telemetry_control.moe_routing
                && task.params.moe_routing_telemetry_permitted;

            task.params.message_spans = task.tokens.find_message_spans(delimiters);

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
        res->headers["X-Llama-Trace-Id"] = request_trace_id;
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(
        const common_params & params,
        server_context & ctx_server,
        const server_http_context & ctx_http)
        : params(params),
          ctx_server(*ctx_server.impl),
          ctx_http(ctx_http),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();

    // note: this must be registered before load_model()
    //       so that on sleep phase, the callback is called before ctx is destroyed
    queue_tasks.on_sleeping_state([this](bool is_sleeping) {
        update_cached_responses(is_sleeping);
    });
}

static json get_res_model_info(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
            {"ftype",       meta.model_ftype},
        }},
    };
}

static json get_res_models(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return json{
        {"models", json::array({
            {
                {"name",  meta.model_name},
                {"model", meta.model_name},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                {"type", "model"},
                {"description", ""},
                {"tags", json::array({""})},
                {"capabilities", meta.has_mtmd ? json::array({"completion","multimodal"}) : json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "gguf"},
                    {"family", ""},
                    {"families", json::array({""})},
                    {"parameter_size", ""},
                    {"quantization_level", ""}
                }}
            }
        })},
        {"object", "list"},
        {"data", json::array({
            get_res_model_info(meta),
        })}
    };
}

static json get_res_props(const server_context_meta & meta, const common_params & params, bool is_sleeping) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    task_params tparams;
    tparams.sampling = params.sampling;
    json default_generation_settings_for_props = json {
        { "params", tparams.to_json(true) },
        { "n_ctx",  meta.slot_n_ctx },
    };

    std::string tmpl_default = common_chat_templates_source(meta.chat_params.tmpls.get(), "");
    std::string tmpl_tools   = common_chat_templates_source(meta.chat_params.tmpls.get(), "tool_use");

    json props = {
        { "default_generation_settings", default_generation_settings_for_props },
        { "total_slots",                 params.n_parallel },
        { "model_alias",                 meta.model_name },
        { "model_ftype",                 meta.model_ftype },
        { "model_path",                  meta.model_path },
        { "modalities",                  json {
            {"vision", meta.has_inp_image},
            {"video",  meta.has_inp_video},
            {"audio",  meta.has_inp_audio},
        } },
        { "media_marker",                get_media_marker() },
        { "endpoint_slots",              params.endpoint_slots },
        { "endpoint_props",              params.endpoint_props },
        { "endpoint_metrics",            params.endpoint_metrics },
        { "ui",                          params.ui },
        { "ui_settings",                 meta.json_ui_settings },
        { "chat_template",               tmpl_default },
        { "chat_template_caps",          meta.chat_template_caps },
        { "bos_token",                   meta.bos_token_str },
        { "eos_token",                   meta.eos_token_str },
        { "build_info",                  meta.build_info },
        { "is_sleeping",                 is_sleeping },
        { "cors_proxy_enabled",          params.ui_mcp_proxy },
    };
    if (params.use_jinja) {
        if (!tmpl_tools.empty()) {
            props["chat_template_tool_use"] = tmpl_tools;
        }
    }

    return props;
}

json server_routes::get_model_info() const {
    return get_res_model_info(*meta);
}

std::unique_ptr<server_res_generator> server_routes::handle_telemetry(
        const server_http_req & req,
        server_task_type type,
        uint64_t cursor,
        size_t limit,
        const std::string & trace_id,
        bool deep_detail) {
    auto res = create_response();
    server_task task(type);
    task.id = res->rd.get_new_id();
    task.telemetry_cursor = cursor;
    task.telemetry_limit = limit;
    task.telemetry_trace_id = trace_id;
    task.telemetry_deep_detail = deep_detail;
    res->rd.post_task(std::move(task), true);

    auto result = res->rd.next(req.should_stop);
    if (!result) {
        return res;
    }
    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }
    auto * telemetry = dynamic_cast<server_task_result_telemetry *>(result.get());
    GGML_ASSERT(telemetry != nullptr);
    res->headers["Cache-Control"] = "no-store";
    if (telemetry->has_serialized_data) {
        res->ok_serialized(std::move(telemetry->serialized_data));
    } else {
        res->ok(telemetry->to_json());
    }
    return res;
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_telemetry_capabilities = [this](const server_http_req &) {
        auto res = create_response(true);
        res->headers["Cache-Control"] = "no-store";
        res->ok({
            {"schema_version", 1},
            {"server_instance_id", ctx_server.telemetry_instance_id()},
            {"server", {
                {"build", std::string(llama_build_info())},
                {"model", meta->model_name},
            }},
            {"clock", {
                {"duration_clock", "monotonic_microseconds"},
                {"event_clock", "unix_milliseconds_anchored_at_http_handler_dispatch_after_body_read"},
                {"process_start_unix_s", ctx_server.get_metrics().t_start_unix},
            }},
            {"configuration", {
                {"parallel_slots", params.n_parallel},
                {"logical_batch_size", params.n_batch},
                {"physical_ubatch_size", params.n_ubatch},
                {"context_size", params.n_ctx},
                {"continuous_batching", params.cont_batching},
                {"unified_kv", params.kv_unified},
            }},
            {"capabilities", {
                {"request_lifecycle", {{"state", "available"}, {"version", 1}}},
                {"ttft", {{"state", "available"}, {"semantics", "first_model_token_minus_http_handler_dispatch_after_body_read"}}},
                {"queue_latency", {{"state", "available"}, {"semantics", "slot_start_minus_first_enqueue"}}},
                {"cache_reuse", {{"state", "available"}, {"full_hit_replays_one_token", true}}},
                {"logical_batch", {{"state", "available"}}},
                {"physical_ubatch_observed", {{"state", "available"}, {"semantics", "actual llama_ubatch graph submissions"}}},
                {"speculative", {{"state", "available"}, {"enabled", common_speculative_n_max(&params.speculative) > 0}}},
                {"response_perplexity", {
                    {"state", "conditional"},
                    {"requires", "POST /props telemetry_control.output_token_detail=true plus n_probs > 0 and full raw target logits"},
                    {"supports_speculative_output", true},
                    {"semantics", "committed emitted tokens only; rejected and replayed speculative tokens are excluded"},
                }},
                {"prompt_perplexity", {
                    {"state", "conditional"},
                    {"enable_with", "POST /props telemetry_control.prompt_perplexity=true plus request prompt_perplexity=true"},
                    {"semantics", "exact raw-target next-token probability for a cold contiguous text prompt"},
                    {"overhead", "forces one scored prompt token per sequence and performs O(prompt_tokens * vocabulary) CPU work"},
                    {"multimodal", "unsupported"},
                }},
                {"kv_allocation", {{"state", "available"}, {"semantics", "typed memory components plus model/context/compute allocation by buffer type"}}},
                {"kv_live_occupancy", {{"state", "available"}, {"semantics", "authoritative memory metadata; occupied bytes are a labeled dense-allocation estimate"}}},
                {"kv_pressure", ctx_server.telemetry_kv_pressure_capability_json()},
                {"physical_prefix_sharing", {{"state", "available"}, {"semantics", "authoritative multi-sequence membership in physical memory entries"}}},
                {"duplicate_prefix_opportunities", {{"state", "available"}, {"semantics", "bounded metadata-only optimization estimate; multimodal histories are skipped"}}},
                {"gpu_gpm", ctx_server.telemetry_gpu_capability_json()},
                {"moe_routing", llama_model_n_expert(ctx_server.model_tgt) <= 0
                    ? json {{"state", "not_applicable"}, {"reason", "The loaded target model has no routed MoE experts."}}
                    : json {
                        {"state", "conditional"},
                        {"configured_experts", llama_model_n_expert(ctx_server.model_tgt)},
                        {"experts_per_token", llama_model_n_expert_used(ctx_server.model_tgt)},
                        {"reason", "Versioned full-request routing chunks retain every selected routed expert, exact effective coefficient, K/K+1 boundary score status, physical microbatch identity, and shared-expert metadata while POST /props enables telemetry_control.moe_routing."},
                        {"enable_with", "POST /props telemetry_control.moe_routing=true; request moe_routing_telemetry=false permanently opts a request out"},
                        {"endpoint", "/telemetry/v1/events"},
                        {"chunk_event", "moe_routing_chunk"},
                        {"chunk_schema_version", 2},
                        {"chunk_max_serialized_bytes", 1024 * 1024},
                        {"full_request_capture", true},
                        {"transport_gap_ranges", "response gap_ranges report only missing global event-sequence intervals"},
                        {"population", "target_model_routed_token_layer_decisions"},
                        {"token_detail_population", "target_model_output_logit_rows_by_layer"},
                        {"token_detail_schema_version", 2},
                        {"retained_token_detail_linkage", json::array({"model_position", "phase", "logical_step", "actual_target_pass", "proposal_position", "replay_pass"})},
                        {"routing_weights_state", "conditional"},
                        {"routing_weights_reason", "Exact effective expert coefficients and normalized selected-expert shares require the telemetry control and request opt-in."},
                        {"router_margin_state", llama_model_n_expert_used(ctx_server.model_tgt) >= 2
                            ? "conditional"
                            : "not_applicable"},
                        {"router_margin_reason", llama_model_n_expert_used(ctx_server.model_tgt) >= 2
                            ? "The normalized top-two selected-expert share difference requires the telemetry control and request opt-in."
                            : "A top-two margin does not apply when fewer than two experts are selected per token-layer decision."},
                        {"maximum_captured_activations", ctx_server.telemetry_moe_activation_limit_value()},
                        {"disabled_path_changes_graph", false},
                    }},
                {"output_token_telemetry", {
                    {"state", "conditional"},
                    {"reason", "Bounded committed-token timing, target-model position, decode origin, MTP commit linkage, and actual target-pass records require the telemetry control; raw target probability additionally requires n_probs > 0."},
                    {"enable_with", "POST /props telemetry_control.output_token_detail=true plus request output_token_telemetry=true"},
                    {"probability_enable_with", "n_probs > 0"},
                    {"automatic_request_defaults", true},
                    {"token_identity_policy", "token IDs and authoritative tokenizer-piece bytes require POST /props telemetry_control.request_content=true"},
                    {"population", "committed_generation_tokens"},
                    {"record_schema_version", 3},
                    {"mtp_pass_record_schema_version", 2},
                    {"retained_linkage", json::array({"model_ready_monotonic_us", "model_position", "origin", "logical_step", "actual_target_pass", "proposal_position", "accepted_depth", "proposed_count", "replay_pass"})},
                    {"retained_mtp_proposal_fields", json::array({"position", "disposition", "evaluated_actual_target_pass", "draft_token_id", "draft_token_piece_base64", "target_selected_token_id", "target_selected_token_piece_base64", "target_selected_log_probability_ln", "committed_output_ordinal"})},
                    {"maximum_captured_tokens", ctx_server.telemetry_output_tokens_limit()},
                    {"maximum_captured_mtp_passes", ctx_server.telemetry_mtp_passes_limit()},
                    {"maximum_captured_mtp_proposals", ctx_server.telemetry_mtp_proposals_limit()},
                    {"normal_request_telemetry_unaffected", false},
                }},
                {"output_token_candidates", {
                    {"state", "conditional"},
                    {"reason", "A separately retained target-model top-K block requires the telemetry control and request opt-in."},
                    {"enable_with", "POST /props telemetry_control.output_token_detail=true and telemetry_control.token_candidates=true plus request output_token_telemetry=true and output_token_candidate_telemetry=true"},
                    {"automatic_request_defaults", true},
                    {"endpoint", "/telemetry/v1/token-candidates?trace_id=<exact-trace-id>"},
                    {"schema_version", 2},
                    {"default_target_top_k", 5},
                    {"maximum_target_top_k", 5},
                    {"default_population", "mtp_mismatch_replacement_and_bonus_positions"},
                    {"accepted_positions", "separate_request_opt_in"},
                    {"draft_top_k_state", "unsupported"},
                    {"draft_top_k_reason", "The current producer tier retains draft proposal identity but not the draft-model top-K distribution."},
                    {"maximum_serialized_bytes_per_request", ctx_server.telemetry_token_candidate_max_bytes()},
                    {"maximum_decisions_per_request", ctx_server.telemetry_token_candidate_decisions_limit()},
                    {"token_identity_policy", "token IDs and authoritative tokenizer-piece bytes require POST /props telemetry_control.request_content=true"},
                    {"normal_event_payload_contains_candidates", false},
                    {"normal_request_telemetry_unaffected", false},
                }},
                {"content_events", {{"state", "conditional"}, {"enable_with", "POST /props telemetry_control.request_content=true plus request request_content=true"}}},
            }},
            {"telemetry_control", ctx_server.telemetry_control_capability_json()},
            {"content_policy", {
                {"default", "metadata_only"},
                {"in_memory_event_capacity", 2048},
                {"serialized_event_capacity_bytes", ctx_server.telemetry_event_capacity_bytes()},
                {"event_buffer_env", "LLAMA_TELEMETRY_EVENT_BUFFER_MIB"},
                {"request_capture_limit_bytes", 4 * 1024 * 1024},
                {"output_token_record_limit", ctx_server.telemetry_output_tokens_limit()},
                {"token_candidate_max_serialized_bytes", ctx_server.telemetry_token_candidate_max_bytes()},
                {"moe_activation_record_limit", ctx_server.telemetry_moe_activation_limit_value()},
                {"moe_token_detail_activation_record_limit", ctx_server.telemetry_moe_activation_limit_value()},
                {"moe_routing_chunk_max_serialized_bytes", 1024 * 1024},
                {"prometheus_content", false},
            }},
        });
        return res;
    };

    this->get_telemetry_snapshot = [this](const server_http_req & req) {
        return handle_telemetry(req, SERVER_TASK_TYPE_TELEMETRY_SNAPSHOT);
    };

    this->get_telemetry_events = [this](const server_http_req & req) {
        uint64_t cursor = 0;
        size_t limit = 100;
        try {
            auto parse_unsigned = [](const std::string & value) {
                if (value.empty() || value.front() == '-') {
                    throw std::invalid_argument("expected unsigned integer");
                }
                size_t end = 0;
                const uint64_t parsed = std::stoull(value, &end);
                if (end != value.size()) {
                    throw std::invalid_argument("unexpected trailing characters");
                }
                return parsed;
            };
            if (!req.get_param("cursor").empty()) {
                cursor = parse_unsigned(req.get_param("cursor"));
            }
            if (!req.get_param("limit").empty()) {
                limit = parse_unsigned(req.get_param("limit"));
            }
        } catch (const std::exception &) {
            auto res = create_response(true);
            res->error(format_error_response("cursor and limit must be non-negative integers", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        limit = std::max<size_t>(1, std::min<size_t>(limit, 512));
        return handle_telemetry(req, SERVER_TASK_TYPE_TELEMETRY_EVENTS, cursor, limit);
    };

    this->get_telemetry_kv = [this](const server_http_req & req) {
        const std::string detail = req.get_param("detail");
        if (!detail.empty() && detail != "deep") {
            auto res = create_response(true);
            res->error(format_error_response("detail must be deep when specified", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        return handle_telemetry(req, SERVER_TASK_TYPE_TELEMETRY_KV, 0, 100, {}, detail == "deep");
    };

    this->get_telemetry_kv_pressure = [this](const server_http_req & req) {
        uint64_t cursor = 0;
        size_t limit = 256;
        try {
            auto parse_unsigned = [](const std::string & value) {
                if (value.empty() || value.front() == '-') {
                    throw std::invalid_argument("expected unsigned integer");
                }
                size_t end = 0;
                const uint64_t parsed = std::stoull(value, &end);
                if (end != value.size()) {
                    throw std::invalid_argument("unexpected trailing characters");
                }
                return parsed;
            };
            if (!req.get_param("cursor").empty()) {
                cursor = parse_unsigned(req.get_param("cursor"));
            }
            if (!req.get_param("limit").empty()) {
                limit = parse_unsigned(req.get_param("limit"));
            }
        } catch (const std::exception &) {
            auto res = create_response(true);
            res->error(format_error_response(
                "cursor and limit must be non-negative integers",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (limit < 1 || limit > 4096) {
            auto res = create_response(true);
            res->error(format_error_response(
                "limit must be between 1 and 4096",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        const std::string trace_id = req.get_param("trace_id");
        if (trace_id.size() > 256 ||
                std::any_of(trace_id.begin(), trace_id.end(), [](unsigned char c) { return std::iscntrl(c) != 0; })) {
            auto res = create_response(true);
            res->error(format_error_response(
                "trace_id must contain at most 256 non-control characters",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        return handle_telemetry(
            req,
            SERVER_TASK_TYPE_TELEMETRY_KV_PRESSURE,
            cursor,
            limit,
            trace_id);
    };

    this->get_telemetry_gpu = [this](const server_http_req & req) {
        uint64_t cursor = 0;
        size_t limit = 256;
        try {
            auto parse_unsigned = [](const std::string & value) {
                if (value.empty() || value.front() == '-') {
                    throw std::invalid_argument("expected unsigned integer");
                }
                size_t end = 0;
                const uint64_t parsed = std::stoull(value, &end);
                if (end != value.size()) {
                    throw std::invalid_argument("unexpected trailing characters");
                }
                return parsed;
            };
            if (!req.get_param("cursor").empty()) {
                cursor = parse_unsigned(req.get_param("cursor"));
            }
            if (!req.get_param("limit").empty()) {
                limit = parse_unsigned(req.get_param("limit"));
            }
        } catch (const std::exception &) {
            auto res = create_response(true);
            res->error(format_error_response("cursor and limit must be non-negative integers", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        limit = std::max<size_t>(1, std::min<size_t>(limit, 4096));
        return handle_telemetry(
            req,
            SERVER_TASK_TYPE_TELEMETRY_GPU,
            cursor,
            limit,
            req.get_param("trace_id"));
    };

    this->get_telemetry_token_candidates = [this](const server_http_req & req) {
        const std::string trace_id = req.get_param("trace_id");
        if (trace_id.empty() || trace_id.size() > 256 ||
                std::any_of(trace_id.begin(), trace_id.end(), [](unsigned char c) { return std::iscntrl(c) != 0; })) {
            auto res = create_response(true);
            res->error(format_error_response(
                "trace_id is required and must contain at most 256 non-control characters",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        return handle_telemetry(
            req,
            SERVER_TASK_TYPE_TELEMETRY_TOKEN_CANDIDATES,
            0,
            1,
            trace_id);
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // render response using cached_metrics
        auto use_cached_metrics = [&]() {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->headers["Process-Start-Time-Unix"] = std::to_string(cached_metrics.t_start_unix);
            server_task_result_metrics tmp;
            tmp.metrics = cached_metrics;
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = tmp.to_metrics();
            // the gauges are averaged over the window between two scrapes
            cached_metrics.reset_bucket();
            should_reset_buckets = true;
        };

        if (queue_tasks.is_sleeping()) {
            use_cached_metrics();

        } else {
            // request slots data using task queue
            {
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = res->rd.get_new_id();
                // the gauges are averaged over the window between two scrapes
                task.metrics_reset_bucket = true;
                res->rd.post_task(std::move(task), true); // high-priority task
            }

            // a task posted right before sleeping is never processed, do not wait for it
            auto result = res->rd.next([&]{
                return req.should_stop() || queue_tasks.is_sleeping();
            });
            if (!result) {
                if (!req.should_stop()) {
                    use_cached_metrics();
                }
                return res;
            }

            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }

            auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
            GGML_ASSERT(res_task != nullptr);

            res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start_unix);
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = res_task->to_metrics();
        }

        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->to_json());
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_props);
        } else {
            res->ok(get_res_props(*meta, params, false));
        }
        return res;
    };

    this->post_props = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        if (params.api_keys.empty()) {
            res->error(format_error_response("Telemetry control requires a configured API key.", ERROR_TYPE_PERMISSION));
            return res;
        }
        if (!ctx_http.telemetry_control_is_loopback_listener()) {
            res->error(format_error_response("Telemetry control requires a loopback listener.", ERROR_TYPE_PERMISSION));
            return res;
        }

        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception &) {
            res->error(format_error_response("request body must be valid JSON", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!body.is_object() || !body.contains("telemetry_control") || !body.at("telemetry_control").is_object()) {
            res->error(format_error_response("telemetry_control must be an object", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        const json & source = body.at("telemetry_control");
        static const std::set<std::string> names = {
            "moe_routing",
            "output_token_detail",
            "token_candidates",
            "prompt_perplexity",
            "request_content",
            "kv_pressure_detail",
            "native_gpu_gpm",
        };
        for (const auto & item : source.items()) {
            if (names.count(item.key()) == 0 || !item.value().is_boolean()) {
                res->error(format_error_response(
                    "telemetry_control accepts only the documented boolean controls", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        const auto value = [&](const char * name) {
            return source.contains(name) && source.at(name).get<bool>();
        };
        telemetry_control_state next;
        next.moe_routing = value("moe_routing");
        next.output_token_detail = value("output_token_detail");
        next.token_candidates = value("token_candidates");
        next.prompt_perplexity = value("prompt_perplexity");
        next.request_content = value("request_content");
        next.kv_pressure_detail = value("kv_pressure_detail");
        next.native_gpu_gpm = value("native_gpu_gpm");
        const telemetry_control_application application = ctx_server.telemetry_control_apply(next);
        json telemetry_control = ctx_server.telemetry_control_state_json(application.effective);
        telemetry_control["effective_from"] = application.effective_from;
        telemetry_control["applicability"] = ctx_server.telemetry_control_applicability_json();
        res->ok({
            {"success", true},
            {"telemetry_control", std::move(telemetry_control)},
        });
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true, ctx_server.init_opt);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_models);
        } else {
            res->ok(get_res_models(*meta));
        }
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens").get<llama_tokens>();
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i], ctx_server.init_opt);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                task.t_arrival = req.t_arrival;
                task.t_arrival_unix_ms = req.t_arrival_unix_ms;
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize").get<int>();
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);
            task.t_arrival = req.t_arrival;
            task.t_arrival_unix_ms = req.t_arrival_unix_ms;

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const mtmd_helper_init_opt & init_opt, const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(mctx, prompt.get<std::string>(), files, init_opt, true).size();
    } else {
        n_tokens = tokenize_mixed(vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}

void server_routes::update_cached_responses(bool is_sleeping) {
    // caller is task_queue, so ctx_server can be accessed without holding locks
    std::unique_lock<std::mutex> lock(mutex_cache);

    if (is_sleeping) {
        cached_models  = get_res_models(*meta);
        cached_props   = get_res_props(*meta, params, true);
        cached_metrics = ctx_server.get_metrics();

        should_reset_buckets = false;

        SRV_DBG("%s\n", "cached responses updated");

    } else if (should_reset_buckets) {
        // a scrape during sleep already reported these buckets
        ctx_server.reset_metrics_bucket();

        should_reset_buckets = false;
    }
}
