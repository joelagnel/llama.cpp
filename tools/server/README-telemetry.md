# llama-server telemetry v1

The private-fork telemetry API exposes bounded, local inference events and low-cardinality aggregate metrics. It is independent of the Prometheus endpoint and is available without `--metrics`.

On Windows, run `build-telemetry.ps1` from the repository root to configure and build a native x64 or ARM64 Release server. CUDA is enabled automatically when `nvcc.exe` is available; use `-Cuda On` to require it or `-Cuda Off` for a CPU build. Windows ARM64 uses `clang-cl` for llama.cpp's ARM CPU backend and the Visual Studio ARM64 compiler as nvcc's host compiler. Pass `-LlvmPath` when LLVM is not installed in a standard location and `-CudaArchitecture 121` for an RTX Spark/compute 12.1 build. `run-llama-telemetry.ps1` starts the matching native build with model, context, batching, speculative/MTP, Prometheus, and authenticated telemetry-control options.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\build-telemetry.ps1 -Cuda Auto
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\build-telemetry.ps1 `
  -Architecture ARM64 -Cuda On -CudaArchitecture 121
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run-llama-telemetry.ps1 `
  -ModelPath C:\models\model.gguf -ApiKeyFile C:\secure\llama-server.key `
  -Port 8080 -ContextLength 8192 `
  -GpuLayers all -ParallelSlots 4 -BatchSize 2048 -UBatchSize 512
```

`-ApiKeyFile` is required. It must be an absolute existing local file owned by the current Windows user with a protected ACL containing exactly current user, `SYSTEM`, and `BUILTIN\Administrators` full-control rules. The launcher verifies only its path and ACL: it never reads, prints, hashes, or places the key value in a URI or command line. It starts llama-server with the protected `--api-key-file` path and `--props`; the launcher rejects non-loopback hosts and extra arguments that could override `--host`, `--api-key-file`, or `--props`.

Use `-SpecType` with `-SpecModelPath` for a draft backend, or `-MtpModelPath` to select `draft-mtp`. `-Background` starts a hidden process and waits for `/health`. `-ContentLogging` is deprecated and fails closed. Content retention and every optional producer are controlled only by authenticated `POST /props` from LlamaScope; inherited telemetry activation variables are cleared before launch. Buffer and cap variables remain available for bounded retention tuning.

## Endpoints

- `GET /telemetry/v1/capabilities` reports schema version, server instance, build, configured batching, content policy, clock semantics, and supported, conditional, or unsupported measurements.
- `GET /telemetry/v1/snapshot` reports coherent cumulative counters and current server state. KV occupancy is the immutable shallow snapshot captured at the latest decode boundary.
- `GET /telemetry/v1/events?cursor=N&limit=N` returns retained request events after a cursor. `limit` is clamped to 1 through 512.
- `GET /telemetry/v1/kv` reports model, context, and compute allocation by backend buffer type and the immutable shallow KV snapshot captured at the latest decode boundary. Add `?detail=deep` to collect typed memory components, physical prefix sharing, bounded duplicate-prefix opportunities, and cache churn from one immutable boundary. Dense occupied bytes remain explicitly labeled as an estimate.
- `GET /telemetry/v1/gpu?cursor=N&limit=N&trace_id=ID` reports bounded llama-server-owned asynchronous NVML GPM intervals and correlated prefill, normal-decode, MTP-draft, and MTP-verify operation spans. SM ID 2, Tensor ID 5, and DRAM-bandwidth ID 10 retain independent state, reason, and nullable value fields. Stable source descriptors also retain state/reason when no interval exists, so unsupported and disabled hosts still export all three counters.

The model router proxies these routes to a selected model in the same way as the existing monitoring routes. Normal server API-key policy applies. Responses include `schema_version` and `server_instance_id`; consumers must reset their cursor when the instance changes.

## Runtime telemetry control

`POST /props` is the single control endpoint. It is available only when the server was started with `--props`, has a configured non-empty API key, and is loopback-bound. The normal API-key middleware still authenticates the call. The same restrictions apply to the router before it proxies a model-targeted request.

Use a literal `127.0.0.1` or `::1` listener for control. At this source revision the guard checks the configured host name, not the numeric address returned by the bound socket. The actual-listener loopback hardening correction is a release requirement; do not treat `localhost` or any DNS result as proof that control is safely local.

```json
{
  "model": "optional-router-model-id",
  "telemetry_control": {
    "output_token_detail": true,
    "request_content": true
  }
}
```

`telemetry_control` is a full replacement, not a patch. Its only accepted fields are `moe_routing`, `output_token_detail`, `token_candidates`, `prompt_perplexity`, `request_content`, `kv_pressure_detail`, and `native_gpu_gpm`; every omitted field is false, so `{}` explicitly disables every control. The reply returns the complete effective set, per-control applicability, a monotonically increasing generation, and the boundary at which each control takes effect. `moe_routing`, `kv_pressure_detail`, and `native_gpu_gpm` take effect at the next microbatch. The remaining request-scoped controls take effect for the next request. Last apply wins, and state survives until an explicit replacement or process restart. There is no lease, owner count, union, persistence, or second control route.

`GET /telemetry/v1/capabilities` advertises this static contract. `GET /telemetry/v1/snapshot` includes the current effective set and generation. Environment variables may still set telemetry bounds and buffer sizes, but they cannot enable a telemetry control.

Request booleans remain opt-out switches: an explicit false disables that feature for the request, while true cannot enable a globally disabled control. Requests snapshot `prompt_perplexity`, `output_token_telemetry`, `output_token_candidate_telemetry`, and `request_content` when they are accepted. `moe_routing_telemetry` is a permanent request opt-out only: absent or true permits capture only while the current microbatch control has `moe_routing: true`.

## Request identity and clocks

Every completion or infill sequence has an opaque unique `trace_id`. Parallel completion children use distinct IDs. The server returns the root request ID in `X-Llama-Trace-Id` and includes the sequence ID in JSON results and events.

A valid W3C `traceparent` is retained separately as `w3c_traceparent`, with its normalized 32-hex trace ID in `w3c_trace_id`. These are correlation values, not unique request IDs. None of these identifiers is used as a Prometheus label.

Durations use `ggml_time_us()`, a high-resolution monotonic clock. Event wall timestamps are anchored to Unix milliseconds captured when the HTTP handler is dispatched, after cpp-httplib has read the request body. Every request event also carries a content-independent `lifecycle_clock` v1 block in the `server_process_monotonic_microseconds` domain. Its nullable boundaries cover arrival, enqueue, slot start, cache start/end, prefill start/end, first token, last generation work, finalization start, and actual slot release. A missing boundary stays null; consumers must not stretch a measured phase across it. Therefore v1 TTFT is:

`first actual sampled model token - HTTP handler dispatch after body read`

It is not HTTP TTFB. Queue latency is `slot start - first enqueue`; deferral does not reset enqueue time. The completion event is prepared during finalization but appended only after the slot has entered its idle state, so `slot_release_monotonic_us` and `slot_release_unix_ms` identify the actual release without delaying the API response. `slot_id` is zero-based. `slot_assignment_ordinal` is server-instance-scoped and increases for every assignment, including reuse of the same slot; only adjacent ordinals on the same server instance and slot prove consecutive use.

## Phase and cache semantics

Final events keep cache lookup, actual prefill, first token, steady-state decode, and completion distinct. The following invariant is enforced by the engine's existing counters:

`prompt_tokens = reused_prompt_tokens + evaluated_prompt_tokens`

`matched_prefix_tokens` is captured before llama-server's required one-token logits replay. A full match therefore has `matched_prefix_tokens == prompt_tokens`, while `reused_prompt_tokens` is normally one lower and `evaluated_prompt_tokens` is one. Such events use `cache_status: "full"` and `prefill_meaningful: false`; clients must not present the replay as meaningful uncached prefill throughput.

Live events expose only values that are authoritative at that phase. `request_started` includes `prompt_tokens` and `server_configuration`; cache lookup has not happened yet, so it does not publish a provisional cache outcome. `first_token` includes prompt, matched, reused, and evaluated token counts, cache status and reuse ratio, `prefill_meaningful`, queue and TTFT, `cache_lookup_ms`, `actual_prefill_ms`, and the same server configuration. These fields let a live consumer present cache and prefill state while decoding continues and derive a meaningful prefill rate from evaluated tokens. All durations are milliseconds. On a full cache match, `actual_prefill_ms` describes the required logits replay and `prefill_meaningful` remains false.

TPOT and decode TPS use generation steps after the first model token. Queue and prefill time are excluded. Server output TPS is derived from timestamped deltas of `server_output_tokens_total`, which advances during generation.

Final events also retain the complete parsed sampler configuration, including requested and effective base temperature, requested and effective seed, sampler order, penalties, grammar, stop sequences, adapters, and speculative/MTP configuration. Dynamic-temperature range and exponent remain separate because the effective per-token temperature may vary.

## Event retention and privacy

Content logging is disabled by default. Metadata-only events never contain prompts or responses. Enable `request_content` through `POST /props`, then leave the request-level `request_content` option unset (or set it true) to retain the original structured request, rendered prompt, and response in the in-memory event ring.

The ring retains at most 2,048 whole events and 64 MiB of serialized event data by default. Set `LLAMA_TELEMETRY_EVENT_BUFFER_MIB` to a value from 1 through 4096 to change the byte limit. Original requests above 4 MiB are explicitly marked omitted. Oversized events are dropped whole rather than silently truncating content.

An events response contains `cursor`, `oldest_sequence`, `next_sequence`, `gap`, `gap_ranges`, `dropped_events`, `last_dropped_sequence`, and `retained_serialized_bytes`. Send the last consumed `cursor` on the next request. `next_sequence` is an allocation watermark, not the continuation cursor. If `gap` is true, retained history is incomplete for the supplied cursor. Each `gap_ranges` item names only the exact missing global event-sequence interval; it does not invent token, layer, or model-position coordinates for lost data. A cursor ahead of the server high-water mark is reset to that mark with `gap: true`; `gap_ranges` stays empty because no unallocated future sequence is reported as lost.

## Response probability

Response probability is normally conditional on request `n_probs > 0`. When all three trusted-local diagnostic gates described below are enabled, llama-server raises an omitted or zero request value to `n_probs = 1`. It computes the selected emitted token's log probability from the raw target-model logits before sampler truncation. This adds no model inference, rejected draft tokens never contribute, accepted speculative output tokens contribute once, and replayed/discarded verification passes contribute nothing. The exact event semantic is:

- `raw_target_model_pre_sampler_selected_token_probability`.

The event reports the scored-token count, mean/minimum/maximum selected-token log probability, mean NLL, and `exp(mean_nll)`. Overflow is represented by a null perplexity and `perplexity_state: "overflow"`. Raw log-sum-exp is O(vocabulary) CPU work per emitted token, so it remains an explicit logprobs mode rather than default hot-path work.

Prompt perplexity is an independent, explicit diagnostic requested with `prompt_perplexity: true`. It scores token `i+1` from token `i`'s raw target logits. It performs no second inference pass, but it disables prompt-cache reuse and deliberately limits each scoring sequence to one prompt token per logical decode so the server does not reserve a full `batch size x vocabulary` logits buffer for ordinary requests. This performs O(prompt tokens x vocabulary) CPU work and many small logical decode calls, so it is diagnostic-only. The first token is conditioning only, so `scored_tokens == prompt_tokens - 1`. Multimodal prompts and prompts shorter than two text tokens report an explicit unavailable reason rather than an approximation.

## Bounded output-token diagnostics

Per-token detail is disabled by default. Enable `output_token_detail` through `POST /props`; accepted requests snapshot that setting, and may explicitly opt out with `output_token_telemetry: false`. `token_candidates` is independently controlled and requires both global controls for a request that sets `output_token_candidate_telemetry: true`. Set `request_content` globally only when token IDs and pieces, prompts, and response text may be retained. `LLAMA_TELEMETRY_OUTPUT_TOKEN_LIMIT` sets a hard 32--8192 committed-token cap (default 512). The richer speculative records have independent caps: `LLAMA_TELEMETRY_MTP_PASS_LIMIT` and `LLAMA_TELEMETRY_MTP_PROPOSAL_LIMIT` each accept 32 through 4096 with a default of 512, while `LLAMA_TELEMETRY_TOKEN_CANDIDATE_DECISION_LIMIT` accepts 8 through 4096 with a default of 512. Raising the committed-output cap therefore does not retain thousands of MTP proposal or candidate distributions. The disabled path does not allocate the per-request record buffer, take another token-detail clock sample, serialize token rows, or compute probability.

Output-token record schema v3 retains the zero-based output ordinal, the existing request-relative model-ready offset, the absolute process-monotonic model-ready timestamp used to align concurrent sequences, authoritative Base64 tokenizer-piece bytes under the content policy, and the target-model context position derived directly from the evaluated logits-row position plus one. It also classifies each committed token as ordinary target decode, accepted MTP proposal, target replacement after the first mismatch, or target bonus after a full accepted chain. MTP-origin tokens retain their zero-based logical step, actual target pass, proposal position, accepted depth, proposed depth, and whether the committing pass was a checkpoint replay. Multiple tokens committed by one verification pass intentionally share one model-ready timestamp.

Raw selected-token log probability remains independently conditional on `n_probs > 0`, and token ID remains independently conditional on `request_content`. Every record carries separate state and reason fields for model position, probability, token identity, origin, and MTP linkage; ordinary decode reports MTP linkage as `not_applicable`. Older v1 records do not acquire inferred positions or linkage in consumers--those fields remain `not_captured`.

This committed-token block does not replace the request-level speculative counters or the `/telemetry/v1/gpu` operation ledger. A target verification pass that is discarded before committing output has no token row, but remains real work in actual-pass totals and native GPU operation evidence. Schema v3 also retains a bounded MTP proposal ledger. Target top-K candidate distributions remain in the separate bounded `/telemetry/v1/token-candidates` block and are never placed in the normal event payload.

## MoE routing chunks

Static MoE configuration is always reported when the loaded model exposes it. Dynamic routing capture is disabled by default. Enable `moe_routing` through `POST /props`. A request with `moe_routing_telemetry: false` is permanently excluded; absent or true permits capture only for microbatches where the global control is enabled. Environment variables can tune the existing bounded legacy histogram limit, but cannot enable routing capture.

When active, `GET /telemetry/v1/events` emits canonical schema-v2 `moe_routing_chunk` events. The native readback is copied before another decode or routing toggle can invalidate it. The intended population is every routed token-layer row produced for opted-in prefill, normal decode, and MTP verification passes. There is no per-request byte cap. Each chunk is at most `min(1 MiB, serialized event capacity)` and holds whole decisions only; a decision is never split across chunks. The final retained chunk for a trace carries `is_final_for_trace: true`; a no-record trace can instead have a final availability marker without a `decisions` array.

The v2 chunk schema keeps the global ring `sequence` and `server_instance_id` for transport order. `trace_chunk_sequence` and `trace_decision_sequence` are separate, one-based sequences within one request trace. `serialized_bytes` is the exact UTF-8 size retained for that event. `producer_coverage`, `invalid_records`, explicit coordinate-bearing gaps, and `unlocated_coverage_loss` distinguish retained evidence from unavailable evidence. An unlocated loss must never be projected onto a model position, token, or layer. A global ring overrun is transport loss, not a zero-valued route; use the response `gap_ranges` only for its global event-sequence interval.

Each decision includes physical-step and microbatch identity, graph row, model position, phase, layer, control generation, and request/MTP linkage. MTP decisions carry logical step, actual target pass, proposal position, and replay state. `physical_peer_trace_ids` and `physical_peer_coverage` identify whether the concurrent physical event has complete opted-in peer evidence. `shared_experts` remains separate from routed `selected_experts`.

Each selected expert has its exact final effective coefficient and separate numeric status. `kth_selected_score` is the selected Kth score and `highest_rejected_score` is the highest rejected K+1 score. Their status fields, and the selected-expert ID/weight status fields, use the native ABI: `0` valid, `1` source unavailable, `2` invalid, and `3` nonfinite. A non-valid value is null with an optional reason; consumers must not replace it with zero. The scores are selection-source boundary values after selection bias and group masking, not a full router distribution.

The disabled and dense paths do not add graph outputs, readback, allocation, synchronization, or batch-peer instrumentation. The existing bounded completion-event histogram remains available for compatibility. `LLAMA_TELEMETRY_MOE_ACTIVATION_LIMIT` applies to that legacy summary only; it does not truncate the full-request routing chunks.

Before release acceptance, schema-v2 still requires raw producer-byte validation through the managed mapper, CUDA readback coverage, and forced unavailable, unlocated, overrun, and mid-request control-boundary cases. The current generic server tests do not by themselves prove those cases or make coverage counts safe to upgrade from `partial` to `complete`. Until those focused gates pass, consumers must preserve every availability and gap field and present routing evidence conservatively.

Dense models report `not_applicable`. An MoE model can report configuration `available` while legacy request detail is `disabled`, `not_enabled_for_request`, `no_data`, `available`, `partial`, or `truncated`. Histogram, selected-ID, effective-weight, normalized-share, and margin states and reasons remain independent. A backend that exposes IDs but not a usable value keeps the measured ID state and reports the missing value as unavailable; no consumer should substitute zero.

The reusable `scripts/Benchmark-MoeRoutingTelemetry.ps1` alternates matched request-off/request-on samples and validates every emitted array, selected-share sum, and derived margin. A 100-pair, 128-token, single-thread ARM64 CPU run on `stories15M_MOE-F16.gguf` measured request-off/on medians of 393.82/393.38 token/s, 17.0915/17.2270 ms TTFT, and 2.53925/2.54207 ms inter-token time. The median paired token/s delta was -0.607%. This tiny CPU result is within the provisional one-percent diagnostic-bookkeeping budget, but it is a measurable signal and not a claim that the feature is free; representative GPU, concurrency, and architecture testing remains required.

## Prometheus additions

`/metrics` remains standards-compatible and still requires `--metrics`. Existing metric semantics, including `llamacpp:n_decode_total`, are unchanged. V1 adds cumulative counters and histograms without request IDs or content labels:

- request outcomes and prompt-cache miss, partial-hit, and full-hit counters;
- live server output tokens;
- successful logical decode calls and logical tokens;
- TTFT, queue, actual-prefill, TPOT, E2E, prompt-token, output-token, and reuse-ratio histograms;
- logical tokens and participating slots per decode histograms;
- actual physical target- and draft-context `llama_ubatch` counters and token histograms;
- speculative draft/accepted depth, token and verification-step outcomes, full-chain outcomes, target tokens, and useful output tokens per target pass.

Histogram buckets are cumulative and expose `_bucket`, `_sum`, `_count`, and `le="+Inf"`. Cumulative series are never reset by a scrape. Existing short-window throughput gauges retain their historical scrape-reset behavior.

## Capability boundaries

Physical ubatch observation, standard-KV live entry occupancy, authoritative sequence-membership sharing, bounded duplicate-prefix opportunity estimates, exact optional prompt perplexity, and speculative response perplexity are implemented in v1. Memory diagnostics are polymorphic: recurrent and composite backends describe their own entry semantics, and a field that is not meaningful for that backend reports `not_applicable` rather than zero. Active KV defragmentation is not implemented by current llama.cpp memory backends and is therefore `not_applicable`, not an unsupported numeric metric.

The remaining non-metric boundaries are structured final tool-call extraction in telemetry events, lifecycle events for requests rejected before a slot is assigned, and prompt perplexity for multimodal inputs. The capability response distinguishes disabled, conditional, unavailable, and not-applicable states from a measured value of zero. Slot token totals are labeled `resident_tokens_upper_bound` and must not be presented as physical KV occupancy.
