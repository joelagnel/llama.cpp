# llama-server telemetry v1

The private-fork telemetry API exposes bounded, local inference events and low-cardinality aggregate metrics. It is independent of the Prometheus endpoint and is available without `--metrics`.

On Windows, run `build-telemetry.ps1` from the repository root to configure and build a native x64 or ARM64 Release server. CUDA is enabled automatically when `nvcc.exe` is available; use `-Cuda On` to require it or `-Cuda Off` for a CPU build. Windows ARM64 uses `clang-cl` for llama.cpp's ARM CPU backend and the Visual Studio ARM64 compiler as nvcc's host compiler. Pass `-LlvmPath` when LLVM is not installed in a standard location and `-CudaArchitecture 121` for an RTX Spark/compute 12.1 build. `run-llama-telemetry.ps1` starts the matching native build with model, context, batching, speculative/MTP, Prometheus, and content-policy options.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\build-telemetry.ps1 -Cuda Auto
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\build-telemetry.ps1 `
  -Architecture ARM64 -Cuda On -CudaArchitecture 121
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run-llama-telemetry.ps1 `
  -ModelPath C:\models\model.gguf -Port 8080 -ContextLength 8192 `
  -GpuLayers all -ParallelSlots 4 -BatchSize 2048 -UBatchSize 512
```

Use `-SpecType` with `-SpecModelPath` for a draft backend, or `-MtpModelPath` to select `draft-mtp`. Add `-ContentLogging` only when local prompt and response retention is intended. `-Background` starts a hidden process and waits for `/health`.

## Endpoints

- `GET /telemetry/v1/capabilities` reports schema version, server instance, build, configured batching, content policy, clock semantics, and supported, conditional, or unsupported measurements.
- `GET /telemetry/v1/snapshot` reports coherent cumulative counters and current server state.
- `GET /telemetry/v1/events?cursor=N&limit=N` returns retained request events after a cursor. `limit` is clamped to 1 through 512.
- `GET /telemetry/v1/kv` reports model, context, and compute allocation by backend buffer type, typed memory components, authoritative entry occupancy where the active memory backend provides it, physical prefix sharing, bounded duplicate-prefix opportunities, and cache churn. Dense occupied bytes remain explicitly labeled as an estimate.

The model router proxies these routes to a selected model in the same way as the existing monitoring routes. Normal server API-key policy applies. Responses include `schema_version` and `server_instance_id`; consumers must reset their cursor when the instance changes.

## Request identity and clocks

Every completion or infill sequence has an opaque unique `trace_id`. Parallel completion children use distinct IDs. The server returns the root request ID in `X-Llama-Trace-Id` and includes the sequence ID in JSON results and events.

A valid W3C `traceparent` is retained separately as `w3c_traceparent`, with its normalized 32-hex trace ID in `w3c_trace_id`. These are correlation values, not unique request IDs. None of these identifiers is used as a Prometheus label.

Durations use `ggml_time_us()`, a high-resolution monotonic clock. Event wall timestamps are anchored to Unix milliseconds captured when the HTTP handler is dispatched, after cpp-httplib has read the request body. Therefore v1 TTFT is:

`first actual sampled model token - HTTP handler dispatch after body read`

It is not HTTP TTFB. Queue latency is `slot start - first enqueue`; deferral does not reset enqueue time.

## Phase and cache semantics

Final events keep cache lookup, actual prefill, first token, steady-state decode, and completion distinct. The following invariant is enforced by the engine's existing counters:

`prompt_tokens = reused_prompt_tokens + evaluated_prompt_tokens`

`matched_prefix_tokens` is captured before llama-server's required one-token logits replay. A full match therefore has `matched_prefix_tokens == prompt_tokens`, while `reused_prompt_tokens` is normally one lower and `evaluated_prompt_tokens` is one. Such events use `cache_status: "full"` and `prefill_meaningful: false`; clients must not present the replay as meaningful uncached prefill throughput.

Live events expose only values that are authoritative at that phase. `request_started` includes `prompt_tokens` and `server_configuration`; cache lookup has not happened yet, so it does not publish a provisional cache outcome. `first_token` includes prompt, matched, reused, and evaluated token counts, cache status and reuse ratio, `prefill_meaningful`, queue and TTFT, `cache_lookup_ms`, `actual_prefill_ms`, and the same server configuration. These fields let a live consumer present cache and prefill state while decoding continues and derive a meaningful prefill rate from evaluated tokens. All durations are milliseconds. On a full cache match, `actual_prefill_ms` describes the required logits replay and `prefill_meaningful` remains false.

TPOT and decode TPS use generation steps after the first model token. Queue and prefill time are excluded. Server output TPS is derived from timestamped deltas of `server_output_tokens_total`, which advances during generation.

Final events also retain the complete parsed sampler configuration, including requested and effective base temperature, requested and effective seed, sampler order, penalties, grammar, stop sequences, adapters, and speculative/MTP configuration. Dynamic-temperature range and exponent remain separate because the effective per-token temperature may vary.

## Event retention and privacy

Content logging is disabled by default. Metadata-only events never contain prompts or responses. Set `LLAMA_TELEMETRY_CONTENT=1` before starting the process to retain the original structured request, rendered prompt, and response in the in-memory event ring.

The ring retains at most 2,048 whole events and 64 MiB of serialized event data by default. Set `LLAMA_TELEMETRY_EVENT_BUFFER_MIB` to a value from 1 through 4096 to change the byte limit. Original requests above 4 MiB are explicitly marked omitted. Oversized events are dropped whole rather than silently truncating content.

An events response contains `cursor`, `oldest_sequence`, `next_sequence`, `gap`, `dropped_events`, `last_dropped_sequence`, and `retained_serialized_bytes`. Send the last consumed `cursor` on the next request. `next_sequence` is an allocation watermark, not the continuation cursor. If `gap` is true, retained history is incomplete for the supplied cursor.

## Response probability

Response probability is conditional on request `n_probs > 0`. It computes the selected emitted token's log probability from the raw target-model logits before sampler truncation. This adds no model inference, rejected draft tokens never contribute, accepted speculative output tokens contribute once, and replayed/discarded verification passes contribute nothing. The exact event semantic is:

- `raw_target_model_pre_sampler_selected_token_probability`.

The event reports the scored-token count, mean/minimum/maximum selected-token log probability, mean NLL, and `exp(mean_nll)`. Overflow is represented by a null perplexity and `perplexity_state: "overflow"`. Raw log-sum-exp is O(vocabulary) CPU work per emitted token, so it remains an explicit logprobs mode rather than default hot-path work.

Prompt perplexity is an independent, explicit diagnostic requested with `prompt_perplexity: true`. It scores token `i+1` from token `i`'s raw target logits. It performs no second inference pass, but it disables prompt-cache reuse and deliberately limits each scoring sequence to one prompt token per logical decode so the server does not reserve a full `batch size x vocabulary` logits buffer for ordinary requests. This performs O(prompt tokens x vocabulary) CPU work and many small logical decode calls, so it is diagnostic-only. The first token is conditioning only, so `scored_tokens == prompt_tokens - 1`. Multimodal prompts and prompts shorter than two text tokens report an explicit unavailable reason rather than an approximation.

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
