# Rebasing the LlamaScope telemetry patch stack

This checkout is a private, reviewable fork of `ggml-org/llama.cpp`. The
LlamaScope telemetry work is intentionally kept as a small ordered commit stack
on top of an unmodified upstream commit.

## Current base and stack

- Upstream remote: `https://github.com/ggml-org/llama.cpp.git`
- Baseline commit: `9a286ac98d2cab74231bd3f1fc3f2b8bdf05422e`
- Branch: `llamascope/telemetry-v1`
- Commit 1: typed core memory diagnostics
- Commit 2: versioned server telemetry APIs and metrics
- Commit 3: telemetry contract tests, scripts, and documentation

The canonical mailbox copies of these commits are also stored in the outer
LlamaScope repository under `patches/llama.cpp/`.

## Update procedure

Work in a disposable branch and preserve the old branch until validation is
complete:

```powershell
git fetch upstream
git switch llamascope/telemetry-v1
git branch backup/telemetry-v1-before-rebase
git rebase --onto upstream/master 9a286ac98d2cab74231bd3f1fc3f2b8bdf05422e
```

Resolve conflicts according to current llama.cpp semantics. Do not retain an
old counter merely to avoid a conflict: first classify whether upstream now
exports, internally measures, or makes the value derivable. Keep the following
distinctions intact:

- cache reuse versus actual prefill;
- TTFT versus HTTP TTFB and steady-state decode;
- logical `llama_decode()` batch versus physical `llama_ubatch`;
- speculative token acceptance versus verification-step acceptance;
- target tokens evaluated versus useful output tokens committed;
- allocated context memory versus authoritative live occupancy;
- equivalent prefixes versus physically shared memory state.

After the rebase, regenerate the outer patch series from the new upstream base,
update `patches/llama.cpp/BASE_COMMIT`, and run the native build and telemetry
contract tests described in `tools/server/README-telemetry.md`.

## Validation gates

At minimum validate:

1. Windows x64 Release build of `llama-server`.
2. `tools/server/tests/unit/test_telemetry.py`.
3. Prometheus histogram format and cumulative behavior.
4. Structured capability, snapshot, event cursor, and KV endpoints.
5. Cold, partial-hit, full-hit, concurrent, speculative, cancellation, and
   restart scenarios.
6. Telemetry-off versus default-on throughput and latency overhead.

Never push this branch to the official upstream remote. LlamaScope's outer
repository is the distribution and review boundary for this private patch
stack.
