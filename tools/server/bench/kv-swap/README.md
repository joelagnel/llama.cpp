# KV swap benchmark

`kv_swap_bench.py` runs the two requested workloads against four server configurations:

1. unified KV without swap;
2. unified KV with an 8192 MiB pinned-host arena;
3. non-unified KV at the same total context/device budget;
4. unified KV without swap, with session admission limited to work that fits.

The default is the 128K, four-slot experiment. Use `--quick` for the scaled 5120-cell acceptance run used during development.
The default workload uses 33K-token prompts so aggregate demand reliably exceeds the 128K device cache; the admission control limits concurrency to work that fits.

```powershell
python .\tools\server\bench\kv-swap\kv_swap_bench.py `
  --server .\build\bin\llama-server.exe `
  --model C:\models\Qwen3.6-35B-A3B.gguf `
  --output .\kv-swap-results
```

The output directory contains server logs, `results.json`, sampled slot/metrics/memory timelines, and a self-contained `report.html`. The runner records HTTP failures, useful tokens, task goodput, TTFT, decode rate, latency, inferred queue time, swap stalls, copy bytes/durations/bandwidth, page counts, overlap, and GPU/host/process/system memory high-water marks.

If `test-kv-swap` is next to `llama-server`, the runner also executes it against the supplied model. That test compares a no-swap state snapshot with a fragmented D2H/H2D round trip byte-for-byte and checks the next deterministic token. Use `--skip-correctness` only when that extra model load is undesirable.

The MVP implementation is CUDA-only and requires unified KV. It swaps the growing attention cache; fixed recurrent/GDN state remains device-resident. The CLI is backend-neutral so another backend can implement the same page interface later.
