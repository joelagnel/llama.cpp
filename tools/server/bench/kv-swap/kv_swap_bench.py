#!/usr/bin/env python3
"""Four-configuration KV swap benchmark and self-contained HTML reporter."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import html
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


CONFIGS = {
    "unified_no_swap": ["--kv-unified"],
    "unified_swap": ["--kv-unified", "--kv-swap"],
    "non_unified": ["--no-kv-unified"],
    "admission_queue": ["--kv-unified"],
}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def task_completed(task: dict) -> bool:
    return (task.get("status") == 200 and not task.get("error") and
            len(task.get("tokens", [])) >= task.get("expected_tokens", 0))


def request_json(base: str, method: str, path: str, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(base + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        return response.status, json.loads(raw) if raw else None


def request_text(base: str, path: str, timeout=30) -> str:
    with urllib.request.urlopen(base + path, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def parse_metrics(text: str) -> dict[str, float]:
    result = {}
    for line in text.splitlines():
        if not line.startswith("llamacpp:") or "{" in line:
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                result[parts[0][len("llamacpp:"):]] = float(parts[1])
            except ValueError:
                pass
    return result


def process_memory(pid: int) -> dict[str, int]:
    if os.name != "nt":
        try:
            fields = Path(f"/proc/{pid}/status").read_text().splitlines()
            values = {line.split(":", 1)[0]: line for line in fields}
            rss = int(values["VmRSS"].split()[1]) * 1024
            return {"process_rss": rss, "process_private": rss}
        except (OSError, KeyError, ValueError):
            return {}

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    handle = ctypes.windll.kernel32.OpenProcess(0x1000 | 0x0400, False, pid)
    if not handle:
        return {}
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    ctypes.windll.kernel32.CloseHandle(handle)
    if not ok:
        return {}
    return {"process_rss": counters.WorkingSetSize, "process_private": counters.PrivateUsage}


def system_memory() -> dict[str, int]:
    if os.name != "nt":
        return {}

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {}
    return {"system_ram_used": status.ullTotalPhys - status.ullAvailPhys,
            "system_ram_total": status.ullTotalPhys}


def gpu_sample() -> dict[str, float]:
    try:
        proc = subprocess.run([
            "nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
        used, total, util = [float(v.strip()) for v in proc.stdout.splitlines()[0].split(",")]
        return {"gpu_mib": used, "gpu_total_mib": total, "gpu_util": util}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


class Monitor:
    def __init__(self, base: str, process: subprocess.Popen, interval: float):
        self.base = base
        self.process = process
        self.interval = interval
        self.samples: list[dict] = []
        self.phase = "startup"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.t0 = time.perf_counter()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=10)

    def _run(self):
        gpu = {}
        next_gpu = 0.0
        while not self.stop_event.is_set() and self.process.poll() is None:
            now = time.perf_counter()
            sample = {"t": now - self.t0, "phase": self.phase}
            try:
                sample["slots"] = request_json(self.base, "GET", "/slots", timeout=5)[1]
                sample["metrics"] = parse_metrics(request_text(self.base, "/metrics", timeout=5))
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                pass
            sample.update(process_memory(self.process.pid))
            sample.update(system_memory())
            if now >= next_gpu:
                gpu = gpu_sample()
                next_gpu = now + 1.0
            sample.update(gpu)
            self.samples.append(sample)
            self.stop_event.wait(self.interval)


def stream_completion(base: str, payload: dict, timeout: int) -> dict:
    started = time.perf_counter()
    first = None
    tokens = []
    content = []
    final = {}
    status = 0
    error = None
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(base + "/completion", data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if first is None:
                    first = time.perf_counter()
                if event.get("error"):
                    error = json.dumps(event["error"], ensure_ascii=False)
                tokens.extend(event.get("tokens", []))
                content.append(event.get("content", ""))
                if event.get("stop"):
                    final = event
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read().decode("utf-8", "replace")
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        error = str(exc)
    ended = time.perf_counter()
    timings = final.get("timings", {})
    latency_ms = (ended - started) * 1000
    queue_ms = max(0.0, latency_ms - timings.get("prompt_ms", 0) - timings.get("predicted_ms", 0))
    return {
        "status": status, "error": error, "tokens": tokens, "content": "".join(content),
        "token_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "started": started, "ended": ended, "latency_ms": latency_ms,
        "ttft_ms": ((first or ended) - started) * 1000, "queue_ms": queue_ms,
        "timings": timings, "stop_type": final.get("stop_type"),
        "stopped_eos": final.get("stopped_eos"),
        "stopped_limit": final.get("stopped_limit"),
    }


def erase_slots(base: str, n_parallel: int):
    for slot in range(n_parallel):
        try:
            request_json(base, "POST", f"/slots/{slot}?action=erase", timeout=30)
        except (OSError, urllib.error.URLError):
            pass


def make_prompt(base: str, target: int, label: str) -> tuple[str, int]:
    unit = "alpha beta gamma delta epsilon zeta eta theta iota kappa. "
    status, tokenized = request_json(base, "POST", "/tokenize", {"content": unit}, timeout=30)
    if status != 200:
        raise RuntimeError("tokenization failed")
    per_unit = max(1, len(tokenized["tokens"]) - 1)
    repeats = max(1, math.ceil((target - 24) / per_unit))
    prompt = f"Session {label}. Retain every detail below.\n" + unit * repeats
    _, tokenized = request_json(base, "POST", "/tokenize", {"content": prompt}, timeout=60)
    return prompt, len(tokenized["tokens"])


def run_continuous(args, base: str, config: str) -> list[dict]:
    prompts = [make_prompt(base, args.prompt_tokens, f"continuous-{i}") for i in range(args.parallel)]
    barrier = threading.Barrier(args.parallel)
    semaphore = threading.Semaphore(args.admission_limit) if config == "admission_queue" else None
    results = []
    lock = threading.Lock()

    def worker(slot: int):
        prompt, actual = prompts[slot]
        barrier.wait()
        # Let the three long-lived requests establish their cache first.  The
        # fourth request is deliberately newer and short, which makes this a
        # direct check that it can progress while an older request is evicted.
        if slot == args.parallel - 1:
            time.sleep(args.short_delay)
        if semaphore:
            semaphore.acquire()
        try:
            expected = args.continuous_predict if slot < args.parallel - 1 else max(32, args.continuous_predict // 8)
            payload = {
                "prompt": prompt + "\nContinue the analysis in a deterministic numbered list.",
                "n_predict": expected, "id_slot": slot, "cache_prompt": True,
                "stream": True, "return_tokens": True, "ignore_eos": True,
                "temperature": 0, "seed": 1234,
            }
            result = stream_completion(base, payload, args.request_timeout)
            result.update({"workload": "continuous", "task": f"continuous/{slot}",
                           "slot": slot, "prompt_tokens_requested": args.prompt_tokens,
                           "prompt_tokens_actual": actual, "expected_tokens": expected})
            with lock:
                results.append(result)
        finally:
            if semaphore:
                try:
                    erase_slots(base, args.parallel) if args.admission_limit == 1 else request_json(
                        base, "POST", f"/slots/{slot}?action=erase", timeout=30)
                except (OSError, urllib.error.URLError):
                    pass
                finally:
                    semaphore.release()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.parallel)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def run_agent(args, base: str, config: str) -> list[dict]:
    prompts = [make_prompt(base, args.prompt_tokens, f"agent-{i}") for i in range(args.parallel)]
    barrier = threading.Barrier(args.parallel)
    semaphore = threading.Semaphore(args.admission_limit) if config == "admission_queue" else None
    results = []
    lock = threading.Lock()

    def session(slot: int):
        prompt, actual = prompts[slot]
        barrier.wait()
        if semaphore:
            semaphore.acquire()
        try:
            conversation = prompt
            for turn in range(args.agent_turns):
                payload = {
                    "prompt": conversation + f"\nTurn {turn}: propose the next tool call and explain it.",
                    "n_predict": args.agent_predict, "id_slot": slot, "cache_prompt": True,
                    "stream": True, "return_tokens": True, "ignore_eos": True,
                    "temperature": 0, "seed": 1234,
                }
                result = stream_completion(base, payload, args.request_timeout)
                result.update({"workload": "agent", "task": f"agent/{slot}/{turn}",
                               "slot": slot, "turn": turn,
                               "prompt_tokens_requested": args.prompt_tokens,
                               "prompt_tokens_actual": actual,
                               "expected_tokens": args.agent_predict})
                with lock:
                    results.append(result)
                if not task_completed(result):
                    break
                conversation += result["content"] + f"\nTool result {turn}: success; value={slot * 100 + turn}."
                time.sleep(args.tool_pause)
        finally:
            if semaphore:
                try:
                    request_json(base, "POST", f"/slots/{slot}?action=erase", timeout=30)
                except (OSError, urllib.error.URLError):
                    pass
                semaphore.release()

    threads = [threading.Thread(target=session, args=(i,)) for i in range(args.parallel)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def start_server(args, config: str, port: int, output: Path):
    cmd = [str(args.server), "-m", str(args.model), "--host", "127.0.0.1", "--port", str(port),
           "--device", args.device, "-ngl", args.gpu_layers, "-c", str(args.ctx_size),
           "-np", str(args.parallel), "-b", str(args.batch), "-ub", str(args.ubatch),
           "--metrics", "--slots", "--no-warmup", "--no-context-shift"]
    cmd += CONFIGS[config]
    if config == "unified_swap":
        cmd += ["--kv-swap-max-mib", str(args.swap_mib)]
    cmd += args.extra_server_arg
    log = (output / f"server-{config}.log").open("w", encoding="utf-8")
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    process = subprocess.Popen(cmd, cwd=args.server.parent, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=creationflags)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + args.startup_timeout
    while time.time() < deadline and process.poll() is None:
        try:
            if request_json(base, "GET", "/health", timeout=2)[0] == 200:
                return process, log, base, cmd
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(0.5)
    log.flush()
    raise RuntimeError(f"server {config} failed to become healthy; see {log.name}")


def run_correctness(args) -> dict:
    executable = args.correctness_test
    if executable is None:
        name = "test-kv-swap.exe" if os.name == "nt" else "test-kv-swap"
        candidate = args.server.parent / name
        executable = candidate if candidate.exists() else None
    if executable is None:
        return {"available": False, "error": "test-kv-swap executable not found"}

    executable = executable.resolve()
    command = [str(executable), "-m", str(args.model), "--device", args.device,
               "-ngl", args.gpu_layers, "-c", "1024", "-np", "2"]
    try:
        completed = subprocess.run(command, cwd=executable.parent, capture_output=True, text=True,
                                   timeout=args.startup_timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": True, "command": command, "returncode": -1, "error": str(exc)}

    output = completed.stdout + completed.stderr
    match = re.search(r"KV_SWAP_CORRECTNESS\s+(\{[^\r\n]+\})", output)
    values = json.loads(match.group(1)) if match else {}
    return {"available": True, "command": command, "returncode": completed.returncode,
            "values": values, "output": output}


def stop_server(process: subprocess.Popen, log):
    if process.poll() is None:
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_C_EVENT)
            else:
                process.send_signal(signal.SIGINT)
            process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    log.close()


def summarize_task_group(tasks: list[dict], samples: list[dict], phase: str) -> dict:
    successful = [task for task in tasks if task_completed(task)]
    start = min((task["started"] for task in tasks), default=0)
    end = max((task["ended"] for task in tasks), default=start)
    duration = max(0.001, end - start)
    tokens = sum(len(task["tokens"]) for task in successful)
    phase_samples = [sample for sample in samples if sample.get("phase") == phase]
    metrics_first = next((s.get("metrics", {}) for s in phase_samples if s.get("metrics")), {})
    metrics_last = next((s.get("metrics", {}) for s in reversed(phase_samples) if s.get("metrics")), {})
    delta = lambda key: max(0.0, metrics_last.get(key, 0) - metrics_first.get(key, 0))
    d2h_bytes = delta("kv_swap_d2h_bytes_total")
    h2d_bytes = delta("kv_swap_h2d_bytes_total")
    d2h_s = delta("kv_swap_d2h_seconds_total")
    h2d_s = delta("kv_swap_h2d_seconds_total")
    stall_by_slot = {}
    for sample in phase_samples:
        for slot in sample.get("slots", []):
            swap = slot.get("kv_swap", {})
            stall_by_slot[slot.get("id")] = max(
                stall_by_slot.get(slot.get("id"), 0), swap.get("swap_stall_ms", 0))
    total_peak = max([
        s.get("process_private", 0) + s.get("gpu_mib", 0) * 2**20
        for s in phase_samples
    ] or [0])
    return {
        "tasks": len(tasks), "successful": len(successful), "failed": len(tasks) - len(successful),
        "useful_tokens": tokens, "duration_s": duration,
        "tasks_per_min": len(successful) * 60 / duration, "tokens_per_min": tokens * 60 / duration,
        "ttft_p50_ms": percentile([t["ttft_ms"] for t in successful], 0.5),
        "ttft_p95_ms": percentile([t["ttft_ms"] for t in successful], 0.95),
        "latency_p50_ms": percentile([t["latency_ms"] for t in successful], 0.5),
        "latency_p95_ms": percentile([t["latency_ms"] for t in successful], 0.95),
        "queue_mean_ms": statistics.mean([t["queue_ms"] for t in successful]) if successful else 0,
        "decode_tokens_s": statistics.mean([t["timings"].get("predicted_per_second", 0)
                                               for t in successful]) if successful else 0,
        "d2h_bytes": d2h_bytes, "h2d_bytes": h2d_bytes, "d2h_seconds": d2h_s, "h2d_seconds": h2d_s,
        "d2h_gbps": d2h_bytes / d2h_s / 1e9 if d2h_s else 0,
        "h2d_gbps": h2d_bytes / h2d_s / 1e9 if h2d_s else 0,
        "overlap_seconds": delta("kv_swap_overlap_seconds_total"),
        "pages_out": delta("kv_swap_pages_out_total"), "pages_in": delta("kv_swap_pages_in_total"),
        "swap_stall_ms": sum(stall_by_slot.values()),
        "host_peak_bytes": max([s.get("metrics", {}).get("kv_swap_host_bytes_peak", 0)
                                for s in phase_samples] or [0]),
        "gpu_peak_mib": max([s.get("gpu_mib", 0) for s in phase_samples] or [0]),
        "process_private_peak": max([s.get("process_private", 0) for s in phase_samples] or [0]),
        "system_ram_peak": max([s.get("system_ram_used", 0) for s in phase_samples] or [0]),
        "tracked_total_peak": total_peak,
    }


def svg_bars(rows: list[tuple[str, float]], title: str, unit: str) -> str:
    width, row_h = 700, 30
    maximum = max([value for _, value in rows] or [1]) or 1
    parts = [f'<svg viewBox="0 0 {width} {55 + row_h * len(rows)}" role="img">',
             f'<text x="0" y="18" class="chart-title">{html.escape(title)}</text>']
    for i, (label, value) in enumerate(rows):
        y = 35 + i * row_h
        bar = 430 * value / maximum
        parts += [f'<text x="0" y="{y + 15}">{html.escape(label)}</text>',
                  f'<rect x="220" y="{y}" width="{bar:.1f}" height="18" rx="3"/>',
                  f'<text x="{230 + bar:.1f}" y="{y + 15}">{value:.2f} {unit}</text>']
    parts.append("</svg>")
    return "".join(parts)


def timeline_svg(samples: list[dict], phase: str) -> str:
    data = [s for s in samples if s.get("phase") == phase and s.get("slots")]
    if not data:
        return "<p>No slot samples were collected.</p>"
    t0, t1 = data[0]["t"], data[-1]["t"]
    span = max(0.001, t1 - t0)
    colors = {"resident": "#43aa8b", "generating": "#277da1", "prompt": "#4d908e",
              "swapping_out": "#f8961e", "swapping_in": "#f3722c", "paused": "#f94144",
              "host": "#9b5de5", "idle": "#adb5bd", "started": "#577590"}
    width, left, lane = 900, 70, 26
    parts = [f'<svg viewBox="0 0 {width} {45 + lane * 4}" role="img">',
             '<text x="0" y="17" class="chart-title">Per-sequence residency and scheduling timeline</text>']
    for idx, sample in enumerate(data):
        x = left + (sample["t"] - t0) / span * (width - left - 10)
        x2 = left + ((data[idx + 1]["t"] if idx + 1 < len(data) else t1) - t0) / span * (width - left - 10)
        for slot in sample["slots"]:
            sid = int(slot["id"])
            state = slot.get("state", "idle")
            swap = slot.get("kv_swap", {}).get("state")
            if state == "idle" and swap in ("host", "swapping_out", "swapping_in"):
                state = swap
            color = colors.get(state, "#adb5bd")
            parts.append(f'<rect x="{x:.1f}" y="{30 + sid * lane}" width="{max(1, x2 - x):.1f}" height="18" fill="{color}"/>')
    for sid in range(4):
        parts.append(f'<text x="0" y="{44 + sid * lane}">slot {sid}</text>')
    parts.append("</svg>")
    return "".join(parts)


def write_report(data: dict, output: Path):
    summaries = data.get("summaries", {})
    rows = []
    for config, workloads in summaries.items():
        for workload, value in workloads.items():
            rows.append((config, workload, value))

    reference = data.get("tasks", {}).get("admission_queue", [])
    refs = {task["task"]: task for task in reference if task_completed(task) and
            (task["workload"] == "continuous" or task.get("turn") == 0)}
    swap_tasks = data.get("tasks", {}).get("unified_swap", [])
    comparisons = []
    for task in swap_tasks:
        ref = refs.get(task["task"])
        if ref and task_completed(task):
            comparisons.append(task["tokens"] == ref["tokens"])
    identical = sum(comparisons)
    cross_schedule_text = (f"{identical}/{len(comparisons)} cross-schedule continuations token-identical"
                           if comparisons else "cross-schedule diagnostic not measured")
    correctness = data.get("correctness", {})
    correctness_values = correctness.get("values", {})
    deterministic_identity = (correctness.get("returncode") == 0 and
                              correctness_values.get("byte_identical") is True and
                              correctness_values.get("continuation_identical") is True)
    identity_text = (f"{correctness_values.get('state_bytes', 0):,} state bytes and continuation token "
                     f"{correctness_values.get('continuation_token')} identical"
                     if deterministic_identity else "not verified")

    table = []
    for config, workload, item in rows:
        table.append("<tr>" + "".join([
            f"<td>{html.escape(config)}</td><td>{html.escape(workload)}</td>",
            f"<td>{item['successful']}/{item['tasks']}</td><td>{item['tokens_per_min']:.1f}</td>",
            f"<td>{item['ttft_p50_ms']:.1f}</td><td>{item['latency_p95_ms']:.1f}</td>",
            f"<td>{item['queue_mean_ms']:.1f}</td><td>{item['decode_tokens_s']:.2f}</td>",
            f"<td>{item['swap_stall_ms']:.1f}</td>",
            f"<td>{item['d2h_gbps']:.2f}/{item['h2d_gbps']:.2f}</td>",
            f"<td>{item['host_peak_bytes']/2**20:.1f}</td><td>{item['process_private_peak']/2**30:.2f}</td>",
            f"<td>{item['gpu_peak_mib']:.0f}</td><td>{item['tracked_total_peak']/2**30:.2f}</td>",
        ]) + "</tr>")

    success_chart = svg_bars([(f"{c}/{w}", v["successful"] / max(1, v["tasks"]) * 100)
                              for c, w, v in rows], "Successful requests", "%")
    goodput_chart = svg_bars([(f"{c}/{w}", v["tokens_per_min"]) for c, w, v in rows],
                             "Completed useful-token goodput", "tok/min")
    swap_summary = summaries.get("unified_swap", {})
    baseline_summary = summaries.get("unified_no_swap", {})
    swap_all_tasks = data.get("tasks", {}).get("unified_swap", [])
    no_500 = bool(swap_all_tasks) and all(
        task.get("status") != 500 and not re.search(r'"code"\s*:\s*500', task.get("error") or "")
        for task in swap_all_tasks)
    all_completed = bool(swap_summary) and all(v.get("failed", 1) == 0 for v in swap_summary.values())
    goodput_better = any(swap_summary.get(w, {}).get("tokens_per_min", 0) >
                         baseline_summary.get(w, {}).get("tokens_per_min", 0)
                         for w in swap_summary)
    resumed = any(v.get("pages_in", 0) > 0 for v in swap_summary.values())
    overlap = sum(v.get("overlap_seconds", 0) for v in swap_summary.values())
    transfer = sum(v.get("d2h_seconds", 0) + v.get("h2d_seconds", 0) for v in swap_summary.values())
    overlap_material = overlap > 0.1 * transfer if transfer else False
    continuous = [task for task in swap_all_tasks if task.get("workload") == "continuous"]
    short = next((task for task in continuous if task.get("slot") == data["parameters"]["parallel"] - 1), None)
    older = [task for task in continuous if task.get("slot") != data["parameters"]["parallel"] - 1]
    short_overtook = bool(short and older and task_completed(short) and
                          short["ended"] < max(task["ended"] for task in older))
    checks = [
        (no_500, "No capacity HTTP/SSE server errors in the swap configuration"),
        (all_completed, "All swap requests produced their requested useful tokens without starvation"),
        (deterministic_identity, f"Same-context deterministic reference: {identity_text}"),
        (resumed, "At least one swapped sequence completed H2D restoration"),
        (short_overtook, "The newer short request progressed before all older long requests completed"),
        (goodput_better, "Swap goodput exceeded the unified no-swap failure case"),
        (overlap_material, f"Measured copy/decode overlap was material ({overlap:.3f}s observed)"),
    ]
    checks_html = "".join(f'<li class="{"pass" if ok else "fail"}">{"PASS" if ok else "CHECK"}: {html.escape(text)}</li>'
                          for ok, text in checks)
    timeline = timeline_svg(data.get("samples", {}).get("unified_swap", []), "continuous")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    generated = time.strftime("%Y-%m-%d %H:%M:%S %Z")

    report = f"""<!doctype html><html><head><meta charset="utf-8"><title>llama-server KV swap report</title>
<style>body{{font:14px system-ui;margin:0;background:#f5f7fb;color:#18212f}}main{{max-width:1200px;margin:auto;padding:28px}}
h1,h2{{color:#102a43}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}}
.card{{background:white;border:1px solid #d9e2ec;border-radius:10px;padding:18px;box-shadow:0 2px 7px #0001}}
table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:8px;border:1px solid #d9e2ec;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
svg{{width:100%;height:auto}}svg rect{{fill:#277da1}}svg text{{font:12px system-ui;fill:#243b53}}.chart-title{{font-weight:700;font-size:14px}}
.pass{{color:#137333}}.fail{{color:#a61b1b}}code{{background:#e9eef5;padding:2px 5px}}details{{margin-top:20px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style></head>
<body><main><h1><code>--kv-swap</code> MVP benchmark</h1><p>Generated {html.escape(generated)}. Context={data['parameters']['ctx_size']:,}, slots={data['parameters']['parallel']}, pinned-host budget={data['parameters']['swap_mib']} MiB.</p>
<section class="card"><h2>Outcome</h2><ul>{checks_html}</ul><p>The CUDA MVP swaps only growing attention K/V. Qwen3.6 fixed GDN recurrent state remains device-resident. Eviction uses 128-cell logical pages, grouped behind one completion event; restoration gates decode until every historical page is back on device.</p><p>The identity acceptance test restores a byte-serialized no-swap reference in the same context, forces fragmented cell placement through D2H/H2D, then compares both serialized state and the next greedy token. The separate workload hash comparison is only diagnostic because different concurrency schedules change batching: {html.escape(cross_schedule_text)}.</p></section>
<h2>Comparative results</h2><div class="card"><table><thead><tr><th>Configuration</th><th>Workload</th><th>Success</th><th>Tokens/min</th><th>TTFT p50 ms</th><th>Latency p95 ms</th><th>Queue ms</th><th>Decode tok/s</th><th>Swap stall ms</th><th>D2H/H2D GB/s</th><th>Pinned peak MiB</th><th>RAM peak GiB</th><th>GPU peak MiB</th><th>Tracked total GiB</th></tr></thead><tbody>{''.join(table)}</tbody></table></div>
<div class="cards"><div class="card">{success_chart}</div><div class="card">{goodput_chart}</div></div>
<h2>Scheduling behavior</h2><div class="card">{timeline}<p>Orange/red segments are asynchronous eviction/restore; purple is host-resident; red slot state is a paused open response. A paused stream emits no model tokens until all H2D groups complete. Quantum fairness selects a runnable victim after 32 useful tokens, with last-use time breaking ties.</p></div>
<h2>Memory and performance tradeoffs</h2><div class="card"><p>Pinned host memory extends logical context without enlarging the device KV allocation. The cost is D2H/H2D traffic and per-resume stall; full H2D restore is required because an attention query can address any historical cell. Copy/decode overlap is reported as observed decode wall time while page-group transfers were pending. On systems where this is small relative to transfer time, copy overlap does not materially hide latency.</p></div>
<details><summary>Embedded raw results</summary><pre id="raw"></pre></details><script id="data" type="application/json">{payload}</script><script>document.getElementById('raw').textContent=JSON.stringify(JSON.parse(document.getElementById('data').textContent),null,2);</script>
</main></body></html>"""
    (output / "report.html").write_text(report, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("kv-swap-results"))
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--workloads", default="agent,continuous")
    parser.add_argument("--ctx-size", type=int, default=131072)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--prompt-tokens", type=int, default=33000)
    parser.add_argument("--continuous-predict", type=int, default=4096)
    parser.add_argument("--agent-turns", type=int, default=4)
    parser.add_argument("--agent-predict", type=int, default=512)
    parser.add_argument("--tool-pause", type=float, default=1.0)
    parser.add_argument("--short-delay", type=float, default=2.0)
    parser.add_argument("--swap-mib", type=int, default=8192)
    parser.add_argument("--admission-limit", type=int, default=3)
    parser.add_argument("--device", default="CUDA0")
    parser.add_argument("--gpu-layers", default="all")
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--extra-server-arg", action="append", default=[])
    parser.add_argument("--correctness-test", type=Path)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--quick", action="store_true", help="5K-cell scaled acceptance run")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.ctx_size = 5120
        args.batch = args.ubatch = 512
        args.prompt_tokens = 1400
        args.continuous_predict = 384
        args.agent_turns = 2
        args.agent_predict = 192
        args.tool_pause = 0.2
        args.short_delay = 0.5
        args.swap_mib = 1024
        args.admission_limit = 2
    args.server = args.server.resolve()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "results.json"
    if args.report_only:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        write_report(data, args.output)
        return 0

    configs = [name for name in args.configs.split(",") if name in CONFIGS]
    workloads = [name for name in args.workloads.split(",") if name in ("agent", "continuous")]
    data = {"parameters": vars(args).copy(), "commands": {}, "tasks": {}, "samples": {}, "summaries": {}}
    data["parameters"] = {key: str(value) if isinstance(value, Path) else value for key, value in data["parameters"].items()}
    data["correctness"] = ({"available": False, "error": "skipped by request"}
                           if args.skip_correctness else run_correctness(args))

    for index, config in enumerate(configs):
        process = log = monitor = None
        try:
            print(f"kv-swap bench: starting {config}", flush=True)
            process, log, base, cmd = start_server(args, config, args.port + index, args.output)
            data["commands"][config] = cmd
            monitor = Monitor(base, process, args.sample_interval)
            monitor.start()
            tasks = []
            data["summaries"][config] = {}
            for workload in workloads:
                print(f"kv-swap bench: {config}/{workload}", flush=True)
                erase_slots(base, args.parallel)
                monitor.phase = workload
                current = run_agent(args, base, config) if workload == "agent" else run_continuous(args, base, config)
                tasks.extend(current)
                time.sleep(max(0.5, args.sample_interval * 2))
                data["summaries"][config][workload] = summarize_task_group(current, monitor.samples, workload)
                summary = data["summaries"][config][workload]
                print(f"kv-swap bench: {config}/{workload}: "
                      f"{summary['successful']}/{summary['tasks']} complete", flush=True)
            data["tasks"][config] = tasks
            monitor.stop()
            data["samples"][config] = monitor.samples
        except Exception as exc:
            data.setdefault("errors", {})[config] = str(exc)
            if monitor:
                monitor.stop()
                data["samples"][config] = monitor.samples
        finally:
            if process and log:
                stop_server(process, log)
            result_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            write_report(data, args.output)
            print(f"kv-swap bench: finished {config}", flush=True)
    return 0 if not data.get("errors") else 1


if __name__ == "__main__":
    sys.exit(main())
