import math
import os
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
import base64
import time

import pytest
import requests

from utils import ServerPreset, download_file


server = ServerPreset.tinyllama2()

MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"
MODEL_TINY_FILE_URL = "https://huggingface.co/ggml-org/test-model-stories260K/resolve/main/stories260K-f32.gguf"
KV_PRESSURE_API_KEY = "kv-pressure-test-key"
KV_PRESSURE_AUTH = {"Authorization": f"Bearer {KV_PRESSURE_API_KEY}"}
TELEMETRY_API_KEY = "telemetry-test-key"

KV_PRESSURE_EVENT_KINDS = {
    "utilization_sample",
    "decode_wait_started",
    "decode_retry",
    "decode_wait_finished",
    "idle_slot_evicted",
    "context_shift",
}

CONTEXT_SHIFT_PROMPT = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
""".strip()

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_metrics = True
    configure_telemetry_server()


def configure_telemetry_server():
    server.server_props = True
    server.api_key = TELEMETRY_API_KEY
    test_server = server
    request = test_server.make_request

    def authenticated_request(method, path, data=None, **kwargs):
        headers = {"Authorization": f"Bearer {test_server.api_key}"}
        headers.update(kwargs.pop("headers", None) or {})
        return request(method, path, data=data, headers=headers, **kwargs)

    server.make_request = authenticated_request


def configure_embedded_mtp_fixture(test_server):
    model_path = os.environ.get("LLAMA_TEST_MTP_MODEL_PATH")
    if not model_path:
        pytest.skip("requires LLAMA_TEST_MTP_MODEL_PATH for an embedded-MTP fixture")

    test_server.model_file = model_path
    test_server.model_hf_repo = None
    test_server.model_hf_file = None
    test_server.spec_type = "draft-mtp"
    test_server.spec_draft_n_max = 3


def configure_ngram_simple_speculation(test_server):
    test_server.spec_type = "ngram-simple"
    test_server.spec_ngram_simple_size_n = 2
    test_server.spec_ngram_simple_size_m = 3
    test_server.spec_ngram_simple_min_hits = 1


def apply_telemetry_control(**features):
    response = server.make_request(
        "POST",
        "/props",
        data={"telemetry_control": features},
    )
    assert response.status_code == 200
    assert response.body["telemetry_control"]["effective"] == {
        "moe_routing": features.get("moe_routing", False),
        "output_token_detail": features.get("output_token_detail", False),
        "token_candidates": features.get("token_candidates", False),
        "prompt_perplexity": features.get("prompt_perplexity", False),
        "request_content": features.get("request_content", False),
        "kv_pressure_detail": features.get("kv_pressure_detail", False),
        "native_gpu_gpm": features.get("native_gpu_gpm", False),
    }
    return response.body["telemetry_control"]


def completed_event(trace_id):
    response = server.make_request("GET", "/telemetry/v1/events?cursor=0&limit=100")
    assert response.status_code == 200
    assert response.body["schema_version"] == 1
    assert response.body["cursor"] > 0
    assert response.body["next_sequence"] > response.body["cursor"]
    matches = [
        event for event in response.body["events"]
        if event["trace_id"] == trace_id and event["event"] == "request_completed"
    ]
    assert len(matches) == 1
    return matches[0]


def trace_events(trace_id):
    response = server.make_request("GET", "/telemetry/v1/events?cursor=0&limit=100")
    assert response.status_code == 200
    return [event for event in response.body["events"] if event["trace_id"] == trace_id]


def test_event_ring_reuses_serialized_payload_bytes_for_retention_and_response():
    api_key = "serialized-event-test-key"
    auth = {"Authorization": f"Bearer {api_key}"}
    server.server_props = True
    server.api_key = api_key
    server.extra_env = {"LLAMA_TELEMETRY_EVENT_BUFFER_MIB": "1"}
    server.start()

    control = server.make_request(
        "POST",
        "/props",
        data={"telemetry_control": {"request_content": True}},
        headers=auth,
    )
    assert control.status_code == 200

    for ordinal in range(2):
        completion = server.make_request(
            "POST",
            "/completion",
            data={
                "prompt": "serialized event payload",
                "n_predict": 1,
                "unused_payload": str(ordinal) + "x" * 600_000,
            },
            headers=auth,
        )
        assert completion.status_code == 200

    capabilities = server.make_request("GET", "/telemetry/v1/capabilities", headers=auth)
    assert capabilities.status_code == 200
    byte_cap = capabilities.body["content_policy"]["serialized_event_capacity_bytes"]

    response = requests.get(
        f"http://{server.server_host}:{server.server_port}/telemetry/v1/events?cursor=0&limit=100",
        headers=auth,
        timeout=60,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["events"]

    serialized_events = [
        json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        for event in body["events"]
    ]
    assert body["retained_serialized_bytes"] == sum(map(len, serialized_events))
    assert body["retained_serialized_bytes"] <= byte_cap
    assert body["dropped_events"] > 0
    assert body["gap"] is True
    assert body["gap_ranges"] == [{
        "first_sequence": 1,
        "last_sequence": body["oldest_sequence"] - 1,
    }]
    assert b'"events":[' + b",".join(serialized_events) + b"]" in response.content

    zero_width = server.make_request(
        "GET",
        f"/telemetry/v1/events?cursor={body['oldest_sequence'] - 1}&limit=100",
        headers=auth,
    )
    assert zero_width.status_code == 200
    assert zero_width.body["gap"] is False
    assert zero_width.body["gap_ranges"] == []


def test_event_ring_future_cursor_resets_to_high_water_mark():
    server.start()

    future = server.make_request("GET", "/telemetry/v1/events?cursor=18446744073709551615")
    assert future.status_code == 200
    assert future.body["events"] == []
    assert future.body["gap"] is True
    assert future.body["gap_ranges"] == []
    assert future.body["cursor"] == future.body["next_sequence"] - 1

    response = server.make_request(
        "POST",
        "/completion",
        data={"prompt": "Recover after a future event cursor", "n_predict": 1, "ignore_eos": True},
    )
    assert response.status_code == 200

    resumed = server.make_request(
        "GET", f"/telemetry/v1/events?cursor={future.body['cursor']}&limit=100"
    )
    assert resumed.status_code == 200
    assert resumed.body["events"]
    assert all(event["sequence"] > future.body["cursor"] for event in resumed.body["events"])


def test_event_ring_reports_singleton_trailing_gap_for_dropped_completion():
    api_key = "dropped-final-event-test-key"
    auth = {"Authorization": f"Bearer {api_key}"}
    server.server_props = True
    server.api_key = api_key
    server.extra_env = {"LLAMA_TELEMETRY_EVENT_BUFFER_MIB": "1"}
    server.start()

    control = server.make_request(
        "POST",
        "/props",
        data={"telemetry_control": {"request_content": True}},
        headers=auth,
    )
    assert control.status_code == 200

    completion = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "A small prompt with an oversized retained request payload",
            "n_predict": 1,
            "ignore_eos": True,
            "unused_payload": "x" * (1024 * 1024),
        },
        headers=auth,
    )
    assert completion.status_code == 200

    response = server.make_request("GET", "/telemetry/v1/events?cursor=0&limit=100", headers=auth)
    assert response.status_code == 200
    body = response.body
    final_sequence = body["next_sequence"] - 1
    assert body["dropped_events"] == 1
    assert body["last_dropped_sequence"] == final_sequence
    assert body["events"][-1]["sequence"] == final_sequence - 1
    assert body["gap"] is True
    assert body["gap_ranges"] == [{
        "first_sequence": final_sequence,
        "last_sequence": final_sequence,
    }]


def kv_pressure_batch(cursor=0, limit=4096, trace_id=None, headers=None):
    path = f"/telemetry/v1/kv-pressure?cursor={cursor}&limit={limit}"
    if trace_id is not None:
        path += f"&trace_id={quote(trace_id, safe='')}"
    response = server.make_request(
        "GET", path, headers=KV_PRESSURE_AUTH if headers is None else headers
    )
    assert response.status_code == 200
    assert response.body["schema_version"] == 1
    return response.body


def kv_pressure_request(method, path, data=None, **kwargs):
    return server.make_request(method, path, data=data, headers=KV_PRESSURE_AUTH, **kwargs)


def start_kv_pressure_server():
    server.server_props = True
    server.api_key = KV_PRESSURE_API_KEY
    server.start()

    disabled = kv_pressure_batch()
    assert disabled["events"] == []
    assert disabled["oldest_sequence"] == disabled["next_sequence"] == 1
    assert disabled["dropped_events"] == 0
    assert disabled["last_dropped_sequence"] == 0
    assert disabled["retained_serialized_bytes"] == 0

    enabled = kv_pressure_request(
        "POST",
        "/props",
        {"telemetry_control": {"kv_pressure_detail": True}},
    )
    assert enabled.status_code == 200
    assert enabled.body["telemetry_control"]["effective"]["kv_pressure_detail"] is True
    assert enabled.body["telemetry_control"]["effective_from"] == "next_microbatch"


def one_token_without_special_tokens():
    response = kv_pressure_request(
        "POST",
        "/tokenize",
        data={"content": " hello", "add_special": False},
    )
    assert response.status_code == 200
    assert response.body["tokens"]
    return response.body["tokens"][0]


def test_kv_pressure_capability_and_route_exist():
    start_kv_pressure_server()

    capabilities = kv_pressure_request("GET", "/telemetry/v1/capabilities")
    assert capabilities.status_code == 200
    capability = capabilities.body["capabilities"]["kv_pressure"]
    assert capability["state"] == "available"
    assert capability["endpoint"] == "/telemetry/v1/kv-pressure"
    assert capability["schema_version"] == 1
    assert capability["reason"]
    assert capability["owner"] == "llama.cpp/llama-server"
    assert 10 <= capability["sampling_interval_ms"] <= 5000
    assert capability["event_capacity"] > 0
    assert 1024 * 1024 <= capability["serialized_event_capacity_bytes"] <= 1024 * 1024 * 1024
    assert set(capability["event_kinds"]) == KV_PRESSURE_EVENT_KINDS
    assert capability["decode_wait_semantics"] == "llama_decode_returned_1_no_kv_slot_available"

    pressure = kv_pressure_request("GET", "/telemetry/v1/kv-pressure?cursor=0&limit=1")
    assert pressure.status_code == 200
    assert pressure.body["schema_version"] == 1
    assert pressure.body["state"] == "available"
    assert pressure.body["reason"]
    assert pressure.body["server_instance_id"] == capabilities.body["server_instance_id"]
    assert pressure.body["trace_filter"] is None
    assert pressure.body["request_start_monotonic_us"] is None
    assert pressure.body["request_end_monotonic_us"] is None
    assert len(pressure.body["events"]) <= 1
    assert pressure.body["oldest_sequence"] >= 1
    assert pressure.body["next_sequence"] >= pressure.body["oldest_sequence"]
    assert pressure.body["cursor"] <= pressure.body["next_sequence"]
    assert pressure.body["gap"] is False
    assert pressure.body["dropped_events"] == 0
    assert pressure.body["last_dropped_sequence"] == 0
    assert pressure.body["retained_serialized_bytes"] <= capability["serialized_event_capacity_bytes"]


def test_kv_pressure_capabilities_are_safe_during_decode():
    server.n_ctx = 512
    server.n_batch = 32
    server.n_ubatch = 32
    server.n_threads = 1
    server.n_slots = 1
    server.enable_ctx_shift = True
    start_kv_pressure_server()

    token = one_token_without_special_tokens()
    with ThreadPoolExecutor(max_workers=1) as executor:
        completion = executor.submit(
            kv_pressure_request,
            "POST",
            "/completion",
            {
                "prompt": [token] * 480,
                "n_predict": 256,
                "ignore_eos": True,
                "temperature": 0,
                "cache_prompt": True,
            },
        )
        capability_reads = 0
        while not completion.done():
            response = kv_pressure_request("GET", "/telemetry/v1/capabilities")
            assert response.status_code == 200
            assert response.body["capabilities"]["kv_pressure"]["state"] == "available"
            capability_reads += 1

        result = completion.result()

    assert result.status_code == 200
    assert capability_reads > 0


@pytest.mark.parametrize(
    "query",
    [
        "cursor=-1",
        "cursor=not-a-number",
        "limit=0",
        "limit=-1",
        "limit=4097",
        "limit=not-a-number",
        f"trace_id={'x' * 257}",
    ],
)
def test_kv_pressure_route_rejects_invalid_cursor_limit_and_trace(query):
    start_kv_pressure_server()

    response = kv_pressure_request("GET", f"/telemetry/v1/kv-pressure?{query}")

    assert response.status_code == 400


def test_kv_pressure_route_pages_with_monotonic_bounded_metadata():
    start_kv_pressure_server()

    response = kv_pressure_request(
        "POST",
        "/completion",
        data={"prompt": "A bounded KV-pressure page", "n_predict": 2, "ignore_eos": True},
    )
    assert response.status_code == 200
    first = kv_pressure_batch(limit=1)
    assert len(first["events"]) == 1
    first_event = first["events"][0]
    assert first_event["sequence"] == first["cursor"]
    assert first_event["schema_version"] == 1
    assert first_event["server_instance_id"] == first["server_instance_id"]
    assert first_event["kind"] in KV_PRESSURE_EVENT_KINDS
    assert first_event["timestamp_unix_ms"] > 0
    assert first_event["monotonic_us"] > 0

    response = kv_pressure_request(
        "POST",
        "/completion",
        data={"prompt": "A second bounded KV-pressure page", "n_predict": 2, "ignore_eos": True},
    )
    assert response.status_code == 200
    second = kv_pressure_batch(cursor=first["cursor"], limit=2)
    sequences = [event["sequence"] for event in second["events"]]
    assert 1 <= len(sequences) <= 2
    assert sequences == sorted(sequences)
    assert all(sequence > first["cursor"] for sequence in sequences)
    assert second["cursor"] == sequences[-1]
    assert second["oldest_sequence"] <= first["cursor"] + 1
    assert second["next_sequence"] > second["cursor"]
    assert second["gap"] is False
    assert second["dropped_events"] == 0
    assert second["last_dropped_sequence"] == 0
    assert second["retained_serialized_bytes"] >= first["retained_serialized_bytes"]


def test_kv_pressure_future_cursor_resets_to_high_water_mark():
    start_kv_pressure_server()

    future = kv_pressure_batch(cursor=2**64 - 1)
    assert future["state"] == "partial"
    assert "cursor" in future["reason"].lower()
    assert future["gap"] is True
    assert future["cursor"] == future["next_sequence"] - 1

    response = kv_pressure_request(
        "POST",
        "/completion",
        data={"prompt": "Recover after a future cursor", "n_predict": 2, "ignore_eos": True},
    )
    assert response.status_code == 200
    resumed = kv_pressure_batch(cursor=future["cursor"])
    assert resumed["events"]
    assert all(event["sequence"] > future["cursor"] for event in resumed["events"])


def test_kv_pressure_reports_exact_global_primary_occupancy_after_decode():
    start_kv_pressure_server()

    response = kv_pressure_request(
        "POST",
        "/completion",
        data={"prompt": "Measure occupied KV entries", "n_predict": 4, "ignore_eos": True},
    )
    assert response.status_code == 200
    batch = kv_pressure_batch()
    samples = [event for event in batch["events"] if event["kind"] == "utilization_sample"]
    assert samples
    sample = samples[-1]
    shallow_kv = kv_pressure_request("GET", "/telemetry/v1/kv")
    assert shallow_kv.status_code == 200
    assert shallow_kv.body["components"] == []
    assert shallow_kv.body["physical_prefix_sharing"]["state"] == "not_collected"
    assert shallow_kv.body["duplicate_prefix_opportunities"]["state"] == "not_collected"

    kv = kv_pressure_request("GET", "/telemetry/v1/kv?detail=deep")
    assert kv.status_code == 200
    primary = next(component for component in kv.body["components"] if component["logical_primary"])

    assert sample["utilization_state"] == "available"
    assert sample["utilization_reason"]
    assert sample["component"]
    assert sample["memory_kind"]
    assert sample["entry_semantics"]
    assert sample["capacity_entries"] > 0
    assert 0 < sample["used_entries"] <= sample["capacity_entries"]
    assert sample["free_entries"] == sample["capacity_entries"] - sample["used_entries"]
    assert sample["utilization"] == pytest.approx(
        sample["used_entries"] / sample["capacity_entries"], rel=1e-12
    )
    assert sample["component"] == primary["name"]
    assert sample["memory_kind"] == primary["kind"]
    assert sample["entry_semantics"] == primary["entry_semantics"]
    assert sample["capacity_entries"] == primary["capacity_entries"]
    assert sample["used_entries"] == primary["used_entries"]
    assert sample["free_entries"] == primary["free_entries"]
    assert sample.get("trace_id") is None
    assert sample.get("task_id") is None
    assert sample.get("slot_id") is None
    assert sample.get("role") is None
    assert sample.get("episode_id") is None


def test_kv_pressure_trace_filter_has_exact_identity_and_request_bounds():
    start_kv_pressure_server()

    response = kv_pressure_request(
        "POST",
        "/completion",
        data={"prompt": "Trace-bound KV telemetry", "n_predict": 4, "ignore_eos": True},
    )
    assert response.status_code == 200
    trace_id = response.body["trace_id"]
    batch = kv_pressure_batch(trace_id=trace_id)

    assert batch["trace_filter"] == trace_id
    assert batch["request_start_monotonic_us"] > 0
    assert batch["request_end_monotonic_us"] >= batch["request_start_monotonic_us"]
    assert batch["events"]
    sequences = [event["sequence"] for event in batch["events"]]
    assert sequences == sorted(sequences)
    for event in batch["events"]:
        assert batch["request_start_monotonic_us"] <= event["monotonic_us"] <= batch["request_end_monotonic_us"]
        if event["kind"] == "utilization_sample":
            assert event.get("trace_id") is None
        else:
            assert event["trace_id"] == trace_id

    unknown = kv_pressure_batch(trace_id="unknown-trace-id")
    assert unknown["trace_filter"] == "unknown-trace-id"
    assert unknown["request_start_monotonic_us"] is None
    assert unknown["request_end_monotonic_us"] is None
    assert unknown["events"] == []


def test_kv_pressure_context_shift_preserves_trace_and_distinct_churn_counts():
    server.n_ctx = 512
    server.n_slots = 2
    server.enable_ctx_shift = True
    start_kv_pressure_server()

    response = kv_pressure_request(
        "POST",
        "/completion",
        data={
            "prompt": CONTEXT_SHIFT_PROMPT,
            "n_predict": 96,
            "ignore_eos": True,
        },
    )
    assert response.status_code == 200
    assert response.body["truncated"] is True
    trace_id = response.body["trace_id"]
    batch = kv_pressure_batch(trace_id=trace_id)
    shifts = [event for event in batch["events"] if event["kind"] == "context_shift"]
    assert shifts

    for shift in shifts:
        assert shift["trace_id"] == trace_id
        assert shift["task_id"] >= 0
        assert 0 <= shift["slot_id"] < server.n_slots
        assert shift["role"] == "target"
        assert shift.get("episode_id") is None
        assert shift["discarded_tokens"] > 0
        assert shift["position_delta"] == -shift["discarded_tokens"] < 0
        assert shift["shifted_entries"] >= 0


def test_kv_pressure_retained_trace_events_are_partial_when_window_expires(monkeypatch):
    monkeypatch.setenv("LLAMA_TELEMETRY_KV_PRESSURE_REQUEST_WINDOW_LIMIT", "1")
    server.n_ctx = 512
    server.n_slots = 2
    server.enable_ctx_shift = True
    start_kv_pressure_server()

    first = kv_pressure_request(
        "POST",
        "/completion",
        data={
            "prompt": CONTEXT_SHIFT_PROMPT,
            "n_predict": 96,
            "ignore_eos": True,
        },
    )
    assert first.status_code == 200
    second = kv_pressure_request(
        "POST",
        "/completion",
        data={"prompt": "Expire the prior request window", "n_predict": 2, "ignore_eos": True},
    )
    assert second.status_code == 200

    batch = kv_pressure_batch(trace_id=first.body["trace_id"])
    assert batch["state"] == "partial"
    assert "window" in batch["reason"].lower()
    assert batch["request_start_monotonic_us"] is None
    assert batch["request_end_monotonic_us"] is None
    assert batch["events"]
    assert all(event["kind"] != "utilization_sample" for event in batch["events"])
    assert any(event["kind"] == "context_shift" for event in batch["events"])

    consumed = kv_pressure_batch(cursor=batch["cursor"], trace_id=first.body["trace_id"])
    assert consumed["state"] == "partial"
    assert "window" in consumed["reason"].lower()
    assert consumed["request_start_monotonic_us"] is None
    assert consumed["request_end_monotonic_us"] is None
    assert consumed["events"] == []


def test_kv_pressure_decode_wait_is_a_ret1_ordered_terminal_episode():
    server.n_ctx = 256
    server.n_batch = 32
    server.n_ubatch = 32
    server.n_slots = 2
    server.kv_unified = True
    server.no_cache_idle_slots = True
    start_kv_pressure_server()

    token = one_token_without_special_tokens()
    response = kv_pressure_request(
        "POST",
        "/completion",
        data={
            "prompt": [[token], [token]],
            "n_predict": 129,
            "ignore_eos": True,
            "temperature": 0,
            "cache_prompt": True,
        },
    )
    assert response.status_code == 500
    assert "Context size has been exceeded" in response.body["error"]["message"]
    batch = kv_pressure_batch()
    starts = [event for event in batch["events"] if event["kind"] == "decode_wait_started"]
    assert starts

    chain = None
    for started in starts:
        candidates = [
            event
            for event in batch["events"]
            if event.get("episode_id") == started["episode_id"]
            and event.get("trace_id") == started["trace_id"]
            and event.get("task_id") == started["task_id"]
            and event.get("slot_id") == started["slot_id"]
        ]
        if any(event["kind"] == "decode_retry" for event in candidates) and any(
            event["kind"] == "decode_wait_finished" for event in candidates
        ):
            chain = candidates
            break
    assert chain is not None

    started = next(event for event in chain if event["kind"] == "decode_wait_started")
    retries = [event for event in chain if event["kind"] == "decode_retry"]
    finished = next(event for event in chain if event["kind"] == "decode_wait_finished")
    assert started["role"] == "target"
    assert started["attempted_batch_size"] == 32
    assert [event["sequence"] for event in chain] == sorted(event["sequence"] for event in chain)
    assert started["sequence"] < retries[0]["sequence"] < finished["sequence"]
    assert started["monotonic_us"] <= retries[0]["monotonic_us"] <= finished["monotonic_us"]
    assert [
        (
            event["retry_count"],
            event["action"],
            event["attempted_batch_size"],
            event["next_batch_size"],
        )
        for event in retries
    ] == [
        (1, "batch_halved", 32, 16),
        (2, "batch_halved", 16, 8),
        (3, "batch_halved", 8, 4),
        (4, "batch_halved", 4, 2),
        (5, "batch_halved", 2, 1),
        (6, "terminal", 1, 0),
    ]
    assert finished["outcome"] == "context_exhausted"
    assert finished["wait_duration_us"] > 0
    assert finished["wait_duration_us"] == finished["monotonic_us"] - started["monotonic_us"]


def test_kv_pressure_split_recovery_finishes_each_identity_when_its_subbatch_succeeds():
    server.n_ctx = 256
    server.n_batch = 32
    server.n_ubatch = 32
    server.n_slots = 2
    server.kv_unified = True
    server.no_cache_idle_slots = True
    server.enable_ctx_shift = False
    start_kv_pressure_server()

    token = one_token_without_special_tokens()
    response = kv_pressure_request(
        "POST",
        "/completion",
        data={
            "prompt": [[token], [token, token]],
            "n_predict": 128,
            "ignore_eos": True,
            "temperature": 0,
            "cache_prompt": True,
        },
    )
    assert response.status_code == 200
    assert isinstance(response.body, list)
    assert len(response.body) == 2
    trace_ids = {result["trace_id"] for result in response.body}
    assert len(trace_ids) == 2

    batch = kv_pressure_batch()
    starts = [
        event
        for event in batch["events"]
        if event["kind"] == "decode_wait_started" and event.get("trace_id") in trace_ids
    ]
    shared_episode = next(
        episode_id
        for episode_id in {event["episode_id"] for event in starts}
        if {event["trace_id"] for event in starts if event["episode_id"] == episode_id} == trace_ids
    )
    episode = [
        event for event in batch["events"] if event.get("episode_id") == shared_episode
    ]
    evictions = [
        event
        for event in episode
        if event["kind"] == "idle_slot_evicted" and event["cause"] == "decode_pressure_recovery"
    ]
    assert len(evictions) == 1
    eviction = evictions[0]
    finishes = [event for event in episode if event["kind"] == "decode_wait_finished"]
    assert len(finishes) == 2
    assert {event["trace_id"] for event in finishes} == trace_ids
    assert all(event["outcome"] == "resumed" for event in finishes)

    victim_finish = next(event for event in finishes if event["trace_id"] == eviction["victim_trace_id"])
    target_finish = next(event for event in finishes if event["trace_id"] == eviction["trace_id"])
    assert victim_finish["sequence"] < eviction["sequence"] < target_finish["sequence"]
    for trace_id in trace_ids:
        trace_starts = [event for event in starts if event["trace_id"] == trace_id]
        trace_finishes = [
            event
            for event in batch["events"]
            if event["kind"] == "decode_wait_finished" and event.get("trace_id") == trace_id
        ]
        assert len(trace_starts) == len(trace_finishes) == 1


def test_kv_pressure_identity_stays_waiting_across_multiple_retry_slices():
    server.n_ctx = 256
    server.n_batch = 32
    server.n_ubatch = 32
    server.n_slots = 2
    server.kv_unified = True
    server.no_cache_idle_slots = True
    server.enable_ctx_shift = False
    start_kv_pressure_server()

    token = one_token_without_special_tokens()
    response = kv_pressure_request(
        "POST",
        "/completion",
        data={
            "prompt": [[token], [token]],
            "n_predict": 129,
            "ignore_eos": True,
            "temperature": 0,
            "cache_prompt": True,
        },
    )
    assert response.status_code == 500
    assert "Context size has been exceeded" in response.body["error"]["message"]

    batch = kv_pressure_batch()
    starts = [
        event
        for event in batch["events"]
        if event["kind"] == "decode_wait_started"
    ]
    shared_episode = next(
        episode_id
        for episode_id in {event["episode_id"] for event in starts}
        if len(
            {
                event["trace_id"]
                for event in starts
                if event["episode_id"] == episode_id
            }
        )
        == 2
    )
    episode = [event for event in batch["events"] if event.get("episode_id") == shared_episode]
    trace_ids = {
        event["trace_id"]
        for event in starts
        if event["episode_id"] == shared_episode
    }
    assert len(trace_ids) == 2
    expected_retries = [
        (1, "batch_halved", 32, 16),
        (2, "batch_halved", 16, 8),
        (3, "batch_halved", 8, 4),
        (4, "batch_halved", 4, 2),
        (5, "batch_halved", 2, 1),
        (6, "terminal", 1, 0),
    ]
    for trace_id in trace_ids:
        retries = [
            event
            for event in episode
            if event["kind"] == "decode_retry" and event["trace_id"] == trace_id
        ]
        assert [
            (
                event["retry_count"],
                event["action"],
                event["attempted_batch_size"],
                event["next_batch_size"],
            )
            for event in retries
        ] == expected_retries
        finished = next(
            event for event in episode
            if event["kind"] == "decode_wait_finished"
            and event["trace_id"] == trace_id
        )
        assert finished["outcome"] == "context_exhausted"
        assert finished["sequence"] > retries[-1]["sequence"]


def test_kv_pressure_post_split_error_finishes_the_unresolved_identity():
    model_file = download_file(MODEL_TINY_FILE_URL)
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = model_file
    server.model_draft = model_file
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 1
    server.spec_draft_n_max = 4
    server.n_ctx = 256
    server.n_batch = 32
    server.n_ubatch = 32
    server.n_slots = 2
    server.kv_unified = True
    server.no_cache_idle_slots = True
    server.enable_ctx_shift = False
    start_kv_pressure_server()

    token = one_token_without_special_tokens()
    response = kv_pressure_request(
        "POST",
        "/completion",
        data={
            "prompt": [[token], [token, token]],
            "n_predict": 128,
            "ignore_eos": True,
            "temperature": 0,
            "cache_prompt": True,
        },
    )
    assert response.status_code == 500
    assert "speculative batch index" in response.body["error"]["message"]

    batch = kv_pressure_batch()
    starts = [event for event in batch["events"] if event["kind"] == "decode_wait_started"]
    shared_episode = next(
        episode_id
        for episode_id in {event["episode_id"] for event in starts}
        if len({event["trace_id"] for event in starts if event["episode_id"] == episode_id}) == 2
    )
    episode_starts = [event for event in starts if event["episode_id"] == shared_episode]
    finishes = [
        event
        for event in batch["events"]
        if event["kind"] == "decode_wait_finished" and event.get("episode_id") == shared_episode
    ]
    assert len(finishes) == len(episode_starts) == 2
    assert {event["trace_id"] for event in finishes} == {
        event["trace_id"] for event in episode_starts
    }
    assert {event["outcome"] for event in finishes} == {"resumed", "failed"}


def test_kv_pressure_emergency_eviction_separates_trigger_and_victim_identity():
    server.n_ctx = 256
    server.n_batch = 32
    server.n_ubatch = 32
    server.n_slots = 2
    server.kv_unified = True
    server.no_cache_idle_slots = True
    start_kv_pressure_server()

    token = one_token_without_special_tokens()
    victim = kv_pressure_request(
        "POST",
        "/completion",
        data={
            "prompt": [token] * 128,
            "n_predict": 1,
            "ignore_eos": True,
            "temperature": 0,
            "id_slot": 0,
            "cache_prompt": True,
        },
    )
    assert victim.status_code == 200
    before_target = kv_pressure_batch()
    target = kv_pressure_request(
        "POST",
        "/completion",
        data={
            "prompt": [token] * 129,
            "n_predict": 1,
            "ignore_eos": True,
            "temperature": 0,
            "id_slot": 1,
            "cache_prompt": True,
        },
    )
    assert target.status_code == 200
    batch = kv_pressure_batch(cursor=before_target["cursor"])
    evictions = [
        event
        for event in batch["events"]
        if event["kind"] == "idle_slot_evicted" and event["cause"] == "decode_pressure_recovery"
    ]
    assert len(evictions) == 1
    eviction = evictions[0]

    assert eviction["trace_id"] == target.body["trace_id"]
    assert eviction["task_id"] >= 0
    assert eviction["slot_id"] == 1
    assert eviction["role"] == "target"
    assert eviction["episode_id"]
    assert eviction["eviction_reason"]
    assert eviction["victim_slot_id"] == 0
    assert eviction["victim_slot_id"] != eviction["slot_id"]
    assert eviction["victim_trace_id"] == victim.body["trace_id"]
    assert eviction["victim_trace_id"] != eviction["trace_id"]
    assert eviction["victim_prompt_tokens"] == 128
    assert eviction["released_entries_state"] == "available"
    assert eviction["released_entries_reason"]
    assert eviction["released_entries"] > 0
    assert eviction["memberships_removed"] > 0

    samples_before = [
        event
        for event in batch["events"]
        if event["kind"] == "utilization_sample"
        and event["sequence"] < eviction["sequence"]
        and event["utilization_state"] == "available"
    ]
    samples_after = [
        event
        for event in batch["events"]
        if event["kind"] == "utilization_sample"
        and event["sequence"] > eviction["sequence"]
        and event["utilization_state"] == "available"
    ]
    assert samples_before and samples_after
    occupancy_before = samples_before[-1]
    occupancy_after = samples_after[0]
    assert occupancy_before["component"] == occupancy_after["component"]
    assert occupancy_before["used_entries"] - occupancy_after["used_entries"] == eviction["released_entries"]

    episode = [
        event
        for event in batch["events"]
        if event.get("episode_id") == eviction["episode_id"]
        and event.get("trace_id") == target.body["trace_id"]
    ]
    started = next(event for event in episode if event["kind"] == "decode_wait_started")
    retry = next(
        event
        for event in episode
        if event["kind"] == "decode_retry" and event["action"] == "emergency_idle_slot_eviction"
    )
    finished = next(event for event in episode if event["kind"] == "decode_wait_finished")
    assert started["sequence"] < retry["sequence"] < eviction["sequence"] < finished["sequence"]
    assert started["attempted_batch_size"] == 32
    assert retry["attempted_batch_size"] == retry["next_batch_size"] == 32
    assert finished["outcome"] == "resumed"
    assert finished["wait_duration_us"] == finished["monotonic_us"] - started["monotonic_us"]


def test_kv_pressure_proactive_idle_cache_policy_is_not_pressure_recovery(tmp_path):
    server.n_ctx = 256
    server.n_batch = 32
    server.n_ubatch = 32
    server.n_slots = 2
    server.kv_unified = True
    server.cache_ram = 100
    server.debug = True
    server.log_path = str(tmp_path / "server.log")
    start_kv_pressure_server()

    token = one_token_without_special_tokens()
    victim = kv_pressure_request(
        "POST",
        "/completion",
        data={
            "prompt": [token] * 128,
            "n_predict": 1,
            "ignore_eos": True,
            "temperature": 0,
            "id_slot": 0,
            "cache_prompt": True,
        },
    )
    assert victim.status_code == 200
    before_launch = kv_pressure_batch()
    target = kv_pressure_request(
        "POST",
        "/completion",
        data={
            "prompt": [token] * 129,
            "n_predict": 1,
            "ignore_eos": True,
            "temperature": 0,
            "id_slot": 1,
            "cache_prompt": True,
        },
    )
    assert target.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in (tmp_path / "server.log").read_text()

    batch = kv_pressure_batch(cursor=before_launch["cursor"])
    evictions = [event for event in batch["events"] if event["kind"] == "idle_slot_evicted"]
    assert len(evictions) == 1
    eviction = evictions[0]
    assert eviction["cause"] == "idle_cache_policy"
    assert eviction.get("trace_id") is None
    assert eviction.get("task_id") is None
    assert eviction.get("slot_id") is None
    assert eviction.get("episode_id") is None
    assert eviction["victim_slot_id"] == 0
    assert eviction["victim_trace_id"] == victim.body["trace_id"]


def terminal_event(trace_id, expected_event, expected_outcome):
    matches = [
        event for event in trace_events(trace_id)
        if event["event"] in ["request_completed", "request_ended"]
    ]
    assert len(matches) == 1
    assert matches[0]["event"] == expected_event
    assert matches[0]["outcome"] == expected_outcome
    return matches[0]


def assert_complete_lifecycle_clock(event, require_generation=True, require_handoff=True):
    clock = event["lifecycle_clock"]
    assert clock["schema_version"] == 1
    assert clock["clock_domain"] == "server_process_monotonic_microseconds"
    names = [
        "arrival_monotonic_us",
        "enqueue_monotonic_us",
        "slot_start_monotonic_us",
        "cache_start_monotonic_us",
        "cache_end_monotonic_us",
        "prefill_start_monotonic_us",
        "prefill_end_monotonic_us",
        "first_token_monotonic_us",
        "last_generation_work_monotonic_us",
        "finalization_start_monotonic_us",
        "response_handoff_monotonic_us",
        "slot_release_monotonic_us",
    ]
    assert set(names).issubset(clock)
    values = [clock[name] for name in names if clock[name] is not None]
    assert all(isinstance(value, int) and value > 0 for value in values)
    assert values == sorted(values)
    if require_generation:
        assert all(clock[name] is not None for name in names)
    assert clock["arrival_monotonic_us"] is not None
    assert clock["enqueue_monotonic_us"] is not None
    assert clock["slot_start_monotonic_us"] is not None
    assert clock["finalization_start_monotonic_us"] is not None
    if require_handoff:
        assert clock["response_handoff_monotonic_us"] is not None
        assert clock["response_handoff_monotonic_us"] >= clock["finalization_start_monotonic_us"]
        assert event["timings"]["e2e_ms"] == pytest.approx(
            (clock["response_handoff_monotonic_us"] - clock["arrival_monotonic_us"]) / 1000.0
        )
        assert event["timestamp_unix_ms"] == (
            event["timings"]["arrival_unix_ms"]
            + (clock["response_handoff_monotonic_us"] - clock["arrival_monotonic_us"]) // 1000
        )
    else:
        assert clock["response_handoff_monotonic_us"] is None
        assert event["timings"]["e2e_ms"] == 0
    assert clock["slot_release_monotonic_us"] is not None
    if require_handoff:
        assert clock["slot_release_monotonic_us"] >= clock["response_handoff_monotonic_us"]
    else:
        assert clock["slot_release_monotonic_us"] >= clock["finalization_start_monotonic_us"]
    assert event["slot_release_unix_ms"] >= event["timestamp_unix_ms"]
    assert event["slot_release_unix_ms"] == (
        event["timings"]["arrival_unix_ms"]
        + (clock["slot_release_monotonic_us"] - clock["arrival_monotonic_us"]) // 1000
    )


def assert_rich_diagnostics_finish_by_last_generation_work(event):
    detail = event["output_token_telemetry"]
    records = detail["records"]
    assert records
    assert detail["scored_tokens"] > 0
    latest_model_ready = max(record["model_ready_monotonic_us"] for record in records)
    last_generation_work = event["lifecycle_clock"]["last_generation_work_monotonic_us"]
    finalization_start = event["lifecycle_clock"]["finalization_start_monotonic_us"]
    assert last_generation_work > latest_model_ready
    assert finalization_start >= last_generation_work


@pytest.mark.parametrize("stream", [False, True])
def test_completion_e2e_ends_at_response_handoff(stream):
    server.start()
    data = {
        "prompt": "response handoff timing",
        "n_predict": 2,
        "ignore_eos": True,
        "stream": stream,
    }

    if stream:
        url = f"http://{server.server_host}:{server.server_port}/completion"
        with requests.post(
            url,
            json=data,
            headers={"Authorization": f"Bearer {server.api_key}"},
            stream=True,
            timeout=5.0,
        ) as response:
            assert response.status_code == 200
            trace_id = response.headers["X-Llama-Trace-Id"]
            final = None
            for line in response.iter_lines():
                if line.startswith(b"data: "):
                    chunk = json.loads(line[6:])
                    if chunk.get("stop") is True:
                        final = chunk
            assert final is not None
    else:
        response = server.make_request("POST", "/completion", data=data)
        assert response.status_code == 200
        trace_id = response.body["trace_id"]
        final = response.body

    event = completed_event(trace_id)
    assert_complete_lifecycle_clock(event)
    assert final["timings"]["e2e_ms"] == pytest.approx(event["timings"]["e2e_ms"])


def test_telemetry_lifecycle_cache_and_cursor():
    server.n_slots = 1
    server.start()

    capabilities = server.make_request("GET", "/telemetry/v1/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.body["schema_version"] == 1
    assert capabilities.body["capabilities"]["request_lifecycle"]["state"] == "available"
    assert capabilities.body["capabilities"]["moe_routing"]["state"] == "not_applicable"
    assert capabilities.body["capabilities"]["output_token_telemetry"]["state"] == "conditional"
    assert "POST /props telemetry_control.output_token_detail=true" in capabilities.body["capabilities"]["output_token_telemetry"]["enable_with"]

    invalid_cursor = server.make_request("GET", "/telemetry/v1/events?cursor=-1")
    assert invalid_cursor.status_code == 400

    prompt = "the quick brown fox jumps over the lazy dog"
    tokenized = server.make_request("POST", "/tokenize", data={"content": prompt})
    assert tokenized.status_code == 200
    first = server.make_request(
        "POST",
        "/completion",
        data={"prompt": tokenized.body["tokens"], "n_predict": 2, "return_tokens": True},
    )
    # Continue the exact token stream. Text concatenation can retokenize across the
    # prompt/response boundary, so use the authoritative token IDs for this cache
    # invariant instead of relying on tokenizer-specific string behavior.
    second_prompt = tokenized.body["tokens"] + first.body["tokens"]
    second = server.make_request("POST", "/completion", data={"prompt": second_prompt, "n_predict": 2})
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.body["trace_id"] != second.body["trace_id"]
    assert second.headers["X-Llama-Trace-Id"] == second.body["trace_id"]

    live_events = trace_events(second.body["trace_id"])
    started = next(candidate for candidate in live_events if candidate["event"] == "request_started")
    first_token = next(candidate for candidate in live_events if candidate["event"] == "first_token")
    assert started["prompt_tokens"] > 0
    assert started["server_configuration"]["parallel_slots"] >= 1
    assert first_token["prompt_tokens"] == first_token["reused_prompt_tokens"] + first_token["evaluated_prompt_tokens"]
    assert 0 < first_token["matched_prefix_tokens"] <= first_token["prompt_tokens"]
    assert first_token["reused_prompt_tokens"] > 0
    expected_cache_status = (
        "full"
        if first_token["matched_prefix_tokens"] == first_token["prompt_tokens"]
        else "partial"
    )
    assert first_token["cache_status"] == expected_cache_status
    assert first_token["prefill_meaningful"] is (expected_cache_status != "full")
    assert first_token["cache_lookup_ms"] >= 0
    assert first_token["actual_prefill_ms"] >= 0
    assert first_token["server_configuration"] == started["server_configuration"]

    event = completed_event(second.body["trace_id"])
    assert event["prompt_tokens"] == event["reused_prompt_tokens"] + event["evaluated_prompt_tokens"]
    assert event["cache_status"] == expected_cache_status
    assert event["matched_prefix_tokens"] == first_token["matched_prefix_tokens"]
    assert event["timings"]["ttft_ms"] >= event["timings"]["queue_ms"] >= 0
    assert event["timings"]["e2e_ms"] >= event["timings"]["ttft_ms"]
    token_detail = event["output_token_telemetry"]
    assert token_detail["state"] == "not_enabled_for_request"
    assert token_detail["captured_tokens"] == 0
    assert token_detail["dropped_tokens"] == token_detail["total_committed_tokens"]
    assert token_detail["probability_state"] == "not_enabled_for_request"
    assert token_detail["mtp_pass_state"] == "not_enabled_for_request"
    assert token_detail["mtp_pass_records"] == []
    assert token_detail["records"] == []
    assert_complete_lifecycle_clock(event)

    first_event = completed_event(first.body["trace_id"])
    assert first_event["slot_id"] == event["slot_id"] == 0
    assert event["slot_assignment_ordinal"] == first_event["slot_assignment_ordinal"] + 1
    assert started["slot_id"] == event["slot_id"]
    assert started["slot_assignment_ordinal"] == event["slot_assignment_ordinal"]
    assert first_token["slot_assignment_ordinal"] == event["slot_assignment_ordinal"]
    assert len([
        candidate for candidate in trace_events(second.body["trace_id"])
        if candidate["event"] == "request_completed"
    ]) == 1


def test_output_token_telemetry_is_bounded_opt_in_with_independent_probability_and_identity(monkeypatch):
    monkeypatch.setenv("LLAMA_TELEMETRY_OUTPUT_TOKEN_LIMIT", "32")
    server.start()
    apply_telemetry_control(output_token_detail=True)

    capabilities = server.make_request("GET", "/telemetry/v1/capabilities")
    assert capabilities.status_code == 200
    capability = capabilities.body["capabilities"]["output_token_telemetry"]
    assert capability["state"] == "conditional"
    assert capability["maximum_captured_tokens"] == 32
    assert capability["record_schema_version"] == 3
    assert capability["mtp_pass_record_schema_version"] == 2
    assert capability["maximum_captured_mtp_passes"] == 512
    assert capability["maximum_captured_mtp_proposals"] == 512
    assert set(capability["retained_mtp_proposal_fields"]) >= {
        "position",
        "disposition",
        "evaluated_actual_target_pass",
        "draft_token_id",
        "draft_token_piece_base64",
        "target_selected_token_id",
        "target_selected_token_piece_base64",
        "target_selected_log_probability_ln",
        "committed_output_ordinal",
    }
    assert set(capability["retained_linkage"]) >= {
        "model_ready_monotonic_us",
        "model_position",
        "origin",
        "logical_step",
        "actual_target_pass",
        "proposal_position",
    }
    assert capability["normal_request_telemetry_unaffected"] is False

    response = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "A tiny output-token telemetry test",
            "n_predict": 4,
            "n_probs": 1,
            "output_token_telemetry": True,
            "temperature": 0,
            "ignore_eos": True,
        },
    )
    assert response.status_code == 200
    event = completed_event(response.body["trace_id"])
    detail = event["output_token_telemetry"]
    assert_rich_diagnostics_finish_by_last_generation_work(event)

    assert detail["schema_version"] == 3
    assert detail["mtp_pass_record_schema_version"] == 2
    assert detail["state"] == "available"
    assert detail["population"] == "committed_generation_tokens"
    assert detail["total_committed_tokens"] == event["output_tokens"] == 4
    assert detail["captured_tokens"] == 4
    assert detail["scored_tokens"] == 4
    assert detail["dropped_tokens"] == 0
    assert detail["probability_state"] == "available"
    assert detail["token_identity_state"] == "not_captured"
    assert detail["token_piece_state"] == "not_captured"
    assert detail["mtp_pass_state"] in ["not_applicable", "no_data"]
    assert detail["mtp_passes_total"] == 0
    assert detail["mtp_passes_captured"] == 0
    assert detail["mtp_passes_dropped"] == 0
    assert detail["mtp_proposal_state"] == detail["mtp_pass_state"]
    assert detail["mtp_proposals_total"] == 0
    assert detail["mtp_proposals_captured"] == 0
    assert detail["mtp_proposals_dropped"] == 0
    assert detail["mtp_pass_records"] == []
    assert len(detail["records"]) == 4
    assert [record["ordinal"] for record in detail["records"]] == list(range(4))
    assert [record["model_ready_offset_us"] for record in detail["records"]] == sorted(
        record["model_ready_offset_us"] for record in detail["records"]
    )
    assert [record["model_ready_monotonic_us"] for record in detail["records"]] == sorted(
        record["model_ready_monotonic_us"] for record in detail["records"]
    )
    for record in detail["records"]:
        assert record["model_ready_offset_us"] >= 0
        assert record["model_ready_monotonic_us"] > 0
        assert record["model_position"] >= 0
        assert record["model_position_state"] == "available"
        assert record["selected_log_probability_ln"] <= 0
        assert record["probability_state"] == "available"
        assert record["token_id"] is None
        assert record["token_identity_state"] == "not_captured"
        assert record["token_piece_base64"] is None
        assert record["token_piece_state"] == "not_captured"
        assert record["origin"] == "normal_decode"
        assert record["origin_state"] == "available"
        assert record["mtp_linkage_state"] == "not_applicable"
        assert record["logical_step"] is None

    no_request_opt_in = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "Explicit request-level diagnostic opt out",
            "n_predict": 1,
            "output_token_telemetry": False,
        },
    )
    assert no_request_opt_in.status_code == 200
    not_enabled = completed_event(no_request_opt_in.body["trace_id"])["output_token_telemetry"]
    assert not_enabled["state"] == "not_enabled_for_request"
    assert not_enabled["captured_tokens"] == 0
    assert not_enabled["dropped_tokens"] == not_enabled["total_committed_tokens"]

    capped_response = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "Exercise the hard per-request token-detail cap",
            "n_predict": 40,
            "ignore_eos": True,
            "output_token_telemetry": True,
        },
    )
    assert capped_response.status_code == 200
    capped = completed_event(capped_response.body["trace_id"])["output_token_telemetry"]
    assert capped["state"] == "truncated"
    assert capped["reason"] == "per_request_output_token_limit_reached"
    assert capped["total_committed_tokens"] == 40
    assert capped["captured_tokens"] == 32
    assert capped["dropped_tokens"] == 8
    assert capped["probability_state"] == "not_enabled_for_request"
    assert len(capped["records"]) == 32
    assert [record["ordinal"] for record in capped["records"]] == list(range(32))


@pytest.mark.parametrize("configured_limit,expected_limit", [(8192, 8192), (8193, 8192)])
def test_output_token_limit_accepts_8192_and_clamps_larger_values(monkeypatch, configured_limit, expected_limit):
    monkeypatch.setenv("LLAMA_TELEMETRY_OUTPUT_TOKEN_LIMIT", str(configured_limit))
    server.start()
    apply_telemetry_control(output_token_detail=True)

    capability = server.make_request(
        "GET", "/telemetry/v1/capabilities"
    ).body["capabilities"]["output_token_telemetry"]
    assert capability["maximum_captured_tokens"] == expected_limit
    assert capability["maximum_captured_mtp_passes"] == 512
    assert capability["maximum_captured_mtp_proposals"] == 512


def test_output_token_identity_follows_the_separate_content_policy(monkeypatch):
    server.start()
    apply_telemetry_control(output_token_detail=True, request_content=True)

    response = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "Safe content-policy identity fixture",
            "n_predict": 2,
            "ignore_eos": True,
            "output_token_telemetry": True,
        },
    )
    assert response.status_code == 200
    event = completed_event(response.body["trace_id"])
    detail = event["output_token_telemetry"]

    assert detail["state"] == "available"
    assert detail["probability_state"] == "not_enabled_for_request"
    assert detail["token_identity_state"] == "available"
    assert detail["token_piece_state"] == "available"
    assert len(detail["records"]) == 2
    assert all(isinstance(record["token_id"], int) for record in detail["records"])
    assert all(record["token_identity_state"] == "available" for record in detail["records"])
    assert all(record["token_piece_state"] == "available" for record in detail["records"])
    piece_bytes = b"".join(base64.b64decode(record["token_piece_base64"]) for record in detail["records"])
    assert piece_bytes == response.body["content"].encode("utf-8")
    assert all(record["selected_log_probability_ln"] is None for record in detail["records"])
    assert event["request"]["original_request"]["prompt"] == "Safe content-policy identity fixture"
    assert event["response"] == response.body["content"]


def test_output_token_control_enables_request_defaults():
    configure_ngram_simple_speculation(server)
    server.start()
    apply_telemetry_control(
        output_token_detail=True,
        token_candidates=True,
        request_content=True,
    )

    capabilities = server.make_request("GET", "/telemetry/v1/capabilities")
    assert capabilities.status_code == 200
    output_capability = capabilities.body["capabilities"]["output_token_telemetry"]
    candidate_capability = capabilities.body["capabilities"]["output_token_candidates"]
    assert output_capability["state"] == "conditional"
    assert candidate_capability["state"] == "conditional"
    assert output_capability["automatic_request_defaults"] is True
    assert candidate_capability["automatic_request_defaults"] is True
    assert output_capability["normal_request_telemetry_unaffected"] is False
    assert candidate_capability["normal_request_telemetry_unaffected"] is False

    response = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "repeat:",
            "n_predict": 12,
            "n_probs": 1,
            "temperature": 0,
            "ignore_eos": True,
            "grammar": 'root ::= " a a a a a a a a a a a a"',
        },
    )
    assert response.status_code == 200
    trace_id = response.body["trace_id"]
    event = completed_event(trace_id)
    assert_rich_diagnostics_finish_by_last_generation_work(event)
    detail = event["output_token_telemetry"]
    assert detail["schema_version"] == 3
    assert detail["state"] == "available"
    assert detail["probability_state"] == "available"
    assert detail["token_identity_state"] == "available"
    assert detail["token_piece_state"] == "available"
    assert detail["captured_tokens"] == detail["total_committed_tokens"] == 12
    assert all(record["token_id"] is not None for record in detail["records"])
    assert all(record["token_piece_state"] == "available" for record in detail["records"])
    assert all(
        base64.b64decode(record["token_piece_base64"], validate=True) is not None
        for record in detail["records"]
    )
    assert detail["candidate_detail_state"] != "not_enabled_for_request"

    candidate_detail = server.make_request(
        "GET",
        f"/telemetry/v1/token-candidates?trace_id={trace_id}",
    )
    assert candidate_detail.status_code == 200
    assert candidate_detail.body["schema_version"] == 2
    assert candidate_detail.body["state"] != "not_enabled_for_request"


def test_content_disabled_emits_an_explicit_omission_marker():
    server.start()
    response = server.make_request(
        "POST",
        "/completion",
        data={"prompt": "sensitive content must not enter telemetry", "n_predict": 1},
    )
    assert response.status_code == 200

    event = completed_event(response.body["trace_id"])
    assert event["request"] == {
        "content_omitted": True,
        "reason": "content_logging_disabled",
    }
    assert "response" not in event
    assert isinstance(event["slot_id"], int) and event["slot_id"] >= 0
    assert event["slot_assignment_ordinal"] > 0
    assert_complete_lifecycle_clock(event)


def test_success_error_and_cancel_each_emit_one_release_complete_event():
    server.n_ctx = 512
    server.n_predict = 4096
    server.n_slots = 1
    server.start()

    success = server.make_request(
        "POST",
        "/completion",
        data={"prompt": "lifecycle success", "n_predict": 2, "ignore_eos": True},
    )
    assert success.status_code == 200
    success_event = terminal_event(success.body["trace_id"], "request_completed", "success")
    assert_complete_lifecycle_clock(success_event)

    error = server.make_request(
        "POST",
        "/completion",
        data={"prompt": "context overflow " * 1024, "n_predict": 1},
    )
    assert error.status_code == 400
    error_trace_id = error.body["error"]["trace_id"]
    error_event = terminal_event(error_trace_id, "request_ended", "error")
    assert error_event["error_category"] == "context_size"
    assert_complete_lifecycle_clock(error_event, require_generation=False)

    with requests.post(
        f"http://{server.server_host}:{server.server_port}/completion",
        json={
            "prompt": "lifecycle cancellation",
            "n_predict": 4096,
            "ignore_eos": True,
            "stream": True,
        },
        headers={"Authorization": f"Bearer {server.api_key}"},
        stream=True,
        timeout=5.0,
    ) as streaming_response:
        assert streaming_response.status_code == 200
        started_trace_id = streaming_response.headers["X-Llama-Trace-Id"]
        first_line = next(line for line in streaming_response.iter_lines() if line)
        assert first_line.startswith(b"data: ")

    cancelled = None
    deadline = time.time() + 10.0
    while time.time() < deadline:
        events = server.make_request("GET", "/telemetry/v1/events?cursor=0&limit=100").body["events"]
        matches = [
            candidate for candidate in events
            if candidate["trace_id"] == started_trace_id
            and candidate["event"] == "request_ended"
            and candidate["outcome"] == "cancelled"
        ]
        if matches:
            cancelled = matches[-1]
            break
        time.sleep(0.05)

    assert cancelled is not None
    assert_complete_lifecycle_clock(cancelled, require_generation=False, require_handoff=False)
    terminal_event(cancelled["trace_id"], "request_ended", "cancelled")


def test_w3c_traceparent_valid_and_invalid():
    server.n_slots = 2
    server.start()
    correlation_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    traceparent = f"00-{correlation_id}-00f067aa0ba902b7-01"

    valid = server.make_request(
        "POST",
        "/completion",
        data={"prompt": "valid trace context", "n_predict": 1},
        headers={"traceparent": traceparent},
    )
    event = completed_event(valid.body["trace_id"])
    assert event["w3c_trace_id"] == correlation_id
    assert event["w3c_traceparent"] == traceparent
    assert event["trace_id"] != correlation_id

    invalid = server.make_request(
        "POST",
        "/completion",
        data={"prompt": "invalid trace context", "n_predict": 1},
        headers={"traceparent": "00-00000000000000000000000000000000-0000000000000000-01"},
    )
    event = completed_event(invalid.body["trace_id"])
    assert event["w3c_trace_id"] is None
    assert event["w3c_traceparent"] is None

    child_correlation_id = "11111111111111111111111111111111"
    child_traceparent = f"00-{child_correlation_id}-2222222222222222-01"
    multiple = server.make_request(
        "POST",
        "/completion",
        data={"prompt": "shared correlation with distinct children", "n_predict": 1, "n": 2},
        headers={"traceparent": child_traceparent},
    )
    assert multiple.status_code == 200
    events = server.make_request("GET", "/telemetry/v1/events?cursor=0&limit=100").body["events"]
    completions = [
        candidate for candidate in events
        if candidate["event"] == "request_completed"
        and candidate["w3c_trace_id"] == child_correlation_id
    ]
    assert len(completions) == 2
    assert len({candidate["trace_id"] for candidate in completions}) == 2
    assert all(candidate["w3c_traceparent"] == child_traceparent for candidate in completions)
    assert len({candidate["slot_assignment_ordinal"] for candidate in completions}) == 2
    for completion in completions:
        terminal = terminal_event(completion["trace_id"], "request_completed", "success")
        assert_complete_lifecycle_clock(terminal)


def test_embedding_assignments_emit_one_release_complete_event():
    global server
    server = ServerPreset.tinyllama2()
    server.server_embeddings = True
    server.server_metrics = True
    server.pooling = "last"
    server.start()

    response = server.make_request(
        "POST",
        "/v1/embeddings",
        data={"input": ["first lifecycle embedding", "second lifecycle embedding"]},
    )
    assert response.status_code == 200

    events = server.make_request("GET", "/telemetry/v1/events?cursor=0&limit=100").body["events"]
    starts = [event for event in events if event["event"] == "request_started"]
    assert len(starts) == 2
    assert len({event["trace_id"] for event in starts}) == 2
    assert len({event["slot_assignment_ordinal"] for event in starts}) == 2
    for started in starts:
        terminal = terminal_event(started["trace_id"], "request_completed", "success")
        assert terminal["slot_id"] == started["slot_id"]
        assert terminal["slot_assignment_ordinal"] == started["slot_assignment_ordinal"]
        assert terminal["sampling"]["effective_seed"] is None
        assert terminal["output_tokens"] == 0
        assert terminal["lifecycle_clock"]["first_token_monotonic_us"] is None
        assert terminal["lifecycle_clock"]["last_generation_work_monotonic_us"] is None
        assert_complete_lifecycle_clock(terminal, require_generation=False)


def test_rerank_assignments_emit_one_release_complete_event():
    global server
    server = ServerPreset.jina_reranker_tiny()
    server.server_metrics = True
    server.start()

    response = server.make_request(
        "POST",
        "/rerank",
        data={
            "query": "telemetry lifecycle",
            "documents": ["native request accounting", "unrelated document"],
        },
    )
    assert response.status_code == 200

    events = server.make_request("GET", "/telemetry/v1/events?cursor=0&limit=100").body["events"]
    starts = [event for event in events if event["event"] == "request_started"]
    assert len(starts) == 2
    assert len({event["trace_id"] for event in starts}) == 2
    assert len({event["slot_assignment_ordinal"] for event in starts}) == 2
    for started in starts:
        terminal = terminal_event(started["trace_id"], "request_completed", "success")
        assert terminal["slot_id"] == started["slot_id"] == 0
        assert terminal["slot_assignment_ordinal"] == started["slot_assignment_ordinal"]
        assert terminal["sampling"]["effective_seed"] is None
        assert terminal["output_tokens"] == 0
        assert terminal["lifecycle_clock"]["first_token_monotonic_us"] is None
        assert terminal["lifecycle_clock"]["last_generation_work_monotonic_us"] is None
        assert_complete_lifecycle_clock(terminal, require_generation=False)


def test_prometheus_histograms_are_cumulative():
    server.start()
    server.make_request("POST", "/completion", data={"prompt": "histogram", "n_predict": 2})
    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    lines = metrics.body.splitlines()
    generation_steps = next(
        float(line.split(" ")[-1])
        for line in lines
        if line.startswith("llamacpp:generation_steps_total ")
    )
    assert generation_steps > 0
    for name in ["request_ttft_seconds", "logical_batch_tokens", "physical_ubatch_target_tokens"]:
        buckets = [line for line in lines if line.startswith(f"llamacpp:{name}_bucket")]
        count = next(line.split(" ")[-1] for line in lines if line.startswith(f"llamacpp:{name}_count"))
        assert buckets[-1].split(" ")[-1] == count
        assert float(count) > 0
        values = [float(line.split(" ")[-1]) for line in buckets]
        assert values == sorted(values)


def test_structured_forward_and_kv_diagnostics_are_measured():
    server.start()
    response = server.make_request(
        "POST",
        "/completion",
        data={"prompt": "physical forward and kv diagnostics", "n_predict": 3},
    )
    assert response.status_code == 200

    snapshot = server.make_request("GET", "/telemetry/v1/snapshot")
    assert snapshot.status_code == 200
    throughput = snapshot.body["throughput"]
    assert throughput["generation_steps_total"] > 0
    assert throughput["server_output_tokens_total"] >= throughput["generation_steps_total"]
    assert throughput["server_output_tps"] is None
    assert throughput["server_output_tps_state"] == "derive_from_counter_delta"
    forward = snapshot.body["forward_pass"]
    assert forward["logical_tokens_per_call_distribution"]["count"] == forward["successful_calls"]
    assert forward["participating_slots_per_call_distribution"]["count"] == forward["successful_calls"]
    physical = forward["physical_ubatch"]["target"]
    assert physical["state"] == "available"
    assert physical["attempted"] >= physical["successful"] > 0
    assert physical["tokens"] >= physical["successful"]
    assert physical["p50_tokens_upper_bound"] is not None
    assert physical["p95_tokens_upper_bound"] is not None
    assert physical["token_buckets"][-1]["le"] == "+Inf"
    assert physical["token_buckets"][-1]["count"] == physical["successful"]

    live = snapshot.body["kv"]["live_occupancy"]
    assert live["state"] == "available"
    assert live["reason"]
    assert live["resident_tokens_state"] == "not_collected"
    assert live["resident_tokens_reason"]
    assert live["capacity_entries"] >= live["used_entries"] > 0
    assert live["free_entries"] == live["capacity_entries"] - live["used_entries"]
    assert snapshot.body["kv"]["physical_prefix_sharing"]["state"] == "not_collected"
    assert snapshot.body["kv"]["churn"]["state"] == "not_collected"

    kv = server.make_request("GET", "/telemetry/v1/kv?detail=deep")
    assert kv.status_code == 200
    assert kv.body["allocated"]["state"] == "available"
    assert kv.body["allocated"]["reason"]
    assert kv.body["allocated"]["total_bytes"] > 0
    assert kv.body["slot_metadata"]["state"] == "available"
    assert kv.body["slot_metadata"]["reason"]
    assert kv.body["live_occupancy"]["state"] == "available"
    assert kv.body["live_occupancy"]["resident_tokens_state"] in ("available", "not_applicable")
    assert kv.body["live_occupancy"]["resident_tokens_reason"]
    assert kv.body["physical_prefix_sharing"]["state"] == "available"
    assert kv.body["duplicate_prefix_opportunities"]["state"] == "available"
    assert kv.body["churn"]["state"] == "available"
    assert kv.body["churn"]["reason"]
    assert len(kv.body["components"]) > 0
    for component in kv.body["components"]:
        assert component["resident_tokens_state"] in ("available", "not_applicable")
        assert component["resident_tokens_reason"]

    capabilities = server.make_request("GET", "/telemetry/v1/capabilities").body["capabilities"]
    assert capabilities["physical_ubatch_observed"]["state"] == "available"
    assert capabilities["kv_live_occupancy"]["state"] == "available"
    assert capabilities["physical_prefix_sharing"]["state"] == "available"
    assert capabilities["duplicate_prefix_opportunities"]["state"] == "available"
    assert capabilities["response_perplexity"]["state"] == "conditional"
    assert capabilities["prompt_perplexity"]["state"] == "conditional"
    assert capabilities["moe_routing"]["state"] == "not_applicable"


def test_response_probability_invariant():
    server.start()
    response = server.make_request(
        "POST",
        "/completion",
        data={"prompt": "probability invariant", "n_predict": 2, "n_probs": 1},
    )
    assert response.status_code == 200
    probability = completed_event(response.body["trace_id"])["response_probability"]
    assert probability["available"] is True
    assert probability["semantics"] == "raw_target_model_pre_sampler_selected_token_probability"
    assert probability["perplexity_state"] == "available"
    assert probability["perplexity"] == pytest.approx(math.exp(probability["mean_nll"]))


def test_prompt_perplexity_is_exact_and_opt_in():
    server.start()
    apply_telemetry_control(prompt_perplexity=True)
    response = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "prompt perplexity exact",
            "n_predict": 1,
            "prompt_perplexity": True,
        },
    )
    assert response.status_code == 200
    event = completed_event(response.body["trace_id"])
    probability = event["prompt_probability"]
    assert probability["available"] is True
    assert probability["state"] == "available"
    assert probability["semantics"] == "raw_target_model_next_token_probability_cold_text_prompt"
    assert probability["scored_tokens"] == event["prompt_tokens"] - 1
    assert probability["conditioning_tokens"] == 1
    assert probability["cache_reuse_disabled"] is True
    assert event["cache_status"] == "miss"
    assert event["reused_prompt_tokens"] == 0
    assert event["evaluated_prompt_tokens"] == event["prompt_tokens"]
    assert probability["perplexity_state"] == "available"
    assert probability["perplexity"] == pytest.approx(math.exp(probability["mean_nll"]))

    moe = completed_event(response.body["trace_id"])["moe_routing"]
    assert moe["state"] == "not_applicable"
    assert moe["configuration_state"] == "not_applicable"
    assert moe["configuration_reason"]
    assert moe["expert_activations"] == []
    assert moe["routed_tokens"] is None
    assert moe["token_detail_state"] == "not_applicable"
    assert moe["selected_expert_ids_state"] == "not_applicable"
    assert moe["routing_weights_state"] == "not_applicable"
    assert moe["router_margin_state"] == "not_applicable"
    assert moe["token_decisions"] == []

    disabled = server.make_request(
        "POST",
        "/completion",
        data={"prompt": "prompt perplexity disabled", "n_predict": 1, "prompt_perplexity": False},
    )
    assert disabled.status_code == 200
    disabled_probability = completed_event(disabled.body["trace_id"])["prompt_probability"]
    assert disabled_probability == {
        "state": "disabled",
        "available": False,
        "reason": "request_did_not_enable_prompt_perplexity",
    }


def test_moe_routing_chunks_cover_prefill_and_decode_with_props_control():
    global server

    api_key = "moe-routing-chunk-test-key"
    auth = {"Authorization": f"Bearer {api_key}"}
    server = ServerPreset.stories15m_moe()
    server.server_metrics = True
    server.server_props = True
    server.api_key = api_key
    server.n_batch = 4
    server.n_ubatch = 2
    server.start()

    try:
        control = server.make_request(
            "POST",
            "/props",
            data={"telemetry_control": {"moe_routing": True}},
            headers=auth,
        )
        assert control.status_code == 200
        assert control.body["telemetry_control"]["generation"] == 1

        response = server.make_request(
            "POST",
            "/completion",
            data={
                "prompt": "A routed prefill and decode capture.",
                "n_predict": 3,
                "ignore_eos": True,
                "temperature": 0,
            },
            headers=auth,
        )
        assert response.status_code == 200
        trace_id = response.body["trace_id"]
        events_http = requests.get(
            f"http://{server.server_host}:{server.server_port}/telemetry/v1/events?cursor=0&limit=512",
            headers=auth,
            timeout=60,
        )
        assert events_http.status_code == 200
        events_response = events_http.json()
        chunks = [
            event for event in events_response["events"]
            if event["event"] == "moe_routing_chunk" and event["trace_id"] == trace_id
        ]
        assert chunks
        assert all(chunk["moe_routing_schema_version"] == 2 for chunk in chunks)
        assert [chunk["trace_chunk_sequence"] for chunk in chunks] == list(range(1, len(chunks) + 1))
        assert chunks[-1]["final"] is True
        assert chunks[-1]["decisions"] == []
        assert chunks[-1]["producer_coverage"]["state"] == "complete"

        decisions = [
            decision for chunk in chunks if not chunk["final"] for decision in chunk["decisions"]
        ]
        assert decisions
        assert [decision["trace_decision_sequence"] for decision in decisions] == list(
            range(1, len(decisions) + 1)
        )
        assert {decision["phase"] for decision in decisions} >= {"prefill", "decode"}
        assert all(chunk["serialized_bytes"] <= 1024 * 1024 for chunk in chunks)
        assert all(
            chunk["serialized_bytes"] == len(
                json.dumps(chunk, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
            for chunk in chunks
        )
        serialized_events = [
            json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            for event in events_response["events"]
        ]
        assert b'"events":[' + b",".join(serialized_events) + b"]" in events_http.content
        assert all(chunk["sequence"] > 0 for chunk in chunks)
        assert all(chunk["server_instance_id"] == events_response["server_instance_id"] for chunk in chunks)
        assert all(chunk["physical_peer_coverage"] in ["complete", "partial"] for chunk in chunks)
        assert all("shared_experts" in chunk for chunk in chunks)
        assert len({decision["physical_ubatch_index"] for decision in decisions}) >= 2
        for decision in decisions:
            assert decision["physical_step_id"] > 0
            assert decision["physical_ubatch_index"] >= 0
            assert decision["control_generation"] == 1
            assert decision["kth_selected_score_status"] in [0, 1, 2, 3]
            assert decision["highest_rejected_score_status"] in [0, 1, 2, 3]
            assert decision["selected_experts"]
            for selected in decision["selected_experts"]:
                assert selected["expert_id_status"] in [0, 1, 2, 3]
                assert selected["effective_weight_status"] in [0, 1, 2, 3]
                if selected["effective_weight_status"] == 0:
                    assert selected["effective_weight"] is not None

        opted_out = server.make_request(
            "POST",
            "/completion",
            data={
                "prompt": "The request opts out of routing detail.",
                "n_predict": 1,
                "moe_routing_telemetry": False,
            },
            headers=auth,
        )
        assert opted_out.status_code == 200
        events_response = server.make_request(
            "GET", "/telemetry/v1/events?cursor=0&limit=512", headers=auth
        )
        assert not any(
            event["event"] == "moe_routing_chunk" and event["trace_id"] == opted_out.body["trace_id"]
            for event in events_response.body["events"]
        )
    finally:
        server.stop()


def test_moe_routing_chunks_link_decoder_mtp_context_when_available():
    api_key = "moe-routing-mtp-test-key"
    auth = {"Authorization": f"Bearer {api_key}"}
    mtp_server = ServerPreset.stories15m_moe()
    configure_embedded_mtp_fixture(mtp_server)
    mtp_server.server_props = True
    mtp_server.api_key = api_key
    mtp_server.spec_draft_n_min = 1
    mtp_server.n_batch = 4
    mtp_server.n_ubatch = 2
    mtp_server.start()

    try:
        control = mtp_server.make_request(
            "POST",
            "/props",
            data={"telemetry_control": {"moe_routing": True}},
            headers=auth,
        )
        assert control.status_code == 200

        response = mtp_server.make_request(
            "POST",
            "/completion",
            data={
                "prompt": "MTP routing captures the target verification pass.",
                "n_predict": 16,
                "ignore_eos": True,
                "temperature": 0,
            },
            headers=auth,
        )
        assert response.status_code == 200

        events = mtp_server.make_request(
            "GET", "/telemetry/v1/events?cursor=0&limit=512", headers=auth
        )
        assert events.status_code == 200
        decisions = [
            decision
            for event in events.body["events"]
            if event["event"] == "moe_routing_chunk"
            and event["trace_id"] == response.body["trace_id"]
            and not event["final"]
            for decision in event["decisions"]
        ]
        mtp_decisions = [decision for decision in decisions if decision["phase"] == "mtp_verify"]
        assert mtp_decisions
        # LLAMA_CONTEXT_TYPE_MTP maps directly to LLM_GRAPH_TYPE_DECODER_MTP (3).
        assert all(decision["graph_type"] == 3 for decision in mtp_decisions)
        assert all(decision["model_position"] is not None for decision in mtp_decisions)
        assert all(decision["logical_step"] is not None for decision in mtp_decisions)
        assert all(decision["actual_target_pass"] is not None for decision in mtp_decisions)
        assert all(decision["proposal_position"] is not None for decision in mtp_decisions)
        assert all(isinstance(decision["replay_pass"], bool) for decision in mtp_decisions)
    finally:
        mtp_server.stop()


def test_moe_routing_chunks_mark_on_off_on_intervals_partial():
    global server

    api_key = "moe-routing-toggle-test-key"
    auth = {"Authorization": f"Bearer {api_key}"}
    server = ServerPreset.stories15m_moe()
    server.server_props = True
    server.api_key = api_key
    # Keep this request in flight across multiple physical microbatches so the
    # control boundaries are observed by the decode-side capture path.
    server.n_batch = 4
    server.n_ubatch = 2
    server.n_threads = 1
    server.start()

    try:
        enabled = server.make_request(
            "POST", "/props", data={"telemetry_control": {"moe_routing": True}}, headers=auth
        )
        assert enabled.status_code == 200

        with ThreadPoolExecutor(max_workers=1) as executor:
            completion = executor.submit(
                server.make_request,
                "POST",
                "/completion",
                {
                    "prompt": "routing interval " * 96,
                    "n_predict": 24,
                    "ignore_eos": True,
                    "temperature": 0,
                },
                auth,
            )
            toggled = False
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not completion.done():
                events = server.make_request(
                    "GET", "/telemetry/v1/events?cursor=0&limit=512", headers=auth
                )
                if any(event["event"] == "moe_routing_chunk" for event in events.body["events"]):
                    disabled = server.make_request(
                        "POST", "/props", data={"telemetry_control": {}}, headers=auth
                    )
                    assert disabled.status_code == 200
                    assert disabled.body["telemetry_control"]["generation"] == 2
                    time.sleep(0.05)
                    reenabled = server.make_request(
                        "POST", "/props", data={"telemetry_control": {"moe_routing": True}}, headers=auth
                    )
                    assert reenabled.status_code == 200
                    assert reenabled.body["telemetry_control"]["generation"] == 3
                    toggled = True
                    break
                time.sleep(0.01)
            assert toggled
            response = completion.result(timeout=60)
            assert response.status_code == 200

        events = server.make_request(
            "GET", "/telemetry/v1/events?cursor=0&limit=512", headers=auth
        )
        chunks = [
            event for event in events.body["events"]
            if event["event"] == "moe_routing_chunk" and event["trace_id"] == response.body["trace_id"]
        ]
        assert chunks[-1]["final"] is True
        assert chunks[-1]["producer_coverage"]["state"] == "partial"
        assert chunks[-1]["capture_interruption_reason"] == "telemetry_control_disabled"
        assert any(
            decision["control_generation"] == 1
            for chunk in chunks if not chunk["final"] for decision in chunk["decisions"]
        )
        assert any(
            decision["control_generation"] == 3
            for chunk in chunks if not chunk["final"] for decision in chunk["decisions"]
        )
    finally:
        server.stop()


def test_moe_routing_states_and_bounded_histogram(monkeypatch):
    global server

    monkeypatch.delenv("LLAMA_TELEMETRY_MOE_ROUTING", raising=False)
    monkeypatch.delenv("LLAMA_TELEMETRY_MOE_ACTIVATION_LIMIT", raising=False)
    server = ServerPreset.stories15m_moe()
    server.server_metrics = True
    configure_telemetry_server()
    server.start()

    capability = server.make_request("GET", "/telemetry/v1/capabilities").body["capabilities"]["moe_routing"]
    assert capability["state"] == "conditional"
    assert capability["configured_experts"] == 4
    assert capability["experts_per_token"] == 2
    assert capability["disabled_path_changes_graph"] is False

    disabled_response = server.make_request(
        "POST",
        "/completion",
        data={"prompt": "disabled MoE diagnostic", "n_predict": 1, "moe_routing_telemetry": True},
    )
    assert disabled_response.status_code == 200
    disabled = completed_event(disabled_response.body["trace_id"])["moe_routing"]
    assert disabled["state"] == "not_enabled_for_request"
    assert disabled["configuration_state"] == "available"
    assert disabled["configured_experts"] == 4
    assert disabled["expert_activations"] == []
    assert disabled["token_detail_state"] == "not_enabled_for_request"
    assert disabled["token_detail_reason"]
    assert disabled["selected_expert_ids_state"] == "not_enabled_for_request"
    assert disabled["routing_weights_state"] == "not_enabled_for_request"
    assert disabled["router_margin_state"] == "not_enabled_for_request"
    assert disabled["token_decisions"] == []
    server.stop()

    monkeypatch.setenv("LLAMA_TELEMETRY_MOE_ACTIVATION_LIMIT", "1024")
    server = ServerPreset.stories15m_moe()
    server.server_metrics = True
    configure_telemetry_server()
    server.start()
    apply_telemetry_control(moe_routing=True)

    capability = server.make_request("GET", "/telemetry/v1/capabilities").body["capabilities"]["moe_routing"]
    assert capability["state"] == "conditional"
    assert capability["maximum_captured_activations"] == 1024
    assert "POST /props telemetry_control.moe_routing=true" in capability["enable_with"]
    assert capability["token_detail_schema_version"] == 2
    assert capability["token_detail_population"] == "target_model_output_logit_rows_by_layer"
    assert capability["routing_weights_state"] == "conditional"
    assert capability["router_margin_state"] == "conditional"

    plain_response = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "explicit request opt out",
            "n_predict": 1,
            "moe_routing_telemetry": False,
        },
    )
    assert plain_response.status_code == 200
    plain = completed_event(plain_response.body["trace_id"])["moe_routing"]
    assert plain["state"] == "not_enabled_for_request"
    assert plain["configuration_state"] == "available"
    assert plain["configured_experts"] == 4
    assert plain["routed_tokens"] is None
    assert plain["expert_activations"] == []
    assert plain["token_detail_state"] == "not_enabled_for_request"
    assert plain["selected_expert_ids_state"] == "not_enabled_for_request"
    assert plain["routing_weights_state"] == "not_enabled_for_request"
    assert plain["router_margin_state"] == "not_enabled_for_request"
    assert plain["token_decisions"] == []

    available_response = server.make_request(
        "POST",
        "/completion",
        data={"prompt": "small routed request", "n_predict": 2, "moe_routing_telemetry": True},
    )
    assert available_response.status_code == 200
    available = completed_event(available_response.body["trace_id"])["moe_routing"]
    assert available["state"] == "available"
    assert available["configuration_state"] == "available"
    assert available["routed_tokens"] > 0
    assert available["routed_token_layer_decisions"] > 0
    assert available["expert_activations_total"] == available["routed_token_layer_decisions"] * available["experts_per_token"]
    assert available["expert_activations_captured"] == available["expert_activations_total"]
    assert sum(row["activation_count"] for row in available["expert_activations"]) == available["expert_activations_captured"]
    assert sum(row["share_percent"] for row in available["expert_activations"]) == pytest.approx(100.0)
    assert len({(row["layer_index"], row["expert_index"]) for row in available["expert_activations"]}) == len(available["expert_activations"])
    for layer in {row["layer_index"] for row in available["expert_activations"]}:
        assert sum(row["layer_share_percent"] for row in available["expert_activations"] if row["layer_index"] == layer) == pytest.approx(100.0)
    assert available["token_detail_state"] == "available"
    assert available["selected_expert_ids_state"] == "available"
    assert available["routing_weights_state"] == "available"
    assert available["router_margin_state"] == "available"
    assert available["token_detail_decisions_total"] == available["token_detail_decisions_captured"]
    assert available["token_detail_decisions_dropped"] == 0
    assert available["token_detail_activations_total"] == available["token_detail_activations_captured"]
    assert available["token_detail_activations_dropped"] == 0
    assert available["token_detail_invalid_activations"] == 0
    assert len(available["token_decisions"]) == available["token_detail_decisions_captured"]
    assert available["token_detail_decisions_total"] == available_response.body["timings"]["predicted_n"] * available["moe_layers"]
    assert {decision["phase"] for decision in available["token_decisions"]} == {"prefill_output", "normal_decode"}
    for decision in available["token_decisions"]:
        assert decision["model_position"] >= 1
        assert 0 <= decision["layer_index"]
        assert decision["logical_step"] is None
        assert decision["actual_target_pass"] is None
        assert decision["proposal_position"] is None
        assert decision["replay_pass"] is False
        assert len(decision["selected_expert_ids"]) == available["experts_per_token"]
        assert all(0 <= expert < available["configured_experts"] for expert in decision["selected_expert_ids"])
        assert len(decision["effective_expert_weights"]) == available["experts_per_token"]
        assert len(decision["normalized_expert_weight_shares"]) == available["experts_per_token"]
        assert all(weight >= 0 for weight in decision["effective_expert_weights"])
        assert all(0 <= share <= 1 for share in decision["normalized_expert_weight_shares"])
        assert sum(decision["normalized_expert_weight_shares"]) == pytest.approx(1.0)
        sorted_shares = sorted(decision["normalized_expert_weight_shares"], reverse=True)
        assert decision["normalized_router_margin"] == pytest.approx(sorted_shares[0] - sorted_shares[1])

    truncated_response = server.make_request(
        "POST",
        "/completion",
        data={"prompt": "telemetry " * 55, "n_predict": 4, "moe_routing_telemetry": True},
    )
    assert truncated_response.status_code == 200
    event = completed_event(truncated_response.body["trace_id"])
    truncated = event["moe_routing"]
    assert truncated["state"] == "truncated"
    assert truncated["configuration_state"] == "available"
    assert truncated["routed_tokens"] == event["evaluated_prompt_tokens"] + event["output_tokens"] - 1
    assert truncated["moe_layers"] > 0
    assert truncated["routed_token_layer_decisions"] >= truncated["routed_tokens"]
    assert truncated["expert_activations_total"] == truncated["routed_token_layer_decisions"] * truncated["experts_per_token"]
    assert truncated["expert_activations_captured"] == 1024
    assert truncated["dropped_activations"] == truncated["expert_activations_total"] - 1024
    assert sum(row["activation_count"] for row in truncated["expert_activations"]) == 1024
    assert sum(row["share_percent"] for row in truncated["expert_activations"]) == pytest.approx(100.0)
    assert len({(row["layer_index"], row["expert_index"]) for row in truncated["expert_activations"]}) == len(truncated["expert_activations"])
    for layer in {row["layer_index"] for row in truncated["expert_activations"]}:
        assert sum(row["layer_share_percent"] for row in truncated["expert_activations"] if row["layer_index"] == layer) == pytest.approx(100.0)
    assert all(0 <= row["expert_index"] < truncated["configured_experts"] for row in truncated["expert_activations"])
    assert truncated["token_detail_state"] in ["available", "truncated"]
    assert truncated["selected_expert_ids_state"] == truncated["token_detail_state"]
    assert truncated["routing_weights_state"] == truncated["token_detail_state"]
    assert truncated["router_margin_state"] == truncated["token_detail_state"]
    assert truncated["token_detail_activations_captured"] <= truncated["maximum_captured_token_detail_activations"]
    assert truncated["token_detail_decisions_captured"] == len(truncated["token_decisions"])
    assert truncated["token_detail_activations_dropped"] == (
        truncated["token_detail_activations_total"] - truncated["token_detail_activations_captured"]
    )


def test_moe_exact_token_linkage_survives_speculative_verification(monkeypatch):
    global server

    monkeypatch.setenv("LLAMA_TELEMETRY_MOE_ACTIVATION_LIMIT", "4096")
    server = ServerPreset.stories15m_moe()
    server.server_metrics = True
    configure_telemetry_server()
    server.model_draft = os.environ.get("LLAMA_TEST_DRAFT_MODEL_PATH") or download_file(MODEL_DRAFT_FILE_URL)
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 1
    server.spec_draft_n_max = 4
    server.fa = "off"
    server.start()
    apply_telemetry_control(moe_routing=True)

    response = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "Once upon a time there was a small cat. Once upon a time there was a small cat.",
            "n_predict": 40,
            "ignore_eos": True,
            "temperature": 0,
            "moe_routing_telemetry": True,
        },
    )
    assert response.status_code == 200
    event = completed_event(response.body["trace_id"])
    speculative = event["speculative"]
    routing = event["moe_routing"]
    verify_decisions = [
        decision for decision in routing["token_decisions"]
        if decision["phase"] == "mtp_verify"
    ]

    assert speculative["draft_tokens"] > 0
    assert speculative["actual_target_passes"] > 0
    assert routing["token_detail_state"] in ["available", "truncated"]
    assert routing["routing_weights_state"] == routing["token_detail_state"]
    assert routing["router_margin_state"] == routing["token_detail_state"]
    assert verify_decisions
    assert routing["token_detail_decisions_captured"] == len(routing["token_decisions"])
    assert routing["token_detail_activations_captured"] == sum(
        len(decision["selected_expert_ids"])
        for decision in routing["token_decisions"]
    )
    for decision in verify_decisions:
        assert decision["logical_step"] is not None
        assert decision["actual_target_pass"] is not None
        assert decision["proposal_position"] is not None
        assert isinstance(decision["replay_pass"], bool)
        assert len(decision["selected_expert_ids"]) == routing["experts_per_token"]
        assert all(0 <= expert < routing["configured_experts"] for expert in decision["selected_expert_ids"])
        assert len(decision["effective_expert_weights"]) == routing["experts_per_token"]
        assert len(decision["normalized_expert_weight_shares"]) == routing["experts_per_token"]
        assert sum(decision["normalized_expert_weight_shares"]) == pytest.approx(1.0)
        sorted_shares = sorted(decision["normalized_expert_weight_shares"], reverse=True)
        assert decision["normalized_router_margin"] == pytest.approx(sorted_shares[0] - sorted_shares[1])

    verify_keys = {
        (
            decision["model_position"],
            decision["layer_index"],
            decision["logical_step"],
            decision["actual_target_pass"],
            decision["proposal_position"],
        )
        for decision in verify_decisions
    }
    assert len(verify_keys) == len(verify_decisions)
    assert len({decision["actual_target_pass"] for decision in verify_decisions}) <= speculative["actual_target_passes"]


def test_native_gpu_gpm_capability_and_bounded_endpoint():
    server.start()

    capabilities = server.make_request("GET", "/telemetry/v1/capabilities")
    assert capabilities.status_code == 200
    gpu = capabilities.body["capabilities"]["gpu_gpm"]
    assert gpu["owner"] == "llama.cpp/llama-server"
    assert gpu["llamascope_queries_nvml"] is False
    assert [metric["id"] for metric in gpu["metrics"]] == [2, 5, 10]
    assert gpu["operation_phases"] == ["request", "prefill", "normal_decode", "mtp_draft", "mtp_verify"]
    assert gpu["state"] in [
        "available",
        "disabled",
        "initializing",
        "unsupported",
        "unavailable",
    ]

    invalid = server.make_request("GET", "/telemetry/v1/gpu?cursor=-1")
    assert invalid.status_code == 400

    response = server.make_request("GET", "/telemetry/v1/gpu?cursor=0&limit=2&trace_id=unknown")
    assert response.status_code == 200
    assert response.body["schema_version"] == 1
    assert response.body["server_instance_id"] == capabilities.body["server_instance_id"]
    assert response.body["source"]["collector"] == "asynchronous NVML GPM"
    assert response.body["source"]["measurement_semantics"].startswith("one shared interval-derived")
    assert response.body["trace_filter"] == "unknown"
    assert response.body["operations_truncated"] is False
    assert len(response.body["intervals"]) <= 2
    assert response.body["operations"] == []

    for interval in response.body["intervals"]:
        assert interval["start_monotonic_us"] <= interval["end_monotonic_us"]
        assert set(interval["readings"]) == {
            "sm_utilization",
            "tensor_core_utilization",
            "dram_bandwidth_utilization",
        }
        for reading in interval["readings"].values():
            assert reading["state"] in ["available", "unsupported", "unavailable"]
            assert (reading["value_percent"] is not None) == (reading["state"] == "available")


def test_speculative_invariants_and_ttft(monkeypatch):
    configure_embedded_mtp_fixture(server)
    server.start()
    apply_telemetry_control(output_token_detail=True)
    content_enabled = (
        server.make_request("GET", "/telemetry/v1/capabilities")
        .body["capabilities"]["content_events"]["state"]
        == "enabled"
    )
    response = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "repeat:",
            "n_predict": 20,
            "n_probs": 1,
            "output_token_telemetry": True,
            "temperature": 0,
            "grammar": 'root ::= " a a a a a a a a a a a a a a a a a a a a"',
        },
    )
    assert response.status_code == 200
    event = completed_event(response.body["trace_id"])
    assert_rich_diagnostics_finish_by_last_generation_work(event)
    speculative = event["speculative"]
    assert event["timings"]["ttft_ms"] > 0
    assert speculative["draft_tokens"] > 0
    assert speculative["draft_tokens"] == speculative["accepted_tokens"] + speculative["rejected_tokens"]
    assert speculative["logical_target_passes"] == speculative["hit_steps"] + speculative["miss_steps"]
    assert speculative["actual_target_passes"] >= speculative["logical_target_passes"]
    assert speculative["target_tokens"] >= speculative["actual_target_passes"]
    assert speculative["useful_output_tokens"] > 0
    assert speculative["target_tokens_per_pass"] == pytest.approx(
        speculative["target_tokens"] / speculative["actual_target_passes"]
    )
    assert speculative["useful_output_tokens_per_target_pass"] == pytest.approx(
        speculative["useful_output_tokens"] / speculative["actual_target_passes"]
    )
    assert "draft-mtp" in speculative["configuration"]["types"]
    assert event["sampling"]["requested_temperature"] == 0
    assert event["sampling"]["effective_temperature"] == 0
    assert event["sampling"]["effective_seed"] >= 0
    assert event["sampling"]["grammar"].startswith("root ::=")
    probability = event["response_probability"]
    assert probability["available"] is True
    assert probability["semantics"] == "raw_target_model_pre_sampler_selected_token_probability"
    assert probability["scored_tokens"] == event["output_tokens"]
    assert probability["perplexity"] == pytest.approx(math.exp(probability["mean_nll"]))
    token_detail = event["output_token_telemetry"]
    if token_detail["state"] in ["available", "truncated"]:
        assert token_detail["total_committed_tokens"] == event["output_tokens"]
        assert token_detail["captured_tokens"] == min(
            token_detail["total_committed_tokens"], token_detail["maximum_captured_tokens"]
        )
        assert token_detail["scored_tokens"] == token_detail["captured_tokens"]
        assert [record["ordinal"] for record in token_detail["records"]] == list(
            range(token_detail["captured_tokens"])
        )
        ready_offsets = [record["model_ready_offset_us"] for record in token_detail["records"]]
        assert ready_offsets == sorted(ready_offsets)
        if speculative["accepted_tokens"] > 0:
            # Several tokens committed by one speculative verification share the
            # one authoritative model-ready timestamp. That is interval-derived
            # evidence, not independent zero-latency hardware work.
            assert len(set(ready_offsets)) < len(ready_offsets)
        assert all(record["selected_log_probability_ln"] <= 0 for record in token_detail["records"])
        assert all(record["model_position_state"] == "available" for record in token_detail["records"])
        assert all(record["model_position"] >= 0 for record in token_detail["records"])
        linked = [record for record in token_detail["records"] if record["mtp_linkage_state"] == "available"]
        assert linked
        assert all(record["origin"] in ["mtp_accepted", "target_after_miss", "target_bonus"] for record in linked)
        assert all(record["logical_step"] >= 0 for record in linked)
        assert all(record["actual_target_pass"] >= 0 for record in linked)
        assert all(record["proposal_position"] >= 0 for record in linked)
        assert all(record["accepted_depth"] <= record["proposed_count"] for record in linked)
        assert all(record["proposal_position"] <= record["proposed_count"] for record in linked)

        assert token_detail["mtp_pass_state"] in ["available", "truncated"]
        assert token_detail["mtp_passes_total"] == speculative["actual_target_passes"]
        assert token_detail["mtp_passes_captured"] == len(token_detail["mtp_pass_records"])
        assert token_detail["mtp_passes_dropped"] == (
            token_detail["mtp_passes_total"] - token_detail["mtp_passes_captured"]
        )
        assert token_detail["mtp_proposals_total"] == speculative["draft_tokens"]
        assert token_detail["mtp_proposals_captured"] <= token_detail["maximum_captured_mtp_proposals"]
        assert token_detail["mtp_proposals_dropped"] == (
            token_detail["mtp_proposals_total"] - token_detail["mtp_proposals_captured"]
        )
        assert token_detail["mtp_proposal_state"] in ["available", "truncated"]
        passes = token_detail["mtp_pass_records"]
        assert [record["actual_target_pass"] for record in passes] == list(range(len(passes)))
        assert all(record["accepted_depth"] <= record["proposed_count"] for record in passes)
        assert all(
            record["reached_rejected_tokens"] + record["invalidated_tokens"]
            == record["proposed_count"] - record["accepted_depth"]
            for record in passes
        )
        assert all(record["target_rows_evaluated"] >= 1 for record in passes)
        assert all(record["committed_token_count"] == 0 for record in passes if record["discarded"])
        assert all(
            record["committed_output_start_ordinal"] is None
            for record in passes
            if record["committed_token_count"] == 0
        )
        captured_logical_passes = sum(record["counts_as_logical_step"] for record in passes)
        captured_target_rows = sum(record["target_rows_evaluated"] for record in passes)
        captured_committed_tokens = sum(record["committed_token_count"] for record in passes)
        logical_passes = [record for record in passes if record["counts_as_logical_step"]]
        captured_proposed = sum(record["proposed_count"] for record in logical_passes)
        captured_accepted = sum(record["accepted_depth"] for record in logical_passes)
        captured_proposal_rows = sum(len(record["proposals"]) for record in passes)
        assert captured_proposal_rows == token_detail["mtp_proposals_captured"]
        if token_detail["mtp_pass_state"] == "available":
            assert captured_logical_passes == speculative["logical_target_passes"]
            assert captured_target_rows == speculative["target_tokens"]
            assert captured_committed_tokens == speculative["useful_output_tokens"]
            assert captured_proposed == speculative["draft_tokens"]
            assert captured_accepted == speculative["accepted_tokens"]
        else:
            assert token_detail["mtp_passes_dropped"] > 0
            assert captured_logical_passes <= speculative["logical_target_passes"]
            assert captured_target_rows <= speculative["target_tokens"]
            assert captured_committed_tokens <= speculative["useful_output_tokens"]
            assert captured_proposed <= speculative["draft_tokens"]
            assert captured_accepted <= speculative["accepted_tokens"]
        for record in passes:
            if record["replay_pass"]:
                assert record["replay_of_actual_target_pass"] == record["actual_target_pass"] - 1
            else:
                assert record["replay_of_actual_target_pass"] is None
            if record["committed_token_count"] > 0:
                start = record["committed_output_start_ordinal"]
                assert start is not None
                assert start + record["committed_token_count"] <= token_detail["total_committed_tokens"]
            if record["discarded"]:
                assert record["proposal_state"] == "not_applicable"
                assert record["proposals"] == []
                continue
            assert record["proposal_state"] in ["available", "truncated"]
            proposals = record["proposals"]
            assert [proposal["position"] for proposal in proposals] == list(range(len(proposals)))
            assert all(proposal["evaluated_actual_target_pass"] <= record["actual_target_pass"] for proposal in proposals)
            if record["replay_pass"]:
                assert all(
                    proposal["evaluated_actual_target_pass"] == record["replay_of_actual_target_pass"]
                    for proposal in proposals
                )
            else:
                assert all(
                    proposal["evaluated_actual_target_pass"] == record["actual_target_pass"]
                    for proposal in proposals
                )
            if content_enabled:
                assert all(isinstance(proposal["draft_token_id"], int) for proposal in proposals)
                assert all(proposal["draft_token_identity_state"] == "available" for proposal in proposals)
                assert all(proposal["draft_token_piece_state"] == "available" for proposal in proposals)
                assert all(isinstance(base64.b64decode(proposal["draft_token_piece_base64"]), bytes) for proposal in proposals)
            else:
                assert all(proposal["draft_token_id"] is None for proposal in proposals)
                assert all(proposal["draft_token_identity_state"] == "not_captured" for proposal in proposals)
                assert all(proposal["draft_token_piece_base64"] is None for proposal in proposals)
                assert all(proposal["draft_token_piece_state"] == "not_captured" for proposal in proposals)
            for proposal in proposals:
                if proposal["disposition"] == "invalidated_tail":
                    assert proposal["target_selected_token_id"] is None
                    assert proposal["target_selected_token_identity_state"] == "not_applicable"
                    assert proposal["target_selected_token_piece_base64"] is None
                    assert proposal["target_selected_token_piece_state"] == "not_applicable"
                    assert proposal["target_selected_log_probability_ln"] is None
                    assert proposal["target_selected_probability_state"] == "not_applicable"
                    assert proposal["committed_output_ordinal"] is None
                else:
                    assert proposal["disposition"] in ["accepted", "first_mismatch"]
                    if content_enabled:
                        assert isinstance(proposal["target_selected_token_id"], int)
                        assert proposal["target_selected_token_identity_state"] == "available"
                        assert proposal["target_selected_token_piece_state"] == "available"
                        assert isinstance(base64.b64decode(proposal["target_selected_token_piece_base64"]), bytes)
                    else:
                        assert proposal["target_selected_token_id"] is None
                        assert proposal["target_selected_token_identity_state"] == "not_captured"
                        assert proposal["target_selected_token_piece_base64"] is None
                        assert proposal["target_selected_token_piece_state"] == "not_captured"
                    assert proposal["target_selected_log_probability_ln"] <= 0
                    assert proposal["target_selected_probability_state"] == "available"
                    assert proposal["committed_output_ordinal"] is not None
                    assert proposal["committed_output_ordinal"] < token_detail["total_committed_tokens"]
    else:
        assert token_detail["state"] == "disabled"


def test_mtp_target_candidate_detail_is_lazy_bounded_and_independently_stateful(monkeypatch):
    monkeypatch.setenv("LLAMA_TELEMETRY_TOKEN_CANDIDATE_MAX_BYTES", "32768")
    configure_embedded_mtp_fixture(server)
    server.start()
    apply_telemetry_control(output_token_detail=True, token_candidates=True)

    capability = server.make_request("GET", "/telemetry/v1/capabilities")
    assert capability.status_code == 200
    candidate_capability = capability.body["capabilities"]["output_token_candidates"]
    assert candidate_capability["state"] == "conditional"
    assert candidate_capability["default_target_top_k"] == 5
    assert candidate_capability["draft_top_k_state"] == "unsupported"
    assert candidate_capability["normal_event_payload_contains_candidates"] is False

    missing_trace_id = server.make_request("GET", "/telemetry/v1/token-candidates")
    assert missing_trace_id.status_code == 400

    response = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "repeat:",
            "n_predict": 20,
            "n_probs": 0,
            "output_token_telemetry": True,
            "output_token_candidate_telemetry": True,
            "output_token_candidate_top_k": 5,
            "output_token_candidate_byte_cap": 8192,
            "temperature": 0,
            "grammar": 'root ::= " a a a a a a a a a a a a a a a a a a a a"',
        },
    )
    assert response.status_code == 200
    trace_id = response.body["trace_id"]
    event = completed_event(trace_id)
    summary = event["output_token_telemetry"]
    assert summary["candidate_detail_key"] == trace_id
    assert summary["candidate_detail_state"] in ["available", "truncated"]
    assert summary["candidate_detail_eligible_decisions"] > 0
    assert summary["candidate_detail_stored_bytes"] <= 8192
    assert "decisions" not in summary
    assert "target_candidates" not in json.dumps(summary)
    # Candidate scoring is a separate explicit diagnostic; it must not silently
    # relabel the request-level response probability population as enabled.
    assert summary["probability_state"] == "not_enabled_for_request"

    detail_response = server.make_request(
        "GET",
        f"/telemetry/v1/token-candidates?trace_id={trace_id}",
    )
    assert detail_response.status_code == 200
    detail = detail_response.body
    assert detail["schema_version"] == 2
    assert detail["trace_id"] == trace_id
    assert detail["state"] == summary["candidate_detail_state"]
    assert detail["storage_kind"] == "external_bounded_detail_block"
    assert detail["target_top_k"] == 5
    assert detail["include_accepted_positions"] is False
    assert detail["draft_candidates_state"] == "unsupported"
    assert detail["token_piece_state"] == "not_captured"
    assert detail["stored_bytes"] == summary["candidate_detail_stored_bytes"]
    assert detail["stored_bytes"] <= detail["byte_cap"] == 8192
    assert detail["eligible_decisions"] == summary["candidate_detail_eligible_decisions"]
    assert detail["dropped_decisions"] == summary["candidate_detail_dropped_decisions"]
    assert detail["captured_decisions"] == len(detail["decisions"])
    assert detail["captured_decisions"] + detail["dropped_decisions"] == detail["eligible_decisions"]
    assert all(
        decision["decision_kind"] in ["first_mismatch", "target_bonus"]
        for decision in detail["decisions"]
    )
    assert all(decision["target_probability_state"] == "available" for decision in detail["decisions"])
    assert all(1 <= len(decision["target_candidates"]) <= 5 for decision in detail["decisions"])
    assert all(
        [candidate["rank"] for candidate in decision["target_candidates"]]
        == list(range(1, len(decision["target_candidates"]) + 1))
        for decision in detail["decisions"]
    )
    assert all(
        candidate["log_probability_ln"] <= 0
        and candidate["probability_state"] == "available"
        and candidate["token_id"] is None
        and candidate["token_identity_state"] == "not_captured"
        and candidate["token_piece_base64"] is None
        and candidate["token_piece_state"] == "not_captured"
        for decision in detail["decisions"]
        for candidate in decision["target_candidates"]
    )

    unknown = server.make_request(
        "GET",
        "/telemetry/v1/token-candidates?trace_id=unknown-exact-trace",
    )
    assert unknown.status_code == 200
    assert unknown.body["schema_version"] == 2
    assert unknown.body["state"] == "not_captured"
    assert unknown.body["token_identity_state"] == "not_captured"
    assert unknown.body["token_piece_state"] == "not_captured"
    assert unknown.body["decisions"] == []

    not_enabled_response = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "repeat:",
            "n_predict": 8,
            "output_token_telemetry": True,
            "output_token_candidate_telemetry": False,
            "temperature": 0,
            "grammar": 'root ::= " a a a a a a a a"',
        },
    )
    assert not_enabled_response.status_code == 200
    not_enabled_trace_id = not_enabled_response.body["trace_id"]
    not_enabled_summary = completed_event(not_enabled_trace_id)["output_token_telemetry"]
    assert not_enabled_summary["candidate_detail_state"] == "not_enabled_for_request"
    assert "output_token_candidate_telemetry=true" in not_enabled_summary["candidate_detail_reason"]
    not_enabled_detail = server.make_request(
        "GET",
        f"/telemetry/v1/token-candidates?trace_id={not_enabled_trace_id}",
    )
    assert not_enabled_detail.status_code == 200
    assert not_enabled_detail.body["schema_version"] == 2
    assert not_enabled_detail.body["state"] == "not_captured"
    assert not_enabled_detail.body["token_identity_state"] == "not_captured"
    assert not_enabled_detail.body["token_piece_state"] == "not_captured"
    assert not_enabled_detail.body["decisions"] == []


def test_mtp_target_candidate_content_identity_and_accepted_position_gate(monkeypatch):
    configure_embedded_mtp_fixture(server)
    server.start()
    apply_telemetry_control(
        output_token_detail=True,
        token_candidates=True,
        request_content=True,
    )

    response = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "repeat:",
            "n_predict": 20,
            "n_probs": 0,
            "output_token_telemetry": True,
            "output_token_candidate_telemetry": True,
            "output_token_candidate_include_accepted": True,
            "output_token_candidate_top_k": 5,
            "output_token_candidate_byte_cap": 32768,
            "temperature": 0,
            "grammar": 'root ::= " a a a a a a a a a a a a a a a a a a a a"',
        },
    )
    assert response.status_code == 200
    trace_id = response.body["trace_id"]
    detail_response = server.make_request(
        "GET",
        f"/telemetry/v1/token-candidates?trace_id={trace_id}",
    )
    assert detail_response.status_code == 200
    detail = detail_response.body
    assert detail["state"] in ["available", "truncated"]
    assert detail["include_accepted_positions"] is True
    assert detail["token_identity_state"] == "available"
    assert detail["token_piece_state"] == "available"
    assert detail["captured_decisions"] > 0
    assert all(
        isinstance(candidate["token_id"], int)
        and candidate["token_identity_state"] == "available"
        and isinstance(candidate["token_piece_base64"], str)
        and candidate["token_piece_state"] == "available"
        and isinstance(base64.b64decode(candidate["token_piece_base64"]), bytes)
        and candidate["probability_state"] == "available"
        for decision in detail["decisions"]
        for candidate in decision["target_candidates"]
    )
    assert all(
        decision["decision_kind"] in ["accepted_draft_position", "first_mismatch", "target_bonus"]
        for decision in detail["decisions"]
    )


def test_token_candidate_bounded_ring_reports_expired_trace(monkeypatch):
    server.spec_type = "ngram-simple"
    server.spec_ngram_simple_size_n = 2
    server.spec_ngram_simple_size_m = 3
    server.spec_ngram_simple_min_hits = 1
    server.start()
    apply_telemetry_control(output_token_detail=True, token_candidates=True)

    first_trace_id = None
    latest_trace_id = None
    for index in range(257):
        response = server.make_request(
            "POST",
            "/completion",
            data={
                "prompt": f"bounded candidate retention {index}",
                "n_predict": 1,
                "output_token_telemetry": True,
                "output_token_candidate_telemetry": True,
                "temperature": 0,
            },
        )
        assert response.status_code == 200
        latest_trace_id = response.body["trace_id"]
        if first_trace_id is None:
            first_trace_id = latest_trace_id

    expired = server.make_request(
        "GET",
        f"/telemetry/v1/token-candidates?trace_id={first_trace_id}",
    )
    assert expired.status_code == 200
    assert expired.body["schema_version"] == 2
    assert expired.body["state"] == "expired"
    assert expired.body["token_identity_state"] == "expired"
    assert expired.body["token_piece_state"] == "expired"
    assert expired.body["decisions"] == []

    latest = server.make_request(
        "GET",
        f"/telemetry/v1/token-candidates?trace_id={latest_trace_id}",
    )
    assert latest.status_code == 200
    assert latest.body["state"] in ["no_data", "available", "truncated"]


def test_mtp_target_candidate_requires_global_control():
    configure_embedded_mtp_fixture(server)
    server.start()
    apply_telemetry_control(output_token_detail=True)

    capability = server.make_request("GET", "/telemetry/v1/capabilities")
    candidate_capability = capability.body["capabilities"]["output_token_candidates"]
    assert candidate_capability["state"] == "conditional"
    assert "telemetry_control.token_candidates=true" in candidate_capability["enable_with"]

    response = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "repeat:",
            "n_predict": 8,
            "output_token_telemetry": True,
            "output_token_candidate_telemetry": True,
            "temperature": 0,
            "grammar": 'root ::= " a a a a a a a a"',
        },
    )
    assert response.status_code == 200
    trace_id = response.body["trace_id"]
    summary = completed_event(trace_id)["output_token_telemetry"]
    assert summary["candidate_detail_state"] == "not_enabled_for_request"
    assert "output_token_candidate_telemetry=true" in summary["candidate_detail_reason"]
    detail = server.make_request(
        "GET",
        f"/telemetry/v1/token-candidates?trace_id={trace_id}",
    )
    assert detail.status_code == 200
    assert detail.body["schema_version"] == 2
    assert detail.body["state"] == "not_captured"
    assert detail.body["token_identity_state"] == "not_captured"
    assert detail.body["token_piece_state"] == "not_captured"
    assert detail.body["decisions"] == []
