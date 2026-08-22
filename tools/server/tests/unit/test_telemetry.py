import math

import pytest

from utils import ServerPreset


server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_metrics = True


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


def test_telemetry_lifecycle_cache_and_cursor():
    server.n_slots = 1
    server.start()

    capabilities = server.make_request("GET", "/telemetry/v1/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.body["schema_version"] == 1
    assert capabilities.body["capabilities"]["request_lifecycle"]["state"] == "available"

    invalid_cursor = server.make_request("GET", "/telemetry/v1/events?cursor=-1")
    assert invalid_cursor.status_code == 400

    prompt = "the quick brown fox jumps over the lazy dog"
    first = server.make_request("POST", "/completion", data={"prompt": prompt, "n_predict": 2})
    second = server.make_request("POST", "/completion", data={"prompt": prompt, "n_predict": 2})
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
    assert first_token["matched_prefix_tokens"] == first_token["prompt_tokens"]
    assert first_token["cache_status"] == "full"
    assert first_token["prefill_meaningful"] is False
    assert first_token["cache_lookup_ms"] >= 0
    assert first_token["actual_prefill_ms"] >= 0
    assert first_token["server_configuration"] == started["server_configuration"]

    event = completed_event(second.body["trace_id"])
    assert event["prompt_tokens"] == event["reused_prompt_tokens"] + event["evaluated_prompt_tokens"]
    assert event["cache_status"] == "full"
    assert event["matched_prefix_tokens"] == event["prompt_tokens"]
    assert event["timings"]["ttft_ms"] >= event["timings"]["queue_ms"] >= 0
    assert event["timings"]["e2e_ms"] >= event["timings"]["ttft_ms"]


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


def test_prometheus_histograms_are_cumulative():
    server.start()
    server.make_request("POST", "/completion", data={"prompt": "histogram", "n_predict": 2})
    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    lines = metrics.body.splitlines()
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
    assert live["capacity_entries"] >= live["used_entries"] > 0
    assert live["free_entries"] == live["capacity_entries"] - live["used_entries"]
    assert snapshot.body["kv"]["physical_prefix_sharing"]["state"] == "available"
    assert snapshot.body["kv"]["churn"]["defragmentation"]["state"] == "not_applicable"

    kv = server.make_request("GET", "/telemetry/v1/kv")
    assert kv.status_code == 200
    assert kv.body["allocated"]["total_bytes"] > 0
    assert kv.body["live_occupancy"]["state"] == "available"
    assert kv.body["physical_prefix_sharing"]["state"] == "available"
    assert kv.body["duplicate_prefix_opportunities"]["state"] == "available"
    assert len(kv.body["components"]) > 0

    capabilities = server.make_request("GET", "/telemetry/v1/capabilities").body["capabilities"]
    assert capabilities["physical_ubatch_observed"]["state"] == "available"
    assert capabilities["kv_live_occupancy"]["state"] == "available"
    assert capabilities["physical_prefix_sharing"]["state"] == "available"
    assert capabilities["duplicate_prefix_opportunities"]["state"] == "available"
    assert capabilities["response_perplexity"]["state"] == "conditional"
    assert capabilities["prompt_perplexity"]["state"] == "conditional"


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

    disabled = server.make_request(
        "POST",
        "/completion",
        data={"prompt": "prompt perplexity disabled", "n_predict": 1},
    )
    assert disabled.status_code == 200
    disabled_probability = completed_event(disabled.body["trace_id"])["prompt_probability"]
    assert disabled_probability == {
        "state": "disabled",
        "available": False,
        "reason": "request_did_not_enable_prompt_perplexity",
    }


def test_speculative_invariants_and_ttft():
    server.spec_type = "ngram-simple"
    server.spec_ngram_simple_size_n = 2
    server.spec_ngram_simple_size_m = 3
    server.spec_ngram_simple_min_hits = 1
    server.start()
    response = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "repeat:",
            "n_predict": 20,
            "n_probs": 1,
            "temperature": 0,
            "grammar": 'root ::= " a a a a a a a a a a a a a a a a a a a a"',
        },
    )
    assert response.status_code == 200
    event = completed_event(response.body["trace_id"])
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
    assert speculative["configuration"]["ngram_simple_size_n"] == 2
    assert speculative["configuration"]["ngram_simple_size_m"] == 3
    assert event["sampling"]["requested_temperature"] == 0
    assert event["sampling"]["effective_temperature"] == 0
    assert event["sampling"]["effective_seed"] >= 0
    assert event["sampling"]["grammar"].startswith("root ::=")
    probability = event["response_probability"]
    assert probability["available"] is True
    assert probability["semantics"] == "raw_target_model_pre_sampler_selected_token_probability"
    assert probability["scored_tokens"] == event["output_tokens"]
    assert probability["perplexity"] == pytest.approx(math.exp(probability["mean_nll"]))
