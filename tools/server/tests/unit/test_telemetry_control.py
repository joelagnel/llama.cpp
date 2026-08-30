import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from utils import ServerPreset


API_KEY = "telemetry-control-test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}
CONTROL_NAMES = (
    "moe_routing",
    "output_token_detail",
    "token_candidates",
    "prompt_perplexity",
    "request_content",
    "kv_pressure_detail",
    "native_gpu_gpm",
)


def _controlled_server():
    server = ServerPreset.tinyllama2()
    server.server_props = True
    server.api_key = API_KEY
    return server


def _snapshot(server):
    response = server.make_request("GET", "/telemetry/v1/snapshot", headers=AUTH)
    assert response.status_code == 200
    return response.body["telemetry_control"]


def _kv_pressure(server):
    response = server.make_request(
        "GET", "/telemetry/v1/kv-pressure?cursor=0&limit=4096", headers=AUTH
    )
    assert response.status_code == 200
    return response.body


def _completed_event(server, trace_id):
    response = server.make_request("GET", "/telemetry/v1/events?cursor=0&limit=100", headers=AUTH)
    assert response.status_code == 200
    return next(
        event for event in response.body["events"]
        if event["trace_id"] == trace_id and event["event"] == "request_completed"
    )


def _telemetry_events(server, limit=512):
    response = server.make_request(
        "GET", f"/telemetry/v1/events?cursor=0&limit={limit}", headers=AUTH
    )
    assert response.status_code == 200
    return response.body["events"]


def test_props_telemetry_control_replaces_state_and_resets_on_restart():
    server = _controlled_server()
    server.start()

    capabilities = server.make_request("GET", "/telemetry/v1/capabilities", headers=AUTH)
    assert capabilities.status_code == 200
    control_capability = capabilities.body["telemetry_control"]
    assert control_capability["supported"] is True
    assert control_capability["route"] == "/props"
    assert control_capability["method"] == "POST"
    assert control_capability["requires_props"] is True
    assert control_capability["requires_authentication"] is True
    assert control_capability["requires_loopback"] is True
    assert control_capability["replacement_semantics"] == "full"
    assert control_capability["features"]["moe_routing"]["effective_from"] == "next_microbatch"
    assert control_capability["features"]["token_candidates"]["dependencies"] == ["output_token_detail"]

    initial = _snapshot(server)
    assert initial["generation"] == 0
    assert set(initial["effective"]) == set(CONTROL_NAMES)
    assert all(value is False for value in initial["effective"].values())
    assert "effective_from" not in initial

    enabled = server.make_request(
        "POST",
        "/props",
        data={"telemetry_control": {"output_token_detail": True, "request_content": True}},
        headers=AUTH,
    )
    assert enabled.status_code == 200
    control = enabled.body["telemetry_control"]
    assert control["generation"] == 1
    assert control["effective"]["output_token_detail"] is True
    assert control["effective"]["request_content"] is True
    assert control["effective"] == {
        name: name in {"output_token_detail", "request_content"}
        for name in CONTROL_NAMES
    }
    assert control["effective_from"] == "next_request"
    assert control["applicability"]["moe_routing"]["applicable"] is False

    full = server.make_request(
        "POST",
        "/props",
        data={"telemetry_control": {name: True for name in CONTROL_NAMES}},
        headers=AUTH,
    )
    assert full.status_code == 200
    assert full.body["telemetry_control"]["generation"] == 2
    assert all(full.body["telemetry_control"]["effective"].values())
    assert full.body["telemetry_control"]["effective_from"] == "next_microbatch"

    disabled = server.make_request("POST", "/props", data={"telemetry_control": {}}, headers=AUTH)
    assert disabled.status_code == 200
    assert disabled.body["telemetry_control"]["generation"] == 3
    assert disabled.body["telemetry_control"]["effective_from"] == "next_microbatch"
    assert all(value is False for value in disabled.body["telemetry_control"]["effective"].values())

    server.stop()
    server.start()
    reset = _snapshot(server)
    assert reset["generation"] == 0
    assert all(value is False for value in reset["effective"].values())


def test_request_and_environment_values_cannot_enable_global_control():
    server = _controlled_server()
    server.extra_env = {
        "LLAMA_TELEMETRY_OUTPUT_TOKENS": "1",
        "LLAMA_TELEMETRY_CONTENT": "1",
        "LLAMA_TELEMETRY_TOKEN_CANDIDATES": "1",
        "LLAMA_TELEMETRY_MOE_ROUTING": "1",
    }
    server.start()

    assert all(value is False for value in _snapshot(server)["effective"].values())
    response = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "telemetry control request opt-in",
            "n_predict": 1,
            "output_token_telemetry": True,
            "output_token_candidate_telemetry": True,
            "prompt_perplexity": True,
            "request_content": True,
        },
        headers=AUTH,
    )
    assert response.status_code == 200
    detail = _completed_event(server, response.body["trace_id"])["output_token_telemetry"]
    assert detail["state"] == "not_enabled_for_request"

    enabled = server.make_request(
        "POST",
        "/props",
        data={
            "telemetry_control": {
                "output_token_detail": True,
                "token_candidates": True,
                "prompt_perplexity": True,
                "request_content": True,
            },
        },
        headers=AUTH,
    )
    assert enabled.status_code == 200

    opted_out = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "explicit false remains an opt out",
            "n_predict": 1,
            "output_token_telemetry": False,
            "output_token_candidate_telemetry": False,
            "prompt_perplexity": False,
            "request_content": False,
        },
        headers=AUTH,
    )
    assert opted_out.status_code == 200
    event = _completed_event(server, opted_out.body["trace_id"])
    assert event["output_token_telemetry"]["state"] == "not_enabled_for_request"
    assert event["prompt_probability"] == {
        "state": "disabled",
        "available": False,
        "reason": "request_did_not_enable_prompt_perplexity",
    }
    assert event["request"] == {
        "content_omitted": True,
        "reason": "content_logging_disabled",
    }
    assert "response" not in event


def test_dense_model_never_emits_moe_routing_chunks():
    server = _controlled_server()
    server.start()
    try:
        control = server.make_request(
            "POST",
            "/props",
            data={"telemetry_control": {"moe_routing": True}},
            headers=AUTH,
        )
        assert control.status_code == 200
        assert control.body["telemetry_control"]["applicability"]["moe_routing"]["applicable"] is False

        completion = server.make_request(
            "POST",
            "/completion",
            data={"prompt": "dense telemetry remains available", "n_predict": 1},
            headers=AUTH,
        )
        assert completion.status_code == 200
        events = server.make_request(
            "GET", "/telemetry/v1/events?cursor=0&limit=100", headers=AUTH
        )
        assert events.status_code == 200
        assert not any(event["event"] == "moe_routing_chunk" for event in events.body["events"])
    finally:
        server.stop()


def test_output_token_telemetry_preserves_requested_probability_mode():
    server = _controlled_server()
    server.start()

    globally_disabled = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "globally disabled telemetry probability mode",
            "n_predict": 1,
            "ignore_eos": True,
            "output_token_telemetry": True,
        },
        headers=AUTH,
    )
    assert globally_disabled.status_code == 200
    assert "completion_probabilities" not in globally_disabled.body
    disabled_event = _completed_event(server, globally_disabled.body["trace_id"])
    assert disabled_event["response_probability"]["state"] == "disabled"
    assert disabled_event["output_token_telemetry"]["state"] == "not_enabled_for_request"

    enabled = server.make_request(
        "POST",
        "/props",
        data={"telemetry_control": {"output_token_detail": True}},
        headers=AUTH,
    )
    assert enabled.status_code == 200

    without_n_probs = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "enabled telemetry without probabilities",
            "n_predict": 1,
            "ignore_eos": True,
            "output_token_telemetry": True,
        },
        headers=AUTH,
    )
    assert without_n_probs.status_code == 200
    assert "completion_probabilities" not in without_n_probs.body
    without_n_probs_event = _completed_event(server, without_n_probs.body["trace_id"])
    assert without_n_probs_event["response_probability"]["state"] == "disabled"
    assert without_n_probs_event["output_token_telemetry"]["state"] == "available"
    assert without_n_probs_event["output_token_telemetry"]["probability_state"] == "not_enabled_for_request"

    requested = server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": "explicitly requested probabilities",
            "n_predict": 1,
            "n_probs": 2,
            "ignore_eos": True,
            "output_token_telemetry": True,
        },
        headers=AUTH,
    )
    assert requested.status_code == 200
    assert len(requested.body["completion_probabilities"]) == 1
    assert len(requested.body["completion_probabilities"][0]["top_logprobs"]) == 2
    requested_event = _completed_event(server, requested.body["trace_id"])
    assert requested_event["response_probability"]["state"] == "available"
    assert requested_event["output_token_telemetry"]["probability_state"] == "available"


def test_kv_pressure_control_keeps_the_disabled_path_empty_and_records_when_enabled():
    server = _controlled_server()
    server.start()

    disabled_response = server.make_request(
        "POST",
        "/completion",
        data={"prompt": "KV pressure stays disabled", "n_predict": 2, "temperature": 0},
        headers=AUTH,
    )
    assert disabled_response.status_code == 200
    disabled = _kv_pressure(server)
    assert disabled["events"] == []
    assert disabled["oldest_sequence"] == disabled["next_sequence"] == 1
    assert disabled["dropped_events"] == 0
    assert disabled["last_dropped_sequence"] == 0
    assert disabled["retained_serialized_bytes"] == 0

    enabled_response = server.make_request(
        "POST",
        "/props",
        data={"telemetry_control": {"kv_pressure_detail": True}},
        headers=AUTH,
    )
    assert enabled_response.status_code == 200
    assert enabled_response.body["telemetry_control"]["effective"]["kv_pressure_detail"] is True
    assert enabled_response.body["telemetry_control"]["effective_from"] == "next_microbatch"

    enabled_completion = server.make_request(
        "POST",
        "/completion",
        data={"prompt": "KV pressure is enabled", "n_predict": 2, "temperature": 0},
        headers=AUTH,
    )
    assert enabled_completion.status_code == 200
    enabled = _kv_pressure(server)
    assert enabled["events"]
    assert any(event["kind"] == "utilization_sample" for event in enabled["events"])
    assert enabled["next_sequence"] > 1
    assert enabled["retained_serialized_bytes"] > 0

    disabled_again_response = server.make_request(
        "POST",
        "/props",
        data={"telemetry_control": {}},
        headers=AUTH,
    )
    assert disabled_again_response.status_code == 200
    assert disabled_again_response.body["telemetry_control"]["effective"]["kv_pressure_detail"] is False

    before_disable = _kv_pressure(server)
    after_disable_completion = server.make_request(
        "POST",
        "/completion",
        data={"prompt": "KV pressure is disabled again", "n_predict": 2, "temperature": 0},
        headers=AUTH,
    )
    assert after_disable_completion.status_code == 200
    after_disable = _kv_pressure(server)
    assert after_disable["next_sequence"] == before_disable["next_sequence"]
    assert after_disable["dropped_events"] == before_disable["dropped_events"]
    assert after_disable["retained_serialized_bytes"] == before_disable["retained_serialized_bytes"]


def test_props_requires_auth_props_api_key_and_loopback_listener():
    server = ServerPreset.router()
    server.server_props = True
    server.api_key = API_KEY
    server.start()
    unauthorized = server.make_request("POST", "/props", data={"telemetry_control": {}})
    assert unauthorized.status_code == 401
    server.stop()

    server = ServerPreset.router()
    server.api_key = API_KEY
    server.start()
    props_disabled = server.make_request("POST", "/props", data={"model": "unused", "telemetry_control": {}}, headers=AUTH)
    assert props_disabled.status_code == 501
    server.stop()

    server = ServerPreset.router()
    server.server_props = True
    server.start()
    key_missing = server.make_request("POST", "/props", data={"model": "unused", "telemetry_control": {}})
    assert key_missing.status_code == 403
    server.stop()

    server = ServerPreset.router()
    server.server_props = True
    server.api_key = API_KEY
    server.server_host = "0.0.0.0"
    server.request_host = "127.0.0.1"
    server.start()
    non_loopback = server.make_request("POST", "/props", data={"model": "unused", "telemetry_control": {}}, headers=AUTH)
    assert non_loopback.status_code == 403


def test_router_model_targeted_last_apply_isolated_between_configured_children():
    server = ServerPreset.router()
    server.server_props = True
    server.api_key = API_KEY
    server.models_max = 2
    preset = tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False)
    try:
        model_a = "telemetry-control-a"
        model_b = "telemetry-control-b"
        preset.write(
            f"[{model_a}]\n"
            "hf-repo = ggml-org/test-model-stories260K\n"
            "\n"
            f"[{model_b}]\n"
            "hf-repo = ggml-org/test-model-stories260K-infill\n"
        )
        preset.close()
        server.models_preset = preset.name
        server.start()

        first_a = server.make_request(
            "POST",
            "/props",
            data={"model": model_a, "telemetry_control": {"output_token_detail": True}},
            headers=AUTH,
            timeout=600,
        )
        assert first_a.status_code == 200
        assert first_a.body["telemetry_control"]["generation"] == 1

        first_b = server.make_request(
            "POST",
            "/props",
            data={"model": model_b, "telemetry_control": {"prompt_perplexity": True}},
            headers=AUTH,
            timeout=600,
        )
        assert first_b.status_code == 200
        assert first_b.body["telemetry_control"]["generation"] == 1

        last_a = server.make_request(
            "POST",
            "/props",
            data={"model": model_a, "telemetry_control": {"request_content": True}},
            headers=AUTH,
            timeout=600,
        )
        assert last_a.status_code == 200
        assert last_a.body["telemetry_control"]["generation"] == 2

        models = server.make_request("GET", "/models", headers=AUTH)
        assert models.status_code == 200
        statuses = {
            item["id"]: item["status"]["value"]
            for item in models.body["data"]
            if item["id"] in {model_a, model_b}
        }
        assert statuses == {model_a: "loaded", model_b: "loaded"}

        snapshots = {}
        for model in (model_a, model_b):
            response = server.make_request(
                "GET",
                f"/telemetry/v1/snapshot?model={model}",
                headers=AUTH,
                timeout=600,
            )
            assert response.status_code == 200
            snapshots[model] = response.body["telemetry_control"]

        assert snapshots[model_a]["generation"] == 2
        assert snapshots[model_a]["effective"] == {
            name: name == "request_content" for name in CONTROL_NAMES
        }
        assert snapshots[model_b]["generation"] == 1
        assert snapshots[model_b]["effective"] == {
            name: name == "prompt_perplexity" for name in CONTROL_NAMES
        }
    finally:
        server.stop()
        if not preset.closed:
            preset.close()
        os.unlink(preset.name)


def test_inflight_off_on_off_uses_recorded_microbatch_generation_and_request_snapshot():
    server = ServerPreset.stories15m_moe()
    server.server_props = True
    server.api_key = API_KEY
    server.n_batch = 4
    server.n_ubatch = 2
    server.n_threads = 1
    server.start()

    try:
        initial = _snapshot(server)
        assert initial["generation"] == 0
        assert all(value is False for value in initial["effective"].values())

        with ThreadPoolExecutor(max_workers=1) as executor:
            completion = executor.submit(
                server.make_request,
                "POST",
                "/completion",
                data={
                    "prompt": "in-flight telemetry boundary " * 96,
                    "n_predict": 24,
                    "ignore_eos": True,
                    "temperature": 0,
                    "moe_routing_telemetry": True,
                    "output_token_telemetry": True,
                },
                headers=AUTH,
            )

            deadline = time.monotonic() + 30
            request_active = False
            while time.monotonic() < deadline and not completion.done():
                snapshot = server.make_request(
                    "GET", "/telemetry/v1/snapshot", headers=AUTH
                )
                assert snapshot.status_code == 200
                if snapshot.body["requests"]["active"] > 0:
                    request_active = True
                    break
                time.sleep(0.01)
            if not request_active:
                response = completion.result(timeout=60)
                assert response.status_code == 200
                pytest.skip("MoE fixture completed before an in-flight control boundary was observable")

            enabled = server.make_request(
                "POST",
                "/props",
                data={
                    "telemetry_control": {
                        "moe_routing": True,
                        "output_token_detail": True,
                    },
                },
                headers=AUTH,
            )
            assert enabled.status_code == 200
            assert enabled.body["telemetry_control"]["generation"] == 1
            assert enabled.body["telemetry_control"]["effective"]["moe_routing"] is True
            assert enabled.body["telemetry_control"]["effective"]["output_token_detail"] is True
            assert enabled.body["telemetry_control"]["effective_from"] == "next_microbatch"

            deadline = time.monotonic() + 30
            generation_one_recorded = False
            while time.monotonic() < deadline and not completion.done():
                generation_one_recorded = any(
                    decision["control_generation"] == 1
                    for event in _telemetry_events(server)
                    if event["event"] == "moe_routing_chunk" and not event["final"]
                    for decision in event["decisions"]
                )
                if generation_one_recorded:
                    break
                time.sleep(0.01)
            if not generation_one_recorded:
                response = completion.result(timeout=60)
                assert response.status_code == 200
                pytest.skip("MoE fixture completed before control generation 1 was recorded")
            if completion.done():
                response = completion.result(timeout=60)
                assert response.status_code == 200
                pytest.skip("MoE fixture completed before the disabling boundary could be applied")

            disabled = server.make_request(
                "POST", "/props", data={"telemetry_control": {}}, headers=AUTH
            )
            assert disabled.status_code == 200
            assert disabled.body["telemetry_control"]["generation"] == 2
            assert disabled.body["telemetry_control"]["effective_from"] == "next_microbatch"
            assert all(value is False for value in disabled.body["telemetry_control"]["effective"].values())

            response = completion.result(timeout=90)
            assert response.status_code == 200

        events = _telemetry_events(server)
        chunks = [
            event for event in events
            if event["event"] == "moe_routing_chunk" and event["trace_id"] == response.body["trace_id"]
        ]
        assert chunks[-1]["final"] is True
        assert chunks[-1]["control_generation"] == 2
        assert chunks[-1]["capture_interruption_reason"] == "telemetry_control_disabled"
        decisions = [
            decision for chunk in chunks if not chunk["final"] for decision in chunk["decisions"]
        ]
        assert decisions
        assert {decision["control_generation"] for decision in decisions} == {1}

        completed = _completed_event(server, response.body["trace_id"])
        assert completed["output_token_telemetry"]["state"] == "not_enabled_for_request"
    finally:
        server.stop()
