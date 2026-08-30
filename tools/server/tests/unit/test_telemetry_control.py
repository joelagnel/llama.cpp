from utils import ServerPreset


API_KEY = "telemetry-control-test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}


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
    assert control["effective"]["token_candidates"] is False
    assert control["effective_from"] == "next_request"
    assert control["applicability"]["moe_routing"]["applicable"] is False

    microbatch = server.make_request(
        "POST",
        "/props",
        data={"telemetry_control": {"moe_routing": True}},
        headers=AUTH,
    )
    assert microbatch.status_code == 200
    assert microbatch.body["telemetry_control"]["generation"] == 2
    assert microbatch.body["telemetry_control"]["effective_from"] == "next_microbatch"

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


def test_router_props_targets_only_the_selected_model():
    server = ServerPreset.router()
    server.server_props = True
    server.api_key = API_KEY
    server.start()

    model = "ggml-org/tinygemma3-GGUF:Q8_0"
    response = server.make_request(
        "POST",
        "/props",
        data={"model": model, "telemetry_control": {"prompt_perplexity": True}},
        headers=AUTH,
        timeout=600,
    )
    assert response.status_code == 200
    assert response.body["telemetry_control"]["effective"]["prompt_perplexity"] is True
    assert response.body["telemetry_control"]["generation"] == 1

    snapshot = server.make_request(
        "GET",
        f"/telemetry/v1/snapshot?model={model}",
        headers=AUTH,
        timeout=600,
    )
    assert snapshot.status_code == 200
    assert snapshot.body["telemetry_control"]["effective"]["prompt_perplexity"] is True
