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
    control_capability = capabilities.body["capabilities"]["telemetry_control"]
    assert control_capability["endpoint"] == "POST /props"
    assert control_capability["full_replacement"] is True
    assert control_capability["environment_cannot_enable_global"] is True

    initial = _snapshot(server)
    assert initial["generation"] == 0
    assert all(value is False for value in initial["effective"].values())
    assert initial["effective_from"]["moe_routing"] == "next_microbatch"
    assert initial["effective_from"]["output_token_detail"] == "next_request"

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
    assert control["applicability"]["moe_routing"]["state"] == "not_applicable"

    disabled = server.make_request("POST", "/props", data={"telemetry_control": {}}, headers=AUTH)
    assert disabled.status_code == 200
    assert disabled.body["telemetry_control"]["generation"] == 2
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
