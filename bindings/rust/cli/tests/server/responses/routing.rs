use crate::server::common::{
    model_config, post_json, require_model, send_request, ServerConfig, ServerTestContext,
    TenantSpec,
};

#[test]
fn responses_route_is_mounted() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "previous_response_id": "resp_test"
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_ne!(resp.status, 404, "body: {}", resp.body);
}

#[test]
fn responses_rejects_non_spec_fields() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "input": "hello",
        "session_id": "chat_only_field"
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn responses_alias_is_not_exposed() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "input": "hello"
    });
    let resp = post_json(ctx.addr(), "/responses", &body);
    assert_eq!(resp.status, 404, "body: {}", resp.body);
}

#[test]
fn responses_rejects_invalid_input_shape() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "input": { "bad": "shape" }
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(
        json["error"]["type"].as_str(),
        Some("invalid_request_error")
    );
    assert!(json["error"]["code"].is_string());
    assert!(json["error"]["message"].is_string());
    assert!(json["error"]["param"].is_null());
}

#[test]
fn responses_rejects_invalid_tool_choice() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "input": "hello",
        "tool_choice": "sometimes"
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn responses_accepts_valid_allowed_tools_tool_choice_shape() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "input": "hello",
        "tool_choice": {
            "type": "allowed_tools",
            "tools": [
                { "type": "function", "name": "lookup" }
            ],
            "mode": "auto"
        }
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_ne!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn responses_accepts_spec_valid_engine_unsupported_fields() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "input": "hello",
        "parallel_tool_calls": true,
        "background": true,
        "service_tier": "default",
        "prompt_cache_key": "cache-key-1"
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_ne!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn responses_server_error_envelope_includes_type_and_param() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "model": "model-definitely-not-available",
        "input": "hello"
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 500, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["error"]["type"].as_str(), Some("server_error"));
    assert!(json["error"]["code"].is_string());
    assert!(json["error"]["message"].is_string());
    assert!(json["error"]["param"].is_null());
}

#[test]
fn responses_input_array_accepts_all_itemparam_variants() {
    let model = require_model!();
    let mut cfg = model_config();
    cfg.model = Some(model.clone());
    let ctx = ServerTestContext::new(cfg);
    let body = serde_json::json!({
        "model": model,
        "max_output_tokens": 8,
        "input": [
            { "type": "message", "role": "user", "content": "hello" },
            { "type": "message", "role": "assistant", "content": [ { "type": "output_text", "text": "prior" } ] },
            { "type": "message", "role": "system", "content": "system note" },
            { "type": "message", "role": "developer", "content": "developer note" },
            { "type": "reasoning", "summary": [ { "type": "summary_text", "text": "reasoning summary" } ] },
            { "type": "function_call", "call_id": "call_123", "name": "lookup", "arguments": "{}" },
            { "type": "function_call_output", "call_id": "call_123", "output": "{\"ok\":true}" },
            { "type": "item_reference", "id": "msg_abc" }
        ]
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_ne!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn responses_rejects_invalid_itemparam_in_input_array() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "input": [
            { "type": "message", "role": "user", "content": "hello" },
            { "type": "function_call", "name": "lookup" }
        ]
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(
        json["error"]["type"].as_str(),
        Some("invalid_request_error")
    );
}

#[test]
fn responses_store_false_still_allows_previous_response_id_chaining_in_process() {
    let model = require_model!();
    let mut cfg = model_config();
    cfg.model = Some(model.clone());
    let ctx = ServerTestContext::new(cfg);

    let first = serde_json::json!({
        "model": model,
        "input": "hello",
        "max_output_tokens": 8,
        "store": false,
        "tools": [{
            "type": "function",
            "name": "lookup",
            "description": "lookup",
            "parameters": { "type": "object", "properties": {} }
        }]
    });
    let first_resp = post_json(ctx.addr(), "/v1/responses", &first);
    assert_eq!(first_resp.status, 200, "body: {}", first_resp.body);
    let first_json = first_resp.json();
    let prev_id = first_json["id"].as_str().expect("response id");

    let second = serde_json::json!({
        "model": model,
        "input": "follow-up",
        "max_output_tokens": 8,
        "store": false,
        "previous_response_id": prev_id
    });
    let second_resp = post_json(ctx.addr(), "/v1/responses", &second);
    assert_eq!(second_resp.status, 200, "body: {}", second_resp.body);
    let second_json = second_resp.json();
    assert_eq!(
        second_json["tools"].as_array().map(|v| v.len()),
        Some(1),
        "same-process chaining should resolve previous_response_id even when store=false"
    );
}

#[test]
fn responses_store_flag_round_trips_when_model_is_available() {
    let model = require_model!();
    let mut cfg = model_config();
    cfg.model = Some(model.clone());
    let ctx = ServerTestContext::new(cfg);
    let body = serde_json::json!({
        "model": model,
        "input": "hello",
        "max_output_tokens": 8,
        "store": true
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["store"].as_bool(), Some(true));
    assert!(
        json["metadata"]
            .as_object()
            .map(|m| !m.contains_key("session_id"))
            .unwrap_or(false),
        "strict responses should not inject chat session metadata: {}",
        resp.body
    );
}

#[test]
fn responses_instructions_round_trip_when_model_is_available() {
    let model = require_model!();
    let mut cfg = model_config();
    cfg.model = Some(model.clone());
    let ctx = ServerTestContext::new(cfg);
    let body = serde_json::json!({
        "model": model,
        "input": "hello",
        "instructions": "answer in one sentence",
        "max_output_tokens": 8
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(
        json["instructions"].as_str(),
        Some("answer in one sentence"),
        "response resource must preserve instructions field"
    );
}

#[test]
fn responses_cross_tenant_previous_response_id_does_not_inherit_tools() {
    let model = require_model!();
    let temp = tempfile::TempDir::new().expect("temp dir");
    let bucket = temp.path().join("bucket");
    std::fs::create_dir_all(&bucket).expect("create bucket");

    let mut config = ServerConfig::new();
    config.model = Some(model.clone());
    config.bucket = Some(bucket);
    config.gateway_secret = Some("secret".to_string());
    config.tenants = vec![
        TenantSpec {
            id: "alpha".to_string(),
            storage_prefix: "alpha".to_string(),
            allowed_models: vec![],
        },
        TenantSpec {
            id: "beta".to_string(),
            storage_prefix: "beta".to_string(),
            allowed_models: vec![],
        },
    ];
    let ctx = ServerTestContext::new(config);

    let alpha_body = serde_json::json!({
        "model": &model,
        "input": "Hello from alpha",
        "max_output_tokens": 10,
        "tools": [{
            "type": "function",
            "name": "alpha_lookup",
            "description": "alpha only",
            "parameters": { "type": "object", "properties": {} }
        }]
    });
    let alpha_body_str = serde_json::to_string(&alpha_body).unwrap();
    let alpha_resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/responses",
        &[
            ("Content-Type", "application/json"),
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "alpha"),
        ],
        Some(&alpha_body_str),
    );
    assert_eq!(
        alpha_resp.status, 200,
        "alpha request failed: {}",
        alpha_resp.body
    );
    let alpha_json = alpha_resp.json();
    let alpha_response_id = alpha_json["id"].as_str().expect("alpha should have id");
    assert_eq!(
        alpha_json["tools"].as_array().map(|v| v.len()),
        Some(1),
        "alpha should persist request tools"
    );

    let beta_body = serde_json::json!({
        "model": &model,
        "input": "Hello from beta",
        "max_output_tokens": 10,
        "previous_response_id": alpha_response_id
    });
    let beta_body_str = serde_json::to_string(&beta_body).unwrap();
    let beta_resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/responses",
        &[
            ("Content-Type", "application/json"),
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "beta"),
        ],
        Some(&beta_body_str),
    );
    assert_eq!(
        beta_resp.status, 200,
        "beta request failed: {}",
        beta_resp.body
    );
    let beta_json = beta_resp.json();
    assert_eq!(
        beta_json["tools"].as_array().map(|v| v.len()),
        Some(0),
        "cross-tenant previous_response_id must not inherit tools"
    );
}
