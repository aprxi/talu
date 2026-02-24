//! `/v1/responses` chaining behavior tests.

use crate::server::common::{
    model_config, post_json, require_model, send_request, ServerConfig, ServerTestContext,
    TenantSpec,
};

fn parse_sse_events(body: &str) -> Vec<(String, serde_json::Value)> {
    let mut events = Vec::new();
    for line in body.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                continue;
            }
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                let event_type = json["type"].as_str().unwrap_or("").to_string();
                events.push((event_type, json));
            }
        }
    }
    events
}

#[test]
fn responses_chaining_unknown_id_succeeds() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": &model,
        "input": "Hello",
        "previous_response_id": "resp_nonexistent_00000000",
        "max_output_tokens": 10
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["object"].as_str(), Some("response"));
}

#[test]
fn responses_previous_response_id_without_input_succeeds() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let first = serde_json::json!({
        "model": &model,
        "input": "Remember: the answer is 42.",
        "max_output_tokens": 10
    });
    let first_resp = post_json(ctx.addr(), "/v1/responses", &first);
    assert_eq!(first_resp.status, 200, "body: {}", first_resp.body);
    let response_id = first_resp.json()["id"]
        .as_str()
        .expect("response id")
        .to_string();

    let second = serde_json::json!({
        "model": &model,
        "previous_response_id": response_id,
        "max_output_tokens": 10
    });
    let second_resp = post_json(ctx.addr(), "/v1/responses", &second);
    assert_eq!(second_resp.status, 200, "body: {}", second_resp.body);
    let second_json = second_resp.json();
    assert_eq!(
        second_json["previous_response_id"].as_str(),
        Some(response_id.as_str()),
        "chained response should echo previous_response_id"
    );
}

#[test]
fn responses_streaming_continue_no_input_emits_lifecycle() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let initial = serde_json::json!({
        "model": &model,
        "input": "Tell me about the weather.",
        "stream": true,
        "store": true,
        "max_output_tokens": 10
    });
    let initial_resp = post_json(ctx.addr(), "/v1/responses", &initial);
    assert_eq!(initial_resp.status, 200, "body: {}", initial_resp.body);
    let initial_events = parse_sse_events(&initial_resp.body);
    let terminal = initial_events
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete")
        .expect("missing terminal event in initial stream");
    let response_id = terminal.1["response"]["id"]
        .as_str()
        .expect("terminal event should include response.id");

    let continue_body = serde_json::json!({
        "model": &model,
        "previous_response_id": response_id,
        "stream": true,
        "store": true,
        "max_output_tokens": 10
    });
    let continue_resp = post_json(ctx.addr(), "/v1/responses", &continue_body);
    assert_eq!(continue_resp.status, 200, "body: {}", continue_resp.body);
    let events = parse_sse_events(&continue_resp.body);
    assert!(!events.is_empty(), "continue stream should emit events");
    assert_eq!(events[0].0, "response.queued");
    assert!(
        events.iter().any(|(t, _)| t.ends_with(".delta")),
        "continue stream should emit at least one delta event"
    );
    let last = events.last().expect("at least one event");
    assert!(
        last.0 == "response.completed" || last.0 == "response.incomplete",
        "continue stream should terminate with completed or incomplete, got {}",
        last.0
    );
    let created = events
        .iter()
        .find(|(t, _)| t == "response.created")
        .map(|(_, e)| &e["response"])
        .expect("missing response.created event");
    assert_eq!(
        created["previous_response_id"].as_str(),
        Some(response_id),
        "streamed chained response should echo previous_response_id"
    );
}

#[test]
fn responses_cross_tenant_chaining_does_not_share_response_state() {
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
    let alpha_resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/responses",
        &[
            ("Content-Type", "application/json"),
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "alpha"),
        ],
        Some(&serde_json::to_string(&alpha_body).expect("serialize alpha body")),
    );
    assert_eq!(alpha_resp.status, 200, "alpha failed: {}", alpha_resp.body);
    let alpha_json = alpha_resp.json();
    let alpha_response_id = alpha_json["id"].as_str().expect("alpha response id");
    assert_eq!(alpha_json["tools"].as_array().map(|v| v.len()), Some(1));

    let beta_body = serde_json::json!({
        "model": &model,
        "input": "Hello from beta",
        "max_output_tokens": 10,
        "previous_response_id": alpha_response_id
    });
    let beta_resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/responses",
        &[
            ("Content-Type", "application/json"),
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "beta"),
        ],
        Some(&serde_json::to_string(&beta_body).expect("serialize beta body")),
    );
    assert_eq!(beta_resp.status, 200, "beta failed: {}", beta_resp.body);
    let beta_json = beta_resp.json();
    assert_eq!(
        beta_json["tools"].as_array().map(|v| v.len()),
        Some(0),
        "cross-tenant previous_response_id must not inherit alpha tools"
    );
}
