//! Conversation chaining (previous_response_id) tests over real HTTP.
//!
//! Basic chaining (previous_response_id round-trip) is covered by the
//! in-process `api_compliance` test suite. This module tests advanced
//! chaining scenarios: unknown IDs, streaming chains, continue-generating,
//! and session metadata preservation.

use crate::server::common::{
    model_config, post_json, require_model, send_request, ServerConfig, ServerTestContext,
    TenantSpec,
};

/// Parse SSE events from a raw body string (same helper as streaming.rs).
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

/// Chaining with a nonexistent previous_response_id still succeeds
/// (server treats missing context as a fresh conversation).
#[test]
fn chaining_unknown_id_succeeds() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": &model,
        "input": "Hello",
        "previous_response_id": "resp_nonexistent_00000000",
        "max_output_tokens": 10,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["object"].as_str(), Some("response"));
}

/// Streaming chained request works.
#[test]
fn streaming_chaining() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    // First request (non-streaming).
    let body1 = serde_json::json!({
        "model": &model,
        "input": "Remember: the secret word is banana.",
        "max_output_tokens": 10,
    });
    let resp1 = post_json(ctx.addr(), "/v1/responses", &body1);
    assert_eq!(resp1.status, 200);
    let json1 = resp1.json();
    let response_id = json1["id"].as_str().expect("should have id");

    // Chained streaming request.
    let body2 = serde_json::json!({
        "model": &model,
        "input": "What is the secret word?",
        "previous_response_id": response_id,
        "stream": true,
        "max_output_tokens": 10,
    });
    let resp2 = post_json(ctx.addr(), "/v1/responses", &body2);
    assert_eq!(resp2.status, 200, "streaming chained: {}", resp2.body);
    let ct = resp2
        .header("content-type")
        .expect("should have Content-Type");
    assert!(ct.contains("text/event-stream"), "should be SSE: {ct}");

    // Should contain events.
    let has_events = resp2.body.contains("data: {");
    assert!(has_events, "streaming body should contain SSE events");
}

/// "Continue generating" — streaming request with previous_response_id
/// and no input should produce a full SSE lifecycle ending with a
/// terminal event (response.completed or response.incomplete).
#[test]
fn streaming_continue_no_input() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    // Initial stored request.
    let body1 = serde_json::json!({
        "model": &model,
        "input": "Tell me about the weather.",
        "stream": true,
        "store": true,
        "max_output_tokens": 10,
    });
    let resp1 = post_json(ctx.addr(), "/v1/responses", &body1);
    assert_eq!(resp1.status, 200, "initial: {}", resp1.body);
    let events1 = parse_sse_events(&resp1.body);
    let terminal1 = events1
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete")
        .expect("initial request should have terminal event");
    let response_id = terminal1.1["response"]["id"]
        .as_str()
        .expect("terminal should have response.id");

    // Continue request — no input, just previous_response_id.
    let body2 = serde_json::json!({
        "model": &model,
        "previous_response_id": response_id,
        "stream": true,
        "store": true,
        "max_output_tokens": 10,
    });
    let resp2 = post_json(ctx.addr(), "/v1/responses", &body2);
    assert_eq!(resp2.status, 200, "continue: {}", resp2.body);

    let events2 = parse_sse_events(&resp2.body);
    assert!(!events2.is_empty(), "continue should have SSE events");

    // Must start with response.created.
    assert_eq!(
        events2[0].0, "response.created",
        "continue should start with response.created, got: {}",
        events2[0].0,
    );

    // Must end with a terminal event.
    let last = &events2.last().unwrap().0;
    assert!(
        last == "response.completed" || last == "response.incomplete",
        "continue should end with terminal event, got: {last}",
    );

    // Terminal event should have a response with an id.
    let terminal2 = events2
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete")
        .expect("continue should have terminal event");
    let continue_id = terminal2.1["response"]["id"]
        .as_str()
        .expect("continue terminal should have response.id");
    assert!(
        !continue_id.is_empty(),
        "continue response id should not be empty"
    );

    // Should contain delta events (actual content generated).
    let has_deltas = events2.iter().any(|(t, _)| t.ends_with(".delta"));
    assert!(
        has_deltas,
        "continue should produce delta events (new content)"
    );
}

/// Continue request's terminal event carries session metadata.
#[test]
fn streaming_continue_has_session_metadata() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    // Initial stored streaming request.
    let body1 = serde_json::json!({
        "model": &model,
        "input": "Hello",
        "stream": true,
        "store": true,
        "max_output_tokens": 10,
    });
    let resp1 = post_json(ctx.addr(), "/v1/responses", &body1);
    assert_eq!(resp1.status, 200);
    let events1 = parse_sse_events(&resp1.body);
    let terminal1 = events1
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete")
        .expect("should have terminal event");
    let response_id = terminal1.1["response"]["id"]
        .as_str()
        .expect("should have id");
    let session_id = terminal1.1["response"]["metadata"]["session_id"]
        .as_str()
        .expect("initial response should have session_id in metadata");

    // Continue.
    let body2 = serde_json::json!({
        "model": &model,
        "previous_response_id": response_id,
        "stream": true,
        "store": true,
        "max_output_tokens": 10,
    });
    let resp2 = post_json(ctx.addr(), "/v1/responses", &body2);
    assert_eq!(resp2.status, 200);
    let events2 = parse_sse_events(&resp2.body);
    let terminal2 = events2
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete")
        .expect("continue should have terminal event");
    let continue_session = terminal2.1["response"]["metadata"]["session_id"]
        .as_str()
        .expect("continue should have session_id");
    assert_eq!(
        session_id, continue_session,
        "continue should preserve the same session_id",
    );
}

/// Cross-tenant `response_store` leak: the store is a global `HashMap` with no
/// tenant scoping, so Tenant B can chain off Tenant A's `response_id` and
/// inherit Tenant A's `session_id`.
///
/// If this assertion starts failing (i.e. the sessions differ or the request
/// returns an error), tenant isolation was added to `response_store` — update
/// this test to assert that cross-tenant chaining is correctly rejected.
#[test]
fn cross_tenant_chaining_inherits_session_id() {
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

    // Tenant Alpha: generate with store=true.
    let body_a = serde_json::json!({
        "model": &model,
        "input": "Hello from alpha",
        "max_output_tokens": 10,
        "store": true,
    });
    let body_a_str = serde_json::to_string(&body_a).unwrap();
    let resp_a = send_request(
        ctx.addr(),
        "POST",
        "/v1/responses",
        &[
            ("Content-Type", "application/json"),
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "alpha"),
        ],
        Some(&body_a_str),
    );
    assert_eq!(resp_a.status, 200, "alpha request failed: {}", resp_a.body);
    let json_a = resp_a.json();
    let alpha_response_id = json_a["id"].as_str().expect("alpha should have id");
    let alpha_session_id = json_a["metadata"]["session_id"]
        .as_str()
        .expect("alpha should have session_id");

    // Tenant Beta: chain off Alpha's response_id.
    let body_b = serde_json::json!({
        "model": &model,
        "input": "Hello from beta",
        "max_output_tokens": 10,
        "previous_response_id": alpha_response_id,
        "store": true,
    });
    let body_b_str = serde_json::to_string(&body_b).unwrap();
    let resp_b = send_request(
        ctx.addr(),
        "POST",
        "/v1/responses",
        &[
            ("Content-Type", "application/json"),
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "beta"),
        ],
        Some(&body_b_str),
    );
    assert_eq!(resp_b.status, 200, "beta request failed: {}", resp_b.body);
    let json_b = resp_b.json();
    let beta_session_id = json_b["metadata"]["session_id"]
        .as_str()
        .expect("beta should have session_id");

    // Documents known vulnerability: Beta inherits Alpha's session_id because
    // response_store has no tenant scoping.
    assert_eq!(
        alpha_session_id, beta_session_id,
        "cross-tenant chaining leaks session_id (known: response_store is global)"
    );
}
