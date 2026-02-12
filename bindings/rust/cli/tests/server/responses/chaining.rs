//! Conversation chaining (previous_response_id) tests over real HTTP.

use crate::server::common::{model_config, post_json, require_model, ServerTestContext};

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

/// Chained request references first response and succeeds.
#[test]
fn conversation_chaining() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    // First request.
    let body1 = serde_json::json!({
        "model": &model,
        "input": "Hello, my name is Alice.",
        "max_output_tokens": 50,
    });
    let resp1 = post_json(ctx.addr(), "/v1/responses", &body1);
    assert_eq!(resp1.status, 200, "first request: {}", resp1.body);
    let json1 = resp1.json();
    let response_id = json1["id"].as_str().expect("should have id");

    // Chained request.
    let body2 = serde_json::json!({
        "model": &model,
        "input": "What is my name?",
        "previous_response_id": response_id,
        "max_output_tokens": 50,
    });
    let resp2 = post_json(ctx.addr(), "/v1/responses", &body2);
    assert_eq!(resp2.status, 200, "chained request: {}", resp2.body);
    let json2 = resp2.json();
    assert_eq!(
        json2["previous_response_id"].as_str(),
        Some(response_id),
        "should reference previous response",
    );
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
        "max_output_tokens": 50,
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
        "max_output_tokens": 50,
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
        "max_output_tokens": 20,
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
        "max_output_tokens": 20,
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
