//! `/v1/chat/completions` error-path tests.

use crate::server::common::{
    model_config, post_json, require_model, send_request, ServerConfig, ServerTestContext,
};

#[test]
fn completions_empty_body_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/chat/completions",
        &[("Content-Type", "application/json")],
        Some(""),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    let json = resp.json();
    assert!(
        json["error"]["message"].is_string(),
        "error should have message: {}",
        resp.body
    );
}

#[test]
fn completions_invalid_json_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/chat/completions",
        &[("Content-Type", "application/json")],
        Some("{not json}"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn completions_missing_messages_returns_400() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({ "model": "test" });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn completions_no_model_uses_default() {
    let _ = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert!(
        json["model"].is_string(),
        "response must include model name"
    );
}

#[test]
fn completions_error_shape_matches_openai() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/chat/completions",
        &[("Content-Type", "application/json")],
        Some(""),
    );
    let json = resp.json();
    // OpenAI error shape: { "error": { "message": "...", "type": "...", "code": ..., "param": ... } }
    assert!(
        json["error"].is_object(),
        "must have error object: {}",
        resp.body
    );
    assert!(
        json["error"]["message"].is_string(),
        "must have error.message"
    );
    assert!(json["error"]["type"].is_string(), "must have error.type");
}
