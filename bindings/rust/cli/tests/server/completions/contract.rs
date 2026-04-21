//! Strict contract tests for `/v1/chat/completions`.
//!
//! These tests intentionally enforce request-shape and validation behavior so
//! regressions are caught at the HTTP boundary.

use crate::server::common::{post_json, send_request, ServerConfig, ServerTestContext};

fn assert_invalid_request(resp: &crate::server::common::HttpResponse) {
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(
        json["error"]["type"].as_str(),
        Some("invalid_request_error"),
        "body: {}",
        resp.body
    );
    assert!(json["error"]["message"].is_string(), "body: {}", resp.body);
}

#[test]
fn completions_rejects_unknown_top_level_fields() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "model": "test/model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "unexpected_field": true
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_invalid_request(&resp);
}

#[test]
fn completions_rejects_negative_max_tokens() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "model": "test/model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": -1
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_invalid_request(&resp);
}

#[test]
fn completions_rejects_negative_max_completion_tokens() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "model": "test/model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": -1
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_invalid_request(&resp);
}

#[test]
fn completions_rejects_out_of_range_temperature() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "model": "test/model",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 2.5
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_invalid_request(&resp);
}

#[test]
fn completions_rejects_out_of_range_top_p() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "model": "test/model",
        "messages": [{"role": "user", "content": "hi"}],
        "top_p": 1.1
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_invalid_request(&resp);
}

#[test]
fn completions_rejects_invalid_tool_choice_string() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "model": "test/model",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": "sometimes"
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_invalid_request(&resp);
}

#[test]
fn completions_rejects_invalid_tool_choice_object() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "model": "test/model",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": {"type": "function"}
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_invalid_request(&resp);
}

#[test]
fn completions_rejects_invalid_tools_shape() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "model": "test/model",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": {"bad": "shape"}
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_invalid_request(&resp);
}

#[test]
fn completions_invalid_message_role_is_client_error() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "model": "test/model",
        "messages": [{"role": "bogus", "content": "hi"}]
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_invalid_request(&resp);
}

#[test]
fn completions_tool_message_without_tool_call_id_is_client_error() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "model": "test/model",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "tool output without id"}
        ]
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_invalid_request(&resp);
}

#[test]
fn completions_streaming_invalid_request_returns_400_json() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "model": "test/model",
        "stream": true,
        "messages": [{"role": "bogus", "content": "hi"}]
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    let ct = resp.header("content-type").unwrap_or("");
    assert!(
        ct.contains("application/json"),
        "invalid streaming request must return JSON error, got content-type={ct}, body={}",
        resp.body
    );
    let json = resp.json();
    assert_eq!(
        json["error"]["type"].as_str(),
        Some("invalid_request_error")
    );
}

#[test]
fn completions_streaming_invalid_json_returns_400_json() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/chat/completions",
        &[("Content-Type", "application/json")],
        Some("{invalid json"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    let ct = resp.header("content-type").unwrap_or("");
    assert!(
        ct.contains("application/json"),
        "invalid JSON must return JSON error, got content-type={ct}, body={}",
        resp.body
    );
}
