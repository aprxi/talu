//! Input format variant tests over real HTTP.

use crate::server::common::{model_config, post_json, require_model, ServerTestContext};

/// Plain string input is accepted and produces output.
#[test]
fn string_input() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 5,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["object"].as_str(), Some("response"));
}

/// Structured array input with a user message item.
#[test]
fn structured_user_message() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": [{"type": "message", "role": "user", "content": "Say hello"}],
        "max_output_tokens": 200,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let output = json["output"].as_array().expect("should have output");
    assert!(!output.is_empty());
}

/// System + user messages in structured input.
#[test]
fn system_and_user_messages() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": [
            {"type": "message", "role": "system", "content": "You are a helpful assistant."},
            {"type": "message", "role": "user", "content": "Hello"}
        ],
        "max_output_tokens": 200,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let output = json["output"].as_array().expect("should have output");
    assert!(!output.is_empty());
}

/// Developer message (alias for system) is accepted.
#[test]
fn developer_message() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": [
            {"type": "message", "role": "developer", "content": "Be brief."},
            {"type": "message", "role": "user", "content": "Hi"}
        ],
        "max_output_tokens": 200,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
}

/// Message array input (legacy format without "type" discriminator).
#[test]
fn message_array_input() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": [{"type": "message", "role": "user", "content": "Hi"}],
        "max_output_tokens": 5,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
}
