//! Non-streaming response envelope tests over real HTTP.

use crate::server::common::{
    get, model_config, post_json, require_model, ServerConfig, ServerTestContext,
};

/// Response has correct envelope fields (object, id, status, model, created_at).
#[test]
fn response_envelope() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Say hello",
        "max_output_tokens": 10,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["object"].as_str(), Some("response"));
    let id = json["id"].as_str().expect("should have id");
    assert!(id.starts_with("resp_"), "id should start with resp_: {id}");
    assert!(
        json["created_at"].as_f64().is_some(),
        "should have created_at"
    );
    assert_eq!(json["status"].as_str(), Some("completed"));
    assert_eq!(json["model"].as_str(), Some(model.as_str()));
}

/// Response Content-Type is application/json.
#[test]
fn response_content_type() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Hi",
        "max_output_tokens": 5,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200);
    let ct = resp
        .header("content-type")
        .expect("should have Content-Type");
    assert!(ct.contains("application/json"), "Content-Type: {ct}");
}

/// Output is an array with at least one typed item containing a message.
#[test]
fn response_output_shape() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 200,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let output = json["output"].as_array().expect("output should be array");
    assert!(!output.is_empty(), "output should have items");

    for item in output {
        let t = item["type"].as_str().expect("item should have type");
        assert!(
            ["message", "reasoning", "function_call"].contains(&t),
            "unexpected type: {t}",
        );
    }

    // At least one message or reasoning.
    let has_content = output
        .iter()
        .any(|item| matches!(item["type"].as_str(), Some("message") | Some("reasoning")));
    assert!(
        has_content,
        "output should contain a message or reasoning item"
    );
}

/// Message item has role=assistant and content array with output_text.
#[test]
fn response_content_shape() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Say hi",
        "max_output_tokens": 200,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let output = json["output"].as_array().unwrap();
    let msg = output
        .iter()
        .find(|i| i["type"].as_str() == Some("message"));
    if let Some(msg) = msg {
        assert_eq!(msg["role"].as_str(), Some("assistant"));
        let content = msg["content"].as_array().expect("content should be array");
        assert!(!content.is_empty());
        assert_eq!(content[0]["type"].as_str(), Some("output_text"));
        assert!(content[0]["text"].as_str().is_some());
    }
    // If no message item (only reasoning), that's OK — model may truncate.
}

/// Usage has input_tokens, output_tokens, total_tokens.
#[test]
fn response_usage() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Count to three",
        "max_output_tokens": 20,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let usage = &json["usage"];
    let input_tokens = usage["input_tokens"].as_u64().expect("input_tokens");
    let output_tokens = usage["output_tokens"].as_u64().expect("output_tokens");
    let total_tokens = usage["total_tokens"].as_u64().expect("total_tokens");
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);
    assert_eq!(total_tokens, input_tokens + output_tokens);
}

/// GET /v1/models returns a list with the configured model.
#[test]
fn models_list() {
    let _ = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = get(ctx.addr(), "/v1/models");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    assert_eq!(json["object"].as_str(), Some("list"));
    let data = json["data"].as_array().expect("data array");
    assert!(!data.is_empty());
    assert!(data[0]["id"].as_str().is_some());
}

/// GET /health returns 200 "ok".
#[test]
fn health_check() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = get(ctx.addr(), "/health");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.body.trim(), "ok");
}
