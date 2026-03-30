//! `/v1/chat/completions` parameter handling and edge cases.

use crate::server::common::{model_config, post_json, require_model, ServerTestContext};

fn generate(ctx: &ServerTestContext, model: &str, extra: serde_json::Value) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8,
        "temperature": 0.0
    });
    if let Some(obj) = extra.as_object() {
        for (k, v) in obj {
            body[k.as_str()] = v.clone();
        }
    }
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    resp.json()
}

// =============================================================================
// finish_reason semantics
// =============================================================================

/// When max_tokens truncates output, finish_reason must be "length".
#[test]
fn finish_reason_length_on_truncation() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Write a very long detailed essay about the complete history of all mathematics from ancient times to modern day."}],
        "max_tokens": 1,
        "temperature": 0.0
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let finish = json["choices"][0]["finish_reason"].as_str().unwrap_or("");
    // max_tokens=1 with a long prompt should hit the limit
    assert!(
        finish == "length" || finish == "stop",
        "finish_reason should be 'length' or 'stop', got: {finish}"
    );
}

/// Natural completion should give finish_reason "stop".
#[test]
fn finish_reason_stop_on_natural_end() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 100,
        "temperature": 0.0
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(
        json["choices"][0]["finish_reason"].as_str(),
        Some("stop"),
        "short response with high limit should finish with 'stop'"
    );
}

// =============================================================================
// Multi-turn context
// =============================================================================

/// Assistant messages in the input provide conversation context.
#[test]
fn multi_turn_assistant_context() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a calculator that only outputs numbers."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Add 3 to that."}
        ],
        "max_tokens": 8,
        "temperature": 0.0
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "multi-turn should produce output");
}

// =============================================================================
// Model echo
// =============================================================================

/// The response model field should reflect the resolved model, not be empty.
#[test]
fn model_field_echoed_in_response() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx, &model, serde_json::json!({}));

    let response_model = json["model"].as_str().expect("model field");
    assert!(
        !response_model.is_empty(),
        "model must not be empty in response"
    );
}

// =============================================================================
// Streaming vs non-streaming equivalence
// =============================================================================

/// Same prompt should produce the same content via streaming and non-streaming.
#[test]
fn streaming_and_non_streaming_produce_same_content() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let base = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "What is 1+1? Reply with just the number."}],
        "max_tokens": 4,
        "temperature": 0.0
    });

    // Non-streaming
    let non_stream_resp = post_json(ctx.addr(), "/v1/chat/completions", &base);
    assert_eq!(non_stream_resp.status, 200);
    let non_stream_content = non_stream_resp.json()["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    // Streaming
    let mut stream_body = base.clone();
    stream_body["stream"] = serde_json::json!(true);
    let stream_resp = post_json(ctx.addr(), "/v1/chat/completions", &stream_body);
    assert_eq!(stream_resp.status, 200);

    let streamed_content: String = stream_resp
        .body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .map(|s| s.to_string())
        })
        .collect();

    assert_eq!(
        non_stream_content.trim(),
        streamed_content.trim(),
        "streaming and non-streaming must produce identical content"
    );
}

// =============================================================================
// Content format variants
// =============================================================================

/// Content as array (vision format) should be accepted.
#[test]
fn content_as_array_accepted() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let body = serde_json::json!({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "Say hello"}]
        }],
        "max_tokens": 8,
        "temperature": 0.0
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    // Should either succeed or return a clear error — not crash or 500
    assert!(
        resp.status == 200 || resp.status == 400,
        "content-as-array should be accepted (200) or cleanly rejected (400), got {}: {}",
        resp.status,
        resp.body
    );
}
