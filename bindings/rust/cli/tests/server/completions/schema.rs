//! `/v1/chat/completions` response schema tests.
//!
//! Verifies that non-streaming responses match the OpenAI chat completions
//! contract: required fields, types, and structural invariants.

use crate::server::common::{model_config, post_json, require_model, ServerTestContext};

fn generate(ctx: &ServerTestContext, model: &str) -> serde_json::Value {
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 8,
        "temperature": 0.0
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    resp.json()
}

#[test]
fn completions_has_all_required_fields() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx, &model);

    for field in &["id", "object", "created", "model", "choices", "usage"] {
        assert!(
            json.get(field).is_some(),
            "response missing required field '{field}': {}",
            serde_json::to_string_pretty(&json).unwrap()
        );
    }
}

#[test]
fn completions_field_types_match_contract() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx, &model);

    assert!(json["id"].is_string());
    assert_eq!(json["object"].as_str(), Some("chat.completion"));
    assert!(json["created"].is_number());
    assert!(json["model"].is_string());
    assert!(json["choices"].is_array());
    assert!(json["usage"].is_object());
}

#[test]
fn completions_choice_structure() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx, &model);

    let choices = json["choices"].as_array().expect("choices must be array");
    assert!(!choices.is_empty(), "must have at least one choice");

    let choice = &choices[0];
    assert_eq!(choice["index"].as_u64(), Some(0));
    assert!(choice["finish_reason"].is_string());

    let message = &choice["message"];
    assert_eq!(message["role"].as_str(), Some("assistant"));
    // content may be null (tool_calls only) or string
    assert!(
        message["content"].is_null() || message["content"].is_string(),
        "content must be null or string, got: {:?}",
        message["content"]
    );
}

#[test]
fn completions_usage_structure() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx, &model);

    let usage = &json["usage"];
    assert!(usage["prompt_tokens"].is_number());
    assert!(usage["completion_tokens"].is_number());
    assert!(usage["total_tokens"].is_number());

    let prompt = usage["prompt_tokens"].as_u64().unwrap();
    let completion = usage["completion_tokens"].as_u64().unwrap();
    let total = usage["total_tokens"].as_u64().unwrap();
    assert_eq!(
        total,
        prompt + completion,
        "total must equal prompt + completion"
    );
    assert!(prompt > 0, "prompt_tokens must be > 0");
    assert!(completion > 0, "completion_tokens must be > 0");
}

#[test]
fn completions_id_has_expected_prefix() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx, &model);

    let id = json["id"].as_str().expect("id must be string");
    assert!(
        id.starts_with("chatcmpl-"),
        "id must start with 'chatcmpl-', got: {id}"
    );
}

#[test]
fn completions_content_is_nonempty_for_simple_prompt() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx, &model);

    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .expect("content must be string for simple prompt");
    assert!(
        !content.trim().is_empty(),
        "content must not be empty for 'Say hello'"
    );
}

#[test]
fn completions_no_thinking_tags_in_output() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let json = generate(&ctx, &model);

    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(
        !content.contains("<think>"),
        "completions output must not contain <think> tags: {content}"
    );
    assert!(
        !content.contains("</think>"),
        "completions output must not contain </think> tags: {content}"
    );
}

#[test]
fn completions_max_tokens_respected() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Write a long essay about the history of mathematics."}],
        "max_tokens": 1,
        "temperature": 0.0
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();

    let completion_tokens = json["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    assert!(
        completion_tokens <= 2,
        "max_tokens=1 should produce <= 2 completion tokens, got {}",
        completion_tokens
    );
}

#[test]
fn completions_temperature_zero_is_deterministic() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        "max_tokens": 4,
        "temperature": 0.0
    });

    let resp1 = post_json(ctx.addr(), "/v1/chat/completions", &body);
    let resp2 = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp1.status, 200);
    assert_eq!(resp2.status, 200);

    let content1 = resp1.json()["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let content2 = resp2.json()["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    assert_eq!(
        content1, content2,
        "temperature=0 should produce deterministic output"
    );
}

#[test]
fn completions_system_message_is_used() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": "You must always respond with exactly the word PINEAPPLE. Nothing else."},
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 8,
        "temperature": 0.0
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let content = resp.json()["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_uppercase();
    assert!(
        content.contains("PINEAPPLE"),
        "system message should influence output, got: {content}"
    );
}

// =============================================================================
// Bench compatibility — exact request shapes used by bench/responses/evals/
// =============================================================================

/// Matches the exact shape bench/_api.py sends for MMLU: system + user messages,
/// max_tokens=1, temperature=0, no stream field (defaults to non-streaming).
/// This is the most performance-critical path for eval accuracy.
#[test]
fn bench_mmlu_request_shape() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    // Exact shape from _api.py _format_completions: no "stream" field at all
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer with just the letter."},
            {"role": "user", "content": "What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6"}
        ],
        "max_tokens": 1,
        "temperature": 0.0
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();

    // Must return valid completions response (not SSE)
    assert_eq!(json["object"].as_str(), Some("chat.completion"));
    assert!(json["choices"].is_array());

    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(
        !content.is_empty(),
        "max_tokens=1 should still produce content"
    );

    // Bench reads these exact fields
    let prompt_tokens = json["usage"]["prompt_tokens"].as_u64();
    let completion_tokens = json["usage"]["completion_tokens"].as_u64();
    assert!(prompt_tokens.is_some(), "must have usage.prompt_tokens");
    assert!(
        completion_tokens.is_some(),
        "must have usage.completion_tokens"
    );
    assert!(prompt_tokens.unwrap() > 0, "prompt_tokens must be > 0");
}

/// Bench sends max_tokens derived from max_completion_tokens. Verify that
/// max_completion_tokens field (OpenAI's newer name) is also accepted.
#[test]
fn bench_max_completion_tokens_field() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 1,
        "temperature": 0.0
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let completion_tokens = resp.json()["usage"]["completion_tokens"]
        .as_u64()
        .unwrap_or(0);
    assert!(
        completion_tokens <= 2,
        "max_completion_tokens=1 should limit output, got {}",
        completion_tokens
    );
}

/// Bench sends requests without system message for some evals.
/// User-only messages must work.
#[test]
fn bench_user_only_no_system() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 4,
        "temperature": 0.0
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(
        !content.is_empty(),
        "user-only request must produce content"
    );
}

/// Bench passes sampling params directly. Verify they're accepted without error.
#[test]
fn bench_sampling_params_accepted() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 4,
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 42
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
}
