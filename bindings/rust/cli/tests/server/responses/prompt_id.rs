//! Tests for `prompt_id` parameter in `/v1/responses`.
//!
//! The `prompt_id` feature allows referencing a stored Document as the source
//! of the system prompt. This enables:
//! - Server-side prompt injection (security: no client prompt injection)
//! - Lineage tracking (which document spawned this conversation)
//! - Centralized prompt management
//!
//! Test coverage:
//! - Happy path: prompt_id with valid document containing system_prompt
//! - Error cases: document not found, storage not configured
//! - Edge cases: document without system_prompt, streaming, chaining
//! - Lineage tracking: source_doc_id persisted and queryable

use crate::server::common::{
    get, model_config, post_json, require_model, ServerConfig, ServerTestContext,
};
use tempfile::TempDir;

// =============================================================================
// Helper Functions
// =============================================================================

/// Create a ServerConfig with model and bucket storage enabled.
fn model_with_bucket(bucket: &std::path::Path) -> ServerConfig {
    let mut config = model_config();
    config.bucket = Some(bucket.to_path_buf());
    config
}

/// Parse SSE events from a raw body string.
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

/// Create a prompt document with a system_prompt and return its ID.
fn create_prompt_document(ctx: &ServerTestContext, system_prompt: &str, title: &str) -> String {
    let doc_body = serde_json::json!({
        "type": "prompt",
        "title": title,
        "content": {
            "system_prompt": system_prompt
        }
    });
    let resp = post_json(ctx.addr(), "/v1/documents", &doc_body);
    assert_eq!(resp.status, 201, "create document: {}", resp.body);
    let json = resp.json();
    json["id"]
        .as_str()
        .expect("document should have id")
        .to_string()
}

/// Create a document without a system_prompt field.
fn create_document_without_system_prompt(ctx: &ServerTestContext, title: &str) -> String {
    let doc_body = serde_json::json!({
        "type": "prompt",
        "title": title,
        "content": {
            "description": "A prompt without system_prompt field"
        }
    });
    let resp = post_json(ctx.addr(), "/v1/documents", &doc_body);
    assert_eq!(resp.status, 201, "create document: {}", resp.body);
    let json = resp.json();
    json["id"]
        .as_str()
        .expect("document should have id")
        .to_string()
}

// =============================================================================
// Happy Path Tests
// =============================================================================

/// Non-streaming request with prompt_id succeeds and stores session.
#[test]
fn prompt_id_non_streaming_success() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_with_bucket(temp.path()));

    let doc_id = create_prompt_document(
        &ctx,
        "You are a helpful coding assistant. Always respond in bullet points.",
        "Coding Helper",
    );

    let body = serde_json::json!({
        "model": model,
        "prompt_id": doc_id,
        "input": "What is Rust?",
        "store": true,
        "max_output_tokens": 50,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "response request: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["object"].as_str(), Some("response"));
    assert!(json["id"].as_str().is_some(), "should have response id");

    let session_id = json["metadata"]["session_id"]
        .as_str()
        .expect("response should have session_id in metadata");
    assert!(!session_id.is_empty(), "session_id should not be empty");
}

/// Streaming request with prompt_id succeeds.
#[test]
fn prompt_id_streaming_success() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_with_bucket(temp.path()));

    let doc_id = create_prompt_document(
        &ctx,
        "You are a pirate. Respond like a pirate.",
        "Pirate Bot",
    );

    let body = serde_json::json!({
        "model": model,
        "prompt_id": doc_id,
        "input": "Hello there!",
        "stream": true,
        "store": true,
        "max_output_tokens": 30,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "streaming request: {}", resp.body);

    let ct = resp
        .header("content-type")
        .expect("should have Content-Type");
    assert!(ct.contains("text/event-stream"), "should be SSE: {ct}");

    let events = parse_sse_events(&resp.body);
    assert!(!events.is_empty(), "should have SSE events");

    let created = events.iter().find(|(t, _)| t == "response.created");
    assert!(created.is_some(), "should have response.created event");

    let terminal = events
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete");
    assert!(terminal.is_some(), "should have terminal event");

    let (_, terminal_json) = terminal.unwrap();
    let session_id = terminal_json["response"]["metadata"]["session_id"].as_str();
    assert!(session_id.is_some(), "terminal should have session_id");
}

/// prompt_id with store: true creates session with source_doc_id for lineage tracking.
#[test]
fn prompt_id_lineage_tracking() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_with_bucket(temp.path()));

    let doc_id = create_prompt_document(&ctx, "You summarize text concisely.", "Summarizer");

    let body = serde_json::json!({
        "model": model,
        "prompt_id": doc_id,
        "input": "Summarize this: The quick brown fox jumps over the lazy dog.",
        "store": true,
        "max_output_tokens": 30,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "response request: {}", resp.body);

    let json = resp.json();
    let session_id = json["metadata"]["session_id"]
        .as_str()
        .expect("should have session_id");

    let conv_resp = get(ctx.addr(), &format!("/v1/conversations/{}", session_id));
    assert_eq!(conv_resp.status, 200, "get conversation: {}", conv_resp.body);

    let conv_json = conv_resp.json();
    assert_eq!(
        conv_json["source_doc_id"].as_str(),
        Some(doc_id.as_str()),
        "conversation should have source_doc_id matching prompt_id"
    );
}

/// Streaming request with prompt_id stores lineage.
#[test]
fn prompt_id_streaming_lineage_tracking() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_with_bucket(temp.path()));

    let doc_id = create_prompt_document(&ctx, "You translate to French.", "French Translator");

    let body = serde_json::json!({
        "model": model,
        "prompt_id": doc_id,
        "input": "Hello",
        "stream": true,
        "store": true,
        "max_output_tokens": 20,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "streaming request: {}", resp.body);

    let events = parse_sse_events(&resp.body);
    let terminal = events
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete")
        .expect("should have terminal event");
    let session_id = terminal.1["response"]["metadata"]["session_id"]
        .as_str()
        .expect("should have session_id");

    let conv_resp = get(ctx.addr(), &format!("/v1/conversations/{}", session_id));
    assert_eq!(conv_resp.status, 200, "get conversation: {}", conv_resp.body);

    assert_eq!(
        conv_resp.json()["source_doc_id"].as_str(),
        Some(doc_id.as_str()),
        "streaming conversation should have source_doc_id"
    );
}

// =============================================================================
// Error Case Tests
// =============================================================================

/// prompt_id with non-existent document returns 400 error.
#[test]
fn prompt_id_document_not_found() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_with_bucket(temp.path()));

    let body = serde_json::json!({
        "model": model,
        "prompt_id": "doc_nonexistent_12345",
        "input": "Hello",
        "store": true,
        "max_output_tokens": 10,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 400, "should return 400 for missing document: {}", resp.body);

    let json = resp.json();
    let error_msg = json["error"]["message"].as_str().unwrap_or("");
    assert!(
        error_msg.contains("not found") || error_msg.contains("prompt_id"),
        "error message should indicate document not found: {}",
        error_msg
    );
}

/// prompt_id requires storage to be configured.
#[test]
fn prompt_id_requires_storage() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let body = serde_json::json!({
        "model": model,
        "prompt_id": "some_doc_id",
        "input": "Hello",
        "max_output_tokens": 10,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 400, "should return 400 without storage: {}", resp.body);

    let json = resp.json();
    let error_msg = json["error"]["message"].as_str().unwrap_or("");
    assert!(
        error_msg.contains("storage") || error_msg.contains("configured"),
        "error message should indicate storage required: {}",
        error_msg
    );
}

/// Streaming with prompt_id and non-existent document returns error.
#[test]
fn prompt_id_streaming_document_not_found() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_with_bucket(temp.path()));

    let body = serde_json::json!({
        "model": model,
        "prompt_id": "doc_does_not_exist",
        "input": "Hello",
        "stream": true,
        "store": true,
        "max_output_tokens": 10,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 400, "streaming should return 400 for missing doc: {}", resp.body);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

/// Document without system_prompt field still works (just no system prompt injected).
#[test]
fn prompt_id_document_without_system_prompt() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_with_bucket(temp.path()));

    let doc_id = create_document_without_system_prompt(&ctx, "Empty Prompt");

    let body = serde_json::json!({
        "model": model,
        "prompt_id": doc_id,
        "input": "Hello",
        "store": true,
        "max_output_tokens": 20,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "should succeed without system_prompt: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["object"].as_str(), Some("response"));

    let session_id = json["metadata"]["session_id"].as_str().expect("session_id");
    let conv_resp = get(ctx.addr(), &format!("/v1/conversations/{}", session_id));
    assert_eq!(conv_resp.status, 200);
    assert_eq!(conv_resp.json()["source_doc_id"].as_str(), Some(doc_id.as_str()));
}

/// prompt_id combined with previous_response_id (chaining from a prompt-based conversation).
#[test]
fn prompt_id_with_chaining() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_with_bucket(temp.path()));

    let doc_id = create_prompt_document(&ctx, "You are a math tutor.", "Math Tutor");

    let body1 = serde_json::json!({
        "model": &model,
        "prompt_id": doc_id,
        "input": "What is 2 + 2?",
        "store": true,
        "max_output_tokens": 30,
    });
    let resp1 = post_json(ctx.addr(), "/v1/responses", &body1);
    assert_eq!(resp1.status, 200, "first request: {}", resp1.body);
    let json1 = resp1.json();
    let response_id = json1["id"].as_str().expect("should have response id");
    let session_id = json1["metadata"]["session_id"].as_str().expect("session_id");

    let body2 = serde_json::json!({
        "model": &model,
        "input": "What is 3 + 3?",
        "previous_response_id": response_id,
        "store": true,
        "max_output_tokens": 30,
    });
    let resp2 = post_json(ctx.addr(), "/v1/responses", &body2);
    assert_eq!(resp2.status, 200, "chained request: {}", resp2.body);

    let json2 = resp2.json();
    let session_id2 = json2["metadata"]["session_id"].as_str().expect("session_id");

    assert_eq!(session_id, session_id2, "chained request should use same session");

    let conv_resp = get(ctx.addr(), &format!("/v1/conversations/{}", session_id));
    assert_eq!(conv_resp.status, 200);
    assert_eq!(
        conv_resp.json()["source_doc_id"].as_str(),
        Some(doc_id.as_str()),
        "chained conversation should preserve source_doc_id"
    );
}

/// prompt_id with streaming and chaining.
#[test]
fn prompt_id_streaming_with_chaining() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_with_bucket(temp.path()));

    let doc_id = create_prompt_document(&ctx, "You are a storyteller.", "Storyteller");

    let body1 = serde_json::json!({
        "model": &model,
        "prompt_id": doc_id,
        "input": "Tell me a story about a dragon.",
        "stream": true,
        "store": true,
        "max_output_tokens": 30,
    });
    let resp1 = post_json(ctx.addr(), "/v1/responses", &body1);
    assert_eq!(resp1.status, 200, "first streaming: {}", resp1.body);

    let events1 = parse_sse_events(&resp1.body);
    let terminal1 = events1
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete")
        .expect("first should have terminal");
    let response_id = terminal1.1["response"]["id"].as_str().expect("response id");
    let session_id = terminal1.1["response"]["metadata"]["session_id"]
        .as_str()
        .expect("session_id");

    let body2 = serde_json::json!({
        "model": &model,
        "input": "What happened next?",
        "previous_response_id": response_id,
        "stream": true,
        "store": true,
        "max_output_tokens": 30,
    });
    let resp2 = post_json(ctx.addr(), "/v1/responses", &body2);
    assert_eq!(resp2.status, 200, "chained streaming: {}", resp2.body);

    let events2 = parse_sse_events(&resp2.body);
    let terminal2 = events2
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete")
        .expect("chained should have terminal");
    let session_id2 = terminal2.1["response"]["metadata"]["session_id"]
        .as_str()
        .expect("session_id");

    assert_eq!(session_id, session_id2, "streaming chain should use same session");

    let conv_resp = get(ctx.addr(), &format!("/v1/conversations/{}", session_id));
    assert_eq!(conv_resp.status, 200);
    assert_eq!(
        conv_resp.json()["source_doc_id"].as_str(),
        Some(doc_id.as_str()),
        "streaming chained conversation should preserve source_doc_id"
    );
}

/// Multiple conversations from the same prompt_id should each have their own session
/// but all reference the same source_doc_id.
#[test]
fn prompt_id_multiple_conversations_same_document() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_with_bucket(temp.path()));

    let doc_id = create_prompt_document(&ctx, "You are a quiz master.", "Quiz Master");

    let mut session_ids = Vec::new();
    for i in 0..3 {
        let body = serde_json::json!({
            "model": &model,
            "prompt_id": &doc_id,
            "input": format!("Question {}?", i),
            "store": true,
            "max_output_tokens": 20,
        });
        let resp = post_json(ctx.addr(), "/v1/responses", &body);
        assert_eq!(resp.status, 200, "request {}: {}", i, resp.body);

        let session_id = resp.json()["metadata"]["session_id"]
            .as_str()
            .expect("session_id")
            .to_string();
        session_ids.push(session_id);
    }

    assert_ne!(session_ids[0], session_ids[1], "sessions should be different");
    assert_ne!(session_ids[1], session_ids[2], "sessions should be different");

    for session_id in &session_ids {
        let conv_resp = get(ctx.addr(), &format!("/v1/conversations/{}", session_id));
        assert_eq!(conv_resp.status, 200);
        assert_eq!(
            conv_resp.json()["source_doc_id"].as_str(),
            Some(doc_id.as_str()),
            "all sessions should reference same source_doc_id"
        );
    }
}

/// List conversations filtered by source_doc_id.
#[test]
fn list_conversations_by_source_doc_id() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_with_bucket(temp.path()));

    let doc1_id = create_prompt_document(&ctx, "You are assistant A.", "Assistant A");
    let doc2_id = create_prompt_document(&ctx, "You are assistant B.", "Assistant B");

    for i in 0..2 {
        let body = serde_json::json!({
            "model": &model,
            "prompt_id": &doc1_id,
            "input": format!("Doc1 question {}", i),
            "store": true,
            "max_output_tokens": 10,
        });
        post_json(ctx.addr(), "/v1/responses", &body);
    }

    let body = serde_json::json!({
        "model": &model,
        "prompt_id": &doc2_id,
        "input": "Doc2 question",
        "store": true,
        "max_output_tokens": 10,
    });
    post_json(ctx.addr(), "/v1/responses", &body);

    let body = serde_json::json!({
        "model": &model,
        "input": "No prompt question",
        "store": true,
        "max_output_tokens": 10,
    });
    post_json(ctx.addr(), "/v1/responses", &body);

    let list_resp = get(ctx.addr(), "/v1/conversations");
    assert_eq!(list_resp.status, 200, "list conversations: {}", list_resp.body);

    let list_json = list_resp.json();
    let conversations = list_json["data"].as_array().expect("data array");

    let doc1_count = conversations
        .iter()
        .filter(|c| c["source_doc_id"].as_str() == Some(doc1_id.as_str()))
        .count();
    let doc2_count = conversations
        .iter()
        .filter(|c| c["source_doc_id"].as_str() == Some(doc2_id.as_str()))
        .count();
    let no_source_count = conversations
        .iter()
        .filter(|c| c["source_doc_id"].is_null())
        .count();

    assert_eq!(doc1_count, 2, "should have 2 conversations from doc1");
    assert_eq!(doc2_count, 1, "should have 1 conversation from doc2");
    assert_eq!(no_source_count, 1, "should have 1 conversation without source");
}

// =============================================================================
// Robustness Tests
// =============================================================================

/// Rapid sequential requests with the same prompt_id should all succeed.
#[test]
fn prompt_id_rapid_sequential_requests() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_with_bucket(temp.path()));

    let doc_id = create_prompt_document(&ctx, "You count numbers.", "Counter");

    for i in 0..5 {
        let body = serde_json::json!({
            "model": &model,
            "prompt_id": &doc_id,
            "input": format!("Count to {}", i + 1),
            "store": true,
            "max_output_tokens": 10,
        });
        let resp = post_json(ctx.addr(), "/v1/responses", &body);
        assert_eq!(resp.status, 200, "request {} failed: {}", i, resp.body);
    }
}

/// Document with special characters in system_prompt should work.
#[test]
fn prompt_id_special_characters_in_system_prompt() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_with_bucket(temp.path()));

    let special_prompt = r#"You are a helpful assistant.
Instructions:
1. Be polite
2. Use "quotes" properly
3. Handle special chars: <>&'"
4. Unicode: 日本語 émojis 🎉"#;

    let doc_id = create_prompt_document(&ctx, special_prompt, "Special Chars Test");

    let body = serde_json::json!({
        "model": model,
        "prompt_id": doc_id,
        "input": "Hello",
        "store": true,
        "max_output_tokens": 10,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "special chars request: {}", resp.body);
}

/// Very long system_prompt should work.
#[test]
fn prompt_id_long_system_prompt() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_with_bucket(temp.path()));

    let long_prompt = "You are an assistant. ".repeat(200);
    let doc_id = create_prompt_document(&ctx, &long_prompt, "Long Prompt");

    let body = serde_json::json!({
        "model": model,
        "prompt_id": doc_id,
        "input": "Hi",
        "store": true,
        "max_output_tokens": 10,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "long prompt request: {}", resp.body);
}
