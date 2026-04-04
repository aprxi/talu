//! `/v1/chat/completions` streaming (SSE) tests.

use crate::server::common::{model_config, post_json, require_model, ServerTestContext};

fn parse_sse_chunks(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect()
}

fn streaming_body(model: &str, content: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": true,
        "max_tokens": 16,
        "temperature": 0.0
    })
}

#[test]
fn streaming_returns_sse_content_type() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/chat/completions",
        &streaming_body(&model, "Hi"),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let ct = resp.header("content-type").unwrap_or("");
    assert!(
        ct.contains("text/event-stream"),
        "streaming must return text/event-stream, got: {ct}"
    );
}

#[test]
fn streaming_ends_with_done() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/chat/completions",
        &streaming_body(&model, "Hi"),
    );
    assert_eq!(resp.status, 200);

    assert!(
        resp.body.contains("data: [DONE]"),
        "stream must end with [DONE]: {}",
        &resp.body[resp.body.len().saturating_sub(200)..]
    );
}

#[test]
fn streaming_chunks_have_correct_object() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/chat/completions",
        &streaming_body(&model, "Hi"),
    );
    assert_eq!(resp.status, 200);

    let chunks = parse_sse_chunks(&resp.body);
    assert!(!chunks.is_empty(), "must have at least one chunk");

    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(
            chunk["object"].as_str(),
            Some("chat.completion.chunk"),
            "chunk {} must have object=chat.completion.chunk: {:?}",
            i,
            chunk
        );
    }
}

#[test]
fn streaming_first_chunk_has_role() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/chat/completions",
        &streaming_body(&model, "Hi"),
    );
    assert_eq!(resp.status, 200);

    let chunks = parse_sse_chunks(&resp.body);
    assert!(!chunks.is_empty());

    let first = &chunks[0];
    let role = first["choices"][0]["delta"]["role"].as_str();
    assert_eq!(
        role,
        Some("assistant"),
        "first chunk must set role=assistant"
    );
}

#[test]
fn streaming_content_deltas_are_nonempty() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/chat/completions",
        &streaming_body(&model, "Say hello"),
    );
    assert_eq!(resp.status, 200);

    let chunks = parse_sse_chunks(&resp.body);
    let content_chunks: Vec<&str> = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();

    assert!(!content_chunks.is_empty(), "must have content delta chunks");

    // Concatenated content should form readable text
    let full_content: String = content_chunks.into_iter().collect();
    assert!(
        !full_content.trim().is_empty(),
        "concatenated content must not be empty"
    );
}

#[test]
fn streaming_last_content_chunk_has_finish_reason() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/chat/completions",
        &streaming_body(&model, "Hi"),
    );
    assert_eq!(resp.status, 200);

    let chunks = parse_sse_chunks(&resp.body);
    // Find the last chunk that has a finish_reason
    let final_chunks: Vec<_> = chunks
        .iter()
        .filter(|c| c["choices"][0]["finish_reason"].is_string())
        .collect();

    assert!(
        !final_chunks.is_empty(),
        "at least one chunk must have finish_reason"
    );
    let finish = final_chunks.last().unwrap()["choices"][0]["finish_reason"]
        .as_str()
        .unwrap();
    assert!(
        finish == "stop" || finish == "length",
        "finish_reason must be 'stop' or 'length', got: {finish}"
    );
}

#[test]
fn streaming_no_thinking_tags_in_content() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/chat/completions",
        &streaming_body(&model, "Say hello"),
    );
    assert_eq!(resp.status, 200);

    let chunks = parse_sse_chunks(&resp.body);
    let full_content: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();

    assert!(
        !full_content.contains("<think>"),
        "streaming content must not contain <think> tags: {full_content}"
    );
    assert!(
        !full_content.contains("</think>"),
        "streaming content must not contain </think> tags: {full_content}"
    );
}

#[test]
fn streaming_chunks_share_same_id() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/chat/completions",
        &streaming_body(&model, "Hi"),
    );
    assert_eq!(resp.status, 200);

    let chunks = parse_sse_chunks(&resp.body);
    assert!(
        chunks.len() >= 2,
        "need at least 2 chunks to verify id consistency"
    );

    let first_id = chunks[0]["id"].as_str().expect("first chunk must have id");
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(
            chunk["id"].as_str(),
            Some(first_id),
            "chunk {} has different id",
            i
        );
    }
}

#[test]
fn streaming_usage_in_final_chunk() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/chat/completions",
        &streaming_body(&model, "Hi"),
    );
    assert_eq!(resp.status, 200);

    let chunks = parse_sse_chunks(&resp.body);
    // Check if the last chunk or any chunk with finish_reason has usage
    let final_chunk = chunks
        .iter()
        .filter(|c| c["choices"][0]["finish_reason"].is_string())
        .last();

    if let Some(chunk) = final_chunk {
        if let Some(usage) = chunk.get("usage") {
            if !usage.is_null() {
                assert!(usage["prompt_tokens"].is_number());
                assert!(usage["completion_tokens"].is_number());
                assert!(usage["total_tokens"].is_number());
            }
        }
        // Usage in final chunk is optional per OpenAI spec — pass either way.
    }
}
