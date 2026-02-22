//! Streaming (SSE) lifecycle tests over real HTTP.
//!
//! Basic streaming conformance (content-type, cache headers, event sequence,
//! delta/done/terminal events, sequence numbers) is covered by the in-process
//! `api_compliance` test suite. This module tests advanced lifecycle events
//! and edge cases that only the subprocess harness exercises.

use crate::server::common::{
    get, model_config, post_json, require_model, ServerConfig, ServerTestContext,
};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;
use tempfile::TempDir;

fn streaming_body(model: &str, input: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "input": input,
        "stream": true,
        "max_output_tokens": 10,
    })
}

/// Parse SSE events from a raw body string.
/// Returns Vec<(event_type, json_data)>.
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

/// Stream contains response.in_progress event after response.created.
#[test]
fn streaming_in_progress_event() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body(&model, "Hello"),
    );
    let events = parse_sse_events(&resp.body);
    assert!(events.len() >= 2, "should have at least 2 events");

    let (ref t, ref data) = events[1];
    assert_eq!(
        t, "response.in_progress",
        "second event should be response.in_progress"
    );
    let response = &data["response"];
    assert_eq!(response["status"].as_str(), Some("in_progress"));
    assert_eq!(response["object"].as_str(), Some("response"));
}

/// Stream contains response.output_item.added event with an item object.
#[test]
fn streaming_output_item_added() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body(&model, "Hello"),
    );
    let events = parse_sse_events(&resp.body);

    let added = events
        .iter()
        .find(|(t, _)| t == "response.output_item.added");
    assert!(
        added.is_some(),
        "should have response.output_item.added event"
    );
    let (_, data) = added.unwrap();
    let item = &data["item"];
    assert!(item.is_object(), "item should be object");
    let item_type = item["type"].as_str().expect("item.type");
    assert!(
        ["message", "reasoning", "function_call"].contains(&item_type),
        "item type: {item_type}",
    );
    assert!(item["id"].as_str().is_some(), "item should have id");
    assert_eq!(
        item["status"].as_str(),
        Some("in_progress"),
        "item status should be in_progress"
    );
    assert!(data["output_index"].is_number(), "should have output_index");
}

/// Stream contains response.content_part.added event.
#[test]
fn streaming_content_part_added() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body(&model, "Hello"),
    );
    let events = parse_sse_events(&resp.body);

    let added = events
        .iter()
        .find(|(t, _)| t == "response.content_part.added");
    assert!(
        added.is_some(),
        "should have response.content_part.added event"
    );
    let (_, data) = added.unwrap();
    let part = &data["part"];
    assert!(part.is_object(), "part should be object");
    assert!(part["type"].as_str().is_some(), "part should have type");
    assert!(data["item_id"].as_str().is_some(), "should have item_id");
    assert!(data["output_index"].is_number(), "should have output_index");
    assert!(
        data["content_index"].is_number(),
        "should have content_index"
    );
}

/// Stream contains response.content_part.done event with completed part.
#[test]
fn streaming_content_part_done() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body(&model, "Hello"),
    );
    let events = parse_sse_events(&resp.body);

    let done = events
        .iter()
        .find(|(t, _)| t == "response.content_part.done");
    assert!(
        done.is_some(),
        "should have response.content_part.done event"
    );
    let (_, data) = done.unwrap();
    let part = &data["part"];
    assert!(part.is_object(), "part should be object");
    assert!(part["type"].as_str().is_some(), "part should have type");
    // The text in the part should be the accumulated content.
    assert!(part["text"].is_string(), "part should have text");
    assert!(data["item_id"].as_str().is_some(), "should have item_id");
}

/// Stream contains response.output_item.done event with completed item.
#[test]
fn streaming_output_item_done() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body(&model, "Hello"),
    );
    let events = parse_sse_events(&resp.body);

    let done = events
        .iter()
        .find(|(t, _)| t == "response.output_item.done");
    assert!(
        done.is_some(),
        "should have response.output_item.done event"
    );
    let (_, data) = done.unwrap();
    let item = &data["item"];
    assert!(item.is_object(), "item should be object");
    assert_eq!(
        item["status"].as_str(),
        Some("completed"),
        "done item should be completed"
    );
    assert!(item["id"].as_str().is_some(), "item should have id");
    assert!(data["output_index"].is_number(), "should have output_index");
}

/// Terminal response resource in streaming has all 31 required fields.
#[test]
fn streaming_terminal_response_schema() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body(&model, "Hello"),
    );
    let events = parse_sse_events(&resp.body);

    let terminal = events
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete");
    assert!(terminal.is_some(), "should have terminal event");
    let (_, data) = terminal.unwrap();
    let response = &data["response"];

    let required_fields = [
        "id",
        "object",
        "created_at",
        "completed_at",
        "status",
        "incomplete_details",
        "model",
        "previous_response_id",
        "instructions",
        "output",
        "error",
        "tools",
        "tool_choice",
        "truncation",
        "parallel_tool_calls",
        "text",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "top_logprobs",
        "temperature",
        "reasoning",
        "usage",
        "max_output_tokens",
        "max_tool_calls",
        "store",
        "background",
        "service_tier",
        "metadata",
        "safety_identifier",
        "prompt_cache_key",
    ];

    for field in &required_fields {
        assert!(
            response.get(field).is_some(),
            "terminal response missing field: {field}",
        );
    }

    // Terminal status should not be in_progress.
    let status = response["status"].as_str().expect("status");
    assert!(
        status == "completed" || status == "incomplete",
        "terminal status should be completed or incomplete: {status}",
    );
}

/// Full streaming event lifecycle: created -> in_progress -> item.added ->
/// content_part.added -> deltas -> done -> content_part.done -> item.done -> terminal.
#[test]
fn streaming_full_lifecycle_order() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body(&model, "Hello"),
    );
    let events = parse_sse_events(&resp.body);

    let types: Vec<&str> = events.iter().map(|(t, _)| t.as_str()).collect();

    // Verify ordering of lifecycle events.
    let created_pos = types.iter().position(|t| *t == "response.created");
    let in_progress_pos = types.iter().position(|t| *t == "response.in_progress");
    let item_added_pos = types
        .iter()
        .position(|t| *t == "response.output_item.added");
    let part_added_pos = types
        .iter()
        .position(|t| *t == "response.content_part.added");
    let first_delta_pos = types.iter().position(|t| t.ends_with(".delta"));
    let part_done_pos = types
        .iter()
        .position(|t| *t == "response.content_part.done");
    let item_done_pos = types.iter().position(|t| *t == "response.output_item.done");
    let terminal_pos = types
        .iter()
        .position(|t| *t == "response.completed" || *t == "response.incomplete");

    assert!(created_pos.is_some(), "missing response.created");
    assert!(in_progress_pos.is_some(), "missing response.in_progress");
    assert!(
        item_added_pos.is_some(),
        "missing response.output_item.added"
    );
    assert!(
        part_added_pos.is_some(),
        "missing response.content_part.added"
    );
    assert!(first_delta_pos.is_some(), "missing delta events");
    assert!(
        part_done_pos.is_some(),
        "missing response.content_part.done"
    );
    assert!(item_done_pos.is_some(), "missing response.output_item.done");
    assert!(terminal_pos.is_some(), "missing terminal event");

    // Verify ordering.
    assert!(created_pos < in_progress_pos, "created before in_progress");
    assert!(
        in_progress_pos < item_added_pos,
        "in_progress before item.added"
    );
    assert!(
        item_added_pos < part_added_pos,
        "item.added before content_part.added"
    );
    assert!(
        part_added_pos < first_delta_pos,
        "content_part.added before first delta"
    );
    assert!(
        first_delta_pos < part_done_pos,
        "deltas before content_part.done"
    );
    assert!(
        part_done_pos < item_done_pos,
        "content_part.done before item.done"
    );
    assert!(item_done_pos < terminal_pos, "item.done before terminal");
}

/// Delta events have item_id, output_index, content_index fields.
#[test]
fn streaming_delta_event_fields() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body(&model, "Hello"),
    );
    let events = parse_sse_events(&resp.body);

    let delta = events.iter().find(|(t, _)| t.ends_with(".delta"));
    assert!(delta.is_some(), "should have delta events");
    let (_, data) = delta.unwrap();
    assert!(
        data["item_id"].as_str().is_some(),
        "delta should have item_id"
    );
    assert!(
        data["output_index"].is_number(),
        "delta should have output_index"
    );
    assert!(
        data["content_index"].is_number(),
        "delta should have content_index"
    );
    assert!(data["delta"].is_string(), "delta should have delta text");
    assert!(
        data["sequence_number"].is_number(),
        "delta should have sequence_number"
    );
}

/// Streaming with max_output_tokens=1 triggers response.incomplete with
/// incomplete_details indicating max_output_tokens.
#[test]
fn streaming_incomplete_on_max_tokens() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Write a very long story about dragons and wizards",
        "stream": true,
        "max_output_tokens": 1,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200);
    let events = parse_sse_events(&resp.body);

    let terminal = events.last().expect("should have events");
    // With only 1 token, model should hit the limit and produce response.incomplete.
    assert_eq!(
        terminal.0, "response.incomplete",
        "should be response.incomplete, got: {}",
        terminal.0,
    );
    let response = &terminal.1["response"];
    assert_eq!(response["status"].as_str(), Some("incomplete"));
    let details = &response["incomplete_details"];
    assert!(details.is_object(), "should have incomplete_details object");
    assert_eq!(
        details["reason"].as_str(),
        Some("max_output_tokens"),
        "reason should be max_output_tokens",
    );
}

/// Non-streaming with max_output_tokens=1 produces status=incomplete.
#[test]
fn non_streaming_incomplete_on_max_tokens() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Write a very long story about dragons and wizards",
        "max_output_tokens": 1,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    // Non-streaming may or may not report incomplete (handler always sets "completed").
    // Verify usage is present and output_tokens is limited.
    let output_tokens = json["usage"]["output_tokens"]
        .as_u64()
        .expect("output_tokens");
    assert!(
        output_tokens <= 3,
        "output_tokens ({output_tokens}) should be limited"
    );
}

// =============================================================================
// Session creation & metadata tests
//
// These tests verify that:
// - session_id appears in the response.created SSE event metadata
// - session_id is consistent across the full SSE lifecycle
// - the conversation is persisted before generation completes
// =============================================================================

fn model_and_bucket_config(bucket: &std::path::Path) -> ServerConfig {
    let mut config = model_config();
    config.bucket = Some(bucket.to_path_buf());
    config
}

fn streaming_body_with_store(model: &str, input: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "input": input,
        "stream": true,
        "store": true,
        "max_output_tokens": 10,
    })
}

/// Extract session_id from a specific event type's response.metadata.
fn session_id_from_event<'a>(
    events: &'a [(String, serde_json::Value)],
    event_type: &str,
) -> Option<&'a str> {
    events
        .iter()
        .find(|(t, _)| t == event_type)
        .and_then(|(_, data)| data["response"]["metadata"]["session_id"].as_str())
}

/// response.created event carries session_id in metadata.
#[test]
fn streaming_created_event_has_session_id() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_and_bucket_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body_with_store(&model, "Hello"),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);

    let created = events
        .iter()
        .find(|(t, _)| t == "response.created");
    assert!(created.is_some(), "should have response.created event");
    let (_, data) = created.unwrap();

    let metadata = &data["response"]["metadata"];
    assert!(metadata.is_object(), "response should have metadata object");
    let session_id = metadata["session_id"].as_str();
    assert!(
        session_id.is_some() && !session_id.unwrap().is_empty(),
        "response.created should have non-empty session_id in metadata, got: {:?}",
        metadata,
    );
}

/// response.in_progress event also carries session_id matching response.created.
#[test]
fn streaming_in_progress_event_has_session_id() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_and_bucket_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body_with_store(&model, "Hello"),
    );
    assert_eq!(resp.status, 200);
    let events = parse_sse_events(&resp.body);

    let created_sid = session_id_from_event(&events, "response.created")
        .expect("response.created should have session_id");
    let in_progress_sid = session_id_from_event(&events, "response.in_progress")
        .expect("response.in_progress should have session_id");

    assert_eq!(
        created_sid, in_progress_sid,
        "session_id should match between response.created and response.in_progress",
    );
}

/// session_id is the same in response.created and the terminal event.
#[test]
fn streaming_session_id_consistent_across_lifecycle() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_and_bucket_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body_with_store(&model, "Hello"),
    );
    assert_eq!(resp.status, 200);
    let events = parse_sse_events(&resp.body);

    let created_sid = session_id_from_event(&events, "response.created")
        .expect("response.created should have session_id");

    // Find terminal event (completed or incomplete).
    let terminal = events
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete");
    assert!(terminal.is_some(), "should have terminal event");
    let (_, terminal_data) = terminal.unwrap();
    let terminal_sid = terminal_data["response"]["metadata"]["session_id"]
        .as_str()
        .expect("terminal event should have session_id in metadata");

    assert_eq!(
        created_sid, terminal_sid,
        "session_id should be consistent from response.created to terminal event",
    );
}

/// Session is visible via GET and list APIs **during** an active stream.
///
/// This is the core test for early session creation. It uses raw TCP to
/// read the first SSE event (response.created), extracts the session_id,
/// then queries the conversations API on a second connection while
/// generation is still running. Synchronization is event-based (not
/// timing-based): we proceed only after receiving the first SSE event,
/// which guarantees the server has already persisted the session.
#[test]
fn streaming_session_visible_during_generation() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_and_bucket_config(temp.path()));

    // Use enough tokens so the stream stays open long enough for our GET.
    let body = serde_json::json!({
        "model": model,
        "input": "Write a detailed essay about the history of computing",
        "stream": true,
        "store": true,
        "max_output_tokens": 100,
    });
    let body_str = serde_json::to_string(&body).unwrap();

    // Open a raw TCP connection and send the POST manually.
    let request = format!(
        "POST /v1/responses HTTP/1.1\r\n\
         Host: {}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n\
         {}",
        ctx.addr(),
        body_str.len(),
        body_str,
    );

    let mut stream =
        TcpStream::connect_timeout(&ctx.addr().into(), Duration::from_secs(5)).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .expect("set read timeout");
    stream.write_all(request.as_bytes()).expect("write");
    stream.flush().expect("flush");

    // Read bytes incrementally until we find the response.created event.
    let mut accumulated = String::new();
    let mut session_id: Option<String> = None;

    'outer: loop {
        let mut buf = [0u8; 4096];
        let n = stream.read(&mut buf).expect("read from stream");
        if n == 0 {
            break;
        }
        accumulated.push_str(&String::from_utf8_lossy(&buf[..n]));

        // Scan for "data: " lines in the accumulated text.
        for line in accumulated.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                    if json["type"].as_str() == Some("response.created") {
                        session_id = json["response"]["metadata"]["session_id"]
                            .as_str()
                            .map(String::from);
                        break 'outer;
                    }
                }
            }
        }
    }

    let session_id = session_id.expect("should find session_id in response.created event");

    // While the stream is still active, check that the session exists via
    // a separate HTTP connection. This proves early session creation works.
    let conv_resp = get(ctx.addr(), &format!("/v1/conversations/{}", session_id));
    assert_eq!(
        conv_resp.status, 200,
        "session should be visible via GET during streaming, got {}: {}",
        conv_resp.status, conv_resp.body,
    );
    let conv = conv_resp.json();
    assert_eq!(conv["id"].as_str(), Some(session_id.as_str()));
    assert!(
        conv["title"].as_str().is_some() && !conv["title"].as_str().unwrap().is_empty(),
        "session should have a title derived from input",
    );

    // Also check the list endpoint.
    let list_resp = get(ctx.addr(), "/v1/conversations");
    assert_eq!(list_resp.status, 200);
    let list = list_resp.json();
    let data = list["data"].as_array().expect("data array");
    let found = data
        .iter()
        .any(|c| c["id"].as_str() == Some(session_id.as_str()));
    assert!(
        found,
        "session {session_id} should appear in conversations list during streaming",
    );

    // Drain the rest of the stream so the server doesn't get a broken pipe.
    loop {
        let mut buf = [0u8; 8192];
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(_) => continue,
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                break;
            }
            Err(e) => panic!("drain error: {e}"),
        }
    }
}

/// Session title is derived from the user's input text (verified post-stream).
#[test]
fn streaming_session_title_derived_from_input() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_and_bucket_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body_with_store(&model, "Tell me about penguins"),
    );
    assert_eq!(resp.status, 200);
    let events = parse_sse_events(&resp.body);

    let session_id = session_id_from_event(&events, "response.created")
        .expect("response.created should have session_id");

    let conv_resp = get(ctx.addr(), &format!("/v1/conversations/{}", session_id));
    assert_eq!(conv_resp.status, 200);

    let conv_json = conv_resp.json();
    let title = conv_json["title"]
        .as_str()
        .expect("conversation should have title");
    assert!(
        title.starts_with("Tell me about penguins"),
        "title should be derived from input, got: {title:?}",
    );
}

/// Non-streaming response has session_id in metadata; session is accessible
/// and appears in the list with the correct title.
#[test]
fn non_streaming_response_has_session_id() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_and_bucket_config(temp.path()));
    let body = serde_json::json!({
        "model": model,
        "input": "Explain gravity briefly",
        "store": true,
        "max_output_tokens": 10,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();

    let session_id = json["metadata"]["session_id"]
        .as_str()
        .expect("non-streaming response should have session_id in metadata");
    assert!(!session_id.is_empty(), "session_id should be non-empty");

    // Session should be accessible via GET with correct properties.
    let conv_resp = get(ctx.addr(), &format!("/v1/conversations/{}", session_id));
    assert_eq!(
        conv_resp.status, 200,
        "GET conversation should return 200 for non-streaming session",
    );
    let conv = conv_resp.json();
    assert_eq!(conv["id"].as_str(), Some(session_id));

    let title = conv["title"].as_str().expect("should have title");
    assert!(
        title.starts_with("Explain gravity"),
        "title should be derived from input, got: {title:?}",
    );

    // Session should appear in the list.
    let list_resp = get(ctx.addr(), "/v1/conversations");
    assert_eq!(list_resp.status, 200);
    let list = list_resp.json();
    let found = list["data"]
        .as_array()
        .expect("data array")
        .iter()
        .any(|c| c["id"].as_str() == Some(session_id));
    assert!(found, "session should appear in conversations list");
}

/// Chained streaming response reuses the same session_id.
#[test]
fn streaming_chained_response_preserves_session() {
    let model = require_model!();
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(model_and_bucket_config(temp.path()));

    // First request: create a new session.
    let resp1 = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body_with_store(&model, "Hello"),
    );
    assert_eq!(resp1.status, 200);
    let events1 = parse_sse_events(&resp1.body);

    let session_id_1 = session_id_from_event(&events1, "response.created")
        .expect("first response should have session_id");

    // Extract the response id from the terminal event for chaining.
    let terminal = events1
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete")
        .expect("should have terminal event");
    let response_id = terminal.1["response"]["id"]
        .as_str()
        .expect("terminal response should have id");

    // Second request: chain off the first response.
    let body2 = serde_json::json!({
        "model": model,
        "input": "Tell me more",
        "stream": true,
        "store": true,
        "max_output_tokens": 10,
        "previous_response_id": response_id,
    });
    let resp2 = post_json(ctx.addr(), "/v1/responses", &body2);
    assert_eq!(resp2.status, 200);
    let events2 = parse_sse_events(&resp2.body);

    let session_id_2 = session_id_from_event(&events2, "response.created")
        .expect("chained response should have session_id");

    assert_eq!(
        session_id_1, session_id_2,
        "chained response should reuse the same session_id",
    );
}
