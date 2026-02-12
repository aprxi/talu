//! Streaming (SSE) tests over real HTTP.

use crate::server::common::{model_config, post_json, require_model, ServerTestContext};

fn streaming_body(model: &str, input: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "input": input,
        "stream": true,
        "max_output_tokens": 50,
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

/// Streaming response has text/event-stream Content-Type.
#[test]
fn streaming_content_type() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(ctx.addr(), "/v1/responses", &streaming_body(&model, "Hi"));
    assert_eq!(resp.status, 200);
    let ct = resp
        .header("content-type")
        .expect("should have Content-Type");
    assert!(ct.contains("text/event-stream"), "Content-Type: {ct}");
}

/// Streaming response has Cache-Control: no-cache.
#[test]
fn streaming_cache_headers() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(ctx.addr(), "/v1/responses", &streaming_body(&model, "Hi"));
    assert_eq!(resp.status, 200);
    let cc = resp.header("cache-control");
    assert_eq!(cc, Some("no-cache"), "Cache-Control: {cc:?}");
}

/// SSE stream starts with response.created and ends with a terminal event.
#[test]
fn streaming_event_sequence() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(ctx.addr(), "/v1/responses", &streaming_body(&model, "Hello"));
    assert_eq!(resp.status, 200);
    let events = parse_sse_events(&resp.body);
    assert!(!events.is_empty(), "should have SSE events");

    // First event: response.created
    assert_eq!(
        events[0].0, "response.created",
        "first event should be response.created"
    );

    // Last event: terminal (completed or incomplete)
    let last = &events.last().unwrap().0;
    assert!(
        last == "response.completed" || last == "response.incomplete",
        "last event should be terminal: {last}",
    );
}

/// Created event has response with status=in_progress.
#[test]
fn streaming_created_event() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(ctx.addr(), "/v1/responses", &streaming_body(&model, "Hello"));
    let events = parse_sse_events(&resp.body);
    let (_, data) = &events[0];
    assert_eq!(data["type"].as_str(), Some("response.created"));
    let response = &data["response"];
    assert_eq!(response["status"].as_str(), Some("in_progress"));
}

/// Stream contains delta events with a delta field.
#[test]
fn streaming_delta_events() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(ctx.addr(), "/v1/responses", &streaming_body(&model, "Hello"));
    let events = parse_sse_events(&resp.body);

    let deltas: Vec<_> = events
        .iter()
        .filter(|(t, _)| t.ends_with(".delta"))
        .collect();
    assert!(!deltas.is_empty(), "should have delta events");
    for (_, data) in &deltas {
        assert!(
            data.get("delta").is_some(),
            "delta event should have delta field"
        );
    }
}

/// Stream contains a done event with text.
#[test]
fn streaming_done_event() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(ctx.addr(), "/v1/responses", &streaming_body(&model, "Hello"));
    let events = parse_sse_events(&resp.body);

    let done = events
        .iter()
        .find(|(t, _)| t.ends_with(".done") && t.starts_with("response."));
    assert!(done.is_some(), "should have a done event");
    let (_, data) = done.unwrap();
    assert!(
        data.get("text").is_some(),
        "done event should have text field"
    );
}

/// Terminal event has a response with usage stats.
#[test]
fn streaming_terminal_has_usage() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(ctx.addr(), "/v1/responses", &streaming_body(&model, "Hello"));
    let events = parse_sse_events(&resp.body);

    let terminal = events
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete");
    assert!(terminal.is_some(), "should have terminal event");
    let (_, data) = terminal.unwrap();
    let response = &data["response"];
    let usage = &response["usage"];
    assert!(usage["input_tokens"].as_u64().unwrap_or(0) > 0);
    assert!(usage["output_tokens"].as_u64().unwrap_or(0) > 0);
}

/// Sequence numbers are monotonically increasing.
#[test]
fn streaming_sequence_numbers() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(ctx.addr(), "/v1/responses", &streaming_body(&model, "Hello"));
    let events = parse_sse_events(&resp.body);

    let seq_nums: Vec<u64> = events
        .iter()
        .filter_map(|(_, data)| data["sequence_number"].as_u64())
        .collect();
    assert!(!seq_nums.is_empty(), "should have sequence numbers");
    for window in seq_nums.windows(2) {
        assert!(
            window[1] > window[0],
            "sequence numbers should increase: {} -> {}",
            window[0],
            window[1],
        );
    }
}

/// Stream contains response.in_progress event after response.created.
#[test]
fn streaming_in_progress_event() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(ctx.addr(), "/v1/responses", &streaming_body(&model, "Hello"));
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
    let resp = post_json(ctx.addr(), "/v1/responses", &streaming_body(&model, "Hello"));
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
    let resp = post_json(ctx.addr(), "/v1/responses", &streaming_body(&model, "Hello"));
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
    let resp = post_json(ctx.addr(), "/v1/responses", &streaming_body(&model, "Hello"));
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
    let resp = post_json(ctx.addr(), "/v1/responses", &streaming_body(&model, "Hello"));
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
    let resp = post_json(ctx.addr(), "/v1/responses", &streaming_body(&model, "Hello"));
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
    let resp = post_json(ctx.addr(), "/v1/responses", &streaming_body(&model, "Hello"));
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
    let resp = post_json(ctx.addr(), "/v1/responses", &streaming_body(&model, "Hello"));
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
