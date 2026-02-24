use crate::server::common::{model_config, post_json, require_model, ServerTestContext};

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

fn is_allowed_responses_stream_event(event_type: &str) -> bool {
    matches!(
        event_type,
        "response.queued"
            | "response.created"
            | "response.in_progress"
            | "response.output_item.added"
            | "response.content_part.added"
            | "response.output_text.delta"
            | "response.output_text.done"
            | "response.refusal.delta"
            | "response.refusal.done"
            | "response.reasoning.delta"
            | "response.reasoning.done"
            | "response.reasoning_summary_part.added"
            | "response.reasoning_summary_part.done"
            | "response.reasoning_summary_text.delta"
            | "response.reasoning_summary_text.done"
            | "response.function_call_arguments.delta"
            | "response.function_call_arguments.done"
            | "response.output_text.annotation.added"
            | "response.output_text.annotation.done"
            | "response.code_interpreter_call.code.delta"
            | "response.code_interpreter_call.code.done"
            | "response.code_interpreter_call.interpreting"
            | "response.code_interpreter_call.completed"
            | "response.code_interpreter_call.in_progress"
            | "response.image_generation_call.completed"
            | "response.image_generation_call.generating"
            | "response.image_generation_call.in_progress"
            | "response.web_search_call.completed"
            | "response.web_search_call.in_progress"
            | "response.web_search_call.searching"
            | "response.file_search_call.completed"
            | "response.file_search_call.in_progress"
            | "response.file_search_call.searching"
            | "response.content_part.done"
            | "response.output_item.done"
            | "response.completed"
            | "response.incomplete"
            | "response.failed"
            | "error"
    )
}

fn streaming_body(model: &str, input: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "input": input,
        "stream": true,
        "max_output_tokens": 10
    })
}

fn validate_stream_state_machine(events: &[(String, serde_json::Value)]) -> Result<(), String> {
    let mut saw_queued = false;
    let mut saw_created = false;
    let mut saw_in_progress = false;
    let mut saw_terminal = false;
    let mut item_open = false;
    let mut part_open = false;

    for (event_type, _event) in events {
        match event_type.as_str() {
            "response.queued" => {
                if saw_queued {
                    return Err("duplicate response.queued".to_string());
                }
                saw_queued = true;
            }
            "response.created" => {
                if !saw_queued {
                    return Err("response.created before response.queued".to_string());
                }
                saw_created = true;
            }
            "response.in_progress" => {
                if !saw_created {
                    return Err("response.in_progress before response.created".to_string());
                }
                saw_in_progress = true;
            }
            "response.output_item.added" => {
                if item_open {
                    return Err("new output item opened before closing previous item".to_string());
                }
                item_open = true;
                part_open = false;
            }
            "response.content_part.added" => {
                if !item_open {
                    return Err("content part opened without an active output item".to_string());
                }
                if part_open {
                    return Err("content part opened before previous part was done".to_string());
                }
                part_open = true;
            }
            "response.output_text.delta"
            | "response.reasoning.delta"
            | "response.reasoning_summary_text.delta"
            | "response.refusal.delta"
            | "response.function_call_arguments.delta" => {
                if !part_open {
                    return Err(format!("{event_type} emitted without open content part"));
                }
            }
            "response.output_text.done"
            | "response.reasoning.done"
            | "response.reasoning_summary_text.done"
            | "response.refusal.done"
            | "response.function_call_arguments.done" => {
                if !part_open {
                    return Err(format!("{event_type} emitted without open content part"));
                }
            }
            "response.content_part.done" => {
                if !part_open {
                    return Err("response.content_part.done without open part".to_string());
                }
                part_open = false;
            }
            "response.output_item.done" => {
                if !item_open {
                    return Err("response.output_item.done without open item".to_string());
                }
                if part_open {
                    return Err(
                        "response.output_item.done while content part still open".to_string()
                    );
                }
                item_open = false;
            }
            "response.completed" | "response.incomplete" | "response.failed" => {
                if !saw_in_progress {
                    return Err(format!("{event_type} before response.in_progress"));
                }
                saw_terminal = true;
                if item_open || part_open {
                    return Err(format!("{event_type} emitted while item/part still open"));
                }
            }
            "error" => {}
            _ => {}
        }
    }

    if !saw_queued {
        return Err("missing response.queued".to_string());
    }
    if !saw_created {
        return Err("missing response.created".to_string());
    }
    if !saw_in_progress {
        return Err("missing response.in_progress".to_string());
    }
    if !saw_terminal {
        return Err("missing terminal event".to_string());
    }
    Ok(())
}

#[test]
fn responses_stream_emits_queued_and_omits_progress() {
    let model = require_model!();
    let mut cfg = model_config();
    cfg.model = Some(model.clone());
    let ctx = ServerTestContext::new(cfg);
    let body = serde_json::json!({
        "model": model,
        "input": "hello",
        "stream": true,
        "max_output_tokens": 12
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);
    assert!(!events.is_empty(), "stream should emit events");
    assert_eq!(events[0].0, "response.queued");
    assert!(events.iter().any(|(t, _)| t == "response.created"));
    assert!(events.iter().any(|(t, _)| t == "response.in_progress"));
    assert!(
        events.iter().all(|(t, _)| t != "response.progress"),
        "strict /v1/responses should not expose non-spec progress events"
    );
}

#[test]
fn responses_stream_does_not_inject_session_metadata() {
    let model = require_model!();
    let mut cfg = model_config();
    cfg.model = Some(model.clone());
    let ctx = ServerTestContext::new(cfg);
    let body = serde_json::json!({
        "model": model,
        "input": "hello",
        "stream": true,
        "max_output_tokens": 12
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);

    let created = events
        .iter()
        .find(|(t, _)| t == "response.created")
        .map(|(_, e)| &e["response"])
        .expect("missing response.created");
    assert!(
        created["metadata"]
            .as_object()
            .map(|m| !m.contains_key("session_id"))
            .unwrap_or(false),
        "created metadata should not contain chat session_id",
    );

    let terminal = events
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete")
        .map(|(_, e)| &e["response"])
        .expect("missing terminal response event");
    assert!(
        terminal["metadata"]
            .as_object()
            .map(|m| !m.contains_key("session_id"))
            .unwrap_or(false),
        "terminal metadata should not contain chat session_id",
    );
}

#[test]
fn responses_stream_sequence_obeys_state_machine() {
    let model = require_model!();
    let mut cfg = model_config();
    cfg.model = Some(model.clone());
    let ctx = ServerTestContext::new(cfg);
    let body = serde_json::json!({
        "model": model,
        "input": "hello",
        "stream": true,
        "max_output_tokens": 16
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);
    assert!(
        validate_stream_state_machine(&events).is_ok(),
        "invalid event sequence: {:#?}",
        events
    );
}

#[test]
fn responses_stream_delta_obfuscation_fields_are_spec_compatible() {
    let model = require_model!();
    let mut cfg = model_config();
    cfg.model = Some(model.clone());
    let ctx = ServerTestContext::new(cfg);
    let body = serde_json::json!({
        "model": model,
        "input": "hello",
        "stream": true,
        "max_output_tokens": 12
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);

    for (event_type, event) in events {
        match event_type.as_str() {
            "response.output_text.delta"
            | "response.reasoning.delta"
            | "response.reasoning_summary_text.delta"
            | "response.function_call_arguments.delta" => {
                assert!(
                    event
                        .get("obfuscation")
                        .map(|v| v.is_string())
                        .unwrap_or(false),
                    "{event_type} must include string obfuscation field: {event}"
                );
            }
            _ => {}
        }
    }
}

#[test]
fn responses_stream_includes_instructions_in_response_resources() {
    let model = require_model!();
    let mut cfg = model_config();
    cfg.model = Some(model.clone());
    let ctx = ServerTestContext::new(cfg);
    let body = serde_json::json!({
        "model": model,
        "input": "hello",
        "instructions": "be concise",
        "stream": true,
        "max_output_tokens": 12
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);

    let created = events
        .iter()
        .find(|(t, _)| t == "response.created")
        .map(|(_, e)| &e["response"])
        .expect("missing response.created");
    assert_eq!(created["instructions"].as_str(), Some("be concise"));

    let terminal = events
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete")
        .map(|(_, e)| &e["response"])
        .expect("missing terminal response");
    assert_eq!(terminal["instructions"].as_str(), Some("be concise"));
}

#[test]
fn responses_stream_in_progress_event_has_valid_response_shape() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body(&model, "hello"),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);

    let in_progress = events
        .iter()
        .find(|(t, _)| t == "response.in_progress")
        .map(|(_, e)| &e["response"])
        .expect("missing response.in_progress event");
    assert_eq!(in_progress["status"].as_str(), Some("in_progress"));
    assert_eq!(in_progress["object"].as_str(), Some("response"));
}

#[test]
fn responses_stream_item_and_part_events_have_required_fields() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body(&model, "hello"),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);

    let item_added = events
        .iter()
        .find(|(t, _)| t == "response.output_item.added")
        .map(|(_, e)| e)
        .expect("missing response.output_item.added");
    assert!(item_added["item"].is_object(), "item should be object");
    assert!(item_added["item"]["id"].is_string(), "item should have id");
    assert!(
        item_added["output_index"].is_number(),
        "output_index should be number"
    );

    let part_added = events
        .iter()
        .find(|(t, _)| t == "response.content_part.added")
        .map(|(_, e)| e)
        .expect("missing response.content_part.added");
    assert!(part_added["part"].is_object(), "part should be object");
    assert!(
        part_added["part"]["type"].is_string(),
        "part should have type"
    );
    assert!(
        part_added["item_id"].is_string(),
        "item_id should be present"
    );
    assert!(
        part_added["output_index"].is_number(),
        "output_index should be number"
    );
    assert!(
        part_added["content_index"].is_number(),
        "content_index should be number"
    );

    let part_done = events
        .iter()
        .find(|(t, _)| t == "response.content_part.done")
        .map(|(_, e)| e)
        .expect("missing response.content_part.done");
    assert!(part_done["part"].is_object(), "done part should be object");
    assert!(
        part_done["item_id"].is_string(),
        "done part should include item_id"
    );

    let item_done = events
        .iter()
        .find(|(t, _)| t == "response.output_item.done")
        .map(|(_, e)| e)
        .expect("missing response.output_item.done");
    assert!(item_done["item"].is_object(), "done item should be object");
    assert!(
        item_done["item"]["id"].is_string(),
        "done item should have id"
    );
    assert!(
        item_done["item"]["status"].is_string(),
        "done item should have status"
    );
}

#[test]
fn responses_stream_delta_events_have_indices_and_sequence() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body(&model, "hello"),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);

    let delta = events
        .iter()
        .find(|(t, _)| t.ends_with(".delta"))
        .map(|(_, e)| e)
        .expect("missing delta event");
    assert!(delta["item_id"].is_string(), "delta should include item_id");
    assert!(
        delta["output_index"].is_number(),
        "delta should include output_index"
    );
    assert!(
        delta["content_index"].is_number(),
        "delta should include content_index"
    );
    assert!(
        delta["sequence_number"].is_number(),
        "delta should include sequence_number"
    );
}

#[test]
fn responses_stream_full_lifecycle_order() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body(&model, "hello"),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);
    let types: Vec<&str> = events.iter().map(|(t, _)| t.as_str()).collect();

    let queued_pos = types.iter().position(|t| *t == "response.queued");
    let created_pos = types.iter().position(|t| *t == "response.created");
    let in_progress_pos = types.iter().position(|t| *t == "response.in_progress");
    let item_added_pos = types
        .iter()
        .position(|t| *t == "response.output_item.added");
    let part_added_pos = types
        .iter()
        .position(|t| *t == "response.content_part.added");
    let delta_pos = types.iter().position(|t| t.ends_with(".delta"));
    let part_done_pos = types
        .iter()
        .position(|t| *t == "response.content_part.done");
    let item_done_pos = types.iter().position(|t| *t == "response.output_item.done");
    let terminal_pos = types
        .iter()
        .position(|t| *t == "response.completed" || *t == "response.incomplete");

    assert!(queued_pos.is_some(), "missing response.queued");
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
    assert!(delta_pos.is_some(), "missing delta event");
    assert!(
        part_done_pos.is_some(),
        "missing response.content_part.done"
    );
    assert!(item_done_pos.is_some(), "missing response.output_item.done");
    assert!(terminal_pos.is_some(), "missing terminal event");

    assert!(queued_pos < created_pos);
    assert!(created_pos < in_progress_pos);
    assert!(in_progress_pos < item_added_pos);
    assert!(item_added_pos < part_added_pos);
    assert!(part_added_pos < delta_pos);
    assert!(delta_pos < part_done_pos);
    assert!(part_done_pos < item_done_pos);
    assert!(item_done_pos < terminal_pos);
}

#[test]
fn responses_stream_terminal_resource_has_required_fields() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body(&model, "hello"),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);

    let terminal = events
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete")
        .map(|(_, e)| &e["response"])
        .expect("missing terminal response event");

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
    for field in required_fields {
        assert!(
            terminal.get(field).is_some(),
            "terminal response missing required field: {field}"
        );
    }
    let status = terminal["status"].as_str().expect("terminal status");
    assert!(
        status == "completed" || status == "incomplete",
        "terminal status should be completed or incomplete, got {status}"
    );
}

#[test]
fn responses_stream_incomplete_event_contains_incomplete_details() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Write a very long story about dragons and wizards",
        "stream": true,
        "max_output_tokens": 1
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);

    let terminal = events.last().expect("missing stream events");
    if terminal.0 == "response.incomplete" {
        assert_eq!(
            terminal.1["response"]["status"].as_str(),
            Some("incomplete")
        );
        assert_eq!(
            terminal.1["response"]["incomplete_details"]["reason"].as_str(),
            Some("max_output_tokens")
        );
    } else {
        assert_eq!(
            terminal.0, "response.completed",
            "terminal event should be completed or incomplete, got {}",
            terminal.0
        );
    }
}

#[test]
fn responses_stream_emits_only_spec_event_types() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body(&model, "hello"),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);
    assert!(!events.is_empty(), "stream should emit events");
    for (event_type, _) in events {
        assert!(
            is_allowed_responses_stream_event(&event_type),
            "non-spec stream event emitted on /v1/responses: {event_type}"
        );
    }
}

#[test]
fn responses_stream_sequence_numbers_are_monotonic() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body(&model, "hello"),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);

    let mut previous: Option<i64> = None;
    for (_, e) in events {
        let Some(current) = e.get("sequence_number").and_then(|v| v.as_i64()) else {
            continue;
        };
        if let Some(prev) = previous {
            assert!(
                current > prev,
                "sequence_number must be strictly increasing, got {prev} then {current}"
            );
        }
        previous = Some(current);
    }
}

#[test]
fn responses_stream_created_and_terminal_response_ids_match() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/responses",
        &streaming_body(&model, "hello"),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);

    let created_id = events
        .iter()
        .find(|(t, _)| t == "response.created")
        .and_then(|(_, e)| e["response"]["id"].as_str())
        .expect("created event must include response.id");
    let terminal_id = events
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete")
        .and_then(|(_, e)| e["response"]["id"].as_str())
        .expect("terminal event must include response.id");
    assert_eq!(
        created_id, terminal_id,
        "created and terminal response ids must match for one stream"
    );
}
