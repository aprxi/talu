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
