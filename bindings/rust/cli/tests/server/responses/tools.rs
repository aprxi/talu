//! `/v1/responses` tools and tool_choice round-trip tests.

use crate::server::common::{model_config, post_json, require_model, ServerTestContext};

fn tool_definition() -> serde_json::Value {
    serde_json::json!([{
        "type": "function",
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            },
            "required": ["city"]
        }
    }])
}

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

#[test]
fn responses_tools_echo_in_response() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 10,
        "tools": tool_definition()
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let tools = json["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["name"].as_str(), Some("get_weather"));
}

#[test]
fn responses_tool_choice_string_echo_and_defaults() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 10,
        "tools": tool_definition(),
        "tool_choice": "auto"
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["tool_choice"].as_str(), Some("auto"));

    let default_body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 10
    });
    let default_resp = post_json(ctx.addr(), "/v1/responses", &default_body);
    assert_eq!(default_resp.status, 200, "body: {}", default_resp.body);
    let json = default_resp.json();
    assert_eq!(json["tool_choice"].as_str(), Some("none"));
    assert_eq!(
        json["tools"].as_array().map(|a| a.len()),
        Some(0),
        "default tools should be empty array"
    );
}

#[test]
fn responses_object_tool_choice_echo_in_response() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 10,
        "tools": tool_definition(),
        "tool_choice": {
            "type": "function",
            "function": { "name": "get_weather" }
        }
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let tc = &resp.json()["tool_choice"];
    assert!(tc.is_object(), "tool_choice should be object, got: {tc}");
    assert_eq!(tc["type"].as_str(), Some("function"));
    assert_eq!(tc["function"]["name"].as_str(), Some("get_weather"));
}

#[test]
fn responses_rejects_invalid_object_tool_choice_shapes() {
    let ctx = ServerTestContext::new(crate::server::common::ServerConfig::new());
    let missing_name = serde_json::json!({
        "input": "hello",
        "tool_choice": {
            "type": "function"
        }
    });
    let missing_name_resp = post_json(ctx.addr(), "/v1/responses", &missing_name);
    assert_eq!(
        missing_name_resp.status, 400,
        "body: {}",
        missing_name_resp.body
    );

    let invalid_allowed_tool_entry = serde_json::json!({
        "input": "hello",
        "tool_choice": {
            "type": "allowed_tools",
            "tools": [{ "type": "not_function", "name": "lookup" }],
            "mode": "auto"
        }
    });
    let invalid_allowed_tool_entry_resp =
        post_json(ctx.addr(), "/v1/responses", &invalid_allowed_tool_entry);
    assert_eq!(
        invalid_allowed_tool_entry_resp.status, 400,
        "body: {}",
        invalid_allowed_tool_entry_resp.body
    );
}

#[test]
fn responses_tools_chaining_merge_and_override() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let first = serde_json::json!({
        "model": &model,
        "input": "Hello",
        "max_output_tokens": 10,
        "tools": tool_definition(),
        "tool_choice": "auto"
    });
    let first_resp = post_json(ctx.addr(), "/v1/responses", &first);
    assert_eq!(first_resp.status, 200, "body: {}", first_resp.body);
    let response_id = first_resp.json()["id"]
        .as_str()
        .expect("response id")
        .to_string();

    let merged = serde_json::json!({
        "model": &model,
        "input": "What tools do you have?",
        "max_output_tokens": 10,
        "previous_response_id": response_id
    });
    let merged_resp = post_json(ctx.addr(), "/v1/responses", &merged);
    assert_eq!(merged_resp.status, 200, "body: {}", merged_resp.body);
    let merged_json = merged_resp.json();
    assert_eq!(
        merged_json["tools"].as_array().map(|v| v.len()),
        Some(1),
        "chained request should inherit tools"
    );
    assert_eq!(merged_json["tool_choice"].as_str(), Some("auto"));

    let override_tools = serde_json::json!([{
        "type": "function",
        "name": "search",
        "description": "Search the web",
        "parameters": { "type": "object", "properties": {} }
    }]);
    let overridden = serde_json::json!({
        "model": &model,
        "input": "Search for something",
        "max_output_tokens": 10,
        "previous_response_id": merged_json["id"].as_str().expect("response id"),
        "tools": override_tools,
        "tool_choice": "required"
    });
    let overridden_resp = post_json(ctx.addr(), "/v1/responses", &overridden);
    assert_eq!(
        overridden_resp.status, 200,
        "body: {}",
        overridden_resp.body
    );
    let json = overridden_resp.json();
    let tools = json["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["name"].as_str(), Some("search"));
    assert_eq!(json["tool_choice"].as_str(), Some("required"));
}

#[test]
fn responses_streaming_events_include_tools_on_created_and_terminal() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "stream": true,
        "max_output_tokens": 10,
        "tools": tool_definition(),
        "tool_choice": "auto"
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let events = parse_sse_events(&resp.body);

    let created = events
        .iter()
        .find(|(t, _)| t == "response.created")
        .map(|(_, e)| &e["response"])
        .expect("missing response.created event");
    let created_tools = created["tools"].as_array().expect("created tools array");
    assert_eq!(created_tools.len(), 1);
    assert_eq!(created["tool_choice"].as_str(), Some("auto"));

    let terminal = events
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete")
        .map(|(_, e)| &e["response"])
        .expect("missing terminal response event");
    let terminal_tools = terminal["tools"].as_array().expect("terminal tools array");
    assert_eq!(terminal_tools.len(), 1);
    assert_eq!(terminal["tool_choice"].as_str(), Some("auto"));
}
