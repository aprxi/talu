//! Tool calling passthrough and round-trip tests.

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

/// Tools array is echoed back in the response resource.
#[test]
fn tools_echo_in_response() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let tools = tool_definition();
    let body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 10,
        "tools": tools,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let resp_tools = json["tools"].as_array().expect("tools should be array");
    assert_eq!(resp_tools.len(), 1, "should have 1 tool");
    assert_eq!(resp_tools[0]["name"].as_str(), Some("get_weather"));
}

/// tool_choice is echoed back in the response resource.
#[test]
fn tool_choice_echo_in_response() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 10,
        "tools": tool_definition(),
        "tool_choice": "auto",
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["tool_choice"].as_str(), Some("auto"));
}

/// Default tool_choice (when not specified) is "none".
#[test]
fn default_tool_choice() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 10,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200);
    let json = resp.json();
    assert_eq!(json["tool_choice"].as_str(), Some("none"));
}

/// Default tools (when not specified) is an empty array.
#[test]
fn default_tools_empty() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "max_output_tokens": 10,
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let tools = json["tools"].as_array().expect("tools array");
    assert!(tools.is_empty(), "default tools should be empty");
}

/// Tools are round-tripped through conversation chaining.
#[test]
fn tools_chaining_merge() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    // First request with tools.
    let body1 = serde_json::json!({
        "model": &model,
        "input": "Hello",
        "max_output_tokens": 10,
        "tools": tool_definition(),
        "tool_choice": "auto",
    });
    let resp1 = post_json(ctx.addr(), "/v1/responses", &body1);
    assert_eq!(resp1.status, 200);
    let json1 = resp1.json();
    let response_id = json1["id"].as_str().expect("id");

    // Chained request WITHOUT tools — should inherit from previous.
    let body2 = serde_json::json!({
        "model": &model,
        "input": "What tools do you have?",
        "max_output_tokens": 10,
        "previous_response_id": response_id,
    });
    let resp2 = post_json(ctx.addr(), "/v1/responses", &body2);
    assert_eq!(resp2.status, 200, "body: {}", resp2.body);
    let json2 = resp2.json();
    let resp_tools = json2["tools"].as_array().expect("tools array");
    assert_eq!(resp_tools.len(), 1, "chained request should inherit tools");
    assert_eq!(resp_tools[0]["name"].as_str(), Some("get_weather"));
    assert_eq!(
        json2["tool_choice"].as_str(),
        Some("auto"),
        "should inherit tool_choice"
    );
}

/// Chained request can override tools from previous response.
#[test]
fn tools_chaining_override() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    // First request with tools.
    let body1 = serde_json::json!({
        "model": &model,
        "input": "Hello",
        "max_output_tokens": 10,
        "tools": tool_definition(),
    });
    let resp1 = post_json(ctx.addr(), "/v1/responses", &body1);
    assert_eq!(resp1.status, 200);
    let json1 = resp1.json();
    let response_id = json1["id"].as_str().expect("id");

    // Chained request with different tools — should override.
    let new_tools = serde_json::json!([{
        "type": "function",
        "name": "search",
        "description": "Search the web",
        "parameters": { "type": "object", "properties": {} }
    }]);
    let body2 = serde_json::json!({
        "model": &model,
        "input": "Search for something",
        "max_output_tokens": 10,
        "previous_response_id": response_id,
        "tools": new_tools,
        "tool_choice": "required",
    });
    let resp2 = post_json(ctx.addr(), "/v1/responses", &body2);
    assert_eq!(resp2.status, 200, "body: {}", resp2.body);
    let json2 = resp2.json();
    let resp_tools = json2["tools"].as_array().expect("tools array");
    assert_eq!(resp_tools.len(), 1);
    assert_eq!(
        resp_tools[0]["name"].as_str(),
        Some("search"),
        "should use overridden tools"
    );
    assert_eq!(
        json2["tool_choice"].as_str(),
        Some("required"),
        "should use overridden tool_choice"
    );
}

/// Streaming with tools echoes them in the terminal response resource.
#[test]
fn streaming_tools_echo() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "stream": true,
        "max_output_tokens": 10,
        "tools": tool_definition(),
        "tool_choice": "auto",
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200);

    // Find terminal event and check tools in its response resource.
    let events = parse_sse_events(&resp.body);
    let terminal = events
        .iter()
        .find(|(t, _)| t == "response.completed" || t == "response.incomplete");
    assert!(terminal.is_some(), "should have terminal event");
    let (_, data) = terminal.unwrap();
    let response = &data["response"];
    let tools = response["tools"].as_array().expect("tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["name"].as_str(), Some("get_weather"));
    assert_eq!(response["tool_choice"].as_str(), Some("auto"));
}

/// Streaming created event includes tools in the response resource.
#[test]
fn streaming_created_includes_tools() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());
    let body = serde_json::json!({
        "model": model,
        "input": "Hello",
        "stream": true,
        "max_output_tokens": 10,
        "tools": tool_definition(),
        "tool_choice": "auto",
    });
    let resp = post_json(ctx.addr(), "/v1/responses", &body);
    assert_eq!(resp.status, 200);
    let events = parse_sse_events(&resp.body);
    let (_, data) = &events[0];
    let response = &data["response"];
    let tools = response["tools"].as_array().expect("tools");
    assert_eq!(tools.len(), 1, "created event should include tools");
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
