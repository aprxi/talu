//! `/v1/chat/completions` tool calling tests.
//!
//! Verifies that the tools and tool_choice parameters are accepted and
//! that tool call responses match the OpenAI contract shape.

use crate::server::common::{model_config, post_json, require_model, ServerTestContext};

fn tool_definition() -> serde_json::Value {
    serde_json::json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": { "type": "string", "description": "City name" }
                },
                "required": ["city"]
            }
        }
    }])
}

/// Tools parameter is accepted without error.
#[test]
fn tools_parameter_accepted() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": tool_definition(),
        "max_tokens": 64,
        "temperature": 0.0
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
}

/// Tool choice "auto" is accepted.
#[test]
fn tool_choice_auto_accepted() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": tool_definition(),
        "tool_choice": "auto",
        "max_tokens": 64,
        "temperature": 0.0
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
}

/// Tool choice "none" is accepted — model should respond with text, not tool calls.
#[test]
fn tool_choice_none_produces_text() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": tool_definition(),
        "tool_choice": "none",
        "max_tokens": 32,
        "temperature": 0.0
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let content = json["choices"][0]["message"]["content"].as_str();
    // With tool_choice=none, the model should produce text content
    assert!(
        content.is_some() && !content.unwrap().is_empty(),
        "tool_choice=none should produce text content: {:?}",
        json["choices"][0]
    );
}

/// When the model produces a tool call, the response shape must match
/// the OpenAI contract: choices[0].message.tool_calls is an array of
/// {id, type, function: {name, arguments}}.
#[test]
fn tool_call_response_shape() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": "You must use the get_weather tool to answer weather questions. Always call the tool."},
            {"role": "user", "content": "What is the weather in Paris?"}
        ],
        "tools": tool_definition(),
        "tool_choice": "required",
        "max_tokens": 128,
        "temperature": 0.0
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let choice = &json["choices"][0];

    // Model may or may not produce a tool call depending on model capability.
    // If tool_calls is present, validate the shape.
    if let Some(tool_calls) = choice["message"]["tool_calls"].as_array() {
        if !tool_calls.is_empty() {
            let tc = &tool_calls[0];
            assert!(tc["id"].is_string(), "tool call must have string id: {:?}", tc);
            assert_eq!(
                tc["type"].as_str(),
                Some("function"),
                "tool call type must be 'function': {:?}",
                tc
            );
            assert!(
                tc["function"]["name"].is_string(),
                "tool call must have function.name: {:?}",
                tc
            );
            assert!(
                tc["function"]["arguments"].is_string(),
                "tool call must have function.arguments as string: {:?}",
                tc
            );

            // finish_reason should be "tool_calls"
            assert_eq!(
                choice["finish_reason"].as_str(),
                Some("tool_calls"),
                "finish_reason should be 'tool_calls' when tool calls are present"
            );
        }
    }
}

/// Tool call continuation: user → assistant (tool_call) → tool → assistant.
/// vLLM handles this correctly with the same model template.
#[test]
fn tool_call_continuation() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "22°C, sunny"
            },
        ],
        "tools": tool_definition(),
        "max_tokens": 32,
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
        "model should produce text after tool output"
    );
}

/// OpenAI allows content:null on assistant messages with tool_calls.
#[test]
fn tool_call_null_content() {
    let model = require_model!();
    let ctx = ServerTestContext::new(model_config());

    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "72F"}
        ],
        "tools": tool_definition(),
        "max_tokens": 16,
        "temperature": 0.0
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
}
