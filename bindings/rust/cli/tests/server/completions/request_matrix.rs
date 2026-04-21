//! Broad request-shape/type coverage for `/v1/chat/completions`.

use crate::server::common::{post_json, ServerConfig, ServerTestContext};

#[derive(Clone, Copy)]
enum Expectation {
    InvalidRequest,
    ValidNonStreamingNoBackend,
    ValidStreamingNoBackend,
}

struct Case {
    name: &'static str,
    body: serde_json::Value,
    expect: Expectation,
}

fn run_case(ctx: &ServerTestContext, case: &Case) -> Result<(), String> {
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &case.body);
    match case.expect {
        Expectation::InvalidRequest => assert_invalid_request(case, &resp)?,
        Expectation::ValidNonStreamingNoBackend => assert_server_error(case, &resp)?,
        Expectation::ValidStreamingNoBackend => assert_streaming_server_error(case, &resp)?,
    }
    Ok(())
}

fn assert_invalid_request(
    case: &Case,
    resp: &crate::server::common::HttpResponse,
) -> Result<(), String> {
    if resp.status != 400 {
        return Err(format!(
            "[{}] expected 400 invalid_request_error got {} body={}",
            case.name, resp.status, resp.body
        ));
    }

    let ct = resp.header("content-type").unwrap_or("");
    if !ct.contains("application/json") {
        return Err(format!(
            "[{}] expected JSON content-type for invalid request, got {} body={}",
            case.name, ct, resp.body
        ));
    }

    let json = resp.json();
    if json["error"]["type"].as_str() != Some("invalid_request_error") {
        return Err(format!(
            "[{}] expected error.type=invalid_request_error, got body={}",
            case.name, resp.body
        ));
    }
    if !json["error"]["message"].is_string() {
        return Err(format!(
            "[{}] expected error.message string, got body={}",
            case.name, resp.body
        ));
    }

    Ok(())
}

fn assert_server_error(
    case: &Case,
    resp: &crate::server::common::HttpResponse,
) -> Result<(), String> {
    if resp.status != 500 {
        return Err(format!(
            "[{}] expected 500 server_error (valid request without backend) got {} body={}",
            case.name, resp.status, resp.body
        ));
    }

    let ct = resp.header("content-type").unwrap_or("");
    if !ct.contains("application/json") {
        return Err(format!(
            "[{}] expected JSON content-type for server error, got {} body={}",
            case.name, ct, resp.body
        ));
    }

    let json = resp.json();
    if json["error"]["type"].as_str() != Some("server_error") {
        return Err(format!(
            "[{}] expected error.type=server_error, got body={}",
            case.name, resp.body
        ));
    }
    if !json["error"]["message"].is_string() {
        return Err(format!(
            "[{}] expected error.message string, got body={}",
            case.name, resp.body
        ));
    }

    Ok(())
}

fn assert_streaming_server_error(
    case: &Case,
    resp: &crate::server::common::HttpResponse,
) -> Result<(), String> {
    if resp.status != 200 {
        return Err(format!(
            "[{}] expected 200 SSE stream got {} body={}",
            case.name, resp.status, resp.body
        ));
    }

    let ct = resp.header("content-type").unwrap_or("");
    if !ct.contains("text/event-stream") {
        return Err(format!(
            "[{}] expected text/event-stream content-type, got {} body={}",
            case.name, ct, resp.body
        ));
    }

    if !resp.body.contains("data: [DONE]") {
        return Err(format!(
            "[{}] expected streaming response to terminate with [DONE], body={}",
            case.name, resp.body
        ));
    }

    let data_events: Vec<&str> = resp
        .body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|payload| *payload != "[DONE]")
        .collect();
    if data_events.is_empty() {
        return Err(format!(
            "[{}] expected at least one SSE JSON chunk",
            case.name
        ));
    }

    let mut saw_server_error = false;
    for payload in data_events {
        let chunk: serde_json::Value = serde_json::from_str(payload).map_err(|e| {
            format!(
                "[{}] failed to parse SSE chunk as JSON: {} chunk={}",
                case.name, e, payload
            )
        })?;
        if chunk["error"]["type"].as_str() == Some("server_error")
            && chunk["error"]["message"].is_string()
        {
            saw_server_error = true;
        }
    }

    if !saw_server_error {
        return Err(format!(
            "[{}] expected at least one SSE server_error chunk, got body={}",
            case.name, resp.body
        ));
    }

    Ok(())
}

#[test]
fn completions_request_type_matrix() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let cases = vec![
        Case {
            name: "model_wrong_type",
            body: serde_json::json!({
                "model": 123,
                "messages": [{"role": "user", "content": "hi"}]
            }),
            expect: Expectation::InvalidRequest,
        },
        Case {
            name: "messages_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": {"role": "user", "content": "hi"}
            }),
            expect: Expectation::InvalidRequest,
        },
        Case {
            name: "messages_entry_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [1]
            }),
            expect: Expectation::InvalidRequest,
        },
        Case {
            name: "messages_role_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": 1, "content": "hi"}]
            }),
            expect: Expectation::InvalidRequest,
        },
        Case {
            name: "max_tokens_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": "8"
            }),
            expect: Expectation::InvalidRequest,
        },
        Case {
            name: "max_completion_tokens_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": "8"
            }),
            expect: Expectation::InvalidRequest,
        },
        Case {
            name: "temperature_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": "0.0"
            }),
            expect: Expectation::InvalidRequest,
        },
        Case {
            name: "top_p_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "top_p": "1.0"
            }),
            expect: Expectation::InvalidRequest,
        },
        Case {
            name: "top_k_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "top_k": "20"
            }),
            expect: Expectation::InvalidRequest,
        },
        Case {
            name: "stream_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": "true"
            }),
            expect: Expectation::InvalidRequest,
        },
        Case {
            name: "seed_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "seed": "42"
            }),
            expect: Expectation::InvalidRequest,
        },
        Case {
            name: "presence_penalty_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "presence_penalty": "0.1"
            }),
            expect: Expectation::InvalidRequest,
        },
        Case {
            name: "frequency_penalty_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "frequency_penalty": "0.1"
            }),
            expect: Expectation::InvalidRequest,
        },
    ];

    let mut failures = Vec::new();
    for case in &cases {
        if let Err(err) = run_case(&ctx, case) {
            failures.push(err);
        }
    }
    assert!(
        failures.is_empty(),
        "request type matrix failures:\n{}",
        failures.join("\n")
    );
}

#[test]
fn completions_request_shape_acceptance_matrix() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let cases = vec![
        Case {
            name: "minimal_valid_shape",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}]
            }),
            expect: Expectation::ValidNonStreamingNoBackend,
        },
        Case {
            name: "stream_true_valid_shape",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            }),
            expect: Expectation::ValidStreamingNoBackend,
        },
        Case {
            name: "sampling_valid_shape",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 20,
                "seed": 123
            }),
            expect: Expectation::ValidNonStreamingNoBackend,
        },
        Case {
            name: "tooling_valid_shape",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "parameters": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}}
                        }
                    }
                }],
                "tool_choice": "auto"
            }),
            expect: Expectation::ValidNonStreamingNoBackend,
        },
        Case {
            name: "tool_choice_required_valid_shape",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "parameters": {"type": "object"}
                    }
                }],
                "tool_choice": "required"
            }),
            expect: Expectation::ValidNonStreamingNoBackend,
        },
        Case {
            name: "tool_choice_function_object_valid_shape",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "parameters": {"type": "object"}
                    }
                }],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "echo"}
                }
            }),
            expect: Expectation::ValidNonStreamingNoBackend,
        },
        Case {
            name: "content_array_valid_shape",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "text": "say hi"}]
                }]
            }),
            expect: Expectation::ValidNonStreamingNoBackend,
        },
        Case {
            name: "assistant_tool_call_and_tool_message_valid_shape",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [
                    {"role": "user", "content": "Weather?"},
                    {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                        }]
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "22C"}
                ]
            }),
            expect: Expectation::ValidNonStreamingNoBackend,
        },
    ];

    let mut failures = Vec::new();
    for case in &cases {
        if let Err(err) = run_case(&ctx, case) {
            failures.push(err);
        }
    }
    assert!(
        failures.is_empty(),
        "request acceptance matrix failures:\n{}",
        failures.join("\n")
    );
}
