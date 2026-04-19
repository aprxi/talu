//! Broad request-shape/type coverage for `/v1/chat/completions`.

use crate::server::common::{post_json, ServerConfig, ServerTestContext};

#[derive(Clone, Copy)]
enum Expectation {
    Status(u16),
    NotStatus(u16),
}

struct Case {
    name: &'static str,
    body: serde_json::Value,
    expect: Expectation,
}

fn run_case(ctx: &ServerTestContext, case: &Case) -> Result<(), String> {
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &case.body);
    match case.expect {
        Expectation::Status(code) => {
            if resp.status != code {
                return Err(format!(
                    "[{}] expected status {} got {} body={}",
                    case.name, code, resp.status, resp.body
                ));
            }
        }
        Expectation::NotStatus(code) => {
            if resp.status == code {
                return Err(format!(
                    "[{}] expected status != {} got {} body={}",
                    case.name, code, resp.status, resp.body
                ));
            }
        }
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
            expect: Expectation::Status(400),
        },
        Case {
            name: "messages_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": {"role": "user", "content": "hi"}
            }),
            expect: Expectation::Status(400),
        },
        Case {
            name: "messages_entry_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [1]
            }),
            expect: Expectation::Status(400),
        },
        Case {
            name: "messages_role_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": 1, "content": "hi"}]
            }),
            expect: Expectation::Status(400),
        },
        Case {
            name: "max_tokens_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": "8"
            }),
            expect: Expectation::Status(400),
        },
        Case {
            name: "max_completion_tokens_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": "8"
            }),
            expect: Expectation::Status(400),
        },
        Case {
            name: "temperature_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": "0.0"
            }),
            expect: Expectation::Status(400),
        },
        Case {
            name: "top_p_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "top_p": "1.0"
            }),
            expect: Expectation::Status(400),
        },
        Case {
            name: "top_k_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "top_k": "20"
            }),
            expect: Expectation::Status(400),
        },
        Case {
            name: "stream_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": "true"
            }),
            expect: Expectation::Status(400),
        },
        Case {
            name: "seed_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "seed": "42"
            }),
            expect: Expectation::Status(400),
        },
        Case {
            name: "presence_penalty_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "presence_penalty": "0.1"
            }),
            expect: Expectation::Status(400),
        },
        Case {
            name: "frequency_penalty_wrong_type",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "frequency_penalty": "0.1"
            }),
            expect: Expectation::Status(400),
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
            expect: Expectation::NotStatus(400),
        },
        Case {
            name: "stream_true_valid_shape",
            body: serde_json::json!({
                "model": "test/model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            }),
            expect: Expectation::NotStatus(400),
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
            expect: Expectation::NotStatus(400),
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
            expect: Expectation::NotStatus(400),
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
            expect: Expectation::NotStatus(400),
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
