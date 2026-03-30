//! OpenAI-compatible `/v1/chat/completions` endpoint.
//!
//! Pure passthrough: parses HTTP request, calls C API, serializes HTTP response.
//! All domain logic lives in core/src/ (Zig).

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::anyhow;
use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::{Request, Response, StatusCode};
use serde_json::json;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use crate::server::auth_gateway::AuthContext;
use crate::server::completions_types::*;
use crate::server::state::AppState;
use talu::ChatHandle;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub async fn handle_create(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth_ctx: Option<AuthContext>,
) -> Response<BoxBody> {
    let body_bytes = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(_) => return error_response(StatusCode::BAD_REQUEST, "invalid body"),
    };

    let parsed: CreateChatCompletionBody = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Invalid JSON: {e}"),
            )
        }
    };

    if parsed.messages.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "messages array is empty");
    }

    let stream = parsed.stream.unwrap_or(false);

    if stream {
        handle_streaming(state, parsed).await
    } else {
        handle_non_streaming(state, parsed).await
    }
}

// ---------------------------------------------------------------------------
// Non-streaming
// ---------------------------------------------------------------------------

async fn handle_non_streaming(
    state: Arc<AppState>,
    body: CreateChatCompletionBody,
) -> Response<BoxBody> {
    let model_id = match resolve_model(&state, body.model.as_deref()).await {
        Ok(m) => m,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &e.to_string()),
    };

    let messages_json = match serde_json::to_string(&body.messages) {
        Ok(j) => j,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &format!("Failed to serialize messages: {e}")),
    };

    let system_prompt = extract_system_prompt(&body.messages);
    let max_tokens = body.max_completion_tokens.or(body.max_tokens);
    let cfg = build_generate_config(&body, max_tokens);

    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_for_gen = stop_flag.clone();

    let backend = state.backend.clone();
    let batch_scheduler = state.batch_scheduler.lock().unwrap().clone();
    let model_id_for_task = model_id.clone();
    let created = now_unix_seconds();

    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let chat = ChatHandle::new(system_prompt.as_deref())?;
        chat.load_completions_json(&messages_json)
            .map_err(|e| anyhow!("failed to load messages: {e}"))?;

        let mut cfg = cfg;
        cfg.stop_flag = Some(stop_flag_for_gen);

        // Generate. BatchResult.text is the raw model output (with <think>
        // tags intact). BatchResult.tool_calls has parsed tool calls.
        // No further extraction or format conversion needed here.
        if let Some(ref sched) = batch_scheduler {
            let (request_id, event_rx) = sched
                .submit(&chat, cfg, stop_flag)
                .map_err(|e| anyhow!("batch submit failed: {e}"))?;

            loop {
                match event_rx.recv() {
                    Ok(event) if event.is_final => break,
                    Ok(_) => {}
                    Err(_) => break,
                }
            }

            let r = sched.take_result(request_id);
            match r {
                Some(r) => {
                    let has_tool_calls = !r.tool_calls.is_empty();
                    let tool_calls = if has_tool_calls {
                        let tc: Vec<serde_json::Value> = r.tool_calls.iter().map(|tc| json!({
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        })).collect();
                        Some(serde_json::Value::Array(tc))
                    } else {
                        None
                    };
                    // When tool calls are present, content should be null
                    // (the model's text output contains thinking + tool tags, not user-visible content).
                    let content = if has_tool_calls {
                        None
                    } else {
                        r.text.map(|t| t.trim().to_string()).filter(|t| !t.is_empty())
                    };
                    Ok((
                        content,
                        tool_calls,
                        r.prompt_tokens as u64,
                        r.completion_tokens as u64,
                        finish_reason_str(r.finish_reason),
                    ))
                }
                None => Ok((None::<String>, None::<serde_json::Value>, 0u64, 0u64, "stop")),
            }
        } else {
            let mut guard = backend.blocking_lock();
            let be = guard.backend.as_mut()
                .ok_or_else(|| anyhow!("no backend available"))?;
            let result = talu::router::generate(&chat, &[], be, &cfg)
                .map_err(|e| anyhow!("generation failed: {e}"))?;
            Ok((
                result.text().map(|t| t.trim().to_string()).filter(|t| !t.is_empty()),
                None,
                result.prompt_tokens() as u64,
                result.completion_tokens() as u64,
                match result.finish_reason() {
                    talu::FinishReason::Length => "length",
                    talu::FinishReason::ToolCalls => "tool_calls",
                    _ => "stop",
                },
            ))
        }
    })
    .await;

    let (content, tool_calls, prompt_tokens, completion_tokens, finish_reason) = match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };

    let response = ChatCompletion {
        id: format!("chatcmpl-{}", random_id()),
        object: "chat.completion".to_string(),
        created,
        model: model_id_for_task,
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".to_string(),
                content,
                tool_calls,
            },
            finish_reason: finish_reason.to_string(),
        }],
        usage: CompletionUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    json_response(StatusCode::OK, &response)
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

async fn handle_streaming(
    state: Arc<AppState>,
    body: CreateChatCompletionBody,
) -> Response<BoxBody> {
    let model_id = match resolve_model(&state, body.model.as_deref()).await {
        Ok(m) => m,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &e.to_string()),
    };

    let messages_json = match serde_json::to_string(&body.messages) {
        Ok(j) => j,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &format!("Failed to serialize messages: {e}")),
    };

    let system_prompt = extract_system_prompt(&body.messages);
    let max_tokens = body.max_completion_tokens.or(body.max_tokens);
    let cfg = build_generate_config(&body, max_tokens);

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Bytes>();
    let created = now_unix_seconds();
    let completion_id = format!("chatcmpl-{}", random_id());
    let backend = state.backend.clone();
    let batch_scheduler = state.batch_scheduler.lock().unwrap().clone();

    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_for_gen = stop_flag.clone();

    tokio::task::spawn_blocking(move || {
        let chat = match ChatHandle::new(system_prompt.as_deref()) {
            Ok(c) => c,
            Err(e) => {
                let _ = send_error_chunk(&tx, &e.to_string());
                return;
            }
        };

        if let Err(e) = chat.load_completions_json(&messages_json) {
            let _ = send_error_chunk(&tx, &format!("failed to load messages: {e}"));
            return;
        }

        // Initial chunk with role.
        let initial_chunk = ChatCompletionChunk {
            id: completion_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_id.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };
        if send_chunk(&tx, &initial_chunk).is_err() {
            return;
        }

        let mut cfg = cfg;
        cfg.stop_flag = Some(stop_flag_for_gen.clone());

        let finish_reason;
        let prompt_tokens;
        let completion_tokens;

        if let Some(ref sched) = batch_scheduler {
            let submit_result = sched.submit(&chat, cfg, stop_flag);
            let (request_id, event_rx) = match submit_result {
                Ok(v) => v,
                Err(e) => {
                    let _ = send_error_chunk(&tx, &format!("batch submit failed: {e}"));
                    return;
                }
            };

            // Stream batch events as completions chunks.
            // Each event.text is raw decoded text from the C API.
            loop {
                match event_rx.recv() {
                    Ok(event) => {
                        if !event.text.is_empty() {
                            let chunk = ChatCompletionChunk {
                                id: completion_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model_id.clone(),
                                choices: vec![ChunkChoice {
                                    index: 0,
                                    delta: Delta {
                                        role: None,
                                        content: Some(event.text.clone()),
                                    },
                                    finish_reason: None,
                                }],
                                usage: None,
                            };
                            if send_chunk(&tx, &chunk).is_err() {
                                let _ = sched.cancel(request_id);
                                return;
                            }
                        }
                        if event.is_final {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }

            let batch_result = sched.take_result(request_id);
            if let Some(r) = batch_result {
                prompt_tokens = r.prompt_tokens as u64;
                completion_tokens = r.completion_tokens as u64;
                finish_reason = finish_reason_str(r.finish_reason);
            } else {
                prompt_tokens = 0;
                completion_tokens = 0;
                finish_reason = "stop";
            }
        } else {
            // Direct generate_stream path (remote/provider backends).
            let stop_flag_for_cb = stop_flag_for_gen.clone();
            let tx_for_cb = tx.clone();
            let id_for_cb = completion_id.clone();
            let model_for_cb = model_id.clone();

            let callback: talu::router::StreamCallback = Box::new(move |token| {
                if stop_flag_for_cb.load(Ordering::Acquire) {
                    return false;
                }
                if !token.text.is_empty() {
                    let chunk = ChatCompletionChunk {
                        id: id_for_cb.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_for_cb.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: Some(token.text.to_string()),
                            },
                            finish_reason: None,
                        }],
                        usage: None,
                    };
                    if send_chunk(&tx_for_cb, &chunk).is_err() {
                        return false;
                    }
                }
                true
            });

            let mut guard = backend.blocking_lock();
            let be = match guard.backend.as_mut() {
                Some(b) => b,
                None => {
                    let _ = send_error_chunk(&tx, "no backend available");
                    return;
                }
            };

            match talu::router::generate_stream(&chat, &[], be, &cfg, callback) {
                Ok(result) => {
                    prompt_tokens = result.prompt_tokens as u64;
                    completion_tokens = result.completion_tokens as u64;
                    finish_reason = match result.finish_reason {
                        talu::FinishReason::Length => "length",
                        _ => "stop",
                    };
                }
                Err(e) => {
                    let _ = send_error_chunk(&tx, &format!("generation failed: {e}"));
                    return;
                }
            }
        };

        // Final chunk with finish_reason and usage.
        let final_chunk = ChatCompletionChunk {
            id: completion_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model_id,
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                },
                finish_reason: Some(finish_reason.to_string()),
            }],
            usage: Some(CompletionUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }),
        };
        let _ = send_chunk(&tx, &final_chunk);
        let _ = tx.send(Bytes::from("data: [DONE]\n\n"));
    });

    let stream =
        UnboundedReceiverStream::new(rx).map(|chunk| Ok::<_, Infallible>(Frame::data(chunk)));
    let body = StreamBody::new(stream).boxed();

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .body(body)
        .unwrap()
}

// ---------------------------------------------------------------------------
// Helpers (HTTP plumbing only — no domain logic)
// ---------------------------------------------------------------------------

async fn resolve_model(state: &AppState, requested: Option<&str>) -> anyhow::Result<String> {
    if let Some(m) = requested {
        return Ok(m.to_string());
    }
    let guard = state.backend.lock().await;
    guard
        .current_model
        .clone()
        .or_else(|| state.configured_model.clone())
        .ok_or_else(|| anyhow!("no model specified and no default model configured"))
}

fn extract_system_prompt(messages: &[ChatMessage]) -> Option<String> {
    messages.first().and_then(|m| {
        if m.role == "system" {
            m.content.as_ref().and_then(|c| match c {
                serde_json::Value::String(s) => Some(s.clone()),
                _ => None,
            })
        } else {
            None
        }
    })
}

/// Build GenerateConfig from request params. Pure field mapping.
fn build_generate_config(
    body: &CreateChatCompletionBody,
    max_tokens: Option<i64>,
) -> talu::router::GenerateConfig {
    let mut cfg = talu::router::GenerateConfig::default();
    cfg.completions_mode = true;
    if let Some(mt) = max_tokens {
        cfg.max_tokens = mt as usize;
    }
    if let Some(t) = body.temperature { cfg.temperature = t as f32; }
    if let Some(p) = body.top_p { cfg.top_p = p as f32; }
    if let Some(k) = body.top_k { cfg.top_k = k as usize; }
    if let Some(s) = body.seed { cfg.seed = s; }
    if let Some(pp) = body.presence_penalty { cfg.presence_penalty = pp as f32; }
    if let Some(fp) = body.frequency_penalty { cfg.frequency_penalty = fp as f32; }
    cfg.tools_json = body.tools.as_ref().map(|v| v.to_string());
    cfg.tool_choice = body.tool_choice.as_ref().map(|v| match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    });
    cfg
}

fn finish_reason_str(code: u8) -> &'static str {
    match code {
        1 => "length",
        3 => "tool_calls",
        _ => "stop",
    }
}

fn send_chunk(
    tx: &tokio::sync::mpsc::UnboundedSender<Bytes>,
    chunk: &ChatCompletionChunk,
) -> Result<(), ()> {
    let json = serde_json::to_string(chunk).map_err(|_| ())?;
    tx.send(Bytes::from(format!("data: {json}\n\n"))).map_err(|_| ())
}

fn send_error_chunk(
    tx: &tokio::sync::mpsc::UnboundedSender<Bytes>,
    message: &str,
) -> Result<(), ()> {
    let payload = json!({"error": {"message": message, "type": "server_error"}});
    let sse = format!("data: {}\n\n", serde_json::to_string(&payload).unwrap_or_default());
    tx.send(Bytes::from(sse)).map_err(|_| ())?;
    tx.send(Bytes::from("data: [DONE]\n\n")).map_err(|_| ())
}

fn error_response(status: StatusCode, message: &str) -> Response<BoxBody> {
    let payload = json!({
        "error": {
            "message": message,
            "type": if status.is_client_error() { "invalid_request_error" } else { "server_error" },
            "code": null,
            "param": null,
        }
    });
    let body = serde_json::to_vec(&payload).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}

fn json_response<T: serde::Serialize>(status: StatusCode, value: &T) -> Response<BoxBody> {
    let body = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}

fn random_id() -> String {
    format!("{:x}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos())
}

fn now_unix_seconds() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64
}
