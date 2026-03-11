//! `/v1/agent/processes/*` long-lived process session endpoints.
//!
//! Transport/glue only: parse HTTP, call core `talu_process_*` APIs via
//! `talu::process::ProcessSession`, and stream bytes as SSE or WebSocket.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures_util::SinkExt;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::upgrade::Upgraded;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::Mutex;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use tokio_tungstenite::tungstenite::protocol::Role;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;
use utoipa::ToSchema;

use crate::server::auth_gateway::AuthContext;
use crate::server::code_ws;
use crate::server::state::{AppState, ProcessSession};

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;
type WsStream = WebSocketStream<TokioIo<Upgraded>>;

const STREAM_READ_BUFFER_BYTES: usize = 16 * 1024;
const STREAM_POLL_INTERVAL_MS: u64 = 10;
const MAX_PROCESSES_PER_OWNER: usize = 10;
const MAX_PROCESSES_TOTAL: usize = 50;

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct ProcessSpawnRequest {
    command: String,
    #[serde(default)]
    cwd: Option<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct ProcessSendRequest {
    data: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ProcessSessionResponse {
    process_id: String,
    command: String,
    cwd: Option<String>,
    attached_streams: usize,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ProcessListResponse {
    data: Vec<ProcessSessionResponse>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ProcessSendResponse {
    process_id: String,
    bytes_written: usize,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ProcessDeleteResponse {
    process_id: String,
    terminated: bool,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ProcessEvent {
    #[serde(rename = "type")]
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ProcessControlMessage {
    #[serde(rename = "type")]
    msg_type: String,
    signal: Option<String>,
}

#[utoipa::path(post, path = "/v1/agent/processes/spawn", tag = "Agent::Process",
    request_body = ProcessSpawnRequest,
    responses(
        (status = 200, description = "Create process session. In strict mode, descendant exec/file access is sandbox-enforced; in host mode, passthrough has no firewall guarantee.", body = ProcessSessionResponse),
        (status = 400, body = crate::server::http::ErrorResponse),
        (status = 429, body = crate::server::http::ErrorResponse),
        (status = 500, body = crate::server::http::ErrorResponse),
    ))]
pub async fn handle_spawn(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e.to_string()),
    };
    let request: ProcessSpawnRequest = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };
    if request.command.trim().is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "command is required",
        );
    }

    let owner = owner_key(auth.as_ref());
    let mut sessions = state.process_sessions.lock().await;
    if sessions.len() >= MAX_PROCESSES_TOTAL {
        return json_error(
            StatusCode::TOO_MANY_REQUESTS,
            "resource_exhausted",
            "server process session limit reached",
        );
    }
    let owner_count = sessions
        .values()
        .filter(|session| session.owner_key == owner)
        .count();
    if owner_count >= MAX_PROCESSES_PER_OWNER {
        return json_error(
            StatusCode::TOO_MANY_REQUESTS,
            "resource_exhausted",
            "process session limit reached for this auth context",
        );
    }

    let policy = super::load_runtime_policy(&state);
    let runtime_mode = super::runtime_mode_for_talu(&state);
    let sandbox_backend = super::sandbox_backend_for_talu(state.sandbox_backend);
    let effective_cwd = if state.agent_runtime_mode == crate::server::AgentRuntimeMode::Strict {
        request
            .cwd
            .clone()
            .or_else(|| super::default_workdir(&state))
    } else {
        request.cwd.clone()
    };

    let process = match talu::process::ProcessSession::open_with_policy_runtime(
        &request.command,
        effective_cwd.as_deref(),
        policy.as_deref(),
        runtime_mode,
        sandbox_backend,
    ) {
        Ok(value) => value,
        Err(e) => {
            let (status, code) = match &e {
                talu::process::ProcessError::CommandDenied(_) => {
                    (StatusCode::FORBIDDEN, "command_denied")
                }
                talu::process::ProcessError::PolicyDeniedCwd(_) => {
                    (StatusCode::FORBIDDEN, "policy_denied_cwd")
                }
                talu::process::ProcessError::PolicyDeniedExec(_) => {
                    (StatusCode::FORBIDDEN, "policy_denied_exec")
                }
                talu::process::ProcessError::StrictUnavailable(_) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "strict_runtime_unavailable",
                ),
                talu::process::ProcessError::StrictSetupFailed(_) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "strict_runtime_setup_failed",
                ),
                _ => (StatusCode::INTERNAL_SERVER_ERROR, "process_error"),
            };
            return json_error(
                status,
                code,
                &format!("failed to open process session: {e}"),
            );
        }
    };

    let process_id = uuid::Uuid::new_v4().to_string();
    let now = Instant::now();
    let response = ProcessSessionResponse {
        process_id: process_id.clone(),
        command: request.command.clone(),
        cwd: effective_cwd.clone(),
        attached_streams: 0,
    };
    sessions.insert(
        process_id,
        ProcessSession {
            process: Arc::new(tokio::sync::Mutex::new(process)),
            owner_key: owner,
            command: request.command,
            cwd: effective_cwd,
            created_at: now,
            last_access: now,
            attached_streams: 0,
        },
    );
    drop(sessions);

    json_response(StatusCode::OK, &response)
}

#[utoipa::path(get, path = "/v1/agent/processes", tag = "Agent::Process",
    responses((status = 200, body = ProcessListResponse)))]
pub async fn handle_list(
    state: Arc<AppState>,
    _req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let owner = owner_key(auth.as_ref());
    let sessions = state.process_sessions.lock().await;
    let data = sessions
        .iter()
        .filter(|(_, session)| session.owner_key == owner)
        .map(|(process_id, session)| ProcessSessionResponse {
            process_id: process_id.clone(),
            command: session.command.clone(),
            cwd: session.cwd.clone(),
            attached_streams: session.attached_streams,
        })
        .collect();
    json_response(StatusCode::OK, &ProcessListResponse { data })
}

#[utoipa::path(post, path = "/v1/agent/processes/{id}/send", tag = "Agent::Process",
    params(("id" = String, Path, description = "Process session id")),
    request_body = ProcessSendRequest,
    responses(
        (status = 200, body = ProcessSendResponse),
        (status = 400, body = crate::server::http::ErrorResponse),
        (status = 403, body = crate::server::http::ErrorResponse),
        (status = 404, body = crate::server::http::ErrorResponse),
    ))]
pub async fn handle_send(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let process_id = match process_id_from_send_path(req.uri().path()) {
        Some(value) => value.to_string(),
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "invalid process send path",
            )
        }
    };

    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e.to_string()),
    };
    let request: ProcessSendRequest = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    let owner = owner_key(auth.as_ref());
    let process = {
        let mut sessions = state.process_sessions.lock().await;
        let session = match sessions.get_mut(&process_id) {
            Some(value) => value,
            None => return json_error(StatusCode::NOT_FOUND, "not_found", "process not found"),
        };
        if session.owner_key != owner {
            return json_error(
                StatusCode::FORBIDDEN,
                "forbidden",
                "process belongs to another auth context",
            );
        }
        session.last_access = Instant::now();
        session.process.clone()
    };

    let mut guard = process.lock().await;
    if let Err(e) = guard.write(request.data.as_bytes()) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "process_error",
            &format!("failed to write process stdin: {e}"),
        );
    }

    json_response(
        StatusCode::OK,
        &ProcessSendResponse {
            process_id,
            bytes_written: request.data.len(),
        },
    )
}

#[utoipa::path(get, path = "/v1/agent/processes/{id}/stream", tag = "Agent::Process",
    params(("id" = String, Path, description = "Process session id")),
    responses(
        (status = 200, description = "SSE stream with output/exit events"),
        (status = 400, body = crate::server::http::ErrorResponse),
        (status = 403, body = crate::server::http::ErrorResponse),
        (status = 404, body = crate::server::http::ErrorResponse),
        (status = 409, body = crate::server::http::ErrorResponse),
    ))]
pub async fn handle_stream(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let process_id = match process_id_from_stream_path(req.uri().path()) {
        Some(value) => value.to_string(),
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "invalid process stream path",
            )
        }
    };

    let owner = owner_key(auth.as_ref());
    let process = {
        let mut sessions = state.process_sessions.lock().await;
        let session = match sessions.get_mut(&process_id) {
            Some(value) => value,
            None => return json_error(StatusCode::NOT_FOUND, "not_found", "process not found"),
        };
        if session.owner_key != owner {
            return json_error(
                StatusCode::FORBIDDEN,
                "forbidden",
                "process belongs to another auth context",
            );
        }
        if session.attached_streams > 0 {
            return json_error(
                StatusCode::CONFLICT,
                "already_streaming",
                "process stream already attached",
            );
        }
        session.attached_streams += 1;
        session.last_access = Instant::now();
        session.process.clone()
    };

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Bytes>();
    let state_for_stream = state.clone();
    let process_id_for_stream = process_id.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(STREAM_POLL_INTERVAL_MS));
        let mut read_buf = vec![0u8; STREAM_READ_BUFFER_BYTES];

        loop {
            interval.tick().await;

            let (output_len, should_exit, exit_code) = {
                let mut should_exit = false;
                let mut exit_code: Option<i32> = None;
                let mut guard = process.lock().await;
                let output_len = match guard.read(&mut read_buf) {
                    Ok(n) => n,
                    Err(e) => {
                        let _ = tx.send(sse_data(&ProcessEvent {
                            kind: "error".to_string(),
                            data: None,
                            code: None,
                            message: Some(e.to_string()),
                        }));
                        break;
                    }
                };

                match guard.is_alive() {
                    Ok(alive) => {
                        if !alive {
                            should_exit = true;
                            exit_code = guard.exit_code().ok().flatten();
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(sse_data(&ProcessEvent {
                            kind: "error".to_string(),
                            data: None,
                            code: None,
                            message: Some(e.to_string()),
                        }));
                        break;
                    }
                }
                (output_len, should_exit, exit_code)
            };

            if output_len > 0 {
                let _ = tx.send(sse_data(&ProcessEvent {
                    kind: "output".to_string(),
                    data: Some(String::from_utf8_lossy(&read_buf[..output_len]).into_owned()),
                    code: None,
                    message: None,
                }));
                touch_process_session(&state_for_stream, &process_id_for_stream).await;
            }

            if should_exit {
                let _ = tx.send(sse_data(&ProcessEvent {
                    kind: "exit".to_string(),
                    data: None,
                    code: exit_code,
                    message: None,
                }));
                break;
            }
        }

        detach_stream(&state_for_stream, &process_id_for_stream).await;
    });

    let stream =
        UnboundedReceiverStream::new(rx).map(|chunk| Ok::<_, Infallible>(Frame::data(chunk)));
    let body = StreamBody::new(stream).boxed();

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream; charset=utf-8")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .body(body)
        .unwrap()
}

#[utoipa::path(get, path = "/v1/agent/processes/{id}/ws", tag = "Agent::Process",
    params(("id" = String, Path, description = "Process session id")),
    responses(
        (status = 101, description = "WebSocket upgrade"),
        (status = 400, body = crate::server::http::ErrorResponse),
        (status = 403, body = crate::server::http::ErrorResponse),
        (status = 404, body = crate::server::http::ErrorResponse),
        (status = 409, body = crate::server::http::ErrorResponse),
    ))]
pub async fn handle_ws(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let process_id = match process_id_from_ws_path(&path) {
        Some(value) => value.to_string(),
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "invalid process ws path",
            )
        }
    };

    let owner = owner_key(auth.as_ref());
    let process = {
        let mut sessions = state.process_sessions.lock().await;
        let session = match sessions.get_mut(&process_id) {
            Some(value) => value,
            None => return json_error(StatusCode::NOT_FOUND, "not_found", "process not found"),
        };

        if session.owner_key != owner {
            return json_error(
                StatusCode::FORBIDDEN,
                "forbidden",
                "process belongs to another auth context",
            );
        }

        if session.attached_streams > 0 {
            return json_error(
                StatusCode::CONFLICT,
                "already_streaming",
                "process stream already attached",
            );
        }

        session.attached_streams += 1;
        session.last_access = Instant::now();
        session.process.clone()
    };

    let key = match req.headers().get("sec-websocket-key") {
        Some(value) => value.as_bytes().to_vec(),
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "Missing Sec-WebSocket-Key header",
            )
        }
    };
    let accept = code_ws::compute_accept_key(&key);

    let upgrade = hyper::upgrade::on(req);
    let state_for_ws = state.clone();
    tokio::spawn(async move {
        match upgrade.await {
            Ok(upgraded) => {
                if let Err(err) = handle_ws_connection(
                    state_for_ws.clone(),
                    process_id.clone(),
                    process,
                    upgraded,
                )
                .await
                {
                    log::warn!(target: "server::agent_process", "process ws {} ended with error: {}", process_id, err);
                }
                detach_stream(&state_for_ws, &process_id).await;
            }
            Err(e) => {
                log::error!(target: "server::agent_process", "WebSocket upgrade failed: {e}");
                detach_stream(&state_for_ws, &process_id).await;
            }
        }
    });

    Response::builder()
        .status(StatusCode::SWITCHING_PROTOCOLS)
        .header("upgrade", "websocket")
        .header("connection", "Upgrade")
        .header("sec-websocket-accept", accept)
        .body(Full::new(Bytes::new()).boxed())
        .unwrap()
}

async fn handle_ws_connection(
    state: Arc<AppState>,
    process_id: String,
    process: Arc<Mutex<talu::process::ProcessSession>>,
    upgraded: Upgraded,
) -> Result<(), String> {
    let mut ws: WsStream =
        WebSocketStream::from_raw_socket(TokioIo::new(upgraded), Role::Server, None).await;

    let mut interval = tokio::time::interval(Duration::from_millis(STREAM_POLL_INTERVAL_MS));
    let mut read_buf = vec![0u8; STREAM_READ_BUFFER_BYTES];

    loop {
        tokio::select! {
            _ = interval.tick() => {
                let mut should_exit = false;
                let mut output_len = 0usize;
                let mut exit_code: Option<i32> = None;
                {
                    let mut guard = process.lock().await;
                    match guard.read(&mut read_buf) {
                        Ok(n) => output_len = n,
                        Err(e) => {
                            let _ = ws_send_error(&mut ws, "read_failed", &e.to_string()).await;
                            should_exit = true;
                        }
                    }

                    if !should_exit {
                        match guard.is_alive() {
                            Ok(alive) => {
                                if !alive {
                                    should_exit = true;
                                    exit_code = guard.exit_code().ok().flatten();
                                }
                            }
                            Err(e) => {
                                let _ = ws_send_error(&mut ws, "alive_failed", &e.to_string()).await;
                                should_exit = true;
                            }
                        }
                    }
                }

                if output_len > 0 {
                    ws.send(Message::Binary(read_buf[..output_len].to_vec().into()))
                        .await
                        .map_err(|e| format!("failed to send process output: {e}"))?;
                    touch_process_session(&state, &process_id).await;
                }

                if should_exit {
                    let _ = ws.send(Message::Text(
                        json!({"type":"exit","code":exit_code}).to_string().into(),
                    )).await;
                    break;
                }
            }
            maybe_msg = futures_util::StreamExt::next(&mut ws) => {
                let msg = match maybe_msg {
                    Some(Ok(m)) => m,
                    Some(Err(e)) => return Err(format!("ws receive failed: {e}")),
                    None => break,
                };

                match msg {
                    Message::Binary(data) => {
                        let mut guard = process.lock().await;
                        if let Err(e) = guard.write(&data) {
                            let _ = ws_send_error(&mut ws, "write_failed", &e.to_string()).await;
                            break;
                        }
                        touch_process_session(&state, &process_id).await;
                    }
                    Message::Text(text) => {
                        let control: ProcessControlMessage = match serde_json::from_str(&text) {
                            Ok(value) => value,
                            Err(e) => {
                                let _ = ws_send_error(&mut ws, "invalid_json", &e.to_string()).await;
                                continue;
                            }
                        };

                        match control.msg_type.as_str() {
                            "signal" => {
                                let signal_name = control.signal.as_deref().unwrap_or("");
                                let signal = match parse_signal(signal_name) {
                                    Some(value) => value,
                                    None => {
                                        let _ = ws_send_error(
                                            &mut ws,
                                            "invalid_signal",
                                            "signal must be one of INT, TERM, KILL, HUP, QUIT",
                                        ).await;
                                        continue;
                                    }
                                };

                                let mut guard = process.lock().await;
                                if let Err(e) = guard.signal(signal) {
                                    let _ = ws_send_error(&mut ws, "signal_failed", &e.to_string()).await;
                                }
                                touch_process_session(&state, &process_id).await;
                            }
                            other => {
                                let _ = ws_send_error(
                                    &mut ws,
                                    "unsupported_message",
                                    &format!("unsupported control message type: {other}"),
                                ).await;
                            }
                        }
                    }
                    Message::Ping(payload) => {
                        ws.send(Message::Pong(payload))
                            .await
                            .map_err(|e| format!("failed to send pong: {e}"))?;
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
        }
    }

    Ok(())
}

async fn ws_send_error(ws: &mut WsStream, code: &str, message: &str) -> Result<(), String> {
    ws.send(Message::Text(
        json!({
            "type": "error",
            "code": code,
            "message": message,
        })
        .to_string()
        .into(),
    ))
    .await
    .map_err(|e| e.to_string())
}

fn parse_signal(value: &str) -> Option<u8> {
    match value.to_ascii_uppercase().as_str() {
        "INT" | "SIGINT" => Some(2),
        "TERM" | "SIGTERM" => Some(15),
        "KILL" | "SIGKILL" => Some(9),
        "HUP" | "SIGHUP" => Some(1),
        "QUIT" | "SIGQUIT" => Some(3),
        _ => None,
    }
}

#[utoipa::path(delete, path = "/v1/agent/processes/{id}", tag = "Agent::Process",
    params(("id" = String, Path, description = "Process session id")),
    responses(
        (status = 200, body = ProcessDeleteResponse),
        (status = 403, body = crate::server::http::ErrorResponse),
        (status = 404, body = crate::server::http::ErrorResponse),
    ))]
pub async fn handle_delete(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let process_id = match process_id_from_path(req.uri().path()) {
        Some(value) => value.to_string(),
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "invalid process path",
            )
        }
    };

    let owner = owner_key(auth.as_ref());
    let removed = {
        let mut sessions = state.process_sessions.lock().await;
        if let Some(existing) = sessions.get(&process_id) {
            if existing.owner_key != owner {
                return json_error(
                    StatusCode::FORBIDDEN,
                    "forbidden",
                    "process belongs to another auth context",
                );
            }
        } else {
            return json_error(StatusCode::NOT_FOUND, "not_found", "process not found");
        }
        sessions.remove(&process_id)
    };

    if let Some(entry) = removed {
        let mut process = entry.process.lock().await;
        process.close();
    }

    json_response(
        StatusCode::OK,
        &ProcessDeleteResponse {
            process_id,
            terminated: true,
        },
    )
}

fn process_id_from_path(path: &str) -> Option<&str> {
    let prefix = "/v1/agent/processes/";
    if !path.starts_with(prefix) {
        return None;
    }
    let rest = &path[prefix.len()..];
    if rest.is_empty() || rest.contains('/') {
        return None;
    }
    Some(rest)
}

fn process_id_from_ws_path(path: &str) -> Option<&str> {
    let prefix = "/v1/agent/processes/";
    let suffix = "/ws";
    let rest = path.strip_prefix(prefix)?;
    if !rest.ends_with(suffix) {
        return None;
    }
    let id = &rest[..rest.len().saturating_sub(suffix.len())];
    if id.is_empty() || id.contains('/') {
        return None;
    }
    Some(id)
}

fn process_id_from_send_path(path: &str) -> Option<&str> {
    let prefix = "/v1/agent/processes/";
    let suffix = "/send";
    if !path.starts_with(prefix) || !path.ends_with(suffix) {
        return None;
    }
    let id = &path[prefix.len()..path.len() - suffix.len()];
    if id.is_empty() || id.contains('/') {
        return None;
    }
    Some(id)
}

fn process_id_from_stream_path(path: &str) -> Option<&str> {
    let prefix = "/v1/agent/processes/";
    let suffix = "/stream";
    if !path.starts_with(prefix) || !path.ends_with(suffix) {
        return None;
    }
    let id = &path[prefix.len()..path.len() - suffix.len()];
    if id.is_empty() || id.contains('/') {
        return None;
    }
    Some(id)
}

async fn touch_process_session(state: &AppState, process_id: &str) {
    let mut sessions = state.process_sessions.lock().await;
    if let Some(session) = sessions.get_mut(process_id) {
        session.last_access = Instant::now();
    }
}

async fn detach_stream(state: &AppState, process_id: &str) {
    let mut sessions = state.process_sessions.lock().await;
    if let Some(session) = sessions.get_mut(process_id) {
        session.attached_streams = session.attached_streams.saturating_sub(1);
        session.last_access = Instant::now();
    }
}

fn owner_key(auth: Option<&AuthContext>) -> String {
    match auth {
        Some(ctx) => {
            let group = ctx.group_id.as_deref().unwrap_or("");
            let user = ctx.user_id.as_deref().unwrap_or("");
            format!("{}:{}:{}", ctx.tenant_id, group, user)
        }
        None => "default".to_string(),
    }
}

fn sse_data<T: Serialize>(payload: &T) -> Bytes {
    let value = serde_json::to_string(payload).unwrap_or_else(|_| "{}".to_string());
    Bytes::from(format!("data: {value}\n\n"))
}

fn json_response<T: Serialize>(status: StatusCode, value: &T) -> Response<BoxBody> {
    let body = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}

fn json_error(status: StatusCode, code: &str, message: &str) -> Response<BoxBody> {
    let body = serde_json::to_vec(&json!({
        "error": {
            "code": code,
            "message": message,
        }
    }))
    .unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}
