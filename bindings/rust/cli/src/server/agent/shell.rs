//! `/v1/agent/shells/*` interactive PTY shell endpoints.
//!
//! HTTP and WebSocket transport only. Process and PTY logic lives in core
//! (`talu_shell_*` via `talu::shell::ShellSession`).

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::upgrade::Upgraded;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::protocol::Role;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;
use utoipa::ToSchema;

use crate::server::auth_gateway::AuthContext;
use crate::server::code_ws;
use crate::server::state::{AppState, ShellSession};

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;
type WsStream = WebSocketStream<TokioIo<Upgraded>>;

const DEFAULT_COLS: u16 = 120;
const DEFAULT_ROWS: u16 = 40;
const WS_READ_BUFFER_BYTES: usize = 16 * 1024;
const WS_POLL_INTERVAL_MS: u64 = 10;
const MAX_SHELLS_PER_OWNER: usize = 10;
const MAX_SHELLS_TOTAL: usize = 50;

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct ShellCreateRequest {
    #[serde(default)]
    cols: Option<u16>,
    #[serde(default)]
    rows: Option<u16>,
    #[serde(default)]
    cwd: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ShellSessionResponse {
    shell_id: String,
    cols: u16,
    rows: u16,
    cwd: Option<String>,
    attached_clients: usize,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ShellListResponse {
    data: Vec<ShellSessionResponse>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ShellDeleteResponse {
    shell_id: String,
    terminated: bool,
}

#[derive(Debug, Deserialize)]
struct ShellControlMessage {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(default)]
    cols: Option<u16>,
    #[serde(default)]
    rows: Option<u16>,
    #[serde(default)]
    signal: Option<String>,
}

#[utoipa::path(post, path = "/v1/agent/shells", tag = "Agent::Shell",
    request_body = ShellCreateRequest,
    responses(
        (status = 200, description = "Create shell session. In strict mode, descendant exec/file access is sandbox-enforced; in host mode, passthrough has no firewall guarantee.", body = ShellSessionResponse),
        (status = 400, body = crate::server::http::ErrorResponse),
        (status = 403, body = crate::server::http::ErrorResponse),
        (status = 429, body = crate::server::http::ErrorResponse),
        (status = 500, body = crate::server::http::ErrorResponse),
    ))]
pub async fn handle_create(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e.to_string()),
    };

    let request: ShellCreateRequest = if body.is_empty() {
        ShellCreateRequest {
            cols: None,
            rows: None,
            cwd: None,
        }
    } else {
        match serde_json::from_slice(&body) {
            Ok(v) => v,
            Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
        }
    };

    let cols = request.cols.unwrap_or(DEFAULT_COLS);
    let rows = request.rows.unwrap_or(DEFAULT_ROWS);
    if cols == 0 || rows == 0 {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "cols and rows must be greater than zero",
        );
    }

    let owner = owner_key(auth.as_ref());

    // Hold the lock across check + open + insert to prevent TOCTOU races
    // on session limits. The FFI open call is synchronous and fast (forkpty).
    let mut sessions = state.shell_sessions.lock().await;

    if sessions.len() >= MAX_SHELLS_TOTAL {
        return json_error(
            StatusCode::TOO_MANY_REQUESTS,
            "resource_exhausted",
            "server shell session limit reached",
        );
    }
    let owner_count = sessions
        .values()
        .filter(|session| session.owner_key == owner)
        .count();
    if owner_count >= MAX_SHELLS_PER_OWNER {
        return json_error(
            StatusCode::TOO_MANY_REQUESTS,
            "resource_exhausted",
            "shell session limit reached for this auth context",
        );
    }

    let policy = super::load_runtime_policy(&state);
    let runtime_mode = super::runtime_mode_for_talu(&state);
    let sandbox_backend = super::sandbox_backend_for_talu(state.sandbox_backend);
    let effective_cwd = if state.agent_runtime_mode == crate::server::AgentRuntimeMode::Strict {
        request
            .cwd
            .clone()
            .or_else(|| Some(state.workspace_dir.to_string_lossy().into_owned()))
    } else {
        request.cwd.clone()
    };

    let shell = match talu::shell::ShellSession::open_with_policy_runtime(
        cols,
        rows,
        effective_cwd.as_deref(),
        policy.as_deref(),
        runtime_mode,
        sandbox_backend,
    ) {
        Ok(value) => value,
        Err(e) => {
            let (status, code) = match &e {
                talu::shell::ShellError::PolicyDeniedCwd(_) => {
                    (StatusCode::FORBIDDEN, "policy_denied_cwd")
                }
                talu::shell::ShellError::PolicyDeniedExec(_)
                | talu::shell::ShellError::CommandDenied(_) => {
                    (StatusCode::FORBIDDEN, "policy_denied_exec")
                }
                talu::shell::ShellError::StrictUnavailable(_) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "strict_runtime_unavailable",
                ),
                talu::shell::ShellError::StrictSetupFailed(_) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "strict_runtime_setup_failed",
                ),
                _ => (StatusCode::INTERNAL_SERVER_ERROR, "shell_error"),
            };
            return json_error(status, code, &format!("failed to open shell: {e}"));
        }
    };

    let shell_id = uuid::Uuid::new_v4().to_string();
    let now = Instant::now();
    let response = ShellSessionResponse {
        shell_id: shell_id.clone(),
        cols,
        rows,
        cwd: effective_cwd.clone(),
        attached_clients: 0,
    };

    sessions.insert(
        shell_id,
        ShellSession {
            shell: Arc::new(Mutex::new(shell)),
            owner_key: owner,
            cwd: effective_cwd,
            cols,
            rows,
            created_at: now,
            last_access: now,
            attached_clients: 0,
        },
    );
    drop(sessions);

    json_response(StatusCode::OK, &response)
}

#[utoipa::path(get, path = "/v1/agent/shells", tag = "Agent::Shell",
    responses((status = 200, body = ShellListResponse)))]
pub async fn handle_list(
    state: Arc<AppState>,
    _req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let owner = owner_key(auth.as_ref());
    let sessions = state.shell_sessions.lock().await;

    let data = sessions
        .iter()
        .filter(|(_, session)| session.owner_key == owner)
        .map(|(shell_id, session)| ShellSessionResponse {
            shell_id: shell_id.clone(),
            cols: session.cols,
            rows: session.rows,
            cwd: session.cwd.clone(),
            attached_clients: session.attached_clients,
        })
        .collect();

    json_response(StatusCode::OK, &ShellListResponse { data })
}

#[utoipa::path(get, path = "/v1/agent/shells/{id}", tag = "Agent::Shell",
    params(("id" = String, Path, description = "Shell session id")),
    responses(
        (status = 200, body = ShellSessionResponse),
        (status = 403, body = crate::server::http::ErrorResponse),
        (status = 404, body = crate::server::http::ErrorResponse),
    ))]
pub async fn handle_get(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let shell_id = match shell_id_from_path(req.uri().path()) {
        Some(value) => value,
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "invalid shell path",
            )
        }
    };

    let owner = owner_key(auth.as_ref());
    let mut sessions = state.shell_sessions.lock().await;
    let session = match sessions.get_mut(shell_id) {
        Some(value) => value,
        None => return json_error(StatusCode::NOT_FOUND, "not_found", "shell not found"),
    };

    if session.owner_key != owner {
        return json_error(
            StatusCode::FORBIDDEN,
            "forbidden",
            "shell belongs to another auth context",
        );
    }

    session.last_access = Instant::now();

    json_response(
        StatusCode::OK,
        &ShellSessionResponse {
            shell_id: shell_id.to_string(),
            cols: session.cols,
            rows: session.rows,
            cwd: session.cwd.clone(),
            attached_clients: session.attached_clients,
        },
    )
}

#[utoipa::path(delete, path = "/v1/agent/shells/{id}", tag = "Agent::Shell",
    params(("id" = String, Path, description = "Shell session id")),
    responses(
        (status = 200, body = ShellDeleteResponse),
        (status = 403, body = crate::server::http::ErrorResponse),
        (status = 404, body = crate::server::http::ErrorResponse),
    ))]
pub async fn handle_delete(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let shell_id = match shell_id_from_path(req.uri().path()) {
        Some(value) => value.to_string(),
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "invalid shell path",
            )
        }
    };

    let owner = owner_key(auth.as_ref());
    let removed = {
        let mut sessions = state.shell_sessions.lock().await;
        if let Some(existing) = sessions.get(&shell_id) {
            if existing.owner_key != owner {
                return json_error(
                    StatusCode::FORBIDDEN,
                    "forbidden",
                    "shell belongs to another auth context",
                );
            }
        } else {
            return json_error(StatusCode::NOT_FOUND, "not_found", "shell not found");
        }
        sessions.remove(&shell_id)
    };

    if let Some(entry) = removed {
        let mut shell = entry.shell.lock().await;
        shell.close();
    }

    json_response(
        StatusCode::OK,
        &ShellDeleteResponse {
            shell_id,
            terminated: true,
        },
    )
}

#[utoipa::path(get, path = "/v1/agent/shells/{id}/ws", tag = "Agent::Shell",
    params(("id" = String, Path, description = "Shell session id")),
    responses(
        (status = 101, description = "WebSocket upgrade"),
        (status = 400, body = crate::server::http::ErrorResponse),
        (status = 403, body = crate::server::http::ErrorResponse),
        (status = 404, body = crate::server::http::ErrorResponse),
    ))]
pub async fn handle_ws(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let shell_id = match shell_id_from_ws_path(&path) {
        Some(value) => value.to_string(),
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "invalid shell ws path",
            )
        }
    };

    let owner = owner_key(auth.as_ref());
    let shell = {
        let mut sessions = state.shell_sessions.lock().await;
        let session = match sessions.get_mut(&shell_id) {
            Some(value) => value,
            None => return json_error(StatusCode::NOT_FOUND, "not_found", "shell not found"),
        };

        if session.owner_key != owner {
            return json_error(
                StatusCode::FORBIDDEN,
                "forbidden",
                "shell belongs to another auth context",
            );
        }

        session.attached_clients = session.attached_clients.saturating_add(1);
        session.last_access = Instant::now();
        session.shell.clone()
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
                if let Err(err) =
                    handle_ws_connection(state_for_ws.clone(), shell_id.clone(), shell, upgraded)
                        .await
                {
                    log::warn!(target: "server::agent_shell", "shell ws {} ended with error: {}", shell_id, err);
                }
                detach_client(&state_for_ws, &shell_id).await;
            }
            Err(e) => {
                log::error!(target: "server::agent_shell", "WebSocket upgrade failed: {e}");
                detach_client(&state_for_ws, &shell_id).await;
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
    shell_id: String,
    shell: Arc<Mutex<talu::shell::ShellSession>>,
    upgraded: Upgraded,
) -> Result<(), String> {
    let mut ws: WsStream =
        WebSocketStream::from_raw_socket(TokioIo::new(upgraded), Role::Server, None).await;

    {
        let mut guard = shell.lock().await;
        let scrollback = guard
            .scrollback()
            .map_err(|e| format!("scrollback failed: {e}"))?;
        if !scrollback.is_empty() {
            ws.send(Message::Binary(scrollback.into()))
                .await
                .map_err(|e| format!("failed to replay scrollback: {e}"))?;
        }
    }

    let mut interval = tokio::time::interval(Duration::from_millis(WS_POLL_INTERVAL_MS));
    let mut read_buf = vec![0u8; WS_READ_BUFFER_BYTES];

    loop {
        tokio::select! {
            _ = interval.tick() => {
                let mut should_exit = false;
                let mut output_len = 0usize;
                {
                    let mut guard = shell.lock().await;
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
                        .map_err(|e| format!("failed to send shell output: {e}"))?;
                    touch_session(&state, &shell_id).await;
                }

                if should_exit {
                    let _ = ws.send(Message::Text(json!({"type":"exit","code":null}).to_string().into())).await;
                    break;
                }
            }
            maybe_msg = ws.next() => {
                let msg = match maybe_msg {
                    Some(Ok(m)) => m,
                    Some(Err(e)) => return Err(format!("ws receive failed: {e}")),
                    None => break,
                };

                match msg {
                    Message::Binary(data) => {
                        let mut guard = shell.lock().await;
                        if let Err(e) = guard.write(&data) {
                            let _ = ws_send_error(&mut ws, "write_failed", &e.to_string()).await;
                            break;
                        }
                        touch_session(&state, &shell_id).await;
                    }
                    Message::Text(text) => {
                        let control: ShellControlMessage = match serde_json::from_str(&text) {
                            Ok(value) => value,
                            Err(e) => {
                                let _ = ws_send_error(&mut ws, "invalid_json", &e.to_string()).await;
                                continue;
                            }
                        };

                        match control.msg_type.as_str() {
                            "resize" => {
                                let (cols, rows) = match (control.cols, control.rows) {
                                    (Some(c), Some(r)) if c > 0 && r > 0 => (c, r),
                                    _ => {
                                        let _ = ws_send_error(
                                            &mut ws,
                                            "invalid_resize",
                                            "resize requires positive cols and rows",
                                        ).await;
                                        continue;
                                    }
                                };

                                let mut guard = shell.lock().await;
                                if let Err(e) = guard.resize(cols, rows) {
                                    let _ = ws_send_error(&mut ws, "resize_failed", &e.to_string()).await;
                                    continue;
                                }
                                update_session_resize(&state, &shell_id, cols, rows).await;
                            }
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

                                let mut guard = shell.lock().await;
                                if let Err(e) = guard.signal(signal) {
                                    let _ = ws_send_error(&mut ws, "signal_failed", &e.to_string()).await;
                                }
                                touch_session(&state, &shell_id).await;
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

async fn detach_client(state: &Arc<AppState>, shell_id: &str) {
    let mut sessions = state.shell_sessions.lock().await;
    if let Some(session) = sessions.get_mut(shell_id) {
        session.attached_clients = session.attached_clients.saturating_sub(1);
        session.last_access = Instant::now();
    }
}

async fn touch_session(state: &Arc<AppState>, shell_id: &str) {
    let mut sessions = state.shell_sessions.lock().await;
    if let Some(session) = sessions.get_mut(shell_id) {
        session.last_access = Instant::now();
    }
}

async fn update_session_resize(state: &Arc<AppState>, shell_id: &str, cols: u16, rows: u16) {
    let mut sessions = state.shell_sessions.lock().await;
    if let Some(session) = sessions.get_mut(shell_id) {
        session.cols = cols;
        session.rows = rows;
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

fn shell_id_from_path(path: &str) -> Option<&str> {
    let prefix = "/v1/agent/shells/";
    let rest = path.strip_prefix(prefix)?;
    if rest.is_empty() || rest.contains('/') {
        return None;
    }
    Some(rest)
}

fn shell_id_from_ws_path(path: &str) -> Option<&str> {
    let prefix = "/v1/agent/shells/";
    let suffix = "/ws";
    let rest = path.strip_prefix(prefix)?;
    if !rest.ends_with(suffix) {
        return None;
    }
    let shell_id = &rest[..rest.len().saturating_sub(suffix.len())];
    if shell_id.is_empty() || shell_id.contains('/') {
        return None;
    }
    Some(shell_id)
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
    json_response(
        status,
        &json!({
            "error": {
                "code": code,
                "message": message,
            }
        }),
    )
}
