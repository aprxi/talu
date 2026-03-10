//! `/v1/agent/exec` one-shot command execution streamed as SSE.
//!
//! HTTP layer only: parse request, validate command policy, call core shell APIs.

use std::convert::Infallible;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::Frame;
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use utoipa::ToSchema;

use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

const DEFAULT_TIMEOUT_MS: u64 = 120_000;

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct ExecRequest {
    command: String,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    timeout_ms: Option<u64>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ExecEvent {
    #[serde(rename = "type")]
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<i32>,
}

#[utoipa::path(post, path = "/v1/agent/exec", tag = "Agent::Exec",
    request_body = ExecRequest,
    responses(
        (status = 200, description = "SSE stream with stdout/stderr/exit events. In strict mode, execution is sandbox-enforced; in host mode, passthrough has no firewall guarantee."),
        (status = 400, body = crate::server::http::ErrorResponse),
        (status = 403, body = crate::server::http::ErrorResponse),
        (status = 500, body = crate::server::http::ErrorResponse),
    ))]
pub async fn handle_exec(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e.to_string()),
    };

    let request: ExecRequest = match serde_json::from_slice(&body) {
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

    let timeout_ms = request.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS);
    let command = request.command;
    let cwd = if state.agent_runtime_mode == crate::server::AgentRuntimeMode::Strict {
        request
            .cwd
            .or_else(|| Some(state.workspace_dir.to_string_lossy().into_owned()))
    } else {
        request.cwd
    };
    let policy = super::load_runtime_policy(&state);
    let runtime_mode = super::runtime_mode_for_talu(&state);
    let sandbox_backend = super::sandbox_backend_for_talu(state.sandbox_backend);

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Bytes>();
    tokio::task::spawn_blocking(move || {
        let tx_stdout = tx.clone();
        let tx_stderr = tx.clone();

        let exec_result = talu::shell::exec_streaming_with_policy_runtime(
            &command,
            cwd.as_deref(),
            timeout_ms,
            policy.as_deref(),
            runtime_mode,
            sandbox_backend,
            move |chunk| {
                let payload = ExecEvent {
                    kind: "stdout".to_string(),
                    data: Some(String::from_utf8_lossy(chunk).into_owned()),
                    code: None,
                };
                tx_stdout.send(sse_data(&payload)).is_ok()
            },
            move |chunk| {
                let payload = ExecEvent {
                    kind: "stderr".to_string(),
                    data: Some(String::from_utf8_lossy(chunk).into_owned()),
                    code: None,
                };
                tx_stderr.send(sse_data(&payload)).is_ok()
            },
        );

        match exec_result {
            Ok(code) => {
                let payload = ExecEvent {
                    kind: "exit".to_string(),
                    data: None,
                    code,
                };
                let _ = tx.send(sse_data(&payload));
            }
            Err(err) => {
                let _ = tx.send(sse_data(&json!({
                    "type": "error",
                    "message": err.to_string(),
                })));
            }
        }
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

fn sse_data<T: Serialize>(payload: &T) -> Bytes {
    let value = serde_json::to_string(payload).unwrap_or_else(|_| "{}".to_string());
    Bytes::from(format!("data: {value}\n\n"))
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
