//! Collaboration API plane (`/v1/collab/resources/*`).
//!
//! This is the current server contract for collaborative resource sessions.
//! It persists collaboration state in KV durability lanes:
//! - strong: operation history + authoritative checkpoints
//! - batched: participant metadata
//! - ephemeral(+ttl): presence/liveness state

use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use base64::Engine as _;
use bytes::Bytes;
use futures_util::{stream, SinkExt, StreamExt};
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::upgrade::Upgraded;
use hyper::{Request, Response, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::protocol::Role;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;
use utoipa::ToSchema;
use hyper_util::rt::TokioIo;

use crate::server::auth_gateway::AuthContext;
use crate::server::code_ws;
use crate::server::events::{global_event_bus, EventDraft};
use crate::server::state::{AppState, CollabHandleEntry};

use talu::collab::{
    CollabError, CollabHandle, ParticipantKind as CoreParticipantKind,
    WatchDurability as CoreWatchDurability, WatchEventType as CoreWatchEventType,
};

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;
pub(crate) type SharedCollabHandle = Arc<CollabHandle>;
type WsStream = WebSocketStream<TokioIo<Upgraded>>;

const COLLAB_RESOURCE_PREFIX: &str = "/v1/collab/resources/";
const MAX_KIND_LEN: usize = 64;
const MAX_RESOURCE_ID_LEN: usize = 8 * 1024;
const MAX_PARTICIPANT_ID_LEN: usize = 256;
const MAX_OP_ID_LEN: usize = 256;
const MAX_HISTORY_LIMIT: usize = 5_000;
const WATCH_STREAM_CAPACITY: usize = 128;
const WATCH_HEARTBEAT_SECONDS: u64 = 15;
const WATCH_DRAIN_MAX_EVENTS: usize = 128;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ParticipantKind {
    Human,
    Agent,
    External,
    System,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OpenSessionRequest {
    pub participant_id: Option<String>,
    pub participant_kind: Option<ParticipantKind>,
    pub role: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct OpenSessionResponse {
    pub resource_kind: String,
    pub resource_id: String,
    pub namespace: String,
    pub participant_id: String,
    pub participant_kind: ParticipantKind,
    pub status: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ResourceSummaryResponse {
    pub resource_kind: String,
    pub resource_id: String,
    pub namespace: String,
    pub meta: Option<Value>,
    pub total_live_entries: usize,
    pub batched_pending: usize,
    pub ephemeral_live_entries: usize,
    pub watch_published: u64,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct SnapshotResponse {
    pub resource_kind: String,
    pub resource_id: String,
    pub namespace: String,
    pub snapshot_base64: Option<String>,
    pub updated_at_ms: Option<i64>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct SubmitOpRequest {
    pub actor_id: String,
    pub actor_seq: u64,
    pub op_id: String,
    pub payload_base64: String,
    pub issued_at_ms: Option<i64>,
    pub snapshot_base64: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct SubmitOpResponse {
    pub resource_kind: String,
    pub resource_id: String,
    pub namespace: String,
    pub op_key: String,
    pub accepted: bool,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct OpHistoryEntry {
    pub actor_id: String,
    pub actor_seq: u64,
    pub op_id: String,
    pub payload_base64: String,
    pub updated_at_ms: i64,
    pub key: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct HistoryResponse {
    pub resource_kind: String,
    pub resource_id: String,
    pub namespace: String,
    pub count: usize,
    pub next_cursor: Option<String>,
    pub data: Vec<OpHistoryEntry>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct PresencePutRequest {
    pub presence: Value,
    pub ttl_ms: Option<u64>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct PresencePutResponse {
    pub resource_kind: String,
    pub resource_id: String,
    pub namespace: String,
    pub participant_id: String,
    pub ttl_ms: u64,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct PresenceGetResponse {
    pub resource_kind: String,
    pub resource_id: String,
    pub namespace: String,
    pub participant_id: String,
    pub found: bool,
    pub presence: Option<Value>,
    pub updated_at_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct CollabWatchEvent {
    pub seq: u64,
    #[serde(rename = "type")]
    pub event_type: String,
    pub resource_kind: String,
    pub resource_id: String,
    pub namespace: String,
    pub key: String,
    pub value_len: usize,
    pub durability: Option<String>,
    pub ttl_ms: Option<u64>,
    pub updated_at_ms: i64,
}

#[derive(Debug, Deserialize)]
struct CollabWsOpenMessage {
    participant_id: Option<String>,
    participant_kind: Option<ParticipantKind>,
    role: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CollabWsSubmitOpMessage {
    actor_seq: u64,
    op_id: String,
    #[serde(default)]
    payload_base64: Option<String>,
    snapshot_base64: String,
    #[serde(default)]
    issued_at_ms: Option<i64>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum CollabWsClientMessage {
    Open(CollabWsOpenMessage),
    SubmitOp(CollabWsSubmitOpMessage),
    Ping,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum CollabWsServerMessage {
    Ready {
        resource_kind: String,
        resource_id: String,
        namespace: String,
        participant_id: String,
        participant_kind: ParticipantKind,
        status: String,
        snapshot_base64: Option<String>,
        updated_at_ms: Option<i64>,
    },
    Ack {
        op_key: String,
        actor_seq: u64,
        op_id: String,
    },
    Snapshot {
        resource_kind: String,
        resource_id: String,
        namespace: String,
        seq: u64,
        key: String,
        snapshot_base64: Option<String>,
        updated_at_ms: Option<i64>,
    },
    Error {
        code: String,
        message: String,
    },
    Pong,
}

#[utoipa::path(
    post,
    path = "/v1/collab/resources/{kind}/{id}/sessions",
    tag = "Collab::Resources",
    request_body = OpenSessionRequest,
    responses(
        (status = 200, description = "Session opened or joined", body = OpenSessionResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("kind" = String, Path, description = "Resource kind"),
        ("id" = String, Path, description = "Resource identifier")
    )
)]
pub async fn handle_open_session(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let parsed = match parse_resource_path(&path) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    if parsed.tail.as_slice() != ["sessions"] {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /v1/collab/resources/{kind}/{id}/sessions",
        );
    }

    let body_bytes = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                "failed to read request body",
            );
        }
    };
    let open_req = if body_bytes.is_empty() {
        OpenSessionRequest {
            participant_id: None,
            participant_kind: None,
            role: None,
        }
    } else {
        match serde_json::from_slice::<OpenSessionRequest>(&body_bytes) {
            Ok(v) => v,
            Err(e) => {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_json",
                    &format!("invalid JSON body: {e}"),
                );
            }
        }
    };

    let participant_id = open_req
        .participant_id
        .unwrap_or_else(|| "human:anonymous".to_string());
    if let Err(resp) = validate_participant_id(&participant_id) {
        return resp;
    }
    let participant_kind = open_req.participant_kind.unwrap_or(ParticipantKind::Human);

    let root = match resolve_storage_root(&state, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let collab = match open_collab_handle(&state, &root, &parsed.kind, &parsed.id).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let opened = match collab.open_session(
        &participant_id,
        core_participant_kind(participant_kind),
        open_req.role.as_deref(),
    ) {
        Ok(v) => v,
        Err(err) => return collab_error_response(err),
    };
    let namespace = opened.namespace.clone();

    publish_collab_event(
        auth.as_ref(),
        "session_open",
        "session opened",
        Some(serde_json::json!({
            "resource_kind": parsed.kind,
            "resource_id": parsed.id,
            "namespace": namespace,
            "participant_id": participant_id,
        })),
    );

    json_response(
        StatusCode::OK,
        &OpenSessionResponse {
            resource_kind: parsed.kind,
            resource_id: parsed.id,
            namespace,
            participant_id: opened.participant_id,
            participant_kind,
            status: opened.status,
        },
    )
}

#[utoipa::path(
    get,
    path = "/v1/collab/resources/{kind}/{id}",
    tag = "Collab::Resources",
    responses(
        (status = 200, description = "Resource summary", body = ResourceSummaryResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("kind" = String, Path, description = "Resource kind"),
        ("id" = String, Path, description = "Resource identifier")
    )
)]
pub async fn handle_get_resource(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let parsed = match parse_resource_path(&path) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    if !parsed.tail.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /v1/collab/resources/{kind}/{id}",
        );
    }

    let root = match resolve_storage_root(&state, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let collab = match open_collab_handle(&state, &root, &parsed.kind, &parsed.id).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let summary = match collab.summary() {
        Ok(v) => v,
        Err(err) => return collab_error_response(err),
    };
    let meta = summary
        .meta_json
        .as_deref()
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok());

    json_response(
        StatusCode::OK,
        &ResourceSummaryResponse {
            resource_kind: parsed.kind,
            resource_id: parsed.id,
            namespace: summary.namespace,
            meta,
            total_live_entries: summary.total_live_entries,
            batched_pending: summary.batched_pending,
            ephemeral_live_entries: summary.ephemeral_live_entries,
            watch_published: summary.watch_published,
        },
    )
}

#[utoipa::path(
    get,
    path = "/v1/collab/resources/{kind}/{id}/snapshot",
    tag = "Collab::Resources",
    responses(
        (status = 200, description = "Latest snapshot checkpoint", body = SnapshotResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("kind" = String, Path, description = "Resource kind"),
        ("id" = String, Path, description = "Resource identifier")
    )
)]
pub async fn handle_get_snapshot(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let parsed = match parse_resource_path(&path) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    if parsed.tail.as_slice() != ["snapshot"] {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /v1/collab/resources/{kind}/{id}/snapshot",
        );
    }

    let root = match resolve_storage_root(&state, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let collab = match open_collab_handle(&state, &root, &parsed.kind, &parsed.id).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let summary = match collab.summary() {
        Ok(v) => v,
        Err(err) => return collab_error_response(err),
    };
    let snapshot = match collab.snapshot() {
        Ok(v) => v,
        Err(err) => return collab_error_response(err),
    };

    json_response(
        StatusCode::OK,
        &SnapshotResponse {
            resource_kind: parsed.kind,
            resource_id: parsed.id,
            namespace: summary.namespace,
            snapshot_base64: snapshot
                .as_ref()
                .map(|v| base64::engine::general_purpose::STANDARD.encode(&v.data)),
            updated_at_ms: snapshot.map(|v| v.updated_at_ms),
        },
    )
}

#[utoipa::path(
    post,
    path = "/v1/collab/resources/{kind}/{id}/ops",
    tag = "Collab::Resources",
    request_body = SubmitOpRequest,
    responses(
        (status = 200, description = "Operation accepted", body = SubmitOpResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("kind" = String, Path, description = "Resource kind"),
        ("id" = String, Path, description = "Resource identifier")
    )
)]
pub async fn handle_submit_op(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let parsed = match parse_resource_path(&path) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    if parsed.tail.as_slice() != ["ops"] {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /v1/collab/resources/{kind}/{id}/ops",
        );
    }

    let body_bytes = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                "failed to read request body",
            );
        }
    };
    let op_req: SubmitOpRequest = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_json",
                &format!("invalid JSON body: {e}"),
            );
        }
    };
    if let Err(resp) = validate_actor_id(&op_req.actor_id) {
        return resp;
    }
    if op_req.actor_seq == 0 {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "actor_seq must be >= 1",
        );
    }
    if let Err(resp) = validate_op_id(&op_req.op_id) {
        return resp;
    }
    let payload = match base64::engine::general_purpose::STANDARD.decode(&op_req.payload_base64) {
        Ok(v) => v,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_argument",
                &format!("payload_base64 is invalid: {e}"),
            );
        }
    };
    if payload.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "payload_base64 must decode to non-empty bytes",
        );
    }

    let root = match resolve_storage_root(&state, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let collab = match open_collab_handle(&state, &root, &parsed.kind, &parsed.id).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let snapshot = match op_req.snapshot_base64.as_ref() {
        Some(snapshot_base64) => {
            match base64::engine::general_purpose::STANDARD.decode(snapshot_base64) {
                Ok(v) => Some(v),
                Err(e) => {
                    return json_error(
                        StatusCode::BAD_REQUEST,
                        "invalid_argument",
                        &format!("snapshot_base64 is invalid: {e}"),
                    );
                }
            }
        }
        None => None,
    };
    let submitted = match collab.submit_op(
        &op_req.actor_id,
        op_req.actor_seq,
        &op_req.op_id,
        &payload,
        op_req.issued_at_ms,
        snapshot.as_deref(),
    ) {
        Ok(v) => v,
        Err(err) => return collab_error_response(err),
    };
    let namespace = match collab.summary() {
        Ok(v) => v.namespace,
        Err(err) => return collab_error_response(err),
    };

    publish_collab_event(
        auth.as_ref(),
        "op_accepted",
        "collab op accepted",
        Some(serde_json::json!({
            "resource_kind": parsed.kind,
            "resource_id": parsed.id,
            "namespace": namespace,
            "op_key": submitted.op_key,
        })),
    );

    json_response(
        StatusCode::OK,
        &SubmitOpResponse {
            resource_kind: parsed.kind,
            resource_id: parsed.id,
            namespace,
            op_key: submitted.op_key,
            accepted: submitted.accepted,
        },
    )
}

#[utoipa::path(
    get,
    path = "/v1/collab/resources/{kind}/{id}/history",
    tag = "Collab::Resources",
    responses(
        (status = 200, description = "Operation history", body = HistoryResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("kind" = String, Path, description = "Resource kind"),
        ("id" = String, Path, description = "Resource identifier"),
        ("limit" = Option<usize>, Query, description = "Max operation records to return"),
        ("cursor" = Option<String>, Query, description = "Opaque pagination cursor from a prior history page")
    )
)]
pub async fn handle_get_history(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let parsed = match parse_resource_path(&path) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    if parsed.tail.as_slice() != ["history"] {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /v1/collab/resources/{kind}/{id}/history",
        );
    }
    let limit = match parse_limit(req.uri().query()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let cursor = match parse_cursor(req.uri().query()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let root = match resolve_storage_root(&state, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let cache_key = collab_cache_key(&root, &parsed.kind, &parsed.id);
    let collab = match open_collab_handle(&state, &root, &parsed.kind, &parsed.id).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    attach_collab_stream(&state, &cache_key).await;
    let namespace = match collab.summary() {
        Ok(v) => v.namespace,
        Err(err) => return collab_error_response(err),
    };
    let mut ops = match collab.history(cursor.as_deref(), limit.saturating_add(1)) {
        Ok(v) => v,
        Err(err) => return collab_error_response(err),
    };
    let next_cursor = if ops.len() > limit {
        ops.get(limit.saturating_sub(1)).map(|entry| entry.key.clone())
    } else {
        None
    };
    if ops.len() > limit {
        ops.truncate(limit);
    }
    let ops = ops
        .into_iter()
        .map(|entry| OpHistoryEntry {
            actor_id: entry.actor_id,
            actor_seq: entry.actor_seq,
            op_id: entry.op_id,
            payload_base64: base64::engine::general_purpose::STANDARD.encode(entry.payload),
            updated_at_ms: entry.updated_at_ms,
            key: entry.key,
        })
        .collect::<Vec<_>>();

    json_response(
        StatusCode::OK,
        &HistoryResponse {
            resource_kind: parsed.kind,
            resource_id: parsed.id,
            namespace,
            count: ops.len(),
            next_cursor,
            data: ops,
        },
    )
}

#[utoipa::path(
    put,
    path = "/v1/collab/resources/{kind}/{id}/presence/{participant_id}",
    tag = "Collab::Resources",
    request_body = PresencePutRequest,
    responses(
        (status = 200, description = "Presence upserted", body = PresencePutResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("kind" = String, Path, description = "Resource kind"),
        ("id" = String, Path, description = "Resource identifier"),
        ("participant_id" = String, Path, description = "Participant identifier")
    )
)]
pub async fn handle_put_presence(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let parsed = match parse_resource_path(&path) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    if parsed.tail.len() != 2 || parsed.tail[0] != "presence" {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /v1/collab/resources/{kind}/{id}/presence/{participant_id}",
        );
    }
    let participant_id = percent_decode(&parsed.tail[1]);
    if let Err(resp) = validate_participant_id(&participant_id) {
        return resp;
    }

    let body_bytes = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                "failed to read request body",
            );
        }
    };
    let put_req: PresencePutRequest = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_json",
                &format!("invalid JSON body: {e}"),
            );
        }
    };
    let root = match resolve_storage_root(&state, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let collab = match open_collab_handle(&state, &root, &parsed.kind, &parsed.id).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let payload = match serde_json::to_vec(&put_req.presence) {
        Ok(v) => v,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_argument",
                &format!("failed to serialize presence: {e}"),
            );
        }
    };
    let ttl_ms = match collab.put_presence(
        &participant_id,
        &payload,
        put_req.ttl_ms.unwrap_or(0),
    ) {
        Ok(v) => v,
        Err(err) => return collab_error_response(err),
    };
    let namespace = match collab.summary() {
        Ok(v) => v.namespace,
        Err(err) => return collab_error_response(err),
    };

    json_response(
        StatusCode::OK,
        &PresencePutResponse {
            resource_kind: parsed.kind,
            resource_id: parsed.id,
            namespace,
            participant_id,
            ttl_ms,
        },
    )
}

#[utoipa::path(
    get,
    path = "/v1/collab/resources/{kind}/{id}/presence/{participant_id}",
    tag = "Collab::Resources",
    responses(
        (status = 200, description = "Presence payload", body = PresenceGetResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("kind" = String, Path, description = "Resource kind"),
        ("id" = String, Path, description = "Resource identifier"),
        ("participant_id" = String, Path, description = "Participant identifier")
    )
)]
pub async fn handle_get_presence(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let parsed = match parse_resource_path(&path) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    if parsed.tail.len() != 2 || parsed.tail[0] != "presence" {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /v1/collab/resources/{kind}/{id}/presence/{participant_id}",
        );
    }
    let participant_id = percent_decode(&parsed.tail[1]);
    if let Err(resp) = validate_participant_id(&participant_id) {
        return resp;
    }

    let root = match resolve_storage_root(&state, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let collab = match open_collab_handle(&state, &root, &parsed.kind, &parsed.id).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let value = match collab.presence(&participant_id) {
        Ok(v) => v,
        Err(err) => return collab_error_response(err),
    };
    let namespace = match collab.summary() {
        Ok(v) => v.namespace,
        Err(err) => return collab_error_response(err),
    };
    let (found, presence, updated_at_ms) = if let Some(v) = value {
        (
            true,
            serde_json::from_slice::<Value>(&v.data).ok(),
            Some(v.updated_at_ms),
        )
    } else {
        (false, None, None)
    };

    json_response(
        StatusCode::OK,
        &PresenceGetResponse {
            resource_kind: parsed.kind,
            resource_id: parsed.id,
            namespace,
            participant_id,
            found,
            presence,
            updated_at_ms,
        },
    )
}

#[utoipa::path(
    get,
    path = "/v1/collab/resources/{kind}/{id}/events/stream",
    tag = "Collab::Resources",
    responses(
        (status = 200, description = "Collab watch stream (SSE)"),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("kind" = String, Path, description = "Resource kind"),
        ("id" = String, Path, description = "Resource identifier"),
        ("after_seq" = Option<u64>, Query, description = "Drain events with sequence > after_seq")
    )
)]
pub async fn handle_stream_events(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let accepts_sse = req
        .headers()
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .map(|v| {
            v.split(',')
                .map(|part| part.trim().to_ascii_lowercase())
                .any(|part| part.starts_with("text/event-stream"))
        })
        .unwrap_or(false);
    if !accepts_sse {
        return json_error(
            StatusCode::NOT_ACCEPTABLE,
            "invalid_accept",
            "events stream requires Accept: text/event-stream",
        );
    }

    let path = req.uri().path().to_string();
    let parsed = match parse_resource_path(&path) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    if parsed.tail.as_slice() != ["events", "stream"] {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /v1/collab/resources/{kind}/{id}/events/stream",
        );
    }
    let initial_after_seq = match parse_after_seq(req.uri().query()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let root = match resolve_storage_root(&state, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let cache_key = collab_cache_key(&root, &parsed.kind, &parsed.id);
    let collab = match open_collab_handle(&state, &root, &parsed.kind, &parsed.id).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    attach_collab_stream(&state, &cache_key).await;
    let namespace = match collab.summary() {
        Ok(v) => v.namespace,
        Err(err) => return collab_error_response(err),
    };

    let resource_kind = parsed.kind.clone();
    let resource_id = parsed.id.clone();
    let namespace_for_stream = namespace.clone();
    let queue = Arc::new(CollabWatchQueue::new(WATCH_STREAM_CAPACITY));
    let producer_queue = queue.clone();
    let state_for_stream = state.clone();
    tokio::spawn(async move {
        let mut after_seq = initial_after_seq;
        'producer: loop {
            let wait = match collab_watch_wait_blocking(
                collab.clone(),
                after_seq,
                WATCH_HEARTBEAT_SECONDS * 1_000,
            )
            .await
            {
                Ok(result) => result,
                Err(_) => {
                    producer_queue.close().await;
                    break 'producer;
                }
            };

            if wait.timed_out {
                if !producer_queue.push(Bytes::from_static(b": keepalive\n\n")).await {
                    producer_queue
                        .push_gap_and_close(watch_gap_frame(
                            &resource_kind,
                            &resource_id,
                            &namespace_for_stream,
                            "consumer_too_slow",
                            "consumer too slow",
                        ))
                        .await;
                    break;
                }
                continue;
            }

            loop {
                let batch = match collab.watch_drain(after_seq, WATCH_DRAIN_MAX_EVENTS) {
                    Ok(batch) => batch,
                    Err(_) => {
                        producer_queue.close().await;
                        break 'producer;
                    }
                };
                if batch.lost {
                    producer_queue
                        .push_gap_and_close(watch_gap_frame(
                            &resource_kind,
                            &resource_id,
                            &namespace_for_stream,
                            "source_gap",
                            "watch source gap",
                        ))
                        .await;
                    break;
                }
                if batch.events.is_empty() {
                    break;
                }
                for event in batch.events {
                    after_seq = event.seq;
                    let payload = watch_event_response(
                        &resource_kind,
                        &resource_id,
                        &namespace_for_stream,
                        event,
                    );
                    if !producer_queue.push(sse_json("event", &payload)).await {
                        producer_queue
                            .push_gap_and_close(watch_gap_frame(
                                &resource_kind,
                                &resource_id,
                                &namespace_for_stream,
                                "consumer_too_slow",
                                "consumer too slow",
                            ))
                            .await;
                        break 'producer;
                    }
                }
            }
        }
        detach_collab_stream(&state_for_stream, &cache_key).await;
    });

    let stream = stream::unfold(WatchStreamState { queue }, |state| async move {
        state
            .queue
            .next_frame()
            .await
            .map(|chunk| (Ok::<_, Infallible>(Frame::data(chunk)), state))
    });
    let body = BodyExt::boxed(StreamBody::new(stream));

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream; charset=utf-8")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .body(body)
        .unwrap()
}

#[utoipa::path(
    get,
    path = "/v1/collab/resources/{kind}/{id}/ws",
    tag = "Collab::Resources",
    responses(
        (status = 101, description = "WebSocket upgrade for active collaboration"),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("kind" = String, Path, description = "Resource kind"),
        ("id" = String, Path, description = "Resource identifier")
    )
)]
pub async fn handle_ws(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let parsed = match parse_resource_path(&path) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    if parsed.tail.as_slice() != ["ws"] {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /v1/collab/resources/{kind}/{id}/ws",
        );
    }

    let root = match resolve_storage_root(&state, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let cache_key = collab_cache_key(&root, &parsed.kind, &parsed.id);
    let collab = match open_collab_handle(&state, &root, &parsed.kind, &parsed.id).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    attach_collab_stream(&state, &cache_key).await;

    let key = match req.headers().get("sec-websocket-key") {
        Some(value) => value.as_bytes().to_vec(),
        None => {
            detach_collab_stream(&state, &cache_key).await;
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "Missing Sec-WebSocket-Key header",
            );
        }
    };
    let accept = code_ws::compute_accept_key(&key);

    let upgrade = hyper::upgrade::on(req);
    let state_for_ws = state.clone();
    let auth_for_ws = auth.clone();
    let resource_kind = parsed.kind.clone();
    let resource_id = parsed.id.clone();
    tokio::spawn(async move {
        match upgrade.await {
            Ok(upgraded) => {
                if let Err(err) = handle_ws_connection(
                    collab,
                    resource_kind.clone(),
                    resource_id.clone(),
                    auth_for_ws,
                    upgraded,
                )
                .await
                {
                    log::warn!(
                        target: "server::collab",
                        "collab ws {}/{} ended with error: {}",
                        resource_kind,
                        resource_id,
                        err,
                    );
                }
            }
            Err(err) => {
                log::warn!(
                    target: "server::collab",
                    "collab ws upgrade failed for {}/{}: {}",
                    resource_kind,
                    resource_id,
                    err,
                );
            }
        }
        detach_collab_stream(&state_for_ws, &cache_key).await;
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
    collab: SharedCollabHandle,
    resource_kind: String,
    resource_id: String,
    auth: Option<AuthContext>,
    upgraded: Upgraded,
) -> Result<(), String> {
    let ws: WsStream =
        WebSocketStream::from_raw_socket(TokioIo::new(upgraded), Role::Server, None).await;
    let (mut ws_sink, mut ws_stream) = ws.split();
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<CollabWsServerMessage>();
    let mut session_participant_id: Option<String> = None;
    let mut watch_task: Option<tokio::task::JoinHandle<()>> = None;

    loop {
        tokio::select! {
            maybe_out = out_rx.recv() => {
                let Some(message) = maybe_out else {
                    break;
                };
                ws_send_json_sink(&mut ws_sink, &message).await?;
            }
            maybe_msg = ws_stream.next() => {
                let msg = match maybe_msg {
                    Some(Ok(Message::Text(text))) => text,
                    Some(Ok(Message::Ping(data))) => {
                        ws_sink.send(Message::Pong(data)).await.map_err(|err| err.to_string())?;
                        continue;
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(err)) => return Err(format!("ws receive failed: {err}")),
                    _ => continue,
                };

                let request: CollabWsClientMessage = match serde_json::from_str(&msg) {
                    Ok(value) => value,
                    Err(err) => {
                        ws_send_collab_error_sink(&mut ws_sink, "invalid_json", &err.to_string()).await?;
                        continue;
                    }
                };

                match request {
                    CollabWsClientMessage::Open(open) => {
                        if session_participant_id.is_some() {
                            ws_send_collab_error_sink(&mut ws_sink, "already_open", "session already opened on this websocket").await?;
                            continue;
                        }
                        let participant_id = open
                            .participant_id
                            .unwrap_or_else(|| "human:anonymous".to_string());
                        if let Err(resp) = validate_participant_id(&participant_id) {
                            ws_send_http_error_sink(&mut ws_sink, resp).await?;
                            continue;
                        }
                        let participant_kind = open.participant_kind.unwrap_or(ParticipantKind::Human);
                        let opened = collab
                            .open_session(
                                &participant_id,
                                core_participant_kind(participant_kind),
                                open.role.as_deref(),
                            )
                            .map_err(|err| err.to_string())?;
                        let summary = collab.summary().map_err(|err| err.to_string())?;
                        let snapshot = collab.snapshot().map_err(|err| err.to_string())?;
                        let watch_tx = out_tx.clone();
                        let watch_collab = collab.clone();
                        let watch_resource_kind = resource_kind.clone();
                        let watch_resource_id = resource_id.clone();
                        let watch_namespace = opened.namespace.clone();
                        let mut watch_after_seq = summary.watch_published;
                        session_participant_id = Some(opened.participant_id.clone());

                        watch_task = Some(tokio::spawn(async move {
                            loop {
                                let wait = match collab_watch_wait_blocking(
                                    watch_collab.clone(),
                                    watch_after_seq,
                                    WATCH_HEARTBEAT_SECONDS * 1_000,
                                )
                                .await
                                {
                                    Ok(result) => result,
                                    Err(err) => {
                                        let _ = watch_tx.send(CollabWsServerMessage::Error {
                                            code: "watch_failed".to_string(),
                                            message: err,
                                        });
                                        return;
                                    }
                                };
                                if wait.timed_out {
                                    continue;
                                }
                                loop {
                                    let batch = match watch_collab.watch_drain(watch_after_seq, WATCH_DRAIN_MAX_EVENTS) {
                                        Ok(batch) => batch,
                                        Err(err) => {
                                            let _ = watch_tx.send(CollabWsServerMessage::Error {
                                                code: "watch_failed".to_string(),
                                                message: err.to_string(),
                                            });
                                            return;
                                        }
                                    };
                                    if batch.lost {
                                        let _ = watch_tx.send(CollabWsServerMessage::Error {
                                            code: "source_gap".to_string(),
                                            message: "watch source gap".to_string(),
                                        });
                                        return;
                                    }
                                    if batch.events.is_empty() {
                                        break;
                                    }

                                    let mut latest_snapshot_event: Option<(u64, String)> = None;
                                    for event in batch.events {
                                        watch_after_seq = event.seq;
                                        if event_affects_snapshot(&event.key) {
                                            latest_snapshot_event = Some((event.seq, event.key));
                                        }
                                    }

                                    if let Some((seq, key)) = latest_snapshot_event {
                                        let snapshot = match watch_collab.snapshot() {
                                            Ok(value) => value,
                                            Err(err) => {
                                                let _ = watch_tx.send(CollabWsServerMessage::Error {
                                                    code: "snapshot_failed".to_string(),
                                                    message: err.to_string(),
                                                });
                                                return;
                                            }
                                        };
                                        if watch_tx.send(CollabWsServerMessage::Snapshot {
                                            resource_kind: watch_resource_kind.clone(),
                                            resource_id: watch_resource_id.clone(),
                                            namespace: watch_namespace.clone(),
                                            seq,
                                            key,
                                            snapshot_base64: snapshot
                                                .as_ref()
                                                .map(|value| base64::engine::general_purpose::STANDARD.encode(&value.data)),
                                            updated_at_ms: snapshot.map(|value| value.updated_at_ms),
                                        }).is_err() {
                                            return;
                                        }
                                    }
                                }
                            }
                        }));

                        publish_collab_event(
                            auth.as_ref(),
                            "session_open",
                            "session opened",
                            Some(json!({
                                "resource_kind": resource_kind,
                                "resource_id": resource_id,
                                "namespace": opened.namespace,
                                "participant_id": opened.participant_id,
                                "transport": "websocket",
                            })),
                        );

                        ws_send_json_sink(
                            &mut ws_sink,
                            &CollabWsServerMessage::Ready {
                                resource_kind: resource_kind.clone(),
                                resource_id: resource_id.clone(),
                                namespace: opened.namespace,
                                participant_id: opened.participant_id,
                                participant_kind,
                                status: opened.status,
                                snapshot_base64: snapshot
                                    .as_ref()
                                    .map(|value| base64::engine::general_purpose::STANDARD.encode(&value.data)),
                                updated_at_ms: snapshot.map(|value| value.updated_at_ms),
                            },
                        ).await?;
                    }
                    CollabWsClientMessage::SubmitOp(submit) => {
                        let Some(actor_id) = session_participant_id.as_deref() else {
                            ws_send_collab_error_sink(&mut ws_sink, "session_not_open", "send an open message before submit_op").await?;
                            continue;
                        };
                        if submit.actor_seq == 0 {
                            ws_send_collab_error_sink(&mut ws_sink, "invalid_argument", "actor_seq must be >= 1").await?;
                            continue;
                        }
                        if let Err(resp) = validate_op_id(&submit.op_id) {
                            ws_send_http_error_sink(&mut ws_sink, resp).await?;
                            continue;
                        }
                        let snapshot = match base64::engine::general_purpose::STANDARD.decode(&submit.snapshot_base64) {
                            Ok(value) => value,
                            Err(err) => {
                                ws_send_collab_error_sink(&mut ws_sink, "invalid_argument", &format!("snapshot_base64 is invalid: {err}")).await?;
                                continue;
                            }
                        };
                        let payload = match submit.payload_base64 {
                            Some(payload_base64) => match base64::engine::general_purpose::STANDARD.decode(payload_base64) {
                                Ok(value) => value,
                                Err(err) => {
                                    ws_send_collab_error_sink(&mut ws_sink, "invalid_argument", &format!("payload_base64 is invalid: {err}")).await?;
                                    continue;
                                }
                            },
                            None => serde_json::to_vec(&json!({
                                "type": "snapshot",
                                "source": "collab.ws",
                                "resource_kind": resource_kind,
                                "resource_id": resource_id,
                                "bytes": snapshot.len(),
                            }))
                            .map_err(|err| err.to_string())?,
                        };
                        let submitted = collab
                            .submit_op(
                                actor_id,
                                submit.actor_seq,
                                &submit.op_id,
                                &payload,
                                submit.issued_at_ms,
                                Some(&snapshot),
                            )
                            .map_err(|err| err.to_string())?;

                        publish_collab_event(
                            auth.as_ref(),
                            "op_accepted",
                            "collab op accepted",
                            Some(json!({
                                "resource_kind": resource_kind,
                                "resource_id": resource_id,
                                "namespace": collab_namespace(&collab).await?,
                                "op_key": submitted.op_key,
                                "transport": "websocket",
                            })),
                        );

                        ws_send_json_sink(
                            &mut ws_sink,
                            &CollabWsServerMessage::Ack {
                                op_key: submitted.op_key,
                                actor_seq: submit.actor_seq,
                                op_id: submit.op_id,
                            },
                        ).await?;
                    }
                    CollabWsClientMessage::Ping => {
                        ws_send_json_sink(&mut ws_sink, &CollabWsServerMessage::Pong).await?;
                    }
                }
            }
        }
    }

    if let Some(task) = watch_task {
        task.abort();
    }
    Ok(())
}

pub fn is_collab_resource_prefix(path: &str) -> bool {
    path.starts_with(COLLAB_RESOURCE_PREFIX)
}

pub fn is_collab_resource_root_path(path: &str) -> bool {
    let Ok(parsed) = parse_resource_path(path) else {
        return false;
    };
    parsed.tail.is_empty()
}

pub fn is_collab_session_open_path(path: &str) -> bool {
    let Ok(parsed) = parse_resource_path(path) else {
        return false;
    };
    parsed.tail.as_slice() == ["sessions"]
}

pub fn is_collab_snapshot_path(path: &str) -> bool {
    let Ok(parsed) = parse_resource_path(path) else {
        return false;
    };
    parsed.tail.as_slice() == ["snapshot"]
}

pub fn is_collab_ops_path(path: &str) -> bool {
    let Ok(parsed) = parse_resource_path(path) else {
        return false;
    };
    parsed.tail.as_slice() == ["ops"]
}

pub fn is_collab_history_path(path: &str) -> bool {
    let Ok(parsed) = parse_resource_path(path) else {
        return false;
    };
    parsed.tail.as_slice() == ["history"]
}

pub fn is_collab_presence_path(path: &str) -> bool {
    let Ok(parsed) = parse_resource_path(path) else {
        return false;
    };
    parsed.tail.len() == 2 && parsed.tail[0] == "presence"
}

pub fn is_collab_events_stream_path(path: &str) -> bool {
    let Ok(parsed) = parse_resource_path(path) else {
        return false;
    };
    parsed.tail.as_slice() == ["events", "stream"]
}

pub fn is_collab_ws_path(path: &str) -> bool {
    let Ok(parsed) = parse_resource_path(path) else {
        return false;
    };
    parsed.tail.as_slice() == ["ws"]
}

#[derive(Debug)]
struct ParsedResourcePath {
    kind: String,
    id: String,
    tail: Vec<String>,
}

fn parse_resource_path(path: &str) -> Result<ParsedResourcePath, Response<BoxBody>> {
    let Some(stripped) = path.strip_prefix(COLLAB_RESOURCE_PREFIX) else {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "missing collab resource prefix",
        ));
    };
    let mut parts = stripped.split('/');
    let kind_raw = parts.next().unwrap_or("");
    let id_raw = parts.next().unwrap_or("");
    if kind_raw.is_empty() || id_raw.is_empty() {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /v1/collab/resources/{kind}/{id}",
        ));
    }

    let kind = percent_decode(kind_raw);
    let id = percent_decode(id_raw);
    if let Err(resp) = validate_kind(&kind) {
        return Err(resp);
    }
    if let Err(resp) = validate_resource_id(&id) {
        return Err(resp);
    }

    let tail = parts.map(percent_decode).collect::<Vec<_>>();
    Ok(ParsedResourcePath { kind, id, tail })
}

fn validate_kind(kind: &str) -> Result<(), Response<BoxBody>> {
    if kind.is_empty() || kind.trim().is_empty() {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "resource kind must be non-empty",
        ));
    }
    if kind.len() > MAX_KIND_LEN {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "resource kind is too long",
        ));
    }
    if kind
        .bytes()
        .any(|b| b == 0 || b.is_ascii_control() || b == b'/')
    {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "resource kind contains invalid characters",
        ));
    }
    Ok(())
}

fn validate_resource_id(id: &str) -> Result<(), Response<BoxBody>> {
    if id.is_empty() {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "resource id must be non-empty",
        ));
    }
    if id.len() > MAX_RESOURCE_ID_LEN {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "resource id is too long",
        ));
    }
    if id.bytes().any(|b| b == 0 || b.is_ascii_control()) {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "resource id contains invalid control characters",
        ));
    }
    Ok(())
}

fn validate_participant_id(participant_id: &str) -> Result<(), Response<BoxBody>> {
    if participant_id.is_empty() || participant_id.len() > MAX_PARTICIPANT_ID_LEN {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "participant_id must be non-empty and <= 256 bytes",
        ));
    }
    if participant_id
        .bytes()
        .any(|b| b == 0 || b.is_ascii_control() || b == b'/')
    {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "participant_id contains invalid characters",
        ));
    }
    Ok(())
}

fn validate_actor_id(actor_id: &str) -> Result<(), Response<BoxBody>> {
    if actor_id.is_empty() || actor_id.len() > MAX_PARTICIPANT_ID_LEN {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "actor_id must be non-empty and <= 256 bytes",
        ));
    }
    if actor_id.bytes().any(|b| b == 0 || b.is_ascii_control()) {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "actor_id contains invalid control characters",
        ));
    }
    Ok(())
}

fn validate_op_id(op_id: &str) -> Result<(), Response<BoxBody>> {
    if op_id.is_empty() || op_id.len() > MAX_OP_ID_LEN {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "op_id must be non-empty and <= 256 bytes",
        ));
    }
    if op_id
        .bytes()
        .any(|b| b == 0 || b.is_ascii_control() || b == b':' || b == b'/')
    {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "op_id contains invalid characters",
        ));
    }
    Ok(())
}

fn parse_limit(query: Option<&str>) -> Result<usize, Response<BoxBody>> {
    let Some(query) = query else {
        return Ok(200);
    };
    let raw = parse_query_param(query, "limit");
    let Some(raw) = raw else {
        return Ok(200);
    };
    let value = raw.parse::<usize>().map_err(|_| {
        json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "limit must be a positive integer",
        )
    })?;
    if value == 0 || value > MAX_HISTORY_LIMIT {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "limit must be in range 1..=5000",
        ));
    }
    Ok(value)
}

fn parse_after_seq(query: Option<&str>) -> Result<u64, Response<BoxBody>> {
    let Some(query) = query else {
        return Ok(0);
    };
    let raw = parse_query_param(query, "after_seq");
    let Some(raw) = raw else {
        return Ok(0);
    };
    raw.parse::<u64>().map_err(|_| {
        json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "after_seq must be a non-negative integer",
        )
    })
}

fn parse_cursor(query: Option<&str>) -> Result<Option<String>, Response<BoxBody>> {
    let Some(query) = query else {
        return Ok(None);
    };
    let raw = parse_query_param(query, "cursor");
    let Some(raw) = raw else {
        return Ok(None);
    };
    let decoded = percent_decode(raw);
    if decoded.is_empty()
        || decoded
            .bytes()
            .any(|b| b == 0 || b.is_ascii_control())
    {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "cursor must be a non-empty opaque token",
        ));
    }
    Ok(Some(decoded))
}

fn parse_query_param<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query
        .split('&')
        .filter_map(|part| part.split_once('='))
        .find_map(|(k, v)| if k == key { Some(v) } else { None })
}

fn percent_decode(raw: &str) -> String {
    percent_encoding::percent_decode_str(raw)
        .decode_utf8_lossy()
        .into_owned()
}

fn core_participant_kind(kind: ParticipantKind) -> CoreParticipantKind {
    match kind {
        ParticipantKind::Human => CoreParticipantKind::Human,
        ParticipantKind::Agent => CoreParticipantKind::Agent,
        ParticipantKind::External => CoreParticipantKind::External,
        ParticipantKind::System => CoreParticipantKind::System,
    }
}

pub(crate) async fn open_collab_handle(
    state: &Arc<AppState>,
    root: &str,
    resource_kind: &str,
    resource_id: &str,
) -> Result<SharedCollabHandle, Response<BoxBody>> {
    let cache_key = collab_cache_key(root, resource_kind, resource_id);
    {
        let mut cache = state.collab_handles.lock().await;
        if let Some(existing) = cache.get_mut(&cache_key) {
            existing.last_access = Instant::now();
            return Ok(existing.handle.clone());
        }
    }

    let opened =
        CollabHandle::open(root, resource_kind, resource_id).map_err(collab_error_response)?;
    let shared = Arc::new(opened);

    let mut cache = state.collab_handles.lock().await;
    if let Some(existing) = cache.get_mut(&cache_key) {
        existing.last_access = Instant::now();
        return Ok(existing.handle.clone());
    }
    cache.insert(
        cache_key,
        CollabHandleEntry {
            handle: shared.clone(),
            last_access: Instant::now(),
            active_streams: 0,
        },
    );
    Ok(shared)
}

async fn collab_namespace(collab: &SharedCollabHandle) -> Result<String, String> {
    collab.summary().map(|summary| summary.namespace).map_err(|err| err.to_string())
}

async fn collab_watch_wait_blocking(
    collab: SharedCollabHandle,
    after_seq: u64,
    timeout_ms: u64,
) -> Result<talu::collab::WatchWaitResult, String> {
    tokio::task::spawn_blocking(move || collab.watch_wait(after_seq, timeout_ms))
        .await
        .map_err(|err| err.to_string())?
        .map_err(|err| err.to_string())
}

fn collab_cache_key(root: &str, resource_kind: &str, resource_id: &str) -> String {
    format!("{root}\0{resource_kind}\0{resource_id}")
}

async fn attach_collab_stream(state: &Arc<AppState>, cache_key: &str) {
    let mut cache = state.collab_handles.lock().await;
    if let Some(entry) = cache.get_mut(cache_key) {
        entry.active_streams = entry.active_streams.saturating_add(1);
        entry.last_access = Instant::now();
    }
}

async fn detach_collab_stream(state: &Arc<AppState>, cache_key: &str) {
    let mut cache = state.collab_handles.lock().await;
    if let Some(entry) = cache.get_mut(cache_key) {
        entry.active_streams = entry.active_streams.saturating_sub(1);
        entry.last_access = Instant::now();
    }
}

pub(crate) fn resolve_storage_root(
    state: &AppState,
    auth: Option<&AuthContext>,
) -> Result<String, Response<BoxBody>> {
    let bucket = match state.bucket_path.as_ref() {
        Some(v) => v,
        None => {
            return Err(json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no_storage",
                "Storage not configured",
            ));
        }
    };
    let base = match auth {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };
    let root = base.join("kv");
    std::fs::create_dir_all(&root).map_err(|e| {
        json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage_error",
            &format!("failed to create storage root: {e}"),
        )
    })?;
    Ok(root.to_string_lossy().to_string())
}

fn publish_collab_event(
    auth: Option<&AuthContext>,
    topic: &str,
    message: &str,
    data: Option<Value>,
) {
    global_event_bus().publish(EventDraft {
        verbosity_min: 0,
        level: "info".to_string(),
        domain: "collab".to_string(),
        topic: topic.to_string(),
        event_class: "collab".to_string(),
        message: message.to_string(),
        tenant_id: auth.map(|ctx| ctx.tenant_id.clone()),
        correlation: None,
        data,
    });
}

fn collab_error_response(err: CollabError) -> Response<BoxBody> {
    match err {
        CollabError::InvalidArgument(msg) => {
            json_error(StatusCode::BAD_REQUEST, "invalid_argument", &msg)
        }
        CollabError::Busy(msg) => json_error(StatusCode::CONFLICT, "resource_busy", &msg),
        CollabError::ResourceExhausted(msg) => {
            json_error(StatusCode::TOO_MANY_REQUESTS, "resource_exhausted", &msg)
        }
        CollabError::Storage(msg) => {
            json_error(StatusCode::SERVICE_UNAVAILABLE, "storage_error", &msg)
        }
    }
}

fn event_affects_snapshot(key: &str) -> bool {
    key == "checkpoints/snapshot" || key.starts_with("ops/")
}

async fn ws_send_json_sink<S>(ws: &mut S, message: &CollabWsServerMessage) -> Result<(), String>
where
    S: futures_util::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
{
    let payload = serde_json::to_string(message).map_err(|err| err.to_string())?;
    ws.send(Message::Text(payload.into()))
        .await
        .map_err(|err| err.to_string())
}

async fn ws_send_collab_error_sink<S>(
    ws: &mut S,
    code: &str,
    message: &str,
) -> Result<(), String>
where
    S: futures_util::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
{
    ws_send_json_sink(
        ws,
        &CollabWsServerMessage::Error {
            code: code.to_string(),
            message: message.to_string(),
        },
    )
    .await
}

async fn ws_send_http_error_sink<S>(ws: &mut S, resp: Response<BoxBody>) -> Result<(), String>
where
    S: futures_util::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
{
    let body_bytes = resp
        .into_body()
        .collect()
        .await
        .map_err(|err| err.to_string())?
        .to_bytes();
    let json = serde_json::from_slice::<Value>(&body_bytes).map_err(|err| err.to_string())?;
    let code = json
        .get("error")
        .and_then(|value| value.get("code"))
        .and_then(Value::as_str)
        .unwrap_or("invalid_request");
    let message = json
        .get("error")
        .and_then(|value| value.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("request failed");
    ws_send_collab_error_sink(ws, code, message).await
}

fn json_response<T: Serialize>(status: StatusCode, body: &T) -> Response<BoxBody> {
    let bytes = serde_json::to_vec(body).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(bytes)).boxed())
        .unwrap()
}

fn json_error(status: StatusCode, code: &str, message: &str) -> Response<BoxBody> {
    json_response(
        status,
        &serde_json::json!({
            "error": {
                "code": code,
                "message": message,
            }
        }),
    )
}

fn sse_json<T: Serialize>(event: &str, payload: &T) -> Bytes {
    let data = serde_json::to_string(payload).unwrap_or_else(|_| "{}".to_string());
    Bytes::from(format!("event: {event}\ndata: {data}\n\n"))
}

fn watch_gap_frame(
    resource_kind: &str,
    resource_id: &str,
    namespace: &str,
    gap_type: &str,
    message: &str,
) -> Bytes {
    sse_json(
        "gap",
        &serde_json::json!({
            "type": gap_type,
            "message": message,
            "resource_kind": resource_kind,
            "resource_id": resource_id,
            "namespace": namespace,
        }),
    )
}

fn watch_event_response(
    resource_kind: &str,
    resource_id: &str,
    namespace: &str,
    event: talu::collab::WatchEvent,
) -> CollabWatchEvent {
    CollabWatchEvent {
        seq: event.seq,
        event_type: match event.event_type {
            CoreWatchEventType::Put => "put".to_string(),
            CoreWatchEventType::Delete => "delete".to_string(),
        },
        resource_kind: resource_kind.to_string(),
        resource_id: resource_id.to_string(),
        namespace: namespace.to_string(),
        key: event.key,
        value_len: event.value_len,
        durability: event
            .durability
            .map(durability_to_str)
            .map(ToString::to_string),
        ttl_ms: event.ttl_ms,
        updated_at_ms: event.updated_at_ms,
    }
}

fn durability_to_str(durability: CoreWatchDurability) -> &'static str {
    match durability {
        CoreWatchDurability::Strong => "strong",
        CoreWatchDurability::Batched => "batched",
        CoreWatchDurability::Ephemeral => "ephemeral",
    }
}

struct CollabWatchQueue {
    state: tokio::sync::Mutex<CollabWatchQueueState>,
    notify: tokio::sync::Notify,
}

struct CollabWatchQueueState {
    frames: VecDeque<Bytes>,
    closed: bool,
    capacity: usize,
}

impl CollabWatchQueue {
    fn new(capacity: usize) -> Self {
        Self {
            state: tokio::sync::Mutex::new(CollabWatchQueueState {
                frames: VecDeque::with_capacity(capacity),
                closed: false,
                capacity,
            }),
            notify: tokio::sync::Notify::new(),
        }
    }

    async fn push(&self, frame: Bytes) -> bool {
        let mut state = self.state.lock().await;
        if state.closed {
            return false;
        }
        if state.frames.len() >= state.capacity {
            return false;
        }
        state.frames.push_back(frame);
        drop(state);
        self.notify.notify_one();
        true
    }

    async fn push_gap_and_close(&self, frame: Bytes) {
        let mut state = self.state.lock().await;
        if state.closed {
            return;
        }
        state.frames.clear();
        state.frames.push_back(frame);
        state.closed = true;
        drop(state);
        self.notify.notify_waiters();
    }

    async fn close(&self) {
        let mut state = self.state.lock().await;
        state.closed = true;
        drop(state);
        self.notify.notify_waiters();
    }

    async fn next_frame(&self) -> Option<Bytes> {
        loop {
            let notified = {
                let mut state = self.state.lock().await;
                if let Some(frame) = state.frames.pop_front() {
                    return Some(frame);
                }
                if state.closed {
                    return None;
                }
                self.notify.notified()
            };
            notified.await;
        }
    }
}

struct WatchStreamState {
    queue: Arc<CollabWatchQueue>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_participant_kind_maps_server_schema() {
        assert_eq!(
            core_participant_kind(ParticipantKind::Human),
            CoreParticipantKind::Human
        );
        assert_eq!(
            core_participant_kind(ParticipantKind::Agent),
            CoreParticipantKind::Agent
        );
    }

    #[test]
    fn collab_path_matchers_detect_expected_shapes() {
        let root = "/v1/collab/resources/text_document/doc-1";
        assert!(is_collab_resource_root_path(root));
        assert!(is_collab_session_open_path(
            &(root.to_string() + "/sessions")
        ));
        assert!(is_collab_ops_path(&(root.to_string() + "/ops")));
        assert!(is_collab_snapshot_path(&(root.to_string() + "/snapshot")));
        assert!(is_collab_history_path(&(root.to_string() + "/history")));
        assert!(is_collab_events_stream_path(
            &(root.to_string() + "/events/stream")
        ));
        assert!(is_collab_presence_path(
            &(root.to_string() + "/presence/human%3A1")
        ));
    }

    #[tokio::test]
    async fn collab_watch_queue_overflow_rejects_new_frames() {
        let queue = CollabWatchQueue::new(1);
        assert!(queue.push(Bytes::from_static(b"event-1")).await);
        assert!(!queue.push(Bytes::from_static(b"event-2")).await);
    }
}
