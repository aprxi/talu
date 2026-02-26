//! Session management handlers for `/v1/chat/sessions` endpoints.

use std::convert::Infallible;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde::Serialize;
use serde_json::json;
use utoipa::ToSchema;

use talu::responses::ResponsesView;
use talu::storage::{SessionCursor, SessionUpdate, StorageError, StorageHandle};

use crate::server::auth_gateway::AuthContext;
use crate::server::http;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

/// Tag object embedded in session responses.
#[derive(Serialize, ToSchema)]
pub(crate) struct SessionTag {
    pub id: String,
    pub name: String,
    pub color: Option<String>,
}

/// Documentation-only: shape returned by session list/get endpoints.
#[derive(Serialize, ToSchema)]
pub(crate) struct SessionResponse {
    pub id: String,
    pub title: Option<String>,
    pub model: Option<String>,
    pub marker: Option<String>,
    pub project_id: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
    pub tags: Vec<SessionTag>,
    /// Session items (messages). Present on single-get, absent on list.
    #[schema(value_type = Option<Vec<Object>>)]
    pub items: Option<Vec<serde_json::Value>>,
}

/// Documentation-only: shape returned by GET /v1/chat/sessions.
#[derive(Serialize, ToSchema)]
pub(crate) struct SessionListResponse {
    pub object: String,
    pub data: Vec<SessionResponse>,
    pub has_more: bool,
    pub cursor: Option<String>,
    pub total: usize,
}

/// Request body for PATCH /v1/chat/sessions/:id.
#[derive(Serialize, ToSchema)]
pub(crate) struct SessionPatchRequest {
    pub title: Option<String>,
    pub marker: Option<String>,
    pub pinned: Option<bool>,
    pub project_id: Option<String>,
}

/// Request body for POST /v1/chat/sessions/batch.
#[derive(Serialize, ToSchema)]
pub(crate) struct BatchRequest {
    pub action: String,
    pub ids: Vec<String>,
}

/// Request body for POST /v1/chat/sessions/:id/fork.
#[derive(Serialize, ToSchema)]
pub(crate) struct ForkRequest {
    pub after_item: Option<u64>,
}

/// Request body for tag operations on sessions.
#[derive(Serialize, ToSchema)]
pub(crate) struct TagsRequest {
    pub tags: Vec<String>,
}

/// Open a StorageHandle from the app state, returning 503 if no bucket configured.
///
/// When gateway auth provides a `storage_prefix`, the storage path is
/// resolved as `<base_bucket>/<storage_prefix>/`, giving each tenant an
/// isolated subdirectory. Without auth (or without gateway mode), the
/// base bucket path is used directly.
fn open_storage(
    state: &AppState,
    auth: &Option<AuthContext>,
) -> Result<StorageHandle, Response<BoxBody>> {
    let base = state.bucket_path.as_ref().ok_or_else(|| {
        json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "storage_unavailable",
            "Storage disabled (running with --no-bucket)",
        )
    })?;

    let path = match auth {
        Some(ctx) => base.join(&ctx.storage_prefix),
        None => base.to_path_buf(),
    };

    StorageHandle::open(&path).map_err(|e| {
        json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage_error",
            &format!("{e}"),
        )
    })
}

/// Extract session ID from path like `/v1/chat/sessions/{id}`.
fn extract_session_id(path: &str) -> Option<&str> {
    let rest = path.strip_prefix("/v1/chat/sessions/")?;
    // Take everything up to the next '/' (or end) as the ID
    let id = rest.split('/').next()?;
    if id.is_empty() {
        None
    } else {
        Some(id)
    }
}

/// Parse query string into key-value pairs.
fn parse_query(uri: &hyper::Uri) -> Vec<(String, String)> {
    uri.query()
        .map(|q| {
            q.split('&')
                .filter_map(|pair| {
                    let mut parts = pair.splitn(2, '=');
                    let key = parts.next()?;
                    let value = parts.next().unwrap_or("");
                    Some((urldecode(key), urldecode(value)))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Minimal percent-decoding (covers the common cases for cursor/group_id).
fn urldecode(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.bytes();
    while let Some(b) = chars.next() {
        if b == b'+' {
            result.push(' ');
        } else if b == b'%' {
            let hi = chars.next().and_then(|c| hex_val(c));
            let lo = chars.next().and_then(|c| hex_val(c));
            if let (Some(h), Some(l)) = (hi, lo) {
                result.push((h << 4 | l) as char);
            }
        } else {
            result.push(b as char);
        }
    }
    result
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Encode a cursor as an opaque string: `"{updated_at_ms}:{session_id}"`.
pub(crate) fn encode_cursor(cursor: &SessionCursor) -> String {
    let raw = format!("{}:{}", cursor.updated_at_ms, cursor.session_id);
    base64_encode(raw.as_bytes())
}

/// Decode an opaque cursor string back to SessionCursor.
pub(crate) fn decode_cursor(encoded: &str) -> Option<SessionCursor> {
    let decoded = base64_decode(encoded)?;
    let s = std::str::from_utf8(&decoded).ok()?;
    let (ts_str, session_id) = s.split_once(':')?;
    let updated_at_ms = ts_str.parse::<i64>().ok()?;
    Some(SessionCursor {
        updated_at_ms,
        session_id: session_id.to_string(),
    })
}

/// Simple base64 encode (standard alphabet, with padding).
fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity((input.len() + 2) / 3 * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        result.push(ALPHABET[(n >> 18 & 0x3F) as usize] as char);
        result.push(ALPHABET[(n >> 12 & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(ALPHABET[(n >> 6 & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(ALPHABET[(n & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

/// Simple base64 decode (standard alphabet, with padding).
fn base64_decode(input: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'+' => Some(62),
            b'/' => Some(63),
            b'=' => Some(0),
            _ => None,
        }
    }
    let bytes = input.as_bytes();
    if bytes.len() % 4 != 0 {
        return None;
    }
    let mut result = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let a = val(chunk[0])?;
        let b = val(chunk[1])?;
        let c = val(chunk[2])?;
        let d = val(chunk[3])?;
        let n = (a << 18) | (b << 12) | (c << 6) | d;
        result.push((n >> 16) as u8);
        if chunk[2] != b'=' {
            result.push((n >> 8 & 0xFF) as u8);
        }
        if chunk[3] != b'=' {
            result.push((n & 0xFF) as u8);
        }
    }
    Some(result)
}

/// Convert a SessionRecordFull to a JSON session resource (without items).
/// Fields are aligned with CSessionRecord in core/src/capi/db.zig.
pub(crate) fn session_to_session_json(
    rec: &talu::storage::SessionRecordFull,
    tags: Option<Vec<serde_json::Value>>,
) -> serde_json::Value {
    let metadata: serde_json::Value = rec
        .metadata_json
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_else(|| json!({}));

    let config: serde_json::Value = rec
        .config_json
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_else(|| json!(null));

    json!({
        "id": rec.session_id,
        "object": "session",
        "model": rec.model,
        "title": rec.title,
        "system_prompt": rec.system_prompt,
        "config": config,
        "marker": rec.marker,
        "parent_session_id": rec.parent_session_id,
        "group_id": rec.group_id,
        "head_item_id": rec.head_item_id,
        "ttl_ts": rec.ttl_ts,
        "metadata": metadata,
        "tags": tags.unwrap_or_default(),
        "search_snippet": rec.search_snippet,
        "source_doc_id": rec.source_doc_id,
        "project_id": rec.project_id,
        "created_at": rec.created_at,
        "updated_at": rec.updated_at
    })
}

/// Convert a slim SessionRecord to a JSON session resource (without items).
/// Fields are aligned with CSessionRecord in core/src/capi/db.zig.
/// Fields not available in the slim record are set to null/default.
fn session_slim_to_session_json(
    rec: &talu::storage::SessionRecord,
    tags: Option<Vec<serde_json::Value>>,
) -> serde_json::Value {
    json!({
        "id": rec.session_id,
        "object": "session",
        "model": rec.model,
        "title": rec.title,
        "system_prompt": null,
        "config": null,
        "marker": rec.marker,
        "parent_session_id": null,
        "group_id": null,
        "head_item_id": 0,
        "ttl_ts": 0,
        "metadata": {},
        "tags": tags.unwrap_or_default(),
        "created_at": rec.created_at,
        "updated_at": rec.updated_at
    })
}

/// Resolve tag IDs for a session into tag objects [{id, name, color}].
pub(crate) fn resolve_tags_for_session(
    storage: &StorageHandle,
    session_id: &str,
) -> Vec<serde_json::Value> {
    let tag_ids = match storage.get_session_tags(session_id) {
        Ok(ids) => ids,
        Err(_) => return Vec::new(),
    };

    let mut tags = Vec::with_capacity(tag_ids.len());
    for tag_id in &tag_ids {
        if let Ok(tag) = storage.get_tag(tag_id) {
            tags.push(json!({
                "id": tag.tag_id,
                "name": tag.name,
                "color": tag.color,
            }));
        }
    }
    tags
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

#[utoipa::path(get, path = "/v1/chat/sessions", tag = "Chat",
    params(
        ("limit" = Option<usize>, Query, description = "Max items to return (default 50, max 100)"),
        ("offset" = Option<usize>, Query, description = "Number of items to skip"),
        ("group_id" = Option<String>, Query, description = "Filter by group ID"),
        ("marker" = Option<String>, Query, description = "Filter by marker value"),
        ("search" = Option<String>, Query, description = "Full-text search on titles"),
        ("tags_any" = Option<String>, Query, description = "Comma-separated tag IDs (OR filter)"),
        ("project_id" = Option<String>, Query, description = "Filter by project ID"),
    ),
    responses((status = 200, body = SessionListResponse)))]
/// GET /v1/chat/sessions — list sessions with offset-based pagination.
pub async fn handle_list(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let storage = match open_storage(&state, &auth) {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    let query = parse_query(req.uri());
    let limit = query
        .iter()
        .find(|(k, _)| k == "limit")
        .and_then(|(_, v)| v.parse::<usize>().ok())
        .unwrap_or(50)
        .clamp(1, 100);

    let offset = query
        .iter()
        .find(|(k, _)| k == "offset")
        .and_then(|(_, v)| v.parse::<usize>().ok())
        .unwrap_or(0);

    // group_id: query param takes precedence, then auth header.
    let group_id: Option<String> = query
        .iter()
        .find(|(k, _)| k == "group_id")
        .map(|(_, v)| v.clone())
        .or_else(|| auth.and_then(|a| a.group_id));

    let marker_filter: Option<String> = query
        .iter()
        .find(|(k, _)| k == "marker")
        .map(|(_, v)| v.clone());

    let search_query: Option<String> = query
        .iter()
        .find(|(k, _)| k == "search")
        .map(|(_, v)| v.clone())
        .filter(|v| !v.is_empty());

    let tags_any: Option<String> = query
        .iter()
        .find(|(k, _)| k == "tags_any")
        .map(|(_, v)| v.clone())
        .filter(|v| !v.is_empty());

    let project_id_raw: Option<String> = query
        .iter()
        .find(|(k, _)| k == "project_id")
        .map(|(_, v)| v.clone())
        .filter(|v| !v.is_empty());

    let project_id_null = project_id_raw.as_deref() == Some("__default__");
    let project_id: Option<String> = if project_id_null {
        None
    } else {
        project_id_raw
    };

    let result = tokio::task::spawn_blocking(move || {
        let batch = storage.list_sessions_batch(
            offset,
            limit,
            group_id.as_deref(),
            marker_filter.as_deref(),
            search_query.as_deref(),
            tags_any.as_deref(),
            project_id.as_deref(),
            project_id_null,
        )?;

        // Batch-resolve tags: one scan for all sessions on this page.
        let session_ids: Vec<&str> = batch
            .sessions
            .iter()
            .map(|s| s.session_id.as_str())
            .collect();
        let tags_by_session = storage
            .get_sessions_tags_batch(&session_ids)
            .unwrap_or_default();

        // Pre-fetch all tag records for detail lookup (name, color).
        let tag_details: std::collections::HashMap<String, serde_json::Value> = storage
            .list_tags(None)
            .unwrap_or_default()
            .into_iter()
            .map(|t| {
                let val = json!({ "id": t.tag_id, "name": t.name, "color": t.color });
                (t.tag_id, val)
            })
            .collect();

        let page_data: Vec<serde_json::Value> = batch
            .sessions
            .iter()
            .map(|session| {
                let tags: Vec<serde_json::Value> = tags_by_session
                    .get(&session.session_id)
                    .map(|tag_ids| {
                        tag_ids
                            .iter()
                            .filter_map(|tid| tag_details.get(tid).cloned())
                            .collect()
                    })
                    .unwrap_or_default();
                session_to_session_json(session, Some(tags))
            })
            .collect();

        let has_more = batch.has_more;
        let total = batch.total;

        // Compute cursor from last session in page for backward compatibility.
        let next_cursor = if has_more {
            batch.sessions.last().map(|s| SessionCursor {
                updated_at_ms: s.updated_at,
                session_id: s.session_id.clone(),
            })
        } else {
            None
        };

        Ok((page_data, has_more, next_cursor, total))
    })
    .await;

    let (data, has_more, next_cursor, total) = match result {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return storage_error_response(e),
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                &format!("{e}"),
            )
        }
    };

    let mut payload = json!({
        "object": "list",
        "data": data,
        "has_more": has_more,
        "total": total,
    });

    if let Some(ref cursor) = next_cursor {
        payload["cursor"] = json!(encode_cursor(cursor));
    } else {
        payload["cursor"] = serde_json::Value::Null;
    }

    json_response(StatusCode::OK, &payload)
}

#[utoipa::path(get, path = "/v1/chat/sessions/{session_id}", tag = "Chat",
    params(("session_id" = String, Path, description = "Session ID")),
    responses(
        (status = 200, body = SessionResponse),
        (status = 404, body = http::ErrorResponse, description = "Session not found"),
    ))]
/// GET /v1/chat/sessions/{id} — retrieve full session.
pub async fn handle_get(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let session_id = match extract_session_id(&path) {
        Some(id) => id.to_string(),
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "Missing session ID",
            )
        }
    };

    let storage = match open_storage(&state, &auth) {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    let id_for_session = session_id.clone();
    let id_for_conv = session_id.clone();
    let id_for_tags = session_id.clone();

    let result: Result<Result<_, StorageError>, _> = tokio::task::spawn_blocking(move || {
        // Get session metadata (full record with source_doc_id)
        let session = storage.get_session_full(&id_for_session)?;

        // Load session items
        let conv = storage.load_session(&id_for_conv)?;
        let items_json: String = conv
            .to_responses_json(1)
            .map_err(|e| StorageError::IoError(format!("serialization failed: {e}")))?;

        // Resolve tags
        let tags = resolve_tags_for_session(&storage, &id_for_tags);

        Ok((session, items_json, tags))
    })
    .await;

    let (session, items_json, tags) = match result {
        Ok(Ok((s, j, t))) => (s, j, t),
        Ok(Err(StorageError::SessionNotFound(_))) => {
            return json_error(StatusCode::NOT_FOUND, "not_found", "Session not found")
        }
        Ok(Err(e)) => return storage_error_response(e),
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                &format!("{e}"),
            )
        }
    };

    let items: serde_json::Value = serde_json::from_str(&items_json).unwrap_or_else(|_| json!([]));

    let mut session_payload = session_to_session_json(&session, Some(tags));
    session_payload["items"] = items;

    json_response(StatusCode::OK, &session_payload)
}

#[utoipa::path(delete, path = "/v1/chat/sessions/{session_id}", tag = "Chat",
    params(("session_id" = String, Path, description = "Session ID")),
    responses((status = 204)))]
/// DELETE /v1/chat/sessions/{id} — delete session (idempotent).
pub async fn handle_delete(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let session_id = match extract_session_id(&path) {
        Some(id) => id.to_string(),
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "Missing session ID",
            )
        }
    };

    let storage = match open_storage(&state, &auth) {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    let result = tokio::task::spawn_blocking(move || storage.delete_session(&session_id)).await;

    match result {
        Ok(Ok(())) | Ok(Err(StorageError::SessionNotFound(_))) => {
            // Idempotent: 204 whether it existed or not
            Response::builder()
                .status(StatusCode::NO_CONTENT)
                .body(Full::new(Bytes::new()).boxed())
                .unwrap()
        }
        Ok(Err(e)) => storage_error_response(e),
        Err(e) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            &format!("{e}"),
        ),
    }
}

#[utoipa::path(patch, path = "/v1/chat/sessions/{session_id}", tag = "Chat",
    params(("session_id" = String, Path, description = "Session ID")),
    request_body = SessionPatchRequest,
    responses((status = 200, body = SessionResponse)))]
/// PATCH /v1/chat/sessions/{id} — update session metadata.
pub async fn handle_patch(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let session_id = match extract_session_id(&path) {
        Some(id) => id.to_string(),
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "Missing session ID",
            )
        }
    };

    let (_parts, body) = req.into_parts();
    let body_bytes = match body.collect().await {
        Ok(b) => b.to_bytes(),
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "invalid_request", "Invalid body"),
    };

    let patch: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &format!("Invalid JSON: {e}"),
            )
        }
    };

    // Extract project_id from metadata or top-level patch body.
    // If metadata.project_id is a string, set it. If explicitly null, clear it.
    let (patch_project_id, clear_project_id) = if let Some(meta) = patch.get("metadata") {
        if let Some(pid_val) = meta.get("project_id") {
            if pid_val.is_null() {
                (None, true)
            } else if let Some(pid_str) = pid_val.as_str() {
                (Some(pid_str.to_string()), false)
            } else {
                (None, false)
            }
        } else {
            (None, false)
        }
    } else if let Some(pid_val) = patch.get("project_id") {
        if pid_val.is_null() {
            (None, true)
        } else if let Some(pid_str) = pid_val.as_str() {
            (Some(pid_str.to_string()), false)
        } else {
            (None, false)
        }
    } else {
        (None, false)
    };

    let updates = SessionUpdate {
        title: patch
            .get("title")
            .and_then(|v| v.as_str())
            .map(String::from),
        marker: patch
            .get("marker")
            .and_then(|v| v.as_str())
            .map(String::from),
        metadata_json: patch.get("metadata").map(|v| v.to_string()),
        project_id: patch_project_id,
        clear_project_id,
        ..Default::default()
    };

    let storage = match open_storage(&state, &auth) {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    let id_for_update = session_id.clone();
    let id_for_get = session_id.clone();
    let id_for_tags = session_id.clone();

    let result = tokio::task::spawn_blocking(move || {
        storage.update_session(&id_for_update, &updates)?;
        // Return updated session with tags
        let session = storage.get_session(&id_for_get)?;
        let tags = resolve_tags_for_session(&storage, &id_for_tags);
        Ok((session, tags))
    })
    .await;

    let (session, tags) = match result {
        Ok(Ok((s, t))) => (s, t),
        Ok(Err(StorageError::SessionNotFound(_))) => {
            return json_error(StatusCode::NOT_FOUND, "not_found", "Session not found")
        }
        Ok(Err(e)) => return storage_error_response(e),
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                &format!("{e}"),
            )
        }
    };

    let session_payload = session_slim_to_session_json(&session, Some(tags));
    json_response(StatusCode::OK, &session_payload)
}

#[utoipa::path(post, path = "/v1/chat/sessions/{session_id}/fork", tag = "Chat",
    params(("session_id" = String, Path, description = "Session ID")),
    request_body = ForkRequest,
    responses((status = 200, body = SessionResponse)))]
/// POST /v1/chat/sessions/{id}/fork — fork session at item.
pub async fn handle_fork(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let session_id = match extract_session_id(&path) {
        Some(id) => id.to_string(),
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "Missing session ID",
            )
        }
    };

    let (_parts, body) = req.into_parts();
    let body_bytes = match body.collect().await {
        Ok(b) => b.to_bytes(),
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "invalid_request", "Invalid body"),
    };

    let fork_req: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &format!("Invalid JSON: {e}"),
            )
        }
    };

    let target_item_id = match fork_req.get("target_item_id").and_then(|v| v.as_u64()) {
        Some(id) => id,
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "Missing or invalid target_item_id (must be a positive integer)",
            )
        }
    };

    let new_session_id = format!("sess_{}", uuid::Uuid::new_v4());

    let storage = match open_storage(&state, &auth) {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    let source_id = session_id.clone();
    let new_id = new_session_id.clone();
    let new_id_for_tags = new_session_id.clone();

    let result = tokio::task::spawn_blocking(move || {
        storage.fork_session(&source_id, target_item_id, &new_id)?;
        let session = storage.get_session(&new_id)?;
        let tags = resolve_tags_for_session(&storage, &new_id_for_tags);
        Ok((session, tags))
    })
    .await;

    let (session, tags) = match result {
        Ok(Ok((s, t))) => (s, t),
        Ok(Err(StorageError::SessionNotFound(_))) => {
            return json_error(
                StatusCode::NOT_FOUND,
                "not_found",
                "Source session not found",
            )
        }
        Ok(Err(StorageError::ItemNotFound(_))) => {
            return json_error(
                StatusCode::NOT_FOUND,
                "not_found",
                "Target item not found in source session",
            )
        }
        Ok(Err(e)) => return storage_error_response(e),
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                &format!("{e}"),
            )
        }
    };

    let session_payload = session_slim_to_session_json(&session, Some(tags));
    json_response(StatusCode::CREATED, &session_payload)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn storage_error_response(err: StorageError) -> Response<BoxBody> {
    let (status, code) = match &err {
        StorageError::SessionNotFound(_) => (StatusCode::NOT_FOUND, "not_found"),
        StorageError::ItemNotFound(_) => (StatusCode::NOT_FOUND, "not_found"),
        StorageError::StorageNotFound(_) => {
            (StatusCode::SERVICE_UNAVAILABLE, "storage_unavailable")
        }
        StorageError::InvalidArgument(_) => (StatusCode::BAD_REQUEST, "invalid_request"),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, "storage_error"),
    };
    json_error(status, code, &format!("{err}"))
}

fn json_response(status: StatusCode, value: &serde_json::Value) -> Response<BoxBody> {
    let body = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}

fn json_error(status: StatusCode, code: &str, message: &str) -> Response<BoxBody> {
    let payload = json!({
        "error": {
            "code": code,
            "message": message
        }
    });
    let body = serde_json::to_vec(&payload).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}

// ---------------------------------------------------------------------------
// Session Tag Handlers
// ---------------------------------------------------------------------------

#[utoipa::path(get, path = "/v1/chat/sessions/{session_id}/tags", tag = "Chat",
    params(("session_id" = String, Path, description = "Session ID")),
    responses((status = 200)))]
/// GET /v1/chat/sessions/:id/tags - Get tags for a session
pub async fn handle_get_tags(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let session_id = match extract_session_id(req.uri().path()) {
        Some(id) => id,
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "Missing session ID",
            )
        }
    };

    let storage = match open_storage(&state, &auth) {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    let tag_ids = match storage.get_session_tags(session_id) {
        Ok(ids) => ids,
        Err(e) => return storage_error_response(e),
    };

    // Resolve tag IDs to full tag objects
    let mut tags = Vec::with_capacity(tag_ids.len());
    for tag_id in &tag_ids {
        if let Ok(tag) = storage.get_tag(tag_id) {
            tags.push(json!({
                "id": tag.tag_id,
                "name": tag.name,
                "color": tag.color,
            }));
        }
    }

    json_response(StatusCode::OK, &json!({ "tags": tags }))
}

#[utoipa::path(post, path = "/v1/chat/sessions/{session_id}/tags", tag = "Chat",
    params(("session_id" = String, Path, description = "Session ID")),
    request_body = TagsRequest,
    responses((status = 200)))]
/// POST /v1/chat/sessions/:id/tags - Add tags to a session
pub async fn handle_add_tags(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let session_id = match extract_session_id(&path) {
        Some(id) => id.to_string(),
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "Missing session ID",
            )
        }
    };

    let body = match read_body(req).await {
        Ok(b) => b,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e),
    };

    #[derive(serde::Deserialize)]
    struct AddTagsRequest {
        tags: Vec<String>,
    }

    let add_req: AddTagsRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    let storage = match open_storage(&state, &auth) {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    let group_id = auth.as_ref().and_then(|a| a.group_id.clone());

    // Add each tag (accepting either tag_id or tag name)
    for tag_ref in &add_req.tags {
        // First try to get tag by ID
        let tag_id = if storage.get_tag(tag_ref).is_ok() {
            tag_ref.clone()
        } else {
            // Try by name, auto-create if doesn't exist
            match storage.get_tag_by_name(tag_ref, group_id.as_deref()) {
                Ok(tag) => tag.tag_id,
                Err(StorageError::TagNotFound(_)) => {
                    // Auto-create the tag
                    let new_id = uuid::Uuid::new_v4().to_string();
                    let create = talu::storage::TagCreate {
                        tag_id: new_id.clone(),
                        name: tag_ref.clone(),
                        color: None,
                        description: None,
                        group_id: group_id.clone(),
                    };
                    if let Err(e) = storage.create_tag(&create) {
                        return storage_error_response(e);
                    }
                    new_id
                }
                Err(e) => return storage_error_response(e),
            }
        };

        if let Err(e) = storage.add_session_tag(&session_id, &tag_id) {
            return storage_error_response(e);
        }
    }

    // Return current tags
    let tag_ids = match storage.get_session_tags(&session_id) {
        Ok(ids) => ids,
        Err(e) => return storage_error_response(e),
    };

    let mut tags = Vec::with_capacity(tag_ids.len());
    for tag_id in &tag_ids {
        if let Ok(tag) = storage.get_tag(tag_id) {
            tags.push(json!({
                "id": tag.tag_id,
                "name": tag.name,
            }));
        }
    }

    json_response(StatusCode::OK, &json!({ "tags": tags }))
}

#[utoipa::path(delete, path = "/v1/chat/sessions/{session_id}/tags", tag = "Chat",
    params(("session_id" = String, Path, description = "Session ID")),
    request_body = TagsRequest,
    responses((status = 200)))]
/// DELETE /v1/chat/sessions/:id/tags - Remove tags from a session
pub async fn handle_remove_tags(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let session_id = match extract_session_id(&path) {
        Some(id) => id.to_string(),
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "Missing session ID",
            )
        }
    };

    let body = match read_body(req).await {
        Ok(b) => b,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e),
    };

    #[derive(serde::Deserialize)]
    struct RemoveTagsRequest {
        tags: Vec<String>,
    }

    let remove_req: RemoveTagsRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    let storage = match open_storage(&state, &auth) {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    for tag_id in &remove_req.tags {
        if let Err(e) = storage.remove_session_tag(&session_id, tag_id) {
            // Ignore "not found" errors when removing
            if !matches!(e, StorageError::TagNotFound(_)) {
                return storage_error_response(e);
            }
        }
    }

    // Return remaining tags
    let tag_ids = match storage.get_session_tags(&session_id) {
        Ok(ids) => ids,
        Err(e) => return storage_error_response(e),
    };

    let mut tags = Vec::with_capacity(tag_ids.len());
    for tag_id in &tag_ids {
        if let Ok(tag) = storage.get_tag(tag_id) {
            tags.push(json!({
                "id": tag.tag_id,
                "name": tag.name,
            }));
        }
    }

    json_response(StatusCode::OK, &json!({ "tags": tags }))
}

#[utoipa::path(put, path = "/v1/chat/sessions/{session_id}/tags", tag = "Chat",
    params(("session_id" = String, Path, description = "Session ID")),
    request_body = TagsRequest,
    responses((status = 200)))]
/// PUT /v1/chat/sessions/:id/tags - Replace all tags on a session
pub async fn handle_set_tags(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let session_id = match extract_session_id(&path) {
        Some(id) => id.to_string(),
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "Missing session ID",
            )
        }
    };

    let body = match read_body(req).await {
        Ok(b) => b,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e),
    };

    #[derive(serde::Deserialize)]
    struct SetTagsRequest {
        tags: Vec<String>,
    }

    let set_req: SetTagsRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    let storage = match open_storage(&state, &auth) {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    let group_id = auth.as_ref().and_then(|a| a.group_id.clone());

    // Remove all existing tags
    let existing_tags = match storage.get_session_tags(&session_id) {
        Ok(ids) => ids,
        Err(e) => return storage_error_response(e),
    };

    for tag_id in existing_tags {
        let _ = storage.remove_session_tag(&session_id, &tag_id);
    }

    // Add the new tags
    for tag_ref in &set_req.tags {
        // First try to get tag by ID
        let tag_id = if storage.get_tag(tag_ref).is_ok() {
            tag_ref.clone()
        } else {
            // Try by name, auto-create if doesn't exist
            match storage.get_tag_by_name(tag_ref, group_id.as_deref()) {
                Ok(tag) => tag.tag_id,
                Err(StorageError::TagNotFound(_)) => {
                    // Auto-create the tag
                    let new_id = uuid::Uuid::new_v4().to_string();
                    let create = talu::storage::TagCreate {
                        tag_id: new_id.clone(),
                        name: tag_ref.clone(),
                        color: None,
                        description: None,
                        group_id: group_id.clone(),
                    };
                    if let Err(e) = storage.create_tag(&create) {
                        return storage_error_response(e);
                    }
                    new_id
                }
                Err(e) => return storage_error_response(e),
            }
        };

        if let Err(e) = storage.add_session_tag(&session_id, &tag_id) {
            return storage_error_response(e);
        }
    }

    // Return final tags
    let tag_ids = match storage.get_session_tags(&session_id) {
        Ok(ids) => ids,
        Err(e) => return storage_error_response(e),
    };

    let mut tags = Vec::with_capacity(tag_ids.len());
    for tag_id in &tag_ids {
        if let Ok(tag) = storage.get_tag(tag_id) {
            tags.push(json!({
                "id": tag.tag_id,
                "name": tag.name,
            }));
        }
    }

    json_response(StatusCode::OK, &json!({ "tags": tags }))
}

async fn read_body(req: Request<Incoming>) -> Result<Vec<u8>, String> {
    let body = req.into_body();
    let collected = body
        .collect()
        .await
        .map_err(|e| format!("Failed to read body: {e}"))?;
    Ok(collected.to_bytes().to_vec())
}

// ---------------------------------------------------------------------------
// Batch Operations Handler
// ---------------------------------------------------------------------------

#[utoipa::path(post, path = "/v1/chat/sessions/batch", tag = "Chat",
    request_body = BatchRequest,
    responses((status = 200)))]
/// POST /v1/chat/sessions/batch - Perform batch operations on sessions.
///
/// Supported actions:
/// - "delete": Delete multiple sessions
/// - "archive": Set marker="archived" on multiple sessions
/// - "unarchive": Remove marker from multiple sessions
/// - "add_tags": Add tags to multiple sessions (requires `tags` field)
/// - "remove_tags": Remove tags from multiple sessions (requires `tags` field)
pub async fn handle_batch(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body = match read_body(req).await {
        Ok(b) => b,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e),
    };

    #[derive(serde::Deserialize)]
    struct BatchRequest {
        action: String,
        ids: Vec<String>,
        #[serde(default)]
        tags: Vec<String>,
    }

    let batch_req: BatchRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    // Validate action
    let valid_actions = ["delete", "archive", "unarchive", "add_tags", "remove_tags"];
    if !valid_actions.contains(&batch_req.action.as_str()) {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_action",
            &format!(
                "Invalid action '{}'. Must be one of: {}",
                batch_req.action,
                valid_actions.join(", ")
            ),
        );
    }

    // Validate tags are provided for tag operations
    if (batch_req.action == "add_tags" || batch_req.action == "remove_tags")
        && batch_req.tags.is_empty()
    {
        return json_error(
            StatusCode::BAD_REQUEST,
            "missing_tags",
            &format!("'tags' field is required for '{}' action", batch_req.action),
        );
    }

    // Validate non-empty ids
    if batch_req.ids.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "missing_ids",
            "'ids' field must contain at least one session ID",
        );
    }

    // Limit batch size to prevent abuse
    if batch_req.ids.len() > 100 {
        return json_error(
            StatusCode::BAD_REQUEST,
            "batch_too_large",
            "Batch operations are limited to 100 sessions at a time",
        );
    }

    let storage = match open_storage(&state, &auth) {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    let group_id = auth.as_ref().and_then(|a| a.group_id.clone());
    let action = batch_req.action.clone();
    let ids = batch_req.ids.clone();
    let tags = batch_req.tags.clone();

    let result = tokio::task::spawn_blocking(move || {
        match action.as_str() {
            "delete" => {
                for id in &ids {
                    // Ignore not-found errors for idempotent deletes
                    let _ = storage.delete_session(id);
                }
                Ok(())
            }
            "archive" => {
                for id in &ids {
                    let update = SessionUpdate {
                        marker: Some("archived".to_string()),
                        ..Default::default()
                    };
                    storage.update_session(id, &update)?;
                }
                Ok(())
            }
            "unarchive" => {
                for id in &ids {
                    let update = SessionUpdate {
                        marker: Some(String::new()), // Empty string clears marker
                        ..Default::default()
                    };
                    storage.update_session(id, &update)?;
                }
                Ok(())
            }
            "add_tags" => {
                // Resolve tags (by ID or name, auto-create if needed)
                let mut tag_ids = Vec::new();
                for tag_ref in &tags {
                    let tag_id = if storage.get_tag(tag_ref).is_ok() {
                        tag_ref.clone()
                    } else {
                        match storage.get_tag_by_name(tag_ref, group_id.as_deref()) {
                            Ok(tag) => tag.tag_id,
                            Err(StorageError::TagNotFound(_)) => {
                                // Auto-create the tag
                                let new_id = uuid::Uuid::new_v4().to_string();
                                let create = talu::storage::TagCreate {
                                    tag_id: new_id.clone(),
                                    name: tag_ref.clone(),
                                    color: None,
                                    description: None,
                                    group_id: group_id.clone(),
                                };
                                storage.create_tag(&create)?;
                                new_id
                            }
                            Err(e) => return Err(e),
                        }
                    };
                    tag_ids.push(tag_id);
                }

                // Add tags to all sessions
                for id in &ids {
                    for tag_id in &tag_ids {
                        let _ = storage.add_session_tag(id, tag_id);
                    }
                }
                Ok(())
            }
            "remove_tags" => {
                for id in &ids {
                    for tag_ref in &tags {
                        // Try as ID first, then as name
                        if storage.get_tag(tag_ref).is_ok() {
                            let _ = storage.remove_session_tag(id, tag_ref);
                        } else if let Ok(tag) =
                            storage.get_tag_by_name(tag_ref, group_id.as_deref())
                        {
                            let _ = storage.remove_session_tag(id, &tag.tag_id);
                        }
                    }
                }
                Ok(())
            }
            _ => unreachable!(),
        }
    })
    .await;

    match result {
        Ok(Ok(())) => Response::builder()
            .status(StatusCode::NO_CONTENT)
            .body(Full::new(Bytes::new()).boxed())
            .unwrap(),
        Ok(Err(e)) => storage_error_response(e),
        Err(e) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            &format!("{e}"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_session_id_v1_prefix() {
        assert_eq!(
            extract_session_id("/v1/chat/sessions/abc123"),
            Some("abc123")
        );
    }

    #[test]
    fn test_extract_session_id_without_v1_prefix() {
        assert_eq!(extract_session_id("/sessions/abc123"), None);
    }

    #[test]
    fn test_extract_session_id_with_fork_suffix() {
        assert_eq!(
            extract_session_id("/v1/chat/sessions/abc123/fork"),
            Some("abc123")
        );
    }

    #[test]
    fn test_extract_session_id_empty() {
        assert_eq!(extract_session_id("/v1/chat/sessions/"), None);
    }

    #[test]
    fn test_extract_session_id_no_match() {
        assert_eq!(extract_session_id("/v1/responses/abc"), None);
    }

    #[test]
    fn test_cursor_round_trip() {
        let cursor = SessionCursor {
            updated_at_ms: 1700000000000,
            session_id: "sess_abc123".to_string(),
        };
        let encoded = encode_cursor(&cursor);
        let decoded = decode_cursor(&encoded).unwrap();
        assert_eq!(decoded.updated_at_ms, 1700000000000);
        assert_eq!(decoded.session_id, "sess_abc123");
    }

    #[test]
    fn test_decode_cursor_invalid() {
        assert!(decode_cursor("not-valid-base64!!!").is_none());
    }

    #[test]
    fn test_base64_round_trip() {
        let input = b"1700000000000:sess_abc123";
        let encoded = base64_encode(input);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_base64_empty() {
        let encoded = base64_encode(b"");
        let decoded = base64_decode(&encoded).unwrap();
        assert!(decoded.is_empty());
    }
}
