//! `/v1/agent/fs/*` filesystem capability endpoints.
//!
//! HTTP handlers are glue only: parse request JSON, call `talu::FsHandle`,
//! and shape responses.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::Engine;
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde::{Deserialize, Serialize};
use talu::{FsError, FsHandle};
use utoipa::ToSchema;

use crate::server::collab::{open_collab_handle, resolve_storage_root};
use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;

const DEFAULT_MAX_READ_BYTES: usize = 256 * 1024;
const DEFAULT_LIST_LIMIT: usize = 1000;
const COLLAB_WORKDIR_RESOURCE_KIND: &str = "workdir_file";
const COLLAB_SYNC_ACTOR_ID: &str = "system:agent_fs";

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct FsReadRequest {
    path: String,
    #[serde(default)]
    encoding: Option<String>,
    #[serde(default)]
    max_bytes: Option<usize>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct FsReadResponse {
    path: String,
    content: String,
    encoding: String,
    size: u64,
    /// Reserved for future partial-read mode; current core behavior returns
    /// 413 `too_large` instead of a truncated payload.
    truncated: bool,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct FsWriteRequest {
    path: String,
    content: String,
    #[serde(default)]
    encoding: Option<String>,
    #[serde(default)]
    mkdir: Option<bool>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct FsWriteResponse {
    path: String,
    bytes_written: usize,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct FsEditRequest {
    path: String,
    old_text: String,
    new_text: String,
    #[serde(default)]
    replace_all: Option<bool>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct FsEditResponse {
    path: String,
    replacements: usize,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct FsStatRequest {
    path: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct FsStatResponse {
    path: String,
    exists: bool,
    is_file: bool,
    is_dir: bool,
    is_symlink: bool,
    size: u64,
    mode: String,
    modified_at: i64,
    created_at: i64,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct FsListRequest {
    path: String,
    #[serde(default)]
    glob: Option<String>,
    #[serde(default)]
    recursive: Option<bool>,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub(crate) struct FsListEntry {
    name: String,
    path: String,
    is_dir: bool,
    is_symlink: bool,
    size: u64,
    modified_at: i64,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct FsListResponse {
    path: String,
    entries: Vec<FsListEntry>,
    truncated: bool,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct FsRemoveRequest {
    path: String,
    #[serde(default)]
    recursive: Option<bool>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct FsRemoveResponse {
    path: String,
    removed: bool,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct FsMkdirRequest {
    path: String,
    #[serde(default)]
    recursive: Option<bool>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct FsMkdirResponse {
    path: String,
    created: bool,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct FsRenameRequest {
    from: String,
    to: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct FsRenameResponse {
    from: String,
    to: String,
}

#[derive(Debug, Deserialize)]
struct CoreListPayload {
    entries: Vec<FsListEntry>,
    truncated: bool,
}

fn parse_json<T: for<'de> Deserialize<'de>>(body: &[u8]) -> Result<T, Response<BoxBody>> {
    serde_json::from_slice(body)
        .map_err(|e| json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()))
}

fn missing_workdir_response() -> Response<BoxBody> {
    json_error(
        StatusCode::BAD_REQUEST,
        "no_workdir",
        "no workdir was passed",
    )
}

fn open_fs<'a>(
    state: &'a AppState,
    policy: Option<&talu::policy::Policy>,
) -> Result<(FsHandle, &'a Path), Response<BoxBody>> {
    let workdir = state
        .workdir
        .as_deref()
        .ok_or_else(missing_workdir_response)?;
    FsHandle::open_with_policy(&workdir.to_string_lossy(), policy)
        .map(|handle| (handle, workdir))
        .map_err(|e| fs_error_response(e, "failed to initialize workdir fs"))
}

fn load_policy(state: &AppState) -> Option<Arc<talu::policy::Policy>> {
    super::load_runtime_policy(state)
}

fn encoding_or_default(encoding: Option<String>) -> String {
    encoding.unwrap_or_else(|| "utf-8".to_string())
}

fn decode_content(content: &str, encoding: &str) -> Result<Vec<u8>, String> {
    match encoding {
        "utf-8" => Ok(content.as_bytes().to_vec()),
        "base64" => base64::engine::general_purpose::STANDARD
            .decode(content)
            .map_err(|e| format!("invalid base64 content: {e}")),
        _ => Err("encoding must be 'utf-8' or 'base64'".to_string()),
    }
}

fn encode_content(content: &[u8], encoding: &str) -> Result<String, String> {
    match encoding {
        "utf-8" => String::from_utf8(content.to_vec())
            .map_err(|e| format!("file content is not valid utf-8: {e}")),
        "base64" => Ok(base64::engine::general_purpose::STANDARD.encode(content)),
        _ => Err("encoding must be 'utf-8' or 'base64'".to_string()),
    }
}

fn normalized_absolute_for_prefix(path: &Path) -> PathBuf {
    if let Ok(canon) = std::fs::canonicalize(path) {
        return canon;
    }
    if let (Some(parent), Some(name)) = (path.parent(), path.file_name()) {
        if let Ok(canon_parent) = std::fs::canonicalize(parent) {
            return canon_parent.join(name);
        }
    }
    path.to_path_buf()
}

fn to_workspace_relative(workspace: &Path, requested: &str) -> String {
    let requested_path = Path::new(requested);
    let absolute: PathBuf = if requested_path.is_absolute() {
        requested_path.to_path_buf()
    } else {
        workspace.join(requested_path)
    };
    let workspace_abs = normalized_absolute_for_prefix(workspace);
    let absolute_norm = normalized_absolute_for_prefix(&absolute);

    match absolute_norm
        .strip_prefix(&workspace_abs)
        .or_else(|_| absolute.strip_prefix(workspace))
    {
        Ok(rel) if rel.as_os_str().is_empty() => ".".to_string(),
        Ok(rel) => rel.to_string_lossy().to_string(),
        Err(_) => requested.to_string(),
    }
}

fn collab_resource_id_for_workdir_file(workdir: &Path, requested: &str) -> String {
    to_workspace_relative(workdir, requested)
}

fn full_read_limit_for_current_file(fs: &FsHandle, path: &str) -> Result<usize, FsError> {
    let stat = fs.stat(path)?;
    let size = usize::try_from(stat.size)
        .map_err(|_| FsError::TooLarge("file is too large to mirror into collab".to_string()))?;
    Ok(size.saturating_add(1).max(1))
}

async fn sync_workdir_collab_snapshot(
    state: &Arc<AppState>,
    auth: Option<&AuthContext>,
    resource_id: &str,
    snapshot: &[u8],
    op_kind: &str,
) {
    if state.bucket_path.is_none() {
        return;
    }

    let Ok(root) = resolve_storage_root(state, auth) else {
        return;
    };
    let Ok(collab) =
        open_collab_handle(state, &root, COLLAB_WORKDIR_RESOURCE_KIND, resource_id).await
    else {
        return;
    };

    let issued_at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0);
    let actor_seq = if issued_at_ms <= 0 {
        1
    } else {
        u64::try_from(issued_at_ms).unwrap_or(u64::MAX)
    };
    let payload = serde_json::to_vec(&serde_json::json!({
        "type": op_kind,
        "source": "agent.fs",
        "resource_kind": COLLAB_WORKDIR_RESOURCE_KIND,
        "resource_id": resource_id,
        "snapshot_bytes": snapshot.len(),
        "issued_at_ms": issued_at_ms,
    }))
    .unwrap_or_else(|_| b"{}".to_vec());
    let op_id = format!("{op_kind}-{actor_seq}");
    let _ = collab.submit_op(
        COLLAB_SYNC_ACTOR_ID,
        actor_seq,
        &op_id,
        &payload,
        Some(issued_at_ms),
        Some(snapshot),
    );
}

async fn clear_workdir_collab_snapshot(
    state: &Arc<AppState>,
    auth: Option<&AuthContext>,
    resource_id: &str,
    op_kind: &str,
) {
    if state.bucket_path.is_none() {
        return;
    }

    let Ok(root) = resolve_storage_root(state, auth) else {
        return;
    };
    let Ok(collab) =
        open_collab_handle(state, &root, COLLAB_WORKDIR_RESOURCE_KIND, resource_id).await
    else {
        return;
    };
    let _ = collab.clear_snapshot(
        COLLAB_SYNC_ACTOR_ID,
        talu::collab::ParticipantKind::System,
        Some("sync"),
        op_kind,
    );
}

fn absolute_workdir_path(workdir: &Path, requested: &str) -> PathBuf {
    let requested_path = Path::new(requested);
    if requested_path.is_absolute() {
        requested_path.to_path_buf()
    } else {
        workdir.join(requested_path)
    }
}

fn collect_workdir_file_resource_ids(workdir: &Path, requested: &str) -> std::io::Result<Vec<String>> {
    let mut resource_ids = Vec::new();
    collect_workdir_file_resource_ids_from_absolute(
        workdir,
        &absolute_workdir_path(workdir, requested),
        &mut resource_ids,
    )?;
    Ok(resource_ids)
}

fn collect_workdir_file_resource_ids_from_absolute(
    workdir: &Path,
    absolute: &Path,
    resource_ids: &mut Vec<String>,
) -> std::io::Result<()> {
    let metadata = std::fs::symlink_metadata(absolute)?;
    if metadata.is_dir() {
        for entry in std::fs::read_dir(absolute)? {
            let entry = entry?;
            collect_workdir_file_resource_ids_from_absolute(workdir, &entry.path(), resource_ids)?;
        }
        return Ok(());
    }

    resource_ids.push(to_workspace_relative(workdir, &absolute.to_string_lossy()));
    Ok(())
}

fn map_renamed_resource_id(from: &str, to: &str, existing: &str) -> Option<String> {
    let suffix = Path::new(existing).strip_prefix(Path::new(from)).ok()?;
    let mapped = if suffix.as_os_str().is_empty() {
        PathBuf::from(to)
    } else {
        Path::new(to).join(suffix)
    };
    Some(mapped.to_string_lossy().to_string())
}

fn fs_error_response(err: FsError, fallback: &str) -> Response<BoxBody> {
    let message = err.to_string();
    match err {
        FsError::InvalidArgument(_) => {
            json_error(StatusCode::BAD_REQUEST, "invalid_request", &message)
        }
        FsError::InvalidPath(_) => json_error(StatusCode::BAD_REQUEST, "invalid_path", &message),
        FsError::PermissionDenied(_) => {
            json_error(StatusCode::FORBIDDEN, "permission_denied", &message)
        }
        FsError::NotFound(_) | FsError::ParentNotFound(_) => {
            json_error(StatusCode::NOT_FOUND, "not_found", &message)
        }
        FsError::IsDirectory(_) => json_error(StatusCode::BAD_REQUEST, "is_directory", &message),
        FsError::NotDirectory(_) => json_error(StatusCode::BAD_REQUEST, "not_directory", &message),
        FsError::NotEmpty(_) => json_error(StatusCode::CONFLICT, "not_empty", &message),
        FsError::TooLarge(_) => json_error(StatusCode::PAYLOAD_TOO_LARGE, "too_large", &message),
        FsError::PolicyDeniedFileRead(_) => {
            json_error(StatusCode::FORBIDDEN, "policy_denied_file_read", &message)
        }
        FsError::PolicyDeniedFileWrite(_) => {
            json_error(StatusCode::FORBIDDEN, "policy_denied_file_write", &message)
        }
        FsError::PolicyDeniedFileDelete(_) => {
            json_error(StatusCode::FORBIDDEN, "policy_denied_file_delete", &message)
        }
        FsError::PolicyInvalid(_) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "policy_invalid",
            &message,
        ),
        FsError::Io(_) => json_error(StatusCode::INTERNAL_SERVER_ERROR, "io_error", fallback),
    }
}

#[utoipa::path(post, path = "/v1/agent/fs/read", tag = "Agent::FS",
    request_body = FsReadRequest,
    responses(
        (status = 200, body = FsReadResponse),
        (status = 400, body = crate::server::http::ErrorResponse),
        (status = 403, body = crate::server::http::ErrorResponse),
        (status = 404, body = crate::server::http::ErrorResponse),
        (status = 413, body = crate::server::http::ErrorResponse),
    ))]
pub async fn handle_read(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e.to_string()),
    };
    let request: FsReadRequest = match parse_json(&body) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let encoding = encoding_or_default(request.encoding);
    let max_bytes = request.max_bytes.unwrap_or(DEFAULT_MAX_READ_BYTES);
    let policy = load_policy(&state);
    let (fs, workdir) = match open_fs(&state, policy.as_deref()) {
        Ok(value) => value,
        Err(resp) => return resp,
    };
    let read = match fs.read(&request.path, max_bytes) {
        Ok(v) => v,
        Err(e) => return fs_error_response(e, "failed to read file"),
    };

    let content = match encode_content(&read.content, &encoding) {
        Ok(v) => v,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_encoding", &e),
    };

    json_response(
        StatusCode::OK,
        &FsReadResponse {
            path: to_workspace_relative(workdir, &request.path),
            content,
            encoding,
            size: read.size,
            truncated: read.truncated,
        },
    )
}

#[utoipa::path(post, path = "/v1/agent/fs/write", tag = "Agent::FS",
    request_body = FsWriteRequest,
    responses((status = 200, body = FsWriteResponse)))]
pub async fn handle_write(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e.to_string()),
    };
    let request: FsWriteRequest = match parse_json(&body) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let encoding = encoding_or_default(request.encoding);
    let content = match decode_content(&request.content, &encoding) {
        Ok(v) => v,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_encoding", &e),
    };

    let policy = load_policy(&state);
    let (result, response_path, resource_id) = {
        let (fs, workdir) = match open_fs(&state, policy.as_deref()) {
            Ok(value) => value,
            Err(resp) => return resp,
        };
        let result = match fs.write(&request.path, &content, request.mkdir.unwrap_or(false)) {
            Ok(v) => v,
            Err(e) => return fs_error_response(e, "failed to write file"),
        };
        let response_path = to_workspace_relative(workdir, &request.path);
        let resource_id = collab_resource_id_for_workdir_file(workdir, &request.path);
        (result, response_path, resource_id)
    };
    sync_workdir_collab_snapshot(
        &state,
        auth.as_ref(),
        &resource_id,
        &content,
        "fs_write",
    )
    .await;

    json_response(
        StatusCode::OK,
        &FsWriteResponse {
            path: response_path,
            bytes_written: result.bytes_written,
        },
    )
}

#[utoipa::path(post, path = "/v1/agent/fs/edit", tag = "Agent::FS",
    request_body = FsEditRequest,
    responses((status = 200, body = FsEditResponse)))]
pub async fn handle_edit(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e.to_string()),
    };
    let request: FsEditRequest = match parse_json(&body) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let policy = load_policy(&state);
    let (result, response_path, resource_id, snapshot_bytes) = {
        let (fs, workdir) = match open_fs(&state, policy.as_deref()) {
            Ok(value) => value,
            Err(resp) => return resp,
        };
        let result = match fs.edit(
            &request.path,
            request.old_text.as_bytes(),
            request.new_text.as_bytes(),
            request.replace_all.unwrap_or(false),
        ) {
            Ok(v) => v,
            Err(e) => return fs_error_response(e, "failed to edit file"),
        };
        let snapshot_bytes =
            if let Ok(read_limit) = full_read_limit_for_current_file(&fs, &request.path) {
                fs.read(&request.path, read_limit)
                    .ok()
                    .map(|read_back| read_back.content)
            } else {
                None
            };
        let response_path = to_workspace_relative(workdir, &request.path);
        let resource_id = collab_resource_id_for_workdir_file(workdir, &request.path);
        (result, response_path, resource_id, snapshot_bytes)
    };
    if let Some(snapshot_bytes) = snapshot_bytes.as_deref() {
        sync_workdir_collab_snapshot(
            &state,
            auth.as_ref(),
            &resource_id,
            snapshot_bytes,
            "fs_edit",
        )
        .await;
    }

    json_response(
        StatusCode::OK,
        &FsEditResponse {
            path: response_path,
            replacements: result.replacements,
        },
    )
}

#[utoipa::path(post, path = "/v1/agent/fs/stat", tag = "Agent::FS",
    request_body = FsStatRequest,
    responses((status = 200, body = FsStatResponse)))]
pub async fn handle_stat(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e.to_string()),
    };
    let request: FsStatRequest = match parse_json(&body) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let policy = load_policy(&state);
    let (fs, workdir) = match open_fs(&state, policy.as_deref()) {
        Ok(value) => value,
        Err(resp) => return resp,
    };
    let stat = match fs.stat(&request.path) {
        Ok(v) => v,
        Err(e) => return fs_error_response(e, "failed to stat path"),
    };

    json_response(
        StatusCode::OK,
        &FsStatResponse {
            path: to_workspace_relative(workdir, &request.path),
            exists: stat.exists,
            is_file: stat.is_file,
            is_dir: stat.is_dir,
            is_symlink: stat.is_symlink,
            size: stat.size,
            mode: format!("{:04o}", stat.mode & 0o7777),
            modified_at: stat.modified_at,
            created_at: stat.created_at,
        },
    )
}

#[utoipa::path(post, path = "/v1/agent/fs/ls", tag = "Agent::FS",
    request_body = FsListRequest,
    responses((status = 200, body = FsListResponse)))]
pub async fn handle_list(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e.to_string()),
    };
    let request: FsListRequest = match parse_json(&body) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let policy = load_policy(&state);
    let (fs, workdir) = match open_fs(&state, policy.as_deref()) {
        Ok(value) => value,
        Err(resp) => return resp,
    };
    let raw_json = match fs.list_json(
        &request.path,
        request.glob.as_deref(),
        request.recursive.unwrap_or(false),
        request.limit.unwrap_or(DEFAULT_LIST_LIMIT),
    ) {
        Ok(v) => v,
        Err(e) => return fs_error_response(e, "failed to list path"),
    };
    let payload: CoreListPayload = match serde_json::from_str(&raw_json) {
        Ok(v) => v,
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "io_error",
                &format!("invalid list payload from core: {e}"),
            )
        }
    };

    json_response(
        StatusCode::OK,
        &FsListResponse {
            path: to_workspace_relative(workdir, &request.path),
            entries: payload.entries,
            truncated: payload.truncated,
        },
    )
}

#[utoipa::path(delete, path = "/v1/agent/fs/rm", tag = "Agent::FS",
    request_body = FsRemoveRequest,
    responses((status = 200, body = FsRemoveResponse)))]
pub async fn handle_remove(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e.to_string()),
    };
    let request: FsRemoveRequest = match parse_json(&body) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let policy = load_policy(&state);
    let (response_path, removed_resource_ids) = {
        let (fs, workdir) = match open_fs(&state, policy.as_deref()) {
            Ok(value) => value,
            Err(resp) => return resp,
        };
        let removed_resource_ids = collect_workdir_file_resource_ids(workdir, &request.path).ok();
        let response_path = to_workspace_relative(workdir, &request.path);
        match fs.remove(&request.path, request.recursive.unwrap_or(false)) {
            Ok(()) => (response_path, removed_resource_ids.unwrap_or_default()),
            Err(e) => return fs_error_response(e, "failed to remove path"),
        }
    };
    for resource_id in &removed_resource_ids {
        clear_workdir_collab_snapshot(&state, auth.as_ref(), resource_id, "fs_delete").await;
    }
    json_response(
        StatusCode::OK,
        &FsRemoveResponse {
            path: response_path,
            removed: true,
        },
    )
}

#[utoipa::path(post, path = "/v1/agent/fs/mkdir", tag = "Agent::FS",
    request_body = FsMkdirRequest,
    responses((status = 200, body = FsMkdirResponse)))]
pub async fn handle_mkdir(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e.to_string()),
    };
    let request: FsMkdirRequest = match parse_json(&body) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let policy = load_policy(&state);
    let (fs, workdir) = match open_fs(&state, policy.as_deref()) {
        Ok(value) => value,
        Err(resp) => return resp,
    };
    match fs.mkdir(&request.path, request.recursive.unwrap_or(false)) {
        Ok(()) => json_response(
            StatusCode::OK,
            &FsMkdirResponse {
                path: to_workspace_relative(workdir, &request.path),
                created: true,
            },
        ),
        Err(e) => fs_error_response(e, "failed to create directory"),
    }
}

#[utoipa::path(post, path = "/v1/agent/fs/rename", tag = "Agent::FS",
    request_body = FsRenameRequest,
    responses((status = 200, body = FsRenameResponse)))]
pub async fn handle_rename(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e.to_string()),
    };
    let request: FsRenameRequest = match parse_json(&body) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let policy = load_policy(&state);
    let (from_path, to_path, removed_resource_ids, renamed_targets) = {
        let (fs, workdir) = match open_fs(&state, policy.as_deref()) {
            Ok(value) => value,
            Err(resp) => return resp,
        };
        let from_path = to_workspace_relative(workdir, &request.from);
        let to_path = to_workspace_relative(workdir, &request.to);
        let source_resource_ids = collect_workdir_file_resource_ids(workdir, &request.from).ok();
        match fs.rename(&request.from, &request.to) {
            Ok(()) => {
                let renamed_targets = source_resource_ids
                    .as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .filter_map(|resource_id| {
                        map_renamed_resource_id(&from_path, &to_path, resource_id).map(|mapped| {
                            let bytes = full_read_limit_for_current_file(&fs, &mapped)
                                .ok()
                                .and_then(|read_limit| fs.read(&mapped, read_limit).ok())
                                .map(|read| read.content);
                            (mapped, bytes)
                        })
                    })
                    .collect::<Vec<_>>();
                (from_path, to_path, source_resource_ids.unwrap_or_default(), renamed_targets)
            }
            Err(e) => return fs_error_response(e, "failed to rename path"),
        }
    };
    for resource_id in &removed_resource_ids {
        clear_workdir_collab_snapshot(&state, auth.as_ref(), resource_id, "fs_rename").await;
    }
    for (resource_id, snapshot_bytes) in &renamed_targets {
        if let Some(snapshot_bytes) = snapshot_bytes.as_deref() {
            sync_workdir_collab_snapshot(
                &state,
                auth.as_ref(),
                resource_id,
                snapshot_bytes,
                "fs_rename",
            )
            .await;
        }
    }
    json_response(
        StatusCode::OK,
        &FsRenameResponse {
            from: from_path,
            to: to_path,
        },
    )
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
        &serde_json::json!({
            "error": {
                "code": code,
                "message": message,
            }
        }),
    )
}
