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

use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;

const DEFAULT_MAX_READ_BYTES: usize = 256 * 1024;
const DEFAULT_LIST_LIMIT: usize = 1000;

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

fn open_fs(state: &AppState) -> Result<FsHandle, Response<BoxBody>> {
    FsHandle::open(&state.workspace_dir.to_string_lossy())
        .map_err(|e| fs_error_response(e, "failed to initialize workspace fs"))
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

fn to_workspace_relative(workspace: &Path, requested: &str) -> String {
    let requested_path = Path::new(requested);
    let absolute: PathBuf = if requested_path.is_absolute() {
        requested_path.to_path_buf()
    } else {
        workspace.join(requested_path)
    };

    match absolute.strip_prefix(workspace) {
        Ok(rel) if rel.as_os_str().is_empty() => ".".to_string(),
        Ok(rel) => rel.to_string_lossy().to_string(),
        Err(_) => requested.to_string(),
    }
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
    let fs = match open_fs(&state) {
        Ok(fs) => fs,
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
            path: to_workspace_relative(&state.workspace_dir, &request.path),
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
    _auth: Option<AuthContext>,
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

    let fs = match open_fs(&state) {
        Ok(fs) => fs,
        Err(resp) => return resp,
    };
    let result = match fs.write(&request.path, &content, request.mkdir.unwrap_or(false)) {
        Ok(v) => v,
        Err(e) => return fs_error_response(e, "failed to write file"),
    };

    json_response(
        StatusCode::OK,
        &FsWriteResponse {
            path: to_workspace_relative(&state.workspace_dir, &request.path),
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
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e.to_string()),
    };
    let request: FsEditRequest = match parse_json(&body) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let fs = match open_fs(&state) {
        Ok(fs) => fs,
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

    json_response(
        StatusCode::OK,
        &FsEditResponse {
            path: to_workspace_relative(&state.workspace_dir, &request.path),
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

    let fs = match open_fs(&state) {
        Ok(fs) => fs,
        Err(resp) => return resp,
    };
    let stat = match fs.stat(&request.path) {
        Ok(v) => v,
        Err(e) => return fs_error_response(e, "failed to stat path"),
    };

    json_response(
        StatusCode::OK,
        &FsStatResponse {
            path: to_workspace_relative(&state.workspace_dir, &request.path),
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

    let fs = match open_fs(&state) {
        Ok(fs) => fs,
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
            path: to_workspace_relative(&state.workspace_dir, &request.path),
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
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e.to_string()),
    };
    let request: FsRemoveRequest = match parse_json(&body) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let fs = match open_fs(&state) {
        Ok(fs) => fs,
        Err(resp) => return resp,
    };
    match fs.remove(&request.path, request.recursive.unwrap_or(false)) {
        Ok(()) => json_response(
            StatusCode::OK,
            &FsRemoveResponse {
                path: to_workspace_relative(&state.workspace_dir, &request.path),
                removed: true,
            },
        ),
        Err(e) => fs_error_response(e, "failed to remove path"),
    }
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

    let fs = match open_fs(&state) {
        Ok(fs) => fs,
        Err(resp) => return resp,
    };
    match fs.mkdir(&request.path, request.recursive.unwrap_or(false)) {
        Ok(()) => json_response(
            StatusCode::OK,
            &FsMkdirResponse {
                path: to_workspace_relative(&state.workspace_dir, &request.path),
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
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e.to_string()),
    };
    let request: FsRenameRequest = match parse_json(&body) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let fs = match open_fs(&state) {
        Ok(fs) => fs,
        Err(resp) => return resp,
    };
    match fs.rename(&request.from, &request.to) {
        Ok(()) => json_response(
            StatusCode::OK,
            &FsRenameResponse {
                from: to_workspace_relative(&state.workspace_dir, &request.from),
                to: to_workspace_relative(&state.workspace_dir, &request.to),
            },
        ),
        Err(e) => fs_error_response(e, "failed to rename path"),
    }
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
