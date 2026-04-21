//! Repository management endpoints (`/v1/repo/...`).
//!
//! Exposes cache listing, hub search, file listing, and model downloads over
//! HTTP so API clients can manage local model availability.

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::{Request, Response, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use utoipa::ToSchema;

use crate::quant_scheme as quant_scheme_display;
use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

/// A cached model entry returned by the list endpoint.
#[derive(Serialize, ToSchema)]
pub(crate) struct CachedModelResponse {
    /// Model ID (e.g., "meta-llama/Llama-3.2-1B").
    pub id: String,
    /// Local filesystem path.
    pub path: String,
    /// Origin: "hub" or "managed".
    pub source: &'static str,
    /// Size in bytes on disk.
    pub size_bytes: u64,
    /// Last modification time (Unix timestamp, seconds).
    pub mtime: i64,
    /// Model architecture (e.g., "Llama", "Qwen2"), if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    /// Quantization scheme (e.g., "F16", "TQ4", "MXFP4"), if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quant_scheme: Option<String>,
    /// Source model ID this was converted from, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_model_id: Option<String>,
}

/// Response for GET /v1/repo/models.
#[derive(Serialize, ToSchema)]
pub(crate) struct RepoListResponse {
    pub models: Vec<CachedModelResponse>,
    /// Total size of all cached models in bytes.
    pub total_size_bytes: u64,
}

/// A search result from HuggingFace Hub.
#[derive(Serialize, ToSchema)]
pub(crate) struct SearchResultResponse {
    pub model_id: String,
    pub downloads: i64,
    pub likes: i64,
    pub last_modified: String,
    pub params_total: i64,
}

/// Response for GET /v1/repo/search.
#[derive(Serialize, ToSchema)]
pub(crate) struct RepoSearchResponse {
    pub results: Vec<SearchResultResponse>,
}

/// Request body for POST /v1/repo/models.
#[derive(Deserialize, Serialize, ToSchema)]
pub(crate) struct RepoFetchRequest {
    /// Model ID to download (e.g., "meta-llama/Llama-3.2-1B").
    pub model_id: String,
    /// Force re-download even if already cached.
    #[serde(default)]
    pub force: bool,
    /// HuggingFace token for private repos.
    #[serde(default)]
    pub token: Option<String>,
    /// Custom endpoint URL (e.g., a HuggingFace mirror).
    #[serde(default)]
    pub endpoint_url: Option<String>,
    /// Skip downloading weight files (.safetensors), only sync metadata/tokenizer.
    #[serde(default)]
    pub skip_weights: bool,
}

/// Non-streaming response for POST /v1/repo/models.
#[derive(Serialize, ToSchema)]
pub(crate) struct RepoFetchResponse {
    /// The model ID that was downloaded.
    pub model_id: String,
    /// Local filesystem path to the downloaded model.
    pub path: String,
}

/// Response for DELETE /v1/repo/models/{model_id}.
#[derive(Serialize, ToSchema)]
pub(crate) struct RepoDeleteResponse {
    /// Whether the model was successfully deleted.
    pub deleted: bool,
    /// The model ID that was requested for deletion.
    pub model_id: String,
}

/// A single file entry in a model.
#[derive(Serialize, ToSchema)]
pub(crate) struct FileEntry {
    pub filename: String,
}

/// Response for GET /v1/repo/models/{model_id}/files.
#[derive(Serialize, ToSchema)]
pub(crate) struct FileListResponse {
    pub model_id: String,
    pub files: Vec<FileEntry>,
}

/// GET /v1/repo/models — list all locally cached models with metadata.
#[utoipa::path(get, path = "/v1/repo/models", tag = "Repository",
    params(("source" = Option<String>, Query, description = "Filter by source: 'hub' or 'managed'")),
    responses(
        (status = 200, body = RepoListResponse),
        (status = 500, body = super::http::ErrorResponse, description = "Internal error"),
    ))]
pub async fn handle_list(
    _state: Arc<AppState>,
    req: Request<Incoming>,
    _auth_ctx: Option<AuthContext>,
) -> Response<BoxBody> {
    let uri = req.uri().clone();
    let params: Vec<(String, String)> = uri
        .query()
        .map(|q| {
            url::form_urlencoded::parse(q.as_bytes())
                .into_owned()
                .collect()
        })
        .unwrap_or_default();

    let source_filter: Option<String> = params
        .iter()
        .find(|(k, _)| k == "source")
        .map(|(_, v)| v.clone())
        .filter(|v| v == "hub" || v == "managed");

    let result = tokio::task::spawn_blocking(move || {
        let cached = talu::repo::repo_list_models(false).unwrap_or_default();
        let total_size = talu::repo::repo_total_size();

        let models: Vec<CachedModelResponse> = cached
            .into_iter()
            .filter(|m| {
                if let Some(ref filter) = source_filter {
                    let src = match m.source {
                        talu::repo::CacheOrigin::Managed => "managed",
                        talu::repo::CacheOrigin::Hub => "hub",
                    };
                    if src != filter {
                        return false;
                    }
                }
                true
            })
            .map(|m| {
                let size_bytes = talu::repo::repo_size(&m.id);
                let mtime = talu::repo::repo_mtime(&m.id);
                let source = match m.source {
                    talu::repo::CacheOrigin::Managed => "managed",
                    talu::repo::CacheOrigin::Hub => "hub",
                };

                let (architecture, quant_scheme) = match talu::model::describe(&m.path) {
                    Ok(info) => {
                        let arch = if info.model_type.is_empty() {
                            None
                        } else {
                            Some(info.model_type)
                        };
                        let quant = Some(quant_scheme_display::format_quant_scheme_for_path(
                            &m.path,
                            info.quant_method,
                            info.quant_bits,
                            info.quant_group_size,
                        ));
                        (arch, quant)
                    }
                    Err(_) => (None, None),
                };

                let source_model_id = if source == "managed" {
                    std::fs::read_to_string(std::path::Path::new(&m.path).join("talu_meta.json"))
                        .ok()
                        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                        .and_then(|v| v.get("source_model_id")?.as_str().map(String::from))
                } else {
                    None
                };

                CachedModelResponse {
                    id: m.id,
                    path: m.path,
                    source,
                    size_bytes,
                    mtime,
                    architecture,
                    quant_scheme,
                    source_model_id,
                }
            })
            .collect();

        RepoListResponse {
            models,
            total_size_bytes: total_size,
        }
    })
    .await;

    match result {
        Ok(list) => json_response(StatusCode::OK, &list),
        Err(e) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "list_error",
            &format!("Failed to list models: {e}"),
        ),
    }
}

/// GET /v1/repo/search — search HuggingFace Hub for models.
#[utoipa::path(get, path = "/v1/repo/search", tag = "Repository",
    params(
        ("query" = Option<String>, Query, description = "Search query (empty returns trending models)"),
        ("limit" = Option<usize>, Query, description = "Max results (default 20)"),
        ("token" = Option<String>, Query, description = "HuggingFace token for private repos"),
        ("endpoint_url" = Option<String>, Query, description = "Custom endpoint URL"),
        ("sort" = Option<String>, Query, description = "Sort order: trending, downloads, likes, last_modified (default: trending)"),
        ("direction" = Option<String>, Query, description = "Sort direction: descending, ascending (default: descending)"),
        ("filter" = Option<String>, Query, description = "HuggingFace pipeline tag filter (e.g., text-generation)"),
        ("library" = Option<String>, Query, description = "Library/framework filter (e.g., safetensors, transformers)"),
    ),
    responses(
        (status = 200, body = RepoSearchResponse),
        (status = 500, body = super::http::ErrorResponse, description = "Search failed"),
    ))]
pub async fn handle_search(
    _state: Arc<AppState>,
    req: Request<Incoming>,
    _auth_ctx: Option<AuthContext>,
) -> Response<BoxBody> {
    let uri = req.uri().clone();
    let params: Vec<(String, String)> = uri
        .query()
        .map(|q| {
            url::form_urlencoded::parse(q.as_bytes())
                .into_owned()
                .collect()
        })
        .unwrap_or_default();

    let query = params
        .iter()
        .find(|(k, _)| k == "query")
        .map(|(_, v)| v.clone())
        .unwrap_or_default();

    let limit: usize = params
        .iter()
        .find(|(k, _)| k == "limit")
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(20);

    let token: Option<String> = params
        .iter()
        .find(|(k, _)| k == "token")
        .map(|(_, v)| v.clone())
        .filter(|v| !v.is_empty());

    let endpoint_url: Option<String> = params
        .iter()
        .find(|(k, _)| k == "endpoint_url")
        .map(|(_, v)| v.clone())
        .filter(|v| !v.is_empty());

    let sort = match param_str(&params, "sort") {
        Some("downloads") => talu::repo::SearchSort::Downloads,
        Some("likes") => talu::repo::SearchSort::Likes,
        Some("last_modified") => talu::repo::SearchSort::LastModified,
        _ => talu::repo::SearchSort::Trending,
    };

    let direction = match param_str(&params, "direction") {
        Some("ascending") => talu::repo::SearchDirection::Ascending,
        _ => talu::repo::SearchDirection::Descending,
    };

    let filter: Option<String> = params
        .iter()
        .find(|(k, _)| k == "filter")
        .map(|(_, v)| v.clone())
        .filter(|v| !v.is_empty());

    let library: Option<String> = params
        .iter()
        .find(|(k, _)| k == "library")
        .map(|(_, v)| v.clone())
        .filter(|v| !v.is_empty());

    let result = tokio::task::spawn_blocking(move || {
        talu::repo::repo_search_rich(
            &query,
            limit,
            token.as_deref(),
            endpoint_url.as_deref(),
            filter.as_deref(),
            sort,
            direction,
            library.as_deref(),
        )
    })
    .await;

    match result {
        Ok(Ok(hits)) => {
            let results: Vec<SearchResultResponse> = hits
                .into_iter()
                .map(|h| SearchResultResponse {
                    model_id: h.model_id,
                    downloads: h.downloads,
                    likes: h.likes,
                    last_modified: h.last_modified,
                    params_total: h.params_total,
                })
                .collect();
            json_response(StatusCode::OK, &RepoSearchResponse { results })
        }
        Ok(Err(e)) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "search_error",
            &format!("Search failed: {e}"),
        ),
        Err(e) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "search_error",
            &format!("Search task failed: {e}"),
        ),
    }
}

/// POST /v1/repo/models — download a model from HuggingFace Hub.
///
/// When the `Accept` header contains `text/event-stream`, the response streams
/// SSE progress events. Otherwise, the response blocks until the download
/// completes and returns a JSON body.
#[utoipa::path(post, path = "/v1/repo/models", tag = "Repository",
    request_body = RepoFetchRequest,
    responses(
        (status = 200, body = RepoFetchResponse, description = "Download complete (non-streaming)"),
        (status = 400, body = super::http::ErrorResponse, description = "Invalid request"),
        (status = 500, body = super::http::ErrorResponse, description = "Download failed"),
    ))]
pub async fn handle_fetch(
    _state: Arc<AppState>,
    req: Request<Incoming>,
    _auth_ctx: Option<AuthContext>,
) -> Response<BoxBody> {
    let wants_stream = req
        .headers()
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains("text/event-stream"));

    let body_bytes = match req.into_body().collect().await {
        Ok(body) => body.to_bytes(),
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "invalid_request", "Invalid body"),
    };

    let request: RepoFetchRequest = match serde_json::from_slice(&body_bytes) {
        Ok(val) => val,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &format!("Invalid JSON: {e}"),
            );
        }
    };

    if request.model_id.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "model_id is required",
        );
    }

    if wants_stream {
        return handle_fetch_streaming(request);
    }

    let model_id = request.model_id.clone();
    let options = talu::repo::DownloadOptions {
        token: request.token,
        force: request.force,
        endpoint_url: request.endpoint_url,
        skip_weights: request.skip_weights,
        cancel_flag: None,
    };

    let result =
        tokio::task::spawn_blocking(move || talu::repo::repo_fetch(&model_id, options, None)).await;

    match result {
        Ok(Ok(path)) => json_response(
            StatusCode::OK,
            &RepoFetchResponse {
                model_id: request.model_id,
                path,
            },
        ),
        Ok(Err(e)) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "fetch_error",
            &format!("Download failed: {e}"),
        ),
        Err(e) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "fetch_error",
            &format!("Download task failed: {e}"),
        ),
    }
}

/// Streaming variant: sends SSE progress events during download.
fn handle_fetch_streaming(request: RepoFetchRequest) -> Response<BoxBody> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Bytes>();

    let model_id = request.model_id.clone();
    let cancel = Arc::new(AtomicBool::new(false));
    let options = talu::repo::DownloadOptions {
        token: request.token,
        force: request.force,
        endpoint_url: request.endpoint_url,
        skip_weights: request.skip_weights,
        cancel_flag: Some(cancel.clone()),
    };

    std::thread::spawn(move || {
        let tx_progress = tx.clone();
        let cancel_clone = cancel;
        let callback: talu::repo::ProgressCallback =
            Box::new(move |progress: talu::repo::DownloadProgress| {
                let event = json!({
                    "action": match progress.action {
                        talu::repo::ProgressAction::Add => "add",
                        talu::repo::ProgressAction::Update => "update",
                        talu::repo::ProgressAction::Complete => "complete",
                    },
                    "line_id": progress.line_id,
                    "label": progress.label,
                    "message": progress.message,
                    "current": progress.current,
                    "total": progress.total,
                });
                let chunk = format!("data: {}\n\n", event);
                if tx_progress.send(Bytes::from(chunk)).is_err() {
                    cancel_clone.store(true, Ordering::Relaxed);
                }
            });

        let result = talu::repo::repo_fetch(&model_id, options, Some(callback));

        match result {
            Ok(path) => {
                let done = json!({
                    "event": "done",
                    "model_id": model_id,
                    "path": path,
                });
                let chunk = format!("data: {}\n\n", done);
                let _ = tx.send(Bytes::from(chunk));
            }
            Err(e) => {
                let err = json!({
                    "event": "error",
                    "message": format!("{e}"),
                });
                let chunk = format!("data: {}\n\n", err);
                let _ = tx.send(Bytes::from(chunk));
            }
        }
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

/// DELETE /v1/repo/models/{model_id} — evict a model from the local cache.
#[utoipa::path(delete, path = "/v1/repo/models/{model_id}", tag = "Repository",
    params(("model_id" = String, Path, description = "Model ID to delete (URL-encoded)")),
    responses(
        (status = 200, body = RepoDeleteResponse),
        (status = 400, body = super::http::ErrorResponse, description = "Missing model_id"),
    ))]
pub async fn handle_delete(
    _state: Arc<AppState>,
    _req: Request<Incoming>,
    _auth_ctx: Option<AuthContext>,
    model_id: &str,
) -> Response<BoxBody> {
    if model_id.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "model_id is required",
        );
    }

    let model_id_owned = model_id.to_string();
    let result = tokio::task::spawn_blocking(move || talu::repo::repo_delete(&model_id_owned))
        .await
        .unwrap_or(false);

    json_response(
        StatusCode::OK,
        &RepoDeleteResponse {
            deleted: result,
            model_id: model_id.to_string(),
        },
    )
}

/// GET /v1/repo/models/{model_id}/files — list files inside a cached or remote model.
#[utoipa::path(get, path = "/v1/repo/models/{model_id}/files", tag = "Repository",
    params(
        ("model_id" = String, Path, description = "Model ID (URL-encoded)"),
        ("token" = Option<String>, Query, description = "HuggingFace token for private repos"),
    ),
    responses(
        (status = 200, body = FileListResponse),
        (status = 400, body = super::http::ErrorResponse, description = "Missing model_id"),
        (status = 500, body = super::http::ErrorResponse, description = "File listing failed"),
    ))]
pub async fn handle_list_files(
    _state: Arc<AppState>,
    req: Request<Incoming>,
    _auth_ctx: Option<AuthContext>,
    model_id: &str,
) -> Response<BoxBody> {
    if model_id.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "model_id is required",
        );
    }

    let uri = req.uri().clone();
    let params: Vec<(String, String)> = uri
        .query()
        .map(|q| {
            url::form_urlencoded::parse(q.as_bytes())
                .into_owned()
                .collect()
        })
        .unwrap_or_default();

    let token: Option<String> = params
        .iter()
        .find(|(k, _)| k == "token")
        .map(|(_, v)| v.clone())
        .filter(|v| !v.is_empty());

    let model_id_owned = model_id.to_string();
    let result = tokio::task::spawn_blocking(move || {
        talu::repo::repo_list_files(&model_id_owned, token.as_deref())
    })
    .await;

    match result {
        Ok(Ok(files)) => {
            let entries: Vec<FileEntry> = files
                .into_iter()
                .map(|f| FileEntry { filename: f })
                .collect();
            json_response(
                StatusCode::OK,
                &FileListResponse {
                    model_id: model_id.to_string(),
                    files: entries,
                },
            )
        }
        Ok(Err(e)) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "list_files_error",
            &format!("Failed to list files: {e}"),
        ),
        Err(e) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "list_files_error",
            &format!("File list task failed: {e}"),
        ),
    }
}

fn param_str<'a>(params: &'a [(String, String)], key: &str) -> Option<&'a str> {
    params
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .filter(|v| !v.is_empty())
}

fn json_response<T: Serialize>(status: StatusCode, payload: &T) -> Response<BoxBody> {
    let body = serde_json::to_vec(payload).unwrap_or_else(|_| b"{}".to_vec());
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
