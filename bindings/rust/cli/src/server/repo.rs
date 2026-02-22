//! Repository management endpoints (`/v1/repo/...`).
//!
//! Exposes the CLI's `talu ls`, `talu get`, `talu rm`, and pin management
//! functionality over HTTP so the web UI can manage the local model cache.

use std::collections::HashSet;
use std::convert::Infallible;
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

use crate::pin_store::PinStore;
use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

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
    /// Quantization scheme (e.g., "F16", "GAF4_64", "MXFP4"), if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quant_scheme: Option<String>,
    /// Whether this model is pinned in the active profile.
    pub pinned: bool,
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

/// A pinned model entry.
#[derive(Serialize, ToSchema)]
pub(crate) struct PinnedModelResponse {
    /// Model ID (e.g., "meta-llama/Llama-3.2-1B").
    pub model_uri: String,
    /// When the model was pinned (Unix timestamp, milliseconds).
    pub pinned_at_ms: i64,
    /// Cached size in bytes, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
}

/// Response for GET /v1/repo/pins.
#[derive(Serialize, ToSchema)]
pub(crate) struct PinListResponse {
    pub pins: Vec<PinnedModelResponse>,
}

/// Request body for POST /v1/repo/pins.
#[derive(Deserialize, Serialize, ToSchema)]
pub(crate) struct PinRequest {
    /// Model ID to pin.
    pub model_id: String,
}

/// Response for pin/unpin operations.
#[derive(Serialize, ToSchema)]
pub(crate) struct PinActionResponse {
    /// The model ID.
    pub model_id: String,
    /// Whether the model is now pinned.
    pub pinned: bool,
}

/// Request body for POST /v1/repo/sync-pins.
#[derive(Deserialize, Serialize, ToSchema)]
pub(crate) struct SyncPinsRequest {
    /// Dry run (report only, no downloads). Defaults to true for safety.
    #[serde(default = "default_true")]
    pub dry_run: bool,
    /// Skip downloading weight files.
    #[serde(default)]
    pub skip_weights: bool,
    /// HuggingFace token for private repos.
    #[serde(default)]
    pub token: Option<String>,
    /// Custom endpoint URL.
    #[serde(default)]
    pub endpoint_url: Option<String>,
}

fn default_true() -> bool {
    true
}

/// Response for POST /v1/repo/sync-pins.
#[derive(Serialize, ToSchema)]
pub(crate) struct SyncPinsResponse {
    /// Total number of pinned models.
    pub total: usize,
    /// Number already cached locally.
    pub cached: usize,
    /// Number missing from local cache.
    pub missing: usize,
    /// Number downloaded in this sync (0 if dry_run).
    pub downloaded: usize,
    /// Estimated total bytes for missing models (from HF API, best-effort).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub missing_size_bytes: Option<u64>,
    /// Errors encountered during download.
    pub errors: Vec<String>,
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

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// GET /v1/repo/models — list all locally cached models with metadata.
#[utoipa::path(get, path = "/v1/repo/models", tag = "Repository",
    params(
        ("source" = Option<String>, Query, description = "Filter by source: 'hub' or 'managed'"),
        ("pinned" = Option<bool>, Query, description = "Filter to pinned models only"),
    ),
    responses(
        (status = 200, body = RepoListResponse),
        (status = 500, body = super::http::ErrorResponse, description = "Internal error"),
    ))]
pub async fn handle_list(
    state: Arc<AppState>,
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

    let pinned_only: bool = params
        .iter()
        .find(|(k, _)| k == "pinned")
        .map(|(_, v)| v == "true" || v == "1")
        .unwrap_or(false);

    let bucket_path = state.bucket_path.clone();

    let result = tokio::task::spawn_blocking(move || {
        let cached = talu::repo::repo_list_models(false).unwrap_or_default();
        let total_size = talu::repo::repo_total_size();

        // Load pinned set for filtering and annotation.
        let pinned_set: HashSet<String> = bucket_path
            .as_ref()
            .and_then(|bp| PinStore::open(&bp.join("meta.sqlite")).ok())
            .and_then(|store| store.list_pinned().ok())
            .map(|v| v.into_iter().collect())
            .unwrap_or_default();

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
                if pinned_only && !pinned_set.contains(&m.id) {
                    return false;
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
                        let quant = Some(format_quant_scheme(
                            info.quant_method,
                            info.quant_bits,
                            info.quant_group_size,
                        ));
                        (arch, quant)
                    }
                    Err(_) => (None, None),
                };

                let is_pinned = pinned_set.contains(&m.id);

                CachedModelResponse {
                    id: m.id,
                    path: m.path,
                    source,
                    size_bytes,
                    mtime,
                    architecture,
                    quant_scheme,
                    pinned: is_pinned,
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
        (status = 400, body = super::http::ErrorResponse, description = "Bad request"),
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
            )
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

    // Non-streaming: block until download completes.
    let model_id = request.model_id.clone();
    let options = talu::repo::DownloadOptions {
        token: request.token,
        force: request.force,
        endpoint_url: request.endpoint_url,
        skip_weights: request.skip_weights,
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
    let options = talu::repo::DownloadOptions {
        token: request.token,
        force: request.force,
        endpoint_url: request.endpoint_url,
        skip_weights: request.skip_weights,
    };

    std::thread::spawn(move || {
        let tx_progress = tx.clone();
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
                let _ = tx_progress.send(Bytes::from(chunk));
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
///
/// The `model_id` is extracted from the URL path. Model IDs containing
/// slashes (e.g., "meta-llama/Llama-3.2-1B") are URL-encoded in the path.
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

// ---------------------------------------------------------------------------
// Pin endpoints
// ---------------------------------------------------------------------------

/// GET /v1/repo/pins — list all pinned models.
#[utoipa::path(get, path = "/v1/repo/pins", tag = "Repository",
    responses(
        (status = 200, body = PinListResponse),
        (status = 400, body = super::http::ErrorResponse, description = "No bucket configured"),
    ))]
pub async fn handle_list_pins(
    state: Arc<AppState>,
    _req: Request<Incoming>,
    _auth_ctx: Option<AuthContext>,
) -> Response<BoxBody> {
    let bucket_path = match state.bucket_path.clone() {
        Some(bp) => bp,
        None => return no_bucket_error(),
    };

    let result = tokio::task::spawn_blocking(move || {
        let store = PinStore::open(&bucket_path.join("meta.sqlite"))?;
        store.list_pinned_entries()
    })
    .await;

    match result {
        Ok(Ok(entries)) => {
            let pins: Vec<PinnedModelResponse> = entries
                .into_iter()
                .map(|e| PinnedModelResponse {
                    model_uri: e.model_uri,
                    pinned_at_ms: e.pinned_at_ms,
                    size_bytes: e.size_bytes,
                })
                .collect();
            json_response(StatusCode::OK, &PinListResponse { pins })
        }
        Ok(Err(e)) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "pin_store_error",
            &format!("Failed to list pins: {e}"),
        ),
        Err(e) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "pin_store_error",
            &format!("Pin list task failed: {e}"),
        ),
    }
}

/// POST /v1/repo/pins — pin a model.
#[utoipa::path(post, path = "/v1/repo/pins", tag = "Repository",
    request_body = PinRequest,
    responses(
        (status = 200, body = PinActionResponse),
        (status = 400, body = super::http::ErrorResponse, description = "Invalid request"),
    ))]
pub async fn handle_pin(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth_ctx: Option<AuthContext>,
) -> Response<BoxBody> {
    let bucket_path = match state.bucket_path.clone() {
        Some(bp) => bp,
        None => return no_bucket_error(),
    };

    let body_bytes = match req.into_body().collect().await {
        Ok(body) => body.to_bytes(),
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "invalid_request", "Invalid body"),
    };

    let request: PinRequest = match serde_json::from_slice(&body_bytes) {
        Ok(val) => val,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &format!("Invalid JSON: {e}"),
            )
        }
    };

    if request.model_id.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "model_id is required",
        );
    }

    let model_id = request.model_id.clone();
    let result = tokio::task::spawn_blocking(move || {
        let store = PinStore::open(&bucket_path.join("meta.sqlite"))?;
        store.pin(&model_id)?;

        // Update size if model is already cached.
        let size = talu::repo::repo_size(&model_id);
        if size > 0 {
            let _ = store.upsert_size_bytes(&model_id, size);
        }

        Ok::<_, anyhow::Error>(())
    })
    .await;

    match result {
        Ok(Ok(())) => json_response(
            StatusCode::OK,
            &PinActionResponse {
                model_id: request.model_id,
                pinned: true,
            },
        ),
        Ok(Err(e)) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "pin_store_error",
            &format!("Failed to pin: {e}"),
        ),
        Err(e) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "pin_store_error",
            &format!("Pin task failed: {e}"),
        ),
    }
}

/// DELETE /v1/repo/pins/{model_id} — unpin a model.
#[utoipa::path(delete, path = "/v1/repo/pins/{model_id}", tag = "Repository",
    params(("model_id" = String, Path, description = "Model ID to unpin (URL-encoded)")),
    responses(
        (status = 200, body = PinActionResponse),
        (status = 400, body = super::http::ErrorResponse, description = "Invalid request"),
    ))]
pub async fn handle_unpin(
    state: Arc<AppState>,
    _req: Request<Incoming>,
    _auth_ctx: Option<AuthContext>,
    model_id: &str,
) -> Response<BoxBody> {
    let bucket_path = match state.bucket_path.clone() {
        Some(bp) => bp,
        None => return no_bucket_error(),
    };

    if model_id.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "model_id is required",
        );
    }

    let model_id_owned = model_id.to_string();
    let result = tokio::task::spawn_blocking(move || {
        let store = PinStore::open(&bucket_path.join("meta.sqlite"))?;
        store.unpin(&model_id_owned)?;
        Ok::<_, anyhow::Error>(())
    })
    .await;

    match result {
        Ok(Ok(())) => json_response(
            StatusCode::OK,
            &PinActionResponse {
                model_id: model_id.to_string(),
                pinned: false,
            },
        ),
        Ok(Err(e)) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "pin_store_error",
            &format!("Failed to unpin: {e}"),
        ),
        Err(e) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "pin_store_error",
            &format!("Unpin task failed: {e}"),
        ),
    }
}

/// POST /v1/repo/sync-pins — synchronize pinned models (download missing ones).
///
/// When the `Accept` header contains `text/event-stream`, the response streams
/// SSE progress events per model. Otherwise, blocks until sync completes and
/// returns a JSON summary.
#[utoipa::path(post, path = "/v1/repo/sync-pins", tag = "Repository",
    request_body = SyncPinsRequest,
    responses(
        (status = 200, body = SyncPinsResponse),
        (status = 400, body = super::http::ErrorResponse, description = "No bucket configured"),
    ))]
pub async fn handle_sync_pins(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth_ctx: Option<AuthContext>,
) -> Response<BoxBody> {
    let bucket_path = match state.bucket_path.clone() {
        Some(bp) => bp,
        None => return no_bucket_error(),
    };

    let wants_stream = req
        .headers()
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains("text/event-stream"));

    let body_bytes = match req.into_body().collect().await {
        Ok(body) => body.to_bytes(),
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "invalid_request", "Invalid body"),
    };

    let request: SyncPinsRequest = match serde_json::from_slice(&body_bytes) {
        Ok(val) => val,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &format!("Invalid JSON: {e}"),
            )
        }
    };

    if wants_stream && !request.dry_run {
        return handle_sync_pins_streaming(bucket_path, request);
    }

    let result = tokio::task::spawn_blocking(move || sync_pins_blocking(&bucket_path, &request))
        .await;

    match result {
        Ok(Ok(resp)) => json_response(StatusCode::OK, &resp),
        Ok(Err(e)) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "sync_error",
            &format!("Sync failed: {e}"),
        ),
        Err(e) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "sync_error",
            &format!("Sync task failed: {e}"),
        ),
    }
}

/// Blocking sync-pins logic shared between JSON and SSE paths.
fn sync_pins_blocking(
    bucket_path: &std::path::Path,
    request: &SyncPinsRequest,
) -> Result<SyncPinsResponse, anyhow::Error> {
    let store = PinStore::open(&bucket_path.join("meta.sqlite"))?;
    let entries = store.list_pinned_entries()?;
    let total = entries.len();

    // Classify each pinned model as cached or missing.
    let cached_models = talu::repo::repo_list_models(false).unwrap_or_default();
    let cached_ids: HashSet<String> = cached_models.into_iter().map(|m| m.id).collect();

    let mut cached = 0usize;
    let mut missing_ids = Vec::new();
    for entry in &entries {
        if cached_ids.contains(&entry.model_uri) {
            cached += 1;
        } else {
            missing_ids.push(entry.model_uri.clone());
        }
    }
    let missing = missing_ids.len();

    // Hydrate sizes for missing models from HF API (best-effort).
    let missing_size_bytes = if !missing_ids.is_empty() {
        hydrate_missing_sizes(&missing_ids, request.token.as_deref(), request.endpoint_url.as_deref(), request.skip_weights)
    } else {
        None
    };

    let mut downloaded = 0usize;
    let mut errors = Vec::new();

    if !request.dry_run {
        for model_id in &missing_ids {
            let options = talu::repo::DownloadOptions {
                token: request.token.clone(),
                force: false,
                endpoint_url: request.endpoint_url.clone(),
                skip_weights: request.skip_weights,
            };
            match talu::repo::repo_fetch(model_id, options, None) {
                Ok(_) => {
                    downloaded += 1;
                    let size = talu::repo::repo_size(model_id);
                    if size > 0 {
                        let _ = store.upsert_size_bytes(model_id, size);
                    }
                }
                Err(e) => errors.push(format!("{model_id}: {e}")),
            }
        }
    }

    Ok(SyncPinsResponse {
        total,
        cached,
        missing,
        downloaded,
        missing_size_bytes,
        errors,
    })
}

/// SSE streaming variant for sync-pins: emits per-model progress events.
fn handle_sync_pins_streaming(
    bucket_path: std::path::PathBuf,
    request: SyncPinsRequest,
) -> Response<BoxBody> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Bytes>();

    std::thread::spawn(move || {
        let store = match PinStore::open(&bucket_path.join("meta.sqlite")) {
            Ok(s) => s,
            Err(e) => {
                let err = json!({ "event": "error", "message": format!("{e}") });
                let _ = tx.send(Bytes::from(format!("data: {}\n\n", err)));
                return;
            }
        };

        let entries = match store.list_pinned_entries() {
            Ok(e) => e,
            Err(e) => {
                let err = json!({ "event": "error", "message": format!("{e}") });
                let _ = tx.send(Bytes::from(format!("data: {}\n\n", err)));
                return;
            }
        };

        let total = entries.len();
        let cached_models = talu::repo::repo_list_models(false).unwrap_or_default();
        let cached_ids: HashSet<String> = cached_models.into_iter().map(|m| m.id).collect();

        let mut cached = 0usize;
        let mut missing_ids = Vec::new();
        for entry in &entries {
            if cached_ids.contains(&entry.model_uri) {
                cached += 1;
            } else {
                missing_ids.push(entry.model_uri.clone());
            }
        }
        let missing = missing_ids.len();

        // Emit scan summary.
        let scan = json!({
            "event": "scan",
            "total": total,
            "cached": cached,
            "missing": missing,
        });
        let _ = tx.send(Bytes::from(format!("data: {}\n\n", scan)));

        let mut downloaded = 0usize;
        let mut errors = Vec::new();

        for (i, model_id) in missing_ids.iter().enumerate() {
            let start_evt = json!({
                "event": "download_start",
                "model_id": model_id,
                "index": i,
                "of": missing,
            });
            let _ = tx.send(Bytes::from(format!("data: {}\n\n", start_evt)));

            let tx_progress = tx.clone();
            let mid = model_id.clone();
            let callback: talu::repo::ProgressCallback =
                Box::new(move |progress: talu::repo::DownloadProgress| {
                    let event = json!({
                        "event": "progress",
                        "model_id": mid,
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
                    let _ = tx_progress.send(Bytes::from(format!("data: {}\n\n", event)));
                });

            let options = talu::repo::DownloadOptions {
                token: request.token.clone(),
                force: false,
                endpoint_url: request.endpoint_url.clone(),
                skip_weights: request.skip_weights,
            };

            match talu::repo::repo_fetch(model_id, options, Some(callback)) {
                Ok(_) => {
                    downloaded += 1;
                    let size = talu::repo::repo_size(model_id);
                    if size > 0 {
                        let _ = store.upsert_size_bytes(model_id, size);
                    }
                    let ok_evt = json!({
                        "event": "download_complete",
                        "model_id": model_id,
                    });
                    let _ = tx.send(Bytes::from(format!("data: {}\n\n", ok_evt)));
                }
                Err(e) => {
                    let err_msg = format!("{model_id}: {e}");
                    errors.push(err_msg.clone());
                    let err_evt = json!({
                        "event": "download_error",
                        "model_id": model_id,
                        "message": err_msg,
                    });
                    let _ = tx.send(Bytes::from(format!("data: {}\n\n", err_evt)));
                }
            }
        }

        // Final done event with summary.
        let done = json!({
            "event": "done",
            "total": total,
            "cached": cached,
            "missing": missing,
            "downloaded": downloaded,
            "errors": errors,
        });
        let _ = tx.send(Bytes::from(format!("data: {}\n\n", done)));
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn no_bucket_error() -> Response<BoxBody> {
    json_error(
        StatusCode::BAD_REQUEST,
        "pin_store_unavailable",
        "Pin operations require a storage bucket (do not use --no-bucket)",
    )
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

/// Hydrate missing model sizes from the HuggingFace API (best-effort).
///
/// Returns the sum of sizes for models where the API returned data, or None
/// if no sizes could be retrieved.
fn hydrate_missing_sizes(
    model_ids: &[String],
    token: Option<&str>,
    endpoint_url: Option<&str>,
    skip_weights: bool,
) -> Option<u64> {
    let endpoint = effective_hf_endpoint(endpoint_url);

    // Build a blocking reqwest client for HF API calls.
    let client = match reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
    {
        Ok(c) => c,
        Err(_) => return None,
    };

    let mut total: u64 = 0;
    let mut any_success = false;

    for model_id in model_ids {
        if let Some(size) = fetch_hf_model_size_blocking(&client, &endpoint, model_id, token, skip_weights) {
            total = total.saturating_add(size);
            any_success = true;
        }
    }

    if any_success { Some(total) } else { None }
}

/// Fetch model size from HF API (blocking). Returns None on failure.
fn fetch_hf_model_size_blocking(
    client: &reqwest::blocking::Client,
    endpoint: &str,
    model_id: &str,
    token: Option<&str>,
    skip_weights: bool,
) -> Option<u64> {
    // Try /api/models/{id}?blobs=true&files_metadata=true first.
    let info_url = format!("{}/api/models/{}?blobs=true&files_metadata=true", endpoint, model_id);
    if let Some(body) = hf_get_text_blocking(client, &info_url, token) {
        if let Some(size) = parse_hf_model_size_bytes(&body, skip_weights) {
            return Some(size);
        }
    }

    // Fallback: /api/models/{id}/tree/main?recursive=0
    let tree_url = format!("{}/api/models/{}/tree/main?recursive=0", endpoint, model_id);
    let body = hf_get_text_blocking(client, &tree_url, token)?;
    parse_hf_tree_size_bytes(&body, skip_weights)
}

fn hf_get_text_blocking(
    client: &reqwest::blocking::Client,
    url: &str,
    token: Option<&str>,
) -> Option<String> {
    let mut req = client.get(url);
    if let Some(t) = token {
        if !t.trim().is_empty() {
            req = req.bearer_auth(t);
        }
    }
    let resp = req.send().ok()?;
    if !resp.status().is_success() {
        return None;
    }
    resp.text().ok()
}

fn effective_hf_endpoint(endpoint_url: Option<&str>) -> String {
    let endpoint = endpoint_url
        .map(str::to_string)
        .or_else(|| std::env::var("HF_ENDPOINT").ok())
        .unwrap_or_else(|| "https://huggingface.co".to_string());
    endpoint.trim_end_matches('/').to_string()
}

/// Parse model size from HF /api/models response (siblings array).
fn parse_hf_model_size_bytes(body: &str, skip_weights: bool) -> Option<u64> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let siblings = value.get("siblings")?.as_array()?;

    let mut total_size = 0u64;
    for sibling in siblings {
        let filename = sibling.get("rfilename").and_then(|v| v.as_str()).unwrap_or_default();
        if filename.starts_with('.') || filename.contains('/') {
            continue;
        }
        if skip_weights && is_weight_filename(filename) {
            continue;
        }
        let size = value_as_u64(sibling.get("size")?)
            .or_else(|| value_as_u64(sibling.get("lfs")?.get("size")?))
            .unwrap_or(0);
        total_size = total_size.saturating_add(size);
    }

    if total_size > 0 { Some(total_size) } else { None }
}

/// Parse model size from HF /tree response.
fn parse_hf_tree_size_bytes(body: &str, skip_weights: bool) -> Option<u64> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let entries = value.as_array()?;

    let mut total_size = 0u64;
    for entry in entries {
        let filename = entry.get("path")
            .or_else(|| entry.get("rfilename"))
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if filename.is_empty() || filename.starts_with('.') || filename.contains('/') {
            continue;
        }
        if let Some(entry_type) = entry.get("type").and_then(|v| v.as_str()) {
            if entry_type != "file" {
                continue;
            }
        }
        if skip_weights && is_weight_filename(filename) {
            continue;
        }
        let size = entry.get("size").and_then(value_as_u64)
            .or_else(|| entry.get("lfs").and_then(|v| v.get("size")).and_then(value_as_u64))
            .unwrap_or(0);
        total_size = total_size.saturating_add(size);
    }

    if total_size > 0 { Some(total_size) } else { None }
}

fn is_weight_filename(filename: &str) -> bool {
    filename.ends_with(".safetensors") || filename.ends_with(".safetensors.index.json")
}

fn value_as_u64(value: &serde_json::Value) -> Option<u64> {
    if let Some(v) = value.as_u64() {
        return Some(v);
    }
    if let Some(v) = value.as_i64() {
        return u64::try_from(v).ok();
    }
    value.as_str().and_then(|s| s.parse::<u64>().ok())
}

/// Format quantization method as a scheme name.
///
/// Mirrors `format_quant_scheme` in `cli/models.rs`.
fn format_quant_scheme(
    method: talu::model::QuantMethod,
    bits: i32,
    group_size: i32,
) -> String {
    match method {
        talu::model::QuantMethod::None => "F16".to_string(),
        talu::model::QuantMethod::Gaffine => format!("GAF{}_{}", bits, group_size),
        talu::model::QuantMethod::Mxfp4 => "MXFP4".to_string(),
        talu::model::QuantMethod::Native => format!("GAF{}_{}", bits, group_size),
    }
}
