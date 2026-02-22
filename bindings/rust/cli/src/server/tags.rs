//! Tag management endpoints.
//!
//! Provides REST API handlers for tag CRUD operations.

use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde::{Deserialize, Serialize};
use talu::documents::DocumentsHandle;
use talu::storage::{StorageError, StorageHandle, TagCreate, TagRecord, TagUpdate};
use utoipa::ToSchema;

use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;

// =============================================================================
// Request/Response types
// =============================================================================

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct TagResponse {
    id: String,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    created_at: i64,
    updated_at: i64,
}

/// Tag usage statistics.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct TagUsage {
    conversations: usize,
    documents: usize,
    total: usize,
}

/// Tag response with usage statistics (for GET /v1/tags/:id).
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct TagResponseWithUsage {
    id: String,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    created_at: i64,
    updated_at: i64,
    usage: TagUsage,
}

impl From<TagRecord> for TagResponse {
    fn from(r: TagRecord) -> Self {
        Self {
            id: r.tag_id,
            name: r.name,
            color: r.color,
            description: r.description,
            created_at: r.created_at,
            updated_at: r.updated_at,
        }
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct TagListResponse {
    data: Vec<TagResponse>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct CreateTagRequest {
    name: String,
    #[serde(default)]
    color: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct UpdateTagRequest {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    color: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

// =============================================================================
// Handlers
// =============================================================================

#[utoipa::path(get, path = "/v1/tags", tag = "Tags",
    responses((status = 200, body = TagListResponse)))]
/// GET /v1/tags - List all tags
pub async fn handle_list(
    state: Arc<AppState>,
    _req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let bucket = match state.bucket_path.as_ref() {
        Some(p) => p,
        None => {
            return json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no_storage",
                "Storage not configured",
            )
        }
    };

    let group_id = auth.as_ref().and_then(|a| a.group_id.as_deref());

    let handle = match StorageHandle::open(bucket) {
        Ok(h) => h,
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "storage_error",
                &e.to_string(),
            )
        }
    };

    let tags = match handle.list_tags(group_id) {
        Ok(t) => t,
        Err(e) => return storage_error_response(e),
    };

    let response = TagListResponse {
        data: tags.into_iter().map(TagResponse::from).collect(),
    };

    json_response(StatusCode::OK, &response)
}

#[utoipa::path(get, path = "/v1/tags/{tag_id}", tag = "Tags",
    params(("tag_id" = String, Path, description = "Tag ID")),
    responses((status = 200, body = TagResponseWithUsage)))]
/// GET /v1/tags/:id - Get a specific tag with usage statistics
pub async fn handle_get(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path();
    let tag_id = extract_tag_id(path);

    let bucket = match state.bucket_path.as_ref() {
        Some(p) => p,
        None => {
            return json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no_storage",
                "Storage not configured",
            )
        }
    };

    let handle = match StorageHandle::open(bucket) {
        Ok(h) => h,
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "storage_error",
                &e.to_string(),
            )
        }
    };

    let tag = match handle.get_tag(tag_id) {
        Ok(t) => t,
        Err(e) => return storage_error_response(e),
    };

    // Get conversation count for this tag
    let conversation_count = match handle.get_tag_conversations(tag_id) {
        Ok(ids) => ids.len(),
        Err(_) => 0, // On error, report 0 instead of failing the request
    };

    // Get document count for this tag
    let document_count = match DocumentsHandle::open(bucket) {
        Ok(docs) => match docs.get_by_tag(tag_id) {
            Ok(ids) => ids.len(),
            Err(_) => 0,
        },
        Err(_) => 0,
    };

    let response = TagResponseWithUsage {
        id: tag.tag_id,
        name: tag.name,
        color: tag.color,
        description: tag.description,
        created_at: tag.created_at,
        updated_at: tag.updated_at,
        usage: TagUsage {
            conversations: conversation_count,
            documents: document_count,
            total: conversation_count + document_count,
        },
    };

    json_response(StatusCode::OK, &response)
}

#[utoipa::path(post, path = "/v1/tags", tag = "Tags",
    request_body = CreateTagRequest,
    responses((status = 201, body = TagResponse)))]
/// POST /v1/tags - Create a new tag
pub async fn handle_create(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let bucket = match state.bucket_path.as_ref() {
        Some(p) => p,
        None => {
            return json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no_storage",
                "Storage not configured",
            )
        }
    };

    let body = match read_body(req).await {
        Ok(b) => b,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e),
    };

    let create_req: CreateTagRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    // Generate tag ID
    let tag_id = uuid::Uuid::new_v4().to_string();
    let group_id = auth.as_ref().and_then(|a| a.group_id.clone());

    let tag_create = TagCreate {
        tag_id: tag_id.clone(),
        name: create_req.name.clone(),
        color: create_req.color,
        description: create_req.description,
        group_id,
    };

    let handle = match StorageHandle::open(bucket) {
        Ok(h) => h,
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "storage_error",
                &e.to_string(),
            )
        }
    };

    if let Err(e) = handle.create_tag(&tag_create) {
        return storage_error_response(e);
    }

    // Fetch the created tag to return with timestamps
    let tag = match handle.get_tag(&tag_id) {
        Ok(t) => t,
        Err(e) => return storage_error_response(e),
    };

    json_response(StatusCode::CREATED, &TagResponse::from(tag))
}

#[utoipa::path(patch, path = "/v1/tags/{tag_id}", tag = "Tags",
    params(("tag_id" = String, Path, description = "Tag ID")),
    request_body = UpdateTagRequest,
    responses((status = 200, body = TagResponse)))]
/// PATCH /v1/tags/:id - Update a tag
pub async fn handle_patch(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let tag_id = extract_tag_id(&path);

    let bucket = match state.bucket_path.as_ref() {
        Some(p) => p,
        None => {
            return json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no_storage",
                "Storage not configured",
            )
        }
    };

    let body = match read_body(req).await {
        Ok(b) => b,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e),
    };

    let update_req: UpdateTagRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    let tag_update = TagUpdate {
        name: update_req.name,
        color: update_req.color,
        description: update_req.description,
    };

    let handle = match StorageHandle::open(bucket) {
        Ok(h) => h,
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "storage_error",
                &e.to_string(),
            )
        }
    };

    if let Err(e) = handle.update_tag(tag_id, &tag_update) {
        return storage_error_response(e);
    }

    // Fetch the updated tag
    let tag = match handle.get_tag(tag_id) {
        Ok(t) => t,
        Err(e) => return storage_error_response(e),
    };

    json_response(StatusCode::OK, &TagResponse::from(tag))
}

#[utoipa::path(delete, path = "/v1/tags/{tag_id}", tag = "Tags",
    params(("tag_id" = String, Path, description = "Tag ID")),
    responses((status = 204)))]
/// DELETE /v1/tags/:id - Delete a tag
pub async fn handle_delete(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path();
    let tag_id = extract_tag_id(path);

    let bucket = match state.bucket_path.as_ref() {
        Some(p) => p,
        None => {
            return json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no_storage",
                "Storage not configured",
            )
        }
    };

    let handle = match StorageHandle::open(bucket) {
        Ok(h) => h,
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "storage_error",
                &e.to_string(),
            )
        }
    };

    if let Err(e) = handle.delete_tag(tag_id) {
        return storage_error_response(e);
    }

    Response::builder()
        .status(StatusCode::NO_CONTENT)
        .body(Full::new(Bytes::new()).boxed())
        .unwrap()
}

// =============================================================================
// Helper functions
// =============================================================================

fn extract_tag_id(path: &str) -> &str {
    // Path is /v1/tags/:id or /tags/:id
    let stripped = path
        .strip_prefix("/v1/tags/")
        .or_else(|| path.strip_prefix("/tags/"))
        .unwrap_or("");
    // Remove any trailing path segments
    stripped.split('/').next().unwrap_or("")
}

async fn read_body(req: Request<Incoming>) -> Result<Vec<u8>, String> {
    let body = req.into_body();
    let collected = body
        .collect()
        .await
        .map_err(|e| format!("Failed to read body: {}", e))?;
    Ok(collected.to_bytes().to_vec())
}

fn json_response<T: Serialize>(status: StatusCode, data: &T) -> Response<BoxBody> {
    let body = serde_json::to_vec(data).unwrap_or_default();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}

fn json_error(status: StatusCode, code: &str, message: &str) -> Response<BoxBody> {
    let body = serde_json::json!({
        "error": {
            "code": code,
            "message": message
        }
    });
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body.to_string())).boxed())
        .unwrap()
}

fn storage_error_response(err: StorageError) -> Response<BoxBody> {
    match err {
        StorageError::TagNotFound(id) => json_error(
            StatusCode::NOT_FOUND,
            "tag_not_found",
            &format!("Tag not found: {}", id),
        ),
        StorageError::InvalidArgument(msg) => {
            json_error(StatusCode::BAD_REQUEST, "invalid_argument", &msg)
        }
        StorageError::StorageNotFound(p) => json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "storage_not_found",
            &format!("Storage not found: {}", p.display()),
        ),
        _ => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage_error",
            &err.to_string(),
        ),
    }
}
