//! Document management endpoints.
//!
//! Provides REST API handlers for document CRUD, search, and tag operations.

use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde::{Deserialize, Serialize};
use talu::documents::{
    DocumentError, DocumentRecord, DocumentSummary, DocumentsHandle, SearchResult,
};

use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;

// =============================================================================
// Request/Response types
// =============================================================================

#[derive(Debug, Serialize)]
struct DocumentResponse {
    id: String,
    #[serde(rename = "type")]
    doc_type: String,
    title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tags_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    marker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    group_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    owner_id: Option<String>,
    created_at: i64,
    updated_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    expires_at: Option<i64>,
}

impl From<DocumentRecord> for DocumentResponse {
    fn from(r: DocumentRecord) -> Self {
        let content: Option<serde_json::Value> = if r.doc_json.is_empty() {
            None
        } else {
            serde_json::from_str(&r.doc_json).ok()
        };
        Self {
            id: r.doc_id,
            doc_type: r.doc_type,
            title: r.title,
            content,
            tags_text: r.tags_text,
            parent_id: r.parent_id,
            marker: r.marker,
            group_id: r.group_id,
            owner_id: r.owner_id,
            created_at: r.created_at_ms,
            updated_at: r.updated_at_ms,
            expires_at: if r.expires_at_ms > 0 {
                Some(r.expires_at_ms)
            } else {
                None
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct DocumentSummaryResponse {
    id: String,
    #[serde(rename = "type")]
    doc_type: String,
    title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    marker: Option<String>,
    created_at: i64,
    updated_at: i64,
}

impl From<DocumentSummary> for DocumentSummaryResponse {
    fn from(s: DocumentSummary) -> Self {
        Self {
            id: s.doc_id,
            doc_type: s.doc_type,
            title: s.title,
            marker: s.marker,
            created_at: s.created_at_ms,
            updated_at: s.updated_at_ms,
        }
    }
}

#[derive(Debug, Serialize)]
struct DocumentListResponse {
    data: Vec<DocumentSummaryResponse>,
    has_more: bool,
}

#[derive(Debug, Serialize)]
struct SearchResultResponse {
    id: String,
    #[serde(rename = "type")]
    doc_type: String,
    title: String,
    snippet: String,
}

impl From<SearchResult> for SearchResultResponse {
    fn from(r: SearchResult) -> Self {
        Self {
            id: r.doc_id,
            doc_type: r.doc_type,
            title: r.title,
            snippet: r.snippet,
        }
    }
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    data: Vec<SearchResultResponse>,
}

#[derive(Debug, Deserialize)]
struct CreateDocumentRequest {
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "type")]
    doc_type: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    content: Option<serde_json::Value>,
    #[serde(default)]
    tags_text: Option<String>,
    #[serde(default)]
    parent_id: Option<String>,
    #[serde(default)]
    marker: Option<String>,
    #[serde(default)]
    group_id: Option<String>,
    #[serde(default)]
    owner_id: Option<String>,
    #[serde(default)]
    ttl_seconds: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct UpdateDocumentRequest {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    content: Option<serde_json::Value>,
    #[serde(default)]
    tags_text: Option<String>,
    #[serde(default)]
    marker: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SearchRequest {
    query: String,
    #[serde(default)]
    limit: Option<u32>,
    #[serde(default, rename = "type")]
    doc_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TagsRequest {
    tags: Vec<String>,
}

// =============================================================================
// Handlers
// =============================================================================

/// GET /v1/documents - List documents
pub async fn handle_list(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
    plugin_owner: Option<String>,
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

    // Parse query params manually
    let query_str = req.uri().query().unwrap_or("");
    let limit = parse_query_param(query_str, "limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(100u32);
    let doc_type = parse_query_param(query_str, "type");
    let marker = parse_query_param(query_str, "marker");
    let group_id = parse_query_param(query_str, "group_id");

    // Plugin storage requires a valid capability token; the token's plugin_id
    // becomes the owner_id filter so plugins only see their own documents.
    let owner_id = if doc_type.as_deref() == Some("plugin_storage") {
        match plugin_owner {
            Some(pid) => Some(pid),
            None => {
                return json_error(
                    StatusCode::FORBIDDEN,
                    "forbidden",
                    "type=plugin_storage requires a plugin capability token",
                )
            }
        }
    } else {
        parse_query_param(query_str, "owner_id")
    };

    let storage_path = match auth.as_ref() {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };

    let handle = match DocumentsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    let docs = match handle.list(
        doc_type.as_deref(),
        group_id.as_deref(),
        owner_id.as_deref(),
        marker.as_deref(),
        limit,
    ) {
        Ok(d) => d,
        Err(e) => return document_error_response(e),
    };

    let has_more = docs.len() >= limit as usize;
    let response = DocumentListResponse {
        data: docs
            .into_iter()
            .map(DocumentSummaryResponse::from)
            .collect(),
        has_more,
    };

    json_response(StatusCode::OK, &response)
}

/// GET /v1/documents/:id - Get a document
pub async fn handle_get(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path();
    let doc_id = extract_doc_id(path);

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

    let storage_path = match auth.as_ref() {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };

    let handle = match DocumentsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    let doc = match handle.get(doc_id) {
        Ok(Some(d)) => d,
        Ok(None) => {
            return json_error(
                StatusCode::NOT_FOUND,
                "not_found",
                &format!("Document not found: {}", doc_id),
            )
        }
        Err(e) => return document_error_response(e),
    };

    json_response(StatusCode::OK, &DocumentResponse::from(doc))
}

/// POST /v1/documents - Create a document
pub async fn handle_create(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
    plugin_owner: Option<String>,
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

    let create_req: CreateDocumentRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    // Plugin storage requires a valid capability token; the token's plugin_id
    // becomes the owner_id so each plugin's data is isolated.
    let plugin_owner_id = if create_req.doc_type == "plugin_storage" {
        match plugin_owner {
            Some(pid) => Some(pid),
            None => {
                return json_error(
                    StatusCode::FORBIDDEN,
                    "forbidden",
                    "type=plugin_storage requires a plugin capability token",
                )
            }
        }
    } else {
        None
    };

    let doc_id = create_req
        .id
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let title = create_req.title.unwrap_or_default();
    let content_json = create_req
        .content
        .map(|v| v.to_string())
        .unwrap_or_else(|| "{}".to_string());
    let group_id = create_req
        .group_id
        .or_else(|| auth.as_ref().and_then(|a| a.group_id.clone()));

    let storage_path = match auth.as_ref() {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };

    let handle = match DocumentsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    // Plugin storage: force owner_id to the authenticated plugin_id.
    let owner_id = plugin_owner_id.or(create_req.owner_id);

    if let Err(e) = handle.create(
        &doc_id,
        &create_req.doc_type,
        &title,
        &content_json,
        create_req.tags_text.as_deref(),
        create_req.parent_id.as_deref(),
        create_req.marker.as_deref(),
        group_id.as_deref(),
        owner_id.as_deref(),
    ) {
        return document_error_response(e);
    }

    // Set TTL if specified
    if let Some(ttl) = create_req.ttl_seconds {
        if let Err(e) = handle.set_ttl(&doc_id, ttl) {
            return document_error_response(e);
        }
    }

    // Fetch the created document
    let doc = match handle.get(&doc_id) {
        Ok(Some(d)) => d,
        Ok(None) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "create_failed",
                "Document created but not found",
            )
        }
        Err(e) => return document_error_response(e),
    };

    json_response(StatusCode::CREATED, &DocumentResponse::from(doc))
}

/// PATCH /v1/documents/:id - Update a document
pub async fn handle_update(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let doc_id = extract_doc_id(&path);

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

    let update_req: UpdateDocumentRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    let storage_path = match auth.as_ref() {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };

    let handle = match DocumentsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    let content_json = update_req.content.map(|v| v.to_string());

    if let Err(e) = handle.update(
        doc_id,
        update_req.title.as_deref(),
        content_json.as_deref(),
        update_req.tags_text.as_deref(),
        update_req.marker.as_deref(),
    ) {
        return document_error_response(e);
    }

    // Fetch the updated document
    let doc = match handle.get(doc_id) {
        Ok(Some(d)) => d,
        Ok(None) => {
            return json_error(
                StatusCode::NOT_FOUND,
                "not_found",
                &format!("Document not found: {}", doc_id),
            )
        }
        Err(e) => return document_error_response(e),
    };

    json_response(StatusCode::OK, &DocumentResponse::from(doc))
}

/// DELETE /v1/documents/:id - Delete a document
pub async fn handle_delete(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path();
    let doc_id = extract_doc_id(path);

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

    let storage_path = match auth.as_ref() {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };

    let handle = match DocumentsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    if let Err(e) = handle.delete(doc_id) {
        return document_error_response(e);
    }

    Response::builder()
        .status(StatusCode::NO_CONTENT)
        .body(Full::new(Bytes::new()).boxed())
        .unwrap()
}

/// POST /v1/documents/search - Search documents
pub async fn handle_search(
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

    let search_req: SearchRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    let storage_path = match auth.as_ref() {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };

    let handle = match DocumentsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    let limit = search_req.limit.unwrap_or(20);
    let results = match handle.search(&search_req.query, search_req.doc_type.as_deref(), limit) {
        Ok(r) => r,
        Err(e) => return document_error_response(e),
    };

    let response = SearchResponse {
        data: results
            .into_iter()
            .map(SearchResultResponse::from)
            .collect(),
    };

    json_response(StatusCode::OK, &response)
}

/// GET /v1/documents/:id/tags - Get document tags
pub async fn handle_get_tags(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path();
    let doc_id = extract_doc_id_before_tags(path);

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

    let storage_path = match auth.as_ref() {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };

    let handle = match DocumentsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    let tags = match handle.get_tags(doc_id) {
        Ok(t) => t,
        Err(e) => return document_error_response(e),
    };

    json_response(StatusCode::OK, &serde_json::json!({ "tags": tags }))
}

/// POST /v1/documents/:id/tags - Add tags to document
pub async fn handle_add_tags(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let doc_id = extract_doc_id_before_tags(&path);

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

    let tags_req: TagsRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    let storage_path = match auth.as_ref() {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };

    let group_id = auth.as_ref().and_then(|a| a.group_id.clone());

    let handle = match DocumentsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    for tag in &tags_req.tags {
        if let Err(e) = handle.add_tag(doc_id, tag, group_id.as_deref()) {
            return document_error_response(e);
        }
    }

    // Return updated tags
    let tags = match handle.get_tags(doc_id) {
        Ok(t) => t,
        Err(e) => return document_error_response(e),
    };

    json_response(StatusCode::OK, &serde_json::json!({ "tags": tags }))
}

/// DELETE /v1/documents/:id/tags - Remove tags from document
pub async fn handle_remove_tags(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let doc_id = extract_doc_id_before_tags(&path);

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

    let tags_req: TagsRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    let storage_path = match auth.as_ref() {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };

    let group_id = auth.as_ref().and_then(|a| a.group_id.clone());

    let handle = match DocumentsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    for tag in &tags_req.tags {
        if let Err(e) = handle.remove_tag(doc_id, tag, group_id.as_deref()) {
            return document_error_response(e);
        }
    }

    // Return updated tags
    let tags = match handle.get_tags(doc_id) {
        Ok(t) => t,
        Err(e) => return document_error_response(e),
    };

    json_response(StatusCode::OK, &serde_json::json!({ "tags": tags }))
}

// =============================================================================
// Helper functions
// =============================================================================

fn extract_doc_id(path: &str) -> &str {
    // Path is /v1/documents/:id or /documents/:id
    let stripped = path
        .strip_prefix("/v1/documents/")
        .or_else(|| path.strip_prefix("/documents/"))
        .unwrap_or("");
    // Remove any trailing path segments
    stripped.split('/').next().unwrap_or("")
}

fn extract_doc_id_before_tags(path: &str) -> &str {
    // Path is /v1/documents/:id/tags or /documents/:id/tags
    let stripped = path
        .strip_prefix("/v1/documents/")
        .or_else(|| path.strip_prefix("/documents/"))
        .unwrap_or("");
    // Get the ID before /tags
    stripped
        .strip_suffix("/tags")
        .unwrap_or(stripped)
        .split('/')
        .next()
        .unwrap_or("")
}

/// Simple query parameter parser.
fn parse_query_param(query: &str, key: &str) -> Option<String> {
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            if k == key {
                // URL decode the value (simple version)
                return Some(v.replace("%20", " ").replace("+", " "));
            }
        }
    }
    None
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

fn document_error_response(err: DocumentError) -> Response<BoxBody> {
    match err {
        DocumentError::DocumentNotFound(id) => json_error(
            StatusCode::NOT_FOUND,
            "not_found",
            &format!("Document not found: {}", id),
        ),
        DocumentError::InvalidArgument(msg) => {
            json_error(StatusCode::BAD_REQUEST, "invalid_argument", &msg)
        }
        DocumentError::StorageNotFound(p) => json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "storage_not_found",
            &format!("Storage not found: {}", p.display()),
        ),
        DocumentError::StorageError(msg) => {
            json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &msg)
        }
    }
}
