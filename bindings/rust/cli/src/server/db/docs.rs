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
use utoipa::ToSchema;

use crate::server::auth_gateway::AuthContext;
use crate::server::http;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;

// =============================================================================
// Request/Response types
// =============================================================================

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct DocumentResponse {
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

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct DocumentSummaryResponse {
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

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct DocumentListResponse {
    data: Vec<DocumentSummaryResponse>,
    has_more: bool,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct SearchResultResponse {
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

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct DocumentSearchResponse {
    data: Vec<SearchResultResponse>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct CreateDocumentRequest {
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

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct UpdateDocumentRequest {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    content: Option<serde_json::Value>,
    #[serde(default)]
    tags_text: Option<String>,
    #[serde(default)]
    marker: Option<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct DocumentSearchRequest {
    query: String,
    #[serde(default)]
    limit: Option<u32>,
    #[serde(default, rename = "type")]
    doc_type: Option<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct DocumentTagsRequest {
    tags: Vec<String>,
}

// =============================================================================
// Handlers
// =============================================================================

#[utoipa::path(get, path = "/v1/db/tables/{table}", tag = "DB::Tables",
    params(
        ("table" = String, Path, description = "Table name (alphanumeric, underscore, hyphen; 1-64 chars)"),
        ("limit" = Option<u32>, Query, description = "Max items to return (default 100)"),
        ("type" = Option<String>, Query, description = "Filter by document type"),
        ("marker" = Option<String>, Query, description = "Filter by marker value"),
        ("group_id" = Option<String>, Query, description = "Filter by group ID"),
        ("owner_id" = Option<String>, Query, description = "Filter by owner ID"),
    ),
    responses((status = 200, body = DocumentListResponse)))]
/// GET /v1/db/tables/{table} - List documents
pub async fn handle_list(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
    plugin_owner: Option<String>,
) -> Response<BoxBody> {
    let (_table_name, table_path) =
        match resolve_table_path(&state, req.uri().path(), auth.as_ref()) {
            Ok(v) => v,
            Err(resp) => return resp,
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

    let handle = match DocumentsHandle::open(&table_path) {
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

#[utoipa::path(get, path = "/v1/db/tables/{table}/{doc_id}", tag = "DB::Tables",
    params(
        ("table" = String, Path, description = "Table name (alphanumeric, underscore, hyphen; 1-64 chars)"),
        ("doc_id" = String, Path, description = "Document ID")
    ),
    responses(
        (status = 200, body = DocumentResponse),
        (status = 404, body = http::ErrorResponse, description = "Document not found"),
    ))]
/// GET /v1/db/tables/{table}/:id - Get a document
pub async fn handle_get(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path();
    let doc_id = extract_doc_id(path);

    let (_table_name, table_path) = match resolve_table_path(&state, path, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let handle = match DocumentsHandle::open(&table_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    let doc = match handle.get(&doc_id) {
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

#[utoipa::path(post, path = "/v1/db/tables/{table}", tag = "DB::Tables",
    params(("table" = String, Path, description = "Table name (alphanumeric, underscore, hyphen; 1-64 chars)")),
    request_body = CreateDocumentRequest,
    responses(
        (status = 201, body = DocumentResponse),
        (status = 400, body = http::ErrorResponse, description = "Invalid request body"),
    ))]
/// POST /v1/db/tables/{table} - Create a document
pub async fn handle_create(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
    plugin_owner: Option<String>,
) -> Response<BoxBody> {
    let (_table_name, table_path) =
        match resolve_table_path(&state, req.uri().path(), auth.as_ref()) {
            Ok(v) => v,
            Err(resp) => return resp,
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

    let explicit_id = create_req.id.is_some();
    let doc_id = create_req
        .id
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    // Validate explicit IDs: reject path separators and reserved route segments.
    if explicit_id {
        if doc_id.contains('/') {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_argument",
                "document ID must not contain path separators",
            );
        }
        const RESERVED_DOC_IDS: &[&str] = &["search", "insert", "rows", "_meta"];
        if RESERVED_DOC_IDS.contains(&doc_id.as_str()) {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_argument",
                &format!(
                    "document ID '{}' conflicts with a reserved route segment",
                    doc_id
                ),
            );
        }
    }

    let title = create_req.title.unwrap_or_default();
    let content_json = create_req
        .content
        .map(|v| v.to_string())
        .unwrap_or_else(|| "{}".to_string());
    let group_id = create_req
        .group_id
        .or_else(|| auth.as_ref().and_then(|a| a.group_id.clone()));

    let handle = match DocumentsHandle::open(&table_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    // Reject duplicate explicit IDs.
    if explicit_id {
        if let Ok(Some(_)) = handle.get(&doc_id) {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_argument",
                &format!("Document already exists: {}", doc_id),
            );
        }
    }

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

#[utoipa::path(post, path = "/v1/db/tables/{table}/insert", tag = "DB::Tables",
    params(("table" = String, Path, description = "Table name (alphanumeric, underscore, hyphen; 1-64 chars)")),
    request_body = CreateDocumentRequest,
    responses(
        (status = 201, body = DocumentResponse),
        (status = 400, body = http::ErrorResponse, description = "Invalid request body"),
    ))]
/// POST /v1/db/tables/{table}/insert - Alias for insert/create
pub async fn handle_insert(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
    plugin_owner: Option<String>,
) -> Response<BoxBody> {
    handle_create(state, req, auth, plugin_owner).await
}

#[utoipa::path(patch, path = "/v1/db/tables/{table}/{doc_id}", tag = "DB::Tables",
    params(
        ("table" = String, Path, description = "Table name (alphanumeric, underscore, hyphen; 1-64 chars)"),
        ("doc_id" = String, Path, description = "Document ID")
    ),
    request_body = UpdateDocumentRequest,
    responses((status = 200, body = DocumentResponse)))]
/// PATCH /v1/db/tables/{table}/:id - Update a document
pub async fn handle_update(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let doc_id = extract_doc_id(&path);

    let (_table_name, table_path) = match resolve_table_path(&state, &path, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let body = match read_body(req).await {
        Ok(b) => b,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e),
    };

    let update_req: UpdateDocumentRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    let handle = match DocumentsHandle::open(&table_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    let content_json = update_req.content.map(|v| v.to_string());

    if let Err(e) = handle.update(
        &doc_id,
        update_req.title.as_deref(),
        content_json.as_deref(),
        update_req.tags_text.as_deref(),
        update_req.marker.as_deref(),
    ) {
        return document_error_response(e);
    }

    // Fetch the updated document
    let doc = match handle.get(&doc_id) {
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

#[utoipa::path(delete, path = "/v1/db/tables/{table}/{doc_id}", tag = "DB::Tables",
    params(
        ("table" = String, Path, description = "Table name (alphanumeric, underscore, hyphen; 1-64 chars)"),
        ("doc_id" = String, Path, description = "Document ID")
    ),
    responses((status = 204)))]
/// DELETE /v1/db/tables/{table}/:id - Delete a document
pub async fn handle_delete(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path();
    let doc_id = extract_doc_id(path);

    let (_table_name, table_path) = match resolve_table_path(&state, path, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let handle = match DocumentsHandle::open(&table_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    if let Err(e) = handle.delete(&doc_id) {
        return document_error_response(e);
    }

    Response::builder()
        .status(StatusCode::NO_CONTENT)
        .body(Full::new(Bytes::new()).boxed())
        .unwrap()
}

#[utoipa::path(post, path = "/v1/db/tables/{table}/search", tag = "DB::Tables",
    params(("table" = String, Path, description = "Table name (alphanumeric, underscore, hyphen; 1-64 chars)")),
    request_body = DocumentSearchRequest,
    responses((status = 200, body = DocumentSearchResponse)))]
/// POST /v1/db/tables/{table}/search - Search documents
pub async fn handle_search(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let (_table_name, table_path) =
        match resolve_table_path(&state, req.uri().path(), auth.as_ref()) {
            Ok(v) => v,
            Err(resp) => return resp,
        };

    let body = match read_body(req).await {
        Ok(b) => b,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e),
    };

    let search_req: DocumentSearchRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    let handle = match DocumentsHandle::open(&table_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    let limit = search_req.limit.unwrap_or(20);
    let results = match handle.search(&search_req.query, search_req.doc_type.as_deref(), limit) {
        Ok(r) => r,
        Err(e) => return document_error_response(e),
    };

    let response = DocumentSearchResponse {
        data: results
            .into_iter()
            .map(SearchResultResponse::from)
            .collect(),
    };

    json_response(StatusCode::OK, &response)
}

#[utoipa::path(get, path = "/v1/db/tables/{table}/{doc_id}/tags", tag = "DB::Tables",
    params(
        ("table" = String, Path, description = "Table name (alphanumeric, underscore, hyphen; 1-64 chars)"),
        ("doc_id" = String, Path, description = "Document ID")
    ),
    responses((status = 200)))]
/// GET /v1/db/tables/{table}/:id/tags - Get document tags
pub async fn handle_get_tags(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path();
    let doc_id = extract_doc_id_before_tags(path);

    let (_table_name, table_path) = match resolve_table_path(&state, path, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let handle = match DocumentsHandle::open(&table_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    match handle.get(&doc_id) {
        Ok(Some(_)) => {}
        Ok(None) | Err(DocumentError::DocumentNotFound(_)) => {
            return json_error(
                StatusCode::NOT_FOUND,
                "not_found",
                &format!("Document not found: {}", doc_id),
            );
        }
        Err(e) => return document_error_response(e),
    }

    let tags = match handle.get_tags(&doc_id) {
        Ok(t) => t,
        Err(e) => return document_error_response(e),
    };

    json_response(StatusCode::OK, &serde_json::json!({ "tags": tags }))
}

#[utoipa::path(post, path = "/v1/db/tables/{table}/{doc_id}/tags", tag = "DB::Tables",
    params(
        ("table" = String, Path, description = "Table name (alphanumeric, underscore, hyphen; 1-64 chars)"),
        ("doc_id" = String, Path, description = "Document ID")
    ),
    request_body = DocumentTagsRequest,
    responses((status = 200)))]
/// POST /v1/db/tables/{table}/:id/tags - Add tags to document
pub async fn handle_add_tags(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let doc_id = extract_doc_id_before_tags(&path);

    let (_table_name, table_path) = match resolve_table_path(&state, &path, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let body = match read_body(req).await {
        Ok(b) => b,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e),
    };

    let tags_req: DocumentTagsRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    let group_id = auth.as_ref().and_then(|a| a.group_id.clone());

    let handle = match DocumentsHandle::open(&table_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    match handle.get(&doc_id) {
        Ok(Some(_)) => {}
        Ok(None) | Err(DocumentError::DocumentNotFound(_)) => {
            return json_error(
                StatusCode::NOT_FOUND,
                "not_found",
                &format!("Document not found: {}", doc_id),
            );
        }
        Err(e) => return document_error_response(e),
    }

    for tag in &tags_req.tags {
        if let Err(e) = handle.add_tag(&doc_id, tag, group_id.as_deref()) {
            return document_error_response(e);
        }
    }

    // Return updated tags
    let tags = match handle.get_tags(&doc_id) {
        Ok(t) => t,
        Err(e) => return document_error_response(e),
    };

    json_response(StatusCode::OK, &serde_json::json!({ "tags": tags }))
}

#[utoipa::path(delete, path = "/v1/db/tables/{table}/{doc_id}/tags", tag = "DB::Tables",
    params(
        ("table" = String, Path, description = "Table name (alphanumeric, underscore, hyphen; 1-64 chars)"),
        ("doc_id" = String, Path, description = "Document ID")
    ),
    request_body = DocumentTagsRequest,
    responses((status = 200)))]
/// DELETE /v1/db/tables/{table}/:id/tags - Remove tags from document
pub async fn handle_remove_tags(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let doc_id = extract_doc_id_before_tags(&path);

    let (_table_name, table_path) = match resolve_table_path(&state, &path, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let body = match read_body(req).await {
        Ok(b) => b,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e),
    };

    let tags_req: DocumentTagsRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    let group_id = auth.as_ref().and_then(|a| a.group_id.clone());

    let handle = match DocumentsHandle::open(&table_path) {
        Ok(h) => h,
        Err(e) => return document_error_response(e),
    };

    match handle.get(&doc_id) {
        Ok(Some(_)) => {}
        Ok(None) | Err(DocumentError::DocumentNotFound(_)) => {
            return json_error(
                StatusCode::NOT_FOUND,
                "not_found",
                &format!("Document not found: {}", doc_id),
            );
        }
        Err(e) => return document_error_response(e),
    }

    for tag in &tags_req.tags {
        if let Err(e) = handle.remove_tag(&doc_id, tag, group_id.as_deref()) {
            return document_error_response(e);
        }
    }

    // Return updated tags
    let tags = match handle.get_tags(&doc_id) {
        Ok(t) => t,
        Err(e) => return document_error_response(e),
    };

    json_response(StatusCode::OK, &serde_json::json!({ "tags": tags }))
}

// =============================================================================
// Helper functions
// =============================================================================

fn extract_doc_id(path: &str) -> String {
    let Some(stripped) = path.strip_prefix("/v1/db/tables/") else {
        return String::new();
    };
    let mut parts = stripped.split('/');
    let _table_name = parts.next();
    let raw = parts.next().unwrap_or("");
    percent_encoding::percent_decode_str(raw)
        .decode_utf8_lossy()
        .into_owned()
}

fn extract_doc_id_before_tags(path: &str) -> String {
    let Some(stripped) = path.strip_prefix("/v1/db/tables/") else {
        return String::new();
    };
    let without_tags = stripped.strip_suffix("/tags").unwrap_or(stripped);
    let mut parts = without_tags.split('/');
    let _table_name = parts.next();
    let raw = parts.next().unwrap_or("");
    percent_encoding::percent_decode_str(raw)
        .decode_utf8_lossy()
        .into_owned()
}

fn extract_table_name(path: &str) -> Option<&str> {
    let stripped = path.strip_prefix("/v1/db/tables/")?;
    let table_name = stripped.split('/').next().unwrap_or("");
    if table_name.is_empty() {
        None
    } else {
        Some(table_name)
    }
}

/// Reserved names that collide with internal storage directories.
const RESERVED_TABLE_NAMES: &[&str] = &[
    "docs", "blobs", "chat", "tables", "vector", "sql", "kv_state", "_meta",
];

/// Validate that a table name is safe and well-formed.
fn validate_table_name(name: &str) -> Result<(), Response<BoxBody>> {
    if name.is_empty() || name.len() > 64 {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_table_name",
            "table name must be 1-64 characters",
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_table_name",
            "table name may only contain alphanumeric characters, underscores, and hyphens",
        ));
    }
    if RESERVED_TABLE_NAMES.contains(&name) {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "reserved_table_name",
            &format!("table name '{name}' is reserved"),
        ));
    }
    Ok(())
}

/// Map a table name to its on-disk storage path.
///
/// All tables are isolated under `tables/{name}/`.
fn resolve_table_storage_path(
    storage_root: &std::path::Path,
    table_name: &str,
) -> std::path::PathBuf {
    storage_root.join("tables").join(table_name)
}

/// Extract, validate, and resolve a table name + storage path from a request.
fn resolve_table_path(
    state: &AppState,
    uri_path: &str,
    auth: Option<&AuthContext>,
) -> Result<(String, std::path::PathBuf), Response<BoxBody>> {
    let table_name = extract_table_name(uri_path).ok_or_else(|| {
        json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "missing table name in path",
        )
    })?;
    validate_table_name(table_name)?;

    let bucket = state.bucket_path.as_ref().ok_or_else(|| {
        json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "no_storage",
            "Storage not configured",
        )
    })?;

    let storage_root = match auth {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };

    let table_path = resolve_table_storage_path(&storage_root, table_name);
    Ok((table_name.to_string(), table_path))
}

/// Simple query parameter parser.
fn parse_query_param(query: &str, key: &str) -> Option<String> {
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            if k == key {
                // In query strings, '+' represents a space (x-www-form-urlencoded).
                let with_spaces = v.replace('+', " ");
                return Some(
                    percent_encoding::percent_decode_str(&with_spaces)
                        .decode_utf8_lossy()
                        .into_owned(),
                );
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
