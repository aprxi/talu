//! Project management endpoints.
//!
//! Domain-specific API layer over the generic document table plane.
//! Projects are stored as documents with `doc_type = "project"`.

use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::json;
use talu::documents::{DocumentError, DocumentRecord, DocumentsHandle};
use talu::storage::StorageHandle;
use utoipa::ToSchema;

use crate::server::auth_gateway::AuthContext;
use crate::server::http;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;

const DOC_TYPE: &str = "project";

// =============================================================================
// Request/Response types
// =============================================================================

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct CreateProjectRequest {
    /// Display name for the project.
    pub name: String,
    /// Optional description.
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct UpdateProjectRequest {
    /// New display name.
    #[serde(default)]
    pub name: Option<String>,
    /// New description.
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ProjectResponse {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ProjectListResponse {
    pub data: Vec<ProjectResponse>,
    pub has_more: bool,
}

// =============================================================================
// Conversion helpers
// =============================================================================

fn project_from_doc(doc: DocumentRecord) -> ProjectResponse {
    let content: serde_json::Value = if doc.doc_json.is_empty() {
        json!({})
    } else {
        serde_json::from_str(&doc.doc_json).unwrap_or_else(|_| json!({}))
    };

    ProjectResponse {
        id: doc.doc_id,
        name: doc.title,
        description: content
            .get("description")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(String::from),
        created_at: doc.created_at_ms,
        updated_at: doc.updated_at_ms,
    }
}

fn open_documents(
    state: &AppState,
    auth: &Option<AuthContext>,
) -> Result<DocumentsHandle, Response<BoxBody>> {
    let bucket = state.bucket_path.as_ref().ok_or_else(|| {
        json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "storage_unavailable",
            "Storage disabled (running with --no-bucket)",
        )
    })?;

    let path = match auth {
        Some(ctx) => bucket
            .join(&ctx.storage_prefix)
            .join("tables")
            .join("documents"),
        None => bucket.join("tables").join("documents"),
    };

    DocumentsHandle::open(&path).map_err(|e| document_error_response(e))
}

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

// =============================================================================
// Handlers
// =============================================================================

#[utoipa::path(get, path = "/v1/projects", tag = "Projects",
    params(
        ("limit" = Option<u32>, Query, description = "Max items to return (default 50, max 100)"),
        ("search" = Option<String>, Query, description = "Filter by name (case-insensitive substring)"),
    ),
    responses(
        (status = 200, body = ProjectListResponse),
    ))]
/// GET /v1/projects — List projects
pub async fn handle_list(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let handle = match open_documents(&state, &auth) {
        Ok(h) => h,
        Err(resp) => return resp,
    };

    let query_str = req.uri().query().unwrap_or("");
    let limit = parse_query_param(query_str, "limit")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(50)
        .min(100);
    let search = parse_query_param(query_str, "search");

    let docs = if let Some(ref q) = search {
        // Use document search with doc_type filter.
        match handle.search(q, Some(DOC_TYPE), limit) {
            Ok(results) => {
                // Search returns summaries; fetch full records for the response.
                let mut projects = Vec::with_capacity(results.len());
                for sr in &results {
                    if let Ok(Some(doc)) = handle.get(&sr.doc_id) {
                        projects.push(project_from_doc(doc));
                    }
                }
                let has_more = projects.len() >= limit as usize;
                return json_response(
                    StatusCode::OK,
                    &ProjectListResponse {
                        data: projects,
                        has_more,
                    },
                );
            }
            Err(e) => return document_error_response(e),
        }
    } else {
        match handle.list(Some(DOC_TYPE), None, None, None, limit) {
            Ok(d) => d,
            Err(e) => return document_error_response(e),
        }
    };

    let has_more = docs.len() >= limit as usize;

    // Summaries → full records for consistent response shape.
    let mut projects = Vec::with_capacity(docs.len());
    for summary in &docs {
        if let Ok(Some(doc)) = handle.get(&summary.doc_id) {
            projects.push(project_from_doc(doc));
        }
    }

    json_response(
        StatusCode::OK,
        &ProjectListResponse {
            data: projects,
            has_more,
        },
    )
}

#[utoipa::path(post, path = "/v1/projects", tag = "Projects",
    request_body = CreateProjectRequest,
    responses(
        (status = 201, body = ProjectResponse),
        (status = 400, body = http::ErrorResponse, description = "Invalid request"),
    ))]
/// POST /v1/projects — Create a project
pub async fn handle_create(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let handle = match open_documents(&state, &auth) {
        Ok(h) => h,
        Err(resp) => return resp,
    };

    let body = match read_body(req).await {
        Ok(b) => b,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e),
    };

    let create_req: CreateProjectRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    let name = create_req.name.trim();
    if name.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "name is required",
        );
    }

    let project_id = format!("proj_{}", uuid::Uuid::new_v4());
    let content = json!({
        "description": create_req.description.as_deref().unwrap_or(""),
    });

    let group_id = auth.as_ref().and_then(|a| a.group_id.as_deref());

    if let Err(e) = handle.create(
        &project_id,
        DOC_TYPE,
        name,
        &content.to_string(),
        None, // tags_text
        None, // parent_id
        None, // marker
        group_id,
        None, // owner_id
    ) {
        return document_error_response(e);
    }

    let doc = match handle.get(&project_id) {
        Ok(Some(d)) => d,
        Ok(None) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "create_failed",
                "Project created but not found",
            )
        }
        Err(e) => return document_error_response(e),
    };

    json_response(StatusCode::CREATED, &project_from_doc(doc))
}

#[utoipa::path(get, path = "/v1/projects/{id}", tag = "Projects",
    params(
        ("id" = String, Path, description = "Project ID"),
    ),
    responses(
        (status = 200, body = ProjectResponse),
        (status = 404, body = http::ErrorResponse, description = "Project not found"),
    ))]
/// GET /v1/projects/:id — Get a project
pub async fn handle_get(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let project_id = extract_project_id(req.uri().path());
    if project_id.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "missing project ID",
        );
    }

    let handle = match open_documents(&state, &auth) {
        Ok(h) => h,
        Err(resp) => return resp,
    };

    let doc = match handle.get(project_id) {
        Ok(Some(d)) => d,
        Ok(None) => {
            return json_error(
                StatusCode::NOT_FOUND,
                "not_found",
                &format!("Project not found: {}", project_id),
            )
        }
        Err(e) => return document_error_response(e),
    };

    if doc.doc_type != DOC_TYPE {
        return json_error(
            StatusCode::NOT_FOUND,
            "not_found",
            &format!("Project not found: {}", project_id),
        );
    }

    json_response(StatusCode::OK, &project_from_doc(doc))
}

#[utoipa::path(patch, path = "/v1/projects/{id}", tag = "Projects",
    params(
        ("id" = String, Path, description = "Project ID"),
    ),
    request_body = UpdateProjectRequest,
    responses(
        (status = 200, body = ProjectResponse),
        (status = 404, body = http::ErrorResponse, description = "Project not found"),
    ))]
/// PATCH /v1/projects/:id — Update a project
pub async fn handle_update(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let project_id = extract_project_id(&path);
    if project_id.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "missing project ID",
        );
    }

    let handle = match open_documents(&state, &auth) {
        Ok(h) => h,
        Err(resp) => return resp,
    };

    let body = match read_body(req).await {
        Ok(b) => b,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e),
    };

    let update_req: UpdateProjectRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    // Fetch existing to merge content fields.
    let existing = match handle.get(project_id) {
        Ok(Some(d)) => d,
        Ok(None) => {
            return json_error(
                StatusCode::NOT_FOUND,
                "not_found",
                &format!("Project not found: {}", project_id),
            )
        }
        Err(e) => return document_error_response(e),
    };

    if existing.doc_type != DOC_TYPE {
        return json_error(
            StatusCode::NOT_FOUND,
            "not_found",
            &format!("Project not found: {}", project_id),
        );
    }

    // Build updated content JSON, merging with existing.
    let mut content: serde_json::Value = if existing.doc_json.is_empty() {
        json!({})
    } else {
        serde_json::from_str(&existing.doc_json).unwrap_or_else(|_| json!({}))
    };

    if let Some(ref desc) = update_req.description {
        content["description"] = json!(desc);
    }

    let new_title = update_req.name.as_deref();
    let new_content = Some(content.to_string());

    // Detect name change so we can update linked sessions.
    let old_name = existing.title.clone();
    let name_changed = new_title.is_some() && new_title != Some(&old_name);

    if let Err(e) = handle.update(
        project_id,
        new_title,
        new_content.as_deref(),
        None, // tags_text
        None, // marker
    ) {
        return document_error_response(e);
    }

    // If name changed, update project_id on all linked sessions.
    // Sessions store the project name (not UUID) as project_id.
    if name_changed {
        if let (Some(new_name), Ok(storage)) = (new_title, open_storage(&state, &auth)) {
            let mut offset = 0usize;
            loop {
                let batch = match storage.list_sessions_batch(
                    offset,
                    100,
                    None,            // group_id
                    None,            // marker
                    None,            // search
                    None,            // tags_any
                    Some(&old_name), // project_id (old name)
                    false,           // project_id_null
                ) {
                    Ok(b) => b,
                    Err(_) => break,
                };

                if batch.sessions.is_empty() {
                    break;
                }

                for session in &batch.sessions {
                    let update = talu::storage::SessionUpdate {
                        title: None,
                        marker: None,
                        metadata_json: None,
                        source_doc_id: None,
                        project_id: Some(new_name.to_string()),
                        clear_project_id: false,
                    };
                    let _ = storage.update_session(&session.session_id, &update);
                }

                if !batch.has_more {
                    break;
                }
                offset += 100;
            }
        }
    }

    let doc = match handle.get(project_id) {
        Ok(Some(d)) => d,
        Ok(None) => {
            return json_error(
                StatusCode::NOT_FOUND,
                "not_found",
                &format!("Project not found: {}", project_id),
            )
        }
        Err(e) => return document_error_response(e),
    };

    json_response(StatusCode::OK, &project_from_doc(doc))
}

#[utoipa::path(delete, path = "/v1/projects/{id}", tag = "Projects",
    params(
        ("id" = String, Path, description = "Project ID"),
    ),
    responses(
        (status = 204, description = "Project deleted"),
        (status = 404, body = http::ErrorResponse, description = "Project not found"),
    ))]
/// DELETE /v1/projects/:id — Delete a project and unlink sessions
pub async fn handle_delete(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let project_id = extract_project_id(req.uri().path());
    if project_id.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "missing project ID",
        );
    }

    let handle = match open_documents(&state, &auth) {
        Ok(h) => h,
        Err(resp) => return resp,
    };

    // Verify the document exists and is a project.
    match handle.get(project_id) {
        Ok(Some(d)) if d.doc_type == DOC_TYPE => {}
        Ok(_) => {
            return json_error(
                StatusCode::NOT_FOUND,
                "not_found",
                &format!("Project not found: {}", project_id),
            )
        }
        Err(e) => return document_error_response(e),
    }

    // Clear project_id on all linked sessions so they become unassigned.
    if let Ok(storage) = open_storage(&state, &auth) {
        // List sessions with this project_id and clear each one.
        // Use a reasonable batch size; projects with many sessions will
        // need multiple passes but this is fine for the deletion path.
        let mut offset = 0usize;
        loop {
            let batch = match storage.list_sessions_batch(
                offset,
                100,
                None,             // group_id
                None,             // marker
                None,             // search
                None,             // tags_any
                Some(project_id), // project_id
                false,            // project_id_null
            ) {
                Ok(b) => b,
                Err(_) => break,
            };

            if batch.sessions.is_empty() {
                break;
            }

            for session in &batch.sessions {
                let update = talu::storage::SessionUpdate {
                    title: None,
                    marker: None,
                    metadata_json: None,
                    source_doc_id: None,
                    project_id: None,
                    clear_project_id: true,
                };
                let _ = storage.update_session(&session.session_id, &update);
            }

            if !batch.has_more {
                break;
            }
            offset += 100;
        }
    }

    // Delete the project document.
    if let Err(e) = handle.delete(project_id) {
        return document_error_response(e);
    }

    Response::builder()
        .status(StatusCode::NO_CONTENT)
        .body(Full::new(Bytes::new()).boxed())
        .unwrap()
}

// =============================================================================
// Helpers
// =============================================================================

/// Extract project ID from `/v1/projects/{id}`.
fn extract_project_id(path: &str) -> &str {
    let rest = path
        .strip_prefix("/v1/projects/")
        .or_else(|| path.strip_prefix("/projects/"))
        .unwrap_or("");
    rest.split('/').next().unwrap_or("")
}

fn parse_query_param(query: &str, key: &str) -> Option<String> {
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            if k == key {
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
    let body = json!({
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
            &format!("Project not found: {}", id),
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
