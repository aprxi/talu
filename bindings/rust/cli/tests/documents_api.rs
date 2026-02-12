//! Documents API integration tests.
//!
//! Tests for the Documents HTTP API endpoints.
//! Uses in-process hyper connections (no TCP socket) for speed and determinism.
//!
//! Run: `LD_LIBRARY_PATH=zig-out/lib cargo test --test documents_api`

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::client::conn::http1 as client_http1;
use hyper::server::conn::http1 as server_http1;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;
use serde_json::Value;
use tempfile::TempDir;
use tokio::sync::Mutex;

use talu_cli::server::http::Router;
use talu_cli::server::state::{AppState, BackendState};

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

async fn send_request(router: &Router, req: Request<Full<Bytes>>) -> Response<Incoming> {
    let (client_io, server_io) = tokio::io::duplex(1024 * 1024);

    let service = TowerToHyperService::new(router.clone());
    tokio::spawn(async move {
        let _ = server_http1::Builder::new()
            .serve_connection(TokioIo::new(server_io), service)
            .await;
    });

    let (mut sender, conn) = client_http1::handshake(TokioIo::new(client_io))
        .await
        .expect("client handshake failed");

    tokio::spawn(async move {
        let _ = conn.await;
    });

    sender.send_request(req).await.expect("send_request failed")
}

async fn body_json(resp: Response<Incoming>) -> (StatusCode, Value) {
    let status = resp.status();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap_or_else(|e| {
        panic!(
            "invalid JSON in response body: {e}\nbody: {}",
            String::from_utf8_lossy(&body)
        )
    });
    (status, json)
}

async fn body_bytes(resp: Response<Incoming>) -> (StatusCode, Bytes) {
    let status = resp.status();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    (status, body)
}

/// Build a Router with a temporary storage bucket.
fn build_app_with_storage(temp_dir: &TempDir) -> Router {
    let state = AppState {
        backend: Arc::new(Mutex::new(BackendState {
            backend: None,
            current_model: None,
        })),
        configured_model: None,
        response_store: Mutex::new(HashMap::new()),
        gateway_secret: None,
        tenant_registry: None,
        bucket_path: Some(temp_dir.path().to_path_buf()),
        html_dir: None,
        plugin_tokens: Mutex::new(HashMap::new()),
    };

    Router::new(Arc::new(state))
}

/// Build a Router without storage (for testing error cases).
fn build_app_no_storage() -> Router {
    let state = AppState {
        backend: Arc::new(Mutex::new(BackendState {
            backend: None,
            current_model: None,
        })),
        configured_model: None,
        response_store: Mutex::new(HashMap::new()),
        gateway_secret: None,
        tenant_registry: None,
        bucket_path: None,
        html_dir: None,
        plugin_tokens: Mutex::new(HashMap::new()),
    };

    Router::new(Arc::new(state))
}

fn get(uri: &str) -> Request<Full<Bytes>> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .body(Full::new(Bytes::new()))
        .unwrap()
}

fn post_json(uri: &str, body: &Value) -> Request<Full<Bytes>> {
    let bytes = serde_json::to_vec(body).unwrap();
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(bytes)))
        .unwrap()
}

fn patch_json(uri: &str, body: &Value) -> Request<Full<Bytes>> {
    let bytes = serde_json::to_vec(body).unwrap();
    Request::builder()
        .method("PATCH")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(bytes)))
        .unwrap()
}

fn delete(uri: &str) -> Request<Full<Bytes>> {
    Request::builder()
        .method("DELETE")
        .uri(uri)
        .body(Full::new(Bytes::new()))
        .unwrap()
}

fn delete_json(uri: &str, body: &Value) -> Request<Full<Bytes>> {
    let bytes = serde_json::to_vec(body).unwrap();
    Request::builder()
        .method("DELETE")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(bytes)))
        .unwrap()
}

// ===========================================================================
// D1. Storage requirement
// ===========================================================================

#[tokio::test]
async fn test_documents_requires_storage() {
    let app = build_app_no_storage();
    let resp = send_request(&app, get("/v1/documents")).await;
    let (status, json) = body_json(resp).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    let error = json.get("error").expect("should have error");
    assert_eq!(
        error.get("code").and_then(|v| v.as_str()),
        Some("no_storage")
    );
}

// ===========================================================================
// D2. Create document
// ===========================================================================

#[tokio::test]
async fn test_create_document_minimal() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    let body = serde_json::json!({
        "type": "note",
        "title": "Test Note",
        "content": "This is test content."
    });
    let resp = send_request(&app, post_json("/v1/documents", &body)).await;
    let (status, json) = body_json(resp).await;

    assert_eq!(
        status,
        StatusCode::CREATED,
        "create should return 201: {json}"
    );
    assert!(
        json.get("id").and_then(|v| v.as_str()).is_some(),
        "should have id"
    );
    assert_eq!(json.get("type").and_then(|v| v.as_str()), Some("note"));
    assert_eq!(
        json.get("title").and_then(|v| v.as_str()),
        Some("Test Note")
    );
}

#[tokio::test]
async fn test_create_document_with_all_fields() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    let body = serde_json::json!({
        "type": "article",
        "title": "Full Article",
        "content": "Article content here.",
        "group_id": "group-123",
        "owner_id": "user-456",
        "tags": ["tech", "news"],
        "metadata": {
            "author": "Test Author",
            "priority": 5
        }
    });
    let resp = send_request(&app, post_json("/v1/documents", &body)).await;
    let (status, json) = body_json(resp).await;

    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(json.get("type").and_then(|v| v.as_str()), Some("article"));
    assert_eq!(
        json.get("group_id").and_then(|v| v.as_str()),
        Some("group-123")
    );
    assert_eq!(
        json.get("owner_id").and_then(|v| v.as_str()),
        Some("user-456")
    );
}

#[tokio::test]
async fn test_create_document_missing_doc_type() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    let body = serde_json::json!({
        "title": "No Type",
        "content": "Content without doc_type"
    });
    let resp = send_request(&app, post_json("/v1/documents", &body)).await;
    let (status, json) = body_json(resp).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    let error = json.get("error").expect("should have error");
    assert_eq!(
        error.get("code").and_then(|v| v.as_str()),
        Some("invalid_json")
    );
}

// ===========================================================================
// D3. Get document
// ===========================================================================

#[tokio::test]
async fn test_get_document() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    // Create a document first
    let create_body = serde_json::json!({
        "type": "memo",
        "title": "Get Test",
        "content": "Content to retrieve"
    });
    let create_resp = send_request(&app, post_json("/v1/documents", &create_body)).await;
    let (_, create_json) = body_json(create_resp).await;
    let doc_id = create_json.get("id").and_then(|v| v.as_str()).unwrap();

    // Get the document
    let get_resp = send_request(&app, get(&format!("/v1/documents/{}", doc_id))).await;
    let (status, json) = body_json(get_resp).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json.get("id").and_then(|v| v.as_str()), Some(doc_id));
    assert_eq!(json.get("title").and_then(|v| v.as_str()), Some("Get Test"));
}

#[tokio::test]
async fn test_get_document_not_found() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    let resp = send_request(&app, get("/v1/documents/nonexistent-id")).await;
    let (status, json) = body_json(resp).await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    let error = json.get("error").expect("should have error");
    assert_eq!(
        error.get("code").and_then(|v| v.as_str()),
        Some("not_found")
    );
}

// ===========================================================================
// D4. Update document
// ===========================================================================

#[tokio::test]
async fn test_update_document() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    // Create a document
    let create_body = serde_json::json!({
        "type": "draft",
        "title": "Original Title",
        "content": "Original content"
    });
    let create_resp = send_request(&app, post_json("/v1/documents", &create_body)).await;
    let (_, create_json) = body_json(create_resp).await;
    let doc_id = create_json.get("id").and_then(|v| v.as_str()).unwrap();

    // Update the document
    let update_body = serde_json::json!({
        "title": "Updated Title",
        "content": "Updated content"
    });
    let update_resp = send_request(
        &app,
        patch_json(&format!("/v1/documents/{}", doc_id), &update_body),
    )
    .await;
    let (status, json) = body_json(update_resp).await;

    assert_eq!(status, StatusCode::OK, "update should succeed: {json}");
    assert_eq!(
        json.get("title").and_then(|v| v.as_str()),
        Some("Updated Title")
    );
}

#[tokio::test]
async fn test_update_document_not_found() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    let update_body = serde_json::json!({
        "title": "Won't Work"
    });
    let resp = send_request(
        &app,
        patch_json("/v1/documents/nonexistent-id", &update_body),
    )
    .await;
    let (status, _) = body_json(resp).await;

    assert_eq!(status, StatusCode::NOT_FOUND);
}

// ===========================================================================
// D5. Delete document
// ===========================================================================

#[tokio::test]
async fn test_delete_document() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    // Create a document
    let create_body = serde_json::json!({
        "type": "temp",
        "title": "To Delete",
        "content": "This will be deleted"
    });
    let create_resp = send_request(&app, post_json("/v1/documents", &create_body)).await;
    let (_, create_json) = body_json(create_resp).await;
    let doc_id = create_json.get("id").and_then(|v| v.as_str()).unwrap();

    // Delete the document
    let delete_resp = send_request(&app, delete(&format!("/v1/documents/{}", doc_id))).await;
    let (status, _) = body_bytes(delete_resp).await;
    assert_eq!(status, StatusCode::NO_CONTENT);

    // Verify it's gone
    let get_resp = send_request(&app, get(&format!("/v1/documents/{}", doc_id))).await;
    let (get_status, _) = body_json(get_resp).await;
    assert_eq!(get_status, StatusCode::NOT_FOUND);
}

// ===========================================================================
// D6. List documents
// ===========================================================================

#[tokio::test]
async fn test_list_documents_empty() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    let resp = send_request(&app, get("/v1/documents")).await;
    let (status, json) = body_json(resp).await;

    assert_eq!(status, StatusCode::OK);
    let data = json
        .get("data")
        .and_then(|v| v.as_array())
        .expect("should have data array");
    assert!(data.is_empty(), "should be empty initially");
    assert_eq!(json.get("has_more").and_then(|v| v.as_bool()), Some(false));
}

#[tokio::test]
async fn test_list_documents_with_items() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    // Create some documents
    for i in 0..3 {
        let body = serde_json::json!({
            "type": "item",
            "title": format!("Item {}", i),
            "content": format!("Content {}", i)
        });
        send_request(&app, post_json("/v1/documents", &body)).await;
    }

    let resp = send_request(&app, get("/v1/documents")).await;
    let (status, json) = body_json(resp).await;

    assert_eq!(status, StatusCode::OK);
    let data = json
        .get("data")
        .and_then(|v| v.as_array())
        .expect("should have data");
    assert_eq!(data.len(), 3, "should have 3 documents");
}

#[tokio::test]
async fn test_list_documents_filter_by_type() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    // Create documents of different types
    let body1 = serde_json::json!({"type": "note", "title": "Note 1", "content": "c"});
    let body2 = serde_json::json!({"type": "article", "title": "Article 1", "content": "c"});
    let body3 = serde_json::json!({"type": "note", "title": "Note 2", "content": "c"});
    send_request(&app, post_json("/v1/documents", &body1)).await;
    send_request(&app, post_json("/v1/documents", &body2)).await;
    send_request(&app, post_json("/v1/documents", &body3)).await;

    // Filter by type (API uses 'type' not 'doc_type')
    let resp = send_request(&app, get("/v1/documents?type=note")).await;
    let (status, json) = body_json(resp).await;

    assert_eq!(status, StatusCode::OK);
    let data = json
        .get("data")
        .and_then(|v| v.as_array())
        .expect("should have data");
    assert_eq!(data.len(), 2, "should have 2 notes");
    for item in data {
        assert_eq!(item.get("type").and_then(|v| v.as_str()), Some("note"));
    }
}

#[tokio::test]
async fn test_list_documents_pagination() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    // Create 5 documents
    for i in 0..5 {
        let body = serde_json::json!({
            "type": "page",
            "title": format!("Page {}", i),
            "content": format!("Content {}", i)
        });
        send_request(&app, post_json("/v1/documents", &body)).await;
    }

    // Get first page with limit 2
    let resp = send_request(&app, get("/v1/documents?limit=2")).await;
    let (status, json) = body_json(resp).await;

    assert_eq!(status, StatusCode::OK);
    let data = json
        .get("data")
        .and_then(|v| v.as_array())
        .expect("should have data");
    assert_eq!(data.len(), 2, "should have 2 items in first page");
    assert_eq!(json.get("has_more").and_then(|v| v.as_bool()), Some(true));
}

// ===========================================================================
// D7. Search documents
// ===========================================================================

#[tokio::test]
async fn test_search_documents() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    // Create documents with searchable content
    let body1 = serde_json::json!({
        "type": "article",
        "title": "Rust Programming",
        "content": "Learn about Rust programming language"
    });
    let body2 = serde_json::json!({
        "type": "article",
        "title": "Python Guide",
        "content": "Python is a great language"
    });
    send_request(&app, post_json("/v1/documents", &body1)).await;
    send_request(&app, post_json("/v1/documents", &body2)).await;

    // Search for "Rust"
    let search_body = serde_json::json!({
        "query": "Rust"
    });
    let resp = send_request(&app, post_json("/v1/documents/search", &search_body)).await;
    let (status, json) = body_json(resp).await;

    assert_eq!(status, StatusCode::OK, "search should succeed: {json}");
    let data = json
        .get("data")
        .and_then(|v| v.as_array())
        .expect("should have data");
    assert!(!data.is_empty(), "should find at least one result");
}

#[tokio::test]
async fn test_search_documents_with_type_filter() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    // Create documents of different types
    let body1 = serde_json::json!({
        "type": "note",
        "title": "Programming Note",
        "content": "Notes about programming"
    });
    let body2 = serde_json::json!({
        "type": "article",
        "title": "Programming Article",
        "content": "Article about programming"
    });
    send_request(&app, post_json("/v1/documents", &body1)).await;
    send_request(&app, post_json("/v1/documents", &body2)).await;

    // Search with type filter
    let search_body = serde_json::json!({
        "query": "programming",
        "type": "note"
    });
    let resp = send_request(&app, post_json("/v1/documents/search", &search_body)).await;
    let (status, json) = body_json(resp).await;

    assert_eq!(status, StatusCode::OK);
    let data = json
        .get("data")
        .and_then(|v| v.as_array())
        .expect("should have data");
    // All results should be notes
    for result in data {
        assert_eq!(result.get("type").and_then(|v| v.as_str()), Some("note"));
    }
}

// ===========================================================================
// D8. Tag operations
// ===========================================================================

#[tokio::test]
async fn test_add_tags() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    // Create a document
    let create_body = serde_json::json!({
        "type": "taggable",
        "title": "Tag Test",
        "content": "Content"
    });
    let create_resp = send_request(&app, post_json("/v1/documents", &create_body)).await;
    let (_, create_json) = body_json(create_resp).await;
    let doc_id = create_json.get("id").and_then(|v| v.as_str()).unwrap();

    // Add tags
    let tags_body = serde_json::json!({
        "tags": ["important", "urgent"]
    });
    let add_resp = send_request(
        &app,
        post_json(&format!("/v1/documents/{}/tags", doc_id), &tags_body),
    )
    .await;
    let (status, body) = body_bytes(add_resp).await;
    assert!(
        status.is_success(),
        "add tags should succeed: status={} body={}",
        status,
        String::from_utf8_lossy(&body)
    );

    // Get tags
    let get_resp = send_request(&app, get(&format!("/v1/documents/{}/tags", doc_id))).await;
    let (get_status, json) = body_json(get_resp).await;

    assert_eq!(get_status, StatusCode::OK);
    let tags = json
        .get("tags")
        .and_then(|v| v.as_array())
        .expect("should have tags");
    assert!(tags.iter().any(|t| t.as_str() == Some("important")));
    assert!(tags.iter().any(|t| t.as_str() == Some("urgent")));
}

#[tokio::test]
async fn test_remove_tags() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    // Create a document
    let create_body = serde_json::json!({
        "type": "taggable",
        "title": "Remove Tag Test",
        "content": "Content"
    });
    let create_resp = send_request(&app, post_json("/v1/documents", &create_body)).await;
    let (_, create_json) = body_json(create_resp).await;
    let doc_id = create_json.get("id").and_then(|v| v.as_str()).unwrap();

    // Add tags first
    let add_body = serde_json::json!({
        "tags": ["keep", "remove"]
    });
    let add_resp = send_request(
        &app,
        post_json(&format!("/v1/documents/{}/tags", doc_id), &add_body),
    )
    .await;
    let (add_status, _) = body_bytes(add_resp).await;
    assert!(add_status.is_success(), "add tags should succeed");

    // Remove a tag
    let remove_body = serde_json::json!({
        "tags": ["remove"]
    });
    let remove_resp = send_request(
        &app,
        delete_json(&format!("/v1/documents/{}/tags", doc_id), &remove_body),
    )
    .await;
    let (status, body) = body_bytes(remove_resp).await;
    assert!(
        status.is_success(),
        "remove tags should succeed: status={} body={}",
        status,
        String::from_utf8_lossy(&body)
    );

    // Verify the tag was removed
    let get_resp = send_request(&app, get(&format!("/v1/documents/{}/tags", doc_id))).await;
    let (_, json) = body_json(get_resp).await;
    let tags = json
        .get("tags")
        .and_then(|v| v.as_array())
        .expect("should have tags");
    assert!(
        tags.iter().any(|t| t.as_str() == Some("keep")),
        "should have 'keep' tag, got: {:?}",
        tags
    );
    assert!(
        !tags.iter().any(|t| t.as_str() == Some("remove")),
        "should not have 'remove' tag, got: {:?}",
        tags
    );
}

#[tokio::test]
async fn test_get_tags_empty() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    // Create a document without tags
    let create_body = serde_json::json!({
        "type": "untagged",
        "title": "No Tags",
        "content": "Content"
    });
    let create_resp = send_request(&app, post_json("/v1/documents", &create_body)).await;
    let (_, create_json) = body_json(create_resp).await;
    let doc_id = create_json.get("id").and_then(|v| v.as_str()).unwrap();

    // Get tags
    let get_resp = send_request(&app, get(&format!("/v1/documents/{}/tags", doc_id))).await;
    let (status, json) = body_json(get_resp).await;

    assert_eq!(status, StatusCode::OK);
    let tags = json
        .get("tags")
        .and_then(|v| v.as_array())
        .expect("should have tags array");
    assert!(tags.is_empty(), "should have no tags");
}

// ===========================================================================
// D9. URL prefix variations
// ===========================================================================

#[tokio::test]
async fn test_documents_with_v1_prefix() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    // Both /v1/documents and /documents should work
    let resp_v1 = send_request(&app, get("/v1/documents")).await;
    let resp_bare = send_request(&app, get("/documents")).await;

    let (s1, _) = body_json(resp_v1).await;
    let (s2, _) = body_json(resp_bare).await;

    assert_eq!(s1, StatusCode::OK);
    assert_eq!(s2, StatusCode::OK);
}

// ===========================================================================
// D10. Error response format
// ===========================================================================

#[tokio::test]
async fn test_error_json_structure() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    let resp = send_request(&app, get("/v1/documents/nonexistent")).await;
    let (status, json) = body_json(resp).await;

    assert!(status.is_client_error());
    let error = json.get("error").expect("should have error object");
    assert!(
        error.get("code").and_then(|v| v.as_str()).is_some(),
        "should have code"
    );
    assert!(
        error.get("message").and_then(|v| v.as_str()).is_some(),
        "should have message"
    );
}

#[tokio::test]
async fn test_error_content_type() {
    let temp_dir = TempDir::new().unwrap();
    let app = build_app_with_storage(&temp_dir);

    let resp = send_request(&app, get("/v1/documents/nonexistent")).await;
    let ct = resp
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap().to_string());

    assert!(
        ct.as_deref()
            .map_or(false, |c| c.contains("application/json")),
        "error Content-Type should be application/json: {:?}",
        ct
    );
}
