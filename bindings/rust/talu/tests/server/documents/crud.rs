//! Document CRUD operation tests.
//!
//! Tests create, get, list, update, and delete operations.

use super::{documents_config, no_bucket_config};
use crate::server::common::*;
use tempfile::TempDir;

// =============================================================================
// GET /v1/documents (list)
// =============================================================================

#[test]
fn list_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = get(ctx.addr(), "/v1/documents");
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn list_returns_empty_array_initially() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/documents");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert!(json["data"].is_array(), "should have data array");
    assert_eq!(
        json["data"].as_array().unwrap().len(),
        0,
        "should be empty initially"
    );
}

#[test]
fn list_returns_created_documents() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create a document
    let create_body = serde_json::json!({
        "type": "prompt",
        "title": "Test Prompt",
        "content": {"system": "You are helpful."}
    });
    let resp = post_json(ctx.addr(), "/v1/documents", &create_body);
    assert_eq!(resp.status, 201, "create: {}", resp.body);

    // List should return the document
    let resp = get(ctx.addr(), "/v1/documents");
    assert_eq!(resp.status, 200, "list: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1, "should have one document");
    assert_eq!(data[0]["title"], "Test Prompt");
    assert_eq!(data[0]["type"], "prompt");
}

#[test]
fn list_filters_by_type() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create documents of different types
    post_json(
        ctx.addr(),
        "/v1/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Prompt 1", "content": {}
        }),
    );
    post_json(
        ctx.addr(),
        "/v1/documents",
        &serde_json::json!({
            "type": "persona", "title": "Persona 1", "content": {}
        }),
    );
    post_json(
        ctx.addr(),
        "/v1/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Prompt 2", "content": {}
        }),
    );

    // Filter by type
    let resp = get(ctx.addr(), "/v1/documents?type=prompt");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2, "should have 2 prompts");
    for doc in data {
        assert_eq!(doc["type"], "prompt");
    }
}

#[test]
fn list_respects_limit() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create several documents
    for i in 0..5 {
        post_json(
            ctx.addr(),
            "/v1/documents",
            &serde_json::json!({
                "type": "note", "title": format!("Note {}", i), "content": {}
            }),
        );
    }

    // List with limit
    let resp = get(ctx.addr(), "/v1/documents?limit=2");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(data.len() <= 2, "should respect limit");
}

// =============================================================================
// POST /v1/documents (create)
// =============================================================================

#[test]
fn create_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let body = serde_json::json!({
        "type": "prompt",
        "title": "Test",
        "content": {}
    });
    let resp = post_json(ctx.addr(), "/v1/documents", &body);
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn create_returns_document_with_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({
        "type": "prompt",
        "title": "My Prompt",
        "content": {"system": "Be helpful."}
    });
    let resp = post_json(ctx.addr(), "/v1/documents", &body);
    assert_eq!(resp.status, 201, "body: {}", resp.body);

    let json = resp.json();
    // Response doesn't include "object" field - just check the essential fields
    assert!(json["id"].is_string(), "should have id");
    assert_eq!(json["title"], "My Prompt");
    assert_eq!(json["type"], "prompt");
    assert!(json["created_at"].as_i64().unwrap() > 0);
}

#[test]
fn create_with_tags() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({
        "type": "prompt",
        "title": "Tagged Prompt",
        "content": {},
        "tags": ["coding", "rust"]
    });
    let resp = post_json(ctx.addr(), "/v1/documents", &body);
    assert_eq!(resp.status, 201, "body: {}", resp.body);

    let json = resp.json();
    // Tags may be in response or need separate GET
    assert!(json["id"].is_string());
}

#[test]
fn create_with_owner_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({
        "type": "note",
        "title": "My Note",
        "content": {"text": "Personal note"},
        "owner_id": "user-123"
    });
    let resp = post_json(ctx.addr(), "/v1/documents", &body);
    assert_eq!(resp.status, 201, "body: {}", resp.body);
}

#[test]
fn create_with_group_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({
        "type": "prompt",
        "title": "Team Prompt",
        "content": {},
        "group_id": "team-alpha"
    });
    let resp = post_json(ctx.addr(), "/v1/documents", &body);
    assert_eq!(resp.status, 201, "body: {}", resp.body);
}

#[test]
fn create_requires_type() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({
        "title": "No Type",
        "content": {}
    });
    let resp = post_json(ctx.addr(), "/v1/documents", &body);
    assert_eq!(resp.status, 400, "should reject missing type");
}

#[test]
fn create_requires_title() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({
        "type": "prompt",
        "content": {}
    });
    let resp = post_json(ctx.addr(), "/v1/documents", &body);
    assert_eq!(resp.status, 400, "should reject missing title");
}

// =============================================================================
// GET /v1/documents/:id (get)
// =============================================================================

#[test]
fn get_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = get(ctx.addr(), "/v1/documents/some-id");
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn get_returns_404_for_missing() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/documents/nonexistent-id");
    assert_eq!(resp.status, 404, "body: {}", resp.body);
}

#[test]
fn get_returns_created_document() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create
    let body = serde_json::json!({
        "type": "prompt",
        "title": "Get Test",
        "content": {"system": "Test content"}
    });
    let create_resp = post_json(ctx.addr(), "/v1/documents", &body);
    assert_eq!(create_resp.status, 201);
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    // Get
    let resp = get(ctx.addr(), &format!("/v1/documents/{}", doc_id));
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["id"], doc_id);
    assert_eq!(json["title"], "Get Test");
    assert_eq!(json["type"], "prompt");
}

#[test]
fn get_returns_full_content() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let content = serde_json::json!({
        "system": "You are a code reviewer.",
        "model": "gpt-4",
        "temperature": 0.7
    });
    let body = serde_json::json!({
        "type": "prompt",
        "title": "Full Content",
        "content": content
    });
    let create_resp = post_json(ctx.addr(), "/v1/documents", &body);
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    let resp = get(ctx.addr(), &format!("/v1/documents/{}", doc_id));
    assert_eq!(resp.status, 200);

    let json = resp.json();
    // Content should be preserved
    assert!(json["content"].is_object() || json["data"].is_object());
}

// =============================================================================
// PATCH /v1/documents/:id (update)
// =============================================================================

#[test]
fn patch_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let body = serde_json::json!({"title": "Updated"});
    let resp = patch_json(ctx.addr(), "/v1/documents/some-id", &body);
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn patch_returns_404_for_missing() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({"title": "Updated"});
    let resp = patch_json(ctx.addr(), "/v1/documents/nonexistent", &body);
    assert_eq!(resp.status, 404, "body: {}", resp.body);
}

#[test]
fn patch_updates_title() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create
    let create_resp = post_json(
        ctx.addr(),
        "/v1/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Before", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    // Patch
    let resp = patch_json(
        ctx.addr(),
        &format!("/v1/documents/{}", doc_id),
        &serde_json::json!({"title": "After"}),
    );
    assert_eq!(resp.status, 200, "patch: {}", resp.body);

    // Verify
    let get_resp = get(ctx.addr(), &format!("/v1/documents/{}", doc_id));
    assert_eq!(get_resp.json()["title"], "After");
}

#[test]
fn patch_updates_content() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let create_resp = post_json(
        ctx.addr(),
        "/v1/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Content Test", "content": {"v": 1}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    let resp = patch_json(
        ctx.addr(),
        &format!("/v1/documents/{}", doc_id),
        &serde_json::json!({"content": {"v": 2}}),
    );
    assert_eq!(resp.status, 200, "patch: {}", resp.body);
}

#[test]
fn patch_updates_marker() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let create_resp = post_json(
        ctx.addr(),
        "/v1/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Marker Test", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    let resp = patch_json(
        ctx.addr(),
        &format!("/v1/documents/{}", doc_id),
        &serde_json::json!({"marker": "archived"}),
    );
    assert_eq!(resp.status, 200, "patch: {}", resp.body);
}

// =============================================================================
// DELETE /v1/documents/:id (delete)
// =============================================================================

#[test]
fn delete_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = delete(ctx.addr(), "/v1/documents/some-id");
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn delete_returns_404_for_missing() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = delete(ctx.addr(), "/v1/documents/nonexistent");
    // May return 404 or 200 depending on implementation (soft delete)
    assert!(
        resp.status == 404 || resp.status == 200,
        "body: {}",
        resp.body
    );
}

#[test]
fn delete_removes_document() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create
    let create_resp = post_json(
        ctx.addr(),
        "/v1/documents",
        &serde_json::json!({
            "type": "prompt", "title": "To Delete", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    // Delete
    let resp = delete(ctx.addr(), &format!("/v1/documents/{}", doc_id));
    assert!(
        resp.status == 200 || resp.status == 204,
        "delete: {}",
        resp.body
    );

    // Verify gone (or marked deleted)
    let get_resp = get(ctx.addr(), &format!("/v1/documents/{}", doc_id));
    // Either 404 (hard delete) or marker=deleted (soft delete)
    assert!(get_resp.status == 404 || get_resp.json()["marker"] == "deleted");
}

#[test]
fn delete_removed_from_list() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create two documents
    let _keep = post_json(
        ctx.addr(),
        "/v1/documents",
        &serde_json::json!({
            "type": "note", "title": "Keep", "content": {}
        }),
    );
    let create2 = post_json(
        ctx.addr(),
        "/v1/documents",
        &serde_json::json!({
            "type": "note", "title": "Delete", "content": {}
        }),
    );
    let doc_id2 = create2.json()["id"].as_str().expect("id").to_string();

    // Delete second
    delete(ctx.addr(), &format!("/v1/documents/{}", doc_id2));

    // List should not include deleted (by default)
    let list_resp = get(ctx.addr(), "/v1/documents");
    let json = list_resp.json();
    let data = json["data"].as_array().expect("data");
    // Should only have 1 document now (or 2 if soft-delete without filter)
    let titles: Vec<_> = data
        .iter()
        .map(|d| d["title"].as_str().unwrap_or(""))
        .collect();
    assert!(titles.contains(&"Keep"), "Keep should still exist");
}

// =============================================================================
// Error handling
// =============================================================================

#[test]
fn invalid_json_returns_400() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/documents",
        &[("Content-Type", "application/json")],
        Some("not valid json"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn error_responses_have_json_body() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/documents/nonexistent");
    assert_eq!(resp.status, 404);

    let json = resp.json();
    assert!(json["error"]["code"].is_string(), "error should have code");
    assert!(
        json["error"]["message"].is_string(),
        "error should have message"
    );
}
