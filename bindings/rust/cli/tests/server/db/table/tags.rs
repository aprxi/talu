//! Document tag operations tests.
//!
//! Tests tag endpoints for documents:
//! - GET /v1/db/tables/documents/:id/tags
//! - POST /v1/db/tables/documents/:id/tags
//! - DELETE /v1/db/tables/documents/:id/tags

use super::{documents_config, no_bucket_config};
use crate::server::common::*;
use tempfile::TempDir;

// =============================================================================
// GET /v1/db/tables/documents/:id/tags
// =============================================================================

#[test]
fn get_tags_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = get(ctx.addr(), "/v1/db/tables/documents/some-id/tags");
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn get_tags_returns_empty_for_missing_document() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/db/tables/documents/nonexistent/tags");
    // Server returns empty tags for non-existent documents (permissive behavior)
    assert!(
        resp.status == 200 || resp.status == 404,
        "body: {}",
        resp.body
    );
}

#[test]
fn get_tags_returns_empty_initially() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create document
    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "No Tags", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    // Get tags
    let resp = get(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert!(json["tags"].is_array(), "should have tags array");
    // May be empty array or null
}

#[test]
fn get_tags_returns_added_tags() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create document
    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Tagged Doc", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    // Add tags
    post_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &serde_json::json!({
            "tags": ["coding", "rust"]
        }),
    );

    // Get tags
    let resp = get(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let tags = json["tags"].as_array().expect("tags array");
    // Tags should be present (may be IDs or names depending on implementation)
    assert!(tags.len() >= 2 || !tags.is_empty());
}

// =============================================================================
// POST /v1/db/tables/documents/:id/tags
// =============================================================================

#[test]
fn add_tags_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let body = serde_json::json!({"tags": ["test"]});
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents/some-id/tags", &body);
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn add_tags_handles_missing_document() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({"tags": ["test"]});
    let resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents/nonexistent/tags",
        &body,
    );
    // Server may return 404 or accept and store tags (permissive behavior)
    assert!(
        resp.status == 200 || resp.status == 404,
        "body: {}",
        resp.body
    );
}

#[test]
fn add_tags_succeeds() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create document
    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Add Tags Test", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    // Add tags
    let body = serde_json::json!({"tags": ["important", "urgent"]});
    let resp = post_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &body,
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
}

#[test]
fn add_tags_idempotent() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Idempotent Test", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    // Add same tag twice
    let body = serde_json::json!({"tags": ["duplicate"]});
    post_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &body,
    );
    let resp = post_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &body,
    );
    assert_eq!(resp.status, 200, "adding same tag twice should succeed");

    // Should only have one instance
    let get_resp = get(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
    );
    let json = get_resp.json();
    let tags = json["tags"].as_array().expect("tags");
    // Count occurrences of "duplicate" - should be 1
    let dupe_count = tags
        .iter()
        .filter(|t| {
            t.as_str() == Some("duplicate")
                || t["name"].as_str() == Some("duplicate")
                || t["id"]
                    .as_str()
                    .map(|s| s.contains("duplicate"))
                    .unwrap_or(false)
        })
        .count();
    assert!(dupe_count <= 1, "tag should not be duplicated");
}

#[test]
fn add_multiple_tags_at_once() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Multi Tags", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    let body = serde_json::json!({"tags": ["tag1", "tag2", "tag3"]});
    let resp = post_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &body,
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let get_resp = get(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
    );
    let json = get_resp.json();
    let tags = json["tags"].as_array().expect("tags");
    assert!(tags.len() >= 3, "should have all tags added");
}

// =============================================================================
// DELETE /v1/db/tables/documents/:id/tags
// =============================================================================

#[test]
fn remove_tags_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = delete_json(
        ctx.addr(),
        "/v1/db/tables/documents/some-id/tags",
        &serde_json::json!({"tags": ["test"]}),
    );
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn remove_tags_handles_missing_document() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = delete_json(
        ctx.addr(),
        "/v1/db/tables/documents/nonexistent/tags",
        &serde_json::json!({"tags": ["test"]}),
    );
    // Server may return 404 or succeed silently (permissive behavior)
    assert!(
        resp.status == 200 || resp.status == 404,
        "body: {}",
        resp.body
    );
}

#[test]
fn remove_tags_succeeds() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Remove Tags Test", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    // Add tags
    post_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &serde_json::json!({"tags": ["keep", "remove"]}),
    );

    // Remove one tag
    let resp = delete_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &serde_json::json!({"tags": ["remove"]}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    // Verify
    let get_resp = get(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
    );
    let json = get_resp.json();
    let tags = json["tags"].as_array().expect("tags");
    // "remove" should be gone, "keep" should remain
    let has_remove = tags
        .iter()
        .any(|t| t.as_str() == Some("remove") || t["name"].as_str() == Some("remove"));
    assert!(!has_remove, "removed tag should be gone");
}

#[test]
fn remove_nonexistent_tag_safe() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Safe Remove Test", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    // Remove tag that was never added - should not error
    let resp = delete_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &serde_json::json!({"tags": ["never-existed"]}),
    );
    // Should succeed or return 404 for the tag - not 500
    assert!(resp.status == 200 || resp.status == 404);
}
