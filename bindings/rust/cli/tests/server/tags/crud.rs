//! Integration tests for `/v1/tags` CRUD endpoints.

use super::tags_config;
use crate::server::common::*;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// POST /v1/tags - Create tag
// ---------------------------------------------------------------------------

/// Create a tag with name only.
#[test]
fn create_tag_minimal() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(tags_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/tags",
        &serde_json::json!({
            "name": "work"
        }),
    );
    assert_eq!(resp.status, 201, "body: {}", resp.body);

    let json = resp.json();
    assert!(json["id"].is_string(), "id should be present");
    assert_eq!(json["name"], "work");
    assert!(json["color"].is_null(), "color should be null");
    assert!(json["description"].is_null(), "description should be null");
    assert!(
        json["created_at"].is_number(),
        "created_at should be present"
    );
    assert!(
        json["updated_at"].is_number(),
        "updated_at should be present"
    );
}

/// Create a tag with all fields.
#[test]
fn create_tag_full() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(tags_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/tags",
        &serde_json::json!({
            "name": "urgent",
            "color": "#ff6b6b",
            "description": "High priority items"
        }),
    );
    assert_eq!(resp.status, 201, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["name"], "urgent");
    assert_eq!(json["color"], "#ff6b6b");
    assert_eq!(json["description"], "High priority items");
}

/// Create tag without name fails.
#[test]
fn create_tag_missing_name() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(tags_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/tags",
        &serde_json::json!({
            "color": "#ff0000"
        }),
    );
    assert_eq!(resp.status, 400, "should fail without name");
}

// ---------------------------------------------------------------------------
// GET /v1/tags - List tags
// ---------------------------------------------------------------------------

/// List tags returns empty when no tags exist.
#[test]
fn list_tags_empty() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(tags_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/tags");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(data.is_empty());
}

/// List tags returns created tags.
#[test]
fn list_tags_returns_all() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Create two tags
    let resp1 = post_json(ctx.addr(), "/v1/tags", &serde_json::json!({"name": "work"}));
    assert_eq!(resp1.status, 201);

    let resp2 = post_json(
        ctx.addr(),
        "/v1/tags",
        &serde_json::json!({"name": "personal"}),
    );
    assert_eq!(resp2.status, 201);

    // List all tags
    let resp = get(ctx.addr(), "/v1/tags");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2);

    let names: Vec<&str> = data.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"work"));
    assert!(names.contains(&"personal"));
}

// ---------------------------------------------------------------------------
// GET /v1/tags/:id - Get tag
// ---------------------------------------------------------------------------

/// Get a tag by ID.
#[test]
fn get_tag_by_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Create a tag
    let create_resp = post_json(
        ctx.addr(),
        "/v1/tags",
        &serde_json::json!({
            "name": "project-x",
            "color": "#9b59b6"
        }),
    );
    assert_eq!(create_resp.status, 201);
    let create_json = create_resp.json();
    let tag_id = create_json["id"].as_str().unwrap();

    // Get by ID
    let resp = get(ctx.addr(), &format!("/v1/tags/{}", tag_id));
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["id"].as_str().unwrap(), tag_id);
    assert_eq!(json["name"], "project-x");
    assert_eq!(json["color"], "#9b59b6");
}

/// Get tag by ID includes usage statistics with zero counts when no sessions/documents.
#[test]
fn get_tag_includes_usage_stats() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Create a tag
    let create_resp = post_json(
        ctx.addr(),
        "/v1/tags",
        &serde_json::json!({
            "name": "usage-test"
        }),
    );
    assert_eq!(create_resp.status, 201);
    let create_json = create_resp.json();
    let tag_id = create_json["id"].as_str().unwrap();

    // Get by ID - should include usage stats
    let resp = get(ctx.addr(), &format!("/v1/tags/{}", tag_id));
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["name"], "usage-test");

    // Verify usage field exists with correct structure
    let usage = &json["usage"];
    assert!(usage.is_object(), "usage should be an object");
    assert_eq!(usage["sessions"], 0, "sessions should be 0");
    assert_eq!(usage["documents"], 0, "documents should be 0");
    assert_eq!(usage["total"], 0, "total should be 0");
}

/// Get non-existent tag returns 404.
#[test]
fn get_tag_not_found() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(tags_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/tags/nonexistent-id");
    assert_eq!(resp.status, 404, "body: {}", resp.body);
}

// ---------------------------------------------------------------------------
// PATCH /v1/tags/:id - Update tag
// ---------------------------------------------------------------------------

/// Update tag name.
#[test]
fn update_tag_name() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Create a tag
    let create_resp = post_json(
        ctx.addr(),
        "/v1/tags",
        &serde_json::json!({"name": "old-name"}),
    );
    assert_eq!(create_resp.status, 201);
    let create_json = create_resp.json();
    let tag_id = create_json["id"].as_str().unwrap();

    // Update name
    let resp = patch_json(
        ctx.addr(),
        &format!("/v1/tags/{}", tag_id),
        &serde_json::json!({
            "name": "new-name"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["name"], "new-name");
}

/// Update tag color.
#[test]
fn update_tag_color() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Create a tag
    let create_resp = post_json(ctx.addr(), "/v1/tags", &serde_json::json!({"name": "test"}));
    assert_eq!(create_resp.status, 201);
    let create_json = create_resp.json();
    let tag_id = create_json["id"].as_str().unwrap();

    // Update color
    let resp = patch_json(
        ctx.addr(),
        &format!("/v1/tags/{}", tag_id),
        &serde_json::json!({
            "color": "#00ff00"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["color"], "#00ff00");
}

/// Update tag description.
#[test]
fn update_tag_description() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Create a tag
    let create_resp = post_json(ctx.addr(), "/v1/tags", &serde_json::json!({"name": "test"}));
    assert_eq!(create_resp.status, 201);
    let create_json = create_resp.json();
    let tag_id = create_json["id"].as_str().unwrap();

    // Update description
    let resp = patch_json(
        ctx.addr(),
        &format!("/v1/tags/{}", tag_id),
        &serde_json::json!({
            "description": "A test tag with description"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["description"], "A test tag with description");
}

/// Update non-existent tag returns 404.
#[test]
fn update_tag_not_found() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(tags_config(temp.path()));

    let resp = patch_json(
        ctx.addr(),
        "/v1/tags/nonexistent",
        &serde_json::json!({
            "name": "new"
        }),
    );
    assert_eq!(resp.status, 404, "body: {}", resp.body);
}

// ---------------------------------------------------------------------------
// DELETE /v1/tags/:id - Delete tag
// ---------------------------------------------------------------------------

/// Delete a tag.
#[test]
fn delete_tag() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Create a tag
    let create_resp = post_json(
        ctx.addr(),
        "/v1/tags",
        &serde_json::json!({"name": "to-delete"}),
    );
    assert_eq!(create_resp.status, 201);
    let create_json = create_resp.json();
    let tag_id = create_json["id"].as_str().unwrap().to_string();

    // Delete
    let resp = delete(ctx.addr(), &format!("/v1/tags/{}", tag_id));
    assert_eq!(resp.status, 204, "body: {}", resp.body);

    // Verify deleted
    let get_resp = get(ctx.addr(), &format!("/v1/tags/{}", tag_id));
    assert_eq!(get_resp.status, 404);
}

/// Delete non-existent tag returns 404.
#[test]
fn delete_tag_not_found() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(tags_config(temp.path()));

    let resp = delete(ctx.addr(), "/v1/tags/nonexistent");
    assert_eq!(resp.status, 404, "body: {}", resp.body);
}

// ---------------------------------------------------------------------------
// Bare paths (without /v1 prefix)
// ---------------------------------------------------------------------------

/// Bare /tags path works.
#[test]
fn bare_tags_path() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Create via /tags
    let resp = post_json(
        ctx.addr(),
        "/tags",
        &serde_json::json!({"name": "bare-test"}),
    );
    assert_eq!(resp.status, 201);

    // List via /tags
    let list_resp = get(ctx.addr(), "/tags");
    assert_eq!(list_resp.status, 200);
    let list_json = list_resp.json();
    let data = list_json["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
}

// ---------------------------------------------------------------------------
// Tag usage.documents count
// ---------------------------------------------------------------------------

/// Tag usage.documents count reflects tagged documents.
#[test]
fn tag_usage_counts_documents() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Create a tag.
    let tag_resp = post_json(
        ctx.addr(),
        "/v1/tags",
        &serde_json::json!({"name": "doc-usage"}),
    );
    assert_eq!(tag_resp.status, 201);
    let tag_id = tag_resp.json()["id"].as_str().unwrap().to_string();

    // Initially usage.documents should be 0.
    let resp = get(ctx.addr(), &format!("/v1/tags/{}", tag_id));
    let json = resp.json();
    assert_eq!(json["usage"]["documents"], 0);
    assert_eq!(json["usage"]["total"], 0);

    // Create two documents.
    let doc1_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt",
            "title": "Doc One",
            "content": {}
        }),
    );
    assert_eq!(doc1_resp.status, 201);
    let doc1_id = doc1_resp.json()["id"].as_str().unwrap().to_string();

    let doc2_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt",
            "title": "Doc Two",
            "content": {}
        }),
    );
    assert_eq!(doc2_resp.status, 201);
    let doc2_id = doc2_resp.json()["id"].as_str().unwrap().to_string();

    // Tag both documents.
    let tag_body = serde_json::json!({"tags": [&tag_id]});
    post_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc1_id),
        &tag_body,
    );
    post_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc2_id),
        &tag_body,
    );

    // Usage should now reflect 2 documents.
    let resp = get(ctx.addr(), &format!("/v1/tags/{}", tag_id));
    let json = resp.json();
    assert_eq!(json["usage"]["documents"], 2, "expected 2 tagged documents");
    assert_eq!(json["usage"]["total"], 2);
}
