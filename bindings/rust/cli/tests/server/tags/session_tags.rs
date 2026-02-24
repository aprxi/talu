//! Integration tests for `/v1/chat/sessions/:id/tags` endpoints.

use super::tags_config;
use crate::server::common::*;
use crate::server::sessions::{seed_session, seed_session_with_tags};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// GET /v1/chat/sessions/:id/tags
// ---------------------------------------------------------------------------

/// Get tags for a session with no tags.
#[test]
fn get_session_tags_empty() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Test Chat", "model");

    let ctx = ServerTestContext::new(tags_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-1/tags");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let tags = json["tags"].as_array().expect("tags array");
    assert!(tags.is_empty());
}

// ---------------------------------------------------------------------------
// POST /v1/chat/sessions/:id/tags - Add tags
// ---------------------------------------------------------------------------

/// Add tags by ID to a session.
#[test]
fn add_tags_by_id() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Test Chat", "model");

    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Create a tag first
    let tag_resp = post_json(ctx.addr(), "/v1/tags", &serde_json::json!({"name": "work"}));
    assert_eq!(tag_resp.status, 201);
    let tag_id = tag_resp.json()["id"].as_str().unwrap().to_string();

    // Add tag to session
    let resp = post_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-1/tags",
        &serde_json::json!({
            "tags": [tag_id]
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let tags = json["tags"].as_array().expect("tags array");
    assert_eq!(tags.len(), 1);
    assert_eq!(tags[0]["name"], "work");
}

/// Add tags by name (auto-creates if not exists).
#[test]
fn add_tags_by_name_auto_creates() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Test Chat", "model");

    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Add tag by name (should auto-create)
    let resp = post_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-1/tags",
        &serde_json::json!({
            "tags": ["new-tag"]
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let tags = json["tags"].as_array().expect("tags array");
    assert_eq!(tags.len(), 1);
    assert_eq!(tags[0]["name"], "new-tag");

    // Verify tag was created in tags list
    let tags_list = get(ctx.addr(), "/v1/tags");
    let tags_list_json = tags_list.json();
    let tags_data = tags_list_json["data"].as_array().unwrap();
    assert_eq!(tags_data.len(), 1);
    assert_eq!(tags_data[0]["name"], "new-tag");
}

/// Add multiple tags at once.
#[test]
fn add_multiple_tags() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Test Chat", "model");

    let ctx = ServerTestContext::new(tags_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-1/tags",
        &serde_json::json!({
            "tags": ["work", "urgent", "project-x"]
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let tags = json["tags"].as_array().expect("tags array");
    assert_eq!(tags.len(), 3);
}

/// Adding the same tag twice is idempotent.
#[test]
fn add_tag_idempotent() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Test Chat", "model");

    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Add tag first time
    let resp1 = post_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-1/tags",
        &serde_json::json!({
            "tags": ["work"]
        }),
    );
    assert_eq!(resp1.status, 200);

    // Add same tag again
    let resp2 = post_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-1/tags",
        &serde_json::json!({
            "tags": ["work"]
        }),
    );
    assert_eq!(resp2.status, 200);

    // Should still have only one tag
    let json = resp2.json();
    let tags = json["tags"].as_array().expect("tags array");
    assert_eq!(tags.len(), 1);
}

// ---------------------------------------------------------------------------
// DELETE /v1/chat/sessions/:id/tags - Remove tags
// ---------------------------------------------------------------------------

/// Remove tags from a session.
#[test]
fn remove_tags() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Test Chat", "model");

    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Add tags
    post_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-1/tags",
        &serde_json::json!({
            "tags": ["work", "urgent"]
        }),
    );

    // Get the tag ID for "work"
    let tags_resp = get(ctx.addr(), "/v1/chat/sessions/sess-1/tags");
    let tags_json = tags_resp.json();
    let tags = tags_json["tags"].as_array().unwrap();
    let work_tag_id = tags.iter().find(|t| t["name"] == "work").unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();

    // Remove "work" tag
    let resp = delete_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-1/tags",
        &serde_json::json!({
            "tags": [work_tag_id]
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let remaining_tags = json["tags"].as_array().expect("tags array");
    assert_eq!(remaining_tags.len(), 1);
    assert_eq!(remaining_tags[0]["name"], "urgent");
}

/// Remove non-existent tag is ignored.
#[test]
fn remove_nonexistent_tag() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Test Chat", "model");

    let ctx = ServerTestContext::new(tags_config(temp.path()));

    let resp = delete_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-1/tags",
        &serde_json::json!({
            "tags": ["nonexistent-tag-id"]
        }),
    );
    // Should succeed but do nothing
    assert_eq!(resp.status, 200, "body: {}", resp.body);
}

// ---------------------------------------------------------------------------
// PUT /v1/chat/sessions/:id/tags - Replace all tags
// ---------------------------------------------------------------------------

/// Replace all tags on a session.
#[test]
fn set_tags_replace_all() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Test Chat", "model");

    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Add initial tags
    post_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-1/tags",
        &serde_json::json!({
            "tags": ["old-tag-1", "old-tag-2"]
        }),
    );

    // Replace with new tags
    let resp = put_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-1/tags",
        &serde_json::json!({
            "tags": ["new-tag-1", "new-tag-2", "new-tag-3"]
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let tags = json["tags"].as_array().expect("tags array");
    assert_eq!(tags.len(), 3);

    let names: Vec<&str> = tags.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"new-tag-1"));
    assert!(names.contains(&"new-tag-2"));
    assert!(names.contains(&"new-tag-3"));
    assert!(!names.contains(&"old-tag-1"));
    assert!(!names.contains(&"old-tag-2"));
}

/// Set empty tags removes all.
#[test]
fn set_tags_empty_removes_all() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Test Chat", "model");

    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Add tags
    post_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-1/tags",
        &serde_json::json!({
            "tags": ["work", "urgent"]
        }),
    );

    // Replace with empty
    let resp = put_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-1/tags",
        &serde_json::json!({
            "tags": []
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let tags = json["tags"].as_array().expect("tags array");
    assert!(tags.is_empty());
}

// ---------------------------------------------------------------------------
// Resolved tags in session responses
// ---------------------------------------------------------------------------

/// GET /v1/chat/sessions includes resolved tags array.
#[test]
fn session_list_includes_tags() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Test Chat", "model");

    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Add tags to session
    post_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-1/tags",
        &serde_json::json!({
            "tags": ["work", "urgent"]
        }),
    );

    // List sessions
    let resp = get(ctx.addr(), "/v1/chat/sessions");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);

    // Check tags field
    let tags = data[0]["tags"].as_array().expect("tags array");
    assert_eq!(tags.len(), 2);

    // Tags should be objects with id, name, color
    let names: Vec<&str> = tags.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"work"));
    assert!(names.contains(&"urgent"));

    // Each tag should have an id
    for tag in tags {
        assert!(tag["id"].is_string(), "tag should have id");
        assert!(tag["name"].is_string(), "tag should have name");
    }
}

/// GET /v1/chat/sessions/:id includes resolved tags array.
#[test]
fn session_get_includes_tags() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Test Chat", "model");

    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Add tags to session
    post_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-1/tags",
        &serde_json::json!({
            "tags": ["project-x"]
        }),
    );

    // Get single session
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-1");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();

    // Check tags field
    let tags = json["tags"].as_array().expect("tags array");
    assert_eq!(tags.len(), 1);
    assert_eq!(tags[0]["name"], "project-x");
    assert!(tags[0]["id"].is_string());
}

/// Session without tags has empty tags array.
#[test]
fn session_without_tags_has_empty_array() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Test Chat", "model");

    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Get session without adding tags
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-1");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let tags = json["tags"].as_array().expect("tags array");
    assert!(tags.is_empty());
}

// ---------------------------------------------------------------------------
// Tag usage statistics with sessions
// ---------------------------------------------------------------------------

/// Tag usage.sessions count reflects tagged sessions.
#[test]
fn tag_usage_counts_sessions() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat One", "model");
    seed_session(temp.path(), "sess-2", "Chat Two", "model");
    seed_session(temp.path(), "sess-3", "Chat Three", "model");

    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Create a tag
    let tag_resp = post_json(ctx.addr(), "/v1/tags", &serde_json::json!({"name": "work"}));
    assert_eq!(tag_resp.status, 201);
    let tag_id = tag_resp.json()["id"].as_str().unwrap().to_string();

    // Initially usage should be 0
    let resp = get(ctx.addr(), &format!("/v1/tags/{}", tag_id));
    let json = resp.json();
    assert_eq!(json["usage"]["sessions"], 0);
    assert_eq!(json["usage"]["total"], 0);

    // Tag two sessions
    post_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-1/tags",
        &serde_json::json!({
            "tags": [&tag_id]
        }),
    );
    post_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-2/tags",
        &serde_json::json!({
            "tags": [&tag_id]
        }),
    );

    // Usage should now reflect 2 sessions
    let resp = get(ctx.addr(), &format!("/v1/tags/{}", tag_id));
    let json = resp.json();
    assert_eq!(json["usage"]["sessions"], 2, "expected 2 tagged sessions");
    assert_eq!(json["usage"]["total"], 2);

    // Tag another session
    post_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-3/tags",
        &serde_json::json!({
            "tags": [&tag_id]
        }),
    );

    // Usage should now be 3
    let resp = get(ctx.addr(), &format!("/v1/tags/{}", tag_id));
    let json = resp.json();
    assert_eq!(json["usage"]["sessions"], 3, "expected 3 tagged sessions");
    assert_eq!(json["usage"]["total"], 3);
}

// ---------------------------------------------------------------------------
// Orphan handling: tag deleted after being assigned to a session
// ---------------------------------------------------------------------------

/// Deleting a tag removes it from resolved tags on sessions.
///
/// `resolve_tags_for_session` calls `get_tag()` for each tag reference;
/// if the tag is deleted, `get_tag()` returns `None` and the reference is
/// silently skipped.
///
/// **Known issue**: `talu_db_table_tag_delete` (Zig core) returns error 999
/// (internal_error) when the tag has session associations, even though
/// `deleteTagAndAssociations` is designed to cascade-remove them.  Verified
/// the failure reproduces via direct FFI (no HTTP handler involved).
/// This test is `#[ignore]`d until the storage bug is fixed; once
/// `DELETE /v1/tags/:id` returns 204, remove the `#[ignore]` attribute.
#[test]
#[ignore = "storage bug: talu_db_table_tag_delete returns 999 for tags with session associations"]
fn deleted_tag_silently_dropped_from_session() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags(
        temp.path(),
        "sess-orphan",
        "Orphan Test Chat",
        "model",
        &["keep-me", "remove-me"],
    );

    let ctx = ServerTestContext::new(tags_config(temp.path()));

    // Verify both tags are visible on the session.
    let before = get(ctx.addr(), "/v1/chat/sessions/sess-orphan");
    assert_eq!(before.status, 200, "body: {}", before.body);
    let before_tags = before.json()["tags"].as_array().expect("tags").clone();
    assert_eq!(before_tags.len(), 2, "should start with 2 tags");

    // Delete "remove-me" tag via the tags CRUD endpoint.
    let del_resp = delete(ctx.addr(), "/v1/tags/tag-remove-me");
    assert_eq!(del_resp.status, 204, "body: {}", del_resp.body);

    // GET single session — only "keep-me" should appear.
    let get_resp = get(ctx.addr(), "/v1/chat/sessions/sess-orphan");
    assert_eq!(get_resp.status, 200, "body: {}", get_resp.body);
    let resolved_tags = get_resp.json()["tags"]
        .as_array()
        .expect("tags array")
        .clone();
    let names: Vec<&str> = resolved_tags
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();
    assert_eq!(
        names,
        vec!["keep-me"],
        "deleted tag should be silently dropped"
    );

    // LIST sessions — same assertion.
    let list_resp = get(ctx.addr(), "/v1/chat/sessions");
    assert_eq!(list_resp.status, 200, "body: {}", list_resp.body);
    let data = list_resp.json()["data"].as_array().expect("data").clone();
    let sess = data
        .iter()
        .find(|s| s["id"] == "sess-orphan")
        .expect("session in list");
    let list_names: Vec<&str> = sess["tags"]
        .as_array()
        .expect("tags")
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();
    assert_eq!(
        list_names,
        vec!["keep-me"],
        "deleted tag dropped from list too"
    );
}
