//! Integration tests for `POST /v1/conversations/batch` endpoint.

use super::{conversation_config, no_bucket_config, seed_session};
use crate::server::common::*;
use serde_json::json;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Batch delete
// ---------------------------------------------------------------------------

/// Batch delete removes multiple conversations.
#[test]
fn batch_delete_removes_conversations() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat 1", "model");
    seed_session(temp.path(), "sess-2", "Chat 2", "model");
    seed_session(temp.path(), "sess-3", "Chat 3", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    // Delete sess-1 and sess-2
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/batch",
        &json!({
            "action": "delete",
            "ids": ["sess-1", "sess-2"]
        }),
    );
    assert_eq!(resp.status, 204, "body: {}", resp.body);

    // Verify only sess-3 remains
    let list_resp = get(ctx.addr(), "/v1/conversations");
    let list_json = list_resp.json();
    let data = list_json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "sess-3");
}

/// Batch delete is idempotent (deleting non-existent is OK).
#[test]
fn batch_delete_idempotent() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat 1", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    // Delete sess-1 and non-existent sess-99
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/batch",
        &json!({
            "action": "delete",
            "ids": ["sess-1", "sess-99"]
        }),
    );
    assert_eq!(resp.status, 204);
}

// ---------------------------------------------------------------------------
// Batch archive/unarchive
// ---------------------------------------------------------------------------

/// Batch archive sets marker to "archived".
#[test]
fn batch_archive_sets_marker() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat 1", "model");
    seed_session(temp.path(), "sess-2", "Chat 2", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    // Archive sess-1
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/batch",
        &json!({
            "action": "archive",
            "ids": ["sess-1"]
        }),
    );
    assert_eq!(resp.status, 204);

    // Verify sess-1 has marker "archived"
    let get_resp = get(ctx.addr(), "/v1/conversations/sess-1");
    let get_json = get_resp.json();
    assert_eq!(get_json["marker"], "archived");

    // sess-2 should still have "active" marker
    let get_resp2 = get(ctx.addr(), "/v1/conversations/sess-2");
    let get_json2 = get_resp2.json();
    assert_eq!(get_json2["marker"], "active");
}

/// Batch unarchive clears the marker.
#[test]
fn batch_unarchive_clears_marker() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat 1", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    // First archive
    post_json(
        ctx.addr(),
        "/v1/conversations/batch",
        &json!({
            "action": "archive",
            "ids": ["sess-1"]
        }),
    );

    // Then unarchive
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/batch",
        &json!({
            "action": "unarchive",
            "ids": ["sess-1"]
        }),
    );
    assert_eq!(resp.status, 204);

    // Verify marker is cleared (empty string or null, not "archived")
    let get_resp = get(ctx.addr(), "/v1/conversations/sess-1");
    let get_json = get_resp.json();
    let marker = &get_json["marker"];
    let is_cleared = marker.is_null() || marker.as_str().map_or(false, |s| s.is_empty());
    assert!(
        is_cleared,
        "marker should be cleared (null or empty), got: {:?}",
        marker
    );
}

// ---------------------------------------------------------------------------
// Batch tag operations
// ---------------------------------------------------------------------------

/// Batch add_tags adds tags to multiple conversations.
#[test]
fn batch_add_tags() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat 1", "model");
    seed_session(temp.path(), "sess-2", "Chat 2", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    // First create tag via API
    let create_resp = post_json(ctx.addr(), "/v1/tags", &json!({"name": "work"}));
    assert_eq!(create_resp.status, 201, "create tag: {}", create_resp.body);
    let tag_id = create_resp.json()["id"].as_str().unwrap().to_string();

    // Add tag to both conversations using tag ID
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/batch",
        &json!({
            "action": "add_tags",
            "ids": ["sess-1", "sess-2"],
            "tags": [tag_id]
        }),
    );
    assert_eq!(resp.status, 204, "body: {}", resp.body);

    // Verify both have the tag
    let get1 = get(ctx.addr(), "/v1/conversations/sess-1");
    let json1 = get1.json();
    let tags1 = json1["tags"].as_array().expect("tags array");
    assert_eq!(tags1.len(), 1);
    assert_eq!(tags1[0]["name"], "work");

    let get2 = get(ctx.addr(), "/v1/conversations/sess-2");
    let json2 = get2.json();
    let tags2 = json2["tags"].as_array().expect("tags array");
    assert_eq!(tags2.len(), 1);
    assert_eq!(tags2[0]["name"], "work");
}

/// Batch remove_tags removes tags from multiple conversations.
#[test]
fn batch_remove_tags() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat 1", "model");
    seed_session(temp.path(), "sess-2", "Chat 2", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    // Create tags via API first
    let work_resp = post_json(ctx.addr(), "/v1/tags", &json!({"name": "work"}));
    assert_eq!(work_resp.status, 201);
    let work_tag_id = work_resp.json()["id"].as_str().unwrap().to_string();

    let urgent_resp = post_json(ctx.addr(), "/v1/tags", &json!({"name": "urgent"}));
    assert_eq!(urgent_resp.status, 201);
    let urgent_tag_id = urgent_resp.json()["id"].as_str().unwrap().to_string();

    // Add both tags to both conversations
    let add_resp = post_json(
        ctx.addr(),
        "/v1/conversations/batch",
        &json!({
            "action": "add_tags",
            "ids": ["sess-1", "sess-2"],
            "tags": [&work_tag_id, &urgent_tag_id]
        }),
    );
    assert_eq!(add_resp.status, 204, "add_tags failed: {}", add_resp.body);

    // Remove "work" tag from both
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/batch",
        &json!({
            "action": "remove_tags",
            "ids": ["sess-1", "sess-2"],
            "tags": [work_tag_id]
        }),
    );
    assert_eq!(resp.status, 204);

    // Verify only "urgent" remains
    let get1 = get(ctx.addr(), "/v1/conversations/sess-1");
    let json1 = get1.json();
    let tags1 = json1["tags"].as_array().expect("tags array");
    assert_eq!(tags1.len(), 1);
    assert_eq!(tags1[0]["name"], "urgent");
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Invalid action returns 400.
#[test]
fn batch_invalid_action() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/batch",
        &json!({
            "action": "invalid_action",
            "ids": ["sess-1"]
        }),
    );
    assert_eq!(resp.status, 400);
    let json = resp.json();
    assert_eq!(json["error"]["code"], "invalid_action");
}

/// Empty ids returns 400.
#[test]
fn batch_empty_ids() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/batch",
        &json!({
            "action": "delete",
            "ids": []
        }),
    );
    assert_eq!(resp.status, 400);
    let json = resp.json();
    assert_eq!(json["error"]["code"], "missing_ids");
}

/// add_tags without tags returns 400.
#[test]
fn batch_add_tags_missing_tags() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/batch",
        &json!({
            "action": "add_tags",
            "ids": ["sess-1"]
        }),
    );
    assert_eq!(resp.status, 400);
    let json = resp.json();
    assert_eq!(json["error"]["code"], "missing_tags");
}

/// Batch size limit is enforced.
#[test]
fn batch_size_limit() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    // Create 101 IDs (over the 100 limit)
    let ids: Vec<String> = (0..101).map(|i| format!("sess-{}", i)).collect();

    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/batch",
        &json!({
            "action": "delete",
            "ids": ids
        }),
    );
    assert_eq!(resp.status, 400);
    let json = resp.json();
    assert_eq!(json["error"]["code"], "batch_too_large");
}

/// Batch operations require storage (no_bucket returns 503).
#[test]
fn batch_no_bucket_returns_503() {
    let ctx = ServerTestContext::new(no_bucket_config());

    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/batch",
        &json!({
            "action": "delete",
            "ids": ["sess-1"]
        }),
    );
    assert_eq!(resp.status, 503);
    let json = resp.json();
    assert_eq!(json["error"]["code"], "storage_unavailable");
}

/// Batch archive is idempotent (archiving already-archived is OK).
#[test]
fn batch_archive_idempotent() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat 1", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    // Archive twice
    let resp1 = post_json(
        ctx.addr(),
        "/v1/conversations/batch",
        &json!({
            "action": "archive",
            "ids": ["sess-1"]
        }),
    );
    assert_eq!(resp1.status, 204);

    let resp2 = post_json(
        ctx.addr(),
        "/v1/conversations/batch",
        &json!({
            "action": "archive",
            "ids": ["sess-1"]
        }),
    );
    assert_eq!(resp2.status, 204);

    // Verify still archived
    let get_resp = get(ctx.addr(), "/v1/conversations/sess-1");
    let get_json = get_resp.json();
    assert_eq!(get_json["marker"], "archived");
}

/// Batch remove_tags is idempotent (removing non-existent tag from conversation is OK).
#[test]
fn batch_remove_tags_idempotent() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat 1", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    // Create a tag but don't assign it
    let create_resp = post_json(ctx.addr(), "/v1/tags", &json!({"name": "unused"}));
    assert_eq!(create_resp.status, 201);
    let tag_id = create_resp.json()["id"].as_str().unwrap().to_string();

    // Remove tag that was never assigned (should be OK)
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/batch",
        &json!({
            "action": "remove_tags",
            "ids": ["sess-1"],
            "tags": [tag_id]
        }),
    );
    assert_eq!(resp.status, 204);
}
