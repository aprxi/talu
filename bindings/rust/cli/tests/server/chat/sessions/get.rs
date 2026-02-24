use super::{no_bucket_config, seed_session, seed_session_with_messages, session_config};
use crate::server::common::*;
use tempfile::TempDir;

#[test]
fn get_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = get(ctx.addr(), "/v1/chat/sessions/some-id");
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn get_503_error_body_is_json() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = get(ctx.addr(), "/v1/chat/sessions/some-id");
    assert_eq!(resp.status, 503);
    let json = resp.json();
    assert!(
        json["error"]["code"].is_string(),
        "503 error should have code"
    );
    assert!(
        json["error"]["message"].is_string(),
        "503 error should have message"
    );
}

#[test]
fn get_returns_session_with_items() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-get-1", "My Chat", "test-model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-get-1");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["object"], "session");
    assert_eq!(json["id"], "sess-get-1");
    assert_eq!(json["title"], "My Chat");

    // Should have items array with at least the user message
    let items = json["items"].as_array().expect("items array");
    assert!(!items.is_empty(), "should have at least one item");
}

#[test]
fn get_returns_404_for_missing_session() {
    let temp = TempDir::new().expect("temp dir");
    // Seed one session so the DB exists
    seed_session(temp.path(), "sess-exists", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/chat/sessions/nonexistent-session");
    assert_eq!(resp.status, 404, "body: {}", resp.body);
}

#[test]
fn get_without_prefix() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-noprefix", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-noprefix");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["id"], "sess-noprefix");
}

#[test]
fn get_returns_all_metadata_fields() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-meta-get", "Full Meta Chat", "model-x");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-meta-get");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["object"], "session");
    assert_eq!(json["id"], "sess-meta-get");
    assert_eq!(json["title"], "Full Meta Chat");
    assert_eq!(json["model"], "model-x");
    assert_eq!(json["marker"], "active");
    assert!(json["created_at"].as_i64().unwrap() > 0);
    assert!(json["updated_at"].as_i64().unwrap() > 0);
    // Items should be present (it's the full GET)
    assert!(json["items"].is_array(), "GET should include items array");
}

#[test]
fn get_items_contain_user_message() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-item-check", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-item-check");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let items = json["items"].as_array().expect("items");
    assert!(!items.is_empty());

    // Verify items have expected structure (type, role, content)
    let first = &items[0];
    assert_eq!(first["type"], "message", "item should be a message type");
    assert!(first["role"].is_string(), "item should have a role");
    assert!(
        first["content"].is_array(),
        "item should have content array"
    );
    // Verify content has input_text
    let content = first["content"].as_array().expect("content array");
    assert!(!content.is_empty(), "content should not be empty");
    assert_eq!(content[0]["type"], "input_text");
}

#[test]
fn get_multiple_items_preserved() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_messages(
        temp.path(),
        "sess-multi",
        "Multi-msg Chat",
        "model",
        &["First message", "Second message", "Third message"],
    );

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-multi");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let items = json["items"].as_array().expect("items");
    // Should have at least 3 message items (one per appended message)
    let msg_items: Vec<_> = items
        .iter()
        .filter(|item| item["type"].as_str() == Some("message"))
        .collect();
    assert!(
        msg_items.len() >= 3,
        "expected at least 3 message items, got {}",
        msg_items.len()
    );
}

#[test]
fn get_404_error_body_is_json() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-x", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/chat/sessions/nonexistent");
    assert_eq!(resp.status, 404);
    let json = resp.json();
    assert!(
        json["error"]["code"].is_string(),
        "error body should have code"
    );
    assert!(
        json["error"]["message"].is_string(),
        "error body should have message"
    );
}

#[test]
fn get_timestamps_are_consistent() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-ts", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-ts");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let created = json["created_at"].as_i64().expect("created_at");
    let updated = json["updated_at"].as_i64().expect("updated_at");
    assert!(created > 0, "created_at should be positive ms timestamp");
    assert!(updated > 0, "updated_at should be positive ms timestamp");
    assert!(
        updated >= created,
        "updated_at ({updated}) should be >= created_at ({created})"
    );
}

#[test]
fn get_items_have_id_and_type() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-item-fields", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-item-fields");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let items = json["items"].as_array().expect("items");
    for item in items {
        assert!(item["id"].is_string(), "each item should have a string id");
        assert!(
            item["type"].is_string(),
            "each item should have a string type"
        );
    }
}

#[test]
fn get_items_is_array_even_when_single_message() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-single", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-single");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    assert!(json["items"].is_array(), "items should always be an array");
}

#[test]
fn get_response_includes_model() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-model-check", "Chat", "my-specific-model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-model-check");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["model"], "my-specific-model");
}

#[test]
fn get_response_includes_marker() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-marker-check", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-marker-check");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["marker"], "active");
}

// ---------------------------------------------------------------------------
// GET response completeness
// ---------------------------------------------------------------------------

#[test]
fn get_response_includes_parent_session_id_null() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-parent", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-parent");
    assert_eq!(resp.status, 200);
    // Root session should have null parent_session_id
    assert!(
        resp.json()["parent_session_id"].is_null(),
        "root session parent_session_id should be null"
    );
}

#[test]
fn get_response_includes_group_id_null() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-group", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-group");
    assert_eq!(resp.status, 200);
    // Session without group should have null group_id
    assert!(
        resp.json()["group_id"].is_null(),
        "session without group should have null group_id"
    );
}

#[test]
fn get_response_includes_metadata_empty_object() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-meta-empty", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-meta-empty");
    assert_eq!(resp.status, 200);
    // Default metadata should be empty object {}
    let metadata = &resp.json()["metadata"];
    assert!(
        metadata.is_object() || metadata.is_null(),
        "default metadata should be object or null: {:?}",
        metadata
    );
}

#[test]
fn get_patched_title_reflected() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-get-patch", "Before", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Patch the title
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-get-patch",
        &serde_json::json!({"title": "After"}),
    );
    assert_eq!(resp.status, 200);

    // GET should show the new title
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-get-patch");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["title"], "After");
    // Items should still be present
    let json = resp.json();
    let items = json["items"].as_array().expect("items");
    assert!(!items.is_empty(), "items should survive a title patch");
}

#[test]
fn get_patched_marker_reflected() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-get-marker", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-get-marker",
        &serde_json::json!({"marker": "completed"}),
    );
    assert_eq!(resp.status, 200);

    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-get-marker");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["marker"], "completed");
}

// ---------------------------------------------------------------------------
// Generation metadata
// ---------------------------------------------------------------------------

#[test]
fn get_manual_items_have_no_generation() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-gen-none", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-gen-none");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let items = json["items"].as_array().expect("items");

    // Manually seeded items should have generation=null or absent
    for item in items {
        let gen = item.get("generation");
        assert!(
            gen.is_none() || gen.unwrap().is_null(),
            "manual item should have generation=null or absent, got {:?}",
            gen
        );
    }
}
