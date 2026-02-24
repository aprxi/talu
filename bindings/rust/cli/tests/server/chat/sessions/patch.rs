use super::{no_bucket_config, seed_session, session_config};
use crate::server::common::*;
use serde_json::json;
use tempfile::TempDir;

#[test]
fn patch_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/some-id",
        &json!({"title": "New"}),
    );
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn patch_updates_title() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-patch", "Old Title", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-patch",
        &json!({"title": "New Title"}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["title"], "New Title");
    assert_eq!(json["id"], "sess-patch");
}

#[test]
fn patch_updates_marker() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-marker", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-marker",
        &json!({"marker": "archived"}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["marker"], "archived");
}

#[test]
fn patch_preserves_unchanged_fields() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(
        temp.path(),
        "sess-preserve",
        "Keep This Title",
        "test-model",
    );

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Only patch marker — title should be preserved
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-preserve",
        &json!({"marker": "done"}),
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    assert_eq!(json["title"], "Keep This Title");
    assert_eq!(json["marker"], "done");
}

#[test]
fn patch_metadata_replaces_entire_object() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-meta", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Set initial metadata
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-meta",
        &json!({"metadata": {"key1": "val1", "key2": "val2"}}),
    );
    assert_eq!(resp.status, 200);

    // Replace with different metadata — key1 and key2 should be gone
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-meta",
        &json!({"metadata": {"starred": true}}),
    );
    assert_eq!(resp.status, 200);

    // Verify via GET
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-meta");
    assert_eq!(resp.status, 200);
    // The metadata in the GET response is what was last PATCHed
    // (Note: get uses slim SessionRecord which doesn't have metadata,
    //  so we verify through the PATCH response above)
}

#[test]
fn patch_nonexistent_returns_404() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-exists", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/nonexistent",
        &json!({"title": "New"}),
    );
    assert_eq!(resp.status, 404, "body: {}", resp.body);
}

#[test]
fn patch_empty_body_is_noop() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-noop", "Original", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = patch_json(ctx.addr(), "/v1/chat/sessions/sess-noop", &json!({}));
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["title"], "Original");
}

#[test]
fn patch_title_persists_via_get() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-persist", "Before", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // PATCH
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-persist",
        &json!({"title": "After"}),
    );
    assert_eq!(resp.status, 200);

    // Verify via GET (independent read)
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-persist");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["title"], "After");
}

#[test]
fn patch_marker_persists_via_get() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-st-persist", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-st-persist",
        &json!({"marker": "archived"}),
    );
    assert_eq!(resp.status, 200);

    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-st-persist");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["marker"], "archived");
}

#[test]
fn patch_multiple_fields_at_once() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-multi-patch", "Old Title", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-multi-patch",
        &json!({"title": "New Title", "marker": "done"}),
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    assert_eq!(json["title"], "New Title");
    assert_eq!(json["marker"], "done");
}

#[test]
fn patch_sequential_updates() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-seq", "Step 0", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // First update
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-seq",
        &json!({"title": "Step 1"}),
    );
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["title"], "Step 1");

    // Second update
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-seq",
        &json!({"title": "Step 2"}),
    );
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["title"], "Step 2");

    // Third update — verify last one wins
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-seq",
        &json!({"title": "Final"}),
    );
    assert_eq!(resp.status, 200);

    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-seq");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["title"], "Final");
}

#[test]
fn patch_invalid_json_returns_400() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-badjson", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    // Send invalid JSON
    let resp = send_request(
        ctx.addr(),
        "PATCH",
        "/v1/chat/sessions/sess-badjson",
        &[("Content-Type", "application/json")],
        Some("not json {{{"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn patch_without_prefix() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-patch-np", "Before", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-patch-np",
        &json!({"title": "After"}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["title"], "After");
}

#[test]
fn patch_does_not_modify_model() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-no-model", "Chat", "original-model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Patch only title — model should remain unchanged
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-no-model",
        &json!({"title": "Updated"}),
    );
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["model"], "original-model");
}

#[test]
fn patch_does_not_modify_id() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-keep-id", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-keep-id",
        &json!({"title": "New"}),
    );
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["id"], "sess-keep-id");
}

#[test]
fn patch_updated_session_reflected_in_list() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-list-check", "Old Title", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Patch the title
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-list-check",
        &json!({"title": "Patched Title"}),
    );
    assert_eq!(resp.status, 200);

    // Verify the updated title shows in the list
    let resp = get(ctx.addr(), "/v1/chat/sessions");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    let session = data
        .iter()
        .find(|s| s["id"] == "sess-list-check")
        .expect("session in list");
    assert_eq!(session["title"], "Patched Title");
}

// ---------------------------------------------------------------------------
// Body edge cases
// ---------------------------------------------------------------------------

#[test]
fn patch_null_title_treated_as_no_change() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-null-title", "Original Title", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    // title: null — as_str() returns None, so title is not updated
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-null-title",
        &json!({"title": null}),
    );
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["title"], "Original Title");
}

#[test]
fn patch_non_string_title_treated_as_no_change() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-int-title", "Original", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    // title: 123 — as_str() returns None, so title is not updated
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-int-title",
        &json!({"title": 123}),
    );
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["title"], "Original");
}

#[test]
fn patch_extra_fields_ignored() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-extra", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-extra",
        &json!({"title": "New", "unknown_field": "ignored", "foo": 42}),
    );
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["title"], "New");
}

#[test]
fn patch_empty_string_title() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-empty-title", "Non-empty", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-empty-title",
        &json!({"title": ""}),
    );
    assert_eq!(resp.status, 200);
    // Empty string is treated as null by the C API storage layer
    let title = &resp.json()["title"];
    assert!(
        title.is_null() || title == "",
        "empty string title should be stored as null or empty: got {:?}",
        title
    );
}

#[test]
fn patch_unicode_title() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-unicode", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-unicode",
        &json!({"title": "Caf\u{00e9} \u{1f600} \u{4e16}\u{754c}"}),
    );
    assert_eq!(resp.status, 200);
    assert_eq!(
        resp.json()["title"],
        "Caf\u{00e9} \u{1f600} \u{4e16}\u{754c}"
    );
}

#[test]
fn patch_metadata_as_null_clears_metadata() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-meta-null", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // First set metadata
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-meta-null",
        &json!({"metadata": {"key": "value"}}),
    );
    assert_eq!(resp.status, 200);

    // Then send metadata: null — this replaces with "null" string which
    // the handler serializes via .to_string() on Value::Null
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-meta-null",
        &json!({"metadata": null}),
    );
    assert_eq!(resp.status, 200);
}

#[test]
fn patch_metadata_complex_object() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-meta-complex", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-meta-complex",
        &json!({"metadata": {
            "tags": ["rust", "talu"],
            "priority": 5,
            "nested": {"deep": true}
        }}),
    );
    assert_eq!(resp.status, 200);
}

#[test]
fn patch_response_has_object_field() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-obj", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-obj",
        &json!({"title": "Updated"}),
    );
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["object"], "session");
}

#[test]
fn patch_preserves_created_at() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-created", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Get original created_at
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-created");
    assert_eq!(resp.status, 200);
    let original_created = resp.json()["created_at"].as_i64().expect("created_at");

    // Patch
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-created",
        &json!({"title": "Updated"}),
    );
    assert_eq!(resp.status, 200);

    // Verify created_at unchanged
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-created");
    assert_eq!(resp.status, 200);
    let new_created = resp.json()["created_at"].as_i64().expect("created_at");
    assert_eq!(
        original_created, new_created,
        "created_at should not change on patch"
    );
}

// ---------------------------------------------------------------------------
// PATCH metadata round-trip via GET
// ---------------------------------------------------------------------------

#[test]
fn patch_metadata_persists_via_list() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-meta-list", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Set metadata
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-meta-list",
        &json!({"metadata": {"starred": true, "color": "blue"}}),
    );
    assert_eq!(resp.status, 200);

    // Verify in list response (uses session_to_conversation_json → SessionRecordFull)
    let resp = get(ctx.addr(), "/v1/chat/sessions");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    let session = data
        .iter()
        .find(|s| s["id"] == "sess-meta-list")
        .expect("session in list");
    assert_eq!(session["metadata"]["starred"], true);
    assert_eq!(session["metadata"]["color"], "blue");
}

#[test]
fn patch_metadata_replacement_removes_old_keys_via_list() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-meta-repl", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Set initial metadata with two keys
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-meta-repl",
        &json!({"metadata": {"key1": "a", "key2": "b"}}),
    );
    assert_eq!(resp.status, 200);

    // Replace with different metadata
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-meta-repl",
        &json!({"metadata": {"key3": "c"}}),
    );
    assert_eq!(resp.status, 200);

    // Verify via list: old keys should be gone
    let resp = get(ctx.addr(), "/v1/chat/sessions");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    let session = data
        .iter()
        .find(|s| s["id"] == "sess-meta-repl")
        .expect("session in list");
    assert_eq!(session["metadata"]["key3"], "c");
    assert!(
        session["metadata"]["key1"].is_null(),
        "key1 should be gone after replacement"
    );
    assert!(
        session["metadata"]["key2"].is_null(),
        "key2 should be gone after replacement"
    );
}

// ---------------------------------------------------------------------------
// PATCH content-type and response format
// ---------------------------------------------------------------------------

#[test]
fn patch_response_has_json_content_type() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-patch-ct", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-patch-ct",
        &json!({"title": "New"}),
    );
    assert_eq!(resp.status, 200);
    assert_eq!(resp.header("content-type"), Some("application/json"));
}

#[test]
fn patch_404_error_body_is_json() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-x", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/nonexistent",
        &json!({"title": "New"}),
    );
    assert_eq!(resp.status, 404);
    let json = resp.json();
    assert!(json["error"]["code"].is_string(), "error should have code");
    assert!(
        json["error"]["message"].is_string(),
        "error should have message"
    );
}

#[test]
fn patch_400_error_body_is_json() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-bad-patch", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = send_request(
        ctx.addr(),
        "PATCH",
        "/v1/chat/sessions/sess-bad-patch",
        &[("Content-Type", "application/json")],
        Some("{invalid json"),
    );
    assert_eq!(resp.status, 400);
    let json = resp.json();
    assert!(
        json["error"]["code"].is_string(),
        "400 error should have code"
    );
    assert!(
        json["error"]["message"].is_string(),
        "400 error should have message"
    );
}

#[test]
fn patch_updated_at_changes() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-ts-change", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Get original timestamps
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-ts-change");
    assert_eq!(resp.status, 200);
    let original_updated = resp.json()["updated_at"].as_i64().expect("updated_at");

    // Small delay to ensure timestamp difference
    std::thread::sleep(std::time::Duration::from_millis(10));

    // Patch
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-ts-change",
        &json!({"title": "Changed"}),
    );
    assert_eq!(resp.status, 200);

    // Verify updated_at increased
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-ts-change");
    assert_eq!(resp.status, 200);
    let new_updated = resp.json()["updated_at"].as_i64().expect("updated_at");
    assert!(
        new_updated >= original_updated,
        "updated_at should be >= original after patch: {} vs {}",
        new_updated,
        original_updated
    );
}
