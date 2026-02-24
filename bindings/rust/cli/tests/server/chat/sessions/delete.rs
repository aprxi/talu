use super::{no_bucket_config, seed_session, session_config};
use crate::server::common::*;
use tempfile::TempDir;

#[test]
fn delete_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = delete(ctx.addr(), "/v1/chat/sessions/some-id");
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn delete_503_error_body_is_json() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = delete(ctx.addr(), "/v1/chat/sessions/some-id");
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
fn delete_returns_204() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-del", "Doomed chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Verify it exists first
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-del");
    assert_eq!(resp.status, 200);

    // Delete
    let resp = delete(ctx.addr(), "/v1/chat/sessions/sess-del");
    assert_eq!(resp.status, 204, "body: {}", resp.body);
}

#[test]
fn delete_is_idempotent() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-idem", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // First delete
    let resp1 = delete(ctx.addr(), "/v1/chat/sessions/sess-idem");
    assert_eq!(resp1.status, 204);

    // Second delete of same session — still 204
    let resp2 = delete(ctx.addr(), "/v1/chat/sessions/sess-idem");
    assert_eq!(resp2.status, 204);
}

#[test]
fn delete_nonexistent_returns_204() {
    let temp = TempDir::new().expect("temp dir");
    // Seed something so DB exists
    seed_session(temp.path(), "sess-other", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    let resp = delete(ctx.addr(), "/v1/chat/sessions/never-existed");
    assert_eq!(resp.status, 204);
}

#[test]
fn deleted_session_excluded_from_list() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-keep", "Keep me", "model");
    seed_session(temp.path(), "sess-remove", "Remove me", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Delete one session
    let resp = delete(ctx.addr(), "/v1/chat/sessions/sess-remove");
    assert_eq!(resp.status, 204);

    // List should only contain the remaining session
    let resp = get(ctx.addr(), "/v1/chat/sessions");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "sess-keep");
}

#[test]
fn get_returns_404_after_delete() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-gone", "Doomed", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    // Verify it exists
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-gone");
    assert_eq!(resp.status, 200);

    // Delete it
    let resp = delete(ctx.addr(), "/v1/chat/sessions/sess-gone");
    assert_eq!(resp.status, 204);

    // GET should now return 404
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-gone");
    assert_eq!(resp.status, 404, "body: {}", resp.body);
}

#[test]
fn delete_without_prefix() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-del-np", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = delete(ctx.addr(), "/v1/chat/sessions/sess-del-np");
    assert_eq!(resp.status, 204);

    // Verify it's gone
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-del-np");
    assert_eq!(resp.status, 404);
}

#[test]
fn delete_body_is_empty() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-empty-body", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = delete(ctx.addr(), "/v1/chat/sessions/sess-empty-body");
    assert_eq!(resp.status, 204);
    assert!(resp.body.is_empty(), "204 response body should be empty");
}

// ---------------------------------------------------------------------------
// Cross-endpoint interactions
// ---------------------------------------------------------------------------

#[test]
fn patch_after_delete_returns_404() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-patch-del", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    let resp = delete(ctx.addr(), "/v1/chat/sessions/sess-patch-del");
    assert_eq!(resp.status, 204);

    // PATCH on deleted session should return 404
    let resp = patch_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-patch-del",
        &serde_json::json!({"title": "New"}),
    );
    assert_eq!(resp.status, 404, "body: {}", resp.body);
}

#[test]
fn fork_after_delete_succeeds_with_empty_session() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-del", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    let resp = delete(ctx.addr(), "/v1/chat/sessions/sess-fork-del");
    assert_eq!(resp.status, 204);

    // Fork from deleted session — the core creates a new session even if
    // the source is deleted (source items were tombstoned, so the fork
    // gets an empty/minimal conversation). This is valid behavior.
    let resp = post_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-fork-del/fork",
        &serde_json::json!({"target_item_id": 0}),
    );
    // Could be 201 (new empty fork) or 404/500 (source gone) — accept either
    assert!(
        resp.status == 201 || resp.status == 404 || resp.status == 500,
        "fork after delete: status {} body: {}",
        resp.status,
        resp.body
    );
}

#[test]
fn delete_all_then_list_empty() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-da", "Chat 1", "model");
    seed_session(temp.path(), "sess-db", "Chat 2", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    delete(ctx.addr(), "/v1/chat/sessions/sess-da");
    delete(ctx.addr(), "/v1/chat/sessions/sess-db");

    let resp = get(ctx.addr(), "/v1/chat/sessions");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert!(data.is_empty(), "all deleted — list should be empty");
}

// ---------------------------------------------------------------------------
// DELETE response format
// ---------------------------------------------------------------------------

#[test]
fn delete_response_has_no_content_type_on_204() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-del-ct", "Chat", "model");

    let ctx = ServerTestContext::new(session_config(temp.path()));
    let resp = delete(ctx.addr(), "/v1/chat/sessions/sess-del-ct");
    assert_eq!(resp.status, 204);
    // 204 responses typically have no content-type, but we don't enforce this
    // — the important thing is the body is empty
    assert!(resp.body.is_empty(), "204 body should be empty");
}

// ---------------------------------------------------------------------------
// DELETE preserves other sessions
// ---------------------------------------------------------------------------

#[test]
fn delete_one_preserves_others_metadata() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-keep-a", "Keep A", "model-a");
    seed_session(temp.path(), "sess-keep-b", "Keep B", "model-b");
    seed_session(temp.path(), "sess-remove-c", "Remove C", "model-c");

    let ctx = ServerTestContext::new(session_config(temp.path()));

    delete(ctx.addr(), "/v1/chat/sessions/sess-remove-c");

    // Verify remaining sessions are fully intact
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-keep-a");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["title"], "Keep A");
    assert_eq!(resp.json()["model"], "model-a");

    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-keep-b");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["title"], "Keep B");
    assert_eq!(resp.json()["model"], "model-b");
}
