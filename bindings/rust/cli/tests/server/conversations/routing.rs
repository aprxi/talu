//! Tests for route matching edge cases on `/v1/conversations` endpoints.

use super::{conversation_config, no_bucket_config, seed_session};
use crate::server::common::*;
use serde_json::json;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Wrong HTTP methods on valid paths
// ---------------------------------------------------------------------------

#[test]
fn put_on_conversation_returns_not_found() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-put", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = send_request(ctx.addr(), "PUT", "/v1/conversations/sess-put", &[], None);
    // PUT is not a recognized method for conversations — falls through to default handler
    assert!(
        resp.status == 404 || resp.status == 501,
        "PUT on conversation should be 404 or 501, got: {}",
        resp.status
    );
}

#[test]
fn post_on_conversation_without_fork_returns_not_found() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-post", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-post",
        &json!({"data": "something"}),
    );
    // POST without /fork suffix is not routed to any handler
    assert!(
        resp.status == 404 || resp.status == 501,
        "POST on conversation (no /fork) should be 404 or 501, got: {}",
        resp.status
    );
}

#[test]
fn get_on_fork_path_returns_not_found() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-get-fork", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    // GET on /fork path — GET handler explicitly excludes paths ending in /fork
    let resp = get(ctx.addr(), "/v1/conversations/sess-get-fork/fork");
    assert!(
        resp.status == 404 || resp.status == 501,
        "GET on fork path should be 404 or 501, got: {}",
        resp.status
    );
}

#[test]
fn delete_on_list_path_returns_not_found() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-x", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    // DELETE on /v1/conversations (list endpoint, no ID) — no route matches
    let resp = delete(ctx.addr(), "/v1/conversations");
    assert!(
        resp.status == 404 || resp.status == 501,
        "DELETE on list path should be 404 or 501, got: {}",
        resp.status
    );
}

#[test]
fn patch_on_list_path_returns_not_found() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-x", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    // PATCH on /v1/conversations (list endpoint) — no route matches
    let resp = patch_json(ctx.addr(), "/v1/conversations", &json!({"title": "x"}));
    assert!(
        resp.status == 404 || resp.status == 501,
        "PATCH on list path should be 404 or 501, got: {}",
        resp.status
    );
}

#[test]
fn post_on_list_path_returns_not_found() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-x", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = post_json(ctx.addr(), "/v1/conversations", &json!({}));
    assert!(
        resp.status == 404 || resp.status == 501,
        "POST on list path should be 404 or 501, got: {}",
        resp.status
    );
}

// ---------------------------------------------------------------------------
// Path confusion edge cases
// ---------------------------------------------------------------------------

#[test]
fn conversation_id_with_special_characters() {
    let temp = TempDir::new().expect("temp dir");
    // Session IDs with dashes, underscores, dots — all common in real usage
    seed_session(temp.path(), "sess-with-dashes", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations/sess-with-dashes");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["id"], "sess-with-dashes");
}

#[test]
fn conversation_id_with_uuid_format() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(
        temp.path(),
        "123e4567-e89b-12d3-a456-426614174000",
        "Chat",
        "model",
    );

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(
        ctx.addr(),
        "/v1/conversations/123e4567-e89b-12d3-a456-426614174000",
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.json()["id"], "123e4567-e89b-12d3-a456-426614174000");
}

#[test]
fn conversation_id_named_fork_is_valid() {
    // A session ID literally named "fork" — the GET handler excludes paths
    // ending in /fork, so GET /v1/conversations/fork would not hit handle_get.
    // This is a known route matching quirk.
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "fork", "Chat named fork", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    // GET /v1/conversations/fork ends with /fork so the GET handler skips it
    let resp = get(ctx.addr(), "/v1/conversations/fork");
    // This goes to the fallthrough — it will be 404 or 501, NOT a GET on "fork"
    assert!(
        resp.status != 200,
        "GET /v1/conversations/fork should NOT match the GET handler (conflicts with fork route)"
    );
}

// ---------------------------------------------------------------------------
// Content-Type header validation on responses
// ---------------------------------------------------------------------------

#[test]
fn list_response_has_json_content_type() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-ct", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations");
    assert_eq!(resp.status, 200);
    let ct = resp.header("content-type");
    assert_eq!(ct, Some("application/json"), "response content-type");
}

#[test]
fn get_response_has_json_content_type() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-ct2", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations/sess-ct2");
    assert_eq!(resp.status, 200);
    let ct = resp.header("content-type");
    assert_eq!(ct, Some("application/json"));
}

#[test]
fn error_response_has_json_content_type() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-x", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations/nonexistent");
    assert_eq!(resp.status, 404);
    let ct = resp.header("content-type");
    assert_eq!(ct, Some("application/json"), "error response content-type");
}

// ---------------------------------------------------------------------------
// Ambiguous fork-path routing
// ---------------------------------------------------------------------------

#[test]
fn delete_on_fork_path_deletes_session() {
    // DELETE /v1/conversations/sess-id/fork — the route matches PATCH/DELETE since
    // they use starts_with("/v1/conversations/"). The conversation ID extracted will be "sess-id"
    // (extract_conversation_id takes up to the first '/').
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-del-fork", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = delete(ctx.addr(), "/v1/conversations/sess-del-fork/fork");
    // DELETE matches starts_with("/v1/conversations/") so it goes to handle_delete.
    // extract_conversation_id("/v1/conversations/sess-del-fork/fork") returns "sess-del-fork"
    assert_eq!(resp.status, 204);

    // Session should be deleted
    let resp = get(ctx.addr(), "/v1/conversations/sess-del-fork");
    assert_eq!(resp.status, 404);
}

#[test]
fn patch_on_fork_path_patches_session() {
    // PATCH /v1/conversations/sess-id/fork — similarly matches PATCH handler
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-patch-fork", "Original", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = patch_json(
        ctx.addr(),
        "/v1/conversations/sess-patch-fork/fork",
        &json!({"title": "Patched via fork path"}),
    );
    // PATCH matches starts_with("/v1/conversations/") so it goes to handle_patch.
    // extract_conversation_id returns "sess-patch-fork"
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["title"], "Patched via fork path");
}

// ---------------------------------------------------------------------------
// Trailing slash edge cases
// ---------------------------------------------------------------------------

#[test]
fn list_with_trailing_slash() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-ts1", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    // /v1/conversations/ (trailing slash) — the exact match is "/v1/conversations", so
    // /v1/conversations/ goes to the GET handler with starts_with check and
    // extract_conversation_id gets empty string → returns 400 or goes to GET handler
    let resp = get(ctx.addr(), "/v1/conversations/");
    // This either returns the list (if trailing-slash is normalized) or
    // goes to GET handler with empty ID → 400
    assert!(
        resp.status == 200 || resp.status == 400,
        "trailing slash on list: status {} body: {}",
        resp.status,
        resp.body,
    );
}

// ---------------------------------------------------------------------------
// 503 when no bucket configured
// ---------------------------------------------------------------------------

#[test]
fn list_503_has_json_content_type() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = get(ctx.addr(), "/v1/conversations");
    assert_eq!(resp.status, 503);
    let ct = resp.header("content-type");
    assert_eq!(ct, Some("application/json"), "503 response content-type");
}

// ---------------------------------------------------------------------------
// Error content-type
// ---------------------------------------------------------------------------

#[test]
fn fork_400_has_json_content_type() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-400ct", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-400ct/fork",
        &json!({}),
    );
    assert_eq!(resp.status, 400);
    assert_eq!(resp.header("content-type"), Some("application/json"));
}

#[test]
fn patch_content_type_on_200() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-pct", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = patch_json(
        ctx.addr(),
        "/v1/conversations/sess-pct",
        &json!({"title": "New"}),
    );
    assert_eq!(resp.status, 200);
    assert_eq!(resp.header("content-type"), Some("application/json"));
}

#[test]
fn fork_201_has_json_content_type() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fct", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fct/fork",
        &json!({"target_item_id": 0}),
    );
    assert_eq!(resp.status, 201);
    assert_eq!(resp.header("content-type"), Some("application/json"));
}
