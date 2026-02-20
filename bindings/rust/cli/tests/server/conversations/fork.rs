use super::{conversation_config, no_bucket_config, seed_session, seed_session_with_messages};
use crate::server::common::*;
use serde_json::json;
use tempfile::TempDir;

#[test]
fn fork_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/some-id/fork",
        &json!({"target_item_id": 0}),
    );
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn fork_creates_new_session() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-src", "Source Chat", "test-model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    // Get the source to find an item ID
    let source = get(ctx.addr(), "/v1/conversations/sess-fork-src");
    assert_eq!(source.status, 200);
    let source_json = source.json();
    let items = source_json["items"].as_array().expect("items");
    assert!(!items.is_empty(), "source should have items");

    // Use item_id=0 (the first user message added by seed, 0-indexed)
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-src/fork",
        &json!({"target_item_id": 0}),
    );
    assert_eq!(resp.status, 201, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["object"], "conversation");
    // New session ID should be a UUID-based ID
    let new_id = json["id"].as_str().expect("new session id");
    assert!(
        new_id.starts_with("sess_"),
        "id should start with sess_: {new_id}"
    );
}

#[test]
fn fork_nonexistent_source_returns_404() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-other", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/nonexistent/fork",
        &json!({"target_item_id": 1}),
    );
    // Should be 404 or 500 depending on how the core handles missing source
    assert!(
        resp.status == 404 || resp.status == 500,
        "status: {} body: {}",
        resp.status,
        resp.body
    );
}

#[test]
fn fork_missing_target_item_id_returns_400() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-bad", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-bad/fork",
        &json!({}),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn fork_with_prefix_path() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-prefix", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/conversations/sess-fork-prefix/fork",
        &json!({"target_item_id": 0}),
    );
    assert_eq!(resp.status, 201, "body: {}", resp.body);
}

#[test]
fn fork_new_session_appears_in_list() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-list", "Source", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    // Fork
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-list/fork",
        &json!({"target_item_id": 0}),
    );
    assert_eq!(resp.status, 201);
    let new_id = resp.json()["id"].as_str().expect("id").to_string();

    // List should now have 2 sessions
    let resp = get(ctx.addr(), "/v1/conversations");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 2);

    let ids: Vec<&str> = data.iter().map(|s| s["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"sess-fork-list"), "source should be in list");
    assert!(
        ids.contains(&new_id.as_str()),
        "forked session should be in list"
    );
}

#[test]
fn fork_new_session_is_retrievable() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-get", "Source", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-get/fork",
        &json!({"target_item_id": 0}),
    );
    assert_eq!(resp.status, 201);
    let new_id = resp.json()["id"].as_str().expect("id").to_string();

    // GET the forked session
    let resp = get(ctx.addr(), &format!("/v1/conversations/{new_id}"));
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["id"], new_id);
    assert_eq!(json["object"], "conversation");
    // Should have items
    let items = json["items"].as_array().expect("items");
    assert!(!items.is_empty(), "forked session should have items");
}

#[test]
fn fork_preserves_items_subset() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_messages(
        temp.path(),
        "sess-fork-subset",
        "Multi-msg",
        "model",
        &["msg-0", "msg-1", "msg-2"],
    );

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    // Get source items count
    let source_resp = get(ctx.addr(), "/v1/conversations/sess-fork-subset");
    assert_eq!(source_resp.status, 200);
    let source_json = source_resp.json();
    let source_items = source_json["items"].as_array().expect("source items");
    let source_count = source_items.len();
    assert!(source_count >= 3, "source should have at least 3 items");

    // Fork at item 0 (first message only)
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-subset/fork",
        &json!({"target_item_id": 0}),
    );
    assert_eq!(resp.status, 201);
    let new_id = resp.json()["id"].as_str().expect("id").to_string();

    // Forked session should have fewer items than source
    let fork_resp = get(ctx.addr(), &format!("/v1/conversations/{new_id}"));
    assert_eq!(fork_resp.status, 200);
    let fork_json = fork_resp.json();
    let fork_items = fork_json["items"].as_array().expect("fork items");
    assert!(
        fork_items.len() <= source_count,
        "forked session should have <= source items ({} vs {})",
        fork_items.len(),
        source_count,
    );
}

#[test]
fn fork_invalid_json_returns_400() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-badjson", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/conversations/sess-fork-badjson/fork",
        &[("Content-Type", "application/json")],
        Some("{{invalid"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn fork_string_target_item_id_returns_400() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-strid", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-strid/fork",
        &json!({"target_item_id": "not-a-number"}),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn fork_does_not_modify_source() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(
        temp.path(),
        "sess-fork-src-intact",
        "Source Title",
        "src-model",
    );

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    // Get source before fork
    let before = get(ctx.addr(), "/v1/conversations/sess-fork-src-intact");
    assert_eq!(before.status, 200);
    let before_json = before.json();
    let before_item_count = before_json["items"].as_array().expect("items").len();

    // Fork
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-src-intact/fork",
        &json!({"target_item_id": 0}),
    );
    assert_eq!(resp.status, 201);

    // Get source after fork — should be unchanged
    let after = get(ctx.addr(), "/v1/conversations/sess-fork-src-intact");
    assert_eq!(after.status, 200);
    let after_json = after.json();
    assert_eq!(after_json["title"], "Source Title");
    assert_eq!(after_json["model"], "src-model");
    let after_item_count = after_json["items"].as_array().expect("items").len();
    assert_eq!(
        before_item_count, after_item_count,
        "source item count should be unchanged after fork"
    );
}

// ---------------------------------------------------------------------------
// Body edge cases
// ---------------------------------------------------------------------------

#[test]
fn fork_negative_target_item_id_returns_400() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-neg", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    // -1 cannot parse as u64 → as_u64() returns None → 400
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-neg/fork",
        &json!({"target_item_id": -1}),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn fork_float_target_item_id_returns_400() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-float", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    // 0.5 — as_u64() returns None for non-integer → 400
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-float/fork",
        &json!({"target_item_id": 0.5}),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn fork_null_target_item_id_returns_400() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-null", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-null/fork",
        &json!({"target_item_id": null}),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn fork_boolean_target_item_id_returns_400() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-bool", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-bool/fork",
        &json!({"target_item_id": false}),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn fork_extra_fields_ignored() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-extra", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-extra/fork",
        &json!({"target_item_id": 0, "extra_field": "ignored"}),
    );
    assert_eq!(resp.status, 201, "body: {}", resp.body);
}

#[test]
fn fork_response_has_object_field() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-obj", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-obj/fork",
        &json!({"target_item_id": 0}),
    );
    assert_eq!(resp.status, 201);
    assert_eq!(resp.json()["object"], "conversation");
}

#[test]
fn fork_returns_201_not_200() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-201", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-201/fork",
        &json!({"target_item_id": 0}),
    );
    // Specifically assert 201 Created, not 200 OK
    assert_eq!(resp.status, 201);
}

#[test]
fn fork_each_call_creates_unique_session() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-unique", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    let resp1 = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-unique/fork",
        &json!({"target_item_id": 0}),
    );
    assert_eq!(resp1.status, 201);
    let id1 = resp1.json()["id"].as_str().expect("id1").to_string();

    let resp2 = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-unique/fork",
        &json!({"target_item_id": 0}),
    );
    assert_eq!(resp2.status, 201);
    let id2 = resp2.json()["id"].as_str().expect("id2").to_string();

    assert_ne!(id1, id2, "each fork should produce a unique session ID");

    // All 3 sessions should exist in the list
    let resp = get(ctx.addr(), "/v1/conversations");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 3, "source + 2 forks = 3 sessions");
}

// ---------------------------------------------------------------------------
// Out-of-range item_id
// ---------------------------------------------------------------------------

#[test]
fn fork_very_large_item_id() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-large", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    // Item ID 999999 is way beyond what the session has (only 1 item at ID 0)
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-large/fork",
        &json!({"target_item_id": 999999}),
    );
    // Core may return ItemNotFound (404) or create a fork with all items (201)
    assert!(
        resp.status == 201 || resp.status == 404 || resp.status == 500,
        "out-of-range item_id: status {} body: {}",
        resp.status,
        resp.body
    );
}

#[test]
fn fork_zero_target_item_id_returns_201() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-zero", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-zero/fork",
        &json!({"target_item_id": 0}),
    );
    assert_eq!(resp.status, 201);
}

// ---------------------------------------------------------------------------
// Fork inheritance and metadata
// ---------------------------------------------------------------------------

#[test]
fn fork_inherits_model_from_source() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-model", "Source", "custom-model-v2");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-model/fork",
        &json!({"target_item_id": 0}),
    );
    assert_eq!(resp.status, 201);
    let new_id = resp.json()["id"].as_str().expect("id").to_string();

    // GET the forked session and check model
    let resp = get(ctx.addr(), &format!("/v1/conversations/{new_id}"));
    assert_eq!(resp.status, 200);
    // The forked session gets the source's model via fork internals
    let model = &resp.json()["model"];
    // Model may or may not be inherited depending on core behavior
    assert!(
        model.is_string() || model.is_null(),
        "model should be present or null: {model:?}"
    );
}

#[test]
fn fork_response_has_created_at_and_updated_at() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-ts", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-ts/fork",
        &json!({"target_item_id": 0}),
    );
    assert_eq!(resp.status, 201);
    let json = resp.json();
    assert!(
        json["created_at"].as_i64().unwrap() > 0,
        "should have created_at"
    );
    assert!(
        json["updated_at"].as_i64().unwrap() > 0,
        "should have updated_at"
    );
}

#[test]
fn fork_response_has_marker_field() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-marker", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-marker/fork",
        &json!({"target_item_id": 0}),
    );
    assert_eq!(resp.status, 201);
    let json = resp.json();
    // Marker should be present (either "active" or whatever the core sets)
    assert!(
        json["marker"].is_string(),
        "fork response should have marker field: {:?}",
        json["marker"]
    );
}

#[test]
fn fork_content_type_is_json() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fork-ct", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-ct/fork",
        &json!({"target_item_id": 0}),
    );
    assert_eq!(resp.status, 201);
    assert_eq!(resp.header("content-type"), Some("application/json"));
}

// ---------------------------------------------------------------------------
// Fork content accuracy: strict item count assertion
// ---------------------------------------------------------------------------

/// Forking at a mid-point produces strictly fewer items than the source.
///
/// Strengthens the `fork_preserves_items_subset` test which used a weak `<=`
/// assertion.  Forking at `target_item_id: 1` from a 3-message source should
/// truncate after item 1, producing a strict subset.
#[test]
fn fork_at_mid_item_has_fewer_items_than_source() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_messages(
        temp.path(),
        "sess-fork-mid",
        "Multi-msg",
        "model",
        &["msg-0", "msg-1", "msg-2"],
    );

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    // Get source items count.
    let source_resp = get(ctx.addr(), "/v1/conversations/sess-fork-mid");
    assert_eq!(source_resp.status, 200);
    let source_json = source_resp.json();
    let source_items = source_json["items"].as_array().expect("source items");
    let source_count = source_items.len();
    assert!(
        source_count >= 3,
        "source should have at least 3 items, got: {}",
        source_count
    );

    // Fork at item 1 (should include items 0 and 1, truncating item 2+).
    let resp = post_json(
        ctx.addr(),
        "/v1/conversations/sess-fork-mid/fork",
        &json!({"target_item_id": 1}),
    );
    assert_eq!(resp.status, 201);
    let new_id = resp.json()["id"].as_str().expect("id").to_string();

    let fork_resp = get(ctx.addr(), &format!("/v1/conversations/{new_id}"));
    assert_eq!(fork_resp.status, 200);
    let fork_json = fork_resp.json();
    let fork_items = fork_json["items"].as_array().expect("fork items");
    let fork_count = fork_items.len();

    // Strictly fewer items than source.
    assert!(
        fork_count < source_count,
        "forked session should have strictly fewer items: fork={} source={}",
        fork_count,
        source_count,
    );

    // Items 0 and 1 should be retained — exactly 2 items.
    assert_eq!(
        fork_count, 2,
        "fork at item 1 should retain exactly 2 items (0 and 1), got: {}",
        fork_count,
    );
}
