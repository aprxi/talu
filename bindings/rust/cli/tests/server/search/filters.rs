//! Integration tests for search filters in `POST /v1/search`.

use super::{search_config, seed_session, seed_session_with_marker, seed_session_with_tags};
use crate::server::common::*;
use serde_json::json;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Model filter tests
// ---------------------------------------------------------------------------

/// Model filter with exact match (case-insensitive).
#[test]
fn filter_model_exact_match() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-a", "Chat A", "qwen3-0.6b");
    seed_session(temp.path(), "sess-b", "Chat B", "llama-3");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": { "model": "qwen3-0.6b" }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["model"], "qwen3-0.6b");
}

/// Model filter with wildcard suffix (qwen*).
#[test]
fn filter_model_wildcard_suffix() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-a", "Chat A", "qwen3-0.6b");
    seed_session(temp.path(), "sess-b", "Chat B", "qwen2-1.5b");
    seed_session(temp.path(), "sess-c", "Chat C", "llama-3");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": { "model": "qwen*" }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2);
}

/// Model filter with wildcard prefix (*llama).
#[test]
fn filter_model_wildcard_prefix() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-a", "Chat A", "meta-llama");
    seed_session(temp.path(), "sess-b", "Chat B", "qwen3-0.6b");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": { "model": "*llama" }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["model"], "meta-llama");
}

/// Model filter with wildcard both sides (*llama*).
#[test]
fn filter_model_wildcard_contains() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-a", "Chat A", "meta-llama-3");
    seed_session(temp.path(), "sess-b", "Chat B", "llama-2");
    seed_session(temp.path(), "sess-c", "Chat C", "qwen3-0.6b");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": { "model": "*llama*" }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2);
}

/// Model filter is case-insensitive.
#[test]
fn filter_model_case_insensitive() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-a", "Chat A", "Qwen3-0.6B");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": { "model": "qwen*" }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
}

// ---------------------------------------------------------------------------
// Marker filter tests
// ---------------------------------------------------------------------------

/// Marker filter with exact match.
#[test]
fn filter_marker_exact_match() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_marker(temp.path(), "sess-a", "Chat A", "gpt-4", "pinned");
    seed_session_with_marker(temp.path(), "sess-b", "Chat B", "gpt-4", "archived");
    seed_session(temp.path(), "sess-c", "Chat C", "gpt-4"); // default "active"

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": { "marker": "pinned" }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "sess-a");
}

/// Marker filter with OR logic (marker_any).
#[test]
fn filter_marker_any_or_logic() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_marker(temp.path(), "sess-a", "Chat A", "gpt-4", "pinned");
    seed_session_with_marker(temp.path(), "sess-b", "Chat B", "gpt-4", "archived");
    seed_session_with_marker(temp.path(), "sess-c", "Chat C", "gpt-4", "active");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": { "marker_any": ["pinned", "archived"] }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2);
}

// ---------------------------------------------------------------------------
// has_tags filter tests
// ---------------------------------------------------------------------------

/// has_tags=true returns only sessions with tags.
#[test]
fn filter_has_tags_true() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags(
        temp.path(),
        "sess-tagged",
        "Tagged Chat",
        "gpt-4",
        &["work"],
    );
    seed_session(temp.path(), "sess-untagged", "Untagged Chat", "gpt-4");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": { "has_tags": true }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "sess-tagged");
}

/// has_tags=false returns only sessions without tags.
#[test]
fn filter_has_tags_false() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags(
        temp.path(),
        "sess-tagged",
        "Tagged Chat",
        "gpt-4",
        &["work"],
    );
    seed_session(temp.path(), "sess-untagged", "Untagged Chat", "gpt-4");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": { "has_tags": false }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "sess-untagged");
}

// ---------------------------------------------------------------------------
// Timestamp filter tests
// ---------------------------------------------------------------------------

/// created_after filter excludes older sessions.
#[test]
fn filter_created_after() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-a", "Chat A", "gpt-4");
    // Small delay to ensure different timestamps
    std::thread::sleep(std::time::Duration::from_millis(50));
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;
    std::thread::sleep(std::time::Duration::from_millis(50));
    seed_session(temp.path(), "sess-b", "Chat B", "gpt-4");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": { "created_after": now }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "sess-b");
}

/// created_before filter excludes newer sessions.
#[test]
fn filter_created_before() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-a", "Chat A", "gpt-4");
    std::thread::sleep(std::time::Duration::from_millis(50));
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;
    std::thread::sleep(std::time::Duration::from_millis(50));
    seed_session(temp.path(), "sess-b", "Chat B", "gpt-4");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": { "created_before": now }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "sess-a");
}

// ---------------------------------------------------------------------------
// Combined filters
// ---------------------------------------------------------------------------

/// Multiple filters are combined with AND logic.
#[test]
fn filter_combined_and_logic() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_marker(temp.path(), "sess-a", "Chat A", "qwen3-0.6b", "pinned");
    seed_session_with_marker(temp.path(), "sess-b", "Chat B", "qwen3-0.6b", "active");
    seed_session_with_marker(temp.path(), "sess-c", "Chat C", "llama-3", "pinned");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": {
                "model": "qwen*",
                "marker": "pinned"
            }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "sess-a");
}

/// Text search combined with filters.
#[test]
fn filter_with_text_search() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_marker(
        temp.path(),
        "sess-a",
        "Rust programming",
        "qwen3-0.6b",
        "pinned",
    );
    seed_session_with_marker(
        temp.path(),
        "sess-b",
        "Python scripting",
        "qwen3-0.6b",
        "pinned",
    );
    seed_session_with_marker(temp.path(), "sess-c", "Rust basics", "llama-3", "active");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "text": "rust",
            "filters": {
                "marker": "pinned"
            }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "sess-a");
}

// ---------------------------------------------------------------------------
// Empty array edge case tests (regression tests)
// ---------------------------------------------------------------------------

/// Empty tags array should return all sessions (not none).
///
/// This is a regression test for a bug where `tags: []` was interpreted as
/// "filter by nothing" rather than "no filter". The correct behavior is that
/// an empty array means no filter is applied.
#[test]
fn filter_tags_empty_array_returns_all() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags(
        temp.path(),
        "sess-tagged",
        "Tagged Chat",
        "gpt-4",
        &["work"],
    );
    seed_session(temp.path(), "sess-untagged", "Untagged Chat", "gpt-4");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": { "tags": [] }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    // Empty tags array = no filter = return all sessions
    assert_eq!(data.len(), 2, "empty tags array should return all sessions");
}

/// Empty tags_any array should return all sessions (not none).
///
/// This is a regression test for a bug where `tags_any: []` was interpreted as
/// "filter by nothing" rather than "no filter".
#[test]
fn filter_tags_any_empty_array_returns_all() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags(
        temp.path(),
        "sess-tagged",
        "Tagged Chat",
        "gpt-4",
        &["work"],
    );
    seed_session(temp.path(), "sess-untagged", "Untagged Chat", "gpt-4");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": { "tags_any": [] }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    // Empty tags_any array = no filter = return all sessions
    assert_eq!(
        data.len(),
        2,
        "empty tags_any array should return all sessions"
    );
}

/// Empty marker_any array should return all sessions (not none).
///
/// This is a regression test for a bug where `marker_any: []` was interpreted as
/// "filter by nothing" rather than "no filter". The correct behavior is that
/// an empty array means no filter is applied.
#[test]
fn filter_marker_any_empty_array_returns_all() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_marker(temp.path(), "sess-a", "Chat A", "gpt-4", "pinned");
    seed_session_with_marker(temp.path(), "sess-b", "Chat B", "gpt-4", "archived");
    seed_session_with_marker(temp.path(), "sess-c", "Chat C", "gpt-4", "active");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": { "marker_any": [] }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    // Empty marker_any array = no filter = return all sessions
    assert_eq!(
        data.len(),
        3,
        "empty marker_any array should return all sessions"
    );
}
