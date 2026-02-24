//! Integration tests for text mode in `POST /v1/search`.

use super::{
    no_bucket_config, search_config, seed_session, seed_session_with_group,
    seed_session_with_marker, seed_session_with_messages, seed_session_with_project,
    seed_session_with_system_prompt, seed_session_with_tags,
};
use crate::server::common::*;
use serde_json::json;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Basic text search
// ---------------------------------------------------------------------------

/// Search by title substring (case-insensitive).
#[test]
fn text_search_by_title() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-rust", "Rust programming", "gpt-4");
    seed_session(temp.path(), "sess-py", "Python scripting", "gpt-4");
    seed_session(temp.path(), "sess-go", "Go concurrency", "gpt-4");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "text": "rust"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["title"], "Rust programming");
}

/// Search by model substring.
#[test]
fn text_search_by_model() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-a", "Chat A", "qwen3-0.6b");
    seed_session(temp.path(), "sess-b", "Chat B", "llama-3");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "text": "qwen"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["model"], "qwen3-0.6b");
}

/// Search by system_prompt substring.
#[test]
fn text_search_by_system_prompt() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_system_prompt(
        temp.path(),
        "sess-code",
        "Code Help",
        "gpt-4",
        "You are a helpful coding assistant",
    );
    seed_session_with_system_prompt(
        temp.path(),
        "sess-math",
        "Math Help",
        "gpt-4",
        "You are a math tutor",
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "text": "coding"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["title"], "Code Help");
}

// ---------------------------------------------------------------------------
// Case insensitivity
// ---------------------------------------------------------------------------

/// Search is case-insensitive.
#[test]
fn text_search_case_insensitive() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-upper", "IMPORTANT Meeting", "m");
    seed_session(temp.path(), "sess-lower", "casual chat", "m");

    let ctx = ServerTestContext::new(search_config(temp.path()));

    // Lowercase query matches uppercase title
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "text": "important"
        }),
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["title"], "IMPORTANT Meeting");

    // Uppercase query matches lowercase title
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "text": "CASUAL"
        }),
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["title"], "casual chat");
}

// ---------------------------------------------------------------------------
// No match
// ---------------------------------------------------------------------------

/// Query that matches nothing returns empty list.
#[test]
fn text_search_no_match() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-a", "Chat about Rust", "gpt-4");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "text": "nonexistent"
        }),
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(data.is_empty());
    assert_eq!(json["has_more"], false);
}

// ---------------------------------------------------------------------------
// Filter-only search (no text)
// ---------------------------------------------------------------------------

/// No search mode (filter-only) returns all.
#[test]
fn filter_only_returns_all() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-a", "Alpha", "m");
    seed_session(temp.path(), "sess-b", "Beta", "m");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions"
        }),
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2);
}

// ---------------------------------------------------------------------------
// Search + pagination
// ---------------------------------------------------------------------------

/// Search with limit returns correct page size and has_more.
#[test]
fn text_search_with_limit() {
    let temp = TempDir::new().expect("temp dir");
    for i in 0..5 {
        seed_session(
            temp.path(),
            &format!("sess-rust-{i}"),
            &format!("Rust topic {i}"),
            "m",
        );
    }
    seed_session(temp.path(), "sess-py", "Python topic", "m");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "text": "rust",
            "limit": 2
        }),
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2);
    assert_eq!(json["has_more"], true);

    // All returned sessions should match the query
    for item in data {
        let title = item["title"].as_str().unwrap();
        assert!(
            title.to_lowercase().contains("rust"),
            "returned session should match query, got: {title}",
        );
    }
}

/// Search + cursor pagination covers all matching results.
#[test]
fn text_search_pagination_covers_all() {
    let temp = TempDir::new().expect("temp dir");
    for i in 0..4 {
        seed_session(
            temp.path(),
            &format!("sess-match-{i}"),
            &format!("Matching {i}"),
            "m",
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    // Non-matching session
    seed_session(temp.path(), "sess-other", "Other topic", "m");

    let ctx = ServerTestContext::new(search_config(temp.path()));

    let mut all_ids: Vec<String> = Vec::new();
    let mut cursor: Option<String> = None;

    for _ in 0..10 {
        let mut req = json!({
            "scope": "sessions",
            "text": "matching",
            "limit": 2
        });
        if let Some(c) = cursor.as_ref() {
            req["cursor"] = json!(c);
        }

        let resp = post_json(ctx.addr(), "/v1/search", &req);
        assert_eq!(resp.status, 200);
        let json = resp.json();
        let data = json["data"].as_array().expect("data array");

        for item in data {
            all_ids.push(item["id"].as_str().unwrap().to_string());
        }

        if json["has_more"] == false {
            break;
        }
        cursor = json["cursor"].as_str().map(String::from);
    }

    assert_eq!(all_ids.len(), 4, "should find all 4 matching sessions");
}

// ---------------------------------------------------------------------------
// Search + group_id filter
// ---------------------------------------------------------------------------

/// Search combined with group_id filters by both.
#[test]
fn text_search_with_group_id_filter() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_group(temp.path(), "sess-g1-rust", "Rust in group1", "m", "group1");
    seed_session_with_group(temp.path(), "sess-g2-rust", "Rust in group2", "m", "group2");
    seed_session_with_group(temp.path(), "sess-g1-py", "Python in group1", "m", "group1");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "text": "rust",
            "filters": {
                "group_id": "group1"
            }
        }),
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1, "should only match Rust in group1");
    assert_eq!(data[0]["id"], "sess-g1-rust");
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Invalid scope returns error.
#[test]
fn invalid_scope_returns_error() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "invalid"
        }),
    );
    assert_eq!(resp.status, 400);
    let json = resp.json();
    assert_eq!(json["error"]["code"], "invalid_scope");
}

/// Multiple search modes returns error.
#[test]
fn multiple_search_modes_returns_error() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "text": "hello",
            "regex": "world"
        }),
    );
    assert_eq!(resp.status, 400);
    let json = resp.json();
    assert_eq!(json["error"]["code"], "invalid_search_mode");
}

/// Regex mode returns not implemented.
#[test]
fn regex_mode_not_implemented() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "regex": "hello.*world"
        }),
    );
    assert_eq!(resp.status, 501);
    let json = resp.json();
    assert_eq!(json["error"]["code"], "not_implemented");
}

/// Vector mode returns not implemented.
#[test]
fn vector_mode_not_implemented() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "vector": {
                "text": "semantic query"
            }
        }),
    );
    assert_eq!(resp.status, 501);
    let json = resp.json();
    assert_eq!(json["error"]["code"], "not_implemented");
}

/// Items scope searches within message content.
#[test]
fn items_scope_searches_message_content() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_messages(
        temp.path(),
        "sess-1",
        "Chat about Rust",
        "m",
        &["I love programming in Rust because it's safe"],
    );
    seed_session_with_messages(
        temp.path(),
        "sess-2",
        "Python Chat",
        "m",
        &["Python is great for scripting"],
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "items",
            "text": "Rust"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["session_id"], "sess-1");
    assert!(data[0]["snippet"]
        .as_str()
        .unwrap()
        .to_lowercase()
        .contains("rust"));
}

/// Items scope with highlight returns **highlighted** snippets.
#[test]
fn items_scope_highlight() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_messages(
        temp.path(),
        "sess-1",
        "Chat",
        "m",
        &["I love Rust programming"],
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "items",
            "text": "Rust",
            "highlight": true
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    // Highlighted text should contain **Rust**
    let snippet = data[0]["snippet"].as_str().unwrap();
    assert!(
        snippet.contains("**Rust**"),
        "snippet should contain **Rust**: {}",
        snippet
    );
}

/// Items scope requires text parameter.
#[test]
fn items_scope_requires_text() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "items"
        }),
    );
    assert_eq!(resp.status, 400);
    let json = resp.json();
    assert_eq!(json["error"]["code"], "missing_text");
}

/// Items scope respects limit parameter.
#[test]
fn items_scope_with_limit() {
    let temp = TempDir::new().expect("temp dir");
    // Create multiple sessions with the keyword
    for i in 0..4 {
        seed_session_with_messages(
            temp.path(),
            &format!("sess-{i}"),
            &format!("Chat {i}"),
            "m",
            &[&format!("Message containing LIMITKEY {i}")],
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    let ctx = ServerTestContext::new(search_config(temp.path()));

    // Request with limit=2
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "items",
            "text": "LIMITKEY",
            "limit": 2
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2, "should respect limit=2");

    // Request without limit (default)
    let resp2 = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "items",
            "text": "LIMITKEY"
        }),
    );
    assert_eq!(resp2.status, 200, "body: {}", resp2.body);
    let json2 = resp2.json();
    let data2 = json2["data"].as_array().expect("data array");
    assert_eq!(data2.len(), 4, "should return all 4 items without limit");
}

// ---------------------------------------------------------------------------
// Content search (message body)
// ---------------------------------------------------------------------------

/// Search matches content in user messages.
#[test]
fn text_search_by_message_content() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_messages(
        temp.path(),
        "sess-1",
        "Generic Title",
        "m",
        &["This message contains a unique keyword: xyzzy123"],
    );
    seed_session_with_messages(
        temp.path(),
        "sess-2",
        "Another Chat",
        "m",
        &["Regular message without the keyword"],
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "text": "xyzzy123"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "sess-1");
}

/// Content search returns snippet for matching content.
#[test]
fn text_search_content_returns_snippet() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_messages(
        temp.path(),
        "sess-1",
        "Generic Title",
        "m",
        &["The error code ABC123 appeared in the logs"],
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "text": "ABC123"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);

    // Content match should include a snippet
    let snippet = &data[0]["search_snippet"];
    assert!(
        snippet.is_string(),
        "search_snippet should be present for content match"
    );
    let snippet_text = snippet.as_str().unwrap();
    assert!(
        snippet_text.to_lowercase().contains("abc123"),
        "snippet should contain the match: got {snippet_text}"
    );
}

/// Snippet is extracted from deep in a long message.
#[test]
fn text_search_snippet_deep_in_long_message() {
    let temp = TempDir::new().expect("temp dir");
    // Create a message with the keyword deep inside
    let prefix = "x".repeat(500);
    let long_msg = format!("{prefix} UNIQUE_MARKER_DEEP {prefix}");
    seed_session_with_messages(
        temp.path(),
        "sess-1",
        "Long Message Chat",
        "m",
        &[&long_msg],
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "text": "UNIQUE_MARKER_DEEP"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);

    let snippet = &data[0]["search_snippet"];
    assert!(snippet.is_string(), "search_snippet should be present");
    let snippet_text = snippet.as_str().unwrap();
    assert!(
        snippet_text.to_lowercase().contains("unique_marker_deep"),
        "snippet should contain the match"
    );
}

/// Snippet is extracted from second message in session.
#[test]
fn text_search_snippet_in_second_message() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_messages(
        temp.path(),
        "sess-1",
        "Multi Message Chat",
        "m",
        &[
            "First message without keyword",
            "Second message has TARGET_WORD here",
        ],
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "text": "TARGET_WORD"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);

    let snippet = &data[0]["search_snippet"];
    assert!(snippet.is_string(), "search_snippet should be present");
    let snippet_text = snippet.as_str().unwrap();
    assert!(
        snippet_text.to_lowercase().contains("target_word"),
        "snippet should contain the match from second message"
    );
}

/// No snippet for title-only match.
#[test]
fn text_search_no_snippet_for_title_match() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_messages(
        temp.path(),
        "sess-1",
        "UniqueTitle12345",
        "m",
        &["Generic message content"],
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "text": "UniqueTitle12345"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);

    // Title match should not have a snippet (snippet is for content matches)
    let snippet = &data[0]["search_snippet"];
    assert!(
        snippet.is_null(),
        "search_snippet should be null for title-only match, got: {snippet}"
    );
}

/// Content search with no match returns empty results.
#[test]
fn text_search_content_no_match() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_messages(
        temp.path(),
        "sess-1",
        "Test Chat",
        "m",
        &["Hello world", "Another message"],
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "text": "nonexistent_keyword_xyz"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(
        data.is_empty(),
        "should return no results for non-matching content search"
    );
}

/// Search matches title OR content (OR logic).
#[test]
fn text_search_matches_title_or_content() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_messages(
        temp.path(),
        "sess-title",
        "SearchKeyword in title",
        "m",
        &["Regular content"],
    );
    seed_session_with_messages(
        temp.path(),
        "sess-content",
        "Generic Title",
        "m",
        &["Content contains SearchKeyword here"],
    );
    seed_session_with_messages(
        temp.path(),
        "sess-neither",
        "Unrelated Chat",
        "m",
        &["Nothing matching here"],
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "text": "SearchKeyword"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2, "should match both title and content matches");

    let ids: Vec<&str> = data.iter().map(|d| d["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"sess-title"), "should include title match");
    assert!(
        ids.contains(&"sess-content"),
        "should include content match"
    );
}

// ---------------------------------------------------------------------------
// Storage disabled (no_bucket)
// ---------------------------------------------------------------------------

/// Search requires storage (no_bucket returns 503).
#[test]
fn search_no_bucket_returns_503() {
    let ctx = ServerTestContext::new(no_bucket_config());

    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "text": "anything"
        }),
    );
    assert_eq!(resp.status, 503);
    let json = resp.json();
    assert_eq!(json["error"]["code"], "no_storage");
}

/// Items scope search also requires storage (no_bucket returns 503).
#[test]
fn items_search_no_bucket_returns_503() {
    let ctx = ServerTestContext::new(no_bucket_config());

    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "items",
            "text": "anything"
        }),
    );
    assert_eq!(resp.status, 503);
    let json = resp.json();
    assert_eq!(json["error"]["code"], "no_storage");
}

// ---------------------------------------------------------------------------
// Unicode handling (regression tests)
// ---------------------------------------------------------------------------

/// Items search with Unicode content that has different byte lengths when lowercased.
///
/// This is a regression test for a bug where using byte positions from
/// `to_lowercase()` to slice the original text caused incorrect snippets
/// or panics. For example, Turkish İ (2 bytes) lowercases to "i̇" (3 bytes
/// with combining dot), causing all subsequent byte positions to be off.
#[test]
fn items_search_unicode_turkish_i() {
    let temp = TempDir::new().expect("temp dir");
    // Turkish İ (U+0130) has different byte length when lowercased
    seed_session_with_messages(
        temp.path(),
        "sess-unicode",
        "Unicode Chat",
        "m",
        &["İstanbul is a beautiful city and world famous"],
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));

    // Search for "world" which comes after the Turkish İ
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "items",
            "text": "world"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1, "should find the message containing 'world'");

    // The snippet should contain "world" without being corrupted
    let snippet = data[0]["snippet"].as_str().unwrap();
    assert!(
        snippet.to_lowercase().contains("world"),
        "snippet should contain 'world', got: {snippet}"
    );
}

/// Items search with highlight on Unicode content.
///
/// Verifies that highlighting works correctly with Unicode text that has
/// different byte lengths when lowercased.
#[test]
fn items_search_unicode_highlight() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_messages(
        temp.path(),
        "sess-unicode",
        "Unicode Chat",
        "m",
        &["İstanbul is beautiful and the world loves it"],
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "items",
            "text": "world",
            "highlight": true
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);

    // The highlighted snippet should contain **world**
    let snippet = data[0]["snippet"].as_str().unwrap();
    assert!(
        snippet.contains("**world**"),
        "snippet should contain **world**, got: {snippet}"
    );
}

/// Items search with Chinese characters.
///
/// Chinese characters are multi-byte UTF-8 but lowercase() is a no-op,
/// so this tests basic multi-byte handling.
#[test]
fn items_search_unicode_chinese() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_messages(
        temp.path(),
        "sess-chinese",
        "Chinese Chat",
        "m",
        &["你好世界 hello programming 再见"],
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));

    // Search for "programming" which is between Chinese characters
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "items",
            "text": "programming"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);

    let snippet = data[0]["snippet"].as_str().unwrap();
    assert!(
        snippet.contains("programming"),
        "snippet should contain 'programming', got: {snippet}"
    );
}

/// Items search matching Chinese characters themselves.
#[test]
fn items_search_unicode_chinese_match() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_messages(
        temp.path(),
        "sess-chinese",
        "Chinese Chat",
        "m",
        &["你好世界 means hello world in Chinese"],
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));

    // Search for Chinese characters
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "items",
            "text": "你好"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);

    let snippet = data[0]["snippet"].as_str().unwrap();
    assert!(
        snippet.contains("你好"),
        "snippet should contain '你好', got: {snippet}"
    );
}

// ---------------------------------------------------------------------------
// Pagination edge cases
// ---------------------------------------------------------------------------

/// limit=0 should be clamped to 1 (minimum limit).
#[test]
fn pagination_limit_zero_clamped_to_one() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-a", "Chat A", "m");
    seed_session(temp.path(), "sess-b", "Chat B", "m");
    seed_session(temp.path(), "sess-c", "Chat C", "m");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "limit": 0
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    // limit=0 should be clamped to 1, so we get exactly 1 result
    assert_eq!(data.len(), 1, "limit=0 should be clamped to 1");
}

/// Very large limit should be clamped to 100 (maximum limit).
#[test]
fn pagination_limit_large_clamped_to_max() {
    let temp = TempDir::new().expect("temp dir");
    // Create 5 sessions - we just need to verify clamping, not actually hit 100
    for i in 0..5 {
        seed_session(temp.path(), &format!("sess-{i}"), &format!("Chat {i}"), "m");
    }

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "limit": 99999
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    // Should return all 5 (under the 100 cap)
    assert_eq!(data.len(), 5);
}

/// Negative limit should return 400 (invalid JSON parse for usize).
#[test]
fn pagination_limit_negative_returns_400() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-a", "Chat A", "m");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "limit": -1
        }),
    );
    // Negative values can't deserialize to usize, should be 400 bad request
    assert_eq!(resp.status, 400, "negative limit should return 400");
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

/// Invalid scope returns 400.
#[test]
fn validation_invalid_scope_returns_400() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-a", "Chat A", "m");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "invalid_scope_xyz"
        }),
    );
    assert_eq!(resp.status, 400, "invalid scope should return 400");
    let json = resp.json();
    assert_eq!(json["error"]["code"], "invalid_scope");
}

/// Missing scope (invalid JSON structure) returns 400.
#[test]
fn validation_missing_scope_returns_400() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-a", "Chat A", "m");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "text": "hello"
        }),
    );
    // Missing required field "scope" should return 400
    assert_eq!(resp.status, 400, "missing scope should return 400");
}

/// Invalid JSON body returns 400.
#[test]
fn validation_invalid_json_returns_400() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(search_config(temp.path()));

    // Send malformed JSON using post_raw if available, or just test with wrong type
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": 123  // scope should be a string, not a number
        }),
    );
    assert_eq!(resp.status, 400, "invalid type for scope should return 400");
}

// ---------------------------------------------------------------------------
// Federated search (scope: "all")
// ---------------------------------------------------------------------------

/// Federated search returns both session and document results.
#[test]
fn federated_search_returns_both_sessions_and_documents() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(
        temp.path(),
        "sess-fed-1",
        "Quantum computing basics",
        "gpt-4",
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));

    // Create a document with "Quantum" in the title so both backends match.
    let doc_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &json!({
            "type": "prompt",
            "title": "Quantum physics notes",
            "content": {"text": "Quantum entanglement"}
        }),
    );
    assert_eq!(doc_resp.status, 201, "body: {}", doc_resp.body);

    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "all",
            "text": "Quantum"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(
        data.len() >= 2,
        "should have results from both backends, got {}",
        data.len()
    );

    let has_session = data.iter().any(|d| d["object"] == "session");
    let has_document = data.iter().any(|d| d["object"] == "document");
    assert!(has_session, "should include session results");
    assert!(has_document, "should include document results");
}

/// Federated search requires text field.
#[test]
fn federated_search_requires_text() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-fed-2", "Chat", "m");
    let ctx = ServerTestContext::new(search_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "all"
        }),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "missing_text");
}

/// Federated search respects limit.
#[test]
fn federated_search_respects_limit() {
    let temp = TempDir::new().expect("temp dir");
    // Seed several sessions with matching keyword.
    for i in 0..5 {
        seed_session(
            temp.path(),
            &format!("sess-fed-lim-{}", i),
            &format!("Federated topic {}", i),
            "gpt-4",
        );
    }

    let ctx = ServerTestContext::new(search_config(temp.path()));

    // Also create documents matching.
    for i in 0..5 {
        post_json(
            ctx.addr(),
            "/v1/db/tables/documents",
            &json!({
                "type": "note",
                "title": format!("Federated note {}", i),
                "content": {}
            }),
        );
    }

    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "all",
            "text": "Federated",
            "limit": 3
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(data.len() <= 3, "should respect limit, got {}", data.len());
    assert_eq!(json["has_more"], true);
}

/// Federated search works when only sessions match (no documents).
#[test]
fn federated_search_graceful_without_documents() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(
        temp.path(),
        "sess-fed-only",
        "Unique session topic",
        "gpt-4",
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "all",
            "text": "Unique session"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(!data.is_empty(), "should still return session results");
    assert!(data.iter().any(|d| d["object"] == "session"));
}

// ---------------------------------------------------------------------------
// Aggregations
// ---------------------------------------------------------------------------

/// Models aggregation returns counts grouped by model.
#[test]
fn aggregations_models_returns_counts() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "agg-m1", "Chat 1", "gpt-4");
    seed_session(temp.path(), "agg-m2", "Chat 2", "gpt-4");
    seed_session(temp.path(), "agg-m3", "Chat 3", "claude-3");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "aggregations": ["models"]
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let aggs = &json["aggregations"];
    assert!(aggs.is_object(), "aggregations should be an object");

    let models = aggs["models"].as_array().expect("models array");
    assert!(!models.is_empty(), "should have model entries");

    // Each entry should have value and count.
    for entry in models {
        assert!(
            entry["value"].is_string(),
            "model entry should have 'value'"
        );
        assert!(
            entry["count"].is_number(),
            "model entry should have 'count'"
        );
    }

    // Find gpt-4 entry — should have count 2.
    let gpt4 = models.iter().find(|e| e["value"] == "gpt-4");
    assert!(gpt4.is_some(), "should have gpt-4 entry");
    assert_eq!(gpt4.unwrap()["count"], 2);

    // Models should be sorted by count descending.
    let counts: Vec<u64> = models
        .iter()
        .map(|e| e["count"].as_u64().unwrap())
        .collect();
    let mut sorted = counts.clone();
    sorted.sort_by(|a, b| b.cmp(a));
    assert_eq!(counts, sorted, "models should be sorted by count desc");
}

/// Markers aggregation returns counts grouped by marker.
#[test]
fn aggregations_markers_returns_counts() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_marker(temp.path(), "agg-mk1", "Chat A", "m", "active");
    seed_session_with_marker(temp.path(), "agg-mk2", "Chat B", "m", "active");
    seed_session_with_marker(temp.path(), "agg-mk3", "Chat C", "m", "archived");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "aggregations": ["markers"]
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let markers = json["aggregations"]["markers"]
        .as_array()
        .expect("markers array");
    assert!(!markers.is_empty(), "should have marker entries");

    for entry in markers {
        assert!(entry["value"].is_string());
        assert!(entry["count"].is_number());
    }

    let active = markers.iter().find(|e| e["value"] == "active");
    assert!(active.is_some(), "should have active marker entry");
    assert_eq!(active.unwrap()["count"], 2);
}

/// Tags aggregation returns counts with id, name, count.
#[test]
fn aggregations_tags_returns_counts() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags(temp.path(), "agg-t1", "Chat T1", "m", &["work"]);
    seed_session_with_tags(temp.path(), "agg-t2", "Chat T2", "m", &["work"]);
    seed_session_with_tags(temp.path(), "agg-t3", "Chat T3", "m", &["personal"]);

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "aggregations": ["tags"]
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let tags = json["aggregations"]["tags"].as_array().expect("tags array");
    assert!(!tags.is_empty(), "should have tag entries");

    for entry in tags {
        assert!(entry["id"].is_string(), "tag entry should have 'id'");
        assert!(entry["name"].is_string(), "tag entry should have 'name'");
        assert!(entry["count"].is_number(), "tag entry should have 'count'");
    }

    let work = tags.iter().find(|e| e["name"] == "work");
    assert!(work.is_some(), "should have 'work' tag entry");
    assert_eq!(work.unwrap()["count"], 2);
}

/// Empty aggregations array returns no aggregations field.
#[test]
fn aggregations_empty_array_returns_no_aggregations() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "agg-e1", "Chat", "m");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "aggregations": []
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert!(
        json["aggregations"].is_null(),
        "empty aggregation array should yield null aggregations"
    );
}

/// Unknown aggregation type is silently skipped.
#[test]
fn aggregations_unknown_type_skipped() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "agg-u1", "Chat", "m");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "aggregations": ["nonexistent_type"]
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let aggs = &json["aggregations"];
    // Should be an object but without the unknown key.
    assert!(aggs.is_object(), "aggregations should be an object");
    assert!(
        aggs["nonexistent_type"].is_null(),
        "unknown type should not appear"
    );
}

/// Tags aggregation returns an empty array when no sessions have tags.
#[test]
fn aggregations_tags_empty_when_no_tags() {
    let temp = TempDir::new().expect("temp dir");
    // Seed sessions WITHOUT any tags.
    seed_session(temp.path(), "agg-no-tags-1", "Chat A", "m");
    seed_session(temp.path(), "agg-no-tags-2", "Chat B", "m");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "aggregations": ["tags"]
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let tags = &json["aggregations"]["tags"];
    assert!(
        tags.is_array(),
        "tags aggregation should be an array even when empty, got: {tags}"
    );
    assert_eq!(
        tags.as_array().unwrap().len(),
        0,
        "tags array should be empty when no sessions have tags"
    );
}

// ---------------------------------------------------------------------------
// Unicode: multi-byte characters in the search query itself
// ---------------------------------------------------------------------------

/// Search query containing multi-byte Unicode characters (German ß).
///
/// The `find_case_insensitive` function in search.rs uses character-by-character
/// matching. This test verifies that multi-byte chars in the *query* (not just
/// the corpus) don't cause byte-boundary panics or incorrect results.
#[test]
fn search_query_with_multibyte_unicode_chars() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_messages(
        temp.path(),
        "sess-german",
        "German Chat",
        "m",
        &["The German word Straße means street"],
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));

    // Search with a query containing ß (U+00DF, 2 bytes in UTF-8).
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "items",
            "text": "Straße",
            "highlight": true
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1, "should find the message containing 'Straße'");

    let snippet = data[0]["snippet"].as_str().unwrap();
    assert!(
        snippet.contains("**Straße**"),
        "snippet should contain highlighted **Straße**, got: {snippet}"
    );
}

// ---------------------------------------------------------------------------
// Items search: cursor parameter handling
// ---------------------------------------------------------------------------

/// Items search ignores cursor parameter (not yet supported).
///
/// The handler at search.rs:764 returns `cursor: None` regardless of input.
/// Passing a bogus cursor should not cause an error.
#[test]
fn items_search_cursor_param_ignored() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_messages(
        temp.path(),
        "sess-cursor-test",
        "Cursor Chat",
        "m",
        &["findable content for cursor test"],
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "items",
            "text": "findable",
            "cursor": "some-opaque-cursor-value"
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1, "should still return the matching item");

    // Cursor should be null (items search doesn't support it).
    assert!(
        json["cursor"].is_null(),
        "items search cursor should be null, got: {}",
        json["cursor"]
    );
}

// ---------------------------------------------------------------------------
// Items search: has_more set correctly with limit
// ---------------------------------------------------------------------------

/// Items search sets `has_more` when more results exist than the limit.
#[test]
fn items_search_has_more_with_limit() {
    let temp = TempDir::new().expect("temp dir");
    // Seed 3 sessions each with a message containing the search term.
    for i in 0..3 {
        seed_session_with_messages(
            temp.path(),
            &format!("sess-hm-{i}"),
            &format!("HasMore Chat {i}"),
            "m",
            &[&format!("HASMORE_KEYWORD content {i}")],
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    let ctx = ServerTestContext::new(search_config(temp.path()));

    // Limit to 1 result.
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "items",
            "text": "HASMORE_KEYWORD",
            "limit": 1
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1, "should respect limit=1");
    assert_eq!(
        json["has_more"], true,
        "has_more should be true when more results exist"
    );
}

// ---------------------------------------------------------------------------
// Projects aggregation
// ---------------------------------------------------------------------------

/// Projects aggregation returns counts grouped by project_id.
#[test]
fn aggregations_projects_returns_counts() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_project(temp.path(), "agg-p1", "Chat P1", "m", "proj_alpha");
    seed_session_with_project(temp.path(), "agg-p2", "Chat P2", "m", "proj_alpha");
    seed_session_with_project(temp.path(), "agg-p3", "Chat P3", "m", "proj_beta");
    seed_session(temp.path(), "agg-p4", "Chat P4", "m"); // no project

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "aggregations": ["projects"]
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let projects = json["aggregations"]["projects"]
        .as_array()
        .expect("projects array");
    assert_eq!(projects.len(), 3, "should have 3 entries (__default__ + 2 projects)");

    for entry in projects {
        assert!(entry["value"].is_string(), "entry should have 'value'");
        assert!(entry["count"].is_u64(), "entry should have 'count'");
    }

    // __default__ bucket is always prepended first (sessions with no project_id).
    assert_eq!(projects[0]["value"], "__default__");
    assert_eq!(projects[0]["count"], 1);
    // proj_alpha has 2 sessions → should be next (sorted by count desc).
    assert_eq!(projects[1]["value"], "proj_alpha");
    assert_eq!(projects[1]["count"], 2);
    assert_eq!(projects[2]["value"], "proj_beta");
    assert_eq!(projects[2]["count"], 1);
}
