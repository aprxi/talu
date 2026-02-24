//! Integration tests for tag filtering in `POST /v1/search`.

use super::{
    search_config, seed_session, seed_session_with_tags, seed_session_with_tags_and_group,
};
use crate::server::common::*;
use serde_json::json;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Basic tag filter (filters.tags AND logic)
// ---------------------------------------------------------------------------

/// Filter by a single tag using exact word match.
#[test]
fn tag_filter_single_tag() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags(
        temp.path(),
        "sess-rust",
        "Rust Chat",
        "m",
        &["rust", "work"],
    );
    seed_session_with_tags(
        temp.path(),
        "sess-py",
        "Python Chat",
        "m",
        &["python", "work"],
    );
    seed_session_with_tags(
        temp.path(),
        "sess-rusty",
        "Rusty Tools",
        "m",
        &["rusty", "tools"],
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": {
                "tags": ["rust"]
            }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    // Only sess-rust has the exact tag "rust" (not "rusty")
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "sess-rust");
}

/// Tag filter with multiple tags uses AND logic (must have ALL tags).
#[test]
fn tag_filter_and_logic() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags(
        temp.path(),
        "sess-both",
        "Both Tags",
        "m",
        &["rust", "python"],
    );
    seed_session_with_tags(temp.path(), "sess-rust-only", "Rust Only", "m", &["rust"]);
    seed_session_with_tags(temp.path(), "sess-py-only", "Python Only", "m", &["python"]);

    let ctx = ServerTestContext::new(search_config(temp.path()));
    // Session must have BOTH "rust" AND "python"
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": {
                "tags": ["rust", "python"]
            }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "sess-both");
}

/// Tag filter is case-insensitive.
#[test]
fn tag_filter_case_insensitive() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags(temp.path(), "sess-a", "Chat A", "m", &["Rust", "Python"]);

    let ctx = ServerTestContext::new(search_config(temp.path()));

    // Lowercase query should match uppercase tags
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": {
                "tags": ["rust"]
            }
        }),
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);

    // Uppercase query should match
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": {
                "tags": ["PYTHON"]
            }
        }),
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
}

/// Tag filter with no matching sessions returns empty list.
#[test]
fn tag_filter_no_match() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags(temp.path(), "sess-a", "Chat", "m", &["rust", "python"]);

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": {
                "tags": ["java"]
            }
        }),
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(data.is_empty());
}

/// Sessions without tags are excluded when tag filter is applied.
#[test]
fn tag_filter_excludes_untagged() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags(temp.path(), "sess-tagged", "Tagged", "m", &["rust"]);
    seed_session(temp.path(), "sess-untagged", "Untagged", "m");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": {
                "tags": ["rust"]
            }
        }),
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "sess-tagged");
}

// ---------------------------------------------------------------------------
// OR tag filter (filters.tags_any)
// ---------------------------------------------------------------------------

/// Filter by tags using OR logic (must have ANY of the tags).
#[test]
fn tag_any_filter_or_logic() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags(temp.path(), "sess-rust", "Rust Only", "m", &["rust"]);
    seed_session_with_tags(temp.path(), "sess-py", "Python Only", "m", &["python"]);
    seed_session_with_tags(temp.path(), "sess-go", "Go Only", "m", &["go"]);

    let ctx = ServerTestContext::new(search_config(temp.path()));
    // Session must have "rust" OR "python"
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": {
                "tags_any": ["rust", "python"]
            }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2);
    let ids: Vec<&str> = data.iter().map(|d| d["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"sess-rust"));
    assert!(ids.contains(&"sess-py"));
    assert!(!ids.contains(&"sess-go"));
}

/// Single tag with tags_any works like tags filter.
#[test]
fn tag_any_single_tag() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags(temp.path(), "sess-a", "Chat A", "m", &["work"]);
    seed_session_with_tags(temp.path(), "sess-b", "Chat B", "m", &["personal"]);

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": {
                "tags_any": ["work"]
            }
        }),
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "sess-a");
}

/// tags_any with no matching sessions returns empty list.
#[test]
fn tag_any_no_match() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags(temp.path(), "sess-a", "Chat", "m", &["rust"]);

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": {
                "tags_any": ["java", "kotlin"]
            }
        }),
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(data.is_empty());
}

// ---------------------------------------------------------------------------
// Empty / missing tag params
// ---------------------------------------------------------------------------

/// Empty tags array is treated as no filter (returns all).
#[test]
fn tag_filter_empty_returns_all() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags(temp.path(), "sess-a", "Alpha", "m", &["rust"]);
    seed_session(temp.path(), "sess-b", "Beta", "m");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": {
                "tags": []
            }
        }),
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2, "empty tags should not filter");
}

/// Empty tags_any array is treated as no filter (returns all).
#[test]
fn tag_any_empty_returns_all() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags(temp.path(), "sess-a", "Alpha", "m", &["rust"]);
    seed_session(temp.path(), "sess-b", "Beta", "m");

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": {
                "tags_any": []
            }
        }),
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2, "empty tags_any should not filter");
}

// ---------------------------------------------------------------------------
// Combined filters
// ---------------------------------------------------------------------------

/// Tag filter combined with text search.
#[test]
fn tag_filter_with_text_search() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags(
        temp.path(),
        "sess-rust-work",
        "Rust Work",
        "m",
        &["rust", "work"],
    );
    seed_session_with_tags(
        temp.path(),
        "sess-rust-play",
        "Rust Play",
        "m",
        &["rust", "personal"],
    );
    seed_session_with_tags(
        temp.path(),
        "sess-py-work",
        "Python Work",
        "m",
        &["python", "work"],
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));
    // Search for "Rust" in title AND tag=work
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "text": "rust",
            "filters": {
                "tags": ["work"]
            }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "sess-rust-work");
}

/// Tag filter combined with group_id.
#[test]
fn tag_filter_with_group_id() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags_and_group(
        temp.path(),
        "sess-g1-rust",
        "Rust G1",
        "m",
        &["rust"],
        "group1",
    );
    seed_session_with_tags_and_group(
        temp.path(),
        "sess-g2-rust",
        "Rust G2",
        "m",
        &["rust"],
        "group2",
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": {
                "tags": ["rust"],
                "group_id": "group1"
            }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "sess-g1-rust");
}

/// Tag filter with pagination.
#[test]
fn tag_filter_with_pagination() {
    let temp = TempDir::new().expect("temp dir");
    for i in 0..5 {
        seed_session_with_tags(
            temp.path(),
            &format!("sess-rust-{i}"),
            &format!("Rust Chat {i}"),
            "m",
            &["rust"],
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    // Non-matching session
    seed_session_with_tags(temp.path(), "sess-py", "Python", "m", &["python"]);

    let ctx = ServerTestContext::new(search_config(temp.path()));

    let mut all_ids: Vec<String> = Vec::new();
    let mut cursor: Option<String> = None;

    for _ in 0..10 {
        let mut req = json!({
            "scope": "sessions",
            "filters": {
                "tags": ["rust"]
            },
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

    assert_eq!(all_ids.len(), 5, "should find all 5 rust-tagged sessions");
}

// ---------------------------------------------------------------------------
// Tags API vs. search index integration
//
// Search resolves tag filters via the relational `session_tags` table.
// Tags added via the dedicated API are the single source of truth for both
// search filtering and response display.
// ---------------------------------------------------------------------------

/// Tags added via the dedicated API must be searchable.
#[test]
fn tags_via_api_searchable() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-api-tag", "API Tagged", "m");

    let ctx = ServerTestContext::new(search_config(temp.path()));

    // Add tag via dedicated tags API
    let resp = post_json(
        ctx.addr(),
        "/v1/chat/sessions/sess-api-tag/tags",
        &json!({"tags": ["api-tag"]}),
    );
    assert_eq!(resp.status, 200, "add tag failed: {}", resp.body);

    // Tag visible in session response
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-api-tag");
    assert_eq!(resp.status, 200);
    let conv = resp.json();
    let tags = conv["tags"].as_array().expect("tags array");
    assert_eq!(tags.len(), 1, "tag should appear in response");
    assert_eq!(tags[0]["name"], "api-tag");

    // Tag must also be findable via search
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": {
                "tags": ["api-tag"]
            }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(
        data.len(),
        1,
        "Tags added via the dedicated API must be searchable"
    );
    assert_eq!(data[0]["id"], "sess-api-tag");
}

/// Tags created via seed helper must be searchable and visible in responses.
///
/// Verifies that relational tags created by `seed_session_with_tags` are
/// both filterable in search and visible in the session response.
#[test]
fn metadata_tags_visible_in_response() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags(
        temp.path(),
        "sess-meta",
        "Metadata Tagged",
        "m",
        &["meta-tag"],
    );

    let ctx = ServerTestContext::new(search_config(temp.path()));

    // Search finds it via relational tag table
    let resp = post_json(
        ctx.addr(),
        "/v1/search",
        &json!({
            "scope": "sessions",
            "filters": {
                "tags": ["meta-tag"]
            }
        }),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1, "metadata tags must be searchable");
    assert_eq!(data[0]["id"], "sess-meta");

    // Tag must also appear in the session response tags array
    let resp = get(ctx.addr(), "/v1/chat/sessions/sess-meta");
    assert_eq!(resp.status, 200);
    let conv = resp.json();
    let tags = conv["tags"].as_array().expect("tags array");
    assert_eq!(
        tags.len(),
        1,
        "Metadata tags must be visible in the response tags array"
    );
    assert_eq!(tags[0]["name"], "meta-tag");
}
