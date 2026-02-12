use super::{
    conversation_config, no_bucket_config, seed_session, seed_session_with_group,
    seed_session_with_tags,
};
use crate::server::common::*;
use tempfile::TempDir;

#[test]
fn list_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = get(ctx.addr(), "/v1/conversations");
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn list_empty_db() {
    let temp = TempDir::new().expect("temp dir");
    std::fs::create_dir_all(temp.path()).ok();
    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["object"], "list");
    assert!(json["data"].as_array().unwrap().is_empty());
    assert_eq!(json["has_more"], false);
}

#[test]
fn list_returns_seeded_sessions() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-a", "First chat", "test-model");
    seed_session(temp.path(), "sess-b", "Second chat", "test-model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations?limit=10");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2);

    // All sessions should have "chat" object type
    for item in data {
        assert_eq!(item["object"], "conversation");
    }
}

#[test]
fn list_respects_limit() {
    let temp = TempDir::new().expect("temp dir");
    for i in 0..5 {
        seed_session(temp.path(), &format!("sess-{i}"), &format!("Chat {i}"), "m");
    }

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations?limit=2");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2);
    assert_eq!(json["has_more"], true);
    assert!(
        json["cursor"].is_string(),
        "cursor should be present when has_more"
    );
}

#[test]
fn list_cursor_pagination_round_trip() {
    let temp = TempDir::new().expect("temp dir");
    for i in 0..5 {
        seed_session(temp.path(), &format!("sess-{i}"), &format!("Chat {i}"), "m");
        // Small sleep to ensure different timestamps
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    // Page 1: limit=3
    let resp1 = get(ctx.addr(), "/v1/conversations?limit=3");
    assert_eq!(resp1.status, 200);
    let json1 = resp1.json();
    let page1 = json1["data"].as_array().expect("page1 data");
    assert_eq!(page1.len(), 3);
    assert_eq!(json1["has_more"], true);
    let cursor = json1["cursor"].as_str().expect("cursor string");

    // Page 2: use cursor
    let resp2 = get(
        ctx.addr(),
        &format!("/v1/conversations?limit=3&cursor={cursor}"),
    );
    assert_eq!(resp2.status, 200);
    let json2 = resp2.json();
    let page2 = json2["data"].as_array().expect("page2 data");
    assert_eq!(page2.len(), 2);
    assert_eq!(json2["has_more"], false);

    // Verify no overlap: page1 and page2 session IDs should be disjoint
    let page1_ids: Vec<&str> = page1.iter().map(|v| v["id"].as_str().unwrap()).collect();
    let page2_ids: Vec<&str> = page2.iter().map(|v| v["id"].as_str().unwrap()).collect();
    for id in &page2_ids {
        assert!(
            !page1_ids.contains(id),
            "session {id} appears in both pages"
        );
    }
}

#[test]
fn list_without_prefix() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat", "model");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/conversations");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["data"].as_array().unwrap().len(), 1);
}

#[test]
fn list_has_more_false_when_exact_limit() {
    let temp = TempDir::new().expect("temp dir");
    for i in 0..3 {
        seed_session(temp.path(), &format!("sess-{i}"), &format!("Chat {i}"), "m");
    }

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations?limit=3");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    assert_eq!(json["data"].as_array().unwrap().len(), 3);
    assert_eq!(json["has_more"], false);
}

#[test]
fn list_default_limit_is_20() {
    let temp = TempDir::new().expect("temp dir");
    for i in 0..25 {
        seed_session(
            temp.path(),
            &format!("sess-{i:02}"),
            &format!("Chat {i}"),
            "m",
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    // No limit= param — should default to 20
    let resp = get(ctx.addr(), "/v1/conversations");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 20, "default limit should be 20");
    assert_eq!(json["has_more"], true);
}

#[test]
fn list_clamps_limit_to_100() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat", "m");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    // Request limit=999 — server should clamp to 100 and not error
    let resp = get(ctx.addr(), "/v1/conversations?limit=999");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
}

#[test]
fn list_clamps_limit_to_1_minimum() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat", "m");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    // Request limit=0 — server should clamp to 1
    let resp = get(ctx.addr(), "/v1/conversations?limit=0");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 1, "limit=0 should be clamped to 1");
}

#[test]
fn list_cursor_null_when_no_more() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat", "m");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations?limit=10");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    assert_eq!(json["has_more"], false);
    assert!(
        json["cursor"].is_null(),
        "cursor should be null when has_more is false"
    );
}

#[test]
fn list_sessions_have_expected_fields() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(
        temp.path(),
        "sess-fields",
        "Field Test Chat",
        "test-model-v1",
    );

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 1);

    let session = &data[0];
    assert_eq!(session["id"], "sess-fields");
    assert_eq!(session["object"], "conversation");
    assert_eq!(session["title"], "Field Test Chat");
    assert_eq!(session["model"], "test-model-v1");
    assert_eq!(session["marker"], "active");
    // Timestamps should be positive integers
    assert!(
        session["created_at"].as_i64().unwrap() > 0,
        "created_at should be positive"
    );
    assert!(
        session["updated_at"].as_i64().unwrap() > 0,
        "updated_at should be positive"
    );
    // parent_session_id and group_id should be null for a root session
    assert!(session["parent_session_id"].is_null());
    assert!(session["group_id"].is_null());
    // metadata should be an empty object by default
    assert_eq!(session["metadata"], serde_json::json!({}));
}

#[test]
fn list_invalid_cursor_ignored() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat", "m");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    // Garbage cursor — should be treated as no cursor and return all results
    let resp = get(ctx.addr(), "/v1/conversations?cursor=not-valid-base64!!!");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 1);
}

#[test]
fn list_pagination_covers_all_sessions() {
    let temp = TempDir::new().expect("temp dir");
    let mut all_ids: Vec<String> = Vec::new();
    for i in 0..7 {
        let id = format!("sess-page-{i}");
        seed_session(temp.path(), &id, &format!("Chat {i}"), "m");
        all_ids.push(id);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let mut collected_ids: Vec<String> = Vec::new();
    let mut cursor: Option<String> = None;

    // Walk all pages with limit=2
    for _ in 0..10 {
        let url = match &cursor {
            Some(c) => format!("/v1/conversations?limit=2&cursor={c}"),
            None => "/v1/conversations?limit=2".to_string(),
        };
        let resp = get(ctx.addr(), &url);
        assert_eq!(resp.status, 200);
        let json = resp.json();
        let data = json["data"].as_array().expect("data");
        for item in data {
            collected_ids.push(item["id"].as_str().unwrap().to_string());
        }
        if json["has_more"] == false {
            break;
        }
        cursor = Some(json["cursor"].as_str().expect("cursor").to_string());
    }

    // All 7 sessions should appear exactly once
    assert_eq!(collected_ids.len(), 7, "should collect all 7 sessions");
    // Every seeded ID should be present
    for id in &all_ids {
        assert!(
            collected_ids.contains(id),
            "session {id} not found in paginated results"
        );
    }
}

// ---------------------------------------------------------------------------
// Query parameter edge cases
// ---------------------------------------------------------------------------

#[test]
fn list_negative_limit_uses_default() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat", "m");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    // Negative limit can't parse to usize, falls back to default (20)
    let resp = get(ctx.addr(), "/v1/conversations?limit=-1");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
}

#[test]
fn list_non_numeric_limit_uses_default() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat", "m");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations?limit=abc");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    // Should still return data (uses default limit)
    assert!(json["data"].is_array());
}

#[test]
fn list_empty_limit_uses_default() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat", "m");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations?limit=");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
}

#[test]
fn list_float_limit_uses_default() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat", "m");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations?limit=2.5");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
}

#[test]
fn list_empty_cursor_returns_all() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat", "m");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    // Empty cursor value — decode fails, treated as no cursor
    let resp = get(ctx.addr(), "/v1/conversations?cursor=");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["data"].as_array().unwrap().len(), 1);
}

#[test]
fn list_cursor_valid_base64_but_bad_format() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat", "m");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    // "hello" base64-encoded = "aGVsbG8=" — valid base64 but not "ts:session_id" format
    let resp = get(ctx.addr(), "/v1/conversations?cursor=aGVsbG8=");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    // Should treat as no cursor (decode_cursor fails on split_once(':'))
    let json = resp.json();
    assert_eq!(json["data"].as_array().unwrap().len(), 1);
}

#[test]
fn list_empty_query_string_same_as_no_query() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat", "m");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations?");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["data"].as_array().unwrap().len(), 1);
}

#[test]
fn list_unknown_query_params_ignored() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat", "m");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations?foo=bar&baz=qux&limit=10");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["data"].as_array().unwrap().len(), 1);
}

#[test]
fn list_limit_1_returns_single_session() {
    let temp = TempDir::new().expect("temp dir");
    for i in 0..3 {
        seed_session(temp.path(), &format!("sess-{i}"), &format!("Chat {i}"), "m");
    }

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations?limit=1");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 1);
    assert_eq!(json["has_more"], true);
}

#[test]
fn list_response_object_field() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-1", "Chat", "m");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    assert_eq!(json["object"], "list", "top-level object should be 'list'");
    assert!(json["data"].is_array(), "data should be an array");
    assert!(json["has_more"].is_boolean(), "has_more should be boolean");
}

// ---------------------------------------------------------------------------
// group_id filtering
// ---------------------------------------------------------------------------

#[test]
fn list_group_id_filters_sessions() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_group(temp.path(), "sess-g1a", "Chat A", "m", "tenant-1");
    seed_session_with_group(temp.path(), "sess-g1b", "Chat B", "m", "tenant-1");
    seed_session_with_group(temp.path(), "sess-g2a", "Chat C", "m", "tenant-2");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    // Filter by tenant-1
    let resp = get(ctx.addr(), "/v1/conversations?group_id=tenant-1");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 2, "should return 2 sessions for tenant-1");
    let ids: Vec<&str> = data.iter().map(|s| s["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"sess-g1a"));
    assert!(ids.contains(&"sess-g1b"));
}

#[test]
fn list_group_id_excludes_other_tenants() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_group(temp.path(), "sess-ga", "Chat", "m", "tenant-1");
    seed_session_with_group(temp.path(), "sess-gb", "Chat", "m", "tenant-2");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/conversations?group_id=tenant-2");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "sess-gb");
}

#[test]
fn list_group_id_nonexistent_returns_empty() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_group(temp.path(), "sess-ga", "Chat", "m", "tenant-1");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/conversations?group_id=nonexistent-tenant");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert!(
        data.is_empty(),
        "no sessions should match nonexistent group_id"
    );
}

#[test]
fn list_without_group_id_returns_all() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_group(temp.path(), "sess-x1", "Chat", "m", "tenant-1");
    seed_session_with_group(temp.path(), "sess-x2", "Chat", "m", "tenant-2");
    seed_session(temp.path(), "sess-x3", "Chat", "m"); // no group_id

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/conversations");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(
        data.len(),
        3,
        "all sessions returned without group_id filter"
    );
}

#[test]
fn list_group_id_with_pagination() {
    let temp = TempDir::new().expect("temp dir");
    for i in 0..5 {
        seed_session_with_group(
            temp.path(),
            &format!("sess-pg-{i}"),
            &format!("Chat {i}"),
            "m",
            "paged-tenant",
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    // Add sessions from another group that should not appear
    seed_session_with_group(temp.path(), "sess-other", "Other", "m", "other-tenant");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    // Page 1
    let resp = get(
        ctx.addr(),
        "/v1/conversations?group_id=paged-tenant&limit=3",
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let page1 = json["data"].as_array().expect("data");
    assert_eq!(page1.len(), 3);
    assert_eq!(json["has_more"], true);
    let cursor = json["cursor"].as_str().expect("cursor");

    // Page 2
    let resp = get(
        ctx.addr(),
        &format!("/v1/conversations?group_id=paged-tenant&limit=3&cursor={cursor}"),
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let page2 = json["data"].as_array().expect("data");
    assert_eq!(page2.len(), 2);
    assert_eq!(json["has_more"], false);
}

#[test]
fn list_group_id_shown_in_response() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_group(temp.path(), "sess-gshow", "Chat", "m", "my-group");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data[0]["group_id"], "my-group");
}

// ---------------------------------------------------------------------------
// Ordering verification
// ---------------------------------------------------------------------------

#[test]
fn list_sessions_ordered_by_updated_at_descending() {
    let temp = TempDir::new().expect("temp dir");
    for i in 0..5 {
        seed_session(
            temp.path(),
            &format!("sess-ord-{i}"),
            &format!("Chat {i}"),
            "m",
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations?limit=5");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 5);

    // Verify descending order by updated_at
    let timestamps: Vec<i64> = data
        .iter()
        .map(|s| s["updated_at"].as_i64().expect("updated_at"))
        .collect();
    for i in 0..timestamps.len() - 1 {
        assert!(
            timestamps[i] >= timestamps[i + 1],
            "sessions should be ordered newest-first: {:?}",
            timestamps,
        );
    }
}

#[test]
fn list_ordering_stable_across_pages() {
    let temp = TempDir::new().expect("temp dir");
    for i in 0..6 {
        seed_session(
            temp.path(),
            &format!("sess-ost-{i}"),
            &format!("Chat {i}"),
            "m",
        );
        std::thread::sleep(std::time::Duration::from_millis(15));
    }

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let mut all_timestamps: Vec<i64> = Vec::new();
    let mut cursor: Option<String> = None;

    for _ in 0..5 {
        let url = match &cursor {
            Some(c) => format!("/v1/conversations?limit=2&cursor={c}"),
            None => "/v1/conversations?limit=2".to_string(),
        };
        let resp = get(ctx.addr(), &url);
        assert_eq!(resp.status, 200);
        let json = resp.json();
        let data = json["data"].as_array().expect("data");
        for item in data {
            all_timestamps.push(item["updated_at"].as_i64().expect("updated_at"));
        }
        if json["has_more"] == false {
            break;
        }
        cursor = Some(json["cursor"].as_str().expect("cursor").to_string());
    }

    assert_eq!(all_timestamps.len(), 6);
    // Verify monotonically non-increasing across all pages
    for i in 0..all_timestamps.len() - 1 {
        assert!(
            all_timestamps[i] >= all_timestamps[i + 1],
            "cross-page ordering should be descending: {:?}",
            all_timestamps,
        );
    }
}

// ---------------------------------------------------------------------------
// List metadata fields from SessionRecordFull
// ---------------------------------------------------------------------------

#[test]
fn list_sessions_include_metadata_when_set() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-lm", "Chat", "m");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));

    // Set metadata via PATCH
    let resp = patch_json(
        ctx.addr(),
        "/v1/conversations/sess-lm",
        &serde_json::json!({"metadata": {"key": "value"}}),
    );
    assert_eq!(resp.status, 200);

    // Verify list includes the metadata
    let resp = get(ctx.addr(), "/v1/conversations");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    let session = data
        .iter()
        .find(|s| s["id"] == "sess-lm")
        .expect("session in list");
    assert_eq!(session["metadata"]["key"], "value");
}

// ---------------------------------------------------------------------------
// Tags in list responses
// ---------------------------------------------------------------------------

/// Sessions with tags include resolved tag objects in response.
#[test]
fn list_includes_tags_for_tagged_session() {
    let temp = TempDir::new().expect("temp dir");
    seed_session_with_tags(
        temp.path(),
        "sess-t",
        "Tagged Chat",
        "m",
        &["rust", "python", "work"],
    );

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);

    let tags = data[0]["tags"].as_array();
    assert!(tags.is_some(), "tags field should be present");
    let tags = tags.unwrap();

    // Tags should be resolved to objects with at least name field
    // (auto-created from metadata.tags strings)
    assert!(
        !tags.is_empty(),
        "tags should not be empty for tagged session"
    );

    // Collect tag names
    let tag_names: Vec<&str> = tags.iter().filter_map(|t| t["name"].as_str()).collect();

    assert!(tag_names.contains(&"rust"), "tags should contain rust");
    assert!(tag_names.contains(&"python"), "tags should contain python");
    assert!(tag_names.contains(&"work"), "tags should contain work");
}

/// Sessions without tags have empty tags array.
#[test]
fn list_untagged_session_has_empty_tags() {
    let temp = TempDir::new().expect("temp dir");
    seed_session(temp.path(), "sess-u", "Untagged Chat", "m");

    let ctx = ServerTestContext::new(conversation_config(temp.path()));
    let resp = get(ctx.addr(), "/v1/conversations");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);

    let tags = &data[0]["tags"];
    assert!(tags.is_array(), "tags should be an array");
    assert!(
        tags.as_array().unwrap().is_empty(),
        "untagged session should have empty tags array"
    );
}
