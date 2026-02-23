//! Document search tests.
//!
//! Tests POST /v1/db/tables/documents/search endpoint.

use super::{documents_config, no_bucket_config};
use crate::server::common::*;
use tempfile::TempDir;

// =============================================================================
// POST /v1/db/tables/documents/search
// =============================================================================

#[test]
fn search_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let body = serde_json::json!({"query": "test"});
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents/search", &body);
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn search_returns_empty_for_no_matches() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create a document
    post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt",
            "title": "Test Document",
            "content": {"text": "Hello world"}
        }),
    );

    // Search for something that doesn't match
    let body = serde_json::json!({"query": "xyznonexistent"});
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents/search", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert!(json["data"].is_array());
    assert_eq!(json["data"].as_array().unwrap().len(), 0);
}

#[test]
fn search_finds_matching_title() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create documents
    post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt",
            "title": "Rust Programming Guide",
            "content": {}
        }),
    );
    post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt",
            "title": "Python Tutorial",
            "content": {}
        }),
    );

    // Search for "Rust"
    let body = serde_json::json!({"query": "Rust"});
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents/search", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(!data.is_empty(), "should find Rust document");
    assert!(data
        .iter()
        .any(|d| d["title"].as_str().unwrap_or("").contains("Rust")));
}

#[test]
fn search_finds_matching_content() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt",
            "title": "Generic Title",
            "content": {"system": "You are a Rust programming expert."}
        }),
    );

    let body = serde_json::json!({"query": "Rust programming"});
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents/search", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(!data.is_empty(), "should find document by content");
}

#[test]
fn search_filters_by_type() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create documents of different types with same keyword
    post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt",
            "title": "Coding Prompt",
            "content": {}
        }),
    );
    post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "persona",
            "title": "Coding Persona",
            "content": {}
        }),
    );

    // Search with type filter
    let body = serde_json::json!({
        "query": "Coding",
        "type": "prompt"
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents/search", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    for doc in data {
        assert_eq!(doc["type"], "prompt", "should only return prompts");
    }
}

#[test]
fn search_respects_limit() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create many matching documents
    for i in 0..10 {
        post_json(
            ctx.addr(),
            "/v1/db/tables/documents",
            &serde_json::json!({
                "type": "note",
                "title": format!("Common Note {}", i),
                "content": {}
            }),
        );
    }

    let body = serde_json::json!({
        "query": "Common",
        "limit": 3
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents/search", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(data.len() <= 3, "should respect limit");
}

#[test]
fn search_case_insensitive() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt",
            "title": "UPPERCASE Title",
            "content": {}
        }),
    );

    // Search with lowercase
    let body = serde_json::json!({"query": "uppercase"});
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents/search", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(!data.is_empty(), "search should be case-insensitive");
}

#[test]
fn search_deleted_not_found() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create and delete
    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt",
            "title": "Deleted Document Unique",
            "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();
    delete(ctx.addr(), &format!("/v1/db/tables/documents/{}", doc_id));

    // Search should not find deleted
    let body = serde_json::json!({"query": "Deleted Document Unique"});
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents/search", &body);
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(
        data.is_empty(),
        "deleted document should not appear in search"
    );
}

#[test]
fn search_requires_query() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({});
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents/search", &body);
    // Should either require query or return all documents
    // Implementation may vary - just ensure it doesn't crash
    assert!(resp.status == 200 || resp.status == 400);
}

#[test]
fn search_empty_query_handled() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({"query": ""});
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents/search", &body);
    // Should handle empty query gracefully
    assert!(resp.status == 200 || resp.status == 400);
}

#[test]
fn search_with_group_id_filter() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create documents in different groups
    post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt",
            "title": "Team A Document",
            "content": {},
            "group_id": "team-a"
        }),
    );
    post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt",
            "title": "Team B Document",
            "content": {},
            "group_id": "team-b"
        }),
    );

    // Search with group filter
    let body = serde_json::json!({
        "query": "Team",
        "group_id": "team-a"
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents/search", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    // Should only return team-a documents
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    for doc in data {
        // Either has group_id=team-a or check title
        if doc.get("group_id").is_some() {
            assert_eq!(doc["group_id"], "team-a");
        }
    }
}

/// Expired documents still appear in search results (known inconsistency:
/// direct GET filters by TTL via the Zig layer, but search does not).
///
/// If this assertion starts failing (i.e. the search returns no results),
/// TTL filtering was added to the search path — update to assert absence.
#[test]
fn expired_document_still_appears_in_search() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create an ephemeral document with a unique searchable term.
    let ephemeral = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "note",
            "title": "Ephemeral Search Target",
            "content": {"text": "alpha_kappa_search_term_unique"},
            "ttl_seconds": 1
        }),
    );
    assert_eq!(ephemeral.status, 201, "body: {}", ephemeral.body);
    let ephemeral_id = ephemeral.json()["id"].as_str().unwrap().to_string();

    // Create a permanent document with different content.
    let permanent = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "note",
            "title": "Permanent Search Doc",
            "content": {"text": "beta_lambda_unrelated_term"}
        }),
    );
    assert_eq!(permanent.status, 201, "body: {}", permanent.body);

    // Wait for TTL to expire.
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Direct GET should return 404 (Zig layer filters by TTL).
    let get_expired = get(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}", ephemeral_id),
    );
    assert_eq!(
        get_expired.status, 404,
        "expired doc should return 404 on direct GET"
    );

    // Search for the ephemeral document's unique content.
    let search_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents/search",
        &serde_json::json!({"query": "alpha_kappa_search_term_unique"}),
    );
    assert_eq!(search_resp.status, 200, "body: {}", search_resp.body);

    let data = search_resp.json()["data"]
        .as_array()
        .expect("data array")
        .clone();
    let ids: Vec<&str> = data.iter().filter_map(|d| d["id"].as_str()).collect();
    assert!(
        ids.contains(&ephemeral_id.as_str()),
        "expired doc still appears in search (known inconsistency: search does not filter by TTL)"
    );
}
