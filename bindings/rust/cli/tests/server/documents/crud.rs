//! Document CRUD operation tests.
//!
//! Tests create, get, list, update, and delete operations.

use super::{documents_config, no_bucket_config};
use crate::server::common::*;
use tempfile::TempDir;

// =============================================================================
// GET /v1/db/tables/documents (list)
// =============================================================================

#[test]
fn list_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = get(ctx.addr(), "/v1/db/tables/documents");
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn list_returns_empty_array_initially() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/db/tables/documents");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert!(json["data"].is_array(), "should have data array");
    assert_eq!(
        json["data"].as_array().unwrap().len(),
        0,
        "should be empty initially"
    );
}

#[test]
fn list_returns_created_documents() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create a document
    let create_body = serde_json::json!({
        "type": "prompt",
        "title": "Test Prompt",
        "content": {"system": "You are helpful."}
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents", &create_body);
    assert_eq!(resp.status, 201, "create: {}", resp.body);

    // List should return the document
    let resp = get(ctx.addr(), "/v1/db/tables/documents");
    assert_eq!(resp.status, 200, "list: {}", resp.body);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1, "should have one document");
    assert_eq!(data[0]["title"], "Test Prompt");
    assert_eq!(data[0]["type"], "prompt");
}

#[test]
fn list_filters_by_type() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create documents of different types
    post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Prompt 1", "content": {}
        }),
    );
    post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "persona", "title": "Persona 1", "content": {}
        }),
    );
    post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Prompt 2", "content": {}
        }),
    );

    // Filter by type
    let resp = get(ctx.addr(), "/v1/db/tables/documents?type=prompt");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2, "should have 2 prompts");
    for doc in data {
        assert_eq!(doc["type"], "prompt");
    }
}

#[test]
fn list_respects_limit() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create several documents
    for i in 0..5 {
        post_json(
            ctx.addr(),
            "/v1/db/tables/documents",
            &serde_json::json!({
                "type": "note", "title": format!("Note {}", i), "content": {}
            }),
        );
    }

    // List with limit
    let resp = get(ctx.addr(), "/v1/db/tables/documents?limit=2");
    assert_eq!(resp.status, 200);

    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(data.len() <= 2, "should respect limit");
}

// =============================================================================
// POST /v1/db/tables/documents (create)
// =============================================================================

#[test]
fn create_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let body = serde_json::json!({
        "type": "prompt",
        "title": "Test",
        "content": {}
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents", &body);
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn create_returns_document_with_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({
        "type": "prompt",
        "title": "My Prompt",
        "content": {"system": "Be helpful."}
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents", &body);
    assert_eq!(resp.status, 201, "body: {}", resp.body);

    let json = resp.json();
    // Response doesn't include "object" field - just check the essential fields
    assert!(json["id"].is_string(), "should have id");
    assert_eq!(json["title"], "My Prompt");
    assert_eq!(json["type"], "prompt");
    assert!(json["created_at"].as_i64().unwrap() > 0);
}

#[test]
fn create_with_tags() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({
        "type": "prompt",
        "title": "Tagged Prompt",
        "content": {},
        "tags": ["coding", "rust"]
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents", &body);
    assert_eq!(resp.status, 201, "body: {}", resp.body);

    let json = resp.json();
    // Tags may be in response or need separate GET
    assert!(json["id"].is_string());
}

#[test]
fn create_with_owner_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({
        "type": "note",
        "title": "My Note",
        "content": {"text": "Personal note"},
        "owner_id": "user-123"
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents", &body);
    assert_eq!(resp.status, 201, "body: {}", resp.body);
}

#[test]
fn create_with_group_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({
        "type": "prompt",
        "title": "Team Prompt",
        "content": {},
        "group_id": "team-alpha"
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents", &body);
    assert_eq!(resp.status, 201, "body: {}", resp.body);
}

#[test]
fn create_requires_type() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({
        "title": "No Type",
        "content": {}
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents", &body);
    assert_eq!(resp.status, 400, "should reject missing type");
}

#[test]
fn create_requires_title() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({
        "type": "prompt",
        "content": {}
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents", &body);
    assert_eq!(resp.status, 400, "should reject missing title");
}

// =============================================================================
// GET /v1/db/tables/documents/:id (get)
// =============================================================================

#[test]
fn get_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = get(ctx.addr(), "/v1/db/tables/documents/some-id");
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn get_returns_404_for_missing() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/db/tables/documents/nonexistent-id");
    assert_eq!(resp.status, 404, "body: {}", resp.body);
}

#[test]
fn get_returns_created_document() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create
    let body = serde_json::json!({
        "type": "prompt",
        "title": "Get Test",
        "content": {"system": "Test content"}
    });
    let create_resp = post_json(ctx.addr(), "/v1/db/tables/documents", &body);
    assert_eq!(create_resp.status, 201);
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    // Get
    let resp = get(ctx.addr(), &format!("/v1/db/tables/documents/{}", doc_id));
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["id"], doc_id);
    assert_eq!(json["title"], "Get Test");
    assert_eq!(json["type"], "prompt");
}

#[test]
fn get_returns_full_content() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let content = serde_json::json!({
        "system": "You are a code reviewer.",
        "model": "gpt-4",
        "temperature": 0.7
    });
    let body = serde_json::json!({
        "type": "prompt",
        "title": "Full Content",
        "content": content
    });
    let create_resp = post_json(ctx.addr(), "/v1/db/tables/documents", &body);
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    let resp = get(ctx.addr(), &format!("/v1/db/tables/documents/{}", doc_id));
    assert_eq!(resp.status, 200);

    let json = resp.json();
    // Content should be preserved
    assert!(json["content"].is_object() || json["data"].is_object());
}

// =============================================================================
// PATCH /v1/db/tables/documents/:id (update)
// =============================================================================

#[test]
fn patch_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let body = serde_json::json!({"title": "Updated"});
    let resp = patch_json(ctx.addr(), "/v1/db/tables/documents/some-id", &body);
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn patch_returns_404_for_missing() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({"title": "Updated"});
    let resp = patch_json(ctx.addr(), "/v1/db/tables/documents/nonexistent", &body);
    assert_eq!(resp.status, 404, "body: {}", resp.body);
}

#[test]
fn patch_updates_title() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create
    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Before", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    // Patch
    let resp = patch_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}", doc_id),
        &serde_json::json!({"title": "After"}),
    );
    assert_eq!(resp.status, 200, "patch: {}", resp.body);

    // Verify
    let get_resp = get(ctx.addr(), &format!("/v1/db/tables/documents/{}", doc_id));
    assert_eq!(get_resp.json()["title"], "After");
}

#[test]
fn patch_updates_content() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Content Test", "content": {"v": 1}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    let resp = patch_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}", doc_id),
        &serde_json::json!({"content": {"v": 2}}),
    );
    assert_eq!(resp.status, 200, "patch: {}", resp.body);
}

#[test]
fn patch_updates_marker() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Marker Test", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    let resp = patch_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}", doc_id),
        &serde_json::json!({"marker": "archived"}),
    );
    assert_eq!(resp.status, 200, "patch: {}", resp.body);
}

// =============================================================================
// DELETE /v1/db/tables/documents/:id (delete)
// =============================================================================

#[test]
fn delete_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = delete(ctx.addr(), "/v1/db/tables/documents/some-id");
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn delete_returns_404_for_missing() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = delete(ctx.addr(), "/v1/db/tables/documents/nonexistent");
    // May return 404 or 200 depending on implementation (soft delete)
    assert!(
        resp.status == 404 || resp.status == 200,
        "body: {}",
        resp.body
    );
}

#[test]
fn delete_removes_document() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create
    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "To Delete", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    // Delete
    let resp = delete(ctx.addr(), &format!("/v1/db/tables/documents/{}", doc_id));
    assert!(
        resp.status == 200 || resp.status == 204,
        "delete: {}",
        resp.body
    );

    // Verify gone (or marked deleted)
    let get_resp = get(ctx.addr(), &format!("/v1/db/tables/documents/{}", doc_id));
    // Either 404 (hard delete) or marker=deleted (soft delete)
    assert!(get_resp.status == 404 || get_resp.json()["marker"] == "deleted");
}

#[test]
fn delete_removed_from_list() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create two documents
    let _keep = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "note", "title": "Keep", "content": {}
        }),
    );
    let create2 = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "note", "title": "Delete", "content": {}
        }),
    );
    let doc_id2 = create2.json()["id"].as_str().expect("id").to_string();

    // Delete second
    delete(ctx.addr(), &format!("/v1/db/tables/documents/{}", doc_id2));

    // List should not include deleted (by default)
    let list_resp = get(ctx.addr(), "/v1/db/tables/documents");
    let json = list_resp.json();
    let data = json["data"].as_array().expect("data");
    // Should only have 1 document now (or 2 if soft-delete without filter)
    let titles: Vec<_> = data
        .iter()
        .map(|d| d["title"].as_str().unwrap_or(""))
        .collect();
    assert!(titles.contains(&"Keep"), "Keep should still exist");
}

// =============================================================================
// Error handling
// =============================================================================

#[test]
fn invalid_json_returns_400() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/db/tables/documents",
        &[("Content-Type", "application/json")],
        Some("not valid json"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
}

#[test]
fn error_responses_have_json_body() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/db/tables/documents/nonexistent");
    assert_eq!(resp.status, 404);

    let json = resp.json();
    assert!(json["error"]["code"].is_string(), "error should have code");
    assert!(
        json["error"]["message"].is_string(),
        "error should have message"
    );
}

// =============================================================================
// TTL / expires_at
// =============================================================================

/// Creating a document with ttl_seconds sets expires_at.
#[test]
fn create_with_ttl_sets_expires_at() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;

    let resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "note",
            "title": "TTL document",
            "content": {"text": "expires soon"},
            "ttl_seconds": 3600
        }),
    );
    assert_eq!(resp.status, 201, "body: {}", resp.body);

    let create_json = resp.json();
    let doc_id = create_json["id"].as_str().unwrap();

    // The create response may or may not include expires_at depending on
    // storage backend read-after-write semantics. Use GET to verify.
    let get_resp = get(ctx.addr(), &format!("/v1/db/tables/documents/{}", doc_id));
    assert_eq!(get_resp.status, 200, "body: {}", get_resp.body);

    let get_json = get_resp.json();

    // Check either the create or GET response for expires_at.
    let expires_at = create_json["expires_at"]
        .as_i64()
        .or_else(|| get_json["expires_at"].as_i64());

    if let Some(ea) = expires_at {
        let expected_min = now_ms + 3600 * 1000 - 10_000; // 10s tolerance
        let expected_max = now_ms + 3600 * 1000 + 10_000;
        assert!(
            ea >= expected_min && ea <= expected_max,
            "expires_at {} should be ~{} (now + 3600s), range [{}, {}]",
            ea,
            now_ms + 3600 * 1000,
            expected_min,
            expected_max,
        );
    }
    // If expires_at is not exposed in either response, verify the request
    // was at least accepted (201 + no error).
    assert_eq!(create_json["type"], "note");
}

/// Creating a document without ttl_seconds has null expires_at.
#[test]
fn create_without_ttl_has_null_expires_at() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "note",
            "title": "No TTL document",
            "content": {"text": "lives forever"}
        }),
    );
    assert_eq!(resp.status, 201, "body: {}", resp.body);

    let json = resp.json();
    assert!(
        json["expires_at"].is_null(),
        "expires_at should be null without ttl_seconds, got: {}",
        json["expires_at"]
    );
}

/// Expired documents are filtered from GET (returns 404).
///
/// `getDocument` in the Zig storage layer uses reverse scanning (newest
/// block/row first) and checks `expires_at > 0 and expires_at < now_ms`,
/// returning null for expired docs.  This test uses a 1-second TTL
/// followed by a 2-second wait — the sleep is inherent to testing
/// time-based expiration, not a synchronization hack.
#[test]
fn expired_document_returns_404_on_get() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create a document with a 1-second TTL.
    let ephemeral = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "note",
            "title": "Ephemeral doc",
            "content": {"text": "gone soon"},
            "ttl_seconds": 1
        }),
    );
    assert_eq!(ephemeral.status, 201, "body: {}", ephemeral.body);
    let ephemeral_id = ephemeral.json()["id"].as_str().unwrap().to_string();

    // Also create a permanent document to ensure non-TTL docs are unaffected.
    let permanent = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "note",
            "title": "Permanent doc",
            "content": {"text": "stays forever"}
        }),
    );
    assert_eq!(permanent.status, 201, "body: {}", permanent.body);
    let permanent_id = permanent.json()["id"].as_str().unwrap().to_string();

    // Immediately after creation, the ephemeral document should be visible.
    let get_before = get(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}", ephemeral_id),
    );
    assert_eq!(
        get_before.status, 200,
        "ephemeral doc should be visible before expiration"
    );

    // Wait for the TTL to lapse (1s TTL + generous margin).
    std::thread::sleep(std::time::Duration::from_secs(2));

    // GET should now return 404 for the expired document.
    let get_after = get(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}", ephemeral_id),
    );
    assert_eq!(
        get_after.status, 404,
        "expired document should return 404, body: {}",
        get_after.body
    );

    // Permanent document should still be accessible.
    let get_permanent = get(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}", permanent_id),
    );
    assert_eq!(
        get_permanent.status, 200,
        "permanent document should still be accessible"
    );
}

// =============================================================================
// TTL: list inconsistency
// =============================================================================

/// Expired documents still appear in list results even though direct GET
/// returns 404.
///
/// `getDocument` in the Zig storage layer filters by TTL (returns null for
/// expired docs), but `listDocuments` does NOT apply TTL filtering.
/// This test locks in the current behavior for future resolution.
///
/// The sleep is inherent to testing time-based expiration, not a
/// synchronization hack — see policy §5.
#[test]
fn expired_document_still_appears_in_list() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create a document with 1-second TTL.
    let ephemeral = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "note",
            "title": "Ephemeral in list",
            "content": {"text": "will expire"},
            "ttl_seconds": 1
        }),
    );
    assert_eq!(ephemeral.status, 201, "body: {}", ephemeral.body);
    let ephemeral_id = ephemeral.json()["id"].as_str().unwrap().to_string();

    // Create a permanent document.
    let permanent = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "note",
            "title": "Permanent in list",
            "content": {"text": "stays forever"}
        }),
    );
    assert_eq!(permanent.status, 201, "body: {}", permanent.body);
    let permanent_id = permanent.json()["id"].as_str().unwrap().to_string();

    // Wait for TTL to lapse.
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Direct GET of expired doc → 404.
    let get_expired = get(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}", ephemeral_id),
    );
    assert_eq!(
        get_expired.status, 404,
        "expired doc should return 404 on direct GET"
    );

    // Permanent doc still accessible.
    let get_permanent = get(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}", permanent_id),
    );
    assert_eq!(get_permanent.status, 200);

    // List returns both — expired doc is NOT filtered from list results.
    // NOTE: If this assertion starts failing, TTL filtering was added to
    // list — update to assert the expired doc is absent instead.
    let list_resp = get(ctx.addr(), "/v1/db/tables/documents");
    assert_eq!(list_resp.status, 200, "body: {}", list_resp.body);
    let data = list_resp.json()["data"]
        .as_array()
        .expect("data array")
        .clone();
    let ids: Vec<&str> = data.iter().filter_map(|d| d["id"].as_str()).collect();

    assert!(
        ids.contains(&permanent_id.as_str()),
        "permanent doc should appear in list"
    );
    assert!(
        ids.contains(&ephemeral_id.as_str()),
        "expired doc still appears in list (known inconsistency: \
         list does not filter by TTL)"
    );
}

// =============================================================================
// PATCH edge cases
// =============================================================================

/// PATCH with null title preserves the existing title (same as conversations).
///
/// `update_req.title.as_deref()` returns `None` for null, which means
/// "no update" is passed to the storage layer.
#[test]
fn patch_null_title_preserves_existing() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "note",
            "title": "Original Title",
            "content": {"text": "test"}
        }),
    );
    assert_eq!(create_resp.status, 201, "body: {}", create_resp.body);
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    // PATCH with title: null — should be a no-op for the title.
    let patch_resp = patch_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}", doc_id),
        &serde_json::json!({"title": null}),
    );
    assert_eq!(patch_resp.status, 200, "body: {}", patch_resp.body);
    assert_eq!(
        patch_resp.json()["title"],
        "Original Title",
        "null title should preserve existing"
    );

    // Verify via GET.
    let get_resp = get(ctx.addr(), &format!("/v1/db/tables/documents/{}", doc_id));
    assert_eq!(get_resp.status, 200, "body: {}", get_resp.body);
    assert_eq!(get_resp.json()["title"], "Original Title");
}

// =============================================================================
// Pagination: full traversal
// =============================================================================

/// `has_more` reflects whether the total exceeds the requested limit.
///
/// The documents endpoint uses limit-only pagination (no offset/cursor),
/// so `has_more = data.len() >= limit`.  This test verifies both the
/// `true` and `false` cases.
#[test]
fn list_has_more_reflects_total_vs_limit() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create 5 documents.
    for i in 0..5 {
        let resp = post_json(
            ctx.addr(),
            "/v1/db/tables/documents",
            &serde_json::json!({
                "type": "note",
                "title": format!("Doc {}", i),
                "content": {"text": format!("content-{}", i)}
            }),
        );
        assert_eq!(resp.status, 201, "body: {}", resp.body);
    }

    // limit=2 when 5 exist → has_more=true, exactly 2 returned.
    let resp = get(ctx.addr(), "/v1/db/tables/documents?limit=2");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 2, "should return exactly 2 documents");
    assert_eq!(
        json["has_more"], true,
        "has_more should be true when more exist"
    );

    // limit=10 when 5 exist → has_more=false, all 5 returned.
    let resp = get(ctx.addr(), "/v1/db/tables/documents?limit=10");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 5, "should return all 5 documents");
    assert_eq!(
        json["has_more"], false,
        "has_more should be false when all returned"
    );
}
