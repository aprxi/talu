//! Document tag operations tests.
//!
//! Tests tag endpoints for documents:
//! - GET /v1/db/tables/documents/:id/tags
//! - POST /v1/db/tables/documents/:id/tags
//! - DELETE /v1/db/tables/documents/:id/tags

use super::{documents_config, no_bucket_config};
use crate::server::common::*;
use std::collections::BTreeSet;
use tempfile::TempDir;

fn tag_names_from_response(resp: &HttpResponse) -> BTreeSet<String> {
    resp.json()["tags"]
        .as_array()
        .expect("tags array")
        .iter()
        .filter_map(|tag| {
            tag.as_str()
                .map(ToOwned::to_owned)
                .or_else(|| tag["name"].as_str().map(ToOwned::to_owned))
                .or_else(|| tag["id"].as_str().map(ToOwned::to_owned))
        })
        .collect()
}

// =============================================================================
// GET /v1/db/tables/documents/:id/tags
// =============================================================================

#[test]
fn get_tags_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = get(ctx.addr(), "/v1/db/tables/documents/some-id/tags");
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn get_tags_returns_empty_for_missing_document() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/db/tables/documents/nonexistent/tags");
    assert_eq!(resp.status, 404, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_found");
}

#[test]
fn get_tags_returns_empty_initially() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create document
    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "No Tags", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    // Get tags
    let resp = get(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert!(json["tags"].is_array(), "should have tags array");
    assert_eq!(json["tags"], serde_json::json!([]));
}

#[test]
fn get_tags_returns_added_tags() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create document
    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Tagged Doc", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    // Add tags
    post_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &serde_json::json!({
            "tags": ["coding", "rust"]
        }),
    );

    // Get tags
    let resp = get(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let tags = tag_names_from_response(&resp);
    assert!(tags.contains("coding"), "missing tag 'coding': {tags:?}");
    assert!(tags.contains("rust"), "missing tag 'rust': {tags:?}");
}

#[test]
fn tags_doc_id_path_params_are_percent_decoded() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "id": "doc with space",
            "type": "note",
            "title": "Encoded Tag ID",
            "content": {}
        }),
    );
    assert_eq!(create_resp.status, 201, "body: {}", create_resp.body);

    let encoded = "/v1/db/tables/documents/doc%20with%20space/tags";
    let add_resp = post_json(
        ctx.addr(),
        encoded,
        &serde_json::json!({
            "tags": ["encoded"]
        }),
    );
    assert_eq!(
        add_resp.status, 200,
        "doc id path params should be percent-decoded for tags add; body: {}",
        add_resp.body
    );

    let get_resp = get(ctx.addr(), encoded);
    assert_eq!(
        get_resp.status, 200,
        "doc id path params should be percent-decoded for tags get; body: {}",
        get_resp.body
    );
    let tags = tag_names_from_response(&get_resp);
    assert!(tags.contains("encoded"), "missing tag 'encoded': {tags:?}");

    let del_resp = delete_json(
        ctx.addr(),
        encoded,
        &serde_json::json!({ "tags": ["encoded"] }),
    );
    assert_eq!(
        del_resp.status, 200,
        "doc id path params should be percent-decoded for tags delete; body: {}",
        del_resp.body
    );
}

// =============================================================================
// POST /v1/db/tables/documents/:id/tags
// =============================================================================

#[test]
fn add_tags_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let body = serde_json::json!({"tags": ["test"]});
    let resp = post_json(ctx.addr(), "/v1/db/tables/documents/some-id/tags", &body);
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn add_tags_handles_missing_document() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({"tags": ["test"]});
    let resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents/nonexistent/tags",
        &body,
    );
    assert_eq!(resp.status, 404, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_found");
}

#[test]
fn add_tags_rejects_invalid_json_shapes() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let malformed = send_request(
        ctx.addr(),
        "POST",
        "/v1/db/tables/documents/some-id/tags",
        &[("Content-Type", "application/json")],
        Some("{broken"),
    );
    assert_eq!(malformed.status, 400, "body: {}", malformed.body);
    assert_eq!(malformed.json()["error"]["code"], "invalid_json");

    let missing_tags = post_json(
        ctx.addr(),
        "/v1/db/tables/documents/some-id/tags",
        &serde_json::json!({}),
    );
    assert_eq!(missing_tags.status, 400, "body: {}", missing_tags.body);
    assert_eq!(missing_tags.json()["error"]["code"], "invalid_json");
}

#[test]
fn add_tags_succeeds() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Create document
    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Add Tags Test", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    // Add tags
    let body = serde_json::json!({"tags": ["important", "urgent"]});
    let resp = post_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &body,
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let tags = tag_names_from_response(&resp);
    assert!(tags.contains("important"), "missing tag 'important': {tags:?}");
    assert!(tags.contains("urgent"), "missing tag 'urgent': {tags:?}");
}

#[test]
fn add_tags_with_empty_array_is_noop() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "No-op add", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    let add = post_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &serde_json::json!({"tags": ["seed"]}),
    );
    assert_eq!(add.status, 200, "body: {}", add.body);

    let noop = post_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &serde_json::json!({"tags": []}),
    );
    assert_eq!(noop.status, 200, "body: {}", noop.body);
    let tags = tag_names_from_response(&noop);
    assert_eq!(tags, ["seed".to_string()].into_iter().collect());
}

#[test]
fn add_tags_idempotent() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Idempotent Test", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    // Add same tag twice
    let body = serde_json::json!({"tags": ["duplicate"]});
    post_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &body,
    );
    let resp = post_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &body,
    );
    assert_eq!(resp.status, 200, "adding same tag twice should succeed");

    // Should only have one instance
    let get_resp = get(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
    );
    let tags = tag_names_from_response(&get_resp);
    assert_eq!(
        tags.iter().filter(|t| t.as_str() == "duplicate").count(),
        1,
        "tag should not be duplicated: {tags:?}"
    );
}

#[test]
fn add_multiple_tags_at_once() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Multi Tags", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    let body = serde_json::json!({"tags": ["tag1", "tag2", "tag3"]});
    let resp = post_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &body,
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let get_resp = get(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
    );
    let tags = tag_names_from_response(&get_resp);
    let expected: BTreeSet<String> = ["tag1", "tag2", "tag3"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(tags, expected, "tags should match exactly");
}

// =============================================================================
// DELETE /v1/db/tables/documents/:id/tags
// =============================================================================

#[test]
fn remove_tags_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = delete_json(
        ctx.addr(),
        "/v1/db/tables/documents/some-id/tags",
        &serde_json::json!({"tags": ["test"]}),
    );
    assert_eq!(resp.status, 503, "body: {}", resp.body);
}

#[test]
fn remove_tags_handles_missing_document() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = delete_json(
        ctx.addr(),
        "/v1/db/tables/documents/nonexistent/tags",
        &serde_json::json!({"tags": ["test"]}),
    );
    assert_eq!(resp.status, 404, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_found");
}

#[test]
fn remove_tags_rejects_invalid_json_shapes() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let malformed = send_request(
        ctx.addr(),
        "DELETE",
        "/v1/db/tables/documents/some-id/tags",
        &[("Content-Type", "application/json")],
        Some("{broken"),
    );
    assert_eq!(malformed.status, 400, "body: {}", malformed.body);
    assert_eq!(malformed.json()["error"]["code"], "invalid_json");

    let missing_tags = delete_json(
        ctx.addr(),
        "/v1/db/tables/documents/some-id/tags",
        &serde_json::json!({}),
    );
    assert_eq!(missing_tags.status, 400, "body: {}", missing_tags.body);
    assert_eq!(missing_tags.json()["error"]["code"], "invalid_json");
}

#[test]
fn remove_tags_succeeds() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Remove Tags Test", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    // Add tags
    post_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &serde_json::json!({"tags": ["keep", "remove"]}),
    );

    // Remove one tag
    let resp = delete_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &serde_json::json!({"tags": ["remove"]}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    // Verify
    let get_resp = get(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
    );
    let tags = tag_names_from_response(&get_resp);
    assert!(!tags.contains("remove"), "removed tag should be gone");
    assert!(tags.contains("keep"), "remaining tag should still exist");
}

#[test]
fn remove_tags_with_empty_array_is_noop() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "No-op remove", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    let add = post_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &serde_json::json!({"tags": ["keep"]}),
    );
    assert_eq!(add.status, 200, "body: {}", add.body);

    let noop = delete_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &serde_json::json!({"tags": []}),
    );
    assert_eq!(noop.status, 200, "body: {}", noop.body);
    let tags = tag_names_from_response(&noop);
    assert_eq!(tags, ["keep".to_string()].into_iter().collect());
}

#[test]
fn remove_nonexistent_tag_safe() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "prompt", "title": "Safe Remove Test", "content": {}
        }),
    );
    let doc_id = create_resp.json()["id"].as_str().expect("id").to_string();

    // Remove tag that was never added - should not error
    let resp = delete_json(
        ctx.addr(),
        &format!("/v1/db/tables/documents/{}/tags", doc_id),
        &serde_json::json!({"tags": ["never-existed"]}),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let tags = tag_names_from_response(&resp);
    assert!(
        !tags.contains("never-existed"),
        "non-existent tag should remain absent"
    );
}

#[test]
fn tags_reject_invalid_and_reserved_table_names() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let get_bad = get(ctx.addr(), "/v1/db/tables/bad.name/some-id/tags");
    assert_eq!(get_bad.status, 400, "body: {}", get_bad.body);
    assert_eq!(get_bad.json()["error"]["code"], "invalid_table_name");

    let add_reserved = post_json(
        ctx.addr(),
        "/v1/db/tables/vector/some-id/tags",
        &serde_json::json!({"tags": ["x"]}),
    );
    assert_eq!(add_reserved.status, 400, "body: {}", add_reserved.body);
    assert_eq!(add_reserved.json()["error"]["code"], "reserved_table_name");

    let remove_bad = delete_json(
        ctx.addr(),
        "/v1/db/tables/bad.name/some-id/tags",
        &serde_json::json!({"tags": ["x"]}),
    );
    assert_eq!(remove_bad.status, 400, "body: {}", remove_bad.body);
    assert_eq!(remove_bad.json()["error"]["code"], "invalid_table_name");
}

#[test]
fn tags_are_isolated_per_table_name_even_with_same_doc_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));
    let doc_id = "shared-doc-id";

    let create_a = post_json(
        ctx.addr(),
        "/v1/db/tables/team_a",
        &serde_json::json!({
            "id": doc_id,
            "type": "note",
            "title": "A",
            "content": {}
        }),
    );
    assert_eq!(create_a.status, 201, "body: {}", create_a.body);

    let create_b = post_json(
        ctx.addr(),
        "/v1/db/tables/team_b",
        &serde_json::json!({
            "id": doc_id,
            "type": "note",
            "title": "B",
            "content": {}
        }),
    );
    assert_eq!(create_b.status, 201, "body: {}", create_b.body);

    let add_a = post_json(
        ctx.addr(),
        &format!("/v1/db/tables/team_a/{doc_id}/tags"),
        &serde_json::json!({"tags": ["a-only"]}),
    );
    assert_eq!(add_a.status, 200, "body: {}", add_a.body);

    let add_b = post_json(
        ctx.addr(),
        &format!("/v1/db/tables/team_b/{doc_id}/tags"),
        &serde_json::json!({"tags": ["b-only"]}),
    );
    assert_eq!(add_b.status, 200, "body: {}", add_b.body);

    let get_a = get(ctx.addr(), &format!("/v1/db/tables/team_a/{doc_id}/tags"));
    assert_eq!(get_a.status, 200, "body: {}", get_a.body);
    let tags_a = tag_names_from_response(&get_a);
    assert!(tags_a.contains("a-only"), "team_a missing tag: {tags_a:?}");
    assert!(!tags_a.contains("b-only"), "team_a leaked team_b tag: {tags_a:?}");

    let get_b = get(ctx.addr(), &format!("/v1/db/tables/team_b/{doc_id}/tags"));
    assert_eq!(get_b.status, 200, "body: {}", get_b.body);
    let tags_b = tag_names_from_response(&get_b);
    assert!(tags_b.contains("b-only"), "team_b missing tag: {tags_b:?}");
    assert!(!tags_b.contains("a-only"), "team_b leaked team_a tag: {tags_b:?}");
}
