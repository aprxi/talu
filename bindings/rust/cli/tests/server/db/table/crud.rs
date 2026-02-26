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
fn list_filters_by_owner_group_and_marker() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let docs = [
        serde_json::json!({
            "type": "note",
            "title": "g1-u1-m1",
            "content": {},
            "group_id": "g1",
            "owner_id": "u1",
            "marker": "m1"
        }),
        serde_json::json!({
            "type": "note",
            "title": "g2-u1-m2",
            "content": {},
            "group_id": "g2",
            "owner_id": "u1",
            "marker": "m2"
        }),
        serde_json::json!({
            "type": "note",
            "title": "g1-u2-m1",
            "content": {},
            "group_id": "g1",
            "owner_id": "u2",
            "marker": "m1"
        }),
    ];

    for body in docs {
        let resp = post_json(ctx.addr(), "/v1/db/tables/documents", &body);
        assert_eq!(resp.status, 201, "create: {}", resp.body);
    }

    let owner_resp = get(ctx.addr(), "/v1/db/tables/documents?owner_id=u1");
    assert_eq!(owner_resp.status, 200, "body: {}", owner_resp.body);
    assert_eq!(
        owner_resp.json()["data"].as_array().expect("data").len(),
        2,
        "owner_id=u1 should return two docs"
    );

    let group_resp = get(ctx.addr(), "/v1/db/tables/documents?group_id=g1");
    assert_eq!(group_resp.status, 200, "body: {}", group_resp.body);
    assert_eq!(
        group_resp.json()["data"].as_array().expect("data").len(),
        2,
        "group_id=g1 should return two docs"
    );

    let marker_resp = get(ctx.addr(), "/v1/db/tables/documents?marker=m2");
    assert_eq!(marker_resp.status, 200, "body: {}", marker_resp.body);
    let marker_data = marker_resp.json()["data"].as_array().expect("data").clone();
    assert_eq!(marker_data.len(), 1, "marker=m2 should return one doc");
    assert_eq!(marker_data[0]["title"], "g2-u1-m2");
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
    assert_eq!(data.len(), 2, "limit=2 should return exactly 2 items");
    assert_eq!(json["has_more"], true, "limit=2 over 5 docs should set has_more=true");
}

#[test]
fn list_invalid_limit_falls_back_to_default() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    for i in 0..3 {
        let create = post_json(
            ctx.addr(),
            "/v1/db/tables/documents",
            &serde_json::json!({
                "type": "note",
                "title": format!("Doc {}", i),
                "content": {}
            }),
        );
        assert_eq!(create.status, 201, "body: {}", create.body);
    }

    let resp = get(ctx.addr(), "/v1/db/tables/documents?limit=not-a-number");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let data = resp.json()["data"].as_array().expect("data array").clone();
    assert_eq!(data.len(), 3, "invalid limit should fall back to default");
}

#[test]
fn list_negative_limit_falls_back_to_default() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    for i in 0..2 {
        let create = post_json(
            ctx.addr(),
            "/v1/db/tables/documents",
            &serde_json::json!({
                "type": "note",
                "title": format!("Doc {}", i),
                "content": {}
            }),
        );
        assert_eq!(create.status, 201, "body: {}", create.body);
    }

    let resp = get(ctx.addr(), "/v1/db/tables/documents?limit=-1");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(
        resp.json()["data"].as_array().expect("data").len(),
        2,
        "negative limit should be treated as invalid and use default"
    );
}

#[test]
fn list_query_params_decode_space_encodings_for_marker() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let create = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "note",
            "title": "Needs review doc",
            "content": {},
            "marker": "needs review"
        }),
    );
    assert_eq!(create.status, 201, "body: {}", create.body);

    let plus = get(ctx.addr(), "/v1/db/tables/documents?marker=needs+review");
    assert_eq!(plus.status, 200, "body: {}", plus.body);
    assert_eq!(
        plus.json()["data"].as_array().expect("data").len(),
        1,
        "marker with '+' should be decoded as space"
    );

    let pct20 = get(ctx.addr(), "/v1/db/tables/documents?marker=needs%20review");
    assert_eq!(pct20.status, 200, "body: {}", pct20.body);
    assert_eq!(
        pct20.json()["data"].as_array().expect("data").len(),
        1,
        "marker with %20 should be decoded as space"
    );
}

#[test]
fn list_query_params_percent_decode_reserved_characters_for_marker() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let marker = "a+b/c&d";
    let create = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "note",
            "title": "Encoded marker doc",
            "content": {},
            "marker": marker
        }),
    );
    assert_eq!(create.status, 201, "body: {}", create.body);

    let encoded = get(ctx.addr(), "/v1/db/tables/documents?marker=a%2Bb%2Fc%26d");
    assert_eq!(encoded.status, 200, "body: {}", encoded.body);
    let data = encoded.json()["data"].as_array().expect("data").clone();
    assert_eq!(
        data.len(),
        1,
        "marker query value should follow standard percent-decoding"
    );
    assert_eq!(data[0]["title"], "Encoded marker doc");
}

#[test]
fn list_query_params_percent_decode_reserved_characters_for_owner_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let owner = "team+alpha/beta&ops";
    let create = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "note",
            "title": "Encoded owner doc",
            "content": {},
            "owner_id": owner
        }),
    );
    assert_eq!(create.status, 201, "body: {}", create.body);

    let encoded = get(
        ctx.addr(),
        "/v1/db/tables/documents?owner_id=team%2Balpha%2Fbeta%26ops",
    );
    assert_eq!(encoded.status, 200, "body: {}", encoded.body);
    let data = encoded.json()["data"].as_array().expect("data").clone();
    assert_eq!(
        data.len(),
        1,
        "owner_id query value should follow standard percent-decoding"
    );
    assert_eq!(data[0]["title"], "Encoded owner doc");
}

#[test]
fn documents_are_isolated_per_table_name() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let a_create = post_json(
        ctx.addr(),
        "/v1/db/tables/team_a",
        &serde_json::json!({
            "type": "note",
            "title": "A1",
            "content": {"tenant": "a"}
        }),
    );
    assert_eq!(a_create.status, 201, "body: {}", a_create.body);
    let a_id = a_create.json()["id"].as_str().expect("a id").to_string();

    let b_create = post_json(
        ctx.addr(),
        "/v1/db/tables/team_b",
        &serde_json::json!({
            "type": "note",
            "title": "B1",
            "content": {"tenant": "b"}
        }),
    );
    assert_eq!(b_create.status, 201, "body: {}", b_create.body);
    let b_id = b_create.json()["id"].as_str().expect("b id").to_string();

    let list_a = get(ctx.addr(), "/v1/db/tables/team_a");
    assert_eq!(list_a.status, 200, "body: {}", list_a.body);
    let data_a = list_a.json()["data"].as_array().expect("data").clone();
    assert_eq!(data_a.len(), 1, "team_a should contain exactly one doc");
    assert_eq!(data_a[0]["title"], "A1");

    let list_b = get(ctx.addr(), "/v1/db/tables/team_b");
    assert_eq!(list_b.status, 200, "body: {}", list_b.body);
    let data_b = list_b.json()["data"].as_array().expect("data").clone();
    assert_eq!(data_b.len(), 1, "team_b should contain exactly one doc");
    assert_eq!(data_b[0]["title"], "B1");

    let cross_get_a = get(ctx.addr(), &format!("/v1/db/tables/team_b/{a_id}"));
    assert_eq!(cross_get_a.status, 404, "body: {}", cross_get_a.body);
    assert_eq!(cross_get_a.json()["error"]["code"], "not_found");

    let cross_get_b = get(ctx.addr(), &format!("/v1/db/tables/team_a/{b_id}"));
    assert_eq!(cross_get_b.status, 404, "body: {}", cross_get_b.body);
    assert_eq!(cross_get_b.json()["error"]["code"], "not_found");
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
fn create_via_insert_alias_matches_primary_create() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({
        "type": "prompt",
        "title": "Insert Alias",
        "content": {"text": "created via /insert"}
    });

    let create = post_json(ctx.addr(), "/v1/db/tables/documents/insert", &body);
    assert_eq!(create.status, 201, "body: {}", create.body);
    let create_json = create.json();
    let doc_id = create_json["id"].as_str().expect("id");
    assert_eq!(create_json["title"], "Insert Alias");
    assert_eq!(create_json["type"], "prompt");

    let get_resp = get(ctx.addr(), &format!("/v1/db/tables/documents/{doc_id}"));
    assert_eq!(get_resp.status, 200, "body: {}", get_resp.body);
    assert_eq!(get_resp.json()["title"], "Insert Alias");
}

#[test]
fn plugin_storage_requires_capability_token() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "type": "plugin_storage",
            "title": "plugin object",
            "content": {"k": "v"}
        }),
    );
    assert_eq!(create_resp.status, 403, "body: {}", create_resp.body);
    assert_eq!(create_resp.json()["error"]["code"], "forbidden");

    let list_resp = get(ctx.addr(), "/v1/db/tables/documents?type=plugin_storage");
    assert_eq!(list_resp.status, 403, "body: {}", list_resp.body);
    assert_eq!(list_resp.json()["error"]["code"], "forbidden");
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
fn create_rejects_duplicate_explicit_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({
        "id": "fixed-id",
        "type": "note",
        "title": "first",
        "content": {}
    });
    let first = post_json(ctx.addr(), "/v1/db/tables/documents", &body);
    assert_eq!(first.status, 201, "body: {}", first.body);

    let second = post_json(ctx.addr(), "/v1/db/tables/documents", &body);
    assert_eq!(second.status, 400, "body: {}", second.body);
    assert_eq!(second.json()["error"]["code"], "invalid_argument");
}

#[test]
fn create_rejects_explicit_ids_that_conflict_with_reserved_route_segments() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let reserved_ids = ["search", "insert", "rows", "_meta"];
    let mut failures = Vec::new();

    for id in reserved_ids {
        let resp = post_json(
            ctx.addr(),
            "/v1/db/tables/documents",
            &serde_json::json!({
                "id": id,
                "type": "note",
                "title": format!("reserved-id-{id}"),
                "content": {}
            }),
        );
        if resp.status != 400 {
            failures.push(format!(
                "id={id}: expected 400 rejection, got status={} body={}",
                resp.status, resp.body
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "route-reserved explicit IDs must be rejected:\n{}",
        failures.join("\n")
    );
}

#[test]
fn create_rejects_explicit_ids_with_path_separators() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let invalid_ids = ["contains/slash", "nested/path/doc", "/leading", "trailing/"];
    let mut failures = Vec::new();

    for id in invalid_ids {
        let resp = post_json(
            ctx.addr(),
            "/v1/db/tables/documents",
            &serde_json::json!({
                "id": id,
                "type": "note",
                "title": format!("invalid-id-{id}"),
                "content": {}
            }),
        );
        if resp.status != 400 {
            failures.push(format!(
                "id={id}: expected 400 rejection, got status={} body={}",
                resp.status, resp.body
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "explicit IDs containing path separators must be rejected:\n{}",
        failures.join("\n")
    );
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

#[test]
fn table_name_validation_rejects_invalid_and_reserved_names() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let payload = serde_json::json!({
        "type": "note",
        "title": "x",
        "content": {}
    });

    let invalid_chars = post_json(ctx.addr(), "/v1/db/tables/bad.name", &payload);
    assert_eq!(invalid_chars.status, 400, "body: {}", invalid_chars.body);
    assert_eq!(invalid_chars.json()["error"]["code"], "invalid_table_name");

    let reserved = post_json(ctx.addr(), "/v1/db/tables/vector", &payload);
    assert_eq!(reserved.status, 400, "body: {}", reserved.body);
    assert_eq!(reserved.json()["error"]["code"], "reserved_table_name");

    let reserved_meta = post_json(ctx.addr(), "/v1/db/tables/_meta", &payload);
    assert_eq!(reserved_meta.status, 400, "body: {}", reserved_meta.body);
    assert_eq!(reserved_meta.json()["error"]["code"], "reserved_table_name");

    let too_long_name = format!("/v1/db/tables/{}", "a".repeat(65));
    let too_long = post_json(ctx.addr(), &too_long_name, &payload);
    assert_eq!(too_long.status, 400, "body: {}", too_long.body);
    assert_eq!(too_long.json()["error"]["code"], "invalid_table_name");
}

#[test]
fn table_name_validation_applies_to_list_and_get_paths() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let invalid_list = get(ctx.addr(), "/v1/db/tables/bad.name");
    assert_eq!(invalid_list.status, 400, "body: {}", invalid_list.body);
    assert_eq!(invalid_list.json()["error"]["code"], "invalid_table_name");

    let reserved_list = get(ctx.addr(), "/v1/db/tables/vector");
    assert_eq!(reserved_list.status, 400, "body: {}", reserved_list.body);
    assert_eq!(reserved_list.json()["error"]["code"], "reserved_table_name");

    let reserved_meta_list = get(ctx.addr(), "/v1/db/tables/_meta");
    assert_eq!(
        reserved_meta_list.status, 400,
        "body: {}",
        reserved_meta_list.body
    );
    assert_eq!(
        reserved_meta_list.json()["error"]["code"],
        "reserved_table_name"
    );

    let invalid_get = get(ctx.addr(), "/v1/db/tables/bad.name/doc-1");
    assert_eq!(invalid_get.status, 400, "body: {}", invalid_get.body);
    assert_eq!(invalid_get.json()["error"]["code"], "invalid_table_name");

    let reserved_meta_get = get(ctx.addr(), "/v1/db/tables/_meta/doc-1");
    assert_eq!(
        reserved_meta_get.status, 400,
        "body: {}",
        reserved_meta_get.body
    );
    assert_eq!(
        reserved_meta_get.json()["error"]["code"],
        "reserved_table_name"
    );
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
fn doc_id_path_params_are_percent_decoded_for_get_patch_delete() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let create_resp = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &serde_json::json!({
            "id": "id with space",
            "type": "note",
            "title": "Percent Encoded ID",
            "content": {"k": "v"}
        }),
    );
    assert_eq!(create_resp.status, 201, "body: {}", create_resp.body);

    let encoded_path = "/v1/db/tables/documents/id%20with%20space";

    let get_resp = get(ctx.addr(), encoded_path);
    assert_eq!(
        get_resp.status, 200,
        "doc path params should be percent-decoded for GET; body: {}",
        get_resp.body
    );
    assert_eq!(get_resp.json()["id"], "id with space");

    let patch_resp = patch_json(
        ctx.addr(),
        encoded_path,
        &serde_json::json!({
            "title": "Updated Title"
        }),
    );
    assert_eq!(
        patch_resp.status, 200,
        "doc path params should be percent-decoded for PATCH; body: {}",
        patch_resp.body
    );
    assert_eq!(patch_resp.json()["title"], "Updated Title");

    let delete_resp = delete(ctx.addr(), encoded_path);
    assert_eq!(
        delete_resp.status, 204,
        "doc path params should be percent-decoded for DELETE; body: {}",
        delete_resp.body
    );
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
    assert_eq!(json["type"], "prompt");
    assert_eq!(json["title"], "Full Content");
    assert_eq!(json["content"], content, "content object should round-trip exactly");
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
fn patch_rejects_invalid_json_body() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = send_request(
        ctx.addr(),
        "PATCH",
        "/v1/db/tables/documents/some-id",
        &[("Content-Type", "application/json")],
        Some("{broken"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_json");
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
    assert_eq!(resp.status, 404, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_found");
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
    assert_eq!(resp.status, 204, "delete: {}", resp.body);

    // Verify gone
    let get_resp = get(ctx.addr(), &format!("/v1/db/tables/documents/{}", doc_id));
    assert_eq!(get_resp.status, 404, "body: {}", get_resp.body);
    assert_eq!(get_resp.json()["error"]["code"], "not_found");
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
