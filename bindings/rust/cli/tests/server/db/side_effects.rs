//! Failure-path side-effect contracts for `/v1/db/*`.
//!
//! Errors must be safe: no partial writes, no hidden state changes.

use super::db_config;
use crate::server::common::*;
use serde_json::json;
use tempfile::TempDir;

#[test]
fn invalid_document_create_does_not_mutate_table() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let before = get(ctx.addr(), "/v1/db/tables/documents");
    assert_eq!(before.status, 200, "body: {}", before.body);
    assert_eq!(
        before.json()["data"].as_array().expect("data").len(),
        0,
        "precondition: documents table must be empty"
    );

    let invalid = post_json(
        ctx.addr(),
        "/v1/db/tables/documents",
        &json!({
            "title": "missing type",
            "content": {}
        }),
    );
    assert_eq!(invalid.status, 400, "body: {}", invalid.body);

    let after = get(ctx.addr(), "/v1/db/tables/documents");
    assert_eq!(after.status, 200, "body: {}", after.body);
    assert_eq!(
        after.json()["data"].as_array().expect("data").len(),
        0,
        "invalid create must not persist any document"
    );
}

#[test]
fn vector_append_dimension_error_does_not_persist_partial_rows() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let create = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections",
        &json!({
            "name": "emb",
            "dims": 3
        }),
    );
    assert_eq!(create.status, 201, "body: {}", create.body);

    let bad_append = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [
                { "id": 1, "values": [1.0, 0.0, 0.0] },
                { "id": 2, "values": [1.0, 0.0] }
            ]
        }),
    );
    assert_eq!(bad_append.status, 400, "body: {}", bad_append.body);
    assert_eq!(bad_append.json()["error"]["code"], "dimension_mismatch");

    let query = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/query",
        &json!({
            "vector": [1.0, 0.0, 0.0],
            "top_k": 10
        }),
    );
    assert_eq!(query.status, 200, "body: {}", query.body);
    let query_json = query.json();
    let matches = query_json["results"][0]["matches"]
        .as_array()
        .expect("matches");
    assert_eq!(
        matches.len(),
        0,
        "failed append must not persist any subset of points"
    );
}

#[test]
fn kv_invalid_path_error_does_not_create_entries() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let invalid_put = send_request(
        ctx.addr(),
        "PUT",
        "/v1/db/kv/namespaces/ns/entries/",
        &[("Content-Type", "application/octet-stream")],
        Some("value"),
    );
    assert_eq!(invalid_put.status, 400, "body: {}", invalid_put.body);
    assert_eq!(invalid_put.json()["error"]["code"], "invalid_path");

    let list = get(ctx.addr(), "/v1/db/kv/namespaces/ns/entries");
    assert_eq!(list.status, 200, "body: {}", list.body);
    assert_eq!(
        list.json()["count"],
        0,
        "invalid path must not create entries"
    );
}

#[test]
fn unauthorized_table_rows_write_does_not_mutate_tenant_storage() {
    let temp = TempDir::new().expect("temp dir");
    let mut cfg = db_config(temp.path());
    cfg.gateway_secret = Some("secret".to_string());
    cfg.tenants = vec![TenantSpec {
        id: "tenant-a".to_string(),
        storage_prefix: "tenant-a".to_string(),
        allowed_models: vec![],
    }];
    let ctx = ServerTestContext::new(cfg);

    let write = post_json(
        ctx.addr(),
        "/v1/db/tables/isolated/rows",
        &json!({
            "schema_id": 10,
            "columns": [
                {"column_id": 1, "type": "scalar_u64", "value": 1},
                {"column_id": 2, "type": "scalar_i64", "value": 123},
                {"column_id": 20, "type": "string", "value": "unauth-write"}
            ]
        }),
    );
    assert_eq!(write.status, 401, "body: {}", write.body);
    assert_eq!(write.json()["error"]["code"], "unauthorized");

    let authorized_scan = send_request(
        ctx.addr(),
        "GET",
        "/v1/db/tables/isolated/rows?schema_id=10",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "tenant-a"),
        ],
        None,
    );
    assert_eq!(
        authorized_scan.status, 200,
        "body: {}",
        authorized_scan.body
    );
    assert_eq!(
        authorized_scan.json()["rows"]
            .as_array()
            .expect("rows")
            .len(),
        0,
        "unauthorized write attempt must not mutate tenant data"
    );
}
