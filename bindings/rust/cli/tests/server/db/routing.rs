//! Routing contract tests for `/v1/db/*` endpoints.

use super::db_config;
use crate::server::common::*;
use tempfile::TempDir;

#[test]
fn known_db_paths_with_wrong_methods_return_not_implemented() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let sql_get = send_request(ctx.addr(), "GET", "/v1/db/sql/query", &[], None);
    assert_eq!(sql_get.status, 501, "body: {}", sql_get.body);
    assert_eq!(sql_get.json()["error"]["code"], "not_implemented");

    let sql_explain_get = send_request(ctx.addr(), "GET", "/v1/db/sql/explain", &[], None);
    assert_eq!(sql_explain_get.status, 501, "body: {}", sql_explain_get.body);
    assert_eq!(sql_explain_get.json()["error"]["code"], "not_implemented");

    let blobs_post = send_request(
        ctx.addr(),
        "POST",
        "/v1/db/blobs",
        &[("Content-Type", "application/json")],
        Some("{}"),
    );
    assert_eq!(blobs_post.status, 501, "body: {}", blobs_post.body);
    assert_eq!(blobs_post.json()["error"]["code"], "not_implemented");

    let vectors_patch = send_request(
        ctx.addr(),
        "PATCH",
        "/v1/db/vectors/collections",
        &[("Content-Type", "application/json")],
        Some("{}"),
    );
    assert_eq!(vectors_patch.status, 501, "body: {}", vectors_patch.body);
    assert_eq!(vectors_patch.json()["error"]["code"], "not_implemented");

    let kv_patch = send_request(
        ctx.addr(),
        "PATCH",
        "/v1/db/kv/namespaces/ns/entries/key",
        &[("Content-Type", "application/json")],
        Some("{}"),
    );
    assert_eq!(kv_patch.status, 501, "body: {}", kv_patch.body);
    assert_eq!(kv_patch.json()["error"]["code"], "not_implemented");
}

#[test]
fn unknown_db_paths_return_404_plain_not_found() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let unknown = get(ctx.addr(), "/v1/db/not-a-real-endpoint");
    assert_eq!(unknown.status, 404, "body: {}", unknown.body);
    assert_eq!(unknown.body, "not found");

    let non_v1 = send_request(
        ctx.addr(),
        "POST",
        "/db/sql/query",
        &[("Content-Type", "application/json")],
        Some(r#"{"query":"SELECT 1"}"#),
    );
    assert_eq!(non_v1.status, 404, "body: {}", non_v1.body);
    assert_eq!(non_v1.body, "not found");
}

#[test]
fn known_dynamic_db_paths_with_wrong_methods_return_not_implemented() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let vector_query_get = send_request(
        ctx.addr(),
        "GET",
        "/v1/db/vectors/collections/emb/points/query",
        &[],
        None,
    );
    assert_eq!(vector_query_get.status, 501, "body: {}", vector_query_get.body);
    assert_eq!(
        vector_query_get.json()["error"]["code"],
        "not_implemented",
        "wrong method on known vector path should return not_implemented"
    );

    let table_search_patch = send_request(
        ctx.addr(),
        "PATCH",
        "/v1/db/tables/documents/search",
        &[("Content-Type", "application/json")],
        Some(r#"{"query":"hello"}"#),
    );
    assert_eq!(table_search_patch.status, 501, "body: {}", table_search_patch.body);
    assert_eq!(table_search_patch.json()["error"]["code"], "not_implemented");
}
