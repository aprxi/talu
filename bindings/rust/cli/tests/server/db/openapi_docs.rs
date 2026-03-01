//! Integration tests for DB-scoped OpenAPI and docs endpoints.

use super::db_config;
use crate::server::common::*;
use tempfile::TempDir;

#[test]
fn docs_hubs_serve_navigation_index() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let docs = get(ctx.addr(), "/docs");
    assert_eq!(docs.status, 200, "body: {}", docs.body);
    let docs_type = docs.header("content-type").unwrap_or("");
    assert!(docs_type.contains("text/html"), "content-type={docs_type}");
    assert!(
        docs.body.contains("/docs/db/tables"),
        "expected /docs/db/tables link"
    );
    assert!(
        docs.body.contains("/openapi.json"),
        "expected /openapi.json link"
    );

    let db_docs = get(ctx.addr(), "/docs/db");
    assert_eq!(db_docs.status, 200, "body: {}", db_docs.body);
    let db_docs_type = db_docs.header("content-type").unwrap_or("");
    assert!(
        db_docs_type.contains("text/html"),
        "content-type={db_docs_type}"
    );
    assert!(
        db_docs.body.contains("/docs/db/kv"),
        "expected /docs/db/kv link"
    );
    assert!(
        db_docs.body.contains("/openapi/db.json"),
        "expected /openapi/db.json link"
    );
}

#[test]
fn docs_db_plane_pages_serve_swagger_html() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    for (path, expected_spec) in [
        ("/docs/db/tables", "/openapi/db/tables.json"),
        ("/docs/db/vectors", "/openapi/db/vectors.json"),
        ("/docs/db/kv", "/openapi/db/kv.json"),
        ("/docs/db/blobs", "/openapi/db/blobs.json"),
        ("/docs/db/sql", "/openapi/db/sql.json"),
        ("/docs/db/ops", "/openapi/db/ops.json"),
    ] {
        let resp = get(ctx.addr(), path);
        assert_eq!(resp.status, 200, "path={path} body={}", resp.body);
        let content_type = resp.header("content-type").unwrap_or("");
        assert!(
            content_type.contains("text/html"),
            "path={path} content-type={content_type}"
        );
        assert!(
            resp.body.contains(expected_spec),
            "path={path} expected spec URL {expected_spec} in body"
        );
        assert!(
            resp.body.contains(r#"href="/docs""#),
            "path={path} expected docs-home link in header"
        );
    }
}

#[test]
fn openapi_db_specs_are_plane_scoped() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let all_db = get(ctx.addr(), "/openapi/db.json");
    assert_eq!(all_db.status, 200, "body: {}", all_db.body);
    let all_db_json = all_db.json();
    let all_db_paths = all_db_json["paths"].as_object().expect("db paths object");
    assert!(!all_db_paths.is_empty(), "db paths should not be empty");
    assert!(
        all_db_paths.keys().all(|k| k.starts_with("/v1/db/")),
        "all /openapi/db.json paths must be /v1/db/*"
    );

    for (path, prefix) in [
        ("/openapi/db/tables.json", "/v1/db/tables/"),
        ("/openapi/db/vectors.json", "/v1/db/vectors/"),
        ("/openapi/db/kv.json", "/v1/db/kv/"),
        ("/openapi/db/blobs.json", "/v1/db/blobs"),
        ("/openapi/db/sql.json", "/v1/db/sql/"),
        ("/openapi/db/ops.json", "/v1/db/ops/"),
    ] {
        let resp = get(ctx.addr(), path);
        assert_eq!(resp.status, 200, "path={path} body={}", resp.body);
        let json = resp.json();
        let paths = json["paths"].as_object().expect("paths object");
        assert!(
            paths.keys().all(|k| k.starts_with(prefix)),
            "path={path} contains non-{prefix} endpoint"
        );
    }
}

#[test]
fn openapi_sql_spec_includes_query_and_explain_paths() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let resp = get(ctx.addr(), "/openapi/db/sql.json");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let paths = json["paths"].as_object().expect("paths object");

    let query = paths
        .get("/v1/db/sql/query")
        .expect("missing /v1/db/sql/query");
    assert!(
        query.get("post").is_some(),
        "/v1/db/sql/query should expose POST"
    );

    let explain = paths
        .get("/v1/db/sql/explain")
        .expect("missing /v1/db/sql/explain");
    assert!(
        explain.get("post").is_some(),
        "/v1/db/sql/explain should expose POST"
    );
}

#[test]
fn openapi_plane_specs_include_expected_paths_and_methods() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let cases: [(&str, &[(&str, &[&str])]); 6] = [
        (
            "/openapi/db/tables.json",
            &[
                ("/v1/db/tables/{table}", &["get", "post"]),
                ("/v1/db/tables/{table}/insert", &["post"]),
                (
                    "/v1/db/tables/{table}/{doc_id}",
                    &["get", "patch", "delete"],
                ),
                ("/v1/db/tables/{table}/search", &["post"]),
                (
                    "/v1/db/tables/{table}/{doc_id}/tags",
                    &["get", "post", "delete"],
                ),
            ],
        ),
        (
            "/openapi/db/vectors.json",
            &[
                ("/v1/db/vectors/collections", &["get", "post"]),
                ("/v1/db/vectors/collections/{name}", &["get", "delete"]),
                ("/v1/db/vectors/collections/{name}/points/append", &["post"]),
                ("/v1/db/vectors/collections/{name}/points/upsert", &["post"]),
                ("/v1/db/vectors/collections/{name}/points/delete", &["post"]),
                ("/v1/db/vectors/collections/{name}/points/fetch", &["post"]),
                ("/v1/db/vectors/collections/{name}/points/query", &["post"]),
                ("/v1/db/vectors/collections/{name}/stats", &["get"]),
                ("/v1/db/vectors/collections/{name}/compact", &["post"]),
                ("/v1/db/vectors/collections/{name}/indexes/build", &["post"]),
                ("/v1/db/vectors/collections/{name}/changes", &["get"]),
            ],
        ),
        (
            "/openapi/db/kv.json",
            &[
                ("/v1/db/kv/namespaces/{namespace}/entries", &["get"]),
                (
                    "/v1/db/kv/namespaces/{namespace}/entries/{key}",
                    &["put", "get", "delete"],
                ),
                ("/v1/db/kv/namespaces/{namespace}/flush", &["post"]),
                ("/v1/db/kv/namespaces/{namespace}/compact", &["post"]),
            ],
        ),
        (
            "/openapi/db/blobs.json",
            &[
                ("/v1/db/blobs", &["get"]),
                ("/v1/db/blobs/{blob_ref}", &["get"]),
            ],
        ),
        (
            "/openapi/db/sql.json",
            &[
                ("/v1/db/sql/query", &["post"]),
                ("/v1/db/sql/explain", &["post"]),
            ],
        ),
        (
            "/openapi/db/ops.json",
            &[
                ("/v1/db/ops/compact", &["post"]),
                ("/v1/db/ops/simulate_crash", &["post"]),
            ],
        ),
    ];

    for (spec_path, expected) in cases {
        let resp = get(ctx.addr(), spec_path);
        assert_eq!(resp.status, 200, "spec={spec_path} body={}", resp.body);
        let json = resp.json();
        let paths = json["paths"].as_object().expect("paths object");

        for (path, methods) in expected {
            let entry = paths
                .get(*path)
                .unwrap_or_else(|| panic!("spec={spec_path} missing path {path}"));
            for method in *methods {
                assert!(
                    entry.get(*method).is_some(),
                    "spec={spec_path} path={path} missing method {}",
                    method
                );
            }
        }
    }
}
