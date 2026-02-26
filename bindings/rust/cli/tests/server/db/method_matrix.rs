//! Unsupported-method matrix for `/v1/db/*` endpoints.
//!
//! Contract: known DB routes with unsupported HTTP methods should return
//! `501 not_implemented` (not router-miss `404 not found`).

use super::db_config;
use crate::server::common::*;
use tempfile::TempDir;

fn request_for_method(addr: std::net::SocketAddr, method: &str, path: &str) -> HttpResponse {
    match method {
        "GET" => send_request(addr, "GET", path, &[], None),
        "POST" => send_request(
            addr,
            "POST",
            path,
            &[("Content-Type", "application/json")],
            Some("{}"),
        ),
        "PUT" => send_request(
            addr,
            "PUT",
            path,
            &[("Content-Type", "application/json")],
            Some("{}"),
        ),
        "PATCH" => send_request(
            addr,
            "PATCH",
            path,
            &[("Content-Type", "application/json")],
            Some("{}"),
        ),
        "DELETE" => send_request(addr, "DELETE", path, &[], None),
        other => panic!("unsupported method in matrix: {other}"),
    }
}

#[test]
fn db_unsupported_methods_return_not_implemented_matrix() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let cases: &[(&str, &str)] = &[
        // Table document endpoints.
        ("PUT", "/v1/db/tables/documents"),
        ("PATCH", "/v1/db/tables/documents"),
        ("DELETE", "/v1/db/tables/documents"),
        ("GET", "/v1/db/tables/documents/insert"),
        ("PATCH", "/v1/db/tables/documents/insert"),
        ("DELETE", "/v1/db/tables/documents/insert"),
        ("PUT", "/v1/db/tables/documents/insert"),
        ("GET", "/v1/db/tables/documents/search"),
        ("PATCH", "/v1/db/tables/documents/search"),
        ("DELETE", "/v1/db/tables/documents/search"),
        ("PUT", "/v1/db/tables/documents/search"),
        ("POST", "/v1/db/tables/documents/doc-1"),
        ("PUT", "/v1/db/tables/documents/doc-1"),
        ("PUT", "/v1/db/tables/documents/doc-1/tags"),
        ("PATCH", "/v1/db/tables/documents/doc-1/tags"),
        // Low-level table rows/meta endpoints.
        ("PUT", "/v1/db/tables/rowsns/rows"),
        ("PATCH", "/v1/db/tables/rowsns/rows"),
        ("DELETE", "/v1/db/tables/rowsns/rows"),
        ("POST", "/v1/db/tables/rowsns/rows/123"),
        ("PUT", "/v1/db/tables/rowsns/rows/123"),
        ("PATCH", "/v1/db/tables/rowsns/rows/123"),
        ("GET", "/v1/db/tables/rowsns/rows/scan"),
        ("PUT", "/v1/db/tables/rowsns/rows/scan"),
        ("PATCH", "/v1/db/tables/rowsns/rows/scan"),
        ("DELETE", "/v1/db/tables/rowsns/rows/scan"),
        ("POST", "/v1/db/tables/_meta/namespaces"),
        ("PUT", "/v1/db/tables/_meta/namespaces"),
        ("PATCH", "/v1/db/tables/_meta/namespaces"),
        ("DELETE", "/v1/db/tables/_meta/namespaces"),
        ("POST", "/v1/db/tables/rowsns/_meta/policy"),
        ("PUT", "/v1/db/tables/rowsns/_meta/policy"),
        ("PATCH", "/v1/db/tables/rowsns/_meta/policy"),
        ("DELETE", "/v1/db/tables/rowsns/_meta/policy"),
        // Vector endpoints.
        ("PATCH", "/v1/db/vectors/collections"),
        ("PUT", "/v1/db/vectors/collections"),
        ("DELETE", "/v1/db/vectors/collections"),
        ("POST", "/v1/db/vectors/collections/coll"),
        ("PUT", "/v1/db/vectors/collections/coll"),
        ("PATCH", "/v1/db/vectors/collections/coll"),
        ("GET", "/v1/db/vectors/collections/coll/points/append"),
        ("PUT", "/v1/db/vectors/collections/coll/points/append"),
        ("PATCH", "/v1/db/vectors/collections/coll/points/append"),
        ("DELETE", "/v1/db/vectors/collections/coll/points/append"),
        ("GET", "/v1/db/vectors/collections/coll/points/upsert"),
        ("PUT", "/v1/db/vectors/collections/coll/points/upsert"),
        ("PATCH", "/v1/db/vectors/collections/coll/points/upsert"),
        ("DELETE", "/v1/db/vectors/collections/coll/points/upsert"),
        ("GET", "/v1/db/vectors/collections/coll/points/delete"),
        ("PUT", "/v1/db/vectors/collections/coll/points/delete"),
        ("PATCH", "/v1/db/vectors/collections/coll/points/delete"),
        ("DELETE", "/v1/db/vectors/collections/coll/points/delete"),
        ("GET", "/v1/db/vectors/collections/coll/points/fetch"),
        ("PUT", "/v1/db/vectors/collections/coll/points/fetch"),
        ("PATCH", "/v1/db/vectors/collections/coll/points/fetch"),
        ("DELETE", "/v1/db/vectors/collections/coll/points/fetch"),
        ("GET", "/v1/db/vectors/collections/coll/points/query"),
        ("PUT", "/v1/db/vectors/collections/coll/points/query"),
        ("PATCH", "/v1/db/vectors/collections/coll/points/query"),
        ("DELETE", "/v1/db/vectors/collections/coll/points/query"),
        ("POST", "/v1/db/vectors/collections/coll/stats"),
        ("PUT", "/v1/db/vectors/collections/coll/stats"),
        ("PATCH", "/v1/db/vectors/collections/coll/stats"),
        ("DELETE", "/v1/db/vectors/collections/coll/stats"),
        ("GET", "/v1/db/vectors/collections/coll/compact"),
        ("PUT", "/v1/db/vectors/collections/coll/compact"),
        ("PATCH", "/v1/db/vectors/collections/coll/compact"),
        ("DELETE", "/v1/db/vectors/collections/coll/compact"),
        ("GET", "/v1/db/vectors/collections/coll/indexes/build"),
        ("PUT", "/v1/db/vectors/collections/coll/indexes/build"),
        ("PATCH", "/v1/db/vectors/collections/coll/indexes/build"),
        ("DELETE", "/v1/db/vectors/collections/coll/indexes/build"),
        ("POST", "/v1/db/vectors/collections/coll/changes"),
        ("PUT", "/v1/db/vectors/collections/coll/changes"),
        ("PATCH", "/v1/db/vectors/collections/coll/changes"),
        ("DELETE", "/v1/db/vectors/collections/coll/changes"),
        // KV endpoints.
        ("POST", "/v1/db/kv/namespaces/ns/entries"),
        ("PUT", "/v1/db/kv/namespaces/ns/entries"),
        ("PATCH", "/v1/db/kv/namespaces/ns/entries"),
        ("DELETE", "/v1/db/kv/namespaces/ns/entries"),
        ("POST", "/v1/db/kv/namespaces/ns/entries/key"),
        ("PATCH", "/v1/db/kv/namespaces/ns/entries/key"),
        ("GET", "/v1/db/kv/namespaces/ns/flush"),
        ("PUT", "/v1/db/kv/namespaces/ns/flush"),
        ("PATCH", "/v1/db/kv/namespaces/ns/flush"),
        ("DELETE", "/v1/db/kv/namespaces/ns/flush"),
        ("GET", "/v1/db/kv/namespaces/ns/compact"),
        ("PUT", "/v1/db/kv/namespaces/ns/compact"),
        ("PATCH", "/v1/db/kv/namespaces/ns/compact"),
        ("DELETE", "/v1/db/kv/namespaces/ns/compact"),
        // Blob endpoints.
        ("POST", "/v1/db/blobs"),
        ("PUT", "/v1/db/blobs"),
        ("PATCH", "/v1/db/blobs"),
        ("DELETE", "/v1/db/blobs"),
        ("POST", "/v1/db/blobs/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        ("PUT", "/v1/db/blobs/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        ("PATCH", "/v1/db/blobs/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        ("DELETE", "/v1/db/blobs/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        // SQL endpoints.
        ("GET", "/v1/db/sql/query"),
        ("PUT", "/v1/db/sql/query"),
        ("PATCH", "/v1/db/sql/query"),
        ("DELETE", "/v1/db/sql/query"),
        ("GET", "/v1/db/sql/explain"),
        ("PUT", "/v1/db/sql/explain"),
        ("PATCH", "/v1/db/sql/explain"),
        ("DELETE", "/v1/db/sql/explain"),
        // Ops endpoints.
        ("GET", "/v1/db/ops/compact"),
        ("PUT", "/v1/db/ops/compact"),
        ("PATCH", "/v1/db/ops/compact"),
        ("DELETE", "/v1/db/ops/compact"),
        ("GET", "/v1/db/ops/simulate_crash"),
        ("PUT", "/v1/db/ops/simulate_crash"),
        ("PATCH", "/v1/db/ops/simulate_crash"),
        ("DELETE", "/v1/db/ops/simulate_crash"),
    ];

    let mut failures = Vec::new();
    for (method, path) in cases {
        let resp = request_for_method(ctx.addr(), method, path);
        if resp.status != 501 {
            failures.push(format!(
                "{method} {path}: expected status=501/not_implemented, got status={} body={}",
                resp.status, resp.body
            ));
            continue;
        }
        let json: serde_json::Value = match serde_json::from_str(&resp.body) {
            Ok(v) => v,
            Err(err) => {
                failures.push(format!(
                    "{method} {path}: expected JSON not_implemented body, parse error={err}, body={}",
                    resp.body
                ));
                continue;
            }
        };
        if json["error"]["code"] != "not_implemented" {
            failures.push(format!(
                "{method} {path}: expected error.code=not_implemented, got body={}",
                resp.body
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "unsupported-method matrix violations ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
