//! Cross-plane error payload contracts for `/v1/db/*`.

use super::db_config;
use crate::server::common::*;
use serde_json::Value;
use tempfile::TempDir;

fn assert_structured_error_json(resp: &HttpResponse) -> Value {
    let content_type = resp.header("content-type").unwrap_or("");
    assert!(
        content_type.starts_with("application/json"),
        "error responses should be JSON; got content-type={content_type}, body={}",
        resp.body
    );
    let json = resp.json();
    assert!(
        json["error"]["code"].is_string(),
        "error.code must be string; body={}",
        resp.body
    );
    assert!(
        json["error"]["message"].is_string(),
        "error.message must be string; body={}",
        resp.body
    );
    json
}

#[test]
fn known_db_resource_not_found_errors_use_structured_json_shape() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let docs_missing = get(ctx.addr(), "/v1/db/tables/documents/not-there");
    assert_eq!(docs_missing.status, 404, "body: {}", docs_missing.body);
    assert_eq!(
        assert_structured_error_json(&docs_missing)["error"]["code"],
        "not_found"
    );

    let vector_missing = get(ctx.addr(), "/v1/db/vectors/collections/missing/stats");
    assert_eq!(vector_missing.status, 404, "body: {}", vector_missing.body);
    assert_eq!(
        assert_structured_error_json(&vector_missing)["error"]["code"],
        "collection_not_found"
    );

    let kv_missing = get(ctx.addr(), "/v1/db/kv/namespaces/ns/entries/missing");
    assert_eq!(kv_missing.status, 404, "body: {}", kv_missing.body);
    assert_eq!(
        assert_structured_error_json(&kv_missing)["error"]["code"],
        "not_found"
    );

    let blob_missing = get(
        ctx.addr(),
        "/v1/db/blobs/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    );
    assert_eq!(blob_missing.status, 404, "body: {}", blob_missing.body);
    assert_eq!(
        assert_structured_error_json(&blob_missing)["error"]["code"],
        "not_found"
    );
}

#[test]
fn malformed_known_db_paths_return_structured_bad_request_errors() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let cases: &[(&str, &str)] = &[
        ("GET", "/v1/db/tables//rows?schema_id=10"),
        ("GET", "/v1/db/tables/_meta"),
        ("GET", "/v1/db/vectors/collections/bad%2Fname/stats"),
        ("GET", "/v1/db/kv/namespaces//entries"),
    ];

    let mut failures = Vec::new();
    for (method, path) in cases {
        let resp = send_request(ctx.addr(), method, path, &[], None);
        if resp.status != 400 {
            failures.push(format!(
                "{method} {path}: expected 400 structured bad request, got status={} body={}",
                resp.status, resp.body
            ));
            continue;
        }
        let json = match serde_json::from_str::<serde_json::Value>(&resp.body) {
            Ok(v) => v,
            Err(err) => {
                failures.push(format!(
                    "{method} {path}: expected JSON error body, parse error={err}, body={}",
                    resp.body
                ));
                continue;
            }
        };
        if !json["error"]["code"].is_string() || !json["error"]["message"].is_string() {
            failures.push(format!(
                "{method} {path}: expected structured error shape, body={}",
                resp.body
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "malformed-path bad-request contract violations ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
