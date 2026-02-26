//! Gateway auth contracts for `/v1/db/*` endpoints.

use super::db_config;
use crate::server::common::*;
use tempfile::TempDir;

fn gateway_db_config(bucket: &std::path::Path) -> ServerConfig {
    let mut cfg = db_config(bucket);
    cfg.gateway_secret = Some("secret".to_string());
    cfg.tenants = vec![TenantSpec {
        id: "tenant-a".to_string(),
        storage_prefix: "tenant-a".to_string(),
        allowed_models: vec![],
    }];
    cfg
}

fn request_with_method_and_body(
    addr: std::net::SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&str>,
) -> HttpResponse {
    let mut request_headers = Vec::with_capacity(headers.len() + 1);
    if body.is_some() {
        let content_type = if method == "PUT" {
            "application/octet-stream"
        } else {
            "application/json"
        };
        request_headers.push(("Content-Type", content_type));
    }
    request_headers.extend_from_slice(headers);
    send_request(addr, method, path, &request_headers, body)
}

#[test]
fn db_endpoints_require_secret_when_gateway_enabled() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(gateway_db_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/db/blobs");
    assert_eq!(resp.status, 401, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "unauthorized");
}

#[test]
fn db_endpoints_require_tenant_when_gateway_enabled() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(gateway_db_config(temp.path()));

    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/db/blobs",
        &[("X-Talu-Gateway-Secret", "secret")],
        None,
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "forbidden");
}

#[test]
fn db_endpoints_reject_unknown_tenant() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(gateway_db_config(temp.path()));

    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/db/kv/namespaces/ns/entries",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "tenant-missing"),
        ],
        None,
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "forbidden");
}

#[test]
fn db_endpoints_reject_invalid_secret_even_for_unknown_paths() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(gateway_db_config(temp.path()));

    let known = send_request(
        ctx.addr(),
        "POST",
        "/v1/db/sql/query",
        &[
            ("Content-Type", "application/json"),
            ("X-Talu-Gateway-Secret", "wrong-secret"),
            ("X-Talu-Tenant-Id", "tenant-a"),
        ],
        Some(r#"{"query":"SELECT 1"}"#),
    );
    assert_eq!(known.status, 401, "body: {}", known.body);
    assert_eq!(known.json()["error"]["code"], "unauthorized");

    let unknown = send_request(
        ctx.addr(),
        "GET",
        "/v1/db/not-real",
        &[
            ("X-Talu-Gateway-Secret", "wrong-secret"),
            ("X-Talu-Tenant-Id", "tenant-a"),
        ],
        None,
    );
    assert_eq!(unknown.status, 401, "body: {}", unknown.body);
    assert_eq!(unknown.json()["error"]["code"], "unauthorized");
}

#[test]
fn db_unknown_path_with_valid_auth_returns_404_plain_not_found() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(gateway_db_config(temp.path()));

    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/db/this-path-does-not-exist",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "tenant-a"),
        ],
        None,
    );
    assert_eq!(resp.status, 404, "body: {}", resp.body);
    assert_eq!(resp.body, "not found");
}

#[test]
fn db_unknown_path_still_requires_tenant_when_gateway_enabled() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(gateway_db_config(temp.path()));

    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/db/this-path-does-not-exist",
        &[("X-Talu-Gateway-Secret", "secret")],
        None,
    );
    assert_eq!(resp.status, 403, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "forbidden");
}

#[test]
fn db_auth_is_checked_before_body_and_path_validation() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(gateway_db_config(temp.path()));

    let invalid_secret_bad_json = send_request(
        ctx.addr(),
        "POST",
        "/v1/db/tables/documents",
        &[
            ("Content-Type", "application/json"),
            ("X-Talu-Gateway-Secret", "wrong-secret"),
            ("X-Talu-Tenant-Id", "tenant-a"),
        ],
        Some("{invalid-json"),
    );
    assert_eq!(
        invalid_secret_bad_json.status, 401,
        "auth should fail before request body parsing; body: {}",
        invalid_secret_bad_json.body
    );
    assert_eq!(
        invalid_secret_bad_json.json()["error"]["code"],
        "unauthorized"
    );

    let missing_tenant_invalid_path = send_request(
        ctx.addr(),
        "POST",
        "/v1/db/kv/namespaces//flush",
        &[("X-Talu-Gateway-Secret", "secret")],
        None,
    );
    assert_eq!(
        missing_tenant_invalid_path.status, 403,
        "auth should fail before route/path validation; body: {}",
        missing_tenant_invalid_path.body
    );
    assert_eq!(
        missing_tenant_invalid_path.json()["error"]["code"],
        "forbidden"
    );
}

#[test]
fn db_auth_precedes_not_implemented_on_known_method_mismatch_paths() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(gateway_db_config(temp.path()));

    let missing_secret = send_request(ctx.addr(), "GET", "/v1/db/sql/query", &[], None);
    assert_eq!(
        missing_secret.status, 401,
        "auth should run before not_implemented dispatch; body: {}",
        missing_secret.body
    );
    assert_eq!(missing_secret.json()["error"]["code"], "unauthorized");

    let missing_tenant = send_request(
        ctx.addr(),
        "GET",
        "/v1/db/sql/query",
        &[("X-Talu-Gateway-Secret", "secret")],
        None,
    );
    assert_eq!(
        missing_tenant.status, 403,
        "tenant auth should run before not_implemented dispatch; body: {}",
        missing_tenant.body
    );
    assert_eq!(missing_tenant.json()["error"]["code"], "forbidden");

    let authed = send_request(
        ctx.addr(),
        "GET",
        "/v1/db/sql/query",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "tenant-a"),
        ],
        None,
    );
    assert_eq!(
        authed.status, 501,
        "with valid auth, method mismatch should surface as not_implemented; body: {}",
        authed.body
    );
    assert_eq!(authed.json()["error"]["code"], "not_implemented");
}

#[test]
fn db_gateway_auth_matrix_covers_all_db_planes() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(gateway_db_config(temp.path()));

    let cases: &[(&str, &str, Option<&str>)] = &[
        ("GET", "/v1/db/tables/documents", None),
        (
            "POST",
            "/v1/db/tables/documents/search",
            Some(r#"{"query":"x"}"#),
        ),
        (
            "POST",
            "/v1/db/tables/rowsns/rows",
            Some(r#"{"schema_id":10,"columns":[]}"#),
        ),
        ("GET", "/v1/db/tables/_meta/namespaces", None),
        (
            "POST",
            "/v1/db/vectors/collections",
            Some(r#"{"name":"m","dims":3}"#),
        ),
        (
            "POST",
            "/v1/db/vectors/collections/m/points/query",
            Some(r#"{"vector":[1,0,0],"top_k":1}"#),
        ),
        ("GET", "/v1/db/kv/namespaces/ns/entries", None),
        ("PUT", "/v1/db/kv/namespaces/ns/entries/key", Some("value")),
        ("GET", "/v1/db/blobs", None),
        ("POST", "/v1/db/sql/query", Some(r#"{"query":"SELECT 1"}"#)),
        (
            "POST",
            "/v1/db/ops/compact",
            Some(r#"{"collection":"c","dims":3}"#),
        ),
    ];

    let mut failures = Vec::new();
    for (method, path, body) in cases {
        let no_auth = request_with_method_and_body(ctx.addr(), method, path, &[], *body);
        if no_auth.status != 401 || no_auth.json()["error"]["code"] != "unauthorized" {
            failures.push(format!(
                "{method} {path}: expected 401 unauthorized without auth, got status={} body={}",
                no_auth.status, no_auth.body
            ));
        }

        let missing_tenant = request_with_method_and_body(
            ctx.addr(),
            method,
            path,
            &[("X-Talu-Gateway-Secret", "secret")],
            *body,
        );
        if missing_tenant.status != 403 || missing_tenant.json()["error"]["code"] != "forbidden" {
            failures.push(format!(
                "{method} {path}: expected 403 forbidden without tenant, got status={} body={}",
                missing_tenant.status, missing_tenant.body
            ));
        }

        let unknown_tenant = request_with_method_and_body(
            ctx.addr(),
            method,
            path,
            &[
                ("X-Talu-Gateway-Secret", "secret"),
                ("X-Talu-Tenant-Id", "tenant-missing"),
            ],
            *body,
        );
        if unknown_tenant.status != 403 || unknown_tenant.json()["error"]["code"] != "forbidden" {
            failures.push(format!(
                "{method} {path}: expected 403 forbidden for unknown tenant, got status={} body={}",
                unknown_tenant.status, unknown_tenant.body
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "gateway auth matrix failures ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn db_gateway_wrong_secret_is_rejected_across_all_db_planes() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(gateway_db_config(temp.path()));

    let cases: &[(&str, &str, Option<&str>)] = &[
        ("GET", "/v1/db/tables/documents", None),
        (
            "POST",
            "/v1/db/tables/documents/search",
            Some(r#"{"query":"x"}"#),
        ),
        (
            "POST",
            "/v1/db/tables/rowsns/rows",
            Some(r#"{"schema_id":10,"columns":[]}"#),
        ),
        ("GET", "/v1/db/tables/_meta/namespaces", None),
        (
            "POST",
            "/v1/db/vectors/collections",
            Some(r#"{"name":"m","dims":3}"#),
        ),
        (
            "POST",
            "/v1/db/vectors/collections/m/points/query",
            Some(r#"{"vector":[1,0,0],"top_k":1}"#),
        ),
        ("GET", "/v1/db/kv/namespaces/ns/entries", None),
        ("PUT", "/v1/db/kv/namespaces/ns/entries/key", Some("value")),
        ("GET", "/v1/db/blobs", None),
        ("POST", "/v1/db/sql/query", Some(r#"{"query":"SELECT 1"}"#)),
        (
            "POST",
            "/v1/db/ops/compact",
            Some(r#"{"collection":"c","dims":3}"#),
        ),
    ];

    let mut failures = Vec::new();
    for (method, path, body) in cases {
        let resp = request_with_method_and_body(
            ctx.addr(),
            method,
            path,
            &[
                ("X-Talu-Gateway-Secret", "wrong-secret"),
                ("X-Talu-Tenant-Id", "tenant-a"),
            ],
            *body,
        );
        if resp.status != 401 || resp.json()["error"]["code"] != "unauthorized" {
            failures.push(format!(
                "{method} {path}: expected 401 unauthorized for wrong secret, got status={} body={}",
                resp.status, resp.body
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "wrong-secret auth matrix failures ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
