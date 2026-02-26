//! Cross-plane contract tests for `/v1/db/*`.
//!
//! These tests verify router/spec parity and catch drift across planes.

use super::db_config;
use crate::server::common::*;
use std::collections::{BTreeSet, HashMap};
use tempfile::TempDir;

fn normalized_path_template(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    let mut in_braces = false;
    for ch in path.chars() {
        if ch == '{' {
            in_braces = true;
            out.push('{');
            out.push('}');
            continue;
        }
        if ch == '}' {
            in_braces = false;
            continue;
        }
        if !in_braces {
            out.push(ch);
        }
    }
    out
}

fn materialize_db_path(template: &str, suffix: &str) -> String {
    let replacements: HashMap<&str, String> = HashMap::from([
        ("table", "documents".to_string()),
        ("doc_id", format!("missing-{suffix}")),
        ("namespace", format!("ns-{suffix}")),
        ("key", format!("k-{suffix}")),
        ("name", format!("coll-{suffix}")),
        ("blob_ref", "a".repeat(64)),
    ]);

    let mut out = template.to_string();
    for (key, value) in replacements {
        out = out.replace(&format!("{{{key}}}"), &value);
    }
    out
}

fn request_for_method(addr: std::net::SocketAddr, method: &str, path: &str) -> HttpResponse {
    match method {
        "get" => send_request(addr, "GET", path, &[], None),
        "post" => send_request(
            addr,
            "POST",
            path,
            &[("Content-Type", "application/json")],
            Some("{}"),
        ),
        "patch" => send_request(
            addr,
            "PATCH",
            path,
            &[("Content-Type", "application/json")],
            Some("{}"),
        ),
        "delete" => send_request(
            addr,
            "DELETE",
            path,
            &[("Content-Type", "application/json")],
            Some("{}"),
        ),
        "put" => send_request(
            addr,
            "PUT",
            path,
            &[("Content-Type", "application/octet-stream")],
            Some("x"),
        ),
        other => panic!("unsupported OpenAPI method key: {other}"),
    }
}

fn path_item_methods(path_item: &serde_json::Value) -> BTreeSet<String> {
    path_item
        .as_object()
        .expect("path item object")
        .keys()
        .filter(|k| {
            matches!(
                k.as_str(),
                "get" | "post" | "put" | "patch" | "delete" | "head" | "options" | "trace"
            )
        })
        .cloned()
        .collect()
}

#[test]
fn openapi_db_documented_operations_are_routable_through_http_dispatcher() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let spec = get(ctx.addr(), "/openapi/db.json");
    assert_eq!(spec.status, 200, "body: {}", spec.body);
    let spec_json = spec.json();
    let paths = spec_json["paths"].as_object().expect("paths object");
    assert!(!paths.is_empty(), "db spec should define at least one path");

    let suffix = format!("{}-{}", std::process::id(), paths.len());
    for (template, methods_obj) in paths {
        let methods = methods_obj.as_object().expect("methods object");
        let concrete_path = materialize_db_path(template, &suffix);

        for method in methods.keys() {
            let resp = request_for_method(ctx.addr(), method, &concrete_path);

            assert_ne!(
                resp.status, 501,
                "documented operation should not hit not_implemented: method={method} path={concrete_path} body={}",
                resp.body
            );

            assert!(
                !(resp.status == 404 && resp.body == "not found"),
                "documented operation should not miss router match: method={method} path={concrete_path}"
            );
        }
    }
}

#[test]
fn openapi_plane_specs_cover_low_level_table_and_blob_routes() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let tables = get(ctx.addr(), "/openapi/db/tables.json");
    assert_eq!(tables.status, 200, "body: {}", tables.body);
    let table_paths = tables.json()["paths"]
        .as_object()
        .expect("tables paths")
        .keys()
        .map(|p| normalized_path_template(p))
        .collect::<BTreeSet<_>>();

    for expected in [
        "/v1/db/tables/{}/rows",
        "/v1/db/tables/{}/rows/{}",
        "/v1/db/tables/{}/rows/scan",
        "/v1/db/tables/_meta/namespaces",
        "/v1/db/tables/{}/_meta/policy",
    ] {
        assert!(
            table_paths.contains(expected),
            "tables OpenAPI is missing low-level route template {expected}"
        );
    }

    let blobs = get(ctx.addr(), "/openapi/db/blobs.json");
    assert_eq!(blobs.status, 200, "body: {}", blobs.body);
    let blob_paths = blobs.json()["paths"]
        .as_object()
        .expect("blobs paths")
        .keys()
        .map(|p| normalized_path_template(p))
        .collect::<BTreeSet<_>>();

    assert!(
        blob_paths.contains("/v1/db/blobs"),
        "blobs OpenAPI should include list route /v1/db/blobs"
    );
    assert!(
        blob_paths.contains("/v1/db/blobs/{}"),
        "blobs OpenAPI should include blob-get route /v1/db/blobs/{{...}}"
    );
}

#[test]
fn openapi_db_aggregate_spec_includes_low_level_table_routes() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let db = get(ctx.addr(), "/openapi/db.json");
    assert_eq!(db.status, 200, "body: {}", db.body);
    let db_paths = db.json()["paths"]
        .as_object()
        .expect("db paths")
        .keys()
        .map(|p| normalized_path_template(p))
        .collect::<BTreeSet<_>>();

    for expected in [
        "/v1/db/tables/{}/rows",
        "/v1/db/tables/{}/rows/{}",
        "/v1/db/tables/{}/rows/scan",
        "/v1/db/tables/_meta/namespaces",
        "/v1/db/tables/{}/_meta/policy",
    ] {
        assert!(
            db_paths.contains(expected),
            "aggregate db OpenAPI is missing low-level route template {expected}"
        );
    }
}

#[test]
fn openapi_low_level_table_routes_have_expected_methods_in_plane_and_aggregate_specs() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let expected: [(&str, &[&str]); 5] = [
        ("/v1/db/tables/{}/rows", &["get", "post"]),
        ("/v1/db/tables/{}/rows/{}", &["get", "delete"]),
        ("/v1/db/tables/{}/rows/scan", &["post"]),
        ("/v1/db/tables/_meta/namespaces", &["get"]),
        ("/v1/db/tables/{}/_meta/policy", &["get"]),
    ];

    for (spec_name, spec_path) in [
        ("tables", "/openapi/db/tables.json"),
        ("aggregate", "/openapi/db.json"),
    ] {
        let resp = get(ctx.addr(), spec_path);
        assert_eq!(resp.status, 200, "body: {}", resp.body);
        let spec_json = resp.json();
        let paths = spec_json["paths"].as_object().expect("paths object");

        let mut by_template: HashMap<String, BTreeSet<String>> = HashMap::new();
        for (raw_template, path_item) in paths {
            let norm = normalized_path_template(raw_template);
            by_template.insert(norm, path_item_methods(path_item));
        }

        for (template, expected_methods) in expected {
            let got = by_template
                .get(template)
                .unwrap_or_else(|| panic!("{spec_name} spec missing {template}"));
            let want = expected_methods
                .iter()
                .map(|m| m.to_string())
                .collect::<BTreeSet<_>>();
            assert_eq!(
                got, &want,
                "{spec_name} spec has wrong methods for {template}"
            );
        }
    }
}
