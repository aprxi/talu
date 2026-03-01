//! Integration tests for low-level `/v1/db/ops/*` endpoints.

use super::db_config;
use crate::server::common::*;
use serde_json::json;
use tempfile::TempDir;

fn create_collection(addr: std::net::SocketAddr, name: &str, dims: u32) {
    let resp = post_json(
        addr,
        "/v1/db/vectors/collections",
        &json!({
            "name": name,
            "dims": dims
        }),
    );
    assert_eq!(resp.status, 201, "create collection: {}", resp.body);
}

#[test]
fn ops_unknown_endpoint_returns_not_implemented() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let resp = post_json(ctx.addr(), "/v1/db/ops/unknown", &json!({}));
    assert_eq!(resp.status, 501, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_implemented");
}

#[test]
fn ops_wrong_method_on_known_paths_returns_not_implemented() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let compact_get = send_request(ctx.addr(), "GET", "/v1/db/ops/compact", &[], None);
    assert_eq!(compact_get.status, 501, "body: {}", compact_get.body);
    assert_eq!(compact_get.json()["error"]["code"], "not_implemented");

    let crash_put = send_request(
        ctx.addr(),
        "PUT",
        "/v1/db/ops/simulate_crash",
        &[("Content-Type", "application/json")],
        Some(r#"{"collection":"x"}"#),
    );
    assert_eq!(crash_put.status, 501, "body: {}", crash_put.body);
    assert_eq!(crash_put.json()["error"]["code"], "not_implemented");
}

#[test]
fn ops_endpoints_require_storage() {
    let mut cfg = ServerConfig::new();
    cfg.no_bucket = true;
    let ctx = ServerTestContext::new(cfg);

    let compact = post_json(
        ctx.addr(),
        "/v1/db/ops/compact",
        &json!({"collection": "missing", "dims": 3}),
    );
    assert_eq!(compact.status, 503, "body: {}", compact.body);
    assert_eq!(compact.json()["error"]["code"], "no_storage");

    let crash = post_json(
        ctx.addr(),
        "/v1/db/ops/simulate_crash",
        &json!({"collection": "missing"}),
    );
    assert_eq!(crash.status, 503, "body: {}", crash.body);
    assert_eq!(crash.json()["error"]["code"], "no_storage");
}

#[test]
fn ops_endpoints_require_gateway_auth_when_enabled() {
    let temp = TempDir::new().expect("temp dir");
    let mut cfg = db_config(temp.path());
    cfg.gateway_secret = Some("secret".to_string());
    cfg.tenants = vec![TenantSpec {
        id: "tenant-a".to_string(),
        storage_prefix: "tenant-a".to_string(),
        allowed_models: vec![],
    }];
    let ctx = ServerTestContext::new(cfg);

    let compact = post_json(
        ctx.addr(),
        "/v1/db/ops/compact",
        &json!({"collection": "ops", "dims": 3}),
    );
    assert_eq!(compact.status, 401, "body: {}", compact.body);
    assert_eq!(compact.json()["error"]["code"], "unauthorized");

    let crash_missing_tenant = send_request(
        ctx.addr(),
        "POST",
        "/v1/db/ops/simulate_crash",
        &[
            ("Content-Type", "application/json"),
            ("X-Talu-Gateway-Secret", "secret"),
        ],
        Some(r#"{"collection":"ops"}"#),
    );
    assert_eq!(
        crash_missing_tenant.status, 403,
        "body: {}",
        crash_missing_tenant.body
    );
    assert_eq!(crash_missing_tenant.json()["error"]["code"], "forbidden");
}

#[test]
fn compact_validates_request_body_fields() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let invalid_json = send_request(
        ctx.addr(),
        "POST",
        "/v1/db/ops/compact",
        &[("Content-Type", "application/json")],
        Some("not-json"),
    );
    assert_eq!(invalid_json.status, 400, "body: {}", invalid_json.body);
    assert_eq!(invalid_json.json()["error"]["code"], "invalid_json");

    let missing_collection = post_json(
        ctx.addr(),
        "/v1/db/ops/compact",
        &json!({"collection": "   ", "dims": 3}),
    );
    assert_eq!(
        missing_collection.status, 400,
        "body: {}",
        missing_collection.body
    );
    assert_eq!(
        missing_collection.json()["error"]["code"],
        "invalid_argument"
    );

    let zero_dims = post_json(
        ctx.addr(),
        "/v1/db/ops/compact",
        &json!({"collection": "ops", "dims": 0}),
    );
    assert_eq!(zero_dims.status, 400, "body: {}", zero_dims.body);
    assert_eq!(zero_dims.json()["error"]["code"], "invalid_argument");
}

#[test]
fn simulate_crash_validates_collection_field() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let invalid_json = send_request(
        ctx.addr(),
        "POST",
        "/v1/db/ops/simulate_crash",
        &[("Content-Type", "application/json")],
        Some("{broken"),
    );
    assert_eq!(invalid_json.status, 400, "body: {}", invalid_json.body);
    assert_eq!(invalid_json.json()["error"]["code"], "invalid_json");

    let missing_collection = post_json(
        ctx.addr(),
        "/v1/db/ops/simulate_crash",
        &json!({"collection": ""}),
    );
    assert_eq!(
        missing_collection.status, 400,
        "body: {}",
        missing_collection.body
    );
    assert_eq!(
        missing_collection.json()["error"]["code"],
        "invalid_argument"
    );
}

#[test]
fn ops_rejects_collection_names_with_path_separators() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let compact = post_json(
        ctx.addr(),
        "/v1/db/ops/compact",
        &json!({"collection": "bad/name", "dims": 3}),
    );
    assert_eq!(compact.status, 400, "body: {}", compact.body);
    assert_eq!(compact.json()["error"]["code"], "invalid_argument");

    let crash = post_json(
        ctx.addr(),
        "/v1/db/ops/simulate_crash",
        &json!({"collection": "bad/name"}),
    );
    assert_eq!(crash.status, 400, "body: {}", crash.body);
    assert_eq!(crash.json()["error"]["code"], "invalid_argument");
}

#[test]
fn compact_and_simulate_crash_operate_on_vector_store() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(ctx.addr(), "ops-coll", 3);
    let append = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/ops-coll/points/append",
        &json!({
            "vectors": [
                { "id": 1, "values": [1.0, 0.0, 0.0] },
                { "id": 2, "values": [0.0, 1.0, 0.0] }
            ]
        }),
    );
    assert_eq!(append.status, 200, "append: {}", append.body);

    let compact = post_json(
        ctx.addr(),
        "/v1/db/ops/compact",
        &json!({"collection": "ops-coll", "dims": 3}),
    );
    assert_eq!(compact.status, 200, "compact: {}", compact.body);
    let compact_json = compact.json();
    assert_eq!(compact_json["collection"], "ops-coll");
    assert_eq!(compact_json["dims"], 3);
    assert!(compact_json["kept_count"].as_u64().unwrap_or(0) >= 1);
    assert!(compact_json["removed_tombstones"].is_number());

    let crash = post_json(
        ctx.addr(),
        "/v1/db/ops/simulate_crash",
        &json!({"collection": "ops-coll"}),
    );
    assert_eq!(crash.status, 200, "simulate_crash: {}", crash.body);
    let crash_json = crash.json();
    assert_eq!(crash_json["collection"], "ops-coll");
    assert_eq!(crash_json["status"], "simulated_crash");
}

#[test]
fn compact_rejects_dimension_mismatch_for_existing_store() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(ctx.addr(), "ops-dims", 3);
    let append = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/ops-dims/points/append",
        &json!({
            "vectors": [{ "id": 7, "values": [0.1, 0.2, 0.3] }]
        }),
    );
    assert_eq!(append.status, 200, "append: {}", append.body);

    let compact = post_json(
        ctx.addr(),
        "/v1/db/ops/compact",
        &json!({"collection": "ops-dims", "dims": 4}),
    );
    assert_eq!(compact.status, 400, "body: {}", compact.body);
    assert_eq!(compact.json()["error"]["code"], "dimension_mismatch");
}
