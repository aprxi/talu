//! Integration tests for low-level `/v1/db/kv/*` endpoints.

use super::db_config;
use crate::server::common::*;
use tempfile::TempDir;

fn put_kv(addr: std::net::SocketAddr, namespace: &str, key: &str, value: &str) -> HttpResponse {
    let path = format!("/v1/db/kv/namespaces/{namespace}/entries/{key}");
    send_request(
        addr,
        "PUT",
        &path,
        &[("Content-Type", "application/octet-stream")],
        Some(value),
    )
}

#[test]
fn kv_put_get_delete_roundtrip() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let put = put_kv(ctx.addr(), "ns1", "alpha", "hello");
    assert_eq!(put.status, 200, "body: {}", put.body);
    let put_json = put.json();
    assert_eq!(put_json["namespace"], "ns1");
    assert_eq!(put_json["key"], "alpha");
    assert_eq!(put_json["value_len"], 5);

    let get_resp = get(ctx.addr(), "/v1/db/kv/namespaces/ns1/entries/alpha");
    assert_eq!(get_resp.status, 200, "body: {}", get_resp.body);
    let get_json = get_resp.json();
    assert_eq!(get_json["key"], "alpha");
    assert_eq!(get_json["value_len"], 5);
    assert_eq!(get_json["value_hex"], "68656c6c6f");
    assert!(get_json["updated_at_ms"].as_i64().unwrap_or(0) > 0);

    let del = delete(ctx.addr(), "/v1/db/kv/namespaces/ns1/entries/alpha");
    assert_eq!(del.status, 200, "body: {}", del.body);
    assert_eq!(del.json()["deleted"], true);

    let missing = get(ctx.addr(), "/v1/db/kv/namespaces/ns1/entries/alpha");
    assert_eq!(missing.status, 404, "body: {}", missing.body);
    assert_eq!(missing.json()["error"]["code"], "not_found");
}

#[test]
fn kv_list_is_namespace_scoped() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let put_a = put_kv(ctx.addr(), "ns_a", "k", "a");
    assert_eq!(put_a.status, 200, "body: {}", put_a.body);
    let put_b = put_kv(ctx.addr(), "ns_b", "k", "b");
    assert_eq!(put_b.status, 200, "body: {}", put_b.body);

    let list_a = get(ctx.addr(), "/v1/db/kv/namespaces/ns_a/entries");
    assert_eq!(list_a.status, 200, "body: {}", list_a.body);
    let json_a = list_a.json();
    assert_eq!(json_a["namespace"], "ns_a");
    assert_eq!(json_a["count"], 1);
    assert_eq!(json_a["data"][0]["key"], "k");
    assert_eq!(json_a["data"][0]["value_hex"], "61");

    let list_b = get(ctx.addr(), "/v1/db/kv/namespaces/ns_b/entries");
    assert_eq!(list_b.status, 200, "body: {}", list_b.body);
    let json_b = list_b.json();
    assert_eq!(json_b["namespace"], "ns_b");
    assert_eq!(json_b["count"], 1);
    assert_eq!(json_b["data"][0]["key"], "k");
    assert_eq!(json_b["data"][0]["value_hex"], "62");
}

#[test]
fn kv_flush_and_compact_endpoints() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let put = put_kv(ctx.addr(), "ops", "key", "value");
    assert_eq!(put.status, 200, "body: {}", put.body);

    let flush = post_json(
        ctx.addr(),
        "/v1/db/kv/namespaces/ops/flush",
        &serde_json::json!({}),
    );
    assert_eq!(flush.status, 200, "body: {}", flush.body);
    assert_eq!(flush.json()["status"], "flushed");

    let compact = post_json(
        ctx.addr(),
        "/v1/db/kv/namespaces/ops/compact",
        &serde_json::json!({}),
    );
    assert_eq!(compact.status, 200, "body: {}", compact.body);
    assert_eq!(compact.json()["status"], "compacted");
}

#[test]
fn kv_endpoints_require_storage() {
    let mut cfg = ServerConfig::new();
    cfg.no_bucket = true;
    let ctx = ServerTestContext::new(cfg);

    let list = get(ctx.addr(), "/v1/db/kv/namespaces/ns/entries");
    assert_eq!(list.status, 503, "body: {}", list.body);
    assert_eq!(list.json()["error"]["code"], "no_storage");
}
