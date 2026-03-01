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

fn put_kv_with_headers(
    addr: std::net::SocketAddr,
    namespace: &str,
    key: &str,
    value: &str,
    headers: &[(&str, &str)],
) -> HttpResponse {
    let path = format!("/v1/db/kv/namespaces/{namespace}/entries/{key}");
    let mut request_headers = Vec::with_capacity(headers.len() + 1);
    request_headers.push(("Content-Type", "application/octet-stream"));
    request_headers.extend_from_slice(headers);
    send_request(addr, "PUT", &path, &request_headers, Some(value))
}

fn get_with_headers(
    addr: std::net::SocketAddr,
    path: &str,
    headers: &[(&str, &str)],
) -> HttpResponse {
    send_request(addr, "GET", path, headers, None)
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
fn kv_put_accepts_empty_values() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let put = put_kv(ctx.addr(), "empty", "k", "");
    assert_eq!(put.status, 200, "body: {}", put.body);
    assert_eq!(put.json()["value_len"], 0);

    let get_resp = get(ctx.addr(), "/v1/db/kv/namespaces/empty/entries/k");
    assert_eq!(get_resp.status, 200, "body: {}", get_resp.body);
    assert_eq!(get_resp.json()["value_len"], 0);
    assert_eq!(get_resp.json()["value_hex"], "");
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
fn kv_delete_missing_entry_is_not_found_false() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let resp = delete(ctx.addr(), "/v1/db/kv/namespaces/ns/entries/missing");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["namespace"], "ns");
    assert_eq!(json["key"], "missing");
    assert_eq!(json["deleted"], false);
}

#[test]
fn kv_rejects_invalid_namespace_and_entry_paths() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let missing_namespace = get(ctx.addr(), "/v1/db/kv/namespaces//entries");
    assert_eq!(
        missing_namespace.status, 400,
        "body: {}",
        missing_namespace.body
    );
    assert_eq!(missing_namespace.json()["error"]["code"], "invalid_path");

    let missing_key = get(ctx.addr(), "/v1/db/kv/namespaces/ns/entries/");
    assert_eq!(missing_key.status, 400, "body: {}", missing_key.body);
    assert_eq!(missing_key.json()["error"]["code"], "invalid_path");

    let whitespace_namespace = get(ctx.addr(), "/v1/db/kv/namespaces/%20/entries");
    assert_eq!(
        whitespace_namespace.status, 400,
        "body: {}",
        whitespace_namespace.body
    );
    assert_eq!(
        whitespace_namespace.json()["error"]["code"],
        "invalid_argument"
    );
}

#[test]
fn kv_paths_are_percent_decoded_for_namespace_and_key() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let put = send_request(
        ctx.addr(),
        "PUT",
        "/v1/db/kv/namespaces/ns%20x/entries/key%2Fpart",
        &[("Content-Type", "application/octet-stream")],
        Some("decoded"),
    );
    assert_eq!(put.status, 200, "body: {}", put.body);
    assert_eq!(put.json()["namespace"], "ns x");
    assert_eq!(put.json()["key"], "key/part");

    let get_resp = get(ctx.addr(), "/v1/db/kv/namespaces/ns%20x/entries/key%2Fpart");
    assert_eq!(get_resp.status, 200, "body: {}", get_resp.body);
    assert_eq!(get_resp.json()["key"], "key/part");
    assert_eq!(get_resp.json()["value_hex"], "6465636f646564");
}

#[test]
fn kv_path_segments_preserve_literal_plus_signs() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let put = send_request(
        ctx.addr(),
        "PUT",
        "/v1/db/kv/namespaces/ns+plus/entries/key+plus",
        &[("Content-Type", "application/octet-stream")],
        Some("plus-value"),
    );
    assert_eq!(put.status, 200, "body: {}", put.body);
    assert_eq!(put.json()["namespace"], "ns+plus");
    assert_eq!(put.json()["key"], "key+plus");

    let get_resp = get(ctx.addr(), "/v1/db/kv/namespaces/ns+plus/entries/key+plus");
    assert_eq!(get_resp.status, 200, "body: {}", get_resp.body);
    assert_eq!(get_resp.json()["key"], "key+plus");
    assert_eq!(get_resp.json()["value_hex"], "706c75732d76616c7565");
}

#[test]
fn kv_flush_and_compact_accept_percent_encoded_namespace() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let put = send_request(
        ctx.addr(),
        "PUT",
        "/v1/db/kv/namespaces/ops%20space/entries/key",
        &[("Content-Type", "application/octet-stream")],
        Some("value"),
    );
    assert_eq!(put.status, 200, "body: {}", put.body);
    assert_eq!(put.json()["namespace"], "ops space");

    let flush = post_json(
        ctx.addr(),
        "/v1/db/kv/namespaces/ops%20space/flush",
        &serde_json::json!({}),
    );
    assert_eq!(flush.status, 200, "body: {}", flush.body);
    assert_eq!(flush.json()["namespace"], "ops space");
    assert_eq!(flush.json()["status"], "flushed");

    let compact = post_json(
        ctx.addr(),
        "/v1/db/kv/namespaces/ops%20space/compact",
        &serde_json::json!({}),
    );
    assert_eq!(compact.status, 200, "body: {}", compact.body);
    assert_eq!(compact.json()["namespace"], "ops space");
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

    let put = send_request(
        ctx.addr(),
        "PUT",
        "/v1/db/kv/namespaces/ns/entries/key",
        &[("Content-Type", "application/octet-stream")],
        Some("value"),
    );
    assert_eq!(put.status, 503, "body: {}", put.body);
    assert_eq!(put.json()["error"]["code"], "no_storage");

    let get_resp = get(ctx.addr(), "/v1/db/kv/namespaces/ns/entries/key");
    assert_eq!(get_resp.status, 503, "body: {}", get_resp.body);
    assert_eq!(get_resp.json()["error"]["code"], "no_storage");

    let delete_resp = delete(ctx.addr(), "/v1/db/kv/namespaces/ns/entries/key");
    assert_eq!(delete_resp.status, 503, "body: {}", delete_resp.body);
    assert_eq!(delete_resp.json()["error"]["code"], "no_storage");

    let flush = post_json(
        ctx.addr(),
        "/v1/db/kv/namespaces/ns/flush",
        &serde_json::json!({}),
    );
    assert_eq!(flush.status, 503, "body: {}", flush.body);
    assert_eq!(flush.json()["error"]["code"], "no_storage");

    let compact = post_json(
        ctx.addr(),
        "/v1/db/kv/namespaces/ns/compact",
        &serde_json::json!({}),
    );
    assert_eq!(compact.status, 503, "body: {}", compact.body);
    assert_eq!(compact.json()["error"]["code"], "no_storage");
}

#[test]
fn kv_flush_and_compact_reject_missing_namespace_path_segment() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let flush = post_json(
        ctx.addr(),
        "/v1/db/kv/namespaces//flush",
        &serde_json::json!({}),
    );
    assert_eq!(flush.status, 400, "body: {}", flush.body);
    assert_eq!(flush.json()["error"]["code"], "invalid_path");

    let compact = post_json(
        ctx.addr(),
        "/v1/db/kv/namespaces//compact",
        &serde_json::json!({}),
    );
    assert_eq!(compact.status, 400, "body: {}", compact.body);
    assert_eq!(compact.json()["error"]["code"], "invalid_path");
}

#[test]
fn kv_requires_gateway_auth_when_gateway_is_enabled() {
    let temp = TempDir::new().expect("temp dir");
    let mut cfg = db_config(temp.path());
    cfg.gateway_secret = Some("secret".to_string());
    cfg.tenants = vec![TenantSpec {
        id: "tenant-a".to_string(),
        storage_prefix: "tenant-a".to_string(),
        allowed_models: vec![],
    }];
    let ctx = ServerTestContext::new(cfg);

    let list = get(ctx.addr(), "/v1/db/kv/namespaces/ns/entries");
    assert_eq!(list.status, 401, "body: {}", list.body);
    assert_eq!(list.json()["error"]["code"], "unauthorized");
}

#[test]
fn kv_storage_is_tenant_isolated_by_storage_prefix() {
    let temp = TempDir::new().expect("temp dir");
    let mut cfg = db_config(temp.path());
    cfg.gateway_secret = Some("secret".to_string());
    cfg.tenants = vec![
        TenantSpec {
            id: "tenant-a".to_string(),
            storage_prefix: "tenant-a".to_string(),
            allowed_models: vec![],
        },
        TenantSpec {
            id: "tenant-b".to_string(),
            storage_prefix: "tenant-b".to_string(),
            allowed_models: vec![],
        },
    ];
    let ctx = ServerTestContext::new(cfg);

    let tenant_a = [
        ("X-Talu-Gateway-Secret", "secret"),
        ("X-Talu-Tenant-Id", "tenant-a"),
    ];
    let tenant_b = [
        ("X-Talu-Gateway-Secret", "secret"),
        ("X-Talu-Tenant-Id", "tenant-b"),
    ];

    let put_a = put_kv_with_headers(ctx.addr(), "shared", "alpha", "from-a", &tenant_a);
    assert_eq!(put_a.status, 200, "body: {}", put_a.body);

    let get_a = get_with_headers(
        ctx.addr(),
        "/v1/db/kv/namespaces/shared/entries/alpha",
        &tenant_a,
    );
    assert_eq!(get_a.status, 200, "body: {}", get_a.body);
    assert_eq!(get_a.json()["value_hex"], "66726f6d2d61");

    let get_b = get_with_headers(
        ctx.addr(),
        "/v1/db/kv/namespaces/shared/entries/alpha",
        &tenant_b,
    );
    assert_eq!(get_b.status, 404, "body: {}", get_b.body);
    assert_eq!(get_b.json()["error"]["code"], "not_found");

    let list_b = get_with_headers(ctx.addr(), "/v1/db/kv/namespaces/shared/entries", &tenant_b);
    assert_eq!(list_b.status, 200, "body: {}", list_b.body);
    assert_eq!(list_b.json()["count"], 0);

    let put_b = put_kv_with_headers(ctx.addr(), "shared", "alpha", "from-b", &tenant_b);
    assert_eq!(put_b.status, 200, "body: {}", put_b.body);

    let get_a_again = get_with_headers(
        ctx.addr(),
        "/v1/db/kv/namespaces/shared/entries/alpha",
        &tenant_a,
    );
    assert_eq!(get_a_again.status, 200, "body: {}", get_a_again.body);
    assert_eq!(get_a_again.json()["value_hex"], "66726f6d2d61");

    let get_b_again = get_with_headers(
        ctx.addr(),
        "/v1/db/kv/namespaces/shared/entries/alpha",
        &tenant_b,
    );
    assert_eq!(get_b_again.status, 200, "body: {}", get_b_again.body);
    assert_eq!(get_b_again.json()["value_hex"], "66726f6d2d62");
}
