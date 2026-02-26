//! Integration tests for low-level `/v1/db/blobs*` endpoints.

use super::db_config;
use crate::server::common::*;
use serde_json::json;
use tempfile::TempDir;

fn upload_text_file(
    addr: std::net::SocketAddr,
    filename: &str,
    payload: &str,
    extra_headers: &[(&str, &str)],
) -> String {
    let boundary = "----talu-db-blobs-upload";
    let body = format!(
        "--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: text/plain\r\n\r\n{payload}\r\n--{b}--\r\n",
        b = boundary,
        filename = filename,
        payload = payload,
    );
    let content_type = format!("multipart/form-data; boundary={boundary}");
    let mut headers = Vec::with_capacity(extra_headers.len() + 1);
    headers.push(("Content-Type", content_type.as_str()));
    headers.extend_from_slice(extra_headers);

    let upload = send_request(addr, "POST", "/v1/files", &headers, Some(&body));
    assert_eq!(upload.status, 200, "upload failed: {}", upload.body);
    upload.json()["id"]
        .as_str()
        .expect("uploaded file id")
        .to_string()
}

fn get_with_headers(
    addr: std::net::SocketAddr,
    path: &str,
    headers: &[(&str, &str)],
) -> HttpResponse {
    send_request(addr, "GET", path, headers, None)
}

fn blob_hash_for_file(
    addr: std::net::SocketAddr,
    file_id: &str,
    headers: &[(&str, &str)],
) -> String {
    let doc = get_with_headers(addr, &format!("/v1/db/tables/documents/{file_id}"), headers);
    assert_eq!(doc.status, 200, "document fetch failed: {}", doc.body);
    let doc_json = doc.json();
    let blob_ref = doc_json["content"]["blob_ref"]
        .as_str()
        .expect("blob_ref in file document");
    blob_ref
        .strip_prefix("sha256:")
        .expect("sha256 blob ref")
        .to_string()
}

#[test]
fn blob_list_requires_storage() {
    let mut cfg = ServerConfig::new();
    cfg.no_bucket = true;
    let ctx = ServerTestContext::new(cfg);

    let resp = get(ctx.addr(), "/v1/db/blobs");
    assert_eq!(resp.status, 503, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "no_storage");
}

#[test]
fn blob_list_is_empty_initially() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/db/blobs");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["count"], 0);
    assert_eq!(json["data"], json!([]));
}

#[test]
fn blob_list_returns_hashes_and_honors_limit() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let file_a = upload_text_file(ctx.addr(), "a.txt", "alpha-bytes", &[]);
    let file_b = upload_text_file(ctx.addr(), "b.txt", "beta-bytes", &[]);
    let hash_a = blob_hash_for_file(ctx.addr(), &file_a, &[]);
    let hash_b = blob_hash_for_file(ctx.addr(), &file_b, &[]);

    let full = get(ctx.addr(), "/v1/db/blobs");
    assert_eq!(full.status, 200, "body: {}", full.body);
    let full_json = full.json();
    let data = full_json["data"].as_array().expect("data array");
    let expected_a = format!("sha256:{hash_a}");
    let expected_b = format!("sha256:{hash_b}");
    assert!(
        data.iter()
            .filter_map(|v| v.as_str())
            .any(|r| r == expected_a),
        "missing blob ref for alpha"
    );
    assert!(
        data.iter()
            .filter_map(|v| v.as_str())
            .any(|r| r == expected_b),
        "missing blob ref for beta"
    );
    assert_eq!(
        full_json["count"].as_u64().expect("count u64") as usize,
        data.len()
    );

    let limited = get(ctx.addr(), "/v1/db/blobs?limit=1");
    assert_eq!(limited.status, 200, "body: {}", limited.body);
    let limited_json = limited.json();
    assert_eq!(limited_json["count"], 1);
    assert_eq!(limited_json["data"].as_array().expect("data array").len(), 1);
}

#[test]
fn blob_list_invalid_limit_falls_back_to_default() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    upload_text_file(ctx.addr(), "a.txt", "alpha", &[]);
    upload_text_file(ctx.addr(), "b.txt", "beta", &[]);

    let resp = get(ctx.addr(), "/v1/db/blobs?limit=not-a-number");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["count"], 2, "invalid limit should use default listing");
    assert_eq!(json["data"].as_array().expect("data array").len(), 2);
}

#[test]
fn blob_list_negative_limit_falls_back_to_default() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    upload_text_file(ctx.addr(), "a.txt", "alpha", &[]);
    upload_text_file(ctx.addr(), "b.txt", "beta", &[]);

    let resp = get(ctx.addr(), "/v1/db/blobs?limit=-1");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["count"], 2, "negative limit should use default listing");
    assert_eq!(json["data"].as_array().expect("data array").len(), 2);
}

#[test]
fn blob_list_limit_zero_returns_empty() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    upload_text_file(ctx.addr(), "a.txt", "alpha", &[]);
    upload_text_file(ctx.addr(), "b.txt", "beta", &[]);

    let resp = get(ctx.addr(), "/v1/db/blobs?limit=0");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert_eq!(json["count"], 0, "limit=0 should return no refs");
    assert_eq!(json["data"], json!([]));
}

#[test]
fn blob_get_serves_raw_bytes_by_hash() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let payload = "blob-get-payload";
    let file_id = upload_text_file(ctx.addr(), "blob.txt", payload, &[]);
    let hash = blob_hash_for_file(ctx.addr(), &file_id, &[]);

    let resp = get(ctx.addr(), &format!("/v1/db/blobs/{hash}"));
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.body, payload);
    let content_type = resp.header("content-type").unwrap_or("");
    assert!(
        content_type.contains("application/octet-stream"),
        "expected octet-stream, got: {content_type}"
    );
}

#[test]
fn blob_get_accepts_uppercase_hex_hash() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let payload = "uppercase-hash-payload";
    let file_id = upload_text_file(ctx.addr(), "blob.txt", payload, &[]);
    let hash = blob_hash_for_file(ctx.addr(), &file_id, &[]);
    let uppercase = hash.to_ascii_uppercase();

    let resp = get(ctx.addr(), &format!("/v1/db/blobs/{uppercase}"));
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.body, payload);
}

#[test]
fn blob_get_rejects_invalid_hash_shape() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/db/blobs/not-a-valid-hash");
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_argument");
}

#[test]
fn blob_get_rejects_prefixed_sha256_reference() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let resp = get(ctx.addr(), &format!("/v1/db/blobs/sha256:{}", "a".repeat(64)));
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_argument");
}

#[test]
fn blob_get_returns_not_found_for_unknown_hash() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let hash = "a".repeat(64);
    let resp = get(ctx.addr(), &format!("/v1/db/blobs/{hash}"));
    assert_eq!(resp.status, 404, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_found");
}

#[test]
fn blob_get_requires_storage_configuration() {
    let mut cfg = ServerConfig::new();
    cfg.no_bucket = true;
    let ctx = ServerTestContext::new(cfg);

    let hash = "a".repeat(64);
    let resp = get(ctx.addr(), &format!("/v1/db/blobs/{hash}"));
    assert_eq!(resp.status, 503, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "no_storage");
}

#[test]
fn blob_endpoints_require_gateway_auth_when_gateway_enabled() {
    let temp = TempDir::new().expect("temp dir");
    let mut cfg = db_config(temp.path());
    cfg.gateway_secret = Some("secret".to_string());
    cfg.tenants = vec![TenantSpec {
        id: "tenant-a".to_string(),
        storage_prefix: "tenant-a".to_string(),
        allowed_models: vec![],
    }];
    let ctx = ServerTestContext::new(cfg);

    let list = get(ctx.addr(), "/v1/db/blobs");
    assert_eq!(list.status, 401, "body: {}", list.body);
    assert_eq!(list.json()["error"]["code"], "unauthorized");

    let get_resp = get(ctx.addr(), &format!("/v1/db/blobs/{}", "a".repeat(64)));
    assert_eq!(get_resp.status, 401, "body: {}", get_resp.body);
    assert_eq!(get_resp.json()["error"]["code"], "unauthorized");
}

#[test]
fn blob_list_is_tenant_isolated() {
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

    let file_a = upload_text_file(ctx.addr(), "a.txt", "tenant-a-bytes", &tenant_a);
    let file_b = upload_text_file(ctx.addr(), "b.txt", "tenant-b-bytes", &tenant_b);
    let hash_a = blob_hash_for_file(ctx.addr(), &file_a, &tenant_a);
    let hash_b = blob_hash_for_file(ctx.addr(), &file_b, &tenant_b);

    let list_a = get_with_headers(ctx.addr(), "/v1/db/blobs", &tenant_a);
    assert_eq!(list_a.status, 200, "body: {}", list_a.body);
    let list_a_json = list_a.json();
    let refs_a = list_a_json["data"].as_array().expect("tenant-a data");
    assert_eq!(refs_a.len(), 1, "tenant-a should see exactly one blob");
    let expected_a = format!("sha256:{hash_a}");
    assert_eq!(
        refs_a[0].as_str(),
        Some(expected_a.as_str())
    );

    let list_b = get_with_headers(ctx.addr(), "/v1/db/blobs", &tenant_b);
    assert_eq!(list_b.status, 200, "body: {}", list_b.body);
    let list_b_json = list_b.json();
    let refs_b = list_b_json["data"].as_array().expect("tenant-b data");
    assert_eq!(refs_b.len(), 1, "tenant-b should see exactly one blob");
    let expected_b = format!("sha256:{hash_b}");
    assert_eq!(
        refs_b[0].as_str(),
        Some(expected_b.as_str())
    );
}

#[test]
fn blob_get_is_tenant_scoped_for_same_hash() {
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

    let payload_a = "tenant-a-blob";
    let file_a = upload_text_file(ctx.addr(), "a.txt", payload_a, &tenant_a);
    let hash_a = blob_hash_for_file(ctx.addr(), &file_a, &tenant_a);

    let list_a = get_with_headers(ctx.addr(), "/v1/db/blobs", &tenant_a);
    assert_eq!(list_a.status, 200, "body: {}", list_a.body);
    assert!(
        list_a
            .json()["data"]
            .as_array()
            .expect("data")
            .iter()
            .filter_map(|v| v.as_str())
            .any(|r| r == format!("sha256:{hash_a}")),
        "tenant-a list should contain uploaded hash"
    );

    let get_a = get_with_headers(ctx.addr(), &format!("/v1/db/blobs/{hash_a}"), &tenant_a);
    assert_eq!(get_a.status, 200, "body: {}", get_a.body);
    assert_eq!(get_a.body, payload_a);

    let get_b = get_with_headers(ctx.addr(), &format!("/v1/db/blobs/{hash_a}"), &tenant_b);
    assert_eq!(get_b.status, 404, "body: {}", get_b.body);
    assert_eq!(get_b.json()["error"]["code"], "not_found");
}
