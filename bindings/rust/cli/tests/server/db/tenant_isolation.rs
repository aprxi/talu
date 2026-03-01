//! Cross-plane tenant-isolation contracts for `/v1/db/*`.

use super::db_config;
use crate::server::common::*;
use tempfile::TempDir;

fn gateway_two_tenant_db_config(bucket: &std::path::Path) -> ServerConfig {
    let mut cfg = db_config(bucket);
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
    cfg
}

fn auth_headers<'a>(tenant: &'a str) -> [(&'a str, &'a str); 2] {
    [
        ("X-Talu-Gateway-Secret", "secret"),
        ("X-Talu-Tenant-Id", tenant),
    ]
}

fn post_json_with_headers(
    addr: std::net::SocketAddr,
    path: &str,
    headers: &[(&str, &str)],
    body: &serde_json::Value,
) -> HttpResponse {
    let json = serde_json::to_string(body).expect("serialize body");
    let mut all_headers = Vec::with_capacity(headers.len() + 1);
    all_headers.push(("Content-Type", "application/json"));
    all_headers.extend_from_slice(headers);
    send_request(addr, "POST", path, &all_headers, Some(&json))
}

fn get_with_headers(
    addr: std::net::SocketAddr,
    path: &str,
    headers: &[(&str, &str)],
) -> HttpResponse {
    send_request(addr, "GET", path, headers, None)
}

#[test]
fn docs_table_storage_is_tenant_isolated_even_with_same_explicit_doc_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(gateway_two_tenant_db_config(temp.path()));
    let tenant_a = auth_headers("tenant-a");
    let tenant_b = auth_headers("tenant-b");

    let create_a = post_json_with_headers(
        ctx.addr(),
        "/v1/db/tables/documents",
        &tenant_a,
        &serde_json::json!({
            "id": "shared-id",
            "type": "note",
            "title": "tenant-a-doc",
            "content": {"tenant": "a"}
        }),
    );
    assert_eq!(create_a.status, 201, "body: {}", create_a.body);

    let create_b = post_json_with_headers(
        ctx.addr(),
        "/v1/db/tables/documents",
        &tenant_b,
        &serde_json::json!({
            "id": "shared-id",
            "type": "note",
            "title": "tenant-b-doc",
            "content": {"tenant": "b"}
        }),
    );
    assert_eq!(create_b.status, 201, "body: {}", create_b.body);

    let get_a = get_with_headers(ctx.addr(), "/v1/db/tables/documents/shared-id", &tenant_a);
    assert_eq!(get_a.status, 200, "body: {}", get_a.body);
    assert_eq!(get_a.json()["title"], "tenant-a-doc");

    let get_b = get_with_headers(ctx.addr(), "/v1/db/tables/documents/shared-id", &tenant_b);
    assert_eq!(get_b.status, 200, "body: {}", get_b.body);
    assert_eq!(get_b.json()["title"], "tenant-b-doc");

    let list_a = get_with_headers(ctx.addr(), "/v1/db/tables/documents", &tenant_a);
    assert_eq!(list_a.status, 200, "body: {}", list_a.body);
    let list_a_data = list_a.json()["data"].as_array().expect("data").clone();
    assert_eq!(
        list_a_data.len(),
        1,
        "tenant-a should only see one document"
    );
    assert_eq!(list_a_data[0]["id"], "shared-id");
    assert_eq!(list_a_data[0]["title"], "tenant-a-doc");

    let list_b = get_with_headers(ctx.addr(), "/v1/db/tables/documents", &tenant_b);
    assert_eq!(list_b.status, 200, "body: {}", list_b.body);
    let list_b_data = list_b.json()["data"].as_array().expect("data").clone();
    assert_eq!(
        list_b_data.len(),
        1,
        "tenant-b should only see one document"
    );
    assert_eq!(list_b_data[0]["id"], "shared-id");
    assert_eq!(list_b_data[0]["title"], "tenant-b-doc");
}

#[test]
fn docs_search_is_tenant_scoped() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(gateway_two_tenant_db_config(temp.path()));
    let tenant_a = auth_headers("tenant-a");
    let tenant_b = auth_headers("tenant-b");

    let create_a = post_json_with_headers(
        ctx.addr(),
        "/v1/db/tables/documents",
        &tenant_a,
        &serde_json::json!({
            "type": "note",
            "title": "Alpha-only keyword: ORBITAL-FOX-001",
            "content": {}
        }),
    );
    assert_eq!(create_a.status, 201, "body: {}", create_a.body);

    let search_a = post_json_with_headers(
        ctx.addr(),
        "/v1/db/tables/documents/search",
        &tenant_a,
        &serde_json::json!({
            "query": "ORBITAL-FOX-001",
            "limit": 10
        }),
    );
    assert_eq!(search_a.status, 200, "body: {}", search_a.body);
    assert_eq!(
        search_a.json()["data"].as_array().expect("data").len(),
        1,
        "tenant-a should find its own indexed document"
    );

    let search_b = post_json_with_headers(
        ctx.addr(),
        "/v1/db/tables/documents/search",
        &tenant_b,
        &serde_json::json!({
            "query": "ORBITAL-FOX-001",
            "limit": 10
        }),
    );
    assert_eq!(search_b.status, 200, "body: {}", search_b.body);
    assert_eq!(
        search_b.json()["data"].as_array().expect("data").len(),
        0,
        "tenant-b must not observe tenant-a documents in search results"
    );
}

#[test]
fn docs_tags_are_tenant_isolated_for_same_doc_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(gateway_two_tenant_db_config(temp.path()));
    let tenant_a = auth_headers("tenant-a");
    let tenant_b = auth_headers("tenant-b");

    let create_a = post_json_with_headers(
        ctx.addr(),
        "/v1/db/tables/documents",
        &tenant_a,
        &serde_json::json!({
            "id": "shared-tag-id",
            "type": "note",
            "title": "tenant-a-tagged",
            "content": {}
        }),
    );
    assert_eq!(create_a.status, 201, "body: {}", create_a.body);

    let add_tags_a = post_json_with_headers(
        ctx.addr(),
        "/v1/db/tables/documents/shared-tag-id/tags",
        &tenant_a,
        &serde_json::json!({ "tags": ["a-only"] }),
    );
    assert_eq!(add_tags_a.status, 200, "body: {}", add_tags_a.body);

    let tags_b_before_create = get_with_headers(
        ctx.addr(),
        "/v1/db/tables/documents/shared-tag-id/tags",
        &tenant_b,
    );
    assert_eq!(
        tags_b_before_create.status, 404,
        "tenant-b should not see tenant-a doc/tag state; body: {}",
        tags_b_before_create.body
    );
    assert_eq!(tags_b_before_create.json()["error"]["code"], "not_found");

    let create_b = post_json_with_headers(
        ctx.addr(),
        "/v1/db/tables/documents",
        &tenant_b,
        &serde_json::json!({
            "id": "shared-tag-id",
            "type": "note",
            "title": "tenant-b-doc",
            "content": {}
        }),
    );
    assert_eq!(create_b.status, 201, "body: {}", create_b.body);

    let tags_b_after_create = get_with_headers(
        ctx.addr(),
        "/v1/db/tables/documents/shared-tag-id/tags",
        &tenant_b,
    );
    assert_eq!(
        tags_b_after_create.status, 200,
        "body: {}",
        tags_b_after_create.body
    );
    assert_eq!(
        tags_b_after_create.json()["tags"]
            .as_array()
            .expect("tags")
            .len(),
        0,
        "tenant-b tags must start empty even when tenant-a used same document ID"
    );
}
