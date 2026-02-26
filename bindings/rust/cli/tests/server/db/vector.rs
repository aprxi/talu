//! Integration tests for low-level `/v1/db/*` vector endpoints.

use super::db_config;
use crate::server::common::*;
use serde_json::json;
use tempfile::TempDir;

fn create_collection(ctx: &ServerTestContext, name: &str, dims: u32) {
    let resp = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections",
        &json!({
            "name": name,
            "dims": dims
        }),
    );
    assert_eq!(resp.status, 201, "status={}, body={}", resp.status, resp.body);
}

fn post_json_with_headers(
    addr: std::net::SocketAddr,
    path: &str,
    headers: &[(&str, &str)],
    body: &serde_json::Value,
) -> HttpResponse {
    let payload = serde_json::to_string(body).expect("serialize json");
    let mut request_headers = Vec::with_capacity(headers.len() + 1);
    request_headers.push(("Content-Type", "application/json"));
    request_headers.extend_from_slice(headers);
    send_request(addr, "POST", path, &request_headers, Some(&payload))
}

fn get_with_headers(
    addr: std::net::SocketAddr,
    path: &str,
    headers: &[(&str, &str)],
) -> HttpResponse {
    send_request(addr, "GET", path, headers, None)
}

#[test]
fn create_collection_minimal_defaults() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections",
        &json!({
            "name": "docs-1536",
            "dims": 1536
        }),
    );
    assert_eq!(resp.status, 201, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["name"], "docs-1536");
    assert_eq!(json["dims"], 1536);
    assert_eq!(json["metric"], "dot");
    assert_eq!(json["normalization"], "none");
    assert_eq!(json["id_type"], "u64");
    assert!(json["created_at"].is_number());
    assert!(json["updated_at"].is_number());
}

#[test]
fn vector_endpoints_require_storage() {
    let mut cfg = ServerConfig::new();
    cfg.no_bucket = true;
    let ctx = ServerTestContext::new(cfg);

    let create = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections",
        &json!({"name": "x", "dims": 3}),
    );
    assert_eq!(create.status, 503, "body: {}", create.body);
    assert_eq!(create.json()["error"]["code"], "no_storage");

    let list = get(ctx.addr(), "/v1/db/vectors/collections");
    assert_eq!(list.status, 503, "body: {}", list.body);
    assert_eq!(list.json()["error"]["code"], "no_storage");
}

#[test]
fn vector_endpoints_enforce_gateway_auth_and_tenant() {
    let temp = TempDir::new().expect("temp dir");
    let mut cfg = db_config(temp.path());
    cfg.gateway_secret = Some("secret".to_string());
    cfg.tenants = vec![TenantSpec {
        id: "tenant-a".to_string(),
        storage_prefix: "tenant-a".to_string(),
        allowed_models: vec![],
    }];
    let ctx = ServerTestContext::new(cfg);

    let missing_secret = get(ctx.addr(), "/v1/db/vectors/collections");
    assert_eq!(missing_secret.status, 401, "body: {}", missing_secret.body);
    assert_eq!(missing_secret.json()["error"]["code"], "unauthorized");

    let missing_tenant = send_request(
        ctx.addr(),
        "GET",
        "/v1/db/vectors/collections",
        &[("X-Talu-Gateway-Secret", "secret")],
        None,
    );
    assert_eq!(missing_tenant.status, 403, "body: {}", missing_tenant.body);
    assert_eq!(missing_tenant.json()["error"]["code"], "forbidden");

    let unknown_tenant = send_request(
        ctx.addr(),
        "GET",
        "/v1/db/vectors/collections",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "tenant-missing"),
        ],
        None,
    );
    assert_eq!(unknown_tenant.status, 403, "body: {}", unknown_tenant.body);
    assert_eq!(unknown_tenant.json()["error"]["code"], "forbidden");
}

#[test]
fn create_collection_rejects_zero_dims_and_path_separators() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let zero_dims = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections",
        &json!({
            "name": "bad-zero",
            "dims": 0
        }),
    );
    assert_eq!(zero_dims.status, 400, "body: {}", zero_dims.body);
    assert_eq!(zero_dims.json()["error"]["code"], "invalid_argument");

    let bad_name = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections",
        &json!({
            "name": "../escape",
            "dims": 3
        }),
    );
    assert_eq!(bad_name.status, 400, "body: {}", bad_name.body);
    assert_eq!(bad_name.json()["error"]["code"], "invalid_argument");

    let backslash_name = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections",
        &json!({
            "name": "bad\\name",
            "dims": 3
        }),
    );
    assert_eq!(backslash_name.status, 400, "body: {}", backslash_name.body);
    assert_eq!(backslash_name.json()["error"]["code"], "invalid_argument");
}

#[test]
fn create_collection_rejects_blank_name() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections",
        &json!({
            "name": "   ",
            "dims": 3
        }),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_argument");
}

#[test]
fn create_collection_idempotent_same_config() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = json!({
        "name": "embeddings",
        "dims": 1024,
        "metric": "dot",
        "normalization": "l2",
        "id_type": "u64"
    });

    let first = post_json(ctx.addr(), "/v1/db/vectors/collections", &body);
    assert_eq!(first.status, 201, "body: {}", first.body);

    let second = post_json(ctx.addr(), "/v1/db/vectors/collections", &body);
    assert_eq!(second.status, 200, "body: {}", second.body);

    let second_json = second.json();
    assert_eq!(second_json["name"], "embeddings");
    assert_eq!(second_json["dims"], 1024);
    assert_eq!(second_json["normalization"], "l2");
}

#[test]
fn create_collection_conflict_different_config() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let first = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections",
        &json!({
            "name": "embeddings",
            "dims": 768
        }),
    );
    assert_eq!(first.status, 201, "body: {}", first.body);

    let second = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections",
        &json!({
            "name": "embeddings",
            "dims": 1536
        }),
    );
    assert_eq!(second.status, 409, "body: {}", second.body);
    let json = second.json();
    assert_eq!(json["error"]["code"], "collection_conflict");
}

#[test]
fn list_get_and_delete_collection() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "c1", 256);

    let list = get(ctx.addr(), "/v1/db/vectors/collections");
    assert_eq!(list.status, 200, "body: {}", list.body);
    let list_json = list.json();
    let data = list_json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["name"], "c1");

    let get_one = get(ctx.addr(), "/v1/db/vectors/collections/c1");
    assert_eq!(get_one.status, 200, "body: {}", get_one.body);
    let one_json = get_one.json();
    assert_eq!(one_json["name"], "c1");
    assert_eq!(one_json["dims"], 256);

    let del = delete(ctx.addr(), "/v1/db/vectors/collections/c1");
    assert_eq!(del.status, 204, "body: {}", del.body);

    let missing = get(ctx.addr(), "/v1/db/vectors/collections/c1");
    assert_eq!(missing.status, 404, "body: {}", missing.body);
    let missing_json = missing.json();
    assert_eq!(missing_json["error"]["code"], "collection_not_found");
}

#[test]
fn append_and_query_single_vector() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);

    let append = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [
                { "id": 1, "values": [1.0, 0.0, 0.0] },
                { "id": "2", "values": [0.0, 1.0, 0.0] }
            ]
        }),
    );
    assert_eq!(append.status, 200, "body: {}", append.body);
    let append_json = append.json();
    assert_eq!(append_json["appended_count"], 2);

    let query = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/query",
        &json!({
            "vector": [1.0, 0.0, 0.0],
            "top_k": 1
        }),
    );
    assert_eq!(query.status, 200, "body: {}", query.body);
    let query_json = query.json();
    let results = query_json["results"].as_array().expect("results array");
    assert_eq!(results.len(), 1);
    let matches = results[0]["matches"].as_array().expect("matches array");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0]["id"], "1");
}

#[test]
fn append_rejects_dimension_mismatch_and_duplicate_request_ids() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);

    let mismatch = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [
                { "id": 1, "values": [1.0, 0.0] }
            ]
        }),
    );
    assert_eq!(mismatch.status, 400, "body: {}", mismatch.body);
    let json = mismatch.json();
    assert_eq!(json["error"]["code"], "dimension_mismatch");

    let duplicate = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [
                { "id": 42, "values": [1.0, 0.0, 0.0] },
                { "id": "42", "values": [0.0, 1.0, 0.0] }
            ]
        }),
    );
    assert_eq!(duplicate.status, 409, "body: {}", duplicate.body);
    let json = duplicate.json();
    assert_eq!(json["error"]["code"], "id_conflict");
}

#[test]
fn append_rejects_request_dims_field_mismatch() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);

    let resp = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "dims": 4,
            "vectors": [
                { "id": 1, "values": [1.0, 0.0, 0.0] }
            ]
        }),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "dimension_mismatch");
}

#[test]
fn append_rejects_ids_that_already_exist_in_collection() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);

    let first = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [
                { "id": 7, "values": [1.0, 0.0, 0.0] }
            ]
        }),
    );
    assert_eq!(first.status, 200, "body: {}", first.body);

    let duplicate = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [
                { "id": 7, "values": [0.0, 1.0, 0.0] }
            ]
        }),
    );
    assert_eq!(duplicate.status, 409, "body: {}", duplicate.body);
    assert_eq!(duplicate.json()["error"]["code"], "id_conflict");
}

#[test]
fn append_rejects_zero_vector_when_l2_normalization_enabled() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let create = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections",
        &json!({
            "name": "l2-emb",
            "dims": 3,
            "normalization": "l2"
        }),
    );
    assert_eq!(create.status, 201, "body: {}", create.body);

    let append = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/l2-emb/points/append",
        &json!({
            "vectors": [
                { "id": 1, "values": [0.0, 0.0, 0.0] }
            ]
        }),
    );
    assert_eq!(append.status, 400, "body: {}", append.body);
    assert_eq!(append.json()["error"]["code"], "invalid_argument");
}

#[test]
fn query_batch_returns_per_query_matches() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);

    let append = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [
                { "id": 10, "values": [1.0, 0.0, 0.0] },
                { "id": 20, "values": [0.0, 1.0, 0.0] }
            ]
        }),
    );
    assert_eq!(append.status, 200, "body: {}", append.body);

    let query = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/query",
        &json!({
            "queries": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]
            ],
            "top_k": 1
        }),
    );
    assert_eq!(query.status, 200, "body: {}", query.body);
    let json = query.json();
    let results = json["results"].as_array().expect("results array");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["matches"][0]["id"], "10");
    assert_eq!(results[1]["matches"][0]["id"], "20");
}

#[test]
fn query_accepts_approximate_flag() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);

    let append = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [
                { "id": 10, "values": [1.0, 0.0, 0.0] },
                { "id": 20, "values": [0.0, 1.0, 0.0] }
            ]
        }),
    );
    assert_eq!(append.status, 200, "body: {}", append.body);

    let query = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/query",
        &json!({
            "vector": [1.0, 0.0, 0.0],
            "top_k": 1,
            "approximate": true
        }),
    );
    assert_eq!(query.status, 200, "body: {}", query.body);
    assert_eq!(query.json()["results"][0]["matches"][0]["id"], "10");
}

#[test]
fn query_rejects_invalid_shape_and_top_k_zero() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);

    let invalid_shape = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/query",
        &json!({
            "vector": [1.0, 0.0, 0.0],
            "queries": [[1.0, 0.0, 0.0]],
            "top_k": 1
        }),
    );
    assert_eq!(invalid_shape.status, 400, "body: {}", invalid_shape.body);
    assert_eq!(
        invalid_shape.json()["error"]["code"],
        "invalid_argument",
        "body: {}",
        invalid_shape.body
    );

    let zero_top_k = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/query",
        &json!({
            "vector": [1.0, 0.0, 0.0],
            "top_k": 0
        }),
    );
    assert_eq!(zero_top_k.status, 400, "body: {}", zero_top_k.body);
    assert_eq!(zero_top_k.json()["error"]["code"], "invalid_argument");
}

#[test]
fn query_rejects_empty_queries() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);

    let resp = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/query",
        &json!({
            "queries": [],
            "top_k": 1
        }),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_argument");
}

#[test]
fn query_rejects_dimension_mismatch() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);

    let resp = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/query",
        &json!({
            "vector": [1.0, 0.0],
            "top_k": 1
        }),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "dimension_mismatch");
}

#[test]
fn query_requires_exactly_one_of_vector_or_queries() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);

    let resp = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/query",
        &json!({
            "top_k": 1
        }),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_argument");
}

#[test]
fn query_min_score_filters_low_similarity_matches() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);
    let append = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [
                { "id": 10, "values": [1.0, 0.0, 0.0] },
                { "id": 20, "values": [0.0, 1.0, 0.0] }
            ]
        }),
    );
    assert_eq!(append.status, 200, "body: {}", append.body);

    let query = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/query",
        &json!({
            "vector": [1.0, 0.0, 0.0],
            "top_k": 2,
            "min_score": 0.5
        }),
    );
    assert_eq!(query.status, 200, "body: {}", query.body);
    let query_json = query.json();
    let matches = query_json["results"][0]["matches"]
        .as_array()
        .expect("matches");
    assert_eq!(matches.len(), 1, "min_score should filter weaker match");
    assert_eq!(matches[0]["id"], "10");
}

#[test]
fn db_openapi_includes_query_compact_and_index_build_request_fields() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let resp = get(ctx.addr(), "/openapi.json");
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let schemas = &json["components"]["schemas"];
    assert!(
        schemas["QueryPointsRequest"]["properties"]
            .get("approximate")
            .is_some(),
        "missing QueryPointsRequest.approximate"
    );
    assert!(
        schemas["CompactCollectionRequest"]["properties"]
            .get("expected_generation")
            .is_some(),
        "missing CompactCollectionRequest.expected_generation"
    );
    assert!(
        schemas["CompactCollectionRequest"]["properties"]
            .get("ttl_max_age_ms")
            .is_some(),
        "missing CompactCollectionRequest.ttl_max_age_ms"
    );
    assert!(
        schemas["CompactCollectionRequest"]["properties"]
            .get("now_ms")
            .is_some(),
        "missing CompactCollectionRequest.now_ms"
    );
    assert!(
        schemas["BuildIndexesRequest"]["properties"]
            .get("expected_generation")
            .is_some(),
        "missing BuildIndexesRequest.expected_generation"
    );
    assert!(
        schemas["BuildIndexesRequest"]["properties"]
            .get("max_segments")
            .is_some(),
        "missing BuildIndexesRequest.max_segments"
    );
    let has_indexes_build_path = json["paths"]
        .as_object()
        .is_some_and(|paths| paths.keys().any(|key| key.contains("/indexes/build")));
    assert!(has_indexes_build_path, "missing indexes/build path");
    assert!(
        schemas["CollectionStatsResponse"]["properties"]
            .get("manifest_generation")
            .is_some(),
        "missing CollectionStatsResponse.manifest_generation"
    );
    assert!(
        schemas["CollectionStatsResponse"]["properties"]
            .get("index_ready_segments")
            .is_some(),
        "missing CollectionStatsResponse.index_ready_segments"
    );
    assert!(
        schemas["CollectionStatsResponse"]["properties"]
            .get("index_pending_segments")
            .is_some(),
        "missing CollectionStatsResponse.index_pending_segments"
    );
    assert!(
        schemas["CollectionStatsResponse"]["properties"]
            .get("index_failed_segments")
            .is_some(),
        "missing CollectionStatsResponse.index_failed_segments"
    );
}

#[test]
fn fetch_delete_and_upsert_work_with_tombstone_visibility() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);

    let append = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [
                { "id": 1, "values": [1.0, 0.0, 0.0] },
                { "id": 2, "values": [0.0, 1.0, 0.0] }
            ]
        }),
    );
    assert_eq!(append.status, 200, "body: {}", append.body);

    let fetch_without_values = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/fetch",
        &json!({
            "ids": [1, "2"],
            "include_values": false
        }),
    );
    assert_eq!(
        fetch_without_values.status, 200,
        "body: {}",
        fetch_without_values.body
    );
    let fetched = fetch_without_values.json();
    assert_eq!(fetched["data"].as_array().expect("data").len(), 2);
    assert!(
        fetched["data"][0].get("values").is_none(),
        "values should be omitted when include_values=false"
    );

    let delete_resp = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/delete",
        &json!({ "ids": [2, 999] }),
    );
    assert_eq!(delete_resp.status, 200, "body: {}", delete_resp.body);
    let delete_json = delete_resp.json();
    assert_eq!(delete_json["deleted_count"], 1);
    assert_eq!(delete_json["not_found_count"], 1);

    let fetch_after_delete = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/fetch",
        &json!({ "ids": [1, 2] }),
    );
    assert_eq!(
        fetch_after_delete.status, 200,
        "body: {}",
        fetch_after_delete.body
    );
    let fetch_json = fetch_after_delete.json();
    assert_eq!(fetch_json["data"].as_array().expect("data").len(), 1);
    assert_eq!(fetch_json["data"][0]["id"], "1");
    assert_eq!(fetch_json["not_found_ids"], json!(["2"]));

    let upsert = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/upsert",
        &json!({
            "vectors": [
                { "id": 2, "values": [0.0, 0.0, 1.0] }
            ]
        }),
    );
    assert_eq!(upsert.status, 200, "body: {}", upsert.body);
    assert_eq!(upsert.json()["upserted_count"], 1);

    let fetch_after_upsert = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/fetch",
        &json!({ "ids": [2] }),
    );
    assert_eq!(
        fetch_after_upsert.status, 200,
        "body: {}",
        fetch_after_upsert.body
    );
    let fetch_json = fetch_after_upsert.json();
    assert_eq!(fetch_json["data"].as_array().expect("data").len(), 1);
    assert_eq!(fetch_json["data"][0]["id"], "2");
    assert_eq!(fetch_json["data"][0]["values"], json!([0.0, 0.0, 1.0]));
}

#[test]
fn fetch_and_delete_reject_invalid_id_shapes() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);

    let fetch = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/fetch",
        &json!({
            "ids": [true]
        }),
    );
    assert_eq!(fetch.status, 400, "body: {}", fetch.body);
    assert_eq!(fetch.json()["error"]["code"], "invalid_json");

    let delete_resp = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/delete",
        &json!({
            "ids": ["ok", {"bad": "shape"}]
        }),
    );
    assert_eq!(delete_resp.status, 400, "body: {}", delete_resp.body);
    assert_eq!(delete_resp.json()["error"]["code"], "invalid_json");
}

#[test]
fn upsert_rejects_dimension_mismatch_via_request_dims_field() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);

    let resp = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/upsert",
        &json!({
            "dims": 4,
            "vectors": [
                { "id": 1, "values": [1.0, 0.0, 0.0] }
            ]
        }),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "dimension_mismatch");
}

#[test]
fn upsert_delete_fetch_and_query_return_not_found_for_missing_collection() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let upsert = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/missing/points/upsert",
        &json!({
            "vectors": [
                { "id": 1, "values": [1.0, 0.0, 0.0] }
            ]
        }),
    );
    assert_eq!(upsert.status, 404, "body: {}", upsert.body);
    assert_eq!(upsert.json()["error"]["code"], "collection_not_found");

    let delete_resp = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/missing/points/delete",
        &json!({ "ids": [1] }),
    );
    assert_eq!(delete_resp.status, 404, "body: {}", delete_resp.body);
    assert_eq!(delete_resp.json()["error"]["code"], "collection_not_found");

    let fetch = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/missing/points/fetch",
        &json!({ "ids": [1] }),
    );
    assert_eq!(fetch.status, 404, "body: {}", fetch.body);
    assert_eq!(fetch.json()["error"]["code"], "collection_not_found");

    let query = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/missing/points/query",
        &json!({
            "vector": [1.0, 0.0, 0.0],
            "top_k": 1
        }),
    );
    assert_eq!(query.status, 404, "body: {}", query.body);
    assert_eq!(query.json()["error"]["code"], "collection_not_found");
}

#[test]
fn fetch_includes_values_by_default() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);
    let append = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [{ "id": 9, "values": [0.2, 0.4, 0.8] }]
        }),
    );
    assert_eq!(append.status, 200, "body: {}", append.body);

    let fetch = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/fetch",
        &json!({ "ids": [9] }),
    );
    assert_eq!(fetch.status, 200, "body: {}", fetch.body);
    let fetch_json = fetch.json();
    let values = fetch_json["data"][0]["values"]
        .as_array()
        .expect("values array");
    assert_eq!(values.len(), 3);
}

#[test]
fn stats_changes_and_compact_reflect_lifecycle() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);

    let append = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [
                { "id": 1, "values": [1.0, 0.0, 0.0] },
                { "id": 2, "values": [0.0, 1.0, 0.0] }
            ]
        }),
    );
    assert_eq!(append.status, 200, "body: {}", append.body);

    let delete_resp = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/delete",
        &json!({ "ids": [2] }),
    );
    assert_eq!(delete_resp.status, 200, "body: {}", delete_resp.body);

    let stats_before = get(ctx.addr(), "/v1/db/vectors/collections/emb/stats");
    assert_eq!(stats_before.status, 200, "body: {}", stats_before.body);
    let stats_json = stats_before.json();
    assert_eq!(stats_json["visible_count"], 1);
    assert_eq!(stats_json["tombstone_count"], 1);
    assert_eq!(stats_json["total_vector_count"], 2);
    assert!(stats_json["manifest_generation"].is_number());
    assert!(stats_json["index_ready_segments"].is_number());
    assert!(stats_json["index_pending_segments"].is_number());
    assert!(stats_json["index_failed_segments"].is_number());

    let changes = get(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/changes?since=0&limit=10",
    );
    assert_eq!(changes.status, 200, "body: {}", changes.body);
    let changes_json = changes.json();
    let data = changes_json["data"].as_array().expect("changes data");
    assert_eq!(data.len(), 3, "append, append, delete");

    let compact = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/compact",
        &json!({}),
    );
    assert_eq!(compact.status, 200, "body: {}", compact.body);
    let compact_json = compact.json();
    assert_eq!(compact_json["kept_count"], 1);
    assert_eq!(compact_json["removed_tombstones"], 1);

    let stats_after = get(ctx.addr(), "/v1/db/vectors/collections/emb/stats");
    assert_eq!(stats_after.status, 200, "body: {}", stats_after.body);
    let stats_json = stats_after.json();
    assert_eq!(stats_json["visible_count"], 1);
    assert_eq!(stats_json["tombstone_count"], 0);
    assert!(stats_json["manifest_generation"].is_number());
    assert!(stats_json["index_ready_segments"].is_number());
    assert!(stats_json["index_pending_segments"].is_number());
    assert!(stats_json["index_failed_segments"].is_number());

    let changes_after = get(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/changes?since=0&limit=10",
    );
    assert_eq!(changes_after.status, 200, "body: {}", changes_after.body);
    let data = changes_after.json()["data"]
        .as_array()
        .expect("changes data")
        .clone();
    assert!(
        data.iter().any(|event| event["op"] == "compact"),
        "expected compact event in change feed"
    );
}

#[test]
fn compact_with_expected_generation_returns_generation_conflict() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);
    let append = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [
                { "id": 1, "values": [1.0, 0.0, 0.0] }
            ]
        }),
    );
    assert_eq!(append.status, 200, "body: {}", append.body);

    let compact = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/compact",
        &json!({
            "expected_generation": u64::MAX
        }),
    );
    assert_eq!(compact.status, 409, "body: {}", compact.body);
    assert_eq!(compact.json()["error"]["code"], "generation_conflict");
}

#[test]
fn compact_with_ttl_removes_expired_tombstones() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);

    let append = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [
                { "id": 1, "values": [1.0, 0.0, 0.0] },
                { "id": 2, "values": [0.0, 1.0, 0.0] }
            ]
        }),
    );
    assert_eq!(append.status, 200, "body: {}", append.body);

    let delete_resp = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/delete",
        &json!({ "ids": [2] }),
    );
    assert_eq!(delete_resp.status, 200, "body: {}", delete_resp.body);

    let compact = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/compact",
        &json!({
            "ttl_max_age_ms": 0,
            "now_ms": 4_102_444_800_000_i64
        }),
    );
    assert_eq!(compact.status, 200, "body: {}", compact.body);
    let compact_json = compact.json();
    assert_eq!(compact_json["kept_count"], 1);
    assert_eq!(compact_json["removed_tombstones"], 1);

    let stats = get(ctx.addr(), "/v1/db/vectors/collections/emb/stats");
    assert_eq!(stats.status, 200, "body: {}", stats.body);
    assert_eq!(stats.json()["tombstone_count"], 0);
}

#[test]
fn compact_rejects_combined_ttl_and_generation_options() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);
    let compact = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/compact",
        &json!({
            "expected_generation": 1,
            "ttl_max_age_ms": 0
        }),
    );
    assert_eq!(compact.status, 400, "body: {}", compact.body);
    assert_eq!(compact.json()["error"]["code"], "invalid_argument");
}

#[test]
fn compact_rejects_malformed_json_body() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/db/vectors/collections/emb/compact",
        &[("Content-Type", "application/json")],
        Some("{not-json"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_json");
}

#[test]
fn build_indexes_with_expected_generation_returns_generation_conflict() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);
    let build = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/indexes/build",
        &json!({
            "expected_generation": u64::MAX,
            "max_segments": 8
        }),
    );
    assert_eq!(build.status, 409, "body: {}", build.body);
    assert_eq!(build.json()["error"]["code"], "generation_conflict");
}

#[test]
fn build_indexes_with_expected_generation_succeeds_for_current_manifest() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);
    let build = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/indexes/build",
        &json!({
            "expected_generation": 0,
            "max_segments": 8
        }),
    );
    assert_eq!(build.status, 200, "body: {}", build.body);
    let json = build.json();
    assert_eq!(json["collection"], "emb");
    assert_eq!(json["built_segments"], 0);
    assert_eq!(json["failed_segments"], 0);
    assert_eq!(json["pending_segments"], 0);
}

#[test]
fn build_indexes_requires_non_empty_body_and_positive_max_segments() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);

    let missing_body = send_request(
        ctx.addr(),
        "POST",
        "/v1/db/vectors/collections/emb/indexes/build",
        &[("Content-Type", "application/json")],
        Some(""),
    );
    assert_eq!(missing_body.status, 400, "body: {}", missing_body.body);
    assert_eq!(missing_body.json()["error"]["code"], "invalid_argument");

    let zero_max_segments = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/indexes/build",
        &json!({
            "expected_generation": 0,
            "max_segments": 0
        }),
    );
    assert_eq!(zero_max_segments.status, 400, "body: {}", zero_max_segments.body);
    assert_eq!(zero_max_segments.json()["error"]["code"], "invalid_argument");
}

#[test]
fn build_indexes_rejects_malformed_json_body() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/db/vectors/collections/emb/indexes/build",
        &[("Content-Type", "application/json")],
        Some("{broken"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_json");
}

#[test]
fn build_indexes_returns_not_found_for_missing_collection() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/missing/indexes/build",
        &json!({
            "expected_generation": 0,
            "max_segments": 8
        }),
    );
    assert_eq!(resp.status, 404, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "collection_not_found");
}

#[test]
fn stats_changes_and_compact_return_not_found_for_missing_collection() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let stats = get(ctx.addr(), "/v1/db/vectors/collections/missing/stats");
    assert_eq!(stats.status, 404, "body: {}", stats.body);
    assert_eq!(stats.json()["error"]["code"], "collection_not_found");

    let changes = get(ctx.addr(), "/v1/db/vectors/collections/missing/changes?since=0&limit=10");
    assert_eq!(changes.status, 404, "body: {}", changes.body);
    assert_eq!(changes.json()["error"]["code"], "collection_not_found");

    let compact = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/missing/compact",
        &json!({}),
    );
    assert_eq!(compact.status, 404, "body: {}", compact.body);
    assert_eq!(compact.json()["error"]["code"], "collection_not_found");
}

#[test]
fn changes_limit_parameter_is_clamped_to_at_least_one() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);
    let append = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [
                { "id": 1, "values": [1.0, 0.0, 0.0] },
                { "id": 2, "values": [0.0, 1.0, 0.0] }
            ]
        }),
    );
    assert_eq!(append.status, 200, "body: {}", append.body);

    let changes = get(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/changes?since=0&limit=0",
    );
    assert_eq!(changes.status, 200, "body: {}", changes.body);
    let changes_json = changes.json();
    let data = changes_json["data"].as_array().expect("data");
    assert_eq!(data.len(), 1, "limit=0 should clamp to one change");
}

#[test]
fn changes_invalid_since_and_limit_fall_back_to_defaults() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);
    let append = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [
                { "id": 1, "values": [1.0, 0.0, 0.0] },
                { "id": 2, "values": [0.0, 1.0, 0.0] }
            ]
        }),
    );
    assert_eq!(append.status, 200, "body: {}", append.body);

    let changes = get(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/changes?since=abc&limit=xyz",
    );
    assert_eq!(changes.status, 200, "body: {}", changes.body);
    let changes_json = changes.json();
    let data = changes_json["data"].as_array().expect("data");
    assert!(
        data.len() >= 2,
        "fallback defaults should include existing append events"
    );
}

#[test]
fn changes_since_filters_out_older_events() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);
    let append_1 = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [{ "id": 1, "values": [1.0, 0.0, 0.0] }]
        }),
    );
    assert_eq!(append_1.status, 200, "body: {}", append_1.body);

    let append_2 = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [{ "id": 2, "values": [0.0, 1.0, 0.0] }]
        }),
    );
    assert_eq!(append_2.status, 200, "body: {}", append_2.body);

    let all = get(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/changes?since=0&limit=10",
    );
    assert_eq!(all.status, 200, "body: {}", all.body);
    let all_data = all.json()["data"].as_array().expect("data").clone();
    assert!(all_data.len() >= 2, "expected at least two change events");

    let first_seq = all_data[0]["seq"].as_u64().expect("first seq");
    let filtered = get(
        ctx.addr(),
        &format!("/v1/db/vectors/collections/emb/changes?since={first_seq}&limit=10"),
    );
    assert_eq!(filtered.status, 200, "body: {}", filtered.body);
    let filtered_data = filtered.json()["data"].as_array().expect("data").clone();
    assert!(
        filtered_data.iter().all(|event| {
            event["seq"]
                .as_u64()
                .expect("event seq")
                > first_seq
        }),
        "all returned seq values must be strictly greater than since"
    );
    assert!(
        filtered_data.len() < all_data.len(),
        "since filter should drop at least one older event"
    );
}

#[test]
fn write_endpoints_support_idempotency_replay_and_conflict() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);

    let first = post_json_with_headers(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &[("Idempotency-Key", "append-key-1")],
        &json!({
            "vectors": [
                { "id": 1, "values": [1.0, 0.0, 0.0] }
            ]
        }),
    );
    assert_eq!(first.status, 200, "body: {}", first.body);

    let replay = post_json_with_headers(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &[("Idempotency-Key", "append-key-1")],
        &json!({
            "vectors": [
                { "id": 1, "values": [1.0, 0.0, 0.0] }
            ]
        }),
    );
    assert_eq!(replay.status, 200, "body: {}", replay.body);

    let conflict = post_json_with_headers(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &[("Idempotency-Key", "append-key-1")],
        &json!({
            "vectors": [
                { "id": 2, "values": [0.0, 1.0, 0.0] }
            ]
        }),
    );
    assert_eq!(conflict.status, 409, "body: {}", conflict.body);
    assert_eq!(conflict.json()["error"]["code"], "idempotency_conflict");

    let changes = get(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/changes?since=0&limit=20",
    );
    assert_eq!(changes.status, 200, "body: {}", changes.body);
    let changes_json = changes.json();
    let data = changes_json["data"].as_array().expect("changes data");
    assert_eq!(data.len(), 1, "replayed write must not emit extra changes");
    assert_eq!(data[0]["op"], "append");
}

#[test]
fn upsert_and_delete_support_idempotency_replay_and_conflict() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);
    let append = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [
                { "id": 1, "values": [1.0, 0.0, 0.0] },
                { "id": 2, "values": [0.0, 1.0, 0.0] }
            ]
        }),
    );
    assert_eq!(append.status, 200, "body: {}", append.body);

    let upsert_first = post_json_with_headers(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/upsert",
        &[("Idempotency-Key", "upsert-key-1")],
        &json!({
            "vectors": [
                { "id": 1, "values": [0.0, 0.0, 1.0] }
            ]
        }),
    );
    assert_eq!(upsert_first.status, 200, "body: {}", upsert_first.body);

    let upsert_replay = post_json_with_headers(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/upsert",
        &[("Idempotency-Key", "upsert-key-1")],
        &json!({
            "vectors": [
                { "id": 1, "values": [0.0, 0.0, 1.0] }
            ]
        }),
    );
    assert_eq!(upsert_replay.status, 200, "body: {}", upsert_replay.body);

    let upsert_conflict = post_json_with_headers(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/upsert",
        &[("Idempotency-Key", "upsert-key-1")],
        &json!({
            "vectors": [
                { "id": 1, "values": [0.3, 0.3, 0.3] }
            ]
        }),
    );
    assert_eq!(upsert_conflict.status, 409, "body: {}", upsert_conflict.body);
    assert_eq!(
        upsert_conflict.json()["error"]["code"],
        "idempotency_conflict"
    );

    let delete_first = post_json_with_headers(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/delete",
        &[("Idempotency-Key", "delete-key-1")],
        &json!({
            "ids": [2]
        }),
    );
    assert_eq!(delete_first.status, 200, "body: {}", delete_first.body);

    let delete_replay = post_json_with_headers(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/delete",
        &[("Idempotency-Key", "delete-key-1")],
        &json!({
            "ids": [2]
        }),
    );
    assert_eq!(delete_replay.status, 200, "body: {}", delete_replay.body);

    let delete_conflict = post_json_with_headers(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/delete",
        &[("Idempotency-Key", "delete-key-1")],
        &json!({
            "ids": [1]
        }),
    );
    assert_eq!(delete_conflict.status, 409, "body: {}", delete_conflict.body);
    assert_eq!(
        delete_conflict.json()["error"]["code"],
        "idempotency_conflict"
    );
}

#[test]
fn compact_supports_idempotency_conflict_detection() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);
    let append = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append",
        &json!({
            "vectors": [{ "id": 1, "values": [1.0, 0.0, 0.0] }]
        }),
    );
    assert_eq!(append.status, 200, "body: {}", append.body);

    let first = post_json_with_headers(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/compact",
        &[("Idempotency-Key", "compact-key-1")],
        &json!({}),
    );
    assert_eq!(first.status, 200, "body: {}", first.body);

    let replay = post_json_with_headers(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/compact",
        &[("Idempotency-Key", "compact-key-1")],
        &json!({}),
    );
    assert_eq!(replay.status, 200, "body: {}", replay.body);

    let conflict = post_json_with_headers(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/compact",
        &[("Idempotency-Key", "compact-key-1")],
        &json!({"extra": true}),
    );
    assert_eq!(conflict.status, 409, "body: {}", conflict.body);
    assert_eq!(conflict.json()["error"]["code"], "idempotency_conflict");
}

#[test]
fn oversized_payload_is_rejected_with_resource_exhausted() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let oversized = "x".repeat((16 * 1024 * 1024) + 256);
    let response = send_request(
        ctx.addr(),
        "POST",
        "/v1/db/vectors/collections",
        &[("Content-Type", "application/json")],
        Some(&oversized),
    );
    assert_eq!(response.status, 413, "body: {}", response.body);
    assert_eq!(response.json()["error"]["code"], "resource_exhausted");
}

#[test]
fn db_routes_are_tenant_isolated_by_storage_prefix() {
    let temp = TempDir::new().expect("temp dir");

    let mut config = db_config(temp.path());
    config.gateway_secret = Some("secret".to_string());
    config.tenants = vec![
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

    let ctx = ServerTestContext::new(config);

    let tenant_a = [
        ("X-Talu-Gateway-Secret", "secret"),
        ("X-Talu-Tenant-Id", "tenant-a"),
    ];
    let tenant_b = [
        ("X-Talu-Gateway-Secret", "secret"),
        ("X-Talu-Tenant-Id", "tenant-b"),
    ];

    let create_a = post_json_with_headers(
        ctx.addr(),
        "/v1/db/vectors/collections",
        &tenant_a,
        &json!({ "name": "shared", "dims": 3 }),
    );
    assert_eq!(create_a.status, 201, "body: {}", create_a.body);

    let append_a = post_json_with_headers(
        ctx.addr(),
        "/v1/db/vectors/collections/shared/points/append",
        &tenant_a,
        &json!({
            "vectors": [
                { "id": 1, "values": [1.0, 0.0, 0.0] }
            ]
        }),
    );
    assert_eq!(append_a.status, 200, "body: {}", append_a.body);

    let query_b_before_create = post_json_with_headers(
        ctx.addr(),
        "/v1/db/vectors/collections/shared/points/query",
        &tenant_b,
        &json!({
            "vector": [1.0, 0.0, 0.0],
            "top_k": 1
        }),
    );
    assert_eq!(
        query_b_before_create.status, 404,
        "body: {}",
        query_b_before_create.body
    );
    assert_eq!(
        query_b_before_create.json()["error"]["code"],
        "collection_not_found"
    );

    let create_b = post_json_with_headers(
        ctx.addr(),
        "/v1/db/vectors/collections",
        &tenant_b,
        &json!({ "name": "shared", "dims": 3 }),
    );
    assert_eq!(create_b.status, 201, "body: {}", create_b.body);

    let query_b_after_create = post_json_with_headers(
        ctx.addr(),
        "/v1/db/vectors/collections/shared/points/query",
        &tenant_b,
        &json!({
            "vector": [1.0, 0.0, 0.0],
            "top_k": 1
        }),
    );
    assert_eq!(
        query_b_after_create.status, 200,
        "body: {}",
        query_b_after_create.body
    );
    assert!(query_b_after_create.json()["results"][0]["matches"]
        .as_array()
        .expect("matches")
        .is_empty());

    let list_a = get_with_headers(ctx.addr(), "/v1/db/vectors/collections", &tenant_a);
    assert_eq!(list_a.status, 200, "body: {}", list_a.body);
    let list_b = get_with_headers(ctx.addr(), "/v1/db/vectors/collections", &tenant_b);
    assert_eq!(list_b.status, 200, "body: {}", list_b.body);

    let data_a = list_a.json()["data"]
        .as_array()
        .expect("tenant-a data")
        .len();
    let data_b = list_b.json()["data"]
        .as_array()
        .expect("tenant-b data")
        .len();
    assert_eq!(data_a, 1);
    assert_eq!(data_b, 1);
}

#[test]
fn append_to_missing_collection_returns_404() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let append = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/missing/points/append",
        &json!({
            "vectors": [
                { "id": 1, "values": [1.0, 0.0, 0.0] }
            ]
        }),
    );
    assert_eq!(append.status, 404, "body: {}", append.body);
    let json = append.json();
    assert_eq!(json["error"]["code"], "collection_not_found");
}

#[test]
fn vector_endpoints_reject_trailing_path_segments() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    create_collection(&ctx, "emb", 3);

    let append_extra = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/emb/points/append/extra",
        &json!({
            "vectors": [{ "id": 1, "values": [1.0, 0.0, 0.0] }]
        }),
    );
    assert_eq!(append_extra.status, 404, "body: {}", append_extra.body);
    assert_eq!(append_extra.body, "not found");

    let stats_extra = get(ctx.addr(), "/v1/db/vectors/collections/emb/stats/extra");
    assert_eq!(stats_extra.status, 404, "body: {}", stats_extra.body);
    assert_eq!(stats_extra.json()["error"]["code"], "not_found");

    let changes_extra = get(ctx.addr(), "/v1/db/vectors/collections/emb/changes/extra");
    assert_eq!(changes_extra.status, 404, "body: {}", changes_extra.body);
    assert_eq!(changes_extra.json()["error"]["code"], "not_found");
}

#[test]
fn vector_collection_name_path_params_are_percent_decoded() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let collection_name = "my collection+v1";
    let encoded_name = "my%20collection%2Bv1";
    create_collection(&ctx, collection_name, 3);

    let append = post_json(
        ctx.addr(),
        &format!("/v1/db/vectors/collections/{encoded_name}/points/append"),
        &json!({
            "vectors": [{ "id": 42, "values": [1.0, 0.0, 0.0] }]
        }),
    );
    assert_eq!(
        append.status, 200,
        "collection path params should be percent-decoded for append; body: {}",
        append.body
    );

    let stats = get(
        ctx.addr(),
        &format!("/v1/db/vectors/collections/{encoded_name}/stats"),
    );
    assert_eq!(
        stats.status, 200,
        "collection path params should be percent-decoded for stats; body: {}",
        stats.body
    );
    assert_eq!(stats.json()["name"], collection_name);

    let query = post_json(
        ctx.addr(),
        &format!("/v1/db/vectors/collections/{encoded_name}/points/query"),
        &json!({
            "vector": [1.0, 0.0, 0.0],
            "top_k": 1
        }),
    );
    assert_eq!(
        query.status, 200,
        "collection path params should be percent-decoded for query; body: {}",
        query.body
    );
    assert_eq!(
        query.json()["results"][0]["matches"]
            .as_array()
            .expect("matches")
            .len(),
        1
    );

    let delete = delete(
        ctx.addr(),
        &format!("/v1/db/vectors/collections/{encoded_name}"),
    );
    assert_eq!(
        delete.status, 200,
        "collection path params should be percent-decoded for delete; body: {}",
        delete.body
    );
}

#[test]
fn vector_collection_name_path_params_reject_percent_decoded_path_separators() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let append = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections/bad%2Fname/points/append",
        &json!({
            "vectors": [{ "id": 1, "values": [1.0, 0.0, 0.0] }]
        }),
    );
    assert_eq!(
        append.status, 400,
        "collection names containing decoded path separators should be rejected; body: {}",
        append.body
    );
    assert_eq!(append.json()["error"]["code"], "invalid_argument");

    let stats = get(ctx.addr(), "/v1/db/vectors/collections/bad%2Fname/stats");
    assert_eq!(
        stats.status, 400,
        "collection names containing decoded path separators should be rejected; body: {}",
        stats.body
    );
    assert_eq!(stats.json()["error"]["code"], "invalid_argument");
}
