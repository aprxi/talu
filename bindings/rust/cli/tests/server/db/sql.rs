//! Integration tests for `POST /v1/db/sql/query` and `POST /v1/db/sql/explain`.
//!
//! Tests parameterized queries (structured response), the legacy
//! no-params path (raw JSON array backward compat), vector search
//! with filter_ids pre-filtering, and EXPLAIN QUERY PLAN.

use crate::server::common::*;
use crate::server::db::db_config;
use tempfile::TempDir;

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

// =============================================================================
// Storage unavailable
// =============================================================================

#[test]
fn query_returns_503_without_bucket() {
    let mut config = ServerConfig::new();
    config.no_bucket = true;
    let ctx = ServerTestContext::new(config);

    let body = serde_json::json!({"query": "SELECT 1"});
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 503, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["error"]["code"], "no_storage");
}

#[test]
fn query_requires_gateway_auth_when_gateway_is_enabled() {
    let temp = TempDir::new().expect("temp dir");
    let mut config = db_config(temp.path());
    config.gateway_secret = Some("secret".to_string());
    config.tenants = vec![TenantSpec {
        id: "tenant-a".to_string(),
        storage_prefix: "tenant-a".to_string(),
        allowed_models: vec![],
    }];
    let ctx = ServerTestContext::new(config);

    let body = serde_json::json!({"query": "SELECT 1"});
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 401, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "unauthorized");
}

#[test]
fn explain_returns_503_without_bucket_legacy_contract() {
    let mut config = ServerConfig::new();
    config.no_bucket = true;
    let ctx = ServerTestContext::new(config);

    let body = serde_json::json!({"query": "SELECT 1"});
    let resp = post_json(ctx.addr(), "/v1/db/sql/explain", &body);
    assert_eq!(resp.status, 503, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "no_storage");
}

#[test]
fn explain_requires_gateway_auth_when_gateway_is_enabled() {
    let temp = TempDir::new().expect("temp dir");
    let mut config = db_config(temp.path());
    config.gateway_secret = Some("secret".to_string());
    config.tenants = vec![TenantSpec {
        id: "tenant-a".to_string(),
        storage_prefix: "tenant-a".to_string(),
        allowed_models: vec![],
    }];
    let ctx = ServerTestContext::new(config);

    let body = serde_json::json!({"query": "SELECT 1"});
    let resp = post_json(ctx.addr(), "/v1/db/sql/explain", &body);
    assert_eq!(resp.status, 401, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "unauthorized");
}

// =============================================================================
// Invalid requests
// =============================================================================

#[test]
fn query_returns_400_for_invalid_json() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/db/sql/query",
        &[("Content-Type", "application/json")],
        Some("not valid json"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["error"]["code"], "invalid_json");
}

#[test]
fn explain_returns_400_for_invalid_json() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/db/sql/explain",
        &[("Content-Type", "application/json")],
        Some("not valid json"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_json");
}

#[test]
fn explain_accepts_json_body_without_content_type_header() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/db/sql/explain",
        &[],
        Some(r#"{"query":"SELECT 1 AS v"}"#),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert!(json["columns"].is_array(), "missing columns: {json}");
    assert!(json["rows"].is_array(), "missing rows: {json}");
    assert!(json["row_count"].is_number(), "missing row_count: {json}");
}

#[test]
fn query_returns_400_for_sql_error() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({"query": "SELECT * FROM nonexistent_table_xyz"});
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["error"]["code"], "sql_error");
    assert!(json["error"]["message"].as_str().unwrap().len() > 0);
}

#[test]
fn query_rejects_empty_sql_string() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let resp = post_json(
        ctx.addr(),
        "/v1/db/sql/query",
        &serde_json::json!({"query": ""}),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "sql_error");
}

#[test]
fn query_requires_query_field() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({"params": [1]});
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_json");
}

#[test]
fn explain_requires_query_field() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({"params": [1]});
    let resp = post_json(ctx.addr(), "/v1/db/sql/explain", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_json");
}

#[test]
fn explain_rejects_non_array_params_shape() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({
        "query": "SELECT 1",
        "params": {"bad": "shape"}
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/explain", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_json");
}

#[test]
fn query_rejects_non_array_params_shape() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({
        "query": "SELECT 1",
        "params": {"bad": "shape"}
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_json");
}

#[test]
fn query_rejects_object_params() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({
        "query": "SELECT ?1",
        "params": [{"key": "value"}]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["error"]["code"], "invalid_params");
}

#[test]
fn explain_rejects_object_params() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({
        "query": "SELECT ?1",
        "params": [{"key": "value"}]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/explain", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_params");
}

// =============================================================================
// Legacy path: no params → raw JSON array
// =============================================================================

#[test]
fn query_without_params_returns_raw_json_array() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({"query": "SELECT 1 AS val, 'hello' AS msg"});
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert_eq!(resp.header("content-type").unwrap(), "application/json",);

    // Legacy path returns a raw JSON array of row objects.
    let json = resp.json();
    assert!(json.is_array(), "expected array, got: {json}");
    let rows = json.as_array().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["val"], 1);
    assert_eq!(rows[0]["msg"], "hello");
}

#[test]
fn query_with_null_params_uses_legacy_path() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({"query": "SELECT 42 AS answer", "params": null});
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert!(
        json.is_array(),
        "null params should use legacy path: {json}"
    );
}

#[test]
fn query_accepts_json_body_without_content_type_header() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/db/sql/query",
        &[],
        Some(r#"{"query":"SELECT 7 AS v"}"#),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    assert!(json.is_array(), "legacy response should be array");
    assert_eq!(json[0]["v"], 7);
}

#[test]
fn query_with_empty_params_uses_structured_path() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({"query": "SELECT 42 AS answer", "params": []});
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert!(
        json.is_object(),
        "empty params should use structured path: {json}"
    );
    assert!(json["columns"].is_array(), "missing columns: {json}");
    assert!(json["rows"].is_array(), "missing rows: {json}");
    assert_eq!(json["row_count"], 1);
}

#[test]
fn explain_returns_structured_plan_response() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({
        "query": "SELECT ?1 AS val",
        "params": [42]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/explain", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert!(json["columns"].is_array(), "missing columns: {json}");
    assert!(json["rows"].is_array(), "missing rows: {json}");
    assert!(json["row_count"].is_number(), "missing row_count: {json}");
    assert!(
        json["row_count"].as_u64().unwrap_or(0) >= 1,
        "EXPLAIN should return at least one plan row: {json}"
    );
}

#[test]
fn explain_with_null_params_matches_no_params_path() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let none = post_json(
        ctx.addr(),
        "/v1/db/sql/explain",
        &serde_json::json!({"query": "SELECT 1"}),
    );
    assert_eq!(none.status, 200, "body: {}", none.body);
    let none_json = none.json();

    let null = post_json(
        ctx.addr(),
        "/v1/db/sql/explain",
        &serde_json::json!({"query": "SELECT 1", "params": null}),
    );
    assert_eq!(null.status, 200, "body: {}", null.body);
    let null_json = null.json();

    assert_eq!(none_json["row_count"], null_json["row_count"]);
    assert_eq!(none_json["columns"], null_json["columns"]);
}

// =============================================================================
// Parameterized path: structured response (rows are objects)
// =============================================================================

#[test]
fn query_with_integer_param_returns_structured_response() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({
        "query": "SELECT ?1 AS val",
        "params": [42]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    // Structured response has columns, rows, row_count.
    assert!(json["columns"].is_array(), "missing columns: {json}");
    assert!(json["rows"].is_array(), "missing rows: {json}");
    assert!(json["row_count"].is_number(), "missing row_count: {json}");

    let columns = json["columns"].as_array().unwrap();
    assert_eq!(columns.len(), 1);
    assert_eq!(columns[0], "val");

    let rows = json["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["val"], 42);

    assert_eq!(json["row_count"], 1);
}

#[test]
fn query_with_text_param() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({
        "query": "SELECT ?1 AS greeting",
        "params": ["hello world"]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let rows = json["rows"].as_array().unwrap();
    assert_eq!(rows[0]["greeting"], "hello world");
}

#[test]
fn query_with_float_param() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({
        "query": "SELECT ?1 AS pi",
        "params": [3.14]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let rows = json["rows"].as_array().unwrap();
    let val = rows[0]["pi"].as_f64().unwrap();
    assert!((val - 3.14).abs() < 0.001, "expected ~3.14, got {val}");
}

#[test]
fn query_with_null_param() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({
        "query": "SELECT ?1 AS empty",
        "params": [null]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let rows = json["rows"].as_array().unwrap();
    assert!(
        rows[0]["empty"].is_null(),
        "expected null, got: {}",
        rows[0]["empty"]
    );
}

#[test]
fn query_with_bool_param_maps_to_integer() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({
        "query": "SELECT ?1 AS flag_true, ?2 AS flag_false",
        "params": [true, false]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let rows = json["rows"].as_array().unwrap();
    assert_eq!(rows[0]["flag_true"], 1, "true should map to 1");
    assert_eq!(rows[0]["flag_false"], 0, "false should map to 0");
}

#[test]
fn query_with_multiple_mixed_params() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({
        "query": "SELECT ?1 AS name, ?2 AS age, ?3 AS score, ?4 AS notes",
        "params": ["Alice", 30, 95.5, null]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["row_count"], 1);

    let columns = json["columns"].as_array().unwrap();
    assert_eq!(columns, &["name", "age", "score", "notes"]);

    let rows = json["rows"].as_array().unwrap();
    assert_eq!(rows[0]["name"], "Alice");
    assert_eq!(rows[0]["age"], 30);
    assert!((rows[0]["score"].as_f64().unwrap() - 95.5).abs() < 0.001);
    assert!(rows[0]["notes"].is_null());
}

#[test]
fn query_with_blob_param() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    // Array of numbers → packed f32 blob parameter.
    let body = serde_json::json!({
        "query": "SELECT length(?1) AS blob_len",
        "params": [[1.0, 2.0, 3.0]]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let rows = json["rows"].as_array().unwrap();
    // 3 floats × 4 bytes each = 12 bytes.
    assert_eq!(rows[0]["blob_len"], 12, "3 f32s should be 12 bytes");
}

#[test]
fn query_rejects_non_numeric_array_elements() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({
        "query": "SELECT ?1",
        "params": [["not", "numbers"]]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["error"]["code"], "invalid_params");
}

// =============================================================================
// Structured response format
// =============================================================================

#[test]
fn structured_response_has_correct_shape() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({
        "query": "SELECT ?1 AS a, ?2 AS b UNION ALL SELECT ?3, ?4",
        "params": [1, "x", 2, "y"]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();

    // Must have exactly these three top-level keys.
    assert!(json.is_object());
    let obj = json.as_object().unwrap();
    assert!(obj.contains_key("columns"), "missing 'columns'");
    assert!(obj.contains_key("rows"), "missing 'rows'");
    assert!(obj.contains_key("row_count"), "missing 'row_count'");

    assert_eq!(json["columns"].as_array().unwrap(), &["a", "b"]);
    assert_eq!(json["row_count"], 2);

    let rows = json["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    // Each row is an object with column-name keys.
    assert!(rows[0].is_object(), "rows should be objects: {}", rows[0]);
    assert_eq!(rows[0].as_object().unwrap().len(), 2);
}

// =============================================================================
// Vector search TVF via SQL
// =============================================================================

/// Helper: create a vector collection and append vectors via REST, then return
/// the ServerTestContext for SQL queries.
fn setup_vector_collection(
    ctx: &ServerTestContext,
    collection: &str,
    dims: u32,
    vectors: &[(u64, Vec<f32>)],
) {
    // Create collection.
    let resp = post_json(
        ctx.addr(),
        "/v1/db/vectors/collections",
        &serde_json::json!({
            "name": collection,
            "dims": dims
        }),
    );
    assert_eq!(
        resp.status, 201,
        "create collection: status={}, body={}",
        resp.status, resp.body
    );

    // Append vectors.
    let vecs: Vec<serde_json::Value> = vectors
        .iter()
        .map(|(id, values)| {
            serde_json::json!({
                "id": id,
                "values": values
            })
        })
        .collect();
    let resp = post_json(
        ctx.addr(),
        &format!("/v1/db/vectors/collections/{collection}/points/append"),
        &serde_json::json!({ "vectors": vecs }),
    );
    assert_eq!(resp.status, 200, "append: body={}", resp.body);
}

#[test]
fn sql_vector_search_returns_results() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    setup_vector_collection(
        &ctx,
        "emb",
        3,
        &[
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0]),
        ],
    );

    let body = serde_json::json!({
        "query": "SELECT id, score FROM vector_search(?1, ?2, ?3)",
        "params": ["emb", [1.0, 0.0, 0.0], 2]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert!(json["columns"].is_array(), "missing columns: {json}");
    let columns = json["columns"].as_array().unwrap();
    assert_eq!(columns, &["id", "score"]);

    assert!(json["rows"].is_array(), "missing rows: {json}");
    let rows = json["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2, "expected 2 rows (k=2), got: {json}");

    // Top result should be id=1 (exact match for [1,0,0]).
    assert_eq!(rows[0]["id"], 1, "top result should be id=1: {json}");
}

#[test]
fn sql_vector_search_respects_top_k() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let mut vectors = Vec::new();
    for i in 1..=5u64 {
        vectors.push((i, vec![i as f32, 0.0, 0.0]));
    }
    setup_vector_collection(&ctx, "emb5", 3, &vectors);

    let body = serde_json::json!({
        "query": "SELECT id, score FROM vector_search(?1, ?2, ?3)",
        "params": ["emb5", [1.0, 0.0, 0.0], 2]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["row_count"], 2, "k=2 should return 2 rows: {json}");
}

#[test]
fn sql_vector_search_scores_descending() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    setup_vector_collection(
        &ctx,
        "emb_sort",
        3,
        &[
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.5, 0.5, 0.0]),
            (3, vec![0.0, 0.0, 1.0]),
        ],
    );

    let body = serde_json::json!({
        "query": "SELECT id, score FROM vector_search(?1, ?2, ?3)",
        "params": ["emb_sort", [1.0, 0.0, 0.0], 3]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let rows = json["rows"].as_array().unwrap();
    assert!(rows.len() >= 2, "need at least 2 rows: {json}");

    // Scores should be in descending order.
    for i in 0..rows.len() - 1 {
        let s1 = rows[i]["score"].as_f64().unwrap();
        let s2 = rows[i + 1]["score"].as_f64().unwrap();
        assert!(
            s1 >= s2,
            "scores not descending: row[{i}]={s1} < row[{}]={s2}",
            i + 1
        );
    }
}

#[test]
fn sql_vector_search_missing_collection_returns_empty() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    // VectorAdapter auto-creates the path; search returns 0 results.
    let body = serde_json::json!({
        "query": "SELECT id, score FROM vector_search(?1, ?2, ?3)",
        "params": ["nonexistent_collection", [1.0, 0.0, 0.0], 5]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(
        json["row_count"], 0,
        "empty collection should return 0 rows: {json}"
    );
}

#[test]
fn sql_vector_search_default_k() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let mut vectors = Vec::new();
    for i in 1..=15u64 {
        let mut v = vec![0.0f32; 3];
        v[0] = i as f32;
        vectors.push((i, v));
    }
    setup_vector_collection(&ctx, "emb_dk", 3, &vectors);

    // No k argument — should use default k=10.
    let body = serde_json::json!({
        "query": "SELECT id, score FROM vector_search(?1, ?2)",
        "params": ["emb_dk", [1.0, 0.0, 0.0]]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["row_count"], 10, "default k should be 10: {json}");
}

#[test]
fn sql_query_uses_tenant_scoped_storage_prefix() {
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

    let create = post_json_with_headers(
        ctx.addr(),
        "/v1/db/vectors/collections",
        &tenant_a,
        &serde_json::json!({
            "name": "shared",
            "dims": 3
        }),
    );
    assert_eq!(create.status, 201, "body: {}", create.body);

    let append = post_json_with_headers(
        ctx.addr(),
        "/v1/db/vectors/collections/shared/points/append",
        &tenant_a,
        &serde_json::json!({
            "vectors": [
                { "id": 1, "values": [1.0, 0.0, 0.0] }
            ]
        }),
    );
    assert_eq!(append.status, 200, "body: {}", append.body);

    let query_a = post_json_with_headers(
        ctx.addr(),
        "/v1/db/sql/query",
        &tenant_a,
        &serde_json::json!({
            "query": "SELECT id, score FROM vector_search(?1, ?2, ?3)",
            "params": ["shared", [1.0, 0.0, 0.0], 1]
        }),
    );
    assert_eq!(query_a.status, 200, "body: {}", query_a.body);
    assert_eq!(
        query_a.json()["row_count"],
        1,
        "tenant-a should see one row"
    );

    let query_b = post_json_with_headers(
        ctx.addr(),
        "/v1/db/sql/query",
        &tenant_b,
        &serde_json::json!({
            "query": "SELECT id, score FROM vector_search(?1, ?2, ?3)",
            "params": ["shared", [1.0, 0.0, 0.0], 1]
        }),
    );
    assert_eq!(query_b.status, 200, "body: {}", query_b.body);
    assert_eq!(
        query_b.json()["row_count"],
        0,
        "tenant-b should see no rows"
    );
}

// =============================================================================
// Error response format
// =============================================================================

#[test]
fn error_responses_have_standard_shape() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({"query": "INVALID SQL !!!"});
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 400);

    let json = resp.json();
    assert!(json["error"].is_object(), "error should be object: {json}");
    assert!(
        json["error"]["code"].is_string(),
        "error.code missing: {json}"
    );
    assert!(
        json["error"]["message"].is_string(),
        "error.message missing: {json}"
    );
}

// =============================================================================
// EXPLAIN endpoint
// =============================================================================

#[test]
fn explain_returns_structured_plan() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({"query": "SELECT 1"});
    let resp = post_json(ctx.addr(), "/v1/db/sql/explain", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert!(json["columns"].is_array(), "missing columns: {json}");
    assert!(json["rows"].is_array(), "missing rows: {json}");
    assert!(json["row_count"].is_number(), "missing row_count: {json}");
}

#[test]
fn explain_with_vector_search() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    setup_vector_collection(&ctx, "emb_explain", 3, &[(1, vec![1.0, 0.0, 0.0])]);

    let body = serde_json::json!({
        "query": "SELECT id, score FROM vector_search(?1, ?2, ?3)",
        "params": ["emb_explain", [1.0, 0.0, 0.0], 5]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/explain", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    // Plan detail should mention vector_search.
    let body_str = resp.body.to_lowercase();
    assert!(
        body_str.contains("vector_search"),
        "EXPLAIN should mention vector_search: {json}"
    );
}

#[test]
fn explain_returns_400_for_invalid_sql() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({"query": "INVALID SQL !!!"});
    let resp = post_json(ctx.addr(), "/v1/db/sql/explain", &body);
    assert_eq!(resp.status, 400, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["error"]["code"], "sql_error");
}

// =============================================================================
// Vector search with filter_ids
// =============================================================================

#[test]
fn sql_vector_search_with_filter_ids() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    setup_vector_collection(
        &ctx,
        "emb_filt",
        3,
        &[
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0]),
        ],
    );

    let body = serde_json::json!({
        "query": "SELECT id, score FROM vector_search(?1, ?2, ?3, ?4)",
        "params": ["emb_filt", [1.0, 0.0, 0.0], 10, "[1, 3]"]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(
        json["row_count"], 2,
        "filter_ids [1,3] should return 2 rows: {json}"
    );

    let rows = json["rows"].as_array().unwrap();
    let ids: Vec<i64> = rows.iter().map(|r| r["id"].as_i64().unwrap()).collect();
    assert!(ids.contains(&1), "should contain id=1: {json}");
    assert!(ids.contains(&3), "should contain id=3: {json}");
    assert!(!ids.contains(&2), "should NOT contain id=2: {json}");
}

#[test]
fn sql_vector_search_with_empty_filter_ids_returns_empty() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    setup_vector_collection(
        &ctx,
        "emb_filt_empty",
        3,
        &[(1, vec![1.0, 0.0, 0.0]), (2, vec![0.0, 1.0, 0.0])],
    );

    let body = serde_json::json!({
        "query": "SELECT id, score FROM vector_search(?1, ?2, ?3, ?4)",
        "params": ["emb_filt_empty", [1.0, 0.0, 0.0], 10, "[]"]
    });
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(
        json["row_count"], 0,
        "empty filter_ids should return 0 rows: {json}"
    );
}
