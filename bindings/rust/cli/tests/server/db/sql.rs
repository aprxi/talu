//! Integration tests for `POST /v1/db/sql/query`.
//!
//! Tests parameterized queries (structured response) and the legacy
//! no-params path (raw JSON array backward compat).

use crate::server::common::*;
use crate::server::db::db_config;
use tempfile::TempDir;

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
    assert_eq!(
        resp.header("content-type").unwrap(),
        "application/json",
    );

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
    assert!(json.is_array(), "null params should use legacy path: {json}");
}

#[test]
fn query_with_empty_params_uses_structured_path() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(db_config(temp.path()));

    let body = serde_json::json!({"query": "SELECT 42 AS answer", "params": []});
    let resp = post_json(ctx.addr(), "/v1/db/sql/query", &body);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert!(json.is_object(), "empty params should use structured path: {json}");
    assert!(json["columns"].is_array(), "missing columns: {json}");
    assert!(json["rows"].is_array(), "missing rows: {json}");
    assert_eq!(json["row_count"], 1);
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
    assert!(rows[0]["empty"].is_null(), "expected null, got: {}", rows[0]["empty"]);
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
    assert!(json["error"]["code"].is_string(), "error.code missing: {json}");
    assert!(
        json["error"]["message"].is_string(),
        "error.message missing: {json}"
    );
}
