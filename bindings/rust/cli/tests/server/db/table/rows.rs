//! Integration tests for generic table rows API (`/v1/db/tables/{ns}/rows`).
//!
//! Tests write, scan, get-by-hash, and delete operations on the low-level
//! table engine HTTP surface.

use super::{documents_config, no_bucket_config};
use crate::server::common::*;
use tempfile::TempDir;

// =============================================================================
// Helpers
// =============================================================================

fn write_row(
    addr: std::net::SocketAddr,
    ns: &str,
    schema_id: u16,
    columns: &[serde_json::Value],
) -> HttpResponse {
    let body = serde_json::json!({
        "schema_id": schema_id,
        "columns": columns,
    });
    post_json(addr, &format!("/v1/db/tables/{ns}/rows"), &body)
}

fn write_row_with_policy(
    addr: std::net::SocketAddr,
    ns: &str,
    schema_id: u16,
    columns: &[serde_json::Value],
    policy: serde_json::Value,
) -> HttpResponse {
    let body = serde_json::json!({
        "schema_id": schema_id,
        "columns": columns,
        "policy": policy,
    });
    post_json(addr, &format!("/v1/db/tables/{ns}/rows"), &body)
}

fn scan_rows(addr: std::net::SocketAddr, ns: &str, schema_id: u16) -> HttpResponse {
    get(addr, &format!("/v1/db/tables/{ns}/rows?schema_id={schema_id}"))
}

fn scan_rows_with_limit(
    addr: std::net::SocketAddr,
    ns: &str,
    schema_id: u16,
    limit: u32,
) -> HttpResponse {
    get(
        addr,
        &format!("/v1/db/tables/{ns}/rows?schema_id={schema_id}&limit={limit}"),
    )
}

/// Standard test columns: dedup key (col 1), timestamp (col 2), payload (col 20).
fn standard_columns(key: u64, payload: &str) -> Vec<serde_json::Value> {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;
    vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": key}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": ts}),
        serde_json::json!({"column_id": 20, "type": "string", "value": payload}),
    ]
}

/// Columns with an explicit timestamp value (for deterministic ordering tests).
fn columns_with_ts(key: u64, ts: i64, payload: &str) -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": key}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": ts}),
        serde_json::json!({"column_id": 20, "type": "string", "value": payload}),
    ]
}

// =============================================================================
// Storage requirement
// =============================================================================

#[test]
fn write_row_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let cols = standard_columns(1, "hello");
    let resp = write_row(ctx.addr(), "test_ns", 10, &cols);
    assert_eq!(resp.status, 503, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "no_storage");
}

#[test]
fn scan_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = scan_rows(ctx.addr(), "test_ns", 10);
    assert_eq!(resp.status, 503, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "no_storage");
}

// =============================================================================
// Write + Scan roundtrip
// =============================================================================

#[test]
fn write_and_scan_roundtrip() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = standard_columns(42, "hello world");
    let resp = write_row(ctx.addr(), "roundtrip", 10, &cols);
    assert_eq!(resp.status, 200, "write: {}", resp.body);
    assert_eq!(resp.json()["status"], "ok");

    let resp = scan_rows(ctx.addr(), "roundtrip", 10);
    assert_eq!(resp.status, 200, "scan: {}", resp.body);

    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows array");
    assert_eq!(rows.len(), 1, "should have one row");
    assert_eq!(json["has_more"], false);

    let row = &rows[0];
    assert_eq!(row["payload"], "hello world");
    assert_eq!(row["payload_encoding"], "utf8");

    // Verify scalars contain our dedup key (col 1).
    let scalars = row["scalars"].as_array().expect("scalars array");
    let col1 = scalars.iter().find(|s| s["column_id"] == 1);
    assert!(col1.is_some(), "should have column 1 scalar");
    assert_eq!(col1.unwrap()["value"], 42);
}

#[test]
fn scan_returns_empty_initially() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = scan_rows(ctx.addr(), "empty_ns", 10);
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows array");
    assert_eq!(rows.len(), 0);
    assert_eq!(json["has_more"], false);
}

// =============================================================================
// Column types
// =============================================================================

#[test]
fn write_scalar_i64_column() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": -999}),
        serde_json::json!({"column_id": 20, "type": "string", "value": "i64 test"}),
    ];
    let resp = write_row(ctx.addr(), "i64ns", 10, &cols);
    assert_eq!(resp.status, 200, "write: {}", resp.body);

    let resp = scan_rows(ctx.addr(), "i64ns", 10);
    assert_eq!(resp.status, 200, "scan: {}", resp.body);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 1);

    // Timestamp column (col 2) should carry -999 as unsigned reinterpretation.
    let scalars = rows[0]["scalars"].as_array().expect("scalars");
    let col2 = scalars.iter().find(|s| s["column_id"] == 2).expect("col 2");
    // -999i64 as u64 = 18446744073709550617
    assert_eq!(col2["value"], 18446744073709550617_u64);
}

#[test]
fn write_scalar_f64_column() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Scan only returns dedup + ts scalars, so a non-indexed f64 column
    // is invisible in results. This test verifies the json_to_column_value
    // f64 path accepts and stores the value without corrupting the row.
    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 100}),
        serde_json::json!({"column_id": 5, "type": "scalar_f64", "value": 3.14}),
        serde_json::json!({"column_id": 20, "type": "string", "value": "f64 test"}),
    ];
    let resp = write_row(ctx.addr(), "f64ns", 10, &cols);
    assert_eq!(resp.status, 200, "write: {}", resp.body);

    let resp = scan_rows(ctx.addr(), "f64ns", 10);
    assert_eq!(resp.status, 200, "scan: {}", resp.body);
    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 1, "row stored despite extra f64 column");
    assert_eq!(rows[0]["payload"], "f64 test", "payload not corrupted");
}

#[test]
fn write_varbytes_column() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 100}),
        serde_json::json!({"column_id": 20, "type": "varbytes", "value": "deadbeef"}),
    ];
    let resp = write_row(ctx.addr(), "varns", 10, &cols);
    assert_eq!(resp.status, 200, "write: {}", resp.body);

    let resp = scan_rows(ctx.addr(), "varns", 10);
    assert_eq!(resp.status, 200, "scan: {}", resp.body);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["payload"], "deadbeef");
    assert_eq!(rows[0]["payload_encoding"], "hex");
}

#[test]
fn write_json_column() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 100}),
        serde_json::json!({"column_id": 20, "type": "json", "value": {"key": "val", "n": 42}}),
    ];
    let resp = write_row(ctx.addr(), "jsonns", 10, &cols);
    assert_eq!(resp.status, 200, "write: {}", resp.body);

    let resp = scan_rows(ctx.addr(), "jsonns", 10);
    assert_eq!(resp.status, 200, "scan: {}", resp.body);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["payload_encoding"], "utf8");

    // Payload is the serialized JSON string.
    let payload_str = rows[0]["payload"].as_str().expect("payload string");
    let parsed: serde_json::Value = serde_json::from_str(payload_str).expect("parse payload json");
    assert_eq!(parsed["key"], "val");
    assert_eq!(parsed["n"], 42);
}

#[test]
fn write_string_column() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 100}),
        serde_json::json!({"column_id": 20, "type": "string", "value": "unicode: \u{1f600}"}),
    ];
    let resp = write_row(ctx.addr(), "strns", 10, &cols);
    assert_eq!(resp.status, 200, "write: {}", resp.body);

    let resp = scan_rows(ctx.addr(), "strns", 10);
    assert_eq!(resp.status, 200, "scan: {}", resp.body);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["payload"], "unicode: \u{1f600}");
    assert_eq!(rows[0]["payload_encoding"], "utf8");
}

// =============================================================================
// Get by hash
// =============================================================================

#[test]
fn get_row_by_hash() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = standard_columns(77, "find me");
    let resp = write_row(ctx.addr(), "getns", 10, &cols);
    assert_eq!(resp.status, 200, "write: {}", resp.body);

    // Scan to discover the dedup hash.
    let resp = scan_rows(ctx.addr(), "getns", 10);
    assert_eq!(resp.status, 200, "scan: {}", resp.body);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 1);

    let scalars = rows[0]["scalars"].as_array().expect("scalars");
    let col1 = scalars.iter().find(|s| s["column_id"] == 1).expect("col 1");
    let hash = col1["value"].as_u64().expect("hash u64");

    // GET by hash.
    let resp = get(
        ctx.addr(),
        &format!("/v1/db/tables/getns/rows/{hash}?schema_id=10"),
    );
    assert_eq!(resp.status, 200, "get: {}", resp.body);

    let json = resp.json();
    assert!(json["row"].is_object(), "row should be present");
    assert_eq!(json["row"]["payload"], "find me");
}

#[test]
fn get_row_returns_null_for_missing_hash() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Write at least one row so the namespace exists.
    let cols = standard_columns(1, "x");
    write_row(ctx.addr(), "missns", 10, &cols);

    let resp = get(
        ctx.addr(),
        "/v1/db/tables/missns/rows/9999999999?schema_id=10",
    );
    assert_eq!(resp.status, 200, "get: {}", resp.body);
    assert!(resp.json()["row"].is_null(), "row should be null for missing hash");
}

// =============================================================================
// Delete
// =============================================================================

#[test]
fn delete_row_hides_from_get() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Write with explicit policy that includes a tombstone_schema_id so
    // the delete handler can write a tombstone row.
    let policy = serde_json::json!({
        "active_schema_ids": [10],
        "tombstone_schema_id": 2,
        "dedup_column_id": 1,
        "ts_column_id": 2,
    });
    let cols = standard_columns(55, "ephemeral");
    let resp = write_row_with_policy(ctx.addr(), "delns", 10, &cols, policy);
    assert_eq!(resp.status, 200, "write: {}", resp.body);

    // Scan to get the hash.
    let resp = scan_rows(ctx.addr(), "delns", 10);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 1);
    let scalars = rows[0]["scalars"].as_array().expect("scalars");
    let hash = scalars
        .iter()
        .find(|s| s["column_id"] == 1)
        .expect("col 1")["value"]
        .as_u64()
        .expect("hash");

    // Verify GET finds the row before delete.
    let resp = get(
        ctx.addr(),
        &format!("/v1/db/tables/delns/rows/{hash}?schema_id=10"),
    );
    assert_eq!(resp.status, 200, "get before delete: {}", resp.body);
    assert!(resp.json()["row"].is_object(), "row should exist before delete");

    // Delete.
    let resp = delete(
        ctx.addr(),
        &format!("/v1/db/tables/delns/rows/{hash}"),
    );
    assert_eq!(resp.status, 200, "delete: {}", resp.body);
    assert_eq!(resp.json()["status"], "deleted");

    // GET by hash should return null (tombstone applied via persisted policy).
    let resp = get(
        ctx.addr(),
        &format!("/v1/db/tables/delns/rows/{hash}?schema_id=10"),
    );
    assert_eq!(resp.status, 200, "get after delete: {}", resp.body);
    assert!(
        resp.json()["row"].is_null(),
        "deleted row should be null in GET: {}",
        resp.body
    );
}

// =============================================================================
// Scan parameters
// =============================================================================

#[test]
fn scan_requires_schema_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/db/tables/anyns/rows");
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "missing_param");
}

#[test]
fn scan_respects_limit() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Write 3 rows with distinct dedup keys.
    for i in 1..=3 {
        let cols = standard_columns(i, &format!("row {i}"));
        let resp = write_row(ctx.addr(), "limitns", 10, &cols);
        assert_eq!(resp.status, 200, "write {i}: {}", resp.body);
    }

    let resp = scan_rows_with_limit(ctx.addr(), "limitns", 10, 1);
    assert_eq!(resp.status, 200, "scan: {}", resp.body);

    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 1, "limit=1 should return 1 row");
    assert_eq!(json["has_more"], true, "should have more rows");
}

// =============================================================================
// Namespace isolation
// =============================================================================

#[test]
fn namespaces_are_isolated() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = standard_columns(1, "alpha data");
    let resp = write_row(ctx.addr(), "alpha", 10, &cols);
    assert_eq!(resp.status, 200, "write: {}", resp.body);

    // Scan a different namespace — should be empty.
    let resp = scan_rows(ctx.addr(), "beta", 10);
    assert_eq!(resp.status, 200, "scan: {}", resp.body);
    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 0, "beta should be empty");

    // Scan alpha — should have data.
    let resp = scan_rows(ctx.addr(), "alpha", 10);
    assert_eq!(resp.status, 200, "scan: {}", resp.body);
    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 1, "alpha should have one row");
}

// =============================================================================
// Error paths
// =============================================================================

#[test]
fn write_rejects_invalid_column_type() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "bogus_type", "value": 1}),
    ];
    let resp = write_row(ctx.addr(), "errns", 10, &cols);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_column");
}

#[test]
fn write_rejects_invalid_json() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/db/tables/errns/rows",
        &[("Content-Type", "application/json")],
        Some("{not valid json}"),
    );
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_json");
}

#[test]
fn write_rejects_type_value_mismatch() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": "not_a_number"}),
    ];
    let resp = write_row(ctx.addr(), "errns", 10, &cols);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_column");
}

// =============================================================================
// Policy
// =============================================================================

#[test]
fn write_with_explicit_policy() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let policy = serde_json::json!({
        "active_schema_ids": [10, 11],
        "tombstone_schema_id": 2,
        "dedup_column_id": 1,
        "ts_column_id": 2,
    });

    let cols = standard_columns(1, "policy row 1");
    let resp = write_row_with_policy(ctx.addr(), "policyns", 10, &cols, policy);
    assert_eq!(resp.status, 200, "write 1: {}", resp.body);
    assert_eq!(resp.json()["status"], "ok");

    // Second write — policy should load from disk (no policy field needed).
    let cols2 = standard_columns(2, "policy row 2");
    let resp = write_row(ctx.addr(), "policyns", 10, &cols2);
    assert_eq!(resp.status, 200, "write 2: {}", resp.body);

    // Both rows should be present.
    let resp = scan_rows(ctx.addr(), "policyns", 10);
    assert_eq!(resp.status, 200, "scan: {}", resp.body);
    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 2, "should have two rows");
}

// =============================================================================
// Dedup behavior
// =============================================================================

/// Writing two rows with the same dedup key keeps only the latest.
#[test]
fn dedup_keeps_latest_row() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols1 = columns_with_ts(100, 1000, "version-1");
    let resp = write_row(ctx.addr(), "dedupns", 10, &cols1);
    assert_eq!(resp.status, 200, "write 1: {}", resp.body);

    let cols2 = columns_with_ts(100, 2000, "version-2");
    let resp = write_row(ctx.addr(), "dedupns", 10, &cols2);
    assert_eq!(resp.status, 200, "write 2: {}", resp.body);

    let resp = scan_rows(ctx.addr(), "dedupns", 10);
    assert_eq!(resp.status, 200, "scan: {}", resp.body);
    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 1, "dedup should collapse to one row");
    assert_eq!(rows[0]["payload"], "version-2", "should keep the latest");
}

/// Multiple distinct dedup keys coexist.
#[test]
fn dedup_distinct_keys_coexist() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    for key in [10, 20, 30] {
        let cols = columns_with_ts(key, 1000, &format!("key-{key}"));
        let resp = write_row(ctx.addr(), "multins", 10, &cols);
        assert_eq!(resp.status, 200, "write key={key}: {}", resp.body);
    }

    let resp = scan_rows(ctx.addr(), "multins", 10);
    assert_eq!(resp.status, 200, "scan: {}", resp.body);
    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 3, "three distinct keys should all appear");
}

// =============================================================================
// Scan ordering and cursor pagination
// =============================================================================

/// Default scan returns rows newest-first (reverse chronological).
#[test]
fn scan_returns_newest_first() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols_old = columns_with_ts(1, 1000, "old");
    let resp = write_row(ctx.addr(), "orderns", 10, &cols_old);
    assert_eq!(resp.status, 200, "write old: {}", resp.body);

    let cols_new = columns_with_ts(2, 2000, "new");
    let resp = write_row(ctx.addr(), "orderns", 10, &cols_new);
    assert_eq!(resp.status, 200, "write new: {}", resp.body);

    let resp = scan_rows(ctx.addr(), "orderns", 10);
    assert_eq!(resp.status, 200, "scan: {}", resp.body);
    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["payload"], "new", "first should be newest");
    assert_eq!(rows[1]["payload"], "old", "second should be oldest");
}

/// Cursor-based pagination: first page returns cursor, second page continues.
#[test]
fn scan_cursor_pagination() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Write 3 rows with distinct timestamps (deterministic ordering).
    for (key, ts) in [(1, 1000), (2, 2000), (3, 3000)] {
        let cols = columns_with_ts(key, ts, &format!("row-{key}"));
        let resp = write_row(ctx.addr(), "cursorns", 10, &cols);
        assert_eq!(resp.status, 200, "write key={key}: {}", resp.body);
    }

    // Page 1: limit=2 (newest first → row-3, row-2).
    let resp = scan_rows_with_limit(ctx.addr(), "cursorns", 10, 2);
    assert_eq!(resp.status, 200, "page 1: {}", resp.body);
    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 2, "page 1 should have 2 rows");
    assert_eq!(json["has_more"], true, "page 1 should signal more");
    assert_eq!(rows[0]["payload"], "row-3");
    assert_eq!(rows[1]["payload"], "row-2");

    // Extract cursor from the last row of page 1.
    let last = &rows[1];
    let scalars = last["scalars"].as_array().expect("scalars");
    let ts_scalar = scalars.iter().find(|s| s["column_id"] == 2).expect("ts col");
    let cursor_ts = ts_scalar["value"].as_u64().expect("cursor ts");
    let pk_scalar = scalars.iter().find(|s| s["column_id"] == 1).expect("pk col");
    let cursor_hash = pk_scalar["value"].as_u64().expect("cursor hash");

    // Page 2: use cursor from page 1.
    let resp = get(
        ctx.addr(),
        &format!(
            "/v1/db/tables/cursorns/rows?schema_id=10&limit=2&cursor_ts={cursor_ts}&cursor_hash={cursor_hash}"
        ),
    );
    assert_eq!(resp.status, 200, "page 2: {}", resp.body);
    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 1, "page 2 should have 1 remaining row");
    assert_eq!(json["has_more"], false, "page 2 should signal no more");
    assert_eq!(rows[0]["payload"], "row-1");
}

// =============================================================================
// Schema isolation
// =============================================================================

/// Different schema IDs in the same namespace are isolated during scan.
#[test]
fn schema_ids_are_isolated() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Write with schema 10.
    let policy = serde_json::json!({
        "active_schema_ids": [10, 11],
        "dedup_column_id": 1,
        "ts_column_id": 2,
    });
    let cols = standard_columns(1, "schema-10-data");
    let resp = write_row_with_policy(ctx.addr(), "schemans", 10, &cols, policy);
    assert_eq!(resp.status, 200, "write schema 10: {}", resp.body);

    // Write with schema 11 (different dedup key).
    let cols = standard_columns(2, "schema-11-data");
    let resp = write_row(ctx.addr(), "schemans", 11, &cols);
    assert_eq!(resp.status, 200, "write schema 11: {}", resp.body);

    // Scan schema 10 — should only see its row.
    let resp = scan_rows(ctx.addr(), "schemans", 10);
    assert_eq!(resp.status, 200, "scan 10: {}", resp.body);
    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 1, "schema 10 should have 1 row");
    assert_eq!(rows[0]["payload"], "schema-10-data");

    // Scan schema 11 — should only see its row.
    let resp = scan_rows(ctx.addr(), "schemans", 11);
    assert_eq!(resp.status, 200, "scan 11: {}", resp.body);
    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 1, "schema 11 should have 1 row");
    assert_eq!(rows[0]["payload"], "schema-11-data");
}

// =============================================================================
// Additional storage-requirement tests (GET and DELETE)
// =============================================================================

#[test]
fn get_row_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = get(ctx.addr(), "/v1/db/tables/anyns/rows/123?schema_id=10");
    assert_eq!(resp.status, 503, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "no_storage");
}

#[test]
fn delete_row_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = delete(ctx.addr(), "/v1/db/tables/anyns/rows/123");
    assert_eq!(resp.status, 503, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "no_storage");
}

// =============================================================================
// GET-by-hash parameter validation
// =============================================================================

#[test]
fn get_row_requires_schema_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/db/tables/anyns/rows/123");
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "missing_param");
}

// =============================================================================
// Additional write error paths
// =============================================================================

/// Writing without the required dedup column (col 1) fails.
#[test]
fn write_rejects_missing_dedup_column() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Only provide ts (col 2) and payload (col 20), skip dedup (col 1).
    let cols = vec![
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 100}),
        serde_json::json!({"column_id": 20, "type": "string", "value": "no dedup"}),
    ];
    let resp = write_row(ctx.addr(), "errns2", 10, &cols);
    assert_eq!(resp.status, 500, "missing dedup should fail: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "storage_error");
}

/// Writing without the required timestamp column (col 2) fails.
#[test]
fn write_rejects_missing_ts_column() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Only provide dedup (col 1) and payload (col 20), skip ts (col 2).
    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 20, "type": "string", "value": "no ts"}),
    ];
    let resp = write_row(ctx.addr(), "errns3", 10, &cols);
    assert_eq!(resp.status, 500, "missing ts should fail: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "storage_error");
}

/// Varbytes with odd-length hex string is rejected.
#[test]
fn write_rejects_odd_length_hex() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 100}),
        serde_json::json!({"column_id": 20, "type": "varbytes", "value": "abc"}),
    ];
    let resp = write_row(ctx.addr(), "hexns", 10, &cols);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_column");
}

/// Varbytes with invalid hex characters is rejected.
#[test]
fn write_rejects_invalid_hex_chars() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 100}),
        serde_json::json!({"column_id": 20, "type": "varbytes", "value": "ZZZZ"}),
    ];
    let resp = write_row(ctx.addr(), "hexns2", 10, &cols);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_column");
}

/// Scalar i64 with a string value is rejected.
#[test]
fn write_rejects_i64_type_mismatch() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": "not_a_number"}),
        serde_json::json!({"column_id": 20, "type": "string", "value": "x"}),
    ];
    let resp = write_row(ctx.addr(), "errns4", 10, &cols);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_column");
}

/// Scalar f64 with a string value is rejected.
#[test]
fn write_rejects_f64_type_mismatch() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 100}),
        serde_json::json!({"column_id": 5, "type": "scalar_f64", "value": "pi"}),
    ];
    let resp = write_row(ctx.addr(), "errns5", 10, &cols);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_column");
}

/// String column with non-string value is rejected.
#[test]
fn write_rejects_string_type_mismatch() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 100}),
        serde_json::json!({"column_id": 20, "type": "string", "value": 12345}),
    ];
    let resp = write_row(ctx.addr(), "errns6", 10, &cols);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_column");
}

/// Varbytes with non-string value is rejected.
#[test]
fn write_rejects_varbytes_type_mismatch() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 100}),
        serde_json::json!({"column_id": 20, "type": "varbytes", "value": 12345}),
    ];
    let resp = write_row(ctx.addr(), "errns7", 10, &cols);
    assert_eq!(resp.status, 400, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_column");
}

// =============================================================================
// payload_column_id
// =============================================================================

/// Scan with a non-default payload_column_id returns that column's data.
#[test]
fn scan_with_custom_payload_column_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Write with payload in column 30 instead of the default 20.
    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 100}),
        serde_json::json!({"column_id": 20, "type": "string", "value": "col-20-data"}),
        serde_json::json!({"column_id": 30, "type": "string", "value": "col-30-data"}),
    ];
    let resp = write_row(ctx.addr(), "paycolns", 10, &cols);
    assert_eq!(resp.status, 200, "write: {}", resp.body);

    // Default scan (payload_column_id=20).
    let resp = scan_rows(ctx.addr(), "paycolns", 10);
    assert_eq!(resp.status, 200, "scan default: {}", resp.body);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows[0]["payload"], "col-20-data");

    // Scan with payload_column_id=30.
    let resp = get(
        ctx.addr(),
        "/v1/db/tables/paycolns/rows?schema_id=10&payload_column_id=30",
    );
    assert_eq!(resp.status, 200, "scan col30: {}", resp.body);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows[0]["payload"], "col-30-data");
}

// =============================================================================
// delete_schema_id scan filtering
// =============================================================================

/// Scan with delete_schema_id parameter is accepted and passed to the engine.
///
/// Note: tombstone filtering via scan requires the tombstone to be visible
/// in the same reader state. Across separate HTTP request handles this is
/// unreliable (the GET-by-hash path uses the persisted policy and works
/// reliably — see delete_row_hides_from_get). This test verifies the param
/// is plumbed through without error rather than asserting filtering behavior.
#[test]
fn scan_accepts_delete_schema_id_param() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = standard_columns(1, "test");
    let resp = write_row(ctx.addr(), "dsparam", 10, &cols);
    assert_eq!(resp.status, 200, "write: {}", resp.body);

    let resp = get(
        ctx.addr(),
        "/v1/db/tables/dsparam/rows?schema_id=10&delete_schema_id=2",
    );
    assert_eq!(resp.status, 200, "scan with delete_schema_id: {}", resp.body);
    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 1, "row should still be present (no tombstone written)");
}

// =============================================================================
// tombstone_schema_id DELETE query parameter
// =============================================================================

/// DELETE with tombstone_schema_id query param works for first-open namespaces.
#[test]
fn delete_with_tombstone_schema_id_query_param() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Write with explicit policy including tombstone_schema_id.
    let policy = serde_json::json!({
        "active_schema_ids": [10],
        "tombstone_schema_id": 2,
        "dedup_column_id": 1,
        "ts_column_id": 2,
    });
    let cols = standard_columns(88, "to-delete");
    let resp = write_row_with_policy(ctx.addr(), "tsqpns", 10, &cols, policy);
    assert_eq!(resp.status, 200, "write: {}", resp.body);

    // Discover hash.
    let resp = scan_rows(ctx.addr(), "tsqpns", 10);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    let hash = rows[0]["scalars"].as_array().expect("scalars")
        .iter().find(|s| s["column_id"] == 1).expect("col 1")["value"]
        .as_u64().expect("hash");

    // Delete using tombstone_schema_id query param.
    let resp = delete(
        ctx.addr(),
        &format!("/v1/db/tables/tsqpns/rows/{hash}?tombstone_schema_id=2"),
    );
    assert_eq!(resp.status, 200, "delete: {}", resp.body);
    assert_eq!(resp.json()["status"], "deleted");

    // GET confirms tombstone applied.
    let resp = get(
        ctx.addr(),
        &format!("/v1/db/tables/tsqpns/rows/{hash}?schema_id=10"),
    );
    assert_eq!(resp.status, 200, "get after delete: {}", resp.body);
    assert!(resp.json()["row"].is_null(), "row should be null after delete");
}

// =============================================================================
// Path parsing errors
// =============================================================================

/// POST to invalid path (no namespace) returns 400.
#[test]
fn write_rejects_invalid_path() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({"schema_id": 10, "columns": []});
    // Path missing namespace segment: /v1/db/tables//rows
    let resp = post_json(ctx.addr(), "/v1/db/tables//rows", &body);
    assert_eq!(resp.status, 404, "empty namespace: {}", resp.body);
}

/// GET row with non-numeric hash returns 404 (path doesn't match route).
#[test]
fn get_row_rejects_non_numeric_hash() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/db/tables/anyns/rows/not_a_number?schema_id=10");
    // The router won't match this to the rows/{hash} handler (hash must parse as u64),
    // so it falls through to 404.
    assert_eq!(resp.status, 404, "non-numeric hash: {}", resp.body);
}

// =============================================================================
// POST /v1/db/tables/{ns}/rows/scan — advanced scan
// =============================================================================

/// POST scan returns same results as GET scan for basic case.
#[test]
fn post_scan_basic() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = standard_columns(42, "basic scan");
    let resp = write_row(ctx.addr(), "postscan", 10, &cols);
    assert_eq!(resp.status, 200, "write: {}", resp.body);

    let body = serde_json::json!({"schema_id": 10});
    let resp = post_json(ctx.addr(), "/v1/db/tables/postscan/rows/scan", &body);
    assert_eq!(resp.status, 200, "post scan: {}", resp.body);

    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["payload"], "basic scan");
    assert_eq!(json["has_more"], false);
}

/// POST scan with eq filter returns only matching rows.
#[test]
fn post_scan_with_filter_eq() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = columns_with_ts(100, 1000, "row-100");
    write_row(ctx.addr(), "filtns", 10, &cols);
    let cols = columns_with_ts(200, 2000, "row-200");
    write_row(ctx.addr(), "filtns", 10, &cols);

    // Filter: dedup_column (col 1) == 100.
    let body = serde_json::json!({
        "schema_id": 10,
        "filters": [{"column_id": 1, "op": "eq", "value": 100}]
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/filtns/rows/scan", &body);
    assert_eq!(resp.status, 200, "post scan filter: {}", resp.body);

    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 1, "eq filter should return one row");
    assert_eq!(rows[0]["payload"], "row-100");
}

/// POST scan with range filter (ge/le) returns rows in range.
#[test]
fn post_scan_with_filter_range() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    for (key, ts) in [(10, 1000), (20, 2000), (30, 3000)] {
        let cols = columns_with_ts(key, ts, &format!("row-{key}"));
        write_row(ctx.addr(), "rangns", 10, &cols);
    }

    let body = serde_json::json!({
        "schema_id": 10,
        "filters": [
            {"column_id": 2, "op": "ge", "value": 2000_u64},
            {"column_id": 2, "op": "le", "value": 3000_u64}
        ]
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/rangns/rows/scan", &body);
    assert_eq!(resp.status, 200, "post scan range: {}", resp.body);

    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 2, "ge+le filter should return 2 rows");
    assert_eq!(rows[0]["payload"], "row-30");
    assert_eq!(rows[1]["payload"], "row-20");
}

/// POST scan with extra_columns returns additional scalars.
#[test]
fn post_scan_with_extra_columns() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 100}),
        serde_json::json!({"column_id": 5, "type": "scalar_u64", "value": 999}),
        serde_json::json!({"column_id": 20, "type": "string", "value": "extra cols test"}),
    ];
    let resp = write_row(ctx.addr(), "extrans", 10, &cols);
    assert_eq!(resp.status, 200, "write: {}", resp.body);

    // Default scan — col 5 should NOT appear in scalars.
    let resp = scan_rows(ctx.addr(), "extrans", 10);
    assert_eq!(resp.status, 200, "default scan: {}", resp.body);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    let has_col5 = rows[0]["scalars"].as_array().expect("scalars")
        .iter().any(|s| s["column_id"] == 5);
    assert!(!has_col5, "col 5 should not appear in default scan");

    // POST scan with extra_columns=[5] — col 5 SHOULD appear.
    let body = serde_json::json!({
        "schema_id": 10,
        "extra_columns": [5]
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/extrans/rows/scan", &body);
    assert_eq!(resp.status, 200, "post scan extra: {}", resp.body);

    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 1);
    let scalars = rows[0]["scalars"].as_array().expect("scalars");
    let col5 = scalars.iter().find(|s| s["column_id"] == 5);
    assert!(col5.is_some(), "col 5 should appear with extra_columns");
    assert_eq!(col5.unwrap()["value"], 999);
}

/// POST scan with reverse=false returns oldest first.
#[test]
fn post_scan_reverse_false() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = columns_with_ts(1, 1000, "old");
    write_row(ctx.addr(), "revns", 10, &cols);
    let cols = columns_with_ts(2, 2000, "new");
    write_row(ctx.addr(), "revns", 10, &cols);

    let body = serde_json::json!({
        "schema_id": 10,
        "reverse": false
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/revns/rows/scan", &body);
    assert_eq!(resp.status, 200, "post scan reverse=false: {}", resp.body);

    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["payload"], "old", "first should be oldest with reverse=false");
    assert_eq!(rows[1]["payload"], "new", "second should be newest with reverse=false");
}

/// POST scan with dedup=false returns all row versions.
#[test]
fn post_scan_dedup_false() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = columns_with_ts(100, 1000, "v1");
    write_row(ctx.addr(), "nodedns", 10, &cols);
    let cols = columns_with_ts(100, 2000, "v2");
    write_row(ctx.addr(), "nodedns", 10, &cols);

    // Default scan (dedup=true) — should return 1 row.
    let body = serde_json::json!({"schema_id": 10});
    let resp = post_json(ctx.addr(), "/v1/db/tables/nodedns/rows/scan", &body);
    assert_eq!(resp.status, 200, "dedup scan: {}", resp.body);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 1, "dedup should collapse to 1");

    // POST scan with dedup=false — should return both versions.
    let body = serde_json::json!({
        "schema_id": 10,
        "dedup": false
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/nodedns/rows/scan", &body);
    assert_eq!(resp.status, 200, "no-dedup scan: {}", resp.body);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 2, "dedup=false should return both versions");
    let payloads: Vec<&str> = rows.iter().map(|r| r["payload"].as_str().unwrap()).collect();
    assert_eq!(payloads[0], "v2", "newest version first");
    assert_eq!(payloads[1], "v1", "oldest version second");
}

/// POST scan with additional_schema_ids returns rows from multiple schemas.
#[test]
fn post_scan_additional_schema_ids() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let policy = serde_json::json!({
        "active_schema_ids": [10, 11],
        "dedup_column_id": 1,
        "ts_column_id": 2,
    });
    let cols = columns_with_ts(1, 1000, "schema-10");
    write_row_with_policy(ctx.addr(), "multischns", 10, &cols, policy);
    let cols = columns_with_ts(2, 2000, "schema-11");
    write_row(ctx.addr(), "multischns", 11, &cols);

    // Single schema scan — only schema 10.
    let body = serde_json::json!({"schema_id": 10});
    let resp = post_json(ctx.addr(), "/v1/db/tables/multischns/rows/scan", &body);
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["rows"].as_array().expect("rows").len(), 1);

    // Multi-schema scan — schema 10 + 11.
    let body = serde_json::json!({
        "schema_id": 10,
        "additional_schema_ids": [11]
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/multischns/rows/scan", &body);
    assert_eq!(resp.status, 200, "multi-schema scan: {}", resp.body);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 2, "should see rows from both schemas");
    let mut payloads: Vec<&str> = rows.iter().map(|r| r["payload"].as_str().unwrap()).collect();
    payloads.sort();
    assert_eq!(payloads, vec!["schema-10", "schema-11"], "should contain both schema rows");
}

/// POST scan requires schema_id.
#[test]
fn post_scan_requires_schema_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = standard_columns(1, "x");
    write_row(ctx.addr(), "reqschns", 10, &cols);

    let body = serde_json::json!({});
    let resp = post_json(ctx.addr(), "/v1/db/tables/reqschns/rows/scan", &body);
    assert_eq!(resp.status, 400, "missing schema_id: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_json");
}

/// POST scan rejects invalid filter op.
#[test]
fn post_scan_invalid_filter_op() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = standard_columns(1, "x");
    write_row(ctx.addr(), "badopns", 10, &cols);

    let body = serde_json::json!({
        "schema_id": 10,
        "filters": [{"column_id": 1, "op": "like", "value": 1}]
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/badopns/rows/scan", &body);
    assert_eq!(resp.status, 400, "invalid op: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "invalid_filter");
}

// =============================================================================
// POST scan 503 without bucket
// =============================================================================

#[test]
fn post_scan_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let body = serde_json::json!({"schema_id": 10});
    let resp = post_json(ctx.addr(), "/v1/db/tables/anyns/rows/scan", &body);
    assert_eq!(resp.status, 503, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "no_storage");
}

// =============================================================================
// GET /v1/db/tables/_meta/namespaces — list namespaces
// =============================================================================

/// List namespaces returns tables that have been written to.
#[test]
fn list_namespaces() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = standard_columns(1, "a");
    write_row(ctx.addr(), "alpha", 10, &cols);
    let cols = standard_columns(1, "b");
    write_row(ctx.addr(), "beta", 10, &cols);

    let resp = get(ctx.addr(), "/v1/db/tables/_meta/namespaces");
    assert_eq!(resp.status, 200, "list: {}", resp.body);

    let json = resp.json();
    let ns = json["namespaces"].as_array().expect("namespaces array");
    let names: Vec<&str> = ns.iter().map(|v| v.as_str().unwrap()).collect();
    assert!(names.contains(&"alpha"), "should contain alpha");
    assert!(names.contains(&"beta"), "should contain beta");
    // Should be sorted.
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted, "namespaces should be sorted");
}

/// List namespaces before user writes returns only server-internal namespaces.
#[test]
fn list_namespaces_before_user_writes() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/db/tables/_meta/namespaces");
    assert_eq!(resp.status, 200, "list: {}", resp.body);

    let json = resp.json();
    let ns = json["namespaces"].as_array().expect("namespaces array");
    let names: Vec<&str> = ns.iter().map(|v| v.as_str().unwrap()).collect();
    assert!(
        !names.contains(&"my_custom_table"),
        "user namespaces should not exist before writing"
    );
}

#[test]
fn list_namespaces_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = get(ctx.addr(), "/v1/db/tables/_meta/namespaces");
    assert_eq!(resp.status, 503, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "no_storage");
}

// =============================================================================
// GET /v1/db/tables/{ns}/_meta/policy — read policy
// =============================================================================

/// Read policy returns persisted policy after a write.
#[test]
fn get_policy() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let policy = serde_json::json!({
        "active_schema_ids": [10],
        "tombstone_schema_id": 2,
        "dedup_column_id": 1,
        "ts_column_id": 2,
    });
    let cols = standard_columns(1, "x");
    write_row_with_policy(ctx.addr(), "polns", 10, &cols, policy);

    let resp = get(ctx.addr(), "/v1/db/tables/polns/_meta/policy");
    assert_eq!(resp.status, 200, "get policy: {}", resp.body);

    let json = resp.json();
    assert!(json["policy"].is_object(), "should have policy object");
    let pol = &json["policy"];
    assert_eq!(pol["dedup_column_id"], 1);
    assert_eq!(pol["ts_column_id"], 2);
    assert_eq!(pol["tombstone_schema_id"], 2);
    let ids = pol["active_schema_ids"].as_array().expect("active_schema_ids");
    assert!(ids.contains(&serde_json::json!(10)), "active_schema_ids should include 10");
}

/// Read policy returns 404 for nonexistent namespace.
#[test]
fn get_policy_not_found() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/db/tables/noexist/_meta/policy");
    assert_eq!(resp.status, 404, "not found: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "not_found");
}

#[test]
fn get_policy_returns_503_without_bucket() {
    let ctx = ServerTestContext::new(no_bucket_config());
    let resp = get(ctx.addr(), "/v1/db/tables/anyns/_meta/policy");
    assert_eq!(resp.status, 503, "body: {}", resp.body);
    assert_eq!(resp.json()["error"]["code"], "no_storage");
}

// =============================================================================
// TTL column
// =============================================================================

/// POST scan with ttl_column_id filters out expired rows.
#[test]
fn post_scan_ttl_filters_expired_rows() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    // Row with TTL=1ms — definitely expired.
    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 100}),
        serde_json::json!({"column_id": 3, "type": "scalar_i64", "value": 1}),
        serde_json::json!({"column_id": 20, "type": "string", "value": "expired"}),
    ];
    write_row(ctx.addr(), "ttlns", 10, &cols);

    // Row with TTL far in the future (year ~2100).
    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 2}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 200}),
        serde_json::json!({"column_id": 3, "type": "scalar_i64", "value": 4102444800000_i64}),
        serde_json::json!({"column_id": 20, "type": "string", "value": "alive"}),
    ];
    write_row(ctx.addr(), "ttlns", 10, &cols);

    // Without ttl_column_id — both rows visible.
    let body = serde_json::json!({"schema_id": 10});
    let resp = post_json(ctx.addr(), "/v1/db/tables/ttlns/rows/scan", &body);
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["rows"].as_array().expect("rows").len(), 2);

    // With ttl_column_id=3 — expired row filtered out.
    let body = serde_json::json!({"schema_id": 10, "ttl_column_id": 3});
    let resp = post_json(ctx.addr(), "/v1/db/tables/ttlns/rows/scan", &body);
    assert_eq!(resp.status, 200, "scan with ttl: {}", resp.body);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 1, "expired row should be filtered out");
    assert_eq!(rows[0]["payload"], "alive");
}

// =============================================================================
// ts_column_id override
// =============================================================================

/// POST scan with non-default ts_column_id includes that column in scalars.
///
/// Note: ts_column_id in ScanParams does NOT change result ordering —
/// ordering comes from disk layout (set by CompactionPolicy.ts_column_id
/// at write time). ScanParams.ts_column_id controls which column appears
/// in result scalars and which column cursor_ts filters on.
#[test]
fn post_scan_custom_ts_column_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 1000}),
        serde_json::json!({"column_id": 4, "type": "scalar_i64", "value": 9000}),
        serde_json::json!({"column_id": 20, "type": "string", "value": "row-A"}),
    ];
    write_row(ctx.addr(), "tscolns", 10, &cols);

    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 2}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 2000}),
        serde_json::json!({"column_id": 4, "type": "scalar_i64", "value": 8000}),
        serde_json::json!({"column_id": 20, "type": "string", "value": "row-B"}),
    ];
    write_row(ctx.addr(), "tscolns", 10, &cols);

    // Default scan (ts_column_id=2) — col 2 in scalars, not col 4.
    let body = serde_json::json!({"schema_id": 10});
    let resp = post_json(ctx.addr(), "/v1/db/tables/tscolns/rows/scan", &body);
    assert_eq!(resp.status, 200);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    let scalars = rows[0]["scalars"].as_array().expect("scalars");
    assert!(scalars.iter().any(|s| s["column_id"] == 2), "default: col 2 in scalars");
    assert!(!scalars.iter().any(|s| s["column_id"] == 4), "default: col 4 NOT in scalars");

    // Custom ts_column_id=4 — col 4 should appear in scalars.
    let body = serde_json::json!({"schema_id": 10, "ts_column_id": 4});
    let resp = post_json(ctx.addr(), "/v1/db/tables/tscolns/rows/scan", &body);
    assert_eq!(resp.status, 200);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    let scalars = rows[0]["scalars"].as_array().expect("scalars");
    assert!(scalars.iter().any(|s| s["column_id"] == 4), "custom: col 4 in scalars");
    let col4 = scalars.iter().find(|s| s["column_id"] == 4).unwrap();
    // First row is row-B (reverse=true, ordered by policy ts_column=2, B has ts=2000>1000).
    assert_eq!(col4["value"], 8000_u64, "row-B should have col 4 = 8000");
}

// =============================================================================
// legacy_hash GET parameter
// =============================================================================

/// GET by-hash with legacy_hash fallback finds a row by alternate hash.
#[test]
fn get_row_with_legacy_hash() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = standard_columns(42, "legacy test");
    write_row(ctx.addr(), "legns", 10, &cols);

    let resp = scan_rows(ctx.addr(), "legns", 10);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    let hash = rows[0]["scalars"].as_array().expect("scalars")
        .iter().find(|s| s["column_id"] == 1).expect("col 1")["value"]
        .as_u64().expect("hash");

    // Wrong primary hash but correct legacy_hash → should find the row.
    let resp = get(
        ctx.addr(),
        &format!("/v1/db/tables/legns/rows/99999?schema_id=10&legacy_hash={hash}"),
    );
    assert_eq!(resp.status, 200, "legacy get: {}", resp.body);
    assert!(resp.json()["row"].is_object(), "legacy_hash should find the row");
    assert_eq!(resp.json()["row"]["payload"], "legacy test");

    // Wrong primary hash and no legacy_hash → should NOT find.
    let resp = get(ctx.addr(), "/v1/db/tables/legns/rows/99999?schema_id=10");
    assert_eq!(resp.status, 200);
    assert!(resp.json()["row"].is_null(), "without legacy_hash, should not find");
}

// =============================================================================
// POST scan cursor pagination
// =============================================================================

/// POST scan with cursor_ts/cursor_hash paginates correctly.
#[test]
fn post_scan_cursor_pagination() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    for (key, ts) in [(1, 1000), (2, 2000), (3, 3000)] {
        let cols = columns_with_ts(key, ts, &format!("row-{key}"));
        write_row(ctx.addr(), "postcurns", 10, &cols);
    }

    // Page 1: limit=2 via POST body.
    let body = serde_json::json!({"schema_id": 10, "limit": 2});
    let resp = post_json(ctx.addr(), "/v1/db/tables/postcurns/rows/scan", &body);
    assert_eq!(resp.status, 200, "page 1: {}", resp.body);
    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 2);
    assert_eq!(json["has_more"], true);
    assert_eq!(rows[0]["payload"], "row-3");
    assert_eq!(rows[1]["payload"], "row-2");

    // Extract cursor from last row.
    let last = &rows[1];
    let scalars = last["scalars"].as_array().expect("scalars");
    let cursor_ts = scalars.iter().find(|s| s["column_id"] == 2).expect("ts")["value"]
        .as_u64().expect("ts val");
    let cursor_hash = scalars.iter().find(|s| s["column_id"] == 1).expect("pk")["value"]
        .as_u64().expect("pk val");

    // Page 2: cursor via POST body.
    let body = serde_json::json!({
        "schema_id": 10,
        "limit": 2,
        "cursor_ts": cursor_ts,
        "cursor_hash": cursor_hash
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/postcurns/rows/scan", &body);
    assert_eq!(resp.status, 200, "page 2: {}", resp.body);
    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 1);
    assert_eq!(json["has_more"], false);
    assert_eq!(rows[0]["payload"], "row-1");
}

// =============================================================================
// Custom dedup_column_id
// =============================================================================

/// POST scan with non-default dedup_column_id deduplicates on that column.
#[test]
fn post_scan_custom_dedup_column_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let policy = serde_json::json!({
        "active_schema_ids": [10],
        "dedup_column_id": 3,
        "ts_column_id": 2,
    });

    // Two rows with same col 3 (dedup key) but different col 1 and timestamps.
    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 1000}),
        serde_json::json!({"column_id": 3, "type": "scalar_u64", "value": 999}),
        serde_json::json!({"column_id": 20, "type": "string", "value": "old-version"}),
    ];
    write_row_with_policy(ctx.addr(), "dedcol3ns", 10, &cols, policy);

    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 2}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 2000}),
        serde_json::json!({"column_id": 3, "type": "scalar_u64", "value": 999}),
        serde_json::json!({"column_id": 20, "type": "string", "value": "new-version"}),
    ];
    write_row(ctx.addr(), "dedcol3ns", 10, &cols);

    // Dedup on col 3 → should collapse to 1 row (latest).
    let body = serde_json::json!({"schema_id": 10, "dedup_column_id": 3});
    let resp = post_json(ctx.addr(), "/v1/db/tables/dedcol3ns/rows/scan", &body);
    assert_eq!(resp.status, 200);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 1, "dedup on col 3 should collapse to 1 row");
    assert_eq!(rows[0]["payload"], "new-version");

    // Dedup on col 1 (each row has distinct col 1) → should see both.
    let body = serde_json::json!({"schema_id": 10, "dedup_column_id": 1});
    let resp = post_json(ctx.addr(), "/v1/db/tables/dedcol3ns/rows/scan", &body);
    assert_eq!(resp.status, 200);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 2, "dedup on col 1 should see both rows");
    let mut payloads: Vec<&str> = rows.iter().map(|r| r["payload"].as_str().unwrap()).collect();
    payloads.sort();
    assert_eq!(payloads, vec!["new-version", "old-version"]);
}

// =============================================================================
// Filter ops: ne, lt, gt
// =============================================================================

/// POST scan with ne (not-equal) filter excludes matching rows.
#[test]
fn post_scan_filter_ne() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    for (key, ts) in [(10, 1000), (20, 2000), (30, 3000)] {
        let cols = columns_with_ts(key, ts, &format!("key-{key}"));
        write_row(ctx.addr(), "nens", 10, &cols);
    }

    let body = serde_json::json!({
        "schema_id": 10,
        "filters": [{"column_id": 1, "op": "ne", "value": 20}]
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/nens/rows/scan", &body);
    assert_eq!(resp.status, 200);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 2, "ne should exclude one row");
    let payloads: Vec<&str> = rows.iter().map(|r| r["payload"].as_str().unwrap()).collect();
    assert!(!payloads.contains(&"key-20"), "key-20 should be excluded");
    assert!(payloads.contains(&"key-10"));
    assert!(payloads.contains(&"key-30"));
}

/// POST scan with lt (less-than) filter.
#[test]
fn post_scan_filter_lt() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    for (key, ts) in [(10, 1000), (20, 2000), (30, 3000)] {
        let cols = columns_with_ts(key, ts, &format!("key-{key}"));
        write_row(ctx.addr(), "ltns", 10, &cols);
    }

    let body = serde_json::json!({
        "schema_id": 10,
        "filters": [{"column_id": 1, "op": "lt", "value": 20}]
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/ltns/rows/scan", &body);
    assert_eq!(resp.status, 200);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 1, "lt 20 should return 1 row");
    assert_eq!(rows[0]["payload"], "key-10");
}

/// POST scan with gt (greater-than) filter.
#[test]
fn post_scan_filter_gt() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    for (key, ts) in [(10, 1000), (20, 2000), (30, 3000)] {
        let cols = columns_with_ts(key, ts, &format!("key-{key}"));
        write_row(ctx.addr(), "gtns", 10, &cols);
    }

    let body = serde_json::json!({
        "schema_id": 10,
        "filters": [{"column_id": 1, "op": "gt", "value": 20}]
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/gtns/rows/scan", &body);
    assert_eq!(resp.status, 200);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 1, "gt 20 should return 1 row");
    assert_eq!(rows[0]["payload"], "key-30");
}

// =============================================================================
// POST scan: payload_column_id and delete_schema_id via body
// =============================================================================

/// POST scan with custom payload_column_id returns that column's data.
#[test]
fn post_scan_with_custom_payload_column_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 100}),
        serde_json::json!({"column_id": 20, "type": "string", "value": "col-20"}),
        serde_json::json!({"column_id": 30, "type": "string", "value": "col-30"}),
    ];
    write_row(ctx.addr(), "postpayns", 10, &cols);

    let body = serde_json::json!({"schema_id": 10});
    let resp = post_json(ctx.addr(), "/v1/db/tables/postpayns/rows/scan", &body);
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["rows"].as_array().expect("rows").clone()[0]["payload"], "col-20");

    let body = serde_json::json!({"schema_id": 10, "payload_column_id": 30});
    let resp = post_json(ctx.addr(), "/v1/db/tables/postpayns/rows/scan", &body);
    assert_eq!(resp.status, 200);
    assert_eq!(resp.json()["rows"].as_array().expect("rows").clone()[0]["payload"], "col-30");
}

/// POST scan with delete_schema_id passes the parameter through.
#[test]
fn post_scan_with_delete_schema_id() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = standard_columns(1, "test");
    write_row(ctx.addr(), "postdelns", 10, &cols);

    let body = serde_json::json!({"schema_id": 10, "delete_schema_id": 2});
    let resp = post_json(ctx.addr(), "/v1/db/tables/postdelns/rows/scan", &body);
    assert_eq!(resp.status, 200);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 1, "row should still be visible (no tombstone)");
}

// =============================================================================
// Path parsing for new endpoints
// =============================================================================

/// POST to /rows/scan with missing namespace returns 404.
#[test]
fn post_scan_rejects_empty_namespace() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({"schema_id": 10});
    let resp = post_json(ctx.addr(), "/v1/db/tables//rows/scan", &body);
    assert_eq!(resp.status, 404, "empty ns: {}", resp.body);
}

/// POST to /rows/scan with extra trailing segment returns 404.
#[test]
fn post_scan_rejects_extra_path_segments() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({"schema_id": 10});
    let resp = post_json(ctx.addr(), "/v1/db/tables/ns/rows/scan/extra", &body);
    assert_eq!(resp.status, 404, "extra segment: {}", resp.body);
}

/// GET policy with extra path segment returns 404.
#[test]
fn get_policy_rejects_extra_path_segments() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/db/tables/ns/_meta/policy/extra");
    assert_eq!(resp.status, 404, "extra segment: {}", resp.body);
}

/// _meta namespace doesn't match document routes.
#[test]
fn meta_namespace_not_intercepted_by_docs() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = get(ctx.addr(), "/v1/db/tables/_meta/namespaces");
    assert_eq!(resp.status, 200, "should be handled by meta endpoint: {}", resp.body);
    assert!(resp.json()["namespaces"].is_array(), "should have namespaces array");
}

// =============================================================================
// Combinatorial / interaction tests
// =============================================================================

/// POST scan combining multiple features: filters + extra_columns + reverse=false + limit.
#[test]
fn post_scan_combined_features() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    for (key, ts, col5_val) in [(1, 1000, 100), (2, 2000, 200), (3, 3000, 300), (4, 4000, 400)] {
        let cols = vec![
            serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": key}),
            serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": ts}),
            serde_json::json!({"column_id": 5, "type": "scalar_u64", "value": col5_val}),
            serde_json::json!({"column_id": 20, "type": "string", "value": format!("row-{key}")}),
        ];
        write_row(ctx.addr(), "combns", 10, &cols);
    }

    // Combine: filter col 1 >= 2 AND col 1 <= 3, extra_columns=[5], reverse=false, limit=10.
    let body = serde_json::json!({
        "schema_id": 10,
        "filters": [
            {"column_id": 1, "op": "ge", "value": 2},
            {"column_id": 1, "op": "le", "value": 3}
        ],
        "extra_columns": [5],
        "reverse": false,
        "limit": 10
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/combns/rows/scan", &body);
    assert_eq!(resp.status, 200, "combined scan: {}", resp.body);

    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 2, "filter should select keys 2 and 3");

    // reverse=false → oldest first.
    assert_eq!(rows[0]["payload"], "row-2", "oldest first with reverse=false");
    assert_eq!(rows[1]["payload"], "row-3", "newest second with reverse=false");

    // extra_columns=[5] → col 5 should appear in scalars with correct values.
    let col5_row2 = rows[0]["scalars"].as_array().unwrap()
        .iter().find(|s| s["column_id"] == 5).expect("col 5 in row-2");
    assert_eq!(col5_row2["value"], 200);
    let col5_row3 = rows[1]["scalars"].as_array().unwrap()
        .iter().find(|s| s["column_id"] == 5).expect("col 5 in row-3");
    assert_eq!(col5_row3["value"], 300);
}

/// POST scan combining filters on different columns.
#[test]
fn post_scan_multi_column_filter() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    for (key, ts, col5_val, payload) in [
        (1, 1000, 10, "a"),
        (2, 2000, 20, "b"),
        (3, 3000, 10, "c"),
        (4, 4000, 30, "d"),
    ] {
        let cols = vec![
            serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": key}),
            serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": ts}),
            serde_json::json!({"column_id": 5, "type": "scalar_u64", "value": col5_val}),
            serde_json::json!({"column_id": 20, "type": "string", "value": payload}),
        ];
        write_row(ctx.addr(), "mcfns", 10, &cols);
    }

    // Filter: col 1 > 1 AND col 5 == 10 → only key=3 (col1=3>1, col5=10).
    let body = serde_json::json!({
        "schema_id": 10,
        "filters": [
            {"column_id": 1, "op": "gt", "value": 1},
            {"column_id": 5, "op": "eq", "value": 10}
        ]
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/mcfns/rows/scan", &body);
    assert_eq!(resp.status, 200);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 1, "only key=3 matches both filters");
    assert_eq!(rows[0]["payload"], "c");
}

// =============================================================================
// Edge cases
// =============================================================================

/// Writing and scanning a row with empty payload.
#[test]
fn write_and_scan_empty_payload() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = vec![
        serde_json::json!({"column_id": 1, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 2, "type": "scalar_i64", "value": 100}),
        serde_json::json!({"column_id": 20, "type": "string", "value": ""}),
    ];
    write_row(ctx.addr(), "emptyns", 10, &cols);

    let resp = scan_rows(ctx.addr(), "emptyns", 10);
    assert_eq!(resp.status, 200);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["payload"], "", "empty payload should roundtrip");
    assert_eq!(rows[0]["payload_encoding"], "utf8");
}

/// DELETE with invalid path (missing hash) returns 404.
#[test]
fn delete_rejects_invalid_path() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = delete(ctx.addr(), "/v1/db/tables/ns/rows");
    assert_eq!(resp.status, 404, "missing hash: {}", resp.body);
}

/// DELETE with non-numeric hash returns 404.
#[test]
fn delete_rejects_non_numeric_hash() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = delete(ctx.addr(), "/v1/db/tables/ns/rows/abc");
    assert_eq!(resp.status, 404, "non-numeric hash: {}", resp.body);
}

/// POST scan with explicit empty arrays is valid and returns results.
#[test]
fn post_scan_with_explicit_empty_arrays() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let cols = standard_columns(1, "data");
    write_row(ctx.addr(), "emptyarrns", 10, &cols);

    let body = serde_json::json!({
        "schema_id": 10,
        "filters": [],
        "extra_columns": [],
        "additional_schema_ids": []
    });
    let resp = post_json(ctx.addr(), "/v1/db/tables/emptyarrns/rows/scan", &body);
    assert_eq!(resp.status, 200);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["payload"], "data");
}

/// POST scan with limit=0 returns all rows (unlimited).
#[test]
fn post_scan_limit_zero_returns_all() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    for key in 1..=5 {
        let cols = columns_with_ts(key, key as i64 * 1000, &format!("row-{key}"));
        write_row(ctx.addr(), "limzns", 10, &cols);
    }

    let body = serde_json::json!({"schema_id": 10, "limit": 0});
    let resp = post_json(ctx.addr(), "/v1/db/tables/limzns/rows/scan", &body);
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 5, "limit=0 should return all 5 rows");
    assert_eq!(json["has_more"], false);
}

/// POST scan with limit=1 returns exactly one row and has_more=true.
#[test]
fn post_scan_limit_one() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    for key in 1..=3 {
        let cols = columns_with_ts(key, key as i64 * 1000, &format!("row-{key}"));
        write_row(ctx.addr(), "lim1ns", 10, &cols);
    }

    let body = serde_json::json!({"schema_id": 10, "limit": 1});
    let resp = post_json(ctx.addr(), "/v1/db/tables/lim1ns/rows/scan", &body);
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let rows = json["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 1, "limit=1 should return exactly 1 row");
    assert_eq!(json["has_more"], true);
    assert_eq!(rows[0]["payload"], "row-3", "newest first (reverse=true default)");
}

/// GET policy with ttl_column_id verifies all fields roundtrip.
#[test]
fn get_policy_with_ttl() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let policy = serde_json::json!({
        "active_schema_ids": [10, 11],
        "tombstone_schema_id": 2,
        "dedup_column_id": 3,
        "ts_column_id": 4,
        "ttl_column_id": 5,
    });
    let cols = vec![
        serde_json::json!({"column_id": 3, "type": "scalar_u64", "value": 1}),
        serde_json::json!({"column_id": 4, "type": "scalar_i64", "value": 100}),
        serde_json::json!({"column_id": 20, "type": "string", "value": "x"}),
    ];
    write_row_with_policy(ctx.addr(), "polttlns", 10, &cols, policy);

    let resp = get(ctx.addr(), "/v1/db/tables/polttlns/_meta/policy");
    assert_eq!(resp.status, 200);

    let pol = &resp.json()["policy"];
    assert_eq!(pol["dedup_column_id"], 3);
    assert_eq!(pol["ts_column_id"], 4);
    assert_eq!(pol["tombstone_schema_id"], 2);
    let ids = pol["active_schema_ids"].as_array().expect("active_schema_ids");
    assert!(ids.contains(&serde_json::json!(10)));
    assert!(ids.contains(&serde_json::json!(11)));
}

/// Scan returns correct content-type header.
#[test]
fn scan_returns_json_content_type() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let resp = scan_rows(ctx.addr(), "ctns", 10);
    assert_eq!(resp.status, 200);
    let ct = resp.header("content-type").expect("content-type header");
    assert_eq!(ct, "application/json");
}

/// POST scan returns correct content-type header.
#[test]
fn post_scan_returns_json_content_type() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let body = serde_json::json!({"schema_id": 10});
    let resp = post_json(ctx.addr(), "/v1/db/tables/ctpostns/rows/scan", &body);
    assert_eq!(resp.status, 200);
    let ct = resp.header("content-type").expect("content-type header");
    assert_eq!(ct, "application/json");
}

/// Write + delete + scan confirms tombstone hides row from scan when
/// delete_schema_id is provided.
#[test]
fn delete_row_hides_from_scan() {
    let temp = TempDir::new().expect("temp dir");
    let ctx = ServerTestContext::new(documents_config(temp.path()));

    let policy = serde_json::json!({
        "active_schema_ids": [10],
        "tombstone_schema_id": 2,
        "dedup_column_id": 1,
        "ts_column_id": 2,
    });
    let cols = columns_with_ts(1, 1000, "keep");
    write_row_with_policy(ctx.addr(), "delscanns", 10, &cols, policy);
    let cols = columns_with_ts(2, 2000, "remove");
    write_row(ctx.addr(), "delscanns", 10, &cols);

    // Discover hash of the row to delete.
    let resp = scan_rows(ctx.addr(), "delscanns", 10);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 2);
    let remove_row = rows.iter().find(|r| r["payload"] == "remove").expect("find remove");
    let hash = remove_row["scalars"].as_array().expect("scalars")
        .iter().find(|s| s["column_id"] == 1).expect("col 1")["value"]
        .as_u64().expect("hash");

    // Delete the "remove" row.
    let resp = delete(ctx.addr(), &format!("/v1/db/tables/delscanns/rows/{hash}"));
    assert_eq!(resp.status, 200, "delete: {}", resp.body);

    // Scan with delete_schema_id to activate tombstone filtering.
    // Unlike GET-by-hash (which uses persisted policy), scan requires explicit
    // delete_schema_id to filter tombstoned rows.
    let body = serde_json::json!({"schema_id": 10, "delete_schema_id": 2});
    let resp = post_json(ctx.addr(), "/v1/db/tables/delscanns/rows/scan", &body);
    assert_eq!(resp.status, 200, "scan after delete: {}", resp.body);
    let rows = resp.json()["rows"].as_array().expect("rows").clone();
    assert_eq!(rows.len(), 1, "deleted row should be hidden from scan");
    assert_eq!(rows[0]["payload"], "keep");
}
