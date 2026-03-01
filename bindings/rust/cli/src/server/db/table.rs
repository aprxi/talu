//! DB table HTTP handlers.
//!
//! Exposes Table operations under `/v1/db/tables/{ns}/rows`.

use std::path::PathBuf;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

use talu::table::{
    ColumnFilter, ColumnShape, ColumnValue, CompactionPolicy, FilterOp, PhysicalType, ScanParams,
    TableError, TableHandle,
};

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;

const PATH_PREFIX: &str = "/v1/db/tables/";

// =========================================================================
// Request/Response types
// =========================================================================

/// JSON column value for write requests.
#[derive(Debug, Deserialize, ToSchema)]
pub struct JsonColumnValue {
    pub column_id: u32,
    /// "scalar_u64", "scalar_i64", "scalar_f64", "varbytes", "json", or "string".
    pub r#type: String,
    /// For scalars: numeric value. For varbytes: hex-encoded string.
    /// For json: any JSON value (serialized to string and stored as bytes).
    /// For string: a plain string (stored as UTF-8 bytes).
    pub value: serde_json::Value,
    #[serde(default = "default_dims")]
    pub dims: u16,
}

fn default_dims() -> u16 {
    1
}

/// JSON request body for POST (write row).
#[derive(Debug, Deserialize, ToSchema)]
pub struct WriteRowRequest {
    pub schema_id: u16,
    pub columns: Vec<JsonColumnValue>,
    /// Optional compaction policy. Persisted on first write; subsequent writes
    /// load the persisted policy (this field is ignored).
    #[serde(default)]
    pub policy: Option<PolicyRequest>,
}

/// JSON request body for opening a table (policy).
#[derive(Debug, Deserialize, ToSchema)]
pub struct PolicyRequest {
    pub active_schema_ids: Vec<u16>,
    #[serde(default)]
    pub tombstone_schema_id: Option<u16>,
    #[serde(default = "default_dedup_col")]
    pub dedup_column_id: u32,
    #[serde(default = "default_ts_col")]
    pub ts_column_id: u32,
    #[serde(default)]
    pub ttl_column_id: Option<u32>,
}

fn default_dedup_col() -> u32 {
    1
}
fn default_ts_col() -> u32 {
    2
}

/// JSON request body for POST scan.
#[derive(Debug, Deserialize, ToSchema)]
pub struct ScanRequest {
    pub schema_id: u16,
    #[serde(default)]
    pub additional_schema_ids: Vec<u16>,
    #[serde(default)]
    pub filters: Vec<JsonColumnFilter>,
    #[serde(default)]
    pub extra_columns: Vec<u32>,
    #[serde(default = "default_dedup_col")]
    pub dedup_column_id: u32,
    #[serde(default = "default_dedup_true")]
    pub dedup: bool,
    #[serde(default)]
    pub delete_schema_id: Option<u16>,
    #[serde(default = "default_ts_col")]
    pub ts_column_id: u32,
    #[serde(default)]
    pub ttl_column_id: Option<u32>,
    #[serde(default)]
    pub limit: u32,
    #[serde(default)]
    pub cursor_ts: Option<i64>,
    #[serde(default)]
    pub cursor_hash: Option<u64>,
    #[serde(default = "default_payload_col")]
    pub payload_column_id: u32,
    #[serde(default = "default_reverse_true")]
    pub reverse: bool,
}

/// A scalar column filter in scan requests.
#[derive(Debug, Deserialize, ToSchema)]
pub struct JsonColumnFilter {
    pub column_id: u32,
    /// One of: "eq", "ne", "lt", "le", "gt", "ge".
    pub op: String,
    pub value: u64,
}

fn default_dedup_true() -> bool {
    true
}
fn default_payload_col() -> u32 {
    20
}
fn default_reverse_true() -> bool {
    true
}

/// Response for namespace listing.
#[derive(Debug, Serialize, ToSchema)]
pub struct NamespacesResponse {
    pub namespaces: Vec<String>,
}

/// Scalar value in scan results.
#[derive(Debug, Serialize, ToSchema)]
pub struct JsonScalar {
    pub column_id: u32,
    pub value: u64,
}

/// A row in scan results.
#[derive(Debug, Serialize, ToSchema)]
pub struct JsonRow {
    pub scalars: Vec<JsonScalar>,
    /// Payload bytes. Encoding indicated by `payload_encoding`.
    pub payload: String,
    /// "utf8" when payload is valid UTF-8; "hex" otherwise.
    pub payload_encoding: String,
}

/// Response for scan/list operations.
#[derive(Debug, Serialize, ToSchema)]
pub struct ScanResponse {
    pub rows: Vec<JsonRow>,
    pub has_more: bool,
}

/// Response for get-by-hash.
#[derive(Debug, Serialize, ToSchema)]
pub struct GetResponse {
    pub row: Option<JsonRow>,
}

/// Response for write operations.
#[derive(Debug, Serialize, ToSchema)]
pub struct WriteResponse {
    pub status: String,
}

// =========================================================================
// Handlers
// =========================================================================

#[utoipa::path(
    post,
    path = "/v1/db/tables/{ns}/rows",
    tag = "DB::Tables",
    params(("ns" = String, Path, description = "Table namespace")),
    request_body = WriteRowRequest,
    responses((status = 200, body = WriteResponse))
)]
/// POST /v1/db/tables/{ns}/rows — write a row.
pub async fn handle_write_row(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let ns = match parse_namespace(&path) {
        Ok(ns) => ns,
        Err(Some(resp)) => return resp,
        Err(None) => {
            return json_error(StatusCode::BAD_REQUEST, "invalid_path", "missing namespace")
        }
    };

    let body = match read_json_body::<WriteRowRequest>(req).await {
        Ok(b) => b,
        Err(resp) => return resp,
    };

    let root = match resolve_storage_root(&state, auth.as_ref()) {
        Ok(r) => r,
        Err(resp) => return resp,
    };

    // Build column values from JSON.
    let columns: Vec<ColumnValue> = match body
        .columns
        .iter()
        .map(json_to_column_value)
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(c) => c,
        Err(msg) => return json_error(StatusCode::BAD_REQUEST, "invalid_column", &msg),
    };

    // Build policy from request or use minimal default.
    // On first open, the Zig layer persists this to disk; subsequent opens
    // load the persisted policy and ignore what we pass here.
    let policy = match body.policy {
        Some(ref p) => CompactionPolicy {
            active_schema_ids: p.active_schema_ids.clone(),
            tombstone_schema_id: p.tombstone_schema_id,
            dedup_column_id: p.dedup_column_id,
            ts_column_id: p.ts_column_id,
            ttl_column_id: p.ttl_column_id,
        },
        None => CompactionPolicy {
            active_schema_ids: vec![body.schema_id],
            tombstone_schema_id: None,
            dedup_column_id: 1,
            ts_column_id: 2,
            ttl_column_id: None,
        },
    };

    let handle = match TableHandle::open(&root, &ns, &policy) {
        Ok(h) => h,
        Err(e) => return table_error_response(e),
    };

    if let Err(e) = handle.append_row(body.schema_id, &columns) {
        return table_error_response(e);
    }
    if let Err(e) = handle.flush() {
        return table_error_response(e);
    }

    json_response(
        StatusCode::OK,
        &WriteResponse {
            status: "ok".to_string(),
        },
    )
}

#[utoipa::path(
    get,
    path = "/v1/db/tables/{ns}/rows",
    tag = "DB::Tables",
    params(("ns" = String, Path, description = "Table namespace")),
    responses((status = 200, body = ScanResponse))
)]
/// GET /v1/db/tables/{ns}/rows — scan rows.
pub async fn handle_scan(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let ns = match parse_namespace(&path) {
        Ok(ns) => ns,
        Err(Some(resp)) => return resp,
        Err(None) => {
            return json_error(StatusCode::BAD_REQUEST, "invalid_path", "missing namespace")
        }
    };

    let query = req
        .uri()
        .query()
        .map(|q| url::form_urlencoded::parse(q.as_bytes()))
        .unwrap_or_else(|| url::form_urlencoded::parse(b""));
    let query_map: std::collections::HashMap<String, String> =
        query.map(|(k, v)| (k.to_string(), v.to_string())).collect();

    let schema_id: u16 = query_map
        .get("schema_id")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if schema_id == 0 {
        return json_error(
            StatusCode::BAD_REQUEST,
            "missing_param",
            "schema_id query parameter is required",
        );
    }

    let limit: u32 = query_map
        .get("limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let cursor_ts: Option<i64> = query_map.get("cursor_ts").and_then(|v| v.parse().ok());
    let cursor_hash: Option<u64> = query_map.get("cursor_hash").and_then(|v| v.parse().ok());
    let payload_column_id: u32 = query_map
        .get("payload_column_id")
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let delete_schema_id: Option<u16> = query_map
        .get("delete_schema_id")
        .and_then(|v| v.parse().ok());

    let root = match resolve_storage_root(&state, auth.as_ref()) {
        Ok(r) => r,
        Err(resp) => return resp,
    };

    let handle = match TableHandle::open_readonly(&root, &ns) {
        Ok(h) => h,
        Err(e) => return table_error_response(e),
    };

    let params = ScanParams {
        schema_id,
        limit,
        cursor_ts,
        cursor_hash,
        payload_column_id,
        delete_schema_id,
        ..ScanParams::default()
    };

    let result = match handle.scan(&params) {
        Ok(r) => r,
        Err(e) => return table_error_response(e),
    };

    let rows: Vec<JsonRow> = result
        .rows
        .iter()
        .map(|r| {
            let (payload, encoding) = encode_payload(&r.payload);
            JsonRow {
                scalars: r
                    .scalars
                    .iter()
                    .map(|s| JsonScalar {
                        column_id: s.column_id,
                        value: s.value,
                    })
                    .collect(),
                payload,
                payload_encoding: encoding.to_string(),
            }
        })
        .collect();

    json_response(
        StatusCode::OK,
        &ScanResponse {
            rows,
            has_more: result.has_more,
        },
    )
}

#[utoipa::path(
    get,
    path = "/v1/db/tables/{ns}/rows/{hash}",
    tag = "DB::Tables",
    params(
        ("ns" = String, Path, description = "Table namespace"),
        ("hash" = u64, Path, description = "Row primary hash"),
    ),
    responses((status = 200, body = GetResponse))
)]
/// GET /v1/db/tables/{ns}/rows/{hash} — get row by primary hash.
pub async fn handle_get_row(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let (ns, hash) = match parse_namespace_and_hash(&path) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_path",
                "expected /v1/db/tables/{ns}/rows/{hash}",
            )
        }
    };

    let query = req
        .uri()
        .query()
        .map(|q| url::form_urlencoded::parse(q.as_bytes()))
        .unwrap_or_else(|| url::form_urlencoded::parse(b""));
    let query_map: std::collections::HashMap<String, String> =
        query.map(|(k, v)| (k.to_string(), v.to_string())).collect();

    let schema_id: u16 = query_map
        .get("schema_id")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if schema_id == 0 {
        return json_error(
            StatusCode::BAD_REQUEST,
            "missing_param",
            "schema_id query parameter is required",
        );
    }

    let legacy_hash: Option<u64> = query_map.get("legacy_hash").and_then(|v| v.parse().ok());

    let root = match resolve_storage_root(&state, auth.as_ref()) {
        Ok(r) => r,
        Err(resp) => return resp,
    };

    let handle = match TableHandle::open_readonly(&root, &ns) {
        Ok(h) => h,
        Err(e) => return table_error_response(e),
    };

    let row = match handle.get(schema_id, hash, legacy_hash) {
        Ok(r) => r,
        Err(e) => return table_error_response(e),
    };

    let json_row = row.map(|r| {
        let (payload, encoding) = encode_payload(&r.payload);
        JsonRow {
            scalars: r
                .scalars
                .iter()
                .map(|s| JsonScalar {
                    column_id: s.column_id,
                    value: s.value,
                })
                .collect(),
            payload,
            payload_encoding: encoding.to_string(),
        }
    });

    json_response(StatusCode::OK, &GetResponse { row: json_row })
}

#[utoipa::path(
    delete,
    path = "/v1/db/tables/{ns}/rows/{hash}",
    tag = "DB::Tables",
    params(
        ("ns" = String, Path, description = "Table namespace"),
        ("hash" = u64, Path, description = "Row primary hash"),
    ),
    responses((status = 200, body = WriteResponse))
)]
/// DELETE /v1/db/tables/{ns}/rows/{hash} — tombstone a row.
pub async fn handle_delete_row(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let (ns, hash) = match parse_namespace_and_hash(&path) {
        Some(v) => v,
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_path",
                "expected /v1/db/tables/{ns}/rows/{hash}",
            )
        }
    };

    let query = req
        .uri()
        .query()
        .map(|q| url::form_urlencoded::parse(q.as_bytes()))
        .unwrap_or_else(|| url::form_urlencoded::parse(b""));
    let query_map: std::collections::HashMap<String, String> =
        query.map(|(k, v)| (k.to_string(), v.to_string())).collect();

    let tombstone_schema_id: Option<u16> = query_map
        .get("tombstone_schema_id")
        .and_then(|v| v.parse().ok());

    let root = match resolve_storage_root(&state, auth.as_ref()) {
        Ok(r) => r,
        Err(resp) => return resp,
    };

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    // Open with minimal policy. The Zig layer loads the persisted policy if
    // available, so the tombstone_schema_id here is only a fallback for the
    // first open (when no policy.json exists yet).
    let policy = CompactionPolicy {
        active_schema_ids: Vec::new(),
        tombstone_schema_id,
        dedup_column_id: 1,
        ts_column_id: 2,
        ttl_column_id: None,
    };

    let handle = match TableHandle::open(&root, &ns, &policy) {
        Ok(h) => h,
        Err(e) => return table_error_response(e),
    };

    if let Err(e) = handle.delete_row(hash, ts) {
        return table_error_response(e);
    }
    if let Err(e) = handle.flush() {
        return table_error_response(e);
    }

    json_response(
        StatusCode::OK,
        &WriteResponse {
            status: "deleted".to_string(),
        },
    )
}

#[utoipa::path(
    post,
    path = "/v1/db/tables/{ns}/rows/scan",
    tag = "DB::Tables",
    params(("ns" = String, Path, description = "Table namespace")),
    request_body = ScanRequest,
    responses((status = 200, body = ScanResponse))
)]
/// POST /v1/db/tables/{ns}/rows/scan — advanced scan with full engine params.
pub async fn handle_scan_post(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let ns = match parse_scan_namespace(&path) {
        Some(ns) => ns,
        None => return json_error(StatusCode::BAD_REQUEST, "invalid_path", "missing namespace"),
    };

    let body = match read_json_body::<ScanRequest>(req).await {
        Ok(b) => b,
        Err(resp) => return resp,
    };

    // Convert filter op strings to FilterOp enum.
    let filters: Vec<ColumnFilter> = match body
        .filters
        .iter()
        .map(|f| {
            let op = match f.op.as_str() {
                "eq" => Ok(FilterOp::Eq),
                "ne" => Ok(FilterOp::Ne),
                "lt" => Ok(FilterOp::Lt),
                "le" => Ok(FilterOp::Le),
                "gt" => Ok(FilterOp::Gt),
                "ge" => Ok(FilterOp::Ge),
                other => Err(format!(
                    "invalid filter op '{other}'; expected eq/ne/lt/le/gt/ge"
                )),
            }?;
            Ok(ColumnFilter {
                column_id: f.column_id,
                op,
                value: f.value,
            })
        })
        .collect::<Result<Vec<_>, String>>()
    {
        Ok(f) => f,
        Err(msg) => return json_error(StatusCode::BAD_REQUEST, "invalid_filter", &msg),
    };

    let root = match resolve_storage_root(&state, auth.as_ref()) {
        Ok(r) => r,
        Err(resp) => return resp,
    };

    let handle = match TableHandle::open_readonly(&root, &ns) {
        Ok(h) => h,
        Err(e) => return table_error_response(e),
    };

    let params = ScanParams {
        schema_id: body.schema_id,
        additional_schema_ids: body.additional_schema_ids,
        filters,
        extra_columns: body.extra_columns,
        dedup_column_id: if body.dedup { body.dedup_column_id } else { 0 },
        delete_schema_id: body.delete_schema_id,
        ts_column_id: body.ts_column_id,
        ttl_column_id: body.ttl_column_id,
        limit: body.limit,
        cursor_ts: body.cursor_ts,
        cursor_hash: body.cursor_hash,
        payload_column_id: body.payload_column_id,
        reverse: body.reverse,
    };

    let result = match handle.scan(&params) {
        Ok(r) => r,
        Err(e) => return table_error_response(e),
    };

    let rows: Vec<JsonRow> = result
        .rows
        .iter()
        .map(|r| {
            let (payload, encoding) = encode_payload(&r.payload);
            JsonRow {
                scalars: r
                    .scalars
                    .iter()
                    .map(|s| JsonScalar {
                        column_id: s.column_id,
                        value: s.value,
                    })
                    .collect(),
                payload,
                payload_encoding: encoding.to_string(),
            }
        })
        .collect();

    json_response(
        StatusCode::OK,
        &ScanResponse {
            rows,
            has_more: result.has_more,
        },
    )
}

#[utoipa::path(
    get,
    path = "/v1/db/tables/_meta/namespaces",
    tag = "DB::Tables",
    responses((status = 200, body = NamespacesResponse))
)]
/// GET /v1/db/tables/_meta/namespaces — list table namespaces.
pub async fn handle_list_namespaces(
    state: Arc<AppState>,
    _req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let root = match resolve_storage_root(&state, auth.as_ref()) {
        Ok(r) => r,
        Err(resp) => return resp,
    };

    let mut namespaces: Vec<String> = Vec::new();
    let entries = match std::fs::read_dir(&root) {
        Ok(e) => e,
        Err(_) => return json_response(StatusCode::OK, &NamespacesResponse { namespaces }),
    };

    for entry in entries.flatten() {
        if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
            if let Some(name) = entry.file_name().to_str() {
                namespaces.push(name.to_string());
            }
        }
    }
    namespaces.sort();

    json_response(StatusCode::OK, &NamespacesResponse { namespaces })
}

#[utoipa::path(
    get,
    path = "/v1/db/tables/{ns}/_meta/policy",
    tag = "DB::Tables",
    params(("ns" = String, Path, description = "Table namespace")),
    responses((status = 200))
)]
/// GET /v1/db/tables/{ns}/_meta/policy — read persisted policy.
pub async fn handle_get_policy(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let ns = match parse_meta_policy_namespace(&path) {
        Some(ns) => ns,
        None => return json_error(StatusCode::BAD_REQUEST, "invalid_path", "missing namespace"),
    };

    let root = match resolve_storage_root(&state, auth.as_ref()) {
        Ok(r) => r,
        Err(resp) => return resp,
    };

    let policy_path = root.join(&ns).join("policy.json");
    let data = match std::fs::read(&policy_path) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return json_error(
                StatusCode::NOT_FOUND,
                "not_found",
                "no policy for this namespace",
            )
        }
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "storage_error",
                &format!("failed to read policy: {e}"),
            )
        }
    };

    // Return the raw policy JSON wrapped in { "policy": ... }.
    let policy_value: serde_json::Value = match serde_json::from_slice(&data) {
        Ok(v) => v,
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "storage_error",
                &format!("failed to parse policy: {e}"),
            )
        }
    };

    json_response(
        StatusCode::OK,
        &serde_json::json!({ "policy": policy_value }),
    )
}

// =========================================================================
// Path parsing
// =========================================================================

/// Extract namespace from /v1/db/tables/{ns}/rows
fn parse_namespace(path: &str) -> Result<String, Option<Response<BoxBody>>> {
    let stripped = path.strip_prefix(PATH_PREFIX).ok_or(None)?;
    let mut parts = stripped.split('/');
    let ns_raw = parts.next().unwrap_or("");
    let action = parts.next().ok_or(None)?;
    if action != "rows" || parts.next().is_some() {
        return Err(None);
    }
    // Pattern matches /v1/db/tables/{ns}/rows — validate namespace.
    if ns_raw.is_empty() {
        return Err(Some(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "missing namespace",
        )));
    }
    let decoded = percent_encoding::percent_decode_str(ns_raw)
        .decode_utf8_lossy()
        .into_owned();
    if decoded.contains('/') {
        return Err(Some(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "namespace must not contain path separators",
        )));
    }
    Ok(decoded)
}

/// Extract namespace and hash from /v1/db/tables/{ns}/rows/{hash}
fn parse_namespace_and_hash(path: &str) -> Option<(String, u64)> {
    let stripped = path.strip_prefix(PATH_PREFIX)?;
    let mut parts = stripped.split('/');
    let ns = parts.next().filter(|s| !s.is_empty())?;
    let action = parts.next()?;
    if action != "rows" {
        return None;
    }
    let hash_str = parts.next().filter(|s| !s.is_empty())?;
    if parts.next().is_some() {
        return None;
    }
    let decoded_hash = percent_encoding::percent_decode_str(hash_str).decode_utf8_lossy();
    let hash: u64 = decoded_hash.parse().ok()?;
    let decoded_ns = percent_encoding::percent_decode_str(ns)
        .decode_utf8_lossy()
        .into_owned();
    if decoded_ns.contains('/') {
        return None;
    }
    Some((decoded_ns, hash))
}

/// Extract namespace from /v1/db/tables/{ns}/rows/scan
fn parse_scan_namespace(path: &str) -> Option<String> {
    let stripped = path.strip_prefix(PATH_PREFIX)?;
    let mut parts = stripped.split('/');
    let ns = parts.next().filter(|s| !s.is_empty())?;
    if parts.next()? != "rows" {
        return None;
    }
    if parts.next()? != "scan" {
        return None;
    }
    if parts.next().is_some() {
        return None;
    }
    let decoded = percent_encoding::percent_decode_str(ns)
        .decode_utf8_lossy()
        .into_owned();
    if decoded.contains('/') {
        return None;
    }
    Some(decoded)
}

/// Extract namespace from /v1/db/tables/{ns}/_meta/policy
fn parse_meta_policy_namespace(path: &str) -> Option<String> {
    let stripped = path.strip_prefix(PATH_PREFIX)?;
    let mut parts = stripped.split('/');
    let ns = parts.next().filter(|s| !s.is_empty())?;
    if parts.next()? != "_meta" {
        return None;
    }
    if parts.next()? != "policy" {
        return None;
    }
    if parts.next().is_some() {
        return None;
    }
    let decoded = percent_encoding::percent_decode_str(ns)
        .decode_utf8_lossy()
        .into_owned();
    if decoded.contains('/') {
        return None;
    }
    Some(decoded)
}

/// Check if path matches a table rows endpoint.
pub fn is_table_rows_path(path: &str) -> bool {
    !matches!(parse_namespace(path), Err(None))
}

/// Check if path matches a table rows/{hash} endpoint.
pub fn is_table_row_path(path: &str) -> bool {
    parse_namespace_and_hash(path).is_some()
}

/// Check if path matches POST /v1/db/tables/{ns}/rows/scan.
pub fn is_table_scan_path(path: &str) -> bool {
    parse_scan_namespace(path).is_some()
}

/// Check if path matches GET /v1/db/tables/{ns}/_meta/policy.
pub fn is_table_meta_policy_path(path: &str) -> bool {
    parse_meta_policy_namespace(path).is_some()
}

/// Check if path matches GET /v1/db/tables/_meta/namespaces.
pub fn is_table_meta_namespaces_path(path: &str) -> bool {
    path == "/v1/db/tables/_meta/namespaces"
}

// =========================================================================
// Helpers
// =========================================================================

fn json_to_column_value(jcv: &JsonColumnValue) -> Result<ColumnValue, String> {
    match jcv.r#type.as_str() {
        "scalar_u64" => {
            let val: u64 = jcv
                .value
                .as_u64()
                .ok_or_else(|| format!("column {} value must be u64", jcv.column_id))?;
            Ok(ColumnValue {
                column_id: jcv.column_id,
                shape: ColumnShape::Scalar,
                phys_type: PhysicalType::U64,
                dims: 1,
                data: val.to_le_bytes().to_vec(),
            })
        }
        "scalar_i64" => {
            let val: i64 = jcv
                .value
                .as_i64()
                .ok_or_else(|| format!("column {} value must be i64", jcv.column_id))?;
            Ok(ColumnValue {
                column_id: jcv.column_id,
                shape: ColumnShape::Scalar,
                phys_type: PhysicalType::I64,
                dims: 1,
                data: val.to_le_bytes().to_vec(),
            })
        }
        "scalar_f64" => {
            let val: f64 = jcv
                .value
                .as_f64()
                .ok_or_else(|| format!("column {} value must be f64", jcv.column_id))?;
            Ok(ColumnValue {
                column_id: jcv.column_id,
                shape: ColumnShape::Scalar,
                phys_type: PhysicalType::F64,
                dims: 1,
                data: val.to_le_bytes().to_vec(),
            })
        }
        "varbytes" => {
            let hex_str = jcv
                .value
                .as_str()
                .ok_or_else(|| format!("column {} value must be hex string", jcv.column_id))?;
            let data = decode_hex(hex_str)
                .map_err(|e| format!("column {} hex decode error: {e}", jcv.column_id))?;
            Ok(ColumnValue {
                column_id: jcv.column_id,
                shape: ColumnShape::VarBytes,
                phys_type: PhysicalType::Binary,
                dims: 0,
                data,
            })
        }
        "json" => {
            let json_str = serde_json::to_string(&jcv.value)
                .map_err(|e| format!("column {} json encode error: {e}", jcv.column_id))?;
            Ok(ColumnValue {
                column_id: jcv.column_id,
                shape: ColumnShape::VarBytes,
                phys_type: PhysicalType::Binary,
                dims: 0,
                data: json_str.into_bytes(),
            })
        }
        "string" => {
            let s = jcv
                .value
                .as_str()
                .ok_or_else(|| format!("column {} value must be a string", jcv.column_id))?;
            Ok(ColumnValue {
                column_id: jcv.column_id,
                shape: ColumnShape::VarBytes,
                phys_type: PhysicalType::Binary,
                dims: 0,
                data: s.as_bytes().to_vec(),
            })
        }
        other => Err(format!(
            "column {} unknown type '{}'; expected scalar_u64, scalar_i64, scalar_f64, varbytes, json, or string",
            jcv.column_id, other
        )),
    }
}

fn resolve_storage_root(
    state: &AppState,
    auth: Option<&AuthContext>,
) -> Result<PathBuf, Response<BoxBody>> {
    let bucket = match state.bucket_path.as_ref() {
        Some(p) => p,
        None => {
            return Err(json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no_storage",
                "Storage not configured",
            ))
        }
    };
    let base = match auth {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };
    let root = base.join("tables");
    std::fs::create_dir_all(&root).map_err(|e| {
        json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage_error",
            &format!("failed to create storage root: {e}"),
        )
    })?;
    Ok(root)
}

async fn read_json_body<T: serde::de::DeserializeOwned>(
    req: Request<Incoming>,
) -> Result<T, Response<BoxBody>> {
    let body = req
        .into_body()
        .collect()
        .await
        .map_err(|_| {
            json_error(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                "Failed to read body",
            )
        })?
        .to_bytes();
    serde_json::from_slice(&body).map_err(|e| {
        json_error(
            StatusCode::BAD_REQUEST,
            "invalid_json",
            &format!("JSON parse error: {e}"),
        )
    })
}

fn table_error_response(err: TableError) -> Response<BoxBody> {
    match err {
        TableError::InvalidArgument(msg) => {
            json_error(StatusCode::BAD_REQUEST, "invalid_argument", &msg)
        }
        TableError::ReadOnly => {
            json_error(StatusCode::BAD_REQUEST, "read_only", "Table is read-only")
        }
        TableError::StorageError(msg) => {
            json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &msg)
        }
    }
}

fn json_response<T: serde::Serialize>(status: StatusCode, value: &T) -> Response<BoxBody> {
    let body = serde_json::to_vec(value).unwrap_or_default();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}

fn json_error(status: StatusCode, code: &str, message: &str) -> Response<BoxBody> {
    let body = serde_json::json!({
        "error": {
            "code": code,
            "message": message
        }
    });
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body.to_string())).boxed())
        .unwrap()
}

/// Encode payload bytes for JSON response. Returns (payload_string, encoding).
/// Valid UTF-8 is returned as-is with encoding "utf8"; otherwise hex-encoded with "hex".
fn encode_payload(bytes: &[u8]) -> (String, &'static str) {
    match std::str::from_utf8(bytes) {
        Ok(s) => (s.to_string(), "utf8"),
        Err(_) => (encode_hex(bytes), "hex"),
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn decode_hex(hex: &str) -> Result<Vec<u8>, String> {
    if hex.len() % 2 != 0 {
        return Err("odd-length hex string".to_string());
    }
    let mut out = Vec::with_capacity(hex.len() / 2);
    for chunk in hex.as_bytes().chunks(2) {
        let hi = hex_nibble(chunk[0])
            .ok_or_else(|| format!("invalid hex char '{}'", chunk[0] as char))?;
        let lo = hex_nibble(chunk[1])
            .ok_or_else(|| format!("invalid hex char '{}'", chunk[1] as char))?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn hex_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}
