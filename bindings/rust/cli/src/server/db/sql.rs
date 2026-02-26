//! DB sql plane HTTP handlers.

use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde::Deserialize;
use utoipa::ToSchema;

use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

use talu::sql::{SqlEngine, SqlParam};

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct SqlQueryRequest {
    query: String,
    /// Optional typed parameters for parameterized queries.
    /// JSON types map to: string→text, integer→i64, float→f64, null→null,
    /// array of numbers→blob (packed f32).
    params: Option<Vec<serde_json::Value>>,
}

/// Convert a serde_json::Value into a SqlParam.
/// Vectors must be stored as allocated byte buffers whose lifetime is managed
/// by the returned Vec<Vec<u8>> (blob_bufs). This ensures the pointers inside
/// SqlParam remain valid for the duration of the FFI call.
fn json_params_to_sql(
    values: &[serde_json::Value],
    blob_bufs: &mut Vec<Vec<u8>>,
    text_bufs: &mut Vec<Vec<u8>>,
) -> Result<Vec<SqlParam>, String> {
    let mut params = Vec::with_capacity(values.len());
    for val in values {
        match val {
            serde_json::Value::Null => params.push(SqlParam::null()),
            serde_json::Value::Bool(b) => params.push(SqlParam::int(if *b { 1 } else { 0 })),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    params.push(SqlParam::int(i));
                } else if let Some(f) = n.as_f64() {
                    params.push(SqlParam::float(f));
                } else {
                    return Err(format!("unsupported number: {n}"));
                }
            }
            serde_json::Value::String(s) => {
                text_bufs.push(s.as_bytes().to_vec());
                let buf = text_bufs.last().unwrap();
                params.push(SqlParam::text(buf));
            }
            serde_json::Value::Array(arr) => {
                // Array of numbers → pack as f32 blob (query vector).
                let mut floats = Vec::with_capacity(arr.len());
                for elem in arr {
                    let f = elem.as_f64().ok_or_else(|| {
                        format!("array param must contain only numbers, got: {elem}")
                    })?;
                    floats.push(f as f32);
                }
                let bytes: Vec<u8> = floats
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();
                blob_bufs.push(bytes);
                let buf = blob_bufs.last().unwrap();
                params.push(SqlParam::blob(buf));
            }
            serde_json::Value::Object(_) => {
                return Err("object params are not supported".to_string());
            }
        }
    }
    Ok(params)
}

#[utoipa::path(
    post,
    path = "/v1/db/sql/query",
    tag = "DB::SQL",
    request_body = SqlQueryRequest,
    responses(
        (status = 200, description = "SQL query results as JSON"),
        (status = 400, description = "Invalid query or request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    )
)]
pub async fn handle_query(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let bucket = match state.bucket_path.as_ref() {
        Some(b) => b,
        None => {
            return json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no_storage",
                "Storage not configured",
            )
        }
    };

    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                "Failed to read request body",
            )
        }
    };

    let parsed: SqlQueryRequest = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_json",
                "Expected JSON body: {\"query\":\"...\"}",
            )
        }
    };

    let storage_path = match auth.as_ref() {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };
    let root = storage_path.to_string_lossy().to_string();

    // If params are provided (even empty), use the parameterized path (structured response).
    // Otherwise, use the legacy path (raw JSON array) for backward compat.
    if let Some(ref json_params) = parsed.params {
        let mut blob_bufs = Vec::new();
        let mut text_bufs = Vec::new();
        let sql_params = match json_params_to_sql(json_params, &mut blob_bufs, &mut text_bufs) {
            Ok(p) => p,
            Err(msg) => return json_error(StatusCode::BAD_REQUEST, "invalid_params", &msg),
        };

        match SqlEngine::query_params(&root, &parsed.query, &sql_params) {
            Ok(json) => Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(json)).boxed())
                .unwrap(),
            Err(err) => json_error(StatusCode::BAD_REQUEST, "sql_error", &err.to_string()),
        }
    } else {
        match SqlEngine::query_json(&root, &parsed.query) {
            Ok(json) => Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(json)).boxed())
                .unwrap(),
            Err(err) => json_error(StatusCode::BAD_REQUEST, "sql_error", &err.to_string()),
        }
    }
}

#[utoipa::path(
    post,
    path = "/v1/db/sql/explain",
    tag = "DB::SQL",
    request_body = SqlQueryRequest,
    responses(
        (status = 200, description = "EXPLAIN QUERY PLAN results as JSON"),
        (status = 400, description = "Invalid query or request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    )
)]
pub async fn handle_explain(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let bucket = match state.bucket_path.as_ref() {
        Some(b) => b,
        None => {
            return json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no_storage",
                "Storage not configured",
            )
        }
    };

    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                "Failed to read request body",
            )
        }
    };

    let parsed: SqlQueryRequest = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_json",
                "Expected JSON body: {\"query\":\"...\"}",
            )
        }
    };

    let storage_path = match auth.as_ref() {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };
    let root = storage_path.to_string_lossy().to_string();

    let explain_query = format!("EXPLAIN QUERY PLAN {}", parsed.query);

    let mut blob_bufs = Vec::new();
    let mut text_bufs = Vec::new();
    let sql_params = if let Some(ref json_params) = parsed.params {
        match json_params_to_sql(json_params, &mut blob_bufs, &mut text_bufs) {
            Ok(p) => p,
            Err(msg) => return json_error(StatusCode::BAD_REQUEST, "invalid_params", &msg),
        }
    } else {
        Vec::new()
    };

    match SqlEngine::query_params(&root, &explain_query, &sql_params) {
        Ok(json) => Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(json)).boxed())
            .unwrap(),
        Err(err) => json_error(StatusCode::BAD_REQUEST, "sql_error", &err.to_string()),
    }
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
