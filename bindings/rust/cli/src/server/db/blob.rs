//! DB blob plane HTTP handlers.

use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};

use talu::blobs::{BlobError, BlobsHandle};

use crate::server::auth_gateway::AuthContext;
use crate::server::files;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;

pub async fn handle_get(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    files::handle_get_blob(state, req, auth).await
}

pub async fn handle_list(
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

    let storage_path = match auth.as_ref() {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };

    let query = req.uri().query().unwrap_or("");
    let limit = parse_query_param(query, "limit")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(100);

    let handle = match BlobsHandle::open(&storage_path) {
        Ok(h) => h,
        Err(err) => return blob_error_response(err),
    };

    let refs = match handle.list(Some(limit)) {
        Ok(items) => items,
        Err(err) => return blob_error_response(err),
    };

    let count = refs.len();
    let body = serde_json::json!({
        "data": refs,
        "count": count,
    });

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body.to_string())).boxed())
        .unwrap()
}

fn parse_query_param<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find_map(|(k, v)| if k == key { Some(v) } else { None })
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

fn blob_error_response(err: BlobError) -> Response<BoxBody> {
    match err {
        BlobError::InvalidArgument(msg) => {
            json_error(StatusCode::BAD_REQUEST, "invalid_argument", &msg)
        }
        BlobError::NotFound(msg) => json_error(StatusCode::NOT_FOUND, "not_found", &msg),
        BlobError::PermissionDenied(msg) => {
            json_error(StatusCode::FORBIDDEN, "permission_denied", &msg)
        }
        BlobError::ResourceExhausted(msg) => {
            json_error(StatusCode::INSUFFICIENT_STORAGE, "resource_exhausted", &msg)
        }
        BlobError::StorageError(msg) => {
            json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &msg)
        }
    }
}
