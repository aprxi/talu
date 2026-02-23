//! DB ops plane HTTP handlers.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Method, Request, Response, StatusCode};
use serde::Deserialize;

use talu::vector::{VectorError, VectorStore};

use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;

const COLLECTIONS_DIR: &str = "vector";
const COLLECTION_STORES_DIR: &str = "collections";

#[derive(Debug, Deserialize)]
struct CompactRequest {
    collection: String,
    dims: u32,
}

#[derive(Debug, Deserialize)]
struct SimulateCrashRequest {
    collection: String,
}

pub async fn handle(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    match (req.method(), req.uri().path()) {
        (&Method::POST, "/v1/db/ops/compact") => handle_compact(state, req, auth).await,
        (&Method::POST, "/v1/db/ops/simulate_crash") => {
            handle_simulate_crash(state, req, auth).await
        }
        _ => json_error(
            StatusCode::NOT_IMPLEMENTED,
            "not_implemented",
            "db ops endpoint not implemented yet",
        ),
    }
}

async fn handle_compact(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let storage_root = match resolve_storage_root(&state, &auth) {
        Ok(path) => path,
        Err(resp) => return resp,
    };

    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e.to_string()),
    };

    let compact_req: CompactRequest = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    if compact_req.collection.trim().is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "collection is required",
        );
    }
    if compact_req.dims == 0 {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "dims must be greater than zero",
        );
    }

    let store_root = collection_store_root(&storage_root, &compact_req.collection);
    let root_str = store_root.display().to_string();
    let store = match VectorStore::open(&root_str) {
        Ok(s) => s,
        Err(err) => return vector_error_response(err),
    };

    let compact = match store.compact(compact_req.dims) {
        Ok(v) => v,
        Err(err) => return vector_error_response(err),
    };

    json_response(
        StatusCode::OK,
        &serde_json::json!({
            "collection": compact_req.collection,
            "dims": compact_req.dims,
            "kept_count": compact.kept_count,
            "removed_tombstones": compact.removed_tombstones,
        }),
    )
}

async fn handle_simulate_crash(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let storage_root = match resolve_storage_root(&state, &auth) {
        Ok(path) => path,
        Err(resp) => return resp,
    };

    let body = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_body", &e.to_string()),
    };

    let crash_req: SimulateCrashRequest = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    if crash_req.collection.trim().is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "collection is required",
        );
    }

    let store_root = collection_store_root(&storage_root, &crash_req.collection);
    let root_str = store_root.display().to_string();
    let mut store = match VectorStore::open(&root_str) {
        Ok(s) => s,
        Err(err) => return vector_error_response(err),
    };

    store.simulate_crash();

    json_response(
        StatusCode::OK,
        &serde_json::json!({
            "collection": crash_req.collection,
            "status": "simulated_crash",
        }),
    )
}

fn resolve_storage_root(
    state: &AppState,
    auth: &Option<AuthContext>,
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

    let root = match auth.as_ref() {
        Some(ctx) => bucket.join(&ctx.storage_prefix),
        None => bucket.to_path_buf(),
    };

    if let Err(e) = std::fs::create_dir_all(&root) {
        return Err(json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage_error",
            &format!("failed to create storage root: {e}"),
        ));
    }

    Ok(root)
}

fn collection_store_root(storage_root: &Path, collection_name: &str) -> PathBuf {
    storage_root
        .join(COLLECTIONS_DIR)
        .join(COLLECTION_STORES_DIR)
        .join(collection_name)
}

fn json_response<T: serde::Serialize>(status: StatusCode, data: &T) -> Response<BoxBody> {
    let body = serde_json::to_vec(data).unwrap_or_default();
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

fn vector_error_response(err: VectorError) -> Response<BoxBody> {
    match err {
        VectorError::InvalidArgument(msg) => {
            let lower = msg.to_ascii_lowercase();
            if lower.contains("dim") {
                json_error(StatusCode::BAD_REQUEST, "dimension_mismatch", &msg)
            } else {
                json_error(StatusCode::BAD_REQUEST, "invalid_argument", &msg)
            }
        }
        VectorError::StoreError(msg) => {
            json_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error", &msg)
        }
    }
}
