//! DB kv plane HTTP handlers.

use std::path::PathBuf;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde::Serialize;
use utoipa::ToSchema;

use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

use talu::kv::{KvError, KvHandle};

type BoxBody = http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>;

#[derive(Debug, Serialize, ToSchema)]
pub struct KvEntryResponse {
    pub key: String,
    pub value_len: usize,
    pub value_hex: String,
    pub updated_at_ms: i64,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct KvListResponse {
    pub namespace: String,
    pub count: usize,
    pub data: Vec<KvEntryResponse>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct KvPutResponse {
    pub namespace: String,
    pub key: String,
    pub value_len: usize,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct KvDeleteResponse {
    pub namespace: String,
    pub key: String,
    pub deleted: bool,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct KvCompactResponse {
    pub namespace: String,
    pub status: String,
}

#[utoipa::path(
    get,
    path = "/v1/db/kv/namespaces/{namespace}/entries",
    tag = "DB::KV",
    responses(
        (status = 200, description = "List KV entries", body = KvListResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("namespace" = String, Path, description = "KV namespace")
    )
)]
pub async fn handle_list(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let (namespace, root) = match parse_namespace_and_root(&state, req.uri().path(), auth.as_ref())
    {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let kv = match KvHandle::open(&root, &namespace) {
        Ok(h) => h,
        Err(err) => return kv_error_response(err),
    };
    let entries = match kv.list() {
        Ok(v) => v,
        Err(err) => return kv_error_response(err),
    };

    let data = entries
        .into_iter()
        .map(|entry| KvEntryResponse {
            key: entry.key,
            value_len: entry.value.len(),
            value_hex: encode_hex(&entry.value),
            updated_at_ms: entry.updated_at_ms,
        })
        .collect::<Vec<_>>();

    json_response(
        StatusCode::OK,
        &KvListResponse {
            namespace,
            count: data.len(),
            data,
        },
    )
}

#[utoipa::path(
    put,
    path = "/v1/db/kv/namespaces/{namespace}/entries/{key}",
    tag = "DB::KV",
    request_body(content = String, content_type = "application/octet-stream"),
    responses(
        (status = 200, description = "KV entry upserted", body = KvPutResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("namespace" = String, Path, description = "KV namespace"),
        ("key" = String, Path, description = "KV key")
    )
)]
pub async fn handle_put(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let (namespace, key, root) = match parse_entry_path_and_root(&state, &path, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let value = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                "Failed to read request body",
            )
        }
    };

    let kv = match KvHandle::open(&root, &namespace) {
        Ok(h) => h,
        Err(err) => return kv_error_response(err),
    };
    if let Err(err) = kv.put(&key, value.as_ref()) {
        return kv_error_response(err);
    }

    json_response(
        StatusCode::OK,
        &KvPutResponse {
            namespace,
            key,
            value_len: value.len(),
        },
    )
}

#[utoipa::path(
    get,
    path = "/v1/db/kv/namespaces/{namespace}/entries/{key}",
    tag = "DB::KV",
    responses(
        (status = 200, description = "KV entry fetched", body = KvEntryResponse),
        (status = 404, description = "Entry not found", body = crate::server::http::ErrorResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("namespace" = String, Path, description = "KV namespace"),
        ("key" = String, Path, description = "KV key")
    )
)]
pub async fn handle_get(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let (namespace, key, root) = match parse_entry_path_and_root(&state, &path, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let kv = match KvHandle::open(&root, &namespace) {
        Ok(h) => h,
        Err(err) => return kv_error_response(err),
    };
    let value = match kv.get(&key) {
        Ok(v) => v,
        Err(err) => return kv_error_response(err),
    };
    let Some(value) = value else {
        return json_error(
            StatusCode::NOT_FOUND,
            "not_found",
            "KV entry does not exist",
        );
    };

    json_response(
        StatusCode::OK,
        &KvEntryResponse {
            key,
            value_len: value.data.len(),
            value_hex: encode_hex(&value.data),
            updated_at_ms: value.updated_at_ms,
        },
    )
}

#[utoipa::path(
    delete,
    path = "/v1/db/kv/namespaces/{namespace}/entries/{key}",
    tag = "DB::KV",
    responses(
        (status = 200, description = "KV entry deleted", body = KvDeleteResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("namespace" = String, Path, description = "KV namespace"),
        ("key" = String, Path, description = "KV key")
    )
)]
pub async fn handle_delete(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path().to_string();
    let (namespace, key, root) = match parse_entry_path_and_root(&state, &path, auth.as_ref()) {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let kv = match KvHandle::open(&root, &namespace) {
        Ok(h) => h,
        Err(err) => return kv_error_response(err),
    };
    let deleted = match kv.delete(&key) {
        Ok(v) => v,
        Err(err) => return kv_error_response(err),
    };

    json_response(
        StatusCode::OK,
        &KvDeleteResponse {
            namespace,
            key,
            deleted,
        },
    )
}

#[utoipa::path(
    post,
    path = "/v1/db/kv/namespaces/{namespace}/flush",
    tag = "DB::KV",
    responses(
        (status = 200, description = "KV namespace flushed", body = KvCompactResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("namespace" = String, Path, description = "KV namespace")
    )
)]
pub async fn handle_flush(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    handle_state_op(state, req.uri().path(), auth.as_ref(), true)
}

#[utoipa::path(
    post,
    path = "/v1/db/kv/namespaces/{namespace}/compact",
    tag = "DB::KV",
    responses(
        (status = 200, description = "KV namespace compacted", body = KvCompactResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse),
        (status = 503, description = "Storage unavailable", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("namespace" = String, Path, description = "KV namespace")
    )
)]
pub async fn handle_compact(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    handle_state_op(state, req.uri().path(), auth.as_ref(), false)
}

fn handle_state_op(
    state: Arc<AppState>,
    path: &str,
    auth: Option<&AuthContext>,
    flush: bool,
) -> Response<BoxBody> {
    let (namespace, root) = match parse_namespace_and_root(&state, path, auth) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let kv = match KvHandle::open(&root, &namespace) {
        Ok(h) => h,
        Err(err) => return kv_error_response(err),
    };
    let result = if flush { kv.flush() } else { kv.compact() };
    if let Err(err) = result {
        return kv_error_response(err);
    }
    json_response(
        StatusCode::OK,
        &KvCompactResponse {
            namespace,
            status: if flush { "flushed" } else { "compacted" }.to_string(),
        },
    )
}

fn parse_namespace_and_root(
    state: &AppState,
    path: &str,
    auth: Option<&AuthContext>,
) -> Result<(String, String), Response<BoxBody>> {
    let Some(stripped) = path.strip_prefix("/v1/db/kv/namespaces/") else {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "missing namespace",
        ));
    };
    let namespace_raw = stripped.split('/').next().unwrap_or("");
    if namespace_raw.is_empty() {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "missing namespace",
        ));
    }
    let namespace = percent_encoding::percent_decode_str(namespace_raw)
        .decode_utf8_lossy()
        .into_owned();
    if namespace.trim().is_empty() {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "namespace must be non-empty",
        ));
    }

    let root = resolve_storage_root(state, auth)?;
    Ok((namespace, root.to_string_lossy().to_string()))
}

fn parse_entry_path_and_root(
    state: &AppState,
    path: &str,
    auth: Option<&AuthContext>,
) -> Result<(String, String, String), Response<BoxBody>> {
    let Some(stripped) = path.strip_prefix("/v1/db/kv/namespaces/") else {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "missing namespace",
        ));
    };
    let mut parts = stripped.splitn(3, '/');
    let namespace_raw = parts.next().unwrap_or("");
    let segment = parts.next().unwrap_or("");
    let key_raw = parts.next().unwrap_or("");

    if namespace_raw.is_empty() || segment != "entries" || key_raw.is_empty() {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_path",
            "expected /v1/db/kv/namespaces/{namespace}/entries/{key}",
        ));
    }

    let namespace = percent_encoding::percent_decode_str(namespace_raw)
        .decode_utf8_lossy()
        .into_owned();
    if namespace.trim().is_empty() {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "namespace must be non-empty",
        ));
    }
    let key = percent_encoding::percent_decode_str(key_raw)
        .decode_utf8_lossy()
        .into_owned();
    if key.is_empty() {
        return Err(json_error(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "key must be non-empty",
        ));
    }
    let root = resolve_storage_root(state, auth)?;
    Ok((namespace, key, root.to_string_lossy().to_string()))
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
    let root = base.join("kv");
    std::fs::create_dir_all(&root).map_err(|e| {
        json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage_error",
            &format!("failed to create storage root: {e}"),
        )
    })?;
    Ok(root)
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

fn kv_error_response(err: KvError) -> Response<BoxBody> {
    match err {
        KvError::InvalidArgument(msg) => {
            json_error(StatusCode::BAD_REQUEST, "invalid_argument", &msg)
        }
        KvError::Busy(msg) => json_error(StatusCode::CONFLICT, "resource_busy", &msg),
        KvError::Storage(msg) => {
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
