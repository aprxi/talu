//! Plugin discovery and asset serving endpoints.
//!
//! - `GET /v1/plugins` — list discovered plugins from ~/.talu/plugins/
//! - `GET /v1/plugins/{id}/{path...}` — serve plugin static assets

use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde::Serialize;

use crate::server::auth_gateway::AuthContext;
use crate::server::state::{AppState, PluginTokenEntry};

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

// =============================================================================
// Discovery endpoint: GET /v1/plugins
// =============================================================================

#[derive(Serialize)]
struct PluginEntry {
    id: String,
    manifest: serde_json::Value,
    #[serde(rename = "entryUrl")]
    entry_url: String,
    token: String,
}

#[derive(Serialize)]
struct PluginListResponse {
    data: Vec<PluginEntry>,
}

pub async fn handle_list(
    state: Arc<AppState>,
    _req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let plugins_dir = talu::plugins::default_plugins_dir();

    let plugins = if !plugins_dir.exists() {
        Vec::new()
    } else {
        match talu::plugins::scan_plugins(Some(&plugins_dir)) {
            Ok(list) => list,
            Err(e) => {
                log::warn!(target: "server::plugins", "plugin scan failed: {}", e);
                return json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "scan_error",
                    &format!("Failed to scan plugins: {}", e),
                );
            }
        }
    };

    // Clear and regenerate capability tokens.
    let mut tokens = state.plugin_tokens.lock().await;
    tokens.clear();

    let entries: Vec<PluginEntry> = plugins
        .into_iter()
        .map(|p| {
            let manifest: serde_json::Value =
                serde_json::from_str(&p.manifest_json).unwrap_or(serde_json::Value::Null);
            let entry_url = format!("/v1/plugins/{}/{}", p.plugin_id, p.entry_path);

            let token = generate_token();
            let network_permissions = extract_network_permissions(&manifest);

            tokens.insert(
                token.clone(),
                PluginTokenEntry {
                    plugin_id: p.plugin_id.clone(),
                    network_permissions,
                },
            );

            PluginEntry {
                id: p.plugin_id,
                manifest,
                entry_url,
                token,
            }
        })
        .collect();

    drop(tokens);

    json_response(StatusCode::OK, &PluginListResponse { data: entries })
}

// =============================================================================
// Asset serving: GET /v1/plugins/{id}/{path...}
// =============================================================================

pub async fn handle_asset(
    _state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let path = req.uri().path();

    // Strip /v1/plugins/ or /plugins/ prefix.
    let remainder = if let Some(r) = path.strip_prefix("/v1/plugins/") {
        r
    } else if let Some(r) = path.strip_prefix("/plugins/") {
        r
    } else {
        return json_error(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "Invalid plugin path",
        );
    };

    // Split into plugin_id and file_path.
    let (plugin_id, file_path) = match remainder.split_once('/') {
        Some((id, rest)) if !id.is_empty() && !rest.is_empty() => (id, rest),
        _ => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "bad_request",
                "Expected /v1/plugins/{id}/{path}",
            );
        }
    };

    // Reject path traversal attempts.
    if file_path.contains("..") {
        return json_error(
            StatusCode::FORBIDDEN,
            "forbidden",
            "Path traversal not allowed",
        );
    }

    let plugins_dir = talu::plugins::default_plugins_dir();
    let plugin_root = plugins_dir.join(plugin_id);

    if !plugin_root.is_dir() {
        return json_error(
            StatusCode::NOT_FOUND,
            "not_found",
            &format!("Plugin not found: {}", plugin_id),
        );
    }

    let file = plugin_root.join(file_path);

    // Canonicalize both paths and verify the file is inside the plugin root.
    let canon_root = match plugin_root.canonicalize() {
        Ok(p) => p,
        Err(_) => {
            return json_error(
                StatusCode::NOT_FOUND,
                "not_found",
                "Plugin directory not found",
            );
        }
    };
    let canon_file = match file.canonicalize() {
        Ok(p) => p,
        Err(_) => {
            return json_error(
                StatusCode::NOT_FOUND,
                "not_found",
                &format!("Asset not found: {}", file_path),
            );
        }
    };
    if !canon_file.starts_with(&canon_root) {
        return json_error(
            StatusCode::FORBIDDEN,
            "forbidden",
            "Path traversal not allowed",
        );
    }

    // Read file and serve with appropriate MIME type.
    let data = match std::fs::read(&canon_file) {
        Ok(d) => d,
        Err(_) => {
            return json_error(
                StatusCode::NOT_FOUND,
                "not_found",
                &format!("Asset not found: {}", file_path),
            );
        }
    };

    let content_type = mime_for_path(&canon_file);

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", content_type)
        .body(Full::new(Bytes::from(data)).boxed())
        .unwrap()
}

// =============================================================================
// Plugin capability tokens
// =============================================================================

/// Resolve a plugin Bearer token from request headers.
///
/// Returns `Some(plugin_id)` if the `Authorization: Bearer <token>` header
/// contains a valid plugin capability token. Returns `None` otherwise.
pub async fn resolve_bearer_token(state: &AppState, headers: &hyper::HeaderMap) -> Option<String> {
    let auth_header = headers.get("authorization")?.to_str().ok()?;
    let token = auth_header.strip_prefix("Bearer ")?;
    let tokens = state.plugin_tokens.lock().await;
    tokens.get(token).map(|entry| entry.plugin_id.clone())
}

/// Generate a cryptographically random 32-byte hex token.
fn generate_token() -> String {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).expect("failed to generate random bytes");
    hex_encode(&bytes)
}

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        write!(s, "{:02x}", b).unwrap();
    }
    s
}

/// Extract `network:` domain permissions from a plugin manifest.
fn extract_network_permissions(manifest: &serde_json::Value) -> Vec<String> {
    manifest
        .get("permissions")
        .and_then(|p| p.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .filter_map(|s| s.strip_prefix("network:"))
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default()
}

// =============================================================================
// Helpers
// =============================================================================

fn mime_for_path(path: &PathBuf) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("js") | Some("mjs") => "application/javascript",
        Some("css") => "text/css",
        Some("html") | Some("htm") => "text/html; charset=utf-8",
        Some("json") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("woff") => "font/woff",
        Some("woff2") => "font/woff2",
        Some("ttf") => "font/ttf",
        Some("otf") => "font/otf",
        Some("wasm") => "application/wasm",
        _ => "application/octet-stream",
    }
}

fn json_response<T: Serialize>(status: StatusCode, data: &T) -> Response<BoxBody> {
    let body = serde_json::to_vec(data).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}

fn json_error(status: StatusCode, code: &str, message: &str) -> Response<BoxBody> {
    let payload = serde_json::json!({
        "error": {
            "code": code,
            "message": message
        }
    });
    let body = serde_json::to_vec(&payload).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}
