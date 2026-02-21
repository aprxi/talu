use std::collections::HashSet;
use std::convert::Infallible;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Method, Request, Response, StatusCode};
use once_cell::sync::Lazy;
use serde_json::Value;
use tower_service::Service;

use serde::Serialize;
use utoipa::ToSchema;

use crate::server::auth_gateway::AuthContext;
use crate::server::code;
use crate::server::code_ws;
use crate::server::conversations;
use crate::server::documents;
use crate::server::file;
use crate::server::files;
use crate::server::handlers;
use crate::server::openapi;
use crate::server::plugins;
use crate::server::proxy;
use crate::server::repo;
use crate::server::search;
use crate::server::settings;
use crate::server::state::AppState;
use crate::server::tags;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

/// Structured error response returned by all endpoints.
#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

/// Error details.
#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
}

static OPENAPI_SPEC: Lazy<Vec<u8>> = Lazy::new(openapi::build_openapi_json);

const SWAGGER_UI_HTML: &[u8] = br##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Talu API</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css"/>
</head>
<body>
<div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>SwaggerUIBundle({url:"/openapi.json",dom_id:"#swagger-ui"});</script>
</body>
</html>
"##;

// Console UI assets — only compiled in when `make ui` has been run.
// build.rs sets cfg(bundled_ui) when ui/dist/ contains the required files.
#[cfg(bundled_ui)]
const CONSOLE_HTML: &[u8] = include_bytes!("../../../../../ui/dist/index.html");
#[cfg(bundled_ui)]
const CONSOLE_CSS: &[u8] = include_bytes!("../../../../../ui/dist/style.css");
#[cfg(bundled_ui)]
const CONSOLE_JS: &[u8] = include_bytes!("../../../../../ui/dist/main.js");

static KNOWN_PATHS: Lazy<HashSet<String>> = Lazy::new(|| {
    let mut paths = HashSet::new();
    let parsed: Value = serde_json::from_slice(&OPENAPI_SPEC).unwrap_or(Value::Null);
    if let Some(obj) = parsed.get("paths").and_then(|val| val.as_object()) {
        for key in obj.keys() {
            paths.insert(key.to_string());
        }
    }
    paths
});

#[derive(Clone)]
pub struct Router {
    state: Arc<AppState>,
}

impl Router {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }
}

impl Service<Request<Incoming>> for Router {
    type Response = Response<BoxBody>;
    type Error = Infallible;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: Request<Incoming>) -> Self::Future {
        let state = self.state.clone();
        Box::pin(async move {
            let method = req.method().clone();
            let method_log = method.clone();
            let path = req.uri().path().to_string();
            let path_log = path.clone();

            let content_length = req.headers().get("content-length")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("-");
            let content_type = req.headers().get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("-");
            log::info!(target: "server::http", "{} {}", method, path);
            log::debug!(target: "server::http", "{} {} (content-length={}, content-type={})",
                method, path, content_length, content_type);

            // Auth-exempt routes.
            let response = match (method.clone(), path.as_str()) {
                (Method::GET, "/health") => {
                    Response::new(Full::new(Bytes::from_static(b"ok")).boxed())
                }
                (Method::GET, "/openapi.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/docs") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "text/html; charset=utf-8")
                    .body(Full::new(Bytes::from_static(SWAGGER_UI_HTML)).boxed())
                    .unwrap(),
                // Console UI (auth-exempt)
                (Method::GET, "/") => {
                    let mut resp = serve_ui_asset(&state, "index.html", "text/html; charset=utf-8");
                    if resp.status() == StatusCode::OK {
                        resp.headers_mut().insert(
                            "content-security-policy",
                            hyper::header::HeaderValue::from_static(
                                "default-src 'self'; \
                                 script-src 'self' 'unsafe-inline'; \
                                 style-src 'self' 'unsafe-inline'; \
                                 connect-src 'self'; \
                                 img-src 'self' data: blob:; \
                                 font-src 'self'; \
                                 media-src 'self' blob:; \
                                 object-src 'none'; \
                                 frame-src 'none'",
                            ),
                        );
                        resp.headers_mut().insert(
                            "referrer-policy",
                            hyper::header::HeaderValue::from_static("no-referrer"),
                        );
                    }
                    resp
                }
                (Method::GET, "/assets/style.css") => {
                    serve_ui_asset(&state, "style.css", "text/css")
                }
                (Method::GET, "/assets/main.js") => {
                    serve_ui_asset(&state, "main.js", "application/javascript")
                }
                // Plugin assets (auth-exempt — JS loads before auth context exists)
                (Method::GET, p)
                    if (p.starts_with("/v1/plugins/") || p.starts_with("/plugins/"))
                        && p.matches('/').count() >= 3 =>
                {
                    plugins::handle_asset(state, req, None).await
                }
                _ => {
                    // Authenticate all other routes.
                    let auth = match authenticate(&state, req.headers()) {
                        Ok(ctx) => ctx,
                        Err(resp) => {
                            let status = resp.status();
                            if status.is_client_error() {
                                log::warn!(target: "server::http", "{} {} -> {}", method_log, path_log, status);
                            }
                            return Ok(resp);
                        }
                    };

                    // Resolve plugin Bearer token for document/proxy routes.
                    let plugin_owner = plugins::resolve_bearer_token(&state, req.headers()).await;

                    match (method, path.as_str()) {
                        (Method::GET, "/v1/models") | (Method::GET, "/models") => {
                            handlers::handle_models(state, req, auth).await
                        }
                        (Method::POST, "/v1/responses") | (Method::POST, "/responses") => {
                            handlers::handle_responses(state, req, auth).await
                        }
                        // Settings endpoints
                        (Method::GET, "/v1/settings") | (Method::GET, "/settings") => {
                            settings::handle_get(state, req, auth).await
                        }
                        (Method::PATCH, "/v1/settings") | (Method::PATCH, "/settings") => {
                            settings::handle_patch(state, req, auth).await
                        }
                        (Method::DELETE, p)
                            if p.starts_with("/v1/settings/models/")
                                || p.starts_with("/settings/models/") =>
                        {
                            let prefix = if p.starts_with("/v1") {
                                "/v1/settings/models/"
                            } else {
                                "/settings/models/"
                            };
                            let model_id = &p[prefix.len()..];
                            settings::handle_reset_model(state, req, auth, model_id).await
                        }
                        // Conversation management endpoints
                        (Method::GET, "/v1/conversations") | (Method::GET, "/conversations") => {
                            conversations::handle_list(state, req, auth).await
                        }
                        // Batch operations (must be before single-conversation routes)
                        (Method::POST, "/v1/conversations/batch")
                        | (Method::POST, "/conversations/batch") => {
                            conversations::handle_batch(state, req, auth).await
                        }
                        (Method::GET, p)
                            if (p.starts_with("/v1/conversations/")
                                || p.starts_with("/conversations/"))
                                && !p.ends_with("/fork") =>
                        {
                            conversations::handle_get(state, req, auth).await
                        }
                        (Method::DELETE, p)
                            if (p.starts_with("/v1/conversations/")
                                || p.starts_with("/conversations/"))
                                && !p.ends_with("/tags") =>
                        {
                            conversations::handle_delete(state, req, auth).await
                        }
                        (Method::PATCH, p)
                            if p.starts_with("/v1/conversations/")
                                || p.starts_with("/conversations/") =>
                        {
                            conversations::handle_patch(state, req, auth).await
                        }
                        (Method::POST, p)
                            if p.ends_with("/fork")
                                && (p.starts_with("/v1/conversations/")
                                    || p.starts_with("/conversations/")) =>
                        {
                            conversations::handle_fork(state, req, auth).await
                        }
                        // Search endpoint
                        (Method::POST, "/v1/search") | (Method::POST, "/search") => {
                            search::handle_search(state, req, auth).await
                        }
                        // Tag management endpoints
                        (Method::GET, "/v1/tags") | (Method::GET, "/tags") => {
                            tags::handle_list(state, req, auth).await
                        }
                        (Method::POST, "/v1/tags") | (Method::POST, "/tags") => {
                            tags::handle_create(state, req, auth).await
                        }
                        (Method::GET, p)
                            if p.starts_with("/v1/tags/") || p.starts_with("/tags/") =>
                        {
                            tags::handle_get(state, req, auth).await
                        }
                        (Method::PATCH, p)
                            if p.starts_with("/v1/tags/") || p.starts_with("/tags/") =>
                        {
                            tags::handle_patch(state, req, auth).await
                        }
                        (Method::DELETE, p)
                            if p.starts_with("/v1/tags/") || p.starts_with("/tags/") =>
                        {
                            tags::handle_delete(state, req, auth).await
                        }
                        // Conversation tag endpoints
                        (Method::GET, p)
                            if (p.starts_with("/v1/conversations/")
                                || p.starts_with("/conversations/"))
                                && p.ends_with("/tags") =>
                        {
                            conversations::handle_get_tags(state, req, auth).await
                        }
                        (Method::POST, p)
                            if (p.starts_with("/v1/conversations/")
                                || p.starts_with("/conversations/"))
                                && p.ends_with("/tags") =>
                        {
                            conversations::handle_add_tags(state, req, auth).await
                        }
                        (Method::PUT, p)
                            if (p.starts_with("/v1/conversations/")
                                || p.starts_with("/conversations/"))
                                && p.ends_with("/tags") =>
                        {
                            conversations::handle_set_tags(state, req, auth).await
                        }
                        (Method::DELETE, p)
                            if (p.starts_with("/v1/conversations/")
                                || p.starts_with("/conversations/"))
                                && p.ends_with("/tags") =>
                        {
                            conversations::handle_remove_tags(state, req, auth).await
                        }
                        // Document management endpoints
                        (Method::GET, "/v1/documents") | (Method::GET, "/documents") => {
                            documents::handle_list(state, req, auth, plugin_owner).await
                        }
                        (Method::POST, "/v1/documents") | (Method::POST, "/documents") => {
                            documents::handle_create(state, req, auth, plugin_owner).await
                        }
                        // Stateless file inspect/transform (no storage required)
                        (Method::POST, "/v1/file/inspect") | (Method::POST, "/file/inspect") => {
                            file::handle_inspect(state, req, auth).await
                        }
                        (Method::POST, "/v1/file/transform")
                        | (Method::POST, "/file/transform") => {
                            file::handle_transform(state, req, auth).await
                        }
                        // Batch must come before single-file routes
                        (Method::POST, "/v1/files/batch")
                        | (Method::POST, "/files/batch") => {
                            files::handle_batch(state, req, auth).await
                        }
                        (Method::POST, "/v1/files") | (Method::POST, "/files") => {
                            files::handle_upload(state, req, auth).await
                        }
                        (Method::GET, "/v1/files") | (Method::GET, "/files") => {
                            files::handle_list(state, req, auth).await
                        }
                        (Method::GET, p)
                            if p.starts_with("/v1/blobs/") || p.starts_with("/blobs/") =>
                        {
                            files::handle_get_blob(state, req, auth).await
                        }
                        (Method::GET, p)
                            if (p.starts_with("/v1/files/") || p.starts_with("/files/"))
                                && p.ends_with("/content") =>
                        {
                            files::handle_get_content(state, req, auth).await
                        }
                        (Method::GET, p)
                            if (p.starts_with("/v1/files/") || p.starts_with("/files/"))
                                && !p.ends_with("/content") =>
                        {
                            files::handle_get(state, req, auth).await
                        }
                        (Method::PATCH, p)
                            if p.starts_with("/v1/files/") || p.starts_with("/files/") =>
                        {
                            files::handle_patch(state, req, auth).await
                        }
                        (Method::DELETE, p)
                            if p.starts_with("/v1/files/") || p.starts_with("/files/") =>
                        {
                            files::handle_delete(state, req, auth).await
                        }
                        // Code analysis endpoints (tree-sitter)
                        (Method::POST, "/v1/code/highlight")
                        | (Method::POST, "/code/highlight") => {
                            code::handle_highlight(state, req, auth).await
                        }
                        (Method::POST, "/v1/code/parse")
                        | (Method::POST, "/code/parse") => {
                            code::handle_parse(state, req, auth).await
                        }
                        (Method::POST, "/v1/code/query")
                        | (Method::POST, "/code/query") => {
                            code::handle_query(state, req, auth).await
                        }
                        (Method::POST, "/v1/code/graph")
                        | (Method::POST, "/code/graph") => {
                            code::handle_graph(state, req, auth).await
                        }
                        (Method::GET, "/v1/code/languages")
                        | (Method::GET, "/code/languages") => {
                            code::handle_languages(state, req, auth).await
                        }
                        // Code session endpoints (incremental parsing)
                        (Method::POST, "/v1/code/session/create")
                        | (Method::POST, "/code/session/create") => {
                            code::handle_session_create(state, req, auth).await
                        }
                        (Method::POST, "/v1/code/session/update")
                        | (Method::POST, "/code/session/update") => {
                            code::handle_session_update(state, req, auth).await
                        }
                        (Method::POST, "/v1/code/session/highlight")
                        | (Method::POST, "/code/session/highlight") => {
                            code::handle_session_highlight(state, req, auth).await
                        }
                        (Method::DELETE, p)
                            if p.starts_with("/v1/code/session/")
                                || p.starts_with("/code/session/") =>
                        {
                            code::handle_session_delete(state, req, auth).await
                        }
                        // WebSocket upgrade for real-time code analysis
                        (Method::GET, "/v1/code/ws") | (Method::GET, "/code/ws")
                            if req.headers().get("upgrade")
                                .and_then(|v| v.to_str().ok())
                                .is_some_and(|v| v.eq_ignore_ascii_case("websocket")) =>
                        {
                            let key = match req.headers().get("sec-websocket-key") {
                                Some(k) => k.as_bytes().to_vec(),
                                None => {
                                    return Ok(json_error(
                                        StatusCode::BAD_REQUEST,
                                        "invalid_request",
                                        "Missing Sec-WebSocket-Key header",
                                    ));
                                }
                            };
                            let accept = code_ws::compute_accept_key(&key);

                            let upgrade = hyper::upgrade::on(req);
                            tokio::spawn(async move {
                                match upgrade.await {
                                    Ok(upgraded) => {
                                        log::info!(target: "server::code_ws", "WebSocket connection established");
                                        code_ws::handle_ws_connection(upgraded).await;
                                    }
                                    Err(e) => {
                                        log::error!(target: "server::code_ws", "WebSocket upgrade failed: {e}");
                                    }
                                }
                            });

                            Response::builder()
                                .status(StatusCode::SWITCHING_PROTOCOLS)
                                .header("upgrade", "websocket")
                                .header("connection", "Upgrade")
                                .header("sec-websocket-accept", accept)
                                .body(Full::new(Bytes::new()).boxed())
                                .unwrap()
                        }
                        (Method::POST, "/v1/documents/search")
                        | (Method::POST, "/documents/search") => {
                            documents::handle_search(state, req, auth).await
                        }
                        (Method::GET, p)
                            if (p.starts_with("/v1/documents/")
                                || p.starts_with("/documents/"))
                                && !p.ends_with("/tags") =>
                        {
                            documents::handle_get(state, req, auth).await
                        }
                        (Method::PATCH, p)
                            if (p.starts_with("/v1/documents/")
                                || p.starts_with("/documents/"))
                                && !p.ends_with("/tags") =>
                        {
                            documents::handle_update(state, req, auth).await
                        }
                        (Method::DELETE, p)
                            if (p.starts_with("/v1/documents/")
                                || p.starts_with("/documents/"))
                                && !p.ends_with("/tags") =>
                        {
                            documents::handle_delete(state, req, auth).await
                        }
                        // Document tag endpoints
                        (Method::GET, p)
                            if (p.starts_with("/v1/documents/")
                                || p.starts_with("/documents/"))
                                && p.ends_with("/tags") =>
                        {
                            documents::handle_get_tags(state, req, auth).await
                        }
                        (Method::POST, p)
                            if (p.starts_with("/v1/documents/")
                                || p.starts_with("/documents/"))
                                && p.ends_with("/tags") =>
                        {
                            documents::handle_add_tags(state, req, auth).await
                        }
                        (Method::DELETE, p)
                            if (p.starts_with("/v1/documents/")
                                || p.starts_with("/documents/"))
                                && p.ends_with("/tags") =>
                        {
                            documents::handle_remove_tags(state, req, auth).await
                        }
                        // Plugin discovery
                        (Method::GET, "/v1/plugins") | (Method::GET, "/plugins") => {
                            plugins::handle_list(state, req, auth).await
                        }
                        // Repository management endpoints
                        (Method::GET, "/v1/repo/models") | (Method::GET, "/repo/models") => {
                            repo::handle_list(state, req, auth).await
                        }
                        (Method::GET, "/v1/repo/search") | (Method::GET, "/repo/search") => {
                            repo::handle_search(state, req, auth).await
                        }
                        (Method::POST, "/v1/repo/models") | (Method::POST, "/repo/models") => {
                            repo::handle_fetch(state, req, auth).await
                        }
                        (Method::DELETE, p)
                            if p.starts_with("/v1/repo/models/")
                                || p.starts_with("/repo/models/") =>
                        {
                            let prefix = if p.starts_with("/v1") {
                                "/v1/repo/models/"
                            } else {
                                "/repo/models/"
                            };
                            let raw = &p[prefix.len()..];
                            let model_id = percent_encoding::percent_decode_str(raw)
                                .decode_utf8_lossy();
                            repo::handle_delete(state, req, auth, &model_id).await
                        }
                        // Pin management endpoints
                        (Method::GET, "/v1/repo/pins") | (Method::GET, "/repo/pins") => {
                            repo::handle_list_pins(state, req, auth).await
                        }
                        (Method::POST, "/v1/repo/pins") | (Method::POST, "/repo/pins") => {
                            repo::handle_pin(state, req, auth).await
                        }
                        (Method::DELETE, p)
                            if p.starts_with("/v1/repo/pins/")
                                || p.starts_with("/repo/pins/") =>
                        {
                            let prefix = if p.starts_with("/v1") {
                                "/v1/repo/pins/"
                            } else {
                                "/repo/pins/"
                            };
                            let raw = &p[prefix.len()..];
                            let model_id = percent_encoding::percent_decode_str(raw)
                                .decode_utf8_lossy();
                            repo::handle_unpin(state, req, auth, &model_id).await
                        }
                        (Method::POST, "/v1/repo/sync-pins")
                        | (Method::POST, "/repo/sync-pins") => {
                            repo::handle_sync_pins(state, req, auth).await
                        }
                        // Proxy endpoint (plugin outbound HTTP)
                        (Method::POST, "/v1/proxy") | (Method::POST, "/proxy") => {
                            proxy::handle_proxy(state, req, auth).await
                        }
                        _ => {
                            if is_known_path(&path) {
                                log::warn!(target: "server::http", "unimplemented endpoint: {} {}", method_log, path_log);
                                json_error(
                                    StatusCode::NOT_IMPLEMENTED,
                                    "not_implemented",
                                    "Not implemented",
                                )
                            } else {
                                Response::builder()
                                    .status(StatusCode::NOT_FOUND)
                                    .body(Full::new(Bytes::from_static(b"not found")).boxed())
                                    .unwrap()
                            }
                        }
                    }
                }
            };

            let status = response.status();
            if status.is_client_error() {
                log::warn!(target: "server::http", "{} {} -> {}", method_log, path_log, status);
            } else if status.is_server_error() {
                log::error!(target: "server::http", "{} {} -> {}", method_log, path_log, status);
            }

            Ok(response)
        })
    }
}

/// Authenticate the request using gateway auth (if configured).
///
/// Returns `Ok(None)` when gateway auth is not enabled (no secret configured).
/// Returns `Ok(Some(ctx))` on successful authentication.
/// Returns `Err(response)` on auth failure.
fn authenticate(
    state: &AppState,
    headers: &hyper::HeaderMap,
) -> Result<Option<AuthContext>, Response<BoxBody>> {
    use crate::server::auth_gateway::{self, AuthError};

    let secret = match state.gateway_secret.as_deref() {
        Some(secret) => secret,
        None => return Ok(None),
    };
    let registry = match state.tenant_registry.as_ref() {
        Some(registry) => registry,
        None => {
            return Err(json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "Gateway auth enabled but tenant registry is missing",
            ))
        }
    };

    match auth_gateway::validate_request(headers, secret, registry) {
        Ok(ctx) => Ok(Some(ctx)),
        Err(AuthError::MissingSecret) => Err(json_error(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "Missing gateway secret",
        )),
        Err(AuthError::InvalidSecret) => Err(json_error(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "Invalid gateway secret",
        )),
        Err(AuthError::MissingTenant) => Err(json_error(
            StatusCode::FORBIDDEN,
            "forbidden",
            "Missing tenant id",
        )),
        Err(AuthError::UnknownTenant) => Err(json_error(
            StatusCode::FORBIDDEN,
            "forbidden",
            "Unknown tenant id",
        )),
    }
}

fn is_known_path(path: &str) -> bool {
    if KNOWN_PATHS.contains(path) {
        return true;
    }
    if let Some(stripped) = path.strip_prefix("/v1") {
        return KNOWN_PATHS.contains(stripped);
    }
    false
}

/// Serve a console UI asset.
///
/// Resolution order:
/// 1. `--html-dir <dir>` → read `<dir>/<filename>` from disk
/// 2. Bundled assets (when built with `make ui` / `cfg(bundled_ui)`)
/// 3. 404 with a helpful message
fn serve_ui_asset(state: &AppState, filename: &str, content_type: &str) -> Response<BoxBody> {
    // --html-dir takes precedence over bundled assets.
    if let Some(ref dir) = state.html_dir {
        return serve_file(dir, filename, content_type);
    }

    // Fall back to bundled assets (compiled in when ui/dist/ exists).
    #[cfg(bundled_ui)]
    {
        if let Some(data) = bundled_asset(filename) {
            return static_response(content_type, data);
        }
    }

    json_error(
        StatusCode::NOT_FOUND,
        "ui_not_available",
        "No UI bundled. Run 'make ui' or use --html-dir",
    )
}

/// Return bundled asset bytes by filename.
#[cfg(bundled_ui)]
fn bundled_asset(filename: &str) -> Option<&'static [u8]> {
    match filename {
        "index.html" => Some(CONSOLE_HTML),
        "style.css" => Some(CONSOLE_CSS),
        "main.js" => Some(CONSOLE_JS),
        _ => None,
    }
}

/// Serve a file from a directory on disk.
fn serve_file(dir: &Path, filename: &str, content_type: &str) -> Response<BoxBody> {
    let path = dir.join(filename);
    match std::fs::read(&path) {
        Ok(data) => Response::builder()
            .status(StatusCode::OK)
            .header("content-type", content_type)
            .header("referrer-policy", "no-referrer")
            .body(Full::new(Bytes::from(data)).boxed())
            .unwrap(),
        Err(_) => json_error(
            StatusCode::NOT_FOUND,
            "not_found",
            &format!("Asset not found: {}", filename),
        ),
    }
}

#[cfg(bundled_ui)]
fn static_response(content_type: &str, body: &'static [u8]) -> Response<BoxBody> {
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", content_type)
        .header("referrer-policy", "no-referrer")
        .body(Full::new(Bytes::from_static(body)).boxed())
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
