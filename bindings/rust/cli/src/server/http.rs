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
use crate::server::db;
use crate::server::events;
use crate::server::file;
use crate::server::files;
use crate::server::handlers;
use crate::server::openapi;
use crate::server::plugins;
use crate::server::projects;
use crate::server::proxy;
use crate::server::repo;
use crate::server::responses;
use crate::server::search;
use crate::server::sessions;
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
static OPENAPI_CHAT_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/chat/"]));
static OPENAPI_RESPONSES_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/responses"]));
static OPENAPI_MODELS_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/models"]));
static OPENAPI_FILES_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/files", "/v1/file"]));
static OPENAPI_REPO_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/repo"]));
static OPENAPI_SEARCH_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/search"]));
static OPENAPI_TAGS_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/tags"]));
static OPENAPI_SETTINGS_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/settings"]));
static OPENAPI_PLUGINS_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/plugins", "/v1/proxy"]));
static OPENAPI_CODE_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/code"]));
static OPENAPI_EVENTS_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/events"]));
static OPENAPI_PROJECTS_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/projects"]));
static OPENAPI_DB_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/db/"]));
static OPENAPI_DB_TABLES_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/db/tables/"]));
static OPENAPI_DB_VECTORS_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/db/vectors/"]));
static OPENAPI_DB_KV_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/db/kv/"]));
static OPENAPI_DB_BLOBS_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/db/blobs"]));
static OPENAPI_DB_SQL_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/db/sql/"]));
static OPENAPI_DB_OPS_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/db/ops/"]));

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

            let content_length = req
                .headers()
                .get("content-length")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("-");
            let content_type = req
                .headers()
                .get("content-type")
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
                (Method::GET, "/openapi/chat.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_CHAT_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/openapi/responses.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_RESPONSES_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/openapi/models.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_MODELS_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/openapi/files.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_FILES_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/openapi/repo.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_REPO_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/openapi/search.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_SEARCH_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/openapi/tags.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_TAGS_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/openapi/settings.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_SETTINGS_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/openapi/plugins.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_PLUGINS_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/openapi/code.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_CODE_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/openapi/events.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_EVENTS_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/openapi/projects.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_PROJECTS_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/openapi/db.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_DB_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/openapi/db/tables.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_DB_TABLES_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/openapi/db/vectors.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_DB_VECTORS_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/openapi/db/kv.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_DB_KV_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/openapi/db/blobs.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_DB_BLOBS_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/openapi/db/sql.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_DB_SQL_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/openapi/db/ops.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_DB_OPS_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/docs") => docs_hub_response(),
                (Method::GET, "/docs/chat") => {
                    swagger_ui_response("/openapi/chat.json", "Talu API :: Chat")
                }
                (Method::GET, "/docs/responses") => {
                    swagger_ui_response("/openapi/responses.json", "Talu API :: Responses")
                }
                (Method::GET, "/docs/models") => {
                    swagger_ui_response("/openapi/models.json", "Talu API :: Models")
                }
                (Method::GET, "/docs/files") => {
                    swagger_ui_response("/openapi/files.json", "Talu API :: Files")
                }
                (Method::GET, "/docs/repo") => {
                    swagger_ui_response("/openapi/repo.json", "Talu API :: Repository")
                }
                (Method::GET, "/docs/search") => {
                    swagger_ui_response("/openapi/search.json", "Talu API :: Search")
                }
                (Method::GET, "/docs/tags") => {
                    swagger_ui_response("/openapi/tags.json", "Talu API :: Tags")
                }
                (Method::GET, "/docs/settings") => {
                    swagger_ui_response("/openapi/settings.json", "Talu API :: Settings")
                }
                (Method::GET, "/docs/plugins") => {
                    swagger_ui_response("/openapi/plugins.json", "Talu API :: Plugins")
                }
                (Method::GET, "/docs/code") => {
                    swagger_ui_response("/openapi/code.json", "Talu API :: Code")
                }
                (Method::GET, "/docs/events") => {
                    swagger_ui_response("/openapi/events.json", "Talu API :: Events")
                }
                (Method::GET, "/docs/projects") => {
                    swagger_ui_response("/openapi/projects.json", "Talu API :: Projects")
                }
                (Method::GET, "/docs/db") => docs_hub_response(),
                (Method::GET, "/docs/db/tables") => {
                    swagger_ui_response("/openapi/db/tables.json", "Talu API :: DB::Tables")
                }
                (Method::GET, "/docs/db/vectors") => {
                    swagger_ui_response("/openapi/db/vectors.json", "Talu API :: DB::Vectors")
                }
                (Method::GET, "/docs/db/kv") => {
                    swagger_ui_response("/openapi/db/kv.json", "Talu API :: DB::KV")
                }
                (Method::GET, "/docs/db/blobs") => {
                    swagger_ui_response("/openapi/db/blobs.json", "Talu API :: DB::Blobs")
                }
                (Method::GET, "/docs/db/sql") => {
                    swagger_ui_response("/openapi/db/sql.json", "Talu API :: DB::SQL")
                }
                (Method::GET, "/docs/db/ops") => {
                    swagger_ui_response("/openapi/db/ops.json", "Talu API :: DB::Ops")
                }
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
                        (Method::POST, "/v1/responses") => {
                            responses::handle_create(state, req, auth).await
                        }
                        (Method::GET, "/v1/events") => {
                            events::handle_replay(state, req, auth).await
                        }
                        (Method::GET, "/v1/events/stream") => {
                            events::handle_stream(state, req, auth).await
                        }
                        (Method::GET, "/v1/events/topics") => {
                            events::handle_topics(state, req, auth).await
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
                        // Session management endpoints
                        (Method::GET, "/v1/chat/sessions") => {
                            sessions::handle_list(state, req, auth).await
                        }
                        // Batch operations (must be before single-session routes)
                        (Method::POST, "/v1/chat/sessions/batch") => {
                            sessions::handle_batch(state, req, auth).await
                        }
                        (Method::GET, p)
                            if p.starts_with("/v1/chat/sessions/") && !p.ends_with("/fork") =>
                        {
                            sessions::handle_get(state, req, auth).await
                        }
                        (Method::DELETE, p)
                            if p.starts_with("/v1/chat/sessions/") && !p.ends_with("/tags") =>
                        {
                            sessions::handle_delete(state, req, auth).await
                        }
                        (Method::PATCH, p) if p.starts_with("/v1/chat/sessions/") => {
                            sessions::handle_patch(state, req, auth).await
                        }
                        (Method::POST, p)
                            if p.ends_with("/fork") && p.starts_with("/v1/chat/sessions/") =>
                        {
                            sessions::handle_fork(state, req, auth).await
                        }
                        // Search endpoint
                        (Method::POST, "/v1/search") | (Method::POST, "/search") => {
                            search::handle_search(state, req, auth).await
                        }
                        // Project management endpoints
                        (Method::GET, "/v1/projects") | (Method::GET, "/projects") => {
                            projects::handle_list(state, req, auth).await
                        }
                        (Method::POST, "/v1/projects") | (Method::POST, "/projects") => {
                            projects::handle_create(state, req, auth).await
                        }
                        (Method::GET, p)
                            if p.starts_with("/v1/projects/") || p.starts_with("/projects/") =>
                        {
                            projects::handle_get(state, req, auth).await
                        }
                        (Method::PATCH, p)
                            if p.starts_with("/v1/projects/") || p.starts_with("/projects/") =>
                        {
                            projects::handle_update(state, req, auth).await
                        }
                        (Method::DELETE, p)
                            if p.starts_with("/v1/projects/") || p.starts_with("/projects/") =>
                        {
                            projects::handle_delete(state, req, auth).await
                        }
                        // DB vector plane endpoints
                        (Method::POST, "/v1/db/vectors/collections") => {
                            db::vector::handle_create_collection(state, req, auth).await
                        }
                        (Method::GET, "/v1/db/vectors/collections") => {
                            db::vector::handle_list_collections(state, req, auth).await
                        }
                        (Method::POST, p)
                            if p.ends_with("/points/append")
                                && p.starts_with("/v1/db/vectors/collections/") =>
                        {
                            db::vector::handle_append_points(state, req, auth).await
                        }
                        (Method::POST, p)
                            if p.ends_with("/points/upsert")
                                && p.starts_with("/v1/db/vectors/collections/") =>
                        {
                            db::vector::handle_upsert_points(state, req, auth).await
                        }
                        (Method::POST, p)
                            if p.ends_with("/points/delete")
                                && p.starts_with("/v1/db/vectors/collections/") =>
                        {
                            db::vector::handle_delete_points(state, req, auth).await
                        }
                        (Method::POST, p)
                            if p.ends_with("/points/fetch")
                                && p.starts_with("/v1/db/vectors/collections/") =>
                        {
                            db::vector::handle_fetch_points(state, req, auth).await
                        }
                        (Method::POST, p)
                            if p.ends_with("/points/query")
                                && p.starts_with("/v1/db/vectors/collections/") =>
                        {
                            db::vector::handle_query_points(state, req, auth).await
                        }
                        (Method::POST, p)
                            if p.ends_with("/indexes/build")
                                && p.starts_with("/v1/db/vectors/collections/") =>
                        {
                            db::vector::handle_build_collection_indexes(state, req, auth).await
                        }
                        (Method::POST, p)
                            if p.ends_with("/compact")
                                && p.starts_with("/v1/db/vectors/collections/") =>
                        {
                            db::vector::handle_compact_collection(state, req, auth).await
                        }
                        (Method::GET, p)
                            if p.ends_with("/stats")
                                && p.starts_with("/v1/db/vectors/collections/") =>
                        {
                            db::vector::handle_collection_stats(state, req, auth).await
                        }
                        (Method::GET, p)
                            if p.ends_with("/changes")
                                && p.starts_with("/v1/db/vectors/collections/") =>
                        {
                            db::vector::handle_collection_changes(state, req, auth).await
                        }
                        (Method::GET, p)
                            if p.starts_with("/v1/db/vectors/collections/")
                                && !is_dynamic_vector_subpath(p) =>
                        {
                            db::vector::handle_get_collection(state, req, auth).await
                        }
                        (Method::DELETE, p)
                            if p.starts_with("/v1/db/vectors/collections/")
                                && !is_dynamic_vector_subpath(p) =>
                        {
                            db::vector::handle_delete_collection(state, req, auth).await
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
                        // Session tag endpoints
                        (Method::GET, p)
                            if p.starts_with("/v1/chat/sessions/") && p.ends_with("/tags") =>
                        {
                            sessions::handle_get_tags(state, req, auth).await
                        }
                        (Method::POST, p)
                            if p.starts_with("/v1/chat/sessions/") && p.ends_with("/tags") =>
                        {
                            sessions::handle_add_tags(state, req, auth).await
                        }
                        (Method::PUT, p)
                            if p.starts_with("/v1/chat/sessions/") && p.ends_with("/tags") =>
                        {
                            sessions::handle_set_tags(state, req, auth).await
                        }
                        (Method::DELETE, p)
                            if p.starts_with("/v1/chat/sessions/") && p.ends_with("/tags") =>
                        {
                            sessions::handle_remove_tags(state, req, auth).await
                        }
                        // Table meta endpoints
                        (Method::GET, "/v1/db/tables/_meta/namespaces") => {
                            db::table::handle_list_namespaces(state, req, auth).await
                        }
                        (Method::GET, p) if db::table::is_table_meta_policy_path(p) => {
                            db::table::handle_get_policy(state, req, auth).await
                        }
                        // Document/table plane endpoints
                        (Method::GET, p) if is_db_table_root_path(p) => {
                            db::docs::handle_list(state, req, auth, plugin_owner).await
                        }
                        (Method::POST, p)
                            if is_db_table_root_path(p) || is_db_table_insert_path(p) =>
                        {
                            db::docs::handle_create(state, req, auth, plugin_owner).await
                        }
                        (Method::POST, p) if is_db_table_search_path(p) => {
                            db::docs::handle_search(state, req, auth).await
                        }
                        (Method::GET, p) if is_db_table_tags_path(p) => {
                            db::docs::handle_get_tags(state, req, auth).await
                        }
                        (Method::POST, p) if is_db_table_tags_path(p) => {
                            db::docs::handle_add_tags(state, req, auth).await
                        }
                        (Method::DELETE, p) if is_db_table_tags_path(p) => {
                            db::docs::handle_remove_tags(state, req, auth).await
                        }
                        // Table rows endpoints (must come before item-path match)
                        (Method::POST, p) if db::table::is_table_scan_path(p) => {
                            db::table::handle_scan_post(state, req, auth).await
                        }
                        (Method::POST, p) if db::table::is_table_rows_path(p) => {
                            db::table::handle_write_row(state, req, auth).await
                        }
                        (Method::GET, p) if db::table::is_table_row_path(p) => {
                            db::table::handle_get_row(state, req, auth).await
                        }
                        (Method::GET, p) if db::table::is_table_rows_path(p) => {
                            db::table::handle_scan(state, req, auth).await
                        }
                        (Method::DELETE, p) if db::table::is_table_row_path(p) => {
                            db::table::handle_delete_row(state, req, auth).await
                        }
                        (Method::GET, p) if is_db_table_item_path(p) => {
                            db::docs::handle_get(state, req, auth).await
                        }
                        (Method::PATCH, p) if is_db_table_item_path(p) => {
                            db::docs::handle_update(state, req, auth).await
                        }
                        (Method::DELETE, p) if is_db_table_item_path(p) => {
                            db::docs::handle_delete(state, req, auth).await
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
                        (Method::POST, "/v1/files/batch") | (Method::POST, "/files/batch") => {
                            files::handle_batch(state, req, auth).await
                        }
                        (Method::POST, "/v1/files") | (Method::POST, "/files") => {
                            files::handle_upload(state, req, auth).await
                        }
                        (Method::GET, "/v1/files") | (Method::GET, "/files") => {
                            files::handle_list(state, req, auth).await
                        }
                        (Method::GET, "/v1/db/blobs") => {
                            db::blob::handle_list(state, req, auth).await
                        }
                        (Method::GET, p) if p.starts_with("/v1/db/blobs/") => {
                            db::blob::handle_get(state, req, auth).await
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
                        (Method::POST, "/v1/code/parse") | (Method::POST, "/code/parse") => {
                            code::handle_parse(state, req, auth).await
                        }
                        (Method::POST, "/v1/code/query") | (Method::POST, "/code/query") => {
                            code::handle_query(state, req, auth).await
                        }
                        (Method::POST, "/v1/code/graph") | (Method::POST, "/code/graph") => {
                            code::handle_graph(state, req, auth).await
                        }
                        (Method::GET, "/v1/code/languages") | (Method::GET, "/code/languages") => {
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
                            if req
                                .headers()
                                .get("upgrade")
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
                        // DB KV plane endpoints
                        (Method::GET, p)
                            if p.starts_with("/v1/db/kv/namespaces/")
                                && p.ends_with("/entries") =>
                        {
                            db::kv::handle_list(state, req, auth).await
                        }
                        (Method::PUT, p)
                            if p.starts_with("/v1/db/kv/namespaces/")
                                && p.contains("/entries/") =>
                        {
                            db::kv::handle_put(state, req, auth).await
                        }
                        (Method::GET, p)
                            if p.starts_with("/v1/db/kv/namespaces/")
                                && p.contains("/entries/") =>
                        {
                            db::kv::handle_get(state, req, auth).await
                        }
                        (Method::DELETE, p)
                            if p.starts_with("/v1/db/kv/namespaces/")
                                && p.contains("/entries/") =>
                        {
                            db::kv::handle_delete(state, req, auth).await
                        }
                        (Method::POST, p)
                            if p.starts_with("/v1/db/kv/namespaces/") && p.ends_with("/flush") =>
                        {
                            db::kv::handle_flush(state, req, auth).await
                        }
                        (Method::POST, p)
                            if p.starts_with("/v1/db/kv/namespaces/")
                                && p.ends_with("/compact") =>
                        {
                            db::kv::handle_compact(state, req, auth).await
                        }
                        // DB SQL plane endpoints
                        (Method::POST, "/v1/db/sql/query") => {
                            db::sql::handle_query(state, req, auth).await
                        }
                        (Method::POST, "/v1/db/sql/explain") => {
                            db::sql::handle_explain(state, req, auth).await
                        }
                        // DB ops plane endpoints
                        (Method::POST, p) if p.starts_with("/v1/db/ops/") => {
                            db::ops::handle(state, req, auth).await
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
                        // File listing (must come before DELETE /repo/models/{id})
                        (Method::GET, p)
                            if (p.starts_with("/v1/repo/models/")
                                || p.starts_with("/repo/models/"))
                                && p.ends_with("/files") =>
                        {
                            let prefix = if p.starts_with("/v1") {
                                "/v1/repo/models/"
                            } else {
                                "/repo/models/"
                            };
                            let raw = &p[prefix.len()..p.len() - "/files".len()];
                            let model_id =
                                percent_encoding::percent_decode_str(raw).decode_utf8_lossy();
                            repo::handle_list_files(state, req, auth, &model_id).await
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
                            let model_id =
                                percent_encoding::percent_decode_str(raw).decode_utf8_lossy();
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
                            if p.starts_with("/v1/repo/pins/") || p.starts_with("/repo/pins/") =>
                        {
                            let prefix = if p.starts_with("/v1") {
                                "/v1/repo/pins/"
                            } else {
                                "/repo/pins/"
                            };
                            let raw = &p[prefix.len()..];
                            let model_id =
                                percent_encoding::percent_decode_str(raw).decode_utf8_lossy();
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

fn is_db_table_root_path(path: &str) -> bool {
    let Some(stripped) = path.strip_prefix("/v1/db/tables/") else {
        return false;
    };
    !stripped.is_empty() && !stripped.contains('/')
}

fn is_db_table_insert_path(path: &str) -> bool {
    let Some(stripped) = path.strip_prefix("/v1/db/tables/") else {
        return false;
    };
    let mut parts = stripped.split('/');
    let table = parts.next().unwrap_or("");
    let action = parts.next().unwrap_or("");
    table != "" && action == "insert" && parts.next().is_none()
}

fn is_db_table_search_path(path: &str) -> bool {
    let Some(stripped) = path.strip_prefix("/v1/db/tables/") else {
        return false;
    };
    let mut parts = stripped.split('/');
    let table = parts.next().unwrap_or("");
    let action = parts.next().unwrap_or("");
    table != "" && action == "search" && parts.next().is_none()
}

fn is_db_table_item_path(path: &str) -> bool {
    let Some(stripped) = path.strip_prefix("/v1/db/tables/") else {
        return false;
    };
    let mut parts = stripped.split('/');
    let table = parts.next().unwrap_or("");
    let id = parts.next().unwrap_or("");
    table != ""
        && id != ""
        && id != "search"
        && id != "insert"
        && id != "rows"
        && id != "_meta"
        && !(table == "_meta" && (id == "namespaces" || id == "policy"))
        && parts.next().is_none()
}

fn is_db_table_tags_path(path: &str) -> bool {
    let Some(stripped) = path.strip_prefix("/v1/db/tables/") else {
        return false;
    };
    let mut parts = stripped.split('/');
    let table = parts.next().unwrap_or("");
    let id = parts.next().unwrap_or("");
    let tags = parts.next().unwrap_or("");
    table != "" && id != "" && tags == "tags" && parts.next().is_none()
}

fn is_known_path(path: &str) -> bool {
    if KNOWN_PATHS.contains(path) {
        return true;
    }
    if let Some(stripped) = path.strip_prefix("/v1") {
        if KNOWN_PATHS.contains(stripped) {
            return true;
        }
    }
    // Parameterized paths not in KNOWN_PATHS as literals.
    is_kv_entry_path(path)
        || is_kv_state_op_path(path)
        || is_db_table_root_path(path)
        || is_db_table_item_path(path)
        || is_db_table_insert_path(path)
        || is_db_table_search_path(path)
        || is_db_table_tags_path(path)
        || db::table::is_table_rows_path(path)
        || db::table::is_table_row_path(path)
        || db::table::is_table_scan_path(path)
        || db::table::is_table_meta_policy_path(path)
        || is_dynamic_vector_path(path)
        || is_blob_item_path(path)
}

fn is_kv_entry_path(path: &str) -> bool {
    let p = path.strip_prefix("/v1").unwrap_or(path);
    p.starts_with("/db/kv/namespaces/") && p.contains("/entries")
}

fn is_kv_state_op_path(path: &str) -> bool {
    let Some(stripped) = path.strip_prefix("/v1/db/kv/namespaces/") else {
        return false;
    };
    stripped.ends_with("/flush") || stripped.ends_with("/compact")
}

/// True when path is a known vector sub-path (points/*, stats, compact, etc.)
/// as opposed to a simple collection-name path.  Used to prevent GET/DELETE
/// catch-alls from absorbing wrong-method requests on known sub-paths.
fn is_dynamic_vector_subpath(path: &str) -> bool {
    let Some(stripped) = path.strip_prefix("/v1/db/vectors/collections/") else {
        return false;
    };
    let parts: Vec<&str> = stripped.split('/').collect();
    match parts.len() {
        2 => matches!(parts[1], "stats" | "compact" | "changes"),
        3 => {
            (parts[1] == "points"
                && matches!(parts[2], "append" | "upsert" | "delete" | "fetch" | "query"))
                || (parts[1] == "indexes" && parts[2] == "build")
        }
        _ => false,
    }
}

fn is_dynamic_vector_path(path: &str) -> bool {
    let Some(stripped) = path.strip_prefix("/v1/db/vectors/collections/") else {
        return false;
    };
    if stripped.is_empty() {
        return false;
    }
    let parts: Vec<&str> = stripped.split('/').collect();
    match parts.len() {
        // /v1/db/vectors/collections/{name}
        1 => true,
        // /v1/db/vectors/collections/{name}/{suffix}
        2 => matches!(parts[1], "stats" | "compact" | "changes"),
        // /v1/db/vectors/collections/{name}/points/{action} or indexes/build
        3 => {
            (parts[1] == "points"
                && matches!(parts[2], "append" | "upsert" | "delete" | "fetch" | "query"))
                || (parts[1] == "indexes" && parts[2] == "build")
        }
        _ => false,
    }
}

fn is_blob_item_path(path: &str) -> bool {
    let Some(stripped) = path.strip_prefix("/v1/db/blobs/") else {
        return false;
    };
    !stripped.is_empty() && !stripped.contains('/')
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

fn swagger_ui_response(spec_url: &str, title: &str) -> Response<BoxBody> {
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/html; charset=utf-8")
        .body(Full::new(Bytes::from(swagger_ui_html(spec_url, title))).boxed())
        .unwrap()
}

fn docs_hub_response() -> Response<BoxBody> {
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/html; charset=utf-8")
        .body(Full::new(Bytes::from(docs_hub_html())).boxed())
        .unwrap()
}

fn swagger_ui_html(spec_url: &str, title: &str) -> String {
    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css"/>
<style>
body {{ margin: 0; }}
.talu-docs-nav {{
  display: flex;
  align-items: center;
  padding: 0.75rem 1rem;
  border-bottom: 1px solid #e5e7eb;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}}
.talu-docs-nav a {{
  font-weight: 700;
  color: #0f172a;
  text-decoration: none;
}}
.talu-docs-nav a:hover {{ text-decoration: underline; }}
.swagger-ui .info {{ display: none; }}
</style>
</head>
<body>
<div class="talu-docs-nav">
  <a href="/docs">Docs Home</a>
</div>
<div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>SwaggerUIBundle({{url:"{spec_url}",dom_id:"#swagger-ui"}});</script>
</body>
</html>
"##
    )
}

fn docs_hub_html() -> String {
    r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Talu Docs</title>
<style>
body {
  font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  margin: 0;
  background: #f8fafc;
  color: #0f172a;
}
.page {
  max-width: 78rem;
  margin: 0 auto;
  padding: 2rem 1.25rem 2.5rem;
}
h1 {
  margin: 0 0 0.4rem;
  font-size: 1.9rem;
}
.muted {
  color: #475569;
  margin: 0 0 1.2rem;
}
.table-wrap {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  overflow: hidden;
  margin-bottom: 1rem;
}
table {
  width: 100%;
  border-collapse: collapse;
}
th, td {
  text-align: left;
  vertical-align: top;
  padding: 0.62rem 0.8rem;
  border-bottom: 1px solid #e2e8f0;
  border-right: 1px solid #e2e8f0;
}
th:last-child, td:last-child {
  border-right: none;
}
th {
  background: #f1f5f9;
  color: #0f172a;
  font-weight: 700;
  font-size: 0.92rem;
}
th .header-link {
  font-weight: 600;
  margin-left: 0;
}
tbody tr:last-child td {
  border-bottom: none;
}
td {
  font-size: 0.95rem;
}
td.desc {
  color: #334155;
}
.json-cell {
  white-space: nowrap;
  width: 8.5rem;
}
.mono {
  white-space: nowrap;
}
.json-link {
  display: inline-block;
  padding: 0.08rem 0.4rem;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  font-size: 0.82rem;
  font-weight: 600;
  letter-spacing: 0.01em;
}
.copy-btn {
  margin-left: 0.35rem;
  width: 1.55rem;
  height: 1.55rem;
  line-height: 1.35rem;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  background: #fff;
  color: #334155;
  cursor: pointer;
  font-size: 0.88rem;
}
.copy-btn:hover {
  background: #f8fafc;
}
.copy-btn.copied {
  border-color: #16a34a;
  color: #166534;
}
code {
  background: #f1f5f9;
  padding: 0.1rem 0.32rem;
  border-radius: 4px;
}
a {
  color: #0f172a;
  text-decoration: none;
}
a:hover {
  text-decoration: underline;
}
</style>
</head>
<body>
<main class="page">
  <h1>Talu API Docs</h1>
  <p class="muted">Single index for interactive docs and scoped OpenAPI JSON contracts.</p>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th><a class="header-link" href="/docs"><code>/docs</code></a></th>
          <th class="json-cell"><a class="json-link" href="/openapi.json" title="/openapi.json">json</a><button class="copy-btn" data-url="/openapi.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="mono"><a href="/docs/chat"><code>chat</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/chat.json" title="/openapi/chat.json">json</a><button class="copy-btn" data-url="/openapi/chat.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td class="desc">Chat API plane (`/v1/chat/*`) for conversational session management.</td>
        </tr>
        <tr>
          <td class="mono"><a href="/docs/responses"><code>responses</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/responses.json" title="/openapi/responses.json">json</a><button class="copy-btn" data-url="/openapi/responses.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td class="desc">OpenResponses-compatible API surface (`/v1/responses`).</td>
        </tr>
        <tr>
          <td class="mono"><a href="/docs/models"><code>models</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/models.json" title="/openapi/models.json">json</a><button class="copy-btn" data-url="/openapi/models.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td class="desc">Model discovery and listing endpoints.</td>
        </tr>
        <tr>
          <td class="mono"><a href="/docs/files"><code>files</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/files.json" title="/openapi/files.json">json</a><button class="copy-btn" data-url="/openapi/files.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td class="desc">File upload/list/get plus stateless inspect and transform APIs.</td>
        </tr>
        <tr>
          <td class="mono"><a href="/docs/repo"><code>repo</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/repo.json" title="/openapi/repo.json">json</a><button class="copy-btn" data-url="/openapi/repo.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td class="desc">Repository model management, pin lifecycle, and sync endpoints.</td>
        </tr>
        <tr>
          <td class="mono"><a href="/docs/search"><code>search</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/search.json" title="/openapi/search.json">json</a><button class="copy-btn" data-url="/openapi/search.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td class="desc">Cross-domain search with text and filter capabilities.</td>
        </tr>
        <tr>
          <td class="mono"><a href="/docs/tags"><code>tags</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/tags.json" title="/openapi/tags.json">json</a><button class="copy-btn" data-url="/openapi/tags.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td class="desc">Tag CRUD and tag-association endpoints.</td>
        </tr>
        <tr>
          <td class="mono"><a href="/docs/settings"><code>settings</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/settings.json" title="/openapi/settings.json">json</a><button class="copy-btn" data-url="/openapi/settings.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td class="desc">Server and model-default settings APIs.</td>
        </tr>
        <tr>
          <td class="mono"><a href="/docs/plugins"><code>plugins</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/plugins.json" title="/openapi/plugins.json">json</a><button class="copy-btn" data-url="/openapi/plugins.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td class="desc">Plugin discovery, plugin assets, and proxy operations.</td>
        </tr>
        <tr>
          <td class="mono"><a href="/docs/code"><code>code</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/code.json" title="/openapi/code.json">json</a><button class="copy-btn" data-url="/openapi/code.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td class="desc">Tree-sitter parse, highlight, query, graph, and code-session APIs.</td>
        </tr>
        <tr>
          <td class="mono"><a href="/docs/events"><code>events</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/events.json" title="/openapi/events.json">json</a><button class="copy-btn" data-url="/openapi/events.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td class="desc">Unified in-memory observability stream and replay APIs (`/v1/events*`).</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th><a class="header-link" href="/docs/db"><code>/docs/db</code></a></th>
          <th class="json-cell"><a class="json-link" href="/openapi/db.json" title="/openapi/db.json">json</a><button class="copy-btn" data-url="/openapi/db.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="mono"><a href="/docs/db/tables"><code>tables</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/db/tables.json" title="/openapi/db/tables.json">json</a><button class="copy-btn" data-url="/openapi/db/tables.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td class="desc">Table-plane CRUD/search routes (currently documents-backed).</td>
        </tr>
        <tr>
          <td class="mono"><a href="/docs/db/vectors"><code>vectors</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/db/vectors.json" title="/openapi/db/vectors.json">json</a><button class="copy-btn" data-url="/openapi/db/vectors.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td class="desc">Vector collection, points mutation/query, and index workflows.</td>
        </tr>
        <tr>
          <td class="mono"><a href="/docs/db/kv"><code>kv</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/db/kv.json" title="/openapi/db/kv.json">json</a><button class="copy-btn" data-url="/openapi/db/kv.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td class="desc">Generic namespaced key/value entry and maintenance operations.</td>
        </tr>
        <tr>
          <td class="mono"><a href="/docs/db/blobs"><code>blobs</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/db/blobs.json" title="/openapi/db/blobs.json">json</a><button class="copy-btn" data-url="/openapi/db/blobs.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td class="desc">Blob-plane listing and content retrieval endpoints.</td>
        </tr>
        <tr>
          <td class="mono"><a href="/docs/db/sql"><code>sql</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/db/sql.json" title="/openapi/db/sql.json">json</a><button class="copy-btn" data-url="/openapi/db/sql.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td class="desc">Compute SQL query plane over database-backed resources.</td>
        </tr>
        <tr>
          <td class="mono"><a href="/docs/db/ops"><code>ops</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/db/ops.json" title="/openapi/db/ops.json">json</a><button class="copy-btn" data-url="/openapi/db/ops.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td class="desc">Operational DB actions such as compaction and maintenance.</td>
        </tr>
      </tbody>
    </table>
  </div>
</main>
<script>
document.querySelectorAll('.copy-btn').forEach((button) => {
  button.addEventListener('click', async () => {
    const path = button.getAttribute('data-url') || '';
    if (!path) return;
    const url = `${window.location.origin}${path}`;
    try {
      await navigator.clipboard.writeText(url);
      button.classList.add('copied');
      const prev = button.textContent;
      button.textContent = '✓';
      setTimeout(() => {
        button.classList.remove('copied');
        button.textContent = prev;
      }, 1000);
    } catch {
      // Best-effort copy; no-op on clipboard errors.
    }
  });
});
</script>
</body>
</html>
"##
    .to_string()
}

fn filter_openapi_paths(spec: &[u8], prefixes: &[&str]) -> Vec<u8> {
    let mut doc: Value = match serde_json::from_slice(spec) {
        Ok(v) => v,
        Err(_) => return spec.to_vec(),
    };

    let Some(paths) = doc.get_mut("paths").and_then(Value::as_object_mut) else {
        return spec.to_vec();
    };

    paths.retain(|path, _| prefixes.iter().any(|prefix| path.starts_with(prefix)));

    let mut used_tags: HashSet<String> = HashSet::new();
    for item in paths.values() {
        let Some(item_obj) = item.as_object() else {
            continue;
        };
        for method in [
            "get", "post", "put", "patch", "delete", "head", "options", "trace",
        ] {
            let Some(operation) = item_obj.get(method) else {
                continue;
            };
            if let Some(tags) = operation.get("tags").and_then(Value::as_array) {
                for tag in tags {
                    if let Some(tag_name) = tag.as_str() {
                        let _ = used_tags.insert(tag_name.to_string());
                    }
                }
            }
        }
    }

    if let Some(tags) = doc.get_mut("tags").and_then(Value::as_array_mut) {
        tags.retain(|tag| {
            let Some(name) = tag.get("name").and_then(Value::as_str) else {
                return false;
            };
            used_tags.contains(name)
        });
    }

    serde_json::to_vec_pretty(&doc).unwrap_or_else(|_| spec.to_vec())
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
