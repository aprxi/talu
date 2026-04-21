use std::collections::HashSet;
use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::header::HeaderValue;
use hyper::{Method, Request, Response, StatusCode};
use once_cell::sync::Lazy;
use serde_json::Value;
use tower_service::Service;

use serde::Serialize;
use utoipa::ToSchema;

use crate::server::auth_gateway::AuthContext;
use crate::server::completions;
use crate::server::handlers;
use crate::server::openapi;
use crate::server::repo;
use crate::server::responses;
use crate::server::responses_openapi;
use crate::server::state::AppState;
use crate::server::tokenizer;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;
const TALU_INSTANCE: &str = "talu";
const TALU_API_VERSION: &str = "v1";
const TALU_VERSION: &str = env!("TALU_VERSION");
const CORS_ALLOW_ORIGIN: &str = "*";
const CORS_ALLOW_METHODS: &str = "GET, POST, PUT, PATCH, DELETE, OPTIONS";
const CORS_ALLOW_HEADERS_DEFAULT: &str =
    "authorization, content-type, x-talu-gateway-secret, x-talu-tenant-id, x-talu-group-id, x-talu-user-id";
const CORS_EXPOSE_HEADERS: &str = "x-talu-instance, x-talu-version, x-talu-api-version";
const CORS_MAX_AGE_SECONDS: &str = "600";

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

/// Machine-readable server identification and health response.
#[derive(Debug, Serialize, ToSchema)]
pub struct HealthResponse {
    /// Service status.
    pub status: &'static str,
    /// Stable product identifier for this server implementation.
    pub service: &'static str,
    /// API version exposed by this endpoint.
    pub api_version: &'static str,
    /// Talu build version.
    pub version: &'static str,
}

#[utoipa::path(get, path = "/v1/health", tag = "Models",
    responses((status = 200, body = HealthResponse, description = "Health and server identity")))]
pub async fn handle_health(
    _state: Arc<AppState>,
    _req: Request<Incoming>,
    _auth_ctx: Option<AuthContext>,
) -> Response<BoxBody> {
    let payload = HealthResponse {
        status: "ok",
        service: TALU_INSTANCE,
        api_version: TALU_API_VERSION,
        version: TALU_VERSION,
    };
    json_response(StatusCode::OK, &payload)
}

static OPENAPI_SPEC: Lazy<Vec<u8>> = Lazy::new(|| {
    let spec = openapi::build_openapi_json();
    filter_openapi_paths(
        &spec,
        &[
            "/v1/chat/completions",
            "/v1/health",
            "/v1/models",
            "/v1/repo",
            "/v1/responses",
            "/v1/tokenizer",
        ],
    )
});
static OPENAPI_CHAT_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/chat/completions"]));
static OPENAPI_RESPONSES_SPEC: Lazy<Vec<u8>> = Lazy::new(|| {
    responses_openapi::patch_responses_openapi_spec(&filter_openapi_paths(
        &OPENAPI_SPEC,
        &["/v1/responses"],
    ))
});
static OPENAPI_MODELS_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/models"]));
static OPENAPI_REPO_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/repo"]));
static OPENAPI_TOKENIZER_SPEC: Lazy<Vec<u8>> =
    Lazy::new(|| filter_openapi_paths(&OPENAPI_SPEC, &["/v1/tokenizer"]));

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
            let cors_request_headers = req.headers().clone();

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
                (Method::OPTIONS, _) => cors_preflight_response(&cors_request_headers),
                (Method::GET, "/health") => {
                    Response::new(Full::new(Bytes::from_static(b"ok")).boxed())
                }
                (Method::GET, "/v1/health") => handle_health(state, req, None).await,
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
                (Method::GET, "/openapi/repo.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_REPO_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/openapi/tokenizer.json") => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(OPENAPI_TOKENIZER_SPEC.clone())).boxed())
                    .unwrap(),
                (Method::GET, "/docs") => docs_hub_response(),
                (Method::GET, "/docs/chat") => {
                    swagger_ui_response("/openapi/chat.json", "Talu API :: Chat Completions")
                }
                (Method::GET, "/docs/responses") => {
                    swagger_ui_response("/openapi/responses.json", "Talu API :: Responses")
                }
                (Method::GET, "/docs/models") => {
                    swagger_ui_response("/openapi/models.json", "Talu API :: Models")
                }
                (Method::GET, "/docs/repo") => {
                    swagger_ui_response("/openapi/repo.json", "Talu API :: Repository")
                }
                (Method::GET, "/docs/tokenizer") => {
                    swagger_ui_response("/openapi/tokenizer.json", "Talu API :: Tokenizer")
                }
                _ => {
                    // Authenticate all other routes.
                    let auth = match authenticate(&state, req.headers()) {
                        Ok(ctx) => ctx,
                        Err(resp) => {
                            let resp = with_cors(resp, &cors_request_headers);
                            let status = resp.status();
                            if status.is_client_error() {
                                log::warn!(target: "server::http", "{} {} -> {}", method_log, path_log, status);
                            }
                            return Ok(resp);
                        }
                    };

                    match (method, path.as_str()) {
                        (Method::GET, "/v1/models") => {
                            handlers::handle_models(state, req, auth).await
                        }
                        (Method::GET, "/v1/repo/models") | (Method::GET, "/repo/models") => {
                            repo::handle_list(state, req, auth).await
                        }
                        (Method::GET, "/v1/repo/search") | (Method::GET, "/repo/search") => {
                            repo::handle_search(state, req, auth).await
                        }
                        (Method::POST, "/v1/repo/models") | (Method::POST, "/repo/models") => {
                            repo::handle_fetch(state, req, auth).await
                        }
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
                        (Method::POST, "/v1/responses") => {
                            responses::handle_create(state, req, auth).await
                        }
                        (Method::POST, "/v1/chat/completions") => {
                            completions::handle_create(state, req, auth).await
                        }
                        (_, "/v1/models") => method_not_allowed(&["GET"]),
                        (_, "/v1/repo/models") => method_not_allowed(&["GET", "POST"]),
                        (_, "/repo/models") => method_not_allowed(&["GET", "POST"]),
                        (_, "/v1/repo/search") => method_not_allowed(&["GET"]),
                        (_, "/repo/search") => method_not_allowed(&["GET"]),
                        (_, "/v1/responses") => method_not_allowed(&["POST"]),
                        (_, "/v1/chat/completions") => method_not_allowed(&["POST"]),
                        (Method::POST, "/v1/tokenizer/instances") => {
                            tokenizer::handle_create_instance(state, req, auth).await
                        }
                        (Method::GET, p) if p.starts_with("/v1/tokenizer/instances/") => {
                            tokenizer::handle_get_instance(state, req, auth).await
                        }
                        (Method::DELETE, p) if p.starts_with("/v1/tokenizer/instances/") => {
                            tokenizer::handle_delete_instance(state, req, auth).await
                        }
                        (Method::POST, "/v1/tokenizer/encode") => {
                            tokenizer::handle_encode(state, req, auth).await
                        }
                        (Method::POST, "/v1/tokenizer/encode_batch") => {
                            tokenizer::handle_encode_batch(state, req, auth).await
                        }
                        (Method::POST, "/v1/tokenizer/decode") => {
                            tokenizer::handle_decode(state, req, auth).await
                        }
                        (Method::POST, "/v1/tokenizer/decode_batch") => {
                            tokenizer::handle_decode_batch(state, req, auth).await
                        }
                        (Method::GET, "/v1/tokenizer/vocab") => {
                            tokenizer::handle_vocab(state, req, auth).await
                        }
                        (Method::GET, "/v1/tokenizer/vocab_size") => {
                            tokenizer::handle_vocab_size(state, req, auth).await
                        }
                        (Method::POST, "/v1/tokenizer/token_to_id") => {
                            tokenizer::handle_token_to_id(state, req, auth).await
                        }
                        (Method::POST, "/v1/tokenizer/id_to_token") => {
                            tokenizer::handle_id_to_token(state, req, auth).await
                        }
                        (Method::POST, "/v1/tokenizer/add_tokens") => {
                            tokenizer::handle_add_tokens(state, req, auth).await
                        }
                        (Method::POST, "/v1/tokenizer/add_special_tokens") => {
                            tokenizer::handle_add_special_tokens(state, req, auth).await
                        }
                        (Method::POST, "/v1/tokenizer/enable_truncation") => {
                            tokenizer::handle_enable_truncation(state, req, auth).await
                        }
                        (Method::POST, "/v1/tokenizer/disable_truncation") => {
                            tokenizer::handle_disable_truncation(state, req, auth).await
                        }
                        (Method::POST, "/v1/tokenizer/enable_padding") => {
                            tokenizer::handle_enable_padding(state, req, auth).await
                        }
                        (Method::POST, "/v1/tokenizer/disable_padding") => {
                            tokenizer::handle_disable_padding(state, req, auth).await
                        }
                        (Method::POST, "/v1/tokenizer/train") => {
                            tokenizer::handle_train(state, req, auth).await
                        }
                        (Method::POST, "/v1/tokenizer/train_from_iterator") => {
                            tokenizer::handle_train_from_iterator(state, req, auth).await
                        }
                        (Method::POST, "/v1/tokenizer/save") => {
                            tokenizer::handle_save(state, req, auth).await
                        }
                        (Method::POST, "/v1/tokenizer/compare") => {
                            tokenizer::handle_compare(state, req, auth).await
                        }
                        (Method::GET, "/v1/tokenizer/capabilities") => {
                            tokenizer::handle_capabilities(state, req, auth).await
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
            let response = with_cors(response, &cors_request_headers);

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

fn with_cors(mut response: Response<BoxBody>, req_headers: &hyper::HeaderMap) -> Response<BoxBody> {
    apply_cors_headers(response.headers_mut(), req_headers);
    response
}

fn cors_preflight_response(req_headers: &hyper::HeaderMap) -> Response<BoxBody> {
    let mut response = Response::builder()
        .status(StatusCode::NO_CONTENT)
        .body(Full::new(Bytes::new()).boxed())
        .unwrap();
    apply_cors_headers(response.headers_mut(), req_headers);
    response
}

fn apply_cors_headers(headers: &mut hyper::HeaderMap, req_headers: &hyper::HeaderMap) {
    headers.insert(
        "access-control-allow-origin",
        HeaderValue::from_static(CORS_ALLOW_ORIGIN),
    );
    headers.insert(
        "access-control-expose-headers",
        HeaderValue::from_static(CORS_EXPOSE_HEADERS),
    );
    headers.insert(
        "access-control-allow-methods",
        HeaderValue::from_static(CORS_ALLOW_METHODS),
    );
    if let Some(requested_headers) = req_headers.get("access-control-request-headers") {
        headers.insert("access-control-allow-headers", requested_headers.clone());
    } else {
        headers.insert(
            "access-control-allow-headers",
            HeaderValue::from_static(CORS_ALLOW_HEADERS_DEFAULT),
        );
    }
    headers.insert(
        "access-control-max-age",
        HeaderValue::from_static(CORS_MAX_AGE_SECONDS),
    );
    headers.insert(
        "vary",
        HeaderValue::from_static(
            "origin, access-control-request-method, access-control-request-headers",
        ),
    );
    headers.insert("x-talu-instance", HeaderValue::from_static(TALU_INSTANCE));
    headers.insert("x-talu-version", HeaderValue::from_static(TALU_VERSION));
    headers.insert(
        "x-talu-api-version",
        HeaderValue::from_static(TALU_API_VERSION),
    );
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
    if path.starts_with("/v1/tokenizer/instances/") {
        return true;
    }
    if let Some(stripped) = path.strip_prefix("/v1") {
        if KNOWN_PATHS.contains(stripped) {
            return true;
        }
    }
    false
}

fn swagger_ui_response(spec_url: &str, title: &str) -> Response<BoxBody> {
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/html; charset=utf-8")
        .body(Full::new(Bytes::from(swagger_ui_html(spec_url, title))).boxed())
        .unwrap()
}

fn json_response<T: Serialize>(status: StatusCode, payload: &T) -> Response<BoxBody> {
    let body = serde_json::to_vec(payload).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
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
  max-width: 72rem;
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
tr:last-child td {
  border-bottom: none;
}
th {
  background: #f1f5f9;
  font-weight: 700;
}
.json-cell {
  white-space: nowrap;
  width: 8.5rem;
}
.json-link {
  display: inline-block;
  padding: 0.08rem 0.4rem;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  font-size: 0.82rem;
  font-weight: 600;
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
  <p class="muted">Inference-only server surface.</p>

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
          <td><a href="/docs/responses"><code>responses</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/responses.json" title="/openapi/responses.json">json</a><button class="copy-btn" data-url="/openapi/responses.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td>OpenResponses-compatible inference API.</td>
        </tr>
        <tr>
          <td><a href="/docs/chat"><code>chat</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/chat.json" title="/openapi/chat.json">json</a><button class="copy-btn" data-url="/openapi/chat.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td>Chat completions endpoint.</td>
        </tr>
        <tr>
          <td><a href="/docs/models"><code>models</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/models.json" title="/openapi/models.json">json</a><button class="copy-btn" data-url="/openapi/models.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td>Model discovery and listing.</td>
        </tr>
        <tr>
          <td><a href="/docs/repo"><code>repo</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/repo.json" title="/openapi/repo.json">json</a><button class="copy-btn" data-url="/openapi/repo.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td>Repository cache management, hub search, and streaming model downloads.</td>
        </tr>
        <tr>
          <td><a href="/docs/tokenizer"><code>tokenizer</code></a></td>
          <td class="json-cell"><a class="json-link" href="/openapi/tokenizer.json" title="/openapi/tokenizer.json">json</a><button class="copy-btn" data-url="/openapi/tokenizer.json" title="Copy JSON URL" aria-label="Copy JSON URL">⧉</button></td>
          <td>Full tokenizer lifecycle, encode/decode, training, and persistence APIs.</td>
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

fn method_not_allowed(allow: &[&str]) -> Response<BoxBody> {
    let mut response = json_error(
        StatusCode::METHOD_NOT_ALLOWED,
        "method_not_allowed",
        "Method not allowed",
    );
    let allow_value = allow.join(", ");
    if let Ok(header) = HeaderValue::from_str(&allow_value) {
        response.headers_mut().insert("allow", header);
    }
    response
}
