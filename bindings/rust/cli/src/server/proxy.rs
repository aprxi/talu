//! Proxy endpoint for plugin outbound HTTP requests.
//!
//! `POST /v1/proxy` — forwards HTTP requests on behalf of plugins with
//! SSRF protection and per-plugin domain allowlisting.

use std::convert::Infallible;
use std::net::IpAddr;
use std::sync::Arc;

use bytes::Bytes;
#[allow(unused_imports)]
use http_body_util::BodyExt;
use http_body_util::Full;
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde::{Deserialize, Serialize};

use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

// =============================================================================
// Request/Response types
// =============================================================================

#[derive(Deserialize)]
struct ProxyRequest {
    url: String,
    #[serde(default = "default_method")]
    method: String,
    #[serde(default)]
    headers: std::collections::HashMap<String, String>,
    #[serde(default)]
    body: Option<String>,
}

fn default_method() -> String {
    "GET".to_string()
}

#[derive(Serialize)]
struct ProxyResponse {
    status: u16,
    headers: std::collections::HashMap<String, String>,
    body: String,
}

// =============================================================================
// Handler
// =============================================================================

pub async fn handle_proxy(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    // Require a valid plugin capability token (Bearer auth).
    let (plugin_id, network_permissions) = {
        let auth_header = match req
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
        {
            Some(h) => h.to_string(),
            None => {
                return json_error(
                    StatusCode::UNAUTHORIZED,
                    "unauthorized",
                    "Missing Authorization header",
                )
            }
        };
        let token = match auth_header.strip_prefix("Bearer ") {
            Some(t) => t.to_string(),
            None => {
                return json_error(
                    StatusCode::UNAUTHORIZED,
                    "unauthorized",
                    "Expected Bearer token",
                )
            }
        };
        let tokens = state.plugin_tokens.lock().await;
        match tokens.get(&token) {
            Some(entry) => (entry.plugin_id.clone(), entry.network_permissions.clone()),
            None => {
                return json_error(
                    StatusCode::UNAUTHORIZED,
                    "unauthorized",
                    "Invalid plugin token",
                )
            }
        }
    };

    // Parse request body.
    let body_bytes = match req.collect().await {
        Ok(b) => b.to_bytes(),
        Err(_) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "bad_request",
                "Failed to read request body",
            )
        }
    };

    let proxy_req: ProxyRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "bad_request",
                &format!("Invalid proxy request JSON: {}", e),
            );
        }
    };

    // Parse and validate URL.
    let parsed_url = match url::Url::parse(&proxy_req.url) {
        Ok(u) => u,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "bad_request",
                &format!("Invalid URL: {}", e),
            );
        }
    };

    // Only allow http/https.
    match parsed_url.scheme() {
        "http" | "https" => {}
        scheme => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "bad_request",
                &format!("Unsupported scheme: {}", scheme),
            );
        }
    }

    let host = parsed_url.host_str().unwrap_or("");
    let host_allowed = is_domain_allowed(&network_permissions, host);

    // SSRF protection: block private/internal IPs.
    //
    // Explicit network permissions from the plugin manifest (controlled by
    // the server operator) override the SSRF check for the granted hosts.
    // The operator must intentionally place a plugin with `network:127.0.0.1`
    // (or similar) in the plugins directory for this to take effect.
    if is_private_host(host) && !host_allowed {
        return json_error(
            StatusCode::FORBIDDEN,
            "forbidden",
            "Requests to private/internal addresses are not allowed",
        );
    }

    // Domain allowlist: require explicit permission for all destinations.
    if !host_allowed {
        return json_error(
            StatusCode::FORBIDDEN,
            "forbidden",
            &format!(
                "Plugin '{}' does not have network permission for '{}'",
                plugin_id, host
            ),
        );
    }

    // Build outbound request.
    let client = reqwest::Client::new();
    let method = match proxy_req.method.to_uppercase().as_str() {
        "GET" => reqwest::Method::GET,
        "POST" => reqwest::Method::POST,
        "PUT" => reqwest::Method::PUT,
        "PATCH" => reqwest::Method::PATCH,
        "DELETE" => reqwest::Method::DELETE,
        "HEAD" => reqwest::Method::HEAD,
        "OPTIONS" => reqwest::Method::OPTIONS,
        _ => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "bad_request",
                &format!("Unsupported method: {}", proxy_req.method),
            );
        }
    };

    let mut outbound = client.request(method, proxy_req.url.clone());

    // Forward user-provided headers, stripping internal ones.
    for (key, value) in &proxy_req.headers {
        let key_lower = key.to_lowercase();
        if key_lower.starts_with("x-talu-") || key_lower == "cookie" || key_lower == "host" {
            continue;
        }
        outbound = outbound.header(key.as_str(), value.as_str());
    }

    if let Some(ref body) = proxy_req.body {
        outbound = outbound.body(body.clone());
    }

    // Execute.
    let upstream = match outbound.send().await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!(target: "server::proxy", "proxy request to {} failed: {}", proxy_req.url, e);
            return json_error(
                StatusCode::BAD_GATEWAY,
                "upstream_error",
                &format!("Upstream request failed: {}", e),
            );
        }
    };

    // Collect response.
    let status = upstream.status().as_u16();
    let mut resp_headers = std::collections::HashMap::new();
    for key in &["content-type", "cache-control", "etag"] {
        if let Some(val) = upstream.headers().get(*key) {
            if let Ok(v) = val.to_str() {
                resp_headers.insert(key.to_string(), v.to_string());
            }
        }
    }

    let resp_body = match upstream.text().await {
        Ok(t) => t,
        Err(e) => {
            return json_error(
                StatusCode::BAD_GATEWAY,
                "upstream_error",
                &format!("Failed to read upstream response: {}", e),
            );
        }
    };

    json_response(
        StatusCode::OK,
        &ProxyResponse {
            status,
            headers: resp_headers,
            body: resp_body,
        },
    )
}

// =============================================================================
// SSRF Protection
// =============================================================================

fn is_private_host(host: &str) -> bool {
    // Block known internal hostnames.
    let host_lower = host.to_lowercase();
    if host_lower == "localhost"
        || host_lower.ends_with(".local")
        || host_lower.ends_with(".internal")
    {
        return true;
    }

    // Try to parse as IP and check private ranges.
    if let Ok(ip) = host.parse::<IpAddr>() {
        return is_private_ip(ip);
    }

    false
}

fn is_private_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()             // 127.0.0.0/8
                || v4.is_private()       // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
                || v4.is_link_local()    // 169.254.0.0/16
                || v4.is_unspecified() // 0.0.0.0
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()             // ::1
                || v6.is_unspecified()   // ::
                // fe80::/10 (link-local) — check first 10 bits
                || (v6.segments()[0] & 0xffc0) == 0xfe80
        }
    }
}

// =============================================================================
// Domain Allowlist
// =============================================================================

/// Check if a plugin's network permissions allow the given domain.
///
/// Permissions are pre-extracted from the manifest and stored in the token store.
/// Supports exact match (`api.example.com`) and wildcard (`*.example.com`).
fn is_domain_allowed(permissions: &[String], domain: &str) -> bool {
    let domain_lower = domain.to_lowercase();

    for pattern in permissions {
        let pattern_lower = pattern.to_lowercase();
        if pattern_lower == domain_lower {
            return true;
        }
        if let Some(suffix) = pattern_lower.strip_prefix("*.") {
            if domain_lower.ends_with(&format!(".{}", suffix)) || domain_lower == suffix {
                return true;
            }
        }
    }

    false
}

// =============================================================================
// Helpers
// =============================================================================

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
