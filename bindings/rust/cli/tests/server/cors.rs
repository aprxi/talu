//! CORS behavior for browser-based clients.

use crate::server::common::{send_request, ServerConfig, ServerTestContext, TenantSpec};

#[test]
fn models_get_includes_cors_headers() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/models",
        &[("Origin", "http://localhost:3000")],
        None,
    );
    assert_ne!(resp.status, 404, "route must be mounted: {}", resp.body);
    assert_eq!(
        resp.header("Access-Control-Allow-Origin"),
        Some("*"),
        "missing allow-origin header: {}",
        resp.headers
    );
    assert!(
        resp.header("Access-Control-Allow-Methods")
            .unwrap_or("")
            .contains("GET"),
        "missing GET in allow-methods: {}",
        resp.headers
    );
}

#[test]
fn models_preflight_options_returns_no_content_with_cors_headers() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = send_request(
        ctx.addr(),
        "OPTIONS",
        "/v1/models",
        &[
            ("Origin", "http://localhost:3000"),
            ("Access-Control-Request-Method", "GET"),
            (
                "Access-Control-Request-Headers",
                "content-type, x-talu-gateway-secret",
            ),
        ],
        None,
    );
    assert_eq!(resp.status, 204, "preflight failed: {}", resp.body);
    assert_eq!(
        resp.header("Access-Control-Allow-Origin"),
        Some("*"),
        "missing allow-origin header: {}",
        resp.headers
    );
    let allow_headers = resp
        .header("Access-Control-Allow-Headers")
        .unwrap_or("")
        .to_ascii_lowercase();
    assert!(
        allow_headers.contains("content-type"),
        "missing content-type in allow-headers: {}",
        resp.headers
    );
    assert!(
        allow_headers.contains("x-talu-gateway-secret"),
        "missing x-talu-gateway-secret in allow-headers: {}",
        resp.headers
    );
}

#[test]
fn auth_errors_still_include_cors_headers() {
    let mut cfg = ServerConfig::new();
    cfg.gateway_secret = Some("secret".to_string());
    cfg.tenants = vec![TenantSpec {
        id: "acme".to_string(),
        storage_prefix: "acme".to_string(),
        allowed_models: vec![],
    }];
    let ctx = ServerTestContext::new(cfg);
    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/models",
        &[("Origin", "http://localhost:3000")],
        None,
    );
    assert_eq!(resp.status, 401, "expected unauthorized: {}", resp.body);
    assert_eq!(
        resp.header("Access-Control-Allow-Origin"),
        Some("*"),
        "missing allow-origin header on auth error: {}",
        resp.headers
    );
}
