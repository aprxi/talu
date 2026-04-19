//! `/v1/chat/completions` route mounting and HTTP method tests.

use crate::server::common::{post_json, send_request, ServerConfig, ServerTestContext};

#[test]
fn completions_route_is_mounted() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}]
    });
    let resp = post_json(ctx.addr(), "/v1/chat/completions", &body);
    // Without a model loaded, we expect an error — but NOT 404.
    assert_ne!(resp.status, 404, "route must be mounted: {}", resp.body);
}

#[test]
fn completions_get_not_allowed() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = send_request(ctx.addr(), "GET", "/v1/chat/completions", &[], None);
    assert_eq!(resp.status, 405, "GET should be 405, body: {}", resp.body);
    assert_eq!(
        resp.header("allow"),
        Some("POST"),
        "GET /v1/chat/completions must advertise Allow: POST"
    );
}

#[test]
fn completions_put_not_allowed() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = send_request(ctx.addr(), "PUT", "/v1/chat/completions", &[], None);
    assert_eq!(resp.status, 405, "PUT should be 405, body: {}", resp.body);
    assert_eq!(
        resp.header("allow"),
        Some("POST"),
        "PUT /v1/chat/completions must advertise Allow: POST"
    );
}

#[test]
fn completions_delete_not_allowed() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = send_request(ctx.addr(), "DELETE", "/v1/chat/completions", &[], None);
    assert_eq!(
        resp.status, 405,
        "DELETE should be 405, body: {}",
        resp.body
    );
    assert_eq!(
        resp.header("allow"),
        Some("POST"),
        "DELETE /v1/chat/completions must advertise Allow: POST"
    );
}
