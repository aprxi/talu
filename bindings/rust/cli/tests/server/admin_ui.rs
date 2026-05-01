//! Integration tests for the built-in Talu admin UI.

use crate::server::common::*;

#[test]
fn root_serves_admin_ui() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = get(ctx.addr(), "/");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let content_type = resp.header("content-type").unwrap_or("");
    assert!(
        content_type.contains("text/html"),
        "content-type={content_type}"
    );
    assert!(
        resp.body.contains("<title>Talu Admin</title>"),
        "missing admin page title"
    );
    assert!(
        resp.body.contains(r#"data-panel="discovery""#),
        "missing discovery panel tab"
    );
}

#[test]
fn admin_alias_serves_admin_ui() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let resp = get(ctx.addr(), "/admin");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert!(
        resp.body.contains("/v1/repo/downloads"),
        "admin page should wire repo download API"
    );
}
