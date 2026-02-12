//! Integration tests for the Talu Console static asset routes.

use crate::server::common::*;

fn console_config() -> ServerConfig {
    ServerConfig::new()
}

// ---------------------------------------------------------------------------
// GET / — serves index.html
// ---------------------------------------------------------------------------

#[test]
fn get_root_returns_200() {
    let ctx = ServerTestContext::new(console_config());
    let resp = get(ctx.addr(), "/");
    assert_eq!(resp.status, 200);
}

#[test]
fn get_root_has_html_content_type() {
    let ctx = ServerTestContext::new(console_config());
    let resp = get(ctx.addr(), "/");
    let ct = resp.header("content-type").expect("content-type header");
    assert!(ct.contains("text/html"), "expected text/html, got: {ct}");
}

#[test]
fn get_root_body_contains_talu_console() {
    let ctx = ServerTestContext::new(console_config());
    let resp = get(ctx.addr(), "/");
    assert!(
        resp.body.contains("Talu Console"),
        "index.html should contain 'Talu Console'"
    );
}

#[test]
fn get_root_body_references_assets() {
    let ctx = ServerTestContext::new(console_config());
    let resp = get(ctx.addr(), "/");
    assert!(
        resp.body.contains("/assets/main.js"),
        "should reference main.js"
    );
    assert!(
        resp.body.contains("/assets/style.css"),
        "should reference style.css"
    );
}

// ---------------------------------------------------------------------------
// Security headers on GET /
// ---------------------------------------------------------------------------

#[test]
fn get_root_has_referrer_policy_no_referrer() {
    let ctx = ServerTestContext::new(console_config());
    let resp = get(ctx.addr(), "/");
    let rp = resp
        .header("referrer-policy")
        .expect("referrer-policy header");
    assert_eq!(rp, "no-referrer");
}

// ---------------------------------------------------------------------------
// GET /assets/main.js
// ---------------------------------------------------------------------------

#[test]
fn get_main_js_returns_200() {
    let ctx = ServerTestContext::new(console_config());
    let resp = get(ctx.addr(), "/assets/main.js");
    assert_eq!(resp.status, 200);
}

#[test]
fn get_main_js_has_js_content_type() {
    let ctx = ServerTestContext::new(console_config());
    let resp = get(ctx.addr(), "/assets/main.js");
    let ct = resp.header("content-type").expect("content-type header");
    assert!(
        ct.contains("application/javascript"),
        "expected application/javascript, got: {ct}"
    );
}

#[test]
fn get_main_js_body_is_not_empty() {
    let ctx = ServerTestContext::new(console_config());
    let resp = get(ctx.addr(), "/assets/main.js");
    assert!(!resp.body.is_empty(), "main.js should not be empty");
}

#[test]
fn get_main_js_has_referrer_policy_no_referrer() {
    let ctx = ServerTestContext::new(console_config());
    let resp = get(ctx.addr(), "/assets/main.js");
    let rp = resp
        .header("referrer-policy")
        .expect("referrer-policy header");
    assert_eq!(rp, "no-referrer");
}

// ---------------------------------------------------------------------------
// GET /assets/style.css
// ---------------------------------------------------------------------------

#[test]
fn get_style_css_returns_200() {
    let ctx = ServerTestContext::new(console_config());
    let resp = get(ctx.addr(), "/assets/style.css");
    assert_eq!(resp.status, 200);
}

#[test]
fn get_style_css_has_css_content_type() {
    let ctx = ServerTestContext::new(console_config());
    let resp = get(ctx.addr(), "/assets/style.css");
    let ct = resp.header("content-type").expect("content-type header");
    assert!(ct.contains("text/css"), "expected text/css, got: {ct}");
}

#[test]
fn get_style_css_body_is_not_empty() {
    let ctx = ServerTestContext::new(console_config());
    let resp = get(ctx.addr(), "/assets/style.css");
    assert!(!resp.body.is_empty(), "style.css should not be empty");
}

#[test]
fn get_style_css_has_referrer_policy_no_referrer() {
    let ctx = ServerTestContext::new(console_config());
    let resp = get(ctx.addr(), "/assets/style.css");
    let rp = resp
        .header("referrer-policy")
        .expect("referrer-policy header");
    assert_eq!(rp, "no-referrer");
}

// ---------------------------------------------------------------------------
// 404 for unknown assets
// ---------------------------------------------------------------------------

#[test]
fn get_unknown_asset_returns_404() {
    let ctx = ServerTestContext::new(console_config());
    let resp = get(ctx.addr(), "/assets/nonexistent.js");
    assert_eq!(resp.status, 404);
}

#[test]
fn get_assets_root_returns_404() {
    let ctx = ServerTestContext::new(console_config());
    let resp = get(ctx.addr(), "/assets/");
    assert_eq!(resp.status, 404);
}
