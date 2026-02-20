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

// ---------------------------------------------------------------------------
// --html-dir overrides bundled assets
// ---------------------------------------------------------------------------

fn html_dir_config(dir: &std::path::Path) -> ServerConfig {
    let mut config = ServerConfig::new();
    config.html_dir = Some(dir.to_path_buf());
    config
}

#[test]
fn html_dir_serves_custom_index() {
    let temp = tempfile::TempDir::new().expect("temp dir");
    let custom_html = "<html><body>Custom Console</body></html>";
    std::fs::write(temp.path().join("index.html"), custom_html).expect("write index.html");

    let ctx = ServerTestContext::new(html_dir_config(temp.path()));
    let resp = get(ctx.addr(), "/");
    assert_eq!(resp.status, 200);

    let ct = resp.header("content-type").expect("content-type header");
    assert!(ct.contains("text/html"), "expected text/html, got: {ct}");
    assert!(
        resp.body.contains("Custom Console"),
        "should serve custom index.html"
    );
    assert!(
        !resp.body.contains("Talu Console"),
        "should NOT serve bundled index"
    );
}

#[test]
fn html_dir_serves_custom_assets() {
    let temp = tempfile::TempDir::new().expect("temp dir");
    // Write minimal index so the server starts cleanly.
    std::fs::write(temp.path().join("index.html"), "<html></html>").expect("write index");
    std::fs::write(temp.path().join("style.css"), "body { color: red; }").expect("write css");
    std::fs::write(temp.path().join("main.js"), "console.log('custom');").expect("write js");

    let ctx = ServerTestContext::new(html_dir_config(temp.path()));

    let css_resp = get(ctx.addr(), "/assets/style.css");
    assert_eq!(css_resp.status, 200);
    assert!(css_resp.body.contains("color: red"));

    let js_resp = get(ctx.addr(), "/assets/main.js");
    assert_eq!(js_resp.status, 200);
    assert!(js_resp.body.contains("custom"));
}

#[test]
fn html_dir_missing_file_returns_404() {
    let temp = tempfile::TempDir::new().expect("temp dir");
    // Only write index.html — no main.js.
    std::fs::write(temp.path().join("index.html"), "<html></html>").expect("write index");

    let ctx = ServerTestContext::new(html_dir_config(temp.path()));

    let resp = get(ctx.addr(), "/assets/main.js");
    assert_eq!(resp.status, 404, "missing asset should return 404");
}
