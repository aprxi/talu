//! Tests for GET /v1/plugins/{id}/{path} — plugin asset serving.

use std::fs;

use tempfile::TempDir;

use crate::server::common::*;

fn plugins_config(plugins_dir: &std::path::Path) -> ServerConfig {
    let mut config = ServerConfig::new();
    config.env_vars.push((
        "TALU_PLUGINS_DIR".to_string(),
        plugins_dir.to_string_lossy().to_string(),
    ));
    config
}

fn setup_plugin(dir: &std::path::Path) {
    let plugin_dir = dir.join("test-plugin");
    fs::create_dir_all(&plugin_dir).unwrap();
    fs::write(plugin_dir.join("index.js"), "console.log('hello');").unwrap();
    fs::write(plugin_dir.join("style.css"), "body { color: red; }").unwrap();
    fs::write(plugin_dir.join("data.json"), r#"{"key": "value"}"#).unwrap();
}

// ---------------------------------------------------------------------------
// Serving JS files
// ---------------------------------------------------------------------------

#[test]
fn asset_serve_js_file() {
    let dir = TempDir::new().unwrap();
    setup_plugin(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let resp = get(ctx.addr(), "/v1/plugins/test-plugin/index.js");
    assert_eq!(resp.status, 200);
    let ct = resp.header("content-type").expect("content-type");
    assert!(
        ct.contains("application/javascript"),
        "expected JS content type, got: {ct}"
    );
    assert!(resp.body.contains("console.log"));
}

// ---------------------------------------------------------------------------
// Serving CSS files
// ---------------------------------------------------------------------------

#[test]
fn asset_serve_css_file() {
    let dir = TempDir::new().unwrap();
    setup_plugin(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let resp = get(ctx.addr(), "/v1/plugins/test-plugin/style.css");
    assert_eq!(resp.status, 200);
    let ct = resp.header("content-type").expect("content-type");
    assert!(
        ct.contains("text/css"),
        "expected CSS content type, got: {ct}"
    );
}

// ---------------------------------------------------------------------------
// Serving JSON files
// ---------------------------------------------------------------------------

#[test]
fn asset_serve_json_file() {
    let dir = TempDir::new().unwrap();
    setup_plugin(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let resp = get(ctx.addr(), "/v1/plugins/test-plugin/data.json");
    assert_eq!(resp.status, 200);
    let ct = resp.header("content-type").expect("content-type");
    assert!(
        ct.contains("application/json"),
        "expected JSON content type, got: {ct}"
    );
}

// ---------------------------------------------------------------------------
// 404 for missing asset
// ---------------------------------------------------------------------------

#[test]
fn asset_missing_file_returns_404() {
    let dir = TempDir::new().unwrap();
    setup_plugin(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let resp = get(ctx.addr(), "/v1/plugins/test-plugin/nonexistent.js");
    assert_eq!(resp.status, 404);
}

// ---------------------------------------------------------------------------
// 404 for missing plugin
// ---------------------------------------------------------------------------

#[test]
fn asset_missing_plugin_returns_404() {
    let dir = TempDir::new().unwrap();
    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let resp = get(ctx.addr(), "/v1/plugins/no-such-plugin/index.js");
    assert_eq!(resp.status, 404);
}

// ---------------------------------------------------------------------------
// Path traversal protection
// ---------------------------------------------------------------------------

#[test]
fn asset_path_traversal_blocked() {
    let dir = TempDir::new().unwrap();
    setup_plugin(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let resp = get(ctx.addr(), "/v1/plugins/test-plugin/../../../etc/passwd");
    assert_eq!(resp.status, 403);
}

// ---------------------------------------------------------------------------
// Bad request — missing path component
// ---------------------------------------------------------------------------

#[test]
fn asset_no_file_path_returns_400() {
    let dir = TempDir::new().unwrap();
    setup_plugin(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let resp = get(ctx.addr(), "/v1/plugins/test-plugin");
    // This should either be 400 (no file path) or 404 (not a 3-segment path)
    assert!(resp.status == 400 || resp.status == 404);
}

// ---------------------------------------------------------------------------
// Dual-path routing (without /v1 prefix)
// ---------------------------------------------------------------------------

#[test]
fn asset_works_without_v1_prefix() {
    let dir = TempDir::new().unwrap();
    setup_plugin(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let resp = get(ctx.addr(), "/plugins/test-plugin/index.js");
    assert_eq!(resp.status, 200);
    assert!(resp.body.contains("console.log"));
}

// ---------------------------------------------------------------------------
// Nested subdirectory assets
// ---------------------------------------------------------------------------

#[test]
fn asset_serve_nested_subdirectory() {
    let dir = TempDir::new().unwrap();
    let plugin_dir = dir.path().join("nested-plugin");
    let sub_dir = plugin_dir.join("dist").join("lib");
    fs::create_dir_all(&sub_dir).unwrap();
    fs::write(sub_dir.join("bundle.js"), "var x = 42;").unwrap();

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let resp = get(ctx.addr(), "/v1/plugins/nested-plugin/dist/lib/bundle.js");
    assert_eq!(resp.status, 200);
    assert!(resp.body.contains("var x = 42"));
}

// ---------------------------------------------------------------------------
// Path traversal via encoded characters
// ---------------------------------------------------------------------------

#[test]
fn asset_encoded_path_traversal_blocked() {
    let dir = TempDir::new().unwrap();
    setup_plugin(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    // %2e%2e = ".." URL-encoded
    let resp = get(
        ctx.addr(),
        "/v1/plugins/test-plugin/%2e%2e/%2e%2e/etc/passwd",
    );
    // Should be blocked — either 403 (traversal) or 404 (file not found after decode)
    assert!(
        resp.status == 403 || resp.status == 404,
        "encoded traversal should be blocked, got {}",
        resp.status
    );
}

// ---------------------------------------------------------------------------
// Path traversal via plugin ID
// ---------------------------------------------------------------------------

#[test]
fn asset_traversal_in_plugin_id_blocked() {
    let dir = TempDir::new().unwrap();
    setup_plugin(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let resp = get(ctx.addr(), "/v1/plugins/..%2F..%2Fetc/passwd");
    assert!(
        resp.status == 403 || resp.status == 404,
        "traversal in plugin ID should be blocked, got {}",
        resp.status
    );
}
