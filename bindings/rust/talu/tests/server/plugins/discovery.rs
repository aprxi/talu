//! Tests for GET /v1/plugins — plugin discovery.

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

// ---------------------------------------------------------------------------
// Empty plugins directory
// ---------------------------------------------------------------------------

#[test]
fn discovery_empty_dir_returns_empty_list() {
    let dir = TempDir::new().unwrap();
    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let resp = get(ctx.addr(), "/v1/plugins");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(data.is_empty(), "empty dir should return empty list");
}

// ---------------------------------------------------------------------------
// Directory plugin with manifest
// ---------------------------------------------------------------------------

#[test]
fn discovery_directory_plugin_with_manifest() {
    let dir = TempDir::new().unwrap();
    let plugin_dir = dir.path().join("my-plugin");
    fs::create_dir(&plugin_dir).unwrap();
    fs::write(
        plugin_dir.join("talu.json"),
        r#"{"id":"my-plugin","activationEvents":["*"]}"#,
    )
    .unwrap();
    fs::write(plugin_dir.join("index.js"), "export default function(){}").unwrap();

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let resp = get(ctx.addr(), "/v1/plugins");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "my-plugin");
    assert_eq!(data[0]["entryUrl"], "/v1/plugins/my-plugin/index.js");
    assert!(
        data[0]["manifest"].is_object(),
        "manifest should be parsed JSON"
    );
    let token = data[0]["token"].as_str().expect("token should be a string");
    assert_eq!(token.len(), 64, "token should be 64 hex chars (32 bytes)");
    assert!(
        token.chars().all(|c| c.is_ascii_hexdigit()),
        "token should be hex"
    );
}

// ---------------------------------------------------------------------------
// Single-file plugin
// ---------------------------------------------------------------------------

#[test]
fn discovery_single_file_js_plugin() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("hello.js"), "export default function(){}").unwrap();

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let resp = get(ctx.addr(), "/v1/plugins");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "hello");
}

// ---------------------------------------------------------------------------
// Non-existent plugins directory
// ---------------------------------------------------------------------------

#[test]
fn discovery_nonexistent_dir_returns_empty_list() {
    let dir = TempDir::new().unwrap();
    let nonexistent = dir.path().join("does-not-exist");
    let ctx = ServerTestContext::new(plugins_config(&nonexistent));
    let resp = get(ctx.addr(), "/v1/plugins");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(data.is_empty());
}

// ---------------------------------------------------------------------------
// Multiple plugins
// ---------------------------------------------------------------------------

#[test]
fn discovery_multiple_plugins() {
    let dir = TempDir::new().unwrap();

    // Directory plugin
    let p1 = dir.path().join("plugin-a");
    fs::create_dir(&p1).unwrap();
    fs::write(p1.join("index.js"), "// a").unwrap();

    // Single-file plugin
    fs::write(dir.path().join("plugin-b.js"), "// b").unwrap();

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let resp = get(ctx.addr(), "/v1/plugins");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2);

    let ids: Vec<&str> = data.iter().filter_map(|p| p["id"].as_str()).collect();
    assert!(ids.contains(&"plugin-a"), "should contain plugin-a");
    assert!(ids.contains(&"plugin-b"), "should contain plugin-b");
}

// ---------------------------------------------------------------------------
// Dual-path routing (/plugins without /v1 prefix)
// ---------------------------------------------------------------------------

#[test]
fn discovery_works_without_v1_prefix() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("test.js"), "// test").unwrap();

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let resp = get(ctx.addr(), "/plugins");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1);
}

// ---------------------------------------------------------------------------
// Dotfiles and non-JS files are skipped
// ---------------------------------------------------------------------------

#[test]
fn discovery_skips_dotfiles_and_non_js() {
    let dir = TempDir::new().unwrap();

    // Dotfile directory — should be skipped
    let dot = dir.path().join(".hidden-plugin");
    fs::create_dir(&dot).unwrap();
    fs::write(dot.join("index.js"), "// hidden").unwrap();

    // Non-JS file — should be skipped
    fs::write(dir.path().join("readme.txt"), "not a plugin").unwrap();
    fs::write(dir.path().join("data.json"), "{}").unwrap();

    // Valid plugin — should be discovered
    fs::write(dir.path().join("real.js"), "// real").unwrap();

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let resp = get(ctx.addr(), "/v1/plugins");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1, "only the .js file should be discovered");
    assert_eq!(data[0]["id"], "real");
}

// ---------------------------------------------------------------------------
// Manifest permissions appear in discovery response
// ---------------------------------------------------------------------------

#[test]
fn discovery_manifest_permissions_visible() {
    let dir = TempDir::new().unwrap();
    let plugin_dir = dir.path().join("perms-plugin");
    fs::create_dir(&plugin_dir).unwrap();
    fs::write(
        plugin_dir.join("talu.json"),
        r#"{"id":"perms-plugin","permissions":["network:api.example.com","storage","ui"]}"#,
    )
    .unwrap();
    fs::write(plugin_dir.join("index.js"), "// perms").unwrap();

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let resp = get(ctx.addr(), "/v1/plugins");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let manifest = &json["data"][0]["manifest"];
    let perms = manifest["permissions"]
        .as_array()
        .expect("permissions array");
    assert_eq!(perms.len(), 3);
    assert!(perms.contains(&serde_json::json!("network:api.example.com")));
}

// ---------------------------------------------------------------------------
// Directory plugin without entry point is skipped
// ---------------------------------------------------------------------------

#[test]
fn discovery_skips_dir_without_entry_point() {
    let dir = TempDir::new().unwrap();

    // Directory with manifest but no index.js/index.ts
    let no_entry = dir.path().join("no-entry");
    fs::create_dir(&no_entry).unwrap();
    fs::write(
        no_entry.join("talu.json"),
        r#"{"id":"no-entry","activationEvents":["*"]}"#,
    )
    .unwrap();
    fs::write(no_entry.join("readme.md"), "no entry point here").unwrap();

    // Valid plugin
    fs::write(dir.path().join("valid.js"), "// ok").unwrap();

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let resp = get(ctx.addr(), "/v1/plugins");
    let data = resp.json()["data"].as_array().unwrap().clone();
    let ids: Vec<&str> = data.iter().filter_map(|p| p["id"].as_str()).collect();
    assert!(ids.contains(&"valid"), "valid plugin should be present");
    assert!(
        !ids.contains(&"no-entry"),
        "directory without entry point should be skipped"
    );
}
