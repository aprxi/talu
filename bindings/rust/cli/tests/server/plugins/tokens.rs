//! Tests for plugin capability tokens — token generation, proxy auth, and
//! plugin_storage document gating.

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

fn plugins_config_with_bucket(
    plugins_dir: &std::path::Path,
    bucket: &std::path::Path,
) -> ServerConfig {
    let mut config = plugins_config(plugins_dir);
    config.bucket = Some(bucket.to_path_buf());
    config
}

/// Create a directory plugin with a manifest that has network permissions.
fn setup_plugin_with_permissions(dir: &std::path::Path) {
    let plugin_dir = dir.join("net-plugin");
    fs::create_dir_all(&plugin_dir).unwrap();
    fs::write(
        plugin_dir.join("talu.json"),
        r#"{"id":"net-plugin","activationEvents":["*"],"permissions":["network:api.example.com","network:*.google.com"]}"#,
    )
    .unwrap();
    fs::write(plugin_dir.join("index.js"), "export default function(){}").unwrap();
}

/// Helper: discover plugins and return the token for a specific plugin ID.
fn discover_and_get_token(addr: std::net::SocketAddr, plugin_id: &str) -> String {
    let resp = get(addr, "/v1/plugins");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    data.iter()
        .find(|p| p["id"].as_str() == Some(plugin_id))
        .unwrap_or_else(|| panic!("plugin {plugin_id} not found"))["token"]
        .as_str()
        .expect("token string")
        .to_string()
}

/// Helper: discover plugins and return tokens for multiple plugin IDs from a
/// single discovery call (avoids invalidating tokens between lookups).
fn discover_and_get_tokens(addr: std::net::SocketAddr, plugin_ids: &[&str]) -> Vec<String> {
    let resp = get(addr, "/v1/plugins");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    plugin_ids
        .iter()
        .map(|id| {
            data.iter()
                .find(|p| p["id"].as_str() == Some(id))
                .unwrap_or_else(|| panic!("plugin {id} not found"))["token"]
                .as_str()
                .expect("token string")
                .to_string()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Token uniqueness per plugin
// ---------------------------------------------------------------------------

#[test]
fn tokens_unique_per_plugin() {
    let dir = TempDir::new().unwrap();

    let p1 = dir.path().join("plugin-a");
    fs::create_dir(&p1).unwrap();
    fs::write(p1.join("index.js"), "// a").unwrap();

    fs::write(dir.path().join("plugin-b.js"), "// b").unwrap();

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let resp = get(ctx.addr(), "/v1/plugins");
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2);

    let tokens: Vec<&str> = data.iter().filter_map(|p| p["token"].as_str()).collect();
    assert_eq!(tokens.len(), 2, "each plugin should have a token");
    assert_ne!(tokens[0], tokens[1], "tokens should be unique");
}

// ---------------------------------------------------------------------------
// Tokens regenerated on each discovery call
// ---------------------------------------------------------------------------

#[test]
fn tokens_regenerated_on_rediscovery() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("test.js"), "// test").unwrap();

    let ctx = ServerTestContext::new(plugins_config(dir.path()));

    let resp1 = get(ctx.addr(), "/v1/plugins");
    let token1 = resp1.json()["data"][0]["token"]
        .as_str()
        .expect("token")
        .to_string();

    let resp2 = get(ctx.addr(), "/v1/plugins");
    let token2 = resp2.json()["data"][0]["token"]
        .as_str()
        .expect("token")
        .to_string();

    assert_ne!(token1, token2, "tokens should be regenerated on each call");
}

// ---------------------------------------------------------------------------
// Proxy: missing Authorization header → 401
// ---------------------------------------------------------------------------

#[test]
fn proxy_missing_auth_returns_401() {
    let dir = TempDir::new().unwrap();
    setup_plugin_with_permissions(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let body = r#"{"url":"https://api.example.com/v1/test","method":"GET"}"#;
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/proxy",
        &[("Content-Type", "application/json")],
        Some(body),
    );
    assert_eq!(resp.status, 401);
}

// ---------------------------------------------------------------------------
// Proxy: invalid Bearer token → 401
// ---------------------------------------------------------------------------

#[test]
fn proxy_invalid_token_returns_401() {
    let dir = TempDir::new().unwrap();
    setup_plugin_with_permissions(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let body = r#"{"url":"https://api.example.com/v1/test","method":"GET"}"#;
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/proxy",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", "Bearer invalid-token-here"),
        ],
        Some(body),
    );
    assert_eq!(resp.status, 401);
}

// ---------------------------------------------------------------------------
// Proxy: valid token but domain not in permissions → 403
// ---------------------------------------------------------------------------

#[test]
fn proxy_domain_not_allowed_returns_403() {
    let dir = TempDir::new().unwrap();
    setup_plugin_with_permissions(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let token = discover_and_get_token(ctx.addr(), "net-plugin");

    let body = r#"{"url":"https://evil.example.org/steal","method":"GET"}"#;
    let auth = format!("Bearer {token}");
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/proxy",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", &auth),
        ],
        Some(body),
    );
    assert_eq!(resp.status, 403);
}

// ---------------------------------------------------------------------------
// Proxy: SSRF protection still active with valid token
// ---------------------------------------------------------------------------

#[test]
fn proxy_ssrf_blocked_with_valid_token() {
    let dir = TempDir::new().unwrap();

    // Plugin with network:localhost permission (should still be blocked by SSRF).
    let plugin_dir = dir.path().join("ssrf-plugin");
    fs::create_dir_all(&plugin_dir).unwrap();
    fs::write(
        plugin_dir.join("talu.json"),
        r#"{"id":"ssrf-plugin","permissions":["network:localhost"]}"#,
    )
    .unwrap();
    fs::write(plugin_dir.join("index.js"), "// ssrf").unwrap();

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let token = discover_and_get_token(ctx.addr(), "ssrf-plugin");

    let body = r#"{"url":"http://localhost/secret","method":"GET"}"#;
    let auth = format!("Bearer {token}");
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/proxy",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", &auth),
        ],
        Some(body),
    );
    assert_eq!(
        resp.status, 403,
        "SSRF should be blocked even with valid token"
    );
}

// ---------------------------------------------------------------------------
// Plugin storage: type=plugin_storage without token → 403
// ---------------------------------------------------------------------------

#[test]
fn plugin_storage_list_without_token_returns_403() {
    let dir = TempDir::new().unwrap();
    let bucket = TempDir::new().unwrap();
    fs::write(dir.path().join("test.js"), "// test").unwrap();

    let ctx = ServerTestContext::new(plugins_config_with_bucket(dir.path(), bucket.path()));
    let resp = get(ctx.addr(), "/v1/documents?type=plugin_storage");
    assert_eq!(resp.status, 403);
}

// ---------------------------------------------------------------------------
// Plugin storage: type=plugin_storage with valid token → 200
// ---------------------------------------------------------------------------

#[test]
fn plugin_storage_list_with_token_returns_200() {
    let dir = TempDir::new().unwrap();
    let bucket = TempDir::new().unwrap();
    fs::write(dir.path().join("test.js"), "// test").unwrap();

    let ctx = ServerTestContext::new(plugins_config_with_bucket(dir.path(), bucket.path()));
    let token = discover_and_get_token(ctx.addr(), "test");

    let auth = format!("Bearer {token}");
    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/documents?type=plugin_storage",
        &[("Authorization", &auth)],
        None,
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    let data = json["data"].as_array().expect("data array");
    assert!(data.is_empty(), "no documents should exist yet");
}

// ---------------------------------------------------------------------------
// Plugin storage: create without token → 403
// ---------------------------------------------------------------------------

#[test]
fn plugin_storage_create_without_token_returns_403() {
    let dir = TempDir::new().unwrap();
    let bucket = TempDir::new().unwrap();
    fs::write(dir.path().join("test.js"), "// test").unwrap();

    let ctx = ServerTestContext::new(plugins_config_with_bucket(dir.path(), bucket.path()));
    let body = serde_json::json!({
        "type": "plugin_storage",
        "title": "test doc",
        "content": {"key": "value"}
    });
    let resp = post_json(ctx.addr(), "/v1/documents", &body);
    assert_eq!(resp.status, 403);
}

// ---------------------------------------------------------------------------
// Plugin storage: create with valid token → 201, owner_id set
// ---------------------------------------------------------------------------

#[test]
fn plugin_storage_create_with_token_sets_owner_id() {
    let dir = TempDir::new().unwrap();
    let bucket = TempDir::new().unwrap();
    fs::write(dir.path().join("my-plugin.js"), "// plugin").unwrap();

    let ctx = ServerTestContext::new(plugins_config_with_bucket(dir.path(), bucket.path()));
    let token = discover_and_get_token(ctx.addr(), "my-plugin");

    let body = serde_json::to_string(&serde_json::json!({
        "type": "plugin_storage",
        "title": "plugin data",
        "content": {"setting": true}
    }))
    .unwrap();
    let auth = format!("Bearer {token}");
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/documents",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", &auth),
        ],
        Some(&body),
    );
    assert_eq!(resp.status, 201);
    let json = resp.json();
    assert_eq!(
        json["owner_id"].as_str(),
        Some("my-plugin"),
        "owner_id should be set to the plugin ID from the token"
    );
}

// ===========================================================================
// Token invalidation
// ===========================================================================

// ---------------------------------------------------------------------------
// Old token is rejected after rediscovery
// ---------------------------------------------------------------------------

#[test]
fn old_token_rejected_after_rediscovery() {
    let dir = TempDir::new().unwrap();
    setup_plugin_with_permissions(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));

    // Get a token from first discovery.
    let old_token = discover_and_get_token(ctx.addr(), "net-plugin");

    // Rediscover — all old tokens are invalidated.
    let _new_token = discover_and_get_token(ctx.addr(), "net-plugin");

    // Try using the old token — should be rejected.
    let body = r#"{"url":"https://api.example.com/v1/test","method":"GET"}"#;
    let auth = format!("Bearer {old_token}");
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/proxy",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", &auth),
        ],
        Some(body),
    );
    assert_eq!(
        resp.status, 401,
        "old token should be rejected after rediscovery"
    );
}

// ===========================================================================
// Plugin storage isolation
// ===========================================================================

// ---------------------------------------------------------------------------
// Plugin A can't see plugin B's documents
// ---------------------------------------------------------------------------

#[test]
fn plugin_storage_isolation_between_plugins() {
    let dir = TempDir::new().unwrap();
    let bucket = TempDir::new().unwrap();
    fs::write(dir.path().join("plugin-a.js"), "// a").unwrap();
    fs::write(dir.path().join("plugin-b.js"), "// b").unwrap();

    let ctx = ServerTestContext::new(plugins_config_with_bucket(dir.path(), bucket.path()));

    let tokens = discover_and_get_tokens(ctx.addr(), &["plugin-a", "plugin-b"]);
    let token_a = &tokens[0];
    let token_b = &tokens[1];

    // Plugin A creates a document.
    let body = serde_json::to_string(&serde_json::json!({
        "type": "plugin_storage",
        "title": "A's secret",
        "content": {"owner": "a"}
    }))
    .unwrap();
    let auth_a = format!("Bearer {token_a}");
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/documents",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", &auth_a),
        ],
        Some(&body),
    );
    assert_eq!(resp.status, 201);

    // Plugin A can see its own document.
    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/documents?type=plugin_storage",
        &[("Authorization", &auth_a)],
        None,
    );
    assert_eq!(resp.status, 200);
    let data = resp.json()["data"].as_array().unwrap().clone();
    assert_eq!(data.len(), 1, "plugin-a should see its own document");

    // Plugin B sees nothing — different owner_id.
    let auth_b = format!("Bearer {token_b}");
    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/documents?type=plugin_storage",
        &[("Authorization", &auth_b)],
        None,
    );
    assert_eq!(resp.status, 200);
    let data = resp.json()["data"].as_array().unwrap().clone();
    assert!(
        data.is_empty(),
        "plugin-b should not see plugin-a's documents"
    );
}

// ---------------------------------------------------------------------------
// Plugin storage: client-provided owner_id is overridden by token
// ---------------------------------------------------------------------------

#[test]
fn plugin_storage_ignores_client_owner_id() {
    let dir = TempDir::new().unwrap();
    let bucket = TempDir::new().unwrap();
    fs::write(dir.path().join("honest.js"), "// honest").unwrap();

    let ctx = ServerTestContext::new(plugins_config_with_bucket(dir.path(), bucket.path()));
    let token = discover_and_get_token(ctx.addr(), "honest");

    // Try to set owner_id to a different plugin — should be overridden.
    let body = serde_json::to_string(&serde_json::json!({
        "type": "plugin_storage",
        "title": "sneaky doc",
        "content": {},
        "owner_id": "someone-else"
    }))
    .unwrap();
    let auth = format!("Bearer {token}");
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/documents",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", &auth),
        ],
        Some(&body),
    );
    assert_eq!(resp.status, 201);
    let json = resp.json();
    assert_eq!(
        json["owner_id"].as_str(),
        Some("honest"),
        "owner_id should be forced to the token's plugin_id, not the client-provided value"
    );
}

// ---------------------------------------------------------------------------
// Plugin storage: create then list roundtrip
// ---------------------------------------------------------------------------

#[test]
fn plugin_storage_create_then_list_roundtrip() {
    let dir = TempDir::new().unwrap();
    let bucket = TempDir::new().unwrap();
    fs::write(dir.path().join("roundtrip.js"), "// rt").unwrap();

    let ctx = ServerTestContext::new(plugins_config_with_bucket(dir.path(), bucket.path()));
    let token = discover_and_get_token(ctx.addr(), "roundtrip");
    let auth = format!("Bearer {token}");

    // Create two documents.
    for title in &["doc-1", "doc-2"] {
        let body = serde_json::to_string(&serde_json::json!({
            "type": "plugin_storage",
            "title": title,
            "content": {"n": title}
        }))
        .unwrap();
        let resp = send_request(
            ctx.addr(),
            "POST",
            "/v1/documents",
            &[
                ("Content-Type", "application/json"),
                ("Authorization", &auth),
            ],
            Some(&body),
        );
        assert_eq!(resp.status, 201);
    }

    // List should return both.
    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/documents?type=plugin_storage",
        &[("Authorization", &auth)],
        None,
    );
    assert_eq!(resp.status, 200);
    let data = resp.json()["data"].as_array().unwrap().clone();
    assert_eq!(data.len(), 2, "should see both created documents");
}

// ---------------------------------------------------------------------------
// Non-plugin_storage types are unaffected by token presence
// ---------------------------------------------------------------------------

#[test]
fn non_plugin_storage_type_works_without_token() {
    let dir = TempDir::new().unwrap();
    let bucket = TempDir::new().unwrap();
    fs::write(dir.path().join("test.js"), "// test").unwrap();

    let ctx = ServerTestContext::new(plugins_config_with_bucket(dir.path(), bucket.path()));

    // Create a regular (non-plugin_storage) document — no Bearer token needed.
    let body = serde_json::json!({
        "type": "note",
        "title": "regular doc",
        "content": {"data": true}
    });
    let resp = post_json(ctx.addr(), "/v1/documents", &body);
    assert_eq!(
        resp.status, 201,
        "non-plugin_storage should work without token"
    );

    // List regular documents — no token needed.
    let resp = get(ctx.addr(), "/v1/documents?type=note");
    assert_eq!(resp.status, 200);
    let data = resp.json()["data"].as_array().unwrap().clone();
    assert_eq!(data.len(), 1);
}

// ===========================================================================
// Proxy validation edge cases
// ===========================================================================

// ---------------------------------------------------------------------------
// Proxy: invalid URL → 400
// ---------------------------------------------------------------------------

#[test]
fn proxy_invalid_url_returns_400() {
    let dir = TempDir::new().unwrap();
    setup_plugin_with_permissions(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let token = discover_and_get_token(ctx.addr(), "net-plugin");

    let body = r#"{"url":"not-a-url","method":"GET"}"#;
    let auth = format!("Bearer {token}");
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/proxy",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", &auth),
        ],
        Some(body),
    );
    assert_eq!(resp.status, 400);
}

// ---------------------------------------------------------------------------
// Proxy: unsupported scheme → 400
// ---------------------------------------------------------------------------

#[test]
fn proxy_unsupported_scheme_returns_400() {
    let dir = TempDir::new().unwrap();
    setup_plugin_with_permissions(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let token = discover_and_get_token(ctx.addr(), "net-plugin");

    let body = r#"{"url":"ftp://api.example.com/file","method":"GET"}"#;
    let auth = format!("Bearer {token}");
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/proxy",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", &auth),
        ],
        Some(body),
    );
    assert_eq!(resp.status, 400);
}

// ---------------------------------------------------------------------------
// Proxy: missing body → 400
// ---------------------------------------------------------------------------

#[test]
fn proxy_missing_body_returns_400() {
    let dir = TempDir::new().unwrap();
    setup_plugin_with_permissions(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let token = discover_and_get_token(ctx.addr(), "net-plugin");

    let auth = format!("Bearer {token}");
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/proxy",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", &auth),
        ],
        None,
    );
    assert_eq!(resp.status, 400);
}

// ---------------------------------------------------------------------------
// Proxy: invalid JSON body → 400
// ---------------------------------------------------------------------------

#[test]
fn proxy_invalid_json_returns_400() {
    let dir = TempDir::new().unwrap();
    setup_plugin_with_permissions(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let token = discover_and_get_token(ctx.addr(), "net-plugin");

    let auth = format!("Bearer {token}");
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/proxy",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", &auth),
        ],
        Some("{not valid json"),
    );
    assert_eq!(resp.status, 400);
}

// ---------------------------------------------------------------------------
// Proxy: wildcard domain matching
// ---------------------------------------------------------------------------

#[test]
fn proxy_wildcard_domain_blocks_non_subdomain() {
    let dir = TempDir::new().unwrap();
    setup_plugin_with_permissions(dir.path()); // has *.google.com

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let token = discover_and_get_token(ctx.addr(), "net-plugin");

    // notgoogle.com is NOT a subdomain of google.com
    let body = r#"{"url":"https://notgoogle.com/api","method":"GET"}"#;
    let auth = format!("Bearer {token}");
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/proxy",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", &auth),
        ],
        Some(body),
    );
    assert_eq!(
        resp.status, 403,
        "notgoogle.com should not match *.google.com"
    );
}

// ---------------------------------------------------------------------------
// Proxy: SSRF with private IP ranges
// ---------------------------------------------------------------------------

#[test]
fn proxy_ssrf_blocks_private_ips() {
    let dir = TempDir::new().unwrap();

    // Plugin with a wildcard network permission.
    let plugin_dir = dir.path().join("wide-plugin");
    fs::create_dir_all(&plugin_dir).unwrap();
    fs::write(
        plugin_dir.join("talu.json"),
        r#"{"id":"wide-plugin","permissions":["network:*"]}"#,
    )
    .unwrap();
    fs::write(plugin_dir.join("index.js"), "// wide").unwrap();

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let token = discover_and_get_token(ctx.addr(), "wide-plugin");
    let auth = format!("Bearer {token}");

    let private_urls = [
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
    ];

    for url in &private_urls {
        let body = format!(r#"{{"url":"{}","method":"GET"}}"#, url);
        let resp = send_request(
            ctx.addr(),
            "POST",
            "/v1/proxy",
            &[
                ("Content-Type", "application/json"),
                ("Authorization", &auth),
            ],
            Some(&body),
        );
        assert_eq!(resp.status, 403, "SSRF should block {url}");
    }
}

// ---------------------------------------------------------------------------
// Proxy: dual-path routing (/proxy without /v1)
// ---------------------------------------------------------------------------

#[test]
fn proxy_works_without_v1_prefix() {
    let dir = TempDir::new().unwrap();
    setup_plugin_with_permissions(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));

    // Without token — should still return 401 (not 404).
    let body = r#"{"url":"https://api.example.com/test","method":"GET"}"#;
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/proxy",
        &[("Content-Type", "application/json")],
        Some(body),
    );
    assert_eq!(resp.status, 401, "/proxy should route correctly (not 404)");
}

// ---------------------------------------------------------------------------
// Proxy: plugin with no network permissions → 403
// ---------------------------------------------------------------------------

#[test]
fn proxy_no_permissions_returns_403() {
    let dir = TempDir::new().unwrap();

    // Plugin without any network: permissions.
    let plugin_dir = dir.path().join("no-net");
    fs::create_dir_all(&plugin_dir).unwrap();
    fs::write(
        plugin_dir.join("talu.json"),
        r#"{"id":"no-net","permissions":["storage","ui"]}"#,
    )
    .unwrap();
    fs::write(plugin_dir.join("index.js"), "// no-net").unwrap();

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let token = discover_and_get_token(ctx.addr(), "no-net");

    let body = r#"{"url":"https://api.example.com/test","method":"GET"}"#;
    let auth = format!("Bearer {token}");
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/proxy",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", &auth),
        ],
        Some(body),
    );
    assert_eq!(
        resp.status, 403,
        "plugin without network perms should be denied"
    );
}

// ---------------------------------------------------------------------------
// Proxy: single-file plugin (no manifest) → 403 for proxy
// ---------------------------------------------------------------------------

#[test]
fn proxy_single_file_plugin_no_manifest_returns_403() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("bare.js"), "// bare").unwrap();

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let token = discover_and_get_token(ctx.addr(), "bare");

    let body = r#"{"url":"https://api.example.com/test","method":"GET"}"#;
    let auth = format!("Bearer {token}");
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/proxy",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", &auth),
        ],
        Some(body),
    );
    assert_eq!(
        resp.status, 403,
        "single-file plugin has no manifest → no network permissions → 403"
    );
}

// ---------------------------------------------------------------------------
// Proxy: Bearer prefix without token value → 401
// ---------------------------------------------------------------------------

#[test]
fn proxy_bearer_prefix_only_returns_401() {
    let dir = TempDir::new().unwrap();
    setup_plugin_with_permissions(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let body = r#"{"url":"https://api.example.com/v1/test","method":"GET"}"#;
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/proxy",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", "Bearer "),
        ],
        Some(body),
    );
    assert_eq!(resp.status, 401);
}

// ---------------------------------------------------------------------------
// Proxy: non-Bearer auth scheme → 401
// ---------------------------------------------------------------------------

#[test]
fn proxy_non_bearer_scheme_returns_401() {
    let dir = TempDir::new().unwrap();
    setup_plugin_with_permissions(dir.path());

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let body = r#"{"url":"https://api.example.com/v1/test","method":"GET"}"#;
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/proxy",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", "Basic dXNlcjpwYXNz"),
        ],
        Some(body),
    );
    assert_eq!(resp.status, 401);
}
