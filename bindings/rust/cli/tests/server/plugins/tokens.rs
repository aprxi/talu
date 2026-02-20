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
fn proxy_ssrf_blocked_without_explicit_private_permission() {
    let dir = TempDir::new().unwrap();

    // Plugin with permission for an external domain only — no private-IP grant.
    let plugin_dir = dir.path().join("ssrf-plugin");
    fs::create_dir_all(&plugin_dir).unwrap();
    fs::write(
        plugin_dir.join("talu.json"),
        r#"{"id":"ssrf-plugin","permissions":["network:api.example.com"]}"#,
    )
    .unwrap();
    fs::write(plugin_dir.join("index.js"), "// ssrf").unwrap();

    let ctx = ServerTestContext::new(plugins_config(dir.path()));
    let token = discover_and_get_token(ctx.addr(), "ssrf-plugin");

    // Attempt to reach localhost — plugin only has permission for api.example.com.
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
        "SSRF should block private IPs when plugin lacks explicit permission"
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

// ---------------------------------------------------------------------------
// Proxy forwarding (with mock upstream server)
// ---------------------------------------------------------------------------

/// Spin up a minimal HTTP echo server that reads one request and responds
/// with a JSON body echoing the method, path, headers, and body it received.
fn spawn_echo_server() -> (std::net::SocketAddr, std::thread::JoinHandle<()>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind echo server");
    let addr = listener.local_addr().expect("echo server addr");

    let handle = std::thread::spawn(move || {
        // Accept up to 10 connections (enough for our tests).
        for _ in 0..10 {
            let Ok((mut stream, _)) = listener.accept() else {
                break;
            };
            use std::io::{BufRead, BufReader, Write};

            let mut reader = BufReader::new(stream.try_clone().unwrap());

            // Read request line.
            let mut request_line = String::new();
            if reader.read_line(&mut request_line).is_err() {
                continue;
            }
            let parts: Vec<&str> = request_line.trim().splitn(3, ' ').collect();
            let method = parts.first().unwrap_or(&"").to_string();
            let path = parts.get(1).unwrap_or(&"").to_string();

            // Read headers.
            let mut headers = std::collections::HashMap::new();
            let mut content_length: usize = 0;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
                    break;
                }
                if let Some((key, value)) = line.trim().split_once(':') {
                    let key_lower = key.trim().to_lowercase();
                    let val = value.trim().to_string();
                    if key_lower == "content-length" {
                        content_length = val.parse().unwrap_or(0);
                    }
                    headers.insert(key.trim().to_string(), val);
                }
            }

            // Read body.
            let mut body_bytes = vec![0u8; content_length];
            if content_length > 0 {
                use std::io::Read;
                let _ = reader.read_exact(&mut body_bytes);
            }
            let body = String::from_utf8_lossy(&body_bytes).to_string();

            // Respond with JSON echo.
            let echo = serde_json::json!({
                "echo_method": method,
                "echo_path": path,
                "echo_headers": headers,
                "echo_body": body,
            });
            let resp_body = serde_json::to_string(&echo).unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\n\
                 Content-Type: application/json\r\n\
                 Cache-Control: max-age=60\r\n\
                 ETag: \"echo-etag-123\"\r\n\
                 Content-Length: {}\r\n\
                 Connection: close\r\n\r\n{}",
                resp_body.len(),
                resp_body,
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        }
    });

    (addr, handle)
}

/// Create a plugin config for proxy forwarding tests.
///
/// The echo server binds to 127.0.0.1 and the plugin manifest grants
/// `network:127.0.0.1`.  The proxy handler respects explicit operator-granted
/// permissions, so no SSRF bypass is needed.
fn proxy_forwarding_config(plugins_dir: &std::path::Path) -> ServerConfig {
    plugins_config(plugins_dir)
}

/// Setup a plugin with network permissions for a specific host.
fn setup_plugin_with_host_permission(dir: &std::path::Path, host: &str) {
    let plugin_dir = dir.join("proxy-fwd-plugin");
    fs::create_dir_all(&plugin_dir).unwrap();
    fs::write(
        plugin_dir.join("talu.json"),
        serde_json::json!({
            "id": "proxy-fwd-plugin",
            "activationEvents": ["*"],
            "permissions": [format!("network:{}", host)]
        })
        .to_string(),
    )
    .unwrap();
    fs::write(plugin_dir.join("index.js"), "export default function(){}").unwrap();
}


/// Proxy forwards POST body to upstream.
#[test]
fn proxy_forwards_post_body() {
    let (echo_addr, _handle) = spawn_echo_server();
    let dir = TempDir::new().unwrap();
    setup_plugin_with_host_permission(dir.path(), "127.0.0.1");

    let ctx = ServerTestContext::new(proxy_forwarding_config(dir.path()));
    let token = discover_and_get_token(ctx.addr(), "proxy-fwd-plugin");

    let proxy_body = serde_json::json!({
        "url": format!("http://{}/test-path", echo_addr),
        "method": "POST",
        "headers": {"Content-Type": "text/plain"},
        "body": "hello proxy body"
    });
    let proxy_json = serde_json::to_string(&proxy_body).unwrap();
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/proxy",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", &format!("Bearer {}", token)),
        ],
        Some(&proxy_json),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    assert_eq!(json["status"], 200, "upstream should return 200");

    // Parse the upstream response body (which is the echo JSON).
    let upstream_body: serde_json::Value =
        serde_json::from_str(json["body"].as_str().unwrap()).expect("parse echo body");
    assert_eq!(upstream_body["echo_method"], "POST");
    assert_eq!(upstream_body["echo_body"], "hello proxy body");
}

/// Proxy forwards custom headers and strips internal ones (X-Talu-*, Cookie, Host).
#[test]
fn proxy_forwards_custom_headers_and_strips_internal() {
    let (echo_addr, _handle) = spawn_echo_server();
    let dir = TempDir::new().unwrap();
    setup_plugin_with_host_permission(dir.path(), "127.0.0.1");

    let ctx = ServerTestContext::new(proxy_forwarding_config(dir.path()));
    let token = discover_and_get_token(ctx.addr(), "proxy-fwd-plugin");

    let proxy_body = serde_json::json!({
        "url": format!("http://{}/headers-test", echo_addr),
        "method": "GET",
        "headers": {
            "X-Custom-Auth": "my-token",
            "X-Talu-Internal": "secret-value",
            "Cookie": "session=abc123",
            "Host": "evil.com",
            "Accept": "application/json"
        }
    });
    let proxy_json = serde_json::to_string(&proxy_body).unwrap();
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/proxy",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", &format!("Bearer {}", token)),
        ],
        Some(&proxy_json),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let upstream_body: serde_json::Value =
        serde_json::from_str(json["body"].as_str().unwrap()).expect("parse echo body");
    let echo_headers = &upstream_body["echo_headers"];

    // Custom header should be forwarded.
    assert_eq!(
        echo_headers["x-custom-auth"].as_str().or_else(|| echo_headers["X-Custom-Auth"].as_str()),
        Some("my-token"),
        "custom header should be forwarded, got headers: {}",
        echo_headers
    );
    assert_eq!(
        echo_headers["accept"].as_str().or_else(|| echo_headers["Accept"].as_str()),
        Some("application/json"),
        "Accept header should be forwarded"
    );

    // Internal headers should NOT be forwarded.
    assert!(
        echo_headers["X-Talu-Internal"].is_null() && echo_headers["x-talu-internal"].is_null(),
        "X-Talu-Internal should be stripped"
    );
    assert!(
        echo_headers["Cookie"].is_null() && echo_headers["cookie"].is_null(),
        "Cookie should be stripped"
    );
}

/// Proxy response includes selectively-forwarded upstream headers.
#[test]
fn proxy_response_includes_upstream_headers() {
    let (echo_addr, _handle) = spawn_echo_server();
    let dir = TempDir::new().unwrap();
    setup_plugin_with_host_permission(dir.path(), "127.0.0.1");

    let ctx = ServerTestContext::new(proxy_forwarding_config(dir.path()));
    let token = discover_and_get_token(ctx.addr(), "proxy-fwd-plugin");

    let proxy_body = serde_json::json!({
        "url": format!("http://{}/resp-headers", echo_addr),
        "method": "GET"
    });
    let proxy_json = serde_json::to_string(&proxy_body).unwrap();
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/proxy",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", &format!("Bearer {}", token)),
        ],
        Some(&proxy_json),
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);

    let json = resp.json();
    let headers = &json["headers"];

    // The echo server responds with content-type, cache-control, and etag.
    // The proxy should selectively forward these 3 headers (lines 220-226 in proxy.rs).
    assert_eq!(
        headers["content-type"].as_str(),
        Some("application/json"),
        "should forward content-type"
    );
    assert_eq!(
        headers["cache-control"].as_str(),
        Some("max-age=60"),
        "should forward cache-control"
    );
    assert_eq!(
        headers["etag"].as_str(),
        Some("\"echo-etag-123\""),
        "should forward etag"
    );
}

/// Wildcard permission `*.X` also matches the root domain `X` itself.
///
/// Uses the local echo server with permission `*.127.0.0.1` and URL host
/// `127.0.0.1`. This exercises the `domain_lower == suffix` branch in
/// `is_domain_allowed` (proxy.rs:305) without any outbound network call.
#[test]
fn proxy_wildcard_matches_root_domain() {
    let (echo_addr, _handle) = spawn_echo_server();
    let dir = TempDir::new().unwrap();

    // Permission is *.127.0.0.1 — the URL host is 127.0.0.1 (the root).
    // is_domain_allowed strips "*." to get suffix "127.0.0.1", then checks
    // domain_lower == suffix, which is true.
    let plugin_dir = dir.path().join("proxy-wild-plugin");
    fs::create_dir_all(&plugin_dir).unwrap();
    fs::write(
        plugin_dir.join("talu.json"),
        r#"{"id":"proxy-wild-plugin","activationEvents":["*"],"permissions":["network:*.127.0.0.1"]}"#,
    )
    .unwrap();
    fs::write(plugin_dir.join("index.js"), "export default function(){}").unwrap();

    let ctx = ServerTestContext::new(proxy_forwarding_config(dir.path()));
    let token = discover_and_get_token(ctx.addr(), "proxy-wild-plugin");

    let proxy_body = serde_json::json!({
        "url": format!("http://{}/wildcard-root", echo_addr),
        "method": "GET"
    });
    let proxy_json = serde_json::to_string(&proxy_body).unwrap();
    let resp = send_request(
        ctx.addr(),
        "POST",
        "/v1/proxy",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", &format!("Bearer {}", token)),
        ],
        Some(&proxy_json),
    );
    // Should be 200 (echo server responds), NOT 403 (domain check failed).
    assert_eq!(
        resp.status, 200,
        "wildcard *.127.0.0.1 should match root domain 127.0.0.1, body: {}",
        resp.body
    );

    // Verify we actually hit the echo server.
    let json = resp.json();
    let upstream_body: serde_json::Value =
        serde_json::from_str(json["body"].as_str().unwrap()).expect("parse echo body");
    assert_eq!(upstream_body["echo_method"], "GET");
}
