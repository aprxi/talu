use super::common::{send_request, ServerConfig, ServerTestContext, TenantSpec};
use super::conversations::{seed_session, seed_session_with_group};
use tempfile::TempDir;

#[test]
fn no_auth_allows_request_without_headers() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let response = send_request(ctx.addr(), "POST", "/v1/responses", &[], Some("not-json"));
    assert_eq!(response.status, 400);
}

#[test]
fn auth_rejects_missing_secret() {
    let mut config = ServerConfig::new();
    config.gateway_secret = Some("secret".to_string());
    config.tenants = vec![TenantSpec {
        id: "acme".to_string(),
        storage_prefix: "acme".to_string(),
        allowed_models: vec![],
    }];

    let ctx = ServerTestContext::new(config);
    let response = send_request(ctx.addr(), "GET", "/v1/models", &[], None);
    assert_eq!(response.status, 401);
}

#[test]
fn auth_rejects_invalid_secret() {
    let mut config = ServerConfig::new();
    config.gateway_secret = Some("secret".to_string());
    config.tenants = vec![TenantSpec {
        id: "acme".to_string(),
        storage_prefix: "acme".to_string(),
        allowed_models: vec![],
    }];

    let ctx = ServerTestContext::new(config);
    let headers = [("X-Talu-Gateway-Secret", "wrong")];
    let response = send_request(ctx.addr(), "GET", "/v1/models", &headers, None);
    assert_eq!(response.status, 401);
}

#[test]
fn auth_rejects_missing_tenant() {
    let mut config = ServerConfig::new();
    config.gateway_secret = Some("secret".to_string());
    config.tenants = vec![TenantSpec {
        id: "acme".to_string(),
        storage_prefix: "acme".to_string(),
        allowed_models: vec![],
    }];

    let ctx = ServerTestContext::new(config);
    let headers = [("X-Talu-Gateway-Secret", "secret")];
    let response = send_request(ctx.addr(), "GET", "/v1/models", &headers, None);
    assert_eq!(response.status, 403);
}

#[test]
fn auth_rejects_unknown_tenant() {
    let mut config = ServerConfig::new();
    config.gateway_secret = Some("secret".to_string());
    config.tenants = vec![TenantSpec {
        id: "acme".to_string(),
        storage_prefix: "acme".to_string(),
        allowed_models: vec![],
    }];

    let ctx = ServerTestContext::new(config);
    let headers = [
        ("X-Talu-Gateway-Secret", "secret"),
        ("X-Talu-Tenant-Id", "unknown"),
    ];
    let response = send_request(ctx.addr(), "GET", "/v1/models", &headers, None);
    assert_eq!(response.status, 403);
}

#[test]
fn auth_allows_valid_tenant() {
    let mut config = ServerConfig::new();
    config.gateway_secret = Some("secret".to_string());
    config.tenants = vec![TenantSpec {
        id: "acme".to_string(),
        storage_prefix: "acme".to_string(),
        allowed_models: vec![],
    }];

    let ctx = ServerTestContext::new(config);
    let headers = [
        ("X-Talu-Gateway-Secret", "secret"),
        ("X-Talu-Tenant-Id", "acme"),
    ];
    let response = send_request(ctx.addr(), "GET", "/v1/models", &headers, None);
    assert_eq!(response.status, 200);
}

#[test]
fn auth_health_exempt() {
    let mut config = ServerConfig::new();
    config.gateway_secret = Some("secret".to_string());
    config.tenants = vec![TenantSpec {
        id: "acme".to_string(),
        storage_prefix: "acme".to_string(),
        allowed_models: vec![],
    }];

    let ctx = ServerTestContext::new(config);
    let response = send_request(ctx.addr(), "GET", "/health", &[], None);
    assert_eq!(response.status, 200);
}

#[test]
fn auth_filters_models_by_tenant() {
    let mut config = ServerConfig::new();
    config.model = Some("openai::gpt-4o".to_string());
    config.gateway_secret = Some("secret".to_string());
    config.tenants = vec![
        TenantSpec {
            id: "tenant-a".to_string(),
            storage_prefix: "tenant-a".to_string(),
            allowed_models: vec!["openai::gpt-4o".to_string()],
        },
        TenantSpec {
            id: "tenant-b".to_string(),
            storage_prefix: "tenant-b".to_string(),
            allowed_models: vec!["other-model".to_string()],
        },
    ];

    let ctx = ServerTestContext::new(config);

    let headers = [
        ("X-Talu-Gateway-Secret", "secret"),
        ("X-Talu-Tenant-Id", "tenant-a"),
    ];
    let response = send_request(ctx.addr(), "GET", "/v1/models", &headers, None);
    assert_eq!(response.status, 200);
    let payload: serde_json::Value = serde_json::from_str(&response.body).expect("json");
    let data = payload
        .get("data")
        .and_then(|val| val.as_array())
        .expect("data array");
    assert_eq!(data.len(), 1);
    assert_eq!(
        data[0].get("id").and_then(|val| val.as_str()),
        Some("openai::gpt-4o")
    );

    let headers = [
        ("X-Talu-Gateway-Secret", "secret"),
        ("X-Talu-Tenant-Id", "tenant-b"),
    ];
    let response = send_request(ctx.addr(), "GET", "/v1/models", &headers, None);
    assert_eq!(response.status, 200);
    let payload: serde_json::Value = serde_json::from_str(&response.body).expect("json");
    let data = payload
        .get("data")
        .and_then(|val| val.as_array())
        .expect("data array");
    assert!(data.is_empty());
}

// ---------------------------------------------------------------------------
// Tenant storage isolation
// ---------------------------------------------------------------------------

#[test]
fn auth_tenant_storage_isolation() {
    let temp = TempDir::new().expect("temp dir");

    // Seed sessions into tenant-specific subdirectories.
    let acme_dir = temp.path().join("acme");
    let globex_dir = temp.path().join("globex");
    seed_session(&acme_dir, "sess-acme-1", "Acme Chat", "m");
    seed_session(&globex_dir, "sess-globex-1", "Globex Chat", "m");
    seed_session(&globex_dir, "sess-globex-2", "Globex Chat 2", "m");

    let mut config = ServerConfig::new();
    config.gateway_secret = Some("secret".to_string());
    config.bucket = Some(temp.path().to_path_buf());
    config.tenants = vec![
        TenantSpec {
            id: "acme".to_string(),
            storage_prefix: "acme".to_string(),
            allowed_models: vec![],
        },
        TenantSpec {
            id: "globex".to_string(),
            storage_prefix: "globex".to_string(),
            allowed_models: vec![],
        },
    ];

    let ctx = ServerTestContext::new(config);

    // Acme should only see its own session.
    let headers_acme = [
        ("X-Talu-Gateway-Secret", "secret"),
        ("X-Talu-Tenant-Id", "acme"),
    ];
    let resp = send_request(ctx.addr(), "GET", "/v1/conversations", &headers_acme, None);
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 1, "acme should see 1 session");
    assert_eq!(data[0]["id"], "sess-acme-1");

    // Globex should see its own two sessions.
    let headers_globex = [
        ("X-Talu-Gateway-Secret", "secret"),
        ("X-Talu-Tenant-Id", "globex"),
    ];
    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/conversations",
        &headers_globex,
        None,
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 2, "globex should see 2 sessions");
    let ids: Vec<&str> = data.iter().map(|s| s["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"sess-globex-1"));
    assert!(ids.contains(&"sess-globex-2"));
}

#[test]
fn auth_group_id_header_filters_conversation_list() {
    let temp = TempDir::new().expect("temp dir");

    // Seed sessions with different group_ids into the tenant's subdirectory.
    let acme_dir = temp.path().join("acme");
    seed_session_with_group(&acme_dir, "sess-g1", "Chat 1", "m", "team-a");
    seed_session_with_group(&acme_dir, "sess-g2", "Chat 2", "m", "team-a");
    seed_session_with_group(&acme_dir, "sess-g3", "Chat 3", "m", "team-b");

    let mut config = ServerConfig::new();
    config.gateway_secret = Some("secret".to_string());
    config.bucket = Some(temp.path().to_path_buf());
    config.tenants = vec![TenantSpec {
        id: "acme".to_string(),
        storage_prefix: "acme".to_string(),
        allowed_models: vec![],
    }];

    let ctx = ServerTestContext::new(config);

    // Request with X-Talu-Group-Id: team-a → should see 2 sessions.
    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/conversations",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "acme"),
            ("X-Talu-Group-Id", "team-a"),
        ],
        None,
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 2, "should see 2 sessions for team-a");
    let ids: Vec<&str> = data.iter().map(|s| s["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"sess-g1"));
    assert!(ids.contains(&"sess-g2"));

    // Request with X-Talu-Group-Id: team-b → should see 1 session.
    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/conversations",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "acme"),
            ("X-Talu-Group-Id", "team-b"),
        ],
        None,
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 1, "should see 1 session for team-b");
    assert_eq!(data[0]["id"], "sess-g3");
}

#[test]
fn auth_query_param_group_id_overrides_header() {
    let temp = TempDir::new().expect("temp dir");

    let acme_dir = temp.path().join("acme");
    seed_session_with_group(&acme_dir, "sess-ha", "Chat A", "m", "header-group");
    seed_session_with_group(&acme_dir, "sess-qa", "Chat B", "m", "query-group");

    let mut config = ServerConfig::new();
    config.gateway_secret = Some("secret".to_string());
    config.bucket = Some(temp.path().to_path_buf());
    config.tenants = vec![TenantSpec {
        id: "acme".to_string(),
        storage_prefix: "acme".to_string(),
        allowed_models: vec![],
    }];

    let ctx = ServerTestContext::new(config);

    // Header says "header-group" but query param says "query-group".
    // Query param should take precedence.
    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/conversations?group_id=query-group",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "acme"),
            ("X-Talu-Group-Id", "header-group"),
        ],
        None,
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(data.len(), 1, "query param group_id should override header");
    assert_eq!(data[0]["id"], "sess-qa");
}

#[test]
fn auth_without_group_id_returns_all_tenant_sessions() {
    let temp = TempDir::new().expect("temp dir");

    let acme_dir = temp.path().join("acme");
    seed_session_with_group(&acme_dir, "sess-1", "Chat 1", "m", "group-a");
    seed_session_with_group(&acme_dir, "sess-2", "Chat 2", "m", "group-b");
    seed_session(&acme_dir, "sess-3", "Chat 3", "m"); // no group

    let mut config = ServerConfig::new();
    config.gateway_secret = Some("secret".to_string());
    config.bucket = Some(temp.path().to_path_buf());
    config.tenants = vec![TenantSpec {
        id: "acme".to_string(),
        storage_prefix: "acme".to_string(),
        allowed_models: vec![],
    }];

    let ctx = ServerTestContext::new(config);

    // No group_id header or query param → all sessions for this tenant.
    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/conversations",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "acme"),
        ],
        None,
    );
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let json = resp.json();
    let data = json["data"].as_array().expect("data");
    assert_eq!(
        data.len(),
        3,
        "should see all 3 sessions without group filter"
    );
}

// ---------------------------------------------------------------------------
// Cross-tenant access denial (individual resource operations)
// ---------------------------------------------------------------------------

/// Prove that tenant A cannot GET a session that belongs to tenant B,
/// even if they know the exact session ID. Storage is physically isolated
/// via the storage_prefix subdirectory.
#[test]
fn auth_cross_tenant_get_returns_404() {
    let temp = TempDir::new().expect("temp dir");

    // Seed a session only in globex's subdirectory.
    let globex_dir = temp.path().join("globex");
    seed_session(&globex_dir, "sess-secret", "Globex Secret", "m");

    let mut config = ServerConfig::new();
    config.gateway_secret = Some("secret".to_string());
    config.bucket = Some(temp.path().to_path_buf());
    config.tenants = vec![
        TenantSpec {
            id: "acme".to_string(),
            storage_prefix: "acme".to_string(),
            allowed_models: vec![],
        },
        TenantSpec {
            id: "globex".to_string(),
            storage_prefix: "globex".to_string(),
            allowed_models: vec![],
        },
    ];

    let ctx = ServerTestContext::new(config);

    // Acme tries to GET a session that only exists in Globex's storage.
    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/conversations/sess-secret",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "acme"),
        ],
        None,
    );
    assert_eq!(resp.status, 404, "acme should not see globex session");

    // Globex can access its own session.
    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/conversations/sess-secret",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "globex"),
        ],
        None,
    );
    assert_eq!(resp.status, 200, "globex should see its own session");
    let json = resp.json();
    assert_eq!(json["id"], "sess-secret");
}

/// Prove that tenant A cannot DELETE a session belonging to tenant B.
#[test]
fn auth_cross_tenant_delete_returns_204_without_effect() {
    let temp = TempDir::new().expect("temp dir");

    let globex_dir = temp.path().join("globex");
    seed_session(&globex_dir, "sess-keep", "Keep Me", "m");

    let mut config = ServerConfig::new();
    config.gateway_secret = Some("secret".to_string());
    config.bucket = Some(temp.path().to_path_buf());
    config.tenants = vec![
        TenantSpec {
            id: "acme".to_string(),
            storage_prefix: "acme".to_string(),
            allowed_models: vec![],
        },
        TenantSpec {
            id: "globex".to_string(),
            storage_prefix: "globex".to_string(),
            allowed_models: vec![],
        },
    ];

    let ctx = ServerTestContext::new(config);

    // Acme tries to DELETE a session in globex's storage.
    // DELETE is idempotent: returns 204 even if not found. The key
    // guarantee is that globex's session remains intact.
    let resp = send_request(
        ctx.addr(),
        "DELETE",
        "/v1/conversations/sess-keep",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "acme"),
        ],
        None,
    );
    assert_eq!(
        resp.status, 204,
        "delete is idempotent (returns 204 even for missing)"
    );

    // Verify globex's session is still there.
    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/conversations/sess-keep",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "globex"),
        ],
        None,
    );
    assert_eq!(resp.status, 200, "globex session should still exist");
    let json = resp.json();
    assert_eq!(json["id"], "sess-keep");
    assert_eq!(json["title"], "Keep Me");
}

/// Prove that tenant A cannot PATCH a session belonging to tenant B.
#[test]
fn auth_cross_tenant_patch_returns_404() {
    let temp = TempDir::new().expect("temp dir");

    let globex_dir = temp.path().join("globex");
    seed_session(&globex_dir, "sess-immutable", "Original Title", "m");

    let mut config = ServerConfig::new();
    config.gateway_secret = Some("secret".to_string());
    config.bucket = Some(temp.path().to_path_buf());
    config.tenants = vec![
        TenantSpec {
            id: "acme".to_string(),
            storage_prefix: "acme".to_string(),
            allowed_models: vec![],
        },
        TenantSpec {
            id: "globex".to_string(),
            storage_prefix: "globex".to_string(),
            allowed_models: vec![],
        },
    ];

    let ctx = ServerTestContext::new(config);

    // Acme tries to PATCH a session in globex's storage.
    let resp = send_request(
        ctx.addr(),
        "PATCH",
        "/v1/conversations/sess-immutable",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "acme"),
            ("Content-Type", "application/json"),
        ],
        Some(r#"{"title": "Hacked Title"}"#),
    );
    assert_eq!(
        resp.status, 404,
        "acme should not be able to patch globex session"
    );

    // Verify globex's session title is unchanged.
    let resp = send_request(
        ctx.addr(),
        "GET",
        "/v1/conversations/sess-immutable",
        &[
            ("X-Talu-Gateway-Secret", "secret"),
            ("X-Talu-Tenant-Id", "globex"),
        ],
        None,
    );
    assert_eq!(resp.status, 200);
    let json = resp.json();
    assert_eq!(json["title"], "Original Title", "title should be unchanged");
}
