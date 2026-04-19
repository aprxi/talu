use crate::server::common::{get, send_request, ServerConfig, ServerTestContext, TenantSpec};

#[test]
fn no_auth_allows_request_without_headers() {
    let ctx = ServerTestContext::new(ServerConfig::new());
    let response = get(ctx.addr(), "/v1/models");
    assert_eq!(response.status, 200, "body: {}", response.body);
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
    let response = get(ctx.addr(), "/v1/models");
    assert_eq!(response.status, 401, "body: {}", response.body);
    assert_eq!(response.json()["error"]["code"], "unauthorized");
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
    assert_eq!(response.status, 401, "body: {}", response.body);
    assert_eq!(response.json()["error"]["code"], "unauthorized");
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
    assert_eq!(response.status, 403, "body: {}", response.body);
    assert_eq!(response.json()["error"]["code"], "forbidden");
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
    assert_eq!(response.status, 403, "body: {}", response.body);
    assert_eq!(response.json()["error"]["code"], "forbidden");
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
    assert_eq!(response.status, 200, "body: {}", response.body);
}

#[test]
fn auth_filters_models_by_tenant_allowlist() {
    let mut config = ServerConfig::new();
    config.gateway_secret = Some("secret".to_string());
    config.tenants = vec![TenantSpec {
        id: "acme".to_string(),
        storage_prefix: "acme".to_string(),
        allowed_models: vec![
            "allow-1".to_string(),
            "allow-2".to_string(),
            "Qwen/Qwen3.5-4B-NVFP4".to_string(),
        ],
    }];

    let ctx = ServerTestContext::new(config);
    let headers = [
        ("X-Talu-Gateway-Secret", "secret"),
        ("X-Talu-Tenant-Id", "acme"),
    ];
    let response = send_request(ctx.addr(), "GET", "/v1/models", &headers, None);
    assert_eq!(response.status, 200, "body: {}", response.body);

    let payload = response.json();
    let models = payload["data"].as_array().expect("data array");
    for model in models {
        let id = model["id"].as_str().expect("model id string");
        assert!(
            matches!(id, "allow-1" | "allow-2" | "Qwen/Qwen3.5-4B-NVFP4"),
            "model {id} is outside tenant allowlist"
        );
    }
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
    let response = get(ctx.addr(), "/health");
    assert_eq!(response.status, 200, "body: {}", response.body);
    assert_eq!(response.body, "ok");
}
