//! Trusted-gateway authentication for multi-tenant deployments.
//!
//! # Trust model
//!
//! This module implements a **trusted gateway** pattern. The talu server
//! never faces end users directly — it binds to localhost / a Unix socket
//! and sits behind a gateway (reverse proxy, API gateway, etc.) that:
//!
//! 1. Authenticates end users (OAuth, API keys, sessions, …).
//! 2. Resolves the authenticated user to a tenant.
//! 3. Forwards the request to talu with internal headers:
//!    - `X-Talu-Gateway-Secret` — shared secret proving the request
//!      came from the gateway, not from an end user.
//!    - `X-Talu-Tenant-Id` — the tenant the gateway resolved.
//!    - `X-Talu-Group-Id` / `X-Talu-User-Id` — optional scoping.
//!
//! The shared secret is **not** a per-user credential. Any request bearing
//! the correct secret is trusted to set arbitrary tenant/group/user headers.
//! This is safe because only the gateway possesses the secret, and the
//! server is not network-reachable by end users.
//!
//! Tenant isolation is enforced at the storage layer: each tenant's data
//! lives under `<bucket>/<storage_prefix>/`, so a request scoped to one
//! tenant physically cannot read or write another tenant's data.

use hyper::HeaderMap;

use crate::server::tenant::TenantRegistry;

pub const HEADER_GATEWAY_SECRET: &str = "x-talu-gateway-secret";
pub const HEADER_TENANT_ID: &str = "x-talu-tenant-id";
pub const HEADER_GROUP_ID: &str = "x-talu-group-id";
pub const HEADER_USER_ID: &str = "x-talu-user-id";

#[derive(Debug, Clone)]
pub struct AuthContext {
    pub tenant_id: String,
    pub storage_prefix: String,
    pub group_id: Option<String>,
    pub user_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthError {
    MissingSecret,
    InvalidSecret,
    MissingTenant,
    UnknownTenant,
}

/// Validate that a request came from the trusted gateway.
///
/// Checks the shared secret (constant-time), then resolves the tenant
/// from the `X-Talu-Tenant-Id` header against the registry. The gateway
/// is trusted to set the correct tenant — see module-level docs for the
/// trust model.
pub fn validate_request(
    headers: &HeaderMap,
    secret: &str,
    registry: &TenantRegistry,
) -> Result<AuthContext, AuthError> {
    let provided = header_str(headers, HEADER_GATEWAY_SECRET).ok_or(AuthError::MissingSecret)?;
    if !constant_time_eq(provided, secret) {
        return Err(AuthError::InvalidSecret);
    }

    let tenant_id = header_str(headers, HEADER_TENANT_ID)
        .ok_or(AuthError::MissingTenant)?
        .to_string();
    let tenant = registry.get(&tenant_id).ok_or(AuthError::UnknownTenant)?;

    let group_id = header_str(headers, HEADER_GROUP_ID).map(|val| val.to_string());
    let user_id = header_str(headers, HEADER_USER_ID).map(|val| val.to_string());

    Ok(AuthContext {
        tenant_id: tenant.id.clone(),
        storage_prefix: tenant.storage_prefix.clone(),
        group_id,
        user_id,
    })
}

fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}

fn constant_time_eq(a: &str, b: &str) -> bool {
    let a = a.as_bytes();
    let b = b.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    let mut acc = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        acc |= x ^ y;
    }
    acc == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::header::HeaderValue;

    fn registry() -> TenantRegistry {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "acme".to_string(),
            crate::server::tenant::Tenant {
                id: "acme".to_string(),
                storage_prefix: "acme".to_string(),
                allowed_models: vec!["model-a".to_string()],
            },
        );
        TenantRegistry::from_map(map)
    }

    #[test]
    fn constant_time_eq_matches() {
        assert!(constant_time_eq("secret", "secret"));
        assert!(!constant_time_eq("secret", "other"));
        assert!(!constant_time_eq("short", "longer"));
    }

    #[test]
    fn validate_request_success() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_GATEWAY_SECRET, HeaderValue::from_static("secret"));
        headers.insert(HEADER_TENANT_ID, HeaderValue::from_static("acme"));
        headers.insert(HEADER_GROUP_ID, HeaderValue::from_static("engineering"));
        headers.insert(HEADER_USER_ID, HeaderValue::from_static("user-1"));

        let ctx = validate_request(&headers, "secret", &registry()).expect("auth ok");
        assert_eq!(ctx.tenant_id, "acme");
        assert_eq!(ctx.storage_prefix, "acme");
        assert_eq!(ctx.group_id.as_deref(), Some("engineering"));
        assert_eq!(ctx.user_id.as_deref(), Some("user-1"));
    }

    #[test]
    fn validate_request_missing_secret() {
        let headers = HeaderMap::new();
        let err = validate_request(&headers, "secret", &registry()).unwrap_err();
        assert_eq!(err, AuthError::MissingSecret);
    }

    #[test]
    fn validate_request_invalid_secret() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_GATEWAY_SECRET, HeaderValue::from_static("bad"));
        let err = validate_request(&headers, "secret", &registry()).unwrap_err();
        assert_eq!(err, AuthError::InvalidSecret);
    }

    #[test]
    fn validate_request_missing_tenant() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_GATEWAY_SECRET, HeaderValue::from_static("secret"));
        let err = validate_request(&headers, "secret", &registry()).unwrap_err();
        assert_eq!(err, AuthError::MissingTenant);
    }

    #[test]
    fn validate_request_unknown_tenant() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_GATEWAY_SECRET, HeaderValue::from_static("secret"));
        headers.insert(HEADER_TENANT_ID, HeaderValue::from_static("unknown"));
        let err = validate_request(&headers, "secret", &registry()).unwrap_err();
        assert_eq!(err, AuthError::UnknownTenant);
    }
}
