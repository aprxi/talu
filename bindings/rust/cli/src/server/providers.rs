use std::convert::Infallible;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde_json::json;

use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

/// Resolve the KV root path from the app state's bucket path.
/// Returns `None` if storage is disabled (no bucket).
fn kv_root(state: &AppState) -> Option<String> {
    state
        .bucket_path
        .as_ref()
        .map(|b| b.join("kv").to_string_lossy().into_owned())
}

/// GET /v1/providers — list all providers with their runtime configuration.
pub async fn handle_list(
    state: Arc<AppState>,
    _req: Request<Incoming>,
    _auth_ctx: Option<AuthContext>,
) -> Response<BoxBody> {
    let db_root = match kv_root(&state) {
        Some(r) => r,
        None => return json_error(StatusCode::SERVICE_UNAVAILABLE, "Storage not configured"),
    };

    let providers =
        match tokio::task::spawn_blocking(move || talu::provider_config_list(&db_root)).await {
            Ok(Ok(list)) => list,
            Ok(Err(e)) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}")),
            Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}")),
        };

    let payload = json!({ "providers": providers_to_json(&providers) });
    json_response(StatusCode::OK, &payload)
}

/// PATCH /v1/providers/{name} — update configuration for a named provider.
///
/// Merge semantics: omitted fields are kept, `null` clears, values set.
/// Body: `{ "enabled"?: bool, "api_key"?: string|null, "base_url"?: string|null }`
pub async fn handle_update(
    state: Arc<AppState>,
    req: Request<Incoming>,
    _auth_ctx: Option<AuthContext>,
    provider_name: String,
) -> Response<BoxBody> {
    let db_root = match kv_root(&state) {
        Some(r) => r,
        None => return json_error(StatusCode::SERVICE_UNAVAILABLE, "Storage not configured"),
    };

    let body_bytes = match req.into_body().collect().await {
        Ok(body) => body.to_bytes(),
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "Invalid body"),
    };

    let patch: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(val) => val,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &format!("Invalid JSON: {e}")),
    };

    // Trinary enabled: missing = keep existing, true/false = set.
    let enabled: Option<bool> = patch.get("enabled").and_then(|v| v.as_bool());

    // Distinguish missing (keep) vs null (clear) vs string (set).
    // None = field missing → keep existing.  Some("") = JSON null → clear.  Some(val) = set.
    let api_key: Option<String> = match patch.get("api_key") {
        None => None,
        Some(serde_json::Value::Null) => Some(String::new()),
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(_) => return json_error(StatusCode::BAD_REQUEST, "api_key must be a string or null"),
    };
    let base_url: Option<String> = match patch.get("base_url") {
        None => None,
        Some(serde_json::Value::Null) => Some(String::new()),
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(_) => {
            return json_error(StatusCode::BAD_REQUEST, "base_url must be a string or null")
        }
    };

    let name = provider_name.clone();
    let db_root_set = db_root.clone();
    let result = tokio::task::spawn_blocking(move || {
        talu::provider_config_set(
            &db_root_set,
            &name,
            enabled,
            api_key.as_deref(),
            base_url.as_deref(),
        )
    })
    .await;

    match result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return json_error(StatusCode::BAD_REQUEST, &format!("{e}")),
        Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}")),
    }

    // Return updated provider list.
    let providers =
        match tokio::task::spawn_blocking(move || talu::provider_config_list(&db_root)).await {
            Ok(Ok(list)) => list,
            Ok(Err(e)) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}")),
            Err(e) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}")),
        };

    let payload = json!({ "providers": providers_to_json(&providers) });
    json_response(StatusCode::OK, &payload)
}

/// POST /v1/providers/{name}/health — check connectivity to a provider's endpoint.
pub async fn handle_health(
    state: Arc<AppState>,
    _req: Request<Incoming>,
    _auth_ctx: Option<AuthContext>,
    provider_name: String,
) -> Response<BoxBody> {
    let db_root = match kv_root(&state) {
        Some(r) => r,
        None => return json_error(StatusCode::SERVICE_UNAVAILABLE, "Storage not configured"),
    };

    let result = tokio::task::spawn_blocking(move || {
        talu::provider_config_health(&db_root, &provider_name)
    })
    .await;

    match result {
        Ok(health) => {
            let payload = if health.ok {
                json!({ "ok": true, "model_count": health.model_count })
            } else {
                json!({ "ok": false, "error": health.error_message })
            };
            json_response(StatusCode::OK, &payload)
        }
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}")),
    }
}

/// GET /v1/providers/{name}/models — list models from a single provider.
pub async fn handle_list_models(
    state: Arc<AppState>,
    _req: Request<Incoming>,
    _auth_ctx: Option<AuthContext>,
    provider_name: String,
) -> Response<BoxBody> {
    let db_root = match kv_root(&state) {
        Some(r) => r,
        None => return json_error(StatusCode::SERVICE_UNAVAILABLE, "Storage not configured"),
    };

    let result = tokio::task::spawn_blocking(move || {
        talu::provider_config_list_provider_models(&db_root, &provider_name)
    })
    .await;

    match result {
        Ok(Ok(models)) => {
            let models_json: Vec<serde_json::Value> = models
                .iter()
                .map(|m| {
                    json!({
                        "id": m.id,
                        "object": m.object,
                        "created": m.created,
                        "owned_by": m.owned_by,
                    })
                })
                .collect();
            json_response(StatusCode::OK, &json!({ "models": models_json }))
        }
        Ok(Err(e)) => json_error(StatusCode::BAD_REQUEST, &format!("{e}")),
        Err(e) => json_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e}")),
    }
}

fn providers_to_json(providers: &[talu::ProviderWithConfig]) -> Vec<serde_json::Value> {
    providers
        .iter()
        .map(|p| {
            json!({
                "name": p.name,
                "default_endpoint": p.default_endpoint,
                "api_key_env": p.api_key_env,
                "enabled": p.enabled,
                "has_api_key": p.has_api_key,
                "base_url_override": p.base_url_override,
                "effective_endpoint": p.effective_endpoint,
            })
        })
        .collect()
}

fn json_response(status: StatusCode, payload: &serde_json::Value) -> Response<BoxBody> {
    let body = serde_json::to_vec(payload).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}

fn json_error(status: StatusCode, message: &str) -> Response<BoxBody> {
    let payload = json!({
        "error": {
            "code": "provider_error",
            "message": message
        }
    });
    let body = serde_json::to_vec(&payload).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}
