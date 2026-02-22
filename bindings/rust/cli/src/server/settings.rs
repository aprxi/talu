use std::convert::Infallible;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde::Serialize;
use serde_json::json;
use utoipa::ToSchema;

use crate::bucket_settings::{self, ModelOverrides};
use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

/// Documentation-only type for the settings GET response shape.
#[derive(Serialize, ToSchema)]
pub(crate) struct SettingsResponse {
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub max_output_tokens: Option<u32>,
    pub context_length: Option<u32>,
    pub auto_title: bool,
    pub default_prompt_id: Option<String>,
    pub system_prompt_enabled: bool,
    pub available_models: Vec<ModelEntry>,
}

/// GET /v1/settings — return current bucket settings + enriched model list.
#[utoipa::path(get, path = "/v1/settings", tag = "Settings",
    responses((status = 200, body = SettingsResponse)))]
pub async fn handle_get(
    state: Arc<AppState>,
    _req: Request<Incoming>,
    auth_ctx: Option<AuthContext>,
) -> Response<BoxBody> {
    let bucket = match resolve_bucket(&state, auth_ctx.as_ref()) {
        Some(b) => b,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "no_storage",
                "Storage not configured",
            )
        }
    };

    let settings = match bucket_settings::load_bucket_settings(&bucket) {
        Ok(s) => s,
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "settings_error",
                &format!("Failed to load settings: {e}"),
            )
        }
    };

    let models = list_models_enriched(&settings).await;

    let payload = json!({
        "model": settings.model,
        "system_prompt": settings.system_prompt,
        "max_output_tokens": settings.max_output_tokens,
        "context_length": settings.context_length,
        "auto_title": settings.auto_title,
        "default_prompt_id": settings.default_prompt_id,
        "system_prompt_enabled": settings.system_prompt_enabled,
        "available_models": models,
    });

    json_response(StatusCode::OK, &payload)
}

/// Request body for PATCH /v1/settings.
#[derive(Serialize, ToSchema)]
#[allow(dead_code)]
pub(crate) struct SettingsPatchRequest {
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub max_output_tokens: Option<u32>,
    pub context_length: Option<u32>,
    pub auto_title: Option<bool>,
    pub default_prompt_id: Option<String>,
    pub system_prompt_enabled: Option<bool>,
    pub model_overrides: Option<OverridesJson>,
}

/// PATCH /v1/settings — merge partial update into bucket settings.
///
/// Accepts top-level `model`, `system_prompt`, and per-model generation
/// overrides under `model_overrides: { temperature, top_p, top_k, … }`.
/// The overrides are stored under `[models."<active-model>"]` in the TOML.
#[utoipa::path(patch, path = "/v1/settings", tag = "Settings",
    request_body = SettingsPatchRequest,
    responses((status = 200, body = SettingsResponse)))]
pub async fn handle_patch(
    state: Arc<AppState>,
    req: Request<Incoming>,
    auth_ctx: Option<AuthContext>,
) -> Response<BoxBody> {
    let bucket = match resolve_bucket(&state, auth_ctx.as_ref()) {
        Some(b) => b,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "no_storage",
                "Storage not configured",
            )
        }
    };

    let body_bytes = match req.into_body().collect().await {
        Ok(body) => body.to_bytes(),
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "invalid_request", "Invalid body"),
    };

    let patch: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(val) => val,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &format!("Invalid JSON: {e}"),
            )
        }
    };

    let mut settings = match bucket_settings::load_bucket_settings(&bucket) {
        Ok(s) => s,
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "settings_error",
                &format!("Failed to load settings: {e}"),
            )
        }
    };

    // Merge top-level fields.
    if let Some(v) = patch.get("model") {
        settings.model = v.as_str().map(|s| s.to_string());
    }
    if let Some(v) = patch.get("system_prompt") {
        settings.system_prompt = v.as_str().map(|s| s.to_string());
    }
    if let Some(v) = patch.get("max_output_tokens") {
        settings.max_output_tokens = v.as_u64().map(|n| n as u32);
    }
    if let Some(v) = patch.get("context_length") {
        settings.context_length = v.as_u64().map(|n| n as u32);
    }
    if let Some(v) = patch.get("auto_title") {
        settings.auto_title = v.as_bool().unwrap_or(true);
    }
    if let Some(v) = patch.get("default_prompt_id") {
        settings.default_prompt_id = v.as_str().map(|s| s.to_string());
    }
    if let Some(v) = patch.get("system_prompt_enabled") {
        settings.system_prompt_enabled = v.as_bool().unwrap_or(true);
    }

    // Merge per-model generation overrides (keyed to the active model).
    if let Some(overrides) = patch.get("model_overrides") {
        if let Some(model_id) = &settings.model {
            let entry = settings.models.entry(model_id.clone()).or_default();
            if let Some(v) = overrides.get("temperature") {
                entry.temperature = v.as_f64();
            }
            if let Some(v) = overrides.get("top_p") {
                entry.top_p = v.as_f64();
            }
            if let Some(v) = overrides.get("top_k") {
                entry.top_k = v.as_u64().map(|n| n as u32);
            }
            // Remove the entry if all fields cleared.
            if entry.is_empty() {
                settings.models.remove(model_id);
            }
        }
    }

    if let Err(e) = bucket_settings::save_bucket_settings(&bucket, &settings) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "settings_error",
            &format!("Failed to save settings: {e}"),
        );
    }

    // Return full updated settings (same shape as GET).
    let models = list_models_enriched(&settings).await;
    let payload = json!({
        "model": settings.model,
        "system_prompt": settings.system_prompt,
        "max_output_tokens": settings.max_output_tokens,
        "context_length": settings.context_length,
        "auto_title": settings.auto_title,
        "default_prompt_id": settings.default_prompt_id,
        "system_prompt_enabled": settings.system_prompt_enabled,
        "available_models": models,
    });

    json_response(StatusCode::OK, &payload)
}

/// DELETE /v1/settings/models/{model_id} — reset a model's overrides to defaults.
#[utoipa::path(delete, path = "/v1/settings/models/{model_id}", tag = "Settings",
    params(("model_id" = String, Path, description = "Model ID")),
    responses((status = 200, body = SettingsResponse)))]
pub async fn handle_reset_model(
    state: Arc<AppState>,
    _req: Request<Incoming>,
    auth_ctx: Option<AuthContext>,
    model_id: &str,
) -> Response<BoxBody> {
    let bucket = match resolve_bucket(&state, auth_ctx.as_ref()) {
        Some(b) => b,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                "no_storage",
                "Storage not configured",
            )
        }
    };

    let mut settings = match bucket_settings::load_bucket_settings(&bucket) {
        Ok(s) => s,
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "settings_error",
                &format!("Failed to load settings: {e}"),
            )
        }
    };

    // Remove all overrides for this model.
    settings.models.remove(model_id);

    if let Err(e) = bucket_settings::save_bucket_settings(&bucket, &settings) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "settings_error",
            &format!("Failed to save settings: {e}"),
        );
    }

    // Return full settings (same shape as GET).
    let models = list_models_enriched(&settings).await;
    let payload = json!({
        "model": settings.model,
        "system_prompt": settings.system_prompt,
        "max_output_tokens": settings.max_output_tokens,
        "context_length": settings.context_length,
        "auto_title": settings.auto_title,
        "default_prompt_id": settings.default_prompt_id,
        "system_prompt_enabled": settings.system_prompt_enabled,
        "available_models": models,
    });

    json_response(StatusCode::OK, &payload)
}

/// Resolve the effective bucket path (with tenant prefix if applicable).
fn resolve_bucket(state: &AppState, auth_ctx: Option<&AuthContext>) -> Option<std::path::PathBuf> {
    state.bucket_path.as_ref().map(|base| match auth_ctx {
        Some(ctx) => base.join(&ctx.storage_prefix),
        None => base.to_path_buf(),
    })
}

/// Enriched model info returned in the settings response.
///
/// Each model carries its `generation_config.json` defaults so the UI can
/// show placeholders, plus any user overrides from the bucket settings.
#[derive(Serialize, ToSchema)]
pub(crate) struct ModelEntry {
    id: String,
    /// Model origin: "managed" (managed by Talu) or "hub" (HuggingFace cache).
    source: &'static str,
    /// Defaults from the model's generation_config.json.
    defaults: ModelDefaults,
    /// User overrides from bucket settings (empty = use defaults).
    overrides: OverridesJson,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct ModelDefaults {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    do_sample: bool,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct OverridesJson {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
}

impl From<&ModelOverrides> for OverridesJson {
    fn from(o: &ModelOverrides) -> Self {
        Self {
            temperature: o.temperature,
            top_p: o.top_p,
            top_k: o.top_k,
        }
    }
}

/// List managed models enriched with generation_config defaults + user overrides.
///
/// Only returns models with `source = Managed` (Talu-managed models in
/// `~/.cache/talu/models/`). Hub-cached models are excluded from the UI
/// model selector — they may be incomplete downloads or not usable.
async fn list_models_enriched(settings: &bucket_settings::BucketSettings) -> Vec<ModelEntry> {
    let overrides = settings.models.clone();
    tokio::task::spawn_blocking(move || {
        let cached = talu::repo::repo_list_models(false).unwrap_or_default();
        cached
            .into_iter()
            .filter(|m| m.source == talu::repo::CacheOrigin::Managed)
            .map(|m| {
                let defaults = talu::model::get_generation_config(&m.path)
                    .map(|gc| ModelDefaults {
                        temperature: gc.temperature,
                        top_k: gc.top_k,
                        top_p: gc.top_p,
                        do_sample: gc.do_sample,
                    })
                    .unwrap_or(ModelDefaults {
                        temperature: 1.0,
                        top_k: 50,
                        top_p: 1.0,
                        do_sample: true,
                    });
                let user_overrides =
                    overrides
                        .get(&m.id)
                        .map(OverridesJson::from)
                        .unwrap_or(OverridesJson {
                            temperature: None,
                            top_p: None,
                            top_k: None,
                        });
                ModelEntry {
                    id: m.id,
                    source: "managed",
                    defaults,
                    overrides: user_overrides,
                }
            })
            .collect()
    })
    .await
    .unwrap_or_default()
}

fn json_response(status: StatusCode, payload: &serde_json::Value) -> Response<BoxBody> {
    let body = serde_json::to_vec(payload).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)).boxed())
        .unwrap()
}

fn json_error(status: StatusCode, code: &str, message: &str) -> Response<BoxBody> {
    let payload = json!({
        "error": {
            "code": code,
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
