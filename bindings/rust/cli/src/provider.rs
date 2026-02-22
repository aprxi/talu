use std::env;

use anyhow::{anyhow, Result};

use talu::provider as talu_provider;
use talu::InferenceBackend;

/// Provider info retrieved from core via C API.
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    pub name: String,
    pub default_endpoint: String,
    pub api_key_env: Option<String>,
}

impl From<talu_provider::ProviderInfo> for ProviderInfo {
    fn from(p: talu_provider::ProviderInfo) -> Self {
        Self {
            name: p.name,
            default_endpoint: p.default_endpoint,
            api_key_env: if p.api_key_env.is_empty() {
                None
            } else {
                Some(p.api_key_env)
            },
        }
    }
}

/// Get provider info by name using C API.
pub fn get_provider(name: &str) -> Option<ProviderInfo> {
    talu_provider::provider_get_by_name(name)
        .ok()
        .map(ProviderInfo::from)
}

pub enum ModelTarget<'a> {
    Local(&'a str),
    Remote { provider: String, model: &'a str },
}

/// Parse a model string into a local or remote target.
/// If the input contains `::`, it is treated as remote syntax and must match a registered provider.
pub fn parse_model_target(model: &str) -> Result<ModelTarget<'_>> {
    if let Some((provider_prefix, _)) = model.split_once("::") {
        if provider_prefix.is_empty() {
            return Err(anyhow!("Invalid provider prefix in '{}'", model));
        }
        if let Some(parsed) = talu_provider::parse_model_target(model) {
            return Ok(ModelTarget::Remote {
                provider: parsed.provider.name,
                model: &model[model.len() - parsed.model_id.len()..],
            });
        }

        let providers = list_providers();
        let list = if providers.is_empty() {
            "no providers registered".to_string()
        } else {
            providers.join(", ")
        };
        return Err(anyhow!(
            "Unknown provider '{}'. Available: {}. If this is a local model, remove the '::' prefix.",
            provider_prefix,
            list
        ));
    }

    Ok(ModelTarget::Local(model))
}

/// Check if a string is a known provider name (for ls command).
pub fn is_provider_prefix(s: &str) -> bool {
    if !s.ends_with("::") {
        return false;
    }
    talu_provider::has_provider_prefix(s)
}

/// Get provider name from "vllm::" format.
pub fn provider_from_prefix(s: &str) -> Option<String> {
    if !s.ends_with("::") {
        return None;
    }
    let name = &s[..s.len() - 2];
    get_provider(name).map(|p| p.name)
}

pub fn create_backend_for_model(model: &str) -> Result<InferenceBackend> {
    create_backend_for_model_with_progress(model, None)
}

pub fn create_backend_for_model_with_progress(
    model: &str,
    callback: Option<talu::LoadProgressCallback>,
) -> Result<InferenceBackend> {
    match parse_model_target(model)? {
        ModelTarget::Local(local_id) => {
            Ok(InferenceBackend::new_with_progress(local_id, callback)?)
        }
        ModelTarget::Remote { provider, model } => {
            let provider_info =
                get_provider(&provider).ok_or_else(|| anyhow!("Unknown provider: {}", provider))?;

            let base_url = env::var(format!("{}_ENDPOINT", provider_info.name.to_uppercase()))
                .ok()
                .unwrap_or_else(|| provider_info.default_endpoint.clone());
            let api_key = provider_info
                .api_key_env
                .as_deref()
                .and_then(|key| env::var(key).ok());

            Ok(InferenceBackend::new_openai_compatible(
                model,
                &base_url,
                api_key.as_deref(),
                30_000,
            )?)
        }
    }
}

fn list_providers() -> Vec<String> {
    talu_provider::list_providers()
        .into_iter()
        .map(|p| p.name)
        .collect()
}
