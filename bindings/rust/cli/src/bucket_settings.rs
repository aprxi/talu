//! Per-bucket settings (`<bucket>/settings.toml`).
//!
//! Each profile bucket stores its own runtime settings, separate from the
//! global config registry (`~/.talu/config.toml`). This keeps the global
//! file minimal (profile names + bucket paths) and avoids multi-writer
//! conflicts when multiple clients share a bucket.
//!
//! Generation parameters (temperature, top_p, …) are stored per-model under
//! `[models."<model-id>"]`. The top-level `model` field selects the active
//! model; per-model overrides take effect when that model is active.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Per-model sampling parameter overrides (temperature, top_p, top_k).
///
/// These are the model-specific parameters that come from generation_config.json
/// and can be overridden per model. Generic generation parameters (max_tokens,
/// context_length) live at the top level of BucketSettings because they apply
/// uniformly regardless of which model is active.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ModelOverrides {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
}

impl ModelOverrides {
    /// True when every field is None (nothing to persist).
    pub fn is_empty(&self) -> bool {
        self.temperature.is_none() && self.top_p.is_none() && self.top_k.is_none()
    }
}

/// Runtime settings stored in `<bucket>/settings.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketSettings {
    /// Active model for this profile (e.g., "Qwen/Qwen3-0.6B-GAF4").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Default system prompt prepended to every conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,

    /// Maximum number of tokens to generate per response.
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "max_tokens")]
    pub max_output_tokens: Option<u32>,

    /// Context window length (prompt + completion tokens).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,

    /// Automatically generate a descriptive conversation title after the
    /// first response completes (uses a short model inference).
    #[serde(default = "default_true")]
    pub auto_title: bool,

    /// Default prompt document ID. When set, new conversations use this
    /// prompt's system message automatically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_prompt_id: Option<String>,

    /// Whether system prompts are applied to new conversations.
    /// When false, the default prompt is ignored.
    #[serde(default = "default_true")]
    pub system_prompt_enabled: bool,

    /// Per-model sampling parameter overrides, keyed by model ID.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub models: BTreeMap<String, ModelOverrides>,
}

impl Default for BucketSettings {
    fn default() -> Self {
        Self {
            model: None,
            system_prompt: None,
            max_output_tokens: None,
            context_length: None,
            auto_title: true,
            default_prompt_id: None,
            system_prompt_enabled: true,
            models: BTreeMap::new(),
        }
    }
}

fn default_true() -> bool {
    true
}

/// Load settings from `<bucket>/settings.toml`. Returns defaults if the file
/// doesn't exist.
pub fn load_bucket_settings(bucket: &Path) -> Result<BucketSettings> {
    let path = bucket.join("settings.toml");
    if !path.exists() {
        return Ok(BucketSettings::default());
    }
    let contents = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let settings: BucketSettings =
        toml::from_str(&contents).with_context(|| format!("Failed to parse {}", path.display()))?;
    Ok(settings)
}

/// Save settings to `<bucket>/settings.toml`.
pub fn save_bucket_settings(bucket: &Path, settings: &BucketSettings) -> Result<()> {
    let path = bucket.join("settings.toml");
    let contents = toml::to_string_pretty(settings).context("Failed to serialize settings")?;
    std::fs::write(&path, contents)
        .with_context(|| format!("Failed to write {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_tokens_alias() {
        // Old format with max_tokens should deserialize into max_output_tokens
        let toml_str = r#"
            model = "test-model"
            max_tokens = 5000
        "#;
        let settings: BucketSettings = toml::from_str(toml_str).unwrap();
        assert_eq!(settings.max_output_tokens, Some(5000));
    }

    #[test]
    fn test_max_output_tokens_direct() {
        // New format with max_output_tokens
        let toml_str = r#"
            model = "test-model"
            max_output_tokens = 2048
        "#;
        let settings: BucketSettings = toml::from_str(toml_str).unwrap();
        assert_eq!(settings.max_output_tokens, Some(2048));
    }

    #[test]
    fn test_serializes_as_max_output_tokens() {
        // When we save, it should use max_output_tokens (not the alias)
        let settings = BucketSettings {
            model: Some("test".to_string()),
            max_output_tokens: Some(4096),
            ..Default::default()
        };
        let toml_str = toml::to_string_pretty(&settings).unwrap();
        assert!(toml_str.contains("max_output_tokens = 4096"));
        assert!(!toml_str.contains("max_tokens"));
    }

    #[test]
    fn test_load_real_settings_file() {
        // Test loading the actual user's settings file with max_tokens
        let toml_str = r#"
model = "Qwen/Qwen3-0.6B-GAF4"
system_prompt = "You are a helpful AI assistant."
max_tokens = 5000
context_length = 1

[models."Qwen/Qwen3-0.6B-GAF4"]
temperature = 1.0
top_p = 0.0
top_k = 50
        "#;
        let settings: BucketSettings = toml::from_str(toml_str).unwrap();
        assert_eq!(settings.max_output_tokens, Some(5000));
        assert_eq!(settings.model, Some("Qwen/Qwen3-0.6B-GAF4".to_string()));
    }
}
