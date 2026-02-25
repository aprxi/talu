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

    /// On-disk layout version. `0` = legacy (KV namespaces at root, docs/chat
    /// at root). `1` = consolidated (KV under `kv/`, tables under `tables/`).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub layout_version: u32,
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
            layout_version: 0,
        }
    }
}

fn default_true() -> bool {
    true
}

fn is_zero(v: &u32) -> bool {
    *v == 0
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

/// Known KV namespace directory names that exist at the bucket root in layout v0.
const KNOWN_KV_NAMESPACES: &[&str] = &["ui", "repo_meta"];

/// Migrate on-disk layout from v0 to v1 if needed.
///
/// v0 layout: KV namespaces (`ui/`, `plugin:*/`, `repo_meta/`) and table data
/// (`docs/`, `chat/`) sit at the bucket root.
///
/// v1 layout: KV namespaces move under `kv/`, legacy tables move under `tables/`.
///
/// The migration is idempotent — each move checks if the source exists before
/// attempting the rename. A partially completed migration will resume cleanly.
pub fn migrate_layout_if_needed(bucket: &Path) -> Result<()> {
    let settings = load_bucket_settings(bucket)?;
    if settings.layout_version >= 1 {
        return Ok(());
    }

    log::info!(target: "server::init", "migrating on-disk layout v0 → v1");

    // 1. Move docs/ → tables/documents/docs/
    let docs_src = bucket.join("docs");
    if docs_src.is_dir() {
        let docs_dst = bucket.join("tables").join("documents");
        std::fs::create_dir_all(&docs_dst)
            .with_context(|| format!("create {}", docs_dst.display()))?;
        let docs_final = docs_dst.join("docs");
        if !docs_final.exists() {
            std::fs::rename(&docs_src, &docs_final)
                .with_context(|| format!("move {} → {}", docs_src.display(), docs_final.display()))?;
        }
    }

    // 2. Move chat/ → tables/chat/chat/
    let chat_src = bucket.join("chat");
    if chat_src.is_dir() {
        let chat_dst = bucket.join("tables").join("chat");
        std::fs::create_dir_all(&chat_dst)
            .with_context(|| format!("create {}", chat_dst.display()))?;
        let chat_final = chat_dst.join("chat");
        if !chat_final.exists() {
            std::fs::rename(&chat_src, &chat_final)
                .with_context(|| format!("move {} → {}", chat_src.display(), chat_final.display()))?;
        }
    }

    // 3. Move KV namespaces to kv/
    let kv_dir = bucket.join("kv");

    // Move known namespaces.
    for ns in KNOWN_KV_NAMESPACES {
        let src = bucket.join(ns);
        if src.is_dir() {
            std::fs::create_dir_all(&kv_dir)
                .with_context(|| format!("create {}", kv_dir.display()))?;
            let dst = kv_dir.join(ns);
            if !dst.exists() {
                std::fs::rename(&src, &dst)
                    .with_context(|| format!("move {} → {}", src.display(), dst.display()))?;
            }
        }
    }

    // Move plugin:* namespaces.
    if let Ok(entries) = std::fs::read_dir(bucket) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("plugin:") && entry.path().is_dir() {
                std::fs::create_dir_all(&kv_dir)
                    .with_context(|| format!("create {}", kv_dir.display()))?;
                let dst = kv_dir.join(&*name_str);
                if !dst.exists() {
                    std::fs::rename(&entry.path(), &dst).with_context(|| {
                        format!("move {} → {}", entry.path().display(), dst.display())
                    })?;
                }
            }
        }
    }

    // 4. Persist layout_version = 1.
    let mut updated = load_bucket_settings(bucket)?;
    updated.layout_version = 1;
    save_bucket_settings(bucket, &updated)?;

    log::info!(target: "server::init", "layout migration complete (v1)");
    Ok(())
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

    #[test]
    fn test_layout_version_defaults_to_zero() {
        let settings: BucketSettings = toml::from_str("").unwrap();
        assert_eq!(settings.layout_version, 0);
    }

    #[test]
    fn test_layout_version_round_trip() {
        let mut settings = BucketSettings::default();
        settings.layout_version = 1;
        let toml_str = toml::to_string_pretty(&settings).unwrap();
        assert!(toml_str.contains("layout_version = 1"));
        let loaded: BucketSettings = toml::from_str(&toml_str).unwrap();
        assert_eq!(loaded.layout_version, 1);
    }

    #[test]
    fn test_migrate_layout_v0_to_v1() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();

        // Create old-layout directories.
        std::fs::create_dir_all(root.join("docs")).unwrap();
        std::fs::write(root.join("docs").join("block-0000.col"), b"d").unwrap();
        std::fs::create_dir_all(root.join("chat")).unwrap();
        std::fs::write(root.join("chat").join("block-0000.col"), b"c").unwrap();
        std::fs::create_dir_all(root.join("ui")).unwrap();
        std::fs::write(root.join("ui").join("kv.db"), b"u").unwrap();
        std::fs::create_dir_all(root.join("plugin:talu.chat")).unwrap();
        std::fs::write(root.join("plugin:talu.chat").join("kv.db"), b"p").unwrap();
        std::fs::create_dir_all(root.join("repo_meta")).unwrap();
        std::fs::write(root.join("repo_meta").join("kv.db"), b"r").unwrap();

        // Existing tables/ dir with user table (should be preserved).
        std::fs::create_dir_all(root.join("tables").join("my_table")).unwrap();
        std::fs::write(root.join("tables").join("my_table").join("data"), b"t").unwrap();

        // Run migration.
        migrate_layout_if_needed(root).unwrap();

        // Verify target layout.
        assert!(root.join("tables/documents/docs/block-0000.col").exists());
        assert!(root.join("tables/chat/chat/block-0000.col").exists());
        assert!(root.join("kv/ui/kv.db").exists());
        assert!(root.join("kv/plugin:talu.chat/kv.db").exists());
        assert!(root.join("kv/repo_meta/kv.db").exists());
        assert!(root.join("tables/my_table/data").exists());

        // Old dirs should be gone.
        assert!(!root.join("docs").exists());
        assert!(!root.join("chat").exists());
        assert!(!root.join("ui").exists());
        assert!(!root.join("plugin:talu.chat").exists());
        assert!(!root.join("repo_meta").exists());

        // layout_version should be 1.
        let settings = load_bucket_settings(root).unwrap();
        assert_eq!(settings.layout_version, 1);
    }

    #[test]
    fn test_migrate_layout_idempotent() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();

        std::fs::create_dir_all(root.join("docs")).unwrap();
        std::fs::write(root.join("docs").join("data"), b"d").unwrap();

        migrate_layout_if_needed(root).unwrap();
        assert_eq!(load_bucket_settings(root).unwrap().layout_version, 1);

        // Second run is a no-op.
        migrate_layout_if_needed(root).unwrap();
        assert!(root.join("tables/documents/docs/data").exists());
    }

    #[test]
    fn test_migrate_layout_fresh_bucket() {
        // No old directories — migration should succeed and set version.
        let tmp = tempfile::TempDir::new().unwrap();
        migrate_layout_if_needed(tmp.path()).unwrap();
        assert_eq!(load_bucket_settings(tmp.path()).unwrap().layout_version, 1);
    }
}
