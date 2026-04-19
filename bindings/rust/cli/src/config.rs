//! Configuration file support (`~/.talu/config.toml`).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Top-level configuration file structure.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TaluConfig {
    /// Default model for inference commands (e.g., "Qwen/Qwen3-0.6B-NVFP4").
    #[serde(default)]
    pub default_model: Option<String>,
}

/// Return `~/.talu/`.
pub fn talu_home() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".talu")
}

/// Return `~/.talu/config.toml`.
pub fn config_path() -> PathBuf {
    config_path_with_home(&talu_home())
}

/// Load config from `~/.talu/config.toml`. Returns default config if file
/// doesn't exist.
pub fn load_config() -> Result<TaluConfig> {
    load_config_from(&config_path())
}

fn load_config_from(path: &Path) -> Result<TaluConfig> {
    if !path.exists() {
        return Ok(TaluConfig::default());
    }
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let config: TaluConfig =
        toml::from_str(&contents).with_context(|| format!("Failed to parse {}", path.display()))?;
    Ok(config)
}

/// Save config to `~/.talu/config.toml`.
pub fn save_config(config: &TaluConfig) -> Result<()> {
    save_config_to(&config_path(), config)
}

fn save_config_to(path: &Path, config: &TaluConfig) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }
    let contents = toml::to_string_pretty(config).context("Failed to serialize config")?;
    std::fs::write(path, contents)
        .with_context(|| format!("Failed to write {}", path.display()))?;
    Ok(())
}

fn config_path_with_home(home: &Path) -> PathBuf {
    home.join("config.toml")
}

/// Get the default model from config, or `None` if not set.
pub fn get_default_model() -> Option<String> {
    load_config().ok()?.default_model
}

/// Set the default model in config.
pub fn set_default_model(model: &str) -> Result<()> {
    let mut config = load_config()?;
    config.default_model = Some(model.to_string());
    save_config(&config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn missing_config_returns_default() {
        let temp = tempdir().expect("tempdir");
        let path = temp.path().join("missing-config.toml");
        let cfg = load_config_from(&path).expect("load");
        assert!(cfg.default_model.is_none());
    }

    #[test]
    fn roundtrip_default_model() {
        let temp = tempdir().expect("tempdir");
        let path = temp.path().join("config.toml");

        let mut cfg = TaluConfig::default();
        cfg.default_model = Some("Qwen/Qwen3-0.6B".to_string());
        save_config_to(&path, &cfg).expect("save");

        let loaded = load_config_from(&path).expect("load");
        assert_eq!(loaded.default_model.as_deref(), Some("Qwen/Qwen3-0.6B"));
    }
}
