//! Configuration file support (`~/.talu/config.toml`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Top-level configuration file structure.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TaluConfig {
    /// Named profiles. Each profile maps to a storage bucket.
    #[serde(default)]
    pub profiles: BTreeMap<String, ProfileConfig>,

    /// Default model for inference commands (e.g., "Qwen/Qwen3-0.6B-GAF4").
    #[serde(default)]
    pub default_model: Option<String>,
}

/// Per-profile configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileConfig {
    /// Path to the storage bucket directory.
    pub bucket: PathBuf,
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
    let contents = std::fs::read_to_string(&path)
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
    std::fs::write(&path, contents)
        .with_context(|| format!("Failed to write {}", path.display()))?;
    Ok(())
}

fn config_path_with_home(home: &Path) -> PathBuf {
    home.join("config.toml")
}

fn default_profile_bucket_with_home(home: &Path, name: &str) -> PathBuf {
    home.join("db").join(name)
}

/// Resolve a profile name to its bucket path.
///
/// - Known profile → bucket path from config
/// - Existing on-disk bucket (`~/.talu/db/<name>/`) → accepted and registered
/// - Missing profile name → auto-creates profile entry at `~/.talu/db/<name>/`
pub fn resolve_profile(profile: &str) -> Result<PathBuf> {
    resolve_profile_with_home(profile, &talu_home())
}

fn resolve_profile_with_home(profile: &str, home: &Path) -> Result<PathBuf> {
    let config_path = config_path_with_home(home);
    let mut config = load_config_from(&config_path)?;

    if let Some(entry) = config.profiles.get(profile) {
        return Ok(entry.bucket.clone());
    }

    // Profile may exist on disk but not be registered in config.toml
    let discovered_bucket = default_profile_bucket_with_home(home, profile);
    if discovered_bucket.is_dir() {
        config.profiles.insert(
            profile.to_string(),
            ProfileConfig {
                bucket: discovered_bucket.clone(),
            },
        );
        save_config_to(&config_path, &config)?;
        return Ok(discovered_bucket);
    }

    // Allow arbitrary profile names by creating a new profile entry on first use.
    let bucket = default_profile_bucket_with_home(home, profile);
    config.profiles.insert(
        profile.to_string(),
        ProfileConfig {
            bucket: bucket.clone(),
        },
    );
    save_config_to(&config_path, &config)?;
    Ok(bucket)
}

/// Resolve a profile and ensure the bucket directory exists (auto-create).
pub fn resolve_and_ensure_bucket(profile: &str) -> Result<PathBuf> {
    let bucket = resolve_profile(profile)?;
    ensure_bucket(&bucket)?;
    Ok(bucket)
}

/// Ensure a bucket directory exists, initializing it if needed.
pub fn ensure_bucket(bucket_path: &Path) -> Result<()> {
    if bucket_path.exists() {
        return Ok(());
    }

    std::fs::create_dir_all(bucket_path)
        .with_context(|| format!("Failed to create bucket at {}", bucket_path.display()))?;

    // Generate store.key
    let key_path = bucket_path.join("store.key");
    let mut key_data = [0u8; 16];
    getrandom::fill(&mut key_data)
        .map_err(|e| anyhow::anyhow!("Failed to generate random key: {}", e))?;
    std::fs::write(&key_path, &key_data)?;

    // Initialize manifest.json
    let manifest_path = bucket_path.join("manifest.json");
    let manifest = r#"{"version": 1, "segments": [], "last_compaction_ts": 0}"#;
    std::fs::write(&manifest_path, manifest)?;

    log::info!("Initialized storage bucket at: {}", bucket_path.display());
    Ok(())
}

/// Resolve bucket path from the 3-tier flag set: --no-bucket, --bucket, --profile.
///
/// Returns `None` when storage is disabled, `Some(path)` otherwise.
/// Errors if both `--bucket` and `--no-bucket` are specified.
pub fn resolve_bucket(
    no_bucket: bool,
    bucket: Option<PathBuf>,
    profile: &str,
) -> Result<Option<PathBuf>> {
    if no_bucket && bucket.is_some() {
        anyhow::bail!("--bucket and --no-bucket are mutually exclusive");
    }
    if no_bucket {
        return Ok(None);
    }
    if let Some(explicit) = bucket {
        return Ok(Some(explicit));
    }
    Ok(Some(resolve_and_ensure_bucket(profile)?))
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
    fn resolves_existing_disk_profile_and_registers_it() {
        let temp = tempdir().expect("tempdir");
        let home = temp.path().join(".talu");
        let dev_bucket = home.join("db").join("dev");
        std::fs::create_dir_all(&dev_bucket).expect("create dev bucket");

        let resolved = resolve_profile_with_home("dev", &home).expect("resolve dev");
        assert_eq!(resolved, dev_bucket);

        let config = load_config_from(&config_path_with_home(&home)).expect("load config");
        assert_eq!(
            config.profiles.get("dev").map(|p| p.bucket.as_path()),
            Some(dev_bucket.as_path())
        );
    }

    #[test]
    fn missing_profile_auto_registers_default_bucket_path() {
        let temp = tempdir().expect("tempdir");
        let home = temp.path().join(".talu");
        std::fs::create_dir_all(&home).expect("create home");

        let resolved =
            resolve_profile_with_home("test_minimal", &home).expect("resolve new profile");
        let expected = home.join("db").join("test_minimal");
        assert_eq!(resolved, expected);

        let config = load_config_from(&config_path_with_home(&home)).expect("load config");
        assert_eq!(
            config
                .profiles
                .get("test_minimal")
                .map(|p| p.bucket.as_path()),
            Some(expected.as_path())
        );
    }
}
