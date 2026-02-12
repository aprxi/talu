use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Tenant {
    pub id: String,
    pub storage_prefix: String,
    #[serde(default)]
    pub allowed_models: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TenantRegistry {
    tenants: HashMap<String, Tenant>,
}

impl TenantRegistry {
    pub fn load(path: &Path) -> Result<Self> {
        let payload = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read tenant config {}", path.display()))?;
        let tenants: Vec<Tenant> = serde_json::from_str(&payload)
            .with_context(|| format!("Failed to parse tenant config {}", path.display()))?;

        let mut registry = HashMap::new();
        for tenant in tenants {
            let id = tenant.id.clone();
            if registry.insert(id.clone(), tenant).is_some() {
                bail!("Duplicate tenant id in config: {id}", id = id);
            }
        }

        Ok(Self { tenants: registry })
    }

    pub fn get(&self, id: &str) -> Option<&Tenant> {
        self.tenants.get(id)
    }
}

#[cfg(test)]
impl TenantRegistry {
    pub fn from_map(tenants: HashMap<String, Tenant>) -> Self {
        Self { tenants }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_temp(contents: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        path.push(format!("talu-tenant-test-{nanos}.json"));
        std::fs::write(&path, contents).expect("write temp file");
        path
    }

    #[test]
    fn load_valid_config() {
        let path = write_temp(
            r#"[
                {"id": "acme", "storage_prefix": "acme", "allowed_models": ["model-a"]},
                {"id": "globex", "storage_prefix": "globex"}
            ]"#,
        );
        let registry = TenantRegistry::load(&path).expect("load registry");
        let tenant = registry.get("acme").expect("tenant exists");
        assert_eq!(tenant.storage_prefix, "acme");
        assert_eq!(tenant.allowed_models, vec!["model-a".to_string()]);
        let tenant = registry.get("globex").expect("tenant exists");
        assert!(tenant.allowed_models.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_rejects_invalid_json() {
        let path = write_temp(r#"{"bad": true}"#);
        let err = TenantRegistry::load(&path).expect_err("expected error");
        let msg = format!("{err}");
        assert!(msg.contains("tenant config"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_rejects_duplicate_ids() {
        let path = write_temp(
            r#"[
                {"id": "acme", "storage_prefix": "one"},
                {"id": "acme", "storage_prefix": "two"}
            ]"#,
        );
        let err = TenantRegistry::load(&path).expect_err("expected error");
        let msg = format!("{err}");
        assert!(msg.contains("Duplicate tenant id"));
        let _ = std::fs::remove_file(&path);
    }
}
