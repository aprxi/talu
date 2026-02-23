#[cfg(test)]
use std::collections::HashSet;
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};

use talu::{RepoMetaError, RepoMetaStore, RepoPinEntry};

const BUSY_RETRY_WINDOW: Duration = Duration::from_secs(5);
const BUSY_RETRY_BASE_SLEEP: Duration = Duration::from_millis(25);
const BUSY_RETRY_MAX_SLEEP: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PinnedModelEntry {
    pub model_uri: String,
    pub pinned_at_ms: i64,
    pub size_bytes: Option<u64>,
    pub size_updated_at_ms: Option<i64>,
}

/// Profile-local pin storage backed by TaluDB (`<bucket>/repo_meta/*`).
pub(crate) struct PinStore {
    store: RepoMetaStore,
}

impl PinStore {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let db_root = resolve_db_root(path);
        std::fs::create_dir_all(db_root)
            .with_context(|| format!("Failed to create {}", db_root.display()))?;

        let db_root_str = db_root.to_string_lossy();
        let store = RepoMetaStore::open(db_root_str.as_ref())
            .map_err(|err| anyhow!("Failed to open repo metadata store: {err}"))?;

        Ok(Self { store })
    }

    pub(crate) fn pin(&self, model_uri: &str) -> Result<bool> {
        let was_pinned = self.is_pinned(model_uri)?;
        self.with_busy_retry("pin", || self.store.pin(model_uri))?;
        Ok(!was_pinned)
    }

    pub(crate) fn unpin(&self, model_uri: &str) -> Result<bool> {
        let was_pinned = self.is_pinned(model_uri)?;
        self.with_busy_retry("unpin", || self.store.unpin(model_uri))?;
        Ok(was_pinned)
    }

    pub(crate) fn upsert_size_bytes(&self, model_uri: &str, size_bytes: u64) -> Result<()> {
        self.with_busy_retry("update_size", || {
            self.store.update_size(model_uri, size_bytes)
        })
    }

    pub(crate) fn clear_size_bytes(&self, model_uri: &str) -> Result<()> {
        self.with_busy_retry("clear_size", || self.store.clear_size(model_uri))
    }

    pub(crate) fn list_pinned_entries(&self) -> Result<Vec<PinnedModelEntry>> {
        let entries = self
            .with_busy_retry("list_pins", || self.store.list_pins())?
            .into_iter()
            .map(convert_pin_entry)
            .collect();
        Ok(entries)
    }

    pub(crate) fn list_pinned(&self) -> Result<Vec<String>> {
        Ok(self
            .list_pinned_entries()?
            .into_iter()
            .map(|entry| entry.model_uri)
            .collect())
    }

    #[cfg(test)]
    pub(crate) fn list_pinned_set(&self) -> Result<HashSet<String>> {
        Ok(self.list_pinned()?.into_iter().collect())
    }

    fn is_pinned(&self, model_uri: &str) -> Result<bool> {
        let entries = self.with_busy_retry("list_pins", || self.store.list_pins())?;
        Ok(entries.iter().any(|entry| entry.model_uri == model_uri))
    }

    fn with_busy_retry<T, F>(&self, op_name: &str, mut op: F) -> Result<T>
    where
        F: FnMut() -> std::result::Result<T, RepoMetaError>,
    {
        let start = Instant::now();
        let mut attempt: u32 = 0;

        loop {
            match op() {
                Ok(value) => return Ok(value),
                Err(err) if err.is_busy() && start.elapsed() < BUSY_RETRY_WINDOW => {
                    let exp = 1u64 << attempt.min(4);
                    let sleep_ms = (BUSY_RETRY_BASE_SLEEP.as_millis() as u64)
                        .saturating_mul(exp)
                        .min(BUSY_RETRY_MAX_SLEEP.as_millis() as u64);
                    thread::sleep(Duration::from_millis(sleep_ms));
                    attempt = attempt.saturating_add(1);
                }
                Err(err) => {
                    return Err(anyhow!("repo meta {op_name} failed: {err}"));
                }
            }
        }
    }
}

fn resolve_db_root(path: &Path) -> &Path {
    if path.file_name().and_then(|name| name.to_str()) == Some("meta.sqlite") {
        return path.parent().unwrap_or(path);
    }
    path
}

fn convert_pin_entry(entry: RepoPinEntry) -> PinnedModelEntry {
    PinnedModelEntry {
        model_uri: entry.model_uri,
        pinned_at_ms: entry.pinned_at_ms,
        size_bytes: entry.size_bytes,
        size_updated_at_ms: entry.size_updated_at_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pin_unpin_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = dir.path().join("meta.sqlite");
        let store = PinStore::open(&db).expect("open");

        assert!(store.pin("Qwen/Qwen3-0.6B").expect("pin"));
        assert!(!store.pin("Qwen/Qwen3-0.6B").expect("idempotent pin"));

        let set = store.list_pinned_set().expect("list set");
        assert!(set.contains("Qwen/Qwen3-0.6B"));

        assert!(store.unpin("Qwen/Qwen3-0.6B").expect("unpin"));
        assert!(!store.unpin("Qwen/Qwen3-0.6B").expect("idempotent unpin"));

        let set = store.list_pinned_set().expect("list set after");
        assert!(!set.contains("Qwen/Qwen3-0.6B"));
    }

    #[test]
    fn pin_size_roundtrip_and_cascade() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = dir.path().join("meta.sqlite");
        let store = PinStore::open(&db).expect("open");

        assert!(store.pin("Qwen/Qwen3-0.6B").expect("pin"));
        store
            .upsert_size_bytes("Qwen/Qwen3-0.6B", 1_234_567_890)
            .expect("upsert size");

        let entries = store.list_pinned_entries().expect("list entries");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].model_uri, "Qwen/Qwen3-0.6B");
        assert_eq!(entries[0].size_bytes, Some(1_234_567_890));
        assert!(entries[0].size_updated_at_ms.is_some());

        assert!(store.unpin("Qwen/Qwen3-0.6B").expect("unpin"));

        assert!(store.pin("Qwen/Qwen3-0.6B").expect("re-pin"));
        let entries = store.list_pinned_entries().expect("list entries");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].size_bytes, None);
        assert_eq!(entries[0].size_updated_at_ms, None);
    }
}
