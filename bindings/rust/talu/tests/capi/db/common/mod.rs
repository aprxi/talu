//! Shared test fixtures for the DB integration test suite.

use tempfile::TempDir;

/// Ephemeral database context that auto-cleans on drop.
///
/// Creates a temporary directory for TaluDB storage. The directory
/// is deleted when `TestContext` goes out of scope.
pub struct TestContext {
    _dir: TempDir,
    path: String,
}

impl TestContext {
    /// Create a new ephemeral TaluDB root directory.
    pub fn new() -> Self {
        let dir = TempDir::new().expect("failed to create temp dir");
        let path = dir.path().to_string_lossy().into_owned();
        Self { _dir: dir, path }
    }

    /// Returns the DB root path as `&str` (C-API compatible).
    pub fn db_path(&self) -> &str {
        &self.path
    }

    /// Generate a unique session ID.
    pub fn unique_session_id() -> String {
        uuid::Uuid::new_v4().to_string()
    }
}

/// Find all WAL files (`wal-*.wal`) in a namespace directory.
///
/// Returns paths to all per-writer WAL files. With per-writer WALs,
/// each writer creates a unique `wal-<hex>.wal` file. On clean close
/// the file is deleted; orphaned files indicate a crash.
pub fn find_wal_files(db_root: &str, namespace: &str) -> Vec<std::path::PathBuf> {
    let ns_dir = std::path::Path::new(db_root).join(namespace);
    let mut wals = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&ns_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("wal-") && name_str.ends_with(".wal") {
                wals.push(entry.path());
            }
        }
    }
    wals
}

/// Total size of all WAL files in a namespace directory.
pub fn total_wal_size(db_root: &str, namespace: &str) -> u64 {
    find_wal_files(db_root, namespace)
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .sum()
}

/// Generate deterministic test vectors.
///
/// Returns `(ids, flat_vectors)` where `flat_vectors.len() == count * dims`.
/// Each vector is L2-normalized so dot-product similarity works as cosine.
/// Uses a simple deterministic formula (no RNG dependency).
pub fn generate_vectors(count: usize, dims: usize) -> (Vec<u64>, Vec<f32>) {
    let ids: Vec<u64> = (1..=count as u64).collect();
    let mut vectors = Vec::with_capacity(count * dims);

    for i in 0..count {
        let mut v = Vec::with_capacity(dims);
        for d in 0..dims {
            // Deterministic: different per (i, d) but reproducible.
            let val = ((i * 7 + d * 13 + 3) % 100) as f32 / 100.0;
            v.push(val);
        }
        // L2-normalize so dot-product ≈ cosine similarity.
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        vectors.extend_from_slice(&v);
    }

    (ids, vectors)
}
