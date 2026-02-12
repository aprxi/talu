//! Persistent benchmark dataset resolver.
//!
//! Locates pre-generated datasets at `~/.talu/db/bench-{name}/`.
//! Override with `TALU_BENCH_DB=<path>` for CI or custom locations.

/// Vector dimensions used by all benchmark datasets.
pub const DIMS: u32 = 384;

/// Returns the path to a persistent bench dataset.
///
/// Resolution order:
/// 1. `TALU_BENCH_DB` env var (if set, used as-is)
/// 2. `~/.talu/db/bench-{name}/`
///
/// # Panics
///
/// Panics with a helpful message if the dataset doesn't exist.
pub fn dataset_path(name: &str) -> String {
    if let Ok(override_path) = std::env::var("TALU_BENCH_DB") {
        if std::path::Path::new(&override_path).exists() {
            return override_path;
        }
        panic!(
            "TALU_BENCH_DB={override_path} does not exist.\n\
             Unset TALU_BENCH_DB or point it to a valid dataset."
        );
    }

    let home = std::env::var("HOME").expect("HOME not set");
    let path = format!("{home}/.talu/db/bench-{name}");
    if !std::path::Path::new(&path).exists() {
        panic!(
            "Dataset 'bench-{name}' not found at {path}.\n\
             Run: make bench-prep dataset={name}"
        );
    }
    path
}
