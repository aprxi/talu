//! "Hot Document" benchmark: measures getDocument and list latency as a
//! function of update count.
//!
//! Reverse scanning makes the document-lookup phase O(1), but setup costs
//! (block index scan, tombstone collection) still scale with total blocks.
//! This benchmark quantifies both the improvement from reverse scanning and
//! the remaining bottleneck that compaction would address.

use criterion::{BenchmarkId, Criterion};
use tempfile::TempDir;

use talu::documents::DocumentsHandle;

/// Pre-populate a store with a single document updated `n` times.
fn setup_hot_document(n: usize) -> (TempDir, DocumentsHandle) {
    let dir = TempDir::new().expect("tmpdir");
    let handle = DocumentsHandle::open(dir.path()).expect("open");

    handle
        .create(
            "hot-doc",
            "prompt",
            "Initial Title",
            r#"{"version": 0}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("create");

    for i in 1..=n {
        let json = format!(r#"{{"version": {i}}}"#);
        let title = format!("Title v{i}");
        handle
            .update("hot-doc", Some(&title), Some(&json), None, None)
            .expect("update");
    }

    (dir, handle)
}

pub fn bench_hot_document(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_document");
    group.sample_size(20);

    // Benchmark get() after increasing update counts.
    for &update_count in &[0, 10, 100, 500, 1000, 5000] {
        let (_dir, handle) = setup_hot_document(update_count);

        group.bench_with_input(
            BenchmarkId::new("get", update_count),
            &update_count,
            |b, _| {
                b.iter(|| {
                    let doc = handle.get("hot-doc").expect("get").expect("found");
                    assert!(doc.title.contains('v') || doc.title == "Initial Title");
                });
            },
        );
    }

    // Benchmark list() after increasing update counts on a single document.
    // This shows the scan-all-blocks cost that compaction would address.
    for &update_count in &[0, 100, 1000, 5000] {
        let (_dir, handle) = setup_hot_document(update_count);

        group.bench_with_input(
            BenchmarkId::new("list", update_count),
            &update_count,
            |b, _| {
                b.iter(|| {
                    let docs = handle.list(None, None, None, None, 100).expect("list");
                    assert_eq!(docs.len(), 1);
                });
            },
        );
    }

    group.finish();
}
