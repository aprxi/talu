//! Session listing + client-side metadata scan benchmark.
//!
//! Uses the persistent "medium" dataset (200 sessions × 50 messages).
//! Measures the RAG pattern of listing sessions and filtering client-side.

use criterion::Criterion;
use talu::StorageHandle;

#[allow(dead_code)]
#[path = "../common/mod.rs"]
mod common;

pub fn bench_session_list_scan(c: &mut Criterion) {
    let db_path = common::dataset::dataset_path("medium");

    let mut group = c.benchmark_group("rag_session_scan");

    group.bench_function("list_200_sessions", |b| {
        b.iter(|| {
            let storage = StorageHandle::open(&db_path).unwrap();
            let sessions = storage.list_sessions(Some(200)).unwrap();
            assert_eq!(sessions.len(), 200);
        });
    });

    // List + filter: find sessions with title starting with "Session 19" (client-side).
    group.bench_function("list_and_filter_200", |b| {
        b.iter(|| {
            let storage = StorageHandle::open(&db_path).unwrap();
            let sessions = storage.list_sessions(Some(200)).unwrap();
            let filtered: Vec<_> = sessions
                .iter()
                .filter(|s| {
                    s.title
                        .as_ref()
                        .map(|t| t.starts_with("Session 19"))
                        .unwrap_or(false)
                })
                .collect();
            // "Session 19", "Session 190"..."Session 199" = 11 matches
            assert_eq!(filtered.len(), 11);
        });
    });

    group.finish();
}
