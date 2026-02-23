//! Session list and text search benchmarks.
//!
//! Uses the persistent "medium" dataset (200 sessions × 50 messages = 10k messages).
//! Content is varied by topic so we can benchmark partial matches vs full scans.
//!
//! Search path: StorageHandle::list_sessions_paginated(query=...)
//!   → FFI talu_db_table_session_list → core scanSessionsFiltered
//!   → collectContentMatchHashes (content scan) + metadata substring match.

use criterion::Criterion;
use talu::StorageHandle;

#[allow(dead_code)]
#[path = "../common/mod.rs"]
mod common;

pub fn bench_list_sessions(c: &mut Criterion) {
    let db_path = common::dataset::dataset_path("medium");

    let mut group = c.benchmark_group("sessions");

    // Baseline: list without search query.
    group.bench_function("list_no_query", |b| {
        b.iter(|| {
            let storage = StorageHandle::open(&db_path).unwrap();
            let result = storage
                .list_sessions_paginated(100, None, None, None)
                .unwrap();
            assert!(!result.sessions.is_empty());
        });
    });

    // Text search: match in title metadata (no content scan needed for these).
    group.bench_function("search_title_hit", |b| {
        b.iter(|| {
            let storage = StorageHandle::open(&db_path).unwrap();
            let result = storage
                .list_sessions_paginated(100, None, None, Some("Session 19"))
                .unwrap();
            // "Session 19", "Session 190"..."Session 199" = 11 matches
            assert_eq!(result.sessions.len(), 11);
        });
    });

    // Text search: match ~40 sessions (1 of 5 topics).
    // Exercises content scan + snippet extraction on a partial match.
    group.bench_function("search_content_partial", |b| {
        b.iter(|| {
            let storage = StorageHandle::open(&db_path).unwrap();
            let result = storage
                .list_sessions_paginated(100, None, None, Some("quantum"))
                .unwrap();
            // 200 sessions / 5 topics = 40 sessions contain "quantum"
            assert_eq!(result.sessions.len(), 40);
            assert!(result.sessions[0].search_snippet.is_some());
        });
    });

    // Text search: match all 200 sessions (content present in every session).
    group.bench_function("search_content_all", |b| {
        b.iter(|| {
            let storage = StorageHandle::open(&db_path).unwrap();
            let result = storage
                .list_sessions_paginated(100, None, None, Some("benchmark payload"))
                .unwrap();
            assert_eq!(result.sessions.len(), 100); // clamped by limit
            assert!(result.sessions[0].search_snippet.is_some());
        });
    });

    // Text search: no matches (worst case — full scan, zero results).
    group.bench_function("search_no_match", |b| {
        b.iter(|| {
            let storage = StorageHandle::open(&db_path).unwrap();
            let result = storage
                .list_sessions_paginated(100, None, None, Some("zzzznotfound"))
                .unwrap();
            assert_eq!(result.sessions.len(), 0);
        });
    });

    group.finish();
}
