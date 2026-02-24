//! Context retrieval benchmark.
//!
//! Load a conversation from the persistent "medium" dataset and read N
//! message texts by index (the real "vector ID -> text" path).
//!
//! Uses session-0000 which has 50 messages in the medium dataset.

use criterion::Criterion;
use talu::responses::ResponsesView;
use talu::StorageHandle;

#[allow(dead_code)]
#[path = "../common/mod.rs"]
mod common;

/// Session ID to load (first session in the medium dataset).
const SESSION_ID: &str = "session-0000";

/// Messages per session in the medium dataset.
const MSGS_PER_SESSION: usize = 50;

pub fn bench_context_retrieval(c: &mut Criterion) {
    let db_path = common::dataset::dataset_path("medium");

    // Pre-select indices spread across the conversation.
    let indices: Vec<usize> = (0..20).map(|i| i % MSGS_PER_SESSION).collect();

    c.bench_function("rag_context_retrieval_20_of_50", |b| {
        b.iter(|| {
            let storage = StorageHandle::open(&db_path).unwrap();
            let conv = storage.load_session(SESSION_ID).unwrap();
            assert_eq!(conv.item_count(), MSGS_PER_SESSION);

            let mut total_len = 0usize;
            for &idx in &indices {
                let text = conv.message_text(idx).unwrap();
                total_len += text.len();
            }
            assert!(total_len > 0);
        });
    });
}
