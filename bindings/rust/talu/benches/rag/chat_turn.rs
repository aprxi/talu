//! Chat turn latency: append message -> embed -> search.
//!
//! Measures the full end-to-end loop: append a message, embed (simulated via
//! vector append), then search against the persistent "medium" dataset's
//! vector store (10k vectors, 384d).
//!
//! The chat append goes to a fresh temp session (write bench), while the
//! vector search reads from the persistent dataset (read bench).

use criterion::Criterion;
use uuid::Uuid;

use talu::vector::VectorStore;
use talu::ChatHandle;

#[allow(dead_code)]
#[path = "../common/mod.rs"]
mod common;

pub fn bench_chat_turn(c: &mut Criterion) {
    let dataset_path = common::dataset::dataset_path("medium");

    // Open the persistent vector store for search (read-only in the hot loop).
    let store = VectorStore::open(&dataset_path).unwrap();

    // Create a fresh temp session for the write side of the loop.
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let sid = Uuid::new_v4().to_string();
    let chat = ChatHandle::new(None).unwrap();
    chat.set_storage_db(db_path, &sid).unwrap();

    let query_vec = common::make_query(common::dataset::DIMS);
    let mut turn_counter = 0u64;

    c.bench_function("rag_chat_turn_latency", |b| {
        b.iter(|| {
            // 1. Append user message (write to temp session).
            let msg = format!("User turn {turn_counter}");
            common::append_message(&chat, msg.as_bytes());

            // 2. Search persistent vector store (read).
            let result = store.search(&query_vec, 10).unwrap();
            assert_eq!(result.ids.len(), 10);

            turn_counter += 1;
        });
    });
}
