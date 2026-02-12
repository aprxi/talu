//! Shared helpers for Criterion benchmark harnesses.
//!
//! Provides corpus builders, FFI helpers, and setup utilities.
//! Domain-specific constants (CORPUS_SIZE, TOP_K, etc.) belong in each harness,
//! not here — these functions accept parameters.

pub mod dataset;

use tempfile::TempDir;
use uuid::Uuid;

use talu::responses::{MessageRole, ResponsesView};
use talu::vector::VectorStore;
use talu::ChatHandle;

/// Append a single user message via raw FFI.
pub fn append_message(chat: &ChatHandle, content: &[u8]) {
    let rc = unsafe {
        talu_sys::talu_responses_append_message(
            chat.responses().as_ptr(),
            MessageRole::User as u8,
            content.as_ptr(),
            content.len(),
        )
    };
    assert!(rc >= 0, "append_message failed: {rc}");
}

/// Open a fresh VectorStore in a new temp directory.
pub fn fresh_vector_store() -> (TempDir, VectorStore) {
    let dir = TempDir::new().expect("tmpdir");
    let store = VectorStore::open(dir.path().to_str().unwrap()).expect("open");
    (dir, store)
}

/// Build a deterministic vector corpus: id=i, all components = (i as f32) * 0.001.
pub fn make_corpus(count: usize, dims: u32) -> (Vec<u64>, Vec<f32>) {
    let ids: Vec<u64> = (0..count as u64).collect();
    let vectors: Vec<f32> = (0..count)
        .flat_map(|i| {
            let val = i as f32 * 0.001;
            std::iter::repeat(val).take(dims as usize)
        })
        .collect();
    (ids, vectors)
}

/// Build a single query vector (all components = 0.5).
pub fn make_query(dims: u32) -> Vec<f32> {
    vec![0.5_f32; dims as usize]
}

/// Build a batch of `n` query vectors.
pub fn make_query_batch(n: usize, dims: u32) -> Vec<f32> {
    (0..n)
        .flat_map(|i| {
            let val = i as f32 * 0.01;
            std::iter::repeat(val).take(dims as usize)
        })
        .collect()
}

/// Create a ChatHandle with storage in a temp directory.
pub fn fresh_chat() -> (TempDir, ChatHandle, String) {
    let dir = TempDir::new().expect("tmpdir");
    let sid = Uuid::new_v4().to_string();
    let chat = ChatHandle::new(None).expect("new");
    chat.set_storage_db(dir.path().to_str().unwrap(), &sid)
        .expect("set_storage_db");
    (dir, chat, sid)
}
