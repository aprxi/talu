//! Vector store basic integration tests.
//!
//! Validates: init -> append -> search -> load -> persistence across reopens.

use crate::capi::db::common::{generate_vectors, TestContext};
use talu::vector::VectorStore;

/// Append vectors and search for an exact match.
#[test]
fn append_and_search_exact() {
    let ctx = TestContext::new();
    let store = VectorStore::open(ctx.db_path()).expect("open failed");

    let dims: u32 = 4;
    let ids: Vec<u64> = vec![10, 20, 30];
    let vectors: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, // id=10: unit vector along dim 0
        0.0, 1.0, 0.0, 0.0, // id=20: unit vector along dim 1
        0.0, 0.0, 1.0, 0.0, // id=30: unit vector along dim 2
    ];

    store.append(&ids, &vectors, dims).expect("append failed");

    // Query for the exact vector of id=10.
    let query = &[1.0, 0.0, 0.0, 0.0];
    let result = store.search(query, 3).expect("search failed");

    assert_eq!(result.ids.len(), 3);
    // Top result must be id=10 (dot product = 1.0).
    assert_eq!(result.ids[0], 10);
    assert!(
        (result.scores[0] - 1.0).abs() < 1e-5,
        "Expected score ≈ 1.0, got {}",
        result.scores[0],
    );
    // Other results should have score 0.0 (orthogonal).
    assert!(
        result.scores[1].abs() < 1e-5,
        "Expected score ≈ 0.0 for orthogonal vector, got {}",
        result.scores[1],
    );
}

/// Search with k < total vectors returns exactly k results.
#[test]
fn search_top_k_limited() {
    let ctx = TestContext::new();
    let store = VectorStore::open(ctx.db_path()).expect("open failed");

    let dims: u32 = 4;
    let (ids, vectors) = generate_vectors(20, dims as usize);
    store.append(&ids, &vectors, dims).expect("append failed");

    let query = &vectors[0..dims as usize]; // Query = first vector
    let result = store.search(query, 5).expect("search failed");

    assert_eq!(result.ids.len(), 5, "Expected exactly k=5 results");
    // First result should be the vector itself (id=1).
    assert_eq!(result.ids[0], 1);
    assert!(
        (result.scores[0] - 1.0).abs() < 1e-5,
        "Self-similarity should be ≈ 1.0, got {}",
        result.scores[0],
    );
}

/// Vectors persist across close/reopen cycles.
#[test]
fn persistence_across_reopen() {
    let ctx = TestContext::new();
    let dims: u32 = 4;
    let ids: Vec<u64> = vec![100, 200];
    let vectors: Vec<f32> = vec![
        0.5, 0.5, 0.5, 0.5, // id=100
        1.0, 0.0, 0.0, 0.0, // id=200
    ];

    // Phase 1: Write and close.
    {
        let store = VectorStore::open(ctx.db_path()).expect("open failed");
        store.append(&ids, &vectors, dims).expect("append failed");
        // VectorStore dropped here
    }

    // Phase 2: Reopen and search.
    {
        let store = VectorStore::open(ctx.db_path()).expect("reopen failed");
        let result = store
            .search(&[1.0, 0.0, 0.0, 0.0], 2)
            .expect("search failed");

        assert_eq!(result.ids.len(), 2);
        assert_eq!(result.ids[0], 200, "id=200 should match [1,0,0,0] best");
    }
}

/// Load returns all appended vectors with correct dimensions.
#[test]
fn load_returns_all() {
    let ctx = TestContext::new();
    let store = VectorStore::open(ctx.db_path()).expect("open failed");

    let dims: u32 = 8;
    let (ids, vectors) = generate_vectors(10, dims as usize);
    store.append(&ids, &vectors, dims).expect("append failed");

    let loaded = store.load().expect("load failed");

    assert_eq!(loaded.ids.len(), 10);
    assert_eq!(loaded.dims, dims);
    assert_eq!(loaded.vectors.len(), 10 * dims as usize);
    // Verify IDs match (order may differ, so check set equality).
    let mut loaded_ids = loaded.ids.clone();
    loaded_ids.sort();
    assert_eq!(loaded_ids, ids);
}

/// Append validates dimension consistency on the Rust side.
#[test]
fn append_rejects_dimension_mismatch() {
    let ctx = TestContext::new();
    let store = VectorStore::open(ctx.db_path()).expect("open failed");

    let ids: Vec<u64> = vec![1, 2];
    let vectors: Vec<f32> = vec![1.0, 2.0, 3.0]; // 3 floats, but 2 ids * dims should be even

    let result = store.append(&ids, &vectors, 4);
    assert!(result.is_err(), "Should reject mismatched dimensions");
}

/// Empty append is a no-op (does not error).
#[test]
fn empty_append_is_noop() {
    let ctx = TestContext::new();
    let store = VectorStore::open(ctx.db_path()).expect("open failed");

    store
        .append(&[], &[], 4)
        .expect("empty append should succeed");

    let loaded = store.load().expect("load failed");
    assert_eq!(loaded.ids.len(), 0);
}

/// Multiple appends accumulate correctly.
#[test]
fn multiple_appends_accumulate() {
    let ctx = TestContext::new();
    let store = VectorStore::open(ctx.db_path()).expect("open failed");

    let dims: u32 = 2;

    store.append(&[1], &[1.0, 0.0], dims).expect("append 1");
    store.append(&[2], &[0.0, 1.0], dims).expect("append 2");
    store.append(&[3], &[0.7, 0.7], dims).expect("append 3");

    let loaded = store.load().expect("load failed");
    assert_eq!(loaded.ids.len(), 3, "Should have 3 vectors after 3 appends");
}

/// Larger batch: 1000 vectors at dim=128.
#[test]
fn batch_1k_dim128() {
    let ctx = TestContext::new();
    let store = VectorStore::open(ctx.db_path()).expect("open failed");

    let dims: u32 = 128;
    let (ids, vectors) = generate_vectors(1000, dims as usize);
    store.append(&ids, &vectors, dims).expect("append failed");

    // Search for the first vector — should return itself as top result.
    let query = &vectors[0..dims as usize];
    let result = store.search(query, 1).expect("search failed");
    assert_eq!(result.ids[0], 1);
    assert!((result.scores[0] - 1.0).abs() < 1e-4);

    let loaded = store.load().expect("load failed");
    assert_eq!(loaded.ids.len(), 1000);
}
