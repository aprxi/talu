//! Batch search integration tests.
//!
//! Validates search_batch: multi-query in one pass.

use crate::capi::db::common::{generate_vectors, TestContext};
use talu::vector::VectorStore;

/// Batch search with orthogonal unit vectors returns correct per-query results.
#[test]
fn batch_search_orthogonal() {
    let ctx = TestContext::new();
    let store = VectorStore::open(ctx.db_path()).expect("open failed");

    let dims: u32 = 4;
    let ids: Vec<u64> = vec![10, 20, 30, 40];
    let vectors: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, // id=10
        0.0, 1.0, 0.0, 0.0, // id=20
        0.0, 0.0, 1.0, 0.0, // id=30
        0.0, 0.0, 0.0, 1.0, // id=40
    ];
    store.append(&ids, &vectors, dims).expect("append failed");

    // Two queries: [1,0,0,0] should match id=10, [0,0,0,1] should match id=40
    let queries: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    let result = store
        .search_batch(&queries, dims, 2)
        .expect("search_batch failed");

    assert_eq!(result.query_count, 2);
    let cpq = result.count_per_query as usize;
    assert!(cpq >= 1, "Expected at least 1 result per query");

    // Query 0 top result
    assert_eq!(result.ids[0], 10, "Query 0 top should be id=10");
    assert!((result.scores[0] - 1.0).abs() < 1e-5);

    // Query 1 top result
    assert_eq!(result.ids[cpq], 40, "Query 1 top should be id=40");
    assert!((result.scores[cpq] - 1.0).abs() < 1e-5);
}

/// Batch search result sizes match expectations.
#[test]
fn batch_search_result_layout() {
    let ctx = TestContext::new();
    let store = VectorStore::open(ctx.db_path()).expect("open failed");

    let dims: u32 = 8;
    let (ids, vectors) = generate_vectors(50, dims as usize);
    store.append(&ids, &vectors, dims).expect("append failed");

    // 3 queries, k=5
    let queries: Vec<f32> = vectors[0..3 * dims as usize].to_vec();
    let result = store
        .search_batch(&queries, dims, 5)
        .expect("search_batch failed");

    assert_eq!(result.query_count, 3);
    let cpq = result.count_per_query as usize;
    assert!(cpq <= 5, "count_per_query should be <= k");
    assert_eq!(result.ids.len(), cpq * 3);
    assert_eq!(result.scores.len(), cpq * 3);
}

/// Batch search with single query matches regular search.
#[test]
fn batch_search_single_matches_search() {
    let ctx = TestContext::new();
    let store = VectorStore::open(ctx.db_path()).expect("open failed");

    let dims: u32 = 4;
    let (ids, vectors) = generate_vectors(20, dims as usize);
    store.append(&ids, &vectors, dims).expect("append failed");

    let query = &vectors[0..dims as usize];

    let single = store.search(query, 3).expect("search failed");
    let batch = store
        .search_batch(query, dims, 3)
        .expect("search_batch failed");

    assert_eq!(batch.query_count, 1);
    let cpq = batch.count_per_query as usize;
    assert_eq!(single.ids.len(), cpq);
    assert_eq!(single.ids, batch.ids);
    for (s, b) in single.scores.iter().zip(batch.scores.iter()) {
        assert!((s - b).abs() < 1e-6, "Score mismatch: {} vs {}", s, b);
    }
}

/// search_batch rejects queries not a multiple of dims.
#[test]
fn batch_search_rejects_bad_query_len() {
    let ctx = TestContext::new();
    let store = VectorStore::open(ctx.db_path()).expect("open failed");

    store
        .append(&[1], &[1.0, 0.0, 0.0, 0.0], 4)
        .expect("append failed");

    // 5 floats is not a multiple of 4
    let bad_queries = vec![1.0, 0.0, 0.0, 0.0, 0.5];
    let result = store.search_batch(&bad_queries, 4, 1);
    assert!(
        result.is_err(),
        "Should reject query_len not multiple of dims"
    );
}

/// Batch search with k larger than stored count still works.
#[test]
fn batch_search_k_exceeds_stored() {
    let ctx = TestContext::new();
    let store = VectorStore::open(ctx.db_path()).expect("open failed");

    let dims: u32 = 4;
    store
        .append(&[1, 2], &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dims)
        .expect("append failed");

    // Ask for k=100 but only 2 vectors exist
    let queries: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let result = store
        .search_batch(&queries, dims, 100)
        .expect("search_batch failed");

    assert_eq!(result.query_count, 2);
    let cpq = result.count_per_query as usize;
    assert!(cpq <= 2, "Can't return more than stored: got {}", cpq);
}
