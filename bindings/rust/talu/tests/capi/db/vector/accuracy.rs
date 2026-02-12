//! Vector search ranking accuracy tests.
//!
//! Verifies that the brute-force dot-product search returns results in
//! the correct order: descending by similarity score, with the closest
//! vector first.

use crate::capi::db::common::TestContext;
use talu::vector::VectorStore;

/// L2-normalize a vector in-place.
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Ranking accuracy
// ---------------------------------------------------------------------------

/// Three vectors at known angles: search returns [Target, Close, Far].
///
/// Setup (4D, L2-normalized):
///   - Query:  [1, 0, 0, 0]  (unit vector along axis 0)
///   - Target: [0.95, 0.05, 0, 0]  → nearly aligned with query
///   - Close:  [0.5, 0.5, 0, 0]   → 45° from query
///   - Far:    [0, 0, 0, 1]        → orthogonal to query (dot = 0)
///
/// Expected ranking by dot product (descending):
///   1. Target (highest dot product, closest to query)
///   2. Close  (moderate dot product)
///   3. Far    (dot product ≈ 0)
#[test]
fn ranking_accuracy() {
    let ctx = TestContext::new();
    let dims: u32 = 4;

    // Build vectors (L2-normalized).
    let mut target = vec![0.95_f32, 0.05, 0.0, 0.0];
    let mut close = vec![0.5_f32, 0.5, 0.0, 0.0];
    let mut far = vec![0.0_f32, 0.0, 0.0, 1.0];
    normalize(&mut target);
    normalize(&mut close);
    normalize(&mut far);

    // IDs: Target=1, Close=2, Far=3.
    let ids: Vec<u64> = vec![1, 2, 3];
    let mut vectors = Vec::with_capacity(3 * dims as usize);
    vectors.extend_from_slice(&target);
    vectors.extend_from_slice(&close);
    vectors.extend_from_slice(&far);

    // Store and search.
    let store = VectorStore::open(ctx.db_path()).expect("open");
    store.append(&ids, &vectors, dims).expect("append");

    let query = vec![1.0_f32, 0.0, 0.0, 0.0]; // unit vector on axis 0
    let results = store.search(&query, 3).expect("search");

    assert_eq!(
        results.ids.len(),
        3,
        "Expected 3 results, got {}",
        results.ids.len(),
    );

    // Verify exact ranking order.
    assert_eq!(
        results.ids[0], 1,
        "Rank 1 should be Target (id=1), got id={}",
        results.ids[0],
    );
    assert_eq!(
        results.ids[1], 2,
        "Rank 2 should be Close (id=2), got id={}",
        results.ids[1],
    );
    assert_eq!(
        results.ids[2], 3,
        "Rank 3 should be Far (id=3), got id={}",
        results.ids[2],
    );

    // Verify scores are strictly descending.
    assert!(
        results.scores[0] > results.scores[1],
        "Score[0]={} should be > Score[1]={}",
        results.scores[0],
        results.scores[1],
    );
    assert!(
        results.scores[1] > results.scores[2],
        "Score[1]={} should be > Score[2]={}",
        results.scores[1],
        results.scores[2],
    );

    // Far vector is orthogonal to query → dot product ≈ 0.
    assert!(
        results.scores[2].abs() < 0.01,
        "Far vector score should be ~0, got {}",
        results.scores[2],
    );
}

/// Requesting fewer than all results returns the top-k correctly.
///
/// With 5 vectors of decreasing similarity, search(k=2) returns
/// only the 2 most similar.
#[test]
fn top_k_truncation() {
    let ctx = TestContext::new();
    let dims: u32 = 4;

    // 5 vectors with decreasing alignment to [1,0,0,0].
    let raw_vectors: Vec<[f32; 4]> = vec![
        [1.0, 0.0, 0.0, 0.0], // id=1: perfect match
        [0.9, 0.1, 0.0, 0.0], // id=2: very close
        [0.5, 0.5, 0.0, 0.0], // id=3: moderate
        [0.1, 0.9, 0.0, 0.0], // id=4: far
        [0.0, 0.0, 0.0, 1.0], // id=5: orthogonal
    ];

    let ids: Vec<u64> = (1..=5).collect();
    let mut vectors = Vec::with_capacity(5 * dims as usize);
    for mut v in raw_vectors {
        normalize(&mut v);
        vectors.extend_from_slice(&v);
    }

    let store = VectorStore::open(ctx.db_path()).expect("open");
    store.append(&ids, &vectors, dims).expect("append");

    let query = vec![1.0_f32, 0.0, 0.0, 0.0];
    let results = store.search(&query, 2).expect("search k=2");

    assert_eq!(results.ids.len(), 2, "Expected 2 results for k=2");
    assert_eq!(
        results.ids[0], 1,
        "Top result should be id=1 (perfect match)"
    );
    assert_eq!(
        results.ids[1], 2,
        "Second result should be id=2 (very close)"
    );
}

/// Identical vectors return equal scores.
///
/// Two vectors identical to the query should both score 1.0 (dot product
/// of two identical unit vectors).
#[test]
fn identical_vectors_equal_scores() {
    let ctx = TestContext::new();
    let dims: u32 = 4;

    let unit = vec![1.0_f32, 0.0, 0.0, 0.0];

    let ids: Vec<u64> = vec![1, 2];
    let mut vectors = Vec::with_capacity(2 * dims as usize);
    vectors.extend_from_slice(&unit);
    vectors.extend_from_slice(&unit);

    let store = VectorStore::open(ctx.db_path()).expect("open");
    store.append(&ids, &vectors, dims).expect("append");

    let results = store.search(&unit, 2).expect("search");

    assert_eq!(results.ids.len(), 2);

    // Both scores should be 1.0 (within floating point tolerance).
    for (i, &score) in results.scores.iter().enumerate() {
        assert!(
            (score - 1.0).abs() < 1e-5,
            "Score[{i}] should be ~1.0, got {score}",
        );
    }
}

/// Opposite vectors score negatively.
///
/// A vector pointing in the exact opposite direction to the query
/// should have a dot product of -1.0.
#[test]
fn opposite_vectors_negative_score() {
    let ctx = TestContext::new();
    let dims: u32 = 4;

    let same = vec![1.0_f32, 0.0, 0.0, 0.0];
    let opposite = vec![-1.0_f32, 0.0, 0.0, 0.0];

    let ids: Vec<u64> = vec![1, 2];
    let mut vectors = Vec::with_capacity(2 * dims as usize);
    vectors.extend_from_slice(&same);
    vectors.extend_from_slice(&opposite);

    let store = VectorStore::open(ctx.db_path()).expect("open");
    store.append(&ids, &vectors, dims).expect("append");

    let query = vec![1.0_f32, 0.0, 0.0, 0.0];
    let results = store.search(&query, 2).expect("search");

    assert_eq!(results.ids.len(), 2);

    // Same direction should be first (score ≈ 1.0).
    assert_eq!(results.ids[0], 1);
    assert!(
        (results.scores[0] - 1.0).abs() < 1e-5,
        "Same-direction score should be ~1.0, got {}",
        results.scores[0],
    );

    // Opposite direction should be last (score ≈ -1.0).
    assert_eq!(results.ids[1], 2);
    assert!(
        (results.scores[1] + 1.0).abs() < 1e-5,
        "Opposite-direction score should be ~-1.0, got {}",
        results.scores[1],
    );
}
