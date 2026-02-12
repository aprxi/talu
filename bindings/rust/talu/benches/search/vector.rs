//! Vector search benchmarks on the persistent "medium" dataset (10k vectors, 384d).

use criterion::Criterion;

#[allow(dead_code)]
#[path = "../common/mod.rs"]
mod common;

const TOP_K: u32 = 10;
const BATCH_QUERIES: usize = 100;

pub fn bench_search_10k(c: &mut Criterion) {
    let db_path = common::dataset::dataset_path("medium");
    let store = talu::vector::VectorStore::open(&db_path).unwrap();
    let query = common::make_query(common::dataset::DIMS);

    c.bench_function("search_10k_vectors", |b| {
        b.iter(|| {
            let result = store.search(&query, TOP_K).expect("search");
            assert_eq!(result.ids.len(), TOP_K as usize);
        });
    });
}

pub fn bench_search_batch(c: &mut Criterion) {
    let db_path = common::dataset::dataset_path("medium");
    let store = talu::vector::VectorStore::open(&db_path).unwrap();
    let queries = common::make_query_batch(BATCH_QUERIES, common::dataset::DIMS);

    c.bench_function("search_batch_throughput", |b| {
        b.iter(|| {
            let result = store
                .search_batch(&queries, common::dataset::DIMS, TOP_K)
                .expect("search_batch");
            assert_eq!(result.query_count, BATCH_QUERIES as u32);
        });
    });
}
