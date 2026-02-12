//! High-scale vector benchmarks (1M vectors).
//!
//! Stress test memory bandwidth by exceeding L3 cache size.
//! Dataset: 1M vectors * 128 dims * 4 bytes = 512 MB.

use criterion::Criterion;

#[allow(dead_code)]
#[path = "../common/mod.rs"]
mod common;

const CORPUS_SIZE: usize = 1_000_000;
const DIMS: u32 = 128;
const TOP_K: u32 = 10;

fn preloaded_large_store() -> (tempfile::TempDir, talu::vector::VectorStore) {
    let (dir, store) = common::fresh_vector_store();

    let chunk_size = 100_000;
    let mut vec_buffer = Vec::with_capacity(chunk_size * DIMS as usize);
    let mut id_buffer = Vec::with_capacity(chunk_size);

    for chunk_start in (0..CORPUS_SIZE).step_by(chunk_size) {
        vec_buffer.clear();
        id_buffer.clear();

        for i in 0..chunk_size {
            let val = (i as f32) * 0.001;
            id_buffer.push((chunk_start + i) as u64);
            vec_buffer.extend(std::iter::repeat(val).take(DIMS as usize));
        }
        store
            .append(&id_buffer, &vec_buffer, DIMS)
            .expect("append chunk");
    }

    (dir, store)
}

pub fn bench_search_1m(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale_1m");
    group.sample_size(10);

    let (_dir, store) = preloaded_large_store();
    let query = common::make_query(DIMS);

    group.bench_function("search_1m_single", |b| {
        b.iter(|| {
            let result = store.search(&query, TOP_K).expect("search");
            assert_eq!(result.ids.len(), TOP_K as usize);
        });
    });

    group.finish();
}
