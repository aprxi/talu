//! Cold start benchmarks: open + load latency with data on disk.
//!
//! Uses the persistent "medium" dataset (10k vectors, 384d).

use criterion::{BatchSize, Criterion};
use talu::vector::VectorStore;

#[allow(dead_code)]
#[path = "../common/mod.rs"]
mod common;

pub fn bench_cold_start(c: &mut Criterion) {
    let db_path = common::dataset::dataset_path("medium");

    let mut group = c.benchmark_group("lifecycle");

    group.bench_function("cold_open_10k", |b| {
        b.iter(|| {
            let _store = VectorStore::open(&db_path).unwrap();
        });
    });

    group.bench_function("cold_load_10k", |b| {
        b.iter_batched(
            || VectorStore::open(&db_path).unwrap(),
            |store| {
                let _data = store.load().unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}
