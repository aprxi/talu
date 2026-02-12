//! Ingest throughput benchmarks.
//!
//! Measures write throughput variance by batch size (IOPS vs bandwidth)
//! and bulk ingest performance.

use criterion::{BatchSize, Criterion, Throughput};

#[allow(dead_code)]
#[path = "../common/mod.rs"]
mod common;

const DIMS: u32 = 128;

pub fn bench_ingest_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest_throughput");
    group.sample_size(10);

    let total_items = 100;
    let (ids, vectors) = common::make_corpus(total_items, DIMS);

    let payload_size = (total_items * 8) + (total_items * DIMS as usize * 4);
    group.throughput(Throughput::Bytes(payload_size as u64));

    // One at a time (IOPS-bound — 1 fsync per row).
    group.bench_function("batch_size_1", |b| {
        b.iter_batched(
            common::fresh_vector_store,
            |(_dir, store)| {
                for i in 0..total_items {
                    let v_slice = &vectors[i * DIMS as usize..(i + 1) * DIMS as usize];
                    store.append(&[ids[i]], v_slice, DIMS).unwrap();
                }
            },
            BatchSize::PerIteration,
        );
    });

    // All at once (bandwidth-bound — 1 fsync total).
    group.bench_function("batch_size_100", |b| {
        b.iter_batched(
            common::fresh_vector_store,
            |(_dir, store)| {
                store.append(&ids, &vectors, DIMS).unwrap();
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

pub fn bench_ingest_10k(c: &mut Criterion) {
    let dims: u32 = 384;
    let (ids, vectors) = common::make_corpus(10_000, dims);

    c.bench_function("ingest_10k_vectors", |b| {
        b.iter_batched(
            common::fresh_vector_store,
            |(_dir, store)| {
                store.append(&ids, &vectors, dims).expect("append");
            },
            BatchSize::PerIteration,
        );
    });
}
