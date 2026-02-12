mod ingest;
mod lifecycle;

use criterion::{criterion_group, criterion_main};

// @bench: ingest_throughput/batch_size_1   — 100 individual appends (IOPS-bound)
// @bench: ingest_throughput/batch_size_100 — single batch of 100 (bandwidth-bound)
// @bench: ingest_10k_vectors              — bulk ingest 10k vectors (384d)
// @bench: lifecycle/cold_open_10k         — open existing store with 10k vectors
// @bench: lifecycle/cold_load_10k         — open + load 10k vectors (384d)

criterion_group!(
    storage_benches,
    ingest::bench_ingest_strategies,
    ingest::bench_ingest_10k,
    lifecycle::bench_cold_start,
);
criterion_main!(storage_benches);
