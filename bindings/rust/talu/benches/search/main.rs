mod scale;
mod vector;

use criterion::{criterion_group, criterion_main};

// @bench: search_10k_vectors        — single query on 10k corpus (384d)
// @bench: search_batch_throughput    — 100 queries on 10k corpus
// @bench: scale_1m/search_1m_single — single query on 1M corpus (128d, 512MB)

criterion_group!(
    search_benches,
    vector::bench_search_10k,
    vector::bench_search_batch,
    scale::bench_search_1m,
);
criterion_main!(search_benches);
