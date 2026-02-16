mod updates;

use criterion::{criterion_group, criterion_main};

// @bench: hot_document/get_after_N_updates — point lookup after 1..5000 updates
// @bench: hot_document/list_after_N_updates — list scan after 1..5000 updates

criterion_group!(document_benches, updates::bench_hot_document,);
criterion_main!(document_benches);
