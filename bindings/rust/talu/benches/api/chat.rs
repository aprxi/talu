//! Chat append latency through the full FFI path.
//!
//! Measures: FFI call -> Lock acquire -> WAL append -> fsync.

use criterion::Criterion;

#[allow(dead_code)]
#[path = "../common/mod.rs"]
mod common;

pub fn bench_append_message(c: &mut Criterion) {
    let (_dir, chat, _sid) = common::fresh_chat();
    let payload = b"Hello, this is a benchmark message for append latency.";

    c.bench_function("append_message_latency", |b| {
        b.iter(|| {
            common::append_message(&chat, payload);
        });
    });
}
