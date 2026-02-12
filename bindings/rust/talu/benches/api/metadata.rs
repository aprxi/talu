//! Metadata update latency (notify_session_update).

use criterion::Criterion;
use talu::ChatHandle;

#[allow(dead_code)]
#[path = "../common/mod.rs"]
mod common;

pub fn bench_metadata_update(c: &mut Criterion) {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().to_str().unwrap();
    let sid = "bench-session";

    let chat = ChatHandle::new(None).unwrap();
    chat.set_storage_db(db_path, sid).unwrap();

    c.bench_function("metadata_update_latency", |b| {
        b.iter(|| {
            chat.notify_session_update(Some("model-v2"), Some("Updated Title"), Some("active"))
                .unwrap();
        });
    });
}
