//! Blob concurrent-write tests.

use crate::capi::db::common::TestContext;
use std::sync::Arc;
use std::thread;
use talu::blobs::BlobsHandle;

#[test]
fn concurrent_same_payload_writes_return_identical_blob_refs() {
    let ctx = TestContext::new();
    let db_path = Arc::new(ctx.db_path().to_string());
    let payload = Arc::new(b"same-payload-from-many-writers".to_vec());

    let mut handles = Vec::new();
    for _ in 0..10 {
        let db_path = db_path.clone();
        let payload = payload.clone();
        handles.push(thread::spawn(move || {
            let blobs = BlobsHandle::open(&*db_path).expect("open blobs");
            blobs.put(&payload).expect("put")
        }));
    }

    let mut refs = Vec::new();
    for h in handles {
        refs.push(h.join().expect("join"));
    }

    let first = refs
        .first()
        .cloned()
        .expect("at least one reference should be produced");
    for r in &refs {
        assert_eq!(
            r, &first,
            "expected all writers to resolve to same CAS reference"
        );
    }

    let blobs = BlobsHandle::open(&*db_path).expect("open blobs for read");
    let loaded = blobs.read_all(&first).expect("read_all");
    assert_eq!(loaded, payload.as_slice());
}
