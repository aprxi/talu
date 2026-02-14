//! Blob lifecycle tests.

use crate::capi::db::common::TestContext;
use talu::blobs::BlobsHandle;

#[test]
fn put_and_stream_read_roundtrip() {
    let ctx = TestContext::new();
    let blobs = BlobsHandle::open(ctx.db_path()).expect("open blobs");

    let payload = b"blob-lifecycle-payload";
    let blob_ref = blobs.put(payload).expect("put");
    assert!(
        blob_ref.starts_with("sha256:") || blob_ref.starts_with("multi:"),
        "unexpected blob ref: {}",
        blob_ref
    );

    let mut stream = blobs.open_stream(&blob_ref).expect("open stream");
    assert_eq!(
        stream.total_size().expect("total size"),
        payload.len() as u64
    );

    let mut out = Vec::new();
    let mut buf = [0u8; 5];
    loop {
        let read = stream.read(&mut buf).expect("read");
        if read == 0 {
            break;
        }
        out.extend_from_slice(&buf[..read]);
    }
    assert_eq!(out, payload);
}

#[test]
fn put_empty_blob_and_read_all() {
    let ctx = TestContext::new();
    let blobs = BlobsHandle::open(ctx.db_path()).expect("open blobs");

    let blob_ref = blobs.put(&[]).expect("put empty");
    let loaded = blobs.read_all(&blob_ref).expect("read_all");
    assert!(loaded.is_empty(), "expected empty payload");
}

#[test]
fn stream_can_move_to_worker_thread() {
    let ctx = TestContext::new();
    let blobs = BlobsHandle::open(ctx.db_path()).expect("open blobs");

    let payload = b"cross-thread-blob-read";
    let blob_ref = blobs.put(payload).expect("put");
    let mut stream = blobs.open_stream(&blob_ref).expect("open stream");

    let handle = std::thread::spawn(move || {
        let mut out = Vec::new();
        let mut buf = [0u8; 8];
        loop {
            let read = stream.read(&mut buf).expect("read");
            if read == 0 {
                break;
            }
            out.extend_from_slice(&buf[..read]);
        }
        out
    });

    let out = handle.join().expect("worker join");
    assert_eq!(out, payload);
}

#[test]
fn write_stream_roundtrip() {
    let ctx = TestContext::new();
    let blobs = BlobsHandle::open(ctx.db_path()).expect("open blobs");

    let mut writer = blobs.open_write_stream().expect("open write stream");
    writer.write(b"write-").expect("write part 1");
    writer.write(b"stream").expect("write part 2");
    let blob_ref = writer.finish().expect("finish");

    let loaded = blobs.read_all(&blob_ref).expect("read_all");
    assert_eq!(loaded, b"write-stream");
}

#[test]
fn contains_reports_existing_and_missing_refs() {
    let ctx = TestContext::new();
    let blobs = BlobsHandle::open(ctx.db_path()).expect("open blobs");

    let blob_ref = blobs.put(b"contains-payload").expect("put");
    assert!(blobs.contains(&blob_ref).expect("contains existing"));

    let missing = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
    assert!(!blobs.contains(missing).expect("contains missing"));
}

#[test]
fn list_returns_written_blob_refs() {
    let ctx = TestContext::new();
    let blobs = BlobsHandle::open(ctx.db_path()).expect("open blobs");

    let ref_a = blobs.put(b"list-a").expect("put a");
    let ref_b = blobs.put(b"list-b").expect("put b");
    let refs = blobs.list(None).expect("list blobs");

    assert!(refs.contains(&ref_a), "expected list to contain first blob");
    assert!(
        refs.contains(&ref_b),
        "expected list to contain second blob"
    );
}
