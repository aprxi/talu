//! Blob corruption-resilience and malformed-reference tests.

use crate::capi::db::common::TestContext;
use talu::blobs::{BlobError, BlobsHandle};

#[test]
fn drop_write_stream_before_finish_does_not_block_subsequent_writes() {
    let ctx = TestContext::new();
    let blobs = BlobsHandle::open(ctx.db_path()).expect("open blobs");

    {
        let mut writer = blobs.open_write_stream().expect("open writer");
        writer
            .write(b"incomplete-upload-data")
            .expect("write incomplete data");
        // Drop without finish; should not poison future uploads.
    }

    let blob_ref = blobs.put(b"complete-payload").expect("put complete");
    let loaded = blobs.read_all(&blob_ref).expect("read complete");
    assert_eq!(loaded, b"complete-payload");
}

#[test]
fn open_stream_rejects_malformed_blob_refs() {
    let ctx = TestContext::new();
    let blobs = BlobsHandle::open(ctx.db_path()).expect("open blobs");

    let bad_refs = [
        "",
        "sha256:xyz",
        "sha256:zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
        "../../etc/passwd",
        "multi:not-a-valid-digest",
    ];

    for bad in bad_refs {
        let err = blobs
            .open_stream(bad)
            .expect_err("expected malformed ref to fail");
        assert!(
            matches!(err, BlobError::InvalidArgument(_)),
            "expected InvalidArgument for bad ref `{}`, got {:?}",
            bad,
            err
        );
    }
}
