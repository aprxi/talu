//! Blob stream boundary and chunking tests.

use crate::capi::db::common::TestContext;
use talu::blobs::BlobsHandle;

#[test]
fn write_stream_zero_byte_payload_produces_readable_empty_blob() {
    let ctx = TestContext::new();
    let blobs = BlobsHandle::open(ctx.db_path()).expect("open blobs");

    let writer = blobs.open_write_stream().expect("open write stream");
    let blob_ref = writer.finish().expect("finish empty stream");
    let loaded = blobs.read_all(&blob_ref).expect("read_all");
    assert!(loaded.is_empty(), "expected empty payload");
}

#[test]
fn write_stream_handles_misaligned_chunk_sizes() {
    let ctx = TestContext::new();
    let blobs = BlobsHandle::open(ctx.db_path()).expect("open blobs");

    let mut writer = blobs.open_write_stream().expect("open write stream");

    let tiny = [0xABu8; 1];
    writer.write(&tiny).expect("write 1-byte chunk");

    let medium = vec![0xCDu8; 4097];
    writer.write(&medium).expect("write 4097-byte chunk");

    let large = vec![0xEFu8; 65537];
    writer.write(&large).expect("write 65537-byte chunk");

    let blob_ref = writer.finish().expect("finish");
    let loaded = blobs.read_all(&blob_ref).expect("read_all");

    assert_eq!(loaded.len(), tiny.len() + medium.len() + large.len());
    assert_eq!(&loaded[..1], &tiny);
    assert_eq!(&loaded[1..(1 + 4097)], medium.as_slice());
    assert_eq!(&loaded[(1 + 4097)..], large.as_slice());
}
