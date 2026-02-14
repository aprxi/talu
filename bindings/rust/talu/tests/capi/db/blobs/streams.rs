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

#[test]
fn read_stream_seek_repositions_for_subsequent_reads() {
    let ctx = TestContext::new();
    let blobs = BlobsHandle::open(ctx.db_path()).expect("open blobs");

    let blob_ref = blobs.put(b"seek-stream-payload").expect("put");
    let mut stream = blobs.open_stream(&blob_ref).expect("open stream");

    stream.seek(5).expect("seek");
    let mut out = Vec::new();
    let mut buf = [0u8; 4];
    loop {
        let read = stream.read(&mut buf).expect("read");
        if read == 0 {
            break;
        }
        out.extend_from_slice(&buf[..read]);
    }
    assert_eq!(out, b"stream-payload");
}
