//! Large I/O and chunk-alignment tests for blob streaming.

use crate::capi::db::common::TestContext;
use talu::blobs::BlobsHandle;

#[test]
fn write_stream_roundtrips_multi_megabyte_payload() {
    let ctx = TestContext::new();
    let blobs = BlobsHandle::open(ctx.db_path()).expect("open blobs");

    // 3 MiB deterministic payload to exceed small internal buffers.
    let mut payload = vec![0u8; 3 * 1024 * 1024];
    for (i, byte) in payload.iter_mut().enumerate() {
        *byte = (i % 251) as u8;
    }

    let mut writer = blobs.open_write_stream().expect("open write stream");
    writer.write(&payload).expect("write full payload");
    let blob_ref = writer.finish().expect("finish stream");

    let loaded = blobs.read_all(&blob_ref).expect("read_all");
    assert_eq!(loaded, payload);
}

#[test]
fn write_stream_roundtrips_prime_sized_chunks() {
    let ctx = TestContext::new();
    let blobs = BlobsHandle::open(ctx.db_path()).expect("open blobs");

    const CHUNK: usize = 1009; // Prime chunk size to stress alignment boundaries.
    const CHUNKS: usize = 2048;

    let mut writer = blobs.open_write_stream().expect("open write stream");
    let mut expected = Vec::with_capacity(CHUNK * CHUNKS);

    for i in 0..CHUNKS {
        let mut chunk = vec![0u8; CHUNK];
        for (j, byte) in chunk.iter_mut().enumerate() {
            *byte = ((i + j) % 251) as u8;
        }
        writer.write(&chunk).expect("write chunk");
        expected.extend_from_slice(&chunk);
    }

    let blob_ref = writer.finish().expect("finish stream");
    let loaded = blobs.read_all(&blob_ref).expect("read_all");
    assert_eq!(loaded, expected);
}
