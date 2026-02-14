//! Content hash correctness tests for CAS blob refs.

use crate::capi::db::common::TestContext;
use sha2::{Digest, Sha256};
use talu::blobs::BlobsHandle;

#[test]
fn blob_ref_matches_sha256_of_payload() {
    let ctx = TestContext::new();
    let blobs = BlobsHandle::open(ctx.db_path()).expect("open blobs");

    // 1 MiB deterministic payload.
    let mut payload = vec![0u8; 1024 * 1024];
    for (i, byte) in payload.iter_mut().enumerate() {
        *byte = ((i * 31) % 251) as u8;
    }

    let blob_ref = blobs.put(&payload).expect("put payload");

    let mut hasher = Sha256::new();
    hasher.update(&payload);
    let expected_hex = format!("{:x}", hasher.finalize());
    let expected_ref = format!("sha256:{}", expected_hex);

    assert_eq!(
        blob_ref, expected_ref,
        "blob reference should equal sha256 digest of stored bytes"
    );
}
