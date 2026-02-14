//! Blob GC behavior tests.

use crate::capi::db::common::TestContext;
use talu::blobs::BlobsHandle;
use talu::documents::DocumentsHandle;

#[test]
fn gc_preserves_inline_file_metadata_blob_ref_and_deletes_orphan_blob() {
    let ctx = TestContext::new();
    let blobs = BlobsHandle::open(ctx.db_path()).expect("open blobs");
    let docs = DocumentsHandle::open(ctx.db_path()).expect("open docs");

    let referenced_blob = blobs.put(b"referenced-content").expect("put referenced");
    let orphan_blob = blobs.put(b"orphan-content").expect("put orphan");

    let file_doc_id = "file_gc_inline_ref";
    let metadata = format!(
        "{{\"blob_ref\":\"{}\",\"original_name\":\"gc.txt\",\"size\":17}}",
        referenced_blob
    );
    docs.create(
        file_doc_id,
        "file",
        "gc.txt",
        &metadata,
        None,
        None,
        Some("active"),
        None,
        None,
    )
    .expect("create file metadata doc");

    let stats = blobs.gc_with_min_age(0).expect("run gc");
    assert!(
        stats.deleted_blob_files >= 1,
        "expected at least one deleted blob, stats: {stats:?}"
    );

    assert!(
        blobs.contains(&referenced_blob).expect("contains referenced"),
        "referenced blob should remain"
    );
    assert!(
        !blobs.contains(&orphan_blob).expect("contains orphan"),
        "orphan blob should be deleted"
    );
}
