//! Interrupted upload behavior and cleanup tests.

use crate::capi::db::common::TestContext;
use std::fs;
use std::path::Path;
use talu::blobs::BlobsHandle;

#[test]
fn dropping_unfinished_stream_leaves_no_tmp_files_and_store_remains_writable() {
    let ctx = TestContext::new();
    let blobs = BlobsHandle::open(ctx.db_path()).expect("open blobs");

    {
        let mut writer = blobs.open_write_stream().expect("open writer");
        let chunk = vec![0xABu8; 256 * 1024];
        writer.write(&chunk).expect("write partial payload");
        // Drop without finish() to simulate interrupted upload.
    }

    let blobs_dir = Path::new(ctx.db_path()).join("blobs");
    assert!(
        count_tmp_files(&blobs_dir).expect("scan tmp files") == 0,
        "unfinished stream should not leave temporary files"
    );

    let blob_ref = blobs.put(b"post-interruption-write").expect("put");
    let loaded = blobs.read_all(&blob_ref).expect("read_all");
    assert_eq!(loaded, b"post-interruption-write");
}

fn count_tmp_files(root: &Path) -> std::io::Result<usize> {
    if !root.exists() {
        return Ok(0);
    }

    let mut count = 0usize;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with(".tmp-") {
                    count += 1;
                }
            }
        }
    }

    Ok(count)
}
