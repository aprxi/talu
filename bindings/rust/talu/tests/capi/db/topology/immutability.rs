//! Append-only block file invariants.
//!
//! TaluDB's storage model requires that once bytes are written to a block
//! file (`current.talu` or sealed `seg-*.talu`), they are never modified.
//! New data, metadata updates, and tombstones are always appended as new
//! blocks at the end of the file.
//!
//! This invariant is critical for S3 replication: sealed segments can be
//! uploaded once and never re-uploaded. Violations would cause silent data
//! corruption in eventually-consistent object stores.

use sha2::{Digest, Sha256};
use std::path::PathBuf;

use crate::capi::db::common::{find_wal_files, TestContext};
use talu::responses::{MessageRole, ResponsesView};
use talu::vector::VectorStore;
use talu::{ChatHandle, StorageHandle};

/// Read a file and return its SHA-256 hash.
fn sha256_file(path: &std::path::Path) -> Vec<u8> {
    let data = std::fs::read(path).expect("read file for hashing");
    let mut hasher = Sha256::new();
    hasher.update(&data);
    hasher.finalize().to_vec()
}

/// Read the first `n` bytes of a file.
fn read_prefix(path: &std::path::Path, n: usize) -> Vec<u8> {
    let data = std::fs::read(path).expect("read file for prefix");
    data[..n.min(data.len())].to_vec()
}

/// Get the size of a file.
fn file_size(path: &std::path::Path) -> u64 {
    std::fs::metadata(path)
        .expect("metadata for file size")
        .len()
}

/// Resolve the chat namespace directory for a DB root.
fn chat_dir(db_root: &str) -> PathBuf {
    PathBuf::from(db_root).join("chat")
}

/// Resolve the vector namespace directory for a DB root.
fn vector_dir(db_root: &str) -> PathBuf {
    PathBuf::from(db_root).join("vector")
}

// ---------------------------------------------------------------------------
// Chat: append-only current.talu
// ---------------------------------------------------------------------------

/// Writing new messages appends blocks; existing bytes are untouched.
///
/// Strategy: write batch A, snapshot the file prefix, write batch B,
/// verify the prefix bytes are unchanged and the file only grew.
#[test]
fn chat_blocks_are_append_only() {
    let ctx = TestContext::new();
    let sid = TestContext::unique_session_id();

    // Batch A: one message + metadata.
    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid).expect("set");
        chat.notify_session_update(None, Some("Title A"), Some("active"))
            .expect("notify");
        let msg = b"Batch A message";
        let rc = unsafe {
            talu_sys::talu_responses_append_message(
                chat.responses().as_ptr(),
                MessageRole::User as u8,
                msg.as_ptr(),
                msg.len(),
            )
        };
        assert!(rc >= 0, "append failed: {rc}");
    }

    let data_file = chat_dir(ctx.db_path()).join("current.talu");
    assert!(
        data_file.exists(),
        "current.talu should exist after batch A"
    );

    let size_after_a = file_size(&data_file);
    let prefix_after_a = read_prefix(&data_file, size_after_a as usize);
    let hash_after_a = sha256_file(&data_file);

    // Batch B: new session, new message.
    let sid_b = TestContext::unique_session_id();
    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid_b).expect("set");
        chat.notify_session_update(None, Some("Title B"), None)
            .expect("notify");
        let msg = b"Batch B message";
        let rc = unsafe {
            talu_sys::talu_responses_append_message(
                chat.responses().as_ptr(),
                MessageRole::User as u8,
                msg.as_ptr(),
                msg.len(),
            )
        };
        assert!(rc >= 0);
    }

    let size_after_b = file_size(&data_file);
    assert!(
        size_after_b > size_after_a,
        "File should grow: {size_after_a} -> {size_after_b}",
    );

    // The first `size_after_a` bytes must be identical (append-only).
    let prefix_after_b = read_prefix(&data_file, size_after_a as usize);
    assert_eq!(
        prefix_after_a, prefix_after_b,
        "Existing blocks modified! Append-only invariant violated.",
    );

    // Full-file hash must differ (new data appended).
    let hash_after_b = sha256_file(&data_file);
    assert_ne!(
        hash_after_a, hash_after_b,
        "File hash should change after appending new blocks",
    );
}

/// Metadata updates append new blocks; they do not modify existing ones.
#[test]
fn metadata_update_is_append_only() {
    let ctx = TestContext::new();
    let sid = TestContext::unique_session_id();

    // Initial write.
    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid).expect("set");
        chat.notify_session_update(Some("model-v1"), Some("Original Title"), Some("active"))
            .expect("notify");
    }

    let data_file = chat_dir(ctx.db_path()).join("current.talu");
    let size_before = file_size(&data_file);
    let prefix_before = read_prefix(&data_file, size_before as usize);

    // Metadata overwrite (separate ChatHandle → separate block).
    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid).expect("set");
        chat.notify_session_update(Some("model-v2"), Some("Updated Title"), Some("done"))
            .expect("notify");
    }

    let size_after = file_size(&data_file);
    assert!(
        size_after > size_before,
        "Metadata update should append a new block, not modify in place",
    );

    let prefix_after = read_prefix(&data_file, size_before as usize);
    assert_eq!(
        prefix_before, prefix_after,
        "Original blocks modified by metadata update",
    );
}

/// Session deletion appends tombstone blocks; existing data is untouched.
#[test]
fn deletion_is_append_only() {
    let ctx = TestContext::new();
    let sid = TestContext::unique_session_id();

    // Create session with a message.
    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid).expect("set");
        chat.notify_session_update(None, Some("Doomed"), Some("active"))
            .expect("notify");
        let msg = b"To be deleted";
        let rc = unsafe {
            talu_sys::talu_responses_append_message(
                chat.responses().as_ptr(),
                MessageRole::User as u8,
                msg.as_ptr(),
                msg.len(),
            )
        };
        assert!(rc >= 0);
    }

    let data_file = chat_dir(ctx.db_path()).join("current.talu");
    let size_before = file_size(&data_file);
    let prefix_before = read_prefix(&data_file, size_before as usize);

    // Delete the session.
    let storage = StorageHandle::open(ctx.db_path()).expect("open");
    storage.delete_session(&sid).expect("delete");

    let size_after = file_size(&data_file);
    assert!(
        size_after > size_before,
        "Deletion should append tombstone blocks, not truncate",
    );

    let prefix_after = read_prefix(&data_file, size_before as usize);
    assert_eq!(
        prefix_before, prefix_after,
        "Original blocks modified by deletion",
    );
}

// ---------------------------------------------------------------------------
// Vector: append-only current.talu
// ---------------------------------------------------------------------------

/// Vector appends grow the file; existing blocks stay put.
#[test]
fn vector_blocks_are_append_only() {
    let ctx = TestContext::new();
    let dims: u32 = 4;

    // Batch A: one vector.
    {
        let store = VectorStore::open(ctx.db_path()).expect("open");
        store
            .append(&[1], &[1.0, 0.0, 0.0, 0.0], dims)
            .expect("append");
    }

    let data_file = vector_dir(ctx.db_path()).join("current.talu");
    assert!(data_file.exists(), "vector current.talu should exist");

    let size_after_a = file_size(&data_file);
    let prefix_after_a = read_prefix(&data_file, size_after_a as usize);

    // Batch B: more vectors via a new handle.
    {
        let store = VectorStore::open(ctx.db_path()).expect("open");
        store
            .append(&[2, 3], &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dims)
            .expect("append");
    }

    let size_after_b = file_size(&data_file);
    assert!(
        size_after_b > size_after_a,
        "Vector file should grow after second append",
    );

    let prefix_after_b = read_prefix(&data_file, size_after_a as usize);
    assert_eq!(
        prefix_after_a, prefix_after_b,
        "Vector: existing blocks modified by new append",
    );
}

// ---------------------------------------------------------------------------
// Namespace isolation
// ---------------------------------------------------------------------------

/// Chat and vector operations create files in separate namespace directories.
/// Neither namespace directory is affected by the other's operations.
#[test]
fn namespaces_are_isolated() {
    let ctx = TestContext::new();

    // Write chat data only.
    let sid = TestContext::unique_session_id();
    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid).expect("set");
        chat.notify_session_update(None, Some("Chat Only"), None)
            .expect("notify");
    }

    assert!(
        chat_dir(ctx.db_path()).join("current.talu").exists(),
        "Chat namespace should have current.talu",
    );
    assert!(
        !vector_dir(ctx.db_path()).join("current.talu").exists(),
        "Vector namespace should NOT exist after chat-only operations",
    );

    // Now write vector data.
    {
        let store = VectorStore::open(ctx.db_path()).expect("open");
        store
            .append(&[1], &[1.0, 0.0, 0.0, 0.0], 4)
            .expect("append");
    }

    assert!(
        vector_dir(ctx.db_path()).join("current.talu").exists(),
        "Vector namespace should now have current.talu",
    );

    // Verify chat file was not modified by vector write.
    let chat_file = chat_dir(ctx.db_path()).join("current.talu");
    let chat_hash_before = sha256_file(&chat_file);

    // Another vector write.
    {
        let store = VectorStore::open(ctx.db_path()).expect("open");
        store
            .append(&[2], &[0.0, 1.0, 0.0, 0.0], 4)
            .expect("append");
    }

    let chat_hash_after = sha256_file(&chat_file);
    assert_eq!(
        chat_hash_before, chat_hash_after,
        "Chat file modified by vector operation — namespace leak",
    );
}

// ---------------------------------------------------------------------------
// Block structure
// ---------------------------------------------------------------------------

/// Every block in current.talu starts with the TALU magic (0x554C4154).
///
/// Walks the file, reading each BlockHeader, verifying the magic and
/// advancing by block_len. This ensures the file is a valid chain of
/// blocks with no gaps or corruption.
#[test]
fn all_blocks_have_valid_magic() {
    let ctx = TestContext::new();

    // Write several sessions to produce multiple blocks.
    for i in 0..5 {
        let sid = TestContext::unique_session_id();
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid).expect("set");
        chat.notify_session_update(None, Some(&format!("Session {i}")), None)
            .expect("notify");
        let msg = format!("Message {i}");
        let content = msg.as_bytes();
        let rc = unsafe {
            talu_sys::talu_responses_append_message(
                chat.responses().as_ptr(),
                MessageRole::User as u8,
                content.as_ptr(),
                content.len(),
            )
        };
        assert!(rc >= 0);
        drop(chat);
    }

    let data_file = chat_dir(ctx.db_path()).join("current.talu");
    let data = std::fs::read(&data_file).expect("read current.talu");

    // BlockHeader is 64 bytes (extern struct, C layout).
    // Field offsets (from types.zig):
    //   magic:     u32 @ 0
    //   version:   u16 @ 4
    //   header_len:u16 @ 6
    //   flags:     u16 @ 8
    //   schema_id: u16 @ 10
    //   row_count: u32 @ 12
    //   block_len: u32 @ 16
    const HEADER_SIZE: usize = 64;
    const MAGIC_TALU: u32 = 0x554C4154;
    const BLOCK_LEN_OFFSET: usize = 16;

    let mut offset = 0usize;
    let mut block_count = 0usize;

    while offset + HEADER_SIZE <= data.len() {
        let magic = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        assert_eq!(
            magic, MAGIC_TALU,
            "Block {block_count} at offset {offset}: bad magic 0x{magic:08X}, expected 0x{MAGIC_TALU:08X}",
        );

        let block_len = u32::from_le_bytes(
            data[offset + BLOCK_LEN_OFFSET..offset + BLOCK_LEN_OFFSET + 4]
                .try_into()
                .unwrap(),
        ) as usize;
        assert!(
            block_len >= HEADER_SIZE,
            "Block {block_count}: block_len {block_len} < header size {HEADER_SIZE}",
        );

        offset += block_len;
        block_count += 1;
    }

    assert_eq!(
        offset,
        data.len(),
        "File has {remaining} trailing bytes after {block_count} blocks (file={total})",
        remaining = data.len() - offset,
        total = data.len(),
    );
    assert!(
        block_count >= 5,
        "Expected at least 5 blocks from 5 sessions, got {block_count}",
    );
}

/// WAL is empty (truncated) after clean shutdown.
///
/// When a ChatHandle is dropped, it flushes pending data to blocks and
/// truncates the WAL to zero. A subsequent read should find an empty WAL.
#[test]
fn wal_empty_after_clean_shutdown() {
    let ctx = TestContext::new();
    let sid = TestContext::unique_session_id();

    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid).expect("set");
        chat.notify_session_update(None, Some("WAL test"), None)
            .expect("notify");
        let msg = b"WAL payload";
        let rc = unsafe {
            talu_sys::talu_responses_append_message(
                chat.responses().as_ptr(),
                MessageRole::User as u8,
                msg.as_ptr(),
                msg.len(),
            )
        };
        assert!(rc >= 0);
        // Drop flushes and truncates WAL.
    }

    // With per-writer WALs, clean close deletes the WAL file.
    // No orphaned wal-*.wal files should remain.
    let wals = find_wal_files(ctx.db_path(), "chat");
    assert!(
        wals.is_empty(),
        "No WAL files should remain after clean shutdown, found: {:?}",
        wals,
    );
}
