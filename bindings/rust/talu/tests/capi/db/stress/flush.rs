//! Auto-flush threshold tests.
//!
//! The TaluDB writer buffers columnar data in memory and WAL. When the
//! accumulated buffer exceeds 64KB (`buffer_bytes >= 64 * 1024`), the
//! writer auto-flushes: materializing an on-disk block in `current.talu`
//! and truncating the WAL to zero.
//!
//! These tests verify that auto-flush occurs at the expected threshold
//! while the handle is still open (not triggered by close/drop).

use std::path::PathBuf;

use crate::capi::db::common::{total_wal_size, TestContext};
use talu::responses::{MessageRole, ResponsesView};
use talu::ChatHandle;

/// Resolve the chat namespace directory for a DB root.
fn chat_dir(db_root: &str) -> PathBuf {
    PathBuf::from(db_root).join("chat")
}

/// File size, or 0 if the file does not exist.
fn file_size_or_zero(path: &std::path::Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Append a user message of exactly `len` bytes (filled with 'x').
///
/// Returns the C API return code.
fn append_padding_message(chat: &ChatHandle, len: usize) -> i64 {
    let content: Vec<u8> = vec![b'x'; len];
    unsafe {
        talu_sys::talu_responses_append_message(
            chat.responses().as_ptr(),
            MessageRole::User as u8,
            content.as_ptr(),
            content.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Auto-flush threshold
// ---------------------------------------------------------------------------

/// Auto-flush materializes a block in current.talu when buffered data
/// exceeds 64KB — while the ChatHandle is still open.
///
/// Strategy:
///   1. Open a ChatHandle (keeps writer alive).
///   2. Write small messages totalling well under 64KB.
///   3. Record current.talu size (should be 0 — data only in WAL).
///   4. Write enough additional data to exceed 64KB total.
///   5. current.talu should have grown (auto-flush fired).
///   6. Handle is still open — growth is from auto-flush, not close.
#[test]
fn auto_flush_threshold() {
    let ctx = TestContext::new();
    let sid = TestContext::unique_session_id();

    let chat = ChatHandle::new(None).expect("new");
    chat.set_storage_db(ctx.db_path(), &sid).expect("set");

    let data_file = chat_dir(ctx.db_path()).join("current.talu");

    // Phase 1: Write ~4KB of small messages (well under 64KB threshold).
    // Each message is 1000 bytes; 4 messages ≈ 4KB of payload.
    for _ in 0..4 {
        let rc = append_padding_message(&chat, 1000);
        assert!(rc >= 0, "append failed: {rc}");
    }

    // current.talu should be small or non-existent — data is in WAL only.
    let size_before = file_size_or_zero(&data_file);

    // Phase 2: Write enough to exceed 64KB threshold.
    // We need ~60KB more. Write 60 × 1100-byte messages ≈ 66KB additional.
    // Total payload will be ~70KB, well over the 64KB threshold.
    for _ in 0..60 {
        let rc = append_padding_message(&chat, 1100);
        assert!(rc >= 0, "append failed during large batch");
    }

    // Auto-flush should have fired: current.talu must have grown.
    let size_after = file_size_or_zero(&data_file);
    assert!(
        size_after > size_before,
        "current.talu should grow from auto-flush while handle is open: \
         before={size_before}, after={size_after}",
    );

    // The handle is still alive — this growth came from auto-flush, not drop.
    // Verify WAL was truncated (auto-flush clears the writer's WAL).
    // With per-writer WALs, the WAL file is named wal-<hex>.wal.
    let wal_size = total_wal_size(ctx.db_path(), "chat");
    // WAL may contain residual data from post-flush appends (the last
    // few messages that didn't trigger another flush). But it should be
    // significantly smaller than the total data written.
    assert!(
        wal_size < 64 * 1024,
        "WAL should have been truncated by auto-flush, but is {wal_size} bytes",
    );

    // Drop flushes remaining buffer.
    drop(chat);

    // After close, current.talu may grow further (residual buffer flush).
    let size_final = file_size_or_zero(&data_file);
    assert!(
        size_final >= size_after,
        "current.talu should not shrink after close: after_flush={size_after}, final={size_final}",
    );
}

/// A single row exceeding the flush threshold triggers immediate flush.
///
/// If a single append produces > 64KB of column data, the writer flushes
/// immediately after that row, materializing a block with just that one row.
#[test]
fn single_large_row_triggers_flush() {
    let ctx = TestContext::new();
    let sid = TestContext::unique_session_id();

    let chat = ChatHandle::new(None).expect("new");
    chat.set_storage_db(ctx.db_path(), &sid).expect("set");

    let data_file = chat_dir(ctx.db_path()).join("current.talu");
    let size_before = file_size_or_zero(&data_file);

    // Write a single message larger than 64KB.
    let rc = append_padding_message(&chat, 80_000);
    assert!(rc >= 0, "append 80KB message failed: {rc}");

    // Auto-flush should have fired after this single row.
    let size_after = file_size_or_zero(&data_file);
    assert!(
        size_after > size_before,
        "Single 80KB message should trigger immediate auto-flush: \
         before={size_before}, after={size_after}",
    );

    // Handle still open — growth is from auto-flush.
    drop(chat);
}

/// Multiple flushes accumulate blocks in current.talu.
///
/// Writing data in batches that each exceed 64KB should produce multiple
/// auto-flush events, each appending a new block to current.talu.
#[test]
fn multiple_flush_cycles() {
    let ctx = TestContext::new();
    let sid = TestContext::unique_session_id();

    let chat = ChatHandle::new(None).expect("new");
    chat.set_storage_db(ctx.db_path(), &sid).expect("set");

    let data_file = chat_dir(ctx.db_path()).join("current.talu");
    let mut sizes = Vec::new();
    sizes.push(file_size_or_zero(&data_file));

    // Perform 3 rounds, each writing > 64KB.
    for round in 0..3 {
        for _ in 0..70 {
            let rc = append_padding_message(&chat, 1024);
            assert!(rc >= 0, "append failed in round {round}");
        }
        sizes.push(file_size_or_zero(&data_file));
    }

    // Each round should have caused at least one flush, growing the file.
    for i in 1..sizes.len() {
        assert!(
            sizes[i] > sizes[i - 1],
            "Round {}: current.talu should grow (size[{}]={}, size[{}]={})",
            i - 1,
            i - 1,
            sizes[i - 1],
            i,
            sizes[i],
        );
    }

    drop(chat);
}
