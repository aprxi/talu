//! Crash recovery tests.
//!
//! Validates that data written to WAL but NOT flushed to blocks
//! survives a simulated process crash.
//!
//! The write path is:
//!   appendRow → WAL write (fsync) → in-memory buffer accumulation
//!   → auto-flush or deinit → block materialization → WAL truncation
//!
//! If the process crashes after WAL write but before flush, the WAL
//! contains unflushed rows. On next open, `Writer.open` replays the
//! WAL under lock, recovering all pending rows into the in-memory
//! buffer. A clean close then flushes them to blocks.
//!
//! Crash simulation: `simulate_crash()` closes all file descriptors
//! (releasing flocks) without flushing or deleting the WAL file. This
//! accurately simulates what the OS does when a process dies: locks are
//! released, but files remain on disk.

use crate::capi::db::common::{find_wal_files, total_wal_size, TestContext};
use talu::responses::{MessageRole, ResponsesView};
use talu::vector::VectorStore;
use talu::{ChatHandle, StorageHandle};

/// Append a user message via raw FFI.
fn append_message(chat: &ChatHandle, content: &[u8]) {
    let rc = unsafe {
        talu_sys::talu_responses_append_message(
            chat.responses().as_ptr(),
            MessageRole::User as u8,
            content.as_ptr(),
            content.len(),
        )
    };
    assert!(rc >= 0, "append_message failed: {rc}");
}

// ---------------------------------------------------------------------------
// Chat: WAL replay after simulated crash
// ---------------------------------------------------------------------------

/// Data written to WAL survives a simulated crash (no flush/deinit).
///
/// Strategy:
///   1. Open ChatHandle with storage.
///   2. Append a message (goes to WAL + in-memory buffer).
///   3. `simulate_crash()` — releases fds/flocks without flush or WAL delete.
///   4. Verify WAL file is non-empty (data was persisted to WAL).
///   5. Open a NEW ChatHandle on the same DB+session.
///      Writer.open replays WAL → rows recovered in memory.
///   6. Drop the new handle (flush occurs).
///   7. Verify message via StorageHandle.load_conversation.
#[test]
fn chat_wal_replay_after_crash() {
    let ctx = TestContext::new();
    let sid = TestContext::unique_session_id();

    // Phase 1: Write data, then "crash".
    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid).expect("set");

        append_message(&chat, b"Unflushed message from crash");

        // Simulate crash: close fds/flocks, leave WAL on disk.
        chat.simulate_crash().expect("simulate_crash");
    }

    // Phase 2: Verify WAL has data (crash left unflushed rows).
    // With per-writer WALs, the orphaned file is named wal-<hex>.wal.
    let wals = find_wal_files(ctx.db_path(), "chat");
    assert!(
        !wals.is_empty(),
        "Orphaned WAL file should exist after crash",
    );
    let wal_size = total_wal_size(ctx.db_path(), "chat");
    assert!(
        wal_size > 0,
        "WAL should be non-empty after crash (unflushed data), got {wal_size} bytes",
    );

    // Phase 3: Reopen — Writer.open replays WAL, recovering the row.
    // Drop triggers flush → data materialized in blocks.
    {
        let chat = ChatHandle::new(None).expect("new after crash");
        chat.set_storage_db(ctx.db_path(), &sid)
            .expect("set after crash");
        // Drop triggers deinit → flushBlock → blocks written, WAL truncated.
    }

    // Phase 4: Verify data survived the crash via read-only path.
    let storage = StorageHandle::open(ctx.db_path()).expect("open storage");
    let conv = storage.load_conversation(&sid).expect("load conversation");

    assert_eq!(
        conv.item_count(),
        1,
        "WAL replay should recover the unflushed message",
    );
    assert_eq!(
        conv.message_text(0).expect("message text"),
        "Unflushed message from crash",
    );
}

// ---------------------------------------------------------------------------
// Chat: Multiple messages survive crash
// ---------------------------------------------------------------------------

/// Multiple messages written before crash are all recovered from WAL.
///
/// This tests that WAL replay handles multiple rows correctly,
/// not just a single row.
#[test]
fn chat_multiple_messages_survive_crash() {
    let ctx = TestContext::new();
    let sid = TestContext::unique_session_id();

    let message_count = 5;

    // Phase 1: Write multiple messages, then crash.
    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid).expect("set");

        for i in 0..message_count {
            let msg = format!("Crash msg {i}");
            append_message(&chat, msg.as_bytes());
        }

        chat.simulate_crash().expect("simulate_crash");
    }

    // Phase 2: Reopen (WAL replay), then clean close.
    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid).expect("set");
        // Drop → flush
    }

    // Phase 3: Verify all messages recovered.
    let storage = StorageHandle::open(ctx.db_path()).expect("open");
    let conv = storage.load_conversation(&sid).expect("load");

    assert_eq!(
        conv.item_count(),
        message_count,
        "All {message_count} messages should survive WAL replay after crash",
    );

    for i in 0..message_count {
        let expected = format!("Crash msg {i}");
        let actual = conv.message_text(i).expect("message text");
        assert_eq!(
            actual, expected,
            "Message {i} content mismatch after WAL replay",
        );
    }
}

// ---------------------------------------------------------------------------
// Vector: WAL replay after simulated crash
// ---------------------------------------------------------------------------

/// Vector data written to WAL survives a simulated crash.
///
/// Same strategy as chat crash test but using VectorStore.
#[test]
fn vector_wal_replay_after_crash() {
    let ctx = TestContext::new();
    let dims: u32 = 4;

    // Phase 1: Write vectors, then crash.
    {
        let mut store = VectorStore::open(ctx.db_path()).expect("open");
        store
            .append(&[1, 2], &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dims)
            .expect("append");

        store.simulate_crash();
    }

    // Phase 2: Verify WAL has data.
    let wals = find_wal_files(ctx.db_path(), "vector");
    assert!(
        !wals.is_empty(),
        "Orphaned vector WAL should exist after crash",
    );
    let wal_size = total_wal_size(ctx.db_path(), "vector");
    assert!(
        wal_size > 0,
        "Vector WAL should be non-empty after crash, got {wal_size} bytes",
    );

    // Phase 3: Reopen (WAL replayed) → load → verify.
    {
        let store = VectorStore::open(ctx.db_path()).expect("reopen");
        let loaded = store.load().expect("load");

        assert_eq!(
            loaded.ids.len(),
            2,
            "Both vectors should survive WAL replay after crash",
        );

        let mut ids = loaded.ids.clone();
        ids.sort();
        assert_eq!(ids, vec![1, 2], "Vector IDs should match");
    }
}

// ---------------------------------------------------------------------------
// Mixed: Flushed + unflushed data coexists after crash
// ---------------------------------------------------------------------------

/// Data split across flushed blocks and unflushed WAL both survive.
///
/// Phase 1: Write batch A (enough for auto-flush → blocks on disk).
/// Phase 2: Write batch B (small, stays in WAL buffer).
/// Phase 3: Crash (simulate_crash) → batch B is WAL-only.
/// Phase 4: Reopen → WAL replay recovers batch B; batch A in blocks.
/// Phase 5: Verify all data present.
#[test]
fn flushed_and_unflushed_coexist_after_crash() {
    let ctx = TestContext::new();
    let sid = TestContext::unique_session_id();

    // Phase 1+2+3: Write large batch (triggers auto-flush) + small batch, then crash.
    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid).expect("set");

        // Write enough to trigger auto-flush (>64KB).
        // ~1200 bytes per message × 60 messages = ~72KB → auto-flush.
        for i in 0..60 {
            let msg = format!("Flushed msg {i}: {}", "F".repeat(1100));
            append_message(&chat, msg.as_bytes());
        }

        // Write a small message that stays in WAL (not enough for another flush).
        append_message(&chat, b"Unflushed tail message");

        chat.simulate_crash().expect("simulate_crash");
    }

    // Phase 4: Reopen (WAL replay recovers unflushed tail).
    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid).expect("set");
        // Drop → flush remaining
    }

    // Phase 5: Verify ALL messages present.
    let storage = StorageHandle::open(ctx.db_path()).expect("open");
    let conv = storage.load_conversation(&sid).expect("load");

    // 60 flushed + 1 unflushed = 61 total.
    assert_eq!(
        conv.item_count(),
        61,
        "Expected 61 messages (60 flushed + 1 WAL-only), got {}",
        conv.item_count(),
    );

    // Verify the unflushed tail message survived.
    let last = conv.message_text(conv.item_count() - 1).expect("last msg");
    assert_eq!(
        last, "Unflushed tail message",
        "WAL-only tail message should survive crash recovery",
    );
}
