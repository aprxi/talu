//! Concurrent reader/writer tests.
//!
//! Validates that a reader (StorageHandle) can safely query the DB
//! while a writer (ChatHandle) is actively appending data.
//!
//! Design constraint (from AGENTS.md):
//!   "Tests can't use timing as synchronization for correctness."
//!   We use barriers and channels — not sleep/polling — to coordinate
//!   writer and reader threads deterministically.

use std::sync::mpsc;
use std::sync::{Arc, Barrier};

use crate::capi::db::common::TestContext;
use talu::responses::{MessageRole, ResponsesView};
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
// Reader sees data from completed writer sessions
// ---------------------------------------------------------------------------

/// A reader thread sees progressively more data as writers complete.
///
/// Strategy:
///   Main thread creates 5 sequential writer sessions, each adding
///   one message. After each session closes (flush completes), a
///   reader verifies the accumulated message count.
///
/// This tests the reader's ability to see data from completed writes
/// without requiring a reader restart.
#[test]
fn reader_sees_completed_writes() {
    let ctx = TestContext::new();
    let sid = TestContext::unique_session_id();
    let total_rounds = 5;

    for round in 0..total_rounds {
        // Writer: append one message and close (flush).
        {
            let chat = ChatHandle::new(None).expect("new");
            chat.set_storage_db(ctx.db_path(), &sid).expect("set");
            let msg = format!("Round {round} message");
            append_message(&chat, msg.as_bytes());
            // Drop → flush
        }

        // Reader: verify accumulated count.
        let storage = StorageHandle::open(ctx.db_path()).expect("open");
        let conv = storage.load_session(&sid).expect("load");
        assert_eq!(
            conv.item_count(),
            round + 1,
            "After round {round}, reader should see {} messages, got {}",
            round + 1,
            conv.item_count(),
        );
    }
}

// ---------------------------------------------------------------------------
// Writer and reader threads interleave safely
// ---------------------------------------------------------------------------

/// A writer thread and reader thread operate concurrently on the same DB.
///
/// Strategy:
///   Writer thread: opens/writes/closes ChatHandle in N rounds.
///   After each round, signals completion via channel.
///   Reader thread: waits for each signal, then reads and verifies.
///
/// This ensures:
///   - No file-handle contention between writer and reader.
///   - Reader always sees a consistent snapshot after writer closes.
///   - No data corruption from concurrent file access.
#[test]
fn writer_and_reader_threads_interleave() {
    let ctx = TestContext::new();
    let db_path = ctx.db_path().to_string();
    let sid = TestContext::unique_session_id();
    let rounds = 5;

    let (tx, rx) = mpsc::channel::<usize>();

    // Writer thread: opens, writes, closes, signals.
    let writer_path = db_path.clone();
    let writer_sid = sid.clone();
    let writer = std::thread::spawn(move || {
        for round in 0..rounds {
            let chat = ChatHandle::new(None).expect("writer: new");
            chat.set_storage_db(&writer_path, &writer_sid)
                .expect("writer: set");
            let msg = format!("Thread msg {round}");
            append_message(&chat, msg.as_bytes());
            drop(chat); // Flush before signaling.
            tx.send(round + 1).expect("writer: send");
        }
    });

    // Reader thread: waits for signal, reads, verifies.
    let reader_path = db_path.clone();
    let reader_sid = sid.clone();
    let reader = std::thread::spawn(move || {
        for _ in 0..rounds {
            let expected_count = rx.recv().expect("reader: recv");
            let storage = StorageHandle::open(&reader_path).expect("reader: open");
            let conv = storage.load_session(&reader_sid).expect("reader: load");
            assert_eq!(
                conv.item_count(),
                expected_count,
                "Reader expected {expected_count} messages, got {}",
                conv.item_count(),
            );
        }
    });

    writer.join().expect("writer panicked");
    reader.join().expect("reader panicked");
}

// ---------------------------------------------------------------------------
// Multiple concurrent writers + final reader
// ---------------------------------------------------------------------------

/// Multiple writer threads write to separate sessions simultaneously.
/// A final reader verifies all sessions are intact and isolated.
///
/// This tests namespace-level concurrency: multiple ChatHandle instances
/// writing to the same "chat" namespace directory but different sessions.
#[test]
fn parallel_writers_different_sessions() {
    let ctx = TestContext::new();
    let db_path = ctx.db_path().to_string();
    let num_writers = 4;
    let messages_per_writer = 5;

    let barrier = Arc::new(Barrier::new(num_writers));

    let session_ids: Vec<String> = (0..num_writers)
        .map(|_| TestContext::unique_session_id())
        .collect();

    let handles: Vec<_> = (0..num_writers)
        .map(|i| {
            let path = db_path.clone();
            let sid = session_ids[i].clone();
            let bar = barrier.clone();

            std::thread::spawn(move || {
                bar.wait(); // Maximize concurrency.
                let chat = ChatHandle::new(None).expect("new");
                chat.set_storage_db(&path, &sid).expect("set");

                for j in 0..messages_per_writer {
                    let msg = format!("Writer {i} msg {j}");
                    append_message(&chat, msg.as_bytes());
                }
                // Drop → flush
            })
        })
        .collect();

    for h in handles {
        h.join().expect("writer thread panicked");
    }

    // Verify each session independently.
    let storage = StorageHandle::open(ctx.db_path()).expect("open");
    for (i, sid) in session_ids.iter().enumerate() {
        let conv = storage.load_session(sid).expect("load");
        assert_eq!(
            conv.item_count(),
            messages_per_writer,
            "Writer {i}: expected {messages_per_writer} messages, got {}",
            conv.item_count(),
        );

        // Verify first and last message content.
        let first = conv.message_text(0).expect("first text");
        assert!(
            first.starts_with(&format!("Writer {i} msg 0")),
            "Writer {i}: first message mismatch: '{first}'",
        );
        let last = conv
            .message_text(messages_per_writer - 1)
            .expect("last text");
        assert!(
            last.starts_with(&format!("Writer {i} msg {}", messages_per_writer - 1)),
            "Writer {i}: last message mismatch: '{last}'",
        );
    }
}

// ---------------------------------------------------------------------------
// Concurrent readers don't interfere
// ---------------------------------------------------------------------------

/// Multiple reader threads reading the same session concurrently
/// all get consistent, identical results.
#[test]
fn concurrent_readers_consistent() {
    let ctx = TestContext::new();
    let sid = TestContext::unique_session_id();
    let message_count = 10;

    // Write data first.
    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid).expect("set");
        for i in 0..message_count {
            let msg = format!("Msg {i}");
            append_message(&chat, msg.as_bytes());
        }
    }

    // Spawn N reader threads that all read the same session.
    let db_path = ctx.db_path().to_string();
    let num_readers = 8;
    let barrier = Arc::new(Barrier::new(num_readers));

    let handles: Vec<_> = (0..num_readers)
        .map(|thread_id| {
            let path = db_path.clone();
            let session = sid.clone();
            let bar = barrier.clone();

            std::thread::spawn(move || {
                bar.wait(); // Maximize read contention.
                let storage = StorageHandle::open(&path).expect("open");
                let conv = storage.load_session(&session).expect("load");

                assert_eq!(
                    conv.item_count(),
                    message_count,
                    "Thread {thread_id}: expected {message_count} items, got {}",
                    conv.item_count(),
                );

                // Verify content integrity.
                for i in 0..message_count {
                    let text = conv.message_text(i).expect("text");
                    assert_eq!(
                        text,
                        format!("Msg {i}"),
                        "Thread {thread_id}: message {i} corrupt",
                    );
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("reader thread panicked");
    }
}
