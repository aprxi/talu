//! Schema interleaving tests.
//!
//! The TaluDB writer uses a block builder that accumulates rows for a
//! single schema. When the schema changes, the builder must flush the
//! pending block before starting a new one (via `resetSchema`).
//!
//! The chat adapter uses multiple schemas within a single session:
//!   - Schema 3: chat items (messages)
//!   - Schema 4: session metadata (PutSession)
//!   - Schema 2: deletion markers (ClearItems)
//!
//! These tests verify that interleaving different schema writes within
//! a single ChatHandle lifetime doesn't corrupt the block builder or
//! lose data.

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
// Messages + Metadata interleaved
// ---------------------------------------------------------------------------

/// Interleaving messages (Schema 3) and metadata (Schema 4) within
/// a single ChatHandle doesn't lose either.
///
/// Write order: Msg → Metadata → Msg → Metadata
/// Verify: Both messages and final metadata persist.
#[test]
fn interleaved_messages_and_metadata() {
    let ctx = TestContext::new();
    let sid = TestContext::unique_session_id();

    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid).expect("set");

        // Schema 3: message
        append_message(&chat, b"First question");

        // Schema 4: metadata
        chat.notify_session_update(Some("model-v1"), Some("Title A"), Some("active"))
            .expect("notify 1");

        // Schema 3: message (forces schema switch back)
        append_message(&chat, b"Second question");

        // Schema 4: metadata (another switch)
        chat.notify_session_update(Some("model-v2"), Some("Title B"), Some("done"))
            .expect("notify 2");
    }

    // Verify messages.
    let storage = StorageHandle::open(ctx.db_path()).expect("open");
    let conv = storage.load_conversation(&sid).expect("load");
    assert_eq!(
        conv.item_count(),
        2,
        "Both messages should survive schema interleaving",
    );
    assert_eq!(conv.message_text(0).unwrap(), "First question");
    assert_eq!(conv.message_text(1).unwrap(), "Second question");

    // Verify metadata (last-write-wins).
    let session = storage.get_session(&sid).expect("get session");
    assert_eq!(session.title.as_deref(), Some("Title B"));
    assert_eq!(session.model.as_deref(), Some("model-v2"));
    assert_eq!(session.marker.as_deref(), Some("done"));
}

// ---------------------------------------------------------------------------
// Many rapid schema switches
// ---------------------------------------------------------------------------

/// Rapidly alternating between messages and metadata doesn't corrupt data.
///
/// 20 cycles of (message + metadata update), verifying all 20 messages
/// and the final metadata state.
#[test]
fn rapid_schema_alternation() {
    let ctx = TestContext::new();
    let sid = TestContext::unique_session_id();

    let cycles = 20;

    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid).expect("set");

        for i in 0..cycles {
            // Schema 3: message
            let msg = format!("Cycle {i} message");
            append_message(&chat, msg.as_bytes());

            // Schema 4: metadata
            let title = format!("Title after cycle {i}");
            chat.notify_session_update(Some("model"), Some(&title), Some("active"))
                .expect("notify");
        }
    }

    let storage = StorageHandle::open(ctx.db_path()).expect("open");
    let conv = storage.load_conversation(&sid).expect("load");

    assert_eq!(
        conv.item_count(),
        cycles,
        "All {cycles} messages should survive rapid schema alternation",
    );

    for i in 0..cycles {
        let expected = format!("Cycle {i} message");
        let actual = conv.message_text(i).expect("text");
        assert_eq!(actual, expected, "Message {i} content mismatch");
    }

    // Final metadata should be from the last cycle.
    let session = storage.get_session(&sid).expect("get");
    assert_eq!(
        session.title.as_deref(),
        Some(format!("Title after cycle {}", cycles - 1).as_str()),
    );
}

// ---------------------------------------------------------------------------
// Messages + Deletion + Messages
// ---------------------------------------------------------------------------

/// Deletion marker (Schema 2) interleaved with messages doesn't corrupt
/// subsequent writes.
///
/// Session A: write messages → close.
/// Delete session A (writes Schema 2 tombstone + Schema 4 delete marker).
/// Session B (same DB): write messages → verify B is intact.
///
/// This tests that the delete operation's schema writes don't
/// interfere with subsequent message writes in a different session.
#[test]
fn deletion_interleaved_with_writes() {
    let ctx = TestContext::new();
    let sid_a = TestContext::unique_session_id();
    let sid_b = TestContext::unique_session_id();

    // Session A: write messages.
    {
        let chat = ChatHandle::new(None).expect("new A");
        chat.set_storage_db(ctx.db_path(), &sid_a).expect("set A");
        append_message(&chat, b"Session A message 1");
        append_message(&chat, b"Session A message 2");
    }

    // Delete session A (writes Schema 2 + Schema 4).
    {
        let storage = StorageHandle::open(ctx.db_path()).expect("open");
        storage.delete_session(&sid_a).expect("delete A");
    }

    // Session B: write messages after deletion occurred.
    {
        let chat = ChatHandle::new(None).expect("new B");
        chat.set_storage_db(ctx.db_path(), &sid_b).expect("set B");
        append_message(&chat, b"Session B message 1");
        append_message(&chat, b"Session B message 2");
        append_message(&chat, b"Session B message 3");
    }

    // Verify session B is intact (not corrupted by A's deletion).
    let storage = StorageHandle::open(ctx.db_path()).expect("open");
    let conv_b = storage.load_conversation(&sid_b).expect("load B");
    assert_eq!(
        conv_b.item_count(),
        3,
        "Session B should have 3 messages after deletion of A",
    );
    assert_eq!(conv_b.message_text(0).unwrap(), "Session B message 1");
    assert_eq!(conv_b.message_text(1).unwrap(), "Session B message 2");
    assert_eq!(conv_b.message_text(2).unwrap(), "Session B message 3");

    // Session A should be deleted.
    assert!(
        storage.get_session(&sid_a).is_err(),
        "Session A should be deleted",
    );
}

// ---------------------------------------------------------------------------
// All three schemas in one session lifetime
// ---------------------------------------------------------------------------

/// Messages, metadata, and deletion all happen within a short window,
/// exercising all three schema types.
///
/// 1. Session X: messages + metadata.
/// 2. Session Y: messages.
/// 3. Delete session X.
/// 4. Session Y: more messages.
/// 5. Verify Y is intact, X is deleted.
#[test]
fn all_schemas_in_sequence() {
    let ctx = TestContext::new();
    let sid_x = TestContext::unique_session_id();
    let sid_y = TestContext::unique_session_id();

    // Step 1: Session X with messages + metadata.
    {
        let chat = ChatHandle::new(None).expect("new X");
        chat.set_storage_db(ctx.db_path(), &sid_x).expect("set X");
        append_message(&chat, b"X msg 1");
        chat.notify_session_update(Some("model-x"), Some("Session X"), Some("active"))
            .expect("notify X");
        append_message(&chat, b"X msg 2");
    }

    // Step 2: Session Y with messages.
    {
        let chat = ChatHandle::new(None).expect("new Y");
        chat.set_storage_db(ctx.db_path(), &sid_y).expect("set Y");
        append_message(&chat, b"Y msg 1");
    }

    // Step 3: Delete session X (Schema 2 + 4).
    {
        let storage = StorageHandle::open(ctx.db_path()).expect("open for delete");
        storage.delete_session(&sid_x).expect("delete X");
    }

    // Step 4: Session Y continues.
    {
        let chat = ChatHandle::new(None).expect("new Y2");
        chat.set_storage_db(ctx.db_path(), &sid_y).expect("set Y2");
        append_message(&chat, b"Y msg 2");
        chat.notify_session_update(Some("model-y"), Some("Session Y"), Some("done"))
            .expect("notify Y");
    }

    // Step 5: Verify.
    let storage = StorageHandle::open(ctx.db_path()).expect("open final");

    // X is deleted.
    assert!(storage.get_session(&sid_x).is_err(), "X should be deleted");

    // Y is intact with all messages.
    let conv_y = storage.load_conversation(&sid_y).expect("load Y");
    assert_eq!(conv_y.item_count(), 2, "Y should have 2 messages");
    assert_eq!(conv_y.message_text(0).unwrap(), "Y msg 1");
    assert_eq!(conv_y.message_text(1).unwrap(), "Y msg 2");

    let session_y = storage.get_session(&sid_y).expect("get Y");
    assert_eq!(session_y.title.as_deref(), Some("Session Y"));
    assert_eq!(session_y.marker.as_deref(), Some("done"));
}
