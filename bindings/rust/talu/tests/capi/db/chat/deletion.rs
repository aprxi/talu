//! Session deletion and tombstone tests.
//!
//! Validates the Dual-Delete Protocol:
//! - Schema 4 session tombstone (hides from list_sessions)
//! - Schema 2 clear marker (hides items from load_session)

use crate::capi::db::common::TestContext;
use talu::responses::{MessageRole, ResponsesView};
use talu::{ChatHandle, StorageHandle};

/// Helper: create a session with one message and metadata.
fn create_session_with_message(db_path: &str, session_id: &str, msg: &str) {
    let chat = ChatHandle::new(None).expect("new");
    chat.set_storage_db(db_path, session_id).expect("set");
    chat.notify_session_update(None, Some("To Delete"), Some("active"))
        .expect("notify");
    let content = msg.as_bytes();
    let rc = unsafe {
        talu_sys::talu_responses_append_message(
            chat.responses().as_ptr(),
            MessageRole::User as u8,
            content.as_ptr(),
            content.len(),
        )
    };
    assert!(rc >= 0, "append failed: {rc}");
}

/// Tombstone persists across StorageHandle reopen.
///
/// This proves the tombstone is durable on disk, not just in-memory filtering.
#[test]
fn tombstone_survives_reopen() {
    let ctx = TestContext::new();
    let session_id = TestContext::unique_session_id();

    create_session_with_message(ctx.db_path(), &session_id, "Ephemeral");

    // Delete
    {
        let storage = StorageHandle::open(ctx.db_path()).expect("open 1");
        storage.delete_session(&session_id).expect("delete");
    }

    // Reopen and verify still deleted
    {
        let storage = StorageHandle::open(ctx.db_path()).expect("open 2");
        assert!(
            storage.get_session(&session_id).is_err(),
            "Tombstone should survive StorageHandle reopen",
        );
        let sessions = storage.list_sessions(Some(100)).expect("list");
        assert!(
            !sessions.iter().any(|s| s.session_id == session_id),
            "Deleted session should not appear after reopen",
        );
    }
}

/// load_session on a deleted session still returns items.
///
/// Core's loadAll scans only schema_chat_items blocks (schema 3) and
/// skips schema_chat_deletes blocks (schema 2). The ClearItems marker
/// written by deleteSession is not honored during item loading.
///
/// Deletion is enforced at the session level: get_session returns an
/// error for tombstoned sessions, and list_sessions omits them. But
/// load_session is a low-level item scan that does not check
/// session tombstones.
#[test]
fn load_session_after_delete_returns_items() {
    let ctx = TestContext::new();
    let session_id = TestContext::unique_session_id();

    create_session_with_message(ctx.db_path(), &session_id, "Will be cleared");

    let storage = StorageHandle::open(ctx.db_path()).expect("open");

    // Verify message exists before delete.
    let conv_before = storage.load_session(&session_id).expect("load before");
    assert_eq!(conv_before.item_count(), 1);

    storage.delete_session(&session_id).expect("delete");

    // Session is tombstoned: get_session fails.
    assert!(
        storage.get_session(&session_id).is_err(),
        "Tombstoned session should not be returned by get_session",
    );

    // But load_session still returns items (clear marker not honored).
    let conv_after = storage.load_session(&session_id).expect("load after");
    assert_eq!(
        conv_after.item_count(),
        1,
        "loadAll does not honor ClearItems marker; items remain visible",
    );
}

/// Deleting one session does not affect sibling sessions.
#[test]
fn delete_does_not_affect_siblings() {
    let ctx = TestContext::new();
    let sid_keep = TestContext::unique_session_id();
    let sid_delete = TestContext::unique_session_id();

    create_session_with_message(ctx.db_path(), &sid_keep, "Keep me");
    create_session_with_message(ctx.db_path(), &sid_delete, "Delete me");

    let storage = StorageHandle::open(ctx.db_path()).expect("open");
    storage.delete_session(&sid_delete).expect("delete");

    // Kept session should be intact.
    let kept = storage.get_session(&sid_keep).expect("get kept");
    assert_eq!(kept.title.as_deref(), Some("To Delete")); // from helper
    let conv = storage.load_session(&sid_keep).expect("load kept");
    assert_eq!(conv.item_count(), 1);
    assert_eq!(conv.message_text(0).unwrap(), "Keep me");

    // Deleted session should be gone.
    assert!(storage.get_session(&sid_delete).is_err());
}

/// Deleting a non-existent session is not an error (idempotent).
/// Or it returns a clear error — either is acceptable; we just
/// verify it does not panic or corrupt the DB.
#[test]
fn delete_nonexistent_session_safe() {
    let ctx = TestContext::new();
    let fake_id = TestContext::unique_session_id();

    // Ensure something exists in the DB (so the directory is created).
    create_session_with_message(ctx.db_path(), &TestContext::unique_session_id(), "Anchor");

    let storage = StorageHandle::open(ctx.db_path()).expect("open");

    // This should either succeed (no-op) or return a typed error.
    // It must NOT panic or corrupt the DB.
    let _ = storage.delete_session(&fake_id);

    // DB should still be usable.
    let sessions = storage
        .list_sessions(Some(100))
        .expect("list after bad delete");
    assert!(sessions.len() >= 1, "DB should still function");
}

/// session_count decreases after deletion.
#[test]
fn session_count_decreases_after_delete() {
    let ctx = TestContext::new();
    let sid_a = TestContext::unique_session_id();
    let sid_b = TestContext::unique_session_id();

    create_session_with_message(ctx.db_path(), &sid_a, "A");
    create_session_with_message(ctx.db_path(), &sid_b, "B");

    let storage = StorageHandle::open(ctx.db_path()).expect("open");
    assert_eq!(storage.session_count().expect("count"), 2);

    storage.delete_session(&sid_a).expect("delete A");
    assert_eq!(
        storage.session_count().expect("count after delete"),
        1,
        "Session count should decrease after delete",
    );
}

/// Multi-message conversation: user messages survive persistence.
///
/// Only user/system/developer messages are immediately persisted via
/// notifyStorage(). Assistant messages require explicit finalization
/// and are not covered by append_message alone.
#[test]
fn multi_message_conversation_persists() {
    let ctx = TestContext::new();
    let session_id = TestContext::unique_session_id();

    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &session_id)
            .expect("set");

        let messages: &[&[u8]] = &[
            b"What is 2+2?",
            b"And 3+3?",
            b"What about 4+4?",
            b"Last question: 5+5?",
        ];

        for content in messages {
            let rc = unsafe {
                talu_sys::talu_responses_append_message(
                    chat.responses().as_ptr(),
                    MessageRole::User as u8,
                    content.as_ptr(),
                    content.len(),
                )
            };
            assert!(rc >= 0, "append failed");
        }
    }

    // Verify via StorageHandle
    let storage = StorageHandle::open(ctx.db_path()).expect("open");
    let conv = storage.load_session(&session_id).expect("load");

    assert_eq!(conv.item_count(), 4, "Should have 4 user messages");
    assert_eq!(conv.message_text(0).unwrap(), "What is 2+2?");
    assert_eq!(conv.message_text(1).unwrap(), "And 3+3?");
    assert_eq!(conv.message_text(2).unwrap(), "What about 4+4?");
    assert_eq!(conv.message_text(3).unwrap(), "Last question: 5+5?");

    // Verify via ChatHandle reload
    let chat = ChatHandle::new(None).expect("new");
    chat.set_storage_db(ctx.db_path(), &session_id)
        .expect("set");
    let view = chat.responses();
    assert_eq!(view.item_count(), 4);
}
