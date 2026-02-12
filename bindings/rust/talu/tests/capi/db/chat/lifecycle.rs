//! Chat session lifecycle tests.
//!
//! Validates: create -> persist -> close -> reopen -> verify.

use crate::capi::db::common::TestContext;
use talu::responses::{MessageRole, ResponsesView};
use talu::{ChatHandle, StorageHandle};

/// A new ChatHandle with storage_db creates a session visible to StorageHandle.
#[test]
fn session_appears_after_notify() {
    let ctx = TestContext::new();
    let session_id = TestContext::unique_session_id();

    // Phase 1: Create session and persist metadata
    {
        let chat =
            ChatHandle::new(Some("You are a test assistant.")).expect("ChatHandle::new failed");
        chat.set_storage_db(ctx.db_path(), &session_id)
            .expect("set_storage_db failed");
        chat.notify_session_update(Some("test-model"), Some("Test Session"), Some("active"))
            .expect("notify_session_update failed");
        // ChatHandle dropped here -> storage backend flushed
    }

    // Phase 2: Verify session is visible via StorageHandle
    let storage = StorageHandle::open(ctx.db_path()).expect("StorageHandle::open failed");

    let sessions = storage
        .list_sessions(Some(100))
        .expect("list_sessions failed");
    assert!(
        sessions.iter().any(|s| s.session_id == session_id),
        "Session '{}' not found in session list ({} sessions total)",
        session_id,
        sessions.len(),
    );

    let session = storage
        .get_session(&session_id)
        .expect("get_session failed");
    assert_eq!(session.session_id, session_id);
    assert_eq!(session.title.as_deref(), Some("Test Session"));
    assert_eq!(session.model.as_deref(), Some("test-model"));
    assert_eq!(session.marker.as_deref(), Some("active"));
}

/// Setting storage_db on a ChatHandle loads previously persisted items.
///
/// This is the core persistence lifecycle:
///   1. Create ChatHandle with storage -> append message via FFI -> drop
///   2. Create new ChatHandle with same storage -> verify items loaded
#[test]
fn message_persists_across_sessions() {
    let ctx = TestContext::new();
    let session_id = TestContext::unique_session_id();

    // Phase 1: Create a session and append a user message
    {
        let chat = ChatHandle::new(None).expect("ChatHandle::new failed");
        chat.set_storage_db(ctx.db_path(), &session_id)
            .expect("set_storage_db failed");

        // Append a user message via talu_sys (the safe API does not expose
        // direct append on ChatHandle's borrowed conversation; messages are
        // normally added via the generation pipeline).
        let content = b"Hello Persistence";
        let rc = unsafe {
            talu_sys::talu_responses_append_message(
                chat.responses().as_ptr(),
                MessageRole::User as u8,
                content.as_ptr(),
                content.len(),
            )
        };
        assert!(rc >= 0, "talu_responses_append_message failed: {rc}");

        // Verify item is in memory
        let view = chat.responses();
        assert_eq!(view.item_count(), 1);
        assert_eq!(view.message_text(0).unwrap(), "Hello Persistence",);
        // ChatHandle dropped -> storage backend flushed & freed
    }

    // Phase 2: Reopen with a new ChatHandle pointing to the same session
    {
        let chat = ChatHandle::new(None).expect("ChatHandle::new failed (reopen)");
        chat.set_storage_db(ctx.db_path(), &session_id)
            .expect("set_storage_db failed (reopen)");

        let view = chat.responses();
        assert_eq!(
            view.item_count(),
            1,
            "Expected 1 item after reload, got {}",
            view.item_count(),
        );
        assert_eq!(view.message_text(0).unwrap(), "Hello Persistence",);
    }
}

/// Verify StorageHandle::load_conversation returns persisted items.
#[test]
fn load_conversation_returns_items() {
    let ctx = TestContext::new();
    let session_id = TestContext::unique_session_id();

    // Phase 1: Persist a message
    {
        let chat = ChatHandle::new(None).expect("ChatHandle::new failed");
        chat.set_storage_db(ctx.db_path(), &session_id)
            .expect("set_storage_db failed");

        let content = b"Stored via ChatHandle";
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

    // Phase 2: Load via StorageHandle (independent read path)
    let storage = StorageHandle::open(ctx.db_path()).expect("StorageHandle::open failed");
    let conv = storage
        .load_conversation(&session_id)
        .expect("load_conversation failed");

    assert_eq!(conv.item_count(), 1);
    assert_eq!(conv.message_text(0).unwrap(), "Stored via ChatHandle",);
}

/// Deleting a session removes it from the listing and clears items.
#[test]
fn delete_session_removes_data() {
    let ctx = TestContext::new();
    let session_id = TestContext::unique_session_id();

    // Phase 1: Create and persist
    {
        let chat = ChatHandle::new(None).expect("ChatHandle::new failed");
        chat.set_storage_db(ctx.db_path(), &session_id)
            .expect("set_storage_db failed");
        chat.notify_session_update(Some("model"), Some("To Be Deleted"), Some("active"))
            .expect("notify failed");

        let content = b"Delete me";
        let rc = unsafe {
            talu_sys::talu_responses_append_message(
                chat.responses().as_ptr(),
                MessageRole::User as u8,
                content.as_ptr(),
                content.len(),
            )
        };
        assert!(rc >= 0);
    }

    // Phase 2: Verify exists, then delete
    let storage = StorageHandle::open(ctx.db_path()).expect("open failed");
    assert!(
        storage.get_session(&session_id).is_ok(),
        "Session should exist before delete",
    );

    storage
        .delete_session(&session_id)
        .expect("delete_session failed");

    // Phase 3: Verify gone
    assert!(
        storage.get_session(&session_id).is_err(),
        "Session should not be found after delete",
    );

    let sessions = storage
        .list_sessions(Some(100))
        .expect("list_sessions failed");
    assert!(
        !sessions.iter().any(|s| s.session_id == session_id),
        "Deleted session should not appear in list",
    );
}

/// Multiple sessions in the same DB are isolated.
#[test]
fn multiple_sessions_isolated() {
    let ctx = TestContext::new();
    let sid_a = TestContext::unique_session_id();
    let sid_b = TestContext::unique_session_id();

    // Create session A with one message
    {
        let chat = ChatHandle::new(None).unwrap();
        chat.set_storage_db(ctx.db_path(), &sid_a).unwrap();
        chat.notify_session_update(None, Some("Session A"), None)
            .unwrap();
        let msg = b"Message A";
        unsafe {
            talu_sys::talu_responses_append_message(
                chat.responses().as_ptr(),
                MessageRole::User as u8,
                msg.as_ptr(),
                msg.len(),
            );
        }
    }

    // Create session B with a different message
    {
        let chat = ChatHandle::new(None).unwrap();
        chat.set_storage_db(ctx.db_path(), &sid_b).unwrap();
        chat.notify_session_update(None, Some("Session B"), None)
            .unwrap();
        let msg = b"Message B";
        unsafe {
            talu_sys::talu_responses_append_message(
                chat.responses().as_ptr(),
                MessageRole::User as u8,
                msg.as_ptr(),
                msg.len(),
            );
        }
    }

    // Verify isolation
    let storage = StorageHandle::open(ctx.db_path()).unwrap();

    let conv_a = storage.load_conversation(&sid_a).unwrap();
    assert_eq!(conv_a.item_count(), 1);
    assert_eq!(conv_a.message_text(0).unwrap(), "Message A");

    let conv_b = storage.load_conversation(&sid_b).unwrap();
    assert_eq!(conv_b.item_count(), 1);
    assert_eq!(conv_b.message_text(0).unwrap(), "Message B");

    let sessions = storage.list_sessions(Some(100)).unwrap();
    assert_eq!(sessions.len(), 2);
}
