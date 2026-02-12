//! Session metadata persistence and update tests.
//!
//! Validates that notify_session_update correctly persists and overwrites
//! session metadata (title, model, marker) to the Schema 4 session record.

use crate::capi::db::common::TestContext;
use talu::responses::ResponsesView;
use talu::{ChatHandle, StorageHandle};

/// Metadata fields survive close → reopen of StorageHandle.
#[test]
fn metadata_survives_storage_reopen() {
    let ctx = TestContext::new();
    let session_id = TestContext::unique_session_id();

    {
        let chat = ChatHandle::new(None).expect("ChatHandle::new failed");
        chat.set_storage_db(ctx.db_path(), &session_id)
            .expect("set_storage_db failed");
        chat.notify_session_update(Some("qwen3-0.6b"), Some("Durable Title"), Some("active"))
            .expect("notify failed");
    }

    // Open StorageHandle, verify, close, reopen, verify again.
    {
        let storage = StorageHandle::open(ctx.db_path()).expect("open 1");
        let s = storage.get_session(&session_id).expect("get 1");
        assert_eq!(s.title.as_deref(), Some("Durable Title"));
    }
    {
        let storage = StorageHandle::open(ctx.db_path()).expect("open 2");
        let s = storage.get_session(&session_id).expect("get 2");
        assert_eq!(s.title.as_deref(), Some("Durable Title"));
        assert_eq!(s.model.as_deref(), Some("qwen3-0.6b"));
        assert_eq!(s.marker.as_deref(), Some("active"));
    }
}

/// Metadata updates across separate ChatHandle sessions use last-write-wins.
///
/// Core scans blocks newest-first; each block contains a PutSession row.
/// Closing and reopening the ChatHandle forces a flush, producing separate
/// blocks. The newest block's metadata is the one returned by get_session.
#[test]
fn metadata_last_write_wins() {
    let ctx = TestContext::new();
    let session_id = TestContext::unique_session_id();

    // First session: set initial metadata, then close (flushes block).
    {
        let chat = ChatHandle::new(None).expect("new failed");
        chat.set_storage_db(ctx.db_path(), &session_id)
            .expect("set_storage_db failed");
        chat.notify_session_update(Some("model-v1"), Some("Title A"), Some("active"))
            .expect("notify 1");
    }

    // Second session: overwrite metadata, then close (flushes new block).
    {
        let chat = ChatHandle::new(None).expect("new failed");
        chat.set_storage_db(ctx.db_path(), &session_id)
            .expect("set_storage_db failed");
        chat.notify_session_update(Some("model-v2"), Some("Title B"), Some("done"))
            .expect("notify 2");
    }

    let storage = StorageHandle::open(ctx.db_path()).expect("open");
    let s = storage.get_session(&session_id).expect("get");

    assert_eq!(s.title.as_deref(), Some("Title B"));
    assert_eq!(s.model.as_deref(), Some("model-v2"));
    assert_eq!(s.marker.as_deref(), Some("done"));
}

/// notifySessionUpdate is a full-replace: None fields become null.
///
/// Core writes a complete PutSession record on every call. There is no
/// read-modify-write merge. Passing None for a field means null in the
/// record, and the newest record (newest block) wins. Callers that need
/// partial updates must read-then-write at the application layer.
#[test]
fn full_replace_semantics() {
    let ctx = TestContext::new();
    let session_id = TestContext::unique_session_id();

    // First session: set all fields.
    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &session_id)
            .expect("set");
        chat.notify_session_update(Some("initial-model"), Some("Initial Title"), Some("active"))
            .expect("notify 1");
    }

    // Second session: set only title (model and status are None → null).
    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &session_id)
            .expect("set");
        chat.notify_session_update(None, Some("Updated Title"), None)
            .expect("notify 2");
    }

    let storage = StorageHandle::open(ctx.db_path()).expect("open");
    let s = storage.get_session(&session_id).expect("get");

    // Title comes from the newest block (second session).
    assert_eq!(s.title.as_deref(), Some("Updated Title"));
    // Model is null in the newest record — full-replace, not merge.
    assert_eq!(
        s.model.as_deref(),
        None,
        "Full-replace: None fields become null, not preserved from prior record",
    );
    assert_eq!(
        s.marker.as_deref(),
        None,
        "Full-replace: None fields become null",
    );
}

/// Session with metadata but no messages still appears in list.
#[test]
fn metadata_only_session_listed() {
    let ctx = TestContext::new();
    let session_id = TestContext::unique_session_id();

    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &session_id)
            .expect("set");
        chat.notify_session_update(Some("model"), Some("Empty Chat"), Some("active"))
            .expect("notify");
        // No messages appended
    }

    let storage = StorageHandle::open(ctx.db_path()).expect("open");
    let sessions = storage.list_sessions(Some(100)).expect("list");
    assert!(
        sessions.iter().any(|s| s.session_id == session_id),
        "Session with metadata-only should appear in list",
    );

    // load_conversation should succeed but return empty.
    let conv = storage.load_conversation(&session_id).expect("load");
    assert_eq!(conv.item_count(), 0, "No messages = empty conversation");
}

/// Timestamps: created_at and updated_at are non-zero after notify.
#[test]
fn timestamps_populated() {
    let ctx = TestContext::new();
    let session_id = TestContext::unique_session_id();

    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &session_id)
            .expect("set");
        chat.notify_session_update(None, Some("Timestamped"), None)
            .expect("notify");
    }

    let storage = StorageHandle::open(ctx.db_path()).expect("open");
    let s = storage.get_session(&session_id).expect("get");

    assert!(
        s.created_at > 0,
        "created_at should be populated, got {}",
        s.created_at,
    );
    assert!(
        s.updated_at > 0,
        "updated_at should be populated, got {}",
        s.updated_at,
    );
    assert!(
        s.updated_at >= s.created_at,
        "updated_at ({}) should be >= created_at ({})",
        s.updated_at,
        s.created_at,
    );
}
