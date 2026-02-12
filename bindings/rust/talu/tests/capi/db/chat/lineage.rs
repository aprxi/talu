//! C API tests for session lineage tracking via source_doc_id.
//!
//! Tests:
//! - source_doc_id filter in talu_storage_list_sessions_ex
//! - talu_storage_list_sessions_by_source convenience function
//! - talu_chat_inherit_tags for copying document tags to conversations

use crate::capi::db::common::TestContext;
use std::ffi::CString;
use std::ptr;
use talu::ChatHandle;
use talu_sys::CSessionList;

/// Helper to seed a session with source_doc_id via notify_session_update.
fn seed_session_with_source(
    ctx: &TestContext,
    session_id: &str,
    title: &str,
    source_doc_id: Option<&str>,
) {
    let chat = ChatHandle::new(None).expect("ChatHandle::new failed");
    chat.set_storage_db(ctx.db_path(), session_id)
        .expect("set_storage_db failed");

    let c_model = CString::new("test-model").expect("model cstr");
    let c_title = CString::new(title).expect("title cstr");
    let c_marker = CString::new("active").expect("marker cstr");
    let c_source_doc_id = source_doc_id.map(|s| CString::new(s).expect("source_doc_id cstr"));

    let rc = unsafe {
        talu_sys::talu_chat_notify_session_update(
            chat.as_ptr(),
            c_model.as_ptr(),
            c_title.as_ptr(),
            ptr::null(), // system_prompt
            ptr::null(), // config_json
            c_marker.as_ptr(),
            ptr::null(), // parent_session_id
            ptr::null(), // group_id
            ptr::null(), // metadata_json
            c_source_doc_id.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
        )
    };
    assert_eq!(rc, 0, "notify_session_update with source_doc_id failed");
}

/// Helper to list sessions filtered by source_doc_id using list_sessions_ex.
fn list_sessions_by_source_ex(db_path: &str, source_doc_id: &str) -> Vec<String> {
    let c_db_path = CString::new(db_path).expect("db_path cstr");
    let c_source_doc_id = CString::new(source_doc_id).expect("source_doc_id cstr");

    let mut c_list: *mut CSessionList = ptr::null_mut();

    let result = unsafe {
        talu_sys::talu_storage_list_sessions_ex(
            c_db_path.as_ptr(),
            0,                        // limit
            0,                        // before_updated_at_ms
            ptr::null(),              // before_session_id
            ptr::null(),              // group_id
            ptr::null(),              // search_query
            ptr::null(),              // tags_filter
            ptr::null(),              // tags_filter_any
            ptr::null(),              // marker_filter
            ptr::null(),              // marker_filter_any
            ptr::null(),              // model_filter
            0,                        // created_after_ms
            0,                        // created_before_ms
            0,                        // updated_after_ms
            0,                        // updated_before_ms
            -1,                       // has_tags (-1 = don't filter)
            c_source_doc_id.as_ptr(), // source_doc_id filter
            &mut c_list as *mut _ as *mut std::ffi::c_void,
        )
    };

    assert_eq!(result, 0, "talu_storage_list_sessions_ex failed");

    extract_session_ids(c_list)
}

/// Helper to list sessions using the by_source convenience function.
fn list_sessions_by_source(db_path: &str, source_doc_id: &str) -> Vec<String> {
    let c_db_path = CString::new(db_path).expect("db_path cstr");
    let c_source_doc_id = CString::new(source_doc_id).expect("source_doc_id cstr");

    let mut c_list: *mut CSessionList = ptr::null_mut();

    let result = unsafe {
        talu_sys::talu_storage_list_sessions_by_source(
            c_db_path.as_ptr(),
            c_source_doc_id.as_ptr(),
            0,           // no limit
            0,           // no cursor timestamp
            ptr::null(), // no cursor session_id
            &mut c_list as *mut _ as *mut std::ffi::c_void,
        )
    };

    assert_eq!(result, 0, "talu_storage_list_sessions_by_source failed");

    extract_session_ids(c_list)
}

/// Helper to extract session IDs from CSessionList and free it.
fn extract_session_ids(c_list: *mut CSessionList) -> Vec<String> {
    let mut ids = Vec::new();
    if !c_list.is_null() {
        let list = unsafe { &*c_list };
        if !list.sessions.is_null() && list.count > 0 {
            for i in 0..list.count {
                let record = unsafe { &*list.sessions.add(i) };
                if !record.session_id.is_null() {
                    let id = unsafe { std::ffi::CStr::from_ptr(record.session_id) }
                        .to_string_lossy()
                        .to_string();
                    ids.push(id);
                }
            }
        }
        unsafe { talu_sys::talu_storage_free_sessions(c_list) };
    }
    ids
}

// ---------------------------------------------------------------------------
// source_doc_id filter tests (via list_sessions_ex)
// ---------------------------------------------------------------------------

/// Filter sessions by source_doc_id using list_sessions_ex.
#[test]
fn capi_source_doc_id_filter_basic() {
    let ctx = TestContext::new();

    // Seed sessions from different documents
    seed_session_with_source(&ctx, "sess-doc1-a", "From Doc 1 A", Some("doc_001"));
    seed_session_with_source(&ctx, "sess-doc1-b", "From Doc 1 B", Some("doc_001"));
    seed_session_with_source(&ctx, "sess-doc2", "From Doc 2", Some("doc_002"));
    seed_session_with_source(&ctx, "sess-no-doc", "No Document", None);

    // Filter by doc_001 - should return 2 sessions
    let ids = list_sessions_by_source_ex(ctx.db_path(), "doc_001");
    assert_eq!(ids.len(), 2, "doc_001 should have 2 sessions");
    assert!(ids.contains(&"sess-doc1-a".to_string()));
    assert!(ids.contains(&"sess-doc1-b".to_string()));

    // Filter by doc_002 - should return 1 session
    let ids = list_sessions_by_source_ex(ctx.db_path(), "doc_002");
    assert_eq!(ids.len(), 1, "doc_002 should have 1 session");
    assert!(ids.contains(&"sess-doc2".to_string()));
}

/// Filter by nonexistent source_doc_id returns empty.
#[test]
fn capi_source_doc_id_filter_no_match() {
    let ctx = TestContext::new();

    seed_session_with_source(&ctx, "sess-a", "Session A", Some("doc_001"));

    let ids = list_sessions_by_source_ex(ctx.db_path(), "doc_nonexistent");
    assert!(ids.is_empty(), "nonexistent doc should return empty");
}

// ---------------------------------------------------------------------------
// list_sessions_by_source convenience function tests
// ---------------------------------------------------------------------------

/// Basic test for list_sessions_by_source.
#[test]
fn capi_list_sessions_by_source_basic() {
    let ctx = TestContext::new();

    seed_session_with_source(&ctx, "sess-src-a", "Source A", Some("prompt_doc_1"));
    seed_session_with_source(&ctx, "sess-src-b", "Source B", Some("prompt_doc_1"));
    seed_session_with_source(&ctx, "sess-other", "Other Source", Some("prompt_doc_2"));

    let ids = list_sessions_by_source(ctx.db_path(), "prompt_doc_1");
    assert_eq!(ids.len(), 2, "prompt_doc_1 should have 2 sessions");
    assert!(ids.contains(&"sess-src-a".to_string()));
    assert!(ids.contains(&"sess-src-b".to_string()));
}

/// Null source_doc_id returns error.
#[test]
fn capi_list_sessions_by_source_null_source() {
    let ctx = TestContext::new();
    let c_db_path = CString::new(ctx.db_path()).expect("db_path cstr");

    let mut c_list: *mut CSessionList = ptr::null_mut();

    let result = unsafe {
        talu_sys::talu_storage_list_sessions_by_source(
            c_db_path.as_ptr(),
            ptr::null(), // null source_doc_id
            0,
            0,
            ptr::null(),
            &mut c_list as *mut _ as *mut std::ffi::c_void,
        )
    };

    assert_ne!(result, 0, "null source_doc_id should return error");
}

// ---------------------------------------------------------------------------
// talu_chat_inherit_tags tests
// ---------------------------------------------------------------------------

/// Create document with tags, then inherit tags to chat session.
#[test]
fn capi_inherit_tags_basic() {
    use talu::documents::DocumentsHandle;
    use talu::StorageHandle;

    let ctx = TestContext::new();

    // Create tags first via StorageHandle
    let storage = StorageHandle::open(ctx.db_path()).expect("open storage");
    storage
        .create_tag(&talu::storage::TagCreate {
            tag_id: "rust".to_string(),
            name: "Rust".to_string(),
            color: None,
            description: None,
            group_id: None,
        })
        .expect("create rust tag");
    storage
        .create_tag(&talu::storage::TagCreate {
            tag_id: "tutorial".to_string(),
            name: "Tutorial".to_string(),
            color: None,
            description: None,
            group_id: None,
        })
        .expect("create tutorial tag");
    drop(storage);

    // Create a document via DocumentsHandle
    let docs = DocumentsHandle::open(ctx.db_path()).expect("open docs");
    let doc_json = r#"{"system_prompt": "You are a Rust expert."}"#;
    docs.create(
        "doc_rust_tutorial",
        "prompt",
        "Rust Tutorial",
        doc_json,
        None, // tags_text
        None, // parent_id
        None, // marker
        None, // group_id
        None, // owner_id
    )
    .expect("create document");

    // Add tags to document via proper API
    docs.add_tag("doc_rust_tutorial", "rust", None)
        .expect("add rust tag to doc");
    docs.add_tag("doc_rust_tutorial", "tutorial", None)
        .expect("add tutorial tag to doc");
    drop(docs);

    // Create chat and set prompt_id
    let chat = ChatHandle::new(None).expect("ChatHandle::new");
    chat.set_storage_db(ctx.db_path(), "sess-inherit-test")
        .expect("set_storage_db");

    let c_prompt_id = CString::new("doc_rust_tutorial").expect("CString");
    let rc = unsafe { talu_sys::talu_chat_set_prompt_id(chat.as_ptr(), c_prompt_id.as_ptr()) };
    assert_eq!(rc, 0, "set_prompt_id failed");

    // Inherit tags from document
    let c_db_path = CString::new(ctx.db_path()).expect("db_path cstr");
    let rc = unsafe { talu_sys::talu_chat_inherit_tags(chat.as_ptr(), c_db_path.as_ptr()) };
    assert_eq!(rc, 0, "inherit_tags failed");

    // Notify session to persist
    chat.notify_session_update(
        Some("test-model"),
        Some("Inherited Tags Session"),
        Some("active"),
    )
    .expect("notify");
    drop(chat);

    // Verify tags were inherited
    let storage = StorageHandle::open(ctx.db_path()).expect("reopen storage");
    let tags = storage
        .get_conversation_tags("sess-inherit-test")
        .expect("get_conversation_tags");

    assert!(tags.contains(&"rust".to_string()), "should have rust tag");
    assert!(
        tags.contains(&"tutorial".to_string()),
        "should have tutorial tag"
    );
}

/// Inherit tags with no prompt_id set returns error.
#[test]
fn capi_inherit_tags_no_prompt_id() {
    let ctx = TestContext::new();

    let chat = ChatHandle::new(None).expect("ChatHandle::new");
    chat.set_storage_db(ctx.db_path(), "sess-no-prompt")
        .expect("set_storage_db");

    let c_db_path = CString::new(ctx.db_path()).expect("db_path cstr");
    let rc = unsafe { talu_sys::talu_chat_inherit_tags(chat.as_ptr(), c_db_path.as_ptr()) };
    assert_ne!(rc, 0, "inherit_tags without prompt_id should return error");
}

/// Inherit tags with null chat handle returns error.
#[test]
fn capi_inherit_tags_null_handle() {
    let ctx = TestContext::new();
    let c_db_path = CString::new(ctx.db_path()).expect("db_path cstr");

    let rc = unsafe { talu_sys::talu_chat_inherit_tags(ptr::null_mut(), c_db_path.as_ptr()) };
    assert_ne!(rc, 0, "inherit_tags with null handle should return error");
}

/// Inherit tags with null db_path returns error.
#[test]
fn capi_inherit_tags_null_db_path() {
    let chat = ChatHandle::new(None).expect("ChatHandle::new");

    let c_prompt_id = CString::new("some_doc").expect("CString");
    let rc = unsafe { talu_sys::talu_chat_set_prompt_id(chat.as_ptr(), c_prompt_id.as_ptr()) };
    assert_eq!(rc, 0);

    let rc = unsafe { talu_sys::talu_chat_inherit_tags(chat.as_ptr(), ptr::null()) };
    assert_ne!(rc, 0, "inherit_tags with null db_path should return error");
}

/// Inherit tags from document that doesn't exist succeeds with no tags added.
/// This is by design - a document with no tags (or nonexistent) just results in
/// no tags being inherited, not an error. This is useful for optional prompt_id.
#[test]
fn capi_inherit_tags_document_not_found() {
    use talu::StorageHandle;

    let ctx = TestContext::new();

    let chat = ChatHandle::new(None).expect("ChatHandle::new");
    chat.set_storage_db(ctx.db_path(), "sess-missing-doc")
        .expect("set_storage_db");

    let c_prompt_id = CString::new("doc_nonexistent_12345").expect("CString");
    let rc = unsafe { talu_sys::talu_chat_set_prompt_id(chat.as_ptr(), c_prompt_id.as_ptr()) };
    assert_eq!(rc, 0);

    let c_db_path = CString::new(ctx.db_path()).expect("db_path cstr");
    let rc = unsafe { talu_sys::talu_chat_inherit_tags(chat.as_ptr(), c_db_path.as_ptr()) };
    // Succeeds because missing document just means no tags to inherit
    assert_eq!(
        rc, 0,
        "inherit_tags with missing document should succeed (no tags to inherit)"
    );

    // Notify session to persist
    chat.notify_session_update(
        Some("test-model"),
        Some("Missing Doc Session"),
        Some("active"),
    )
    .expect("notify");
    drop(chat);

    // Verify no tags were added
    let storage = StorageHandle::open(ctx.db_path()).expect("reopen storage");
    let tags = storage
        .get_conversation_tags("sess-missing-doc")
        .expect("get_conversation_tags");
    assert!(
        tags.is_empty(),
        "should have no tags when document doesn't exist"
    );
}

// ---------------------------------------------------------------------------
// Integration test: prompt_id -> source_doc_id -> lineage query
// ---------------------------------------------------------------------------

/// End-to-end test: create document, create sessions with source_doc_id,
/// verify source_doc_id is stored and queryable.
#[test]
fn capi_lineage_end_to_end() {
    use talu::documents::DocumentsHandle;

    let ctx = TestContext::new();

    // Create a prompt document
    let docs = DocumentsHandle::open(ctx.db_path()).expect("open docs");
    let doc_json = r#"{"system_prompt": "You are a helpful assistant."}"#;
    docs.create(
        "prompt_assistant",
        "prompt",
        "AI Assistant",
        doc_json,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("create document");
    drop(docs);

    // Create multiple sessions referencing this document
    for i in 0..3 {
        let session_id = format!("lineage-sess-{}", i);
        seed_session_with_source(
            &ctx,
            &session_id,
            &format!("Session {}", i),
            Some("prompt_assistant"),
        );
    }

    // Create session without source
    seed_session_with_source(&ctx, "lineage-sess-orphan", "Orphan Session", None);

    // Query sessions by source document
    let linked_sessions = list_sessions_by_source(ctx.db_path(), "prompt_assistant");
    assert_eq!(linked_sessions.len(), 3, "should have 3 linked sessions");

    for i in 0..3 {
        assert!(
            linked_sessions.contains(&format!("lineage-sess-{}", i)),
            "should contain lineage-sess-{}",
            i
        );
    }
    assert!(
        !linked_sessions.contains(&"lineage-sess-orphan".to_string()),
        "should not contain orphan session"
    );
}
