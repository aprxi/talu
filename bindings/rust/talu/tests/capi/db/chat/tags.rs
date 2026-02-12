//! C API tests for talu_storage_list_sessions tag filtering.
//!
//! Tests the tags_filter and tags_filter_any parameters added to the C API.

use crate::capi::db::common::TestContext;
use std::ffi::CString;
use std::ptr;
use talu::ChatHandle;
use talu_sys::{CSessionList, CSessionRecord};

/// Helper to seed a session with tags via C API.
fn seed_session_with_tags(ctx: &TestContext, session_id: &str, title: &str, tags: &[&str]) {
    let chat = ChatHandle::new(None).expect("ChatHandle::new failed");
    chat.set_storage_db(ctx.db_path(), session_id)
        .expect("set_storage_db failed");

    // Build metadata_json with tags
    let tags_json: Vec<String> = tags.iter().map(|t| format!("\"{}\"", t)).collect();
    let metadata_json = format!("{{\"tags\":[{}]}}", tags_json.join(","));

    let c_model = CString::new("test-model").expect("model cstr");
    let c_title = CString::new(title).expect("title cstr");
    let c_marker = CString::new("active").expect("marker cstr");
    let c_metadata = CString::new(metadata_json).expect("metadata cstr");

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
            c_metadata.as_ptr(),
            ptr::null(), // source_doc_id
        )
    };
    assert_eq!(rc, 0, "notify_session_update with tags failed");
    // Drop flushes storage
}

/// Helper to list sessions with tag filters via C API.
fn list_sessions_with_tags(
    db_path: &str,
    tags_filter: Option<&str>,
    tags_filter_any: Option<&str>,
) -> Vec<String> {
    let c_db_path = CString::new(db_path).expect("db_path cstr");
    let c_tags_filter = tags_filter.map(|t| CString::new(t).expect("tags_filter cstr"));
    let c_tags_filter_any = tags_filter_any.map(|t| CString::new(t).expect("tags_filter_any cstr"));

    let mut c_list: *mut CSessionList = ptr::null_mut();

    let result = unsafe {
        talu_sys::talu_storage_list_sessions(
            c_db_path.as_ptr(),
            0,           // no limit
            0,           // no cursor
            ptr::null(), // no cursor session_id
            ptr::null(), // no group_id
            ptr::null(), // no search_query
            c_tags_filter.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
            c_tags_filter_any
                .as_ref()
                .map_or(ptr::null(), |c| c.as_ptr()),
            &mut c_list as *mut _ as *mut std::ffi::c_void,
        )
    };

    assert_eq!(result, 0, "talu_storage_list_sessions failed");

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

/// Helper to get tags_text for a session via C API.
fn get_session_tags_text(db_path: &str, session_id: &str) -> Option<String> {
    let c_db_path = CString::new(db_path).expect("db_path cstr");
    let c_session_id = CString::new(session_id).expect("session_id cstr");

    let mut c_record = CSessionRecord::default();

    let result = unsafe {
        talu_sys::talu_storage_get_session_info(
            c_db_path.as_ptr(),
            c_session_id.as_ptr(),
            &mut c_record,
        )
    };

    if result != 0 {
        return None;
    }

    if c_record.tags_text.is_null() {
        None
    } else {
        Some(
            unsafe { std::ffi::CStr::from_ptr(c_record.tags_text) }
                .to_string_lossy()
                .to_string(),
        )
    }
}

// ---------------------------------------------------------------------------
// tags_filter (AND logic) tests
// ---------------------------------------------------------------------------

/// Single tag filter with exact word match.
#[test]
fn capi_tags_filter_single_tag() {
    let ctx = TestContext::new();

    seed_session_with_tags(&ctx, "sess-rust", "Rust Chat", &["rust", "work"]);
    seed_session_with_tags(&ctx, "sess-py", "Python Chat", &["python", "work"]);
    seed_session_with_tags(&ctx, "sess-rusty", "Rusty Tools", &["rusty"]); // "rusty" != "rust"

    let ids = list_sessions_with_tags(ctx.db_path(), Some("rust"), None);
    assert_eq!(ids.len(), 1, "should match only exact 'rust' tag");
    assert!(ids.contains(&"sess-rust".to_string()));
}

/// Multiple tags with AND logic.
#[test]
fn capi_tags_filter_and_logic() {
    let ctx = TestContext::new();

    seed_session_with_tags(&ctx, "sess-both", "Both Tags", &["rust", "python"]);
    seed_session_with_tags(&ctx, "sess-rust-only", "Rust Only", &["rust"]);
    seed_session_with_tags(&ctx, "sess-py-only", "Python Only", &["python"]);

    // Space-separated tags: must have BOTH
    let ids = list_sessions_with_tags(ctx.db_path(), Some("rust python"), None);
    assert_eq!(ids.len(), 1);
    assert!(ids.contains(&"sess-both".to_string()));
}

/// Tag filter is case-insensitive.
#[test]
fn capi_tags_filter_case_insensitive() {
    let ctx = TestContext::new();

    seed_session_with_tags(&ctx, "sess-a", "Chat", &["Rust", "Python"]);

    // Lowercase filter should match uppercase tag
    let ids = list_sessions_with_tags(ctx.db_path(), Some("rust"), None);
    assert_eq!(ids.len(), 1);

    // Uppercase filter should match
    let ids = list_sessions_with_tags(ctx.db_path(), Some("PYTHON"), None);
    assert_eq!(ids.len(), 1);
}

/// Tag filter with no matches returns empty.
#[test]
fn capi_tags_filter_no_match() {
    let ctx = TestContext::new();

    seed_session_with_tags(&ctx, "sess-a", "Chat", &["rust"]);

    let ids = list_sessions_with_tags(ctx.db_path(), Some("java"), None);
    assert!(ids.is_empty());
}

// ---------------------------------------------------------------------------
// tags_filter_any (OR logic) tests
// ---------------------------------------------------------------------------

/// OR logic with multiple tags.
#[test]
fn capi_tags_filter_any_or_logic() {
    let ctx = TestContext::new();

    seed_session_with_tags(&ctx, "sess-rust", "Rust Only", &["rust"]);
    seed_session_with_tags(&ctx, "sess-py", "Python Only", &["python"]);
    seed_session_with_tags(&ctx, "sess-go", "Go Only", &["go"]);

    // Should match "rust" OR "python"
    let ids = list_sessions_with_tags(ctx.db_path(), None, Some("rust python"));
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&"sess-rust".to_string()));
    assert!(ids.contains(&"sess-py".to_string()));
    assert!(!ids.contains(&"sess-go".to_string()));
}

/// Single tag with tag_any.
#[test]
fn capi_tags_filter_any_single() {
    let ctx = TestContext::new();

    seed_session_with_tags(&ctx, "sess-a", "Chat A", &["work"]);
    seed_session_with_tags(&ctx, "sess-b", "Chat B", &["personal"]);

    let ids = list_sessions_with_tags(ctx.db_path(), None, Some("work"));
    assert_eq!(ids.len(), 1);
    assert!(ids.contains(&"sess-a".to_string()));
}

// ---------------------------------------------------------------------------
// tags_text field tests
// ---------------------------------------------------------------------------

/// Session with tags has tags_text populated.
#[test]
fn capi_session_tags_text_populated() {
    let ctx = TestContext::new();

    seed_session_with_tags(&ctx, "sess-t", "Tagged", &["rust", "python", "work"]);

    let tags_text = get_session_tags_text(ctx.db_path(), "sess-t");
    assert!(tags_text.is_some(), "tags_text should be populated");

    let text = tags_text.unwrap();
    assert!(text.contains("rust"), "should contain rust");
    assert!(text.contains("python"), "should contain python");
    assert!(text.contains("work"), "should contain work");
}

/// Session without tags has null tags_text.
#[test]
fn capi_session_tags_text_null_for_untagged() {
    let ctx = TestContext::new();

    // Seed session without tags
    let chat = ChatHandle::new(None).expect("ChatHandle::new");
    chat.set_storage_db(ctx.db_path(), "sess-u")
        .expect("set_storage_db");
    chat.notify_session_update(Some("model"), Some("Untagged"), Some("active"))
        .expect("notify");
    drop(chat);

    let tags_text = get_session_tags_text(ctx.db_path(), "sess-u");
    assert!(
        tags_text.is_none(),
        "untagged session should have null tags_text"
    );
}

// ---------------------------------------------------------------------------
// Combined filters
// ---------------------------------------------------------------------------

/// tags_filter takes precedence over tags_filter_any when both provided.
#[test]
fn capi_tags_filter_takes_precedence() {
    let ctx = TestContext::new();

    seed_session_with_tags(&ctx, "sess-both", "Both", &["rust", "python"]);
    seed_session_with_tags(&ctx, "sess-rust", "Rust", &["rust"]);
    seed_session_with_tags(&ctx, "sess-py", "Python", &["python"]);

    // Both provided: tags_filter (AND) should take precedence
    // tags_filter="rust python" requires BOTH, tags_filter_any="go" would add none
    let ids = list_sessions_with_tags(ctx.db_path(), Some("rust python"), Some("go"));
    assert_eq!(ids.len(), 1, "AND filter should take precedence");
    assert!(ids.contains(&"sess-both".to_string()));
}

/// Null filters return all sessions.
#[test]
fn capi_null_filters_return_all() {
    let ctx = TestContext::new();

    seed_session_with_tags(&ctx, "sess-a", "A", &["rust"]);
    seed_session_with_tags(&ctx, "sess-b", "B", &["python"]);

    // Seed untagged session
    let chat = ChatHandle::new(None).expect("ChatHandle::new");
    chat.set_storage_db(ctx.db_path(), "sess-c")
        .expect("set_storage_db");
    chat.notify_session_update(Some("model"), Some("C"), Some("active"))
        .expect("notify");
    drop(chat);

    let ids = list_sessions_with_tags(ctx.db_path(), None, None);
    assert_eq!(ids.len(), 3, "null filters should return all sessions");
}

// ---------------------------------------------------------------------------
// Stress tests for tag list memory management
// ---------------------------------------------------------------------------

/// Stress test for list_tags and get_conversation_tags.
/// Exercises talu_storage_free_tags and talu_storage_free_string_list
/// through many alloc/free cycles to detect memory corruption.
/// This is a regression test for the fix where these free functions
/// were incorrectly taking struct by value instead of pointer.
#[test]
fn capi_tags_stress_alloc_free() {
    use talu::StorageHandle;

    let ctx = TestContext::new();

    // Create tags using StorageHandle
    let storage = StorageHandle::open(ctx.db_path()).expect("open");
    let tag_names = ["rust", "python", "work", "project"];
    for name in &tag_names {
        storage
            .create_tag(&talu::storage::TagCreate {
                tag_id: name.to_string(),
                name: name.to_string(),
                color: None,
                description: None,
                group_id: None,
            })
            .expect("create_tag");
    }

    // Seed sessions and add tags
    for i in 0..5 {
        let session_id = format!("stress-sess-{}", i);
        let chat = ChatHandle::new(None).expect("ChatHandle::new");
        chat.set_storage_db(ctx.db_path(), &session_id)
            .expect("set_storage_db");
        chat.notify_session_update(Some("test"), Some(&format!("Sess {}", i)), Some("active"))
            .expect("notify");
        drop(chat);

        // Add tags to session
        for tag in &tag_names {
            let _ = storage.add_conversation_tag(&session_id, tag);
        }
    }

    // Rapid list_tags/get_conversation_tags cycles
    for iteration in 0..100 {
        // list_tags exercises talu_storage_free_tags
        let tags = storage.list_tags(None).expect("list_tags failed");
        assert!(
            !tags.is_empty(),
            "iteration {}: should have tags",
            iteration
        );

        // get_conversation_tags exercises talu_storage_free_string_list
        let conv_tags = storage
            .get_conversation_tags("stress-sess-0")
            .expect("get_conversation_tags failed");
        assert!(
            !conv_tags.is_empty(),
            "iteration {}: should have conversation tags",
            iteration
        );

        // get_tag_conversations also exercises talu_storage_free_string_list
        let sessions = storage
            .get_tag_conversations("rust")
            .expect("get_tag_conversations failed");
        assert!(
            !sessions.is_empty(),
            "iteration {}: should have sessions with rust tag",
            iteration
        );
    }

    // If we get here without crash, the fix is working
    eprintln!("Completed 100 iterations of tag list alloc/free cycles");
}

/// Multithreaded stress test for tag list memory management.
#[test]
fn capi_tags_stress_multithreaded() {
    use std::sync::Arc;
    use std::thread;
    use talu::StorageHandle;

    let ctx = TestContext::new();
    let db_path = Arc::new(ctx.db_path().to_string());

    // Create tags using StorageHandle
    let storage = StorageHandle::open(ctx.db_path()).expect("open");
    let tag_names = ["rust", "python", "go", "java"];
    for name in &tag_names {
        storage
            .create_tag(&talu::storage::TagCreate {
                tag_id: name.to_string(),
                name: name.to_string(),
                color: None,
                description: None,
                group_id: None,
            })
            .expect("create_tag");
    }

    // Seed sessions and add tags
    for i in 0..10 {
        let session_id = format!("mt-stress-sess-{}", i);
        let chat = ChatHandle::new(None).expect("ChatHandle::new");
        chat.set_storage_db(ctx.db_path(), &session_id)
            .expect("set_storage_db");
        chat.notify_session_update(
            Some("test"),
            Some(&format!("MT Sess {}", i)),
            Some("active"),
        )
        .expect("notify");
        drop(chat);

        // Add tags to session
        for tag in &tag_names {
            let _ = storage.add_conversation_tag(&session_id, tag);
        }
    }
    drop(storage);

    let num_threads = 8;
    let iterations_per_thread = 25;
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let db_path = Arc::clone(&db_path);
        let handle = thread::spawn(move || {
            let storage = StorageHandle::open(&*db_path).expect("open");

            for i in 0..iterations_per_thread {
                // list_tags
                let tags = storage.list_tags(None).expect("list_tags");
                assert!(!tags.is_empty(), "thread {} iter {}: tags", thread_id, i);

                // get_conversation_tags
                let session_idx = (thread_id + i) % 10;
                let session_id = format!("mt-stress-sess-{}", session_idx);
                let conv_tags = storage
                    .get_conversation_tags(&session_id)
                    .expect("get_conversation_tags");
                assert!(
                    !conv_tags.is_empty(),
                    "thread {} iter {}: conv_tags",
                    thread_id,
                    i
                );

                // get_tag_conversations
                let tag_names = ["rust", "python", "go", "java"];
                let tag = tag_names[(thread_id + i) % tag_names.len()];
                let sessions = storage
                    .get_tag_conversations(tag)
                    .expect("get_tag_conversations");
                assert!(
                    !sessions.is_empty(),
                    "thread {} iter {}: sessions for {}",
                    thread_id,
                    i,
                    tag
                );
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for (i, handle) in handles.into_iter().enumerate() {
        handle
            .join()
            .unwrap_or_else(|e| panic!("Thread {} panicked: {:?}", i, e));
    }

    eprintln!(
        "Completed {} threads x {} iterations = {} concurrent tag operations",
        num_threads,
        iterations_per_thread,
        num_threads * iterations_per_thread
    );
}
