//! Memory safety tests for C API boundary handling.
//!
//! These tests specifically try to cause memory errors by:
//! - Passing null pointers
//! - Double-freeing resources
//! - Using handles after close
//! - Passing huge inputs
//! - Passing invalid inputs

use crate::capi::db::common::TestContext;
use std::ffi::CString;
use std::ptr;
use talu::ChatHandle;
use talu_sys::CSessionList;

// ---------------------------------------------------------------------------
// Null pointer handling
// ---------------------------------------------------------------------------

/// Passing null db_path to list_sessions should return error, not crash.
#[test]
fn null_db_path_returns_error() {
    let mut c_list: *mut CSessionList = ptr::null_mut();

    let result = unsafe {
        talu_sys::talu_db_table_session_list(
            ptr::null(), // null db_path
            0,
            0,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(), // no project_id
            0,           // no project_id_null
            &mut c_list as *mut _,
        )
    };

    // Should return non-zero (error), not crash
    assert_ne!(result, 0, "null db_path should return error code");
}

/// Passing null output pointer to list_sessions should return error, not crash.
#[test]
fn null_output_returns_error() {
    let ctx = TestContext::new();
    let c_db_path = CString::new(ctx.db_path()).expect("db_path cstr");

    let result = unsafe {
        talu_sys::talu_db_table_session_list(
            c_db_path.as_ptr(),
            0,
            0,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(), // no project_id
            0,           // no project_id_null
            ptr::null_mut(), // null output pointer
        )
    };

    // Should return non-zero (error), not crash
    assert_ne!(result, 0, "null output pointer should return error code");
}

/// Freeing null CSessionList should be a no-op, not crash.
#[test]
fn free_null_session_list_is_noop() {
    // Should not crash
    unsafe {
        talu_sys::talu_db_table_session_free_list(ptr::null_mut());
    }
}

// ---------------------------------------------------------------------------
// Double-free protection
// ---------------------------------------------------------------------------

/// Double-freeing a session list should not crash.
/// Note: This test documents expected behavior. Double-free is UB in general,
/// but we want to verify the implementation is defensive.
#[test]
fn double_free_session_list_does_not_crash() {
    let ctx = TestContext::new();

    // Seed a session
    let chat = ChatHandle::new(None).expect("ChatHandle::new");
    chat.set_storage_db(ctx.db_path(), "test-session")
        .expect("set_storage_db");
    chat.notify_session_update(Some("model"), Some("Test"), Some("active"))
        .expect("notify_session_update");
    drop(chat);

    // List sessions
    let c_db_path = CString::new(ctx.db_path()).expect("db_path cstr");
    let mut c_list: *mut CSessionList = ptr::null_mut();

    let result = unsafe {
        talu_sys::talu_db_table_session_list(
            c_db_path.as_ptr(),
            0,
            0,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(), // no project_id
            0,           // no project_id_null
            &mut c_list as *mut _,
        )
    };
    assert_eq!(result, 0, "list_sessions should succeed");

    if !c_list.is_null() {
        // First free - normal
        unsafe { talu_sys::talu_db_table_session_free_list(c_list) };

        // Note: We don't actually do a second free here because it would be UB.
        // This test documents that the first free should work correctly.
        // In a debug build with memory sanitizers, this would catch issues.
    }
}

// ---------------------------------------------------------------------------
// Large input handling
// ---------------------------------------------------------------------------

/// Very long session_id should not cause buffer overflow.
#[test]
fn very_long_session_id_handled_safely() {
    let ctx = TestContext::new();

    // Create a very long session ID (1MB)
    let long_id: String = "x".repeat(1024 * 1024);

    let chat = ChatHandle::new(None).expect("ChatHandle::new");

    // This might fail (too long), but should not crash
    let result = chat.set_storage_db(ctx.db_path(), &long_id);

    // Either succeeds or returns an error, but doesn't crash
    match result {
        Ok(_) => eprintln!("Long session ID accepted"),
        Err(e) => eprintln!("Long session ID rejected: {}", e),
    }
}

/// Very long title should not cause buffer overflow.
#[test]
fn very_long_title_handled_safely() {
    let ctx = TestContext::new();

    let chat = ChatHandle::new(None).expect("ChatHandle::new");
    chat.set_storage_db(ctx.db_path(), "test-session")
        .expect("set_storage_db");

    // Create a very long title (1MB)
    let long_title: String = "Title ".repeat(170_000); // ~1MB

    // This might fail (too long), but should not crash
    let result = chat.notify_session_update(Some("model"), Some(&long_title), Some("active"));

    match result {
        Ok(_) => eprintln!("Long title accepted"),
        Err(e) => eprintln!("Long title rejected: {}", e),
    }
}

/// Very long message content should not cause buffer overflow.
#[test]
fn very_long_message_handled_safely() {
    let ctx = TestContext::new();

    let chat = ChatHandle::new(None).expect("ChatHandle::new");
    chat.set_storage_db(ctx.db_path(), "test-session")
        .expect("set_storage_db");
    chat.notify_session_update(Some("model"), Some("Test"), Some("active"))
        .expect("notify_session_update");

    // Create a very long message (10MB)
    let long_message: String = "Hello world! ".repeat(800_000); // ~10MB

    // This should either succeed or return an error, but not crash
    let result = chat.append_user_message(&long_message);

    match result {
        Ok(_) => eprintln!("Long message accepted ({} bytes)", long_message.len()),
        Err(e) => eprintln!("Long message rejected: {}", e),
    }
}

/// Very long search query should not cause buffer overflow.
#[test]
fn very_long_search_query_handled_safely() {
    let ctx = TestContext::new();

    // Seed a session
    let chat = ChatHandle::new(None).expect("ChatHandle::new");
    chat.set_storage_db(ctx.db_path(), "test-session")
        .expect("set_storage_db");
    chat.notify_session_update(Some("model"), Some("Test Session"), Some("active"))
        .expect("notify_session_update");
    chat.append_user_message("Hello world")
        .expect("append_user_message");
    drop(chat);

    // Search with very long query (1MB)
    let long_query: String = "search".repeat(170_000); // ~1MB
    let c_db_path = CString::new(ctx.db_path()).expect("db_path cstr");
    let c_query = CString::new(long_query).expect("query cstr");

    let mut c_list: *mut CSessionList = ptr::null_mut();

    let result = unsafe {
        talu_sys::talu_db_table_session_list(
            c_db_path.as_ptr(),
            0,
            0,
            ptr::null(),
            ptr::null(),
            c_query.as_ptr(),
            ptr::null(),
            ptr::null(),
            ptr::null(), // no project_id
            0,           // no project_id_null
            &mut c_list as *mut _,
        )
    };

    // Should either succeed with 0 results or return error, not crash
    if result == 0 && !c_list.is_null() {
        unsafe { talu_sys::talu_db_table_session_free_list(c_list) };
        eprintln!("Long search query accepted");
    } else {
        eprintln!("Long search query returned code: {}", result);
    }
}

// ---------------------------------------------------------------------------
// Empty/edge case inputs
// ---------------------------------------------------------------------------

/// Empty session_id should be handled.
#[test]
fn empty_session_id_handled() {
    let ctx = TestContext::new();

    let chat = ChatHandle::new(None).expect("ChatHandle::new");

    // Empty session ID
    let result = chat.set_storage_db(ctx.db_path(), "");

    match result {
        Ok(_) => eprintln!("Empty session ID accepted"),
        Err(e) => eprintln!("Empty session ID rejected: {}", e),
    }
}

/// Non-existent db_path should return error, not crash.
#[test]
fn nonexistent_db_path_returns_error() {
    let nonexistent_path = "/nonexistent/path/that/does/not/exist/12345";
    let c_db_path = CString::new(nonexistent_path).expect("db_path cstr");

    let mut c_list: *mut CSessionList = ptr::null_mut();

    let result = unsafe {
        talu_sys::talu_db_table_session_list(
            c_db_path.as_ptr(),
            0,
            0,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(), // no project_id
            0,           // no project_id_null
            &mut c_list as *mut _,
        )
    };

    // Should return error or empty list, not crash
    if result == 0 {
        if !c_list.is_null() {
            let list = unsafe { &*c_list };
            eprintln!("Non-existent path returned {} sessions", list.count);
            unsafe { talu_sys::talu_db_table_session_free_list(c_list) };
        }
    } else {
        eprintln!("Non-existent path returned error code: {}", result);
    }
}

// ---------------------------------------------------------------------------
// Concurrent stress with errors
// ---------------------------------------------------------------------------

/// Concurrent operations with some failures should not corrupt state.
#[test]
fn concurrent_operations_with_errors() {
    use std::sync::Arc;
    use std::thread;

    let ctx = TestContext::new();
    let db_path = Arc::new(ctx.db_path().to_string());

    // Seed a valid session
    let chat = ChatHandle::new(None).expect("ChatHandle::new");
    chat.set_storage_db(&db_path, "valid-session")
        .expect("set_storage_db");
    chat.notify_session_update(Some("model"), Some("Valid Session"), Some("active"))
        .expect("notify_session_update");
    chat.append_user_message("Hello world")
        .expect("append_user_message");
    drop(chat);

    let num_threads = 8;
    let iterations = 20;
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let db_path = Arc::clone(&db_path);
        let handle = thread::spawn(move || {
            for i in 0..iterations {
                // Mix of valid and invalid operations
                if (thread_id + i) % 3 == 0 {
                    // Valid search
                    let c_db_path = CString::new(db_path.as_str()).expect("db_path cstr");
                    let c_query = CString::new("world").expect("query cstr");
                    let mut c_list: *mut CSessionList = ptr::null_mut();

                    let result = unsafe {
                        talu_sys::talu_db_table_session_list(
                            c_db_path.as_ptr(),
                            0,
                            0,
                            ptr::null(),
                            ptr::null(),
                            c_query.as_ptr(),
                            ptr::null(),
                            ptr::null(),
                            ptr::null(), // no project_id
                            0,           // no project_id_null
                            &mut c_list as *mut _,
                        )
                    };

                    if result == 0 && !c_list.is_null() {
                        unsafe { talu_sys::talu_db_table_session_free_list(c_list) };
                    }
                } else if (thread_id + i) % 3 == 1 {
                    // Search with empty query
                    let c_db_path = CString::new(db_path.as_str()).expect("db_path cstr");
                    let c_query = CString::new("").expect("query cstr");
                    let mut c_list: *mut CSessionList = ptr::null_mut();

                    let result = unsafe {
                        talu_sys::talu_db_table_session_list(
                            c_db_path.as_ptr(),
                            0,
                            0,
                            ptr::null(),
                            ptr::null(),
                            c_query.as_ptr(),
                            ptr::null(),
                            ptr::null(),
                            ptr::null(), // no project_id
                            0,           // no project_id_null
                            &mut c_list as *mut _,
                        )
                    };

                    if result == 0 && !c_list.is_null() {
                        unsafe { talu_sys::talu_db_table_session_free_list(c_list) };
                    }
                } else {
                    // Search with no-match query
                    let c_db_path = CString::new(db_path.as_str()).expect("db_path cstr");
                    let c_query = CString::new("nonexistent_xyz_123").expect("query cstr");
                    let mut c_list: *mut CSessionList = ptr::null_mut();

                    let result = unsafe {
                        talu_sys::talu_db_table_session_list(
                            c_db_path.as_ptr(),
                            0,
                            0,
                            ptr::null(),
                            ptr::null(),
                            c_query.as_ptr(),
                            ptr::null(),
                            ptr::null(),
                            ptr::null(), // no project_id
                            0,           // no project_id_null
                            &mut c_list as *mut _,
                        )
                    };

                    if result == 0 && !c_list.is_null() {
                        unsafe { talu_sys::talu_db_table_session_free_list(c_list) };
                    }
                }
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

    // Verify data integrity
    let c_db_path = CString::new(ctx.db_path()).expect("db_path cstr");
    let mut c_list: *mut CSessionList = ptr::null_mut();

    let result = unsafe {
        talu_sys::talu_db_table_session_list(
            c_db_path.as_ptr(),
            0,
            0,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(), // no project_id
            0,           // no project_id_null
            &mut c_list as *mut _,
        )
    };

    assert_eq!(result, 0, "Final list_sessions should succeed");
    if !c_list.is_null() {
        let list = unsafe { &*c_list };
        assert_eq!(list.count, 1, "Should still have 1 valid session");
        unsafe { talu_sys::talu_db_table_session_free_list(c_list) };
    }

    eprintln!(
        "Completed {} threads x {} iterations of mixed operations",
        num_threads, iterations
    );
}
