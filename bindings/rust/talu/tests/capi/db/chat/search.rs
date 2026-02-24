//! C API tests for session search functionality.
//!
//! Tests the search_query parameter in talu_db_table_session_list to ensure
//! sessions can be filtered by title/content text search.

use crate::capi::db::common::TestContext;
use std::ffi::CString;
use std::ptr;
use talu::ChatHandle;
use talu_sys::CSessionList;

/// Helper to seed a session with a specific title via C API.
fn seed_session(ctx: &TestContext, session_id: &str, title: &str, message_content: &str) {
    let chat = ChatHandle::new(None).expect("ChatHandle::new failed");
    chat.set_storage_db(ctx.db_path(), session_id)
        .expect("set_storage_db failed");

    chat.notify_session_update(Some("test-model"), Some(title), Some("active"))
        .expect("notify_session_update failed");

    // Append a user message using the safe wrapper
    chat.append_user_message(message_content)
        .expect("append_user_message failed");
    // Drop flushes storage
}

/// Helper to list sessions with a search query via C API.
fn list_sessions_with_search(db_path: &str, search_query: &str) -> Vec<String> {
    let c_db_path = CString::new(db_path).expect("db_path cstr");
    let c_search_query = CString::new(search_query).expect("search_query cstr");

    let mut c_list: *mut CSessionList = ptr::null_mut();

    let result = unsafe {
        talu_sys::talu_db_table_session_list(
            c_db_path.as_ptr(),
            0,                       // no limit
            0,                       // no cursor timestamp
            ptr::null(),             // no cursor session_id
            ptr::null(),             // no group_id
            c_search_query.as_ptr(), // search query
            ptr::null(),             // no tags_filter
            ptr::null(),             // no tags_filter_any
            ptr::null(),             // no project_id
            0,                       // no project_id_null
            &mut c_list as *mut _,
        )
    };

    assert_eq!(
        result, 0,
        "talu_db_table_session_list with search_query failed"
    );

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
        unsafe { talu_sys::talu_db_table_session_free_list(c_list) };
    }

    ids
}

/// Helper to list sessions without any filters via C API.
fn list_all_sessions(db_path: &str) -> Vec<String> {
    let c_db_path = CString::new(db_path).expect("db_path cstr");

    let mut c_list: *mut CSessionList = ptr::null_mut();

    let result = unsafe {
        talu_sys::talu_db_table_session_list(
            c_db_path.as_ptr(),
            0,           // no limit
            0,           // no cursor timestamp
            ptr::null(), // no cursor session_id
            ptr::null(), // no group_id
            ptr::null(), // no search query
            ptr::null(), // no tags_filter
            ptr::null(), // no tags_filter_any
            ptr::null(), // no project_id
            0,           // no project_id_null
            &mut c_list as *mut _,
        )
    };

    assert_eq!(result, 0, "talu_db_table_session_list failed");

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
        unsafe { talu_sys::talu_db_table_session_free_list(c_list) };
    }

    ids
}

/// Search sessions by title - basic matching.
#[test]
fn search_sessions_by_title() {
    let ctx = TestContext::new();

    // Seed sessions with distinct titles
    let session1 = TestContext::unique_session_id();
    let session2 = TestContext::unique_session_id();
    let session3 = TestContext::unique_session_id();

    seed_session(&ctx, &session1, "Rust programming guide", "Hello rust");
    seed_session(&ctx, &session2, "Python basics tutorial", "Hello python");
    seed_session(&ctx, &session3, "Advanced Rust concepts", "More rust");

    // Verify all sessions exist
    let all_ids = list_all_sessions(ctx.db_path());
    assert_eq!(
        all_ids.len(),
        3,
        "Expected 3 sessions, got {}",
        all_ids.len()
    );

    // Search for "Rust" - should match session1 and session3
    let rust_results = list_sessions_with_search(ctx.db_path(), "Rust");
    assert!(
        rust_results.contains(&session1),
        "Search for 'Rust' should find session1"
    );
    assert!(
        rust_results.contains(&session3),
        "Search for 'Rust' should find session3"
    );
    assert!(
        !rust_results.contains(&session2),
        "Search for 'Rust' should not find session2"
    );

    // Search for "Python" - should match only session2
    let python_results = list_sessions_with_search(ctx.db_path(), "Python");
    assert!(
        python_results.contains(&session2),
        "Search for 'Python' should find session2"
    );
    assert_eq!(
        python_results.len(),
        1,
        "Search for 'Python' should find exactly 1 session"
    );
}

/// Search sessions with no matches returns empty.
#[test]
fn search_sessions_no_match() {
    let ctx = TestContext::new();

    let session1 = TestContext::unique_session_id();
    seed_session(&ctx, &session1, "Test session", "Hello world");

    // Search for something not in any title
    let results = list_sessions_with_search(ctx.db_path(), "nonexistent_xyz_123");
    assert!(
        results.is_empty(),
        "Search for nonexistent term should return empty"
    );
}

/// Search with empty query returns all sessions (or is treated as no filter).
#[test]
fn search_sessions_empty_query() {
    let ctx = TestContext::new();

    let session1 = TestContext::unique_session_id();
    let session2 = TestContext::unique_session_id();
    seed_session(&ctx, &session1, "Session one", "Message one");
    seed_session(&ctx, &session2, "Session two", "Message two");

    // Empty search should return all sessions
    let results = list_sessions_with_search(ctx.db_path(), "");
    assert_eq!(
        results.len(),
        2,
        "Empty search should return all 2 sessions"
    );
}

/// Multiple searches in sequence don't corrupt memory.
/// This tests for memory safety issues in the search path.
#[test]
fn search_sessions_multiple_sequential() {
    let ctx = TestContext::new();

    // Seed several sessions
    let mut session_ids = Vec::new();
    for i in 0..5 {
        let id = TestContext::unique_session_id();
        seed_session(
            &ctx,
            &id,
            &format!("Session {} title", i),
            &format!("Message {}", i),
        );
        session_ids.push(id);
    }

    // Run multiple searches to test memory handling
    for i in 0..10 {
        let query = format!("{}", i % 5);
        let results = list_sessions_with_search(ctx.db_path(), &query);
        // Just verify we don't crash - the actual matching behavior
        // depends on the search implementation
        eprintln!("Search {} for '{}': {} results", i, query, results.len());
    }

    // Final sanity check - list all
    let all = list_all_sessions(ctx.db_path());
    assert_eq!(all.len(), 5, "Should still have 5 sessions after searches");
}

/// Search and free cycle stress test.
/// Tests for memory leaks and use-after-free bugs.
#[test]
fn search_sessions_stress_alloc_free() {
    let ctx = TestContext::new();

    // Create sessions with searchable content
    for i in 0..3 {
        let id = TestContext::unique_session_id();
        seed_session(
            &ctx,
            &id,
            &format!("Rust Python JavaScript topic{}", i),
            &format!("Content with words {}", i),
        );
    }

    // Rapid search/free cycles
    for iteration in 0..50 {
        let queries = ["Rust", "Python", "JavaScript", "topic", "nonexistent"];
        for query in queries {
            let c_db_path = CString::new(ctx.db_path()).expect("db_path cstr");
            let c_query = CString::new(query).expect("query cstr");
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

            assert_eq!(
                result, 0,
                "Iteration {}, query '{}': list_sessions failed",
                iteration, query
            );

            if !c_list.is_null() {
                unsafe { talu_sys::talu_db_table_session_free_list(c_list) };
            }
        }
    }

    // If we get here without crashing or memory errors, the test passes
    eprintln!("Completed 50 iterations of 5 search queries each = 250 alloc/free cycles");
}

/// Multithreaded search stress test.
/// Multiple threads doing concurrent searches on the same database.
#[test]
fn search_sessions_multithreaded() {
    use std::sync::Arc;
    use std::thread;

    let ctx = TestContext::new();
    let db_path = Arc::new(ctx.db_path().to_string());

    // Seed sessions
    for i in 0..10 {
        let id = TestContext::unique_session_id();
        seed_session(
            &ctx,
            &id,
            &format!("Thread test session {} Rust Python", i),
            &format!("Content {} with searchable words", i),
        );
    }

    // Verify sessions exist
    let initial_count = list_all_sessions(&db_path).len();
    assert_eq!(initial_count, 10, "Should have 10 sessions");

    // Spawn threads doing concurrent searches
    let num_threads = 8;
    let iterations_per_thread = 20;
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let db_path = Arc::clone(&db_path);
        let handle = thread::spawn(move || {
            let queries = ["Rust", "Python", "session", "Content", "nonexistent"];
            for i in 0..iterations_per_thread {
                let query = queries[(thread_id + i) % queries.len()];
                let results = list_sessions_with_search(&db_path, query);
                // Just verify no crash - results may vary
                eprintln!(
                    "Thread {} iter {}: '{}' -> {} results",
                    thread_id,
                    i,
                    query,
                    results.len()
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

    // Verify data integrity after concurrent access
    let final_count = list_all_sessions(&db_path).len();
    assert_eq!(
        final_count, 10,
        "Should still have 10 sessions after concurrent searches"
    );

    eprintln!(
        "Completed {} threads x {} iterations = {} concurrent search operations",
        num_threads,
        iterations_per_thread,
        num_threads * iterations_per_thread
    );
}

/// Multithreaded seed + search stress test.
/// Some threads seed new sessions while others search.
#[test]
fn search_sessions_concurrent_seed_and_search() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;

    let ctx = TestContext::new();
    let db_path = Arc::new(ctx.db_path().to_string());
    let sessions_created = Arc::new(AtomicUsize::new(0));

    // Seed initial sessions
    for i in 0..5 {
        let id = TestContext::unique_session_id();
        seed_session(
            &ctx,
            &id,
            &format!("Initial session {}", i),
            "Initial content",
        );
    }
    sessions_created.fetch_add(5, Ordering::SeqCst);

    let num_seeders = 4;
    let num_searchers = 4;
    let iterations = 10;
    let mut handles = Vec::new();

    // Seeder threads
    for seeder_id in 0..num_seeders {
        let db_path = Arc::clone(&db_path);
        let counter = Arc::clone(&sessions_created);
        let handle = thread::spawn(move || {
            for i in 0..iterations {
                let session_id = format!("seeder-{}-iter-{}-{}", seeder_id, i, std::process::id());
                let chat = ChatHandle::new(None).expect("ChatHandle::new");
                chat.set_storage_db(&db_path, &session_id)
                    .expect("set_storage_db");
                chat.notify_session_update(
                    Some("test-model"),
                    Some(&format!("Seeder {} session {} Rust Python", seeder_id, i)),
                    Some("active"),
                )
                .expect("notify_session_update");
                chat.append_user_message(&format!(
                    "Message from seeder {} iteration {}",
                    seeder_id, i
                ))
                .expect("append_user_message");
                drop(chat);
                counter.fetch_add(1, Ordering::SeqCst);
                eprintln!("Seeder {} created session {}", seeder_id, i);
            }
        });
        handles.push(handle);
    }

    // Searcher threads
    for searcher_id in 0..num_searchers {
        let db_path = Arc::clone(&db_path);
        let handle = thread::spawn(move || {
            let queries = ["Rust", "Python", "Seeder", "Initial", "nonexistent"];
            for i in 0..iterations {
                let query = queries[(searcher_id + i) % queries.len()];
                let results = list_sessions_with_search(&db_path, query);
                eprintln!(
                    "Searcher {} iter {}: '{}' -> {} results",
                    searcher_id,
                    i,
                    query,
                    results.len()
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

    let expected_count = 5 + (num_seeders * iterations);
    let final_count = list_all_sessions(&db_path).len();
    assert_eq!(
        final_count, expected_count,
        "Should have {} sessions (5 initial + {} seeder threads * {} iterations)",
        expected_count, num_seeders, iterations
    );

    eprintln!(
        "Completed {} seeders + {} searchers, {} total sessions",
        num_seeders, num_searchers, final_count
    );
}
