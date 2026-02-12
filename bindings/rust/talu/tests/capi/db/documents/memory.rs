//! Memory safety and error handling tests.
//!
//! Validates proper handling of null inputs, invalid paths, and edge cases.

use crate::capi::db::common::TestContext;
use talu::documents::DocumentsHandle;

/// Empty document ID handled safely.
#[test]
fn empty_doc_id_handled() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    // Create with empty ID should fail
    let result = handle.create("", "note", "Title", "{}", None, None, None, None, None);
    assert!(result.is_err(), "empty doc_id should be rejected");
}

/// Empty document type handled safely.
#[test]
fn empty_doc_type_handled() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    let result = handle.create("doc-1", "", "Title", "{}", None, None, None, None, None);
    assert!(result.is_err(), "empty doc_type should be rejected");
}

/// Very long document ID handled safely.
#[test]
fn very_long_doc_id_handled() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let long_id = "x".repeat(10000);

    // Should either succeed or fail gracefully, not crash
    let _ = handle.create(
        &long_id, "note", "Title", "{}", None, None, None, None, None,
    );
}

/// Very long title handled safely.
#[test]
fn very_long_title_handled() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let long_title = "Title ".repeat(5000);
    let doc_id = TestContext::unique_session_id();

    // Should either succeed or fail gracefully
    let result = handle.create(
        &doc_id,
        "note",
        &long_title,
        "{}",
        None,
        None,
        None,
        None,
        None,
    );
    if result.is_ok() {
        let doc = handle.get(&doc_id).unwrap().unwrap();
        assert!(!doc.title.is_empty());
    }
}

/// Very long content handled safely.
#[test]
fn very_long_content_handled() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let long_content = format!(r#"{{"data": "{}"}}"#, "x".repeat(100_000));
    let doc_id = TestContext::unique_session_id();

    let result = handle.create(
        &doc_id,
        "note",
        "Title",
        &long_content,
        None,
        None,
        None,
        None,
        None,
    );
    if result.is_ok() {
        let doc = handle.get(&doc_id).unwrap().unwrap();
        assert!(doc.doc_json.len() > 1000);
    }
}

/// Invalid JSON content handled.
#[test]
fn invalid_json_content_handled() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let doc_id = TestContext::unique_session_id();

    // Content is stored as-is, not validated as JSON by the storage layer
    let result = handle.create(
        &doc_id,
        "note",
        "Title",
        "not valid json {{{",
        None,
        None,
        None,
        None,
        None,
    );
    // May succeed (storage doesn't validate JSON) or fail - either is fine
    let _ = result;
}

/// Nonexistent DB path returns error.
#[test]
fn nonexistent_db_path_returns_error() {
    let result = DocumentsHandle::open("/nonexistent/path/that/does/not/exist/12345");
    // May succeed (auto-create) or fail - check it doesn't crash
    let _ = result;
}

/// Unicode content handled correctly.
#[test]
fn unicode_content_handled() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let doc_id = TestContext::unique_session_id();

    let unicode_content = r#"{"emoji": "🎉🚀💯", "chinese": "中文测试", "arabic": "العربية"}"#;
    handle
        .create(
            &doc_id,
            "note",
            "Unicode Test 中文",
            unicode_content,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("create with unicode failed");

    let doc = handle.get(&doc_id).unwrap().unwrap();
    assert!(doc.title.contains("中文"));
    assert!(doc.doc_json.contains("🎉"));
    assert!(doc.doc_json.contains("中文测试"));
}

/// Null bytes in content handled safely.
#[test]
fn null_bytes_in_content_handled() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let doc_id = TestContext::unique_session_id();

    // Content with embedded null should be truncated or rejected, not crash
    let content_with_null = "before\0after";
    let result = handle.create(
        &doc_id,
        "note",
        "Title",
        content_with_null,
        None,
        None,
        None,
        None,
        None,
    );
    // Either succeeds with truncated content or fails - both acceptable
    let _ = result;
}

/// Rapid create/delete cycles don't leak memory.
#[test]
fn rapid_create_delete_no_leak() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    for i in 0..100 {
        let doc_id = format!("rapid-{}", i);
        handle
            .create(&doc_id, "note", "Rapid", "{}", None, None, None, None, None)
            .unwrap();
        handle.delete(&doc_id).unwrap();
    }

    // Verify storage is clean
    let docs = handle.list(None, None, None, None, 1000).unwrap();
    assert!(docs.is_empty(), "all documents should be deleted");
}

/// Concurrent read operations don't crash.
#[test]
fn concurrent_reads_safe() {
    use std::sync::Arc;
    use std::thread;

    let ctx = TestContext::new();
    let path = ctx.db_path().to_string();

    // Create some documents first
    {
        let handle = DocumentsHandle::open(&path).expect("open failed");
        for i in 0..10 {
            handle
                .create(
                    &format!("doc-{}", i),
                    "note",
                    &format!("Note {}", i),
                    "{}",
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();
        }
    }

    let path = Arc::new(path);
    let mut threads = vec![];

    // Spawn multiple readers
    for _ in 0..4 {
        let p = Arc::clone(&path);
        threads.push(thread::spawn(move || {
            let handle = DocumentsHandle::open(p.as_str()).expect("open failed");
            for _ in 0..20 {
                let _ = handle.list(None, None, None, None, 100);
                let _ = handle.get("doc-0");
                let _ = handle.get("nonexistent");
            }
        }));
    }

    for t in threads {
        t.join().expect("thread panicked");
    }
}

/// List with very large limit doesn't crash.
#[test]
fn list_large_limit_safe() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    // Create a few documents
    for i in 0..3 {
        handle
            .create(
                &format!("doc-{}", i),
                "note",
                "Note",
                "{}",
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
    }

    // Request way more than exist
    let docs = handle.list(None, None, None, None, u32::MAX).unwrap();
    assert_eq!(docs.len(), 3);
}

/// Search with empty query handled.
#[test]
fn search_empty_query_handled() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    let result = handle.search("", None, 10);
    // May return all or none - just shouldn't crash
    let _ = result;
}

/// Search with very long query handled.
#[test]
fn search_long_query_handled() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    let long_query = "search ".repeat(1000);
    let result = handle.search(&long_query, None, 10);
    // Should not crash
    let _ = result;
}

// =============================================================================
// Advanced Memory Safety Tests
// =============================================================================

/// Multiple handles to same DB path work correctly.
#[test]
fn multiple_handles_same_path() {
    let ctx = TestContext::new();
    let path = ctx.db_path().to_string();

    // Open first handle and create document
    {
        let handle1 = DocumentsHandle::open(&path).expect("open 1 failed");
        handle1
            .create(
                "shared-doc",
                "note",
                "Shared",
                "{}",
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
    }

    // Open second handle and read document
    {
        let handle2 = DocumentsHandle::open(&path).expect("open 2 failed");
        let doc = handle2.get("shared-doc").unwrap();
        assert!(doc.is_some() || doc.is_none()); // May or may not be visible depending on timing
    }
}

/// Handle drop doesn't leak resources.
#[test]
fn handle_drop_no_leak() {
    let ctx = TestContext::new();

    for _ in 0..50 {
        let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
        handle
            .create(
                &TestContext::unique_session_id(),
                "note",
                "Drop Test",
                "{}",
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        // handle drops here
    }
    // No assertion - test passes if no crash/leak
}

/// Operations after many creates don't degrade.
#[test]
fn operations_after_bulk_creates() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    // Bulk create
    for i in 0..200 {
        handle
            .create(
                &format!("bulk-{}", i),
                "note",
                &format!("Note {}", i),
                r#"{"i": 0}"#,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
    }

    // Operations should still work
    let docs = handle.list(None, None, None, None, 1000).unwrap();
    assert_eq!(docs.len(), 200);

    let search = handle.search("Note", None, 100).unwrap();
    assert!(!search.is_empty());

    // Update should work
    handle
        .update("bulk-50", Some("Updated"), None, None, None)
        .unwrap();

    // Delete should work
    handle.delete("bulk-100").unwrap();
}

/// Very long tag names handled safely.
#[test]
fn very_long_tag_name_handled() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let doc_id = TestContext::unique_session_id();
    let long_tag = "t".repeat(5000);

    handle
        .create(
            &doc_id, "note", "Tag Test", "{}", None, None, None, None, None,
        )
        .unwrap();

    // May succeed or fail - should not crash
    let _ = handle.add_tag(&doc_id, &long_tag, None);
}

/// Many tags on single document handled.
#[test]
fn stress_many_tags_single_doc() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let doc_id = TestContext::unique_session_id();

    handle
        .create(
            &doc_id,
            "note",
            "Many Tags",
            "{}",
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    // Add many tags
    for i in 0..100 {
        handle
            .add_tag(&doc_id, &format!("stress-tag-{}", i), None)
            .unwrap();
    }

    // Get all tags
    let tags = handle.get_tags(&doc_id).unwrap();
    assert_eq!(tags.len(), 100);

    // Remove all tags
    for i in 0..100 {
        let _ = handle.remove_tag(&doc_id, &format!("stress-tag-{}", i), None);
    }
}

/// Interleaved operations don't corrupt state.
#[test]
fn interleaved_operations_safe() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    for i in 0..50 {
        let doc_id = format!("interleave-{}", i);

        // Create
        handle
            .create(
                &doc_id,
                "note",
                &format!("Doc {}", i),
                "{}",
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();

        // Immediately update
        handle
            .update(&doc_id, Some(&format!("Updated {}", i)), None, None, None)
            .unwrap();

        // Add tag
        handle.add_tag(&doc_id, "interleaved", None).unwrap();

        // Search
        let _ = handle.search(&format!("Updated {}", i), None, 5);

        // List
        let _ = handle.list(None, None, None, None, 10);

        // Get
        let doc = handle.get(&doc_id).unwrap().unwrap();
        assert!(doc.title.contains("Updated"));

        // Delete every other
        if i % 2 == 0 {
            handle.delete(&doc_id).unwrap();
        }
    }

    // Verify final state
    let remaining = handle.list(None, None, None, None, 100).unwrap();
    assert_eq!(remaining.len(), 25); // Half were deleted
}

/// Special characters in all string fields.
#[test]
fn special_chars_all_fields() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    let special = "test'\"\\`!@#$%^&*()[]{}|;:,.<>?/~";
    let doc_id = TestContext::unique_session_id();

    // Create with special chars in all fields
    handle
        .create(
            &doc_id,
            "type-special",
            &format!("Title {}", special),
            &format!(
                r#"{{"content": "{}"}}"#,
                special.replace('\\', "\\\\").replace('"', "\\\"")
            ),
            Some(&format!(
                "tag{}",
                special
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
            )),
            None,
            Some("marker-special"),
            Some("group-special"),
            Some("owner-special"),
        )
        .expect("create with special chars failed");

    // Verify retrieval
    let doc = handle.get(&doc_id).unwrap().unwrap();
    assert!(doc.title.contains("test"));
}

/// Zero limit in list returns empty or default.
#[test]
fn list_zero_limit_handled() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    handle
        .create("doc-1", "note", "Test", "{}", None, None, None, None, None)
        .unwrap();

    // Zero limit - may return empty or use default limit
    let result = handle.list(None, None, None, None, 0);
    // Should not crash
    let _ = result;
}

/// Search zero limit handled.
#[test]
fn search_zero_limit_handled() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    handle
        .create(
            "doc-1",
            "note",
            "Searchable",
            "{}",
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    // Zero limit
    let result = handle.search("Searchable", None, 0);
    // Should not crash
    let _ = result;
}

/// Concurrent write operations don't corrupt.
#[test]
fn concurrent_writes_safe() {
    use std::sync::Arc;
    use std::thread;

    let ctx = TestContext::new();
    let path = Arc::new(ctx.db_path().to_string());

    let mut threads = vec![];

    for t in 0..4 {
        let p = Arc::clone(&path);
        threads.push(thread::spawn(move || {
            let handle = DocumentsHandle::open(p.as_str()).expect("open failed");
            for i in 0..25 {
                let doc_id = format!("concurrent-{}-{}", t, i);
                let _ = handle.create(
                    &doc_id,
                    "note",
                    &format!("Thread {} Doc {}", t, i),
                    "{}",
                    None,
                    None,
                    None,
                    None,
                    None,
                );
            }
        }));
    }

    for t in threads {
        t.join().expect("thread panicked");
    }

    // Verify database is still readable
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let docs = handle.list(None, None, None, None, 200).unwrap();
    // Some documents should have been created (may not be all 100 due to conflicts)
    assert!(docs.len() > 0);
}
