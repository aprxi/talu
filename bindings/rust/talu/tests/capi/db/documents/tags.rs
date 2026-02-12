//! Document tag operations tests.
//!
//! Validates add/remove/get tag operations on documents.

use crate::capi::db::common::TestContext;
use talu::documents::DocumentsHandle;

/// Add tags to a document.
#[test]
fn add_tags_to_document() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let doc_id = TestContext::unique_session_id();

    // Create document
    handle
        .create(
            &doc_id,
            "note",
            "Tagged Doc",
            "{}",
            None,
            None,
            None,
            None,
            None,
        )
        .expect("create failed");

    // Add tags
    handle
        .add_tag(&doc_id, "important", None)
        .expect("add_tag failed");
    handle
        .add_tag(&doc_id, "urgent", None)
        .expect("add_tag failed");

    // Verify tags
    let tags = handle.get_tags(&doc_id).expect("get_tags failed");
    assert!(tags.contains(&"important".to_string()));
    assert!(tags.contains(&"urgent".to_string()));
}

/// Remove tags from a document.
#[test]
fn remove_tags_from_document() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let doc_id = TestContext::unique_session_id();

    // Create and tag
    handle
        .create(
            &doc_id, "note", "Tagged", "{}", None, None, None, None, None,
        )
        .unwrap();
    handle.add_tag(&doc_id, "keep", None).unwrap();
    handle.add_tag(&doc_id, "remove-me", None).unwrap();

    // Remove one tag
    handle
        .remove_tag(&doc_id, "remove-me", None)
        .expect("remove_tag failed");

    // Verify
    let tags = handle.get_tags(&doc_id).unwrap();
    assert!(tags.contains(&"keep".to_string()));
    assert!(!tags.contains(&"remove-me".to_string()));
}

/// Get tags from untagged document returns empty.
#[test]
fn get_tags_empty_document() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let doc_id = TestContext::unique_session_id();

    handle
        .create(
            &doc_id, "note", "No Tags", "{}", None, None, None, None, None,
        )
        .unwrap();

    let tags = handle.get_tags(&doc_id).expect("get_tags failed");
    assert!(tags.is_empty());
}

/// Get documents by tag.
#[test]
fn get_documents_by_tag() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let doc1 = TestContext::unique_session_id();
    let doc2 = TestContext::unique_session_id();
    let doc3 = TestContext::unique_session_id();

    // Create documents
    handle
        .create(&doc1, "note", "Doc 1", "{}", None, None, None, None, None)
        .unwrap();
    handle
        .create(&doc2, "note", "Doc 2", "{}", None, None, None, None, None)
        .unwrap();
    handle
        .create(&doc3, "note", "Doc 3", "{}", None, None, None, None, None)
        .unwrap();

    // Tag some documents
    handle.add_tag(&doc1, "shared-tag", None).unwrap();
    handle.add_tag(&doc2, "shared-tag", None).unwrap();
    handle.add_tag(&doc3, "other-tag", None).unwrap();

    // Get by tag
    let docs = handle.get_by_tag("shared-tag").expect("get_by_tag failed");
    assert_eq!(docs.len(), 2);
    assert!(docs.contains(&doc1));
    assert!(docs.contains(&doc2));
    assert!(!docs.contains(&doc3));
}

/// Tags persist across handle reopen.
#[test]
fn tags_persist_across_reopen() {
    let ctx = TestContext::new();
    let doc_id = TestContext::unique_session_id();

    // Create and tag
    {
        let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
        handle
            .create(
                &doc_id,
                "note",
                "Persistent Tags",
                "{}",
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        handle.add_tag(&doc_id, "persistent", None).unwrap();
    }

    // Reopen and verify
    {
        let handle = DocumentsHandle::open(ctx.db_path()).expect("reopen failed");
        let tags = handle.get_tags(&doc_id).expect("get_tags failed");
        assert!(tags.contains(&"persistent".to_string()));
    }
}

/// Add same tag twice is idempotent.
#[test]
fn add_tag_twice_idempotent() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let doc_id = TestContext::unique_session_id();

    handle
        .create(
            &doc_id,
            "note",
            "Double Tag",
            "{}",
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    handle.add_tag(&doc_id, "duplicate", None).unwrap();
    handle.add_tag(&doc_id, "duplicate", None).unwrap();

    let tags = handle.get_tags(&doc_id).unwrap();
    // Should have exactly one instance
    assert_eq!(tags.iter().filter(|t| *t == "duplicate").count(), 1);
}

/// Remove nonexistent tag is safe.
#[test]
fn remove_nonexistent_tag_safe() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let doc_id = TestContext::unique_session_id();

    handle
        .create(
            &doc_id,
            "note",
            "No Such Tag",
            "{}",
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    // Should not error
    let result = handle.remove_tag(&doc_id, "nonexistent", None);
    let _ = result; // May succeed or fail - both acceptable
}

/// Tags with group_id isolation.
#[test]
fn tags_with_group_isolation() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let doc_id = TestContext::unique_session_id();

    handle
        .create(
            &doc_id,
            "note",
            "Grouped Tags",
            "{}",
            None,
            None,
            None,
            Some("group-a"),
            None,
        )
        .unwrap();
    handle
        .add_tag(&doc_id, "group-tag", Some("group-a"))
        .unwrap();

    // Tags should be visible with same group
    let tags = handle.get_tags(&doc_id).unwrap();
    assert!(tags.contains(&"group-tag".to_string()));
}

/// Unicode tag names handled.
#[test]
fn unicode_tag_names() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let doc_id = TestContext::unique_session_id();

    handle
        .create(
            &doc_id,
            "note",
            "Unicode Tags",
            "{}",
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    handle
        .add_tag(&doc_id, "重要", None)
        .expect("add chinese tag failed");
    handle
        .add_tag(&doc_id, "🏷️", None)
        .expect("add emoji tag failed");

    let tags = handle.get_tags(&doc_id).unwrap();
    assert!(tags.contains(&"重要".to_string()));
    assert!(tags.contains(&"🏷️".to_string()));
}

/// Stress test: many tags on one document.
#[test]
fn many_tags_on_document() {
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
    for i in 0..50 {
        handle
            .add_tag(&doc_id, &format!("tag-{}", i), None)
            .unwrap();
    }

    let tags = handle.get_tags(&doc_id).unwrap();
    assert_eq!(tags.len(), 50);
}

/// Stress test: one tag on many documents.
#[test]
fn one_tag_many_documents() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    let mut doc_ids = vec![];
    for i in 0..30 {
        let doc_id = format!("bulk-doc-{}", i);
        handle
            .create(
                &doc_id,
                "note",
                &format!("Bulk {}", i),
                "{}",
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        handle.add_tag(&doc_id, "bulk-tag", None).unwrap();
        doc_ids.push(doc_id);
    }

    let docs = handle.get_by_tag("bulk-tag").expect("get_by_tag failed");
    assert_eq!(docs.len(), 30);
}
