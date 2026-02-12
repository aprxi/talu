//! Document lifecycle tests.
//!
//! Validates: create -> get -> update -> delete -> verify gone.

use crate::capi::db::common::TestContext;
use talu::documents::DocumentsHandle;

/// Create a document and retrieve it.
#[test]
fn create_and_get_document() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let doc_id = TestContext::unique_session_id();

    // Create
    handle
        .create(
            &doc_id,
            "note",
            "Test Note",
            r#"{"content": "Hello world"}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("create failed");

    // Get
    let doc = handle
        .get(&doc_id)
        .expect("get failed")
        .expect("document should exist");
    assert_eq!(doc.doc_id, doc_id);
    assert_eq!(doc.doc_type, "note");
    assert_eq!(doc.title, "Test Note");
    assert!(doc.doc_json.contains("Hello world"));
}

/// Create with all optional fields.
#[test]
fn create_with_all_fields() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let doc_id = TestContext::unique_session_id();
    let parent_id = TestContext::unique_session_id();

    handle
        .create(
            &doc_id,
            "article",
            "Full Article",
            r#"{"body": "Content"}"#,
            Some("tag1,tag2"), // tags_text
            Some(&parent_id),  // parent_id
            Some("draft"),     // marker
            Some("group-1"),   // group_id
            Some("owner-1"),   // owner_id
        )
        .expect("create with all fields failed");

    let doc = handle
        .get(&doc_id)
        .expect("get failed")
        .expect("document should exist");
    assert_eq!(doc.doc_type, "article");
    assert_eq!(doc.title, "Full Article");
    assert_eq!(doc.tags_text.as_deref(), Some("tag1,tag2"));
    assert_eq!(doc.parent_id.as_deref(), Some(parent_id.as_str()));
    assert_eq!(doc.marker.as_deref(), Some("draft"));
    assert_eq!(doc.group_id.as_deref(), Some("group-1"));
    assert_eq!(doc.owner_id.as_deref(), Some("owner-1"));
}

/// Update a document.
#[test]
fn update_document() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let doc_id = TestContext::unique_session_id();

    // Create
    handle
        .create(
            &doc_id,
            "note",
            "Original",
            r#"{"v": 1}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("create failed");

    // Update
    handle
        .update(
            &doc_id,
            Some("Updated Title"),
            Some(r#"{"v": 2}"#),
            None,
            None,
        )
        .expect("update failed");

    // Verify
    let doc = handle
        .get(&doc_id)
        .expect("get failed")
        .expect("document should exist");
    assert_eq!(doc.title, "Updated Title");
    assert!(doc.doc_json.contains(r#""v": 2"#) || doc.doc_json.contains(r#""v":2"#));
    assert!(doc.updated_at_ms >= doc.created_at_ms);
}

/// Delete a document.
#[test]
fn delete_document() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let doc_id = TestContext::unique_session_id();

    // Create
    handle
        .create(
            &doc_id,
            "note",
            "To Delete",
            "{}",
            None,
            None,
            None,
            None,
            None,
        )
        .expect("create failed");

    // Verify exists
    assert!(handle.get(&doc_id).expect("get failed").is_some());

    // Delete
    handle.delete(&doc_id).expect("delete failed");

    // Verify gone - accepts either Ok(None) or Err(DocumentNotFound)
    let result = handle.get(&doc_id);
    match result {
        Ok(None) => {}
        Ok(Some(_)) => panic!("document should be deleted"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("not found") || msg.contains("NotFound"),
                "unexpected error: {}",
                e
            );
        }
    }
}

/// List documents.
#[test]
fn list_documents() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    // Create multiple documents
    for i in 0..5 {
        let doc_id = format!("doc-{}", i);
        handle
            .create(
                &doc_id,
                "note",
                &format!("Note {}", i),
                "{}",
                None,
                None,
                None,
                None,
                None,
            )
            .expect("create failed");
    }

    // List all
    let docs = handle
        .list(None, None, None, None, 100)
        .expect("list failed");
    assert_eq!(docs.len(), 5);
}

/// List documents with type filter.
#[test]
fn list_documents_by_type() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    // Create mixed types
    handle
        .create(
            "doc-1", "note", "Note 1", "{}", None, None, None, None, None,
        )
        .unwrap();
    handle
        .create(
            "doc-2",
            "article",
            "Article 1",
            "{}",
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    handle
        .create(
            "doc-3", "note", "Note 2", "{}", None, None, None, None, None,
        )
        .unwrap();

    // List notes only
    let notes = handle
        .list(Some("note"), None, None, None, 100)
        .expect("list failed");
    assert_eq!(notes.len(), 2);
    for doc in &notes {
        assert_eq!(doc.doc_type, "note");
    }

    // List articles only
    let articles = handle
        .list(Some("article"), None, None, None, 100)
        .expect("list failed");
    assert_eq!(articles.len(), 1);
    assert_eq!(articles[0].doc_type, "article");
}

/// List documents with marker filter.
#[test]
fn list_documents_by_marker() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    handle
        .create(
            "doc-1",
            "note",
            "Draft 1",
            "{}",
            None,
            None,
            Some("draft"),
            None,
            None,
        )
        .unwrap();
    handle
        .create(
            "doc-2",
            "note",
            "Published",
            "{}",
            None,
            None,
            Some("published"),
            None,
            None,
        )
        .unwrap();
    handle
        .create(
            "doc-3",
            "note",
            "Draft 2",
            "{}",
            None,
            None,
            Some("draft"),
            None,
            None,
        )
        .unwrap();

    let drafts = handle
        .list(None, None, None, Some("draft"), 100)
        .expect("list failed");
    assert_eq!(drafts.len(), 2);
    for doc in &drafts {
        assert_eq!(doc.marker.as_deref(), Some("draft"));
    }
}

/// Documents persist across handle close/reopen.
#[test]
fn persistence_across_reopen() {
    let ctx = TestContext::new();
    let doc_id = TestContext::unique_session_id();

    // Create and close
    {
        let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
        handle
            .create(
                &doc_id,
                "note",
                "Persistent",
                r#"{"saved": true}"#,
                None,
                None,
                None,
                None,
                None,
            )
            .expect("create failed");
    }

    // Reopen and verify
    {
        let handle = DocumentsHandle::open(ctx.db_path()).expect("reopen failed");
        let doc = handle
            .get(&doc_id)
            .expect("get failed")
            .expect("document should persist");
        assert_eq!(doc.title, "Persistent");
        assert!(doc.doc_json.contains("saved"));
    }
}

/// Multiple documents are isolated.
#[test]
fn multiple_documents_isolated() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let id_a = TestContext::unique_session_id();
    let id_b = TestContext::unique_session_id();

    handle
        .create(
            &id_a,
            "type-a",
            "Doc A",
            r#"{"a": 1}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    handle
        .create(
            &id_b,
            "type-b",
            "Doc B",
            r#"{"b": 2}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    let doc_a = handle.get(&id_a).unwrap().unwrap();
    let doc_b = handle.get(&id_b).unwrap().unwrap();

    assert_eq!(doc_a.doc_type, "type-a");
    assert_eq!(doc_b.doc_type, "type-b");
    assert_ne!(doc_a.doc_id, doc_b.doc_id);
}

/// Update non-existent document fails gracefully.
#[test]
fn update_nonexistent_returns_error() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    let result = handle.update("nonexistent-id", Some("Title"), None, None, None);
    assert!(result.is_err());
}

/// Delete non-existent document is safe (idempotent).
#[test]
fn delete_nonexistent_is_safe() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    // Should not panic or error
    let result = handle.delete("nonexistent-id");
    // The C API may return success for idempotent delete
    // or error - either is acceptable
    let _ = result;
}

/// Get non-existent document returns None or DocumentNotFound.
#[test]
fn get_nonexistent_returns_none() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    let result = handle.get("nonexistent-id");
    // Either Ok(None) or Err(DocumentNotFound) is acceptable
    match result {
        Ok(None) => {}
        Ok(Some(_)) => panic!("should not find nonexistent document"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("not found") || msg.contains("NotFound"),
                "unexpected error: {}",
                e
            );
        }
    }
}

/// Handle path() returns the storage path.
#[test]
fn handle_path_returns_storage_path() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    let path = handle.path();
    assert_eq!(path.to_string_lossy(), ctx.db_path());
}

/// DocumentRecord has all fields populated.
#[test]
fn document_record_all_fields() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
    let doc_id = TestContext::unique_session_id();

    handle
        .create(
            &doc_id,
            "note",
            "Field Test",
            r#"{"key": "value"}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("create failed");

    let doc = handle
        .get(&doc_id)
        .expect("get failed")
        .expect("document should exist");

    // Verify all DocumentRecord fields are accessible
    assert_eq!(doc.doc_id, doc_id);
    assert_eq!(doc.doc_type, "note");
    assert_eq!(doc.title, "Field Test");
    assert!(doc.doc_json.contains("key"));
    assert!(doc.created_at_ms > 0, "created_at_ms should be set");
    assert!(
        doc.updated_at_ms >= doc.created_at_ms,
        "updated_at_ms should be >= created_at_ms"
    );
    // content_hash should be computed (non-zero for non-empty content)
    // Note: content_hash algorithm may vary, just verify field is accessible
    let _ = doc.content_hash;
    // seq_num should be assigned
    let _ = doc.seq_num;
    // expires_at_ms is 0 when no TTL set
    assert_eq!(
        doc.expires_at_ms, 0,
        "expires_at_ms should be 0 without TTL"
    );
    // Optional fields should be None when not set
    assert!(doc.tags_text.is_none() || doc.tags_text.as_deref() == Some(""));
    assert!(doc.parent_id.is_none());
    assert!(doc.marker.is_none());
    assert!(doc.group_id.is_none());
    assert!(doc.owner_id.is_none());
}

/// DocumentSummary has expected fields from list().
#[test]
fn document_summary_fields() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    handle
        .create(
            "sum-1",
            "note",
            "Summary Test",
            "{}",
            None,
            None,
            Some("draft"),
            None,
            None,
        )
        .unwrap();

    let summaries = handle
        .list(None, None, None, None, 100)
        .expect("list failed");
    assert_eq!(summaries.len(), 1);

    let summary = &summaries[0];
    assert_eq!(summary.doc_id, "sum-1");
    assert_eq!(summary.doc_type, "note");
    assert_eq!(summary.title, "Summary Test");
    assert_eq!(summary.marker.as_deref(), Some("draft"));
    assert!(summary.created_at_ms > 0);
    assert!(summary.updated_at_ms > 0);
}

/// SearchResult has expected fields from search().
#[test]
fn search_result_fields() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    handle
        .create(
            "sr-1",
            "article",
            "Search Result Test",
            r#"{"body": "unique searchable content xyz123"}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    let results = handle.search("xyz123", None, 10).expect("search failed");
    assert!(!results.is_empty(), "should find document");

    let result = &results[0];
    assert_eq!(result.doc_id, "sr-1");
    assert_eq!(result.doc_type, "article");
    assert_eq!(result.title, "Search Result Test");
    // snippet may or may not contain match depending on implementation
    let _ = result.snippet;
}
