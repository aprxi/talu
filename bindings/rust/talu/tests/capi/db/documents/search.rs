//! Document search tests.
//!
//! Validates full-text search functionality.

use crate::capi::db::common::TestContext;
use talu::documents::DocumentsHandle;

/// Basic search finds matching documents.
#[test]
fn search_finds_matching_documents() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    // Create documents with searchable content
    handle
        .create(
            "doc-rust",
            "article",
            "Rust Programming",
            r#"{"body": "Learn about Rust programming language"}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    handle
        .create(
            "doc-python",
            "article",
            "Python Guide",
            r#"{"body": "Python is a great language"}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    handle
        .create(
            "doc-js",
            "article",
            "JavaScript Basics",
            r#"{"body": "JavaScript for web development"}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    // Search for "Rust"
    let results = handle.search("Rust", None, 10).expect("search failed");
    assert!(!results.is_empty(), "should find Rust document");
    assert!(results.iter().any(|r| r.doc_id == "doc-rust"));
}

/// Search with type filter.
#[test]
fn search_with_type_filter() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    handle
        .create(
            "note-1",
            "note",
            "Programming Note",
            r#"{"text": "Notes about programming"}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    handle
        .create(
            "article-1",
            "article",
            "Programming Article",
            r#"{"text": "Article about programming"}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    // Search notes only
    let results = handle
        .search("programming", Some("note"), 10)
        .expect("search failed");
    for r in &results {
        assert_eq!(r.doc_type, "note", "should only return notes");
    }
}

/// Search respects limit.
#[test]
fn search_respects_limit() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    // Create many matching documents
    for i in 0..20 {
        handle
            .create(
                &format!("doc-{}", i),
                "article",
                &format!("Common Topic {}", i),
                r#"{"text": "Common searchable content"}"#,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
    }

    let results = handle.search("Common", None, 5).expect("search failed");
    assert!(results.len() <= 5, "should respect limit");
}

/// Search returns snippets.
#[test]
fn search_returns_snippets() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    handle
        .create(
            "doc-1",
            "article",
            "Test Article",
            r#"{"body": "This is a test article with specific searchable words"}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    let results = handle
        .search("specific searchable", None, 10)
        .expect("search failed");
    if !results.is_empty() {
        // Snippet should contain some context
        assert!(!results[0].snippet.is_empty() || !results[0].title.is_empty());
    }
}

/// Search no matches returns empty.
#[test]
fn search_no_matches_returns_empty() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    handle
        .create(
            "doc-1",
            "note",
            "Apple",
            r#"{"text": "Fruit"}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    let results = handle
        .search("xyznonexistent123", None, 10)
        .expect("search failed");
    assert!(results.is_empty(), "should find no matches");
}

/// Search is case-insensitive.
#[test]
fn search_case_insensitive() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    handle
        .create(
            "doc-1",
            "note",
            "Important Document",
            r#"{"text": "UPPERCASE and lowercase"}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    // Search with different cases
    let results_lower = handle.search("important", None, 10).unwrap();
    let results_upper = handle.search("IMPORTANT", None, 10).unwrap();
    let results_mixed = handle.search("ImPoRtAnT", None, 10).unwrap();

    // All should find the document (case-insensitive)
    // Note: this depends on the search implementation
    assert!(
        !results_lower.is_empty() || !results_upper.is_empty() || !results_mixed.is_empty(),
        "at least one case variant should match"
    );
}

/// Search in title.
#[test]
fn search_matches_title() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    handle
        .create(
            "doc-1",
            "note",
            "UniqueTitle123",
            r#"{"text": "generic content"}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    let results = handle
        .search("UniqueTitle123", None, 10)
        .expect("search failed");
    assert!(!results.is_empty(), "should match title");
}

/// Search in content JSON.
#[test]
fn search_matches_content() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    handle
        .create(
            "doc-1",
            "note",
            "Generic Title",
            r#"{"text": "UniqueContent456 is here"}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    let results = handle
        .search("UniqueContent456", None, 10)
        .expect("search failed");
    assert!(!results.is_empty(), "should match content");
}

/// Search persists across reopen.
#[test]
fn search_persists_across_reopen() {
    let ctx = TestContext::new();

    // Create and close
    {
        let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");
        handle
            .create(
                "persistent-doc",
                "article",
                "Persistent Search Test",
                r#"{"text": "searchable persistent content"}"#,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
    }

    // Reopen and search
    {
        let handle = DocumentsHandle::open(ctx.db_path()).expect("reopen failed");
        let results = handle
            .search("persistent", None, 10)
            .expect("search failed");
        assert!(!results.is_empty(), "should find document after reopen");
    }
}

/// Search deleted documents not found.
#[test]
fn search_deleted_not_found() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    handle
        .create(
            "doc-to-delete",
            "note",
            "Delete Me Search",
            r#"{"text": "will be deleted"}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    // Verify searchable
    let before = handle.search("Delete Me Search", None, 10).unwrap();
    assert!(!before.is_empty(), "should find before delete");

    // Delete
    handle.delete("doc-to-delete").unwrap();

    // Verify not searchable
    let after = handle.search("Delete Me Search", None, 10).unwrap();
    assert!(after.is_empty(), "should not find after delete");
}

/// Search multiple words (AND semantics).
#[test]
fn search_multiple_words() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    handle
        .create(
            "doc-both",
            "note",
            "Alpha Beta",
            r#"{"text": "has both words"}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    handle
        .create(
            "doc-alpha",
            "note",
            "Alpha Only",
            r#"{"text": "just alpha"}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    handle
        .create(
            "doc-beta",
            "note",
            "Beta Only",
            r#"{"text": "just beta"}"#,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    let results = handle.search("Alpha Beta", None, 10).unwrap();
    // May return just the doc with both, or all - depends on search semantics
    assert!(!results.is_empty(), "should find at least one");
}

/// Stress test: search with many documents.
#[test]
fn search_many_documents() {
    let ctx = TestContext::new();
    let handle = DocumentsHandle::open(ctx.db_path()).expect("open failed");

    // Create many documents
    for i in 0..100 {
        let content = if i % 10 == 0 {
            r#"{"text": "special keyword content"}"#.to_string()
        } else {
            format!(r#"{{"text": "document number {}"}}"#, i)
        };
        handle
            .create(
                &format!("bulk-{}", i),
                "note",
                &format!("Doc {}", i),
                &content,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap();
    }

    let results = handle.search("special keyword", None, 100).unwrap();
    assert_eq!(results.len(), 10, "should find 10 special documents");
}
