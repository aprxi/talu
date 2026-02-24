//! Manifest portability tests.
//!
//! The manifest (`manifest.json`) is the root of trust for sealed segments.
//! All segment paths must be **relative** to the DB root directory so that
//! the database is portable across machines and S3 buckets.
//!
//! Segment rotation is not yet implemented — these tests verify the manifest
//! infrastructure and prepare regression coverage for when rotation lands.

use serde::Deserialize;
use std::path::PathBuf;

use crate::capi::db::common::TestContext;
use talu::responses::{MessageRole, ResponsesView};
use talu::vector::VectorStore;
use talu::ChatHandle;

/// Manifest JSON schema (mirrors Core's ManifestJson).
#[derive(Deserialize)]
struct Manifest {
    version: u32,
    segments: Vec<SegmentEntry>,
    last_compaction_ts: i64,
}

/// Segment entry within the manifest.
#[derive(Deserialize)]
#[allow(dead_code)]
struct SegmentEntry {
    id: String,
    path: String,
    min_ts: i64,
    max_ts: i64,
    row_count: u64,
}

fn manifest_path(db_root: &str) -> PathBuf {
    PathBuf::from(db_root).join("manifest.json")
}

// ---------------------------------------------------------------------------
// Synthetic manifest tests
// ---------------------------------------------------------------------------

/// A hand-crafted manifest with relative paths round-trips correctly.
///
/// Writes a valid manifest.json, reads it back, and verifies all segment
/// paths are relative. This exercises the expected on-disk format that
/// the reader will consume once segment rotation is implemented.
#[test]
fn synthetic_manifest_paths_are_relative() {
    let ctx = TestContext::new();

    let manifest = serde_json::json!({
        "version": 1,
        "last_compaction_ts": 1700000000,
        "segments": [
            {
                "id": "0123456789abcdef0123456789abcdef",
                "path": "chat/seg-1.talu",
                "min_ts": 100,
                "max_ts": 200,
                "row_count": 42
            },
            {
                "id": "fedcba9876543210fedcba9876543210",
                "path": "vector/seg-2.talu",
                "min_ts": 150,
                "max_ts": 300,
                "row_count": 100
            }
        ]
    });

    let path = manifest_path(ctx.db_path());
    std::fs::write(&path, serde_json::to_string_pretty(&manifest).unwrap())
        .expect("write manifest");

    // Read back and verify.
    let data = std::fs::read_to_string(&path).expect("read manifest");
    let parsed: Manifest = serde_json::from_str(&data).expect("parse manifest");

    assert_eq!(parsed.version, 1);
    assert_eq!(parsed.segments.len(), 2);
    assert_eq!(parsed.last_compaction_ts, 1700000000);

    for seg in &parsed.segments {
        assert!(
            !seg.path.starts_with('/'),
            "Segment path '{}' is absolute (Unix) — must be relative",
            seg.path,
        );
        assert!(
            !seg.path.contains(":\\"),
            "Segment path '{}' is absolute (Windows) — must be relative",
            seg.path,
        );
        // Paths should be namespace-prefixed: "chat/..." or "vector/..."
        assert!(
            seg.path.contains('/'),
            "Segment path '{}' should include namespace prefix (e.g. chat/seg-1.talu)",
            seg.path,
        );
    }

    assert_eq!(parsed.segments[0].path, "chat/seg-1.talu");
    assert_eq!(parsed.segments[1].path, "vector/seg-2.talu");
    assert_eq!(parsed.segments[0].row_count, 42);
    assert_eq!(parsed.segments[1].row_count, 100);
}

/// Manifest version field is present and valid.
#[test]
fn manifest_version_field() {
    let ctx = TestContext::new();

    let manifest = serde_json::json!({
        "version": 1,
        "last_compaction_ts": 0,
        "segments": []
    });

    let path = manifest_path(ctx.db_path());
    std::fs::write(&path, serde_json::to_string(&manifest).unwrap()).expect("write manifest");

    let data = std::fs::read_to_string(&path).expect("read manifest");
    let parsed: Manifest = serde_json::from_str(&data).expect("parse manifest");

    assert_eq!(parsed.version, 1, "Manifest version should be 1");
    assert!(parsed.segments.is_empty());
}

// ---------------------------------------------------------------------------
// Normal operations — manifest behavior
// ---------------------------------------------------------------------------

/// Normal chat operations do not create a manifest.json.
///
/// Segment rotation is not implemented. Until it is, no manifest should
/// appear on disk during normal read/write operations.
#[test]
fn no_manifest_from_normal_chat_ops() {
    let ctx = TestContext::new();
    let sid = TestContext::unique_session_id();

    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid).expect("set");
        chat.notify_session_update(Some("model"), Some("Test"), Some("active"))
            .expect("notify");
        let msg = b"Hello";
        let rc = unsafe {
            talu_sys::talu_responses_append_message(
                chat.responses().as_ptr(),
                MessageRole::User as u8,
                msg.as_ptr(),
                msg.len(),
            )
        };
        assert!(rc >= 0);
    }

    let path = manifest_path(ctx.db_path());
    assert!(
        !path.exists(),
        "manifest.json should not exist from normal operations \
         (segment rotation not yet implemented)",
    );
}

/// Normal vector operations do not create a manifest.json.
#[test]
fn no_manifest_from_normal_vector_ops() {
    let ctx = TestContext::new();

    {
        let store = VectorStore::open(ctx.db_path()).expect("open");
        store
            .append(
                &[1, 2, 3],
                &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                4,
            )
            .expect("append");
    }

    let path = manifest_path(ctx.db_path());
    assert!(
        !path.exists(),
        "manifest.json should not exist from normal vector operations",
    );
}

// ---------------------------------------------------------------------------
// Pre-existing manifest — reader compatibility
// ---------------------------------------------------------------------------

/// The reader gracefully handles a manifest with zero segments.
///
/// An empty manifest (no sealed segments) should not interfere with
/// reading data from current.talu.
#[test]
fn empty_manifest_does_not_block_reads() {
    let ctx = TestContext::new();
    let sid = TestContext::unique_session_id();

    // Write chat data.
    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid).expect("set");
        chat.notify_session_update(None, Some("With Manifest"), None)
            .expect("notify");
        let msg = b"Still readable";
        let rc = unsafe {
            talu_sys::talu_responses_append_message(
                chat.responses().as_ptr(),
                MessageRole::User as u8,
                msg.as_ptr(),
                msg.len(),
            )
        };
        assert!(rc >= 0);
    }

    // Drop an empty manifest into the DB root.
    let manifest = serde_json::json!({
        "version": 1,
        "last_compaction_ts": 0,
        "segments": []
    });
    let path = manifest_path(ctx.db_path());
    std::fs::write(&path, serde_json::to_string(&manifest).unwrap()).expect("write manifest");

    // Data should still be readable.
    let storage = talu::StorageHandle::open(ctx.db_path()).expect("open");
    let conv = storage.load_session(&sid).expect("load");
    assert_eq!(conv.item_count(), 1);

    use talu::responses::ResponsesView;
    assert_eq!(conv.message_text(0).unwrap(), "Still readable");
}

/// Segment paths in a manifest must follow the namespace/filename convention.
///
/// This verifies our understanding of the expected path format by parsing
/// the path components. When rotation is implemented, the actual paths
/// must match this convention.
#[test]
fn segment_path_convention() {
    // Expected format: "{namespace}/seg-{hex_id}.talu"
    // or at minimum: "{namespace}/{filename}.talu"
    let valid_paths = [
        "chat/seg-1.talu",
        "vector/seg-abc123.talu",
        "chat/seg-0123456789abcdef0123456789abcdef.talu",
    ];

    for path in &valid_paths {
        assert!(!path.starts_with('/'), "Absolute path: {path}");
        assert!(path.ends_with(".talu"), "Missing .talu extension: {path}");
        assert!(path.contains('/'), "Missing namespace prefix: {path}");

        let parts: Vec<&str> = path.splitn(2, '/').collect();
        assert_eq!(parts.len(), 2, "Expected namespace/filename: {path}");

        let namespace = parts[0];
        assert!(
            namespace == "chat" || namespace == "vector",
            "Unknown namespace '{namespace}' in path: {path}",
        );
    }
}
