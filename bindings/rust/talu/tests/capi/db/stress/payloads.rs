//! Huge payload boundary tests.
//!
//! Verifies that the storage engine correctly handles messages that far
//! exceed the auto-flush threshold (64KB). A single 2MB message must be
//! written, persisted, and read back with full integrity.

use crate::capi::db::common::TestContext;
use talu::responses::{MessageRole, ResponsesView};
use talu::{ChatHandle, StorageHandle};

// ---------------------------------------------------------------------------
// Huge message payload
// ---------------------------------------------------------------------------

/// A single 2MB message round-trips through storage without corruption.
///
/// Strategy:
///   1. Build a deterministic 2MB payload (repeating pattern, verifiable).
///   2. Append it as a single user message via ChatHandle.
///   3. Drop the handle (flushes to disk).
///   4. Reopen via StorageHandle::load_session and verify full content.
#[test]
fn huge_message_payload() {
    let ctx = TestContext::new();
    let sid = TestContext::unique_session_id();

    // Build a deterministic 2MB payload.
    // Pattern: repeating "ABCDEFGHIJ" (10 bytes) to fill 2MB.
    const TWO_MB: usize = 2 * 1024 * 1024;
    let pattern = b"ABCDEFGHIJ";
    let payload: Vec<u8> = pattern.iter().copied().cycle().take(TWO_MB).collect();
    assert_eq!(payload.len(), TWO_MB);

    // Write the payload as a single message.
    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid).expect("set");

        let rc = unsafe {
            talu_sys::talu_responses_append_message(
                chat.responses().as_ptr(),
                MessageRole::User as u8,
                payload.as_ptr(),
                payload.len(),
            )
        };
        assert!(rc >= 0, "append 2MB message failed: {rc}");

        // Verify in-memory before flush.
        let view = chat.responses();
        assert_eq!(view.item_count(), 1);
        let text = view.message_text(0).expect("message_text");
        assert_eq!(text.len(), TWO_MB, "in-memory text length mismatch");
    }
    // ChatHandle dropped → flushBlock() → current.talu written.

    // Read back via StorageHandle (independent read path).
    let storage = StorageHandle::open(ctx.db_path()).expect("open");
    let conv = storage.load_session(&sid).expect("load");

    assert_eq!(conv.item_count(), 1, "expected 1 item after reload");

    let text = conv.message_text(0).expect("message_text after reload");
    assert_eq!(
        text.len(),
        TWO_MB,
        "Reloaded text length mismatch: got {}, expected {TWO_MB}",
        text.len(),
    );

    // Verify content integrity — check the full payload.
    let expected = std::str::from_utf8(&payload).expect("payload is utf8");
    assert_eq!(
        text, expected,
        "2MB payload content mismatch after round-trip",
    );
}

/// Multiple large messages in the same session round-trip correctly.
///
/// Writes 3 × 100KB messages, then verifies all three are intact after
/// reopen. Each message has a distinct prefix for easy identification.
#[test]
fn multiple_large_messages() {
    let ctx = TestContext::new();
    let sid = TestContext::unique_session_id();

    const SIZE: usize = 100 * 1024; // 100KB each
    let messages: Vec<Vec<u8>> = (0..3)
        .map(|i| {
            let prefix = format!("MSG{i}:");
            let filler = vec![b'A' + i as u8; SIZE - prefix.len()];
            let mut msg = prefix.into_bytes();
            msg.extend_from_slice(&filler);
            msg
        })
        .collect();

    // Write all three messages.
    {
        let chat = ChatHandle::new(None).expect("new");
        chat.set_storage_db(ctx.db_path(), &sid).expect("set");

        for (i, msg) in messages.iter().enumerate() {
            let rc = unsafe {
                talu_sys::talu_responses_append_message(
                    chat.responses().as_ptr(),
                    MessageRole::User as u8,
                    msg.as_ptr(),
                    msg.len(),
                )
            };
            assert!(rc >= 0, "append message {i} failed: {rc}");
        }
    }

    // Read back and verify each message.
    let storage = StorageHandle::open(ctx.db_path()).expect("open");
    let conv = storage.load_session(&sid).expect("load");

    assert_eq!(conv.item_count(), 3, "expected 3 items after reload");

    for (i, expected_bytes) in messages.iter().enumerate() {
        let text = conv.message_text(i).expect("message_text");
        let expected = std::str::from_utf8(expected_bytes).expect("utf8");
        assert_eq!(
            text.len(),
            expected.len(),
            "Message {i}: length mismatch (got {}, expected {})",
            text.len(),
            expected.len(),
        );
        assert_eq!(
            text, expected,
            "Message {i}: content mismatch after round-trip",
        );
    }
}
