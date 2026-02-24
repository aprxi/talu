//! Quick chat benchmarks: append latency, context retrieval.

use std::time::Instant;
use tempfile::TempDir;
use uuid::Uuid;

use talu::responses::{MessageRole, ResponsesView};
use talu::{ChatHandle, StorageHandle};

use super::BenchResult;

fn append_message(chat: &ChatHandle, content: &[u8]) {
    let rc = unsafe {
        talu_sys::talu_responses_append_message(
            chat.responses().as_ptr(),
            MessageRole::User as u8,
            content.as_ptr(),
            content.len(),
        )
    };
    assert!(rc >= 0, "append_message failed: {rc}");
}

pub fn run() -> Vec<BenchResult> {
    let mut results = Vec::new();

    // Append message latency (10 messages)
    {
        let dir = TempDir::new().unwrap();
        let sid = Uuid::new_v4().to_string();
        let chat = ChatHandle::new(None).unwrap();
        chat.set_storage_db(dir.path().to_str().unwrap(), &sid)
            .unwrap();

        let payload = b"Benchmark message for append latency measurement.";
        let start = Instant::now();
        for _ in 0..10 {
            append_message(&chat, payload);
        }
        results.push(BenchResult {
            name: "append_message",
            elapsed: start.elapsed(),
            ops: 10,
        });
    }

    // Context retrieval (load conversation + read 20 texts)
    {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().to_str().unwrap();
        let sid = Uuid::new_v4().to_string();
        {
            let chat = ChatHandle::new(None).unwrap();
            chat.set_storage_db(db_path, &sid).unwrap();
            for i in 0..1000 {
                let msg = format!("Message {i}: context payload for retrieval benchmark.");
                append_message(&chat, msg.as_bytes());
            }
        }

        let indices: Vec<usize> = (0..20).map(|i| i * 50).collect();
        let start = Instant::now();
        let storage = StorageHandle::open(db_path).unwrap();
        let conv = storage.load_session(&sid).unwrap();
        for &idx in &indices {
            let _ = conv.message_text(idx).unwrap();
        }
        results.push(BenchResult {
            name: "context_retrieval_20",
            elapsed: start.elapsed(),
            ops: 1,
        });
    }

    results
}
