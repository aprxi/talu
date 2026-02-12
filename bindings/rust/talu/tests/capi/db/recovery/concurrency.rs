//! Concurrency tests for TaluDB storage.
//!
//! TaluDB uses granular flock-based locking: the lock is held only during
//! the microseconds of actual WAL/block I/O. Multiple handles on the same
//! path coexist with serialized writes.

use crate::capi::db::common::TestContext;
use talu::vector::VectorStore;
use talu::{ChatHandle, StorageHandle};

/// Two VectorStore handles on the same path can both write successfully.
/// Flock serializes their WAL writes; no data is lost.
#[test]
fn two_handles_both_write_successfully() {
    let ctx = TestContext::new();
    let dims: u32 = 4;

    let store_a = VectorStore::open(ctx.db_path()).expect("open A failed");
    let store_b = VectorStore::open(ctx.db_path()).expect("open B failed");

    store_a
        .append(&[1], &[1.0, 0.0, 0.0, 0.0], dims)
        .expect("append A failed");
    store_b
        .append(&[2], &[0.0, 1.0, 0.0, 0.0], dims)
        .expect("append B failed");

    // Both should see their own data (each has its own in-memory buffer).
    let load_a = store_a.load().expect("load A failed");
    let load_b = store_b.load().expect("load B failed");

    // store_a's in-memory buffer has id=1; store_b's has id=2.
    // load() flushes first, so store_a's block will contain id=1.
    // store_b will see its own id=2 plus whatever was flushed by a.
    assert!(
        load_a.ids.contains(&1),
        "Store A should see its own vector (id=1)",
    );
    assert!(
        load_b.ids.contains(&2),
        "Store B should see its own vector (id=2)",
    );

    // After both close and reopen, all data should be visible.
    drop(store_a);
    drop(store_b);

    let store = VectorStore::open(ctx.db_path()).expect("reopen failed");
    let loaded = store.load().expect("final load failed");

    let mut ids = loaded.ids.clone();
    ids.sort();
    assert_eq!(
        ids,
        vec![1, 2],
        "Both vectors should survive concurrent writes"
    );
}

/// Sequential open/close cycles on the same path never corrupt data.
#[test]
fn sequential_open_close_no_corruption() {
    let ctx = TestContext::new();
    let dims: u32 = 4;

    for i in 0u64..10 {
        let store = VectorStore::open(ctx.db_path()).expect("open failed");
        let vec = vec![(i + 1) as f32, 0.0, 0.0, 0.0];
        store.append(&[i + 1], &vec, dims).expect("append failed");
        // Drop closes cleanly
    }

    let store = VectorStore::open(ctx.db_path()).expect("final open");
    let loaded = store.load().expect("final load");
    assert_eq!(
        loaded.ids.len(),
        10,
        "All 10 sequential writes should survive"
    );
}

/// Concurrent get_session calls must not return corrupted session_ids.
///
/// Regression test for a static-buffer race in talu_storage_get_session_info.
/// The C function copies output strings into process-global static buffers.
/// Without thread-local storage, concurrent calls overwrite each other's
/// output between the C return and the Rust cstr_to_string copy.
///
/// Strategy: create N sessions in a shared DB, then spawn N threads that
/// each call get_session in a tight loop and assert the returned session_id
/// matches the requested one.
#[test]
fn get_session_no_cross_thread_corruption() {
    let ctx = TestContext::new();
    let num_threads = 8;
    let iterations = 200;

    // Create N sessions, each with distinct metadata.
    let session_ids: Vec<String> = (0..num_threads)
        .map(|i| {
            let sid = TestContext::unique_session_id();
            let chat = ChatHandle::new(None).expect("new");
            chat.set_storage_db(ctx.db_path(), &sid).expect("set");
            chat.notify_session_update(
                Some(&format!("model-{i}")),
                Some(&format!("Title-{i}")),
                Some("active"),
            )
            .expect("notify");
            drop(chat);
            sid
        })
        .collect();

    // Hammer get_session from N threads concurrently.
    let db_path = ctx.db_path().to_string();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let path = db_path.clone();
            let sid = session_ids[i].clone();
            let expected_title = format!("Title-{i}");
            let bar = barrier.clone();
            std::thread::spawn(move || {
                let storage = StorageHandle::open(&path).expect("open");
                bar.wait(); // Maximize contention.
                for _ in 0..iterations {
                    let session = storage.get_session(&sid).expect("get_session");
                    assert_eq!(
                        session.session_id, sid,
                        "Thread {i}: got session_id '{}', expected '{sid}'",
                        session.session_id,
                    );
                    assert_eq!(
                        session.title.as_deref(),
                        Some(expected_title.as_str()),
                        "Thread {i}: title mismatch",
                    );
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }
}

/// Threaded concurrent appends to the same path.
/// Each thread opens its own handle, writes one vector, and closes.
/// All data should be present after all threads complete.
#[test]
fn threaded_concurrent_appends() {
    let ctx = TestContext::new();
    let db_path = ctx.db_path().to_string();
    let dims: u32 = 4;
    let num_threads = 8;

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let path = db_path.clone();
            std::thread::spawn(move || {
                let store = VectorStore::open(&path).expect("thread open failed");
                let id = (i + 1) as u64;
                let vec = vec![id as f32, 0.0, 0.0, 0.0];
                store
                    .append(&[id], &vec, dims)
                    .expect("thread append failed");
                // Drop closes cleanly
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    // Verify all vectors present.
    let store = VectorStore::open(&db_path).expect("final open");
    let loaded = store.load().expect("final load");

    let mut ids = loaded.ids.clone();
    ids.sort();
    let expected: Vec<u64> = (1..=num_threads as u64).collect();
    assert_eq!(
        ids, expected,
        "All {} threaded writes should survive",
        num_threads,
    );
}
