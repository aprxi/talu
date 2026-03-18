//! Tests for singleflight model-loading coordination.
//!
//! These exercise the `model_load_inflight` map + `ModelLoadResult` watch
//! channel protocol directly, without requiring a real model or backend.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use talu_cli::server::state::ModelLoadResult;

/// Simulate the singleflight check-and-register pattern used by
/// `ensure_backend_for_model`. Returns `true` if this caller became the
/// loader (created the watch channel), `false` if it became a waiter.
fn try_become_loader(
    inflight: &std::sync::Mutex<
        std::collections::HashMap<String, tokio::sync::watch::Receiver<ModelLoadResult>>,
    >,
    model_id: &str,
    tx: &tokio::sync::watch::Sender<ModelLoadResult>,
    rx: tokio::sync::watch::Receiver<ModelLoadResult>,
) -> Option<tokio::sync::watch::Receiver<ModelLoadResult>> {
    let mut map = inflight.lock().unwrap();
    if let Some(existing) = map.get(model_id) {
        // Another task is already loading — return its receiver.
        Some(existing.clone())
    } else {
        map.insert(model_id.to_string(), rx);
        // Suppress unused-variable warning — tx is used by the caller.
        let _ = tx;
        None
    }
}

/// 5 tasks race to "load" the same model. Exactly one becomes the loader;
/// the rest wait and see the same success result.
#[tokio::test]
async fn singleflight_deduplicates_concurrent_model_loads() {
    let inflight: Arc<
        std::sync::Mutex<
            std::collections::HashMap<String, tokio::sync::watch::Receiver<ModelLoadResult>>,
        >,
    > = Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));

    let loader_count = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(tokio::sync::Barrier::new(5));

    let mut handles = Vec::new();
    for _ in 0..5 {
        let inflight = inflight.clone();
        let loader_count = loader_count.clone();
        let barrier = barrier.clone();

        handles.push(tokio::spawn(async move {
            // All 5 tasks arrive at the barrier before any proceeds.
            barrier.wait().await;

            let (tx, rx) = tokio::sync::watch::channel(ModelLoadResult::Pending);
            let waiter_rx = try_become_loader(&inflight, "test-model", &tx, rx);

            if waiter_rx.is_none() {
                // We are the loader.
                loader_count.fetch_add(1, Ordering::SeqCst);
                // Simulate model load.
                tokio::task::yield_now().await;
                // Signal success and clean up.
                let _ = tx.send(ModelLoadResult::Ok);
                inflight.lock().unwrap().remove("test-model");
                true // is_loader
            } else {
                // We are a waiter.
                let mut rx = waiter_rx.unwrap();
                let _ = rx.changed().await;
                let result = rx.borrow().clone();
                match result {
                    ModelLoadResult::Ok => false,
                    other => panic!("expected Ok, got {:?}", other),
                }
            }
        }));
    }

    let mut loaders = 0;
    let mut waiters = 0;
    for handle in handles {
        if handle.await.unwrap() {
            loaders += 1;
        } else {
            waiters += 1;
        }
    }

    assert_eq!(loaders, 1, "exactly one task should become the loader");
    assert_eq!(waiters, 4, "remaining tasks should wait");
    assert_eq!(loader_count.load(Ordering::SeqCst), 1);
    assert!(
        inflight.lock().unwrap().is_empty(),
        "inflight map should be cleaned up after load completes"
    );
}

/// Loader fails with an error. All waiters receive the same error message.
#[tokio::test]
async fn singleflight_propagates_load_failure_to_waiters() {
    let inflight: Arc<
        std::sync::Mutex<
            std::collections::HashMap<String, tokio::sync::watch::Receiver<ModelLoadResult>>,
        >,
    > = Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));

    let barrier = Arc::new(tokio::sync::Barrier::new(3));

    let mut handles = Vec::new();
    for _ in 0..3 {
        let inflight = inflight.clone();
        let barrier = barrier.clone();

        handles.push(tokio::spawn(async move {
            barrier.wait().await;

            let (tx, rx) = tokio::sync::watch::channel(ModelLoadResult::Pending);
            let waiter_rx = try_become_loader(&inflight, "bad-model", &tx, rx);

            if waiter_rx.is_none() {
                // Loader: simulate failure.
                tokio::task::yield_now().await;
                let _ = tx.send(ModelLoadResult::Err("gpu on fire".to_string()));
                inflight.lock().unwrap().remove("bad-model");
                None // loader returns no error string
            } else {
                // Waiter: observe the error.
                let mut rx = waiter_rx.unwrap();
                let _ = rx.changed().await;
                let result = rx.borrow().clone();
                match result {
                    ModelLoadResult::Err(msg) => Some(msg),
                    other => panic!("expected Err, got {:?}", other),
                }
            }
        }));
    }

    let mut loader_seen = false;
    for handle in handles {
        match handle.await.unwrap() {
            None => {
                assert!(!loader_seen, "only one loader expected");
                loader_seen = true;
            }
            Some(msg) => {
                assert_eq!(msg, "gpu on fire");
            }
        }
    }
    assert!(loader_seen, "one task should have been the loader");
    assert!(inflight.lock().unwrap().is_empty());
}

/// Two different model IDs load concurrently without cross-blocking.
#[tokio::test]
async fn singleflight_allows_independent_loads_for_different_models() {
    let inflight: Arc<
        std::sync::Mutex<
            std::collections::HashMap<String, tokio::sync::watch::Receiver<ModelLoadResult>>,
        >,
    > = Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));

    let barrier = Arc::new(tokio::sync::Barrier::new(2));

    let mut handles = Vec::new();
    for model_id in &["model-a", "model-b"] {
        let inflight = inflight.clone();
        let barrier = barrier.clone();
        let model_id = model_id.to_string();

        handles.push(tokio::spawn(async move {
            barrier.wait().await;

            let (tx, rx) = tokio::sync::watch::channel(ModelLoadResult::Pending);
            let waiter_rx = try_become_loader(&inflight, &model_id, &tx, rx);

            assert!(
                waiter_rx.is_none(),
                "each model should get its own loader, not a waiter for {model_id}"
            );

            // Both are loaders for their respective models.
            tokio::task::yield_now().await;
            let _ = tx.send(ModelLoadResult::Ok);
            inflight.lock().unwrap().remove(model_id.as_str());
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }

    assert!(inflight.lock().unwrap().is_empty());
}
