//! Batch scheduler runtime for continuous batched GPU decode.
//!
//! Wraps `talu::batch::BatchHandle` with a dedicated step thread and
//! per-request event channels, allowing concurrent `/v1/responses` requests
//! to share the GPU without holding a backend mutex for the entire duration.
//!
//! # Architecture
//!
//! ```text
//!   HTTP handler → SchedulerState::submit() → cmd_tx → step thread
//!   HTTP handler ← rx (BatchEvent stream) ←──────────── step thread
//! ```
//!
//! The step thread owns the `BatchHandle` (which is NOT thread-safe) and
//! serializes all operations: submit, cancel, step.
//!
//! Completed results are stored in a shared map accessible to handlers
//! without going through the step thread, so results survive shutdown.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use talu::batch::{BatchConfig, BatchEvent, BatchHandle, BatchResult};
use talu::router::GenerateConfig;
use talu::InferenceBackend;

// =============================================================================
// Commands sent from HTTP handlers to the step thread
// =============================================================================

enum SchedulerCommand {
    Submit {
        /// Raw ChatHandle pointer, borrowed from the caller. Valid for the
        /// duration of the submit because the caller blocks on reply_rx.
        chat_ptr: *mut c_void,
        config: GenerateConfig,
        /// Event sender for this request — registered by the step thread
        /// before any step() call, preventing lost events.
        event_tx: std::sync::mpsc::Sender<BatchEvent>,
        stop_flag: Arc<AtomicBool>,
        reply: std::sync::mpsc::Sender<Result<u64>>,
    },
    Cancel {
        request_id: u64,
    },
    Shutdown,
}

// SAFETY: SchedulerCommand contains raw pointers (chat_ptr) and GenerateConfig
// which may contain Arc<AtomicBool>. The chat_ptr is only dereferenced on the
// step thread during Submit handling, and the caller blocks until that completes,
// guaranteeing the pointer remains valid. GenerateConfig's Arc fields are
// thread-safe by design.
unsafe impl Send for SchedulerCommand {}

// =============================================================================
// Per-request slot
// =============================================================================

struct RequestSlot {
    tx: std::sync::mpsc::Sender<BatchEvent>,
    stop_flag: Arc<AtomicBool>,
}

/// Entry in the completed-results map, with insertion time for stale eviction.
struct CompletedEntry {
    result: BatchResult,
    inserted_at: std::time::Instant,
}

/// Entries older than this are evicted from the completed map during idle sweeps.
/// Handlers normally call take_result() within milliseconds of the final event;
/// 60 seconds is generous enough for any realistic retrieval delay.
const COMPLETED_ENTRY_TTL: std::time::Duration = std::time::Duration::from_secs(60);

// =============================================================================
// Public API
// =============================================================================

/// Shared scheduler state. Cloneable via `Arc`.
///
/// HTTP handlers use this to submit/cancel requests and receive event streams.
/// The actual scheduler runs on a dedicated OS thread.
pub struct SchedulerState {
    cmd_tx: std::sync::mpsc::Sender<SchedulerCommand>,
    step_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// Completed request results. Populated by the step thread when a
    /// request's final event is dispatched; read by handlers via
    /// `take_result()`. This decouples result retrieval from the step
    /// thread's lifetime, so results survive graceful shutdown.
    completed: Arc<Mutex<HashMap<u64, CompletedEntry>>>,
}

impl SchedulerState {
    /// Create a scheduler bound to a local inference backend.
    ///
    /// Spawns a dedicated OS thread for the step loop.
    pub fn new(backend: &InferenceBackend, config: Option<&BatchConfig>) -> Result<Self> {
        let batch = BatchHandle::new(backend, config)
            .map_err(|e| anyhow!("failed to create batch handle: {}", e))?;

        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<SchedulerCommand>();
        let requests: Arc<Mutex<HashMap<u64, RequestSlot>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let completed: Arc<Mutex<HashMap<u64, CompletedEntry>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let completed_clone = completed.clone();

        let handle = std::thread::Builder::new()
            .name("batch-scheduler".into())
            .spawn(move || {
                step_loop(batch, cmd_rx, requests, completed_clone);
            })
            .map_err(|e| anyhow!("failed to spawn scheduler thread: {}", e))?;

        Ok(Self {
            cmd_tx,
            step_thread: Mutex::new(Some(handle)),
            completed,
        })
    }

    /// Submit a generation request.
    ///
    /// Borrows the `ChatHandle` for tokenization only — the handle remains
    /// available to the caller after this returns (e.g. for serialization).
    ///
    /// Returns `(request_id, event_receiver)`. The receiver yields `BatchEvent`s
    /// until the final event (is_final=true) or the channel closes.
    pub fn submit(
        &self,
        chat: &talu::ChatHandle,
        config: GenerateConfig,
        stop_flag: Arc<AtomicBool>,
    ) -> Result<(u64, std::sync::mpsc::Receiver<BatchEvent>)> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        let (event_tx, event_rx) = std::sync::mpsc::channel();

        // Send submit command with event_tx/stop_flag so the step thread
        // registers the slot BEFORE any step() can produce events.
        self.cmd_tx
            .send(SchedulerCommand::Submit {
                chat_ptr: chat.as_ptr(),
                config,
                event_tx,
                stop_flag,
                reply: reply_tx,
            })
            .map_err(|_| anyhow!("scheduler thread has shut down"))?;

        // Block until the step thread completes the submit. This guarantees
        // chat_ptr (borrowed from the caller) is no longer accessed.
        let request_id = reply_rx
            .recv()
            .map_err(|_| anyhow!("scheduler thread died before replying"))??;

        Ok((request_id, event_rx))
    }

    /// Cancel a running request.
    pub fn cancel(&self, request_id: u64) {
        let _ = self.cmd_tx.send(SchedulerCommand::Cancel { request_id });
    }

    /// Take the completion result for a finished request.
    ///
    /// Reads from the shared completed-results map, which is populated by
    /// the step thread on final events. This does NOT require the step
    /// thread to be alive — results survive scheduler shutdown.
    ///
    /// Returns `None` if the request hasn't completed or was already taken.
    pub fn take_result(&self, request_id: u64) -> Option<BatchResult> {
        self.completed.lock().ok()?.remove(&request_id).map(|e| e.result)
    }

    /// Shut down the scheduler. Blocks until the step thread exits.
    ///
    /// The step thread performs a graceful drain: it finishes all active
    /// requests before exiting, so in-flight handlers receive their
    /// final events and results.
    pub fn shutdown(&self) {
        let _ = self.cmd_tx.send(SchedulerCommand::Shutdown);
        if let Ok(mut guard) = self.step_thread.lock() {
            if let Some(handle) = guard.take() {
                let _ = handle.join();
            }
        }
    }
}

impl Drop for SchedulerState {
    fn drop(&mut self) {
        // Best-effort shutdown signal; don't block in drop.
        let _ = self.cmd_tx.send(SchedulerCommand::Shutdown);
    }
}

// =============================================================================
// Step loop (runs on dedicated OS thread)
// =============================================================================

fn step_loop(
    batch: BatchHandle,
    cmd_rx: std::sync::mpsc::Receiver<SchedulerCommand>,
    requests: Arc<Mutex<HashMap<u64, RequestSlot>>>,
    completed: Arc<Mutex<HashMap<u64, CompletedEntry>>>,
) {
    let mut events_buf = vec![BatchEvent::default(); 64];
    let mut draining = false;
    let mut step_count: u64 = 0;

    loop {
        // 1. Drain all pending commands (non-blocking).
        loop {
            match cmd_rx.try_recv() {
                Ok(SchedulerCommand::Shutdown) => {
                    // Graceful drain: finish active requests before exiting.
                    if !batch.has_active() {
                        return;
                    }
                    draining = true;
                    break; // Stop processing commands, start draining.
                }
                Ok(SchedulerCommand::Submit { reply, .. }) if draining => {
                    // Reject new submits during drain.
                    let _ = reply.send(Err(anyhow!("scheduler is shutting down")));
                }
                Ok(cmd) => {
                    handle_command(&batch, cmd, &requests);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    // All senders dropped. Drain remaining active requests.
                    if !batch.has_active() {
                        return;
                    }
                    draining = true;
                    break;
                }
            }
        }

        // 2. Check for cancelled requests (stop_flag set by client disconnect).
        check_stop_flags(&batch, &requests);

        // 3. If draining and no more active requests, exit.
        if draining && !batch.has_active() {
            return;
        }

        // 4. If idle (no active requests, not draining), sweep stale
        //    completed entries then block waiting for the next command.
        if !batch.has_active() {
            sweep_stale_completed(&completed);
            match cmd_rx.recv() {
                Ok(SchedulerCommand::Shutdown) => return,
                Ok(cmd) => {
                    handle_command(&batch, cmd, &requests);
                }
                Err(_) => return, // Channel disconnected
            }
            continue; // Re-drain commands before stepping
        }

        // 5. One decode step — produces events for all active requests.
        let count = match batch.step(&mut events_buf) {
            Ok(n) => n,
            Err(e) => {
                log::error!(target: "batch_scheduler", "step error: {}", e);
                0
            }
        };
        step_count += 1;

        // 5b. Periodic sweep under sustained load (every 256 steps).
        if step_count % 256 == 0 {
            sweep_stale_completed(&completed);
        }

        // 6. Dispatch events and clean up completed requests.
        //
        // For final events: store the result in `completed` BEFORE
        // dispatching the event. This guarantees that when the handler
        // receives is_final and calls take_result(), the result is
        // already available.
        if count > 0 {
            let mut reqs = requests.lock().unwrap();
            for event in &events_buf[..count] {
                if event.is_final {
                    if reqs.contains_key(&event.request_id) {
                        // Store result before the handler can observe the
                        // final event via the channel.
                        if let Some(result) = batch.take_result(event.request_id) {
                            if let Ok(mut comp) = completed.lock() {
                                comp.insert(event.request_id, CompletedEntry {
                                    result,
                                    inserted_at: std::time::Instant::now(),
                                });
                            }
                        }
                    } else {
                        // Slot already removed (cancelled/disconnected).
                        // Consume the result from batch to free it, but
                        // don't store — no handler will call take_result().
                        let _ = batch.take_result(event.request_id);
                    }
                }
                if let Some(slot) = reqs.get(&event.request_id) {
                    let _ = slot.tx.send(event.clone());
                }
                if event.is_final {
                    reqs.remove(&event.request_id);
                }
            }
        }
    }
}

/// Handle a single non-Shutdown command.
fn handle_command(
    batch: &BatchHandle,
    cmd: SchedulerCommand,
    requests: &Arc<Mutex<HashMap<u64, RequestSlot>>>,
) {
    match cmd {
        SchedulerCommand::Submit {
            chat_ptr,
            config,
            event_tx,
            stop_flag,
            reply,
        } => {
            // SAFETY: chat_ptr is a valid ChatHandle pointer borrowed from the
            // caller, who blocks on reply_rx.recv() until we send the reply.
            let result = batch
                .submit_raw(chat_ptr, &config)
                .map_err(|e| anyhow!("batch submit failed: {}", e));
            // Register the event slot BEFORE replying, so no step() can
            // produce events for this request before the slot exists.
            if let Ok(request_id) = &result {
                if let Ok(mut reqs) = requests.lock() {
                    reqs.insert(
                        *request_id,
                        RequestSlot {
                            tx: event_tx,
                            stop_flag,
                        },
                    );
                }
            }
            let _ = reply.send(result);
        }
        SchedulerCommand::Cancel { request_id } => {
            batch.cancel(request_id);
            if let Ok(mut reqs) = requests.lock() {
                reqs.remove(&request_id);
            }
        }
        SchedulerCommand::Shutdown => {
            // Handled in step_loop directly, not here.
        }
    }
}

/// Evict stale entries from the completed-results map.
fn sweep_stale_completed(completed: &Arc<Mutex<HashMap<u64, CompletedEntry>>>) {
    if let Ok(mut comp) = completed.lock() {
        let before = comp.len();
        comp.retain(|_, entry| entry.inserted_at.elapsed() < COMPLETED_ENTRY_TTL);
        let evicted = before - comp.len();
        if evicted > 0 {
            log::debug!(target: "batch_scheduler",
                "evicted {evicted} stale completed entries");
        }
    }
}

/// Check stop flags set by CancelOnDrop / client disconnect.
fn check_stop_flags(batch: &BatchHandle, requests: &Arc<Mutex<HashMap<u64, RequestSlot>>>) {
    let to_cancel: Vec<u64> = {
        let reqs = requests.lock().unwrap();
        reqs.iter()
            .filter(|(_, slot)| slot.stop_flag.load(Ordering::Acquire))
            .map(|(id, _)| *id)
            .collect()
    };

    for id in to_cancel {
        batch.cancel(id);
        if let Ok(mut reqs) = requests.lock() {
            reqs.remove(&id);
        }
    }
}
