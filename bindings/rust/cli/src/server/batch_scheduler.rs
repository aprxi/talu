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
//! serializes all operations: submit, cancel, step, take_result.

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
        reply: std::sync::mpsc::Sender<Result<u64>>,
    },
    Cancel {
        request_id: u64,
    },
    TakeResult {
        request_id: u64,
        reply: std::sync::mpsc::Sender<Option<BatchResult>>,
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

// =============================================================================
// Public API
// =============================================================================

/// Shared scheduler state. Cloneable via `Arc`.
///
/// HTTP handlers use this to submit/cancel requests and receive event streams.
/// The actual scheduler runs on a dedicated OS thread.
pub struct SchedulerState {
    cmd_tx: std::sync::mpsc::Sender<SchedulerCommand>,
    requests: Arc<Mutex<HashMap<u64, RequestSlot>>>,
    step_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
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
        let requests_clone = requests.clone();

        let handle = std::thread::Builder::new()
            .name("batch-scheduler".into())
            .spawn(move || {
                step_loop(batch, cmd_rx, requests_clone);
            })
            .map_err(|e| anyhow!("failed to spawn scheduler thread: {}", e))?;

        Ok(Self {
            cmd_tx,
            requests,
            step_thread: Mutex::new(Some(handle)),
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

        // Create the event channel before submitting so events can't be missed.
        let (event_tx, event_rx) = std::sync::mpsc::channel();

        self.cmd_tx
            .send(SchedulerCommand::Submit {
                chat_ptr: chat.as_ptr(),
                config,
                reply: reply_tx,
            })
            .map_err(|_| anyhow!("scheduler thread has shut down"))?;

        // Block until the step thread completes the submit. This guarantees
        // chat_ptr (borrowed from the caller) is no longer accessed.
        let request_id = reply_rx
            .recv()
            .map_err(|_| anyhow!("scheduler thread died before replying"))??;

        // Register the event channel for this request.
        if let Ok(mut reqs) = self.requests.lock() {
            reqs.insert(
                request_id,
                RequestSlot {
                    tx: event_tx,
                    stop_flag,
                },
            );
        }

        Ok((request_id, event_rx))
    }

    /// Cancel a running request.
    pub fn cancel(&self, request_id: u64) {
        let _ = self.cmd_tx.send(SchedulerCommand::Cancel { request_id });
    }

    /// Take the completion result for a finished request.
    ///
    /// Returns `None` if the request hasn't completed or was already taken.
    pub fn take_result(&self, request_id: u64) -> Option<BatchResult> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        self.cmd_tx
            .send(SchedulerCommand::TakeResult {
                request_id,
                reply: reply_tx,
            })
            .ok()?;
        reply_rx.recv().ok()?
    }

    /// Shut down the scheduler. Blocks until the step thread exits.
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
) {
    let mut events_buf = vec![BatchEvent::default(); 64];

    loop {
        // 1. Drain all pending commands (non-blocking).
        loop {
            match cmd_rx.try_recv() {
                Ok(cmd) => {
                    if handle_command(&batch, cmd, &requests) {
                        return; // Shutdown
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return,
            }
        }

        // 2. Check for cancelled requests (stop_flag set by client disconnect).
        check_stop_flags(&batch, &requests);

        // 3. If idle, block waiting for next command.
        if !batch.has_active() {
            match cmd_rx.recv() {
                Ok(cmd) => {
                    if handle_command(&batch, cmd, &requests) {
                        return; // Shutdown
                    }
                }
                Err(_) => return, // Channel disconnected
            }
            continue; // Re-drain commands before stepping
        }

        // 4. One decode step — produces events for all active requests.
        let count = match batch.step(&mut events_buf) {
            Ok(n) => n,
            Err(e) => {
                log::error!(target: "batch_scheduler", "step error: {}", e);
                0
            }
        };

        // 5. Dispatch events to per-request channels.
        if count > 0 {
            let reqs = requests.lock().unwrap();
            for event in &events_buf[..count] {
                if let Some(slot) = reqs.get(&event.request_id) {
                    let _ = slot.tx.send(event.clone());
                }
            }
        }

        // 6. Clean up completed requests.
        {
            let mut reqs = requests.lock().unwrap();
            for event in &events_buf[..count] {
                if event.is_final {
                    reqs.remove(&event.request_id);
                }
            }
        }
    }
}

/// Handle a single command. Returns `true` if the loop should exit (Shutdown).
fn handle_command(
    batch: &BatchHandle,
    cmd: SchedulerCommand,
    requests: &Arc<Mutex<HashMap<u64, RequestSlot>>>,
) -> bool {
    match cmd {
        SchedulerCommand::Submit {
            chat_ptr,
            config,
            reply,
        } => {
            // SAFETY: chat_ptr is a valid ChatHandle pointer borrowed from the
            // caller, who blocks on reply_rx.recv() until we send the reply.
            let result = batch
                .submit_raw(chat_ptr, &config)
                .map_err(|e| anyhow!("batch submit failed: {}", e));
            let _ = reply.send(result);
        }
        SchedulerCommand::Cancel { request_id } => {
            batch.cancel(request_id);
            if let Ok(mut reqs) = requests.lock() {
                reqs.remove(&request_id);
            }
        }
        SchedulerCommand::TakeResult { request_id, reply } => {
            let result = batch.take_result(request_id);
            let _ = reply.send(result);
        }
        SchedulerCommand::Shutdown => {
            return true;
        }
    }
    false
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
