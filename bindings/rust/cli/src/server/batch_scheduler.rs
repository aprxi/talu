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
use std::time::Duration;

use anyhow::{anyhow, Result};
use talu::batch::{BatchConfig, BatchEvent, BatchHandle, BatchResult, EventType};
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
        /// Whether to forward incremental token events (`true`) or only
        /// the terminal event (`false`).
        ///
        /// Non-streaming handlers should set this to `false` to avoid per-token
        /// channel overhead while still receiving completion notification.
        stream_events: bool,
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
    stream_events: bool,
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

/// Brief coalescing window after idle wake-up to collect near-simultaneous
/// submits before entering prefill.
///
/// Keep a modest default window so bursty arrivals can join the first prefill
/// wave instead of fragmenting into serial waves. The N=1 path remains gated
/// by `INITIAL_SUBMIT_PAIR_WAIT_MS`, so single-request latency impact stays low.
const DEFAULT_SUBMIT_COALESCE_MS: u64 = 128;
/// Initial idle-wake wait for a second submit before deciding whether this is
/// a true burst. Kept short to avoid adding noticeable N=1 latency.
const INITIAL_SUBMIT_PAIR_WAIT_MS: u64 = 32;
/// In non-streaming mode, poll pending submit/cancel commands at this token
/// cadence while decode is active. Polling every token causes run_loop churn
/// and collapses effective batching under steady submit pressure.
const DEFAULT_NON_STREAM_CMD_POLL_TOKENS: usize = 4;

/// Maximum total in-flight requests (active + pending) accepted by the
/// server-side scheduler wrapper. `0` means unbounded.
///
/// This caps host-side request accumulation under overload independently of
/// backend slot capacity. It is a queueing/backpressure control and does NOT
/// by itself guarantee any specific GPU memory bound.
fn resolve_batch_max_inflight() -> usize {
    let raw = match std::env::var("TALU_BATCH_MAX_INFLIGHT") {
        Ok(v) => v,
        Err(_) => return 0,
    };
    let trimmed = raw.trim();
    match trimmed.parse::<usize>() {
        Ok(v) => v,
        Err(_) => {
            log::warn!(target: "batch_scheduler",
                "invalid TALU_BATCH_MAX_INFLIGHT='{}'; using 0 (unbounded)", trimmed);
            0
        }
    }
}

fn resolve_batch_submit_coalesce() -> Duration {
    let raw = match std::env::var("TALU_BATCH_SUBMIT_COALESCE_MS") {
        Ok(v) => v,
        Err(_) => return Duration::from_millis(DEFAULT_SUBMIT_COALESCE_MS),
    };
    let trimmed = raw.trim();
    match trimmed.parse::<u64>() {
        Ok(v) => Duration::from_millis(v),
        Err(_) => {
            log::warn!(
                target: "batch_scheduler",
                "invalid TALU_BATCH_SUBMIT_COALESCE_MS='{}'; using {}ms",
                trimmed,
                DEFAULT_SUBMIT_COALESCE_MS
            );
            Duration::from_millis(DEFAULT_SUBMIT_COALESCE_MS)
        }
    }
}

/// Maximum number of submit commands to process in one command-drain cycle
/// while the scheduler already has active/pending work.
///
/// Lower values reduce decode stalls from submit-side CPU work; higher values
/// admit large bursts faster. Must be >=1.
fn resolve_active_submit_budget() -> usize {
    let raw = match std::env::var("TALU_BATCH_ACTIVE_SUBMIT_BUDGET") {
        Ok(v) => v,
        Err(_) => return 8,
    };
    let trimmed = raw.trim();
    match trimmed.parse::<usize>() {
        Ok(0) => usize::MAX,
        Ok(v) if v >= 1 => v,
        Ok(_) | Err(_) => {
            log::warn!(
                target: "batch_scheduler",
                "invalid TALU_BATCH_ACTIVE_SUBMIT_BUDGET='{}'; using unbounded",
                trimmed
            );
            usize::MAX
        }
    }
}

/// Number of token callbacks between command-pending polls in non-streaming
/// run_loop mode.
fn resolve_non_stream_cmd_poll_tokens() -> usize {
    let raw = match std::env::var("TALU_BATCH_NON_STREAM_CMD_POLL_TOKENS") {
        Ok(v) => v,
        Err(_) => return DEFAULT_NON_STREAM_CMD_POLL_TOKENS,
    };
    let trimmed = raw.trim();
    match trimmed.parse::<usize>() {
        Ok(0) => 1,
        Ok(v) => v,
        Err(_) => {
            log::warn!(
                target: "batch_scheduler",
                "invalid TALU_BATCH_NON_STREAM_CMD_POLL_TOKENS='{}'; using {}",
                trimmed,
                DEFAULT_NON_STREAM_CMD_POLL_TOKENS
            );
            DEFAULT_NON_STREAM_CMD_POLL_TOKENS
        }
    }
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
    step_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// Completed request results. Populated by the step thread when a
    /// request's final event is dispatched; read by handlers via
    /// `take_result()`. This decouples result retrieval from the step
    /// thread's lifetime, so results survive graceful shutdown.
    completed: Arc<Mutex<HashMap<u64, CompletedEntry>>>,
    /// Signalled by senders (submit/cancel/shutdown) before enqueueing a
    /// command. The decode-loop callback checks this and sets the Zig
    /// `pending_flag` to break out, so the step thread can drain commands.
    cmd_pending: Arc<AtomicBool>,
}

impl SchedulerState {
    /// Create a scheduler bound to a local inference backend.
    ///
    /// Spawns a dedicated OS thread for the step loop.
    pub fn new(backend: &InferenceBackend, config: Option<&BatchConfig>) -> Result<Self> {
        let batch = BatchHandle::new(backend, config)
            .map_err(|e| anyhow!("failed to create batch handle: {}", e))?;
        let max_inflight = resolve_batch_max_inflight();
        let submit_coalesce = resolve_batch_submit_coalesce();
        let active_submit_budget = resolve_active_submit_budget();
        let non_stream_cmd_poll_tokens = resolve_non_stream_cmd_poll_tokens();
        if max_inflight > 0 {
            log::info!(target: "batch_scheduler",
                "TALU_BATCH_MAX_INFLIGHT={} (server in-flight cap)", max_inflight);
        } else {
            log::info!(target: "batch_scheduler",
                "TALU_BATCH_MAX_INFLIGHT=0 (unbounded in-flight)");
        }
        log::info!(
            target: "batch_scheduler",
            "TALU_BATCH_SUBMIT_COALESCE_MS={} (idle burst coalescing)",
            submit_coalesce.as_millis()
        );
        if active_submit_budget == usize::MAX {
            log::info!(
                target: "batch_scheduler",
                "TALU_BATCH_ACTIVE_SUBMIT_BUDGET=unbounded (submit commands per active drain)"
            );
        } else {
            log::info!(
                target: "batch_scheduler",
                "TALU_BATCH_ACTIVE_SUBMIT_BUDGET={} (submit commands per active drain)",
                active_submit_budget
            );
        }
        log::info!(
            target: "batch_scheduler",
            "TALU_BATCH_NON_STREAM_CMD_POLL_TOKENS={} (non-stream command poll cadence)",
            non_stream_cmd_poll_tokens
        );

        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<SchedulerCommand>();
        let requests: Arc<Mutex<HashMap<u64, RequestSlot>>> = Arc::new(Mutex::new(HashMap::new()));
        let completed: Arc<Mutex<HashMap<u64, CompletedEntry>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let cmd_pending = Arc::new(AtomicBool::new(false));
        let completed_clone = completed.clone();
        let cmd_pending_clone = cmd_pending.clone();
        let handle = std::thread::Builder::new()
            .name("batch-scheduler".into())
            .spawn(move || {
                step_loop(
                    batch,
                    cmd_rx,
                    requests,
                    completed_clone,
                    max_inflight,
                    cmd_pending_clone,
                    submit_coalesce,
                    active_submit_budget,
                    non_stream_cmd_poll_tokens,
                );
            })
            .map_err(|e| anyhow!("failed to spawn scheduler thread: {}", e))?;

        Ok(Self {
            cmd_tx,
            step_thread: Mutex::new(Some(handle)),
            completed,
            cmd_pending,
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
        self.submit_with_event_mode(chat, config, stop_flag, true)
    }

    /// Submit a generation request and receive only a terminal event.
    ///
    /// This keeps batched execution but avoids forwarding per-token events,
    /// reducing channel/clone overhead for non-streaming endpoints.
    pub fn submit_final_only(
        &self,
        chat: &talu::ChatHandle,
        config: GenerateConfig,
        stop_flag: Arc<AtomicBool>,
    ) -> Result<(u64, std::sync::mpsc::Receiver<BatchEvent>)> {
        self.submit_with_event_mode(chat, config, stop_flag, false)
    }

    fn submit_with_event_mode(
        &self,
        chat: &talu::ChatHandle,
        config: GenerateConfig,
        stop_flag: Arc<AtomicBool>,
        stream_events: bool,
    ) -> Result<(u64, std::sync::mpsc::Receiver<BatchEvent>)> {
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        let (event_tx, event_rx) = std::sync::mpsc::channel();

        // Signal the step thread to break out of the Zig decode loop so it
        // can drain this command promptly.
        self.cmd_pending.store(true, Ordering::Release);

        // Send submit command with event_tx/stop_flag so the step thread
        // registers the slot BEFORE any step() can produce events.
        self.cmd_tx
            .send(SchedulerCommand::Submit {
                chat_ptr: chat.as_ptr(),
                config,
                event_tx,
                stream_events,
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
        self.cmd_pending.store(true, Ordering::Release);
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
        self.completed
            .lock()
            .ok()?
            .remove(&request_id)
            .map(|e| e.result)
    }

    /// Shut down the scheduler. Blocks until the step thread exits.
    ///
    /// The step thread performs a graceful drain: it finishes all active
    /// requests before exiting, so in-flight handlers receive their
    /// final events and results.
    pub fn shutdown(&self) {
        self.cmd_pending.store(true, Ordering::Release);
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
        self.cmd_pending.store(true, Ordering::Release);
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
    max_inflight: usize,
    cmd_pending: Arc<AtomicBool>,
    submit_coalesce: Duration,
    active_submit_budget: usize,
    non_stream_cmd_poll_tokens: usize,
) {
    // Local flag passed into the Zig decode loop. The event callback
    // sets this to break out of the loop when there are commands to
    // drain or stop-flagged requests to cancel.
    let pending_flag = AtomicBool::new(false);
    let mut draining = false;
    let mut step_count: u64 = 0;
    let non_stream_cmd_poll_tokens_u32: u32 = non_stream_cmd_poll_tokens
        .try_into()
        .unwrap_or(u32::MAX);

    loop {
        // 1. Clear pending flags, then drain all queued commands.
        cmd_pending.store(false, Ordering::Release);
        pending_flag.store(false, Ordering::Release);
        let had_work_in_scheduler = batch.has_active();
        let mut serviced_submit_count = 0usize;

        loop {
            match cmd_rx.try_recv() {
                Ok(SchedulerCommand::Shutdown) => {
                    if !batch.has_active() {
                        return;
                    }
                    log_scheduler_snapshot(&batch, &requests, &completed, max_inflight);
                    draining = true;
                    break;
                }
                Ok(SchedulerCommand::Submit { reply, .. }) if draining => {
                    let tracked = requests.lock().ok().map(|r| r.len()).unwrap_or(0);
                    let active = batch.active_count();
                    let pending = tracked.saturating_sub(active);
                    log::warn!(target: "batch_scheduler",
                        "rejecting submit during drain (tracked={}, active={}, pending={})",
                        tracked, active, pending);
                    let _ = reply.send(Err(anyhow!(
                        "scheduler is shutting down (tracked={}, active={}, pending={})",
                        tracked,
                        active,
                        pending
                    )));
                }
                Ok(cmd) => {
                    let is_submit = matches!(cmd, SchedulerCommand::Submit { .. });
                    handle_command(&batch, cmd, &requests, max_inflight);
                    if had_work_in_scheduler && is_submit {
                        serviced_submit_count = serviced_submit_count.saturating_add(1);
                    }
                    // Under load, avoid draining a long submit queue in one
                    // go: each submit does prompt/template CPU work, and
                    // draining all of them can stall decode for hundreds of ms.
                    // Process a bounded number of submits, return to run_loop,
                    // and let further submits join on subsequent token rounds.
                    if active_submit_budget != usize::MAX
                        && serviced_submit_count >= active_submit_budget
                    {
                        break;
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    if !batch.has_active() {
                        return;
                    }
                    log_scheduler_snapshot(&batch, &requests, &completed, max_inflight);
                    draining = true;
                    break;
                }
            }
        }

        // 2. If draining and no more active requests, exit.
        if draining && !batch.has_active() {
            return;
        }

        // 3. If idle (no active requests, not draining), sweep stale
        //    completed entries then block waiting for the next command.
        if !batch.has_active() {
            sweep_stale_completed(&completed);
            let first_command_was_submit: bool;
            match cmd_rx.recv() {
                Ok(SchedulerCommand::Shutdown) => return,
                Ok(cmd) => {
                    first_command_was_submit = matches!(cmd, SchedulerCommand::Submit { .. });
                    handle_command(&batch, cmd, &requests, max_inflight);
                }
                Err(_) => return,
            }

            // After waking from idle on the first command, briefly keep
            // draining command arrivals so near-simultaneous submits enter the
            // same first prefill wave instead of waiting behind it.
            //
            // Important: handle_command(Submit) performs template/tokenize CPU
            // work, so wall-time windows must not cut off burst coalescing just
            // because command processing itself took time.
            if !draining && !submit_coalesce.is_zero() {
                let mut observed_submit_count: usize = if first_command_was_submit { 1 } else { 0 };
                let mut idle_submit_budget_remaining = if active_submit_budget == usize::MAX {
                    usize::MAX
                } else {
                    active_submit_budget.saturating_sub(if first_command_was_submit { 1 } else { 0 })
                };

                // Drain already-queued commands first (not time-gated).
                while !draining {
                    if idle_submit_budget_remaining == 0 {
                        break;
                    }
                    match cmd_rx.try_recv() {
                        Ok(SchedulerCommand::Shutdown) => {
                            if !batch.has_active() {
                                return;
                            }
                            log_scheduler_snapshot(&batch, &requests, &completed, max_inflight);
                            draining = true;
                            break;
                        }
                        Ok(SchedulerCommand::Submit { reply, .. }) if draining => {
                            let tracked = requests.lock().ok().map(|r| r.len()).unwrap_or(0);
                            let active = batch.active_count();
                            let pending = tracked.saturating_sub(active);
                            let _ = reply.send(Err(anyhow!(
                                "scheduler is shutting down (tracked={}, active={}, pending={})",
                                tracked,
                                active,
                                pending
                            )));
                        }
                        Ok(cmd) => {
                            if matches!(cmd, SchedulerCommand::Submit { .. }) {
                                observed_submit_count += 1;
                                if idle_submit_budget_remaining != usize::MAX {
                                    idle_submit_budget_remaining =
                                        idle_submit_budget_remaining.saturating_sub(1);
                                }
                            }
                            handle_command(&batch, cmd, &requests, max_inflight);
                        }
                        Err(std::sync::mpsc::TryRecvError::Empty) => break,
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            if !batch.has_active() {
                                return;
                            }
                            log_scheduler_snapshot(&batch, &requests, &completed, max_inflight);
                            draining = true;
                            break;
                        }
                    }
                }

                // If we saw only one submit, wait briefly for a second one.
                // This preserves N=1 responsiveness while still capturing the
                // common "several requests arrive almost together" burst.
                if !draining && observed_submit_count == 1 && idle_submit_budget_remaining != 0 {
                    let pair_wait =
                        submit_coalesce.min(Duration::from_millis(INITIAL_SUBMIT_PAIR_WAIT_MS));
                    if !pair_wait.is_zero() {
                        match cmd_rx.recv_timeout(pair_wait) {
                            Ok(SchedulerCommand::Shutdown) => {
                                if !batch.has_active() {
                                    return;
                                }
                                log_scheduler_snapshot(&batch, &requests, &completed, max_inflight);
                                draining = true;
                            }
                            Ok(SchedulerCommand::Submit { reply, .. }) if draining => {
                                let tracked = requests.lock().ok().map(|r| r.len()).unwrap_or(0);
                                let active = batch.active_count();
                                let pending = tracked.saturating_sub(active);
                                let _ = reply.send(Err(anyhow!(
                                    "scheduler is shutting down (tracked={}, active={}, pending={})",
                                    tracked,
                                    active,
                                    pending
                                )));
                            }
                            Ok(cmd) => {
                                if matches!(cmd, SchedulerCommand::Submit { .. }) {
                                    observed_submit_count += 1;
                                    if idle_submit_budget_remaining != usize::MAX {
                                        idle_submit_budget_remaining =
                                            idle_submit_budget_remaining.saturating_sub(1);
                                    }
                                }
                                handle_command(&batch, cmd, &requests, max_inflight);
                            }
                            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                                if !batch.has_active() {
                                    return;
                                }
                                log_scheduler_snapshot(&batch, &requests, &completed, max_inflight);
                                draining = true;
                            }
                        }
                    }
                }

                // Extend to a short rolling coalesce window only after we've
                // seen a real submit burst (>=2 submits). Keep draining queued
                // commands first, then wait up to `submit_coalesce` for the
                // next arrival; stop on timeout.
                while !draining && observed_submit_count >= 2 && idle_submit_budget_remaining != 0 {
                    loop {
                        if idle_submit_budget_remaining == 0 || draining {
                            break;
                        }
                        match cmd_rx.try_recv() {
                            Ok(SchedulerCommand::Shutdown) => {
                                if !batch.has_active() {
                                    return;
                                }
                                log_scheduler_snapshot(&batch, &requests, &completed, max_inflight);
                                draining = true;
                                break;
                            }
                            Ok(SchedulerCommand::Submit { reply, .. }) if draining => {
                                let tracked = requests.lock().ok().map(|r| r.len()).unwrap_or(0);
                                let active = batch.active_count();
                                let pending = tracked.saturating_sub(active);
                                let _ = reply.send(Err(anyhow!(
                                    "scheduler is shutting down (tracked={}, active={}, pending={})",
                                    tracked,
                                    active,
                                    pending
                                )));
                            }
                            Ok(cmd) => {
                                if matches!(cmd, SchedulerCommand::Submit { .. }) {
                                    observed_submit_count += 1;
                                    if idle_submit_budget_remaining != usize::MAX {
                                        idle_submit_budget_remaining =
                                            idle_submit_budget_remaining.saturating_sub(1);
                                    }
                                }
                                handle_command(&batch, cmd, &requests, max_inflight);
                            }
                            Err(std::sync::mpsc::TryRecvError::Empty) => break,
                            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                                if !batch.has_active() {
                                    return;
                                }
                                log_scheduler_snapshot(&batch, &requests, &completed, max_inflight);
                                draining = true;
                                break;
                            }
                        }
                    }
                    if draining || idle_submit_budget_remaining == 0 {
                        break;
                    }
                    match cmd_rx.recv_timeout(submit_coalesce) {
                        Ok(SchedulerCommand::Shutdown) => {
                            if !batch.has_active() {
                                return;
                            }
                            log_scheduler_snapshot(&batch, &requests, &completed, max_inflight);
                            draining = true;
                            break;
                        }
                        Ok(SchedulerCommand::Submit { reply, .. }) if draining => {
                            let tracked = requests.lock().ok().map(|r| r.len()).unwrap_or(0);
                            let active = batch.active_count();
                            let pending = tracked.saturating_sub(active);
                            let _ = reply.send(Err(anyhow!(
                                "scheduler is shutting down (tracked={}, active={}, pending={})",
                                tracked,
                                active,
                                pending
                            )));
                        }
                        Ok(cmd) => {
                            if matches!(cmd, SchedulerCommand::Submit { .. }) {
                                observed_submit_count += 1;
                                if idle_submit_budget_remaining != usize::MAX {
                                    idle_submit_budget_remaining =
                                        idle_submit_budget_remaining.saturating_sub(1);
                                }
                            }
                            handle_command(&batch, cmd, &requests, max_inflight);
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => break,
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                            if !batch.has_active() {
                                return;
                            }
                            log_scheduler_snapshot(&batch, &requests, &completed, max_inflight);
                            draining = true;
                            break;
                        }
                    }
                }
            }
            continue;
        }

        // 4. Tight decode loop in Zig — replaces per-token step()+FFI.
        //
        // The callback dispatches events and checks stop flags. When
        // it detects a cancellation or a new command on the channel,
        // it sets pending_flag to break the Zig loop. Cancellations
        // are collected and applied after run_loop returns.
        {
            let mut to_cancel: Vec<u64> = Vec::new();
            let mut deferred_finals: Vec<(BatchEvent, std::sync::mpsc::Sender<BatchEvent>)> =
                Vec::new();
            let has_stream_subscribers = requests
                .lock()
                .ok()
                .map(|reqs| reqs.values().any(|slot| slot.stream_events))
                .unwrap_or(true);

            let mut non_stream_scan_ticks: u32 = 0;
            let mut dispatch_event = |event: &BatchEvent, to_cancel: &mut Vec<u64>| {
                step_count += 1;

                // Non-streaming hot path: most events are token deltas that do
                // not need per-request channel dispatch. Keep command wakeups
                // immediate while avoiding per-token map lock/scans.
                if !has_stream_subscribers && !event.is_final {
                    non_stream_scan_ticks = non_stream_scan_ticks.wrapping_add(1);
                    // Poll command-pending at a bounded cadence to avoid
                    // breaking the Zig run loop on every token when submits are
                    // continuously arriving.
                    if cmd_pending.load(Ordering::Acquire)
                        && (non_stream_scan_ticks % non_stream_cmd_poll_tokens_u32 == 0)
                    {
                        pending_flag.store(true, Ordering::Release);
                    }

                    // Periodic cancel sweep to honor stop flags without
                    // imposing O(active) work on every token callback.
                    if (non_stream_scan_ticks & 0x0f) == 0 {
                        if let Ok(mut reqs) = requests.lock() {
                            let cancelled: Vec<u64> = reqs
                                .iter()
                                .filter(|(_, slot)| slot.stop_flag.load(Ordering::Acquire))
                                .map(|(id, _)| *id)
                                .collect();
                            for id in &cancelled {
                                reqs.remove(id);
                                to_cancel.push(*id);
                            }
                        }
                    }
                    if !to_cancel.is_empty() {
                        pending_flag.store(true, Ordering::Release);
                    }
                    return;
                }

                // --- Dispatch event + check stop flags ---
                let mut reqs = requests.lock().unwrap();

                if event.is_final {
                    if let Some(slot) = reqs.remove(&event.request_id) {
                        // Do not call batch.take_result() from inside the
                        // run_loop callback. BatchWrapper callback paths are
                        // non-reentrant by contract.
                        deferred_finals.push((event.clone(), slot.tx));
                    }
                } else if let Some(slot) = reqs.get(&event.request_id) {
                    if slot.stream_events {
                        let _ = slot.tx.send(event.clone());
                    }
                }

                // Check stop flags for client disconnects.
                let cancelled: Vec<u64> = reqs
                    .iter()
                    .filter(|(_, slot)| slot.stop_flag.load(Ordering::Acquire))
                    .map(|(id, _)| *id)
                    .collect();
                for id in &cancelled {
                    reqs.remove(id);
                    to_cancel.push(*id);
                }

                // Break the Zig loop if there are cancellations or
                // new commands waiting on the channel.
                if !to_cancel.is_empty() || cmd_pending.load(Ordering::Acquire) {
                    pending_flag.store(true, Ordering::Release);
                }
            };

            let result = if has_stream_subscribers {
                batch.run_loop(&pending_flag, |event| dispatch_event(event, &mut to_cancel))
            } else {
                batch.run_loop_no_text(&pending_flag, |event| dispatch_event(event, &mut to_cancel))
            };

            // Apply collected cancellations now that we're outside
            // the Zig loop (safe to call cancel on the batch handle).
            for id in to_cancel {
                batch.cancel(id);
            }

            // Materialize and dispatch deferred finals now that we are outside
            // the non-reentrant run_loop callback.
            for (event, tx) in deferred_finals {
                if let Some(result) = batch.take_result(event.request_id) {
                    if let Ok(mut comp) = completed.lock() {
                        comp.insert(
                            event.request_id,
                            CompletedEntry {
                                result,
                                inserted_at: std::time::Instant::now(),
                            },
                        );
                    }
                } else {
                    log::warn!(
                        target: "batch_scheduler",
                        "final event without result: request_id={} tracked={} active={}",
                        event.request_id,
                        requests.lock().ok().map(|r| r.len()).unwrap_or(0),
                        batch.active_count()
                    );
                }
                let _ = tx.send(event);
            }

            if let Err(e) = result {
                log::error!(target: "batch_scheduler", "run_loop error: {}", e);
                fail_tracked_requests_after_run_loop_error(&batch, &requests, &e.to_string());
            }
        }

        // 5. Periodic sweep under sustained load.
        if step_count % 256 == 0 {
            sweep_stale_completed(&completed);
        }
    }
}

fn fail_tracked_requests_after_run_loop_error(
    batch: &BatchHandle,
    requests: &Arc<Mutex<HashMap<u64, RequestSlot>>>,
    error_message: &str,
) {
    let tracked_ids: Vec<u64> = requests
        .lock()
        .ok()
        .map(|reqs| reqs.keys().copied().collect())
        .unwrap_or_default();

    for id in &tracked_ids {
        batch.cancel(*id);
        let _ = batch.take_result(*id);
    }

    let drained: Vec<(u64, RequestSlot)> = requests
        .lock()
        .ok()
        .map(|mut reqs| reqs.drain().collect())
        .unwrap_or_default();

    for (request_id, slot) in drained {
        let _ = slot.tx.send(BatchEvent {
            request_id,
            event_type: EventType::Error,
            item_type: 0,
            content_type: 0,
            is_final: true,
            text: error_message.to_string(),
            token_id: 0,
            tokens_generated: 0,
            timestamp_ns: 0,
        });
    }
}

/// Handle a single non-Shutdown command.
fn handle_command(
    batch: &BatchHandle,
    cmd: SchedulerCommand,
    requests: &Arc<Mutex<HashMap<u64, RequestSlot>>>,
    max_inflight: usize,
) {
    match cmd {
        SchedulerCommand::Submit {
            chat_ptr,
            config,
            event_tx,
            stream_events,
            stop_flag,
            reply,
        } => {
            // SAFETY: chat_ptr is a valid ChatHandle pointer borrowed from the
            // caller, who blocks on reply_rx.recv() until we send the reply.
            if max_inflight > 0 {
                let tracked = match requests.lock() {
                    Ok(reqs) => reqs.len(),
                    Err(_) => {
                        let _ = reply.send(Err(anyhow!(
                            "scheduler internal error: request registry unavailable"
                        )));
                        return;
                    }
                };
                if tracked >= max_inflight {
                    let active = batch.active_count();
                    let pending = tracked.saturating_sub(active);
                    log::warn!(target: "batch_scheduler",
                        "rejecting submit: in-flight limit reached \
                         (limit={}, tracked={}, active={}, pending={})",
                        max_inflight, tracked, active, pending);
                    let _ = reply.send(Err(anyhow!(
                        "batch queue is full (limit={}, tracked={}, active={}, pending={})",
                        max_inflight,
                        tracked,
                        active,
                        pending
                    )));
                    return;
                }
            }
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
                            stream_events,
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

/// Debug snapshot of scheduler occupancy.
fn log_scheduler_snapshot(
    batch: &BatchHandle,
    requests: &Arc<Mutex<HashMap<u64, RequestSlot>>>,
    completed: &Arc<Mutex<HashMap<u64, CompletedEntry>>>,
    max_inflight: usize,
) {
    let tracked = requests.lock().ok().map(|r| r.len()).unwrap_or(0);
    let active = batch.active_count();
    let pending = tracked.saturating_sub(active);
    let completed_entries = completed.lock().ok().map(|c| c.len()).unwrap_or(0);
    log::debug!(target: "batch_scheduler",
        "snapshot: tracked={} active={} pending={} completed_cache={} inflight_cap={}",
        tracked, active, pending, completed_entries, max_inflight);
}

#[cfg(test)]
mod tests {
    use super::{
        resolve_active_submit_budget, resolve_batch_submit_coalesce,
        resolve_non_stream_cmd_poll_tokens,
    };
    use std::sync::Mutex;
    use std::time::Duration;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_env<R>(
        key: &str,
        value: Option<&str>,
        f: impl FnOnce() -> R,
    ) -> R {
        let _guard = ENV_LOCK.lock().expect("env lock");
        let prev = std::env::var(key).ok();
        match value {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
        let out = f();
        match prev {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
        out
    }

    #[test]
    fn active_submit_budget_defaults_to_eight() {
        let got = with_env("TALU_BATCH_ACTIVE_SUBMIT_BUDGET", None, resolve_active_submit_budget);
        assert_eq!(got, 8);
    }

    #[test]
    fn active_submit_budget_parses_valid_values() {
        let got = with_env(
            "TALU_BATCH_ACTIVE_SUBMIT_BUDGET",
            Some("4"),
            resolve_active_submit_budget,
        );
        assert_eq!(got, 4);
    }

    #[test]
    fn active_submit_budget_rejects_invalid_values() {
        let got_zero = with_env(
            "TALU_BATCH_ACTIVE_SUBMIT_BUDGET",
            Some("0"),
            resolve_active_submit_budget,
        );
        assert_eq!(got_zero, usize::MAX);

        let got_invalid = with_env(
            "TALU_BATCH_ACTIVE_SUBMIT_BUDGET",
            Some("abc"),
            resolve_active_submit_budget,
        );
        assert_eq!(got_invalid, usize::MAX);
    }

    #[test]
    fn submit_coalesce_defaults_to_one_hundred_twenty_eight_ms() {
        let got = with_env("TALU_BATCH_SUBMIT_COALESCE_MS", None, resolve_batch_submit_coalesce);
        assert_eq!(got, Duration::from_millis(128));
    }

    #[test]
    fn submit_coalesce_parses_valid_value() {
        let got = with_env(
            "TALU_BATCH_SUBMIT_COALESCE_MS",
            Some("7"),
            resolve_batch_submit_coalesce,
        );
        assert_eq!(got, Duration::from_millis(7));
    }

    #[test]
    fn non_stream_cmd_poll_tokens_defaults_to_four() {
        let got = with_env(
            "TALU_BATCH_NON_STREAM_CMD_POLL_TOKENS",
            None,
            resolve_non_stream_cmd_poll_tokens,
        );
        assert_eq!(got, 4);
    }

    #[test]
    fn non_stream_cmd_poll_tokens_parses_valid_value() {
        let got = with_env(
            "TALU_BATCH_NON_STREAM_CMD_POLL_TOKENS",
            Some("4"),
            resolve_non_stream_cmd_poll_tokens,
        );
        assert_eq!(got, 4);
    }

    #[test]
    fn non_stream_cmd_poll_tokens_clamps_zero_to_one() {
        let got = with_env(
            "TALU_BATCH_NON_STREAM_CMD_POLL_TOKENS",
            Some("0"),
            resolve_non_stream_cmd_poll_tokens,
        );
        assert_eq!(got, 1);
    }

    #[test]
    fn non_stream_cmd_poll_tokens_invalid_falls_back_to_default() {
        let got = with_env(
            "TALU_BATCH_NON_STREAM_CMD_POLL_TOKENS",
            Some("bad"),
            resolve_non_stream_cmd_poll_tokens,
        );
        assert_eq!(got, 4);
    }
}
