//! Responses-aware batch generation for continuous batching.
//!
//! Wraps the `talu_batch_*` C API with a safe RAII handle. The batch handle
//! owns a scheduler bound to a local inference backend and supports concurrent
//! generation requests with per-request state for reasoning tag filtering,
//! grammar-constrained sampling (tools/structured output), and timing.
//!
//! # Lifecycle
//!
//! ```text
//! create → (submit | cancel | step)* → drop
//! ```
//!
//! # Thread Safety
//!
//! The underlying scheduler is NOT thread-safe. All calls must be serialized.
//! In a server context, run the step loop on a dedicated thread and communicate
//! via channels.
//!
//! # Example
//!
//! ```no_run
//! use talu::batch::{BatchHandle, BatchConfig};
//! use talu::{ChatHandle, InferenceBackend};
//! use talu::router::GenerateConfig;
//!
//! let backend = InferenceBackend::new("path/to/model")?;
//! let batch = BatchHandle::new(&backend, None)?;
//!
//! let chat = ChatHandle::new(None)?;
//! // ... append messages to chat ...
//!
//! let config = GenerateConfig { max_tokens: 64, ..Default::default() };
//! let request_id = batch.submit(&chat, &config)?;
//!
//! // Step loop
//! let mut events = vec![BatchEvent::default(); 16];
//! loop {
//!     let n = batch.step(&mut events)?;
//!     for event in &events[..n] {
//!         // Process event (text delta, completion, etc.)
//!     }
//!     if !batch.has_active() { break; }
//! }
//!
//! // Retrieve completion result
//! if let Some(result) = batch.take_result(request_id) {
//!     println!("Generated {} tokens", result.completion_tokens);
//! }
//! # Ok::<(), talu::Error>(())
//! ```

use crate::error::error_from_last_or;
use crate::router::GenerateConfig;
use crate::{ChatHandle, InferenceBackend, Result};
use std::ffi::{c_void, CStr, CString};
use std::sync::atomic::AtomicBool;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for creating a batch handle.
#[derive(Debug, Clone, Default)]
pub struct BatchConfig {
    /// Maximum concurrent requests (0 = use backend default).
    pub max_concurrent: usize,
}

// =============================================================================
// Event Types
// =============================================================================

/// Event type discriminator for batch events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EventType {
    /// Decoded text delta.
    TextDelta = 0,
    /// Request completed (final event).
    Completed = 1,
    /// Request failed with error.
    Error = 2,
}

impl From<u8> for EventType {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::TextDelta,
            1 => Self::Completed,
            2 => Self::Error,
            _ => Self::Error,
        }
    }
}

/// A rich event from a batch step.
#[derive(Debug, Clone)]
pub struct BatchEvent {
    /// Request ID that produced this event.
    pub request_id: u64,
    /// Event type (text delta, completed, error).
    pub event_type: EventType,
    /// Item type from the responses type system (0=message, 1=function_call, 3=reasoning).
    pub item_type: u8,
    /// Content type (5=output_text, 8=reasoning_text).
    pub content_type: u8,
    /// Whether this is the final event for this request.
    pub is_final: bool,
    /// Decoded text delta (empty for completion events).
    pub text: String,
    /// Raw token ID.
    pub token_id: u32,
    /// Cumulative tokens generated for this request.
    pub tokens_generated: usize,
    /// Timestamp in nanoseconds.
    pub timestamp_ns: i64,
}

impl Default for BatchEvent {
    fn default() -> Self {
        Self {
            request_id: 0,
            event_type: EventType::TextDelta,
            item_type: 0,
            content_type: 0,
            is_final: false,
            text: String::new(),
            token_id: 0,
            tokens_generated: 0,
            timestamp_ns: 0,
        }
    }
}

/// Completion result for a finished request.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,
    /// Number of tokens generated.
    pub completion_tokens: usize,
    /// Prefill time in nanoseconds.
    pub prefill_ns: u64,
    /// Total generation time in nanoseconds.
    pub generation_ns: u64,
    /// Time to first token in nanoseconds.
    pub ttft_ns: u64,
    /// Finish reason (0=eos, 1=length, 2=stop_sequence, 3=tool_calls, 4=content_filter, 5=cancelled).
    pub finish_reason: u8,
    /// Full generated text.
    pub text: Option<String>,
    /// Tool calls (if finish_reason == tool_calls).
    pub tool_calls: Vec<crate::ToolCall>,
}

// =============================================================================
// BatchHandle — RAII wrapper
// =============================================================================

/// RAII handle for responses-aware batch generation.
///
/// Wraps an opaque `TaluBatch` handle. The underlying scheduler and all
/// per-request state are freed when this handle is dropped.
pub struct BatchHandle {
    ptr: *mut c_void,
}

// SAFETY: The batch handle is single-threaded by contract. Callers must
// serialize access (e.g. via a dedicated step thread with command channels).
// Send is required to move the handle to that dedicated thread.
unsafe impl Send for BatchHandle {}

impl BatchHandle {
    /// Create a batch handle bound to a local inference backend.
    ///
    /// The backend must remain valid for the lifetime of this handle.
    pub fn new(backend: &InferenceBackend, config: Option<&BatchConfig>) -> Result<Self> {
        let c_config = config.map(|cfg| talu_sys::CBatchConfig {
            max_concurrent: cfg.max_concurrent,
        });

        let ptr = unsafe {
            talu_sys::talu_batch_create(
                backend.as_ptr(),
                c_config
                    .as_ref()
                    .map(|c| c as *const _)
                    .unwrap_or(std::ptr::null()),
            )
        };

        if ptr.is_null() {
            return Err(error_from_last_or("Failed to create batch handle"));
        }

        Ok(Self { ptr })
    }

    /// Submit a generation request.
    ///
    /// Applies the chat template, tokenizes, sets up grammar constraints,
    /// and submits to the scheduler. Returns a non-zero request ID.
    pub fn submit(&self, chat: &ChatHandle, config: &GenerateConfig) -> Result<u64> {
        self.submit_raw(chat.as_ptr(), config)
    }

    /// Submit using a raw chat handle pointer.
    ///
    /// # Safety
    ///
    /// `chat_ptr` must be a valid, non-null pointer to a `TaluChatHandle`
    /// that remains valid for the duration of this call.
    pub fn submit_raw(&self, chat_ptr: *mut c_void, config: &GenerateConfig) -> Result<u64> {
        let config_holder = ConfigHolder::new(config)?;

        let request_id =
            unsafe { talu_sys::talu_batch_submit(self.ptr, chat_ptr, config_holder.as_ptr()) };

        if request_id == 0 {
            return Err(error_from_last_or("Batch submit failed"));
        }

        Ok(request_id)
    }

    /// Run one generation step for all active requests.
    ///
    /// Produces rich events with decoded text and metadata. Returns the number
    /// of events written to the output buffer.
    ///
    /// Text in events is copied and valid independently of the next step() call.
    pub fn step(&self, events_out: &mut [BatchEvent]) -> Result<usize> {
        // Stack-allocated C event buffer — avoids heap alloc/free per step.
        let mut c_events_buf = [talu_sys::CBatchEvent::default(); 64];
        let max_events = events_out.len().min(64);

        let count =
            unsafe { talu_sys::talu_batch_step(self.ptr, c_events_buf.as_mut_ptr(), max_events) };

        // Convert C events to Rust events (copy text).
        for i in 0..count.min(max_events) {
            let c = &c_events_buf[i];
            let text = if !c.text_ptr.is_null() && c.text_len > 0 {
                // SAFETY: text_ptr is valid for text_len bytes until next step().
                let bytes = unsafe { std::slice::from_raw_parts(c.text_ptr, c.text_len) };
                String::from_utf8_lossy(bytes).into_owned()
            } else {
                String::new()
            };

            events_out[i] = BatchEvent {
                request_id: c.request_id,
                event_type: EventType::from(c.event_type),
                item_type: c.item_type,
                content_type: c.content_type,
                is_final: c.is_final != 0,
                text,
                token_id: c.token_id,
                tokens_generated: c.tokens_generated,
                timestamp_ns: c.timestamp_ns,
            };
        }

        Ok(count.min(max_events))
    }

    /// Cancel a request.
    ///
    /// Returns `true` if the request was found and cancelled.
    pub fn cancel(&self, request_id: u64) -> bool {
        unsafe { talu_sys::talu_batch_cancel(self.ptr, request_id) != 0 }
    }

    /// Take the completion result for a finished request.
    ///
    /// Returns `None` if the request is not yet complete or was already taken.
    pub fn take_result(&self, request_id: u64) -> Option<BatchResult> {
        let ptr = unsafe { talu_sys::talu_batch_take_result(self.ptr, request_id) };
        if ptr.is_null() {
            return None;
        }

        // SAFETY: ptr is non-null and points to a valid CBatchResult.
        let c_result = unsafe { &*ptr };

        let text = if c_result.text.is_null() {
            None
        } else {
            // SAFETY: Non-null text pointer from the C API.
            Some(
                unsafe { CStr::from_ptr(c_result.text) }
                    .to_string_lossy()
                    .into_owned(),
            )
        };

        let tool_calls = if c_result.tool_calls.is_null() || c_result.tool_call_count == 0 {
            Vec::new()
        } else {
            // SAFETY: Non-null tool_calls ptr with valid count from C API.
            let slice = unsafe {
                std::slice::from_raw_parts(c_result.tool_calls, c_result.tool_call_count)
            };
            slice
                .iter()
                .map(|tc| {
                    let id = if tc.call_id.is_null() {
                        String::new()
                    } else {
                        unsafe { CStr::from_ptr(tc.call_id) }
                            .to_string_lossy()
                            .into_owned()
                    };
                    let name = if tc.name.is_null() {
                        String::new()
                    } else {
                        unsafe { CStr::from_ptr(tc.name) }
                            .to_string_lossy()
                            .into_owned()
                    };
                    let arguments = if tc.arguments.is_null() {
                        String::new()
                    } else {
                        unsafe { CStr::from_ptr(tc.arguments) }
                            .to_string_lossy()
                            .into_owned()
                    };
                    crate::ToolCall {
                        id,
                        name,
                        arguments,
                        item_index: tc.item_index,
                    }
                })
                .collect()
        };

        let result = BatchResult {
            prompt_tokens: c_result.prompt_tokens,
            completion_tokens: c_result.completion_tokens,
            prefill_ns: c_result.prefill_ns,
            generation_ns: c_result.generation_ns,
            ttft_ns: c_result.ttft_ns,
            finish_reason: c_result.finish_reason,
            text,
            tool_calls,
        };

        // SAFETY: ptr was returned by talu_batch_take_result and must be freed.
        unsafe { talu_sys::talu_batch_result_free(ptr) };

        Some(result)
    }

    /// Check if there are any active or pending requests.
    pub fn has_active(&self) -> bool {
        unsafe { talu_sys::talu_batch_has_active(self.ptr) != 0 }
    }

    /// Get the number of active requests.
    pub fn active_count(&self) -> usize {
        unsafe { talu_sys::talu_batch_active_count(self.ptr) }
    }

    /// Run a tight decode loop, calling `callback` for each batch of events.
    ///
    /// Keeps the decode loop entirely in Zig, eliminating per-token FFI
    /// round-trip overhead that occurs when calling `step()` in a Rust loop.
    ///
    /// Works for any number of concurrent requests (N=1 or N>1).
    ///
    /// The loop runs until:
    /// - All requests complete
    /// - `pending_flag` is set (caller has commands to process)
    /// - An error occurs
    ///
    /// Per-request cancellation: the callback can set `pending_flag` to
    /// break out of the loop. The caller then calls `cancel()` for specific
    /// requests before re-entering `run_loop`.
    pub fn run_loop(
        &self,
        pending_flag: &AtomicBool,
        mut callback: impl FnMut(&BatchEvent),
    ) -> Result<()> {
        self.run_loop_impl(pending_flag, true, &mut callback)
    }

    /// Run a tight decode loop, skipping per-token text decoding.
    ///
    /// This is intended for non-streaming server routes that only need
    /// completion/error signaling and will fetch final text from take_result().
    pub fn run_loop_no_text(
        &self,
        pending_flag: &AtomicBool,
        mut callback: impl FnMut(&BatchEvent),
    ) -> Result<()> {
        self.run_loop_impl(pending_flag, false, &mut callback)
    }

    fn run_loop_impl(
        &self,
        pending_flag: &AtomicBool,
        decode_text: bool,
        callback: &mut dyn FnMut(&BatchEvent),
    ) -> Result<()> {
        // Trampoline: extern "C" fn that receives a *mut FnMut(&BatchEvent)
        // through callback_data, converts CEvents to BatchEvents, and invokes it.
        struct CallbackCtx<'a> {
            decode_text: bool,
            callback: &'a mut dyn FnMut(&BatchEvent),
        }

        extern "C" fn trampoline(
            events: *const talu_sys::CEvent,
            count: usize,
            userdata: *mut c_void,
        ) {
            if events.is_null() || count == 0 || userdata.is_null() {
                return;
            }
            let ctx: &mut CallbackCtx<'_> = unsafe { &mut *(userdata as *mut CallbackCtx<'_>) };
            let c_events = unsafe { std::slice::from_raw_parts(events, count) };
            for c in c_events {
                let text = if ctx.decode_text && !c.text_ptr.is_null() && c.text_len > 0 {
                    let bytes = unsafe { std::slice::from_raw_parts(c.text_ptr, c.text_len) };
                    String::from_utf8_lossy(bytes).into_owned()
                } else {
                    String::new()
                };
                let event = BatchEvent {
                    request_id: c.request_id,
                    event_type: EventType::from(c.event_type),
                    item_type: c.item_type,
                    content_type: c.content_type,
                    is_final: c.is_final != 0,
                    text,
                    token_id: c.token_id,
                    tokens_generated: c.tokens_generated,
                    timestamp_ns: c.timestamp_ns,
                };
                (ctx.callback)(&event);
            }
        }

        let mut ctx = CallbackCtx {
            decode_text,
            callback,
        };
        let cb_ptr: *mut c_void = &mut ctx as *mut CallbackCtx<'_> as *mut c_void;

        let rc = unsafe {
            if decode_text {
                talu_sys::talu_batch_run_loop(
                    self.ptr,
                    pending_flag as *const AtomicBool as *mut c_void,
                    trampoline as *mut c_void,
                    cb_ptr,
                )
            } else {
                talu_sys::talu_batch_run_loop_no_text(
                    self.ptr,
                    pending_flag as *const AtomicBool as *mut c_void,
                    trampoline as *mut c_void,
                    cb_ptr,
                )
            }
        };

        if rc != 0 {
            return Err(error_from_last_or("batch run_loop failed"));
        }

        Ok(())
    }
}

impl Drop for BatchHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: self.ptr was allocated by talu_batch_create and is non-null.
            unsafe { talu_sys::talu_batch_destroy(self.ptr) };
        }
    }
}

// =============================================================================
// Internal: ConfigHolder (keeps CStrings alive for C API)
// =============================================================================

/// Holds CString temporaries that the CGenerateConfig borrows.
struct ConfigHolder {
    _template_override: Option<CString>,
    _extra_context: Option<CString>,
    _tools_json: Option<CString>,
    _tool_choice: Option<CString>,
    _extra_body: Option<CString>,
    _reasoning_effort: Option<CString>,
    _stop_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    config: talu_sys::CGenerateConfig,
}

impl ConfigHolder {
    fn new(cfg: &GenerateConfig) -> Result<Self> {
        let template_override = cfg
            .template_override
            .as_ref()
            .map(|s| CString::new(s.as_str()))
            .transpose()?;

        let tools_json = cfg
            .tools_json
            .as_ref()
            .map(|s| CString::new(s.as_str()))
            .transpose()?;

        let tool_choice = cfg
            .tool_choice
            .as_ref()
            .map(|s| CString::new(s.as_str()))
            .transpose()?;

        let extra_body = cfg
            .extra_body_json
            .as_ref()
            .map(|s| CString::new(s.as_str()))
            .transpose()?;

        let reasoning_effort = cfg
            .reasoning_effort
            .as_ref()
            .map(|s| CString::new(s.as_str()))
            .transpose()?;

        let stop_flag = cfg.stop_flag.clone();

        let mut c_config = talu_sys::CGenerateConfig::default();
        c_config.max_tokens = cfg.max_tokens;
        if let Some(mct) = cfg.max_completion_tokens {
            c_config.max_completion_tokens = mct;
        }
        if let Some(mrt) = cfg.max_reasoning_tokens {
            c_config.max_reasoning_tokens = mrt;
        }
        c_config.temperature = cfg.temperature;
        c_config.top_k = cfg.top_k;
        c_config.top_p = cfg.top_p;
        c_config.min_p = cfg.min_p;
        c_config.repetition_penalty = cfg.repetition_penalty;
        c_config.presence_penalty = cfg.presence_penalty;
        c_config.frequency_penalty = cfg.frequency_penalty;
        c_config.seed = cfg.seed;
        c_config.raw_output = if cfg.raw_output { 1 } else { 0 };
        c_config.completions_mode = if cfg.completions_mode { 1 } else { 0 };

        if let Some(ref tpl) = template_override {
            c_config.template_override = tpl.as_ptr();
        }
        if let Some(ref tools) = tools_json {
            c_config.tools_json = tools.as_ptr();
        }
        if let Some(ref choice) = tool_choice {
            c_config.tool_choice = choice.as_ptr();
        }
        if let Some(ref extra) = extra_body {
            c_config.extra_body_json = extra.as_ptr();
        }
        if let Some(ref effort) = reasoning_effort {
            c_config.reasoning_effort = effort.as_ptr();
        }
        if let Some(ref flag) = stop_flag {
            c_config.stop_flag =
                flag.as_ptr() as *const std::sync::atomic::AtomicBool as *mut c_void;
        }

        Ok(Self {
            _template_override: template_override,
            _extra_context: None,
            _tools_json: tools_json,
            _tool_choice: tool_choice,
            _extra_body: extra_body,
            _reasoning_effort: reasoning_effort,
            _stop_flag: stop_flag,
            config: c_config,
        })
    }

    fn as_ptr(&self) -> *const talu_sys::CGenerateConfig {
        &self.config
    }
}
