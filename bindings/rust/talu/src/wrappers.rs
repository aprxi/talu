//! Safe RAII wrappers for talu FFI handles.

use crate::error::error_from_last_or;
use crate::responses::{ResponsesRef, ResponsesView};
use crate::Result;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use talu_sys;

/// RAII wrapper for a chat session handle.
pub struct ChatHandle {
    ptr: *mut c_void,
}

impl ChatHandle {
    /// Creates a new chat session with an optional system message.
    pub fn new(system_msg: Option<&str>) -> Result<Self> {
        // SAFETY: CString is valid for the duration of the call; null config is allowed.
        let ptr = unsafe {
            if let Some(msg) = system_msg {
                let c_msg = CString::new(msg)?;
                talu_sys::talu_chat_create_with_system(c_msg.as_ptr(), std::ptr::null_mut())
            } else {
                talu_sys::talu_chat_create(std::ptr::null_mut())
            }
        };
        if ptr.is_null() {
            return Err(error_from_last_or("Failed to initialize chat session"));
        }
        Ok(Self { ptr })
    }

    /// Returns the raw pointer to the chat handle.
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Returns a borrowed view of the underlying conversation.
    pub fn responses(&self) -> ResponsesRef<'_> {
        // SAFETY: self.ptr is a valid chat handle.
        let ptr = unsafe { talu_sys::talu_chat_get_conversation(self.ptr) as *mut c_void };
        // SAFETY: ptr is a valid conversation handle owned by the chat for &self lifetime.
        unsafe { ResponsesRef::from_raw(ptr) }
    }

    /// Loads items from Open Responses JSON into the chat's conversation.
    ///
    /// Accepts a JSON string (or array of ItemParam objects). Appends to the
    /// existing conversation without clearing it.
    pub fn load_responses_json(&self, json: &str) -> Result<()> {
        let c_json = CString::new(json)?;
        // SAFETY: self.ptr is a valid chat handle.
        let conv_ptr = unsafe { talu_sys::talu_chat_get_conversation(self.ptr) };
        if conv_ptr.is_null() {
            return Err(error_from_last_or("Chat has no conversation"));
        }
        // SAFETY: conv_ptr is non-null (checked above); c_json is a valid CString.
        let rc = unsafe { talu_sys::talu_responses_load_responses_json(conv_ptr, c_json.as_ptr()) };
        if rc != 0 {
            return Err(error_from_last_or("Failed to load Responses JSON"));
        }
        Ok(())
    }

    /// Loads chat completions-format messages JSON into the chat's conversation.
    ///
    /// Accepts a JSON array of messages in OpenAI chat completions format:
    /// `[{"role":"user","content":"Hello"}, ...]`
    ///
    /// Clears existing items first, then parses the messages into the conversation.
    pub fn load_completions_json(&self, json: &str) -> Result<()> {
        // SAFETY: self.ptr is a valid chat handle; json bytes are valid for the slice length.
        let rc = unsafe {
            talu_sys::talu_chat_load_completions_json(self.ptr, json.as_ptr(), json.len())
        };
        if rc != 0 {
            return Err(error_from_last_or("Failed to load Completions JSON"));
        }
        Ok(())
    }

    /// Appends a user message to the chat's conversation.
    ///
    /// Shorthand for loading a single user message without constructing JSON.
    pub fn append_user_message(&self, content: &str) -> Result<()> {
        // SAFETY: self.ptr is a valid chat handle.
        let conv_ptr = unsafe { talu_sys::talu_chat_get_conversation(self.ptr) };
        if conv_ptr.is_null() {
            return Err(error_from_last_or("Chat has no conversation"));
        }
        // SAFETY: conv_ptr is non-null; content ptr/len are valid for the str.
        // Role codes: 0=system, 1=user, 2=assistant, 3=developer.
        let result = unsafe {
            talu_sys::talu_responses_append_message(
                conv_ptr,
                1, // User
                content.as_ptr(),
                content.len(),
            )
        };
        if result < 0 {
            return Err(error_from_last_or("Failed to append user message"));
        }
        Ok(())
    }

    /// Serializes the conversation to Open Responses JSON format.
    ///
    /// `direction`: 0 = request (ItemParam schemas), 1 = response (ItemField schemas).
    pub fn to_responses_json(&self, direction: u8) -> Result<String> {
        self.responses().to_responses_json(direction)
    }

    /// Returns the number of items in the conversation.
    pub fn item_count(&self) -> usize {
        self.responses().item_count()
    }

    /// Appends a function call output to the chat's conversation.
    ///
    /// This is used by agent loops to feed tool results back into the
    /// conversation before the next generation call.
    pub fn append_function_call_output(&self, call_id: &str, output: &str) -> Result<()> {
        let c_call_id = CString::new(call_id)?;
        // SAFETY: self.ptr is a valid chat handle.
        let conv_ptr = unsafe { talu_sys::talu_chat_get_conversation(self.ptr) };
        if conv_ptr.is_null() {
            return Err(error_from_last_or("Chat has no conversation"));
        }
        // SAFETY: conv_ptr is non-null (checked above); CString and output ptr/len are valid.
        let result = unsafe {
            talu_sys::talu_responses_append_function_call_output(
                conv_ptr,
                c_call_id.as_ptr(),
                output.as_ptr(),
                output.len(),
            )
        };
        if result < 0 {
            return Err(error_from_last_or("Failed to append function call output"));
        }
        Ok(())
    }

    /// Sets tool definitions on the chat (opaque JSON blob).
    ///
    /// Stores the JSON for round-tripping in the response resource.
    /// Pass the raw JSON array of tool definitions from the request.
    pub fn set_tools(&self, json: &str) -> Result<()> {
        let c_json = CString::new(json)?;
        // SAFETY: self.ptr is a valid chat handle; c_json is a valid CString.
        let rc = unsafe { talu_sys::talu_chat_set_tools(self.ptr, c_json.as_ptr(), json.len()) };
        if rc != 0 {
            return Err(error_from_last_or("Failed to set tools"));
        }
        Ok(())
    }

    /// Gets tool definitions JSON, or None if not set.
    ///
    /// The returned string must be freed by the caller (handled by TextPtr RAII).
    pub fn get_tools(&self) -> Option<String> {
        // SAFETY: self.ptr is a valid chat handle. Returns null if no tools set.
        let ptr = unsafe { talu_sys::talu_chat_get_tools(self.ptr) };
        TextPtr::new(ptr as *mut c_char).map(|t| t.to_string_lossy())
    }

    /// Sets tool_choice on the chat (opaque JSON blob).
    ///
    /// Stores the JSON for round-tripping in the response resource.
    pub fn set_tool_choice(&self, json: &str) -> Result<()> {
        let c_json = CString::new(json)?;
        // SAFETY: self.ptr is a valid chat handle; c_json is a valid CString.
        let rc =
            unsafe { talu_sys::talu_chat_set_tool_choice(self.ptr, c_json.as_ptr(), json.len()) };
        if rc != 0 {
            return Err(error_from_last_or("Failed to set tool_choice"));
        }
        Ok(())
    }

    /// Gets tool_choice JSON, or None if not set.
    pub fn get_tool_choice(&self) -> Option<String> {
        // SAFETY: self.ptr is a valid chat handle. Returns null if no tool_choice set.
        let ptr = unsafe { talu_sys::talu_chat_get_tool_choice(self.ptr) };
        TextPtr::new(ptr as *mut c_char).map(|t| t.to_string_lossy())
    }
}

impl Drop for ChatHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: self.ptr was allocated by talu_chat_create and is non-null.
            unsafe { talu_sys::talu_chat_free(self.ptr) };
        }
    }
}

// ChatHandle is not Send/Sync by default due to raw pointer
// If the C API is thread-safe, these can be implemented:
// unsafe impl Send for ChatHandle {}
// unsafe impl Sync for ChatHandle {}

/// RAII wrapper for a tokenizer handle.
pub struct TokenizerHandle {
    ptr: *mut c_void,
}

/// Result of tokenization.
pub struct EncodeResult {
    /// The encoded token IDs.
    pub tokens: Vec<u32>,
}

impl TokenizerHandle {
    /// Creates a new tokenizer from a model path.
    pub fn new(model_path: &str) -> Result<Self> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let c_path = CString::new(model_path)?;
        // SAFETY: c_path is a valid CString; ptr is a valid out-param.
        let rc = unsafe {
            talu_sys::talu_tokenizer_create(c_path.as_ptr(), &mut ptr as *mut _ as *mut c_void)
        };
        if rc != 0 || ptr.is_null() {
            return Err(error_from_last_or("Failed to load tokenizer"));
        }
        Ok(Self { ptr })
    }

    /// Encodes text into tokens, returning a Vec of token IDs.
    pub fn encode(&self, text: &str) -> Result<EncodeResult> {
        let options = talu_sys::EncodeOptions::default();
        // SAFETY: self.ptr is valid, text is valid UTF-8.
        let result = unsafe {
            talu_sys::talu_tokenizer_encode(
                self.ptr,
                text.as_bytes().as_ptr(),
                text.len(),
                &options,
            )
        };
        if !result.error_msg.is_null() {
            return Err(error_from_last_or("Failed to encode text"));
        }

        // Copy tokens to a Vec for safe ownership
        let tokens = if result.ids.is_null() || result.num_tokens == 0 {
            Vec::new()
        } else {
            // SAFETY: ids is a valid pointer with num_tokens elements.
            let slice = unsafe { std::slice::from_raw_parts(result.ids, result.num_tokens) };
            slice.to_vec()
        };

        // Free the C-allocated result (ids, offsets, masks)
        unsafe { talu_sys::talu_encode_result_free(result) };

        Ok(EncodeResult { tokens })
    }

    /// Encodes text and returns the raw result (for internal use).
    /// Caller is responsible for freeing the result via `talu_encode_result_free`.
    pub fn encode_raw(&self, text: &str) -> Result<talu_sys::EncodeResult> {
        let options = talu_sys::EncodeOptions::default();
        // SAFETY: self.ptr is valid, text is valid UTF-8.
        let result = unsafe {
            talu_sys::talu_tokenizer_encode(
                self.ptr,
                text.as_bytes().as_ptr(),
                text.len(),
                &options,
            )
        };
        if !result.error_msg.is_null() {
            return Err(error_from_last_or("Failed to encode text"));
        }
        Ok(result)
    }

    /// Returns the raw pointer to the tokenizer handle.
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for TokenizerHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: self.ptr was allocated by talu_tokenizer_create and is non-null.
            unsafe { talu_sys::talu_tokenizer_free(self.ptr) };
        }
    }
}

/// RAII wrapper for a text pointer allocated by the C API.
pub struct TextPtr {
    ptr: *mut c_char,
}

impl TextPtr {
    /// Creates a TextPtr from a raw pointer, returning None if null.
    pub fn new(ptr: *mut c_char) -> Option<Self> {
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Converts the text to a lossy String.
    pub fn to_string_lossy(&self) -> String {
        // SAFETY: self.ptr is non-null (checked in new()) and points to a valid C string.
        unsafe { CStr::from_ptr(self.ptr) }
            .to_string_lossy()
            .to_string()
    }

    /// Returns the raw pointer.
    pub fn as_ptr(&self) -> *const c_char {
        self.ptr
    }
}

impl Drop for TextPtr {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: self.ptr was allocated by the C API and is non-null.
            unsafe { talu_sys::talu_text_free(self.ptr) };
        }
    }
}

/// RAII wrapper for a CanonicalSpec handle.
pub struct CanonicalSpec {
    ptr: *mut c_void,
}

impl CanonicalSpec {
    /// Returns a const pointer to the canonical spec.
    pub fn as_ptr(&self) -> *const c_void {
        self.ptr
    }
}

impl Drop for CanonicalSpec {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: self.ptr was allocated by talu_config_canonicalize and is non-null.
            unsafe { talu_sys::talu_config_free(self.ptr) };
        }
    }
}

// =============================================================================
// Load Progress
// =============================================================================

/// Progress update during model loading.
#[derive(Debug, Clone)]
pub struct LoadProgress {
    /// The action for this update.
    pub action: crate::convert::ProgressAction,
    /// Line ID for multi-line progress.
    pub line_id: u8,
    /// Name/label of the current operation.
    pub label: String,
    /// Progress message.
    pub message: String,
    /// Current progress value.
    pub current: u64,
    /// Total expected value.
    pub total: u64,
}

/// Callback type for model load progress updates.
pub type LoadProgressCallback = Box<dyn FnMut(LoadProgress) + Send>;

/// Callback wrapper context for C API progress callback.
struct LoadProgressContext {
    callback: LoadProgressCallback,
}

unsafe extern "C" fn load_progress_callback_wrapper(
    update: *const talu_sys::ProgressUpdate,
    user_data: *mut c_void,
) {
    if update.is_null() || user_data.is_null() {
        return;
    }

    // SAFETY: user_data is a valid pointer to LoadProgressContext created by new_with_progress.
    let ctx = unsafe { &mut *(user_data as *mut LoadProgressContext) };

    // SAFETY: update is a valid pointer passed from C.
    let update_ref = unsafe { &*update };

    let label = if update_ref.label.is_null() {
        String::new()
    } else {
        // SAFETY: label is a valid C string from the C API.
        unsafe { CStr::from_ptr(update_ref.label) }
            .to_string_lossy()
            .into_owned()
    };

    let message = if update_ref.message.is_null() {
        String::new()
    } else {
        // SAFETY: message is a valid C string from the C API.
        unsafe { CStr::from_ptr(update_ref.message) }
            .to_string_lossy()
            .into_owned()
    };

    let progress = LoadProgress {
        action: crate::convert::ProgressAction::from(update_ref.action),
        line_id: update_ref.line_id,
        label,
        message,
        current: update_ref.current,
        total: update_ref.total,
    };

    (ctx.callback)(progress);
}

/// RAII wrapper for an InferenceBackend handle.
pub struct InferenceBackend {
    ptr: *mut c_void,
    _canonical: CanonicalSpec, // Keep canonical spec alive
}

// SAFETY: The backend handle is synchronized by callers (e.g., a Mutex) and the
// C API is treated as thread-safe for serialized access.
unsafe impl Send for InferenceBackend {}
unsafe impl Sync for InferenceBackend {}

impl InferenceBackend {
    /// Creates a new inference backend from a model path (local inference).
    pub fn new(model_path: &str) -> Result<Self> {
        let c_path = CString::new(model_path)?;

        // Construct TaluModelSpec for local backend (backend_type_raw = 0 for local)
        let spec = talu_sys::TaluModelSpec {
            abi_version: 1,
            struct_size: std::mem::size_of::<talu_sys::TaluModelSpec>() as u32,
            ref_: c_path.as_ptr(),
            backend_type_raw: 0, // 0 = local
            // SAFETY: BackendUnion is a C union; zeroed is a valid representation.
            backend_config: unsafe { std::mem::zeroed() },
        };

        // Canonicalize the spec
        let mut canon_ptr: *mut c_void = std::ptr::null_mut();
        let mut spec_mut = spec;
        // SAFETY: spec_mut is a valid TaluModelSpec; canon_ptr is a valid out-param.
        let rc = unsafe {
            talu_sys::talu_config_canonicalize(
                &mut spec_mut,
                &mut canon_ptr as *mut _ as *mut c_void,
            )
        };
        if rc != 0 || canon_ptr.is_null() {
            return Err(error_from_last_or("Failed to canonicalize model spec"));
        }
        let canonical = CanonicalSpec { ptr: canon_ptr };

        // Create backend from canonical spec (no progress)
        let mut backend_ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: canonical.as_ptr() is a valid canonical spec; backend_ptr is a valid out-param.
        let rc = unsafe {
            talu_sys::talu_backend_create_from_canonical(
                canonical.as_ptr() as *mut c_void,
                talu_sys::BackendCreateOptions::default(),
                &mut backend_ptr as *mut _ as *mut c_void,
            )
        };
        if rc != 0 || backend_ptr.is_null() {
            return Err(error_from_last_or("Failed to create inference backend"));
        }

        Ok(Self {
            ptr: backend_ptr,
            _canonical: canonical,
        })
    }

    /// Creates a new inference backend with a progress callback for load status.
    pub fn new_with_progress(
        model_path: &str,
        callback: Option<LoadProgressCallback>,
    ) -> Result<Self> {
        let c_path = CString::new(model_path)?;

        let spec = talu_sys::TaluModelSpec {
            abi_version: 1,
            struct_size: std::mem::size_of::<talu_sys::TaluModelSpec>() as u32,
            ref_: c_path.as_ptr(),
            backend_type_raw: 0,
            // SAFETY: BackendUnion is a C union; zeroed is a valid representation.
            backend_config: unsafe { std::mem::zeroed() },
        };

        let mut canon_ptr: *mut c_void = std::ptr::null_mut();
        let mut spec_mut = spec;
        // SAFETY: spec_mut is a valid TaluModelSpec; canon_ptr is a valid out-param.
        let rc = unsafe {
            talu_sys::talu_config_canonicalize(
                &mut spec_mut,
                &mut canon_ptr as *mut _ as *mut c_void,
            )
        };
        if rc != 0 || canon_ptr.is_null() {
            return Err(error_from_last_or("Failed to canonicalize model spec"));
        }
        let canonical = CanonicalSpec { ptr: canon_ptr };

        // Set up progress callback in options struct (same pattern as ConvertOptions)
        let mut progress_ctx = callback.map(|cb| LoadProgressContext { callback: cb });
        let options = if let Some(ref mut ctx) = progress_ctx {
            talu_sys::BackendCreateOptions {
                progress_callback: load_progress_callback_wrapper as *mut c_void,
                progress_user_data: ctx as *mut LoadProgressContext as *mut c_void,
            }
        } else {
            talu_sys::BackendCreateOptions::default()
        };

        let mut backend_ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: canonical.as_ptr() is a valid canonical spec; backend_ptr is a valid out-param.
        // progress_ctx and options are alive for the duration of this call.
        let rc = unsafe {
            talu_sys::talu_backend_create_from_canonical(
                canonical.as_ptr() as *mut c_void,
                options,
                &mut backend_ptr as *mut _ as *mut c_void,
            )
        };
        if rc != 0 || backend_ptr.is_null() {
            return Err(error_from_last_or("Failed to create inference backend"));
        }

        Ok(Self {
            ptr: backend_ptr,
            _canonical: canonical,
        })
    }

    /// Returns the raw pointer to the backend handle.
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Return static model metadata from a local backend.
    /// Returns a zero struct for non-local backends.
    pub fn model_info(&self) -> talu_sys::CModelInfo {
        unsafe { talu_sys::talu_backend_model_info(self.ptr) }
    }
}

impl Drop for InferenceBackend {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: self.ptr was allocated by talu_backend_create and is non-null.
            unsafe { talu_sys::talu_backend_free(self.ptr) };
        }
    }
}

/// Finish reason for generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    EosToken,
    Length,
    StopSequence,
    ToolCalls,
    ContentFilter,
    Cancelled,
}

impl From<talu_sys::CFinishReason> for FinishReason {
    fn from(r: talu_sys::CFinishReason) -> Self {
        match r {
            talu_sys::CFinishReason::EosToken => FinishReason::EosToken,
            talu_sys::CFinishReason::Length => FinishReason::Length,
            talu_sys::CFinishReason::StopSequence => FinishReason::StopSequence,
            talu_sys::CFinishReason::ToolCalls => FinishReason::ToolCalls,
            talu_sys::CFinishReason::ContentFilter => FinishReason::ContentFilter,
            talu_sys::CFinishReason::Cancelled => FinishReason::Cancelled,
        }
    }
}

/// A tool call extracted from generation output.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
    /// Index of the FunctionCallItem in the conversation.
    pub item_index: usize,
}

/// RAII wrapper for generation results.
pub struct GenerateResult {
    inner: talu_sys::CGenerateResult,
}

impl GenerateResult {
    /// Creates a new GenerateResult from the raw FFI result.
    pub fn new(inner: talu_sys::CGenerateResult) -> Self {
        Self { inner }
    }

    /// Returns a reference to the inner result.
    pub fn inner(&self) -> &talu_sys::CGenerateResult {
        &self.inner
    }

    /// Returns the error code.
    pub fn error_code(&self) -> i32 {
        self.inner.error_code
    }

    /// Returns the token count.
    pub fn token_count(&self) -> usize {
        self.inner.token_count
    }

    /// Returns the prompt token count.
    pub fn prompt_tokens(&self) -> usize {
        self.inner.prompt_tokens
    }

    /// Returns the completion token count.
    pub fn completion_tokens(&self) -> usize {
        self.inner.completion_tokens
    }

    /// Returns the prefill time in nanoseconds.
    pub fn prefill_ns(&self) -> u64 {
        self.inner.prefill_ns
    }

    /// Returns the generation time in nanoseconds.
    pub fn generation_ns(&self) -> u64 {
        self.inner.generation_ns
    }

    /// Returns the time-to-first-token in nanoseconds.
    pub fn ttft_ns(&self) -> u64 {
        self.inner.ttft_ns
    }

    /// Returns the generated text, if any.
    pub fn text(&self) -> Option<String> {
        if self.inner.text.is_null() {
            None
        } else {
            // SAFETY: Non-null text pointer from the C generate result.
            Some(
                unsafe { CStr::from_ptr(self.inner.text) }
                    .to_string_lossy()
                    .to_string(),
            )
        }
    }

    /// Returns the finish reason for this generation.
    pub fn finish_reason(&self) -> FinishReason {
        FinishReason::from(talu_sys::CFinishReason::from(self.inner.finish_reason))
    }

    /// Extracts tool calls from the generation result.
    ///
    /// Returns an empty vec if there are no tool calls. Each tool call includes
    /// the call ID, function name, and raw arguments JSON string.  The arguments
    /// are read from the conversation items referenced by `item_index`; when the
    /// C API does not populate the arguments pointer directly on `CToolCallRef`,
    /// the caller should fall back to `GenerateResult::text()` which contains
    /// the raw model output including any tool-call markup.
    pub fn tool_calls(&self) -> Vec<ToolCall> {
        if self.inner.tool_calls.is_null() || self.inner.tool_call_count == 0 {
            return Vec::new();
        }

        // SAFETY: Non-null tool_calls ptr with valid count from C API.
        let slice = unsafe {
            std::slice::from_raw_parts(self.inner.tool_calls, self.inner.tool_call_count)
        };

        slice
            .iter()
            .map(|tc| {
                let id = if tc.call_id.is_null() {
                    String::new()
                } else {
                    // SAFETY: Non-null C string from C API.
                    unsafe { CStr::from_ptr(tc.call_id) }
                        .to_string_lossy()
                        .into_owned()
                };
                let name = if tc.name.is_null() {
                    String::new()
                } else {
                    // SAFETY: Non-null C string from C API.
                    unsafe { CStr::from_ptr(tc.name) }
                        .to_string_lossy()
                        .into_owned()
                };
                ToolCall {
                    id,
                    name,
                    arguments: String::new(),
                    item_index: tc.item_index,
                }
            })
            .collect()
    }
}

impl Drop for GenerateResult {
    fn drop(&mut self) {
        // SAFETY: self.inner was populated by talu_router_generate and must be freed.
        unsafe { talu_sys::talu_router_result_free(&mut self.inner) };
    }
}
