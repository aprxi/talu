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

    /// Sets TaluDB storage backend for this chat session.
    ///
    /// This enables persistence of the conversation to TaluDB storage.
    /// All subsequent messages will be persisted to the specified database
    /// and session. If the session already exists, items are loaded from storage.
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to TaluDB storage directory
    /// * `session_id` - Unique session identifier
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or the lock cannot be acquired.
    pub fn set_storage_db(&self, db_path: &str, session_id: &str) -> Result<()> {
        let c_db_path = CString::new(db_path)?;
        let c_session_id = CString::new(session_id)?;

        // SAFETY: self.ptr is a valid chat handle; CStrings are valid.
        let rc = unsafe {
            talu_sys::talu_db_ops_set_storage_db(
                self.ptr,
                c_db_path.as_ptr(),
                c_session_id.as_ptr(),
            )
        };

        if rc != 0 {
            return Err(error_from_last_or("Failed to set TaluDB storage backend"));
        }

        Ok(())
    }

    /// Sets the maximum segment size for TaluDB storage.
    ///
    /// When the active segment (`current.talu`) would exceed this size after
    /// a flush, the writer seals it as `seg-<uuid>.talu`, updates
    /// `manifest.json`, and creates a fresh `current.talu`.
    ///
    /// Must be called after [`set_storage_db`]. Pass 0 to restore the
    /// default (64 MB).
    ///
    /// # Errors
    ///
    /// Returns an error if no TaluDB storage backend is set.
    pub fn set_max_segment_size(&self, max_bytes: u64) -> Result<()> {
        // SAFETY: self.ptr is a valid chat handle.
        let rc = unsafe { talu_sys::talu_db_ops_set_max_segment_size(self.ptr, max_bytes) };

        if rc != 0 {
            return Err(error_from_last_or("Failed to set max segment size"));
        }

        Ok(())
    }

    /// Sets the write durability mode for the TaluDB storage backend.
    ///
    /// Must be called after [`set_storage_db`]. Controls whether writes
    /// are fsynced to disk on every append (full durability) or buffered
    /// in the OS page cache (async durability).
    ///
    /// # Errors
    ///
    /// Returns an error if no TaluDB storage backend is set or the mode is invalid.
    pub fn set_durability(&self, mode: crate::Durability) -> Result<()> {
        // SAFETY: self.ptr is a valid chat handle.
        let rc = unsafe { talu_sys::talu_db_ops_set_durability(self.ptr, mode as u8) };
        if rc != 0 {
            return Err(error_from_last_or("Failed to set durability"));
        }
        Ok(())
    }

    /// Simulates a process crash for testing.
    ///
    /// Closes all file descriptors (releasing flocks) WITHOUT flushing
    /// pending data or deleting the WAL file. This accurately simulates
    /// what the OS does when a process dies: all locks are released, but
    /// files remain on disk. The orphaned WAL will be replayed by the
    /// next `Writer::open`.
    ///
    /// After calling this, the storage backend is detached. The
    /// `ChatHandle` should still be dropped normally to free memory.
    ///
    /// # Errors
    ///
    /// Returns an error if no TaluDB storage backend is set.
    pub fn simulate_crash(&self) -> Result<()> {
        // SAFETY: self.ptr is a valid chat handle.
        let rc = unsafe { talu_sys::talu_db_ops_simulate_crash(self.ptr) };
        if rc != 0 {
            return Err(error_from_last_or("Failed to simulate crash"));
        }
        Ok(())
    }

    /// Notifies the storage backend of a session metadata update.
    ///
    /// Call this after generation completes to persist session metadata
    /// (model, title, marker) to TaluDB. This writes a Schema 4 session
    /// record that enables `talu db list` to show the session.
    ///
    /// # Arguments
    ///
    /// * `model` - Model identifier (e.g., "Qwen/Qwen3-0.6B")
    /// * `title` - Optional session title (derived from first message if None)
    /// * `marker` - Session marker (e.g., "pinned", "archived", "deleted")
    pub fn notify_session_update(
        &self,
        model: Option<&str>,
        title: Option<&str>,
        marker: Option<&str>,
    ) -> Result<()> {
        self.notify_session_update_ex(model, title, marker, None, None)
    }

    /// Extended session update with source document ID and project ID.
    ///
    /// Call this after generation completes to persist session metadata
    /// to TaluDB. This writes a Schema 5 session record that enables
    /// `talu db list` to show the session.
    ///
    /// # Arguments
    ///
    /// * `model` - Model identifier (e.g., "Qwen/Qwen3-0.6B")
    /// * `title` - Optional session title (derived from first message if None)
    /// * `marker` - Session marker (e.g., "pinned", "archived", "deleted")
    /// * `source_doc_id` - Source document ID for lineage tracking (prompt_id)
    /// * `project_id` - Project identifier for multi-project session organization
    pub fn notify_session_update_ex(
        &self,
        model: Option<&str>,
        title: Option<&str>,
        marker: Option<&str>,
        source_doc_id: Option<&str>,
        project_id: Option<&str>,
    ) -> Result<()> {
        let c_model = model.map(|s| CString::new(s)).transpose()?;
        let c_title = title.map(|s| CString::new(s)).transpose()?;
        let c_marker = marker.map(|s| CString::new(s)).transpose()?;
        let c_source_doc_id = source_doc_id.map(|s| CString::new(s)).transpose()?;
        let c_project_id = project_id.map(|s| CString::new(s)).transpose()?;

        // SAFETY: self.ptr is a valid chat handle; CStrings/nulls are valid.
        let rc = unsafe {
            talu_sys::talu_chat_notify_session_update(
                self.ptr,
                c_model
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
                c_title
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
                std::ptr::null(), // system_prompt
                std::ptr::null(), // config_json
                c_marker
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
                std::ptr::null(), // parent_session_id
                std::ptr::null(), // group_id
                std::ptr::null(), // metadata_json
                c_source_doc_id
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
                c_project_id
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
            )
        };

        if rc != 0 {
            return Err(error_from_last_or("Failed to notify session update"));
        }

        Ok(())
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
        // MessageRole::User = 0.
        let result = unsafe {
            talu_sys::talu_responses_append_message(
                conv_ptr,
                0, // User
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

    /// Attaches an IAM-style policy to this chat for tool call filtering.
    ///
    /// When a policy is set, tool calls generated during inference are
    /// evaluated against it before being committed. Denied tool calls
    /// are committed with `status: failed`.
    ///
    /// # Safety contract
    ///
    /// The `Policy` must outlive this `ChatHandle`. The core stores a raw
    /// pointer — dropping the policy while the chat is alive is undefined
    /// behavior. Pass `None` to detach.
    pub fn set_policy(&self, policy: Option<&crate::policy::Policy>) -> Result<()> {
        let policy_ptr = policy.map(|p| p.as_ptr()).unwrap_or(std::ptr::null_mut());
        // SAFETY: policy_ptr is either null (detach) or valid and will
        // outlive the chat per the caller's contract.
        let rc = unsafe { talu_sys::talu_chat_set_policy(self.ptr, policy_ptr) };
        if rc != 0 {
            return Err(error_from_last_or("Failed to set policy"));
        }
        Ok(())
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
            talu_sys::talu_tokenizer_encode(self.ptr, text.as_bytes().as_ptr(), text.len(), options)
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
            talu_sys::talu_tokenizer_encode(self.ptr, text.as_bytes().as_ptr(), text.len(), options)
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
    // Keep CStrings alive for the duration of the backend
    _base_url: Option<CString>,
    _api_key: Option<CString>,
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
            _base_url: None,
            _api_key: None,
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
            _base_url: None,
            _api_key: None,
        })
    }

    /// Creates a new inference backend for a remote OpenAI-compatible server.
    pub fn new_openai_compatible(
        model_id: &str,
        base_url: &str,
        api_key: Option<&str>,
        timeout_ms: i32,
    ) -> Result<Self> {
        let c_model = CString::new(model_id)?;
        let c_base_url = CString::new(base_url)?;
        let c_api_key = api_key.map(|k| CString::new(k)).transpose()?;

        let openai_config = talu_sys::OpenAICompatibleConfig {
            base_url: c_base_url.as_ptr(),
            api_key: c_api_key
                .as_ref()
                .map(|k| k.as_ptr())
                .unwrap_or(std::ptr::null()),
            org_id: std::ptr::null(),
            timeout_ms,
            max_retries: 2,
            custom_headers_json: std::ptr::null(),
            _reserved: [0; 24],
        };

        let spec = talu_sys::TaluModelSpec {
            abi_version: 1,
            struct_size: std::mem::size_of::<talu_sys::TaluModelSpec>() as u32,
            ref_: c_model.as_ptr(),
            backend_type_raw: 1, // 1 = OpenAICompatible
            backend_config: talu_sys::BackendUnion {
                openai_compat: openai_config,
            },
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
            return Err(error_from_last_or(
                "Failed to canonicalize remote model spec",
            ));
        }
        let canonical = CanonicalSpec { ptr: canon_ptr };

        // Create backend from canonical spec (no progress for remote)
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
            return Err(error_from_last_or(
                "Failed to create remote inference backend",
            ));
        }

        Ok(Self {
            ptr: backend_ptr,
            _canonical: canonical,
            _base_url: Some(c_base_url),
            _api_key: c_api_key,
        })
    }

    /// Returns the raw pointer to the backend handle.
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// List models available on a remote backend.
    /// Only works for OpenAI-compatible backends.
    pub fn list_models(&self) -> Result<Vec<RemoteModelInfo>> {
        // SAFETY: self.ptr is a valid backend handle.
        let result = unsafe { talu_sys::talu_backend_list_models(self.ptr) };

        if result.error_code != 0 {
            return Err(error_from_last_or("Failed to list models"));
        }

        let mut models = Vec::new();
        if !result.models.is_null() && result.count > 0 {
            // SAFETY: Non-null models ptr with valid count from C API.
            let slice = unsafe { std::slice::from_raw_parts(result.models, result.count) };
            for info in slice {
                models.push(RemoteModelInfo {
                    id: if info.id.is_null() {
                        String::new()
                    } else {
                        // SAFETY: Non-null C string from C API.
                        unsafe { CStr::from_ptr(info.id) }
                            .to_string_lossy()
                            .to_string()
                    },
                    object: if info.object.is_null() {
                        String::new()
                    } else {
                        // SAFETY: Non-null C string from C API.
                        unsafe { CStr::from_ptr(info.object) }
                            .to_string_lossy()
                            .to_string()
                    },
                    created: info.created,
                    owned_by: if info.owned_by.is_null() {
                        String::new()
                    } else {
                        // SAFETY: Non-null C string from C API.
                        unsafe { CStr::from_ptr(info.owned_by) }
                            .to_string_lossy()
                            .to_string()
                    },
                });
            }
        }

        // Free the C result
        let mut result = result;
        // SAFETY: result was returned by talu_backend_list_models and must be freed.
        unsafe { talu_sys::talu_backend_list_models_free(&mut result as *mut _ as *mut c_void) };

        Ok(models)
    }
}

/// Model info from a remote backend
#[derive(Debug, Clone)]
pub struct RemoteModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
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
