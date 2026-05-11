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

    pub(crate) fn as_ptr(&self) -> *mut c_void {
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

// SAFETY: A ChatHandle owns a single conversation handle. Moving ownership to
// another thread is sound when no references remain on the source thread. The
// type intentionally does not implement Sync; shared concurrent access must be
// externally serialized by higher-level code.
unsafe impl Send for ChatHandle {}

/// RAII wrapper for a tokenizer handle.
#[derive(Debug)]
pub struct TokenizerHandle {
    ptr: *mut c_void,
}

// SAFETY: Tokenizer handles are immutable after construction. Sharing across
// threads is valid for read-only tokenizer operations; Drop still owns and frees
// exactly one handle.
unsafe impl Send for TokenizerHandle {}
unsafe impl Sync for TokenizerHandle {}

/// Result of tokenization.
pub struct EncodeResult {
    /// The encoded token IDs.
    pub tokens: Vec<u32>,
}

/// Options for tokenizer encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenizerEncodeOptions {
    pub add_bos: bool,
    pub add_eos: bool,
    pub truncation: Option<TokenizerTruncation>,
}

impl Default for TokenizerEncodeOptions {
    fn default() -> Self {
        Self {
            add_bos: false,
            add_eos: false,
            truncation: None,
        }
    }
}

/// Tokenizer truncation settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenizerTruncation {
    pub max_length: usize,
    pub side: TokenizerTruncationSide,
}

/// Which side is truncated when encoded IDs exceed `max_length`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerTruncationSide {
    Right,
    Left,
}

/// Owned tokenizer encoding result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerEncoding {
    pub ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub special_tokens_mask: Vec<u32>,
    pub offsets: Vec<[u32; 2]>,
}

/// Owned tokenizer vocabulary entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerVocabEntry {
    pub token: String,
    pub id: u32,
}

/// Special token IDs reported by the tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenizerSpecialTokens {
    pub bos_token_id: i32,
    pub unk_token_id: i32,
    pub pad_token_id: i32,
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

    /// Creates a new tokenizer from a tokenizer.json payload.
    pub fn from_json(json: &str) -> Result<Self> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: json bytes and out pointer are valid for the duration of the call.
        let rc = unsafe {
            talu_sys::talu_tokenizer_create_from_json(
                json.as_bytes().as_ptr(),
                json.len(),
                &mut ptr as *mut _ as *mut c_void,
            )
        };
        if rc != 0 || ptr.is_null() {
            return Err(error_from_last_or("Failed to load tokenizer from JSON"));
        }
        Ok(Self { ptr })
    }

    /// Encodes text into tokens, returning a Vec of token IDs.
    pub fn encode(&self, text: &str) -> Result<EncodeResult> {
        self.encode_with_options(text, TokenizerEncodeOptions::default())
            .map(|encoding| EncodeResult {
                tokens: encoding.ids,
            })
    }

    /// Encodes text with explicit options.
    pub fn encode_with_options(
        &self,
        text: &str,
        options: TokenizerEncodeOptions,
    ) -> Result<TokenizerEncoding> {
        let c_options = encode_options_to_c(options);
        // SAFETY: self.ptr is valid, text is valid UTF-8.
        let result = unsafe {
            talu_sys::talu_tokenizer_encode(
                self.ptr,
                text.as_bytes().as_ptr(),
                text.len(),
                &c_options,
            )
        };
        if !result.error_msg.is_null() {
            return Err(error_from_last_or("Failed to encode text"));
        }
        let encoding = tokenizer_encoding_from_c(&result);
        // SAFETY: result was returned by talu_tokenizer_encode and must be freed once.
        unsafe { talu_sys::talu_encode_result_free(result) };
        Ok(encoding)
    }

    /// Encodes a batch of texts and returns owned token ID rows.
    pub fn encode_batch_ids(
        &self,
        texts: &[String],
        options: TokenizerEncodeOptions,
    ) -> Result<Vec<Vec<u32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let c_options = encode_options_to_c(options);
        let text_ptrs: Vec<*const u8> = texts.iter().map(|text| text.as_bytes().as_ptr()).collect();
        let lengths: Vec<usize> = texts.iter().map(String::len).collect();

        // SAFETY: tokenizer handle is valid; text pointers and length arrays
        // stay alive for the duration of the call.
        let result = unsafe {
            talu_sys::talu_tokenizer_encode_batch(
                self.ptr,
                text_ptrs.as_ptr(),
                lengths.as_ptr(),
                texts.len(),
                &c_options,
            )
        };
        if !result.error_msg.is_null() {
            return Err(error_from_last_or("Batch encode failed"));
        }

        let ids = copy_u32_slice(result.ids, result.total_tokens);
        let offsets = copy_usize_slice(result.offsets, result.num_sequences.saturating_add(1));

        // SAFETY: result was returned by the tokenizer C API.
        unsafe {
            talu_sys::talu_batch_encode_result_free(
                result.ids,
                result.offsets,
                result.total_tokens,
                result.num_sequences,
            );
        }

        if result.num_sequences != texts.len()
            || offsets.len() != result.num_sequences.saturating_add(1)
            || offsets.first().copied().unwrap_or(usize::MAX) != 0
            || offsets.last().copied().unwrap_or(usize::MAX) != ids.len()
        {
            return Err(error_from_last_or("Batch encode returned invalid offsets"));
        }

        let mut rows = Vec::with_capacity(result.num_sequences);
        for pair in offsets.windows(2) {
            let start = pair[0];
            let end = pair[1];
            if start > end || end > ids.len() {
                return Err(error_from_last_or("Batch encode returned invalid offsets"));
            }
            rows.push(ids[start..end].to_vec());
        }
        Ok(rows)
    }

    /// Decodes token IDs into owned UTF-8 text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let options = talu_sys::DecodeOptionsC {
            skip_special_tokens: if skip_special_tokens { 1 } else { 0 },
        };
        // SAFETY: tokenizer handle is valid and ids points to ids.len elements.
        let result =
            unsafe { talu_sys::talu_tokenizer_decode(self.ptr, ids.as_ptr(), ids.len(), &options) };
        if !result.error_msg.is_null() {
            return Err(error_from_last_or("Decode failed"));
        }
        let text = if result.text.is_null() || result.text_len == 0 {
            String::new()
        } else {
            // SAFETY: result.text is valid for result.text_len bytes.
            let bytes = unsafe { std::slice::from_raw_parts(result.text, result.text_len) };
            String::from_utf8_lossy(bytes).into_owned()
        };
        // SAFETY: result buffers were allocated by the C API.
        unsafe { talu_sys::talu_decode_result_free(result.text, result.text_len) };
        Ok(text)
    }

    /// Returns the tokenizer vocabulary size.
    pub fn vocab_size(&self) -> usize {
        // SAFETY: self.ptr is a valid tokenizer handle.
        unsafe { talu_sys::talu_tokenizer_get_vocab_size(self.ptr) }
    }

    /// Returns the full tokenizer vocabulary.
    pub fn vocab(&self) -> Result<Vec<TokenizerVocabEntry>> {
        // SAFETY: self.ptr is a valid tokenizer handle.
        let vocab = unsafe { talu_sys::talu_tokenizer_get_vocab(self.ptr) };
        if !vocab.error_msg.is_null() {
            return Err(error_from_last_or("Failed to fetch vocab"));
        }

        let mut entries = Vec::with_capacity(vocab.num_entries);
        for idx in 0..vocab.num_entries {
            // SAFETY: returned arrays have num_entries elements per C API contract.
            let token_ptr = unsafe { *vocab.tokens.add(idx) };
            let token_len = unsafe { *vocab.lengths.add(idx) as usize };
            let token_id = unsafe { *vocab.ids.add(idx) };
            let bytes = unsafe { std::slice::from_raw_parts(token_ptr as *const u8, token_len) };
            entries.push(TokenizerVocabEntry {
                token: String::from_utf8_lossy(bytes).into_owned(),
                id: token_id,
            });
        }

        // SAFETY: frees buffers returned by talu_tokenizer_get_vocab.
        unsafe {
            talu_sys::talu_vocab_result_free(
                vocab.tokens,
                vocab.lengths,
                vocab.ids,
                vocab.num_entries,
            )
        };
        Ok(entries)
    }

    /// Looks up a token string and returns its token ID.
    pub fn token_to_id(&self, token: &str) -> Result<u32> {
        // SAFETY: Clears the thread-local error buffer before a sentinel-returning lookup.
        unsafe { talu_sys::talu_clear_error() };
        // SAFETY: self.ptr is valid and token bytes are valid for the call.
        let id = unsafe {
            talu_sys::talu_tokenizer_token_to_id(self.ptr, token.as_bytes().as_ptr(), token.len())
        };
        if let Some(err) = crate::error::last_error_message() {
            return Err(crate::Error::talu(err));
        }
        u32::try_from(id).map_err(|_| crate::Error::talu(format!("token {token:?} was not found")))
    }

    /// Looks up a token ID and returns its token string.
    pub fn id_to_token(&self, token_id: i32) -> Result<String> {
        let mut out_ptr: *mut c_char = std::ptr::null_mut();
        // SAFETY: out pointer is valid and self.ptr is a valid tokenizer handle.
        let rc = unsafe {
            talu_sys::talu_tokenizer_id_to_token(
                self.ptr,
                token_id,
                &mut out_ptr as *mut _ as *mut c_void,
            )
        };
        if rc != 0 {
            let detail = crate::error::last_error_message()
                .unwrap_or_else(|| "token id is not in vocabulary".to_string());
            return Err(crate::Error::talu(format!(
                "token id {token_id} lookup failed: {detail}"
            )));
        }
        let text = TextPtr::new(out_ptr)
            .ok_or_else(|| {
                crate::Error::generic(format!("token id {token_id} returned null token"))
            })?
            .to_string_lossy();
        Ok(text)
    }

    /// Returns tokenizer special token IDs.
    pub fn special_tokens(&self) -> TokenizerSpecialTokens {
        // SAFETY: self.ptr is a valid tokenizer handle.
        let special = unsafe { talu_sys::talu_tokenizer_get_special_tokens(self.ptr) };
        TokenizerSpecialTokens {
            bos_token_id: special.bos_token_id,
            unk_token_id: special.unk_token_id,
            pad_token_id: special.pad_token_id,
        }
    }

    /// Returns the resolved tokenizer model directory, if available.
    pub fn model_dir(&self) -> Option<String> {
        let mut out_ptr: *mut c_char = std::ptr::null_mut();
        // SAFETY: out pointer is valid and self.ptr is a valid tokenizer handle.
        let rc = unsafe {
            talu_sys::talu_tokenizer_get_model_dir(self.ptr, &mut out_ptr as *mut _ as *mut c_void)
        };
        if rc != 0 || out_ptr.is_null() {
            return None;
        }
        let text = TextPtr::new(out_ptr)?.to_string_lossy();
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
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

fn encode_options_to_c(options: TokenizerEncodeOptions) -> talu_sys::EncodeOptions {
    let mut c_options = talu_sys::EncodeOptions::default();
    c_options.add_bos = if options.add_bos { 1 } else { 0 };
    c_options.add_eos = if options.add_eos { 1 } else { 0 };
    if let Some(truncation) = options.truncation {
        c_options.truncation = 1;
        c_options.max_length = truncation.max_length;
        c_options.truncation_side = match truncation.side {
            TokenizerTruncationSide::Right => 0,
            TokenizerTruncationSide::Left => 1,
        };
    }
    c_options
}

fn tokenizer_encoding_from_c(result: &talu_sys::EncodeResult) -> TokenizerEncoding {
    TokenizerEncoding {
        ids: copy_u32_slice(result.ids, result.num_tokens),
        attention_mask: copy_u32_slice(result.attention_mask, result.num_tokens),
        special_tokens_mask: copy_u32_slice(result.special_tokens_mask, result.num_tokens),
        offsets: copy_token_offsets_slice(result.offsets, result.num_tokens),
    }
}

fn copy_u32_slice(ptr: *mut u32, len: usize) -> Vec<u32> {
    if ptr.is_null() || len == 0 {
        return Vec::new();
    }
    // SAFETY: caller passes pointers returned by the C API with the matching length.
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}

fn copy_usize_slice(ptr: *mut usize, len: usize) -> Vec<usize> {
    if ptr.is_null() || len == 0 {
        return Vec::new();
    }
    // SAFETY: caller passes pointers returned by the C API with the matching length.
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}

fn copy_token_offsets_slice(ptr: *mut talu_sys::TokenOffset, len: usize) -> Vec<[u32; 2]> {
    if ptr.is_null() || len == 0 {
        return Vec::new();
    }
    // SAFETY: caller passes pointers returned by the C API with the matching length.
    unsafe { std::slice::from_raw_parts(ptr, len) }
        .iter()
        .map(|offset| [offset.start, offset.end])
        .collect()
}

/// RAII wrapper for a text pointer allocated by the C API.
pub(crate) struct TextPtr {
    ptr: *mut c_char,
}

impl TextPtr {
    /// Creates a TextPtr from a raw pointer, returning None if null.
    pub(crate) fn new(ptr: *mut c_char) -> Option<Self> {
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Converts the text to a lossy String.
    pub(crate) fn to_string_lossy(&self) -> String {
        // SAFETY: self.ptr is non-null (checked in new()) and points to a valid C string.
        unsafe { CStr::from_ptr(self.ptr) }
            .to_string_lossy()
            .to_string()
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
pub(crate) struct CanonicalSpec {
    ptr: *mut c_void,
}

impl CanonicalSpec {
    /// Returns a const pointer to the canonical spec.
    pub(crate) fn as_ptr(&self) -> *const c_void {
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

/// Static model metadata returned by an inference backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BackendModelInfo {
    pub file_size: u64,
    pub tensor_count: u64,
    pub vocab_size: i32,
    pub d_model: i32,
    pub n_layers: i32,
    pub n_heads: i32,
    pub n_kv_groups: i32,
    pub d_ff: i32,
    pub max_seq_len: i32,
    pub gaffine_group_size: i32,
    pub weight_dtype: u8,
}

impl BackendModelInfo {
    pub fn is_empty(&self) -> bool {
        self.file_size == 0 && self.d_model == 0
    }
}

/// Negative log-likelihood scoring result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenNllScore {
    pub nll_sum: f64,
    pub scored_tokens: usize,
}

/// Joint scoring result for two backends on the same token stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointTokenScore {
    pub reference_nll_sum: f64,
    pub model_nll_sum: f64,
    pub kld_sum: f64,
    pub scored_tokens: usize,
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

    pub(crate) fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Return static model metadata from a local backend.
    /// Returns a zero struct for non-local backends.
    pub fn model_metadata(&self) -> BackendModelInfo {
        let info = unsafe { talu_sys::talu_backend_model_info(self.ptr) };
        BackendModelInfo {
            file_size: info.file_size,
            tensor_count: info.tensor_count,
            vocab_size: info.vocab_size,
            d_model: info.d_model,
            n_layers: info.n_layers,
            n_heads: info.n_heads,
            n_kv_groups: info.n_kv_groups,
            d_ff: info.d_ff,
            max_seq_len: info.max_seq_len,
            gaffine_group_size: info.gaffine_group_size,
            weight_dtype: info.weight_dtype,
        }
    }

    /// Synchronizes backend work and returns any backend error.
    pub fn synchronize(&self) -> Result<()> {
        let rc = unsafe { talu_sys::talu_backend_synchronize(self.ptr) };
        if rc != 0 {
            return Err(error_from_last_or("backend synchronize failed"));
        }
        Ok(())
    }

    /// Scores target tokens using negative log-likelihood.
    pub fn score_tokens_nll(
        &self,
        context: &[u32],
        targets: &[u32],
        max_context: usize,
    ) -> Result<TokenNllScore> {
        let mut nll_sum = 0.0;
        let mut scored_tokens = 0usize;
        let rc = unsafe {
            talu_sys::talu_scheduler_score_tokens_nll(
                self.ptr,
                nullable_u32_ptr(context),
                context.len(),
                nullable_u32_ptr(targets),
                targets.len(),
                max_context,
                &mut nll_sum as *mut f64 as *mut c_void,
                &mut scored_tokens as *mut usize as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(error_from_last_or("scoring failed"));
        }
        Ok(TokenNllScore {
            nll_sum,
            scored_tokens,
        })
    }

    /// Scores target tokens against a reference backend and this backend.
    pub fn score_tokens_joint(
        reference: &InferenceBackend,
        model: &InferenceBackend,
        context: &[u32],
        targets: &[u32],
        max_context: usize,
    ) -> Result<JointTokenScore> {
        let mut reference_nll_sum = 0.0;
        let mut model_nll_sum = 0.0;
        let mut kld_sum = 0.0;
        let mut scored_tokens = 0usize;
        let rc = unsafe {
            talu_sys::talu_scheduler_score_tokens_joint(
                reference.ptr,
                model.ptr,
                nullable_u32_ptr(context),
                context.len(),
                nullable_u32_ptr(targets),
                targets.len(),
                max_context,
                &mut reference_nll_sum as *mut f64 as *mut c_void,
                &mut model_nll_sum as *mut f64 as *mut c_void,
                &mut kld_sum as *mut f64 as *mut c_void,
                &mut scored_tokens as *mut usize as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(error_from_last_or("joint scoring failed"));
        }
        Ok(JointTokenScore {
            reference_nll_sum,
            model_nll_sum,
            kld_sum,
            scored_tokens,
        })
    }
}

fn nullable_u32_ptr(values: &[u32]) -> *const u32 {
    if values.is_empty() {
        std::ptr::null()
    } else {
        values.as_ptr()
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

struct OwnedGenerateResult {
    text: Option<String>,
    token_count: usize,
    prompt_tokens: usize,
    completion_tokens: usize,
    prefill_ns: u64,
    generation_ns: u64,
    ttft_ns: u64,
    error_code: i32,
    finish_reason: FinishReason,
    tool_calls: Vec<ToolCall>,
}

/// RAII wrapper for generation results.
pub struct GenerateResult {
    inner: OwnedGenerateResult,
}

impl GenerateResult {
    pub(crate) fn from_batch(result: crate::batch::BatchResult) -> Self {
        Self {
            inner: OwnedGenerateResult {
                token_count: result.completion_tokens,
                prompt_tokens: result.prompt_tokens,
                completion_tokens: result.completion_tokens,
                prefill_ns: result.prefill_ns,
                generation_ns: result.generation_ns,
                ttft_ns: result.ttft_ns,
                error_code: 0,
                finish_reason: result.finish_reason,
                text: result.text,
                tool_calls: result.tool_calls,
            },
        }
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
        self.inner.text.clone()
    }

    /// Returns the finish reason for this generation.
    pub fn finish_reason(&self) -> FinishReason {
        self.inner.finish_reason
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
        self.inner.tool_calls.clone()
    }
}
