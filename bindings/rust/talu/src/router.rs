//! Safe wrappers for talu generation/inference routing.

use crate::error::error_from_last_or;
use crate::{ChatHandle, GenerateResult, InferenceBackend, Result};
use std::ffi::{c_void, CStr, CString};

/// Content type for generation input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ContentType {
    Text = 0,
    ImageUrl = 1,
    ImageBase64 = 2,
    AudioUrl = 3,
    AudioBase64 = 4,
}

/// A content part for generation input.
#[derive(Debug, Clone)]
pub enum ContentPart {
    /// Plain text content.
    Text(String),
    /// Image from URL.
    ImageUrl { url: String, mime: Option<String> },
    /// Image from base64 data.
    ImageBase64 { data: Vec<u8>, mime: String },
    /// Audio from URL.
    AudioUrl { url: String, mime: Option<String> },
    /// Audio from base64 data.
    AudioBase64 { data: Vec<u8>, mime: String },
}

/// Configuration for text generation.
#[derive(Default)]
pub struct GenerateConfig {
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature (0.0 = deterministic).
    pub temperature: f32,
    /// Top-k sampling parameter.
    pub top_k: usize,
    /// Top-p (nucleus) sampling parameter.
    pub top_p: f32,
    /// Min-p sampling parameter.
    pub min_p: f32,
    /// Repetition penalty.
    pub repetition_penalty: f32,
    /// Random seed (0 = random).
    pub seed: u64,
    /// Override the chat template.
    pub template_override: Option<String>,
    /// Tools definition JSON.
    pub tools_json: Option<String>,
    /// Tool choice specification.
    pub tool_choice: Option<String>,
    /// Extra body JSON for API requests.
    pub extra_body_json: Option<String>,
    /// Optional stop flag for cancellation (pointer to AtomicBool).
    /// When set to true, generation will stop gracefully.
    pub stop_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Preserve raw model output without reasoning-tag filtering.
    pub raw_output: bool,
    /// Optional prefill progress callback. Called once per transformer layer
    /// during prefill (not decode). Arguments: (completed_layers, total_layers).
    pub prefill_progress: Option<PrefillProgressCallback>,
}

/// Callback for prefill progress. Called once per transformer layer.
pub type PrefillProgressCallback = Box<dyn Fn(usize, usize) + Send + Sync>;

/// A single streamed token with content classification metadata.
///
/// Each token carries `item_type` and `content_type` discriminators from the
/// responses type system, enabling correct SSE event routing and display.
#[derive(Debug, Clone)]
pub struct StreamToken<'a> {
    /// Decoded token text.
    pub text: &'a str,
    /// Item type (e.g. Message, FunctionCall, Reasoning).
    pub item_type: talu_sys::ItemType,
    /// Content type (e.g. OutputText, ReasoningText).
    pub content_type: talu_sys::ContentType,
}

/// Streaming callback for token-by-token generation.
///
/// The callback receives a `StreamToken` with text and content classification,
/// and should return `true` to continue streaming or `false` to stop.
pub type StreamCallback = Box<dyn FnMut(&StreamToken) -> bool + Send>;

/// Result from streaming generation, containing stats after completion.
#[derive(Debug, Clone)]
pub struct StreamResult {
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,
    /// Number of tokens generated.
    pub completion_tokens: usize,
    /// Prefill time in nanoseconds.
    pub prefill_ns: u64,
    /// Generation time in nanoseconds.
    pub generation_ns: u64,
    /// Reason generation stopped.
    pub finish_reason: crate::FinishReason,
}

impl StreamResult {
    /// Calculate tokens per second for generation.
    pub fn tokens_per_second(&self) -> f64 {
        if self.generation_ns > 0 {
            (self.completion_tokens as f64) / (self.generation_ns as f64 / 1_000_000_000.0)
        } else {
            0.0
        }
    }

    /// Calculate tokens per second for prefill.
    pub fn prefill_tokens_per_second(&self) -> f64 {
        if self.prefill_ns > 0 {
            (self.prompt_tokens as f64) / (self.prefill_ns as f64 / 1_000_000_000.0)
        } else {
            0.0
        }
    }
}

/// Holds C strings and raw parts for the duration of a generate call.
struct ContentPartsHolder {
    _strings: Vec<CString>,
    _data: Vec<Vec<u8>>,
    parts: Vec<talu_sys::GenerateContentPart>,
}

impl ContentPartsHolder {
    fn new(content: &[ContentPart]) -> Result<Self> {
        let mut strings = Vec::new();
        let mut data_vecs = Vec::new();
        let mut parts = Vec::with_capacity(content.len());

        for part in content {
            match part {
                ContentPart::Text(text) => {
                    let c_str = CString::new(text.as_str())?;
                    let ptr = c_str.as_ptr() as *const u8;
                    let len = text.len();
                    strings.push(c_str);
                    parts.push(talu_sys::GenerateContentPart {
                        content_type: ContentType::Text as u8,
                        data_ptr: ptr,
                        data_len: len,
                        mime_ptr: std::ptr::null(),
                    });
                }
                ContentPart::ImageUrl { url, mime } => {
                    let c_url = CString::new(url.as_str())?;
                    let c_mime = mime
                        .as_ref()
                        .map(|m| CString::new(m.as_str()))
                        .transpose()?;
                    let url_ptr = c_url.as_ptr() as *const u8;
                    let url_len = url.len();
                    let mime_ptr = c_mime
                        .as_ref()
                        .map(|m| m.as_ptr())
                        .unwrap_or(std::ptr::null());
                    strings.push(c_url);
                    if let Some(m) = c_mime {
                        strings.push(m);
                    }
                    parts.push(talu_sys::GenerateContentPart {
                        content_type: ContentType::ImageUrl as u8,
                        data_ptr: url_ptr,
                        data_len: url_len,
                        mime_ptr,
                    });
                }
                ContentPart::ImageBase64 { data, mime } => {
                    let c_mime = CString::new(mime.as_str())?;
                    let mime_ptr = c_mime.as_ptr();
                    strings.push(c_mime);
                    let data_clone = data.clone();
                    let data_ptr = data_clone.as_ptr();
                    let data_len = data_clone.len();
                    data_vecs.push(data_clone);
                    parts.push(talu_sys::GenerateContentPart {
                        content_type: ContentType::ImageBase64 as u8,
                        data_ptr,
                        data_len,
                        mime_ptr,
                    });
                }
                ContentPart::AudioUrl { url, mime } => {
                    let c_url = CString::new(url.as_str())?;
                    let c_mime = mime
                        .as_ref()
                        .map(|m| CString::new(m.as_str()))
                        .transpose()?;
                    let url_ptr = c_url.as_ptr() as *const u8;
                    let url_len = url.len();
                    let mime_ptr = c_mime
                        .as_ref()
                        .map(|m| m.as_ptr())
                        .unwrap_or(std::ptr::null());
                    strings.push(c_url);
                    if let Some(m) = c_mime {
                        strings.push(m);
                    }
                    parts.push(talu_sys::GenerateContentPart {
                        content_type: ContentType::AudioUrl as u8,
                        data_ptr: url_ptr,
                        data_len: url_len,
                        mime_ptr,
                    });
                }
                ContentPart::AudioBase64 { data, mime } => {
                    let c_mime = CString::new(mime.as_str())?;
                    let mime_ptr = c_mime.as_ptr();
                    strings.push(c_mime);
                    let data_clone = data.clone();
                    let data_ptr = data_clone.as_ptr();
                    let data_len = data_clone.len();
                    data_vecs.push(data_clone);
                    parts.push(talu_sys::GenerateContentPart {
                        content_type: ContentType::AudioBase64 as u8,
                        data_ptr,
                        data_len,
                        mime_ptr,
                    });
                }
            }
        }

        Ok(Self {
            _strings: strings,
            _data: data_vecs,
            parts,
        })
    }

    fn as_ptr(&self) -> *const talu_sys::GenerateContentPart {
        if self.parts.is_empty() {
            std::ptr::null()
        } else {
            self.parts.as_ptr()
        }
    }

    fn len(&self) -> usize {
        self.parts.len()
    }
}

/// C trampoline for prefill progress callback.
extern "C" fn prefill_progress_trampoline(completed: usize, total: usize, userdata: *mut c_void) {
    if !userdata.is_null() {
        // SAFETY: userdata points to a PrefillProgressCallback (Box<dyn Fn(usize,usize)>)
        // that outlives this call (owned by the GenerateConfig passed to generate_stream).
        let cb = unsafe { &*(userdata as *const PrefillProgressCallback) };
        cb(completed, total);
    }
}

/// Holds C strings for the GenerateConfig.
struct ConfigHolder {
    _template_override: Option<CString>,
    _tools_json: Option<CString>,
    _tool_choice: Option<CString>,
    _extra_body: Option<CString>,
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

        let stop_flag = cfg.stop_flag.clone();

        let mut c_config = talu_sys::CGenerateConfig::default();
        c_config.max_tokens = cfg.max_tokens;
        c_config.temperature = cfg.temperature;
        c_config.top_k = cfg.top_k;
        c_config.top_p = cfg.top_p;
        c_config.min_p = cfg.min_p;
        c_config.repetition_penalty = cfg.repetition_penalty;
        c_config.seed = cfg.seed;
        c_config.raw_output = if cfg.raw_output { 1 } else { 0 };

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

        if let Some(ref flag) = stop_flag {
            // SAFETY: We store a reference to the Arc's inner AtomicBool.
            // The Arc is kept alive in _stop_flag for the duration of this struct.
            c_config.stop_flag =
                flag.as_ref() as *const std::sync::atomic::AtomicBool as *mut c_void;
        }

        // Prefill progress: pass pointer to the original config's callback.
        // SAFETY: The GenerateConfig reference outlives both ConfigHolder and the C call.
        if let Some(ref cb) = cfg.prefill_progress {
            c_config.prefill_progress_fn = prefill_progress_trampoline as *mut c_void;
            c_config.prefill_progress_data = cb as *const PrefillProgressCallback as *mut c_void;
        }

        Ok(Self {
            _template_override: template_override,
            _tools_json: tools_json,
            _tool_choice: tool_choice,
            _extra_body: extra_body,
            _stop_flag: stop_flag,
            config: c_config,
        })
    }

    fn as_ptr(&self) -> *const talu_sys::CGenerateConfig {
        &self.config
    }
}

/// Generates text using the specified backend (non-streaming).
pub fn generate(
    chat: &ChatHandle,
    content: &[ContentPart],
    backend: &InferenceBackend,
    config: &GenerateConfig,
) -> Result<GenerateResult> {
    let parts_holder = ContentPartsHolder::new(content)?;
    let config_holder = ConfigHolder::new(config)?;

    // SAFETY: All pointers are valid. chat, backend are valid handles.
    // parts_holder and config_holder keep the underlying data alive.
    let result = unsafe {
        talu_sys::talu_router_generate_with_backend(
            chat.as_ptr(),
            parts_holder.as_ptr(),
            parts_holder.len(),
            backend.as_ptr(),
            config_holder.as_ptr(),
        )
    };

    if result.error_code != 0 {
        return Err(error_from_last_or("Generation failed"));
    }

    Ok(GenerateResult::new(result))
}

/// Generates text using the specified backend with streaming via iterator API.
///
/// Uses a pull-based iterator internally for reliable streaming without
/// callback lifetime issues. Returns stats after streaming completes.
pub fn generate_stream(
    chat: &ChatHandle,
    content: &[ContentPart],
    backend: &InferenceBackend,
    config: &GenerateConfig,
    mut callback: StreamCallback,
) -> Result<StreamResult> {
    let parts_holder = ContentPartsHolder::new(content)?;
    let config_holder = ConfigHolder::new(config)?;

    // Create iterator
    // SAFETY: All pointers are valid. chat, backend are valid handles.
    // parts_holder and config_holder keep the underlying data alive.
    let iterator = unsafe {
        talu_sys::talu_router_create_iterator(
            chat.as_ptr(),
            parts_holder.as_ptr(),
            parts_holder.len(),
            backend.as_ptr(),
            config_holder.as_ptr(),
        )
    };

    if iterator.is_null() {
        return Err(error_from_last_or("Failed to create iterator"));
    }

    // Poll for tokens
    loop {
        // SAFETY: iterator is valid (checked above)
        let token_ptr = unsafe { talu_sys::talu_router_iterator_next(iterator) };

        if token_ptr.is_null() {
            // Check for errors
            // SAFETY: iterator is valid
            if unsafe { talu_sys::talu_router_iterator_has_error(iterator) } {
                // Try to get error message first, fall back to error code
                let error_msg_ptr = unsafe { talu_sys::talu_router_iterator_error_msg(iterator) };
                let error_message = if !error_msg_ptr.is_null() {
                    // SAFETY: error_msg_ptr is valid C string from the iterator
                    unsafe { CStr::from_ptr(error_msg_ptr) }
                        .to_str()
                        .ok()
                        .map(|s| s.to_string())
                } else {
                    None
                };
                let error_code = unsafe { talu_sys::talu_router_iterator_error_code(iterator) };
                // SAFETY: iterator is valid
                unsafe { talu_sys::talu_router_iterator_free(iterator) };
                return Err(crate::error::Error::generic(error_message.unwrap_or_else(
                    || format!("Generation failed with error code {}", error_code),
                )));
            }
            break;
        }

        // SAFETY: token_ptr is valid C string from the iterator
        let token = unsafe { CStr::from_ptr(token_ptr) }.to_str().unwrap_or("");

        // Read content classification for this token
        // SAFETY: iterator is valid
        let item_type =
            talu_sys::ItemType::from(unsafe { talu_sys::talu_router_iterator_item_type(iterator) });
        let content_type = talu_sys::ContentType::from(unsafe {
            talu_sys::talu_router_iterator_content_type(iterator)
        });

        let stream_token = StreamToken {
            text: token,
            item_type,
            content_type,
        };

        // Call user callback
        if !callback(&stream_token) {
            // User requested stop
            // SAFETY: iterator is valid
            unsafe { talu_sys::talu_router_iterator_cancel(iterator) };
            break;
        }
    }

    // Get stats before freeing iterator
    // SAFETY: iterator is valid
    let result = StreamResult {
        prompt_tokens: unsafe { talu_sys::talu_router_iterator_prompt_tokens(iterator) },
        completion_tokens: unsafe { talu_sys::talu_router_iterator_completion_tokens(iterator) },
        prefill_ns: unsafe { talu_sys::talu_router_iterator_prefill_ns(iterator) },
        generation_ns: unsafe { talu_sys::talu_router_iterator_generation_ns(iterator) },
        finish_reason: crate::FinishReason::from(talu_sys::CFinishReason::from(unsafe {
            talu_sys::talu_router_iterator_finish_reason(iterator)
        })),
    };

    // Free iterator
    // SAFETY: iterator is valid
    unsafe { talu_sys::talu_router_iterator_free(iterator) };

    Ok(result)
}

/// Serializes the chat conversation to an OpenAI Completions-format JSON string.
///
/// Returns the serialized messages, or an empty string on failure.
pub fn to_json(chat: &ChatHandle) -> Result<String> {
    // SAFETY: chat.as_ptr() is a valid chat handle.
    let ptr = unsafe { talu_sys::talu_chat_to_json(chat.as_ptr()) };
    if ptr.is_null() {
        return Err(error_from_last_or("Failed to serialize chat to JSON"));
    }
    // SAFETY: Non-null ptr is a heap-allocated C string from the C API.
    let text = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().to_string();
    // SAFETY: ptr was allocated by the C API and must be freed exactly once.
    unsafe { talu_sys::talu_text_free(ptr as *mut std::os::raw::c_char) };
    Ok(text)
}

/// Loads conversation messages from an OpenAI Completions-format JSON string.
///
/// Replaces the current conversation in the chat with the parsed messages.
pub fn set_messages(chat: &ChatHandle, json: &str) -> Result<()> {
    let c_json = CString::new(json)?;
    // SAFETY: chat.as_ptr() is a valid chat handle; c_json is a valid CString.
    let rc = unsafe { talu_sys::talu_chat_set_messages(chat.as_ptr(), c_json.as_ptr()) };
    if rc != 0 {
        return Err(error_from_last_or("Failed to set messages from JSON"));
    }
    Ok(())
}

/// Convenience function to generate text from a simple string prompt.
pub fn generate_text(
    chat: &ChatHandle,
    text: &str,
    backend: &InferenceBackend,
    config: &GenerateConfig,
) -> Result<GenerateResult> {
    let content = vec![ContentPart::Text(text.to_string())];
    generate(chat, &content, backend, config)
}

/// Convenience function to stream text from a simple string prompt.
pub fn stream_text(
    chat: &ChatHandle,
    text: &str,
    backend: &InferenceBackend,
    config: &GenerateConfig,
    callback: StreamCallback,
) -> Result<StreamResult> {
    let content = vec![ContentPart::Text(text.to_string())];
    generate_stream(chat, &content, backend, config, callback)
}
