//! Safe wrappers for talu generation/inference routing.

use crate::batch::{BatchHandle, EventType};
use crate::error::error_from_last_or;
use crate::responses::ResponsesView;
use crate::{ChatHandle, GenerateResult, InferenceBackend, Result};
use std::ffi::{CStr, CString};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn has_visible_text(text: &str) -> bool {
    text.chars().any(|ch| {
        if ch.is_ascii() {
            !ch.is_ascii_whitespace() && !ch.is_ascii_control()
        } else {
            !ch.is_whitespace()
        }
    })
}

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
///
/// Sampling parameters default to `-1.0` (unset), matching the C API sentinel.
/// The Zig core treats negative values as "use model/chat defaults". Callers
/// must explicitly set any parameter they want to override.
#[derive(Debug, Clone)]
pub struct VisionPrefillImage {
    pub pixels: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub grid_temporal: u32,
    pub grid_height: u32,
    pub grid_width: u32,
    pub token_count: usize,
}

#[derive(Debug, Clone)]
pub struct VisionPrefillInput {
    pub image_token_id: u32,
    pub images: Vec<VisionPrefillImage>,
}

pub struct GenerateConfig {
    /// Maximum number of tokens to generate (total: thinking + answer).
    pub max_tokens: usize,
    /// Maximum tokens for the answer/completion only (excludes thinking).
    pub max_completion_tokens: Option<usize>,
    /// Maximum thinking/reasoning tokens. Overrides effort-derived budget when set.
    pub max_reasoning_tokens: Option<usize>,
    /// Sampling temperature (0.0 = deterministic, -1.0 = unset).
    pub temperature: f32,
    /// Top-k sampling parameter.
    pub top_k: usize,
    /// Top-p (nucleus) sampling parameter (-1.0 = unset).
    pub top_p: f32,
    /// Min-p sampling parameter (-1.0 = unset).
    pub min_p: f32,
    /// Repetition penalty (multiplicative, 1.0 = neutral, -1.0 = unset).
    pub repetition_penalty: f32,
    /// Additive presence penalty (0.0 = disabled, -1.0 = unset).
    pub presence_penalty: f32,
    /// Additive frequency penalty (0.0 = disabled, -1.0 = unset).
    pub frequency_penalty: f32,
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
    /// Reasoning effort level (e.g. "none", "low", "medium", "high").
    /// Passed through to the Zig core which maps it to template variables like `enable_thinking`.
    pub reasoning_effort: Option<String>,
    /// Optional stop flag for cancellation (pointer to AtomicBool).
    /// When set to true, generation will stop gracefully.
    pub stop_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Preserve raw model output without reasoning-tag filtering.
    pub raw_output: bool,
    /// When true, behave like a standard completions endpoint: no thinking
    /// intervention, no reasoning separation, just raw token generation.
    pub completions_mode: bool,
    /// Optional prefill progress callback. Called once per transformer layer
    /// during prefill (not decode). Arguments: (completed_layers, total_layers).
    pub prefill_progress: Option<PrefillProgressCallback>,
    /// Optional externally prepared vision input. When set, local image
    /// decode/preprocess is skipped and this payload is used directly.
    pub vision_prefill: Option<VisionPrefillInput>,
}

/// Core validation inputs for chat-completions request contracts.
pub struct CompletionsValidationRequest<'a> {
    pub messages_json: &'a [u8],
    pub max_tokens: Option<i64>,
    pub max_completion_tokens: Option<i64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub tools_json: Option<&'a [u8]>,
    pub tool_choice_json: Option<&'a [u8]>,
}

/// Validates chat-completions request inputs using the core contract.
pub fn validate_completions_request(request: CompletionsValidationRequest<'_>) -> Result<()> {
    let (tools_ptr, tools_len) = nullable_bytes(request.tools_json);
    let (tool_choice_ptr, tool_choice_len) = nullable_bytes(request.tool_choice_json);

    let rc = unsafe {
        talu_sys::talu_completions_validate_request(
            request.messages_json.as_ptr(),
            request.messages_json.len(),
            usize::from(request.max_tokens.is_some()),
            request.max_tokens.unwrap_or_default(),
            usize::from(request.max_completion_tokens.is_some()),
            request.max_completion_tokens.unwrap_or_default(),
            request.temperature.unwrap_or(f64::NAN),
            request.top_p.unwrap_or(f64::NAN),
            request.presence_penalty.unwrap_or(f64::NAN),
            request.frequency_penalty.unwrap_or(f64::NAN),
            tools_ptr,
            tools_len,
            tool_choice_ptr,
            tool_choice_len,
        )
    };
    if rc != 0 {
        return Err(error_from_last_or("invalid chat/completions request"));
    }
    Ok(())
}

fn nullable_bytes(value: Option<&[u8]>) -> (*const u8, usize) {
    match value {
        Some(bytes) => (bytes.as_ptr(), bytes.len()),
        None => (std::ptr::null(), 0),
    }
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 0,
            max_completion_tokens: None,
            max_reasoning_tokens: None,
            temperature: -1.0,
            top_k: 0,
            top_p: -1.0,
            min_p: -1.0,
            repetition_penalty: -1.0,
            presence_penalty: -1.0,
            frequency_penalty: -1.0,
            seed: 0,
            template_override: None,
            tools_json: None,
            tool_choice: None,
            extra_body_json: None,
            reasoning_effort: None,
            stop_flag: None,
            raw_output: false,
            completions_mode: false,
            prefill_progress: None,
            vision_prefill: None,
        }
    }
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
    pub item_type: crate::responses::ItemType,
    /// Content type (e.g. OutputText, ReasoningText).
    pub content_type: crate::responses::ContentType,
    /// Cumulative count of tokens streamed so far (from engine).
    pub tokens_generated: usize,
    /// Nanoseconds elapsed since first token (from engine timestamps).
    pub elapsed_ns: u64,
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
    /// Time-to-first-token in nanoseconds.
    pub ttft_ns: u64,
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

/// Generates text using the specified backend (non-streaming).
pub fn generate(
    chat: &ChatHandle,
    content: &[ContentPart],
    backend: &InferenceBackend,
    config: &GenerateConfig,
) -> Result<GenerateResult> {
    if config.vision_prefill.is_some() {
        return Err(crate::Error::generic(
            "local non-stream generation with prepared vision input is not supported by the batch scheduler yet",
        ));
    }

    append_batch_user_content(chat, content)?;

    let batch = BatchHandle::new(backend, None)?;
    let request_id = batch.submit(chat, config)?;
    let pending = AtomicBool::new(false);
    let mut saw_completion = false;

    batch.run_loop_final_only(&pending, |event| {
        if event.request_id == request_id && matches!(event.event_type, EventType::Completed) {
            saw_completion = true;
        }
    })?;

    let result = batch.take_result(request_id).ok_or_else(|| {
        if saw_completion {
            crate::Error::generic("batch generation completed without a result")
        } else {
            crate::Error::generic("batch generation ended before the request completed")
        }
    })?;

    Ok(GenerateResult::from_batch(result))
}

fn append_batch_user_content(chat: &ChatHandle, content: &[ContentPart]) -> Result<()> {
    if content.is_empty() {
        if chat.item_count() == 0 {
            return Err(crate::Error::generic(
                "generation requires either content or an existing conversation",
            ));
        }
        return Ok(());
    }

    let mut text = String::new();
    for part in content {
        match part {
            ContentPart::Text(part_text) => text.push_str(part_text),
            ContentPart::ImageUrl { .. } | ContentPart::ImageBase64 { .. } => {
                return Err(crate::Error::generic(
                    "local image generation is not supported by the batch scheduler yet",
                ));
            }
            ContentPart::AudioUrl { .. } | ContentPart::AudioBase64 { .. } => {
                return Err(crate::Error::generic(
                    "local audio generation is not supported by the batch scheduler yet",
                ));
            }
        }
    }

    if text.is_empty() {
        return Err(crate::Error::generic("generation prompt is empty"));
    }

    chat.append_user_message(&text)
}

fn config_with_batch_stop_flag(
    config: &GenerateConfig,
    stop_flag: Arc<AtomicBool>,
) -> GenerateConfig {
    GenerateConfig {
        max_tokens: config.max_tokens,
        max_completion_tokens: config.max_completion_tokens,
        max_reasoning_tokens: config.max_reasoning_tokens,
        temperature: config.temperature,
        top_k: config.top_k,
        top_p: config.top_p,
        min_p: config.min_p,
        repetition_penalty: config.repetition_penalty,
        presence_penalty: config.presence_penalty,
        frequency_penalty: config.frequency_penalty,
        seed: config.seed,
        template_override: config.template_override.clone(),
        tools_json: config.tools_json.clone(),
        tool_choice: config.tool_choice.clone(),
        extra_body_json: config.extra_body_json.clone(),
        reasoning_effort: config.reasoning_effort.clone(),
        stop_flag: Some(stop_flag),
        raw_output: config.raw_output,
        completions_mode: config.completions_mode,
        prefill_progress: None,
        vision_prefill: config.vision_prefill.clone(),
    }
}

/// Generates text using the specified backend with streaming callbacks.
///
/// Local streaming generation is driven through the batch scheduler. Text
/// decoding, reasoning filtering, grammar constraints, and final stats all
/// come from the same batch request lifecycle used by non-stream generation.
pub fn generate_stream(
    chat: &ChatHandle,
    content: &[ContentPart],
    backend: &InferenceBackend,
    config: &GenerateConfig,
    mut callback: StreamCallback,
) -> Result<StreamResult> {
    if config.vision_prefill.is_some() {
        return Err(crate::Error::generic(
            "local streaming generation with prepared vision input is not supported by the batch scheduler yet",
        ));
    }

    append_batch_user_content(chat, content)?;

    let stop_flag = config
        .stop_flag
        .clone()
        .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
    let batch_config = config_with_batch_stop_flag(config, stop_flag.clone());
    let batch = BatchHandle::new(backend, None)?;
    let request_id = batch.submit(chat, &batch_config)?;
    let pending = AtomicBool::new(false);
    let mut emitted_visible = false;
    let mut callback_stopped = false;

    batch.run_loop_borrowed(&pending, |event| {
        if event.request_id != request_id || callback_stopped {
            return;
        }
        if !matches!(event.event_type, EventType::TextDelta) || event.text.is_empty() {
            return;
        }
        if !emitted_visible && has_visible_text(event.text) {
            emitted_visible = true;
        }
        let stream_token = StreamToken {
            text: event.text,
            item_type: event.item_type,
            content_type: event.content_type,
            tokens_generated: event.tokens_generated,
            elapsed_ns: event.timestamp_ns.max(0) as u64,
        };
        if !callback(&stream_token) {
            callback_stopped = true;
            stop_flag.store(true, Ordering::Release);
            pending.store(true, Ordering::Release);
        }
    })?;

    if callback_stopped && batch.has_active() {
        let drain_pending = AtomicBool::new(false);
        batch.run_loop_final_only(&drain_pending, |_| {})?;
    }

    let result = batch
        .take_result(request_id)
        .ok_or_else(|| crate::Error::generic("batch stream completed without a result"))?;

    if !callback_stopped && !emitted_visible {
        if let Ok(Some(text)) = chat.responses().last_assistant_message_text() {
            if has_visible_text(&text) {
                let fallback = StreamToken {
                    text: text.as_str(),
                    item_type: crate::responses::ItemType::Message,
                    content_type: crate::responses::ContentType::OutputText,
                    tokens_generated: result.completion_tokens,
                    elapsed_ns: result.generation_ns,
                };
                let _ = callback(&fallback);
            }
        }
    }

    Ok(StreamResult {
        prompt_tokens: result.prompt_tokens,
        completion_tokens: result.completion_tokens,
        prefill_ns: result.prefill_ns,
        generation_ns: result.generation_ns,
        ttft_ns: result.ttft_ns,
        finish_reason: result.finish_reason,
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_batch_user_content_rejects_empty_conversation_without_content() {
        let chat = ChatHandle::new(None).expect("chat should be created");
        let err = append_batch_user_content(&chat, &[]).expect_err("empty input must fail");
        assert!(
            err.to_string().contains("requires either content"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn append_batch_user_content_rejects_image_content() {
        let chat = ChatHandle::new(None).expect("chat should be created");
        let content = [ContentPart::ImageUrl {
            url: "data:image/png;base64,AAAA".to_string(),
            mime: Some("image/png".to_string()),
        }];
        let err = append_batch_user_content(&chat, &content).expect_err("image input must fail");
        assert!(
            err.to_string().contains("image generation"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn append_batch_user_content_appends_text_message() {
        let chat = ChatHandle::new(None).expect("chat should be created");
        let content = [ContentPart::Text("hello".to_string())];

        append_batch_user_content(&chat, &content).expect("text append should succeed");

        assert_eq!(chat.item_count(), 1);
    }
}
