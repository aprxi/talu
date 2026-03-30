//! Serde types for the OpenAI-compatible `/v1/chat/completions` endpoint.

use serde::{Deserialize, Serialize};

/// Request body for `POST /v1/chat/completions`.
#[derive(Debug, Deserialize)]
pub struct CreateChatCompletionBody {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<i64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub stream: Option<bool>,
    pub seed: Option<u64>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub tools: Option<serde_json::Value>,
    pub tool_choice: Option<serde_json::Value>,
    pub max_completion_tokens: Option<i64>,
}

/// A single message in the chat completions request.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_calls: Option<serde_json::Value>,
    pub tool_call_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Response types (non-streaming)
// ---------------------------------------------------------------------------

/// Non-streaming response for `POST /v1/chat/completions`.
#[derive(Debug, Serialize)]
pub struct ChatCompletion {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: CompletionUsage,
}

/// A single choice in a chat completion response.
#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: String,
}

/// The assistant message returned in a choice.
#[derive(Debug, Serialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
}

/// Token usage statistics.
#[derive(Debug, Serialize, Clone)]
pub struct CompletionUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

// ---------------------------------------------------------------------------
// Streaming types (SSE chunks)
// ---------------------------------------------------------------------------

/// A single streaming chunk for `POST /v1/chat/completions` with `stream=true`.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<CompletionUsage>,
}

/// A single choice within a streaming chunk.
#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

/// The delta content in a streaming chunk.
#[derive(Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}
