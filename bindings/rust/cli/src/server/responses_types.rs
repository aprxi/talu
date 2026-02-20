use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Request body for POST /v1/responses.
///
/// The full request JSON is forwarded to the Zig core as-is; the fields
/// listed here are the ones the server inspects or that Swagger UI needs
/// to render useful input forms.
#[derive(Debug, Deserialize, ToSchema)]
pub struct CreateResponseBody {
    pub model: Option<String>,
    /// The prompt. Can be a plain string or an array of structured messages.
    #[schema(example = "What is the capital of France?")]
    pub input: serde_json::Value,
    pub stream: Option<bool>,
    pub store: Option<bool>,
    pub previous_response_id: Option<String>,
    pub session_id: Option<String>,
    pub prompt_id: Option<String>,
    pub tools: Option<serde_json::Value>,
    pub tool_choice: Option<serde_json::Value>,
    pub max_output_tokens: Option<i64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub seed: Option<u32>,
}

/// Canonical response shape for POST /v1/responses (non-streaming).
///
/// Used for round-trip normalization (deserialize then re-serialize strips
/// unknown fields). With `serde(flatten)` + `Value` this is effectively a
/// pass-through — the schema documents the response shape for OpenAPI.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ResponseResource {
    #[serde(flatten)]
    pub fields: serde_json::Value,
}
