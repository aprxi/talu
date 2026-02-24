use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Request body for POST /v1/responses.
///
/// This follows the OpenResponses request surface. Fields are optional in the
/// contract; validation of field combinations happens in the handler.
#[derive(Debug, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct CreateResponseBody {
    pub background: Option<bool>,
    pub frequency_penalty: Option<f64>,
    pub include: Option<serde_json::Value>,
    /// Input can be a string shorthand or an array of typed items.
    pub input: Option<serde_json::Value>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<i64>,
    pub max_tool_calls: Option<i64>,
    pub metadata: Option<serde_json::Value>,
    pub model: Option<String>,
    pub parallel_tool_calls: Option<bool>,
    pub presence_penalty: Option<f64>,
    pub previous_response_id: Option<String>,
    pub prompt_cache_key: Option<String>,
    pub reasoning: Option<serde_json::Value>,
    pub safety_identifier: Option<String>,
    pub service_tier: Option<String>,
    pub store: Option<bool>,
    pub stream: Option<bool>,
    pub stream_options: Option<serde_json::Value>,
    pub temperature: Option<f64>,
    pub text: Option<serde_json::Value>,
    pub tool_choice: Option<serde_json::Value>,
    pub tools: Option<serde_json::Value>,
    pub top_logprobs: Option<i64>,
    pub top_p: Option<f64>,
    pub truncation: Option<String>,
}

/// Response shape for POST /v1/responses.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ResponseResource {
    #[serde(flatten)]
    pub fields: serde_json::Value,
}

/// Error envelope for `/v1/responses` non-streaming failures.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ResponsesErrorResponse {
    pub error: ResponsesErrorBody,
}

/// Error details for `/v1/responses`.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ResponsesErrorBody {
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: String,
    pub message: String,
    pub param: Option<String>,
}
