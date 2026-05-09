//! Central OpenAPI spec generation for the kept inference HTTP surface.

use utoipa::openapi::security::{ApiKey, ApiKeyValue, SecurityScheme};
use utoipa::{Modify, OpenApi};

use crate::server::{
    completions, completions_types, handlers, http, model, repo, responses, responses_types,
    tokenizer,
};

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Talu API",
        version = env!("TALU_VERSION"),
        description = "Inference API for local models"
    ),
    tags(
        (name = "Responses", description = "OpenResponses-compatible API surface"),
        (name = "Models", description = "Available model listing"),
        (name = "Repository", description = "Model cache management and hub downloads"),
        (name = "Chat", description = "OpenAI-compatible chat completions"),
        (name = "Tokenizer", description = "Tokenizer management and encode/decode API"),
    ),
    security(
        ("gateway_secret" = []),
        ("tenant_id" = []),
    ),
    paths(
        http::handle_health,
        handlers::handle_models,
        model::handle_config,
        repo::handle_list,
        repo::handle_search,
        repo::handle_stats,
        repo::handle_downloads,
        repo::handle_download_enqueue,
        repo::handle_download_clear_finished,
        repo::handle_download_cancel_all,
        repo::handle_fetch,
        repo::handle_delete,
        repo::handle_list_files,
        responses::handle_create,
        completions::handle_create,
        tokenizer::handle_create_instance,
        tokenizer::handle_get_instance,
        tokenizer::handle_delete_instance,
        tokenizer::handle_encode,
        tokenizer::handle_encode_batch,
        tokenizer::handle_decode,
        tokenizer::handle_decode_batch,
        tokenizer::handle_vocab,
        tokenizer::handle_vocab_size,
        tokenizer::handle_token_to_id,
        tokenizer::handle_id_to_token,
        tokenizer::handle_enable_truncation,
        tokenizer::handle_disable_truncation,
        tokenizer::handle_enable_padding,
        tokenizer::handle_disable_padding,
        tokenizer::handle_add_tokens,
        tokenizer::handle_add_special_tokens,
        tokenizer::handle_train,
        tokenizer::handle_train_from_iterator,
        tokenizer::handle_save,
        tokenizer::handle_compare,
        tokenizer::handle_capabilities,
    ),
    components(schemas(
        http::ErrorResponse,
        http::ErrorBody,
        http::HealthResponse,
        model::ModelConfigRequest,
        model::ModelConfigResponse,
        responses_types::CreateResponseBody,
        responses_types::ResponseResource,
        responses_types::ResponsesErrorResponse,
        responses_types::ResponsesErrorBody,
        completions_types::CreateChatCompletionBody,
        completions_types::ChatMessage,
        completions_types::ChatCompletion,
        completions_types::Choice,
        completions_types::ResponseMessage,
        completions_types::CompletionUsage,
        completions_types::ChatCompletionChunk,
        completions_types::ChunkChoice,
        completions_types::Delta,
        repo::CachedModelResponse,
        repo::RepoListResponse,
        repo::SearchResultResponse,
        repo::RepoSearchResponse,
        repo::RepoStatsResponse,
        repo::RepoDownloadResponse,
        repo::RepoDownloadsResponse,
        repo::RepoFetchRequest,
        repo::RepoFetchResponse,
        repo::RepoDeleteResponse,
        repo::FileEntry,
        repo::FileListResponse,
    ))
)]
pub struct ApiDoc;

struct SecurityAddon;

impl Modify for SecurityAddon {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        let components = openapi.components.as_mut().unwrap();
        components.add_security_scheme(
            "gateway_secret",
            SecurityScheme::ApiKey(ApiKey::Header(ApiKeyValue::new("x-talu-gateway-secret"))),
        );
        components.add_security_scheme(
            "tenant_id",
            SecurityScheme::ApiKey(ApiKey::Header(ApiKeyValue::new("x-talu-tenant-id"))),
        );
    }
}

pub fn build_openapi_json() -> Vec<u8> {
    let mut doc = ApiDoc::openapi();
    SecurityAddon.modify(&mut doc);
    let mut value = serde_json::to_value(&doc).unwrap_or_else(|_| serde_json::json!({}));

    // Keep generated spec aligned with the runtime contract:
    // chat-completions request rejects unknown top-level fields.
    if let Some(schema) = value.pointer_mut("/components/schemas/CreateChatCompletionBody") {
        if !schema.is_object() {
            return serde_json::to_vec_pretty(&value).unwrap_or_else(|_| b"{}".to_vec());
        }
        if schema.get("additionalProperties").is_none() {
            schema["additionalProperties"] = serde_json::Value::Bool(false);
        }
    }

    serde_json::to_vec_pretty(&value).unwrap_or_else(|_| b"{}".to_vec())
}
