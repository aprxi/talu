//! Central OpenAPI spec generation for the kept inference HTTP surface.

use utoipa::openapi::security::{ApiKey, ApiKeyValue, SecurityScheme};
use utoipa::{Modify, OpenApi};

use crate::server::{completions, completions_types, handlers, http, responses, responses_types};

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
        (name = "Chat", description = "OpenAI-compatible chat completions"),
    ),
    security(
        ("gateway_secret" = []),
        ("tenant_id" = []),
    ),
    paths(
        handlers::handle_models,
        responses::handle_create,
        completions::handle_create,
    ),
    components(schemas(
        http::ErrorResponse,
        http::ErrorBody,
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
    serde_json::to_vec_pretty(&doc).unwrap()
}
