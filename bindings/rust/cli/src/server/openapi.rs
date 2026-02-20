//! Central OpenAPI spec generation via `utoipa`.
//!
//! All handler paths and schema types are registered in `ApiDoc`.
//! `build_openapi_json()` produces the JSON spec served at `/openapi.json`.

use utoipa::OpenApi;

use crate::server::{
    conversations, documents, file, files, handlers, http, plugins, proxy, responses_types, search,
    settings, tags,
};

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Talu API",
        version = env!("TALU_VERSION"),
        description = "Local-first LLM inference and knowledge management API"
    ),
    paths(
        // Models + Responses
        handlers::handle_models,
        handlers::handle_responses,
        // Tags
        tags::handle_list,
        tags::handle_create,
        tags::handle_get,
        tags::handle_patch,
        tags::handle_delete,
        // Settings
        settings::handle_get,
        settings::handle_patch,
        settings::handle_reset_model,
        // Conversations
        conversations::handle_list,
        conversations::handle_get,
        conversations::handle_delete,
        conversations::handle_patch,
        conversations::handle_batch,
        conversations::handle_fork,
        conversations::handle_get_tags,
        conversations::handle_add_tags,
        conversations::handle_set_tags,
        conversations::handle_remove_tags,
        // Documents
        documents::handle_list,
        documents::handle_create,
        documents::handle_get,
        documents::handle_update,
        documents::handle_delete,
        documents::handle_search,
        documents::handle_get_tags,
        documents::handle_add_tags,
        documents::handle_remove_tags,
        // Files
        files::handle_upload,
        files::handle_list,
        files::handle_get,
        files::handle_get_content,
        files::handle_get_blob,
        files::handle_patch,
        files::handle_delete,
        files::handle_batch,
        // File (stateless)
        file::handle_inspect,
        file::handle_transform,
        // Search
        search::handle_search,
        // Plugins
        plugins::handle_list,
        // Proxy
        proxy::handle_proxy,
    ),
    components(schemas(
        // Shared
        http::ErrorResponse,
        http::ErrorBody,
        // Responses
        responses_types::CreateResponseBody,
        responses_types::ResponseResource,
        // Tags
        tags::TagResponse,
        tags::TagUsage,
        tags::TagResponseWithUsage,
        tags::TagListResponse,
        tags::CreateTagRequest,
        tags::UpdateTagRequest,
        // Settings
        settings::SettingsResponse,
        settings::ModelEntry,
        settings::ModelDefaults,
        settings::OverridesJson,
        // Conversations (doc-only)
        conversations::ConversationResponse,
        conversations::ConversationListResponse,
        conversations::ConversationPatchRequest,
        conversations::BatchRequest,
        conversations::ForkRequest,
        conversations::TagsRequest,
        // Documents
        documents::DocumentResponse,
        documents::DocumentSummaryResponse,
        documents::DocumentListResponse,
        documents::SearchResultResponse,
        documents::DocumentSearchResponse,
        documents::CreateDocumentRequest,
        documents::UpdateDocumentRequest,
        documents::DocumentSearchRequest,
        documents::DocumentTagsRequest,
        // Files
        files::FileObjectResponse,
        files::FileImageMetadata,
        files::FileListResponse,
        files::FileDeleteResponse,
        files::FileBatchRequest,
        files::FilePatchRequest,
        // File (stateless)
        file::FileInspectResponse,
        file::ImageMetadata,
        // Search
        search::SearchRequest,
        search::SearchResponse,
        search::SearchFilters,
        search::VectorSearch,
        // Plugins
        plugins::PluginEntry,
        plugins::PluginListResponse,
        // Proxy
        proxy::ProxyRequest,
        proxy::ProxyResponse,
    ))
)]
pub struct ApiDoc;

pub fn build_openapi_json() -> Vec<u8> {
    serde_json::to_vec_pretty(&ApiDoc::openapi()).unwrap()
}
