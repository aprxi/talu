//! Central OpenAPI spec generation via `utoipa`.
//!
//! All handler paths and schema types are registered in `ApiDoc`.
//! `build_openapi_json()` produces the JSON spec served at `/openapi.json`.

use utoipa::openapi::security::{ApiKey, ApiKeyValue, SecurityScheme};
use utoipa::{Modify, OpenApi};

use crate::server::{
    code, conversations, documents, file, files, handlers, http, plugins, proxy, repo,
    responses_types, search, settings, tags,
};

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Talu API",
        version = env!("TALU_VERSION"),
        description = "Local-first LLM inference and knowledge management API"
    ),
    tags(
        (name = "Responses", description = "LLM inference (chat completions)"),
        (name = "Models", description = "Available model listing"),
        (name = "Conversations", description = "Chat session management"),
        (name = "Documents", description = "Structured document storage"),
        (name = "Files", description = "Binary file upload and management"),
        (name = "File", description = "Stateless file inspect/transform"),
        (name = "Tags", description = "Tag CRUD and assignment"),
        (name = "Search", description = "Full-text and vector search"),
        (name = "Settings", description = "Server and model configuration"),
        (name = "Plugins", description = "Plugin discovery"),
        (name = "Proxy", description = "Plugin outbound HTTP proxy"),
        (name = "Repository", description = "Model cache management (list, search, download, delete)"),
        (name = "Code", description = "Tree-sitter code analysis and incremental parsing"),
    ),
    security(
        ("gateway_secret" = []),
        ("tenant_id" = []),
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
        // Repository management
        repo::handle_list,
        repo::handle_search,
        repo::handle_fetch,
        repo::handle_delete,
        // Pin management
        repo::handle_list_pins,
        repo::handle_pin,
        repo::handle_unpin,
        repo::handle_sync_pins,
        // Code (tree-sitter)
        code::handle_highlight,
        code::handle_parse,
        code::handle_query,
        code::handle_graph,
        code::handle_languages,
        code::handle_session_create,
        code::handle_session_update,
        code::handle_session_highlight,
        code::handle_session_delete,
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
        settings::SettingsPatchRequest,
        settings::ModelEntry,
        settings::ModelDefaults,
        settings::OverridesJson,
        // Conversations (doc-only)
        conversations::ConversationTag,
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
        // Repository
        repo::CachedModelResponse,
        repo::RepoListResponse,
        repo::SearchResultResponse,
        repo::RepoSearchResponse,
        repo::RepoFetchRequest,
        repo::RepoFetchResponse,
        repo::RepoDeleteResponse,
        repo::PinnedModelResponse,
        repo::PinListResponse,
        repo::PinRequest,
        repo::PinActionResponse,
        repo::SyncPinsRequest,
        repo::SyncPinsResponse,
        // Code (tree-sitter)
        code::HighlightRequest,
        code::ParseRequest,
        code::QueryRequest,
        code::GraphRequest,
        code::LanguagesResponse,
        code::SessionCreateRequest,
        code::SessionUpdateRequest,
        code::SessionTextEdit,
        code::SessionHighlightRequest,
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
