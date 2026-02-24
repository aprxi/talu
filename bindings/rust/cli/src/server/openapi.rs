//! Central OpenAPI spec generation via `utoipa`.
//!
//! All handler paths and schema types are registered in `ApiDoc`.
//! `build_openapi_json()` produces the JSON spec served at `/openapi.json`.

use utoipa::openapi::security::{ApiKey, ApiKeyValue, SecurityScheme};
use utoipa::{Modify, OpenApi};

use crate::server::{
    chat_generate_types, code, db, file, files, handlers, http, plugins, proxy, repo, responses,
    responses_types, search, sessions, settings, tags,
};

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Talu API",
        version = env!("TALU_VERSION"),
        description = "Local-first LLM inference and knowledge management API"
    ),
    tags(
        (name = "Chat::Generate", description = "Current chat generation endpoint"),
        (name = "Responses", description = "OpenResponses-compatible API surface"),
        (name = "Models", description = "Available model listing"),
        (name = "Chat", description = "Chat session state management"),
        (name = "Files", description = "Binary file upload and management"),
        (name = "File", description = "Stateless file inspect/transform"),
        (name = "Tags", description = "Tag CRUD and assignment"),
        (name = "Search", description = "Full-text and vector search"),
        (name = "DB::Tables", description = "Table-plane operations (currently documents table)"),
        (name = "DB::Vectors", description = "Vector-plane operations"),
        (name = "DB::KV", description = "Key/value-plane operations"),
        (name = "DB::Blobs", description = "Blob-plane operations"),
        (name = "DB::SQL", description = "SQL compute-plane operations"),
        (name = "DB::Ops", description = "Operational and maintenance endpoints"),
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
        // Models + chat generation
        handlers::handle_models,
        handlers::handle_chat_generate,
        responses::handle_create,
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
        // Sessions
        sessions::handle_list,
        sessions::handle_get,
        sessions::handle_delete,
        sessions::handle_patch,
        sessions::handle_batch,
        sessions::handle_fork,
        sessions::handle_get_tags,
        sessions::handle_add_tags,
        sessions::handle_set_tags,
        sessions::handle_remove_tags,
        // Documents
        db::table::handle_list,
        db::table::handle_create,
        db::table::handle_insert,
        db::table::handle_get,
        db::table::handle_update,
        db::table::handle_delete,
        db::table::handle_search,
        db::table::handle_get_tags,
        db::table::handle_add_tags,
        db::table::handle_remove_tags,
        // Files
        files::handle_upload,
        files::handle_list,
        files::handle_get,
        files::handle_get_content,
        files::handle_patch,
        files::handle_delete,
        files::handle_batch,
        // File (stateless)
        file::handle_inspect,
        file::handle_transform,
        // Search
        search::handle_search,
        // DB
        db::vector::handle_create_collection,
        db::vector::handle_list_collections,
        db::vector::handle_get_collection,
        db::vector::handle_delete_collection,
        db::vector::handle_append_points,
        db::vector::handle_upsert_points,
        db::vector::handle_delete_points,
        db::vector::handle_fetch_points,
        db::vector::handle_query_points,
        db::vector::handle_collection_stats,
        db::vector::handle_compact_collection,
        db::vector::handle_build_collection_indexes,
        db::vector::handle_collection_changes,
        db::kv::handle_list,
        db::kv::handle_put,
        db::kv::handle_get,
        db::kv::handle_delete,
        db::kv::handle_flush,
        db::kv::handle_compact,
        db::blob::handle_list,
        db::blob::handle_get,
        db::sql::handle_query,
        db::ops::handle_compact,
        db::ops::handle_simulate_crash,
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
        repo::handle_list_files,
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
        // Chat generate
        chat_generate_types::CreateChatGenerateBody,
        chat_generate_types::ChatGenerateResponseResource,
        responses_types::CreateResponseBody,
        responses_types::ResponseResource,
        responses_types::ResponsesErrorResponse,
        responses_types::ResponsesErrorBody,
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
        // Sessions (doc-only)
        sessions::SessionTag,
        sessions::SessionResponse,
        sessions::SessionListResponse,
        sessions::SessionPatchRequest,
        sessions::BatchRequest,
        sessions::ForkRequest,
        sessions::TagsRequest,
        // Documents
        db::table::DocumentResponse,
        db::table::DocumentSummaryResponse,
        db::table::DocumentListResponse,
        db::table::SearchResultResponse,
        db::table::DocumentSearchResponse,
        db::table::CreateDocumentRequest,
        db::table::UpdateDocumentRequest,
        db::table::DocumentSearchRequest,
        db::table::DocumentTagsRequest,
        db::blob::BlobListResponse,
        db::sql::SqlQueryRequest,
        db::ops::CompactRequest,
        db::ops::CompactResponse,
        db::ops::SimulateCrashRequest,
        db::ops::SimulateCrashResponse,
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
        // DB
        db::vector::CreateCollectionRequest,
        db::vector::CollectionResponse,
        db::vector::CollectionListResponse,
        db::vector::CollectionMetric,
        db::vector::NormalizationPolicy,
        db::vector::CollectionIdType,
        db::vector::VectorPointInput,
        db::vector::AppendPointsRequest,
        db::vector::AppendPointsResponse,
        db::vector::UpsertPointsRequest,
        db::vector::UpsertPointsResponse,
        db::vector::IdListRequest,
        db::vector::DeletePointsResponse,
        db::vector::FetchPointsRequest,
        db::vector::FetchedPoint,
        db::vector::FetchPointsResponse,
        db::vector::QueryPointsRequest,
        db::vector::QueryMatch,
        db::vector::QueryResultSet,
        db::vector::QueryPointsResponse,
        db::vector::CollectionStatsResponse,
        db::vector::CompactCollectionRequest,
        db::vector::CompactResponse,
        db::vector::BuildIndexesRequest,
        db::vector::BuildIndexesResponse,
        db::vector::ChangeEventResponse,
        db::vector::ChangesResponse,
        db::kv::KvEntryResponse,
        db::kv::KvListResponse,
        db::kv::KvPutResponse,
        db::kv::KvDeleteResponse,
        db::kv::KvCompactResponse,
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
        repo::FileEntry,
        repo::FileListResponse,
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
