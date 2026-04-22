//! Central OpenAPI spec generation for the kept inference HTTP surface.

use utoipa::openapi::security::{ApiKey, ApiKeyValue, SecurityScheme};
use utoipa::{Modify, OpenApi};

use crate::server::{
    agent::{exec as agent_exec, fs as agent_fs, process as agent_process, shell as agent_shell},
    collab, completions, completions_types, handlers, http, repo, responses, responses_types,
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
        (name = "Collab::Resources", description = "Canonical resource-scoped collaboration API"),
        (name = "Agent::FS", description = "Workdir filesystem runtime APIs"),
        (name = "Agent::Exec", description = "One-shot shell command execution (SSE)"),
        (name = "Agent::Shell", description = "Interactive PTY shell lifecycle and WebSocket attach"),
        (name = "Agent::Process", description = "Long-lived process sessions with stdin send and SSE/WebSocket streaming"),
    ),
    security(
        ("gateway_secret" = []),
        ("tenant_id" = []),
    ),
    paths(
        http::handle_health,
        handlers::handle_models,
        repo::handle_list,
        repo::handle_search,
        repo::handle_fetch,
        repo::handle_delete,
        repo::handle_list_files,
        responses::handle_create,
        completions::handle_create,
        collab::handle_open_session,
        collab::handle_get_resource,
        collab::handle_get_snapshot,
        collab::handle_submit_op,
        collab::handle_get_history,
        collab::handle_stream_events,
        collab::handle_ws,
        collab::handle_put_presence,
        collab::handle_get_presence,
        agent_fs::handle_read,
        agent_fs::handle_write,
        agent_fs::handle_edit,
        agent_fs::handle_stat,
        agent_fs::handle_list,
        agent_fs::handle_remove,
        agent_fs::handle_mkdir,
        agent_fs::handle_rename,
        agent_exec::handle_exec,
        agent_shell::handle_create,
        agent_shell::handle_list,
        agent_shell::handle_get,
        agent_shell::handle_delete,
        agent_shell::handle_ws,
        agent_process::handle_spawn,
        agent_process::handle_list,
        agent_process::handle_send,
        agent_process::handle_stream,
        agent_process::handle_ws,
        agent_process::handle_delete,
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
        collab::ParticipantKind,
        collab::OpenSessionRequest,
        collab::OpenSessionResponse,
        collab::ResourceSummaryResponse,
        collab::SnapshotResponse,
        collab::SubmitOpRequest,
        collab::SubmitOpResponse,
        collab::OpHistoryEntry,
        collab::HistoryResponse,
        collab::PresencePutRequest,
        collab::PresencePutResponse,
        collab::PresenceGetResponse,
        collab::CollabWatchEvent,
        agent_fs::FsReadRequest,
        agent_fs::FsReadResponse,
        agent_fs::FsWriteRequest,
        agent_fs::FsWriteResponse,
        agent_fs::FsEditRequest,
        agent_fs::FsEditResponse,
        agent_fs::FsStatRequest,
        agent_fs::FsStatResponse,
        agent_fs::FsListRequest,
        agent_fs::FsListResponse,
        agent_fs::FsListEntry,
        agent_fs::FsRemoveRequest,
        agent_fs::FsRemoveResponse,
        agent_fs::FsMkdirRequest,
        agent_fs::FsMkdirResponse,
        agent_fs::FsRenameRequest,
        agent_fs::FsRenameResponse,
        agent_exec::ExecRequest,
        agent_exec::ExecEvent,
        agent_shell::ShellCreateRequest,
        agent_shell::ShellSessionResponse,
        agent_shell::ShellListResponse,
        agent_shell::ShellDeleteResponse,
        agent_process::ProcessSpawnRequest,
        agent_process::ProcessSessionResponse,
        agent_process::ProcessListResponse,
        agent_process::ProcessSendRequest,
        agent_process::ProcessSendResponse,
        agent_process::ProcessDeleteResponse,
        agent_process::ProcessEvent,
        repo::CachedModelResponse,
        repo::RepoListResponse,
        repo::SearchResultResponse,
        repo::RepoSearchResponse,
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
