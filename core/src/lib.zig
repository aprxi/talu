// Library entry point for C API exports
//
// This file is the root for the shared library build.
// It exports all C API functions for FFI usage.

pub const capi = @import("capi/root.zig");

// =============================================================================
// ABI Version - Global Handshake for FFI Safety
// =============================================================================
//
// This version number is checked by Python/Rust bindings when the library is
// loaded. If there's a mismatch, the import fails immediately with a clear
// error, preventing memory corruption from mismatched struct layouts.
//
// IMPORTANT: Bump this version whenever ANY of these change:
//   - CStorageRecord, CStorageEvent, CSessionRecord (capi/storage.zig)
//   - CGenerateOptions, CGenerateCallbacks (capi/generate.zig)
//   - Any other extern struct in the C-API
//
// This is the ONLY place you need to update - Python checks automatically.

/// ABI version for FFI compatibility checking.
/// Bump this when extern struct layouts change.
pub const ABI_VERSION: i32 = 1;

/// Get the ABI version of this library.
/// Called by bindings during library load to verify compatibility.
pub export fn talu_get_abi_version() callconv(.c) i32 {
    return ABI_VERSION;
}

// Additional exports needed by integration tests
pub const core = @import("root.zig");
pub const tokenizer = @import("tokenizer/root.zig");
pub const template = @import("template/root.zig");
pub const nn = struct {
    pub const sampling = inference.sampling;
};
pub const io = @import("io/root.zig");
pub const models = struct {
    pub const dispatcher = @import("models/root.zig");
};
pub const inference = @import("inference/root.zig");
pub const responses = @import("responses/root.zig");
pub const router = @import("router/root.zig");
pub const generation_config = @import("inference/config/generation.zig");
pub const converter = @import("converter/root.zig");
pub const compute = @import("compute/root.zig");
pub const xray = @import("xray/root.zig");
pub const validate = @import("validate/root.zig");
pub const db = @import("db/root.zig");
pub const policy = @import("policy/root.zig");
pub const dump = @import("xray/dump/root.zig");
pub const agent = @import("agent/root.zig");

// Force the linker to export these symbols by referencing them in a comptime block
comptime {
    // ABI Version (Global Handshake)
    _ = &talu_get_abi_version;

    // Tensor API
    _ = &capi.talu_hello;
    _ = &capi.talu_tensor_create;
    _ = &capi.talu_tensor_zeros;
    _ = &capi.talu_tensor_test_embeddings;
    _ = &capi.talu_tensor_free;
    _ = &capi.talu_tensor_data_ptr;
    _ = &capi.talu_tensor_ndim;
    _ = &capi.talu_tensor_shape;
    _ = &capi.talu_tensor_strides;
    _ = &capi.talu_tensor_dtype;
    _ = &capi.talu_tensor_typestr;
    _ = &capi.talu_tensor_device_type;
    _ = &capi.talu_tensor_device_id;
    _ = &capi.talu_tensor_is_cpu;
    _ = &capi.talu_tensor_numel;
    _ = &capi.talu_tensor_element_size;
    _ = &capi.talu_tensor_to_dlpack;
    _ = &capi.talu_dlpack_capsule_name;
    _ = &capi.talu_dlpack_used_capsule_name;

    // Error API
    _ = &capi.talu_last_error;
    _ = &capi.talu_last_error_code;
    _ = &capi.talu_take_last_error;
    _ = &capi.talu_error_buf_size;
    _ = &capi.talu_clear_error;

    // Log API
    _ = &capi.talu_set_log_level;
    _ = &capi.talu_set_log_format;
    _ = &capi.talu_get_log_level;
    _ = &capi.talu_get_log_format;

    // Memory API
    _ = &capi.talu_alloc_string;
    _ = &capi.talu_free_string;

    // Config API
    _ = &capi.talu_config_validate;
    _ = &capi.talu_config_canonicalize;
    _ = &capi.talu_config_get_view;
    _ = &capi.talu_config_free;
    _ = &capi.talu_backend_get_capabilities;
    _ = &capi.talu_backend_create_from_canonical;
    _ = &capi.talu_backend_free;

    // Session utilities (model resolution, chat templates, EOS tokens)
    _ = &capi.talu_resolve_model_path;
    _ = &capi.talu_get_eos_tokens;
    _ = &capi.talu_get_generation_config;
    _ = &capi.talu_apply_chat_template;

    // Tokenizer utilities
    _ = &capi.talu_tokens_free;
    _ = &capi.talu_tokens_concat;
    _ = &capi.talu_decode_result_free;
    _ = &capi.talu_text_free;

    // DLPack API
    _ = &capi.talu_buffer_to_dlpack;
    _ = &capi.talu_batch_to_dlpack;
    _ = &capi.talu_batch_mask_to_dlpack;

    // SharedBuffer API (refcounted buffers for zero-copy interop)
    _ = &capi.talu_buffer_create_from_owned;
    _ = &capi.talu_buffer_create_from_copy;
    _ = &capi.talu_buffer_retain;
    _ = &capi.talu_buffer_release;
    _ = &capi.talu_buffer_get_data_ptr;
    _ = &capi.talu_buffer_get_capacity;
    _ = &capi.talu_buffer_get_refcount;

    // Tokenizer-only API (lightweight, no model weights)
    _ = &capi.talu_tokenizer_create;
    _ = &capi.talu_tokenizer_free;
    _ = &capi.talu_tokenizer_encode;
    _ = &capi.talu_tokenizer_decode;
    _ = &capi.talu_tokenizer_get_eos_tokens;
    _ = &capi.talu_tokenizer_get_model_dir;
    _ = &capi.talu_tokenizer_get_vocab_size;
    _ = &capi.talu_tokenizer_get_vocab;
    _ = &capi.talu_vocab_result_free;
    _ = &capi.talu_tokenizer_get_special_tokens;
    _ = &capi.talu_tokenizer_id_to_token;
    _ = &capi.talu_tokenizer_token_to_id;
    _ = &capi.talu_tokenizer_tokenize;
    _ = &capi.talu_tokenize_result_free;
    _ = &capi.talu_tokenizer_tokenize_bytes;
    _ = &capi.talu_tokenize_bytes_result_free;
    _ = &capi.talu_encode_result_free;
    _ = &capi.talu_tokenizer_encode_batch;
    _ = &capi.talu_batch_encode_result_free;
    _ = &capi.talu_batch_to_padded_tensor;
    _ = &capi.talu_padded_tensor_result_free;

    // Model description API
    _ = &capi.talu_describe;
    _ = &capi.talu_model_info_free;

    // Architecture API
    _ = &capi.talu_arch_init;
    _ = &capi.talu_arch_deinit;
    _ = &capi.talu_arch_register;
    _ = &capi.talu_arch_exists;
    _ = &capi.talu_arch_count;
    _ = &capi.talu_arch_list;
    _ = &capi.talu_arch_free_string;
    _ = &capi.talu_arch_detect;

    // Convert API
    _ = &capi.talu_convert;
    _ = &capi.talu_convert_free_string;
    _ = &capi.talu_convert_schemes;

    // Image API
    _ = &capi.talu_image_decode;
    _ = &capi.talu_image_convert;
    _ = &capi.talu_image_to_model_input;
    _ = &capi.talu_image_encode;
    _ = &capi.talu_image_free;
    _ = &capi.talu_model_buffer_free;
    _ = &capi.talu_image_encode_free;
    _ = &capi.talu_file_inspect;
    _ = &capi.talu_file_info_free;
    _ = &capi.talu_file_transform;
    _ = &capi.talu_file_bytes_free;

    // Repo API
    _ = &capi.talu_repo_is_cached;
    _ = &capi.talu_repo_get_cached_path;
    _ = &capi.talu_repo_get_hf_home;
    _ = &capi.talu_repo_get_cache_dir;
    _ = &capi.talu_repo_list_models;
    _ = &capi.talu_repo_list_count;
    _ = &capi.talu_repo_list_get_id;
    _ = &capi.talu_repo_list_get_path;
    _ = &capi.talu_repo_list_free;
    _ = &capi.talu_repo_delete;
    _ = &capi.talu_repo_size;
    _ = &capi.talu_repo_total_size;
    _ = &capi.talu_repo_is_model_id;
    _ = &capi.talu_repo_fetch;
    _ = &capi.talu_repo_exists;
    _ = &capi.talu_repo_list;
    _ = &capi.talu_repo_search;
    _ = &capi.talu_repo_string_list_count;
    _ = &capi.talu_repo_string_list_get;
    _ = &capi.talu_repo_string_list_free;

    // Validate API (high-level sampler)
    _ = &capi.talu_validate_create;
    _ = &capi.talu_validate_free;
    _ = &capi.talu_validate_apply;
    _ = &capi.talu_validate_accept;
    _ = &capi.talu_validate_is_complete;
    _ = &capi.talu_validate_reset;
    _ = &capi.talu_set_response_format;
    _ = &capi.talu_clear_response_format;

    // Validate Engine API (low-level isolated operations)
    _ = &capi.talu_validate_engine_create;
    _ = &capi.talu_validate_engine_destroy;
    _ = &capi.talu_validate_engine_reset;
    _ = &capi.talu_validate_engine_is_complete;
    _ = &capi.talu_validate_engine_get_position;
    _ = &capi.talu_validate_engine_get_state_count;
    _ = &capi.talu_validate_engine_get_valid_bytes;
    _ = &capi.talu_validate_engine_count_valid_bytes;
    _ = &capi.talu_validate_engine_can_accept;
    _ = &capi.talu_validate_engine_advance_byte;
    _ = &capi.talu_validate_engine_advance;
    _ = &capi.talu_validate_engine_validate;
    _ = &capi.talu_validate_engine_get_valid_tokens;
    _ = &capi.talu_validate_engine_get_valid_tokens_with_tokenizer;
    _ = &capi.talu_validate_engine_get_deterministic_continuation;

    // Token Mask API
    _ = &capi.talu_token_mask_create;
    _ = &capi.talu_token_mask_destroy;
    _ = &capi.talu_token_mask_clear;
    _ = &capi.talu_token_mask_set_all;
    _ = &capi.talu_token_mask_is_valid;
    _ = &capi.talu_token_mask_set;
    _ = &capi.talu_token_mask_get_size;
    _ = &capi.talu_token_mask_get_bits;
    _ = &capi.talu_token_mask_get_word_count;
    _ = &capi.talu_token_mask_count_valid;
    _ = &capi.talu_token_mask_apply;

    // Chat API (maps to talu/chat/)
    _ = &capi.talu_chat_create;
    _ = &capi.talu_chat_create_with_system;
    _ = &capi.talu_chat_create_with_session;
    _ = &capi.talu_chat_create_with_system_and_session;
    _ = &capi.talu_chat_free;
    _ = &capi.talu_chat_get_conversation;
    _ = &capi.talu_chat_get_session_id;
    _ = &capi.talu_chat_set_ttl_ts;
    _ = &capi.talu_responses_clone_prefix;
    _ = &capi.talu_responses_truncate_after;
    _ = &capi.talu_responses_load_storage_records;
    _ = &capi.talu_responses_begin_fork;
    _ = &capi.talu_responses_end_fork;
    _ = &capi.talu_responses_set_item_parent;
    _ = &capi.talu_responses_set_item_validation_flags;
    _ = &capi.talu_chat_get_temperature;
    _ = &capi.talu_chat_set_temperature;
    _ = &capi.talu_chat_get_max_tokens;
    _ = &capi.talu_chat_set_max_tokens;
    _ = &capi.talu_chat_get_top_k;
    _ = &capi.talu_chat_set_top_k;
    _ = &capi.talu_chat_get_top_p;
    _ = &capi.talu_chat_set_top_p;
    _ = &capi.talu_chat_get_min_p;
    _ = &capi.talu_chat_set_min_p;
    _ = &capi.talu_chat_get_repetition_penalty;
    _ = &capi.talu_chat_set_repetition_penalty;
    _ = &capi.talu_chat_set_system;
    _ = &capi.talu_chat_get_system;
    _ = &capi.talu_chat_clear;
    _ = &capi.talu_chat_reset;
    _ = &capi.talu_chat_len;
    _ = &capi.talu_chat_to_json;
    _ = &capi.talu_chat_set_messages;
    _ = &capi.talu_chat_get_messages;

    // Template API
    _ = &capi.talu_template_render;

    // Router API (backend-based)
    _ = &capi.talu_router_generate_with_backend;
    _ = &capi.talu_router_result_free;
    _ = &capi.talu_router_close_all;
    _ = &capi.talu_router_embedding_dim;
    _ = &capi.talu_router_embed;
    _ = &capi.talu_router_embedding_free;

    // Iterator API (pull-based streaming)
    _ = &capi.talu_router_create_iterator;
    _ = &capi.talu_router_iterator_next;
    _ = &capi.talu_router_iterator_has_error;
    _ = &capi.talu_router_iterator_error_code;
    _ = &capi.talu_router_iterator_error_msg;
    _ = &capi.talu_router_iterator_cancel;
    _ = &capi.talu_router_iterator_free;
    _ = &capi.talu_router_iterator_prompt_tokens;
    _ = &capi.talu_router_iterator_completion_tokens;
    _ = &capi.talu_router_iterator_prefill_ns;
    _ = &capi.talu_router_iterator_generation_ns;

    // X-Ray API (tensor inspection during inference)
    _ = &capi.talu_xray_capture_create;
    _ = &capi.talu_xray_capture_create_all;
    _ = &capi.talu_xray_capture_enable;
    _ = &capi.talu_xray_capture_disable;
    _ = &capi.talu_xray_capture_is_enabled;
    _ = &capi.talu_xray_capture_clear;
    _ = &capi.talu_xray_capture_count;
    _ = &capi.talu_xray_capture_overflow;
    _ = &capi.talu_xray_capture_destroy;
    _ = &capi.talu_xray_get;
    _ = &capi.talu_xray_find_anomaly;
    _ = &capi.talu_xray_count_matching;
    _ = &capi.talu_xray_get_samples;
    _ = &capi.talu_xray_point_name;

    // Provider API (remote provider registry)
    _ = &capi.talu_provider_count;
    _ = &capi.talu_provider_get;
    _ = &capi.talu_provider_get_by_name;
    _ = &capi.talu_provider_parse;
    _ = &capi.talu_provider_has_prefix;

    // Policy API (tool call firewall)
    _ = &capi.talu_policy_create;
    _ = &capi.talu_policy_free;
    _ = &capi.talu_policy_evaluate;
    _ = &capi.talu_policy_get_mode;
    _ = &capi.talu_chat_set_policy;

    // Plugins API (UI plugin discovery)
    _ = &capi.talu_plugins_scan;
    _ = &capi.talu_plugins_list_count;
    _ = &capi.talu_plugins_list_get;
    _ = &capi.talu_plugins_list_free;

    // Documents API (universal document storage)
    _ = &capi.talu_documents_create;
    _ = &capi.talu_documents_get;
    _ = &capi.talu_documents_update;
    _ = &capi.talu_documents_delete;
    _ = &capi.talu_documents_delete_batch;
    _ = &capi.talu_documents_set_marker_batch;
    _ = &capi.talu_documents_list;
    _ = &capi.talu_documents_free_list;
    _ = &capi.talu_documents_search;
    _ = &capi.talu_documents_free_search_results;
    _ = &capi.talu_documents_search_batch;
    _ = &capi.talu_documents_free_json;
    _ = &capi.talu_documents_get_changes;
    _ = &capi.talu_documents_free_changes;
    _ = &capi.talu_documents_set_ttl;
    _ = &capi.talu_documents_count_expired;
    _ = &capi.talu_documents_create_delta;
    _ = &capi.talu_documents_get_delta_chain;
    _ = &capi.talu_documents_free_delta_chain;
    _ = &capi.talu_documents_is_delta;
    _ = &capi.talu_documents_get_base_id;
    _ = &capi.talu_documents_get_compaction_stats;
    _ = &capi.talu_documents_purge_expired;
    _ = &capi.talu_documents_get_garbage_candidates;
    _ = &capi.talu_documents_add_tag;
    _ = &capi.talu_documents_remove_tag;
    _ = &capi.talu_documents_get_tags;
    _ = &capi.talu_documents_get_by_tag;
    _ = &capi.talu_documents_free_string_list;
    _ = &capi.talu_documents_get_blob_ref;

    // Blob API (raw content-addressable blob storage)
    _ = &capi.talu_blobs_put;
    _ = &capi.talu_blobs_open_stream;
    _ = &capi.talu_blobs_stream_read;
    _ = &capi.talu_blobs_stream_total_size;
    _ = &capi.talu_blobs_stream_close;

    // Agent API (tool registry + agent loop)
    _ = &capi.talu_agent_registry_create;
    _ = &capi.talu_agent_registry_free;
    _ = &capi.talu_agent_registry_add;
    _ = &capi.talu_agent_registry_count;
    _ = &capi.talu_agent_run;

    // Stateful Agent API
    _ = &capi.talu_agent_create;
    _ = &capi.talu_agent_free;
    _ = &capi.talu_agent_set_system;
    _ = &capi.talu_agent_register_tool;
    _ = &capi.talu_agent_set_bus;
    _ = &capi.talu_agent_prompt;
    _ = &capi.talu_agent_continue;
    _ = &capi.talu_agent_heartbeat;
    _ = &capi.talu_agent_abort;
    _ = &capi.talu_agent_get_chat;
    _ = &capi.talu_agent_get_id;

    // Agent goal API
    _ = &capi.talu_agent_add_goal;
    _ = &capi.talu_agent_remove_goal;
    _ = &capi.talu_agent_clear_goals;
    _ = &capi.talu_agent_goal_count;

    // Agent context injection API
    _ = &capi.talu_agent_set_context_inject;

    // MessageBus API
    _ = &capi.talu_agent_bus_create;
    _ = &capi.talu_agent_bus_free;
    _ = &capi.talu_agent_bus_register;
    _ = &capi.talu_agent_bus_unregister;
    _ = &capi.talu_agent_bus_add_peer;
    _ = &capi.talu_agent_bus_remove_peer;
    _ = &capi.talu_agent_bus_send;
    _ = &capi.talu_agent_bus_deliver;
    _ = &capi.talu_agent_bus_broadcast;
    _ = &capi.talu_agent_bus_pending;
}
