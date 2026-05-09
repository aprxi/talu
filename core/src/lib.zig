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
//   - Any extern struct in capi/types.zig or capi/* public C API modules
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

// Force the linker to export these symbols by referencing them in a comptime block
comptime {
    // ABI Version (Global Handshake)
    _ = &talu_get_abi_version;

    // Tensor API

    // Error API
    _ = &capi.talu_last_error;
    _ = &capi.talu_last_error_code;
    _ = &capi.talu_take_last_error;
    _ = &capi.talu_error_buf_size;
    _ = &capi.talu_clear_error;

    // Log API
    _ = &capi.talu_set_log_level;
    _ = &capi.talu_set_log_format;
    _ = &capi.talu_set_log_filter;
    _ = &capi.talu_set_log_callback;
    _ = &capi.talu_get_log_level;
    _ = &capi.talu_get_log_format;

    // Memory API
    _ = &capi.talu_alloc_string;
    _ = &capi.talu_free_string;

    // Config API
    _ = &capi.talu_config_canonicalize;
    _ = &capi.talu_config_get_view;
    _ = &capi.talu_config_free;
    _ = &capi.talu_backend_get_capabilities;
    _ = &capi.talu_backend_create_from_canonical;
    _ = &capi.talu_backend_free;

    // Session utilities (model resolution, chat templates, EOS tokens)
    _ = &capi.talu_resolve_model_path;
    _ = &capi.talu_model_performance_hints;
    _ = &capi.talu_model_hf_config_json;
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
    _ = &capi.talu_buffer_release;
    _ = &capi.talu_buffer_get_data_ptr;

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

    // Convert API
    _ = &capi.talu_convert;
    _ = &capi.talu_convert_free_string;
    _ = &capi.talu_convert_schemes;

    // Repo API
    _ = &capi.talu_repo_is_cached;
    _ = &capi.talu_repo_get_cached_path;
    _ = &capi.talu_repo_get_hf_home;
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
    _ = &capi.talu_repo_list;
    _ = &capi.talu_repo_search;
    _ = &capi.talu_repo_string_list_count;
    _ = &capi.talu_repo_string_list_get;
    _ = &capi.talu_repo_string_list_free;

    // Validate API (high-level sampler)
    _ = &capi.talu_set_response_format;
    _ = &capi.talu_clear_response_format;

    // Validate Engine API (low-level isolated operations)
    _ = &capi.talu_validate_engine_create;
    _ = &capi.talu_validate_engine_destroy;
    _ = &capi.talu_validate_engine_reset;
    _ = &capi.talu_validate_engine_is_complete;
    _ = &capi.talu_validate_engine_get_position;
    _ = &capi.talu_validate_engine_get_valid_bytes;
    _ = &capi.talu_validate_engine_can_accept;
    _ = &capi.talu_validate_engine_advance;
    _ = &capi.talu_validate_engine_validate;

    // Token Mask API

    // Chat API (maps to talu/chat/)
    _ = &capi.talu_chat_create;
    _ = &capi.talu_chat_create_with_system;
    _ = &capi.talu_chat_create_with_session;
    _ = &capi.talu_chat_create_with_system_and_session;
    _ = &capi.talu_chat_free;
    _ = &capi.talu_chat_get_conversation;
    _ = &capi.talu_chat_set_ttl_ts;
    _ = &capi.talu_responses_clone_prefix;
    _ = &capi.talu_responses_truncate_after;
    _ = &capi.talu_chat_set_system;
    _ = &capi.talu_chat_get_system;
    _ = &capi.talu_chat_to_json;
    _ = &capi.talu_chat_set_messages;
    _ = &capi.talu_chat_load_completions_json;

    // Template API
    _ = &capi.talu_template_render;

    // Router API (backend-based)
    _ = &capi.talu_router_generate_with_backend;
    _ = &capi.talu_router_generate_streaming;
    _ = &capi.talu_router_result_free;
    _ = &capi.talu_router_close_all;
    _ = &capi.talu_router_embedding_dim;
    _ = &capi.talu_router_embed;
    _ = &capi.talu_router_embedding_free;

    // Scheduler API (continuous batching)
    _ = &capi.talu_scheduler_score_tokens_nll;

    // Batch API (responses-aware continuous batching)
    _ = &capi.talu_batch_create;
    _ = &capi.talu_batch_destroy;
    _ = &capi.talu_batch_submit;
    _ = &capi.talu_batch_cancel;
    _ = &capi.talu_batch_step;
    _ = &capi.talu_batch_has_active;
    _ = &capi.talu_batch_active_count;
    _ = &capi.talu_batch_run_loop;
    _ = &capi.talu_batch_run_loop_no_text;
    _ = &capi.talu_batch_take_result;
    _ = &capi.talu_batch_result_free;

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

    // Training API (LoRA)
    _ = &capi.talu_train_create;
    _ = &capi.talu_train_destroy;
    _ = &capi.talu_train_load_model;
    _ = &capi.talu_train_configure;
    _ = &capi.talu_train_load_data;
    _ = &capi.talu_train_run;
    _ = &capi.talu_train_save_checkpoint;
    _ = &capi.talu_train_get_info;

    // Training API (full / from-scratch)
    _ = &capi.talu_train_full_create;
    _ = &capi.talu_train_full_destroy;
    _ = &capi.talu_train_full_init_model;
    _ = &capi.talu_train_full_configure;
    _ = &capi.talu_train_full_set_data;
    _ = &capi.talu_train_full_load_data;
    _ = &capi.talu_train_full_step;
    _ = &capi.talu_train_full_run;
    _ = &capi.talu_train_full_get_info;
    _ = &capi.talu_train_full_copy_weights_f32;
    _ = &capi.talu_train_full_load_weights_f32;
    _ = &capi.talu_train_full_copy_optimizer_state_f32;
    _ = &capi.talu_train_full_load_optimizer_state_f32;
}
