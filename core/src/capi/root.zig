// C API module - exports all C-callable functions.
//
// This module aggregates all C API exports for the library.
// Usage: const capi = @import("capi/root.zig");
//
// Primary APIs for talu/chat/:
// - responses.zig: Chat/Session management (talu_chat_*), Item-based conversation (talu_responses_*)
// - router.zig: Generation routing (talu_router_*), backend management (talu_backend_*), config (talu_config_*)
//
// Utility APIs:
// - session.zig: Model resolution, chat templates, EOS tokens
// - tokenizer.zig: Tokenization (talu_tokenizer_*)

const session = @import("session.zig");
const template = @import("template.zig");
const tokenizer = @import("tokenizer.zig");
const dlpack = @import("dlpack.zig");
const converter = @import("converter.zig");
const repo = @import("repository.zig");
const err = @import("error.zig");
const responses = @import("responses.zig");
const xray = @import("xray.zig");
const validate = @import("validate.zig");
const buffer = @import("buffer.zig");

const log_api = @import("log.zig");
pub const progress = @import("progress.zig");
const memory = @import("memory.zig");

const router = @import("router.zig");
const scheduler = @import("scheduler.zig");
const batch = @import("batch.zig");
const train = @import("train.zig");
const train_full = @import("train_full.zig");

// Re-export Chat/session lifecycle APIs.
pub const talu_chat_set_ttl_ts = responses.talu_chat_set_ttl_ts;
pub const talu_responses_clone_prefix = responses.talu_responses_clone_prefix;
pub const talu_responses_truncate_after = responses.talu_responses_truncate_after;

// Re-export utility C API functions (model resolution, templates, EOS)
pub const talu_resolve_model_path = session.talu_resolve_model_path;
pub const talu_get_eos_tokens = session.talu_get_eos_tokens;
pub const talu_get_generation_config = session.talu_get_generation_config;
pub const talu_apply_chat_template = session.talu_apply_chat_template;
pub const talu_apply_chat_template_string = session.talu_apply_chat_template_string;

// Re-export tokenizer utility functions
pub const talu_tokens_free = tokenizer.talu_tokens_free;
pub const talu_tokens_concat = tokenizer.talu_tokens_concat;
pub const talu_decode_result_free = tokenizer.talu_decode_result_free;
pub const talu_text_free = tokenizer.talu_text_free;

// Re-export DLPack C API functions.
pub const talu_buffer_to_dlpack = dlpack.talu_buffer_to_dlpack;
pub const talu_batch_to_dlpack = dlpack.talu_batch_to_dlpack;
pub const talu_batch_mask_to_dlpack = dlpack.talu_batch_mask_to_dlpack;

// Re-export tokenizer-only C API functions (lightweight, no model weights).
pub const talu_tokenizer_create = tokenizer.talu_tokenizer_create;
pub const talu_tokenizer_create_from_json = tokenizer.talu_tokenizer_create_from_json;
pub const talu_tokenizer_free = tokenizer.talu_tokenizer_free;
pub const talu_tokenizer_encode = tokenizer.talu_tokenizer_encode;
pub const talu_tokenizer_decode = tokenizer.talu_tokenizer_decode;
pub const talu_tokenizer_get_eos_tokens = tokenizer.talu_tokenizer_get_eos_tokens;
pub const talu_tokenizer_get_model_dir = tokenizer.talu_tokenizer_get_model_dir;
pub const talu_tokenizer_get_vocab_size = tokenizer.talu_tokenizer_get_vocab_size;
pub const talu_tokenizer_get_vocab = tokenizer.talu_tokenizer_get_vocab;
pub const talu_vocab_result_free = tokenizer.talu_vocab_result_free;
pub const talu_tokenizer_get_special_tokens = tokenizer.talu_tokenizer_get_special_tokens;
pub const talu_tokenizer_id_to_token = tokenizer.talu_tokenizer_id_to_token;
pub const talu_tokenizer_token_to_id = tokenizer.talu_tokenizer_token_to_id;
pub const talu_tokenizer_tokenize = tokenizer.talu_tokenizer_tokenize;
pub const talu_tokenize_result_free = tokenizer.talu_tokenize_result_free;
pub const talu_tokenizer_tokenize_bytes = tokenizer.talu_tokenizer_tokenize_bytes;
pub const talu_tokenize_bytes_result_free = tokenizer.talu_tokenize_bytes_result_free;
pub const talu_encode_result_free = tokenizer.talu_encode_result_free;
pub const talu_tokenizer_encode_batch = tokenizer.talu_tokenizer_encode_batch;
pub const talu_batch_encode_result_free = tokenizer.talu_batch_encode_result_free;
pub const talu_batch_to_padded_tensor = tokenizer.talu_batch_to_padded_tensor;
pub const talu_padded_tensor_result_free = tokenizer.talu_padded_tensor_result_free;

// Re-export model description C API functions.
pub const talu_describe = converter.talu_describe;
pub const talu_model_performance_hints = converter.talu_model_performance_hints;
pub const talu_model_info_free = converter.talu_model_info_free;
pub const talu_execution_plan = converter.talu_execution_plan;

// Re-export template C API functions.
pub const talu_template_render = template.talu_template_render;

// Re-export convert C API functions.
pub const talu_convert = converter.talu_convert;
pub const talu_convert_free_string = converter.talu_convert_free_string;
pub const talu_convert_schemes = converter.talu_convert_schemes;

// Re-export repo C API functions.
pub const talu_repo_is_cached = repo.talu_repo_is_cached;
pub const talu_repo_get_cached_path = repo.talu_repo_get_cached_path;
pub const talu_repo_get_hf_home = repo.talu_repo_get_hf_home;
pub const talu_repo_list_models = repo.talu_repo_list_models;
pub const talu_repo_list_count = repo.talu_repo_list_count;
pub const talu_repo_list_get_id = repo.talu_repo_list_get_id;
pub const talu_repo_list_get_path = repo.talu_repo_list_get_path;
pub const talu_repo_list_free = repo.talu_repo_list_free;
pub const talu_repo_delete = repo.talu_repo_delete;
pub const talu_repo_size = repo.talu_repo_size;
pub const talu_repo_total_size = repo.talu_repo_total_size;
pub const talu_repo_is_model_id = repo.talu_repo_is_model_id;
pub const talu_repo_fetch = repo.talu_repo_fetch;
pub const talu_repo_resolve_path = repo.talu_repo_resolve_path;
pub const talu_repo_list = repo.talu_repo_list;
pub const talu_repo_search = repo.talu_repo_search;
pub const talu_repo_string_list_count = repo.talu_repo_string_list_count;
pub const talu_repo_string_list_get = repo.talu_repo_string_list_get;
pub const talu_repo_string_list_free = repo.talu_repo_string_list_free;
pub const talu_repo_search_rich = repo.talu_repo_search_rich;
pub const talu_repo_search_result_count = repo.talu_repo_search_result_count;
pub const talu_repo_search_result_get_id = repo.talu_repo_search_result_get_id;
pub const talu_repo_search_result_get_downloads = repo.talu_repo_search_result_get_downloads;
pub const talu_repo_search_result_get_likes = repo.talu_repo_search_result_get_likes;
pub const talu_repo_search_result_get_params = repo.talu_repo_search_result_get_params;
pub const talu_repo_search_result_get_last_modified = repo.talu_repo_search_result_get_last_modified;
pub const talu_repo_search_result_get_pipeline_tag = repo.talu_repo_search_result_get_pipeline_tag;
pub const talu_repo_search_result_free = repo.talu_repo_search_result_free;

// Re-export error C API functions.
pub const talu_last_error = err.talu_last_error;
pub const talu_last_error_code = err.talu_last_error_code;
pub const talu_clear_error = err.talu_clear_error;
pub const talu_error_buf_size = err.talu_error_buf_size;
pub const talu_take_last_error = err.talu_take_last_error;

// Re-export log configuration C API functions.
pub const talu_set_log_level = log_api.talu_set_log_level;
pub const talu_set_log_format = log_api.talu_set_log_format;
pub const talu_set_log_filter = log_api.talu_set_log_filter;
pub const talu_set_log_callback = log_api.talu_set_log_callback;
pub const talu_get_log_level = log_api.talu_get_log_level;
pub const talu_get_log_format = log_api.talu_get_log_format;

// Re-export memory allocation C API functions.
pub const talu_alloc_string = memory.talu_alloc_string;
pub const talu_free_string = memory.talu_free_string;

// Re-export validate C API functions (chat response-format API).
pub const talu_set_response_format = validate.talu_set_response_format;
pub const talu_clear_response_format = validate.talu_clear_response_format;
pub const talu_validate_response_format = validate.talu_validate_response_format;

// Re-export low-level engine C API functions (isolated validate operations).
pub const talu_validate_engine_create = validate.talu_validate_engine_create;
pub const talu_validate_engine_destroy = validate.talu_validate_engine_destroy;
pub const talu_validate_engine_reset = validate.talu_validate_engine_reset;
pub const talu_validate_engine_is_complete = validate.talu_validate_engine_is_complete;
pub const talu_validate_engine_get_position = validate.talu_validate_engine_get_position;
pub const talu_validate_engine_get_valid_bytes = validate.talu_validate_engine_get_valid_bytes;
pub const talu_validate_engine_can_accept = validate.talu_validate_engine_can_accept;
pub const talu_validate_engine_advance = validate.talu_validate_engine_advance;
pub const talu_validate_engine_validate = validate.talu_validate_engine_validate;

// Re-export SharedBuffer C API functions.
pub const talu_buffer_create_from_owned = buffer.talu_buffer_create_from_owned;
pub const talu_buffer_create_from_copy = buffer.talu_buffer_create_from_copy;
pub const talu_buffer_release = buffer.talu_buffer_release;
pub const talu_buffer_get_data_ptr = buffer.talu_buffer_get_data_ptr;

// Re-export config C API functions.
pub const talu_config_canonicalize = router.talu_config_canonicalize;
pub const talu_config_get_view = router.talu_config_get_view;
pub const talu_config_free = router.talu_config_free;
pub const talu_backend_get_capabilities = router.talu_backend_get_capabilities;
pub const talu_backend_create_from_canonical = router.talu_backend_create_from_canonical;
pub const talu_backend_free = router.talu_backend_free;
pub const talu_backend_synchronize = router.talu_backend_synchronize;
pub const talu_backend_model_info = router.talu_backend_model_info;

// Re-export Chat C API functions (maps to talu/chat/).
pub const talu_chat_create = responses.talu_chat_create;
pub const talu_chat_create_with_system = responses.talu_chat_create_with_system;
pub const talu_chat_create_with_session = responses.talu_chat_create_with_session;
pub const talu_chat_create_with_system_and_session = responses.talu_chat_create_with_system_and_session;
pub const talu_chat_free = responses.talu_chat_free;
pub const talu_chat_get_conversation = responses.talu_chat_get_conversation;
pub const talu_session_id_new = responses.talu_session_id_new;
pub const talu_chat_set_system = responses.talu_chat_set_system;
pub const talu_chat_get_system = responses.talu_chat_get_system;
pub const talu_chat_to_json = responses.talu_chat_to_json;
pub const talu_chat_set_messages = responses.talu_chat_set_messages;
pub const talu_chat_load_completions_json = responses.talu_chat_load_completions_json;
pub const talu_chat_count_tokens = responses.talu_chat_count_tokens;
pub const talu_chat_max_context_length = responses.talu_chat_max_context_length;

// Re-export Router C API functions (routes generation to inference backends).
pub const talu_router_generate_with_backend = router.talu_router_generate_with_backend;
pub const talu_router_generate_streaming = router.talu_router_generate_streaming;
pub const talu_router_result_free = router.talu_router_result_free;
pub const talu_router_close_all = router.talu_router_close_all;
pub const talu_router_embedding_dim = router.talu_router_embedding_dim;
pub const talu_router_embed = router.talu_router_embed;
pub const talu_router_embedding_free = router.talu_router_embedding_free;

// Re-export Scheduler C API functions (scoring).
pub const talu_scheduler_score_tokens_nll = scheduler.talu_scheduler_score_tokens_nll;

// Re-export Batch C API functions (responses-aware continuous batching).
pub const talu_batch_create = batch.talu_batch_create;
pub const talu_batch_destroy = batch.talu_batch_destroy;
pub const talu_batch_submit = batch.talu_batch_submit;
pub const talu_batch_cancel = batch.talu_batch_cancel;
pub const talu_batch_step = batch.talu_batch_step;
pub const talu_batch_has_active = batch.talu_batch_has_active;
pub const talu_batch_active_count = batch.talu_batch_active_count;
pub const talu_batch_run_loop = batch.talu_batch_run_loop;
pub const talu_batch_run_loop_no_text = batch.talu_batch_run_loop_no_text;
pub const talu_batch_take_result = batch.talu_batch_take_result;
pub const talu_batch_result_free = batch.talu_batch_result_free;

// Re-export X-Ray C API functions (tensor inspection during inference).
pub const talu_xray_capture_create = xray.talu_xray_capture_create;
pub const talu_xray_capture_create_all = xray.talu_xray_capture_create_all;
pub const talu_xray_capture_enable = xray.talu_xray_capture_enable;
pub const talu_xray_capture_disable = xray.talu_xray_capture_disable;
pub const talu_xray_capture_is_enabled = xray.talu_xray_capture_is_enabled;
pub const talu_xray_capture_clear = xray.talu_xray_capture_clear;
pub const talu_xray_capture_count = xray.talu_xray_capture_count;
pub const talu_xray_capture_overflow = xray.talu_xray_capture_overflow;
pub const talu_xray_capture_destroy = xray.talu_xray_capture_destroy;
pub const talu_xray_get = xray.talu_xray_get;
pub const talu_xray_find_anomaly = xray.talu_xray_find_anomaly;
pub const talu_xray_count_matching = xray.talu_xray_count_matching;
pub const talu_xray_get_samples = xray.talu_xray_get_samples;
pub const talu_xray_get_data_size = xray.talu_xray_get_data_size;
pub const talu_xray_get_data = xray.talu_xray_get_data;
pub const talu_xray_point_name = xray.talu_xray_point_name;

// Re-export Responses C API functions (Item-based data access).
pub const talu_responses_create = responses.talu_responses_create;
pub const talu_responses_create_with_session = responses.talu_responses_create_with_session;
pub const talu_responses_free = responses.talu_responses_free;
pub const talu_responses_item_count = responses.talu_responses_item_count;
pub const talu_responses_item_type = responses.talu_responses_item_type;
pub const talu_responses_get_item = responses.talu_responses_get_item;
pub const talu_responses_item_as_message = responses.talu_responses_item_as_message;
pub const talu_responses_item_as_function_call = responses.talu_responses_item_as_function_call;
pub const talu_responses_item_as_function_call_output = responses.talu_responses_item_as_function_call_output;
pub const talu_responses_item_as_reasoning = responses.talu_responses_item_as_reasoning;
pub const talu_responses_item_as_item_reference = responses.talu_responses_item_as_item_reference;
pub const talu_responses_item_message_content_count = responses.talu_responses_item_message_content_count;
pub const talu_responses_item_message_get_content = responses.talu_responses_item_message_get_content;
pub const talu_responses_item_reasoning_content_count = responses.talu_responses_item_reasoning_content_count;
pub const talu_responses_item_reasoning_get_content = responses.talu_responses_item_reasoning_get_content;
pub const talu_responses_item_reasoning_summary_count = responses.talu_responses_item_reasoning_summary_count;
pub const talu_responses_item_reasoning_get_summary = responses.talu_responses_item_reasoning_get_summary;
pub const talu_responses_item_fco_get_part = responses.talu_responses_item_fco_get_part;
pub const talu_responses_to_responses_json = responses.talu_responses_to_responses_json;
pub const talu_responses_to_completions_json = responses.talu_responses_to_completions_json;

// Re-export Training C API functions (LoRA fine-tuning sessions).
pub const talu_train_create = train.talu_train_create;
pub const talu_train_destroy = train.talu_train_destroy;
pub const talu_train_load_model = train.talu_train_load_model;
pub const talu_train_configure = train.talu_train_configure;
pub const talu_train_load_data = train.talu_train_load_data;
pub const talu_train_run = train.talu_train_run;
pub const talu_train_save_checkpoint = train.talu_train_save_checkpoint;
pub const talu_train_get_info = train.talu_train_get_info;

// Re-export Full Training C API functions (from-scratch training sessions).
pub const talu_train_full_create = train_full.talu_train_full_create;
pub const talu_train_full_destroy = train_full.talu_train_full_destroy;
pub const talu_train_full_init_model = train_full.talu_train_full_init_model;
pub const talu_train_full_configure = train_full.talu_train_full_configure;
pub const talu_train_full_set_data = train_full.talu_train_full_set_data;
pub const talu_train_full_load_data = train_full.talu_train_full_load_data;
pub const talu_train_full_step = train_full.talu_train_full_step;
pub const talu_train_full_run = train_full.talu_train_full_run;
pub const talu_train_full_get_info = train_full.talu_train_full_get_info;
pub const talu_train_full_copy_weights_f32 = train_full.talu_train_full_copy_weights_f32;
pub const talu_train_full_load_weights_f32 = train_full.talu_train_full_load_weights_f32;
pub const talu_train_full_copy_optimizer_state_f32 = train_full.talu_train_full_copy_optimizer_state_f32;
pub const talu_train_full_load_optimizer_state_f32 = train_full.talu_train_full_load_optimizer_state_f32;
