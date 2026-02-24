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

pub const tensor = @import("tensor.zig");
pub const session = @import("session.zig");
pub const template = @import("template.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const dlpack = @import("dlpack.zig");
pub const converter = @import("converter.zig");
pub const repo = @import("repository.zig");
pub const err = @import("error.zig");
pub const responses = @import("responses.zig");
pub const xray = @import("xray.zig");
pub const validate = @import("validate.zig");
pub const buffer = @import("buffer.zig");
pub const types = @import("types.zig");
pub const provider = @import("provider.zig");
pub const db_table = @import("db/table.zig");
pub const db_vector = @import("db/vector.zig");
pub const db_blob = @import("db/blob.zig");
pub const db_kv = @import("db/kv.zig");
pub const db_sql = @import("db/sql.zig");
pub const db_ops = @import("db/ops.zig");

// Force-include all DB submodule exports (needed for FFI symbol generation).
comptime {
    _ = db_table;
    _ = db_vector;
    _ = db_blob;
    _ = db_kv;
    _ = db_sql;
    _ = db_ops;
}

pub const repo_meta = @import("repo_meta.zig");
pub const sql_api = @import("sql.zig");
pub const log_api = @import("log.zig");
pub const progress = @import("progress.zig");
pub const memory = @import("memory.zig");
pub const policy = @import("policy.zig");
pub const plugins = @import("plugins.zig");
pub const file_api = @import("file.zig");
pub const treesitter = @import("code.zig");

pub const router = @import("router.zig");
pub const agent = @import("agent.zig");

// Re-export Chat/session lifecycle APIs.
pub const talu_chat_set_ttl_ts = responses.talu_chat_set_ttl_ts;
pub const talu_responses_clone_prefix = responses.talu_responses_clone_prefix;
pub const talu_responses_truncate_after = responses.talu_responses_truncate_after;
pub const talu_responses_load_storage_records = responses.talu_responses_load_storage_records;
pub const talu_responses_begin_fork = responses.talu_responses_begin_fork;
pub const talu_responses_end_fork = responses.talu_responses_end_fork;
pub const talu_responses_set_item_parent = responses.talu_responses_set_item_parent;
pub const talu_responses_set_item_validation_flags = responses.talu_responses_set_item_validation_flags;

// Re-export tensor C API functions at the top level.
pub const talu_hello = tensor.talu_hello;
pub const talu_tensor_create = tensor.talu_tensor_create;
pub const talu_tensor_zeros = tensor.talu_tensor_zeros;
pub const talu_tensor_test_embeddings = tensor.talu_tensor_test_embeddings;
pub const talu_tensor_free = tensor.talu_tensor_free;
pub const talu_tensor_data_ptr = tensor.talu_tensor_data_ptr;
pub const talu_tensor_ndim = tensor.talu_tensor_ndim;
pub const talu_tensor_shape = tensor.talu_tensor_shape;
pub const talu_tensor_strides = tensor.talu_tensor_strides;
pub const talu_tensor_dtype = tensor.talu_tensor_dtype;
pub const talu_tensor_typestr = tensor.talu_tensor_typestr;
pub const talu_tensor_device_type = tensor.talu_tensor_device_type;
pub const talu_tensor_device_id = tensor.talu_tensor_device_id;
pub const talu_tensor_is_cpu = tensor.talu_tensor_is_cpu;
pub const talu_tensor_numel = tensor.talu_tensor_numel;
pub const talu_tensor_element_size = tensor.talu_tensor_element_size;
pub const talu_tensor_to_dlpack = tensor.talu_tensor_to_dlpack;
pub const talu_dlpack_capsule_name = tensor.talu_dlpack_capsule_name;
pub const talu_dlpack_used_capsule_name = tensor.talu_dlpack_used_capsule_name;

// Re-export utility C API functions (model resolution, templates, EOS)
pub const talu_resolve_model_path = session.talu_resolve_model_path;
pub const talu_get_eos_tokens = session.talu_get_eos_tokens;
pub const talu_get_generation_config = session.talu_get_generation_config;
pub const talu_apply_chat_template = session.talu_apply_chat_template;
pub const talu_apply_chat_template_string = session.talu_apply_chat_template_string;
pub const SamplingParams = session.SamplingParams;

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
pub const EncodeOptions = tokenizer.EncodeOptions;
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
pub const TokenizeBytesResult = tokenizer.TokenizeBytesResult;
pub const VocabResult = tokenizer.VocabResult;
pub const TokenOffset = tokenizer.TokenOffset;
pub const BatchEncodeResult = tokenizer.BatchEncodeResult;
pub const PaddedTensorOptions = tokenizer.PaddedTensorOptions;
pub const PaddedTensorResult = tokenizer.PaddedTensorResult;
pub const DecodeOptionsC = tokenizer.DecodeOptionsC;

// Re-export model description C API functions.
pub const talu_describe = converter.talu_describe;
pub const talu_model_info_free = converter.talu_model_info_free;
pub const talu_execution_plan = converter.talu_execution_plan;
pub const ExecutionPlanInfo = converter.ExecutionPlanInfo;

// Re-export template C API functions.
pub const talu_template_render = template.talu_template_render;

// Re-export convert C API functions.
pub const talu_convert = converter.talu_convert;
pub const talu_convert_free_string = converter.talu_convert_free_string;
pub const talu_convert_schemes = converter.talu_convert_schemes;

// Re-export image C API functions.
pub const talu_image_decode = file_api.talu_image_decode;
pub const talu_image_convert = file_api.talu_image_convert;
pub const talu_image_to_model_input = file_api.talu_image_to_model_input;
pub const talu_image_encode = file_api.talu_image_encode;
pub const talu_image_free = file_api.talu_image_free;
pub const talu_model_buffer_free = file_api.talu_model_buffer_free;
pub const talu_image_encode_free = file_api.talu_image_encode_free;
pub const TaluImage = file_api.TaluImage;
pub const TaluImageDecodeOptions = file_api.TaluImageDecodeOptions;
pub const TaluImageConvertOptions = file_api.TaluImageConvertOptions;
pub const TaluImageResizeOptions = file_api.TaluImageResizeOptions;
pub const TaluModelInputSpec = file_api.TaluModelInputSpec;
pub const TaluModelBuffer = file_api.TaluModelBuffer;
pub const TaluImageEncodeOptions = file_api.TaluImageEncodeOptions;

// Re-export file-level C API functions.
pub const talu_file_inspect = file_api.talu_file_inspect;
pub const talu_file_info_free = file_api.talu_file_info_free;
pub const talu_file_transform = file_api.talu_file_transform;
pub const talu_file_bytes_free = file_api.talu_file_bytes_free;
pub const talu_pdf_render_page = file_api.talu_pdf_render_page;
pub const talu_pdf_page_count = file_api.talu_pdf_page_count;
pub const talu_pdf_transform_page = file_api.talu_pdf_transform_page;
pub const TaluFileInfo = file_api.TaluFileInfo;
pub const TaluImageInfo = file_api.TaluImageInfo;
pub const TaluFileTransformOptions = file_api.TaluFileTransformOptions;

// Re-export repo C API functions.
pub const talu_repo_is_cached = repo.talu_repo_is_cached;
pub const talu_repo_get_cached_path = repo.talu_repo_get_cached_path;
pub const talu_repo_get_hf_home = repo.talu_repo_get_hf_home;
pub const talu_repo_get_cache_dir = repo.talu_repo_get_cache_dir;
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
pub const talu_repo_exists = repo.talu_repo_exists;
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
pub const CachedModelList = repo.CachedModelList;
pub const SearchResultList = repo.SearchResultList;
pub const StringList = repo.StringList;
pub const DownloadOptions = repo.DownloadOptions;

// Re-export repo metadata C API functions.
pub const talu_repo_meta_init = repo_meta.talu_repo_meta_init;
pub const talu_repo_meta_free = repo_meta.talu_repo_meta_free;
pub const talu_repo_meta_pin = repo_meta.talu_repo_meta_pin;
pub const talu_repo_meta_unpin = repo_meta.talu_repo_meta_unpin;
pub const talu_repo_meta_update_size = repo_meta.talu_repo_meta_update_size;
pub const talu_repo_meta_clear_size = repo_meta.talu_repo_meta_clear_size;
pub const talu_repo_meta_list_pins = repo_meta.talu_repo_meta_list_pins;
pub const talu_repo_meta_free_list = repo_meta.talu_repo_meta_free_list;
pub const PinAdapterHandle = repo_meta.PinAdapterHandle;
pub const CPinRecord = repo_meta.CPinRecord;
pub const CPinList = repo_meta.CPinList;

// Re-export unified DB kv plane symbols.
pub const KvHandle = db_kv.KvHandle;
pub const CKvValue = db_kv.CKvValue;
pub const CKvEntry = db_kv.CKvEntry;
pub const CKvList = db_kv.CKvList;
pub const talu_db_kv_init = db_kv.talu_db_kv_init;
pub const talu_db_kv_free = db_kv.talu_db_kv_free;
pub const talu_db_kv_put = db_kv.talu_db_kv_put;
pub const talu_db_kv_get = db_kv.talu_db_kv_get;
pub const talu_db_kv_delete = db_kv.talu_db_kv_delete;
pub const talu_db_kv_list = db_kv.talu_db_kv_list;
pub const talu_db_kv_free_value = db_kv.talu_db_kv_free_value;
pub const talu_db_kv_free_list = db_kv.talu_db_kv_free_list;
pub const talu_db_kv_flush = db_kv.talu_db_kv_flush;
pub const talu_db_kv_compact = db_kv.talu_db_kv_compact;

// Re-export SQL C API functions.
pub const talu_sql_query = sql_api.talu_sql_query;
pub const talu_sql_query_free = sql_api.talu_sql_query_free;
pub const talu_db_sql_query = db_sql.talu_db_sql_query;
pub const talu_db_sql_query_free = db_sql.talu_db_sql_query_free;

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

// Re-export validate C API functions (high-level sampler API).
pub const talu_validate_create = validate.talu_validate_create;
pub const talu_validate_free = validate.talu_validate_free;
pub const talu_validate_apply = validate.talu_validate_apply;
pub const talu_validate_accept = validate.talu_validate_accept;
pub const talu_validate_is_complete = validate.talu_validate_is_complete;
pub const talu_validate_reset = validate.talu_validate_reset;
pub const ValidateHandle = validate.ValidateHandle;
pub const ValidateConfigC = validate.ValidateConfigC;
pub const SemanticValidationResultC = validate.SemanticValidationResultC;
pub const talu_set_response_format = validate.talu_set_response_format;
pub const talu_clear_response_format = validate.talu_clear_response_format;
pub const talu_validate_response_format = validate.talu_validate_response_format;

// Re-export low-level engine C API functions (isolated validate operations).
pub const talu_validate_engine_create = validate.talu_validate_engine_create;
pub const talu_validate_engine_destroy = validate.talu_validate_engine_destroy;
pub const talu_validate_engine_reset = validate.talu_validate_engine_reset;
pub const talu_validate_engine_is_complete = validate.talu_validate_engine_is_complete;
pub const talu_validate_engine_get_position = validate.talu_validate_engine_get_position;
pub const talu_validate_engine_get_state_count = validate.talu_validate_engine_get_state_count;
pub const talu_validate_engine_get_valid_bytes = validate.talu_validate_engine_get_valid_bytes;
pub const talu_validate_engine_count_valid_bytes = validate.talu_validate_engine_count_valid_bytes;
pub const talu_validate_engine_can_accept = validate.talu_validate_engine_can_accept;
pub const talu_validate_engine_advance_byte = validate.talu_validate_engine_advance_byte;
pub const talu_validate_engine_advance = validate.talu_validate_engine_advance;
pub const talu_validate_engine_validate = validate.talu_validate_engine_validate;
pub const talu_validate_engine_get_valid_tokens = validate.talu_validate_engine_get_valid_tokens;
pub const talu_validate_engine_get_valid_tokens_with_tokenizer = validate.talu_validate_engine_get_valid_tokens_with_tokenizer;
pub const talu_validate_engine_get_deterministic_continuation = validate.talu_validate_engine_get_deterministic_continuation;
pub const ValidateEngineHandle = validate.ValidateEngineHandle;
pub const TokenInfoCallback = validate.TokenInfoCallback;

// Re-export token mask C API functions.
pub const talu_token_mask_create = validate.talu_token_mask_create;
pub const talu_token_mask_destroy = validate.talu_token_mask_destroy;
pub const talu_token_mask_clear = validate.talu_token_mask_clear;
pub const talu_token_mask_set_all = validate.talu_token_mask_set_all;
pub const talu_token_mask_is_valid = validate.talu_token_mask_is_valid;
pub const talu_token_mask_set = validate.talu_token_mask_set;
pub const talu_token_mask_get_size = validate.talu_token_mask_get_size;
pub const talu_token_mask_get_bits = validate.talu_token_mask_get_bits;
pub const talu_token_mask_get_word_count = validate.talu_token_mask_get_word_count;
pub const talu_token_mask_count_valid = validate.talu_token_mask_count_valid;
pub const talu_token_mask_apply = validate.talu_token_mask_apply;
pub const TokenMaskHandle = validate.TokenMaskHandle;

// Re-export SharedBuffer C API functions.
pub const talu_buffer_create_from_owned = buffer.talu_buffer_create_from_owned;
pub const talu_buffer_create_from_copy = buffer.talu_buffer_create_from_copy;
pub const talu_buffer_create_uninitialized = buffer.talu_buffer_create_uninitialized;
pub const talu_buffer_retain = buffer.talu_buffer_retain;
pub const talu_buffer_release = buffer.talu_buffer_release;
pub const talu_buffer_get_data_ptr = buffer.talu_buffer_get_data_ptr;
pub const talu_buffer_get_capacity = buffer.talu_buffer_get_capacity;
pub const talu_buffer_get_refcount = buffer.talu_buffer_get_refcount;
pub const BufferHandle = buffer.BufferHandle;
pub const SharedBuffer = buffer.SharedBuffer;

// Re-export config C API functions.
pub const talu_config_validate = router.talu_config_validate;
pub const talu_config_canonicalize = router.talu_config_canonicalize;
pub const talu_config_get_view = router.talu_config_get_view;
pub const talu_config_free = router.talu_config_free;
pub const talu_backend_get_capabilities = router.talu_backend_get_capabilities;
pub const talu_backend_create_from_canonical = router.talu_backend_create_from_canonical;
pub const talu_backend_free = router.talu_backend_free;
pub const talu_backend_list_models = router.talu_backend_list_models;
pub const talu_backend_list_models_free = router.talu_backend_list_models_free;
pub const RemoteModelInfo = router.RemoteModelInfo;
pub const RemoteModelListResult = router.RemoteModelListResult;

// Re-export config types.
pub const BackendType = types.BackendType;
pub const LocalConfig = types.LocalConfig;
pub const OpenAICompatibleConfig = types.OpenAICompatibleConfig;
pub const BackendUnion = types.BackendUnion;
pub const TaluModelSpec = types.TaluModelSpec;
pub const TaluCapabilities = types.TaluCapabilities;
pub const TaluCanonicalSpec = router.TaluCanonicalSpec;
pub const TaluInferenceBackend = router.TaluInferenceBackend;

// Re-export types.
pub const Tensor = tensor.Tensor;
pub const DType = tensor.DType;
pub const Device = tensor.Device;
pub const DLManagedTensor = tensor.DLManagedTensor;
pub const TokenizerHandle = tokenizer.TokenizerHandle;
pub const EncodeResult = tokenizer.EncodeResult;
pub const DecodeResult = tokenizer.DecodeResult;
pub const TokenizeResult = tokenizer.TokenizeResult;
pub const SpecialTokensResult = tokenizer.SpecialTokensResult;
pub const EosTokenResult = tokenizer.EosTokenResult;
pub const ModelInfo = converter.ModelInfo;
pub const ConvertOptions = converter.ConvertOptions;
pub const ConvertResult = converter.ConvertResult;
pub const Scheme = converter.Scheme;
pub const ProgressUpdate = progress.ProgressUpdate;
pub const ProgressAction = progress.ProgressAction;
pub const CProgressCallback = progress.CProgressCallback;

// Re-export Chat C API functions (maps to talu/chat/).
pub const talu_chat_create = responses.talu_chat_create;
pub const talu_chat_create_with_system = responses.talu_chat_create_with_system;
pub const talu_chat_create_with_session = responses.talu_chat_create_with_session;
pub const talu_chat_create_with_system_and_session = responses.talu_chat_create_with_system_and_session;
pub const talu_chat_free = responses.talu_chat_free;
pub const talu_chat_get_conversation = responses.talu_chat_get_conversation;
pub const talu_chat_get_session_id = responses.talu_chat_get_session_id;
pub const talu_session_id_new = responses.talu_session_id_new;
pub const talu_chat_get_temperature = responses.talu_chat_get_temperature;
pub const talu_chat_set_temperature = responses.talu_chat_set_temperature;
pub const talu_chat_get_max_tokens = responses.talu_chat_get_max_tokens;
pub const talu_chat_set_max_tokens = responses.talu_chat_set_max_tokens;
pub const talu_chat_get_top_k = responses.talu_chat_get_top_k;
pub const talu_chat_set_top_k = responses.talu_chat_set_top_k;
pub const talu_chat_get_top_p = responses.talu_chat_get_top_p;
pub const talu_chat_set_top_p = responses.talu_chat_set_top_p;
pub const talu_chat_get_min_p = responses.talu_chat_get_min_p;
pub const talu_chat_set_min_p = responses.talu_chat_set_min_p;
pub const talu_chat_get_repetition_penalty = responses.talu_chat_get_repetition_penalty;
pub const talu_chat_set_repetition_penalty = responses.talu_chat_set_repetition_penalty;
pub const talu_chat_set_system = responses.talu_chat_set_system;
pub const talu_chat_get_system = responses.talu_chat_get_system;
pub const talu_chat_clear = responses.talu_chat_clear;
pub const talu_chat_reset = responses.talu_chat_reset;
pub const talu_chat_len = responses.talu_chat_len;
pub const talu_chat_to_json = responses.talu_chat_to_json;
pub const talu_chat_set_messages = responses.talu_chat_set_messages;
pub const talu_chat_get_messages = responses.talu_chat_get_messages;
pub const talu_chat_count_tokens = responses.talu_chat_count_tokens;
pub const talu_chat_max_context_length = responses.talu_chat_max_context_length;
pub const ChatHandle = responses.ChatHandle;
pub const ChatCreateOptions = responses.ChatCreateOptions;

pub const CStorageRecord = db_ops.CStorageRecord;
pub const CStorageEvent = db_ops.CStorageEvent;
pub const CStorageEventType = db_ops.CStorageEventType;
pub const CItemType = db_ops.CItemType;
pub const CMessageRole = db_ops.CMessageRole;
pub const CSessionRecord = db_ops.CSessionRecord;
pub const VectorStoreHandle = db_vector.VectorStoreHandle;

// Re-export unified DB vector plane symbols.
pub const talu_db_vector_init = db_vector.talu_db_vector_init;
pub const talu_db_vector_free = db_vector.talu_db_vector_free;
pub const talu_db_vector_simulate_crash = db_vector.talu_db_vector_simulate_crash;
pub const talu_db_vector_set_durability = db_vector.talu_db_vector_set_durability;
pub const talu_db_vector_append = db_vector.talu_db_vector_append;
pub const talu_db_vector_append_ex = db_vector.talu_db_vector_append_ex;
pub const talu_db_vector_append_idempotent_ex = db_vector.talu_db_vector_append_idempotent_ex;
pub const talu_db_vector_upsert = db_vector.talu_db_vector_upsert;
pub const talu_db_vector_upsert_ex = db_vector.talu_db_vector_upsert_ex;
pub const talu_db_vector_upsert_idempotent_ex = db_vector.talu_db_vector_upsert_idempotent_ex;
pub const talu_db_vector_delete = db_vector.talu_db_vector_delete;
pub const talu_db_vector_delete_idempotent = db_vector.talu_db_vector_delete_idempotent;
pub const talu_db_vector_fetch = db_vector.talu_db_vector_fetch;
pub const talu_db_vector_free_fetch = db_vector.talu_db_vector_free_fetch;
pub const talu_db_vector_stats = db_vector.talu_db_vector_stats;
pub const talu_db_vector_compact = db_vector.talu_db_vector_compact;
pub const talu_db_vector_compact_idempotent = db_vector.talu_db_vector_compact_idempotent;
pub const talu_db_vector_compact_with_generation = db_vector.talu_db_vector_compact_with_generation;
pub const talu_db_vector_compact_expired_tombstones = db_vector.talu_db_vector_compact_expired_tombstones;
pub const talu_db_vector_build_indexes_with_generation = db_vector.talu_db_vector_build_indexes_with_generation;
pub const talu_db_vector_changes = db_vector.talu_db_vector_changes;
pub const talu_db_vector_free_changes = db_vector.talu_db_vector_free_changes;
pub const talu_db_vector_load = db_vector.talu_db_vector_load;
pub const talu_db_vector_free_load = db_vector.talu_db_vector_free_load;
pub const talu_db_vector_search = db_vector.talu_db_vector_search;
pub const talu_db_vector_free_search = db_vector.talu_db_vector_free_search;
pub const talu_db_vector_search_batch = db_vector.talu_db_vector_search_batch;
pub const talu_db_vector_search_batch_ex = db_vector.talu_db_vector_search_batch_ex;
pub const talu_db_vector_scan_batch = db_vector.talu_db_vector_scan_batch;
pub const talu_db_vector_free_search_batch = db_vector.talu_db_vector_free_search_batch;
pub const talu_db_ops_set_storage_db = db_ops.talu_db_ops_set_storage_db;
pub const talu_db_ops_set_max_segment_size = db_ops.talu_db_ops_set_max_segment_size;
pub const talu_db_ops_set_durability = db_ops.talu_db_ops_set_durability;
pub const talu_db_ops_simulate_crash = db_ops.talu_db_ops_simulate_crash;
pub const talu_db_ops_vector_compact = db_ops.talu_db_ops_vector_compact;
pub const talu_db_ops_vector_build_indexes_with_generation = db_ops.talu_db_ops_vector_build_indexes_with_generation;
pub const talu_db_ops_snapshot_create = db_ops.talu_db_ops_snapshot_create;
pub const talu_db_ops_snapshot_release = db_ops.talu_db_ops_snapshot_release;

// Re-export Router C API functions (routes generation to inference backends).
pub const talu_router_generate_with_backend = router.talu_router_generate_with_backend;
pub const talu_router_result_free = router.talu_router_result_free;
pub const talu_router_close_all = router.talu_router_close_all;
pub const talu_router_embedding_dim = router.talu_router_embedding_dim;
pub const talu_router_embed = router.talu_router_embed;
pub const talu_router_embedding_free = router.talu_router_embedding_free;
pub const RouterGenerateResult = router.RouterGenerateResult;
pub const RouterGenerateConfig = router.RouterGenerateConfig;
pub const GenerateContentPart = router.GenerateContentPart;

// Re-export Iterator API functions (pull-based streaming).
pub const talu_router_create_iterator = router.talu_router_create_iterator;
pub const talu_router_iterator_next = router.talu_router_iterator_next;
pub const talu_router_iterator_has_error = router.talu_router_iterator_has_error;
pub const talu_router_iterator_error_code = router.talu_router_iterator_error_code;
pub const talu_router_iterator_error_msg = router.talu_router_iterator_error_msg;
pub const talu_router_iterator_cancel = router.talu_router_iterator_cancel;
pub const talu_router_iterator_free = router.talu_router_iterator_free;
pub const talu_router_iterator_prompt_tokens = router.talu_router_iterator_prompt_tokens;
pub const talu_router_iterator_completion_tokens = router.talu_router_iterator_completion_tokens;
pub const talu_router_iterator_prefill_ns = router.talu_router_iterator_prefill_ns;
pub const talu_router_iterator_generation_ns = router.talu_router_iterator_generation_ns;
pub const TaluTokenIterator = router.TaluTokenIterator;

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
pub const XrayCaptureHandle = xray.CaptureHandle;
pub const XrayTensorStats = xray.TensorStats;
pub const XrayCapturedTensorInfo = xray.CapturedTensorInfo;
pub const XRAY_CAPTURE_MODE_STATS = xray.CAPTURE_MODE_STATS;
pub const XRAY_CAPTURE_MODE_SAMPLE = xray.CAPTURE_MODE_SAMPLE;
pub const XRAY_CAPTURE_MODE_FULL = xray.CAPTURE_MODE_FULL;
pub const XRAY_POINT_ALL = xray.POINT_ALL;

// Re-export Provider C API functions (remote provider registry).
pub const talu_provider_count = provider.talu_provider_count;
pub const talu_provider_get = provider.talu_provider_get;
pub const talu_provider_get_by_name = provider.talu_provider_get_by_name;
pub const talu_provider_parse = provider.talu_provider_parse;
pub const talu_provider_has_prefix = provider.talu_provider_has_prefix;
pub const CProviderInfo = provider.CProviderInfo;

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
pub const talu_responses_validate = responses.talu_responses_validate;
pub const talu_chat_validate = responses.talu_chat_validate;
pub const ResponsesHandle = responses.ResponsesHandle;
pub const CItem = responses.CItem;
pub const CMessageItem = responses.CMessageItem;
pub const CFunctionCallItem = responses.CFunctionCallItem;
pub const CFunctionCallOutputItem = responses.CFunctionCallOutputItem;
pub const CReasoningItem = responses.CReasoningItem;
pub const CItemReferenceItem = responses.CItemReferenceItem;
pub const ConversationCContentPart = responses.CContentPart;

// Re-export Policy C API functions.
pub const talu_policy_create = policy.talu_policy_create;
pub const talu_policy_free = policy.talu_policy_free;
pub const talu_policy_evaluate = policy.talu_policy_evaluate;
pub const talu_policy_get_mode = policy.talu_policy_get_mode;
pub const talu_chat_set_policy = policy.talu_chat_set_policy;
pub const PolicyHandle = policy.PolicyHandle;

pub const CDocumentRecord = db_table.CDocumentRecord;
pub const CDocumentSummary = db_table.CDocumentSummary;
pub const CDocumentList = db_table.CDocumentList;
pub const CDocStringList = db_table.CStringList;
pub const CSearchResult = db_table.CSearchResult;
pub const CSearchResultList = db_table.CSearchResultList;
pub const CChangeRecord = db_table.CChangeRecord;
pub const CChangeList = db_table.CChangeList;
pub const CDeltaChain = db_table.CDeltaChain;
pub const CCompactionStats = db_table.CCompactionStats;

// Re-export unified DB table plane symbols (documents).
pub const talu_db_table_create = db_table.talu_db_table_create;
pub const talu_db_table_get = db_table.talu_db_table_get;
pub const talu_db_table_get_blob_ref = db_table.talu_db_table_get_blob_ref;
pub const talu_db_table_update = db_table.talu_db_table_update;
pub const talu_db_table_delete = db_table.talu_db_table_delete;
pub const talu_db_table_delete_batch = db_table.talu_db_table_delete_batch;
pub const talu_db_table_set_marker_batch = db_table.talu_db_table_set_marker_batch;
pub const talu_db_table_list = db_table.talu_db_table_list;
pub const talu_db_table_free_list = db_table.talu_db_table_free_list;
pub const talu_db_table_search = db_table.talu_db_table_search;
pub const talu_db_table_free_search_results = db_table.talu_db_table_free_search_results;
pub const talu_db_table_search_batch = db_table.talu_db_table_search_batch;
pub const talu_db_table_free_json = db_table.talu_db_table_free_json;
pub const talu_db_table_get_changes = db_table.talu_db_table_get_changes;
pub const talu_db_table_free_changes = db_table.talu_db_table_free_changes;
pub const talu_db_table_set_ttl = db_table.talu_db_table_set_ttl;
pub const talu_db_table_count_expired = db_table.talu_db_table_count_expired;
pub const talu_db_table_create_delta = db_table.talu_db_table_create_delta;
pub const talu_db_table_get_delta_chain = db_table.talu_db_table_get_delta_chain;
pub const talu_db_table_free_delta_chain = db_table.talu_db_table_free_delta_chain;
pub const talu_db_table_is_delta = db_table.talu_db_table_is_delta;
pub const talu_db_table_get_base_id = db_table.talu_db_table_get_base_id;
pub const talu_db_table_get_compaction_stats = db_table.talu_db_table_get_compaction_stats;
pub const talu_db_table_purge_expired = db_table.talu_db_table_purge_expired;
pub const talu_db_table_get_garbage_candidates = db_table.talu_db_table_get_garbage_candidates;
pub const talu_db_table_add_tag = db_table.talu_db_table_add_tag;
pub const talu_db_table_remove_tag = db_table.talu_db_table_remove_tag;
pub const talu_db_table_get_tags = db_table.talu_db_table_get_tags;
pub const talu_db_table_get_by_tag = db_table.talu_db_table_get_by_tag;
pub const talu_db_table_free_string_list = db_table.talu_db_table_free_string_list;

// Re-export unified DB table plane symbols (sessions + tags).
pub const CSessionList = db_table.CSessionList;
pub const CTagRecord = db_table.CTagRecord;
pub const CTagList = db_table.CTagList;
pub const CRelationStringList = db_table.CRelationStringList;
pub const talu_db_table_session_list = db_table.talu_db_table_session_list;
pub const talu_db_table_session_list_ex = db_table.talu_db_table_session_list_ex;
pub const talu_db_table_session_list_by_source = db_table.talu_db_table_session_list_by_source;
pub const talu_db_table_session_list_batch = db_table.talu_db_table_session_list_batch;
pub const talu_db_table_session_get = db_table.talu_db_table_session_get;
pub const talu_db_table_session_update = db_table.talu_db_table_session_update;
pub const talu_db_table_session_update_ex = db_table.talu_db_table_session_update_ex;
pub const talu_db_table_session_fork = db_table.talu_db_table_session_fork;
pub const talu_db_table_session_delete = db_table.talu_db_table_session_delete;
pub const talu_db_table_session_load_conversation = db_table.talu_db_table_session_load_conversation;
pub const talu_db_table_session_free_list = db_table.talu_db_table_session_free_list;
pub const talu_db_table_tag_list = db_table.talu_db_table_tag_list;
pub const talu_db_table_tag_get = db_table.talu_db_table_tag_get;
pub const talu_db_table_tag_get_by_name = db_table.talu_db_table_tag_get_by_name;
pub const talu_db_table_tag_create = db_table.talu_db_table_tag_create;
pub const talu_db_table_tag_update = db_table.talu_db_table_tag_update;
pub const talu_db_table_tag_delete = db_table.talu_db_table_tag_delete;
pub const talu_db_table_tag_free_list = db_table.talu_db_table_tag_free_list;
pub const talu_db_table_session_add_tag = db_table.talu_db_table_session_add_tag;
pub const talu_db_table_session_remove_tag = db_table.talu_db_table_session_remove_tag;
pub const talu_db_table_session_get_tags = db_table.talu_db_table_session_get_tags;
pub const talu_db_table_tag_get_conversations = db_table.talu_db_table_tag_get_conversations;
pub const talu_db_table_free_relation_string_list = db_table.talu_db_table_free_relation_string_list;
pub const talu_db_table_sessions_get_tags_batch = db_table.talu_db_table_sessions_get_tags_batch;
pub const talu_db_table_free_session_tag_batch = db_table.talu_db_table_free_session_tag_batch;
pub const CSessionTagBatch = db_table.CSessionTagBatch;
pub const talu_chat_inherit_tags = db_table.talu_chat_inherit_tags;
pub const talu_chat_notify_session_update = responses.talu_chat_notify_session_update;

pub const BlobGcStats = db_blob.BlobGcStats;
pub const BlobStreamHandle = db_blob.BlobStreamHandle;
pub const BlobWriteStreamHandle = db_blob.BlobWriteStreamHandle;
pub const CBlobStringList = db_blob.CStringList;

// Re-export unified DB blob plane symbols.
pub const talu_db_blob_put = db_blob.talu_db_blob_put;
pub const talu_db_blob_gc = db_blob.talu_db_blob_gc;
pub const talu_db_blob_exists = db_blob.talu_db_blob_exists;
pub const talu_db_blob_list = db_blob.talu_db_blob_list;
pub const talu_db_blob_free_string_list = db_blob.talu_db_blob_free_string_list;
pub const talu_db_blob_open_write_stream = db_blob.talu_db_blob_open_write_stream;
pub const talu_db_blob_write_stream_write = db_blob.talu_db_blob_write_stream_write;
pub const talu_db_blob_write_stream_finish = db_blob.talu_db_blob_write_stream_finish;
pub const talu_db_blob_write_stream_close = db_blob.talu_db_blob_write_stream_close;
pub const talu_db_blob_open_stream = db_blob.talu_db_blob_open_stream;
pub const talu_db_blob_stream_read = db_blob.talu_db_blob_stream_read;
pub const talu_db_blob_stream_total_size = db_blob.talu_db_blob_stream_total_size;
pub const talu_db_blob_stream_seek = db_blob.talu_db_blob_stream_seek;
pub const talu_db_blob_stream_close = db_blob.talu_db_blob_stream_close;

// Re-export Plugins C API functions (UI plugin discovery).
pub const talu_plugins_scan = plugins.talu_plugins_scan;
pub const talu_plugins_list_count = plugins.talu_plugins_list_count;
pub const talu_plugins_list_get = plugins.talu_plugins_list_get;
pub const talu_plugins_list_free = plugins.talu_plugins_list_free;
pub const CPluginInfo = plugins.CPluginInfo;
pub const CPluginList = plugins.CPluginList;

// Re-export Agent C API functions (tool registry + agent loop).
pub const talu_agent_registry_create = agent.talu_agent_registry_create;
pub const talu_agent_registry_free = agent.talu_agent_registry_free;
pub const talu_agent_registry_add = agent.talu_agent_registry_add;
pub const talu_agent_registry_count = agent.talu_agent_registry_count;
pub const talu_agent_run = agent.talu_agent_run;
pub const TaluToolRegistry = agent.TaluToolRegistry;
pub const CAgentLoopConfig = agent.CAgentLoopConfig;
pub const CAgentLoopResult = agent.CAgentLoopResult;
pub const CToolExecuteFn = agent.CToolExecuteFn;

// Re-export Stateful Agent C API functions.
pub const talu_agent_create = agent.talu_agent_create;
pub const talu_agent_free = agent.talu_agent_free;
pub const talu_agent_set_system = agent.talu_agent_set_system;
pub const talu_agent_register_tool = agent.talu_agent_register_tool;
pub const talu_agent_set_bus = agent.talu_agent_set_bus;
pub const talu_agent_prompt = agent.talu_agent_prompt;
pub const talu_agent_continue = agent.talu_agent_continue;
pub const talu_agent_heartbeat = agent.talu_agent_heartbeat;
pub const talu_agent_abort = agent.talu_agent_abort;
pub const talu_agent_get_chat = agent.talu_agent_get_chat;
pub const talu_agent_get_id = agent.talu_agent_get_id;
pub const TaluAgent = agent.TaluAgent;
pub const CAgentCreateConfig = agent.CAgentCreateConfig;

// Re-export Agent goal C API functions.
pub const talu_agent_add_goal = agent.talu_agent_add_goal;
pub const talu_agent_remove_goal = agent.talu_agent_remove_goal;
pub const talu_agent_clear_goals = agent.talu_agent_clear_goals;
pub const talu_agent_goal_count = agent.talu_agent_goal_count;

// Re-export Agent context injection C API.
pub const talu_agent_set_context_inject = agent.talu_agent_set_context_inject;

// Re-export Agent tool middleware C API.
pub const talu_agent_set_before_tool = agent.talu_agent_set_before_tool;
pub const talu_agent_set_after_tool = agent.talu_agent_set_after_tool;

// Re-export Agent active receiver C API.
pub const talu_agent_wait_for_message = agent.talu_agent_wait_for_message;
pub const talu_agent_run_loop = agent.talu_agent_run_loop;

// Re-export Agent vector store RAG C API.
pub const talu_agent_set_vector_store = agent.talu_agent_set_vector_store;
pub const CRagConfig = agent.CRagConfig;

// Re-export MessageBus C API functions.
pub const talu_agent_bus_create = agent.talu_agent_bus_create;
pub const talu_agent_bus_free = agent.talu_agent_bus_free;
pub const talu_agent_bus_register = agent.talu_agent_bus_register;
pub const talu_agent_bus_unregister = agent.talu_agent_bus_unregister;
pub const talu_agent_bus_add_peer = agent.talu_agent_bus_add_peer;
pub const talu_agent_bus_remove_peer = agent.talu_agent_bus_remove_peer;
pub const talu_agent_bus_send = agent.talu_agent_bus_send;
pub const talu_agent_bus_deliver = agent.talu_agent_bus_deliver;
pub const talu_agent_bus_broadcast = agent.talu_agent_bus_broadcast;
pub const talu_agent_bus_pending = agent.talu_agent_bus_pending;
pub const talu_agent_bus_set_notify = agent.talu_agent_bus_set_notify;
pub const TaluAgentBus = agent.TaluAgentBus;

// Re-export Tree-sitter C API functions (code parsing, highlighting, querying).
pub const talu_treesitter_parser_create = treesitter.talu_treesitter_parser_create;
pub const talu_treesitter_parser_free = treesitter.talu_treesitter_parser_free;
pub const talu_treesitter_parse = treesitter.talu_treesitter_parse;
pub const talu_treesitter_tree_edit = treesitter.talu_treesitter_tree_edit;
pub const talu_treesitter_tree_free = treesitter.talu_treesitter_tree_free;
pub const talu_treesitter_tree_sexp = treesitter.talu_treesitter_tree_sexp;
pub const talu_treesitter_highlight = treesitter.talu_treesitter_highlight;
pub const talu_treesitter_query_create = treesitter.talu_treesitter_query_create;
pub const talu_treesitter_query_free = treesitter.talu_treesitter_query_free;
pub const talu_treesitter_query_exec = treesitter.talu_treesitter_query_exec;
pub const talu_treesitter_languages = treesitter.talu_treesitter_languages;
pub const talu_treesitter_language_from_filename = treesitter.talu_treesitter_language_from_filename;
pub const talu_treesitter_tree_json = treesitter.talu_treesitter_tree_json;
pub const talu_treesitter_highlight_rich = treesitter.talu_treesitter_highlight_rich;
pub const talu_treesitter_parse_incremental = treesitter.talu_treesitter_parse_incremental;
pub const talu_treesitter_highlight_from_tree = treesitter.talu_treesitter_highlight_from_tree;
pub const talu_treesitter_highlight_rich_from_tree = treesitter.talu_treesitter_highlight_rich_from_tree;
pub const talu_treesitter_extract_callables = treesitter.talu_treesitter_extract_callables;
pub const talu_treesitter_extract_call_sites = treesitter.talu_treesitter_extract_call_sites;
pub const TreeSitterParserHandle = treesitter.TreeSitterParserHandle;
pub const TreeSitterTreeHandle = treesitter.TreeSitterTreeHandle;
pub const TreeSitterQueryHandle = treesitter.TreeSitterQueryHandle;

// ABI validation - comptime assertions ensure struct sizes match expected values.
// When struct layouts change, update abi.zig and bindings/python/talu/_abi.py.
pub const abi = @import("abi.zig");
