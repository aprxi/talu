//! Tokenizer pipeline facade.
//!
//! Re-exports tokenizer modules (normalization, pretokenization, encoding,
//! post-processing, and C API entrypoints).

pub const pretokenize = @import("pretokenize.zig");
pub const normalize = @import("normalize.zig");
pub const encode = @import("encode.zig");
pub const postprocess = @import("postprocess.zig");
pub const c_api = @import("c_api.zig");

pub const tokenizer_pretokenizer_free = pretokenize.tokenizer_pretokenizer_free;
pub const tokenizer_pretokenizer_set = pretokenize.tokenizer_pretokenizer_set;
pub const tokenizer_apply_pretokenizer_spec = pretokenize.tokenizer_apply_pretokenizer_spec;

pub const tokenizer_apply_normalizer_spec = normalize.tokenizer_apply_normalizer_spec;

pub const tokenizer_apply_postprocessor_spec = postprocess.tokenizer_apply_postprocessor_spec;

// C API entrypoints (FFI bridge).
pub const tokenizer_from_pretrained = c_api.tokenizer_from_pretrained;
pub const tokenizer_from_json_string = c_api.tokenizer_from_json_string;
pub const tokenizer_set_error = c_api.tokenizer_set_error;
pub const tokenizer_added_token_add = c_api.tokenizer_added_token_add;
pub const tokenizer_added_token_find = c_api.tokenizer_added_token_find;
pub const tokenizer_free = c_api.tokenizer_free;
pub const tokenizer_encode_ids = c_api.tokenizer_encode_ids;
pub const tokenizer_encode_ids_slice = c_api.tokenizer_encode_ids_slice;
pub const tokenizer_encode_ids_slice_with_options = c_api.tokenizer_encode_ids_slice_with_options;
pub const tokenizer_decode = c_api.tokenizer_decode;
pub const tokenizer_decode_with_options = c_api.tokenizer_decode_with_options;
pub const tokenizer_tokenize = c_api.tokenizer_tokenize;
pub const tokenizer_string_free = c_api.tokenizer_string_free;
pub const tokenizer_string_free_with_len = c_api.tokenizer_string_free_with_len;
pub const tokenizer_get_last_error = c_api.tokenizer_get_last_error;
