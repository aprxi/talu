//! Tokenizer Integration Tests
//!
//! Integration tests for the tokenizer public API.
//! Tests types exported from core/src/tokenizer/root.zig.

const std = @import("std");

// Tokenizer types
pub const tokenizer = @import("tokenizer_test.zig");
pub const tokenizer_handle = @import("tokenizer_handle_test.zig");
pub const streaming_decoder = @import("streaming_decoder_test.zig");
pub const c_tokenizer = @import("c_tokenizer_test.zig");

// Result types
pub const vocab_result = @import("vocab_result_test.zig");
pub const vocab_list_c = @import("vocab_list_c_test.zig");
pub const tokenize_bytes_result = @import("tokenize_bytes_result_test.zig");
pub const offsets_result = @import("offsets_result_test.zig");
pub const encoding = @import("encoding_test.zig");
pub const batch_encode_result = @import("batch_encode_result_test.zig");
pub const batch_encode_context = @import("batch_encode_context_test.zig");
pub const padded_tensor_result = @import("padded_tensor_result_test.zig");

test {
    std.testing.refAllDecls(@This());
}
