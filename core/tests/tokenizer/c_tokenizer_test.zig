//! Integration tests for tokenizer.CTokenizer
//!
//! Tests the C ABI tokenizer type exported from tokenizer/root.zig.

const std = @import("std");
const main = @import("main");
const CTokenizer = main.tokenizer.CTokenizer;

test "CTokenizer type is accessible" {
    // Basic type accessibility test
    // CTokenizer is an opaque type used for C FFI, so we just verify it exists
    const type_info = @typeInfo(CTokenizer);
    try std.testing.expect(type_info == .@"struct" or type_info == .@"opaque");
}
