//! Shared helpers for tokenizer model backends.

const std = @import("std");
const ct = @import("c_types.zig");
const tok_fns = @import("pipeline.zig");

pub fn isAddedSpecialToken(tokenizer: *ct.Tokenizer, token_id: i32) bool {
    var cursor = tokenizer.added;
    while (cursor) |node| {
        if (node.id == token_id and node.special != 0) return true;
        cursor = node.next;
    }
    return false;
}

pub fn attachRegexPretokenizer(tokenizer: *ct.Tokenizer, pattern: [:0]const u8, error_message: [*:0]const u8) !void {
    if (tok_fns.tokenizer_pretokenizer_set(&tokenizer.pretokenizer, pattern.ptr) != 0) {
        tok_fns.tokenizer_set_error(tokenizer, error_message);
        return error.PretokenizerInitFailed;
    }
}

pub fn allocIdToToken(allocator: std.mem.Allocator, size: usize) ![]?[*:0]u8 {
    const id_to_token = try allocator.alloc(?[*:0]u8, size);
    @memset(id_to_token, null);
    return id_to_token;
}

pub fn setFixedString(dst: []u8, text: []const u8) void {
    @memset(dst, 0);
    const copy_len = @min(dst.len - 1, text.len);
    @memcpy(dst[0..copy_len], text[0..copy_len]);
    dst[copy_len] = 0;
}
