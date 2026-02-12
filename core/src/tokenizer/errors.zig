//! Tokenizer Error Handling
//!
//! Error message management for tokenizer operations.
//! Stores per-tokenizer error state for FFI retrieval.

const std = @import("std");
const ct = @import("c_types.zig");
const types = @import("types.zig");
const strings = @import("strings.zig");

const Allocator = types.Allocator;

pub fn freeLastError(tok: *ct.Tokenizer) void {
    if (tok.last_error) |ptr| {
        const slice = std.mem.span(@as([*:0]u8, @ptrCast(ptr)));
        Allocator.free(slice);
    }
    tok.last_error = null;
}

pub fn tokenizer_set_error_internal(tok: *ct.Tokenizer, comptime fmt: []const u8, args: anytype) void {
    freeLastError(tok);
    const msg = std.fmt.allocPrint(Allocator, fmt, args) catch return;
    const dup = Allocator.dupeZ(u8, msg) catch {
        Allocator.free(msg);
        return;
    };
    Allocator.free(msg);
    tok.last_error = @ptrCast(dup.ptr);
}

pub fn tokenizer_set_error(tok: ?*ct.Tokenizer, msg: ?[*:0]const u8) void {
    if (tok == null or msg == null) return;
    const tokenizer = tok.?;
    freeLastError(tokenizer);
    const dup = strings.tokenizer_strdup(msg) orelse return;
    tokenizer.last_error = @ptrCast(dup);
}

// =============================================================================
// Tests
// =============================================================================

test "freeLastError clears error" {
    var tok = ct.Tokenizer{
        .model = null,
        .type = .bpe,
        .normalizer = std.mem.zeroes(ct.Normalizer),
        .pretokenizer = std.mem.zeroes(ct.PreTokenizer),
        .postproc = std.mem.zeroes(ct.PostProcessor),
        .decoder = std.mem.zeroes(ct.Decoder),
        .padding = std.mem.zeroes(ct.Padding),
        .truncation = std.mem.zeroes(ct.Truncation),
        .added = null,
        .last_error = null,
    };

    // Set an error
    const error_msg = "test error";
    const dup = strings.tokenizer_strdup(error_msg) orelse unreachable;
    tok.last_error = @ptrCast(dup);

    // Free it
    freeLastError(&tok);
    try std.testing.expect(tok.last_error == null);
}

test "tokenizer_set_error_internal formats message" {
    var tok = ct.Tokenizer{
        .model = null,
        .type = .bpe,
        .normalizer = std.mem.zeroes(ct.Normalizer),
        .pretokenizer = std.mem.zeroes(ct.PreTokenizer),
        .postproc = std.mem.zeroes(ct.PostProcessor),
        .decoder = std.mem.zeroes(ct.Decoder),
        .padding = std.mem.zeroes(ct.Padding),
        .truncation = std.mem.zeroes(ct.Truncation),
        .added = null,
        .last_error = null,
    };
    defer freeLastError(&tok);

    tokenizer_set_error_internal(&tok, "error code: {}", .{42});
    try std.testing.expect(tok.last_error != null);

    const error_slice = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(tok.last_error.?)), 0);
    try std.testing.expectEqualStrings("error code: 42", error_slice);
}

test "tokenizer_set_error copies message" {
    var tok = ct.Tokenizer{
        .model = null,
        .type = .bpe,
        .normalizer = std.mem.zeroes(ct.Normalizer),
        .pretokenizer = std.mem.zeroes(ct.PreTokenizer),
        .postproc = std.mem.zeroes(ct.PostProcessor),
        .decoder = std.mem.zeroes(ct.Decoder),
        .padding = std.mem.zeroes(ct.Padding),
        .truncation = std.mem.zeroes(ct.Truncation),
        .added = null,
        .last_error = null,
    };
    defer freeLastError(&tok);

    const msg: [*:0]const u8 = "test message";
    tokenizer_set_error(&tok, msg);
    try std.testing.expect(tok.last_error != null);

    const error_slice = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(tok.last_error.?)), 0);
    try std.testing.expectEqualStrings("test message", error_slice);
}
