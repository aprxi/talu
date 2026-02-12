//! Encoding Buffers
//!
//! Pre-allocated buffer management for tokenizer encoding results.
//! Handles IDs, tokens, attention masks, type IDs, and offsets.

const std = @import("std");
const ct = @import("c_types.zig");
const types = @import("types.zig");
const strings = @import("strings.zig");

const Allocator = types.Allocator;

pub const Buffers = struct {
    ids: []i32,
    tokens: [][*c]u8,
    attention_mask: []i32,
    type_ids: []i32,
    special: []i32,
    offsets: []ct.Offset,

    pub fn deinit(self: *Buffers) void {
        if (self.ids.len > 0) Allocator.free(self.ids);
        if (self.tokens.len > 0) Allocator.free(self.tokens);
        if (self.attention_mask.len > 0) Allocator.free(self.attention_mask);
        if (self.type_ids.len > 0) Allocator.free(self.type_ids);
        if (self.special.len > 0) Allocator.free(self.special);
        if (self.offsets.len > 0) Allocator.free(self.offsets);
        self.* = .{
            .ids = &.{},
            .tokens = &.{},
            .attention_mask = &.{},
            .type_ids = &.{},
            .special = &.{},
            .offsets = &.{},
        };
    }
};

pub fn allocBuffers(buffer_len: usize) !Buffers {
    var buffers_out: Buffers = .{
        .ids = &.{},
        .tokens = &.{},
        .attention_mask = &.{},
        .type_ids = &.{},
        .special = &.{},
        .offsets = &.{},
    };
    buffers_out.ids = try Allocator.alloc(i32, buffer_len);
    errdefer buffers_out.deinit();
    buffers_out.tokens = try Allocator.alloc([*c]u8, buffer_len);
    // Initialize tokens to null so freeEncodingArrays doesn't try to free garbage pointers
    @memset(buffers_out.tokens, null);
    buffers_out.attention_mask = try Allocator.alloc(i32, buffer_len);
    buffers_out.type_ids = try Allocator.alloc(i32, buffer_len);
    buffers_out.special = try Allocator.alloc(i32, buffer_len);
    buffers_out.offsets = try Allocator.alloc(ct.Offset, buffer_len);
    return buffers_out;
}

pub fn fillFromEncoding(
    buffers_out: *Buffers,
    write_index: *usize,
    source_encoding: *ct.TokenizerEncoding,
    type_id: i32,
    default_offset: ct.Offset,
) void {
    if (source_encoding.ids == null) return;
    const id_values: [*]i32 = @ptrCast(source_encoding.ids.?);
    const token_ptrs: ?[*][*c]u8 = if (source_encoding.tokens) |t| @ptrCast(t) else null;
    const mask_values: ?[*]i32 = if (source_encoding.attention_mask) |m| @ptrCast(m) else null;
    const special_values: ?[*]i32 = if (source_encoding.special_tokens_mask) |s| @ptrCast(s) else null;
    const offset_values: ?[*]ct.Offset = if (source_encoding.offsets) |o| @ptrCast(o) else null;

    for (0..source_encoding.ids_len) |source_index| {
        buffers_out.ids[write_index.*] = id_values[source_index];
        // Dup token strings so the source encoding can be freed independently.
        // Without this, freeEncodingArrays on the source would invalidate our
        // pointers, causing a double-free when the output encoding is freed.
        buffers_out.tokens[write_index.*] = if (token_ptrs) |ts| blk: {
            if (ts[source_index]) |src_ptr| {
                const src_str: [*:0]const u8 = @ptrCast(src_ptr);
                break :blk @ptrCast(strings.tokenizer_strdup(src_str) orelse null);
            }
            break :blk null;
        } else null;
        buffers_out.attention_mask[write_index.*] = if (mask_values) |ms| ms[source_index] else 1;
        buffers_out.type_ids[write_index.*] = type_id;
        buffers_out.special[write_index.*] = if (special_values) |ss| ss[source_index] else 0;
        buffers_out.offsets[write_index.*] = if (offset_values) |os| os[source_index] else default_offset;
        write_index.* += 1;
    }
}

pub fn initEncoding(
    encoding: *ct.TokenizerEncoding,
    buffers_out: *Buffers,
    encoding_len: usize,
    overflow_ptr: ?*ct.TokenizerEncoding,
    overflow_len: usize,
) void {
    encoding.* = .{
        .ids = @ptrCast(buffers_out.ids.ptr),
        .ids_len = encoding_len,
        .tokens = @ptrCast(buffers_out.tokens.ptr),
        .tokens_len = encoding_len,
        .attention_mask = @ptrCast(buffers_out.attention_mask.ptr),
        .type_ids = @ptrCast(buffers_out.type_ids.ptr),
        .special_tokens_mask = @ptrCast(buffers_out.special.ptr),
        .offsets = @ptrCast(buffers_out.offsets.ptr),
        .overflows = overflow_ptr,
        .overflow_count = overflow_len,
    };
    buffers_out.* = .{
        .ids = &.{},
        .tokens = &.{},
        .attention_mask = &.{},
        .type_ids = &.{},
        .special = &.{},
        .offsets = &.{},
    };
}

pub fn freeEncodingArrays(encoding: *ct.TokenizerEncoding) void {
    if (encoding.tokens) |tokens_ptr| {
        const token_ptrs: [*][*c]u8 = @ptrCast(tokens_ptr);
        for (0..encoding.tokens_len) |token_idx| {
            if (token_ptrs[token_idx]) |token_ptr| Allocator.free(std.mem.span(@as([*:0]u8, @ptrCast(token_ptr))));
        }
        Allocator.free(token_ptrs[0..encoding.tokens_len]);
    }
    if (encoding.attention_mask) |mask_ptr| {
        const slice: [*]i32 = @ptrCast(mask_ptr);
        Allocator.free(slice[0..encoding.ids_len]);
    }
    if (encoding.type_ids) |type_ids_ptr| {
        const slice: [*]i32 = @ptrCast(type_ids_ptr);
        Allocator.free(slice[0..encoding.ids_len]);
    }
    if (encoding.special_tokens_mask) |special_ptr| {
        const slice: [*]i32 = @ptrCast(special_ptr);
        Allocator.free(slice[0..encoding.ids_len]);
    }
    if (encoding.offsets) |offsets_ptr| {
        const slice: [*]ct.Offset = @ptrCast(offsets_ptr);
        Allocator.free(slice[0..encoding.ids_len]);
    }
    if (encoding.ids) |ids_ptr| {
        const slice: [*]i32 = @ptrCast(ids_ptr);
        Allocator.free(slice[0..encoding.ids_len]);
    }
    encoding.ids = null;
    encoding.tokens = null;
    encoding.attention_mask = null;
    encoding.type_ids = null;
    encoding.special_tokens_mask = null;
    encoding.offsets = null;
    encoding.ids_len = 0;
    encoding.tokens_len = 0;
}

// =============================================================================
// Tests
// =============================================================================

test "allocBuffers creates all buffers" {
    var buffers = try allocBuffers(10);
    defer buffers.deinit();

    try std.testing.expectEqual(@as(usize, 10), buffers.ids.len);
    try std.testing.expectEqual(@as(usize, 10), buffers.tokens.len);
    try std.testing.expectEqual(@as(usize, 10), buffers.attention_mask.len);
    try std.testing.expectEqual(@as(usize, 10), buffers.type_ids.len);
    try std.testing.expectEqual(@as(usize, 10), buffers.special.len);
    try std.testing.expectEqual(@as(usize, 10), buffers.offsets.len);
}

test "fillFromEncoding copies data correctly" {
    var buffers = try allocBuffers(5);
    defer buffers.deinit();

    // Create a source encoding with test data
    var ids = [_]i32{ 1, 2, 3 };
    const token1 = strings.dupTokenString("tok1") orelse unreachable;
    const token2 = strings.dupTokenString("tok2") orelse unreachable;
    const token3 = strings.dupTokenString("tok3") orelse unreachable;
    defer Allocator.free(std.mem.sliceTo(token1, 0));
    defer Allocator.free(std.mem.sliceTo(token2, 0));
    defer Allocator.free(std.mem.sliceTo(token3, 0));

    var token_ptrs = [_][*c]u8{ @ptrCast(token1), @ptrCast(token2), @ptrCast(token3) };
    var source_encoding = ct.TokenizerEncoding{
        .ids = @ptrCast(&ids),
        .ids_len = 3,
        .tokens = @ptrCast(&token_ptrs),
        .tokens_len = 3,
        .attention_mask = null,
        .type_ids = null,
        .special_tokens_mask = null,
        .offsets = null,
        .overflows = null,
        .overflow_count = 0,
    };

    var write_index: usize = 0;
    fillFromEncoding(&buffers, &write_index, &source_encoding, 0, .{ .start = -1, .end = -1 });

    try std.testing.expectEqual(@as(usize, 3), write_index);
    try std.testing.expectEqual(@as(i32, 1), buffers.ids[0]);
    try std.testing.expectEqual(@as(i32, 2), buffers.ids[1]);
    try std.testing.expectEqual(@as(i32, 3), buffers.ids[2]);
}

test "initEncoding sets up encoding structure" {
    var buffers = try allocBuffers(3);
    // Don't defer deinit - initEncoding transfers ownership

    buffers.ids[0] = 1;
    buffers.ids[1] = 2;
    buffers.ids[2] = 3;

    var encoding = std.mem.zeroes(ct.TokenizerEncoding);
    initEncoding(&encoding, &buffers, 3, null, 0);

    // Verify encoding was set up
    try std.testing.expectEqual(@as(usize, 3), encoding.ids_len);
    try std.testing.expect(encoding.ids != null);

    // Verify buffers were cleared (ownership transferred)
    try std.testing.expectEqual(@as(usize, 0), buffers.ids.len);

    // Clean up the encoding
    freeEncodingArrays(&encoding);
}

test "freeEncodingArrays cleans up all arrays" {
    var buffers = try allocBuffers(2);

    buffers.ids[0] = 100;
    buffers.ids[1] = 200;
    buffers.tokens[0] = @ptrCast(strings.dupTokenString("test") orelse unreachable);
    buffers.tokens[1] = null;

    var encoding = std.mem.zeroes(ct.TokenizerEncoding);
    initEncoding(&encoding, &buffers, 2, null, 0);

    // Free it
    freeEncodingArrays(&encoding);

    try std.testing.expect(encoding.ids == null);
    try std.testing.expect(encoding.tokens == null);
    try std.testing.expectEqual(@as(usize, 0), encoding.ids_len);
}

test "deinit frees allocated buffers" {
    var buffers = try allocBuffers(5);

    // Verify buffers were allocated
    try std.testing.expectEqual(@as(usize, 5), buffers.ids.len);
    try std.testing.expectEqual(@as(usize, 5), buffers.tokens.len);
    try std.testing.expectEqual(@as(usize, 5), buffers.attention_mask.len);
    try std.testing.expectEqual(@as(usize, 5), buffers.type_ids.len);
    try std.testing.expectEqual(@as(usize, 5), buffers.special.len);
    try std.testing.expectEqual(@as(usize, 5), buffers.offsets.len);

    // Call deinit
    buffers.deinit();

    // Verify all buffers were freed and reset to empty slices
    try std.testing.expectEqual(@as(usize, 0), buffers.ids.len);
    try std.testing.expectEqual(@as(usize, 0), buffers.tokens.len);
    try std.testing.expectEqual(@as(usize, 0), buffers.attention_mask.len);
    try std.testing.expectEqual(@as(usize, 0), buffers.type_ids.len);
    try std.testing.expectEqual(@as(usize, 0), buffers.special.len);
    try std.testing.expectEqual(@as(usize, 0), buffers.offsets.len);
}
