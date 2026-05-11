//! Encoding accumulator for tokenizer encode paths.

const std = @import("std");
const ct = @import("c_types.zig");
const buffers = @import("encoding_buffers.zig");
const types = @import("types.zig");

const Allocator = types.Allocator;

/// Builds TokenizerEncoding arrays from token IDs.
///
/// Token strings are resolved lazily by consumers that need them. Special-token
/// flags use sparse positions so normal BPE word encoding does not append a
/// zero mask entry for every token.
pub const EncodeAccum = struct {
    ids: std.ArrayListUnmanaged(i32) = .{},
    special_positions: std.ArrayListUnmanaged(u32) = .{},

    pub fn deinit(self: *EncodeAccum) void {
        self.special_positions.deinit(Allocator);
        self.ids.deinit(Allocator);
        self.* = .{};
    }

    pub fn appendAdded(self: *EncodeAccum, added_token: *const ct.AddedToken) !void {
        if (added_token.special != 0) {
            try self.special_positions.append(Allocator, @intCast(self.ids.items.len));
        }
        try self.ids.append(Allocator, added_token.id);
    }

    pub fn appendCachedIds(self: *EncodeAccum, cached_ids: []const i32) !void {
        try self.ids.appendSlice(Allocator, cached_ids);
    }

    pub fn appendEncoding(self: *EncodeAccum, encoding: *const ct.TokenizerEncoding, added_head: ?*ct.AddedToken) !void {
        if (encoding.ids == null) return;
        const ids_ptr: [*]i32 = @ptrCast(encoding.ids.?);

        for (0..encoding.ids_len) |id_index| {
            if (isSpecialAddedToken(added_head, ids_ptr[id_index])) {
                try self.special_positions.append(Allocator, @intCast(self.ids.items.len));
            }
            try self.ids.append(Allocator, ids_ptr[id_index]);
        }
    }

    pub fn buildOutput(self: *EncodeAccum, out_encoding: *ct.TokenizerEncoding) !void {
        const token_count = self.ids.items.len;
        if (token_count == 0) {
            out_encoding.* = std.mem.zeroes(ct.TokenizerEncoding);
            return;
        }

        var buffers_out = buffers.Buffers{
            .ids = try self.ids.toOwnedSlice(Allocator),
            .tokens = &.{},
            .attention_mask = &.{},
            .type_ids = &.{},
            .special = &.{},
            .offsets = &.{},
        };
        errdefer buffers_out.deinit();

        buffers_out.tokens = try Allocator.alloc([*c]u8, token_count);
        @memset(buffers_out.tokens, null);
        buffers_out.attention_mask = try Allocator.alloc(i32, token_count);
        @memset(buffers_out.attention_mask, 1);
        buffers_out.type_ids = try Allocator.alloc(i32, token_count);
        @memset(buffers_out.type_ids, 0);
        buffers_out.special = try Allocator.alloc(i32, token_count);
        @memset(buffers_out.special, 0);
        for (self.special_positions.items) |pos| {
            buffers_out.special[pos] = 1;
        }
        buffers_out.offsets = try Allocator.alloc(ct.Offset, token_count);
        @memset(std.mem.sliceAsBytes(buffers_out.offsets), 0);

        buffers.initEncoding(out_encoding, &buffers_out, token_count, null, 0);
        self.special_positions.deinit(Allocator);
        self.* = .{};
    }
};

fn isSpecialAddedToken(added_head: ?*ct.AddedToken, id: i32) bool {
    var added_iter = added_head;
    while (added_iter) |token| : (added_iter = token.next) {
        if (token.special != 0 and token.id == id) return true;
    }
    return false;
}

fn testAddedToken(id: i32, special: i32, next: ?*ct.AddedToken) ct.AddedToken {
    return .{
        .content = null,
        .id = id,
        .special = special,
        .single_word = 0,
        .lstrip = 0,
        .rstrip = 0,
        .normalized = 0,
        .next = next,
    };
}

test "EncodeAccum deinit resets owned buffers" {
    var accum = EncodeAccum{};
    try accum.appendCachedIds(&[_]i32{ 1, 2 });
    try accum.special_positions.append(Allocator, 0);

    accum.deinit();

    try std.testing.expectEqual(@as(usize, 0), accum.ids.items.len);
    try std.testing.expectEqual(@as(usize, 0), accum.special_positions.items.len);
}

test "EncodeAccum appendAdded records special token position" {
    var accum = EncodeAccum{};
    defer accum.deinit();
    var added = testAddedToken(123, 1, null);

    try accum.appendAdded(&added);

    try std.testing.expectEqualSlices(i32, &[_]i32{123}, accum.ids.items);
    try std.testing.expectEqualSlices(u32, &[_]u32{0}, accum.special_positions.items);
}

test "EncodeAccum appendCachedIds appends non-special ids" {
    var accum = EncodeAccum{};
    defer accum.deinit();

    try accum.appendCachedIds(&[_]i32{ 4, 5, 6 });

    try std.testing.expectEqualSlices(i32, &[_]i32{ 4, 5, 6 }, accum.ids.items);
    try std.testing.expectEqual(@as(usize, 0), accum.special_positions.items.len);
}

test "EncodeAccum appendEncoding copies ids and marks special added tokens" {
    var normal_added = testAddedToken(7, 0, null);
    var special_added = testAddedToken(42, 1, &normal_added);
    var ids = [_]i32{ 1, 42, 7, 42 };
    var encoding = ct.TokenizerEncoding{
        .ids = @ptrCast(&ids),
        .ids_len = ids.len,
        .tokens = null,
        .tokens_len = 0,
        .attention_mask = null,
        .type_ids = null,
        .special_tokens_mask = null,
        .offsets = null,
        .overflows = null,
        .overflow_count = 0,
    };
    var accum = EncodeAccum{};
    defer accum.deinit();

    try accum.appendEncoding(&encoding, &special_added);

    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 42, 7, 42 }, accum.ids.items);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 1, 3 }, accum.special_positions.items);
}

test "EncodeAccum buildOutput transfers ids and initializes arrays" {
    var accum = EncodeAccum{};
    defer accum.deinit();
    try accum.appendCachedIds(&[_]i32{ 7, 8, 9 });
    try accum.special_positions.append(Allocator, 1);

    var output = std.mem.zeroes(ct.TokenizerEncoding);
    try accum.buildOutput(&output);
    defer buffers.freeEncodingArrays(&output);

    try std.testing.expectEqual(@as(usize, 0), accum.ids.items.len);
    try std.testing.expectEqual(@as(usize, 0), accum.special_positions.items.len);
    try std.testing.expectEqual(@as(usize, 3), output.ids_len);
    try std.testing.expectEqual(@as(usize, 3), output.tokens_len);

    const ids: [*]i32 = @ptrCast(output.ids.?);
    const tokens: [*][*c]u8 = @ptrCast(output.tokens.?);
    const attention_mask: [*]i32 = @ptrCast(output.attention_mask.?);
    const type_ids: [*]i32 = @ptrCast(output.type_ids.?);
    const special: [*]i32 = @ptrCast(output.special_tokens_mask.?);
    const offsets: [*]ct.Offset = @ptrCast(output.offsets.?);

    try std.testing.expectEqualSlices(i32, &[_]i32{ 7, 8, 9 }, ids[0..3]);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 1 }, attention_mask[0..3]);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 0, 0, 0 }, type_ids[0..3]);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 0, 1, 0 }, special[0..3]);
    for (0..3) |idx| {
        try std.testing.expect(tokens[idx] == null);
        try std.testing.expectEqual(@as(i32, 0), offsets[idx].start);
        try std.testing.expectEqual(@as(i32, 0), offsets[idx].end);
    }
}

test "EncodeAccum buildOutput zeroes empty output" {
    var accum = EncodeAccum{};
    var output = std.mem.zeroes(ct.TokenizerEncoding);
    output.ids_len = 99;
    output.tokens_len = 99;

    try accum.buildOutput(&output);

    try std.testing.expect(output.ids == null);
    try std.testing.expect(output.tokens == null);
    try std.testing.expect(output.attention_mask == null);
    try std.testing.expect(output.type_ids == null);
    try std.testing.expect(output.special_tokens_mask == null);
    try std.testing.expect(output.offsets == null);
    try std.testing.expectEqual(@as(usize, 0), output.ids_len);
    try std.testing.expectEqual(@as(usize, 0), output.tokens_len);
}
