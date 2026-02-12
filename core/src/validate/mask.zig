//! Token mask bit vector.

const std = @import("std");

pub const TokenMask = struct {
    allocator: std.mem.Allocator,
    bits: []u64,
    len: usize,

    pub fn init(allocator: std.mem.Allocator, len: usize) !TokenMask {
        const word_count = (len + 63) / 64;
        const bits = try allocator.alloc(u64, word_count);
        @memset(bits, 0);
        return .{ .allocator = allocator, .bits = bits, .len = len };
    }

    pub fn deinit(self: *TokenMask) void {
        self.allocator.free(self.bits);
        self.* = undefined;
    }

    pub fn setAll(self: *TokenMask) void {
        @memset(self.bits, std.math.maxInt(u64));
        self.clearPadding();
    }

    pub fn clearAll(self: *TokenMask) void {
        @memset(self.bits, 0);
    }

    pub fn set(self: *TokenMask, idx: usize) void {
        if (idx >= self.len) return;
        const word = idx / 64;
        const bit = idx % 64;
        self.bits[word] |= (@as(u64, 1) << @intCast(bit));
    }

    pub fn setValid(self: *TokenMask, idx: usize) void {
        self.set(idx);
    }

    pub fn isSet(self: *const TokenMask, idx: usize) bool {
        if (idx >= self.len) return false;
        const word = idx / 64;
        const bit = idx % 64;
        return (self.bits[word] & (@as(u64, 1) << @intCast(bit))) != 0;
    }

    fn clearPadding(self: *TokenMask) void {
        const extra = self.len % 64;
        if (extra == 0) return;
        const mask = (@as(u64, 1) << @intCast(extra)) - 1;
        self.bits[self.bits.len - 1] &= mask;
    }

    pub fn allValid(allocator: std.mem.Allocator, len: usize) !TokenMask {
        var mask = try init(allocator, len);
        mask.setAll();
        return mask;
    }
};

/// Apply mask to logits - optimized with word-level fast paths.
pub fn applyMask(logits: []f32, mask: *const TokenMask) void {
    const neg_inf = -std.math.inf(f32);

    // Process in chunks of 64 (one word)
    var i: usize = 0;
    while (i < logits.len) {
        const word_idx = i / 64;
        const word = mask.bits[word_idx];
        const chunk_end = @min(i + 64 - (i % 64), logits.len);
        const chunk_start = i;

        // Fast path: if entire word is 0, all tokens in this chunk are invalid
        if (word == 0 and chunk_start % 64 == 0 and chunk_end - chunk_start == 64) {
            @memset(logits[chunk_start..chunk_end], neg_inf);
            i = chunk_end;
            continue;
        }

        // Slow path: check individual bits
        while (i < chunk_end) : (i += 1) {
            if (!mask.isSet(i)) {
                logits[i] = neg_inf;
            }
        }
    }
}

test "mask set and apply" {
    const allocator = std.testing.allocator;

    var mask = try TokenMask.init(allocator, 8);
    defer mask.deinit();

    mask.setValid(2);
    mask.setValid(5);

    var logits = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1 };
    applyMask(&logits, &mask);

    try std.testing.expect(logits[2] == 1);
    try std.testing.expect(logits[5] == 1);
    try std.testing.expect(std.math.isInf(logits[0]));
}

test "mask allValid sets all bits" {
    const allocator = std.testing.allocator;

    var mask = try TokenMask.allValid(allocator, 10);
    defer mask.deinit();

    for (0..10) |idx| {
        try std.testing.expect(mask.isSet(idx));
    }
}
