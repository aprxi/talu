//! Padding Utilities for Batched Sequences
//!
//! Functions for calculating max sequence lengths, padding, and mask generation.
//! Used by capi modules to avoid inline loop logic.

const std = @import("std");

/// Result of batch padding operation.
pub const BatchPaddingResult = struct {
    /// Padded 2D data: [num_sequences, padded_len]
    data: []u32,
    /// Number of sequences
    num_sequences: usize,
    /// Padded sequence length
    padded_len: usize,

    pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }

    pub fn shape(self: *const @This()) [2]i64 {
        return .{ @intCast(self.num_sequences), @intCast(self.padded_len) };
    }

    pub fn strides(self: *const @This()) [2]i64 {
        return .{ @intCast(self.padded_len), 1 };
    }
};

/// Result of mask generation operation.
pub const BatchMaskResult = struct {
    /// Mask 2D data: [num_sequences, padded_len], 1=token, 0=padding
    data: []i32,
    /// Number of sequences
    num_sequences: usize,
    /// Padded sequence length
    padded_len: usize,

    pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }

    pub fn shape(self: *const @This()) [2]i64 {
        return .{ @intCast(self.num_sequences), @intCast(self.padded_len) };
    }

    pub fn strides(self: *const @This()) [2]i64 {
        return .{ @intCast(self.padded_len), 1 };
    }
};

/// Calculate the maximum sequence length from CSR offsets.
pub fn maxSequenceLength(offsets: []const usize, num_sequences: usize) usize {
    var max_len: usize = 0;
    for (0..num_sequences) |i| {
        const seq_len = offsets[i + 1] - offsets[i];
        if (seq_len > max_len) max_len = seq_len;
    }
    return max_len;
}

/// Pad sequences from CSR format to a 2D array.
///
/// Args:
///   allocator: Memory allocator
///   ids: Token IDs in CSR data array
///   offsets: CSR offsets (num_sequences + 1 elements)
///   num_sequences: Number of sequences
///   pad_id: Token ID to use for padding
///   max_length: Maximum length (0 = use longest sequence)
///   pad_left: If true, pad on left side
///
/// Returns BatchPaddingResult on success, error on failure.
pub fn padSequences(
    allocator: std.mem.Allocator,
    ids: [*]const u32,
    offsets: [*]const usize,
    num_sequences: usize,
    pad_id: u32,
    max_length: usize,
    pad_left: bool,
) !BatchPaddingResult {
    const offsets_slice = offsets[0 .. num_sequences + 1];

    // Calculate max sequence length
    const max_seq_len = maxSequenceLength(offsets_slice, num_sequences);

    // Determine padded length
    const padded_len = if (max_length > 0) @min(max_length, max_seq_len) else max_seq_len;
    if (padded_len == 0) return error.EmptyBatch;

    const total_elements = num_sequences * padded_len;

    // Allocate and fill with padding
    const padded_data = try allocator.alloc(u32, total_elements);
    errdefer allocator.free(padded_data);
    @memset(padded_data, pad_id);

    // Copy sequences
    for (0..num_sequences) |i| {
        const seq_start = offsets_slice[i];
        const seq_end = offsets_slice[i + 1];
        var seq_len = seq_end - seq_start;
        if (seq_len > padded_len) seq_len = padded_len;

        const row_start = i * padded_len;
        const src = ids[seq_start .. seq_start + seq_len];

        if (pad_left) {
            const dst_start = row_start + (padded_len - seq_len);
            @memcpy(padded_data[dst_start .. dst_start + seq_len], src);
        } else {
            @memcpy(padded_data[row_start .. row_start + seq_len], src);
        }
    }

    return .{
        .data = padded_data,
        .num_sequences = num_sequences,
        .padded_len = padded_len,
    };
}

/// Generate attention mask for padded sequences.
///
/// Args:
///   allocator: Memory allocator
///   offsets: CSR offsets (num_sequences + 1 elements)
///   num_sequences: Number of sequences
///   max_length: Maximum length (0 = use longest sequence)
///   pad_left: If true, pad on left side
///
/// Returns BatchMaskResult on success, error on failure.
pub fn generateMask(
    allocator: std.mem.Allocator,
    offsets: [*]const usize,
    num_sequences: usize,
    max_length: usize,
    pad_left: bool,
) !BatchMaskResult {
    const offsets_slice = offsets[0 .. num_sequences + 1];

    // Calculate max sequence length
    const max_seq_len = maxSequenceLength(offsets_slice, num_sequences);

    // Determine padded length
    const padded_len = if (max_length > 0) @min(max_length, max_seq_len) else max_seq_len;
    if (padded_len == 0) return error.EmptyBatch;

    const total_elements = num_sequences * padded_len;

    // Allocate and fill with zeros
    const mask_data = try allocator.alloc(i32, total_elements);
    errdefer allocator.free(mask_data);
    @memset(mask_data, 0);

    // Set 1s for real tokens
    for (0..num_sequences) |i| {
        const seq_start = offsets_slice[i];
        const seq_end = offsets_slice[i + 1];
        var seq_len = seq_end - seq_start;
        if (seq_len > padded_len) seq_len = padded_len;

        const row_start = i * padded_len;

        if (pad_left) {
            const dst_start = row_start + (padded_len - seq_len);
            for (0..seq_len) |j| {
                mask_data[dst_start + j] = 1;
            }
        } else {
            for (0..seq_len) |j| {
                mask_data[row_start + j] = 1;
            }
        }
    }

    return .{
        .data = mask_data,
        .num_sequences = num_sequences,
        .padded_len = padded_len,
    };
}

// =============================================================================
// Tests
// =============================================================================

test "maxSequenceLength calculates correctly" {
    const offsets = [_]usize{ 0, 3, 5, 10 }; // seq lengths: 3, 2, 5
    const max_len = maxSequenceLength(&offsets, 3);
    try std.testing.expectEqual(@as(usize, 5), max_len);
}

test "padSequences right padding" {
    const allocator = std.testing.allocator;
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const offsets = [_]usize{ 0, 3, 5, 10 }; // seq lengths: 3, 2, 5

    var result = try padSequences(allocator, &ids, &offsets, 3, 0, 0, false);
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), result.num_sequences);
    try std.testing.expectEqual(@as(usize, 5), result.padded_len);

    // First row: [1, 2, 3, 0, 0]
    try std.testing.expectEqual(@as(u32, 1), result.data[0]);
    try std.testing.expectEqual(@as(u32, 2), result.data[1]);
    try std.testing.expectEqual(@as(u32, 3), result.data[2]);
    try std.testing.expectEqual(@as(u32, 0), result.data[3]);
    try std.testing.expectEqual(@as(u32, 0), result.data[4]);
}

test "padSequences left padding" {
    const allocator = std.testing.allocator;
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const offsets = [_]usize{ 0, 3, 5, 10 }; // seq lengths: 3, 2, 5

    var result = try padSequences(allocator, &ids, &offsets, 3, 0, 0, true);
    defer result.deinit(allocator);

    // First row: [0, 0, 1, 2, 3]
    try std.testing.expectEqual(@as(u32, 0), result.data[0]);
    try std.testing.expectEqual(@as(u32, 0), result.data[1]);
    try std.testing.expectEqual(@as(u32, 1), result.data[2]);
    try std.testing.expectEqual(@as(u32, 2), result.data[3]);
    try std.testing.expectEqual(@as(u32, 3), result.data[4]);
}

test "generateMask right padding" {
    const allocator = std.testing.allocator;
    const offsets = [_]usize{ 0, 3, 5, 10 }; // seq lengths: 3, 2, 5

    var result = try generateMask(allocator, &offsets, 3, 0, false);
    defer result.deinit(allocator);

    // First row: [1, 1, 1, 0, 0]
    try std.testing.expectEqual(@as(i32, 1), result.data[0]);
    try std.testing.expectEqual(@as(i32, 1), result.data[1]);
    try std.testing.expectEqual(@as(i32, 1), result.data[2]);
    try std.testing.expectEqual(@as(i32, 0), result.data[3]);
    try std.testing.expectEqual(@as(i32, 0), result.data[4]);
}
