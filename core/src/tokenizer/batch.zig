//! Batch Tokenization Operations
//!
//! Utilities for batch encoding and padded tensor conversion.
//! Used by the C API for efficient multi-sequence processing.

const std = @import("std");
const api_mod = @import("api.zig");
const parallel = @import("../system/parallel.zig");

// =============================================================================
// Batch Encoding
// =============================================================================

/// Result of batch encoding operation (CSR-style layout).
pub const BatchEncodeResult = struct {
    /// Flattened token IDs for all sequences
    ids: []u32,
    /// Offsets into ids array (length = num_sequences + 1)
    offsets: []usize,
    /// Total number of tokens across all sequences
    total_tokens: usize,
    /// Number of sequences
    num_sequences: usize,
    /// Allocator used for allocation
    allocator: std.mem.Allocator,

    pub fn deinit(self: *BatchEncodeResult) void {
        if (self.ids.len > 0) {
            self.allocator.free(self.ids);
        }
        if (self.offsets.len > 0) {
            self.allocator.free(self.offsets);
        }
        self.* = undefined;
    }
};

/// Context for parallel batch encoding.
pub const BatchEncodeContext = struct {
    tokenizer: *api_mod.Tokenizer,
    text_ptrs: [*]const [*]const u8,
    text_lengths: [*]const usize,
    add_special_tokens: bool,
    encoded_batches: [][]u32,
    had_error_flag: std.atomic.Value(bool),

    pub fn init(
        allocator: std.mem.Allocator,
        tokenizer: *api_mod.Tokenizer,
        text_ptrs: [*]const [*]const u8,
        text_lengths: [*]const usize,
        num_texts: usize,
        add_special_tokens: bool,
    ) !*BatchEncodeContext {
        const context = try allocator.create(BatchEncodeContext);
        errdefer allocator.destroy(context);
        const encoded_batches = try allocator.alloc([]u32, num_texts);
        context.* = .{
            .tokenizer = tokenizer,
            .text_ptrs = text_ptrs,
            .text_lengths = text_lengths,
            .add_special_tokens = add_special_tokens,
            .encoded_batches = encoded_batches,
            .had_error_flag = std.atomic.Value(bool).init(false),
        };
        for (context.encoded_batches) |*encoded_ids| {
            encoded_ids.* = &.{};
        }
        return context;
    }

    pub fn deinit(self: *BatchEncodeContext, allocator: std.mem.Allocator) void {
        for (self.encoded_batches) |encoded_ids| {
            if (encoded_ids.len > 0) {
                allocator.free(encoded_ids);
            }
        }
        allocator.free(self.encoded_batches);
        allocator.destroy(self);
    }
};

/// Worker function for parallel batch encoding.
pub fn batchEncodeWorker(start: usize, end: usize, ctx: *BatchEncodeContext) void {
    const encode_opts = api_mod.Tokenizer.EncodeOptions{
        .add_special_tokens = ctx.add_special_tokens,
    };

    for (start..end) |seq_idx| {
        const text_bytes = ctx.text_ptrs[seq_idx][0..ctx.text_lengths[seq_idx]];
        const encoded_ids = ctx.tokenizer.encodeSliceWithOptions(text_bytes, encode_opts) catch {
            ctx.had_error_flag.store(true, .release);
            return;
        };
        ctx.encoded_batches[seq_idx] = encoded_ids;
    }
}

/// Encode multiple texts in parallel.
pub fn encodeBatch(
    allocator: std.mem.Allocator,
    tokenizer: *api_mod.Tokenizer,
    text_ptrs: [*]const [*]const u8,
    text_lengths: [*]const usize,
    num_texts: usize,
    add_special_tokens: bool,
) !BatchEncodeResult {
    if (num_texts == 0) {
        return BatchEncodeResult{
            .ids = &.{},
            .offsets = &.{},
            .total_tokens = 0,
            .num_sequences = 0,
            .allocator = allocator,
        };
    }

    const batch_context = try BatchEncodeContext.init(
        allocator,
        tokenizer,
        text_ptrs,
        text_lengths,
        num_texts,
        add_special_tokens,
    );
    defer batch_context.deinit(allocator);

    const pool = parallel.global();
    pool.parallelFor(num_texts, batchEncodeWorker, batch_context);

    if (batch_context.had_error_flag.load(.acquire)) {
        return error.BatchEncodingFailed;
    }

    var total_token_count: usize = 0;
    for (batch_context.encoded_batches) |encoded_ids| {
        total_token_count += encoded_ids.len;
    }

    const offsets = try allocator.alloc(usize, num_texts + 1);
    errdefer allocator.free(offsets);

    if (total_token_count == 0) {
        for (offsets) |*off| {
            off.* = 0;
        }
        return BatchEncodeResult{
            .ids = &.{},
            .offsets = offsets,
            .total_tokens = 0,
            .num_sequences = num_texts,
            .allocator = allocator,
        };
    }

    const ids = try allocator.alloc(u32, total_token_count);

    var write_offset: usize = 0;
    for (batch_context.encoded_batches, 0..) |encoded_ids, seq_idx| {
        offsets[seq_idx] = write_offset;
        if (encoded_ids.len > 0) {
            @memcpy(ids[write_offset .. write_offset + encoded_ids.len], encoded_ids);
            write_offset += encoded_ids.len;
        }
    }
    offsets[num_texts] = total_token_count;

    return BatchEncodeResult{
        .ids = ids,
        .offsets = offsets,
        .total_tokens = total_token_count,
        .num_sequences = num_texts,
        .allocator = allocator,
    };
}

// =============================================================================
// Padded Tensor Conversion
// =============================================================================

/// Padding side options.
pub const PaddingSide = enum(u8) {
    right = 0,
    left = 1,
};

/// Options for padded tensor conversion.
pub const PaddedTensorOptions = struct {
    pad_id: u32 = 0,
    padding_side: PaddingSide = .right,
    max_length: usize = 0,
    truncate: bool = false,
    return_attention_mask: bool = true,
};

/// Result from padded tensor conversion.
pub const PaddedTensorResult = struct {
    input_ids: []u32,
    attention_mask: ?[]u32,
    num_sequences: usize,
    padded_length: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *PaddedTensorResult) void {
        const total_size = self.num_sequences * self.padded_length;
        if (total_size > 0) {
            self.allocator.free(self.input_ids);
            if (self.attention_mask) |mask| {
                self.allocator.free(mask);
            }
        }
        self.* = undefined;
    }
};

/// Convert batch encoding to padded tensors.
pub fn batchToPaddedTensor(
    allocator: std.mem.Allocator,
    ids: []const u32,
    offsets: []const usize,
    num_sequences: usize,
    options: PaddedTensorOptions,
) !PaddedTensorResult {
    if (num_sequences == 0) {
        return PaddedTensorResult{
            .input_ids = &.{},
            .attention_mask = null,
            .num_sequences = 0,
            .padded_length = 0,
            .allocator = allocator,
        };
    }

    // Find max sequence length
    var max_sequence_len: usize = 0;
    for (0..num_sequences) |seq_idx| {
        const seq_len = offsets[seq_idx + 1] - offsets[seq_idx];
        if (seq_len > max_sequence_len) max_sequence_len = seq_len;
    }

    // Determine padded length
    var padded_length = if (options.max_length > 0) options.max_length else max_sequence_len;
    if (options.truncate and options.max_length > 0) {
        padded_length = options.max_length;
    } else if (options.max_length > 0 and max_sequence_len > options.max_length) {
        padded_length = max_sequence_len;
    }

    const total_elements = num_sequences * padded_length;
    const input_ids_buffer = try allocator.alloc(u32, total_elements);
    errdefer allocator.free(input_ids_buffer);

    const attention_mask_buffer = if (options.return_attention_mask)
        try allocator.alloc(u32, total_elements)
    else
        null;
    errdefer if (attention_mask_buffer) |buf| allocator.free(buf);

    const pad_left = options.padding_side == .left;

    for (0..num_sequences) |seq_idx| {
        const sequence_start = offsets[seq_idx];
        const sequence_end = offsets[seq_idx + 1];
        var sequence_len = sequence_end - sequence_start;

        if (options.truncate and sequence_len > padded_length) {
            sequence_len = padded_length;
        }

        const pad_count = if (padded_length > sequence_len) padded_length - sequence_len else 0;
        const row_offset = seq_idx * padded_length;

        if (pad_left) {
            // Padding on left side
            for (0..pad_count) |elem_idx| {
                input_ids_buffer[row_offset + elem_idx] = options.pad_id;
            }
            for (0..sequence_len) |elem_idx| {
                input_ids_buffer[row_offset + pad_count + elem_idx] = ids[sequence_start + elem_idx];
            }
            if (attention_mask_buffer) |mask| {
                for (0..pad_count) |elem_idx| {
                    mask[row_offset + elem_idx] = 0;
                }
                for (0..sequence_len) |elem_idx| {
                    mask[row_offset + pad_count + elem_idx] = 1;
                }
            }
        } else {
            // Padding on right side
            for (0..sequence_len) |elem_idx| {
                input_ids_buffer[row_offset + elem_idx] = ids[sequence_start + elem_idx];
            }
            for (0..pad_count) |elem_idx| {
                input_ids_buffer[row_offset + sequence_len + elem_idx] = options.pad_id;
            }
            if (attention_mask_buffer) |mask| {
                for (0..sequence_len) |elem_idx| {
                    mask[row_offset + elem_idx] = 1;
                }
                for (0..pad_count) |elem_idx| {
                    mask[row_offset + sequence_len + elem_idx] = 0;
                }
            }
        }
    }

    return PaddedTensorResult{
        .input_ids = input_ids_buffer,
        .attention_mask = attention_mask_buffer,
        .num_sequences = num_sequences,
        .padded_length = padded_length,
        .allocator = allocator,
    };
}

// =============================================================================
// Tests
// =============================================================================

test "PaddingSide enum values" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(PaddingSide.right));
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(PaddingSide.left));
}

test "PaddedTensorOptions defaults" {
    const opts = PaddedTensorOptions{};
    try std.testing.expectEqual(@as(u32, 0), opts.pad_id);
    try std.testing.expectEqual(PaddingSide.right, opts.padding_side);
    try std.testing.expectEqual(@as(usize, 0), opts.max_length);
    try std.testing.expectEqual(false, opts.truncate);
    try std.testing.expectEqual(true, opts.return_attention_mask);
}

test "batchToPaddedTensor handles empty input" {
    var result = try batchToPaddedTensor(
        std.testing.allocator,
        &.{},
        &.{},
        0,
        .{},
    );
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 0), result.num_sequences);
    try std.testing.expectEqual(@as(usize, 0), result.padded_length);
}

test "batchToPaddedTensor pads right" {
    const ids = [_]u32{ 1, 2, 3, 4, 5 };
    const offsets = [_]usize{ 0, 3, 5 }; // [1,2,3] and [4,5]

    var result = try batchToPaddedTensor(
        std.testing.allocator,
        &ids,
        &offsets,
        2,
        .{ .pad_id = 0 },
    );
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 2), result.num_sequences);
    try std.testing.expectEqual(@as(usize, 3), result.padded_length);

    // First sequence: [1, 2, 3]
    try std.testing.expectEqual(@as(u32, 1), result.input_ids[0]);
    try std.testing.expectEqual(@as(u32, 2), result.input_ids[1]);
    try std.testing.expectEqual(@as(u32, 3), result.input_ids[2]);

    // Second sequence: [4, 5, 0] (padded right)
    try std.testing.expectEqual(@as(u32, 4), result.input_ids[3]);
    try std.testing.expectEqual(@as(u32, 5), result.input_ids[4]);
    try std.testing.expectEqual(@as(u32, 0), result.input_ids[5]);

    // Attention mask
    const mask = result.attention_mask.?;
    try std.testing.expectEqual(@as(u32, 1), mask[0]);
    try std.testing.expectEqual(@as(u32, 1), mask[1]);
    try std.testing.expectEqual(@as(u32, 1), mask[2]);
    try std.testing.expectEqual(@as(u32, 1), mask[3]);
    try std.testing.expectEqual(@as(u32, 1), mask[4]);
    try std.testing.expectEqual(@as(u32, 0), mask[5]);
}

test "batchToPaddedTensor pads left" {
    const ids = [_]u32{ 1, 2, 3, 4, 5 };
    const offsets = [_]usize{ 0, 3, 5 }; // [1,2,3] and [4,5]

    var result = try batchToPaddedTensor(
        std.testing.allocator,
        &ids,
        &offsets,
        2,
        .{ .pad_id = 0, .padding_side = .left },
    );
    defer result.deinit();

    // First sequence: [1, 2, 3]
    try std.testing.expectEqual(@as(u32, 1), result.input_ids[0]);
    try std.testing.expectEqual(@as(u32, 2), result.input_ids[1]);
    try std.testing.expectEqual(@as(u32, 3), result.input_ids[2]);

    // Second sequence: [0, 4, 5] (padded left)
    try std.testing.expectEqual(@as(u32, 0), result.input_ids[3]);
    try std.testing.expectEqual(@as(u32, 4), result.input_ids[4]);
    try std.testing.expectEqual(@as(u32, 5), result.input_ids[5]);

    // Attention mask
    const mask = result.attention_mask.?;
    try std.testing.expectEqual(@as(u32, 0), mask[3]);
    try std.testing.expectEqual(@as(u32, 1), mask[4]);
    try std.testing.expectEqual(@as(u32, 1), mask[5]);
}

test "batchToPaddedTensor truncates" {
    const ids = [_]u32{ 1, 2, 3, 4, 5 };
    const offsets = [_]usize{ 0, 3, 5 }; // [1,2,3] and [4,5]

    var result = try batchToPaddedTensor(
        std.testing.allocator,
        &ids,
        &offsets,
        2,
        .{ .pad_id = 0, .max_length = 2, .truncate = true },
    );
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 2), result.padded_length);

    // First sequence truncated: [1, 2]
    try std.testing.expectEqual(@as(u32, 1), result.input_ids[0]);
    try std.testing.expectEqual(@as(u32, 2), result.input_ids[1]);

    // Second sequence: [4, 5]
    try std.testing.expectEqual(@as(u32, 4), result.input_ids[2]);
    try std.testing.expectEqual(@as(u32, 5), result.input_ids[3]);
}

test "batchToPaddedTensor without attention mask" {
    const ids = [_]u32{ 1, 2, 3 };
    const offsets = [_]usize{ 0, 3 };

    var result = try batchToPaddedTensor(
        std.testing.allocator,
        &ids,
        &offsets,
        1,
        .{ .return_attention_mask = false },
    );
    defer result.deinit();

    try std.testing.expect(result.attention_mask == null);
}
