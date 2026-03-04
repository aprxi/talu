//! Training data: batch sampling from tokenized data.
//!
//! Provides sequential batch iteration over pre-tokenized data with
//! input_ids and targets (shifted by 1 position for next-token prediction).

const std = @import("std");
const Allocator = std.mem.Allocator;

/// A single training batch.
pub const Batch = struct {
    /// Input token IDs: [batch_size * seq_len].
    input_ids: []const u32,
    /// Target token IDs: [batch_size * seq_len] (shifted by 1 from source).
    targets: []const u32,
    /// Number of sequences in this batch.
    batch_size: usize,
    /// Sequence length per batch element.
    seq_len: usize,
};

/// Sequential batch sampler from a flat token array.
///
/// Given tokens [t0, t1, t2, ...], produces batches where:
///   input_ids  = [t_i, t_i+1, ..., t_i+seq_len-1]
///   targets    = [t_i+1, t_i+2, ..., t_i+seq_len]
pub const DataLoader = struct {
    /// The full tokenized dataset.
    tokens: []const u32,
    /// Sequence length per sample.
    seq_len: usize,
    /// Batch size.
    batch_size: usize,
    /// Current position in the token stream.
    cursor: usize,

    /// Pre-allocated batch buffers.
    input_buf: []u32,
    target_buf: []u32,
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        tokens: []const u32,
        batch_size: usize,
        seq_len: usize,
    ) !DataLoader {
        const buf_size = batch_size * seq_len;
        const input_buf = try allocator.alloc(u32, buf_size);
        errdefer allocator.free(input_buf);
        const target_buf = try allocator.alloc(u32, buf_size);

        return .{
            .tokens = tokens,
            .seq_len = seq_len,
            .batch_size = batch_size,
            .cursor = 0,
            .input_buf = input_buf,
            .target_buf = target_buf,
            .allocator = allocator,
        };
    }

    /// Get the next batch. Returns null when data is exhausted.
    /// Call reset() to start a new epoch.
    pub fn nextBatch(self: *DataLoader) ?Batch {
        const stride = self.seq_len + 1; // +1 for the target at the end
        const total_needed = self.batch_size * stride;

        if (self.cursor + total_needed > self.tokens.len) {
            return null;
        }

        for (0..self.batch_size) |b| {
            const start = self.cursor + b * stride;
            const src = self.tokens[start..][0..stride];
            const in_dest = self.input_buf[b * self.seq_len ..][0..self.seq_len];
            const tgt_dest = self.target_buf[b * self.seq_len ..][0..self.seq_len];

            @memcpy(in_dest, src[0..self.seq_len]);
            @memcpy(tgt_dest, src[1..stride]);
        }

        self.cursor += total_needed;

        return .{
            .input_ids = self.input_buf,
            .targets = self.target_buf,
            .batch_size = self.batch_size,
            .seq_len = self.seq_len,
        };
    }

    /// Reset to the beginning for a new epoch.
    pub fn reset(self: *DataLoader) void {
        self.cursor = 0;
    }

    /// Number of complete batches available.
    pub fn numBatches(self: *const DataLoader) usize {
        const stride = self.seq_len + 1;
        const total_needed = self.batch_size * stride;
        if (total_needed == 0) return 0;
        return self.tokens.len / total_needed;
    }

    pub fn deinit(self: *DataLoader) void {
        self.allocator.free(self.input_buf);
        self.allocator.free(self.target_buf);
        self.* = undefined;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "DataLoader produces correct input/target pairs" {
    const allocator = std.testing.allocator;
    // tokens: 0,1,2,3,4,5,6,7,8,9
    const tokens = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    var loader = try DataLoader.init(allocator, &tokens, 1, 3);
    defer loader.deinit();

    // Batch 1: input=[0,1,2], target=[1,2,3]
    const batch1 = loader.nextBatch().?;
    try std.testing.expectEqual(@as(u32, 0), batch1.input_ids[0]);
    try std.testing.expectEqual(@as(u32, 1), batch1.input_ids[1]);
    try std.testing.expectEqual(@as(u32, 2), batch1.input_ids[2]);
    try std.testing.expectEqual(@as(u32, 1), batch1.targets[0]);
    try std.testing.expectEqual(@as(u32, 2), batch1.targets[1]);
    try std.testing.expectEqual(@as(u32, 3), batch1.targets[2]);
}

test "DataLoader returns null when exhausted" {
    const allocator = std.testing.allocator;
    const tokens = [_]u32{ 0, 1, 2, 3, 4 };

    var loader = try DataLoader.init(allocator, &tokens, 1, 4);
    defer loader.deinit();

    // First batch uses 5 tokens (seq_len=4 + 1 for target)
    const batch1 = loader.nextBatch();
    try std.testing.expect(batch1 != null);

    // No more data
    try std.testing.expect(loader.nextBatch() == null);
}

test "DataLoader reset enables re-iteration" {
    const allocator = std.testing.allocator;
    const tokens = [_]u32{ 0, 1, 2, 3, 4 };

    var loader = try DataLoader.init(allocator, &tokens, 1, 4);
    defer loader.deinit();

    _ = loader.nextBatch();
    try std.testing.expect(loader.nextBatch() == null);

    loader.reset();

    const batch = loader.nextBatch();
    try std.testing.expect(batch != null);
}

test "DataLoader numBatches" {
    const allocator = std.testing.allocator;
    // 10 tokens, batch=2, seq_len=2: stride=3, total_per_batch=6
    // 10/6 = 1 complete batch
    const tokens = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    var loader = try DataLoader.init(allocator, &tokens, 2, 2);
    defer loader.deinit();

    try std.testing.expectEqual(@as(usize, 1), loader.numBatches());
}

test "DataLoader batch_size > 1" {
    const allocator = std.testing.allocator;
    // 12 tokens, batch=2, seq_len=2: stride=3, total=6
    const tokens = [_]u32{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 };

    var loader = try DataLoader.init(allocator, &tokens, 2, 2);
    defer loader.deinit();

    const batch = loader.nextBatch().?;
    try std.testing.expectEqual(@as(usize, 2), batch.batch_size);

    // Sequence 0: input=[10,20], target=[20,30]
    try std.testing.expectEqual(@as(u32, 10), batch.input_ids[0]);
    try std.testing.expectEqual(@as(u32, 20), batch.input_ids[1]);
    try std.testing.expectEqual(@as(u32, 20), batch.targets[0]);
    try std.testing.expectEqual(@as(u32, 30), batch.targets[1]);

    // Sequence 1: input=[40,50], target=[50,60]
    try std.testing.expectEqual(@as(u32, 40), batch.input_ids[2]);
    try std.testing.expectEqual(@as(u32, 50), batch.input_ids[3]);
    try std.testing.expectEqual(@as(u32, 50), batch.targets[2]);
    try std.testing.expectEqual(@as(u32, 60), batch.targets[3]);
}
