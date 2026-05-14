//! CUDA route preflight helpers.
//!
//! These helpers own route-local validation and row setup only. They must not
//! choose routes, allocate hot-path buffers, launch kernels, or perform sampling.

const std = @import("std");
const log = @import("log_pkg");
const common_mrope = @import("../../vision_mrope.zig");

pub const TopKBufferShape = struct {
    total_candidates: usize,
};

pub inline fn slotIndexSupported(self: anytype, slot_index: usize) bool {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasDecl(SelfType, "slotIndexSupported")) return self.slotIndexSupported(slot_index);
    if (comptime @hasField(SelfType, "max_batch_size")) return slot_index < self.max_batch_size;
    if (comptime @hasDecl(SelfType, "max_batch_size")) return slot_index < SelfType.max_batch_size;
    return slot_index == 0;
}

pub inline fn slotInUse(self: anytype, slot_index: usize) bool {
    if (!slotIndexSupported(self, slot_index)) return false;
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "slot_in_use")) {
        if (slot_index >= self.slot_in_use.len) return false;
        return self.slot_in_use[slot_index];
    }
    return true;
}

pub inline fn requireStateBlocksBoundIfPresent(self: anytype, slot_index: usize) !void {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasDecl(SelfType, "ensureSlotStateBlocksBoundForScheduler")) {
        try self.ensureSlotStateBlocksBoundForScheduler(slot_index);
    }
}

pub inline fn requireSlotInUse(comptime operation: []const u8, self: anytype, slot_index: usize) !void {
    const in_use = slotInUse(self, slot_index);
    if (!in_use) {
        log.warn("inference", "CUDA " ++ operation ++ " invalid args", .{
            .reason = "slot_state",
            .slot_index = slot_index,
            .slot_in_use = @as(u8, @intFromBool(in_use)),
        });
        return error.InvalidArgument;
    }
}

pub inline fn requirePrefillRequest(
    comptime operation: []const u8,
    self: anytype,
    tokens: []const u32,
    logits_len: usize,
) !void {
    try requireStateBlocksBoundIfPresent(self, 0);
    if (tokens.len == 0) {
        log.warn("inference", "CUDA " ++ operation ++ " invalid args", .{
            .reason = "empty_tokens",
        });
        return error.InvalidArgument;
    }
    if (logits_len != self.vocab_size) {
        log.warn("inference", "CUDA " ++ operation ++ " invalid args", .{
            .reason = "logits_len_mismatch",
            .logits_len = logits_len,
            .vocab_size = self.vocab_size,
        });
        return error.InvalidArgument;
    }
    if (tokens.len > self.max_seq_len) return error.InvalidArgument;
}

pub inline fn requirePrefillSlotRequest(
    comptime operation: []const u8,
    self: anytype,
    slot_index: usize,
    tokens: []const u32,
    logits_len: usize,
) !void {
    try requireStateBlocksBoundIfPresent(self, slot_index);
    if (tokens.len == 0) {
        log.warn("inference", "CUDA " ++ operation ++ " invalid args", .{
            .reason = "empty_tokens",
            .slot_index = slot_index,
        });
        return error.InvalidArgument;
    }
    if (logits_len != self.vocab_size) {
        log.warn("inference", "CUDA " ++ operation ++ " invalid args", .{
            .reason = "logits_len_mismatch",
            .slot_index = slot_index,
            .logits_len = logits_len,
            .vocab_size = self.vocab_size,
        });
        return error.InvalidArgument;
    }
    try requireSlotInUse(operation, self, slot_index);
    if (tokens.len > self.max_seq_len) return error.InvalidArgument;
}

pub inline fn requireDecodeRequest(
    comptime operation: []const u8,
    self: anytype,
    slot_index: usize,
    logits_len: usize,
) !void {
    try requireStateBlocksBoundIfPresent(self, slot_index);
    if (logits_len != self.vocab_size) {
        log.warn("inference", "CUDA " ++ operation ++ " invalid args", .{
            .reason = "logits_len_mismatch",
            .logits_len = logits_len,
            .vocab_size = self.vocab_size,
        });
        return error.InvalidArgument;
    }
}

pub inline fn requireDecodeBatchResultCapacity(
    comptime operation: []const u8,
    request_count: usize,
    result_count: usize,
) !void {
    if (result_count < request_count) {
        log.warn("inference", "CUDA " ++ operation ++ " invalid args", .{
            .reason = "results_short",
            .requests = request_count,
            .results = result_count,
        });
        return error.InvalidArgument;
    }
}

pub inline fn requireTopKCandidateBuffers(
    row_count: usize,
    top_k: usize,
    candidate_logits_len: usize,
    candidate_ids_len: usize,
    candidate_counts_len: usize,
    comptime max_top_k: usize,
) !TopKBufferShape {
    if (row_count == 0) return error.InvalidArgument;
    if (top_k == 0 or top_k > max_top_k) return error.InvalidArgument;
    if (candidate_counts_len < row_count) return error.InvalidArgument;
    const total_candidates = std.math.mul(usize, row_count, top_k) catch return error.InvalidArgument;
    if (candidate_logits_len < total_candidates or candidate_ids_len < total_candidates) return error.InvalidArgument;
    return .{ .total_candidates = total_candidates };
}

pub inline fn requireTopKCandidateRowBuffers(
    top_k: usize,
    candidate_logits_len: usize,
    candidate_ids_len: usize,
    comptime max_top_k: usize,
) !void {
    if (top_k == 0 or top_k > max_top_k) return error.InvalidArgument;
    if (candidate_logits_len < top_k or candidate_ids_len < top_k) return error.InvalidArgument;
}

pub inline fn requireDecodeRows(
    comptime operation: []const u8,
    self: anytype,
    requests: anytype,
) !void {
    for (requests) |req| {
        try requireStateBlocksBoundIfPresent(self, req.slot_index);
        try requireSlotInUse(operation, self, req.slot_index);
    }
}

inline fn requireDecodeRowBuffers(
    request_count: usize,
    tokens_len: usize,
    slot_indices_len: usize,
    positions_len: usize,
) !void {
    if (tokens_len < request_count or slot_indices_len < request_count or positions_len < request_count) {
        return error.InvalidArgument;
    }
}

/// Fills decode row buffers after `requireDecodeRows` accepts the requests.
pub inline fn fillDecodeRows(
    self: anytype,
    requests: anytype,
    tokens_out: []u32,
    slot_indices_out: []usize,
    positions_out: []usize,
) !void {
    try requireDecodeRowBuffers(requests.len, tokens_out.len, slot_indices_out.len, positions_out.len);
    for (requests, 0..) |req, i| {
        const raw_position = self.slot_positions[req.slot_index];
        tokens_out[i] = req.token;
        slot_indices_out[i] = req.slot_index;
        positions_out[i] = try common_mrope.applyPositionDelta(raw_position, self.slot_rope_position_deltas[req.slot_index]);
    }
}

/// Validates slots and fills decode row buffers.
pub inline fn prepareDecodeRows(
    comptime operation: []const u8,
    self: anytype,
    requests: anytype,
    tokens_out: []u32,
    slot_indices_out: []usize,
    positions_out: []usize,
) !void {
    try requireDecodeRowBuffers(requests.len, tokens_out.len, slot_indices_out.len, positions_out.len);
    try requireDecodeRows(operation, self, requests);
    try fillDecodeRows(self, requests, tokens_out, slot_indices_out, positions_out);
}

/// Fills decode row buffers and raw positions after slot validation.
pub inline fn fillDecodeRowsWithRaw(
    self: anytype,
    requests: anytype,
    tokens_out: []u32,
    slot_indices_out: []usize,
    positions_out: []usize,
    raw_positions_out: []usize,
) !void {
    if (raw_positions_out.len < requests.len) return error.InvalidArgument;
    try fillDecodeRows(self, requests, tokens_out, slot_indices_out, positions_out);
    for (requests, 0..) |req, i| {
        raw_positions_out[i] = self.slot_positions[req.slot_index];
    }
}

/// Validates slots and fills decode row buffers, preserving raw positions.
pub inline fn prepareDecodeRowsWithRaw(
    comptime operation: []const u8,
    self: anytype,
    requests: anytype,
    tokens_out: []u32,
    slot_indices_out: []usize,
    positions_out: []usize,
    raw_positions_out: []usize,
) !void {
    if (raw_positions_out.len < requests.len) return error.InvalidArgument;
    try requireDecodeRowBuffers(requests.len, tokens_out.len, slot_indices_out.len, positions_out.len);
    try requireDecodeRows(operation, self, requests);
    try fillDecodeRowsWithRaw(self, requests, tokens_out, slot_indices_out, positions_out, raw_positions_out);
}

const TestBackend = struct {
    const max_batch_size = 2;

    vocab_size: usize = 4,
    max_seq_len: usize = 8,
    slot_in_use: [max_batch_size]bool = .{ true, false },
    slot_positions: [max_batch_size]usize = .{ 3, 5 },
    slot_rope_position_deltas: [max_batch_size]isize = .{ 0, 2 },
    state_bound: bool = true,

    fn ensureSlotStateBlocksBoundForScheduler(self: *TestBackend, slot_index: usize) !void {
        if (slot_index >= max_batch_size) return error.InvalidArgument;
        if (!self.state_bound) return error.InvalidStateDescriptorBinding;
    }
};

test "slotIndexSupported and slotInUse validate slot bounds" {
    var backend = TestBackend{};
    try std.testing.expect(slotIndexSupported(&backend, 0));
    try std.testing.expect(!slotIndexSupported(&backend, 2));
    try std.testing.expect(slotInUse(&backend, 0));
    try std.testing.expect(!slotInUse(&backend, 1));
    try std.testing.expect(!slotInUse(&backend, 2));
}

test "requireStateBlocksBoundIfPresent and requireSlotInUse validate slot state" {
    var backend = TestBackend{};
    try requireStateBlocksBoundIfPresent(&backend, 0);
    try requireSlotInUse("test", &backend, 0);

    try std.testing.expectError(error.InvalidArgument, requireSlotInUse("test", &backend, 1));
    try std.testing.expectError(error.InvalidArgument, requireSlotInUse("test", &backend, 2));

    backend.state_bound = false;
    try std.testing.expectError(error.InvalidStateDescriptorBinding, requireStateBlocksBoundIfPresent(&backend, 0));
}

test "requirePrefillRequest and requirePrefillSlotRequest validate payload and slot" {
    var backend = TestBackend{};
    try requirePrefillRequest("prefill", &backend, &.{ 1, 2 }, 4);
    try std.testing.expectError(error.InvalidArgument, requirePrefillRequest("prefill", &backend, &.{}, 4));
    try std.testing.expectError(error.InvalidArgument, requirePrefillRequest("prefill", &backend, &.{1}, 3));
    try requirePrefillSlotRequest("prefillSlot", &backend, 0, &.{1}, 4);
    try std.testing.expectError(error.InvalidArgument, requirePrefillSlotRequest("prefillSlot", &backend, 1, &.{1}, 4));
}

test "requireDecodeRequest and requireDecodeBatchResultCapacity validate decode shapes" {
    var backend = TestBackend{};
    try requireDecodeRequest("decode", &backend, 0, 4);
    try std.testing.expectError(error.InvalidArgument, requireDecodeRequest("decode", &backend, 0, 3));
    try requireDecodeBatchResultCapacity("decodeBatch", 2, 2);
    try std.testing.expectError(error.InvalidArgument, requireDecodeBatchResultCapacity("decodeBatch", 2, 1));
}

test "requireTopKCandidateBuffers and requireTopKCandidateRowBuffers validate candidate slices" {
    const shape = try requireTopKCandidateBuffers(2, 3, 6, 6, 2, 256);
    try std.testing.expectEqual(@as(usize, 6), shape.total_candidates);
    try std.testing.expectError(error.InvalidArgument, requireTopKCandidateBuffers(0, 3, 6, 6, 2, 256));
    try std.testing.expectError(error.InvalidArgument, requireTopKCandidateBuffers(2, 0, 6, 6, 2, 256));
    try std.testing.expectError(error.InvalidArgument, requireTopKCandidateBuffers(2, 257, 514, 514, 2, 256));
    try std.testing.expectError(error.InvalidArgument, requireTopKCandidateBuffers(2, 3, 6, 6, 1, 256));
    try std.testing.expectError(error.InvalidArgument, requireTopKCandidateBuffers(2, 3, 5, 6, 2, 256));
    try std.testing.expectError(error.InvalidArgument, requireTopKCandidateBuffers(2, 3, 6, 5, 2, 256));
    try requireTopKCandidateRowBuffers(3, 3, 3, 256);
    try std.testing.expectError(error.InvalidArgument, requireTopKCandidateRowBuffers(0, 3, 3, 256));
    try std.testing.expectError(error.InvalidArgument, requireTopKCandidateRowBuffers(257, 257, 257, 256));
    try std.testing.expectError(error.InvalidArgument, requireTopKCandidateRowBuffers(3, 2, 3, 256));
    try std.testing.expectError(error.InvalidArgument, requireTopKCandidateRowBuffers(3, 3, 2, 256));
}

test "requireDecodeRows, fillDecodeRows, fillDecodeRowsWithRaw, prepareDecodeRows, and prepareDecodeRowsWithRaw handle decode rows" {
    var backend = TestBackend{};
    const requests = [_]struct { slot_index: usize, token: u32 }{
        .{ .slot_index = 0, .token = 42 },
    };
    var tokens: [1]u32 = undefined;
    var slots: [1]usize = undefined;
    var positions: [1]usize = undefined;
    var raw_positions: [1]usize = undefined;

    try requireDecodeRows("decodeBatch", &backend, requests[0..]);
    try fillDecodeRows(&backend, requests[0..], tokens[0..], slots[0..], positions[0..]);
    try std.testing.expectEqual(@as(u32, 42), tokens[0]);
    try std.testing.expectEqual(@as(usize, 0), slots[0]);
    try std.testing.expectEqual(@as(usize, 3), positions[0]);

    try prepareDecodeRows("decodeBatch", &backend, requests[0..], tokens[0..], slots[0..], positions[0..]);
    try std.testing.expectEqual(@as(u32, 42), tokens[0]);

    try fillDecodeRowsWithRaw(&backend, requests[0..], tokens[0..], slots[0..], positions[0..], raw_positions[0..]);
    try std.testing.expectEqual(@as(usize, 3), raw_positions[0]);

    try prepareDecodeRowsWithRaw("decodeBatchTopKCandidates", &backend, requests[0..], tokens[0..], slots[0..], positions[0..], raw_positions[0..]);
    try std.testing.expectEqual(@as(usize, 3), raw_positions[0]);

    try std.testing.expectError(
        error.InvalidArgument,
        fillDecodeRows(&backend, requests[0..], tokens[0..0], slots[0..], positions[0..]),
    );
    try std.testing.expectError(
        error.InvalidArgument,
        fillDecodeRowsWithRaw(&backend, requests[0..], tokens[0..], slots[0..], positions[0..], raw_positions[0..0]),
    );

    backend.state_bound = false;
    try std.testing.expectError(
        error.InvalidArgument,
        prepareDecodeRows("decodeBatch", &backend, requests[0..], tokens[0..0], slots[0..], positions[0..]),
    );
    backend.state_bound = true;

    const invalid = [_]struct { slot_index: usize, token: u32 }{
        .{ .slot_index = 1, .token = 7 },
    };
    try std.testing.expectError(
        error.InvalidArgument,
        prepareDecodeRows("decodeBatch", &backend, invalid[0..], tokens[0..], slots[0..], positions[0..]),
    );
}
