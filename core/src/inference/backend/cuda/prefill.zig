//! Prefill path entrypoints extracted from CUDA engine.

const std = @import("std");
const log = @import("../../../log.zig");

pub fn prefill(self: anytype, tokens: []const u32, logits_out: []f32) !void {
    if (tokens.len == 0) {
        log.warn("inference", "CUDA prefill invalid args", .{
            .reason = "empty_tokens",
        });
        return error.InvalidArgument;
    }
    if (logits_out.len != self.vocab_size) {
        log.warn("inference", "CUDA prefill invalid args", .{
            .reason = "logits_len_mismatch",
            .logits_len = logits_out.len,
            .vocab_size = self.vocab_size,
        });
        return error.InvalidArgument;
    }
    if (tokens.len > self.max_seq_len) return error.InvalidArgument;
    self.slot_rope_position_delta = 0;
    const prefill_start_ns: i128 = std.time.nanoTimestamp();
    try self.ensureKvCapacity(tokens.len);

    var i: usize = 0;
    while (i < tokens.len) : (i += 1) {
        const download_logits = self.shouldDownloadPrefillLogitsImpl(i, tokens.len);
        try self.computeGpuPrototypeLogitsWithLayerLimit(
            tokens[i],
            i,
            if (download_logits) self.slot_logits else null,
            self.block_runtime.blocks.len,
            download_logits,
            download_logits,
            false,
            null,
            null,
            null,
        );
    }
    const prefill_elapsed_ns: u64 = @intCast(std.time.nanoTimestamp() - prefill_start_ns);
    self.logPrefillTimingImpl("prefill", tokens.len, prefill_elapsed_ns);
    @memcpy(logits_out, self.slot_logits);
    self.slot_position = tokens.len;
}

pub fn prefillSlot(
    self: anytype,
    slot_index: usize,
    tokens: []const u32,
    logits_out: []f32,
) !void {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasDecl(SelfType, "ensureSlotStateBlocksBoundForScheduler")) {
        try self.ensureSlotStateBlocksBoundForScheduler(slot_index);
    }
    if (tokens.len == 0) {
        log.warn("inference", "CUDA prefillSlot invalid args", .{
            .reason = "empty_tokens",
            .slot_index = slot_index,
        });
        return error.InvalidArgument;
    }
    if (logits_out.len != self.vocab_size) {
        log.warn("inference", "CUDA prefillSlot invalid args", .{
            .reason = "logits_len_mismatch",
            .slot_index = slot_index,
            .logits_len = logits_out.len,
            .vocab_size = self.vocab_size,
        });
        return error.InvalidArgument;
    }
    if (!self.slot_in_use or slot_index != 0) {
        log.warn("inference", "CUDA prefillSlot invalid args", .{
            .reason = "slot_state",
            .slot_index = slot_index,
            .slot_in_use = @as(u8, @intFromBool(self.slot_in_use)),
        });
        return error.InvalidArgument;
    }
    if (tokens.len > self.max_seq_len) return error.InvalidArgument;
    self.slot_rope_position_delta = 0;
    const prefill_start_ns: i128 = std.time.nanoTimestamp();
    try self.ensureKvCapacity(tokens.len);
    var i: usize = 0;
    while (i < tokens.len) : (i += 1) {
        const download_logits = self.shouldDownloadPrefillLogitsImpl(i, tokens.len);
        try self.computeGpuPrototypeLogitsWithLayerLimit(
            tokens[i],
            i,
            if (download_logits) self.slot_logits else null,
            self.block_runtime.blocks.len,
            download_logits,
            download_logits,
            false,
            null,
            null,
            null,
        );
    }
    const prefill_elapsed_ns: u64 = @intCast(std.time.nanoTimestamp() - prefill_start_ns);
    self.logPrefillTimingImpl("prefill_slot", tokens.len, prefill_elapsed_ns);
    @memcpy(logits_out, self.slot_logits);
    self.slot_position = tokens.len;
}

const testing = std.testing;

const MockBackend = struct {
    const MockBlockRuntime = struct {
        blocks: [1]u8 = .{0},
    };

    vocab_size: usize = 8,
    max_seq_len: usize = 64,
    block_runtime: MockBlockRuntime = .{},
    slot_rope_position_delta: isize = 0,
    slot_position: usize = 0,
    slot_in_use: bool = true,
    slot_logits_storage: [8]f32 = [_]f32{0.0} ** 8,
    slot_logits: []f32 = undefined,
    ensure_kv_capacity_calls: usize = 0,
    compute_calls: usize = 0,
    timing_calls: usize = 0,

    fn init() MockBackend {
        var backend = MockBackend{};
        backend.slot_logits = backend.slot_logits_storage[0..];
        return backend;
    }

    fn ensureKvCapacity(self: *MockBackend, required_tokens: usize) !void {
        _ = required_tokens;
        self.ensure_kv_capacity_calls += 1;
    }

    fn shouldDownloadPrefillLogitsImpl(self: *const MockBackend, token_index: usize, token_count: usize) bool {
        _ = self;
        return token_index + 1 == token_count;
    }

    fn computeGpuPrototypeLogitsWithLayerLimit(
        self: *MockBackend,
        token: u32,
        position: usize,
        logits_out_opt: ?[]f32,
        layer_limit: usize,
        compute_logits: bool,
        download_logits: bool,
        ensure_kv_capacity: bool,
        hidden_override: ?[]const f32,
        deepstack_layer_features_opt: ?[]const []const f32,
        deepstack_feature_index_opt: ?usize,
    ) !void {
        _ = token;
        _ = position;
        _ = layer_limit;
        _ = compute_logits;
        _ = download_logits;
        _ = ensure_kv_capacity;
        _ = hidden_override;
        _ = deepstack_layer_features_opt;
        _ = deepstack_feature_index_opt;
        self.compute_calls += 1;
        if (logits_out_opt) |logits_out| {
            for (logits_out, 0..) |*logit, idx| {
                logit.* = @floatFromInt(idx);
            }
        }
    }

    fn logPrefillTimingImpl(self: *MockBackend, mode: []const u8, token_count: usize, elapsed_ns: u64) void {
        _ = mode;
        _ = token_count;
        _ = elapsed_ns;
        self.timing_calls += 1;
    }
};

test "prefill executes token-wise prefill path" {
    var backend = MockBackend.init();
    const tokens = [_]u32{ 1, 2, 3 };
    var logits_out: [8]f32 = undefined;

    try prefill(&backend, tokens[0..], logits_out[0..]);

    try testing.expectEqual(@as(usize, 1), backend.ensure_kv_capacity_calls);
    try testing.expectEqual(tokens.len, backend.compute_calls);
    try testing.expectEqual(tokens.len, backend.slot_position);
}

test "prefillSlot executes token-wise prefill path" {
    var backend = MockBackend.init();
    const tokens = [_]u32{ 4, 5 };
    var logits_out: [8]f32 = undefined;

    try prefillSlot(&backend, 0, tokens[0..], logits_out[0..]);

    try testing.expectEqual(@as(usize, 1), backend.ensure_kv_capacity_calls);
    try testing.expectEqual(tokens.len, backend.compute_calls);
    try testing.expectEqual(tokens.len, backend.slot_position);
}
