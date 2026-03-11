//! Prefill path entrypoints extracted from CUDA engine.

const std = @import("std");
const log = @import("../../../log.zig");
const trace = @import("../../../xray/trace.zig");

fn slotIndexSupported(self: anytype, slot_index: usize) bool {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasDecl(SelfType, "slotIndexSupported")) return self.slotIndexSupported(slot_index);
    if (comptime @hasField(SelfType, "max_batch_size")) return slot_index < self.max_batch_size;
    return slot_index == 0;
}

pub fn prefill(self: anytype, tokens: []const u32, logits_out: []f32) !void {
    const prev_backend = trace.setBackendContext(.cuda);
    defer _ = trace.setBackendContext(prev_backend);
    const SelfType = @TypeOf(self.*);
    if (comptime @hasDecl(SelfType, "ensureSlotStateBlocksBoundForScheduler")) {
        try self.ensureSlotStateBlocksBoundForScheduler(0);
    }
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
    try self.beginParityPrefillCapture(tokens.len);
    defer self.endParityPrefillCapture();
    const prefill_start_ns: i128 = std.time.nanoTimestamp();
    self.parity_prefill_token_index = tokens.len - 1;
    self.computeGpuPrototypePrefillLogitsWithLayerLimit(
        tokens,
        0,
        self.slot_logits,
        self.block_runtime.blocks.len,
    ) catch |err| {
        log.warn("inference", "CUDA prefill batched path failed", .{
            .token_count = tokens.len,
            .reason = @errorName(err),
        });
        return err;
    };
    const prefill_elapsed_ns: u64 = @intCast(std.time.nanoTimestamp() - prefill_start_ns);
    self.logPrefillTimingImpl("prefill", tokens.len, prefill_elapsed_ns);
    @memcpy(logits_out, self.slot_logits);
    trace.emitFinal(
        .logits_ready,
        0,
        @intCast(tokens.len),
        @ptrCast(logits_out.ptr),
        .f32,
        .{ @intCast(self.vocab_size), 0, 0, 0 },
        1,
        "cuda_logits_host",
    );
    self.slot_position = tokens.len;
}

pub fn prefillSlot(
    self: anytype,
    slot_index: usize,
    tokens: []const u32,
    logits_out: []f32,
) !void {
    const prev_backend = trace.setBackendContext(.cuda);
    defer _ = trace.setBackendContext(prev_backend);
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
    if (!self.slot_in_use or !slotIndexSupported(self, slot_index)) {
        log.warn("inference", "CUDA prefillSlot invalid args", .{
            .reason = "slot_state",
            .slot_index = slot_index,
            .slot_in_use = @as(u8, @intFromBool(self.slot_in_use)),
        });
        return error.InvalidArgument;
    }
    if (tokens.len > self.max_seq_len) return error.InvalidArgument;
    self.slot_rope_position_delta = 0;
    try self.beginParityPrefillCapture(tokens.len);
    defer self.endParityPrefillCapture();
    const prefill_start_ns: i128 = std.time.nanoTimestamp();
    self.parity_prefill_token_index = tokens.len - 1;
    self.computeGpuPrototypePrefillLogitsWithLayerLimit(
        tokens,
        slot_index,
        self.slot_logits,
        self.block_runtime.blocks.len,
    ) catch |err| {
        log.warn("inference", "CUDA prefillSlot batched path failed", .{
            .slot_index = slot_index,
            .token_count = tokens.len,
            .reason = @errorName(err),
        });
        return err;
    };
    const prefill_elapsed_ns: u64 = @intCast(std.time.nanoTimestamp() - prefill_start_ns);
    self.logPrefillTimingImpl("prefill_slot", tokens.len, prefill_elapsed_ns);
    @memcpy(logits_out, self.slot_logits);
    trace.emitFinal(
        .logits_ready,
        0,
        @intCast(tokens.len),
        @ptrCast(logits_out.ptr),
        .f32,
        .{ @intCast(self.vocab_size), 0, 0, 0 },
        1,
        "cuda_logits_host",
    );
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
    ensure_state_binding_calls: usize = 0,
    slot_state_bound: bool = true,
    compute_calls: usize = 0,
    timing_calls: usize = 0,
    parity_prefill_token_index: usize = 0,

    fn init() MockBackend {
        var backend = MockBackend{};
        backend.slot_logits = backend.slot_logits_storage[0..];
        return backend;
    }

    fn ensureKvCapacity(self: *MockBackend, required_tokens: usize) !void {
        _ = required_tokens;
        self.ensure_kv_capacity_calls += 1;
    }

    fn ensureSlotStateBlocksBoundForScheduler(self: *MockBackend, slot_index: usize) !void {
        self.ensure_state_binding_calls += 1;
        if (!slotIndexSupported(self, slot_index)) return error.InvalidArgument;
        if (!self.slot_state_bound) return error.InvalidStateDescriptorBinding;
    }

    fn beginParityPrefillCapture(self: *MockBackend, seq_len: usize) !void {
        _ = self;
        _ = seq_len;
    }

    fn endParityPrefillCapture(self: *MockBackend) void {
        _ = self;
    }

    fn computeGpuPrototypePrefillLogitsWithLayerLimit(
        self: *MockBackend,
        tokens: []const u32,
        slot_index: usize,
        logits_out: []f32,
        layer_limit: usize,
    ) !void {
        _ = tokens;
        _ = slot_index;
        _ = layer_limit;
        self.ensure_kv_capacity_calls += 1;
        self.compute_calls += 1;
        for (logits_out, 0..) |*logit, idx| {
            logit.* = @floatFromInt(idx);
        }
    }

    fn logPrefillTimingImpl(self: *MockBackend, mode: []const u8, token_count: usize, elapsed_ns: u64) void {
        _ = mode;
        _ = token_count;
        _ = elapsed_ns;
        self.timing_calls += 1;
    }
};

test "prefill executes batched prefill path" {
    var backend = MockBackend.init();
    const tokens = [_]u32{ 1, 2, 3 };
    var logits_out: [8]f32 = undefined;

    try prefill(&backend, tokens[0..], logits_out[0..]);

    try testing.expectEqual(@as(usize, 1), backend.ensure_state_binding_calls);
    try testing.expectEqual(@as(usize, 1), backend.ensure_kv_capacity_calls);
    try testing.expectEqual(@as(usize, 1), backend.compute_calls);
    try testing.expectEqual(tokens.len, backend.slot_position);
}

test "prefillSlot executes batched prefill path" {
    var backend = MockBackend.init();
    const tokens = [_]u32{ 4, 5 };
    var logits_out: [8]f32 = undefined;

    try prefillSlot(&backend, 0, tokens[0..], logits_out[0..]);

    try testing.expectEqual(@as(usize, 1), backend.ensure_state_binding_calls);
    try testing.expectEqual(@as(usize, 1), backend.ensure_kv_capacity_calls);
    try testing.expectEqual(@as(usize, 1), backend.compute_calls);
    try testing.expectEqual(tokens.len, backend.slot_position);
}

test "prefill fails when slot state blocks are unbound" {
    var backend = MockBackend.init();
    backend.slot_state_bound = false;
    const tokens = [_]u32{1};
    var logits_out: [8]f32 = undefined;

    try testing.expectError(
        error.InvalidStateDescriptorBinding,
        prefill(&backend, tokens[0..], logits_out[0..]),
    );
    try testing.expectEqual(@as(usize, 1), backend.ensure_state_binding_calls);
    try testing.expectEqual(@as(usize, 0), backend.compute_calls);
}
