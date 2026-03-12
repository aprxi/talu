//! Decode path entrypoints extracted from CUDA engine.

const contract = @import("../contract.zig");
const common_mrope = @import("../../vision_mrope.zig");
const log = @import("../../../log.zig");
const trace = @import("../../../xray/trace.zig");

fn slotIndexSupported(self: anytype, slot_index: usize) bool {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasDecl(SelfType, "slotIndexSupported")) return self.slotIndexSupported(slot_index);
    if (comptime @hasField(SelfType, "max_batch_size")) return slot_index < self.max_batch_size;
    return slot_index == 0;
}

pub fn decode(self: anytype, token: u32, position: usize, logits_out: []f32) !void {
    const prev_backend = trace.setBackendContext(.cuda);
    defer _ = trace.setBackendContext(prev_backend);
    const SelfType = @TypeOf(self.*);
    if (comptime @hasDecl(SelfType, "ensureSlotStateBlocksBoundForScheduler")) {
        try self.ensureSlotStateBlocksBoundForScheduler(0);
    }
    if (logits_out.len != self.vocab_size) {
        log.warn("inference", "CUDA decode invalid args", .{
            .reason = "logits_len_mismatch",
            .logits_len = logits_out.len,
            .vocab_size = self.vocab_size,
        });
        return error.InvalidArgument;
    }
    const effective_position = try common_mrope.applyPositionDelta(position, self.slot_rope_position_delta);
    try self.computeGpuPrototypeLogits(token, effective_position, logits_out);
    trace.emitFinal(
        .logits_ready,
        0,
        @intCast(position + 1),
        @ptrCast(logits_out.ptr),
        .f32,
        .{ @intCast(self.vocab_size), 0, 0, 0 },
        1,
        "cuda_logits_host",
    );
    self.slot_position = position + 1;
}

pub fn decodeBatch(
    self: anytype,
    requests: []const contract.DecodeRequest,
    results: []contract.DecodeResult,
) !void {
    const prev_backend = trace.setBackendContext(.cuda);
    defer _ = trace.setBackendContext(prev_backend);
    const SelfType = @TypeOf(self.*);
    if (results.len < requests.len) {
        log.warn("inference", "CUDA decodeBatch invalid args", .{
            .reason = "results_short",
            .requests = requests.len,
            .results = results.len,
        });
        return error.InvalidArgument;
    }
    if (requests.len == 0) return;
    if (requests.len > 1) {
        log.warn("inference", "CUDA decodeBatch invalid args", .{
            .reason = "batch_gt_one",
            .requests = requests.len,
        });
        return error.InvalidArgument;
    }

    const req = requests[0];
    if (comptime @hasDecl(SelfType, "ensureSlotStateBlocksBoundForScheduler")) {
        try self.ensureSlotStateBlocksBoundForScheduler(req.slot_index);
    }
    if (!self.slot_in_use or !slotIndexSupported(self, req.slot_index)) {
        log.warn("inference", "CUDA decodeBatch invalid args", .{
            .reason = "slot_state",
            .slot_index = req.slot_index,
            .slot_in_use = @as(u8, @intFromBool(self.slot_in_use)),
        });
        return error.InvalidArgument;
    }

    const effective_position = try common_mrope.applyPositionDelta(self.slot_position, self.slot_rope_position_delta);
    try self.computeGpuPrototypeLogitsWithLayerLimit(
        req.token,
        effective_position,
        req.slot_index,
        self.slot_logits,
        self.block_runtime.blocks.len,
        true,
        true,
        true,
        1,
        self.slot_position,
        null,
        null,
        null,
    );
    results[0] = .{
        .slot_index = req.slot_index,
        .logits = self.slot_logits,
    };
    trace.emitFinal(
        .logits_ready,
        0,
        @intCast(self.slot_position + 1),
        @ptrCast(self.slot_logits.ptr),
        .f32,
        .{ @intCast(self.vocab_size), 0, 0, 0 },
        1,
        "cuda_logits_host",
    );
    self.slot_position += 1;
}

pub fn decodeStreaming(
    self: anytype,
    first_token: u32,
    start_position: usize,
    max_tokens: usize,
    eos_token_ids: []const u32,
    output_tokens: []u32,
    callback: ?*const fn (u32, ?*anyopaque) void,
    callback_data: ?*anyopaque,
) !usize {
    const prev_backend = trace.setBackendContext(.cuda);
    defer _ = trace.setBackendContext(prev_backend);
    const SelfType = @TypeOf(self.*);
    if (comptime @hasDecl(SelfType, "ensureSlotStateBlocksBoundForScheduler")) {
        try self.ensureSlotStateBlocksBoundForScheduler(0);
    }
    if (max_tokens == 0 or output_tokens.len == 0) return 0;
    if (!self.slot_in_use) {
        self.slot_in_use = true;
        self.slot_position = start_position;
    }

    var current_token = first_token;
    var generated: usize = 0;
    var position = self.slot_position;
    const budget = @min(max_tokens, output_tokens.len);
    if (comptime @hasDecl(SelfType, "ensureKvCapacity")) {
        if (budget > 0) {
            const required_capacity = std.math.add(usize, position, budget) catch return error.InvalidArgument;
            try self.ensureKvCapacity(required_capacity);
        }
    }
    while (generated < budget) : (generated += 1) {
        const effective_position = try common_mrope.applyPositionDelta(position, self.slot_rope_position_delta);
        try self.computeGpuPrototypeLogitsWithLayerLimit(
            current_token,
            effective_position,
            0,
            null,
            self.block_runtime.blocks.len,
            true,
            false,
            false,
            1,
            position,
            null,
            null,
            null,
        );
        const next_token = try self.selectNextTokenFromDeviceLogitsImpl();
        trace.emitFinal(
            .token_select,
            0,
            @intCast(position + 1),
            @ptrCast(std.mem.asBytes(&next_token).ptr),
            .u32,
            .{ 1, 0, 0, 0 },
            1,
            "gpu_argmax",
        );
        output_tokens[generated] = next_token;
        position += 1;
        self.slot_position = position;
        if (callback) |cb| cb(next_token, callback_data);

        for (eos_token_ids) |eos_id| {
            if (next_token == eos_id) {
                return generated + 1;
            }
        }
        current_token = next_token;
    }
    return generated;
}

pub fn allocSlot(self: anytype) ?usize {
    const SelfType = @TypeOf(self.*);
    if (self.slot_in_use) return null;
    self.slot_in_use = true;
    self.slot_position = 0;
    self.slot_rope_position_delta = 0;
    if (comptime @hasField(SelfType, "slot_state_bound")) {
        self.slot_state_bound = false;
        self.slot_state_block_count = 0;
    }
    return 0;
}

pub fn freeSlot(self: anytype, slot_index: usize) void {
    const SelfType = @TypeOf(self.*);
    if (!slotIndexSupported(self, slot_index)) return;
    self.slot_in_use = false;
    self.slot_position = 0;
    self.slot_rope_position_delta = 0;
    if (comptime @hasField(SelfType, "slot_state_bound")) {
        self.slot_state_bound = false;
        self.slot_state_block_count = 0;
    }
}

pub fn resetSlot(self: anytype, slot_index: usize) void {
    if (!slotIndexSupported(self, slot_index)) return;
    self.slot_position = 0;
    self.slot_rope_position_delta = 0;
}

pub fn getPosition(self: anytype, slot_index: usize) usize {
    if (!slotIndexSupported(self, slot_index)) return 0;
    return self.slot_position;
}

const std = @import("std");
const testing = std.testing;

const MockDecodeBackend = struct {
    const MockBlockRuntime = struct {
        blocks: [1]u8 = .{0},
    };

    vocab_size: usize = 8,
    slot_rope_position_delta: isize = 0,
    slot_position: usize = 0,
    slot_in_use: bool = true,
    block_runtime: MockBlockRuntime = .{},
    slot_logits_storage: [8]f32 = [_]f32{0.0} ** 8,
    slot_logits: []f32 = undefined,
    ensure_state_binding_calls: usize = 0,
    slot_state_bound: bool = true,
    compute_calls: usize = 0,
    next_token: u32 = 7,

    fn init() MockDecodeBackend {
        var backend = MockDecodeBackend{};
        backend.slot_logits = backend.slot_logits_storage[0..];
        return backend;
    }

    fn ensureSlotStateBlocksBoundForScheduler(self: *MockDecodeBackend, slot_index: usize) !void {
        self.ensure_state_binding_calls += 1;
        if (!slotIndexSupported(self, slot_index)) return error.InvalidArgument;
        if (!self.slot_state_bound) return error.InvalidStateDescriptorBinding;
    }

    fn computeGpuPrototypeLogits(
        self: *MockDecodeBackend,
        token: u32,
        position: usize,
        logits_out: []f32,
    ) !void {
        _ = token;
        _ = position;
        self.compute_calls += 1;
        for (logits_out, 0..) |*value, idx| {
            value.* = @floatFromInt(idx);
        }
    }

    fn computeGpuPrototypeLogitsWithLayerLimit(
        self: *MockDecodeBackend,
        token: u32,
        position: usize,
        slot_index: usize,
        logits_out_opt: ?[]f32,
        layer_limit: usize,
        compute_logits: bool,
        download_logits: bool,
        ensure_kv_capacity: bool,
        trace_seq_len_u32: u32,
        trace_pos_offset: usize,
        hidden_override: ?[]const f32,
        deepstack_layer_features_opt: ?[]const []const f32,
        deepstack_feature_index_opt: ?usize,
    ) !void {
        _ = token;
        _ = position;
        _ = slot_index;
        _ = logits_out_opt;
        _ = layer_limit;
        _ = compute_logits;
        _ = download_logits;
        _ = ensure_kv_capacity;
        _ = trace_seq_len_u32;
        _ = trace_pos_offset;
        _ = hidden_override;
        _ = deepstack_layer_features_opt;
        _ = deepstack_feature_index_opt;
        self.compute_calls += 1;
    }

    fn selectNextTokenFromDeviceLogitsImpl(self: *MockDecodeBackend) !u32 {
        return self.next_token;
    }
};

test "decode enforces state binding guard" {
    var backend = MockDecodeBackend.init();
    backend.slot_state_bound = false;
    var logits_out: [8]f32 = undefined;

    try testing.expectError(
        error.InvalidStateDescriptorBinding,
        decode(&backend, 1, 0, logits_out[0..]),
    );
    try testing.expectEqual(@as(usize, 1), backend.ensure_state_binding_calls);
    try testing.expectEqual(@as(usize, 0), backend.compute_calls);
}

test "decodeStreaming enforces state binding guard" {
    var backend = MockDecodeBackend.init();
    backend.slot_state_bound = false;
    var output_tokens: [4]u32 = undefined;

    try testing.expectError(
        error.InvalidStateDescriptorBinding,
        decodeStreaming(
            &backend,
            1,
            0,
            output_tokens.len,
            &[_]u32{2},
            output_tokens[0..],
            null,
            null,
        ),
    );
    try testing.expectEqual(@as(usize, 1), backend.ensure_state_binding_calls);
    try testing.expectEqual(@as(usize, 0), backend.compute_calls);
}
