//! Decode path entrypoints extracted from CUDA engine.

const std = @import("std");
const contract = @import("../contract.zig");
const common_mrope = @import("../../vision_mrope.zig");
const log = @import("log_pkg");
const preflight = @import("route_preflight.zig");
const trace = @import("xray_pkg").trace;

fn markDecodePointerTablesDirty(self: anytype) void {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "decode_ptr_tables_dirty")) {
        self.decode_ptr_tables_dirty = true;
        if (comptime @hasField(SelfType, "decode_ptr_tables_cached_rows")) {
            self.decode_ptr_tables_cached_rows = 0;
        }
    }
    if (comptime @hasField(SelfType, "batched_decode_graph_exec")) {
        if (self.batched_decode_graph_exec) |exec| {
            self.device.graphExecDestroy(exec);
            self.batched_decode_graph_exec = null;
        }
    }
}

pub fn decode(self: anytype, token: u32, position: usize, logits_out: []f32) !void {
    const prev_backend = trace.setBackendContext(.cuda);
    defer _ = trace.setBackendContext(prev_backend);
    try preflight.requireDecodeRequest("decode", self, 0, logits_out.len);
    const effective_position = try common_mrope.applyPositionDelta(position, self.slot_rope_position_deltas[0]);
    try self.executeDecode(token, effective_position, logits_out);
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
    self.slot_positions[0] = position + 1;
}

pub fn decodeBatch(
    self: anytype,
    requests: []const contract.DecodeRequest,
    results: []contract.DecodeResult,
) !void {
    const prev_backend = trace.setBackendContext(.cuda);
    defer _ = trace.setBackendContext(prev_backend);
    const SelfType = @TypeOf(self.*);
    try preflight.requireDecodeBatchResultCapacity("decodeBatch", requests.len, results.len);
    if (requests.len == 0) return;
    try preflight.requireDecodeRows("decodeBatch", self, requests);

    // Canonical path: use batched decode implementation for all batch sizes,
    // including N=1, so decode behavior stays on one route.
    const use_batched = comptime @hasDecl(SelfType, "computeBatchedDecodeLogits");
    if (use_batched) {
        const max_n = 128;
        if (requests.len > max_n) return error.InvalidArgument;
        var tokens_buf: [max_n]u32 = undefined;
        var slot_indices_buf: [max_n]usize = undefined;
        var positions_buf: [max_n]usize = undefined;
        try preflight.fillDecodeRows(self, requests, tokens_buf[0..], slot_indices_buf[0..], positions_buf[0..]);
        self.computeBatchedDecodeLogits(
            tokens_buf[0..requests.len],
            slot_indices_buf[0..requests.len],
            positions_buf[0..requests.len],
        ) catch |err| {
            log.warn("inference", "CUDA decodeBatch batched decode failed", .{
                .batch_rows = requests.len,
                .slot0 = slot_indices_buf[0],
                .position0 = positions_buf[0],
                .token0 = tokens_buf[0],
                .reason = @errorName(err),
            });
            return err;
        };
        for (requests, results[0..requests.len], 0..) |req, *result, row_i| {
            const position = self.slot_positions[req.slot_index];
            const slot_logits = self.slotLogits(req.slot_index);
            var result_logits = slot_logits;
            if (comptime @hasDecl(SelfType, "batchedHostLogitsRow")) {
                if (self.batchedHostLogitsRow(row_i)) |row_logits| {
                    if (row_logits.len == self.vocab_size) {
                        result_logits = row_logits;
                    } else {
                        @memset(slot_logits, -1.0e9);
                        const copy_len = @min(slot_logits.len, row_logits.len);
                        @memcpy(slot_logits[0..copy_len], row_logits[0..copy_len]);
                    }
                }
            }
            result.* = .{
                .slot_index = req.slot_index,
                .logits = result_logits,
            };
            trace.emitFinal(
                .logits_ready,
                0,
                @intCast(position + 1),
                @ptrCast(result_logits.ptr),
                .f32,
                .{ @intCast(result_logits.len), 0, 0, 0 },
                1,
                "cuda_logits_host",
            );
            self.slot_positions[req.slot_index] = position + 1;
        }
        return;
    }

    // Sequential path: N == 1 or backend lacks batched support.
    for (requests, results[0..requests.len]) |req, *result| {
        const position = self.slot_positions[req.slot_index];
        const effective_position = try common_mrope.applyPositionDelta(position, self.slot_rope_position_deltas[req.slot_index]);
        const slot_logits = self.slotLogits(req.slot_index);

        if (comptime @hasDecl(SelfType, "activateKvSlot")) {
            self.activateKvSlot(req.slot_index);
        }

        try self.executeDecodeWithLayerLimit(
            req.token,
            effective_position,
            req.slot_index,
            slot_logits,
            self.block_runtime.blocks.len,
            true,
            true,
            true,
            1,
            position,
            null,
            null,
            null,
            false,
        );

        if (comptime @hasDecl(SelfType, "saveActiveKvSlot")) {
            self.saveActiveKvSlot();
        }

        result.* = .{
            .slot_index = req.slot_index,
            .logits = slot_logits,
        };
        trace.emitFinal(
            .logits_ready,
            0,
            @intCast(position + 1),
            @ptrCast(slot_logits.ptr),
            .f32,
            .{ @intCast(self.vocab_size), 0, 0, 0 },
            1,
            "cuda_logits_host",
        );
        self.slot_positions[req.slot_index] = position + 1;
    }
}

pub fn allocSlot(self: anytype) ?usize {
    const SelfType = @TypeOf(self.*);
    for (self.slot_in_use, 0..) |in_use, i| {
        if (!in_use) {
            self.slot_in_use[i] = true;
            self.slot_positions[i] = 0;
            self.slot_rope_position_deltas[i] = 0;
            if (comptime @hasField(SelfType, "slot_request_ids") and @hasField(SelfType, "next_slot_request_id")) {
                if (i < self.slot_request_ids.len) {
                    self.slot_request_ids[i] = self.next_slot_request_id;
                    self.next_slot_request_id +%= 1;
                    if (self.next_slot_request_id == 0) self.next_slot_request_id = 1;
                }
            }
            markDecodePointerTablesDirty(self);
            return i;
        }
    }
    return null;
}

pub fn freeSlot(self: anytype, slot_index: usize) void {
    const SelfType = @TypeOf(self.*);
    if (!preflight.slotIndexSupported(self, slot_index)) return;
    self.slot_in_use[slot_index] = false;
    self.slot_positions[slot_index] = 0;
    self.slot_rope_position_deltas[slot_index] = 0;
    if (comptime @hasField(SelfType, "slot_request_ids")) {
        if (slot_index < self.slot_request_ids.len) self.slot_request_ids[slot_index] = null;
    }
    markDecodePointerTablesDirty(self);
}

pub fn resetSlot(self: anytype, slot_index: usize) void {
    if (!preflight.slotIndexSupported(self, slot_index)) return;
    self.slot_positions[slot_index] = 0;
    self.slot_rope_position_deltas[slot_index] = 0;
    markDecodePointerTablesDirty(self);
}

pub fn getPosition(self: anytype, slot_index: usize) usize {
    if (!preflight.slotIndexSupported(self, slot_index)) return 0;
    return self.slot_positions[slot_index];
}

const testing = std.testing;

const MockDecodeBackend = struct {
    const MockBlockRuntime = struct {
        blocks: [1]u8 = .{0},
    };
    const max_batch_size: usize = 2;

    vocab_size: usize = 8,
    slot_rope_position_deltas: [max_batch_size]isize = .{ 0, 0 },
    slot_positions: [max_batch_size]usize = .{ 0, 0 },
    slot_in_use: [max_batch_size]bool = .{ true, false },
    block_runtime: MockBlockRuntime = .{},
    slot_logits_storage: [max_batch_size * 8]f32 = [_]f32{0.0} ** (max_batch_size * 8),
    slot_logits: []f32 = undefined,
    ensure_state_binding_calls: usize = 0,
    slot_state_bound: bool = true,
    compute_calls: usize = 0,

    fn init() MockDecodeBackend {
        var backend = MockDecodeBackend{};
        backend.slot_logits = backend.slot_logits_storage[0..];
        return backend;
    }

    fn slotLogits(self: *MockDecodeBackend, slot_index: usize) []f32 {
        const offset = slot_index * self.vocab_size;
        return self.slot_logits[offset .. offset + self.vocab_size];
    }

    fn slotIndexSupported(_: *const MockDecodeBackend, slot_index: usize) bool {
        return slot_index < max_batch_size;
    }

    fn ensureSlotStateBlocksBoundForScheduler(self: *MockDecodeBackend, slot_index: usize) !void {
        self.ensure_state_binding_calls += 1;
        if (slot_index >= max_batch_size) return error.InvalidArgument;
        if (!self.slot_state_bound) return error.InvalidStateDescriptorBinding;
    }

    fn executeDecode(
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

    fn executeDecodeWithLayerLimit(
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
        use_preloaded_input: bool,
    ) !void {
        _ = token;
        _ = position;
        _ = slot_index;
        _ = layer_limit;
        _ = compute_logits;
        _ = download_logits;
        _ = ensure_kv_capacity;
        _ = trace_seq_len_u32;
        _ = trace_pos_offset;
        _ = hidden_override;
        _ = deepstack_layer_features_opt;
        _ = deepstack_feature_index_opt;
        _ = use_preloaded_input;
        self.compute_calls += 1;
        // Fill logits if provided so tests can verify content.
        if (logits_out_opt) |logits_out| {
            for (logits_out, 0..) |*value, idx| {
                value.* = @floatFromInt(idx);
            }
        }
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

test "allocSlot returns first free slot" {
    var backend = MockDecodeBackend.init();
    // slot 0 is in_use by default, slot 1 is free
    const slot = allocSlot(&backend);
    try testing.expectEqual(@as(?usize, 1), slot);
    try testing.expect(backend.slot_in_use[1]);
    try testing.expectEqual(@as(usize, 0), backend.slot_positions[1]);
}

test "allocSlot returns null when all slots used" {
    var backend = MockDecodeBackend.init();
    backend.slot_in_use = .{ true, true };
    try testing.expectEqual(@as(?usize, null), allocSlot(&backend));
}

test "freeSlot releases slot" {
    var backend = MockDecodeBackend.init();
    backend.slot_positions[0] = 42;
    freeSlot(&backend, 0);
    try testing.expect(!backend.slot_in_use[0]);
    try testing.expectEqual(@as(usize, 0), backend.slot_positions[0]);
}

test "decodeBatch processes single request" {
    var backend = MockDecodeBackend.init();
    const requests = [_]contract.DecodeRequest{.{ .slot_index = 0, .token = 5 }};
    var results: [1]contract.DecodeResult = undefined;

    try decodeBatch(&backend, requests[0..], results[0..]);

    try testing.expectEqual(@as(usize, 1), backend.compute_calls);
    try testing.expectEqual(@as(usize, 0), results[0].slot_index);
    try testing.expectEqual(@as(usize, 1), backend.slot_positions[0]);
}

test "decodeBatch processes multiple requests" {
    var backend = MockDecodeBackend.init();
    backend.slot_in_use = .{ true, true };
    backend.slot_positions = .{ 10, 20 };
    const requests = [_]contract.DecodeRequest{
        .{ .slot_index = 0, .token = 5 },
        .{ .slot_index = 1, .token = 9 },
    };
    var results: [2]contract.DecodeResult = undefined;

    try decodeBatch(&backend, requests[0..], results[0..]);

    try testing.expectEqual(@as(usize, 2), backend.compute_calls);
    try testing.expectEqual(@as(usize, 0), results[0].slot_index);
    try testing.expectEqual(@as(usize, 1), results[1].slot_index);
    try testing.expectEqual(@as(usize, 11), backend.slot_positions[0]);
    try testing.expectEqual(@as(usize, 21), backend.slot_positions[1]);
}

test "decodeBatch lifecycle stress remains deterministic under interleaved slot operations" {
    var backend = MockDecodeBackend.init();
    backend.slot_in_use = .{ false, false };
    var results: [2]contract.DecodeResult = undefined;
    var requests: [2]contract.DecodeRequest = undefined;
    var expected_compute_calls: usize = 0;

    for (0..80) |iter| {
        if (!backend.slot_in_use[0]) _ = allocSlot(&backend);
        if (!backend.slot_in_use[1] and iter % 2 == 0) _ = allocSlot(&backend);

        var req_count: usize = 0;
        for (backend.slot_in_use, 0..) |in_use, slot_index| {
            if (!in_use) continue;
            requests[req_count] = .{
                .slot_index = slot_index,
                .token = @intCast(100 + iter + slot_index),
            };
            req_count += 1;
        }
        if (req_count > 0) {
            try decodeBatch(&backend, requests[0..req_count], results[0..req_count]);
            expected_compute_calls += req_count;
        }

        if (iter % 5 == 0 and backend.slot_in_use[1]) {
            resetSlot(&backend, 1);
        }
        if (iter % 7 == 0 and backend.slot_in_use[1]) {
            freeSlot(&backend, 1);
        }
        if (iter % 11 == 0) {
            backend.slot_state_bound = false;
            if (backend.slot_in_use[0]) {
                requests[0] = .{ .slot_index = 0, .token = @intCast(iter) };
                try testing.expectError(
                    error.InvalidStateDescriptorBinding,
                    decodeBatch(&backend, requests[0..1], results[0..1]),
                );
            }
            backend.slot_state_bound = true;
        }
    }

    try testing.expectEqual(expected_compute_calls, backend.compute_calls);
}
