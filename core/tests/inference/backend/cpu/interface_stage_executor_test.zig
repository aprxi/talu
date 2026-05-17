//! Tests for CPU backend stage executor interface forwarding.

const std = @import("std");
const main = @import("main");

const endpoint = main.inference.backend.cpu.interface.stage_executor;

fn preserveUnexpectedDecodeBoundaryFailure(
    root_backend: anytype,
    boundary: anytype,
    location_hint: anytype,
    slot_indices: []const usize,
    positions: []const usize,
    active_side: anytype,
    source_error: anyerror,
) anyerror {
    _ = root_backend;
    _ = boundary;
    _ = location_hint;
    _ = slot_indices;
    _ = positions;
    _ = active_side;
    _ = source_error;
    return error.TestUnexpectedResult;
}

test "executeDecodeLayerRange forwards final logits controls" {
    const MockCpuBackend = struct {
        saw_logits: bool = false,
        compute_logits: bool = false,
        download_logits: bool = false,
        use_preloaded_input: bool = false,
        layer_start: usize = 0,
        layer_end: usize = 0,

        pub fn executeDecodeLayerRange(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_start: usize,
            layer_end: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            use_preloaded_input: bool,
        ) !void {
            try std.testing.expectEqual(@as(u32, 17), token);
            try std.testing.expectEqual(@as(usize, 23), position);
            try std.testing.expectEqual(@as(usize, 3), slot_index);
            try std.testing.expect(ensure_kv_capacity);
            self.saw_logits = logits_out_opt != null;
            self.compute_logits = compute_logits;
            self.download_logits = download_logits;
            self.use_preloaded_input = use_preloaded_input;
            self.layer_start = layer_start;
            self.layer_end = layer_end;
        }
    };
    const DecodeContext = struct {
        token: u32,
        position: usize,
        slot_index: usize,
        ensure_kv_capacity: bool,
    };

    var backend = MockCpuBackend{};
    var logits = [_]f32{0} ** 4;
    const ctx = DecodeContext{
        .token = 17,
        .position = 23,
        .slot_index = 3,
        .ensure_kv_capacity = true,
    };

    try endpoint.executeDecodeLayerRange(&backend, ctx, 4, 7, logits[0..], true, true, true);

    try std.testing.expect(backend.saw_logits);
    try std.testing.expect(backend.compute_logits);
    try std.testing.expect(backend.download_logits);
    try std.testing.expect(backend.use_preloaded_input);
    try std.testing.expectEqual(@as(usize, 4), backend.layer_start);
    try std.testing.expectEqual(@as(usize, 7), backend.layer_end);
}

test "executePrefillLayerRange forwards generic prefill controls" {
    const MockCpuBackend = struct {
        slot_seen: usize = 0,
        sequence_start_seen: usize = 0,
        layer_start_seen: usize = 0,
        layer_end_seen: usize = 0,
        use_preloaded_seen: bool = false,
        compute_logits_seen: bool = false,
        logits_len_seen: usize = 0,
        source_embeddings_len_seen: usize = 0,

        pub fn executePrefillLayerRange(
            self: *@This(),
            slot_index: usize,
            tokens: []const u32,
            sequence_start: usize,
            layer_start: usize,
            layer_end: usize,
            use_preloaded_input: bool,
            compute_logits: bool,
            logits_out_opt: ?[]f32,
            source_embeddings_out: ?[]f32,
        ) !void {
            try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, tokens);
            self.slot_seen = slot_index;
            self.sequence_start_seen = sequence_start;
            self.layer_start_seen = layer_start;
            self.layer_end_seen = layer_end;
            self.use_preloaded_seen = use_preloaded_input;
            self.compute_logits_seen = compute_logits;
            self.logits_len_seen = if (logits_out_opt) |logits| logits.len else 0;
            self.source_embeddings_len_seen = if (source_embeddings_out) |embeddings| embeddings.len else 0;
        }
    };

    var backend = MockCpuBackend{};
    var logits = [_]f32{0} ** 8;
    var embeddings = [_]f32{0} ** 12;
    try endpoint.executePrefillLayerRange(
        &backend,
        4,
        &.{ 1, 2, 3 },
        9,
        2,
        5,
        true,
        true,
        logits[0..],
        embeddings[0..],
    );

    try std.testing.expectEqual(@as(usize, 4), backend.slot_seen);
    try std.testing.expectEqual(@as(usize, 9), backend.sequence_start_seen);
    try std.testing.expectEqual(@as(usize, 2), backend.layer_start_seen);
    try std.testing.expectEqual(@as(usize, 5), backend.layer_end_seen);
    try std.testing.expect(backend.use_preloaded_seen);
    try std.testing.expect(backend.compute_logits_seen);
    try std.testing.expectEqual(@as(usize, 8), backend.logits_len_seen);
    try std.testing.expectEqual(@as(usize, 12), backend.source_embeddings_len_seen);
}

test "prepareBatchedDecodeSegments exposes CPU activation rows and batched logits" {
    const MockRoot = struct {
        const RuntimeBuffers = struct {
            projected_vocab: usize,
            projected_logits_batch_host: []f32,
        };

        runtime_buffers: RuntimeBuffers,
        activated: [4]usize = .{0} ** 4,
        activated_count: usize = 0,

        pub fn activateKvSlot(self: *@This(), slot_index: usize) void {
            self.activated[self.activated_count] = slot_index;
            self.activated_count += 1;
        }
    };
    const MockCpuStage = struct {
        activation_rows: [2][4]u8 = .{
            .{ 1, 2, 3, 4 },
            .{ 5, 6, 7, 8 },
        },
        layer_start_seen: usize = 0,
        layer_end_seen: usize = 0,

        pub fn prepareBatchedDecodeSegments(
            self: *@This(),
            comptime preserve_decode_boundary_failure: anytype,
            root_backend: anytype,
            activate_intermediate: bool,
            intermediate_backend: anytype,
            boundary: anytype,
            tokens: []const u32,
            slot_indices: []const usize,
            positions: []const usize,
            layer_start: usize,
            layer_end: usize,
            use_preloaded_input: bool,
            compute_logits: bool,
            row_bytes: usize,
            host_segments: [][]const u8,
        ) !void {
            _ = preserve_decode_boundary_failure;
            _ = activate_intermediate;
            _ = intermediate_backend;
            _ = boundary;
            try std.testing.expectEqualSlices(usize, &.{ 10, 11 }, positions);
            try std.testing.expect(compute_logits);
            try std.testing.expect(use_preloaded_input);
            try std.testing.expectEqual(@as(usize, 4), row_bytes);
            self.layer_start_seen = layer_start;
            self.layer_end_seen = layer_end;
            for (tokens, slot_indices, 0..) |token, slot_index, row_index| {
                root_backend.activateKvSlot(slot_index);
                const start = row_index * root_backend.runtime_buffers.projected_vocab;
                const end = start + root_backend.runtime_buffers.projected_vocab;
                for (root_backend.runtime_buffers.projected_logits_batch_host[start..end]) |*value| {
                    value.* = @floatFromInt(token);
                }
                host_segments[row_index] = self.activation_rows[slot_index][0..];
            }
        }
    };

    var logits = [_]f32{0} ** 6;
    var root = MockRoot{
        .runtime_buffers = .{
            .projected_vocab = 3,
            .projected_logits_batch_host = logits[0..],
        },
    };
    var cpu = MockCpuStage{};
    var host_segments: [2][]const u8 = undefined;
    const tokens = [_]u32{ 17, 18 };
    const slots = [_]usize{ 0, 1 };
    const positions = [_]usize{ 10, 11 };

    try endpoint.prepareBatchedDecodeSegments(
        preserveUnexpectedDecodeBoundaryFailure,
        &root,
        &cpu,
        false,
        &root,
        .{},
        tokens[0..],
        slots[0..],
        positions[0..],
        2,
        5,
        true,
        true,
        4,
        host_segments[0..],
    );

    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, root.activated[0..root.activated_count]);
    try std.testing.expectEqual(@as(usize, 2), cpu.layer_start_seen);
    try std.testing.expectEqual(@as(usize, 5), cpu.layer_end_seen);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, host_segments[0]);
    try std.testing.expectEqualSlices(u8, &.{ 5, 6, 7, 8 }, host_segments[1]);
    try std.testing.expectEqualSlices(f32, &.{ 17, 17, 17, 18, 18, 18 }, logits[0..]);
}
