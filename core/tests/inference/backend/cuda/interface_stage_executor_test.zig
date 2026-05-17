//! Tests for CUDA backend stage executor interface forwarding.

const std = @import("std");
const main = @import("main");

const stage_executor = main.inference.backend.cuda.interface.stage_executor;

test "executeDecodeLayerRange forwards CUDA decode controls" {
    const MockCudaBackend = struct {
        token_seen: u32 = 0,
        position_seen: usize = 0,
        slot_seen: usize = 0,
        layer_start_seen: usize = 0,
        layer_end_seen: usize = 0,
        compute_logits_seen: bool = false,
        download_logits_seen: bool = false,
        ensure_kv_capacity_seen: bool = false,
        use_preloaded_seen: bool = false,
        logits_len_seen: usize = 0,

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
            self.token_seen = token;
            self.position_seen = position;
            self.slot_seen = slot_index;
            self.layer_start_seen = layer_start;
            self.layer_end_seen = layer_end;
            self.compute_logits_seen = compute_logits;
            self.download_logits_seen = download_logits;
            self.ensure_kv_capacity_seen = ensure_kv_capacity;
            self.use_preloaded_seen = use_preloaded_input;
            self.logits_len_seen = if (logits_out_opt) |logits| logits.len else 0;
        }
    };
    const DecodeContext = struct {
        token: u32,
        position: usize,
        slot_index: usize,
        ensure_kv_capacity: bool,
    };

    var backend = MockCudaBackend{};
    var logits = [_]f32{0} ** 6;
    try stage_executor.executeDecodeLayerRange(
        &backend,
        DecodeContext{
            .token = 17,
            .position = 23,
            .slot_index = 3,
            .ensure_kv_capacity = true,
        },
        4,
        7,
        logits[0..],
        true,
        true,
        true,
    );

    try std.testing.expectEqual(@as(u32, 17), backend.token_seen);
    try std.testing.expectEqual(@as(usize, 23), backend.position_seen);
    try std.testing.expectEqual(@as(usize, 3), backend.slot_seen);
    try std.testing.expectEqual(@as(usize, 4), backend.layer_start_seen);
    try std.testing.expectEqual(@as(usize, 7), backend.layer_end_seen);
    try std.testing.expect(backend.compute_logits_seen);
    try std.testing.expect(backend.download_logits_seen);
    try std.testing.expect(backend.ensure_kv_capacity_seen);
    try std.testing.expect(backend.use_preloaded_seen);
    try std.testing.expectEqual(@as(usize, 6), backend.logits_len_seen);
}

test "executePrefillLayerRange forwards CUDA prefill controls" {
    const MockCudaBackend = struct {
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

    var backend = MockCudaBackend{};
    var logits = [_]f32{0} ** 8;
    var embeddings = [_]f32{0} ** 12;
    try stage_executor.executePrefillLayerRange(
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
