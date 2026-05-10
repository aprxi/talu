//! Backend-agnostic staged forward orchestrator.
//!
//! Provides a thin generic entrypoint that executes a two-stage forward pass
//! via PipelineRuntime. Backends provide stage adapters + optional transfer.

const std = @import("std");
const pipeline = @import("pipeline.zig");
const tensor_frame = @import("tensor_frame.zig");
const xray_bridge = @import("../diagnostics/xray_bridge.zig");

pub const LocalDecodeHandoffConfig = struct {
    metadata: tensor_frame.TensorFrameMetadata,
    activation_byte_count: usize,
    host_staging: ?[]align(64) u8,
    stage0_input: []const u8 = &.{},
    stage1_input: []const u8 = &.{},
};

pub fn executeTwoStageForward(
    comptime Stage0Type: type,
    comptime Stage1Type: type,
    comptime TransferType: ?type,
    stage0: Stage0Type,
    stage1: Stage1Type,
    split_layer: usize,
    total_layers: usize,
    stage0_input: []const u8,
    stage1_input: []const u8,
    activation_byte_count: usize,
    host_staging: ?[]align(64) u8,
    custom_transfer: if (TransferType != null) TransferType.? else void,
) !void {
    const Runtime = pipeline.PipelineRuntime(Stage0Type, Stage1Type, TransferType);
    var runtime = Runtime{
        .stage0 = stage0,
        .stage1 = stage1,
        .split_layer = split_layer,
        .total_layers = total_layers,
        .host_staging = host_staging,
        .custom_transfer = custom_transfer,
    };
    try runtime.executeForward(stage0_input, stage1_input, activation_byte_count);
}

pub fn executeLocalDecodeHandoff(
    comptime Stage0Type: type,
    comptime Stage1Type: type,
    stage0: Stage0Type,
    stage1: Stage1Type,
    config: LocalDecodeHandoffConfig,
) !void {
    try config.metadata.validate();
    if (config.metadata.role != .activation) return error.InvalidStageBoundary;
    if (config.metadata.boundary.producer_layer_start != 0) return error.InvalidLayerRange;

    const split_layer: usize = config.metadata.boundary.consumer_layer_start;
    const total_layers: usize = config.metadata.boundary.consumer_layer_end;

    const Transfer = struct {
        metadata: tensor_frame.TensorFrameMetadata,
        host_staging: ?[]align(64) u8,

        pub fn transfer(transfer_ctx: *@This(), src: *Stage0Type, dst: *Stage1Type, byte_count: usize) anyerror!void {
            try tensor_frame.validateActivationFrameByteCount(&transfer_ctx.metadata, @intCast(byte_count));
            const staging = transfer_ctx.host_staging orelse return error.PipelineTransferNotInitialized;
            if (byte_count > staging.len) return error.PipelineTransferBufferTooSmall;
            try src.downloadActivation(staging[0..byte_count], byte_count);
            try dst.uploadActivation(staging[0..byte_count], byte_count);
            xray_bridge.emitActivationHandoffLayerInput(&transfer_ctx.metadata, staging.ptr);
        }

        pub fn deinit(_: *@This(), _: std.mem.Allocator) void {}
    };

    try executeTwoStageForward(
        Stage0Type,
        Stage1Type,
        Transfer,
        stage0,
        stage1,
        split_layer,
        total_layers,
        config.stage0_input,
        config.stage1_input,
        config.activation_byte_count,
        null,
        .{ .metadata = config.metadata, .host_staging = config.host_staging },
    );
}

pub fn executeThreeStageForward(
    comptime Stage0Type: type,
    comptime Stage1Type: type,
    comptime Stage2Type: type,
    stage0: Stage0Type,
    stage1: Stage1Type,
    stage2: Stage2Type,
    split_layer_01: usize,
    split_layer_12: usize,
    total_layers: usize,
    stage0_input: []const u8,
    stage1_input: []const u8,
    stage2_input: []const u8,
    activation01_byte_count: usize,
    activation12_byte_count: usize,
    host_staging_01: ?[]align(64) u8,
    host_staging_12: ?[]align(64) u8,
) !void {
    const Runtime = pipeline.PipelineRuntime3(Stage0Type, Stage1Type, Stage2Type);
    var runtime = Runtime{
        .stage0 = stage0,
        .stage1 = stage1,
        .stage2 = stage2,
        .split_layer_01 = split_layer_01,
        .split_layer_12 = split_layer_12,
        .total_layers = total_layers,
        .host_staging_01 = host_staging_01,
        .host_staging_12 = host_staging_12,
    };
    try runtime.executeForward(
        stage0_input,
        stage1_input,
        stage2_input,
        activation01_byte_count,
        activation12_byte_count,
    );
}

const TestLog = struct {
    entries: [16]Entry = [_]Entry{.{ .kind = .none, .arg = 0 }} ** 16,
    count: usize = 0,

    const Kind = enum { execute, sync, download, upload, transfer, none };
    const Entry = struct { kind: Kind, arg: usize };

    fn push(self: *TestLog, kind: Kind, arg: usize) void {
        self.entries[self.count] = .{ .kind = kind, .arg = arg };
        self.count += 1;
    }
};

const MockStage = struct {
    id: usize,
    log: *TestLog,

    pub fn executeLayers(self: *MockStage, _: []const u8, _: usize, layer_end: usize) !void {
        self.log.push(.execute, layer_end + self.id);
    }

    pub fn downloadActivation(self: *MockStage, _: []u8, byte_count: usize) !void {
        self.log.push(.download, byte_count + self.id);
    }

    pub fn uploadActivation(self: *MockStage, _: []const u8, byte_count: usize) !void {
        self.log.push(.upload, byte_count + self.id);
    }

    pub fn synchronize(self: *MockStage) !void {
        self.log.push(.sync, self.id);
    }

    pub fn deinit(_: *MockStage, _: std.mem.Allocator) void {}
};

const MockTransfer = struct {
    log: *TestLog,

    pub fn transfer(self: *MockTransfer, _: *MockStage, _: *MockStage, byte_count: usize) !void {
        self.log.push(.transfer, byte_count);
    }

    pub fn deinit(_: *MockTransfer, _: std.mem.Allocator) void {}
};

test "executeTwoStageForward runs both stages via custom transfer" {
    var log = TestLog{};
    try executeTwoStageForward(
        MockStage,
        MockStage,
        MockTransfer,
        .{ .id = 0, .log = &log },
        .{ .id = 1, .log = &log },
        2,
        4,
        &.{},
        &.{},
        32,
        null,
        .{ .log = &log },
    );

    try std.testing.expectEqual(@as(usize, 4), log.count);
    try std.testing.expectEqual(TestLog.Kind.execute, log.entries[0].kind);
    try std.testing.expectEqual(TestLog.Kind.sync, log.entries[1].kind);
    try std.testing.expectEqual(TestLog.Kind.transfer, log.entries[2].kind);
    try std.testing.expectEqual(TestLog.Kind.execute, log.entries[3].kind);
}

test "executeLocalDecodeHandoff runs stages through metadata validated transfer" {
    var log = TestLog{};
    var staging: [64]u8 align(64) = undefined;
    const metadata = try tensor_frame.activationHandoffFrame(.{
        .frame_id = 1,
        .graph_id = 2,
        .request_id = 3,
        .source = .{ .stage_id = 0, .backend = .cpu },
        .target = .{ .stage_id = 1, .backend = .cuda },
        .producer_layer_start = 0,
        .producer_layer_end = 2,
        .consumer_layer_start = 2,
        .consumer_layer_end = 4,
        .dtype = .f32,
        .shape = try tensor_frame.TensorFrameShape.contiguous(3, .{ 1, 1, 8, 0 }),
        .device = .{ .cuda = 0 },
        .sequence_start = 7,
        .sequence_len = 1,
        .batch_size = 1,
        .slot_index = 0,
    });

    try executeLocalDecodeHandoff(
        MockStage,
        MockStage,
        .{ .id = 0, .log = &log },
        .{ .id = 1, .log = &log },
        .{
            .metadata = metadata,
            .activation_byte_count = 32,
            .host_staging = staging[0..],
        },
    );

    try std.testing.expectEqual(@as(usize, 5), log.count);
    try std.testing.expectEqual(TestLog.Kind.execute, log.entries[0].kind);
    try std.testing.expectEqual(@as(usize, 2), log.entries[0].arg);
    try std.testing.expectEqual(TestLog.Kind.sync, log.entries[1].kind);
    try std.testing.expectEqual(TestLog.Kind.download, log.entries[2].kind);
    try std.testing.expectEqual(@as(usize, 32), log.entries[2].arg);
    try std.testing.expectEqual(TestLog.Kind.upload, log.entries[3].kind);
    try std.testing.expectEqual(@as(usize, 33), log.entries[3].arg);
    try std.testing.expectEqual(TestLog.Kind.execute, log.entries[4].kind);
    try std.testing.expectEqual(@as(usize, 5), log.entries[4].arg);
}

test "executeLocalDecodeHandoff emits xray layer input from bridge metadata when requested" {
    const trace = @import("xray_pkg").trace;
    const Capture = struct {
        var count: usize = 0;
        var last: ?trace.TraceEmission = null;

        fn handler(emission: trace.TraceEmission) void {
            count += 1;
            last = emission;
        }

        fn reset() void {
            count = 0;
            last = null;
        }
    };

    Capture.reset();
    trace.setHandler(&Capture.handler);
    trace.setActiveBuiltInPointMask(@as(u64, 1) << @intFromEnum(trace.TracePoint.layer_input));
    trace.setActiveExactEmissionFilter(null);
    defer {
        trace.setActiveBuiltInPointMask(0);
        trace.setActiveExactEmissionFilter(null);
        trace.setHandler(null);
    }

    var log = TestLog{};
    var staging: [64]u8 align(64) = undefined;
    const metadata = try tensor_frame.activationHandoffFrame(.{
        .frame_id = 1,
        .graph_id = 2,
        .request_id = 3,
        .source = .{ .stage_id = 0, .backend = .cpu },
        .target = .{ .stage_id = 1, .backend = .cuda },
        .producer_layer_start = 0,
        .producer_layer_end = 2,
        .consumer_layer_start = 2,
        .consumer_layer_end = 4,
        .dtype = .f32,
        .shape = try tensor_frame.TensorFrameShape.contiguous(3, .{ 1, 1, 8, 0 }),
        .device = .{ .cuda = 0 },
        .sequence_start = 7,
        .sequence_len = 1,
        .batch_size = 1,
        .slot_index = 0,
    });

    try executeLocalDecodeHandoff(
        MockStage,
        MockStage,
        .{ .id = 0, .log = &log },
        .{ .id = 1, .log = &log },
        .{
            .metadata = metadata,
            .activation_byte_count = 32,
            .host_staging = staging[0..],
        },
    );

    if (!trace.isEnabled()) {
        try std.testing.expectEqual(@as(usize, 0), Capture.count);
        return;
    }

    try std.testing.expectEqual(@as(usize, 1), Capture.count);
    const emission = Capture.last.?;
    try std.testing.expectEqual(trace.TracePoint.layer_input, emission.point);
    try std.testing.expectEqual(@as(u16, 2), emission.layer);
    try std.testing.expectEqual(@as(u32, 0), emission.token);
    try std.testing.expectEqual(@as(u32, 1), emission.position);
    try std.testing.expectEqual(trace.Backend.cuda, emission.backend);
    try std.testing.expectEqual(trace.DType.f32, emission.tensor.dtype);
    try std.testing.expectEqual(@as(u8, 3), emission.tensor.ndim);
    try std.testing.expectEqual([4]u32{ 1, 1, 8, 0 }, emission.tensor.shape);
    try std.testing.expectEqual(@as(u64, 32), emission.work_bytes);
}

test "executeLocalDecodeHandoff rejects stale byte count before transfer" {
    var log = TestLog{};
    var staging: [64]u8 align(64) = undefined;
    const metadata = try tensor_frame.activationHandoffFrame(.{
        .frame_id = 1,
        .graph_id = 2,
        .request_id = 3,
        .source = .{ .stage_id = 0, .backend = .cpu },
        .target = .{ .stage_id = 1, .backend = .cuda },
        .producer_layer_start = 0,
        .producer_layer_end = 2,
        .consumer_layer_start = 2,
        .consumer_layer_end = 4,
        .dtype = .f32,
        .shape = try tensor_frame.TensorFrameShape.contiguous(3, .{ 1, 1, 8, 0 }),
        .device = .{ .cuda = 0 },
        .sequence_start = 7,
        .sequence_len = 1,
        .batch_size = 1,
        .slot_index = 0,
    });

    try std.testing.expectError(
        error.InvalidTensorByteCount,
        executeLocalDecodeHandoff(
            MockStage,
            MockStage,
            .{ .id = 0, .log = &log },
            .{ .id = 1, .log = &log },
            .{
                .metadata = metadata,
                .activation_byte_count = 36,
                .host_staging = staging[0..],
            },
        ),
    );

    try std.testing.expectEqual(@as(usize, 2), log.count);
    try std.testing.expectEqual(TestLog.Kind.execute, log.entries[0].kind);
    try std.testing.expectEqual(TestLog.Kind.sync, log.entries[1].kind);
}

test "executeLocalDecodeHandoff rejects unsupported producer layer start" {
    var log = TestLog{};
    var staging: [64]u8 align(64) = undefined;
    const metadata = try tensor_frame.activationHandoffFrame(.{
        .frame_id = 1,
        .graph_id = 2,
        .request_id = 3,
        .source = .{ .stage_id = 0, .backend = .cpu },
        .target = .{ .stage_id = 1, .backend = .cuda },
        .producer_layer_start = 1,
        .producer_layer_end = 2,
        .consumer_layer_start = 2,
        .consumer_layer_end = 4,
        .dtype = .f32,
        .shape = try tensor_frame.TensorFrameShape.contiguous(3, .{ 1, 1, 8, 0 }),
        .device = .{ .cuda = 0 },
        .sequence_start = 7,
        .sequence_len = 1,
        .batch_size = 1,
        .slot_index = 0,
    });

    try std.testing.expectError(
        error.InvalidLayerRange,
        executeLocalDecodeHandoff(
            MockStage,
            MockStage,
            .{ .id = 0, .log = &log },
            .{ .id = 1, .log = &log },
            .{
                .metadata = metadata,
                .activation_byte_count = 32,
                .host_staging = staging[0..],
            },
        ),
    );
    try std.testing.expectEqual(@as(usize, 0), log.count);
}

test "executeThreeStageForward runs all stages through both transfers" {
    var log = TestLog{};
    var staging01: [64]u8 align(64) = undefined;
    var staging12: [64]u8 align(64) = undefined;
    try executeThreeStageForward(
        MockStage,
        MockStage,
        MockStage,
        .{ .id = 0, .log = &log },
        .{ .id = 1, .log = &log },
        .{ .id = 2, .log = &log },
        2,
        4,
        6,
        &.{},
        &.{},
        &.{},
        32,
        48,
        staging01[0..],
        staging12[0..],
    );

    try std.testing.expectEqual(@as(usize, 9), log.count);
    try std.testing.expectEqual(TestLog.Kind.execute, log.entries[0].kind);
    try std.testing.expectEqual(TestLog.Kind.sync, log.entries[1].kind);
    try std.testing.expectEqual(TestLog.Kind.download, log.entries[2].kind);
    try std.testing.expectEqual(TestLog.Kind.upload, log.entries[3].kind);
    try std.testing.expectEqual(TestLog.Kind.execute, log.entries[4].kind);
    try std.testing.expectEqual(TestLog.Kind.sync, log.entries[5].kind);
    try std.testing.expectEqual(TestLog.Kind.download, log.entries[6].kind);
    try std.testing.expectEqual(TestLog.Kind.upload, log.entries[7].kind);
    try std.testing.expectEqual(TestLog.Kind.execute, log.entries[8].kind);
}
