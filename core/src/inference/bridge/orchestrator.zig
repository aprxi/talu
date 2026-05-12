//! Backend-agnostic staged forward orchestrator.
//!
//! Provides a thin generic entrypoint that executes a two-stage forward pass
//! via PipelineRuntime. Backends provide stage adapters + optional transfer.

const std = @import("std");
const pipeline = @import("pipeline.zig");
const tensor_frame = @import("tensor_frame.zig");

pub const LocalDecodeHandoffConfig = struct {
    plan_ref: *const tensor_frame.TensorFramePlanRef,
    boundary_index: usize,
    metadata: tensor_frame.TensorFrameMetadata,
    activation_byte_count: usize,
    host_staging: ?[]align(64) u8,
    stage0_input: []const u8 = &.{},
    stage1_input: []const u8 = &.{},
    observer: tensor_frame.TensorFrameObserver = .{},
    observer_mode: tensor_frame.TensorFrameObserverMode = .best_effort,
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
    try tensor_frame.validateTensorFrameForPlanBoundary(&config.metadata, config.plan_ref, config.boundary_index);
    try tensor_frame.validatePayloadBufferLength(&config.metadata, config.activation_byte_count);
    if (config.metadata.boundary.producer_layer_start != 0) return error.InvalidProducerLayerRange;

    const split_layer: usize = config.metadata.boundary.consumer_layer_start;
    const total_layers: usize = config.metadata.boundary.consumer_layer_end;

    const Transfer = struct {
        metadata: tensor_frame.TensorFrameMetadata,
        host_staging: ?[]align(64) u8,
        observer: tensor_frame.TensorFrameObserver,
        observer_mode: tensor_frame.TensorFrameObserverMode,

        pub fn transfer(transfer_ctx: *@This(), src: *Stage0Type, dst: *Stage1Type, byte_count: usize) anyerror!void {
            const staging = transfer_ctx.host_staging orelse return error.PipelineTransferNotInitialized;
            if (byte_count > staging.len) return error.PipelineTransferBufferTooSmall;
            try src.downloadActivation(staging[0..byte_count], byte_count);
            try dst.uploadActivation(staging[0..byte_count], byte_count);
            try tensor_frame.emitTensorFrame(transfer_ctx.observer, transfer_ctx.observer_mode, &transfer_ctx.metadata);
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
        .{
            .metadata = config.metadata,
            .host_staging = config.host_staging,
            .observer = config.observer,
            .observer_mode = config.observer_mode,
        },
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

const observer_test_batch = [_]tensor_frame.TensorFrameBatchEntry{.{
    .batch_index = 0,
    .request_id = 42,
    .slot_id = 7,
    .sequence_start = 7,
    .token_count = 1,
}};

const observer_test_boundaries = [_]tensor_frame.TensorFrameBoundaryRef{.{
    .boundary_index = 0,
    .source_stage_id = 0,
    .target_stage_id = 1,
    .producer_layer_start = 0,
    .producer_layer_end = 2,
    .consumer_layer_start = 2,
    .consumer_layer_end = 4,
}};

const mismatched_observer_test_boundaries = [_]tensor_frame.TensorFrameBoundaryRef{.{
    .boundary_index = 0,
    .source_stage_id = 0,
    .target_stage_id = 1,
    .producer_layer_start = 0,
    .producer_layer_end = 2,
    .consumer_layer_start = 2,
    .consumer_layer_end = 4,
}};

fn testObserverPlanRef() tensor_frame.TensorFramePlanRef {
    return .{
        .allocator = std.testing.allocator,
        .identity = .{
            .graph_digest = [_]u8{1} ** 32,
            .graph_contract_version = 1,
            .stage_plan_contract_version = 1,
            .stage_plan_id = .{ .digest = [_]u8{2} ** 32 },
        },
        .boundaries = &observer_test_boundaries,
    };
}

fn testObserverFrame(plan_ref: *const tensor_frame.TensorFramePlanRef) !tensor_frame.TensorFrameMetadata {
    const boundary = try plan_ref.boundary(0);
    const tensor = try tensor_frame.TensorFrameTensorDesc.contiguousActivation(.f32, .{ 1, 1, 8, 0 });
    return .{
        .frame_id = try tensor_frame.TensorFrameInstanceId.init(1),
        .plan = plan_ref.identity,
        .boundary = boundary,
        .selected_contract = .{
            .boundary = boundary,
            .dtype = .f32,
            .layout = .row_major,
            .source = .explicit,
        },
        .role = .activation,
        .step_kind = .decode,
        .shape_context = .{ .expected_hidden_size = 8 },
        .tensor = tensor,
        .batch = .{ .entries = &observer_test_batch },
        .payload = .{
            .byte_count = tensor.payload_byte_count,
            .location_hint = .{ .cuda = 0 },
            .ownership = .borrowed_until_next_stage_call,
            .lifetime = .step_scoped,
        },
    };
}

fn mismatchedObserverPlanRef() tensor_frame.TensorFramePlanRef {
    return .{
        .allocator = std.testing.allocator,
        .identity = .{
            .graph_digest = [_]u8{1} ** 32,
            .graph_contract_version = 1,
            .stage_plan_contract_version = 1,
            .stage_plan_id = .{ .digest = [_]u8{3} ** 32 },
        },
        .boundaries = &mismatched_observer_test_boundaries,
    };
}

test "inference bridge executeTwoStageForward runs both stages via custom transfer" {
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

test "inference bridge executeLocalDecodeHandoff runs stages through metadata validated transfer" {
    var log = TestLog{};
    var staging: [64]u8 align(64) = undefined;
    const plan_ref = testObserverPlanRef();
    const metadata = try testObserverFrame(&plan_ref);

    try executeLocalDecodeHandoff(
        MockStage,
        MockStage,
        .{ .id = 0, .log = &log },
        .{ .id = 1, .log = &log },
        .{
            .plan_ref = &plan_ref,
            .boundary_index = 0,
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

test "inference bridge executeLocalDecodeHandoff emits generic observer frame" {
    const Capture = struct {
        count: usize = 0,

        fn emit(ctx: ?*anyopaque, metadata: *const tensor_frame.TensorFrameMetadata) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            try metadata.validate();
            self.count += 1;
        }
    };

    var log = TestLog{};
    var capture = Capture{};
    var staging: [64]u8 align(64) = undefined;
    const plan_ref = testObserverPlanRef();
    const frame = try testObserverFrame(&plan_ref);

    try executeLocalDecodeHandoff(
        MockStage,
        MockStage,
        .{ .id = 0, .log = &log },
        .{ .id = 1, .log = &log },
        .{
            .plan_ref = &plan_ref,
            .boundary_index = 0,
            .metadata = frame,
            .activation_byte_count = 32,
            .host_staging = staging[0..],
            .observer = .{ .ctx = &capture, .emit_fn = Capture.emit },
            .observer_mode = .strict,
        },
    );

    try std.testing.expectEqual(@as(usize, 1), capture.count);
}

test "inference bridge executeLocalDecodeHandoff rejects stale byte count before transfer" {
    var log = TestLog{};
    var staging: [64]u8 align(64) = undefined;
    const plan_ref = testObserverPlanRef();
    const metadata = try testObserverFrame(&plan_ref);

    try std.testing.expectError(
        error.PayloadBufferLengthMismatch,
        executeLocalDecodeHandoff(
            MockStage,
            MockStage,
            .{ .id = 0, .log = &log },
            .{ .id = 1, .log = &log },
            .{
                .plan_ref = &plan_ref,
                .boundary_index = 0,
                .metadata = metadata,
                .activation_byte_count = 36,
                .host_staging = staging[0..],
            },
        ),
    );

    try std.testing.expectEqual(@as(usize, 0), log.count);
}

test "inference bridge executeLocalDecodeHandoff rejects unsupported producer layer start" {
    var log = TestLog{};
    var staging: [64]u8 align(64) = undefined;
    const plan_ref = testObserverPlanRef();
    var metadata = try testObserverFrame(&plan_ref);
    metadata.boundary.producer_layer_start = 1;
    metadata.selected_contract.boundary.producer_layer_start = 1;

    try std.testing.expectError(
        error.InvalidProducerLayerRange,
        executeLocalDecodeHandoff(
            MockStage,
            MockStage,
            .{ .id = 0, .log = &log },
            .{ .id = 1, .log = &log },
            .{
                .plan_ref = &plan_ref,
                .boundary_index = 0,
                .metadata = metadata,
                .activation_byte_count = 32,
                .host_staging = staging[0..],
            },
        ),
    );
    try std.testing.expectEqual(@as(usize, 0), log.count);
}

test "inference bridge executeLocalDecodeHandoff rejects self-consistent fake plan identity" {
    var log = TestLog{};
    var staging: [64]u8 align(64) = undefined;
    const plan_ref = testObserverPlanRef();
    const fake_plan_ref = mismatchedObserverPlanRef();
    const metadata = try testObserverFrame(&fake_plan_ref);

    try std.testing.expectError(
        error.StagePlanIdentityMismatch,
        executeLocalDecodeHandoff(
            MockStage,
            MockStage,
            .{ .id = 0, .log = &log },
            .{ .id = 1, .log = &log },
            .{
                .plan_ref = &plan_ref,
                .boundary_index = 0,
                .metadata = metadata,
                .activation_byte_count = 32,
                .host_staging = staging[0..],
            },
        ),
    );
    try std.testing.expectEqual(@as(usize, 0), log.count);
}

test "inference bridge executeThreeStageForward runs all stages through both transfers" {
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
