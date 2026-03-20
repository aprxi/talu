//! Backend-agnostic staged forward orchestrator.
//!
//! Provides a thin generic entrypoint that executes a two-stage forward pass
//! via PipelineRuntime. Backends provide stage adapters + optional transfer.

const std = @import("std");
const pipeline = @import("pipeline.zig");

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
