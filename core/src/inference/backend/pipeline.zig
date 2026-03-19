//! Backend-agnostic pipeline runtime for layer-parallel inference.
//!
//! Splits a model's decoder layers across two stages, each of which may
//! run on a different device or backend. Activation tensors are transferred
//! at the stage boundary.
//!
//! The orchestration pattern is generic: any type satisfying the stage
//! contract (executeLayers, downloadActivation, uploadActivation,
//! synchronize) can be used. Same-backend optimizations (e.g. CUDA P2P)
//! are injected via an optional TransferType parameter.

const std = @import("std");

/// Verify that StageType satisfies the pipeline stage contract at comptime.
fn assertStageContract(comptime S: type) void {
    const required_methods = .{
        .{ "executeLayers", 4 },
        .{ "downloadActivation", 3 },
        .{ "uploadActivation", 3 },
        .{ "synchronize", 1 },
    };
    inline for (required_methods) |entry| {
        const name = entry[0];
        const arity = entry[1];
        if (!@hasDecl(S, name)) {
            @compileError("Pipeline stage type '" ++ @typeName(S) ++
                "' missing required method '" ++ name ++ "'");
        }
        const decl = @field(S, name);
        const info = @typeInfo(@TypeOf(decl));
        if (info != .@"fn") {
            @compileError("Pipeline stage '" ++ @typeName(S) ++ "." ++ name ++
                "' must be a function");
        }
        if (info.@"fn".params.len != arity) {
            @compileError("Pipeline stage '" ++ @typeName(S) ++ "." ++ name ++
                "' must have arity " ++ std.fmt.comptimePrint("{}", .{arity}));
        }
    }
}

/// Verify that TransferType satisfies the transfer contract at comptime.
fn assertTransferContract(comptime T: type, comptime S: type) void {
    if (!@hasDecl(T, "transfer")) {
        @compileError("Pipeline transfer type '" ++ @typeName(T) ++
            "' missing required method 'transfer'");
    }
    _ = S;
}

/// Generic pipeline runtime that orchestrates two stages for layer-parallel
/// inference. Each stage executes a disjoint range of decoder layers.
///
/// StageType must provide:
///   fn executeLayers(self: *StageType, input: []const u8, layer_start: usize, layer_end: usize) !void
///   fn downloadActivation(self: *StageType, host_buf: []u8, byte_count: usize) !void
///   fn uploadActivation(self: *StageType, host_buf: []const u8, byte_count: usize) !void
///   fn synchronize(self: *StageType) !void
///
/// TransferType (optional, pass null for host-staged default) must provide:
///   fn transfer(self: *TransferType, src: *StageType, dst: *StageType, byte_count: usize) !void
pub fn PipelineRuntime(comptime StageType: type, comptime TransferType: ?type) type {
    comptime {
        assertStageContract(StageType);
        if (TransferType) |T| assertTransferContract(T, StageType);
    }

    const HasCustomTransfer = TransferType != null;

    return struct {
        const Self = @This();

        stage0: StageType,
        stage1: StageType,
        split_layer: usize,
        total_layers: usize,
        host_staging: ?[]align(64) u8,
        custom_transfer: if (HasCustomTransfer) TransferType.? else void,

        /// Execute the full forward pass across both stages.
        ///
        /// 1. Stage 0 runs layers [0, split_layer) on its device.
        /// 2. Activation is transferred from stage 0 to stage 1.
        /// 3. Stage 1 runs layers [split_layer, total_layers) on its device.
        ///
        /// `stage0_input` and `stage1_input` are opaque execution payloads for each stage.
        /// Keeping these separate prevents accidental stage-ambiguous control flags.
        /// `activation_byte_count` is the boundary activation transfer size.
        pub fn executeForward(
            self: *Self,
            stage0_input: []const u8,
            stage1_input: []const u8,
            activation_byte_count: usize,
        ) !void {
            try self.stage0.executeLayers(stage0_input, 0, self.split_layer);
            try self.stage0.synchronize();
            try self.transferActivation(activation_byte_count);
            try self.stage1.executeLayers(stage1_input, self.split_layer, self.total_layers);
        }

        fn transferActivation(self: *Self, byte_count: usize) !void {
            if (HasCustomTransfer) {
                try self.custom_transfer.transfer(&self.stage0, &self.stage1, byte_count);
            } else {
                const staging = self.host_staging orelse return error.PipelineTransferNotInitialized;
                if (byte_count > staging.len) return error.PipelineTransferBufferTooSmall;
                try self.stage0.downloadActivation(staging[0..byte_count], byte_count);
                try self.stage1.uploadActivation(staging[0..byte_count], byte_count);
            }
        }

        /// Release both stages and the transfer buffer.
        /// Stage 1 is released first to ensure correct cleanup ordering
        /// when P2P access was enabled from stage 0 to stage 1.
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            if (HasCustomTransfer) {
                if (@hasDecl(TransferType.?, "deinit")) {
                    self.custom_transfer.deinit(allocator);
                }
            }
            self.stage1.deinit(allocator);
            self.stage0.deinit(allocator);
            if (self.host_staging) |buf| {
                allocator.free(buf);
                self.host_staging = null;
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Tests using a mock stage type
// ---------------------------------------------------------------------------

const TestLog = struct {
    entries: [32]Entry = [_]Entry{.{ .kind = .none, .arg = 0 }} ** 32,
    count: usize = 0,

    const Kind = enum { execute, download, upload, sync, transfer, none };
    const Entry = struct { kind: Kind, arg: usize };

    fn append(self: *TestLog, kind: Kind, arg: usize) void {
        if (self.count < self.entries.len) {
            self.entries[self.count] = .{ .kind = kind, .arg = arg };
            self.count += 1;
        }
    }
};

const MockStage = struct {
    stage_id: usize,
    log: *TestLog,
    input_len_out: ?*usize = null,

    pub fn executeLayers(self: *MockStage, input: []const u8, layer_start: usize, layer_end: usize) !void {
        if (self.input_len_out) |out| out.* = input.len;
        _ = layer_start;
        self.log.append(.execute, layer_end);
    }

    pub fn downloadActivation(self: *MockStage, _: []u8, byte_count: usize) !void {
        self.log.append(.download, byte_count);
    }

    pub fn uploadActivation(self: *MockStage, _: []const u8, byte_count: usize) !void {
        self.log.append(.upload, byte_count);
    }

    pub fn synchronize(self: *MockStage) !void {
        self.log.append(.sync, self.stage_id);
    }

    pub fn deinit(_: *MockStage, _: std.mem.Allocator) void {}
};

const MockTransfer = struct {
    log: *TestLog,

    pub fn transfer(self: *MockTransfer, _: *MockStage, _: *MockStage, byte_count: usize) !void {
        self.log.append(.transfer, byte_count);
    }

    pub fn deinit(_: *MockTransfer, _: std.mem.Allocator) void {}
};

test "PipelineRuntime executeForward calls stages in correct order with host-staged transfer" {
    var test_log = TestLog{};
    const HostPipeline = PipelineRuntime(MockStage, null);
    var staging_buf: [256]u8 align(64) = undefined;

    var pipeline = HostPipeline{
        .stage0 = .{ .stage_id = 0, .log = &test_log },
        .stage1 = .{ .stage_id = 1, .log = &test_log },
        .split_layer = 16,
        .total_layers = 32,
        .host_staging = &staging_buf,
        .custom_transfer = {},
    };

    var dummy_input: [64]u8 = [_]u8{0} ** 64;
    try pipeline.executeForward(dummy_input[0..], dummy_input[0..], 64);

    // Expected order: stage0.execute → stage0.sync → download → upload → stage1.execute
    try std.testing.expectEqual(@as(usize, 5), test_log.count);
    try std.testing.expectEqual(TestLog.Kind.execute, test_log.entries[0].kind);
    try std.testing.expectEqual(@as(usize, 16), test_log.entries[0].arg);
    try std.testing.expectEqual(TestLog.Kind.sync, test_log.entries[1].kind);
    try std.testing.expectEqual(@as(usize, 0), test_log.entries[1].arg);
    try std.testing.expectEqual(TestLog.Kind.download, test_log.entries[2].kind);
    try std.testing.expectEqual(@as(usize, 64), test_log.entries[2].arg);
    try std.testing.expectEqual(TestLog.Kind.upload, test_log.entries[3].kind);
    try std.testing.expectEqual(@as(usize, 64), test_log.entries[3].arg);
    try std.testing.expectEqual(TestLog.Kind.execute, test_log.entries[4].kind);
    try std.testing.expectEqual(@as(usize, 32), test_log.entries[4].arg);
}

test "PipelineRuntime executeForward calls custom transfer when provided" {
    var test_log = TestLog{};
    const CustomPipeline = PipelineRuntime(MockStage, MockTransfer);

    var pipeline = CustomPipeline{
        .stage0 = .{ .stage_id = 0, .log = &test_log },
        .stage1 = .{ .stage_id = 1, .log = &test_log },
        .split_layer = 8,
        .total_layers = 24,
        .host_staging = null,
        .custom_transfer = .{ .log = &test_log },
    };

    var dummy_input: [128]u8 = [_]u8{0} ** 128;
    try pipeline.executeForward(dummy_input[0..], dummy_input[0..], 128);

    // Expected: stage0.execute → stage0.sync → custom_transfer → stage1.execute
    try std.testing.expectEqual(@as(usize, 4), test_log.count);
    try std.testing.expectEqual(TestLog.Kind.execute, test_log.entries[0].kind);
    try std.testing.expectEqual(TestLog.Kind.sync, test_log.entries[1].kind);
    try std.testing.expectEqual(TestLog.Kind.transfer, test_log.entries[2].kind);
    try std.testing.expectEqual(@as(usize, 128), test_log.entries[2].arg);
    try std.testing.expectEqual(TestLog.Kind.execute, test_log.entries[3].kind);
}

test "PipelineRuntime transferActivation returns error when host staging is null and no custom transfer" {
    var test_log = TestLog{};
    const HostPipeline = PipelineRuntime(MockStage, null);

    var pipeline = HostPipeline{
        .stage0 = .{ .stage_id = 0, .log = &test_log },
        .stage1 = .{ .stage_id = 1, .log = &test_log },
        .split_layer = 4,
        .total_layers = 8,
        .host_staging = null,
        .custom_transfer = {},
    };

    const result = pipeline.transferActivation(64);
    try std.testing.expectError(error.PipelineTransferNotInitialized, result);
}

test "PipelineRuntime transferActivation returns error when byte_count exceeds staging buffer" {
    var test_log = TestLog{};
    const HostPipeline = PipelineRuntime(MockStage, null);
    var staging_buf: [32]u8 align(64) = undefined;

    var pipeline = HostPipeline{
        .stage0 = .{ .stage_id = 0, .log = &test_log },
        .stage1 = .{ .stage_id = 1, .log = &test_log },
        .split_layer = 4,
        .total_layers = 8,
        .host_staging = &staging_buf,
        .custom_transfer = {},
    };

    const result = pipeline.transferActivation(64);
    try std.testing.expectError(error.PipelineTransferBufferTooSmall, result);
}

test "PipelineRuntime executeForward forwards stage-specific payloads" {
    var test_log = TestLog{};
    var stage0_input_len: usize = 0;
    var stage1_input_len: usize = 0;
    const HostPipeline = PipelineRuntime(MockStage, null);
    var staging_buf: [64]u8 align(64) = undefined;
    var stage0_payload: [12]u8 = [_]u8{0} ** 12;
    var stage1_payload: [24]u8 = [_]u8{0} ** 24;

    var pipeline = HostPipeline{
        .stage0 = .{ .stage_id = 0, .log = &test_log, .input_len_out = &stage0_input_len },
        .stage1 = .{ .stage_id = 1, .log = &test_log, .input_len_out = &stage1_input_len },
        .split_layer = 4,
        .total_layers = 8,
        .host_staging = &staging_buf,
        .custom_transfer = {},
    };

    try pipeline.executeForward(stage0_payload[0..], stage1_payload[0..], 32);
    try std.testing.expectEqual(stage0_payload.len, stage0_input_len);
    try std.testing.expectEqual(stage1_payload.len, stage1_input_len);
}
