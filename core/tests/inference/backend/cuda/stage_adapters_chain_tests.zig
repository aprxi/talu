//! Local stage-chain adapter integration tests.

const std = @import("std");
const main = @import("main");

const bridge = main.inference.bridge;
const transport = main.inference.transport;
const models = main.models.dispatcher;
const cuda_testing = main.inference.backend.cuda.testing;
const engine = cuda_testing.engine;
const stage_adapters = cuda_testing.exec.stage_adapters;

test "peerCopyCudaActivation issues event ordered device peer copy" {
    const Trace = struct {
        record_calls: usize = 0,
        wait_calls: usize = 0,
        make_current_calls: usize = 0,
        peer_calls: usize = 0,
        peer_bytes: usize = 0,
        peer_stream: ?usize = null,
        sync_calls: usize = 0,
    };
    const Buffer = struct {};
    const Device = struct {
        trace: *Trace,

        pub fn canAccessPeer(_: *@This(), _: *@This()) bool {
            return true;
        }

        pub fn enablePeerAccess(_: *@This(), _: *@This()) !void {}

        pub fn recordEvent(device: *@This(), event: usize, stream: ?usize) !void {
            try std.testing.expectEqual(@as(usize, 99), event);
            try std.testing.expectEqual(@as(?usize, 11), stream);
            device.trace.record_calls += 1;
        }

        pub fn streamWaitEvent(device: *@This(), stream: ?usize, event: usize) !void {
            try std.testing.expectEqual(@as(?usize, 22), stream);
            try std.testing.expectEqual(@as(usize, 99), event);
            device.trace.wait_calls += 1;
        }

        pub fn makeCurrent(device: *@This()) !void {
            device.trace.make_current_calls += 1;
        }

        pub fn memcpyPeerAsync(device: *@This(), _: *@This(), _: *Buffer, _: *Buffer, byte_count: usize, stream: ?usize) !void {
            device.trace.peer_calls += 1;
            device.trace.peer_bytes = byte_count;
            device.trace.peer_stream = stream;
        }

        pub fn synchronizeStream(device: *@This(), _: usize) !void {
            device.trace.sync_calls += 1;
        }

        pub fn synchronize(device: *@This()) !void {
            device.trace.sync_calls += 1;
        }
    };
    const Backend = struct {
        const RuntimeBuffers = struct {
            input_dev: Buffer = .{},
        };

        device: Device,
        runtime_buffers: RuntimeBuffers = .{},
        compute_stream: ?usize,
        local_stage_peer_copy_event: ?usize = null,
    };

    var trace_data = Trace{};
    var source = Backend{
        .device = .{ .trace = &trace_data },
        .compute_stream = 11,
        .local_stage_peer_copy_event = 99,
    };
    var target = Backend{
        .device = .{ .trace = &trace_data },
        .compute_stream = 22,
    };

    try transport.peerCopyCudaActivation(&source, &target, 64, .source_event_target_stream);

    try std.testing.expectEqual(@as(usize, 1), trace_data.record_calls);
    try std.testing.expectEqual(@as(usize, 1), trace_data.wait_calls);
    try std.testing.expectEqual(@as(usize, 1), trace_data.make_current_calls);
    try std.testing.expectEqual(@as(usize, 1), trace_data.peer_calls);
    try std.testing.expectEqual(@as(usize, 64), trace_data.peer_bytes);
    try std.testing.expectEqual(@as(?usize, 22), trace_data.peer_stream);
    try std.testing.expectEqual(@as(usize, 0), trace_data.sync_calls);
    try std.testing.expect(transport.peerCopyCudaActivationHandlesStageSync(&source, .source_event_target_stream));
}

test "peerCopyCudaActivationHandlesStageSync reports only event-synchronized copies" {
    const Backend = struct {
        local_stage_peer_copy_event: ?usize = null,
    };
    var without_event = Backend{};
    var with_event = Backend{ .local_stage_peer_copy_event = 99 };

    try std.testing.expect(!transport.peerCopyCudaActivationHandlesStageSync(&without_event, .source_event_target_stream));
    try std.testing.expect(transport.peerCopyCudaActivationHandlesStageSync(&with_event, .source_event_target_stream));
    try std.testing.expect(!transport.peerCopyCudaActivationHandlesStageSync(&with_event, .source_stream));
}

test "localCudaPeerCopyAvailable matches bridge adapter peer-copy direction" {
    const Trace = struct {
        source_enable_calls: usize = 0,
        target_enable_calls: usize = 0,
        probe_calls: usize = 0,
    };
    const Device = struct {
        trace: *Trace,
        is_target: bool,
        can_access_peer: bool,

        pub fn canAccessPeer(device: *@This(), _: *@This()) bool {
            return device.can_access_peer;
        }

        pub fn enablePeerAccess(device: *@This(), _: *@This()) !void {
            if (device.is_target) {
                device.trace.target_enable_calls += 1;
            } else {
                device.trace.source_enable_calls += 1;
            }
        }
    };
    const Backend = struct {
        device: Device,
        probe_result: bool = true,

        pub fn probeLocalCudaPeerCopy(backend: *@This(), _: *@This()) bool {
            backend.device.trace.probe_calls += 1;
            return backend.probe_result;
        }
    };

    var trace_data = Trace{};
    var source = Backend{
        .device = .{ .trace = &trace_data, .is_target = false, .can_access_peer = true },
    };
    var target = Backend{
        .device = .{ .trace = &trace_data, .is_target = true, .can_access_peer = false },
    };

    try std.testing.expect(!engine.testing.testLocalCudaPeerCopyAvailable(&source, &target));
    try std.testing.expectEqual(@as(usize, 0), trace_data.probe_calls);

    target.device.can_access_peer = true;
    try std.testing.expect(engine.testing.testLocalCudaPeerCopyAvailable(&source, &target));
    try std.testing.expectEqual(@as(usize, 1), trace_data.source_enable_calls);
    try std.testing.expectEqual(@as(usize, 1), trace_data.target_enable_calls);
    try std.testing.expectEqual(@as(usize, 1), trace_data.probe_calls);
}

test "executeLocalStageChain moves decode activation through selected bridge mode" {
    const Trace = struct {
        sync_calls: usize = 0,
        upload_calls: usize = 0,
        uploaded_bytes: usize = 0,
    };
    const Stage = struct {
        trace: *Trace,

        pub fn executeLayers(_: *@This(), input: []const u8, _: usize, _: usize) anyerror!void {
            try stage_adapters.validateEmptyInput(input);
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            stage.trace.sync_calls += 1;
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            stage.trace.upload_calls += 1;
            stage.trace.uploaded_bytes = byte_count;
        }
    };

    var bundle = try buildPlacement(.decode);
    defer bundle.deinit();
    const placement = &bundle.placement_plan.?;
    var metadata = try metadataForBoundary(placement, 0, .decode);
    var payload = [_]u8{0xaa} ** 16;
    const image = bridge.hostActivationByteImage(&metadata, payload[0..]);
    var trace_data = Trace{};
    var source = Stage{ .trace = &trace_data };
    var target = Stage{ .trace = &trace_data };
    var stages = [_]bridge.LocalStageChainStage{
        bridge.localStageAdapter(Stage, metadata.boundary.source_stage_id, &source),
        bridge.localStageAdapter(Stage, metadata.boundary.target_stage_id, &target),
    };
    const boundaries = [_]bridge.LocalStageChainBoundaryStep{.{
        .boundary_index = 0,
        .step_kind = .decode,
        .metadata = &metadata,
        .image = &image,
        .allow_borrow = false,
    }};

    try bridge.executeLocalStageChain(.{
        .allocator = std.testing.allocator,
        .plan_ref = &bundle.local_stage_runner_plan_ref.?,
        .placement_plan = placement,
        .stages = &stages,
        .boundaries = &boundaries,
    });

    try std.testing.expectEqual(@as(usize, 1), trace_data.sync_calls);
    try std.testing.expectEqual(@as(usize, 1), trace_data.upload_calls);
    try std.testing.expectEqual(@as(usize, 16), trace_data.uploaded_bytes);
}

test "executeLocalStageChain moves prefill activation through selected bridge mode" {
    const Trace = struct {
        sync_calls: usize = 0,
        upload_calls: usize = 0,
    };
    const Stage = struct {
        trace: *Trace,

        pub fn executeLayers(_: *@This(), input: []const u8, _: usize, _: usize) anyerror!void {
            try stage_adapters.validateEmptyInput(input);
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            stage.trace.sync_calls += 1;
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            stage.trace.upload_calls += 1;
        }
    };

    var bundle = try buildPlacement(.prefill);
    defer bundle.deinit();
    const placement = &bundle.placement_plan.?;
    var metadata = try metadataForBoundary(placement, 0, .prefill);
    var payload = [_]u8{0xbb} ** 16;
    const image = bridge.hostActivationByteImage(&metadata, payload[0..]);
    var trace_data = Trace{};
    var source = Stage{ .trace = &trace_data };
    var target = Stage{ .trace = &trace_data };
    var stages = [_]bridge.LocalStageChainStage{
        bridge.localStageAdapter(Stage, metadata.boundary.source_stage_id, &source),
        bridge.localStageAdapter(Stage, metadata.boundary.target_stage_id, &target),
    };
    const boundaries = [_]bridge.LocalStageChainBoundaryStep{.{
        .boundary_index = 0,
        .step_kind = .prefill,
        .metadata = &metadata,
        .image = &image,
        .allow_borrow = false,
    }};

    try bridge.executeLocalStageChain(.{
        .allocator = std.testing.allocator,
        .plan_ref = &bundle.local_stage_runner_plan_ref.?,
        .placement_plan = placement,
        .stages = &stages,
        .boundaries = &boundaries,
    });

    try std.testing.expectEqual(@as(usize, 1), trace_data.sync_calls);
    try std.testing.expectEqual(@as(usize, 1), trace_data.upload_calls);
}

test "executeLocalStageChain uses peer copy when selected by bridge" {
    const Trace = struct {
        sync_calls: usize = 0,
        peer_calls: usize = 0,
        upload_calls: usize = 0,
    };
    const Source = struct {
        trace: *Trace,

        pub fn executeLayers(_: *@This(), input: []const u8, _: usize, _: usize) anyerror!void {
            try stage_adapters.validateEmptyInput(input);
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            stage.trace.sync_calls += 1;
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(_: *@This(), _: []const u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn peerCopyActivationToErased(stage: *@This(), target_ptr: *anyopaque, byte_count: usize) anyerror!void {
            _ = target_ptr;
            if (byte_count == 0) return error.InvalidArgument;
            stage.trace.peer_calls += 1;
        }
    };
    const Target = struct {
        trace: *Trace,

        pub fn executeLayers(_: *@This(), input: []const u8, _: usize, _: usize) anyerror!void {
            try stage_adapters.validateEmptyInput(input);
        }

        pub fn synchronize(_: *@This()) anyerror!void {}
        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }
        pub fn uploadActivation(stage: *@This(), _: []const u8, _: usize) anyerror!void {
            stage.trace.upload_calls += 1;
        }
    };

    var bundle = try buildPlacement(.decode);
    defer bundle.deinit();
    const placement = &bundle.placement_plan.?;
    var metadata = try metadataForBoundary(placement, 0, .decode);
    metadata.payload.location_hint = .{ .cuda = 0 };
    const image = bridge.deviceActivationByteImage(&metadata);
    var trace_data = Trace{};
    var source = Source{ .trace = &trace_data };
    var target = Target{ .trace = &trace_data };
    var stages = [_]bridge.LocalStageChainStage{
        bridge.localStageAdapter(Source, metadata.boundary.source_stage_id, &source),
        bridge.localStageAdapter(Target, metadata.boundary.target_stage_id, &target),
    };
    const boundaries = [_]bridge.LocalStageChainBoundaryStep{.{
        .boundary_index = 0,
        .step_kind = .decode,
        .metadata = &metadata,
        .image = &image,
        .allow_borrow = false,
        .local_device_peer_copy_available = true,
    }};

    try bridge.executeLocalStageChain(.{
        .allocator = std.testing.allocator,
        .plan_ref = &bundle.local_stage_runner_plan_ref.?,
        .placement_plan = placement,
        .stages = &stages,
        .boundaries = &boundaries,
    });

    try std.testing.expectEqual(@as(usize, 1), trace_data.sync_calls);
    try std.testing.expectEqual(@as(usize, 1), trace_data.peer_calls);
    try std.testing.expectEqual(@as(usize, 0), trace_data.upload_calls);
}

test "executeLocalPipelineStep executes source handoff target in order" {
    const TraceStep = enum { stage0_execute, stage0_sync, stage1_upload, stage1_execute, stage1_sync, stage2_upload, stage2_execute };
    const Trace = struct {
        steps: [8]TraceStep = undefined,
        len: usize = 0,

        fn push(trace_data: *@This(), step: TraceStep) void {
            trace_data.steps[trace_data.len] = step;
            trace_data.len += 1;
        }
    };
    const Stage = chainStage(Trace, TraceStep);

    var bundle = try buildPlacement(.decode);
    defer bundle.deinit();
    const placement = &bundle.placement_plan.?;
    var metadata = try metadataForBoundary(placement, 0, .decode);
    var payload = [_]u8{0xcc} ** 16;
    const image = bridge.hostActivationByteImage(&metadata, payload[0..]);
    var trace_data = Trace{};
    var stage0 = Stage{ .trace = &trace_data, .id = 0 };
    var stage1 = Stage{ .trace = &trace_data, .id = 1 };
    var stages = [_]bridge.LocalStageChainStage{
        bridge.localStageAdapter(Stage, metadata.boundary.source_stage_id, &stage0),
        bridge.localStageAdapter(Stage, metadata.boundary.target_stage_id, &stage1),
    };
    const boundary_payloads = [_]bridge.LocalPipelineBoundaryPayload{.{
        .metadata = &metadata,
        .image = &image,
        .runtime = .{ .allow_borrow = false },
    }};

    try bridge.executeLocalPipelineStep(.{
        .allocator = std.testing.allocator,
        .plan_ref = &bundle.local_stage_runner_plan_ref.?,
        .placement_plan = placement,
    }, &stages, &boundary_payloads, .decode, &.{});

    try std.testing.expectEqual(@as(usize, 4), trace_data.len);
    try std.testing.expectEqual(TraceStep.stage0_execute, trace_data.steps[0]);
    try std.testing.expectEqual(TraceStep.stage0_sync, trace_data.steps[1]);
    try std.testing.expectEqual(TraceStep.stage1_upload, trace_data.steps[2]);
    try std.testing.expectEqual(TraceStep.stage1_execute, trace_data.steps[3]);
}

test "executeLocalPipelineStep executes both adjacent boundaries in order" {
    const TraceStep = enum { stage0_execute, stage0_sync, stage1_upload, stage1_execute, stage1_sync, stage2_upload, stage2_execute };
    const Trace = struct {
        steps: [10]TraceStep = undefined,
        len: usize = 0,

        fn push(trace_data: *@This(), step: TraceStep) void {
            trace_data.steps[trace_data.len] = step;
            trace_data.len += 1;
        }
    };
    const Stage = chainStage(Trace, TraceStep);

    var bundle = try buildThreeStagePlacement();
    defer bundle.deinit();
    const placement = &bundle.placement_plan.?;
    var metadata01 = try metadataForBoundary(placement, 0, .decode);
    var metadata12 = try metadataForBoundary(placement, 1, .decode);
    var payload01 = [_]u8{0xcd} ** 16;
    var payload12 = [_]u8{0xce} ** 16;
    const image01 = bridge.hostActivationByteImage(&metadata01, payload01[0..]);
    const image12 = bridge.hostActivationByteImage(&metadata12, payload12[0..]);
    var trace_data = Trace{};
    var stage0 = Stage{ .trace = &trace_data, .id = 0 };
    var stage1 = Stage{ .trace = &trace_data, .id = 1 };
    var stage2 = Stage{ .trace = &trace_data, .id = 2 };
    var stages = [_]bridge.LocalStageChainStage{
        bridge.localStageAdapter(Stage, metadata01.boundary.source_stage_id, &stage0),
        bridge.localStageAdapter(Stage, metadata01.boundary.target_stage_id, &stage1),
        bridge.localStageAdapter(Stage, metadata12.boundary.target_stage_id, &stage2),
    };
    const boundary_payloads = [_]bridge.LocalPipelineBoundaryPayload{
        .{ .metadata = &metadata01, .image = &image01, .runtime = .{ .allow_borrow = false } },
        .{ .metadata = &metadata12, .image = &image12, .runtime = .{ .allow_borrow = false } },
    };

    try bridge.executeLocalPipelineStep(.{
        .allocator = std.testing.allocator,
        .plan_ref = &bundle.local_stage_runner_plan_ref.?,
        .placement_plan = placement,
    }, &stages, &boundary_payloads, .decode, &.{});

    try std.testing.expectEqual(@as(usize, 7), trace_data.len);
    try std.testing.expectEqual(TraceStep.stage0_execute, trace_data.steps[0]);
    try std.testing.expectEqual(TraceStep.stage0_sync, trace_data.steps[1]);
    try std.testing.expectEqual(TraceStep.stage1_upload, trace_data.steps[2]);
    try std.testing.expectEqual(TraceStep.stage1_execute, trace_data.steps[3]);
    try std.testing.expectEqual(TraceStep.stage1_sync, trace_data.steps[4]);
    try std.testing.expectEqual(TraceStep.stage2_upload, trace_data.steps[5]);
    try std.testing.expectEqual(TraceStep.stage2_execute, trace_data.steps[6]);
}

test "executeLocalPipelineStep executes three adjacent boundaries in order" {
    const TraceStep = enum {
        stage0_execute,
        stage0_sync,
        stage1_upload,
        stage1_execute,
        stage1_sync,
        stage2_upload,
        stage2_execute,
        stage2_sync,
        stage3_upload,
        stage3_execute,
    };
    const Trace = struct {
        steps: [12]TraceStep = undefined,
        len: usize = 0,

        fn push(trace_data: *@This(), step: TraceStep) void {
            trace_data.steps[trace_data.len] = step;
            trace_data.len += 1;
        }
    };
    const Stage = struct {
        trace: *Trace,
        id: usize,

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            try stage_adapters.validateEmptyInput(input);
            try std.testing.expect(layer_end > layer_start);
            stage.trace.push(switch (stage.id) {
                0 => TraceStep.stage0_execute,
                1 => TraceStep.stage1_execute,
                2 => TraceStep.stage2_execute,
                else => TraceStep.stage3_execute,
            });
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            switch (stage.id) {
                0 => stage.trace.push(TraceStep.stage0_sync),
                1 => stage.trace.push(TraceStep.stage1_sync),
                2 => stage.trace.push(TraceStep.stage2_sync),
                else => {},
            }
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            switch (stage.id) {
                1 => stage.trace.push(TraceStep.stage1_upload),
                2 => stage.trace.push(TraceStep.stage2_upload),
                3 => stage.trace.push(TraceStep.stage3_upload),
                else => {},
            }
        }
    };

    var bundle = try buildFourStagePlacement();
    defer bundle.deinit();
    const placement = &bundle.placement_plan.?;
    var metadata01 = try metadataForBoundary(placement, 0, .decode);
    var metadata12 = try metadataForBoundary(placement, 1, .decode);
    var metadata23 = try metadataForBoundary(placement, 2, .decode);
    var payload01 = [_]u8{0xc1} ** 16;
    var payload12 = [_]u8{0xc2} ** 16;
    var payload23 = [_]u8{0xc3} ** 16;
    const image01 = bridge.hostActivationByteImage(&metadata01, payload01[0..]);
    const image12 = bridge.hostActivationByteImage(&metadata12, payload12[0..]);
    const image23 = bridge.hostActivationByteImage(&metadata23, payload23[0..]);
    var trace_data = Trace{};
    var stage0 = Stage{ .trace = &trace_data, .id = 0 };
    var stage1 = Stage{ .trace = &trace_data, .id = 1 };
    var stage2 = Stage{ .trace = &trace_data, .id = 2 };
    var stage3 = Stage{ .trace = &trace_data, .id = 3 };
    var stages = [_]bridge.LocalStageChainStage{
        bridge.localStageAdapter(Stage, metadata01.boundary.source_stage_id, &stage0),
        bridge.localStageAdapter(Stage, metadata01.boundary.target_stage_id, &stage1),
        bridge.localStageAdapter(Stage, metadata12.boundary.target_stage_id, &stage2),
        bridge.localStageAdapter(Stage, metadata23.boundary.target_stage_id, &stage3),
    };
    const boundary_payloads = [_]bridge.LocalPipelineBoundaryPayload{
        .{ .metadata = &metadata01, .image = &image01, .runtime = .{ .allow_borrow = false } },
        .{ .metadata = &metadata12, .image = &image12, .runtime = .{ .allow_borrow = false } },
        .{ .metadata = &metadata23, .image = &image23, .runtime = .{ .allow_borrow = false } },
    };

    try bridge.executeLocalPipelineStep(.{
        .allocator = std.testing.allocator,
        .plan_ref = &bundle.local_stage_runner_plan_ref.?,
        .placement_plan = placement,
    }, &stages, &boundary_payloads, .decode, &.{});

    try std.testing.expectEqual(@as(usize, 10), trace_data.len);
    try std.testing.expectEqual(TraceStep.stage0_execute, trace_data.steps[0]);
    try std.testing.expectEqual(TraceStep.stage0_sync, trace_data.steps[1]);
    try std.testing.expectEqual(TraceStep.stage1_upload, trace_data.steps[2]);
    try std.testing.expectEqual(TraceStep.stage1_execute, trace_data.steps[3]);
    try std.testing.expectEqual(TraceStep.stage1_sync, trace_data.steps[4]);
    try std.testing.expectEqual(TraceStep.stage2_upload, trace_data.steps[5]);
    try std.testing.expectEqual(TraceStep.stage2_execute, trace_data.steps[6]);
    try std.testing.expectEqual(TraceStep.stage2_sync, trace_data.steps[7]);
    try std.testing.expectEqual(TraceStep.stage3_upload, trace_data.steps[8]);
    try std.testing.expectEqual(TraceStep.stage3_execute, trace_data.steps[9]);
}

test "executeLocalDecodePipelineStep accepts generic four stage local chain" {
    const TraceStep = enum {
        stage0_execute,
        stage0_sync,
        stage1_upload,
        stage1_execute,
        stage1_sync,
        stage2_upload,
        stage2_execute,
        stage2_sync,
        stage3_upload,
        stage3_execute,
    };
    const Trace = struct {
        steps: [12]TraceStep = undefined,
        len: usize = 0,

        fn push(trace_data: *@This(), step: TraceStep) void {
            trace_data.steps[trace_data.len] = step;
            trace_data.len += 1;
        }
    };
    const Stage = struct {
        trace: *Trace,
        id: usize,

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            try stage_adapters.validateEmptyInput(input);
            try std.testing.expect(layer_end > layer_start);
            stage.trace.push(switch (stage.id) {
                0 => TraceStep.stage0_execute,
                1 => TraceStep.stage1_execute,
                2 => TraceStep.stage2_execute,
                else => TraceStep.stage3_execute,
            });
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            switch (stage.id) {
                0 => stage.trace.push(TraceStep.stage0_sync),
                1 => stage.trace.push(TraceStep.stage1_sync),
                2 => stage.trace.push(TraceStep.stage2_sync),
                else => {},
            }
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            switch (stage.id) {
                1 => stage.trace.push(TraceStep.stage1_upload),
                2 => stage.trace.push(TraceStep.stage2_upload),
                3 => stage.trace.push(TraceStep.stage3_upload),
                else => {},
            }
        }
    };

    var bundle = try buildFourStagePlacement();
    defer bundle.deinit();
    var payload01 = [_]u8{0xc4} ** 16;
    var payload12 = [_]u8{0xc5} ** 16;
    var payload23 = [_]u8{0xc6} ** 16;
    var slot_request_ids = [_]?u64{null} ** 8;
    slot_request_ids[2] = 202;
    const slot_indices = [_]usize{2};
    const positions = [_]usize{5};
    var trace_data = Trace{};
    var stage0 = Stage{ .trace = &trace_data, .id = 0 };
    var stage1 = Stage{ .trace = &trace_data, .id = 1 };
    var stage2 = Stage{ .trace = &trace_data, .id = 2 };
    var stage3 = Stage{ .trace = &trace_data, .id = 3 };
    const runner = &bundle.local_stage_runner_plan_ref.?;
    var stages = [_]bridge.LocalStageChainStage{
        bridge.localStageAdapter(Stage, runner.stages[0].stage_id, &stage0),
        bridge.localStageAdapter(Stage, runner.stages[1].stage_id, &stage1),
        bridge.localStageAdapter(Stage, runner.stages[2].stage_id, &stage2),
        bridge.localStageAdapter(Stage, runner.stages[3].stage_id, &stage3),
    };
    const boundary_payloads = [_]bridge.LocalDecodeBoundaryPayloadSpec{
        .{
            .frame = .{ .boundary_index = 0, .dtype = .f32, .layout = .row_major },
            .activation_byte_count = payload01.len,
            .location_hint = .cpu,
            .image = .{ .host_bytes = payload01[0..] },
        },
        .{
            .frame = .{ .boundary_index = 1, .dtype = .f32, .layout = .row_major },
            .activation_byte_count = payload12.len,
            .location_hint = .cpu,
            .image = .{ .host_bytes = payload12[0..] },
        },
        .{
            .frame = .{ .boundary_index = 2, .dtype = .f32, .layout = .row_major },
            .activation_byte_count = payload23.len,
            .location_hint = .cpu,
            .image = .{ .host_bytes = payload23[0..] },
        },
    };

    try bridge.executeLocalDecodePipelineStep(.{
        .allocator = std.testing.allocator,
        .plan_ref = runner,
        .placement_plan = &bundle.placement_plan.?,
    }, &stages, .{
        .tensor_frame_plan_ref = &bundle.tensor_frame_plan_ref.?,
        .hidden_size = 4,
        .slot_request_ids = &slot_request_ids,
        .slot_indices = &slot_indices,
        .positions = &positions,
        .boundary_payloads = &boundary_payloads,
    });

    try std.testing.expectEqual(@as(usize, 10), trace_data.len);
    try std.testing.expectEqual(TraceStep.stage0_execute, trace_data.steps[0]);
    try std.testing.expectEqual(TraceStep.stage0_sync, trace_data.steps[1]);
    try std.testing.expectEqual(TraceStep.stage1_upload, trace_data.steps[2]);
    try std.testing.expectEqual(TraceStep.stage1_execute, trace_data.steps[3]);
    try std.testing.expectEqual(TraceStep.stage1_sync, trace_data.steps[4]);
    try std.testing.expectEqual(TraceStep.stage2_upload, trace_data.steps[5]);
    try std.testing.expectEqual(TraceStep.stage2_execute, trace_data.steps[6]);
    try std.testing.expectEqual(TraceStep.stage2_sync, trace_data.steps[7]);
    try std.testing.expectEqual(TraceStep.stage3_upload, trace_data.steps[8]);
    try std.testing.expectEqual(TraceStep.stage3_execute, trace_data.steps[9]);
}

test "executeLocalPrefillPipelineStep builds prefill activation handoff" {
    const TraceStep = enum { stage0_execute, stage0_sync, stage1_upload, stage1_execute };
    const Trace = struct {
        steps: [4]TraceStep = undefined,
        len: usize = 0,

        fn push(trace_data: *@This(), step: TraceStep) void {
            trace_data.steps[trace_data.len] = step;
            trace_data.len += 1;
        }
    };
    const Stage = struct {
        trace: *Trace,
        id: usize,

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            try stage_adapters.validateEmptyInput(input);
            try std.testing.expect(layer_end > layer_start);
            stage.trace.push(if (stage.id == 0) TraceStep.stage0_execute else TraceStep.stage1_execute);
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            if (stage.id == 0) stage.trace.push(TraceStep.stage0_sync);
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            if (stage.id == 1) stage.trace.push(TraceStep.stage1_upload);
        }
    };

    var bundle = try buildPlacement(.prefill);
    defer bundle.deinit();
    var payload = [_]u8{0xc7} ** 16;
    var slot_request_ids = [_]?u64{null} ** 4;
    slot_request_ids[1] = 101;
    var trace_data = Trace{};
    var stage0 = Stage{ .trace = &trace_data, .id = 0 };
    var stage1 = Stage{ .trace = &trace_data, .id = 1 };
    const runner = &bundle.local_stage_runner_plan_ref.?;
    var stages = [_]bridge.LocalStageChainStage{
        bridge.localStageAdapter(Stage, runner.stages[0].stage_id, &stage0),
        bridge.localStageAdapter(Stage, runner.stages[1].stage_id, &stage1),
    };
    const boundary_payloads = [_]bridge.LocalPrefillBoundaryPayloadSpec{.{
        .frame = .{ .boundary_index = 0, .dtype = .f32, .layout = .row_major },
        .slot_index = 1,
        .sequence_start = 3,
        .token_count = 1,
        .activation_byte_count = payload.len,
        .location_hint = .cpu,
        .image = .{ .host_bytes = payload[0..] },
    }};

    try bridge.executeLocalPrefillPipelineStep(.{
        .allocator = std.testing.allocator,
        .plan_ref = runner,
        .placement_plan = &bundle.placement_plan.?,
    }, &stages, .{
        .tensor_frame_plan_ref = &bundle.tensor_frame_plan_ref.?,
        .hidden_size = 4,
        .slot_request_ids = &slot_request_ids,
        .boundary_payloads = &boundary_payloads,
    });

    try std.testing.expectEqual(@as(usize, 4), trace_data.len);
    try std.testing.expectEqual(TraceStep.stage0_execute, trace_data.steps[0]);
    try std.testing.expectEqual(TraceStep.stage0_sync, trace_data.steps[1]);
    try std.testing.expectEqual(TraceStep.stage1_upload, trace_data.steps[2]);
    try std.testing.expectEqual(TraceStep.stage1_execute, trace_data.steps[3]);
}

test "executeLocalPipelineStep stops before stage two when boundary zero fails" {
    const TraceStep = enum { stage0_execute, stage0_sync, stage1_upload, stage1_execute, stage1_sync, stage2_upload, stage2_execute };
    const Trace = struct {
        steps: [8]TraceStep = undefined,
        len: usize = 0,

        fn push(trace_data: *@This(), step: TraceStep) void {
            trace_data.steps[trace_data.len] = step;
            trace_data.len += 1;
        }
    };
    const Stage = chainStage(Trace, TraceStep);

    var bundle = try buildThreeStagePlacement();
    defer bundle.deinit();
    const placement = &bundle.placement_plan.?;
    var metadata01 = try metadataForBoundary(placement, 0, .decode);
    var metadata12 = try metadataForBoundary(placement, 1, .decode);
    var payload01 = [_]u8{0xdd} ** 16;
    var payload12 = [_]u8{0xee} ** 16;
    const image01 = bridge.hostActivationByteImage(&metadata01, payload01[0..]);
    const image12 = bridge.hostActivationByteImage(&metadata12, payload12[0..]);
    var trace_data = Trace{};
    var stage0 = Stage{ .trace = &trace_data, .id = 0 };
    var stage1 = Stage{ .trace = &trace_data, .id = 1, .fail_upload = true };
    var stage2 = Stage{ .trace = &trace_data, .id = 2 };
    var stages = [_]bridge.LocalStageChainStage{
        bridge.localStageAdapter(Stage, metadata01.boundary.source_stage_id, &stage0),
        bridge.localStageAdapter(Stage, metadata01.boundary.target_stage_id, &stage1),
        bridge.localStageAdapter(Stage, metadata12.boundary.target_stage_id, &stage2),
    };
    const boundary_payloads = [_]bridge.LocalPipelineBoundaryPayload{
        .{ .metadata = &metadata01, .image = &image01, .runtime = .{ .allow_borrow = false } },
        .{ .metadata = &metadata12, .image = &image12, .runtime = .{ .allow_borrow = false } },
    };

    try std.testing.expectError(
        error.AccessDenied,
        bridge.executeLocalPipelineStep(.{
            .allocator = std.testing.allocator,
            .plan_ref = &bundle.local_stage_runner_plan_ref.?,
            .placement_plan = placement,
        }, &stages, &boundary_payloads, .decode, &.{}),
    );

    try std.testing.expectEqual(@as(usize, 3), trace_data.len);
    try std.testing.expectEqual(TraceStep.stage0_execute, trace_data.steps[0]);
    try std.testing.expectEqual(TraceStep.stage0_sync, trace_data.steps[1]);
    try std.testing.expectEqual(TraceStep.stage1_upload, trace_data.steps[2]);
}

fn chainStage(comptime Trace: type, comptime TraceStep: type) type {
    return struct {
        trace: *Trace,
        id: usize,
        fail_upload: bool = false,

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            try stage_adapters.validateEmptyInput(input);
            try std.testing.expect(layer_end > layer_start);
            stage.trace.push(switch (stage.id) {
                0 => TraceStep.stage0_execute,
                1 => TraceStep.stage1_execute,
                else => TraceStep.stage2_execute,
            });
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            if (stage.id == 0) stage.trace.push(TraceStep.stage0_sync);
            if (stage.id == 1) stage.trace.push(TraceStep.stage1_sync);
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            if (stage.id == 1) stage.trace.push(TraceStep.stage1_upload);
            if (stage.id == 2) stage.trace.push(TraceStep.stage2_upload);
            if (stage.fail_upload) return error.AccessDenied;
        }
    };
}

const test_entries = [_]bridge.TensorFrameBatchEntry{.{
    .batch_index = 0,
    .request_id = 101,
    .slot_id = 7,
    .sequence_start = 3,
    .token_count = 1,
}};

fn buildPlacement(step_kind: bridge.TensorFrameStepKind) !engine.testing.LocalTopologyContractBundle {
    _ = step_kind;
    const plan = try buildTestStagePlan(std.testing.allocator, 4, &.{2});
    const kinds = [_]bridge.HostBackendKind{ .cpu, .cuda };
    const configs = [_]engine.testing.BoundaryConfig{
        engine.testing.localCpuCudaBoundaryConfig(.f32, .row_major, 4, 1),
    };
    var bundle = try engine.testing.buildLocalTopologyContractBundleFromOwnedPlan(
        std.testing.allocator,
        4,
        plan,
        &kinds,
        &configs,
    );
    errdefer bundle.deinit();
    return bundle;
}

fn buildThreeStagePlacement() !engine.testing.LocalTopologyContractBundle {
    const plan = try buildTestStagePlan(std.testing.allocator, 4, &.{ 2, 3 });
    const kinds = [_]bridge.HostBackendKind{ .cpu, .cuda, .cuda };
    const configs = engine.testing.localCpuCudaCudaBoundaryConfigs(.f32, .row_major, .f32, .row_major, 4, 4, 1, 1);
    return engine.testing.buildLocalTopologyContractBundleFromOwnedPlan(
        std.testing.allocator,
        4,
        plan,
        &kinds,
        &configs,
    );
}

fn buildFourStagePlacement() !engine.testing.LocalTopologyContractBundle {
    const plan = try buildTestStagePlan(std.testing.allocator, 4, &.{ 1, 2, 3 });
    const kinds = [_]bridge.HostBackendKind{ .cuda, .cpu, .cuda, .cpu };
    const configs = [_]engine.testing.BoundaryConfig{
        engine.testing.localCpuCudaBoundaryConfig(.f32, .row_major, 4, 1),
        engine.testing.localCpuCudaBoundaryConfig(.f32, .row_major, 4, 1),
        engine.testing.localCpuCudaBoundaryConfig(.f32, .row_major, 4, 1),
    };
    return engine.testing.buildLocalTopologyContractBundleFromOwnedPlan(
        std.testing.allocator,
        4,
        plan,
        &kinds,
        &configs,
    );
}

fn metadataForBoundary(
    placement: *const bridge.PlacementPlan,
    boundary_index: usize,
    step_kind: bridge.TensorFrameStepKind,
) !bridge.TensorFrameMetadata {
    const summary = placement.boundary_summaries[boundary_index];
    const boundary = bridge.TensorFrameBoundaryRef{
        .boundary_index = summary.boundary_index,
        .source_stage_id = summary.source_stage_id,
        .target_stage_id = summary.target_stage_id,
        .producer_layer_start = summary.producer_layer_start,
        .producer_layer_end = summary.producer_layer_end,
        .consumer_layer_start = summary.consumer_layer_start,
        .consumer_layer_end = summary.consumer_layer_end,
    };
    const tensor_desc = try bridge.TensorFrameTensorDesc.contiguousActivation(.f32, .{ 1, 1, 4, 0 });
    return .{
        .frame_id = try bridge.TensorFrameInstanceId.init(77 + boundary_index),
        .plan = .{
            .graph_digest = placement.graph_digest,
            .graph_contract_version = placement.graph_contract_version,
            .stage_plan_contract_version = placement.stage_plan_contract_version,
            .stage_plan_id = placement.stage_plan_id,
        },
        .boundary = boundary,
        .selected_contract = .{
            .boundary = boundary,
            .dtype = .f32,
            .layout = .row_major,
            .source = .explicit,
        },
        .role = .activation,
        .step_kind = step_kind,
        .shape_context = .{ .expected_hidden_size = 4, .expected_step_kind = step_kind },
        .tensor = tensor_desc,
        .batch = .{ .entries = &test_entries },
        .payload = .{
            .byte_count = tensor_desc.payload_byte_count,
            .location_hint = .cpu,
            .ownership = .borrowed_until_next_stage_call,
            .lifetime = .step_scoped,
        },
    };
}

fn buildTestStagePlan(
    allocator: std.mem.Allocator,
    layer_count: usize,
    splits: []const usize,
) !models.stage_plan.StagePlan {
    var arch = localTestArch();
    var config = localTestConfig(layer_count);
    var manifest = try localTestManifest(allocator, layer_count);
    defer manifest.deinit();
    return models.stage_plan.buildStagePlan(allocator, .{
        .n_layers = layer_count,
        .split_points = splits,
        .architecture = &arch,
        .model_config = &config,
        .manifest = &manifest,
        .partition_constraints = .{
            .decoder_cuts_allowed = true,
            .dependency_overrides = &.{},
        },
    });
}

fn localTestConfig(layer_count: usize) models.config.ModelConfig {
    return .{
        .vocab_size = 64,
        .d_model = 8,
        .n_layers = @intCast(layer_count),
        .n_heads = 2,
        .n_kv_groups = 2,
        .d_ff = 16,
        .max_seq_len = 32,
        .head_dim = 4,
        .rope_theta = 10000,
        .norm_eps = 0.00001,
        .gaffine_group_size = 0,
        .tie_word_embeddings = false,
    };
}

fn localTestArch() models.op_types.Architecture {
    return .{
        .name = "cuda_stage_chain_test",
        .model_types = &.{"cuda_stage_chain_test"},
    };
}

fn localTestManifest(allocator: std.mem.Allocator, layer_count: usize) !models.manifest.ModelManifest {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();
    const entry_count = layer_count + 3;
    const entries = try arena_allocator.alloc(models.manifest.TensorManifestEntry, entry_count);
    entries[0] = .{
        .name = "model.embed_tokens.weight",
        .dtype = .f32,
        .shape = &.{ 64, 8 },
        .checkpoint_bytes = 128,
        .role = .token_embeddings,
        .weight_id = "token_embeddings",
        .status = .architecture_weight,
    };
    for (0..layer_count) |layer_index| {
        entries[layer_index + 1] = .{
            .name = "model.layers.self_attn.q_proj.weight",
            .dtype = .f32,
            .shape = &.{ 8, 8 },
            .checkpoint_bytes = 64,
            .role = .decoder_layer,
            .layer_index = layer_index,
            .weight_id = "self_attn.q_proj.weight",
            .status = .architecture_weight,
        };
    }
    entries[layer_count + 1] = .{
        .name = "model.norm.weight",
        .dtype = .f32,
        .shape = &.{8},
        .checkpoint_bytes = 32,
        .role = .final_norm,
        .weight_id = "ln_final",
        .status = .architecture_weight,
    };
    entries[layer_count + 2] = .{
        .name = "lm_head.weight",
        .dtype = .f32,
        .shape = &.{ 64, 8 },
        .checkpoint_bytes = 128,
        .role = .lm_head,
        .weight_id = "lm_head",
        .status = .architecture_weight,
    };

    var role_bytes = [_]usize{0} ** models.manifest.role_count;
    var total_bytes: usize = 0;
    for (entries) |entry| {
        total_bytes += entry.checkpoint_bytes;
        role_bytes[@intFromEnum(entry.role)] += entry.checkpoint_bytes;
    }

    return .{
        .arena = arena,
        .architecture_id = "cuda_stage_chain_test",
        .layer_count = layer_count,
        .entries = entries,
        .total_checkpoint_bytes = total_bytes,
        .role_bytes = role_bytes,
    };
}
