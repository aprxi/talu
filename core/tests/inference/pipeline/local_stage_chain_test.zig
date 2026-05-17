//! Pipeline integration tests for local stage-chain execution.

const std = @import("std");
const main = @import("main");

const pipeline = main.inference.pipeline;
const transport = main.inference.transport;
const models = main.models.dispatcher;
const local_stage_testing = @import("local_stage_test_helpers.zig");

fn validateEmptyInput(input: []const u8) !void {
    if (input.len != 0) return error.InvalidArgument;
}

test "peerCopyCudaActivation source-event mode falls back to source-stream synchronization" {
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
    };

    var trace_data = Trace{};
    var source = Backend{
        .device = .{ .trace = &trace_data },
        .compute_stream = 11,
    };
    var target = Backend{
        .device = .{ .trace = &trace_data },
        .compute_stream = 22,
    };

    try transport.peerCopyCudaActivation(&source, &target, 64, .source_event_target_stream);

    try std.testing.expectEqual(@as(usize, 0), trace_data.record_calls);
    try std.testing.expectEqual(@as(usize, 0), trace_data.wait_calls);
    try std.testing.expectEqual(@as(usize, 0), trace_data.make_current_calls);
    try std.testing.expectEqual(@as(usize, 1), trace_data.peer_calls);
    try std.testing.expectEqual(@as(usize, 64), trace_data.peer_bytes);
    try std.testing.expectEqual(@as(?usize, 11), trace_data.peer_stream);
    try std.testing.expectEqual(@as(usize, 1), trace_data.sync_calls);
    try std.testing.expect(!transport.peerCopyCudaActivationHandlesStageSync(&source, .source_event_target_stream));
}

test "peerCopyCudaActivationHandlesStageSync reports transport-owned source sync" {
    const Backend = struct {};
    var backend = Backend{};

    try std.testing.expect(!transport.peerCopyCudaActivationHandlesStageSync(&backend, .source_event_target_stream));
    try std.testing.expect(!transport.peerCopyCudaActivationHandlesStageSync(&backend, .source_stream));
}

test "executeLocalStageChain moves decode activation through selected pipeline mode" {
    const Trace = struct {
        sync_calls: usize = 0,
        upload_calls: usize = 0,
        uploaded_bytes: usize = 0,
    };
    const Stage = struct {
        trace: *Trace,

        pub fn executeLayers(_: *@This(), input: []const u8, _: usize, _: usize) anyerror!void {
            try validateEmptyInput(input);
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
    const image = pipeline.hostActivationByteImage(&metadata, payload[0..]);
    var trace_data = Trace{};
    var source = Stage{ .trace = &trace_data };
    var target = Stage{ .trace = &trace_data };
    var stages = [_]pipeline.LocalStageChainStage{
        pipeline.localStageAdapter(Stage, metadata.boundary.source_stage_id, &source),
        pipeline.localStageAdapter(Stage, metadata.boundary.target_stage_id, &target),
    };
    const boundaries = [_]pipeline.LocalStageChainBoundaryStep{.{
        .boundary_index = 0,
        .step_kind = .decode,
        .metadata = &metadata,
        .image = &image,
        .allow_borrow = false,
    }};

    try pipeline.executeLocalStageChain(.{
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

test "executeLocalStageChain moves prefill activation through selected pipeline mode" {
    const Trace = struct {
        sync_calls: usize = 0,
        upload_calls: usize = 0,
    };
    const Stage = struct {
        trace: *Trace,

        pub fn executeLayers(_: *@This(), input: []const u8, _: usize, _: usize) anyerror!void {
            try validateEmptyInput(input);
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
    const image = pipeline.hostActivationByteImage(&metadata, payload[0..]);
    var trace_data = Trace{};
    var source = Stage{ .trace = &trace_data };
    var target = Stage{ .trace = &trace_data };
    var stages = [_]pipeline.LocalStageChainStage{
        pipeline.localStageAdapter(Stage, metadata.boundary.source_stage_id, &source),
        pipeline.localStageAdapter(Stage, metadata.boundary.target_stage_id, &target),
    };
    const boundaries = [_]pipeline.LocalStageChainBoundaryStep{.{
        .boundary_index = 0,
        .step_kind = .prefill,
        .metadata = &metadata,
        .image = &image,
        .allow_borrow = false,
    }};

    try pipeline.executeLocalStageChain(.{
        .allocator = std.testing.allocator,
        .plan_ref = &bundle.local_stage_runner_plan_ref.?,
        .placement_plan = placement,
        .stages = &stages,
        .boundaries = &boundaries,
    });

    try std.testing.expectEqual(@as(usize, 1), trace_data.sync_calls);
    try std.testing.expectEqual(@as(usize, 1), trace_data.upload_calls);
}

test "executeLocalStageChain uses peer copy when selected by pipeline" {
    const Trace = struct {
        sync_calls: usize = 0,
        peer_calls: usize = 0,
        upload_calls: usize = 0,
    };
    const Source = struct {
        trace: *Trace,

        pub fn executeLayers(_: *@This(), input: []const u8, _: usize, _: usize) anyerror!void {
            try validateEmptyInput(input);
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
            try validateEmptyInput(input);
        }

        pub fn synchronize(_: *@This()) anyerror!void {}
        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }
        pub fn uploadActivation(stage: *@This(), _: []const u8, _: usize) anyerror!void {
            stage.trace.upload_calls += 1;
        }
    };

    var bundle = try buildDevicePairPlacement();
    defer bundle.deinit();
    const placement = &bundle.placement_plan.?;
    var metadata = try metadataForBoundary(placement, 0, .decode);
    metadata.payload.location_hint = .{ .cuda = 0 };
    const image = pipeline.deviceActivationByteImage(&metadata);
    var trace_data = Trace{};
    var source = Source{ .trace = &trace_data };
    var target = Target{ .trace = &trace_data };
    var stages = [_]pipeline.LocalStageChainStage{
        pipeline.localStageAdapter(Source, metadata.boundary.source_stage_id, &source),
        pipeline.localStageAdapter(Target, metadata.boundary.target_stage_id, &target),
    };
    const boundaries = [_]pipeline.LocalStageChainBoundaryStep{.{
        .boundary_index = 0,
        .step_kind = .decode,
        .metadata = &metadata,
        .image = &image,
        .allow_borrow = false,
        .local_device_peer_copy_available = true,
    }};

    try pipeline.executeLocalStageChain(.{
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

test "executeLocalStageTransport handles generic device local fallback paths" {
    const Trace = struct {
        sync_calls: usize = 0,
        download_calls: usize = 0,
        upload_calls: usize = 0,
        peer_calls: usize = 0,
        uploaded_first_byte: u8 = 0,
    };
    const Source = struct {
        trace: *Trace,

        pub fn synchronize(stage: *@This()) anyerror!void {
            stage.trace.sync_calls += 1;
        }

        pub fn downloadActivation(stage: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            @memset(host_buf[0..byte_count], 0x5a);
            stage.trace.download_calls += 1;
        }

        pub fn uploadActivation(_: *@This(), _: []const u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn peerCopyActivationTo(stage: *@This(), target: anytype, byte_count: usize) anyerror!void {
            _ = target;
            if (byte_count == 0) return error.InvalidArgument;
            stage.trace.peer_calls += 1;
        }
    };
    const Target = struct {
        trace: *Trace,

        pub fn synchronize(_: *@This()) anyerror!void {}

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            stage.trace.upload_calls += 1;
            stage.trace.uploaded_first_byte = host_buf[0];
        }
    };

    const cases = [_]struct {
        source_kind: pipeline.HostBackendKind,
        target_kind: pipeline.HostBackendKind,
        hint: pipeline.TensorFramePayloadLocationHint,
    }{
        .{ .source_kind = .cuda, .target_kind = .cuda, .hint = .{ .cuda = 0 } },
        .{ .source_kind = .metal, .target_kind = .metal, .hint = .{ .metal = 0 } },
    };

    for (cases) |case| {
        var bundle = try buildTwoStagePlacementWithKinds(case.source_kind, case.target_kind);
        defer bundle.deinit();
        const placement = &bundle.placement_plan.?;
        var metadata = try metadataForBoundary(placement, 0, .decode);
        metadata.payload.location_hint = case.hint;
        const image = pipeline.deviceActivationByteImage(&metadata);
        var trace_data = Trace{};
        var source = Source{ .trace = &trace_data };
        var target = Target{ .trace = &trace_data };
        var staging_storage: [64]u8 align(64) = [_]u8{0} ** 64;
        const fallback_decision = try pipeline.chooseStageTransferMode(.{
            .placement_plan = placement,
            .metadata = &metadata,
            .image = &image,
            .local_device_peer_copy_available = false,
        });
        try std.testing.expectEqual(pipeline.StageTransferMode.device_download_then_copy, fallback_decision.mode);
        const fallback_envelope = try pipeline.buildStageTransportActivationEnvelope(.{
            .metadata = &metadata,
            .image = &image,
            .decision = fallback_decision,
        });
        try transport.executeLocalStageTransport(Source, Target, &source, &target, .{
            .placement_plan = placement,
            .metadata = &metadata,
            .image = &image,
            .decision = fallback_decision,
            .envelope = &fallback_envelope,
            .staging = staging_storage[0..],
        });
        try std.testing.expectEqual(@as(usize, 1), trace_data.sync_calls);
        try std.testing.expectEqual(@as(usize, 1), trace_data.download_calls);
        try std.testing.expectEqual(@as(usize, 1), trace_data.upload_calls);
        try std.testing.expectEqual(@as(u8, 0x5a), trace_data.uploaded_first_byte);

        trace_data = .{};
        source = .{ .trace = &trace_data };
        target = .{ .trace = &trace_data };
        const peer_decision = try pipeline.chooseStageTransferMode(.{
            .placement_plan = placement,
            .metadata = &metadata,
            .image = &image,
            .local_device_peer_copy_available = true,
        });
        try std.testing.expectEqual(pipeline.StageTransferMode.device_peer_copy_in_process, peer_decision.mode);
        const peer_envelope = try pipeline.buildStageTransportActivationEnvelope(.{
            .metadata = &metadata,
            .image = &image,
            .decision = peer_decision,
        });
        try transport.executeLocalStageTransport(Source, Target, &source, &target, .{
            .placement_plan = placement,
            .metadata = &metadata,
            .image = &image,
            .decision = peer_decision,
            .envelope = &peer_envelope,
            .local_device_peer_copy_available = true,
        });
        try std.testing.expectEqual(@as(usize, 1), trace_data.sync_calls);
        try std.testing.expectEqual(@as(usize, 0), trace_data.download_calls);
        try std.testing.expectEqual(@as(usize, 0), trace_data.upload_calls);
        try std.testing.expectEqual(@as(usize, 1), trace_data.peer_calls);
    }
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
    const image = pipeline.hostActivationByteImage(&metadata, payload[0..]);
    var trace_data = Trace{};
    var stage0 = Stage{ .trace = &trace_data, .id = 0 };
    var stage1 = Stage{ .trace = &trace_data, .id = 1 };
    var stages = [_]pipeline.LocalStageChainStage{
        pipeline.localStageAdapter(Stage, metadata.boundary.source_stage_id, &stage0),
        pipeline.localStageAdapter(Stage, metadata.boundary.target_stage_id, &stage1),
    };
    const boundary_payloads = [_]pipeline.LocalPipelineBoundaryPayload{.{
        .metadata = &metadata,
        .image = &image,
        .runtime = .{ .allow_borrow = false },
    }};

    try pipeline.executeLocalPipelineStep(.{
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

test "executeLocalPipelineStep runs erased source prepare before target upload" {
    const TraceStep = enum { stage0_execute, stage0_sync, stage0_prepare, stage1_upload, stage1_execute };
    const Trace = struct {
        steps: [8]TraceStep = undefined,
        len: usize = 0,

        fn push(trace_data: *@This(), step: TraceStep) void {
            trace_data.steps[trace_data.len] = step;
            trace_data.len += 1;
        }
    };
    const Stage = struct {
        trace: *Trace,
        id: usize,
        prepared_boundary: usize = std.math.maxInt(usize),

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            try validateEmptyInput(input);
            try std.testing.expect(layer_end > layer_start);
            stage.trace.push(if (stage.id == 0) TraceStep.stage0_execute else TraceStep.stage1_execute);
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            if (stage.id == 0) stage.trace.push(TraceStep.stage0_sync);
        }

        pub fn prepareBoundaryTransferToErased(
            stage: *@This(),
            target_ptr: *anyopaque,
            metadata: *const pipeline.TensorFrameMetadata,
        ) anyerror!void {
            if (stage.id != 0) return;
            const target: *@This() = @ptrCast(@alignCast(target_ptr));
            try std.testing.expectEqual(@as(usize, 1), target.id);
            stage.prepared_boundary = metadata.boundary.boundary_index;
            stage.trace.push(TraceStep.stage0_prepare);
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            if (stage.id == 1) stage.trace.push(TraceStep.stage1_upload);
        }
    };

    var bundle = try buildPlacement(.decode);
    defer bundle.deinit();
    const placement = &bundle.placement_plan.?;
    var metadata = try metadataForBoundary(placement, 0, .decode);
    var payload = [_]u8{0xca} ** 16;
    const image = pipeline.hostActivationByteImage(&metadata, payload[0..]);
    var trace_data = Trace{};
    var stage0 = Stage{ .trace = &trace_data, .id = 0 };
    var stage1 = Stage{ .trace = &trace_data, .id = 1 };
    var stages = [_]pipeline.LocalStageChainStage{
        pipeline.localStageAdapter(Stage, metadata.boundary.source_stage_id, &stage0),
        pipeline.localStageAdapter(Stage, metadata.boundary.target_stage_id, &stage1),
    };
    const boundary_payloads = [_]pipeline.LocalPipelineBoundaryPayload{.{
        .metadata = &metadata,
        .image = &image,
        .runtime = .{ .allow_borrow = false },
    }};

    try pipeline.executeLocalPipelineStep(.{
        .allocator = std.testing.allocator,
        .plan_ref = &bundle.local_stage_runner_plan_ref.?,
        .placement_plan = placement,
    }, &stages, &boundary_payloads, .decode, &.{});

    try std.testing.expectEqual(@as(usize, 5), trace_data.len);
    try std.testing.expectEqual(TraceStep.stage0_execute, trace_data.steps[0]);
    try std.testing.expectEqual(TraceStep.stage0_sync, trace_data.steps[1]);
    try std.testing.expectEqual(TraceStep.stage0_prepare, trace_data.steps[2]);
    try std.testing.expectEqual(TraceStep.stage1_upload, trace_data.steps[3]);
    try std.testing.expectEqual(TraceStep.stage1_execute, trace_data.steps[4]);
    try std.testing.expectEqual(@as(usize, 0), stage0.prepared_boundary);
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
    const image01 = pipeline.hostActivationByteImage(&metadata01, payload01[0..]);
    const image12 = pipeline.hostActivationByteImage(&metadata12, payload12[0..]);
    var trace_data = Trace{};
    var stage0 = Stage{ .trace = &trace_data, .id = 0 };
    var stage1 = Stage{ .trace = &trace_data, .id = 1 };
    var stage2 = Stage{ .trace = &trace_data, .id = 2 };
    var stages = [_]pipeline.LocalStageChainStage{
        pipeline.localStageAdapter(Stage, metadata01.boundary.source_stage_id, &stage0),
        pipeline.localStageAdapter(Stage, metadata01.boundary.target_stage_id, &stage1),
        pipeline.localStageAdapter(Stage, metadata12.boundary.target_stage_id, &stage2),
    };
    const boundary_payloads = [_]pipeline.LocalPipelineBoundaryPayload{
        .{ .metadata = &metadata01, .image = &image01, .runtime = .{ .allow_borrow = false } },
        .{ .metadata = &metadata12, .image = &image12, .runtime = .{ .allow_borrow = false } },
    };

    try pipeline.executeLocalPipelineStep(.{
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
            try validateEmptyInput(input);
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
    const image01 = pipeline.hostActivationByteImage(&metadata01, payload01[0..]);
    const image12 = pipeline.hostActivationByteImage(&metadata12, payload12[0..]);
    const image23 = pipeline.hostActivationByteImage(&metadata23, payload23[0..]);
    var trace_data = Trace{};
    var stage0 = Stage{ .trace = &trace_data, .id = 0 };
    var stage1 = Stage{ .trace = &trace_data, .id = 1 };
    var stage2 = Stage{ .trace = &trace_data, .id = 2 };
    var stage3 = Stage{ .trace = &trace_data, .id = 3 };
    var stages = [_]pipeline.LocalStageChainStage{
        pipeline.localStageAdapter(Stage, metadata01.boundary.source_stage_id, &stage0),
        pipeline.localStageAdapter(Stage, metadata01.boundary.target_stage_id, &stage1),
        pipeline.localStageAdapter(Stage, metadata12.boundary.target_stage_id, &stage2),
        pipeline.localStageAdapter(Stage, metadata23.boundary.target_stage_id, &stage3),
    };
    const boundary_payloads = [_]pipeline.LocalPipelineBoundaryPayload{
        .{ .metadata = &metadata01, .image = &image01, .runtime = .{ .allow_borrow = false } },
        .{ .metadata = &metadata12, .image = &image12, .runtime = .{ .allow_borrow = false } },
        .{ .metadata = &metadata23, .image = &image23, .runtime = .{ .allow_borrow = false } },
    };

    try pipeline.executeLocalPipelineStep(.{
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
            try validateEmptyInput(input);
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
    var stages = [_]pipeline.LocalStageChainStage{
        pipeline.localStageAdapter(Stage, runner.stages[0].stage_id, &stage0),
        pipeline.localStageAdapter(Stage, runner.stages[1].stage_id, &stage1),
        pipeline.localStageAdapter(Stage, runner.stages[2].stage_id, &stage2),
        pipeline.localStageAdapter(Stage, runner.stages[3].stage_id, &stage3),
    };
    const boundary_payloads = [_]pipeline.LocalDecodeBoundaryPayloadSpec{
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

    try pipeline.executeLocalDecodePipelineStep(.{
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

test "executeLocalDecodePipelineStep validates host segments after source execution" {
    const TraceStep = enum { source_execute, source_sync, target_upload_segments, target_execute };
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
        source_payload: []const u8 = &.{},
        host_segments: ?[][]const u8 = null,

        pub fn executeDecodeLayerRange(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            try validateEmptyInput(input);
            try std.testing.expect(layer_end > layer_start);
            if (stage.id == 0) {
                const segments = stage.host_segments orelse return error.InvalidTopologyConfig;
                segments[0] = stage.source_payload;
                stage.trace.push(TraceStep.source_execute);
            } else {
                stage.trace.push(TraceStep.target_execute);
            }
        }

        pub fn executePrefillLayerRange(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            try stage.executeDecodeLayerRange(input, layer_start, layer_end);
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            if (stage.id == 0) stage.trace.push(TraceStep.source_sync);
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(_: *@This(), _: []const u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivationSegments(stage: *@This(), host_segments: []const []const u8, byte_count: usize) anyerror!void {
            try std.testing.expectEqual(@as(usize, 1), host_segments.len);
            try std.testing.expectEqual(byte_count, host_segments[0].len);
            stage.trace.push(TraceStep.target_upload_segments);
        }
    };

    var bundle = try buildPlacement(.decode);
    defer bundle.deinit();
    var payload = [_]u8{0xb7} ** 16;
    var host_segments = [_][]const u8{&.{}};
    var slot_request_ids = [_]?u64{null} ** 4;
    slot_request_ids[0] = 101;
    const slot_indices = [_]usize{0};
    const positions = [_]usize{7};
    var trace_data = Trace{};
    var stage0 = Stage{
        .trace = &trace_data,
        .id = 0,
        .source_payload = payload[0..],
        .host_segments = host_segments[0..],
    };
    var stage1 = Stage{ .trace = &trace_data, .id = 1 };
    const runner = &bundle.local_stage_runner_plan_ref.?;
    var endpoints = [_]pipeline.LocalStageEndpoint{
        pipeline.localStageAdapter(Stage, runner.stages[0].stage_id, &stage0),
        pipeline.localStageAdapter(Stage, runner.stages[1].stage_id, &stage1),
    };
    const boundary_payloads = [_]pipeline.LocalDecodeBoundaryPayloadSpec{.{
        .frame = .{ .boundary_index = 0, .dtype = .f32, .layout = .row_major },
        .activation_byte_count = payload.len,
        .location_hint = .cpu,
        .image = .{ .host_segments = host_segments[0..] },
    }};

    try pipeline.executeLocalDecodePipelineStepWithEndpointRegistry(.{
        .allocator = std.testing.allocator,
        .plan_ref = runner,
        .placement_plan = &bundle.placement_plan.?,
    }, .{ .endpoints = &endpoints }, .{
        .tensor_frame_plan_ref = &bundle.tensor_frame_plan_ref.?,
        .hidden_size = 4,
        .slot_request_ids = &slot_request_ids,
        .slot_indices = &slot_indices,
        .positions = &positions,
        .boundary_payloads = &boundary_payloads,
    }, false);

    try std.testing.expectEqual(@as(usize, 4), trace_data.len);
    try std.testing.expectEqual(TraceStep.source_execute, trace_data.steps[0]);
    try std.testing.expectEqual(TraceStep.source_sync, trace_data.steps[1]);
    try std.testing.expectEqual(TraceStep.target_upload_segments, trace_data.steps[2]);
    try std.testing.expectEqual(TraceStep.target_execute, trace_data.steps[3]);
}

test "executeLocalPipelineStepWithEndpointRegistry executes five stage chain by stage id" {
    const Phase = enum(u8) { execute, sync, upload, project };
    const Trace = struct {
        steps: [20]u8 = undefined,
        len: usize = 0,

        fn code(stage_id: usize, phase: Phase) u8 {
            return @intCast(stage_id * 4 + @intFromEnum(phase));
        }

        fn push(trace_data: *@This(), stage_id: usize, phase: Phase) void {
            trace_data.steps[trace_data.len] = code(stage_id, phase);
            trace_data.len += 1;
        }
    };
    const Stage = struct {
        trace: *Trace,
        id: usize,

        pub fn executeDecodeLayerRange(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            try validateEmptyInput(input);
            try std.testing.expect(layer_end > layer_start);
            stage.trace.push(stage.id, .execute);
        }

        pub fn executePrefillLayerRange(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            try stage.executeDecodeLayerRange(input, layer_start, layer_end);
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            stage.trace.push(stage.id, .sync);
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            stage.trace.push(stage.id, .upload);
        }

        pub fn projectFinalLogits(stage: *@This()) anyerror!void {
            stage.trace.push(stage.id, .project);
        }
    };

    var bundle = try buildFiveStagePlacement();
    defer bundle.deinit();
    const placement = &bundle.placement_plan.?;
    var metadata01 = try metadataForBoundary(placement, 0, .decode);
    var metadata12 = try metadataForBoundary(placement, 1, .decode);
    var metadata23 = try metadataForBoundary(placement, 2, .decode);
    var metadata34 = try metadataForBoundary(placement, 3, .decode);
    var payload01 = [_]u8{0xd1} ** 16;
    var payload12 = [_]u8{0xd2} ** 16;
    var payload23 = [_]u8{0xd3} ** 16;
    var payload34 = [_]u8{0xd4} ** 16;
    const image01 = pipeline.hostActivationByteImage(&metadata01, payload01[0..]);
    const image12 = pipeline.hostActivationByteImage(&metadata12, payload12[0..]);
    const image23 = pipeline.hostActivationByteImage(&metadata23, payload23[0..]);
    const image34 = pipeline.hostActivationByteImage(&metadata34, payload34[0..]);
    var trace_data = Trace{};
    var stage0 = Stage{ .trace = &trace_data, .id = 0 };
    var stage1 = Stage{ .trace = &trace_data, .id = 1 };
    var stage2 = Stage{ .trace = &trace_data, .id = 2 };
    var stage3 = Stage{ .trace = &trace_data, .id = 3 };
    var stage4 = Stage{ .trace = &trace_data, .id = 4 };
    var endpoints = [_]pipeline.LocalStageEndpoint{
        pipeline.localStageAdapter(Stage, 3, &stage3),
        pipeline.localStageAdapter(Stage, 1, &stage1),
        pipeline.localStageAdapter(Stage, 4, &stage4),
        pipeline.localStageAdapter(Stage, 0, &stage0),
        pipeline.localStageAdapter(Stage, 2, &stage2),
    };
    const boundary_payloads = [_]pipeline.LocalPipelineBoundaryPayload{
        .{ .metadata = &metadata01, .image = &image01, .runtime = .{ .allow_borrow = false } },
        .{ .metadata = &metadata12, .image = &image12, .runtime = .{ .allow_borrow = false } },
        .{ .metadata = &metadata23, .image = &image23, .runtime = .{ .allow_borrow = false } },
        .{ .metadata = &metadata34, .image = &image34, .runtime = .{ .allow_borrow = false } },
    };

    try pipeline.executeLocalPipelineStepWithEndpointRegistry(.{
        .allocator = std.testing.allocator,
        .plan_ref = &bundle.local_stage_runner_plan_ref.?,
        .placement_plan = placement,
    }, .{ .endpoints = &endpoints }, &boundary_payloads, .decode, &.{}, true);

    const expected = [_]u8{
        Trace.code(0, .execute),
        Trace.code(0, .sync),
        Trace.code(1, .upload),
        Trace.code(1, .execute),
        Trace.code(1, .sync),
        Trace.code(2, .upload),
        Trace.code(2, .execute),
        Trace.code(2, .sync),
        Trace.code(3, .upload),
        Trace.code(3, .execute),
        Trace.code(3, .sync),
        Trace.code(4, .upload),
        Trace.code(4, .execute),
        Trace.code(4, .project),
    };
    try std.testing.expectEqualSlices(u8, &expected, trace_data.steps[0..trace_data.len]);
}

test "executeLocalPipelineStepWithEndpointRegistry rejects duplicate and missing endpoints before mutation" {
    const Trace = struct {
        mutations: usize = 0,
    };
    const Stage = struct {
        trace: *Trace,

        pub fn executeDecodeLayerRange(stage: *@This(), _: []const u8, _: usize, _: usize) anyerror!void {
            stage.trace.mutations += 1;
        }

        pub fn executePrefillLayerRange(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            try stage.executeDecodeLayerRange(input, layer_start, layer_end);
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            stage.trace.mutations += 1;
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(stage: *@This(), _: []const u8, _: usize) anyerror!void {
            stage.trace.mutations += 1;
        }
    };

    var bundle = try buildFiveStagePlacement();
    defer bundle.deinit();
    const placement = &bundle.placement_plan.?;
    var metadata01 = try metadataForBoundary(placement, 0, .decode);
    var metadata12 = try metadataForBoundary(placement, 1, .decode);
    var metadata23 = try metadataForBoundary(placement, 2, .decode);
    var metadata34 = try metadataForBoundary(placement, 3, .decode);
    var payload = [_]u8{0xe1} ** 16;
    const image01 = pipeline.hostActivationByteImage(&metadata01, payload[0..]);
    const image12 = pipeline.hostActivationByteImage(&metadata12, payload[0..]);
    const image23 = pipeline.hostActivationByteImage(&metadata23, payload[0..]);
    const image34 = pipeline.hostActivationByteImage(&metadata34, payload[0..]);
    const boundary_payloads = [_]pipeline.LocalPipelineBoundaryPayload{
        .{ .metadata = &metadata01, .image = &image01, .runtime = .{ .allow_borrow = false } },
        .{ .metadata = &metadata12, .image = &image12, .runtime = .{ .allow_borrow = false } },
        .{ .metadata = &metadata23, .image = &image23, .runtime = .{ .allow_borrow = false } },
        .{ .metadata = &metadata34, .image = &image34, .runtime = .{ .allow_borrow = false } },
    };

    var trace_data = Trace{};
    var stage0 = Stage{ .trace = &trace_data };
    var stage1 = Stage{ .trace = &trace_data };
    var stage2 = Stage{ .trace = &trace_data };
    var stage3 = Stage{ .trace = &trace_data };
    var stage4 = Stage{ .trace = &trace_data };
    var duplicate_endpoints = [_]pipeline.LocalStageEndpoint{
        pipeline.localStageAdapter(Stage, 0, &stage0),
        pipeline.localStageAdapter(Stage, 0, &stage1),
        pipeline.localStageAdapter(Stage, 2, &stage2),
        pipeline.localStageAdapter(Stage, 3, &stage3),
        pipeline.localStageAdapter(Stage, 4, &stage4),
    };

    try std.testing.expectError(
        error.DuplicateStageRef,
        pipeline.executeLocalPipelineStepWithEndpointRegistry(.{
            .allocator = std.testing.allocator,
            .plan_ref = &bundle.local_stage_runner_plan_ref.?,
            .placement_plan = placement,
        }, .{ .endpoints = &duplicate_endpoints }, &boundary_payloads, .decode, &.{}, false),
    );
    try std.testing.expectEqual(@as(usize, 0), trace_data.mutations);

    var missing_endpoints = [_]pipeline.LocalStageEndpoint{
        pipeline.localStageAdapter(Stage, 0, &stage0),
        pipeline.localStageAdapter(Stage, 9, &stage1),
        pipeline.localStageAdapter(Stage, 2, &stage2),
        pipeline.localStageAdapter(Stage, 3, &stage3),
        pipeline.localStageAdapter(Stage, 4, &stage4),
    };
    try std.testing.expectError(
        error.MissingStageRef,
        pipeline.executeLocalPipelineStepWithEndpointRegistry(.{
            .allocator = std.testing.allocator,
            .plan_ref = &bundle.local_stage_runner_plan_ref.?,
            .placement_plan = placement,
        }, .{ .endpoints = &missing_endpoints }, &boundary_payloads, .decode, &.{}, false),
    );
    try std.testing.expectEqual(@as(usize, 0), trace_data.mutations);
}

test "executeLocalPipelineStepWithEndpointRegistry validates all boundaries before mutation" {
    const Trace = struct {
        mutations: usize = 0,
    };
    const Stage = struct {
        trace: *Trace,

        pub fn executeDecodeLayerRange(stage: *@This(), _: []const u8, _: usize, _: usize) anyerror!void {
            stage.trace.mutations += 1;
        }

        pub fn executePrefillLayerRange(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            try stage.executeDecodeLayerRange(input, layer_start, layer_end);
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            stage.trace.mutations += 1;
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(stage: *@This(), _: []const u8, _: usize) anyerror!void {
            stage.trace.mutations += 1;
        }
    };

    var bundle = try buildThreeStagePlacement();
    defer bundle.deinit();
    const placement = &bundle.placement_plan.?;
    var metadata01 = try metadataForBoundary(placement, 0, .decode);
    var metadata12 = try metadataForBoundary(placement, 1, .decode);
    var payload01 = [_]u8{0xf1} ** 16;
    const image01 = pipeline.hostActivationByteImage(&metadata01, payload01[0..]);
    var trace_data = Trace{};
    var stage0 = Stage{ .trace = &trace_data };
    var stage1 = Stage{ .trace = &trace_data };
    var stage2 = Stage{ .trace = &trace_data };
    var endpoints = [_]pipeline.LocalStageEndpoint{
        pipeline.localStageAdapter(Stage, 0, &stage0),
        pipeline.localStageAdapter(Stage, 1, &stage1),
        pipeline.localStageAdapter(Stage, 2, &stage2),
    };
    const boundary_payloads = [_]pipeline.LocalPipelineBoundaryPayload{
        .{ .metadata = &metadata01, .image = &image01, .runtime = .{ .allow_borrow = false } },
        .{ .metadata = &metadata12, .image = &image01, .runtime = .{ .allow_borrow = false } },
    };

    try std.testing.expectError(
        error.BoundaryTensorContractMismatch,
        pipeline.executeLocalPipelineStepWithEndpointRegistry(.{
            .allocator = std.testing.allocator,
            .plan_ref = &bundle.local_stage_runner_plan_ref.?,
            .placement_plan = placement,
        }, .{ .endpoints = &endpoints }, &boundary_payloads, .decode, &.{}, false),
    );
    try std.testing.expectEqual(@as(usize, 0), trace_data.mutations);
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
            try validateEmptyInput(input);
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
    var stages = [_]pipeline.LocalStageChainStage{
        pipeline.localStageAdapter(Stage, runner.stages[0].stage_id, &stage0),
        pipeline.localStageAdapter(Stage, runner.stages[1].stage_id, &stage1),
    };
    const boundary_payloads = [_]pipeline.LocalPrefillBoundaryPayloadSpec{.{
        .frame = .{ .boundary_index = 0, .dtype = .f32, .layout = .row_major },
        .slot_index = 1,
        .sequence_start = 3,
        .token_count = 1,
        .activation_byte_count = payload.len,
        .location_hint = .cpu,
        .image = .{ .host_bytes = payload[0..] },
    }};

    try pipeline.executeLocalPrefillPipelineStep(.{
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
    const image01 = pipeline.hostActivationByteImage(&metadata01, payload01[0..]);
    const image12 = pipeline.hostActivationByteImage(&metadata12, payload12[0..]);
    var trace_data = Trace{};
    var stage0 = Stage{ .trace = &trace_data, .id = 0 };
    var stage1 = Stage{ .trace = &trace_data, .id = 1, .fail_upload = true };
    var stage2 = Stage{ .trace = &trace_data, .id = 2 };
    var stages = [_]pipeline.LocalStageChainStage{
        pipeline.localStageAdapter(Stage, metadata01.boundary.source_stage_id, &stage0),
        pipeline.localStageAdapter(Stage, metadata01.boundary.target_stage_id, &stage1),
        pipeline.localStageAdapter(Stage, metadata12.boundary.target_stage_id, &stage2),
    };
    const boundary_payloads = [_]pipeline.LocalPipelineBoundaryPayload{
        .{ .metadata = &metadata01, .image = &image01, .runtime = .{ .allow_borrow = false } },
        .{ .metadata = &metadata12, .image = &image12, .runtime = .{ .allow_borrow = false } },
    };

    try std.testing.expectError(
        error.AccessDenied,
        pipeline.executeLocalPipelineStep(.{
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
            try validateEmptyInput(input);
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

const test_entries = [_]pipeline.TensorFrameBatchEntry{.{
    .batch_index = 0,
    .request_id = 101,
    .slot_id = 7,
    .sequence_start = 3,
    .token_count = 1,
}};

fn buildPlacement(step_kind: pipeline.TensorFrameStepKind) !local_stage_testing.LocalStageContractBundle {
    _ = step_kind;
    const plan = try buildTestStagePlan(std.testing.allocator, 4, &.{2});
    const kinds = [_]pipeline.HostBackendKind{ .cpu, .cuda };
    const configs = [_]local_stage_testing.BoundaryConfig{
        local_stage_testing.localBoundaryConfig(.f32, .row_major, 4, 1),
    };
    var bundle = try local_stage_testing.buildLocalStageContractBundleFromOwnedPlan(
        std.testing.allocator,
        4,
        plan,
        &kinds,
        &configs,
    );
    errdefer bundle.deinit();
    return bundle;
}

fn buildDevicePairPlacement() !local_stage_testing.LocalStageContractBundle {
    return buildTwoStagePlacementWithKinds(.cuda, .cuda);
}

fn buildTwoStagePlacementWithKinds(
    source_kind: pipeline.HostBackendKind,
    target_kind: pipeline.HostBackendKind,
) !local_stage_testing.LocalStageContractBundle {
    const plan = try buildTestStagePlan(std.testing.allocator, 4, &.{2});
    const kinds = [_]pipeline.HostBackendKind{ source_kind, target_kind };
    const configs = [_]local_stage_testing.BoundaryConfig{
        local_stage_testing.localBoundaryConfig(.f32, .row_major, 4, 1),
    };
    return local_stage_testing.buildLocalStageContractBundleFromOwnedPlan(
        std.testing.allocator,
        4,
        plan,
        &kinds,
        &configs,
    );
}

fn buildThreeStagePlacement() !local_stage_testing.LocalStageContractBundle {
    const plan = try buildTestStagePlan(std.testing.allocator, 4, &.{ 2, 3 });
    const kinds = [_]pipeline.HostBackendKind{ .cpu, .cuda, .cuda };
    const configs = local_stage_testing.localTwoBoundaryConfigs(.f32, .row_major, .f32, .row_major, 4, 4, 1, 1);
    return local_stage_testing.buildLocalStageContractBundleFromOwnedPlan(
        std.testing.allocator,
        4,
        plan,
        &kinds,
        &configs,
    );
}

fn buildFourStagePlacement() !local_stage_testing.LocalStageContractBundle {
    const plan = try buildTestStagePlan(std.testing.allocator, 4, &.{ 1, 2, 3 });
    const kinds = [_]pipeline.HostBackendKind{ .cuda, .cpu, .cuda, .cpu };
    const configs = [_]local_stage_testing.BoundaryConfig{
        local_stage_testing.localBoundaryConfig(.f32, .row_major, 4, 1),
        local_stage_testing.localBoundaryConfig(.f32, .row_major, 4, 1),
        local_stage_testing.localBoundaryConfig(.f32, .row_major, 4, 1),
    };
    return local_stage_testing.buildLocalStageContractBundleFromOwnedPlan(
        std.testing.allocator,
        4,
        plan,
        &kinds,
        &configs,
    );
}

fn buildFiveStagePlacement() !local_stage_testing.LocalStageContractBundle {
    const plan = try buildTestStagePlan(std.testing.allocator, 5, &.{ 1, 2, 3, 4 });
    const kinds = [_]pipeline.HostBackendKind{ .cpu, .cuda, .metal, .cuda, .cpu };
    const configs = [_]local_stage_testing.BoundaryConfig{
        local_stage_testing.localBoundaryConfig(.f32, .row_major, 4, 1),
        local_stage_testing.localBoundaryConfig(.f32, .row_major, 4, 1),
        local_stage_testing.localBoundaryConfig(.f32, .row_major, 4, 1),
        local_stage_testing.localBoundaryConfig(.f32, .row_major, 4, 1),
    };
    return local_stage_testing.buildLocalStageContractBundleFromOwnedPlan(
        std.testing.allocator,
        4,
        plan,
        &kinds,
        &configs,
    );
}

fn metadataForBoundary(
    placement: *const pipeline.PlacementPlan,
    boundary_index: usize,
    step_kind: pipeline.TensorFrameStepKind,
) !pipeline.TensorFrameMetadata {
    const summary = placement.boundary_summaries[boundary_index];
    const boundary = pipeline.TensorFrameBoundaryRef{
        .boundary_index = summary.boundary_index,
        .source_stage_id = summary.source_stage_id,
        .target_stage_id = summary.target_stage_id,
        .producer_layer_start = summary.producer_layer_start,
        .producer_layer_end = summary.producer_layer_end,
        .consumer_layer_start = summary.consumer_layer_start,
        .consumer_layer_end = summary.consumer_layer_end,
    };
    const tensor_desc = try pipeline.TensorFrameTensorDesc.contiguousActivation(.f32, .{ 1, 1, 4, 0 });
    return .{
        .frame_id = try pipeline.TensorFrameInstanceId.init(77 + boundary_index),
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
