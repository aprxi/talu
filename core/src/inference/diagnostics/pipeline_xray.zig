//! Tensor-frame observer adapter for xray staged-frame diagnostics.
//!
//! This module is the boundary where inference-owned tensor-frame metadata is
//! copied into xray-owned diagnostic storage. It does not read payload bytes or
//! install any global xray tracing state.

const std = @import("std");
const models = @import("models_pkg");
const pipeline = @import("../pipeline/root.zig");
const xray = @import("xray_pkg");

const stage_plan = models.stage_plan;

pub const XrayStagedFrameError =
    std.mem.Allocator.Error ||
    pipeline.TensorFrameValidationError ||
    error{
        MissingObserverContext,
        UnsupportedFrameStepKind,
        UnsupportedFrameDType,
        UnsupportedFrameLayout,
    };

pub const XrayStagedFrameByteImageError =
    XrayStagedFrameError ||
    pipeline.BoundaryByteImageError ||
    error{
        XrayByteImageMetadataMismatch,
        XrayByteImageTransferDecisionMismatch,
    };

pub const XrayStagedFrameByteImageOptions = struct {
    transfer_decision: ?pipeline.StageTransferModeDecision = null,
};

pub const XrayStagedFrameObserver = struct {
    capture: *xray.staged_frame.StagedFrameCapture,

    pub fn observer(self: *XrayStagedFrameObserver) pipeline.TensorFrameObserver {
        return .{ .ctx = self, .emit_fn = emitTensorFrame };
    }

    fn emitTensorFrame(ctx: ?*anyopaque, metadata: *const pipeline.TensorFrameMetadata) anyerror!void {
        const observer_ctx = ctx orelse return error.MissingObserverContext;
        const self: *XrayStagedFrameObserver = @ptrCast(@alignCast(observer_ctx));
        try appendTensorFrameMetadata(self.capture, metadata);
    }
};

pub fn stagedFrameByteImageFactsFromPipeline(
    image: *const pipeline.BoundaryByteImageRef,
    options: XrayStagedFrameByteImageOptions,
) XrayStagedFrameByteImageError!xray.staged_frame.StagedFrameByteImageFacts {
    try pipeline.validateBoundaryByteImage(image, .{});

    if (options.transfer_decision) |decision| {
        if (!transferDecisionProfileMatchesImage(decision, image)) {
            return error.XrayByteImageTransferDecisionMismatch;
        }
        if (!transferDecisionModeConsistent(decision, image)) {
            return error.XrayByteImageTransferDecisionMismatch;
        }
    }

    return .{
        .readiness = stagedFrameByteImageReadiness(image.readiness),
        .transfer_mode = if (options.transfer_decision) |decision|
            stagedFrameTransferMode(decision.mode)
        else
            .unknown,
        .host_readable = image.readiness == .host_readable_now,
        .remote_readable = pipeline.boundaryByteImageIsRemoteReadable(image),
        .device_download_required = image.readiness == .device_download_required,
    };
}

pub fn appendTensorFrameByteImageMetadata(
    capture: *xray.staged_frame.StagedFrameCapture,
    metadata: *const pipeline.TensorFrameMetadata,
    image: *const pipeline.BoundaryByteImageRef,
    options: XrayStagedFrameByteImageOptions,
) XrayStagedFrameByteImageError!void {
    try metadata.validate();
    if (image.metadata != metadata) return error.XrayByteImageMetadataMismatch;
    const facts = try stagedFrameByteImageFactsFromPipeline(image, options);
    for (metadata.batch.entries) |entry| {
        var record = try stagedFrameRecordFromTensorFrame(metadata, entry);
        record.byte_image_readiness = facts.readiness;
        record.transfer_mode = facts.transfer_mode;
        record.host_readable = facts.host_readable;
        record.remote_readable = facts.remote_readable;
        record.device_download_required = facts.device_download_required;
        try capture.append(record);
    }
}

pub fn appendTensorFrameMetadata(
    capture: *xray.staged_frame.StagedFrameCapture,
    metadata: *const pipeline.TensorFrameMetadata,
) XrayStagedFrameError!void {
    try metadata.validate();
    if (metadata.batch.entries.len == 0) return error.InvalidBatch;
    for (metadata.batch.entries) |entry| {
        try capture.append(try stagedFrameRecordFromTensorFrame(metadata, entry));
    }
}

pub fn stagedFrameRecordFromTensorFrame(
    metadata: *const pipeline.TensorFrameMetadata,
    batch_entry: pipeline.TensorFrameBatchEntry,
) XrayStagedFrameError!xray.staged_frame.StagedFrameRecord {
    try metadata.validate();
    return .{
        .graph_digest = metadata.plan.graph_digest,
        .graph_contract_version = metadata.plan.graph_contract_version,
        .stage_plan_contract_version = metadata.plan.stage_plan_contract_version,
        .stage_plan_id_digest = metadata.plan.stage_plan_id.digest,
        .frame_id = metadata.frame_id.value,
        .boundary_index = metadata.boundary.boundary_index,
        .source_stage_id = metadata.boundary.source_stage_id,
        .target_stage_id = metadata.boundary.target_stage_id,
        .producer_layer_start = metadata.boundary.producer_layer_start,
        .producer_layer_end = metadata.boundary.producer_layer_end,
        .consumer_layer_start = metadata.boundary.consumer_layer_start,
        .consumer_layer_end = metadata.boundary.consumer_layer_end,
        .step_kind = try stagedFrameStepKind(metadata.step_kind),
        .dtype = try stagedFrameDType(metadata.tensor.dtype),
        .layout = try stagedFrameLayout(metadata.tensor.layout),
        .shape = metadata.tensor.shape,
        .rank = metadata.tensor.rank,
        .payload_byte_count = metadata.payload.byte_count,
        .batch_index = batch_entry.batch_index,
        .request_id = batch_entry.request_id,
        .slot_id = batch_entry.slot_id,
        .sequence_start = batch_entry.sequence_start,
        .token_count = batch_entry.token_count,
        .payload_location = stagedFramePayloadLocation(metadata.payload.location_hint),
    };
}

fn stagedFrameStepKind(kind: pipeline.TensorFrameStepKind) XrayStagedFrameError!xray.staged_frame.StagedFrameStepKind {
    return switch (kind) {
        .prefill => .prefill,
        .decode => .decode,
        else => error.UnsupportedFrameStepKind,
    };
}

fn stagedFrameDType(dtype: pipeline.TensorFrameDType) XrayStagedFrameError!xray.staged_frame.StagedFrameDType {
    return switch (dtype) {
        .bf16 => .bf16,
        .f16 => .f16,
        .f32 => .f32,
        else => error.UnsupportedFrameDType,
    };
}

fn stagedFrameLayout(layout: pipeline.TensorFrameLayout) XrayStagedFrameError!xray.staged_frame.StagedFrameLayout {
    return switch (layout) {
        .row_major => .row_major,
        else => error.UnsupportedFrameLayout,
    };
}

fn stagedFramePayloadLocation(location: ?pipeline.TensorFramePayloadLocationHint) xray.staged_frame.StagedFramePayloadLocation {
    const payload = location orelse return .none;
    return switch (payload) {
        .cpu => .cpu,
        .cuda => |ordinal| .{ .cuda = ordinal },
        .metal => |ordinal| .{ .metal = ordinal },
        .opaque_local => |value| .{ .opaque_local = value },
    };
}

fn stagedFrameByteImageReadiness(
    readiness: pipeline.BoundaryByteImageReadiness,
) xray.staged_frame.StagedFrameByteImageReadiness {
    return switch (readiness) {
        .host_readable_now => .host_readable_now,
        .producer_sync_required => .producer_sync_required,
        .device_download_required => .device_download_required,
        .local_only_opaque => .local_only_opaque,
    };
}

fn stagedFrameTransferMode(
    mode: pipeline.StageTransferMode,
) xray.staged_frame.StagedFrameTransferMode {
    return switch (mode) {
        .borrow_in_process => .borrow_in_process,
        .copy_in_process => .copy_in_process,
        .device_download_then_copy => .device_download_then_copy,
        .device_peer_copy_in_process => .device_peer_copy_in_process,
        .remote_stream => .remote_stream,
        .device_download_then_remote_stream => .device_download_then_remote_stream,
    };
}

fn transferDecisionProfileMatchesImage(
    decision: pipeline.StageTransferModeDecision,
    image: *const pipeline.BoundaryByteImageRef,
) bool {
    const profile = decision.boundary_profile;
    const metadata = image.metadata;
    return profile.boundary_index == metadata.boundary.boundary_index and
        profile.source_stage_id == metadata.boundary.source_stage_id and
        profile.target_stage_id == metadata.boundary.target_stage_id and
        profile.step_kind == metadata.step_kind and
        profile.dtype == metadata.tensor.dtype and
        profile.layout == metadata.tensor.layout;
}

fn transferDecisionModeConsistent(
    decision: pipeline.StageTransferModeDecision,
    image: *const pipeline.BoundaryByteImageRef,
) bool {
    return switch (image.readiness) {
        .host_readable_now => switch (decision.boundary_profile.handoff_mode) {
            .same_host_direct => if (image.ownership == .borrowed_until_next_stage_call)
                decision.mode == .borrow_in_process or decision.mode == .copy_in_process
            else
                decision.mode == .copy_in_process,
            .local_in_process, .mock => decision.mode == .copy_in_process,
            .remote_declared => decision.mode == .remote_stream,
        },
        .device_download_required => switch (decision.boundary_profile.handoff_mode) {
            .same_host_direct, .local_in_process => decision.mode == .device_download_then_copy or
                decision.mode == .device_peer_copy_in_process,
            .mock => decision.mode == .device_download_then_copy,
            .remote_declared => decision.mode == .device_download_then_remote_stream,
        },
        .producer_sync_required, .local_only_opaque => false,
    };
}

const test_single_batch = [_]pipeline.TensorFrameBatchEntry{.{
    .batch_index = 0,
    .request_id = 3,
    .slot_id = 7,
    .sequence_start = 9,
    .token_count = 1,
}};

const test_second_boundary_batch = [_]pipeline.TensorFrameBatchEntry{.{
    .batch_index = 0,
    .request_id = 3,
    .slot_id = 7,
    .sequence_start = 10,
    .token_count = 1,
}};

const test_multi_batch = [_]pipeline.TensorFrameBatchEntry{
    .{ .batch_index = 0, .request_id = 3, .slot_id = 7, .sequence_start = 9, .token_count = 1 },
    .{ .batch_index = 1, .request_id = 4, .slot_id = 8, .sequence_start = 9, .token_count = 1 },
};

fn testFrame(
    plan_ref: *const pipeline.TensorFramePlanRef,
    boundary_index: usize,
    frame_id: u64,
    entries: []const pipeline.TensorFrameBatchEntry,
    location_hint: ?pipeline.TensorFramePayloadLocationHint,
) !pipeline.TensorFrameMetadata {
    const batch_count: u64 = @intCast(entries.len);
    const contract = try pipeline.selectedBoundaryTensorContract(plan_ref, boundary_index, .f32, .row_major, .explicit);
    const tensor = try pipeline.TensorFrameTensorDesc.contiguousActivation(.f32, .{ batch_count, 1, 8, 0 });
    return pipeline.activationDecodeFrame(.{
        .frame_id = try pipeline.TensorFrameInstanceId.init(frame_id),
        .plan_ref = plan_ref,
        .boundary_index = boundary_index,
        .selected_contract = &contract,
        .shape_context = .{ .expected_hidden_size = 8, .expected_step_kind = .decode },
        .tensor = tensor,
        .batch = .{ .entries = entries },
        .payload = .{
            .byte_count = tensor.payload_byte_count,
            .location_hint = location_hint,
            .ownership = .borrowed_until_next_stage_call,
            .lifetime = .step_scoped,
        },
    });
}

fn testByteImage(
    metadata: *const pipeline.TensorFrameMetadata,
    readiness: pipeline.BoundaryByteImageReadiness,
    host_bytes: ?[]const u8,
) pipeline.BoundaryByteImageRef {
    return .{
        .metadata = metadata,
        .byte_count = metadata.payload.byte_count,
        .host_bytes = host_bytes,
        .location_hint = metadata.payload.location_hint,
        .readiness = readiness,
        .ownership = metadata.payload.ownership,
        .lifetime = metadata.payload.lifetime,
    };
}

fn testTransferDecision(
    metadata: *const pipeline.TensorFrameMetadata,
    handoff_mode: pipeline.BoundaryHandoffMode,
    mode: pipeline.StageTransferMode,
) pipeline.StageTransferModeDecision {
    return .{
        .mode = mode,
        .boundary_profile = .{
            .boundary_index = metadata.boundary.boundary_index,
            .source_stage_id = metadata.boundary.source_stage_id,
            .target_stage_id = metadata.boundary.target_stage_id,
            .step_kind = metadata.step_kind,
            .dtype = metadata.tensor.dtype,
            .layout = metadata.tensor.layout,
            .max_batch_entries = 8,
            .max_token_count_per_frame = 16,
            .max_activation_payload_bytes = metadata.payload.byte_count,
            .handoff_mode = handoff_mode,
        },
        .source_host_id = .{ .value = 1 },
        .target_host_id = .{ .value = 2 },
    };
}

test "inference diagnostics stagedFrameRecordFromTensorFrame maps tensor frame metadata" {
    var fixture = try TestPlanFixture.init(std.testing.allocator);
    defer fixture.deinit();
    const metadata = try testFrame(&fixture.plan_ref, 0, 41, &test_single_batch, .{ .cuda = 2 });

    const record = try stagedFrameRecordFromTensorFrame(&metadata, metadata.batch.entries[0]);

    try std.testing.expectEqual(metadata.plan.graph_digest, record.graph_digest);
    try std.testing.expectEqual(metadata.plan.graph_contract_version, record.graph_contract_version);
    try std.testing.expectEqual(metadata.plan.stage_plan_contract_version, record.stage_plan_contract_version);
    try std.testing.expectEqual(metadata.plan.stage_plan_id.digest, record.stage_plan_id_digest);
    try std.testing.expectEqual(@as(u64, 41), record.frame_id);
    try std.testing.expectEqual(@as(usize, 0), record.boundary_index);
    try std.testing.expectEqual(@as(usize, 0), record.source_stage_id);
    try std.testing.expectEqual(@as(usize, 1), record.target_stage_id);
    try std.testing.expectEqual(@as(usize, 0), record.producer_layer_start);
    try std.testing.expectEqual(@as(usize, 1), record.producer_layer_end);
    try std.testing.expectEqual(@as(usize, 1), record.consumer_layer_start);
    try std.testing.expectEqual(@as(usize, 3), record.consumer_layer_end);
    try std.testing.expectEqual(xray.staged_frame.StagedFrameStepKind.decode, record.step_kind);
    try std.testing.expectEqual(xray.staged_frame.StagedFrameDType.f32, record.dtype);
    try std.testing.expectEqual(xray.staged_frame.StagedFrameLayout.row_major, record.layout);
    try std.testing.expectEqual([4]u64{ 1, 1, 8, 0 }, record.shape);
    try std.testing.expectEqual(@as(u8, 3), record.rank);
    try std.testing.expectEqual(@as(u64, 32), record.payload_byte_count);
    try std.testing.expectEqual(@as(u32, 0), record.batch_index);
    try std.testing.expectEqual(@as(u64, 3), record.request_id);
    try std.testing.expectEqual(@as(u64, 7), record.slot_id);
    try std.testing.expectEqual(@as(u64, 9), record.sequence_start);
    try std.testing.expectEqual(@as(u64, 1), record.token_count);
    try std.testing.expect(record.payload_location == .cuda);
    try std.testing.expectEqual(@as(u16, 2), record.payload_location.cuda);
}

test "inference diagnostics stagedFrameByteImageFactsFromPipeline maps host readable device resident and transfer mode facts" {
    var fixture = try TestPlanFixture.init(std.testing.allocator);
    defer fixture.deinit();

    const host_metadata = try testFrame(&fixture.plan_ref, 0, 49, &test_single_batch, .cpu);
    const host_bytes = [_]u8{0xaa} ** 32;
    const host_image = testByteImage(&host_metadata, .host_readable_now, &host_bytes);
    const host_decision = testTransferDecision(&host_metadata, .same_host_direct, .borrow_in_process);

    const host_facts = try stagedFrameByteImageFactsFromPipeline(&host_image, .{
        .transfer_decision = host_decision,
    });
    try std.testing.expectEqual(xray.staged_frame.StagedFrameByteImageReadiness.host_readable_now, host_facts.readiness);
    try std.testing.expectEqual(xray.staged_frame.StagedFrameTransferMode.borrow_in_process, host_facts.transfer_mode);
    try std.testing.expect(host_facts.host_readable);
    try std.testing.expect(host_facts.remote_readable);
    try std.testing.expect(!host_facts.device_download_required);

    const device_metadata = try testFrame(&fixture.plan_ref, 0, 50, &test_single_batch, .{ .cuda = 0 });
    const device_image = testByteImage(&device_metadata, .device_download_required, null);
    const device_decision = testTransferDecision(&device_metadata, .local_in_process, .device_download_then_copy);

    const device_facts = try stagedFrameByteImageFactsFromPipeline(&device_image, .{
        .transfer_decision = device_decision,
    });
    try std.testing.expectEqual(xray.staged_frame.StagedFrameByteImageReadiness.device_download_required, device_facts.readiness);
    try std.testing.expectEqual(xray.staged_frame.StagedFrameTransferMode.device_download_then_copy, device_facts.transfer_mode);
    try std.testing.expect(!device_facts.host_readable);
    try std.testing.expect(!device_facts.remote_readable);
    try std.testing.expect(device_facts.device_download_required);

    const peer_decision = testTransferDecision(&device_metadata, .local_in_process, .device_peer_copy_in_process);
    const peer_facts = try stagedFrameByteImageFactsFromPipeline(&device_image, .{
        .transfer_decision = peer_decision,
    });
    try std.testing.expectEqual(xray.staged_frame.StagedFrameTransferMode.device_peer_copy_in_process, peer_facts.transfer_mode);
    try std.testing.expect(peer_facts.device_download_required);

    const remote_decision = testTransferDecision(&host_metadata, .remote_declared, .remote_stream);
    const remote_facts = try stagedFrameByteImageFactsFromPipeline(&host_image, .{
        .transfer_decision = remote_decision,
    });
    try std.testing.expectEqual(xray.staged_frame.StagedFrameTransferMode.remote_stream, remote_facts.transfer_mode);
}

test "inference diagnostics stagedFrameByteImageFactsFromPipeline rejects invalid image mismatched transfer decision facts and inconsistent transfer mode" {
    var fixture = try TestPlanFixture.init(std.testing.allocator);
    defer fixture.deinit();
    const metadata = try testFrame(&fixture.plan_ref, 0, 51, &test_single_batch, .cpu);
    const host_bytes = [_]u8{0xbb} ** 32;

    var invalid_image = testByteImage(&metadata, .host_readable_now, &host_bytes);
    invalid_image.byte_count = 0;
    try std.testing.expectError(error.InvalidPayloadByteCount, stagedFrameByteImageFactsFromPipeline(&invalid_image, .{}));

    const valid_image = testByteImage(&metadata, .host_readable_now, &host_bytes);
    var mismatched_decision = testTransferDecision(&metadata, .same_host_direct, .borrow_in_process);
    mismatched_decision.boundary_profile.dtype = .f16;
    try std.testing.expectError(error.XrayByteImageTransferDecisionMismatch, stagedFrameByteImageFactsFromPipeline(&valid_image, .{
        .transfer_decision = mismatched_decision,
    }));

    const inconsistent_decision = testTransferDecision(&metadata, .local_in_process, .remote_stream);
    try std.testing.expectError(error.XrayByteImageTransferDecisionMismatch, stagedFrameByteImageFactsFromPipeline(&valid_image, .{
        .transfer_decision = inconsistent_decision,
    }));

    const device_metadata = try testFrame(&fixture.plan_ref, 0, 52, &test_single_batch, .{ .cuda = 0 });
    const device_image = testByteImage(&device_metadata, .device_download_required, null);
    const mock_peer_decision = testTransferDecision(&device_metadata, .mock, .device_peer_copy_in_process);
    try std.testing.expectError(error.XrayByteImageTransferDecisionMismatch, stagedFrameByteImageFactsFromPipeline(&device_image, .{
        .transfer_decision = mock_peer_decision,
    }));
}

test "inference diagnostics appendTensorFrameMetadata appends one record per batch entry" {
    var fixture = try TestPlanFixture.init(std.testing.allocator);
    defer fixture.deinit();
    const metadata = try testFrame(&fixture.plan_ref, 0, 42, &test_multi_batch, .cpu);
    var capture = xray.staged_frame.StagedFrameCapture.init(std.testing.allocator);
    defer capture.deinit();

    try appendTensorFrameMetadata(&capture, &metadata);

    try std.testing.expectEqual(@as(usize, 2), capture.count());
    try std.testing.expectEqual(@as(u32, 0), capture.get(0).?.batch_index);
    try std.testing.expectEqual(@as(u32, 1), capture.get(1).?.batch_index);
    try std.testing.expectEqual(@as(u64, 3), capture.get(0).?.request_id);
    try std.testing.expectEqual(@as(u64, 4), capture.get(1).?.request_id);
    try std.testing.expect(capture.get(0).?.payload_location == .cpu);
    try std.testing.expect(capture.get(1).?.payload_location == .cpu);
}

test "inference diagnostics appendTensorFrameByteImageMetadata appends byte image facts without reading payload bytes" {
    var fixture = try TestPlanFixture.init(std.testing.allocator);
    defer fixture.deinit();
    const metadata = try testFrame(&fixture.plan_ref, 0, 52, &test_single_batch, .cpu);
    const host_bytes = [_]u8{0xcc} ** 32;
    const image = testByteImage(&metadata, .host_readable_now, &host_bytes);
    const decision = testTransferDecision(&metadata, .same_host_direct, .copy_in_process);
    var capture = xray.staged_frame.StagedFrameCapture.init(std.testing.allocator);
    defer capture.deinit();

    try appendTensorFrameByteImageMetadata(&capture, &metadata, &image, .{
        .transfer_decision = decision,
    });

    try std.testing.expectEqual(@as(usize, 1), capture.count());
    const record = capture.get(0).?;
    try std.testing.expectEqual(xray.staged_frame.StagedFrameByteImageReadiness.host_readable_now, record.byte_image_readiness);
    try std.testing.expectEqual(xray.staged_frame.StagedFrameTransferMode.copy_in_process, record.transfer_mode);
    try std.testing.expect(record.host_readable);
    try std.testing.expect(record.remote_readable);
    try std.testing.expect(!record.device_download_required);
    try std.testing.expectEqual(@as(u64, metadata.payload.byte_count), record.payload_byte_count);
}

test "inference diagnostics appendTensorFrameByteImageMetadata rejects mismatched metadata and image" {
    var fixture = try TestPlanFixture.init(std.testing.allocator);
    defer fixture.deinit();
    const image_metadata = try testFrame(&fixture.plan_ref, 0, 53, &test_single_batch, .cpu);
    const append_metadata = try testFrame(&fixture.plan_ref, 0, 54, &test_single_batch, .cpu);
    const host_bytes = [_]u8{0xdd} ** 32;
    const image = testByteImage(&image_metadata, .host_readable_now, &host_bytes);
    var capture = xray.staged_frame.StagedFrameCapture.init(std.testing.allocator);
    defer capture.deinit();

    try std.testing.expectError(error.XrayByteImageMetadataMismatch, appendTensorFrameByteImageMetadata(&capture, &append_metadata, &image, .{}));
}

test "inference diagnostics appendTensorFrameByteImageMetadata preserves batch order and base tensor frame facts" {
    var fixture = try TestPlanFixture.init(std.testing.allocator);
    defer fixture.deinit();
    const metadata = try testFrame(&fixture.plan_ref, 0, 55, &test_multi_batch, .cpu);
    const host_bytes = [_]u8{0xee} ** 64;
    const image = testByteImage(&metadata, .host_readable_now, &host_bytes);
    const decision = testTransferDecision(&metadata, .remote_declared, .remote_stream);
    var capture = xray.staged_frame.StagedFrameCapture.init(std.testing.allocator);
    defer capture.deinit();

    try appendTensorFrameByteImageMetadata(&capture, &metadata, &image, .{
        .transfer_decision = decision,
    });

    try std.testing.expectEqual(@as(usize, 2), capture.count());
    try std.testing.expectEqual(@as(u32, 0), capture.get(0).?.batch_index);
    try std.testing.expectEqual(@as(u32, 1), capture.get(1).?.batch_index);
    try std.testing.expectEqual(@as(u64, 3), capture.get(0).?.request_id);
    try std.testing.expectEqual(@as(u64, 4), capture.get(1).?.request_id);
    try std.testing.expectEqual(@as(u64, 7), capture.get(0).?.slot_id);
    try std.testing.expectEqual(@as(u64, 8), capture.get(1).?.slot_id);
    try std.testing.expectEqual(@as(u64, metadata.payload.byte_count), capture.get(0).?.payload_byte_count);
    try std.testing.expect(capture.get(0).?.payload_location == .cpu);
    try std.testing.expect(capture.get(1).?.payload_location == .cpu);
    try std.testing.expectEqual(xray.staged_frame.StagedFrameByteImageReadiness.host_readable_now, capture.get(0).?.byte_image_readiness);
    try std.testing.expectEqual(xray.staged_frame.StagedFrameTransferMode.remote_stream, capture.get(1).?.transfer_mode);
    try std.testing.expect(capture.get(0).?.host_readable);
    try std.testing.expect(capture.get(1).?.remote_readable);
}

test "inference diagnostics appendTensorFrameMetadata rejects empty batch" {
    var fixture = try TestPlanFixture.init(std.testing.allocator);
    defer fixture.deinit();
    var metadata = try testFrame(&fixture.plan_ref, 0, 43, &test_single_batch, null);
    metadata.batch = .{ .entries = &.{} };
    var capture = xray.staged_frame.StagedFrameCapture.init(std.testing.allocator);
    defer capture.deinit();

    try std.testing.expectError(error.InvalidBatch, appendTensorFrameMetadata(&capture, &metadata));
}

test "inference diagnostics XrayStagedFrameObserver.observer rejects missing context" {
    var fixture = try TestPlanFixture.init(std.testing.allocator);
    defer fixture.deinit();
    const metadata = try testFrame(&fixture.plan_ref, 0, 44, &test_single_batch, null);
    var capture = xray.staged_frame.StagedFrameCapture.init(std.testing.allocator);
    defer capture.deinit();
    var xray_observer = XrayStagedFrameObserver{ .capture = &capture };
    const observer = xray_observer.observer();

    try std.testing.expectError(error.MissingObserverContext, observer.emit_fn.?(null, &metadata));
}

test "inference diagnostics XrayStagedFrameObserver.observer captures through TensorFrameObserver" {
    var fixture = try TestPlanFixture.init(std.testing.allocator);
    defer fixture.deinit();
    const metadata = try testFrame(&fixture.plan_ref, 0, 45, &test_single_batch, .{ .metal = 1 });
    var capture = xray.staged_frame.StagedFrameCapture.init(std.testing.allocator);
    defer capture.deinit();
    var xray_observer = XrayStagedFrameObserver{ .capture = &capture };

    try pipeline.emitTensorFrame(xray_observer.observer(), .strict, &metadata);

    try std.testing.expectEqual(@as(usize, 1), capture.count());
    try std.testing.expect(capture.get(0).?.payload_location == .metal);
    try std.testing.expectEqual(@as(u16, 1), capture.get(0).?.payload_location.metal);
}

test "inference diagnostics appendTensorFrameMetadata captures two adjacent boundaries" {
    var fixture = try TestPlanFixture.init(std.testing.allocator);
    defer fixture.deinit();
    const frame0 = try testFrame(&fixture.plan_ref, 0, 46, &test_single_batch, null);
    const frame1 = try testFrame(&fixture.plan_ref, 1, 47, &test_second_boundary_batch, null);
    var capture = xray.staged_frame.StagedFrameCapture.init(std.testing.allocator);
    defer capture.deinit();

    try appendTensorFrameMetadata(&capture, &frame0);
    try appendTensorFrameMetadata(&capture, &frame1);
    try xray.staged_frame.validateAdjacentBoundarySequence(capture.records());

    try std.testing.expectEqual(@as(usize, 2), capture.count());
    try expectBoundaryRecord(capture.get(0).?.*, 0, 0, 1, 0, 1, 1, 3);
    try expectBoundaryRecord(capture.get(1).?.*, 1, 1, 2, 1, 3, 3, 4);
    try std.testing.expectEqual(@as(u64, 46), capture.get(0).?.frame_id);
    try std.testing.expectEqual(@as(u64, 47), capture.get(1).?.frame_id);
    try std.testing.expectEqual(@as(u64, 3), capture.get(0).?.request_id);
    try std.testing.expectEqual(@as(u64, 3), capture.get(1).?.request_id);
    try std.testing.expectEqual(@as(u64, 7), capture.get(0).?.slot_id);
    try std.testing.expectEqual(@as(u64, 7), capture.get(1).?.slot_id);
    try std.testing.expectEqual(xray.staged_frame.StagedFrameDType.f32, capture.get(0).?.dtype);
    try std.testing.expectEqual(xray.staged_frame.StagedFrameDType.f32, capture.get(1).?.dtype);
    try std.testing.expectEqual([4]u64{ 1, 1, 8, 0 }, capture.get(0).?.shape);
    try std.testing.expectEqual([4]u64{ 1, 1, 8, 0 }, capture.get(1).?.shape);
    try std.testing.expectEqual(@as(u64, 32), capture.get(0).?.payload_byte_count);
    try std.testing.expectEqual(@as(u64, 32), capture.get(1).?.payload_byte_count);
}

test "inference diagnostics stagedFrameRecordFromTensorFrame does not use payload bytes" {
    var fixture = try TestPlanFixture.init(std.testing.allocator);
    defer fixture.deinit();
    const metadata = try testFrame(&fixture.plan_ref, 0, 48, &test_single_batch, .{ .opaque_local = 99 });

    const record = try stagedFrameRecordFromTensorFrame(&metadata, metadata.batch.entries[0]);

    try std.testing.expectEqual(@as(u64, metadata.payload.byte_count), record.payload_byte_count);
    try std.testing.expect(record.payload_location == .opaque_local);
    try std.testing.expectEqual(@as(u32, 99), record.payload_location.opaque_local);
}

fn expectBoundaryRecord(
    record: xray.staged_frame.StagedFrameRecord,
    boundary_index: usize,
    source_stage_id: usize,
    target_stage_id: usize,
    producer_layer_start: usize,
    producer_layer_end: usize,
    consumer_layer_start: usize,
    consumer_layer_end: usize,
) !void {
    try std.testing.expectEqual(boundary_index, record.boundary_index);
    try std.testing.expectEqual(source_stage_id, record.source_stage_id);
    try std.testing.expectEqual(target_stage_id, record.target_stage_id);
    try std.testing.expectEqual(producer_layer_start, record.producer_layer_start);
    try std.testing.expectEqual(producer_layer_end, record.producer_layer_end);
    try std.testing.expectEqual(consumer_layer_start, record.consumer_layer_start);
    try std.testing.expectEqual(consumer_layer_end, record.consumer_layer_end);
}

const TestPlanFixture = struct {
    manifest: models.manifest.ModelManifest,
    plan: stage_plan.StagePlan,
    plan_ref: pipeline.TensorFramePlanRef,

    fn init(allocator: std.mem.Allocator) !TestPlanFixture {
        var manifest = try testManifest(allocator, 4);
        errdefer manifest.deinit();
        var arch = testArch();
        var config = testConfig(4);
        const split_points = [_]usize{ 1, 3 };
        var plan = try stage_plan.buildStagePlan(allocator, .{
            .n_layers = 4,
            .split_points = &split_points,
            .architecture = &arch,
            .model_config = &config,
            .manifest = &manifest,
            .partition_constraints = .{ .decoder_cuts_allowed = true },
        });
        errdefer plan.deinit();
        var plan_ref = try pipeline.TensorFramePlanRef.fromStagePlan(allocator, &plan);
        errdefer plan_ref.deinit();
        return .{
            .manifest = manifest,
            .plan = plan,
            .plan_ref = plan_ref,
        };
    }

    fn deinit(self: *TestPlanFixture) void {
        self.plan_ref.deinit();
        self.plan.deinit();
        self.manifest.deinit();
    }
};

fn testConfig(layer_count: usize) models.config.ModelConfig {
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

fn testArch() models.op_types.Architecture {
    return .{
        .name = "xray_staged_frame_test",
        .model_types = &.{"xray_staged_frame_test"},
    };
}

fn testManifest(allocator: std.mem.Allocator, layer_count: usize) !models.manifest.ModelManifest {
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
        .architecture_id = "xray_staged_frame_test",
        .layer_count = layer_count,
        .entries = entries,
        .total_checkpoint_bytes = total_bytes,
        .role_bytes = role_bytes,
    };
}
