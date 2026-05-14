//! Process-local staged transport envelope contract.
//!
//! This module validates envelope metadata for adjacent stage handoff and
//! delegates local byte movement to the fixed-header byte harness.

const std = @import("std");
const tensor_frame = @import("tensor_frame.zig");
const boundary_byte_image = @import("boundary_byte_image.zig");
const stage_frame_header = @import("stage_frame_header.zig");
const stage_byte_harness = @import("stage_byte_harness.zig");
const stage_transfer_mode = @import("stage_transfer_mode.zig");
const staged_error = @import("staged_error.zig");

pub const StageTransportContractVersion = u32;
pub const stage_transport_contract_version: StageTransportContractVersion = 1;

pub const StageTransportError =
    tensor_frame.TensorFrameValidationError ||
    boundary_byte_image.BoundaryByteImageError ||
    stage_frame_header.StageFrameHeaderError ||
    stage_byte_harness.StageByteHarnessError ||
    staged_error.StagedErrorError ||
    error{
        InvalidStageTransportContractVersion,
        MissingStageTransportFrameHeader,
        UnexpectedStageTransportFrameHeader,
        MissingStageTransportFailure,
        UnexpectedStageTransportFailure,
        InvalidStageTransportFailureKind,
        StageTransportMetadataMismatch,
        StageTransportDecisionMismatch,
        StageTransportPayloadForbidden,
        StageTransportPayloadRequired,
        StageTransportPayloadByteCountMismatch,
    };

pub const StageTransportEnvelopeKind = enum(u8) {
    activation_payload,
    propagated_failure,
    request_cancelled,
};

pub const StageTransportEnvelope = struct {
    version: StageTransportContractVersion = stage_transport_contract_version,
    kind: StageTransportEnvelopeKind,
    header: ?stage_frame_header.StageFrameHeader = null,
    transfer_mode: ?stage_transfer_mode.StageTransferMode = null,
    payload_byte_count: u64 = 0,
    failure: ?staged_error.StagedFailure = null,
};

pub const StageTransportActivationRequest = struct {
    metadata: *const tensor_frame.TensorFrameMetadata,
    image: *const boundary_byte_image.BoundaryByteImageRef,
    decision: stage_transfer_mode.StageTransferModeDecision,
};

pub fn buildStageTransportActivationEnvelope(
    request: StageTransportActivationRequest,
) StageTransportError!StageTransportEnvelope {
    try request.metadata.validate();
    try boundary_byte_image.validateBoundaryByteImage(request.image, .{});
    if (request.image.metadata != request.metadata) return error.StageTransportMetadataMismatch;
    if (!decisionProfileMatchesMetadata(request.decision, request.metadata)) {
        return error.StageTransportDecisionMismatch;
    }
    if (!activationDecisionModeConsistent(request.decision, request.image)) {
        return error.StageTransportDecisionMismatch;
    }

    const header = try stage_frame_header.stageFrameHeaderFromMetadata(request.metadata, .{
        .source_host_id = request.decision.source_host_id,
        .target_host_id = request.decision.target_host_id,
    });
    const envelope = StageTransportEnvelope{
        .kind = .activation_payload,
        .header = header,
        .transfer_mode = request.decision.mode,
        .payload_byte_count = request.image.byte_count,
    };
    try validateStageTransportEnvelope(&envelope);
    return envelope;
}

pub fn buildStageTransportFailureEnvelope(
    failure: staged_error.StagedFailure,
) StageTransportError!StageTransportEnvelope {
    try staged_error.validateStagedFailure(&failure, .{});
    if (failure.kind == .request_cancelled or failure.kind == .cleanup_failed) {
        return error.InvalidStageTransportFailureKind;
    }
    const envelope = StageTransportEnvelope{
        .kind = .propagated_failure,
        .failure = failure,
    };
    try validateStageTransportEnvelope(&envelope);
    return envelope;
}

pub fn buildStageTransportCancellationEnvelope(
    failure: staged_error.StagedFailure,
) StageTransportError!StageTransportEnvelope {
    try staged_error.validateStagedFailure(&failure, .{});
    if (failure.kind != .request_cancelled) return error.InvalidStageTransportFailureKind;
    const envelope = StageTransportEnvelope{
        .kind = .request_cancelled,
        .failure = failure,
    };
    try validateStageTransportEnvelope(&envelope);
    return envelope;
}

pub fn validateStageTransportEnvelope(
    envelope: *const StageTransportEnvelope,
) StageTransportError!void {
    if (envelope.version != stage_transport_contract_version) {
        return error.InvalidStageTransportContractVersion;
    }

    switch (envelope.kind) {
        .activation_payload => {
            const header = envelope.header orelse return error.MissingStageTransportFrameHeader;
            if (envelope.transfer_mode == null) return error.StageTransportPayloadRequired;
            if (envelope.payload_byte_count == 0) return error.StageTransportPayloadRequired;
            if (envelope.failure != null) return error.UnexpectedStageTransportFailure;
            if (header.payload_byte_count != envelope.payload_byte_count) {
                return error.StageTransportPayloadByteCountMismatch;
            }
            var header_bytes: [stage_frame_header.stage_frame_header_encoded_len]u8 = undefined;
            try stage_frame_header.encodeStageFrameHeader(&header_bytes, header);
        },
        .propagated_failure => {
            if (envelope.header != null) return error.UnexpectedStageTransportFrameHeader;
            if (envelope.transfer_mode != null) return error.StageTransportPayloadForbidden;
            if (envelope.payload_byte_count != 0) return error.StageTransportPayloadForbidden;
            const failure = envelope.failure orelse return error.MissingStageTransportFailure;
            if (failure.kind == .request_cancelled or failure.kind == .cleanup_failed) {
                return error.InvalidStageTransportFailureKind;
            }
            try staged_error.validateStagedFailure(&failure, .{});
        },
        .request_cancelled => {
            if (envelope.header != null) return error.UnexpectedStageTransportFrameHeader;
            if (envelope.transfer_mode != null) return error.StageTransportPayloadForbidden;
            if (envelope.payload_byte_count != 0) return error.StageTransportPayloadForbidden;
            const failure = envelope.failure orelse return error.MissingStageTransportFailure;
            if (failure.kind != .request_cancelled) return error.InvalidStageTransportFailureKind;
            try staged_error.validateStagedFailure(&failure, .{});
        },
    }
}

pub fn writeStageTransportEnvelopeLocal(
    writer: anytype,
    envelope: *const StageTransportEnvelope,
    metadata: *const tensor_frame.TensorFrameMetadata,
    image: *const boundary_byte_image.BoundaryByteImageRef,
) StageTransportError!void {
    try validateStageTransportEnvelope(envelope);
    if (envelope.kind != .activation_payload) return error.StageTransportPayloadForbidden;
    if (envelope.transfer_mode.? != .copy_in_process) return error.StageTransportPayloadForbidden;
    const header = envelope.header.?;
    try stage_frame_header.validateStageFrameHeaderForMetadata(header, metadata);
    if (image.metadata != metadata) return error.StageTransportMetadataMismatch;
    if (image.byte_count != envelope.payload_byte_count) return error.StageTransportPayloadByteCountMismatch;

    const written_header = try stage_byte_harness.writeStageFrameBytes(writer, metadata, image, .{
        .source_host_id = header.source_host_id,
        .target_host_id = header.target_host_id,
    });
    if (!stageFrameHeadersEql(written_header, header)) return error.StageTransportDecisionMismatch;
}

pub fn readStageTransportEnvelopeLocal(
    reader: anytype,
    expected_metadata: *const tensor_frame.TensorFrameMetadata,
    payload_dest: []u8,
) StageTransportError!StageTransportEnvelope {
    const result = try stage_byte_harness.readStageFrameBytes(reader, expected_metadata, payload_dest);
    const envelope = StageTransportEnvelope{
        .kind = .activation_payload,
        .header = result.header,
        .transfer_mode = .copy_in_process,
        .payload_byte_count = result.header.payload_byte_count,
    };
    try validateStageTransportEnvelope(&envelope);
    return envelope;
}

fn decisionProfileMatchesMetadata(
    decision: stage_transfer_mode.StageTransferModeDecision,
    metadata: *const tensor_frame.TensorFrameMetadata,
) bool {
    const profile = decision.boundary_profile;
    return profile.boundary_index == metadata.boundary.boundary_index and
        profile.source_stage_id == metadata.boundary.source_stage_id and
        profile.target_stage_id == metadata.boundary.target_stage_id and
        profile.step_kind == metadata.step_kind and
        profile.dtype == metadata.tensor.dtype and
        profile.layout == metadata.tensor.layout;
}

fn activationDecisionModeConsistent(
    decision: stage_transfer_mode.StageTransferModeDecision,
    image: *const boundary_byte_image.BoundaryByteImageRef,
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
            .same_host_direct, .local_in_process, .mock => decision.mode == .device_download_then_copy,
            .remote_declared => decision.mode == .device_download_then_remote_stream,
        },
        .producer_sync_required, .local_only_opaque => false,
    };
}

fn stageFrameHeadersEql(
    lhs: stage_frame_header.StageFrameHeader,
    rhs: stage_frame_header.StageFrameHeader,
) bool {
    return lhs.flags == rhs.flags and
        std.mem.eql(u8, &lhs.graph_digest, &rhs.graph_digest) and
        lhs.graph_contract_version == rhs.graph_contract_version and
        lhs.stage_plan_contract_version == rhs.stage_plan_contract_version and
        std.mem.eql(u8, &lhs.stage_plan_id_digest, &rhs.stage_plan_id_digest) and
        lhs.frame_id == rhs.frame_id and
        lhs.boundary_index == rhs.boundary_index and
        lhs.source_stage_id == rhs.source_stage_id and
        lhs.target_stage_id == rhs.target_stage_id and
        hostIdOptionsEql(lhs.source_host_id, rhs.source_host_id) and
        hostIdOptionsEql(lhs.target_host_id, rhs.target_host_id) and
        lhs.request_id == rhs.request_id and
        lhs.slot_id == rhs.slot_id and
        lhs.sequence_start == rhs.sequence_start and
        lhs.token_count == rhs.token_count and
        lhs.batch_index == rhs.batch_index and
        lhs.step_kind == rhs.step_kind and
        lhs.dtype == rhs.dtype and
        lhs.layout == rhs.layout and
        lhs.rank == rhs.rank and
        lhs.shape == rhs.shape and
        lhs.payload_byte_count == rhs.payload_byte_count;
}

fn hostIdOptionsEql(lhs: anytype, rhs: @TypeOf(lhs)) bool {
    const lhs_id = lhs orelse return rhs == null;
    const rhs_id = rhs orelse return false;
    return lhs_id.value == rhs_id.value;
}

const test_payload_len: usize = 16;
const test_large_payload_len: usize = 32;

const TestWriter = struct {
    dest: []u8,
    len: usize = 0,
    call_count: usize = 0,
    call_lengths: [4]usize = [_]usize{0} ** 4,

    fn write(self: *TestWriter, bytes: []const u8) !usize {
        const call_idx = self.call_count;
        self.call_count += 1;
        if (call_idx < self.call_lengths.len) self.call_lengths[call_idx] = bytes.len;
        if (bytes.len > self.dest.len - self.len) return error.InjectedWriteFailure;
        @memcpy(self.dest[self.len..][0..bytes.len], bytes);
        self.len += bytes.len;
        return bytes.len;
    }
};

const TestReader = struct {
    source: []const u8,
    offset: usize = 0,
    call_count: usize = 0,

    fn read(self: *TestReader, dest: []u8) !usize {
        self.call_count += 1;
        var count = dest.len;
        if (count > self.source.len - self.offset) count = self.source.len - self.offset;
        @memcpy(dest[0..count], self.source[self.offset..][0..count]);
        self.offset += count;
        return count;
    }
};

fn testDigest(seed: u8) [32]u8 {
    var digest: [32]u8 = undefined;
    for (&digest, 0..) |*byte, byte_idx| {
        byte.* = seed +% @as(u8, @intCast(byte_idx));
    }
    return digest;
}

fn testBoundary(
    boundary_index: usize,
    source_stage_id: usize,
    target_stage_id: usize,
) tensor_frame.TensorFrameBoundaryRef {
    return .{
        .boundary_index = boundary_index,
        .source_stage_id = source_stage_id,
        .target_stage_id = target_stage_id,
        .producer_layer_start = boundary_index * 4,
        .producer_layer_end = (boundary_index + 1) * 4,
        .consumer_layer_start = (boundary_index + 1) * 4,
        .consumer_layer_end = (boundary_index + 2) * 4,
    };
}

fn testBatchEntry(
    batch_index: u32,
    request_id: u64,
    slot_id: u64,
    sequence_start: u64,
    token_count: u64,
) tensor_frame.TensorFrameBatchEntry {
    return .{
        .batch_index = batch_index,
        .request_id = request_id,
        .slot_id = slot_id,
        .sequence_start = sequence_start,
        .token_count = token_count,
    };
}

const test_decode_entries = [_]tensor_frame.TensorFrameBatchEntry{
    testBatchEntry(0, 101, 88, 12, 1),
};

const test_multi_entries = [_]tensor_frame.TensorFrameBatchEntry{
    testBatchEntry(0, 201, 91, 20, 1),
    testBatchEntry(1, 202, 92, 20, 1),
};

fn testMetadata(
    frame_id: u64,
    boundary: tensor_frame.TensorFrameBoundaryRef,
    entries: []const tensor_frame.TensorFrameBatchEntry,
    shape: [4]u64,
    location_hint: ?tensor_frame.TensorFramePayloadLocationHint,
    ownership: tensor_frame.TensorFrameOwnership,
) StageTransportError!tensor_frame.TensorFrameMetadata {
    const tensor = try tensor_frame.TensorFrameTensorDesc.contiguousActivation(.f32, shape);
    return .{
        .frame_id = try tensor_frame.TensorFrameInstanceId.init(frame_id),
        .plan = .{
            .graph_digest = testDigest(0x10),
            .graph_contract_version = 7,
            .stage_plan_contract_version = 9,
            .stage_plan_id = .{ .digest = testDigest(0x40) },
        },
        .boundary = boundary,
        .selected_contract = .{
            .boundary = boundary,
            .dtype = .f32,
            .layout = .row_major,
            .source = .explicit,
        },
        .role = .activation,
        .step_kind = .decode,
        .shape_context = .{
            .expected_hidden_size = shape[2],
            .expected_step_kind = .decode,
        },
        .tensor = tensor,
        .batch = .{ .entries = entries },
        .payload = .{
            .byte_count = tensor.payload_byte_count,
            .location_hint = location_hint,
            .ownership = ownership,
            .lifetime = .step_scoped,
        },
    };
}

fn testDecodeMetadata() StageTransportError!tensor_frame.TensorFrameMetadata {
    return testMetadata(55, testBoundary(2, 10, 11), &test_decode_entries, .{ 1, 1, 4, 0 }, .cpu, .borrowed_until_next_stage_call);
}

fn testMetadataForBoundary(
    frame_id: u64,
    boundary_index: usize,
    source_stage_id: usize,
    target_stage_id: usize,
) StageTransportError!tensor_frame.TensorFrameMetadata {
    return testMetadata(frame_id, testBoundary(boundary_index, source_stage_id, target_stage_id), &test_decode_entries, .{ 1, 1, 4, 0 }, .cpu, .borrowed_until_next_stage_call);
}

fn testMultiBatchMetadata() StageTransportError!tensor_frame.TensorFrameMetadata {
    return testMetadata(57, testBoundary(2, 10, 11), &test_multi_entries, .{ 2, 1, 4, 0 }, .cpu, .borrowed_until_next_stage_call);
}

fn testImage(
    metadata: *const tensor_frame.TensorFrameMetadata,
    readiness: boundary_byte_image.BoundaryByteImageReadiness,
    host_bytes: ?[]const u8,
) boundary_byte_image.BoundaryByteImageRef {
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

fn testDecision(
    metadata: *const tensor_frame.TensorFrameMetadata,
    handoff_mode: anytype,
    mode: stage_transfer_mode.StageTransferMode,
) stage_transfer_mode.StageTransferModeDecision {
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
        .source_host_id = .{ .value = 31 },
        .target_host_id = .{ .value = 32 },
    };
}

fn transportFailure() staged_error.StagedFailure {
    return .{
        .kind = .transfer_failed,
        .phase = .frame_handoff,
        .scope = .transport,
        .context = .{ .boundary_index = 2 },
        .source = .{ .domain = .transport },
    };
}

fn cancellationFailure() staged_error.StagedFailure {
    return .{
        .kind = .request_cancelled,
        .phase = .validation_before_mutation,
        .scope = .request,
        .context = .{ .request_id = 101 },
        .source = .{ .domain = .transport },
    };
}

fn cleanupFailure() staged_error.StagedFailure {
    return .{
        .kind = .cleanup_failed,
        .phase = .cleanup,
        .scope = .cleanup,
        .source = .{ .domain = .cleanup },
    };
}

fn buildCopyEnvelope(
    metadata: *const tensor_frame.TensorFrameMetadata,
    image: *const boundary_byte_image.BoundaryByteImageRef,
) StageTransportError!StageTransportEnvelope {
    return buildStageTransportActivationEnvelope(.{
        .metadata = metadata,
        .image = image,
        .decision = testDecision(metadata, .local_in_process, .copy_in_process),
    });
}

fn expectFilled(bytes: []const u8, value: u8) !void {
    for (bytes) |byte| {
        try std.testing.expectEqual(value, byte);
    }
}

test "inference bridge stage_transport buildStageTransportActivationEnvelope validateStageTransportEnvelope accepts adjacent activation payload" {
    var metadata = try testDecodeMetadata();
    const payload = [_]u8{ 0x10, 0x11, 0x12, 0x13, 0x20, 0x21, 0x22, 0x23, 0x30, 0x31, 0x32, 0x33, 0x40, 0x41, 0x42, 0x43 };
    const image = testImage(&metadata, .host_readable_now, &payload);
    const decision = testDecision(&metadata, .local_in_process, .copy_in_process);

    const envelope = try buildStageTransportActivationEnvelope(.{
        .metadata = &metadata,
        .image = &image,
        .decision = decision,
    });
    try validateStageTransportEnvelope(&envelope);

    try std.testing.expectEqual(StageTransportEnvelopeKind.activation_payload, envelope.kind);
    try std.testing.expectEqual(stage_transfer_mode.StageTransferMode.copy_in_process, envelope.transfer_mode.?);
    try std.testing.expectEqual(metadata.payload.byte_count, envelope.payload_byte_count);
    try std.testing.expectEqual(@as(u64, 31), envelope.header.?.source_host_id.?.value);
    try std.testing.expectEqual(@as(u64, 32), envelope.header.?.target_host_id.?.value);

    var multi_metadata = try testMultiBatchMetadata();
    const multi_payload = [_]u8{0xaa} ** test_large_payload_len;
    const multi_image = testImage(&multi_metadata, .host_readable_now, &multi_payload);
    try std.testing.expectError(error.UnsupportedStageFrameHeaderBatch, buildStageTransportActivationEnvelope(.{
        .metadata = &multi_metadata,
        .image = &multi_image,
        .decision = testDecision(&multi_metadata, .local_in_process, .copy_in_process),
    }));

    var zero_metadata = metadata;
    zero_metadata.batch = .{ .entries = &.{} };
    const zero_image = testImage(&zero_metadata, .host_readable_now, &payload);
    try std.testing.expectError(error.InvalidBatch, buildStageTransportActivationEnvelope(.{
        .metadata = &zero_metadata,
        .image = &zero_image,
        .decision = testDecision(&zero_metadata, .local_in_process, .copy_in_process),
    }));
}

test "inference bridge stage_transport buildStageTransportActivationEnvelope rejects impossible activation transfer decisions" {
    var metadata = try testDecodeMetadata();
    const payload = [_]u8{0xbb} ** test_payload_len;
    const image = testImage(&metadata, .host_readable_now, &payload);

    var mismatched_decision = testDecision(&metadata, .local_in_process, .copy_in_process);
    mismatched_decision.boundary_profile.dtype = .f16;
    try std.testing.expectError(error.StageTransportDecisionMismatch, buildStageTransportActivationEnvelope(.{
        .metadata = &metadata,
        .image = &image,
        .decision = mismatched_decision,
    }));

    const remote_wrong_mode = testDecision(&metadata, .remote_declared, .copy_in_process);
    try std.testing.expectError(error.StageTransportDecisionMismatch, buildStageTransportActivationEnvelope(.{
        .metadata = &metadata,
        .image = &image,
        .decision = remote_wrong_mode,
    }));

    var producer_metadata = try testMetadata(58, testBoundary(2, 10, 11), &test_decode_entries, .{ 1, 1, 4, 0 }, .cpu, .borrowed_until_next_stage_call);
    const producer_image = testImage(&producer_metadata, .producer_sync_required, null);
    try std.testing.expectError(error.StageTransportDecisionMismatch, buildStageTransportActivationEnvelope(.{
        .metadata = &producer_metadata,
        .image = &producer_image,
        .decision = testDecision(&producer_metadata, .local_in_process, .copy_in_process),
    }));

    var owned_metadata = try testMetadata(59, testBoundary(2, 10, 11), &test_decode_entries, .{ 1, 1, 4, 0 }, .cpu, .owned_by_sender_until_completion);
    const owned_image = testImage(&owned_metadata, .host_readable_now, &payload);
    try std.testing.expectError(error.StageTransportDecisionMismatch, buildStageTransportActivationEnvelope(.{
        .metadata = &owned_metadata,
        .image = &owned_image,
        .decision = testDecision(&owned_metadata, .same_host_direct, .borrow_in_process),
    }));
}

test "inference bridge stage_transport validateStageTransportEnvelope rejects contract header failure transfer mode and byte count mismatches" {
    var metadata = try testDecodeMetadata();
    const payload = [_]u8{0xcc} ** test_payload_len;
    const image = testImage(&metadata, .host_readable_now, &payload);
    const envelope = try buildCopyEnvelope(&metadata, &image);
    const failure = transportFailure();
    const cancellation = cancellationFailure();

    var invalid = envelope;
    invalid.version = 0;
    try std.testing.expectError(error.InvalidStageTransportContractVersion, validateStageTransportEnvelope(&invalid));

    invalid = envelope;
    invalid.header = null;
    try std.testing.expectError(error.MissingStageTransportFrameHeader, validateStageTransportEnvelope(&invalid));

    invalid = envelope;
    invalid.transfer_mode = null;
    try std.testing.expectError(error.StageTransportPayloadRequired, validateStageTransportEnvelope(&invalid));

    invalid = envelope;
    invalid.payload_byte_count = 0;
    try std.testing.expectError(error.StageTransportPayloadRequired, validateStageTransportEnvelope(&invalid));

    invalid = envelope;
    invalid.failure = failure;
    try std.testing.expectError(error.UnexpectedStageTransportFailure, validateStageTransportEnvelope(&invalid));

    invalid = envelope;
    invalid.payload_byte_count += 1;
    try std.testing.expectError(error.StageTransportPayloadByteCountMismatch, validateStageTransportEnvelope(&invalid));

    invalid = envelope;
    var invalid_header = invalid.header.?;
    invalid_header.flags = 99;
    invalid.header = invalid_header;
    try std.testing.expectError(error.InvalidStageFrameHeaderFlags, validateStageTransportEnvelope(&invalid));

    var propagated = try buildStageTransportFailureEnvelope(failure);
    propagated.header = envelope.header;
    try std.testing.expectError(error.UnexpectedStageTransportFrameHeader, validateStageTransportEnvelope(&propagated));

    propagated = try buildStageTransportFailureEnvelope(failure);
    propagated.transfer_mode = .copy_in_process;
    try std.testing.expectError(error.StageTransportPayloadForbidden, validateStageTransportEnvelope(&propagated));

    propagated = try buildStageTransportFailureEnvelope(failure);
    propagated.payload_byte_count = 1;
    try std.testing.expectError(error.StageTransportPayloadForbidden, validateStageTransportEnvelope(&propagated));

    propagated = try buildStageTransportFailureEnvelope(failure);
    propagated.failure = null;
    try std.testing.expectError(error.MissingStageTransportFailure, validateStageTransportEnvelope(&propagated));

    propagated = .{ .kind = .propagated_failure, .failure = cancellation };
    try std.testing.expectError(error.InvalidStageTransportFailureKind, validateStageTransportEnvelope(&propagated));

    var cancelled = try buildStageTransportCancellationEnvelope(cancellation);
    cancelled.failure = failure;
    try std.testing.expectError(error.InvalidStageTransportFailureKind, validateStageTransportEnvelope(&cancelled));
}

test "inference bridge stage_transport writeStageTransportEnvelopeLocal readStageTransportEnvelopeLocal moves fixed header plus raw payload bytes" {
    var metadata = try testDecodeMetadata();
    const payload = [_]u8{ 0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf };
    const image = testImage(&metadata, .host_readable_now, &payload);
    const envelope = try buildCopyEnvelope(&metadata, &image);
    var written: [stage_frame_header.stage_frame_header_encoded_len + test_payload_len]u8 = [_]u8{0} ** (stage_frame_header.stage_frame_header_encoded_len + test_payload_len);
    var writer = TestWriter{ .dest = written[0..] };

    try writeStageTransportEnvelopeLocal(&writer, &envelope, &metadata, &image);

    try std.testing.expectEqual(@as(usize, 2), writer.call_count);
    try std.testing.expectEqual(stage_frame_header.stage_frame_header_encoded_len, writer.call_lengths[0]);
    try std.testing.expectEqual(payload.len, writer.call_lengths[1]);
    try std.testing.expectEqual(stage_frame_header.stage_frame_header_encoded_len + payload.len, writer.len);
    try std.testing.expectEqualSlices(u8, &payload, written[stage_frame_header.stage_frame_header_encoded_len..writer.len]);

    var reader = TestReader{ .source = written[0..writer.len] };
    var received: [test_payload_len]u8 = [_]u8{0} ** test_payload_len;
    const read_envelope = try readStageTransportEnvelopeLocal(&reader, &metadata, received[0..]);

    try std.testing.expectEqual(@as(usize, 2), reader.call_count);
    try std.testing.expectEqual(StageTransportEnvelopeKind.activation_payload, read_envelope.kind);
    try std.testing.expectEqual(stage_transfer_mode.StageTransferMode.copy_in_process, read_envelope.transfer_mode.?);
    try stage_frame_header.validateStageFrameHeaderForMetadata(read_envelope.header.?, &metadata);
    try std.testing.expectEqualSlices(u8, &payload, &received);

    var multi_metadata = try testMultiBatchMetadata();
    const multi_payload = [_]u8{0xee} ** test_large_payload_len;
    const multi_image = testImage(&multi_metadata, .host_readable_now, &multi_payload);
    var multi_writer = TestWriter{ .dest = written[0..] };
    try std.testing.expectError(error.UnsupportedStageFrameHeaderBatch, writeStageTransportEnvelopeLocal(&multi_writer, &envelope, &multi_metadata, &multi_image));
    try std.testing.expectEqual(@as(usize, 0), multi_writer.call_count);

    var multi_reader = TestReader{ .source = written[0..writer.len] };
    var multi_dest: [test_payload_len]u8 = [_]u8{0x7a} ** test_payload_len;
    try std.testing.expectError(error.UnsupportedStageFrameHeaderBatch, readStageTransportEnvelopeLocal(&multi_reader, &multi_metadata, multi_dest[0..]));
    try std.testing.expectEqual(@as(usize, 1), multi_reader.call_count);
    try expectFilled(&multi_dest, 0x7a);

    var zero_metadata = metadata;
    zero_metadata.batch = .{ .entries = &.{} };
    var zero_reader = TestReader{ .source = written[0..writer.len] };
    var zero_dest: [test_payload_len]u8 = [_]u8{0x6b} ** test_payload_len;
    try std.testing.expectError(error.InvalidBatch, readStageTransportEnvelopeLocal(&zero_reader, &zero_metadata, zero_dest[0..]));
    try expectFilled(&zero_dest, 0x6b);
}

test "inference bridge stage_transport writeStageTransportEnvelopeLocal rejects non copy transfer modes and mismatched metadata before payload write" {
    var metadata = try testDecodeMetadata();
    const payload = [_]u8{0xef} ** test_payload_len;
    const image = testImage(&metadata, .host_readable_now, &payload);
    const copy_envelope = try buildCopyEnvelope(&metadata, &image);
    const borrow_envelope = try buildStageTransportActivationEnvelope(.{
        .metadata = &metadata,
        .image = &image,
        .decision = testDecision(&metadata, .same_host_direct, .borrow_in_process),
    });
    var dest: [stage_frame_header.stage_frame_header_encoded_len + test_payload_len]u8 = [_]u8{0x5a} ** (stage_frame_header.stage_frame_header_encoded_len + test_payload_len);

    var writer = TestWriter{ .dest = dest[0..] };
    try std.testing.expectError(error.StageTransportPayloadForbidden, writeStageTransportEnvelopeLocal(&writer, &borrow_envelope, &metadata, &image));
    try std.testing.expectEqual(@as(usize, 0), writer.call_count);
    try expectFilled(&dest, 0x5a);

    var other_metadata = try testMetadata(66, testBoundary(2, 10, 11), &test_decode_entries, .{ 1, 1, 4, 0 }, .cpu, .borrowed_until_next_stage_call);
    const other_image = testImage(&other_metadata, .host_readable_now, &payload);
    writer = TestWriter{ .dest = dest[0..] };
    try std.testing.expectError(error.StageTransportMetadataMismatch, writeStageTransportEnvelopeLocal(&writer, &copy_envelope, &metadata, &other_image));
    try std.testing.expectEqual(@as(usize, 0), writer.call_count);
    try expectFilled(&dest, 0x5a);

    var count_mismatch_image = image;
    count_mismatch_image.byte_count += 1;
    writer = TestWriter{ .dest = dest[0..] };
    try std.testing.expectError(error.StageTransportPayloadByteCountMismatch, writeStageTransportEnvelopeLocal(&writer, &copy_envelope, &metadata, &count_mismatch_image));
    try std.testing.expectEqual(@as(usize, 0), writer.call_count);
    try expectFilled(&dest, 0x5a);

    var zero_metadata = metadata;
    zero_metadata.batch = .{ .entries = &.{} };
    writer = TestWriter{ .dest = dest[0..] };
    try std.testing.expectError(error.InvalidBatch, writeStageTransportEnvelopeLocal(&writer, &copy_envelope, &zero_metadata, &image));
    try std.testing.expectEqual(@as(usize, 0), writer.call_count);
}

test "inference bridge stage_transport buildStageTransportFailureEnvelope validateStageTransportEnvelope accepts propagated staged failure without payload bytes" {
    const failure = transportFailure();
    const envelope = try buildStageTransportFailureEnvelope(failure);

    try validateStageTransportEnvelope(&envelope);
    try std.testing.expectEqual(StageTransportEnvelopeKind.propagated_failure, envelope.kind);
    try std.testing.expect(envelope.header == null);
    try std.testing.expect(envelope.transfer_mode == null);
    try std.testing.expectEqual(@as(u64, 0), envelope.payload_byte_count);
    try std.testing.expectEqual(staged_error.StagedFailureKind.transfer_failed, envelope.failure.?.kind);

    try std.testing.expectError(error.InvalidStageTransportFailureKind, buildStageTransportFailureEnvelope(cancellationFailure()));
    try std.testing.expectError(error.InvalidStageTransportFailureKind, buildStageTransportFailureEnvelope(cleanupFailure()));
}

test "inference bridge stage_transport buildStageTransportCancellationEnvelope validateStageTransportEnvelope accepts request_cancelled without payload bytes" {
    const failure = cancellationFailure();
    const envelope = try buildStageTransportCancellationEnvelope(failure);

    try validateStageTransportEnvelope(&envelope);
    try std.testing.expectEqual(StageTransportEnvelopeKind.request_cancelled, envelope.kind);
    try std.testing.expect(envelope.header == null);
    try std.testing.expect(envelope.transfer_mode == null);
    try std.testing.expectEqual(@as(u64, 0), envelope.payload_byte_count);
    try std.testing.expectEqual(staged_error.StagedFailureKind.request_cancelled, envelope.failure.?.kind);

    try std.testing.expectError(error.InvalidStageTransportFailureKind, buildStageTransportCancellationEnvelope(transportFailure()));
}

test "inference bridge stage_transport validates repeated adjacent activation envelopes for a three stage chain" {
    var first = try testMetadataForBoundary(70, 0, 0, 1);
    var second = try testMetadataForBoundary(71, 1, 1, 2);
    const first_payload = [_]u8{0x10} ** test_payload_len;
    const second_payload = [_]u8{0x20} ** test_payload_len;
    const first_image = testImage(&first, .host_readable_now, &first_payload);
    const second_image = testImage(&second, .host_readable_now, &second_payload);

    const first_envelope = try buildStageTransportActivationEnvelope(.{
        .metadata = &first,
        .image = &first_image,
        .decision = testDecision(&first, .local_in_process, .copy_in_process),
    });
    const second_envelope = try buildStageTransportActivationEnvelope(.{
        .metadata = &second,
        .image = &second_image,
        .decision = testDecision(&second, .local_in_process, .copy_in_process),
    });

    try validateStageTransportEnvelope(&first_envelope);
    try validateStageTransportEnvelope(&second_envelope);
    try std.testing.expectEqual(@as(u64, 0), first_envelope.header.?.source_stage_id);
    try std.testing.expectEqual(@as(u64, 1), first_envelope.header.?.target_stage_id);
    try std.testing.expectEqual(first_envelope.header.?.target_stage_id, second_envelope.header.?.source_stage_id);
    try std.testing.expectEqual(@as(u64, 2), second_envelope.header.?.target_stage_id);
}
