//! Tensor-frame metadata for validated staged decoder activation handoffs.
//!
//! A tensor frame is metadata only. It identifies a Story 03 `StagePlan`
//! boundary, validates the logical activation tensor contract, and describes
//! payload ownership without moving or retaining payload bytes.

const std = @import("std");
const compute = @import("compute_pkg");
const models = @import("models_pkg");
const pipeline = @import("pipeline.zig");

const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;
const stage_plan = models.stage_plan;

pub const tensor_frame_contract_version: u32 = 1;
pub const activation_payload_layout_contract_version: u32 = 1;
pub const max_tensor_rank: u8 = 4;

pub const TensorFrameContractVersion = u32;
pub const TensorFrameDType = pipeline.BoundaryDType;
pub const TensorFrameLayout = pipeline.BoundaryLayout;

pub const TensorFrameValidationError = stage_plan.StagePlanError || error{
    InvalidTensorFrameContractVersion,
    InvalidFrameId,
    GraphIdentityMismatch,
    StagePlanIdentityMismatch,
    MissingSelectedBoundaryTensorContract,
    BoundaryTensorContractMismatch,
    BoundaryIndexOutOfRange,
    AbsentStageBoundary,
    InvalidSourceStage,
    InvalidTargetStage,
    InvalidProducerLayerRange,
    InvalidConsumerLayerRange,
    InvalidActivationRole,
    InvalidStepKind,
    InvalidDType,
    UnsupportedActivationDType,
    InvalidTensorRank,
    InvalidTensorShape,
    InvalidHiddenSize,
    InvalidTensorStride,
    UnsupportedTensorLayout,
    NonContiguousPayload,
    ByteCountOverflow,
    InvalidLogicalByteCount,
    InvalidPayloadByteCount,
    PayloadBufferLengthMismatch,
    InvalidBatch,
    DuplicateBatchIndex,
    DuplicateRequestId,
    DuplicateSlotId,
    InvalidRequestId,
    InvalidSlotId,
    InvalidSequenceRange,
    RaggedBatchUnsupported,
    InvalidOwnership,
    InvalidLifetime,
    ObserverFailure,
};

pub const TensorFrameInstanceId = struct {
    value: u64,

    pub fn init(value: u64) TensorFrameValidationError!TensorFrameInstanceId {
        if (value == 0) return error.InvalidFrameId;
        return .{ .value = value };
    }
};

pub const TensorFramePlanIdentity = struct {
    graph_digest: [32]u8,
    graph_contract_version: u32,
    stage_plan_contract_version: u32,
    stage_plan_id: stage_plan.StagePlanId,
};

pub const TensorFrameBoundaryRef = struct {
    boundary_index: usize,
    source_stage_id: usize,
    target_stage_id: usize,
    producer_layer_start: usize,
    producer_layer_end: usize,
    consumer_layer_start: usize,
    consumer_layer_end: usize,
};

pub const TensorFramePlanRef = struct {
    allocator: Allocator,
    identity: TensorFramePlanIdentity,
    boundaries: []const TensorFrameBoundaryRef,

    /// Caller owns the returned ref and must call `deinit`.
    pub fn fromStagePlan(allocator: Allocator, plan: *const stage_plan.StagePlan) TensorFrameValidationError!TensorFramePlanRef {
        try stage_plan.validateStagePlan(plan, .{});

        const boundaries = try allocator.alloc(TensorFrameBoundaryRef, plan.boundaries.len);
        errdefer allocator.free(boundaries);
        for (plan.boundaries, 0..) |stage_boundary, index| {
            boundaries[index] = boundaryRefFromStageBoundary(index, stage_boundary);
        }

        return .{
            .allocator = allocator,
            .identity = .{
                .graph_digest = plan.graph_identity.digest,
                .graph_contract_version = plan.graph_identity.graph_contract_version,
                .stage_plan_contract_version = plan.stage_contract_version,
                .stage_plan_id = plan.plan_id,
            },
            .boundaries = boundaries,
        };
    }

    pub fn deinit(self: *TensorFramePlanRef) void {
        self.allocator.free(self.boundaries);
        self.* = undefined;
    }

    pub fn boundary(self: *const TensorFramePlanRef, boundary_index: usize) TensorFrameValidationError!TensorFrameBoundaryRef {
        if (self.boundaries.len == 0) return error.AbsentStageBoundary;
        if (boundary_index >= self.boundaries.len) return error.BoundaryIndexOutOfRange;
        return self.boundaries[boundary_index];
    }
};

pub const BoundaryTensorContractSource = enum(u8) {
    explicit,
    negotiated,
};

pub const TensorFrameBoundaryTensorContract = struct {
    boundary: TensorFrameBoundaryRef,
    dtype: TensorFrameDType,
    layout: TensorFrameLayout = .row_major,
    source: BoundaryTensorContractSource = .explicit,
};

pub const TensorFrameShapeContext = struct {
    expected_hidden_size: u64,
    expected_step_kind: ?TensorFrameStepKind = null,
};

pub const TensorFrameRole = enum(u8) {
    activation,
    logits_diagnostic,
};

pub const TensorFrameStepKind = enum(u8) {
    prefill,
    decode,
};

pub const TensorFrameBatchEntry = struct {
    batch_index: u32,
    request_id: u64,
    slot_id: u64,
    sequence_start: u64,
    token_count: u64,
    state_epoch: ?u64 = null,
};

pub const TensorFrameBatch = struct {
    entries: []const TensorFrameBatchEntry,
};

pub const TensorFrameTensorDesc = struct {
    dtype: TensorFrameDType,
    rank: u8,
    shape: [max_tensor_rank]u64,
    stride_elems: [max_tensor_rank]u64,
    layout: TensorFrameLayout = .row_major,
    logical_element_count: u64,
    logical_byte_count: u64,
    payload_byte_count: u64,

    pub fn contiguousActivation(dtype: TensorFrameDType, shape: [max_tensor_rank]u64) TensorFrameValidationError!TensorFrameTensorDesc {
        var stride_elems = [_]u64{0} ** max_tensor_rank;
        if (shape[0] == 0 or shape[1] == 0 or shape[2] == 0) return error.InvalidTensorShape;
        if (shape[3] != 0) return error.InvalidTensorShape;

        stride_elems[2] = 1;
        stride_elems[1] = shape[2];
        stride_elems[0] = std.math.mul(u64, shape[1], shape[2]) catch return error.ByteCountOverflow;

        const element_count = checkedActivationElementCount(shape) catch |err| return err;
        const byte_count = checkedByteCount(element_count, dtype) catch |err| return err;
        return .{
            .dtype = dtype,
            .rank = 3,
            .shape = shape,
            .stride_elems = stride_elems,
            .layout = .row_major,
            .logical_element_count = element_count,
            .logical_byte_count = byte_count,
            .payload_byte_count = byte_count,
        };
    }

    pub fn validateActivation(self: *const TensorFrameTensorDesc, expected_hidden_size: u64, step_kind: TensorFrameStepKind) TensorFrameValidationError!void {
        if (self.rank != 3) return error.InvalidTensorRank;
        if (self.layout != .row_major) return error.UnsupportedTensorLayout;
        if (expected_hidden_size == 0) return error.InvalidHiddenSize;
        if (self.shape[0] == 0 or self.shape[1] == 0 or self.shape[2] == 0) return error.InvalidTensorShape;
        if (self.shape[2] != expected_hidden_size) return error.InvalidHiddenSize;
        if (step_kind == .decode and self.shape[1] != 1) return error.InvalidTensorShape;
        if (self.shape[3] != 0 or self.stride_elems[3] != 0) return error.InvalidTensorShape;

        if (self.stride_elems[2] != 1) return error.InvalidTensorStride;
        if (self.stride_elems[1] != self.shape[2]) return error.InvalidTensorStride;
        const expected_batch_stride = std.math.mul(u64, self.shape[1], self.shape[2]) catch return error.ByteCountOverflow;
        if (self.stride_elems[0] != expected_batch_stride) return error.InvalidTensorStride;

        const expected_elements = checkedActivationElementCount(self.shape) catch |err| return err;
        if (self.logical_element_count == 0 or self.logical_element_count != expected_elements) {
            return error.InvalidLogicalByteCount;
        }
        const expected_bytes = checkedByteCount(expected_elements, self.dtype) catch |err| return err;
        if (self.logical_byte_count == 0 or self.logical_byte_count != expected_bytes) {
            return error.InvalidLogicalByteCount;
        }
        if (self.payload_byte_count == 0 or self.payload_byte_count != expected_bytes) {
            return error.InvalidPayloadByteCount;
        }
    }
};

pub const TensorFramePayloadLocationHint = union(enum(u8)) {
    cpu,
    cuda: u16,
    metal: u16,
    opaque_local: u32,
};

pub const TensorFrameOwnership = enum(u8) {
    borrowed_until_next_stage_call,
    owned_by_sender_until_completion,
    owned_by_receiver_after_handoff,
    external_handle,
};

pub const TensorFrameLifetime = enum(u8) {
    step_scoped,
    request_scoped,
    slot_persistent_metadata_ref,
};

pub const TensorFramePayload = struct {
    byte_count: u64,
    layout: TensorFrameLayout = .row_major,
    location_hint: ?TensorFramePayloadLocationHint = null,
    ownership: TensorFrameOwnership = .borrowed_until_next_stage_call,
    lifetime: TensorFrameLifetime = .step_scoped,
    checksum: ?u64 = null,
};

/// `batch.entries` is caller-owned and must outlive this metadata value.
pub const TensorFrameMetadata = struct {
    version: TensorFrameContractVersion = tensor_frame_contract_version,
    payload_layout_contract_version: TensorFrameContractVersion = activation_payload_layout_contract_version,
    frame_id: TensorFrameInstanceId,
    plan: TensorFramePlanIdentity,
    boundary: TensorFrameBoundaryRef,
    selected_contract: TensorFrameBoundaryTensorContract,
    role: TensorFrameRole,
    step_kind: TensorFrameStepKind,
    shape_context: TensorFrameShapeContext,
    tensor: TensorFrameTensorDesc,
    batch: TensorFrameBatch,
    payload: TensorFramePayload,

    pub fn validate(self: *const TensorFrameMetadata) TensorFrameValidationError!void {
        if (self.version != tensor_frame_contract_version or
            self.payload_layout_contract_version != activation_payload_layout_contract_version)
        {
            return error.InvalidTensorFrameContractVersion;
        }
        if (self.frame_id.value == 0) return error.InvalidFrameId;
        if (self.role != .activation) return error.InvalidActivationRole;
        if (self.shape_context.expected_step_kind) |expected| {
            if (expected != self.step_kind) return error.InvalidStepKind;
        }
        try validateBoundaryRefMatches(self.boundary, self.selected_contract.boundary);
        if (self.selected_contract.layout != .row_major) return error.UnsupportedTensorLayout;
        if (self.tensor.dtype != self.selected_contract.dtype or self.tensor.layout != self.selected_contract.layout) {
            return error.BoundaryTensorContractMismatch;
        }
        try self.tensor.validateActivation(self.shape_context.expected_hidden_size, self.step_kind);
        try validateBatch(self.batch, self.tensor.shape[0], self.tensor.shape[1], self.step_kind);
        try validatePayload(self.payload, self.tensor.payload_byte_count);
    }
};

pub const ActivationFrameArgs = struct {
    frame_id: TensorFrameInstanceId,
    plan_ref: *const TensorFramePlanRef,
    boundary_index: usize,
    selected_contract: ?*const TensorFrameBoundaryTensorContract,
    shape_context: TensorFrameShapeContext,
    tensor: TensorFrameTensorDesc,
    batch: TensorFrameBatch,
    payload: TensorFramePayload,
};

pub const TensorFrameObserverMode = enum(u8) {
    best_effort,
    strict,
};

pub const TensorFrameObserver = struct {
    ctx: ?*anyopaque = null,
    emit_fn: ?*const fn (?*anyopaque, *const TensorFrameMetadata) anyerror!void = null,

    pub fn noop() TensorFrameObserver {
        return .{};
    }
};

pub fn dtypeByteSize(dtype: TensorFrameDType) u64 {
    return switch (dtype) {
        .bf16, .f16 => 2,
        .f32 => 4,
    };
}

pub fn boundaryRefFromPlanRef(plan_ref: *const TensorFramePlanRef, boundary_index: usize) TensorFrameValidationError!TensorFrameBoundaryRef {
    return plan_ref.boundary(boundary_index);
}

pub fn selectedBoundaryTensorContract(
    plan_ref: *const TensorFramePlanRef,
    boundary_index: usize,
    dtype: TensorFrameDType,
    layout: TensorFrameLayout,
    source: BoundaryTensorContractSource,
) TensorFrameValidationError!TensorFrameBoundaryTensorContract {
    if (layout != .row_major) return error.UnsupportedTensorLayout;
    return .{
        .boundary = try boundaryRefFromPlanRef(plan_ref, boundary_index),
        .dtype = dtype,
        .layout = layout,
        .source = source,
    };
}

pub fn activationPrefillFrame(args: ActivationFrameArgs) TensorFrameValidationError!TensorFrameMetadata {
    return activationFrame(.prefill, args);
}

pub fn activationDecodeFrame(args: ActivationFrameArgs) TensorFrameValidationError!TensorFrameMetadata {
    return activationFrame(.decode, args);
}

pub fn validateTensorFramePlanIdentity(
    metadata: *const TensorFrameMetadata,
    plan_ref: *const TensorFramePlanRef,
) TensorFrameValidationError!void {
    if (!std.mem.eql(u8, &metadata.plan.graph_digest, &plan_ref.identity.graph_digest) or
        metadata.plan.graph_contract_version != plan_ref.identity.graph_contract_version)
    {
        return error.GraphIdentityMismatch;
    }
    if (metadata.plan.stage_plan_contract_version != plan_ref.identity.stage_plan_contract_version or
        !std.mem.eql(u8, &metadata.plan.stage_plan_id.digest, &plan_ref.identity.stage_plan_id.digest))
    {
        return error.StagePlanIdentityMismatch;
    }
}

pub fn validateTensorFrameForPlanBoundary(
    metadata: *const TensorFrameMetadata,
    plan_ref: *const TensorFramePlanRef,
    boundary_index: usize,
) TensorFrameValidationError!void {
    try metadata.validate();
    try validateTensorFramePlanIdentity(metadata, plan_ref);
    const expected_boundary = try plan_ref.boundary(boundary_index);
    try validateBoundaryRefMatches(expected_boundary, metadata.boundary);
    try validateBoundaryRefMatches(expected_boundary, metadata.selected_contract.boundary);
}

pub fn validatePayloadBufferLength(
    metadata: *const TensorFrameMetadata,
    payload_buffer_len: usize,
) TensorFrameValidationError!void {
    try metadata.validate();
    const expected = std.math.cast(usize, metadata.payload.byte_count) orelse return error.InvalidPayloadByteCount;
    if (payload_buffer_len != expected) return error.PayloadBufferLengthMismatch;
}

pub fn emitTensorFrame(
    observer: TensorFrameObserver,
    mode: TensorFrameObserverMode,
    metadata: *const TensorFrameMetadata,
) TensorFrameValidationError!void {
    try metadata.validate();
    const emit_fn = observer.emit_fn orelse return;
    emit_fn(observer.ctx, metadata) catch {
        if (mode == .strict) return error.ObserverFailure;
    };
}

pub fn tensorFrameLogicalEql(lhs: *const TensorFrameMetadata, rhs: *const TensorFrameMetadata) bool {
    return std.mem.eql(u8, &tensorFrameLogicalHash(lhs), &tensorFrameLogicalHash(rhs));
}

pub fn tensorFrameLogicalHash(metadata: *const TensorFrameMetadata) [32]u8 {
    var encoder = HashEncoder.init();
    writeLogicalFrame(&encoder, metadata);
    return encoder.finish();
}

pub fn toComputeDType(dtype: TensorFrameDType) compute.DType {
    return switch (dtype) {
        .bf16 => .bf16,
        .f16 => .f16,
        .f32 => .f32,
    };
}

pub fn fromComputeDType(dtype: compute.DType) TensorFrameValidationError!TensorFrameDType {
    return switch (dtype) {
        .bf16 => .bf16,
        .f16 => .f16,
        .f32 => .f32,
        else => error.UnsupportedActivationDType,
    };
}

fn activationFrame(step_kind: TensorFrameStepKind, args: ActivationFrameArgs) TensorFrameValidationError!TensorFrameMetadata {
    const contract = args.selected_contract orelse return error.MissingSelectedBoundaryTensorContract;
    const boundary = try args.plan_ref.boundary(args.boundary_index);
    try validateBoundaryRefMatches(boundary, contract.boundary);

    const metadata = TensorFrameMetadata{
        .frame_id = args.frame_id,
        .plan = args.plan_ref.identity,
        .boundary = boundary,
        .selected_contract = contract.*,
        .role = .activation,
        .step_kind = step_kind,
        .shape_context = args.shape_context,
        .tensor = args.tensor,
        .batch = args.batch,
        .payload = args.payload,
    };
    try metadata.validate();
    try validateTensorFramePlanIdentity(&metadata, args.plan_ref);
    return metadata;
}

fn boundaryRefFromStageBoundary(index: usize, boundary: stage_plan.StageBoundary) TensorFrameBoundaryRef {
    return .{
        .boundary_index = index,
        .source_stage_id = boundary.source_stage_id,
        .target_stage_id = boundary.target_stage_id,
        .producer_layer_start = boundary.producer_layer_start,
        .producer_layer_end = boundary.producer_layer_end,
        .consumer_layer_start = boundary.consumer_layer_start,
        .consumer_layer_end = boundary.consumer_layer_end,
    };
}

fn boundaryRefEql(lhs: TensorFrameBoundaryRef, rhs: TensorFrameBoundaryRef) bool {
    return lhs.boundary_index == rhs.boundary_index and
        lhs.source_stage_id == rhs.source_stage_id and
        lhs.target_stage_id == rhs.target_stage_id and
        lhs.producer_layer_start == rhs.producer_layer_start and
        lhs.producer_layer_end == rhs.producer_layer_end and
        lhs.consumer_layer_start == rhs.consumer_layer_start and
        lhs.consumer_layer_end == rhs.consumer_layer_end;
}

fn validateBoundaryRefMatches(expected: TensorFrameBoundaryRef, actual: TensorFrameBoundaryRef) TensorFrameValidationError!void {
    if (actual.boundary_index != expected.boundary_index) return error.BoundaryTensorContractMismatch;
    if (actual.source_stage_id != expected.source_stage_id) return error.InvalidSourceStage;
    if (actual.target_stage_id != expected.target_stage_id) return error.InvalidTargetStage;
    if (actual.producer_layer_start != expected.producer_layer_start or
        actual.producer_layer_end != expected.producer_layer_end)
    {
        return error.InvalidProducerLayerRange;
    }
    if (actual.consumer_layer_start != expected.consumer_layer_start or
        actual.consumer_layer_end != expected.consumer_layer_end)
    {
        return error.InvalidConsumerLayerRange;
    }
}

fn checkedActivationElementCount(shape: [max_tensor_rank]u64) TensorFrameValidationError!u64 {
    const batch_tokens = std.math.mul(u64, shape[0], shape[1]) catch return error.ByteCountOverflow;
    return std.math.mul(u64, batch_tokens, shape[2]) catch error.ByteCountOverflow;
}

fn checkedByteCount(element_count: u64, dtype: TensorFrameDType) TensorFrameValidationError!u64 {
    return std.math.mul(u64, element_count, dtypeByteSize(dtype)) catch error.ByteCountOverflow;
}

fn validateBatch(
    batch: TensorFrameBatch,
    expected_batch_count: u64,
    expected_token_count: u64,
    step_kind: TensorFrameStepKind,
) TensorFrameValidationError!void {
    if (batch.entries.len == 0) return error.InvalidBatch;
    const expected_len = std.math.cast(usize, expected_batch_count) orelse return error.InvalidBatch;
    if (batch.entries.len != expected_len) return error.InvalidBatch;

    for (batch.entries, 0..) |entry, index| {
        if (entry.batch_index >= expected_batch_count) return error.InvalidBatch;
        if (entry.request_id == 0) return error.InvalidRequestId;
        if (entry.slot_id == 0) return error.InvalidSlotId;
        if (entry.token_count != expected_token_count) return error.RaggedBatchUnsupported;
        if (step_kind == .decode and entry.token_count != 1) return error.InvalidSequenceRange;
        _ = std.math.add(u64, entry.sequence_start, entry.token_count) catch return error.InvalidSequenceRange;

        for (batch.entries[0..index]) |previous| {
            if (previous.batch_index == entry.batch_index) return error.DuplicateBatchIndex;
            if (previous.request_id == entry.request_id) return error.DuplicateRequestId;
            if (previous.slot_id == entry.slot_id) return error.DuplicateSlotId;
        }
    }
}

fn validatePayload(payload: TensorFramePayload, expected_byte_count: u64) TensorFrameValidationError!void {
    if (payload.layout != .row_major) return error.UnsupportedTensorLayout;
    if (payload.byte_count == 0 or payload.byte_count != expected_byte_count) {
        return error.InvalidPayloadByteCount;
    }
    if (payload.lifetime == .slot_persistent_metadata_ref and payload.ownership != .external_handle) {
        return error.InvalidLifetime;
    }
}

const HashEncoder = struct {
    hasher: Sha256 = Sha256.init(.{}),

    fn init() HashEncoder {
        return .{};
    }

    fn writeU8(self: *HashEncoder, value: u8) void {
        self.hasher.update(&.{value});
    }

    fn writeU32(self: *HashEncoder, value: u32) void {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, value, .little);
        self.hasher.update(&bytes);
    }

    fn writeU64(self: *HashEncoder, value: u64) void {
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, value, .little);
        self.hasher.update(&bytes);
    }

    fn writeUsize(self: *HashEncoder, value: usize) void {
        self.writeU64(@intCast(value));
    }

    fn writeBytes(self: *HashEncoder, bytes: []const u8) void {
        self.hasher.update(bytes);
    }

    fn finish(self: *HashEncoder) [32]u8 {
        var digest: [32]u8 = undefined;
        self.hasher.final(&digest);
        return digest;
    }
};

fn writeLogicalFrame(encoder: *HashEncoder, metadata: *const TensorFrameMetadata) void {
    encoder.writeBytes("talu.tensor_frame.logical");
    encoder.writeU32(metadata.version);
    encoder.writeU32(metadata.payload_layout_contract_version);
    encoder.writeU64(metadata.frame_id.value);
    encoder.writeBytes(&metadata.plan.graph_digest);
    encoder.writeU32(metadata.plan.graph_contract_version);
    encoder.writeU32(metadata.plan.stage_plan_contract_version);
    encoder.writeBytes(&metadata.plan.stage_plan_id.digest);
    writeBoundaryRef(encoder, metadata.boundary);
    encoder.writeU8(@intFromEnum(metadata.selected_contract.dtype));
    encoder.writeU8(@intFromEnum(metadata.selected_contract.layout));
    encoder.writeU8(@intFromEnum(metadata.selected_contract.source));
    encoder.writeU8(@intFromEnum(metadata.role));
    encoder.writeU8(@intFromEnum(metadata.step_kind));
    encoder.writeU64(metadata.shape_context.expected_hidden_size);
    encoder.writeU8(metadata.tensor.rank);
    for (metadata.tensor.shape) |value| encoder.writeU64(value);
    for (metadata.tensor.stride_elems) |value| encoder.writeU64(value);
    encoder.writeU64(metadata.tensor.logical_element_count);
    encoder.writeU64(metadata.tensor.logical_byte_count);
    encoder.writeU64(metadata.tensor.payload_byte_count);
    encoder.writeUsize(metadata.batch.entries.len);
    for (metadata.batch.entries) |entry| {
        encoder.writeU32(entry.batch_index);
        encoder.writeU64(entry.request_id);
        encoder.writeU64(entry.slot_id);
        encoder.writeU64(entry.sequence_start);
        encoder.writeU64(entry.token_count);
    }
    encoder.writeU64(metadata.payload.byte_count);
    encoder.writeU8(@intFromEnum(metadata.payload.layout));
    encoder.writeU8(@intFromEnum(metadata.payload.ownership));
    encoder.writeU8(@intFromEnum(metadata.payload.lifetime));
}

fn writeBoundaryRef(encoder: *HashEncoder, boundary: TensorFrameBoundaryRef) void {
    encoder.writeUsize(boundary.boundary_index);
    encoder.writeUsize(boundary.source_stage_id);
    encoder.writeUsize(boundary.target_stage_id);
    encoder.writeUsize(boundary.producer_layer_start);
    encoder.writeUsize(boundary.producer_layer_end);
    encoder.writeUsize(boundary.consumer_layer_start);
    encoder.writeUsize(boundary.consumer_layer_end);
}

test "inference bridge dtypeByteSize returns transfer element sizes" {
    try std.testing.expectEqual(@as(u64, 2), dtypeByteSize(.bf16));
    try std.testing.expectEqual(@as(u64, 2), dtypeByteSize(.f16));
    try std.testing.expectEqual(@as(u64, 4), dtypeByteSize(.f32));
}

test "inference bridge TensorFramePlanRef.fromStagePlan copies fixed identity and boundary summaries" {
    var fixture = try PlanFixture.init(std.testing.allocator, &.{2});
    defer fixture.deinit();

    var plan_ref = try TensorFramePlanRef.fromStagePlan(std.testing.allocator, &fixture.plan);
    defer plan_ref.deinit();

    try std.testing.expectEqualSlices(u8, &fixture.plan.graph_identity.digest, &plan_ref.identity.graph_digest);
    try std.testing.expectEqual(fixture.plan.graph_identity.graph_contract_version, plan_ref.identity.graph_contract_version);
    try std.testing.expectEqual(fixture.plan.stage_contract_version, plan_ref.identity.stage_plan_contract_version);
    try std.testing.expectEqualSlices(u8, &fixture.plan.plan_id.digest, &plan_ref.identity.stage_plan_id.digest);
    try std.testing.expectEqual(@as(usize, 1), plan_ref.boundaries.len);
    try std.testing.expectEqual(@as(usize, 0), plan_ref.boundaries[0].boundary_index);
    try std.testing.expectEqual(@as(usize, 0), plan_ref.boundaries[0].producer_layer_start);
    try std.testing.expectEqual(@as(usize, 2), plan_ref.boundaries[0].producer_layer_end);
}

test "inference bridge TensorFramePlanRef.fromStagePlan rejects invalid StagePlan" {
    var fixture = try PlanFixture.init(std.testing.allocator, &.{2});
    defer fixture.deinit();
    fixture.plan.stage_contract_version += 1;

    try std.testing.expectError(error.InvalidContractVersion, TensorFramePlanRef.fromStagePlan(std.testing.allocator, &fixture.plan));
}

test "inference bridge boundaryRefFromPlanRef rejects missing and out-of-range boundaries" {
    var one_stage = try PlanFixture.init(std.testing.allocator, &.{});
    defer one_stage.deinit();
    var one_ref = try TensorFramePlanRef.fromStagePlan(std.testing.allocator, &one_stage.plan);
    defer one_ref.deinit();
    try std.testing.expectError(error.AbsentStageBoundary, boundaryRefFromPlanRef(&one_ref, 0));

    var two_stage = try PlanFixture.init(std.testing.allocator, &.{2});
    defer two_stage.deinit();
    var two_ref = try TensorFramePlanRef.fromStagePlan(std.testing.allocator, &two_stage.plan);
    defer two_ref.deinit();
    try std.testing.expectError(error.BoundaryIndexOutOfRange, boundaryRefFromPlanRef(&two_ref, 1));
}

test "inference bridge activationDecodeFrame builds a StagePlan-derived frame" {
    var fixture = try PlanFixture.init(std.testing.allocator, &.{2});
    defer fixture.deinit();
    var plan_ref = try TensorFramePlanRef.fromStagePlan(std.testing.allocator, &fixture.plan);
    defer plan_ref.deinit();

    const frame = try makeDecodeFrame(&plan_ref, .{ .cuda = 7 });
    try frame.validate();
    try validateTensorFramePlanIdentity(&frame, &plan_ref);
    try std.testing.expectEqual(TensorFrameRole.activation, frame.role);
    try std.testing.expectEqual(@as(usize, 0), frame.boundary.boundary_index);
    try std.testing.expectEqual(@as(u64, 32), frame.payload.byte_count);
    try std.testing.expectEqual(@as(u64, 123), frame.batch.entries[0].request_id);
}

test "inference bridge activationDecodeFrame rejects missing selected boundary tensor contract" {
    var fixture = try PlanFixture.init(std.testing.allocator, &.{2});
    defer fixture.deinit();
    var plan_ref = try TensorFramePlanRef.fromStagePlan(std.testing.allocator, &fixture.plan);
    defer plan_ref.deinit();

    const tensor = try TensorFrameTensorDesc.contiguousActivation(.f32, .{ 1, 1, 8, 0 });
    const batch = [_]TensorFrameBatchEntry{batchEntry(0, 123, 7, 9, 1)};
    try std.testing.expectError(error.MissingSelectedBoundaryTensorContract, activationDecodeFrame(.{
        .frame_id = try TensorFrameInstanceId.init(1),
        .plan_ref = &plan_ref,
        .boundary_index = 0,
        .selected_contract = null,
        .shape_context = .{ .expected_hidden_size = 8 },
        .tensor = tensor,
        .batch = .{ .entries = &batch },
        .payload = .{ .byte_count = tensor.payload_byte_count },
    }));
}

test "inference bridge activationDecodeFrame rejects boundary tensor contract mismatch" {
    var fixture = try PlanFixture.init(std.testing.allocator, &.{2});
    defer fixture.deinit();
    var plan_ref = try TensorFramePlanRef.fromStagePlan(std.testing.allocator, &fixture.plan);
    defer plan_ref.deinit();

    var contract = try selectedBoundaryTensorContract(&plan_ref, 0, .f32, .row_major, .explicit);
    contract.boundary.target_stage_id += 1;
    const tensor = try TensorFrameTensorDesc.contiguousActivation(.f32, .{ 1, 1, 8, 0 });
    const batch = [_]TensorFrameBatchEntry{batchEntry(0, 123, 7, 9, 1)};
    try std.testing.expectError(error.InvalidTargetStage, activationDecodeFrame(.{
        .frame_id = try TensorFrameInstanceId.init(1),
        .plan_ref = &plan_ref,
        .boundary_index = 0,
        .selected_contract = &contract,
        .shape_context = .{ .expected_hidden_size = 8 },
        .tensor = tensor,
        .batch = .{ .entries = &batch },
        .payload = .{ .byte_count = tensor.payload_byte_count },
    }));
}

test "inference bridge validateTensorFramePlanIdentity rejects graph and stage plan mismatch" {
    var fixture = try PlanFixture.init(std.testing.allocator, &.{2});
    defer fixture.deinit();
    var plan_ref = try TensorFramePlanRef.fromStagePlan(std.testing.allocator, &fixture.plan);
    defer plan_ref.deinit();

    var frame = try makeDecodeFrame(&plan_ref, null);
    frame.plan.graph_digest[0] ^= 0x01;
    try std.testing.expectError(error.GraphIdentityMismatch, validateTensorFramePlanIdentity(&frame, &plan_ref));

    frame = try makeDecodeFrame(&plan_ref, null);
    frame.plan.stage_plan_id.digest[0] ^= 0x01;
    try std.testing.expectError(error.StagePlanIdentityMismatch, validateTensorFramePlanIdentity(&frame, &plan_ref));
}

test "inference bridge activation frame validates shape hidden width stride and step kind" {
    var fixture = try PlanFixture.init(std.testing.allocator, &.{2});
    defer fixture.deinit();
    var plan_ref = try TensorFramePlanRef.fromStagePlan(std.testing.allocator, &fixture.plan);
    defer plan_ref.deinit();
    const contract = try selectedBoundaryTensorContract(&plan_ref, 0, .f32, .row_major, .explicit);
    const batch = [_]TensorFrameBatchEntry{batchEntry(0, 123, 7, 9, 1)};

    var tensor = try TensorFrameTensorDesc.contiguousActivation(.f32, .{ 1, 1, 8, 0 });
    try std.testing.expectError(error.InvalidHiddenSize, activationDecodeFrame(.{
        .frame_id = try TensorFrameInstanceId.init(1),
        .plan_ref = &plan_ref,
        .boundary_index = 0,
        .selected_contract = &contract,
        .shape_context = .{ .expected_hidden_size = 4 },
        .tensor = tensor,
        .batch = .{ .entries = &batch },
        .payload = .{ .byte_count = tensor.payload_byte_count },
    }));

    tensor.stride_elems[1] += 1;
    try std.testing.expectError(error.InvalidTensorStride, activationDecodeFrame(.{
        .frame_id = try TensorFrameInstanceId.init(2),
        .plan_ref = &plan_ref,
        .boundary_index = 0,
        .selected_contract = &contract,
        .shape_context = .{ .expected_hidden_size = 8 },
        .tensor = tensor,
        .batch = .{ .entries = &batch },
        .payload = .{ .byte_count = tensor.payload_byte_count },
    }));

    const prefill_tensor = try TensorFrameTensorDesc.contiguousActivation(.f32, .{ 1, 4, 8, 0 });
    const prefill_batch = [_]TensorFrameBatchEntry{batchEntry(0, 123, 7, 9, 4)};
    _ = try activationPrefillFrame(.{
        .frame_id = try TensorFrameInstanceId.init(3),
        .plan_ref = &plan_ref,
        .boundary_index = 0,
        .selected_contract = &contract,
        .shape_context = .{ .expected_hidden_size = 8, .expected_step_kind = .prefill },
        .tensor = prefill_tensor,
        .batch = .{ .entries = &prefill_batch },
        .payload = .{ .byte_count = prefill_tensor.payload_byte_count },
    });
    try std.testing.expectError(error.InvalidTensorShape, activationDecodeFrame(.{
        .frame_id = try TensorFrameInstanceId.init(4),
        .plan_ref = &plan_ref,
        .boundary_index = 0,
        .selected_contract = &contract,
        .shape_context = .{ .expected_hidden_size = 8 },
        .tensor = prefill_tensor,
        .batch = .{ .entries = &prefill_batch },
        .payload = .{ .byte_count = prefill_tensor.payload_byte_count },
    }));
}

test "inference bridge activation frame validates byte counts and payload buffer length" {
    var fixture = try PlanFixture.init(std.testing.allocator, &.{2});
    defer fixture.deinit();
    var plan_ref = try TensorFramePlanRef.fromStagePlan(std.testing.allocator, &fixture.plan);
    defer plan_ref.deinit();

    var frame = try makeDecodeFrame(&plan_ref, null);
    try validatePayloadBufferLength(&frame, 32);
    try std.testing.expectError(error.PayloadBufferLengthMismatch, validatePayloadBufferLength(&frame, 36));

    frame.payload.byte_count += 4;
    try std.testing.expectError(error.InvalidPayloadByteCount, frame.validate());
}

test "inference bridge activation frame rejects invalid frame id and unsupported compute dtypes" {
    try std.testing.expectError(error.InvalidFrameId, TensorFrameInstanceId.init(0));
    try std.testing.expectEqual(compute.DType.f32, toComputeDType(.f32));
    try std.testing.expectEqual(TensorFrameDType.bf16, try fromComputeDType(.bf16));
    try std.testing.expectError(error.UnsupportedActivationDType, fromComputeDType(.i8));
    try std.testing.expectError(error.UnsupportedActivationDType, fromComputeDType(.grouped_affine_u4));
}

test "inference bridge activation frame validates explicit non-ragged unique batch entries" {
    var fixture = try PlanFixture.init(std.testing.allocator, &.{2});
    defer fixture.deinit();
    var plan_ref = try TensorFramePlanRef.fromStagePlan(std.testing.allocator, &fixture.plan);
    defer plan_ref.deinit();
    const contract = try selectedBoundaryTensorContract(&plan_ref, 0, .f32, .row_major, .explicit);
    const tensor = try TensorFrameTensorDesc.contiguousActivation(.f32, .{ 2, 1, 8, 0 });

    const ok = [_]TensorFrameBatchEntry{
        batchEntry(0, 123, 7, 9, 1),
        batchEntry(1, 124, 8, 9, 1),
    };
    _ = try activationDecodeFrame(.{
        .frame_id = try TensorFrameInstanceId.init(1),
        .plan_ref = &plan_ref,
        .boundary_index = 0,
        .selected_contract = &contract,
        .shape_context = .{ .expected_hidden_size = 8 },
        .tensor = tensor,
        .batch = .{ .entries = &ok },
        .payload = .{ .byte_count = tensor.payload_byte_count },
    });

    const duplicate_batch = [_]TensorFrameBatchEntry{
        batchEntry(0, 123, 7, 9, 1),
        batchEntry(0, 124, 8, 9, 1),
    };
    try std.testing.expectError(error.DuplicateBatchIndex, activationDecodeFrame(.{
        .frame_id = try TensorFrameInstanceId.init(2),
        .plan_ref = &plan_ref,
        .boundary_index = 0,
        .selected_contract = &contract,
        .shape_context = .{ .expected_hidden_size = 8 },
        .tensor = tensor,
        .batch = .{ .entries = &duplicate_batch },
        .payload = .{ .byte_count = tensor.payload_byte_count },
    }));

    const duplicate_request = [_]TensorFrameBatchEntry{
        batchEntry(0, 123, 7, 9, 1),
        batchEntry(1, 123, 8, 9, 1),
    };
    try std.testing.expectError(error.DuplicateRequestId, activationDecodeFrame(.{
        .frame_id = try TensorFrameInstanceId.init(3),
        .plan_ref = &plan_ref,
        .boundary_index = 0,
        .selected_contract = &contract,
        .shape_context = .{ .expected_hidden_size = 8 },
        .tensor = tensor,
        .batch = .{ .entries = &duplicate_request },
        .payload = .{ .byte_count = tensor.payload_byte_count },
    }));

    const duplicate_slot = [_]TensorFrameBatchEntry{
        batchEntry(0, 123, 7, 9, 1),
        batchEntry(1, 124, 7, 9, 1),
    };
    try std.testing.expectError(error.DuplicateSlotId, activationDecodeFrame(.{
        .frame_id = try TensorFrameInstanceId.init(4),
        .plan_ref = &plan_ref,
        .boundary_index = 0,
        .selected_contract = &contract,
        .shape_context = .{ .expected_hidden_size = 8 },
        .tensor = tensor,
        .batch = .{ .entries = &duplicate_slot },
        .payload = .{ .byte_count = tensor.payload_byte_count },
    }));
}

test "inference bridge activation frame rejects zero request slot and sequence overflow" {
    var fixture = try PlanFixture.init(std.testing.allocator, &.{2});
    defer fixture.deinit();
    var plan_ref = try TensorFramePlanRef.fromStagePlan(std.testing.allocator, &fixture.plan);
    defer plan_ref.deinit();
    const contract = try selectedBoundaryTensorContract(&plan_ref, 0, .f32, .row_major, .explicit);
    const tensor = try TensorFrameTensorDesc.contiguousActivation(.f32, .{ 1, 1, 8, 0 });

    const zero_request = [_]TensorFrameBatchEntry{batchEntry(0, 0, 7, 9, 1)};
    try std.testing.expectError(error.InvalidRequestId, activationDecodeFrame(.{
        .frame_id = try TensorFrameInstanceId.init(1),
        .plan_ref = &plan_ref,
        .boundary_index = 0,
        .selected_contract = &contract,
        .shape_context = .{ .expected_hidden_size = 8 },
        .tensor = tensor,
        .batch = .{ .entries = &zero_request },
        .payload = .{ .byte_count = tensor.payload_byte_count },
    }));

    const zero_slot = [_]TensorFrameBatchEntry{batchEntry(0, 123, 0, 9, 1)};
    try std.testing.expectError(error.InvalidSlotId, activationDecodeFrame(.{
        .frame_id = try TensorFrameInstanceId.init(2),
        .plan_ref = &plan_ref,
        .boundary_index = 0,
        .selected_contract = &contract,
        .shape_context = .{ .expected_hidden_size = 8 },
        .tensor = tensor,
        .batch = .{ .entries = &zero_slot },
        .payload = .{ .byte_count = tensor.payload_byte_count },
    }));

    const sequence_overflow = [_]TensorFrameBatchEntry{batchEntry(0, 123, 7, std.math.maxInt(u64), 1)};
    try std.testing.expectError(error.InvalidSequenceRange, activationDecodeFrame(.{
        .frame_id = try TensorFrameInstanceId.init(3),
        .plan_ref = &plan_ref,
        .boundary_index = 0,
        .selected_contract = &contract,
        .shape_context = .{ .expected_hidden_size = 8 },
        .tensor = tensor,
        .batch = .{ .entries = &sequence_overflow },
        .payload = .{ .byte_count = tensor.payload_byte_count },
    }));
}

test "inference bridge payload location hint is excluded from logical equality and hash" {
    var fixture = try PlanFixture.init(std.testing.allocator, &.{2});
    defer fixture.deinit();
    var plan_ref = try TensorFramePlanRef.fromStagePlan(std.testing.allocator, &fixture.plan);
    defer plan_ref.deinit();

    var cpu_frame = try makeDecodeFrame(&plan_ref, .cpu);
    var cuda_frame = try makeDecodeFrame(&plan_ref, .{ .cuda = 7 });
    try std.testing.expect(tensorFrameLogicalEql(&cpu_frame, &cuda_frame));
    try std.testing.expectEqualSlices(u8, &tensorFrameLogicalHash(&cpu_frame), &tensorFrameLogicalHash(&cuda_frame));
}

test "inference bridge payload ownership and lifetime validation rejects invalid combinations" {
    var fixture = try PlanFixture.init(std.testing.allocator, &.{2});
    defer fixture.deinit();
    var plan_ref = try TensorFramePlanRef.fromStagePlan(std.testing.allocator, &fixture.plan);
    defer plan_ref.deinit();

    var frame = try makeDecodeFrame(&plan_ref, null);
    frame.payload.lifetime = .slot_persistent_metadata_ref;
    frame.payload.ownership = .borrowed_until_next_stage_call;
    try std.testing.expectError(error.InvalidLifetime, frame.validate());
    frame.payload.ownership = .external_handle;
    try frame.validate();
}

test "inference bridge emitTensorFrame no-op best-effort and strict observer behavior" {
    var fixture = try PlanFixture.init(std.testing.allocator, &.{2});
    defer fixture.deinit();
    var plan_ref = try TensorFramePlanRef.fromStagePlan(std.testing.allocator, &fixture.plan);
    defer plan_ref.deinit();
    const frame = try makeDecodeFrame(&plan_ref, null);

    try emitTensorFrame(TensorFrameObserver.noop(), .strict, &frame);

    const Capture = struct {
        count: usize = 0,

        fn emit(ctx: ?*anyopaque, metadata: *const TensorFrameMetadata) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            try metadata.validate();
            self.count += 1;
        }

        fn fail(_: ?*anyopaque, _: *const TensorFrameMetadata) anyerror!void {
            return error.TestObserverFailure;
        }
    };

    var capture = Capture{};
    try emitTensorFrame(.{ .ctx = &capture, .emit_fn = Capture.emit }, .strict, &frame);
    try std.testing.expectEqual(@as(usize, 1), capture.count);
    try emitTensorFrame(.{ .emit_fn = Capture.fail }, .best_effort, &frame);
    try std.testing.expectError(error.ObserverFailure, emitTensorFrame(.{ .emit_fn = Capture.fail }, .strict, &frame));
}

fn batchEntry(
    batch_index: u32,
    request_id: u64,
    slot_id: u64,
    sequence_start: u64,
    token_count: u64,
) TensorFrameBatchEntry {
    return .{
        .batch_index = batch_index,
        .request_id = request_id,
        .slot_id = slot_id,
        .sequence_start = sequence_start,
        .token_count = token_count,
    };
}

const single_decode_batch = [_]TensorFrameBatchEntry{.{
    .batch_index = 0,
    .request_id = 123,
    .slot_id = 7,
    .sequence_start = 9,
    .token_count = 1,
}};

fn makeDecodeFrame(
    plan_ref: *const TensorFramePlanRef,
    location_hint: ?TensorFramePayloadLocationHint,
) TensorFrameValidationError!TensorFrameMetadata {
    const contract = try selectedBoundaryTensorContract(plan_ref, 0, .f32, .row_major, .explicit);
    const tensor = try TensorFrameTensorDesc.contiguousActivation(.f32, .{ 1, 1, 8, 0 });
    return activationDecodeFrame(.{
        .frame_id = try TensorFrameInstanceId.init(1),
        .plan_ref = plan_ref,
        .boundary_index = 0,
        .selected_contract = &contract,
        .shape_context = .{ .expected_hidden_size = 8, .expected_step_kind = .decode },
        .tensor = tensor,
        .batch = .{ .entries = &single_decode_batch },
        .payload = .{
            .byte_count = tensor.payload_byte_count,
            .location_hint = location_hint,
            .ownership = .borrowed_until_next_stage_call,
            .lifetime = .step_scoped,
        },
    });
}

const PlanFixture = struct {
    manifest: models.manifest.ModelManifest,
    plan: stage_plan.StagePlan,

    fn init(allocator: Allocator, split_points: []const usize) !PlanFixture {
        var manifest = try makeStagePlanManifest(allocator);
        errdefer manifest.deinit();

        var config = models.config.ModelConfig{
            .vocab_size = 16,
            .d_model = 8,
            .n_layers = 4,
            .n_heads = 1,
            .n_kv_groups = 1,
            .d_ff = 32,
            .max_seq_len = 32,
            .head_dim = 8,
            .rope_theta = 10000.0,
            .norm_eps = 0.00001,
            .gaffine_group_size = 32,
            .tie_word_embeddings = false,
        };
        const arch = models.op_types.Architecture{
            .name = "tensor_frame_test",
            .model_types = &.{"tensor_frame_test"},
        };

        var plan = try stage_plan.buildStagePlan(allocator, .{
            .n_layers = 4,
            .split_points = split_points,
            .architecture = &arch,
            .model_config = &config,
            .manifest = &manifest,
            .partition_constraints = if (split_points.len == 0) null else .{ .decoder_cuts_allowed = true },
        });
        errdefer plan.deinit();

        return .{ .manifest = manifest, .plan = plan };
    }

    fn deinit(self: *PlanFixture) void {
        self.plan.deinit();
        self.manifest.deinit();
    }
};

fn makeStagePlanManifest(allocator: Allocator) !models.manifest.ModelManifest {
    const manifest_mod = models.manifest;
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    const entries = try arena_allocator.alloc(manifest_mod.TensorManifestEntry, 6);
    entries[0] = .{
        .name = try arena_allocator.dupe(u8, "model.embed_tokens.weight"),
        .dtype = .f16,
        .shape = try arena_allocator.dupe(usize, &.{ 16, 8 }),
        .checkpoint_bytes = 256,
        .role = .token_embeddings,
        .weight_id = try arena_allocator.dupe(u8, "token_embeddings"),
        .status = .architecture_weight,
    };
    for (0..4) |layer| {
        entries[layer + 1] = .{
            .name = try std.fmt.allocPrint(arena_allocator, "model.layers.{d}.self_attn.q_proj.weight", .{layer}),
            .dtype = .f16,
            .shape = try arena_allocator.dupe(usize, &.{ 8, 8 }),
            .checkpoint_bytes = 128,
            .role = .decoder_layer,
            .layer_index = layer,
            .weight_id = try arena_allocator.dupe(u8, "self_attn.q_proj.weight"),
            .status = .architecture_weight,
        };
    }
    entries[5] = .{
        .name = try arena_allocator.dupe(u8, "lm_head.weight"),
        .dtype = .f16,
        .shape = try arena_allocator.dupe(usize, &.{ 16, 8 }),
        .checkpoint_bytes = 256,
        .role = .lm_head,
        .weight_id = try arena_allocator.dupe(u8, "lm_head"),
        .status = .architecture_weight,
    };

    var role_bytes = [_]usize{0} ** manifest_mod.role_count;
    var total_checkpoint_bytes: usize = 0;
    for (entries) |entry| {
        total_checkpoint_bytes += entry.checkpoint_bytes;
        role_bytes[@intFromEnum(entry.role)] += entry.checkpoint_bytes;
    }

    return .{
        .arena = arena,
        .architecture_id = try arena_allocator.dupe(u8, "tensor_frame_test"),
        .layer_count = 4,
        .entries = entries,
        .total_checkpoint_bytes = total_checkpoint_bytes,
        .role_bytes = role_bytes,
    };
}
