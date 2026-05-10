//! Tensor-frame metadata for staged inference boundaries.
//!
//! This file defines cold-path contracts for describing activation handoffs
//! between stage owners. It carries metadata only: payload bytes or device
//! handles stay owned by the backend/transport path that performs the transfer.

const std = @import("std");
const pipeline = @import("pipeline.zig");

pub const tensor_frame_contract_version: u8 = 1;
pub const max_tensor_rank: u8 = 4;

pub const TensorFrameDType = pipeline.BoundaryDType;
pub const TensorFrameLayout = pipeline.BoundaryLayout;

pub const StageBackend = enum(u8) {
    cpu,
    cuda,
    metal,
    remote,
};

pub const StageEndpoint = struct {
    stage_id: u16,
    backend: StageBackend,
};

pub const TensorFrameRole = enum(u8) {
    activation,
    logits,
    embedding,
    vision_embedding,
    kv_state,
};

pub const TensorFrameOwnership = enum(u8) {
    borrowed_until_next_stage_call,
    owned_by_sender,
    owned_by_receiver,
    external_handle,
};

pub const TensorFrameLifetime = enum(u8) {
    step_scoped,
    request_scoped,
    slot_persistent,
};

pub const TensorFrameDevice = union(enum(u8)) {
    cpu,
    cuda: u16,
    metal: u16,
    remote: u32,
};

pub const TensorFrameValidationError = error{
    InvalidBridgeContractVersion,
    InvalidStageBoundary,
    InvalidLayerRange,
    InvalidTensorRank,
    InvalidTensorShape,
    InvalidTensorStride,
    InvalidTensorByteCount,
    InvalidBatchSize,
    InvalidSequenceRange,
};

pub const StageBoundary = struct {
    source: StageEndpoint,
    target: StageEndpoint,
    producer_layer_start: u32,
    producer_layer_end: u32,
    consumer_layer_start: u32,
    consumer_layer_end: u32,
    dtype: TensorFrameDType,
    layout: TensorFrameLayout = .row_major,

    pub fn validate(self: *const StageBoundary) TensorFrameValidationError!void {
        try validateStageBoundary(self.*);
    }
};

pub const TensorFrameShape = struct {
    rank: u8,
    dims: [max_tensor_rank]u64,
    stride_elems: [max_tensor_rank]u64,

    pub fn contiguous(rank: u8, dims: [max_tensor_rank]u64) TensorFrameValidationError!TensorFrameShape {
        if (rank == 0 or rank > max_tensor_rank) return error.InvalidTensorRank;

        var shape = TensorFrameShape{
            .rank = rank,
            .dims = dims,
            .stride_elems = [_]u64{0} ** max_tensor_rank,
        };

        var stride: u64 = 1;
        var reverse_idx: usize = rank;
        while (reverse_idx > 0) {
            reverse_idx -= 1;
            if (dims[reverse_idx] == 0) return error.InvalidTensorShape;
            shape.stride_elems[reverse_idx] = stride;
            stride = std.math.mul(u64, stride, dims[reverse_idx]) catch return error.InvalidTensorShape;
        }

        var idx: usize = rank;
        while (idx < max_tensor_rank) : (idx += 1) {
            if (dims[idx] != 0) return error.InvalidTensorShape;
        }

        return shape;
    }

    pub fn validate(self: *const TensorFrameShape) TensorFrameValidationError!void {
        if (self.rank == 0 or self.rank > max_tensor_rank) return error.InvalidTensorRank;

        var idx: usize = 0;
        while (idx < self.rank) : (idx += 1) {
            if (self.dims[idx] == 0) return error.InvalidTensorShape;
            if (self.stride_elems[idx] == 0) return error.InvalidTensorStride;
        }

        while (idx < max_tensor_rank) : (idx += 1) {
            if (self.dims[idx] != 0) return error.InvalidTensorShape;
            if (self.stride_elems[idx] != 0) return error.InvalidTensorStride;
        }
    }

    pub fn elementCount(self: *const TensorFrameShape) TensorFrameValidationError!u64 {
        try self.validate();
        var count: u64 = 1;
        for (0..self.rank) |idx| {
            count = std.math.mul(u64, count, self.dims[idx]) catch return error.InvalidTensorShape;
        }
        return count;
    }

    pub fn byteCount(self: *const TensorFrameShape, dtype: TensorFrameDType) TensorFrameValidationError!u64 {
        const elems = try self.elementCount();
        return std.math.mul(u64, elems, dtypeByteSize(dtype)) catch error.InvalidTensorByteCount;
    }
};

pub const TensorFrameMetadata = struct {
    version: u8 = tensor_frame_contract_version,
    frame_id: u64,
    graph_id: u64,
    request_id: u64,
    boundary: StageBoundary,
    role: TensorFrameRole,
    shape: TensorFrameShape,
    device: TensorFrameDevice,
    sequence_start: u32,
    sequence_len: u32,
    batch_size: u32,
    slot_index: ?u32 = null,
    byte_count: u64,
    checksum: ?u64 = null,
    ownership: TensorFrameOwnership,
    lifetime: TensorFrameLifetime,

    pub fn validate(self: *const TensorFrameMetadata) TensorFrameValidationError!void {
        if (self.version != tensor_frame_contract_version) return error.InvalidBridgeContractVersion;
        try self.boundary.validate();
        try self.shape.validate();
        if (self.batch_size == 0) return error.InvalidBatchSize;
        if (self.sequence_len == 0) return error.InvalidSequenceRange;
        const expected_byte_count = try self.shape.byteCount(self.boundary.dtype);
        if (self.byte_count == 0 or self.byte_count != expected_byte_count) {
            return error.InvalidTensorByteCount;
        }
    }
};

pub fn dtypeByteSize(dtype: TensorFrameDType) u64 {
    return switch (dtype) {
        .bf16, .f16 => 2,
        .f32 => 4,
    };
}

pub fn validateStageBoundary(boundary: StageBoundary) TensorFrameValidationError!void {
    if (boundary.source.stage_id == boundary.target.stage_id and
        boundary.source.backend == boundary.target.backend)
    {
        return error.InvalidStageBoundary;
    }
    if (boundary.producer_layer_end <= boundary.producer_layer_start) return error.InvalidLayerRange;
    if (boundary.consumer_layer_end <= boundary.consumer_layer_start) return error.InvalidLayerRange;
    if (boundary.producer_layer_end != boundary.consumer_layer_start) return error.InvalidLayerRange;
}

pub const ActivationFrameArgs = struct {
    frame_id: u64,
    graph_id: u64,
    request_id: u64,
    boundary: StageBoundary,
    shape: TensorFrameShape,
    device: TensorFrameDevice,
    sequence_start: u32,
    sequence_len: u32,
    batch_size: u32,
    slot_index: ?u32 = null,
    ownership: TensorFrameOwnership = .borrowed_until_next_stage_call,
    lifetime: TensorFrameLifetime = .step_scoped,
    checksum: ?u64 = null,
};

pub fn activationFrameFromBoundary(args: ActivationFrameArgs) TensorFrameValidationError!TensorFrameMetadata {
    const metadata = TensorFrameMetadata{
        .frame_id = args.frame_id,
        .graph_id = args.graph_id,
        .request_id = args.request_id,
        .boundary = args.boundary,
        .role = .activation,
        .shape = args.shape,
        .device = args.device,
        .sequence_start = args.sequence_start,
        .sequence_len = args.sequence_len,
        .batch_size = args.batch_size,
        .slot_index = args.slot_index,
        .byte_count = try args.shape.byteCount(args.boundary.dtype),
        .checksum = args.checksum,
        .ownership = args.ownership,
        .lifetime = args.lifetime,
    };
    try metadata.validate();
    return metadata;
}

test "dtypeByteSize returns transfer element sizes" {
    try std.testing.expectEqual(@as(u64, 2), dtypeByteSize(.bf16));
    try std.testing.expectEqual(@as(u64, 2), dtypeByteSize(.f16));
    try std.testing.expectEqual(@as(u64, 4), dtypeByteSize(.f32));
}

test "validateStageBoundary accepts contiguous producer consumer ranges" {
    try validateStageBoundary(.{
        .source = .{ .stage_id = 0, .backend = .cpu },
        .target = .{ .stage_id = 1, .backend = .cuda },
        .producer_layer_start = 0,
        .producer_layer_end = 8,
        .consumer_layer_start = 8,
        .consumer_layer_end = 24,
        .dtype = .f32,
    });
}

test "validateStageBoundary rejects same endpoint" {
    try std.testing.expectError(error.InvalidStageBoundary, validateStageBoundary(.{
        .source = .{ .stage_id = 0, .backend = .cuda },
        .target = .{ .stage_id = 0, .backend = .cuda },
        .producer_layer_start = 0,
        .producer_layer_end = 8,
        .consumer_layer_start = 8,
        .consumer_layer_end = 24,
        .dtype = .f16,
    }));
}

test "validateStageBoundary rejects split gaps" {
    try std.testing.expectError(error.InvalidLayerRange, validateStageBoundary(.{
        .source = .{ .stage_id = 0, .backend = .cpu },
        .target = .{ .stage_id = 1, .backend = .cuda },
        .producer_layer_start = 0,
        .producer_layer_end = 7,
        .consumer_layer_start = 8,
        .consumer_layer_end = 24,
        .dtype = .f32,
    }));
}

test "TensorFrameShape contiguous builds row-major strides" {
    const shape = try TensorFrameShape.contiguous(3, .{ 2, 4, 8, 0 });
    try std.testing.expectEqual(@as(u64, 32), shape.stride_elems[0]);
    try std.testing.expectEqual(@as(u64, 8), shape.stride_elems[1]);
    try std.testing.expectEqual(@as(u64, 1), shape.stride_elems[2]);
    try std.testing.expectEqual(@as(u64, 64), try shape.elementCount());
    try std.testing.expectEqual(@as(u64, 128), try shape.byteCount(.bf16));
}

test "TensorFrameShape validate rejects inactive metadata" {
    const shape = TensorFrameShape{
        .rank = 2,
        .dims = .{ 2, 4, 1, 0 },
        .stride_elems = .{ 4, 1, 1, 0 },
    };
    try std.testing.expectError(error.InvalidTensorShape, shape.validate());
}

test "TensorFrameShape elementCount rejects zero dimensions" {
    const shape = TensorFrameShape{
        .rank = 2,
        .dims = .{ 2, 0, 0, 0 },
        .stride_elems = .{ 1, 1, 0, 0 },
    };
    try std.testing.expectError(error.InvalidTensorShape, shape.elementCount());
}

test "TensorFrameShape byteCount reports dtype-sized payload" {
    const shape = try TensorFrameShape.contiguous(2, .{ 3, 5, 0, 0 });
    try std.testing.expectEqual(@as(u64, 60), try shape.byteCount(.f32));
}

test "TensorFrameMetadata validate accepts activation boundary metadata" {
    const shape = try TensorFrameShape.contiguous(3, .{ 1, 16, 4096, 0 });
    const metadata = try activationFrameFromBoundary(.{
        .frame_id = 7,
        .graph_id = 11,
        .request_id = 13,
        .boundary = .{
            .source = .{ .stage_id = 0, .backend = .cpu },
            .target = .{ .stage_id = 1, .backend = .cuda },
            .producer_layer_start = 0,
            .producer_layer_end = 12,
            .consumer_layer_start = 12,
            .consumer_layer_end = 24,
            .dtype = .f32,
        },
        .shape = shape,
        .device = .{ .cuda = 0 },
        .sequence_start = 0,
        .sequence_len = 16,
        .batch_size = 1,
        .slot_index = 3,
    });
    try metadata.validate();
    try std.testing.expectEqual(TensorFrameRole.activation, metadata.role);
    try std.testing.expectEqual(@as(u64, 1 * 16 * 4096 * 4), metadata.byte_count);
}

test "TensorFrameMetadata validate rejects stale byte counts" {
    const shape = try TensorFrameShape.contiguous(2, .{ 2, 4, 0, 0 });
    var metadata = try activationFrameFromBoundary(.{
        .frame_id = 1,
        .graph_id = 2,
        .request_id = 3,
        .boundary = .{
            .source = .{ .stage_id = 0, .backend = .cuda },
            .target = .{ .stage_id = 1, .backend = .cuda },
            .producer_layer_start = 0,
            .producer_layer_end = 4,
            .consumer_layer_start = 4,
            .consumer_layer_end = 8,
            .dtype = .f16,
        },
        .shape = shape,
        .device = .{ .cuda = 1 },
        .sequence_start = 0,
        .sequence_len = 1,
        .batch_size = 1,
    });
    metadata.byte_count += 1;
    try std.testing.expectError(error.InvalidTensorByteCount, metadata.validate());
}

test "activationFrameFromBoundary rejects zero batch" {
    const shape = try TensorFrameShape.contiguous(2, .{ 2, 4, 0, 0 });
    try std.testing.expectError(error.InvalidBatchSize, activationFrameFromBoundary(.{
        .frame_id = 1,
        .graph_id = 2,
        .request_id = 3,
        .boundary = .{
            .source = .{ .stage_id = 0, .backend = .metal },
            .target = .{ .stage_id = 1, .backend = .remote },
            .producer_layer_start = 0,
            .producer_layer_end = 4,
            .consumer_layer_start = 4,
            .consumer_layer_end = 8,
            .dtype = .bf16,
        },
        .shape = shape,
        .device = .{ .remote = 2 },
        .sequence_start = 0,
        .sequence_len = 1,
        .batch_size = 0,
    }));
}
