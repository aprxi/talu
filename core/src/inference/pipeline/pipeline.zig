//! Boundary negotiation primitives for local multi-stage inference.

const std = @import("std");

pub const BoundaryDType = enum(u8) {
    bf16,
    f16,
    f32,
};

pub const BoundaryLayout = enum(u8) {
    row_major,
};

pub const BoundaryNegotiationRequest = struct {
    stage0_native_dtype: BoundaryDType,
    stage1_native_dtype: BoundaryDType,
    stage0_supported_boundary_dtypes: []const BoundaryDType,
    stage1_supported_boundary_dtypes: []const BoundaryDType,
    preferred_boundary_dtypes: []const BoundaryDType = &.{ .bf16, .f16, .f32 },
    layout: BoundaryLayout = .row_major,
};

pub const BoundaryNegotiationResult = struct {
    boundary_dtype: BoundaryDType,
    layout: BoundaryLayout,
    stage0_requires_conversion: bool,
    stage1_requires_conversion: bool,
};

fn containsBoundaryDType(list: []const BoundaryDType, dtype: BoundaryDType) bool {
    for (list) |candidate| {
        if (candidate == dtype) return true;
    }
    return false;
}

/// Negotiate an explicit stage boundary contract.
/// Chooses the highest-priority mutually-supported dtype and records whether
/// stage0/stage1 need conversion from their native execution dtype.
pub fn negotiateBoundaryContract(req: BoundaryNegotiationRequest) !BoundaryNegotiationResult {
    for (req.preferred_boundary_dtypes) |candidate| {
        if (!containsBoundaryDType(req.stage0_supported_boundary_dtypes, candidate)) continue;
        if (!containsBoundaryDType(req.stage1_supported_boundary_dtypes, candidate)) continue;
        return .{
            .boundary_dtype = candidate,
            .layout = req.layout,
            .stage0_requires_conversion = candidate != req.stage0_native_dtype,
            .stage1_requires_conversion = candidate != req.stage1_native_dtype,
        };
    }
    return error.NoCompatibleBoundaryDType;
}

test "negotiateBoundaryContract chooses highest-preference mutually supported dtype" {
    const result = try negotiateBoundaryContract(.{
        .stage0_native_dtype = .f16,
        .stage1_native_dtype = .bf16,
        .stage0_supported_boundary_dtypes = &.{ .f16, .f32 },
        .stage1_supported_boundary_dtypes = &.{ .bf16, .f16, .f32 },
        .preferred_boundary_dtypes = &.{ .bf16, .f16, .f32 },
    });
    try std.testing.expectEqual(BoundaryDType.f16, result.boundary_dtype);
    try std.testing.expect(!result.stage0_requires_conversion);
    try std.testing.expect(result.stage1_requires_conversion);
}

test "negotiateBoundaryContract reports conversion flags when f32 is the only mutual dtype" {
    const result = try negotiateBoundaryContract(.{
        .stage0_native_dtype = .bf16,
        .stage1_native_dtype = .f16,
        .stage0_supported_boundary_dtypes = &.{ .bf16, .f32 },
        .stage1_supported_boundary_dtypes = &.{ .f16, .f32 },
        .preferred_boundary_dtypes = &.{ .bf16, .f16, .f32 },
    });
    try std.testing.expectEqual(BoundaryDType.f32, result.boundary_dtype);
    try std.testing.expect(result.stage0_requires_conversion);
    try std.testing.expect(result.stage1_requires_conversion);
}

test "negotiateBoundaryContract fails when no mutual dtype exists" {
    try std.testing.expectError(error.NoCompatibleBoundaryDType, negotiateBoundaryContract(.{
        .stage0_native_dtype = .bf16,
        .stage1_native_dtype = .f16,
        .stage0_supported_boundary_dtypes = &.{.bf16},
        .stage1_supported_boundary_dtypes = &.{.f16},
        .preferred_boundary_dtypes = &.{ .bf16, .f16, .f32 },
    }));
}
