//! Integration tests for inference.executor.RMSNorm
//!
//! Tests the RMSNorm type from the executor module, including:
//! - RMSNorm configuration and fields
//! - Forward pass operations
//! - Type verification

const std = @import("std");
const main = @import("main");

const RMSNorm = main.inference.executor.RMSNorm;
const Tensor = main.Tensor;
const DType = main.DType;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "RMSNorm type is accessible" {
    const T = RMSNorm;
    _ = T;
}

test "RMSNorm is a struct" {
    const info = @typeInfo(RMSNorm);
    try std.testing.expect(info == .@"struct");
}

test "RMSNorm has expected fields" {
    const info = @typeInfo(RMSNorm);
    const fields = info.@"struct".fields;

    var has_weight = false;
    var has_dim = false;
    var has_eps = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "weight")) has_weight = true;
        if (comptime std.mem.eql(u8, field.name, "dim")) has_dim = true;
        if (comptime std.mem.eql(u8, field.name, "eps")) has_eps = true;
    }

    try std.testing.expect(has_weight);
    try std.testing.expect(has_dim);
    try std.testing.expect(has_eps);
}

test "RMSNorm has weight_offset field" {
    const info = @typeInfo(RMSNorm);
    const fields = info.@"struct".fields;

    var has_offset = false;
    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "weight_offset")) has_offset = true;
    }

    try std.testing.expect(has_offset);
}

// =============================================================================
// Method Existence Tests
// =============================================================================

test "RMSNorm has forward method" {
    try std.testing.expect(@hasDecl(RMSNorm, "forward"));
}

test "RMSNorm has forwardTraced method" {
    try std.testing.expect(@hasDecl(RMSNorm, "forwardTraced"));
}

test "RMSNorm has describe method" {
    try std.testing.expect(@hasDecl(RMSNorm, "describe"));
}

// =============================================================================
// Configuration Tests
// =============================================================================

test "RMSNorm weight_offset default is zero" {
    // Check that the default value is 0.0
    const info = @typeInfo(RMSNorm);
    const fields = info.@"struct".fields;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "weight_offset")) {
            if (field.default_value_ptr) |default_ptr| {
                const default_value: f32 = @as(*const f32, @ptrCast(@alignCast(default_ptr))).*;
                try std.testing.expectApproxEqAbs(@as(f32, 0.0), default_value, 0.001);
            }
        }
    }
}
