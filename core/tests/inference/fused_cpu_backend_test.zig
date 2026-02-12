//! Integration tests for inference.FusedCpuBackend
//!
//! Tests the FusedCpuBackend struct which provides fused CPU execution.
//! Note: Full inference tests require a loaded model.

const std = @import("std");
const main = @import("main");

const FusedCpuBackend = main.inference.backend.FusedCpuBackend;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "FusedCpuBackend type is accessible" {
    const T = FusedCpuBackend;
    _ = T;
}

test "FusedCpuBackend has expected structure" {
    const info = @typeInfo(FusedCpuBackend);
    try std.testing.expect(info == .@"struct");
}

test "FusedCpuBackend has expected fields" {
    const info = @typeInfo(FusedCpuBackend);
    const fields = info.@"struct".fields;

    var has_vocab_size = false;
    var has_max_batch_size = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "vocab_size")) has_vocab_size = true;
        if (comptime std.mem.eql(u8, field.name, "max_batch_size")) has_max_batch_size = true;
    }

    try std.testing.expect(has_vocab_size);
    try std.testing.expect(has_max_batch_size);
}
