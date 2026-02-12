//! Integration tests for inference.SwiGLU
//!
//! Tests the SwiGLU struct which implements the SwiGLU activation function layer.
//! Note: Full forward pass tests require loaded weights.

const std = @import("std");
const main = @import("main");

const SwiGLU = main.inference.backend.SwiGLU;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "SwiGLU type is accessible" {
    const T = SwiGLU;
    _ = T;
}

test "SwiGLU has expected structure" {
    const info = @typeInfo(SwiGLU);
    try std.testing.expect(info == .@"struct");
}
