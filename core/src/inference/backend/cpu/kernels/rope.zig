//! CPU Rotary Position Embedding (RoPE) Kernel
//!
//! Re-exports RoPE from compute/ops/math_primitives.zig for organization.
//! The actual implementation lives in ops/math_primitives.zig alongside other math operations.

const std = @import("std");
const compute = @import("../../../../compute/root.zig");

// Re-export RoPE from ops/math_primitives.zig
pub const RoPE = compute.ops.math.RoPE;

pub const RotaryEmbedding = struct {
    /// Canonical kernel-call contract for backend parity checks.
    pub const ForwardParams = struct {
        input_vector: []const f32,
        output_vector: []f32,
        position: usize,
    };

    rope: *RoPE,

    pub fn forward(
        self: *RotaryEmbedding,
        input_vector: []const f32,
        output_vector: []f32,
        position: usize,
    ) void {
        std.debug.assert(output_vector.len == input_vector.len);
        @memcpy(output_vector, input_vector);
        self.rope.applyInPlace(output_vector, position);
    }
};

test "RotaryEmbedding.forward at position 0 preserves vector" {
    const allocator = std.testing.allocator;
    var rope = try RoPE.init(allocator, 4, 16, 10_000.0, 1.0);
    defer rope.deinit(allocator);

    var kernel = RotaryEmbedding{ .rope = &rope };
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output: [4]f32 = undefined;
    kernel.forward(&input, &output, 0);
    try std.testing.expectEqualSlices(f32, &input, &output);
}
