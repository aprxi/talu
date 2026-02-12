//! Integration tests for ShapeDim
//!
//! ShapeDim represents a tensor dimension that can be either static (fixed size)
//! or dynamic (varies with sequence length).

const std = @import("std");
const main = @import("main");
const ShapeDim = main.inspect.ShapeDim;

// =============================================================================
// Construction Tests
// =============================================================================

test "ShapeDim.static holds fixed size" {
    const dim = ShapeDim{ .static = 512 };

    switch (dim) {
        .static => |s| try std.testing.expectEqual(@as(usize, 512), s),
        .seq => unreachable,
    }
}

test "ShapeDim.seq represents dynamic dimension" {
    const dim = ShapeDim{ .seq = {} };

    switch (dim) {
        .static => unreachable,
        .seq => {}, // Expected
    }
}

// =============================================================================
// Format Tests
// =============================================================================

test "ShapeDim.formatTo writes static value" {
    const dim = ShapeDim{ .static = 1024 };

    var buf: [32]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try dim.formatTo(stream.writer());

    const output = stream.getWritten();
    try std.testing.expectEqualStrings("1024", output);
}

test "ShapeDim.formatTo writes seq for dynamic" {
    const dim = ShapeDim{ .seq = {} };

    var buf: [32]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try dim.formatTo(stream.writer());

    const output = stream.getWritten();
    try std.testing.expectEqualStrings("seq", output);
}

// =============================================================================
// Usage in Matmul Context Tests
// =============================================================================

test "ShapeDim static in matmul ignores seq_len" {
    const KernelOp = main.inspect.KernelOp;

    const op = KernelOp{ .matmul = .{
        .m = .{ .static = 64 },
        .k = 128,
        .n = 256,
        .dtype = .f32,
        .kernel_name = "test",
    } };

    // Same FLOPs regardless of seq_len
    const flops_10 = op.estimateFlops(10);
    const flops_100 = op.estimateFlops(100);
    try std.testing.expectEqual(flops_10, flops_100);
}

test "ShapeDim seq in matmul scales with seq_len" {
    const KernelOp = main.inspect.KernelOp;

    const op = KernelOp{ .matmul = .{
        .m = .seq,
        .k = 128,
        .n = 256,
        .dtype = .f32,
        .kernel_name = "test",
    } };

    // FLOPs scale linearly with seq_len
    const flops_10 = op.estimateFlops(10);
    const flops_20 = op.estimateFlops(20);
    try std.testing.expectEqual(flops_10 * 2, flops_20);
}

// =============================================================================
// Comparison Tests
// =============================================================================

test "ShapeDim static values can be compared" {
    const dim1 = ShapeDim{ .static = 256 };
    const dim2 = ShapeDim{ .static = 256 };
    const dim3 = ShapeDim{ .static = 512 };

    try std.testing.expectEqual(dim1, dim2);
    try std.testing.expect(!std.meta.eql(dim1, dim3));
}

test "ShapeDim seq values are equal" {
    const dim1 = ShapeDim{ .seq = {} };
    const dim2 = ShapeDim{ .seq = {} };

    try std.testing.expectEqual(dim1, dim2);
}

test "ShapeDim static and seq are not equal" {
    const static_dim = ShapeDim{ .static = 0 };
    const seq_dim = ShapeDim{ .seq = {} };

    try std.testing.expect(!std.meta.eql(static_dim, seq_dim));
}
