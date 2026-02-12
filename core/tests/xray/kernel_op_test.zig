//! Integration tests for xray.KernelOp
//!
//! KernelOp is a tagged union representing different kernel operations
//! (matmul, bias_add, rmsnorm, rope, silu, gelu, mul, add, etc.).
//! Each operation provides FLOPs and memory bandwidth estimates.

const std = @import("std");
const main = @import("main");
const xray = main.xray;

const KernelOp = xray.KernelOp;
const ShapeDim = xray.ShapeDim;

test "KernelOp: matmul FLOPs estimation with static dims" {
    const op = KernelOp{
        .matmul = .{
            .m = .{ .static = 10 },
            .k = 128,
            .n = 256,
            .dtype = .f32,
            .kernel_name = "matmul_f32",
        },
    };

    const flops = op.estimateFlops(1);
    // FLOPs = 2 * M * K * N = 2 * 10 * 128 * 256
    try std.testing.expectEqual(@as(u64, 2 * 10 * 128 * 256), flops);
}

test "KernelOp: matmul FLOPs estimation with seq dims" {
    const op = KernelOp{
        .matmul = .{
            .m = .seq,
            .k = 128,
            .n = 256,
            .dtype = .f32,
            .kernel_name = "matmul_f32",
        },
    };

    const flops = op.estimateFlops(8);
    // FLOPs = 2 * seq_len * K * N = 2 * 8 * 128 * 256
    try std.testing.expectEqual(@as(u64, 2 * 8 * 128 * 256), flops);
}

test "KernelOp: matmul memory estimation" {
    const op = KernelOp{
        .matmul = .{
            .m = .{ .static = 10 },
            .k = 128,
            .n = 256,
            .dtype = .f32,
            .kernel_name = "matmul_f32",
        },
    };

    const mem = op.estimateMemory(1);
    // Read A: 10 * 128 * 4 = 5120
    // Read B: 128 * 256 * 4 = 131072
    // Write C: 10 * 256 * 4 = 10240
    try std.testing.expectEqual(@as(u64, 5120 + 131072 + 10240), mem);
}

test "KernelOp: bias_add" {
    const op = KernelOp{ .bias_add = .{ .size = 1024 } };

    const flops = op.estimateFlops(10);
    // bias_add: size FLOPs (one add per element)
    try std.testing.expectEqual(@as(u64, 1024), flops);

    const mem = op.estimateMemory(10);
    // Read input + bias, write output: size * 4 * 2 (read/write)
    try std.testing.expect(mem > 0);
}

test "KernelOp: rmsnorm" {
    const op = KernelOp{ .rmsnorm = .{ .dim = 768, .eps = 1e-5 } };

    const flops = op.estimateFlops(10);
    try std.testing.expect(flops > 0);
}

test "KernelOp: rope" {
    const op = KernelOp{ .rope = .{ .dim = 128, .theta = 10000.0 } };

    const flops = op.estimateFlops(10);
    try std.testing.expect(flops > 0);
}

test "KernelOp: silu" {
    const op = KernelOp{ .silu = .{ .size = 1000 } };

    const flops = op.estimateFlops(10);
    // silu: size * 4 FLOPs (x * sigmoid(x))
    try std.testing.expectEqual(@as(u64, 1000 * 4), flops);
}

test "KernelOp: gelu" {
    const op = KernelOp{ .gelu = .{ .size = 500 } };

    const flops = op.estimateFlops(10);
    try std.testing.expect(flops > 0);
}

test "KernelOp: mul" {
    const op = KernelOp{ .mul = .{ .size = 200 } };

    const flops = op.estimateFlops(10);
    // mul: size FLOPs
    try std.testing.expectEqual(@as(u64, 200), flops);
}

test "KernelOp: add" {
    const op = KernelOp{ .add = .{ .scale = 1.0, .size = 100 } };

    const flops = op.estimateFlops(10);
    // add: size FLOPs
    try std.testing.expectEqual(@as(u64, 100), flops);
}

test "KernelOp: various operation types compile" {
    // Test that all op types can be constructed
    const ops = [_]KernelOp{
        .{ .bias_add = .{ .size = 1024 } },
        .{ .rmsnorm = .{ .dim = 768, .eps = 1e-5 } },
        .{ .rope = .{ .dim = 128, .theta = 10000.0 } },
        .{ .silu = .{ .size = 1000 } },
        .{ .gelu = .{ .size = 500 } },
        .{ .mul = .{ .size = 200 } },
        .{ .add = .{ .scale = 1.0, .size = 100 } },
    };

    for (ops) |op| {
        const flops = op.estimateFlops(10);
        const mem = op.estimateMemory(10);
        try std.testing.expect(flops >= 0);
        try std.testing.expect(mem >= 0);
    }
}
