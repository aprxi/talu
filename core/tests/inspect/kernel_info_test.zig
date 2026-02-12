//! Integration tests for KernelInfo
//!
//! KernelInfo describes the operations performed by a neural network module.
//! It aggregates KernelOps and provides FLOPs/memory estimation.

const std = @import("std");
const main = @import("main");
const KernelInfo = main.inspect.KernelInfo;
const KernelOp = main.inspect.KernelOp;

// =============================================================================
// Basic Structure Tests
// =============================================================================

test "KernelInfo can be constructed with operations" {
    const ops = [_]KernelOp{
        .{ .rmsnorm = .{ .dim = 512, .eps = 1e-6 } },
        .{ .silu = .{} },
    };

    const info = KernelInfo{
        .name = "TestModule",
        .ops = &ops,
    };

    try std.testing.expectEqualStrings("TestModule", info.name);
    try std.testing.expectEqual(@as(usize, 2), info.ops.len);
}

test "KernelInfo can have input/output shape descriptions" {
    const ops = [_]KernelOp{};

    const info = KernelInfo{
        .name = "Linear",
        .input_shape = "[batch, seq, 512]",
        .output_shape = "[batch, seq, 1024]",
        .ops = &ops,
    };

    try std.testing.expectEqualStrings("[batch, seq, 512]", info.input_shape.?);
    try std.testing.expectEqualStrings("[batch, seq, 1024]", info.output_shape.?);
}

// =============================================================================
// FLOPs Estimation Tests
// =============================================================================

test "KernelInfo.estimateFlops sums operation FLOPs" {
    const ops = [_]KernelOp{
        .{ .rmsnorm = .{ .dim = 512, .eps = 1e-6 } },
        .{ .rmsnorm = .{ .dim = 512, .eps = 1e-6 } },
    };

    const info = KernelInfo{
        .name = "TwoNorms",
        .ops = &ops,
    };

    const seq_len: usize = 10;
    // Each rmsnorm: seq_len * dim * 3 = 10 * 512 * 3 = 15360
    // Two of them: 30720
    const flops = info.estimateFlops(seq_len);
    try std.testing.expectEqual(@as(u64, 30720), flops);
}

test "KernelInfo.estimateFlops returns zero for empty ops" {
    const ops = [_]KernelOp{};

    const info = KernelInfo{
        .name = "Empty",
        .ops = &ops,
    };

    try std.testing.expectEqual(@as(u64, 0), info.estimateFlops(100));
}

// =============================================================================
// Memory Estimation Tests
// =============================================================================

test "KernelInfo.estimateMemory sums operation memory" {
    const ops = [_]KernelOp{
        .{ .rmsnorm = .{ .dim = 256, .eps = 1e-6 } },
    };

    const info = KernelInfo{
        .name = "SingleNorm",
        .ops = &ops,
    };

    const seq_len: usize = 8;
    // rmsnorm memory: seq_len * dim * 4 * 2 = 8 * 256 * 4 * 2 = 16384
    const mem = info.estimateMemory(seq_len);
    try std.testing.expectEqual(@as(u64, 16384), mem);
}

// =============================================================================
// Format Tests
// =============================================================================

test "KernelInfo.format writes operations" {
    const ops = [_]KernelOp{
        .{ .silu = .{} },
    };

    const info = KernelInfo{
        .name = "Activation",
        .ops = &ops,
    };

    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try info.format(stream.writer(), 0);

    const output = stream.getWritten();
    try std.testing.expect(output.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, output, "SiLU") != null);
}

// =============================================================================
// Submodule Tests
// =============================================================================

test "KernelInfo handles submodule references" {
    const inner_ops = [_]KernelOp{
        .{ .rmsnorm = .{ .dim = 128, .eps = 1e-5 } },
    };

    const inner_info = KernelInfo{
        .name = "InnerModule",
        .ops = &inner_ops,
    };

    const outer_ops = [_]KernelOp{
        .{ .submodule = .{ .name = "inner", .info = &inner_info } },
    };

    const outer_info = KernelInfo{
        .name = "OuterModule",
        .ops = &outer_ops,
    };

    // FLOPs should include submodule
    const seq_len: usize = 4;
    const flops = outer_info.estimateFlops(seq_len);
    // submodule -> rmsnorm: 4 * 128 * 3 = 1536
    try std.testing.expectEqual(@as(u64, 1536), flops);
}
