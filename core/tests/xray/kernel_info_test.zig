//! Integration tests for xray.KernelInfo
//!
//! KernelInfo describes a kernel's operations for performance analysis.
//! Contains a name and array of KernelOp. Provides estimateFlops and
//! estimateMemory methods that aggregate across all ops.

const std = @import("std");
const main = @import("main");
const xray = main.xray;

const KernelInfo = xray.KernelInfo;
const KernelOp = xray.KernelOp;

test "KernelInfo: total FLOPs estimation" {
    const info = KernelInfo{
        .name = "mlp",
        .ops = &.{
            .{ .matmul = .{ .m = .seq, .k = 128, .n = 512, .dtype = .f32, .kernel_name = "matmul_f32" } },
            .{ .bias_add = .{ .size = 512 } },
            .{ .silu = .{ .size = 512 } },
        },
    };

    const seq_len = 4;
    const flops = info.estimateFlops(seq_len);

    // matmul: 2 * 4 * 128 * 512 = 524288
    // bias_add: 512
    // silu: 512 * 4 = 2048
    const expected = 2 * seq_len * 128 * 512 + 512 + 512 * 4;
    try std.testing.expectEqual(@as(u64, expected), flops);
}

test "KernelInfo: total memory estimation" {
    const info = KernelInfo{
        .name = "block",
        .ops = &.{
            .{ .bias_add = .{ .size = 100 } },
            .{ .mul = .{ .size = 100 } },
        },
    };

    const mem = info.estimateMemory(1);
    // bias_add: 100 * 4 * 2 = 800
    // mul: 100 * 4 * 3 = 1200
    try std.testing.expectEqual(@as(u64, 800 + 1200), mem);
}

test "KernelInfo: formatting works" {
    const info = KernelInfo{
        .name = "linear",
        .ops = &.{
            .{ .matmul = .{
                .m = .seq,
                .k = 1024,
                .n = 4096,
                .dtype = .f32,
                .kernel_name = "matmul_f32",
            } },
        },
    };

    var buf: [512]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    try info.format(stream.writer(), 0);

    // Should have written something
    try std.testing.expect(stream.getWritten().len > 0);
}

test "KernelInfo: empty ops array" {
    const info = KernelInfo{
        .name = "empty",
        .ops = &.{},
    };

    try std.testing.expectEqual(@as(u64, 0), info.estimateFlops(10));
    try std.testing.expectEqual(@as(u64, 0), info.estimateMemory(10));
}

test "KernelInfo: name is accessible" {
    const info = KernelInfo{
        .name = "test_kernel",
        .ops = &.{},
    };

    try std.testing.expectEqualStrings("test_kernel", info.name);
}
