//! Integration tests for KernelOp
//!
//! KernelOp represents a single computational kernel operation.
//! Supports matmul, rmsnorm, rope, sdpa, activations, and more.

const std = @import("std");
const main = @import("main");
const KernelOp = main.inspect.KernelOp;
const ShapeDim = main.inspect.ShapeDim;
const DType = main.DType;

// =============================================================================
// Matmul Operation Tests
// =============================================================================

test "KernelOp.matmul with static dimension" {
    const op = KernelOp{ .matmul = .{
        .m = .{ .static = 32 },
        .k = 512,
        .n = 1024,
        .dtype = .f32,
        .kernel_name = "gemm_f32",
    } };

    // FLOPs = 2 * M * K * N = 2 * 32 * 512 * 1024 = 33554432
    const flops = op.estimateFlops(100); // seq_len ignored for static
    try std.testing.expectEqual(@as(u64, 33554432), flops);
}

test "KernelOp.matmul with seq dimension" {
    const op = KernelOp{ .matmul = .{
        .m = .seq,
        .k = 256,
        .n = 512,
        .dtype = .f16,
        .kernel_name = "gemm_f16",
    } };

    const seq_len: usize = 16;
    // FLOPs = 2 * seq * K * N = 2 * 16 * 256 * 512 = 4194304
    const flops = op.estimateFlops(seq_len);
    try std.testing.expectEqual(@as(u64, 4194304), flops);
}

test "KernelOp.matmul memory estimation" {
    const op = KernelOp{ .matmul = .{
        .m = .{ .static = 8 },
        .k = 64,
        .n = 128,
        .dtype = .f32,
        .kernel_name = "test",
    } };

    const mem = op.estimateMemory(10);
    // Read A[8,64] + B[64,128], Write C[8,128]
    // f32 = 4 bytes: 8*64*4 + 64*128*4 + 8*128*4 = 2048 + 32768 + 4096 = 38912
    try std.testing.expectEqual(@as(u64, 38912), mem);
}

// =============================================================================
// Normalization Operation Tests
// =============================================================================

test "KernelOp.rmsnorm FLOPs estimation" {
    const op = KernelOp{ .rmsnorm = .{
        .dim = 1024,
        .eps = 1e-6,
    } };

    const seq_len: usize = 32;
    // FLOPs = seq_len * dim * 3 = 32 * 1024 * 3 = 98304
    const flops = op.estimateFlops(seq_len);
    try std.testing.expectEqual(@as(u64, 98304), flops);
}

test "KernelOp.rmsnorm memory estimation" {
    const op = KernelOp{ .rmsnorm = .{
        .dim = 512,
        .eps = 1e-5,
    } };

    const seq_len: usize = 8;
    // Memory = seq_len * dim * 4 * 2 (read + write f32)
    const mem = op.estimateMemory(seq_len);
    try std.testing.expectEqual(@as(u64, 32768), mem);
}

// =============================================================================
// RoPE Operation Tests
// =============================================================================

test "KernelOp.rope FLOPs estimation" {
    const op = KernelOp{ .rope = .{
        .dim = 64,
        .theta = 10000.0,
    } };

    const seq_len: usize = 16;
    // FLOPs = seq_len * dim * 4 = 16 * 64 * 4 = 4096
    const flops = op.estimateFlops(seq_len);
    try std.testing.expectEqual(@as(u64, 4096), flops);
}

// =============================================================================
// SDPA Operation Tests
// =============================================================================

test "KernelOp.sdpa FLOPs estimation" {
    const op = KernelOp{ .sdpa = .{
        .n_heads = 8,
        .n_kv_heads = 8,
        .head_dim = 64,
        .scale = 0.125,
        .causal = true,
    } };

    const seq_len: usize = 4;
    // Per head: 2 * seq^2 * head_dim + 5 * seq^2 = 2*16*64 + 5*16 = 2048 + 80 = 2128
    // Total: 2128 * 8 = 17024
    const flops = op.estimateFlops(seq_len);
    try std.testing.expectEqual(@as(u64, 17024), flops);
}

// =============================================================================
// Activation Operation Tests
// =============================================================================

test "KernelOp.silu FLOPs estimation" {
    const op = KernelOp{ .silu = .{ .size = 1024 } };

    // silu: 4 ops per element
    const flops = op.estimateFlops(100); // seq_len not used when size specified
    try std.testing.expectEqual(@as(u64, 4096), flops);
}

test "KernelOp.gelu FLOPs estimation" {
    const op = KernelOp{ .gelu = .{ .size = 512 } };

    // gelu: 8 ops per element
    const flops = op.estimateFlops(100);
    try std.testing.expectEqual(@as(u64, 4096), flops);
}

// =============================================================================
// Element-wise Operation Tests
// =============================================================================

test "KernelOp.add FLOPs estimation" {
    const op = KernelOp{ .add = .{ .scale = 1.0, .size = 256 } };

    // add with scale=1.0: size ops
    const flops = op.estimateFlops(100);
    try std.testing.expectEqual(@as(u64, 256), flops);
}

test "KernelOp.add with scale FLOPs estimation" {
    const op = KernelOp{ .add = .{ .scale = 0.5, .size = 256 } };

    // add with scale!=1.0: size * 2 ops
    const flops = op.estimateFlops(100);
    try std.testing.expectEqual(@as(u64, 512), flops);
}

test "KernelOp.mul FLOPs estimation" {
    const op = KernelOp{ .mul = .{ .size = 128 } };

    const flops = op.estimateFlops(100);
    try std.testing.expectEqual(@as(u64, 128), flops);
}

// =============================================================================
// MoE Routing Tests
// =============================================================================

test "KernelOp.moe_route FLOPs estimation" {
    const op = KernelOp{ .moe_route = .{
        .num_experts = 8,
        .experts_per_token = 2,
        .d_model = 512,
    } };

    // router_flops = 2 * d_model * num_experts = 2 * 512 * 8 = 8192
    // + num_experts * 6 = 48
    // total = 8240
    const flops = op.estimateFlops(100);
    try std.testing.expectEqual(@as(u64, 8240), flops);
}

// =============================================================================
// Gather Operation Tests
// =============================================================================

test "KernelOp.gather returns zero FLOPs" {
    const op = KernelOp{ .gather = .{
        .vocab_size = 32000,
        .embed_dim = 512,
        .dtype = .f16,
    } };

    // Gather is memory-bound, no FLOPs
    const flops = op.estimateFlops(16);
    try std.testing.expectEqual(@as(u64, 0), flops);
}

test "KernelOp.gather memory estimation" {
    const op = KernelOp{ .gather = .{
        .vocab_size = 1000,
        .embed_dim = 256,
        .dtype = .f16,
    } };

    const seq_len: usize = 4;
    // Read: seq * embed_dim * dtype_size, Write: seq * embed_dim * 4 (f32 output)
    // f16 = 2 bytes: 4 * 256 * (2 + 4) = 6144
    const mem = op.estimateMemory(seq_len);
    try std.testing.expectEqual(@as(u64, 6144), mem);
}

// =============================================================================
// Format Tests
// =============================================================================

test "KernelOp.format writes operation description" {
    const op = KernelOp{ .rmsnorm = .{
        .dim = 512,
        .eps = 1e-6,
    } };

    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try op.format(stream.writer(), 0);

    const output = stream.getWritten();
    try std.testing.expect(output.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, output, "RMSNorm") != null);
}
