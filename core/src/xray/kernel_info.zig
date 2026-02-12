//! Kernel Information and Tracing
//!
//! This module provides introspection into the compute kernels used by each
//! neural network module. It enables:
//! - Describing what operations a module performs
//! - Tracing kernel execution at runtime
//! - Understanding the computational graph
//!
//! Design: Each nn module implements kernelInfo() which returns a tree of
//! operations. This can be printed for debugging or used for optimization.

const std = @import("std");
const dtype_mod = @import("../dtype.zig");
const DType = dtype_mod.DType;

/// Represents a single computational kernel operation
pub const KernelOp = union(enum) {
    /// Matrix multiplication: C[m,n] = A[m,k] @ B[k,n]
    matmul: struct {
        m: ShapeDim,
        k: usize,
        n: usize,
        dtype: DType,
        kernel_name: []const u8,
    },

    /// Bias addition: x[..., n] += bias[n]
    bias_add: struct {
        size: usize,
    },

    /// Embedding gather: out[seq, dim] = weight[tokens, dim]
    gather: struct {
        vocab_size: usize,
        embed_dim: usize,
        dtype: DType,
    },

    /// RMS normalization
    rmsnorm: struct {
        dim: usize,
        eps: f32,
    },

    /// Rotary position embedding
    rope: struct {
        dim: usize,
        theta: f32,
    },

    /// Scaled dot-product attention
    sdpa: struct {
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        causal: bool,
    },

    /// SiLU activation: silu(x) = x * sigmoid(x)
    silu: struct {
        size: usize = 0, // Optional size for FLOPs estimation
    },

    /// GELU activation
    gelu: struct {
        size: usize = 0,
    },

    /// Element-wise multiply: a * b
    mul: struct {
        size: usize = 0,
    },

    /// Residual add: x + residual
    add: struct {
        scale: f32,
        size: usize = 0,
    },

    /// Softmax routing for MoE
    moe_route: struct {
        num_experts: usize,
        experts_per_token: usize,
        d_model: usize = 0,
    },

    /// Reference to a submodule's operations
    submodule: struct {
        name: []const u8,
        info: *const KernelInfo,
    },

    /// Estimate FLOPs for this operation given a sequence length.
    /// Returns floating-point operations (multiply-adds count as 2 FLOPs).
    pub fn estimateFlops(self: KernelOp, seq_len: usize) u64 {
        return switch (self) {
            .matmul => |m| blk: {
                const rows: u64 = switch (m.m) {
                    .static => |s| s,
                    .seq => seq_len,
                };
                // FLOPs = 2 * M * K * N (one multiply + one add per output element)
                break :blk 2 * rows * m.k * m.n;
            },
            .bias_add => |b| b.size, // One add per element
            .gather => 0, // Memory-bound, no FLOPs
            .rmsnorm => |r| blk: {
                // Per position: sum squares (dim), divide (1), sqrt (1), multiply (dim)
                // Total: ~3*dim ops per position
                break :blk seq_len * r.dim * 3;
            },
            .rope => |r| blk: {
                // Per position: compute sin/cos (2*dim), apply rotation (2*dim)
                // Approximate as 4*dim FLOPs per position
                break :blk seq_len * r.dim * 4;
            },
            .sdpa => |s| blk: {
                // Q @ K^T: seq * seq * head_dim per head
                // Softmax: ~5 * seq * seq per head
                // Scores @ V: seq * seq * head_dim per head
                // Total per head: 2 * seq^2 * head_dim + 5 * seq^2
                const per_head: u64 = 2 * seq_len * seq_len * s.head_dim + 5 * seq_len * seq_len;
                break :blk per_head * s.n_heads;
            },
            .silu => |s| blk: {
                // silu(x) = x * sigmoid(x)
                // sigmoid: exp + add + div ≈ 3 ops, then multiply = 1
                // Total: 4 ops per element
                const size = if (s.size > 0) s.size else seq_len;
                break :blk size * 4;
            },
            .gelu => |g| blk: {
                // GELU ≈ 8 ops per element (tanh approximation)
                const size = if (g.size > 0) g.size else seq_len;
                break :blk size * 8;
            },
            .mul => |m| if (m.size > 0) m.size else seq_len,
            .add => |r| blk: {
                const size = if (r.size > 0) r.size else seq_len;
                break :blk if (r.scale != 1.0) size * 2 else size;
            },
            .moe_route => |m| blk: {
                // Router matmul: d_model * num_experts
                // Softmax: ~5 * num_experts
                // Top-k selection: ~num_experts
                const router_flops: u64 = if (m.d_model > 0)
                    2 * m.d_model * m.num_experts
                else
                    m.num_experts * 100; // Approximate when d_model unknown
                break :blk router_flops + m.num_experts * 6;
            },
            .submodule => |s| s.info.estimateFlops(seq_len),
        };
    }

    /// Estimate memory bandwidth in bytes for this operation.
    /// Includes both reads and writes.
    pub fn estimateMemory(self: KernelOp, seq_len: usize) u64 {
        return switch (self) {
            .matmul => |m| blk: {
                const rows: u64 = switch (m.m) {
                    .static => |s| s,
                    .seq => seq_len,
                };
                const elem_size: u64 = dtypeSize(m.dtype);
                // Read: A[m,k] + B[k,n], Write: C[m,n]
                const read_a = rows * m.k * elem_size;
                const read_b = m.k * m.n * elem_size;
                const write_c = rows * m.n * 4; // Output is f32
                break :blk read_a + read_b + write_c;
            },
            .bias_add => |b| b.size * 4 * 2, // Read + write f32
            .gather => |g| blk: {
                // Read: seq embeddings from weight table
                // Write: seq * embed_dim output
                break :blk seq_len * g.embed_dim * (dtypeSize(g.dtype) + 4);
            },
            .rmsnorm => |r| seq_len * r.dim * 4 * 2, // Read + write f32
            .rope => |r| seq_len * r.dim * 4 * 2, // Read + write f32 (Q and K)
            .sdpa => |s| blk: {
                // This is complex; approximate as reading Q,K,V and writing O
                const total_dim = s.n_heads * s.head_dim;
                break :blk seq_len * total_dim * 4 * 4; // Q, K, V read + O write
            },
            .silu => |s| blk: {
                const size = if (s.size > 0) s.size else seq_len;
                break :blk size * 4 * 2; // Read + write f32
            },
            .gelu => |g| blk: {
                const size = if (g.size > 0) g.size else seq_len;
                break :blk size * 4 * 2; // Read + write f32
            },
            .mul => |m| blk: {
                const size = if (m.size > 0) m.size else seq_len;
                break :blk size * 4 * 3; // Read 2 inputs + write output
            },
            .add => |r| blk: {
                const size = if (r.size > 0) r.size else seq_len;
                break :blk size * 4 * 3; // Read 2 inputs + write output
            },
            .moe_route => |m| blk: {
                // Router weights + logits
                const weight_bytes = if (m.d_model > 0) m.d_model * m.num_experts * 4 else 0;
                break :blk weight_bytes + m.num_experts * 4;
            },
            .submodule => |s| s.info.estimateMemory(seq_len),
        };
    }

    /// Format a single op for display
    pub fn format(self: KernelOp, writer: anytype, indent: usize) !void {
        try writer.writeByteNTimes(' ', indent);
        try writer.writeAll("└─ ");

        switch (self) {
            .matmul => |matmul_op| {
                try writer.print("{s}(x[", .{matmul_op.kernel_name});
                try matmul_op.m.formatTo(writer);
                try writer.print(", {}], weight[{}, {}], dtype={s}) → [", .{
                    matmul_op.k,
                    matmul_op.n,
                    matmul_op.k,
                    dtypeName(matmul_op.dtype),
                });
                try matmul_op.m.formatTo(writer);
                try writer.print(", {}]", .{matmul_op.n});
            },
            .bias_add => |bias_op| {
                try writer.print("bias_add(size={})", .{bias_op.size});
            },
            .gather => |gather_op| {
                try writer.print("gather(indices, weight[{}, {}], dtype={s})", .{
                    gather_op.vocab_size,
                    gather_op.embed_dim,
                    dtypeName(gather_op.dtype),
                });
            },
            .rmsnorm => |rms_op| {
                try writer.print("rmsnorm(x, weight[{}], eps={e})", .{ rms_op.dim, rms_op.eps });
            },
            .rope => |rope_op| {
                try writer.print("rope(q, k, dim={}, theta={d})", .{ rope_op.dim, rope_op.theta });
            },
            .sdpa => |sdpa_op| {
                try writer.print("sdpa(q, k, v, heads={}, kv_heads={}, head_dim={}, scale={d:.4}, causal={})", .{
                    sdpa_op.n_heads,
                    sdpa_op.n_kv_heads,
                    sdpa_op.head_dim,
                    sdpa_op.scale,
                    sdpa_op.causal,
                });
            },
            .silu => {
                try writer.writeAll("silu(x)");
            },
            .gelu => {
                try writer.writeAll("gelu(x)");
            },
            .mul => {
                try writer.writeAll("mul(a, b)");
            },
            .add => |add_op| {
                if (add_op.scale != 1.0) {
                    try writer.print("add(x, r, scale={d})", .{add_op.scale});
                } else {
                    try writer.writeAll("add(x, r)");
                }
            },
            .moe_route => |moe_op| {
                try writer.print("moe_route(x, num_experts={}, top_k={})", .{
                    moe_op.num_experts,
                    moe_op.experts_per_token,
                });
            },
            .submodule => |submodule| {
                try writer.print("[see {s}]", .{submodule.name});
            },
        }
        try writer.writeAll("\n");
    }
};

/// Dimension that can be static or dynamic (sequence length)
pub const ShapeDim = union(enum) {
    static: usize,
    seq: void, // Dynamic sequence length

    pub fn formatTo(self: ShapeDim, writer: anytype) !void {
        switch (self) {
            .static => |s| try writer.print("{}", .{s}),
            .seq => try writer.writeAll("seq"),
        }
    }
};

/// Information about kernels used by a module
pub const KernelInfo = struct {
    /// Module/operation name
    name: []const u8,
    /// Input shape description
    input_shape: ?[]const u8 = null,
    /// Output shape description
    output_shape: ?[]const u8 = null,
    /// Sequence of operations performed
    ops: []const KernelOp,

    /// Format kernel info with operations
    pub fn format(self: *const KernelInfo, writer: anytype, indent: usize) !void {
        for (self.ops) |op| {
            switch (op) {
                .submodule => |s| {
                    // Recursively format submodule
                    try s.info.format(writer, indent);
                },
                else => {
                    try op.format(writer, indent);
                },
            }
        }
    }

    /// Estimate total FLOPs for all operations given a sequence length
    pub fn estimateFlops(self: *const KernelInfo, seq_len: usize) u64 {
        var total: u64 = 0;
        for (self.ops) |op| {
            total += op.estimateFlops(seq_len);
        }
        return total;
    }

    /// Estimate total memory bandwidth in bytes for all operations
    pub fn estimateMemory(self: *const KernelInfo, seq_len: usize) u64 {
        var total: u64 = 0;
        for (self.ops) |op| {
            total += op.estimateMemory(seq_len);
        }
        return total;
    }
};

// =============================================================================
// Helpers
// =============================================================================

fn dtypeName(dtype: DType) []const u8 {
    return switch (dtype) {
        .f32 => "f32",
        .f16 => "f16",
        .bf16 => "bf16",
        .grouped_affine_u4 => "grouped_affine_u4",
        .grouped_affine_u8 => "grouped_affine_u8",
        else => "unknown",
    };
}

/// Get element size in bytes for a dtype (average for quantized types)
fn dtypeSize(dtype: DType) u64 {
    return switch (dtype) {
        .f32 => 4,
        .f16 => 2,
        .bf16 => 2,
        .grouped_affine_u4 => 1, // 4 bits + scales/biases ≈ 0.5-1 byte effective
        .grouped_affine_u8 => 1,
        else => 4, // Default to f32
    };
}

/// Get kernel name from matmul function pointer (best effort)
pub fn matmulKernelName(dtype: DType) []const u8 {
    return switch (dtype) {
        .f32 => "matmul_f32",
        .f16 => "matmul_f16",
        .bf16 => "matmul_bf16",
        .grouped_affine_u4 => "matmul_grouped_affine_u4",
        .grouped_affine_u8 => "matmul_grouped_affine_u8",
        else => "matmul_unknown",
    };
}

// =============================================================================
// Tests
// =============================================================================

test "kernel info formatting" {
    const info = KernelInfo{
        .name = "linear",
        .ops = &.{
            .{ .matmul = .{
                .m = .seq,
                .k = 1024,
                .n = 4096,
                .dtype = .grouped_affine_u4,
                .kernel_name = "matmul_grouped_affine_u4",
            } },
        },
    };

    var format_buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&format_buf);
    try info.format(stream.writer(), 0);
}

test "KernelOp.estimateFlops - matmul with static dims" {
    const op = KernelOp{
        .matmul = .{
            .m = .{ .static = 10 },
            .k = 128,
            .n = 256,
            .dtype = .f32,
            .kernel_name = "matmul_f32",
        },
    };
    const flops = op.estimateFlops(1); // seq_len ignored for static
    // FLOPs = 2 * M * K * N = 2 * 10 * 128 * 256 = 655360
    try std.testing.expectEqual(@as(u64, 2 * 10 * 128 * 256), flops);
}

test "KernelOp.estimateFlops - matmul with seq dims" {
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
    // FLOPs = 2 * seq_len * K * N = 2 * 8 * 128 * 256 = 524288
    try std.testing.expectEqual(@as(u64, 2 * 8 * 128 * 256), flops);
}

test "KernelOp.estimateFlops - bias_add" {
    const op = KernelOp{ .bias_add = .{ .size = 1024 } };
    const flops = op.estimateFlops(1);
    try std.testing.expectEqual(@as(u64, 1024), flops);
}

test "KernelOp.estimateFlops - gather" {
    const op = KernelOp{
        .gather = .{
            .vocab_size = 50000,
            .embed_dim = 768,
            .dtype = .f32,
        },
    };
    const flops = op.estimateFlops(10);
    try std.testing.expectEqual(@as(u64, 0), flops); // Memory-bound, no FLOPs
}

test "KernelOp.estimateFlops - rmsnorm" {
    const op = KernelOp{ .rmsnorm = .{ .dim = 768, .eps = 1e-5 } };
    const flops = op.estimateFlops(4);
    // ~3 * seq_len * dim = 3 * 4 * 768 = 9216
    try std.testing.expectEqual(@as(u64, 4 * 768 * 3), flops);
}

test "KernelOp.estimateFlops - rope" {
    const op = KernelOp{ .rope = .{ .dim = 128, .theta = 10000.0 } };
    const flops = op.estimateFlops(5);
    // ~4 * seq_len * dim = 4 * 5 * 128 = 2560
    try std.testing.expectEqual(@as(u64, 5 * 128 * 4), flops);
}

test "KernelOp.estimateFlops - sdpa" {
    const op = KernelOp{
        .sdpa = .{
            .n_heads = 12,
            .n_kv_heads = 12,
            .head_dim = 64,
            .scale = 0.125,
            .causal = true,
        },
    };
    const seq_len = 4;
    const flops = op.estimateFlops(seq_len);
    // Per head: 2 * seq^2 * head_dim + 5 * seq^2
    // = 2 * 16 * 64 + 5 * 16 = 2048 + 80 = 2128
    // Total: 2128 * 12 = 25536
    const per_head: u64 = 2 * seq_len * seq_len * 64 + 5 * seq_len * seq_len;
    try std.testing.expectEqual(per_head * 12, flops);
}

test "KernelOp.estimateFlops - silu with size" {
    const op = KernelOp{ .silu = .{ .size = 1000 } };
    const flops = op.estimateFlops(10);
    // 4 ops per element, size specified
    try std.testing.expectEqual(@as(u64, 1000 * 4), flops);
}

test "KernelOp.estimateFlops - silu with seq_len" {
    const op = KernelOp{ .silu = .{ .size = 0 } };
    const flops = op.estimateFlops(10);
    // 4 ops per element, use seq_len when size is 0
    try std.testing.expectEqual(@as(u64, 10 * 4), flops);
}

test "KernelOp.estimateFlops - gelu with size" {
    const op = KernelOp{ .gelu = .{ .size = 500 } };
    const flops = op.estimateFlops(10);
    // 8 ops per element
    try std.testing.expectEqual(@as(u64, 500 * 8), flops);
}

test "KernelOp.estimateFlops - gelu with seq_len" {
    const op = KernelOp{ .gelu = .{ .size = 0 } };
    const flops = op.estimateFlops(10);
    try std.testing.expectEqual(@as(u64, 10 * 8), flops);
}

test "KernelOp.estimateFlops - mul" {
    const op = KernelOp{ .mul = .{ .size = 200 } };
    const flops = op.estimateFlops(10);
    try std.testing.expectEqual(@as(u64, 200), flops);
}

test "KernelOp.estimateFlops - add with scale" {
    const op = KernelOp{ .add = .{ .scale = 2.0, .size = 100 } };
    const flops = op.estimateFlops(10);
    // scale != 1.0 means 2 ops per element
    try std.testing.expectEqual(@as(u64, 100 * 2), flops);
}

test "KernelOp.estimateFlops - add without scale" {
    const op = KernelOp{ .add = .{ .scale = 1.0, .size = 100 } };
    const flops = op.estimateFlops(10);
    // scale == 1.0 means 1 op per element
    try std.testing.expectEqual(@as(u64, 100), flops);
}

test "KernelOp.estimateFlops - moe_route with d_model" {
    const op = KernelOp{
        .moe_route = .{
            .num_experts = 8,
            .experts_per_token = 2,
            .d_model = 512,
        },
    };
    const flops = op.estimateFlops(1);
    // Router matmul: 2 * d_model * num_experts + num_experts * 6
    // = 2 * 512 * 8 + 8 * 6 = 8192 + 48 = 8240
    try std.testing.expectEqual(@as(u64, 2 * 512 * 8 + 8 * 6), flops);
}

test "KernelOp.estimateFlops - moe_route without d_model" {
    const op = KernelOp{
        .moe_route = .{
            .num_experts = 8,
            .experts_per_token = 2,
            .d_model = 0,
        },
    };
    const flops = op.estimateFlops(1);
    // Approximate when d_model unknown: num_experts * 100 + num_experts * 6
    // = 8 * 100 + 8 * 6 = 800 + 48 = 848
    try std.testing.expectEqual(@as(u64, 8 * 100 + 8 * 6), flops);
}

test "KernelOp.estimateFlops - submodule" {
    const child_info = KernelInfo{
        .name = "child",
        .ops = &.{
            .{ .bias_add = .{ .size = 100 } },
            .{ .silu = .{ .size = 100 } },
        },
    };
    const op = KernelOp{
        .submodule = .{
            .name = "child",
            .info = &child_info,
        },
    };
    const flops = op.estimateFlops(1);
    // Sum of child ops: 100 + 400 = 500
    try std.testing.expectEqual(@as(u64, 500), flops);
}

test "KernelOp.estimateMemory - matmul with static dims" {
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
    // Total: 146432
    try std.testing.expectEqual(@as(u64, 10 * 128 * 4 + 128 * 256 * 4 + 10 * 256 * 4), mem);
}

test "KernelOp.estimateMemory - matmul with seq dims and quantized dtype" {
    const op = KernelOp{
        .matmul = .{
            .m = .seq,
            .k = 128,
            .n = 256,
            .dtype = .grouped_affine_u4,
            .kernel_name = "matmul_grouped_affine_u4",
        },
    };
    const mem = op.estimateMemory(8);
    // Read A: 8 * 128 * 1 = 1024
    // Read B: 128 * 256 * 1 = 32768
    // Write C: 8 * 256 * 4 = 8192 (output is f32)
    // Total: 41984
    try std.testing.expectEqual(@as(u64, 8 * 128 * 1 + 128 * 256 * 1 + 8 * 256 * 4), mem);
}

test "KernelOp.estimateMemory - bias_add" {
    const op = KernelOp{ .bias_add = .{ .size = 1024 } };
    const mem = op.estimateMemory(1);
    // Read + write: 1024 * 4 * 2 = 8192
    try std.testing.expectEqual(@as(u64, 1024 * 4 * 2), mem);
}

test "KernelOp.estimateMemory - gather" {
    const op = KernelOp{
        .gather = .{
            .vocab_size = 50000,
            .embed_dim = 768,
            .dtype = .f16,
        },
    };
    const mem = op.estimateMemory(4);
    // seq * embed_dim * (dtype_size + 4)
    // 4 * 768 * (2 + 4) = 18432
    try std.testing.expectEqual(@as(u64, 4 * 768 * (2 + 4)), mem);
}

test "KernelOp.estimateMemory - rmsnorm" {
    const op = KernelOp{ .rmsnorm = .{ .dim = 768, .eps = 1e-5 } };
    const mem = op.estimateMemory(4);
    // seq * dim * 4 * 2 = 4 * 768 * 8 = 24576
    try std.testing.expectEqual(@as(u64, 4 * 768 * 4 * 2), mem);
}

test "KernelOp.estimateMemory - rope" {
    const op = KernelOp{ .rope = .{ .dim = 128, .theta = 10000.0 } };
    const mem = op.estimateMemory(5);
    // seq * dim * 4 * 2 = 5 * 128 * 8 = 5120
    try std.testing.expectEqual(@as(u64, 5 * 128 * 4 * 2), mem);
}

test "KernelOp.estimateMemory - sdpa" {
    const op = KernelOp{
        .sdpa = .{
            .n_heads = 12,
            .n_kv_heads = 12,
            .head_dim = 64,
            .scale = 0.125,
            .causal = true,
        },
    };
    const seq_len = 4;
    const mem = op.estimateMemory(seq_len);
    // total_dim * seq * 4 * 4 (Q, K, V read + O write)
    const total_dim: u64 = 12 * 64;
    try std.testing.expectEqual(seq_len * total_dim * 4 * 4, mem);
}

test "KernelOp.estimateMemory - silu" {
    const op = KernelOp{ .silu = .{ .size = 1000 } };
    const mem = op.estimateMemory(10);
    // size * 4 * 2 = 1000 * 8 = 8000
    try std.testing.expectEqual(@as(u64, 1000 * 4 * 2), mem);
}

test "KernelOp.estimateMemory - gelu" {
    const op = KernelOp{ .gelu = .{ .size = 500 } };
    const mem = op.estimateMemory(10);
    try std.testing.expectEqual(@as(u64, 500 * 4 * 2), mem);
}

test "KernelOp.estimateMemory - mul" {
    const op = KernelOp{ .mul = .{ .size = 200 } };
    const mem = op.estimateMemory(10);
    // Read 2 inputs + write: 200 * 4 * 3 = 2400
    try std.testing.expectEqual(@as(u64, 200 * 4 * 3), mem);
}

test "KernelOp.estimateMemory - add" {
    const op = KernelOp{ .add = .{ .scale = 2.0, .size = 100 } };
    const mem = op.estimateMemory(10);
    // Read 2 inputs + write: 100 * 4 * 3 = 1200
    try std.testing.expectEqual(@as(u64, 100 * 4 * 3), mem);
}

test "KernelOp.estimateMemory - moe_route with d_model" {
    const op = KernelOp{
        .moe_route = .{
            .num_experts = 8,
            .experts_per_token = 2,
            .d_model = 512,
        },
    };
    const mem = op.estimateMemory(1);
    // weight_bytes + logits: 512 * 8 * 4 + 8 * 4 = 16384 + 32 = 16416
    try std.testing.expectEqual(@as(u64, 512 * 8 * 4 + 8 * 4), mem);
}

test "KernelOp.estimateMemory - moe_route without d_model" {
    const op = KernelOp{
        .moe_route = .{
            .num_experts = 8,
            .experts_per_token = 2,
            .d_model = 0,
        },
    };
    const mem = op.estimateMemory(1);
    // Only logits when d_model is 0: 8 * 4 = 32
    try std.testing.expectEqual(@as(u64, 8 * 4), mem);
}

test "KernelOp.estimateMemory - submodule" {
    const child_info = KernelInfo{
        .name = "child",
        .ops = &.{
            .{ .bias_add = .{ .size = 100 } },
            .{ .mul = .{ .size = 100 } },
        },
    };
    const op = KernelOp{
        .submodule = .{
            .name = "child",
            .info = &child_info,
        },
    };
    const mem = op.estimateMemory(1);
    // Sum of child ops: (100 * 4 * 2) + (100 * 4 * 3) = 800 + 1200 = 2000
    try std.testing.expectEqual(@as(u64, 2000), mem);
}

test "ShapeDim.formatTo - static" {
    const dim: ShapeDim = .{ .static = 42 };
    var buf: [16]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    try dim.formatTo(stream.writer());
    try std.testing.expectEqualStrings("42", stream.getWritten());
}

test "ShapeDim.formatTo - seq" {
    const dim: ShapeDim = .seq;
    var buf: [16]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    try dim.formatTo(stream.writer());
    try std.testing.expectEqualStrings("seq", stream.getWritten());
}

test "KernelInfo.estimateFlops - multiple ops" {
    const info = KernelInfo{
        .name = "mlp",
        .ops = &.{
            .{ .matmul = .{ .m = .seq, .k = 128, .n = 512, .dtype = .f32, .kernel_name = "matmul_f32" } },
            .{ .bias_add = .{ .size = 512 } },
            .{ .silu = .{ .size = 512 } },
        },
    };
    const flops = info.estimateFlops(4);
    // matmul: 2 * 4 * 128 * 512 = 524288
    // bias_add: 512
    // silu: 512 * 4 = 2048
    // Total: 526848
    try std.testing.expectEqual(@as(u64, 2 * 4 * 128 * 512 + 512 + 512 * 4), flops);
}

test "KernelInfo.estimateMemory - multiple ops" {
    const info = KernelInfo{
        .name = "mlp",
        .ops = &.{
            .{ .bias_add = .{ .size = 100 } },
            .{ .mul = .{ .size = 100 } },
            .{ .add = .{ .scale = 1.0, .size = 100 } },
        },
    };
    const mem = info.estimateMemory(1);
    // bias_add: 100 * 4 * 2 = 800
    // mul: 100 * 4 * 3 = 1200
    // add: 100 * 4 * 3 = 1200
    // Total: 3200
    try std.testing.expectEqual(@as(u64, 800 + 1200 + 1200), mem);
}

test "matmulKernelName - f32" {
    const name = matmulKernelName(.f32);
    try std.testing.expectEqualStrings("matmul_f32", name);
}

test "matmulKernelName - f16" {
    const name = matmulKernelName(.f16);
    try std.testing.expectEqualStrings("matmul_f16", name);
}

test "matmulKernelName - bf16" {
    const name = matmulKernelName(.bf16);
    try std.testing.expectEqualStrings("matmul_bf16", name);
}

test "matmulKernelName - grouped_affine_u4" {
    const name = matmulKernelName(.grouped_affine_u4);
    try std.testing.expectEqualStrings("matmul_grouped_affine_u4", name);
}

test "matmulKernelName - grouped_affine_u8" {
    const name = matmulKernelName(.grouped_affine_u8);
    try std.testing.expectEqualStrings("matmul_grouped_affine_u8", name);
}

test "matmulKernelName - unknown dtype" {
    const name = matmulKernelName(.u8);
    try std.testing.expectEqualStrings("matmul_unknown", name);
}
