//! Performance Estimation
//!
//! Estimates FLOPs and memory bandwidth for transformer models.
//! Independent of runtime types - takes primitive configuration values.

const std = @import("std");
const dtype_mod = @import("../dtype.zig");

const DType = dtype_mod.DType;

pub const PerfEstimate = struct {
    prefill_flops: u64,
    per_token_flops: u64,
    prefill_mem_bytes: u64,
    per_token_mem_bytes: u64,
    seq_len: usize,
    weight_dtype: DType,

    pub fn format(self: PerfEstimate, writer: anytype) !void {
        try writer.print("Performance estimate (seq_len={}):\n", .{self.seq_len});

        try writer.writeAll("\n  FLOPs:\n");
        try writer.writeAll("    Prefill:   ");
        try formatFlops(writer, self.prefill_flops);
        try writer.writeAll("\n");
        try writer.writeAll("    Per-token: ");
        try formatFlops(writer, self.per_token_flops);
        try writer.writeAll("\n");

        try writer.writeAll("\n  Memory bandwidth:\n");
        try writer.writeAll("    Prefill:   ");
        try formatBytes(writer, self.prefill_mem_bytes);
        try writer.writeAll("\n");
        try writer.writeAll("    Per-token: ");
        try formatBytes(writer, self.per_token_mem_bytes);
        try writer.writeAll("\n");

        const prefill_intensity = @as(f64, @floatFromInt(self.prefill_flops)) / @as(f64, @floatFromInt(self.prefill_mem_bytes));
        const per_token_intensity = @as(f64, @floatFromInt(self.per_token_flops)) / @as(f64, @floatFromInt(self.per_token_mem_bytes));
        try writer.writeAll("\n  Arithmetic intensity (FLOP/byte):\n");
        try writer.print("    Prefill:   {d:.1}\n", .{prefill_intensity});
        try writer.print("    Per-token: {d:.1}\n", .{per_token_intensity});

        try writer.writeAll("\n  Theoretical decode tok/s:\n");
        const profiles = [_]struct { name: []const u8, tflops: f64, mem_gbps: f64 }{
            .{ .name = "CPU (AVX2)", .tflops = 0.5, .mem_gbps = 50 },
            .{ .name = "M1 Pro", .tflops = 5.2, .mem_gbps = 200 },
            .{ .name = "M2 Ultra", .tflops = 27.2, .mem_gbps = 800 },
            .{ .name = "RTX 4090", .tflops = 82.6, .mem_gbps = 1008 },
            .{ .name = "H100 SXM", .tflops = 989, .mem_gbps = 3350 },
        };

        for (profiles) |profile| {
            const compute_limit = profile.tflops * 1e12 / @as(f64, @floatFromInt(self.per_token_flops));
            const mem_limit = profile.mem_gbps * 1e9 / @as(f64, @floatFromInt(self.per_token_mem_bytes));
            const actual = @min(compute_limit, mem_limit);
            const bottleneck: []const u8 = if (compute_limit < mem_limit) "compute" else "memory";
            try writer.print("    {s}: {d:.1} tok/s ({s}-bound)\n", .{ profile.name, actual, bottleneck });
        }
    }
};

/// Attention configuration for performance estimation.
/// Decoupled from runtime types - pass primitive values.
pub const AttnConfig = struct {
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    d_model: usize,
    has_q_bias: bool = false,
    has_k_bias: bool = false,
    has_v_bias: bool = false,
    has_o_bias: bool = false,
};

/// FFN configuration for performance estimation.
/// Decoupled from runtime types - pass primitive values.
pub const FfnConfig = union(enum) {
    swiglu: struct {
        d_model: usize,
        d_ff: usize,
    },
    moe_ffn: struct {
        d_model: usize,
        d_ff: usize,
        num_experts: usize,
        experts_per_token: usize,
    },
};

pub const EstimateArgs = struct {
    seq_len: usize,
    weight_dtype: DType,
    hidden_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    attn: AttnConfig,
    ffn: FfnConfig,
};

pub const LayerGeometry = struct {
    q_dim: usize,
    kv_dim: usize,
    qkv_proj_weights: usize,

    ffn_weights: usize,
    attn_bias_params: usize,
    router_weights: usize,
    expert_weights: usize,
    total_layer_params: usize,

    pub fn init(attn: AttnConfig, ffn: FfnConfig) LayerGeometry {
        const q_dim = attn.n_heads * attn.head_dim;
        const kv_dim = attn.n_kv_heads * attn.head_dim;

        const q_proj_weights = attn.d_model * q_dim;
        const k_proj_weights = attn.d_model * kv_dim;
        const v_proj_weights = attn.d_model * kv_dim;
        const o_proj_weights = q_dim * attn.d_model;
        const qkv_proj_weights = q_proj_weights + k_proj_weights + v_proj_weights + o_proj_weights;
        const attn_bias_params: usize =
            (if (attn.has_q_bias) q_dim else 0) +
            (if (attn.has_k_bias) kv_dim else 0) +
            (if (attn.has_v_bias) kv_dim else 0) +
            (if (attn.has_o_bias) attn.d_model else 0);

        var ffn_weights: usize = 0;
        var router_weights: usize = 0;
        var expert_weights: usize = 0;

        switch (ffn) {
            .swiglu => |mlp| {
                ffn_weights = mlp.d_model * mlp.d_ff * 3;
            },
            .moe_ffn => |moe| {
                router_weights = moe.d_model * moe.num_experts;
                expert_weights = moe.num_experts * moe.d_model * moe.d_ff * 3;
                ffn_weights = router_weights + expert_weights;
            },
        }

        return .{
            .q_dim = q_dim,
            .kv_dim = kv_dim,
            .qkv_proj_weights = qkv_proj_weights,
            .ffn_weights = ffn_weights,
            .attn_bias_params = attn_bias_params,
            .router_weights = router_weights,
            .expert_weights = expert_weights,
            .total_layer_params = qkv_proj_weights + attn_bias_params + ffn_weights,
        };
    }
};

pub fn estimatePerf(args: EstimateArgs) PerfEstimate {
    var prefill_flops: u64 = 0;
    var decode_flops: u64 = 0;
    var prefill_mem: u64 = 0;
    var decode_mem: u64 = 0;

    const weight_bytes_times_2: u64 = switch (args.weight_dtype) {
        .grouped_affine_u4 => 1, // 0.5 bytes (4 bits per weight)
        .grouped_affine_u8 => 2, // 1 byte
        .f16, .bf16 => 4, // 2 bytes
        .f32 => 8, // 4 bytes
        else => 4, // Default to f16
    };

    prefill_mem += args.seq_len * args.hidden_size * weight_bytes_times_2;
    decode_mem += args.hidden_size * weight_bytes_times_2;

    const geom = LayerGeometry.init(args.attn, args.ffn);
    const n_heads = args.attn.n_heads;
    const head_dim = args.attn.head_dim;
    const n_kv_heads = args.attn.n_kv_heads;

    var layer_prefill_flops: u64 = 0;
    var layer_decode_flops: u64 = 0;
    var layer_weight_bytes_times_2: u64 = 0;

    // Weights (quantized; preserve prior behavior: biases are not included here)
    layer_weight_bytes_times_2 += @as(u64, @intCast(geom.qkv_proj_weights + geom.ffn_weights)) * weight_bytes_times_2;

    // Projections FLOPs: 2 * seq * (Q + K + V + O)
    layer_prefill_flops += 2 * args.seq_len * @as(u64, @intCast(geom.qkv_proj_weights));
    layer_decode_flops += 2 * @as(u64, @intCast(geom.qkv_proj_weights));

    // SDPA: Q @ K^T + scores @ V
    layer_prefill_flops += 2 * args.seq_len * args.seq_len * head_dim * n_heads;
    layer_decode_flops += 2 * args.seq_len * head_dim * n_heads;

    const kv_cache_per_layer = args.seq_len * n_kv_heads * head_dim * 4 * 2; // f32, K+V

    // FFN FLOPs
    switch (args.ffn) {
        .swiglu => {
            layer_prefill_flops += 2 * args.seq_len * @as(u64, @intCast(geom.ffn_weights));
            layer_decode_flops += 2 * @as(u64, @intCast(geom.ffn_weights));
        },
        .moe_ffn => |moe_layer| {
            // Only count active experts for compute.
            layer_prefill_flops += 2 * args.seq_len * @as(u64, @intCast(geom.router_weights));
            layer_prefill_flops += 2 * args.seq_len * moe_layer.d_model * moe_layer.d_ff * 3 * moe_layer.experts_per_token;

            layer_decode_flops += 2 * @as(u64, @intCast(geom.router_weights));
            layer_decode_flops += 2 * moe_layer.d_model * moe_layer.d_ff * 3 * moe_layer.experts_per_token;
        },
    }

    prefill_flops += layer_prefill_flops * args.num_hidden_layers;
    decode_flops += layer_decode_flops * args.num_hidden_layers;

    prefill_mem += layer_weight_bytes_times_2 * args.num_hidden_layers;
    decode_mem += layer_weight_bytes_times_2 * args.num_hidden_layers;
    decode_mem += kv_cache_per_layer * args.num_hidden_layers * 2; // Convert to x2 units

    // LM head
    const lm_head_size = args.hidden_size * args.vocab_size;
    const lm_head_bytes_times_2 = lm_head_size * weight_bytes_times_2;
    prefill_flops += 2 * lm_head_size;
    decode_flops += 2 * lm_head_size;
    prefill_mem += lm_head_bytes_times_2;
    decode_mem += lm_head_bytes_times_2;

    return .{
        .prefill_flops = prefill_flops,
        .per_token_flops = decode_flops,
        .prefill_mem_bytes = prefill_mem / 2,
        .per_token_mem_bytes = decode_mem / 2,
        .seq_len = args.seq_len,
        .weight_dtype = args.weight_dtype,
    };
}

fn formatFlops(writer: anytype, flops: u64) !void {
    const flops_value = @as(f64, @floatFromInt(flops));
    if (flops_value >= 1e15) {
        try writer.print("{d:.2} PFLOP", .{flops_value / 1e15});
    } else if (flops_value >= 1e12) {
        try writer.print("{d:.2} TFLOP", .{flops_value / 1e12});
    } else if (flops_value >= 1e9) {
        try writer.print("{d:.2} GFLOP", .{flops_value / 1e9});
    } else if (flops_value >= 1e6) {
        try writer.print("{d:.2} MFLOP", .{flops_value / 1e6});
    } else {
        try writer.print("{} FLOP", .{flops});
    }
}

pub fn formatBytes(writer: anytype, bytes: u64) !void {
    const bytes_value = @as(f64, @floatFromInt(bytes));
    if (bytes_value >= 1e12) {
        try writer.print("{d:.2} TB", .{bytes_value / 1e12});
    } else if (bytes_value >= 1e9) {
        try writer.print("{d:.2} GB", .{bytes_value / 1e9});
    } else if (bytes_value >= 1e6) {
        try writer.print("{d:.2} MB", .{bytes_value / 1e6});
    } else if (bytes_value >= 1e3) {
        try writer.print("{d:.2} KB", .{bytes_value / 1e3});
    } else {
        try writer.print("{} B", .{bytes});
    }
}

// ============================================================================
// Tests
// ============================================================================

test "LayerGeometry.init - swiglu standard config" {
    const attn = AttnConfig{
        .n_heads = 32,
        .n_kv_heads = 8,
        .head_dim = 128,
        .d_model = 4096,
    };
    const ffn = FfnConfig{ .swiglu = .{
        .d_model = 4096,
        .d_ff = 14336,
    } };

    const geom = LayerGeometry.init(attn, ffn);

    // q_dim = n_heads * head_dim = 32 * 128 = 4096
    try std.testing.expectEqual(@as(usize, 4096), geom.q_dim);
    // kv_dim = n_kv_heads * head_dim = 8 * 128 = 1024
    try std.testing.expectEqual(@as(usize, 1024), geom.kv_dim);

    // QKV projections: q_proj (4096*4096) + k_proj (4096*1024) + v_proj (4096*1024) + o_proj (4096*4096)
    const expected_qkv = 4096 * 4096 + 4096 * 1024 + 4096 * 1024 + 4096 * 4096;
    try std.testing.expectEqual(expected_qkv, geom.qkv_proj_weights);

    // FFN: d_model * d_ff * 3 = 4096 * 14336 * 3
    const expected_ffn = 4096 * 14336 * 3;
    try std.testing.expectEqual(expected_ffn, geom.ffn_weights);

    // No biases
    try std.testing.expectEqual(@as(usize, 0), geom.attn_bias_params);
    try std.testing.expectEqual(@as(usize, 0), geom.router_weights);
    try std.testing.expectEqual(@as(usize, 0), geom.expert_weights);

    // Total = qkv + ffn
    try std.testing.expectEqual(expected_qkv + expected_ffn, geom.total_layer_params);
}

test "LayerGeometry.init - with biases" {
    const attn = AttnConfig{
        .n_heads = 16,
        .n_kv_heads = 4,
        .head_dim = 64,
        .d_model = 1024,
        .has_q_bias = true,
        .has_k_bias = true,
        .has_v_bias = true,
        .has_o_bias = true,
    };
    const ffn = FfnConfig{ .swiglu = .{
        .d_model = 1024,
        .d_ff = 4096,
    } };

    const geom = LayerGeometry.init(attn, ffn);

    const q_dim = 16 * 64; // 1024
    const kv_dim = 4 * 64; // 256

    // q_bias + k_bias + v_bias + o_bias
    const expected_bias = q_dim + kv_dim + kv_dim + 1024;
    try std.testing.expectEqual(expected_bias, geom.attn_bias_params);
}

test "LayerGeometry.init - moe_ffn config" {
    const attn = AttnConfig{
        .n_heads = 8,
        .n_kv_heads = 8,
        .head_dim = 128,
        .d_model = 1024,
    };
    const ffn = FfnConfig{ .moe_ffn = .{
        .d_model = 1024,
        .d_ff = 2816,
        .num_experts = 8,
        .experts_per_token = 2,
    } };

    const geom = LayerGeometry.init(attn, ffn);

    // Router: d_model * num_experts = 1024 * 8
    const expected_router = 1024 * 8;
    try std.testing.expectEqual(expected_router, geom.router_weights);

    // Expert weights: num_experts * d_model * d_ff * 3
    const expected_expert = 8 * 1024 * 2816 * 3;
    try std.testing.expectEqual(expected_expert, geom.expert_weights);

    // FFN = router + expert
    try std.testing.expectEqual(expected_router + expected_expert, geom.ffn_weights);
}

test "estimatePerf - basic swiglu model" {
    const args = EstimateArgs{
        .seq_len = 512,
        .weight_dtype = .f16,
        .hidden_size = 2048,
        .vocab_size = 32000,
        .num_hidden_layers = 24,
        .attn = .{
            .n_heads = 32,
            .n_kv_heads = 8,
            .head_dim = 64,
            .d_model = 2048,
        },
        .ffn = .{ .swiglu = .{
            .d_model = 2048,
            .d_ff = 8192,
        } },
    };

    const result = estimatePerf(args);

    // Basic sanity checks
    try std.testing.expectEqual(@as(usize, 512), result.seq_len);
    try std.testing.expectEqual(DType.f16, result.weight_dtype);

    // Prefill should have higher FLOPs than per-token (seq_len^2 vs seq_len scaling)
    try std.testing.expect(result.prefill_flops > result.per_token_flops);

    // Memory should be non-zero
    try std.testing.expect(result.prefill_mem_bytes > 0);
    try std.testing.expect(result.per_token_mem_bytes > 0);

    // Per-token memory should include KV cache
    try std.testing.expect(result.per_token_mem_bytes > result.prefill_mem_bytes);
}

test "estimatePerf - different dtypes" {
    const base_args = EstimateArgs{
        .seq_len = 128,
        .weight_dtype = .f32,
        .hidden_size = 512,
        .vocab_size = 10000,
        .num_hidden_layers = 6,
        .attn = .{
            .n_heads = 8,
            .n_kv_heads = 8,
            .head_dim = 64,
            .d_model = 512,
        },
        .ffn = .{ .swiglu = .{
            .d_model = 512,
            .d_ff = 2048,
        } },
    };

    // Test different dtypes - FLOPs should be the same, memory should differ
    var args_f32 = base_args;
    args_f32.weight_dtype = .f32;
    const result_f32 = estimatePerf(args_f32);

    var args_f16 = base_args;
    args_f16.weight_dtype = .f16;
    const result_f16 = estimatePerf(args_f16);

    var args_u8 = base_args;
    args_u8.weight_dtype = .grouped_affine_u8;
    const result_u8 = estimatePerf(args_u8);

    var args_u4 = base_args;
    args_u4.weight_dtype = .grouped_affine_u4;
    const result_u4 = estimatePerf(args_u4);

    // FLOPs should be identical (independent of dtype)
    try std.testing.expectEqual(result_f32.prefill_flops, result_f16.prefill_flops);
    try std.testing.expectEqual(result_f32.prefill_flops, result_u8.prefill_flops);
    try std.testing.expectEqual(result_f32.prefill_flops, result_u4.prefill_flops);

    // Memory should decrease with smaller dtypes
    try std.testing.expect(result_f32.prefill_mem_bytes > result_f16.prefill_mem_bytes);
    try std.testing.expect(result_f16.prefill_mem_bytes > result_u8.prefill_mem_bytes);
    try std.testing.expect(result_u8.prefill_mem_bytes > result_u4.prefill_mem_bytes);
}

test "estimatePerf - moe model" {
    const args = EstimateArgs{
        .seq_len = 256,
        .weight_dtype = .f16,
        .hidden_size = 1024,
        .vocab_size = 20000,
        .num_hidden_layers = 12,
        .attn = .{
            .n_heads = 16,
            .n_kv_heads = 4,
            .head_dim = 64,
            .d_model = 1024,
        },
        .ffn = .{ .moe_ffn = .{
            .d_model = 1024,
            .d_ff = 2816,
            .num_experts = 8,
            .experts_per_token = 2,
        } },
    };

    const result = estimatePerf(args);

    // Should produce valid results
    try std.testing.expect(result.prefill_flops > 0);
    try std.testing.expect(result.per_token_flops > 0);
    try std.testing.expect(result.prefill_mem_bytes > 0);
    try std.testing.expect(result.per_token_mem_bytes > 0);
}

test "estimatePerf - minimal model" {
    const args = EstimateArgs{
        .seq_len = 1,
        .weight_dtype = .f16,
        .hidden_size = 64,
        .vocab_size = 100,
        .num_hidden_layers = 1,
        .attn = .{
            .n_heads = 2,
            .n_kv_heads = 2,
            .head_dim = 32,
            .d_model = 64,
        },
        .ffn = .{ .swiglu = .{
            .d_model = 64,
            .d_ff = 256,
        } },
    };

    const result = estimatePerf(args);

    // Even minimal models should have non-zero values
    try std.testing.expect(result.prefill_flops > 0);
    try std.testing.expect(result.per_token_flops > 0);
    try std.testing.expect(result.prefill_mem_bytes > 0);
    try std.testing.expect(result.per_token_mem_bytes > 0);
}

test "formatBytes - various sizes" {
    // Test TB range
    {
        var buf: [64]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buf);
        try formatBytes(stream.writer(), 1500000000000);
        try std.testing.expectEqualStrings("1.50 TB", stream.getWritten());
    }

    // Test GB range
    {
        var buf: [64]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buf);
        try formatBytes(stream.writer(), 2500000000);
        try std.testing.expectEqualStrings("2.50 GB", stream.getWritten());
    }

    // Test MB range
    {
        var buf: [64]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buf);
        try formatBytes(stream.writer(), 3500000);
        try std.testing.expectEqualStrings("3.50 MB", stream.getWritten());
    }

    // Test KB range
    {
        var buf: [64]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buf);
        try formatBytes(stream.writer(), 4500);
        try std.testing.expectEqualStrings("4.50 KB", stream.getWritten());
    }

    // Test B range
    {
        var buf: [64]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buf);
        try formatBytes(stream.writer(), 512);
        try std.testing.expectEqualStrings("512 B", stream.getWritten());
    }

    // Test zero
    {
        var buf: [64]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buf);
        try formatBytes(stream.writer(), 0);
        try std.testing.expectEqualStrings("0 B", stream.getWritten());
    }

    // Test boundary conditions
    {
        var buf: [64]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buf);
        try formatBytes(stream.writer(), 1000);
        try std.testing.expectEqualStrings("1.00 KB", stream.getWritten());
    }

    {
        var buf: [64]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buf);
        try formatBytes(stream.writer(), 1000000);
        try std.testing.expectEqualStrings("1.00 MB", stream.getWritten());
    }

    {
        var buf: [64]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buf);
        try formatBytes(stream.writer(), 1000000000);
        try std.testing.expectEqualStrings("1.00 GB", stream.getWritten());
    }

    {
        var buf: [64]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buf);
        try formatBytes(stream.writer(), 1000000000000);
        try std.testing.expectEqualStrings("1.00 TB", stream.getWritten());
    }
}

test "PerfEstimate.format - basic output" {
    const estimate = PerfEstimate{
        .prefill_flops = 1500000000000,
        .per_token_flops = 75000000,
        .prefill_mem_bytes = 2500000000,
        .per_token_mem_bytes = 1250000000,
        .seq_len = 512,
        .weight_dtype = .f16,
    };

    var buf: [2048]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try estimate.format(stream.writer());

    const output = stream.getWritten();

    // Check for expected sections
    try std.testing.expect(std.mem.indexOf(u8, output, "Performance estimate (seq_len=512)") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "FLOPs:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Prefill:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Per-token:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Memory bandwidth:") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Arithmetic intensity") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Theoretical decode tok/s:") != null);

    // Check for hardware profiles
    try std.testing.expect(std.mem.indexOf(u8, output, "CPU (AVX2)") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "M1 Pro") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "H100 SXM") != null);

    // Check for bound annotations
    try std.testing.expect(std.mem.indexOf(u8, output, "-bound") != null);
}

test "PerfEstimate.format - edge case values" {
    // Test with very small values
    const estimate = PerfEstimate{
        .prefill_flops = 1000,
        .per_token_flops = 100,
        .prefill_mem_bytes = 500,
        .per_token_mem_bytes = 250,
        .seq_len = 1,
        .weight_dtype = .f32,
    };

    var buf: [2048]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);

    try estimate.format(stream.writer());

    // Should not crash and should contain basic structure
    const output = stream.getWritten();
    try std.testing.expect(output.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, output, "Performance estimate") != null);
}
