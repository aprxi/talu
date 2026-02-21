//! Model-owned RoPE scaling materialization.
//!
//! This module translates model configuration policy (`tensor.RopeScaling`) into
//! plain numeric inverse-frequency tables consumed by compute primitives.

const std = @import("std");
const tensor = @import("../tensor.zig");

pub const MaterializedRope = struct {
    inv_freq: []f32,
    attention_scaling: f32,

    pub fn deinit(self: *MaterializedRope, allocator: std.mem.Allocator) void {
        allocator.free(self.inv_freq);
        self.* = undefined;
    }
};

pub fn materializeInverseFrequencies(
    allocator: std.mem.Allocator,
    dim: usize,
    theta: f32,
    rope_scaling: tensor.RopeScaling,
) !MaterializedRope {
    if ((dim % 2) != 0) return error.InvalidShape;
    if (theta <= 0.0) return error.InvalidShape;
    const half_dim = dim / 2;

    const inv_freq = try allocator.alloc(f32, half_dim);
    errdefer allocator.free(inv_freq);

    switch (rope_scaling.rope_type) {
        .llama3 => {
            fillLlama3InverseFrequencies(inv_freq, dim, theta, rope_scaling);
            return .{ .inv_freq = inv_freq, .attention_scaling = 1.0 };
        },
        .yarn => {
            const attention_scaling = fillYarnInverseFrequencies(inv_freq, dim, theta, rope_scaling);
            return .{ .inv_freq = inv_freq, .attention_scaling = attention_scaling };
        },
        .linear, .none => {
            const inv_freq_scale: f32 = if (rope_scaling.rope_type == .linear and rope_scaling.factor > 0)
                1.0 / rope_scaling.factor
            else
                1.0;
            fillStandardInverseFrequencies(inv_freq, dim, theta, inv_freq_scale);
            return .{ .inv_freq = inv_freq, .attention_scaling = 1.0 };
        },
    }
}

fn fillStandardInverseFrequencies(inv_freq: []f32, dim: usize, theta: f32, inv_freq_scale: f32) void {
    for (0..inv_freq.len) |dim_idx| {
        const exponent = @as(f64, @floatFromInt(2 * dim_idx)) / @as(f64, @floatFromInt(dim));
        inv_freq[dim_idx] = @floatCast((1.0 / std.math.pow(f64, @as(f64, theta), exponent)) * @as(f64, inv_freq_scale));
    }
}

fn fillLlama3InverseFrequencies(inv_freq: []f32, dim: usize, theta: f32, rope_scaling: tensor.RopeScaling) void {
    const factor = rope_scaling.factor;
    const low_freq_factor = rope_scaling.low_freq_factor;
    const high_freq_factor = rope_scaling.high_freq_factor;
    const old_ctx: f32 = @floatFromInt(rope_scaling.original_max_position_embeddings);

    const low_freq_wavelen = old_ctx / low_freq_factor;
    const high_freq_wavelen = old_ctx / high_freq_factor;
    const dims_f: f32 = @floatFromInt(dim);

    const denom = high_freq_factor - low_freq_factor;
    const inv_denom: f32 = if (denom != 0) 1.0 / denom else 0;

    for (0..inv_freq.len) |dim_idx| {
        const dim_pos: f32 = @floatFromInt(dim_idx * 2);
        const exponent = dim_pos / dims_f;
        const freq = std.math.pow(f32, theta, exponent);
        const wavelen = 2.0 * std.math.pi * freq;

        const freq_scaled: f32 = if (factor > 0 and wavelen > low_freq_wavelen) blk: {
            break :blk freq * factor;
        } else if (wavelen < high_freq_wavelen or factor <= 0 or inv_denom == 0) blk: {
            break :blk freq;
        } else blk: {
            const smooth_factor = (old_ctx / wavelen - low_freq_factor) * inv_denom;
            const smooth_mix = @min(@max(smooth_factor, 0.0), 1.0);
            break :blk freq / (((1.0 - smooth_mix) / factor) + smooth_mix);
        };

        inv_freq[dim_idx] = 1.0 / freq_scaled;
    }
}

fn fillYarnInverseFrequencies(inv_freq: []f32, dim: usize, theta: f32, rope_scaling: tensor.RopeScaling) f32 {
    const factor: f64 = if (rope_scaling.factor > 0) @as(f64, rope_scaling.factor) else 1.0;
    const beta_fast: f64 = @as(f64, rope_scaling.beta_fast);
    const beta_slow: f64 = @as(f64, rope_scaling.beta_slow);
    const original_max_pos: f64 = @as(f64, @floatFromInt(rope_scaling.original_max_position_embeddings));
    const base: f64 = @as(f64, theta);
    const dim_f64: f64 = @as(f64, @floatFromInt(dim));

    const attention_scaling = computeYarnAttentionScaling(@as(f32, @floatCast(factor)), rope_scaling);

    var low = findYarnCorrectionDim(beta_fast, dim_f64, base, original_max_pos);
    var high = findYarnCorrectionDim(beta_slow, dim_f64, base, original_max_pos);
    if (rope_scaling.truncate) {
        low = @floor(low);
        high = @ceil(high);
    }
    if (low < 0) low = 0;
    if (high > dim_f64 - 1) high = dim_f64 - 1;
    if (low == high) {
        high += 0.001;
    }

    const denom = high - low;
    for (0..inv_freq.len) |dim_idx| {
        const dim_pos: f64 = @as(f64, @floatFromInt(dim_idx * 2));
        const exponent = dim_pos / dim_f64;
        const pos_freq = std.math.pow(f64, base, exponent);
        const inv_extrap = 1.0 / pos_freq;
        const inv_interp = 1.0 / (factor * pos_freq);
        var linear = (@as(f64, @floatFromInt(dim_idx)) - low) / denom;
        if (linear < 0) linear = 0;
        if (linear > 1) linear = 1;
        const inv_extrap_factor = 1.0 - linear;
        const inv_val = inv_interp * (1.0 - inv_extrap_factor) + inv_extrap * inv_extrap_factor;
        inv_freq[dim_idx] = @floatCast(inv_val);
    }

    return attention_scaling;
}

fn computeYarnAttentionScaling(factor: f32, rope_scaling: tensor.RopeScaling) f32 {
    if (rope_scaling.attention_factor > 0) return rope_scaling.attention_factor;

    const mscale_value: f32 = if (rope_scaling.mscale > 0) rope_scaling.mscale else 1.0;
    if (rope_scaling.mscale > 0 and rope_scaling.mscale_all_dim > 0) {
        const numerator = yarnMscale(factor, rope_scaling.mscale);
        const denominator = yarnMscale(factor, rope_scaling.mscale_all_dim);
        if (denominator != 0) return numerator / denominator;
    }
    return yarnMscale(factor, mscale_value);
}

fn yarnMscale(scale: f32, mscale_value: f32) f32 {
    if (scale <= 1.0) return 1.0;
    return 0.1 * mscale_value * @as(f32, @floatCast(std.math.log(f64, std.math.e, @as(f64, scale)))) + 1.0;
}

fn findYarnCorrectionDim(num_rotations: f64, dim_value: f64, base_value: f64, max_pos: f64) f64 {
    const denom = 2.0 * std.math.log(f64, std.math.e, base_value);
    if (denom == 0) return 0;
    return (dim_value * std.math.log(f64, std.math.e, max_pos / (num_rotations * 2.0 * std.math.pi))) / denom;
}

test "materializeInverseFrequencies llama3 matches expected" {
    const allocator = std.testing.allocator;
    const dim: usize = 128;
    const theta: f32 = 500_000.0;
    const scaling = tensor.RopeScaling{
        .rope_type = .llama3,
        .factor = 8.0,
        .low_freq_factor = 1.0,
        .high_freq_factor = 4.0,
        .original_max_position_embeddings = 8192,
    };

    var materialized = try materializeInverseFrequencies(allocator, dim, theta, scaling);
    defer materialized.deinit(allocator);

    const half = dim / 2;
    const old_ctx: f32 = @floatFromInt(scaling.original_max_position_embeddings);
    const low_freq_wavelen = old_ctx / scaling.low_freq_factor;
    const high_freq_wavelen = old_ctx / scaling.high_freq_factor;
    const dims_f: f32 = @floatFromInt(dim);
    const denom = scaling.high_freq_factor - scaling.low_freq_factor;
    const inv_denom: f32 = if (denom != 0) 1.0 / denom else 0;

    var dim_idx: usize = 0;
    while (dim_idx < half) : (dim_idx += 1) {
        const dim_pos: f32 = @floatFromInt(dim_idx * 2);
        const freq = std.math.pow(f32, theta, dim_pos / dims_f);
        const wavelen = 2.0 * std.math.pi * freq;
        const freq_scaled: f32 = if (wavelen > low_freq_wavelen) blk: {
            break :blk freq * scaling.factor;
        } else if (wavelen < high_freq_wavelen or inv_denom == 0) blk: {
            break :blk freq;
        } else blk: {
            const smooth_factor = (old_ctx / wavelen - scaling.low_freq_factor) * inv_denom;
            const smooth_mix = @min(@max(smooth_factor, 0.0), 1.0);
            break :blk freq / (((1.0 - smooth_mix) / scaling.factor) + smooth_mix);
        };
        const expected_inv = 1.0 / freq_scaled;
        try std.testing.expectApproxEqRel(expected_inv, materialized.inv_freq[dim_idx], 1e-5);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), materialized.attention_scaling, 1e-6);
}

test "materializeInverseFrequencies yarn matches reference values" {
    const allocator = std.testing.allocator;
    const dim: usize = 128;
    const theta: f32 = 150000.0;
    const scaling = tensor.RopeScaling{
        .rope_type = .yarn,
        .factor = 32.0,
        .beta_fast = 32.0,
        .beta_slow = 1.0,
        .original_max_position_embeddings = 4096,
        .truncate = false,
    };

    var materialized = try materializeInverseFrequencies(allocator, dim, theta, scaling);
    defer materialized.deinit(allocator);

    const expected_head = [_]f32{
        1.0,
        0.83008695,
        0.6890443,
        0.57196665,
        0.47478205,
        0.39411038,
    };
    for (expected_head, 0..) |expected, idx| {
        try std.testing.expectApproxEqAbs(expected, materialized.inv_freq[idx], 1e-5);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.3465736), materialized.attention_scaling, 1e-6);
}
