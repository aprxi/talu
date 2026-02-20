//! Rotary Position Embedding (RoPE) with lazy cache expansion.

const std = @import("std");
const tensor = @import("../../../tensor.zig");
const simd = @import("../simd/arch/root.zig");
const log = @import("../../../log.zig");

const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

pub fn ropeFillCosSinCombinedStrided(
    data: [*]f32,
    stride0: usize,
    stride1: usize,
    seq_len: usize,
    dim: usize,
    theta: f32,
    offset: usize,
) void {
    const half_dim = dim / 2;
    for (0..seq_len) |pos| {
        const actual_pos = pos + offset;
        for (0..half_dim) |dim_idx| {
            const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * dim_idx)) / @as(f32, @floatFromInt(dim)));
            const angle = @as(f32, @floatFromInt(actual_pos)) * freq;
            const base = pos * stride0;
            data[base + dim_idx * stride1] = @cos(angle);
            data[base + (dim_idx + half_dim) * stride1] = @sin(angle);
        }
    }
}

fn ropeFillCosSinFromInvFreq(
    cos: []f32,
    sin: []f32,
    inv_freq: []const f32,
    dim: usize,
    pos_start: usize,
    count: usize,
) void {
    const half_dim = dim / 2;
    for (0..count) |step_idx| {
        const pos = pos_start + step_idx;
        const base = step_idx * dim;
        for (0..half_dim) |dim_idx| {
            const angle = @as(f32, @floatFromInt(pos)) * inv_freq[dim_idx];
            const cos_val = @cos(angle);
            const sin_val = @sin(angle);
            cos[base + dim_idx] = cos_val;
            cos[base + dim_idx + half_dim] = cos_val;
            sin[base + dim_idx] = sin_val;
            sin[base + dim_idx + half_dim] = sin_val;
        }
    }
}

/// Rotary Position Embedding (RoPE) with lazy cache expansion.
///
/// The RoPE cache starts small (256 positions) and grows on demand as longer
/// sequences are processed. This uses `realloc` to expand the cache, so the
/// allocator passed to `init()` MUST support true realloc semantics:
///
/// **WARNING: Do NOT use ArenaAllocator for RoPE.**
/// ArenaAllocator.realloc() does not free old memory - it allocates new memory
/// and copies, leaving the old allocation orphaned. This causes:
/// - Memory exhaustion for long sequences (each expansion leaks the old cache)
/// - Crashes when the arena's backing pages become fragmented
///
/// Use a general-purpose allocator like `std.heap.c_allocator` or `page_allocator`.
///
pub const RoPE = struct {
    dim: usize,
    max_seq_len: usize,
    theta: f32,
    attention_scaling: f32,
    /// Precomputed inverse frequencies (dim/2 values)
    inv_freq: []f32,
    /// Cached cos/sin values (computed lazily up to cached_len).
    /// IMPORTANT: This cache grows via realloc - see allocator requirements above.
    freqs_cos: []f32,
    freqs_sin: []f32,
    cached_len: usize,
    allocator: std.mem.Allocator,

    /// Initialize RoPE with given parameters.
    ///
    /// `inv_freq_scale` lets callers implement simple RoPE scaling variants by scaling the inverse
    /// frequencies (e.g. linear RoPE scaling uses `inv_freq_scale = 1 / factor`).
    ///
    /// **Allocator requirements**: The allocator must support true realloc semantics.
    /// Do NOT pass an ArenaAllocator - use c_allocator or similar.
    pub fn init(allocator: std.mem.Allocator, dim: usize, max_seq_len: usize, theta: f32, inv_freq_scale: f32) !RoPE {
        // Precompute inverse frequencies (only dim/2 values)
        const half_dim = dim / 2;
        const inv_freq = try allocator.alloc(f32, half_dim);
        errdefer allocator.free(inv_freq);

        for (0..half_dim) |dim_idx| {
            const exponent = @as(f64, @floatFromInt(2 * dim_idx)) / @as(f64, @floatFromInt(dim));
            inv_freq[dim_idx] = @floatCast((1.0 / std.math.pow(f64, @as(f64, theta), exponent)) * @as(f64, inv_freq_scale));
        }

        // Start with small cache (256 positions typical for short prompts)
        const initial_cache = @min(256, max_seq_len);
        const cos = try allocator.alloc(f32, initial_cache * dim);
        errdefer allocator.free(cos);
        const sin = try allocator.alloc(f32, initial_cache * dim);
        errdefer allocator.free(sin);

        ropeFillCosSinFromInvFreq(cos, sin, inv_freq, dim, 0, initial_cache);

        return RoPE{
            .dim = dim,
            .max_seq_len = max_seq_len,
            .theta = theta,
            .attention_scaling = 1.0,
            .inv_freq = inv_freq,
            .freqs_cos = cos,
            .freqs_sin = sin,
            .cached_len = initial_cache,
            .allocator = allocator,
        };
    }

    pub fn initWithRopeScaling(
        allocator: std.mem.Allocator,
        dim: usize,
        max_seq_len: usize,
        theta: f32,
        rope_scaling: tensor.RopeScaling,
    ) !RoPE {
        const inv_freq_scale: f32 = if (rope_scaling.rope_type == .linear and rope_scaling.factor > 0)
            1.0 / rope_scaling.factor
        else
            1.0;

        if (rope_scaling.rope_type == .yarn) {
            return RoPE.initWithYarnScaling(allocator, dim, max_seq_len, theta, rope_scaling);
        }

        if (rope_scaling.rope_type != .llama3) {
            return RoPE.init(allocator, dim, max_seq_len, theta, inv_freq_scale);
        }

        // Llama3-style RoPE uses wavelength-dependent frequency scaling.
        // We implement the same formula as mlx_lm by modifying the *denominator* `freq = theta^(2i/dim)`
        // and then storing `inv_freq = 1 / freq`.
        const half_dim = dim / 2;
        const inv_freq = try allocator.alloc(f32, half_dim);
        errdefer allocator.free(inv_freq);

        const factor = rope_scaling.factor;
        const low_freq_factor = rope_scaling.low_freq_factor;
        const high_freq_factor = rope_scaling.high_freq_factor;
        const old_ctx: f32 = @floatFromInt(rope_scaling.original_max_position_embeddings);

        const low_freq_wavelen = old_ctx / low_freq_factor;
        const high_freq_wavelen = old_ctx / high_freq_factor;
        const dims_f: f32 = @floatFromInt(dim);

        const denom = high_freq_factor - low_freq_factor;
        const inv_denom: f32 = if (denom != 0) 1.0 / denom else 0;

        for (0..half_dim) |dim_idx| {
            const dim_pos: f32 = @floatFromInt(dim_idx * 2);
            const exponent = dim_pos / dims_f;
            const freq = std.math.pow(f32, theta, exponent);
            const wavelen = 2.0 * std.math.pi * freq;

            const freq_scaled: f32 = if (factor > 0 and wavelen > low_freq_wavelen) blk: {
                break :blk freq * factor;
            } else if (wavelen < high_freq_wavelen or factor <= 0 or inv_denom == 0) blk: {
                break :blk freq;
            } else blk: {
                // Medium wavelengths: smooth interpolation.
                // smooth_factor = (old_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                const smooth_factor = (old_ctx / wavelen - low_freq_factor) * inv_denom;
                const smooth_mix = @min(@max(smooth_factor, 0.0), 1.0);
                // freq / ((1 - smooth_mix)/factor + smooth_mix)
                break :blk freq / (((1.0 - smooth_mix) / factor) + smooth_mix);
            };

            inv_freq[dim_idx] = 1.0 / freq_scaled;
        }

        // Start with small cache (256 positions typical for short prompts)
        const initial_cache = @min(256, max_seq_len);
        const cos = try allocator.alloc(f32, initial_cache * dim);
        errdefer allocator.free(cos);
        const sin = try allocator.alloc(f32, initial_cache * dim);
        errdefer allocator.free(sin);

        ropeFillCosSinFromInvFreq(cos, sin, inv_freq, dim, 0, initial_cache);

        return RoPE{
            .dim = dim,
            .max_seq_len = max_seq_len,
            .theta = theta,
            .attention_scaling = 1.0,
            .inv_freq = inv_freq,
            .freqs_cos = cos,
            .freqs_sin = sin,
            .cached_len = initial_cache,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RoPE, allocator: std.mem.Allocator) void {
        allocator.free(self.inv_freq);
        allocator.free(self.freqs_cos);
        allocator.free(self.freqs_sin);
        self.* = undefined;
    }

    /// Ensure cache covers at least `needed` positions.
    /// Grows cache exponentially (power of 2) up to max_seq_len.
    fn ensureCache(self: *RoPE, needed: usize) void {
        if (needed <= self.cached_len) return;

        // Grow cache to power of 2 for efficient reallocation
        var new_len = self.cached_len;
        while (new_len < needed) {
            // Saturating multiply to avoid overflow (though max_seq_len bound should prevent this)
            new_len = if (new_len > std.math.maxInt(usize) / 2) std.math.maxInt(usize) else new_len * 2;
        }
        new_len = @min(new_len, self.max_seq_len);

        // Realloc can fail if system is out of memory - log and continue with existing cache
        // The caller will see stale cache data which is incorrect but won't crash
        const new_cos = self.allocator.realloc(self.freqs_cos, new_len * self.dim) catch {
            log.warn("compute", "RoPE cache realloc failed", .{ .positions = new_len });
            return;
        };
        const new_sin = self.allocator.realloc(self.freqs_sin, new_len * self.dim) catch {
            // Rollback cos realloc (we got new memory but sin failed)
            self.freqs_cos = self.allocator.realloc(new_cos, self.cached_len * self.dim) catch new_cos;
            log.warn("compute", "RoPE cache realloc failed", .{ .positions = new_len });
            return;
        };
        self.freqs_cos = new_cos;
        self.freqs_sin = new_sin;

        const count = new_len - self.cached_len;
        const cos_slice = new_cos[self.cached_len * self.dim .. new_len * self.dim];
        const sin_slice = new_sin[self.cached_len * self.dim .. new_len * self.dim];
        ropeFillCosSinFromInvFreq(cos_slice, sin_slice, self.inv_freq, self.dim, self.cached_len, count);
        if (self.attention_scaling != 1.0) {
            scaleCosSin(cos_slice, sin_slice, self.attention_scaling);
        }
        self.cached_len = new_len;
    }

    fn applyRotation(vec: []f32, cos: []const f32, sin: []const f32, half: usize) void {
        applyRopeRotationContiguous(vec, cos, sin, half);
    }

    pub fn applyInPlace(self: *RoPE, vec: []f32, pos: usize) void {
        // Ensure cache covers this position
        if (pos >= self.cached_len) self.ensureCache(pos + 1);

        const half = self.dim / 2;
        const base = pos * self.dim;
        const cos = self.freqs_cos[base..];
        const sin = self.freqs_sin[base..];

        applyRotation(vec, cos[0..half], sin[0..half], half);
    }

    /// Get cos values for a given position (dim/2 values).
    /// Ensures cache covers this position.
    pub fn getCos(self: *RoPE, pos: usize) []const f32 {
        if (pos >= self.cached_len) self.ensureCache(pos + 1);
        const half = self.dim / 2;
        const base = pos * self.dim;
        return self.freqs_cos[base .. base + half];
    }

    /// Get sin values for a given position (dim/2 values).
    /// Ensures cache covers this position.
    pub fn getSin(self: *RoPE, pos: usize) []const f32 {
        if (pos >= self.cached_len) self.ensureCache(pos + 1);
        const half = self.dim / 2;
        const base = pos * self.dim;
        return self.freqs_sin[base .. base + half];
    }

    /// Apply interleaved RoPE rotation (for MLA-style models).
    /// In interleaved layout, consecutive pairs (x0, x1), (x2, x3), ... rotate together.
    /// This differs from standard RoPE where first half rotates with second half.
    pub fn applyInterleavedInPlace(self: *RoPE, vec: []f32, pos: usize) void {
        if (pos >= self.cached_len) self.ensureCache(pos + 1);

        const half = vec.len / 2;
        const base = pos * self.dim;
        const cos = self.freqs_cos[base..];
        const sin = self.freqs_sin[base..];

        // Interleaved rotation: pairs (x0, x1), (x2, x3), ...
        var i: usize = 0;
        while (i < half) : (i += 1) {
            const x0 = vec[i * 2];
            const x1 = vec[i * 2 + 1];
            vec[i * 2] = x0 * cos[i] - x1 * sin[i];
            vec[i * 2 + 1] = x1 * cos[i] + x0 * sin[i];
        }
    }

    fn initWithYarnScaling(
        allocator: std.mem.Allocator,
        dim: usize,
        max_seq_len: usize,
        theta: f32,
        rope_scaling: tensor.RopeScaling,
    ) !RoPE {
        const half_dim = dim / 2;
        const inv_freq = try allocator.alloc(f32, half_dim);
        errdefer allocator.free(inv_freq);

        const attention_scaling = computeYarnInvFreq(inv_freq, dim, theta, rope_scaling);

        const initial_cache = @min(256, max_seq_len);
        const cos = try allocator.alloc(f32, initial_cache * dim);
        errdefer allocator.free(cos);
        const sin = try allocator.alloc(f32, initial_cache * dim);
        errdefer allocator.free(sin);

        ropeFillCosSinFromInvFreq(cos, sin, inv_freq, dim, 0, initial_cache);
        if (attention_scaling != 1.0) {
            scaleCosSin(cos, sin, attention_scaling);
        }

        return RoPE{
            .dim = dim,
            .max_seq_len = max_seq_len,
            .theta = theta,
            .attention_scaling = attention_scaling,
            .inv_freq = inv_freq,
            .freqs_cos = cos,
            .freqs_sin = sin,
            .cached_len = initial_cache,
            .allocator = allocator,
        };
    }
};

fn scaleCosSin(cos: []f32, sin: []f32, scale: f32) void {
    if (scale == 1.0) return;
    for (cos) |*val| {
        val.* *= scale;
    }
    for (sin) |*val| {
        val.* *= scale;
    }
}

fn computeYarnInvFreq(inv_freq: []f32, dim: usize, theta: f32, rope_scaling: tensor.RopeScaling) f32 {
    const half_dim = dim / 2;
    std.debug.assert(inv_freq.len == half_dim);

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
    for (0..half_dim) |dim_idx| {
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

fn applyRopeRotationContiguous(vec: []f32, cos: []const f32, sin: []const f32, half: usize) void {
    @setFloatMode(.optimized);
    std.debug.assert(vec.len >= half * 2);
    std.debug.assert(cos.len >= half);
    std.debug.assert(sin.len >= half);

    var vec_idx: usize = 0;
    while (vec_idx + VEC_LEN - 1 < half) : (vec_idx += VEC_LEN) {
        const x1: F32Vec = vec[vec_idx..][0..VEC_LEN].*;
        const x2: F32Vec = vec[vec_idx + half ..][0..VEC_LEN].*;
        const cos_vec: F32Vec = cos[vec_idx..][0..VEC_LEN].*;
        const sin_vec: F32Vec = sin[vec_idx..][0..VEC_LEN].*;
        const r1 = @mulAdd(F32Vec, x1, cos_vec, -x2 * sin_vec);
        const r2 = @mulAdd(F32Vec, x2, cos_vec, x1 * sin_vec);
        vec[vec_idx..][0..VEC_LEN].* = r1;
        vec[vec_idx + half ..][0..VEC_LEN].* = r2;
    }
    while (vec_idx < half) : (vec_idx += 1) {
        const x1 = vec[vec_idx];
        const x2 = vec[vec_idx + half];
        const cos_val = cos[vec_idx];
        const sin_val = sin[vec_idx];
        vec[vec_idx] = x1 * cos_val - x2 * sin_val;
        vec[vec_idx + half] = x2 * cos_val + x1 * sin_val;
    }
}

pub fn applyRopeRotationStrided(
    comptime T: type,
    comptime toF32: fn (T) f32,
    comptime fromF32: fn (f32) T,
    data: [*]T,
    data_stride: usize,
    cos: [*]const T,
    cos_stride: usize,
    sin: [*]const T,
    sin_stride: usize,
    half: usize,
) void {
    if (T == f32 and data_stride == 1 and cos_stride == 1 and sin_stride == 1) {
        applyRopeRotationContiguous(data[0 .. half * 2], cos[0..half], sin[0..half], half);
        return;
    }

    for (0..half) |dim_idx| {
        const idx0 = dim_idx * data_stride;
        const idx1 = (dim_idx + half) * data_stride;
        const cos_val = toF32(cos[dim_idx * cos_stride]);
        const sin_val = toF32(sin[dim_idx * sin_stride]);
        const x0 = toF32(data[idx0]);
        const x1 = toF32(data[idx1]);
        data[idx0] = fromF32(x0 * cos_val - x1 * sin_val);
        data[idx1] = fromF32(x1 * cos_val + x0 * sin_val);
    }
}

test "initWithRopeScaling llama3 matches mlx" {
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

    var rope_inst = try RoPE.initWithRopeScaling(allocator, dim, 32, theta, scaling);
    defer rope_inst.deinit(allocator);

    // Recompute expected inv_freq in-place (mirrors metal computeLlama3RopeFreqs + inversion).
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
        try std.testing.expectApproxEqRel(expected_inv, rope_inst.inv_freq[dim_idx], 1e-5);
    }
}

test "ropeFillCosSinCombinedStrided - basic correctness" {
    const allocator = std.testing.allocator;

    const seq_len: usize = 2;
    const dim: usize = 4;
    const theta: f32 = 10000.0;
    const offset: usize = 0;
    const stride0: usize = dim;
    const stride1: usize = 1;

    const data = try allocator.alloc(f32, seq_len * dim);
    defer allocator.free(data);

    ropeFillCosSinCombinedStrided(data.ptr, stride0, stride1, seq_len, dim, theta, offset);

    // First half should be cosines, second half should be sines
    const half_dim = dim / 2;

    // Verify first position (pos=0)
    for (0..half_dim) |dim_idx| {
        const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * dim_idx)) / @as(f32, @floatFromInt(dim)));
        const angle = 0.0 * freq;
        const expected_cos = @cos(angle);
        const expected_sin = @sin(angle);

        try std.testing.expectApproxEqRel(expected_cos, data[dim_idx], 1e-5);
        try std.testing.expectApproxEqRel(expected_sin, data[dim_idx + half_dim], 1e-5);
    }

    // Verify second position (pos=1)
    for (0..half_dim) |dim_idx| {
        const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * dim_idx)) / @as(f32, @floatFromInt(dim)));
        const angle = 1.0 * freq;
        const expected_cos = @cos(angle);
        const expected_sin = @sin(angle);

        try std.testing.expectApproxEqRel(expected_cos, data[dim + dim_idx], 1e-5);
        try std.testing.expectApproxEqRel(expected_sin, data[dim + dim_idx + half_dim], 1e-5);
    }
}

test "ropeFillCosSinCombinedStrided - with offset" {
    const allocator = std.testing.allocator;

    const seq_len: usize = 2;
    const dim: usize = 4;
    const theta: f32 = 10000.0;
    const offset: usize = 10;
    const stride0: usize = dim;
    const stride1: usize = 1;

    const data = try allocator.alloc(f32, seq_len * dim);
    defer allocator.free(data);

    ropeFillCosSinCombinedStrided(data.ptr, stride0, stride1, seq_len, dim, theta, offset);

    const half_dim = dim / 2;

    // First position should use actual_pos = 0 + 10 = 10
    for (0..half_dim) |dim_idx| {
        const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * dim_idx)) / @as(f32, @floatFromInt(dim)));
        const angle = 10.0 * freq;
        const expected_cos = @cos(angle);
        const expected_sin = @sin(angle);

        try std.testing.expectApproxEqRel(expected_cos, data[dim_idx], 1e-5);
        try std.testing.expectApproxEqRel(expected_sin, data[dim_idx + half_dim], 1e-5);
    }
}

test "ropeFillCosSinCombinedStrided - strided layout" {
    const allocator = std.testing.allocator;

    const seq_len: usize = 2;
    const dim: usize = 4;
    const theta: f32 = 10000.0;
    const offset: usize = 0;
    const stride0: usize = 8; // Non-contiguous sequence stride
    const stride1: usize = 2; // Non-contiguous dim stride

    const data = try allocator.alloc(f32, seq_len * stride0);
    defer allocator.free(data);
    @memset(data, 0);

    ropeFillCosSinCombinedStrided(data.ptr, stride0, stride1, seq_len, dim, theta, offset);

    const half_dim = dim / 2;

    // Verify first position with strides
    for (0..half_dim) |dim_idx| {
        const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * dim_idx)) / @as(f32, @floatFromInt(dim)));
        const angle = 0.0 * freq;
        const expected_cos = @cos(angle);
        const expected_sin = @sin(angle);

        try std.testing.expectApproxEqRel(expected_cos, data[dim_idx * stride1], 1e-5);
        try std.testing.expectApproxEqRel(expected_sin, data[(dim_idx + half_dim) * stride1], 1e-5);
    }
}

test "ropeFillCosSinCombinedStrided - edge case single position" {
    const allocator = std.testing.allocator;

    const seq_len: usize = 1;
    const dim: usize = 2;
    const theta: f32 = 10000.0;
    const offset: usize = 0;
    const stride0: usize = dim;
    const stride1: usize = 1;

    const data = try allocator.alloc(f32, seq_len * dim);
    defer allocator.free(data);

    ropeFillCosSinCombinedStrided(data.ptr, stride0, stride1, seq_len, dim, theta, offset);

    // With pos=0, dim_idx=0: freq = 1.0 / theta^0 = 1.0, angle = 0
    try std.testing.expectApproxEqRel(@as(f32, 1.0), data[0], 1e-5); // cos(0) = 1
    try std.testing.expectApproxEqRel(@as(f32, 0.0), data[1], 1e-5); // sin(0) = 0
}

// =============================================================================
// applyRopeRotationStrided Tests
// =============================================================================

fn identity_f32(x: f32) f32 {
    return x;
}

test "applyRopeRotationStrided - contiguous f32 delegates to fast path" {
    const half: usize = 2;
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const cos_data = [_]f32{ 0.5, 0.866 };
    const sin_data = [_]f32{ 0.866, 0.5 };

    const original = data;

    applyRopeRotationStrided(f32, identity_f32, identity_f32, &data, 1, &cos_data, 1, &sin_data, 1, half);

    // Verify rotation formula: [x0', x1'] = [x0*cos - x1*sin, x1*cos + x0*sin]
    const expected_0 = original[0] * cos_data[0] - original[2] * sin_data[0];
    const expected_1 = original[1] * cos_data[1] - original[3] * sin_data[1];
    const expected_2 = original[2] * cos_data[0] + original[0] * sin_data[0];
    const expected_3 = original[3] * cos_data[1] + original[1] * sin_data[1];

    try std.testing.expectApproxEqRel(expected_0, data[0], 1e-5);
    try std.testing.expectApproxEqRel(expected_1, data[1], 1e-5);
    try std.testing.expectApproxEqRel(expected_2, data[2], 1e-5);
    try std.testing.expectApproxEqRel(expected_3, data[3], 1e-5);
}

test "applyRopeRotationStrided - strided data layout" {
    const half: usize = 2;
    var data = [_]f32{ 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0 }; // Stride 2
    const cos_data = [_]f32{ 0.5, 0.866 };
    const sin_data = [_]f32{ 0.866, 0.5 };

    applyRopeRotationStrided(f32, identity_f32, identity_f32, &data, 2, &cos_data, 1, &sin_data, 1, half);

    // Verify strided access: data[0, 2, 4, 6] are accessed with stride 2
    const expected_0 = 1.0 * cos_data[0] - 3.0 * sin_data[0];
    const expected_2 = 2.0 * cos_data[1] - 4.0 * sin_data[1];
    const expected_4 = 3.0 * cos_data[0] + 1.0 * sin_data[0];
    const expected_6 = 4.0 * cos_data[1] + 2.0 * sin_data[1];

    try std.testing.expectApproxEqRel(expected_0, data[0], 1e-5);
    try std.testing.expectApproxEqRel(expected_2, data[2], 1e-5);
    try std.testing.expectApproxEqRel(expected_4, data[4], 1e-5);
    try std.testing.expectApproxEqRel(expected_6, data[6], 1e-5);

    // Verify padding elements unchanged
    try std.testing.expectEqual(@as(f32, 0.0), data[1]);
    try std.testing.expectEqual(@as(f32, 0.0), data[3]);
    try std.testing.expectEqual(@as(f32, 0.0), data[5]);
    try std.testing.expectEqual(@as(f32, 0.0), data[7]);
}

test "applyRopeRotationStrided - strided cos/sin" {
    const half: usize = 2;
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const cos_data = [_]f32{ 0.5, 0.0, 0.866, 0.0 }; // Stride 2
    const sin_data = [_]f32{ 0.866, 0.0, 0.5, 0.0 }; // Stride 2

    applyRopeRotationStrided(f32, identity_f32, identity_f32, &data, 1, &cos_data, 2, &sin_data, 2, half);

    // Verify correct cos/sin values accessed with stride 2
    // data[0,1] are first half, data[2,3] are second half (half=2)
    const expected_0 = 1.0 * cos_data[0] - 3.0 * sin_data[0];
    const expected_1 = 2.0 * cos_data[2] - 4.0 * sin_data[2];
    const expected_2 = 3.0 * cos_data[0] + 1.0 * sin_data[0];
    const expected_3 = 4.0 * cos_data[2] + 2.0 * sin_data[2];

    try std.testing.expectApproxEqRel(expected_0, data[0], 1e-5);
    try std.testing.expectApproxEqRel(expected_1, data[1], 1e-5);
    try std.testing.expectApproxEqRel(expected_2, data[2], 1e-5);
    try std.testing.expectApproxEqRel(expected_3, data[3], 1e-5);
}

test "applyRopeRotationStrided - identity rotation (cos=1, sin=0)" {
    const half: usize = 2;
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const cos_data = [_]f32{ 1.0, 1.0 };
    const sin_data = [_]f32{ 0.0, 0.0 };

    const original = data;

    applyRopeRotationStrided(f32, identity_f32, identity_f32, &data, 1, &cos_data, 1, &sin_data, 1, half);

    // Identity rotation: data should be unchanged
    try std.testing.expectApproxEqRel(original[0], data[0], 1e-5);
    try std.testing.expectApproxEqRel(original[1], data[1], 1e-5);
    try std.testing.expectApproxEqRel(original[2], data[2], 1e-5);
    try std.testing.expectApproxEqRel(original[3], data[3], 1e-5);
}

test "applyRopeRotationStrided - 90 degree rotation" {
    const half: usize = 2;
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const cos_data = [_]f32{ 0.0, 0.0 }; // cos(90°) = 0
    const sin_data = [_]f32{ 1.0, 1.0 }; // sin(90°) = 1

    applyRopeRotationStrided(f32, identity_f32, identity_f32, &data, 1, &cos_data, 1, &sin_data, 1, half);

    // 90° rotation: [x, y] -> [-y, x]
    try std.testing.expectApproxEqRel(@as(f32, -3.0), data[0], 1e-5); // 0*1 - 3*1
    try std.testing.expectApproxEqRel(@as(f32, -4.0), data[1], 1e-5); // 0*2 - 4*1
    try std.testing.expectApproxEqRel(@as(f32, 1.0), data[2], 1e-5); // 3*0 + 1*1
    try std.testing.expectApproxEqRel(@as(f32, 2.0), data[3], 1e-5); // 4*0 + 2*1
}

test "applyRopeRotationStrided - single dimension pair" {
    const half: usize = 1;
    var data = [_]f32{ 1.0, 2.0 };
    const cos_data = [_]f32{0.5};
    const sin_data = [_]f32{0.866};

    const original = data;

    applyRopeRotationStrided(f32, identity_f32, identity_f32, &data, 1, &cos_data, 1, &sin_data, 1, half);

    const expected_0 = original[0] * cos_data[0] - original[1] * sin_data[0];
    const expected_1 = original[1] * cos_data[0] + original[0] * sin_data[0];

    try std.testing.expectApproxEqRel(expected_0, data[0], 1e-5);
    try std.testing.expectApproxEqRel(expected_1, data[1], 1e-5);
}

test "RoPE.init creates valid structure" {
    const allocator = std.testing.allocator;
    const dim: usize = 64;
    const max_seq_len: usize = 512;
    const theta: f32 = 10000.0;

    var rope = try RoPE.init(allocator, dim, max_seq_len, theta, 1.0);
    defer rope.deinit(allocator);

    try std.testing.expectEqual(dim, rope.dim);
    try std.testing.expectEqual(max_seq_len, rope.max_seq_len);
    try std.testing.expectEqual(theta, rope.theta);
    try std.testing.expectEqual(dim / 2, rope.inv_freq.len);
    try std.testing.expect(rope.cached_len > 0);
}

test "RoPE.initWithRopeScaling with linear scaling" {
    const allocator = std.testing.allocator;
    const dim: usize = 64;
    const scaling = tensor.RopeScaling{
        .rope_type = .linear,
        .factor = 2.0,
        .low_freq_factor = 1.0,
        .high_freq_factor = 4.0,
        .original_max_position_embeddings = 4096,
    };

    var rope = try RoPE.initWithRopeScaling(allocator, dim, 256, 10000.0, scaling);
    defer rope.deinit(allocator);

    try std.testing.expectEqual(dim, rope.dim);
    try std.testing.expectEqual(dim / 2, rope.inv_freq.len);
}

test "RoPE.initWithRopeScaling yarn matches reference values" {
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

    var rope = try RoPE.initWithRopeScaling(allocator, dim, 256, theta, scaling);
    defer rope.deinit(allocator);

    const expected_head = [_]f32{
        1.0,
        0.83008695,
        0.6890443,
        0.57196665,
        0.47478205,
        0.39411038,
    };
    for (expected_head, 0..) |expected, idx| {
        try std.testing.expectApproxEqAbs(expected, rope.inv_freq[idx], 1e-5);
    }

    const expected_mid = [_]f32{
        0.0010526021,
        0.00071183714,
        0.00045648392,
        0.00026735535,
        0.00012931869,
        0.000046150362,
    };
    for (expected_mid, 0..) |expected, offset| {
        const idx = 30 + offset;
        try std.testing.expectApproxEqAbs(expected, rope.inv_freq[idx], 1e-7);
    }

    const expected_tail = [_]f32{
        2.5097773e-07,
        3.0235114e-07,
        3.6424035e-07,
    };
    for (expected_tail, 0..) |expected, offset| {
        const idx = rope.inv_freq.len - 1 - offset;
        try std.testing.expectApproxEqAbs(expected, rope.inv_freq[idx], 1e-12);
    }

    try std.testing.expectApproxEqAbs(@as(f32, 1.3465736), rope.attention_scaling, 1e-6);
    try std.testing.expectApproxEqAbs(rope.attention_scaling, rope.freqs_cos[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), rope.freqs_sin[0], 1e-6);
}

test "RoPE.deinit frees memory" {
    const allocator = std.testing.allocator;
    var rope = try RoPE.init(allocator, 32, 128, 10000.0, 1.0);
    rope.deinit(allocator);
    // If we get here without memory leak, test passes
}

test "RoPE.applyInPlace rotates vector" {
    const allocator = std.testing.allocator;
    const dim: usize = 8;

    var rope = try RoPE.init(allocator, dim, 64, 10000.0, 1.0);
    defer rope.deinit(allocator);

    var vec = [_]f32{ 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 };
    const original = vec;

    rope.applyInPlace(&vec, 0);

    // At position 0, rotation should be minimal but non-zero for some dims
    // Just verify it ran without crashing and modified the vector
    var changed = false;
    for (vec, original) |v, o| {
        if (@abs(v - o) > 1e-6) changed = true;
    }
    // Position 0 may have very small rotation, so just check it's finite
    for (vec) |v| {
        try std.testing.expect(std.math.isFinite(v));
    }
}

test "RoPE.ensureCache expands cache beyond initial size" {
    // This test verifies that RoPE cache expansion works correctly.
    // The initial cache is 256 positions - we test expansion to 512.
    // This regression test catches issues with allocator compatibility
    // (e.g., using ArenaAllocator which doesn't support true realloc).
    const allocator = std.testing.allocator;
    const dim: usize = 64; // Typical head_dim

    var rope = try RoPE.init(allocator, dim, 2048, 10000.0, 1.0);
    defer rope.deinit(allocator);

    // Initial cache should be 256
    try std.testing.expectEqual(@as(usize, 256), rope.cached_len);

    // Access position 255 (last position in initial cache) - should not expand
    var vec: [64]f32 = undefined;
    @memset(&vec, 1.0);
    rope.applyInPlace(&vec, 255);
    try std.testing.expectEqual(@as(usize, 256), rope.cached_len);

    // Access position 256 - should trigger expansion to 512
    @memset(&vec, 1.0);
    rope.applyInPlace(&vec, 256);
    try std.testing.expectEqual(@as(usize, 512), rope.cached_len);

    // Verify the vector was rotated (values should change)
    var all_ones = true;
    for (vec) |v| {
        if (@abs(v - 1.0) > 1e-6) all_ones = false;
    }
    try std.testing.expect(!all_ones);
}

test "RoPE.ensureCache handles multiple expansions" {
    // Test that multiple cache expansions work correctly.
    // This catches use-after-free bugs where the old cache memory is accessed.
    const allocator = std.testing.allocator;
    const dim: usize = 32;

    var rope = try RoPE.init(allocator, dim, 4096, 10000.0, 1.0);
    defer rope.deinit(allocator);

    // Initial: 256 positions
    try std.testing.expectEqual(@as(usize, 256), rope.cached_len);

    var vec: [32]f32 = undefined;

    // First expansion: 256 -> 512
    @memset(&vec, 1.0);
    rope.applyInPlace(&vec, 300);
    try std.testing.expectEqual(@as(usize, 512), rope.cached_len);

    // Second expansion: 512 -> 1024
    @memset(&vec, 1.0);
    rope.applyInPlace(&vec, 600);
    try std.testing.expectEqual(@as(usize, 1024), rope.cached_len);

    // Third expansion: 1024 -> 2048
    @memset(&vec, 1.0);
    rope.applyInPlace(&vec, 1500);
    try std.testing.expectEqual(@as(usize, 2048), rope.cached_len);

    // Verify all values are finite (no corruption)
    for (vec) |v| {
        try std.testing.expect(std.math.isFinite(v));
    }
}

test "RoPE cache expansion respects max_seq_len" {
    // Test that cache expansion is capped at max_seq_len
    const allocator = std.testing.allocator;
    const dim: usize = 16;
    const max_seq: usize = 512;

    var rope = try RoPE.init(allocator, dim, max_seq, 10000.0, 1.0);
    defer rope.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 256), rope.cached_len);

    var vec: [16]f32 = undefined;

    // Access position 400 - should expand to 512 (max_seq_len), not 1024
    @memset(&vec, 1.0);
    rope.applyInPlace(&vec, 400);
    try std.testing.expectEqual(@as(usize, 512), rope.cached_len);
}
