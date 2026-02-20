//! Integration tests for compute.ops.math.RoPE
//!
//! RoPE (Rotary Position Embedding) is a position encoding scheme used by
//! transformer models. It precomputes cos/sin caches that grow lazily as
//! longer sequences are processed. These tests verify lifecycle, cache
//! growth, rotation correctness, and scaling variants.

const std = @import("std");
const main = @import("main");

const RoPE = main.compute.cpu.math.RoPE;

// ===== init / deinit =====

test "RoPE: init and deinit do not leak" {
    var rope = try RoPE.init(std.testing.allocator, 8, 512, 10000.0, 1.0);
    defer rope.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 8), rope.dim);
    try std.testing.expectEqual(@as(usize, 512), rope.max_seq_len);
    try std.testing.expectEqual(@as(f32, 10000.0), rope.theta);
    try std.testing.expectEqual(@as(f32, 1.0), rope.attention_scaling);
}

test "RoPE: init with small max_seq_len caps initial cache" {
    // When max_seq_len < 256, initial cache should be capped to max_seq_len
    var rope = try RoPE.init(std.testing.allocator, 4, 32, 10000.0, 1.0);
    defer rope.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 32), rope.cached_len);
}

test "RoPE: init with large max_seq_len starts at 256" {
    var rope = try RoPE.init(std.testing.allocator, 8, 4096, 10000.0, 1.0);
    defer rope.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 256), rope.cached_len);
}

// ===== applyInPlace =====

test "RoPE: applyInPlace at position 0 rotates vector" {
    var rope = try RoPE.init(std.testing.allocator, 4, 512, 10000.0, 1.0);
    defer rope.deinit(std.testing.allocator);

    // Position 0: cos(0)=1, sin(0)=0 → rotation is identity
    var vec = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const original = vec;
    rope.applyInPlace(&vec, 0);

    // At pos=0, angle=0 for all dims → cos=1, sin=0 → vec unchanged
    for (vec, original) |v, o| {
        try std.testing.expectApproxEqAbs(o, v, 1e-5);
    }
}

test "RoPE: applyInPlace at nonzero position changes vector" {
    var rope = try RoPE.init(std.testing.allocator, 4, 512, 10000.0, 1.0);
    defer rope.deinit(std.testing.allocator);

    var vec = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    rope.applyInPlace(&vec, 5);

    // At pos=5 the rotation should change the vector
    // vec[0] and vec[2] form a rotation pair
    try std.testing.expect(vec[0] != 1.0 or vec[2] != 0.0);
}

test "RoPE: applyInPlace preserves vector magnitude" {
    var rope = try RoPE.init(std.testing.allocator, 8, 512, 10000.0, 1.0);
    defer rope.deinit(std.testing.allocator);

    var vec = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

    // Compute original magnitude per rotation pair
    const half = vec.len / 2;
    var orig_mags: [4]f32 = undefined;
    for (0..half) |i| {
        orig_mags[i] = vec[i] * vec[i] + vec[i + half] * vec[i + half];
    }

    rope.applyInPlace(&vec, 10);

    // Magnitude of each (x_i, x_{i+half}) pair should be preserved
    for (0..half) |i| {
        const new_mag = vec[i] * vec[i] + vec[i + half] * vec[i + half];
        try std.testing.expectApproxEqRel(orig_mags[i], new_mag, 1e-4);
    }
}

// ===== cache growth =====

test "RoPE: accessing beyond initial cache triggers growth" {
    var rope = try RoPE.init(std.testing.allocator, 4, 4096, 10000.0, 1.0);
    defer rope.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 256), rope.cached_len);

    // Access position 300 → triggers cache growth
    var vec = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    rope.applyInPlace(&vec, 300);

    // Cache should have grown (power-of-2 growth: 256 → 512)
    try std.testing.expect(rope.cached_len >= 301);
    try std.testing.expectEqual(@as(usize, 512), rope.cached_len);
}

// ===== getCos / getSin =====

test "RoPE: getCos and getSin return dim/2 slices" {
    var rope = try RoPE.init(std.testing.allocator, 8, 512, 10000.0, 1.0);
    defer rope.deinit(std.testing.allocator);

    const cos = rope.getCos(0);
    const sin = rope.getSin(0);

    try std.testing.expectEqual(@as(usize, 4), cos.len);
    try std.testing.expectEqual(@as(usize, 4), sin.len);
}

test "RoPE: getCos/getSin at position 0 returns cos(0)=1, sin(0)=0" {
    var rope = try RoPE.init(std.testing.allocator, 4, 512, 10000.0, 1.0);
    defer rope.deinit(std.testing.allocator);

    const cos = rope.getCos(0);
    const sin = rope.getSin(0);

    for (cos) |c| try std.testing.expectApproxEqAbs(@as(f32, 1.0), c, 1e-5);
    for (sin) |s| try std.testing.expectApproxEqAbs(@as(f32, 0.0), s, 1e-5);
}

test "RoPE: getCos/getSin triggers cache growth for distant position" {
    var rope = try RoPE.init(std.testing.allocator, 4, 2048, 10000.0, 1.0);
    defer rope.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 256), rope.cached_len);

    _ = rope.getCos(500);

    try std.testing.expect(rope.cached_len >= 501);
}

// ===== applyInterleavedInPlace =====

test "RoPE: applyInterleavedInPlace at position 0 is identity" {
    var rope = try RoPE.init(std.testing.allocator, 4, 512, 10000.0, 1.0);
    defer rope.deinit(std.testing.allocator);

    var vec = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const original = vec;
    rope.applyInterleavedInPlace(&vec, 0);

    for (vec, original) |v, o| {
        try std.testing.expectApproxEqAbs(o, v, 1e-5);
    }
}

test "RoPE: applyInterleavedInPlace preserves pair magnitudes" {
    var rope = try RoPE.init(std.testing.allocator, 4, 512, 10000.0, 1.0);
    defer rope.deinit(std.testing.allocator);

    var vec = [_]f32{ 3.0, 4.0, 1.0, 2.0 };

    // Original magnitudes of interleaved pairs
    const mag0 = vec[0] * vec[0] + vec[1] * vec[1]; // (3,4) → 25
    const mag1 = vec[2] * vec[2] + vec[3] * vec[3]; // (1,2) → 5

    rope.applyInterleavedInPlace(&vec, 7);

    const new_mag0 = vec[0] * vec[0] + vec[1] * vec[1];
    const new_mag1 = vec[2] * vec[2] + vec[3] * vec[3];

    try std.testing.expectApproxEqRel(mag0, new_mag0, 1e-4);
    try std.testing.expectApproxEqRel(mag1, new_mag1, 1e-4);
}

// ===== initFromInvFreq =====

test "RoPE: initFromInvFreq with explicit table initializes successfully" {
    const inv_freq = [_]f32{ 1.0, 0.5, 0.25, 0.125 };
    var rope = try RoPE.initFromInvFreq(std.testing.allocator, 8, 4096, &inv_freq, 1.0);
    defer rope.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 8), rope.dim);
    try std.testing.expectEqual(@as(f32, 1.0), rope.attention_scaling);
}

test "RoPE: initFromInvFreq applies attention scaling to cached tables" {
    const inv_freq = [_]f32{ 1.0, 0.5 };
    var rope = try RoPE.initFromInvFreq(std.testing.allocator, 4, 512, &inv_freq, 1.25);
    defer rope.deinit(std.testing.allocator);

    try std.testing.expectApproxEqAbs(@as(f32, 1.25), rope.freqs_cos[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), rope.freqs_sin[0], 1e-6);
}

test "RoPE: initFromInvFreq with standard table is equivalent to default init" {
    var rope_default = try RoPE.init(std.testing.allocator, 8, 512, 10000.0, 1.0);
    defer rope_default.deinit(std.testing.allocator);

    const theta: f32 = 10000.0;
    const dim: usize = 8;
    const half_dim = dim / 2;
    var inv_freq: [half_dim]f32 = undefined;
    for (0..half_dim) |idx| {
        const exponent = @as(f64, @floatFromInt(2 * idx)) / @as(f64, @floatFromInt(dim));
        inv_freq[idx] = @floatCast(1.0 / std.math.pow(f64, @as(f64, theta), exponent));
    }
    var rope_from_table = try RoPE.initFromInvFreq(std.testing.allocator, dim, 512, &inv_freq, 1.0);
    defer rope_from_table.deinit(std.testing.allocator);

    // Both should produce the same cos/sin values
    const cos_default = rope_default.getCos(3);
    const cos_from_table = rope_from_table.getCos(3);

    for (cos_default, cos_from_table) |d, n| {
        try std.testing.expectApproxEqRel(d, n, 1e-5);
    }
}

// ===== inv_freq_scale =====

test "RoPE: inv_freq_scale affects frequency computation" {
    var rope_normal = try RoPE.init(std.testing.allocator, 8, 512, 10000.0, 1.0);
    defer rope_normal.deinit(std.testing.allocator);

    var rope_scaled = try RoPE.init(std.testing.allocator, 8, 512, 10000.0, 0.5);
    defer rope_scaled.deinit(std.testing.allocator);

    // At position > 0, different scaling should produce different values
    const cos_normal = rope_normal.getCos(10);
    const cos_scaled = rope_scaled.getCos(10);

    var any_different = false;
    for (cos_normal, cos_scaled) |n, s| {
        if (@abs(n - s) > 1e-5) {
            any_different = true;
            break;
        }
    }
    try std.testing.expect(any_different);
}
