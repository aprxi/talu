//! Softmax convenience wrappers for CPU kernels.
//!
//! Canonical SIMD/math implementation lives in `math_primitives/softmax.zig`.

const std = @import("std");
const softmax_impl = @import("math_primitives/softmax.zig");

/// Numerically-stable softmax in-place over one contiguous row.
pub fn stableInPlace(values: []f32) void {
    if (values.len == 0) return;
    softmax_impl.softmaxContiguous(values, values, 1, values.len);
}

/// In-place masked softmax over `scores` with explicit active interval.
pub fn maskedInPlaceWithMax(
    scores: []f32,
    active_start: usize,
    active_end: usize,
    sink_logit: ?f32,
    exact: bool,
    maxv: f32,
    mask_cutoff: ?f32,
) void {
    softmax_impl.softmaxMaskedInPlaceWithMax(
        scores,
        active_start,
        active_end,
        sink_logit,
        exact,
        maxv,
        mask_cutoff,
    );
}

test "stableInPlace normalizes row to 1" {
    var v = [_]f32{ 1.0, 2.0, 3.0 };
    stableInPlace(&v);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), v[0] + v[1] + v[2], 1e-6);
}

test "maskedInPlaceWithMax zeroes masked region and normalizes active window" {
    var scores = [_]f32{ 1, 2, 3, 4, 5, 6 };
    maskedInPlaceWithMax(&scores, 2, 5, null, true, 5.0, null);
    try std.testing.expectEqual(@as(f32, 0.0), scores[0]);
    try std.testing.expectEqual(@as(f32, 0.0), scores[1]);
    try std.testing.expectEqual(@as(f32, 0.0), scores[5]);
    const active_sum = scores[2] + scores[3] + scores[4];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), active_sum, 1e-6);
}

