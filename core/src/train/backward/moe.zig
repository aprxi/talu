//! Mixture-of-Experts backward pass.
//!
//! MoE forward: for each token, select top-k experts via softmax router,
//! run each expert's SwiGLU FFN, then combine with softmax weights.
//!
//! Backward decomposes into:
//!   1. Router gradient (softmax over top-k logits)
//!   2. Per-expert FFN gradient (reuses linear + activation backward)
//!   3. Expert weight gradient (dot product of grad_output and expert output)

const std = @import("std");

/// Compute MoE router backward: gradient through softmax-normalized top-k weights.
///
/// Given the gradient w.r.t. expert weights (dL/d_weights), compute gradient
/// w.r.t. router logits. Only the selected (top-k) experts receive gradient.
///
/// grad_logits:     [num_experts]   — overwritten (zeros for unselected experts)
/// grad_weights:    [k]             — gradient w.r.t. softmax-normalized weights
/// expert_weights:  [k]             — saved softmax weights from forward
/// expert_indices:  [k]             — which experts were selected
/// num_experts:     total number of experts
pub fn routerBackward(
    grad_logits: []f32,
    grad_weights: []const f32,
    expert_weights: []const f32,
    expert_indices: []const u32,
    num_experts: usize,
) void {
    const k = expert_indices.len;
    std.debug.assert(grad_logits.len == num_experts);
    std.debug.assert(grad_weights.len == k);
    std.debug.assert(expert_weights.len == k);

    // Zero unselected experts
    @memset(grad_logits, 0);

    // Softmax backward: d_logits[i] = w[i] * (d_w[i] - sum_j(w[j] * d_w[j]))
    var dot_wd: f32 = 0.0;
    for (0..k) |i| {
        dot_wd += expert_weights[i] * grad_weights[i];
    }

    for (0..k) |i| {
        const idx: usize = expert_indices[i];
        if (idx < num_experts) {
            grad_logits[idx] = expert_weights[i] * (grad_weights[i] - dot_wd);
        }
    }
}

/// Compute gradient w.r.t. expert weights from the weighted sum of expert outputs.
///
/// y = sum_i(w[i] * expert_out[i])
/// dL/d(w[i]) = dot(dL/dy, expert_out[i])
///
/// grad_weights:    [k]             — overwritten
/// grad_output:     [d_model]       — gradient from upstream
/// expert_outputs:  [k * d_model]   — saved expert outputs from forward
/// k:               number of selected experts
/// d_model:         model dimension
pub fn expertWeightGrad(
    grad_weights: []f32,
    grad_output: []const f32,
    expert_outputs: []const f32,
    k: usize,
    d_model: usize,
) void {
    std.debug.assert(grad_weights.len == k);
    std.debug.assert(grad_output.len == d_model);
    std.debug.assert(expert_outputs.len == k * d_model);

    for (0..k) |i| {
        var dot_val: f32 = 0.0;
        const expert_out = expert_outputs[i * d_model ..][0..d_model];
        for (grad_output, expert_out) |g, e| {
            dot_val += g * e;
        }
        grad_weights[i] = dot_val;
    }
}

/// Compute gradient w.r.t. each expert's output from the weighted combination.
///
/// y = sum_i(w[i] * expert_out[i])
/// dL/d(expert_out[i]) = w[i] * dL/dy
///
/// grad_expert_outputs: [k * d_model] — overwritten
/// grad_output:         [d_model]
/// expert_weights:      [k]
/// k:                   number of selected experts
/// d_model:             model dimension
pub fn expertOutputGrad(
    grad_expert_outputs: []f32,
    grad_output: []const f32,
    expert_weights: []const f32,
    k: usize,
    d_model: usize,
) void {
    std.debug.assert(grad_expert_outputs.len == k * d_model);
    std.debug.assert(grad_output.len == d_model);
    std.debug.assert(expert_weights.len == k);

    for (0..k) |i| {
        const w = expert_weights[i];
        const dest = grad_expert_outputs[i * d_model ..][0..d_model];
        for (dest, grad_output) |*d, g| {
            d.* = w * g;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

test "routerBackward softmax gradient sums to zero" {
    // With 4 experts, top-2 selected
    var grad_logits = [_]f32{ 0, 0, 0, 0 };
    const expert_weights = [_]f32{ 0.6, 0.4 };
    const grad_weights = [_]f32{ 1.0, -1.0 };
    const expert_indices = [_]u32{ 1, 3 };

    routerBackward(&grad_logits, &grad_weights, &expert_weights, &expert_indices, 4);

    // Unselected experts (0, 2) should be zero
    try std.testing.expectEqual(@as(f32, 0.0), grad_logits[0]);
    try std.testing.expectEqual(@as(f32, 0.0), grad_logits[2]);

    // Sum of all gradients should be close to zero (softmax property)
    var total: f32 = 0.0;
    for (grad_logits) |v| total += v;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), total, 1e-5);
}

test "routerBackward equal weights equal grad" {
    var grad_logits = [_]f32{ 0, 0 };
    const expert_weights = [_]f32{ 0.5, 0.5 };
    const grad_weights = [_]f32{ 1.0, 1.0 };
    const expert_indices = [_]u32{ 0, 1 };

    routerBackward(&grad_logits, &grad_weights, &expert_weights, &expert_indices, 2);

    // With equal weights and equal grad, softmax grad should be zero
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), grad_logits[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), grad_logits[1], 1e-5);
}

test "expertWeightGrad computes dot products" {
    var grad_w = [_]f32{ 0, 0 };
    const grad_output = [_]f32{ 1, 0, 0 };
    // Expert 0 output: [1, 0, 0], Expert 1 output: [0, 1, 0]
    const expert_outputs = [_]f32{ 1, 0, 0, 0, 1, 0 };

    expertWeightGrad(&grad_w, &grad_output, &expert_outputs, 2, 3);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), grad_w[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), grad_w[1], 1e-5);
}

test "expertOutputGrad scales by weight" {
    var grad_expert = [_]f32{ 0, 0, 0, 0 };
    const grad_output = [_]f32{ 2.0, 4.0 };
    const expert_weights = [_]f32{ 0.7, 0.3 };

    expertOutputGrad(&grad_expert, &grad_output, &expert_weights, 2, 2);

    // Expert 0: 0.7 * [2, 4] = [1.4, 2.8]
    try std.testing.expectApproxEqAbs(@as(f32, 1.4), grad_expert[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.8), grad_expert[1], 1e-5);
    // Expert 1: 0.3 * [2, 4] = [0.6, 1.2]
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), grad_expert[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.2), grad_expert[3], 1e-5);
}
