//! Training loop orchestration.
//!
//! Ties together forward pass, backward pass, gradient clipping,
//! optimizer step, and logging. The loop operates on a LoRA adapter
//! applied to a frozen base model.

const std = @import("std");
const grad_mod = @import("grad.zig");
const optimizer_mod = @import("optimizer.zig");
const scheduler_mod = @import("scheduler.zig");
const loss_mod = @import("loss.zig");

const GradTensor = grad_mod.GradTensor;
const AdamW = optimizer_mod.AdamW;
const ParamState = optimizer_mod.ParamState;
const Scheduler = scheduler_mod.Scheduler;
const Allocator = std.mem.Allocator;

/// Configuration for a training run.
pub const TrainingConfig = struct {
    /// Maximum gradient norm for clipping.
    max_grad_norm: f32 = 1.0,
    /// Gradient accumulation steps before optimizer update.
    gradient_accumulation_steps: u32 = 1,
    /// Log metrics every N steps.
    log_interval: u32 = 10,
    /// Save checkpoint every N steps.
    save_interval: u32 = 100,
    /// Total training steps.
    total_steps: u64 = 1000,
};

/// Step metrics returned after each optimizer step.
pub const StepMetrics = struct {
    loss: f32,
    learning_rate: f32,
    grad_norm: f32,
    step: u64,
};

/// Clip gradient norm across all parameter groups.
///
/// If the global L2 norm exceeds max_norm, all gradients are scaled down
/// proportionally so the global norm equals max_norm.
///
/// Returns the original (unclipped) global norm.
pub fn clipGradNorm(grads: []GradTensor, max_norm: f32) f32 {
    // Compute global L2 norm
    var global_norm_sq: f32 = 0.0;
    for (grads) |*g| {
        const n = g.norm();
        global_norm_sq += n * n;
    }
    const global_norm = @sqrt(global_norm_sq);

    // Clip if needed
    if (global_norm > max_norm and global_norm > 0.0) {
        const scale = max_norm / global_norm;
        for (grads) |*g| {
            g.scale(scale);
        }
    }

    return global_norm;
}

// =============================================================================
// Tests
// =============================================================================

test "clipGradNorm does not clip below threshold" {
    const allocator = std.testing.allocator;
    var g = try GradTensor.init(allocator, &.{3});
    defer g.deinit();

    const data = g.asSliceMut();
    data[0] = 0.1;
    data[1] = 0.1;
    data[2] = 0.1;

    var grads = [_]GradTensor{g};
    const norm = clipGradNorm(&grads, 10.0);

    // Norm is small, should not clip
    try std.testing.expect(norm < 10.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), grads[0].asSlice()[0], 1e-6);
}

test "clipGradNorm clips above threshold" {
    const allocator = std.testing.allocator;
    var g = try GradTensor.init(allocator, &.{2});
    defer g.deinit();

    const data = g.asSliceMut();
    data[0] = 3.0;
    data[1] = 4.0;
    // norm = 5.0

    var grads = [_]GradTensor{g};
    const norm = clipGradNorm(&grads, 1.0);

    try std.testing.expectApproxEqAbs(@as(f32, 5.0), norm, 1e-5);

    // After clipping to max_norm=1.0, new norm should be ~1.0
    const new_norm = grads[0].norm();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), new_norm, 1e-4);
}

test "clipGradNorm global norm across multiple tensors" {
    const allocator = std.testing.allocator;
    var g1 = try GradTensor.init(allocator, &.{1});
    defer g1.deinit();
    var g2 = try GradTensor.init(allocator, &.{1});
    defer g2.deinit();

    g1.asSliceMut()[0] = 3.0;
    g2.asSliceMut()[0] = 4.0;
    // global norm = sqrt(9 + 16) = 5

    var grads = [_]GradTensor{ g1, g2 };
    const norm = clipGradNorm(&grads, 5.0);

    try std.testing.expectApproxEqAbs(@as(f32, 5.0), norm, 1e-5);
    // Should not clip (exactly at threshold)
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), grads[0].asSlice()[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), grads[1].asSlice()[0], 1e-5);
}

test "TrainingConfig defaults" {
    const cfg = TrainingConfig{};
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cfg.max_grad_norm, 1e-6);
    try std.testing.expectEqual(@as(u32, 1), cfg.gradient_accumulation_steps);
}

test "StepMetrics fields" {
    const metrics = StepMetrics{
        .loss = 2.5,
        .learning_rate = 1e-4,
        .grad_norm = 0.8,
        .step = 42,
    };
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), metrics.loss, 1e-6);
    try std.testing.expectEqual(@as(u64, 42), metrics.step);
}
