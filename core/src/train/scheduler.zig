//! Learning rate schedulers.
//!
//! Provides cosine annealing with optional linear warmup, matching
//! the scheduling used in typical LLM training.

const std = @import("std");

pub const SchedulerConfig = struct {
    /// Maximum learning rate (after warmup).
    max_lr: f32 = 1e-4,
    /// Minimum learning rate (at end of cosine decay).
    min_lr: f32 = 1e-6,
    /// Number of warmup steps (linear ramp from 0 to max_lr).
    warmup_steps: u64 = 0,
    /// Total number of training steps.
    total_steps: u64,
};

pub const Scheduler = struct {
    config: SchedulerConfig,

    pub fn init(config: SchedulerConfig) Scheduler {
        return .{ .config = config };
    }

    /// Get the learning rate for a given step.
    ///
    /// During warmup: linear ramp from 0 to max_lr.
    /// After warmup: cosine decay from max_lr to min_lr.
    pub fn getLr(self: *const Scheduler, step: u64) f32 {
        const cfg = self.config;

        if (step < cfg.warmup_steps) {
            // Linear warmup
            const progress = @as(f32, @floatFromInt(step)) / @as(f32, @floatFromInt(cfg.warmup_steps));
            return cfg.max_lr * progress;
        }

        if (step >= cfg.total_steps) {
            return cfg.min_lr;
        }

        // Cosine decay
        const decay_steps = cfg.total_steps - cfg.warmup_steps;
        const current = step - cfg.warmup_steps;
        const progress = @as(f32, @floatFromInt(current)) / @as(f32, @floatFromInt(decay_steps));
        const cosine = 0.5 * (1.0 + @cos(std.math.pi * progress));

        return cfg.min_lr + (cfg.max_lr - cfg.min_lr) * cosine;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "Scheduler warmup linear ramp" {
    const sched = Scheduler.init(.{
        .max_lr = 1.0,
        .min_lr = 0.0,
        .warmup_steps = 10,
        .total_steps = 100,
    });

    // Step 0: lr = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sched.getLr(0), 1e-6);

    // Step 5: lr = 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sched.getLr(5), 1e-6);

    // Step 10 (end of warmup): lr = max_lr
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sched.getLr(10), 1e-6);
}

test "Scheduler cosine decay" {
    const sched = Scheduler.init(.{
        .max_lr = 1.0,
        .min_lr = 0.0,
        .warmup_steps = 0,
        .total_steps = 100,
    });

    // Step 0: max_lr
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sched.getLr(0), 1e-5);

    // Step 50: midpoint of cosine = (max+min)/2
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sched.getLr(50), 1e-5);

    // Step 100: min_lr
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sched.getLr(100), 1e-5);
}

test "Scheduler beyond total_steps returns min_lr" {
    const sched = Scheduler.init(.{
        .max_lr = 1.0,
        .min_lr = 0.1,
        .warmup_steps = 0,
        .total_steps = 50,
    });

    try std.testing.expectApproxEqAbs(@as(f32, 0.1), sched.getLr(100), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), sched.getLr(1000), 1e-6);
}

test "Scheduler warmup then decay" {
    const sched = Scheduler.init(.{
        .max_lr = 0.001,
        .min_lr = 0.0001,
        .warmup_steps = 100,
        .total_steps = 1100,
    });

    // During warmup
    const lr_50 = sched.getLr(50);
    try std.testing.expect(lr_50 > 0.0);
    try std.testing.expect(lr_50 < 0.001);

    // At warmup boundary
    try std.testing.expectApproxEqAbs(@as(f32, 0.001), sched.getLr(100), 1e-6);

    // During decay: should be between min and max
    const lr_600 = sched.getLr(600);
    try std.testing.expect(lr_600 >= 0.0001);
    try std.testing.expect(lr_600 <= 0.001);

    // Monotonically decreasing after warmup
    const lr_200 = sched.getLr(200);
    const lr_800 = sched.getLr(800);
    try std.testing.expect(lr_200 > lr_800);
}
