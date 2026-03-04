//! AdamW optimizer with decoupled weight decay.
//!
//! Implements the AdamW algorithm:
//!   m = beta1 * m + (1 - beta1) * grad
//!   v = beta2 * v + (1 - beta2) * grad^2
//!   m_hat = m / (1 - beta1^t)
//!   v_hat = v / (1 - beta2^t)
//!   param -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)

const std = @import("std");

pub const AdamWConfig = struct {
    lr: f32 = 1e-4,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    weight_decay: f32 = 0.01,
};

/// Per-parameter optimizer state (first and second moment estimates).
pub const ParamState = struct {
    /// First moment (mean of gradients).
    m: []f32,
    /// Second moment (mean of squared gradients).
    v: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n: usize) !ParamState {
        const m = try allocator.alloc(f32, n);
        errdefer allocator.free(m);
        const v = try allocator.alloc(f32, n);
        @memset(m, 0);
        @memset(v, 0);
        return .{ .m = m, .v = v, .allocator = allocator };
    }

    pub fn deinit(self: *ParamState) void {
        self.allocator.free(self.m);
        self.allocator.free(self.v);
        self.* = undefined;
    }
};

pub const AdamW = struct {
    config: AdamWConfig,
    step_count: u64,

    pub fn init(config: AdamWConfig) AdamW {
        return .{ .config = config, .step_count = 0 };
    }

    /// Perform one optimization step on a parameter.
    ///
    /// params: the parameter values to update (modified in-place)
    /// grads:  the computed gradients
    /// state:  per-parameter optimizer state (m, v)
    /// lr:     learning rate (allows external scheduling)
    pub fn step(self: *AdamW, params: []f32, grads: []const f32, state: *ParamState, lr: f32) void {
        std.debug.assert(params.len == grads.len);
        std.debug.assert(params.len == state.m.len);
        std.debug.assert(params.len == state.v.len);

        self.step_count += 1;
        const t = self.step_count;

        const beta1 = self.config.beta1;
        const beta2 = self.config.beta2;
        const eps = self.config.eps;
        const wd = self.config.weight_decay;

        // Bias correction
        const bc1 = 1.0 - std.math.pow(f32, beta1, @floatFromInt(t));
        const bc2 = 1.0 - std.math.pow(f32, beta2, @floatFromInt(t));

        for (params, grads, state.m, state.v) |*p, g, *m, *v| {
            // Update moments
            m.* = beta1 * m.* + (1.0 - beta1) * g;
            v.* = beta2 * v.* + (1.0 - beta2) * g * g;

            // Bias-corrected estimates
            const m_hat = m.* / bc1;
            const v_hat = v.* / bc2;

            // AdamW update: decoupled weight decay
            p.* -= lr * (m_hat / (@sqrt(v_hat) + eps) + wd * p.*);
        }
    }
};

// =============================================================================
// Tests
// =============================================================================

test "AdamW step decreases loss on simple gradient" {
    const allocator = std.testing.allocator;
    var optimizer = AdamW.init(.{ .lr = 0.1, .weight_decay = 0.0 });

    var params = [_]f32{5.0};
    const grads = [_]f32{1.0}; // positive gradient => decrease param
    var state = try ParamState.init(allocator, 1);
    defer state.deinit();

    const initial = params[0];
    optimizer.step(&params, &grads, &state, 0.1);

    try std.testing.expect(params[0] < initial);
}

test "AdamW step count increments" {
    const allocator = std.testing.allocator;
    var optimizer = AdamW.init(.{});

    var params = [_]f32{1.0};
    const grads = [_]f32{0.1};
    var state = try ParamState.init(allocator, 1);
    defer state.deinit();

    optimizer.step(&params, &grads, &state, 1e-3);
    try std.testing.expectEqual(@as(u64, 1), optimizer.step_count);

    optimizer.step(&params, &grads, &state, 1e-3);
    try std.testing.expectEqual(@as(u64, 2), optimizer.step_count);
}

test "AdamW weight decay shrinks parameters" {
    const allocator = std.testing.allocator;
    var optimizer = AdamW.init(.{ .lr = 0.01, .weight_decay = 0.1 });

    var params = [_]f32{10.0};
    const grads = [_]f32{0.0}; // zero gradient — only weight decay acts
    var state = try ParamState.init(allocator, 1);
    defer state.deinit();

    optimizer.step(&params, &grads, &state, 0.01);

    // Weight decay should shrink the parameter
    try std.testing.expect(params[0] < 10.0);
}

test "AdamW momentum accumulates" {
    const allocator = std.testing.allocator;
    var optimizer = AdamW.init(.{ .lr = 0.01, .weight_decay = 0.0, .beta1 = 0.9 });

    var params = [_]f32{0.0};
    const grads = [_]f32{1.0};
    var state = try ParamState.init(allocator, 1);
    defer state.deinit();

    // After step 1
    optimizer.step(&params, &grads, &state, 0.01);
    const after_1 = params[0];

    // After step 2 with same gradient, should move further per step
    optimizer.step(&params, &grads, &state, 0.01);
    const after_2 = params[0];

    // Both steps should move in negative direction
    try std.testing.expect(after_1 < 0.0);
    try std.testing.expect(after_2 < after_1);
}

test "ParamState init zeros moments" {
    const allocator = std.testing.allocator;
    var state = try ParamState.init(allocator, 4);
    defer state.deinit();

    for (state.m) |v| try std.testing.expectEqual(@as(f32, 0.0), v);
    for (state.v) |v| try std.testing.expectEqual(@as(f32, 0.0), v);
}
