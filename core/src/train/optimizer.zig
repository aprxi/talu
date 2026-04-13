//! AdamW optimizer with decoupled weight decay.
//!
//! Implements the AdamW algorithm:
//!   m = beta1 * m + (1 - beta1) * grad
//!   v = beta2 * v + (1 - beta2) * grad^2
//!   m_hat = m / (1 - beta1^t)
//!   v_hat = v / (1 - beta2^t)
//!   param -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)

const std = @import("std");
const compute = @import("compute_pkg");

const simd = compute.cpu.simd.arch;
const VEC = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

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
        const t = self.step_count + 1;
        self.stepAt(params, grads, state, lr, t);
        self.step_count = t;
    }

    /// Perform one optimization step using an explicit logical step index.
    ///
    /// This is used by FullTrainingSession so all parameter tensors in the
    /// model use the same Adam bias-correction step within a training step.
    pub fn stepAt(
        self: *const AdamW,
        params: []f32,
        grads: []const f32,
        state: *ParamState,
        lr: f32,
        t: u64,
    ) void {
        @setFloatMode(.optimized);
        std.debug.assert(params.len == grads.len);
        std.debug.assert(params.len == state.m.len);
        std.debug.assert(params.len == state.v.len);

        const beta1 = self.config.beta1;
        const beta2 = self.config.beta2;
        const eps = self.config.eps;
        const wd = self.config.weight_decay;

        // Bias correction
        const bc1 = 1.0 - std.math.pow(f32, beta1, @floatFromInt(t));
        const bc2 = 1.0 - std.math.pow(f32, beta2, @floatFromInt(t));
        const inv_bc1 = 1.0 / bc1;
        const inv_bc2 = 1.0 / bc2;

        const n = params.len;

        // SIMD path
        const b1_v: F32Vec = @splat(beta1);
        const b2_v: F32Vec = @splat(beta2);
        const one_minus_b1_v: F32Vec = @splat(1.0 - beta1);
        const one_minus_b2_v: F32Vec = @splat(1.0 - beta2);
        const inv_bc1_v: F32Vec = @splat(inv_bc1);
        const inv_bc2_v: F32Vec = @splat(inv_bc2);
        const eps_v: F32Vec = @splat(eps);
        const lr_v: F32Vec = @splat(lr);
        const wd_v: F32Vec = @splat(wd);

        var i: usize = 0;
        while (i + VEC <= n) : (i += VEC) {
            var m_vec: F32Vec = state.m[i..][0..VEC].*;
            var v_vec: F32Vec = state.v[i..][0..VEC].*;
            const g_vec: F32Vec = grads[i..][0..VEC].*;
            var p_vec: F32Vec = params[i..][0..VEC].*;

            // m = beta1 * m + (1-beta1) * g
            m_vec = @mulAdd(F32Vec, b1_v, m_vec, one_minus_b1_v * g_vec);
            // v = beta2 * v + (1-beta2) * g^2
            v_vec = @mulAdd(F32Vec, b2_v, v_vec, one_minus_b2_v * g_vec * g_vec);

            state.m[i..][0..VEC].* = m_vec;
            state.v[i..][0..VEC].* = v_vec;

            // Bias-corrected: m_hat / (sqrt(v_hat) + eps)
            const m_hat = m_vec * inv_bc1_v;
            const v_hat = v_vec * inv_bc2_v;
            const update = m_hat / (@sqrt(v_hat) + eps_v) + wd_v * p_vec;
            p_vec -= lr_v * update;

            params[i..][0..VEC].* = p_vec;
        }

        // Scalar tail
        while (i < n) : (i += 1) {
            var m = state.m[i];
            var v = state.v[i];
            const g = grads[i];

            m = beta1 * m + (1.0 - beta1) * g;
            v = beta2 * v + (1.0 - beta2) * g * g;

            state.m[i] = m;
            state.v[i] = v;

            const m_hat = m * inv_bc1;
            const v_hat = v * inv_bc2;

            params[i] -= lr * (m_hat / (@sqrt(v_hat) + eps) + wd * params[i]);
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

test "AdamW stepAt does not mutate step_count" {
    const allocator = std.testing.allocator;
    var optimizer = AdamW.init(.{});

    var params = [_]f32{1.0};
    const grads = [_]f32{0.1};
    var state = try ParamState.init(allocator, 1);
    defer state.deinit();

    optimizer.stepAt(&params, &grads, &state, 1e-3, 4);
    try std.testing.expectEqual(@as(u64, 0), optimizer.step_count);
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
