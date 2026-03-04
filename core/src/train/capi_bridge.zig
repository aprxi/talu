//! C API bridge for training session.
//!
//! Converts between C-compatible extern structs and Zig domain types.
//! This layer owns no domain logic — it validates, converts, and delegates
//! to session.zig.

const std = @import("std");
const session_mod = @import("session.zig");

const TrainingSession = session_mod.TrainingSession;
const SessionConfig = session_mod.SessionConfig;
const SessionInfo = session_mod.SessionInfo;
const SessionState = session_mod.State;
const StepMetrics = @import("loop.zig").StepMetrics;
const LoraAdapter = @import("adapter.zig").LoraAdapter;
const LoraConfig = @import("adapter.zig").LoraConfig;
const LoraLayer = @import("adapter.zig").LoraLayer;
const Allocator = std.mem.Allocator;

// =============================================================================
// C-Compatible Extern Structs
// =============================================================================

/// LoRA adapter configuration (C-compatible).
pub const CLoraConfig = extern struct {
    rank: u32 = 16,
    alpha: f32 = 32.0,
};

/// Training hyperparameters (C-compatible).
/// Must be initialized with std.mem.zeroes() on the C side.
pub const CTrainingConfig = extern struct {
    learning_rate: f32 = 1e-4,
    min_learning_rate: f32 = 1e-6,
    weight_decay: f32 = 0.01,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    epsilon: f32 = 1e-8,
    batch_size: u32 = 4,
    seq_len: u32 = 128,
    warmup_steps: u64 = 100,
    total_steps: u64 = 1000,
    max_grad_norm: f32 = 1.0,
    gradient_accumulation_steps: u32 = 1,
    log_interval: u32 = 10,
    save_interval: u32 = 0,
};

/// Per-step training metrics (C-compatible).
pub const CStepMetrics = extern struct {
    step: u64,
    loss: f32,
    learning_rate: f32,
    grad_norm: f32,
};

/// Session info snapshot (C-compatible).
pub const CTrainingInfo = extern struct {
    current_step: u64,
    total_steps: u64,
    trainable_params: u64,
    adapter_layers: u32,
    state: u8,
    _padding: [3]u8 = .{0} ** 3,
};

/// Training step callback. Returns 0 to continue, non-zero to cancel.
pub const CStepCallback = *const fn (
    metrics: *const CStepMetrics,
    user_data: ?*anyopaque,
) callconv(.c) i32;

// =============================================================================
// Type Conversion
// =============================================================================

/// Convert CTrainingConfig → SessionConfig.
pub fn toSessionConfig(c: *const CTrainingConfig) SessionConfig {
    return .{
        .learning_rate = c.learning_rate,
        .min_learning_rate = c.min_learning_rate,
        .weight_decay = c.weight_decay,
        .beta1 = c.beta1,
        .beta2 = c.beta2,
        .epsilon = c.epsilon,
        .batch_size = c.batch_size,
        .seq_len = c.seq_len,
        .warmup_steps = c.warmup_steps,
        .total_steps = c.total_steps,
        .max_grad_norm = c.max_grad_norm,
        .gradient_accumulation_steps = c.gradient_accumulation_steps,
        .log_interval = c.log_interval,
        .save_interval = c.save_interval,
    };
}

/// Convert CLoraConfig → LoraConfig.
pub fn toLoraConfig(c: *const CLoraConfig) LoraConfig {
    return .{
        .rank = c.rank,
        .alpha = c.alpha,
    };
}

/// Convert StepMetrics → CStepMetrics.
pub fn toCStepMetrics(m: StepMetrics) CStepMetrics {
    return .{
        .step = m.step,
        .loss = m.loss,
        .learning_rate = m.learning_rate,
        .grad_norm = m.grad_norm,
    };
}

/// Convert SessionInfo → CTrainingInfo.
pub fn toCTrainingInfo(info: SessionInfo) CTrainingInfo {
    return .{
        .current_step = info.current_step,
        .total_steps = info.total_steps,
        .trainable_params = @intCast(info.trainable_params),
        .adapter_layers = @intCast(info.adapter_layers),
        .state = @intFromEnum(info.state),
    };
}

// =============================================================================
// Bridge Functions
// =============================================================================

/// Create a new training session. Caller must free via destroySession.
pub fn createSession(allocator: Allocator) !*TrainingSession {
    const session = try allocator.create(TrainingSession);
    session.* = TrainingSession.init(allocator);
    return session;
}

/// Destroy a training session and free its resources.
pub fn destroySession(allocator: Allocator, session: *TrainingSession) void {
    session.deinit();
    allocator.destroy(session);
}

/// Configure the session with hyperparameters from a C struct.
pub fn configureSession(
    session: *TrainingSession,
    config: *const CTrainingConfig,
) !void {
    const zig_config = toSessionConfig(config);
    try session.configure(zig_config);
}

/// Load tokenized training data from a file path.
pub fn loadData(
    session: *TrainingSession,
    path: [*:0]const u8,
) !void {
    const path_slice = std.mem.sliceTo(path, 0);
    try session.loadData(path_slice);
}

/// Run the training loop with an optional C callback.
pub fn trainSession(
    session: *TrainingSession,
    c_callback: ?CStepCallback,
    c_user_data: ?*anyopaque,
) !void {
    if (c_callback) |cb| {
        var wrapper = CallbackWrapper{
            .c_callback = cb,
            .c_user_data = c_user_data,
        };
        try session.train(bridgeCallback, @ptrCast(&wrapper));
    } else {
        try session.train(null, null);
    }
}

/// Get session info as a C-compatible struct.
pub fn getInfo(session: *const TrainingSession) CTrainingInfo {
    return toCTrainingInfo(session.getInfo());
}

// =============================================================================
// Callback Wrapper
// =============================================================================

/// Bridges the Zig StepCallback signature to a C CStepCallback.
const CallbackWrapper = struct {
    c_callback: CStepCallback,
    c_user_data: ?*anyopaque,
};

fn bridgeCallback(metrics: StepMetrics, user_data: ?*anyopaque) i32 {
    const wrapper: *CallbackWrapper = @ptrCast(@alignCast(user_data));
    var c_metrics = toCStepMetrics(metrics);
    return wrapper.c_callback(&c_metrics, wrapper.c_user_data);
}

// =============================================================================
// Tests
// =============================================================================

test "CTrainingConfig defaults match SessionConfig defaults" {
    const c_cfg = CTrainingConfig{};
    const zig_cfg = toSessionConfig(&c_cfg);

    const default_zig = SessionConfig{};
    try std.testing.expectApproxEqAbs(default_zig.learning_rate, zig_cfg.learning_rate, 1e-10);
    try std.testing.expectApproxEqAbs(default_zig.beta1, zig_cfg.beta1, 1e-10);
    try std.testing.expectApproxEqAbs(default_zig.beta2, zig_cfg.beta2, 1e-10);
    try std.testing.expectEqual(default_zig.batch_size, zig_cfg.batch_size);
    try std.testing.expectEqual(default_zig.total_steps, zig_cfg.total_steps);
}

test "CLoraConfig defaults" {
    const c_cfg = CLoraConfig{};
    const zig_cfg = toLoraConfig(&c_cfg);
    try std.testing.expectEqual(@as(u32, 16), zig_cfg.rank);
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), zig_cfg.alpha, 1e-6);
}

test "toCStepMetrics preserves values" {
    const m = StepMetrics{
        .step = 42,
        .loss = 2.5,
        .learning_rate = 1e-4,
        .grad_norm = 0.8,
    };
    const c = toCStepMetrics(m);
    try std.testing.expectEqual(@as(u64, 42), c.step);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), c.loss, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1e-4), c.learning_rate, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), c.grad_norm, 1e-6);
}

test "toCTrainingInfo state mapping" {
    const info = SessionInfo{
        .state = .configured,
        .current_step = 10,
        .total_steps = 100,
        .trainable_params = 1024,
        .adapter_layers = 4,
    };
    const c = toCTrainingInfo(info);
    try std.testing.expectEqual(@as(u8, 2), c.state); // configured = 2
    try std.testing.expectEqual(@as(u64, 10), c.current_step);
    try std.testing.expectEqual(@as(u64, 100), c.total_steps);
    try std.testing.expectEqual(@as(u64, 1024), c.trainable_params);
    try std.testing.expectEqual(@as(u32, 4), c.adapter_layers);
}

test "capi_bridge createSession and destroySession" {
    const allocator = std.testing.allocator;
    const session = try createSession(allocator);
    defer destroySession(allocator, session);

    try std.testing.expectEqual(SessionState.created, session.state);
}

test "capi_bridge configure roundtrip" {
    const allocator = std.testing.allocator;
    const session = try createSession(allocator);
    defer destroySession(allocator, session);

    // Need adapter first
    var adapter = LoraAdapter.initExplicit(allocator, .{ .rank = 4, .alpha = 8.0 });
    errdefer adapter.deinit();
    var layer = try LoraLayer.init(allocator, "q_proj", 0, 8, 8, .{ .rank = 4, .alpha = 8.0 });
    errdefer layer.deinit();
    try adapter.addLayer(layer);
    try session.setAdapter(adapter);

    const c_cfg = CTrainingConfig{
        .learning_rate = 3e-4,
        .total_steps = 500,
        .batch_size = 2,
        .seq_len = 64,
    };
    try configureSession(session, &c_cfg);

    try std.testing.expectEqual(SessionState.configured, session.state);
    try std.testing.expectApproxEqAbs(@as(f32, 3e-4), session.config.learning_rate, 1e-10);
    try std.testing.expectEqual(@as(u32, 2), session.config.batch_size);
}

test "capi_bridge getInfo" {
    const allocator = std.testing.allocator;
    const session = try createSession(allocator);
    defer destroySession(allocator, session);

    const info = getInfo(session);
    try std.testing.expectEqual(@as(u8, 0), info.state); // created = 0
    try std.testing.expectEqual(@as(u64, 0), info.current_step);
}

test "capi_bridge callback wrapper" {
    const allocator = std.testing.allocator;
    const session = try createSession(allocator);
    defer destroySession(allocator, session);

    var adapter = LoraAdapter.initExplicit(allocator, .{ .rank = 2, .alpha = 4.0 });
    errdefer adapter.deinit();
    var layer = try LoraLayer.init(allocator, "q_proj", 0, 4, 4, .{ .rank = 2, .alpha = 4.0 });
    errdefer layer.deinit();
    try adapter.addLayer(layer);
    try session.setAdapter(adapter);

    try session.configure(.{ .total_steps = 100, .batch_size = 1, .seq_len = 2, .warmup_steps = 0 });

    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6 };
    try session.setData(&tokens);

    // C callback that cancels after 2 steps
    var call_count: u32 = 0;
    const Ctx = struct {
        fn cb(_: *const CStepMetrics, ud: ?*anyopaque) callconv(.c) i32 {
            const count: *u32 = @ptrCast(@alignCast(ud));
            count.* += 1;
            return if (count.* >= 2) 1 else 0;
        }
    };

    try std.testing.expectError(
        error.Cancelled,
        trainSession(session, Ctx.cb, @ptrCast(&call_count)),
    );
    try std.testing.expectEqual(@as(u32, 2), call_count);
}

test "CTrainingInfo padding is zeroed" {
    const info = CTrainingInfo{
        .current_step = 0,
        .total_steps = 0,
        .trainable_params = 0,
        .adapter_layers = 0,
        .state = 0,
    };
    for (info._padding) |b| {
        try std.testing.expectEqual(@as(u8, 0), b);
    }
}
