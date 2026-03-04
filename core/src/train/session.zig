//! Training session: stateful lifecycle for LoRA fine-tuning.
//!
//! Owns all training state: adapter, optimizer, scheduler, data loader.
//! State machine enforces correct operation ordering:
//!   created → model_loaded → configured → data_loaded → training → completed
//!
//! Usage (explicit setup for testing / pre-built adapters):
//!   var session = TrainingSession.init(allocator);
//!   try session.setAdapter(adapter);
//!   try session.configure(config);
//!   try session.setData(&tokens);
//!   const metrics = try session.optimizerStep();
//!   session.deinit();

const std = @import("std");
const adapter_mod = @import("adapter.zig");
const optimizer_mod = @import("optimizer.zig");
const scheduler_mod = @import("scheduler.zig");
const param_state_mod = @import("param_state.zig");
const data_mod = @import("data.zig");
const loop_mod = @import("loop.zig");
const grad_mod = @import("grad.zig");

const LoraAdapter = adapter_mod.LoraAdapter;
const LoraConfig = adapter_mod.LoraConfig;
const LoraLayer = adapter_mod.LoraLayer;
const AdamW = optimizer_mod.AdamW;
const OptimizerParamState = optimizer_mod.ParamState;
const Scheduler = scheduler_mod.Scheduler;
const TrainableParams = param_state_mod.TrainableParams;
const TrainableParam = param_state_mod.TrainableParam;
const DataLoader = data_mod.DataLoader;
const GradTensor = grad_mod.GradTensor;
const StepMetrics = loop_mod.StepMetrics;
const Allocator = std.mem.Allocator;

/// Session lifecycle state.
pub const State = enum(u8) {
    created = 0,
    model_loaded = 1,
    configured = 2,
    data_loaded = 3,
    training = 4,
    completed = 5,
};

/// Training hyperparameters.
pub const SessionConfig = struct {
    // Optimizer
    learning_rate: f32 = 1e-4,
    min_learning_rate: f32 = 1e-6,
    weight_decay: f32 = 0.01,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    epsilon: f32 = 1e-8,
    // Data
    batch_size: u32 = 4,
    seq_len: u32 = 128,
    // Schedule
    warmup_steps: u64 = 100,
    total_steps: u64 = 1000,
    // Training loop
    max_grad_norm: f32 = 1.0,
    gradient_accumulation_steps: u32 = 1,
    log_interval: u32 = 10,
    save_interval: u32 = 0,
};

/// Read-only snapshot of session state.
pub const SessionInfo = struct {
    state: State,
    current_step: u64,
    total_steps: u64,
    trainable_params: usize,
    adapter_layers: usize,
};

/// Callback invoked after each optimizer step.
/// Return 0 to continue, non-zero to cancel training.
pub const StepCallback = *const fn (metrics: StepMetrics, user_data: ?*anyopaque) i32;

pub const Error = error{
    InvalidState,
    InvalidDataFormat,
    Cancelled,
    OutOfMemory,
};

/// Stateful training session with lifecycle enforcement.
pub const TrainingSession = struct {
    state: State,
    allocator: Allocator,

    // Set during setAdapter (model_loaded state)
    adapter: ?LoraAdapter,

    // Set during configure (configured state)
    config: SessionConfig,
    params: ?TrainableParams,
    optimizer: ?AdamW,
    scheduler: ?Scheduler,

    // Set during setData/loadData (data_loaded state)
    data_loader: ?DataLoader,
    tokens: ?[]u32, // owned copy when loaded from file

    // Training progress
    current_step: u64,

    pub fn init(allocator: Allocator) TrainingSession {
        return .{
            .state = .created,
            .allocator = allocator,
            .adapter = null,
            .config = .{},
            .params = null,
            .optimizer = null,
            .scheduler = null,
            .data_loader = null,
            .tokens = null,
            .current_step = 0,
        };
    }

    /// Set a LoRA adapter explicitly.
    /// Transitions: created → model_loaded.
    pub fn setAdapter(self: *TrainingSession, adapter: LoraAdapter) error{InvalidState}!void {
        if (self.state != .created) return error.InvalidState;
        self.adapter = adapter;
        self.state = .model_loaded;
    }

    /// Configure training hyperparameters. Creates optimizer, scheduler,
    /// and TrainableParams from the adapter's LoRA matrices.
    /// Transitions: model_loaded → configured.
    pub fn configure(self: *TrainingSession, config: SessionConfig) !void {
        if (self.state != .model_loaded) return error.InvalidState;

        self.config = config;

        // Build TrainableParams: one entry per LoRA matrix (A and B per layer)
        var params = TrainableParams.init(self.allocator);
        errdefer params.deinit();

        const layers = self.adapter.?.layers.items;
        for (layers) |*layer| {
            var param_a = try TrainableParam.init(
                self.allocator,
                layer.weight_id,
                @constCast(layer.A.asSlice(f32)),
            );
            errdefer param_a.deinit();
            try params.addParam(param_a);

            var param_b = try TrainableParam.init(
                self.allocator,
                layer.weight_id,
                @constCast(layer.B.asSlice(f32)),
            );
            errdefer param_b.deinit();
            try params.addParam(param_b);
        }

        self.params = params;

        self.optimizer = AdamW.init(.{
            .lr = config.learning_rate,
            .beta1 = config.beta1,
            .beta2 = config.beta2,
            .eps = config.epsilon,
            .weight_decay = config.weight_decay,
        });

        self.scheduler = Scheduler.init(.{
            .max_lr = config.learning_rate,
            .min_lr = config.min_learning_rate,
            .warmup_steps = config.warmup_steps,
            .total_steps = config.total_steps,
        });

        self.state = .configured;
    }

    /// Set tokenized training data from an existing slice.
    /// The caller retains ownership of the slice; it must outlive the session.
    /// Transitions: configured → data_loaded.
    pub fn setData(self: *TrainingSession, tokens: []const u32) !void {
        if (self.state != .configured) return error.InvalidState;

        self.data_loader = try DataLoader.init(
            self.allocator,
            tokens,
            self.config.batch_size,
            self.config.seq_len,
        );

        self.state = .data_loaded;
    }

    /// Load tokenized training data from a flat binary file of u32 tokens.
    /// The session takes ownership of the loaded data.
    /// Transitions: configured → data_loaded.
    pub fn loadData(self: *TrainingSession, path: []const u8) !void {
        if (self.state != .configured) return error.InvalidState;

        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const stat = try file.stat();
        if (stat.size == 0 or stat.size % 4 != 0) return error.InvalidDataFormat;

        const num_tokens: usize = @intCast(stat.size / 4);
        const buf = try self.allocator.alloc(u32, num_tokens);
        errdefer self.allocator.free(buf);

        const bytes = std.mem.sliceAsBytes(buf);
        const bytes_read = try file.readAll(bytes);
        if (bytes_read != stat.size) return error.InvalidDataFormat;

        self.tokens = buf;

        self.data_loader = try DataLoader.init(
            self.allocator,
            buf,
            self.config.batch_size,
            self.config.seq_len,
        );

        self.state = .data_loaded;
    }

    /// Run one optimizer step on all trainable parameters.
    ///
    /// Gradients must be populated externally before calling this.
    /// Clips gradients, steps the optimizer, and advances the step counter.
    /// Transitions: data_loaded → training (on first call).
    pub fn optimizerStep(self: *TrainingSession) !StepMetrics {
        if (self.state != .data_loaded and self.state != .training)
            return error.InvalidState;

        const metrics = self.stepInternal();

        if (self.state == .data_loaded) self.state = .training;
        return metrics;
    }

    /// Run the full training loop with an optional per-step callback.
    /// Returns error.Cancelled if the callback returns non-zero.
    /// Transitions: data_loaded → training → completed.
    pub fn train(
        self: *TrainingSession,
        callback: ?StepCallback,
        user_data: ?*anyopaque,
    ) !void {
        if (self.state != .data_loaded) return error.InvalidState;
        self.state = .training;

        while (self.current_step < self.config.total_steps) {
            // Zero gradients
            if (self.params) |*p| p.zeroAllGrads();

            // TODO: forward + backward pass (pending training forward implementation)

            const metrics = self.stepInternal();

            if (callback) |cb| {
                if (cb(metrics, user_data) != 0) return error.Cancelled;
            }
        }

        self.state = .completed;
    }

    /// Query current session state.
    pub fn getInfo(self: *const TrainingSession) SessionInfo {
        return .{
            .state = self.state,
            .current_step = self.current_step,
            .total_steps = self.config.total_steps,
            .trainable_params = if (self.params) |*p| p.totalParamCount() else 0,
            .adapter_layers = if (self.adapter) |*a| a.layerCount() else 0,
        };
    }

    /// Free all resources. Order: data_loader → params → adapter → tokens.
    pub fn deinit(self: *TrainingSession) void {
        if (self.data_loader) |*dl| dl.deinit();
        if (self.params) |*p| p.deinit();
        if (self.adapter) |*a| a.deinit();
        if (self.tokens) |t| self.allocator.free(t);
        self.* = undefined;
    }

    // =========================================================================
    // Internal
    // =========================================================================

    /// Performs gradient clipping and optimizer step without state transition.
    fn stepInternal(self: *TrainingSession) StepMetrics {
        const params = &self.params.?;
        var opt = &self.optimizer.?;
        const sched = &self.scheduler.?;

        const lr = sched.getLr(self.current_step);

        // Global gradient norm + clipping
        var global_norm_sq: f32 = 0.0;
        for (params.params.items) |*p| {
            const n = p.grad.norm();
            global_norm_sq += n * n;
        }
        const global_norm = @sqrt(global_norm_sq);

        if (global_norm > self.config.max_grad_norm and global_norm > 0.0) {
            const scale = self.config.max_grad_norm / global_norm;
            for (params.params.items) |*p| {
                p.grad.scale(scale);
            }
        }

        // Optimizer step
        for (params.params.items) |*p| {
            opt.step(p.data, p.grad.asSlice(), &p.opt_state, lr);
        }

        self.current_step += 1;

        return .{
            .loss = 0.0, // Forward pass not yet wired
            .learning_rate = lr,
            .grad_norm = global_norm,
            .step = self.current_step,
        };
    }
};

// =============================================================================
// Tests
// =============================================================================

fn createTestAdapter(allocator: Allocator) !LoraAdapter {
    var adapter = LoraAdapter.initExplicit(allocator, .{ .rank = 2, .alpha = 4.0 });
    errdefer adapter.deinit();

    var layer = try LoraLayer.init(allocator, "q_proj", 0, 4, 4, .{ .rank = 2, .alpha = 4.0 });
    errdefer layer.deinit();
    try adapter.addLayer(layer);

    return adapter;
}

test "TrainingSession state transitions: happy path" {
    const allocator = std.testing.allocator;
    var session = TrainingSession.init(allocator);
    defer session.deinit();

    try std.testing.expectEqual(State.created, session.getInfo().state);

    var adapter = try createTestAdapter(allocator);
    errdefer adapter.deinit();
    try session.setAdapter(adapter);
    try std.testing.expectEqual(State.model_loaded, session.getInfo().state);

    try session.configure(.{ .total_steps = 5, .batch_size = 1, .seq_len = 2 });
    try std.testing.expectEqual(State.configured, session.getInfo().state);

    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6 };
    try session.setData(&tokens);
    try std.testing.expectEqual(State.data_loaded, session.getInfo().state);
}

test "TrainingSession rejects out-of-order calls" {
    const allocator = std.testing.allocator;

    // Cannot configure before setAdapter
    {
        var session = TrainingSession.init(allocator);
        defer session.deinit();
        try std.testing.expectError(error.InvalidState, session.configure(.{}));
    }

    // Cannot setData before configure
    {
        var session = TrainingSession.init(allocator);
        defer session.deinit();
        var adapter = try createTestAdapter(allocator);
        errdefer adapter.deinit();
        try session.setAdapter(adapter);
        const tokens = [_]u32{ 1, 2, 3 };
        try std.testing.expectError(error.InvalidState, session.setData(&tokens));
    }

    // Cannot setAdapter twice
    {
        var session = TrainingSession.init(allocator);
        defer session.deinit();
        var adapter = try createTestAdapter(allocator);
        errdefer adapter.deinit();
        try session.setAdapter(adapter);
        var adapter2 = try createTestAdapter(allocator);
        defer adapter2.deinit();
        try std.testing.expectError(error.InvalidState, session.setAdapter(adapter2));
    }
}

test "TrainingSession getInfo reflects adapter" {
    const allocator = std.testing.allocator;
    var session = TrainingSession.init(allocator);
    defer session.deinit();

    // Before adapter: zero params
    try std.testing.expectEqual(@as(usize, 0), session.getInfo().trainable_params);
    try std.testing.expectEqual(@as(usize, 0), session.getInfo().adapter_layers);

    var adapter = try createTestAdapter(allocator);
    errdefer adapter.deinit();
    try session.setAdapter(adapter);

    try session.configure(.{ .total_steps = 10, .batch_size = 1, .seq_len = 2 });

    const info = session.getInfo();
    try std.testing.expectEqual(@as(usize, 1), info.adapter_layers);
    // A: 2*4=8, B: 4*2=8 → 16 trainable params
    try std.testing.expectEqual(@as(usize, 16), info.trainable_params);
    try std.testing.expectEqual(@as(u64, 10), info.total_steps);
}

test "TrainingSession optimizerStep advances step and transitions state" {
    const allocator = std.testing.allocator;
    var session = TrainingSession.init(allocator);
    defer session.deinit();

    var adapter = try createTestAdapter(allocator);
    errdefer adapter.deinit();
    try session.setAdapter(adapter);
    try session.configure(.{ .total_steps = 100, .batch_size = 1, .seq_len = 2, .warmup_steps = 0 });

    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6 };
    try session.setData(&tokens);

    try std.testing.expectEqual(State.data_loaded, session.state);

    const metrics = try session.optimizerStep();
    try std.testing.expectEqual(State.training, session.state);
    try std.testing.expectEqual(@as(u64, 1), metrics.step);
    try std.testing.expect(metrics.learning_rate > 0.0);
}

test "TrainingSession optimizerStep with nonzero gradients changes params" {
    const allocator = std.testing.allocator;
    var session = TrainingSession.init(allocator);
    defer session.deinit();

    var adapter = try createTestAdapter(allocator);
    errdefer adapter.deinit();
    try session.setAdapter(adapter);
    try session.configure(.{
        .total_steps = 100,
        .learning_rate = 0.1,
        .batch_size = 1,
        .seq_len = 2,
        .warmup_steps = 0,
        .weight_decay = 0.0,
    });

    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6 };
    try session.setData(&tokens);

    // Record initial param values
    const initial_a0 = session.params.?.params.items[0].data[0];

    // Set nonzero gradients on the first parameter group
    for (session.params.?.params.items[0].grad.asSliceMut()) |*g| g.* = 1.0;

    const metrics = try session.optimizerStep();
    try std.testing.expect(metrics.grad_norm > 0.0);

    // Param should have moved
    const new_a0 = session.params.?.params.items[0].data[0];
    try std.testing.expect(new_a0 != initial_a0);
}

test "TrainingSession train with cancellation callback" {
    const allocator = std.testing.allocator;
    var session = TrainingSession.init(allocator);
    defer session.deinit();

    var adapter = try createTestAdapter(allocator);
    errdefer adapter.deinit();
    try session.setAdapter(adapter);
    try session.configure(.{ .total_steps = 1000, .batch_size = 1, .seq_len = 2, .warmup_steps = 0 });

    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6 };
    try session.setData(&tokens);

    // Cancel after 3 steps
    const Ctx = struct {
        fn cb(metrics: StepMetrics, _: ?*anyopaque) i32 {
            return if (metrics.step >= 3) 1 else 0;
        }
    };

    try std.testing.expectError(error.Cancelled, session.train(Ctx.cb, null));
    try std.testing.expectEqual(State.training, session.state);
    try std.testing.expectEqual(@as(u64, 3), session.current_step);
}

test "TrainingSession train completes" {
    const allocator = std.testing.allocator;
    var session = TrainingSession.init(allocator);
    defer session.deinit();

    var adapter = try createTestAdapter(allocator);
    errdefer adapter.deinit();
    try session.setAdapter(adapter);
    try session.configure(.{ .total_steps = 3, .batch_size = 1, .seq_len = 2, .warmup_steps = 0 });

    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6 };
    try session.setData(&tokens);

    try session.train(null, null);
    try std.testing.expectEqual(State.completed, session.state);
    try std.testing.expectEqual(@as(u64, 3), session.current_step);
}

test "TrainingSession config defaults match expected values" {
    const cfg = SessionConfig{};
    try std.testing.expectApproxEqAbs(@as(f32, 1e-4), cfg.learning_rate, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), cfg.beta1, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f32, 0.999), cfg.beta2, 1e-10);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cfg.max_grad_norm, 1e-10);
    try std.testing.expectEqual(@as(u32, 1), cfg.gradient_accumulation_steps);
}
