//! From-scratch training session for decoder-only transformers.
//!
//! Owns all training state: weights, activations, optimizer, scheduler, data.
//! State machine enforces correct operation ordering:
//!   created → initialized → configured → data_loaded → training → completed
//!
//! Usage:
//!   var session = FullTrainingSession.init(allocator);
//!   try session.initModel(config, seed);
//!   try session.configure(training_config);
//!   try session.setData(&tokens);
//!   const metrics = try session.trainStep();
//!   session.deinit();

const std = @import("std");
const model_config = @import("model_config.zig");
const model_weights_mod = @import("model_weights.zig");
const activations_mod = @import("activations.zig");
const forward_mod = @import("forward/root.zig");
const backward_mod = @import("backward/pass.zig");
const optimizer_mod = @import("optimizer.zig");
const scheduler_mod = @import("scheduler.zig");
const data_mod = @import("data.zig");
const loop_mod = @import("loop.zig");
const compute = @import("../compute/root.zig");

const Allocator = std.mem.Allocator;
const TransformerConfig = model_config.TransformerConfig;
const ModelWeights = model_weights_mod.ModelWeights;
const ActivationCache = activations_mod.ActivationCache;
const AdamW = optimizer_mod.AdamW;
const ParamState = optimizer_mod.ParamState;
const Scheduler = scheduler_mod.Scheduler;
const DataLoader = data_mod.DataLoader;
const StepMetrics = loop_mod.StepMetrics;
const MatmulScratch = compute.cpu.linalg.MatmulScratch;

/// Number of weight tensors per transformer layer.
const PARAMS_PER_LAYER: usize = 9; // attn_norm, q/k/v/o_proj, ffn_norm, gate/up/down_proj
/// Number of global weight tensors (outside layers).
const GLOBAL_PARAMS: usize = 3; // token_embedding, final_norm, lm_head

/// Session lifecycle state.
pub const State = enum(u8) {
    created = 0,
    initialized = 1,
    configured = 2,
    data_loaded = 3,
    training = 4,
    completed = 5,
};

/// Training hyperparameters.
pub const FullSessionConfig = struct {
    // Optimizer
    learning_rate: f32 = 3e-4,
    min_learning_rate: f32 = 3e-5,
    weight_decay: f32 = 0.1,
    beta1: f32 = 0.9,
    beta2: f32 = 0.95,
    epsilon: f32 = 1e-8,
    // Data
    batch_size: u32 = 32,
    // Schedule
    warmup_steps: u64 = 500,
    total_steps: u64 = 10000,
    // Training
    max_grad_norm: f32 = 1.0,
};

/// Read-only snapshot of session state.
pub const FullSessionInfo = struct {
    state: State,
    current_step: u64,
    total_steps: u64,
    total_params: u64,
    batch_size: u32,
};

/// Callback invoked after each training step.
/// Return 0 to continue, non-zero to cancel training.
pub const StepCallback = *const fn (metrics: StepMetrics, user_data: ?*anyopaque) i32;

pub const Error = error{
    InvalidState,
    InvalidDataFormat,
    Cancelled,
    OutOfMemory,
};

/// Stateful from-scratch training session with lifecycle enforcement.
pub const FullTrainingSession = struct {
    state: State,
    allocator: Allocator,

    // Model (set during initModel)
    model_config: ?TransformerConfig,
    weights: ?ModelWeights,
    cache: ?ActivationCache,
    mm_scratch: ?MatmulScratch,

    // Training (set during configure)
    training_config: FullSessionConfig,
    optimizer: ?AdamW,
    scheduler: ?Scheduler,
    opt_states: ?[]ParamState,

    // Data (set during setData/loadData)
    data_loader: ?DataLoader,
    tokens: ?[]u32, // owned copy when loaded from file

    // Progress
    current_step: u64,

    pub fn init(allocator: Allocator) FullTrainingSession {
        return .{
            .state = .created,
            .allocator = allocator,
            .model_config = null,
            .weights = null,
            .cache = null,
            .mm_scratch = null,
            .training_config = .{},
            .optimizer = null,
            .scheduler = null,
            .opt_states = null,
            .data_loader = null,
            .tokens = null,
            .current_step = 0,
        };
    }

    /// Initialize model with random weights.
    /// Transitions: created → initialized.
    pub fn initModel(self: *FullTrainingSession, config: TransformerConfig, seed: u64) !void {
        if (self.state != .created) return error.InvalidState;

        var weights = try ModelWeights.init(self.allocator, config);
        errdefer weights.deinit();
        weights.initRandom(seed);

        self.model_config = config;
        self.weights = weights;
        self.state = .initialized;
    }

    /// Configure training hyperparameters. Creates optimizer, scheduler,
    /// activation cache, matmul scratch, and per-weight optimizer states.
    /// Transitions: initialized → configured.
    pub fn configure(self: *FullTrainingSession, config: FullSessionConfig) !void {
        if (self.state != .initialized) return error.InvalidState;
        const mc = self.model_config.?;

        self.training_config = config;

        // Activation cache
        var cache = try ActivationCache.init(self.allocator, mc, config.batch_size);
        errdefer cache.deinit();
        self.cache = cache;

        // Matmul scratch
        var mm = try MatmulScratch.init(self.allocator);
        errdefer mm.deinit();
        self.mm_scratch = mm;

        // Optimizer
        self.optimizer = AdamW.init(.{
            .lr = config.learning_rate,
            .beta1 = config.beta1,
            .beta2 = config.beta2,
            .eps = config.epsilon,
            .weight_decay = config.weight_decay,
        });

        // Scheduler
        self.scheduler = Scheduler.init(.{
            .max_lr = config.learning_rate,
            .min_lr = config.min_learning_rate,
            .warmup_steps = config.warmup_steps,
            .total_steps = config.total_steps,
        });

        // Per-weight optimizer states
        const num_params = GLOBAL_PARAMS + PARAMS_PER_LAYER * mc.num_layers;
        const opt_states = try self.allocator.alloc(ParamState, num_params);
        var initialized: usize = 0;
        errdefer {
            for (opt_states[0..initialized]) |*os| os.deinit();
            self.allocator.free(opt_states);
        }

        const weights = &self.weights.?;
        // Global weights
        opt_states[0] = try ParamState.init(self.allocator, weights.token_embedding.numElements());
        initialized += 1;
        opt_states[1] = try ParamState.init(self.allocator, weights.final_norm.numElements());
        initialized += 1;
        opt_states[2] = try ParamState.init(self.allocator, weights.lm_head.numElements());
        initialized += 1;

        // Per-layer weights
        for (weights.layers, 0..) |*layer, li| {
            const base = GLOBAL_PARAMS + li * PARAMS_PER_LAYER;
            opt_states[base + 0] = try ParamState.init(self.allocator, layer.attn_norm.numElements());
            initialized += 1;
            opt_states[base + 1] = try ParamState.init(self.allocator, layer.q_proj.numElements());
            initialized += 1;
            opt_states[base + 2] = try ParamState.init(self.allocator, layer.k_proj.numElements());
            initialized += 1;
            opt_states[base + 3] = try ParamState.init(self.allocator, layer.v_proj.numElements());
            initialized += 1;
            opt_states[base + 4] = try ParamState.init(self.allocator, layer.o_proj.numElements());
            initialized += 1;
            opt_states[base + 5] = try ParamState.init(self.allocator, layer.ffn_norm.numElements());
            initialized += 1;
            opt_states[base + 6] = try ParamState.init(self.allocator, layer.gate_proj.numElements());
            initialized += 1;
            opt_states[base + 7] = try ParamState.init(self.allocator, layer.up_proj.numElements());
            initialized += 1;
            opt_states[base + 8] = try ParamState.init(self.allocator, layer.down_proj.numElements());
            initialized += 1;
        }

        self.opt_states = opt_states;
        self.state = .configured;
    }

    /// Set tokenized training data from an existing slice.
    /// The caller retains ownership; slice must outlive the session.
    /// Transitions: configured → data_loaded.
    pub fn setData(self: *FullTrainingSession, data_tokens: []const u32) !void {
        if (self.state != .configured) return error.InvalidState;
        const mc = self.model_config.?;

        self.data_loader = try DataLoader.init(
            self.allocator,
            data_tokens,
            @intCast(self.training_config.batch_size),
            @intCast(mc.seq_len),
        );
        self.syncDataLoaderCursor();

        self.state = .data_loaded;
    }

    /// Load tokenized training data from a flat binary file of u32 tokens.
    /// The session takes ownership of the loaded data.
    /// Transitions: configured → data_loaded.
    pub fn loadData(self: *FullTrainingSession, path: []const u8) !void {
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

        const mc = self.model_config.?;
        self.data_loader = try DataLoader.init(
            self.allocator,
            buf,
            @intCast(self.training_config.batch_size),
            @intCast(mc.seq_len),
        );
        self.syncDataLoaderCursor();

        self.state = .data_loaded;
    }

    /// Run one training step: forward → backward → clip → optimizer.
    ///
    /// Returns step metrics (loss, lr, grad_norm, step number).
    /// Transitions: data_loaded → training (on first call).
    pub fn trainStep(self: *FullTrainingSession) !StepMetrics {
        if (self.state != .data_loaded and self.state != .training)
            return error.InvalidState;

        const mc = self.model_config.?;
        var weights = &self.weights.?;
        const cache = &self.cache.?;
        const mm = &self.mm_scratch.?;
        var opt = &self.optimizer.?;
        const sched = &self.scheduler.?;

        const bs: usize = @as(usize, self.training_config.batch_size) * mc.seq_len;

        // Get batch from data loader (reset epoch if exhausted)
        var dl = &self.data_loader.?;
        const batch = dl.nextBatch() orelse blk: {
            dl.reset();
            break :blk dl.nextBatch().?;
        };

        std.debug.assert(batch.input_ids.len == bs);
        std.debug.assert(batch.targets.len == bs);

        // Zero gradients
        weights.zeroGrads();

        // Forward pass
        const loss = forward_mod.forward(weights, cache, batch.input_ids, batch.targets, mm);

        // Backward pass
        backward_mod.backward(weights, cache, batch.input_ids, batch.targets, mm);

        // Gradient clipping
        const grad_norm = self.computeAndClipGradNorm();

        // Optimizer step
        const lr = sched.getLr(self.current_step);
        self.stepAllParams(opt, lr);

        self.current_step += 1;
        opt.step_count = self.current_step;
        if (self.state == .data_loaded) self.state = .training;

        return .{
            .loss = loss,
            .learning_rate = lr,
            .grad_norm = grad_norm,
            .step = self.current_step,
        };
    }

    /// Run the full training loop with an optional per-step callback.
    /// Returns error.Cancelled if the callback returns non-zero.
    /// Transitions: data_loaded → training → completed.
    pub fn run(
        self: *FullTrainingSession,
        callback: ?StepCallback,
        user_data: ?*anyopaque,
    ) !void {
        if (self.state != .data_loaded) return error.InvalidState;

        while (self.current_step < self.training_config.total_steps) {
            const metrics = try self.trainStep();

            if (callback) |cb| {
                if (cb(metrics, user_data) != 0) return error.Cancelled;
            }
        }

        self.state = .completed;
    }

    /// Query current session state.
    pub fn getInfo(self: *const FullTrainingSession) FullSessionInfo {
        return .{
            .state = self.state,
            .current_step = self.current_step,
            .total_steps = self.training_config.total_steps,
            .total_params = if (self.weights) |*w| w.totalParams() else 0,
            .batch_size = self.training_config.batch_size,
        };
    }

    /// Copy current model weights into a caller-provided flat f32 buffer.
    pub fn copyWeightsF32(self: *const FullTrainingSession, out: []f32) !void {
        if (self.weights == null) return error.InvalidState;
        const weights = &self.weights.?;
        weights.copyFlatF32(out);
    }

    /// Restore model weights and step counter from a flat checkpoint buffer.
    pub fn loadWeightsF32(self: *FullTrainingSession, flat: []const f32, step: u64) !void {
        if (self.weights == null) return error.InvalidState;
        const weights = &self.weights.?;
        weights.loadFlatF32(flat);
        self.current_step = step;
        self.syncResumeState();
    }

    /// Total number of f32 elements needed to serialize optimizer state.
    pub fn optimizerStateLen(self: *const FullTrainingSession) !usize {
        if (self.weights == null or self.opt_states == null) return error.InvalidState;
        return @as(usize, @intCast(self.weights.?.totalParams())) * 2;
    }

    /// Copy Adam first/second moments into a flat f32 buffer.
    /// Layout per tensor: m values, then v values, in ModelWeights parameter order.
    pub fn copyOptimizerStateF32(self: *const FullTrainingSession, out: []f32) !void {
        const expected_len = try self.optimizerStateLen();
        if (out.len != expected_len) return error.InvalidState;

        const states = self.opt_states.?;
        var cursor: usize = 0;
        for (states) |*state| {
            cursor += copyOptSlice(out[cursor..], state.m);
            cursor += copyOptSlice(out[cursor..], state.v);
        }
        std.debug.assert(cursor == out.len);
    }

    /// Restore Adam first/second moments from a flat f32 buffer.
    pub fn loadOptimizerStateF32(self: *FullTrainingSession, flat: []const f32) !void {
        const expected_len = try self.optimizerStateLen();
        if (flat.len != expected_len) return error.InvalidState;

        const states = self.opt_states.?;
        var cursor: usize = 0;
        for (states) |*state| {
            cursor += loadOptSlice(state.m, flat[cursor..]);
            cursor += loadOptSlice(state.v, flat[cursor..]);
        }
        std.debug.assert(cursor == flat.len);
        self.syncResumeState();
    }


    fn syncResumeState(self: *FullTrainingSession) void {
        self.syncDataLoaderCursor();
        if (self.optimizer) |*opt| opt.step_count = self.current_step;
    }

    fn syncDataLoaderCursor(self: *FullTrainingSession) void {
        if (self.data_loader) |*dl| {
            const batches_per_epoch = dl.numBatches();
            if (batches_per_epoch == 0) {
                dl.cursor = 0;
                return;
            }

            const step_mod = @as(usize, @intCast(self.current_step % batches_per_epoch));
            const batch_stride = dl.batch_size * (dl.seq_len + 1);
            dl.cursor = step_mod * batch_stride;
        }
    }

    /// Free all resources.
    pub fn deinit(self: *FullTrainingSession) void {
        if (self.data_loader) |*dl| dl.deinit();
        if (self.opt_states) |states| {
            for (states) |*s| s.deinit();
            self.allocator.free(states);
        }
        if (self.mm_scratch) |*mm| mm.deinit();
        if (self.cache) |*c| c.deinit();
        if (self.weights) |*w| w.deinit();
        if (self.tokens) |t| self.allocator.free(t);
        self.* = undefined;
    }

    // =========================================================================
    // Internal
    // =========================================================================

    /// Compute global gradient norm across all weights and clip if needed.
    /// Returns the pre-clip global norm.
    fn computeAndClipGradNorm(self: *FullTrainingSession) f32 {
        const weights = &self.weights.?;
        const max_norm = self.training_config.max_grad_norm;

        var global_norm_sq: f32 = 0.0;

        // Global weights
        global_norm_sq += normSq(weights.grad_token_embedding.asSlice());
        global_norm_sq += normSq(weights.grad_final_norm.asSlice());
        global_norm_sq += normSq(weights.grad_lm_head.asSlice());

        // Per-layer weights
        for (weights.layers) |*layer| {
            global_norm_sq += normSq(layer.grad_attn_norm.asSlice());
            global_norm_sq += normSq(layer.grad_q_proj.asSlice());
            global_norm_sq += normSq(layer.grad_k_proj.asSlice());
            global_norm_sq += normSq(layer.grad_v_proj.asSlice());
            global_norm_sq += normSq(layer.grad_o_proj.asSlice());
            global_norm_sq += normSq(layer.grad_ffn_norm.asSlice());
            global_norm_sq += normSq(layer.grad_gate_proj.asSlice());
            global_norm_sq += normSq(layer.grad_up_proj.asSlice());
            global_norm_sq += normSq(layer.grad_down_proj.asSlice());
        }

        const global_norm = @sqrt(global_norm_sq);

        if (global_norm > max_norm and global_norm > 0.0) {
            const scale = max_norm / global_norm;
            scaleGrad(weights.grad_token_embedding.asSliceMut(), scale);
            scaleGrad(weights.grad_final_norm.asSliceMut(), scale);
            scaleGrad(weights.grad_lm_head.asSliceMut(), scale);
            for (weights.layers) |*layer| {
                scaleGrad(layer.grad_attn_norm.asSliceMut(), scale);
                scaleGrad(layer.grad_q_proj.asSliceMut(), scale);
                scaleGrad(layer.grad_k_proj.asSliceMut(), scale);
                scaleGrad(layer.grad_v_proj.asSliceMut(), scale);
                scaleGrad(layer.grad_o_proj.asSliceMut(), scale);
                scaleGrad(layer.grad_ffn_norm.asSliceMut(), scale);
                scaleGrad(layer.grad_gate_proj.asSliceMut(), scale);
                scaleGrad(layer.grad_up_proj.asSliceMut(), scale);
                scaleGrad(layer.grad_down_proj.asSliceMut(), scale);
            }
        }

        return global_norm;
    }

    /// Apply optimizer step to all weight tensors.
    fn stepAllParams(self: *FullTrainingSession, opt: *AdamW, lr: f32) void {
        var weights = &self.weights.?;
        const states = self.opt_states.?;
        const step_index = self.current_step + 1;

        // Global weights
        opt.stepAt(weights.token_embedding.asSlice(f32), weights.grad_token_embedding.asSlice(), &states[0], lr, step_index);
        opt.stepAt(weights.final_norm.asSlice(f32), weights.grad_final_norm.asSlice(), &states[1], lr, step_index);
        opt.stepAt(weights.lm_head.asSlice(f32), weights.grad_lm_head.asSlice(), &states[2], lr, step_index);

        // Per-layer weights
        for (weights.layers, 0..) |*layer, li| {
            const base = GLOBAL_PARAMS + li * PARAMS_PER_LAYER;
            opt.stepAt(layer.attn_norm.asSlice(f32), layer.grad_attn_norm.asSlice(), &states[base + 0], lr, step_index);
            opt.stepAt(layer.q_proj.asSlice(f32), layer.grad_q_proj.asSlice(), &states[base + 1], lr, step_index);
            opt.stepAt(layer.k_proj.asSlice(f32), layer.grad_k_proj.asSlice(), &states[base + 2], lr, step_index);
            opt.stepAt(layer.v_proj.asSlice(f32), layer.grad_v_proj.asSlice(), &states[base + 3], lr, step_index);
            opt.stepAt(layer.o_proj.asSlice(f32), layer.grad_o_proj.asSlice(), &states[base + 4], lr, step_index);
            opt.stepAt(layer.ffn_norm.asSlice(f32), layer.grad_ffn_norm.asSlice(), &states[base + 5], lr, step_index);
            opt.stepAt(layer.gate_proj.asSlice(f32), layer.grad_gate_proj.asSlice(), &states[base + 6], lr, step_index);
            opt.stepAt(layer.up_proj.asSlice(f32), layer.grad_up_proj.asSlice(), &states[base + 7], lr, step_index);
            opt.stepAt(layer.down_proj.asSlice(f32), layer.grad_down_proj.asSlice(), &states[base + 8], lr, step_index);

            // Sync fused QKV weight buffer after q/k/v weights updated.
            layer.syncQkvBuf();
        }
    }
};

const simd = compute.cpu.simd.arch;
const VEC = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

fn normSq(data: []const f32) f32 {
    @setFloatMode(.optimized);
    var acc: F32Vec = @splat(0.0);
    var i: usize = 0;
    while (i + VEC <= data.len) : (i += VEC) {
        const v: F32Vec = data[i..][0..VEC].*;
        acc = @mulAdd(F32Vec, v, v, acc);
    }
    var sum = @reduce(.Add, acc);
    while (i < data.len) : (i += 1) {
        sum += data[i] * data[i];
    }
    return sum;
}

fn scaleGrad(data: []f32, factor: f32) void {
    @setFloatMode(.optimized);
    const f: F32Vec = @splat(factor);
    var i: usize = 0;
    while (i + VEC <= data.len) : (i += VEC) {
        var v: F32Vec = data[i..][0..VEC].*;
        v *= f;
        data[i..][0..VEC].* = v;
    }
    while (i < data.len) : (i += 1) {
        data[i] *= factor;
    }
}

fn copyOptSlice(dst: []f32, src: []const f32) usize {
    std.debug.assert(dst.len >= src.len);
    @memcpy(dst[0..src.len], src);
    return src.len;
}

fn loadOptSlice(dst: []f32, src: []const f32) usize {
    std.debug.assert(src.len >= dst.len);
    @memcpy(dst, src[0..dst.len]);
    return dst.len;
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

fn testConfig() TransformerConfig {
    return .{
        .vocab_size = 8,
        .d_model = 4,
        .num_layers = 1,
        .num_heads = 1,
        .num_kv_heads = 1,
        .d_ff = 8,
        .seq_len = 2,
    };
}

test "FullTrainingSession state transitions: happy path" {
    var session = FullTrainingSession.init(testing.allocator);
    defer session.deinit();

    try testing.expectEqual(State.created, session.getInfo().state);

    try session.initModel(testConfig(), 42);
    try testing.expectEqual(State.initialized, session.getInfo().state);
    try testing.expect(session.getInfo().total_params > 0);

    try session.configure(.{
        .total_steps = 5,
        .batch_size = 1,
        .warmup_steps = 0,
    });
    try testing.expectEqual(State.configured, session.getInfo().state);

    // Need enough tokens for at least one batch: batch_size * seq_len + 1
    const tokens = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    try session.setData(&tokens);
    try testing.expectEqual(State.data_loaded, session.getInfo().state);
}

test "FullTrainingSession rejects out-of-order calls" {
    // Cannot configure before initModel
    {
        var session = FullTrainingSession.init(testing.allocator);
        defer session.deinit();
        try testing.expectError(error.InvalidState, session.configure(.{}));
    }

    // Cannot setData before configure
    {
        var session = FullTrainingSession.init(testing.allocator);
        defer session.deinit();
        try session.initModel(testConfig(), 42);
        const tokens = [_]u32{ 0, 1, 2, 3 };
        try testing.expectError(error.InvalidState, session.setData(&tokens));
    }

    // Cannot initModel twice
    {
        var session = FullTrainingSession.init(testing.allocator);
        defer session.deinit();
        try session.initModel(testConfig(), 42);
        try testing.expectError(error.InvalidState, session.initModel(testConfig(), 99));
    }
}

test "FullTrainingSession trainStep produces valid metrics" {
    var session = FullTrainingSession.init(testing.allocator);
    defer session.deinit();

    try session.initModel(testConfig(), 42);
    try session.configure(.{
        .total_steps = 100,
        .batch_size = 1,
        .warmup_steps = 0,
        .learning_rate = 1e-3,
    });

    const tokens = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    try session.setData(&tokens);

    const metrics = try session.trainStep();

    // Loss should be finite and positive
    try testing.expect(!std.math.isNan(metrics.loss));
    try testing.expect(!std.math.isInf(metrics.loss));
    try testing.expect(metrics.loss > 0.0);

    // Learning rate should be positive
    try testing.expect(metrics.learning_rate > 0.0);

    // Step should be 1
    try testing.expectEqual(@as(u64, 1), metrics.step);

    // State should be training
    try testing.expectEqual(State.training, session.state);
}

test "FullTrainingSession loss decreases over steps" {
    var session = FullTrainingSession.init(testing.allocator);
    defer session.deinit();

    try session.initModel(testConfig(), 42);
    try session.configure(.{
        .total_steps = 50,
        .batch_size = 1,
        .warmup_steps = 0,
        .learning_rate = 1e-2,
        .weight_decay = 0.0,
    });

    // Small dataset that cycles: model should memorize it
    const tokens = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    try session.setData(&tokens);

    const first_metrics = try session.trainStep();
    var last_loss = first_metrics.loss;

    // Train more steps
    for (0..29) |_| {
        const m = try session.trainStep();
        last_loss = m.loss;
    }

    // After 30 steps, loss should be lower than initial
    try testing.expect(last_loss < first_metrics.loss);
}

test "FullTrainingSession run with cancellation" {
    var session = FullTrainingSession.init(testing.allocator);
    defer session.deinit();

    try session.initModel(testConfig(), 42);
    try session.configure(.{
        .total_steps = 1000,
        .batch_size = 1,
        .warmup_steps = 0,
    });

    const tokens = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    try session.setData(&tokens);

    // Cancel after 3 steps
    const Ctx = struct {
        fn cb(metrics: StepMetrics, _: ?*anyopaque) i32 {
            return if (metrics.step >= 3) 1 else 0;
        }
    };

    try testing.expectError(error.Cancelled, session.run(Ctx.cb, null));
    try testing.expectEqual(State.training, session.state);
    try testing.expectEqual(@as(u64, 3), session.current_step);
}

test "FullTrainingSession run completes" {
    var session = FullTrainingSession.init(testing.allocator);
    defer session.deinit();

    try session.initModel(testConfig(), 42);
    try session.configure(.{
        .total_steps = 3,
        .batch_size = 1,
        .warmup_steps = 0,
    });

    const tokens = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    try session.setData(&tokens);

    try session.run(null, null);
    try testing.expectEqual(State.completed, session.state);
    try testing.expectEqual(@as(u64, 3), session.current_step);
}

test "FullTrainingSession getInfo reflects configuration" {
    var session = FullTrainingSession.init(testing.allocator);
    defer session.deinit();

    try session.initModel(testConfig(), 42);
    try session.configure(.{
        .total_steps = 100,
        .batch_size = 2,
        .warmup_steps = 10,
    });

    const info = session.getInfo();
    try testing.expectEqual(@as(u64, 100), info.total_steps);
    try testing.expectEqual(@as(u32, 2), info.batch_size);
    try testing.expect(info.total_params > 0);
}
