//! Full training C API — from-scratch transformer training.
//!
//! Exports C-callable functions for the full training session lifecycle:
//!   talu_train_full_create, talu_train_full_destroy, talu_train_full_init_model,
//!   talu_train_full_configure, talu_train_full_set_data, talu_train_full_load_data,
//!   talu_train_full_step, talu_train_full_run, talu_train_full_get_info,
//!   talu_train_full_copy_weights_f32, talu_train_full_load_weights_f32,
//!   talu_train_full_copy_optimizer_state_f32, talu_train_full_load_optimizer_state_f32
//!
//! Each function: clearError → validate → delegate → catch → setError → return code.

const std = @import("std");
const train_mod = @import("../train/root.zig");
const capi_bridge = train_mod.capi_bridge;

const FullTrainingSession = train_mod.FullTrainingSession;
const FullSessionConfig = train_mod.FullSessionConfig;
const FullSessionInfo = train_mod.FullSessionInfo;
const TransformerConfig = train_mod.TransformerConfig;
const StepMetrics = train_mod.StepMetrics;

const CStepMetrics = capi_bridge.CStepMetrics;
const CStepCallback = capi_bridge.CStepCallback;

const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");
const ErrorCode = error_codes.ErrorCode;

const allocator = std.heap.c_allocator;

// =============================================================================
// C-Compatible Extern Structs
// =============================================================================

/// Transformer architecture configuration (C-compatible).
pub const CTransformerConfig = extern struct {
    vocab_size: u32,
    d_model: u32,
    num_layers: u32,
    num_heads: u32,
    num_kv_heads: u32,
    d_ff: u32,
    seq_len: u32,
    rope_theta: f32,
    norm_eps: f32,
};

/// Full training session hyperparameters (C-compatible).
///
/// Field order avoids implicit padding: f32 group (28 bytes) + u32 (32 bytes)
/// provides natural 8-byte alignment for the trailing u64 fields.
pub const CFullSessionConfig = extern struct {
    learning_rate: f32,
    min_learning_rate: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    max_grad_norm: f32,
    batch_size: u32,
    warmup_steps: u64,
    total_steps: u64,
};

/// Session info snapshot (C-compatible).
pub const CFullSessionInfo = extern struct {
    current_step: u64,
    total_steps: u64,
    total_params: u64,
    batch_size: u32,
    state: u8,
    _padding: [3]u8 = .{0} ** 3,
};

// =============================================================================
// Opaque Handle
// =============================================================================

/// Opaque handle for full training sessions.
pub const TaluTrainFullSession = opaque {};

// =============================================================================
// Type Conversion
// =============================================================================

fn toTransformerConfig(c: *const CTransformerConfig) TransformerConfig {
    return .{
        .vocab_size = c.vocab_size,
        .d_model = c.d_model,
        .num_layers = c.num_layers,
        .num_heads = c.num_heads,
        .num_kv_heads = c.num_kv_heads,
        .d_ff = c.d_ff,
        .seq_len = c.seq_len,
        .rope_theta = c.rope_theta,
        .norm_eps = c.norm_eps,
    };
}

fn toFullSessionConfig(c: *const CFullSessionConfig) FullSessionConfig {
    return .{
        .learning_rate = c.learning_rate,
        .min_learning_rate = c.min_learning_rate,
        .weight_decay = c.weight_decay,
        .beta1 = c.beta1,
        .beta2 = c.beta2,
        .epsilon = c.epsilon,
        .batch_size = c.batch_size,
        .warmup_steps = c.warmup_steps,
        .total_steps = c.total_steps,
        .max_grad_norm = c.max_grad_norm,
    };
}

fn toCFullSessionInfo(info: FullSessionInfo) CFullSessionInfo {
    return .{
        .current_step = info.current_step,
        .total_steps = info.total_steps,
        .total_params = info.total_params,
        .batch_size = info.batch_size,
        .state = @intFromEnum(info.state),
    };
}

// =============================================================================
// Session Lifecycle
// =============================================================================

/// Create a new full training session.
///
/// Returns 0 on success, non-zero error code on failure.
/// Caller must free via talu_train_full_destroy().
pub export fn talu_train_full_create(
    out: ?*?*TaluTrainFullSession,
) callconv(.c) i32 {
    capi_error.clearError();

    const out_ptr = out orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    };
    out_ptr.* = null;

    const session = allocator.create(FullTrainingSession) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate full training session", .{});
        return @intFromEnum(ErrorCode.out_of_memory);
    };

    session.* = FullTrainingSession.init(allocator);
    out_ptr.* = @ptrCast(session);
    return 0;
}

/// Destroy a full training session. Passing null is a safe no-op.
pub export fn talu_train_full_destroy(
    handle: ?*TaluTrainFullSession,
) callconv(.c) void {
    const session: *FullTrainingSession = @ptrCast(@alignCast(handle orelse return));
    session.deinit();
    allocator.destroy(session);
}

// =============================================================================
// Model & Configuration
// =============================================================================

/// Initialize model with random weights.
///
/// Transitions: created → initialized.
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_train_full_init_model(
    handle: ?*TaluTrainFullSession,
    config: ?*const CTransformerConfig,
    seed: u64,
) callconv(.c) i32 {
    capi_error.clearError();

    const session: *FullTrainingSession = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }));

    const cfg = config orelse {
        capi_error.setErrorWithCode(.invalid_argument, "config is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    };

    session.initModel(toTransformerConfig(cfg), seed) catch |err| {
        capi_error.setError(err, "init_model failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

/// Set training hyperparameters.
///
/// Transitions: initialized → configured.
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_train_full_configure(
    handle: ?*TaluTrainFullSession,
    config: ?*const CFullSessionConfig,
) callconv(.c) i32 {
    capi_error.clearError();

    const session: *FullTrainingSession = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }));

    const cfg = config orelse {
        capi_error.setErrorWithCode(.invalid_argument, "config is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    };

    session.configure(toFullSessionConfig(cfg)) catch |err| {
        capi_error.setError(err, "configure failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

// =============================================================================
// Data Loading
// =============================================================================

/// Set tokenized training data from an in-memory buffer.
///
/// The caller retains ownership of `tokens_ptr`; it must outlive the session.
/// Transitions: configured → data_loaded.
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_train_full_set_data(
    handle: ?*TaluTrainFullSession,
    tokens_ptr: ?[*]const u32,
    tokens_len: usize,
) callconv(.c) i32 {
    capi_error.clearError();

    const session: *FullTrainingSession = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }));

    const ptr = tokens_ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, "tokens_ptr is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    };

    if (tokens_len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "tokens_len is 0", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }

    session.setData(ptr[0..tokens_len]) catch |err| {
        capi_error.setError(err, "set_data failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

/// Load tokenized training data from a flat binary file of u32 tokens.
///
/// Transitions: configured → data_loaded.
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_train_full_load_data(
    handle: ?*TaluTrainFullSession,
    data_path: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const session: *FullTrainingSession = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }));

    const path = data_path orelse {
        capi_error.setErrorWithCode(.invalid_argument, "data_path is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    };

    const path_slice = std.mem.sliceTo(path, 0);
    session.loadData(path_slice) catch |err| {
        capi_error.setError(err, "load_data failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

// =============================================================================
// Training
// =============================================================================

/// Run one training step: forward → backward → clip → optimizer.
///
/// Returns step metrics through `out_metrics` (may be null to discard).
/// Transitions: data_loaded → training (on first call).
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_train_full_step(
    handle: ?*TaluTrainFullSession,
    out_metrics: ?*CStepMetrics,
) callconv(.c) i32 {
    capi_error.clearError();

    const session: *FullTrainingSession = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }));

    const metrics = session.trainStep() catch |err| {
        capi_error.setError(err, "train_step failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    if (out_metrics) |out| {
        out.* = capi_bridge.toCStepMetrics(metrics);
    }

    return 0;
}

/// Run the full training loop.
///
/// `callback` is called after each step. Return 0 to continue, non-zero to
/// cancel. Passing null runs without callbacks.
///
/// Transitions: data_loaded → training → completed.
/// Returns 0 on success, train_cancelled (1010) if cancelled via callback.
pub export fn talu_train_full_run(
    handle: ?*TaluTrainFullSession,
    callback: ?CStepCallback,
    user_data: ?*anyopaque,
) callconv(.c) i32 {
    capi_error.clearError();

    const session: *FullTrainingSession = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }));

    if (callback) |cb| {
        var wrapper = CallbackWrapper{
            .c_callback = cb,
            .c_user_data = user_data,
        };
        session.run(bridgeCallback, @ptrCast(&wrapper)) catch |err| {
            capi_error.setError(err, "training failed", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
    } else {
        session.run(null, null) catch |err| {
            capi_error.setError(err, "training failed", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
    }

    return 0;
}

// =============================================================================
// Info / Query
// =============================================================================

/// Query current session state.
///
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_train_full_get_info(
    handle: ?*TaluTrainFullSession,
    out_info: ?*CFullSessionInfo,
) callconv(.c) i32 {
    capi_error.clearError();

    const session: *FullTrainingSession = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }));

    const out = out_info orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_info is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    };

    out.* = toCFullSessionInfo(session.getInfo());
    return 0;
}

/// Copy current model weights into a caller-provided flat f32 buffer.
///
/// The buffer length must equal `talu_train_full_get_info(...).total_params`.
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_train_full_copy_weights_f32(
    handle: ?*TaluTrainFullSession,
    out_weights: ?[*]f32,
    out_len: usize,
) callconv(.c) i32 {
    capi_error.clearError();

    const session: *FullTrainingSession = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }));

    const expected_len: usize = @intCast(session.getInfo().total_params);
    if (out_len != expected_len) {
        capi_error.setErrorWithCode(.invalid_argument, "out_len does not match total_params", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }

    if (out_len == 0) {
        const empty = [_]f32{};
        session.copyWeightsF32(empty[0..]) catch |err| {
            capi_error.setError(err, "copy_weights_f32 failed", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
        return 0;
    }

    const out_ptr = out_weights orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_weights is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    };

    session.copyWeightsF32(out_ptr[0..out_len]) catch |err| {
        capi_error.setError(err, "copy_weights_f32 failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Load model weights and step counter from a flat f32 buffer.
pub export fn talu_train_full_load_weights_f32(
    handle: ?*TaluTrainFullSession,
    in_weights: ?[*]const f32,
    in_len: usize,
    step: u64,
) callconv(.c) i32 {
    capi_error.clearError();

    const session: *FullTrainingSession = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }));

    const expected_len: usize = @intCast(session.getInfo().total_params);
    if (in_len != expected_len) {
        capi_error.setErrorWithCode(.invalid_argument, "in_len does not match total_params", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }

    if (in_len == 0) {
        const empty = [_]f32{};
        session.loadWeightsF32(empty[0..], step) catch |err| {
            capi_error.setError(err, "load_weights_f32 failed", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
        return 0;
    }

    const in_ptr = in_weights orelse {
        capi_error.setErrorWithCode(.invalid_argument, "in_weights is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    };

    session.loadWeightsF32(in_ptr[0..in_len], step) catch |err| {
        capi_error.setError(err, "load_weights_f32 failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Copy Adam optimizer state into a caller-provided flat f32 buffer.
pub export fn talu_train_full_copy_optimizer_state_f32(
    handle: ?*TaluTrainFullSession,
    out_state: ?[*]f32,
    out_len: usize,
) callconv(.c) i32 {
    capi_error.clearError();

    const session: *FullTrainingSession = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }));

    const expected_len = session.optimizerStateLen() catch |err| {
        capi_error.setError(err, "copy_optimizer_state_f32 failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    if (out_len != expected_len) {
        capi_error.setErrorWithCode(.invalid_argument, "out_len does not match optimizer state size", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }

    if (out_len == 0) return 0;

    const out_ptr = out_state orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_state is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    };

    session.copyOptimizerStateF32(out_ptr[0..out_len]) catch |err| {
        capi_error.setError(err, "copy_optimizer_state_f32 failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Load Adam optimizer state from a flat f32 buffer.
pub export fn talu_train_full_load_optimizer_state_f32(
    handle: ?*TaluTrainFullSession,
    in_state: ?[*]const f32,
    in_len: usize,
) callconv(.c) i32 {
    capi_error.clearError();

    const session: *FullTrainingSession = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }));

    const expected_len = session.optimizerStateLen() catch |err| {
        capi_error.setError(err, "load_optimizer_state_f32 failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    if (in_len != expected_len) {
        capi_error.setErrorWithCode(.invalid_argument, "in_len does not match optimizer state size", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }

    if (in_len == 0) return 0;

    const in_ptr = in_state orelse {
        capi_error.setErrorWithCode(.invalid_argument, "in_state is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    };

    session.loadOptimizerStateF32(in_ptr[0..in_len]) catch |err| {
        capi_error.setError(err, "load_optimizer_state_f32 failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
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
    var c_metrics = capi_bridge.toCStepMetrics(metrics);
    return wrapper.c_callback(&c_metrics, wrapper.c_user_data);
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "CTransformerConfig conversion preserves values" {
    const c_cfg = CTransformerConfig{
        .vocab_size = 32000,
        .d_model = 256,
        .num_layers = 4,
        .num_heads = 4,
        .num_kv_heads = 2,
        .d_ff = 1024,
        .seq_len = 128,
        .rope_theta = 10000.0,
        .norm_eps = 1e-5,
    };
    const zig_cfg = toTransformerConfig(&c_cfg);

    try testing.expectEqual(@as(u32, 32000), zig_cfg.vocab_size);
    try testing.expectEqual(@as(u32, 256), zig_cfg.d_model);
    try testing.expectEqual(@as(u32, 4), zig_cfg.num_layers);
    try testing.expectEqual(@as(u32, 4), zig_cfg.num_heads);
    try testing.expectEqual(@as(u32, 2), zig_cfg.num_kv_heads);
    try testing.expectEqual(@as(u32, 1024), zig_cfg.d_ff);
    try testing.expectEqual(@as(u32, 128), zig_cfg.seq_len);
    try testing.expectApproxEqAbs(@as(f32, 10000.0), zig_cfg.rope_theta, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1e-5), zig_cfg.norm_eps, 1e-12);
}

test "CFullSessionConfig conversion preserves values" {
    const c_cfg = CFullSessionConfig{
        .learning_rate = 3e-4,
        .min_learning_rate = 3e-5,
        .weight_decay = 0.1,
        .beta1 = 0.9,
        .beta2 = 0.95,
        .epsilon = 1e-8,
        .max_grad_norm = 1.0,
        .batch_size = 32,
        .warmup_steps = 500,
        .total_steps = 10000,
    };
    const zig_cfg = toFullSessionConfig(&c_cfg);

    try testing.expectApproxEqAbs(@as(f32, 3e-4), zig_cfg.learning_rate, 1e-10);
    try testing.expectApproxEqAbs(@as(f32, 3e-5), zig_cfg.min_learning_rate, 1e-10);
    try testing.expectApproxEqAbs(@as(f32, 0.1), zig_cfg.weight_decay, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.9), zig_cfg.beta1, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.95), zig_cfg.beta2, 1e-6);
    try testing.expectEqual(@as(u32, 32), zig_cfg.batch_size);
    try testing.expectEqual(@as(u64, 500), zig_cfg.warmup_steps);
    try testing.expectEqual(@as(u64, 10000), zig_cfg.total_steps);
}

test "CFullSessionInfo conversion and padding" {
    const info = FullSessionInfo{
        .state = .configured,
        .current_step = 42,
        .total_steps = 1000,
        .total_params = 4_000_000,
        .batch_size = 32,
    };
    const c_info = toCFullSessionInfo(info);

    try testing.expectEqual(@as(u64, 42), c_info.current_step);
    try testing.expectEqual(@as(u64, 1000), c_info.total_steps);
    try testing.expectEqual(@as(u64, 4_000_000), c_info.total_params);
    try testing.expectEqual(@as(u32, 32), c_info.batch_size);
    try testing.expectEqual(@as(u8, 2), c_info.state); // configured = 2
    for (c_info._padding) |b| {
        try testing.expectEqual(@as(u8, 0), b);
    }
}

test "CFullSessionConfig has no implicit padding" {
    // Verify the struct size matches the sum of field sizes.
    // 7 × f32 (28) + 1 × u32 (4) + 2 × u64 (16) = 48
    try testing.expectEqual(@as(usize, 48), @sizeOf(CFullSessionConfig));
}

test "CTransformerConfig has no implicit padding" {
    // 7 × u32 (28) + 2 × f32 (8) = 36
    try testing.expectEqual(@as(usize, 36), @sizeOf(CTransformerConfig));
}

test "CFullSessionInfo size" {
    // 3 × u64 (24) + 1 × u32 (4) + 1 × u8 (1) + 3 × padding (3) = 32
    try testing.expectEqual(@as(usize, 32), @sizeOf(CFullSessionInfo));
}
