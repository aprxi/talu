//! Training C API — thin boundary layer.
//!
//! Exports 8 C-callable functions for the training session lifecycle:
//!   talu_train_create, talu_train_destroy, talu_train_load_model,
//!   talu_train_configure, talu_train_load_data, talu_train_run,
//!   talu_train_save_checkpoint, talu_train_get_info
//!
//! Each function: clearError → validate → delegate to capi_bridge → catch → setError → return code.

const std = @import("std");
const train_mod = @import("../train/root.zig");
const capi_bridge = train_mod.capi_bridge;
const session_mod = train_mod.session;

const TrainingSession = session_mod.TrainingSession;
const CLoraConfig = capi_bridge.CLoraConfig;
const CTrainingConfig = capi_bridge.CTrainingConfig;
const CStepMetrics = capi_bridge.CStepMetrics;
const CTrainingInfo = capi_bridge.CTrainingInfo;
const CStepCallback = capi_bridge.CStepCallback;

const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");
const ErrorCode = error_codes.ErrorCode;

const allocator = std.heap.c_allocator;

/// Opaque handle for training sessions.
pub const TaluTrainSession = opaque {};

// =============================================================================
// Session Lifecycle
// =============================================================================

/// Create a new training session.
///
/// Returns 0 on success, non-zero error code on failure.
/// Caller must free via talu_train_destroy().
pub export fn talu_train_create(
    out: ?*?*TaluTrainSession,
) callconv(.c) i32 {
    capi_error.clearError();

    const out_ptr = out orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    };
    out_ptr.* = null;

    const session = capi_bridge.createSession(allocator) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate training session", .{});
        return @intFromEnum(ErrorCode.out_of_memory);
    };

    out_ptr.* = @ptrCast(session);
    return 0;
}

/// Destroy a training session. Passing null is a safe no-op.
pub export fn talu_train_destroy(
    handle: ?*TaluTrainSession,
) callconv(.c) void {
    const session: *TrainingSession = @ptrCast(@alignCast(handle orelse return));
    capi_bridge.destroySession(allocator, session);
}

// =============================================================================
// Session Configuration
// =============================================================================

/// Load a model and create a LoRA adapter.
///
/// `target_modules` is an array of null-terminated C strings specifying which
/// weight IDs to adapt (matched by substring). For example: {"q_proj", "v_proj"}.
///
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_train_load_model(
    handle: ?*TaluTrainSession,
    model_path: ?[*:0]const u8,
    lora_config: ?*const CLoraConfig,
    target_modules: ?[*]const [*:0]const u8,
    num_targets: usize,
) callconv(.c) i32 {
    capi_error.clearError();

    _ = model_path;
    _ = lora_config;
    _ = target_modules;
    _ = num_targets;

    const session: *TrainingSession = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }));
    _ = session;

    // Model loading requires integration with model I/O subsystem.
    // For now, use talu_train_set_adapter (or call setAdapter via the bridge)
    // to set up adapters explicitly.
    capi_error.setErrorWithCode(.train_model_load_failed, "model loading not yet implemented; use explicit adapter setup", .{});
    return @intFromEnum(ErrorCode.train_model_load_failed);
}

/// Set training hyperparameters.
///
/// Requires a model/adapter to be loaded first.
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_train_configure(
    handle: ?*TaluTrainSession,
    config: ?*const CTrainingConfig,
) callconv(.c) i32 {
    capi_error.clearError();

    const session: *TrainingSession = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }));

    const cfg = config orelse {
        capi_error.setErrorWithCode(.invalid_argument, "config is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    };

    capi_bridge.configureSession(session, cfg) catch |err| {
        capi_error.setError(err, "configure failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

// =============================================================================
// Data Loading
// =============================================================================

/// Load tokenized training data from a file.
///
/// The file must contain flat binary u32 tokens (native endianness).
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_train_load_data(
    handle: ?*TaluTrainSession,
    data_path: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const session: *TrainingSession = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }));

    const path = data_path orelse {
        capi_error.setErrorWithCode(.invalid_argument, "data_path is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    };

    capi_bridge.loadData(session, path) catch |err| {
        capi_error.setError(err, "data loading failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

// =============================================================================
// Training
// =============================================================================

/// Run the training loop.
///
/// `callback` is called after each optimizer step. Return 0 to continue,
/// non-zero to cancel training. Passing null runs without callbacks.
///
/// Returns 0 on success, non-zero error code on failure.
/// Returns train_cancelled (1010) if cancelled via callback.
pub export fn talu_train_run(
    handle: ?*TaluTrainSession,
    callback: ?CStepCallback,
    user_data: ?*anyopaque,
) callconv(.c) i32 {
    capi_error.clearError();

    const session: *TrainingSession = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }));

    capi_bridge.trainSession(session, callback, user_data) catch |err| {
        capi_error.setError(err, "training failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

// =============================================================================
// Checkpointing
// =============================================================================

/// Save adapter weights to a file.
///
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_train_save_checkpoint(
    handle: ?*TaluTrainSession,
    output_path: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const session: *TrainingSession = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }));
    _ = session;
    _ = output_path;

    // Checkpoint saving requires SafeTensors I/O integration.
    capi_error.setErrorWithCode(.train_checkpoint_save_failed, "checkpoint saving not yet implemented", .{});
    return @intFromEnum(ErrorCode.train_checkpoint_save_failed);
}

// =============================================================================
// Info / Query
// =============================================================================

/// Query current training session state.
///
/// Returns 0 on success, non-zero error code on failure.
pub export fn talu_train_get_info(
    handle: ?*TaluTrainSession,
    out_info: ?*CTrainingInfo,
) callconv(.c) i32 {
    capi_error.clearError();

    const session: *TrainingSession = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    }));

    const out = out_info orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_info is null", .{});
        return @intFromEnum(ErrorCode.invalid_argument);
    };

    out.* = capi_bridge.getInfo(session);
    return 0;
}
