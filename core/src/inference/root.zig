//! Inference - sampling, scheduling, loading, and backend execution.
//!
//! This module is the inference boundary used by router/bindings:
//! - `types` - generation request/result types
//! - `sampling` - token sampling policies
//! - `scheduler` - continuous batching runtime
//! - `backend` - CPU/Metal inference backends

const std = @import("std");

pub const sampling = @import("sampling.zig");
pub const scheduler = @import("scheduler.zig");
pub const config = @import("config/root.zig");
pub const vision_types = @import("vision_types.zig");

/// Generation request/response types.
pub const types = struct {
    /// Callback function type for streaming token output.
    /// Called with each newly generated token ID and optional user data.
    pub const TokenCallback = *const fn (token_id: u32, user_data: ?*anyopaque) void;

    /// Request configuration for single generation runs.
    pub const InferenceConfig = struct {
        max_new_tokens: usize = 32,
        sampling: sampling.SamplingConfig = .{},
        eos_token_ids: []const u32 = &.{},
        /// BOS token to prepend to input (from model config)
        bos_token_id: ?u32 = null,
        /// Optional callback for streaming output. Called after each token is sampled.
        token_callback: ?types.TokenCallback = null,
        /// User data passed to the token callback.
        callback_data: ?*anyopaque = null,
        /// Stop sequences (already tokenized). Generation stops when any sequence matches.
        /// Each inner slice is a tokenized stop sequence.
        stop_sequences: []const []const u32 = &.{},
        /// Optional stop flag for cancellation. When set to true, generation stops.
        stop_flag: ?*const std.atomic.Value(bool) = null,
    };

    /// Reason why generation stopped.
    pub const FinishReason = enum(u8) {
        /// Generation stopped due to EOS token.
        eos_token = 0,
        /// Maximum token limit reached.
        length = 1,
        /// A stop sequence was matched.
        stop_sequence = 2,
        /// Model requested tool/function calls.
        tool_calls = 3,
        /// Content was filtered (safety).
        content_filter = 4,
        /// Request was cancelled (e.g., client disconnect, stop flag set).
        cancelled = 5,

        /// Convert to C-compatible integer for C-API.
        pub fn toInt(self: types.FinishReason) u8 {
            return @intFromEnum(self);
        }
    };

    /// Full state returned by low-level `run()` APIs.
    pub const InferenceState = struct {
        tokens: []u32,
        final_logits: []f32,
        prompt_len: usize,
        generated_len: usize,
        prefill_ns: u64,
        decode_ns: u64,
        finish_reason: types.FinishReason = .eos_token,
    };
};

/// Pooling strategy for embedding extraction.
pub const PoolingStrategy = @import("backend/contract.zig").PoolingStrategy;
pub const pooling = struct {
    pub const PoolingStrategy = @import("backend/contract.zig").PoolingStrategy;
};

// Re-export common generation types
pub const InferenceConfig = types.InferenceConfig;
pub const InferenceState = types.InferenceState;
pub const TokenCallback = types.TokenCallback;
pub const FinishReason = types.FinishReason;

pub const Sampler = sampling.Sampler;
pub const SamplingConfig = sampling.SamplingConfig;
pub const SamplingStrategy = sampling.SamplingStrategy;

pub const Scheduler = scheduler.Scheduler;
pub const SchedulerConfig = scheduler.SchedulerConfig;
pub const RequestState = scheduler.RequestState;
pub const TokenEvent = scheduler.TokenEvent;
pub const Request = scheduler.Request;
pub const SchedulerFinishReason = scheduler.FinishReason;
pub const GenerateSyncResult = scheduler.Scheduler.GenerateSyncResult;
pub const SchedulerSubmitOptions = scheduler.Scheduler.SubmitOptions;

// Re-export sampling behavioral types so check_coverage.sh --integration can verify test coverage
pub const SamplingWorkspace = sampling.Workspace;

// =============================================================================
// Internal API (for core/src/ only)
// =============================================================================

/// Backend implementations
pub const backend = struct {
    pub const cpu = @import("backend/cpu/root.zig");
    pub const metal = @import("backend/metal/root.zig");
    // Re-export inference behavioral type used by dump/xray tooling.
    pub const FusedCpuBackend = cpu.FusedCpuBackend;
};
