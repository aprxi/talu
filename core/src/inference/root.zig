//! Inference - sampling, scheduling, loading, and backend execution.
//!
//! This module is the inference boundary used by responses/bindings:
//! - `types` - generation request/result types
//! - `sampling` - token sampling policies
//! - `scheduler` - continuous batching runtime
//! - `backend` - CPU/Metal/CUDA inference backends

const std = @import("std");

pub const sampling = @import("sampling.zig");
pub const scheduler = @import("scheduler.zig");
pub const vision_types = @import("vision_types.zig");
pub const runtime_contract = @import("runtime_contract_pkg");
pub const bridge = @import("bridge/root.zig");
pub const transport = @import("transport/root.zig");
pub const diagnostics = @import("diagnostics/root.zig");

/// Generation request/response types.
pub const types = struct {
    /// Callback function type for streaming token output.
    /// Called with each newly generated token ID and optional user data.
    pub const TokenCallback = *const fn (token_id: u32, in_thinking: bool, user_data: ?*anyopaque) void;

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
pub const GenerateSyncResult = scheduler.Scheduler.GenerateSyncResult;
pub const SchedulerSubmitOptions = scheduler.Scheduler.SubmitOptions;

// =============================================================================
// Internal API (for core/src/ only)
// =============================================================================

/// Backend implementations
pub const backend = struct {
    pub const cpu = @import("backend/cpu/root.zig");
    pub const metal = @import("backend/metal/root.zig");
    pub const cuda = @import("backend/cuda/root.zig");
    pub const local_runtime = @import("backend/local_runtime.zig");
    pub const local_stage = @import("backend/local_stage.zig");
    pub const local_stage_adapters = @import("backend/local_stage_adapters.zig");
    pub const local_decode_pipeline = @import("backend/local_decode_pipeline.zig");
    pub const local_prefill_pipeline = @import("backend/local_prefill_pipeline.zig");
    // Re-export inference behavioral type used by dump/xray tooling.
    pub const FusedCpuBackend = cpu.FusedCpuBackend;
};

test "inference backend local_runtime buildPlan bridge-neutral runtime facts" {
    const local_runtime = backend.local_runtime;
    const stages = [_]@import("backend/topology.zig").LocalStageSpec{
        .{ .backend_kind = .cpu, .layer_start = 0, .layer_end = 1 },
        .{ .backend_kind = .cuda, .device_ordinal = 0, .layer_start = 1, .layer_end = 3 },
        .{ .backend_kind = .cpu, .layer_start = 3, .layer_end = 4 },
    };
    const caps = [_]local_runtime.StageRuntimeCapability{
        .{ .stage_id = 0, .backend_kind = .cpu, .max_batch_size = 2, .prefill_chunk_rows_cap = 5, .supported_boundary_dtypes = &.{.f32} },
        .{ .stage_id = 1, .backend_kind = .cuda, .max_batch_size = 3, .prefill_chunk_rows_cap = 4, .supported_boundary_dtypes = &.{.f32} },
        .{ .stage_id = 2, .backend_kind = .cpu, .max_batch_size = 1, .prefill_chunk_rows_cap = 6, .supported_boundary_dtypes = &.{.f32} },
    };
    var plan = try local_runtime.buildPlan(.{
        .allocator = std.testing.allocator,
        .d_model = 8,
        .total_layers = 4,
        .stages = &stages,
        .stage_capabilities = &caps,
        .boundary_peer_copy_available = &.{ false, false },
    });
    defer plan.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, plan.split_points);
    try std.testing.expectEqual(bridge.HostBackendKind.cpu, plan.stage_specs[0].backend_kind);
    try std.testing.expectEqual(bridge.HostBackendKind.cuda, plan.stage_specs[1].backend_kind);
    try std.testing.expectEqual(bridge.HostBackendKind.cpu, plan.stage_specs[2].backend_kind);
    try std.testing.expect(plan.boundary_runtimes[0].staging != null);
    try std.testing.expect(plan.boundary_runtimes[1].staging != null);
}
