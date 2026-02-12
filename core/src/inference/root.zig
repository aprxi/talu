//! Inference - Sampling, scheduling, and inference execution.
//!
//! This module provides infrastructure for inference:
//!
//! - `Sampler` - Token sampling strategies
//! - `Scheduler` - Continuous batching
//! - `Session` - Legacy inference session (deprecated)
//!
//! For chat state and responses, see `responses/` module.
//! For the inference engine, see `router/` module.
//!
//! ```zig
//! const router = @import("router/root.zig");
//! const responses = @import("responses/root.zig");
//!
//! // Load model once
//! var engine = try router.LocalEngine.init(allocator, "path/to/model");
//! defer engine.deinit();
//!
//! // Create lightweight chats per user (from responses module)
//! var chat = responses.Chat.init(allocator);
//! defer chat.deinit();
//! try chat.append(.user, "Hello!");
//!
//! // Generate response
//! const result = try engine.generate(&chat, .{});
//! defer result.deinit(allocator);
//! ```

// =============================================================================
// Public API
// =============================================================================

// Inference session and sampling
pub const session = @import("session.zig");
pub const sampling = @import("sampling.zig");
pub const scheduler = @import("scheduler.zig");

// Re-export session types
pub const Session = session.Session;
pub const InferenceConfig = session.InferenceConfig;
pub const InferenceState = session.InferenceState;
pub const TokenCallback = session.TokenCallback;
pub const FinishReason = session.FinishReason;

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

/// Executor - LayerOp bytecode execution
pub const executor = @import("executor/root.zig");

/// Backend implementations
pub const backend = struct {
    pub const cpu_fused = @import("backend/cpu/fused.zig");

    /// CPU block kernels - weight types and block building
    pub const block_kernels = @import("backend/cpu/block_kernels.zig");

    /// CPU kernel implementations
    pub const kernels = struct {
        pub const moe = @import("backend/cpu/kernels/moe.zig");
        pub const attention = @import("backend/cpu/kernels/attention.zig");
        pub const ffn = @import("backend/cpu/kernels/ffn.zig");
        pub const kv_cache = @import("backend/cpu/kernels/kv_cache.zig");
    };

    // Re-export inference behavioral types so check_coverage.sh --integration can verify test coverage
    pub const FusedCpuBackend = cpu_fused.FusedCpuBackend;
    pub const MultiHeadAttention = block_kernels.MultiHeadAttention;
    pub const SwiGLU = block_kernels.SwiGLU;
    pub const FfnLayer = block_kernels.FfnLayer;
    pub const AttnTemp = block_kernels.AttnTemp;
    pub const AttnCache = block_kernels.AttnCache;
    pub const FfnScratch = block_kernels.FfnScratch;
    pub const ScratchBuffer = block_kernels.ScratchBuffer;
    pub const TransformerBlock = block_kernels.TransformerBlock;
};
