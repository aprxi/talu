//! Integration tests for the inference module.
//!
//! Tests cover all exports from inference/root.zig:
//! - Session, FinishReason, InferenceState
//! - Sampler, SamplingConfig, SamplingStrategy, SamplingWorkspace
//! - Scheduler, SchedulerConfig, RequestState, TokenEvent, Request
//! - Backend types: FusedCpuBackend, MultiHeadAttention, SwiGLU, etc.
//! - Scratch buffers: AttnTemp, AttnCache, FfnScratch, ScratchBuffer

test {
    // Core inference types
    _ = @import("finish_reason_test.zig");
    _ = @import("sampler_test.zig");
    _ = @import("sampling_workspace_test.zig");
    _ = @import("session_test.zig");

    // Scheduler types
    _ = @import("scheduler_test.zig");
    _ = @import("request_test.zig");

    // Backend types
    _ = @import("fused_cpu_backend_test.zig");
    _ = @import("multi_head_attention_test.zig");
    _ = @import("swi_g_l_u_test.zig");
    _ = @import("ffn_layer_test.zig");
    _ = @import("transformer_block_test.zig");

    // Scratch buffer types
    _ = @import("attn_temp_test.zig");
    _ = @import("attn_cache_test.zig");
    _ = @import("ffn_scratch_test.zig");
    _ = @import("scratch_buffer_test.zig");

    // Executor submodule
    _ = @import("executor/root.zig");

    // Backend submodules
    _ = @import("backend/root.zig");
}
