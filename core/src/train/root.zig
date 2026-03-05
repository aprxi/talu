//! Training module for talu.
//!
//! Provides training, fine-tuning, and LoRA adapter support for any model
//! architecture talu supports. Adapters match against WeightSpec IDs so they
//! work model-agnostically across Llama, Qwen, Gemma, Granite, MoE, etc.
//!
//! This module imports from compute/ and models/ but NOT from inference/.
//! Training uses its own forward pass that saves activations for backward.
//!
//! Public API:
//!   - grad       - GradTensor: f32 gradient storage
//!   - adapter    - LoRA adapter types and pattern matching
//!   - backward   - Backward pass kernels for each operation type

const std = @import("std");

// =============================================================================
// Public API
// =============================================================================

pub const grad = @import("grad.zig");
pub const adapter = @import("adapter.zig");
pub const backward = @import("backward/root.zig");

pub const loss = @import("loss.zig");

pub const optimizer = @import("optimizer.zig");
pub const scheduler = @import("scheduler.zig");
pub const param_state = @import("param_state.zig");

pub const data = @import("data.zig");
pub const checkpoint = @import("checkpoint.zig");
pub const loop = @import("loop.zig");
pub const session = @import("session.zig");
pub const capi_bridge = @import("capi_bridge.zig");

pub const model_config = @import("model_config.zig");
pub const model_weights = @import("model_weights.zig");
pub const activations = @import("activations.zig");
pub const forward = @import("forward.zig");
pub const backward_pass = @import("backward_pass.zig");
pub const full_session = @import("full_session.zig");

// =============================================================================
// Re-exported types
// =============================================================================

pub const GradTensor = grad.GradTensor;
pub const LoraLayer = adapter.LoraLayer;
pub const LoraAdapter = adapter.LoraAdapter;
pub const LoraConfig = adapter.LoraConfig;
pub const TargetPattern = adapter.TargetPattern;

pub const crossEntropyLoss = loss.crossEntropyLoss;
pub const AdamW = optimizer.AdamW;
pub const AdamWConfig = optimizer.AdamWConfig;
pub const Scheduler = scheduler.Scheduler;
pub const SchedulerConfig = scheduler.SchedulerConfig;
pub const TrainableParam = param_state.TrainableParam;
pub const TrainableParams = param_state.TrainableParams;
pub const DataLoader = data.DataLoader;
pub const Batch = data.Batch;
pub const TrainingConfig = loop.TrainingConfig;
pub const StepMetrics = loop.StepMetrics;
pub const clipGradNorm = loop.clipGradNorm;
pub const TrainingSession = session.TrainingSession;
pub const SessionConfig = session.SessionConfig;
pub const SessionInfo = session.SessionInfo;
pub const TransformerConfig = model_config.TransformerConfig;
pub const ModelWeights = model_weights.ModelWeights;
pub const LayerWeights = model_weights.LayerWeights;
pub const ActivationCache = activations.ActivationCache;
pub const FullTrainingSession = full_session.FullTrainingSession;
pub const FullSessionConfig = full_session.FullSessionConfig;
pub const FullSessionInfo = full_session.FullSessionInfo;

test {
    _ = grad;
    _ = adapter;
    _ = backward;
    _ = loss;
    _ = optimizer;
    _ = scheduler;
    _ = param_state;
    _ = data;
    _ = checkpoint;
    _ = loop;
    _ = session;
    _ = capi_bridge;
    _ = model_config;
    _ = model_weights;
    _ = activations;
    _ = forward;
    _ = backward_pass;
    _ = full_session;
}
