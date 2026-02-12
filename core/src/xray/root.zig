//! Inspect Subsystem
//!
//! Introspection and performance analysis tools for the inference engine.
//!
//! - `trace` - Tensor tracing contract (emit points during inference)
//! - `capture` - Capture configuration and storage
//! - `query` - Query engine for captured data
//! - `stats` - Statistics computation
//! - `kernel_info` - Kernel operation description and analysis
//! - `perf_estimate` - Performance estimation (FLOPs, memory bandwidth)
//! - `execution_plan` - Static analysis of kernel selection from config

// Tensor inspection system
pub const trace = @import("trace.zig");
pub const capture = @import("capture.zig");
pub const query = @import("query.zig");
pub const stats = @import("stats.zig");

// Re-export commonly used types for tensor inspection
pub const TracePoint = trace.TracePoint;
pub const TraceEmission = trace.TraceEmission;
pub const TracedTensor = trace.TracedTensor;
pub const TensorStats = stats.TensorStats;
pub const TraceCapture = capture.TraceCapture;
pub const TraceCaptureConfig = capture.TraceCaptureConfig;
pub const TraceCaptureMode = capture.TraceCaptureMode;
pub const TracePointSet = capture.TracePointSet;
pub const CaptureQuery = query.CaptureQuery;

// Kernel/perf analysis
pub const kernel_info = @import("kernel_info.zig");
pub const perf_estimate = @import("perf_estimate.zig");
pub const execution_plan = @import("execution_plan.zig");

// Re-export commonly used types
pub const KernelInfo = kernel_info.KernelInfo;
pub const PerfEstimate = perf_estimate.PerfEstimate;
pub const ExecutionPlan = execution_plan.ExecutionPlan;

// Re-export behavioral types so check_coverage.sh --integration can verify test coverage
pub const KernelOp = kernel_info.KernelOp;
pub const ShapeDim = kernel_info.ShapeDim;
pub const LayerGeometry = perf_estimate.LayerGeometry;
pub const EstimateArgs = perf_estimate.EstimateArgs;
pub const AttnConfig = perf_estimate.AttnConfig;
pub const FfnConfig = perf_estimate.FfnConfig;

// Execution plan types
pub const MatmulKernel = execution_plan.MatmulKernel;
pub const AttentionType = execution_plan.AttentionType;
pub const FfnType = execution_plan.FfnType;
pub const ExecutionPlanConfig = execution_plan.ModelConfig;

// ============================================================================
// Convenience API
// ============================================================================

/// Enable tensor capture with the given configuration.
pub fn enableCapture(cap: *TraceCapture) void {
    capture.enable(cap);
}

/// Disable tensor capture.
pub fn disableCapture() void {
    capture.disable();
}

/// Check if tensor capture is enabled.
pub fn isCaptureEnabled() bool {
    return capture.isEnabled();
}

/// Check if tracing is active (handler installed).
pub fn isTraceEnabled() bool {
    return trace.isEnabled();
}
