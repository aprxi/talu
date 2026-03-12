//! Tensor Trace Contract
//!
//! Minimal stable interface between inference and inspection systems.
//! This file defines the contract - inference calls trace.emit(), inspection
//! sets trace.handler to receive emissions.
//!
//! Design: Zero overhead when disabled (single null check, inlined away).
//!
//! STABLE CONTRACT - changes here affect both inference and inspection code.

const std = @import("std");

/// Trace points in the inference pipeline.
/// Names correspond to actual method signatures in the codebase.
pub const TracePoint = enum(u8) {
    // Embedding - from EmbeddingKernel.forward()
    embed = 0,
    embed_pos, // After positional encoding

    // Per-layer attention - from AttentionKernel.forward()
    layer_input,
    layer_attn_norm,
    attn_q, // Q projection output
    attn_k, // K projection output
    attn_v, // V projection output
    attn_qk, // Q @ K^T (attention scores before softmax)
    attn_weights, // After softmax
    attn_out, // After output projection

    // Per-layer FFN - from FfnKernel.forward()
    layer_ffn_norm,
    ffn_gate, // Gate/up projection output
    ffn_up,
    ffn_act, // After activation
    ffn_down, // Down projection output

    // Block output - from model.forward()
    block_out, // After residual add

    // Per-layer Mamba - from MambaKernel.forward()
    mamba_out,

    // Per-layer conv (ShortConv) - from ShortConvKernel.forward()
    conv_in_proj, // Input projection output (B, C, x_proj)
    conv_conv, // After depthwise convolution
    conv_out_proj, // After output projection

    // Final
    final_norm,
    lm_head, // From lm_head matmul
    logits_scaled, // After temperature scaling
    logits_ready, // Logits materialized and available to sampler
    token_select, // Next-token selection (argmax/sampling)
    ffn_act_map, // Activation map stage (e.g. SiLU/GELU map)
    ffn_act_mix, // Activation mix stage (e.g. gate*up or fused act*mul)
    gdelta_in_proj, // Gated-Delta input projection output
    gdelta_conv, // Gated-Delta depthwise conv output (pre-SiLU)
    gdelta_ssm, // Gated-Delta state-space step output
    gdelta_norm, // Gated-Delta gated RMS norm output
    gdelta_out, // Gated-Delta output projection

    // Extensible - custom points can use values >= 128
    _,

    pub fn name(self: TracePoint) []const u8 {
        return switch (self) {
            .embed => "embed_tokens",
            .embed_pos => "embed_pos",
            .layer_input => "layer_input",
            .layer_attn_norm => "layer_attn_norm",
            .attn_q => "attn.q",
            .attn_k => "attn.k",
            .attn_v => "attn.v",
            .attn_qk => "attn.qk",
            .attn_weights => "attn.weights",
            .attn_out => "attn.out",
            .layer_ffn_norm => "layer_ffn_norm",
            .ffn_gate => "ffn.gate",
            .ffn_up => "ffn.up",
            .ffn_act => "ffn.act",
            .ffn_down => "ffn.down",
            .block_out => "block.out",
            .mamba_out => "mamba.out",
            .conv_in_proj => "conv.in_proj",
            .conv_conv => "conv.conv",
            .conv_out_proj => "conv.out_proj",
            .final_norm => "final_norm",
            .lm_head => "lm_head",
            .logits_scaled => "logits_scaled",
            .logits_ready => "logits_ready",
            .token_select => "token_select",
            .ffn_act_map => "ffn.act.map",
            .ffn_act_mix => "ffn.act.mix",
            .gdelta_in_proj => "gdelta.in_proj",
            .gdelta_conv => "gdelta.conv",
            .gdelta_ssm => "gdelta.ssm",
            .gdelta_norm => "gdelta.norm",
            .gdelta_out => "gdelta.out",
            _ => "custom",
        };
    }
};

/// Data type enum - mirrors dtype.zig but standalone for stability.
/// Inspection system should not import from inference code.
pub const DType = enum(u8) {
    f32 = 0,
    f64 = 1,
    i32 = 2,
    i64 = 3,
    f16 = 4,
    bf16 = 5,
    i8 = 6,
    i16 = 7,
    u8 = 8,
    u16 = 9,
    u32 = 10,
    u64 = 11,
    grouped_affine_u4 = 25,
    grouped_affine_u8 = 26,

    pub fn name(self: DType) []const u8 {
        return switch (self) {
            .f32 => "f32",
            .f16 => "f16",
            .bf16 => "bf16",
            .grouped_affine_u4 => "q4",
            .grouped_affine_u8 => "q8",
            else => "other",
        };
    }

    pub fn byteSize(self: DType) usize {
        return switch (self) {
            .f32, .i32, .u32 => 4,
            .f64, .i64, .u64 => 8,
            .f16, .bf16, .i16, .u16 => 2,
            .i8, .u8, .grouped_affine_u4, .grouped_affine_u8 => 1,
        };
    }
};

/// Convert compute DType (from dtype.zig) to trace DType.
/// Since both enums have matching u8 values, we can cast directly.
pub fn convertDType(compute_dtype: @import("../dtype.zig").DType) DType {
    return @enumFromInt(@intFromEnum(compute_dtype));
}

/// Compute backend type.
pub const Backend = enum(u8) {
    cpu = 0,
    metal = 1,
    cuda = 2,

    pub fn name(self: Backend) []const u8 {
        return switch (self) {
            .cpu => "cpu",
            .metal => "metal",
            .cuda => "cuda",
        };
    }
};

/// SIMD instruction set variant.
pub const SimdVariant = enum(u8) {
    none = 0,
    sse = 1,
    avx2 = 2,
    avx512 = 3,
    neon = 4,

    pub fn name(self: SimdVariant) []const u8 {
        return switch (self) {
            .none => "",
            .sse => "sse",
            .avx2 => "avx2",
            .avx512 => "avx512",
            .neon => "neon",
        };
    }

    /// Detect SIMD variant based on target CPU.
    pub fn detect() SimdVariant {
        const builtin = @import("builtin");
        const arch = builtin.cpu.arch;
        if (arch == .x86_64 or arch == .x86) {
            const features = builtin.cpu.features;
            if (std.Target.x86.featureSetHas(features, .avx512f)) return .avx512;
            if (std.Target.x86.featureSetHas(features, .avx2)) return .avx2;
            if (std.Target.x86.featureSetHas(features, .sse)) return .sse;
            return .none;
        } else if (arch == .aarch64) {
            if (std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon)) return .neon;
            return .none;
        } else if (arch == .arm) {
            return .neon;
        }
        return .none;
    }
};

/// Reference to tensor data - just enough info to read the tensor.
pub const TracedTensor = struct {
    /// Pointer to raw tensor data
    ptr: [*]const u8,
    /// Data type
    dtype: DType,
    /// Shape dimensions (up to 4D)
    shape: [4]u32,
    /// Number of dimensions
    ndim: u8,

    pub fn elementCount(self: TracedTensor) usize {
        var count: usize = 1;
        for (0..self.ndim) |i| {
            count *= self.shape[i];
        }
        return count;
    }

    pub fn byteSize(self: TracedTensor) usize {
        return self.elementCount() * self.dtype.byteSize();
    }
};

/// A single trace emission - all context needed to identify and capture a tensor.
pub const TraceEmission = struct {
    /// Where in the pipeline
    point: TracePoint,
    /// Which layer (0xFFFF = not applicable, e.g., embed/logits)
    layer: u16,
    /// Token index in current batch
    token: u32,
    /// Position in sequence (for KV cache alignment)
    position: u32,
    /// Backend that emitted this trace point.
    backend: Backend,
    /// The tensor data
    tensor: TracedTensor,
    /// Timestamp for ordering
    timestamp_ns: i128,
    /// Kernel name that produced this tensor (null-terminated, max 48 chars)
    kernel_name: [48]u8,
    /// Exact work counters provided by runtime call sites.
    /// These are semantic workload counters (not hardware counters).
    work_flops: u64 = 0,
    work_bytes: u64 = 0,

    pub const NO_LAYER: u16 = 0xFFFF;
};

/// Handler function type - inspection system implements this.
pub const Handler = *const fn (TraceEmission) void;

/// Runtime-provided work counters attached to a trace emission.
pub const Work = struct {
    flops: u64 = 0,
    bytes: u64 = 0,
};

/// The global handler - set by inspection system, null when disabled.
/// Using atomic to be safe across threads (inspection might be enabled/disabled).
var handler_atomic: std.atomic.Value(?Handler) = std.atomic.Value(?Handler).init(null);
// Active built-in trace points for the current handler. Backends that need to
// materialize host-readable tensors must gate that work on shouldEmit(point),
// not merely on handler presence, otherwise "token-only" verify passes still
// pay for and can be destabilized by unused trace-copy paths.
var built_in_point_mask_atomic: std.atomic.Value(u64) = std.atomic.Value(u64).init(0);
var exact_filter_enabled_atomic: std.atomic.Value(bool) = std.atomic.Value(bool).init(false);
var exact_filter_point_atomic: std.atomic.Value(u8) = std.atomic.Value(u8).init(0);
var exact_filter_layer_atomic: std.atomic.Value(u16) = std.atomic.Value(u16).init(0);
var exact_filter_position_atomic: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);
threadlocal var backend_context: Backend = .cpu;

pub const ExactEmissionFilter = struct {
    point: TracePoint,
    layer: u16,
    position: u32,
};

/// Set the trace handler. Called by inspection system on setup.
pub fn setHandler(h: ?Handler) void {
    handler_atomic.store(h, .release);
    if (h == null) {
        built_in_point_mask_atomic.store(0, .release);
        exact_filter_enabled_atomic.store(false, .release);
    }
}

/// Set the active built-in trace-point mask for the current handler.
pub fn setActiveBuiltInPointMask(mask: u64) void {
    built_in_point_mask_atomic.store(mask, .release);
}

pub fn setActiveExactEmissionFilter(filter: ?ExactEmissionFilter) void {
    if (filter) |value| {
        exact_filter_point_atomic.store(@intFromEnum(value.point), .release);
        exact_filter_layer_atomic.store(value.layer, .release);
        exact_filter_position_atomic.store(value.position, .release);
        exact_filter_enabled_atomic.store(true, .release);
        return;
    }
    exact_filter_enabled_atomic.store(false, .release);
}

/// Get current handler (for inspection system to check if enabled).
pub fn getHandler() ?Handler {
    return handler_atomic.load(.acquire);
}

/// Set backend context for subsequent trace emissions on this thread.
/// Returns previous backend context.
pub fn setBackendContext(backend: Backend) Backend {
    const prev = backend_context;
    backend_context = backend;
    return prev;
}

/// Get backend context for this thread.
pub fn getBackendContext() Backend {
    return backend_context;
}

/// Check if tracing is enabled (for conditional expensive operations).
pub inline fn isEnabled() bool {
    return handler_atomic.load(.acquire) != null;
}

/// Check whether a specific trace point should be emitted by the backend.
/// This is the correct guard for expensive host materialization.
pub inline fn shouldEmit(point: TracePoint) bool {
    if (handler_atomic.load(.acquire) == null) return false;
    const point_idx = @intFromEnum(point);
    if (point_idx >= 64) return true;
    const mask = built_in_point_mask_atomic.load(.acquire);
    return ((mask >> @as(u6, @intCast(point_idx))) & 1) != 0;
}

/// Check whether a specific emission matches the active exact-emission filter.
/// This is the correct guard for host materialization in targeted verification
/// runs: it preserves the production compute path while avoiding host copies
/// for checkpoints that the verifier will never inspect.
pub inline fn shouldEmitEmission(point: TracePoint, layer: u16, position: u32) bool {
    if (!shouldEmit(point)) return false;
    if (!exact_filter_enabled_atomic.load(.acquire)) return true;
    return exact_filter_point_atomic.load(.acquire) == @intFromEnum(point) and
        exact_filter_layer_atomic.load(.acquire) == layer and
        exact_filter_position_atomic.load(.acquire) == position;
}

/// Emit a trace point. No-op if handler is null.
/// This is the hot path - must be as fast as possible when disabled.
pub inline fn emit(
    point: TracePoint,
    layer: u16,
    token: u32,
    position: u32,
    ptr: [*]const u8,
    dtype: DType,
    shape: [4]u32,
    ndim: u8,
    kernel_name: ?[]const u8,
) void {
    emitWithWork(point, layer, token, position, ptr, dtype, shape, ndim, kernel_name, .{});
}

/// Emit a trace point with exact runtime work counters.
pub inline fn emitWithWork(
    point: TracePoint,
    layer: u16,
    token: u32,
    position: u32,
    ptr: [*]const u8,
    dtype: DType,
    shape: [4]u32,
    ndim: u8,
    kernel_name: ?[]const u8,
    work: Work,
) void {
    const h = handler_atomic.load(.acquire) orelse return;
    var name_buf: [48]u8 = std.mem.zeroes([48]u8);
    if (kernel_name) |name| {
        const copy_len = @min(name.len, name_buf.len - 1);
        @memcpy(name_buf[0..copy_len], name[0..copy_len]);
    }
    h(.{
        .point = point,
        .layer = layer,
        .token = token,
        .position = position,
        .backend = backend_context,
        .tensor = .{
            .ptr = ptr,
            .dtype = dtype,
            .shape = shape,
            .ndim = ndim,
        },
        .timestamp_ns = std.time.nanoTimestamp(),
        .kernel_name = name_buf,
        .work_flops = work.flops,
        .work_bytes = work.bytes,
    });
}

/// Emit for logits/final outputs (no layer).
pub inline fn emitFinal(
    point: TracePoint,
    token: u32,
    position: u32,
    ptr: [*]const u8,
    dtype: DType,
    shape: [4]u32,
    ndim: u8,
    kernel_name: ?[]const u8,
) void {
    emitFinalWithWork(
        point,
        token,
        position,
        ptr,
        dtype,
        shape,
        ndim,
        kernel_name,
        .{},
    );
}

/// Emit final-stage trace point with exact runtime work counters.
pub inline fn emitFinalWithWork(
    point: TracePoint,
    token: u32,
    position: u32,
    ptr: [*]const u8,
    dtype: DType,
    shape: [4]u32,
    ndim: u8,
    kernel_name: ?[]const u8,
    work: Work,
) void {
    emitWithWork(
        point,
        TraceEmission.NO_LAYER,
        token,
        position,
        ptr,
        dtype,
        shape,
        ndim,
        kernel_name,
        work,
    );
}

// ============================================================================
// Tests
// ============================================================================

test "trace disabled is no-op" {
    // Ensure no handler by default
    try std.testing.expectEqual(@as(?Handler, null), getHandler());
    try std.testing.expect(!isEnabled());

    // Emit should be safe to call (no-op)
    emit(.lm_head, 0, 0, 0, @ptrFromInt(0x1000), .f32, .{ 1, 128, 0, 0 }, 2, null);
}

test "trace handler receives emissions" {
    const TestCapture = struct {
        var last_emission: ?TraceEmission = null;
        var call_count: usize = 0;

        fn handler(e: TraceEmission) void {
            last_emission = e;
            call_count += 1;
        }

        fn reset() void {
            last_emission = null;
            call_count = 0;
        }
    };

    TestCapture.reset();
    setHandler(&TestCapture.handler);
    defer setHandler(null);

    try std.testing.expect(isEnabled());

    emit(.attn_out, 5, 0, 10, @ptrFromInt(0x2000), .bf16, .{ 1, 8, 64, 0 }, 3, null);

    try std.testing.expectEqual(@as(usize, 1), TestCapture.call_count);
    const e = TestCapture.last_emission.?;
    try std.testing.expectEqual(TracePoint.attn_out, e.point);
    try std.testing.expectEqual(Backend.cpu, e.backend);
    try std.testing.expectEqual(@as(u16, 5), e.layer);
    try std.testing.expectEqual(@as(u32, 10), e.position);
    try std.testing.expectEqual(DType.bf16, e.tensor.dtype);
    try std.testing.expectEqual(@as(u8, 3), e.tensor.ndim);
}

test "shouldEmit honors active built-in point mask" {
    const TestCapture = struct {
        fn handler(_: TraceEmission) void {}
    };

    setHandler(&TestCapture.handler);
    defer setHandler(null);
    setActiveBuiltInPointMask((@as(u64, 1) << @intFromEnum(TracePoint.lm_head)));
    defer setActiveBuiltInPointMask(0);

    try std.testing.expect(shouldEmit(.lm_head));
    try std.testing.expect(!shouldEmit(.token_select));
    try std.testing.expect(!shouldEmit(.gdelta_out));
}

test "shouldEmitEmission honors exact checkpoint filter" {
    const TestCapture = struct {
        fn handler(_: TraceEmission) void {}
    };

    setHandler(&TestCapture.handler);
    defer setHandler(null);
    setActiveBuiltInPointMask((@as(u64, 1) << @intFromEnum(TracePoint.gdelta_out)));
    defer setActiveBuiltInPointMask(0);
    setActiveExactEmissionFilter(.{
        .point = .gdelta_out,
        .layer = 22,
        .position = 10,
    });
    defer setActiveExactEmissionFilter(null);

    try std.testing.expect(shouldEmitEmission(.gdelta_out, 22, 10));
    try std.testing.expect(!shouldEmitEmission(.gdelta_out, 22, 11));
    try std.testing.expect(!shouldEmitEmission(.gdelta_out, 21, 10));
    try std.testing.expect(!shouldEmitEmission(.lm_head, TraceEmission.NO_LAYER, 10));
}

test "TracedTensor calculations" {
    const ref = TracedTensor{
        .ptr = @ptrFromInt(0x1000),
        .dtype = .f32,
        .shape = .{ 2, 8, 64, 0 },
        .ndim = 3,
    };

    try std.testing.expectEqual(@as(usize, 2 * 8 * 64), ref.elementCount());
    try std.testing.expectEqual(@as(usize, 2 * 8 * 64 * 4), ref.byteSize());
}
