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
    /// The tensor data
    tensor: TracedTensor,
    /// Timestamp for ordering
    timestamp_ns: i128,
    /// Kernel name that produced this tensor (null-terminated, max 48 chars)
    kernel_name: [48]u8,

    pub const NO_LAYER: u16 = 0xFFFF;
};

/// Handler function type - inspection system implements this.
pub const Handler = *const fn (TraceEmission) void;

/// The global handler - set by inspection system, null when disabled.
/// Using atomic to be safe across threads (inspection might be enabled/disabled).
var handler_atomic: std.atomic.Value(?Handler) = std.atomic.Value(?Handler).init(null);

/// Set the trace handler. Called by inspection system on setup.
pub fn setHandler(h: ?Handler) void {
    handler_atomic.store(h, .release);
}

/// Get current handler (for inspection system to check if enabled).
pub fn getHandler() ?Handler {
    return handler_atomic.load(.acquire);
}

/// Check if tracing is enabled (for conditional expensive operations).
pub inline fn isEnabled() bool {
    return handler_atomic.load(.acquire) != null;
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
        .tensor = .{
            .ptr = ptr,
            .dtype = dtype,
            .shape = shape,
            .ndim = ndim,
        },
        .timestamp_ns = std.time.nanoTimestamp(),
        .kernel_name = name_buf,
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
    emit(point, TraceEmission.NO_LAYER, token, position, ptr, dtype, shape, ndim, kernel_name);
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
    try std.testing.expectEqual(@as(u16, 5), e.layer);
    try std.testing.expectEqual(@as(u32, 10), e.position);
    try std.testing.expectEqual(DType.bf16, e.tensor.dtype);
    try std.testing.expectEqual(@as(u8, 3), e.tensor.ndim);
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

