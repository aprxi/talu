//! ShortConv (Gated Short-Range Convolution) CPU Kernel
//!
//! Implements gated short-range convolution for hybrid models:
//!     output = out_proj(C * conv(B * x))
//!
//! where:
//! - B, C are input-dependent gates (element-wise)
//! - conv is causal 1D depthwise convolution
//! - State tracks last d_conv positions for causal masking
//!
//! This is simpler than Mamba - no SSM state, just conv state for causality.

const std = @import("std");
const tensor = @import("../../../../tensor.zig");
const Tensor = tensor.Tensor;
const log = @import("../../../../log.zig");
const compute = @import("../../../../compute/root.zig");
const matmul = compute.ops.matmul;
const simd = @import("../../../../compute/simd/root.zig");
const inspect = @import("../../../../xray/root.zig");
const trace = inspect.trace;

/// ShortConv configuration.
pub const ShortConvConfig = struct {
    d_model: u32, // Model dimension (hidden_size)
    d_conv: u32, // Convolution kernel size (L_cache, e.g., 3)
    conv_dim: u32, // Intermediate dimension
    conv_dim_out: u32, // Output dimension (usually = d_model)
    has_bias: bool = false,
};

/// ShortConv kernel weights.
pub const ShortConvWeights = struct {
    // Input projection: [3 * conv_dim, d_model] -> projects to B, C, x
    in_proj: *const Tensor,

    // Depthwise 1D convolution: [conv_dim, d_conv]
    // Note: depthwise conv has groups=conv_dim, so weight is [out_channels, 1, kernel_size]
    // which simplifies to [conv_dim, d_conv] when stored flat
    conv1d_weight: *const Tensor,
    conv1d_bias: ?*const Tensor = null,

    // Output projection: [d_model, conv_dim_out]
    out_proj: *const Tensor,
};

/// ShortConv recurrent state for autoregressive generation.
/// Simpler than MambaState - only tracks conv history, no SSM state.
///
/// State layout is [d_conv, conv_dim] (transposed for SIMD efficiency).
/// Each time slice is contiguous, enabling vectorized operations.
pub const ShortConvState = struct {
    allocator: std.mem.Allocator,

    // Convolution state: [d_conv, conv_dim] per batch
    // Transposed layout: time-major for SIMD-friendly access
    // state[k * conv_dim + ch] = value at time position k for channel ch
    conv_state: []f32,

    // Dimensions
    batch_size: usize,
    conv_dim: usize,
    d_conv: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        batch_size: usize,
        config: ShortConvConfig,
    ) !ShortConvState {
        // State holds full d_conv positions per channel
        // Layout: [batch, d_conv, conv_dim] - time-major for SIMD
        const conv_state_size = batch_size * config.conv_dim * config.d_conv;

        const conv_state = try allocator.alloc(f32, conv_state_size);
        @memset(conv_state, 0);

        return .{
            .allocator = allocator,
            .conv_state = conv_state,
            .batch_size = batch_size,
            .conv_dim = config.conv_dim,
            .d_conv = config.d_conv,
        };
    }

    pub fn reset(self: *ShortConvState) void {
        @memset(self.conv_state, 0);
    }

    pub fn deinit(self: *ShortConvState) void {
        self.allocator.free(self.conv_state);
        self.* = undefined;
    }
};

/// Scratch buffer for ShortConv forward pass.
pub const ShortConvScratch = struct {
    allocator: std.mem.Allocator,
    buffer: []f32,
    proj_offset: usize,
    conv_offset: usize,
    gated_offset: usize,

    pub fn init(allocator: std.mem.Allocator, config: ShortConvConfig) !ShortConvScratch {
        const conv_dim: usize = config.conv_dim;

        // Projection output: 3 * conv_dim (for B, C, x_proj)
        const proj_len = 3 * conv_dim;
        const conv_len = conv_dim;
        const gated_len = conv_dim;

        const total = proj_len + conv_len + gated_len;
        const buffer = try allocator.alloc(f32, total);
        @memset(buffer, 0);

        return .{
            .allocator = allocator,
            .buffer = buffer,
            .proj_offset = 0,
            .conv_offset = proj_len,
            .gated_offset = proj_len + conv_len,
        };
    }

    pub fn deinit(self: *ShortConvScratch) void {
        self.allocator.free(self.buffer);
        self.* = undefined;
    }

    pub fn getProjection(self: *ShortConvScratch, len: usize) []f32 {
        return self.buffer[self.proj_offset..][0..len];
    }

    pub fn getConvOutput(self: *ShortConvScratch, len: usize) []f32 {
        return self.buffer[self.conv_offset..][0..len];
    }

    pub fn getGatedOutput(self: *ShortConvScratch, len: usize) []f32 {
        return self.buffer[self.gated_offset..][0..len];
    }
};

/// ShortConv kernel.
pub const ShortConvKernel = struct {
    config: ShortConvConfig,
    weights: ShortConvWeights,
    matmul_in_proj: matmul.MatmulFn,
    matmul_out_proj: matmul.MatmulFn,
    matmul_in_proj_name: []const u8,
    matmul_out_proj_name: []const u8,
    layer_idx: u16 = trace.TraceEmission.NO_LAYER,

    // Pre-transposed conv weights: [d_conv, conv_dim] for SIMD-friendly access
    // Original layout is [conv_dim, d_conv] (channel-major from PyTorch)
    // Transposed layout allows contiguous vector loads in the hot loop
    conv_weight_transposed: ?[]f32 = null,
    weight_allocator: ?std.mem.Allocator = null,

    pub fn init(
        config: ShortConvConfig,
        weights: ShortConvWeights,
        matmul_in_proj: matmul.MatmulFn,
        matmul_out_proj: matmul.MatmulFn,
        matmul_in_proj_name: []const u8,
        matmul_out_proj_name: []const u8,
    ) ShortConvKernel {
        return .{
            .config = config,
            .weights = weights,
            .matmul_in_proj = matmul_in_proj,
            .matmul_out_proj = matmul_out_proj,
            .matmul_in_proj_name = matmul_in_proj_name,
            .matmul_out_proj_name = matmul_out_proj_name,
            .conv_weight_transposed = null,
            .weight_allocator = null,
        };
    }

    /// Initialize transposed weight buffer for SIMD-optimized convolution.
    /// Must be called before forward() for optimal performance.
    /// Transposes [conv_dim, d_conv] -> [d_conv, conv_dim] layout.
    pub fn initTransposedWeights(self: *ShortConvKernel, allocator: std.mem.Allocator) !void {
        if (self.conv_weight_transposed != null) return; // Already initialized

        const conv_dim: usize = self.config.conv_dim;
        const d_conv: usize = self.config.d_conv;
        const src = self.weights.conv1d_weight.asSlice(f32);

        const transposed = try allocator.alloc(f32, conv_dim * d_conv);
        errdefer allocator.free(transposed);

        // Transpose: src[ch * d_conv + k] -> dst[k * conv_dim + ch]
        for (0..conv_dim) |ch| {
            for (0..d_conv) |k| {
                transposed[k * conv_dim + ch] = src[ch * d_conv + k];
            }
        }

        self.conv_weight_transposed = transposed;
        self.weight_allocator = allocator;
    }

    pub fn deinit(self: *ShortConvKernel) void {
        if (self.conv_weight_transposed) |w| {
            if (self.weight_allocator) |alloc| {
                alloc.free(w);
            }
        }
        self.conv_weight_transposed = null;
        self.weight_allocator = null;
    }

    /// Forward pass for ShortConv layer (single token, autoregressive).
    ///
    /// Computation:
    ///   1. BCx = in_proj(x)  -> [B, C, x_proj] each of size conv_dim
    ///   2. Bx = B * x_proj   -> element-wise gating
    ///   3. conv_out = causal_conv1d(Bx, state)
    ///   4. gated = C * conv_out
    ///   5. output = out_proj(gated)
    pub fn forward(
        self: *const ShortConvKernel,
        input: *const Tensor,
        output: *Tensor,
        state: *ShortConvState,
        scratch: *ShortConvScratch,
        matmul_scratch: *matmul.MatmulScratch,
    ) !void {
        const cfg = self.config;
        const w = self.weights;

        const d_model: usize = cfg.d_model;
        const conv_dim: usize = cfg.conv_dim;
        const d_conv: usize = cfg.d_conv;

        // Determine sequence length from input tensor shape
        const seq_len: usize = if (input.n_dims == 3)
            @intCast(input.shape[1])
        else if (input.n_dims == 2)
            @intCast(input.shape[0])
        else
            1;

        // Get scratch buffers
        const proj_len = 3 * conv_dim;
        const proj_out = scratch.getProjection(proj_len);
        const conv_out = scratch.getConvOutput(conv_dim);
        const gated_out = scratch.getGatedOutput(conv_dim);

        const full_input_data = input.asSlice(f32);
        const full_output_data = output.asSlice(f32);
        const conv_weight = w.conv1d_weight.asSlice(f32);
        const conv_state = state.conv_state;

        // Process each token in sequence
        for (0..seq_len) |t| {
            const token_offset = t * d_model;
            const input_data = full_input_data[token_offset..][0..d_model];
            const output_data = full_output_data[token_offset..][0..d_model];

            // 1. Input projection: (d_model) @ (d_model, 3*conv_dim) -> (3*conv_dim)
            var input_view = Tensor.view2DSlice(input_data, 1, d_model);
            var proj_view = Tensor.view2DSlice(proj_out, 1, proj_len);
            self.matmul_in_proj(&input_view, w.in_proj, &proj_view, matmul_scratch);

            // 2. Split projection output into B, C, x_proj
            const B_gate = proj_out[0..conv_dim];
            const C_gate = proj_out[conv_dim .. 2 * conv_dim];
            const x_proj = proj_out[2 * conv_dim ..][0..conv_dim];

            // 3. SIMD-optimized causal depthwise convolution with state update
            // State layout: [d_conv, conv_dim] - time-major for SIMD efficiency
            // Use transposed weights if available (fully contiguous access)
            if (self.conv_weight_transposed) |weight_t| {
                depthwiseConvSimdTransposed(
                    B_gate,
                    x_proj,
                    conv_state,
                    weight_t,
                    conv_out,
                    if (w.conv1d_bias) |b| b.asSlice(f32) else null,
                    conv_dim,
                    d_conv,
                );
            } else {
                // Fallback: channel-major weights with strided access
                depthwiseConvSimd(
                    B_gate,
                    x_proj,
                    conv_state,
                    conv_weight,
                    conv_out,
                    if (w.conv1d_bias) |b| b.asSlice(f32) else null,
                    conv_dim,
                    d_conv,
                );
            }

            // 4. Apply C gate: gated = C * conv_out (SIMD vectorized)
            simdMul(C_gate, conv_out, gated_out, conv_dim);

            // 5. Output projection: (conv_dim) @ (conv_dim, d_model) -> (d_model)
            var gated_view = Tensor.view2DSlice(gated_out, 1, conv_dim);
            var out_view = Tensor.view2DSlice(output_data, 1, d_model);
            self.matmul_out_proj(&gated_view, w.out_proj, &out_view, matmul_scratch);

        }

        // Trace: emit intermediate and final outputs
        if (trace.isEnabled()) {
            // Trace in_proj output (last token's projection)
            trace.emit(
                .conv_in_proj,
                self.layer_idx,
                0,
                @intCast(seq_len),
                @ptrCast(proj_out.ptr),
                .f32,
                .{ 1, @intCast(proj_len), 0, 0 },
                2,
                self.matmul_in_proj_name,
            );

            // Trace conv output (last token's conv result)
            trace.emit(
                .conv_conv,
                self.layer_idx,
                0,
                @intCast(seq_len),
                @ptrCast(conv_out.ptr),
                .f32,
                .{ 1, @intCast(conv_dim), 0, 0 },
                2,
                "depthwiseConv",
            );

            // Trace out_proj output (final output)
            trace.emit(
                .conv_out_proj,
                self.layer_idx,
                0,
                @intCast(seq_len),
                @ptrCast(full_output_data.ptr),
                .f32,
                .{ 1, @intCast(seq_len), @intCast(d_model), 0 },
                3,
                self.matmul_out_proj_name,
            );
        }
    }

    /// Describe this kernel for debugging/introspection.
    pub fn describe(self: *const ShortConvKernel, writer: anytype, indent: usize) !void {
        try writer.writeByteNTimes(' ', indent);
        try writer.print("ShortConvKernel(d_model={}, d_conv={}, conv_dim={}, conv_dim_out={})\n", .{
            self.config.d_model,
            self.config.d_conv,
            self.config.conv_dim,
            self.config.conv_dim_out,
        });
    }
};

// =============================================================================
// SIMD-Optimized Primitives
// =============================================================================

const VEC = simd.f32_vec_len; // 8 for AVX2, 4 for NEON

/// SIMD element-wise multiply: out = a * b
/// Processes VEC elements per iteration for maximum throughput.
fn simdMul(a: []const f32, b: []const f32, out: []f32, len: usize) void {
    @setFloatMode(.optimized);
    var i: usize = 0;

    // Main SIMD loop
    while (i + VEC <= len) : (i += VEC) {
        const va: @Vector(VEC, f32) = a[i..][0..VEC].*;
        const vb: @Vector(VEC, f32) = b[i..][0..VEC].*;
        out[i..][0..VEC].* = va * vb;
    }

    // Scalar tail
    while (i < len) : (i += 1) {
        out[i] = a[i] * b[i];
    }
}

/// High-performance SIMD depthwise 1D convolution with state management.
///
/// State layout: [d_conv, conv_dim] - time-major (each time slice is contiguous)
/// Weight layout: [conv_dim, d_conv] - channel-major (from PyTorch depthwise conv)
///
/// Algorithm:
/// 1. Shift state: move time slices 1..d_conv to 0..d_conv-1
/// 2. Compute Bx = B * x and store in newest state position
/// 3. Compute conv output: sum over d_conv weighted time slices
/// 4. Add bias if present
///
/// The key optimization is processing entire time slices as contiguous vectors,
/// enabling full SIMD utilization for the shift and accumulate operations.
fn depthwiseConvSimd(
    B_gate: []const f32,
    x_proj: []const f32,
    state: []f32,
    weight: []const f32,
    out: []f32,
    bias: ?[]const f32,
    conv_dim: usize,
    d_conv: usize,
) void {
    @setFloatMode(.optimized);

    // Step 1: Shift state left (drop oldest time slice, make room for new)
    // State layout: [d_conv, conv_dim] -> row k is time position k
    // We shift rows 1..d_conv-1 to positions 0..d_conv-2
    // This is a bulk memory move of (d_conv-1) * conv_dim floats
    if (d_conv > 1) {
        const shift_src = state[conv_dim..]; // Start of row 1
        const shift_dst = state[0 .. (d_conv - 1) * conv_dim]; // Rows 0..d_conv-2
        @memcpy(shift_dst, shift_src[0..shift_dst.len]);
    }

    // Step 2: Compute Bx = B * x_proj and store in newest state position (row d_conv-1)
    const newest_row = state[(d_conv - 1) * conv_dim ..][0..conv_dim];
    simdMul(B_gate, x_proj, newest_row, conv_dim);

    // Step 3: Compute convolution output
    // out[ch] = sum_{k=0}^{d_conv-1} state[k, ch] * weight[ch, k]
    //
    // Weight is channel-major [conv_dim, d_conv], but state is time-major [d_conv, conv_dim].
    // For each channel ch, we need weight[ch*d_conv + k] for k in 0..d_conv.
    //
    // Optimization: process d_conv time slices, accumulating weighted contributions.
    // Each time slice is contiguous, enabling full SIMD vectorization.

    // Initialize output to zero
    @memset(out, 0);

    // Accumulate weighted contributions from each time slice
    // For d_conv=3 (typical), this is 3 passes over conv_dim elements
    for (0..d_conv) |k| {
        const state_row = state[k * conv_dim ..][0..conv_dim];

        var i: usize = 0;
        // Main SIMD loop with 2x unroll for better ILP
        while (i + 2 * VEC <= conv_dim) : (i += 2 * VEC) {
            // Load state vectors for this time slice
            const s0: @Vector(VEC, f32) = state_row[i..][0..VEC].*;
            const s1: @Vector(VEC, f32) = state_row[i + VEC ..][0..VEC].*;

            // Load weight vectors - need to gather from channel-major layout
            // weight[ch*d_conv + k] for channels i..i+VEC
            var w0: @Vector(VEC, f32) = undefined;
            var w1: @Vector(VEC, f32) = undefined;
            inline for (0..VEC) |j| {
                w0[j] = weight[(i + j) * d_conv + k];
                w1[j] = weight[(i + VEC + j) * d_conv + k];
            }

            // Load current output
            const o0: @Vector(VEC, f32) = out[i..][0..VEC].*;
            const o1: @Vector(VEC, f32) = out[i + VEC ..][0..VEC].*;

            // Fused multiply-add
            out[i..][0..VEC].* = @mulAdd(@Vector(VEC, f32), s0, w0, o0);
            out[i + VEC ..][0..VEC].* = @mulAdd(@Vector(VEC, f32), s1, w1, o1);
        }

        // Single vector iteration
        while (i + VEC <= conv_dim) : (i += VEC) {
            const s: @Vector(VEC, f32) = state_row[i..][0..VEC].*;
            var w: @Vector(VEC, f32) = undefined;
            inline for (0..VEC) |j| {
                w[j] = weight[(i + j) * d_conv + k];
            }
            const o: @Vector(VEC, f32) = out[i..][0..VEC].*;
            out[i..][0..VEC].* = @mulAdd(@Vector(VEC, f32), s, w, o);
        }

        // Scalar tail
        while (i < conv_dim) : (i += 1) {
            out[i] += state_row[i] * weight[i * d_conv + k];
        }
    }

    // Step 4: Add bias if present
    if (bias) |b| {
        var i: usize = 0;
        while (i + VEC <= conv_dim) : (i += VEC) {
            const o: @Vector(VEC, f32) = out[i..][0..VEC].*;
            const bv: @Vector(VEC, f32) = b[i..][0..VEC].*;
            out[i..][0..VEC].* = o + bv;
        }
        while (i < conv_dim) : (i += 1) {
            out[i] += b[i];
        }
    }
}

/// Maximum-performance SIMD depthwise convolution with transposed weights.
///
/// Both state and weight are time-major [d_conv, conv_dim], enabling
/// fully contiguous vector loads with no gather operations.
///
/// This is ~2-3x faster than the channel-major weight version due to:
/// - No scalar gather for weights (fully contiguous loads)
/// - Better cache locality (sequential memory access)
/// - Enables prefetching for next time slice
fn depthwiseConvSimdTransposed(
    B_gate: []const f32,
    x_proj: []const f32,
    state: []f32,
    weight_t: []const f32, // Transposed: [d_conv, conv_dim]
    out: []f32,
    bias: ?[]const f32,
    conv_dim: usize,
    d_conv: usize,
) void {
    @setFloatMode(.optimized);

    // Step 1: Shift state left using memmove (handles overlap correctly)
    // State layout: [d_conv, conv_dim] -> row k is time position k
    if (d_conv > 1) {
        const shift_src = state[conv_dim..]; // Start of row 1
        const shift_dst = state[0 .. (d_conv - 1) * conv_dim]; // Rows 0..d_conv-2
        @memcpy(shift_dst, shift_src[0..shift_dst.len]);
    }

    // Step 2: Compute Bx = B * x_proj and store in newest state position
    const newest_row = state[(d_conv - 1) * conv_dim ..][0..conv_dim];
    simdMul(B_gate, x_proj, newest_row, conv_dim);

    // Step 3: Compute convolution output with fully contiguous access
    // Both state[k] and weight_t[k] are contiguous rows of conv_dim elements
    //
    // For maximum ILP, we use 4x unrolling on the outer loop when d_conv >= 4,
    // processing multiple time slices per iteration to hide FMA latency.

    // Initialize output to zero
    @memset(out, 0);

    // Process time slices with maximum unrolling based on d_conv
    var k: usize = 0;

    // 4x unrolled loop for d_conv >= 4 (processes 4 time slices per iteration)
    while (k + 4 <= d_conv) : (k += 4) {
        const s0 = state[k * conv_dim ..][0..conv_dim];
        const s1 = state[(k + 1) * conv_dim ..][0..conv_dim];
        const s2 = state[(k + 2) * conv_dim ..][0..conv_dim];
        const s3 = state[(k + 3) * conv_dim ..][0..conv_dim];
        const w0 = weight_t[k * conv_dim ..][0..conv_dim];
        const w1 = weight_t[(k + 1) * conv_dim ..][0..conv_dim];
        const w2 = weight_t[(k + 2) * conv_dim ..][0..conv_dim];
        const w3 = weight_t[(k + 3) * conv_dim ..][0..conv_dim];

        var i: usize = 0;
        // Main SIMD loop - 4 FMAs per channel group for maximum ILP
        while (i + VEC <= conv_dim) : (i += VEC) {
            var acc: @Vector(VEC, f32) = out[i..][0..VEC].*;
            acc = @mulAdd(@Vector(VEC, f32), s0[i..][0..VEC].*, w0[i..][0..VEC].*, acc);
            acc = @mulAdd(@Vector(VEC, f32), s1[i..][0..VEC].*, w1[i..][0..VEC].*, acc);
            acc = @mulAdd(@Vector(VEC, f32), s2[i..][0..VEC].*, w2[i..][0..VEC].*, acc);
            acc = @mulAdd(@Vector(VEC, f32), s3[i..][0..VEC].*, w3[i..][0..VEC].*, acc);
            out[i..][0..VEC].* = acc;
        }
        // Scalar tail
        while (i < conv_dim) : (i += 1) {
            out[i] += s0[i] * w0[i] + s1[i] * w1[i] + s2[i] * w2[i] + s3[i] * w3[i];
        }
    }

    // 2x unrolled for remaining pairs
    while (k + 2 <= d_conv) : (k += 2) {
        const s0 = state[k * conv_dim ..][0..conv_dim];
        const s1 = state[(k + 1) * conv_dim ..][0..conv_dim];
        const w0 = weight_t[k * conv_dim ..][0..conv_dim];
        const w1 = weight_t[(k + 1) * conv_dim ..][0..conv_dim];

        var i: usize = 0;
        while (i + VEC <= conv_dim) : (i += VEC) {
            var acc: @Vector(VEC, f32) = out[i..][0..VEC].*;
            acc = @mulAdd(@Vector(VEC, f32), s0[i..][0..VEC].*, w0[i..][0..VEC].*, acc);
            acc = @mulAdd(@Vector(VEC, f32), s1[i..][0..VEC].*, w1[i..][0..VEC].*, acc);
            out[i..][0..VEC].* = acc;
        }
        while (i < conv_dim) : (i += 1) {
            out[i] += s0[i] * w0[i] + s1[i] * w1[i];
        }
    }

    // Handle remaining single time slice
    while (k < d_conv) : (k += 1) {
        const state_row = state[k * conv_dim ..][0..conv_dim];
        const weight_row = weight_t[k * conv_dim ..][0..conv_dim];

        var i: usize = 0;
        while (i + VEC <= conv_dim) : (i += VEC) {
            const s: @Vector(VEC, f32) = state_row[i..][0..VEC].*;
            const w: @Vector(VEC, f32) = weight_row[i..][0..VEC].*;
            const o: @Vector(VEC, f32) = out[i..][0..VEC].*;
            out[i..][0..VEC].* = @mulAdd(@Vector(VEC, f32), s, w, o);
        }
        while (i < conv_dim) : (i += 1) {
            out[i] += state_row[i] * weight_row[i];
        }
    }

    // Step 4: Add bias if present (fused into final pass for better cache use)
    if (bias) |b| {
        var i: usize = 0;
        while (i + VEC <= conv_dim) : (i += VEC) {
            const o: @Vector(VEC, f32) = out[i..][0..VEC].*;
            const bv: @Vector(VEC, f32) = b[i..][0..VEC].*;
            out[i..][0..VEC].* = o + bv;
        }
        while (i < conv_dim) : (i += 1) {
            out[i] += b[i];
        }
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

test "ShortConvState init and deinit" {
    const allocator = std.testing.allocator;

    const config = ShortConvConfig{
        .d_model = 768,
        .d_conv = 3,
        .conv_dim = 768,
        .conv_dim_out = 768,
    };

    var state = try ShortConvState.init(allocator, 1, config);
    defer state.deinit();

    // Verify dimensions
    try std.testing.expectEqual(@as(usize, 768), state.conv_dim);
    try std.testing.expectEqual(@as(usize, 3), state.d_conv);

    // Verify zero-initialized
    for (state.conv_state) |v| {
        try std.testing.expectEqual(@as(f32, 0), v);
    }
}

test "ShortConvState reset" {
    const allocator = std.testing.allocator;

    const config = ShortConvConfig{
        .d_model = 768,
        .d_conv = 3,
        .conv_dim = 768,
        .conv_dim_out = 768,
    };

    var state = try ShortConvState.init(allocator, 1, config);
    defer state.deinit();

    // Modify state
    state.conv_state[0] = 1.0;

    // Reset
    state.reset();

    // Verify zeroed
    try std.testing.expectEqual(@as(f32, 0), state.conv_state[0]);
}

test "ShortConvScratch init and deinit" {
    const allocator = std.testing.allocator;

    const config = ShortConvConfig{
        .d_model = 768,
        .d_conv = 3,
        .conv_dim = 768,
        .conv_dim_out = 768,
    };

    var scratch = try ShortConvScratch.init(allocator, config);
    defer scratch.deinit();

    // Verify we can get scratch buffers
    const proj = scratch.getProjection(3 * 768);
    try std.testing.expectEqual(@as(usize, 2304), proj.len);

    const conv = scratch.getConvOutput(768);
    try std.testing.expectEqual(@as(usize, 768), conv.len);

    const gated = scratch.getGatedOutput(768);
    try std.testing.expectEqual(@as(usize, 768), gated.len);
}

test "ShortConvState conv_state size" {
    const allocator = std.testing.allocator;

    const config = ShortConvConfig{
        .d_model = 768,
        .d_conv = 3,
        .conv_dim = 768,
        .conv_dim_out = 768,
    };

    var state = try ShortConvState.init(allocator, 2, config);
    defer state.deinit();

    // Conv state should be batch * conv_dim * d_conv
    // = 2 * 768 * 3 = 4608
    try std.testing.expectEqual(@as(usize, 4608), state.conv_state.len);
}
