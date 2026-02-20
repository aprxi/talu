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

pub const supported = true;

const std = @import("std");
const tensor = @import("../../../../tensor.zig");
const Tensor = tensor.Tensor;
const log = @import("../../../../log.zig");
const compute = @import("../../../../compute/root.zig");
const cpu_linalg = compute.cpu.linalg;
const cpu_conv1d = compute.cpu.conv1d_depthwise;
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
    /// Canonical kernel-call contract for backend parity checks.
    pub const ForwardParams = struct {
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        state: *ShortConvState,
        scratch: *ShortConvScratch,
        matmul_scratch: *cpu_linalg.MatmulScratch,
    };

    config: ShortConvConfig,
    weights: ShortConvWeights,
    matmul_in_proj: cpu_linalg.MatmulFn,
    matmul_out_proj: cpu_linalg.MatmulFn,
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
        matmul_in_proj: cpu_linalg.MatmulFn,
        matmul_out_proj: cpu_linalg.MatmulFn,
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

        try cpu_conv1d.transposeChannelMajorToTimeMajor(src, transposed, conv_dim, d_conv);

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
        matmul_scratch: *cpu_linalg.MatmulScratch,
    ) !void {
        const cfg = self.config;
        const w = self.weights;

        const d_model: usize = cfg.d_model;
        const conv_dim: usize = cfg.conv_dim;
        const d_conv: usize = cfg.d_conv;

        // Determine sequence length from input tensor shape.
        // Supported inputs are (d_model,), (seq_len, d_model), or (1, seq_len, d_model).
        const batch_size: usize = if (input.n_dims == 3) @intCast(input.shape[0]) else 1;
        if (batch_size != 1) {
            return error.InvalidShape;
        }
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
                cpu_conv1d.runTimeMajor(
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
                cpu_conv1d.runChannelMajor(
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
            cpu_conv1d.simdMul(C_gate, conv_out, gated_out, conv_dim);

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

test "ShortConvKernel.forward rejects batch > 1 for 3D input" {
    const allocator = std.testing.allocator;

    const config = ShortConvConfig{
        .d_model = 4,
        .d_conv = 2,
        .conv_dim = 4,
        .conv_dim_out = 4,
    };

    var dummy_weight_owned = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, 1 });
    defer dummy_weight_owned.deinit();
    var dummy_weight = dummy_weight_owned.view();

    const weights = ShortConvWeights{
        .in_proj = &dummy_weight,
        .conv1d_weight = &dummy_weight,
        .out_proj = &dummy_weight,
    };

    const kernel_inst = ShortConvKernel.init(
        config,
        weights,
        cpu_linalg.matmulF32,
        cpu_linalg.matmulF32,
        "matmulF32",
        "matmulF32",
    );

    var input_owned = try tensor.OwnedTensor.init(allocator, .f32, &.{ 2, 3, 4 });
    defer input_owned.deinit();
    var output_owned = try tensor.OwnedTensor.init(allocator, .f32, &.{ 2, 3, 4 });
    defer output_owned.deinit();
    var input = input_owned.view();
    var output = output_owned.view();

    var state = try ShortConvState.init(allocator, 1, config);
    defer state.deinit();
    var scratch = try ShortConvScratch.init(allocator, config);
    defer scratch.deinit();
    var matmul_scratch = try cpu_linalg.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    const result = kernel_inst.forward(&input, &output, &state, &scratch, &matmul_scratch);
    try std.testing.expectError(error.InvalidShape, result);
}
