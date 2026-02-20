//! Mamba2 (SSD) CPU Kernel
//!
//! Implementation of Mamba2 State Space Duality layers.
//! Used in heterogeneous models like Granite 4.0 Hybrid.
//!
//! Mamba2 uses a simplified SSM formulation (SSD) with:
//! - Scalar-times-identity structure for A (diagonal)
//! - Head-based parallelism (similar to attention heads)
//! - Efficient hardware-friendly computation
//!
//! Reference: "Transformers are SSMs" (Mamba-2 paper)
//!
//! Architecture:
//!   1. Input projection: x -> (z, xBC, dt) where z is gate, xBC feeds conv/SSM
//!   2. Causal 1D convolution with state
//!   3. SSM scan: h_t = A * h_{t-1} + B * x_t, y_t = C * h_t + D * x_t
//!   4. Gating: y = y * silu(z)
//!   5. Output projection

pub const supported = true;

const std = @import("std");
const tensor = @import("../../../../tensor.zig");
const Tensor = tensor.Tensor;
const log = @import("../../../../log.zig");
const compute = @import("../../../../compute/root.zig");
const matmul = compute.cpu.linalg.matmul;
const cpu_conv1d = compute.cpu.conv1d_depthwise;
const cpu_norm = compute.cpu.normalization;
const cpu_state_space = compute.cpu.recurrence.state_space;
const ssm_scan_mod = compute.cpu.simd.ssm_scan;
const inspect = @import("../../../../xray/root.zig");
const trace = inspect.trace;

/// Mamba2 SSM configuration.
/// These parameters define the state space model structure.
pub const MambaConfig = struct {
    d_model: u32, // Model dimension (e.g., 768)
    d_state: u32, // SSM state dimension (e.g., 128)
    d_conv: u32, // Convolution kernel size (e.g., 4)
    n_heads: u32, // Number of SSM heads (e.g., 48)
    d_head: u32, // Head dimension (e.g., 32)
    n_groups: u32 = 1, // Groups for B/C projection
};

/// Mamba2 kernel weights.
/// All weights needed for a single Mamba layer.
pub const MambaWeights = struct {
    // Input projection: projects input to (2 * d_inner + 2 * n_groups * d_state + n_heads)
    in_proj: *const Tensor,

    // 1D convolution weights for temporal mixing
    conv1d_weight: *const Tensor,
    conv1d_bias: ?*const Tensor = null,

    // SSM parameters
    A_log: *const Tensor, // Log of A diagonal (learned)
    D: *const Tensor, // Skip connection parameter
    dt_bias: ?*const Tensor = null, // Delta time bias

    // Normalization before output
    norm_weight: ?*const Tensor = null,

    // Output projection
    out_proj: *const Tensor,
};

/// Mamba2 recurrent state for a single layer.
/// This state persists across tokens during autoregressive generation.
pub const MambaState = struct {
    allocator: std.mem.Allocator,

    // Convolution state: (batch, d_inner, d_conv)
    conv_state: []f32,

    // SSM state: (batch, n_heads, d_head, d_state)
    ssm_state: []f32,

    // Dimensions for state management
    batch_size: usize,
    d_inner: usize,
    d_conv: usize,
    n_heads: usize,
    d_head: usize,
    d_state: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        batch_size: usize,
        config: MambaConfig,
    ) !MambaState {
        const d_inner = @as(usize, config.n_heads) * @as(usize, config.d_head);
        // Conv is applied to xBC which is d_inner + 2*n_groups*d_state
        const xBC_len = d_inner + 2 * @as(usize, config.n_groups) * @as(usize, config.d_state);

        const conv_state_size = batch_size * xBC_len * config.d_conv;
        const ssm_state_size = batch_size * config.n_heads * config.d_head * config.d_state;

        const conv_state = try allocator.alloc(f32, conv_state_size);
        errdefer allocator.free(conv_state);
        @memset(conv_state, 0);

        const ssm_state = try allocator.alloc(f32, ssm_state_size);
        @memset(ssm_state, 0);

        return .{
            .allocator = allocator,
            .conv_state = conv_state,
            .ssm_state = ssm_state,
            .batch_size = batch_size,
            .d_inner = d_inner,
            .d_conv = config.d_conv,
            .n_heads = config.n_heads,
            .d_head = config.d_head,
            .d_state = config.d_state,
        };
    }

    pub fn reset(self: *MambaState) void {
        @memset(self.conv_state, 0);
        @memset(self.ssm_state, 0);
    }

    pub fn deinit(self: *MambaState) void {
        self.allocator.free(self.conv_state);
        self.allocator.free(self.ssm_state);
        self.* = undefined;
    }
};

/// Mamba2 SSM kernel.
/// Provides the forward pass for Mamba layers in heterogeneous models.
pub const MambaKernel = struct {
    config: MambaConfig,
    weights: MambaWeights,
    matmul_in_proj: matmul.MatmulFn,
    matmul_out_proj: matmul.MatmulFn,
    ssm_scan: ssm_scan_mod.SsmScanFn,
    layer_idx: u16 = trace.TraceEmission.NO_LAYER,

    /// Initialize from configuration and weights.
    pub fn init(
        config: MambaConfig,
        weights: MambaWeights,
        matmul_in_proj: matmul.MatmulFn,
        matmul_out_proj: matmul.MatmulFn,
        ssm_scan: ssm_scan_mod.SsmScanFn,
    ) MambaKernel {
        return .{
            .config = config,
            .weights = weights,
            .matmul_in_proj = matmul_in_proj,
            .matmul_out_proj = matmul_out_proj,
            .ssm_scan = ssm_scan,
        };
    }

    /// Forward pass for Mamba2 layer (single token, autoregressive).
    ///
    /// This implements the Mamba2 SSD algorithm for single-token inference.
    /// The recurrent state is updated in-place for efficient autoregressive generation.
    ///
    /// Args:
    ///   input: Input tensor (d_model,) - single token embedding
    ///   output: Output tensor (d_model,) - output embedding
    ///   state: Mamba recurrent state (updated in-place)
    ///   scratch: Scratch buffer for intermediate computations
    pub fn forward(
        self: *const MambaKernel,
        input: *const Tensor,
        output: *Tensor,
        state: *MambaState,
        scratch: *MambaScratch,
        matmul_scratch: *matmul.MatmulScratch,
    ) !void {
        const cfg = self.config;
        const w = self.weights;

        const d_model: usize = cfg.d_model;
        const d_inner: usize = @as(usize, cfg.n_heads) * @as(usize, cfg.d_head);
        const d_state: usize = cfg.d_state;
        const d_conv: usize = cfg.d_conv;
        const n_heads: usize = cfg.n_heads;
        const d_head: usize = cfg.d_head;
        const n_groups: usize = cfg.n_groups;

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

        // Input projection output size: 2*d_inner + 2*n_groups*d_state + n_heads
        // Splits into: z (d_inner), xBC (d_inner + 2*n_groups*d_state), dt (n_heads)
        const xBC_len = d_inner + 2 * n_groups * d_state;
        const proj_len = 2 * d_inner + 2 * n_groups * d_state + n_heads;

        // Get scratch buffers (reused for each token)
        const proj_out = scratch.getProjection(proj_len);
        _ = scratch.getConvOutput(d_inner); // allocated but conv writes to proj_out directly
        const ssm_out = scratch.getSsmOutput(d_inner);
        const dt = scratch.getDt(n_heads);

        // Get base pointers for input/output data
        const full_input_data = input.asSlice(f32);
        const full_output_data = output.asSlice(f32);

        // Weight data (constant across tokens)
        const conv_weight = w.conv1d_weight.asSlice(f32);
        const conv_bias = if (w.conv1d_bias) |bias| bias.asSlice(f32) else null;
        const A_log = w.A_log.asSlice(f32);
        const D_skip = w.D.asSlice(f32);
        const dt_bias = if (w.dt_bias) |bias| bias.asSlice(f32) else null;

        // State arrays (updated across tokens)
        const conv_state = state.conv_state;
        const ssm_state = state.ssm_state;

        // Process each token in sequence (Mamba is inherently sequential due to recurrence)
        for (0..seq_len) |t| {
            // Get input/output slice for this token: offset by t * d_model
            const token_offset = t * d_model;
            const input_data = full_input_data[token_offset..][0..d_model];
            const output_data = full_output_data[token_offset..][0..d_model];

            // 1. Input projection: (d_model) @ (d_model, proj_len) -> (proj_len)
            var input_view = Tensor.view2DSlice(input_data, 1, d_model);
            var proj_view = Tensor.view2DSlice(proj_out, 1, proj_len);
            self.matmul_in_proj(&input_view, w.in_proj, &proj_view, matmul_scratch);

            // Split projection output: [z | xBC | dt]
            const z = proj_out[0..d_inner];
            const xBC = proj_out[d_inner .. d_inner + xBC_len];
            const dt_raw = proj_out[d_inner + xBC_len ..][0..n_heads];

            // 2. Causal depthwise convolution with state update over xBC in-place.
            try cpu_conv1d.stepDepthwiseState(xBC, conv_state, conv_weight, conv_bias, xBC_len, d_conv);

            // Apply SiLU to the ENTIRE xBC conv output before splitting
            // This matches HuggingFace: xBC_conv = F.silu(conv1d(xBC))
            const xBC_conv = proj_out[d_inner .. d_inner + xBC_len];
            cpu_state_space.siluInPlace(xBC_conv);

            // Now split the conv output after silu
            // xBC layout after conv+silu: [x_conv_out (d_inner) | B (n_groups*d_state) | C (n_groups*d_state)]
            const x_conv_out = xBC_conv[0..d_inner];
            const B_raw = xBC_conv[d_inner .. d_inner + n_groups * d_state];
            const C_raw = xBC_conv[d_inner + n_groups * d_state ..][0 .. n_groups * d_state];

            // 3. Discretize dt: dt = softplus(dt_raw + dt_bias)
            try cpu_state_space.discretizeDtSoftplus(dt, dt_raw, dt_bias);

            // 4. SSM scan: h_t = exp(A * dt) * h_{t-1} + B * x_t
            //              y_t = C * h_t + D * x_t
            cpu_state_space.scanStep(
                self.ssm_scan,
                ssm_state,
                ssm_out,
                x_conv_out,
                B_raw,
                C_raw,
                A_log,
                D_skip,
                dt,
                d_head,
                d_state,
                n_heads,
                n_groups,
            );

            // 5. Gating: y = y * silu(z)
            // NOTE: Gating happens BEFORE norm per HuggingFace reference
            try cpu_state_space.applySiluGateInPlace(ssm_out, z);

            // 6. Apply optional norm AFTER gating
            if (w.norm_weight) |norm_w| {
                const norm_data = norm_w.asSlice(f32);
                cpu_norm.rmsnormInPlace(ssm_out, norm_data, 1e-5);
            }

            // 7. Output projection: (d_inner) @ (d_inner, d_model) -> (d_model)
            var ssm_view = Tensor.view2DSlice(ssm_out, 1, d_inner);
            var out_view = Tensor.view2DSlice(output_data, 1, d_model);
            self.matmul_out_proj(&ssm_view, w.out_proj, &out_view, matmul_scratch);
        }

        // Trace: emit final output
        if (trace.isEnabled()) {
            trace.emit(
                .mamba_out,
                self.layer_idx,
                0,
                @intCast(seq_len),
                @ptrCast(full_output_data.ptr),
                .f32,
                .{ 1, @intCast(seq_len), @intCast(d_model), 0 },
                3,
                null,
            );
        }
    }

    /// Describe this kernel for debugging/introspection.
    pub fn describe(self: *const MambaKernel, writer: anytype, indent: usize) !void {
        try writer.writeByteNTimes(' ', indent);
        try writer.print("MambaKernel(d_model={}, d_state={}, d_conv={}, n_heads={}, d_head={})\n", .{
            self.config.d_model,
            self.config.d_state,
            self.config.d_conv,
            self.config.n_heads,
            self.config.d_head,
        });
    }
};

/// Scratch buffer for Mamba forward pass.
/// Pre-allocated to avoid allocations in the hot path.
pub const MambaScratch = struct {
    allocator: std.mem.Allocator,
    buffer: []f32,
    proj_offset: usize,
    conv_offset: usize,
    ssm_offset: usize,
    dt_offset: usize,

    pub fn init(allocator: std.mem.Allocator, config: MambaConfig) !MambaScratch {
        const d_inner: usize = @as(usize, config.n_heads) * @as(usize, config.d_head);
        const n_groups: usize = config.n_groups;
        const d_state: usize = config.d_state;
        const n_heads: usize = config.n_heads;

        // Projection output: 2*d_inner + 2*n_groups*d_state + n_heads
        const proj_len = 2 * d_inner + 2 * n_groups * d_state + n_heads;
        const conv_len = d_inner;
        const ssm_len = d_inner;
        const dt_len = n_heads;

        const total = proj_len + conv_len + ssm_len + dt_len;
        const buffer = try allocator.alloc(f32, total);
        @memset(buffer, 0);

        return .{
            .allocator = allocator,
            .buffer = buffer,
            .proj_offset = 0,
            .conv_offset = proj_len,
            .ssm_offset = proj_len + conv_len,
            .dt_offset = proj_len + conv_len + ssm_len,
        };
    }

    pub fn deinit(self: *MambaScratch) void {
        self.allocator.free(self.buffer);
        self.* = undefined;
    }

    pub fn getProjection(self: *MambaScratch, len: usize) []f32 {
        return self.buffer[self.proj_offset..][0..len];
    }

    pub fn getConvOutput(self: *MambaScratch, len: usize) []f32 {
        return self.buffer[self.conv_offset..][0..len];
    }

    pub fn getSsmOutput(self: *MambaScratch, len: usize) []f32 {
        return self.buffer[self.ssm_offset..][0..len];
    }

    pub fn getDt(self: *MambaScratch, len: usize) []f32 {
        return self.buffer[self.dt_offset..][0..len];
    }
};

// =============================================================================
// Unit Tests
// =============================================================================

test "MambaState init and deinit" {
    const allocator = std.testing.allocator;

    const config = MambaConfig{
        .d_model = 768,
        .d_state = 128,
        .d_conv = 4,
        .n_heads = 48,
        .d_head = 32,
    };

    var state = try MambaState.init(allocator, 1, config);
    defer state.deinit();

    // Verify state dimensions
    const d_inner = @as(usize, config.n_heads) * @as(usize, config.d_head);
    try std.testing.expectEqual(d_inner, state.d_inner);
    try std.testing.expectEqual(@as(usize, config.d_conv), state.d_conv);

    // Verify state is zero-initialized
    for (state.conv_state) |v| {
        try std.testing.expectEqual(@as(f32, 0), v);
    }
    for (state.ssm_state) |v| {
        try std.testing.expectEqual(@as(f32, 0), v);
    }
}

test "MambaState reset" {
    const allocator = std.testing.allocator;

    const config = MambaConfig{
        .d_model = 768,
        .d_state = 128,
        .d_conv = 4,
        .n_heads = 48,
        .d_head = 32,
    };

    var state = try MambaState.init(allocator, 1, config);
    defer state.deinit();

    // Modify state
    state.conv_state[0] = 1.0;
    state.ssm_state[0] = 2.0;

    // Reset
    state.reset();

    // Verify state is zero again
    try std.testing.expectEqual(@as(f32, 0), state.conv_state[0]);
    try std.testing.expectEqual(@as(f32, 0), state.ssm_state[0]);
}

test "MambaScratch init and deinit" {
    const allocator = std.testing.allocator;

    const config = MambaConfig{
        .d_model = 768,
        .d_state = 128,
        .d_conv = 4,
        .n_heads = 48,
        .d_head = 32,
    };

    var scratch = try MambaScratch.init(allocator, config);
    defer scratch.deinit();

    // Verify we can get scratch buffers
    const d_inner: usize = @as(usize, config.n_heads) * @as(usize, config.d_head);
    const proj_len = 2 * d_inner + 2 * config.n_groups * config.d_state + config.n_heads;

    const proj = scratch.getProjection(proj_len);
    try std.testing.expectEqual(proj_len, proj.len);

    const conv = scratch.getConvOutput(d_inner);
    try std.testing.expectEqual(d_inner, conv.len);

    const ssm = scratch.getSsmOutput(d_inner);
    try std.testing.expectEqual(d_inner, ssm.len);

    const dt = scratch.getDt(config.n_heads);
    try std.testing.expectEqual(@as(usize, config.n_heads), dt.len);
}

test "MambaKernel.forward rejects batch > 1 for 3D input" {
    const allocator = std.testing.allocator;

    const config = MambaConfig{
        .d_model = 4,
        .d_state = 2,
        .d_conv = 2,
        .n_heads = 1,
        .d_head = 4,
    };

    var dummy_weight_owned = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, 1 });
    defer dummy_weight_owned.deinit();
    var dummy_weight = dummy_weight_owned.view();

    const weights = MambaWeights{
        .in_proj = &dummy_weight,
        .conv1d_weight = &dummy_weight,
        .A_log = &dummy_weight,
        .D = &dummy_weight,
        .out_proj = &dummy_weight,
    };

    const kernel_inst = MambaKernel.init(
        config,
        weights,
        matmul.matmulF32,
        matmul.matmulF32,
        compute.cpu.simd.ssm_scan.ssmScanF32,
    );

    var input_owned = try tensor.OwnedTensor.init(allocator, .f32, &.{ 2, 3, 4 });
    defer input_owned.deinit();
    var output_owned = try tensor.OwnedTensor.init(allocator, .f32, &.{ 2, 3, 4 });
    defer output_owned.deinit();
    var input = input_owned.view();
    var output = output_owned.view();

    var state = try MambaState.init(allocator, 1, config);
    defer state.deinit();
    var scratch = try MambaScratch.init(allocator, config);
    defer scratch.deinit();
    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    const result = kernel_inst.forward(&input, &output, &state, &scratch, &matmul_scratch);
    try std.testing.expectError(error.InvalidShape, result);
}
