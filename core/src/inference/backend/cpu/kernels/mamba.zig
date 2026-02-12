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

const std = @import("std");
const tensor = @import("../../../../tensor.zig");
const Tensor = tensor.Tensor;
const log = @import("../../../../log.zig");
const compute = @import("../../../../compute/root.zig");
const kernel = compute.kernel;
const matmul = compute.ops.matmul;
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
    ssm_scan: kernel.SsmScanFn,
    layer_idx: u16 = trace.TraceEmission.NO_LAYER,

    /// Initialize from configuration and weights.
    pub fn init(
        config: MambaConfig,
        weights: MambaWeights,
        matmul_in_proj: matmul.MatmulFn,
        matmul_out_proj: matmul.MatmulFn,
        ssm_scan: kernel.SsmScanFn,
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

        // Determine sequence length from input tensor shape
        // Input can be (d_model,) for single token or (1, seq_len, d_model) for batched
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
        const A_log = w.A_log.asSlice(f32);
        const D_skip = w.D.asSlice(f32);

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

            // 2. Causal 1D convolution with state update
            // Conv is applied to the FULL xBC (d_inner + 2*n_groups*d_state channels)
            // Conv1d weight shape is [xBC_len, 1, d_conv] -> [1792, 1, 4]
            // For each channel in xBC (not just d_inner!)
            for (0..xBC_len) |ch| {
                // Shift state left (drop oldest, keep d_conv-1 most recent)
                const state_offset = ch * d_conv;
                for (0..d_conv - 1) |i| {
                    conv_state[state_offset + i] = conv_state[state_offset + i + 1];
                }
                // Append new input
                conv_state[state_offset + d_conv - 1] = xBC[ch];

                // Depthwise conv: sum over kernel
                var sum: f32 = 0;
                for (0..d_conv) |k| {
                    sum += conv_state[state_offset + k] * conv_weight[ch * d_conv + k];
                }

                // Add bias if present
                if (w.conv1d_bias) |bias| {
                    sum += bias.asSlice(f32)[ch];
                }

                // Store in conv_out buffer (needs to be xBC_len size now!)
                // But conv_out is only d_inner size... we need to use xBC directly
                // Actually, conv output goes back into xBC in-place for the reference impl
                // For now, let's write to the xBC buffer (it's already a slice)
                // Wait, xBC is const. We need a mutable buffer.
                // Let's use proj_out directly since we're done with the input proj part
                proj_out[d_inner + ch] = sum;
            }

            // Apply SiLU to the ENTIRE xBC conv output before splitting
            // This matches HuggingFace: xBC_conv = F.silu(conv1d(xBC))
            const xBC_conv = proj_out[d_inner .. d_inner + xBC_len];
            for (xBC_conv) |*v| {
                v.* = silu(v.*);
            }

            // Now split the conv output after silu
            // xBC layout after conv+silu: [x_conv_out (d_inner) | B (n_groups*d_state) | C (n_groups*d_state)]
            const x_conv_out = xBC_conv[0..d_inner];
            const B_raw = xBC_conv[d_inner .. d_inner + n_groups * d_state];
            const C_raw = xBC_conv[d_inner + n_groups * d_state ..][0 .. n_groups * d_state];

            // 3. Discretize dt: dt = softplus(dt_raw + dt_bias)
            for (0..n_heads) |h| {
                var dt_val = dt_raw[h];
                if (w.dt_bias) |bias| {
                    dt_val += bias.asSlice(f32)[h];
                }
                dt[h] = softplus(dt_val);
            }

            // 4. SSM scan: h_t = exp(A * dt) * h_{t-1} + B * x_t
            //              y_t = C * h_t + D * x_t
            self.ssm_scan(
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
            for (0..d_inner) |i| {
                ssm_out[i] *= silu(z[i]);
            }

            // 6. Apply optional norm AFTER gating
            if (w.norm_weight) |norm_w| {
                const norm_data = norm_w.asSlice(f32);
                rmsnorm(ssm_out, norm_data, 1e-5);
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
// Helper Functions
// =============================================================================

/// SiLU activation: x * sigmoid(x)
inline fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// Softplus activation: log(1 + exp(x))
inline fn softplus(x: f32) f32 {
    // Numerically stable version
    if (x > 20.0) return x;
    return @log(1.0 + @exp(x));
}

/// Simple RMSNorm in-place
fn rmsnorm(x: []f32, weight: []const f32, eps: f32) void {
    // Compute RMS
    var sum_sq: f32 = 0;
    for (x) |v| {
        sum_sq += v * v;
    }
    const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(x.len)) + eps);
    const inv_rms = 1.0 / rms;

    // Normalize and scale
    for (x, 0..) |*v, i| {
        v.* = v.* * inv_rms * weight[i];
    }
}

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

test "silu activation" {
    // silu(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0), silu(0), 1e-6);

    // silu(x) approaches x as x -> inf
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), silu(10.0), 0.01);

    // silu(-x) approaches 0 as x -> inf
    try std.testing.expectApproxEqAbs(@as(f32, 0), silu(-10.0), 0.01);
}

test "softplus activation" {
    // softplus(0) = ln(2)
    try std.testing.expectApproxEqAbs(@as(f32, 0.693147), softplus(0), 1e-4);

    // softplus(x) approaches x for large x
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), softplus(25.0), 0.01);

    // softplus(-x) approaches 0 for large negative x
    try std.testing.expectApproxEqAbs(@as(f32, 0), softplus(-10.0), 0.001);
}
