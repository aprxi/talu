//! Gated DeltaNet CPU Kernel
//!
//! Explicit CPU kernel for gated linear attention. This is distinct from
//! Mamba and intentionally does not share Mamba execution switches.

pub const supported = true;

const std = @import("std");
const tensor = @import("../../../../tensor.zig");
const Tensor = tensor.Tensor;
const log = @import("../../../../log.zig");
const compute = @import("../../../../compute/root.zig");
const cpu_linalg = compute.cpu.linalg;
const cpu_conv1d = compute.cpu.conv1d_depthwise;
const cpu_gated_delta = compute.cpu.gated_delta;
const trace = @import("../../../../xray/root.zig").trace;

pub const GatedDeltaConfig = struct {
    d_model: u32,
    d_conv: u32,
    n_heads: u32,
    d_head: u32,
};

pub const GatedDeltaWeights = struct {
    in_proj: *const Tensor,
    conv1d_weight: *const Tensor,
    conv1d_bias: ?*const Tensor = null,
    A_log: *const Tensor,
    dt_bias: ?*const Tensor = null,
    norm_weight: ?*const Tensor = null,
    out_proj: *const Tensor,
};

pub const GatedDeltaState = struct {
    allocator: std.mem.Allocator,
    conv_state: []f32,
    ssm_state: []f32,
    batch_size: usize,
    d_inner: usize,
    d_conv: usize,
    n_heads: usize,
    d_head: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        batch_size: usize,
        config: GatedDeltaConfig,
    ) !GatedDeltaState {
        const d_inner = @as(usize, config.n_heads) * @as(usize, config.d_head);
        // Allocate for the maximal historical qkv packing (3*d_inner). Some
        // architectures (e.g. Qwen3.5 4B/9B) use asymmetric q/k vs v widths
        // and consume a smaller runtime qkv_len.
        const qkv_len = d_inner * 3;
        // Time-major state layout: [batch, d_conv, qkv_len].
        // This matches the SIMD-friendly conv1d_depthwise.runTimeMajor path.
        const conv_state_size = batch_size * qkv_len * config.d_conv;
        const ssm_state_size = batch_size * config.n_heads * config.d_head * config.d_head;

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
        };
    }

    pub fn reset(self: *GatedDeltaState) void {
        @memset(self.conv_state, 0);
        @memset(self.ssm_state, 0);
    }

    pub fn deinit(self: *GatedDeltaState) void {
        self.allocator.free(self.conv_state);
        self.allocator.free(self.ssm_state);
        self.* = undefined;
    }
};

pub const GatedDeltaScratch = struct {
    allocator: std.mem.Allocator,
    buffer: []f32,
    proj_offset: usize,
    conv_offset: usize,
    ssm_offset: usize,

    pub fn init(allocator: std.mem.Allocator, config: GatedDeltaConfig) !GatedDeltaScratch {
        const d_inner: usize = @as(usize, config.n_heads) * @as(usize, config.d_head);
        const proj_len = 4 * d_inner + 2 * @as(usize, config.n_heads);
        const conv_len = d_inner;
        const ssm_len = d_inner;
        const total = proj_len + conv_len + ssm_len;
        const buffer = try allocator.alloc(f32, total);
        @memset(buffer, 0);
        return .{
            .allocator = allocator,
            .buffer = buffer,
            .proj_offset = 0,
            .conv_offset = proj_len,
            .ssm_offset = proj_len + conv_len,
        };
    }

    pub fn deinit(self: *GatedDeltaScratch) void {
        self.allocator.free(self.buffer);
        self.* = undefined;
    }

    pub fn getProjection(self: *GatedDeltaScratch, len: usize) []f32 {
        return self.buffer[self.proj_offset..][0..len];
    }

    pub fn getConvOutput(self: *GatedDeltaScratch, len: usize) []f32 {
        return self.buffer[self.conv_offset..][0..len];
    }

    pub fn getSsmOutput(self: *GatedDeltaScratch, len: usize) []f32 {
        return self.buffer[self.ssm_offset..][0..len];
    }
};

pub const GatedDeltaKernel = struct {
    pub const ForwardParams = struct {
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        state: *GatedDeltaState,
        scratch: *GatedDeltaScratch,
        matmul_scratch: *cpu_linalg.MatmulScratch,
    };

    config: GatedDeltaConfig,
    weights: GatedDeltaWeights,
    matmul_in_proj: cpu_linalg.MatmulFn,
    matmul_out_proj: cpu_linalg.MatmulFn,
    layer_idx: u16 = trace.TraceEmission.NO_LAYER,
    conv_weight_transposed: ?[]f32 = null,
    weight_allocator: ?std.mem.Allocator = null,

    pub fn init(
        config: GatedDeltaConfig,
        weights: GatedDeltaWeights,
        matmul_in_proj: cpu_linalg.MatmulFn,
        matmul_out_proj: cpu_linalg.MatmulFn,
    ) GatedDeltaKernel {
        return .{
            .config = config,
            .weights = weights,
            .matmul_in_proj = matmul_in_proj,
            .matmul_out_proj = matmul_out_proj,
            .conv_weight_transposed = null,
            .weight_allocator = null,
        };
    }

    pub fn initTransposedWeights(self: *GatedDeltaKernel, allocator: std.mem.Allocator) !void {
        if (self.conv_weight_transposed != null) return;

        const d_conv: usize = self.config.d_conv;
        const conv_dim = try convChannelDim(self.weights.conv1d_weight, d_conv);
        const src = self.weights.conv1d_weight.asSlice(f32);

        const transposed = try allocator.alloc(f32, conv_dim * d_conv);
        errdefer allocator.free(transposed);

        try cpu_conv1d.transposeChannelMajorToTimeMajor(src, transposed, conv_dim, d_conv);
        self.conv_weight_transposed = transposed;
        self.weight_allocator = allocator;
    }

    pub fn deinit(self: *GatedDeltaKernel) void {
        if (self.conv_weight_transposed) |weight_t| {
            if (self.weight_allocator) |alloc| alloc.free(weight_t);
        }
        self.conv_weight_transposed = null;
        self.weight_allocator = null;
    }

    fn projectionOutputDim(weight: *const Tensor, d_model: usize) !usize {
        if (weight.n_dims != 2) return error.InvalidShape;
        const dim0: usize = @intCast(weight.shape[0]);
        const dim1: usize = @intCast(weight.shape[1]);
        if (dim0 == 0 or dim1 == 0) return error.InvalidShape;
        if (dim0 == d_model and dim1 != d_model) return dim1;
        if (dim1 == d_model and dim0 != d_model) return dim0;
        if (dim0 == d_model and dim1 == d_model) return d_model;
        return dim0;
    }

    fn convChannelDim(weight: *const Tensor, d_conv: usize) !usize {
        if (d_conv == 0) return error.InvalidShape;
        const values = weight.asSlice(f32);
        if (values.len == 0 or (values.len % d_conv) != 0) return error.InvalidShape;
        return values.len / d_conv;
    }

    pub fn forward(
        self: *const GatedDeltaKernel,
        input: *const Tensor,
        output: *Tensor,
        state: *GatedDeltaState,
        scratch: *GatedDeltaScratch,
        matmul_scratch: *cpu_linalg.MatmulScratch,
    ) !void {
        const cfg = self.config;
        const w = self.weights;

        const d_model: usize = cfg.d_model;
        const d_inner: usize = @as(usize, cfg.n_heads) * @as(usize, cfg.d_head);
        const d_conv: usize = cfg.d_conv;
        const n_v_heads: usize = cfg.n_heads;
        const d_head: usize = cfg.d_head;

        const batch_size: usize = if (input.n_dims == 3) @intCast(input.shape[0]) else 1;
        if (batch_size != 1) return error.InvalidShape;
        const seq_len: usize = if (input.n_dims == 3)
            @intCast(input.shape[1])
        else if (input.n_dims == 2)
            @intCast(input.shape[0])
        else
            1;

        const weight_proj_len = try projectionOutputDim(w.in_proj, d_model);
        const qkv_len = try convChannelDim(w.conv1d_weight, d_conv);
        const minimum_proj = d_inner + (2 * n_v_heads);
        if (weight_proj_len <= minimum_proj) return error.InvalidShape;
        if (state.conv_state.len < qkv_len * d_conv) return error.InvalidShape;
        const proj_len = weight_proj_len;
        const proj_out = scratch.getProjection(proj_len);
        const temp = scratch.getConvOutput(d_inner);
        const ssm_out = scratch.getSsmOutput(d_inner);

        const full_input_data = input.asSlice(f32);
        const full_output_data = output.asSlice(f32);

        const conv_weight_t = self.conv_weight_transposed orelse {
            log.warn("inference", "CPU gated-delta missing time-major conv weights at execute", .{
                .layer_idx = self.layer_idx,
                .d_model = d_model,
                .d_conv = d_conv,
                .n_heads = n_v_heads,
                .d_head = d_head,
                .conv_dtype = @tagName(w.conv1d_weight.dtype),
                .conv_dims = w.conv1d_weight.n_dims,
                .conv_shape0 = if (w.conv1d_weight.n_dims > 0) w.conv1d_weight.shape[0] else 0,
                .conv_shape1 = if (w.conv1d_weight.n_dims > 1) w.conv1d_weight.shape[1] else 0,
            });
            return error.InvalidConfiguration;
        };
        const conv_bias = if (w.conv1d_bias) |bias| bias.asSlice(f32) else null;
        const A_log = w.A_log.asSlice(f32);
        const dt_bias = if (w.dt_bias) |bias| bias.asSlice(f32) else null;
        const conv_state = state.conv_state;
        const ssm_state = state.ssm_state;
        const norm_data = if (w.norm_weight) |norm_w| norm_w.asSlice(f32) else null;

        for (0..seq_len) |t| {
            const token_offset = t * d_model;
            const input_data = full_input_data[token_offset..][0..d_model];
            const output_data = full_output_data[token_offset..][0..d_model];

            var input_view = Tensor.view2DSlice(input_data, 1, d_model);
            var proj_view = Tensor.view2DSlice(proj_out, 1, proj_len);
            self.matmul_in_proj(&input_view, w.in_proj, &proj_view, matmul_scratch);

            const qkv = proj_out[0..qkv_len];
            const z = proj_out[qkv_len .. qkv_len + d_inner];
            const beta_raw = proj_out[qkv_len + d_inner .. qkv_len + d_inner + n_v_heads];
            const a_raw = proj_out[qkv_len + d_inner + n_v_heads .. qkv_len + d_inner + 2 * n_v_heads];

            if (qkv_len <= d_inner) return error.InvalidShape;
            const qk_total = qkv_len - d_inner;
            if ((qk_total % 2) != 0) return error.InvalidShape;
            const qk_inner = qk_total / 2;
            if ((qk_inner % d_head) != 0) return error.InvalidShape;
            const n_qk_heads = qk_inner / d_head;
            if (n_qk_heads == 0 or (n_v_heads % n_qk_heads) != 0) return error.InvalidShape;

            cpu_conv1d.runTimeMajorValues(qkv, conv_state, conv_weight_t, qkv, conv_bias, qkv_len, d_conv);
            cpu_gated_delta.applySiluInPlace(qkv);

            const query = qkv[0..qk_inner];
            const key = qkv[qk_inner .. 2 * qk_inner];
            const value = qkv[2 * qk_inner .. 2 * qk_inner + d_inner];
            try cpu_gated_delta.normalizeQueryKeyInPlace(query, key, n_qk_heads, d_head);
            try cpu_gated_delta.runStateSpaceStep(
                temp,
                ssm_out,
                ssm_state,
                query,
                key,
                value,
                beta_raw,
                a_raw,
                A_log,
                dt_bias,
                n_qk_heads,
                n_v_heads,
                d_head,
            );

            for (0..n_v_heads) |head_idx| {
                const out_head = ssm_out[head_idx * d_head ..][0..d_head];
                const z_head = z[head_idx * d_head ..][0..d_head];
                const norm_head = try cpu_gated_delta.normWeightSlice(norm_data, head_idx, d_head, d_inner);
                try cpu_gated_delta.applyGatedRmsNormInPlace(out_head, z_head, norm_head);
            }

            var ssm_view = Tensor.view2DSlice(ssm_out, 1, d_inner);
            var out_view = Tensor.view2DSlice(output_data, 1, d_model);
            self.matmul_out_proj(&ssm_view, w.out_proj, &out_view, matmul_scratch);
        }
    }
};

test "normWeightForHead rejects invalid norm shape" {
    const invalid = [_]f32{ 1.0, 2.0, 3.0 };
    try std.testing.expectError(
        error.InvalidShape,
        cpu_gated_delta.normWeightSlice(&invalid, 0, 4, 8),
    );
}
