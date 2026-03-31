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
const parallel = @import("../../../../system/parallel.zig");
const trace = @import("../../../../xray/root.zig").trace;

fn saturatingU64FromU128(value: u128) u64 {
    return if (value > std.math.maxInt(u64)) std.math.maxInt(u64) else @intCast(value);
}

fn tensorStorageBytes(weight: *const Tensor) u64 {
    var bytes: u128 = @intCast(weight.data_size);
    if (weight.gaffine) |meta| {
        bytes += meta.scales.len;
        bytes += meta.biases.len;
    }
    return saturatingU64FromU128(bytes);
}

fn matmulWork(rows: usize, k: usize, n: usize, weight: *const Tensor) trace.Work {
    const rows128: u128 = @intCast(rows);
    const k128: u128 = @intCast(k);
    const n128: u128 = @intCast(n);
    const flops = saturatingU64FromU128(2 * rows128 * k128 * n128);
    const input_bytes = rows128 * k128 * @sizeOf(f32);
    const output_bytes = rows128 * n128 * @sizeOf(f32);
    const bytes = saturatingU64FromU128(input_bytes + output_bytes + @as(u128, tensorStorageBytes(weight)));
    return .{ .flops = flops, .bytes = bytes };
}

/// Context for parallel SSM execution over heads.
/// Each head's state, kv_mem, and output slices are non-overlapping,
/// so heads can safely execute concurrently across threads.
const SsmParallelCtx = struct {
    proj_buf: []f32,
    ssm_buf: []f32,
    ssm_pre_norm_buf: ?[]f32,
    kv_mem: []f32,
    ssm_state: []f32,
    A_log: []const f32,
    dt_bias: ?[]const f32,
    norm_data: ?[]const f32,
    seq_len: usize,
    proj_len: usize,
    qkv_len: usize,
    qk_inner: usize,
    d_inner: usize,
    d_head: usize,
    n_qk_heads: usize,
    n_v_heads: usize,
};

/// parallelFor task: process a range of heads for all tokens.
/// Item count is inflated by INFLATE (16) to survive parallelFor's
/// cache-line alignment rounding with small head counts.
fn ssmHeadTask(start: usize, end: usize, ctx: *SsmParallelCtx) void {
    const INFLATE = 16;
    const head_start = start / INFLATE;
    const head_end = @min((end + INFLATE - 1) / INFLATE, ctx.n_v_heads);
    const qk_repeat = ctx.n_v_heads / ctx.n_qk_heads;

    for (head_start..head_end) |head_idx| {
        const qk_head_idx = head_idx / qk_repeat;
        const state_base = head_idx * ctx.d_head * ctx.d_head;
        const state_head = ctx.ssm_state[state_base..][0 .. ctx.d_head * ctx.d_head];
        const kv_mem_head = ctx.kv_mem[head_idx * ctx.d_head ..][0..ctx.d_head];
        const dt_bias_val: f32 = if (ctx.dt_bias) |bias| bias[head_idx] else 0.0;
        const norm_head: ?[]const f32 = if (ctx.norm_data) |nd| blk: {
            if (nd.len == ctx.d_head) break :blk nd;
            if (nd.len == ctx.d_inner) break :blk nd[head_idx * ctx.d_head ..][0..ctx.d_head];
            break :blk null;
        } else null;

        for (0..ctx.seq_len) |t| {
            const proj_t = ctx.proj_buf[t * ctx.proj_len ..];
            const qkv = proj_t[0..ctx.qkv_len];
            const query_head = qkv[qk_head_idx * ctx.d_head ..][0..ctx.d_head];
            const key_head = qkv[ctx.qk_inner + qk_head_idx * ctx.d_head ..][0..ctx.d_head];
            const value_head = qkv[2 * ctx.qk_inner + head_idx * ctx.d_head ..][0..ctx.d_head];
            const out_head = ctx.ssm_buf[t * ctx.d_inner + head_idx * ctx.d_head ..][0..ctx.d_head];

            cpu_gated_delta.runStateSpaceStepOneHead(
                kv_mem_head,
                out_head,
                state_head,
                query_head,
                key_head,
                value_head,
                proj_t[ctx.qkv_len + ctx.d_inner + head_idx],
                proj_t[ctx.qkv_len + ctx.d_inner + ctx.n_v_heads + head_idx],
                ctx.A_log[head_idx],
                dt_bias_val,
                ctx.d_head,
            );
            if (ctx.ssm_pre_norm_buf) |pre_norm_buf| {
                const pre_norm_head = pre_norm_buf[t * ctx.d_inner + head_idx * ctx.d_head ..][0..ctx.d_head];
                @memcpy(pre_norm_head, out_head);
            }

            const z_head = proj_t[ctx.qkv_len + head_idx * ctx.d_head ..][0..ctx.d_head];
            cpu_gated_delta.applyGatedRmsNormInPlace(out_head, z_head, norm_head) catch unreachable;
        }
    }
}

pub const GatedDeltaConfig = struct {
    d_model: u32,
    d_conv: u32,
    n_heads: u32,
    d_head: u32,
    n_key_heads: u32 = 0,
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
    n_key_heads: usize,
    d_head: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        batch_size: usize,
        config: GatedDeltaConfig,
    ) !GatedDeltaState {
        const n_v_heads: usize = @as(usize, config.n_heads);
        const n_qk_heads: usize = if (config.n_key_heads > 0) @as(usize, config.n_key_heads) else n_v_heads;
        if (n_v_heads == 0 or n_qk_heads == 0 or config.d_head == 0 or config.d_conv == 0) return error.InvalidShape;

        const d_head: usize = @as(usize, config.d_head);
        const d_inner = std.math.mul(usize, n_v_heads, d_head) catch return error.InvalidShape;
        const qkv_len = blk: {
            const qk_inner = std.math.mul(usize, n_qk_heads, d_head) catch return error.InvalidShape;
            const qk_total = std.math.mul(usize, qk_inner, 2) catch return error.InvalidShape;
            break :blk std.math.add(usize, qk_total, d_inner) catch return error.InvalidShape;
        };
        // Time-major state layout: [batch, d_conv, qkv_len].
        // This matches the SIMD-friendly conv1d_depthwise.runTimeMajor path.
        const conv_state_size = std.math.mul(usize, batch_size, std.math.mul(usize, qkv_len, @as(usize, config.d_conv)) catch return error.InvalidShape) catch return error.InvalidShape;
        const ssm_state_size = std.math.mul(usize, batch_size, std.math.mul(usize, d_inner, d_head) catch return error.InvalidShape) catch return error.InvalidShape;

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
            .n_key_heads = n_qk_heads,
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
        const n_v_heads: usize = @as(usize, config.n_heads);
        const n_qk_heads: usize = if (config.n_key_heads > 0) @as(usize, config.n_key_heads) else n_v_heads;
        if (n_v_heads == 0 or n_qk_heads == 0 or config.d_head == 0) return error.InvalidShape;

        const d_head: usize = @as(usize, config.d_head);
        const d_inner = std.math.mul(usize, n_v_heads, d_head) catch return error.InvalidShape;
        const qkv_len = blk: {
            const qk_inner = std.math.mul(usize, n_qk_heads, d_head) catch return error.InvalidShape;
            const qk_total = std.math.mul(usize, qk_inner, 2) catch return error.InvalidShape;
            break :blk std.math.add(usize, qk_total, d_inner) catch return error.InvalidShape;
        };
        const proj_len = blk: {
            const qkv_z = std.math.add(usize, qkv_len, d_inner) catch return error.InvalidShape;
            const ba = std.math.mul(usize, n_v_heads, 2) catch return error.InvalidShape;
            break :blk std.math.add(usize, qkv_z, ba) catch return error.InvalidShape;
        };
        const conv_len = d_inner;
        const ssm_len = d_inner;
        const total = blk: {
            const proj_conv = std.math.add(usize, proj_len, conv_len) catch return error.InvalidShape;
            break :blk std.math.add(usize, proj_conv, ssm_len) catch return error.InvalidShape;
        };
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
    trace_position_offset: usize = 0,
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
        const n_qk_heads_expected: usize = if (cfg.n_key_heads > 0) @as(usize, cfg.n_key_heads) else n_v_heads;
        const d_head: usize = cfg.d_head;
        if (n_v_heads == 0 or n_qk_heads_expected == 0 or d_head == 0 or d_conv == 0) return error.InvalidShape;

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
        const qkv_expected = blk: {
            const qk_inner = std.math.mul(usize, n_qk_heads_expected, d_head) catch return error.InvalidShape;
            const qk_total = std.math.mul(usize, qk_inner, 2) catch return error.InvalidShape;
            break :blk std.math.add(usize, qk_total, d_inner) catch return error.InvalidShape;
        };
        if (qkv_len != qkv_expected) return error.InvalidShape;

        const proj_len = blk: {
            const qkv_z = std.math.add(usize, qkv_expected, d_inner) catch return error.InvalidShape;
            const ba = std.math.mul(usize, n_v_heads, 2) catch return error.InvalidShape;
            break :blk std.math.add(usize, qkv_z, ba) catch return error.InvalidShape;
        };
        if (weight_proj_len != proj_len) return error.InvalidShape;
        if (state.conv_state.len < qkv_expected * d_conv) return error.InvalidShape;
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
        const trace_enabled = trace.isEnabled();

        // Validate shape invariants (independent of seq_len).
        if (qkv_len <= d_inner) return error.InvalidShape;
        const qk_total = qkv_len - d_inner;
        if ((qk_total % 2) != 0) return error.InvalidShape;
        const qk_inner = qk_total / 2;
        if ((qk_inner % d_head) != 0) return error.InvalidShape;
        const n_qk_heads = qk_inner / d_head;
        if (n_qk_heads != n_qk_heads_expected) return error.InvalidShape;
        if (n_qk_heads == 0 or (n_v_heads % n_qk_heads) != 0) return error.InvalidShape;

        if (seq_len > 1) {
            // Batched prefill: batch in_proj and out_proj, keep conv+SSM sequential.
            const proj_buf = try scratch.allocator.alloc(f32, seq_len * proj_len);
            defer scratch.allocator.free(proj_buf);
            const ssm_buf = try scratch.allocator.alloc(f32, seq_len * d_inner);
            defer scratch.allocator.free(ssm_buf);
            const conv_trace_buf: ?[]f32 = if (trace_enabled)
                try scratch.allocator.alloc(f32, seq_len * qkv_len)
            else
                null;
            defer if (conv_trace_buf) |buf| scratch.allocator.free(buf);
            const ssm_pre_norm_buf: ?[]f32 = if (trace_enabled)
                try scratch.allocator.alloc(f32, seq_len * d_inner)
            else
                null;
            defer if (ssm_pre_norm_buf) |buf| scratch.allocator.free(buf);

            // Batched in_proj: [seq_len × d_model] → [seq_len × proj_len]
            {
                var batch_input = Tensor.view2DSlice(full_input_data, seq_len, d_model);
                var batch_proj = Tensor.view2DSlice(proj_buf, seq_len, proj_len);
                self.matmul_in_proj(&batch_input, w.in_proj, &batch_proj, matmul_scratch);
                if (trace_enabled) {
                    trace.emitWithWork(
                        .gdelta_in_proj,
                        self.layer_idx,
                        0,
                        @intCast(self.trace_position_offset),
                        batch_proj.data().ptr,
                        .f32,
                        .{ 1, @intCast(seq_len), @intCast(proj_len), 0 },
                        3,
                        null,
                        matmulWork(seq_len, d_model, proj_len, w.in_proj),
                    );
                }
            }

            // Phase 1: Conv1d + SiLU + QK normalization (sequential — conv1d has cross-token state).
            for (0..seq_len) |t| {
                const proj_t = proj_buf[t * proj_len ..][0..proj_len];
                const qkv = proj_t[0..qkv_len];
                cpu_conv1d.runTimeMajorValues(qkv, conv_state, conv_weight_t, qkv, conv_bias, qkv_len, d_conv);
                if (conv_trace_buf) |buf| {
                    @memcpy(buf[t * qkv_len ..][0..qkv_len], qkv);
                }
                cpu_gated_delta.applySiluInPlace(qkv);
                const query = qkv[0..qk_inner];
                const key = qkv[qk_inner .. 2 * qk_inner];
                try cpu_gated_delta.normalizeQueryKeyInPlace(query, key, n_qk_heads, d_head);
            }
            if (trace_enabled and conv_trace_buf != null) {
                trace.emit(
                    .gdelta_conv,
                    self.layer_idx,
                    0,
                    @intCast(self.trace_position_offset),
                    @ptrCast(conv_trace_buf.?.ptr),
                    .f32,
                    .{ 1, @intCast(seq_len), @intCast(qkv_len), 0 },
                    3,
                    null,
                );
            }

            // Phase 2: SSM state step + gated RMS norm (parallel over heads).
            // Each head has independent state, kv_mem, and output slices.
            // One parallelFor per layer — each thread processes ~2 heads × all 510 tokens.
            var ssm_ctx = SsmParallelCtx{
                .proj_buf = proj_buf,
                .ssm_buf = ssm_buf,
                .ssm_pre_norm_buf = ssm_pre_norm_buf,
                .kv_mem = temp,
                .ssm_state = ssm_state,
                .A_log = A_log,
                .dt_bias = dt_bias,
                .norm_data = norm_data,
                .seq_len = seq_len,
                .proj_len = proj_len,
                .qkv_len = qkv_len,
                .qk_inner = qk_inner,
                .d_inner = d_inner,
                .d_head = d_head,
                .n_qk_heads = n_qk_heads,
                .n_v_heads = n_v_heads,
            };
            parallel.global().parallelFor(n_v_heads * 16, ssmHeadTask, &ssm_ctx);
            if (trace_enabled and ssm_pre_norm_buf != null) {
                trace.emit(
                    .gdelta_ssm,
                    self.layer_idx,
                    0,
                    @intCast(self.trace_position_offset),
                    @ptrCast(ssm_pre_norm_buf.?.ptr),
                    .f32,
                    .{ 1, @intCast(seq_len), @intCast(d_inner), 0 },
                    3,
                    null,
                );
                trace.emit(
                    .gdelta_norm,
                    self.layer_idx,
                    0,
                    @intCast(self.trace_position_offset),
                    @ptrCast(ssm_buf.ptr),
                    .f32,
                    .{ 1, @intCast(seq_len), @intCast(d_inner), 0 },
                    3,
                    null,
                );
            }

            // Batched out_proj: [seq_len × d_inner] → [seq_len × d_model]
            {
                var batch_ssm = Tensor.view2DSlice(ssm_buf, seq_len, d_inner);
                var batch_out = Tensor.view2DSlice(full_output_data, seq_len, d_model);
                self.matmul_out_proj(&batch_ssm, w.out_proj, &batch_out, matmul_scratch);
                if (trace_enabled) {
                    trace.emitWithWork(
                        .gdelta_out,
                        self.layer_idx,
                        0,
                        @intCast(self.trace_position_offset),
                        batch_out.data().ptr,
                        .f32,
                        .{ 1, @intCast(seq_len), @intCast(d_model), 0 },
                        3,
                        null,
                        matmulWork(seq_len, d_inner, d_model, w.out_proj),
                    );
                }
            }
            if (trace_enabled) {
                const state_pos: u32 = @intCast(self.trace_position_offset + seq_len - 1);
                trace.emit(
                    .gdelta_state_conv,
                    self.layer_idx,
                    0,
                    state_pos,
                    @ptrCast(conv_state.ptr),
                    .f32,
                    .{ 1, @intCast(d_conv), @intCast(qkv_len), 0 },
                    3,
                    null,
                );
                trace.emit(
                    .gdelta_state_ssm,
                    self.layer_idx,
                    0,
                    state_pos,
                    @ptrCast(ssm_state.ptr),
                    .f32,
                    .{ 1, @intCast(n_v_heads), @intCast(d_head), @intCast(d_head) },
                    4,
                    null,
                );
            }
            return;
        }

        // Single-token decode path (seq_len ≤ 1).
        for (0..seq_len) |t| {
            const token_offset = t * d_model;
            const input_data = full_input_data[token_offset..][0..d_model];
            const output_data = full_output_data[token_offset..][0..d_model];

            var input_view = Tensor.view2DSlice(input_data, 1, d_model);
            var proj_view = Tensor.view2DSlice(proj_out, 1, proj_len);
            self.matmul_in_proj(&input_view, w.in_proj, &proj_view, matmul_scratch);
            if (trace_enabled) {
                trace.emitWithWork(
                    .gdelta_in_proj,
                    self.layer_idx,
                    0,
                    @intCast(self.trace_position_offset + t),
                    proj_view.data().ptr,
                    .f32,
                    .{ 1, 1, @intCast(proj_len), 0 },
                    3,
                    null,
                    matmulWork(1, d_model, proj_len, w.in_proj),
                );
            }

            const qkv = proj_out[0..qkv_len];
            const z = proj_out[qkv_len .. qkv_len + d_inner];
            const beta_raw = proj_out[qkv_len + d_inner .. qkv_len + d_inner + n_v_heads];
            const a_raw = proj_out[qkv_len + d_inner + n_v_heads .. qkv_len + d_inner + 2 * n_v_heads];

            cpu_conv1d.runTimeMajorValues(qkv, conv_state, conv_weight_t, qkv, conv_bias, qkv_len, d_conv);
            if (trace_enabled) {
                trace.emit(
                    .gdelta_conv,
                    self.layer_idx,
                    0,
                    @intCast(self.trace_position_offset + t),
                    @ptrCast(qkv.ptr),
                    .f32,
                    .{ 1, 1, @intCast(qkv_len), 0 },
                    3,
                    null,
                );
            }
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
            if (trace_enabled) {
                trace.emit(
                    .gdelta_ssm,
                    self.layer_idx,
                    0,
                    @intCast(self.trace_position_offset + t),
                    @ptrCast(ssm_out.ptr),
                    .f32,
                    .{ 1, 1, @intCast(d_inner), 0 },
                    3,
                    null,
                );
            }

            for (0..n_v_heads) |head_idx| {
                const out_head = ssm_out[head_idx * d_head ..][0..d_head];
                const z_head = z[head_idx * d_head ..][0..d_head];
                const norm_head = try cpu_gated_delta.normWeightSlice(norm_data, head_idx, d_head, d_inner);
                try cpu_gated_delta.applyGatedRmsNormInPlace(out_head, z_head, norm_head);
            }
            if (trace_enabled) {
                trace.emit(
                    .gdelta_norm,
                    self.layer_idx,
                    0,
                    @intCast(self.trace_position_offset + t),
                    @ptrCast(ssm_out.ptr),
                    .f32,
                    .{ 1, 1, @intCast(d_inner), 0 },
                    3,
                    null,
                );
            }

            var ssm_view = Tensor.view2DSlice(ssm_out, 1, d_inner);
            var out_view = Tensor.view2DSlice(output_data, 1, d_model);
            self.matmul_out_proj(&ssm_view, w.out_proj, &out_view, matmul_scratch);
            if (trace_enabled) {
                trace.emitWithWork(
                    .gdelta_out,
                    self.layer_idx,
                    0,
                    @intCast(self.trace_position_offset + t),
                    out_view.data().ptr,
                    .f32,
                    .{ 1, 1, @intCast(d_model), 0 },
                    3,
                    null,
                    matmulWork(1, d_inner, d_model, w.out_proj),
                );
                trace.emit(
                    .gdelta_state_conv,
                    self.layer_idx,
                    0,
                    @intCast(self.trace_position_offset + t),
                    @ptrCast(conv_state.ptr),
                    .f32,
                    .{ 1, @intCast(d_conv), @intCast(qkv_len), 0 },
                    3,
                    null,
                );
                trace.emit(
                    .gdelta_state_ssm,
                    self.layer_idx,
                    0,
                    @intCast(self.trace_position_offset + t),
                    @ptrCast(ssm_state.ptr),
                    .f32,
                    .{ 1, @intCast(n_v_heads), @intCast(d_head), @intCast(d_head) },
                    4,
                    null,
                );
            }
        }
    }

    /// Batched decode across multiple scheduler slots.
    /// Input/output are [1, batch_size, d_model]. Each slot has independent
    /// conv/SSM state. Matmuls (in_proj, out_proj) are batched; conv1d and
    /// SSM state updates run per-slot.
    pub fn forwardBatchedSlots(
        self: *const GatedDeltaKernel,
        input: *const Tensor,
        output: *Tensor,
        slot_states: []const *GatedDeltaState,
        scratch: *GatedDeltaScratch,
        matmul_scratch: *cpu_linalg.MatmulScratch,
    ) !void {
        // Sequential per-slot fallback to isolate dispatch vs kernel issues.
        const d_model: usize = self.config.d_model;
        const batch_size = slot_states.len;
        if (batch_size == 0) return;

        const full_input_data = input.asSlice(f32);
        const full_output_data = output.asSlice(f32);

        for (slot_states, 0..) |slot_state, s| {
            const row_input = full_input_data[s * d_model ..][0..d_model];
            const row_output = full_output_data[s * d_model ..][0..d_model];
            var input_view = Tensor.view2DSlice(@constCast(row_input), 1, d_model);
            var output_view = Tensor.view2DSlice(row_output, 1, d_model);
            try self.forward(&input_view, &output_view, slot_state, scratch, matmul_scratch);
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

test "GatedDeltaState.init symmetric heads defaults n_key_heads to n_heads" {
    const cfg = GatedDeltaConfig{
        .d_model = 1024,
        .d_conv = 4,
        .n_heads = 16,
        .d_head = 128,
        // n_key_heads defaults to 0, should fall back to n_heads=16
    };
    var state = try GatedDeltaState.init(std.testing.allocator, 1, cfg);
    defer state.deinit();
    // n_qk_heads = 16, d_inner = 16*128 = 2048
    // qkv_len = 2*16*128 + 2048 = 6144
    // conv_state = 1 * 6144 * 4 = 24576
    // ssm_state = 1 * 2048 * 128 = 262144
    try std.testing.expectEqual(@as(usize, 24576), state.conv_state.len);
    try std.testing.expectEqual(@as(usize, 262144), state.ssm_state.len);
}

test "GatedDeltaState.init asymmetric heads uses n_key_heads for conv sizing" {
    const cfg = GatedDeltaConfig{
        .d_model = 1024,
        .d_conv = 4,
        .n_heads = 16,
        .n_key_heads = 8,
        .d_head = 128,
    };
    var state = try GatedDeltaState.init(std.testing.allocator, 1, cfg);
    defer state.deinit();
    // n_qk_heads = 8, d_inner = 16*128 = 2048
    // qkv_len = 2*8*128 + 2048 = 4096
    // conv_state = 1 * 4096 * 4 = 16384  (smaller than symmetric)
    // ssm_state = 1 * 2048 * 128 = 262144  (unchanged, uses n_heads)
    try std.testing.expectEqual(@as(usize, 16384), state.conv_state.len);
    try std.testing.expectEqual(@as(usize, 262144), state.ssm_state.len);
}

test "GatedDeltaScratch.init asymmetric heads sizes buffer for actual projection split" {
    const asym_cfg = GatedDeltaConfig{
        .d_model = 1024,
        .d_conv = 4,
        .n_heads = 16,
        .n_key_heads = 8,
        .d_head = 128,
    };
    var asym_scratch = try GatedDeltaScratch.init(std.testing.allocator, asym_cfg);
    defer asym_scratch.deinit();

    const sym_cfg = GatedDeltaConfig{
        .d_model = 1024,
        .d_conv = 4,
        .n_heads = 16,
        .d_head = 128,
        // n_key_heads=0 defaults to n_heads=16
    };
    var sym_scratch = try GatedDeltaScratch.init(std.testing.allocator, sym_cfg);
    defer sym_scratch.deinit();

    // Asymmetric: qkv_len=4096, proj_len=4096+2048+32=6176, total=6176+2048+2048=10272
    // Symmetric:  qkv_len=6144, proj_len=6144+2048+32=8224, total=8224+2048+2048=12320
    try std.testing.expectEqual(@as(usize, 10272), asym_scratch.buffer.len);
    try std.testing.expectEqual(@as(usize, 12320), sym_scratch.buffer.len);
    // Asymmetric must be smaller due to fewer key/query heads in projection
    try std.testing.expect(asym_scratch.buffer.len < sym_scratch.buffer.len);
}
