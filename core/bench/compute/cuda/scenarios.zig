const std = @import("std");
const main = @import("main");
const compute = main.compute;
const cuda = compute.cuda;
const dtype = main.core.dtype;
const models = main.models.dispatcher;
const perf_hints = models.perf_hints;
const harness = @import("harness.zig");

pub const Scenario = enum {
    decode_gdelta_ssm_i8,
    decode_gdelta_in_proj,
    decode_gdelta_out_proj,
    decode_gdelta_conv_silu_ptrs,
    decode_gdelta_norm_gate_rows,
    decode_attn_rope,
    decode_attn_scores,
    decode_attn_softmax,
    decode_attn_weighted_sum,
    decode_attn_q,
    decode_attn_out,
    decode_attn_i8_kv_ptrs,
    decode_attn_i8_kv_fused,
    decode_attn_i8_flash,
    decode_attn_qkv_fused,
    decode_token_chain,
    decode_ffn_gate,
    decode_ffn_down,
    decode_ffn_gate_up_fused_silu,
    decode_lm_head,
    prefill_attn_q,
    prefill_attn_qkv_fused,
    prefill_ffn_gate,
    prefill_ffn_gate_up_fused_silu,
    prefill_ffn_down,
};

pub const QuantKind = enum {
    tq4,
    nvfp4,
    nvfp4_i8cache,
    nvfp4_native,
};

pub const RunConfig = struct {
    warmup: usize = 10,
    iters: usize = 30,
    model_id: []const u8 = "qwen3_5",
    rows_override: ?usize = null,
    seq_len_override: ?usize = null,
    layers_override: ?usize = null,
};

pub const ScenarioResult = struct {
    name: []const u8,
    quant: QuantKind,
    samples: []harness.Sample,
    flops_per_iter: u64,
    bytes_per_iter: u64,
    rows: usize,
    in_dim: usize,
    out_dim: usize,

    pub fn deinit(self: *ScenarioResult, allocator: std.mem.Allocator) void {
        allocator.free(self.samples);
    }
};

const RoleMatmulDims = struct {
    tokens: usize,
    hidden: usize,
    out: usize,
};

const QkvDims = struct {
    rows: usize,
    hidden: usize,
    q_out: usize,
    k_out: usize,
    v_out: usize,

    fn totalOut(self: QkvDims) usize {
        return self.q_out + self.k_out + self.v_out;
    }
};

const GateUpDims = struct {
    rows: usize,
    hidden: usize,
    out: usize,
};

const DecodeAttentionDims = struct {
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    rope_theta: f32,

    fn qDim(self: DecodeAttentionDims) usize {
        return self.n_heads * self.head_dim;
    }

    fn kvRowStride(self: DecodeAttentionDims) usize {
        return self.n_kv_heads * self.head_dim;
    }

    fn kvGroups(self: DecodeAttentionDims) usize {
        return self.n_heads / self.n_kv_heads;
    }
};

const DecodeGatedDeltaDims = struct {
    rows: usize,
    hidden: usize,
    qkv_len: usize,
    d_inner: usize,
    n_qk_heads: usize,
    n_v_heads: usize,
    d_head: usize,
    d_conv: usize,

    fn projLen(self: DecodeGatedDeltaDims) usize {
        return self.qkv_len + self.d_inner + (2 * self.n_v_heads);
    }
};

const DecodeTokenChainDims = struct {
    qkv: QkvDims,
    attention: DecodeAttentionDims,
    attn_out: RoleMatmulDims,
    gate_up: GateUpDims,
    ffn_down: RoleMatmulDims,
    lm_head: RoleMatmulDims,
    layers: usize,
};

fn roleBenchRowName(which: Scenario) []const u8 {
    return switch (which) {
        .decode_gdelta_in_proj,
        .decode_gdelta_out_proj,
        .decode_gdelta_conv_silu_ptrs,
        .decode_gdelta_norm_gate_rows,
        .decode_attn_rope,
        .decode_attn_scores,
        .decode_attn_softmax,
        .decode_attn_weighted_sum,
        .decode_attn_q => "decode.attn_q",
        .decode_attn_out => "decode.attn_out",
        .decode_ffn_gate => "decode.ffn_gate",
        .decode_ffn_down => "decode.ffn_down",
        .decode_lm_head => "decode.lm_head_runtime_f32",
        .prefill_attn_q => "prefill.attn_q",
        .prefill_ffn_gate => "prefill.ffn_gate",
        .prefill_ffn_down => "prefill.ffn_down",
        .decode_attn_i8_kv_ptrs,
        .decode_attn_i8_kv_fused,
        .decode_attn_i8_flash,
        .decode_gdelta_ssm_i8,
        .decode_attn_qkv_fused,
        .decode_token_chain,
        .prefill_attn_qkv_fused,
        .decode_ffn_gate_up_fused_silu,
        .prefill_ffn_gate_up_fused_silu,
        => unreachable,
    };
}

fn qkvBenchRowNames(which: Scenario) struct { q: []const u8, k: []const u8, v: []const u8 } {
    return switch (which) {
        .decode_attn_qkv_fused => .{ .q = "decode.attn_q", .k = "decode.attn_k", .v = "decode.attn_v" },
        .prefill_attn_qkv_fused => .{ .q = "prefill.attn_q", .k = "prefill.attn_k", .v = "prefill.attn_v" },
        else => unreachable,
    };
}

fn gateUpBenchRowName(which: Scenario) []const u8 {
    return switch (which) {
        .decode_ffn_gate_up_fused_silu => "decode.ffn_gate",
        .prefill_ffn_gate_up_fused_silu => "prefill.ffn_gate",
        else => unreachable,
    };
}

fn modelRoleDims(model_id: []const u8, which: Scenario, rows_override: ?usize) !RoleMatmulDims {
    switch (which) {
        .decode_gdelta_in_proj => {
            const dims = try decodeGatedDeltaDims(model_id, rows_override);
            return .{ .tokens = dims.rows, .hidden = dims.hidden, .out = dims.projLen() };
        },
        .decode_gdelta_out_proj => {
            const dims = try decodeGatedDeltaDims(model_id, rows_override);
            return .{ .tokens = dims.rows, .hidden = dims.n_qk_heads * dims.d_head, .out = dims.hidden };
        },
        else => {},
    }
    const bench_row = roleBenchRowName(which);
    const dims = resolveRoleDims(model_id, bench_row) orelse return error.InvalidArgument;
    return .{
        .tokens = rows_override orelse dims.tokens,
        .hidden = dims.hidden,
        .out = dims.out,
    };
}

fn modelQkvDims(model_id: []const u8, which: Scenario, rows_override: ?usize) !QkvDims {
    const rowspec = qkvBenchRowNames(which);
    const q = resolveRoleDims(model_id, rowspec.q) orelse return error.InvalidArgument;
    const k = resolveRoleDims(model_id, rowspec.k) orelse return error.InvalidArgument;
    const v = resolveRoleDims(model_id, rowspec.v) orelse return error.InvalidArgument;
    if (q.hidden != k.hidden or q.hidden != v.hidden) return error.InvalidArgument;
    return .{
        .rows = rows_override orelse q.tokens,
        .hidden = q.hidden,
        .q_out = q.out,
        .k_out = k.out,
        .v_out = v.out,
    };
}

fn modelGateUpDims(model_id: []const u8, which: Scenario, rows_override: ?usize) !GateUpDims {
    const dims = resolveRoleDims(model_id, gateUpBenchRowName(which)) orelse return error.InvalidArgument;
    return .{
        .rows = rows_override orelse dims.tokens,
        .hidden = dims.hidden,
        .out = dims.out,
    };
}

fn decodeAttentionDims(model_id: []const u8, seq_len_override: ?usize) !DecodeAttentionDims {
    const seq_len = seq_len_override orelse 2048;
    if (seq_len == 0) return error.InvalidArgument;

    if (std.mem.eql(u8, model_id, "qwen3_5_4b")) {
        return .{
            .seq_len = seq_len,
            .n_heads = 16,
            .n_kv_heads = 4,
            .head_dim = 256,
            .rope_dim = 64,
            .rope_theta = 10_000_000.0,
        };
    }
    if (std.mem.eql(u8, model_id, "qwen3_5")) {
        return .{
            .seq_len = seq_len,
            .n_heads = 8,
            .n_kv_heads = 2,
            .head_dim = 256,
            .rope_dim = 64,
            .rope_theta = 10_000_000.0,
        };
    }
    return error.InvalidArgument;
}

fn modelLayerCount(model_id: []const u8) ?usize {
    if (std.mem.eql(u8, model_id, "qwen3_5_4b")) return 32;
    if (std.mem.eql(u8, model_id, "qwen3_5")) return 24;
    return null;
}

fn decodeGatedDeltaDims(model_id: []const u8, rows_override: ?usize) !DecodeGatedDeltaDims {
    const rows = rows_override orelse 1;
    if (rows == 0) return error.InvalidArgument;

    if (std.mem.eql(u8, model_id, "qwen3_5_4b")) {
        return .{
            .rows = rows,
            .hidden = 2560,
            .qkv_len = 8192,
            .d_inner = 4096,
            .n_qk_heads = 16,
            .n_v_heads = 32,
            .d_head = 128,
            .d_conv = 4,
        };
    }
    if (std.mem.eql(u8, model_id, "qwen3_5")) {
        return .{
            .rows = rows,
            .hidden = 1024,
            .qkv_len = 4096,
            .d_inner = 2048,
            .n_qk_heads = 8,
            .n_v_heads = 16,
            .d_head = 128,
            .d_conv = 4,
        };
    }
    return error.InvalidArgument;
}

fn decodeTokenChainDims(model_id: []const u8, seq_len_override: ?usize, layers_override: ?usize) !DecodeTokenChainDims {
    return .{
        .qkv = try modelQkvDims(model_id, .decode_attn_qkv_fused, 1),
        .attention = try decodeAttentionDims(model_id, seq_len_override),
        .attn_out = try modelRoleDims(model_id, .decode_attn_out, 1),
        .gate_up = try modelGateUpDims(model_id, .decode_ffn_gate_up_fused_silu, 1),
        .ffn_down = try modelRoleDims(model_id, .decode_ffn_down, 1),
        .lm_head = try modelRoleDims(model_id, .decode_lm_head, 1),
        .layers = layers_override orelse modelLayerCount(model_id) orelse return error.InvalidArgument,
    };
}

fn resolveRoleDims(model_id: []const u8, bench_row: []const u8) ?perf_hints.RoleDims {
    if (models.performanceHintsByName(model_id)) |hints| {
        if (findRoleDims(hints, bench_row)) |dims| return dims;
    }
    return perf_hints.defaultRoleDimsForModel(model_id, bench_row);
}

fn findRoleDims(hints: *const perf_hints.PerfHints, bench_row: []const u8) ?perf_hints.RoleDims {
    for (hints.role_dims) |dims| {
        if (std.mem.eql(u8, dims.bench_row, bench_row)) return dims;
    }
    return null;
}

fn selectQuantLabel(tq4: []const u8, nvfp4: []const u8, nvfp4_i8cache: []const u8, nvfp4_native: []const u8, quant: QuantKind) []const u8 {
    return switch (quant) {
        .tq4 => tq4,
        .nvfp4 => nvfp4,
        .nvfp4_i8cache => nvfp4_i8cache,
        .nvfp4_native => nvfp4_native,
    };
}

fn scenarioName(which: Scenario, quant: QuantKind) []const u8 {
    return switch (which) {
        .decode_gdelta_ssm_i8 => selectQuantLabel("decode.gdelta_ssm_i8.tq4", "decode.gdelta_ssm_i8.nvfp4", "decode.gdelta_ssm_i8.nvfp4_i8cache", "decode.gdelta_ssm_i8.nvfp4_native", quant),
        .decode_gdelta_in_proj => selectQuantLabel("decode.gdelta_in_proj.tq4", "decode.gdelta_in_proj.nvfp4", "decode.gdelta_in_proj.nvfp4_i8cache", "decode.gdelta_in_proj.nvfp4_native", quant),
        .decode_gdelta_out_proj => selectQuantLabel("decode.gdelta_out_proj.tq4", "decode.gdelta_out_proj.nvfp4", "decode.gdelta_out_proj.nvfp4_i8cache", "decode.gdelta_out_proj.nvfp4_native", quant),
        .decode_gdelta_conv_silu_ptrs => selectQuantLabel("decode.gdelta_conv_silu_ptrs.tq4", "decode.gdelta_conv_silu_ptrs.nvfp4", "decode.gdelta_conv_silu_ptrs.nvfp4_i8cache", "decode.gdelta_conv_silu_ptrs.nvfp4_native", quant),
        .decode_gdelta_norm_gate_rows => selectQuantLabel("decode.gdelta_norm_gate_rows.tq4", "decode.gdelta_norm_gate_rows.nvfp4", "decode.gdelta_norm_gate_rows.nvfp4_i8cache", "decode.gdelta_norm_gate_rows.nvfp4_native", quant),
        .decode_attn_rope => selectQuantLabel("decode.attn_rope.tq4", "decode.attn_rope.nvfp4", "decode.attn_rope.nvfp4_i8cache", "decode.attn_rope.nvfp4_native", quant),
        .decode_attn_scores => selectQuantLabel("decode.attn_scores.tq4", "decode.attn_scores.nvfp4", "decode.attn_scores.nvfp4_i8cache", "decode.attn_scores.nvfp4_native", quant),
        .decode_attn_softmax => selectQuantLabel("decode.attn_softmax.tq4", "decode.attn_softmax.nvfp4", "decode.attn_softmax.nvfp4_i8cache", "decode.attn_softmax.nvfp4_native", quant),
        .decode_attn_weighted_sum => selectQuantLabel("decode.attn_weighted_sum.tq4", "decode.attn_weighted_sum.nvfp4", "decode.attn_weighted_sum.nvfp4_i8cache", "decode.attn_weighted_sum.nvfp4_native", quant),
        .decode_attn_q => selectQuantLabel("decode.attn_q.tq4", "decode.attn_q.nvfp4", "decode.attn_q.nvfp4_i8cache", "decode.attn_q.nvfp4_native", quant),
        .decode_attn_out => selectQuantLabel("decode.attn_out.tq4", "decode.attn_out.nvfp4", "decode.attn_out.nvfp4_i8cache", "decode.attn_out.nvfp4_native", quant),
        .decode_attn_i8_kv_ptrs => selectQuantLabel("decode.attn_i8_kv_ptrs.tq4", "decode.attn_i8_kv_ptrs.nvfp4", "decode.attn_i8_kv_ptrs.nvfp4_i8cache", "decode.attn_i8_kv_ptrs.nvfp4_native", quant),
        .decode_attn_i8_kv_fused => selectQuantLabel("decode.attn_i8_kv_fused.tq4", "decode.attn_i8_kv_fused.nvfp4", "decode.attn_i8_kv_fused.nvfp4_i8cache", "decode.attn_i8_kv_fused.nvfp4_native", quant),
        .decode_attn_i8_flash => selectQuantLabel("decode.attn.i8_flash.tq4", "decode.attn.i8_flash.nvfp4", "decode.attn.i8_flash.nvfp4_i8cache", "decode.attn.i8_flash.nvfp4_native", quant),
        .decode_attn_qkv_fused => selectQuantLabel("decode.attn_qkv_fused.tq4", "decode.attn_qkv_fused.nvfp4", "decode.attn_qkv_fused.nvfp4_i8cache", "decode.attn_qkv_fused.nvfp4_native", quant),
        .decode_token_chain => selectQuantLabel("decode.token_chain.tq4", "decode.token_chain.nvfp4", "decode.token_chain.nvfp4_i8cache", "decode.token_chain.nvfp4_native", quant),
        .decode_ffn_gate => selectQuantLabel("decode.ffn_gate.tq4", "decode.ffn_gate.nvfp4", "decode.ffn_gate.nvfp4_i8cache", "decode.ffn_gate.nvfp4_native", quant),
        .decode_ffn_down => selectQuantLabel("decode.ffn_down.tq4", "decode.ffn_down.nvfp4", "decode.ffn_down.nvfp4_i8cache", "decode.ffn_down.nvfp4_native", quant),
        .decode_ffn_gate_up_fused_silu => selectQuantLabel("decode.ffn_gate_up_fused_silu.tq4", "decode.ffn_gate_up_fused_silu.nvfp4", "decode.ffn_gate_up_fused_silu.nvfp4_i8cache", "decode.ffn_gate_up_fused_silu.nvfp4_native", quant),
        .decode_lm_head => selectQuantLabel("decode.lm_head.tq4", "decode.lm_head.nvfp4", "decode.lm_head.nvfp4_i8cache", "decode.lm_head.nvfp4_native", quant),
        .prefill_attn_q => selectQuantLabel("prefill.attn_q.tq4", "prefill.attn_q.nvfp4", "prefill.attn_q.nvfp4_i8cache", "prefill.attn_q.nvfp4_native", quant),
        .prefill_attn_qkv_fused => selectQuantLabel("prefill.attn_qkv_fused.tq4", "prefill.attn_qkv_fused.nvfp4", "prefill.attn_qkv_fused.nvfp4_i8cache", "prefill.attn_qkv_fused.nvfp4_native", quant),
        .prefill_ffn_gate => selectQuantLabel("prefill.ffn_gate.tq4", "prefill.ffn_gate.nvfp4", "prefill.ffn_gate.nvfp4_i8cache", "prefill.ffn_gate.nvfp4_native", quant),
        .prefill_ffn_gate_up_fused_silu => selectQuantLabel("prefill.ffn_gate_up_fused_silu.tq4", "prefill.ffn_gate_up_fused_silu.nvfp4", "prefill.ffn_gate_up_fused_silu.nvfp4_i8cache", "prefill.ffn_gate_up_fused_silu.nvfp4_native", quant),
        .prefill_ffn_down => selectQuantLabel("prefill.ffn_down.tq4", "prefill.ffn_down.nvfp4", "prefill.ffn_down.nvfp4_i8cache", "prefill.ffn_down.nvfp4_native", quant),
    };
}

pub fn runScenario(allocator: std.mem.Allocator, which: Scenario, quant: QuantKind, cfg: RunConfig) !ScenarioResult {
    var device = try cuda.Device.init();
    defer device.deinit();
    var registry = cuda.Registry.init(allocator, &device);
    defer registry.deinit();
    var arg_pack = cuda.ArgPack.init(allocator);
    defer arg_pack.deinit();
    const stream = try device.createStream();
    defer device.destroyStream(stream);
    device.setLaunchStream(stream);
    defer device.setLaunchStream(null);

    if (!device.supportsEventTiming()) return error.CudaEventApiUnavailable;

    const start_evt = try device.createTimingEvent();
    defer device.destroyEvent(start_evt);
    const stop_evt = try device.createTimingEvent();
    defer device.destroyEvent(stop_evt);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    return switch (which) {
        .decode_gdelta_ssm_i8 => try runDecodeGatedDeltaSsmI8Scenario(allocator, quant, cfg, &device, &registry, &arg_pack, stream, start_evt, stop_evt, samples),
        .decode_gdelta_conv_silu_ptrs => try runDecodeGatedDeltaConvSiluPtrsScenario(allocator, quant, cfg, &device, &registry, &arg_pack, stream, start_evt, stop_evt, samples),
        .decode_gdelta_norm_gate_rows => try runDecodeGatedDeltaNormGateRowsScenario(allocator, quant, cfg, &device, &registry, &arg_pack, stream, start_evt, stop_evt, samples),
        .decode_attn_rope => try runDecodeAttentionRopeScenario(allocator, quant, cfg, &device, &registry, &arg_pack, stream, start_evt, stop_evt, samples),
        .decode_attn_scores => try runDecodeAttentionScoresScenario(allocator, quant, cfg, &device, &registry, &arg_pack, stream, start_evt, stop_evt, samples),
        .decode_attn_softmax => try runDecodeAttentionSoftmaxScenario(allocator, quant, cfg, &device, &registry, &arg_pack, stream, start_evt, stop_evt, samples),
        .decode_attn_weighted_sum => try runDecodeAttentionWeightedSumScenario(allocator, quant, cfg, &device, &registry, &arg_pack, stream, start_evt, stop_evt, samples),
        .decode_gdelta_in_proj,
        .decode_gdelta_out_proj,
        .decode_attn_q,
        .decode_attn_out,
        .decode_ffn_gate,
        .decode_ffn_down,
        .decode_lm_head,
        .prefill_attn_q,
        .prefill_ffn_gate,
        .prefill_ffn_down,
        => try runMatvecScenario(allocator, which, quant, cfg, &device, &registry, &arg_pack, stream, start_evt, stop_evt, samples),
        .decode_attn_i8_kv_ptrs => try runDecodeAttentionI8PtrsScenario(allocator, quant, cfg, &device, &registry, &arg_pack, stream, start_evt, stop_evt, samples),
        .decode_attn_i8_kv_fused => try runDecodeAttentionI8FusedScenario(allocator, quant, cfg, &device, &registry, &arg_pack, stream, start_evt, stop_evt, samples),
        .decode_attn_i8_flash => try runDecodeAttentionI8FlashScenario(allocator, quant, cfg, &device, &registry, &arg_pack, stream, start_evt, stop_evt, samples),
        .decode_attn_qkv_fused,
        .prefill_attn_qkv_fused,
        => try runQkvScenario(allocator, which, quant, cfg, &device, &registry, &arg_pack, stream, start_evt, stop_evt, samples),
        .decode_token_chain => try runDecodeTokenChainScenario(allocator, quant, cfg, &device, &registry, &arg_pack, stream, start_evt, stop_evt, samples),
        .decode_ffn_gate_up_fused_silu,
        .prefill_ffn_gate_up_fused_silu,
        => try runGateUpSiluScenario(allocator, which, quant, cfg, &device, &registry, &arg_pack, stream, start_evt, stop_evt, samples),
    };
}

fn runDecodeGatedDeltaSsmI8Scenario(
    allocator: std.mem.Allocator,
    quant: QuantKind,
    cfg: RunConfig,
    device: *cuda.Device,
    registry: *cuda.Registry,
    arg_pack: *cuda.ArgPack,
    stream: cuda.StreamHandle,
    start_evt: cuda.EventHandle,
    stop_evt: cuda.EventHandle,
    samples: []harness.Sample,
) !ScenarioResult {
    const dims = try decodeGatedDeltaDims(cfg.model_id, cfg.rows_override);
    const rows = dims.rows;
    const row_stride = dims.projLen();
    const out_row_stride = dims.d_inner;
    const beta_offset = dims.qkv_len + dims.d_inner;
    const a_offset = beta_offset + dims.n_v_heads;
    const qkv_rows_count = std.math.mul(usize, rows, row_stride) catch return error.InvalidArgument;
    const out_count = std.math.mul(usize, rows, out_row_stride) catch return error.InvalidArgument;
    const state_data_bytes = std.math.mul(usize, dims.n_v_heads * dims.d_head, dims.d_head) catch return error.InvalidArgument;
    const state_scales_count = std.math.mul(usize, dims.n_v_heads, dims.d_head) catch return error.InvalidArgument;
    const state_scales_offset = state_data_bytes;
    const state_total_bytes = state_data_bytes + state_scales_count * @sizeOf(f32);

    const qkv_rows_host = try allocator.alloc(f32, qkv_rows_count);
    defer allocator.free(qkv_rows_host);
    const a_log_host = try allocator.alloc(f32, dims.n_v_heads);
    defer allocator.free(a_log_host);
    const dt_bias_host = try allocator.alloc(f32, dims.n_v_heads);
    defer allocator.free(dt_bias_host);
    const state_i8_host = try allocator.alloc(i8, state_data_bytes);
    defer allocator.free(state_i8_host);
    const state_scales_host = try allocator.alloc(f32, state_scales_count);
    defer allocator.free(state_scales_host);
    fillInputPattern(qkv_rows_host);
    fillF32ScalePattern(a_log_host);
    fillF32ScalePattern(dt_bias_host);
    fillI8Pattern(state_i8_host);
    fillF32ScalePattern(state_scales_host);

    const state_bytes_host = try allocator.alloc(u8, state_total_bytes);
    defer allocator.free(state_bytes_host);
    @memcpy(state_bytes_host[0..state_data_bytes], std.mem.sliceAsBytes(state_i8_host));
    @memcpy(state_bytes_host[state_scales_offset .. state_scales_offset + state_scales_count * @sizeOf(f32)], std.mem.sliceAsBytes(state_scales_host));

    var qkv_rows_dev = try device.allocBuffer(qkv_rows_count * @sizeOf(f32));
    defer qkv_rows_dev.deinit(device);
    var a_log_dev = try device.allocBuffer(dims.n_v_heads * @sizeOf(f32));
    defer a_log_dev.deinit(device);
    var dt_bias_dev = try device.allocBuffer(dims.n_v_heads * @sizeOf(f32));
    defer dt_bias_dev.deinit(device);
    var out_dev = try device.allocBuffer(out_count * @sizeOf(f32));
    defer out_dev.deinit(device);
    var state_dev = try device.allocBuffer(state_total_bytes);
    defer state_dev.deinit(device);
    var state_ptrs_dev = try device.allocBuffer(rows * @sizeOf(u64));
    defer state_ptrs_dev.deinit(device);

    try qkv_rows_dev.upload(device, std.mem.sliceAsBytes(qkv_rows_host));
    try a_log_dev.upload(device, std.mem.sliceAsBytes(a_log_host));
    try dt_bias_dev.upload(device, std.mem.sliceAsBytes(dt_bias_host));
    try state_dev.upload(device, state_bytes_host);

    const state_ptrs_host = try allocator.alloc(u64, rows);
    defer allocator.free(state_ptrs_host);
    for (state_ptrs_host, 0..) |*ptr, row| {
        ptr.* = state_dev.pointer;
        _ = row;
    }
    try state_ptrs_dev.upload(device, std.mem.sliceAsBytes(state_ptrs_host));

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(cuda.gated_delta_ssm_rows_ptrs_i8.embedded_module);
    const fn_info = try registry.resolveFunction(cuda.gated_delta_ssm_rows_ptrs_i8.op_name, cuda.gated_delta_ssm_rows_ptrs_i8.embedded_symbol);

    var warm: usize = 0;
    while (warm < cfg.warmup) : (warm += 1) {
        try cuda.gated_delta_ssm_rows_ptrs_i8.runWithFunction(
            arg_pack,
            device,
            fn_info.function,
            &qkv_rows_dev,
            &state_ptrs_dev,
            &a_log_dev,
            &dt_bias_dev,
            &out_dev,
            @intCast(dims.n_qk_heads),
            @intCast(dims.n_v_heads),
            @intCast(dims.d_head),
            @intCast(rows),
            @intCast(row_stride),
            @intCast(beta_offset),
            @intCast(a_offset),
            @intCast(out_row_stride),
            @intCast(state_scales_offset),
        );
    }

    for (samples) |*sample| {
        try device.recordEvent(start_evt, stream);
        try cuda.gated_delta_ssm_rows_ptrs_i8.runWithFunction(
            arg_pack,
            device,
            fn_info.function,
            &qkv_rows_dev,
            &state_ptrs_dev,
            &a_log_dev,
            &dt_bias_dev,
            &out_dev,
            @intCast(dims.n_qk_heads),
            @intCast(dims.n_v_heads),
            @intCast(dims.d_head),
            @intCast(rows),
            @intCast(row_stride),
            @intCast(beta_offset),
            @intCast(a_offset),
            @intCast(out_row_stride),
            @intCast(state_scales_offset),
        );
        try device.recordEvent(stop_evt, stream);
        try device.synchronizeEvent(stop_evt);
        sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
    }

    return .{
        .name = scenarioName(.decode_gdelta_ssm_i8, quant),
        .quant = quant,
        .samples = samples,
        .flops_per_iter = 0,
        .bytes_per_iter = std.math.cast(u64, std.math.mul(u128, qkv_rows_count + out_count, @sizeOf(f32)) catch 0) orelse 0,
        .rows = rows,
        .in_dim = row_stride,
        .out_dim = out_row_stride,
    };
}

fn runDecodeGatedDeltaConvSiluPtrsScenario(
    allocator: std.mem.Allocator,
    quant: QuantKind,
    cfg: RunConfig,
    device: *cuda.Device,
    registry: *cuda.Registry,
    arg_pack: *cuda.ArgPack,
    stream: cuda.StreamHandle,
    start_evt: cuda.EventHandle,
    stop_evt: cuda.EventHandle,
    samples: []harness.Sample,
) !ScenarioResult {
    const dims = try decodeGatedDeltaDims(cfg.model_id, cfg.rows_override);
    const rows = dims.rows;
    const row_stride = dims.projLen();
    const conv_dim = dims.qkv_len;
    const state_elems_per_row = std.math.mul(usize, conv_dim, dims.d_conv) catch return error.InvalidArgument;
    const row_count = std.math.mul(usize, rows, row_stride) catch return error.InvalidArgument;

    const values_host = try allocator.alloc(f32, row_count);
    defer allocator.free(values_host);
    const weight_host = try allocator.alloc(f32, state_elems_per_row);
    defer allocator.free(weight_host);
    const bias_host = try allocator.alloc(f32, conv_dim);
    defer allocator.free(bias_host);
    const positions_host = try allocator.alloc(u32, rows);
    defer allocator.free(positions_host);
    fillInputPattern(values_host);
    fillF32ScalePattern(weight_host);
    fillF32ScalePattern(bias_host);
    for (positions_host, 0..) |*position, row| {
        position.* = @intCast(row % dims.d_conv);
    }

    var values_dev = try device.allocBuffer(row_count * @sizeOf(f32));
    defer values_dev.deinit(device);
    var weight_dev = try device.allocBuffer(state_elems_per_row * @sizeOf(f32));
    defer weight_dev.deinit(device);
    var bias_dev = try device.allocBuffer(conv_dim * @sizeOf(f32));
    defer bias_dev.deinit(device);
    var out_dev = try device.allocBuffer(row_count * @sizeOf(f32));
    defer out_dev.deinit(device);
    var state_dev = try device.allocBuffer(rows * state_elems_per_row * @sizeOf(f32));
    defer state_dev.deinit(device);
    var state_ptrs_dev = try device.allocBuffer(rows * @sizeOf(u64));
    defer state_ptrs_dev.deinit(device);
    var positions_dev = try device.allocBuffer(rows * @sizeOf(u32));
    defer positions_dev.deinit(device);

    try values_dev.upload(device, std.mem.sliceAsBytes(values_host));
    try weight_dev.upload(device, std.mem.sliceAsBytes(weight_host));
    try bias_dev.upload(device, std.mem.sliceAsBytes(bias_host));
    try positions_dev.upload(device, std.mem.sliceAsBytes(positions_host));

    const state_ptrs_host = try allocator.alloc(u64, rows);
    defer allocator.free(state_ptrs_host);
    for (state_ptrs_host, 0..) |*ptr, row| {
        ptr.* = state_dev.pointer + row * state_elems_per_row * @sizeOf(f32);
    }
    try state_ptrs_dev.upload(device, std.mem.sliceAsBytes(state_ptrs_host));

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(cuda.gated_delta_conv_silu_rows_ptrs.embedded_module);
    const fn_info = try registry.resolveFunction(cuda.gated_delta_conv_silu_rows_ptrs.op_name, cuda.gated_delta_conv_silu_rows_ptrs.embedded_symbol);

    var warm: usize = 0;
    while (warm < cfg.warmup) : (warm += 1) {
        try cuda.gated_delta_conv_silu_rows_ptrs.runWithFunction(
            arg_pack,
            device,
            fn_info.function,
            &values_dev,
            &state_ptrs_dev,
            &positions_dev,
            &weight_dev,
            &bias_dev,
            &out_dev,
            @intCast(conv_dim),
            @intCast(dims.d_conv),
            @intCast(rows),
            @intCast(row_stride),
        );
    }

    for (samples) |*sample| {
        try device.recordEvent(start_evt, stream);
        try cuda.gated_delta_conv_silu_rows_ptrs.runWithFunction(
            arg_pack,
            device,
            fn_info.function,
            &values_dev,
            &state_ptrs_dev,
            &positions_dev,
            &weight_dev,
            &bias_dev,
            &out_dev,
            @intCast(conv_dim),
            @intCast(dims.d_conv),
            @intCast(rows),
            @intCast(row_stride),
        );
        try device.recordEvent(stop_evt, stream);
        try device.synchronizeEvent(stop_evt);
        sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
    }

    return .{
        .name = scenarioName(.decode_gdelta_conv_silu_ptrs, quant),
        .quant = quant,
        .samples = samples,
        .flops_per_iter = 0,
        .bytes_per_iter = std.math.cast(u64, std.math.mul(u128, row_count * 2 + state_elems_per_row, @sizeOf(f32)) catch 0) orelse 0,
        .rows = rows,
        .in_dim = row_stride,
        .out_dim = conv_dim,
    };
}

fn runDecodeGatedDeltaNormGateRowsScenario(
    allocator: std.mem.Allocator,
    quant: QuantKind,
    cfg: RunConfig,
    device: *cuda.Device,
    registry: *cuda.Registry,
    arg_pack: *cuda.ArgPack,
    stream: cuda.StreamHandle,
    start_evt: cuda.EventHandle,
    stop_evt: cuda.EventHandle,
    samples: []harness.Sample,
) !ScenarioResult {
    const dims = try decodeGatedDeltaDims(cfg.model_id, cfg.rows_override);
    const rows = dims.rows;
    const rows_total = std.math.mul(usize, rows, dims.n_v_heads) catch return error.InvalidArgument;
    const input_count = std.math.mul(usize, rows_total, dims.d_head) catch return error.InvalidArgument;
    const gate_count = std.math.mul(usize, rows, dims.projLen()) catch return error.InvalidArgument;

    const input_host = try allocator.alloc(f32, input_count);
    defer allocator.free(input_host);
    const gate_host = try allocator.alloc(f32, gate_count);
    defer allocator.free(gate_host);
    const weight_host = try allocator.alloc(f32, dims.d_head);
    defer allocator.free(weight_host);
    fillInputPattern(input_host);
    fillInputPattern(gate_host);
    fillF32ScalePattern(weight_host);

    var input_dev = try device.allocBuffer(input_count * @sizeOf(f32));
    defer input_dev.deinit(device);
    var gate_dev = try device.allocBuffer(gate_count * @sizeOf(f32));
    defer gate_dev.deinit(device);
    var weight_dev = try device.allocBuffer(dims.d_head * @sizeOf(f32));
    defer weight_dev.deinit(device);
    var out_dev = try device.allocBuffer(input_count * @sizeOf(f32));
    defer out_dev.deinit(device);
    try input_dev.upload(device, std.mem.sliceAsBytes(input_host));
    try gate_dev.upload(device, std.mem.sliceAsBytes(gate_host));
    try weight_dev.upload(device, std.mem.sliceAsBytes(weight_host));

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(cuda.gated_delta_rmsnorm_silu_mul_rows.embedded_module);
    const fn_info = try registry.resolveFunction(cuda.gated_delta_rmsnorm_silu_mul_rows.op_name, cuda.gated_delta_rmsnorm_silu_mul_rows.embedded_symbol);

    var warm: usize = 0;
    while (warm < cfg.warmup) : (warm += 1) {
        try cuda.gated_delta_rmsnorm_silu_mul_rows.runWithFunction(
            arg_pack,
            device,
            fn_info.function,
            &input_dev,
            &gate_dev,
            &weight_dev,
            &out_dev,
            @intCast(rows_total),
            @intCast(dims.d_head),
            @intCast(dims.n_v_heads),
            @intCast(dims.projLen()),
            @intCast(dims.qkv_len),
            1.0e-6,
            0,
        );
    }

    for (samples) |*sample| {
        try device.recordEvent(start_evt, stream);
        try cuda.gated_delta_rmsnorm_silu_mul_rows.runWithFunction(
            arg_pack,
            device,
            fn_info.function,
            &input_dev,
            &gate_dev,
            &weight_dev,
            &out_dev,
            @intCast(rows_total),
            @intCast(dims.d_head),
            @intCast(dims.n_v_heads),
            @intCast(dims.projLen()),
            @intCast(dims.qkv_len),
            1.0e-6,
            0,
        );
        try device.recordEvent(stop_evt, stream);
        try device.synchronizeEvent(stop_evt);
        sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
    }

    return .{
        .name = scenarioName(.decode_gdelta_norm_gate_rows, quant),
        .quant = quant,
        .samples = samples,
        .flops_per_iter = 0,
        .bytes_per_iter = std.math.cast(u64, std.math.mul(u128, input_count * 2 + gate_count + dims.d_head, @sizeOf(f32)) catch 0) orelse 0,
        .rows = rows_total,
        .in_dim = dims.d_head,
        .out_dim = dims.d_head,
    };
}

fn runMatvecScenario(
    allocator: std.mem.Allocator,
    which: Scenario,
    quant: QuantKind,
    cfg: RunConfig,
    device: *cuda.Device,
    registry: *cuda.Registry,
    arg_pack: *cuda.ArgPack,
    stream: cuda.StreamHandle,
    start_evt: cuda.EventHandle,
    stop_evt: cuda.EventHandle,
    samples: []harness.Sample,
) !ScenarioResult {
    const dims = try modelRoleDims(cfg.model_id, which, cfg.rows_override);
    const rows = dims.tokens;
    const in_dim = dims.hidden;
    const out_dim = dims.out;
    if (rows == 0 or in_dim == 0 or out_dim == 0) return error.InvalidArgument;

    const input_count = std.math.mul(usize, rows, in_dim) catch return error.InvalidArgument;
    const output_count = std.math.mul(usize, rows, out_dim) catch return error.InvalidArgument;
    const input_host = try allocator.alloc(f32, input_count);
    defer allocator.free(input_host);
    fillInputPattern(input_host);

    var input_dev = try device.allocBuffer(input_count * @sizeOf(f32));
    defer input_dev.deinit(device);
    var output_dev = try device.allocBuffer(output_count * @sizeOf(f32));
    defer output_dev.deinit(device);
    try input_dev.upload(device, std.mem.sliceAsBytes(input_host));

    const batch_rows_u32: u32 = @intCast(rows);
    const in_dim_u32: u32 = @intCast(in_dim);
    const out_dim_u32: u32 = @intCast(out_dim);

    switch (quant) {
        .tq4 => {
            const group_size: u32 = 64;
            const words_per_row = in_dim / 8;
            const groups_per_row = in_dim / group_size;
            const packed_count = std.math.mul(usize, out_dim, words_per_row) catch return error.InvalidArgument;
            const sb_count = std.math.mul(usize, out_dim, groups_per_row) catch return error.InvalidArgument;
            const packed_host = try allocator.alloc(u32, packed_count);
            defer allocator.free(packed_host);
            const scales_host = try allocator.alloc(u16, sb_count);
            defer allocator.free(scales_host);
            const biases_host = try allocator.alloc(u16, sb_count);
            defer allocator.free(biases_host);
            fillU4PackedPattern(packed_host);
            fillScaleBiasPattern(scales_host, biases_host);

            var packed_dev = try device.allocBuffer(packed_count * @sizeOf(u32));
            defer packed_dev.deinit(device);
            var scales_dev = try device.allocBuffer(sb_count * @sizeOf(u16));
            defer scales_dev.deinit(device);
            var biases_dev = try device.allocBuffer(sb_count * @sizeOf(u16));
            defer biases_dev.deinit(device);
            try packed_dev.upload(device, std.mem.sliceAsBytes(packed_host));
            try scales_dev.upload(device, std.mem.sliceAsBytes(scales_host));
            try biases_dev.upload(device, std.mem.sliceAsBytes(biases_host));

            if (registry.embedded_module == null) try registry.loadEmbeddedModule(cuda.gaffine_u4_matvec.embedded_module);
            const base = try registry.resolveFunction(cuda.gaffine_u4_matvec.op_name, cuda.gaffine_u4_matvec.embedded_symbol);
            const tile8 = registry.resolveFunction(cuda.gaffine_u4_matvec.op_name_tile8, cuda.gaffine_u4_matvec.embedded_symbol_tile8) catch null;

            var warm: usize = 0;
            while (warm < cfg.warmup) : (warm += 1) {
                if (rows > 4 and tile8 != null) {
                    try cuda.gaffine_u4_matvec.runWithFunctionTile8(arg_pack, device, tile8.?.function, &input_dev, &packed_dev, &scales_dev, &biases_dev, &output_dev, in_dim_u32, out_dim_u32, group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, batch_rows_u32, 0);
                } else {
                    try cuda.gaffine_u4_matvec.runWithFunction(arg_pack, device, base.function, &input_dev, &packed_dev, &scales_dev, &biases_dev, &output_dev, in_dim_u32, out_dim_u32, group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, batch_rows_u32, 0);
                }
            }

            for (samples) |*sample| {
                try device.recordEvent(start_evt, stream);
                if (rows > 4 and tile8 != null) {
                    try cuda.gaffine_u4_matvec.runWithFunctionTile8(arg_pack, device, tile8.?.function, &input_dev, &packed_dev, &scales_dev, &biases_dev, &output_dev, in_dim_u32, out_dim_u32, group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, batch_rows_u32, 0);
                } else {
                    try cuda.gaffine_u4_matvec.runWithFunction(arg_pack, device, base.function, &input_dev, &packed_dev, &scales_dev, &biases_dev, &output_dev, in_dim_u32, out_dim_u32, group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, batch_rows_u32, 0);
                }
                try device.recordEvent(stop_evt, stream);
                try device.synchronizeEvent(stop_evt);
                sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
            }

            return .{
                .name = scenarioName(which, quant),
                .quant = quant,
                .samples = samples,
                .flops_per_iter = denseFlops(rows, in_dim, out_dim),
                .bytes_per_iter = tq4Bytes(rows, in_dim, out_dim, packed_count, sb_count),
                .rows = rows,
                .in_dim = in_dim,
                .out_dim = out_dim,
            };
        },
        .nvfp4 => {
            const scale_cols = in_dim / 16;
            const packed_cols = (in_dim + 1) / 2;
            const packed_count = std.math.mul(usize, out_dim, packed_cols) catch return error.InvalidArgument;
            const scale_count = std.math.mul(usize, out_dim, scale_cols) catch return error.InvalidArgument;
            const packed_host = try allocator.alloc(u8, packed_count);
            defer allocator.free(packed_host);
            const scales_host = try allocator.alloc(u8, scale_count);
            defer allocator.free(scales_host);
            fillNvfp4PackedPattern(packed_host);
            fillNvfp4ScalePattern(scales_host);

            var packed_dev = try device.allocBuffer(packed_count);
            defer packed_dev.deinit(device);
            var scales_dev = try device.allocBuffer(scale_count);
            defer scales_dev.deinit(device);
            try packed_dev.upload(device, packed_host);
            try scales_dev.upload(device, scales_host);

            if (registry.embedded_module == null) try registry.loadEmbeddedModule(cuda.nvfp4_matvec.embedded_module);
            const base = try registry.resolveFunction(cuda.nvfp4_matvec.op_name, cuda.nvfp4_matvec.embedded_symbol);
            const tile8 = registry.resolveFunction(cuda.nvfp4_matvec.op_name_tile8, cuda.nvfp4_matvec.embedded_symbol_tile8) catch null;
            const weight_global_scale: f32 = 1.0;
            const prefer_tile8 = rows > 4 or (rows == 1 and out_dim >= 2048);

            var warm: usize = 0;
            while (warm < cfg.warmup) : (warm += 1) {
                if (prefer_tile8 and tile8 != null) {
                    try cuda.nvfp4_matvec.runWithFunctionTile8(arg_pack, device, tile8.?.function, &input_dev, &packed_dev, &scales_dev, &output_dev, in_dim_u32, out_dim_u32, @intCast(scale_cols), 16, weight_global_scale, batch_rows_u32);
                } else {
                    try cuda.nvfp4_matvec.runWithFunction(arg_pack, device, base.function, &input_dev, &packed_dev, &scales_dev, &output_dev, in_dim_u32, out_dim_u32, @intCast(scale_cols), 16, weight_global_scale, batch_rows_u32);
                }
            }

            for (samples) |*sample| {
                try device.recordEvent(start_evt, stream);
                if (prefer_tile8 and tile8 != null) {
                    try cuda.nvfp4_matvec.runWithFunctionTile8(arg_pack, device, tile8.?.function, &input_dev, &packed_dev, &scales_dev, &output_dev, in_dim_u32, out_dim_u32, @intCast(scale_cols), 16, weight_global_scale, batch_rows_u32);
                } else {
                    try cuda.nvfp4_matvec.runWithFunction(arg_pack, device, base.function, &input_dev, &packed_dev, &scales_dev, &output_dev, in_dim_u32, out_dim_u32, @intCast(scale_cols), 16, weight_global_scale, batch_rows_u32);
                }
                try device.recordEvent(stop_evt, stream);
                try device.synchronizeEvent(stop_evt);
                sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
            }

            return .{
                .name = scenarioName(which, quant),
                .quant = quant,
                .samples = samples,
                .flops_per_iter = denseFlops(rows, in_dim, out_dim),
                .bytes_per_iter = nvfp4Bytes(rows, in_dim, out_dim, packed_count, scale_count),
                .rows = rows,
                .in_dim = in_dim,
                .out_dim = out_dim,
            };
        },
        .nvfp4_i8cache => {
            const scale_cols = in_dim / 16;
            var weights = try allocUploadNvfp4Weights(allocator, device, out_dim, in_dim, scale_cols);
            defer weights.deinit(device, allocator);

            if (registry.embedded_module == null) try registry.loadEmbeddedModule(cuda.nvfp4_matvec.embedded_module);
            const dequant_fn = try registry.resolveFunction("nvfp4_to_i8", "talu_nvfp4_to_i8");
            const i8_fn = try registry.resolveFunction("i8_matvec_f32", "talu_i8_matvec_f32");

            const i8_weight_bytes = std.math.mul(usize, out_dim, in_dim) catch return error.InvalidArgument;
            const row_scale_bytes = std.math.mul(usize, out_dim, @sizeOf(f32)) catch return error.InvalidArgument;
            var i8_weight_dev = try device.allocBuffer(i8_weight_bytes);
            defer i8_weight_dev.deinit(device);
            var row_scales_dev = try device.allocBuffer(row_scale_bytes);
            defer row_scales_dev.deinit(device);

            arg_pack.reset();
            try arg_pack.appendBufferPtr(&weights.packed_data);
            try arg_pack.appendBufferPtr(&weights.scales);
            try arg_pack.appendBufferPtr(&i8_weight_dev);
            try arg_pack.appendBufferPtr(&row_scales_dev);
            try arg_pack.appendScalar(u32, in_dim_u32);
            try arg_pack.appendScalar(u32, weights.scale_cols);
            try arg_pack.appendScalar(f32, weights.weight_global_scale);
            try cuda.launch.launchWithFamily(device, dequant_fn.function, .{
                .grid_x = @intCast(out_dim),
                .block_x = 256,
            }, arg_pack, .other);

            var warm: usize = 0;
            while (warm < cfg.warmup) : (warm += 1) {
                arg_pack.reset();
                try arg_pack.appendBufferPtr(&input_dev);
                try arg_pack.appendBufferPtr(&i8_weight_dev);
                try arg_pack.appendBufferPtr(&row_scales_dev);
                try arg_pack.appendBufferPtr(&output_dev);
                try arg_pack.appendScalar(u32, in_dim_u32);
                try arg_pack.appendScalar(u32, out_dim_u32);
                try arg_pack.appendScalar(u32, batch_rows_u32);
                try arg_pack.appendDevicePtr(0);
                try cuda.launch.launchWithFamily(device, i8_fn.function, .{
                    .grid_x = (out_dim_u32 + 3) / 4,
                    .grid_y = batch_rows_u32,
                    .block_x = 128,
                }, arg_pack, .matvec);
            }

            for (samples) |*sample| {
                try device.recordEvent(start_evt, stream);
                arg_pack.reset();
                try arg_pack.appendBufferPtr(&input_dev);
                try arg_pack.appendBufferPtr(&i8_weight_dev);
                try arg_pack.appendBufferPtr(&row_scales_dev);
                try arg_pack.appendBufferPtr(&output_dev);
                try arg_pack.appendScalar(u32, in_dim_u32);
                try arg_pack.appendScalar(u32, out_dim_u32);
                try arg_pack.appendScalar(u32, batch_rows_u32);
                try arg_pack.appendDevicePtr(0);
                try cuda.launch.launchWithFamily(device, i8_fn.function, .{
                    .grid_x = (out_dim_u32 + 3) / 4,
                    .grid_y = batch_rows_u32,
                    .block_x = 128,
                }, arg_pack, .matvec);
                try device.recordEvent(stop_evt, stream);
                try device.synchronizeEvent(stop_evt);
                sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
            }

            return .{
                .name = scenarioName(which, quant),
                .quant = quant,
                .samples = samples,
                .flops_per_iter = denseFlops(rows, in_dim, out_dim),
                .bytes_per_iter = i8Bytes(rows, in_dim, out_dim, i8_weight_bytes, out_dim),
                .rows = rows,
                .in_dim = in_dim,
                .out_dim = out_dim,
            };
        },
        .nvfp4_native => {
            if (which != .decode_lm_head and which != .decode_attn_out and which != .decode_attn_q and which != .decode_ffn_down and which != .prefill_attn_q and which != .prefill_ffn_down) {
                return error.InvalidArgument;
            }

            var weights = try allocUploadNvfp4Weights(allocator, device, out_dim, in_dim, in_dim / 16);
            defer weights.deinit(device, allocator);
            if (weights.scales_lt_buffer.size == 0) return error.InvalidArgument;

            if (registry.embedded_module == null) try registry.loadEmbeddedModule(cuda.nvfp4_matvec.embedded_module);
            const quant_fn = try registry.resolveFunction("quantize_f32_to_nvfp4", "talu_quantize_f32_to_nvfp4");
            var blas_lt = try cuda.BlasLt.init(device);
            defer blas_lt.deinit(device);

            const packed_in_cols = (in_dim + 1) / 2;
            const input_fp4_bytes = std.math.mul(usize, rows, packed_in_cols) catch return error.InvalidArgument;
            const input_scale_bytes = nvfp4LtScaleTensorSize(in_dim, rows);
            var input_fp4_dev = try device.allocBuffer(input_fp4_bytes);
            defer input_fp4_dev.deinit(device);
            var input_scales_dev = try device.allocBuffer(input_scale_bytes);
            defer input_scales_dev.deinit(device);

            var warm: usize = 0;
            while (warm < cfg.warmup) : (warm += 1) {
                try quantizeInputNvfp4Lt(arg_pack, device, quant_fn.function, &input_dev, &input_fp4_dev, &input_scales_dev, in_dim, rows);
                try blas_lt.matmulNvfp4(device, &weights.packed_data, &weights.scales_lt_buffer, &input_fp4_dev, &input_scales_dev, &output_dev, rows, out_dim, in_dim, 1.0 / weights.weight_global_scale);
            }

            for (samples) |*sample| {
                try device.recordEvent(start_evt, stream);
                try quantizeInputNvfp4Lt(arg_pack, device, quant_fn.function, &input_dev, &input_fp4_dev, &input_scales_dev, in_dim, rows);
                try blas_lt.matmulNvfp4(device, &weights.packed_data, &weights.scales_lt_buffer, &input_fp4_dev, &input_scales_dev, &output_dev, rows, out_dim, in_dim, 1.0 / weights.weight_global_scale);
                try device.recordEvent(stop_evt, stream);
                try device.synchronizeEvent(stop_evt);
                sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
            }

            return .{
                .name = scenarioName(which, quant),
                .quant = quant,
                .samples = samples,
                .flops_per_iter = denseFlops(rows, in_dim, out_dim),
                .bytes_per_iter = nvfp4Bytes(rows, in_dim, out_dim, weights.packed_data.size, weights.scales_lt_buffer.size + input_scale_bytes),
                .rows = rows,
                .in_dim = in_dim,
                .out_dim = out_dim,
            };
        },
    }
}

fn runDecodeAttentionI8PtrsScenario(
    allocator: std.mem.Allocator,
    quant: QuantKind,
    cfg: RunConfig,
    device: *cuda.Device,
    registry: *cuda.Registry,
    arg_pack: *cuda.ArgPack,
    stream: cuda.StreamHandle,
    start_evt: cuda.EventHandle,
    stop_evt: cuda.EventHandle,
    samples: []harness.Sample,
) !ScenarioResult {
    var attn = try initDecodeAttentionI8Buffers(allocator, device, cfg.model_id, cfg.seq_len_override);
    defer attn.deinit(device, allocator);
    const dims = attn.dims;
    const scale = attn.scale;

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(cuda.rope_rows_ptrs.embedded_module);
    const rope_fn = try registry.resolveFunction(cuda.rope_rows_ptrs.op_name, cuda.rope_rows_ptrs.embedded_symbol);
    const scores_fn = try registry.resolveFunction(cuda.attn_scores_heads_i8_kv_ptrs.op_name, cuda.attn_scores_heads_i8_kv_ptrs.embedded_symbol);
    const softmax_fn = try registry.resolveFunction(cuda.softmax_rows_dynamic_cols_ptrs.op_name, cuda.softmax_rows_dynamic_cols_ptrs.embedded_symbol);
    const weighted_sum_fn = try registry.resolveFunction(cuda.attn_weighted_sum_heads_i8_kv_ptrs.op_name, cuda.attn_weighted_sum_heads_i8_kv_ptrs.embedded_symbol);

    var warm: usize = 0;
    while (warm < cfg.warmup) : (warm += 1) {
        try runDecodeAttentionI8PtrsChain(
            arg_pack,
            device,
            rope_fn.function,
            scores_fn.function,
            softmax_fn.function,
            weighted_sum_fn.function,
            &attn.query_dev,
            &attn.scores_dev,
            &attn.out_dev,
            &attn.positions_dev,
            &attn.seq_lens_dev,
            &attn.key_ptrs_dev,
            &attn.value_ptrs_dev,
            &attn.k_scale_ptrs_dev,
            &attn.v_scale_ptrs_dev,
            dims,
            scale,
        );
    }

    for (samples) |*sample| {
        try device.recordEvent(start_evt, stream);
        try runDecodeAttentionI8PtrsChain(
            arg_pack,
            device,
            rope_fn.function,
            scores_fn.function,
            softmax_fn.function,
            weighted_sum_fn.function,
            &attn.query_dev,
            &attn.scores_dev,
            &attn.out_dev,
            &attn.positions_dev,
            &attn.seq_lens_dev,
            &attn.key_ptrs_dev,
            &attn.value_ptrs_dev,
            &attn.k_scale_ptrs_dev,
            &attn.v_scale_ptrs_dev,
            dims,
            scale,
        );
        try device.recordEvent(stop_evt, stream);
        try device.synchronizeEvent(stop_evt);
        sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
    }

    return .{
        .name = scenarioName(.decode_attn_i8_kv_ptrs, quant),
        .quant = quant,
        .samples = samples,
        .flops_per_iter = decodeAttentionI8Flops(dims),
        .bytes_per_iter = decodeAttentionI8Bytes(dims),
        .rows = dims.seq_len,
        .in_dim = dims.qDim(),
        .out_dim = dims.qDim(),
    };
}

const DecodeAttentionI8Buffers = struct {
    dims: DecodeAttentionDims,
    scale: f32,
    query_dev: cuda.Buffer,
    scores_dev: cuda.Buffer,
    out_dev: cuda.Buffer,
    key_dev: cuda.Buffer,
    value_dev: cuda.Buffer,
    k_scales_dev: cuda.Buffer,
    v_scales_dev: cuda.Buffer,
    key_ptrs_dev: cuda.Buffer,
    value_ptrs_dev: cuda.Buffer,
    k_scale_ptrs_dev: cuda.Buffer,
    v_scale_ptrs_dev: cuda.Buffer,
    seq_lens_dev: cuda.Buffer,
    positions_dev: cuda.Buffer,

    fn deinit(self: *DecodeAttentionI8Buffers, device: *cuda.Device, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.positions_dev.deinit(device);
        self.seq_lens_dev.deinit(device);
        self.v_scale_ptrs_dev.deinit(device);
        self.k_scale_ptrs_dev.deinit(device);
        self.value_ptrs_dev.deinit(device);
        self.key_ptrs_dev.deinit(device);
        self.v_scales_dev.deinit(device);
        self.k_scales_dev.deinit(device);
        self.value_dev.deinit(device);
        self.key_dev.deinit(device);
        self.out_dev.deinit(device);
        self.scores_dev.deinit(device);
        self.query_dev.deinit(device);
    }
};

fn initDecodeAttentionI8Buffers(
    allocator: std.mem.Allocator,
    device: *cuda.Device,
    model_id: []const u8,
    seq_len_override: ?usize,
) !DecodeAttentionI8Buffers {
    const dims = try decodeAttentionDims(model_id, seq_len_override);
    const batch_rows: usize = 1;
    const query_dim = dims.qDim();
    const row_stride = dims.kvRowStride();
    const scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(dims.head_dim)));
    const position: u32 = @intCast(dims.seq_len - 1);
    const seq_len_u32: u32 = @intCast(dims.seq_len);

    const query_host = try allocator.alloc(f32, batch_rows * query_dim);
    defer allocator.free(query_host);
    fillInputPattern(query_host);
    const key_host = try allocator.alloc(i8, dims.seq_len * row_stride);
    defer allocator.free(key_host);
    const value_host = try allocator.alloc(i8, dims.seq_len * row_stride);
    defer allocator.free(value_host);
    fillI8Pattern(key_host);
    fillI8Pattern(value_host);
    const k_scales_host = try allocator.alloc(f32, dims.seq_len * dims.n_kv_heads);
    defer allocator.free(k_scales_host);
    const v_scales_host = try allocator.alloc(f32, dims.seq_len * dims.n_kv_heads);
    defer allocator.free(v_scales_host);
    fillF32ScalePattern(k_scales_host);
    fillF32ScalePattern(v_scales_host);

    var query_dev = try device.allocBuffer(query_host.len * @sizeOf(f32));
    errdefer query_dev.deinit(device);
    var scores_dev = try device.allocBuffer(batchRowsHeads(batch_rows, dims.n_heads) * dims.seq_len * @sizeOf(f32));
    errdefer scores_dev.deinit(device);
    var out_dev = try device.allocBuffer(batchRowsHeads(batch_rows, dims.n_heads) * dims.head_dim * @sizeOf(f32));
    errdefer out_dev.deinit(device);
    var key_dev = try device.allocBuffer(key_host.len);
    errdefer key_dev.deinit(device);
    var value_dev = try device.allocBuffer(value_host.len);
    errdefer value_dev.deinit(device);
    var k_scales_dev = try device.allocBuffer(k_scales_host.len * @sizeOf(f32));
    errdefer k_scales_dev.deinit(device);
    var v_scales_dev = try device.allocBuffer(v_scales_host.len * @sizeOf(f32));
    errdefer v_scales_dev.deinit(device);
    try query_dev.upload(device, std.mem.sliceAsBytes(query_host));
    try key_dev.upload(device, std.mem.sliceAsBytes(key_host));
    try value_dev.upload(device, std.mem.sliceAsBytes(value_host));
    try k_scales_dev.upload(device, std.mem.sliceAsBytes(k_scales_host));
    try v_scales_dev.upload(device, std.mem.sliceAsBytes(v_scales_host));

    const key_ptrs_host = [_]u64{key_dev.pointer};
    const value_ptrs_host = [_]u64{value_dev.pointer};
    const k_scale_ptrs_host = [_]u64{k_scales_dev.pointer};
    const v_scale_ptrs_host = [_]u64{v_scales_dev.pointer};
    const seq_lens_host = [_]u32{seq_len_u32};
    const positions_host = [_]u32{position};

    var key_ptrs_dev = try device.allocBuffer(@sizeOf(u64));
    errdefer key_ptrs_dev.deinit(device);
    var value_ptrs_dev = try device.allocBuffer(@sizeOf(u64));
    errdefer value_ptrs_dev.deinit(device);
    var k_scale_ptrs_dev = try device.allocBuffer(@sizeOf(u64));
    errdefer k_scale_ptrs_dev.deinit(device);
    var v_scale_ptrs_dev = try device.allocBuffer(@sizeOf(u64));
    errdefer v_scale_ptrs_dev.deinit(device);
    var seq_lens_dev = try device.allocBuffer(@sizeOf(u32));
    errdefer seq_lens_dev.deinit(device);
    var positions_dev = try device.allocBuffer(@sizeOf(u32));
    errdefer positions_dev.deinit(device);
    try key_ptrs_dev.upload(device, std.mem.sliceAsBytes(&key_ptrs_host));
    try value_ptrs_dev.upload(device, std.mem.sliceAsBytes(&value_ptrs_host));
    try k_scale_ptrs_dev.upload(device, std.mem.sliceAsBytes(&k_scale_ptrs_host));
    try v_scale_ptrs_dev.upload(device, std.mem.sliceAsBytes(&v_scale_ptrs_host));
    try seq_lens_dev.upload(device, std.mem.sliceAsBytes(&seq_lens_host));
    try positions_dev.upload(device, std.mem.sliceAsBytes(&positions_host));

    return .{
        .dims = dims,
        .scale = scale,
        .query_dev = query_dev,
        .scores_dev = scores_dev,
        .out_dev = out_dev,
        .key_dev = key_dev,
        .value_dev = value_dev,
        .k_scales_dev = k_scales_dev,
        .v_scales_dev = v_scales_dev,
        .key_ptrs_dev = key_ptrs_dev,
        .value_ptrs_dev = value_ptrs_dev,
        .k_scale_ptrs_dev = k_scale_ptrs_dev,
        .v_scale_ptrs_dev = v_scale_ptrs_dev,
        .seq_lens_dev = seq_lens_dev,
        .positions_dev = positions_dev,
    };
}

const DecodeAttentionPtrsFunctions = struct {
    rope: cuda.module.Function,
    scores: cuda.module.Function,
    softmax: cuda.module.Function,
    weighted_sum: cuda.module.Function,
};

fn resolveDecodeAttentionPtrsFunctions(registry: *cuda.Registry) !DecodeAttentionPtrsFunctions {
    if (registry.embedded_module == null) try registry.loadEmbeddedModule(cuda.rope_rows_ptrs.embedded_module);
    return .{
        .rope = (try registry.resolveFunction(cuda.rope_rows_ptrs.op_name, cuda.rope_rows_ptrs.embedded_symbol)).function,
        .scores = (try registry.resolveFunction(cuda.attn_scores_heads_i8_kv_ptrs.op_name, cuda.attn_scores_heads_i8_kv_ptrs.embedded_symbol)).function,
        .softmax = (try registry.resolveFunction(cuda.softmax_rows_dynamic_cols_ptrs.op_name, cuda.softmax_rows_dynamic_cols_ptrs.embedded_symbol)).function,
        .weighted_sum = (try registry.resolveFunction(cuda.attn_weighted_sum_heads_i8_kv_ptrs.op_name, cuda.attn_weighted_sum_heads_i8_kv_ptrs.embedded_symbol)).function,
    };
}

fn runDecodeAttentionRopeScenario(
    allocator: std.mem.Allocator,
    quant: QuantKind,
    cfg: RunConfig,
    device: *cuda.Device,
    registry: *cuda.Registry,
    arg_pack: *cuda.ArgPack,
    stream: cuda.StreamHandle,
    start_evt: cuda.EventHandle,
    stop_evt: cuda.EventHandle,
    samples: []harness.Sample,
) !ScenarioResult {
    var attn = try initDecodeAttentionI8Buffers(allocator, device, cfg.model_id, cfg.seq_len_override);
    defer attn.deinit(device, allocator);
    const dims = attn.dims;
    const functions = try resolveDecodeAttentionPtrsFunctions(registry);

    var warm: usize = 0;
    while (warm < cfg.warmup) : (warm += 1) {
        try cuda.rope_rows_ptrs.runWithFunction(arg_pack, device, functions.rope, &attn.query_dev, &attn.positions_dev, 1, @intCast(dims.n_heads), @intCast(dims.head_dim), @intCast(dims.rope_dim), dims.rope_theta);
    }

    for (samples) |*sample| {
        try device.recordEvent(start_evt, stream);
        try cuda.rope_rows_ptrs.runWithFunction(arg_pack, device, functions.rope, &attn.query_dev, &attn.positions_dev, 1, @intCast(dims.n_heads), @intCast(dims.head_dim), @intCast(dims.rope_dim), dims.rope_theta);
        try device.recordEvent(stop_evt, stream);
        try device.synchronizeEvent(stop_evt);
        sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
    }

    return .{
        .name = scenarioName(.decode_attn_rope, quant),
        .quant = quant,
        .samples = samples,
        .flops_per_iter = denseFlops(1 * dims.n_heads, dims.rope_dim, 1),
        .bytes_per_iter = 0,
        .rows = dims.seq_len,
        .in_dim = dims.qDim(),
        .out_dim = dims.qDim(),
    };
}

fn runDecodeAttentionScoresScenario(
    allocator: std.mem.Allocator,
    quant: QuantKind,
    cfg: RunConfig,
    device: *cuda.Device,
    registry: *cuda.Registry,
    arg_pack: *cuda.ArgPack,
    stream: cuda.StreamHandle,
    start_evt: cuda.EventHandle,
    stop_evt: cuda.EventHandle,
    samples: []harness.Sample,
) !ScenarioResult {
    var attn = try initDecodeAttentionI8Buffers(allocator, device, cfg.model_id, cfg.seq_len_override);
    defer attn.deinit(device, allocator);
    const dims = attn.dims;
    const scale = attn.scale;
    const functions = try resolveDecodeAttentionPtrsFunctions(registry);

    var warm: usize = 0;
    while (warm < cfg.warmup) : (warm += 1) {
        try cuda.attn_scores_heads_i8_kv_ptrs.runWithFunction(arg_pack, device, functions.scores, &attn.scores_dev, &attn.query_dev, &attn.key_ptrs_dev, &attn.k_scale_ptrs_dev, &attn.seq_lens_dev, &attn.positions_dev, 1, @intCast(dims.n_heads), @intCast(dims.n_kv_heads), @intCast(dims.seq_len), @intCast(dims.kvRowStride()), @intCast(dims.kvGroups()), @intCast(dims.head_dim), scale, 0);
    }

    for (samples) |*sample| {
        try device.recordEvent(start_evt, stream);
        try cuda.attn_scores_heads_i8_kv_ptrs.runWithFunction(arg_pack, device, functions.scores, &attn.scores_dev, &attn.query_dev, &attn.key_ptrs_dev, &attn.k_scale_ptrs_dev, &attn.seq_lens_dev, &attn.positions_dev, 1, @intCast(dims.n_heads), @intCast(dims.n_kv_heads), @intCast(dims.seq_len), @intCast(dims.kvRowStride()), @intCast(dims.kvGroups()), @intCast(dims.head_dim), scale, 0);
        try device.recordEvent(stop_evt, stream);
        try device.synchronizeEvent(stop_evt);
        sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
    }

    return .{
        .name = scenarioName(.decode_attn_scores, quant),
        .quant = quant,
        .samples = samples,
        .flops_per_iter = denseFlops(1 * dims.n_heads, dims.head_dim, dims.seq_len),
        .bytes_per_iter = 0,
        .rows = dims.seq_len,
        .in_dim = dims.head_dim,
        .out_dim = dims.seq_len,
    };
}

fn runDecodeAttentionSoftmaxScenario(
    allocator: std.mem.Allocator,
    quant: QuantKind,
    cfg: RunConfig,
    device: *cuda.Device,
    registry: *cuda.Registry,
    arg_pack: *cuda.ArgPack,
    stream: cuda.StreamHandle,
    start_evt: cuda.EventHandle,
    stop_evt: cuda.EventHandle,
    samples: []harness.Sample,
) !ScenarioResult {
    var attn = try initDecodeAttentionI8Buffers(allocator, device, cfg.model_id, cfg.seq_len_override);
    defer attn.deinit(device, allocator);
    const dims = attn.dims;
    const functions = try resolveDecodeAttentionPtrsFunctions(registry);

    var warm: usize = 0;
    while (warm < cfg.warmup) : (warm += 1) {
        try cuda.softmax_rows_dynamic_cols_ptrs.runWithFunction(arg_pack, device, functions.softmax, &attn.scores_dev, &attn.seq_lens_dev, &attn.positions_dev, 1, @intCast(dims.n_heads), @intCast(dims.seq_len), 0);
    }

    for (samples) |*sample| {
        try device.recordEvent(start_evt, stream);
        try cuda.softmax_rows_dynamic_cols_ptrs.runWithFunction(arg_pack, device, functions.softmax, &attn.scores_dev, &attn.seq_lens_dev, &attn.positions_dev, 1, @intCast(dims.n_heads), @intCast(dims.seq_len), 0);
        try device.recordEvent(stop_evt, stream);
        try device.synchronizeEvent(stop_evt);
        sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
    }

    return .{
        .name = scenarioName(.decode_attn_softmax, quant),
        .quant = quant,
        .samples = samples,
        .flops_per_iter = 0,
        .bytes_per_iter = 0,
        .rows = dims.seq_len,
        .in_dim = dims.seq_len,
        .out_dim = dims.seq_len,
    };
}

fn runDecodeAttentionWeightedSumScenario(
    allocator: std.mem.Allocator,
    quant: QuantKind,
    cfg: RunConfig,
    device: *cuda.Device,
    registry: *cuda.Registry,
    arg_pack: *cuda.ArgPack,
    stream: cuda.StreamHandle,
    start_evt: cuda.EventHandle,
    stop_evt: cuda.EventHandle,
    samples: []harness.Sample,
) !ScenarioResult {
    var attn = try initDecodeAttentionI8Buffers(allocator, device, cfg.model_id, cfg.seq_len_override);
    defer attn.deinit(device, allocator);
    const dims = attn.dims;
    const functions = try resolveDecodeAttentionPtrsFunctions(registry);

    var warm: usize = 0;
    while (warm < cfg.warmup) : (warm += 1) {
        try cuda.attn_weighted_sum_heads_i8_kv_ptrs.runWithFunction(arg_pack, device, functions.weighted_sum, &attn.out_dev, &attn.scores_dev, &attn.value_ptrs_dev, &attn.v_scale_ptrs_dev, &attn.seq_lens_dev, &attn.positions_dev, 1, @intCast(dims.n_heads), @intCast(dims.n_kv_heads), @intCast(dims.seq_len), @intCast(dims.kvRowStride()), @intCast(dims.kvGroups()), @intCast(dims.head_dim), 0);
    }

    for (samples) |*sample| {
        try device.recordEvent(start_evt, stream);
        try cuda.attn_weighted_sum_heads_i8_kv_ptrs.runWithFunction(arg_pack, device, functions.weighted_sum, &attn.out_dev, &attn.scores_dev, &attn.value_ptrs_dev, &attn.v_scale_ptrs_dev, &attn.seq_lens_dev, &attn.positions_dev, 1, @intCast(dims.n_heads), @intCast(dims.n_kv_heads), @intCast(dims.seq_len), @intCast(dims.kvRowStride()), @intCast(dims.kvGroups()), @intCast(dims.head_dim), 0);
        try device.recordEvent(stop_evt, stream);
        try device.synchronizeEvent(stop_evt);
        sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
    }

    return .{
        .name = scenarioName(.decode_attn_weighted_sum, quant),
        .quant = quant,
        .samples = samples,
        .flops_per_iter = denseFlops(1 * dims.n_heads, dims.seq_len, dims.head_dim),
        .bytes_per_iter = 0,
        .rows = dims.seq_len,
        .in_dim = dims.seq_len,
        .out_dim = dims.head_dim,
    };
}

fn runDecodeAttentionI8PtrsChain(
    arg_pack: *cuda.ArgPack,
    device: *cuda.Device,
    rope_fn: cuda.module.Function,
    scores_fn: cuda.module.Function,
    softmax_fn: cuda.module.Function,
    weighted_sum_fn: cuda.module.Function,
    query_dev: *cuda.Buffer,
    scores_dev: *cuda.Buffer,
    out_dev: *cuda.Buffer,
    positions_dev: *const cuda.Buffer,
    seq_lens_dev: *const cuda.Buffer,
    key_ptrs_dev: *const cuda.Buffer,
    value_ptrs_dev: *const cuda.Buffer,
    k_scale_ptrs_dev: *const cuda.Buffer,
    v_scale_ptrs_dev: *const cuda.Buffer,
    dims: DecodeAttentionDims,
    scale: f32,
) !void {
    try cuda.rope_rows_ptrs.runWithFunction(
        arg_pack,
        device,
        rope_fn,
        query_dev,
        positions_dev,
        1,
        @intCast(dims.n_heads),
        @intCast(dims.head_dim),
        @intCast(dims.rope_dim),
        dims.rope_theta,
    );
    try cuda.attn_scores_heads_i8_kv_ptrs.runWithFunction(
        arg_pack,
        device,
        scores_fn,
        scores_dev,
        query_dev,
        key_ptrs_dev,
        k_scale_ptrs_dev,
        seq_lens_dev,
        positions_dev,
        1,
        @intCast(dims.n_heads),
        @intCast(dims.n_kv_heads),
        @intCast(dims.seq_len),
        @intCast(dims.kvRowStride()),
        @intCast(dims.kvGroups()),
        @intCast(dims.head_dim),
        scale,
        0,
    );
    try cuda.softmax_rows_dynamic_cols_ptrs.runWithFunction(
        arg_pack,
        device,
        softmax_fn,
        scores_dev,
        seq_lens_dev,
        positions_dev,
        1,
        @intCast(dims.n_heads),
        @intCast(dims.seq_len),
        0,
    );
    try cuda.attn_weighted_sum_heads_i8_kv_ptrs.runWithFunction(
        arg_pack,
        device,
        weighted_sum_fn,
        out_dev,
        scores_dev,
        value_ptrs_dev,
        v_scale_ptrs_dev,
        seq_lens_dev,
        positions_dev,
        1,
        @intCast(dims.n_heads),
        @intCast(dims.n_kv_heads),
        @intCast(dims.seq_len),
        @intCast(dims.kvRowStride()),
        @intCast(dims.kvGroups()),
        @intCast(dims.head_dim),
        0,
    );
}

fn runDecodeAttentionI8FusedScenario(
    allocator: std.mem.Allocator,
    quant: QuantKind,
    cfg: RunConfig,
    device: *cuda.Device,
    registry: *cuda.Registry,
    arg_pack: *cuda.ArgPack,
    stream: cuda.StreamHandle,
    start_evt: cuda.EventHandle,
    stop_evt: cuda.EventHandle,
    samples: []harness.Sample,
) !ScenarioResult {
    var attn = try initDecodeAttentionI8Buffers(allocator, device, cfg.model_id, cfg.seq_len_override);
    defer attn.deinit(device, allocator);
    const dims = attn.dims;
    const scale = attn.scale;

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(cuda.attn_fused_decode_heads_i8_kv_ptrs.embedded_module);
    const fused_fn = try registry.resolveFunction(cuda.attn_fused_decode_heads_i8_kv_ptrs.op_name, cuda.attn_fused_decode_heads_i8_kv_ptrs.embedded_symbol);

    var warm: usize = 0;
    while (warm < cfg.warmup) : (warm += 1) {
        try runDecodeAttentionI8FusedChain(
            arg_pack,
            device,
            fused_fn.function,
            &attn.out_dev,
            &attn.query_dev,
            &attn.key_ptrs_dev,
            &attn.value_ptrs_dev,
            &attn.k_scale_ptrs_dev,
            &attn.v_scale_ptrs_dev,
            &attn.seq_lens_dev,
            &attn.positions_dev,
            dims,
            scale,
        );
    }

    for (samples) |*sample| {
        try device.recordEvent(start_evt, stream);
        try runDecodeAttentionI8FusedChain(
            arg_pack,
            device,
            fused_fn.function,
            &attn.out_dev,
            &attn.query_dev,
            &attn.key_ptrs_dev,
            &attn.value_ptrs_dev,
            &attn.k_scale_ptrs_dev,
            &attn.v_scale_ptrs_dev,
            &attn.seq_lens_dev,
            &attn.positions_dev,
            dims,
            scale,
        );
        try device.recordEvent(stop_evt, stream);
        try device.synchronizeEvent(stop_evt);
        sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
    }

    return .{
        .name = scenarioName(.decode_attn_i8_kv_fused, quant),
        .quant = quant,
        .samples = samples,
        .flops_per_iter = decodeAttentionI8Flops(dims),
        .bytes_per_iter = decodeAttentionI8Bytes(dims),
        .rows = dims.seq_len,
        .in_dim = dims.qDim(),
        .out_dim = dims.qDim(),
    };
}

fn runDecodeAttentionI8FusedChain(
    arg_pack: *cuda.ArgPack,
    device: *cuda.Device,
    fused_fn: cuda.module.Function,
    out_dev: *cuda.Buffer,
    query_dev: *const cuda.Buffer,
    key_ptrs_dev: *const cuda.Buffer,
    value_ptrs_dev: *const cuda.Buffer,
    k_scale_ptrs_dev: *const cuda.Buffer,
    v_scale_ptrs_dev: *const cuda.Buffer,
    seq_lens_dev: *const cuda.Buffer,
    positions_dev: *const cuda.Buffer,
    dims: DecodeAttentionDims,
    scale: f32,
) !void {
    try cuda.attn_fused_decode_heads_i8_kv_ptrs.runWithFunction(
        arg_pack,
        device,
        fused_fn,
        out_dev,
        query_dev,
        key_ptrs_dev,
        value_ptrs_dev,
        k_scale_ptrs_dev,
        v_scale_ptrs_dev,
        seq_lens_dev,
        positions_dev,
        1,
        @intCast(dims.n_heads),
        @intCast(dims.n_kv_heads),
        @intCast(dims.kvRowStride()),
        @intCast(dims.kvGroups()),
        @intCast(dims.head_dim),
        scale,
        @intCast(dims.rope_dim),
        0,
        dims.rope_theta,
        null,
        0,
    );
}

fn runDecodeAttentionI8FlashScenario(
    allocator: std.mem.Allocator,
    quant: QuantKind,
    cfg: RunConfig,
    device: *cuda.Device,
    registry: *cuda.Registry,
    arg_pack: *cuda.ArgPack,
    stream: cuda.StreamHandle,
    start_evt: cuda.EventHandle,
    stop_evt: cuda.EventHandle,
    samples: []harness.Sample,
) !ScenarioResult {
    var attn = try initDecodeAttentionI8Buffers(allocator, device, cfg.model_id, cfg.seq_len_override);
    defer attn.deinit(device, allocator);
    const dims = attn.dims;
    const scale = attn.scale;

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(cuda.flash_decode.embedded_module);
    const flash_fn = try registry.resolveFunction(cuda.flash_decode.op_name_i8, cuda.flash_decode.symbol_i8);
    const reduce_fn = try registry.resolveFunction(cuda.flash_decode.op_name_reduce, cuda.flash_decode.symbol_reduce);

    const n_seq_chunks = cuda.flash_decode.computeSeqChunks(@intCast(dims.n_kv_heads), 1, @intCast(dims.seq_len), true);
    var partial_m: ?cuda.Buffer = null;
    var partial_s: ?cuda.Buffer = null;
    var partial_out: ?cuda.Buffer = null;
    defer {
        if (partial_out) |*buf| buf.deinit(device);
        if (partial_s) |*buf| buf.deinit(device);
        if (partial_m) |*buf| buf.deinit(device);
    }
    if (n_seq_chunks > 1) {
        const partial_entries = @as(usize, dims.n_heads) * @as(usize, n_seq_chunks);
        partial_m = try device.allocBuffer(partial_entries * @sizeOf(f32));
        partial_s = try device.allocBuffer(partial_entries * @sizeOf(f32));
        partial_out = try device.allocBuffer(partial_entries * @as(usize, dims.head_dim) * @sizeOf(f32));
    }

    var warm: usize = 0;
    while (warm < cfg.warmup) : (warm += 1) {
        try runDecodeAttentionI8FlashChain(
            arg_pack,
            device,
            flash_fn.function,
            reduce_fn.function,
            &attn.out_dev,
            &attn.query_dev,
            &attn.key_ptrs_dev,
            &attn.value_ptrs_dev,
            &attn.k_scale_ptrs_dev,
            &attn.v_scale_ptrs_dev,
            &attn.seq_lens_dev,
            &attn.positions_dev,
            dims,
            scale,
            n_seq_chunks,
            if (partial_m) |*buf| buf else null,
            if (partial_s) |*buf| buf else null,
            if (partial_out) |*buf| buf else null,
        );
    }

    for (samples) |*sample| {
        try device.recordEvent(start_evt, stream);
        try runDecodeAttentionI8FlashChain(
            arg_pack,
            device,
            flash_fn.function,
            reduce_fn.function,
            &attn.out_dev,
            &attn.query_dev,
            &attn.key_ptrs_dev,
            &attn.value_ptrs_dev,
            &attn.k_scale_ptrs_dev,
            &attn.v_scale_ptrs_dev,
            &attn.seq_lens_dev,
            &attn.positions_dev,
            dims,
            scale,
            n_seq_chunks,
            if (partial_m) |*buf| buf else null,
            if (partial_s) |*buf| buf else null,
            if (partial_out) |*buf| buf else null,
        );
        try device.recordEvent(stop_evt, stream);
        try device.synchronizeEvent(stop_evt);
        sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
    }

    return .{
        .name = scenarioName(.decode_attn_i8_flash, quant),
        .quant = quant,
        .samples = samples,
        .flops_per_iter = decodeAttentionI8Flops(dims),
        .bytes_per_iter = decodeAttentionI8Bytes(dims),
        .rows = dims.seq_len,
        .in_dim = dims.qDim(),
        .out_dim = dims.qDim(),
    };
}

fn runDecodeAttentionI8FlashChain(
    arg_pack: *cuda.ArgPack,
    device: *cuda.Device,
    flash_fn: cuda.module.Function,
    reduce_fn: cuda.module.Function,
    out_dev: *cuda.Buffer,
    query_dev: *const cuda.Buffer,
    key_ptrs_dev: *const cuda.Buffer,
    value_ptrs_dev: *const cuda.Buffer,
    k_scale_ptrs_dev: *const cuda.Buffer,
    v_scale_ptrs_dev: *const cuda.Buffer,
    seq_lens_dev: *const cuda.Buffer,
    positions_dev: *const cuda.Buffer,
    dims: DecodeAttentionDims,
    scale: f32,
    n_seq_chunks: u32,
    partial_m: ?*cuda.Buffer,
    partial_s: ?*cuda.Buffer,
    partial_out: ?*cuda.Buffer,
) !void {
    try cuda.flash_decode.runWithScales(
        arg_pack,
        device,
        flash_fn,
        out_dev,
        query_dev,
        key_ptrs_dev,
        value_ptrs_dev,
        k_scale_ptrs_dev,
        v_scale_ptrs_dev,
        seq_lens_dev,
        positions_dev,
        1,
        @intCast(dims.n_heads),
        @intCast(dims.n_kv_heads),
        @intCast(dims.kvRowStride()),
        @intCast(dims.kvGroups()),
        @intCast(dims.head_dim),
        scale,
        @intCast(dims.rope_dim),
        0,
        dims.rope_theta,
        null,
        0,
        n_seq_chunks,
        partial_m,
        partial_s,
        partial_out,
    );
    if (n_seq_chunks > 1) {
        try cuda.flash_decode.runReduce(
            arg_pack,
            device,
            reduce_fn,
            out_dev,
            partial_m.?,
            partial_s.?,
            partial_out.?,
            1,
            @intCast(dims.n_heads),
            @intCast(dims.head_dim),
            n_seq_chunks,
            null,
            0,
        );
    }
}

fn runDecodeTokenChainScenario(
    allocator: std.mem.Allocator,
    quant: QuantKind,
    cfg: RunConfig,
    device: *cuda.Device,
    registry: *cuda.Registry,
    arg_pack: *cuda.ArgPack,
    stream: cuda.StreamHandle,
    start_evt: cuda.EventHandle,
    stop_evt: cuda.EventHandle,
    samples: []harness.Sample,
) !ScenarioResult {
    const dims = try decodeTokenChainDims(cfg.model_id, cfg.seq_len_override, cfg.layers_override);
    if (dims.layers == 0) return error.InvalidArgument;

    const hidden_count = dims.attn_out.out;
    const q_count = dims.qkv.q_out;
    const k_count = dims.qkv.k_out;
    const v_count = dims.qkv.v_out;
    const ffn_count = dims.gate_up.out;
    const lm_count = dims.lm_head.out;
    const attn_scores_count = dims.attention.n_heads * dims.attention.seq_len;

    const hidden_host = try allocator.alloc(f32, hidden_count);
    defer allocator.free(hidden_host);
    fillInputPattern(hidden_host);

    var hidden_a = try device.allocBuffer(hidden_count * @sizeOf(f32));
    defer hidden_a.deinit(device);
    var hidden_b = try device.allocBuffer(hidden_count * @sizeOf(f32));
    defer hidden_b.deinit(device);
    var q_out = try device.allocBuffer(q_count * @sizeOf(f32));
    defer q_out.deinit(device);
    var k_out = try device.allocBuffer(k_count * @sizeOf(f32));
    defer k_out.deinit(device);
    var v_out = try device.allocBuffer(v_count * @sizeOf(f32));
    defer v_out.deinit(device);
    var attn_scores = try device.allocBuffer(attn_scores_count * @sizeOf(f32));
    defer attn_scores.deinit(device);
    var attn_ctx = try device.allocBuffer(q_count * @sizeOf(f32));
    defer attn_ctx.deinit(device);
    var ffn_out = try device.allocBuffer(ffn_count * @sizeOf(f32));
    defer ffn_out.deinit(device);
    var lm_out = try device.allocBuffer(lm_count * @sizeOf(f32));
    defer lm_out.deinit(device);
    try hidden_a.upload(device, std.mem.sliceAsBytes(hidden_host));
    try hidden_b.upload(device, std.mem.sliceAsBytes(hidden_host));

    const row_stride = dims.attention.kvRowStride();
    const kv_elems = dims.attention.seq_len * row_stride;
    const key_host = try allocator.alloc(i8, kv_elems);
    defer allocator.free(key_host);
    const value_host = try allocator.alloc(i8, kv_elems);
    defer allocator.free(value_host);
    fillI8Pattern(key_host);
    fillI8Pattern(value_host);
    const scale_elems = dims.attention.seq_len * dims.attention.n_kv_heads;
    const k_scales_host = try allocator.alloc(f32, scale_elems);
    defer allocator.free(k_scales_host);
    const v_scales_host = try allocator.alloc(f32, scale_elems);
    defer allocator.free(v_scales_host);
    fillF32ScalePattern(k_scales_host);
    fillF32ScalePattern(v_scales_host);

    var key_dev = try device.allocBuffer(key_host.len);
    defer key_dev.deinit(device);
    var value_dev = try device.allocBuffer(value_host.len);
    defer value_dev.deinit(device);
    var k_scales_dev = try device.allocBuffer(k_scales_host.len * @sizeOf(f32));
    defer k_scales_dev.deinit(device);
    var v_scales_dev = try device.allocBuffer(v_scales_host.len * @sizeOf(f32));
    defer v_scales_dev.deinit(device);
    try key_dev.upload(device, std.mem.sliceAsBytes(key_host));
    try value_dev.upload(device, std.mem.sliceAsBytes(value_host));
    try k_scales_dev.upload(device, std.mem.sliceAsBytes(k_scales_host));
    try v_scales_dev.upload(device, std.mem.sliceAsBytes(v_scales_host));

    const key_ptrs_host = [_]u64{key_dev.pointer};
    const value_ptrs_host = [_]u64{value_dev.pointer};
    const k_scale_ptrs_host = [_]u64{k_scales_dev.pointer};
    const v_scale_ptrs_host = [_]u64{v_scales_dev.pointer};
    const seq_lens_host = [_]u32{@intCast(dims.attention.seq_len)};
    const positions_host = [_]u32{@intCast(dims.attention.seq_len - 1)};

    var key_ptrs_dev = try device.allocBuffer(@sizeOf(u64));
    defer key_ptrs_dev.deinit(device);
    var value_ptrs_dev = try device.allocBuffer(@sizeOf(u64));
    defer value_ptrs_dev.deinit(device);
    var k_scale_ptrs_dev = try device.allocBuffer(@sizeOf(u64));
    defer k_scale_ptrs_dev.deinit(device);
    var v_scale_ptrs_dev = try device.allocBuffer(@sizeOf(u64));
    defer v_scale_ptrs_dev.deinit(device);
    var seq_lens_dev = try device.allocBuffer(@sizeOf(u32));
    defer seq_lens_dev.deinit(device);
    var positions_dev = try device.allocBuffer(@sizeOf(u32));
    defer positions_dev.deinit(device);
    try key_ptrs_dev.upload(device, std.mem.sliceAsBytes(&key_ptrs_host));
    try value_ptrs_dev.upload(device, std.mem.sliceAsBytes(&value_ptrs_host));
    try k_scale_ptrs_dev.upload(device, std.mem.sliceAsBytes(&k_scale_ptrs_host));
    try v_scale_ptrs_dev.upload(device, std.mem.sliceAsBytes(&v_scale_ptrs_host));
    try seq_lens_dev.upload(device, std.mem.sliceAsBytes(&seq_lens_host));
    try positions_dev.upload(device, std.mem.sliceAsBytes(&positions_host));

    if (registry.embedded_module == null) try registry.loadEmbeddedModule(cuda.nvfp4_matvec.embedded_module);
    const rope_fn = try registry.resolveFunction(cuda.rope_rows_ptrs.op_name, cuda.rope_rows_ptrs.embedded_symbol);
    const scores_fn = try registry.resolveFunction(cuda.attn_scores_heads_i8_kv_ptrs.op_name, cuda.attn_scores_heads_i8_kv_ptrs.embedded_symbol);
    const softmax_fn = try registry.resolveFunction(cuda.softmax_rows_dynamic_cols_ptrs.op_name, cuda.softmax_rows_dynamic_cols_ptrs.embedded_symbol);
    const weighted_sum_fn = try registry.resolveFunction(cuda.attn_weighted_sum_heads_i8_kv_ptrs.op_name, cuda.attn_weighted_sum_heads_i8_kv_ptrs.embedded_symbol);
    const attn_scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(dims.attention.head_dim)));

    switch (quant) {
        .tq4 => {
            var q_weights = try allocUploadU4Weights(allocator, device, dims.qkv.q_out * (dims.qkv.hidden / 8), dims.qkv.q_out * (dims.qkv.hidden / 64));
            defer q_weights.deinit(device, allocator);
            var k_weights = try allocUploadU4Weights(allocator, device, dims.qkv.k_out * (dims.qkv.hidden / 8), dims.qkv.k_out * (dims.qkv.hidden / 64));
            defer k_weights.deinit(device, allocator);
            var v_weights = try allocUploadU4Weights(allocator, device, dims.qkv.v_out * (dims.qkv.hidden / 8), dims.qkv.v_out * (dims.qkv.hidden / 64));
            defer v_weights.deinit(device, allocator);
            var attn_out_weights = try allocUploadU4Weights(allocator, device, dims.attn_out.out * (dims.attn_out.hidden / 8), dims.attn_out.out * (dims.attn_out.hidden / 64));
            defer attn_out_weights.deinit(device, allocator);
            var gate_weights = try allocUploadU4Weights(allocator, device, dims.gate_up.out * (dims.gate_up.hidden / 8), dims.gate_up.out * (dims.gate_up.hidden / 64));
            defer gate_weights.deinit(device, allocator);
            var up_weights = try allocUploadU4Weights(allocator, device, dims.gate_up.out * (dims.gate_up.hidden / 8), dims.gate_up.out * (dims.gate_up.hidden / 64));
            defer up_weights.deinit(device, allocator);
            var down_weights = try allocUploadU4Weights(allocator, device, dims.ffn_down.out * (dims.ffn_down.hidden / 8), dims.ffn_down.out * (dims.ffn_down.hidden / 64));
            defer down_weights.deinit(device, allocator);
            var lm_weights = try allocUploadU4Weights(allocator, device, dims.lm_head.out * (dims.lm_head.hidden / 8), dims.lm_head.out * (dims.lm_head.hidden / 64));
            defer lm_weights.deinit(device, allocator);

            const qkv_fn = try registry.resolveFunction(cuda.gaffine_u4_matvec_qkv.op_name, cuda.gaffine_u4_matvec_qkv.embedded_symbol);
            const attn_out_fn = try registry.resolveFunction(cuda.gaffine_u4_matvec.op_name, cuda.gaffine_u4_matvec.embedded_symbol);
            const gate_up_fn = try registry.resolveFunction(cuda.gaffine_u4_matvec_gate_up_silu.op_name, cuda.gaffine_u4_matvec_gate_up_silu.embedded_symbol);

            var warm: usize = 0;
            while (warm < cfg.warmup) : (warm += 1) {
                try runDecodeTokenChainTq4(
                    arg_pack, device, qkv_fn.function, attn_out_fn.function, gate_up_fn.function,
                    rope_fn.function, scores_fn.function, softmax_fn.function, weighted_sum_fn.function,
                    dims, &hidden_a, &hidden_b, &q_out, &k_out, &v_out, &attn_scores, &attn_ctx, &ffn_out, &lm_out,
                    &positions_dev, &seq_lens_dev, &key_ptrs_dev, &value_ptrs_dev, &k_scale_ptrs_dev, &v_scale_ptrs_dev, attn_scale,
                    &q_weights, &k_weights, &v_weights, &attn_out_weights, &gate_weights, &up_weights, &down_weights, &lm_weights,
                );
            }

            for (samples) |*sample| {
                try device.recordEvent(start_evt, stream);
                try runDecodeTokenChainTq4(
                    arg_pack, device, qkv_fn.function, attn_out_fn.function, gate_up_fn.function,
                    rope_fn.function, scores_fn.function, softmax_fn.function, weighted_sum_fn.function,
                    dims, &hidden_a, &hidden_b, &q_out, &k_out, &v_out, &attn_scores, &attn_ctx, &ffn_out, &lm_out,
                    &positions_dev, &seq_lens_dev, &key_ptrs_dev, &value_ptrs_dev, &k_scale_ptrs_dev, &v_scale_ptrs_dev, attn_scale,
                    &q_weights, &k_weights, &v_weights, &attn_out_weights, &gate_weights, &up_weights, &down_weights, &lm_weights,
                );
                try device.recordEvent(stop_evt, stream);
                try device.synchronizeEvent(stop_evt);
                sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
            }
        },
        .nvfp4 => {
            var q_weights = try allocUploadNvfp4Weights(allocator, device, dims.qkv.q_out, dims.qkv.hidden, dims.qkv.hidden / 16);
            defer q_weights.deinit(device, allocator);
            var k_weights = try allocUploadNvfp4Weights(allocator, device, dims.qkv.k_out, dims.qkv.hidden, dims.qkv.hidden / 16);
            defer k_weights.deinit(device, allocator);
            var v_weights = try allocUploadNvfp4Weights(allocator, device, dims.qkv.v_out, dims.qkv.hidden, dims.qkv.hidden / 16);
            defer v_weights.deinit(device, allocator);
            var attn_out_weights = try allocUploadNvfp4Weights(allocator, device, dims.attn_out.out, dims.attn_out.hidden, dims.attn_out.hidden / 16);
            defer attn_out_weights.deinit(device, allocator);
            var gate_weights = try allocUploadNvfp4Weights(allocator, device, dims.gate_up.out, dims.gate_up.hidden, dims.gate_up.hidden / 16);
            defer gate_weights.deinit(device, allocator);
            var up_weights = try allocUploadNvfp4Weights(allocator, device, dims.gate_up.out, dims.gate_up.hidden, dims.gate_up.hidden / 16);
            defer up_weights.deinit(device, allocator);
            var down_weights = try allocUploadNvfp4Weights(allocator, device, dims.ffn_down.out, dims.ffn_down.hidden, dims.ffn_down.hidden / 16);
            defer down_weights.deinit(device, allocator);
            var lm_weights = try allocUploadNvfp4Weights(allocator, device, dims.lm_head.out, dims.lm_head.hidden, dims.lm_head.hidden / 16);
            defer lm_weights.deinit(device, allocator);

            const qkv_fn = try registry.resolveFunction("nvfp4_matvec_qkv_f32_tile8", "talu_nvfp4_matvec_qkv_f32_tile8");
            const attn_out_fn = try registry.resolveFunction(cuda.nvfp4_matvec.op_name_tile8, cuda.nvfp4_matvec.embedded_symbol_tile8);
            const gate_up_fn = try registry.resolveFunction("nvfp4_matvec_gate_up_silu_f32_tile8", "talu_nvfp4_matvec_gate_up_silu_f32_tile8");

            var warm: usize = 0;
            while (warm < cfg.warmup) : (warm += 1) {
                try runDecodeTokenChainNvfp4(
                    arg_pack, device, qkv_fn.function, attn_out_fn.function, gate_up_fn.function,
                    rope_fn.function, scores_fn.function, softmax_fn.function, weighted_sum_fn.function,
                    dims, &hidden_a, &hidden_b, &q_out, &k_out, &v_out, &attn_scores, &attn_ctx, &ffn_out, &lm_out,
                    &positions_dev, &seq_lens_dev, &key_ptrs_dev, &value_ptrs_dev, &k_scale_ptrs_dev, &v_scale_ptrs_dev, attn_scale,
                    &q_weights, &k_weights, &v_weights, &attn_out_weights, &gate_weights, &up_weights, &down_weights, &lm_weights,
                );
            }

            for (samples) |*sample| {
                try device.recordEvent(start_evt, stream);
                try runDecodeTokenChainNvfp4(
                    arg_pack, device, qkv_fn.function, attn_out_fn.function, gate_up_fn.function,
                    rope_fn.function, scores_fn.function, softmax_fn.function, weighted_sum_fn.function,
                    dims, &hidden_a, &hidden_b, &q_out, &k_out, &v_out, &attn_scores, &attn_ctx, &ffn_out, &lm_out,
                    &positions_dev, &seq_lens_dev, &key_ptrs_dev, &value_ptrs_dev, &k_scale_ptrs_dev, &v_scale_ptrs_dev, attn_scale,
                    &q_weights, &k_weights, &v_weights, &attn_out_weights, &gate_weights, &up_weights, &down_weights, &lm_weights,
                );
                try device.recordEvent(stop_evt, stream);
                try device.synchronizeEvent(stop_evt);
                sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
            }
        },
        .nvfp4_i8cache => {
            var q_weights = try allocUploadNvfp4Weights(allocator, device, dims.qkv.q_out, dims.qkv.hidden, dims.qkv.hidden / 16);
            defer q_weights.deinit(device, allocator);
            var k_weights = try allocUploadNvfp4Weights(allocator, device, dims.qkv.k_out, dims.qkv.hidden, dims.qkv.hidden / 16);
            defer k_weights.deinit(device, allocator);
            var v_weights = try allocUploadNvfp4Weights(allocator, device, dims.qkv.v_out, dims.qkv.hidden, dims.qkv.hidden / 16);
            defer v_weights.deinit(device, allocator);
            var attn_out_weights = try allocUploadNvfp4Weights(allocator, device, dims.attn_out.out, dims.attn_out.hidden, dims.attn_out.hidden / 16);
            defer attn_out_weights.deinit(device, allocator);
            var gate_weights = try allocUploadNvfp4Weights(allocator, device, dims.gate_up.out, dims.gate_up.hidden, dims.gate_up.hidden / 16);
            defer gate_weights.deinit(device, allocator);
            var up_weights = try allocUploadNvfp4Weights(allocator, device, dims.gate_up.out, dims.gate_up.hidden, dims.gate_up.hidden / 16);
            defer up_weights.deinit(device, allocator);
            var down_weights = try allocUploadNvfp4Weights(allocator, device, dims.ffn_down.out, dims.ffn_down.hidden, dims.ffn_down.hidden / 16);
            defer down_weights.deinit(device, allocator);
            var lm_weights = try allocUploadNvfp4Weights(allocator, device, dims.lm_head.out, dims.lm_head.hidden, dims.lm_head.hidden / 16);
            defer lm_weights.deinit(device, allocator);

            const qkv_fn = try registry.resolveFunction("nvfp4_matvec_qkv_f32_tile8", "talu_nvfp4_matvec_qkv_f32_tile8");
            const raw_matvec_fn = try registry.resolveFunction(cuda.nvfp4_matvec.op_name_tile8, cuda.nvfp4_matvec.embedded_symbol_tile8);
            const gate_up_fn = try registry.resolveFunction("nvfp4_matvec_gate_up_silu_f32_tile8", "talu_nvfp4_matvec_gate_up_silu_f32_tile8");
            const i8_matvec_fn = try registry.resolveFunction("i8_matvec_f32", "talu_i8_matvec_f32");
            const dequant_fn = try registry.resolveFunction("nvfp4_to_i8", "talu_nvfp4_to_i8");

            var attn_out_cached = try allocDequantizedI8Weights(arg_pack, device, dequant_fn.function, &attn_out_weights, dims.attn_out.out, dims.attn_out.hidden);
            defer attn_out_cached.deinit(device);
            var down_cached = try allocDequantizedI8Weights(arg_pack, device, dequant_fn.function, &down_weights, dims.ffn_down.out, dims.ffn_down.hidden);
            defer down_cached.deinit(device);

            var warm: usize = 0;
            while (warm < cfg.warmup) : (warm += 1) {
                try runDecodeTokenChainNvfp4Cached(
                    arg_pack, device, qkv_fn.function, raw_matvec_fn.function, gate_up_fn.function, i8_matvec_fn.function,
                    rope_fn.function, scores_fn.function, softmax_fn.function, weighted_sum_fn.function,
                    dims, &hidden_a, &hidden_b, &q_out, &k_out, &v_out, &attn_scores, &attn_ctx, &ffn_out, &lm_out,
                    &positions_dev, &seq_lens_dev, &key_ptrs_dev, &value_ptrs_dev, &k_scale_ptrs_dev, &v_scale_ptrs_dev, attn_scale,
                    &q_weights, &k_weights, &v_weights, &attn_out_cached, &gate_weights, &up_weights, &down_cached, &lm_weights,
                );
            }

            for (samples) |*sample| {
                try device.recordEvent(start_evt, stream);
                try runDecodeTokenChainNvfp4Cached(
                    arg_pack, device, qkv_fn.function, raw_matvec_fn.function, gate_up_fn.function, i8_matvec_fn.function,
                    rope_fn.function, scores_fn.function, softmax_fn.function, weighted_sum_fn.function,
                    dims, &hidden_a, &hidden_b, &q_out, &k_out, &v_out, &attn_scores, &attn_ctx, &ffn_out, &lm_out,
                    &positions_dev, &seq_lens_dev, &key_ptrs_dev, &value_ptrs_dev, &k_scale_ptrs_dev, &v_scale_ptrs_dev, attn_scale,
                    &q_weights, &k_weights, &v_weights, &attn_out_cached, &gate_weights, &up_weights, &down_cached, &lm_weights,
                );
                try device.recordEvent(stop_evt, stream);
                try device.synchronizeEvent(stop_evt);
                sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
            }
        },
        .nvfp4_native => return error.InvalidArgument,
    }

    return .{
        .name = scenarioName(.decode_token_chain, quant),
        .quant = quant,
        .samples = samples,
        .flops_per_iter = decodeTokenChainFlops(dims),
        .bytes_per_iter = decodeTokenChainBytes(dims),
        .rows = dims.layers,
        .in_dim = dims.qkv.hidden,
        .out_dim = dims.lm_head.out,
    };
}

fn runQkvScenario(
    allocator: std.mem.Allocator,
    which: Scenario,
    quant: QuantKind,
    cfg: RunConfig,
    device: *cuda.Device,
    registry: *cuda.Registry,
    arg_pack: *cuda.ArgPack,
    stream: cuda.StreamHandle,
    start_evt: cuda.EventHandle,
    stop_evt: cuda.EventHandle,
    samples: []harness.Sample,
) !ScenarioResult {
    const dims = try modelQkvDims(cfg.model_id, which, cfg.rows_override);
    const rows = dims.rows;
    const in_dim = dims.hidden;
    const q_out = dims.q_out;
    const k_out = dims.k_out;
    const v_out = dims.v_out;
    const total_out = dims.totalOut();
    if (rows == 0 or in_dim == 0 or total_out == 0) return error.InvalidArgument;

    const input_count = std.math.mul(usize, rows, in_dim) catch return error.InvalidArgument;
    const q_output_count = std.math.mul(usize, rows, q_out) catch return error.InvalidArgument;
    const k_output_count = std.math.mul(usize, rows, k_out) catch return error.InvalidArgument;
    const v_output_count = std.math.mul(usize, rows, v_out) catch return error.InvalidArgument;
    const input_host = try allocator.alloc(f32, input_count);
    defer allocator.free(input_host);
    fillInputPattern(input_host);

    var input_dev = try device.allocBuffer(input_count * @sizeOf(f32));
    defer input_dev.deinit(device);
    var q_out_dev = try device.allocBuffer(q_output_count * @sizeOf(f32));
    defer q_out_dev.deinit(device);
    var k_out_dev = try device.allocBuffer(k_output_count * @sizeOf(f32));
    defer k_out_dev.deinit(device);
    var v_out_dev = try device.allocBuffer(v_output_count * @sizeOf(f32));
    defer v_out_dev.deinit(device);
    try input_dev.upload(device, std.mem.sliceAsBytes(input_host));

    switch (quant) {
        .tq4 => {
            const group_size: u32 = 64;
            const words_per_row = in_dim / 8;
            const groups_per_row = in_dim / group_size;
            const q_packed_count = std.math.mul(usize, q_out, words_per_row) catch return error.InvalidArgument;
            const k_packed_count = std.math.mul(usize, k_out, words_per_row) catch return error.InvalidArgument;
            const v_packed_count = std.math.mul(usize, v_out, words_per_row) catch return error.InvalidArgument;
            const q_sb_count = std.math.mul(usize, q_out, groups_per_row) catch return error.InvalidArgument;
            const k_sb_count = std.math.mul(usize, k_out, groups_per_row) catch return error.InvalidArgument;
            const v_sb_count = std.math.mul(usize, v_out, groups_per_row) catch return error.InvalidArgument;

            var q_weights = try allocUploadU4Weights(allocator, device, q_packed_count, q_sb_count);
            defer q_weights.deinit(device, allocator);
            var k_weights = try allocUploadU4Weights(allocator, device, k_packed_count, k_sb_count);
            defer k_weights.deinit(device, allocator);
            var v_weights = try allocUploadU4Weights(allocator, device, v_packed_count, v_sb_count);
            defer v_weights.deinit(device, allocator);

            if (registry.embedded_module == null) try registry.loadEmbeddedModule(cuda.gaffine_u4_matvec_qkv.embedded_module);
            const base = try registry.resolveFunction(cuda.gaffine_u4_matvec_qkv.op_name, cuda.gaffine_u4_matvec_qkv.embedded_symbol);
            const tile8 = registry.resolveFunction(cuda.gaffine_u4_matvec_qkv.op_name_tile8, cuda.gaffine_u4_matvec_qkv.embedded_symbol_tile8) catch null;
            const batch_rows_u32: u32 = @intCast(rows);
            const in_dim_u32: u32 = @intCast(in_dim);
            const q_packed = &q_weights.packed_data;
            const q_scales = &q_weights.scales;
            const q_biases = &q_weights.biases;
            const k_packed = &k_weights.packed_data;
            const k_scales = &k_weights.scales;
            const k_biases = &k_weights.biases;
            const v_packed = &v_weights.packed_data;
            const v_scales = &v_weights.scales;
            const v_biases = &v_weights.biases;

            var warm: usize = 0;
            while (warm < cfg.warmup) : (warm += 1) {
                if (rows > 4 and tile8 != null) {
                    try cuda.gaffine_u4_matvec_qkv.runWithFunctionTile8(arg_pack, device, tile8.?.function, &input_dev, q_packed, q_scales, q_biases, &q_out_dev, @intCast(q_out), group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, k_packed, k_scales, k_biases, &k_out_dev, @intCast(k_out), group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, v_packed, v_scales, v_biases, &v_out_dev, @intCast(v_out), group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, in_dim_u32, batch_rows_u32);
                } else {
                    try cuda.gaffine_u4_matvec_qkv.runWithFunction(arg_pack, device, base.function, &input_dev, q_packed, q_scales, q_biases, &q_out_dev, @intCast(q_out), group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, k_packed, k_scales, k_biases, &k_out_dev, @intCast(k_out), group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, v_packed, v_scales, v_biases, &v_out_dev, @intCast(v_out), group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, in_dim_u32, batch_rows_u32);
                }
            }

            for (samples) |*sample| {
                try device.recordEvent(start_evt, stream);
                if (rows > 4 and tile8 != null) {
                    try cuda.gaffine_u4_matvec_qkv.runWithFunctionTile8(arg_pack, device, tile8.?.function, &input_dev, q_packed, q_scales, q_biases, &q_out_dev, @intCast(q_out), group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, k_packed, k_scales, k_biases, &k_out_dev, @intCast(k_out), group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, v_packed, v_scales, v_biases, &v_out_dev, @intCast(v_out), group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, in_dim_u32, batch_rows_u32);
                } else {
                    try cuda.gaffine_u4_matvec_qkv.runWithFunction(arg_pack, device, base.function, &input_dev, q_packed, q_scales, q_biases, &q_out_dev, @intCast(q_out), group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, k_packed, k_scales, k_biases, &k_out_dev, @intCast(k_out), group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, v_packed, v_scales, v_biases, &v_out_dev, @intCast(v_out), group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, in_dim_u32, batch_rows_u32);
                }
                try device.recordEvent(stop_evt, stream);
                try device.synchronizeEvent(stop_evt);
                sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
            }

            return .{
                .name = scenarioName(which, quant),
                .quant = quant,
                .samples = samples,
                .flops_per_iter = denseFlops(rows, in_dim, total_out),
                .bytes_per_iter = qkvTq4Bytes(rows, in_dim, q_out, k_out, v_out, q_packed_count + k_packed_count + v_packed_count, q_sb_count + k_sb_count + v_sb_count),
                .rows = rows,
                .in_dim = in_dim,
                .out_dim = total_out,
            };
        },
        .nvfp4 => {
            const q_scale_cols = in_dim / 16;
            const k_scale_cols = in_dim / 16;
            const v_scale_cols = in_dim / 16;
            var q_weights = try allocUploadNvfp4Weights(allocator, device, q_out, in_dim, q_scale_cols);
            defer q_weights.deinit(device, allocator);
            var k_weights = try allocUploadNvfp4Weights(allocator, device, k_out, in_dim, k_scale_cols);
            defer k_weights.deinit(device, allocator);
            var v_weights = try allocUploadNvfp4Weights(allocator, device, v_out, in_dim, v_scale_cols);
            defer v_weights.deinit(device, allocator);

            if (registry.embedded_module == null) try registry.loadEmbeddedModule(cuda.nvfp4_matvec.embedded_module);
            const base = try registry.resolveFunction("nvfp4_matvec_qkv_f32", "talu_nvfp4_matvec_qkv_f32");
            const tile8 = registry.resolveFunction("nvfp4_matvec_qkv_f32_tile8", "talu_nvfp4_matvec_qkv_f32_tile8") catch null;
            const batch_rows_u32: u32 = @intCast(rows);
            const in_dim_u32: u32 = @intCast(in_dim);
            const total_out_u32: u32 = @intCast(total_out);
            const prefer_tile8 = rows > 4 or (rows == 1 and total_out >= 3072);

            var warm: usize = 0;
            while (warm < cfg.warmup) : (warm += 1) {
                try runNvfp4QkvLaunch(arg_pack, device, if (prefer_tile8 and tile8 != null) tile8.?.function else base.function, &input_dev, &q_weights, &q_out_dev, &k_weights, &k_out_dev, &v_weights, &v_out_dev, @intCast(q_out), @intCast(k_out), @intCast(v_out), in_dim_u32, batch_rows_u32, if (prefer_tile8 and tile8 != null) 8 else 4, total_out_u32);
            }

            for (samples) |*sample| {
                try device.recordEvent(start_evt, stream);
                try runNvfp4QkvLaunch(arg_pack, device, if (prefer_tile8 and tile8 != null) tile8.?.function else base.function, &input_dev, &q_weights, &q_out_dev, &k_weights, &k_out_dev, &v_weights, &v_out_dev, @intCast(q_out), @intCast(k_out), @intCast(v_out), in_dim_u32, batch_rows_u32, if (prefer_tile8 and tile8 != null) 8 else 4, total_out_u32);
                try device.recordEvent(stop_evt, stream);
                try device.synchronizeEvent(stop_evt);
                sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
            }

            const packed_count_total = q_weights.packed_data.size + k_weights.packed_data.size + v_weights.packed_data.size;
            const scale_count_total = q_weights.scales.size + k_weights.scales.size + v_weights.scales.size;
            return .{
                .name = scenarioName(which, quant),
                .quant = quant,
                .samples = samples,
                .flops_per_iter = denseFlops(rows, in_dim, total_out),
                .bytes_per_iter = qkvNvfp4Bytes(rows, in_dim, q_out, k_out, v_out, packed_count_total, scale_count_total),
                .rows = rows,
                .in_dim = in_dim,
                .out_dim = total_out,
            };
        },
        .nvfp4_i8cache, .nvfp4_native => return error.InvalidArgument,
    }
}

fn runGateUpSiluScenario(
    allocator: std.mem.Allocator,
    which: Scenario,
    quant: QuantKind,
    cfg: RunConfig,
    device: *cuda.Device,
    registry: *cuda.Registry,
    arg_pack: *cuda.ArgPack,
    stream: cuda.StreamHandle,
    start_evt: cuda.EventHandle,
    stop_evt: cuda.EventHandle,
    samples: []harness.Sample,
) !ScenarioResult {
    const dims = try modelGateUpDims(cfg.model_id, which, cfg.rows_override);
    const rows = dims.rows;
    const in_dim = dims.hidden;
    const out_dim = dims.out;
    if (rows == 0 or in_dim == 0 or out_dim == 0) return error.InvalidArgument;

    const input_count = std.math.mul(usize, rows, in_dim) catch return error.InvalidArgument;
    const output_count = std.math.mul(usize, rows, out_dim) catch return error.InvalidArgument;
    const input_host = try allocator.alloc(f32, input_count);
    defer allocator.free(input_host);
    fillInputPattern(input_host);

    var input_dev = try device.allocBuffer(input_count * @sizeOf(f32));
    defer input_dev.deinit(device);
    var output_dev = try device.allocBuffer(output_count * @sizeOf(f32));
    defer output_dev.deinit(device);
    try input_dev.upload(device, std.mem.sliceAsBytes(input_host));

    switch (quant) {
        .tq4 => {
            const group_size: u32 = 64;
            const words_per_row = in_dim / 8;
            const groups_per_row = in_dim / group_size;
            const packed_count = std.math.mul(usize, out_dim, words_per_row) catch return error.InvalidArgument;
            const sb_count = std.math.mul(usize, out_dim, groups_per_row) catch return error.InvalidArgument;
            var gate_weights = try allocUploadU4Weights(allocator, device, packed_count, sb_count);
            defer gate_weights.deinit(device, allocator);
            var up_weights = try allocUploadU4Weights(allocator, device, packed_count, sb_count);
            defer up_weights.deinit(device, allocator);

            if (registry.embedded_module == null) try registry.loadEmbeddedModule(cuda.gaffine_u4_matvec_gate_up_silu.embedded_module);
            const base = try registry.resolveFunction(cuda.gaffine_u4_matvec_gate_up_silu.op_name, cuda.gaffine_u4_matvec_gate_up_silu.embedded_symbol);
            const tile8 = registry.resolveFunction(cuda.gaffine_u4_matvec_gate_up_silu.op_name_tile8, cuda.gaffine_u4_matvec_gate_up_silu.embedded_symbol_tile8) catch null;
            const batch_rows_u32: u32 = @intCast(rows);
            const in_dim_u32: u32 = @intCast(in_dim);

            var warm: usize = 0;
            while (warm < cfg.warmup) : (warm += 1) {
                if (rows > 4 and tile8 != null) {
                    try cuda.gaffine_u4_matvec_gate_up_silu.runWithFunctionTile8(arg_pack, device, tile8.?.function, &input_dev, &gate_weights.packed_data, &gate_weights.scales, &gate_weights.biases, &up_weights.packed_data, &up_weights.scales, &up_weights.biases, &output_dev, @intCast(out_dim), group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, in_dim_u32, batch_rows_u32);
                } else {
                    try cuda.gaffine_u4_matvec_gate_up_silu.runWithFunction(arg_pack, device, base.function, &input_dev, &gate_weights.packed_data, &gate_weights.scales, &gate_weights.biases, &up_weights.packed_data, &up_weights.scales, &up_weights.biases, &output_dev, @intCast(out_dim), group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, in_dim_u32, batch_rows_u32);
                }
            }

            for (samples) |*sample| {
                try device.recordEvent(start_evt, stream);
                if (rows > 4 and tile8 != null) {
                    try cuda.gaffine_u4_matvec_gate_up_silu.runWithFunctionTile8(arg_pack, device, tile8.?.function, &input_dev, &gate_weights.packed_data, &gate_weights.scales, &gate_weights.biases, &up_weights.packed_data, &up_weights.scales, &up_weights.biases, &output_dev, @intCast(out_dim), group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, in_dim_u32, batch_rows_u32);
                } else {
                    try cuda.gaffine_u4_matvec_gate_up_silu.runWithFunction(arg_pack, device, base.function, &input_dev, &gate_weights.packed_data, &gate_weights.scales, &gate_weights.biases, &up_weights.packed_data, &up_weights.scales, &up_weights.biases, &output_dev, @intCast(out_dim), group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, group_size, cuda.gaffine_u4_matvec.scales_dtype_f16, in_dim_u32, batch_rows_u32);
                }
                try device.recordEvent(stop_evt, stream);
                try device.synchronizeEvent(stop_evt);
                sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
            }

            return .{
                .name = scenarioName(which, quant),
                .quant = quant,
                .samples = samples,
                .flops_per_iter = gateUpFlops(rows, in_dim, out_dim),
                .bytes_per_iter = gateUpTq4Bytes(rows, in_dim, out_dim, packed_count * 2, sb_count * 2),
                .rows = rows,
                .in_dim = in_dim,
                .out_dim = out_dim,
            };
        },
        .nvfp4 => {
            const scale_cols = in_dim / 16;
            var gate_weights = try allocUploadNvfp4Weights(allocator, device, out_dim, in_dim, scale_cols);
            defer gate_weights.deinit(device, allocator);
            var up_weights = try allocUploadNvfp4Weights(allocator, device, out_dim, in_dim, scale_cols);
            defer up_weights.deinit(device, allocator);

            if (registry.embedded_module == null) try registry.loadEmbeddedModule(cuda.nvfp4_matvec.embedded_module);
            const base = try registry.resolveFunction("nvfp4_matvec_gate_up_silu_f32", "talu_nvfp4_matvec_gate_up_silu_f32");
            const tile8 = registry.resolveFunction("nvfp4_matvec_gate_up_silu_f32_tile8", "talu_nvfp4_matvec_gate_up_silu_f32_tile8") catch null;
            const batch_rows_u32: u32 = @intCast(rows);
            const in_dim_u32: u32 = @intCast(in_dim);
            const out_dim_u32: u32 = @intCast(out_dim);
            const prefer_tile8 = rows > 4 or (rows == 1 and out_dim >= 8192);

            var warm: usize = 0;
            while (warm < cfg.warmup) : (warm += 1) {
                try runNvfp4GateUpSiluLaunch(arg_pack, device, if (prefer_tile8 and tile8 != null) tile8.?.function else base.function, &input_dev, &gate_weights, &up_weights, &output_dev, out_dim_u32, in_dim_u32, batch_rows_u32, if (prefer_tile8 and tile8 != null) 8 else 4);
            }

            for (samples) |*sample| {
                try device.recordEvent(start_evt, stream);
                try runNvfp4GateUpSiluLaunch(arg_pack, device, if (prefer_tile8 and tile8 != null) tile8.?.function else base.function, &input_dev, &gate_weights, &up_weights, &output_dev, out_dim_u32, in_dim_u32, batch_rows_u32, if (prefer_tile8 and tile8 != null) 8 else 4);
                try device.recordEvent(stop_evt, stream);
                try device.synchronizeEvent(stop_evt);
                sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
            }

            return .{
                .name = scenarioName(which, quant),
                .quant = quant,
                .samples = samples,
                .flops_per_iter = gateUpFlops(rows, in_dim, out_dim),
                .bytes_per_iter = gateUpNvfp4Bytes(rows, in_dim, out_dim, gate_weights.packed_data.size + up_weights.packed_data.size, gate_weights.scales.size + up_weights.scales.size),
                .rows = rows,
                .in_dim = in_dim,
                .out_dim = out_dim,
            };
        },
        .nvfp4_i8cache => {
            const scale_cols = in_dim / 16;
            var gate_weights = try allocUploadNvfp4Weights(allocator, device, out_dim, in_dim, scale_cols);
            defer gate_weights.deinit(device, allocator);
            var up_weights = try allocUploadNvfp4Weights(allocator, device, out_dim, in_dim, scale_cols);
            defer up_weights.deinit(device, allocator);

            if (registry.embedded_module == null) try registry.loadEmbeddedModule(cuda.i8_matvec_gate_up_silu.embedded_module);
            const base = try registry.resolveFunction(cuda.i8_matvec_gate_up_silu.op_name, cuda.i8_matvec_gate_up_silu.embedded_symbol);
            const tile8 = registry.resolveFunction(cuda.i8_matvec_gate_up_silu.op_name_tile8, cuda.i8_matvec_gate_up_silu.embedded_symbol_tile8) catch null;
            const dequant_fn = try registry.resolveFunction("nvfp4_to_i8", "talu_nvfp4_to_i8");

            const i8_weight_bytes = std.math.mul(usize, out_dim, in_dim) catch return error.InvalidArgument;
            const row_scale_bytes = std.math.mul(usize, out_dim, @sizeOf(f32)) catch return error.InvalidArgument;
            var gate_i8_dev = try device.allocBuffer(i8_weight_bytes);
            defer gate_i8_dev.deinit(device);
            var gate_row_scales_dev = try device.allocBuffer(row_scale_bytes);
            defer gate_row_scales_dev.deinit(device);
            var up_i8_dev = try device.allocBuffer(i8_weight_bytes);
            defer up_i8_dev.deinit(device);
            var up_row_scales_dev = try device.allocBuffer(row_scale_bytes);
            defer up_row_scales_dev.deinit(device);

            arg_pack.reset();
            try arg_pack.appendBufferPtr(&gate_weights.packed_data);
            try arg_pack.appendBufferPtr(&gate_weights.scales);
            try arg_pack.appendBufferPtr(&gate_i8_dev);
            try arg_pack.appendBufferPtr(&gate_row_scales_dev);
            try arg_pack.appendScalar(u32, @intCast(in_dim));
            try arg_pack.appendScalar(u32, gate_weights.scale_cols);
            try arg_pack.appendScalar(f32, gate_weights.weight_global_scale);
            try cuda.launch.launchWithFamily(device, dequant_fn.function, .{
                .grid_x = @intCast(out_dim),
                .block_x = 256,
            }, arg_pack, .other);

            arg_pack.reset();
            try arg_pack.appendBufferPtr(&up_weights.packed_data);
            try arg_pack.appendBufferPtr(&up_weights.scales);
            try arg_pack.appendBufferPtr(&up_i8_dev);
            try arg_pack.appendBufferPtr(&up_row_scales_dev);
            try arg_pack.appendScalar(u32, @intCast(in_dim));
            try arg_pack.appendScalar(u32, up_weights.scale_cols);
            try arg_pack.appendScalar(f32, up_weights.weight_global_scale);
            try cuda.launch.launchWithFamily(device, dequant_fn.function, .{
                .grid_x = @intCast(out_dim),
                .block_x = 256,
            }, arg_pack, .other);

            const batch_rows_u32: u32 = @intCast(rows);
            const in_dim_u32: u32 = @intCast(in_dim);
            const out_dim_u32: u32 = @intCast(out_dim);
            const prefer_tile8 = rows > 4 or (rows == 1 and out_dim >= 8192);

            var warm: usize = 0;
            while (warm < cfg.warmup) : (warm += 1) {
                try runI8GateUpSiluLaunch(arg_pack, device, if (prefer_tile8 and tile8 != null) tile8.?.function else base.function, &input_dev, &gate_i8_dev, &gate_row_scales_dev, &up_i8_dev, &up_row_scales_dev, &output_dev, out_dim_u32, in_dim_u32, batch_rows_u32, if (prefer_tile8 and tile8 != null) 8 else 4);
            }

            for (samples) |*sample| {
                try device.recordEvent(start_evt, stream);
                try runI8GateUpSiluLaunch(arg_pack, device, if (prefer_tile8 and tile8 != null) tile8.?.function else base.function, &input_dev, &gate_i8_dev, &gate_row_scales_dev, &up_i8_dev, &up_row_scales_dev, &output_dev, out_dim_u32, in_dim_u32, batch_rows_u32, if (prefer_tile8 and tile8 != null) 8 else 4);
                try device.recordEvent(stop_evt, stream);
                try device.synchronizeEvent(stop_evt);
                sample.* = .{ .eval_ns = try device.elapsedEventNs(start_evt, stop_evt) };
            }

            return .{
                .name = scenarioName(which, quant),
                .quant = quant,
                .samples = samples,
                .flops_per_iter = gateUpFlops(rows, in_dim, out_dim),
                .bytes_per_iter = i8Bytes(rows, in_dim, out_dim, i8_weight_bytes * 2, out_dim * 2),
                .rows = rows,
                .in_dim = in_dim,
                .out_dim = out_dim,
            };
        },
        .nvfp4_native => return error.InvalidArgument,
    }
}

const U4Weights = struct {
    packed_data: cuda.Buffer,
    scales: cuda.Buffer,
    biases: cuda.Buffer,

    fn deinit(self: *U4Weights, device: *cuda.Device, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.packed_data.deinit(device);
        self.scales.deinit(device);
        self.biases.deinit(device);
    }
};

fn allocUploadU4Weights(allocator: std.mem.Allocator, device: *cuda.Device, packed_count: usize, sb_count: usize) !U4Weights {
    const packed_host = try allocator.alloc(u32, packed_count);
    defer allocator.free(packed_host);
    const scales_host = try allocator.alloc(u16, sb_count);
    defer allocator.free(scales_host);
    const biases_host = try allocator.alloc(u16, sb_count);
    defer allocator.free(biases_host);
    fillU4PackedPattern(packed_host);
    fillScaleBiasPattern(scales_host, biases_host);

    var packed_data = try device.allocBuffer(packed_count * @sizeOf(u32));
    errdefer packed_data.deinit(device);
    var scales = try device.allocBuffer(sb_count * @sizeOf(u16));
    errdefer scales.deinit(device);
    var biases = try device.allocBuffer(sb_count * @sizeOf(u16));
    errdefer biases.deinit(device);
    try packed_data.upload(device, std.mem.sliceAsBytes(packed_host));
    try scales.upload(device, std.mem.sliceAsBytes(scales_host));
    try biases.upload(device, std.mem.sliceAsBytes(biases_host));
    return .{ .packed_data = packed_data, .scales = scales, .biases = biases };
}

const Nvfp4Weights = struct {
    packed_data: cuda.Buffer,
    scales: cuda.Buffer,
    scales_lt_buffer: cuda.Buffer = .{ .pointer = 0, .size = 0 },
    scale_cols: u32,
    group_size: u32 = 16,
    weight_global_scale: f32 = 1.0,

    fn deinit(self: *Nvfp4Weights, device: *cuda.Device, allocator: std.mem.Allocator) void {
        _ = allocator;
        if (self.scales_lt_buffer.size > 0) self.scales_lt_buffer.deinit(device);
        self.packed_data.deinit(device);
        self.scales.deinit(device);
    }
};

const I8CachedWeights = struct {
    weight: cuda.Buffer,
    scales: cuda.Buffer,

    fn deinit(self: *I8CachedWeights, device: *cuda.Device) void {
        self.weight.deinit(device);
        self.scales.deinit(device);
    }
};

fn allocUploadNvfp4Weights(allocator: std.mem.Allocator, device: *cuda.Device, out_dim: usize, in_dim: usize, scale_cols: usize) !Nvfp4Weights {
    const packed_cols = (in_dim + 1) / 2;
    const packed_count = std.math.mul(usize, out_dim, packed_cols) catch return error.InvalidArgument;
    const scale_count = std.math.mul(usize, out_dim, scale_cols) catch return error.InvalidArgument;
    const packed_host = try allocator.alloc(u8, packed_count);
    defer allocator.free(packed_host);
    const scales_host = try allocator.alloc(u8, scale_count);
    defer allocator.free(scales_host);
    fillNvfp4PackedPattern(packed_host);
    fillNvfp4ScalePattern(scales_host);

    var packed_data = try device.allocBuffer(packed_count);
    errdefer packed_data.deinit(device);
    var scales = try device.allocBuffer(scale_count);
    errdefer scales.deinit(device);
    try packed_data.upload(device, packed_host);
    try scales.upload(device, scales_host);

    var scales_lt_buffer: cuda.Buffer = .{ .pointer = 0, .size = 0 };
    if (scale_cols >= (in_dim + 15) / 16) {
        const padded_scale_bytes = nvfp4LtScaleTensorSize(in_dim, out_dim);
        const padded_sf_k = nvfp4Roundoff((in_dim + 15) / 16, 4);
        const n_col_tiles = padded_sf_k / 4;
        const interleaved = try allocator.alloc(u8, padded_scale_bytes);
        defer allocator.free(interleaved);
        @memset(interleaved, 0);
        for (0..out_dim) |m| {
            for (0..((in_dim + 15) / 16)) |k| {
                const src_idx = m * scale_cols + k;
                const dst_idx = (m / 128) * n_col_tiles * 512 +
                    (k / 4) * 512 +
                    (m % 32) * 16 +
                    ((m % 128) / 32) * 4 +
                    (k % 4);
                interleaved[dst_idx] = scales_host[src_idx];
            }
        }
        scales_lt_buffer = try device.allocBuffer(padded_scale_bytes);
        errdefer scales_lt_buffer.deinit(device);
        try scales_lt_buffer.upload(device, interleaved);
    }

    return .{
        .packed_data = packed_data,
        .scales = scales,
        .scales_lt_buffer = scales_lt_buffer,
        .scale_cols = @intCast(scale_cols),
    };
}

fn allocDequantizedI8Weights(
    arg_pack: *cuda.ArgPack,
    device: *cuda.Device,
    dequant_fn: cuda.module.Function,
    src: *const Nvfp4Weights,
    out_dim: usize,
    in_dim: usize,
) !I8CachedWeights {
    const i8_weight_bytes = std.math.mul(usize, out_dim, in_dim) catch return error.InvalidArgument;
    const row_scale_bytes = std.math.mul(usize, out_dim, @sizeOf(f32)) catch return error.InvalidArgument;
    var weight = try device.allocBuffer(i8_weight_bytes);
    errdefer weight.deinit(device);
    var scales = try device.allocBuffer(row_scale_bytes);
    errdefer scales.deinit(device);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(&src.packed_data);
    try arg_pack.appendBufferPtr(&src.scales);
    try arg_pack.appendBufferPtr(&weight);
    try arg_pack.appendBufferPtr(&scales);
    try arg_pack.appendScalar(u32, @intCast(in_dim));
    try arg_pack.appendScalar(u32, src.scale_cols);
    try arg_pack.appendScalar(f32, src.weight_global_scale);
    try cuda.launch.launchWithFamily(device, dequant_fn, .{
        .grid_x = @intCast(out_dim),
        .block_x = 256,
    }, arg_pack, .other);

    return .{ .weight = weight, .scales = scales };
}

fn nvfp4Roundoff(value: usize, multiple: usize) usize {
    return ((value + multiple - 1) / multiple) * multiple;
}

fn nvfp4LtScaleTensorSize(inner: usize, outer: usize) usize {
    return nvfp4Roundoff(outer, 128) * nvfp4Roundoff((inner + 15) / 16, 4);
}

fn quantizeInputNvfp4Lt(
    arg_pack: *cuda.ArgPack,
    device: *cuda.Device,
    quant_fn: cuda.module.Function,
    input: *const cuda.Buffer,
    input_fp4_out: *cuda.Buffer,
    input_scales_out: *cuda.Buffer,
    in_dim: usize,
    rows: usize,
) !void {
    const padded_outer: u32 = @intCast(nvfp4Roundoff(rows, 128));
    const sf_k = (in_dim + 15) / 16;
    const padded_sf_k: u32 = @intCast(nvfp4Roundoff(sf_k, 4));
    const quant_grid_x: u32 = @intCast(sf_k);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendBufferPtr(input_fp4_out);
    try arg_pack.appendBufferPtr(input_scales_out);
    try arg_pack.appendScalar(u32, @intCast(in_dim));
    try arg_pack.appendScalar(u32, @intCast(rows));
    try arg_pack.appendScalar(u32, padded_outer);
    try arg_pack.appendScalar(u32, padded_sf_k);
    try cuda.launch.launchWithFamily(device, quant_fn, .{
        .grid_x = quant_grid_x,
        .grid_y = padded_outer,
        .block_x = 32,
    }, arg_pack, .other);
}

fn runNvfp4QkvLaunch(
    arg_pack: *cuda.ArgPack,
    device: *cuda.Device,
    function: cuda.module.Function,
    input: *const cuda.Buffer,
    q_weights: *const Nvfp4Weights,
    q_out: *cuda.Buffer,
    k_weights: *const Nvfp4Weights,
    k_out: *cuda.Buffer,
    v_weights: *const Nvfp4Weights,
    v_out: *cuda.Buffer,
    q_out_dim: u32,
    k_out_dim: u32,
    v_out_dim: u32,
    in_dim: u32,
    batch_rows: u32,
    batch_tile: u32,
    total_out: u32,
) !void {
    const single_row_dualout = batch_rows == 1 and batch_tile == 8;
    arg_pack.reset();
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendBufferPtr(&q_weights.packed_data);
    try arg_pack.appendBufferPtr(&q_weights.scales);
    try arg_pack.appendBufferPtr(q_out);
    try arg_pack.appendScalar(u32, q_out_dim);
    try arg_pack.appendScalar(u32, q_weights.scale_cols);
    try arg_pack.appendScalar(u32, q_weights.group_size);
    try arg_pack.appendScalar(f32, q_weights.weight_global_scale);
    try arg_pack.appendBufferPtr(&k_weights.packed_data);
    try arg_pack.appendBufferPtr(&k_weights.scales);
    try arg_pack.appendBufferPtr(k_out);
    try arg_pack.appendScalar(u32, k_out_dim);
    try arg_pack.appendScalar(u32, k_weights.scale_cols);
    try arg_pack.appendScalar(u32, k_weights.group_size);
    try arg_pack.appendScalar(f32, k_weights.weight_global_scale);
    try arg_pack.appendBufferPtr(&v_weights.packed_data);
    try arg_pack.appendBufferPtr(&v_weights.scales);
    try arg_pack.appendBufferPtr(v_out);
    try arg_pack.appendScalar(u32, v_out_dim);
    try arg_pack.appendScalar(u32, v_weights.scale_cols);
    try arg_pack.appendScalar(u32, v_weights.group_size);
    try arg_pack.appendScalar(f32, v_weights.weight_global_scale);
    try arg_pack.appendScalar(u32, in_dim);
    try arg_pack.appendScalar(u32, batch_rows);
    try cuda.launch.launchWithFamily(device, function, .{
        .grid_x = if (single_row_dualout) (total_out + 7) / 8 else (total_out + 3) / 4,
        .grid_y = (batch_rows + batch_tile - 1) / batch_tile,
        .block_x = 128,
    }, arg_pack, .matvec_qkv);
}

fn runNvfp4GateUpSiluLaunch(
    arg_pack: *cuda.ArgPack,
    device: *cuda.Device,
    function: cuda.module.Function,
    input: *const cuda.Buffer,
    gate_weights: *const Nvfp4Weights,
    up_weights: *const Nvfp4Weights,
    out: *cuda.Buffer,
    out_dim: u32,
    in_dim: u32,
    batch_rows: u32,
    batch_tile: u32,
) !void {
    arg_pack.reset();
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendBufferPtr(&gate_weights.packed_data);
    try arg_pack.appendBufferPtr(&gate_weights.scales);
    try arg_pack.appendBufferPtr(&up_weights.packed_data);
    try arg_pack.appendBufferPtr(&up_weights.scales);
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendScalar(u32, out_dim);
    try arg_pack.appendScalar(u32, in_dim);
    try arg_pack.appendScalar(u32, gate_weights.scale_cols);
    try arg_pack.appendScalar(u32, gate_weights.group_size);
    try arg_pack.appendScalar(f32, gate_weights.weight_global_scale);
    try arg_pack.appendScalar(u32, up_weights.scale_cols);
    try arg_pack.appendScalar(u32, up_weights.group_size);
    try arg_pack.appendScalar(f32, up_weights.weight_global_scale);
    try arg_pack.appendScalar(u32, batch_rows);
    try cuda.launch.launchWithFamily(device, function, .{
        .grid_x = if (batch_rows == 1 and batch_tile == 8) (out_dim + 7) / 8 else (out_dim + 3) / 4,
        .grid_y = (batch_rows + batch_tile - 1) / batch_tile,
        .block_x = 128,
    }, arg_pack, .matvec_gate_up_silu);
}

fn runI8GateUpSiluLaunch(
    arg_pack: *cuda.ArgPack,
    device: *cuda.Device,
    function: cuda.module.Function,
    input: *const cuda.Buffer,
    gate_weight: *const cuda.Buffer,
    gate_scales: *const cuda.Buffer,
    up_weight: *const cuda.Buffer,
    up_scales: *const cuda.Buffer,
    out: *cuda.Buffer,
    out_dim: u32,
    in_dim: u32,
    batch_rows: u32,
    batch_tile: u32,
) !void {
    arg_pack.reset();
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendBufferPtr(gate_weight);
    try arg_pack.appendBufferPtr(gate_scales);
    try arg_pack.appendBufferPtr(up_weight);
    try arg_pack.appendBufferPtr(up_scales);
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendScalar(u32, out_dim);
    try arg_pack.appendScalar(u32, in_dim);
    try arg_pack.appendScalar(u32, batch_rows);
    try cuda.launch.launchWithFamily(device, function, .{
        .grid_x = (out_dim + 3) / 4,
        .grid_y = (batch_rows + batch_tile - 1) / batch_tile,
        .block_x = 128,
    }, arg_pack, .matvec_gate_up_silu);
}

fn runI8MatvecLaunch(
    arg_pack: *cuda.ArgPack,
    device: *cuda.Device,
    function: cuda.module.Function,
    input: *const cuda.Buffer,
    weights: *const I8CachedWeights,
    out: *cuda.Buffer,
    in_dim: u32,
    out_dim: u32,
    batch_rows: u32,
) !void {
    arg_pack.reset();
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendBufferPtr(&weights.weight);
    try arg_pack.appendBufferPtr(&weights.scales);
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendScalar(u32, in_dim);
    try arg_pack.appendScalar(u32, out_dim);
    try arg_pack.appendScalar(u32, batch_rows);
    try arg_pack.appendDevicePtr(0);
    try cuda.launch.launchWithFamily(device, function, .{
        .grid_x = (out_dim + 3) / 4,
        .grid_y = batch_rows,
        .block_x = 128,
    }, arg_pack, .matvec);
}

fn runTq4MatvecLaunch(
    arg_pack: *cuda.ArgPack,
    device: *cuda.Device,
    function: cuda.module.Function,
    input: *const cuda.Buffer,
    weights: *const U4Weights,
    out: *cuda.Buffer,
    in_dim: u32,
    out_dim: u32,
    batch_rows: u32,
) !void {
    try cuda.gaffine_u4_matvec.runWithFunction(
        arg_pack,
        device,
        function,
        input,
        &weights.packed_data,
        &weights.scales,
        &weights.biases,
        out,
        in_dim,
        out_dim,
        64,
        cuda.gaffine_u4_matvec.scales_dtype_f16,
        batch_rows,
        0,
    );
}

fn runNvfp4MatvecLaunch(
    arg_pack: *cuda.ArgPack,
    device: *cuda.Device,
    function: cuda.module.Function,
    input: *const cuda.Buffer,
    weights: *const Nvfp4Weights,
    out: *cuda.Buffer,
    in_dim: u32,
    out_dim: u32,
    batch_rows: u32,
) !void {
    try cuda.nvfp4_matvec.runWithFunctionTile8(
        arg_pack,
        device,
        function,
        input,
        &weights.packed_data,
        &weights.scales,
        out,
        in_dim,
        out_dim,
        weights.scale_cols,
        weights.group_size,
        weights.weight_global_scale,
        batch_rows,
    );
}

fn runDecodeTokenChainTq4(
    arg_pack: *cuda.ArgPack,
    device: *cuda.Device,
    qkv_fn: cuda.module.Function,
    matvec_fn: cuda.module.Function,
    gate_up_fn: cuda.module.Function,
    rope_fn: cuda.module.Function,
    scores_fn: cuda.module.Function,
    softmax_fn: cuda.module.Function,
    weighted_sum_fn: cuda.module.Function,
    dims: DecodeTokenChainDims,
    hidden_a: *cuda.Buffer,
    hidden_b: *cuda.Buffer,
    q_out: *cuda.Buffer,
    k_out: *cuda.Buffer,
    v_out: *cuda.Buffer,
    attn_scores: *cuda.Buffer,
    attn_ctx: *cuda.Buffer,
    ffn_out: *cuda.Buffer,
    lm_out: *cuda.Buffer,
    positions_dev: *const cuda.Buffer,
    seq_lens_dev: *const cuda.Buffer,
    key_ptrs_dev: *const cuda.Buffer,
    value_ptrs_dev: *const cuda.Buffer,
    k_scale_ptrs_dev: *const cuda.Buffer,
    v_scale_ptrs_dev: *const cuda.Buffer,
    attn_scale: f32,
    q_weights: *const U4Weights,
    k_weights: *const U4Weights,
    v_weights: *const U4Weights,
    attn_out_weights: *const U4Weights,
    gate_weights: *const U4Weights,
    up_weights: *const U4Weights,
    down_weights: *const U4Weights,
    lm_weights: *const U4Weights,
) !void {
    const hidden_in = hidden_a;
    const hidden_tmp = hidden_b;
    const batch_rows: u32 = 1;
    const qkv_in_dim: u32 = @intCast(dims.qkv.hidden);
    const attn_out_in_dim: u32 = @intCast(dims.attn_out.hidden);
    const attn_out_out_dim: u32 = @intCast(dims.attn_out.out);
    const ffn_down_in_dim: u32 = @intCast(dims.ffn_down.hidden);
    const ffn_down_out_dim: u32 = @intCast(dims.ffn_down.out);
    const lm_in_dim: u32 = @intCast(dims.lm_head.hidden);
    const lm_out_dim: u32 = @intCast(dims.lm_head.out);

    var layer_idx: usize = 0;
    while (layer_idx < dims.layers) : (layer_idx += 1) {
        try cuda.gaffine_u4_matvec_qkv.runWithFunction(
            arg_pack,
            device,
            qkv_fn,
            hidden_in,
            &q_weights.packed_data,
            &q_weights.scales,
            &q_weights.biases,
            q_out,
            @intCast(dims.qkv.q_out),
            64,
            cuda.gaffine_u4_matvec.scales_dtype_f16,
            &k_weights.packed_data,
            &k_weights.scales,
            &k_weights.biases,
            k_out,
            @intCast(dims.qkv.k_out),
            64,
            cuda.gaffine_u4_matvec.scales_dtype_f16,
            &v_weights.packed_data,
            &v_weights.scales,
            &v_weights.biases,
            v_out,
            @intCast(dims.qkv.v_out),
            64,
            cuda.gaffine_u4_matvec.scales_dtype_f16,
            qkv_in_dim,
            batch_rows,
        );
        try runDecodeAttentionI8PtrsChain(
            arg_pack,
            device,
            rope_fn,
            scores_fn,
            softmax_fn,
            weighted_sum_fn,
            q_out,
            attn_scores,
            attn_ctx,
            positions_dev,
            seq_lens_dev,
            key_ptrs_dev,
            value_ptrs_dev,
            k_scale_ptrs_dev,
            v_scale_ptrs_dev,
            dims.attention,
            attn_scale,
        );
        try runTq4MatvecLaunch(arg_pack, device, matvec_fn, attn_ctx, attn_out_weights, hidden_tmp, attn_out_in_dim, attn_out_out_dim, batch_rows);
        try cuda.gaffine_u4_matvec_gate_up_silu.runWithFunction(
            arg_pack,
            device,
            gate_up_fn,
            hidden_tmp,
            &gate_weights.packed_data,
            &gate_weights.scales,
            &gate_weights.biases,
            &up_weights.packed_data,
            &up_weights.scales,
            &up_weights.biases,
            ffn_out,
            @intCast(dims.gate_up.out),
            64,
            cuda.gaffine_u4_matvec.scales_dtype_f16,
            64,
            cuda.gaffine_u4_matvec.scales_dtype_f16,
            @intCast(dims.gate_up.hidden),
            batch_rows,
        );
        try runTq4MatvecLaunch(arg_pack, device, matvec_fn, ffn_out, down_weights, hidden_in, ffn_down_in_dim, ffn_down_out_dim, batch_rows);
    }
    try runTq4MatvecLaunch(arg_pack, device, matvec_fn, hidden_in, lm_weights, lm_out, lm_in_dim, lm_out_dim, batch_rows);
}

fn runDecodeTokenChainNvfp4(
    arg_pack: *cuda.ArgPack,
    device: *cuda.Device,
    qkv_fn: cuda.module.Function,
    matvec_fn: cuda.module.Function,
    gate_up_fn: cuda.module.Function,
    rope_fn: cuda.module.Function,
    scores_fn: cuda.module.Function,
    softmax_fn: cuda.module.Function,
    weighted_sum_fn: cuda.module.Function,
    dims: DecodeTokenChainDims,
    hidden_a: *cuda.Buffer,
    hidden_b: *cuda.Buffer,
    q_out: *cuda.Buffer,
    k_out: *cuda.Buffer,
    v_out: *cuda.Buffer,
    attn_scores: *cuda.Buffer,
    attn_ctx: *cuda.Buffer,
    ffn_out: *cuda.Buffer,
    lm_out: *cuda.Buffer,
    positions_dev: *const cuda.Buffer,
    seq_lens_dev: *const cuda.Buffer,
    key_ptrs_dev: *const cuda.Buffer,
    value_ptrs_dev: *const cuda.Buffer,
    k_scale_ptrs_dev: *const cuda.Buffer,
    v_scale_ptrs_dev: *const cuda.Buffer,
    attn_scale: f32,
    q_weights: *const Nvfp4Weights,
    k_weights: *const Nvfp4Weights,
    v_weights: *const Nvfp4Weights,
    attn_out_weights: *const Nvfp4Weights,
    gate_weights: *const Nvfp4Weights,
    up_weights: *const Nvfp4Weights,
    down_weights: *const Nvfp4Weights,
    lm_weights: *const Nvfp4Weights,
) !void {
    const hidden_in = hidden_a;
    const hidden_tmp = hidden_b;
    const batch_rows: u32 = 1;
    const qkv_total_out: u32 = @intCast(dims.qkv.totalOut());
    const attn_out_in_dim: u32 = @intCast(dims.attn_out.hidden);
    const attn_out_out_dim: u32 = @intCast(dims.attn_out.out);
    const ffn_down_in_dim: u32 = @intCast(dims.ffn_down.hidden);
    const ffn_down_out_dim: u32 = @intCast(dims.ffn_down.out);
    const lm_in_dim: u32 = @intCast(dims.lm_head.hidden);
    const lm_out_dim: u32 = @intCast(dims.lm_head.out);

    var layer_idx: usize = 0;
    while (layer_idx < dims.layers) : (layer_idx += 1) {
        try runNvfp4QkvLaunch(
            arg_pack,
            device,
            qkv_fn,
            hidden_in,
            q_weights,
            q_out,
            k_weights,
            k_out,
            v_weights,
            v_out,
            @intCast(dims.qkv.q_out),
            @intCast(dims.qkv.k_out),
            @intCast(dims.qkv.v_out),
            @intCast(dims.qkv.hidden),
            batch_rows,
            8,
            qkv_total_out,
        );
        try runDecodeAttentionI8PtrsChain(
            arg_pack,
            device,
            rope_fn,
            scores_fn,
            softmax_fn,
            weighted_sum_fn,
            q_out,
            attn_scores,
            attn_ctx,
            positions_dev,
            seq_lens_dev,
            key_ptrs_dev,
            value_ptrs_dev,
            k_scale_ptrs_dev,
            v_scale_ptrs_dev,
            dims.attention,
            attn_scale,
        );
        try runNvfp4MatvecLaunch(arg_pack, device, matvec_fn, attn_ctx, attn_out_weights, hidden_tmp, attn_out_in_dim, attn_out_out_dim, batch_rows);
        try runNvfp4GateUpSiluLaunch(arg_pack, device, gate_up_fn, hidden_tmp, gate_weights, up_weights, ffn_out, @intCast(dims.gate_up.out), @intCast(dims.gate_up.hidden), batch_rows, 8);
        try runNvfp4MatvecLaunch(arg_pack, device, matvec_fn, ffn_out, down_weights, hidden_in, ffn_down_in_dim, ffn_down_out_dim, batch_rows);
    }
    try runNvfp4MatvecLaunch(arg_pack, device, matvec_fn, hidden_in, lm_weights, lm_out, lm_in_dim, lm_out_dim, batch_rows);
}

fn runDecodeTokenChainNvfp4Cached(
    arg_pack: *cuda.ArgPack,
    device: *cuda.Device,
    qkv_fn: cuda.module.Function,
    raw_matvec_fn: cuda.module.Function,
    gate_up_fn: cuda.module.Function,
    i8_matvec_fn: cuda.module.Function,
    rope_fn: cuda.module.Function,
    scores_fn: cuda.module.Function,
    softmax_fn: cuda.module.Function,
    weighted_sum_fn: cuda.module.Function,
    dims: DecodeTokenChainDims,
    hidden_a: *cuda.Buffer,
    hidden_b: *cuda.Buffer,
    q_out: *cuda.Buffer,
    k_out: *cuda.Buffer,
    v_out: *cuda.Buffer,
    attn_scores: *cuda.Buffer,
    attn_ctx: *cuda.Buffer,
    ffn_out: *cuda.Buffer,
    lm_out: *cuda.Buffer,
    positions_dev: *const cuda.Buffer,
    seq_lens_dev: *const cuda.Buffer,
    key_ptrs_dev: *const cuda.Buffer,
    value_ptrs_dev: *const cuda.Buffer,
    k_scale_ptrs_dev: *const cuda.Buffer,
    v_scale_ptrs_dev: *const cuda.Buffer,
    attn_scale: f32,
    q_weights: *const Nvfp4Weights,
    k_weights: *const Nvfp4Weights,
    v_weights: *const Nvfp4Weights,
    attn_out_cached: *const I8CachedWeights,
    gate_weights: *const Nvfp4Weights,
    up_weights: *const Nvfp4Weights,
    down_cached: *const I8CachedWeights,
    lm_weights: *const Nvfp4Weights,
) !void {
    const hidden_in = hidden_a;
    const hidden_tmp = hidden_b;
    const batch_rows: u32 = 1;
    const qkv_total_out: u32 = @intCast(dims.qkv.totalOut());
    const attn_out_in_dim: u32 = @intCast(dims.attn_out.hidden);
    const attn_out_out_dim: u32 = @intCast(dims.attn_out.out);
    const ffn_down_in_dim: u32 = @intCast(dims.ffn_down.hidden);
    const ffn_down_out_dim: u32 = @intCast(dims.ffn_down.out);
    const lm_in_dim: u32 = @intCast(dims.lm_head.hidden);
    const lm_out_dim: u32 = @intCast(dims.lm_head.out);

    var layer_idx: usize = 0;
    while (layer_idx < dims.layers) : (layer_idx += 1) {
        try runNvfp4QkvLaunch(
            arg_pack,
            device,
            qkv_fn,
            hidden_in,
            q_weights,
            q_out,
            k_weights,
            k_out,
            v_weights,
            v_out,
            @intCast(dims.qkv.q_out),
            @intCast(dims.qkv.k_out),
            @intCast(dims.qkv.v_out),
            @intCast(dims.qkv.hidden),
            batch_rows,
            8,
            qkv_total_out,
        );
        try runDecodeAttentionI8PtrsChain(
            arg_pack,
            device,
            rope_fn,
            scores_fn,
            softmax_fn,
            weighted_sum_fn,
            q_out,
            attn_scores,
            attn_ctx,
            positions_dev,
            seq_lens_dev,
            key_ptrs_dev,
            value_ptrs_dev,
            k_scale_ptrs_dev,
            v_scale_ptrs_dev,
            dims.attention,
            attn_scale,
        );
        try runI8MatvecLaunch(arg_pack, device, i8_matvec_fn, attn_ctx, attn_out_cached, hidden_tmp, attn_out_in_dim, attn_out_out_dim, batch_rows);
        try runNvfp4GateUpSiluLaunch(arg_pack, device, gate_up_fn, hidden_tmp, gate_weights, up_weights, ffn_out, @intCast(dims.gate_up.out), @intCast(dims.gate_up.hidden), batch_rows, 8);
        try runI8MatvecLaunch(arg_pack, device, i8_matvec_fn, ffn_out, down_cached, hidden_in, ffn_down_in_dim, ffn_down_out_dim, batch_rows);
    }
    try runNvfp4MatvecLaunch(arg_pack, device, raw_matvec_fn, hidden_in, lm_weights, lm_out, lm_in_dim, lm_out_dim, batch_rows);
}

fn fillInputPattern(values: []f32) void {
    for (values, 0..) |*value, idx| {
        const bucket: u32 = @intCast((idx % 23) + 1);
        value.* = @as(f32, @floatFromInt(bucket)) * 0.03125;
    }
}

fn fillI8Pattern(values: []i8) void {
    for (values, 0..) |*value, idx| {
        const bucket: i32 = @intCast((idx % 31) - 15);
        value.* = @intCast(bucket);
    }
}

fn fillF32ScalePattern(values: []f32) void {
    for (values, 0..) |*value, idx| {
        const bucket: u32 = @intCast((idx % 7) + 1);
        value.* = 0.015625 * @as(f32, @floatFromInt(bucket));
    }
}

fn fillU4PackedPattern(words: []u32) void {
    for (words, 0..) |*word, idx| {
        const a: u32 = @intCast(idx & 0xF);
        const b: u32 = @intCast((idx + 3) & 0xF);
        word.* =
            (a << 0) |
            (b << 4) |
            (a << 8) |
            (b << 12) |
            (a << 16) |
            (b << 20) |
            (a << 24) |
            (b << 28);
    }
}

fn fillScaleBiasPattern(scales: []u16, biases: []u16) void {
    for (scales, 0..) |*scale, idx| {
        scale.* = dtype.f32ToFp16(0.125 + 0.015625 * @as(f32, @floatFromInt(idx % 4)));
    }
    @memset(biases, 0);
}

fn fillNvfp4PackedPattern(bytes: []u8) void {
    for (bytes, 0..) |*byte, idx| {
        const lo: u8 = @intCast(idx & 0xF);
        const hi: u8 = @intCast((idx + 5) & 0xF);
        byte.* = lo | (hi << 4);
    }
}

fn fillNvfp4ScalePattern(scales: []u8) void {
    const lut = [_]u8{ 0x30, 0x34, 0x38, 0x3C, 0x40, 0x44 };
    for (scales, 0..) |*value, idx| {
        value.* = lut[idx % lut.len];
    }
}

fn batchRowsHeads(batch_rows: usize, n_heads: usize) usize {
    return batch_rows * n_heads;
}

fn denseFlops(rows: usize, in_dim: usize, out_dim: usize) u64 {
    const product = std.math.mul(u128, rows, in_dim) catch return 0;
    const total = std.math.mul(u128, product, out_dim) catch return 0;
    return std.math.cast(u64, total * 2) orelse std.math.maxInt(u64);
}

fn gateUpFlops(rows: usize, in_dim: usize, out_dim: usize) u64 {
    const product = std.math.mul(u128, rows, in_dim) catch return 0;
    const total = std.math.mul(u128, product, out_dim) catch return 0;
    return std.math.cast(u64, total * 4) orelse std.math.maxInt(u64);
}

fn decodeAttentionI8Flops(dims: DecodeAttentionDims) u64 {
    const batch_rows: usize = 1;
    const score_flops = denseFlops(batch_rows * dims.n_heads, dims.head_dim, dims.seq_len);
    const value_flops = denseFlops(batch_rows * dims.n_heads, dims.seq_len, dims.head_dim);
    const rope_flops = denseFlops(batch_rows * dims.n_heads, dims.rope_dim, 1);
    return score_flops + value_flops + rope_flops;
}

fn decodeAttentionI8Bytes(dims: DecodeAttentionDims) u64 {
    const batch_rows: usize = 1;
    const kv_row_stride = dims.kvRowStride();
    const query_bytes = std.math.mul(u128, batch_rows * dims.qDim(), @sizeOf(f32)) catch return 0;
    const score_bytes = std.math.mul(u128, batchRowsHeads(batch_rows, dims.n_heads) * dims.seq_len, @sizeOf(f32)) catch return 0;
    const out_bytes = std.math.mul(u128, batch_rows * dims.qDim(), @sizeOf(f32)) catch return 0;
    const kv_bytes = std.math.mul(u128, dims.seq_len * kv_row_stride, @sizeOf(i8) * 2) catch return 0;
    const scale_bytes = std.math.mul(u128, dims.seq_len * dims.n_kv_heads, @sizeOf(f32) * 2) catch return 0;
    return std.math.cast(u64, query_bytes + score_bytes + out_bytes + kv_bytes + scale_bytes) orelse std.math.maxInt(u64);
}

fn decodeTokenChainFlops(dims: DecodeTokenChainDims) u64 {
    const layer_flops =
        denseFlops(1, dims.qkv.hidden, dims.qkv.totalOut()) +
        decodeAttentionI8Flops(dims.attention) +
        denseFlops(1, dims.attn_out.hidden, dims.attn_out.out) +
        gateUpFlops(1, dims.gate_up.hidden, dims.gate_up.out) +
        denseFlops(1, dims.ffn_down.hidden, dims.ffn_down.out);
    return layer_flops * dims.layers + denseFlops(1, dims.lm_head.hidden, dims.lm_head.out);
}

fn decodeTokenChainBytes(dims: DecodeTokenChainDims) u64 {
    const qkv_bytes = std.math.cast(u64, std.math.mul(u128, dims.qkv.hidden * dims.qkv.totalOut(), @sizeOf(f32)) catch 0) orelse 0;
    const attn_out_bytes = std.math.cast(u64, std.math.mul(u128, dims.attn_out.hidden * dims.attn_out.out, @sizeOf(f32)) catch 0) orelse 0;
    const gate_up_bytes = std.math.cast(u64, std.math.mul(u128, dims.gate_up.hidden * dims.gate_up.out * 2, @sizeOf(f32)) catch 0) orelse 0;
    const ffn_down_bytes = std.math.cast(u64, std.math.mul(u128, dims.ffn_down.hidden * dims.ffn_down.out, @sizeOf(f32)) catch 0) orelse 0;
    const layer_bytes = qkv_bytes + decodeAttentionI8Bytes(dims.attention) + attn_out_bytes + gate_up_bytes + ffn_down_bytes;
    const lm_bytes = std.math.cast(u64, std.math.mul(u128, dims.lm_head.hidden * dims.lm_head.out, @sizeOf(f32)) catch 0) orelse 0;
    return layer_bytes * dims.layers + lm_bytes;
}

fn tq4Bytes(rows: usize, in_dim: usize, out_dim: usize, packed_count: usize, sb_count: usize) u64 {
    const input_bytes = std.math.mul(u128, rows * in_dim, @sizeOf(f32)) catch return 0;
    const output_bytes = std.math.mul(u128, rows * out_dim, @sizeOf(f32)) catch return 0;
    const weight_bytes = std.math.mul(u128, packed_count, @sizeOf(u32)) catch return 0;
    const scale_bias_bytes = std.math.mul(u128, sb_count, @sizeOf(u16) * 2) catch return 0;
    return std.math.cast(u64, input_bytes + output_bytes + weight_bytes + scale_bias_bytes) orelse std.math.maxInt(u64);
}

fn nvfp4Bytes(rows: usize, in_dim: usize, out_dim: usize, packed_count: usize, scale_count: usize) u64 {
    const input_bytes = std.math.mul(u128, rows * in_dim, @sizeOf(f32)) catch return 0;
    const output_bytes = std.math.mul(u128, rows * out_dim, @sizeOf(f32)) catch return 0;
    return std.math.cast(u64, input_bytes + output_bytes + packed_count + scale_count) orelse std.math.maxInt(u64);
}

fn i8Bytes(rows: usize, in_dim: usize, out_dim: usize, weight_bytes: usize, scale_count: usize) u64 {
    const input_bytes = std.math.mul(u128, rows * in_dim, @sizeOf(f32)) catch return 0;
    const output_bytes = std.math.mul(u128, rows * out_dim, @sizeOf(f32)) catch return 0;
    const scale_bytes = std.math.mul(u128, scale_count, @sizeOf(f32)) catch return 0;
    return std.math.cast(u64, input_bytes + output_bytes + weight_bytes + scale_bytes) orelse std.math.maxInt(u64);
}

fn qkvTq4Bytes(rows: usize, in_dim: usize, q_out: usize, k_out: usize, v_out: usize, packed_count_total: usize, sb_count_total: usize) u64 {
    const out_total = q_out + k_out + v_out;
    return tq4Bytes(rows, in_dim, out_total, packed_count_total, sb_count_total);
}

fn qkvNvfp4Bytes(rows: usize, in_dim: usize, q_out: usize, k_out: usize, v_out: usize, packed_count_total: usize, scale_count_total: usize) u64 {
    const out_total = q_out + k_out + v_out;
    return nvfp4Bytes(rows, in_dim, out_total, packed_count_total, scale_count_total);
}

fn gateUpTq4Bytes(rows: usize, in_dim: usize, out_dim: usize, packed_count_total: usize, sb_count_total: usize) u64 {
    return tq4Bytes(rows, in_dim, out_dim, packed_count_total, sb_count_total);
}

fn gateUpNvfp4Bytes(rows: usize, in_dim: usize, out_dim: usize, packed_count_total: usize, scale_count_total: usize) u64 {
    return nvfp4Bytes(rows, in_dim, out_dim, packed_count_total, scale_count_total);
}

test "scenario names are stable" {
    try std.testing.expectEqualStrings("decode.attn_scores.nvfp4", scenarioName(.decode_attn_scores, .nvfp4));
    try std.testing.expectEqualStrings("decode.attn_q.tq4", scenarioName(.decode_attn_q, .tq4));
    try std.testing.expectEqualStrings("prefill.ffn_down.nvfp4", scenarioName(.prefill_ffn_down, .nvfp4));
    try std.testing.expectEqualStrings("decode.attn_qkv_fused.nvfp4", scenarioName(.decode_attn_qkv_fused, .nvfp4));
}

test "qwen3_5_4b exact dims resolve for fused rows" {
    const qkv = try modelQkvDims("qwen3_5_4b", .decode_attn_qkv_fused, null);
    try std.testing.expectEqual(@as(usize, 1), qkv.rows);
    try std.testing.expectEqual(@as(usize, 2560), qkv.hidden);
    try std.testing.expectEqual(@as(usize, 4096), qkv.q_out);
    try std.testing.expectEqual(@as(usize, 1024), qkv.k_out);
    try std.testing.expectEqual(@as(usize, 1024), qkv.v_out);

    const gate = try modelGateUpDims("qwen3_5_4b", .prefill_ffn_gate_up_fused_silu, 387);
    try std.testing.expectEqual(@as(usize, 387), gate.rows);
    try std.testing.expectEqual(@as(usize, 2560), gate.hidden);
    try std.testing.expectEqual(@as(usize, 9216), gate.out);
}
