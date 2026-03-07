const std = @import("std");
const main = @import("main");
const cpu = main.compute.cpu;
const models = main.models.dispatcher;
const tensor = main.tensor;
const dtype = main.core.dtype;
const harness = @import("harness.zig");

pub const Scenario = enum {
    all,
    role_attn_q_bf16,
    role_attn_k_bf16,
    role_attn_v_bf16,
    role_attn_out_bf16,
    role_ffn_gate_bf16,
    role_ffn_down_bf16,
    gated_delta_conv_f32,
    gated_delta_qk_norm_f32,
    gated_delta_step_f32,
    gated_delta_norm_f32,
    gated_attention_gate_f32,
    gated_attention_gate_long_f32,
    rope_f32,
    rope_f16,
    rope_bf16,
    sdpa_f32,
    sdpa_f16,
    sdpa_bf16,
    rmsnorm_f32,
    rmsnorm_bf16_weight,
    rmsnorm_f16_weight,
    softmax_f32,
    decode_bf16,
    decode_u4,
    decode_u8,
    shortconv_decode_f32,
    mamba_scan_f32,
    matmul_throughput_f32,
    matmul_throughput_bf16,
    matmul_throughput_f16,
    matmul_throughput_gaffine_u4,
    matmul_throughput_gaffine_u8,
    micro_matmul_f32,
    add_f32,
    mul_f32,
};

pub const Profile = enum {
    ci,
    bw,
};

pub const RunConfig = struct {
    warmup: usize = 8,
    iters: usize = 24,
    profile: Profile = .bw,
    model_id: ?[]const u8 = null,
};

pub const ScenarioResult = struct {
    name: []const u8,
    profile: Profile,
    samples: []harness.Sample,
    cold_ns: u64,
    sample_loops: usize,
    flops_per_iter: u64,
    bytes_per_iter: u64,
    note: []const u8,

    pub fn deinit(self: *ScenarioResult, allocator: std.mem.Allocator) void {
        allocator.free(self.samples);
    }
};

fn sampleLoops(which: Scenario, profile: Profile) usize {
    return switch (which) {
        .role_attn_k_bf16, .role_attn_v_bf16 => switch (profile) {
            .ci => 32,
            .bw => 16,
        },
        .role_attn_q_bf16, .role_attn_out_bf16, .role_ffn_gate_bf16, .role_ffn_down_bf16 => switch (profile) {
            .ci => 8,
            .bw => 4,
        },
        .add_f32, .mul_f32 => switch (profile) {
            .ci => 128,
            .bw => 64,
        },
        .rmsnorm_f32, .rmsnorm_bf16_weight, .rmsnorm_f16_weight => switch (profile) {
            .ci => 64,
            .bw => 32,
        },
        .softmax_f32 => switch (profile) {
            .ci => 16,
            .bw => 8,
        },
        .shortconv_decode_f32 => switch (profile) {
            .ci => 256,
            .bw => 128,
        },
        .mamba_scan_f32 => switch (profile) {
            .ci => 64,
            .bw => 32,
        },
        .gated_delta_conv_f32, .gated_delta_qk_norm_f32 => switch (profile) {
            .ci => 256,
            .bw => 128,
        },
        .gated_delta_step_f32 => switch (profile) {
            .ci => 128,
            .bw => 64,
        },
        .gated_delta_norm_f32 => switch (profile) {
            .ci => 256,
            .bw => 128,
        },
        .gated_attention_gate_f32 => switch (profile) {
            .ci => 2048,
            .bw => 1024,
        },
        .gated_attention_gate_long_f32 => switch (profile) {
            .ci => 64,
            .bw => 32,
        },
        .rope_f32, .rope_f16, .rope_bf16 => switch (profile) {
            .ci => 128,
            .bw => 64,
        },
        .sdpa_f32, .sdpa_f16, .sdpa_bf16 => switch (profile) {
            .ci => 16,
            .bw => 8,
        },
        .decode_bf16 => switch (profile) {
            .ci => 128,
            .bw => 64,
        },
        .decode_u4, .decode_u8 => switch (profile) {
            .ci => 64,
            .bw => 32,
        },
        .matmul_throughput_f32, .matmul_throughput_bf16, .matmul_throughput_f16, .matmul_throughput_gaffine_u4, .matmul_throughput_gaffine_u8 => switch (profile) {
            .ci => 2,
            .bw => 1,
        },
        .micro_matmul_f32 => switch (profile) {
            .ci => 2,
            .bw => 1,
        },
        .all => unreachable,
    };
}

const RoleMatmulDims = struct {
    tokens: usize,
    hidden: usize,
    out: usize,
};

fn roleBenchRowName(which: Scenario) ![]const u8 {
    return switch (which) {
        .role_attn_q_bf16 => "role.attn_q",
        .role_attn_k_bf16 => "role.attn_k",
        .role_attn_v_bf16 => "role.attn_v",
        .role_attn_out_bf16 => "role.attn_out",
        .role_ffn_gate_bf16 => "role.ffn_gate",
        .role_ffn_down_bf16 => "role.ffn_down",
        else => error.InvalidArgument,
    };
}

fn modelRoleMatmulDims(model_id: []const u8, which: Scenario) !RoleMatmulDims {
    const hints = models.performanceHintsByName(model_id) orelse return error.InvalidArgument;
    const bench_row = try roleBenchRowName(which);
    // Model-owned overrides win. Otherwise bench falls back to a shared
    // representative text shape so every supported architecture still has an
    // immediate local compute loop, even when the family spans many sizes.
    const dims = models.perf_hints.roleDimsFor(hints, bench_row) orelse
        models.perf_hints.defaultRoleDimsFor(bench_row) orelse return error.InvalidArgument;
    return .{
        .tokens = dims.tokens,
        .hidden = dims.hidden,
        .out = dims.out,
    };
}

const GatedDeltaDims = struct {
    n_heads: usize,
    d_head: usize,
    d_conv: usize,

    fn dInner(self: GatedDeltaDims) usize {
        return self.n_heads * self.d_head;
    }

    fn stateElems(self: GatedDeltaDims) usize {
        return self.n_heads * self.d_head * self.d_head;
    }
};

const GatedAttentionDims = struct {
    head_count: usize,
    head_dim: usize,
    seq_len: usize,

    fn queryDim(self: GatedAttentionDims) usize {
        return self.head_count * self.head_dim;
    }

    fn queryProjectionDim(self: GatedAttentionDims) usize {
        return self.queryDim() * 2;
    }
};

const RopeDims = struct {
    batch: usize,
    groups: usize,
    seq_len: usize,
    width: usize,
};

const SdpaDims = struct {
    batch: usize,
    groups: usize,
    query_steps: usize,
    key_steps: usize,
    width: usize,
};

const DecodeDims = struct {
    row_count: usize,
    row_width: usize,
    take_rows: usize,
    group_size: usize,
};

const SoftmaxDims = struct {
    rows: usize,
    cols: usize,
};

const MatmulDims = struct {
    m: usize,
    k: usize,
    n: usize,
};

const ShortConvDims = struct {
    conv_dim: usize,
    d_conv: usize,
};

const ScanDims = struct {
    n_heads: usize,
    d_head: usize,
    d_state: usize,
    n_groups: usize,

    fn dInner(self: ScanDims) usize {
        return self.n_heads * self.d_head;
    }

    fn stateElems(self: ScanDims) usize {
        return self.n_heads * self.d_head * self.d_state;
    }

    fn groupStateElems(self: ScanDims) usize {
        return self.n_groups * self.d_state;
    }
};

fn profileGatedDeltaDims(profile: Profile) GatedDeltaDims {
    return switch (profile) {
        .ci => .{ .n_heads = 8, .d_head = 64, .d_conv = 4 },
        .bw => .{ .n_heads = 16, .d_head = 128, .d_conv = 4 },
    };
}

fn profileAttentionDims(profile: Profile, long: bool) GatedAttentionDims {
    return switch (profile) {
        .ci => .{
            .head_count = 8,
            .head_dim = 64,
            .seq_len = if (long) 32 else 1,
        },
        .bw => .{
            .head_count = 16,
            .head_dim = 64,
            .seq_len = if (long) 128 else 1,
        },
    };
}

fn profileRopeDims(profile: Profile) RopeDims {
    return switch (profile) {
        .ci => .{ .batch = 1, .groups = 8, .seq_len = 64, .width = 128 },
        .bw => .{ .batch = 1, .groups = 16, .seq_len = 256, .width = 128 },
    };
}

fn profileSdpaDims(profile: Profile) SdpaDims {
    return switch (profile) {
        .ci => .{ .batch = 1, .groups = 8, .query_steps = 32, .key_steps = 32, .width = 64 },
        .bw => .{ .batch = 1, .groups = 16, .query_steps = 128, .key_steps = 128, .width = 64 },
    };
}

fn profileDecodeDims(profile: Profile) DecodeDims {
    return switch (profile) {
        .ci => .{ .row_count = 512, .row_width = 1024, .take_rows = 32, .group_size = 128 },
        .bw => .{ .row_count = 2048, .row_width = 2048, .take_rows = 64, .group_size = 128 },
    };
}

fn profileSoftmaxDims(profile: Profile) SoftmaxDims {
    return switch (profile) {
        .ci => .{ .rows = 128, .cols = 1024 },
        .bw => .{ .rows = 512, .cols = 2048 },
    };
}

fn profileMatmulDims(profile: Profile, micro: bool) MatmulDims {
    return switch (profile) {
        .ci => if (micro)
            .{ .m = 1, .k = 1024, .n = 1024 }
        else
            .{ .m = 64, .k = 1024, .n = 1024 },
        .bw => if (micro)
            .{ .m = 1, .k = 4096, .n = 4096 }
        else
            .{ .m = 256, .k = 2048, .n = 2048 },
    };
}

fn profileShortConvDims(profile: Profile) ShortConvDims {
    return switch (profile) {
        .ci => .{ .conv_dim = 512, .d_conv = 4 },
        .bw => .{ .conv_dim = 2048, .d_conv = 4 },
    };
}

fn profileScanDims(profile: Profile) ScanDims {
    return switch (profile) {
        .ci => .{ .n_heads = 8, .d_head = 64, .d_state = 64, .n_groups = 1 },
        .bw => .{ .n_heads = 16, .d_head = 128, .d_state = 64, .n_groups = 1 },
    };
}

fn profileVectorElems(profile: Profile) usize {
    return switch (profile) {
        .ci => 256 * 1024,
        .bw => 1024 * 1024,
    };
}

fn fillDeterministicF32(values: []align(1) f32, seed: u64) void {
    var state = seed ^ 0x9E3779B185EBCA87;
    for (values, 0..) |*slot, idx| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        const raw = @as(f32, @floatFromInt((state >> 40) & 0x3FF));
        const centered = (raw / 1024.0) - 0.5;
        const tweak = @as(f32, @floatFromInt((idx % 11) + 1)) * 0.00390625;
        slot.* = centered + tweak;
    }
}

fn allocFilledF32(allocator: std.mem.Allocator, len: usize, seed: u64) ![]f32 {
    const values = try allocator.alloc(f32, len);
    fillDeterministicF32(values, seed);
    return values;
}

fn allocZeroF32(allocator: std.mem.Allocator, len: usize) ![]f32 {
    const values = try allocator.alloc(f32, len);
    @memset(values, 0.0);
    return values;
}

fn fillTensorF32(owned: *tensor.OwnedTensor, seed: u64) void {
    fillDeterministicF32(owned.asSlice(f32), seed);
}

fn fillTensorLike(bytes: []u8, tensor_dtype: dtype.DType, seed: u64) void {
    switch (tensor_dtype) {
        .f32 => fillDeterministicF32(std.mem.bytesAsSlice(f32, bytes), seed),
        .f16 => fillU16SliceAsF16(std.mem.bytesAsSlice(u16, bytes), seed),
        .bf16 => fillU16SliceAsBf16(std.mem.bytesAsSlice(u16, bytes), seed),
        else => unreachable,
    }
}

const RopeFreqOwned = struct {
    tensor_owned: tensor.OwnedTensor,

    fn deinit(self: *RopeFreqOwned) void {
        self.tensor_owned.deinit();
        self.* = undefined;
    }

    fn view(self: *RopeFreqOwned) tensor.Tensor {
        return self.tensor_owned.view();
    }
};

fn allocRopeFreqTable(allocator: std.mem.Allocator, tensor_dtype: dtype.DType, seq_len: usize, half_dim: usize, seed: u64) !RopeFreqOwned {
    const owned = try tensor.OwnedTensor.init(allocator, tensor_dtype, &.{ 1, seq_len, half_dim });
    fillTensorLike(owned.data, tensor_dtype, seed);
    return .{ .tensor_owned = owned };
}

fn fillU16SliceAsF16(values: []align(1) u16, seed: u64) void {
    var state = seed ^ 0xD1B54A32D192ED03;
    for (values, 0..) |*slot, idx| {
        state = state *% 2862933555777941757 +% 3037000493;
        const raw = @as(f32, @floatFromInt((state >> 40) & 0x1FF));
        const centered = (raw / 512.0) - 0.5;
        const tweak = @as(f32, @floatFromInt((idx % 17) + 1)) * 0.0078125;
        slot.* = dtype.f32ToFp16(centered + tweak);
    }
}

fn fillU16SliceAsBf16(values: []align(1) u16, seed: u64) void {
    var state = seed ^ 0x94D049BB133111EB;
    for (values, 0..) |*slot, idx| {
        state = state *% 3202034522624059733 +% 1;
        const raw = @as(f32, @floatFromInt((state >> 39) & 0x1FF));
        const centered = (raw / 512.0) - 0.5;
        const tweak = @as(f32, @floatFromInt((idx % 13) + 1)) * 0.0078125;
        slot.* = dtype.f32ToBf16(centered + tweak);
    }
}

const QuantOwnedTensor = struct {
    allocator: std.mem.Allocator,
    data: []align(32) u8,
    scales: []u8,
    biases: []u8,
    tensor_view: tensor.Tensor,

    fn init(allocator: std.mem.Allocator, comptime quant_dtype: dtype.DType, shape: []const usize, group_size: usize, scales_dtype: dtype.DType, seed: u64) !QuantOwnedTensor {
        std.debug.assert(quant_dtype == .grouped_affine_u4 or quant_dtype == .grouped_affine_u8);
        if (shape.len != 2) return error.InvalidShape;

        const n_cols = shape[0];
        const k_dim = shape[1];
        if (k_dim % group_size != 0) return error.InvalidShape;

        const packed_bytes = switch (quant_dtype) {
            .grouped_affine_u4 => (n_cols * k_dim) / 2,
            .grouped_affine_u8 => n_cols * k_dim,
            else => unreachable,
        };
        const data = try allocator.alignedAlloc(u8, .@"32", packed_bytes);
        errdefer allocator.free(data);
        var state = seed ^ 0xF1357AEA2E62A9C5;
        for (data) |*slot| {
            state = state *% 6364136223846793005 +% 1442695040888963407;
            slot.* = @truncate((state >> 32) & 0xFF);
        }

        const groups_per_row = k_dim / group_size;
        const meta_elems = n_cols * groups_per_row;
        const meta_bytes = meta_elems * @sizeOf(u16);
        const scales = try allocator.alloc(u8, meta_bytes);
        errdefer allocator.free(scales);
        const biases = try allocator.alloc(u8, meta_bytes);
        errdefer allocator.free(biases);

        const scale_words = std.mem.bytesAsSlice(u16, scales);
        const bias_words = std.mem.bytesAsSlice(u16, biases);
        switch (scales_dtype) {
            .f16 => {
                fillU16SliceAsF16(scale_words, seed + 1);
                fillU16SliceAsF16(bias_words, seed + 2);
            },
            .bf16 => {
                fillU16SliceAsBf16(scale_words, seed + 1);
                fillU16SliceAsBf16(bias_words, seed + 2);
            },
            else => return error.InvalidArgument,
        }

        var t = tensor.Tensor.view(data.ptr, shape, quant_dtype, data.len);
        t.gaffine = .{
            .scales = scales,
            .biases = biases,
            .group_size = group_size,
            .scales_dtype = scales_dtype,
        };
        return .{
            .allocator = allocator,
            .data = data,
            .scales = scales,
            .biases = biases,
            .tensor_view = t,
        };
    }

    fn deinit(self: *QuantOwnedTensor) void {
        self.allocator.free(self.data);
        self.allocator.free(self.scales);
        self.allocator.free(self.biases);
        self.* = undefined;
    }

    fn view(self: *QuantOwnedTensor) *tensor.Tensor {
        return &self.tensor_view;
    }
};

fn timeAddF32(dst: []f32, a: []const f32, b: []const f32) u64 {
    var timer = std.time.Timer.start() catch unreachable;
    cpu.rowwise.addInto(a, b, dst);
    return timer.read();
}

fn timeMulF32(dst: []f32, a: []const f32, b: []const f32) u64 {
    var timer = std.time.Timer.start() catch unreachable;
    cpu.activation.elementwiseMul(a, b, dst);
    return timer.read();
}

pub fn runAddF32(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    const elems = profileVectorElems(cfg.profile);
    const repeats = sampleLoops(.add_f32, cfg.profile);
    const a = try allocFilledF32(allocator, elems, 1);
    defer allocator.free(a);
    const b = try allocFilledF32(allocator, elems, 2);
    defer allocator.free(b);
    const out = try allocZeroF32(allocator, elems);
    defer allocator.free(out);
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    var timer = std.time.Timer.start() catch unreachable;
    for (0..repeats) |_| cpu.rowwise.addInto(a, b, out);
    const cold_ns = timer.read();
    for (0..cfg.warmup) |_| {
        for (0..repeats) |_| cpu.rowwise.addInto(a, b, out);
    }
    for (samples) |*sample| {
        timer = std.time.Timer.start() catch unreachable;
        for (0..repeats) |_| cpu.rowwise.addInto(a, b, out);
        sample.eval_ns = timer.read();
    }

    return .{
        .name = "add_f32",
        .profile = cfg.profile,
        .samples = samples,
        .cold_ns = cold_ns,
        .sample_loops = repeats,
        .flops_per_iter = elems * repeats,
        .bytes_per_iter = 3 * elems * @sizeOf(f32) * repeats,
        .note = "rowwise add baseline",
    };
}

pub fn runMulF32(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    const elems = profileVectorElems(cfg.profile);
    const repeats = sampleLoops(.mul_f32, cfg.profile);
    const a = try allocFilledF32(allocator, elems, 3);
    defer allocator.free(a);
    const b = try allocFilledF32(allocator, elems, 4);
    defer allocator.free(b);
    const out = try allocZeroF32(allocator, elems);
    defer allocator.free(out);
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    var timer = std.time.Timer.start() catch unreachable;
    for (0..repeats) |_| cpu.activation.elementwiseMul(a, b, out);
    const cold_ns = timer.read();
    for (0..cfg.warmup) |_| {
        for (0..repeats) |_| cpu.activation.elementwiseMul(a, b, out);
    }
    for (samples) |*sample| {
        timer = std.time.Timer.start() catch unreachable;
        for (0..repeats) |_| cpu.activation.elementwiseMul(a, b, out);
        sample.eval_ns = timer.read();
    }

    return .{
        .name = "mul_f32",
        .profile = cfg.profile,
        .samples = samples,
        .cold_ns = cold_ns,
        .sample_loops = repeats,
        .flops_per_iter = elems * repeats,
        .bytes_per_iter = 3 * elems * @sizeOf(f32) * repeats,
        .note = "elementwise multiply baseline",
    };
}

pub fn runRmsNormF32(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    const dims = profileSoftmaxDims(cfg.profile);
    const repeats = sampleLoops(.rmsnorm_f32, cfg.profile);
    const elems = dims.rows * dims.cols;
    const input = try allocFilledF32(allocator, elems, 5);
    defer allocator.free(input);
    const out = try allocZeroF32(allocator, elems);
    defer allocator.free(out);
    const weight = try allocFilledF32(allocator, dims.cols, 6);
    defer allocator.free(weight);
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    var timer = std.time.Timer.start() catch unreachable;
    for (0..repeats) |_| cpu.math.rmsnormContiguous(out, input, weight, null, .f32, dims.rows, dims.cols, 1e-6, 0.0);
    const cold_ns = timer.read();

    for (0..cfg.warmup) |_| {
        for (0..repeats) |_| cpu.math.rmsnormContiguous(out, input, weight, null, .f32, dims.rows, dims.cols, 1e-6, 0.0);
    }
    for (samples) |*sample| {
        timer = std.time.Timer.start() catch unreachable;
        for (0..repeats) |_| cpu.math.rmsnormContiguous(out, input, weight, null, .f32, dims.rows, dims.cols, 1e-6, 0.0);
        sample.eval_ns = timer.read();
    }

    return .{
        .name = "rms_f32",
        .profile = cfg.profile,
        .samples = samples,
        .cold_ns = cold_ns,
        .sample_loops = repeats,
        .flops_per_iter = 4 * elems * repeats,
        .bytes_per_iter = @as(u64, @intCast((2 * elems + dims.cols) * @sizeOf(f32) * repeats)),
        .note = "rmsnorm contiguous baseline",
    };
}

fn runRmsNormWeightDType(allocator: std.mem.Allocator, cfg: RunConfig, comptime weight_dtype: dtype.DType) !ScenarioResult {
    std.debug.assert(weight_dtype == .bf16 or weight_dtype == .f16);
    const dims = profileSoftmaxDims(cfg.profile);
    const scenario_kind: Scenario = if (weight_dtype == .bf16) .rmsnorm_bf16_weight else .rmsnorm_f16_weight;
    const repeats = sampleLoops(scenario_kind, cfg.profile);
    const elems = dims.rows * dims.cols;
    const input = try allocFilledF32(allocator, elems, 63);
    defer allocator.free(input);
    const out = try allocZeroF32(allocator, elems);
    defer allocator.free(out);
    const weight = try allocator.alloc(u16, dims.cols);
    defer allocator.free(weight);
    switch (weight_dtype) {
        .bf16 => fillU16SliceAsBf16(weight, 64),
        .f16 => fillU16SliceAsF16(weight, 64),
        else => unreachable,
    }
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    var timer = std.time.Timer.start() catch unreachable;
    for (0..repeats) |_| cpu.math.rmsnormContiguous(out, input, null, weight, weight_dtype, dims.rows, dims.cols, 1e-6, 0.0);
    const cold_ns = timer.read();

    for (0..cfg.warmup) |_| {
        for (0..repeats) |_| cpu.math.rmsnormContiguous(out, input, null, weight, weight_dtype, dims.rows, dims.cols, 1e-6, 0.0);
    }
    for (samples) |*sample| {
        timer = std.time.Timer.start() catch unreachable;
        for (0..repeats) |_| cpu.math.rmsnormContiguous(out, input, null, weight, weight_dtype, dims.rows, dims.cols, 1e-6, 0.0);
        sample.eval_ns = timer.read();
    }

    return .{
        .name = if (weight_dtype == .bf16) "rms_wbf16" else "rms_wf16",
        .profile = cfg.profile,
        .samples = samples,
        .cold_ns = cold_ns,
        .sample_loops = repeats,
        .flops_per_iter = 4 * elems * repeats,
        .bytes_per_iter = @as(u64, @intCast((2 * elems * @sizeOf(f32) + dims.cols * @sizeOf(u16)) * repeats)),
        .note = if (weight_dtype == .bf16) "rmsnorm with bf16 weights" else "rmsnorm with f16 weights",
    };
}

pub fn runRmsNormBf16Weight(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runRmsNormWeightDType(allocator, cfg, .bf16);
}

pub fn runRmsNormF16Weight(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runRmsNormWeightDType(allocator, cfg, .f16);
}

pub fn runSoftmaxF32(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    const dims = profileSoftmaxDims(cfg.profile);
    const repeats = sampleLoops(.softmax_f32, cfg.profile);
    const elems = dims.rows * dims.cols;
    const input = try allocFilledF32(allocator, elems, 7);
    defer allocator.free(input);
    const out = try allocZeroF32(allocator, elems);
    defer allocator.free(out);
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    var timer = std.time.Timer.start() catch unreachable;
    for (0..repeats) |_| cpu.math.softmaxContiguous(out, input, dims.rows, dims.cols);
    const cold_ns = timer.read();

    for (0..cfg.warmup) |_| for (0..repeats) |_| cpu.math.softmaxContiguous(out, input, dims.rows, dims.cols);
    for (samples) |*sample| {
        timer = std.time.Timer.start() catch unreachable;
        for (0..repeats) |_| cpu.math.softmaxContiguous(out, input, dims.rows, dims.cols);
        sample.eval_ns = timer.read();
    }

    return .{
        .name = "softmax_f32",
        .profile = cfg.profile,
        .samples = samples,
        .cold_ns = cold_ns,
        .sample_loops = repeats,
        .flops_per_iter = 5 * elems * repeats,
        .bytes_per_iter = 2 * elems * @sizeOf(f32) * repeats,
        .note = "softmax contiguous baseline",
    };
}

pub fn runShortconvDecodeF32(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    const dims = profileShortConvDims(cfg.profile);
    const repeats = sampleLoops(.shortconv_decode_f32, cfg.profile);
    const state_elems = dims.conv_dim * dims.d_conv;
    const b_gate = try allocFilledF32(allocator, dims.conv_dim, 8);
    defer allocator.free(b_gate);
    const x_proj = try allocFilledF32(allocator, dims.conv_dim, 9);
    defer allocator.free(x_proj);
    const baseline_state = try allocFilledF32(allocator, state_elems, 10);
    defer allocator.free(baseline_state);
    const state = try allocator.alloc(f32, state_elems);
    defer allocator.free(state);
    const weight_t = try allocFilledF32(allocator, state_elems, 11);
    defer allocator.free(weight_t);
    const out = try allocZeroF32(allocator, dims.conv_dim);
    defer allocator.free(out);
    const bias = try allocFilledF32(allocator, dims.conv_dim, 12);
    defer allocator.free(bias);
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    @memcpy(state, baseline_state);
    var timer = std.time.Timer.start() catch unreachable;
    for (0..repeats) |_| cpu.conv1d_depthwise.runTimeMajor(b_gate, x_proj, state, weight_t, out, bias, dims.conv_dim, dims.d_conv);
    const cold_ns = timer.read();

    for (0..cfg.warmup) |_| {
        @memcpy(state, baseline_state);
        for (0..repeats) |_| cpu.conv1d_depthwise.runTimeMajor(b_gate, x_proj, state, weight_t, out, bias, dims.conv_dim, dims.d_conv);
    }
    for (samples) |*sample| {
        @memcpy(state, baseline_state);
        timer = std.time.Timer.start() catch unreachable;
        for (0..repeats) |_| cpu.conv1d_depthwise.runTimeMajor(b_gate, x_proj, state, weight_t, out, bias, dims.conv_dim, dims.d_conv);
        sample.eval_ns = timer.read();
    }

    const flops = dims.conv_dim * (2 * dims.d_conv + 2);
    const bytes = (4 * dims.conv_dim + 2 * state_elems) * @sizeOf(f32);
    return .{
        .name = "shortconv_f32",
        .profile = cfg.profile,
        .samples = samples,
        .cold_ns = cold_ns,
        .sample_loops = repeats,
        .flops_per_iter = @intCast(flops * repeats),
        .bytes_per_iter = @intCast(bytes * repeats),
        .note = "time-major shortconv decode baseline",
    };
}

pub fn runMambaScanF32(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    const dims = profileScanDims(cfg.profile);
    const repeats = sampleLoops(.mamba_scan_f32, cfg.profile);
    const d_inner = dims.dInner();
    const state_elems = dims.stateElems();
    const grouped_elems = dims.groupStateElems();
    const baseline_state = try allocFilledF32(allocator, state_elems, 13);
    defer allocator.free(baseline_state);
    const ssm_state = try allocator.alloc(f32, state_elems);
    defer allocator.free(ssm_state);
    const ssm_out = try allocZeroF32(allocator, d_inner);
    defer allocator.free(ssm_out);
    const x_conv_out = try allocFilledF32(allocator, d_inner, 14);
    defer allocator.free(x_conv_out);
    const b_raw = try allocFilledF32(allocator, grouped_elems, 15);
    defer allocator.free(b_raw);
    const c_raw = try allocFilledF32(allocator, grouped_elems, 16);
    defer allocator.free(c_raw);
    const a_log = try allocFilledF32(allocator, dims.n_heads, 17);
    defer allocator.free(a_log);
    const d_skip = try allocFilledF32(allocator, dims.n_heads, 18);
    defer allocator.free(d_skip);
    const dt = try allocFilledF32(allocator, dims.n_heads, 19);
    defer allocator.free(dt);
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    @memcpy(ssm_state, baseline_state);
    var timer = std.time.Timer.start() catch unreachable;
    for (0..repeats) |_| cpu.state_space.scanStep(cpu.simd.ssm_scan.stateScanF32, ssm_state, ssm_out, x_conv_out, b_raw, c_raw, a_log, d_skip, dt, dims.d_head, dims.d_state, dims.n_heads, dims.n_groups);
    const cold_ns = timer.read();

    for (0..cfg.warmup) |_| {
        @memcpy(ssm_state, baseline_state);
        for (0..repeats) |_| cpu.state_space.scanStep(cpu.simd.ssm_scan.stateScanF32, ssm_state, ssm_out, x_conv_out, b_raw, c_raw, a_log, d_skip, dt, dims.d_head, dims.d_state, dims.n_heads, dims.n_groups);
    }
    for (samples) |*sample| {
        @memcpy(ssm_state, baseline_state);
        timer = std.time.Timer.start() catch unreachable;
        for (0..repeats) |_| cpu.state_space.scanStep(cpu.simd.ssm_scan.stateScanF32, ssm_state, ssm_out, x_conv_out, b_raw, c_raw, a_log, d_skip, dt, dims.d_head, dims.d_state, dims.n_heads, dims.n_groups);
        sample.eval_ns = timer.read();
    }

    const flops = dims.n_heads * dims.d_head * (3 * dims.d_state + 2);
    const bytes = (3 * d_inner + 2 * grouped_elems + 2 * dims.n_heads + 2 * state_elems) * @sizeOf(f32);
    return .{
        .name = "mamba_scan_f32",
        .profile = cfg.profile,
        .samples = samples,
        .cold_ns = cold_ns,
        .sample_loops = repeats,
        .flops_per_iter = @intCast(flops * repeats),
        .bytes_per_iter = @intCast(bytes * repeats),
        .note = "existing Mamba scan baseline",
    };
}

pub fn runGatedDeltaStepF32(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    const dims = profileGatedDeltaDims(cfg.profile);
    const repeats = sampleLoops(.gated_delta_step_f32, cfg.profile);
    const d_inner = dims.dInner();
    const state_elems = dims.stateElems();

    const query = try allocFilledF32(allocator, d_inner, 21);
    defer allocator.free(query);
    const key = try allocFilledF32(allocator, d_inner, 22);
    defer allocator.free(key);
    const value = try allocFilledF32(allocator, d_inner, 23);
    defer allocator.free(value);
    const beta_raw = try allocFilledF32(allocator, dims.n_heads, 24);
    defer allocator.free(beta_raw);
    const a_raw = try allocFilledF32(allocator, dims.n_heads, 25);
    defer allocator.free(a_raw);
    const a_log = try allocFilledF32(allocator, dims.n_heads, 26);
    defer allocator.free(a_log);
    const dt_bias = try allocFilledF32(allocator, dims.n_heads, 27);
    defer allocator.free(dt_bias);
    const baseline_state = try allocFilledF32(allocator, state_elems, 28);
    defer allocator.free(baseline_state);
    const ssm_state = try allocator.alloc(f32, state_elems);
    defer allocator.free(ssm_state);
    const kv_mem = try allocZeroF32(allocator, d_inner);
    defer allocator.free(kv_mem);
    const ssm_out = try allocZeroF32(allocator, d_inner);
    defer allocator.free(ssm_out);
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    @memcpy(ssm_state, baseline_state);
    var timer = std.time.Timer.start() catch unreachable;
    for (0..repeats) |_| try cpu.gated_delta.runStateSpaceStep(kv_mem, ssm_out, ssm_state, query, key, value, beta_raw, a_raw, a_log, dt_bias, dims.n_heads, dims.d_head);
    const cold_ns = timer.read();

    for (0..cfg.warmup) |_| {
        @memcpy(ssm_state, baseline_state);
        @memset(kv_mem, 0.0);
        @memset(ssm_out, 0.0);
        for (0..repeats) |_| try cpu.gated_delta.runStateSpaceStep(kv_mem, ssm_out, ssm_state, query, key, value, beta_raw, a_raw, a_log, dt_bias, dims.n_heads, dims.d_head);
    }
    for (samples) |*sample| {
        @memcpy(ssm_state, baseline_state);
        @memset(kv_mem, 0.0);
        @memset(ssm_out, 0.0);
        timer = std.time.Timer.start() catch unreachable;
        for (0..repeats) |_| try cpu.gated_delta.runStateSpaceStep(kv_mem, ssm_out, ssm_state, query, key, value, beta_raw, a_raw, a_log, dt_bias, dims.n_heads, dims.d_head);
        sample.eval_ns = timer.read();
    }

    const flops = dims.n_heads * (6 * dims.d_head * dims.d_head + 2 * dims.d_head);
    const bytes = (6 * d_inner + 4 * dims.n_heads + 3 * state_elems) * @sizeOf(f32);
    return .{
        .name = "gdelta_step_f32",
        .profile = cfg.profile,
        .samples = samples,
        .cold_ns = cold_ns,
        .sample_loops = repeats,
        .flops_per_iter = @intCast(flops * repeats),
        .bytes_per_iter = @intCast(bytes * repeats),
        .note = "Qwen3.5 recurrent step",
    };
}

pub fn runGatedDeltaConvF32(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    const dims = profileGatedDeltaDims(cfg.profile);
    const repeats = sampleLoops(.gated_delta_conv_f32, cfg.profile);
    const qkv_len = 3 * dims.dInner();
    const state_elems = qkv_len * dims.d_conv;

    const baseline_values = try allocFilledF32(allocator, qkv_len, 201);
    defer allocator.free(baseline_values);
    const values = try allocator.alloc(f32, qkv_len);
    defer allocator.free(values);
    const state = try allocZeroF32(allocator, state_elems);
    defer allocator.free(state);
    const weight_channel_major = try allocFilledF32(allocator, state_elems, 202);
    defer allocator.free(weight_channel_major);
    const weight_time_major = try allocator.alloc(f32, state_elems);
    defer allocator.free(weight_time_major);
    try cpu.conv1d_depthwise.transposeChannelMajorToTimeMajor(weight_channel_major, weight_time_major, qkv_len, dims.d_conv);
    const bias = try allocFilledF32(allocator, qkv_len, 203);
    defer allocator.free(bias);
    const out = try allocZeroF32(allocator, qkv_len);
    defer allocator.free(out);
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    @memcpy(values, baseline_values);
    @memset(state, 0.0);
    var timer = std.time.Timer.start() catch unreachable;
    for (0..repeats) |_| cpu.conv1d_depthwise.runTimeMajorValues(values, state, weight_time_major, out, bias, qkv_len, dims.d_conv);
    const cold_ns = timer.read();

    for (0..cfg.warmup) |_| {
        @memcpy(values, baseline_values);
        @memset(state, 0.0);
        for (0..repeats) |_| cpu.conv1d_depthwise.runTimeMajorValues(values, state, weight_time_major, out, bias, qkv_len, dims.d_conv);
    }
    for (samples) |*sample| {
        @memcpy(values, baseline_values);
        @memset(state, 0.0);
        timer = std.time.Timer.start() catch unreachable;
        for (0..repeats) |_| cpu.conv1d_depthwise.runTimeMajorValues(values, state, weight_time_major, out, bias, qkv_len, dims.d_conv);
        sample.eval_ns = timer.read();
    }

    const flops = qkv_len * (2 * dims.d_conv + 1);
    const bytes = (3 * state_elems + 4 * qkv_len) * @sizeOf(f32);
    return .{
        .name = "gdelta_conv_f32",
        .profile = cfg.profile,
        .samples = samples,
        .cold_ns = cold_ns,
        .sample_loops = repeats,
        .flops_per_iter = @intCast(flops * repeats),
        .bytes_per_iter = @intCast(bytes * repeats),
        .note = "Qwen3.5 depthwise conv step",
    };
}

pub fn runGatedDeltaQkNormF32(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    const dims = profileGatedDeltaDims(cfg.profile);
    const repeats = sampleLoops(.gated_delta_qk_norm_f32, cfg.profile);
    const d_inner = dims.dInner();
    const baseline_query = try allocFilledF32(allocator, d_inner, 211);
    defer allocator.free(baseline_query);
    const baseline_key = try allocFilledF32(allocator, d_inner, 212);
    defer allocator.free(baseline_key);
    const query = try allocator.alloc(f32, d_inner);
    defer allocator.free(query);
    const key = try allocator.alloc(f32, d_inner);
    defer allocator.free(key);
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    @memcpy(query, baseline_query);
    @memcpy(key, baseline_key);
    var timer = std.time.Timer.start() catch unreachable;
    for (0..repeats) |_| try cpu.gated_delta.normalizeQueryKeyInPlace(query, key, dims.n_heads, dims.d_head);
    const cold_ns = timer.read();

    for (0..cfg.warmup) |_| {
        @memcpy(query, baseline_query);
        @memcpy(key, baseline_key);
        for (0..repeats) |_| try cpu.gated_delta.normalizeQueryKeyInPlace(query, key, dims.n_heads, dims.d_head);
    }
    for (samples) |*sample| {
        @memcpy(query, baseline_query);
        @memcpy(key, baseline_key);
        timer = std.time.Timer.start() catch unreachable;
        for (0..repeats) |_| try cpu.gated_delta.normalizeQueryKeyInPlace(query, key, dims.n_heads, dims.d_head);
        sample.eval_ns = timer.read();
    }

    const flops = 6 * d_inner;
    const bytes = 4 * d_inner * @sizeOf(f32);
    return .{
        .name = "gdelta_qk_norm_f32",
        .profile = cfg.profile,
        .samples = samples,
        .cold_ns = cold_ns,
        .sample_loops = repeats,
        .flops_per_iter = @intCast(flops * repeats),
        .bytes_per_iter = @intCast(bytes * repeats),
        .note = "Qwen3.5 q/k normalization",
    };
}

pub fn runGatedDeltaNormF32(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    const dims = profileGatedDeltaDims(cfg.profile);
    const repeats = sampleLoops(.gated_delta_norm_f32, cfg.profile);
    const d_inner = dims.dInner();
    const baseline_values = try allocFilledF32(allocator, d_inner, 31);
    defer allocator.free(baseline_values);
    const values = try allocator.alloc(f32, d_inner);
    defer allocator.free(values);
    const gate = try allocFilledF32(allocator, d_inner, 32);
    defer allocator.free(gate);
    const norm_weight = try allocFilledF32(allocator, d_inner, 33);
    defer allocator.free(norm_weight);
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    @memcpy(values, baseline_values);
    var timer = std.time.Timer.start() catch unreachable;
    for (0..repeats) |_| try cpu.gated_delta.applyGatedRmsNormInPlace(values, gate, norm_weight);
    const cold_ns = timer.read();

    for (0..cfg.warmup) |_| {
        @memcpy(values, baseline_values);
        for (0..repeats) |_| try cpu.gated_delta.applyGatedRmsNormInPlace(values, gate, norm_weight);
    }
    for (samples) |*sample| {
        @memcpy(values, baseline_values);
        timer = std.time.Timer.start() catch unreachable;
        for (0..repeats) |_| try cpu.gated_delta.applyGatedRmsNormInPlace(values, gate, norm_weight);
        sample.eval_ns = timer.read();
    }

    return .{
        .name = "gdelta_norm_f32",
        .profile = cfg.profile,
        .samples = samples,
        .cold_ns = cold_ns,
        .sample_loops = repeats,
        .flops_per_iter = 6 * d_inner * repeats,
        .bytes_per_iter = @as(u64, @intCast(d_inner * 3 * @sizeOf(f32) * repeats)),
        .note = "Qwen3.5 gated RMSNorm tail",
    };
}

fn runAttentionGate(allocator: std.mem.Allocator, cfg: RunConfig, long: bool) !ScenarioResult {
    const dims = profileAttentionDims(cfg.profile, long);
    const repeats = sampleLoops(if (long) .gated_attention_gate_long_f32 else .gated_attention_gate_f32, cfg.profile);
    const query_dim = dims.queryDim();
    const query_projection_dim = dims.queryProjectionDim();
    const elems = dims.seq_len * query_dim;
    const baseline_context = try allocFilledF32(allocator, elems, 41);
    defer allocator.free(baseline_context);
    const context = try allocator.alloc(f32, elems);
    defer allocator.free(context);
    const query_projection = try allocFilledF32(allocator, dims.seq_len * query_projection_dim, 42);
    defer allocator.free(query_projection);
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    @memcpy(context, baseline_context);
    var timer = std.time.Timer.start() catch unreachable;
    for (0..repeats) |_| try cpu.gated_attention.applyOutputGateInPlace(context, query_projection, dims.seq_len, query_dim, query_projection_dim, dims.head_count, dims.head_dim);
    const cold_ns = timer.read();

    for (0..cfg.warmup) |_| {
        @memcpy(context, baseline_context);
        for (0..repeats) |_| try cpu.gated_attention.applyOutputGateInPlace(context, query_projection, dims.seq_len, query_dim, query_projection_dim, dims.head_count, dims.head_dim);
    }
    for (samples) |*sample| {
        @memcpy(context, baseline_context);
        timer = std.time.Timer.start() catch unreachable;
        for (0..repeats) |_| try cpu.gated_attention.applyOutputGateInPlace(context, query_projection, dims.seq_len, query_dim, query_projection_dim, dims.head_count, dims.head_dim);
        sample.eval_ns = timer.read();
    }

    return .{
        .name = if (long) "gattn_gate_long" else "gattn_gate",
        .profile = cfg.profile,
        .samples = samples,
        .cold_ns = cold_ns,
        .sample_loops = repeats,
        .flops_per_iter = @as(u64, @intCast(6 * elems * repeats)),
        .bytes_per_iter = @as(u64, @intCast((2 * elems + dims.seq_len * query_projection_dim) * @sizeOf(f32) * repeats)),
        .note = if (long) "long-sequence gated attention output gate" else "single-step gated attention output gate",
    };
}

pub fn runGatedAttentionGateF32(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runAttentionGate(allocator, cfg, false);
}

pub fn runGatedAttentionGateLongF32(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runAttentionGate(allocator, cfg, true);
}

fn runRopeDType(allocator: std.mem.Allocator, cfg: RunConfig, comptime rope_dtype: dtype.DType) !ScenarioResult {
    std.debug.assert(rope_dtype == .f32 or rope_dtype == .f16 or rope_dtype == .bf16);
    const dims = profileRopeDims(cfg.profile);
    const scenario_kind: Scenario = switch (rope_dtype) {
        .f32 => .rope_f32,
        .f16 => .rope_f16,
        .bf16 => .rope_bf16,
        else => unreachable,
    };
    const repeats = sampleLoops(scenario_kind, cfg.profile);
    const elems = dims.batch * dims.groups * dims.seq_len * dims.width;
    const half = dims.width / 2;

    var q = try tensor.OwnedTensor.init(allocator, rope_dtype, &.{ dims.batch, dims.groups, dims.seq_len, dims.width });
    defer q.deinit();
    var k = try tensor.OwnedTensor.init(allocator, rope_dtype, &.{ dims.batch, dims.groups, dims.seq_len, dims.width });
    defer k.deinit();
    fillTensorLike(q.data, rope_dtype, 91);
    fillTensorLike(k.data, rope_dtype, 92);

    var cos_view = try allocRopeFreqTable(allocator, rope_dtype, dims.seq_len, half, 93);
    defer cos_view.deinit();
    var sin_view = try allocRopeFreqTable(allocator, rope_dtype, dims.seq_len, half, 94);
    defer sin_view.deinit();

    var q_tensor = q.view();
    var k_tensor = k.view();
    var cos_tensor = cos_view.view();
    var sin_tensor = sin_view.view();
    const q_view = cpu.tensor_view.fromTensor(tensor.Tensor, &q_tensor);
    const k_view = cpu.tensor_view.fromTensor(tensor.Tensor, &k_tensor);
    const cos_tv = cpu.tensor_view.fromTensor(tensor.Tensor, &cos_tensor);
    const sin_tv = cpu.tensor_view.fromTensor(tensor.Tensor, &sin_tensor);
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    var timer = std.time.Timer.start() catch unreachable;
    for (0..repeats) |_| cpu.linalg_sdpa.applyRope(q_view, k_view, cos_tv, sin_tv);
    const cold_ns = timer.read();

    for (0..cfg.warmup) |_| for (0..repeats) |_| cpu.linalg_sdpa.applyRope(q_view, k_view, cos_tv, sin_tv);
    for (samples) |*sample| {
        timer = std.time.Timer.start() catch unreachable;
        for (0..repeats) |_| cpu.linalg_sdpa.applyRope(q_view, k_view, cos_tv, sin_tv);
        sample.eval_ns = timer.read();
    }

    const elem_bytes = switch (rope_dtype) {
        .f32 => @sizeOf(f32),
        .f16, .bf16 => @sizeOf(u16),
        else => unreachable,
    };
    return .{
        .name = switch (rope_dtype) {
            .f32 => "rope_f32",
            .f16 => "rope_f16",
            .bf16 => "rope_bf16",
            else => unreachable,
        },
        .profile = cfg.profile,
        .samples = samples,
        .cold_ns = cold_ns,
        .sample_loops = repeats,
        .flops_per_iter = @as(u64, @intCast(6 * elems * repeats)),
        .bytes_per_iter = @as(u64, @intCast((2 * elems * elem_bytes + 2 * dims.seq_len * half * elem_bytes) * repeats)),
        .note = "rotary in-place transform",
    };
}

pub fn runRopeF32(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runRopeDType(allocator, cfg, .f32);
}

pub fn runRopeF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runRopeDType(allocator, cfg, .f16);
}

pub fn runRopeBf16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runRopeDType(allocator, cfg, .bf16);
}

fn runSdpaDType(allocator: std.mem.Allocator, cfg: RunConfig, comptime sdpa_dtype: dtype.DType) !ScenarioResult {
    std.debug.assert(sdpa_dtype == .f32 or sdpa_dtype == .f16 or sdpa_dtype == .bf16);
    const dims = profileSdpaDims(cfg.profile);
    const scenario_kind: Scenario = switch (sdpa_dtype) {
        .f32 => .sdpa_f32,
        .f16 => .sdpa_f16,
        .bf16 => .sdpa_bf16,
        else => unreachable,
    };
    const repeats = sampleLoops(scenario_kind, cfg.profile);

    var q = try tensor.OwnedTensor.init(allocator, sdpa_dtype, &.{ dims.batch, dims.groups, dims.query_steps, dims.width });
    defer q.deinit();
    var k = try tensor.OwnedTensor.init(allocator, sdpa_dtype, &.{ dims.batch, dims.groups, dims.key_steps, dims.width });
    defer k.deinit();
    var v = try tensor.OwnedTensor.init(allocator, sdpa_dtype, &.{ dims.batch, dims.groups, dims.key_steps, dims.width });
    defer v.deinit();
    var out = try tensor.OwnedTensor.init(allocator, sdpa_dtype, &.{ dims.batch, dims.groups, dims.query_steps, dims.width });
    defer out.deinit();
    fillTensorLike(q.data, sdpa_dtype, 101);
    fillTensorLike(k.data, sdpa_dtype, 102);
    fillTensorLike(v.data, sdpa_dtype, 103);

    var q_tensor = q.view();
    var k_tensor = k.view();
    var v_tensor = v.view();
    var out_tensor = out.view();
    const q_tv = cpu.tensor_view.fromTensor(tensor.Tensor, &q_tensor);
    const k_tv = cpu.tensor_view.fromTensor(tensor.Tensor, &k_tensor);
    const v_tv = cpu.tensor_view.fromTensor(tensor.Tensor, &v_tensor);
    const out_tv = cpu.tensor_view.fromTensor(tensor.Tensor, &out_tensor);
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(dims.width)));
    var timer = std.time.Timer.start() catch unreachable;
    for (0..repeats) |_| try cpu.linalg_sdpa.sdpa(out_tv, q_tv, k_tv, v_tv, null, scale, allocator);
    const cold_ns = timer.read();

    for (0..cfg.warmup) |_| for (0..repeats) |_| try cpu.linalg_sdpa.sdpa(out_tv, q_tv, k_tv, v_tv, null, scale, allocator);
    for (samples) |*sample| {
        timer = std.time.Timer.start() catch unreachable;
        for (0..repeats) |_| try cpu.linalg_sdpa.sdpa(out_tv, q_tv, k_tv, v_tv, null, scale, allocator);
        sample.eval_ns = timer.read();
    }

    const elem_count = dims.batch * dims.groups * (dims.query_steps * dims.width + 2 * dims.key_steps * dims.width);
    const out_elems = dims.batch * dims.groups * dims.query_steps * dims.width;
    const elem_bytes = switch (sdpa_dtype) {
        .f32 => @sizeOf(f32),
        .f16, .bf16 => @sizeOf(u16),
        else => unreachable,
    };
    const qk = 2 * dims.batch * dims.groups * dims.query_steps * dims.key_steps * dims.width;
    const av = 2 * dims.batch * dims.groups * dims.query_steps * dims.key_steps * dims.width;
    const soft = 5 * dims.batch * dims.groups * dims.query_steps * dims.key_steps;
    return .{
        .name = switch (sdpa_dtype) {
            .f32 => "sdpa_f32",
            .f16 => "sdpa_f16",
            .bf16 => "sdpa_bf16",
            else => unreachable,
        },
        .profile = cfg.profile,
        .samples = samples,
        .cold_ns = cold_ns,
        .sample_loops = repeats,
        .flops_per_iter = @intCast((qk + av + soft) * repeats),
        .bytes_per_iter = @intCast((elem_count + out_elems) * elem_bytes * repeats),
        .note = "scaled dot-product attention",
    };
}

pub fn runSdpaF32(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runSdpaDType(allocator, cfg, .f32);
}

pub fn runSdpaF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runSdpaDType(allocator, cfg, .f16);
}

pub fn runSdpaBf16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runSdpaDType(allocator, cfg, .bf16);
}

fn runDecodeScenario(allocator: std.mem.Allocator, cfg: RunConfig, which: Scenario) !ScenarioResult {
    const dims = profileDecodeDims(cfg.profile);
    const repeats = sampleLoops(which, cfg.profile);
    const row_indices = try allocator.alloc(u32, dims.take_rows);
    defer allocator.free(row_indices);
    for (row_indices, 0..) |*slot, idx| slot.* = @intCast((idx * 17) % dims.row_count);
    const out = try allocator.alloc(f32, dims.take_rows * dims.row_width);
    defer allocator.free(out);
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    switch (which) {
        .decode_bf16 => {
            const src = try allocator.alloc(u16, dims.row_count * dims.row_width);
            defer allocator.free(src);
            fillU16SliceAsBf16(src, 111);
            var timer = std.time.Timer.start() catch unreachable;
            for (0..repeats) |_| try cpu.quant_decode.gatherDecodeBf16Rows(src, dims.row_count, dims.row_width, row_indices, out);
            const cold_ns = timer.read();
            for (0..cfg.warmup) |_| for (0..repeats) |_| try cpu.quant_decode.gatherDecodeBf16Rows(src, dims.row_count, dims.row_width, row_indices, out);
            for (samples) |*sample| {
                timer = std.time.Timer.start() catch unreachable;
                for (0..repeats) |_| try cpu.quant_decode.gatherDecodeBf16Rows(src, dims.row_count, dims.row_width, row_indices, out);
                sample.eval_ns = timer.read();
            }
            return .{
                .name = "decode_bf16",
                .profile = cfg.profile,
                .samples = samples,
                .cold_ns = cold_ns,
                .sample_loops = repeats,
                .flops_per_iter = dims.take_rows * dims.row_width * repeats,
                .bytes_per_iter = @intCast((dims.take_rows * dims.row_width * (@sizeOf(u16) + @sizeOf(f32))) * repeats),
                .note = "row gather + bf16 decode",
            };
        },
        .decode_u4, .decode_u8 => {
            const packed_words_per_row = if (which == .decode_u4) dims.row_width / 8 else dims.row_width / 4;
            const packed_words = try allocator.alloc(u32, dims.row_count * packed_words_per_row);
            defer allocator.free(packed_words);
            var state: u64 = 1125899906842597;
            for (packed_words) |*slot| {
                state = state *% 2862933555777941757 +% 3037000493;
                slot.* = @truncate(state);
            }
            const group_count = dims.row_width / dims.group_size;
            const scales = try allocator.alloc(u16, dims.row_count * group_count);
            defer allocator.free(scales);
            const biases = try allocator.alloc(u16, dims.row_count * group_count);
            defer allocator.free(biases);
            fillU16SliceAsBf16(scales, 113);
            fillU16SliceAsBf16(biases, 114);

            var timer = std.time.Timer.start() catch unreachable;
            for (0..repeats) |_| {
                if (which == .decode_u4)
                    try cpu.quant_decode.gatherDecodeGroupedAffineU4Rows(packed_words, scales, biases, .bf16, dims.group_size, dims.row_count, dims.row_width, row_indices, out)
                else
                    try cpu.quant_decode.gatherDecodeGroupedAffineU8Rows(packed_words, scales, biases, .bf16, dims.group_size, dims.row_count, dims.row_width, row_indices, out);
            }
            const cold_ns = timer.read();
            for (0..cfg.warmup) |_| {
                for (0..repeats) |_| {
                    if (which == .decode_u4)
                        try cpu.quant_decode.gatherDecodeGroupedAffineU4Rows(packed_words, scales, biases, .bf16, dims.group_size, dims.row_count, dims.row_width, row_indices, out)
                    else
                        try cpu.quant_decode.gatherDecodeGroupedAffineU8Rows(packed_words, scales, biases, .bf16, dims.group_size, dims.row_count, dims.row_width, row_indices, out);
                }
            }
            for (samples) |*sample| {
                timer = std.time.Timer.start() catch unreachable;
                for (0..repeats) |_| {
                    if (which == .decode_u4)
                        try cpu.quant_decode.gatherDecodeGroupedAffineU4Rows(packed_words, scales, biases, .bf16, dims.group_size, dims.row_count, dims.row_width, row_indices, out)
                    else
                        try cpu.quant_decode.gatherDecodeGroupedAffineU8Rows(packed_words, scales, biases, .bf16, dims.group_size, dims.row_count, dims.row_width, row_indices, out);
                }
                sample.eval_ns = timer.read();
            }
            const packed_bytes = if (which == .decode_u4) dims.take_rows * dims.row_width / 2 else dims.take_rows * dims.row_width;
            const meta_bytes = 2 * dims.take_rows * group_count * @sizeOf(u16);
            return .{
                .name = if (which == .decode_u4) "decode_u4" else "decode_u8",
                .profile = cfg.profile,
                .samples = samples,
                .cold_ns = cold_ns,
                .sample_loops = repeats,
                .flops_per_iter = dims.take_rows * dims.row_width * repeats,
                .bytes_per_iter = @intCast((packed_bytes + meta_bytes + dims.take_rows * dims.row_width * @sizeOf(f32)) * repeats),
                .note = if (which == .decode_u4) "row gather + grouped-affine u4 decode" else "row gather + grouped-affine u8 decode",
            };
        },
        else => unreachable,
    }
}

pub fn runDecodeBf16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runDecodeScenario(allocator, cfg, .decode_bf16);
}

pub fn runDecodeU4(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runDecodeScenario(allocator, cfg, .decode_u4);
}

pub fn runDecodeU8(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runDecodeScenario(allocator, cfg, .decode_u8);
}

fn runMatmul(allocator: std.mem.Allocator, cfg: RunConfig, micro: bool) !ScenarioResult {
    const dims = profileMatmulDims(cfg.profile, micro);
    const repeats = sampleLoops(if (micro) .micro_matmul_f32 else .matmul_throughput_f32, cfg.profile);
    var a = try tensor.OwnedTensor.init(allocator, .f32, &.{ dims.m, dims.k });
    defer a.deinit();
    var b = try tensor.OwnedTensor.init(allocator, .f32, &.{ dims.k, dims.n });
    defer b.deinit();
    var out = try tensor.OwnedTensor.init(allocator, .f32, &.{ dims.m, dims.n });
    defer out.deinit();
    fillTensorF32(&a, 51);
    fillTensorF32(&b, 52);
    var a_view = a.view();
    var b_view = b.view();
    var out_view = out.view();
    var scratch = try cpu.matmul.MatmulScratch.init(allocator);
    defer scratch.deinit();
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    var timer = std.time.Timer.start() catch unreachable;
    for (0..repeats) |_| cpu.matmul.matmulF32(&a_view, &b_view, &out_view, &scratch);
    const cold_ns = timer.read();

    for (0..cfg.warmup) |_| for (0..repeats) |_| cpu.matmul.matmulF32(&a_view, &b_view, &out_view, &scratch);
    for (samples) |*sample| {
        timer = std.time.Timer.start() catch unreachable;
        for (0..repeats) |_| cpu.matmul.matmulF32(&a_view, &b_view, &out_view, &scratch);
        sample.eval_ns = timer.read();
    }

    const flops: u64 = @intCast(2 * dims.m * dims.k * dims.n);
    const bytes: u64 = @intCast((dims.m * dims.k + dims.k * dims.n + 2 * dims.m * dims.n) * @sizeOf(f32));
    return .{
        .name = if (micro) "micro_matmul_f32" else "matmul_thr_f32",
        .profile = cfg.profile,
        .samples = samples,
        .cold_ns = cold_ns,
        .sample_loops = repeats,
        .flops_per_iter = flops * repeats,
        .bytes_per_iter = bytes * repeats,
        .note = if (micro) "single-row micro matmul" else "sustained dense f32 matmul throughput",
    };
}

pub fn runMatmulThroughputF32(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runMatmul(allocator, cfg, false);
}

pub fn runMicroMatmulF32(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runMatmul(allocator, cfg, true);
}

fn runMatmulWeightDType(allocator: std.mem.Allocator, cfg: RunConfig, comptime weight_dtype: dtype.DType) !ScenarioResult {
    std.debug.assert(weight_dtype == .bf16 or weight_dtype == .f16);
    const dims = profileMatmulDims(cfg.profile, false);
    const scenario_kind: Scenario = if (weight_dtype == .bf16) .matmul_throughput_bf16 else .matmul_throughput_f16;
    const repeats = sampleLoops(scenario_kind, cfg.profile);
    var a = try tensor.OwnedTensor.init(allocator, .f32, &.{ dims.m, dims.k });
    defer a.deinit();
    var b = try tensor.OwnedTensor.init(allocator, weight_dtype, &.{ dims.n, dims.k });
    defer b.deinit();
    var out = try tensor.OwnedTensor.init(allocator, .f32, &.{ dims.m, dims.n });
    defer out.deinit();
    fillTensorF32(&a, 73);
    switch (weight_dtype) {
        .bf16 => fillU16SliceAsBf16(b.asSlice(u16), 74),
        .f16 => fillU16SliceAsF16(b.asSlice(u16), 74),
        else => unreachable,
    }
    var a_view = a.view();
    var b_view = b.view();
    var out_view = out.view();
    var scratch = try cpu.matmul.MatmulScratch.init(allocator);
    defer scratch.deinit();
    const dispatched = try cpu.matmul.matmulKernel(weight_dtype);
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    var timer = std.time.Timer.start() catch unreachable;
    for (0..repeats) |_| dispatched.func(&a_view, &b_view, &out_view, &scratch);
    const cold_ns = timer.read();

    for (0..cfg.warmup) |_| for (0..repeats) |_| dispatched.func(&a_view, &b_view, &out_view, &scratch);
    for (samples) |*sample| {
        timer = std.time.Timer.start() catch unreachable;
        for (0..repeats) |_| dispatched.func(&a_view, &b_view, &out_view, &scratch);
        sample.eval_ns = timer.read();
    }

    const flops: u64 = @intCast(2 * dims.m * dims.k * dims.n);
    const bytes: u64 = @intCast(dims.m * dims.k * @sizeOf(f32) + dims.n * dims.k * @sizeOf(u16) + 2 * dims.m * dims.n * @sizeOf(f32));
    return .{
        .name = if (weight_dtype == .bf16) "matmul_thr_bf16" else "matmul_thr_f16",
        .profile = cfg.profile,
        .samples = samples,
        .cold_ns = cold_ns,
        .sample_loops = repeats,
        .flops_per_iter = flops * repeats,
        .bytes_per_iter = bytes * repeats,
        .note = if (weight_dtype == .bf16) "sustained dense bf16-weight matmul" else "sustained dense f16-weight matmul",
    };
}

pub fn runMatmulThroughputBf16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runMatmulWeightDType(allocator, cfg, .bf16);
}

pub fn runMatmulThroughputF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runMatmulWeightDType(allocator, cfg, .f16);
}

fn runRoleMatmulBf16(allocator: std.mem.Allocator, cfg: RunConfig, which: Scenario) !ScenarioResult {
    const model_id = cfg.model_id orelse return error.InvalidArgument;
    const dims = try modelRoleMatmulDims(model_id, which);
    const repeats = sampleLoops(which, cfg.profile);

    var a = try tensor.OwnedTensor.init(allocator, .f32, &.{ dims.tokens, dims.hidden });
    defer a.deinit();
    var b = try tensor.OwnedTensor.init(allocator, .bf16, &.{ dims.out, dims.hidden });
    defer b.deinit();
    var out = try tensor.OwnedTensor.init(allocator, .f32, &.{ dims.tokens, dims.out });
    defer out.deinit();
    fillTensorF32(&a, 141);
    fillU16SliceAsBf16(b.asSlice(u16), 142);

    var a_view = a.view();
    var b_view = b.view();
    var out_view = out.view();
    var scratch = try cpu.matmul.MatmulScratch.init(allocator);
    defer scratch.deinit();
    const dispatched = try cpu.matmul.matmulKernel(.bf16);
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    var timer = std.time.Timer.start() catch unreachable;
    for (0..repeats) |_| dispatched.func(&a_view, &b_view, &out_view, &scratch);
    const cold_ns = timer.read();

    for (0..cfg.warmup) |_| for (0..repeats) |_| dispatched.func(&a_view, &b_view, &out_view, &scratch);
    for (samples) |*sample| {
        timer = std.time.Timer.start() catch unreachable;
        for (0..repeats) |_| dispatched.func(&a_view, &b_view, &out_view, &scratch);
        sample.eval_ns = timer.read();
    }

    const flops: u64 = @intCast(2 * dims.tokens * dims.hidden * dims.out);
    const bytes: u64 = @intCast(dims.tokens * dims.hidden * @sizeOf(f32) + dims.out * dims.hidden * @sizeOf(u16) + 2 * dims.tokens * dims.out * @sizeOf(f32));
    return .{
        .name = try roleBenchRowName(which),
        .profile = cfg.profile,
        .samples = samples,
        .cold_ns = cold_ns,
        .sample_loops = repeats,
        .flops_per_iter = flops * repeats,
        .bytes_per_iter = bytes * repeats,
        .note = switch (which) {
            .role_attn_q_bf16 => "xray role: attn.q -> bf16 projection",
            .role_attn_k_bf16 => "xray role: attn.k -> bf16 projection",
            .role_attn_v_bf16 => "xray role: attn.v -> bf16 projection",
            .role_attn_out_bf16 => "xray role: attn.out -> bf16 projection",
            .role_ffn_gate_bf16 => "xray role: ffn.gate -> bf16 projection",
            .role_ffn_down_bf16 => "xray role: ffn.down -> bf16 projection",
            else => unreachable,
        },
    };
}

pub fn runRoleAttnQBf16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runRoleMatmulBf16(allocator, cfg, .role_attn_q_bf16);
}

pub fn runRoleAttnKBf16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runRoleMatmulBf16(allocator, cfg, .role_attn_k_bf16);
}

pub fn runRoleAttnVBf16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runRoleMatmulBf16(allocator, cfg, .role_attn_v_bf16);
}

pub fn runRoleAttnOutBf16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runRoleMatmulBf16(allocator, cfg, .role_attn_out_bf16);
}

pub fn runRoleFfnGateBf16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runRoleMatmulBf16(allocator, cfg, .role_ffn_gate_bf16);
}

pub fn runRoleFfnDownBf16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runRoleMatmulBf16(allocator, cfg, .role_ffn_down_bf16);
}

fn runMatmulGroupedAffine(allocator: std.mem.Allocator, cfg: RunConfig, comptime weight_dtype: dtype.DType) !ScenarioResult {
    std.debug.assert(weight_dtype == .grouped_affine_u4 or weight_dtype == .grouped_affine_u8);
    const dims = profileMatmulDims(cfg.profile, false);
    const scenario_kind: Scenario = if (weight_dtype == .grouped_affine_u4) .matmul_throughput_gaffine_u4 else .matmul_throughput_gaffine_u8;
    const repeats = sampleLoops(scenario_kind, cfg.profile);
    const group_size: usize = 128;
    if (dims.k % group_size != 0) return error.InvalidShape;

    var a = try tensor.OwnedTensor.init(allocator, .f32, &.{ dims.m, dims.k });
    defer a.deinit();
    var b = try QuantOwnedTensor.init(allocator, weight_dtype, &.{ dims.n, dims.k }, group_size, .bf16, 83);
    defer b.deinit();
    var out = try tensor.OwnedTensor.init(allocator, .f32, &.{ dims.m, dims.n });
    defer out.deinit();
    fillTensorF32(&a, 84);
    var a_view = a.view();
    var out_view = out.view();
    var scratch = try cpu.matmul.MatmulScratch.init(allocator);
    defer scratch.deinit();
    const dispatched = try cpu.matmul.matmulKernel(weight_dtype);
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    var timer = std.time.Timer.start() catch unreachable;
    for (0..repeats) |_| dispatched.func(&a_view, b.view(), &out_view, &scratch);
    const cold_ns = timer.read();

    for (0..cfg.warmup) |_| for (0..repeats) |_| dispatched.func(&a_view, b.view(), &out_view, &scratch);
    for (samples) |*sample| {
        timer = std.time.Timer.start() catch unreachable;
        for (0..repeats) |_| dispatched.func(&a_view, b.view(), &out_view, &scratch);
        sample.eval_ns = timer.read();
    }

    const flops: u64 = @intCast(2 * dims.m * dims.k * dims.n);
    const packed_bytes = switch (weight_dtype) {
        .grouped_affine_u4 => (dims.n * dims.k) / 2,
        .grouped_affine_u8 => dims.n * dims.k,
        else => unreachable,
    };
    const meta_bytes = 2 * (dims.n * (dims.k / group_size) * @sizeOf(u16));
    const bytes: u64 = @intCast(dims.m * dims.k * @sizeOf(f32) + packed_bytes + meta_bytes + 2 * dims.m * dims.n * @sizeOf(f32));
    return .{
        .name = if (weight_dtype == .grouped_affine_u4) "matmul_thr_u4" else "matmul_thr_u8",
        .profile = cfg.profile,
        .samples = samples,
        .cold_ns = cold_ns,
        .sample_loops = repeats,
        .flops_per_iter = flops * repeats,
        .bytes_per_iter = bytes * repeats,
        .note = if (weight_dtype == .grouped_affine_u4) "sustained grouped-affine u4 matmul" else "sustained grouped-affine u8 matmul",
    };
}

pub fn runMatmulThroughputGroupedAffineU4(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runMatmulGroupedAffine(allocator, cfg, .grouped_affine_u4);
}

pub fn runMatmulThroughputGroupedAffineU8(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runMatmulGroupedAffine(allocator, cfg, .grouped_affine_u8);
}

test "bw gated delta profile matches qwen3.5 widths" {
    const dims = profileGatedDeltaDims(.bw);
    try std.testing.expectEqual(@as(usize, 16), dims.n_heads);
    try std.testing.expectEqual(@as(usize, 128), dims.d_head);
    try std.testing.expectEqual(@as(usize, 2048), dims.dInner());
}

test "qwen3.5 prefill role dims match xray-visible projections" {
    const q = try modelRoleMatmulDims("qwen3_5", .role_attn_q_bf16);
    try std.testing.expectEqualDeep(RoleMatmulDims{ .tokens = 14, .hidden = 1024, .out = 2048 }, q);
    const gate = try modelRoleMatmulDims("qwen3_5", .role_ffn_gate_bf16);
    try std.testing.expectEqualDeep(RoleMatmulDims{ .tokens = 14, .hidden = 1024, .out = 7168 }, gate);
    const down = try modelRoleMatmulDims("qwen3_5", .role_ffn_down_bf16);
    try std.testing.expectEqualDeep(RoleMatmulDims{ .tokens = 14, .hidden = 3584, .out = 1024 }, down);
}
