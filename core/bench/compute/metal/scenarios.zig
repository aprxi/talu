const std = @import("std");
const builtin = @import("builtin");
const main = @import("main");
const metal = main.compute.metal;
const graph = metal.graph;
const harness = @import("harness.zig");

pub const Scenario = enum {
    all,
    add_f16,
    mul_f16,
    rms_f16,
    softmax_f16,
    fused_ffn_quantized_decode_u4,
    fused_ffn_quantized_decode_u8,
    fused_ffn_dense_decode_f16,
    quantized_matmul_u4,
    quantized_matmul_u8,
    attention_decode_f16,
    shortconv_decode_bf16,
    shortconv_decode_quantized_u4,
    state_space_decode_f16,
    gated_delta_decode_f16,
    gated_delta_block_f16,
    lm_head_f16,
    lm_head_bf16,
    lm_head_host_f16,
    gated_delta_decode_quantized_u4,
    matmul_throughput_f16,
    micro_matmul_f16,
    decode_synth_f16,
    decode_dense_f16,
    decode_quantized_mix_u4,
};

pub const Profile = enum {
    ci,
    bw,
};

pub const RunConfig = struct {
    warmup: usize = 8,
    iters: usize = 24,
    profile: Profile = .bw,
};

pub const ScenarioResult = struct {
    name: []const u8,
    profile: Profile,
    samples: []harness.Sample,
    cold_first: harness.Sample,
    flops_per_iter: u64,
    bytes_per_iter: u64,
    note: []const u8,

    pub fn deinit(self: *ScenarioResult, allocator: std.mem.Allocator) void {
        allocator.free(self.samples);
    }
};

const OwnedF16 = struct {
    data: []u16,
    handle: graph.ArrayHandle,

    fn initShape(allocator: std.mem.Allocator, shape: []const i64, seed: u64) !OwnedF16 {
        var count: usize = 1;
        for (shape) |dim| count *= @intCast(dim);
        const data = try allocator.alloc(u16, count);
        fillDeterministicF16(data, seed);
        const ptr: [*]align(1) const u16 = @ptrCast(data.ptr);
        const handle = graph.createArrayF16Unaligned(ptr, data.len, shape) orelse return error.OutOfMemory;
        return .{ .data = data, .handle = handle };
    }

    fn init1D(allocator: std.mem.Allocator, len: usize, seed: u64) !OwnedF16 {
        const shape = [_]i64{@intCast(len)};
        return initShape(allocator, &shape, seed);
    }

    fn init2D(allocator: std.mem.Allocator, rows: usize, cols: usize, seed: u64) !OwnedF16 {
        const shape = [_]i64{ @intCast(rows), @intCast(cols) };
        return initShape(allocator, &shape, seed);
    }

    fn deinit(self: *OwnedF16, allocator: std.mem.Allocator) void {
        graph.freeArray(self.handle);
        allocator.free(self.data);
    }
};

const OwnedU32 = struct {
    data: []u32,
    handle: graph.ArrayHandle,

    fn init2D(allocator: std.mem.Allocator, rows: usize, cols: usize, seed: u64) !OwnedU32 {
        const count = rows * cols;
        const data = try allocator.alloc(u32, count);
        fillDeterministicU32(data, seed);
        const shape = [_]i64{ @intCast(rows), @intCast(cols) };
        const ptr: [*]align(1) const u32 = @ptrCast(data.ptr);
        const handle = graph.createArrayU32Unaligned(ptr, data.len, &shape) orelse return error.OutOfMemory;
        return .{ .data = data, .handle = handle };
    }

    fn deinit(self: *OwnedU32, allocator: std.mem.Allocator) void {
        graph.freeArray(self.handle);
        allocator.free(self.data);
    }
};

const OwnedBF16 = struct {
    data: []u16,
    handle: graph.ArrayHandle,

    fn initShape(allocator: std.mem.Allocator, shape: []const i64, seed: u64) !OwnedBF16 {
        var count: usize = 1;
        for (shape) |dim| count *= @intCast(dim);
        const data = try allocator.alloc(u16, count);
        fillDeterministicBF16(data, seed);
        const ptr: [*]align(1) const u16 = @ptrCast(data.ptr);
        const handle = graph.createArrayBF16Unaligned(ptr, data.len, shape) orelse return error.OutOfMemory;
        return .{ .data = data, .handle = handle };
    }

    fn init2D(allocator: std.mem.Allocator, rows: usize, cols: usize, seed: u64) !OwnedBF16 {
        const count = rows * cols;
        const data = try allocator.alloc(u16, count);
        fillDeterministicBF16(data, seed);
        const shape = [_]i64{ @intCast(rows), @intCast(cols) };
        const ptr: [*]align(1) const u16 = @ptrCast(data.ptr);
        const handle = graph.createArrayBF16Unaligned(ptr, data.len, &shape) orelse return error.OutOfMemory;
        return .{ .data = data, .handle = handle };
    }

    fn deinit(self: *OwnedBF16, allocator: std.mem.Allocator) void {
        graph.freeArray(self.handle);
        allocator.free(self.data);
    }
};

const OwnedBF16DenseWeight = struct {
    data: []u16,
    handle: graph.ArrayHandle,

    fn init2D(allocator: std.mem.Allocator, rows: usize, cols: usize, seed: u64) !OwnedBF16DenseWeight {
        const count = rows * cols;
        const data = try allocator.alloc(u16, count);
        fillDeterministicBF16(data, seed);
        const shape = [_]i64{ @intCast(rows), @intCast(cols) };
        const ptr: [*]align(1) const u16 = @ptrCast(data.ptr);
        const handle = graph.createArrayBF16DenseWeightUnaligned(ptr, data.len, &shape) orelse return error.OutOfMemory;
        return .{ .data = data, .handle = handle };
    }

    fn deinit(self: *OwnedBF16DenseWeight, allocator: std.mem.Allocator) void {
        graph.freeArray(self.handle);
        allocator.free(self.data);
    }
};

const OwnedBF16Norm = struct {
    data: []u16,
    handle: graph.ArrayHandle,

    fn init1D(allocator: std.mem.Allocator, len: usize, seed: u64) !OwnedBF16Norm {
        const data = try allocator.alloc(u16, len);
        fillDeterministicBF16(data, seed);
        const shape = [_]i64{@intCast(len)};
        const ptr: [*]align(1) const u16 = @ptrCast(data.ptr);
        const handle = graph.createArrayBF16NormUnaligned(ptr, data.len, &shape) orelse return error.OutOfMemory;
        return .{ .data = data, .handle = handle };
    }

    fn deinit(self: *OwnedBF16Norm, allocator: std.mem.Allocator) void {
        graph.freeArray(self.handle);
        allocator.free(self.data);
    }
};

const DecodeLayer = struct {
    wq: OwnedF16,
    wk: OwnedF16,
    wv: OwnedF16,
    wo: OwnedF16,
    w1: OwnedF16,
    w2: OwnedF16,
    w3: OwnedF16,
    norm: OwnedF16,

    fn init(allocator: std.mem.Allocator, hidden: usize, ff: usize, seed: u64) !DecodeLayer {
        return .{
            .wq = try OwnedF16.init2D(allocator, hidden, hidden, seed +% 1),
            .wk = try OwnedF16.init2D(allocator, hidden, hidden, seed +% 2),
            .wv = try OwnedF16.init2D(allocator, hidden, hidden, seed +% 3),
            .wo = try OwnedF16.init2D(allocator, hidden, hidden, seed +% 4),
            .w1 = try OwnedF16.init2D(allocator, hidden, ff, seed +% 5),
            .w2 = try OwnedF16.init2D(allocator, ff, hidden, seed +% 6),
            .w3 = try OwnedF16.init2D(allocator, hidden, ff, seed +% 7),
            .norm = try OwnedF16.init1D(allocator, hidden, seed +% 8),
        };
    }

    fn deinit(self: *DecodeLayer, allocator: std.mem.Allocator) void {
        self.wq.deinit(allocator);
        self.wk.deinit(allocator);
        self.wv.deinit(allocator);
        self.wo.deinit(allocator);
        self.w1.deinit(allocator);
        self.w2.deinit(allocator);
        self.w3.deinit(allocator);
        self.norm.deinit(allocator);
    }
};

fn fillDeterministicF16(data: []u16, seed: u64) void {
    var state = seed ^ 0x9E3779B185EBCA87;
    for (data, 0..) |*slot, idx| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        const r = @as(f32, @floatFromInt((state >> 40) & 0x1FF));
        const centered = (r / 512.0) - 0.5;
        const tweak = @as(f32, @floatFromInt((idx % 19) + 1)) * 0.0078125;
        const value: f16 = @floatCast(centered + tweak);
        slot.* = @bitCast(value);
    }
}

fn fillDeterministicU32(data: []u32, seed: u64) void {
    var state = seed ^ 0xA24BAED4963EE407;
    for (data) |*slot| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        slot.* = @truncate(state >> 16);
    }
}

fn f32ToBf16Bits(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate(bits >> 16);
}

fn fillDeterministicBF16(data: []u16, seed: u64) void {
    var state = seed ^ 0xD1342543DE82EF95;
    for (data, 0..) |*slot, idx| {
        state = state *% 2862933555777941757 +% 3037000493;
        const r = @as(f32, @floatFromInt((state >> 40) & 0xFF));
        const value = 0.01 + (r / 255.0) * 0.04 + @as(f32, @floatFromInt((idx % 7))) * 0.001;
        slot.* = f32ToBf16Bits(value);
    }
}

fn argmaxHost(values: []const f32) u32 {
    var best_idx: usize = 0;
    var best_val: f32 = -std.math.inf(f32);
    for (values, 0..) |v, idx| {
        if (v > best_val) {
            best_val = v;
            best_idx = idx;
        }
    }
    return @intCast(best_idx);
}

fn profileMicroDims(profile: Profile) struct { m: usize, k: usize, n: usize } {
    return switch (profile) {
        .ci => .{ .m = 1, .k = 1024, .n = 1024 },
        .bw => .{ .m = 1, .k = 4096, .n = 4096 },
    };
}

fn profileMatmulThroughputDims(profile: Profile) struct { m: usize, k: usize, n: usize } {
    return switch (profile) {
        .ci => .{ .m = 64, .k = 1024, .n = 1024 },
        .bw => .{ .m = 256, .k = 2048, .n = 2048 },
    };
}

fn profileDecodeDims(profile: Profile) struct { layers: usize, hidden: usize, ff: usize } {
    return switch (profile) {
        .ci => .{ .layers = 8, .hidden = 512, .ff = 2048 },
        .bw => .{ .layers = 16, .hidden = 1024, .ff = 4096 },
    };
}

fn profileDecodeDenseDims(profile: Profile) struct {
    layers: usize,
    hidden: usize,
    ff: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    vocab: usize,
    d_conv: usize,
    conv_dim: usize,
} {
    return switch (profile) {
        .ci => .{
            .layers = 8,
            .hidden = 512,
            .ff = 2048,
            .heads = 8,
            .kv_heads = 8,
            .head_dim = 64,
            .kv_len = 1024,
            .vocab = 32768,
            .d_conv = 4,
            .conv_dim = 512,
        },
        .bw => .{
            .layers = 16,
            .hidden = 1024,
            .ff = 4608,
            .heads = 16,
            .kv_heads = 16,
            .head_dim = 64,
            .kv_len = 4096,
            .vocab = 65536,
            .d_conv = 4,
            .conv_dim = 1024,
        },
    };
}

fn profileAttentionDecodeDims(profile: Profile) struct { heads: usize, head_dim: usize, kv_len: usize } {
    return switch (profile) {
        .ci => .{ .heads = 16, .head_dim = 64, .kv_len = 2048 },
        .bw => .{ .heads = 16, .head_dim = 64, .kv_len = 8192 },
    };
}

fn profileShortconvDecodeDims(profile: Profile) struct {
    hidden: usize,
    conv_dim: usize,
    d_conv: usize,
    steps: usize,
} {
    return switch (profile) {
        .ci => .{
            .hidden = 512,
            .conv_dim = 512,
            .d_conv = 4,
            .steps = 16,
        },
        .bw => .{
            .hidden = 1024,
            .conv_dim = 1024,
            .d_conv = 4,
            .steps = 48,
        },
    };
}

fn profileStateSpaceDecodeDims(profile: Profile) struct {
    hidden: usize,
    n_heads: usize,
    d_head: usize,
    d_state: usize,
    n_groups: usize,
    d_conv: usize,
    steps: usize,
} {
    return switch (profile) {
        .ci => .{
            .hidden = 512,
            .n_heads = 8,
            .d_head = 64,
            .d_state = 16,
            .n_groups = 4,
            .d_conv = 4,
            .steps = 16,
        },
        .bw => .{
            .hidden = 1024,
            .n_heads = 16,
            .d_head = 64,
            .d_state = 16,
            .n_groups = 8,
            .d_conv = 4,
            .steps = 48,
        },
    };
}

fn profileGatedDeltaDecodeDims(profile: Profile) struct {
    hidden: usize,
    n_heads: usize,
    d_head: usize,
    d_conv: usize,
    steps: usize,
} {
    return switch (profile) {
        .ci => .{
            .hidden = 512,
            .n_heads = 8,
            .d_head = 128,
            .d_conv = 4,
            .steps = 16,
        },
        .bw => .{
            .hidden = 1024,
            .n_heads = 16,
            .d_head = 128,
            .d_conv = 4,
            .steps = 48,
        },
    };
}

fn profileGatedDeltaBlockDims(profile: Profile) struct {
    hidden: usize,
    ff: usize,
    n_heads: usize,
    d_head: usize,
    d_conv: usize,
    steps: usize,
} {
    return switch (profile) {
        .ci => .{
            .hidden = 512,
            .ff = 1792,
            .n_heads = 8,
            .d_head = 128,
            .d_conv = 4,
            .steps = 16,
        },
        .bw => .{
            .hidden = 1024,
            .ff = 3584,
            .n_heads = 16,
            .d_head = 128,
            .d_conv = 4,
            .steps = 48,
        },
    };
}

fn profileLmHeadDims(profile: Profile) struct {
    hidden: usize,
    vocab: usize,
    steps: usize,
} {
    return switch (profile) {
        .ci => .{
            .hidden = 1024,
            .vocab = 65536,
            .steps = 16,
        },
        .bw => .{
            .hidden = 1024,
            .vocab = 248320,
            .steps = 48,
        },
    };
}

fn profileAddElems(profile: Profile) usize {
    return switch (profile) {
        .ci => 16 * 1024 * 1024,
        .bw => 64 * 1024 * 1024,
    };
}

fn profileNormShape(profile: Profile) struct { batch: usize, seq: usize, dim: usize } {
    return switch (profile) {
        .ci => .{ .batch = 8, .seq = 256, .dim = 1024 },
        .bw => .{ .batch = 16, .seq = 512, .dim = 2048 },
    };
}

fn profileSoftmaxShape(profile: Profile) struct { rows: usize, cols: usize } {
    return switch (profile) {
        .ci => .{ .rows = 2048, .cols = 1024 },
        .bw => .{ .rows = 4096, .cols = 2048 },
    };
}

fn profileQuantizedDims(profile: Profile) struct { m: usize, k: usize, n: usize, group_size: usize } {
    return switch (profile) {
        .ci => .{ .m = 1, .k = 4096, .n = 4096, .group_size = 32 },
        // Keep decode-like shape (m=1) so qmm rows reflect token-generation
        // bottlenecks instead of prefill-style compute saturation.
        .bw => .{ .m = 1, .k = 8192, .n = 8192, .group_size = 32 },
    };
}

fn profileFfnQuantDecodeDims(profile: Profile) struct { hidden: usize, ff: usize, group_size: usize } {
    return switch (profile) {
        .ci => .{ .hidden = 1024, .ff = 4096, .group_size = 32 },
        .bw => .{ .hidden = 2048, .ff = 8192, .group_size = 32 },
    };
}

fn profileMethodRepeats(profile: Profile) usize {
    return switch (profile) {
        .ci => 4,
        .bw => 4,
    };
}

fn profileReductionMethodRepeats(profile: Profile) usize {
    return switch (profile) {
        .ci => 4,
        // Reduction-heavy kernels need longer sustained loops to amortize
        // dispatch overhead and expose method-limited throughput.
        .bw => 8,
    };
}

fn elapsedNs(start_ns: i128, end_ns: i128) u64 {
    if (end_ns <= start_ns) return 0;
    return @intCast(end_ns - start_ns);
}

fn recordSample(
    cold_first: *harness.Sample,
    samples: []harness.Sample,
    sample_idx: *usize,
    iter: usize,
    warmup: usize,
    t0: i128,
    t1: i128,
    t2: i128,
    t3: i128,
) void {
    const sample = harness.Sample{
        .build_ns = elapsedNs(t0, t1),
        .eval_ns = elapsedNs(t2, t3),
        .total_ns = elapsedNs(t0, t3),
    };
    if (iter == 0) cold_first.* = sample;
    if (iter >= warmup and sample_idx.* < samples.len) {
        samples[sample_idx.*] = sample;
        sample_idx.* += 1;
    }
}

fn expectMetalReady() !void {
    if (comptime builtin.os.tag != .macos) return error.MetalUnavailable;
    if (!metal.isAvailable()) return error.MetalUnavailable;
}

fn toU64Saturating(value: u128) u64 {
    return if (value > std.math.maxInt(u64))
        std.math.maxInt(u64)
    else
        @intCast(value);
}

pub fn runMicroMatmulF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const dims = profileMicroDims(cfg.profile);
    const repeats = profileMethodRepeats(cfg.profile);
    var lhs = try OwnedF16.init2D(allocator, dims.m, dims.k, 0xC001);
    defer lhs.deinit(allocator);
    var rhs = try OwnedF16.init2D(allocator, dims.k, dims.n, 0xC0DE);
    defer rhs.deinit(allocator);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();
        var out = graph.mlx_lazy_matmul(lhs.handle, rhs.handle);
        var rep: usize = 1;
        while (rep < repeats) : (rep += 1) {
            out = graph.mlx_lazy_matmul(out, rhs.handle);
        }
        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{out};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const flops = toU64Saturating(2 * @as(u128, dims.m) * dims.k * dims.n * repeats);
    const bytes = toU64Saturating((@as(u128, dims.m) * dims.k + @as(u128, dims.k) * dims.n + @as(u128, dims.m) * dims.n) * 2 * repeats);

    return .{
        .name = "p3_mm_micro",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = flops,
        .bytes_per_iter = bytes,
        .note = "single F16 matmul; bytes estimate includes input+weight+output",
    };
}

pub fn runAddF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const elems = profileAddElems(cfg.profile);
    const repeats = profileMethodRepeats(cfg.profile);
    var a = try OwnedF16.init1D(allocator, elems, 0x5101);
    defer a.deinit(allocator);
    var b = try OwnedF16.init1D(allocator, elems, 0x5102);
    defer b.deinit(allocator);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();
        var out = graph.mlx_lazy_add(a.handle, b.handle);
        var rep: usize = 1;
        while (rep < repeats) : (rep += 1) {
            out = graph.mlx_lazy_add(out, b.handle);
        }
        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{out};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const bytes = toU64Saturating(@as(u128, elems) * 3 * 2 * repeats);
    const flops = toU64Saturating(@as(u128, elems) * repeats);

    return .{
        .name = "p2_add",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = flops,
        .bytes_per_iter = bytes,
        .note = "elementwise add method stress (read+read+write)",
    };
}

pub fn runMultiplyF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const elems = profileAddElems(cfg.profile);
    const repeats = profileMethodRepeats(cfg.profile);
    var a = try OwnedF16.init1D(allocator, elems, 0x5201);
    defer a.deinit(allocator);
    var b = try OwnedF16.init1D(allocator, elems, 0x5202);
    defer b.deinit(allocator);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();
        var out = graph.mlx_lazy_multiply(a.handle, b.handle);
        var rep: usize = 1;
        while (rep < repeats) : (rep += 1) {
            out = graph.mlx_lazy_multiply(out, b.handle);
        }
        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{out};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const bytes = toU64Saturating(@as(u128, elems) * 3 * 2 * repeats);
    const flops = toU64Saturating(@as(u128, elems) * repeats);

    return .{
        .name = "p2_mul",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = flops,
        .bytes_per_iter = bytes,
        .note = "elementwise multiply method stress",
    };
}

pub fn runRmsNormF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const dims = profileNormShape(cfg.profile);
    const repeats = profileReductionMethodRepeats(cfg.profile);
    const x_shape = [_]i64{ @intCast(dims.batch), @intCast(dims.seq), @intCast(dims.dim) };
    var x = try OwnedF16.initShape(allocator, &x_shape, 0x5401);
    defer x.deinit(allocator);
    var w = try OwnedF16.init1D(allocator, dims.dim, 0x5402);
    defer w.deinit(allocator);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();
        var out = graph.mlx_lazy_rms_norm(x.handle, w.handle, 1.0e-5);
        var rep: usize = 1;
        while (rep < repeats) : (rep += 1) {
            out = graph.mlx_lazy_rms_norm(out, w.handle, 1.0e-5);
        }
        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{out};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const elems = @as(u128, dims.batch) * dims.seq * dims.dim;
    const bytes = toU64Saturating((elems * 2 + dims.dim) * 2 * repeats);
    const flops = toU64Saturating(elems * 6 * repeats);

    return .{
        .name = "p2_rms",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = flops,
        .bytes_per_iter = bytes,
        .note = "RMSNorm method stress",
    };
}

pub fn runSoftmaxF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const dims = profileSoftmaxShape(cfg.profile);
    const repeats = profileReductionMethodRepeats(cfg.profile);
    const x_shape = [_]i64{ @intCast(dims.rows), @intCast(dims.cols) };
    var x = try OwnedF16.initShape(allocator, &x_shape, 0x5501);
    defer x.deinit(allocator);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();
        var out = graph.mlx_lazy_softmax(x.handle, -1);
        var rep: usize = 1;
        while (rep < repeats) : (rep += 1) {
            out = graph.mlx_lazy_softmax(out, -1);
        }
        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{out};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const elems = @as(u128, dims.rows) * dims.cols;
    const bytes = toU64Saturating(elems * 2 * 2 * repeats);
    const flops = toU64Saturating(elems * 5 * repeats);

    return .{
        .name = "p2_smx",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = flops,
        .bytes_per_iter = bytes,
        .note = "softmax method stress",
    };
}

fn runFusedFfnQuantizedDecode(
    allocator: std.mem.Allocator,
    cfg: RunConfig,
    bits: usize,
    name: []const u8,
) !ScenarioResult {
    try expectMetalReady();

    const dims = profileFfnQuantDecodeDims(cfg.profile);
    const repeats = profileMethodRepeats(cfg.profile);
    if (dims.hidden % dims.group_size != 0) return error.InvalidInput;
    if (dims.ff % dims.group_size != 0) return error.InvalidInput;
    if (bits != 4 and bits != 8) return error.InvalidInput;
    if (bits == 4 and dims.hidden % 8 != 0) return error.InvalidInput;
    if (bits == 4 and dims.ff % 8 != 0) return error.InvalidInput;
    if (bits == 8 and dims.hidden % 4 != 0) return error.InvalidInput;
    if (bits == 8 and dims.ff % 4 != 0) return error.InvalidInput;

    const x_shape = [_]i64{ 1, 1, @intCast(dims.hidden) };
    var x = try OwnedBF16.initShape(allocator, &x_shape, 0x7A01);
    defer x.deinit(allocator);

    const gate_up_packed_k = dims.hidden * bits / 32;
    const down_packed_k = dims.ff * bits / 32;

    var gate_w = try OwnedU32.init2D(allocator, dims.ff, gate_up_packed_k, 0x7A02);
    defer gate_w.deinit(allocator);
    var up_w = try OwnedU32.init2D(allocator, dims.ff, gate_up_packed_k, 0x7A03);
    defer up_w.deinit(allocator);
    var down_w = try OwnedU32.init2D(allocator, dims.hidden, down_packed_k, 0x7A04);
    defer down_w.deinit(allocator);

    const gate_up_groups = dims.hidden / dims.group_size;
    const down_groups = dims.ff / dims.group_size;

    var gate_s = try OwnedBF16.init2D(allocator, dims.ff, gate_up_groups, 0x7A05);
    defer gate_s.deinit(allocator);
    var gate_b = try OwnedBF16.init2D(allocator, dims.ff, gate_up_groups, 0x7A06);
    defer gate_b.deinit(allocator);
    var up_s = try OwnedBF16.init2D(allocator, dims.ff, gate_up_groups, 0x7A07);
    defer up_s.deinit(allocator);
    var up_b = try OwnedBF16.init2D(allocator, dims.ff, gate_up_groups, 0x7A08);
    defer up_b.deinit(allocator);
    var down_s = try OwnedBF16.init2D(allocator, dims.hidden, down_groups, 0x7A09);
    defer down_s.deinit(allocator);
    var down_b = try OwnedBF16.init2D(allocator, dims.hidden, down_groups, 0x7A0A);
    defer down_b.deinit(allocator);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();
        var out = graph.mlx_lazy_fused_ffn(
            x.handle,
            gate_w.handle,
            gate_s.handle,
            gate_b.handle,
            up_w.handle,
            up_s.handle,
            up_b.handle,
            down_w.handle,
            down_s.handle,
            down_b.handle,
            dims.group_size,
            bits,
            false,
        );
        var rep: usize = 1;
        while (rep < repeats) : (rep += 1) {
            out = graph.mlx_lazy_fused_ffn(
                out,
                gate_w.handle,
                gate_s.handle,
                gate_b.handle,
                up_w.handle,
                up_s.handle,
                up_b.handle,
                down_w.handle,
                down_s.handle,
                down_b.handle,
                dims.group_size,
                bits,
                false,
            );
        }
        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{out};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const flops_per_ffn = @as(u128, 6) * dims.hidden * dims.ff;
    const gate_up_weight_bytes = @as(u128, dims.ff) * dims.hidden * bits / 8;
    const down_weight_bytes = @as(u128, dims.hidden) * dims.ff * bits / 8;
    const weight_bytes = gate_up_weight_bytes * 2 + down_weight_bytes;
    const scale_bias_bytes = (@as(u128, dims.ff) * gate_up_groups * 2 * 2 * 2) +
        (@as(u128, dims.hidden) * down_groups * 2 * 2);
    const input_output_bytes = @as(u128, dims.hidden) * 2 * 2;

    return .{
        .name = name,
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = toU64Saturating(flops_per_ffn * repeats),
        .bytes_per_iter = toU64Saturating((weight_bytes + scale_bias_bytes + input_output_bytes) * repeats),
        .note = "fused quantized FFN decode path (gate/up/down)",
    };
}

pub fn runFusedFfnQuantizedDecodeU4(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runFusedFfnQuantizedDecode(allocator, cfg, 4, "p1_ffnq_u4");
}

pub fn runFusedFfnQuantizedDecodeU8(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runFusedFfnQuantizedDecode(allocator, cfg, 8, "p1_ffnq_u8");
}

pub fn runFusedFfnDenseDecodeF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const decode_dims = profileDecodeDenseDims(cfg.profile);
    const repeats = profileMethodRepeats(cfg.profile);
    const token_shape = [_]i64{ 1, 1, @intCast(decode_dims.hidden) };
    var x = try OwnedF16.initShape(allocator, &token_shape, 0x7C01);
    defer x.deinit(allocator);

    var w1 = try OwnedF16.init2D(allocator, decode_dims.hidden, decode_dims.ff, 0x7C02);
    defer w1.deinit(allocator);
    var w3 = try OwnedF16.init2D(allocator, decode_dims.hidden, decode_dims.ff, 0x7C03);
    defer w3.deinit(allocator);
    var w2 = try OwnedF16.init2D(allocator, decode_dims.ff, decode_dims.hidden, 0x7C04);
    defer w2.deinit(allocator);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();
        var out = graph.mlx_lazy_fused_ffn_bf16(
            x.handle,
            w1.handle,
            w3.handle,
            w2.handle,
        );
        var rep: usize = 1;
        while (rep < repeats) : (rep += 1) {
            out = graph.mlx_lazy_fused_ffn_bf16(
                out,
                w1.handle,
                w3.handle,
                w2.handle,
            );
        }
        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{out};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const flops_per_ffn = @as(u128, 6) * decode_dims.hidden * decode_dims.ff;
    const weight_bytes_per_ffn = @as(u128, 3) * decode_dims.hidden * decode_dims.ff * 2;
    const io_bytes_per_ffn = @as(u128, 2) * decode_dims.hidden * 2;

    return .{
        .name = "p1_ffnd",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = toU64Saturating(flops_per_ffn * repeats),
        .bytes_per_iter = toU64Saturating((weight_bytes_per_ffn + io_bytes_per_ffn) * repeats),
        .note = "fused dense BF16 FFN decode path (gate/up/down)",
    };
}

fn runQuantizedMatmul(
    allocator: std.mem.Allocator,
    cfg: RunConfig,
    bits: usize,
    name: []const u8,
) !ScenarioResult {
    try expectMetalReady();
    const dims = profileQuantizedDims(cfg.profile);
    const repeats = profileMethodRepeats(cfg.profile);
    if (bits != 4 and bits != 8) return error.InvalidInput;
    if (dims.k % dims.group_size != 0) return error.InvalidInput;
    if (bits == 4 and dims.k % 8 != 0) return error.InvalidInput;
    if (bits == 8 and dims.k % 4 != 0) return error.InvalidInput;

    var input = try OwnedBF16.init2D(allocator, dims.m, dims.k, 0x6601);
    defer input.deinit(allocator);

    const packed_k = dims.k * bits / 32;
    var weights = try OwnedU32.init2D(allocator, dims.n, packed_k, 0x6602);
    defer weights.deinit(allocator);

    const groups = dims.k / dims.group_size;
    var scales = try OwnedBF16.init2D(allocator, dims.n, groups, 0x6603);
    defer scales.deinit(allocator);
    var biases = try OwnedBF16.init2D(allocator, dims.n, groups, 0x6604);
    defer biases.deinit(allocator);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();
        var out = graph.mlx_lazy_quantized_matmul(
            input.handle,
            weights.handle,
            scales.handle,
            biases.handle,
            dims.group_size,
            bits,
            true,
        );
        var rep: usize = 1;
        while (rep < repeats) : (rep += 1) {
            out = graph.mlx_lazy_quantized_matmul(
                out,
                weights.handle,
                scales.handle,
                biases.handle,
                dims.group_size,
                bits,
                true,
            );
        }
        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{out};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const flops = toU64Saturating(2 * @as(u128, dims.m) * dims.k * dims.n * repeats);
    const weight_bytes = @as(u128, dims.n) * dims.k * bits / 8;
    const scale_bias_bytes = @as(u128, dims.n) * groups * 2 * 2;
    const bytes = toU64Saturating((@as(u128, dims.m) * dims.k * 2 + weight_bytes + scale_bias_bytes + @as(u128, dims.m) * dims.n * 2) * repeats);

    return .{
        .name = name,
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = flops,
        .bytes_per_iter = bytes,
        .note = "quantized matmul method stress",
    };
}

pub fn runQuantizedMatmulU4(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runQuantizedMatmul(allocator, cfg, 4, "p1_qmm_u4");
}

pub fn runQuantizedMatmulU8(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runQuantizedMatmul(allocator, cfg, 8, "p1_qmm_u8");
}

pub fn runMatmulThroughputF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const dims = profileMatmulThroughputDims(cfg.profile);
    const repeats = profileMethodRepeats(cfg.profile);
    const startup_preheat_iters: usize = switch (cfg.profile) {
        .ci => 0,
        .bw => 32,
    };
    var lhs = try OwnedF16.init2D(allocator, dims.m, dims.k, 0x6101);
    defer lhs.deinit(allocator);
    var rhs = try OwnedF16.init2D(allocator, dims.k, dims.n, 0x6102);
    defer rhs.deinit(allocator);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    // BW throughput mode should reflect sustained compute. Prime clocks/state
    // outside measured samples to reduce first-invocation ramp noise.
    var preheat_iter: usize = 0;
    while (preheat_iter < startup_preheat_iters) : (preheat_iter += 1) {
        graph.beginForwardGraphBuild();
        var out = graph.mlx_lazy_matmul(lhs.handle, rhs.handle);
        var rep: usize = 1;
        while (rep < repeats) : (rep += 1) {
            out = graph.mlx_lazy_matmul(out, rhs.handle);
        }
        var handles = [_]graph.ArrayHandle{out};
        graph.eval(&handles);
    }
    while (iter < total_iters) : (iter += 1) {
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();
        var out = graph.mlx_lazy_matmul(lhs.handle, rhs.handle);
        var rep: usize = 1;
        while (rep < repeats) : (rep += 1) {
            out = graph.mlx_lazy_matmul(out, rhs.handle);
        }
        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{out};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const flops = toU64Saturating(2 * @as(u128, dims.m) * dims.k * dims.n * repeats);
    const bytes = toU64Saturating((@as(u128, dims.m) * dims.k + @as(u128, dims.k) * dims.n + @as(u128, dims.m) * dims.n) * 2 * repeats);

    return .{
        .name = "p1_mm_thr",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = flops,
        .bytes_per_iter = bytes,
        .note = "matmul method throughput stress (large M)",
    };
}

pub fn runAttentionDecodeF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const dims = profileAttentionDecodeDims(cfg.profile);
    const hidden = dims.heads * dims.head_dim;
    // Keep decode samples long enough to amortize overhead while preserving
    // realistic token-step cache updates on the fused attention method path.
    const repeats: usize = switch (cfg.profile) {
        .ci => 4,
        .bw => 12,
    };
    const token_shape = [_]i64{ 1, 1, @intCast(hidden) };
    const kv_shape = [_]i64{ 1, @intCast(dims.heads), @intCast(dims.kv_len), @intCast(dims.head_dim) };
    var x = try OwnedF16.initShape(allocator, &token_shape, 0x7101);
    defer x.deinit(allocator);

    const AttnProj = struct {
        q_w: OwnedF16,
        k_w: OwnedF16,
        v_w: OwnedF16,
        o_w: OwnedF16,

        fn init(alloc: std.mem.Allocator, h: usize, seed: u64) !@This() {
            return .{
                .q_w = try OwnedF16.init2D(alloc, h, h, seed +% 1),
                .k_w = try OwnedF16.init2D(alloc, h, h, seed +% 2),
                .v_w = try OwnedF16.init2D(alloc, h, h, seed +% 3),
                .o_w = try OwnedF16.init2D(alloc, h, h, seed +% 4),
            };
        }

        fn deinit(self: *@This(), alloc: std.mem.Allocator) void {
            self.q_w.deinit(alloc);
            self.k_w.deinit(alloc);
            self.v_w.deinit(alloc);
            self.o_w.deinit(alloc);
        }
    };

    var proj_layers = try allocator.alloc(AttnProj, repeats);
    defer allocator.free(proj_layers);
    var proj_initialized: usize = 0;
    errdefer for (proj_layers[0..proj_initialized]) |*layer| layer.deinit(allocator);
    for (proj_layers, 0..) |*layer, idx| {
        layer.* = try AttnProj.init(allocator, hidden, 0x7102 +% @as(u64, @intCast(idx)) *% 29);
        proj_initialized += 1;
    }
    defer for (proj_layers) |*layer| layer.deinit(allocator);

    var k_full = try OwnedF16.initShape(allocator, &kv_shape, 0x7106);
    defer k_full.deinit(allocator);
    var v_full = try OwnedF16.initShape(allocator, &kv_shape, 0x7107);
    defer v_full.deinit(allocator);
    const cache = graph.mlx_cache_create(1, dims.kv_len + repeats + 8);
    defer graph.mlx_cache_free(cache);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        graph.mlx_cache_reset(cache);
        graph.mlx_cache_set_full_bfloat16(cache, 0, k_full.handle, v_full.handle);
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();
        var out: graph.ArrayHandle = x.handle;
        var rep: usize = 0;
        while (rep < repeats) : (rep += 1) {
            const layer = &proj_layers[rep];
            out = graph.mlx_lazy_fused_attention_bf16(
                out,
                layer.q_w.handle,
                layer.k_w.handle,
                layer.v_w.handle,
                layer.o_w.handle,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                cache,
                0,
                dims.heads,
                dims.heads,
                dims.head_dim,
                dims.kv_len + rep,
                10_000.0,
                null,
                null,
                0,
                1.0e-5,
                0.0,
                0.0,
            );
        }
        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{out};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const proj_flops_per_step = @as(u128, 8) * hidden * hidden;
    const attn_flops_per_step = @as(u128, 4) * dims.heads * dims.kv_len * dims.head_dim;
    const flops = toU64Saturating((proj_flops_per_step + attn_flops_per_step) * repeats);

    const proj_weight_bytes_per_step = @as(u128, 4) * hidden * hidden * 2;
    const kv_read_bytes_per_step = @as(u128, 2) * dims.heads * dims.kv_len * dims.head_dim * 2;
    const kv_write_bytes_per_step = @as(u128, 2) * dims.heads * dims.head_dim * 2;
    const io_bytes_per_step = @as(u128, 2) * hidden * 2;
    const bytes = toU64Saturating((proj_weight_bytes_per_step + kv_read_bytes_per_step + kv_write_bytes_per_step + io_bytes_per_step) * repeats);

    return .{
        .name = "p1_attn",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = flops,
        .bytes_per_iter = bytes,
        .note = "fused dense attention decode method with prefilled KV cache",
    };
}

pub fn runShortconvDecodeBF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const dims = profileShortconvDecodeDims(cfg.profile);
    const token_shape = [_]i64{ 1, 1, @intCast(dims.hidden) };
    var x = try OwnedBF16.initShape(allocator, &token_shape, 0x9101);
    defer x.deinit(allocator);
    var in_proj = try OwnedBF16.init2D(allocator, dims.hidden, 3 * dims.conv_dim, 0x9102);
    defer in_proj.deinit(allocator);
    var conv_w = try OwnedBF16.init2D(allocator, dims.d_conv, dims.conv_dim, 0x9103);
    defer conv_w.deinit(allocator);
    var conv_b = try OwnedBF16.init2D(allocator, 1, dims.conv_dim, 0x9104);
    defer conv_b.deinit(allocator);
    var out_proj = try OwnedBF16.init2D(allocator, dims.conv_dim, dims.hidden, 0x9105);
    defer out_proj.deinit(allocator);

    const cache = graph.mlx_causal_conv_cache_create(1);
    defer graph.mlx_causal_conv_cache_free(cache);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        graph.mlx_causal_conv_cache_reset(cache);
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();
        var out: graph.ArrayHandle = x.handle;
        var step: usize = 0;
        while (step < dims.steps) : (step += 1) {
            out = graph.mlx_lazy_causal_conv_mixer_bf16(
                out,
                in_proj.handle,
                conv_w.handle,
                conv_b.handle,
                out_proj.handle,
                cache,
                0,
                dims.d_conv,
                dims.conv_dim,
            );
        }
        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{out};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const flops_per_step =
        @as(u128, 2) * dims.hidden * (3 * dims.conv_dim) +
        @as(u128, 2) * dims.d_conv * dims.conv_dim +
        @as(u128, 3) * dims.conv_dim +
        @as(u128, 2) * dims.conv_dim * dims.hidden;
    const bytes_per_step =
        (@as(u128, dims.hidden) * (3 * dims.conv_dim) +
            @as(u128, dims.d_conv) * dims.conv_dim +
            @as(u128, dims.conv_dim) +
            @as(u128, dims.conv_dim) * dims.hidden) * 2 +
        (@as(u128, 2) * dims.d_conv * dims.conv_dim * 4);

    return .{
        .name = "p1_scv",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = toU64Saturating(flops_per_step * dims.steps),
        .bytes_per_iter = toU64Saturating(bytes_per_step * dims.steps),
        .note = "dense shortconv decode method with recurrent cache updates",
    };
}

pub fn runShortconvDecodeQuantizedU4(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const dims = profileShortconvDecodeDims(cfg.profile);
    const bits: usize = 4;
    const group_size: usize = 64;
    if (dims.hidden % group_size != 0) return error.InvalidInput;
    if (dims.conv_dim % group_size != 0) return error.InvalidInput;
    if (dims.hidden % 8 != 0) return error.InvalidInput;
    if (dims.conv_dim % 8 != 0) return error.InvalidInput;

    const token_shape = [_]i64{ 1, 1, @intCast(dims.hidden) };
    var x = try OwnedBF16.initShape(allocator, &token_shape, 0xA101);
    defer x.deinit(allocator);

    const in_out = 3 * dims.conv_dim;
    const in_packed_k = dims.hidden * bits / 32;
    var in_w = try OwnedU32.init2D(allocator, in_out, in_packed_k, 0xA102);
    defer in_w.deinit(allocator);
    var in_s = try OwnedBF16.init2D(allocator, in_out, dims.hidden / group_size, 0xA103);
    defer in_s.deinit(allocator);
    var in_b = try OwnedBF16.init2D(allocator, in_out, dims.hidden / group_size, 0xA104);
    defer in_b.deinit(allocator);

    var conv_w = try OwnedBF16.init2D(allocator, dims.d_conv, dims.conv_dim, 0xA105);
    defer conv_w.deinit(allocator);
    var conv_b = try OwnedBF16.init2D(allocator, 1, dims.conv_dim, 0xA106);
    defer conv_b.deinit(allocator);

    const out_packed_k = dims.conv_dim * bits / 32;
    var out_w = try OwnedU32.init2D(allocator, dims.hidden, out_packed_k, 0xA107);
    defer out_w.deinit(allocator);
    var out_s = try OwnedBF16.init2D(allocator, dims.hidden, dims.conv_dim / group_size, 0xA108);
    defer out_s.deinit(allocator);
    var out_b = try OwnedBF16.init2D(allocator, dims.hidden, dims.conv_dim / group_size, 0xA109);
    defer out_b.deinit(allocator);

    const cache = graph.mlx_causal_conv_cache_create(1);
    defer graph.mlx_causal_conv_cache_free(cache);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        graph.mlx_causal_conv_cache_reset(cache);
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();
        var out: graph.ArrayHandle = x.handle;
        var step: usize = 0;
        while (step < dims.steps) : (step += 1) {
            out = graph.mlx_lazy_causal_conv_mixer_quantized(
                out,
                in_w.handle,
                in_s.handle,
                in_b.handle,
                conv_w.handle,
                conv_b.handle,
                out_w.handle,
                out_s.handle,
                out_b.handle,
                group_size,
                bits,
                cache,
                0,
                dims.d_conv,
                dims.conv_dim,
            );
        }
        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{out};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const flops_per_step =
        @as(u128, 2) * dims.hidden * (3 * dims.conv_dim) +
        @as(u128, 2) * dims.d_conv * dims.conv_dim +
        @as(u128, 3) * dims.conv_dim +
        @as(u128, 2) * dims.conv_dim * dims.hidden;
    const in_weight_bytes = @as(u128, in_out) * dims.hidden * bits / 8;
    const in_scale_bias_bytes = @as(u128, in_out) * (dims.hidden / group_size) * 2 * 2;
    const out_weight_bytes = @as(u128, dims.hidden) * dims.conv_dim * bits / 8;
    const out_scale_bias_bytes = @as(u128, dims.hidden) * (dims.conv_dim / group_size) * 2 * 2;
    const conv_bytes = (@as(u128, dims.d_conv) * dims.conv_dim + @as(u128, dims.conv_dim)) * 2;
    const state_bytes = (@as(u128, 2) * dims.d_conv * dims.conv_dim) * 4;
    const bytes_per_step = in_weight_bytes + in_scale_bias_bytes + out_weight_bytes + out_scale_bias_bytes + conv_bytes + state_bytes;

    return .{
        .name = "p1_scvq_u4",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = toU64Saturating(flops_per_step * dims.steps),
        .bytes_per_iter = toU64Saturating(bytes_per_step * dims.steps),
        .note = "quantized shortconv decode method with recurrent cache updates (u4)",
    };
}

pub fn runStateSpaceDecodeF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const dims = profileStateSpaceDecodeDims(cfg.profile);
    if (dims.hidden != dims.n_heads * dims.d_head) return error.InvalidInput;
    if ((dims.n_heads % dims.n_groups) != 0) return error.InvalidInput;

    const d_inner = dims.n_heads * dims.d_head;
    const bc_len = dims.n_groups * dims.d_state;
    const xbc_len = d_inner + 2 * bc_len;
    const proj_dim = 2 * d_inner + 2 * bc_len + dims.n_heads;

    const token_shape = [_]i64{ 1, 1, @intCast(dims.hidden) };
    var x = try OwnedF16.initShape(allocator, &token_shape, 0xA201);
    defer x.deinit(allocator);

    var ln1_w = try OwnedF16.init1D(allocator, dims.hidden, 0xA202);
    defer ln1_w.deinit(allocator);
    var in_proj = try OwnedF16.init2D(allocator, dims.hidden, proj_dim, 0xA203);
    defer in_proj.deinit(allocator);
    var conv_w = try OwnedF16.init2D(allocator, dims.d_conv, xbc_len, 0xA204);
    defer conv_w.deinit(allocator);
    var conv_b = try OwnedF16.init1D(allocator, xbc_len, 0xA205);
    defer conv_b.deinit(allocator);
    var a_log = try OwnedF16.init1D(allocator, dims.n_heads, 0xA206);
    defer a_log.deinit(allocator);
    var d_skip = try OwnedF16.init1D(allocator, dims.n_heads, 0xA207);
    defer d_skip.deinit(allocator);
    var dt_bias = try OwnedF16.init1D(allocator, dims.n_heads, 0xA208);
    defer dt_bias.deinit(allocator);
    var out_proj = try OwnedF16.init2D(allocator, d_inner, dims.hidden, 0xA209);
    defer out_proj.deinit(allocator);

    const cache = graph.mlx_state_space_cache_create(1);
    defer graph.mlx_state_space_cache_free(cache);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        graph.mlx_state_space_cache_reset(cache);
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();
        var out: graph.ArrayHandle = x.handle;
        var step: usize = 0;
        while (step < dims.steps) : (step += 1) {
            out = graph.mlx_lazy_state_space_block_bf16(
                out,
                ln1_w.handle,
                in_proj.handle,
                conv_w.handle,
                conv_b.handle,
                a_log.handle,
                d_skip.handle,
                dt_bias.handle,
                null,
                out_proj.handle,
                null,
                null,
                null,
                false,
                1.0,
                1.0e-5,
                cache,
                0,
                dims.d_state,
                dims.d_conv,
                dims.n_heads,
                dims.d_head,
                dims.n_groups,
                0,
            );
        }
        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{out};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const in_proj_flops = @as(u128, 2) * dims.hidden * proj_dim;
    const out_proj_flops = @as(u128, 2) * d_inner * dims.hidden;
    const conv_flops = @as(u128, 2) * dims.d_conv * xbc_len + @as(u128, 3) * xbc_len;
    const ssm_state_elems = @as(u128, dims.n_heads) * dims.d_head * dims.d_state;
    const ssm_flops = @as(u128, 9) * ssm_state_elems + @as(u128, 2) * dims.n_heads * dims.d_head;
    const flops_per_step = in_proj_flops + out_proj_flops + conv_flops + ssm_flops;

    const in_proj_bytes = @as(u128, dims.hidden) * proj_dim * 2;
    const out_proj_bytes = @as(u128, d_inner) * dims.hidden * 2;
    const conv_bytes = (@as(u128, dims.d_conv) * xbc_len + xbc_len) * 2;
    const vec_bytes = (@as(u128, 3) * dims.n_heads + dims.hidden) * 2;
    const conv_state_bytes = (@as(u128, 2) * dims.d_conv * xbc_len) * 4;
    const ssm_state_bytes = (@as(u128, 2) * dims.n_heads * dims.d_head * dims.d_state) * 4;
    const bytes_per_step = in_proj_bytes + out_proj_bytes + conv_bytes + vec_bytes + conv_state_bytes + ssm_state_bytes;

    return .{
        .name = "p1_ssm",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = toU64Saturating(flops_per_step * dims.steps),
        .bytes_per_iter = toU64Saturating(bytes_per_step * dims.steps),
        .note = "state-space decode method with recurrent conv+ssm state updates",
    };
}

pub fn runGatedDeltaDecodeF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const dims = profileGatedDeltaDecodeDims(cfg.profile);

    const d_inner = dims.n_heads * dims.d_head;
    const qkv_len = 3 * d_inner;
    const proj_dim = (4 * d_inner) + (2 * dims.n_heads);

    const token_shape = [_]i64{ 1, 1, @intCast(dims.hidden) };
    var x = try OwnedF16.initShape(allocator, &token_shape, 0xA301);
    defer x.deinit(allocator);

    var in_proj = try OwnedF16.init2D(allocator, dims.hidden, proj_dim, 0xA302);
    defer in_proj.deinit(allocator);
    var conv_w = try OwnedF16.init2D(allocator, dims.d_conv, qkv_len, 0xA303);
    defer conv_w.deinit(allocator);
    var conv_b = try OwnedF16.init1D(allocator, qkv_len, 0xA304);
    defer conv_b.deinit(allocator);
    var a_log = try OwnedF16.init1D(allocator, dims.n_heads, 0xA305);
    defer a_log.deinit(allocator);
    var dt_bias = try OwnedF16.init1D(allocator, dims.n_heads, 0xA306);
    defer dt_bias.deinit(allocator);
    var norm_weight = try OwnedF16.init1D(allocator, d_inner, 0xA307);
    defer norm_weight.deinit(allocator);
    var out_proj = try OwnedF16.init2D(allocator, d_inner, dims.hidden, 0xA308);
    defer out_proj.deinit(allocator);

    const cache = graph.mlx_state_space_cache_create(1);
    defer graph.mlx_state_space_cache_free(cache);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        graph.mlx_state_space_cache_reset(cache);
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();
        var out: graph.ArrayHandle = x.handle;
        var step: usize = 0;
        while (step < dims.steps) : (step += 1) {
            out = graph.mlx_lazy_gated_delta_mixer_bf16(
                out,
                in_proj.handle,
                conv_w.handle,
                conv_b.handle,
                a_log.handle,
                dt_bias.handle,
                norm_weight.handle,
                out_proj.handle,
                cache,
                0,
                dims.d_conv,
                dims.n_heads,
                dims.d_head,
            );
        }
        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{out};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const in_proj_flops = @as(u128, 2) * dims.hidden * proj_dim;
    const out_proj_flops = @as(u128, 2) * d_inner * dims.hidden;
    const conv_flops = @as(u128, 2) * dims.d_conv * qkv_len + @as(u128, 3) * qkv_len;
    const recurrent_state_elems = @as(u128, dims.n_heads) * dims.d_head * dims.d_head;
    const recurrent_flops = @as(u128, 12) * recurrent_state_elems + @as(u128, 8) * dims.n_heads * dims.d_head;
    const flops_per_step = in_proj_flops + out_proj_flops + conv_flops + recurrent_flops;

    const in_proj_bytes = @as(u128, dims.hidden) * proj_dim * 2;
    const out_proj_bytes = @as(u128, d_inner) * dims.hidden * 2;
    const conv_bytes = (@as(u128, dims.d_conv) * qkv_len + qkv_len) * 2;
    const vec_bytes = (@as(u128, 2) * dims.n_heads + d_inner) * 2;
    const conv_state_bytes = (@as(u128, 2) * dims.d_conv * qkv_len) * 4;
    const recurrent_state_bytes = (@as(u128, 2) * dims.n_heads * dims.d_head * dims.d_head) * 4;
    const bytes_per_step = in_proj_bytes + out_proj_bytes + conv_bytes + vec_bytes + conv_state_bytes + recurrent_state_bytes;

    return .{
        .name = "p1_gdm",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = toU64Saturating(flops_per_step * dims.steps),
        .bytes_per_iter = toU64Saturating(bytes_per_step * dims.steps),
        .note = "gated-delta decode method with recurrent conv+delta state updates",
    };
}

pub fn runGatedDeltaBlockF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const dims = profileGatedDeltaBlockDims(cfg.profile);
    const d_inner = dims.n_heads * dims.d_head;
    const qkv_len = 3 * d_inner;
    const proj_dim = (4 * d_inner) + (2 * dims.n_heads);

    const token_shape = [_]i64{ 1, 1, @intCast(dims.hidden) };
    var x = try OwnedF16.initShape(allocator, &token_shape, 0xA321);
    defer x.deinit(allocator);

    var ln1_w = try OwnedF16.init1D(allocator, dims.hidden, 0xA322);
    defer ln1_w.deinit(allocator);
    var in_proj = try OwnedF16.init2D(allocator, dims.hidden, proj_dim, 0xA323);
    defer in_proj.deinit(allocator);
    var conv_w = try OwnedF16.init2D(allocator, dims.d_conv, qkv_len, 0xA324);
    defer conv_w.deinit(allocator);
    var conv_b = try OwnedF16.init1D(allocator, qkv_len, 0xA325);
    defer conv_b.deinit(allocator);
    var a_log = try OwnedF16.init1D(allocator, dims.n_heads, 0xA326);
    defer a_log.deinit(allocator);
    var dt_bias = try OwnedF16.init1D(allocator, dims.n_heads, 0xA327);
    defer dt_bias.deinit(allocator);
    var norm_weight = try OwnedF16.init1D(allocator, d_inner, 0xA328);
    defer norm_weight.deinit(allocator);
    var out_proj = try OwnedF16.init2D(allocator, d_inner, dims.hidden, 0xA329);
    defer out_proj.deinit(allocator);
    var ln2_w = try OwnedF16.init1D(allocator, dims.hidden, 0xA32A);
    defer ln2_w.deinit(allocator);
    var w1 = try OwnedF16.init2D(allocator, dims.hidden, dims.ff, 0xA32B);
    defer w1.deinit(allocator);
    var w2 = try OwnedF16.init2D(allocator, dims.ff, dims.hidden, 0xA32C);
    defer w2.deinit(allocator);
    var w3 = try OwnedF16.init2D(allocator, dims.hidden, dims.ff, 0xA32D);
    defer w3.deinit(allocator);

    const cache = graph.mlx_state_space_cache_create(1);
    defer graph.mlx_state_space_cache_free(cache);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        graph.mlx_state_space_cache_reset(cache);
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();
        var out: graph.ArrayHandle = x.handle;
        var step: usize = 0;
        while (step < dims.steps) : (step += 1) {
            const residual = out;
            const norm1 = graph.mlx_lazy_rms_norm(out, ln1_w.handle, 1.0e-5);
            const mixed = graph.mlx_lazy_gated_delta_mixer_bf16(
                norm1,
                in_proj.handle,
                conv_w.handle,
                conv_b.handle,
                a_log.handle,
                dt_bias.handle,
                norm_weight.handle,
                out_proj.handle,
                cache,
                0,
                dims.d_conv,
                dims.n_heads,
                dims.d_head,
            );
            const hidden_1 = graph.mlx_lazy_add(residual, mixed);
            const ffn = graph.mlx_lazy_rms_norm_fused_ffn_bf16(
                hidden_1,
                ln2_w.handle,
                w1.handle,
                w3.handle,
                w2.handle,
                1.0e-5,
            );
            out = graph.mlx_lazy_add(hidden_1, ffn);
        }
        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{out};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const gdm_in_proj_flops = @as(u128, 2) * dims.hidden * proj_dim;
    const gdm_out_proj_flops = @as(u128, 2) * d_inner * dims.hidden;
    const gdm_conv_flops = @as(u128, 2) * dims.d_conv * qkv_len + @as(u128, 3) * qkv_len;
    const recurrent_state_elems = @as(u128, dims.n_heads) * dims.d_head * dims.d_head;
    const gdm_recurrent_flops = @as(u128, 12) * recurrent_state_elems + @as(u128, 8) * dims.n_heads * dims.d_head;
    const ffn_flops = @as(u128, 6) * dims.hidden * dims.ff;
    const norm_flops = @as(u128, 12) * dims.hidden;
    const add_flops = @as(u128, 2) * dims.hidden;
    const flops_per_step = gdm_in_proj_flops + gdm_out_proj_flops + gdm_conv_flops + gdm_recurrent_flops + ffn_flops + norm_flops + add_flops;

    const gdm_in_proj_bytes = @as(u128, dims.hidden) * proj_dim * 2;
    const gdm_out_proj_bytes = @as(u128, d_inner) * dims.hidden * 2;
    const gdm_conv_bytes = (@as(u128, dims.d_conv) * qkv_len + qkv_len) * 2;
    const gdm_vec_bytes = (@as(u128, 2) * dims.n_heads + d_inner) * 2;
    const gdm_conv_state_bytes = (@as(u128, 2) * dims.d_conv * qkv_len) * 4;
    const gdm_recurrent_state_bytes = (@as(u128, 2) * dims.n_heads * dims.d_head * dims.d_head) * 4;
    const ffn_weight_bytes = @as(u128, 3) * dims.hidden * dims.ff * 2;
    const ffn_io_bytes = @as(u128, 2) * dims.hidden * 2;
    const norm_bytes = @as(u128, 2) * ((@as(u128, 2) * dims.hidden + dims.hidden) * 2);
    const add_bytes = @as(u128, 2) * (@as(u128, 3) * dims.hidden * 2);
    const bytes_per_step = gdm_in_proj_bytes + gdm_out_proj_bytes + gdm_conv_bytes + gdm_vec_bytes + gdm_conv_state_bytes + gdm_recurrent_state_bytes + ffn_weight_bytes + ffn_io_bytes + norm_bytes + add_bytes;

    return .{
        .name = "p1_gdblk",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = toU64Saturating(flops_per_step * dims.steps),
        .bytes_per_iter = toU64Saturating(bytes_per_step * dims.steps),
        .note = "gated-delta block decode method: norm + gated-delta + add + norm + ffn + add",
    };
}

pub fn runLmHeadF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const dims = profileLmHeadDims(cfg.profile);
    const token_shape = [_]i64{ 1, 1, @intCast(dims.hidden) };
    var x = try OwnedF16.initShape(allocator, &token_shape, 0xB101);
    defer x.deinit(allocator);
    var ln_final = try OwnedF16.init1D(allocator, dims.hidden, 0xB102);
    defer ln_final.deinit(allocator);
    var lm_head = try OwnedF16.init2D(allocator, dims.hidden, dims.vocab, 0xB103);
    defer lm_head.deinit(allocator);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();
        var logits: graph.ArrayHandle = null;
        var step: usize = 0;
        while (step < dims.steps) : (step += 1) {
            const final_normed = graph.mlx_lazy_rms_norm(x.handle, ln_final.handle, 1.0e-5);
            logits = graph.mlx_lazy_matmul(final_normed, lm_head.handle);
        }
        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{logits};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const flops_per_step = (@as(u128, 2) * dims.hidden * dims.vocab) + (@as(u128, 6) * dims.hidden);
    const bytes_per_step = (@as(u128, dims.hidden) * dims.vocab * 2) + (@as(u128, 3) * dims.hidden * 2);

    return .{
        .name = "p1_lmh_f16",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = toU64Saturating(flops_per_step * dims.steps),
        .bytes_per_iter = toU64Saturating(bytes_per_step * dims.steps),
        .note = "final rmsnorm plus dense f16 lm_head projection",
    };
}

pub fn runLmHeadBF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const dims = profileLmHeadDims(cfg.profile);
    const token_shape = [_]i64{ 1, 1, @intCast(dims.hidden) };
    var x = try OwnedBF16.initShape(allocator, &token_shape, 0xB201);
    defer x.deinit(allocator);
    var ln_final = try OwnedBF16Norm.init1D(allocator, dims.hidden, 0xB202);
    defer ln_final.deinit(allocator);
    var lm_head = try OwnedBF16DenseWeight.init2D(allocator, dims.hidden, dims.vocab, 0xB203);
    defer lm_head.deinit(allocator);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();
        var logits: graph.ArrayHandle = null;
        var step: usize = 0;
        while (step < dims.steps) : (step += 1) {
            const final_normed = graph.mlx_lazy_rms_norm(x.handle, ln_final.handle, 1.0e-5);
            logits = graph.mlx_lazy_matmul(final_normed, lm_head.handle);
        }
        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{logits};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const flops_per_step = (@as(u128, 2) * dims.hidden * dims.vocab) + (@as(u128, 6) * dims.hidden);
    const bytes_per_step = (@as(u128, dims.hidden) * dims.vocab * 2) + (@as(u128, 3) * dims.hidden * 2);

    return .{
        .name = "p1_lmh_bf16",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = toU64Saturating(flops_per_step * dims.steps),
        .bytes_per_iter = toU64Saturating(bytes_per_step * dims.steps),
        .note = "final rmsnorm plus dense bf16 lm_head projection using live weight-load policy",
    };
}

pub fn runLmHeadHostF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const dims = profileLmHeadDims(cfg.profile);
    const token_shape = [_]i64{ 1, 1, @intCast(dims.hidden) };
    var x = try OwnedF16.initShape(allocator, &token_shape, 0xB301);
    defer x.deinit(allocator);
    var ln_final = try OwnedF16.init1D(allocator, dims.hidden, 0xB302);
    defer ln_final.deinit(allocator);
    var lm_head = try OwnedF16.init2D(allocator, dims.hidden, dims.vocab, 0xB303);
    defer lm_head.deinit(allocator);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);
    const host_logits = try allocator.alloc(f32, dims.vocab);
    defer allocator.free(host_logits);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    var sink_token: u32 = 0;
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        var build_ns: u64 = 0;
        var eval_ns: u64 = 0;
        const total_start = std.time.nanoTimestamp();
        var step: usize = 0;
        while (step < dims.steps) : (step += 1) {
            graph.beginForwardGraphBuild();
            const t0 = std.time.nanoTimestamp();
            const final_normed = graph.mlx_lazy_rms_norm(x.handle, ln_final.handle, 1.0e-5);
            const logits = graph.mlx_lazy_matmul(final_normed, lm_head.handle);
            const t1 = std.time.nanoTimestamp();
            var handles = [_]graph.ArrayHandle{logits};
            const t2 = std.time.nanoTimestamp();
            graph.eval(&handles);
            const t3 = std.time.nanoTimestamp();
            graph.copyToHost(logits, host_logits);
            sink_token +%= argmaxHost(host_logits);
            build_ns +%= elapsedNs(t0, t1);
            eval_ns +%= elapsedNs(t2, t3);
            graph.freeArray(logits);
            graph.freeArray(final_normed);
        }
        const total_end = std.time.nanoTimestamp();
        const sample = harness.Sample{
            .build_ns = build_ns,
            .eval_ns = eval_ns,
            .total_ns = elapsedNs(total_start, total_end),
        };
        if (iter == 0) cold_first = sample;
        if (iter >= cfg.warmup and sample_idx < samples.len) {
            samples[sample_idx] = sample;
            sample_idx += 1;
        }
    }
    std.mem.doNotOptimizeAway(sink_token);

    const flops_per_step = (@as(u128, 2) * dims.hidden * dims.vocab) + (@as(u128, 6) * dims.hidden);
    const bytes_per_step = (@as(u128, dims.hidden) * dims.vocab * 2) + (@as(u128, 3) * dims.hidden * 2);

    return .{
        .name = "p1_lmhost",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = toU64Saturating(flops_per_step * dims.steps),
        .bytes_per_iter = toU64Saturating(bytes_per_step * dims.steps),
        .note = "final rmsnorm plus dense f16 lm_head projection with host logits copy and argmax",
    };
}

pub fn runGatedDeltaDecodeQuantizedU4(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const dims = profileGatedDeltaDecodeDims(cfg.profile);
    const bits: usize = 4;
    const group_size: usize = 64;
    if (dims.hidden % group_size != 0) return error.InvalidInput;
    if (dims.hidden % 8 != 0) return error.InvalidInput;

    const d_inner = dims.n_heads * dims.d_head;
    const qkv_len = 3 * d_inner;
    const proj_dim = (4 * d_inner) + (2 * dims.n_heads);

    const token_shape = [_]i64{ 1, 1, @intCast(dims.hidden) };
    const qkv_shape = [_]i64{@intCast(qkv_len)};
    const head_shape = [_]i64{@intCast(dims.n_heads)};
    const inner_shape = [_]i64{@intCast(d_inner)};
    var x = try OwnedBF16.initShape(allocator, &token_shape, 0xA311);
    defer x.deinit(allocator);

    const in_packed_k = dims.hidden * bits / 32;
    var in_w = try OwnedU32.init2D(allocator, proj_dim, in_packed_k, 0xA312);
    defer in_w.deinit(allocator);
    var in_s = try OwnedBF16.init2D(allocator, proj_dim, dims.hidden / group_size, 0xA313);
    defer in_s.deinit(allocator);
    var in_b = try OwnedBF16.init2D(allocator, proj_dim, dims.hidden / group_size, 0xA314);
    defer in_b.deinit(allocator);

    var conv_w = try OwnedBF16.init2D(allocator, dims.d_conv, qkv_len, 0xA315);
    defer conv_w.deinit(allocator);
    var conv_b = try OwnedBF16.initShape(allocator, &qkv_shape, 0xA316);
    defer conv_b.deinit(allocator);
    var a_log = try OwnedBF16.initShape(allocator, &head_shape, 0xA317);
    defer a_log.deinit(allocator);
    var dt_bias = try OwnedBF16.initShape(allocator, &head_shape, 0xA318);
    defer dt_bias.deinit(allocator);
    var norm_weight = try OwnedBF16.initShape(allocator, &inner_shape, 0xA319);
    defer norm_weight.deinit(allocator);

    const out_packed_k = d_inner * bits / 32;
    var out_w = try OwnedU32.init2D(allocator, dims.hidden, out_packed_k, 0xA31A);
    defer out_w.deinit(allocator);
    var out_s = try OwnedBF16.init2D(allocator, dims.hidden, d_inner / group_size, 0xA31B);
    defer out_s.deinit(allocator);
    var out_b = try OwnedBF16.init2D(allocator, dims.hidden, d_inner / group_size, 0xA31C);
    defer out_b.deinit(allocator);

    const cache = graph.mlx_state_space_cache_create(1);
    defer graph.mlx_state_space_cache_free(cache);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        graph.mlx_state_space_cache_reset(cache);
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();
        var out: graph.ArrayHandle = x.handle;
        var step: usize = 0;
        while (step < dims.steps) : (step += 1) {
            out = graph.mlx_lazy_gated_delta_mixer_quantized(
                out,
                in_w.handle,
                in_s.handle,
                in_b.handle,
                conv_w.handle,
                conv_b.handle,
                a_log.handle,
                dt_bias.handle,
                norm_weight.handle,
                out_w.handle,
                out_s.handle,
                out_b.handle,
                group_size,
                bits,
                cache,
                0,
                dims.d_conv,
                dims.n_heads,
                dims.d_head,
            );
        }
        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{out};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const in_proj_flops = @as(u128, 2) * dims.hidden * proj_dim;
    const out_proj_flops = @as(u128, 2) * d_inner * dims.hidden;
    const conv_flops = @as(u128, 2) * dims.d_conv * qkv_len + @as(u128, 3) * qkv_len;
    const recurrent_state_elems = @as(u128, dims.n_heads) * dims.d_head * dims.d_head;
    const recurrent_flops = @as(u128, 12) * recurrent_state_elems + @as(u128, 8) * dims.n_heads * dims.d_head;
    const flops_per_step = in_proj_flops + out_proj_flops + conv_flops + recurrent_flops;

    const in_weight_bytes = @as(u128, proj_dim) * dims.hidden * bits / 8;
    const in_scale_bias_bytes = @as(u128, proj_dim) * (dims.hidden / group_size) * 2 * 2;
    const out_weight_bytes = @as(u128, dims.hidden) * d_inner * bits / 8;
    const out_scale_bias_bytes = @as(u128, dims.hidden) * (d_inner / group_size) * 2 * 2;
    const conv_bytes = (@as(u128, dims.d_conv) * qkv_len + qkv_len + @as(u128, 2) * dims.n_heads + d_inner) * 2;
    const state_bytes = (@as(u128, 2) * dims.d_conv * qkv_len + @as(u128, 2) * dims.n_heads * dims.d_head * dims.d_head) * 4;
    const bytes_per_step = in_weight_bytes + in_scale_bias_bytes + out_weight_bytes + out_scale_bias_bytes + conv_bytes + state_bytes;

    return .{
        .name = "p1_gdmq_u4",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = toU64Saturating(flops_per_step * dims.steps),
        .bytes_per_iter = toU64Saturating(bytes_per_step * dims.steps),
        .note = "quantized gated-delta decode method with recurrent conv+delta state updates (u4)",
    };
}

pub fn runDecodeSynthF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const dims = profileDecodeDims(cfg.profile);
    const token_shape = [_]i64{ 1, 1, @intCast(dims.hidden) };
    var input_token = try OwnedF16.initShape(allocator, &token_shape, 0xBEEF);
    defer input_token.deinit(allocator);

    var layers = try allocator.alloc(DecodeLayer, dims.layers);
    defer allocator.free(layers);
    var initialized: usize = 0;
    errdefer {
        for (layers[0..initialized]) |*layer| layer.deinit(allocator);
    }
    for (layers, 0..) |*layer, idx| {
        layer.* = try DecodeLayer.init(allocator, dims.hidden, dims.ff, 0xA500 +% idx *% 17);
        initialized += 1;
    }
    defer for (layers) |*layer| layer.deinit(allocator);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();

        var x = input_token.handle;
        for (layers) |*layer| {
            const residual = x;
            const q = graph.mlx_lazy_matmul(x, layer.wq.handle);
            const k = graph.mlx_lazy_matmul(x, layer.wk.handle);
            const v = graph.mlx_lazy_matmul(x, layer.wv.handle);
            const qk = graph.mlx_lazy_add(q, k);
            const attn_mix = graph.mlx_lazy_add(qk, v);
            const o = graph.mlx_lazy_matmul(attn_mix, layer.wo.handle);
            const x_res = graph.mlx_lazy_add(residual, o);

            const norm = graph.mlx_lazy_rms_norm(x_res, layer.norm.handle, 1.0e-5);
            const ffn = graph.mlx_lazy_fused_ffn_bf16(
                norm,
                layer.w1.handle,
                layer.w3.handle,
                layer.w2.handle,
            );
            x = graph.mlx_lazy_add(x_res, ffn);
        }

        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{x};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const per_layer_flops = 2 * (@as(u128, 4) * dims.hidden * dims.hidden + @as(u128, 3) * dims.hidden * dims.ff);
    const per_layer_weight_bytes = (@as(u128, 4) * dims.hidden * dims.hidden + @as(u128, 3) * dims.hidden * dims.ff + dims.hidden) * 2;

    return .{
        .name = "p1_dec_mix",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = toU64Saturating(per_layer_flops * dims.layers),
        .bytes_per_iter = toU64Saturating(per_layer_weight_bytes * dims.layers),
        .note = "synthetic decode token path; bytes estimate is weight-stream lower bound",
    };
}

const DecodeDenseLayer = struct {
    // Attention path
    wq: OwnedF16,
    wo: OwnedF16,
    k_cache: OwnedF16,
    v_cache: OwnedF16,
    // ShortConv path
    conv_in: OwnedF16,
    conv_out: OwnedF16,
    // Shared FFN + norm
    w1: OwnedF16,
    w2: OwnedF16,
    w3: OwnedF16,
    norm: OwnedF16,
    // Layer type
    is_shortconv: bool,

    fn init(
        allocator: std.mem.Allocator,
        hidden: usize,
        ff: usize,
        heads: usize,
        head_dim: usize,
        kv_len: usize,
        _: usize,
        conv_dim: usize,
        is_shortconv: bool,
        seed: u64,
    ) !DecodeDenseLayer {
        const kv_shape = [_]i64{ 1, @intCast(heads), @intCast(kv_len), @intCast(head_dim) };
        return .{
            .wq = try OwnedF16.init2D(allocator, hidden, hidden, seed +% 1),
            .wo = try OwnedF16.init2D(allocator, hidden, hidden, seed +% 2),
            .k_cache = try OwnedF16.initShape(allocator, &kv_shape, seed +% 3),
            .v_cache = try OwnedF16.initShape(allocator, &kv_shape, seed +% 4),
            .conv_in = try OwnedF16.init2D(allocator, hidden, 3 * conv_dim, seed +% 5),
            .conv_out = try OwnedF16.init2D(allocator, conv_dim, hidden, seed +% 6),
            .w1 = try OwnedF16.init2D(allocator, hidden, ff, seed +% 7),
            .w2 = try OwnedF16.init2D(allocator, ff, hidden, seed +% 8),
            .w3 = try OwnedF16.init2D(allocator, hidden, ff, seed +% 9),
            .norm = try OwnedF16.init1D(allocator, hidden, seed +% 10),
            .is_shortconv = is_shortconv,
        };
    }

    fn deinit(self: *DecodeDenseLayer, allocator: std.mem.Allocator) void {
        self.wq.deinit(allocator);
        self.wo.deinit(allocator);
        self.k_cache.deinit(allocator);
        self.v_cache.deinit(allocator);
        self.conv_in.deinit(allocator);
        self.conv_out.deinit(allocator);
        self.w1.deinit(allocator);
        self.w2.deinit(allocator);
        self.w3.deinit(allocator);
        self.norm.deinit(allocator);
    }
};

const QuantLinearU4 = struct {
    w: OwnedU32,
    s: OwnedBF16,
    b: OwnedBF16,

    fn init(
        allocator: std.mem.Allocator,
        out_dim: usize,
        in_dim: usize,
        group_size: usize,
        seed: u64,
    ) !QuantLinearU4 {
        if (in_dim % group_size != 0) return error.InvalidInput;
        if (in_dim % 8 != 0) return error.InvalidInput;
        const packed_k = in_dim * 4 / 32;
        const groups = in_dim / group_size;
        return .{
            .w = try OwnedU32.init2D(allocator, out_dim, packed_k, seed +% 1),
            .s = try OwnedBF16.init2D(allocator, out_dim, groups, seed +% 2),
            .b = try OwnedBF16.init2D(allocator, out_dim, groups, seed +% 3),
        };
    }

    fn deinit(self: *QuantLinearU4, allocator: std.mem.Allocator) void {
        self.w.deinit(allocator);
        self.s.deinit(allocator);
        self.b.deinit(allocator);
    }
};

const DecodeQuantLayer = struct {
    is_shortconv: bool,
    norm: OwnedBF16,
    q_proj: QuantLinearU4,
    k_proj: QuantLinearU4,
    v_proj: QuantLinearU4,
    o_proj: QuantLinearU4,
    conv_in: QuantLinearU4,
    conv_out: QuantLinearU4,
    conv_weight: OwnedBF16,
    conv_bias: OwnedBF16,
    w1: QuantLinearU4,
    w2: QuantLinearU4,
    w3: QuantLinearU4,

    fn init(
        allocator: std.mem.Allocator,
        hidden: usize,
        ff: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        conv_dim: usize,
        d_conv: usize,
        group_size: usize,
        is_shortconv: bool,
        seed: u64,
    ) !DecodeQuantLayer {
        const kv_dim = kv_heads * head_dim;
        const norm_shape = [_]i64{@intCast(hidden)};
        return .{
            .is_shortconv = is_shortconv,
            .norm = try OwnedBF16.initShape(allocator, &norm_shape, seed +% 1),
            .q_proj = try QuantLinearU4.init(allocator, heads * head_dim, hidden, group_size, seed +% 10),
            .k_proj = try QuantLinearU4.init(allocator, kv_dim, hidden, group_size, seed +% 20),
            .v_proj = try QuantLinearU4.init(allocator, kv_dim, hidden, group_size, seed +% 30),
            .o_proj = try QuantLinearU4.init(allocator, hidden, heads * head_dim, group_size, seed +% 40),
            .conv_in = try QuantLinearU4.init(allocator, 3 * conv_dim, hidden, group_size, seed +% 50),
            .conv_out = try QuantLinearU4.init(allocator, hidden, conv_dim, group_size, seed +% 60),
            .conv_weight = try OwnedBF16.init2D(allocator, d_conv, conv_dim, seed +% 70),
            .conv_bias = try OwnedBF16.init2D(allocator, 1, conv_dim, seed +% 71),
            .w1 = try QuantLinearU4.init(allocator, ff, hidden, group_size, seed +% 80),
            .w2 = try QuantLinearU4.init(allocator, hidden, ff, group_size, seed +% 90),
            .w3 = try QuantLinearU4.init(allocator, ff, hidden, group_size, seed +% 100),
        };
    }

    fn deinit(self: *DecodeQuantLayer, allocator: std.mem.Allocator) void {
        self.norm.deinit(allocator);
        self.q_proj.deinit(allocator);
        self.k_proj.deinit(allocator);
        self.v_proj.deinit(allocator);
        self.o_proj.deinit(allocator);
        self.conv_in.deinit(allocator);
        self.conv_out.deinit(allocator);
        self.conv_weight.deinit(allocator);
        self.conv_bias.deinit(allocator);
        self.w1.deinit(allocator);
        self.w2.deinit(allocator);
        self.w3.deinit(allocator);
    }
};

pub fn runDecodeDenseF16(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const dims = profileDecodeDenseDims(cfg.profile);
    if (dims.hidden != dims.heads * dims.head_dim) return error.InvalidInput;

    const token_shape = [_]i64{ 1, 1, @intCast(dims.hidden) };
    var input_token = try OwnedF16.initShape(allocator, &token_shape, 0xD001);
    defer input_token.deinit(allocator);
    var ln_final = try OwnedF16.init1D(allocator, dims.hidden, 0xD002);
    defer ln_final.deinit(allocator);
    var lm_head = try OwnedF16.init2D(allocator, dims.hidden, dims.vocab, 0xD003);
    defer lm_head.deinit(allocator);

    var layers = try allocator.alloc(DecodeDenseLayer, dims.layers);
    defer allocator.free(layers);
    var initialized: usize = 0;
    errdefer for (layers[0..initialized]) |*layer| layer.deinit(allocator);
    for (layers, 0..) |*layer, idx| {
        layer.* = try DecodeDenseLayer.init(
            allocator,
            dims.hidden,
            dims.ff,
            dims.heads,
            dims.head_dim,
            dims.kv_len,
            dims.d_conv,
            dims.conv_dim,
            (idx % 2) == 0,
            0xD100 +% idx *% 17,
        );
        initialized += 1;
    }
    defer for (layers) |*layer| layer.deinit(allocator);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    const attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(dims.head_dim)));
    while (iter < total_iters) : (iter += 1) {
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();

        var x = input_token.handle;
        for (layers) |*layer| {
            const residual = x;
            const mixed = if (layer.is_shortconv) blk: {
                // Stable shortconv proxy for dense decode benchmark:
                // in_proj -> split gates -> multiplicative mixer -> out_proj.
                const bcx = graph.mlx_lazy_matmul(x, layer.conv_in.handle);
                const s0 = [_]c_int{ 0, 0, 0 };
                const e0 = [_]c_int{ 1, 1, @intCast(dims.conv_dim) };
                const s1 = [_]c_int{ 0, 0, @intCast(dims.conv_dim) };
                const e1 = [_]c_int{ 1, 1, @intCast(2 * dims.conv_dim) };
                const s2 = [_]c_int{ 0, 0, @intCast(2 * dims.conv_dim) };
                const e2 = [_]c_int{ 1, 1, @intCast(3 * dims.conv_dim) };
                const b_gate = graph.mlx_lazy_slice(bcx, &s0, &e0, 3);
                const c_gate = graph.mlx_lazy_slice(bcx, &s1, &e1, 3);
                const x_proj = graph.mlx_lazy_slice(bcx, &s2, &e2, 3);
                const bx = graph.mlx_lazy_multiply(b_gate, x_proj);
                const gated = graph.mlx_lazy_multiply(bx, c_gate);
                break :blk graph.mlx_lazy_matmul(gated, layer.conv_out.handle);
            } else blk: {
                const q_lin = graph.mlx_lazy_matmul(x, layer.wq.handle); // [1,1,H]
                const q_shape = [_]usize{ 1, dims.heads, 1, dims.head_dim };
                const q = graph.mlx_lazy_reshape(q_lin, &q_shape, 4);
                const attn = graph.mlx_lazy_attention(
                    q,
                    layer.k_cache.handle,
                    layer.v_cache.handle,
                    attn_scale,
                    true,
                );
                const flat_shape = [_]usize{ 1, 1, dims.hidden };
                const attn_flat = graph.mlx_lazy_reshape(attn, &flat_shape, 3);
                break :blk graph.mlx_lazy_matmul(attn_flat, layer.wo.handle);
            };

            const x_res = graph.mlx_lazy_add(residual, mixed);
            const norm = graph.mlx_lazy_rms_norm(x_res, layer.norm.handle, 1.0e-5);
            const ffn = graph.mlx_lazy_fused_ffn_bf16(
                norm,
                layer.w1.handle,
                layer.w3.handle,
                layer.w2.handle,
            );
            x = graph.mlx_lazy_add(x_res, ffn);
        }

        const final_normed = graph.mlx_lazy_rms_norm(x, ln_final.handle, 1.0e-5);
        const logits = graph.mlx_lazy_matmul(final_normed, lm_head.handle);

        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{logits};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const conv_layer_count = (dims.layers + 1) / 2;
    const attn_layer_count = dims.layers - conv_layer_count;
    const ffn_flops_per_layer = @as(u128, 6) * dims.hidden * dims.ff;
    const shortconv_flops_per_layer = @as(u128, 2) * dims.hidden * (3 * dims.conv_dim) +
        @as(u128, 2) * dims.conv_dim * dims.hidden +
        @as(u128, 2) * dims.conv_dim;
    const attn_proj_flops_per_layer = @as(u128, 4) * dims.hidden * dims.hidden;
    const attn_kernel_flops_per_layer = @as(u128, 4) * dims.heads * dims.kv_len * dims.head_dim;
    const lm_head_flops = @as(u128, 2) * dims.hidden * dims.vocab;
    const flops = toU64Saturating(@as(u128, conv_layer_count) * (shortconv_flops_per_layer + ffn_flops_per_layer) +
        @as(u128, attn_layer_count) * (attn_proj_flops_per_layer + attn_kernel_flops_per_layer + ffn_flops_per_layer) +
        lm_head_flops);

    const shortconv_weight_bytes_per_layer = (@as(u128, dims.hidden) * (3 * dims.conv_dim) +
        @as(u128, dims.conv_dim) * dims.hidden) * 2;
    const attn_weight_bytes_per_layer = (@as(u128, 2) * dims.hidden * dims.hidden) * 2;
    const ffn_weight_bytes_per_layer = (@as(u128, 3) * dims.hidden * dims.ff) * 2;
    const kv_cache_bytes_per_attn_layer = (@as(u128, 2) * dims.heads * dims.kv_len * dims.head_dim) * 2;
    const lm_head_bytes = @as(u128, dims.hidden) * dims.vocab * 2;
    const bytes = toU64Saturating(@as(u128, conv_layer_count) * (shortconv_weight_bytes_per_layer + ffn_weight_bytes_per_layer) +
        @as(u128, attn_layer_count) * (attn_weight_bytes_per_layer + ffn_weight_bytes_per_layer + kv_cache_bytes_per_attn_layer) +
        lm_head_bytes);

    return .{
        .name = "p1_dec_dense",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = flops,
        .bytes_per_iter = bytes,
        .note = "dense decode token path with shortconv+attention mix and logits projection",
    };
}

pub fn runDecodeQuantizedMixU4(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    try expectMetalReady();

    const dims = profileDecodeDenseDims(cfg.profile);
    const bits: usize = 4;
    const group_size: usize = 64;
    const kv_dim = dims.kv_heads * dims.head_dim;
    if (dims.hidden % group_size != 0 or dims.ff % group_size != 0 or dims.conv_dim % group_size != 0) return error.InvalidInput;
    if (dims.hidden % 8 != 0 or dims.ff % 8 != 0 or dims.conv_dim % 8 != 0) return error.InvalidInput;
    if (kv_dim % group_size != 0 or kv_dim % 8 != 0) return error.InvalidInput;

    const token_shape = [_]i64{ 1, 1, @intCast(dims.hidden) };
    var input_token = try OwnedBF16.initShape(allocator, &token_shape, 0xE001);
    defer input_token.deinit(allocator);
    const norm_shape = [_]i64{@intCast(dims.hidden)};
    var ln_final = try OwnedBF16.initShape(allocator, &norm_shape, 0xE002);
    defer ln_final.deinit(allocator);
    var lm_head = try QuantLinearU4.init(allocator, dims.vocab, dims.hidden, group_size, 0xE003);
    defer lm_head.deinit(allocator);

    var layers = try allocator.alloc(DecodeQuantLayer, dims.layers);
    defer allocator.free(layers);
    var initialized: usize = 0;
    errdefer for (layers[0..initialized]) |*layer| layer.deinit(allocator);
    for (layers, 0..) |*layer, idx| {
        layer.* = try DecodeQuantLayer.init(
            allocator,
            dims.hidden,
            dims.ff,
            dims.heads,
            dims.kv_heads,
            dims.head_dim,
            dims.conv_dim,
            dims.d_conv,
            group_size,
            (idx % 2) == 0,
            0xE100 +% idx *% 97,
        );
        initialized += 1;
    }
    defer for (layers) |*layer| layer.deinit(allocator);

    const kv_cache = graph.mlx_cache_create(dims.layers, dims.kv_len);
    defer graph.mlx_cache_free(kv_cache);
    const shortconv_cache = graph.mlx_causal_conv_cache_create(dims.layers);
    defer graph.mlx_causal_conv_cache_free(shortconv_cache);

    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    const total_iters = cfg.warmup + cfg.iters;
    var sample_idx: usize = 0;
    var iter: usize = 0;
    var cold_first: harness.Sample = .{ .build_ns = 0, .eval_ns = 0, .total_ns = 0 };
    graph.mlx_clear_memory_cache();
    while (iter < total_iters) : (iter += 1) {
        graph.mlx_cache_reset(kv_cache);
        graph.mlx_causal_conv_cache_reset(shortconv_cache);
        graph.beginForwardGraphBuild();
        const t0 = std.time.nanoTimestamp();

        var x = input_token.handle;
        for (layers, 0..) |*layer, layer_idx| {
            const residual = x;
            const mixed = if (layer.is_shortconv) blk: {
                break :blk graph.mlx_lazy_causal_conv_mixer_quantized(
                    x,
                    layer.conv_in.w.handle,
                    layer.conv_in.s.handle,
                    layer.conv_in.b.handle,
                    layer.conv_weight.handle,
                    layer.conv_bias.handle,
                    layer.conv_out.w.handle,
                    layer.conv_out.s.handle,
                    layer.conv_out.b.handle,
                    group_size,
                    bits,
                    shortconv_cache,
                    layer_idx,
                    dims.d_conv,
                    dims.conv_dim,
                );
            } else blk: {
                break :blk graph.mlx_lazy_fused_attention(
                    x,
                    layer.q_proj.w.handle,
                    layer.q_proj.s.handle,
                    layer.q_proj.b.handle,
                    layer.k_proj.w.handle,
                    layer.k_proj.s.handle,
                    layer.k_proj.b.handle,
                    layer.v_proj.w.handle,
                    layer.v_proj.s.handle,
                    layer.v_proj.b.handle,
                    layer.o_proj.w.handle,
                    layer.o_proj.s.handle,
                    layer.o_proj.b.handle,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    kv_cache,
                    layer_idx,
                    dims.heads,
                    dims.kv_heads,
                    dims.head_dim,
                    0,
                    10000.0,
                    null,
                    null,
                    0,
                    1.0e-5,
                    group_size,
                    bits,
                    0.0,
                    0.0,
                );
            };

            const x_res = graph.mlx_lazy_add(residual, mixed);
            const norm = graph.mlx_lazy_rms_norm(x_res, layer.norm.handle, 1.0e-5);
            const ffn = graph.mlx_lazy_fused_ffn(
                norm,
                layer.w1.w.handle,
                layer.w1.s.handle,
                layer.w1.b.handle,
                layer.w3.w.handle,
                layer.w3.s.handle,
                layer.w3.b.handle,
                layer.w2.w.handle,
                layer.w2.s.handle,
                layer.w2.b.handle,
                group_size,
                bits,
                false,
            );
            x = graph.mlx_lazy_add(x_res, ffn);
        }

        const final_normed = graph.mlx_lazy_rms_norm(x, ln_final.handle, 1.0e-5);
        const logits = graph.mlx_lazy_quantized_matmul(
            final_normed,
            lm_head.w.handle,
            lm_head.s.handle,
            lm_head.b.handle,
            group_size,
            bits,
            true,
        );

        const t1 = std.time.nanoTimestamp();
        var handles = [_]graph.ArrayHandle{logits};
        const t2 = std.time.nanoTimestamp();
        graph.eval(&handles);
        const t3 = std.time.nanoTimestamp();

        recordSample(&cold_first, samples, &sample_idx, iter, cfg.warmup, t0, t1, t2, t3);
    }

    const conv_layer_count = (dims.layers + 1) / 2;
    const attn_layer_count = dims.layers - conv_layer_count;

    const shortconv_proj_flops_per_layer = @as(u128, 2) * dims.hidden * (3 * dims.conv_dim) +
        @as(u128, 2) * dims.conv_dim * dims.hidden +
        @as(u128, 2) * dims.conv_dim;
    const attn_proj_flops_per_layer = @as(u128, 4) * dims.hidden * dims.hidden +
        @as(u128, 4) * dims.hidden * kv_dim;
    const attn_kernel_flops_per_layer = @as(u128, 4) * dims.heads * dims.kv_len * dims.head_dim;
    const ffn_flops_per_layer = @as(u128, 6) * dims.hidden * dims.ff;
    const lm_head_flops = @as(u128, 2) * dims.hidden * dims.vocab;
    const flops = toU64Saturating(
        @as(u128, conv_layer_count) * (shortconv_proj_flops_per_layer + ffn_flops_per_layer) +
            @as(u128, attn_layer_count) * (attn_proj_flops_per_layer + attn_kernel_flops_per_layer + ffn_flops_per_layer) +
            lm_head_flops,
    );

    const shortconv_in_bytes = @as(u128, 3 * dims.conv_dim) * dims.hidden * bits / 8 +
        @as(u128, 3 * dims.conv_dim) * (dims.hidden / group_size) * 2 * 2;
    const shortconv_out_bytes = @as(u128, dims.hidden) * dims.conv_dim * bits / 8 +
        @as(u128, dims.hidden) * (dims.conv_dim / group_size) * 2 * 2;
    const shortconv_conv_bytes = (@as(u128, dims.d_conv) * dims.conv_dim + @as(u128, dims.conv_dim)) * 2;
    const shortconv_state_bytes = (@as(u128, 2) * dims.d_conv * dims.conv_dim) * 4;
    const shortconv_bytes_per_layer = shortconv_in_bytes + shortconv_out_bytes + shortconv_conv_bytes + shortconv_state_bytes;

    const attn_q_bytes = @as(u128, dims.hidden) * dims.hidden * bits / 8 + @as(u128, dims.hidden) * (dims.hidden / group_size) * 2 * 2;
    const attn_kv_bytes = @as(u128, kv_dim) * dims.hidden * bits / 8 + @as(u128, kv_dim) * (dims.hidden / group_size) * 2 * 2;
    const attn_o_bytes = @as(u128, dims.hidden) * dims.hidden * bits / 8 + @as(u128, dims.hidden) * (dims.hidden / group_size) * 2 * 2;
    const kv_cache_bytes_per_attn_layer = (@as(u128, 2) * dims.kv_heads * dims.kv_len * dims.head_dim) * 2;
    const attn_bytes_per_layer = attn_q_bytes + 2 * attn_kv_bytes + attn_o_bytes + kv_cache_bytes_per_attn_layer;

    const ffn_w1_bytes = @as(u128, dims.ff) * dims.hidden * bits / 8 + @as(u128, dims.ff) * (dims.hidden / group_size) * 2 * 2;
    const ffn_w3_bytes = ffn_w1_bytes;
    const ffn_w2_bytes = @as(u128, dims.hidden) * dims.ff * bits / 8 + @as(u128, dims.hidden) * (dims.ff / group_size) * 2 * 2;
    const ffn_bytes_per_layer = ffn_w1_bytes + ffn_w3_bytes + ffn_w2_bytes;

    const lm_head_bytes = @as(u128, dims.vocab) * dims.hidden * bits / 8 + @as(u128, dims.vocab) * (dims.hidden / group_size) * 2 * 2;

    const bytes = toU64Saturating(
        @as(u128, conv_layer_count) * (shortconv_bytes_per_layer + ffn_bytes_per_layer) +
            @as(u128, attn_layer_count) * (attn_bytes_per_layer + ffn_bytes_per_layer) +
            lm_head_bytes,
    );

    return .{
        .name = "p1_dec_qmix",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .flops_per_iter = flops,
        .bytes_per_iter = bytes,
        .note = "quantized decode token path with shortconv+attention mix and quantized logits",
    };
}

test "runGatedDeltaBlockF16 returns gated-delta block scenario metadata" {
    if (comptime builtin.os.tag != .macos) return;
    if (!metal.isAvailable()) return;

    var result = try runGatedDeltaBlockF16(std.testing.allocator, .{
        .warmup = 0,
        .iters = 1,
        .profile = .ci,
    });
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("p1_gdblk", result.name);
    try std.testing.expect(result.flops_per_iter > 0);
    try std.testing.expect(result.bytes_per_iter > 0);
    try std.testing.expectEqual(@as(usize, 1), result.samples.len);
}

test "runLmHeadF16 returns lm_head f16 scenario metadata" {
    if (comptime builtin.os.tag != .macos) return;
    if (!metal.isAvailable()) return;

    var result = try runLmHeadF16(std.testing.allocator, .{
        .warmup = 0,
        .iters = 1,
        .profile = .ci,
    });
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("p1_lmh_f16", result.name);
    try std.testing.expect(result.flops_per_iter > 0);
    try std.testing.expect(result.bytes_per_iter > 0);
    try std.testing.expectEqual(@as(usize, 1), result.samples.len);
}

test "runLmHeadBF16 returns lm_head bf16 scenario metadata" {
    if (comptime builtin.os.tag != .macos) return;
    if (!metal.isAvailable()) return;

    var result = try runLmHeadBF16(std.testing.allocator, .{
        .warmup = 0,
        .iters = 1,
        .profile = .ci,
    });
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("p1_lmh_bf16", result.name);
    try std.testing.expect(result.flops_per_iter > 0);
    try std.testing.expect(result.bytes_per_iter > 0);
    try std.testing.expectEqual(@as(usize, 1), result.samples.len);
}

test "runLmHeadHostF16 returns lm_head host scenario metadata" {
    if (comptime builtin.os.tag != .macos) return;
    if (!metal.isAvailable()) return;

    var result = try runLmHeadHostF16(std.testing.allocator, .{
        .warmup = 0,
        .iters = 1,
        .profile = .ci,
    });
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("p1_lmhost", result.name);
    try std.testing.expect(result.flops_per_iter > 0);
    try std.testing.expect(result.bytes_per_iter > 0);
    try std.testing.expectEqual(@as(usize, 1), result.samples.len);
}
