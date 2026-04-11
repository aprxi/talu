//! Integration tests for CUDA flash prefill attention wrappers.

const std = @import("std");
const main = @import("main");
const cuda = main.core.compute.cuda;
const dtype = main.core.dtype;

const Shape = struct {
    n_heads: u32,
    kv_groups: u32,
    head_dim: u32,
    rope_dim: u32,
    q_rows: u32,
    seq_len: u32,
    position_base: u32,
};

test "flash_prefill_f16 matches CPU reference on grouped-head geometry" {
    if (cuda.probeRuntime() != .available) return error.SkipZigTest;

    var device = cuda.Device.init() catch |err| {
        if (err == error.CudaInitFailed or err == error.CudaNoDevices) return error.SkipZigTest;
        return err;
    };
    defer device.deinit();
    if (!device.supportsModuleLaunch()) return error.SkipZigTest;

    var registry = cuda.Registry.init(std.testing.allocator, &device);
    defer registry.deinit();
    try registry.loadEmbeddedModule(cuda.flash_prefill.embedded_module);
    const resolved = try registry.resolveFunction(
        cuda.flash_prefill.op_name_f16,
        cuda.flash_prefill.symbol_f16,
    );

    const shape = Shape{
        .n_heads = 24,
        .kv_groups = 6,
        .head_dim = 256,
        .rope_dim = 64,
        .q_rows = 5,
        .seq_len = 160,
        .position_base = 155,
    };
    const n_kv_heads = shape.n_heads / shape.kv_groups;
    const row_stride = n_kv_heads * shape.head_dim;
    const theta: f32 = 10000.0;
    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(shape.head_dim)));

    var prng = std.Random.DefaultPrng.init(0x83B2_A019_D5C1_477Eu64);
    const random = prng.random();

    const query_count = @as(usize, shape.q_rows) * @as(usize, shape.n_heads) * @as(usize, shape.head_dim);
    const cache_count = @as(usize, shape.seq_len) * @as(usize, row_stride);
    const out_count = query_count;

    const query_host = try std.testing.allocator.alloc(f32, query_count);
    defer std.testing.allocator.free(query_host);
    const key_cache_host = try std.testing.allocator.alloc(u16, cache_count);
    defer std.testing.allocator.free(key_cache_host);
    const value_cache_host = try std.testing.allocator.alloc(u16, cache_count);
    defer std.testing.allocator.free(value_cache_host);
    const gpu_out_host = try std.testing.allocator.alloc(f32, out_count);
    defer std.testing.allocator.free(gpu_out_host);
    const expected_host = try std.testing.allocator.alloc(f32, out_count);
    defer std.testing.allocator.free(expected_host);

    fillQuery(query_host, random);
    fillCache(key_cache_host, random);
    fillCache(value_cache_host, random);
    @memset(gpu_out_host, 0.0);
    @memset(expected_host, 0.0);

    computeFlashPrefillReference(
        expected_host,
        query_host,
        key_cache_host,
        value_cache_host,
        shape,
        row_stride,
        scale,
        theta,
    );

    var query_dev = try device.allocBuffer(query_count * @sizeOf(f32));
    defer query_dev.deinit(&device);
    var key_dev = try device.allocBuffer(cache_count * @sizeOf(u16));
    defer key_dev.deinit(&device);
    var value_dev = try device.allocBuffer(cache_count * @sizeOf(u16));
    defer value_dev.deinit(&device);
    var out_dev = try device.allocBuffer(out_count * @sizeOf(f32));
    defer out_dev.deinit(&device);

    try query_dev.upload(&device, std.mem.sliceAsBytes(query_host));
    try key_dev.upload(&device, std.mem.sliceAsBytes(key_cache_host));
    try value_dev.upload(&device, std.mem.sliceAsBytes(value_cache_host));
    try out_dev.upload(&device, std.mem.sliceAsBytes(gpu_out_host));

    var arg_pack = cuda.ArgPack.init(std.testing.allocator);
    defer arg_pack.deinit();

    try cuda.flash_prefill.runF16(
        &arg_pack,
        &device,
        resolved.function,
        &query_dev,
        &key_dev,
        &value_dev,
        &out_dev,
        shape.n_heads,
        shape.q_rows,
        shape.seq_len,
        row_stride,
        shape.kv_groups,
        shape.head_dim,
        scale,
        shape.rope_dim,
        shape.position_base,
        0,
        theta,
    );
    try device.synchronize();
    try out_dev.download(&device, std.mem.sliceAsBytes(gpu_out_host));

    try expectApproxEqSlices(expected_host, gpu_out_host, 0.02);
}

fn fillQuery(query: []f32, random: std.Random) void {
    for (query, 0..) |*value, idx| {
        const centered = random.float(f32) - 0.5;
        value.* = centered * 0.35 + @as(f32, @floatFromInt(idx % 13)) * 0.01;
    }
}

fn fillCache(cache: []u16, random: std.Random) void {
    for (cache, 0..) |*bits, idx| {
        const centered = random.float(f32) - 0.5;
        const value = centered * 0.4 + @as(f32, @floatFromInt(idx % 7)) * 0.015;
        bits.* = encodeF16(value);
    }
}

fn computeFlashPrefillReference(
    out: []f32,
    query: []const f32,
    key_cache: []const u16,
    value_cache: []const u16,
    shape: Shape,
    row_stride: u32,
    scale: f32,
    theta: f32,
) void {
    const head_dim_usize: usize = @intCast(shape.head_dim);
    const rope_dim_usize: usize = @intCast(shape.rope_dim);
    const half_rope = rope_dim_usize / 2;
    const n_kv_heads = shape.n_heads / shape.kv_groups;
    const inv_scale = @as(f32, @floatFromInt(shape.rope_dim));

    var q_rot = std.ArrayList(f32).init(std.testing.allocator);
    defer q_rot.deinit();
    q_rot.resize(head_dim_usize) catch unreachable;

    var scores = std.ArrayList(f32).init(std.testing.allocator);
    defer scores.deinit();
    scores.resize(shape.seq_len) catch unreachable;

    for (0..shape.q_rows) |q_row| {
        const q_pos = shape.position_base + @as(u32, @intCast(q_row));
        for (0..shape.n_heads) |head| {
            const kv_head = head / shape.kv_groups;
            const query_base = (@as(usize, q_row) * @as(usize, shape.n_heads) + @as(usize, head)) * head_dim_usize;
            rotateQueryInto(
                q_rot.items,
                query[query_base .. query_base + head_dim_usize],
                half_rope,
                rope_dim_usize,
                q_pos,
                theta,
                inv_scale,
            );

            var max_score = -std.math.inf(f32);
            for (0..shape.seq_len) |token| {
                if (token > q_pos) {
                    scores.items[token] = -std.math.inf(f32);
                    continue;
                }
                const key_base = (token * @as(usize, row_stride) + @as(usize, kv_head) * head_dim_usize);
                var dot: f32 = 0.0;
                for (0..head_dim_usize) |d| {
                    dot += q_rot.items[d] * dtype.fp16ToF32(key_cache[key_base + d]);
                }
                const score = dot * scale;
                scores.items[token] = score;
                max_score = @max(max_score, score);
            }

            var sum: f32 = 0.0;
            for (0..shape.seq_len) |token| {
                if (token > q_pos) continue;
                const exp_score = @exp(scores.items[token] - max_score);
                scores.items[token] = exp_score;
                sum += exp_score;
            }

            const out_base = query_base;
            @memset(out[out_base .. out_base + head_dim_usize], 0.0);
            const inv_sum = 1.0 / @max(sum, 1.0e-20);
            for (0..shape.seq_len) |token| {
                if (token > q_pos) continue;
                const weight = scores.items[token] * inv_sum;
                const value_base = (token * @as(usize, row_stride) + @as(usize, kv_head) * head_dim_usize);
                for (0..head_dim_usize) |d| {
                    out[out_base + d] += weight * dtype.fp16ToF32(value_cache[value_base + d]);
                }
            }
        }
        _ = n_kv_heads;
    }
}

fn rotateQueryInto(
    out: []f32,
    input: []const f32,
    half_rope: usize,
    rope_dim: usize,
    position: u32,
    theta: f32,
    rope_dim_f32: f32,
) void {
    std.mem.copyForwards(f32, out, input);
    for (0..half_rope) |pair| {
        const pair_f32: f32 = @floatFromInt(pair);
        const inv_freq = std.math.pow(f32, theta, (-2.0 * pair_f32) / rope_dim_f32);
        const angle = @as(f32, @floatFromInt(position)) * inv_freq;
        const sin_v = @sin(angle);
        const cos_v = @cos(angle);
        const lo = input[pair];
        const hi = input[half_rope + pair];
        out[pair] = lo * cos_v - hi * sin_v;
        out[half_rope + pair] = lo * sin_v + hi * cos_v;
    }
    _ = rope_dim;
}

fn encodeF16(value: f32) u16 {
    const narrowed: f16 = @floatCast(value);
    return @bitCast(narrowed);
}

fn expectApproxEqSlices(expected: []const f32, actual: []const f32, tolerance: f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual, 0..) |want, got, idx| {
        std.testing.expectApproxEqAbs(want, got, tolerance) catch |err| {
            std.debug.print(
                "flash_prefill mismatch idx={} want={d:.6} got={d:.6} abs_diff={d:.6}\n",
                .{ idx, want, got, @abs(want - got) },
            );
            return err;
        };
    }
}
