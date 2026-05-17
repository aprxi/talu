//! CPU KV-cache to CUDA mirror transport tests.

const std = @import("std");
const main = @import("main");

const cuda_kv_mirror = main.inference.transport.cuda_kv_mirror;

const MockUploadTrace = struct {
    calls: usize = 0,
    pointers: [4]u64 = [_]u64{0} ** 4,
    sizes: [4]usize = [_]usize{0} ** 4,
    values: [4][8]f32 = [_][8]f32{[_]f32{0} ** 8} ** 4,
    value_counts: [4]usize = [_]usize{0} ** 4,
};

const MockDevice = struct {};

const MockCudaBuffer = struct {
    pointer: u64,
    size: usize,
    trace: *MockUploadTrace,

    pub fn upload(self: *const @This(), _: *MockDevice, data: []const u8) !void {
        const index = self.trace.calls;
        self.trace.calls += 1;
        self.trace.pointers[index] = self.pointer;
        self.trace.sizes[index] = self.size;
        const value_count = data.len / @sizeOf(f16);
        self.trace.value_counts[index] = value_count;
        for (0..value_count) |value_index| {
            var bits: u16 = 0;
            const byte_offset = value_index * @sizeOf(u16);
            @memcpy(std.mem.asBytes(&bits), data[byte_offset..][0..@sizeOf(u16)]);
            const value: f16 = @bitCast(bits);
            self.trace.values[index][value_index] = @floatCast(value);
        }
    }
};

const MockMirror = struct {
    k: MockCudaBuffer,
    v: MockCudaBuffer,
};

const MockReplicatedSource = struct {
    global_layer_idx: usize,
    kv_dim: usize,
};

const MockBlockRuntime = struct {
    replicated_kv_sources: []const MockReplicatedSource,
    mirror_kv: []MockMirror,
};

const MockGpuBackend = struct {
    allocator: std.mem.Allocator,
    block_runtime: MockBlockRuntime,
    head_dim: usize = 2,
    kv_cache_dtype: enum { f16, f32 } = .f16,
    device: MockDevice = .{},
};

const MockLayerKv = struct {
    k: [2][3][2]f32,
    v: [2][3][2]f32,

    pub fn getK(self: *const @This(), _: usize, kv_head: usize, position: usize) []const f32 {
        return self.k[kv_head][position][0..];
    }

    pub fn getV(self: *const @This(), _: usize, kv_head: usize, position: usize) []const f32 {
        return self.v[kv_head][position][0..];
    }
};

const MockCpuKvCache = struct {
    layer: MockLayerKv,

    pub fn getLayer(self: *const @This(), _: usize) *const MockLayerKv {
        return &self.layer;
    }
};

const MockCpuBackend = struct {
    kv_cache: MockCpuKvCache,
};

test "uploadCpuKvToCudaMirrors transposes CPU heads into contiguous CUDA rows" {
    var trace = MockUploadTrace{};
    var mirrors = [_]MockMirror{.{
        .k = .{ .pointer = 100, .size = 64, .trace = &trace },
        .v = .{ .pointer = 200, .size = 64, .trace = &trace },
    }};
    const sources = [_]MockReplicatedSource{.{
        .global_layer_idx = 7,
        .kv_dim = 4,
    }};
    var gpu = MockGpuBackend{
        .allocator = std.testing.allocator,
        .block_runtime = .{
            .replicated_kv_sources = sources[0..],
            .mirror_kv = mirrors[0..],
        },
    };
    var cpu = MockCpuBackend{ .kv_cache = .{ .layer = .{
        .k = .{
            .{ .{ 0, 1 }, .{ 10, 11 }, .{ 20, 21 } },
            .{ .{ 100, 101 }, .{ 110, 111 }, .{ 120, 121 } },
        },
        .v = .{
            .{ .{ 1000, 1001 }, .{ 1010, 1011 }, .{ 1020, 1021 } },
            .{ .{ 1100, 1101 }, .{ 1110, 1111 }, .{ 1120, 1121 } },
        },
    } } };

    try cuda_kv_mirror.uploadCpuKvToCudaMirrors(&gpu, &cpu, 0, 1, 2);

    try std.testing.expectEqual(@as(usize, 2), trace.calls);
    try std.testing.expectEqual(@as(u64, 108), trace.pointers[0]);
    try std.testing.expectEqual(@as(u64, 208), trace.pointers[1]);
    try std.testing.expectEqual(@as(usize, 16), trace.sizes[0]);
    try std.testing.expectEqual(@as(usize, 16), trace.sizes[1]);
    try std.testing.expectEqual(@as(usize, 8), trace.value_counts[0]);
    try std.testing.expectEqualSlices(f32, &.{ 10, 11, 110, 111, 20, 21, 120, 121 }, trace.values[0][0..8]);
    try std.testing.expectEqualSlices(f32, &.{ 1010, 1011, 1110, 1111, 1020, 1021, 1120, 1121 }, trace.values[1][0..8]);
}

test "uploadCpuKvToCudaMirrors rejects non-f16 mirror dtype when replication is active" {
    var trace = MockUploadTrace{};
    var mirrors = [_]MockMirror{.{
        .k = .{ .pointer = 0, .size = 16, .trace = &trace },
        .v = .{ .pointer = 0, .size = 16, .trace = &trace },
    }};
    const sources = [_]MockReplicatedSource{.{
        .global_layer_idx = 0,
        .kv_dim = 2,
    }};
    var gpu = MockGpuBackend{
        .allocator = std.testing.allocator,
        .block_runtime = .{
            .replicated_kv_sources = sources[0..],
            .mirror_kv = mirrors[0..],
        },
        .kv_cache_dtype = .f32,
    };
    var cpu = MockCpuBackend{ .kv_cache = .{ .layer = .{
        .k = [_][3][2]f32{[_][2]f32{[_]f32{ 0, 0 }} ** 3} ** 2,
        .v = [_][3][2]f32{[_][2]f32{[_]f32{ 0, 0 }} ** 3} ** 2,
    } } };

    try std.testing.expectError(error.UnsupportedModel, cuda_kv_mirror.uploadCpuKvToCudaMirrors(&gpu, &cpu, 0, 0, 1));
}

test "uploadCpuKvToCudaMirrors no-ops when backend has no block runtime" {
    const Gpu = struct {};
    var gpu = Gpu{};
    var cpu = MockCpuBackend{ .kv_cache = .{ .layer = .{
        .k = [_][3][2]f32{[_][2]f32{[_]f32{ 0, 0 }} ** 3} ** 2,
        .v = [_][3][2]f32{[_][2]f32{[_]f32{ 0, 0 }} ** 3} ** 2,
    } } };

    try cuda_kv_mirror.uploadCpuKvToCudaMirrors(&gpu, &cpu, 0, 0, 1);
}
