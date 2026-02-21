//! CUDA cuBLAS-backed matrix multiplication primitives.
//!
//! This is the first CUDA compute primitive: f32 GEMM on device buffers.

const std = @import("std");
const builtin = @import("builtin");
const device_mod = @import("device.zig");

const cublas_status_success: c_int = 0;
const cublas_status_alloc_failed: c_int = 3;
const cublas_op_n: c_int = 0;

const CublasCreateFn = *const fn (*?*anyopaque) callconv(.c) c_int;
const CublasDestroyFn = *const fn (?*anyopaque) callconv(.c) c_int;
const CublasSgemmFn = *const fn (
    ?*anyopaque,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const f32,
    [*]const f32,
    c_int,
    [*]const f32,
    c_int,
    *const f32,
    [*]f32,
    c_int,
) callconv(.c) c_int;

const CublasApi = struct {
    cublas_create: CublasCreateFn,
    cublas_destroy: CublasDestroyFn,
    cublas_sgemm: CublasSgemmFn,
};

pub const Blas = struct {
    lib: std.DynLib,
    api: CublasApi,
    handle: ?*anyopaque,

    pub fn init(device: *device_mod.Device) !Blas {
        try device.makeCurrent();

        var lib = try openCublasLibrary();
        errdefer lib.close();

        const api = try loadCublasApi(&lib);

        var handle: ?*anyopaque = null;
        const create_status = api.cublas_create(&handle);
        if (create_status == cublas_status_alloc_failed) return error.OutOfMemory;
        if (create_status != cublas_status_success or handle == null) return error.CublasCreateFailed;

        return .{
            .lib = lib,
            .api = api,
            .handle = handle,
        };
    }

    pub fn deinit(self: *Blas, device: *device_mod.Device) void {
        device.makeCurrent() catch {};
        if (self.handle) |handle| {
            _ = self.api.cublas_destroy(handle);
            self.handle = null;
        }
        self.lib.close();
    }

    /// F32 matrix multiplication: C = A @ B.
    /// A: [m x k], B: [k x n], C: [m x n] in row-major contiguous layout.
    pub fn matmulF32(
        self: *Blas,
        device: *device_mod.Device,
        a: *const device_mod.Buffer,
        m: usize,
        k: usize,
        b: *const device_mod.Buffer,
        n: usize,
        c: *device_mod.Buffer,
    ) !void {
        if (self.handle == null) return error.CublasHandleInvalid;
        if (!dimsFitCublas(m, n, k)) return error.InvalidArgument;
        if (a.size < m * k * @sizeOf(f32)) return error.InvalidArgument;
        if (b.size < k * n * @sizeOf(f32)) return error.InvalidArgument;
        if (c.size < m * n * @sizeOf(f32)) return error.InvalidArgument;

        try device.makeCurrent();

        // cuBLAS is column-major. For row-major C=A@B, compute C^T=B^T@A^T by
        // swapping operands and dimensions in the column-major GEMM call.
        const alpha: f32 = 1.0;
        const beta: f32 = 0.0;

        const b_dev = @as([*]const f32, @ptrFromInt(b.pointer));
        const a_dev = @as([*]const f32, @ptrFromInt(a.pointer));
        const c_dev = @as([*]f32, @ptrFromInt(c.pointer));

        const status = self.api.cublas_sgemm(
            self.handle,
            cublas_op_n,
            cublas_op_n,
            @intCast(n),
            @intCast(m),
            @intCast(k),
            &alpha,
            b_dev,
            @intCast(n),
            a_dev,
            @intCast(k),
            &beta,
            c_dev,
            @intCast(n),
        );
        if (status == cublas_status_alloc_failed) return error.OutOfMemory;
        if (status != cublas_status_success) return error.CublasMatmulFailed;
    }
};

fn dimsFitCublas(m: usize, n: usize, k: usize) bool {
    return m <= std.math.maxInt(c_int) and n <= std.math.maxInt(c_int) and k <= std.math.maxInt(c_int);
}

fn openCublasLibrary() !std.DynLib {
    const names: []const []const u8 = switch (builtin.os.tag) {
        .linux => &.{ "libcublas.so.12", "libcublas.so.11", "libcublas.so" },
        .windows => &.{ "cublas64_12.dll", "cublas64_11.dll" },
        else => &.{},
    };
    for (names) |name| {
        if (std.DynLib.open(name)) |lib| return lib else |_| {}
    }
    return error.CublasUnavailable;
}

fn lookupRequired(comptime T: type, lib: *std.DynLib, symbol: [:0]const u8) !T {
    return lib.lookup(T, symbol) orelse error.CublasSymbolMissing;
}

fn loadCublasApi(lib: *std.DynLib) !CublasApi {
    return .{
        .cublas_create = try lookupRequired(CublasCreateFn, lib, "cublasCreate_v2"),
        .cublas_destroy = try lookupRequired(CublasDestroyFn, lib, "cublasDestroy_v2"),
        .cublas_sgemm = try lookupRequired(CublasSgemmFn, lib, "cublasSgemm_v2"),
    };
}

test "dimsFitCublas accepts small dimensions" {
    try std.testing.expect(dimsFitCublas(4, 8, 16));
}

test "dimsFitCublas rejects c_int overflow dimensions" {
    const overflow = @as(usize, @intCast(std.math.maxInt(c_int))) + 1;
    try std.testing.expect(!dimsFitCublas(overflow, 1, 1));
}
