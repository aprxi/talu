//! CUDA cuBLAS-backed matrix multiplication primitives.
//!
//! This is the first CUDA compute primitive: f32 GEMM on device buffers.

const std = @import("std");
const builtin = @import("builtin");
const device_mod = @import("device.zig");

const cublas_status_success: c_int = 0;
const cublas_status_alloc_failed: c_int = 3;
const cublas_op_n: c_int = 0;
const cublas_op_t: c_int = 1;
const cublas_gemm_default: c_int = -1;
const cublas_default_math: c_int = 0;
const cublas_compute_32f: c_int = 68;
const cublas_compute_32i: c_int = 72;
const cuda_r_32f: c_int = 0;
const cuda_r_16f: c_int = 2;
const cuda_r_8i: c_int = 3;
const cuda_r_32i: c_int = 10;
const cuda_r_16bf: c_int = 14;

const CublasCreateFn = *const fn (*?*anyopaque) callconv(.c) c_int;
const CublasDestroyFn = *const fn (?*anyopaque) callconv(.c) c_int;
const CublasSetMathModeFn = *const fn (?*anyopaque, c_int) callconv(.c) c_int;
const CublasSetStreamFn = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) c_int;
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
const CublasGemmExFn = *const fn (
    ?*anyopaque,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    ?*const anyopaque,
    ?*const anyopaque,
    c_int,
    c_int,
    ?*const anyopaque,
    c_int,
    c_int,
    ?*const anyopaque,
    ?*anyopaque,
    c_int,
    c_int,
    c_int,
    c_int,
) callconv(.c) c_int;

const CublasApi = struct {
    cublas_create: CublasCreateFn,
    cublas_destroy: CublasDestroyFn,
    cublas_set_math_mode: CublasSetMathModeFn,
    cublas_set_stream: CublasSetStreamFn,
    cublas_sgemm: CublasSgemmFn,
    cublas_gemm_ex: CublasGemmExFn,
};

pub const Blas = struct {
    pub const U16Payload = enum {
        f16,
        bf16,
    };

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
        if (api.cublas_set_math_mode(handle, cublas_default_math) != cublas_status_success) {
            _ = api.cublas_destroy(handle);
            return error.CublasMathModeFailed;
        }

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
        if (self.api.cublas_set_stream(self.handle, device.getLaunchStream()) != cublas_status_success) {
            return error.CublasStreamSetFailed;
        }

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

    /// Mixed-precision matrix multiplication:
    /// C = input_f32 @ weight_u16^T
    /// input_f32: [rows x in_dim] row-major f32
    /// weight_u16: [out_dim x in_dim] row-major u16 payload (f16/bf16)
    /// out_f32: [rows x out_dim] row-major f32
    pub fn matmulU16F32(
        self: *Blas,
        device: *device_mod.Device,
        input_f32: *const device_mod.Buffer,
        rows: usize,
        in_dim: usize,
        weight_u16: *const device_mod.Buffer,
        out_dim: usize,
        out_f32: *device_mod.Buffer,
        payload: U16Payload,
    ) !void {
        if (self.handle == null) return error.CublasHandleInvalid;
        if (!dimsFitCublas(rows, out_dim, in_dim)) return error.InvalidArgument;
        if (input_f32.size < rows * in_dim * @sizeOf(f32)) return error.InvalidArgument;
        if (weight_u16.size < out_dim * in_dim * @sizeOf(u16)) return error.InvalidArgument;
        if (out_f32.size < rows * out_dim * @sizeOf(f32)) return error.InvalidArgument;

        try device.makeCurrent();
        if (self.api.cublas_set_stream(self.handle, device.getLaunchStream()) != cublas_status_success) {
            return error.CublasStreamSetFailed;
        }

        const alpha: f32 = 1.0;
        const beta: f32 = 0.0;
        const weight_type: c_int = switch (payload) {
            .f16 => cuda_r_16f,
            .bf16 => cuda_r_16bf,
        };

        const weight_dev: ?*const anyopaque = @ptrFromInt(weight_u16.pointer);
        const input_dev: ?*const anyopaque = @ptrFromInt(input_f32.pointer);
        const out_dev: ?*anyopaque = @ptrFromInt(out_f32.pointer);

        // Row-major C = A @ W^T is computed as column-major C^T = W @ A^T.
        // W row-major [out_dim x in_dim] is column-major [in_dim x out_dim], so
        // use transa=T to interpret it as [out_dim x in_dim].
        const status = self.api.cublas_gemm_ex(
            self.handle,
            cublas_op_t, // W^T view -> [out_dim x in_dim]
            cublas_op_n, // A^T view is already column-major [in_dim x rows]
            @intCast(out_dim),
            @intCast(rows),
            @intCast(in_dim),
            @ptrCast(&alpha),
            weight_dev,
            weight_type,
            @intCast(in_dim),
            input_dev,
            cuda_r_32f,
            @intCast(in_dim),
            @ptrCast(&beta),
            out_dev,
            cuda_r_32f,
            @intCast(out_dim),
            cublas_compute_32f,
            cublas_gemm_default,
        );
        if (status == cublas_status_alloc_failed) return error.OutOfMemory;
        if (status != cublas_status_success) return error.CublasMatmulFailed;
    }

    /// Tensor-core path:
    /// C = input_u16 @ weight_u16^T
    /// input_u16: [rows x in_dim] row-major u16 payload (f16/bf16)
    /// weight_u16: [out_dim x in_dim] row-major u16 payload (f16/bf16)
    /// out_f32: [rows x out_dim] row-major f32
    pub fn matmulU16U16F32(
        self: *Blas,
        device: *device_mod.Device,
        input_u16: *const device_mod.Buffer,
        input_payload: U16Payload,
        rows: usize,
        in_dim: usize,
        weight_u16: *const device_mod.Buffer,
        weight_payload: U16Payload,
        out_dim: usize,
        out_f32: *device_mod.Buffer,
    ) !void {
        if (self.handle == null) return error.CublasHandleInvalid;
        if (!dimsFitCublas(rows, out_dim, in_dim)) return error.InvalidArgument;
        if (input_u16.size < rows * in_dim * @sizeOf(u16)) return error.InvalidArgument;
        if (weight_u16.size < out_dim * in_dim * @sizeOf(u16)) return error.InvalidArgument;
        if (out_f32.size < rows * out_dim * @sizeOf(f32)) return error.InvalidArgument;

        try device.makeCurrent();
        if (self.api.cublas_set_stream(self.handle, device.getLaunchStream()) != cublas_status_success) {
            return error.CublasStreamSetFailed;
        }

        const alpha: f32 = 1.0;
        const beta: f32 = 0.0;
        const input_type: c_int = switch (input_payload) {
            .f16 => cuda_r_16f,
            .bf16 => cuda_r_16bf,
        };
        const weight_type: c_int = switch (weight_payload) {
            .f16 => cuda_r_16f,
            .bf16 => cuda_r_16bf,
        };

        const input_dev: ?*const anyopaque = @ptrFromInt(input_u16.pointer);
        const weight_dev: ?*const anyopaque = @ptrFromInt(weight_u16.pointer);
        const out_dev: ?*anyopaque = @ptrFromInt(out_f32.pointer);

        // Row-major C = A @ W^T is computed as column-major C^T = W @ A^T.
        const status = self.api.cublas_gemm_ex(
            self.handle,
            cublas_op_t, // W^T view -> [out_dim x in_dim]
            cublas_op_n, // A^T view -> [in_dim x rows]
            @intCast(out_dim),
            @intCast(rows),
            @intCast(in_dim),
            @ptrCast(&alpha),
            weight_dev,
            weight_type,
            @intCast(in_dim),
            input_dev,
            input_type,
            @intCast(in_dim),
            @ptrCast(&beta),
            out_dev,
            cuda_r_32f,
            @intCast(out_dim),
            cublas_compute_32f,
            cublas_gemm_default,
        );
        if (status == cublas_status_alloc_failed) return error.OutOfMemory;
        if (status != cublas_status_success) return error.CublasMatmulFailed;
    }

    /// INT8 tensor core path:
    /// C_i32 = A_i8 @ B_i8^T
    /// A_i8: [rows x in_dim] row-major signed int8
    /// B_i8: [out_dim x in_dim] row-major signed int8
    /// C_i32: [rows x out_dim] row-major int32
    pub fn matmulI8I8I32(
        self: *Blas,
        device: *device_mod.Device,
        input_i8: *const device_mod.Buffer,
        rows: usize,
        in_dim: usize,
        weight_i8: *const device_mod.Buffer,
        out_dim: usize,
        out_i32: *device_mod.Buffer,
    ) !void {
        if (self.handle == null) return error.CublasHandleInvalid;
        if (!dimsFitCublas(rows, out_dim, in_dim)) return error.InvalidArgument;
        if (input_i8.size < rows * in_dim) return error.InvalidArgument;
        if (weight_i8.size < out_dim * in_dim) return error.InvalidArgument;
        if (out_i32.size < rows * out_dim * @sizeOf(i32)) return error.InvalidArgument;

        try device.makeCurrent();
        if (self.api.cublas_set_stream(self.handle, device.getLaunchStream()) != cublas_status_success) {
            return error.CublasStreamSetFailed;
        }

        const alpha: i32 = 1;
        const beta: i32 = 0;

        const input_dev: ?*const anyopaque = @ptrFromInt(input_i8.pointer);
        const weight_dev: ?*const anyopaque = @ptrFromInt(weight_i8.pointer);
        const out_dev: ?*anyopaque = @ptrFromInt(out_i32.pointer);

        // Row-major C = A @ W^T is computed as column-major C^T = W @ A^T.
        const status = self.api.cublas_gemm_ex(
            self.handle,
            cublas_op_t, // W^T
            cublas_op_n, // A^T
            @intCast(out_dim),
            @intCast(rows),
            @intCast(in_dim),
            @ptrCast(&alpha),
            weight_dev,
            cuda_r_8i,
            @intCast(in_dim),
            input_dev,
            cuda_r_8i,
            @intCast(in_dim),
            @ptrCast(&beta),
            out_dev,
            cuda_r_32i,
            @intCast(out_dim),
            cublas_compute_32i,
            cublas_gemm_default,
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

fn lookupRequiredAny(comptime T: type, lib: *std.DynLib, symbols: []const [:0]const u8) !T {
    for (symbols) |symbol| {
        if (lib.lookup(T, symbol)) |fn_ptr| return fn_ptr;
    }
    return error.CublasSymbolMissing;
}

fn loadCublasApi(lib: *std.DynLib) !CublasApi {
    return .{
        .cublas_create = try lookupRequired(CublasCreateFn, lib, "cublasCreate_v2"),
        .cublas_destroy = try lookupRequired(CublasDestroyFn, lib, "cublasDestroy_v2"),
        .cublas_set_math_mode = try lookupRequiredAny(CublasSetMathModeFn, lib, &.{ "cublasSetMathMode_v2", "cublasSetMathMode" }),
        .cublas_set_stream = try lookupRequiredAny(CublasSetStreamFn, lib, &.{ "cublasSetStream_v2", "cublasSetStream" }),
        .cublas_sgemm = try lookupRequired(CublasSgemmFn, lib, "cublasSgemm_v2"),
        .cublas_gemm_ex = try lookupRequired(CublasGemmExFn, lib, "cublasGemmEx"),
    };
}

test "dimsFitCublas accepts small dimensions" {
    try std.testing.expect(dimsFitCublas(4, 8, 16));
}

test "dimsFitCublas rejects c_int overflow dimensions" {
    const overflow = @as(usize, @intCast(std.math.maxInt(c_int))) + 1;
    try std.testing.expect(!dimsFitCublas(overflow, 1, 1));
}
