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
const cuda_r_8f_e4m3: c_int = 28;
const cuda_r_4f_e2m1: c_int = 33;

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
const CublasGemmStridedBatchedExFn = *const fn (
    ?*anyopaque, // handle
    c_int, // transa
    c_int, // transb
    c_int, // m
    c_int, // n
    c_int, // k
    ?*const anyopaque, // alpha
    ?*const anyopaque, // A
    c_int, // Atype
    c_int, // lda
    c_longlong, // strideA
    ?*const anyopaque, // B
    c_int, // Btype
    c_int, // ldb
    c_longlong, // strideB
    ?*const anyopaque, // beta
    ?*anyopaque, // C
    c_int, // Ctype
    c_int, // ldc
    c_longlong, // strideC
    c_int, // batchCount
    c_int, // computeType
    c_int, // algo
) callconv(.c) c_int;

const CublasApi = struct {
    cublas_create: CublasCreateFn,
    cublas_destroy: CublasDestroyFn,
    cublas_set_math_mode: CublasSetMathModeFn,
    cublas_set_stream: CublasSetStreamFn,
    cublas_sgemm: CublasSgemmFn,
    cublas_gemm_ex: CublasGemmExFn,
    cublas_gemm_strided_batched_ex: CublasGemmStridedBatchedExFn,
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

    /// FP8 E4M3 tensor core path (W8A8):
    /// C_f32 = alpha * A_fp8 @ B_fp8^T
    /// A_fp8: [rows x in_dim] row-major E4M3 (1 byte each)
    /// B_fp8: [out_dim x in_dim] row-major E4M3 (1 byte each)
    /// C_f32: [rows x out_dim] row-major f32
    pub fn matmulFp8Fp8F32(
        self: *Blas,
        device: *device_mod.Device,
        input_fp8: *const device_mod.Buffer,
        rows: usize,
        in_dim: usize,
        weight_fp8: *const device_mod.Buffer,
        out_dim: usize,
        out_f32: *device_mod.Buffer,
        alpha: f32,
    ) !void {
        if (self.handle == null) return error.CublasHandleInvalid;
        if (!dimsFitCublas(rows, out_dim, in_dim)) return error.InvalidArgument;
        if (input_fp8.size < rows * in_dim) return error.InvalidArgument;
        if (weight_fp8.size < out_dim * in_dim) return error.InvalidArgument;
        if (out_f32.size < rows * out_dim * @sizeOf(f32)) return error.InvalidArgument;

        try device.makeCurrent();
        if (self.api.cublas_set_stream(self.handle, device.getLaunchStream()) != cublas_status_success) {
            return error.CublasStreamSetFailed;
        }

        const beta: f32 = 0.0;
        const input_dev: ?*const anyopaque = @ptrFromInt(input_fp8.pointer);
        const weight_dev: ?*const anyopaque = @ptrFromInt(weight_fp8.pointer);
        const out_dev: ?*anyopaque = @ptrFromInt(out_f32.pointer);

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
            cuda_r_8f_e4m3,
            @intCast(in_dim),
            input_dev,
            cuda_r_8f_e4m3,
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

    /// Strided batched mixed-precision GEMM for attention.
    /// C[i] = alpha * op(A[i]) @ op(B[i]) + beta * C[i], i in [0, batch_count).
    /// A is u16 (f16/bf16), B and C are f32.
    /// Column-major convention: op(A) is [m x k], op(B) is [k x n], C is [m x n].
    /// Strides are in elements of the respective type.
    /// Non-batched GEMM with explicit leading dimensions, both inputs u16.
    /// C = alpha * op(A) @ B + beta * C.
    /// A and B are u16 (f16), C is f32.  Leverages tensor cores.
    /// Column-major: op(A) is [m x k], B is [k x n], C is [m x n].
    pub fn gemmU16(
        self: *Blas,
        device: *device_mod.Device,
        transa: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a_ptr: usize,
        lda: usize,
        b_ptr: usize,
        ldb: usize,
        beta: f32,
        c_ptr: usize,
        ldc: usize,
    ) !void {
        if (self.handle == null) return error.CublasHandleInvalid;
        if (!dimsFitCublas(m, n, k)) return error.InvalidArgument;

        try device.makeCurrent();
        if (self.api.cublas_set_stream(self.handle, device.getLaunchStream()) != cublas_status_success) {
            return error.CublasStreamSetFailed;
        }

        const op_a: c_int = if (transa) cublas_op_t else cublas_op_n;

        const status = self.api.cublas_gemm_ex(
            self.handle,
            op_a,
            cublas_op_n,
            @intCast(m),
            @intCast(n),
            @intCast(k),
            @ptrCast(&alpha),
            @ptrFromInt(a_ptr),
            cuda_r_16f,
            @intCast(lda),
            @ptrFromInt(b_ptr),
            cuda_r_16f,
            @intCast(ldb),
            @ptrCast(&beta),
            @ptrFromInt(c_ptr),
            cuda_r_32f,
            @intCast(ldc),
            cublas_compute_32f,
            cublas_gemm_default,
        );
        if (status == cublas_status_alloc_failed) return error.OutOfMemory;
        if (status != cublas_status_success) return error.CublasMatmulFailed;
    }
};

// =============================================================================
// cuBLASLt for block-scaled MXFP8 GEMM (Blackwell tensor cores)
// =============================================================================

// cuBLASLt attribute IDs
const cublaslt_matmul_desc_compute_type: i32 = 0;
const cublaslt_matmul_desc_transa: i32 = 3;
const cublaslt_matmul_desc_transb: i32 = 4;
const cublaslt_matmul_desc_a_scale_pointer: i32 = 17;
const cublaslt_matmul_desc_b_scale_pointer: i32 = 18;
const cublaslt_matmul_desc_a_scale_mode: i32 = 31;
const cublaslt_matmul_desc_b_scale_mode: i32 = 32;

// Scale modes
const cublaslt_scale_vec16_ue4m3: i32 = 1;
const cublaslt_scale_vec32_ue8m0: i32 = 2;

// Preference attributes
const cublaslt_pref_max_workspace_bytes: i32 = 1;

// cuBLASLt function pointer types
const CublasLtCreateFn = *const fn (*?*anyopaque) callconv(.c) c_int;
const CublasLtDestroyFn = *const fn (?*anyopaque) callconv(.c) c_int;
const CublasLtMatmulDescCreateFn = *const fn (*?*anyopaque, c_int, c_int) callconv(.c) c_int;
const CublasLtMatmulDescDestroyFn = *const fn (?*anyopaque) callconv(.c) c_int;
const CublasLtMatmulDescSetAttributeFn = *const fn (?*anyopaque, c_int, ?*const anyopaque, usize) callconv(.c) c_int;
const CublasLtMatrixLayoutCreateFn = *const fn (*?*anyopaque, c_int, u64, u64, u64) callconv(.c) c_int;
const CublasLtMatrixLayoutDestroyFn = *const fn (?*anyopaque) callconv(.c) c_int;
const CublasLtMatmulPreferenceCreateFn = *const fn (*?*anyopaque) callconv(.c) c_int;
const CublasLtMatmulPreferenceDestroyFn = *const fn (?*anyopaque) callconv(.c) c_int;
const CublasLtMatmulPreferenceSetAttributeFn = *const fn (?*anyopaque, c_int, ?*const anyopaque, usize) callconv(.c) c_int;
// cublasLtMatmulAlgoGetHeuristic returns status, last param is *int returnedResults
const CublasLtMatmulAlgoGetHeuristicFn = *const fn (?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque, c_int, ?*anyopaque, *c_int) callconv(.c) c_int;
// cublasLtMatmul: handle, desc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSize, stream
const CublasLtMatmulFn = *const fn (?*anyopaque, ?*anyopaque, ?*const anyopaque, ?*const anyopaque, ?*anyopaque, ?*const anyopaque, ?*anyopaque, ?*const anyopaque, ?*const anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque, ?*const anyopaque, ?*anyopaque, usize, ?*anyopaque) callconv(.c) c_int;

const CublasLtApi = struct {
    create: CublasLtCreateFn,
    destroy: CublasLtDestroyFn,
    matmul_desc_create: CublasLtMatmulDescCreateFn,
    matmul_desc_destroy: CublasLtMatmulDescDestroyFn,
    matmul_desc_set_attribute: CublasLtMatmulDescSetAttributeFn,
    matrix_layout_create: CublasLtMatrixLayoutCreateFn,
    matrix_layout_destroy: CublasLtMatrixLayoutDestroyFn,
    matmul_preference_create: CublasLtMatmulPreferenceCreateFn,
    matmul_preference_destroy: CublasLtMatmulPreferenceDestroyFn,
    matmul_preference_set_attribute: CublasLtMatmulPreferenceSetAttributeFn,
    matmul_algo_get_heuristic: CublasLtMatmulAlgoGetHeuristicFn,
    matmul: CublasLtMatmulFn,
};

/// cuBLASLt handle for block-scaled FP8 GEMM.
/// Loaded separately from cuBLAS since it requires a different shared library.
pub const BlasLt = struct {
    lib: std.DynLib,
    api: CublasLtApi,
    handle: ?*anyopaque,
    workspace: ?device_mod.Buffer,
    workspace_size: usize,
    /// Cached matmul plans keyed by (M, N, K). Avoids expensive per-call
    /// descriptor creation and heuristic search (~168 calls per decode step).
    cached_plans: [max_cached_plans]CachedPlan = [_]CachedPlan{.{}} ** max_cached_plans,
    n_cached: usize = 0,

    const lt_workspace_size: usize = 32 * 1024 * 1024; // 32MB (matches NVIDIA sample)
    const max_cached_plans = 64;

    const PlanKind = enum(u8) {
        mxfp8,
        nvfp4,
    };

    const CachedPlan = struct {
        kind: PlanKind = .mxfp8,
        m: usize = 0,
        n: usize = 0,
        k: usize = 0,
        matmul_desc: ?*anyopaque = null,
        a_layout: ?*anyopaque = null,
        b_layout: ?*anyopaque = null,
        c_layout: ?*anyopaque = null,
        d_layout: ?*anyopaque = null,
        heuristic_result: [128]u64 = [_]u64{0} ** 128,
    };

    pub fn init(device: *device_mod.Device) !BlasLt {
        try device.makeCurrent();

        var lib = try openCublasLtLibrary();
        errdefer lib.close();

        const api = try loadCublasLtApi(&lib);

        var handle: ?*anyopaque = null;
        const status = api.create(&handle);
        if (status != cublas_status_success or handle == null) {
            return error.CublasLtCreateFailed;
        }
        errdefer _ = api.destroy(handle);

        // Allocate workspace
        var workspace = try device.allocBuffer(lt_workspace_size);
        errdefer workspace.deinit(device);

        return .{
            .lib = lib,
            .api = api,
            .handle = handle,
            .workspace = workspace,
            .workspace_size = lt_workspace_size,
        };
    }

    pub fn deinit(self: *BlasLt, device: *device_mod.Device) void {
        device.makeCurrent() catch {};
        // Destroy cached plans before handle
        for (self.cached_plans[0..self.n_cached]) |*plan| {
            if (plan.d_layout) |l| _ = self.api.matrix_layout_destroy(l);
            if (plan.c_layout) |l| _ = self.api.matrix_layout_destroy(l);
            if (plan.b_layout) |l| _ = self.api.matrix_layout_destroy(l);
            if (plan.a_layout) |l| _ = self.api.matrix_layout_destroy(l);
            if (plan.matmul_desc) |d| _ = self.api.matmul_desc_destroy(d);
        }
        self.n_cached = 0;
        if (self.workspace) |*ws| {
            ws.deinit(device);
            self.workspace = null;
        }
        if (self.handle) |h| {
            _ = self.api.destroy(h);
            self.handle = null;
        }
        // Note: intentionally skip self.lib.close(). The NVIDIA cuBLASLt
        // library registers atexit handlers that can crash if dlclosed before
        // process exit. Leaking the handle is safe — the OS reclaims it.
    }

    /// Look up or create a cached plan for the given (M, N, K) dimensions.
    /// Scale pointers are needed for the heuristic search on cache miss.
    fn getOrCreatePlan(
        self: *BlasLt,
        kind: PlanKind,
        M: usize,
        N: usize,
        K: usize,
        a_scale_ptr: ?*const anyopaque,
        b_scale_ptr: ?*const anyopaque,
    ) !*CachedPlan {
        const h = self.handle orelse return error.CublasLtHandleInvalid;

        // Look up existing plan
        for (self.cached_plans[0..self.n_cached]) |*plan| {
            if (plan.kind == kind and plan.m == M and plan.n == N and plan.k == K) return plan;
        }

        // Cache full — evict oldest entry (index 0)
        if (self.n_cached >= max_cached_plans) {
            const evict = &self.cached_plans[0];
            if (evict.d_layout) |l| _ = self.api.matrix_layout_destroy(l);
            if (evict.c_layout) |l| _ = self.api.matrix_layout_destroy(l);
            if (evict.b_layout) |l| _ = self.api.matrix_layout_destroy(l);
            if (evict.a_layout) |l| _ = self.api.matrix_layout_destroy(l);
            if (evict.matmul_desc) |d| _ = self.api.matmul_desc_destroy(d);
            // Shift remaining entries down
            const remaining = self.n_cached - 1;
            for (0..remaining) |i| {
                self.cached_plans[i] = self.cached_plans[i + 1];
            }
            self.n_cached = remaining;
        }

        // Create new plan
        var plan = CachedPlan{ .kind = kind, .m = M, .n = N, .k = K };
        errdefer {
            if (plan.d_layout) |l| _ = self.api.matrix_layout_destroy(l);
            if (plan.c_layout) |l| _ = self.api.matrix_layout_destroy(l);
            if (plan.b_layout) |l| _ = self.api.matrix_layout_destroy(l);
            if (plan.a_layout) |l| _ = self.api.matrix_layout_destroy(l);
            if (plan.matmul_desc) |d| _ = self.api.matmul_desc_destroy(d);
        }

        // Create matmul descriptor: COMPUTE_32F, output CUDA_R_32F
        if (self.api.matmul_desc_create(&plan.matmul_desc, cublas_compute_32f, cuda_r_32f) != cublas_status_success) {
            return error.CublasLtDescCreateFailed;
        }

        // Set transpose: transa=T (weight), transb=N (input)
        var op_t: c_int = cublas_op_t;
        var op_n: c_int = cublas_op_n;
        if (self.api.matmul_desc_set_attribute(plan.matmul_desc, cublaslt_matmul_desc_transa, @ptrCast(&op_t), @sizeOf(c_int)) != cublas_status_success)
            return error.CublasLtSetAttributeFailed;
        if (self.api.matmul_desc_set_attribute(plan.matmul_desc, cublaslt_matmul_desc_transb, @ptrCast(&op_n), @sizeOf(c_int)) != cublas_status_success)
            return error.CublasLtSetAttributeFailed;

        const scale_mode: i32 = switch (kind) {
            .mxfp8 => cublaslt_scale_vec32_ue8m0,
            .nvfp4 => cublaslt_scale_vec16_ue4m3,
        };
        if (self.api.matmul_desc_set_attribute(plan.matmul_desc, cublaslt_matmul_desc_a_scale_mode, @ptrCast(&scale_mode), @sizeOf(i32)) != cublas_status_success)
            return error.CublasLtSetAttributeFailed;
        if (self.api.matmul_desc_set_attribute(plan.matmul_desc, cublaslt_matmul_desc_b_scale_mode, @ptrCast(&scale_mode), @sizeOf(i32)) != cublas_status_success)
            return error.CublasLtSetAttributeFailed;

        // Set scale pointers (required for heuristic search)
        if (self.api.matmul_desc_set_attribute(plan.matmul_desc, cublaslt_matmul_desc_a_scale_pointer, @ptrCast(&a_scale_ptr), @sizeOf(?*const anyopaque)) != cublas_status_success)
            return error.CublasLtSetAttributeFailed;
        if (self.api.matmul_desc_set_attribute(plan.matmul_desc, cublaslt_matmul_desc_b_scale_pointer, @ptrCast(&b_scale_ptr), @sizeOf(?*const anyopaque)) != cublas_status_success)
            return error.CublasLtSetAttributeFailed;

        // Create matrix layouts (column-major perspective)
        // A (weight): K×N col-major (with transa=T, logical N×K = out_dim × in_dim)
        // B (input): K×M col-major (with transb=N, logical K×M = in_dim × rows)
        // C/D (output): N×M col-major = out_dim × rows
        const ab_dtype: c_int = switch (kind) {
            .mxfp8 => cuda_r_8f_e4m3,
            .nvfp4 => cuda_r_4f_e2m1,
        };
        if (self.api.matrix_layout_create(&plan.a_layout, ab_dtype, K, N, K) != cublas_status_success)
            return error.CublasLtLayoutCreateFailed;
        if (self.api.matrix_layout_create(&plan.b_layout, ab_dtype, K, M, K) != cublas_status_success)
            return error.CublasLtLayoutCreateFailed;
        if (self.api.matrix_layout_create(&plan.c_layout, cuda_r_32f, N, M, N) != cublas_status_success)
            return error.CublasLtLayoutCreateFailed;
        if (self.api.matrix_layout_create(&plan.d_layout, cuda_r_32f, N, M, N) != cublas_status_success)
            return error.CublasLtLayoutCreateFailed;

        // Run heuristic search (the expensive part — only done once per unique M,N,K)
        var preference: ?*anyopaque = null;
        if (self.api.matmul_preference_create(&preference) != cublas_status_success)
            return error.CublasLtPreferenceCreateFailed;
        defer _ = self.api.matmul_preference_destroy(preference);

        var ws_size: usize = self.workspace_size;
        if (self.api.matmul_preference_set_attribute(preference, cublaslt_pref_max_workspace_bytes, @ptrCast(&ws_size), @sizeOf(usize)) != cublas_status_success)
            return error.CublasLtSetAttributeFailed;

        var returned_results: c_int = 0;
        const heur_status = self.api.matmul_algo_get_heuristic(
            h,
            plan.matmul_desc,
            plan.a_layout,
            plan.b_layout,
            plan.c_layout,
            plan.d_layout,
            preference,
            1,
            @ptrCast(&plan.heuristic_result),
            &returned_results,
        );
        if (heur_status != cublas_status_success or returned_results == 0)
            return error.CublasLtNoAlgorithm;

        // Store in cache
        const idx = self.n_cached;
        self.cached_plans[idx] = plan;
        self.n_cached = idx + 1;
        return &self.cached_plans[idx];
    }

    /// Block-scaled MXFP8 GEMM: C_f32 = A_e4m3 @ B_e4m3^T with UE8M0 block-32 scales.
    ///
    /// Weight A: [out_dim × in_dim] row-major E4M3, scales: [out_dim × in_dim/32] UE8M0
    /// Input B: [rows × in_dim] row-major E4M3, scales: [rows × in_dim/32] UE8M0
    /// Output C: [rows × out_dim] row-major F32
    pub fn matmulMxfp8(
        self: *BlasLt,
        device: *device_mod.Device,
        weight_e4m3: *const device_mod.Buffer,
        weight_scales_e8m0: *const device_mod.Buffer,
        input_e4m3: *const device_mod.Buffer,
        input_scales_e8m0: *const device_mod.Buffer,
        out_f32: *device_mod.Buffer,
        rows: usize,
        out_dim: usize,
        in_dim: usize,
    ) !void {
        if (!dimsFitCublas(rows, out_dim, in_dim)) return error.InvalidArgument;

        try device.makeCurrent();
        const stream = device.getLaunchStream();

        const a_scale_ptr: ?*const anyopaque = @ptrFromInt(weight_scales_e8m0.pointer);
        const b_scale_ptr: ?*const anyopaque = @ptrFromInt(input_scales_e8m0.pointer);

        const plan = try self.getOrCreatePlan(.mxfp8, rows, out_dim, in_dim, a_scale_ptr, b_scale_ptr);

        // Update per-call scale pointers on the cached descriptor
        if (self.api.matmul_desc_set_attribute(plan.matmul_desc, cublaslt_matmul_desc_a_scale_pointer, @ptrCast(&a_scale_ptr), @sizeOf(?*const anyopaque)) != cublas_status_success)
            return error.CublasLtSetAttributeFailed;
        if (self.api.matmul_desc_set_attribute(plan.matmul_desc, cublaslt_matmul_desc_b_scale_pointer, @ptrCast(&b_scale_ptr), @sizeOf(?*const anyopaque)) != cublas_status_success)
            return error.CublasLtSetAttributeFailed;

        // Execute matmul with cached plan
        const alpha: f32 = 1.0;
        const beta: f32 = 0.0;
        const weight_dev: ?*const anyopaque = @ptrFromInt(weight_e4m3.pointer);
        const input_dev: ?*const anyopaque = @ptrFromInt(input_e4m3.pointer);
        const out_dev: ?*anyopaque = @ptrFromInt(out_f32.pointer);
        const ws_ptr: ?*anyopaque = if (self.workspace) |ws| @ptrFromInt(ws.pointer) else null;
        const algo_ptr: ?*const anyopaque = @ptrCast(&plan.heuristic_result);

        const matmul_status = self.api.matmul(
            self.handle,
            plan.matmul_desc,
            @ptrCast(&alpha),
            weight_dev,
            plan.a_layout,
            input_dev,
            plan.b_layout,
            @ptrCast(&beta),
            out_dev,
            plan.c_layout,
            out_dev,
            plan.d_layout,
            algo_ptr,
            ws_ptr,
            self.workspace_size,
            stream,
        );
        if (matmul_status != cublas_status_success) {
            return error.CublasLtMatmulFailed;
        }
    }

    /// Block-scaled NVFP4 GEMM: C_f32 = A_fp4 @ B_fp4^T with UE4M3 block-16 scales.
    ///
    /// Weight A: [out_dim × in_dim] row-major FP4 E2M1, scales: interleaved UE4M3
    /// Input B: [rows × in_dim] row-major FP4 E2M1, scales: interleaved UE4M3
    /// Output C: [rows × out_dim] row-major F32
    pub fn matmulNvfp4(
        self: *BlasLt,
        device: *device_mod.Device,
        weight_fp4: *const device_mod.Buffer,
        weight_scales_ue4m3: *const device_mod.Buffer,
        input_fp4: *const device_mod.Buffer,
        input_scales_ue4m3: *const device_mod.Buffer,
        out_f32: *device_mod.Buffer,
        rows: usize,
        out_dim: usize,
        in_dim: usize,
        alpha_scale: f32,
    ) !void {
        if (!dimsFitCublas(rows, out_dim, in_dim)) return error.InvalidArgument;

        try device.makeCurrent();
        const stream = device.getLaunchStream();

        const a_scale_ptr: ?*const anyopaque = @ptrFromInt(weight_scales_ue4m3.pointer);
        const b_scale_ptr: ?*const anyopaque = @ptrFromInt(input_scales_ue4m3.pointer);
        const plan = try self.getOrCreatePlan(.nvfp4, rows, out_dim, in_dim, a_scale_ptr, b_scale_ptr);

        if (self.api.matmul_desc_set_attribute(plan.matmul_desc, cublaslt_matmul_desc_a_scale_pointer, @ptrCast(&a_scale_ptr), @sizeOf(?*const anyopaque)) != cublas_status_success)
            return error.CublasLtSetAttributeFailed;
        if (self.api.matmul_desc_set_attribute(plan.matmul_desc, cublaslt_matmul_desc_b_scale_pointer, @ptrCast(&b_scale_ptr), @sizeOf(?*const anyopaque)) != cublas_status_success)
            return error.CublasLtSetAttributeFailed;

        const alpha: f32 = alpha_scale;
        const beta: f32 = 0.0;
        const weight_dev: ?*const anyopaque = @ptrFromInt(weight_fp4.pointer);
        const input_dev: ?*const anyopaque = @ptrFromInt(input_fp4.pointer);
        const out_dev: ?*anyopaque = @ptrFromInt(out_f32.pointer);
        const ws_ptr: ?*anyopaque = if (self.workspace) |ws| @ptrFromInt(ws.pointer) else null;
        const algo_ptr: ?*const anyopaque = @ptrCast(&plan.heuristic_result);

        const matmul_status = self.api.matmul(
            self.handle,
            plan.matmul_desc,
            @ptrCast(&alpha),
            weight_dev,
            plan.a_layout,
            input_dev,
            plan.b_layout,
            @ptrCast(&beta),
            out_dev,
            plan.c_layout,
            out_dev,
            plan.d_layout,
            algo_ptr,
            ws_ptr,
            self.workspace_size,
            stream,
        );
        if (matmul_status != cublas_status_success) {
            return error.CublasLtMatmulFailed;
        }
    }
};

fn openCublasLtLibrary() !std.DynLib {
    const names: []const []const u8 = switch (builtin.os.tag) {
        .linux => &.{ "libcublasLt.so.12", "libcublasLt.so.13", "libcublasLt.so" },
        .windows => &.{ "cublasLt64_12.dll", "cublasLt64_11.dll" },
        else => &.{},
    };
    for (names) |name| {
        if (std.DynLib.open(name)) |lib| return lib else |_| {}
    }
    return error.CublasLtUnavailable;
}

fn loadCublasLtApi(lib: *std.DynLib) !CublasLtApi {
    return .{
        .create = try lookupRequired(CublasLtCreateFn, lib, "cublasLtCreate"),
        .destroy = try lookupRequired(CublasLtDestroyFn, lib, "cublasLtDestroy"),
        .matmul_desc_create = try lookupRequired(CublasLtMatmulDescCreateFn, lib, "cublasLtMatmulDescCreate"),
        .matmul_desc_destroy = try lookupRequired(CublasLtMatmulDescDestroyFn, lib, "cublasLtMatmulDescDestroy"),
        .matmul_desc_set_attribute = try lookupRequired(CublasLtMatmulDescSetAttributeFn, lib, "cublasLtMatmulDescSetAttribute"),
        .matrix_layout_create = try lookupRequired(CublasLtMatrixLayoutCreateFn, lib, "cublasLtMatrixLayoutCreate"),
        .matrix_layout_destroy = try lookupRequired(CublasLtMatrixLayoutDestroyFn, lib, "cublasLtMatrixLayoutDestroy"),
        .matmul_preference_create = try lookupRequired(CublasLtMatmulPreferenceCreateFn, lib, "cublasLtMatmulPreferenceCreate"),
        .matmul_preference_destroy = try lookupRequired(CublasLtMatmulPreferenceDestroyFn, lib, "cublasLtMatmulPreferenceDestroy"),
        .matmul_preference_set_attribute = try lookupRequired(CublasLtMatmulPreferenceSetAttributeFn, lib, "cublasLtMatmulPreferenceSetAttribute"),
        .matmul_algo_get_heuristic = try lookupRequired(CublasLtMatmulAlgoGetHeuristicFn, lib, "cublasLtMatmulAlgoGetHeuristic"),
        .matmul = try lookupRequired(CublasLtMatmulFn, lib, "cublasLtMatmul"),
    };
}

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
        .cublas_gemm_strided_batched_ex = try lookupRequired(CublasGemmStridedBatchedExFn, lib, "cublasGemmStridedBatchedEx"),
    };
}

test "dimsFitCublas accepts small dimensions" {
    try std.testing.expect(dimsFitCublas(4, 8, 16));
}

test "dimsFitCublas rejects c_int overflow dimensions" {
    const overflow = @as(usize, @intCast(std.math.maxInt(c_int))) + 1;
    try std.testing.expect(!dimsFitCublas(overflow, 1, 1));
}
