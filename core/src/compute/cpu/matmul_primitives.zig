//! Matrix multiplication with quantization support.
//!
//! Implements linear (matmul + bias) for f32, bf16, and grouped affine U4/U8
//! with SIMD acceleration.

const std = @import("std");
const build_options = @import("build_options");
const parallel = @import("../parallel.zig");
const tensor_mod = @import("../../tensor.zig");
const dtype_mod = @import("../../dtype.zig");
const simd = @import("../simd/root.zig");
const grouped_affine_quant = @import("../quant/grouped_affine_quant.zig");
const prefill = @import("matmul_prefill.zig");
const log = @import("../../log.zig");

// Re-export types
pub const Tensor = tensor_mod.Tensor;
pub const DType = dtype_mod.DType;

const fp16ToF32 = dtype_mod.fp16ToF32;
const f32ToFp16 = dtype_mod.f32ToFp16;
const bf16ToF32 = dtype_mod.bf16ToF32;
const fp16VecToF32Bits = dtype_mod.fp16x8ToF32Bits;
const gaffineScaleBiasToF32 = grouped_affine_quant.scaleBiasToF32;
const extractNibbles = grouped_affine_quant.extractNibbles;
const extract32NibblesToFloat = grouped_affine_quant.extract32NibblesToFloat;
const extractBytes = grouped_affine_quant.extractBytes;

const debug_matmul = build_options.debug_matmul;

// =============================================================================
// Matmul Scratch (placeholder for future grouped affine optimizations)
// =============================================================================
pub const MatmulScratch = struct {
    const max_weight_rows: usize = 200_000;

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !MatmulScratch {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MatmulScratch) void {
        self.* = undefined;
    }
};

// =============================================================================
// Kernel Tuning Constants
// =============================================================================
// These control tiling and parallelization strategies. Values are tuned for
// modern x86-64 CPUs with AVX2/AVX-512.

/// Number of output columns per tile in the decode (m=1) path.
/// Smaller tiles mean better load balancing but more overhead.
const TILE_COLS: usize = 4;

/// Column tile size for small-batch prefill path (quantized kernels).
/// Larger tiles improve cache locality at the cost of load balancing.
/// Quantized kernels benefit from larger tiles to amortize unpacking overhead.
const COL_TILE_SIZE: usize = 128;

/// Column tile size for BF16/F16 prefill path.
/// BF16/F16 are memory-bandwidth bound (not compute bound like quantized).
/// Smaller tiles keep the active weight block in L1 cache (32KB typical).
/// With k=1024 (common hidden dim), 32 columns * 1024 * 2 bytes = 64KB fits in L1.
const COL_TILE_SIZE_FP16: usize = 32;

/// Batch size threshold: above this, parallelize over rows only.
/// Below this, use tiled row+column parallelization for better load balance.
const TILE_THRESHOLD: usize = 64;

// =============================================================================
// Kernel Limits
// =============================================================================

/// Maximum number of quantization groups supported per matmul column.
/// For grouped-affine quantization: max_groups = max_k / min_group_size = 32768 / 32 = 1024.
/// Supports models up to ~130B parameters (k=32768 with group_size=32).
pub const MAX_GROUPS: usize = 1024;

/// Function pointer type for matmul kernels. Use `matmulKernel` to get the
/// appropriate kernel for a weight tensor's dtype at load time.
pub const MatmulFn = *const fn (*const Tensor, *const Tensor, *Tensor, *MatmulScratch) void;

/// A dispatched kernel: the function pointer and its real source name.
pub const DispatchedKernel = struct {
    func: MatmulFn,
    name: []const u8,
};

/// Returns the appropriate matmul kernel for a weight tensor's dtype.
/// Call this once at model load time and store the result.
/// Returns both the function pointer and the real function name for tracing.
pub fn matmulKernel(weight_dtype: DType) !DispatchedKernel {
    return switch (weight_dtype) {
        .bf16 => .{ .func = matmulBF16, .name = "matmulBF16" },
        .f16 => .{ .func = matmulF16, .name = "matmulF16" },
        .grouped_affine_u4 => .{ .func = matmulGaffineU4, .name = "matmulGaffineU4" },
        .grouped_affine_u8 => .{ .func = matmulGaffineU8, .name = "matmulGaffineU8" },
        .f32 => .{ .func = matmulF32, .name = "matmulF32" },
        else => error.UnsupportedDType,
    };
}

/// Dispatches to the appropriate matmul kernel based on weight dtype.
/// For hot paths, use `matmulKernel` to get a function pointer at load time instead.
pub fn matmulAuto(a: *const Tensor, b: *const Tensor, out: *Tensor, scratch: *MatmulScratch) !void {
    const dispatched = try matmulKernel(b.dtype);
    dispatched.func(a, b, out, scratch);
}

pub fn matmulF32(a: *const Tensor, b: *const Tensor, out: *Tensor, scratch: *MatmulScratch) void {
    _ = scratch;
    std.debug.assert(a.dtype == .f32 and b.dtype == .f32 and out.dtype == .f32);
    std.debug.assert(a.n_dims == 2 and b.n_dims == 2 and out.n_dims == 2);
    const m_rows: usize = @intCast(a.shape[0]);
    const k_dim: usize = @intCast(a.shape[1]);
    std.debug.assert(b.shape[0] == a.shape[1]);
    const n_cols: usize = @intCast(b.shape[1]);
    std.debug.assert(out.shape[0] == a.shape[0] and out.shape[1] == b.shape[1]);

    const a_data = a.asSlice(f32);
    const b_data = b.asSlice(f32);
    const c_data = out.asSlice(f32);

    const MatmulF32Ctx = struct {
        a: []const f32,
        b: []const f32,
        c: []f32,
        m_rows: usize,
        n_cols: usize,
        k_dim: usize,
    };
    var context = MatmulF32Ctx{ .a = a_data, .b = b_data, .c = c_data, .m_rows = m_rows, .n_cols = n_cols, .k_dim = k_dim };

    const row_tiles_task = struct {
        fn runRowTiles(start: usize, end: usize, task_ctx: *MatmulF32Ctx) void {
            const k_len = task_ctx.k_dim;
            const n_len = task_ctx.n_cols;

            const vec_len = simd.f32_vec_len;
            const accum_chunks = 4;
            for (start..end) |row_idx| {
                const a_row = task_ctx.a[row_idx * k_len ..][0..k_len];
                const out_row = task_ctx.c[row_idx * n_len ..][0..n_len];

                for (0..n_len) |col_idx| {
                    var acc: [accum_chunks]@Vector(vec_len, f32) = .{@as(@Vector(vec_len, f32), @splat(0))} ** accum_chunks;
                    var k_idx: usize = 0;

                    while (k_idx + accum_chunks * vec_len - 1 < k_len) : (k_idx += accum_chunks * vec_len) {
                        inline for (0..accum_chunks) |u| {
                            const off = k_idx + u * vec_len;
                            const a_vec: @Vector(vec_len, f32) = a_row[off..][0..vec_len].*;
                            var b_vec: @Vector(vec_len, f32) = undefined;
                            inline for (0..vec_len) |e| b_vec[e] = task_ctx.b[(off + e) * n_len + col_idx];
                            acc[u] = @mulAdd(@Vector(vec_len, f32), a_vec, b_vec, acc[u]);
                        }
                    }

                    var total: @Vector(vec_len, f32) = @splat(0);
                    inline for (0..accum_chunks) |u| total += acc[u];
                    var sum = @reduce(.Add, total);

                    while (k_idx < k_len) : (k_idx += 1) {
                        sum += a_row[k_idx] * task_ctx.b[k_idx * n_len + col_idx];
                    }
                    out_row[col_idx] = sum;
                }
            }
        }
    }.runRowTiles;

    // Tiled parallelization for better load balancing
    if (m_rows >= TILE_THRESHOLD) {
        parallel.global().parallelFor(m_rows, row_tiles_task, &context);
    } else if (m_rows == 1) {
        // Single row: parallelize over columns
        const decode_task = struct {
            fn runDecodeCols(start: usize, end: usize, task_ctx: *MatmulF32Ctx) void {
                const k_len = task_ctx.k_dim;
                const n_len = task_ctx.n_cols;
                const a_row = task_ctx.a[0..k_len];

                const vec_len = simd.f32_vec_len;
                for (start..end) |col_idx| {
                    var acc: @Vector(vec_len, f32) = @splat(0);
                    var k_idx: usize = 0;
                    while (k_idx + vec_len - 1 < k_len) : (k_idx += vec_len) {
                        const a_vec: @Vector(vec_len, f32) = a_row[k_idx..][0..vec_len].*;
                        var b_vec: @Vector(vec_len, f32) = undefined;
                        inline for (0..vec_len) |e| b_vec[e] = task_ctx.b[(k_idx + e) * n_len + col_idx];
                        acc = @mulAdd(@Vector(vec_len, f32), a_vec, b_vec, acc);
                    }
                    var sum = @reduce(.Add, acc);
                    while (k_idx < k_len) : (k_idx += 1) {
                        sum += a_row[k_idx] * task_ctx.b[k_idx * n_len + col_idx];
                    }
                    task_ctx.c[col_idx] = sum;
                }
            }
        }.runDecodeCols;
        parallel.global().parallelFor(n_cols, decode_task, &context);
    } else {
        // Small batch: tile across rows AND columns
        const tiles_per_row = (n_cols + COL_TILE_SIZE - 1) / COL_TILE_SIZE;
        const total_tiles = m_rows * tiles_per_row;

        const MatmulF32TileCtx = struct {
            a: []const f32,
            b: []const f32,
            c: []f32,
            m_rows: usize,
            n_cols: usize,
            k_dim: usize,
            tiles_per_row: usize,
        };
        var tiled_ctx = MatmulF32TileCtx{
            .a = a_data,
            .b = b_data,
            .c = c_data,
            .m_rows = m_rows,
            .n_cols = n_cols,
            .k_dim = k_dim,
            .tiles_per_row = tiles_per_row,
        };

        const tiled_task = struct {
            fn runRowColTiles(start: usize, end: usize, task_ctx: *MatmulF32TileCtx) void {
                const k_len = task_ctx.k_dim;
                const n_len = task_ctx.n_cols;
                const VEC = simd.f32_vec_len;

                for (start..end) |tile_idx| {
                    const row = tile_idx / task_ctx.tiles_per_row;
                    const col_tile = tile_idx % task_ctx.tiles_per_row;
                    const col_start = col_tile * COL_TILE_SIZE;
                    const col_end = @min(col_start + COL_TILE_SIZE, n_len);

                    const a_row = task_ctx.a[row * k_len ..][0..k_len];
                    const out_row = task_ctx.c[row * n_len ..][0..n_len];

                    for (col_start..col_end) |col_idx| {
                        var acc: @Vector(VEC, f32) = @splat(0);
                        var k_idx: usize = 0;
                        while (k_idx + VEC - 1 < k_len) : (k_idx += VEC) {
                            const a_vec: @Vector(VEC, f32) = a_row[k_idx..][0..VEC].*;
                            var b_vec: @Vector(VEC, f32) = undefined;
                            inline for (0..VEC) |lane| b_vec[lane] = task_ctx.b[(k_idx + lane) * n_len + col_idx];
                            acc = @mulAdd(@Vector(VEC, f32), a_vec, b_vec, acc);
                        }
                        var sum = @reduce(.Add, acc);
                        while (k_idx < k_len) : (k_idx += 1) {
                            sum += a_row[k_idx] * task_ctx.b[k_idx * n_len + col_idx];
                        }
                        out_row[col_idx] = sum;
                    }
                }
            }
        }.runRowColTiles;
        parallel.global().parallelFor(total_tiles, tiled_task, &tiled_ctx);
    }
}

/// BF16 matmul kernel - dedicated kernel for BF16 weights (no runtime dtype branching).
fn matmulBF16(a: *const Tensor, b: *const Tensor, out: *Tensor, scratch: *MatmulScratch) void {
    _ = scratch;
    std.debug.assert(a.dtype == .f32 and b.dtype == .bf16 and out.dtype == .f32);
    std.debug.assert(a.n_dims == 2 and b.n_dims == 2 and out.n_dims == 2);

    // BF16 weights are stored as [out, in] = [n, k] (not transposed)
    const m_rows: usize = @intCast(a.shape[0]);
    const k_dim: usize = @intCast(a.shape[1]);
    const n_cols: usize = @intCast(b.shape[0]);
    std.debug.assert(b.shape[1] == a.shape[1]);
    std.debug.assert(out.shape[0] == a.shape[0] and out.shape[1] == b.shape[0]);

    const a_data = a.asSlice(f32);
    const b_data = b.asSliceUnaligned(u16);
    const c_data = out.asSlice(f32);

    const MatmulBF16Ctx = struct {
        a: []const f32,
        b: []align(1) const u16,
        c: []f32,
        m_rows: usize,
        n_cols: usize,
        k_dim: usize,
    };
    var context = MatmulBF16Ctx{ .a = a_data, .b = b_data, .c = c_data, .m_rows = m_rows, .n_cols = n_cols, .k_dim = k_dim };

    // Process multiple columns per task to amortize activation loads
    // and improve cache locality for memory-bandwidth bound BF16 workloads
    const decode_task = struct {
        fn runDecodeCols(start: usize, end: usize, task_ctx: *MatmulBF16Ctx) void {
            const k_len = task_ctx.k_dim;
            const n_len = task_ctx.n_cols;
            const a_row = task_ctx.a[0..k_len];
            const out_row = task_ctx.c[0..n_len];
            const b_base = task_ctx.b;
            const VEC = simd.f32_vec_len;

            var col_idx = start;

            // Process 4 columns at a time for better ILP and cache efficiency
            while (col_idx + 4 <= end) : (col_idx += 4) {
                const b0 = b_base[col_idx * k_len ..][0..k_len];
                const b1 = b_base[(col_idx + 1) * k_len ..][0..k_len];
                const b2 = b_base[(col_idx + 2) * k_len ..][0..k_len];
                const b3 = b_base[(col_idx + 3) * k_len ..][0..k_len];

                var acc0: @Vector(VEC, f32) = @splat(0);
                var acc1: @Vector(VEC, f32) = @splat(0);
                var acc2: @Vector(VEC, f32) = @splat(0);
                var acc3: @Vector(VEC, f32) = @splat(0);

                var k_idx: usize = 0;
                // Main SIMD loop - process all 4 columns with shared activation load
                while (k_idx + VEC <= k_len) : (k_idx += VEC) {
                    const a_vec: @Vector(VEC, f32) = a_row[k_idx..][0..VEC].*;

                    // Prefetch next cache lines
                    @prefetch(@as([*]const u8, @ptrCast(b0.ptr + k_idx + 64)), .{ .locality = 3 });
                    @prefetch(@as([*]const u8, @ptrCast(b1.ptr + k_idx + 64)), .{ .locality = 3 });
                    @prefetch(@as([*]const u8, @ptrCast(b2.ptr + k_idx + 64)), .{ .locality = 3 });
                    @prefetch(@as([*]const u8, @ptrCast(b3.ptr + k_idx + 64)), .{ .locality = 3 });

                    const w0: @Vector(VEC, f32) = @bitCast(@as(@Vector(VEC, u32), b0[k_idx..][0..VEC].*) << @as(@Vector(VEC, u5), @splat(16)));
                    const w1: @Vector(VEC, f32) = @bitCast(@as(@Vector(VEC, u32), b1[k_idx..][0..VEC].*) << @as(@Vector(VEC, u5), @splat(16)));
                    const w2: @Vector(VEC, f32) = @bitCast(@as(@Vector(VEC, u32), b2[k_idx..][0..VEC].*) << @as(@Vector(VEC, u5), @splat(16)));
                    const w3: @Vector(VEC, f32) = @bitCast(@as(@Vector(VEC, u32), b3[k_idx..][0..VEC].*) << @as(@Vector(VEC, u5), @splat(16)));

                    acc0 = @mulAdd(@Vector(VEC, f32), a_vec, w0, acc0);
                    acc1 = @mulAdd(@Vector(VEC, f32), a_vec, w1, acc1);
                    acc2 = @mulAdd(@Vector(VEC, f32), a_vec, w2, acc2);
                    acc3 = @mulAdd(@Vector(VEC, f32), a_vec, w3, acc3);
                }

                var sum0 = @reduce(.Add, acc0);
                var sum1 = @reduce(.Add, acc1);
                var sum2 = @reduce(.Add, acc2);
                var sum3 = @reduce(.Add, acc3);

                // Scalar tail
                while (k_idx < k_len) : (k_idx += 1) {
                    const a_val = a_row[k_idx];
                    sum0 += a_val * bf16ToF32(b0[k_idx]);
                    sum1 += a_val * bf16ToF32(b1[k_idx]);
                    sum2 += a_val * bf16ToF32(b2[k_idx]);
                    sum3 += a_val * bf16ToF32(b3[k_idx]);
                }

                out_row[col_idx] = sum0;
                out_row[col_idx + 1] = sum1;
                out_row[col_idx + 2] = sum2;
                out_row[col_idx + 3] = sum3;
            }

            // Handle remaining columns (1-3)
            while (col_idx < end) : (col_idx += 1) {
                const b_row = b_base[col_idx * k_len ..][0..k_len];
                var acc: @Vector(VEC, f32) = @splat(0);
                var k_idx: usize = 0;

                while (k_idx + VEC <= k_len) : (k_idx += VEC) {
                    const a_vec: @Vector(VEC, f32) = a_row[k_idx..][0..VEC].*;
                    const b_u16: @Vector(VEC, u16) = b_row[k_idx..][0..VEC].*;
                    const b_vec: @Vector(VEC, f32) = @bitCast(@as(@Vector(VEC, u32), b_u16) << @as(@Vector(VEC, u5), @splat(16)));
                    acc = @mulAdd(@Vector(VEC, f32), a_vec, b_vec, acc);
                }

                var sum = @reduce(.Add, acc);
                while (k_idx < k_len) : (k_idx += 1) {
                    sum += a_row[k_idx] * bf16ToF32(b_row[k_idx]);
                }
                out_row[col_idx] = sum;
            }
        }
    }.runDecodeCols;

    if (m_rows == 1) {
        // Decode: parallelize over output columns.
        parallel.global().parallelFor(n_cols, decode_task, &context);
    } else {
        // Prefill/batch: tile over rows * columns
        // Use smaller tiles for BF16 to keep weights in L1 cache (memory-bound workload)
        const tile_size = COL_TILE_SIZE_FP16;
        const tiles_per_row = (n_cols + tile_size - 1) / tile_size;
        const total_tiles = m_rows * tiles_per_row;

        const tiled_task = struct {
            fn runRowColTiles(start: usize, end: usize, task_ctx: *MatmulBF16Ctx) void {
                const k_len = task_ctx.k_dim;
                const n_len = task_ctx.n_cols;
                const tile_size_local = COL_TILE_SIZE_FP16;
                const tiles_per_row_local = (n_len + tile_size_local - 1) / tile_size_local;
                const VEC = simd.f32_vec_len;
                const b_base = task_ctx.b;

                for (start..end) |tile_idx| {
                    const row = tile_idx / tiles_per_row_local;
                    const col_tile = tile_idx % tiles_per_row_local;
                    const col_start = col_tile * tile_size_local;
                    const col_end = @min(col_start + tile_size_local, n_len);

                    const a_row = task_ctx.a[row * k_len ..][0..k_len];
                    const out_row = task_ctx.c[row * n_len ..][0..n_len];

                    var col_idx = col_start;

                    // Process 4 columns at a time for better ILP
                    while (col_idx + 4 <= col_end) : (col_idx += 4) {
                        const b0 = b_base[col_idx * k_len ..][0..k_len];
                        const b1 = b_base[(col_idx + 1) * k_len ..][0..k_len];
                        const b2 = b_base[(col_idx + 2) * k_len ..][0..k_len];
                        const b3 = b_base[(col_idx + 3) * k_len ..][0..k_len];

                        var acc0: @Vector(VEC, f32) = @splat(0);
                        var acc1: @Vector(VEC, f32) = @splat(0);
                        var acc2: @Vector(VEC, f32) = @splat(0);
                        var acc3: @Vector(VEC, f32) = @splat(0);

                        var k_idx: usize = 0;
                        while (k_idx + VEC <= k_len) : (k_idx += VEC) {
                            const a_vec: @Vector(VEC, f32) = a_row[k_idx..][0..VEC].*;

                            @prefetch(@as([*]const u8, @ptrCast(b0.ptr + k_idx + 64)), .{ .locality = 3 });

                            const w0: @Vector(VEC, f32) = @bitCast(@as(@Vector(VEC, u32), b0[k_idx..][0..VEC].*) << @as(@Vector(VEC, u5), @splat(16)));
                            const w1: @Vector(VEC, f32) = @bitCast(@as(@Vector(VEC, u32), b1[k_idx..][0..VEC].*) << @as(@Vector(VEC, u5), @splat(16)));
                            const w2: @Vector(VEC, f32) = @bitCast(@as(@Vector(VEC, u32), b2[k_idx..][0..VEC].*) << @as(@Vector(VEC, u5), @splat(16)));
                            const w3: @Vector(VEC, f32) = @bitCast(@as(@Vector(VEC, u32), b3[k_idx..][0..VEC].*) << @as(@Vector(VEC, u5), @splat(16)));

                            acc0 = @mulAdd(@Vector(VEC, f32), a_vec, w0, acc0);
                            acc1 = @mulAdd(@Vector(VEC, f32), a_vec, w1, acc1);
                            acc2 = @mulAdd(@Vector(VEC, f32), a_vec, w2, acc2);
                            acc3 = @mulAdd(@Vector(VEC, f32), a_vec, w3, acc3);
                        }

                        var sum0 = @reduce(.Add, acc0);
                        var sum1 = @reduce(.Add, acc1);
                        var sum2 = @reduce(.Add, acc2);
                        var sum3 = @reduce(.Add, acc3);

                        while (k_idx < k_len) : (k_idx += 1) {
                            const a_val = a_row[k_idx];
                            sum0 += a_val * bf16ToF32(b0[k_idx]);
                            sum1 += a_val * bf16ToF32(b1[k_idx]);
                            sum2 += a_val * bf16ToF32(b2[k_idx]);
                            sum3 += a_val * bf16ToF32(b3[k_idx]);
                        }

                        out_row[col_idx] = sum0;
                        out_row[col_idx + 1] = sum1;
                        out_row[col_idx + 2] = sum2;
                        out_row[col_idx + 3] = sum3;
                    }

                    // Handle remaining columns
                    while (col_idx < col_end) : (col_idx += 1) {
                        const b_row = b_base[col_idx * k_len ..][0..k_len];
                        var acc: @Vector(VEC, f32) = @splat(0);
                        var k_idx: usize = 0;

                        while (k_idx + VEC <= k_len) : (k_idx += VEC) {
                            const a_vec: @Vector(VEC, f32) = a_row[k_idx..][0..VEC].*;
                            const b_u16: @Vector(VEC, u16) = b_row[k_idx..][0..VEC].*;
                            const b_vec: @Vector(VEC, f32) = @bitCast(@as(@Vector(VEC, u32), b_u16) << @as(@Vector(VEC, u5), @splat(16)));
                            acc = @mulAdd(@Vector(VEC, f32), a_vec, b_vec, acc);
                        }

                        var sum = @reduce(.Add, acc);
                        while (k_idx < k_len) : (k_idx += 1) {
                            sum += a_row[k_idx] * bf16ToF32(b_row[k_idx]);
                        }
                        out_row[col_idx] = sum;
                    }
                }
            }
        }.runRowColTiles;

        parallel.global().parallelFor(total_tiles, tiled_task, &context);
    }
}

/// F16 matmul kernel - dedicated kernel without runtime dtype branching.
/// Identical structure to matmulBF16 but uses F16 bit manipulation directly.
fn matmulF16(a: *const Tensor, b: *const Tensor, out: *Tensor, scratch: *MatmulScratch) void {
    _ = scratch;
    std.debug.assert(a.dtype == .f32 and b.dtype == .f16 and out.dtype == .f32);
    std.debug.assert(a.n_dims == 2 and b.n_dims == 2 and out.n_dims == 2);

    // F16 weights are stored as [out, in] = [n, k] (not transposed)
    const m_rows: usize = @intCast(a.shape[0]);
    const k_dim: usize = @intCast(a.shape[1]);
    const n_cols: usize = @intCast(b.shape[0]);
    std.debug.assert(b.shape[1] == a.shape[1]);
    std.debug.assert(out.shape[0] == a.shape[0] and out.shape[1] == b.shape[0]);

    const a_data = a.asSlice(f32);
    const b_data = b.asSliceUnaligned(u16);
    const c_data = out.asSlice(f32);

    const MatmulF16Ctx = struct {
        a: []const f32,
        b: []align(1) const u16,
        c: []f32,
        m_rows: usize,
        n_cols: usize,
        k_dim: usize,
    };
    var context = MatmulF16Ctx{ .a = a_data, .b = b_data, .c = c_data, .m_rows = m_rows, .n_cols = n_cols, .k_dim = k_dim };

    const decode_task = struct {
        fn runDecodeCols(start: usize, end: usize, task_ctx: *MatmulF16Ctx) void {
            const k_len = task_ctx.k_dim;
            const a_row = task_ctx.a[0..k_len];
            const out_row = task_ctx.c[0..task_ctx.n_cols];
            const VEC = simd.f32_vec_len;
            const N = 4; // Increased unroll factor for better ILP

            for (start..end) |col_idx| {
                const b_row = task_ctx.b[col_idx * k_len ..][0..k_len];
                var acc: [N]@Vector(VEC, f32) = .{@as(@Vector(VEC, f32), @splat(0))} ** N;
                var k_idx: usize = 0;

                // Main loop with prefetch and 4x unroll
                while (k_idx + N * VEC - 1 < k_len) : (k_idx += N * VEC) {
                    // Prefetch next cache lines for weights (F16 = 2 bytes per element)
                    @prefetch(@as([*]const u8, @ptrCast(b_row.ptr + k_idx + N * VEC)), .{ .locality = 3 });

                    inline for (0..N) |vec_idx| {
                        const off = k_idx + vec_idx * VEC;
                        const a_vec: @Vector(VEC, f32) = a_row[off..][0..VEC].*;
                        const b_u16: @Vector(VEC, u16) = b_row[off..][0..VEC].*;
                        const b_vec: @Vector(VEC, f32) = fp16VecToF32Bits(VEC, b_u16);
                        acc[vec_idx] = @mulAdd(@Vector(VEC, f32), a_vec, b_vec, acc[vec_idx]);
                    }
                }

                while (k_idx + VEC - 1 < k_len) : (k_idx += VEC) {
                    const a_vec: @Vector(VEC, f32) = a_row[k_idx..][0..VEC].*;
                    const b_u16: @Vector(VEC, u16) = b_row[k_idx..][0..VEC].*;
                    const b_vec: @Vector(VEC, f32) = fp16VecToF32Bits(VEC, b_u16);
                    acc[0] = @mulAdd(@Vector(VEC, f32), a_vec, b_vec, acc[0]);
                }

                var total: @Vector(VEC, f32) = @splat(0);
                inline for (0..N) |vec_idx| total += acc[vec_idx];
                var sum = @reduce(.Add, total);

                // Scalar tail
                while (k_idx < k_len) : (k_idx += 1) {
                    sum += a_row[k_idx] * fp16ToF32(b_row[k_idx]);
                }
                out_row[col_idx] = sum;
            }
        }
    }.runDecodeCols;

    if (m_rows == 1) {
        // Decode: parallelize over output columns.
        parallel.global().parallelFor(n_cols, decode_task, &context);
    } else {
        // Prefill/batch: tile over rows * columns
        // Use smaller tiles for F16 to keep weights in L1 cache (memory-bound workload)
        const tile_size = COL_TILE_SIZE_FP16;
        const tiles_per_row = (n_cols + tile_size - 1) / tile_size;
        const total_tiles = m_rows * tiles_per_row;

        const tiled_task = struct {
            fn runRowColTiles(start: usize, end: usize, task_ctx: *MatmulF16Ctx) void {
                const k_len = task_ctx.k_dim;
                const n_len = task_ctx.n_cols;
                const tile_size_local = COL_TILE_SIZE_FP16;
                const tiles_per_row_local = (n_len + tile_size_local - 1) / tile_size_local;
                const VEC = simd.f32_vec_len;
                const N = 4; // Increased unroll factor for better ILP

                for (start..end) |tile_idx| {
                    const row = tile_idx / tiles_per_row_local;
                    const col_tile = tile_idx % tiles_per_row_local;
                    const col_start = col_tile * tile_size_local;
                    const col_end = @min(col_start + tile_size_local, n_len);

                    const a_row = task_ctx.a[row * k_len ..][0..k_len];
                    const out_row = task_ctx.c[row * n_len ..][0..n_len];

                    for (col_start..col_end) |col_idx| {
                        const b_row = task_ctx.b[col_idx * k_len ..][0..k_len];
                        var acc: [N]@Vector(VEC, f32) = .{@as(@Vector(VEC, f32), @splat(0))} ** N;
                        var k_idx: usize = 0;

                        // Main loop with prefetch and 4x unroll
                        while (k_idx + N * VEC - 1 < k_len) : (k_idx += N * VEC) {
                            // Prefetch next cache lines for weights
                            @prefetch(@as([*]const u8, @ptrCast(b_row.ptr + k_idx + N * VEC)), .{ .locality = 3 });

                            inline for (0..N) |vec_idx| {
                                const off = k_idx + vec_idx * VEC;
                                const a_vec: @Vector(VEC, f32) = a_row[off..][0..VEC].*;
                                const b_u16: @Vector(VEC, u16) = b_row[off..][0..VEC].*;
                                const b_vec: @Vector(VEC, f32) = fp16VecToF32Bits(VEC, b_u16);
                                acc[vec_idx] = @mulAdd(@Vector(VEC, f32), a_vec, b_vec, acc[vec_idx]);
                            }
                        }

                        while (k_idx + VEC - 1 < k_len) : (k_idx += VEC) {
                            const a_vec: @Vector(VEC, f32) = a_row[k_idx..][0..VEC].*;
                            const b_u16: @Vector(VEC, u16) = b_row[k_idx..][0..VEC].*;
                            const b_vec: @Vector(VEC, f32) = fp16VecToF32Bits(VEC, b_u16);
                            acc[0] = @mulAdd(@Vector(VEC, f32), a_vec, b_vec, acc[0]);
                        }

                        var total: @Vector(VEC, f32) = @splat(0);
                        inline for (0..N) |vec_idx| total += acc[vec_idx];
                        var sum = @reduce(.Add, total);

                        // Scalar tail
                        while (k_idx < k_len) : (k_idx += 1) {
                            sum += a_row[k_idx] * fp16ToF32(b_row[k_idx]);
                        }
                        out_row[col_idx] = sum;
                    }
                }
            }
        }.runRowColTiles;

        parallel.global().parallelFor(total_tiles, tiled_task, &context);
    }
}

/// Optimized SIMD dot product for grouped-affine u4 with pre-converted scales/biases
pub inline fn gaffineU4DotProductOpt(
    a_ptr: [*]const f32,
    w_ptr: [*]align(1) const u32,
    scales_f32: [*]const f32,
    biases_f32: [*]const f32,
    group: usize,
    k_div_group: usize,
    group_u32: usize,
) f32 {
    @setFloatMode(.optimized);

    // Use native vector width: 4x f32 for ARM NEON, 8x f32 for x86 AVX2
    const VEC = simd.f32_vec_len;
    var acc0: @Vector(VEC, f32) = @splat(0);
    var acc1: @Vector(VEC, f32) = @splat(0);

    var group_idx: usize = 0;
    while (group_idx < k_div_group) : (group_idx += 1) {
        // Use pre-converted f32 scales/biases (no BF16 conversion in hot loop!)
        const scale = scales_f32[group_idx];
        const bias = biases_f32[group_idx];

        const weight_base = w_ptr + group_idx * group_u32;
        const act_base = a_ptr + group_idx * group;

        var local0: @Vector(VEC, f32) = @splat(0);
        var local1: @Vector(VEC, f32) = @splat(0);
        var act0: @Vector(VEC, f32) = @splat(0);
        var act1: @Vector(VEC, f32) = @splat(0);

        var pack_idx: usize = 0;
        // extract32NibblesToFloat always returns 4x @Vector(8, f32)
        // But activation (x) is read in chunks of VEC (4 for ARM, 8 for x86)
        // So we need to process nibbles differently based on VEC
        if (VEC == 4) {
            // ARM: process 4 f32 at a time, but nibbles come in groups of 8
            while (pack_idx + 1 < group_u32) : (pack_idx += 2) {
                const nibs = extract32NibblesToFloat(weight_base + pack_idx);

                const x0: @Vector(4, f32) = (act_base + pack_idx * 8)[0..4].*;
                const x1: @Vector(4, f32) = (act_base + pack_idx * 8 + 4)[0..4].*;
                const x2: @Vector(4, f32) = (act_base + (pack_idx + 1) * 8)[0..4].*;
                const x3: @Vector(4, f32) = (act_base + (pack_idx + 1) * 8 + 4)[0..4].*;

                const n0: @Vector(4, f32) = @shuffle(f32, nibs.n0, undefined, [4]i32{ 0, 1, 2, 3 });
                const n1: @Vector(4, f32) = @shuffle(f32, nibs.n0, undefined, [4]i32{ 4, 5, 6, 7 });
                const n2: @Vector(4, f32) = @shuffle(f32, nibs.n1, undefined, [4]i32{ 0, 1, 2, 3 });
                const n3: @Vector(4, f32) = @shuffle(f32, nibs.n1, undefined, [4]i32{ 4, 5, 6, 7 });

                local0 = @mulAdd(@Vector(4, f32), n0, x0, local0);
                local1 = @mulAdd(@Vector(4, f32), n1, x1, local1);
                local0 = @mulAdd(@Vector(4, f32), n2, x2, local0);
                local1 = @mulAdd(@Vector(4, f32), n3, x3, local1);

                act0 += x0;
                act1 += x1;
                act0 += x2;
                act1 += x3;
            }
        } else {
            // x86: process 8 f32 at a time
            while (pack_idx + 3 < group_u32) : (pack_idx += 4) {
                @prefetch(@as([*]const u8, @ptrCast(weight_base + pack_idx + 16)), .{ .locality = 3 });
                @prefetch(@as([*]const u8, @ptrCast(act_base + (pack_idx + 4) * 8)), .{ .locality = 3 });

                const nibs = extract32NibblesToFloat(weight_base + pack_idx);

                const x0: @Vector(8, f32) = (act_base + pack_idx * 8)[0..8].*;
                const x1: @Vector(8, f32) = (act_base + (pack_idx + 1) * 8)[0..8].*;
                const x2: @Vector(8, f32) = (act_base + (pack_idx + 2) * 8)[0..8].*;
                const x3: @Vector(8, f32) = (act_base + (pack_idx + 3) * 8)[0..8].*;

                local0 = @mulAdd(@Vector(8, f32), nibs.n0, x0, local0);
                local1 = @mulAdd(@Vector(8, f32), nibs.n1, x1, local1);
                local0 = @mulAdd(@Vector(8, f32), nibs.n2, x2, local0);
                local1 = @mulAdd(@Vector(8, f32), nibs.n3, x3, local1);

                act0 += x0;
                act1 += x1;
                act0 += x2;
                act1 += x3;
            }
        }

        while (pack_idx < group_u32) : (pack_idx += 1) {
            const nibble_values = extractNibbles(weight_base[pack_idx]);
            if (VEC == 4) {
                const x0: @Vector(4, f32) = (act_base + pack_idx * 8)[0..4].*;
                const x1: @Vector(4, f32) = (act_base + pack_idx * 8 + 4)[0..4].*;
                const n0: @Vector(4, f32) = @shuffle(f32, nibble_values, undefined, [4]i32{ 0, 1, 2, 3 });
                const n1: @Vector(4, f32) = @shuffle(f32, nibble_values, undefined, [4]i32{ 4, 5, 6, 7 });
                local0 = @mulAdd(@Vector(4, f32), n0, x0, local0);
                local1 = @mulAdd(@Vector(4, f32), n1, x1, local1);
                act0 += x0;
                act1 += x1;
            } else {
                const x: @Vector(8, f32) = (act_base + pack_idx * 8)[0..8].*;
                local0 = @mulAdd(@Vector(8, f32), nibble_values, x, local0);
                act0 += x;
            }
        }

        const local = local0 + local1;
        const act = act0 + act1;
        const scale_vec: @Vector(VEC, f32) = @splat(scale);
        const bias_vec: @Vector(VEC, f32) = @splat(bias);
        acc0 = @mulAdd(@Vector(VEC, f32), local, scale_vec, acc0);
        acc1 = @mulAdd(@Vector(VEC, f32), act, bias_vec, acc1);
    }

    return @reduce(.Add, acc0 + acc1);
}

/// Simple reference implementation for debugging - scalar, no SIMD.
/// Only compiled when debug_matmul build option is enabled.
const gaffineU4DotProductRef = if (debug_matmul) gaffineU4DotProductRefImpl else void;

fn gaffineU4DotProductRefImpl(
    a_ptr: [*]const f32,
    w_ptr: [*]align(1) const u32,
    scales: [*]align(1) const u16,
    biases: [*]align(1) const u16,
    scales_dtype: DType,
    k: usize,
    group: usize,
) f32 {
    var result: f32 = 0;
    const k_div_group = k / group;
    const group_u32 = group / 8;

    var group_idx: usize = 0;
    while (group_idx < k_div_group) : (group_idx += 1) {
        const scale = gaffineScaleBiasToF32(scales_dtype, scales[group_idx]);
        const bias = gaffineScaleBiasToF32(scales_dtype, biases[group_idx]);

        var wx_sum: f32 = 0;
        var x_sum: f32 = 0;

        var pack_idx: usize = 0;
        while (pack_idx < group_u32) : (pack_idx += 1) {
            const packed_w = w_ptr[group_idx * group_u32 + pack_idx];
            // Extract nibbles in shift order (packed nibble order)
            var nib: usize = 0;
            while (nib < 8) : (nib += 1) {
                const nibble: f32 = @floatFromInt((packed_w >> @intCast(nib * 4)) & 0xF);
                const x_idx = group_idx * group + pack_idx * 8 + nib;
                const input_value = a_ptr[x_idx];
                wx_sum += nibble * input_value;
                x_sum += input_value;
            }
        }

        result += scale * wx_sum + bias * x_sum;
    }

    return result;
}

/// Grouped-affine u4 matmul: C = A × B^T where B is [n, k] packed 4-bit weights.
/// INVARIANT: b.gaffine must be non-null. This is guaranteed by weights loader which
/// sets b.gaffine when loading grouped-affine weights. matmulKernel() maps dtype to kernel,
/// ensuring this function is only called for .grouped_affine_u4 tensors which always have
/// .gaffine metadata.
pub fn matmulGaffineU4(a: *const Tensor, b: *const Tensor, out: *Tensor, scratch: *MatmulScratch) void {
    _ = scratch;
    std.debug.assert(a.dtype == .f32 and b.dtype == .grouped_affine_u4 and out.dtype == .f32);
    std.debug.assert(a.n_dims == 2 and b.n_dims == 2 and out.n_dims == 2);

    const m_rows: usize = @intCast(a.shape[0]);
    const k_dim: usize = @intCast(a.shape[1]);
    const n_cols: usize = @intCast(b.shape[0]);

    std.debug.assert(out.shape[0] == a.shape[0] and out.shape[1] == b.shape[0]);
    std.debug.assert(b.shape[1] == k_dim);

    const gaffine = b.gaffine.?;
    const group = gaffine.group_size;
    const scales_dtype = gaffine.scales_dtype;
    const scales: []align(1) const u16 = @as([*]align(1) const u16, @ptrCast(gaffine.scales.ptr))[0 .. gaffine.scales.len / 2];
    const biases: []align(1) const u16 = @as([*]align(1) const u16, @ptrCast(gaffine.biases.ptr))[0 .. gaffine.biases.len / 2];
    const packed_vals: []align(1) const u32 = @as([*]align(1) const u32, @ptrCast(b.data().ptr))[0 .. b.data().len / 4];

    std.debug.assert(packed_vals.len * 8 >= k_dim * n_cols);

    const a_data = a.asSlice(f32);
    const out_data = out.asSlice(f32);

    // IMPORTANT: Keep the CPU backend CPU-only in this kernel.
    // Per-op MLX/Metal offload from the CPU path causes frequent host<->device
    // transfers and synchronization, which regresses small/decode-heavy and
    // heterogeneous workloads (for example Granite with many Mamba blocks).
    // GPU acceleration belongs in the Metal backend selection, not inside a
    // CPU matmul primitive.

    // Prefill (m_rows > 1): use dedicated prefill kernel
    if (m_rows > 1) {
        log.trace("compute", "gaffine_u4 prefill", .{ .m = m_rows, .k = k_dim, .n = n_cols, .group = group }, @src());
        prefill.matmulGaffineU4Prefill(a_data, m_rows, k_dim, packed_vals, scales, biases, scales_dtype, n_cols, group, out_data);
        return;
    }

    // Decode (m_rows == 1): parallelize over columns
    const k_div_8 = k_dim / 8;
    const k_div_group = k_dim / group;
    const group_u32 = group / 8;

    // Defense-in-depth: validate k_div_group fits in stack buffers (should be checked at load time)
    std.debug.assert(k_div_group <= MAX_GROUPS);

    const MatmulGaffineU4Ctx = struct {
        a: []const f32,
        packed_b: []align(1) const u32,
        scales: []align(1) const u16,
        biases: []align(1) const u16,
        scales_dtype: DType,
        out: []f32,
        n_cols: usize,
        k_dim: usize,
        group: usize,
        k_div_8: usize,
        k_div_group: usize,
        group_u32: usize,
    };

    var context = MatmulGaffineU4Ctx{
        .a = a_data,
        .packed_b = packed_vals,
        .scales = scales,
        .biases = biases,
        .scales_dtype = scales_dtype,
        .out = out_data,
        .n_cols = n_cols,
        .k_dim = k_dim,
        .group = group,
        .k_div_8 = k_div_8,
        .k_div_group = k_div_group,
        .group_u32 = group_u32,
    };

    const decode_task = struct {
        fn runDecodeCols(start: usize, end: usize, task_ctx: *MatmulGaffineU4Ctx) void {
            var scales_f32: [MAX_GROUPS]f32 align(64) = undefined; // filled in loop below
            var biases_f32: [MAX_GROUPS]f32 align(64) = undefined; // filled in loop below

            const a_ptr = task_ctx.a.ptr;

            for (start..end) |col| {
                const w_ptr = task_ctx.packed_b.ptr + col * task_ctx.k_div_8;
                const s_ptr = task_ctx.scales.ptr + col * task_ctx.k_div_group;
                const b_ptr = task_ctx.biases.ptr + col * task_ctx.k_div_group;

                for (0..task_ctx.k_div_group) |group_idx| {
                    scales_f32[group_idx] = gaffineScaleBiasToF32(task_ctx.scales_dtype, s_ptr[group_idx]);
                    biases_f32[group_idx] = gaffineScaleBiasToF32(task_ctx.scales_dtype, b_ptr[group_idx]);
                }

                task_ctx.out[col] = gaffineU4DotProductOpt(
                    a_ptr,
                    w_ptr,
                    &scales_f32,
                    &biases_f32,
                    task_ctx.group,
                    task_ctx.k_div_group,
                    task_ctx.group_u32,
                );
            }
        }
    }.runDecodeCols;
    parallel.global().parallelFor(n_cols, decode_task, &context);
}

/// Optimized grouped-affine u8 dot product with pre-converted scales/biases
pub inline fn gaffineU8DotProductOpt(
    a_ptr: [*]const f32,
    w_ptr: [*]align(1) const u32,
    scales_f32: [*]const f32,
    biases_f32: [*]const f32,
    group: usize,
    k_div_group: usize,
    group_u32: usize,
) f32 {
    var acc0: @Vector(4, f32) = @splat(0);
    var acc1: @Vector(4, f32) = @splat(0);

    var group_idx: usize = 0;
    while (group_idx < k_div_group) : (group_idx += 1) {
        const scale = scales_f32[group_idx];
        const bias = biases_f32[group_idx];

        const weight_base = w_ptr + group_idx * group_u32;
        const act_base = a_ptr + group_idx * group;

        var local0: @Vector(4, f32) = @splat(0);
        var local1: @Vector(4, f32) = @splat(0);
        var act0: @Vector(4, f32) = @splat(0);
        var act1: @Vector(4, f32) = @splat(0);

        var pack_idx: usize = 0;
        while (pack_idx + 1 < group_u32) : (pack_idx += 2) {
            const bytes0 = extractBytes(weight_base[pack_idx]);
            const bytes1 = extractBytes(weight_base[pack_idx + 1]);

            const x0: @Vector(4, f32) = (act_base + pack_idx * 4)[0..4].*;
            const x1: @Vector(4, f32) = (act_base + (pack_idx + 1) * 4)[0..4].*;

            local0 = @mulAdd(@Vector(4, f32), bytes0, x0, local0);
            local1 = @mulAdd(@Vector(4, f32), bytes1, x1, local1);

            act0 += x0;
            act1 += x1;
        }

        while (pack_idx < group_u32) : (pack_idx += 1) {
            const bytes = extractBytes(weight_base[pack_idx]);
            const x: @Vector(4, f32) = (act_base + pack_idx * 4)[0..4].*;
            local0 = @mulAdd(@Vector(4, f32), bytes, x, local0);
            act0 += x;
        }

        const local = local0 + local1;
        const act = act0 + act1;
        const scale_vec: @Vector(4, f32) = @splat(scale);
        const bias_vec: @Vector(4, f32) = @splat(bias);
        acc0 = @mulAdd(@Vector(4, f32), local, scale_vec, acc0);
        acc1 = @mulAdd(@Vector(4, f32), act, bias_vec, acc1);
    }

    return @reduce(.Add, acc0 + acc1);
}

/// Grouped-affine u8 matmul: C = A × B^T where B is [n, k] packed 8-bit weights.
/// INVARIANT: b.gaffine must be non-null. See matmulGaffineU4 for details.
fn matmulGaffineU8(a: *const Tensor, b: *const Tensor, out: *Tensor, scratch: *MatmulScratch) void {
    _ = scratch;
    std.debug.assert(a.dtype == .f32 and b.dtype == .grouped_affine_u8 and out.dtype == .f32);
    std.debug.assert(a.n_dims == 2 and b.n_dims == 2 and out.n_dims == 2);

    const m_rows: usize = @intCast(a.shape[0]);
    const k_dim: usize = @intCast(a.shape[1]);
    const n_cols: usize = @intCast(b.shape[0]);

    std.debug.assert(out.shape[0] == a.shape[0] and out.shape[1] == b.shape[0]);
    std.debug.assert(b.shape[1] == k_dim);

    const gaffine = b.gaffine.?;
    const group = gaffine.group_size;
    const scales_dtype = gaffine.scales_dtype;
    const scales: []align(1) const u16 = @as([*]align(1) const u16, @ptrCast(gaffine.scales.ptr))[0 .. gaffine.scales.len / 2];
    const biases: []align(1) const u16 = @as([*]align(1) const u16, @ptrCast(gaffine.biases.ptr))[0 .. gaffine.biases.len / 2];
    const packed_vals: []align(1) const u32 = @as([*]align(1) const u32, @ptrCast(b.data().ptr))[0 .. b.data().len / 4];

    std.debug.assert(packed_vals.len * 4 >= k_dim * n_cols);

    const a_data = a.asSlice(f32);
    const out_data = out.asSlice(f32);

    // Prefill (m_rows > 1): use dedicated prefill kernel
    if (m_rows > 1) {
        prefill.matmulGaffineU8Prefill(a_data, m_rows, k_dim, packed_vals, scales, biases, scales_dtype, n_cols, group, out_data);
        return;
    }

    const k_div_4 = k_dim / 4;
    const k_div_group = k_dim / group;
    const group_u32 = group / 4;

    // Defense-in-depth: validate k_div_group fits in stack buffers (should be checked at load time)
    std.debug.assert(k_div_group <= MAX_GROUPS);

    const MatmulGaffineU8Ctx = struct {
        a: []const f32,
        packed_b: []align(1) const u32,
        scales: []align(1) const u16,
        biases: []align(1) const u16,
        scales_dtype: DType,
        out: []f32,
        m_rows: usize,
        n_cols: usize,
        k_dim: usize,
        group: usize,
        k_div_4: usize,
        k_div_group: usize,
        group_u32: usize,
    };

    var context = MatmulGaffineU8Ctx{
        .a = a_data,
        .packed_b = packed_vals,
        .scales = scales,
        .biases = biases,
        .scales_dtype = scales_dtype,
        .out = out_data,
        .m_rows = m_rows,
        .n_cols = n_cols,
        .k_dim = k_dim,
        .group = group,
        .k_div_4 = k_div_4,
        .k_div_group = k_div_group,
        .group_u32 = group_u32,
    };

    const row_tiles_task = struct {
        fn runRowTiles(start: usize, end: usize, task_ctx: *MatmulGaffineU8Ctx) void {
            var scales_f32: [MAX_GROUPS]f32 align(64) = undefined; // filled in loop below
            var biases_f32: [MAX_GROUPS]f32 align(64) = undefined; // filled in loop below

            for (start..end) |row| {
                const a_ptr = task_ctx.a.ptr + row * task_ctx.k_dim;
                const out_row = task_ctx.out[row * task_ctx.n_cols ..][0..task_ctx.n_cols];

                for (0..task_ctx.n_cols) |col| {
                    const w_ptr = task_ctx.packed_b.ptr + col * task_ctx.k_div_4;
                    const s_ptr = task_ctx.scales.ptr + col * task_ctx.k_div_group;
                    const b_ptr = task_ctx.biases.ptr + col * task_ctx.k_div_group;

                    for (0..task_ctx.k_div_group) |group_idx| {
                        scales_f32[group_idx] = gaffineScaleBiasToF32(task_ctx.scales_dtype, s_ptr[group_idx]);
                        biases_f32[group_idx] = gaffineScaleBiasToF32(task_ctx.scales_dtype, b_ptr[group_idx]);
                    }

                    out_row[col] = gaffineU8DotProductOpt(
                        a_ptr,
                        w_ptr,
                        &scales_f32,
                        &biases_f32,
                        task_ctx.group,
                        task_ctx.k_div_group,
                        task_ctx.group_u32,
                    );
                }
            }
        }
    }.runRowTiles;

    // Tiled parallelization for better load balancing (same strategy as 4-bit)
    if (m_rows >= TILE_THRESHOLD) {
        parallel.global().parallelFor(m_rows, row_tiles_task, &context);
    } else if (m_rows == 1) {
        const decode_task = struct {
            fn runDecodeCols(start: usize, end: usize, task_ctx: *MatmulGaffineU8Ctx) void {
                var scales_f32: [MAX_GROUPS]f32 align(64) = undefined; // filled in loop below
                var biases_f32: [MAX_GROUPS]f32 align(64) = undefined; // filled in loop below

                for (start..end) |col| {
                    const w_ptr = task_ctx.packed_b.ptr + col * task_ctx.k_div_4;
                    const s_ptr = task_ctx.scales.ptr + col * task_ctx.k_div_group;
                    const b_ptr = task_ctx.biases.ptr + col * task_ctx.k_div_group;

                    for (0..task_ctx.k_div_group) |group_idx| {
                        scales_f32[group_idx] = gaffineScaleBiasToF32(task_ctx.scales_dtype, s_ptr[group_idx]);
                        biases_f32[group_idx] = gaffineScaleBiasToF32(task_ctx.scales_dtype, b_ptr[group_idx]);
                    }

                    const a_ptr = task_ctx.a.ptr;
                    task_ctx.out[col] = gaffineU8DotProductOpt(
                        a_ptr,
                        w_ptr,
                        &scales_f32,
                        &biases_f32,
                        task_ctx.group,
                        task_ctx.k_div_group,
                        task_ctx.group_u32,
                    );
                }
            }
        }.runDecodeCols;
        parallel.global().parallelFor(n_cols, decode_task, &context);
    } else {
        // Small batch (2 <= m_rows < 64): tile across rows AND columns
        const tiles_per_row = (n_cols + COL_TILE_SIZE - 1) / COL_TILE_SIZE;
        const total_tiles = m_rows * tiles_per_row;

        const MatmulGaffineU8TileCtx = struct {
            a: []const f32,
            packed_b: []align(1) const u32,
            scales: []align(1) const u16,
            biases: []align(1) const u16,
            scales_dtype: DType,
            out: []f32,
            m_rows: usize,
            n_cols: usize,
            k_dim: usize,
            group: usize,
            k_div_4: usize,
            k_div_group: usize,
            group_u32: usize,
            tiles_per_row: usize,
        };

        var tiled_ctx = MatmulGaffineU8TileCtx{
            .a = a_data,
            .packed_b = packed_vals,
            .scales = scales,
            .biases = biases,
            .scales_dtype = scales_dtype,
            .out = out_data,
            .m_rows = m_rows,
            .n_cols = n_cols,
            .k_dim = k_dim,
            .group = group,
            .k_div_4 = k_div_4,
            .k_div_group = k_div_group,
            .group_u32 = group_u32,
            .tiles_per_row = tiles_per_row,
        };

        const tiled_task = struct {
            fn runRowColTiles(start: usize, end: usize, task_ctx: *MatmulGaffineU8TileCtx) void {
                var scales_f32: [MAX_GROUPS]f32 align(64) = undefined; // filled in loop below
                var biases_f32: [MAX_GROUPS]f32 align(64) = undefined; // filled in loop below

                for (start..end) |tile_idx| {
                    const row = tile_idx / task_ctx.tiles_per_row;
                    const col_tile = tile_idx % task_ctx.tiles_per_row;
                    const col_start = col_tile * COL_TILE_SIZE;
                    const col_end = @min(col_start + COL_TILE_SIZE, task_ctx.n_cols);

                    const a_ptr = task_ctx.a.ptr + row * task_ctx.k_dim;
                    const out_row = task_ctx.out[row * task_ctx.n_cols ..][0..task_ctx.n_cols];

                    for (col_start..col_end) |col| {
                        const w_ptr = task_ctx.packed_b.ptr + col * task_ctx.k_div_4;
                        const s_ptr = task_ctx.scales.ptr + col * task_ctx.k_div_group;
                        const b_ptr = task_ctx.biases.ptr + col * task_ctx.k_div_group;

                        for (0..task_ctx.k_div_group) |group_idx| {
                            scales_f32[group_idx] = gaffineScaleBiasToF32(task_ctx.scales_dtype, s_ptr[group_idx]);
                            biases_f32[group_idx] = gaffineScaleBiasToF32(task_ctx.scales_dtype, b_ptr[group_idx]);
                        }

                        out_row[col] = gaffineU8DotProductOpt(
                            a_ptr,
                            w_ptr,
                            &scales_f32,
                            &biases_f32,
                            task_ctx.group,
                            task_ctx.k_div_group,
                            task_ctx.group_u32,
                        );
                    }
                }
            }
        }.runRowColTiles;
        parallel.global().parallelFor(total_tiles, tiled_task, &tiled_ctx);
    }
}

test "MatmulScratch init deinit" {
    const allocator = std.testing.allocator;
    var scratch = try MatmulScratch.init(allocator);
    defer scratch.deinit();
    // MatmulScratch is now minimal - just verify it can be created and destroyed
}

test "matmulF32 basic 1x1" {
    const allocator = std.testing.allocator;

    // Test 1x1 matrix multiplication: [2.0] × [3.0] = [6.0]
    var a = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 1, 1 });
    defer a.deinit();
    var b = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 1, 1 });
    defer b.deinit();
    var out = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 1, 1 });
    defer out.deinit();

    a.asSlice(f32)[0] = 2.0;
    b.asSlice(f32)[0] = 3.0;

    var a_view = a.view();
    var b_view = b.view();
    var out_view = out.view();
    var scratch = try MatmulScratch.init(allocator);
    defer scratch.deinit();

    matmulF32(&a_view, &b_view, &out_view, &scratch);
    try std.testing.expectApproxEqAbs(6.0, out.asSlice(f32)[0], 1e-5);
}

test "matmulF32 basic 2x2" {
    const allocator = std.testing.allocator;

    // Test 2×2 matrix multiplication:
    // [1 2]   [5 6]   [19 22]
    // [3 4] × [7 8] = [43 50]
    var a = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 2, 2 });
    defer a.deinit();
    var b = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 2, 2 });
    defer b.deinit();
    var out = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 2, 2 });
    defer out.deinit();

    const a_data = a.asSlice(f32);
    a_data[0] = 1.0;
    a_data[1] = 2.0;
    a_data[2] = 3.0;
    a_data[3] = 4.0;

    const b_data = b.asSlice(f32);
    b_data[0] = 5.0;
    b_data[1] = 6.0;
    b_data[2] = 7.0;
    b_data[3] = 8.0;

    var a_view = a.view();
    var b_view = b.view();
    var out_view = out.view();
    var scratch = try MatmulScratch.init(allocator);
    defer scratch.deinit();

    matmulF32(&a_view, &b_view, &out_view, &scratch);

    const out_data = out.asSlice(f32);
    try std.testing.expectApproxEqAbs(19.0, out_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(22.0, out_data[1], 1e-5);
    try std.testing.expectApproxEqAbs(43.0, out_data[2], 1e-5);
    try std.testing.expectApproxEqAbs(50.0, out_data[3], 1e-5);
}

test "matmulF32 1xN row vector" {
    const allocator = std.testing.allocator;

    // Test 1×3 × 3×2 = 1×2
    // [1 2 3] × [[1 2]  = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    //            [3 4]
    //            [5 6]]
    var a = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 1, 3 });
    defer a.deinit();
    var b = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 3, 2 });
    defer b.deinit();
    var out = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 1, 2 });
    defer out.deinit();

    const a_data = a.asSlice(f32);
    a_data[0] = 1.0;
    a_data[1] = 2.0;
    a_data[2] = 3.0;

    const b_data = b.asSlice(f32);
    b_data[0] = 1.0;
    b_data[1] = 2.0;
    b_data[2] = 3.0;
    b_data[3] = 4.0;
    b_data[4] = 5.0;
    b_data[5] = 6.0;

    var a_view = a.view();
    var b_view = b.view();
    var out_view = out.view();
    var scratch = try MatmulScratch.init(allocator);
    defer scratch.deinit();

    matmulF32(&a_view, &b_view, &out_view, &scratch);

    const out_data = out.asSlice(f32);
    try std.testing.expectApproxEqAbs(22.0, out_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(28.0, out_data[1], 1e-5);
}

test "matmulF32 Nx1 column vector" {
    const allocator = std.testing.allocator;

    // Test 3×2 × 2×1 = 3×1
    // [[1 2]     [5]   [19]
    //  [3 4]  ×  [7] = [43]
    //  [5 6]]          [67]
    var a = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 3, 2 });
    defer a.deinit();
    var b = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 2, 1 });
    defer b.deinit();
    var out = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 3, 1 });
    defer out.deinit();

    const a_data = a.asSlice(f32);
    a_data[0] = 1.0;
    a_data[1] = 2.0;
    a_data[2] = 3.0;
    a_data[3] = 4.0;
    a_data[4] = 5.0;
    a_data[5] = 6.0;

    const b_data = b.asSlice(f32);
    b_data[0] = 5.0;
    b_data[1] = 7.0;

    var a_view = a.view();
    var b_view = b.view();
    var out_view = out.view();
    var scratch = try MatmulScratch.init(allocator);
    defer scratch.deinit();

    matmulF32(&a_view, &b_view, &out_view, &scratch);

    const out_data = out.asSlice(f32);
    try std.testing.expectApproxEqAbs(19.0, out_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(43.0, out_data[1], 1e-5);
    try std.testing.expectApproxEqAbs(67.0, out_data[2], 1e-5);
}

test "matmulF32 zeros" {
    const allocator = std.testing.allocator;

    // Test with zero matrices
    var a = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 2, 3 });
    defer a.deinit();
    var b = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 3, 2 });
    defer b.deinit();
    var out = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 2, 2 });
    defer out.deinit();

    // Initialize all to zero
    for (a.asSlice(f32)) |*v| v.* = 0.0;
    for (b.asSlice(f32)) |*v| v.* = 0.0;

    var a_view = a.view();
    var b_view = b.view();
    var out_view = out.view();
    var scratch = try MatmulScratch.init(allocator);
    defer scratch.deinit();

    matmulF32(&a_view, &b_view, &out_view, &scratch);

    for (out.asSlice(f32)) |v| {
        try std.testing.expectApproxEqAbs(0.0, v, 1e-5);
    }
}

test "matmulKernel returns correct function for dtype" {
    // Test that matmulKernel returns the expected function pointer for each dtype
    const bf16_dk = try matmulKernel(.bf16);
    try std.testing.expectEqual(@as(MatmulFn, matmulBF16), bf16_dk.func);
    try std.testing.expectEqualStrings("matmulBF16", bf16_dk.name);

    const f16_dk = try matmulKernel(.f16);
    try std.testing.expectEqual(@as(MatmulFn, matmulF16), f16_dk.func);
    try std.testing.expectEqualStrings("matmulF16", f16_dk.name);

    const f32_dk = try matmulKernel(.f32);
    try std.testing.expectEqual(@as(MatmulFn, matmulF32), f32_dk.func);
    try std.testing.expectEqualStrings("matmulF32", f32_dk.name);

    const gaffine_u4_dk = try matmulKernel(.grouped_affine_u4);
    try std.testing.expectEqual(@as(MatmulFn, matmulGaffineU4), gaffine_u4_dk.func);
    try std.testing.expectEqualStrings("matmulGaffineU4", gaffine_u4_dk.name);

    const gaffine_u8_dk = try matmulKernel(.grouped_affine_u8);
    try std.testing.expectEqual(@as(MatmulFn, matmulGaffineU8), gaffine_u8_dk.func);
    try std.testing.expectEqualStrings("matmulGaffineU8", gaffine_u8_dk.name);
}

test "matmulKernel unsupported dtype" {
    // Test that unsupported dtypes return an error
    const result = matmulKernel(.i8);
    try std.testing.expectError(error.UnsupportedDType, result);
}

test "matmulAuto dispatches to correct kernel" {
    const allocator = std.testing.allocator;

    // Test that matmulAuto correctly dispatches for f32
    var a = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 2, 2 });
    defer a.deinit();
    var b = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 2, 2 });
    defer b.deinit();
    var out = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 2, 2 });
    defer out.deinit();

    const a_data = a.asSlice(f32);
    a_data[0] = 1.0;
    a_data[1] = 2.0;
    a_data[2] = 3.0;
    a_data[3] = 4.0;

    const b_data = b.asSlice(f32);
    b_data[0] = 5.0;
    b_data[1] = 6.0;
    b_data[2] = 7.0;
    b_data[3] = 8.0;

    var a_view = a.view();
    var b_view = b.view();
    var out_view = out.view();
    var scratch = try MatmulScratch.init(allocator);
    defer scratch.deinit();

    try matmulAuto(&a_view, &b_view, &out_view, &scratch);

    const out_data = out.asSlice(f32);
    try std.testing.expectApproxEqAbs(19.0, out_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(22.0, out_data[1], 1e-5);
    try std.testing.expectApproxEqAbs(43.0, out_data[2], 1e-5);
    try std.testing.expectApproxEqAbs(50.0, out_data[3], 1e-5);
}

test "gaffineU4DotProductOpt zero weights" {
    const allocator = std.testing.allocator;
    const group_size: usize = 128;
    const k: usize = 128;
    const k_div_group = k / group_size;
    const group_u32 = group_size / 8;

    // Allocate test data
    const a_data = try allocator.alloc(f32, k);
    defer allocator.free(a_data);
    const w_data = try allocator.alloc(u32, k / 8);
    defer allocator.free(w_data);
    const scales = try allocator.alloc(f32, k_div_group);
    defer allocator.free(scales);
    const biases = try allocator.alloc(f32, k_div_group);
    defer allocator.free(biases);

    // Fill with ones for activation, zeros for weights
    for (a_data) |*v| v.* = 1.0;
    for (w_data) |*v| v.* = 0;
    for (scales) |*v| v.* = 1.0;
    for (biases) |*v| v.* = 0.0;

    const result = gaffineU4DotProductOpt(
        a_data.ptr,
        w_data.ptr,
        scales.ptr,
        biases.ptr,
        group_size,
        k_div_group,
        group_u32,
    );

    // Zero weights should produce zero output
    try std.testing.expectApproxEqAbs(0.0, result, 1e-5);
}

test "gaffineU4DotProductOpt with nonzero data" {
    const allocator = std.testing.allocator;
    const group_size: usize = 128;
    const k: usize = 128;
    const k_div_group = k / group_size;
    const group_u32 = group_size / 8;

    const a_data = try allocator.alloc(f32, k);
    defer allocator.free(a_data);
    const w_data = try allocator.alloc(u32, k / 8);
    defer allocator.free(w_data);
    const scales = try allocator.alloc(f32, k_div_group);
    defer allocator.free(scales);
    const biases = try allocator.alloc(f32, k_div_group);
    defer allocator.free(biases);

    // Fill with simple test pattern
    for (a_data) |*v| v.* = 2.0;
    for (w_data) |*v| v.* = 0x11111111; // All nibbles = 1
    for (scales) |*v| v.* = 0.5;
    for (biases) |*v| v.* = 0.1;

    const result = gaffineU4DotProductOpt(
        a_data.ptr,
        w_data.ptr,
        scales.ptr,
        biases.ptr,
        group_size,
        k_div_group,
        group_u32,
    );

    // Result should be non-zero
    try std.testing.expect(result != 0.0);
    // With scale=0.5, bias=0.1, weights=1, activations=2:
    // result ≈ scale * (sum of w*x) + bias * (sum of x)
    // ≈ 0.5 * (1*2*128) + 0.1 * (2*128) = 0.5*256 + 0.1*256 = 153.6
    try std.testing.expectApproxEqAbs(153.6, result, 1.0);
}

test "gaffineU8DotProductOpt zero weights" {
    const allocator = std.testing.allocator;
    const group_size: usize = 128;
    const k: usize = 128;
    const k_div_group = k / group_size;
    const group_u32 = group_size / 4;

    const a_data = try allocator.alloc(f32, k);
    defer allocator.free(a_data);
    const w_data = try allocator.alloc(u32, k / 4);
    defer allocator.free(w_data);
    const scales = try allocator.alloc(f32, k_div_group);
    defer allocator.free(scales);
    const biases = try allocator.alloc(f32, k_div_group);
    defer allocator.free(biases);

    for (a_data) |*v| v.* = 1.0;
    for (w_data) |*v| v.* = 0;
    for (scales) |*v| v.* = 1.0;
    for (biases) |*v| v.* = 0.0;

    const result = gaffineU8DotProductOpt(
        a_data.ptr,
        w_data.ptr,
        scales.ptr,
        biases.ptr,
        group_size,
        k_div_group,
        group_u32,
    );

    try std.testing.expectApproxEqAbs(0.0, result, 1e-5);
}

test "gaffineU8DotProductOpt with nonzero data" {
    const allocator = std.testing.allocator;
    const group_size: usize = 128;
    const k: usize = 128;
    const k_div_group = k / group_size;
    const group_u32 = group_size / 4;

    const a_data = try allocator.alloc(f32, k);
    defer allocator.free(a_data);
    const w_data = try allocator.alloc(u32, k / 4);
    defer allocator.free(w_data);
    const scales = try allocator.alloc(f32, k_div_group);
    defer allocator.free(scales);
    const biases = try allocator.alloc(f32, k_div_group);
    defer allocator.free(biases);

    for (a_data) |*v| v.* = 2.0;
    for (w_data) |*v| v.* = 0x01010101; // All bytes = 1
    for (scales) |*v| v.* = 0.5;
    for (biases) |*v| v.* = 0.1;

    const result = gaffineU8DotProductOpt(
        a_data.ptr,
        w_data.ptr,
        scales.ptr,
        biases.ptr,
        group_size,
        k_div_group,
        group_u32,
    );

    try std.testing.expect(result != 0.0);
    // With scale=0.5, bias=0.1, weights=1, activations=2:
    // result ≈ 0.5 * (1*2*128) + 0.1 * (2*128) = 153.6
    try std.testing.expectApproxEqAbs(153.6, result, 1.0);
}

test "matmulBF16 basic 1x4" {
    const allocator = std.testing.allocator;

    // Test 1×4 × 4×2 = 1×2 with BF16 weights
    // A = [1.0, 2.0, 3.0, 4.0] (f32)
    // B = [[1.0, 0.5],   (bf16, stored as [n, k] = [2, 4])
    //      [2.0, 1.0],
    //      [3.0, 1.5],
    //      [4.0, 2.0]]
    // Expected: out[0] = 1*1 + 2*2 + 3*3 + 4*4 = 30
    //           out[1] = 1*0.5 + 2*1 + 3*1.5 + 4*2 = 15

    var a = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 1, 4 });
    defer a.deinit();
    var b = try tensor_mod.OwnedTensor.init(allocator, .bf16, &.{ 2, 4 }); // [n, k] layout
    defer b.deinit();
    var out = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 1, 2 });
    defer out.deinit();

    const a_data = a.asSlice(f32);
    a_data[0] = 1.0;
    a_data[1] = 2.0;
    a_data[2] = 3.0;
    a_data[3] = 4.0;

    // BF16 weights: row 0 = [1.0, 2.0, 3.0, 4.0], row 1 = [0.5, 1.0, 1.5, 2.0]
    const b_data = b.asSlice(u16);
    b_data[0] = dtype_mod.f32ToBf16(1.0);
    b_data[1] = dtype_mod.f32ToBf16(2.0);
    b_data[2] = dtype_mod.f32ToBf16(3.0);
    b_data[3] = dtype_mod.f32ToBf16(4.0);
    b_data[4] = dtype_mod.f32ToBf16(0.5);
    b_data[5] = dtype_mod.f32ToBf16(1.0);
    b_data[6] = dtype_mod.f32ToBf16(1.5);
    b_data[7] = dtype_mod.f32ToBf16(2.0);

    var a_view = a.view();
    var b_view = b.view();
    var out_view = out.view();
    var scratch = try MatmulScratch.init(allocator);
    defer scratch.deinit();

    matmulBF16(&a_view, &b_view, &out_view, &scratch);

    const out_data = out.asSlice(f32);
    try std.testing.expectApproxEqAbs(30.0, out_data[0], 1e-3);
    try std.testing.expectApproxEqAbs(15.0, out_data[1], 1e-3);
}

test "matmulF16 basic 1x4" {
    const allocator = std.testing.allocator;

    // Same test as BF16 but with F16 weights using dedicated matmulF16 kernel
    var a = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 1, 4 });
    defer a.deinit();
    var b = try tensor_mod.OwnedTensor.init(allocator, .f16, &.{ 2, 4 }); // [n, k] layout
    defer b.deinit();
    var out = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 1, 2 });
    defer out.deinit();

    const a_data = a.asSlice(f32);
    a_data[0] = 1.0;
    a_data[1] = 2.0;
    a_data[2] = 3.0;
    a_data[3] = 4.0;

    // F16 weights: row 0 = [1.0, 2.0, 3.0, 4.0], row 1 = [0.5, 1.0, 1.5, 2.0]
    const b_data = b.asSlice(u16);
    b_data[0] = dtype_mod.f32ToFp16(1.0);
    b_data[1] = dtype_mod.f32ToFp16(2.0);
    b_data[2] = dtype_mod.f32ToFp16(3.0);
    b_data[3] = dtype_mod.f32ToFp16(4.0);
    b_data[4] = dtype_mod.f32ToFp16(0.5);
    b_data[5] = dtype_mod.f32ToFp16(1.0);
    b_data[6] = dtype_mod.f32ToFp16(1.5);
    b_data[7] = dtype_mod.f32ToFp16(2.0);

    var a_view = a.view();
    var b_view = b.view();
    var out_view = out.view();
    var scratch = try MatmulScratch.init(allocator);
    defer scratch.deinit();

    matmulF16(&a_view, &b_view, &out_view, &scratch);

    const out_data = out.asSlice(f32);
    try std.testing.expectApproxEqAbs(30.0, out_data[0], 1e-3);
    try std.testing.expectApproxEqAbs(15.0, out_data[1], 1e-3);
}

test "matmulF16 batch" {
    const allocator = std.testing.allocator;

    // Test batch matmul (m > 1) with F16 weights to exercise the tiled path
    // A = [[1, 2], [3, 4]] (2x2)
    // B = [[1, 2], [0.5, 1]] (2x2, stored as [n, k])
    // Expected: [[1*1 + 2*2, 1*0.5 + 2*1], [3*1 + 4*2, 3*0.5 + 4*1]]
    //         = [[5, 2.5], [11, 5.5]]

    var a = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 2, 2 });
    defer a.deinit();
    var b = try tensor_mod.OwnedTensor.init(allocator, .f16, &.{ 2, 2 });
    defer b.deinit();
    var out = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 2, 2 });
    defer out.deinit();

    const a_data = a.asSlice(f32);
    a_data[0] = 1.0;
    a_data[1] = 2.0;
    a_data[2] = 3.0;
    a_data[3] = 4.0;

    const b_data = b.asSlice(u16);
    b_data[0] = dtype_mod.f32ToFp16(1.0);
    b_data[1] = dtype_mod.f32ToFp16(2.0);
    b_data[2] = dtype_mod.f32ToFp16(0.5);
    b_data[3] = dtype_mod.f32ToFp16(1.0);

    var a_view = a.view();
    var b_view = b.view();
    var out_view = out.view();
    var scratch = try MatmulScratch.init(allocator);
    defer scratch.deinit();

    matmulF16(&a_view, &b_view, &out_view, &scratch);

    const out_data = out.asSlice(f32);
    try std.testing.expectApproxEqAbs(5.0, out_data[0], 1e-3);
    try std.testing.expectApproxEqAbs(2.5, out_data[1], 1e-3);
    try std.testing.expectApproxEqAbs(11.0, out_data[2], 1e-3);
    try std.testing.expectApproxEqAbs(5.5, out_data[3], 1e-3);
}

test "matmulF16 large k dimension" {
    const allocator = std.testing.allocator;

    // Test with k > VEC*N to ensure vectorized loop is exercised
    const k: usize = 64; // Large enough to trigger SIMD path
    const n: usize = 2;

    var a = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 1, k });
    defer a.deinit();
    var b = try tensor_mod.OwnedTensor.init(allocator, .f16, &.{ n, k });
    defer b.deinit();
    var out = try tensor_mod.OwnedTensor.init(allocator, .f32, &.{ 1, n });
    defer out.deinit();

    // Fill A with 1.0
    const a_data = a.asSlice(f32);
    for (a_data) |*v| v.* = 1.0;

    // Fill B row 0 with 1.0, row 1 with 2.0
    const b_data = b.asSlice(u16);
    for (0..k) |i| {
        b_data[i] = dtype_mod.f32ToFp16(1.0);
        b_data[k + i] = dtype_mod.f32ToFp16(2.0);
    }

    var a_view = a.view();
    var b_view = b.view();
    var out_view = out.view();
    var scratch = try MatmulScratch.init(allocator);
    defer scratch.deinit();

    matmulF16(&a_view, &b_view, &out_view, &scratch);

    const out_data = out.asSlice(f32);
    // out[0] = sum of k ones = k
    // out[1] = sum of k twos = 2k
    try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(k)), out_data[0], 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(2 * k)), out_data[1], 1e-2);
}
