//! Matmul with transposed B: C = A @ B^T.
//!
//! Computes C = A @ B^T where:
//!   A:   [M, K] row-major (f32)
//!   B:   [N, K] row-major (f32) — weights stored as [out_dim, in_dim]
//!   out: [M, N] row-major (f32) — overwritten (NOT accumulated)
//!
//! This arises in linear forward passes:
//!   output[batch, out] = input[batch, in] @ weight[out, in]^T
//!
//! Both A rows and B rows are sequential in memory, enabling efficient SIMD
//! dot products with no gather or transpose overhead.
//!
//! Parallelization:
//!   M=1:    over N columns (4-col unrolling shares A row load)
//!   M<64:   tiled row×column (32-col tiles keep B in L1)
//!   M>=64:  over M rows

const std = @import("std");
const parallel = @import("../../system/parallel.zig");
const tensor_mod = @import("../../tensor.zig");
const simd = @import("simd/arch/root.zig");
const matmul_primitives = @import("matmul_primitives.zig");

const Tensor = tensor_mod.Tensor;
pub const MatmulScratch = matmul_primitives.MatmulScratch;

const VEC: usize = simd.f32_vec_len;
const VecF32 = @Vector(VEC, f32);

/// Batch size threshold: above this, parallelize over rows only.
/// Below this, use tiled row+column parallelization for better load balance.
const TILE_THRESHOLD: usize = 64;

/// Column tile size for small-batch tiled path.
/// With K=256 (typical hidden dim): 32 cols × 256 × 4 bytes = 32KB fits L1.
const COL_TILE: usize = 32;

/// Matmul with transposed B: C = A @ B^T.
///
/// Shapes (all f32, 2D):
///   a:   [M, K]  — input activations
///   b:   [N, K]  — weights stored row-major (each row is one output neuron)
///   out: [M, N]  — overwritten (not accumulated)
///
/// K is the reduction dimension (shared second axis of A and B).
pub fn matmulF32TransB(a: *const Tensor, b: *const Tensor, out: *Tensor, scratch: *MatmulScratch) void {
    _ = scratch;
    std.debug.assert(a.dtype == .f32 and b.dtype == .f32 and out.dtype == .f32);
    std.debug.assert(a.n_dims == 2 and b.n_dims == 2 and out.n_dims == 2);

    const m_rows: usize = @intCast(a.shape[0]);
    const k_dim: usize = @intCast(a.shape[1]);
    const n_cols: usize = @intCast(b.shape[0]);

    std.debug.assert(b.shape[1] == a.shape[1]); // K must match
    std.debug.assert(out.shape[0] == a.shape[0]); // M
    std.debug.assert(out.shape[1] == b.shape[0]); // N

    if (m_rows == 0 or n_cols == 0 or k_dim == 0) return;

    const a_data = a.asSlice(f32);
    const b_data = b.asSlice(f32);
    const c_data = out.asSlice(f32);

    const Ctx = struct {
        a: []const f32,
        b: []const f32,
        c: []f32,
        m_rows: usize,
        n_cols: usize,
        k_dim: usize,
    };
    var ctx = Ctx{
        .a = a_data,
        .b = b_data,
        .c = c_data,
        .m_rows = m_rows,
        .n_cols = n_cols,
        .k_dim = k_dim,
    };

    if (m_rows == 1) {
        // Decode: parallelize over output columns.
        const decode_task = struct {
            fn run(start: usize, end: usize, task_ctx: *Ctx) void {
                dotCols(task_ctx.b, task_ctx.a[0..task_ctx.k_dim], task_ctx.c, task_ctx.k_dim, start, end);
            }
        }.run;
        parallel.global().parallelFor(n_cols, decode_task, &ctx);
    } else if (m_rows >= TILE_THRESHOLD) {
        // Large batch: parallelize over rows.
        const row_task = struct {
            fn run(start: usize, end: usize, task_ctx: *Ctx) void {
                const k = task_ctx.k_dim;
                const n = task_ctx.n_cols;
                for (start..end) |row| {
                    dotCols(task_ctx.b, task_ctx.a[row * k ..][0..k], task_ctx.c[row * n ..], k, 0, n);
                }
            }
        }.run;
        parallel.global().parallelFor(m_rows, row_task, &ctx);
    } else {
        // Small batch: tile over rows × columns.
        const tiles_per_row = (n_cols + COL_TILE - 1) / COL_TILE;
        const total_tiles = m_rows * tiles_per_row;

        const TileCtx = struct {
            a: []const f32,
            b: []const f32,
            c: []f32,
            n_cols: usize,
            k_dim: usize,
            tiles_per_row: usize,
        };
        var tile_ctx = TileCtx{
            .a = a_data,
            .b = b_data,
            .c = c_data,
            .n_cols = n_cols,
            .k_dim = k_dim,
            .tiles_per_row = tiles_per_row,
        };

        const tiled_task = struct {
            fn run(start: usize, end: usize, task_ctx: *TileCtx) void {
                const k = task_ctx.k_dim;
                const n = task_ctx.n_cols;

                for (start..end) |tile_idx| {
                    const row = tile_idx / task_ctx.tiles_per_row;
                    const col_tile = tile_idx % task_ctx.tiles_per_row;
                    const col_start = col_tile * COL_TILE;
                    const col_end = @min(col_start + COL_TILE, n);

                    const a_row = task_ctx.a[row * k ..][0..k];
                    const out_row = task_ctx.c[row * n ..];

                    dotCols(task_ctx.b, a_row, out_row, k, col_start, col_end);
                }
            }
        }.run;
        parallel.global().parallelFor(total_tiles, tiled_task, &tile_ctx);
    }
}

/// Inner loop: compute dot products for a range of output columns.
/// Shared across decode, row, and tiled paths to avoid code duplication.
///
/// 4-column unrolling shares the A row load across 4 B rows for better ILP.
/// Both A and B accesses are sequential (cache-friendly).
fn dotCols(b_data: []const f32, a_row: []const f32, out_row: []f32, k: usize, col_start: usize, col_end: usize) void {
    var col = col_start;

    // 4-column unrolling: share A row load across 4 B rows
    while (col + 4 <= col_end) : (col += 4) {
        const b0 = b_data[col * k ..][0..k];
        const b1 = b_data[(col + 1) * k ..][0..k];
        const b2 = b_data[(col + 2) * k ..][0..k];
        const b3 = b_data[(col + 3) * k ..][0..k];

        var acc0: VecF32 = @splat(0);
        var acc1: VecF32 = @splat(0);
        var acc2: VecF32 = @splat(0);
        var acc3: VecF32 = @splat(0);

        var ki: usize = 0;
        while (ki + VEC <= k) : (ki += VEC) {
            const a_vec: VecF32 = a_row[ki..][0..VEC].*;
            acc0 = @mulAdd(VecF32, a_vec, @as(VecF32, b0[ki..][0..VEC].*), acc0);
            acc1 = @mulAdd(VecF32, a_vec, @as(VecF32, b1[ki..][0..VEC].*), acc1);
            acc2 = @mulAdd(VecF32, a_vec, @as(VecF32, b2[ki..][0..VEC].*), acc2);
            acc3 = @mulAdd(VecF32, a_vec, @as(VecF32, b3[ki..][0..VEC].*), acc3);
        }

        var s0 = @reduce(.Add, acc0);
        var s1 = @reduce(.Add, acc1);
        var s2 = @reduce(.Add, acc2);
        var s3 = @reduce(.Add, acc3);

        while (ki < k) : (ki += 1) {
            const av = a_row[ki];
            s0 += av * b0[ki];
            s1 += av * b1[ki];
            s2 += av * b2[ki];
            s3 += av * b3[ki];
        }

        out_row[col] = s0;
        out_row[col + 1] = s1;
        out_row[col + 2] = s2;
        out_row[col + 3] = s3;
    }

    // Remaining 1-3 columns
    while (col < col_end) : (col += 1) {
        const b_row = b_data[col * k ..][0..k];
        var acc: VecF32 = @splat(0);
        var ki: usize = 0;
        while (ki + VEC <= k) : (ki += VEC) {
            const a_vec: VecF32 = a_row[ki..][0..VEC].*;
            acc = @mulAdd(VecF32, a_vec, @as(VecF32, b_row[ki..][0..VEC].*), acc);
        }
        var sum = @reduce(.Add, acc);
        while (ki < k) : (ki += 1) {
            sum += a_row[ki] * b_row[ki];
        }
        out_row[col] = sum;
    }
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "matmulF32TransB basic: identity weight selects input" {
    // A = [[1, 2, 3], [4, 5, 6]]  M=2, K=3
    // B = [[1, 0, 0], [0, 1, 0]]  N=2, K=3 (identity-like)
    // C = A @ B^T = [[1, 2], [4, 5]]
    var a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b_data = [_]f32{ 1, 0, 0, 0, 1, 0 };
    var c_data: [4]f32 = undefined;

    var a = Tensor.view2DSlice(&a_data, 2, 3);
    var b = Tensor.view2DSlice(&b_data, 2, 3);
    var out = Tensor.view2DSlice(&c_data, 2, 2);

    var scratch = try MatmulScratch.init(testing.allocator);
    defer scratch.deinit();

    matmulF32TransB(&a, &b, &out, &scratch);

    try testing.expectApproxEqAbs(@as(f32, 1.0), c_data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.0), c_data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 4.0), c_data[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 5.0), c_data[3], 1e-5);
}

test "matmulF32TransB non-square" {
    // A = [[1, 2]]  M=1, K=2
    // B = [[3, 4], [5, 6], [7, 8]]  N=3, K=2
    // C = A @ B^T = [[1*3+2*4, 1*5+2*6, 1*7+2*8]] = [[11, 17, 23]]
    var a_data = [_]f32{ 1, 2 };
    var b_data = [_]f32{ 3, 4, 5, 6, 7, 8 };
    var c_data: [3]f32 = undefined;

    var a = Tensor.view2DSlice(&a_data, 1, 2);
    var b = Tensor.view2DSlice(&b_data, 3, 2);
    var out = Tensor.view2DSlice(&c_data, 1, 3);

    var scratch = try MatmulScratch.init(testing.allocator);
    defer scratch.deinit();

    matmulF32TransB(&a, &b, &out, &scratch);

    try testing.expectApproxEqAbs(@as(f32, 11.0), c_data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 17.0), c_data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 23.0), c_data[2], 1e-5);
}

test "matmulF32TransB wide output exercises SIMD + 4-col unrolling" {
    // M=2, K=16 (exercises SIMD), N=9 (9 cols: 2 full 4-unrolls + 1 remainder)
    const M = 2;
    const K = 16;
    const N = 9;

    var a_data: [M * K]f32 = undefined;
    var b_data: [N * K]f32 = undefined;
    var c_data: [M * N]f32 = undefined;

    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    for (&a_data) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (&b_data) |*v| v.* = random.float(f32) * 2.0 - 1.0;

    var a = Tensor.view2DSlice(&a_data, M, K);
    var b = Tensor.view2DSlice(&b_data, N, K);
    var out = Tensor.view2DSlice(&c_data, M, N);

    var scratch = try MatmulScratch.init(testing.allocator);
    defer scratch.deinit();

    matmulF32TransB(&a, &b, &out, &scratch);

    // Verify against naive
    var expected: [M * N]f32 = [_]f32{0} ** (M * N);
    for (0..M) |m| {
        for (0..N) |n| {
            for (0..K) |k| {
                expected[m * N + n] += a_data[m * K + k] * b_data[n * K + k];
            }
        }
    }
    for (0..M * N) |i| {
        try testing.expectApproxEqAbs(expected[i], c_data[i], 1e-3);
    }
}

test "matmulF32TransB matches linearForward contract" {
    // Simulates: output[batch, out] = input[batch, in] @ weight[out, in]^T
    // Same test data as forward/linear.zig test
    // input: [2, 3], weight: [2, 3] (out_dim=2, in_dim=3)
    const input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var weight = [_]f32{
        1, 0, 0, // row 0: select dim 0
        0, 1, 0, // row 1: select dim 1
    };
    var output: [4]f32 = undefined;

    var a = Tensor.view2DSlice(@constCast(&input), 2, 3);
    var b = Tensor.view2DSlice(&weight, 2, 3);
    var out = Tensor.view2DSlice(&output, 2, 2);

    var scratch = try MatmulScratch.init(testing.allocator);
    defer scratch.deinit();

    matmulF32TransB(&a, &b, &out, &scratch);

    // Row 0: [1,2,3] @ [[1,0],[0,1],[0,0]] = [1, 2]
    try testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.0), output[1], 1e-5);
    // Row 1: [4,5,6] @ ... = [4, 5]
    try testing.expectApproxEqAbs(@as(f32, 4.0), output[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 5.0), output[3], 1e-5);
}

test "matmulF32TransB large batch exercises row parallelism" {
    // M=70 (> TILE_THRESHOLD=64), K=8, N=5
    const M = 70;
    const K = 8;
    const N = 5;

    var a_data: [M * K]f32 = undefined;
    var b_data: [N * K]f32 = undefined;
    var c_data: [M * N]f32 = undefined;

    var rng = std.Random.DefaultPrng.init(123);
    const random = rng.random();
    for (&a_data) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (&b_data) |*v| v.* = random.float(f32) * 2.0 - 1.0;

    var a = Tensor.view2DSlice(&a_data, M, K);
    var b = Tensor.view2DSlice(&b_data, N, K);
    var out = Tensor.view2DSlice(&c_data, M, N);

    var scratch = try MatmulScratch.init(testing.allocator);
    defer scratch.deinit();

    matmulF32TransB(&a, &b, &out, &scratch);

    // Verify against naive
    var expected: [M * N]f32 = [_]f32{0} ** (M * N);
    for (0..M) |m| {
        for (0..N) |n| {
            for (0..K) |k| {
                expected[m * N + n] += a_data[m * K + k] * b_data[n * K + k];
            }
        }
    }
    for (0..M * N) |i| {
        try testing.expectApproxEqAbs(expected[i], c_data[i], 1e-3);
    }
}
