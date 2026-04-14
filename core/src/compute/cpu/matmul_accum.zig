//! Fused transpose-accumulate matmul: C += A^T @ B.
//!
//! Computes C += A^T @ B where:
//!   A: [K, M] row-major (f32)
//!   B: [K, N] row-major (f32)
//!   C: [M, N] row-major (f32) — accumulated, NOT overwritten
//!
//! This arises in gradient computations:
//!   grad_weight[out, in] += grad_output[batch, out]^T @ input[batch, in]
//!
//! Fuses transpose, matmul, and accumulate into a single SIMD pass.
//! Multi-row register blocking (ROW_TILE rows share each B load) improves
//! arithmetic intensity over separate transpose + matmul + accumulate.

const std = @import("std");
const parallel = @import("compute_pkg").parallel;
const tensor_mod = @import("tensor_pkg");
const simd = @import("simd/arch/root.zig");
const matmul_primitives = @import("matmul_primitives.zig");

const Tensor = tensor_mod.Tensor;
pub const MatmulScratch = matmul_primitives.MatmulScratch;

/// Output rows processed per register tile.
/// Sharing each B vector load across ROW_TILE output rows improves
/// arithmetic intensity by ROW_TILE×.
/// 8 uses 9 of 16 ymm registers (8 accumulators + 1 B vector), leaving
/// headroom for g broadcasts and compiler temporaries.
const ROW_TILE: usize = 8;

/// K-loop tile size for cache blocking.
/// Tiling the reduction dimension keeps the B chunk (K_TILE × N × 4 bytes)
/// in L2 cache across column iterations. With K_TILE=256 and N≤1024:
/// chunk ≤ 1MB (fits L2 on most modern CPUs).
const K_TILE: usize = 256;

const VEC: usize = simd.f32_vec_len;
const VecF32 = @Vector(VEC, f32);

/// Fused transpose-accumulate matmul: C += A^T @ B.
///
/// Shapes (all f32, 2D):
///   a: [K, M]  — logically transposed to [M, K]
///   b: [K, N]
///   c: [M, N]  — accumulated (existing values preserved)
///
/// K is the reduction dimension (shared first axis of A and B).
/// Parallelized over M (output rows) with ROW_TILE register blocking.
pub fn matmulTransposeAccumF32(a: *const Tensor, b: *const Tensor, c: *Tensor, scratch: *MatmulScratch) void {
    _ = scratch;
    std.debug.assert(a.dtype == .f32 and b.dtype == .f32 and c.dtype == .f32);
    std.debug.assert(a.n_dims == 2 and b.n_dims == 2 and c.n_dims == 2);

    const k_dim: usize = @intCast(a.shape[0]);
    const m_rows: usize = @intCast(a.shape[1]);
    const n_cols: usize = @intCast(b.shape[1]);

    std.debug.assert(b.shape[0] == a.shape[0]); // K must match
    std.debug.assert(c.shape[0] == a.shape[1]); // M
    std.debug.assert(c.shape[1] == b.shape[1]); // N

    if (m_rows == 0 or n_cols == 0 or k_dim == 0) return;

    const a_data = a.asSlice(f32);
    const b_data = b.asSlice(f32);
    const c_data = c.asSlice(f32);

    const Ctx = struct {
        a: []const f32,
        b: []const f32,
        c: []f32,
        k_dim: usize,
        m_rows: usize,
        n_cols: usize,
    };
    var ctx = Ctx{
        .a = a_data,
        .b = b_data,
        .c = c_data,
        .k_dim = k_dim,
        .m_rows = m_rows,
        .n_cols = n_cols,
    };

    const task = struct {
        fn run(start: usize, end: usize, task_ctx: *Ctx) void {
            @setFloatMode(.optimized);
            const k = task_ctx.k_dim;
            const m = task_ctx.m_rows;
            const n = task_ctx.n_cols;
            const vec_end = n - (n % VEC);

            var row = start;

            // ── ROW_TILE rows at a time: 4 accumulators share each B load ──
            while (row + ROW_TILE <= end) : (row += ROW_TILE) {
                // K-loop tiling: process K_TILE elements of the reduction
                // dimension at a time, iterating all columns for each chunk.
                // Keeps B chunk (K_TILE × N × 4 bytes) in L2 across column tiles.
                var k_start: usize = 0;
                while (k_start < k) : (k_start += K_TILE) {
                    const k_end = @min(k_start + K_TILE, k);

                    // Vectorized columns
                    var col: usize = 0;
                    while (col < vec_end) : (col += VEC) {
                        var acc: [ROW_TILE]VecF32 = undefined;
                        inline for (0..ROW_TILE) |r| {
                            acc[r] = task_ctx.c[(row + r) * n + col ..][0..VEC].*;
                        }

                        for (k_start..k_end) |bi| {
                            const b_vec: VecF32 = task_ctx.b[bi * n + col ..][0..VEC].*;
                            inline for (0..ROW_TILE) |r| {
                                const g: VecF32 = @splat(task_ctx.a[bi * m + (row + r)]);
                                acc[r] = @mulAdd(VecF32, g, b_vec, acc[r]);
                            }
                        }

                        inline for (0..ROW_TILE) |r| {
                            task_ctx.c[(row + r) * n + col ..][0..VEC].* = acc[r];
                        }
                    }

                    // Scalar tail
                    if (col < n) {
                        inline for (0..ROW_TILE) |r| {
                            for (col..n) |c_idx| {
                                var sum = task_ctx.c[(row + r) * n + c_idx];
                                for (k_start..k_end) |bi| {
                                    sum = @mulAdd(f32, task_ctx.a[bi * m + (row + r)], task_ctx.b[bi * n + c_idx], sum);
                                }
                                task_ctx.c[(row + r) * n + c_idx] = sum;
                            }
                        }
                    }
                }
            }

            // ── Remaining rows (< ROW_TILE) ──
            while (row < end) : (row += 1) {
                var k_start: usize = 0;
                while (k_start < k) : (k_start += K_TILE) {
                    const k_end = @min(k_start + K_TILE, k);

                    var col: usize = 0;
                    while (col < vec_end) : (col += VEC) {
                        var acc: VecF32 = task_ctx.c[row * n + col ..][0..VEC].*;
                        for (k_start..k_end) |bi| {
                            const g: VecF32 = @splat(task_ctx.a[bi * m + row]);
                            const b_vec: VecF32 = task_ctx.b[bi * n + col ..][0..VEC].*;
                            acc = @mulAdd(VecF32, g, b_vec, acc);
                        }
                        task_ctx.c[row * n + col ..][0..VEC].* = acc;
                    }
                    for (col..n) |c_idx| {
                        var sum = task_ctx.c[row * n + c_idx];
                        for (k_start..k_end) |bi| {
                            sum = @mulAdd(f32, task_ctx.a[bi * m + row], task_ctx.b[bi * n + c_idx], sum);
                        }
                        task_ctx.c[row * n + c_idx] = sum;
                    }
                }
            }
        }
    }.run;

    if (m_rows >= ROW_TILE) {
        parallel.global().parallelFor(m_rows, task, &ctx);
    } else {
        task(0, m_rows, &ctx);
    }
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "matmulTransposeAccumF32 basic 2x2" {
    // A = [[1, 2], [3, 4]]  K=2, M=2
    // A^T = [[1, 3], [2, 4]]
    // B = [[5, 6], [7, 8]]  K=2, N=2
    // A^T @ B = [[1*5+3*7, 1*6+3*8], [2*5+4*7, 2*6+4*8]]
    //         = [[26, 30], [38, 44]]
    var a_data = [_]f32{ 1, 2, 3, 4 };
    var b_data = [_]f32{ 5, 6, 7, 8 };
    var c_data = [_]f32{ 0, 0, 0, 0 };

    var a = Tensor.view2DSlice(&a_data, 2, 2);
    var b = Tensor.view2DSlice(&b_data, 2, 2);
    var c = Tensor.view2DSlice(&c_data, 2, 2);

    var scratch = try MatmulScratch.init(testing.allocator);
    defer scratch.deinit();

    matmulTransposeAccumF32(&a, &b, &c, &scratch);

    try testing.expectApproxEqAbs(@as(f32, 26.0), c_data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 30.0), c_data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 38.0), c_data[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 44.0), c_data[3], 1e-5);
}

test "matmulTransposeAccumF32 non-square" {
    // A = [[1, 2, 3], [4, 5, 6]]  K=2, M=3
    // A^T = [[1, 4], [2, 5], [3, 6]]
    // B = [[7, 8], [9, 10]]  K=2, N=2
    // A^T @ B = [[1*7+4*9, 1*8+4*10], [2*7+5*9, 2*8+5*10], [3*7+6*9, 3*8+6*10]]
    //         = [[43, 48], [59, 66], [75, 84]]
    var a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b_data = [_]f32{ 7, 8, 9, 10 };
    var c_data = [_]f32{ 0, 0, 0, 0, 0, 0 };

    var a = Tensor.view2DSlice(&a_data, 2, 3);
    var b = Tensor.view2DSlice(&b_data, 2, 2);
    var c = Tensor.view2DSlice(&c_data, 3, 2);

    var scratch = try MatmulScratch.init(testing.allocator);
    defer scratch.deinit();

    matmulTransposeAccumF32(&a, &b, &c, &scratch);

    try testing.expectApproxEqAbs(@as(f32, 43.0), c_data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 48.0), c_data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 59.0), c_data[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 66.0), c_data[3], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 75.0), c_data[4], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 84.0), c_data[5], 1e-5);
}

test "matmulTransposeAccumF32 accumulates" {
    // Start C = [[10, 20], [30, 40]], then add A^T @ B
    // Same A, B as basic test → A^T @ B = [[26, 30], [38, 44]]
    // Result: [[36, 50], [68, 84]]
    var a_data = [_]f32{ 1, 2, 3, 4 };
    var b_data = [_]f32{ 5, 6, 7, 8 };
    var c_data = [_]f32{ 10, 20, 30, 40 };

    var a = Tensor.view2DSlice(&a_data, 2, 2);
    var b = Tensor.view2DSlice(&b_data, 2, 2);
    var c = Tensor.view2DSlice(&c_data, 2, 2);

    var scratch = try MatmulScratch.init(testing.allocator);
    defer scratch.deinit();

    matmulTransposeAccumF32(&a, &b, &c, &scratch);

    try testing.expectApproxEqAbs(@as(f32, 36.0), c_data[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 50.0), c_data[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 68.0), c_data[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 84.0), c_data[3], 1e-5);
}

test "matmulTransposeAccumF32 wide columns exercises SIMD path" {
    // M=2, K=3, N=16 — forces at least 2 full SIMD iterations (VEC=8)
    const K = 3;
    const M = 2;
    const N = 16;

    var a_data: [K * M]f32 = undefined;
    var b_data: [K * N]f32 = undefined;
    var c_data: [M * N]f32 = [_]f32{0} ** (M * N);

    // Fill with simple pattern: a[k][m] = k*M + m + 1, b[k][n] = k*N + n + 1
    for (0..K) |k| {
        for (0..M) |m| a_data[k * M + m] = @floatFromInt(k * M + m + 1);
        for (0..N) |n| b_data[k * N + n] = @floatFromInt(k * N + n + 1);
    }

    var a = Tensor.view2DSlice(&a_data, K, M);
    var b = Tensor.view2DSlice(&b_data, K, N);
    var c = Tensor.view2DSlice(&c_data, M, N);

    var scratch = try MatmulScratch.init(testing.allocator);
    defer scratch.deinit();

    matmulTransposeAccumF32(&a, &b, &c, &scratch);

    // Verify against naive computation
    var expected: [M * N]f32 = [_]f32{0} ** (M * N);
    for (0..K) |k| {
        for (0..M) |m| {
            for (0..N) |n| {
                expected[m * N + n] += a_data[k * M + m] * b_data[k * N + n];
            }
        }
    }
    for (0..M * N) |i| {
        try testing.expectApproxEqAbs(expected[i], c_data[i], 1e-3);
    }
}

test "matmulTransposeAccumF32 many rows exercises ROW_TILE and parallelism" {
    // M=17 (not divisible by ROW_TILE=4), K=4, N=9 (not divisible by VEC)
    const K = 4;
    const M = 17;
    const N = 9;

    var a_data: [K * M]f32 = undefined;
    var b_data: [K * N]f32 = undefined;
    var c_data: [M * N]f32 = [_]f32{0} ** (M * N);

    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    for (&a_data) |*v| v.* = random.float(f32) * 2.0 - 1.0;
    for (&b_data) |*v| v.* = random.float(f32) * 2.0 - 1.0;

    var a = Tensor.view2DSlice(&a_data, K, M);
    var b = Tensor.view2DSlice(&b_data, K, N);
    var c = Tensor.view2DSlice(&c_data, M, N);

    var scratch = try MatmulScratch.init(testing.allocator);
    defer scratch.deinit();

    matmulTransposeAccumF32(&a, &b, &c, &scratch);

    // Verify against naive
    var expected: [M * N]f32 = [_]f32{0} ** (M * N);
    for (0..K) |k| {
        for (0..M) |m| {
            for (0..N) |n| {
                expected[m * N + n] += a_data[k * M + m] * b_data[k * N + n];
            }
        }
    }
    for (0..M * N) |i| {
        try testing.expectApproxEqAbs(expected[i], c_data[i], 1e-3);
    }
}

test "matmulTransposeAccumF32 matches gradWeight contract" {
    // Simulates: grad_weight[out, in] += grad_output[batch, out]^T @ input[batch, in]
    // batch=2, out_dim=2, in_dim=3
    // grad_output = [[1, 0], [0, 1]] (A: K=2, M=2)
    // input       = [[1, 2, 3], [4, 5, 6]] (B: K=2, N=3)
    // A^T @ B = [[1,0],[0,1]]^T @ [[1,2,3],[4,5,6]]
    //         = [[1,2,3],[4,5,6]]
    var grad_out = [_]f32{ 1, 0, 0, 1 };
    var input = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var grad_w = [_]f32{ 0, 0, 0, 0, 0, 0 };

    var a = Tensor.view2DSlice(&grad_out, 2, 2);
    var b = Tensor.view2DSlice(&input, 2, 3);
    var c = Tensor.view2DSlice(&grad_w, 2, 3);

    var scratch = try MatmulScratch.init(testing.allocator);
    defer scratch.deinit();

    matmulTransposeAccumF32(&a, &b, &c, &scratch);

    // Same expected values as backward/linear.zig "gradWeight accumulates" test
    try testing.expectApproxEqAbs(@as(f32, 1.0), grad_w[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.0), grad_w[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 3.0), grad_w[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 4.0), grad_w[3], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 5.0), grad_w[4], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 6.0), grad_w[5], 1e-5);
}
