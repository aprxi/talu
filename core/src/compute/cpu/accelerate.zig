//! Apple Accelerate framework integration for optimized BLAS operations.
//!
//! Uses cblas_sgemm for f32 matrix multiplication on Apple Silicon.
//! Falls back to pure Zig implementation on non-Apple platforms.

const std = @import("std");
const builtin = @import("builtin");

/// Whether Apple Accelerate is available
pub const available = builtin.os.tag == .macos;

// Apple Accelerate CBLAS bindings
const cblas = if (available) struct {
    // CBLAS enum values
    pub const CblasRowMajor: c_int = 101;
    pub const CblasNoTrans: c_int = 111;
    pub const CblasTrans: c_int = 112;

    // cblas_sgemm: C = alpha * A @ B + beta * C
    pub extern "Accelerate" fn cblas_sgemm(
        order: c_int, // CblasRowMajor
        transA: c_int, // CblasNoTrans or CblasTrans
        transB: c_int, // CblasNoTrans or CblasTrans
        M: c_int, // rows of A (and C)
        N: c_int, // cols of B (and C)
        K: c_int, // cols of A / rows of B
        alpha: f32,
        A: [*]const f32,
        lda: c_int, // leading dimension of A
        B: [*]const f32,
        ldb: c_int, // leading dimension of B
        beta: f32,
        C: [*]f32,
        ldc: c_int, // leading dimension of C
    ) void;
} else struct {};

/// Perform f32 matrix multiplication with accumulation.
/// C = alpha * A @ B + beta * C where A is [M x K], B is [K x N], C is [M x N]
/// All matrices are row-major.
pub fn sgemmScaled(
    a: []const f32,
    b: []const f32,
    c: []f32,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) void {
    if (comptime !available) {
        @compileError("Apple Accelerate not available on this platform");
    }

    cblas.cblas_sgemm(
        cblas.CblasRowMajor,
        cblas.CblasNoTrans,
        cblas.CblasNoTrans,
        @intCast(m),
        @intCast(n),
        @intCast(k),
        alpha,
        a.ptr,
        @intCast(k), // lda = K for row-major A
        b.ptr,
        @intCast(n), // ldb = N for row-major B
        beta,
        c.ptr,
        @intCast(n), // ldc = N for row-major C
    );
}

/// Perform f32 matrix multiplication with custom A stride.
/// C = alpha * A @ B + beta * C where A is [M x K] with leading dimension lda.
/// Useful when A has larger stride (e.g., accessing submatrix of larger matrix).
pub fn sgemmScaledStrided(
    a: []const f32,
    b: []const f32,
    c: []f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize, // Leading dimension of A (stride between rows)
    alpha: f32,
    beta: f32,
) void {
    if (comptime !available) {
        @compileError("Apple Accelerate not available on this platform");
    }

    cblas.cblas_sgemm(
        cblas.CblasRowMajor,
        cblas.CblasNoTrans,
        cblas.CblasNoTrans,
        @intCast(m),
        @intCast(n),
        @intCast(k),
        alpha,
        a.ptr,
        @intCast(lda), // Custom lda for strided access
        b.ptr,
        @intCast(n), // ldb = N for row-major B
        beta,
        c.ptr,
        @intCast(n), // ldc = N for row-major C
    );
}

/// Perform f32 matrix multiplication using Apple Accelerate.
/// C = A @ B where A is [M x K], B is [K x N], C is [M x N]
/// All matrices are row-major.
pub fn sgemm(
    a: []const f32,
    b: []const f32,
    c: []f32,
    m: usize,
    n: usize,
    k: usize,
) void {
    if (comptime !available) {
        @compileError("Apple Accelerate not available on this platform");
    }

    cblas.cblas_sgemm(
        cblas.CblasRowMajor,
        cblas.CblasNoTrans,
        cblas.CblasNoTrans,
        @intCast(m),
        @intCast(n),
        @intCast(k),
        1.0, // alpha
        a.ptr,
        @intCast(k), // lda = K for row-major A
        b.ptr,
        @intCast(n), // ldb = N for row-major B
        0.0, // beta (overwrite C)
        c.ptr,
        @intCast(n), // ldc = N for row-major C
    );
}

/// Perform f32 matrix multiplication with transposed B.
/// C = A @ B^T where A is [M x K], B is [N x K] (stored as N rows of K), C is [M x N]
pub fn sgemmTransB(
    a: []const f32,
    b: []const f32,
    c: []f32,
    m: usize,
    n: usize,
    k: usize,
) void {
    sgemmTransBScaled(a, b, c, m, n, k, 1.0);
}

/// Perform f32 matrix multiplication with transposed B and scaling.
/// C = alpha * A @ B^T where A is [M x K], B is [N x K], C is [M x N]
pub fn sgemmTransBScaled(
    a: []const f32,
    b: []const f32,
    c: []f32,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
) void {
    if (comptime !available) {
        @compileError("Apple Accelerate not available on this platform");
    }

    cblas.cblas_sgemm(
        cblas.CblasRowMajor,
        cblas.CblasNoTrans,
        cblas.CblasTrans, // B is transposed
        @intCast(m),
        @intCast(n),
        @intCast(k),
        alpha,
        a.ptr,
        @intCast(k), // lda = K for row-major A
        b.ptr,
        @intCast(k), // ldb = K for row-major B (before transpose)
        0.0, // beta (overwrite C)
        c.ptr,
        @intCast(n), // ldc = N for row-major C
    );
}

// =============================================================================
// Tests
// =============================================================================

test "sgemm basic multiply" {
    if (comptime !available) return;

    // A = [[1, 2], [3, 4]]  (2x2)
    // B = [[5, 6], [7, 8]]  (2x2)
    // C = A @ B = [[19, 22], [43, 50]]
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var c = [_]f32{ 0, 0, 0, 0 };

    sgemm(&a, &b, &c, 2, 2, 2);

    try std.testing.expectApproxEqAbs(@as(f32, 19), c[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 22), c[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 43), c[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 50), c[3], 1e-5);
}

test "sgemm identity" {
    if (comptime !available) return;

    // A = [[1, 0], [0, 1]]
    // B = [[3, 4], [5, 6]]
    // C = A @ B = B
    const a = [_]f32{ 1, 0, 0, 1 };
    const b = [_]f32{ 3, 4, 5, 6 };
    var c = [_]f32{ 0, 0, 0, 0 };

    sgemm(&a, &b, &c, 2, 2, 2);

    try std.testing.expectApproxEqAbs(@as(f32, 3), c[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4), c[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5), c[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 6), c[3], 1e-5);
}

test "sgemmTransB basic" {
    if (comptime !available) return;

    // A = [[1, 2]]  (1x2, M=1, K=2)
    // B = [[3, 4], [5, 6]]  (2x2, N=2, K=2, will be transposed)
    // B^T = [[3, 5], [4, 6]]  (K=2 x N=2)
    // C = A @ B^T = [[1*3 + 2*4, 1*5 + 2*6]] = [[11, 17]]
    const a = [_]f32{ 1, 2 };
    const b = [_]f32{ 3, 4, 5, 6 };
    var c = [_]f32{ 0, 0 };

    sgemmTransB(&a, &b, &c, 1, 2, 2);

    try std.testing.expectApproxEqAbs(@as(f32, 11), c[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 17), c[1], 1e-5);
}
