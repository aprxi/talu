//! Prefill-specific GEMM kernels (AVX2-optimized)
//!
//! Based on expert optimization advice:
//! 1. 4x2 microkernel fits AVX2's 16 registers
//! 2. Deferred scaling: accumulate Sum(W*X) and Sum(X) separately
//! 3. @reduce only at group boundary, not in hot loop

const std = @import("std");
const parallel = @import("../../system/parallel.zig");
const matmul = @import("matmul_primitives.zig");
const grouped_affine_quant = @import("quant/grouped_affine_quant.zig");
const dtype_mod = @import("../../dtype.zig");
const DType = dtype_mod.DType;
const fp16ToF32 = dtype_mod.fp16ToF32;
const bf16ToF32 = dtype_mod.bf16ToF32;

// Import shared grouped-affine helpers
const extractNibbles = grouped_affine_quant.extractNibbles;
const extract32NibblesToFloat = grouped_affine_quant.extract32NibblesToFloat;
const extractBytes = grouped_affine_quant.extractBytes;
const scaleBiasToF32 = grouped_affine_quant.scaleBiasToF32;

// =============================================================================
// 4-bit Prefill Entry Point
// =============================================================================

/// f32-based prefill kernel with 4x2 microkernel
pub fn matmulGaffineU4Prefill(
    a_data: []const f32,
    m_rows: usize,
    k_dim: usize,
    packed_vals: []align(1) const u32,
    scales: []align(1) const u16,
    biases: []align(1) const u16,
    scales_dtype: DType,
    n_cols: usize,
    group: usize,
    out_data: []f32,
) void {
    const k_div_8 = k_dim / 8;
    const k_div_group = k_dim / group;
    const group_u32 = group / 8;

    const MatmulU4PrefillCtx = struct {
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
        k_div_8: usize,
        k_div_group: usize,
        group_u32: usize,
    };

    var context = MatmulU4PrefillCtx{
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
        .k_div_8 = k_div_8,
        .k_div_group = k_div_group,
        .group_u32 = group_u32,
    };

    // Parallelize over column pairs
    const num_col_pairs = n_cols / 2;

    const col_pair_task = struct {
        fn runColPairs(start: usize, end: usize, task_ctx: *MatmulU4PrefillCtx) void {
            @setFloatMode(.optimized);

            var col_pair_idx = start;
            while (col_pair_idx < end) : (col_pair_idx += 1) {
                const col_idx = col_pair_idx * 2;

                // Process rows in groups of 4
                var row_idx: usize = 0;
                while (row_idx + 3 < task_ctx.m_rows) : (row_idx += 4) {
                    kernel4x2(task_ctx, row_idx, col_idx);
                }
                // Remainder rows
                while (row_idx < task_ctx.m_rows) : (row_idx += 1) {
                    kernel1x2(task_ctx, row_idx, col_idx);
                }
            }
        }
    }.runColPairs;

    parallel.global().parallelFor(num_col_pairs, col_pair_task, &context);

    // Handle odd column at end
    if (n_cols % 2 == 1) {
        const col_idx = n_cols - 1;
        for (0..m_rows) |row_idx| {
            kernel1x1(&context, row_idx, col_idx);
        }
    }
}

// =============================================================================
// 4x2 Microkernel - The fast path
// =============================================================================
// Register budget (AVX2 has 16 YMM registers):
// - 8 accumulators (4 rows × 2 cols)
// - 4 activation sum accumulators
// - 2 weight vectors
// - 2 activation vectors
// Total: 16 registers (perfect fit!)

fn kernel4x2(context: anytype, row_base_idx: usize, col_idx: usize) void {
    @setFloatMode(.optimized);

    const k_dim = context.k_dim;
    const n_dim = context.n_cols;
    const group = context.group;
    const k_div_group = context.k_div_group;
    const k_div_8 = context.k_div_8;
    const group_u32 = context.group_u32;

    // Weight pointers for 2 columns
    const w0_base = context.packed_b.ptr + col_idx * k_div_8;
    const w1_base = context.packed_b.ptr + (col_idx + 1) * k_div_8;

    // Activation pointers for 4 rows
    const a0_base = context.a.ptr + row_base_idx * k_dim;
    const a1_base = context.a.ptr + (row_base_idx + 1) * k_dim;
    const a2_base = context.a.ptr + (row_base_idx + 2) * k_dim;
    const a3_base = context.a.ptr + (row_base_idx + 3) * k_dim;

    // Scale/bias pointers
    const scale0_base = context.scales.ptr + col_idx * k_div_group;
    const scale1_base = context.scales.ptr + (col_idx + 1) * k_div_group;
    const bias0_base = context.biases.ptr + col_idx * k_div_group;
    const bias1_base = context.biases.ptr + (col_idx + 1) * k_div_group;

    // Vector accumulators for scale*wx and bias*xs (defer reduction to end!)
    // This is the key optimization from the decode kernel
    var acc00: @Vector(8, f32) = @splat(0);
    var acc01: @Vector(8, f32) = @splat(0);
    var acc10: @Vector(8, f32) = @splat(0);
    var acc11: @Vector(8, f32) = @splat(0);
    var acc20: @Vector(8, f32) = @splat(0);
    var acc21: @Vector(8, f32) = @splat(0);
    var acc30: @Vector(8, f32) = @splat(0);
    var acc31: @Vector(8, f32) = @splat(0);
    // Separate accumulators for bias*xs terms
    var bias_acc0: @Vector(8, f32) = @splat(0);
    var bias_acc1: @Vector(8, f32) = @splat(0);
    var bias_acc2: @Vector(8, f32) = @splat(0);
    var bias_acc3: @Vector(8, f32) = @splat(0);

    // Process each quantization group
    var group_idx: usize = 0;
    while (group_idx < k_div_group) : (group_idx += 1) {
        const scale0 = scaleBiasToF32(context.scales_dtype, scale0_base[group_idx]);
        const scale1 = scaleBiasToF32(context.scales_dtype, scale1_base[group_idx]);
        const bias0 = scaleBiasToF32(context.scales_dtype, bias0_base[group_idx]);
        const bias1 = scaleBiasToF32(context.scales_dtype, bias1_base[group_idx]);

        const w0_ptr = w0_base + group_idx * group_u32;
        const w1_ptr = w1_base + group_idx * group_u32;
        const a0_ptr = a0_base + group_idx * group;
        const a1_ptr = a1_base + group_idx * group;
        const a2_ptr = a2_base + group_idx * group;
        const a3_ptr = a3_base + group_idx * group;

        // Per-group vector accumulators for deferred scaling
        // wx = Sum(Weight * Activation), xs = Sum(Activation)
        var wx00: @Vector(8, f32) = @splat(0);
        var wx01: @Vector(8, f32) = @splat(0);
        var wx10: @Vector(8, f32) = @splat(0);
        var wx11: @Vector(8, f32) = @splat(0);
        var wx20: @Vector(8, f32) = @splat(0);
        var wx21: @Vector(8, f32) = @splat(0);
        var wx30: @Vector(8, f32) = @splat(0);
        var wx31: @Vector(8, f32) = @splat(0);

        var xs0: @Vector(8, f32) = @splat(0);
        var xs1: @Vector(8, f32) = @splat(0);
        var xs2: @Vector(8, f32) = @splat(0);
        var xs3: @Vector(8, f32) = @splat(0);

        // Hot loop: process 32 weights (4 u32s) per iteration
        // NO @reduce here - pure FMA operations
        var pack_idx: usize = 0;
        while (pack_idx + 3 < group_u32) : (pack_idx += 4) {
            // Prefetch next iteration's data
            @prefetch(@as([*]const u8, @ptrCast(w0_ptr + pack_idx + 16)), .{ .locality = 3 });
            @prefetch(@as([*]const u8, @ptrCast(w1_ptr + pack_idx + 16)), .{ .locality = 3 });
            @prefetch(@as([*]const u8, @ptrCast(a0_ptr + (pack_idx + 4) * 8)), .{ .locality = 3 });

            // Load 32 nibbles for each column
            const nibs0 = extract32NibblesToFloat(w0_ptr + pack_idx);
            const nibs1 = extract32NibblesToFloat(w1_ptr + pack_idx);

            // Load activations (same for both columns, different for each row)
            const x0_0: @Vector(8, f32) = (a0_ptr + pack_idx * 8)[0..8].*;
            const x0_1: @Vector(8, f32) = (a0_ptr + (pack_idx + 1) * 8)[0..8].*;
            const x0_2: @Vector(8, f32) = (a0_ptr + (pack_idx + 2) * 8)[0..8].*;
            const x0_3: @Vector(8, f32) = (a0_ptr + (pack_idx + 3) * 8)[0..8].*;

            const x1_0: @Vector(8, f32) = (a1_ptr + pack_idx * 8)[0..8].*;
            const x1_1: @Vector(8, f32) = (a1_ptr + (pack_idx + 1) * 8)[0..8].*;
            const x1_2: @Vector(8, f32) = (a1_ptr + (pack_idx + 2) * 8)[0..8].*;
            const x1_3: @Vector(8, f32) = (a1_ptr + (pack_idx + 3) * 8)[0..8].*;

            const x2_0: @Vector(8, f32) = (a2_ptr + pack_idx * 8)[0..8].*;
            const x2_1: @Vector(8, f32) = (a2_ptr + (pack_idx + 1) * 8)[0..8].*;
            const x2_2: @Vector(8, f32) = (a2_ptr + (pack_idx + 2) * 8)[0..8].*;
            const x2_3: @Vector(8, f32) = (a2_ptr + (pack_idx + 3) * 8)[0..8].*;

            const x3_0: @Vector(8, f32) = (a3_ptr + pack_idx * 8)[0..8].*;
            const x3_1: @Vector(8, f32) = (a3_ptr + (pack_idx + 1) * 8)[0..8].*;
            const x3_2: @Vector(8, f32) = (a3_ptr + (pack_idx + 2) * 8)[0..8].*;
            const x3_3: @Vector(8, f32) = (a3_ptr + (pack_idx + 3) * 8)[0..8].*;

            // Row 0: accumulate W*X (CORRECT ORDER: weight first!)
            wx00 = @mulAdd(@Vector(8, f32), nibs0.n0, x0_0, wx00);
            wx00 = @mulAdd(@Vector(8, f32), nibs0.n1, x0_1, wx00);
            wx00 = @mulAdd(@Vector(8, f32), nibs0.n2, x0_2, wx00);
            wx00 = @mulAdd(@Vector(8, f32), nibs0.n3, x0_3, wx00);

            wx01 = @mulAdd(@Vector(8, f32), nibs1.n0, x0_0, wx01);
            wx01 = @mulAdd(@Vector(8, f32), nibs1.n1, x0_1, wx01);
            wx01 = @mulAdd(@Vector(8, f32), nibs1.n2, x0_2, wx01);
            wx01 = @mulAdd(@Vector(8, f32), nibs1.n3, x0_3, wx01);

            xs0 += x0_0 + x0_1 + x0_2 + x0_3;

            // Row 1
            wx10 = @mulAdd(@Vector(8, f32), nibs0.n0, x1_0, wx10);
            wx10 = @mulAdd(@Vector(8, f32), nibs0.n1, x1_1, wx10);
            wx10 = @mulAdd(@Vector(8, f32), nibs0.n2, x1_2, wx10);
            wx10 = @mulAdd(@Vector(8, f32), nibs0.n3, x1_3, wx10);

            wx11 = @mulAdd(@Vector(8, f32), nibs1.n0, x1_0, wx11);
            wx11 = @mulAdd(@Vector(8, f32), nibs1.n1, x1_1, wx11);
            wx11 = @mulAdd(@Vector(8, f32), nibs1.n2, x1_2, wx11);
            wx11 = @mulAdd(@Vector(8, f32), nibs1.n3, x1_3, wx11);

            xs1 += x1_0 + x1_1 + x1_2 + x1_3;

            // Row 2
            wx20 = @mulAdd(@Vector(8, f32), nibs0.n0, x2_0, wx20);
            wx20 = @mulAdd(@Vector(8, f32), nibs0.n1, x2_1, wx20);
            wx20 = @mulAdd(@Vector(8, f32), nibs0.n2, x2_2, wx20);
            wx20 = @mulAdd(@Vector(8, f32), nibs0.n3, x2_3, wx20);

            wx21 = @mulAdd(@Vector(8, f32), nibs1.n0, x2_0, wx21);
            wx21 = @mulAdd(@Vector(8, f32), nibs1.n1, x2_1, wx21);
            wx21 = @mulAdd(@Vector(8, f32), nibs1.n2, x2_2, wx21);
            wx21 = @mulAdd(@Vector(8, f32), nibs1.n3, x2_3, wx21);

            xs2 += x2_0 + x2_1 + x2_2 + x2_3;

            // Row 3
            wx30 = @mulAdd(@Vector(8, f32), nibs0.n0, x3_0, wx30);
            wx30 = @mulAdd(@Vector(8, f32), nibs0.n1, x3_1, wx30);
            wx30 = @mulAdd(@Vector(8, f32), nibs0.n2, x3_2, wx30);
            wx30 = @mulAdd(@Vector(8, f32), nibs0.n3, x3_3, wx30);

            wx31 = @mulAdd(@Vector(8, f32), nibs1.n0, x3_0, wx31);
            wx31 = @mulAdd(@Vector(8, f32), nibs1.n1, x3_1, wx31);
            wx31 = @mulAdd(@Vector(8, f32), nibs1.n2, x3_2, wx31);
            wx31 = @mulAdd(@Vector(8, f32), nibs1.n3, x3_3, wx31);

            xs3 += x3_0 + x3_1 + x3_2 + x3_3;
        }

        // Remainder loop (handles non-multiple-of-4 group sizes)
        while (pack_idx < group_u32) : (pack_idx += 1) {
            const w0 = extractNibbles(w0_ptr[pack_idx]);
            const w1 = extractNibbles(w1_ptr[pack_idx]);

            const x0: @Vector(8, f32) = (a0_ptr + pack_idx * 8)[0..8].*;
            const x1: @Vector(8, f32) = (a1_ptr + pack_idx * 8)[0..8].*;
            const x2: @Vector(8, f32) = (a2_ptr + pack_idx * 8)[0..8].*;
            const x3: @Vector(8, f32) = (a3_ptr + pack_idx * 8)[0..8].*;

            wx00 = @mulAdd(@Vector(8, f32), w0, x0, wx00);
            wx01 = @mulAdd(@Vector(8, f32), w1, x0, wx01);
            xs0 += x0;

            wx10 = @mulAdd(@Vector(8, f32), w0, x1, wx10);
            wx11 = @mulAdd(@Vector(8, f32), w1, x1, wx11);
            xs1 += x1;

            wx20 = @mulAdd(@Vector(8, f32), w0, x2, wx20);
            wx21 = @mulAdd(@Vector(8, f32), w1, x2, wx21);
            xs2 += x2;

            wx30 = @mulAdd(@Vector(8, f32), w0, x3, wx30);
            wx31 = @mulAdd(@Vector(8, f32), w1, x3, wx31);
            xs3 += x3;
        }

        // Apply scale to wx accumulators (VECTOR operations, no reduce yet!)
        const s0_vec: @Vector(8, f32) = @splat(scale0);
        const s1_vec: @Vector(8, f32) = @splat(scale1);
        const b0_vec: @Vector(8, f32) = @splat(bias0);
        const b1_vec: @Vector(8, f32) = @splat(bias1);

        // acc += scale * wx (vector FMA)
        acc00 = @mulAdd(@Vector(8, f32), wx00, s0_vec, acc00);
        acc01 = @mulAdd(@Vector(8, f32), wx01, s1_vec, acc01);
        acc10 = @mulAdd(@Vector(8, f32), wx10, s0_vec, acc10);
        acc11 = @mulAdd(@Vector(8, f32), wx11, s1_vec, acc11);
        acc20 = @mulAdd(@Vector(8, f32), wx20, s0_vec, acc20);
        acc21 = @mulAdd(@Vector(8, f32), wx21, s1_vec, acc21);
        acc30 = @mulAdd(@Vector(8, f32), wx30, s0_vec, acc30);
        acc31 = @mulAdd(@Vector(8, f32), wx31, s1_vec, acc31);

        // bias_acc += bias * xs (vector FMA)
        // Note: bias_acc accumulates bias*xs for BOTH columns (b0 and b1)
        // We need separate terms since b0 != b1
        bias_acc0 = @mulAdd(@Vector(8, f32), xs0, b0_vec, bias_acc0);
        bias_acc1 = @mulAdd(@Vector(8, f32), xs1, b0_vec, bias_acc1);
        bias_acc2 = @mulAdd(@Vector(8, f32), xs2, b0_vec, bias_acc2);
        bias_acc3 = @mulAdd(@Vector(8, f32), xs3, b0_vec, bias_acc3);

        // For column 1, we need xs*b1 - accumulate into acc directly
        acc01 = @mulAdd(@Vector(8, f32), xs0, b1_vec, acc01);
        acc11 = @mulAdd(@Vector(8, f32), xs1, b1_vec, acc11);
        acc21 = @mulAdd(@Vector(8, f32), xs2, b1_vec, acc21);
        acc31 = @mulAdd(@Vector(8, f32), xs3, b1_vec, acc31);
    }

    // NOW reduce (only ONCE at the very end!)
    const out00 = @reduce(.Add, acc00) + @reduce(.Add, bias_acc0);
    const out01 = @reduce(.Add, acc01); // already has bias term
    const out10 = @reduce(.Add, acc10) + @reduce(.Add, bias_acc1);
    const out11 = @reduce(.Add, acc11);
    const out20 = @reduce(.Add, acc20) + @reduce(.Add, bias_acc2);
    const out21 = @reduce(.Add, acc21);
    const out30 = @reduce(.Add, acc30) + @reduce(.Add, bias_acc3);
    const out31 = @reduce(.Add, acc31);

    // Write outputs
    context.out[row_base_idx * n_dim + col_idx] = out00;
    context.out[row_base_idx * n_dim + col_idx + 1] = out01;
    context.out[(row_base_idx + 1) * n_dim + col_idx] = out10;
    context.out[(row_base_idx + 1) * n_dim + col_idx + 1] = out11;
    context.out[(row_base_idx + 2) * n_dim + col_idx] = out20;
    context.out[(row_base_idx + 2) * n_dim + col_idx + 1] = out21;
    context.out[(row_base_idx + 3) * n_dim + col_idx] = out30;
    context.out[(row_base_idx + 3) * n_dim + col_idx + 1] = out31;
}

// =============================================================================
// 1x2 Kernel (for remainder rows)
// =============================================================================

inline fn kernel1x2(context: anytype, row_idx: usize, col_idx: usize) void {
    @setFloatMode(.optimized);

    const k_dim = context.k_dim;
    const n_dim = context.n_cols;
    const group = context.group;
    const k_div_group = context.k_div_group;
    const k_div_8 = context.k_div_8;
    const group_u32 = context.group_u32;

    const w0_base = context.packed_b.ptr + col_idx * k_div_8;
    const w1_base = context.packed_b.ptr + (col_idx + 1) * k_div_8;
    const a_base = context.a.ptr + row_idx * k_dim;

    const scale0_base = context.scales.ptr + col_idx * k_div_group;
    const scale1_base = context.scales.ptr + (col_idx + 1) * k_div_group;
    const bias0_base = context.biases.ptr + col_idx * k_div_group;
    const bias1_base = context.biases.ptr + (col_idx + 1) * k_div_group;

    var out0: f32 = 0;
    var out1: f32 = 0;

    var group_idx: usize = 0;
    while (group_idx < k_div_group) : (group_idx += 1) {
        const scale0 = scaleBiasToF32(context.scales_dtype, scale0_base[group_idx]);
        const scale1 = scaleBiasToF32(context.scales_dtype, scale1_base[group_idx]);
        const bias0 = scaleBiasToF32(context.scales_dtype, bias0_base[group_idx]);
        const bias1 = scaleBiasToF32(context.scales_dtype, bias1_base[group_idx]);

        const w0_ptr = w0_base + group_idx * group_u32;
        const w1_ptr = w1_base + group_idx * group_u32;
        const a_ptr = a_base + group_idx * group;

        var wx0: @Vector(8, f32) = @splat(0);
        var wx1: @Vector(8, f32) = @splat(0);
        var x_sum_vec: @Vector(8, f32) = @splat(0);

        var pack_idx: usize = 0;
        while (pack_idx + 3 < group_u32) : (pack_idx += 4) {
            const nibs0 = extract32NibblesToFloat(w0_ptr + pack_idx);
            const nibs1 = extract32NibblesToFloat(w1_ptr + pack_idx);

            const x0: @Vector(8, f32) = (a_ptr + pack_idx * 8)[0..8].*;
            const x1: @Vector(8, f32) = (a_ptr + (pack_idx + 1) * 8)[0..8].*;
            const x2: @Vector(8, f32) = (a_ptr + (pack_idx + 2) * 8)[0..8].*;
            const x3: @Vector(8, f32) = (a_ptr + (pack_idx + 3) * 8)[0..8].*;

            wx0 = @mulAdd(@Vector(8, f32), nibs0.n0, x0, wx0);
            wx0 = @mulAdd(@Vector(8, f32), nibs0.n1, x1, wx0);
            wx0 = @mulAdd(@Vector(8, f32), nibs0.n2, x2, wx0);
            wx0 = @mulAdd(@Vector(8, f32), nibs0.n3, x3, wx0);

            wx1 = @mulAdd(@Vector(8, f32), nibs1.n0, x0, wx1);
            wx1 = @mulAdd(@Vector(8, f32), nibs1.n1, x1, wx1);
            wx1 = @mulAdd(@Vector(8, f32), nibs1.n2, x2, wx1);
            wx1 = @mulAdd(@Vector(8, f32), nibs1.n3, x3, wx1);

            x_sum_vec += x0 + x1 + x2 + x3;
        }

        while (pack_idx < group_u32) : (pack_idx += 1) {
            const w0 = extractNibbles(w0_ptr[pack_idx]);
            const w1 = extractNibbles(w1_ptr[pack_idx]);
            const x_vec: @Vector(8, f32) = (a_ptr + pack_idx * 8)[0..8].*;

            wx0 = @mulAdd(@Vector(8, f32), w0, x_vec, wx0);
            wx1 = @mulAdd(@Vector(8, f32), w1, x_vec, wx1);
            x_sum_vec += x_vec;
        }

        const sum_wx0 = @reduce(.Add, wx0);
        const sum_wx1 = @reduce(.Add, wx1);
        const sum_xs = @reduce(.Add, x_sum_vec);

        out0 += sum_wx0 * scale0 + sum_xs * bias0;
        out1 += sum_wx1 * scale1 + sum_xs * bias1;
    }

    context.out[row_idx * n_dim + col_idx] = out0;
    context.out[row_idx * n_dim + col_idx + 1] = out1;
}

// =============================================================================
// 1x1 Kernel (for odd column at end) - reuses decode's optimized dot product
// =============================================================================

inline fn kernel1x1(context: anytype, row_idx: usize, col_idx: usize) void {
    const k_dim = context.k_dim;
    const n_dim = context.n_cols;
    const k_div_group = context.k_div_group;
    const k_div_8 = context.k_div_8;

    const w_ptr = context.packed_b.ptr + col_idx * k_div_8;
    const a_ptr = context.a.ptr + row_idx * k_dim;
    const scale_ptr = context.scales.ptr + col_idx * k_div_group;
    const bias_ptr = context.biases.ptr + col_idx * k_div_group;

    // Pre-convert scales/biases (same as decode kernel)
    var scales_f32: [matmul.MAX_GROUPS]f32 align(64) = undefined;
    var biases_f32: [matmul.MAX_GROUPS]f32 align(64) = undefined;

    for (0..k_div_group) |group_idx| {
        scales_f32[group_idx] = scaleBiasToF32(context.scales_dtype, scale_ptr[group_idx]);
        biases_f32[group_idx] = scaleBiasToF32(context.scales_dtype, bias_ptr[group_idx]);
    }

    // Reuse decode's optimized dot product
    context.out[row_idx * n_dim + col_idx] = matmul.gaffineU4DotProductOpt(
        a_ptr,
        w_ptr,
        &scales_f32,
        &biases_f32,
        context.group,
        k_div_group,
        context.group_u32,
    );
}

// =============================================================================
// 8-bit Prefill (optimized 4x2 microkernel)
// =============================================================================
// Same strategy as 4-bit: process 4 rows × 2 columns at once for better
// arithmetic intensity (weight reuse across rows).

pub fn matmulGaffineU8Prefill(
    a_data: []const f32,
    m_rows: usize,
    k_dim: usize,
    packed_vals: []align(1) const u32,
    scales: []align(1) const u16,
    biases: []align(1) const u16,
    scales_dtype: DType,
    n_cols: usize,
    group: usize,
    out_data: []f32,
) void {
    const k_div_4 = k_dim / 4;
    const k_div_group = k_dim / group;
    const group_u32 = group / 4;

    const MatmulU8PrefillCtx = struct {
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

    var context = MatmulU8PrefillCtx{
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

    // Parallelize over column pairs (same as 4-bit)
    const num_col_pairs = n_cols / 2;

    const col_pair_task = struct {
        fn runColPairs(start: usize, end: usize, task_ctx: *MatmulU8PrefillCtx) void {
            @setFloatMode(.optimized);

            var col_pair = start;
            while (col_pair < end) : (col_pair += 1) {
                const col = col_pair * 2;

                // Process rows in groups of 4
                var row: usize = 0;
                while (row + 3 < task_ctx.m_rows) : (row += 4) {
                    kernel4x2_8bit(task_ctx, row, col);
                }
                // Remainder rows
                while (row < task_ctx.m_rows) : (row += 1) {
                    kernel1x2_8bit(task_ctx, row, col);
                }
            }
        }
    }.runColPairs;

    parallel.global().parallelFor(num_col_pairs, col_pair_task, &context);

    // Handle odd column at end
    if (n_cols % 2 == 1) {
        const col = n_cols - 1;
        for (0..m_rows) |row| {
            kernel1x1_8bit(&context, row, col);
        }
    }
}

/// 4x2 kernel for 8-bit: process 4 rows × 2 columns
fn kernel4x2_8bit(context: anytype, row_base_idx: usize, col_idx: usize) void {
    @setFloatMode(.optimized);

    const k_dim = context.k_dim;
    const n_dim = context.n_cols;
    const group = context.group;
    const k_div_group = context.k_div_group;
    const k_div_4 = context.k_div_4;
    const group_u32 = context.group_u32;

    // Weight pointers for 2 columns
    const w0_base = context.packed_b.ptr + col_idx * k_div_4;
    const w1_base = context.packed_b.ptr + (col_idx + 1) * k_div_4;

    // Activation pointers for 4 rows
    const a0_base = context.a.ptr + row_base_idx * k_dim;
    const a1_base = context.a.ptr + (row_base_idx + 1) * k_dim;
    const a2_base = context.a.ptr + (row_base_idx + 2) * k_dim;
    const a3_base = context.a.ptr + (row_base_idx + 3) * k_dim;

    // Scale/bias pointers
    const scale0_base = context.scales.ptr + col_idx * k_div_group;
    const scale1_base = context.scales.ptr + (col_idx + 1) * k_div_group;
    const bias0_base = context.biases.ptr + col_idx * k_div_group;
    const bias1_base = context.biases.ptr + (col_idx + 1) * k_div_group;

    // Accumulators for scale*wx and bias*xs (defer reduction to end)
    var acc00: @Vector(8, f32) = @splat(0);
    var acc01: @Vector(8, f32) = @splat(0);
    var acc10: @Vector(8, f32) = @splat(0);
    var acc11: @Vector(8, f32) = @splat(0);
    var acc20: @Vector(8, f32) = @splat(0);
    var acc21: @Vector(8, f32) = @splat(0);
    var acc30: @Vector(8, f32) = @splat(0);
    var acc31: @Vector(8, f32) = @splat(0);
    // Separate accumulators for bias*xs terms
    var bias_acc0: @Vector(8, f32) = @splat(0);
    var bias_acc1: @Vector(8, f32) = @splat(0);
    var bias_acc2: @Vector(8, f32) = @splat(0);
    var bias_acc3: @Vector(8, f32) = @splat(0);

    // Process each quantization group
    var group_idx: usize = 0;
    while (group_idx < k_div_group) : (group_idx += 1) {
        const scale0 = scaleBiasToF32(context.scales_dtype, scale0_base[group_idx]);
        const scale1 = scaleBiasToF32(context.scales_dtype, scale1_base[group_idx]);
        const bias0 = scaleBiasToF32(context.scales_dtype, bias0_base[group_idx]);
        const bias1 = scaleBiasToF32(context.scales_dtype, bias1_base[group_idx]);

        const w0_ptr = w0_base + group_idx * group_u32;
        const w1_ptr = w1_base + group_idx * group_u32;
        const a0_ptr = a0_base + group_idx * group;
        const a1_ptr = a1_base + group_idx * group;
        const a2_ptr = a2_base + group_idx * group;
        const a3_ptr = a3_base + group_idx * group;

        // Per-group accumulators
        var wx00: @Vector(8, f32) = @splat(0);
        var wx01: @Vector(8, f32) = @splat(0);
        var wx10: @Vector(8, f32) = @splat(0);
        var wx11: @Vector(8, f32) = @splat(0);
        var wx20: @Vector(8, f32) = @splat(0);
        var wx21: @Vector(8, f32) = @splat(0);
        var wx30: @Vector(8, f32) = @splat(0);
        var wx31: @Vector(8, f32) = @splat(0);

        var xs0: @Vector(8, f32) = @splat(0);
        var xs1: @Vector(8, f32) = @splat(0);
        var xs2: @Vector(8, f32) = @splat(0);
        var xs3: @Vector(8, f32) = @splat(0);

        // Process 8 bytes (2 u32s) per iteration for 8-wide vectors
        var pack_idx: usize = 0;
        while (pack_idx + 1 < group_u32) : (pack_idx += 2) {
            // Extract 8 bytes from 2 u32s for each column
            const w0 = extract8BytesToFloat(w0_ptr + pack_idx);
            const w1 = extract8BytesToFloat(w1_ptr + pack_idx);

            // Load activations (8 f32s)
            const x0: @Vector(8, f32) = (a0_ptr + pack_idx * 4)[0..8].*;
            const x1: @Vector(8, f32) = (a1_ptr + pack_idx * 4)[0..8].*;
            const x2: @Vector(8, f32) = (a2_ptr + pack_idx * 4)[0..8].*;
            const x3: @Vector(8, f32) = (a3_ptr + pack_idx * 4)[0..8].*;

            // Row 0
            wx00 = @mulAdd(@Vector(8, f32), w0, x0, wx00);
            wx01 = @mulAdd(@Vector(8, f32), w1, x0, wx01);
            xs0 += x0;

            // Row 1
            wx10 = @mulAdd(@Vector(8, f32), w0, x1, wx10);
            wx11 = @mulAdd(@Vector(8, f32), w1, x1, wx11);
            xs1 += x1;

            // Row 2
            wx20 = @mulAdd(@Vector(8, f32), w0, x2, wx20);
            wx21 = @mulAdd(@Vector(8, f32), w1, x2, wx21);
            xs2 += x2;

            // Row 3
            wx30 = @mulAdd(@Vector(8, f32), w0, x3, wx30);
            wx31 = @mulAdd(@Vector(8, f32), w1, x3, wx31);
            xs3 += x3;
        }

        // Remainder (odd u32)
        while (pack_idx < group_u32) : (pack_idx += 1) {
            const w0 = extractBytes(w0_ptr[pack_idx]);
            const w1 = extractBytes(w1_ptr[pack_idx]);

            const x0: @Vector(4, f32) = (a0_ptr + pack_idx * 4)[0..4].*;
            const x1: @Vector(4, f32) = (a1_ptr + pack_idx * 4)[0..4].*;
            const x2: @Vector(4, f32) = (a2_ptr + pack_idx * 4)[0..4].*;
            const x3: @Vector(4, f32) = (a3_ptr + pack_idx * 4)[0..4].*;

            // Accumulate with 4-wide vectors, pad to 8
            const w0_8: @Vector(8, f32) = .{ w0[0], w0[1], w0[2], w0[3], 0, 0, 0, 0 };
            const w1_8: @Vector(8, f32) = .{ w1[0], w1[1], w1[2], w1[3], 0, 0, 0, 0 };
            const x0_8: @Vector(8, f32) = .{ x0[0], x0[1], x0[2], x0[3], 0, 0, 0, 0 };
            const x1_8: @Vector(8, f32) = .{ x1[0], x1[1], x1[2], x1[3], 0, 0, 0, 0 };
            const x2_8: @Vector(8, f32) = .{ x2[0], x2[1], x2[2], x2[3], 0, 0, 0, 0 };
            const x3_8: @Vector(8, f32) = .{ x3[0], x3[1], x3[2], x3[3], 0, 0, 0, 0 };

            wx00 = @mulAdd(@Vector(8, f32), w0_8, x0_8, wx00);
            wx01 = @mulAdd(@Vector(8, f32), w1_8, x0_8, wx01);
            xs0 += x0_8;

            wx10 = @mulAdd(@Vector(8, f32), w0_8, x1_8, wx10);
            wx11 = @mulAdd(@Vector(8, f32), w1_8, x1_8, wx11);
            xs1 += x1_8;

            wx20 = @mulAdd(@Vector(8, f32), w0_8, x2_8, wx20);
            wx21 = @mulAdd(@Vector(8, f32), w1_8, x2_8, wx21);
            xs2 += x2_8;

            wx30 = @mulAdd(@Vector(8, f32), w0_8, x3_8, wx30);
            wx31 = @mulAdd(@Vector(8, f32), w1_8, x3_8, wx31);
            xs3 += x3_8;
        }

        // Apply scale to wx accumulators (vector FMA)
        const s0_vec: @Vector(8, f32) = @splat(scale0);
        const s1_vec: @Vector(8, f32) = @splat(scale1);
        const b0_vec: @Vector(8, f32) = @splat(bias0);
        const b1_vec: @Vector(8, f32) = @splat(bias1);

        acc00 = @mulAdd(@Vector(8, f32), wx00, s0_vec, acc00);
        acc01 = @mulAdd(@Vector(8, f32), wx01, s1_vec, acc01);
        acc10 = @mulAdd(@Vector(8, f32), wx10, s0_vec, acc10);
        acc11 = @mulAdd(@Vector(8, f32), wx11, s1_vec, acc11);
        acc20 = @mulAdd(@Vector(8, f32), wx20, s0_vec, acc20);
        acc21 = @mulAdd(@Vector(8, f32), wx21, s1_vec, acc21);
        acc30 = @mulAdd(@Vector(8, f32), wx30, s0_vec, acc30);
        acc31 = @mulAdd(@Vector(8, f32), wx31, s1_vec, acc31);

        // bias_acc += bias * xs
        bias_acc0 = @mulAdd(@Vector(8, f32), xs0, b0_vec, bias_acc0);
        bias_acc1 = @mulAdd(@Vector(8, f32), xs1, b0_vec, bias_acc1);
        bias_acc2 = @mulAdd(@Vector(8, f32), xs2, b0_vec, bias_acc2);
        bias_acc3 = @mulAdd(@Vector(8, f32), xs3, b0_vec, bias_acc3);

        // For column 1, accumulate xs*b1 into acc directly
        acc01 = @mulAdd(@Vector(8, f32), xs0, b1_vec, acc01);
        acc11 = @mulAdd(@Vector(8, f32), xs1, b1_vec, acc11);
        acc21 = @mulAdd(@Vector(8, f32), xs2, b1_vec, acc21);
        acc31 = @mulAdd(@Vector(8, f32), xs3, b1_vec, acc31);
    }

    // Final reduction
    const out00 = @reduce(.Add, acc00) + @reduce(.Add, bias_acc0);
    const out01 = @reduce(.Add, acc01);
    const out10 = @reduce(.Add, acc10) + @reduce(.Add, bias_acc1);
    const out11 = @reduce(.Add, acc11);
    const out20 = @reduce(.Add, acc20) + @reduce(.Add, bias_acc2);
    const out21 = @reduce(.Add, acc21);
    const out30 = @reduce(.Add, acc30) + @reduce(.Add, bias_acc3);
    const out31 = @reduce(.Add, acc31);

    // Write outputs
    context.out[row_base_idx * n_dim + col_idx] = out00;
    context.out[row_base_idx * n_dim + col_idx + 1] = out01;
    context.out[(row_base_idx + 1) * n_dim + col_idx] = out10;
    context.out[(row_base_idx + 1) * n_dim + col_idx + 1] = out11;
    context.out[(row_base_idx + 2) * n_dim + col_idx] = out20;
    context.out[(row_base_idx + 2) * n_dim + col_idx + 1] = out21;
    context.out[(row_base_idx + 3) * n_dim + col_idx] = out30;
    context.out[(row_base_idx + 3) * n_dim + col_idx + 1] = out31;
}

/// 1x2 kernel for 8-bit (remainder rows)
inline fn kernel1x2_8bit(context: anytype, row_idx: usize, col_idx: usize) void {
    @setFloatMode(.optimized);

    const k_dim = context.k_dim;
    const n_dim = context.n_cols;
    const group = context.group;
    const k_div_group = context.k_div_group;
    const k_div_4 = context.k_div_4;
    const group_u32 = context.group_u32;

    const w0_base = context.packed_b.ptr + col_idx * k_div_4;
    const w1_base = context.packed_b.ptr + (col_idx + 1) * k_div_4;
    const a_base = context.a.ptr + row_idx * k_dim;

    const scale0_base = context.scales.ptr + col_idx * k_div_group;
    const scale1_base = context.scales.ptr + (col_idx + 1) * k_div_group;
    const bias0_base = context.biases.ptr + col_idx * k_div_group;
    const bias1_base = context.biases.ptr + (col_idx + 1) * k_div_group;

    var out0: f32 = 0;
    var out1: f32 = 0;

    var group_idx: usize = 0;
    while (group_idx < k_div_group) : (group_idx += 1) {
        const scale0 = scaleBiasToF32(context.scales_dtype, scale0_base[group_idx]);
        const scale1 = scaleBiasToF32(context.scales_dtype, scale1_base[group_idx]);
        const bias0 = scaleBiasToF32(context.scales_dtype, bias0_base[group_idx]);
        const bias1 = scaleBiasToF32(context.scales_dtype, bias1_base[group_idx]);

        const w0_ptr = w0_base + group_idx * group_u32;
        const w1_ptr = w1_base + group_idx * group_u32;
        const a_ptr = a_base + group_idx * group;

        var wx0: @Vector(8, f32) = @splat(0);
        var wx1: @Vector(8, f32) = @splat(0);
        var x_sum_vec: @Vector(8, f32) = @splat(0);

        var pack_idx: usize = 0;
        while (pack_idx + 1 < group_u32) : (pack_idx += 2) {
            const w0 = extract8BytesToFloat(w0_ptr + pack_idx);
            const w1 = extract8BytesToFloat(w1_ptr + pack_idx);
            const x_vec: @Vector(8, f32) = (a_ptr + pack_idx * 4)[0..8].*;

            wx0 = @mulAdd(@Vector(8, f32), w0, x_vec, wx0);
            wx1 = @mulAdd(@Vector(8, f32), w1, x_vec, wx1);
            x_sum_vec += x_vec;
        }

        while (pack_idx < group_u32) : (pack_idx += 1) {
            const w0 = extractBytes(w0_ptr[pack_idx]);
            const w1 = extractBytes(w1_ptr[pack_idx]);
            const x_vec: @Vector(4, f32) = (a_ptr + pack_idx * 4)[0..4].*;

            out0 += @reduce(.Add, w0 * x_vec) * scale0 + @reduce(.Add, x_vec) * bias0;
            out1 += @reduce(.Add, w1 * x_vec) * scale1 + @reduce(.Add, x_vec) * bias1;
        }

        out0 += @reduce(.Add, wx0) * scale0 + @reduce(.Add, x_sum_vec) * bias0;
        out1 += @reduce(.Add, wx1) * scale1 + @reduce(.Add, x_sum_vec) * bias1;
    }

    context.out[row_idx * n_dim + col_idx] = out0;
    context.out[row_idx * n_dim + col_idx + 1] = out1;
}

/// 1x1 kernel for 8-bit (odd column at end) - reuses decode's optimized dot product
inline fn kernel1x1_8bit(context: anytype, row_idx: usize, col_idx: usize) void {
    const k_dim = context.k_dim;
    const n_dim = context.n_cols;
    const k_div_group = context.k_div_group;
    const k_div_4 = context.k_div_4;

    const w_ptr = context.packed_b.ptr + col_idx * k_div_4;
    const a_ptr = context.a.ptr + row_idx * k_dim;
    const scale_ptr = context.scales.ptr + col_idx * k_div_group;
    const bias_ptr = context.biases.ptr + col_idx * k_div_group;

    // Pre-convert scales/biases (same as decode kernel)
    var scales_f32: [matmul.MAX_GROUPS]f32 align(64) = undefined;
    var biases_f32: [matmul.MAX_GROUPS]f32 align(64) = undefined;

    for (0..k_div_group) |group_idx| {
        scales_f32[group_idx] = scaleBiasToF32(context.scales_dtype, scale_ptr[group_idx]);
        biases_f32[group_idx] = scaleBiasToF32(context.scales_dtype, bias_ptr[group_idx]);
    }

    // Reuse decode's optimized dot product
    context.out[row_idx * n_dim + col_idx] = matmul.gaffineU8DotProductOpt(
        a_ptr,
        w_ptr,
        &scales_f32,
        &biases_f32,
        context.group,
        k_div_group,
        context.group_u32,
    );
}

/// Extract 8 bytes from 2 u32s as @Vector(8, f32)
inline fn extract8BytesToFloat(w_ptr: [*]align(1) const u32) @Vector(8, f32) {
    const bytes: @Vector(8, u8) = @as(*align(1) const [8]u8, @ptrCast(w_ptr)).*;
    return @floatFromInt(@as(@Vector(8, u32), bytes));
}

// =============================================================================
// Unit Tests
// =============================================================================

test "matmulGaffineU8Prefill extract8BytesToFloat basic" {
    // Test extracting 8 bytes from 2 u32s
    const data = [_]u32{ 0x03020100, 0x07060504 }; // bytes 0-7
    const result = extract8BytesToFloat(&data);

    try std.testing.expectEqual(@as(f32, 0.0), result[0]);
    try std.testing.expectEqual(@as(f32, 1.0), result[1]);
    try std.testing.expectEqual(@as(f32, 2.0), result[2]);
    try std.testing.expectEqual(@as(f32, 3.0), result[3]);
    try std.testing.expectEqual(@as(f32, 4.0), result[4]);
    try std.testing.expectEqual(@as(f32, 5.0), result[5]);
    try std.testing.expectEqual(@as(f32, 6.0), result[6]);
    try std.testing.expectEqual(@as(f32, 7.0), result[7]);
}

test "matmulGaffineU4Prefill - basic 4x4 matrix multiplication" {
    // Test basic functionality: A (4x8) @ B (8x4) = C (4x4)
    // Use small dimensions where we can verify results manually
    const m: usize = 4;
    const k: usize = 8; // Must be multiple of group size
    const n: usize = 4; // Must be even for 4x2 microkernel
    const group: usize = 8;

    const allocator = std.testing.allocator;

    // Activation matrix A (4x8) - row-major
    const a_data = try allocator.alloc(f32, m * k);
    defer allocator.free(a_data);

    // Simple pattern: each row has incrementing values
    for (0..m) |i| {
        for (0..k) |j| {
            a_data[i * k + j] = @floatFromInt(j + 1); // [1,2,3,4,5,6,7,8] for each row
        }
    }

    // Weight matrix B (8x4) - stored as quantized 4-bit
    // Each weight column needs k/8 u32s (8 nibbles per u32)
    const k_div_8 = k / 8;
    const packed_vals = try allocator.alloc(u32, n * k_div_8);
    defer allocator.free(packed_vals);

    // Fill with simple pattern: each nibble = 1
    // This means each weight is 1.0 (before scale/bias)
    for (packed_vals) |*val| {
        val.* = 0x11111111; // All nibbles = 1
    }

    // Scales and biases (one per group per column)
    const k_div_group = k / group;
    var scales = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(scales);
    var biases = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(biases);

    // Use f16 format: scale=1.0, bias=0.0
    const scale_fp16 = dtype_mod.f32ToFp16(1.0);
    const bias_fp16 = dtype_mod.f32ToFp16(0.0);
    for (0..n * k_div_group) |i| {
        scales[i] = scale_fp16;
        biases[i] = bias_fp16;
    }

    // Output matrix C (4x4)
    const out_data = try allocator.alloc(f32, m * n);
    defer allocator.free(out_data);
    @memset(out_data, 0);

    // Run the kernel
    matmulGaffineU4Prefill(
        a_data,
        m,
        k,
        packed_vals,
        scales,
        biases,
        .f16,
        n,
        group,
        out_data,
    );

    // Expected result: each element should be sum(1*weights) = sum([1,2,3,4,5,6,7,8]) * 1 (weight=1)
    // = 1+2+3+4+5+6+7+8 = 36
    const expected: f32 = 36.0;

    for (0..m) |i| {
        for (0..n) |j| {
            const actual = out_data[i * n + j];
            try std.testing.expectApproxEqAbs(expected, actual, 0.01);
        }
    }
}

test "matmulGaffineU4Prefill - scale and bias application" {
    // Test that scale and bias are correctly applied
    const m: usize = 2;
    const k: usize = 8;
    const n: usize = 2;
    const group: usize = 8;

    const allocator = std.testing.allocator;

    const a_data = try allocator.alloc(f32, m * k);
    defer allocator.free(a_data);

    // Fill with ones for simplicity
    @memset(a_data, 1.0);

    const k_div_8 = k / 8;
    const packed_vals = try allocator.alloc(u32, n * k_div_8);
    defer allocator.free(packed_vals);

    // All nibbles = 2
    for (packed_vals) |*val| {
        val.* = 0x22222222;
    }

    const k_div_group = k / group;
    var scales = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(scales);
    var biases = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(biases);

    // scale=2.0, bias=3.0
    const scale_fp16 = dtype_mod.f32ToFp16(2.0);
    const bias_fp16 = dtype_mod.f32ToFp16(3.0);
    for (0..n * k_div_group) |i| {
        scales[i] = scale_fp16;
        biases[i] = bias_fp16;
    }

    const out_data = try allocator.alloc(f32, m * n);
    defer allocator.free(out_data);
    @memset(out_data, 0);

    matmulGaffineU4Prefill(
        a_data,
        m,
        k,
        packed_vals,
        scales,
        biases,
        .f16,
        n,
        group,
        out_data,
    );

    // Expected: scale * sum(w*x) + bias * sum(x)
    // sum(w*x) = 8 weights * 2 (nibble) * 1 (activation) = 16
    // sum(x) = 8 * 1 = 8
    // result = 2.0 * 16 + 3.0 * 8 = 32 + 24 = 56
    const expected: f32 = 56.0;

    for (0..m) |i| {
        for (0..n) |j| {
            const actual = out_data[i * n + j];
            try std.testing.expectApproxEqAbs(expected, actual, 0.01);
        }
    }
}

test "matmulGaffineU4Prefill - odd column handling" {
    // Test handling of odd number of columns (tests kernel1x1 path)
    const m: usize = 2;
    const k: usize = 8;
    const n: usize = 3; // Odd number
    const group: usize = 8;

    const allocator = std.testing.allocator;

    const a_data = try allocator.alloc(f32, m * k);
    defer allocator.free(a_data);
    @memset(a_data, 1.0);

    const k_div_8 = k / 8;
    const packed_vals = try allocator.alloc(u32, n * k_div_8);
    defer allocator.free(packed_vals);

    for (packed_vals) |*val| {
        val.* = 0x11111111;
    }

    const k_div_group = k / group;
    var scales = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(scales);
    var biases = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(biases);

    const scale_fp16 = dtype_mod.f32ToFp16(1.0);
    const bias_fp16 = dtype_mod.f32ToFp16(0.0);
    for (0..n * k_div_group) |i| {
        scales[i] = scale_fp16;
        biases[i] = bias_fp16;
    }

    const out_data = try allocator.alloc(f32, m * n);
    defer allocator.free(out_data);
    @memset(out_data, 0);

    matmulGaffineU4Prefill(
        a_data,
        m,
        k,
        packed_vals,
        scales,
        biases,
        .f16,
        n,
        group,
        out_data,
    );

    // Expected: 8 weights * 1 (nibble) * 1 (activation) = 8
    const expected: f32 = 8.0;

    for (0..m) |i| {
        for (0..n) |j| {
            const actual = out_data[i * n + j];
            try std.testing.expectApproxEqAbs(expected, actual, 0.01);
        }
    }
}

test "matmulGaffineU4Prefill - remainder rows handling" {
    // Test handling of non-multiple-of-4 rows (tests kernel1x2 path)
    const m: usize = 5; // Not divisible by 4
    const k: usize = 8;
    const n: usize = 2;
    const group: usize = 8;

    const allocator = std.testing.allocator;

    const a_data = try allocator.alloc(f32, m * k);
    defer allocator.free(a_data);

    // Different values per row to verify correct indexing
    for (0..m) |i| {
        const row_val: f32 = @floatFromInt(i + 1);
        for (0..k) |j| {
            a_data[i * k + j] = row_val;
        }
    }

    const k_div_8 = k / 8;
    const packed_vals = try allocator.alloc(u32, n * k_div_8);
    defer allocator.free(packed_vals);

    for (packed_vals) |*val| {
        val.* = 0x11111111; // All weights = 1
    }

    const k_div_group = k / group;
    var scales = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(scales);
    var biases = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(biases);

    const scale_fp16 = dtype_mod.f32ToFp16(1.0);
    const bias_fp16 = dtype_mod.f32ToFp16(0.0);
    for (0..n * k_div_group) |i| {
        scales[i] = scale_fp16;
        biases[i] = bias_fp16;
    }

    const out_data = try allocator.alloc(f32, m * n);
    defer allocator.free(out_data);
    @memset(out_data, 0);

    matmulGaffineU4Prefill(
        a_data,
        m,
        k,
        packed_vals,
        scales,
        biases,
        .f16,
        n,
        group,
        out_data,
    );

    // Expected: row i has value (i+1) repeated 8 times, weights are 1
    // result[i] = 8 * (i+1) * 1 = 8 * (i+1)
    for (0..m) |i| {
        const expected: f32 = @as(f32, @floatFromInt((i + 1) * 8));
        for (0..n) |j| {
            const actual = out_data[i * n + j];
            try std.testing.expectApproxEqAbs(expected, actual, 0.01);
        }
    }
}

test "matmulGaffineU4Prefill - multiple groups" {
    // Test with multiple quantization groups
    const m: usize = 2;
    const k: usize = 16; // 2 groups of 8
    const n: usize = 2;
    const group: usize = 8;

    const allocator = std.testing.allocator;

    const a_data = try allocator.alloc(f32, m * k);
    defer allocator.free(a_data);

    // First group: all 1s, second group: all 2s
    for (0..m) |i| {
        for (0..8) |j| {
            a_data[i * k + j] = 1.0;
        }
        for (8..16) |j| {
            a_data[i * k + j] = 2.0;
        }
    }

    const k_div_8 = k / 8;
    const packed_vals = try allocator.alloc(u32, n * k_div_8);
    defer allocator.free(packed_vals);

    // All weights = 1
    for (packed_vals) |*val| {
        val.* = 0x11111111;
    }

    const k_div_group = k / group;
    var scales = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(scales);
    var biases = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(biases);

    // Different scales for each group
    for (0..n) |col| {
        scales[col * k_div_group + 0] = dtype_mod.f32ToFp16(1.0); // First group
        scales[col * k_div_group + 1] = dtype_mod.f32ToFp16(2.0); // Second group
        biases[col * k_div_group + 0] = dtype_mod.f32ToFp16(0.0);
        biases[col * k_div_group + 1] = dtype_mod.f32ToFp16(0.0);
    }

    const out_data = try allocator.alloc(f32, m * n);
    defer allocator.free(out_data);
    @memset(out_data, 0);

    matmulGaffineU4Prefill(
        a_data,
        m,
        k,
        packed_vals,
        scales,
        biases,
        .f16,
        n,
        group,
        out_data,
    );

    // Expected:
    // Group 1: 8 * 1 (act) * 1 (weight) * 1.0 (scale) = 8
    // Group 2: 8 * 2 (act) * 1 (weight) * 2.0 (scale) = 32
    // Total: 8 + 32 = 40
    const expected: f32 = 40.0;

    for (0..m) |i| {
        for (0..n) |j| {
            const actual = out_data[i * n + j];
            try std.testing.expectApproxEqAbs(expected, actual, 0.01);
        }
    }
}

test "matmulGaffineU8Prefill - basic 4x4 matrix multiplication" {
    // Test 8-bit quantization path
    const m: usize = 4;
    const k: usize = 8;
    const n: usize = 4;
    const group: usize = 8;

    const allocator = std.testing.allocator;

    const a_data = try allocator.alloc(f32, m * k);
    defer allocator.free(a_data);

    for (0..m) |i| {
        for (0..k) |j| {
            a_data[i * k + j] = @floatFromInt(j + 1);
        }
    }

    // For 8-bit: each u32 holds 4 bytes
    const k_div_4 = k / 4;
    const packed_vals = try allocator.alloc(u32, n * k_div_4);
    defer allocator.free(packed_vals);

    // Each byte = 2 (8-bit weights)
    for (packed_vals) |*val| {
        val.* = 0x02020202;
    }

    const k_div_group = k / group;
    var scales = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(scales);
    var biases = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(biases);

    const scale_fp16 = dtype_mod.f32ToFp16(1.0);
    const bias_fp16 = dtype_mod.f32ToFp16(0.0);
    for (0..n * k_div_group) |i| {
        scales[i] = scale_fp16;
        biases[i] = bias_fp16;
    }

    const out_data = try allocator.alloc(f32, m * n);
    defer allocator.free(out_data);
    @memset(out_data, 0);

    matmulGaffineU8Prefill(
        a_data,
        m,
        k,
        packed_vals,
        scales,
        biases,
        .f16,
        n,
        group,
        out_data,
    );

    // Expected: sum([1,2,3,4,5,6,7,8] * 2) = 2 * 36 = 72
    const expected: f32 = 72.0;

    for (0..m) |i| {
        for (0..n) |j| {
            const actual = out_data[i * n + j];
            try std.testing.expectApproxEqAbs(expected, actual, 0.01);
        }
    }
}

test "matmulGaffineU8Prefill - scale and bias application" {
    // Test scale/bias for 8-bit path
    const m: usize = 2;
    const k: usize = 8;
    const n: usize = 2;
    const group: usize = 8;

    const allocator = std.testing.allocator;

    const a_data = try allocator.alloc(f32, m * k);
    defer allocator.free(a_data);
    @memset(a_data, 1.0);

    const k_div_4 = k / 4;
    const packed_vals = try allocator.alloc(u32, n * k_div_4);
    defer allocator.free(packed_vals);

    for (packed_vals) |*val| {
        val.* = 0x03030303; // All bytes = 3
    }

    const k_div_group = k / group;
    var scales = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(scales);
    var biases = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(biases);

    const scale_fp16 = dtype_mod.f32ToFp16(2.0);
    const bias_fp16 = dtype_mod.f32ToFp16(5.0);
    for (0..n * k_div_group) |i| {
        scales[i] = scale_fp16;
        biases[i] = bias_fp16;
    }

    const out_data = try allocator.alloc(f32, m * n);
    defer allocator.free(out_data);
    @memset(out_data, 0);

    matmulGaffineU8Prefill(
        a_data,
        m,
        k,
        packed_vals,
        scales,
        biases,
        .f16,
        n,
        group,
        out_data,
    );

    // Expected: scale * sum(w*x) + bias * sum(x)
    // sum(w*x) = 8 * 3 * 1 = 24
    // sum(x) = 8 * 1 = 8
    // result = 2.0 * 24 + 5.0 * 8 = 48 + 40 = 88
    const expected: f32 = 88.0;

    for (0..m) |i| {
        for (0..n) |j| {
            const actual = out_data[i * n + j];
            try std.testing.expectApproxEqAbs(expected, actual, 0.01);
        }
    }
}

test "matmulGaffineU8Prefill - odd column handling" {
    // Test odd number of columns for 8-bit path
    const m: usize = 2;
    const k: usize = 8;
    const n: usize = 3;
    const group: usize = 8;

    const allocator = std.testing.allocator;

    const a_data = try allocator.alloc(f32, m * k);
    defer allocator.free(a_data);
    @memset(a_data, 1.0);

    const k_div_4 = k / 4;
    const packed_vals = try allocator.alloc(u32, n * k_div_4);
    defer allocator.free(packed_vals);

    for (packed_vals) |*val| {
        val.* = 0x02020202;
    }

    const k_div_group = k / group;
    var scales = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(scales);
    var biases = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(biases);

    const scale_fp16 = dtype_mod.f32ToFp16(1.0);
    const bias_fp16 = dtype_mod.f32ToFp16(0.0);
    for (0..n * k_div_group) |i| {
        scales[i] = scale_fp16;
        biases[i] = bias_fp16;
    }

    const out_data = try allocator.alloc(f32, m * n);
    defer allocator.free(out_data);
    @memset(out_data, 0);

    matmulGaffineU8Prefill(
        a_data,
        m,
        k,
        packed_vals,
        scales,
        biases,
        .f16,
        n,
        group,
        out_data,
    );

    // Expected: 8 * 2 * 1 = 16
    const expected: f32 = 16.0;

    for (0..m) |i| {
        for (0..n) |j| {
            const actual = out_data[i * n + j];
            try std.testing.expectApproxEqAbs(expected, actual, 0.01);
        }
    }
}

test "matmulGaffineU4Prefill - bfloat16 scale/bias" {
    // Test bfloat16 format for scales/biases
    const m: usize = 2;
    const k: usize = 8;
    const n: usize = 2;
    const group: usize = 8;

    const allocator = std.testing.allocator;

    const a_data = try allocator.alloc(f32, m * k);
    defer allocator.free(a_data);
    @memset(a_data, 1.0);

    const k_div_8 = k / 8;
    const packed_vals = try allocator.alloc(u32, n * k_div_8);
    defer allocator.free(packed_vals);

    for (packed_vals) |*val| {
        val.* = 0x11111111;
    }

    const k_div_group = k / group;
    var scales = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(scales);
    var biases = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(biases);

    // Use bfloat16 format
    const scale_bf16 = dtype_mod.f32ToBf16(1.5);
    const bias_bf16 = dtype_mod.f32ToBf16(0.5);
    for (0..n * k_div_group) |i| {
        scales[i] = scale_bf16;
        biases[i] = bias_bf16;
    }

    const out_data = try allocator.alloc(f32, m * n);
    defer allocator.free(out_data);
    @memset(out_data, 0);

    matmulGaffineU4Prefill(
        a_data,
        m,
        k,
        packed_vals,
        scales,
        biases,
        .bf16, // Use bfloat16
        n,
        group,
        out_data,
    );

    // Expected: 1.5 * (8 * 1 * 1) + 0.5 * (8 * 1) = 1.5 * 8 + 0.5 * 8 = 12 + 4 = 16
    const expected: f32 = 16.0;

    for (0..m) |i| {
        for (0..n) |j| {
            const actual = out_data[i * n + j];
            try std.testing.expectApproxEqAbs(expected, actual, 0.1); // Slightly larger tolerance for bf16
        }
    }
}

test "matmulGaffineU4Prefill - varying nibble values" {
    // Test that different nibble values are decoded correctly
    const m: usize = 1;
    const k: usize = 8;
    const n: usize = 2;
    const group: usize = 8;

    const allocator = std.testing.allocator;

    const a_data = try allocator.alloc(f32, m * k);
    defer allocator.free(a_data);
    @memset(a_data, 1.0);

    const k_div_8 = k / 8;
    const packed_vals = try allocator.alloc(u32, n * k_div_8);
    defer allocator.free(packed_vals);

    // Column 0: nibbles [0,1,2,3,4,5,6,7] (low nibbles of each byte)
    // Packed as: byte0=[0|1], byte1=[2|3], byte2=[4|5], byte3=[6|7]
    packed_vals[0] = 0x10 | (0x32 << 8) | (0x54 << 16) | (0x76 << 24);

    // Column 1: all nibbles = 15 (max value)
    packed_vals[1] = 0xFFFFFFFF;

    const k_div_group = k / group;
    var scales = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(scales);
    var biases = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(biases);

    const scale_fp16 = dtype_mod.f32ToFp16(1.0);
    const bias_fp16 = dtype_mod.f32ToFp16(0.0);
    for (0..n * k_div_group) |i| {
        scales[i] = scale_fp16;
        biases[i] = bias_fp16;
    }

    const out_data = try allocator.alloc(f32, m * n);
    defer allocator.free(out_data);
    @memset(out_data, 0);

    matmulGaffineU4Prefill(
        a_data,
        m,
        k,
        packed_vals,
        scales,
        biases,
        .f16,
        n,
        group,
        out_data,
    );

    // Column 0: sum([0,1,2,3,4,5,6,7] * 1) = 0+1+2+3+4+5+6+7 = 28
    try std.testing.expectApproxEqAbs(@as(f32, 28.0), out_data[0], 0.01);

    // Column 1: sum([15,15,15,15,15,15,15,15] * 1) = 8 * 15 = 120
    try std.testing.expectApproxEqAbs(@as(f32, 120.0), out_data[1], 0.01);
}

test "matmulGaffineU4Prefill kernel4x2 vs kernel1x2" {
    // Verify that kernel4x2 and kernel1x2 produce the same results
    const m: usize = 4;
    const k: usize = 8;
    const n: usize = 2;
    const group: usize = 8;
    const k_div_8 = k / 8;
    const k_div_group = k / group;
    const group_u32 = group / 8;

    const allocator = std.testing.allocator;

    // Prepare test data
    const a_data = try allocator.alloc(f32, m * k);
    defer allocator.free(a_data);

    for (0..m) |i| {
        for (0..k) |j| {
            a_data[i * k + j] = @floatFromInt((i + 1) * 10 + (j + 1));
        }
    }

    const packed_vals = try allocator.alloc(u32, n * k_div_8);
    defer allocator.free(packed_vals);
    for (packed_vals, 0..) |*val, i| {
        val.* = @as(u32, @truncate(i)) * 0x11111111;
    }

    var scales = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(scales);
    var biases = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(biases);

    const scale_fp16 = dtype_mod.f32ToFp16(1.5);
    const bias_fp16 = dtype_mod.f32ToFp16(0.25);
    for (0..n * k_div_group) |i| {
        scales[i] = scale_fp16;
        biases[i] = bias_fp16;
    }

    // Create context
    const MatmulU4PrefillCtx = struct {
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
        k_div_8: usize,
        k_div_group: usize,
        group_u32: usize,
    };

    // Test kernel4x2
    const out_4x2 = try allocator.alloc(f32, m * n);
    defer allocator.free(out_4x2);
    @memset(out_4x2, 0);

    var ctx_4x2 = MatmulU4PrefillCtx{
        .a = a_data,
        .packed_b = packed_vals,
        .scales = scales,
        .biases = biases,
        .scales_dtype = .f16,
        .out = out_4x2,
        .m_rows = m,
        .n_cols = n,
        .k_dim = k,
        .group = group,
        .k_div_8 = k_div_8,
        .k_div_group = k_div_group,
        .group_u32 = group_u32,
    };

    kernel4x2(&ctx_4x2, 0, 0);

    // Test kernel1x2 on same rows
    const out_1x2 = try allocator.alloc(f32, m * n);
    defer allocator.free(out_1x2);
    @memset(out_1x2, 0);

    var ctx_1x2 = MatmulU4PrefillCtx{
        .a = a_data,
        .packed_b = packed_vals,
        .scales = scales,
        .biases = biases,
        .scales_dtype = .f16,
        .out = out_1x2,
        .m_rows = m,
        .n_cols = n,
        .k_dim = k,
        .group = group,
        .k_div_8 = k_div_8,
        .k_div_group = k_div_group,
        .group_u32 = group_u32,
    };

    for (0..4) |row| {
        kernel1x2(&ctx_1x2, row, 0);
    }

    // Compare results
    for (0..m) |i| {
        for (0..n) |j| {
            const idx = i * n + j;
            try std.testing.expectApproxEqAbs(out_4x2[idx], out_1x2[idx], 0.001);
        }
    }
}

test "matmulGaffineU4Prefill extractNibbles helper" {
    // Test nibble extraction from a u32
    const word: u32 = 0xABCD1234;
    const result = extractNibbles(word);

    // Expected nibbles (interleaved low/high from each byte):
    // Byte 0 (0x34): low=4, high=3
    // Byte 1 (0x12): low=2, high=1
    // Byte 2 (0xCD): low=13, high=12
    // Byte 3 (0xAB): low=11, high=10
    const expected = [_]f32{ 4, 3, 2, 1, 13, 12, 11, 10 };

    for (expected, 0..) |exp, i| {
        try std.testing.expectEqual(exp, result[i]);
    }
}

test "matmulGaffineU4Prefill extract32NibblesToFloat helper" {
    // Test extraction of 32 nibbles from 4 u32s
    const data = [_]u32{ 0x03020100, 0x07060504, 0x0B0A0908, 0x0F0E0D0C };
    const result = extract32NibblesToFloat(&data);

    // Each byte has nibbles: low=X, high=X
    // Verify first few values of each vector
    try std.testing.expectEqual(@as(f32, 0.0), result.n0[0]);
    try std.testing.expectEqual(@as(f32, 0.0), result.n0[1]);
    try std.testing.expectEqual(@as(f32, 1.0), result.n0[2]);
    try std.testing.expectEqual(@as(f32, 0.0), result.n0[3]);
}

test "matmulGaffineU8Prefill extractBytes helper" {
    // Test byte extraction from u32
    const word: u32 = 0x0A0B0C0D;
    const result = extractBytes(word);

    try std.testing.expectEqual(@as(f32, 0x0D), result[0]);
    try std.testing.expectEqual(@as(f32, 0x0C), result[1]);
    try std.testing.expectEqual(@as(f32, 0x0B), result[2]);
    try std.testing.expectEqual(@as(f32, 0x0A), result[3]);
}
