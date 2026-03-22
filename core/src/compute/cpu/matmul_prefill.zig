//! Multi-row GEMM kernels for grouped-affine quantized weights (AVX2-optimized)
//!
//! Panel dequant strategy: parallelize over 4-column tiles. For each tile,
//! dequant weight columns in k-tiles to a 4KB f32 scratch buffer (L1-resident),
//! then run a pure f32 4×4 microkernel across ALL rows. Scratch stays hot in
//! L1 across every row — matching BF16's tiling pattern exactly.

const std = @import("std");
const parallel = @import("../../system/parallel.zig");
const matmul = @import("matmul_primitives.zig");
const simd = @import("simd/arch/root.zig");
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

const VEC = simd.f32_vec_len; // 8 on AVX2

// =============================================================================
// 4-bit Prefill Entry Point — Panel Dequant
// =============================================================================
//
// Strategy: parallelize over col_quads (4 columns). Each task dequants weight
// columns in k-tiles to a 4KB f32 scratch buffer (L1-resident), then runs a
// pure f32 4×4 microkernel across ALL rows. Dequant is paid once per k-tile
// and reused across every row — matching BF16's tiling pattern exactly.

const ROW_BLOCK_SIZE: usize = 64; // Used by U8 path

// K-tile size for dequant scratch. 4 cols × K_TILE × 4 bytes on stack.
// 2048 → 32KB scratch, fits L1 (48KB on Zen 5). Covers most k_dims in
// a single tile, eliminating reduction + accumulation overhead.
const K_TILE: usize = 2048;

// Column tile size for parallelization. Each task processes COL_TILE columns,
// dequanting 4 at a time into scratch. Matches BF16's 16-col tile to get the
// same task count and avoid false sharing (16 f32s = 64 bytes = 1 cache line).
const COL_TILE: usize = 16;

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

    const num_col_tiles = n_cols / COL_TILE;

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

    // Parallelize over COL_TILE-sized tiles (16 columns each), matching BF16's
    // task granularity. Each task writes 16 f32s/row = 64 bytes = 1 cache line,
    // eliminating false sharing between tasks. Within each tile, dequant 4 cols
    // at a time into L1-resident scratch and run the 4×4 FMA across all rows.
    const tile_task = struct {
        fn runColTiles(start: usize, end: usize, task_ctx: *MatmulU4PrefillCtx) void {
            @setFloatMode(.optimized);
            const k = task_ctx.k_dim;
            const n = task_ctx.n_cols;
            const m = task_ctx.m_rows;
            const grp = task_ctx.group;
            const k_dg = task_ctx.k_div_group;

            // Stack scratch: 4 columns × K_TILE f32 values (32KB at K_TILE=2048)
            var scratch: [4 * K_TILE]f32 align(64) = undefined;

            for (start..end) |tile_idx| {
                const tile_col_start = tile_idx * COL_TILE;
                const tile_col_end = @min(tile_col_start + COL_TILE, n);

                // Process col-quads within this tile
                var col_base = tile_col_start;
                while (col_base + 3 < tile_col_end) : (col_base += 4) {
                    // Weight and scale/bias pointers for 4 columns
                    const w_ptrs = [4][*]align(1) const u32{
                        task_ctx.packed_b.ptr + col_base * task_ctx.k_div_8,
                        task_ctx.packed_b.ptr + (col_base + 1) * task_ctx.k_div_8,
                        task_ctx.packed_b.ptr + (col_base + 2) * task_ctx.k_div_8,
                        task_ctx.packed_b.ptr + (col_base + 3) * task_ctx.k_div_8,
                    };
                    const s_ptrs = [4][*]align(1) const u16{
                        task_ctx.scales.ptr + col_base * k_dg,
                        task_ctx.scales.ptr + (col_base + 1) * k_dg,
                        task_ctx.scales.ptr + (col_base + 2) * k_dg,
                        task_ctx.scales.ptr + (col_base + 3) * k_dg,
                    };
                    const b_ptrs = [4][*]align(1) const u16{
                        task_ctx.biases.ptr + col_base * k_dg,
                        task_ctx.biases.ptr + (col_base + 1) * k_dg,
                        task_ctx.biases.ptr + (col_base + 2) * k_dg,
                        task_ctx.biases.ptr + (col_base + 3) * k_dg,
                    };

                    // Zero output cells (kernel uses +=)
                    for (0..m) |r| {
                        const out_row = task_ctx.out.ptr + r * n;
                        out_row[col_base] = 0;
                        out_row[col_base + 1] = 0;
                        out_row[col_base + 2] = 0;
                        out_row[col_base + 3] = 0;
                    }

                    // K-tile loop: dequant once per tile, reuse scratch across
                    // ALL rows. Scratch (32KB) fits L1.
                    var k_start: usize = 0;
                    while (k_start < k) : (k_start += K_TILE) {
                        const k_end = @min(k_start + K_TILE, k);
                        const k_len = k_end - k_start;

                        // Phase 1: Dequant 4 cols × k_len → scratch
                        dequantU4Panel(
                            &w_ptrs,
                            &s_ptrs,
                            &b_ptrs,
                            task_ctx.scales_dtype,
                            &scratch,
                            k_start,
                            k_len,
                            grp,
                            task_ctx.group_u32,
                            k_dg,
                        );

                        // Phase 2: 4×4 FMA across ALL rows
                        var row_idx: usize = 0;
                        while (row_idx + 3 < m) : (row_idx += 4) {
                            panelKernel4x4(task_ctx.a.ptr, &scratch, task_ctx.out.ptr, row_idx, col_base, k_start, k_len, k, n);
                        }
                        while (row_idx < m) : (row_idx += 1) {
                            panelKernel1x4(task_ctx.a.ptr, &scratch, task_ctx.out.ptr, row_idx, col_base, k_start, k_len, k, n);
                        }
                    }
                }
            }
        }
    }.runColTiles;

    if (num_col_tiles > 0) {
        parallel.global().parallelFor(num_col_tiles, tile_task, &context);
    }

    // Handle remainder columns (n_cols % COL_TILE) with existing kernels
    const remainder_start = num_col_tiles * COL_TILE;
    if (remainder_start < n_cols) {
        // Process col-quads in remainder
        var col_base = remainder_start;
        while (col_base + 3 < n_cols) : (col_base += 4) {
            for (0..m_rows) |row_idx| {
                kernel1x2(&context, row_idx, col_base);
                kernel1x2(&context, row_idx, col_base + 2);
            }
        }
        // Process pairs
        while (col_base + 1 < n_cols) : (col_base += 2) {
            for (0..m_rows) |row_idx| {
                kernel1x2(&context, row_idx, col_base);
            }
        }
        // Odd final column
        if (col_base < n_cols) {
            for (0..m_rows) |row_idx| {
                kernel1x1(&context, row_idx, n_cols - 1);
            }
        }
    }
}

// =============================================================================
// Panel Dequant: U4 weights → f32 scratch buffer
// =============================================================================
// Dequantizes 4 columns × k_len elements from packed U4 into scratch[0..4*k_len].
// Layout: scratch[col * K_TILE + k_offset] for col in 0..4.

inline fn dequantU4Panel(
    w_ptrs: *const [4][*]align(1) const u32,
    s_ptrs: *const [4][*]align(1) const u16,
    b_ptrs: *const [4][*]align(1) const u16,
    scales_dtype: DType,
    scratch: *[4 * K_TILE]f32,
    k_start: usize,
    k_len: usize,
    group: usize,
    _: usize, // group_u32 (unused, kept for call-site clarity)
    _: usize, // k_div_group (unused)
) void {
    @setFloatMode(.optimized);

    // k_start in units of u32 packs (8 nibbles per u32 for U4)
    const pack_start = k_start / 8;

    inline for (0..4) |col| {
        const w_base = w_ptrs[col] + pack_start;
        var out_idx: usize = 0;

        // Walk through the k_len elements, respecting group boundaries for scale/bias
        var k_pos = k_start;
        while (k_pos < k_start + k_len) {
            const group_idx = k_pos / group;
            const s: @Vector(VEC, f32) = @splat(scaleBiasToF32(scales_dtype, s_ptrs[col][group_idx]));
            const b: @Vector(VEC, f32) = @splat(scaleBiasToF32(scales_dtype, b_ptrs[col][group_idx]));

            // How many elements left in this group?
            const group_end = (group_idx + 1) * group;
            const tile_end = k_start + k_len;
            const chunk_end = @min(group_end, tile_end);
            const chunk_len = chunk_end - k_pos;

            // Process 32 elements at a time (4 u32s)
            const w_off = (k_pos - k_start) / 8;
            const w_ptr = w_base + w_off;
            var elem: usize = 0;
            while (elem + 32 <= chunk_len) : (elem += 32) {
                const pack_off = elem / 8;
                const nibs = extract32NibblesToFloat(w_ptr + pack_off);
                const s_out = scratch[col * K_TILE + out_idx ..];
                s_out[0..VEC].* = @mulAdd(@Vector(VEC, f32), nibs.n0, s, b);
                s_out[VEC..][0..VEC].* = @mulAdd(@Vector(VEC, f32), nibs.n1, s, b);
                s_out[2 * VEC ..][0..VEC].* = @mulAdd(@Vector(VEC, f32), nibs.n2, s, b);
                s_out[3 * VEC ..][0..VEC].* = @mulAdd(@Vector(VEC, f32), nibs.n3, s, b);
                out_idx += 32;
            }

            // Remainder within group (< 32 elements, 8 at a time)
            while (elem + 8 <= chunk_len) : (elem += 8) {
                const pack_off = elem / 8;
                const nibs_vec = extractNibbles(w_ptr[pack_off]);
                scratch[col * K_TILE + out_idx ..][0..VEC].* = @mulAdd(@Vector(VEC, f32), nibs_vec, s, b);
                out_idx += 8;
            }

            k_pos += chunk_len;
        }
    }
}

// =============================================================================
// Pure f32 4×4 Microkernel (operates on pre-dequanted scratch)
// =============================================================================
// Identical structure to BF16 4×4: 16 accumulators, 4 activation loads,
// 4 weight loads from L1-hot scratch. Zero dequant overhead in the hot loop.

inline fn panelKernel4x4(
    a_ptr: [*]const f32,
    scratch: *const [4 * K_TILE]f32,
    out_ptr: [*]f32,
    row_base: usize,
    col_base: usize,
    k_start: usize,
    k_len: usize,
    k_dim: usize,
    n_dim: usize,
) void {
    @setFloatMode(.optimized);

    const a0 = a_ptr + row_base * k_dim + k_start;
    const a1 = a_ptr + (row_base + 1) * k_dim + k_start;
    const a2 = a_ptr + (row_base + 2) * k_dim + k_start;
    const a3 = a_ptr + (row_base + 3) * k_dim + k_start;

    const w0: [*]const f32 = @ptrCast(&scratch[0 * K_TILE]);
    const w1: [*]const f32 = @ptrCast(&scratch[1 * K_TILE]);
    const w2: [*]const f32 = @ptrCast(&scratch[2 * K_TILE]);
    const w3: [*]const f32 = @ptrCast(&scratch[3 * K_TILE]);

    var acc00: @Vector(VEC, f32) = @splat(0);
    var acc01: @Vector(VEC, f32) = @splat(0);
    var acc02: @Vector(VEC, f32) = @splat(0);
    var acc03: @Vector(VEC, f32) = @splat(0);
    var acc10: @Vector(VEC, f32) = @splat(0);
    var acc11: @Vector(VEC, f32) = @splat(0);
    var acc12: @Vector(VEC, f32) = @splat(0);
    var acc13: @Vector(VEC, f32) = @splat(0);
    var acc20: @Vector(VEC, f32) = @splat(0);
    var acc21: @Vector(VEC, f32) = @splat(0);
    var acc22: @Vector(VEC, f32) = @splat(0);
    var acc23: @Vector(VEC, f32) = @splat(0);
    var acc30: @Vector(VEC, f32) = @splat(0);
    var acc31: @Vector(VEC, f32) = @splat(0);
    var acc32: @Vector(VEC, f32) = @splat(0);
    var acc33: @Vector(VEC, f32) = @splat(0);

    var ki: usize = 0;
    while (ki + VEC <= k_len) : (ki += VEC) {
        const av0: @Vector(VEC, f32) = a0[ki..][0..VEC].*;
        const av1: @Vector(VEC, f32) = a1[ki..][0..VEC].*;
        const av2: @Vector(VEC, f32) = a2[ki..][0..VEC].*;
        const av3: @Vector(VEC, f32) = a3[ki..][0..VEC].*;

        const wv0: @Vector(VEC, f32) = w0[ki..][0..VEC].*;
        const wv1: @Vector(VEC, f32) = w1[ki..][0..VEC].*;
        const wv2: @Vector(VEC, f32) = w2[ki..][0..VEC].*;
        const wv3: @Vector(VEC, f32) = w3[ki..][0..VEC].*;

        acc00 = @mulAdd(@Vector(VEC, f32), av0, wv0, acc00);
        acc01 = @mulAdd(@Vector(VEC, f32), av0, wv1, acc01);
        acc02 = @mulAdd(@Vector(VEC, f32), av0, wv2, acc02);
        acc03 = @mulAdd(@Vector(VEC, f32), av0, wv3, acc03);
        acc10 = @mulAdd(@Vector(VEC, f32), av1, wv0, acc10);
        acc11 = @mulAdd(@Vector(VEC, f32), av1, wv1, acc11);
        acc12 = @mulAdd(@Vector(VEC, f32), av1, wv2, acc12);
        acc13 = @mulAdd(@Vector(VEC, f32), av1, wv3, acc13);
        acc20 = @mulAdd(@Vector(VEC, f32), av2, wv0, acc20);
        acc21 = @mulAdd(@Vector(VEC, f32), av2, wv1, acc21);
        acc22 = @mulAdd(@Vector(VEC, f32), av2, wv2, acc22);
        acc23 = @mulAdd(@Vector(VEC, f32), av2, wv3, acc23);
        acc30 = @mulAdd(@Vector(VEC, f32), av3, wv0, acc30);
        acc31 = @mulAdd(@Vector(VEC, f32), av3, wv1, acc31);
        acc32 = @mulAdd(@Vector(VEC, f32), av3, wv2, acc32);
        acc33 = @mulAdd(@Vector(VEC, f32), av3, wv3, acc33);
    }

    out_ptr[row_base * n_dim + col_base] += @reduce(.Add, acc00);
    out_ptr[row_base * n_dim + col_base + 1] += @reduce(.Add, acc01);
    out_ptr[row_base * n_dim + col_base + 2] += @reduce(.Add, acc02);
    out_ptr[row_base * n_dim + col_base + 3] += @reduce(.Add, acc03);
    out_ptr[(row_base + 1) * n_dim + col_base] += @reduce(.Add, acc10);
    out_ptr[(row_base + 1) * n_dim + col_base + 1] += @reduce(.Add, acc11);
    out_ptr[(row_base + 1) * n_dim + col_base + 2] += @reduce(.Add, acc12);
    out_ptr[(row_base + 1) * n_dim + col_base + 3] += @reduce(.Add, acc13);
    out_ptr[(row_base + 2) * n_dim + col_base] += @reduce(.Add, acc20);
    out_ptr[(row_base + 2) * n_dim + col_base + 1] += @reduce(.Add, acc21);
    out_ptr[(row_base + 2) * n_dim + col_base + 2] += @reduce(.Add, acc22);
    out_ptr[(row_base + 2) * n_dim + col_base + 3] += @reduce(.Add, acc23);
    out_ptr[(row_base + 3) * n_dim + col_base] += @reduce(.Add, acc30);
    out_ptr[(row_base + 3) * n_dim + col_base + 1] += @reduce(.Add, acc31);
    out_ptr[(row_base + 3) * n_dim + col_base + 2] += @reduce(.Add, acc32);
    out_ptr[(row_base + 3) * n_dim + col_base + 3] += @reduce(.Add, acc33);
}

// 1×4 remainder kernel (single row, 4 columns from scratch)
inline fn panelKernel1x4(
    a_ptr: [*]const f32,
    scratch: *const [4 * K_TILE]f32,
    out_ptr: [*]f32,
    row: usize,
    col_base: usize,
    k_start: usize,
    k_len: usize,
    k_dim: usize,
    n_dim: usize,
) void {
    @setFloatMode(.optimized);

    const a = a_ptr + row * k_dim + k_start;
    const w0: [*]const f32 = @ptrCast(&scratch[0 * K_TILE]);
    const w1: [*]const f32 = @ptrCast(&scratch[1 * K_TILE]);
    const w2: [*]const f32 = @ptrCast(&scratch[2 * K_TILE]);
    const w3: [*]const f32 = @ptrCast(&scratch[3 * K_TILE]);

    var acc0: @Vector(VEC, f32) = @splat(0);
    var acc1: @Vector(VEC, f32) = @splat(0);
    var acc2: @Vector(VEC, f32) = @splat(0);
    var acc3: @Vector(VEC, f32) = @splat(0);

    var ki: usize = 0;
    while (ki + VEC <= k_len) : (ki += VEC) {
        const av: @Vector(VEC, f32) = a[ki..][0..VEC].*;
        acc0 = @mulAdd(@Vector(VEC, f32), av, w0[ki..][0..VEC].*, acc0);
        acc1 = @mulAdd(@Vector(VEC, f32), av, w1[ki..][0..VEC].*, acc1);
        acc2 = @mulAdd(@Vector(VEC, f32), av, w2[ki..][0..VEC].*, acc2);
        acc3 = @mulAdd(@Vector(VEC, f32), av, w3[ki..][0..VEC].*, acc3);
    }

    out_ptr[row * n_dim + col_base] += @reduce(.Add, acc0);
    out_ptr[row * n_dim + col_base + 1] += @reduce(.Add, acc1);
    out_ptr[row * n_dim + col_base + 2] += @reduce(.Add, acc2);
    out_ptr[row * n_dim + col_base + 3] += @reduce(.Add, acc3);
}

// =============================================================================
// 1x2 Kernel (for remainder columns < 4)
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

    var acc0: @Vector(8, f32) = @splat(0);
    var acc1: @Vector(8, f32) = @splat(0);

    var group_idx: usize = 0;
    while (group_idx < k_div_group) : (group_idx += 1) {
        const s0: @Vector(8, f32) = @splat(scaleBiasToF32(context.scales_dtype, scale0_base[group_idx]));
        const s1: @Vector(8, f32) = @splat(scaleBiasToF32(context.scales_dtype, scale1_base[group_idx]));
        const b0: @Vector(8, f32) = @splat(scaleBiasToF32(context.scales_dtype, bias0_base[group_idx]));
        const b1: @Vector(8, f32) = @splat(scaleBiasToF32(context.scales_dtype, bias1_base[group_idx]));

        const w0_ptr = w0_base + group_idx * group_u32;
        const w1_ptr = w1_base + group_idx * group_u32;
        const a_ptr = a_base + group_idx * group;

        var pack_idx: usize = 0;
        while (pack_idx + 3 < group_u32) : (pack_idx += 4) {
            const nibs0 = extract32NibblesToFloat(w0_ptr + pack_idx);
            const dw0_0 = @mulAdd(@Vector(8, f32), nibs0.n0, s0, b0);
            const dw0_1 = @mulAdd(@Vector(8, f32), nibs0.n1, s0, b0);
            const dw0_2 = @mulAdd(@Vector(8, f32), nibs0.n2, s0, b0);
            const dw0_3 = @mulAdd(@Vector(8, f32), nibs0.n3, s0, b0);

            const nibs1 = extract32NibblesToFloat(w1_ptr + pack_idx);
            const dw1_0 = @mulAdd(@Vector(8, f32), nibs1.n0, s1, b1);
            const dw1_1 = @mulAdd(@Vector(8, f32), nibs1.n1, s1, b1);
            const dw1_2 = @mulAdd(@Vector(8, f32), nibs1.n2, s1, b1);
            const dw1_3 = @mulAdd(@Vector(8, f32), nibs1.n3, s1, b1);

            const x0: @Vector(8, f32) = (a_ptr + pack_idx * 8)[0..8].*;
            const x1: @Vector(8, f32) = (a_ptr + (pack_idx + 1) * 8)[0..8].*;
            const x2: @Vector(8, f32) = (a_ptr + (pack_idx + 2) * 8)[0..8].*;
            const x3: @Vector(8, f32) = (a_ptr + (pack_idx + 3) * 8)[0..8].*;

            acc0 = @mulAdd(@Vector(8, f32), dw0_0, x0, acc0);
            acc0 = @mulAdd(@Vector(8, f32), dw0_1, x1, acc0);
            acc0 = @mulAdd(@Vector(8, f32), dw0_2, x2, acc0);
            acc0 = @mulAdd(@Vector(8, f32), dw0_3, x3, acc0);

            acc1 = @mulAdd(@Vector(8, f32), dw1_0, x0, acc1);
            acc1 = @mulAdd(@Vector(8, f32), dw1_1, x1, acc1);
            acc1 = @mulAdd(@Vector(8, f32), dw1_2, x2, acc1);
            acc1 = @mulAdd(@Vector(8, f32), dw1_3, x3, acc1);
        }

        while (pack_idx < group_u32) : (pack_idx += 1) {
            const dw0 = @mulAdd(@Vector(8, f32), extractNibbles(w0_ptr[pack_idx]), s0, b0);
            const dw1 = @mulAdd(@Vector(8, f32), extractNibbles(w1_ptr[pack_idx]), s1, b1);
            const x_vec: @Vector(8, f32) = (a_ptr + pack_idx * 8)[0..8].*;

            acc0 = @mulAdd(@Vector(8, f32), dw0, x_vec, acc0);
            acc1 = @mulAdd(@Vector(8, f32), dw1, x_vec, acc1);
        }
    }

    context.out[row_idx * n_dim + col_idx] = @reduce(.Add, acc0);
    context.out[row_idx * n_dim + col_idx + 1] = @reduce(.Add, acc1);
}

// =============================================================================
// 1x1 kernel (for odd column at end) - reuses single-row optimized dot product
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

    // Pre-convert scales/biases (same as single-row kernel)
    var scales_f32: [matmul.MAX_GROUPS]f32 align(64) = undefined;
    var biases_f32: [matmul.MAX_GROUPS]f32 align(64) = undefined;

    for (0..k_div_group) |group_idx| {
        scales_f32[group_idx] = scaleBiasToF32(context.scales_dtype, scale_ptr[group_idx]);
        biases_f32[group_idx] = scaleBiasToF32(context.scales_dtype, bias_ptr[group_idx]);
    }

    // Reuse single-row optimized dot product
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
        num_col_pairs: usize,
    };

    // 2D tiling: row_blocks × col_pairs flattened into a 1D task stream.
    const num_col_pairs = n_cols / 2;
    const row_blocks = (m_rows + ROW_BLOCK_SIZE - 1) / ROW_BLOCK_SIZE;
    const total_tiles = row_blocks * num_col_pairs;

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
        .num_col_pairs = num_col_pairs,
    };

    const tile_task = struct {
        fn run2DTiles(start: usize, end: usize, task_ctx: *MatmulU8PrefillCtx) void {
            @setFloatMode(.optimized);

            for (start..end) |tile_idx| {
                const rb = tile_idx / task_ctx.num_col_pairs;
                const col_pair = tile_idx % task_ctx.num_col_pairs;
                const row_start = rb * ROW_BLOCK_SIZE;
                const row_end = @min(row_start + ROW_BLOCK_SIZE, task_ctx.m_rows);
                const col = col_pair * 2;

                // Process rows in groups of 4
                var row = row_start;
                while (row + 3 < row_end) : (row += 4) {
                    kernel4x2_8bit(task_ctx, row, col);
                }
                // Remainder rows
                while (row < row_end) : (row += 1) {
                    kernel1x2_8bit(task_ctx, row, col);
                }
            }
        }
    }.run2DTiles;

    parallel.global().parallelFor(total_tiles, tile_task, &context);

    // Handle odd column at end
    if (n_cols % 2 == 1) {
        const col = n_cols - 1;
        for (0..m_rows) |row| {
            kernel1x1_8bit(&context, row, col);
        }
    }
}

/// 4x2 kernel for 8-bit: inline pre-dequant, same approach as U4
fn kernel4x2_8bit(context: anytype, row_base_idx: usize, col_idx: usize) void {
    @setFloatMode(.optimized);

    const k_dim = context.k_dim;
    const n_dim = context.n_cols;
    const group = context.group;
    const k_div_group = context.k_div_group;
    const k_div_4 = context.k_div_4;
    const group_u32 = context.group_u32;

    const w0_base = context.packed_b.ptr + col_idx * k_div_4;
    const w1_base = context.packed_b.ptr + (col_idx + 1) * k_div_4;

    const a0_base = context.a.ptr + row_base_idx * k_dim;
    const a1_base = context.a.ptr + (row_base_idx + 1) * k_dim;
    const a2_base = context.a.ptr + (row_base_idx + 2) * k_dim;
    const a3_base = context.a.ptr + (row_base_idx + 3) * k_dim;

    const scale0_base = context.scales.ptr + col_idx * k_div_group;
    const scale1_base = context.scales.ptr + (col_idx + 1) * k_div_group;
    const bias0_base = context.biases.ptr + col_idx * k_div_group;
    const bias1_base = context.biases.ptr + (col_idx + 1) * k_div_group;

    var acc00: @Vector(8, f32) = @splat(0);
    var acc01: @Vector(8, f32) = @splat(0);
    var acc10: @Vector(8, f32) = @splat(0);
    var acc11: @Vector(8, f32) = @splat(0);
    var acc20: @Vector(8, f32) = @splat(0);
    var acc21: @Vector(8, f32) = @splat(0);
    var acc30: @Vector(8, f32) = @splat(0);
    var acc31: @Vector(8, f32) = @splat(0);

    var group_idx: usize = 0;
    while (group_idx < k_div_group) : (group_idx += 1) {
        const s0: @Vector(8, f32) = @splat(scaleBiasToF32(context.scales_dtype, scale0_base[group_idx]));
        const s1: @Vector(8, f32) = @splat(scaleBiasToF32(context.scales_dtype, scale1_base[group_idx]));
        const b0: @Vector(8, f32) = @splat(scaleBiasToF32(context.scales_dtype, bias0_base[group_idx]));
        const b1: @Vector(8, f32) = @splat(scaleBiasToF32(context.scales_dtype, bias1_base[group_idx]));

        const w0_ptr = w0_base + group_idx * group_u32;
        const w1_ptr = w1_base + group_idx * group_u32;
        const a0_ptr = a0_base + group_idx * group;
        const a1_ptr = a1_base + group_idx * group;
        const a2_ptr = a2_base + group_idx * group;
        const a3_ptr = a3_base + group_idx * group;

        var pack_idx: usize = 0;
        while (pack_idx + 1 < group_u32) : (pack_idx += 2) {
            const dw0 = @mulAdd(@Vector(8, f32), extract8BytesToFloat(w0_ptr + pack_idx), s0, b0);
            const dw1 = @mulAdd(@Vector(8, f32), extract8BytesToFloat(w1_ptr + pack_idx), s1, b1);

            const x0: @Vector(8, f32) = (a0_ptr + pack_idx * 4)[0..8].*;
            const x1: @Vector(8, f32) = (a1_ptr + pack_idx * 4)[0..8].*;
            const x2: @Vector(8, f32) = (a2_ptr + pack_idx * 4)[0..8].*;
            const x3: @Vector(8, f32) = (a3_ptr + pack_idx * 4)[0..8].*;

            acc00 = @mulAdd(@Vector(8, f32), dw0, x0, acc00);
            acc01 = @mulAdd(@Vector(8, f32), dw1, x0, acc01);
            acc10 = @mulAdd(@Vector(8, f32), dw0, x1, acc10);
            acc11 = @mulAdd(@Vector(8, f32), dw1, x1, acc11);
            acc20 = @mulAdd(@Vector(8, f32), dw0, x2, acc20);
            acc21 = @mulAdd(@Vector(8, f32), dw1, x2, acc21);
            acc30 = @mulAdd(@Vector(8, f32), dw0, x3, acc30);
            acc31 = @mulAdd(@Vector(8, f32), dw1, x3, acc31);
        }

        // Remainder (odd u32)
        while (pack_idx < group_u32) : (pack_idx += 1) {
            const w0_raw = extractBytes(w0_ptr[pack_idx]);
            const w1_raw = extractBytes(w1_ptr[pack_idx]);
            const s0_4: @Vector(4, f32) = @splat(s0[0]);
            const s1_4: @Vector(4, f32) = @splat(s1[0]);
            const b0_4: @Vector(4, f32) = @splat(b0[0]);
            const b1_4: @Vector(4, f32) = @splat(b1[0]);
            const dw0_4 = @mulAdd(@Vector(4, f32), w0_raw, s0_4, b0_4);
            const dw1_4 = @mulAdd(@Vector(4, f32), w1_raw, s1_4, b1_4);

            const x0: @Vector(4, f32) = (a0_ptr + pack_idx * 4)[0..4].*;
            const x1: @Vector(4, f32) = (a1_ptr + pack_idx * 4)[0..4].*;
            const x2: @Vector(4, f32) = (a2_ptr + pack_idx * 4)[0..4].*;
            const x3: @Vector(4, f32) = (a3_ptr + pack_idx * 4)[0..4].*;

            const dw0_8: @Vector(8, f32) = .{ dw0_4[0], dw0_4[1], dw0_4[2], dw0_4[3], 0, 0, 0, 0 };
            const dw1_8: @Vector(8, f32) = .{ dw1_4[0], dw1_4[1], dw1_4[2], dw1_4[3], 0, 0, 0, 0 };
            const x0_8: @Vector(8, f32) = .{ x0[0], x0[1], x0[2], x0[3], 0, 0, 0, 0 };
            const x1_8: @Vector(8, f32) = .{ x1[0], x1[1], x1[2], x1[3], 0, 0, 0, 0 };
            const x2_8: @Vector(8, f32) = .{ x2[0], x2[1], x2[2], x2[3], 0, 0, 0, 0 };
            const x3_8: @Vector(8, f32) = .{ x3[0], x3[1], x3[2], x3[3], 0, 0, 0, 0 };

            acc00 = @mulAdd(@Vector(8, f32), dw0_8, x0_8, acc00);
            acc01 = @mulAdd(@Vector(8, f32), dw1_8, x0_8, acc01);
            acc10 = @mulAdd(@Vector(8, f32), dw0_8, x1_8, acc10);
            acc11 = @mulAdd(@Vector(8, f32), dw1_8, x1_8, acc11);
            acc20 = @mulAdd(@Vector(8, f32), dw0_8, x2_8, acc20);
            acc21 = @mulAdd(@Vector(8, f32), dw1_8, x2_8, acc21);
            acc30 = @mulAdd(@Vector(8, f32), dw0_8, x3_8, acc30);
            acc31 = @mulAdd(@Vector(8, f32), dw1_8, x3_8, acc31);
        }
    }

    context.out[row_base_idx * n_dim + col_idx] = @reduce(.Add, acc00);
    context.out[row_base_idx * n_dim + col_idx + 1] = @reduce(.Add, acc01);
    context.out[(row_base_idx + 1) * n_dim + col_idx] = @reduce(.Add, acc10);
    context.out[(row_base_idx + 1) * n_dim + col_idx + 1] = @reduce(.Add, acc11);
    context.out[(row_base_idx + 2) * n_dim + col_idx] = @reduce(.Add, acc20);
    context.out[(row_base_idx + 2) * n_dim + col_idx + 1] = @reduce(.Add, acc21);
    context.out[(row_base_idx + 3) * n_dim + col_idx] = @reduce(.Add, acc30);
    context.out[(row_base_idx + 3) * n_dim + col_idx + 1] = @reduce(.Add, acc31);
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

    var acc0: @Vector(8, f32) = @splat(0);
    var acc1: @Vector(8, f32) = @splat(0);

    var group_idx: usize = 0;
    while (group_idx < k_div_group) : (group_idx += 1) {
        const s0: @Vector(8, f32) = @splat(scaleBiasToF32(context.scales_dtype, scale0_base[group_idx]));
        const s1: @Vector(8, f32) = @splat(scaleBiasToF32(context.scales_dtype, scale1_base[group_idx]));
        const b0: @Vector(8, f32) = @splat(scaleBiasToF32(context.scales_dtype, bias0_base[group_idx]));
        const b1: @Vector(8, f32) = @splat(scaleBiasToF32(context.scales_dtype, bias1_base[group_idx]));

        const w0_ptr = w0_base + group_idx * group_u32;
        const w1_ptr = w1_base + group_idx * group_u32;
        const a_ptr = a_base + group_idx * group;

        var pack_idx: usize = 0;
        while (pack_idx + 1 < group_u32) : (pack_idx += 2) {
            const dw0 = @mulAdd(@Vector(8, f32), extract8BytesToFloat(w0_ptr + pack_idx), s0, b0);
            const dw1 = @mulAdd(@Vector(8, f32), extract8BytesToFloat(w1_ptr + pack_idx), s1, b1);
            const x_vec: @Vector(8, f32) = (a_ptr + pack_idx * 4)[0..8].*;

            acc0 = @mulAdd(@Vector(8, f32), dw0, x_vec, acc0);
            acc1 = @mulAdd(@Vector(8, f32), dw1, x_vec, acc1);
        }

        while (pack_idx < group_u32) : (pack_idx += 1) {
            const s0_4: @Vector(4, f32) = @splat(s0[0]);
            const s1_4: @Vector(4, f32) = @splat(s1[0]);
            const b0_4: @Vector(4, f32) = @splat(b0[0]);
            const b1_4: @Vector(4, f32) = @splat(b1[0]);
            const dw0 = @mulAdd(@Vector(4, f32), extractBytes(w0_ptr[pack_idx]), s0_4, b0_4);
            const dw1 = @mulAdd(@Vector(4, f32), extractBytes(w1_ptr[pack_idx]), s1_4, b1_4);
            const x_vec: @Vector(4, f32) = (a_ptr + pack_idx * 4)[0..4].*;

            const dw0_8: @Vector(8, f32) = .{ dw0[0], dw0[1], dw0[2], dw0[3], 0, 0, 0, 0 };
            const dw1_8: @Vector(8, f32) = .{ dw1[0], dw1[1], dw1[2], dw1[3], 0, 0, 0, 0 };
            const x_8: @Vector(8, f32) = .{ x_vec[0], x_vec[1], x_vec[2], x_vec[3], 0, 0, 0, 0 };

            acc0 = @mulAdd(@Vector(8, f32), dw0_8, x_8, acc0);
            acc1 = @mulAdd(@Vector(8, f32), dw1_8, x_8, acc1);
        }
    }

    context.out[row_idx * n_dim + col_idx] = @reduce(.Add, acc0);
    context.out[row_idx * n_dim + col_idx + 1] = @reduce(.Add, acc1);
}

/// 1x1 kernel for 8-bit (odd column at end) - reuses single-row optimized dot product
inline fn kernel1x1_8bit(context: anytype, row_idx: usize, col_idx: usize) void {
    const k_dim = context.k_dim;
    const n_dim = context.n_cols;
    const k_div_group = context.k_div_group;
    const k_div_4 = context.k_div_4;

    const w_ptr = context.packed_b.ptr + col_idx * k_div_4;
    const a_ptr = context.a.ptr + row_idx * k_dim;
    const scale_ptr = context.scales.ptr + col_idx * k_div_group;
    const bias_ptr = context.biases.ptr + col_idx * k_div_group;

    // Pre-convert scales/biases (same as single-row kernel)
    var scales_f32: [matmul.MAX_GROUPS]f32 align(64) = undefined;
    var biases_f32: [matmul.MAX_GROUPS]f32 align(64) = undefined;

    for (0..k_div_group) |group_idx| {
        scales_f32[group_idx] = scaleBiasToF32(context.scales_dtype, scale_ptr[group_idx]);
        biases_f32[group_idx] = scaleBiasToF32(context.scales_dtype, bias_ptr[group_idx]);
    }

    // Reuse single-row optimized dot product
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

test "matmulGaffineU4Prefill panel vs scalar consistency" {
    // Verify that the panel path (4-col tiles) produces the same results as
    // the scalar kernel1x2 remainder path.
    const m: usize = 4;
    const k: usize = 64; // Must be >= K_TILE-compatible and multiple of group
    const n: usize = 4; // Exactly one col_quad
    const group: usize = 64;
    const k_div_8 = k / 8;
    const k_div_group = k / group;

    const allocator = std.testing.allocator;

    const a_data = try allocator.alloc(f32, m * k);
    defer allocator.free(a_data);
    for (0..m) |i| {
        for (0..k) |j| {
            a_data[i * k + j] = @floatFromInt((i + 1) * 10 + (j % 8 + 1));
        }
    }

    const packed_vals = try allocator.alloc(u32, n * k_div_8);
    defer allocator.free(packed_vals);
    for (packed_vals, 0..) |*val, i| {
        val.* = @as(u32, @truncate(i % 16)) * 0x01010101 + 0x02020202;
    }

    var scales_data = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(scales_data);
    var biases_data = try allocator.alloc(u16, n * k_div_group);
    defer allocator.free(biases_data);

    const scale_fp16 = dtype_mod.f32ToFp16(1.5);
    const bias_fp16 = dtype_mod.f32ToFp16(0.25);
    for (0..n * k_div_group) |i| {
        scales_data[i] = scale_fp16;
        biases_data[i] = bias_fp16;
    }

    // Test via full entry point (uses panel path for cols 0-3)
    const out_panel = try allocator.alloc(f32, m * n);
    defer allocator.free(out_panel);
    @memset(out_panel, 0);

    matmulGaffineU4Prefill(a_data, m, k, packed_vals, scales_data, biases_data, .f16, n, group, out_panel);

    // Reference: compute each element via kernel1x2 (known correct)
    const out_ref = try allocator.alloc(f32, m * n);
    defer allocator.free(out_ref);
    @memset(out_ref, 0);

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

    var ctx_ref = MatmulU4PrefillCtx{
        .a = a_data,
        .packed_b = packed_vals,
        .scales = scales_data,
        .biases = biases_data,
        .scales_dtype = .f16,
        .out = out_ref,
        .m_rows = m,
        .n_cols = n,
        .k_dim = k,
        .group = group,
        .k_div_8 = k_div_8,
        .k_div_group = k_div_group,
        .group_u32 = group / 8,
    };

    // Use kernel1x2 for each pair
    for (0..m) |row| {
        kernel1x2(&ctx_ref, row, 0);
        kernel1x2(&ctx_ref, row, 2);
    }

    // Compare
    for (0..m) |i| {
        for (0..n) |j| {
            const idx = i * n + j;
            try std.testing.expectApproxEqAbs(out_ref[idx], out_panel[idx], 0.01);
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
