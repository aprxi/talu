//! MXFP4 (Microsoft Microscaling FP4) Compute Operations
//!
//! MXFP4 is a 4-bit floating point format with E8M0 scales (one scale per 32 values).
//! This module provides optimized matmul operations for MXFP4 tensors.

const std = @import("std");
const parallel = @import("../parallel.zig");
const log = @import("../../log.zig");

// =============================================================================
// MXFP4 Constants and Helpers
// =============================================================================

/// MXFP4 lookup table for 4-bit FP values
/// Format: sign(1) + exponent(2) + mantissa(1)
/// Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and negatives)
pub const MXFP4_LUT: [16]f32 = .{
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, // positive
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, // negative
};

/// Convert E8M0 scale to float multiplier (branchless)
/// E8M0 is just an 8-bit exponent with implied mantissa of 1.0
/// Returns 2^(e8m0 - 127). For e8m0=0, returns 0.0 (close enough to 2^-127)
pub inline fn e8m0ToScale(e8m0: u8) f32 {
    // Branchless: just shift and bitcast. e8m0=0 gives 0.0 (acceptable approximation)
    const exp_bits = @as(u32, e8m0) << 23;
    return @bitCast(exp_bits);
}

/// Convert bfloat16 to f32 (inline, branchless)
/// BF16 is just f32 with lower 16 mantissa bits truncated
pub inline fn bf16ToF32(bf16: u16) f32 {
    const bits: u32 = @as(u32, bf16) << 16;
    return @bitCast(bits);
}

// =============================================================================
// Dequantization
// =============================================================================

/// Dequantize a block of MXFP4 values to f32
/// blocks: packed 4-bit values (2 values per byte)
/// scales: E8M0 scales (1 per 32 values)
/// out: output f32 buffer
/// n_elements: number of elements to dequantize
pub fn dequantize(blocks: []const u8, scales: []const u8, out: []f32, n_elements: usize) void {
    const block_size: usize = 32;
    var out_idx: usize = 0;

    var scale_idx: usize = 0;
    while (out_idx < n_elements and scale_idx < scales.len) : (scale_idx += 1) {
        const scale = e8m0ToScale(scales[scale_idx]);

        // Each scale covers 32 values = 16 bytes of packed data
        var byte_in_block: usize = 0;
        while (byte_in_block < block_size / 2 and out_idx < n_elements) : (byte_in_block += 1) {
            const byte_idx = scale_idx * (block_size / 2) + byte_in_block;
            if (byte_idx >= blocks.len) break;

            const byte = blocks[byte_idx];
            // Lower nibble first, then upper nibble
            const low_nibble = byte & 0x0F;
            const high_nibble = byte >> 4;

            if (out_idx < out.len) {
                out[out_idx] = MXFP4_LUT[low_nibble] * scale;
                out_idx += 1;
            }
            if (out_idx < out.len and out_idx < n_elements) {
                out[out_idx] = MXFP4_LUT[high_nibble] * scale;
                out_idx += 1;
            }
        }
    }
}

// =============================================================================
// Matrix Multiplication
// =============================================================================

/// Dequantize MXFP4 and perform matmul in one pass (for efficiency)
/// This avoids materializing the full dequantized matrix
///
/// Layout: MXFP4 safetensors stores weights as [out_features, n_groups, 16] for blocks
/// and [out_features, n_groups] for scales
///
/// Original HF safetensors layout is "aaaa...bbbb..." where:
/// - First 8 bytes contain lo nibbles for positions 0-15 (2 per byte: byte[i] has pos 2i and 2i+1)
/// - Last 8 bytes contain hi nibbles for positions 16-31 (2 per byte: byte[i+8] has pos 16+2i and 16+2i+1)
///
/// Within each byte: lo nibble (& 0x0F) is first position, hi nibble (>> 4) is second position
pub fn matmulF32(
    input: []const f32,
    blocks: []const u8,
    scales: []const u8,
    output: []f32,
    in_features: usize,
    out_features: usize,
    bias: ?[]const f32,
) void {
    const block_size: usize = 32;
    const bytes_per_block: usize = 16;
    const n_groups = (in_features + block_size - 1) / block_size;
    const bytes_per_row = n_groups * bytes_per_block;

    log.trace("compute", "mxfp4MatmulF32", .{ .in_features = in_features, .out_features = out_features, .n_groups = n_groups }, @src());

    const MatmulF32Ctx = struct {
        input: []const f32,
        blocks: []const u8,
        scales: []const u8,
        output: []f32,
        bias: ?[]const f32,
        in_features: usize,
        n_groups: usize,
        bytes_per_row: usize,
    };

    var context = MatmulF32Ctx{
        .input = input,
        .blocks = blocks,
        .scales = scales,
        .output = output,
        .bias = bias,
        .in_features = in_features,
        .n_groups = n_groups,
        .bytes_per_row = bytes_per_row,
    };

    const row_task = struct {
        // SIMD constants - use 4 accumulators to hide FMA latency
        const F32x8 = @Vector(8, f32);
        const LUT: [16]f32 = MXFP4_LUT;

        // Decode 16 bytes (32 nibbles) into 4 x F32x8 vectors
        // This processes one full MXFP4 group (32 values) at once
        inline fn decodeGroup(bytes: *const [16]u8) [4]F32x8 {
            // Unroll completely to let compiler optimize
            return .{
                .{ LUT[bytes[0] & 0xF], LUT[bytes[0] >> 4], LUT[bytes[1] & 0xF], LUT[bytes[1] >> 4], LUT[bytes[2] & 0xF], LUT[bytes[2] >> 4], LUT[bytes[3] & 0xF], LUT[bytes[3] >> 4] },
                .{ LUT[bytes[4] & 0xF], LUT[bytes[4] >> 4], LUT[bytes[5] & 0xF], LUT[bytes[5] >> 4], LUT[bytes[6] & 0xF], LUT[bytes[6] >> 4], LUT[bytes[7] & 0xF], LUT[bytes[7] >> 4] },
                .{ LUT[bytes[8] & 0xF], LUT[bytes[8] >> 4], LUT[bytes[9] & 0xF], LUT[bytes[9] >> 4], LUT[bytes[10] & 0xF], LUT[bytes[10] >> 4], LUT[bytes[11] & 0xF], LUT[bytes[11] >> 4] },
                .{ LUT[bytes[12] & 0xF], LUT[bytes[12] >> 4], LUT[bytes[13] & 0xF], LUT[bytes[13] >> 4], LUT[bytes[14] & 0xF], LUT[bytes[14] >> 4], LUT[bytes[15] & 0xF], LUT[bytes[15] >> 4] },
            };
        }

        fn runRows(start: usize, end: usize, task_ctx: *MatmulF32Ctx) void {
            for (start..end) |row_idx| {
                // Use 4 accumulators to hide FMA latency (4 cycles on most CPUs)
                var acc0: F32x8 = @splat(0.0);
                var acc1: F32x8 = @splat(0.0);
                var acc2: F32x8 = @splat(0.0);
                var acc3: F32x8 = @splat(0.0);
                var sum_scalar: f32 = 0.0;

                const row_blocks = task_ctx.blocks[row_idx * task_ctx.bytes_per_row ..][0..task_ctx.bytes_per_row];
                const row_scales = task_ctx.scales[row_idx * task_ctx.n_groups ..][0..task_ctx.n_groups];

                // Process groups - unroll by 2 for better pipelining
                var group_idx: usize = 0;
                const safe_groups = if (task_ctx.in_features >= block_size) task_ctx.n_groups -| 1 else 0;

                // Main loop: process 2 groups at a time
                while (group_idx + 1 < safe_groups) : (group_idx += 2) {
                    // Group 1
                    const scale1 = e8m0ToScale(row_scales[group_idx]);
                    const sv1: F32x8 = @splat(scale1);
                    const group_bytes1 = row_blocks[group_idx * bytes_per_block ..][0..bytes_per_block];
                    const in_offset1 = group_idx * block_size;
                    const group_vals1 = decodeGroup(group_bytes1);

                    // Group 2
                    const scale2 = e8m0ToScale(row_scales[group_idx + 1]);
                    const sv2: F32x8 = @splat(scale2);
                    const group_bytes2 = row_blocks[(group_idx + 1) * bytes_per_block ..][0..bytes_per_block];
                    const in_offset2 = (group_idx + 1) * block_size;
                    const group_vals2 = decodeGroup(group_bytes2);

                    // Interleave FMAs for better pipelining
                    const in1_0: F32x8 = task_ctx.input[in_offset1..][0..8].*;
                    const in2_0: F32x8 = task_ctx.input[in_offset2..][0..8].*;
                    acc0 = @mulAdd(F32x8, in1_0, group_vals1[0] * sv1, acc0);
                    acc0 = @mulAdd(F32x8, in2_0, group_vals2[0] * sv2, acc0);

                    const in1_1: F32x8 = task_ctx.input[in_offset1 + 8 ..][0..8].*;
                    const in2_1: F32x8 = task_ctx.input[in_offset2 + 8 ..][0..8].*;
                    acc1 = @mulAdd(F32x8, in1_1, group_vals1[1] * sv1, acc1);
                    acc1 = @mulAdd(F32x8, in2_1, group_vals2[1] * sv2, acc1);

                    const in1_2: F32x8 = task_ctx.input[in_offset1 + 16 ..][0..8].*;
                    const in2_2: F32x8 = task_ctx.input[in_offset2 + 16 ..][0..8].*;
                    acc2 = @mulAdd(F32x8, in1_2, group_vals1[2] * sv1, acc2);
                    acc2 = @mulAdd(F32x8, in2_2, group_vals2[2] * sv2, acc2);

                    const in1_3: F32x8 = task_ctx.input[in_offset1 + 24 ..][0..8].*;
                    const in2_3: F32x8 = task_ctx.input[in_offset2 + 24 ..][0..8].*;
                    acc3 = @mulAdd(F32x8, in1_3, group_vals1[3] * sv1, acc3);
                    acc3 = @mulAdd(F32x8, in2_3, group_vals2[3] * sv2, acc3);
                }

                // Handle remaining full groups one at a time
                while (group_idx < safe_groups) : (group_idx += 1) {
                    const scale = e8m0ToScale(row_scales[group_idx]);
                    const sv: F32x8 = @splat(scale);
                    const group_bytes = row_blocks[group_idx * bytes_per_block ..][0..bytes_per_block];
                    const in_offset = group_idx * block_size;
                    const group_vals = decodeGroup(group_bytes);

                    const inp0: F32x8 = task_ctx.input[in_offset..][0..8].*;
                    const inp1: F32x8 = task_ctx.input[in_offset + 8 ..][0..8].*;
                    const inp2: F32x8 = task_ctx.input[in_offset + 16 ..][0..8].*;
                    const inp3: F32x8 = task_ctx.input[in_offset + 24 ..][0..8].*;

                    acc0 = @mulAdd(F32x8, inp0, group_vals[0] * sv, acc0);
                    acc1 = @mulAdd(F32x8, inp1, group_vals[1] * sv, acc1);
                    acc2 = @mulAdd(F32x8, inp2, group_vals[2] * sv, acc2);
                    acc3 = @mulAdd(F32x8, inp3, group_vals[3] * sv, acc3);
                }

                // Handle last partial group with scalar loop
                while (group_idx < task_ctx.n_groups) : (group_idx += 1) {
                    const scale = e8m0ToScale(row_scales[group_idx]);
                    const group_bytes = row_blocks[group_idx * bytes_per_block ..][0..bytes_per_block];
                    const in_offset = group_idx * block_size;

                    for (0..16) |j| {
                        const byte = group_bytes[j];
                        const pos_first = in_offset + j * 2;
                        const pos_second = in_offset + j * 2 + 1;
                        if (pos_first < task_ctx.in_features) {
                            sum_scalar += task_ctx.input[pos_first] * LUT[byte & 0xF] * scale;
                        }
                        if (pos_second < task_ctx.in_features) {
                            sum_scalar += task_ctx.input[pos_second] * LUT[byte >> 4] * scale;
                        }
                    }
                }

                // Reduce all 4 SIMD accumulators
                const acc_sum = acc0 + acc1 + acc2 + acc3;
                const sum = @reduce(.Add, acc_sum) + sum_scalar;

                if (task_ctx.bias) |b| {
                    task_ctx.output[row_idx] = sum + b[row_idx];
                } else {
                    task_ctx.output[row_idx] = sum;
                }
            }
        }

    };

    parallel.global().parallelFor(out_features, row_task.runRows, &context);
}

/// MXFP4 matmul with bfloat16 input (converts bf16->f32 on-the-fly)
/// This eliminates the need for Python-side dtype conversion.
pub fn matmulBF16(
    input: []const u16,
    blocks: []const u8,
    scales: []const u8,
    output: []f32,
    in_features: usize,
    out_features: usize,
    bias: ?[]const f32,
) void {
    const block_size: usize = 32;
    const bytes_per_block: usize = 16;
    const n_groups = (in_features + block_size - 1) / block_size;
    const bytes_per_row = n_groups * bytes_per_block;

    const MatmulBF16Ctx = struct {
        input: []const u16,
        blocks: []const u8,
        scales: []const u8,
        output: []f32,
        bias: ?[]const f32,
        in_features: usize,
        n_groups: usize,
        bytes_per_row: usize,
    };

    var context = MatmulBF16Ctx{
        .input = input,
        .blocks = blocks,
        .scales = scales,
        .output = output,
        .bias = bias,
        .in_features = in_features,
        .n_groups = n_groups,
        .bytes_per_row = bytes_per_row,
    };

    const row_task = struct {
        const VEC = 8;
        const F32x8 = @Vector(VEC, f32);
        const U16x8 = @Vector(VEC, u16);
        const U32x8 = @Vector(VEC, u32);
        const LUT: [16]f32 = MXFP4_LUT;

        inline fn bf16x8ToF32x8(bf16_arr: *const [8]u16) F32x8 {
            // Convert each bf16 to f32: shift left by 16 and bitcast
            return .{
                bf16ToF32(bf16_arr[0]),
                bf16ToF32(bf16_arr[1]),
                bf16ToF32(bf16_arr[2]),
                bf16ToF32(bf16_arr[3]),
                bf16ToF32(bf16_arr[4]),
                bf16ToF32(bf16_arr[5]),
                bf16ToF32(bf16_arr[6]),
                bf16ToF32(bf16_arr[7]),
            };
        }

        inline fn decodeMxfp4x8(bytes: *const [4]u8) F32x8 {
            const b0 = bytes[0];
            const b1 = bytes[1];
            const b2 = bytes[2];
            const b3 = bytes[3];
            return .{
                LUT[b0 & 0x0F],
                LUT[b0 >> 4],
                LUT[b1 & 0x0F],
                LUT[b1 >> 4],
                LUT[b2 & 0x0F],
                LUT[b2 >> 4],
                LUT[b3 & 0x0F],
                LUT[b3 >> 4],
            };
        }

        // Decode 16 bytes (32 nibbles) into 4 x F32x8 vectors (same as f32 kernel)
        inline fn decodeGroup(bytes: *const [16]u8) [4]F32x8 {
            return .{
                .{ LUT[bytes[0] & 0xF], LUT[bytes[0] >> 4], LUT[bytes[1] & 0xF], LUT[bytes[1] >> 4], LUT[bytes[2] & 0xF], LUT[bytes[2] >> 4], LUT[bytes[3] & 0xF], LUT[bytes[3] >> 4] },
                .{ LUT[bytes[4] & 0xF], LUT[bytes[4] >> 4], LUT[bytes[5] & 0xF], LUT[bytes[5] >> 4], LUT[bytes[6] & 0xF], LUT[bytes[6] >> 4], LUT[bytes[7] & 0xF], LUT[bytes[7] >> 4] },
                .{ LUT[bytes[8] & 0xF], LUT[bytes[8] >> 4], LUT[bytes[9] & 0xF], LUT[bytes[9] >> 4], LUT[bytes[10] & 0xF], LUT[bytes[10] >> 4], LUT[bytes[11] & 0xF], LUT[bytes[11] >> 4] },
                .{ LUT[bytes[12] & 0xF], LUT[bytes[12] >> 4], LUT[bytes[13] & 0xF], LUT[bytes[13] >> 4], LUT[bytes[14] & 0xF], LUT[bytes[14] >> 4], LUT[bytes[15] & 0xF], LUT[bytes[15] >> 4] },
            };
        }

        fn runRows(start: usize, end: usize, task_ctx: *MatmulBF16Ctx) void {
            for (start..end) |row_idx| {
                // Use 4 accumulators to hide FMA latency (same as f32 kernel)
                var acc0: F32x8 = @splat(0.0);
                var acc1: F32x8 = @splat(0.0);
                var acc2: F32x8 = @splat(0.0);
                var acc3: F32x8 = @splat(0.0);
                var sum_scalar: f32 = 0.0;

                const row_blocks = task_ctx.blocks[row_idx * task_ctx.bytes_per_row ..][0..task_ctx.bytes_per_row];
                const row_scales = task_ctx.scales[row_idx * task_ctx.n_groups ..][0..task_ctx.n_groups];

                var group_idx: usize = 0;
                const safe_groups = if (task_ctx.in_features >= block_size) task_ctx.n_groups -| 1 else 0;

                // Main loop: process 2 groups at a time (same as f32 kernel)
                while (group_idx + 1 < safe_groups) : (group_idx += 2) {
                    // Group 1
                    const scale1 = e8m0ToScale(row_scales[group_idx]);
                    const sv1: F32x8 = @splat(scale1);
                    const group_bytes1 = row_blocks[group_idx * bytes_per_block ..][0..bytes_per_block];
                    const in_offset1 = group_idx * block_size;
                    const group_vals1 = decodeGroup(group_bytes1);

                    // Group 2
                    const scale2 = e8m0ToScale(row_scales[group_idx + 1]);
                    const sv2: F32x8 = @splat(scale2);
                    const group_bytes2 = row_blocks[(group_idx + 1) * bytes_per_block ..][0..bytes_per_block];
                    const in_offset2 = (group_idx + 1) * block_size;
                    const group_vals2 = decodeGroup(group_bytes2);

                    // Interleave FMAs for better pipelining
                    const in1_0 = bf16x8ToF32x8(task_ctx.input[in_offset1..][0..8]);
                    const in2_0 = bf16x8ToF32x8(task_ctx.input[in_offset2..][0..8]);
                    acc0 = @mulAdd(F32x8, in1_0, group_vals1[0] * sv1, acc0);
                    acc0 = @mulAdd(F32x8, in2_0, group_vals2[0] * sv2, acc0);

                    const in1_1 = bf16x8ToF32x8(task_ctx.input[in_offset1 + 8 ..][0..8]);
                    const in2_1 = bf16x8ToF32x8(task_ctx.input[in_offset2 + 8 ..][0..8]);
                    acc1 = @mulAdd(F32x8, in1_1, group_vals1[1] * sv1, acc1);
                    acc1 = @mulAdd(F32x8, in2_1, group_vals2[1] * sv2, acc1);

                    const in1_2 = bf16x8ToF32x8(task_ctx.input[in_offset1 + 16 ..][0..8]);
                    const in2_2 = bf16x8ToF32x8(task_ctx.input[in_offset2 + 16 ..][0..8]);
                    acc2 = @mulAdd(F32x8, in1_2, group_vals1[2] * sv1, acc2);
                    acc2 = @mulAdd(F32x8, in2_2, group_vals2[2] * sv2, acc2);

                    const in1_3 = bf16x8ToF32x8(task_ctx.input[in_offset1 + 24 ..][0..8]);
                    const in2_3 = bf16x8ToF32x8(task_ctx.input[in_offset2 + 24 ..][0..8]);
                    acc3 = @mulAdd(F32x8, in1_3, group_vals1[3] * sv1, acc3);
                    acc3 = @mulAdd(F32x8, in2_3, group_vals2[3] * sv2, acc3);
                }

                // Handle remaining full groups one at a time
                while (group_idx < safe_groups) : (group_idx += 1) {
                    const scale = e8m0ToScale(row_scales[group_idx]);
                    const sv: F32x8 = @splat(scale);
                    const group_bytes = row_blocks[group_idx * bytes_per_block ..][0..bytes_per_block];
                    const in_offset = group_idx * block_size;
                    const group_vals = decodeGroup(group_bytes);

                    const inp0 = bf16x8ToF32x8(task_ctx.input[in_offset..][0..8]);
                    const inp1 = bf16x8ToF32x8(task_ctx.input[in_offset + 8 ..][0..8]);
                    const inp2 = bf16x8ToF32x8(task_ctx.input[in_offset + 16 ..][0..8]);
                    const inp3 = bf16x8ToF32x8(task_ctx.input[in_offset + 24 ..][0..8]);

                    acc0 = @mulAdd(F32x8, inp0, group_vals[0] * sv, acc0);
                    acc1 = @mulAdd(F32x8, inp1, group_vals[1] * sv, acc1);
                    acc2 = @mulAdd(F32x8, inp2, group_vals[2] * sv, acc2);
                    acc3 = @mulAdd(F32x8, inp3, group_vals[3] * sv, acc3);
                }

                // Handle last partial group with scalar loop
                while (group_idx < task_ctx.n_groups) : (group_idx += 1) {
                    const scale = e8m0ToScale(row_scales[group_idx]);
                    const group_bytes = row_blocks[group_idx * bytes_per_block ..][0..bytes_per_block];
                    const in_offset = group_idx * block_size;

                    for (0..16) |j| {
                        const byte = group_bytes[j];
                        const pos_first = in_offset + j * 2;
                        const pos_second = in_offset + j * 2 + 1;
                        if (pos_first < task_ctx.in_features) {
                            sum_scalar += bf16ToF32(task_ctx.input[pos_first]) * LUT[byte & 0xF] * scale;
                        }
                        if (pos_second < task_ctx.in_features) {
                            sum_scalar += bf16ToF32(task_ctx.input[pos_second]) * LUT[byte >> 4] * scale;
                        }
                    }
                }

                // Reduce all 4 SIMD accumulators
                const acc_sum = acc0 + acc1 + acc2 + acc3;
                const sum = @reduce(.Add, acc_sum) + sum_scalar;

                if (task_ctx.bias) |b| {
                    task_ctx.output[row_idx] = sum + b[row_idx];
                } else {
                    task_ctx.output[row_idx] = sum;
                }
            }
        }
    };

    parallel.global().parallelFor(out_features, row_task.runRows, &context);
}

/// MXFP4 matmul with transposed weight layout.
///
/// Some models store expert weights as [in_features, packed_out_features] and use
/// `x @ W` (input on left, weight on right). Our standard layout is [out_features, packed_in_features]
/// with `W @ x`.
///
/// This function handles the transposed layout where:
/// - blocks has shape [in_features, n_groups_out * 16] (each input position has a row)
/// - scales has shape [in_features, n_groups_out] (each input position has scales for its output groups)
/// - For each input position i, the packed bytes contain weights for all output positions
///
/// Computes: output[o] = sum_i(input[i] * W[i, o])
pub fn matmulF32Transposed(
    input: []const f32,
    blocks: []const u8,
    scales: []const u8,
    output: []f32,
    in_features: usize,
    out_features: usize,
    bias: ?[]const f32,
) void {
    const block_size: usize = 32;
    const bytes_per_block: usize = 16;
    const n_groups_out = (out_features + block_size - 1) / block_size;
    const bytes_per_row = n_groups_out * bytes_per_block;

    // LUT for MXFP4 values
    const LUT: [16]f32 = MXFP4_LUT;

    log.trace("compute", "mxfp4MatmulF32Transposed", .{ .in_features = in_features, .out_features = out_features, .n_groups_out = n_groups_out }, @src());

    // Parallelized implementation: iterate over output positions
    // For each output position o, compute: sum_i(input[i] * W[i, o])
    // This allows parallel writes to different output positions
    const MatmulF32TransposedCtx = struct {
        input: []const f32,
        blocks: []const u8,
        scales: []const u8,
        output: []f32,
        bias: ?[]const f32,
        in_features: usize,
        out_features: usize,
        n_groups_out: usize,
        bytes_per_row: usize,
    };

    const context = MatmulF32TransposedCtx{
        .input = input,
        .blocks = blocks,
        .scales = scales,
        .output = output,
        .bias = bias,
        .in_features = in_features,
        .out_features = out_features,
        .n_groups_out = n_groups_out,
        .bytes_per_row = bytes_per_row,
    };

    const output_task = struct {
        fn runOutputSlice(start: usize, end: usize, task_ctx: *const MatmulF32TransposedCtx) void {
            // For each output position in this chunk
            for (start..end) |out_idx| {
                // Determine which group this output belongs to and position within group
                const out_group_idx = out_idx / block_size;
                const out_pos_in_group = out_idx % block_size;
                const byte_idx = out_pos_in_group / 2;
                const is_high_nibble = (out_pos_in_group % 2) == 1;

                var sum: f32 = 0.0;

                // Iterate over all input positions
                for (0..task_ctx.in_features) |in_idx| {
                    const input_val = task_ctx.input[in_idx];
                    if (input_val == 0) continue;

                    // Get the scale for this input row and output group
                    const scale = e8m0ToScale(task_ctx.scales[in_idx * task_ctx.n_groups_out + out_group_idx]);

                    // Get the byte containing this output position's weight
                    const byte_offset = in_idx * task_ctx.bytes_per_row + out_group_idx * bytes_per_block + byte_idx;
                    const byte = task_ctx.blocks[byte_offset];

                    // Extract the nibble
                    const nibble = if (is_high_nibble) (byte >> 4) else (byte & 0x0F);
                    const weight = LUT[nibble] * scale;

                    sum += input_val * weight;
                }

                // Add bias if present
                if (task_ctx.bias) |b| {
                    task_ctx.output[out_idx] = sum + b[out_idx];
                } else {
                    task_ctx.output[out_idx] = sum;
                }
            }
        }
    };

    parallel.global().parallelFor(out_features, output_task.runOutputSlice, &context);
}

// =============================================================================
// Unit Tests
// =============================================================================

test "dequantize MXFP4_LUT values" {
    const testing = std.testing;

    // Positive values
    try testing.expectApproxEqAbs(0.0, MXFP4_LUT[0], 1e-6);
    try testing.expectApproxEqAbs(0.5, MXFP4_LUT[1], 1e-6);
    try testing.expectApproxEqAbs(1.0, MXFP4_LUT[2], 1e-6);
    try testing.expectApproxEqAbs(1.5, MXFP4_LUT[3], 1e-6);
    try testing.expectApproxEqAbs(2.0, MXFP4_LUT[4], 1e-6);
    try testing.expectApproxEqAbs(3.0, MXFP4_LUT[5], 1e-6);
    try testing.expectApproxEqAbs(4.0, MXFP4_LUT[6], 1e-6);
    try testing.expectApproxEqAbs(6.0, MXFP4_LUT[7], 1e-6);

    // Negative values
    try testing.expectApproxEqAbs(-0.0, MXFP4_LUT[8], 1e-6);
    try testing.expectApproxEqAbs(-0.5, MXFP4_LUT[9], 1e-6);
    try testing.expectApproxEqAbs(-1.0, MXFP4_LUT[10], 1e-6);
    try testing.expectApproxEqAbs(-1.5, MXFP4_LUT[11], 1e-6);
    try testing.expectApproxEqAbs(-2.0, MXFP4_LUT[12], 1e-6);
    try testing.expectApproxEqAbs(-3.0, MXFP4_LUT[13], 1e-6);
    try testing.expectApproxEqAbs(-4.0, MXFP4_LUT[14], 1e-6);
    try testing.expectApproxEqAbs(-6.0, MXFP4_LUT[15], 1e-6);
}

test "e8m0ToScale: zero exponent" {
    const testing = std.testing;
    const scale = e8m0ToScale(0);
    try testing.expectApproxEqAbs(0.0, scale, 1e-6);
}

test "e8m0ToScale: known values" {
    const testing = std.testing;

    // e8m0 = 127 -> 2^(127-127) = 2^0 = 1.0
    const scale_127 = e8m0ToScale(127);
    try testing.expectApproxEqAbs(1.0, scale_127, 1e-6);

    // e8m0 = 128 -> 2^(128-127) = 2^1 = 2.0
    const scale_128 = e8m0ToScale(128);
    try testing.expectApproxEqAbs(2.0, scale_128, 1e-6);

    // e8m0 = 126 -> 2^(126-127) = 2^-1 = 0.5
    const scale_126 = e8m0ToScale(126);
    try testing.expectApproxEqAbs(0.5, scale_126, 1e-6);

    // e8m0 = 130 -> 2^(130-127) = 2^3 = 8.0
    const scale_130 = e8m0ToScale(130);
    try testing.expectApproxEqAbs(8.0, scale_130, 1e-6);

    // e8m0 = 120 -> 2^(120-127) = 2^-7 = 1/128
    const scale_120 = e8m0ToScale(120);
    try testing.expectApproxEqAbs(1.0 / 128.0, scale_120, 1e-6);
}

test "e8m0ToScale: max value" {
    const testing = std.testing;
    // e8m0 = 255 -> 2^(255-127) = 2^128 (very large)
    const scale_max = e8m0ToScale(255);
    try testing.expect(scale_max > 1e30);
}

test "bf16ToF32: zero" {
    const testing = std.testing;
    const result = bf16ToF32(0);
    try testing.expectApproxEqAbs(0.0, result, 1e-6);
}

test "bf16ToF32: known values" {
    const testing = std.testing;

    // 1.0 in BF16: sign=0, exp=127, mantissa=0
    // In IEEE754: 0 01111111 00000000 = 0x3F80
    const one_bf16: u16 = 0x3F80;
    const result_one = bf16ToF32(one_bf16);
    try testing.expectApproxEqAbs(1.0, result_one, 1e-6);

    // 2.0 in BF16: 0x4000
    const two_bf16: u16 = 0x4000;
    const result_two = bf16ToF32(two_bf16);
    try testing.expectApproxEqAbs(2.0, result_two, 1e-6);

    // -1.0 in BF16: 0xBF80
    const neg_one_bf16: u16 = 0xBF80;
    const result_neg_one = bf16ToF32(neg_one_bf16);
    try testing.expectApproxEqAbs(-1.0, result_neg_one, 1e-6);
}

test "dequantize nibble extraction low" {
    const testing = std.testing;

    // Byte 0xAB: low=B (11), high=A (10)
    const byte: u8 = 0xAB;
    const low = byte & 0x0F;
    const high = byte >> 4;

    try testing.expectEqual(@as(u8, 0x0B), low);
    try testing.expectEqual(@as(u8, 0x0A), high);
}

test "dequantize nibble extraction special" {
    const testing = std.testing;

    // 0x00: both nibbles are 0
    const byte_00: u8 = 0x00;
    try testing.expectEqual(@as(u8, 0), byte_00 & 0x0F);
    try testing.expectEqual(@as(u8, 0), byte_00 >> 4);

    // 0xFF: both nibbles are F (15)
    const byte_FF: u8 = 0xFF;
    try testing.expectEqual(@as(u8, 0x0F), byte_FF & 0x0F);
    try testing.expectEqual(@as(u8, 0x0F), byte_FF >> 4);

    // 0xF0: low=0, high=F
    const byte_F0: u8 = 0xF0;
    try testing.expectEqual(@as(u8, 0x00), byte_F0 & 0x0F);
    try testing.expectEqual(@as(u8, 0x0F), byte_F0 >> 4);

    // 0x0F: low=F, high=0
    const byte_0F: u8 = 0x0F;
    try testing.expectEqual(@as(u8, 0x0F), byte_0F & 0x0F);
    try testing.expectEqual(@as(u8, 0x00), byte_0F >> 4);
}

test "dequantize: single value" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Single byte containing two nibbles
    const blocks = [_]u8{0x12}; // low=2 (1.0), high=1 (0.5)
    const scales = [_]u8{127}; // scale = 1.0

    const output = try allocator.alloc(f32, 2);
    defer allocator.free(output);

    dequantize(&blocks, &scales, output, 2);

    // First value: MXFP4_LUT[2] * 1.0 = 1.0
    // Second value: MXFP4_LUT[1] * 1.0 = 0.5
    try testing.expectApproxEqAbs(1.0, output[0], 1e-6);
    try testing.expectApproxEqAbs(0.5, output[1], 1e-6);
}

test "dequantize: zero scale" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const blocks = [_]u8{0x23}; // low=3 (1.5), high=2 (1.0)
    const scales = [_]u8{0}; // scale = 0.0

    const output = try allocator.alloc(f32, 2);
    defer allocator.free(output);

    dequantize(&blocks, &scales, output, 2);

    // Both values should be 0.0 due to zero scale
    try testing.expectApproxEqAbs(0.0, output[0], 1e-6);
    try testing.expectApproxEqAbs(0.0, output[1], 1e-6);
}

test "dequantize: negative values" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // 0xAB: low=B (11=-1.5), high=A (10=-1.0)
    const blocks = [_]u8{0xAB};
    const scales = [_]u8{127}; // scale = 1.0

    const output = try allocator.alloc(f32, 2);
    defer allocator.free(output);

    dequantize(&blocks, &scales, output, 2);

    try testing.expectApproxEqAbs(-1.5, output[0], 1e-6);
    try testing.expectApproxEqAbs(-1.0, output[1], 1e-6);
}

test "dequantize: with non-unity scale" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const blocks = [_]u8{0x24}; // low=4 (2.0), high=2 (1.0)
    const scales = [_]u8{128}; // scale = 2.0

    const output = try allocator.alloc(f32, 2);
    defer allocator.free(output);

    dequantize(&blocks, &scales, output, 2);

    // First: 2.0 * 2.0 = 4.0
    // Second: 1.0 * 2.0 = 2.0
    try testing.expectApproxEqAbs(4.0, output[0], 1e-6);
    try testing.expectApproxEqAbs(2.0, output[1], 1e-6);
}

test "dequantize: full block (32 values)" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // 32 values = 16 bytes, all with same pattern
    var blocks: [16]u8 = undefined;
    for (0..16) |i| {
        blocks[i] = 0x23; // low=3 (1.5), high=2 (1.0)
    }
    const scales = [_]u8{127}; // scale = 1.0

    const output = try allocator.alloc(f32, 32);
    defer allocator.free(output);

    dequantize(&blocks, &scales, output, 32);

    // Verify alternating pattern: 1.5, 1.0, 1.5, 1.0, ...
    for (0..16) |i| {
        try testing.expectApproxEqAbs(1.5, output[i * 2], 1e-6);
        try testing.expectApproxEqAbs(1.0, output[i * 2 + 1], 1e-6);
    }
}

test "dequantize: group boundary handling" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Two groups: first 32 values with scale=1.0, next 32 with scale=2.0
    var blocks: [32]u8 = undefined;
    for (0..32) |i| {
        blocks[i] = 0x22; // both nibbles = 2 (1.0)
    }
    const scales = [_]u8{ 127, 128 }; // scales: 1.0, 2.0

    const output = try allocator.alloc(f32, 64);
    defer allocator.free(output);

    dequantize(&blocks, &scales, output, 64);

    // First 32 values should be 1.0 (1.0 * 1.0)
    for (0..32) |i| {
        try testing.expectApproxEqAbs(1.0, output[i], 1e-6);
    }

    // Next 32 values should be 2.0 (1.0 * 2.0)
    for (32..64) |i| {
        try testing.expectApproxEqAbs(2.0, output[i], 1e-6);
    }
}

test "dequantize: partial block" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Request only 3 values from a 16-byte block
    const blocks = [_]u8{0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
    const scales = [_]u8{127}; // scale = 1.0

    const output = try allocator.alloc(f32, 3);
    defer allocator.free(output);

    dequantize(&blocks, &scales, output, 3);

    // First byte 0x12: low=2 (1.0), high=1 (0.5)
    try testing.expectApproxEqAbs(1.0, output[0], 1e-6);
    try testing.expectApproxEqAbs(0.5, output[1], 1e-6);

    // Second byte 0x34: low=4 (2.0)
    try testing.expectApproxEqAbs(2.0, output[2], 1e-6);
}

test "dequantize: all special nibbles" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test all 16 possible nibble values
    const blocks = [_]u8{
        0x10, 0x32, 0x54, 0x76, // nibbles 0-7
        0x98, 0xBA, 0xDC, 0xFE, // nibbles 8-15
    };
    const scales = [_]u8{127}; // scale = 1.0

    const output = try allocator.alloc(f32, 16);
    defer allocator.free(output);

    dequantize(&blocks, &scales, output, 16);

    // Verify each value matches the LUT (nibbles 0-15 with scale=1.0)
    for (0..16) |i| {
        try testing.expectApproxEqAbs(MXFP4_LUT[i], output[i], 1e-6);
    }
}

test "dequantize: edge case 0x00 bytes" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // All zero bytes
    const blocks = [_]u8{ 0x00, 0x00, 0x00, 0x00 };
    const scales = [_]u8{128}; // scale = 2.0

    const output = try allocator.alloc(f32, 8);
    defer allocator.free(output);

    dequantize(&blocks, &scales, output, 8);

    // All values should be 0.0 (nibble 0 = 0.0)
    for (output) |val| {
        try testing.expectApproxEqAbs(0.0, val, 1e-6);
    }
}

test "dequantize: edge case 0xFF bytes" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // All 0xFF bytes (nibble F = -6.0)
    const blocks = [_]u8{ 0xFF, 0xFF, 0xFF, 0xFF };
    const scales = [_]u8{127}; // scale = 1.0

    const output = try allocator.alloc(f32, 8);
    defer allocator.free(output);

    dequantize(&blocks, &scales, output, 8);

    // All values should be -6.0
    for (output) |val| {
        try testing.expectApproxEqAbs(-6.0, val, 1e-6);
    }
}

test "matmulF32: simple 2x2" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Input: [1.0, 2.0]
    // Weight matrix (2 outputs, 2 inputs):
    //   output 0: [1.0, 0.5]  (nibbles: 2, 1)
    //   output 1: [2.0, 1.5]  (nibbles: 4, 3)
    // Expected:
    //   output[0] = 1.0*1.0 + 2.0*0.5 = 2.0
    //   output[1] = 1.0*2.0 + 2.0*1.5 = 5.0

    const input = [_]f32{ 1.0, 2.0 };
    const blocks = [_]u8{
        0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // row 0
        0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // row 1
    };
    const scales = [_]u8{ 127, 127 }; // both scales = 1.0

    const output = try allocator.alloc(f32, 2);
    defer allocator.free(output);

    matmulF32(&input, &blocks, &scales, output, 2, 2, null);

    try testing.expectApproxEqAbs(2.0, output[0], 1e-5);
    try testing.expectApproxEqAbs(5.0, output[1], 1e-5);
}

test "matmulF32: with bias" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const input = [_]f32{ 1.0, 1.0 };
    const blocks = [_]u8{
        0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };
    const scales = [_]u8{127}; // scale = 1.0
    const bias = [_]f32{10.0};

    const output = try allocator.alloc(f32, 1);
    defer allocator.free(output);

    matmulF32(&input, &blocks, &scales, output, 2, 1, &bias);

    // Result: 1.0*1.0 + 1.0*1.0 + 10.0 = 12.0
    try testing.expectApproxEqAbs(12.0, output[0], 1e-5);
}

test "matmulF32: multiple groups" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // 64 inputs (2 groups), 1 output
    var input: [64]f32 = undefined;
    for (0..64) |i| {
        input[i] = 1.0;
    }

    // Two groups of weights, all nibbles = 2 (1.0)
    var blocks: [32]u8 = undefined;
    for (0..32) |i| {
        blocks[i] = 0x22;
    }
    const scales = [_]u8{ 127, 128 }; // scales: 1.0, 2.0

    const output = try allocator.alloc(f32, 1);
    defer allocator.free(output);

    matmulF32(&input, &blocks, &scales, output, 64, 1, null);

    // First 32 values: 32 * (1.0 * 1.0) = 32.0
    // Next 32 values: 32 * (1.0 * 2.0) = 64.0
    // Total: 96.0
    try testing.expectApproxEqAbs(96.0, output[0], 1e-4);
}

test "matmulF32: zero input" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const input = [_]f32{ 0.0, 0.0 };
    const blocks = [_]u8{
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    };
    const scales = [_]u8{127};

    const output = try allocator.alloc(f32, 1);
    defer allocator.free(output);

    matmulF32(&input, &blocks, &scales, output, 2, 1, null);

    try testing.expectApproxEqAbs(0.0, output[0], 1e-6);
}

test "matmulF32: negative weights" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Input: [1.0, 1.0]
    // Weights: [-1.0, -1.5] (nibbles: 10, 11)
    const input = [_]f32{ 1.0, 1.0 };
    const blocks = [_]u8{
        0xBA, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };
    const scales = [_]u8{127}; // scale = 1.0

    const output = try allocator.alloc(f32, 1);
    defer allocator.free(output);

    matmulF32(&input, &blocks, &scales, output, 2, 1, null);

    // Result: 1.0*(-1.0) + 1.0*(-1.5) = -2.5
    try testing.expectApproxEqAbs(-2.5, output[0], 1e-5);
}

test "matmulF32 matches scalar reference" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const in_features: usize = 96; // 3 groups of 32
    const out_features: usize = 5;
    const block_size: usize = 32;
    const bytes_per_block: usize = 16;
    const n_groups = (in_features + block_size - 1) / block_size;

    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    const input = try allocator.alloc(f32, in_features);
    defer allocator.free(input);
    for (input) |*v| {
        v.* = rng.float(f32) * 2.0 - 1.0;
    }

    const blocks_len = out_features * n_groups * bytes_per_block;
    const scales_len = out_features * n_groups;

    const blocks = try allocator.alloc(u8, blocks_len);
    defer allocator.free(blocks);
    const scales = try allocator.alloc(u8, scales_len);
    defer allocator.free(scales);

    for (blocks) |*b| {
        b.* = rng.int(u8);
    }
    // Avoid extreme exponent values to prevent overflow/inf that can
    // amplify small accumulation-order differences.
    for (scales) |*s| {
        s.* = 120 + @as(u8, @intCast(rng.int(u8) % 16)); // 120..135
    }

    const out = try allocator.alloc(f32, out_features);
    defer allocator.free(out);
    const out_ref = try allocator.alloc(f32, out_features);
    defer allocator.free(out_ref);

    @memset(out, 0.0);
    @memset(out_ref, 0.0);

    matmulF32(input, blocks, scales, out, in_features, out_features, null);

    // Scalar reference
    for (0..out_features) |row_idx| {
        var sum: f32 = 0.0;
        const row_blocks = blocks[row_idx * n_groups * bytes_per_block ..][0 .. n_groups * bytes_per_block];
        const row_scales = scales[row_idx * n_groups ..][0..n_groups];
        for (0..n_groups) |group_idx| {
            const scale = e8m0ToScale(row_scales[group_idx]);
            const group_bytes = row_blocks[group_idx * bytes_per_block ..][0..bytes_per_block];
            const in_offset = group_idx * block_size;
            for (0..bytes_per_block) |j| {
                const byte = group_bytes[j];
                const pos_first = in_offset + j * 2;
                const pos_second = pos_first + 1;
                if (pos_first < in_features) {
                    sum += input[pos_first] * MXFP4_LUT[byte & 0x0F] * scale;
                }
                if (pos_second < in_features) {
                    sum += input[pos_second] * MXFP4_LUT[byte >> 4] * scale;
                }
            }
        }
        out_ref[row_idx] = sum;
    }

    for (0..out_features) |i| {
        try testing.expect(std.math.isFinite(out[i]));
        try testing.expectApproxEqAbs(out_ref[i], out[i], 1e-3);
    }
}

test "matmulF32 matches scalar reference (fixed case)" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const in_features: usize = 32; // one full group
    const out_features: usize = 2;
    const bytes_per_block: usize = 16;
    const n_groups: usize = 1;

    const input = [_]f32{
        1.0, -2.0, 0.5, 0.0, 1.5, -1.0, 2.0, -0.5,
        0.25, -0.75, 1.25, -1.25, 0.75, -0.25, 0.1, -0.1,
        0.6, -0.6, 0.9, -0.9, 1.1, -1.1, 1.3, -1.3,
        1.4, -1.4, 1.6, -1.6, 1.8, -1.8, 2.2, -2.2,
    };

    const blocks = [_]u8{
        // Row 0
        0x21, 0x43, 0x65, 0x87, 0x10, 0x32, 0x54, 0x76,
        0x98, 0xBA, 0xDC, 0xFE, 0x01, 0x23, 0x45, 0x67,
        // Row 1
        0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE,
        0x21, 0x43, 0x65, 0x87, 0x89, 0xAB, 0xCD, 0xEF,
    };
    const scales = [_]u8{ 127, 127 }; // scale = 1.0 for both rows

    const out = try allocator.alloc(f32, out_features);
    defer allocator.free(out);
    const out_ref = try allocator.alloc(f32, out_features);
    defer allocator.free(out_ref);

    @memset(out, 0.0);
    @memset(out_ref, 0.0);

    matmulF32(&input, &blocks, &scales, out, in_features, out_features, null);

    // Scalar reference
    for (0..out_features) |row_idx| {
        var sum: f32 = 0.0;
        const row_blocks = blocks[row_idx * n_groups * bytes_per_block ..][0..bytes_per_block];
        const scale = e8m0ToScale(scales[row_idx]);
        for (0..bytes_per_block) |j| {
            const byte = row_blocks[j];
            const pos_first = j * 2;
            const pos_second = pos_first + 1;
            sum += input[pos_first] * MXFP4_LUT[byte & 0x0F] * scale;
            sum += input[pos_second] * MXFP4_LUT[byte >> 4] * scale;
        }
        out_ref[row_idx] = sum;
    }

    for (0..out_features) |i| {
        try testing.expect(std.math.isFinite(out[i]));
        try testing.expectApproxEqAbs(out_ref[i], out[i], 1e-3);
    }
}

test "matmulBF16: basic operation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // BF16 input: [1.0, 2.0]
    const input = [_]u16{ 0x3F80, 0x4000 };
    const blocks = [_]u8{
        0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };
    const scales = [_]u8{127}; // scale = 1.0

    const output = try allocator.alloc(f32, 1);
    defer allocator.free(output);

    matmulBF16(&input, &blocks, &scales, output, 2, 1, null);

    // Result: 1.0*1.0 + 2.0*0.5 = 2.0
    try testing.expectApproxEqAbs(2.0, output[0], 1e-5);
}

test "matmulF32Transposed: basic operation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Input: [1.0, 2.0]
    // Transposed weight layout: each input row contains weights for all outputs
    // Row 0 (input 0): weights for output[0]=1.0, output[1]=0.5
    // Row 1 (input 1): weights for output[0]=2.0, output[1]=1.5
    // Expected:
    //   output[0] = 1.0*1.0 + 2.0*2.0 = 5.0
    //   output[1] = 1.0*0.5 + 2.0*1.5 = 3.5

    const input = [_]f32{ 1.0, 2.0 };
    const blocks = [_]u8{
        0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // input 0
        0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // input 1
    };
    const scales = [_]u8{ 127, 127 }; // both scales = 1.0

    const output = try allocator.alloc(f32, 2);
    defer allocator.free(output);

    matmulF32Transposed(&input, &blocks, &scales, output, 2, 2, null);

    try testing.expectApproxEqAbs(5.0, output[0], 1e-5);
    try testing.expectApproxEqAbs(3.5, output[1], 1e-5);
}
