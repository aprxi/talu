//! CUDA backend smoke checks.
//!
//! Keeps startup smoke coverage out of engine orchestration logic.

const std = @import("std");
const compute = @import("../../../compute/root.zig");
const dtype = @import("../../../dtype.zig");
const log = @import("../../../log.zig");

const gaffine_scales_dtype_bf16 = compute.cuda.gaffine_u4_matvec.scales_dtype_bf16;

pub fn runMatmulSmoke(backend: anytype) !void {
    const device = &backend.device;
    const m: usize = 2;
    const k: usize = 2;
    const n: usize = 2;

    const a = [_]f32{
        1.0, 2.0,
        3.0, 4.0,
    };
    const b = [_]f32{
        5.0, 6.0,
        7.0, 8.0,
    };
    const expected = [_]f32{
        19.0, 22.0,
        43.0, 50.0,
    };
    var actual = [_]f32{0.0} ** (m * n);

    var a_dev = try device.allocBuffer(@sizeOf(f32) * a.len);
    defer a_dev.deinit(device);
    var b_dev = try device.allocBuffer(@sizeOf(f32) * b.len);
    defer b_dev.deinit(device);
    var c_dev = try device.allocBuffer(@sizeOf(f32) * actual.len);
    defer c_dev.deinit(device);

    try a_dev.upload(device, std.mem.sliceAsBytes(a[0..]));
    try b_dev.upload(device, std.mem.sliceAsBytes(b[0..]));
    try backend.blas.matmulF32(device, &a_dev, m, k, &b_dev, n, &c_dev);
    try c_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        if (@abs(want - got) > 0.001) return error.CudaSmokeMismatch;
    }

    log.info("inference", "CUDA matmul smoke passed", .{
        .m = m,
        .k = k,
        .n = n,
        .c00 = actual[0],
    });
}

pub fn runKernelSmoke(backend: anytype) !void {
    if (!backend.device.supportsModuleLaunch()) {
        log.info("inference", "CUDA module launch API unavailable; skipping kernel smoke", .{});
        return;
    }
    if (backend.vector_add_function == null or
        backend.mul_function == null or
        backend.copy_function == null or
        backend.copy_u16_function == null or
        backend.cast_f32_to_f16_function == null or
        backend.kv_write_f16_function == null or
        backend.rmsnorm_function == null or
        backend.rope_store_f16_function == null or
        backend.attn_scores_heads_f32_function == null or
        backend.attn_scores_heads_f16_kv_function == null or
        backend.attn_fused_heads_f16_kv_function == null or
        backend.softmax_rows_function == null or
        backend.attn_weighted_sum_heads_f32_function == null or
        backend.attn_weighted_sum_heads_f16_kv_function == null or
        backend.silu_function == null or
        backend.silu_mul_function == null or
        backend.gelu_mul_function == null or
        backend.shortconv_step_function == null or
        backend.argmax_function == null or
        backend.matmul_f16_function == null or
        backend.matmul_bf16_function == null or
        backend.matvec_f16_function == null or
        backend.matvec_bf16_function == null or
        backend.matvec_gate_up_f16_function == null or
        backend.matvec_gate_up_bf16_function == null or
        backend.matvec_qkv_f16_function == null or
        backend.matvec_qkv_bf16_function == null or
        backend.gaffine_u4_matvec_function == null or
        backend.gaffine_u4_matvec_gate_up_function == null or
        backend.gaffine_u4_matvec_qkv_function == null)
    {
        return error.CudaKernelUnavailable;
    }

    try runVectorAddSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.vector_add_function.?,
        backend.vector_add_source orelse .embedded_module,
    );
    try runMulSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.mul_function.?,
        backend.mul_source orelse .embedded_module,
    );
    try runCopySmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.copy_function.?,
        backend.copy_source orelse .embedded_module,
    );
    try runCopyU16Smoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.copy_u16_function.?,
        backend.copy_u16_source orelse .embedded_module,
    );
    try runCastF32ToF16Smoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.cast_f32_to_f16_function.?,
        backend.cast_f32_to_f16_source orelse .embedded_module,
    );
    try runKvWriteF16Smoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.kv_write_f16_function.?,
        backend.kv_write_f16_source orelse .embedded_module,
    );
    try runRmsNormSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.rmsnorm_function.?,
        backend.rmsnorm_source orelse .embedded_module,
    );
    try runRopeStoreF16Smoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.rope_store_f16_function.?,
        backend.rope_store_f16_source orelse .embedded_module,
    );
    try runSiluSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.silu_function.?,
        backend.silu_source orelse .embedded_module,
    );
    try runSiluMulSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.silu_mul_function.?,
        backend.silu_mul_source orelse .embedded_module,
    );
    try runGeluMulSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.gelu_mul_function.?,
        backend.gelu_mul_source orelse .embedded_module,
    );
    try runShortConvStepSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.shortconv_step_function.?,
        backend.shortconv_step_source orelse .embedded_module,
    );
    try runF32KvAttentionSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.attn_scores_heads_f32_function.?,
        backend.softmax_rows_function.?,
        backend.attn_weighted_sum_heads_f32_function.?,
    );
    try runArgmaxSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.argmax_function.?,
        backend.argmax_source orelse .embedded_module,
    );
    try runMatmulU16Smoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.matmul_f16_function.?,
        backend.matmul_f16_source orelse .embedded_module,
        backend.matmul_bf16_function.?,
        backend.matmul_bf16_source orelse .embedded_module,
    );
    try runMatvecU16Smoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.matvec_f16_function.?,
        backend.matvec_f16_source orelse .embedded_module,
        backend.matvec_bf16_function.?,
        backend.matvec_bf16_source orelse .embedded_module,
    );
    try runMatvecU16GateUpSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.matvec_gate_up_f16_function.?,
        backend.matvec_gate_up_f16_source orelse .embedded_module,
        backend.matvec_gate_up_bf16_function.?,
        backend.matvec_gate_up_bf16_source orelse .embedded_module,
    );
    try runMatvecU16QkvSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.matvec_qkv_f16_function.?,
        backend.matvec_qkv_f16_source orelse .embedded_module,
        backend.matvec_qkv_bf16_function.?,
        backend.matvec_qkv_bf16_source orelse .embedded_module,
    );
    try runGaffineU4MatvecSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.gaffine_u4_matvec_function.?,
        backend.gaffine_u4_matvec_source orelse .embedded_module,
    );
    try runGaffineU4MatvecGateUpSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.gaffine_u4_matvec_gate_up_function.?,
        backend.gaffine_u4_matvec_gate_up_source orelse .embedded_module,
    );
    try runGaffineU4MatvecQkvSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.gaffine_u4_matvec_qkv_function.?,
        backend.gaffine_u4_matvec_qkv_source orelse .embedded_module,
    );
    try runF16KvAttentionSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.cast_f32_to_f16_function.?,
        backend.attn_scores_heads_f16_kv_function.?,
        backend.softmax_rows_function.?,
        backend.attn_weighted_sum_heads_f16_kv_function.?,
    );
    try runF16KvAttentionFusedSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.cast_f32_to_f16_function.?,
        backend.attn_fused_heads_f16_kv_function.?,
    );
}

pub fn probeGaffineU4SequenceRowsSupport(backend: anytype) !bool {
    if (!backend.device.supportsModuleLaunch()) return false;
    const function = backend.gaffine_u4_matvec_function orelse return false;

    const in_dim: u32 = 8;
    const out_dim: u32 = 2;
    const group_size: u32 = 8;
    const batch_rows: u32 = 2;

    const input = [_]f32{
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    };
    const packed_words = [_]u32{
        0x7654_3210, // row 0 => 0..7
        0x0123_4567, // row 1 => 7..0
    };
    const scales = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(1.0),
    };
    const biases = [_]u16{
        dtype.f32ToBf16(0.0),
        dtype.f32ToBf16(0.0),
    };
    const expected = [_]f32{
        28.0,  28.0,
        168.0, 84.0,
    };
    var actual = [_]f32{0.0} ** (out_dim * batch_rows);

    var input_dev = try backend.device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(&backend.device);
    var packed_dev = try backend.device.allocBuffer(packed_words.len * @sizeOf(u32));
    defer packed_dev.deinit(&backend.device);
    var scales_dev = try backend.device.allocBuffer(scales.len * @sizeOf(u16));
    defer scales_dev.deinit(&backend.device);
    var biases_dev = try backend.device.allocBuffer(biases.len * @sizeOf(u16));
    defer biases_dev.deinit(&backend.device);
    var out_dev = try backend.device.allocBuffer(actual.len * @sizeOf(f32));
    defer out_dev.deinit(&backend.device);

    try input_dev.upload(&backend.device, std.mem.sliceAsBytes(input[0..]));
    try packed_dev.upload(&backend.device, std.mem.sliceAsBytes(packed_words[0..]));
    try scales_dev.upload(&backend.device, std.mem.sliceAsBytes(scales[0..]));
    try biases_dev.upload(&backend.device, std.mem.sliceAsBytes(biases[0..]));

    try compute.cuda.gaffine_u4_matvec.runWithFunction(
        &backend.kernel_arg_pack,
        &backend.device,
        function,
        &input_dev,
        &packed_dev,
        &scales_dev,
        &biases_dev,
        &out_dev,
        in_dim,
        out_dim,
        group_size,
        gaffine_scales_dtype_bf16,
        batch_rows,
    );
    try out_dev.download(&backend.device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        if (@abs(want - got) > 0.02) return false;
    }

    return true;
}

pub fn probeGaffineU4SequenceFusedQkvSupport(backend: anytype) !bool {
    if (!backend.device.supportsModuleLaunch()) return false;
    const function = backend.gaffine_u4_matvec_qkv_function orelse return false;

    const in_dim: u32 = 8;
    const out_dim: u32 = 2;
    const group_size: u32 = 8;
    const batch_rows: u32 = 2;

    const input = [_]f32{
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    };
    const packed_words = [_]u32{
        0x7654_3210, // row 0 => 0..7
        0x0123_4567, // row 1 => 7..0
    };
    const q_scales = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(1.0),
    };
    const q_biases = [_]u16{
        dtype.f32ToBf16(0.0),
        dtype.f32ToBf16(0.0),
    };
    const k_scales = [_]u16{
        dtype.f32ToBf16(2.0),
        dtype.f32ToBf16(2.0),
    };
    const k_biases = [_]u16{
        dtype.f32ToBf16(0.0),
        dtype.f32ToBf16(0.0),
    };
    const v_scales = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(1.0),
    };
    const v_biases = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(1.0),
    };
    const expected_q = [_]f32{
        28.0,  28.0,
        168.0, 84.0,
    };
    const expected_k = [_]f32{
        56.0,  56.0,
        336.0, 168.0,
    };
    const expected_v = [_]f32{
        36.0,  36.0,
        204.0, 120.0,
    };
    var actual_q = [_]f32{0.0} ** @as(usize, out_dim * batch_rows);
    var actual_k = [_]f32{0.0} ** @as(usize, out_dim * batch_rows);
    var actual_v = [_]f32{0.0} ** @as(usize, out_dim * batch_rows);

    var input_dev = try backend.device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(&backend.device);
    var q_packed_dev = try backend.device.allocBuffer(packed_words.len * @sizeOf(u32));
    defer q_packed_dev.deinit(&backend.device);
    var q_scales_dev = try backend.device.allocBuffer(q_scales.len * @sizeOf(u16));
    defer q_scales_dev.deinit(&backend.device);
    var q_biases_dev = try backend.device.allocBuffer(q_biases.len * @sizeOf(u16));
    defer q_biases_dev.deinit(&backend.device);
    var q_out_dev = try backend.device.allocBuffer(actual_q.len * @sizeOf(f32));
    defer q_out_dev.deinit(&backend.device);

    var k_packed_dev = try backend.device.allocBuffer(packed_words.len * @sizeOf(u32));
    defer k_packed_dev.deinit(&backend.device);
    var k_scales_dev = try backend.device.allocBuffer(k_scales.len * @sizeOf(u16));
    defer k_scales_dev.deinit(&backend.device);
    var k_biases_dev = try backend.device.allocBuffer(k_biases.len * @sizeOf(u16));
    defer k_biases_dev.deinit(&backend.device);
    var k_out_dev = try backend.device.allocBuffer(actual_k.len * @sizeOf(f32));
    defer k_out_dev.deinit(&backend.device);

    var v_packed_dev = try backend.device.allocBuffer(packed_words.len * @sizeOf(u32));
    defer v_packed_dev.deinit(&backend.device);
    var v_scales_dev = try backend.device.allocBuffer(v_scales.len * @sizeOf(u16));
    defer v_scales_dev.deinit(&backend.device);
    var v_biases_dev = try backend.device.allocBuffer(v_biases.len * @sizeOf(u16));
    defer v_biases_dev.deinit(&backend.device);
    var v_out_dev = try backend.device.allocBuffer(actual_v.len * @sizeOf(f32));
    defer v_out_dev.deinit(&backend.device);

    try input_dev.upload(&backend.device, std.mem.sliceAsBytes(input[0..]));
    try q_packed_dev.upload(&backend.device, std.mem.sliceAsBytes(packed_words[0..]));
    try q_scales_dev.upload(&backend.device, std.mem.sliceAsBytes(q_scales[0..]));
    try q_biases_dev.upload(&backend.device, std.mem.sliceAsBytes(q_biases[0..]));
    try k_packed_dev.upload(&backend.device, std.mem.sliceAsBytes(packed_words[0..]));
    try k_scales_dev.upload(&backend.device, std.mem.sliceAsBytes(k_scales[0..]));
    try k_biases_dev.upload(&backend.device, std.mem.sliceAsBytes(k_biases[0..]));
    try v_packed_dev.upload(&backend.device, std.mem.sliceAsBytes(packed_words[0..]));
    try v_scales_dev.upload(&backend.device, std.mem.sliceAsBytes(v_scales[0..]));
    try v_biases_dev.upload(&backend.device, std.mem.sliceAsBytes(v_biases[0..]));

    try compute.cuda.gaffine_u4_matvec_qkv.runWithFunction(
        &backend.kernel_arg_pack,
        &backend.device,
        function,
        &input_dev,
        &q_packed_dev,
        &q_scales_dev,
        &q_biases_dev,
        &q_out_dev,
        out_dim,
        group_size,
        gaffine_scales_dtype_bf16,
        &k_packed_dev,
        &k_scales_dev,
        &k_biases_dev,
        &k_out_dev,
        out_dim,
        group_size,
        gaffine_scales_dtype_bf16,
        &v_packed_dev,
        &v_scales_dev,
        &v_biases_dev,
        &v_out_dev,
        out_dim,
        group_size,
        gaffine_scales_dtype_bf16,
        in_dim,
        batch_rows,
    );

    try q_out_dev.download(&backend.device, std.mem.sliceAsBytes(actual_q[0..]));
    try k_out_dev.download(&backend.device, std.mem.sliceAsBytes(actual_k[0..]));
    try v_out_dev.download(&backend.device, std.mem.sliceAsBytes(actual_v[0..]));

    for (expected_q, actual_q) |want, got| {
        if (@abs(want - got) > 0.02) return false;
    }
    for (expected_k, actual_k) |want, got| {
        if (@abs(want - got) > 0.02) return false;
    }
    for (expected_v, actual_v) |want, got| {
        if (@abs(want - got) > 0.02) return false;
    }

    return true;
}

pub fn probeGaffineU4SequenceFusedGateUpSupport(backend: anytype) !bool {
    if (!backend.device.supportsModuleLaunch()) return false;
    const function = backend.gaffine_u4_matvec_gate_up_function orelse return false;

    const in_dim: u32 = 8;
    const out_dim: u32 = 2;
    const group_size: u32 = 8;
    const batch_rows: u32 = 2;

    const input = [_]f32{
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    };
    const packed_words = [_]u32{
        0x7654_3210, // row 0 => 0..7
        0x0123_4567, // row 1 => 7..0
    };
    const gate_scales = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(1.0),
    };
    const gate_biases = [_]u16{
        dtype.f32ToBf16(0.0),
        dtype.f32ToBf16(0.0),
    };
    const up_scales = [_]u16{
        dtype.f32ToBf16(2.0),
        dtype.f32ToBf16(2.0),
    };
    const up_biases = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(1.0),
    };
    const expected_gate = [_]f32{
        28.0,  28.0,
        168.0, 84.0,
    };
    const expected_up = [_]f32{
        64.0,  64.0,
        372.0, 204.0,
    };
    var actual_gate = [_]f32{0.0} ** @as(usize, out_dim * batch_rows);
    var actual_up = [_]f32{0.0} ** @as(usize, out_dim * batch_rows);

    var input_dev = try backend.device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(&backend.device);
    var gate_packed_dev = try backend.device.allocBuffer(packed_words.len * @sizeOf(u32));
    defer gate_packed_dev.deinit(&backend.device);
    var gate_scales_dev = try backend.device.allocBuffer(gate_scales.len * @sizeOf(u16));
    defer gate_scales_dev.deinit(&backend.device);
    var gate_biases_dev = try backend.device.allocBuffer(gate_biases.len * @sizeOf(u16));
    defer gate_biases_dev.deinit(&backend.device);
    var gate_out_dev = try backend.device.allocBuffer(actual_gate.len * @sizeOf(f32));
    defer gate_out_dev.deinit(&backend.device);

    var up_packed_dev = try backend.device.allocBuffer(packed_words.len * @sizeOf(u32));
    defer up_packed_dev.deinit(&backend.device);
    var up_scales_dev = try backend.device.allocBuffer(up_scales.len * @sizeOf(u16));
    defer up_scales_dev.deinit(&backend.device);
    var up_biases_dev = try backend.device.allocBuffer(up_biases.len * @sizeOf(u16));
    defer up_biases_dev.deinit(&backend.device);
    var up_out_dev = try backend.device.allocBuffer(actual_up.len * @sizeOf(f32));
    defer up_out_dev.deinit(&backend.device);

    try input_dev.upload(&backend.device, std.mem.sliceAsBytes(input[0..]));
    try gate_packed_dev.upload(&backend.device, std.mem.sliceAsBytes(packed_words[0..]));
    try gate_scales_dev.upload(&backend.device, std.mem.sliceAsBytes(gate_scales[0..]));
    try gate_biases_dev.upload(&backend.device, std.mem.sliceAsBytes(gate_biases[0..]));
    try up_packed_dev.upload(&backend.device, std.mem.sliceAsBytes(packed_words[0..]));
    try up_scales_dev.upload(&backend.device, std.mem.sliceAsBytes(up_scales[0..]));
    try up_biases_dev.upload(&backend.device, std.mem.sliceAsBytes(up_biases[0..]));

    try compute.cuda.gaffine_u4_matvec_gate_up.runWithFunction(
        &backend.kernel_arg_pack,
        &backend.device,
        function,
        &input_dev,
        &gate_packed_dev,
        &gate_scales_dev,
        &gate_biases_dev,
        &gate_out_dev,
        out_dim,
        group_size,
        gaffine_scales_dtype_bf16,
        &up_packed_dev,
        &up_scales_dev,
        &up_biases_dev,
        &up_out_dev,
        out_dim,
        group_size,
        gaffine_scales_dtype_bf16,
        in_dim,
        batch_rows,
    );

    try gate_out_dev.download(&backend.device, std.mem.sliceAsBytes(actual_gate[0..]));
    try up_out_dev.download(&backend.device, std.mem.sliceAsBytes(actual_up[0..]));

    for (expected_gate, actual_gate) |want, got| {
        if (@abs(want - got) > 0.02) return false;
    }
    for (expected_up, actual_up) |want, got| {
        if (@abs(want - got) > 0.02) return false;
    }

    return true;
}

fn runVectorAddSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const lhs = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const rhs = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    const expected = [_]f32{ 11.0, 22.0, 33.0, 44.0 };
    var actual = [_]f32{0.0} ** lhs.len;

    var lhs_dev = try device.allocBuffer(lhs.len * @sizeOf(f32));
    defer lhs_dev.deinit(device);
    var rhs_dev = try device.allocBuffer(rhs.len * @sizeOf(f32));
    defer rhs_dev.deinit(device);
    var out_dev = try device.allocBuffer(actual.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try lhs_dev.upload(device, std.mem.sliceAsBytes(lhs[0..]));
    try rhs_dev.upload(device, std.mem.sliceAsBytes(rhs[0..]));
    try compute.cuda.vector_add.runWithFunction(
        arg_pack,
        device,
        function,
        &lhs_dev,
        &rhs_dev,
        &out_dev,
        @intCast(lhs.len),
    );
    try out_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        if (@abs(want - got) > 0.0001) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA vector_add smoke passed", .{
        .n = lhs.len,
        .source = @tagName(source),
        .out0 = actual[0],
    });
}

fn runMulSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const lhs = [_]f32{ 1.0, -2.0, 3.0, -4.0 };
    const rhs = [_]f32{ 10.0, 20.0, -30.0, -40.0 };
    const expected = [_]f32{ 10.0, -40.0, -90.0, 160.0 };
    var actual = [_]f32{0.0} ** lhs.len;

    var lhs_dev = try device.allocBuffer(lhs.len * @sizeOf(f32));
    defer lhs_dev.deinit(device);
    var rhs_dev = try device.allocBuffer(rhs.len * @sizeOf(f32));
    defer rhs_dev.deinit(device);
    var out_dev = try device.allocBuffer(actual.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try lhs_dev.upload(device, std.mem.sliceAsBytes(lhs[0..]));
    try rhs_dev.upload(device, std.mem.sliceAsBytes(rhs[0..]));
    try compute.cuda.mul.runWithFunction(
        arg_pack,
        device,
        function,
        &lhs_dev,
        &rhs_dev,
        &out_dev,
        @intCast(lhs.len),
    );
    try out_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        if (@abs(want - got) > 0.0001) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA mul smoke passed", .{
        .n = lhs.len,
        .source = @tagName(source),
        .out0 = actual[0],
    });
}

fn runCopySmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const input = [_]f32{ 5.5, -2.0, 9.25, 0.125 };
    var actual = [_]f32{0.0} ** input.len;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var out_dev = try device.allocBuffer(actual.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try compute.cuda.copy.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &out_dev,
        @intCast(input.len),
    );
    try out_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (input, actual) |want, got| {
        if (@abs(want - got) > 0.0001) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA copy smoke passed", .{
        .n = input.len,
        .source = @tagName(source),
        .out0 = actual[0],
    });
}

fn runCopyU16Smoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const input = [_]u16{
        @bitCast(@as(f16, 1.0)),
        @bitCast(@as(f16, -2.5)),
        @bitCast(@as(f16, 3.25)),
        @bitCast(@as(f16, 0.125)),
    };
    var actual = [_]u16{0} ** input.len;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(u16));
    defer input_dev.deinit(device);
    var out_dev = try device.allocBuffer(actual.len * @sizeOf(u16));
    defer out_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try compute.cuda.copy_u16.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &out_dev,
        @intCast(input.len),
    );
    try out_dev.download(device, std.mem.sliceAsBytes(actual[0..]));
    try std.testing.expectEqualSlices(u16, input[0..], actual[0..]);

    log.info("inference", "CUDA copy_u16 smoke passed", .{
        .n = input.len,
        .source = @tagName(source),
    });
}

fn runCastF32ToF16Smoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const input = [_]f32{ 1.0, -2.5, 3.25, 0.125 };
    var output_bits = [_]u16{0} ** input.len;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var output_dev = try device.allocBuffer(output_bits.len * @sizeOf(u16));
    defer output_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try compute.cuda.cast_f32_to_f16.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &output_dev,
        @intCast(input.len),
    );
    try output_dev.download(device, std.mem.sliceAsBytes(output_bits[0..]));

    for (input, output_bits) |want, got_bits| {
        const got = dtype.fp16ToF32(got_bits);
        if (@abs(want - got) > 0.001) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA cast_f32_to_f16 smoke passed", .{
        .n = input.len,
        .source = @tagName(source),
        .out0 = dtype.fp16ToF32(output_bits[0]),
    });
}

fn runKvWriteF16Smoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const n_heads: u32 = 1;
    const head_dim: u32 = 4;
    const rope_dim: u32 = 4;
    const position: u32 = 3;
    const theta: f32 = 10000.0;

    const k_input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const v_input = [_]f32{ 5.0, -1.0, 0.5, 2.0 };
    var expected_k = k_input;
    const half: usize = @intCast(rope_dim / 2);
    var pair: usize = 0;
    while (pair < half) : (pair += 1) {
        const pair_f: f32 = @floatFromInt(pair);
        const rope_dim_f: f32 = @floatFromInt(rope_dim);
        const inv_freq = std.math.pow(f32, theta, -2.0 * pair_f / rope_dim_f);
        const angle = @as(f32, @floatFromInt(position)) * inv_freq;
        const s = @sin(angle);
        const c = @cos(angle);
        const lo = pair;
        const hi = half + pair;
        const x0 = k_input[lo];
        const x1 = k_input[hi];
        expected_k[lo] = x0 * c - x1 * s;
        expected_k[hi] = x0 * s + x1 * c;
    }

    var out_k_bits = [_]u16{0} ** k_input.len;
    var out_v_bits = [_]u16{0} ** v_input.len;
    var out_k = [_]f32{0.0} ** k_input.len;
    var out_v = [_]f32{0.0} ** v_input.len;

    var input_k_dev = try device.allocBuffer(k_input.len * @sizeOf(f32));
    defer input_k_dev.deinit(device);
    var input_v_dev = try device.allocBuffer(v_input.len * @sizeOf(f32));
    defer input_v_dev.deinit(device);
    var out_k_dev = try device.allocBuffer(out_k_bits.len * @sizeOf(u16));
    defer out_k_dev.deinit(device);
    var out_v_dev = try device.allocBuffer(out_v_bits.len * @sizeOf(u16));
    defer out_v_dev.deinit(device);

    try input_k_dev.upload(device, std.mem.sliceAsBytes(k_input[0..]));
    try input_v_dev.upload(device, std.mem.sliceAsBytes(v_input[0..]));
    try compute.cuda.kv_write_f16.runWithFunction(
        arg_pack,
        device,
        function,
        &input_k_dev,
        &input_v_dev,
        &out_k_dev,
        &out_v_dev,
        n_heads,
        head_dim,
        rope_dim,
        position,
        theta,
    );
    try out_k_dev.download(device, std.mem.sliceAsBytes(out_k_bits[0..]));
    try out_v_dev.download(device, std.mem.sliceAsBytes(out_v_bits[0..]));

    for (out_k_bits, 0..) |bits, i| out_k[i] = dtype.fp16ToF32(bits);
    for (out_v_bits, 0..) |bits, i| out_v[i] = dtype.fp16ToF32(bits);

    for (expected_k, out_k) |want, got| {
        if (@abs(want - got) > 0.01) return error.CudaKernelSmokeMismatch;
    }
    for (v_input, out_v) |want, got| {
        if (@abs(want - got) > 0.01) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA kv_write_f16 smoke passed", .{
        .source = @tagName(source),
        .k0 = out_k[0],
        .v0 = out_v[0],
    });
}

fn runRopeStoreF16Smoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const n_heads: u32 = 1;
    const head_dim: u32 = 4;
    const rope_dim: u32 = 4;
    const position: u32 = 3;
    const theta: f32 = 10000.0;

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var expected = input;
    const half: usize = @intCast(rope_dim / 2);
    var pair: usize = 0;
    while (pair < half) : (pair += 1) {
        const pair_f: f32 = @floatFromInt(pair);
        const rope_dim_f: f32 = @floatFromInt(rope_dim);
        const inv_freq = std.math.pow(f32, theta, -2.0 * pair_f / rope_dim_f);
        const angle = @as(f32, @floatFromInt(position)) * inv_freq;
        const s = @sin(angle);
        const c = @cos(angle);
        const lo = pair;
        const hi = half + pair;
        const x0 = input[lo];
        const x1 = input[hi];
        expected[lo] = x0 * c - x1 * s;
        expected[hi] = x0 * s + x1 * c;
    }

    var output_bits = [_]u16{0} ** input.len;
    var output = [_]f32{0.0} ** input.len;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var output_dev = try device.allocBuffer(output_bits.len * @sizeOf(u16));
    defer output_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try compute.cuda.rope_store_f16.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &output_dev,
        n_heads,
        head_dim,
        rope_dim,
        position,
        theta,
    );
    try output_dev.download(device, std.mem.sliceAsBytes(output_bits[0..]));

    for (output_bits, 0..) |bits, i| {
        output[i] = dtype.fp16ToF32(bits);
    }
    for (expected, output) |want, got| {
        if (@abs(want - got) > 0.01) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA rope_store_f16 smoke passed", .{
        .source = @tagName(source),
        .out0 = output[0],
    });
}

fn runRmsNormSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const rows: u32 = 2;
    const cols: u32 = 4;
    const eps: f32 = 1e-5;

    const input = [_]f32{
        1.0,  2.0, 3.0, 4.0,
        -1.0, 0.0, 1.0, 2.0,
    };
    const weight = [_]f32{ 1.0, 1.5, 0.5, 2.0 };
    var expected = [_]f32{0.0} ** input.len;
    var actual = [_]f32{0.0} ** input.len;

    computeRmsNormReference(&expected, &input, &weight, rows, cols, eps);

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var weight_dev = try device.allocBuffer(weight.len * @sizeOf(f32));
    defer weight_dev.deinit(device);
    var output_dev = try device.allocBuffer(actual.len * @sizeOf(f32));
    defer output_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try weight_dev.upload(device, std.mem.sliceAsBytes(weight[0..]));
    try compute.cuda.rmsnorm.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &weight_dev,
        &output_dev,
        rows,
        cols,
        eps,
        0.0,
    );
    try output_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        if (@abs(want - got) > 0.001) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA rmsnorm smoke passed", .{
        .rows = rows,
        .cols = cols,
        .source = @tagName(source),
        .out0 = actual[0],
    });
}

fn runSiluSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const input = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    const expected = [_]f32{
        -0.26894143,
        0.0,
        0.7310586,
        1.7615942,
    };
    var actual = [_]f32{0.0} ** input.len;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var output_dev = try device.allocBuffer(actual.len * @sizeOf(f32));
    defer output_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try compute.cuda.silu.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &output_dev,
        @intCast(input.len),
    );
    try output_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        if (@abs(want - got) > 0.001) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA silu smoke passed", .{
        .n = input.len,
        .source = @tagName(source),
        .out0 = actual[0],
    });
}

fn runSiluMulSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const gate = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    const up = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    const expected = [_]f32{
        -0.26894143 * 2.0,
        0.0 * 3.0,
        0.7310586 * 4.0,
        1.7615942 * 5.0,
    };
    var actual = [_]f32{0.0} ** gate.len;

    var gate_dev = try device.allocBuffer(gate.len * @sizeOf(f32));
    defer gate_dev.deinit(device);
    var up_dev = try device.allocBuffer(up.len * @sizeOf(f32));
    defer up_dev.deinit(device);
    var out_dev = try device.allocBuffer(actual.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try gate_dev.upload(device, std.mem.sliceAsBytes(gate[0..]));
    try up_dev.upload(device, std.mem.sliceAsBytes(up[0..]));
    try compute.cuda.silu_mul.runWithFunction(
        arg_pack,
        device,
        function,
        &gate_dev,
        &up_dev,
        &out_dev,
        @intCast(gate.len),
    );
    try out_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        if (@abs(want - got) > 0.001) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA silu_mul smoke passed", .{
        .n = gate.len,
        .source = @tagName(source),
        .out0 = actual[0],
    });
}

fn runGeluMulSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const gate = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    const up = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    var expected = [_]f32{0.0} ** gate.len;
    var actual = [_]f32{0.0} ** gate.len;

    const sqrt_2_over_pi: f32 = 0.7978845608028654;
    const coeff: f32 = 0.044715;
    for (gate, up, 0..) |g, u, idx| {
        const x3 = g * g * g;
        const inner = sqrt_2_over_pi * (g + coeff * x3);
        const gelu_g = 0.5 * g * (1.0 + std.math.tanh(inner));
        expected[idx] = gelu_g * u;
    }

    var gate_dev = try device.allocBuffer(gate.len * @sizeOf(f32));
    defer gate_dev.deinit(device);
    var up_dev = try device.allocBuffer(up.len * @sizeOf(f32));
    defer up_dev.deinit(device);
    var out_dev = try device.allocBuffer(actual.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try gate_dev.upload(device, std.mem.sliceAsBytes(gate[0..]));
    try up_dev.upload(device, std.mem.sliceAsBytes(up[0..]));
    try compute.cuda.gelu_mul.runWithFunction(
        arg_pack,
        device,
        function,
        &gate_dev,
        &up_dev,
        &out_dev,
        @intCast(gate.len),
    );
    try out_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        if (@abs(want - got) > 0.001) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA gelu_mul smoke passed", .{
        .n = gate.len,
        .source = @tagName(source),
        .out0 = actual[0],
    });
}

fn runShortConvStepSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const conv_dim: usize = 4;
    const d_conv: usize = 3;
    const b_gate = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const c_gate = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const x_proj = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    // [d_conv, conv_dim] time-major. Only newest tap contributes.
    const weight_time_major = [_]f32{
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
    };
    const expected = [_]f32{ 10.0, 40.0, 90.0, 160.0 };
    var state = [_]f32{0.0} ** (conv_dim * d_conv);
    var out = [_]f32{0.0} ** conv_dim;

    var b_dev = try device.allocBuffer(b_gate.len * @sizeOf(f32));
    defer b_dev.deinit(device);
    var c_dev = try device.allocBuffer(c_gate.len * @sizeOf(f32));
    defer c_dev.deinit(device);
    var x_dev = try device.allocBuffer(x_proj.len * @sizeOf(f32));
    defer x_dev.deinit(device);
    var w_dev = try device.allocBuffer(weight_time_major.len * @sizeOf(f32));
    defer w_dev.deinit(device);
    var state_dev = try device.allocBuffer(state.len * @sizeOf(f32));
    defer state_dev.deinit(device);
    var out_dev = try device.allocBuffer(out.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try b_dev.upload(device, std.mem.sliceAsBytes(b_gate[0..]));
    try c_dev.upload(device, std.mem.sliceAsBytes(c_gate[0..]));
    try x_dev.upload(device, std.mem.sliceAsBytes(x_proj[0..]));
    try w_dev.upload(device, std.mem.sliceAsBytes(weight_time_major[0..]));
    try state_dev.upload(device, std.mem.sliceAsBytes(state[0..]));

    try compute.cuda.shortconv_step.runWithFunction(
        arg_pack,
        device,
        function,
        &b_dev,
        &c_dev,
        &x_dev,
        &state_dev,
        &w_dev,
        null,
        &out_dev,
        @intCast(conv_dim),
        @intCast(d_conv),
    );
    try out_dev.download(device, std.mem.sliceAsBytes(out[0..]));
    try state_dev.download(device, std.mem.sliceAsBytes(state[0..]));

    for (expected, out) |want, got| {
        if (@abs(want - got) > 0.001) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA shortconv_step smoke passed", .{
        .conv_dim = conv_dim,
        .d_conv = d_conv,
        .source = @tagName(source),
        .out0 = out[0],
        .state_last0 = state[(d_conv - 1) * conv_dim],
    });
}

fn runF32KvAttentionSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    scores_heads_f32_function: compute.cuda.Function,
    softmax_rows_function: compute.cuda.Function,
    weighted_sum_heads_f32_function: compute.cuda.Function,
) !void {
    const n_heads: u32 = 2;
    const kv_groups: u32 = 2;
    const head_dim: u32 = 4;
    const seq_len: u32 = 2;
    const row_stride: u32 = 4;
    const scale: f32 = 0.5;

    const query = [_]f32{
        1.0, 0.0, 0.0, 0.0, // head 0
        0.0, 1.0, 0.0, 0.0, // head 1
    };
    const key_cache = [_]f32{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    };
    const value_cache = [_]f32{
        2.0, 3.0, 0.0, 0.0,
        4.0, 1.0, 0.0, 0.0,
    };
    var probs = [_]f32{0.0} ** (n_heads * seq_len);
    var out = [_]f32{0.0} ** (n_heads * head_dim);

    var query_dev = try device.allocBuffer(query.len * @sizeOf(f32));
    defer query_dev.deinit(device);
    var key_dev = try device.allocBuffer(key_cache.len * @sizeOf(f32));
    defer key_dev.deinit(device);
    var value_dev = try device.allocBuffer(value_cache.len * @sizeOf(f32));
    defer value_dev.deinit(device);
    var scores_dev = try device.allocBuffer(probs.len * @sizeOf(f32));
    defer scores_dev.deinit(device);
    var probs_dev = try device.allocBuffer(probs.len * @sizeOf(f32));
    defer probs_dev.deinit(device);
    var out_dev = try device.allocBuffer(out.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try query_dev.upload(device, std.mem.sliceAsBytes(query[0..]));
    try key_dev.upload(device, std.mem.sliceAsBytes(key_cache[0..]));
    try value_dev.upload(device, std.mem.sliceAsBytes(value_cache[0..]));

    try compute.cuda.attn_scores_heads_f32.runWithFunction(
        arg_pack,
        device,
        scores_heads_f32_function,
        &query_dev,
        &key_dev,
        &scores_dev,
        n_heads,
        seq_len,
        row_stride,
        kv_groups,
        head_dim,
        scale,
    );
    try compute.cuda.softmax_rows.runWithFunction(
        arg_pack,
        device,
        softmax_rows_function,
        &scores_dev,
        &probs_dev,
        n_heads,
        seq_len,
    );
    try compute.cuda.attn_weighted_sum_heads_f32.runWithFunction(
        arg_pack,
        device,
        weighted_sum_heads_f32_function,
        &probs_dev,
        &value_dev,
        &out_dev,
        n_heads,
        seq_len,
        row_stride,
        kv_groups,
        head_dim,
    );
    try out_dev.download(device, std.mem.sliceAsBytes(out[0..]));
    try probs_dev.download(device, std.mem.sliceAsBytes(probs[0..]));

    const expected_p0 = std.math.exp(0.5) / (std.math.exp(0.5) + std.math.exp(0.0));
    const expected_p1 = 1.0 - expected_p0;
    if (@abs(probs[0] - expected_p0) > 0.01) return error.CudaKernelSmokeMismatch;
    if (@abs(probs[1] - expected_p1) > 0.01) return error.CudaKernelSmokeMismatch;
    if (@abs(probs[2] - expected_p1) > 0.01) return error.CudaKernelSmokeMismatch;
    if (@abs(probs[3] - expected_p0) > 0.01) return error.CudaKernelSmokeMismatch;

    const expected_out_h0_d0 = expected_p0 * 2.0 + expected_p1 * 4.0;
    const expected_out_h0_d1 = expected_p0 * 3.0 + expected_p1 * 1.0;
    const expected_out_h1_d0 = expected_p1 * 2.0 + expected_p0 * 4.0;
    const expected_out_h1_d1 = expected_p1 * 3.0 + expected_p0 * 1.0;
    if (@abs(out[0] - expected_out_h0_d0) > 0.03) return error.CudaKernelSmokeMismatch;
    if (@abs(out[1] - expected_out_h0_d1) > 0.03) return error.CudaKernelSmokeMismatch;
    if (@abs(out[4] - expected_out_h1_d0) > 0.03) return error.CudaKernelSmokeMismatch;
    if (@abs(out[5] - expected_out_h1_d1) > 0.03) return error.CudaKernelSmokeMismatch;

    log.info("inference", "CUDA attention f32-kv smoke passed", .{
        .n_heads = n_heads,
        .seq = seq_len,
        .head_dim = head_dim,
        .out0 = out[0],
    });
}

fn runF16KvAttentionSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    cast_f32_to_f16_function: compute.cuda.Function,
    scores_heads_f16_kv_function: compute.cuda.Function,
    softmax_rows_function: compute.cuda.Function,
    weighted_sum_heads_f16_kv_function: compute.cuda.Function,
) !void {
    const n_heads: u32 = 2;
    const kv_groups: u32 = 2;
    const head_dim: u32 = 4;
    const seq_len: u32 = 2;
    const row_stride: u32 = 4;
    const scale: f32 = 0.5;

    const query = [_]f32{
        1.0, 0.0, 0.0, 0.0, // head 0
        0.0, 1.0, 0.0, 0.0, // head 1
    };
    const key_cache_f32 = [_]f32{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    };
    const value_cache_f32 = [_]f32{
        2.0, 3.0, 0.0, 0.0,
        4.0, 1.0, 0.0, 0.0,
    };
    var probs = [_]f32{0.0} ** (n_heads * seq_len);
    var out = [_]f32{0.0} ** (n_heads * head_dim);

    var query_dev = try device.allocBuffer(query.len * @sizeOf(f32));
    defer query_dev.deinit(device);
    var key_f32_dev = try device.allocBuffer(key_cache_f32.len * @sizeOf(f32));
    defer key_f32_dev.deinit(device);
    var value_f32_dev = try device.allocBuffer(value_cache_f32.len * @sizeOf(f32));
    defer value_f32_dev.deinit(device);
    var key_f16_dev = try device.allocBuffer(key_cache_f32.len * @sizeOf(u16));
    defer key_f16_dev.deinit(device);
    var value_f16_dev = try device.allocBuffer(value_cache_f32.len * @sizeOf(u16));
    defer value_f16_dev.deinit(device);
    var scores_dev = try device.allocBuffer(probs.len * @sizeOf(f32));
    defer scores_dev.deinit(device);
    var probs_dev = try device.allocBuffer(probs.len * @sizeOf(f32));
    defer probs_dev.deinit(device);
    var out_dev = try device.allocBuffer(out.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try query_dev.upload(device, std.mem.sliceAsBytes(query[0..]));
    try key_f32_dev.upload(device, std.mem.sliceAsBytes(key_cache_f32[0..]));
    try value_f32_dev.upload(device, std.mem.sliceAsBytes(value_cache_f32[0..]));
    try compute.cuda.cast_f32_to_f16.runWithFunction(
        arg_pack,
        device,
        cast_f32_to_f16_function,
        &key_f32_dev,
        &key_f16_dev,
        @intCast(key_cache_f32.len),
    );
    try compute.cuda.cast_f32_to_f16.runWithFunction(
        arg_pack,
        device,
        cast_f32_to_f16_function,
        &value_f32_dev,
        &value_f16_dev,
        @intCast(value_cache_f32.len),
    );
    try compute.cuda.attn_scores_heads_f16_kv.runWithFunction(
        arg_pack,
        device,
        scores_heads_f16_kv_function,
        &query_dev,
        &key_f16_dev,
        &scores_dev,
        n_heads,
        seq_len,
        row_stride,
        kv_groups,
        head_dim,
        scale,
    );
    try compute.cuda.softmax_rows.runWithFunction(
        arg_pack,
        device,
        softmax_rows_function,
        &scores_dev,
        &probs_dev,
        n_heads,
        seq_len,
    );
    try compute.cuda.attn_weighted_sum_heads_f16_kv.runWithFunction(
        arg_pack,
        device,
        weighted_sum_heads_f16_kv_function,
        &probs_dev,
        &value_f16_dev,
        &out_dev,
        n_heads,
        seq_len,
        row_stride,
        kv_groups,
        head_dim,
    );
    try out_dev.download(device, std.mem.sliceAsBytes(out[0..]));
    try probs_dev.download(device, std.mem.sliceAsBytes(probs[0..]));

    const expected_p0 = std.math.exp(0.5) / (std.math.exp(0.5) + std.math.exp(0.0));
    const expected_p1 = 1.0 - expected_p0;
    if (@abs(probs[0] - expected_p0) > 0.01) return error.CudaKernelSmokeMismatch;
    if (@abs(probs[1] - expected_p1) > 0.01) return error.CudaKernelSmokeMismatch;
    if (@abs(probs[2] - expected_p1) > 0.01) return error.CudaKernelSmokeMismatch;
    if (@abs(probs[3] - expected_p0) > 0.01) return error.CudaKernelSmokeMismatch;

    const expected_out_h0_d0 = expected_p0 * 2.0 + expected_p1 * 4.0;
    const expected_out_h0_d1 = expected_p0 * 3.0 + expected_p1 * 1.0;
    const expected_out_h1_d0 = expected_p1 * 2.0 + expected_p0 * 4.0;
    const expected_out_h1_d1 = expected_p1 * 3.0 + expected_p0 * 1.0;
    if (@abs(out[0] - expected_out_h0_d0) > 0.03) return error.CudaKernelSmokeMismatch;
    if (@abs(out[1] - expected_out_h0_d1) > 0.03) return error.CudaKernelSmokeMismatch;
    if (@abs(out[4] - expected_out_h1_d0) > 0.03) return error.CudaKernelSmokeMismatch;
    if (@abs(out[5] - expected_out_h1_d1) > 0.03) return error.CudaKernelSmokeMismatch;

    log.info("inference", "CUDA attention f16-kv smoke passed", .{
        .n_heads = n_heads,
        .seq = seq_len,
        .head_dim = head_dim,
        .out0 = out[0],
    });
}

fn runF16KvAttentionFusedSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    cast_f32_to_f16_function: compute.cuda.Function,
    fused_function: compute.cuda.Function,
) !void {
    const n_heads: u32 = 2;
    const kv_groups: u32 = 2;
    const head_dim: u32 = 4;
    const seq_len: u32 = 2;
    const row_stride: u32 = 4;
    const scale: f32 = 0.5;
    const rope_dim: u32 = head_dim;
    const position: u32 = 0;
    const theta: f32 = 10000.0;

    const query = [_]f32{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    };
    const key_cache_f32 = [_]f32{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    };
    const value_cache_f32 = [_]f32{
        2.0, 3.0, 0.0, 0.0,
        4.0, 1.0, 0.0, 0.0,
    };
    var out = [_]f32{0.0} ** (n_heads * head_dim);

    var query_dev = try device.allocBuffer(query.len * @sizeOf(f32));
    defer query_dev.deinit(device);
    var key_f32_dev = try device.allocBuffer(key_cache_f32.len * @sizeOf(f32));
    defer key_f32_dev.deinit(device);
    var value_f32_dev = try device.allocBuffer(value_cache_f32.len * @sizeOf(f32));
    defer value_f32_dev.deinit(device);
    var key_f16_dev = try device.allocBuffer(key_cache_f32.len * @sizeOf(u16));
    defer key_f16_dev.deinit(device);
    var value_f16_dev = try device.allocBuffer(value_cache_f32.len * @sizeOf(u16));
    defer value_f16_dev.deinit(device);
    var out_dev = try device.allocBuffer(out.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try query_dev.upload(device, std.mem.sliceAsBytes(query[0..]));
    try key_f32_dev.upload(device, std.mem.sliceAsBytes(key_cache_f32[0..]));
    try value_f32_dev.upload(device, std.mem.sliceAsBytes(value_cache_f32[0..]));
    try compute.cuda.cast_f32_to_f16.runWithFunction(
        arg_pack,
        device,
        cast_f32_to_f16_function,
        &key_f32_dev,
        &key_f16_dev,
        @intCast(key_cache_f32.len),
    );
    try compute.cuda.cast_f32_to_f16.runWithFunction(
        arg_pack,
        device,
        cast_f32_to_f16_function,
        &value_f32_dev,
        &value_f16_dev,
        @intCast(value_cache_f32.len),
    );

    try compute.cuda.attn_fused_heads_f16_kv.runWithFunction(
        arg_pack,
        device,
        fused_function,
        &query_dev,
        &key_f16_dev,
        &value_f16_dev,
        &out_dev,
        n_heads,
        seq_len,
        row_stride,
        kv_groups,
        head_dim,
        scale,
        rope_dim,
        position,
        theta,
    );
    try out_dev.download(device, std.mem.sliceAsBytes(out[0..]));

    const expected_p0 = std.math.exp(0.5) / (std.math.exp(0.5) + std.math.exp(0.0));
    const expected_p1 = 1.0 - expected_p0;
    const expected_out_h0_d0 = expected_p0 * 2.0 + expected_p1 * 4.0;
    const expected_out_h0_d1 = expected_p0 * 3.0 + expected_p1 * 1.0;
    const expected_out_h1_d0 = expected_p1 * 2.0 + expected_p0 * 4.0;
    const expected_out_h1_d1 = expected_p1 * 3.0 + expected_p0 * 1.0;
    if (@abs(out[0] - expected_out_h0_d0) > 0.04) return error.CudaKernelSmokeMismatch;
    if (@abs(out[1] - expected_out_h0_d1) > 0.04) return error.CudaKernelSmokeMismatch;
    if (@abs(out[4] - expected_out_h1_d0) > 0.04) return error.CudaKernelSmokeMismatch;
    if (@abs(out[5] - expected_out_h1_d1) > 0.04) return error.CudaKernelSmokeMismatch;

    log.info("inference", "CUDA attention fused f16-kv smoke passed", .{
        .n_heads = n_heads,
        .seq = seq_len,
        .head_dim = head_dim,
        .out0 = out[0],
    });
}

fn runArgmaxSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const input = [_]f32{ -1.0, 4.5, 3.25, 4.5, 0.0 };
    const expected_index: u32 = 1;
    var actual_index: u32 = 0;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var out_index_dev = try device.allocBuffer(@sizeOf(u32));
    defer out_index_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try compute.cuda.argmax.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &out_index_dev,
        @intCast(input.len),
    );
    try out_index_dev.download(device, std.mem.asBytes(&actual_index));

    if (actual_index != expected_index) return error.CudaKernelSmokeMismatch;

    log.info("inference", "CUDA argmax smoke passed", .{
        .n = input.len,
        .source = @tagName(source),
        .idx = actual_index,
    });
}

fn runMatvecU16Smoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function_f16: compute.cuda.Function,
    source_f16: compute.cuda.registry.KernelSource,
    function_bf16: compute.cuda.Function,
    source_bf16: compute.cuda.registry.KernelSource,
) !void {
    const in_dim_usize: usize = 8;
    const out_dim_usize: usize = 3;
    const in_dim: u32 = in_dim_usize;
    const out_dim: u32 = out_dim_usize;

    var input: [in_dim_usize]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @floatFromInt(i + 1);

    var weights_bf16: [out_dim_usize * in_dim_usize]u16 = undefined;
    var weights_f16: [out_dim_usize * in_dim_usize]u16 = undefined;
    var expected: [out_dim_usize]f32 = [_]f32{0.0} ** out_dim_usize;
    for (0..out_dim_usize) |row| {
        var acc: f32 = 0.0;
        for (0..in_dim_usize) |col| {
            const w: f32 = @floatFromInt((row + 1) * (col + 1));
            const idx = row * in_dim_usize + col;
            weights_bf16[idx] = dtype.f32ToBf16(w);
            weights_f16[idx] = @bitCast(@as(f16, @floatCast(w)));
            acc += input[col] * w;
        }
        expected[row] = acc;
    }

    var actual_bf16 = [_]f32{0.0} ** out_dim_usize;
    var actual_f16 = [_]f32{0.0} ** out_dim_usize;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var weight_bf16_dev = try device.allocBuffer(weights_bf16.len * @sizeOf(u16));
    defer weight_bf16_dev.deinit(device);
    var weight_f16_dev = try device.allocBuffer(weights_f16.len * @sizeOf(u16));
    defer weight_f16_dev.deinit(device);
    var out_bf16_dev = try device.allocBuffer(actual_bf16.len * @sizeOf(f32));
    defer out_bf16_dev.deinit(device);
    var out_f16_dev = try device.allocBuffer(actual_f16.len * @sizeOf(f32));
    defer out_f16_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try weight_bf16_dev.upload(device, std.mem.sliceAsBytes(weights_bf16[0..]));
    try weight_f16_dev.upload(device, std.mem.sliceAsBytes(weights_f16[0..]));
    try compute.cuda.matvec_u16.runWithFunction(
        arg_pack,
        device,
        function_bf16,
        &input_dev,
        &weight_bf16_dev,
        &out_bf16_dev,
        in_dim,
        out_dim,
    );
    try compute.cuda.matvec_u16.runWithFunction(
        arg_pack,
        device,
        function_f16,
        &input_dev,
        &weight_f16_dev,
        &out_f16_dev,
        in_dim,
        out_dim,
    );
    try out_bf16_dev.download(device, std.mem.sliceAsBytes(actual_bf16[0..]));
    try out_f16_dev.download(device, std.mem.sliceAsBytes(actual_f16[0..]));

    for (expected, actual_bf16) |want, got| {
        if (@abs(want - got) > 0.01) return error.CudaKernelSmokeMismatch;
    }
    for (expected, actual_f16) |want, got| {
        if (@abs(want - got) > 0.01) return error.CudaKernelSmokeMismatch;
    }

    // Exercise non-vectorized fallback path with intentionally unaligned input/weight pointers.
    const input_bytes = input.len * @sizeOf(f32);
    const weight_bytes = weights_bf16.len * @sizeOf(u16);
    const input_pad: usize = 4;
    const weight_pad: usize = 2;

    var input_unaligned_raw_dev = try device.allocBuffer(input_bytes + input_pad);
    defer input_unaligned_raw_dev.deinit(device);
    var weight_unaligned_raw_dev = try device.allocBuffer(weight_bytes + weight_pad);
    defer weight_unaligned_raw_dev.deinit(device);
    var out_unaligned_dev = try device.allocBuffer(actual_bf16.len * @sizeOf(f32));
    defer out_unaligned_dev.deinit(device);
    var actual_unaligned = [_]f32{0.0} ** out_dim_usize;

    var input_blob: [input_pad + input_bytes]u8 = [_]u8{0} ** (input_pad + input_bytes);
    @memcpy(input_blob[input_pad..], std.mem.sliceAsBytes(input[0..]));
    var weight_blob: [weight_pad + weight_bytes]u8 = [_]u8{0} ** (weight_pad + weight_bytes);
    @memcpy(weight_blob[weight_pad..], std.mem.sliceAsBytes(weights_bf16[0..]));

    try input_unaligned_raw_dev.upload(device, input_blob[0..]);
    try weight_unaligned_raw_dev.upload(device, weight_blob[0..]);

    var input_unaligned_dev = try bufferSlice(&input_unaligned_raw_dev, input_pad, input_bytes);
    var weight_unaligned_dev = try bufferSlice(&weight_unaligned_raw_dev, weight_pad, weight_bytes);
    try compute.cuda.matvec_u16.runWithFunction(
        arg_pack,
        device,
        function_bf16,
        &input_unaligned_dev,
        &weight_unaligned_dev,
        &out_unaligned_dev,
        in_dim,
        out_dim,
    );
    try out_unaligned_dev.download(device, std.mem.sliceAsBytes(actual_unaligned[0..]));
    for (expected, actual_unaligned) |want, got| {
        if (@abs(want - got) > 0.01) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA matvec_u16 smoke passed", .{
        .in_dim = in_dim,
        .out_dim = out_dim,
        .source_f16 = @tagName(source_f16),
        .source_bf16 = @tagName(source_bf16),
        .out0_f16 = actual_f16[0],
        .out0_bf16 = actual_bf16[0],
    });
}

fn runMatmulU16Smoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function_f16: compute.cuda.Function,
    source_f16: compute.cuda.registry.KernelSource,
    function_bf16: compute.cuda.Function,
    source_bf16: compute.cuda.registry.KernelSource,
) !void {
    const rows_usize: usize = 2;
    const in_dim_usize: usize = 8;
    const out_dim_usize: usize = 3;
    const rows: u32 = rows_usize;
    const in_dim: u32 = in_dim_usize;
    const out_dim: u32 = out_dim_usize;

    var input: [rows_usize * in_dim_usize]f32 = undefined;
    for (0..rows_usize) |r| {
        for (0..in_dim_usize) |c| {
            input[r * in_dim_usize + c] = @floatFromInt((r + 1) * (c + 1));
        }
    }

    var weights_bf16: [out_dim_usize * in_dim_usize]u16 = undefined;
    var weights_f16: [out_dim_usize * in_dim_usize]u16 = undefined;
    var expected: [rows_usize * out_dim_usize]f32 = [_]f32{0.0} ** (rows_usize * out_dim_usize);
    for (0..out_dim_usize) |row| {
        for (0..in_dim_usize) |col| {
            const w: f32 = @floatFromInt((row + 1) * (col + 1));
            const idx = row * in_dim_usize + col;
            weights_bf16[idx] = dtype.f32ToBf16(w);
            weights_f16[idx] = @bitCast(@as(f16, @floatCast(w)));
        }
    }
    for (0..rows_usize) |r| {
        for (0..out_dim_usize) |o| {
            var acc: f32 = 0.0;
            for (0..in_dim_usize) |c| {
                const x = input[r * in_dim_usize + c];
                const w: f32 = @floatFromInt((o + 1) * (c + 1));
                acc += x * w;
            }
            expected[r * out_dim_usize + o] = acc;
        }
    }

    var actual_bf16 = [_]f32{0.0} ** (rows_usize * out_dim_usize);
    var actual_f16 = [_]f32{0.0} ** (rows_usize * out_dim_usize);

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var weight_bf16_dev = try device.allocBuffer(weights_bf16.len * @sizeOf(u16));
    defer weight_bf16_dev.deinit(device);
    var weight_f16_dev = try device.allocBuffer(weights_f16.len * @sizeOf(u16));
    defer weight_f16_dev.deinit(device);
    var out_bf16_dev = try device.allocBuffer(actual_bf16.len * @sizeOf(f32));
    defer out_bf16_dev.deinit(device);
    var out_f16_dev = try device.allocBuffer(actual_f16.len * @sizeOf(f32));
    defer out_f16_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try weight_bf16_dev.upload(device, std.mem.sliceAsBytes(weights_bf16[0..]));
    try weight_f16_dev.upload(device, std.mem.sliceAsBytes(weights_f16[0..]));

    try compute.cuda.matmul_u16.runWithFunction(
        arg_pack,
        device,
        function_bf16,
        &input_dev,
        &weight_bf16_dev,
        &out_bf16_dev,
        rows,
        in_dim,
        out_dim,
    );
    try compute.cuda.matmul_u16.runWithFunction(
        arg_pack,
        device,
        function_f16,
        &input_dev,
        &weight_f16_dev,
        &out_f16_dev,
        rows,
        in_dim,
        out_dim,
    );

    try out_bf16_dev.download(device, std.mem.sliceAsBytes(actual_bf16[0..]));
    try out_f16_dev.download(device, std.mem.sliceAsBytes(actual_f16[0..]));

    for (expected, actual_bf16) |want, got| {
        if (@abs(want - got) > 0.05) return error.CudaKernelSmokeMismatch;
    }
    for (expected, actual_f16) |want, got| {
        if (@abs(want - got) > 0.05) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA matmul_u16 smoke passed", .{
        .rows = rows,
        .in_dim = in_dim,
        .out_dim = out_dim,
        .source_f16 = @tagName(source_f16),
        .source_bf16 = @tagName(source_bf16),
        .out00_f16 = actual_f16[0],
        .out00_bf16 = actual_bf16[0],
    });
}

fn runMatvecU16GateUpSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function_f16: compute.cuda.Function,
    source_f16: compute.cuda.registry.KernelSource,
    function_bf16: compute.cuda.Function,
    source_bf16: compute.cuda.registry.KernelSource,
) !void {
    const in_dim_usize: usize = 8;
    const gate_out_dim_usize: usize = 3;
    const up_out_dim_usize: usize = 4;
    const in_dim: u32 = in_dim_usize;
    const gate_out_dim: u32 = gate_out_dim_usize;
    const up_out_dim: u32 = up_out_dim_usize;

    var input: [in_dim_usize]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @floatFromInt(i + 1);

    var gate_weights_bf16: [gate_out_dim_usize * in_dim_usize]u16 = undefined;
    var gate_weights_f16: [gate_out_dim_usize * in_dim_usize]u16 = undefined;
    var up_weights_bf16: [up_out_dim_usize * in_dim_usize]u16 = undefined;
    var up_weights_f16: [up_out_dim_usize * in_dim_usize]u16 = undefined;
    var expected_gate: [gate_out_dim_usize]f32 = [_]f32{0.0} ** gate_out_dim_usize;
    var expected_up: [up_out_dim_usize]f32 = [_]f32{0.0} ** up_out_dim_usize;

    for (0..gate_out_dim_usize) |row| {
        var acc: f32 = 0.0;
        for (0..in_dim_usize) |col| {
            const w: f32 = @floatFromInt((row + 1) * (col + 2));
            const idx = row * in_dim_usize + col;
            gate_weights_bf16[idx] = dtype.f32ToBf16(w);
            gate_weights_f16[idx] = @bitCast(@as(f16, @floatCast(w)));
            acc += input[col] * w;
        }
        expected_gate[row] = acc;
    }
    for (0..up_out_dim_usize) |row| {
        var acc: f32 = 0.0;
        for (0..in_dim_usize) |col| {
            const w: f32 = @floatFromInt((row + 2) * (col + 1));
            const idx = row * in_dim_usize + col;
            up_weights_bf16[idx] = dtype.f32ToBf16(w);
            up_weights_f16[idx] = @bitCast(@as(f16, @floatCast(w)));
            acc += input[col] * w;
        }
        expected_up[row] = acc;
    }

    var gate_actual_f16 = [_]f32{0.0} ** gate_out_dim_usize;
    var up_actual_f16 = [_]f32{0.0} ** up_out_dim_usize;
    var gate_actual_bf16 = [_]f32{0.0} ** gate_out_dim_usize;
    var up_actual_bf16 = [_]f32{0.0} ** up_out_dim_usize;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var gate_weight_bf16_dev = try device.allocBuffer(gate_weights_bf16.len * @sizeOf(u16));
    defer gate_weight_bf16_dev.deinit(device);
    var up_weight_bf16_dev = try device.allocBuffer(up_weights_bf16.len * @sizeOf(u16));
    defer up_weight_bf16_dev.deinit(device);
    var gate_weight_f16_dev = try device.allocBuffer(gate_weights_f16.len * @sizeOf(u16));
    defer gate_weight_f16_dev.deinit(device);
    var up_weight_f16_dev = try device.allocBuffer(up_weights_f16.len * @sizeOf(u16));
    defer up_weight_f16_dev.deinit(device);
    var gate_out_bf16_dev = try device.allocBuffer(gate_actual_bf16.len * @sizeOf(f32));
    defer gate_out_bf16_dev.deinit(device);
    var up_out_bf16_dev = try device.allocBuffer(up_actual_bf16.len * @sizeOf(f32));
    defer up_out_bf16_dev.deinit(device);
    var gate_out_f16_dev = try device.allocBuffer(gate_actual_f16.len * @sizeOf(f32));
    defer gate_out_f16_dev.deinit(device);
    var up_out_f16_dev = try device.allocBuffer(up_actual_f16.len * @sizeOf(f32));
    defer up_out_f16_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try gate_weight_bf16_dev.upload(device, std.mem.sliceAsBytes(gate_weights_bf16[0..]));
    try up_weight_bf16_dev.upload(device, std.mem.sliceAsBytes(up_weights_bf16[0..]));
    try gate_weight_f16_dev.upload(device, std.mem.sliceAsBytes(gate_weights_f16[0..]));
    try up_weight_f16_dev.upload(device, std.mem.sliceAsBytes(up_weights_f16[0..]));

    try compute.cuda.matvec_u16_gate_up.runWithFunction(
        arg_pack,
        device,
        function_bf16,
        &input_dev,
        &gate_weight_bf16_dev,
        &gate_out_bf16_dev,
        gate_out_dim,
        &up_weight_bf16_dev,
        &up_out_bf16_dev,
        up_out_dim,
        in_dim,
    );
    try compute.cuda.matvec_u16_gate_up.runWithFunction(
        arg_pack,
        device,
        function_f16,
        &input_dev,
        &gate_weight_f16_dev,
        &gate_out_f16_dev,
        gate_out_dim,
        &up_weight_f16_dev,
        &up_out_f16_dev,
        up_out_dim,
        in_dim,
    );
    try gate_out_bf16_dev.download(device, std.mem.sliceAsBytes(gate_actual_bf16[0..]));
    try up_out_bf16_dev.download(device, std.mem.sliceAsBytes(up_actual_bf16[0..]));
    try gate_out_f16_dev.download(device, std.mem.sliceAsBytes(gate_actual_f16[0..]));
    try up_out_f16_dev.download(device, std.mem.sliceAsBytes(up_actual_f16[0..]));

    for (expected_gate, gate_actual_bf16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_gate, gate_actual_f16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_up, up_actual_bf16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_up, up_actual_f16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }

    // Exercise non-vectorized fallback path with intentionally unaligned pointers.
    const input_bytes = input.len * @sizeOf(f32);
    const gate_weight_bytes = gate_weights_bf16.len * @sizeOf(u16);
    const up_weight_bytes = up_weights_bf16.len * @sizeOf(u16);
    const input_pad: usize = 4;
    const weight_pad: usize = 2;

    var input_unaligned_raw_dev = try device.allocBuffer(input_bytes + input_pad);
    defer input_unaligned_raw_dev.deinit(device);
    var gate_weight_unaligned_raw_dev = try device.allocBuffer(gate_weight_bytes + weight_pad);
    defer gate_weight_unaligned_raw_dev.deinit(device);
    var up_weight_unaligned_raw_dev = try device.allocBuffer(up_weight_bytes + weight_pad);
    defer up_weight_unaligned_raw_dev.deinit(device);
    var gate_out_unaligned_dev = try device.allocBuffer(gate_actual_bf16.len * @sizeOf(f32));
    defer gate_out_unaligned_dev.deinit(device);
    var up_out_unaligned_dev = try device.allocBuffer(up_actual_bf16.len * @sizeOf(f32));
    defer up_out_unaligned_dev.deinit(device);

    var gate_actual_unaligned = [_]f32{0.0} ** gate_out_dim_usize;
    var up_actual_unaligned = [_]f32{0.0} ** up_out_dim_usize;

    var input_blob: [input_pad + input_bytes]u8 = [_]u8{0} ** (input_pad + input_bytes);
    @memcpy(input_blob[input_pad..], std.mem.sliceAsBytes(input[0..]));
    var gate_weight_blob: [weight_pad + gate_weight_bytes]u8 = [_]u8{0} ** (weight_pad + gate_weight_bytes);
    @memcpy(gate_weight_blob[weight_pad..], std.mem.sliceAsBytes(gate_weights_bf16[0..]));
    var up_weight_blob: [weight_pad + up_weight_bytes]u8 = [_]u8{0} ** (weight_pad + up_weight_bytes);
    @memcpy(up_weight_blob[weight_pad..], std.mem.sliceAsBytes(up_weights_bf16[0..]));

    try input_unaligned_raw_dev.upload(device, input_blob[0..]);
    try gate_weight_unaligned_raw_dev.upload(device, gate_weight_blob[0..]);
    try up_weight_unaligned_raw_dev.upload(device, up_weight_blob[0..]);

    var input_unaligned_dev = try bufferSlice(&input_unaligned_raw_dev, input_pad, input_bytes);
    var gate_weight_unaligned_dev = try bufferSlice(&gate_weight_unaligned_raw_dev, weight_pad, gate_weight_bytes);
    var up_weight_unaligned_dev = try bufferSlice(&up_weight_unaligned_raw_dev, weight_pad, up_weight_bytes);
    try compute.cuda.matvec_u16_gate_up.runWithFunction(
        arg_pack,
        device,
        function_bf16,
        &input_unaligned_dev,
        &gate_weight_unaligned_dev,
        &gate_out_unaligned_dev,
        gate_out_dim,
        &up_weight_unaligned_dev,
        &up_out_unaligned_dev,
        up_out_dim,
        in_dim,
    );
    try gate_out_unaligned_dev.download(device, std.mem.sliceAsBytes(gate_actual_unaligned[0..]));
    try up_out_unaligned_dev.download(device, std.mem.sliceAsBytes(up_actual_unaligned[0..]));

    for (expected_gate, gate_actual_unaligned) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_up, up_actual_unaligned) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA matvec_u16_gate_up smoke passed", .{
        .source_f16 = @tagName(source_f16),
        .source_bf16 = @tagName(source_bf16),
        .gate0 = gate_actual_f16[0],
        .up0 = up_actual_f16[0],
    });
}

fn runMatvecU16QkvSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function_f16: compute.cuda.Function,
    source_f16: compute.cuda.registry.KernelSource,
    function_bf16: compute.cuda.Function,
    source_bf16: compute.cuda.registry.KernelSource,
) !void {
    const in_dim_usize: usize = 8;
    const q_out_dim_usize: usize = 3;
    const k_out_dim_usize: usize = 2;
    const v_out_dim_usize: usize = 4;
    const in_dim: u32 = in_dim_usize;
    const q_out_dim: u32 = q_out_dim_usize;
    const k_out_dim: u32 = k_out_dim_usize;
    const v_out_dim: u32 = v_out_dim_usize;

    var input: [in_dim_usize]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @floatFromInt(i + 1);

    var q_weights_bf16: [q_out_dim_usize * in_dim_usize]u16 = undefined;
    var q_weights_f16: [q_out_dim_usize * in_dim_usize]u16 = undefined;
    var k_weights_bf16: [k_out_dim_usize * in_dim_usize]u16 = undefined;
    var k_weights_f16: [k_out_dim_usize * in_dim_usize]u16 = undefined;
    var v_weights_bf16: [v_out_dim_usize * in_dim_usize]u16 = undefined;
    var v_weights_f16: [v_out_dim_usize * in_dim_usize]u16 = undefined;

    var expected_q: [q_out_dim_usize]f32 = [_]f32{0.0} ** q_out_dim_usize;
    var expected_k: [k_out_dim_usize]f32 = [_]f32{0.0} ** k_out_dim_usize;
    var expected_v: [v_out_dim_usize]f32 = [_]f32{0.0} ** v_out_dim_usize;

    for (0..q_out_dim_usize) |row| {
        var acc: f32 = 0.0;
        for (0..in_dim_usize) |col| {
            const w: f32 = @floatFromInt((row + 1) * (col + 1));
            const idx = row * in_dim_usize + col;
            q_weights_bf16[idx] = dtype.f32ToBf16(w);
            q_weights_f16[idx] = @bitCast(@as(f16, @floatCast(w)));
            acc += input[col] * w;
        }
        expected_q[row] = acc;
    }
    for (0..k_out_dim_usize) |row| {
        var acc: f32 = 0.0;
        for (0..in_dim_usize) |col| {
            const w: f32 = @floatFromInt((row + 2) * (col + 1));
            const idx = row * in_dim_usize + col;
            k_weights_bf16[idx] = dtype.f32ToBf16(w);
            k_weights_f16[idx] = @bitCast(@as(f16, @floatCast(w)));
            acc += input[col] * w;
        }
        expected_k[row] = acc;
    }
    for (0..v_out_dim_usize) |row| {
        var acc: f32 = 0.0;
        for (0..in_dim_usize) |col| {
            const w: f32 = @floatFromInt((row + 3) * (col + 1));
            const idx = row * in_dim_usize + col;
            v_weights_bf16[idx] = dtype.f32ToBf16(w);
            v_weights_f16[idx] = @bitCast(@as(f16, @floatCast(w)));
            acc += input[col] * w;
        }
        expected_v[row] = acc;
    }

    var q_actual_f16 = [_]f32{0.0} ** q_out_dim_usize;
    var k_actual_f16 = [_]f32{0.0} ** k_out_dim_usize;
    var v_actual_f16 = [_]f32{0.0} ** v_out_dim_usize;
    var q_actual_bf16 = [_]f32{0.0} ** q_out_dim_usize;
    var k_actual_bf16 = [_]f32{0.0} ** k_out_dim_usize;
    var v_actual_bf16 = [_]f32{0.0} ** v_out_dim_usize;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var q_weight_bf16_dev = try device.allocBuffer(q_weights_bf16.len * @sizeOf(u16));
    defer q_weight_bf16_dev.deinit(device);
    var k_weight_bf16_dev = try device.allocBuffer(k_weights_bf16.len * @sizeOf(u16));
    defer k_weight_bf16_dev.deinit(device);
    var v_weight_bf16_dev = try device.allocBuffer(v_weights_bf16.len * @sizeOf(u16));
    defer v_weight_bf16_dev.deinit(device);
    var q_weight_f16_dev = try device.allocBuffer(q_weights_f16.len * @sizeOf(u16));
    defer q_weight_f16_dev.deinit(device);
    var k_weight_f16_dev = try device.allocBuffer(k_weights_f16.len * @sizeOf(u16));
    defer k_weight_f16_dev.deinit(device);
    var v_weight_f16_dev = try device.allocBuffer(v_weights_f16.len * @sizeOf(u16));
    defer v_weight_f16_dev.deinit(device);
    var q_out_bf16_dev = try device.allocBuffer(q_actual_bf16.len * @sizeOf(f32));
    defer q_out_bf16_dev.deinit(device);
    var k_out_bf16_dev = try device.allocBuffer(k_actual_bf16.len * @sizeOf(f32));
    defer k_out_bf16_dev.deinit(device);
    var v_out_bf16_dev = try device.allocBuffer(v_actual_bf16.len * @sizeOf(f32));
    defer v_out_bf16_dev.deinit(device);
    var q_out_f16_dev = try device.allocBuffer(q_actual_f16.len * @sizeOf(f32));
    defer q_out_f16_dev.deinit(device);
    var k_out_f16_dev = try device.allocBuffer(k_actual_f16.len * @sizeOf(f32));
    defer k_out_f16_dev.deinit(device);
    var v_out_f16_dev = try device.allocBuffer(v_actual_f16.len * @sizeOf(f32));
    defer v_out_f16_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try q_weight_bf16_dev.upload(device, std.mem.sliceAsBytes(q_weights_bf16[0..]));
    try k_weight_bf16_dev.upload(device, std.mem.sliceAsBytes(k_weights_bf16[0..]));
    try v_weight_bf16_dev.upload(device, std.mem.sliceAsBytes(v_weights_bf16[0..]));
    try q_weight_f16_dev.upload(device, std.mem.sliceAsBytes(q_weights_f16[0..]));
    try k_weight_f16_dev.upload(device, std.mem.sliceAsBytes(k_weights_f16[0..]));
    try v_weight_f16_dev.upload(device, std.mem.sliceAsBytes(v_weights_f16[0..]));

    try compute.cuda.matvec_u16_qkv.runWithFunction(
        arg_pack,
        device,
        function_bf16,
        &input_dev,
        &q_weight_bf16_dev,
        &q_out_bf16_dev,
        q_out_dim,
        &k_weight_bf16_dev,
        &k_out_bf16_dev,
        k_out_dim,
        &v_weight_bf16_dev,
        &v_out_bf16_dev,
        v_out_dim,
        in_dim,
    );
    try compute.cuda.matvec_u16_qkv.runWithFunction(
        arg_pack,
        device,
        function_f16,
        &input_dev,
        &q_weight_f16_dev,
        &q_out_f16_dev,
        q_out_dim,
        &k_weight_f16_dev,
        &k_out_f16_dev,
        k_out_dim,
        &v_weight_f16_dev,
        &v_out_f16_dev,
        v_out_dim,
        in_dim,
    );
    try q_out_bf16_dev.download(device, std.mem.sliceAsBytes(q_actual_bf16[0..]));
    try k_out_bf16_dev.download(device, std.mem.sliceAsBytes(k_actual_bf16[0..]));
    try v_out_bf16_dev.download(device, std.mem.sliceAsBytes(v_actual_bf16[0..]));
    try q_out_f16_dev.download(device, std.mem.sliceAsBytes(q_actual_f16[0..]));
    try k_out_f16_dev.download(device, std.mem.sliceAsBytes(k_actual_f16[0..]));
    try v_out_f16_dev.download(device, std.mem.sliceAsBytes(v_actual_f16[0..]));

    for (expected_q, q_actual_bf16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_q, q_actual_f16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_k, k_actual_bf16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_k, k_actual_f16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_v, v_actual_bf16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_v, v_actual_f16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }

    // Exercise non-vectorized fallback path with intentionally unaligned pointers.
    const input_bytes = input.len * @sizeOf(f32);
    const q_weight_bytes = q_weights_bf16.len * @sizeOf(u16);
    const k_weight_bytes = k_weights_bf16.len * @sizeOf(u16);
    const v_weight_bytes = v_weights_bf16.len * @sizeOf(u16);
    const input_pad: usize = 4;
    const weight_pad: usize = 2;

    var input_unaligned_raw_dev = try device.allocBuffer(input_bytes + input_pad);
    defer input_unaligned_raw_dev.deinit(device);
    var q_weight_unaligned_raw_dev = try device.allocBuffer(q_weight_bytes + weight_pad);
    defer q_weight_unaligned_raw_dev.deinit(device);
    var k_weight_unaligned_raw_dev = try device.allocBuffer(k_weight_bytes + weight_pad);
    defer k_weight_unaligned_raw_dev.deinit(device);
    var v_weight_unaligned_raw_dev = try device.allocBuffer(v_weight_bytes + weight_pad);
    defer v_weight_unaligned_raw_dev.deinit(device);
    var q_out_unaligned_dev = try device.allocBuffer(q_actual_bf16.len * @sizeOf(f32));
    defer q_out_unaligned_dev.deinit(device);
    var k_out_unaligned_dev = try device.allocBuffer(k_actual_bf16.len * @sizeOf(f32));
    defer k_out_unaligned_dev.deinit(device);
    var v_out_unaligned_dev = try device.allocBuffer(v_actual_bf16.len * @sizeOf(f32));
    defer v_out_unaligned_dev.deinit(device);

    var q_actual_unaligned = [_]f32{0.0} ** q_out_dim_usize;
    var k_actual_unaligned = [_]f32{0.0} ** k_out_dim_usize;
    var v_actual_unaligned = [_]f32{0.0} ** v_out_dim_usize;

    var input_blob: [input_pad + input_bytes]u8 = [_]u8{0} ** (input_pad + input_bytes);
    @memcpy(input_blob[input_pad..], std.mem.sliceAsBytes(input[0..]));
    var q_weight_blob: [weight_pad + q_weight_bytes]u8 = [_]u8{0} ** (weight_pad + q_weight_bytes);
    @memcpy(q_weight_blob[weight_pad..], std.mem.sliceAsBytes(q_weights_bf16[0..]));
    var k_weight_blob: [weight_pad + k_weight_bytes]u8 = [_]u8{0} ** (weight_pad + k_weight_bytes);
    @memcpy(k_weight_blob[weight_pad..], std.mem.sliceAsBytes(k_weights_bf16[0..]));
    var v_weight_blob: [weight_pad + v_weight_bytes]u8 = [_]u8{0} ** (weight_pad + v_weight_bytes);
    @memcpy(v_weight_blob[weight_pad..], std.mem.sliceAsBytes(v_weights_bf16[0..]));

    try input_unaligned_raw_dev.upload(device, input_blob[0..]);
    try q_weight_unaligned_raw_dev.upload(device, q_weight_blob[0..]);
    try k_weight_unaligned_raw_dev.upload(device, k_weight_blob[0..]);
    try v_weight_unaligned_raw_dev.upload(device, v_weight_blob[0..]);

    var input_unaligned_dev = try bufferSlice(&input_unaligned_raw_dev, input_pad, input_bytes);
    var q_weight_unaligned_dev = try bufferSlice(&q_weight_unaligned_raw_dev, weight_pad, q_weight_bytes);
    var k_weight_unaligned_dev = try bufferSlice(&k_weight_unaligned_raw_dev, weight_pad, k_weight_bytes);
    var v_weight_unaligned_dev = try bufferSlice(&v_weight_unaligned_raw_dev, weight_pad, v_weight_bytes);
    try compute.cuda.matvec_u16_qkv.runWithFunction(
        arg_pack,
        device,
        function_bf16,
        &input_unaligned_dev,
        &q_weight_unaligned_dev,
        &q_out_unaligned_dev,
        q_out_dim,
        &k_weight_unaligned_dev,
        &k_out_unaligned_dev,
        k_out_dim,
        &v_weight_unaligned_dev,
        &v_out_unaligned_dev,
        v_out_dim,
        in_dim,
    );
    try q_out_unaligned_dev.download(device, std.mem.sliceAsBytes(q_actual_unaligned[0..]));
    try k_out_unaligned_dev.download(device, std.mem.sliceAsBytes(k_actual_unaligned[0..]));
    try v_out_unaligned_dev.download(device, std.mem.sliceAsBytes(v_actual_unaligned[0..]));

    for (expected_q, q_actual_unaligned) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_k, k_actual_unaligned) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_v, v_actual_unaligned) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA matvec_u16_qkv smoke passed", .{
        .source_f16 = @tagName(source_f16),
        .source_bf16 = @tagName(source_bf16),
        .q0 = q_actual_f16[0],
        .k0 = k_actual_f16[0],
        .v0 = v_actual_f16[0],
    });
}

fn runGaffineU4MatvecSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const in_dim: u32 = 8;
    const out_dim: u32 = 2;
    const group_size: u32 = 8;
    const input = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    const packed_words = [_]u32{
        0x7654_3210, // row 0 => 0..7
        0x0123_4567, // row 1 => 7..0
    };
    const scales = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(2.0),
    };
    const biases = [_]u16{
        dtype.f32ToBf16(0.0),
        dtype.f32ToBf16(1.0),
    };
    const expected = [_]f32{
        28.0,
        64.0,
    };
    var actual = [_]f32{0.0} ** out_dim;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var packed_dev = try device.allocBuffer(packed_words.len * @sizeOf(u32));
    defer packed_dev.deinit(device);
    var scales_dev = try device.allocBuffer(scales.len * @sizeOf(u16));
    defer scales_dev.deinit(device);
    var biases_dev = try device.allocBuffer(biases.len * @sizeOf(u16));
    defer biases_dev.deinit(device);
    var out_dev = try device.allocBuffer(actual.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try packed_dev.upload(device, std.mem.sliceAsBytes(packed_words[0..]));
    try scales_dev.upload(device, std.mem.sliceAsBytes(scales[0..]));
    try biases_dev.upload(device, std.mem.sliceAsBytes(biases[0..]));

    try compute.cuda.gaffine_u4_matvec.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &packed_dev,
        &scales_dev,
        &biases_dev,
        &out_dev,
        in_dim,
        out_dim,
        group_size,
        gaffine_scales_dtype_bf16,
        1,
    );
    try out_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        if (@abs(want - got) > 0.01) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA gaffine_u4_matvec smoke passed", .{
        .in_dim = in_dim,
        .out_dim = out_dim,
        .source = @tagName(source),
        .out0 = actual[0],
    });
}

fn runGaffineU4MatvecGateUpSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const in_dim: u32 = 8;
    const out_dim: u32 = 2;
    const group_size: u32 = 8;
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const packed_words = [_]u32{
        0x7654_3210,
        0x0123_4567,
    };
    const gate_scales = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(1.0),
    };
    const gate_biases = [_]u16{
        dtype.f32ToBf16(0.0),
        dtype.f32ToBf16(0.0),
    };
    const up_scales = [_]u16{
        dtype.f32ToBf16(2.0),
        dtype.f32ToBf16(2.0),
    };
    const up_biases = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(1.0),
    };
    const expected_gate = [_]f32{ 168.0, 84.0 };
    const expected_up = [_]f32{ 372.0, 204.0 };
    var out_gate = [_]f32{0.0} ** out_dim;
    var out_up = [_]f32{0.0} ** out_dim;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var gate_packed_dev = try device.allocBuffer(packed_words.len * @sizeOf(u32));
    defer gate_packed_dev.deinit(device);
    var gate_scales_dev = try device.allocBuffer(gate_scales.len * @sizeOf(u16));
    defer gate_scales_dev.deinit(device);
    var gate_biases_dev = try device.allocBuffer(gate_biases.len * @sizeOf(u16));
    defer gate_biases_dev.deinit(device);
    var out_gate_dev = try device.allocBuffer(out_gate.len * @sizeOf(f32));
    defer out_gate_dev.deinit(device);

    var up_packed_dev = try device.allocBuffer(packed_words.len * @sizeOf(u32));
    defer up_packed_dev.deinit(device);
    var up_scales_dev = try device.allocBuffer(up_scales.len * @sizeOf(u16));
    defer up_scales_dev.deinit(device);
    var up_biases_dev = try device.allocBuffer(up_biases.len * @sizeOf(u16));
    defer up_biases_dev.deinit(device);
    var out_up_dev = try device.allocBuffer(out_up.len * @sizeOf(f32));
    defer out_up_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try gate_packed_dev.upload(device, std.mem.sliceAsBytes(packed_words[0..]));
    try gate_scales_dev.upload(device, std.mem.sliceAsBytes(gate_scales[0..]));
    try gate_biases_dev.upload(device, std.mem.sliceAsBytes(gate_biases[0..]));
    try up_packed_dev.upload(device, std.mem.sliceAsBytes(packed_words[0..]));
    try up_scales_dev.upload(device, std.mem.sliceAsBytes(up_scales[0..]));
    try up_biases_dev.upload(device, std.mem.sliceAsBytes(up_biases[0..]));

    try compute.cuda.gaffine_u4_matvec_gate_up.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &gate_packed_dev,
        &gate_scales_dev,
        &gate_biases_dev,
        &out_gate_dev,
        out_dim,
        group_size,
        gaffine_scales_dtype_bf16,
        &up_packed_dev,
        &up_scales_dev,
        &up_biases_dev,
        &out_up_dev,
        out_dim,
        group_size,
        gaffine_scales_dtype_bf16,
        in_dim,
        1,
    );
    try out_gate_dev.download(device, std.mem.sliceAsBytes(out_gate[0..]));
    try out_up_dev.download(device, std.mem.sliceAsBytes(out_up[0..]));

    for (expected_gate, out_gate) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_up, out_up) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA gaffine_u4_matvec_gate_up smoke passed", .{
        .in_dim = in_dim,
        .out_dim = out_dim,
        .source = @tagName(source),
        .gate0 = out_gate[0],
        .up0 = out_up[0],
    });
}

fn runGaffineU4MatvecQkvSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const in_dim: u32 = 8;
    const out_dim: u32 = 2;
    const group_size: u32 = 8;
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const packed_words = [_]u32{
        0x7654_3210, // row 0 => 0..7
        0x0123_4567, // row 1 => 7..0
    };
    const q_scales = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(1.0),
    };
    const q_biases = [_]u16{
        dtype.f32ToBf16(0.0),
        dtype.f32ToBf16(0.0),
    };
    const k_scales = [_]u16{
        dtype.f32ToBf16(2.0),
        dtype.f32ToBf16(2.0),
    };
    const k_biases = [_]u16{
        dtype.f32ToBf16(0.0),
        dtype.f32ToBf16(0.0),
    };
    const v_scales = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(1.0),
    };
    const v_biases = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(1.0),
    };
    const expected_q = [_]f32{ 168.0, 84.0 };
    const expected_k = [_]f32{ 336.0, 168.0 };
    const expected_v = [_]f32{ 204.0, 120.0 };
    var out_q = [_]f32{0.0} ** out_dim;
    var out_k = [_]f32{0.0} ** out_dim;
    var out_v = [_]f32{0.0} ** out_dim;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var q_packed_dev = try device.allocBuffer(packed_words.len * @sizeOf(u32));
    defer q_packed_dev.deinit(device);
    var q_scales_dev = try device.allocBuffer(q_scales.len * @sizeOf(u16));
    defer q_scales_dev.deinit(device);
    var q_biases_dev = try device.allocBuffer(q_biases.len * @sizeOf(u16));
    defer q_biases_dev.deinit(device);
    var q_out_dev = try device.allocBuffer(out_q.len * @sizeOf(f32));
    defer q_out_dev.deinit(device);

    var k_packed_dev = try device.allocBuffer(packed_words.len * @sizeOf(u32));
    defer k_packed_dev.deinit(device);
    var k_scales_dev = try device.allocBuffer(k_scales.len * @sizeOf(u16));
    defer k_scales_dev.deinit(device);
    var k_biases_dev = try device.allocBuffer(k_biases.len * @sizeOf(u16));
    defer k_biases_dev.deinit(device);
    var k_out_dev = try device.allocBuffer(out_k.len * @sizeOf(f32));
    defer k_out_dev.deinit(device);

    var v_packed_dev = try device.allocBuffer(packed_words.len * @sizeOf(u32));
    defer v_packed_dev.deinit(device);
    var v_scales_dev = try device.allocBuffer(v_scales.len * @sizeOf(u16));
    defer v_scales_dev.deinit(device);
    var v_biases_dev = try device.allocBuffer(v_biases.len * @sizeOf(u16));
    defer v_biases_dev.deinit(device);
    var v_out_dev = try device.allocBuffer(out_v.len * @sizeOf(f32));
    defer v_out_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try q_packed_dev.upload(device, std.mem.sliceAsBytes(packed_words[0..]));
    try q_scales_dev.upload(device, std.mem.sliceAsBytes(q_scales[0..]));
    try q_biases_dev.upload(device, std.mem.sliceAsBytes(q_biases[0..]));
    try k_packed_dev.upload(device, std.mem.sliceAsBytes(packed_words[0..]));
    try k_scales_dev.upload(device, std.mem.sliceAsBytes(k_scales[0..]));
    try k_biases_dev.upload(device, std.mem.sliceAsBytes(k_biases[0..]));
    try v_packed_dev.upload(device, std.mem.sliceAsBytes(packed_words[0..]));
    try v_scales_dev.upload(device, std.mem.sliceAsBytes(v_scales[0..]));
    try v_biases_dev.upload(device, std.mem.sliceAsBytes(v_biases[0..]));

    try compute.cuda.gaffine_u4_matvec_qkv.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &q_packed_dev,
        &q_scales_dev,
        &q_biases_dev,
        &q_out_dev,
        out_dim,
        group_size,
        gaffine_scales_dtype_bf16,
        &k_packed_dev,
        &k_scales_dev,
        &k_biases_dev,
        &k_out_dev,
        out_dim,
        group_size,
        gaffine_scales_dtype_bf16,
        &v_packed_dev,
        &v_scales_dev,
        &v_biases_dev,
        &v_out_dev,
        out_dim,
        group_size,
        gaffine_scales_dtype_bf16,
        in_dim,
        1,
    );
    try q_out_dev.download(device, std.mem.sliceAsBytes(out_q[0..]));
    try k_out_dev.download(device, std.mem.sliceAsBytes(out_k[0..]));
    try v_out_dev.download(device, std.mem.sliceAsBytes(out_v[0..]));

    for (expected_q, out_q) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_k, out_k) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_v, out_v) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA gaffine_u4_matvec_qkv smoke passed", .{
        .in_dim = in_dim,
        .out_dim = out_dim,
        .source = @tagName(source),
        .q0 = out_q[0],
        .k0 = out_k[0],
        .v0 = out_v[0],
    });
}

fn computeRmsNormReference(
    out: []f32,
    input: []const f32,
    weight: []const f32,
    rows: u32,
    cols: u32,
    eps: f32,
) void {
    const rows_usize: usize = @intCast(rows);
    const cols_usize: usize = @intCast(cols);
    var row: usize = 0;
    while (row < rows_usize) : (row += 1) {
        const base = row * cols_usize;
        var sum_sq: f32 = 0.0;
        var col: usize = 0;
        while (col < cols_usize) : (col += 1) {
            const v = input[base + col];
            sum_sq += v * v;
        }
        const mean_sq = sum_sq / @as(f32, @floatFromInt(cols_usize));
        const inv_rms = 1.0 / std.math.sqrt(mean_sq + eps);
        col = 0;
        while (col < cols_usize) : (col += 1) {
            out[base + col] = input[base + col] * inv_rms * weight[col];
        }
    }
}

fn bufferSlice(buffer: *const compute.cuda.Buffer, byte_offset: usize, byte_len: usize) !compute.cuda.Buffer {
    if (byte_offset > buffer.size) return error.InvalidArgument;
    const end = std.math.add(usize, byte_offset, byte_len) catch return error.InvalidArgument;
    if (end > buffer.size) return error.InvalidArgument;
    const ptr = std.math.add(u64, buffer.pointer, @intCast(byte_offset)) catch return error.InvalidArgument;
    return .{
        .pointer = ptr,
        .size = byte_len,
    };
}
