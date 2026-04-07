//! Engine integration tests.
//!
//! Extracted from engine.zig to keep all split files under the line budget.
//! Tests exercise types and functions across engine_types, engine_weights,
//! engine_ops, engine_mixers, and engine_layer_program.

const std = @import("std");
const compute = @import("../../../compute/root.zig");
const tensor = @import("../../../tensor.zig");
const dtype = @import("../../../dtype.zig");
const models = @import("../../../models/root.zig");
const layer_ops = models.layer_ops;
const opcode_map = models.plan.opcode_map;
const plan_compiler = models.plan.compiler;
const runtime_contract = @import("../../runtime_contract/root.zig");
const LoadedModel = models.LoadedModel;
const Tensor = tensor.Tensor;

// --- Types from engine.zig ---
const engine = @import("engine.zig");
const CudaBackend = engine.CudaBackend;

// --- Types from engine_types.zig ---
const engine_types = @import("engine_types.zig");
const DeviceTensor = engine_types.DeviceTensor;
const LinearWeight = engine_types.LinearWeight;
const U16LinearWeight = engine_types.U16LinearWeight;
const GaffineU4LinearWeight = engine_types.GaffineU4LinearWeight;
const GaffineU8LinearWeight = engine_types.GaffineU8LinearWeight;
const Nvfp4LinearWeight = engine_types.Nvfp4LinearWeight;
const LayerAttentionRuntime = engine_types.LayerAttentionRuntime;
const LayerAttentionExecConfig = engine_types.LayerAttentionExecConfig;
const ShortConvBlockRuntime = engine_types.ShortConvBlockRuntime;
const BlockRuntimeLayer = engine_types.BlockRuntimeLayer;
const KvRuntimeState = engine_types.KvRuntimeState;
const ShortConvRuntimeState = engine_types.ShortConvRuntimeState;
const MambaRuntimeState = engine_types.MambaRuntimeState;
const GatedDeltaRuntimeState = engine_types.GatedDeltaRuntimeState;
const gaffine_scales_dtype_bf16 = engine_types.gaffine_scales_dtype_bf16;
const expectedAttentionQProjectionDim = engine_types.expectedAttentionQProjectionDim;
const bufferF32RowCount = engine_types.bufferF32RowCount;
const logicalF32RowSlice = engine_types.logicalF32RowSlice;
const buildCudaLayerProgramRegisterSlotMap = engine_types.buildCudaLayerProgramRegisterSlotMap;
const required_kernels = engine_types.required_kernels;
const KernelSlot = engine_types.KernelSlot;

// --- Functions from engine_weights.zig ---
const engine_weights = @import("engine_weights.zig");
const resolveDenseInOutLayout = engine_weights.resolveDenseInOutLayout;
const resolveDenseOutInLayout = engine_weights.resolveDenseOutInLayout;
const transposeRowMajor = engine_weights.transposeRowMajor;
const tryPopulateProjectionFromWeight = engine_weights.tryPopulateProjectionFromWeight;
const tryPopulateHiddenFromToken = engine_weights.tryPopulateHiddenFromToken;
const populatePrefillHiddenFromTokens = engine_weights.populatePrefillHiddenFromTokens;
const gaffineValueAt = engine_weights.gaffineValueAt;
const tryPopulateFinalNormWeight = engine_weights.tryPopulateFinalNormWeight;
const shouldDownloadPrefillLogits = engine_weights.shouldDownloadPrefillLogits;
const materializeDenseOutInU16 = engine_weights.materializeDenseOutInU16;
const materializeDenseOutInF32 = engine_weights.materializeDenseOutInF32;
const collectTokenPositions = engine_weights.collectTokenPositions;
const findPositionIndex = engine_weights.findPositionIndex;
const deepstackLayersCompatibleWithPrompt = engine_weights.deepstackLayersCompatibleWithPrompt;

// --- Functions from engine_ops.zig ---
const engine_ops = @import("engine_ops.zig");

// --- Functions from engine_mixers.zig ---
const engine_mixers = @import("engine_mixers.zig");
const engine_forward = @import("engine_forward.zig");

test "resolveDenseInOutLayout keeps [in,out] orientation" {
    const layout = try resolveDenseInOutLayout(128, 256, 128);
    try std.testing.expectEqual(@as(usize, 128), layout.in_dim);
    try std.testing.expectEqual(@as(usize, 256), layout.out_dim);
    try std.testing.expect(!layout.needs_transpose);
}

test "finalOutputBuffer returns residual when program ends with add" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .branch_out,
            .debug_type = .multihead_attention,
            .state_block_id = runtime_contract.kv_cache_state_id,
        } },
        .{ .add = .{
            .branch = .branch_out,
            .scale = .one,
        } },
    };
    try std.testing.expectEqual(layer_ops.BufferId.residual, layer_ops.finalOutputBuffer(&program));
}

test "finalOutputBuffer returns kernel output buffer for post-norm endings" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .norm_out,
            .debug_type = .norm,
        } },
    };
    try std.testing.expectEqual(layer_ops.BufferId.norm_out, layer_ops.finalOutputBuffer(&program));
}

test "layer program support envelope accepts kernel-add programs" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .norm_out,
            .debug_type = .norm,
        } },
        .{ .kernel = .{
            .id = 1,
            .in = .norm_out,
            .out = .branch_out,
            .debug_type = .mlp,
        } },
        .{ .add = .{
            .branch = .branch_out,
            .scale = .one,
        } },
    };
    try std.testing.expect(
        runtime_contract.firstLayerProgramCompatibilityIssue(
            &program,
            .attention_mlp,
            CudaBackend.layer_program_adapter_table,
        ) == null,
    );
}

test "layer_program_adapter_table covers CUDA LayerOp execution subset" {
    const supported = [_]opcode_map.Opcode{
        .rmsnorm,
        .multihead_attention,
        .gated_delta_net,
        .swiglu,
        .shortconv,
        .residual_add,
    };
    for (supported) |opcode| {
        try std.testing.expect(CudaBackend.layer_program_adapter_table[@intFromEnum(opcode)] != null);
    }

    try std.testing.expect(CudaBackend.layer_program_adapter_table[@intFromEnum(opcode_map.Opcode.mla_attention)] == null);
    try std.testing.expect(CudaBackend.layer_program_adapter_table[@intFromEnum(opcode_map.Opcode.moe)] == null);
    try std.testing.expect(CudaBackend.layer_program_adapter_table[@intFromEnum(opcode_map.Opcode.mamba_mixer)] == null);
    try std.testing.expect(CudaBackend.layer_program_adapter_table[@intFromEnum(opcode_map.Opcode.mul_scalar)] == null);
    try std.testing.expect(CudaBackend.layer_program_adapter_table[@intFromEnum(opcode_map.Opcode.vision_patch_embed)] == null);
}

test "layer program support envelope rejects unsupported primitive ops" {
    const program = [_]layer_ops.LayerOp{
        .{ .mul_scalar = .{
            .in = .residual,
            .out = .residual,
            .scalar = 0.5,
        } },
    };
    const issue = runtime_contract.firstLayerProgramCompatibilityIssue(
        &program,
        .attention_mlp,
        CudaBackend.layer_program_adapter_table,
    ) orelse return error.TestUnexpectedResult;
    switch (issue) {
        .unsupported_opcode => |unsupported| try std.testing.expectEqual(opcode_map.Opcode.mul_scalar, unsupported.opcode),
        else => return error.TestUnexpectedResult,
    }
}

test "layer program support envelope rejects CUDA-unsupported macro opcodes at load time" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .branch_out,
            .debug_type = .mamba_mixer,
            .state_block_id = runtime_contract.mamba_state_id,
        } },
    };
    const issue = runtime_contract.firstLayerProgramCompatibilityIssue(
        &program,
        .attention_mlp,
        CudaBackend.layer_program_adapter_table,
    ) orelse return error.TestUnexpectedResult;
    switch (issue) {
        .unsupported_opcode => |unsupported| try std.testing.expectEqual(opcode_map.Opcode.mamba_mixer, unsupported.opcode),
        else => return error.TestUnexpectedResult,
    }
}

test "layer program support envelope rejects block-kind state descriptor mismatch" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .branch_out,
            .debug_type = .shortconv,
            .state_block_id = runtime_contract.shortconv_state_id,
        } },
        .{ .add = .{
            .branch = .branch_out,
            .scale = .one,
        } },
    };
    const issue = runtime_contract.firstLayerProgramCompatibilityIssue(
        &program,
        .attention_mlp,
        CudaBackend.layer_program_adapter_table,
    ) orelse return error.TestUnexpectedResult;
    switch (issue) {
        .state_mismatch => |mismatch| {
            try std.testing.expectEqual(@as(usize, 0), mismatch.op_index);
            try std.testing.expectEqual(opcode_map.Opcode.shortconv, mismatch.opcode);
            try std.testing.expectEqual(runtime_contract.shortconv_state_id, mismatch.state_id);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "buildCudaLayerProgramRegisterSlotMap reuses temp slots from liveness" {
    const inputs0 = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(0)};
    const outputs0 = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(1)};
    const inputs1 = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(1)};
    const outputs1 = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(2)};
    const inputs2 = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(2)};
    const outputs2 = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(3)};
    const instructions = [_]runtime_contract.Instruction{
        .{
            .opcode = .rmsnorm,
            .inputs = inputs0[0..],
            .outputs = outputs0[0..],
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = null,
        },
        .{
            .opcode = .swiglu,
            .inputs = inputs1[0..],
            .outputs = outputs1[0..],
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = null,
        },
        .{
            .opcode = .swiglu,
            .inputs = inputs2[0..],
            .outputs = outputs2[0..],
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = null,
        },
    };
    const kill0 = [_]u64{0b0000};
    const kill1 = [_]u64{0b0010};
    const kill2 = [_]u64{0b1100};
    const compiled = runtime_contract.CompiledPlan{
        .plan = .{
            .instructions = instructions[0..],
            .register_count = 4,
            .state_descs = &.{},
        },
        .param_blocks = &.{},
        .weight_bindings = &.{},
        .register_buffer_specs = &.{
            .{ .size = 1, .@"align" = 4 },
            .{ .size = 1, .@"align" = 4 },
            .{ .size = 1, .@"align" = 4 },
            .{ .size = 1, .@"align" = 4 },
        },

        .liveness = .{
            .register_last_read = &.{ 0, 1, 2, 2 },
            .kill_after_instruction = &.{ kill0[0..], kill1[0..], kill2[0..] },
        },
        .peak_registers = 2,
        .diagnostics = &.{},
    };

    const map = try buildCudaLayerProgramRegisterSlotMap(std.testing.allocator, &compiled);
    defer std.testing.allocator.free(map);
    try std.testing.expect(map[1] < 2);
    try std.testing.expect(map[2] < 2);
    try std.testing.expect(map[3] < 2);
    try std.testing.expectEqual(map[1], map[3]);
    try std.testing.expect(map[2] != map[1]);
}

test "resolveDenseInOutLayout transposes [out,in] orientation" {
    const layout = try resolveDenseInOutLayout(256, 128, 128);
    try std.testing.expectEqual(@as(usize, 128), layout.in_dim);
    try std.testing.expectEqual(@as(usize, 256), layout.out_dim);
    try std.testing.expect(layout.needs_transpose);
}

test "resolveDenseInOutLayout rejects mismatched orientation" {
    try std.testing.expectError(error.UnsupportedModel, resolveDenseInOutLayout(96, 64, 128));
}

test "resolveDenseInOutLayout prefers [out,in] for square matrices" {
    const layout = try resolveDenseInOutLayout(256, 256, 256);
    try std.testing.expectEqual(@as(usize, 256), layout.in_dim);
    try std.testing.expectEqual(@as(usize, 256), layout.out_dim);
    try std.testing.expect(layout.needs_transpose);
}

test "resolveDenseOutInLayout keeps [out,in] orientation" {
    const layout = try resolveDenseOutInLayout(256, 128, 128);
    try std.testing.expectEqual(@as(usize, 128), layout.in_dim);
    try std.testing.expectEqual(@as(usize, 256), layout.out_dim);
    try std.testing.expect(!layout.needs_transpose);
}

test "resolveDenseOutInLayout transposes [in,out] orientation" {
    const layout = try resolveDenseOutInLayout(128, 256, 128);
    try std.testing.expectEqual(@as(usize, 128), layout.in_dim);
    try std.testing.expectEqual(@as(usize, 256), layout.out_dim);
    try std.testing.expect(layout.needs_transpose);
}

test "resolveDenseOutInLayout keeps square typed layout untransposed" {
    const layout = try resolveDenseOutInLayout(256, 256, 256);
    try std.testing.expectEqual(@as(usize, 256), layout.in_dim);
    try std.testing.expectEqual(@as(usize, 256), layout.out_dim);
    try std.testing.expect(!layout.needs_transpose);
}

test "transposeRowMajor transposes compact row-major matrix" {
    const src = [_]u16{
        1, 2, 3,
        4, 5, 6,
    };
    const transposed = try transposeRowMajor(u16, std.testing.allocator, src[0..], 2, 3);
    defer std.testing.allocator.free(transposed);

    const expected = [_]u16{
        1, 4,
        2, 5,
        3, 6,
    };
    try std.testing.expectEqualSlices(u16, expected[0..], transposed);
}

test "required_kernels contract has unique slots and operation names" {
    var seen_slots = std.AutoHashMap(KernelSlot, void).init(std.testing.allocator);
    defer seen_slots.deinit();

    var seen_ops = std.StringHashMap(void).init(std.testing.allocator);
    defer seen_ops.deinit();

    for (required_kernels) |entry| {
        const slot_put = try seen_slots.getOrPut(entry.slot);
        try std.testing.expect(!slot_put.found_existing);

        const op_put = try seen_ops.getOrPut(entry.op_name);
        try std.testing.expect(!op_put.found_existing);

        try std.testing.expect(std.mem.startsWith(u8, entry.embedded_symbol, "talu_"));
        try std.testing.expect(!hasVersionSuffixName(entry.op_name));
        try std.testing.expect(!hasVersionSuffixName(entry.embedded_symbol));
    }

    const slot_count = @typeInfo(KernelSlot).@"enum".fields.len;
    try std.testing.expectEqual(slot_count, required_kernels.len);
}

test "required_kernels keeps heads-based attention path canonical" {
    const required_ops = [_][]const u8{
        compute.cuda.attn_scores_heads_f32.op_name,
        compute.cuda.attn_scores_heads_f16_kv.op_name,
        compute.cuda.attn_weighted_sum_heads_f32.op_name,
        compute.cuda.attn_weighted_sum_heads_f16_kv.op_name,
        compute.cuda.softmax_rows.op_name,
    };
    const removed_ops = [_][]const u8{
        "attn_scores_f32",
        "attn_scores_f16_kv",
        "attn_weighted_sum_f32",
        "attn_weighted_sum_f16_kv",
        "softmax_f32",
    };

    for (required_ops) |op| {
        var found = false;
        for (required_kernels) |entry| {
            if (std.mem.eql(u8, entry.op_name, op)) {
                found = true;
                break;
            }
        }
        try std.testing.expect(found);
    }

    for (removed_ops) |op| {
        for (required_kernels) |entry| {
            try std.testing.expect(!std.mem.eql(u8, entry.op_name, op));
        }
    }
}

fn hasVersionSuffixName(name: []const u8) bool {
    const marker = "_v";
    const at = std.mem.lastIndexOf(u8, name, marker) orelse return false;
    const digits = name[at + marker.len ..];
    if (digits.len == 0) return false;
    for (digits) |ch| {
        if (!std.ascii.isDigit(ch)) return false;
    }
    return true;
}

test "tryPopulateProjectionFromWeight supports [d_model, vocab] layout" {
    const d_model: usize = 3;
    const projected_vocab: usize = 2;
    var weights = [_]f32{
        1.0, 2.0,  3.0,  4.0,
        5.0, 6.0,  7.0,  8.0,
        9.0, 10.0, 11.0, 12.0,
    };
    const weight_bytes = std.mem.sliceAsBytes(weights[0..]);
    const weight = Tensor.view(weight_bytes.ptr, &.{ d_model, 4 }, .f32, weight_bytes.len);
    var out = [_]f32{0.0} ** (d_model * projected_vocab);

    try std.testing.expect(tryPopulateProjectionFromWeight(std.testing.allocator, &weight, d_model, projected_vocab, out[0..]));
    const expected = [_]f32{ 1.0, 2.0, 5.0, 6.0, 9.0, 10.0 };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "tryPopulateProjectionFromWeight supports [vocab, d_model] layout" {
    const d_model: usize = 3;
    const projected_vocab: usize = 2;
    var weights = [_]f32{
        1.0,  2.0,  3.0,
        4.0,  5.0,  6.0,
        7.0,  8.0,  9.0,
        10.0, 11.0, 12.0,
    };
    const weight_bytes = std.mem.sliceAsBytes(weights[0..]);
    const weight = Tensor.view(weight_bytes.ptr, &.{ 4, d_model }, .f32, weight_bytes.len);
    var out = [_]f32{0.0} ** (d_model * projected_vocab);

    try std.testing.expect(tryPopulateProjectionFromWeight(std.testing.allocator, &weight, d_model, projected_vocab, out[0..]));
    const expected = [_]f32{ 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "tryPopulateHiddenFromToken supports [vocab, d_model] layout" {
    var embedding_data = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const bytes = std.mem.sliceAsBytes(embedding_data[0..]);
    const embeddings = Tensor.view(bytes.ptr, &.{ 2, 3 }, .f32, bytes.len);

    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = std.mem.zeroes(tensor.ModelConfig),
        .token_embeddings = embeddings,
        .blocks = &.{},
        .original_weight_dtype = .f32,
    };
    defer loaded.arena.deinit();

    var out = [_]f32{0.0} ** 3;
    try std.testing.expect(try tryPopulateHiddenFromToken(&loaded, 1, out[0..]));
    const expected = [_]f32{ 4.0, 5.0, 6.0 };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "tryPopulateHiddenFromToken supports [d_model, vocab] layout" {
    var embedding_data = [_]f32{
        1.0, 4.0,
        2.0, 5.0,
        3.0, 6.0,
    };
    const bytes = std.mem.sliceAsBytes(embedding_data[0..]);
    const embeddings = Tensor.view(bytes.ptr, &.{ 3, 2 }, .f32, bytes.len);

    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = std.mem.zeroes(tensor.ModelConfig),
        .token_embeddings = embeddings,
        .blocks = &.{},
        .original_weight_dtype = .f32,
    };
    defer loaded.arena.deinit();

    var out = [_]f32{0.0} ** 3;
    try std.testing.expect(try tryPopulateHiddenFromToken(&loaded, 1, out[0..]));
    const expected = [_]f32{ 4.0, 5.0, 6.0 };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "populatePrefillHiddenFromTokens applies embedding multiplier" {
    var embedding_data = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const bytes = std.mem.sliceAsBytes(embedding_data[0..]);
    const embeddings = Tensor.view(bytes.ptr, &.{ 2, 3 }, .f32, bytes.len);

    var cfg = std.mem.zeroes(tensor.ModelConfig);
    cfg.embedding_multiplier = 2.0;
    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = cfg,
        .token_embeddings = embeddings,
        .blocks = &.{},
        .original_weight_dtype = .f32,
    };
    defer loaded.arena.deinit();

    const tokens = [_]u32{ 1, 0 };
    var out = [_]f32{0.0} ** 6;
    try populatePrefillHiddenFromTokens(&loaded, tokens[0..], 3, out[0..], null);

    const expected = [_]f32{
        8.0, 10.0, 12.0,
        2.0, 4.0,  6.0,
    };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "populatePrefillHiddenFromTokens zero-fills configured skip token rows" {
    var embedding_data = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const bytes = std.mem.sliceAsBytes(embedding_data[0..]);
    const embeddings = Tensor.view(bytes.ptr, &.{ 2, 3 }, .f32, bytes.len);

    var cfg = std.mem.zeroes(tensor.ModelConfig);
    cfg.embedding_multiplier = 1.0;
    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = cfg,
        .token_embeddings = embeddings,
        .blocks = &.{},
        .original_weight_dtype = .f32,
    };
    defer loaded.arena.deinit();

    const tokens = [_]u32{ 1, 99, 0 };
    var out = [_]f32{0.0} ** 9;
    try populatePrefillHiddenFromTokens(&loaded, tokens[0..], 3, out[0..], 99);

    const expected = [_]f32{
        4.0, 5.0, 6.0,
        0.0, 0.0, 0.0,
        1.0, 2.0, 3.0,
    };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "gaffineValueAt decodes grouped_affine_u4 values" {
    var packed_words = [_]u32{
        // 8 packed 4-bit values: 0,1,2,3,4,5,6,7
        0x7654_3210,
    };
    const packed_bytes = std.mem.sliceAsBytes(packed_words[0..]);

    const one_bf16 = dtype.f32ToBf16(1.0);
    const zero_bf16 = dtype.f32ToBf16(0.0);
    var scales_u16 = [_]u16{one_bf16};
    var biases_u16 = [_]u16{zero_bf16};
    const scales_bytes = std.mem.sliceAsBytes(scales_u16[0..]);
    const biases_bytes = std.mem.sliceAsBytes(biases_u16[0..]);

    var weight = Tensor.view(packed_bytes.ptr, &.{ 1, 8 }, .grouped_affine_u4, packed_bytes.len);
    weight.gaffine = .{
        .scales = scales_bytes,
        .biases = biases_bytes,
        .group_size = 8,
        .scales_dtype = .bf16,
    };

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), try gaffineValueAt(&weight, 0, 0), 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), try gaffineValueAt(&weight, 0, 3), 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), try gaffineValueAt(&weight, 0, 7), 0.0);
}

test "tryPopulateFinalNormWeight supports bf16 weights" {
    var norm_u16 = [_]u16{
        dtype.f32ToBf16(1.25),
        dtype.f32ToBf16(-0.5),
    };
    const norm_bytes = std.mem.sliceAsBytes(norm_u16[0..]);
    const norm_tensor = Tensor.view(norm_bytes.ptr, &.{2}, .bf16, norm_bytes.len);

    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = std.mem.zeroes(tensor.ModelConfig),
        .ln_final = norm_tensor,
        .token_embeddings = std.mem.zeroes(Tensor),
        .blocks = &.{},
        .original_weight_dtype = .bf16,
    };
    defer loaded.arena.deinit();

    var out = [_]f32{0.0} ** 2;
    try std.testing.expect(tryPopulateFinalNormWeight(&loaded, out[0..]));
    try std.testing.expectApproxEqAbs(@as(f32, 1.25), out[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), out[1], 0.01);
}

test "populatePrefillHiddenFromTokens rejects missing embeddings" {
    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = std.mem.zeroes(tensor.ModelConfig),
        .token_embeddings = std.mem.zeroes(Tensor),
        .blocks = &.{},
        .original_weight_dtype = .f32,
    };
    defer loaded.arena.deinit();

    const tokens = [_]u32{0};
    var out = [_]f32{0.0} ** 4;
    try std.testing.expectError(
        error.UnsupportedModel,
        populatePrefillHiddenFromTokens(&loaded, tokens[0..], 4, out[0..], null),
    );
}

test "shouldDownloadPrefillLogits only on final token" {
    try std.testing.expect(!shouldDownloadPrefillLogits(0, 4));
    try std.testing.expect(!shouldDownloadPrefillLogits(1, 4));
    try std.testing.expect(!shouldDownloadPrefillLogits(2, 4));
    try std.testing.expect(shouldDownloadPrefillLogits(3, 4));
}

test "shouldDownloadPrefillLogits true for single-token prefill" {
    try std.testing.expect(shouldDownloadPrefillLogits(0, 1));
}

test "linearWeightSupportsSequenceRows allows gaffine when matvec kernel is loaded" {
    const dummy_buffer: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 };
    var weight = LinearWeight{
        .gaffine_u4 = .{
            .rows = 16,
            .cols = 16,
            .packed_data = dummy_buffer,
            .scales = dummy_buffer,
            .biases = dummy_buffer,
            .group_size = 8,
            .scales_dtype_tag = gaffine_scales_dtype_bf16,
        },
    };

    try std.testing.expect(!CudaBackend.linearWeightSupportsSequenceRowsForKernels(&weight, false, false, false, false, false));
    try std.testing.expect(CudaBackend.linearWeightSupportsSequenceRowsForKernels(&weight, false, false, true, false, false));
}

test "canFuseDenseU16QkvWeights supports GQA-style unequal output dims" {
    const dummy_buffer: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 };
    const q = U16LinearWeight{
        .rows = 2048,
        .cols = 2048,
        .buffer = dummy_buffer,
        .dtype = .bf16,
    };
    const k = U16LinearWeight{
        .rows = 2048,
        .cols = 256,
        .buffer = dummy_buffer,
        .dtype = .bf16,
    };
    const v = U16LinearWeight{
        .rows = 2048,
        .cols = 256,
        .buffer = dummy_buffer,
        .dtype = .bf16,
    };

    try std.testing.expect(engine_ops.canFuseDenseU16QkvWeights(2048, q, k, v));
}

test "canFuseDenseU16QkvWeights rejects mixed dtypes" {
    const dummy_buffer: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 };
    const q = U16LinearWeight{
        .rows = 1024,
        .cols = 1024,
        .buffer = dummy_buffer,
        .dtype = .f16,
    };
    const k = U16LinearWeight{
        .rows = 1024,
        .cols = 256,
        .buffer = dummy_buffer,
        .dtype = .bf16,
    };
    const v = U16LinearWeight{
        .rows = 1024,
        .cols = 256,
        .buffer = dummy_buffer,
        .dtype = .bf16,
    };

    try std.testing.expect(!engine_ops.canFuseDenseU16QkvWeights(1024, q, k, v));
}

test "collectTokenPositions returns all matching positions" {
    const tokens = [_]u32{ 7, 3, 7, 9, 7 };
    const positions = try collectTokenPositions(std.testing.allocator, tokens[0..], 7);
    defer if (positions.len > 0) std.testing.allocator.free(positions);

    const expected = [_]usize{ 0, 2, 4 };
    try std.testing.expectEqualSlices(usize, expected[0..], positions);
}

test "collectTokenPositions returns empty when token is absent" {
    const tokens = [_]u32{ 1, 2, 3 };
    const positions = try collectTokenPositions(std.testing.allocator, tokens[0..], 9);
    try std.testing.expectEqual(@as(usize, 0), positions.len);
}

test "findPositionIndex locates mapped image feature index" {
    const positions = [_]usize{ 2, 5, 9 };
    try std.testing.expectEqual(@as(?usize, 0), findPositionIndex(positions[0..], 2));
    try std.testing.expectEqual(@as(?usize, 1), findPositionIndex(positions[0..], 5));
    try std.testing.expectEqual(@as(?usize, 2), findPositionIndex(positions[0..], 9));
    try std.testing.expectEqual(@as(?usize, null), findPositionIndex(positions[0..], 7));
}

test "deepstackLayersCompatibleWithPrompt accepts valid layers" {
    const d_model: usize = 4;
    const image_positions: usize = 2;
    const layer0 = [_]f32{0} ** (2 * 4);
    const layer1 = [_]f32{0} ** (3 * 4);
    const layers = [_][]const f32{ layer0[0..], layer1[0..] };
    try std.testing.expect(deepstackLayersCompatibleWithPrompt(layers[0..], image_positions, d_model));
}

test "deepstackLayersCompatibleWithPrompt rejects malformed layers" {
    const d_model: usize = 4;
    const image_positions: usize = 2;
    const too_few_rows = [_]f32{0} ** (1 * 4);
    const bad_stride = [_]f32{0} ** 7;
    const valid = [_]f32{0} ** (2 * 4);

    const layers_few = [_][]const f32{too_few_rows[0..]};
    try std.testing.expect(!deepstackLayersCompatibleWithPrompt(layers_few[0..], image_positions, d_model));

    const layers_stride = [_][]const f32{bad_stride[0..]};
    try std.testing.expect(!deepstackLayersCompatibleWithPrompt(layers_stride[0..], image_positions, d_model));

    const layers_zero_dim = [_][]const f32{valid[0..]};
    try std.testing.expect(!deepstackLayersCompatibleWithPrompt(layers_zero_dim[0..], image_positions, 0));
}

test "materializeDenseOutInU16 handles out-in and in-out source layouts" {
    var out_in_data = [_]u16{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    };
    var out_in_tensor = Tensor.view(std.mem.sliceAsBytes(out_in_data[0..]).ptr, &.{ 4, 3 }, .bf16, std.mem.sliceAsBytes(out_in_data[0..]).len);
    var out_in_view = try materializeDenseOutInU16(std.testing.allocator, &out_in_tensor, 3, 4);
    defer out_in_view.deinit(std.testing.allocator);
    try std.testing.expect(out_in_view.owned == null);
    for (out_in_data, out_in_view.values) |want, got| {
        try std.testing.expectEqual(want, got);
    }

    var in_out_data = [_]u16{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    };
    var in_out_tensor = Tensor.view(std.mem.sliceAsBytes(in_out_data[0..]).ptr, &.{ 3, 4 }, .bf16, std.mem.sliceAsBytes(in_out_data[0..]).len);
    var in_out_view = try materializeDenseOutInU16(std.testing.allocator, &in_out_tensor, 3, 4);
    defer in_out_view.deinit(std.testing.allocator);
    try std.testing.expect(in_out_view.owned != null);
    const expected = [_]u16{
        1, 5, 9,
        2, 6, 10,
        3, 7, 11,
        4, 8, 12,
    };
    for (expected, in_out_view.values) |want, got| {
        try std.testing.expectEqual(want, got);
    }
}

test "materializeDenseOutInF32 handles out-in and in-out source layouts" {
    var out_in_data = [_]f32{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    };
    var out_in_tensor = Tensor.view(std.mem.sliceAsBytes(out_in_data[0..]).ptr, &.{ 4, 3 }, .f32, std.mem.sliceAsBytes(out_in_data[0..]).len);
    var out_in_view = try materializeDenseOutInF32(std.testing.allocator, &out_in_tensor, 3, 4);
    defer out_in_view.deinit(std.testing.allocator);
    try std.testing.expect(out_in_view.owned == null);
    try std.testing.expectEqualSlices(f32, out_in_data[0..], out_in_view.values);

    var in_out_data = [_]f32{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    };
    var in_out_tensor = Tensor.view(std.mem.sliceAsBytes(in_out_data[0..]).ptr, &.{ 3, 4 }, .f32, std.mem.sliceAsBytes(in_out_data[0..]).len);
    var in_out_view = try materializeDenseOutInF32(std.testing.allocator, &in_out_tensor, 3, 4);
    defer in_out_view.deinit(std.testing.allocator);
    try std.testing.expect(in_out_view.owned != null);
    const expected = [_]f32{
        1, 5, 9,
        2, 6, 10,
        3, 7, 11,
        4, 8, 12,
    };
    try std.testing.expectEqualSlices(f32, expected[0..], in_out_view.values);
}

test "gaffineValueAt decodes grouped_affine_u8 values" {
    var packed_words = [_]u32{
        0x0302_0100,
        0x0706_0504,
    };
    const packed_bytes = std.mem.sliceAsBytes(packed_words[0..]);

    var scales_u16 = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(1.0),
    };
    var biases_u16 = [_]u16{
        dtype.f32ToBf16(0.0),
        dtype.f32ToBf16(0.0),
    };
    const scales_bytes = std.mem.sliceAsBytes(scales_u16[0..]);
    const biases_bytes = std.mem.sliceAsBytes(biases_u16[0..]);

    var weight = Tensor.view(packed_bytes.ptr, &.{ 1, 8 }, .grouped_affine_u8, packed_bytes.len);
    weight.gaffine = .{
        .scales = scales_bytes,
        .biases = biases_bytes,
        .group_size = 4,
        .scales_dtype = .bf16,
    };

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), try gaffineValueAt(&weight, 0, 0), 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), try gaffineValueAt(&weight, 0, 3), 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), try gaffineValueAt(&weight, 0, 7), 0.0);
}

test "canFuseGaffineGateUpWeights accepts grouped-affine u8 weights with matching metadata" {
    const buf = compute.cuda.Buffer{ .pointer = 0, .size = 0 };
    const gate = GaffineU8LinearWeight{
        .rows = 4096,
        .cols = 11008,
        .packed_data = buf,
        .scales = buf,
        .biases = buf,
        .group_size = 32,
        .scales_dtype_tag = compute.cuda.gaffine_u8_matvec.scales_dtype_bf16,
    };
    const up = GaffineU8LinearWeight{
        .rows = 4096,
        .cols = 11008,
        .packed_data = buf,
        .scales = buf,
        .biases = buf,
        .group_size = 32,
        .scales_dtype_tag = compute.cuda.gaffine_u8_matvec.scales_dtype_bf16,
    };

    try std.testing.expect(engine_ops.canFuseGaffineGateUpWeights(4096, gate, up));

    var bad_up = up;
    bad_up.scales_dtype_tag = compute.cuda.gaffine_u8_matvec.scales_dtype_f16;
    try std.testing.expect(!engine_ops.canFuseGaffineGateUpWeights(4096, gate, bad_up));
}

test "canFuseGaffineQkvWeights accepts grouped-affine u8 weights with matching metadata" {
    const buf = compute.cuda.Buffer{ .pointer = 0, .size = 0 };
    const q = GaffineU8LinearWeight{
        .rows = 4096,
        .cols = 8192,
        .packed_data = buf,
        .scales = buf,
        .biases = buf,
        .group_size = 32,
        .scales_dtype_tag = compute.cuda.gaffine_u8_matvec.scales_dtype_bf16,
    };
    const k = GaffineU8LinearWeight{
        .rows = 4096,
        .cols = 1024,
        .packed_data = buf,
        .scales = buf,
        .biases = buf,
        .group_size = 32,
        .scales_dtype_tag = compute.cuda.gaffine_u8_matvec.scales_dtype_bf16,
    };
    const v = GaffineU8LinearWeight{
        .rows = 4096,
        .cols = 1024,
        .packed_data = buf,
        .scales = buf,
        .biases = buf,
        .group_size = 32,
        .scales_dtype_tag = compute.cuda.gaffine_u8_matvec.scales_dtype_bf16,
    };

    try std.testing.expect(engine_ops.canFuseGaffineQkvWeights(4096, q, k, v));

    var bad_v = v;
    bad_v.scales_dtype_tag = compute.cuda.gaffine_u8_matvec.scales_dtype_f16;
    try std.testing.expect(!engine_ops.canFuseGaffineQkvWeights(4096, q, k, bad_v));
}

test "BlockRuntimeLayer.rebuildInstructionMetadata binds per-op runtime metadata" {
    var layer: BlockRuntimeLayer = .{};
    defer {
        if (layer.instruction_norm_weight_slots.len > 0) std.testing.allocator.free(layer.instruction_norm_weight_slots);
        if (layer.instruction_attention_exec_meta.len > 0) std.testing.allocator.free(layer.instruction_attention_exec_meta);
        if (layer.instruction_attention_weight_slots.len > 0) std.testing.allocator.free(layer.instruction_attention_weight_slots);
        if (layer.instruction_shortconv_exec_meta.len > 0) std.testing.allocator.free(layer.instruction_shortconv_exec_meta);
        if (layer.instruction_shortconv_weight_slots.len > 0) std.testing.allocator.free(layer.instruction_shortconv_weight_slots);
        if (layer.instruction_weight_offsets.len > 0) std.testing.allocator.free(layer.instruction_weight_offsets);
        if (layer.instruction_weight_ptrs.len > 0) std.testing.allocator.free(layer.instruction_weight_ptrs);
        if (layer.compiled_plan) |*compiled| {
            plan_compiler.deinitCompiledPlan(std.testing.allocator, compiled);
            layer.compiled_plan = null;
        }
    }

    const zero_buffer = std.mem.zeroes(compute.cuda.Buffer);
    const zero_tensor = DeviceTensor{
        .rows = 0,
        .cols = 0,
        .buffer = zero_buffer,
    };
    const zero_weight = LinearWeight{ .dense_f32 = zero_tensor };
    var norm0: DeviceTensor = zero_tensor;
    var norm1: DeviceTensor = zero_tensor;
    var attention_runtime: LayerAttentionRuntime = .{
        .q_dim = 0,
        .q_projection_dim = 0,
        .kv_dim = 0,
        .d_ff = 0,
        .sliding_window = 0,
        .is_causal = true,
        .query_gate = false,
        .ln1_weight = zero_tensor,
        .ln2_weight = zero_tensor,
        .pre_ffn_norm_weight = null,
        .post_ffn_norm_weight = null,
        .q_norm_weight = null,
        .k_norm_weight = null,
        .q_proj = zero_weight,
        .k_proj = zero_weight,
        .v_proj = zero_weight,
        .o_proj = zero_weight,
        .w1 = zero_weight,
        .w2 = zero_weight,
        .w3 = zero_weight,
        .k_cache = zero_buffer,
        .v_cache = zero_buffer,
        .kv_capacity = 0,
        .slot_kv_index = 0,
    };
    var shortconv_runtime: ShortConvBlockRuntime = .{
        .conv_dim = 0,
        .d_conv = 0,
        .d_ff = 0,
        .ln1_weight = zero_tensor,
        .ln2_weight = null,
        .in_proj = zero_weight,
        .out_proj = zero_weight,
        .conv_weight_time_major = zero_tensor,
        .conv_bias = null,
        .conv_state = zero_buffer,
        .ffn_w1 = null,
        .ffn_w2 = null,
        .ffn_w3 = null,
    };
    const gate_weight: LinearWeight = zero_weight;
    const up_weight: LinearWeight = zero_weight;
    const down_weight: LinearWeight = zero_weight;

    layer.norm_weights[0] = &norm0;
    layer.norm_weights[1] = &norm1;
    layer.norm_weight_count = 2;
    attention_runtime.w1 = gate_weight;
    attention_runtime.w3 = up_weight;
    attention_runtime.w2 = down_weight;
    attention_runtime.d_ff = 32;
    layer.attention_binding = &attention_runtime;
    layer.shortconv_binding = &shortconv_runtime;

    const ops = [_]layer_ops.LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention, .state_block_id = runtime_contract.kv_cache_state_id } },
        .{ .kernel = .{ .id = 2, .in = .branch_out, .out = .tmp3, .debug_type = .shortconv, .state_block_id = runtime_contract.shortconv_state_id } },
        .{ .kernel = .{ .id = 3, .in = .tmp3, .out = .branch_out, .debug_type = .mlp } },
        .{ .kernel = .{ .id = 4, .in = .residual, .out = .norm_out, .debug_type = .norm } },
    };
    layer.compiled_plan = try plan_compiler.compileLayerProgram(std.testing.allocator, ops[0..], .decode, .{});

    try layer.rebuildInstructionMetadata(std.testing.allocator);

    try std.testing.expect(layer.instruction_norm_weight_slots[0].? == &norm0);
    try std.testing.expect(layer.instruction_norm_weight_slots[4].? == &norm1);
    try std.testing.expectEqual(@as(usize, ops.len + 1), layer.instruction_weight_offsets.len);
    try std.testing.expect(layer.instruction_weight_ptrs.len != 0);
    // Instruction weight pointers are flattened and directly sourced from layer bindings.
    const attn_start = layer.instruction_weight_offsets[1];
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[attn_start + 0].?), @intFromPtr(&attention_runtime.q_proj));
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[attn_start + 1].?), @intFromPtr(&attention_runtime.k_proj));
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[attn_start + 2].?), @intFromPtr(&attention_runtime.v_proj));
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[attn_start + 3].?), @intFromPtr(&attention_runtime.o_proj));
    const shortconv_start = layer.instruction_weight_offsets[2];
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[shortconv_start + 0].?), @intFromPtr(&shortconv_runtime.in_proj));
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[shortconv_start + 1].?), @intFromPtr(&shortconv_runtime.conv_weight_time_major));
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[shortconv_start + 2].?), @intFromPtr(&shortconv_runtime.out_proj));
}

test "BlockRuntimeLayer.rebuildInstructionMetadata rejects norm op without bound norm weights" {
    var layer: BlockRuntimeLayer = .{};
    defer {
        if (layer.instruction_norm_weight_slots.len > 0) std.testing.allocator.free(layer.instruction_norm_weight_slots);
        if (layer.instruction_attention_exec_meta.len > 0) std.testing.allocator.free(layer.instruction_attention_exec_meta);
        if (layer.instruction_attention_weight_slots.len > 0) std.testing.allocator.free(layer.instruction_attention_weight_slots);
        if (layer.instruction_shortconv_exec_meta.len > 0) std.testing.allocator.free(layer.instruction_shortconv_exec_meta);
        if (layer.instruction_shortconv_weight_slots.len > 0) std.testing.allocator.free(layer.instruction_shortconv_weight_slots);
        if (layer.instruction_weight_offsets.len > 0) std.testing.allocator.free(layer.instruction_weight_offsets);
        if (layer.instruction_weight_ptrs.len > 0) std.testing.allocator.free(layer.instruction_weight_ptrs);
        if (layer.compiled_plan) |*compiled| {
            plan_compiler.deinitCompiledPlan(std.testing.allocator, compiled);
            layer.compiled_plan = null;
        }
    }

    const ops = [_]layer_ops.LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
    };
    layer.compiled_plan = try plan_compiler.compileLayerProgram(std.testing.allocator, ops[0..], .decode, .{});

    try std.testing.expectError(error.UnsupportedModel, layer.rebuildInstructionMetadata(std.testing.allocator));
}

test "bindSlotStateBlocks stores typed runtime states by runtime_kind" {
    const payload_bytes: usize = @intCast(runtime_contract.builtin_state_block_bytes);
    var backend: CudaBackend = undefined;
    backend.max_batch_size = 1;
    backend.block_runtime = undefined;
    backend.pipeline_backend1 = null;
    backend.pipeline_backend0_cpu = null;
    backend.state_descriptor_count = 4;
    backend.state_descriptors_storage[0] = .{
        .id = 91,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
    };
    backend.state_descriptors_storage[1] = .{
        .id = 92,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_shortconv_cache,
    };
    backend.state_descriptors_storage[2] = .{
        .id = 93,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_mamba_cache,
    };
    backend.state_descriptors_storage[3] = .{
        .id = 94,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_gated_delta_cache,
    };
    var slot_state_bindings: [1]CudaBackend.SlotStateBinding = .{.{}};
    backend.slot_state_bindings = slot_state_bindings[0..];

    var kv_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    var shortconv_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    var mamba_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    var gated_delta_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    const state_blocks = [_]runtime_contract.StateBlockHandle{
        .{
            .id = 91,
            .ptr = kv_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
        .{
            .id = 92,
            .ptr = shortconv_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
        .{
            .id = 93,
            .ptr = mamba_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
        .{
            .id = 94,
            .ptr = gated_delta_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
    };

    try backend.bindSlotStateBlocks(0, state_blocks[0..]);
    defer backend.unbindSlotStateBlocks(0);
    const bound = backend.slotStateBlocks(0);
    const kv_state = runtime_contract.stateValueFromBlock(*KvRuntimeState, &bound[0]) orelse return error.TestUnexpectedResult;
    const shortconv_state = runtime_contract.stateValueFromBlock(*ShortConvRuntimeState, &bound[1]) orelse return error.TestUnexpectedResult;
    const mamba_state = runtime_contract.stateValueFromBlock(*MambaRuntimeState, &bound[2]) orelse return error.TestUnexpectedResult;
    const gated_delta_state = runtime_contract.stateValueFromBlock(*GatedDeltaRuntimeState, &bound[3]) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_kv_cache, kv_state.runtime_kind);
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_shortconv_cache, shortconv_state.runtime_kind);
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_mamba_cache, mamba_state.runtime_kind);
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_gated_delta_cache, gated_delta_state.runtime_kind);
    try std.testing.expectEqual(@intFromPtr(&backend.block_runtime), @intFromPtr(kv_state.block_runtime));
    try std.testing.expectEqual(@intFromPtr(&backend.block_runtime), @intFromPtr(shortconv_state.block_runtime));
    try std.testing.expectEqual(@intFromPtr(&backend.block_runtime), @intFromPtr(mamba_state.block_runtime));
    try std.testing.expectEqual(@intFromPtr(&backend.block_runtime), @intFromPtr(gated_delta_state.block_runtime));
    try std.testing.expectEqual(@as(usize, 0), kv_state.slot_index);
    try std.testing.expectEqual(@as(usize, 0), shortconv_state.slot_index);
    try std.testing.expectEqual(@as(usize, 0), mamba_state.slot_index);
    try std.testing.expectEqual(@as(usize, 0), gated_delta_state.slot_index);
}

test "bindSlotStateBlocks preserves bound slot index in runtime states" {
    const payload_bytes: usize = @intCast(runtime_contract.builtin_state_block_bytes);
    var backend: CudaBackend = undefined;
    backend.max_batch_size = 2;
    backend.block_runtime = undefined;
    backend.pipeline_backend1 = null;
    backend.pipeline_backend0_cpu = null;
    backend.state_descriptor_count = 4;
    backend.state_descriptors_storage[0] = .{
        .id = 101,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
    };
    backend.state_descriptors_storage[1] = .{
        .id = 102,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_shortconv_cache,
    };
    backend.state_descriptors_storage[2] = .{
        .id = 103,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_mamba_cache,
    };
    backend.state_descriptors_storage[3] = .{
        .id = 104,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_gated_delta_cache,
    };
    var slot_state_bindings: [2]CudaBackend.SlotStateBinding = .{ .{}, .{} };
    backend.slot_state_bindings = slot_state_bindings[0..];

    var kv_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    var shortconv_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    var mamba_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    var gated_delta_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    const state_blocks = [_]runtime_contract.StateBlockHandle{
        .{
            .id = 101,
            .ptr = kv_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
        .{
            .id = 102,
            .ptr = shortconv_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
        .{
            .id = 103,
            .ptr = mamba_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
        .{
            .id = 104,
            .ptr = gated_delta_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
    };

    try backend.bindSlotStateBlocks(1, state_blocks[0..]);
    defer backend.unbindSlotStateBlocks(1);
    const bound = backend.slotStateBlocks(1);
    const kv_state = runtime_contract.stateValueFromBlock(*KvRuntimeState, &bound[0]) orelse return error.TestUnexpectedResult;
    const shortconv_state = runtime_contract.stateValueFromBlock(*ShortConvRuntimeState, &bound[1]) orelse return error.TestUnexpectedResult;
    const mamba_state = runtime_contract.stateValueFromBlock(*MambaRuntimeState, &bound[2]) orelse return error.TestUnexpectedResult;
    const gated_delta_state = runtime_contract.stateValueFromBlock(*GatedDeltaRuntimeState, &bound[3]) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_kv_cache, kv_state.runtime_kind);
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_shortconv_cache, shortconv_state.runtime_kind);
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_mamba_cache, mamba_state.runtime_kind);
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_gated_delta_cache, gated_delta_state.runtime_kind);
    try std.testing.expectEqual(@as(usize, 1), kv_state.slot_index);
    try std.testing.expectEqual(@as(usize, 1), shortconv_state.slot_index);
    try std.testing.expectEqual(@as(usize, 1), mamba_state.slot_index);
    try std.testing.expectEqual(@as(usize, 1), gated_delta_state.slot_index);
}

test "bindSlotStateBlocks rolls back self on cpu stage bind failure" {
    const payload_bytes: usize = @intCast(runtime_contract.builtin_state_block_bytes);
    var backend: CudaBackend = undefined;
    backend.max_batch_size = 1;
    backend.block_runtime = undefined;
    backend.state_descriptor_count = 1;
    backend.pipeline_backend1 = null;
    backend.state_descriptors_storage[0] = .{
        .id = 111,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_none,
    };
    var slot_state_bindings: [1]CudaBackend.SlotStateBinding = .{.{}};
    backend.slot_state_bindings = slot_state_bindings[0..];

    // Zero-initialized CPU backend stub: slot_state_bindings.len == 0
    // triggers the bounds check in CPU bindSlotStateBlocks before any other field access.
    // Uses aligned byte buffer to satisfy the struct's natural alignment requirement.
    const cpu_backend = @import("../cpu/root.zig");
    var cpu_stage0_bytes: [@sizeOf(cpu_backend.BackendType)]u8 align(@alignOf(cpu_backend.BackendType)) = @splat(0);
    const cpu_stage0: *cpu_backend.BackendType = @ptrCast(&cpu_stage0_bytes);
    backend.pipeline_backend0_cpu = cpu_stage0;

    var state_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    const state_blocks = [_]runtime_contract.StateBlockHandle{
        .{
            .id = 111,
            .ptr = state_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
    };

    // Bind should fail because CPU stage rejects slot_index 0 (empty bindings slice).
    try std.testing.expectError(error.InvalidArgument, backend.bindSlotStateBlocks(0, state_blocks[0..]));
    // Self-binding must be rolled back: slot should not be marked bound.
    try std.testing.expect(!backend.slot_state_bindings[0].bound);
}

test "bindSlotStateBlocks preserves opaque descriptor blocks with runtime_kind none" {
    const payload_bytes: usize = @intCast(runtime_contract.builtin_state_block_bytes);
    var backend: CudaBackend = undefined;
    backend.max_batch_size = 1;
    backend.block_runtime = undefined;
    backend.pipeline_backend1 = null;
    backend.pipeline_backend0_cpu = null;
    backend.state_descriptor_count = 1;
    backend.state_descriptors_storage[0] = .{
        .id = 111,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_none,
    };
    var slot_state_bindings: [1]CudaBackend.SlotStateBinding = .{.{}};
    backend.slot_state_bindings = slot_state_bindings[0..];

    var state_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    const state_blocks = [_]runtime_contract.StateBlockHandle{
        .{
            .id = 111,
            .ptr = state_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
    };

    try backend.bindSlotStateBlocks(0, state_blocks[0..]);
    defer backend.unbindSlotStateBlocks(0);
    const bound = backend.slotStateBlocks(0);
    try std.testing.expectEqual(@as(usize, 1), bound.len);
    try std.testing.expectEqual(@intFromPtr(state_blocks[0].ptr), @intFromPtr(bound[0].ptr));
    try std.testing.expectEqual(state_blocks[0].size, bound[0].size);
    try std.testing.expectEqual(state_blocks[0].align_bytes, bound[0].align_bytes);
}

test "mirrorSlotStateBlocksFrom synthesizes stage-local runtime descriptors without aliasing source" {
    const payload_bytes: usize = @intCast(runtime_contract.builtin_state_block_bytes);

    var source: CudaBackend = undefined;
    source.max_batch_size = 1;
    source.block_runtime = undefined;
    source.state_descriptor_count = 1;
    source.state_descriptors_storage[0] = .{
        .id = 201,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
    };
    var source_slot_bindings: [1]CudaBackend.SlotStateBinding = .{.{}};
    source.slot_state_bindings = source_slot_bindings[0..];

    var target: CudaBackend = undefined;
    target.max_batch_size = 1;
    target.block_runtime = undefined;
    target.state_descriptor_count = 2;
    target.state_descriptors_storage[0] = .{
        .id = 201,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
    };
    target.state_descriptors_storage[1] = .{
        .id = 202,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = true,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_mamba_cache,
    };
    var target_slot_bindings: [1]CudaBackend.SlotStateBinding = .{.{}};
    target.slot_state_bindings = target_slot_bindings[0..];

    var source_kv_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    const source_blocks = [_]runtime_contract.StateBlockHandle{
        .{
            .id = 201,
            .ptr = source_kv_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
    };
    try source.bindSlotStateBlocks(0, source_blocks[0..]);
    defer source.unbindSlotStateBlocks(0);
    const source_before = source.slotStateBlocks(0);
    const source_before_kv = runtime_contract.findStateBlock(source_before, 201) orelse return error.TestUnexpectedResult;
    const source_before_kv_state = runtime_contract.stateValueFromBlock(*KvRuntimeState, source_before_kv) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@intFromPtr(&source.block_runtime), @intFromPtr(source_before_kv_state.block_runtime));

    try target.mirrorSlotStateBlocksFrom(&source, 0);
    defer target.unbindSlotStateBlocks(0);

    const mirrored = target.slotStateBlocks(0);
    try std.testing.expectEqual(@as(usize, 2), mirrored.len);

    const mirrored_kv = runtime_contract.findStateBlock(mirrored, 201) orelse return error.TestUnexpectedResult;
    try std.testing.expect(@intFromPtr(source_blocks[0].ptr) != @intFromPtr(mirrored_kv.ptr));
    const kv_state = runtime_contract.stateValueFromBlock(*KvRuntimeState, mirrored_kv) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_kv_cache, kv_state.runtime_kind);
    try std.testing.expectEqual(@as(usize, 0), kv_state.slot_index);
    try std.testing.expectEqual(@intFromPtr(&target.block_runtime), @intFromPtr(kv_state.block_runtime));
    const source_after = source.slotStateBlocks(0);
    const source_after_kv = runtime_contract.findStateBlock(source_after, 201) orelse return error.TestUnexpectedResult;
    const source_after_kv_state = runtime_contract.stateValueFromBlock(*KvRuntimeState, source_after_kv) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@intFromPtr(&source.block_runtime), @intFromPtr(source_after_kv_state.block_runtime));

    const mirrored_mamba = runtime_contract.findStateBlock(mirrored, 202) orelse return error.TestUnexpectedResult;
    const mamba_state = runtime_contract.stateValueFromBlock(*MambaRuntimeState, mirrored_mamba) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_mamba_cache, mamba_state.runtime_kind);
    try std.testing.expectEqual(@as(usize, 0), mamba_state.slot_index);
    try std.testing.expectEqual(@intFromPtr(&target.block_runtime), @intFromPtr(mamba_state.block_runtime));
}

test "mirrorSlotStateBlocksFrom rejects missing opaque descriptor without runtime synthesis" {
    const payload_bytes: usize = @intCast(runtime_contract.builtin_state_block_bytes);

    var source: CudaBackend = undefined;
    source.max_batch_size = 1;
    source.block_runtime = undefined;
    source.state_descriptor_count = 1;
    source.state_descriptors_storage[0] = .{
        .id = 211,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
    };
    var source_slot_bindings: [1]CudaBackend.SlotStateBinding = .{.{}};
    source.slot_state_bindings = source_slot_bindings[0..];

    var target: CudaBackend = undefined;
    target.max_batch_size = 1;
    target.block_runtime = undefined;
    target.state_descriptor_count = 2;
    target.state_descriptors_storage[0] = .{
        .id = 211,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
    };
    target.state_descriptors_storage[1] = .{
        .id = 212,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_none,
    };
    var target_slot_bindings: [1]CudaBackend.SlotStateBinding = .{.{}};
    target.slot_state_bindings = target_slot_bindings[0..];

    var source_kv_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    const source_blocks = [_]runtime_contract.StateBlockHandle{
        .{
            .id = 211,
            .ptr = source_kv_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
    };
    try source.bindSlotStateBlocks(0, source_blocks[0..]);
    defer source.unbindSlotStateBlocks(0);

    try std.testing.expectError(error.InvalidStateDescriptorBinding, target.mirrorSlotStateBlocksFrom(&source, 0));
}

test "mirrorSlotStateBlocksFrom synthesizes runtime descriptors when source slot is unbound" {
    var source: CudaBackend = undefined;
    source.max_batch_size = 1;
    source.block_runtime = undefined;
    source.state_descriptor_count = 0;
    var source_slot_bindings: [1]CudaBackend.SlotStateBinding = .{.{}};
    source.slot_state_bindings = source_slot_bindings[0..];

    var target: CudaBackend = undefined;
    target.max_batch_size = 1;
    target.block_runtime = undefined;
    target.state_descriptor_count = 1;
    target.state_descriptors_storage[0] = .{
        .id = 221,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = true,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_mamba_cache,
    };
    var target_slot_bindings: [1]CudaBackend.SlotStateBinding = .{.{}};
    target.slot_state_bindings = target_slot_bindings[0..];

    try target.mirrorSlotStateBlocksFrom(&source, 0);
    defer target.unbindSlotStateBlocks(0);

    const mirrored = target.slotStateBlocks(0);
    try std.testing.expectEqual(@as(usize, 1), mirrored.len);
    const mamba_state = runtime_contract.stateValueFromBlock(*MambaRuntimeState, &mirrored[0]) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_mamba_cache, mamba_state.runtime_kind);
    try std.testing.expectEqual(@as(usize, 0), mamba_state.slot_index);
    try std.testing.expectEqual(@intFromPtr(&target.block_runtime), @intFromPtr(mamba_state.block_runtime));
}

test "expectedAttentionQProjectionDim uses packed query width only for query-gated attention" {
    const plain = LayerAttentionExecConfig{
        .q_dim = 2048,
        .q_projection_dim = 4096,
        .kv_dim = 512,
        .sliding_window = 0,
        .is_causal = true,
        .query_gate = false,
    };
    try std.testing.expectEqual(@as(usize, 2048), expectedAttentionQProjectionDim(&plain));

    const gated = LayerAttentionExecConfig{
        .q_dim = 2048,
        .q_projection_dim = 4096,
        .kv_dim = 512,
        .sliding_window = 0,
        .is_causal = true,
        .query_gate = true,
    };
    try std.testing.expectEqual(@as(usize, 4096), expectedAttentionQProjectionDim(&gated));
}

test "bufferF32RowCount derives staged row count from buffer bytes" {
    const bytes = 2 * 1024 * @sizeOf(f32);
    const buffer = compute.cuda.Buffer{
        .pointer = 0,
        .size = bytes,
    };
    try std.testing.expectEqual(@as(usize, 2), try bufferF32RowCount(&buffer, 1024));
}

test "logicalF32RowSlice uses packed row offsets for tightly packed buffers" {
    const row_width = 8;
    const row_bytes = row_width * @sizeOf(f32);
    const buffer = compute.cuda.Buffer{
        .pointer = 4096,
        .size = 2 * row_bytes,
    };

    const row1 = try logicalF32RowSlice(&buffer, 2, 1, row_width);
    try std.testing.expectEqual(buffer.pointer + row_bytes, row1.pointer);
    try std.testing.expectEqual(@as(usize, row_bytes), row1.size);
}

test "logicalF32RowSlice uses widened row stride for staged slot buffers" {
    const logical_width = 8;
    const logical_row_bytes = logical_width * @sizeOf(f32);
    const widened_row_bytes = 16 * @sizeOf(f32);
    const buffer = compute.cuda.Buffer{
        .pointer = 8192,
        .size = 2 * widened_row_bytes,
    };

    const row1 = try logicalF32RowSlice(&buffer, 2, 1, logical_width);
    try std.testing.expectEqual(buffer.pointer + widened_row_bytes, row1.pointer);
    try std.testing.expectEqual(@as(usize, logical_row_bytes), row1.size);
}

test "attentionFallbackUsesCache uses decode mode only for single-row execution" {
    try std.testing.expect(engine_mixers.attentionFallbackUsesCache(1));
    try std.testing.expect(!engine_mixers.attentionFallbackUsesCache(2));
    try std.testing.expect(!engine_mixers.attentionFallbackUsesCache(15));
}

test "computeBatchedDecodeLogits routes pipeline2 decode per request through single-token path" {
    const Mock = struct {
        const BlockRuntimeMock = struct {
            blocks: [3]u8 = [_]u8{0} ** 3,
        };

        max_batch_size: usize = 4,
        topology_mode: enum { single, pipeline2 } = .pipeline2,
        block_runtime: BlockRuntimeMock = .{},
        vocab_size: usize = 5,
        slot_logits_storage: [20]f32 = [_]f32{0.0} ** 20,
        activate_calls: usize = 0,
        activated_slots: [8]usize = [_]usize{0} ** 8,
        compute_calls: usize = 0,
        recorded_tokens: [8]u32 = [_]u32{0} ** 8,
        recorded_positions: [8]usize = [_]usize{0} ** 8,
        recorded_slots: [8]usize = [_]usize{0} ** 8,
        recorded_layer_limits: [8]usize = [_]usize{0} ** 8,
        recorded_compute_logits: [8]bool = [_]bool{false} ** 8,
        recorded_download_logits: [8]bool = [_]bool{false} ** 8,
        recorded_use_preloaded_input: [8]bool = [_]bool{false} ** 8,

        pub fn activateKvSlot(self: *@This(), slot_index: usize) void {
            self.activated_slots[self.activate_calls] = slot_index;
            self.activate_calls += 1;
        }

        pub fn slotLogits(self: *@This(), slot_index: usize) []f32 {
            const offset = slot_index * self.vocab_size;
            return self.slot_logits_storage[offset .. offset + self.vocab_size];
        }

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_limit: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            trace_seq_len_u32: u32,
            trace_pos_offset: usize,
            hidden_override: ?[]const f32,
            deepstack_layer_features_opt: ?[]const []const f32,
            deepstack_feature_index_opt: ?usize,
            use_preloaded_input: bool,
        ) !void {
            _ = ensure_kv_capacity;
            _ = trace_seq_len_u32;
            _ = trace_pos_offset;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            const idx = self.compute_calls;
            self.compute_calls += 1;
            self.recorded_tokens[idx] = token;
            self.recorded_positions[idx] = position;
            self.recorded_slots[idx] = slot_index;
            self.recorded_layer_limits[idx] = layer_limit;
            self.recorded_compute_logits[idx] = compute_logits;
            self.recorded_download_logits[idx] = download_logits;
            self.recorded_use_preloaded_input[idx] = use_preloaded_input;
            if (logits_out_opt) |logits_out| {
                for (logits_out, 0..) |*logit, i| {
                    logit.* = @floatFromInt(i);
                }
            }
        }
    };

    var mock = Mock{};
    const tokens = [_]u32{ 11, 22, 33 };
    const slot_indices = [_]usize{ 2, 0, 3 };
    const positions = [_]usize{ 7, 8, 9 };

    try engine_forward.computeBatchedDecodeLogits(&mock, tokens[0..], slot_indices[0..], positions[0..]);

    try std.testing.expectEqual(tokens.len, mock.activate_calls);
    try std.testing.expectEqual(tokens.len, mock.compute_calls);
    for (0..tokens.len) |i| {
        try std.testing.expectEqual(slot_indices[i], mock.activated_slots[i]);
        try std.testing.expectEqual(tokens[i], mock.recorded_tokens[i]);
        try std.testing.expectEqual(positions[i], mock.recorded_positions[i]);
        try std.testing.expectEqual(slot_indices[i], mock.recorded_slots[i]);
        try std.testing.expectEqual(mock.block_runtime.blocks.len, mock.recorded_layer_limits[i]);
        try std.testing.expect(mock.recorded_compute_logits[i]);
        try std.testing.expect(mock.recorded_download_logits[i]);
        try std.testing.expect(!mock.recorded_use_preloaded_input[i]);
    }
}

test "computeBatchedDecodeLogits routes cpu_gpu decode per request through single-token path" {
    const Mock = struct {
        const BlockRuntimeMock = struct {
            blocks: [3]u8 = [_]u8{0} ** 3,
        };

        max_batch_size: usize = 4,
        topology_mode: enum { single, pipeline2, cpu_gpu } = .cpu_gpu,
        block_runtime: BlockRuntimeMock = .{},
        vocab_size: usize = 5,
        slot_logits_storage: [20]f32 = [_]f32{0.0} ** 20,
        activate_calls: usize = 0,
        activated_slots: [8]usize = [_]usize{0} ** 8,
        compute_calls: usize = 0,
        recorded_tokens: [8]u32 = [_]u32{0} ** 8,
        recorded_positions: [8]usize = [_]usize{0} ** 8,
        recorded_slots: [8]usize = [_]usize{0} ** 8,
        recorded_layer_limits: [8]usize = [_]usize{0} ** 8,
        recorded_compute_logits: [8]bool = [_]bool{false} ** 8,
        recorded_download_logits: [8]bool = [_]bool{false} ** 8,
        recorded_use_preloaded_input: [8]bool = [_]bool{false} ** 8,

        pub fn activateKvSlot(self: *@This(), slot_index: usize) void {
            self.activated_slots[self.activate_calls] = slot_index;
            self.activate_calls += 1;
        }

        pub fn slotLogits(self: *@This(), slot_index: usize) []f32 {
            const offset = slot_index * self.vocab_size;
            return self.slot_logits_storage[offset .. offset + self.vocab_size];
        }

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_limit: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            trace_seq_len_u32: u32,
            trace_pos_offset: usize,
            hidden_override: ?[]const f32,
            deepstack_layer_features_opt: ?[]const []const f32,
            deepstack_feature_index_opt: ?usize,
            use_preloaded_input: bool,
        ) !void {
            _ = ensure_kv_capacity;
            _ = trace_seq_len_u32;
            _ = trace_pos_offset;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            const idx = self.compute_calls;
            self.compute_calls += 1;
            self.recorded_tokens[idx] = token;
            self.recorded_positions[idx] = position;
            self.recorded_slots[idx] = slot_index;
            self.recorded_layer_limits[idx] = layer_limit;
            self.recorded_compute_logits[idx] = compute_logits;
            self.recorded_download_logits[idx] = download_logits;
            self.recorded_use_preloaded_input[idx] = use_preloaded_input;
            if (logits_out_opt) |logits_out| {
                for (logits_out, 0..) |*logit, i| {
                    logit.* = @floatFromInt(i);
                }
            }
        }
    };

    var mock = Mock{};
    const tokens = [_]u32{ 11, 22, 33 };
    const slot_indices = [_]usize{ 2, 0, 3 };
    const positions = [_]usize{ 7, 8, 9 };

    try engine_forward.computeBatchedDecodeLogits(&mock, tokens[0..], slot_indices[0..], positions[0..]);

    try std.testing.expectEqual(tokens.len, mock.activate_calls);
    try std.testing.expectEqual(tokens.len, mock.compute_calls);
    for (0..tokens.len) |i| {
        try std.testing.expectEqual(slot_indices[i], mock.activated_slots[i]);
        try std.testing.expectEqual(tokens[i], mock.recorded_tokens[i]);
        try std.testing.expectEqual(positions[i], mock.recorded_positions[i]);
        try std.testing.expectEqual(slot_indices[i], mock.recorded_slots[i]);
        try std.testing.expectEqual(mock.block_runtime.blocks.len, mock.recorded_layer_limits[i]);
        try std.testing.expect(mock.recorded_compute_logits[i]);
        try std.testing.expect(mock.recorded_download_logits[i]);
        try std.testing.expect(!mock.recorded_use_preloaded_input[i]);
    }
}

test "computeGpuPrototypePrefillLogitsWithLayerLimit routes pipeline2 prefill through staged token loop" {
    const Mock = struct {
        const BlockRuntimeMock = struct {
            blocks: [4]u8 = [_]u8{0} ** 4,
        };

        topology_mode: enum { single, pipeline2 } = .pipeline2,
        block_runtime: BlockRuntimeMock = .{},
        vocab_size: usize = 6,
        max_seq_len: usize = 32,
        compute_calls: usize = 0,
        recorded_download_logits: [16]bool = [_]bool{false} ** 16,
        recorded_trace_seq_lens: [16]u32 = [_]u32{0} ** 16,
        recorded_trace_positions: [16]usize = [_]usize{0} ** 16,
        recorded_logits_out_present: [16]bool = [_]bool{false} ** 16,

        pub fn slotIndexSupported(_: *const @This(), slot_index: usize) bool {
            return slot_index < 2;
        }

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_limit: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            trace_seq_len_u32: u32,
            trace_pos_offset: usize,
            hidden_override: ?[]const f32,
            deepstack_layer_features_opt: ?[]const []const f32,
            deepstack_feature_index_opt: ?usize,
            use_preloaded_input: bool,
        ) !void {
            _ = token;
            _ = position;
            _ = slot_index;
            _ = layer_limit;
            _ = compute_logits;
            _ = ensure_kv_capacity;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            _ = use_preloaded_input;
            const idx = self.compute_calls;
            self.compute_calls += 1;
            self.recorded_download_logits[idx] = download_logits;
            self.recorded_trace_seq_lens[idx] = trace_seq_len_u32;
            self.recorded_trace_positions[idx] = trace_pos_offset;
            self.recorded_logits_out_present[idx] = logits_out_opt != null;
            if (logits_out_opt) |logits_out| {
                for (logits_out, 0..) |*logit, i| {
                    logit.* = @floatFromInt(i);
                }
            }
        }
    };

    var mock = Mock{};
    const tokens = [_]u32{ 100, 101, 102, 103 };
    var logits_out: [6]f32 = undefined;

    try engine_forward.computeGpuPrototypePrefillLogitsWithLayerLimit(
        &mock,
        tokens[0..],
        0,
        logits_out[0..],
        mock.block_runtime.blocks.len,
    );

    try std.testing.expectEqual(tokens.len, mock.compute_calls);
    for (0..tokens.len) |i| {
        const is_last = i + 1 == tokens.len;
        try std.testing.expectEqual(is_last, mock.recorded_download_logits[i]);
        try std.testing.expectEqual(is_last, mock.recorded_logits_out_present[i]);
        try std.testing.expectEqual(@as(u32, @intCast(i + 1)), mock.recorded_trace_seq_lens[i]);
        try std.testing.expectEqual(i, mock.recorded_trace_positions[i]);
    }
}

test "computeGpuPrototypePrefillLogitsWithLayerLimit routes cpu_gpu prefill through staged token loop" {
    const Mock = struct {
        const BlockRuntimeMock = struct {
            blocks: [4]u8 = [_]u8{0} ** 4,
        };

        topology_mode: enum { single, pipeline2, cpu_gpu } = .cpu_gpu,
        block_runtime: BlockRuntimeMock = .{},
        vocab_size: usize = 6,
        max_seq_len: usize = 32,
        compute_calls: usize = 0,
        recorded_download_logits: [16]bool = [_]bool{false} ** 16,
        recorded_trace_seq_lens: [16]u32 = [_]u32{0} ** 16,
        recorded_trace_positions: [16]usize = [_]usize{0} ** 16,
        recorded_logits_out_present: [16]bool = [_]bool{false} ** 16,

        pub fn slotIndexSupported(_: *const @This(), slot_index: usize) bool {
            return slot_index < 2;
        }

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_limit: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            trace_seq_len_u32: u32,
            trace_pos_offset: usize,
            hidden_override: ?[]const f32,
            deepstack_layer_features_opt: ?[]const []const f32,
            deepstack_feature_index_opt: ?usize,
            use_preloaded_input: bool,
        ) !void {
            _ = token;
            _ = position;
            _ = slot_index;
            _ = layer_limit;
            _ = compute_logits;
            _ = ensure_kv_capacity;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            _ = use_preloaded_input;
            const idx = self.compute_calls;
            self.compute_calls += 1;
            self.recorded_download_logits[idx] = download_logits;
            self.recorded_trace_seq_lens[idx] = trace_seq_len_u32;
            self.recorded_trace_positions[idx] = trace_pos_offset;
            self.recorded_logits_out_present[idx] = logits_out_opt != null;
            if (logits_out_opt) |logits_out| {
                for (logits_out, 0..) |*logit, i| {
                    logit.* = @floatFromInt(i);
                }
            }
        }
    };

    var mock = Mock{};
    const tokens = [_]u32{ 100, 101, 102, 103 };
    var logits_out: [6]f32 = undefined;

    try engine_forward.computeGpuPrototypePrefillLogitsWithLayerLimit(
        &mock,
        tokens[0..],
        0,
        logits_out[0..],
        mock.block_runtime.blocks.len,
    );

    try std.testing.expectEqual(tokens.len, mock.compute_calls);
    for (0..tokens.len) |i| {
        const is_last = i + 1 == tokens.len;
        try std.testing.expectEqual(is_last, mock.recorded_download_logits[i]);
        try std.testing.expectEqual(is_last, mock.recorded_logits_out_present[i]);
        try std.testing.expectEqual(@as(u32, @intCast(i + 1)), mock.recorded_trace_seq_lens[i]);
        try std.testing.expectEqual(i, mock.recorded_trace_positions[i]);
    }
}

test "computeGpuPrototypeLogitsWithLayerLimit orchestrates pipeline2 stage0 transfer and stage1" {
    const TraceStep = enum(u8) {
        mirror,
        activate,
        stage0_compute,
        transfer,
        stage1_compute,
    };
    const SharedTrace = struct {
        steps: [16]TraceStep = undefined,
        step_count: usize = 0,
        mirror_calls: usize = 0,
        mirror_slot: usize = 0,
        activate_calls: usize = 0,
        activated_slot: usize = 0,
        transfer_calls: usize = 0,
        transfer_bytes: usize = 0,
        stage0_compute_calls: usize = 0,
        stage1_compute_calls: usize = 0,
        stage0_layer_limit: usize = 0,
        stage1_layer_limit: usize = 0,
        stage0_compute_logits: bool = true,
        stage1_compute_logits: bool = false,
        stage0_download_logits: bool = true,
        stage1_download_logits: bool = false,
        stage0_use_preloaded_input: bool = true,
        stage1_use_preloaded_input: bool = false,
        stage0_deepstack_len: usize = 0,
        stage1_deepstack_len: usize = 0,
        stage1_logits_out_present: bool = false,
        stage0_logits_out_present: bool = false,

        fn push(self: *@This(), step: TraceStep) void {
            self.steps[self.step_count] = step;
            self.step_count += 1;
        }
    };

    const Mock = struct {
        const BlockRuntimeMock = struct {
            blocks: []const u8 = &.{},
        };

        trace: *SharedTrace,
        is_stage1: bool = false,
        split_layer: usize = 2,
        d_model: usize = 8,
        state_descriptor_count: usize = 0,
        topology_mode: enum { single, pipeline2 } = .single,
        block_runtime: BlockRuntimeMock = .{},
        pipeline_backend1: ?*@This() = null,

        pub fn pipelineStage1(self: *@This()) ?*@This() {
            return self.pipeline_backend1;
        }

        pub fn mirrorSlotStateBlocksFrom(self: *@This(), source: *const @This(), slot_index: usize) !void {
            _ = source;
            self.trace.mirror_calls += 1;
            self.trace.mirror_slot = slot_index;
            self.trace.push(.mirror);
        }

        pub fn activateKvSlot(self: *@This(), slot_index: usize) void {
            self.trace.activate_calls += 1;
            self.trace.activated_slot = slot_index;
            self.trace.push(.activate);
        }

        pub fn transferPipelineActivation(self: *@This(), stage1: *@This(), byte_count: usize) !void {
            _ = stage1;
            self.trace.transfer_calls += 1;
            self.trace.transfer_bytes = byte_count;
            self.trace.push(.transfer);
        }

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_limit: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            trace_seq_len_u32: u32,
            trace_pos_offset: usize,
            hidden_override: ?[]const f32,
            deepstack_layer_features_opt: ?[]const []const f32,
            deepstack_feature_index_opt: ?usize,
            use_preloaded_input: bool,
        ) !void {
            _ = token;
            _ = position;
            _ = slot_index;
            _ = ensure_kv_capacity;
            _ = trace_seq_len_u32;
            _ = trace_pos_offset;
            _ = hidden_override;
            _ = deepstack_feature_index_opt;
            if (self.is_stage1) {
                self.trace.stage1_compute_calls += 1;
                self.trace.stage1_layer_limit = layer_limit;
                self.trace.stage1_compute_logits = compute_logits;
                self.trace.stage1_download_logits = download_logits;
                self.trace.stage1_use_preloaded_input = use_preloaded_input;
                self.trace.stage1_deepstack_len = if (deepstack_layer_features_opt) |features| features.len else 0;
                self.trace.stage1_logits_out_present = logits_out_opt != null;
                self.trace.push(.stage1_compute);
            } else {
                self.trace.stage0_compute_calls += 1;
                self.trace.stage0_layer_limit = layer_limit;
                self.trace.stage0_compute_logits = compute_logits;
                self.trace.stage0_download_logits = download_logits;
                self.trace.stage0_use_preloaded_input = use_preloaded_input;
                self.trace.stage0_deepstack_len = if (deepstack_layer_features_opt) |features| features.len else 0;
                self.trace.stage0_logits_out_present = logits_out_opt != null;
                self.trace.push(.stage0_compute);
            }
        }
    };

    var trace_state = SharedTrace{};
    var stage1 = Mock{
        .trace = &trace_state,
        .is_stage1 = true,
        .split_layer = 0,
        .d_model = 8,
        .state_descriptor_count = 1,
        .topology_mode = .single,
        .block_runtime = .{ .blocks = &[_]u8{ 0, 0 } },
        .pipeline_backend1 = null,
    };
    var stage0 = Mock{
        .trace = &trace_state,
        .is_stage1 = false,
        .split_layer = 2,
        .d_model = 8,
        .state_descriptor_count = 0,
        .topology_mode = .pipeline2,
        .block_runtime = .{ .blocks = &[_]u8{ 0, 0, 0, 0 } },
        .pipeline_backend1 = &stage1,
    };

    var logits: [7]f32 = undefined;
    const deepstack_features = [_][]const f32{
        &[_]f32{ 0.1, 0.2 },
        &[_]f32{ 0.3, 0.4 },
        &[_]f32{ 0.5, 0.6 },
        &[_]f32{ 0.7, 0.8 },
    };
    try engine_forward.computeGpuPrototypeLogitsWithLayerLimit(
        &stage0,
        123,
        9,
        1,
        logits[0..],
        stage0.block_runtime.blocks.len,
        true,
        true,
        true,
        10,
        9,
        null,
        deepstack_features[0..],
        null,
        false,
    );

    try std.testing.expectEqual(@as(usize, 5), trace_state.step_count);
    try std.testing.expectEqual(TraceStep.mirror, trace_state.steps[0]);
    try std.testing.expectEqual(TraceStep.activate, trace_state.steps[1]);
    try std.testing.expectEqual(TraceStep.stage0_compute, trace_state.steps[2]);
    try std.testing.expectEqual(TraceStep.transfer, trace_state.steps[3]);
    try std.testing.expectEqual(TraceStep.stage1_compute, trace_state.steps[4]);

    try std.testing.expectEqual(@as(usize, 1), trace_state.mirror_calls);
    try std.testing.expectEqual(@as(usize, 1), trace_state.mirror_slot);
    try std.testing.expectEqual(@as(usize, 1), trace_state.activate_calls);
    try std.testing.expectEqual(@as(usize, 1), trace_state.activated_slot);
    try std.testing.expectEqual(@as(usize, 1), trace_state.transfer_calls);
    try std.testing.expectEqual(stage0.d_model * @sizeOf(f32), trace_state.transfer_bytes);

    try std.testing.expectEqual(@as(usize, 1), trace_state.stage0_compute_calls);
    try std.testing.expectEqual(@as(usize, 1), trace_state.stage1_compute_calls);
    try std.testing.expectEqual(stage0.block_runtime.blocks.len, trace_state.stage0_layer_limit);
    try std.testing.expectEqual(stage1.block_runtime.blocks.len, trace_state.stage1_layer_limit);

    try std.testing.expect(!trace_state.stage0_compute_logits);
    try std.testing.expect(!trace_state.stage0_download_logits);
    try std.testing.expect(!trace_state.stage0_use_preloaded_input);
    try std.testing.expectEqual(deepstack_features.len, trace_state.stage0_deepstack_len);
    try std.testing.expect(!trace_state.stage0_logits_out_present);

    try std.testing.expect(trace_state.stage1_compute_logits);
    try std.testing.expect(trace_state.stage1_download_logits);
    try std.testing.expect(trace_state.stage1_use_preloaded_input);
    try std.testing.expectEqual(deepstack_features.len - stage0.split_layer, trace_state.stage1_deepstack_len);
    try std.testing.expect(trace_state.stage1_logits_out_present);
}

test "computeGpuPrototypeLogitsWithLayerLimit orchestrates cpu_gpu stage0 transfer and stage1" {
    const TraceStep = enum(u8) {
        stage0_compute,
        transfer,
        stage1_compute,
    };
    const SharedTrace = struct {
        steps: [16]TraceStep = undefined,
        step_count: usize = 0,
        stage0_compute_calls: usize = 0,
        stage0_token: u32 = 0,
        stage0_position: usize = 0,
        stage0_slot: usize = 0,
        stage0_layer_start: usize = 0,
        stage0_layer_end: usize = 0,
        stage0_compute_logits: bool = true,
        stage0_download_logits: bool = true,
        stage0_use_preloaded_input: bool = true,
        transfer_calls: usize = 0,
        transfer_slot: usize = 0,
        transfer_bytes: usize = 0,
        stage1_compute_calls: usize = 0,
        stage1_layer_limit: usize = 0,
        stage1_compute_logits: bool = false,
        stage1_download_logits: bool = false,
        stage1_use_preloaded_input: bool = false,
        stage1_logits_out_present: bool = false,

        fn push(self: *@This(), step: TraceStep) void {
            self.steps[self.step_count] = step;
            self.step_count += 1;
        }
    };

    const CpuStage0Mock = struct {
        trace: *SharedTrace,

        pub fn computePrototypeLogitsWithLayerRange(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_start: usize,
            layer_end: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            use_preloaded_input: bool,
        ) !void {
            _ = ensure_kv_capacity;
            self.trace.stage0_compute_calls += 1;
            self.trace.stage0_token = token;
            self.trace.stage0_position = position;
            self.trace.stage0_slot = slot_index;
            self.trace.stage0_layer_start = layer_start;
            self.trace.stage0_layer_end = layer_end;
            self.trace.stage0_compute_logits = compute_logits;
            self.trace.stage0_download_logits = download_logits;
            self.trace.stage0_use_preloaded_input = use_preloaded_input;
            try std.testing.expect(logits_out_opt == null);
            self.trace.push(.stage0_compute);
        }
    };

    const Mock = struct {
        const BlockRuntimeMock = struct {
            blocks: []const u8 = &.{},
        };

        trace: *SharedTrace,
        cpu_stage0: ?*CpuStage0Mock = null,
        split_layer: usize = 3,
        d_model: usize = 8,
        topology_mode: enum { single, pipeline2, cpu_gpu } = .cpu_gpu,
        block_runtime: BlockRuntimeMock = .{},

        pub fn pipelineCpuStage0(self: *@This()) ?*CpuStage0Mock {
            return self.cpu_stage0;
        }

        pub fn pipelineSplitLayer(self: *const @This()) usize {
            return self.split_layer;
        }

        pub fn transferPipelineActivationFromCpu(
            self: *@This(),
            src: *CpuStage0Mock,
            slot_index: usize,
            byte_count: usize,
        ) !void {
            _ = src;
            self.trace.transfer_calls += 1;
            self.trace.transfer_slot = slot_index;
            self.trace.transfer_bytes = byte_count;
            self.trace.push(.transfer);
        }

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_limit: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            trace_seq_len_u32: u32,
            trace_pos_offset: usize,
            hidden_override: ?[]const f32,
            deepstack_layer_features_opt: ?[]const []const f32,
            deepstack_feature_index_opt: ?usize,
            use_preloaded_input: bool,
        ) !void {
            _ = token;
            _ = position;
            _ = slot_index;
            _ = ensure_kv_capacity;
            _ = trace_seq_len_u32;
            _ = trace_pos_offset;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            self.trace.stage1_compute_calls += 1;
            self.trace.stage1_layer_limit = layer_limit;
            self.trace.stage1_compute_logits = compute_logits;
            self.trace.stage1_download_logits = download_logits;
            self.trace.stage1_use_preloaded_input = use_preloaded_input;
            self.trace.stage1_logits_out_present = logits_out_opt != null;
            self.trace.push(.stage1_compute);
        }
    };

    var trace_state = SharedTrace{};
    var cpu_stage0 = CpuStage0Mock{ .trace = &trace_state };
    var stage1 = Mock{
        .trace = &trace_state,
        .cpu_stage0 = &cpu_stage0,
        .split_layer = 3,
        .d_model = 8,
        .topology_mode = .cpu_gpu,
        .block_runtime = .{ .blocks = &[_]u8{ 0, 0 } },
    };
    var logits: [7]f32 = undefined;

    try engine_forward.computeGpuPrototypeLogitsWithLayerLimit(
        &stage1,
        123,
        9,
        1,
        logits[0..],
        stage1.block_runtime.blocks.len,
        true,
        true,
        true,
        10,
        9,
        null,
        null,
        null,
        false,
    );

    try std.testing.expectEqual(@as(usize, 3), trace_state.step_count);
    try std.testing.expectEqual(TraceStep.stage0_compute, trace_state.steps[0]);
    try std.testing.expectEqual(TraceStep.transfer, trace_state.steps[1]);
    try std.testing.expectEqual(TraceStep.stage1_compute, trace_state.steps[2]);

    try std.testing.expectEqual(@as(usize, 1), trace_state.stage0_compute_calls);
    try std.testing.expectEqual(@as(u32, 123), trace_state.stage0_token);
    try std.testing.expectEqual(@as(usize, 9), trace_state.stage0_position);
    try std.testing.expectEqual(@as(usize, 1), trace_state.stage0_slot);
    try std.testing.expectEqual(@as(usize, 0), trace_state.stage0_layer_start);
    try std.testing.expectEqual(@as(usize, 3), trace_state.stage0_layer_end);
    try std.testing.expect(!trace_state.stage0_compute_logits);
    try std.testing.expect(!trace_state.stage0_download_logits);
    try std.testing.expect(!trace_state.stage0_use_preloaded_input);

    try std.testing.expectEqual(@as(usize, 1), trace_state.transfer_calls);
    try std.testing.expectEqual(@as(usize, 1), trace_state.transfer_slot);
    try std.testing.expectEqual(stage1.d_model * @sizeOf(f32), trace_state.transfer_bytes);

    try std.testing.expectEqual(@as(usize, 1), trace_state.stage1_compute_calls);
    try std.testing.expectEqual(stage1.block_runtime.blocks.len, trace_state.stage1_layer_limit);
    try std.testing.expect(trace_state.stage1_compute_logits);
    try std.testing.expect(trace_state.stage1_download_logits);
    try std.testing.expect(trace_state.stage1_use_preloaded_input);
    try std.testing.expect(trace_state.stage1_logits_out_present);
}

test "computeGpuPrototypeLogitsWithLayerLimit cpu_gpu returns error when stage0 is missing" {
    const Mock = struct {
        const BlockRuntimeMock = struct {
            blocks: [2]u8 = [_]u8{0} ** 2,
        };
        topology_mode: enum { single, pipeline2, cpu_gpu } = .cpu_gpu,
        split_layer: usize = 1,
        d_model: usize = 8,
        block_runtime: BlockRuntimeMock = .{},

        pub fn pipelineCpuStage0(_: *@This()) ?*u8 {
            return null;
        }

        pub fn pipelineSplitLayer(self: *const @This()) usize {
            return self.split_layer;
        }

        pub fn transferPipelineActivationFromCpu(_: *@This(), _: *u8, _: usize, _: usize) !void {
            return;
        }

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            _: *@This(),
            _: u32,
            _: usize,
            _: usize,
            _: ?[]f32,
            _: usize,
            _: bool,
            _: bool,
            _: bool,
            _: u32,
            _: usize,
            _: ?[]const f32,
            _: ?[]const []const f32,
            _: ?usize,
            _: bool,
        ) !void {
            return;
        }
    };

    var mock = Mock{};
    var logits: [5]f32 = undefined;
    try std.testing.expectError(
        error.InvalidTopologyConfig,
        engine_forward.computeGpuPrototypeLogitsWithLayerLimit(
            &mock,
            1,
            0,
            0,
            logits[0..],
            mock.block_runtime.blocks.len,
            true,
            true,
            true,
            1,
            0,
            null,
            null,
            null,
            false,
        ),
    );
}

test "cpu_gpu decode parity matches single topology across slots and lifecycle cycles" {
    const slot_count: usize = 2;
    const d_model: usize = 4;
    const vocab: usize = 8;
    const split_layer: usize = 2;

    const CpuStage0Mock = struct {
        activations: [slot_count][d_model]f32 = [_][d_model]f32{[_]f32{0.0} ** d_model} ** slot_count,

        pub fn computePrototypeLogitsWithLayerRange(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_start: usize,
            layer_end: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            use_preloaded_input: bool,
        ) !void {
            _ = logits_out_opt;
            _ = compute_logits;
            _ = download_logits;
            _ = ensure_kv_capacity;
            _ = use_preloaded_input;
            if (slot_index >= slot_count) return error.InvalidArgument;
            if (layer_end < layer_start) return error.InvalidArgument;
            const base: f32 = @as(f32, @floatFromInt(token)) * 0.5 + @as(f32, @floatFromInt(position)) * 0.25;
            const layer_contrib: f32 = @floatFromInt(layer_end - layer_start);
            const value = base + layer_contrib;
            @memset(self.activations[slot_index][0..], value);
        }

        pub fn slotActivationBytes(self: *@This(), slot_index: usize) []const u8 {
            return std.mem.sliceAsBytes(self.activations[slot_index][0..]);
        }
    };

    const MockBackend = struct {
        const BlockRuntimeMock = struct {
            blocks: []const u8 = &.{},
        };

        topology_mode: enum { single, cpu_gpu } = .single,
        split_layer: usize = split_layer,
        d_model: usize = d_model,
        vocab_size: usize = vocab,
        max_seq_len: usize = 256,
        max_batch_size: usize = slot_count,
        block_runtime: BlockRuntimeMock = .{},
        pipeline_host_staging: ?[]align(64) u8 = null,
        cpu_stage0: ?*CpuStage0Mock = null,
        preloaded: [slot_count][d_model]f32 = [_][d_model]f32{[_]f32{0.0} ** d_model} ** slot_count,
        slot_in_use: [slot_count]bool = [_]bool{true} ** slot_count,
        slot_positions: [slot_count]usize = [_]usize{0} ** slot_count,

        pub fn pipelineCpuStage0(self: *@This()) ?*CpuStage0Mock {
            return self.cpu_stage0;
        }

        pub fn pipelineSplitLayer(self: *const @This()) usize {
            return self.split_layer;
        }

        pub fn pipelineActivationByteCount(self: *const @This()) !usize {
            return std.math.mul(usize, self.d_model, @sizeOf(f32)) catch error.InvalidArgument;
        }

        pub fn transferPipelineActivationFromCpu(
            self: *@This(),
            src: *CpuStage0Mock,
            slot_index: usize,
            byte_count: usize,
        ) !void {
            const src_bytes = src.slotActivationBytes(slot_index);
            const dst_bytes = std.mem.sliceAsBytes(self.preloaded[slot_index][0..]);
            if (byte_count > src_bytes.len or byte_count > dst_bytes.len) return error.InvalidArgument;
            @memcpy(dst_bytes[0..byte_count], src_bytes[0..byte_count]);
        }

        pub fn uploadPipelineActivationFromHost(
            self: *@This(),
            slot_index: usize,
            host_buf: []const u8,
            byte_count: usize,
        ) !void {
            const dst_bytes = std.mem.sliceAsBytes(self.preloaded[slot_index][0..]);
            if (byte_count > host_buf.len or byte_count > dst_bytes.len) return error.InvalidArgument;
            @memcpy(dst_bytes[0..byte_count], host_buf[0..byte_count]);
        }

        pub fn slotIndexSupported(_: *const @This(), slot_index: usize) bool {
            return slot_index < slot_count;
        }

        pub fn activateKvSlot(_: *@This(), _: usize) void {}

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_limit: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            trace_seq_len_u32: u32,
            trace_pos_offset: usize,
            hidden_override: ?[]const f32,
            deepstack_layer_features_opt: ?[]const []const f32,
            deepstack_feature_index_opt: ?usize,
            use_preloaded_input: bool,
        ) !void {
            _ = ensure_kv_capacity;
            _ = trace_seq_len_u32;
            _ = trace_pos_offset;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            if (slot_index >= slot_count) return error.InvalidArgument;
            var hidden: f32 = 0.0;
            if (use_preloaded_input) {
                hidden = self.preloaded[slot_index][0] + @as(f32, @floatFromInt(layer_limit));
            } else {
                const base: f32 = @as(f32, @floatFromInt(token)) * 0.5 + @as(f32, @floatFromInt(position)) * 0.25;
                hidden = base + @as(f32, @floatFromInt(layer_limit));
                @memset(self.preloaded[slot_index][0..], hidden);
            }
            if (!compute_logits) return;
            _ = download_logits;
            if (logits_out_opt) |logits_out| {
                for (logits_out, 0..) |*v, i| {
                    v.* = hidden + @as(f32, @floatFromInt(i)) * 0.01;
                }
            }
        }
    };

    var cycle: usize = 0;
    while (cycle < 8) : (cycle += 1) {
        var cpu_stage = CpuStage0Mock{};
        var staging: [d_model * @sizeOf(f32)]u8 align(64) = undefined;
        var single = MockBackend{
            .topology_mode = .single,
            .split_layer = split_layer,
            .block_runtime = .{ .blocks = &[_]u8{ 0, 0, 0, 0, 0, 0 } },
        };
        var split = MockBackend{
            .topology_mode = .cpu_gpu,
            .split_layer = split_layer,
            .block_runtime = .{ .blocks = &[_]u8{ 0, 0, 0, 0 } },
            .pipeline_host_staging = staging[0..],
            .cpu_stage0 = &cpu_stage,
        };

        var positions: [slot_count]usize = [_]usize{0} ** slot_count;
        for (0..6) |step| {
            for (0..slot_count) |slot_index| {
                const token: u32 = @intCast(11 + step * 3 + slot_index);
                var logits_single: [vocab]f32 = undefined;
                var logits_split: [vocab]f32 = undefined;

                try engine_forward.computeGpuPrototypeLogitsWithLayerLimit(
                    &single,
                    token,
                    positions[slot_index],
                    slot_index,
                    logits_single[0..],
                    single.block_runtime.blocks.len,
                    true,
                    true,
                    true,
                    @intCast(positions[slot_index] + 1),
                    positions[slot_index],
                    null,
                    null,
                    null,
                    false,
                );
                try engine_forward.computeGpuPrototypeLogitsWithLayerLimit(
                    &split,
                    token,
                    positions[slot_index],
                    slot_index,
                    logits_split[0..],
                    split.block_runtime.blocks.len,
                    true,
                    true,
                    true,
                    @intCast(positions[slot_index] + 1),
                    positions[slot_index],
                    null,
                    null,
                    null,
                    false,
                );
                for (logits_single, logits_split) |lhs, rhs| {
                    try std.testing.expectApproxEqAbs(lhs, rhs, 1.0e-6);
                }
                positions[slot_index] += 1;
            }
        }
    }
}

test "cpu_gpu prefill parity matches single topology across repeated windows" {
    const d_model: usize = 4;
    const vocab: usize = 8;
    const split_layer: usize = 2;

    const CpuStage0Mock = struct {
        activation: [d_model]f32 = [_]f32{0.0} ** d_model,

        pub fn computePrototypeLogitsWithLayerRange(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_start: usize,
            layer_end: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            use_preloaded_input: bool,
        ) !void {
            _ = slot_index;
            _ = logits_out_opt;
            _ = compute_logits;
            _ = download_logits;
            _ = ensure_kv_capacity;
            _ = use_preloaded_input;
            if (layer_end < layer_start) return error.InvalidArgument;
            const base: f32 = @as(f32, @floatFromInt(token)) * 0.5 + @as(f32, @floatFromInt(position)) * 0.25;
            const layer_contrib: f32 = @floatFromInt(layer_end - layer_start);
            @memset(self.activation[0..], base + layer_contrib);
        }

        pub fn slotActivationBytes(self: *@This(), _: usize) []const u8 {
            return std.mem.sliceAsBytes(self.activation[0..]);
        }
    };

    const MockBackend = struct {
        const BlockRuntimeMock = struct {
            blocks: []const u8 = &.{},
        };

        topology_mode: enum { single, cpu_gpu } = .single,
        split_layer: usize = split_layer,
        d_model: usize = d_model,
        vocab_size: usize = vocab,
        max_seq_len: usize = 256,
        block_runtime: BlockRuntimeMock = .{},
        cpu_stage0: ?*CpuStage0Mock = null,
        pipeline_host_staging: ?[]align(64) u8 = null,
        preloaded: [d_model]f32 = [_]f32{0.0} ** d_model,

        pub fn slotIndexSupported(_: *const @This(), slot_index: usize) bool {
            return slot_index == 0;
        }

        pub fn pipelineCpuStage0(self: *@This()) ?*CpuStage0Mock {
            return self.cpu_stage0;
        }

        pub fn pipelineSplitLayer(self: *const @This()) usize {
            return self.split_layer;
        }

        pub fn pipelineActivationByteCount(self: *const @This()) !usize {
            return std.math.mul(usize, self.d_model, @sizeOf(f32)) catch error.InvalidArgument;
        }

        pub fn activateKvSlot(_: *@This(), _: usize) void {}

        pub fn transferPipelineActivationFromCpu(
            self: *@This(),
            src: *CpuStage0Mock,
            _: usize,
            byte_count: usize,
        ) !void {
            const src_bytes = src.slotActivationBytes(0);
            const dst_bytes = std.mem.sliceAsBytes(self.preloaded[0..]);
            if (byte_count > src_bytes.len or byte_count > dst_bytes.len) return error.InvalidArgument;
            @memcpy(dst_bytes[0..byte_count], src_bytes[0..byte_count]);
        }

        pub fn uploadPipelineActivationFromHost(
            self: *@This(),
            _: usize,
            host_buf: []const u8,
            byte_count: usize,
        ) !void {
            const dst_bytes = std.mem.sliceAsBytes(self.preloaded[0..]);
            if (byte_count > host_buf.len or byte_count > dst_bytes.len) return error.InvalidArgument;
            @memcpy(dst_bytes[0..byte_count], host_buf[0..byte_count]);
        }

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_limit: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            trace_seq_len_u32: u32,
            trace_pos_offset: usize,
            hidden_override: ?[]const f32,
            deepstack_layer_features_opt: ?[]const []const f32,
            deepstack_feature_index_opt: ?usize,
            use_preloaded_input: bool,
        ) !void {
            _ = slot_index;
            _ = ensure_kv_capacity;
            _ = trace_seq_len_u32;
            _ = trace_pos_offset;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            var hidden: f32 = 0.0;
            if (use_preloaded_input) {
                hidden = self.preloaded[0] + @as(f32, @floatFromInt(layer_limit));
            } else {
                const base: f32 = @as(f32, @floatFromInt(token)) * 0.5 + @as(f32, @floatFromInt(position)) * 0.25;
                hidden = base + @as(f32, @floatFromInt(layer_limit));
                @memset(self.preloaded[0..], hidden);
            }
            if (!compute_logits) return;
            _ = download_logits;
            if (logits_out_opt) |logits_out| {
                for (logits_out, 0..) |*v, i| {
                    v.* = hidden + @as(f32, @floatFromInt(i)) * 0.01;
                }
            }
        }
    };

    const windows = [_][]const u32{
        &[_]u32{ 3, 5, 7, 11 },
        &[_]u32{ 13, 17 },
        &[_]u32{ 19, 23, 29 },
    };

    var cycle: usize = 0;
    while (cycle < 6) : (cycle += 1) {
        var cpu_stage = CpuStage0Mock{};
        var staging: [d_model * @sizeOf(f32)]u8 align(64) = undefined;
        var split = MockBackend{
            .topology_mode = .cpu_gpu,
            .split_layer = split_layer,
            .block_runtime = .{ .blocks = &[_]u8{ 0, 0, 0, 0 } },
            .cpu_stage0 = &cpu_stage,
            .pipeline_host_staging = staging[0..],
        };
        for (windows) |window_tokens| {
            var logits_split: [vocab]f32 = undefined;
            try engine_forward.computeGpuPrototypePrefillLogitsWithLayerLimit(
                &split,
                window_tokens,
                0,
                logits_split[0..],
                split.block_runtime.blocks.len,
            );
            const last_index = window_tokens.len - 1;
            const base: f32 = @as(f32, @floatFromInt(window_tokens[last_index])) * 0.5 + @as(f32, @floatFromInt(last_index)) * 0.25;
            const expected_hidden = base + 6.0;
            for (logits_split, 0..) |actual, i| {
                const expected = expected_hidden + @as(f32, @floatFromInt(i)) * 0.01;
                try std.testing.expectApproxEqAbs(expected, actual, 1.0e-6);
            }
        }
    }
}

test "cpu_gpu prefill parity remains deterministic across slots and lifecycle cycles" {
    const slot_count: usize = 2;
    const d_model: usize = 4;
    const vocab: usize = 8;
    const split_layer: usize = 2;

    const CpuStage0Mock = struct {
        activations: [slot_count][d_model]f32 = [_][d_model]f32{[_]f32{0.0} ** d_model} ** slot_count,

        pub fn computePrototypeLogitsWithLayerRange(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_start: usize,
            layer_end: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            use_preloaded_input: bool,
        ) !void {
            _ = logits_out_opt;
            _ = compute_logits;
            _ = download_logits;
            _ = ensure_kv_capacity;
            _ = use_preloaded_input;
            if (slot_index >= slot_count) return error.InvalidArgument;
            if (layer_end < layer_start) return error.InvalidArgument;
            const base: f32 = @as(f32, @floatFromInt(token)) * 0.5 + @as(f32, @floatFromInt(position)) * 0.25;
            const layer_contrib: f32 = @floatFromInt(layer_end - layer_start);
            @memset(self.activations[slot_index][0..], base + layer_contrib);
        }

        pub fn slotActivationBytes(self: *@This(), slot_index: usize) []const u8 {
            if (slot_index >= slot_count) @panic("slot_index out of range");
            return std.mem.sliceAsBytes(self.activations[slot_index][0..]);
        }
    };

    const MockBackend = struct {
        const BlockRuntimeMock = struct {
            blocks: []const u8 = &.{},
        };

        topology_mode: enum { single, cpu_gpu } = .single,
        split_layer: usize = split_layer,
        d_model: usize = d_model,
        vocab_size: usize = vocab,
        max_seq_len: usize = 256,
        block_runtime: BlockRuntimeMock = .{},
        cpu_stage0: ?*CpuStage0Mock = null,
        pipeline_host_staging: ?[]align(64) u8 = null,
        preloaded: [slot_count][d_model]f32 = [_][d_model]f32{[_]f32{0.0} ** d_model} ** slot_count,

        pub fn slotIndexSupported(_: *const @This(), slot_index: usize) bool {
            return slot_index < slot_count;
        }

        pub fn pipelineCpuStage0(self: *@This()) ?*CpuStage0Mock {
            return self.cpu_stage0;
        }

        pub fn pipelineSplitLayer(self: *const @This()) usize {
            return self.split_layer;
        }

        pub fn pipelineActivationByteCount(self: *const @This()) !usize {
            return std.math.mul(usize, self.d_model, @sizeOf(f32)) catch error.InvalidArgument;
        }

        pub fn activateKvSlot(_: *@This(), _: usize) void {}

        pub fn transferPipelineActivationFromCpu(
            self: *@This(),
            src: *CpuStage0Mock,
            slot_index: usize,
            byte_count: usize,
        ) !void {
            const src_bytes = src.slotActivationBytes(slot_index);
            const dst_bytes = std.mem.sliceAsBytes(self.preloaded[slot_index][0..]);
            if (byte_count > src_bytes.len or byte_count > dst_bytes.len) return error.InvalidArgument;
            @memcpy(dst_bytes[0..byte_count], src_bytes[0..byte_count]);
        }

        pub fn uploadPipelineActivationFromHost(
            self: *@This(),
            slot_index: usize,
            host_buf: []const u8,
            byte_count: usize,
        ) !void {
            const dst_bytes = std.mem.sliceAsBytes(self.preloaded[slot_index][0..]);
            if (byte_count > host_buf.len or byte_count > dst_bytes.len) return error.InvalidArgument;
            @memcpy(dst_bytes[0..byte_count], host_buf[0..byte_count]);
        }

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_limit: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            trace_seq_len_u32: u32,
            trace_pos_offset: usize,
            hidden_override: ?[]const f32,
            deepstack_layer_features_opt: ?[]const []const f32,
            deepstack_feature_index_opt: ?usize,
            use_preloaded_input: bool,
        ) !void {
            _ = ensure_kv_capacity;
            _ = trace_seq_len_u32;
            _ = trace_pos_offset;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            if (slot_index >= slot_count) return error.InvalidArgument;
            var hidden: f32 = 0.0;
            if (use_preloaded_input) {
                hidden = self.preloaded[slot_index][0] + @as(f32, @floatFromInt(layer_limit));
            } else {
                const base: f32 = @as(f32, @floatFromInt(token)) * 0.5 + @as(f32, @floatFromInt(position)) * 0.25;
                hidden = base + @as(f32, @floatFromInt(layer_limit));
                @memset(self.preloaded[slot_index][0..], hidden);
            }
            if (!compute_logits) return;
            _ = download_logits;
            if (logits_out_opt) |logits_out| {
                for (logits_out, 0..) |*v, i| {
                    v.* = hidden + @as(f32, @floatFromInt(i)) * 0.01;
                }
            }
        }
    };

    const slot_windows = [_][]const []const u32{
        &[_][]const u32{
            &[_]u32{ 3, 5, 7, 11 },
            &[_]u32{ 13, 17 },
            &[_]u32{ 19, 23, 29 },
        },
        &[_][]const u32{
            &[_]u32{ 2, 4, 6 },
            &[_]u32{ 8, 10, 12, 14 },
            &[_]u32{16},
        },
    };

    var cycle: usize = 0;
    while (cycle < 6) : (cycle += 1) {
        var cpu_stage = CpuStage0Mock{};
        var staging: [d_model * @sizeOf(f32)]u8 align(64) = undefined;
        var split = MockBackend{
            .topology_mode = .cpu_gpu,
            .split_layer = split_layer,
            .block_runtime = .{ .blocks = &[_]u8{ 0, 0, 0, 0 } },
            .cpu_stage0 = &cpu_stage,
            .pipeline_host_staging = staging[0..],
        };

        for (slot_windows, 0..) |windows, slot_index| {
            for (windows) |window_tokens| {
                var logits_split: [vocab]f32 = undefined;
                try engine_forward.computeGpuPrototypePrefillLogitsWithLayerLimit(
                    &split,
                    window_tokens,
                    slot_index,
                    logits_split[0..],
                    split.block_runtime.blocks.len,
                );
                const last_index = window_tokens.len - 1;
                const base: f32 = @as(f32, @floatFromInt(window_tokens[last_index])) * 0.5 + @as(f32, @floatFromInt(last_index)) * 0.25;
                const expected_hidden = base + 6.0;
                for (logits_split, 0..) |actual, i| {
                    const expected = expected_hidden + @as(f32, @floatFromInt(i)) * 0.01;
                    try std.testing.expectApproxEqAbs(expected, actual, 1.0e-6);
                }
            }
        }
    }
}

test "computeGpuPrototypeLogitsWithLayerLimit pipeline2 returns error when stage1 is missing" {
    const Mock = struct {
        const BlockRuntimeMock = struct {
            blocks: [4]u8 = [_]u8{0} ** 4,
        };
        topology_mode: enum { single, pipeline2 } = .pipeline2,
        split_layer: usize = 2,
        d_model: usize = 8,
        state_descriptor_count: usize = 0,
        block_runtime: BlockRuntimeMock = .{},

        pub fn pipelineStage1(_: *@This()) ?*@This() {
            return null;
        }

        pub fn mirrorSlotStateBlocksFrom(_: *@This(), _: *const @This(), _: usize) !void {
            return;
        }

        pub fn activateKvSlot(_: *@This(), _: usize) void {}

        pub fn transferPipelineActivation(_: *@This(), _: *@This(), _: usize) !void {
            return;
        }

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            _: *@This(),
            _: u32,
            _: usize,
            _: usize,
            _: ?[]f32,
            _: usize,
            _: bool,
            _: bool,
            _: bool,
            _: u32,
            _: usize,
            _: ?[]const f32,
            _: ?[]const []const f32,
            _: ?usize,
            _: bool,
        ) !void {
            return;
        }
    };

    var mock = Mock{};
    var logits: [5]f32 = undefined;
    try std.testing.expectError(
        error.InvalidTopologyConfig,
        engine_forward.computeGpuPrototypeLogitsWithLayerLimit(
            &mock,
            1,
            0,
            0,
            logits[0..],
            mock.block_runtime.blocks.len,
            true,
            true,
            true,
            1,
            0,
            null,
            null,
            null,
            false,
        ),
    );
}

test "computeBatchedDecodeLogits routes cpu_gpu_gpu decode per request through single-token path" {
    const Mock = struct {
        const BlockRuntimeMock = struct {
            blocks: [3]u8 = [_]u8{0} ** 3,
        };

        max_batch_size: usize = 4,
        topology_mode: enum { single, pipeline2, cpu_gpu, cpu_gpu_gpu } = .cpu_gpu_gpu,
        block_runtime: BlockRuntimeMock = .{},
        vocab_size: usize = 5,
        slot_logits_storage: [20]f32 = [_]f32{0.0} ** 20,
        activate_calls: usize = 0,
        recorded_slots: [8]usize = [_]usize{0} ** 8,
        compute_calls: usize = 0,
        recorded_tokens: [8]u32 = [_]u32{0} ** 8,
        recorded_positions: [8]usize = [_]usize{0} ** 8,
        recorded_layer_limits: [8]usize = [_]usize{0} ** 8,
        recorded_compute_logits: [8]bool = [_]bool{false} ** 8,
        recorded_download_logits: [8]bool = [_]bool{false} ** 8,

        pub fn activateKvSlot(self: *@This(), slot_index: usize) void {
            self.recorded_slots[self.activate_calls] = slot_index;
            self.activate_calls += 1;
        }

        pub fn slotLogits(self: *@This(), slot_index: usize) []f32 {
            const offset = slot_index * self.vocab_size;
            return self.slot_logits_storage[offset .. offset + self.vocab_size];
        }

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_limit: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            trace_seq_len_u32: u32,
            trace_pos_offset: usize,
            hidden_override: ?[]const f32,
            deepstack_layer_features_opt: ?[]const []const f32,
            deepstack_feature_index_opt: ?usize,
            use_preloaded_input: bool,
        ) !void {
            _ = slot_index;
            _ = ensure_kv_capacity;
            _ = trace_seq_len_u32;
            _ = trace_pos_offset;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            _ = use_preloaded_input;
            const idx = self.compute_calls;
            self.compute_calls += 1;
            self.recorded_tokens[idx] = token;
            self.recorded_positions[idx] = position;
            self.recorded_layer_limits[idx] = layer_limit;
            self.recorded_compute_logits[idx] = compute_logits;
            self.recorded_download_logits[idx] = download_logits;
            if (logits_out_opt) |logits_out| {
                for (logits_out, 0..) |*logit, i| {
                    logit.* = @floatFromInt(i);
                }
            }
        }
    };

    var mock = Mock{};
    const tokens = [_]u32{ 11, 22, 33 };
    const slot_indices = [_]usize{ 2, 0, 3 };
    const positions = [_]usize{ 7, 8, 9 };

    try engine_forward.computeBatchedDecodeLogits(&mock, tokens[0..], slot_indices[0..], positions[0..]);

    try std.testing.expectEqual(tokens.len, mock.activate_calls);
    try std.testing.expectEqual(tokens.len, mock.compute_calls);
    for (0..tokens.len) |i| {
        try std.testing.expectEqual(slot_indices[i], mock.recorded_slots[i]);
        try std.testing.expectEqual(tokens[i], mock.recorded_tokens[i]);
        try std.testing.expectEqual(positions[i], mock.recorded_positions[i]);
        try std.testing.expectEqual(mock.block_runtime.blocks.len, mock.recorded_layer_limits[i]);
        try std.testing.expect(mock.recorded_compute_logits[i]);
        try std.testing.expect(mock.recorded_download_logits[i]);
    }
}

test "computeGpuPrototypeLogitsWithLayerLimit orchestrates cpu_gpu_gpu stage chain" {
    const TraceStep = enum(u8) {
        mirror,
        stage1_activate,
        stage2_activate,
        stage0_compute,
        transfer_01,
        stage1_compute,
        transfer_12,
        stage2_compute,
    };
    const SharedTrace = struct {
        steps: [32]TraceStep = undefined,
        step_count: usize = 0,
        transfer01_bytes: usize = 0,
        transfer12_bytes: usize = 0,
        stage1_layer_limit: usize = 0,
        stage2_layer_limit: usize = 0,
        stage1_use_preloaded_input: bool = false,
        stage2_use_preloaded_input: bool = false,
        stage2_logits_present: bool = false,

        fn push(self: *@This(), step: TraceStep) void {
            self.steps[self.step_count] = step;
            self.step_count += 1;
        }
    };

    const CpuStage0Mock = struct {
        trace: *SharedTrace,
        activation: [8]f32 = [_]f32{0.0} ** 8,

        pub fn computePrototypeLogitsWithLayerRange(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_start: usize,
            layer_end: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            use_preloaded_input: bool,
        ) !void {
            _ = token;
            _ = position;
            _ = slot_index;
            _ = layer_start;
            _ = layer_end;
            _ = compute_logits;
            _ = download_logits;
            _ = ensure_kv_capacity;
            _ = use_preloaded_input;
            _ = logits_out_opt;
            self.trace.push(.stage0_compute);
            @memset(self.activation[0..], 2.0);
        }

        pub fn slotActivationBytes(self: *@This(), _: usize) []const u8 {
            return std.mem.sliceAsBytes(self.activation[0..]);
        }
    };

    const Stage1Mock = struct {
        const BlockRuntimeMock = struct {
            blocks: []const u8 = &.{},
        };

        trace: *SharedTrace,
        state_descriptor_count: usize = 0,
        block_runtime: BlockRuntimeMock = .{},

        pub fn mirrorSlotStateBlocksFrom(self: *@This(), _: anytype, _: usize) !void {
            self.trace.push(.mirror);
        }

        pub fn activateKvSlot(self: *@This(), _: usize) void {
            self.trace.push(.stage1_activate);
        }

        pub fn transferPipelineActivationFromCpu(
            self: *@This(),
            _: *CpuStage0Mock,
            _: usize,
            byte_count: usize,
        ) !void {
            self.trace.transfer01_bytes = byte_count;
            self.trace.push(.transfer_01);
        }

        pub fn transferPipelineActivation(self: *@This(), _: anytype, byte_count: usize) !void {
            self.trace.transfer12_bytes = byte_count;
            self.trace.push(.transfer_12);
        }

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_limit: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            trace_seq_len_u32: u32,
            trace_pos_offset: usize,
            hidden_override: ?[]const f32,
            deepstack_layer_features_opt: ?[]const []const f32,
            deepstack_feature_index_opt: ?usize,
            use_preloaded_input: bool,
        ) !void {
            _ = token;
            _ = position;
            _ = slot_index;
            _ = logits_out_opt;
            _ = compute_logits;
            _ = download_logits;
            _ = ensure_kv_capacity;
            _ = trace_seq_len_u32;
            _ = trace_pos_offset;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            self.trace.stage1_layer_limit = layer_limit;
            self.trace.stage1_use_preloaded_input = use_preloaded_input;
            self.trace.push(.stage1_compute);
        }
    };

    const Stage2Mock = struct {
        const BlockRuntimeMock = struct {
            blocks: []const u8 = &.{},
        };

        trace: *SharedTrace,
        cpu_stage0: ?*CpuStage0Mock = null,
        stage1: ?*Stage1Mock = null,
        split_layer: usize = 2,
        split_layer_stage2: usize = 4,
        d_model: usize = 8,
        topology_mode: enum { single, pipeline2, cpu_gpu, cpu_gpu_gpu } = .cpu_gpu_gpu,
        block_runtime: BlockRuntimeMock = .{},
        pipeline_host_staging: ?[]align(64) u8 = null,
        pipeline_host_staging_stage12: ?[]align(64) u8 = null,

        pub fn pipelineCpuStage0(self: *@This()) ?*CpuStage0Mock {
            return self.cpu_stage0;
        }

        pub fn pipelineStage1(self: *@This()) ?*Stage1Mock {
            return self.stage1;
        }

        pub fn pipelineSplitLayer(self: *const @This()) usize {
            return self.split_layer;
        }

        pub fn pipelineSplitLayerStage2(self: *const @This()) usize {
            return self.split_layer_stage2;
        }

        pub fn activateKvSlot(self: *@This(), _: usize) void {
            self.trace.push(.stage2_activate);
        }

        pub fn transferPipelineActivationFromCpu(
            _: *@This(),
            _: *CpuStage0Mock,
            _: usize,
            _: usize,
        ) !void {
            return;
        }

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_limit: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            trace_seq_len_u32: u32,
            trace_pos_offset: usize,
            hidden_override: ?[]const f32,
            deepstack_layer_features_opt: ?[]const []const f32,
            deepstack_feature_index_opt: ?usize,
            use_preloaded_input: bool,
        ) !void {
            _ = token;
            _ = position;
            _ = slot_index;
            _ = compute_logits;
            _ = download_logits;
            _ = ensure_kv_capacity;
            _ = trace_seq_len_u32;
            _ = trace_pos_offset;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            self.trace.stage2_layer_limit = layer_limit;
            self.trace.stage2_use_preloaded_input = use_preloaded_input;
            self.trace.stage2_logits_present = logits_out_opt != null;
            self.trace.push(.stage2_compute);
        }
    };

    var trace_state = SharedTrace{};
    var cpu_stage0 = CpuStage0Mock{ .trace = &trace_state };
    var stage1 = Stage1Mock{
        .trace = &trace_state,
        .state_descriptor_count = 1,
        .block_runtime = .{ .blocks = &[_]u8{ 0, 0 } },
    };
    var stage2 = Stage2Mock{
        .trace = &trace_state,
        .cpu_stage0 = &cpu_stage0,
        .stage1 = &stage1,
        .split_layer = 2,
        .split_layer_stage2 = 4,
        .d_model = 8,
        .topology_mode = .cpu_gpu_gpu,
        .block_runtime = .{ .blocks = &[_]u8{ 0, 0 } },
    };
    var logits: [7]f32 = undefined;

    try engine_forward.computeGpuPrototypeLogitsWithLayerLimit(
        &stage2,
        123,
        9,
        1,
        logits[0..],
        stage2.block_runtime.blocks.len,
        true,
        true,
        true,
        10,
        9,
        null,
        null,
        null,
        false,
    );

    try std.testing.expectEqual(@as(usize, 8), trace_state.step_count);
    try std.testing.expectEqual(TraceStep.mirror, trace_state.steps[0]);
    try std.testing.expectEqual(TraceStep.stage1_activate, trace_state.steps[1]);
    try std.testing.expectEqual(TraceStep.stage2_activate, trace_state.steps[2]);
    try std.testing.expectEqual(TraceStep.stage0_compute, trace_state.steps[3]);
    try std.testing.expectEqual(TraceStep.transfer_01, trace_state.steps[4]);
    try std.testing.expectEqual(TraceStep.stage1_compute, trace_state.steps[5]);
    try std.testing.expectEqual(TraceStep.transfer_12, trace_state.steps[6]);
    try std.testing.expectEqual(TraceStep.stage2_compute, trace_state.steps[7]);
    try std.testing.expectEqual(stage2.d_model * @sizeOf(f32), trace_state.transfer01_bytes);
    try std.testing.expectEqual(stage2.d_model * @sizeOf(f32), trace_state.transfer12_bytes);
    try std.testing.expectEqual(@as(usize, 2), trace_state.stage1_layer_limit);
    try std.testing.expectEqual(@as(usize, 2), trace_state.stage2_layer_limit);
    try std.testing.expect(trace_state.stage1_use_preloaded_input);
    try std.testing.expect(trace_state.stage2_use_preloaded_input);
    try std.testing.expect(trace_state.stage2_logits_present);
}

test "computeGpuPrototypePrefillLogitsWithLayerLimit routes cpu_gpu_gpu prefill through staged token loop" {
    const Mock = struct {
        const BlockRuntimeMock = struct {
            blocks: [4]u8 = [_]u8{0} ** 4,
        };

        topology_mode: enum { single, pipeline2, cpu_gpu, cpu_gpu_gpu } = .cpu_gpu_gpu,
        block_runtime: BlockRuntimeMock = .{},
        vocab_size: usize = 6,
        max_seq_len: usize = 32,
        compute_calls: usize = 0,
        recorded_download_logits: [16]bool = [_]bool{false} ** 16,
        recorded_trace_seq_lens: [16]u32 = [_]u32{0} ** 16,
        recorded_trace_positions: [16]usize = [_]usize{0} ** 16,
        recorded_logits_out_present: [16]bool = [_]bool{false} ** 16,

        pub fn slotIndexSupported(_: *const @This(), slot_index: usize) bool {
            return slot_index < 2;
        }

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_limit: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            trace_seq_len_u32: u32,
            trace_pos_offset: usize,
            hidden_override: ?[]const f32,
            deepstack_layer_features_opt: ?[]const []const f32,
            deepstack_feature_index_opt: ?usize,
            use_preloaded_input: bool,
        ) !void {
            _ = token;
            _ = position;
            _ = slot_index;
            _ = layer_limit;
            _ = compute_logits;
            _ = ensure_kv_capacity;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            _ = use_preloaded_input;
            const idx = self.compute_calls;
            self.compute_calls += 1;
            self.recorded_download_logits[idx] = download_logits;
            self.recorded_trace_seq_lens[idx] = trace_seq_len_u32;
            self.recorded_trace_positions[idx] = trace_pos_offset;
            self.recorded_logits_out_present[idx] = logits_out_opt != null;
            if (logits_out_opt) |logits_out| {
                for (logits_out, 0..) |*logit, i| {
                    logit.* = @floatFromInt(i);
                }
            }
        }
    };

    var mock = Mock{};
    const tokens = [_]u32{ 100, 101, 102, 103 };
    var logits_out: [6]f32 = undefined;

    try engine_forward.computeGpuPrototypePrefillLogitsWithLayerLimit(
        &mock,
        tokens[0..],
        0,
        logits_out[0..],
        mock.block_runtime.blocks.len,
    );

    try std.testing.expectEqual(tokens.len, mock.compute_calls);
    for (0..tokens.len) |i| {
        const is_last = i + 1 == tokens.len;
        try std.testing.expectEqual(is_last, mock.recorded_download_logits[i]);
        try std.testing.expectEqual(is_last, mock.recorded_logits_out_present[i]);
        try std.testing.expectEqual(@as(u32, @intCast(i + 1)), mock.recorded_trace_seq_lens[i]);
        try std.testing.expectEqual(i, mock.recorded_trace_positions[i]);
    }
}

test "cpu_gpu_gpu decode parity matches single topology across slots and lifecycle cycles" {
    const slot_count: usize = 2;
    const d_model: usize = 4;
    const vocab: usize = 8;
    const split_layer: usize = 2;
    const split_layer_stage2: usize = 4;

    const CpuStage0Mock = struct {
        activations: [slot_count][d_model]f32 = [_][d_model]f32{[_]f32{0.0} ** d_model} ** slot_count,

        pub fn computePrototypeLogitsWithLayerRange(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_start: usize,
            layer_end: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            use_preloaded_input: bool,
        ) !void {
            _ = logits_out_opt;
            _ = compute_logits;
            _ = download_logits;
            _ = ensure_kv_capacity;
            _ = use_preloaded_input;
            if (slot_index >= slot_count) return error.InvalidArgument;
            if (layer_end < layer_start) return error.InvalidArgument;
            const base: f32 = @as(f32, @floatFromInt(token)) * 0.5 + @as(f32, @floatFromInt(position)) * 0.25;
            const layer_contrib: f32 = @floatFromInt(layer_end - layer_start);
            const value = base + layer_contrib;
            @memset(self.activations[slot_index][0..], value);
        }

        pub fn slotActivationBytes(self: *@This(), slot_index: usize) []const u8 {
            return std.mem.sliceAsBytes(self.activations[slot_index][0..]);
        }
    };

    const Stage1Mock = struct {
        const BlockRuntimeMock = struct {
            blocks: []const u8 = &.{},
        };

        block_runtime: BlockRuntimeMock = .{},
        state_descriptor_count: usize = 0,
        preloaded: [slot_count][d_model]f32 = [_][d_model]f32{[_]f32{0.0} ** d_model} ** slot_count,
        last_slot: usize = 0,

        pub fn mirrorSlotStateBlocksFrom(_: *@This(), _: anytype, _: usize) !void {}

        pub fn activateKvSlot(_: *@This(), _: usize) void {}

        pub fn transferPipelineActivationFromCpu(
            self: *@This(),
            src: *CpuStage0Mock,
            slot_index: usize,
            byte_count: usize,
        ) !void {
            const src_bytes = src.slotActivationBytes(slot_index);
            const dst_bytes = std.mem.sliceAsBytes(self.preloaded[slot_index][0..]);
            if (byte_count > src_bytes.len or byte_count > dst_bytes.len) return error.InvalidArgument;
            @memcpy(dst_bytes[0..byte_count], src_bytes[0..byte_count]);
        }

        pub fn transferPipelineActivation(self: *@This(), dst: anytype, byte_count: usize) !void {
            const src_bytes = std.mem.sliceAsBytes(self.preloaded[self.last_slot][0..]);
            const dst_bytes = std.mem.sliceAsBytes(dst.preloaded[self.last_slot][0..]);
            if (byte_count > src_bytes.len or byte_count > dst_bytes.len) return error.InvalidArgument;
            @memcpy(dst_bytes[0..byte_count], src_bytes[0..byte_count]);
        }

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_limit: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            trace_seq_len_u32: u32,
            trace_pos_offset: usize,
            hidden_override: ?[]const f32,
            deepstack_layer_features_opt: ?[]const []const f32,
            deepstack_feature_index_opt: ?usize,
            use_preloaded_input: bool,
        ) !void {
            _ = ensure_kv_capacity;
            _ = trace_seq_len_u32;
            _ = trace_pos_offset;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            if (slot_index >= slot_count) return error.InvalidArgument;
            self.last_slot = slot_index;
            var hidden: f32 = 0.0;
            if (use_preloaded_input) {
                hidden = self.preloaded[slot_index][0] + @as(f32, @floatFromInt(layer_limit));
            } else {
                hidden = @as(f32, @floatFromInt(token)) * 0.5 + @as(f32, @floatFromInt(position)) * 0.25 + @as(f32, @floatFromInt(layer_limit));
            }
            // Always write back: transferPipelineActivation reads from preloaded after compute.
            @memset(self.preloaded[slot_index][0..], hidden);
            if (!compute_logits) return;
            _ = download_logits;
            if (logits_out_opt) |logits_out| {
                for (logits_out, 0..) |*v, i| {
                    v.* = hidden + @as(f32, @floatFromInt(i)) * 0.01;
                }
            }
        }
    };

    const MockBackend = struct {
        const BlockRuntimeMock = struct {
            blocks: []const u8 = &.{},
        };

        topology_mode: enum { single, cpu_gpu_gpu } = .single,
        split_layer: usize = split_layer,
        split_layer_stage2: usize = split_layer_stage2,
        d_model: usize = d_model,
        vocab_size: usize = vocab,
        max_seq_len: usize = 256,
        max_batch_size: usize = slot_count,
        block_runtime: BlockRuntimeMock = .{},
        pipeline_host_staging_stage12: ?[]align(64) u8 = null,
        cpu_stage0: ?*CpuStage0Mock = null,
        stage1: ?*Stage1Mock = null,
        preloaded: [slot_count][d_model]f32 = [_][d_model]f32{[_]f32{0.0} ** d_model} ** slot_count,

        pub fn pipelineCpuStage0(self: *@This()) ?*CpuStage0Mock {
            return self.cpu_stage0;
        }

        pub fn pipelineStage1(self: *@This()) ?*Stage1Mock {
            return self.stage1;
        }

        pub fn pipelineSplitLayer(self: *const @This()) usize {
            return self.split_layer;
        }

        pub fn pipelineSplitLayerStage2(self: *const @This()) usize {
            return self.split_layer_stage2;
        }

        pub fn transferPipelineActivationFromCpu(
            _: *@This(),
            _: *CpuStage0Mock,
            _: usize,
            _: usize,
        ) !void {}

        pub fn activateKvSlot(_: *@This(), _: usize) void {}

        pub fn slotIndexSupported(_: *const @This(), slot_index: usize) bool {
            return slot_index < slot_count;
        }

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_limit: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            trace_seq_len_u32: u32,
            trace_pos_offset: usize,
            hidden_override: ?[]const f32,
            deepstack_layer_features_opt: ?[]const []const f32,
            deepstack_feature_index_opt: ?usize,
            use_preloaded_input: bool,
        ) !void {
            _ = ensure_kv_capacity;
            _ = trace_seq_len_u32;
            _ = trace_pos_offset;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            if (slot_index >= slot_count) return error.InvalidArgument;
            var hidden: f32 = 0.0;
            if (use_preloaded_input) {
                hidden = self.preloaded[slot_index][0] + @as(f32, @floatFromInt(layer_limit));
            } else {
                const base: f32 = @as(f32, @floatFromInt(token)) * 0.5 + @as(f32, @floatFromInt(position)) * 0.25;
                hidden = base + @as(f32, @floatFromInt(layer_limit));
                @memset(self.preloaded[slot_index][0..], hidden);
            }
            if (!compute_logits) return;
            _ = download_logits;
            if (logits_out_opt) |logits_out| {
                for (logits_out, 0..) |*v, i| {
                    v.* = hidden + @as(f32, @floatFromInt(i)) * 0.01;
                }
            }
        }
    };

    var cycle: usize = 0;
    while (cycle < 8) : (cycle += 1) {
        var cpu_stage = CpuStage0Mock{};
        var stage1 = Stage1Mock{
            .block_runtime = .{ .blocks = &[_]u8{ 0, 0 } },
        };
        var single = MockBackend{
            .topology_mode = .single,
            .block_runtime = .{ .blocks = &[_]u8{ 0, 0, 0, 0, 0, 0 } },
        };
        var split = MockBackend{
            .topology_mode = .cpu_gpu_gpu,
            .block_runtime = .{ .blocks = &[_]u8{ 0, 0 } },
            .cpu_stage0 = &cpu_stage,
            .stage1 = &stage1,
        };

        var positions: [slot_count]usize = [_]usize{0} ** slot_count;
        for (0..6) |step| {
            for (0..slot_count) |slot_index| {
                const token: u32 = @intCast(11 + step * 3 + slot_index);
                var logits_single: [vocab]f32 = undefined;
                var logits_split: [vocab]f32 = undefined;

                try engine_forward.computeGpuPrototypeLogitsWithLayerLimit(
                    &single,
                    token,
                    positions[slot_index],
                    slot_index,
                    logits_single[0..],
                    single.block_runtime.blocks.len,
                    true,
                    true,
                    true,
                    @intCast(positions[slot_index] + 1),
                    positions[slot_index],
                    null,
                    null,
                    null,
                    false,
                );
                try engine_forward.computeGpuPrototypeLogitsWithLayerLimit(
                    &split,
                    token,
                    positions[slot_index],
                    slot_index,
                    logits_split[0..],
                    split.block_runtime.blocks.len,
                    true,
                    true,
                    true,
                    @intCast(positions[slot_index] + 1),
                    positions[slot_index],
                    null,
                    null,
                    null,
                    false,
                );
                for (logits_single, logits_split) |lhs, rhs| {
                    try std.testing.expectApproxEqAbs(lhs, rhs, 1.0e-6);
                }
                positions[slot_index] += 1;
            }
        }
    }
}

test "cpu_gpu_gpu prefill parity matches single topology across repeated windows" {
    const d_model: usize = 4;
    const vocab: usize = 8;
    const split_layer: usize = 2;
    const split_layer_stage2: usize = 4;

    const CpuStage0Mock = struct {
        activation: [d_model]f32 = [_]f32{0.0} ** d_model,

        pub fn computePrototypeLogitsWithLayerRange(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_start: usize,
            layer_end: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            use_preloaded_input: bool,
        ) !void {
            _ = slot_index;
            _ = logits_out_opt;
            _ = compute_logits;
            _ = download_logits;
            _ = ensure_kv_capacity;
            _ = use_preloaded_input;
            if (layer_end < layer_start) return error.InvalidArgument;
            const base: f32 = @as(f32, @floatFromInt(token)) * 0.5 + @as(f32, @floatFromInt(position)) * 0.25;
            const layer_contrib: f32 = @floatFromInt(layer_end - layer_start);
            @memset(self.activation[0..], base + layer_contrib);
        }

        pub fn slotActivationBytes(self: *@This(), _: usize) []const u8 {
            return std.mem.sliceAsBytes(self.activation[0..]);
        }
    };

    const Stage1Mock = struct {
        const BlockRuntimeMock = struct {
            blocks: []const u8 = &.{},
        };

        block_runtime: BlockRuntimeMock = .{},
        state_descriptor_count: usize = 0,
        preloaded: [d_model]f32 = [_]f32{0.0} ** d_model,

        pub fn mirrorSlotStateBlocksFrom(_: *@This(), _: anytype, _: usize) !void {}

        pub fn activateKvSlot(_: *@This(), _: usize) void {}

        pub fn transferPipelineActivationFromCpu(
            self: *@This(),
            src: *CpuStage0Mock,
            _: usize,
            byte_count: usize,
        ) !void {
            const src_bytes = src.slotActivationBytes(0);
            const dst_bytes = std.mem.sliceAsBytes(self.preloaded[0..]);
            if (byte_count > src_bytes.len or byte_count > dst_bytes.len) return error.InvalidArgument;
            @memcpy(dst_bytes[0..byte_count], src_bytes[0..byte_count]);
        }

        pub fn transferPipelineActivation(self: *@This(), dst: anytype, byte_count: usize) !void {
            const src_bytes = std.mem.sliceAsBytes(self.preloaded[0..]);
            const dst_bytes = std.mem.sliceAsBytes(dst.preloaded[0..]);
            if (byte_count > src_bytes.len or byte_count > dst_bytes.len) return error.InvalidArgument;
            @memcpy(dst_bytes[0..byte_count], src_bytes[0..byte_count]);
        }

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_limit: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            trace_seq_len_u32: u32,
            trace_pos_offset: usize,
            hidden_override: ?[]const f32,
            deepstack_layer_features_opt: ?[]const []const f32,
            deepstack_feature_index_opt: ?usize,
            use_preloaded_input: bool,
        ) !void {
            _ = slot_index;
            _ = ensure_kv_capacity;
            _ = trace_seq_len_u32;
            _ = trace_pos_offset;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            var hidden: f32 = 0.0;
            if (use_preloaded_input) {
                hidden = self.preloaded[0] + @as(f32, @floatFromInt(layer_limit));
            } else {
                const base: f32 = @as(f32, @floatFromInt(token)) * 0.5 + @as(f32, @floatFromInt(position)) * 0.25;
                hidden = base + @as(f32, @floatFromInt(layer_limit));
            }
            // Always write back: transferPipelineActivation reads from preloaded after compute.
            @memset(self.preloaded[0..], hidden);
            if (!compute_logits) return;
            _ = download_logits;
            if (logits_out_opt) |logits_out| {
                for (logits_out, 0..) |*v, i| {
                    v.* = hidden + @as(f32, @floatFromInt(i)) * 0.01;
                }
            }
        }
    };

    const MockBackend = struct {
        const BlockRuntimeMock = struct {
            blocks: []const u8 = &.{},
        };

        topology_mode: enum { single, cpu_gpu_gpu } = .single,
        split_layer: usize = split_layer,
        split_layer_stage2: usize = split_layer_stage2,
        d_model: usize = d_model,
        vocab_size: usize = vocab,
        max_seq_len: usize = 256,
        block_runtime: BlockRuntimeMock = .{},
        pipeline_host_staging_stage12: ?[]align(64) u8 = null,
        cpu_stage0: ?*CpuStage0Mock = null,
        stage1: ?*Stage1Mock = null,
        preloaded: [d_model]f32 = [_]f32{0.0} ** d_model,

        pub fn slotIndexSupported(_: *const @This(), slot_index: usize) bool {
            return slot_index == 0;
        }

        pub fn pipelineCpuStage0(self: *@This()) ?*CpuStage0Mock {
            return self.cpu_stage0;
        }

        pub fn pipelineStage1(self: *@This()) ?*Stage1Mock {
            return self.stage1;
        }

        pub fn pipelineSplitLayer(self: *const @This()) usize {
            return self.split_layer;
        }

        pub fn pipelineSplitLayerStage2(self: *const @This()) usize {
            return self.split_layer_stage2;
        }

        pub fn transferPipelineActivationFromCpu(
            _: *@This(),
            _: *CpuStage0Mock,
            _: usize,
            _: usize,
        ) !void {}

        pub fn activateKvSlot(_: *@This(), _: usize) void {}

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_limit: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            trace_seq_len_u32: u32,
            trace_pos_offset: usize,
            hidden_override: ?[]const f32,
            deepstack_layer_features_opt: ?[]const []const f32,
            deepstack_feature_index_opt: ?usize,
            use_preloaded_input: bool,
        ) !void {
            _ = slot_index;
            _ = ensure_kv_capacity;
            _ = trace_seq_len_u32;
            _ = trace_pos_offset;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            var hidden: f32 = 0.0;
            if (use_preloaded_input) {
                hidden = self.preloaded[0] + @as(f32, @floatFromInt(layer_limit));
            } else {
                const base: f32 = @as(f32, @floatFromInt(token)) * 0.5 + @as(f32, @floatFromInt(position)) * 0.25;
                hidden = base + @as(f32, @floatFromInt(layer_limit));
                @memset(self.preloaded[0..], hidden);
            }
            if (!compute_logits) return;
            _ = download_logits;
            if (logits_out_opt) |logits_out| {
                for (logits_out, 0..) |*v, i| {
                    v.* = hidden + @as(f32, @floatFromInt(i)) * 0.01;
                }
            }
        }
    };

    const windows = [_][]const u32{
        &[_]u32{ 3, 5, 7, 11 },
        &[_]u32{ 13, 17 },
        &[_]u32{ 19, 23, 29 },
    };

    var cycle: usize = 0;
    while (cycle < 6) : (cycle += 1) {
        var cpu_stage = CpuStage0Mock{};
        var stage1 = Stage1Mock{
            .block_runtime = .{ .blocks = &[_]u8{ 0, 0 } },
        };
        var split = MockBackend{
            .topology_mode = .cpu_gpu_gpu,
            .block_runtime = .{ .blocks = &[_]u8{ 0, 0 } },
            .cpu_stage0 = &cpu_stage,
            .stage1 = &stage1,
        };
        for (windows) |window_tokens| {
            var logits_split: [vocab]f32 = undefined;
            try engine_forward.computeGpuPrototypePrefillLogitsWithLayerLimit(
                &split,
                window_tokens,
                0,
                logits_split[0..],
                split.block_runtime.blocks.len,
            );
            const last_index = window_tokens.len - 1;
            const base: f32 = @as(f32, @floatFromInt(window_tokens[last_index])) * 0.5 + @as(f32, @floatFromInt(last_index)) * 0.25;
            const expected_hidden = base + 6.0;
            for (logits_split, 0..) |actual, i| {
                const expected = expected_hidden + @as(f32, @floatFromInt(i)) * 0.01;
                try std.testing.expectApproxEqAbs(expected, actual, 1.0e-6);
            }
        }
    }
}

test "cpu_gpu_gpu prefill parity remains deterministic across slots and lifecycle cycles" {
    const slot_count: usize = 2;
    const d_model: usize = 4;
    const vocab: usize = 8;
    const split_layer: usize = 2;
    const split_layer_stage2: usize = 4;

    const CpuStage0Mock = struct {
        activations: [slot_count][d_model]f32 = [_][d_model]f32{[_]f32{0.0} ** d_model} ** slot_count,

        pub fn computePrototypeLogitsWithLayerRange(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_start: usize,
            layer_end: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            use_preloaded_input: bool,
        ) !void {
            _ = logits_out_opt;
            _ = compute_logits;
            _ = download_logits;
            _ = ensure_kv_capacity;
            _ = use_preloaded_input;
            if (slot_index >= slot_count) return error.InvalidArgument;
            if (layer_end < layer_start) return error.InvalidArgument;
            const base: f32 = @as(f32, @floatFromInt(token)) * 0.5 + @as(f32, @floatFromInt(position)) * 0.25;
            const layer_contrib: f32 = @floatFromInt(layer_end - layer_start);
            @memset(self.activations[slot_index][0..], base + layer_contrib);
        }

        pub fn slotActivationBytes(self: *@This(), slot_index: usize) []const u8 {
            if (slot_index >= slot_count) @panic("slot_index out of range");
            return std.mem.sliceAsBytes(self.activations[slot_index][0..]);
        }
    };

    const Stage1Mock = struct {
        const BlockRuntimeMock = struct {
            blocks: []const u8 = &.{},
        };

        block_runtime: BlockRuntimeMock = .{},
        state_descriptor_count: usize = 0,
        preloaded: [slot_count][d_model]f32 = [_][d_model]f32{[_]f32{0.0} ** d_model} ** slot_count,
        last_slot: usize = 0,

        pub fn mirrorSlotStateBlocksFrom(_: *@This(), _: anytype, _: usize) !void {}

        pub fn activateKvSlot(_: *@This(), _: usize) void {}

        pub fn transferPipelineActivationFromCpu(
            self: *@This(),
            src: *CpuStage0Mock,
            slot_index: usize,
            byte_count: usize,
        ) !void {
            const src_bytes = src.slotActivationBytes(slot_index);
            const dst_bytes = std.mem.sliceAsBytes(self.preloaded[slot_index][0..]);
            if (byte_count > src_bytes.len or byte_count > dst_bytes.len) return error.InvalidArgument;
            @memcpy(dst_bytes[0..byte_count], src_bytes[0..byte_count]);
        }

        pub fn transferPipelineActivation(self: *@This(), dst: anytype, byte_count: usize) !void {
            const src_bytes = std.mem.sliceAsBytes(self.preloaded[self.last_slot][0..]);
            const dst_bytes = std.mem.sliceAsBytes(dst.preloaded[self.last_slot][0..]);
            if (byte_count > src_bytes.len or byte_count > dst_bytes.len) return error.InvalidArgument;
            @memcpy(dst_bytes[0..byte_count], src_bytes[0..byte_count]);
        }

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_limit: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            trace_seq_len_u32: u32,
            trace_pos_offset: usize,
            hidden_override: ?[]const f32,
            deepstack_layer_features_opt: ?[]const []const f32,
            deepstack_feature_index_opt: ?usize,
            use_preloaded_input: bool,
        ) !void {
            _ = ensure_kv_capacity;
            _ = trace_seq_len_u32;
            _ = trace_pos_offset;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            if (slot_index >= slot_count) return error.InvalidArgument;
            self.last_slot = slot_index;
            var hidden: f32 = 0.0;
            if (use_preloaded_input) {
                hidden = self.preloaded[slot_index][0] + @as(f32, @floatFromInt(layer_limit));
            } else {
                hidden = @as(f32, @floatFromInt(token)) * 0.5 + @as(f32, @floatFromInt(position)) * 0.25 + @as(f32, @floatFromInt(layer_limit));
            }
            // Always write back: transferPipelineActivation reads from preloaded after compute.
            @memset(self.preloaded[slot_index][0..], hidden);
            if (!compute_logits) return;
            _ = download_logits;
            if (logits_out_opt) |logits_out| {
                for (logits_out, 0..) |*v, i| {
                    v.* = hidden + @as(f32, @floatFromInt(i)) * 0.01;
                }
            }
        }
    };

    const MockBackend = struct {
        const BlockRuntimeMock = struct {
            blocks: []const u8 = &.{},
        };

        topology_mode: enum { single, cpu_gpu_gpu } = .single,
        split_layer: usize = split_layer,
        split_layer_stage2: usize = split_layer_stage2,
        d_model: usize = d_model,
        vocab_size: usize = vocab,
        max_seq_len: usize = 256,
        block_runtime: BlockRuntimeMock = .{},
        pipeline_host_staging_stage12: ?[]align(64) u8 = null,
        cpu_stage0: ?*CpuStage0Mock = null,
        stage1: ?*Stage1Mock = null,
        preloaded: [slot_count][d_model]f32 = [_][d_model]f32{[_]f32{0.0} ** d_model} ** slot_count,

        pub fn slotIndexSupported(_: *const @This(), slot_index: usize) bool {
            return slot_index < slot_count;
        }

        pub fn pipelineCpuStage0(self: *@This()) ?*CpuStage0Mock {
            return self.cpu_stage0;
        }

        pub fn pipelineStage1(self: *@This()) ?*Stage1Mock {
            return self.stage1;
        }

        pub fn pipelineSplitLayer(self: *const @This()) usize {
            return self.split_layer;
        }

        pub fn pipelineSplitLayerStage2(self: *const @This()) usize {
            return self.split_layer_stage2;
        }

        pub fn transferPipelineActivationFromCpu(
            _: *@This(),
            _: *CpuStage0Mock,
            _: usize,
            _: usize,
        ) !void {}

        pub fn activateKvSlot(_: *@This(), _: usize) void {}

        pub fn computeGpuPrototypeLogitsWithLayerLimitTestHook(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_limit: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            trace_seq_len_u32: u32,
            trace_pos_offset: usize,
            hidden_override: ?[]const f32,
            deepstack_layer_features_opt: ?[]const []const f32,
            deepstack_feature_index_opt: ?usize,
            use_preloaded_input: bool,
        ) !void {
            _ = ensure_kv_capacity;
            _ = trace_seq_len_u32;
            _ = trace_pos_offset;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            if (slot_index >= slot_count) return error.InvalidArgument;
            var hidden: f32 = 0.0;
            if (use_preloaded_input) {
                hidden = self.preloaded[slot_index][0] + @as(f32, @floatFromInt(layer_limit));
            } else {
                const base: f32 = @as(f32, @floatFromInt(token)) * 0.5 + @as(f32, @floatFromInt(position)) * 0.25;
                hidden = base + @as(f32, @floatFromInt(layer_limit));
                @memset(self.preloaded[slot_index][0..], hidden);
            }
            if (!compute_logits) return;
            _ = download_logits;
            if (logits_out_opt) |logits_out| {
                for (logits_out, 0..) |*v, i| {
                    v.* = hidden + @as(f32, @floatFromInt(i)) * 0.01;
                }
            }
        }
    };

    const slot_windows = [_][]const []const u32{
        &[_][]const u32{
            &[_]u32{ 3, 5, 7, 11 },
            &[_]u32{ 13, 17 },
            &[_]u32{ 19, 23, 29 },
        },
        &[_][]const u32{
            &[_]u32{ 2, 4, 6 },
            &[_]u32{ 8, 10, 12, 14 },
            &[_]u32{16},
        },
    };

    var cycle: usize = 0;
    while (cycle < 6) : (cycle += 1) {
        var cpu_stage = CpuStage0Mock{};
        var stage1 = Stage1Mock{
            .block_runtime = .{ .blocks = &[_]u8{ 0, 0 } },
        };
        var split = MockBackend{
            .topology_mode = .cpu_gpu_gpu,
            .block_runtime = .{ .blocks = &[_]u8{ 0, 0 } },
            .cpu_stage0 = &cpu_stage,
            .stage1 = &stage1,
        };

        for (slot_windows, 0..) |windows, slot_index| {
            for (windows) |window_tokens| {
                var logits_split: [vocab]f32 = undefined;
                try engine_forward.computeGpuPrototypePrefillLogitsWithLayerLimit(
                    &split,
                    window_tokens,
                    slot_index,
                    logits_split[0..],
                    split.block_runtime.blocks.len,
                );
                const last_index = window_tokens.len - 1;
                const base: f32 = @as(f32, @floatFromInt(window_tokens[last_index])) * 0.5 + @as(f32, @floatFromInt(last_index)) * 0.25;
                const expected_hidden = base + 6.0;
                for (logits_split, 0..) |actual, i| {
                    const expected = expected_hidden + @as(f32, @floatFromInt(i)) * 0.01;
                    try std.testing.expectApproxEqAbs(expected, actual, 1.0e-6);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 9: Lifecycle + Safety Hardening
// ---------------------------------------------------------------------------

test "unbindSlotStateBlocks is idempotent" {
    const payload_bytes: usize = @intCast(runtime_contract.builtin_state_block_bytes);
    var backend: CudaBackend = undefined;
    backend.max_batch_size = 1;
    backend.block_runtime = undefined;
    backend.state_descriptor_count = 1;
    backend.pipeline_backend1 = null;
    backend.pipeline_backend0_cpu = null;
    backend.state_descriptors_storage[0] = .{
        .id = 51,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_none,
    };
    var slot_state_bindings: [1]CudaBackend.SlotStateBinding = .{.{}};
    backend.slot_state_bindings = slot_state_bindings[0..];

    var state_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    const state_blocks = [_]runtime_contract.StateBlockHandle{
        .{ .id = 51, .ptr = state_storage[0..].ptr, .size = runtime_contract.builtin_state_block_bytes, .align_bytes = 64 },
    };

    try backend.bindSlotStateBlocks(0, state_blocks[0..]);
    try std.testing.expect(backend.slot_state_bindings[0].bound);

    // First unbind.
    backend.unbindSlotStateBlocks(0);
    try std.testing.expect(!backend.slot_state_bindings[0].bound);
    try std.testing.expectEqual(@as(u8, 0), backend.slot_state_bindings[0].count);

    // Second unbind — must not crash and state stays unbound.
    backend.unbindSlotStateBlocks(0);
    try std.testing.expect(!backend.slot_state_bindings[0].bound);
    try std.testing.expectEqual(@as(u8, 0), backend.slot_state_bindings[0].count);
}

test "unbindSlotStateBlocks fans out to pipeline stage and is idempotent" {
    const payload_bytes: usize = @intCast(runtime_contract.builtin_state_block_bytes);

    // Source backend (stage2 / main).
    var source: CudaBackend = undefined;
    source.max_batch_size = 1;
    source.block_runtime = undefined;
    source.state_descriptor_count = 1;
    source.state_descriptors_storage[0] = .{
        .id = 52,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
    };
    var source_bindings: [1]CudaBackend.SlotStateBinding = .{.{}};
    source.slot_state_bindings = source_bindings[0..];

    // Stage1 backend.
    var stage1: CudaBackend = undefined;
    stage1.max_batch_size = 1;
    stage1.block_runtime = undefined;
    stage1.state_descriptor_count = 1;
    stage1.pipeline_backend1 = null;
    stage1.pipeline_backend0_cpu = null;
    stage1.state_descriptors_storage[0] = .{
        .id = 52,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
    };
    var stage1_bindings: [1]CudaBackend.SlotStateBinding = .{.{}};
    stage1.slot_state_bindings = stage1_bindings[0..];

    source.pipeline_backend1 = &stage1;
    // CPU stage not wired during bind — bind fans out to CPU and would fail
    // on the zero-init stub. Wire it after bind so only unbind exercises CPU fan-out.
    source.pipeline_backend0_cpu = null;

    var state_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    const state_blocks = [_]runtime_contract.StateBlockHandle{
        .{ .id = 52, .ptr = state_storage[0..].ptr, .size = runtime_contract.builtin_state_block_bytes, .align_bytes = 64 },
    };

    // Bind — mirrors to stage1 (no CPU stage wired yet).
    try source.bindSlotStateBlocks(0, state_blocks[0..]);

    // Now wire the zero-init CPU stub so unbind exercises the CPU fan-out path.
    // Uses aligned byte buffer to satisfy the struct's natural alignment requirement.
    const cpu_backend = @import("../cpu/root.zig");
    var cpu_stage0_bytes: [@sizeOf(cpu_backend.BackendType)]u8 align(@alignOf(cpu_backend.BackendType)) = @splat(0);
    const cpu_stage0: *cpu_backend.BackendType = @ptrCast(&cpu_stage0_bytes);
    source.pipeline_backend0_cpu = cpu_stage0;
    try std.testing.expect(source.slot_state_bindings[0].bound);
    try std.testing.expect(stage1.slot_state_bindings[0].bound);

    // First unbind — fans out to stage1 + CPU.
    source.unbindSlotStateBlocks(0);
    try std.testing.expect(!source.slot_state_bindings[0].bound);
    try std.testing.expectEqual(@as(u8, 0), source.slot_state_bindings[0].count);
    try std.testing.expect(!stage1.slot_state_bindings[0].bound);
    try std.testing.expectEqual(@as(u8, 0), stage1.slot_state_bindings[0].count);

    // Second unbind — idempotent across all stages.
    source.unbindSlotStateBlocks(0);
    try std.testing.expect(!source.slot_state_bindings[0].bound);
    try std.testing.expect(!stage1.slot_state_bindings[0].bound);
}

test "resetSlot is safe on unbound and bound slots with pipeline stage" {
    // Source backend.
    var source: CudaBackend = undefined;
    source.max_batch_size = 1;
    source.state_descriptor_count = 0;
    source.pipeline_backend0_cpu = null;
    var source_positions: [1]usize = .{42};
    var source_deltas: [1]isize = .{7};
    source.slot_positions = source_positions[0..];
    source.slot_rope_position_deltas = source_deltas[0..];

    // Stage1 backend.
    var stage1: CudaBackend = undefined;
    stage1.max_batch_size = 1;
    stage1.state_descriptor_count = 0;
    stage1.pipeline_backend1 = null;
    stage1.pipeline_backend0_cpu = null;
    var stage1_positions: [1]usize = .{99};
    var stage1_deltas: [1]isize = .{-3};
    stage1.slot_positions = stage1_positions[0..];
    stage1.slot_rope_position_deltas = stage1_deltas[0..];

    source.pipeline_backend1 = &stage1;

    // Reset on slot with non-zero positions — must fan out to stage1.
    source.resetSlot(0);
    try std.testing.expectEqual(@as(usize, 0), source.slot_positions[0]);
    try std.testing.expectEqual(@as(isize, 0), source.slot_rope_position_deltas[0]);
    try std.testing.expectEqual(@as(usize, 0), stage1.slot_positions[0]);
    try std.testing.expectEqual(@as(isize, 0), stage1.slot_rope_position_deltas[0]);

    // Second reset — idempotent, positions stay zero.
    source.resetSlot(0);
    try std.testing.expectEqual(@as(usize, 0), source.slot_positions[0]);
    try std.testing.expectEqual(@as(usize, 0), stage1.slot_positions[0]);
}

test "bind-unbind-rebind produces independent runtime state across pipeline stages" {
    const payload_bytes: usize = @intCast(runtime_contract.builtin_state_block_bytes);

    // Source backend.
    var source: CudaBackend = undefined;
    source.max_batch_size = 1;
    source.block_runtime = undefined;
    source.state_descriptor_count = 1;
    source.pipeline_backend0_cpu = null;
    source.state_descriptors_storage[0] = .{
        .id = 53,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
    };
    var source_bindings: [1]CudaBackend.SlotStateBinding = .{.{}};
    source.slot_state_bindings = source_bindings[0..];

    // Stage1 backend.
    var stage1: CudaBackend = undefined;
    stage1.max_batch_size = 1;
    stage1.block_runtime = undefined;
    stage1.state_descriptor_count = 1;
    stage1.pipeline_backend1 = null;
    stage1.pipeline_backend0_cpu = null;
    stage1.state_descriptors_storage[0] = .{
        .id = 53,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
    };
    var stage1_bindings: [1]CudaBackend.SlotStateBinding = .{.{}};
    stage1.slot_state_bindings = stage1_bindings[0..];

    source.pipeline_backend1 = &stage1;

    // --- First bind ---
    var storage_a: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    const blocks_a = [_]runtime_contract.StateBlockHandle{
        .{ .id = 53, .ptr = storage_a[0..].ptr, .size = runtime_contract.builtin_state_block_bytes, .align_bytes = 64 },
    };
    try source.bindSlotStateBlocks(0, blocks_a[0..]);

    // Stage1 kv runtime must use stage1's block_runtime, not source's.
    const mirrored_a = stage1.slotStateBlocks(0);
    const kv_a = runtime_contract.stateValueFromBlock(*KvRuntimeState, &mirrored_a[0]) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@intFromPtr(&stage1.block_runtime), @intFromPtr(kv_a.block_runtime));

    // Source kv runtime must use source's block_runtime.
    const source_bound_a = source.slotStateBlocks(0);
    const source_kv_a = runtime_contract.stateValueFromBlock(*KvRuntimeState, &source_bound_a[0]) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@intFromPtr(&source.block_runtime), @intFromPtr(source_kv_a.block_runtime));

    // No cross-stage pointer aliasing (ADR Rule 7).
    try std.testing.expect(@intFromPtr(mirrored_a[0].ptr) != @intFromPtr(source_bound_a[0].ptr));

    // --- Unbind ---
    source.unbindSlotStateBlocks(0);
    try std.testing.expect(!source.slot_state_bindings[0].bound);
    try std.testing.expect(!stage1.slot_state_bindings[0].bound);

    // --- Rebind with different backing memory ---
    var storage_b: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    const blocks_b = [_]runtime_contract.StateBlockHandle{
        .{ .id = 53, .ptr = storage_b[0..].ptr, .size = runtime_contract.builtin_state_block_bytes, .align_bytes = 64 },
    };
    try source.bindSlotStateBlocks(0, blocks_b[0..]);
    defer source.unbindSlotStateBlocks(0);

    // Stage1 kv runtime must be freshly synthesized (not stale from first bind).
    const mirrored_b = stage1.slotStateBlocks(0);
    const kv_b = runtime_contract.stateValueFromBlock(*KvRuntimeState, &mirrored_b[0]) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@intFromPtr(&stage1.block_runtime), @intFromPtr(kv_b.block_runtime));

    // Source kv runtime still uses source's block_runtime.
    const source_bound_b = source.slotStateBlocks(0);
    const source_kv_b = runtime_contract.stateValueFromBlock(*KvRuntimeState, &source_bound_b[0]) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@intFromPtr(&source.block_runtime), @intFromPtr(source_kv_b.block_runtime));

    // No cross-stage aliasing after rebind.
    try std.testing.expect(@intFromPtr(mirrored_b[0].ptr) != @intFromPtr(source_bound_b[0].ptr));
}

test "interleaved multi-slot lifecycle with pipeline stage" {
    const payload_bytes: usize = @intCast(runtime_contract.builtin_state_block_bytes);

    // Source backend — 2 slots.
    var source: CudaBackend = undefined;
    source.max_batch_size = 2;
    source.block_runtime = undefined;
    source.state_descriptor_count = 1;
    source.pipeline_backend0_cpu = null;
    source.state_descriptors_storage[0] = .{
        .id = 54,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
    };
    var source_bindings: [2]CudaBackend.SlotStateBinding = .{ .{}, .{} };
    source.slot_state_bindings = source_bindings[0..];

    // Stage1 backend — 2 slots.
    var stage1: CudaBackend = undefined;
    stage1.max_batch_size = 2;
    stage1.block_runtime = undefined;
    stage1.state_descriptor_count = 1;
    stage1.pipeline_backend1 = null;
    stage1.pipeline_backend0_cpu = null;
    stage1.state_descriptors_storage[0] = .{
        .id = 54,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
    };
    var stage1_bindings: [2]CudaBackend.SlotStateBinding = .{ .{}, .{} };
    stage1.slot_state_bindings = stage1_bindings[0..];

    source.pipeline_backend1 = &stage1;

    var storage_s0: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    var storage_s1: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    const blocks_s0 = [_]runtime_contract.StateBlockHandle{
        .{ .id = 54, .ptr = storage_s0[0..].ptr, .size = runtime_contract.builtin_state_block_bytes, .align_bytes = 64 },
    };
    const blocks_s1 = [_]runtime_contract.StateBlockHandle{
        .{ .id = 54, .ptr = storage_s1[0..].ptr, .size = runtime_contract.builtin_state_block_bytes, .align_bytes = 64 },
    };

    // Step 1: Bind slot 0 — only slot 0 bound.
    try source.bindSlotStateBlocks(0, blocks_s0[0..]);
    try std.testing.expect(source.slot_state_bindings[0].bound);
    try std.testing.expect(!source.slot_state_bindings[1].bound);
    try std.testing.expect(stage1.slot_state_bindings[0].bound);
    try std.testing.expect(!stage1.slot_state_bindings[1].bound);

    // Step 2: Bind slot 1 — both slots bound.
    try source.bindSlotStateBlocks(1, blocks_s1[0..]);
    try std.testing.expect(source.slot_state_bindings[0].bound);
    try std.testing.expect(source.slot_state_bindings[1].bound);
    try std.testing.expect(stage1.slot_state_bindings[0].bound);
    try std.testing.expect(stage1.slot_state_bindings[1].bound);

    // Step 3: Unbind slot 0 — slot 1 stays bound.
    source.unbindSlotStateBlocks(0);
    try std.testing.expect(!source.slot_state_bindings[0].bound);
    try std.testing.expect(source.slot_state_bindings[1].bound);
    try std.testing.expect(!stage1.slot_state_bindings[0].bound);
    try std.testing.expect(stage1.slot_state_bindings[1].bound);

    // Step 4: Rebind slot 0 — both slots bound again.
    try source.bindSlotStateBlocks(0, blocks_s0[0..]);
    try std.testing.expect(source.slot_state_bindings[0].bound);
    try std.testing.expect(source.slot_state_bindings[1].bound);

    // Step 5: Verify independent runtime state pointers per slot on stage1.
    const s0_blocks = stage1.slotStateBlocks(0);
    const s1_blocks = stage1.slotStateBlocks(1);
    const s0_kv = runtime_contract.stateValueFromBlock(*KvRuntimeState, &s0_blocks[0]) orelse return error.TestUnexpectedResult;
    const s1_kv = runtime_contract.stateValueFromBlock(*KvRuntimeState, &s1_blocks[0]) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 0), s0_kv.slot_index);
    try std.testing.expectEqual(@as(usize, 1), s1_kv.slot_index);
    try std.testing.expectEqual(@intFromPtr(&stage1.block_runtime), @intFromPtr(s0_kv.block_runtime));
    try std.testing.expectEqual(@intFromPtr(&stage1.block_runtime), @intFromPtr(s1_kv.block_runtime));

    // Step 6: Unbind slot 1 — slot 0 stays bound.
    source.unbindSlotStateBlocks(1);
    try std.testing.expect(source.slot_state_bindings[0].bound);
    try std.testing.expect(!source.slot_state_bindings[1].bound);
    try std.testing.expect(stage1.slot_state_bindings[0].bound);
    try std.testing.expect(!stage1.slot_state_bindings[1].bound);

    // Step 7: Unbind slot 0 — both slots unbound.
    source.unbindSlotStateBlocks(0);
    try std.testing.expect(!source.slot_state_bindings[0].bound);
    try std.testing.expect(!source.slot_state_bindings[1].bound);
    try std.testing.expect(!stage1.slot_state_bindings[0].bound);
    try std.testing.expect(!stage1.slot_state_bindings[1].bound);
}

// ---------------------------------------------------------------------------
// computeInitLayerRange — A10 regression tests (range-scoped init invariant)
// ---------------------------------------------------------------------------
//
// These tests exercise the pure validation function that CudaBackend.init()
// calls to determine its layer range. The debug.assert in init() then verifies
// BlockRuntime.initRange produced the matching block count. Together they
// ensure no full-model GPU allocation occurs for staged topologies.

const computeInitLayerRange = CudaBackend.computeInitLayerRange;
const InitOptions = CudaBackend.InitOptions;
const no_sharing_config: tensor.ModelConfig = .{};

test "computeInitLayerRange: single topology uses full range" {
    const r = try computeInitLayerRange(.{ .topology_mode = .single }, 32, no_sharing_config);
    try std.testing.expectEqual(@as(usize, 0), r.start);
    try std.testing.expectEqual(@as(usize, 32), r.end);
    try std.testing.expectEqual(@as(usize, 0), r.split_layer);
    try std.testing.expectEqual(@as(usize, 0), r.split_layer_stage2);
}

test "computeInitLayerRange: pipeline2 default split is n/2" {
    const r = try computeInitLayerRange(.{ .topology_mode = .pipeline2 }, 32, no_sharing_config);
    try std.testing.expectEqual(@as(usize, 0), r.start);
    try std.testing.expectEqual(@as(usize, 16), r.end);
    try std.testing.expectEqual(@as(usize, 16), r.split_layer);
}

test "computeInitLayerRange: pipeline2 explicit split" {
    const r = try computeInitLayerRange(.{ .topology_mode = .pipeline2, .split_layer = 10 }, 32, no_sharing_config);
    try std.testing.expectEqual(@as(usize, 0), r.start);
    try std.testing.expectEqual(@as(usize, 10), r.end);
    try std.testing.expectEqual(@as(usize, 10), r.split_layer);
}

test "computeInitLayerRange: pipeline2 rejects split_layer=0" {
    try std.testing.expectError(
        error.InvalidTopologyConfig,
        computeInitLayerRange(.{ .topology_mode = .pipeline2, .split_layer = 0 }, 32, no_sharing_config),
    );
}

test "computeInitLayerRange: pipeline2 rejects split_layer>=total" {
    try std.testing.expectError(
        error.InvalidTopologyConfig,
        computeInitLayerRange(.{ .topology_mode = .pipeline2, .split_layer = 32 }, 32, no_sharing_config),
    );
}

test "computeInitLayerRange: pipeline2 rejects 1 layer" {
    try std.testing.expectError(
        error.InvalidTopologyConfig,
        computeInitLayerRange(.{ .topology_mode = .pipeline2 }, 1, no_sharing_config),
    );
}

test "computeInitLayerRange: cpu_gpu default split returns upper half" {
    const r = try computeInitLayerRange(.{ .topology_mode = .cpu_gpu }, 32, no_sharing_config);
    try std.testing.expectEqual(@as(usize, 16), r.start);
    try std.testing.expectEqual(@as(usize, 32), r.end);
    try std.testing.expectEqual(@as(usize, 16), r.split_layer);
}

test "computeInitLayerRange: cpu_gpu explicit split" {
    const r = try computeInitLayerRange(.{ .topology_mode = .cpu_gpu, .split_layer = 8 }, 32, no_sharing_config);
    try std.testing.expectEqual(@as(usize, 8), r.start);
    try std.testing.expectEqual(@as(usize, 32), r.end);
    try std.testing.expectEqual(@as(usize, 8), r.split_layer);
}

test "computeInitLayerRange: cpu_gpu rejects split_layer=0" {
    try std.testing.expectError(
        error.InvalidTopologyConfig,
        computeInitLayerRange(.{ .topology_mode = .cpu_gpu, .split_layer = 0 }, 32, no_sharing_config),
    );
}

test "computeInitLayerRange: cpu_gpu rejects 1 layer" {
    try std.testing.expectError(
        error.InvalidTopologyConfig,
        computeInitLayerRange(.{ .topology_mode = .cpu_gpu }, 1, no_sharing_config),
    );
}

test "computeInitLayerRange: cpu_gpu_gpu default splits 3-way" {
    // 12 layers: split=max(1,12/3)=4, split_stage2=4+max(1,(12-4)/2)=4+4=8.
    // Self backend gets [8, 12).
    const r = try computeInitLayerRange(.{ .topology_mode = .cpu_gpu_gpu }, 12, no_sharing_config);
    try std.testing.expectEqual(@as(usize, 8), r.start);
    try std.testing.expectEqual(@as(usize, 12), r.end);
    try std.testing.expectEqual(@as(usize, 4), r.split_layer);
    try std.testing.expectEqual(@as(usize, 8), r.split_layer_stage2);
}

test "computeInitLayerRange: cpu_gpu_gpu explicit splits" {
    const r = try computeInitLayerRange(.{
        .topology_mode = .cpu_gpu_gpu,
        .split_layer = 4,
        .split_layer_stage2 = 8,
    }, 12, no_sharing_config);
    try std.testing.expectEqual(@as(usize, 8), r.start);
    try std.testing.expectEqual(@as(usize, 12), r.end);
    try std.testing.expectEqual(@as(usize, 4), r.split_layer);
    try std.testing.expectEqual(@as(usize, 8), r.split_layer_stage2);
}

test "computeInitLayerRange: cpu_gpu_gpu rejects split_stage2<=split" {
    try std.testing.expectError(
        error.InvalidTopologyConfig,
        computeInitLayerRange(.{
            .topology_mode = .cpu_gpu_gpu,
            .split_layer = 5,
            .split_layer_stage2 = 5,
        }, 12, no_sharing_config),
    );
}

test "computeInitLayerRange: cpu_gpu_gpu rejects split_stage2>=total" {
    try std.testing.expectError(
        error.InvalidTopologyConfig,
        computeInitLayerRange(.{
            .topology_mode = .cpu_gpu_gpu,
            .split_layer = 4,
            .split_layer_stage2 = 12,
        }, 12, no_sharing_config),
    );
}

test "computeInitLayerRange: cpu_gpu_gpu rejects fewer than 3 layers" {
    try std.testing.expectError(
        error.InvalidTopologyConfig,
        computeInitLayerRange(.{ .topology_mode = .cpu_gpu_gpu }, 2, no_sharing_config),
    );
}

test "computeInitLayerRange: explicit init_layer_range with single mode" {
    const r = try computeInitLayerRange(.{
        .topology_mode = .single,
        .init_layer_range = .{ .start = 5, .end = 10 },
    }, 32, no_sharing_config);
    try std.testing.expectEqual(@as(usize, 5), r.start);
    try std.testing.expectEqual(@as(usize, 10), r.end);
    try std.testing.expectEqual(@as(usize, 0), r.split_layer);
    try std.testing.expectEqual(@as(usize, 0), r.split_layer_stage2);
}

test "computeInitLayerRange: rejects init_layer_range with staged topology" {
    // init_layer_range is internal for stage backends, which must be .single.
    try std.testing.expectError(
        error.InvalidTopologyConfig,
        computeInitLayerRange(.{
            .topology_mode = .pipeline2,
            .init_layer_range = .{ .start = 5, .end = 10 },
        }, 32, no_sharing_config),
    );
    try std.testing.expectError(
        error.InvalidTopologyConfig,
        computeInitLayerRange(.{
            .topology_mode = .cpu_gpu,
            .init_layer_range = .{ .start = 5, .end = 10 },
        }, 32, no_sharing_config),
    );
    try std.testing.expectError(
        error.InvalidTopologyConfig,
        computeInitLayerRange(.{
            .topology_mode = .cpu_gpu_gpu,
            .init_layer_range = .{ .start = 5, .end = 10 },
        }, 32, no_sharing_config),
    );
}

test "computeInitLayerRange: rejects init_layer_range with start>=end" {
    try std.testing.expectError(
        error.InvalidTopologyConfig,
        computeInitLayerRange(.{
            .init_layer_range = .{ .start = 10, .end = 10 },
        }, 32, no_sharing_config),
    );
}

test "computeInitLayerRange: rejects init_layer_range with end>total_layers" {
    try std.testing.expectError(
        error.InvalidTopologyConfig,
        computeInitLayerRange(.{
            .init_layer_range = .{ .start = 0, .end = 33 },
        }, 32, no_sharing_config),
    );
}

// Structural invariant: split points partition [0, total_layers) consistently
// with the self-backend range. The topology switch relies on this relationship,
// and breaking it would cause stage backends to receive wrong layer ranges.
test "computeInitLayerRange: pipeline2 split points cover full model" {
    const total: usize = 24;
    const r = try computeInitLayerRange(.{ .topology_mode = .pipeline2, .split_layer = 10 }, total, no_sharing_config);
    // Self [0, split) + stage1 [split, total) == [0, total).
    try std.testing.expectEqual(@as(usize, 0), r.start);
    try std.testing.expectEqual(r.split_layer, r.end);
    try std.testing.expect(r.split_layer > 0 and r.split_layer < total);
}

test "computeInitLayerRange: cpu_gpu split points cover full model" {
    const total: usize = 24;
    const r = try computeInitLayerRange(.{ .topology_mode = .cpu_gpu, .split_layer = 10 }, total, no_sharing_config);
    // CPU [0, split) + self [split, total) == [0, total).
    try std.testing.expectEqual(r.split_layer, r.start);
    try std.testing.expectEqual(total, r.end);
    try std.testing.expect(r.split_layer > 0 and r.split_layer < total);
}

test "computeInitLayerRange: cpu_gpu_gpu split points cover full model" {
    const total: usize = 24;
    const r = try computeInitLayerRange(.{
        .topology_mode = .cpu_gpu_gpu,
        .split_layer = 6,
        .split_layer_stage2 = 16,
    }, total, no_sharing_config);
    // CPU [0, split) + GPU0 [split, split_stage2) + self [split_stage2, total) == [0, total).
    try std.testing.expect(r.split_layer > 0);
    try std.testing.expect(r.split_layer < r.split_layer_stage2);
    try std.testing.expect(r.split_layer_stage2 < total);
    try std.testing.expectEqual(r.split_layer_stage2, r.start);
    try std.testing.expectEqual(total, r.end);
    // Layer counts per stage.
    const cpu_layers = r.split_layer;
    const gpu0_layers = r.split_layer_stage2 - r.split_layer;
    const gpu1_layers = r.end - r.start;
    try std.testing.expectEqual(total, cpu_layers + gpu0_layers + gpu1_layers);
}

test "Nvfp4LinearWeight.cublasLtScaleTensorSize computes padded VEC16 layout bytes" {
    // 16-scale columns become ceil(inner/16), then rounded to multiples of 4.
    // Outer dimension is rounded to 128.
    try std.testing.expectEqual(@as(usize, 128 * 160), Nvfp4LinearWeight.cublasLtScaleTensorSize(2560, 8));
    try std.testing.expectEqual(@as(usize, 128 * 4), Nvfp4LinearWeight.cublasLtScaleTensorSize(48, 1));
}

test "Nvfp4LinearWeight.roundoff rounds up to granularity" {
    try std.testing.expectEqual(@as(usize, 128), Nvfp4LinearWeight.roundoff(1, 128));
    try std.testing.expectEqual(@as(usize, 128), Nvfp4LinearWeight.roundoff(128, 128));
    try std.testing.expectEqual(@as(usize, 256), Nvfp4LinearWeight.roundoff(129, 128));
}
