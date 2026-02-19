//! Integration tests for inference.executor.Block
//!
//! Tests the Block type from the executor module, including:
//! - Block configuration and validation
//! - Accessor methods (getHiddenSize, getBlockIdx, getAttention, getFFN)
//! - Description methods (describe, describeTopology)
//! - Forward pass operations (forward, forwardWithBatchedCache)

const std = @import("std");
const main = @import("main");

const Block = main.inference.executor.Block;
const Tensor = main.core.Tensor;
const DType = main.core.DType;

// Re-exports for test helpers
const LayerOp = main.models.dispatcher.layer_ops.LayerOp;
const NormSlot = main.models.dispatcher.layer_ops.NormSlot;
const BufferId = main.models.dispatcher.layer_ops.BufferId;
const ResidualScale = main.models.dispatcher.layer_ops.ResidualScale;

// Backend kernel types
const backend = main.inference.backend;
const TransformerBlock = backend.TransformerBlock;
const ScratchBuffer = backend.ScratchBuffer;
const AttnCache = backend.AttnCache;
const MultiHeadAttention = backend.MultiHeadAttention;
const RMSNorm = backend.block_kernels.norm.RMSNorm;
const SwiGLU = backend.SwiGLU;
const BatchedKVCache = backend.kernels.kv_cache.BatchedKVCache;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "Block type is accessible" {
    const T = Block;
    _ = T;
}

test "Block is a struct" {
    const info = @typeInfo(Block);
    try std.testing.expect(info == .@"struct");
}

test "Block has expected fields" {
    const info = @typeInfo(Block);
    const fields = info.@"struct".fields;

    var has_program = false;
    var has_block = false;
    var has_block_idx = false;
    var has_hidden_size = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "program")) has_program = true;
        if (comptime std.mem.eql(u8, field.name, "block")) has_block = true;
        if (comptime std.mem.eql(u8, field.name, "block_idx")) has_block_idx = true;
        if (comptime std.mem.eql(u8, field.name, "hidden_size")) has_hidden_size = true;
    }

    try std.testing.expect(has_program);
    try std.testing.expect(has_block);
    try std.testing.expect(has_block_idx);
    try std.testing.expect(has_hidden_size);
}

// =============================================================================
// Test Helper: Create a minimal TransformerBlock
// =============================================================================

fn createTestBlock(allocator: std.mem.Allocator) !TransformerBlock {
    const hidden_size = 128;
    const n_heads = 4;
    const head_dim = 32;
    const intermediate_size = 256;

    // Create RMSNorm weights
    var ln1_weight = try allocator.alloc(f32, hidden_size);
    var ln2_weight = try allocator.alloc(f32, hidden_size);
    for (0..hidden_size) |i| {
        ln1_weight[i] = 1.0;
        ln2_weight[i] = 1.0;
    }

    // Create attention weights
    const qkv_size = (n_heads + 2) * head_dim; // n_heads for Q, 1 for K, 1 for V (GQA)
    const qkv_weight_size = hidden_size * qkv_size;
    var qkv_weight = try allocator.alloc(f32, qkv_weight_size);
    var o_weight = try allocator.alloc(f32, hidden_size * hidden_size);

    for (0..qkv_weight_size) |i| {
        qkv_weight[i] = 0.01 * @as(f32, @floatFromInt(i % 100));
    }
    for (0..hidden_size * hidden_size) |i| {
        o_weight[i] = 0.01 * @as(f32, @floatFromInt(i % 100));
    }

    // Create FFN weights
    var gate_weight = try allocator.alloc(f32, hidden_size * intermediate_size);
    var up_weight = try allocator.alloc(f32, hidden_size * intermediate_size);
    var down_weight = try allocator.alloc(f32, intermediate_size * hidden_size);

    for (0..hidden_size * intermediate_size) |i| {
        gate_weight[i] = 0.01 * @as(f32, @floatFromInt(i % 100));
        up_weight[i] = 0.01 * @as(f32, @floatFromInt(i % 100));
    }
    for (0..intermediate_size * hidden_size) |i| {
        down_weight[i] = 0.01 * @as(f32, @floatFromInt(i % 100));
    }

    // Create tensors
    const ln1_weight_bytes = std.mem.sliceAsBytes(ln1_weight);
    const ln2_weight_bytes = std.mem.sliceAsBytes(ln2_weight);
    const qkv_weight_bytes = std.mem.sliceAsBytes(qkv_weight);
    const o_weight_bytes = std.mem.sliceAsBytes(o_weight);
    const gate_weight_bytes = std.mem.sliceAsBytes(gate_weight);
    const up_weight_bytes = std.mem.sliceAsBytes(up_weight);
    const down_weight_bytes = std.mem.sliceAsBytes(down_weight);

    const ln1_tensor = Tensor.view(ln1_weight_bytes.ptr, &.{hidden_size}, .f32, null);
    const ln2_tensor = Tensor.view(ln2_weight_bytes.ptr, &.{hidden_size}, .f32, null);
    const qkv_tensor = Tensor.view(qkv_weight_bytes.ptr, &.{ hidden_size, qkv_size }, .f32, null);
    const o_tensor = Tensor.view(o_weight_bytes.ptr, &.{ hidden_size, hidden_size }, .f32, null);
    const gate_tensor = Tensor.view(gate_weight_bytes.ptr, &.{ hidden_size, intermediate_size }, .f32, null);
    const up_tensor = Tensor.view(up_weight_bytes.ptr, &.{ hidden_size, intermediate_size }, .f32, null);
    const down_tensor = Tensor.view(down_weight_bytes.ptr, &.{ intermediate_size, hidden_size }, .f32, null);

    // Create RoPE
    const rope = RoPE.init(head_dim, 10000.0, 2048);

    // Create attention
    const attention = MultiHeadAttention{
        .n_heads = n_heads,
        .n_kv_heads = 1, // GQA
        .head_dim = head_dim,
        .hidden_size = hidden_size,
        .qkv_weight = qkv_tensor,
        .o_weight = o_tensor,
        .rope = rope,
    };

    // Create FFN
    const ffn = SwiGLU{
        .d_model = hidden_size,
        .d_ff = intermediate_size,
        .gate_weight = gate_tensor,
        .up_weight = up_tensor,
        .down_weight = down_tensor,
    };

    // Create weight registry
    var registry = WeightRegistry.init(allocator);
    try registry.put("qkv_proj", qkv_tensor);
    try registry.put("o_proj", o_tensor);
    try registry.put("gate_proj", gate_tensor);
    try registry.put("up_proj", up_tensor);
    try registry.put("down_proj", down_tensor);

    // Create TransformerBlock
    return TransformerBlock{
        .ln1 = RMSNorm{ .dim = hidden_size, .weight = ln1_tensor, .eps = 1e-5 },
        .ln2 = RMSNorm{ .dim = hidden_size, .weight = ln2_tensor, .eps = 1e-5 },
        .attention = attention,
        .ffn_layer = .{ .swiglu = ffn },
        .residual_multiplier = 1.0,
        .pre_ffn_norm = null,
        .post_ffn_norm = null,
        .weight_registry = registry,
    };
}

// =============================================================================
// Accessor Method Tests
// =============================================================================

test "getHiddenSize returns correct hidden size" {
    const allocator = std.testing.allocator;

    var transformer_block = try createTestBlock(allocator);
    defer {
        allocator.free(transformer_block.ln1.weight.asSlice(f32));
        allocator.free(transformer_block.ln2.weight.asSlice(f32));
        allocator.free(transformer_block.attention.qkv_weight.asSlice(f32));
        allocator.free(transformer_block.attention.o_weight.asSlice(f32));
        const swiglu = transformer_block.ffn_layer.swiglu;
        allocator.free(swiglu.gate_weight.asSlice(f32));
        allocator.free(swiglu.up_weight.asSlice(f32));
        allocator.free(swiglu.down_weight.asSlice(f32));
        transformer_block.weight_registry.deinit();
    }

    const program = [_]LayerOp{
        .{ .norm = .{ .in = .residual, .out = .norm_out, .which = .ln1 } },
        .{ .attn = .{ .in = .norm_out, .out = .branch_out } },
        .{ .add = .{ .branch = .branch_out, .scale = .one } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    try std.testing.expectEqual(@as(usize, 128), block.getHiddenSize());
}

test "getBlockIdx returns correct block index" {
    const allocator = std.testing.allocator;

    var transformer_block = try createTestBlock(allocator);
    defer {
        allocator.free(transformer_block.ln1.weight.asSlice(f32));
        allocator.free(transformer_block.ln2.weight.asSlice(f32));
        allocator.free(transformer_block.attention.qkv_weight.asSlice(f32));
        allocator.free(transformer_block.attention.o_weight.asSlice(f32));
        const swiglu = transformer_block.ffn_layer.swiglu;
        allocator.free(swiglu.gate_weight.asSlice(f32));
        allocator.free(swiglu.up_weight.asSlice(f32));
        allocator.free(swiglu.down_weight.asSlice(f32));
        transformer_block.weight_registry.deinit();
    }

    const program = [_]LayerOp{
        .{ .norm = .{ .in = .residual, .out = .norm_out, .which = .ln1 } },
    };

    const block_idx: usize = 5;
    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = block_idx,
        .hidden_size = 128,
    };

    try std.testing.expectEqual(block_idx, block.getBlockIdx());
}

test "getAttention returns attention module" {
    const allocator = std.testing.allocator;

    var transformer_block = try createTestBlock(allocator);
    defer {
        allocator.free(transformer_block.ln1.weight.asSlice(f32));
        allocator.free(transformer_block.ln2.weight.asSlice(f32));
        allocator.free(transformer_block.attention.qkv_weight.asSlice(f32));
        allocator.free(transformer_block.attention.o_weight.asSlice(f32));
        const swiglu = transformer_block.ffn_layer.swiglu;
        allocator.free(swiglu.gate_weight.asSlice(f32));
        allocator.free(swiglu.up_weight.asSlice(f32));
        allocator.free(swiglu.down_weight.asSlice(f32));
        transformer_block.weight_registry.deinit();
    }

    const program = [_]LayerOp{
        .{ .attn = .{ .in = .norm_out, .out = .branch_out } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    const attention = block.getAttention();
    try std.testing.expectEqual(@as(usize, 4), attention.n_heads);
    try std.testing.expectEqual(@as(usize, 32), attention.head_dim);
    try std.testing.expectEqual(@as(usize, 128), attention.hidden_size);
}

test "getFFN returns FFN layer" {
    const allocator = std.testing.allocator;

    var transformer_block = try createTestBlock(allocator);
    defer {
        allocator.free(transformer_block.ln1.weight.asSlice(f32));
        allocator.free(transformer_block.ln2.weight.asSlice(f32));
        allocator.free(transformer_block.attention.qkv_weight.asSlice(f32));
        allocator.free(transformer_block.attention.o_weight.asSlice(f32));
        const swiglu = transformer_block.ffn_layer.swiglu;
        allocator.free(swiglu.gate_weight.asSlice(f32));
        allocator.free(swiglu.up_weight.asSlice(f32));
        allocator.free(swiglu.down_weight.asSlice(f32));
        transformer_block.weight_registry.deinit();
    }

    const program = [_]LayerOp{
        .{ .ffn = .{ .in = .norm_out, .out = .branch_out } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    const ffn = block.getFFN();
    switch (ffn.*) {
        .swiglu => |swiglu| {
            try std.testing.expectEqual(@as(usize, 128), swiglu.d_model);
            try std.testing.expectEqual(@as(usize, 256), swiglu.d_ff);
        },
        else => try std.testing.expect(false), // Should be SwiGLU
    }
}

// =============================================================================
// Validation Tests
// =============================================================================

test "validate succeeds with valid program" {
    const allocator = std.testing.allocator;

    var transformer_block = try createTestBlock(allocator);
    defer {
        allocator.free(transformer_block.ln1.weight.asSlice(f32));
        allocator.free(transformer_block.ln2.weight.asSlice(f32));
        allocator.free(transformer_block.attention.qkv_weight.asSlice(f32));
        allocator.free(transformer_block.attention.o_weight.asSlice(f32));
        const swiglu = transformer_block.ffn_layer.swiglu;
        allocator.free(swiglu.gate_weight.asSlice(f32));
        allocator.free(swiglu.up_weight.asSlice(f32));
        allocator.free(swiglu.down_weight.asSlice(f32));
        transformer_block.weight_registry.deinit();
    }

    const program = [_]LayerOp{
        .{ .norm = .{ .in = .residual, .out = .norm_out, .which = .ln1 } },
        .{ .attn = .{ .in = .norm_out, .out = .branch_out } },
        .{ .add = .{ .branch = .branch_out, .scale = .one } },
        .{ .norm = .{ .in = .residual, .out = .norm_out, .which = .ln2 } },
        .{ .ffn = .{ .in = .norm_out, .out = .branch_out } },
        .{ .add = .{ .branch = .branch_out, .scale = .one } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    try block.validate();
}

test "validate fails with missing weight" {
    const allocator = std.testing.allocator;

    var transformer_block = try createTestBlock(allocator);
    defer {
        allocator.free(transformer_block.ln1.weight.asSlice(f32));
        allocator.free(transformer_block.ln2.weight.asSlice(f32));
        allocator.free(transformer_block.attention.qkv_weight.asSlice(f32));
        allocator.free(transformer_block.attention.o_weight.asSlice(f32));
        const swiglu = transformer_block.ffn_layer.swiglu;
        allocator.free(swiglu.gate_weight.asSlice(f32));
        allocator.free(swiglu.up_weight.asSlice(f32));
        allocator.free(swiglu.down_weight.asSlice(f32));
        transformer_block.weight_registry.deinit();
    }

    const program = [_]LayerOp{
        .{ .linear = .{ .in = .norm_out, .out = .branch_out, .weight_name = "nonexistent_weight" } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    try std.testing.expectError(error.MissingWeight, block.validate());
}

test "validate fails with missing param" {
    const allocator = std.testing.allocator;

    var transformer_block = try createTestBlock(allocator);
    defer {
        allocator.free(transformer_block.ln1.weight.asSlice(f32));
        allocator.free(transformer_block.ln2.weight.asSlice(f32));
        allocator.free(transformer_block.attention.qkv_weight.asSlice(f32));
        allocator.free(transformer_block.attention.o_weight.asSlice(f32));
        const swiglu = transformer_block.ffn_layer.swiglu;
        allocator.free(swiglu.gate_weight.asSlice(f32));
        allocator.free(swiglu.up_weight.asSlice(f32));
        allocator.free(swiglu.down_weight.asSlice(f32));
        transformer_block.weight_registry.deinit();
    }

    const program = [_]LayerOp{
        .{ .add_param = .{ .in = .norm_out, .out = .branch_out, .param_name = "missing_param" } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    try std.testing.expectError(error.MissingParam, block.validate());
}

test "validate fails with invalid split configuration" {
    const allocator = std.testing.allocator;

    var transformer_block = try createTestBlock(allocator);
    defer {
        allocator.free(transformer_block.ln1.weight.asSlice(f32));
        allocator.free(transformer_block.ln2.weight.asSlice(f32));
        allocator.free(transformer_block.attention.qkv_weight.asSlice(f32));
        allocator.free(transformer_block.attention.o_weight.asSlice(f32));
        const swiglu = transformer_block.ffn_layer.swiglu;
        allocator.free(swiglu.gate_weight.asSlice(f32));
        allocator.free(swiglu.up_weight.asSlice(f32));
        allocator.free(swiglu.down_weight.asSlice(f32));
        transformer_block.weight_registry.deinit();
    }

    // Try to split with too many outputs
    const program = [_]LayerOp{
        .{ .split = .{
            .in = .norm_out,
            .out_start = .tmp3,
            .num_outputs = 200, // Too many outputs
            .dim = -1,
        } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    try std.testing.expectError(error.TooManySplitOutputs, block.validate());
}

test "validate succeeds with valid split configuration" {
    const allocator = std.testing.allocator;

    var transformer_block = try createTestBlock(allocator);
    defer {
        allocator.free(transformer_block.ln1.weight.asSlice(f32));
        allocator.free(transformer_block.ln2.weight.asSlice(f32));
        allocator.free(transformer_block.attention.qkv_weight.asSlice(f32));
        allocator.free(transformer_block.attention.o_weight.asSlice(f32));
        const swiglu = transformer_block.ffn_layer.swiglu;
        allocator.free(swiglu.gate_weight.asSlice(f32));
        allocator.free(swiglu.up_weight.asSlice(f32));
        allocator.free(swiglu.down_weight.asSlice(f32));
        transformer_block.weight_registry.deinit();
    }

    // Valid split: 3 outputs for QKV
    const program = [_]LayerOp{
        .{ .split = .{
            .in = .norm_out,
            .out_start = .tmp3,
            .num_outputs = 3,
            .dim = -1,
        } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    try block.validate();
}

// =============================================================================
// Description Method Tests
// =============================================================================

test "describe writes hierarchical block description" {
    const allocator = std.testing.allocator;

    var transformer_block = try createTestBlock(allocator);
    defer {
        allocator.free(transformer_block.ln1.weight.asSlice(f32));
        allocator.free(transformer_block.ln2.weight.asSlice(f32));
        allocator.free(transformer_block.attention.qkv_weight.asSlice(f32));
        allocator.free(transformer_block.attention.o_weight.asSlice(f32));
        const swiglu = transformer_block.ffn_layer.swiglu;
        allocator.free(swiglu.gate_weight.asSlice(f32));
        allocator.free(swiglu.up_weight.asSlice(f32));
        allocator.free(swiglu.down_weight.asSlice(f32));
        transformer_block.weight_registry.deinit();
    }

    const program = [_]LayerOp{
        .{ .norm = .{ .in = .residual, .out = .norm_out, .which = .ln1 } },
        .{ .attn = .{ .in = .norm_out, .out = .branch_out } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 3,
        .hidden_size = 128,
    };

    var output = std.ArrayList(u8).init(allocator);
    defer output.deinit();

    try block.describe(output.writer(), 0, false);

    const result = output.items;

    // Check that output contains expected structure markers
    try std.testing.expect(std.mem.indexOf(u8, result, "(layers.3)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Block(") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(self_attn)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "(ffn)") != null);
}

test "describeTopology writes operation sequence" {
    const allocator = std.testing.allocator;

    var transformer_block = try createTestBlock(allocator);
    defer {
        allocator.free(transformer_block.ln1.weight.asSlice(f32));
        allocator.free(transformer_block.ln2.weight.asSlice(f32));
        allocator.free(transformer_block.attention.qkv_weight.asSlice(f32));
        allocator.free(transformer_block.attention.o_weight.asSlice(f32));
        const swiglu = transformer_block.ffn_layer.swiglu;
        allocator.free(swiglu.gate_weight.asSlice(f32));
        allocator.free(swiglu.up_weight.asSlice(f32));
        allocator.free(swiglu.down_weight.asSlice(f32));
        transformer_block.weight_registry.deinit();
    }

    const program = [_]LayerOp{
        .{ .norm = .{ .in = .residual, .out = .norm_out, .which = .ln1 } },
        .{ .attn = .{ .in = .norm_out, .out = .branch_out } },
        .{ .add = .{ .branch = .branch_out, .scale = .one } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 2,
        .hidden_size = 128,
    };

    var output = std.ArrayList(u8).init(allocator);
    defer output.deinit();

    try block.describeTopology(output.writer(), 0);

    const result = output.items;

    // Check that output contains operation indices and types
    try std.testing.expect(std.mem.indexOf(u8, result, "(layers.2)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "3 ops") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "[0]") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "[1]") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "[2]") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "norm") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "attn") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "residual") != null);
}

test "describeTopology shows different operation types" {
    const allocator = std.testing.allocator;

    var transformer_block = try createTestBlock(allocator);
    defer {
        allocator.free(transformer_block.ln1.weight.asSlice(f32));
        allocator.free(transformer_block.ln2.weight.asSlice(f32));
        allocator.free(transformer_block.attention.qkv_weight.asSlice(f32));
        allocator.free(transformer_block.attention.o_weight.asSlice(f32));
        const swiglu = transformer_block.ffn_layer.swiglu;
        allocator.free(swiglu.gate_weight.asSlice(f32));
        allocator.free(swiglu.up_weight.asSlice(f32));
        allocator.free(swiglu.down_weight.asSlice(f32));
        transformer_block.weight_registry.deinit();
    }

    const program = [_]LayerOp{
        .{ .norm = .{ .in = .residual, .out = .norm_out, .which = .ln1 } },
        .{ .ffn = .{ .in = .norm_out, .out = .branch_out } },
        .{ .add = .{ .branch = .branch_out, .scale = .{ .literal = 0.5 } } },
    };

    const block = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    var output = std.ArrayList(u8).init(allocator);
    defer output.deinit();

    try block.describeTopology(output.writer(), 0);

    const result = output.items;

    // Check FFN operation description
    try std.testing.expect(std.mem.indexOf(u8, result, "ffn") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "MLP") != null);

    // Check scaled residual add
    try std.testing.expect(std.mem.indexOf(u8, result, "0.50") != null);
}

// =============================================================================
// Configuration Tests
// =============================================================================

test "Block can be configured with different hidden sizes" {
    const allocator = std.testing.allocator;

    var transformer_block = try createTestBlock(allocator);
    defer {
        allocator.free(transformer_block.ln1.weight.asSlice(f32));
        allocator.free(transformer_block.ln2.weight.asSlice(f32));
        allocator.free(transformer_block.attention.qkv_weight.asSlice(f32));
        allocator.free(transformer_block.attention.o_weight.asSlice(f32));
        const swiglu = transformer_block.ffn_layer.swiglu;
        allocator.free(swiglu.gate_weight.asSlice(f32));
        allocator.free(swiglu.up_weight.asSlice(f32));
        allocator.free(swiglu.down_weight.asSlice(f32));
        transformer_block.weight_registry.deinit();
    }

    const program = [_]LayerOp{
        .{ .norm = .{ .in = .residual, .out = .norm_out, .which = .ln1 } },
    };

    const block256 = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 256,
    };

    const block512 = Block{
        .program = &program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 512,
    };

    try std.testing.expectEqual(@as(usize, 256), block256.getHiddenSize());
    try std.testing.expectEqual(@as(usize, 512), block512.getHiddenSize());
}

test "Block can be configured with different block indices" {
    const allocator = std.testing.allocator;

    var transformer_block = try createTestBlock(allocator);
    defer {
        allocator.free(transformer_block.ln1.weight.asSlice(f32));
        allocator.free(transformer_block.ln2.weight.asSlice(f32));
        allocator.free(transformer_block.attention.qkv_weight.asSlice(f32));
        allocator.free(transformer_block.attention.o_weight.asSlice(f32));
        const swiglu = transformer_block.ffn_layer.swiglu;
        allocator.free(swiglu.gate_weight.asSlice(f32));
        allocator.free(swiglu.up_weight.asSlice(f32));
        allocator.free(swiglu.down_weight.asSlice(f32));
        transformer_block.weight_registry.deinit();
    }

    const program = [_]LayerOp{
        .{ .norm = .{ .in = .residual, .out = .norm_out, .which = .ln1 } },
    };

    var blocks: [3]Block = undefined;
    for (0..3) |i| {
        blocks[i] = Block{
            .program = &program,
            .block = &transformer_block,
            .block_idx = i,
            .hidden_size = 128,
        };
        try std.testing.expectEqual(i, blocks[i].getBlockIdx());
    }
}

test "Block supports different program lengths" {
    const allocator = std.testing.allocator;

    var transformer_block = try createTestBlock(allocator);
    defer {
        allocator.free(transformer_block.ln1.weight.asSlice(f32));
        allocator.free(transformer_block.ln2.weight.asSlice(f32));
        allocator.free(transformer_block.attention.qkv_weight.asSlice(f32));
        allocator.free(transformer_block.attention.o_weight.asSlice(f32));
        const swiglu = transformer_block.ffn_layer.swiglu;
        allocator.free(swiglu.gate_weight.asSlice(f32));
        allocator.free(swiglu.up_weight.asSlice(f32));
        allocator.free(swiglu.down_weight.asSlice(f32));
        transformer_block.weight_registry.deinit();
    }

    // Short program (2-norm LLaMA-style)
    const short_program = [_]LayerOp{
        .{ .norm = .{ .in = .residual, .out = .norm_out, .which = .ln1 } },
        .{ .attn = .{ .in = .norm_out, .out = .branch_out } },
        .{ .add = .{ .branch = .branch_out, .scale = .one } },
    };

    // Long program (4-norm Gemma-style)
    const long_program = [_]LayerOp{
        .{ .norm = .{ .in = .residual, .out = .norm_out, .which = .ln1 } },
        .{ .attn = .{ .in = .norm_out, .out = .branch_out } },
        .{ .norm = .{ .in = .branch_out, .out = .branch_out, .which = .pre_ffn } },
        .{ .add = .{ .branch = .branch_out, .scale = .one } },
        .{ .norm = .{ .in = .residual, .out = .norm_out, .which = .ln2 } },
        .{ .ffn = .{ .in = .norm_out, .out = .branch_out } },
        .{ .norm = .{ .in = .branch_out, .out = .branch_out, .which = .post_ffn } },
        .{ .add = .{ .branch = .branch_out, .scale = .one } },
    };

    const short_block = Block{
        .program = &short_program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    const long_block = Block{
        .program = &long_program,
        .block = &transformer_block,
        .block_idx = 0,
        .hidden_size = 128,
    };

    try std.testing.expectEqual(@as(usize, 3), short_block.program.len);
    try std.testing.expectEqual(@as(usize, 8), long_block.program.len);
}
