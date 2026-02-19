//! Execution Plan Analysis
//!
//! Static analysis of which kernels will be used for a model based on config.json.
//! This provides visibility into the code paths WITHOUT loading the model.
//!
//! The execution plan tells you:
//! - Which matmul kernel (matmul_f32, matmul_bf16, matmul_grouped_affine_u4, etc.)
//! - Which attention implementation (standard, flash attention, etc.)
//! - Which FFN type (SwiGLU vs MoE)
//! - Which activation (SiLU vs GELU)
//!
//! This enables focused optimization: know exactly which code paths to improve
//! for a given model family.

const std = @import("std");
const dtype_mod = @import("../dtype.zig");
const kernel_info = @import("kernel_info.zig");
const models_registry = @import("../models/registry.zig");

const DType = dtype_mod.DType;

// =============================================================================
// Supported Model Types
// =============================================================================

/// Check if a model type is supported by talu's inference engine.
/// This queries the static models registry used by inference.
pub fn isModelTypeSupported(model_type: ?[]const u8) bool {
    const mt = model_type orelse return false;
    return models_registry.isSupportedModelType(mt);
}

/// Quantization method (matches capi/converter.zig)
pub const QuantMethod = enum(u8) {
    none = 0,
    gaffine = 1, // Grouped affine (MLX-style Q4/Q8)
    mxfp4 = 2, // MXFP4 block floating point
};

/// Model configuration for execution plan analysis.
/// This is the input - typically from talu_describe() or config.json parsing.
pub const ModelConfig = struct {
    // Architecture identification
    model_type: ?[]const u8 = null,

    // Core dimensions
    vocab_size: usize = 0,
    hidden_size: usize = 0,
    num_layers: usize = 0,
    num_heads: usize = 0,
    num_kv_heads: usize = 0,
    intermediate_size: usize = 0,
    head_dim: usize = 0,

    // Quantization
    quant_bits: u8 = 16, // 4, 8, or 16
    quant_group_size: u16 = 64,
    quant_method: QuantMethod = .none,

    // Architecture features
    use_gelu: bool = false,
    tie_word_embeddings: bool = true,

    // MoE
    num_experts: usize = 0,
    experts_per_token: usize = 0,
};

/// Matmul kernel selection
pub const MatmulKernel = enum {
    matmul_f32,
    matmul_f16,
    matmul_bf16,
    matmul_grouped_affine_u4,
    matmul_grouped_affine_u8,
    matmul_mxfp4,

    pub fn name(self: MatmulKernel) [:0]const u8 {
        return switch (self) {
            .matmul_f32 => "matmul_f32",
            .matmul_f16 => "matmul_f16",
            .matmul_bf16 => "matmul_bf16",
            .matmul_grouped_affine_u4 => "matmul_grouped_affine_u4",
            .matmul_grouped_affine_u8 => "matmul_grouped_affine_u8",
            .matmul_mxfp4 => "matmul_mxfp4",
        };
    }

    pub fn weightDtype(self: MatmulKernel) DType {
        return switch (self) {
            .matmul_f32 => .f32,
            .matmul_f16 => .f16,
            .matmul_bf16 => .bf16,
            .matmul_grouped_affine_u4 => .grouped_affine_u4,
            .matmul_grouped_affine_u8 => .grouped_affine_u8,
            .matmul_mxfp4 => .mxfp4,
        };
    }
};

/// FFN type selection
pub const FfnType = enum {
    swiglu_silu,
    swiglu_gelu,
    moe_silu,
    moe_gelu,

    pub fn name(self: FfnType) [:0]const u8 {
        return switch (self) {
            .swiglu_silu => "SwiGLU(SiLU)",
            .swiglu_gelu => "SwiGLU(GELU)",
            .moe_silu => "MoE(SiLU)",
            .moe_gelu => "MoE(GELU)",
        };
    }

    pub fn isMoe(self: FfnType) bool {
        return switch (self) {
            .moe_silu, .moe_gelu => true,
            .swiglu_silu, .swiglu_gelu => false,
        };
    }
};

/// Attention type selection
pub const AttentionType = enum {
    multi_head, // Standard MHA
    grouped_query, // GQA (num_kv_heads < num_heads)
    multi_query, // MQA (num_kv_heads == 1)

    pub fn name(self: AttentionType) [:0]const u8 {
        return switch (self) {
            .multi_head => "MultiHeadAttention",
            .grouped_query => "GroupedQueryAttention",
            .multi_query => "MultiQueryAttention",
        };
    }
};

/// Complete execution plan for a model
pub const ExecutionPlan = struct {
    // Model identification
    model_type: ?[]const u8,

    // Core dimensions (for reference)
    num_layers: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,

    // Kernel selections
    matmul_kernel: MatmulKernel,
    attention_type: AttentionType,
    ffn_type: FfnType,

    // MoE details (if applicable)
    num_experts: usize,
    experts_per_token: usize,

    // Quantization details
    quant_bits: u8,
    quant_group_size: u16,

    // Computed flags
    uses_gqa: bool,
    uses_moe: bool,
    uses_quantization: bool,
    uses_gelu: bool,
    is_supported: bool, // Whether model type is supported by talu

    /// Format execution plan for display
    pub fn format(self: *const ExecutionPlan, writer: anytype) !void {
        try writer.writeAll("╔══════════════════════════════════════════════════════════════╗\n");
        try writer.writeAll("║                     EXECUTION PLAN                           ║\n");
        try writer.writeAll("╠══════════════════════════════════════════════════════════════╣\n");

        // Model type
        if (self.model_type) |mt| {
            try writer.print("║ Model Type:       {s:<42} ║\n", .{mt});
        }

        try writer.writeAll("╠══════════════════════════════════════════════════════════════╣\n");
        try writer.writeAll("║ ARCHITECTURE                                                 ║\n");
        try writer.writeAll("╠══════════════════════════════════════════════════════════════╣\n");
        try writer.print("║ Layers:           {d:<42} ║\n", .{self.num_layers});
        try writer.print("║ Hidden Size:      {d:<42} ║\n", .{self.hidden_size});
        try writer.print("║ Attention Heads:  {d:<42} ║\n", .{self.num_heads});
        try writer.print("║ KV Heads:         {d:<42} ║\n", .{self.num_kv_heads});
        try writer.print("║ Head Dimension:   {d:<42} ║\n", .{self.head_dim});

        try writer.writeAll("╠══════════════════════════════════════════════════════════════╣\n");
        try writer.writeAll("║ KERNEL SELECTION                                             ║\n");
        try writer.writeAll("╠══════════════════════════════════════════════════════════════╣\n");
        try writer.print("║ Matmul:           {s:<42} ║\n", .{self.matmul_kernel.name()});
        try writer.print("║ Attention:        {s:<42} ║\n", .{self.attention_type.name()});
        try writer.print("║ FFN:              {s:<42} ║\n", .{self.ffn_type.name()});

        if (self.uses_moe) {
            try writer.writeAll("╠══════════════════════════════════════════════════════════════╣\n");
            try writer.writeAll("║ MIXTURE OF EXPERTS                                           ║\n");
            try writer.writeAll("╠══════════════════════════════════════════════════════════════╣\n");
            try writer.print("║ Num Experts:      {d:<42} ║\n", .{self.num_experts});
            try writer.print("║ Top-K:            {d:<42} ║\n", .{self.experts_per_token});
        }

        if (self.uses_quantization) {
            try writer.writeAll("╠══════════════════════════════════════════════════════════════╣\n");
            try writer.writeAll("║ QUANTIZATION                                                 ║\n");
            try writer.writeAll("╠══════════════════════════════════════════════════════════════╣\n");
            try writer.print("║ Bits:             {d:<42} ║\n", .{self.quant_bits});
            try writer.print("║ Group Size:       {d:<42} ║\n", .{self.quant_group_size});
        }

        try writer.writeAll("╠══════════════════════════════════════════════════════════════╣\n");
        try writer.writeAll("║ CODE PATHS TO OPTIMIZE                                       ║\n");
        try writer.writeAll("╠══════════════════════════════════════════════════════════════╣\n");

        // List the actual source files/functions that will be hot
        try writer.print("║ • {s:<56} ║\n", .{"compute/cpu/matmul_primitives.zig"});
        try writer.print("║   → {s:<54} ║\n", .{self.matmul_kernel.name()});

        try writer.print("║ • {s:<56} ║\n", .{"inference/backend/cpu/kernels/attention.zig"});
        try writer.print("║   → {s:<54} ║\n", .{self.attention_type.name()});

        if (self.uses_moe) {
            try writer.print("║ • {s:<56} ║\n", .{"inference/backend/cpu/kernels/moe.zig"});
            try writer.print("║   → {s:<54} ║\n", .{self.ffn_type.name()});
        } else {
            try writer.print("║ • {s:<56} ║\n", .{"inference/backend/cpu/kernels/ffn.zig"});
            try writer.print("║   → {s:<54} ║\n", .{self.ffn_type.name()});
        }

        try writer.print("║ • {s:<56} ║\n", .{"inference/backend/cpu/kernels/norm.zig"});
        try writer.print("║   → {s:<54} ║\n", .{"rmsnormForward"});

        try writer.writeAll("╚══════════════════════════════════════════════════════════════╝\n");
    }

    /// Get a summary string (for logging)
    pub fn summary(self: *const ExecutionPlan) [128]u8 {
        var buf: [128]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buf);
        const writer = stream.writer();
        writer.print("{s} | {s} | {s}", .{
            self.matmul_kernel.name(),
            self.attention_type.name(),
            self.ffn_type.name(),
        }) catch {};
        const written = stream.pos;
        @memset(buf[written..], 0);
        return buf;
    }
};

/// Build ModelConfig from C API i32 fields.
/// Handles clamping negative values to zero and enum conversion.
pub fn configFromDescribe(args: struct {
    model_type: ?[]const u8 = null,
    vocab_size: i32 = 0,
    hidden_size: i32 = 0,
    num_layers: i32 = 0,
    num_heads: i32 = 0,
    num_kv_heads: i32 = 0,
    intermediate_size: i32 = 0,
    head_dim: i32 = 0,
    quant_bits: i32 = 0,
    quant_group_size: i32 = 0,
    quant_method: QuantMethod = .none,
    use_gelu: bool = false,
    tie_word_embeddings: bool = true,
    num_experts: i32 = 0,
    experts_per_token: i32 = 0,
}) ModelConfig {
    return .{
        .model_type = args.model_type,
        .vocab_size = @intCast(@max(0, args.vocab_size)),
        .hidden_size = @intCast(@max(0, args.hidden_size)),
        .num_layers = @intCast(@max(0, args.num_layers)),
        .num_heads = @intCast(@max(0, args.num_heads)),
        .num_kv_heads = @intCast(@max(0, args.num_kv_heads)),
        .intermediate_size = @intCast(@max(0, args.intermediate_size)),
        .head_dim = @intCast(@max(0, args.head_dim)),
        .quant_bits = @intCast(@max(0, args.quant_bits)),
        .quant_group_size = @intCast(@max(0, args.quant_group_size)),
        .quant_method = args.quant_method,
        .use_gelu = args.use_gelu,
        .tie_word_embeddings = args.tie_word_embeddings,
        .num_experts = @intCast(@max(0, args.num_experts)),
        .experts_per_token = @intCast(@max(0, args.experts_per_token)),
    };
}

/// Analyze model config and produce execution plan.
/// This is static analysis - no model loading required.
pub fn analyze(config: ModelConfig) ExecutionPlan {
    // Determine matmul kernel from quantization settings
    const matmul_kernel: MatmulKernel = blk: {
        if (config.quant_method == .mxfp4) {
            break :blk .matmul_mxfp4;
        }
        if (config.quant_method == .gaffine) {
            if (config.quant_bits == 4) break :blk .matmul_grouped_affine_u4;
            if (config.quant_bits == 8) break :blk .matmul_grouped_affine_u8;
        }
        // Default to bf16 for non-quantized models
        break :blk .matmul_bf16;
    };

    // Determine attention type from head configuration
    const attention_type: AttentionType = blk: {
        if (config.num_kv_heads == 1) break :blk .multi_query;
        if (config.num_kv_heads < config.num_heads) break :blk .grouped_query;
        break :blk .multi_head;
    };

    // Determine FFN type
    const ffn_type: FfnType = blk: {
        const is_moe = config.num_experts > 0;
        if (is_moe) {
            break :blk if (config.use_gelu) .moe_gelu else .moe_silu;
        }
        break :blk if (config.use_gelu) .swiglu_gelu else .swiglu_silu;
    };

    return ExecutionPlan{
        .model_type = config.model_type,
        .num_layers = config.num_layers,
        .hidden_size = config.hidden_size,
        .num_heads = config.num_heads,
        .num_kv_heads = config.num_kv_heads,
        .head_dim = config.head_dim,
        .matmul_kernel = matmul_kernel,
        .attention_type = attention_type,
        .ffn_type = ffn_type,
        .num_experts = config.num_experts,
        .experts_per_token = config.experts_per_token,
        .quant_bits = config.quant_bits,
        .quant_group_size = config.quant_group_size,
        .uses_gqa = config.num_kv_heads > 0 and config.num_kv_heads < config.num_heads,
        .uses_moe = config.num_experts > 0,
        .uses_quantization = config.quant_bits < 16,
        .uses_gelu = config.use_gelu,
        .is_supported = isModelTypeSupported(config.model_type),
    };
}

// =============================================================================
// Tests
// =============================================================================

test "analyze - bf16 model with GQA" {
    const config = ModelConfig{
        .model_type = "qwen3",
        .num_layers = 28,
        .hidden_size = 1024,
        .num_heads = 16,
        .num_kv_heads = 4,
        .head_dim = 64,
        .intermediate_size = 4096,
        .quant_bits = 16,
        .quant_method = .none,
        .use_gelu = false,
    };

    const plan = analyze(config);

    try std.testing.expectEqual(MatmulKernel.matmul_bf16, plan.matmul_kernel);
    try std.testing.expectEqual(AttentionType.grouped_query, plan.attention_type);
    try std.testing.expectEqual(FfnType.swiglu_silu, plan.ffn_type);
    try std.testing.expect(!plan.uses_quantization);
    try std.testing.expect(plan.uses_gqa);
    try std.testing.expect(!plan.uses_moe);
}

test "analyze - Q4 quantized model" {
    const config = ModelConfig{
        .model_type = "qwen3",
        .num_layers = 28,
        .hidden_size = 1024,
        .num_heads = 16,
        .num_kv_heads = 4,
        .head_dim = 64,
        .quant_bits = 4,
        .quant_group_size = 64,
        .quant_method = .gaffine,
    };

    const plan = analyze(config);

    try std.testing.expectEqual(MatmulKernel.matmul_grouped_affine_u4, plan.matmul_kernel);
    try std.testing.expect(plan.uses_quantization);
}

test "analyze - Q8 quantized model" {
    const config = ModelConfig{
        .model_type = "llama",
        .num_layers = 32,
        .hidden_size = 4096,
        .num_heads = 32,
        .num_kv_heads = 8,
        .head_dim = 128,
        .quant_bits = 8,
        .quant_group_size = 128,
        .quant_method = .gaffine,
    };

    const plan = analyze(config);

    try std.testing.expectEqual(MatmulKernel.matmul_grouped_affine_u8, plan.matmul_kernel);
}

test "analyze - MoE model" {
    const config = ModelConfig{
        .model_type = "mixtral",
        .num_layers = 32,
        .hidden_size = 4096,
        .num_heads = 32,
        .num_kv_heads = 8,
        .head_dim = 128,
        .num_experts = 8,
        .experts_per_token = 2,
        .quant_bits = 16,
    };

    const plan = analyze(config);

    try std.testing.expectEqual(FfnType.moe_silu, plan.ffn_type);
    try std.testing.expect(plan.uses_moe);
    try std.testing.expectEqual(@as(usize, 8), plan.num_experts);
    try std.testing.expectEqual(@as(usize, 2), plan.experts_per_token);
}

test "analyze - GELU activation" {
    const config = ModelConfig{
        .model_type = "phi",
        .num_layers = 24,
        .hidden_size = 2560,
        .num_heads = 32,
        .num_kv_heads = 32,
        .head_dim = 80,
        .use_gelu = true,
    };

    const plan = analyze(config);

    try std.testing.expectEqual(FfnType.swiglu_gelu, plan.ffn_type);
    try std.testing.expect(plan.uses_gelu);
}

test "analyze - MHA (no GQA)" {
    const config = ModelConfig{
        .model_type = "bert",
        .num_layers = 12,
        .hidden_size = 768,
        .num_heads = 12,
        .num_kv_heads = 12, // Same as num_heads
        .head_dim = 64,
    };

    const plan = analyze(config);

    try std.testing.expectEqual(AttentionType.multi_head, plan.attention_type);
    try std.testing.expect(!plan.uses_gqa);
}

test "analyze - MQA (single KV head)" {
    const config = ModelConfig{
        .model_type = "falcon",
        .num_layers = 32,
        .hidden_size = 4096,
        .num_heads = 32,
        .num_kv_heads = 1, // Single KV head
        .head_dim = 128,
    };

    const plan = analyze(config);

    try std.testing.expectEqual(AttentionType.multi_query, plan.attention_type);
}

test "analyze - MXFP4 quantization" {
    const config = ModelConfig{
        .model_type = "qwen",
        .num_layers = 28,
        .hidden_size = 3584,
        .num_heads = 28,
        .num_kv_heads = 4,
        .head_dim = 128,
        .quant_bits = 4,
        .quant_method = .mxfp4,
    };

    const plan = analyze(config);

    try std.testing.expectEqual(MatmulKernel.matmul_mxfp4, plan.matmul_kernel);
}

test "ExecutionPlan.format - produces valid output" {
    const config = ModelConfig{
        .model_type = "test_model",
        .num_layers = 12,
        .hidden_size = 768,
        .num_heads = 12,
        .num_kv_heads = 4,
        .head_dim = 64,
        .quant_bits = 4,
        .quant_group_size = 64,
        .quant_method = .gaffine,
    };

    const plan = analyze(config);

    var buf: [4096]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    try plan.format(stream.writer());

    const output = stream.getWritten();
    try std.testing.expect(output.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, output, "EXECUTION PLAN") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "matmul_grouped_affine_u4") != null);
}

test "ExecutionPlan.summary - produces compact string" {
    const config = ModelConfig{
        .model_type = "llama",
        .num_layers = 32,
        .hidden_size = 4096,
        .num_heads = 32,
        .num_kv_heads = 8,
        .head_dim = 128,
    };

    const plan = analyze(config);
    const summary = plan.summary();

    // Check that it's null-terminated
    const len = std.mem.indexOfScalar(u8, &summary, 0) orelse summary.len;
    try std.testing.expect(len > 0);
    try std.testing.expect(len < 128);
}

test "MatmulKernel.name - all variants" {
    try std.testing.expectEqualStrings("matmul_f32", MatmulKernel.matmul_f32.name());
    try std.testing.expectEqualStrings("matmul_f16", MatmulKernel.matmul_f16.name());
    try std.testing.expectEqualStrings("matmul_bf16", MatmulKernel.matmul_bf16.name());
    try std.testing.expectEqualStrings("matmul_grouped_affine_u4", MatmulKernel.matmul_grouped_affine_u4.name());
    try std.testing.expectEqualStrings("matmul_grouped_affine_u8", MatmulKernel.matmul_grouped_affine_u8.name());
    try std.testing.expectEqualStrings("matmul_mxfp4", MatmulKernel.matmul_mxfp4.name());
}

test "FfnType.name - all variants" {
    try std.testing.expectEqualStrings("SwiGLU(SiLU)", FfnType.swiglu_silu.name());
    try std.testing.expectEqualStrings("SwiGLU(GELU)", FfnType.swiglu_gelu.name());
    try std.testing.expectEqualStrings("MoE(SiLU)", FfnType.moe_silu.name());
    try std.testing.expectEqualStrings("MoE(GELU)", FfnType.moe_gelu.name());
}

test "AttentionType.name - all variants" {
    try std.testing.expectEqualStrings("MultiHeadAttention", AttentionType.multi_head.name());
    try std.testing.expectEqualStrings("GroupedQueryAttention", AttentionType.grouped_query.name());
    try std.testing.expectEqualStrings("MultiQueryAttention", AttentionType.multi_query.name());
}
