//! Layer Operation Types
//!
//! Defines the bytecode format for transformer block execution.
//! Each LayerOp represents a single operation in the execution flow.

const op_types = @import("op_types.zig");

/// Buffer slots for layer operation operands.
/// Maps to physical scratch buffers via array indexing: scratch.tmp[@intFromEnum(id)].
/// Access scratch buffers via ScratchBuffer.getTmp(id, len).
pub const BufferId = enum(u6) {
    /// The residual stream (input/output). NOT in scratch.tmp - uses model output buffer.
    residual = 0,
    /// Post-normalization buffer. Maps to scratch.tmp[1].
    norm_out = 1,
    /// Attention/FFN output buffer. Maps to scratch.tmp[2].
    branch_out = 2,

    // Extended slots for primitive-based execution
    tmp3 = 3, // For split outputs, intermediate results
    tmp4 = 4,
    tmp5 = 5,
    tmp6 = 6,
    tmp7 = 7,
    tmp8 = 8,
    tmp9 = 9,
    tmp10 = 10,
    tmp11 = 11,
    tmp12 = 12,
    tmp13 = 13,
    tmp14 = 14,
    tmp15 = 15,
    tmp16 = 16,
    tmp17 = 17,
    tmp18 = 18,
    tmp19 = 19,
    tmp20 = 20,
    tmp21 = 21,
    tmp22 = 22,
    tmp23 = 23,
    tmp24 = 24,
    tmp25 = 25,
    tmp26 = 26,
    tmp27 = 27,
    tmp28 = 28,
    tmp29 = 29,
    tmp30 = 30,
    tmp31 = 31,
    tmp32 = 32,
    tmp33 = 33,
    tmp34 = 34,
    tmp35 = 35,
    tmp36 = 36,
    tmp37 = 37,
    tmp38 = 38,
    tmp39 = 39,
    tmp40 = 40,
    tmp41 = 41,
    tmp42 = 42,
    tmp43 = 43,
    tmp44 = 44,
    tmp45 = 45,
    tmp46 = 46,
    tmp47 = 47,
    tmp48 = 48,
    tmp49 = 49,
    tmp50 = 50,
    tmp51 = 51,
    tmp52 = 52,
    tmp53 = 53,
    tmp54 = 54,
    tmp55 = 55,
    tmp56 = 56,
    tmp57 = 57,
    tmp58 = 58,
    tmp59 = 59,
    tmp60 = 60,
    tmp61 = 61,
    tmp62 = 62,
    tmp63 = 63,
};

/// Scaling mode for residual add.
pub const ResidualScale = union(enum) {
    one,
    residual_multiplier,
    literal: f32,
};

/// A single layer operation - the "bytecode" for transformer blocks.
/// Each Block contains a sequence of these ops defining its execution flow.
pub const LayerOp = union(enum) {
    /// Generic kernel invocation (norm/attention/mamba/ffn).
    /// The kernel id is an index into the block's kernel list.
    kernel: struct {
        id: u32,
        in: BufferId,
        out: BufferId,
        /// Expected operation type (debug validation only).
        /// Set by compiler, checked by executor in debug builds.
        debug_type: op_types.OpType = .norm,
    },

    /// residual += branch * scale
    add: struct {
        branch: BufferId,
        scale: ResidualScale,
    },

    // =========================================================================
    // Low-level primitive ops (for custom attention/MLP implementations)
    // =========================================================================

    /// y = x @ weight (linear projection)
    linear: struct {
        in: BufferId,
        out: BufferId,
        weight_name: []const u8, // e.g., "qkv_proj", "o_proj"
    },

    /// y = matmul(a, b)
    matmul: struct {
        in_a: BufferId,
        in_b: BufferId,
        out: BufferId,
    },

    /// Split tensor into multiple outputs
    split: struct {
        in: BufferId,
        out_start: BufferId, // First output buffer
        num_outputs: u8,
        dim: i8,
        split_sizes: []const usize = &.{}, // Sizes of each output (empty = equal split)
    },

    /// y = softmax(x, dim)
    softmax: struct {
        in: BufferId,
        out: BufferId,
        dim: i8,
    },

    /// y = silu(x)
    silu: struct {
        in: BufferId,
        out: BufferId,
    },

    /// y = gelu(x) - Gaussian Error Linear Unit
    gelu: struct {
        in: BufferId,
        out: BufferId,
    },

    /// y = x * scale (element-wise multiply by scalar or tensor)
    mul: struct {
        in: BufferId,
        other: BufferId, // Can be same as in for scalar mult
        out: BufferId,
    },

    /// y = x + y (element-wise add)
    add_tensor: struct {
        in_a: BufferId,
        in_b: BufferId,
        out: BufferId,
    },

    /// y = x + scalar (element-wise add)
    add_scalar: struct {
        in: BufferId,
        out: BufferId,
        scalar: f32,
    },

    /// y = x * scalar (element-wise multiply)
    mul_scalar: struct {
        in: BufferId,
        out: BufferId,
        scalar: f32,
    },

    /// y = mean(x, dim)
    mean: struct {
        in: BufferId,
        out: BufferId,
        dim: i8,
        keepdim: bool,
    },

    /// y = pow(x, exponent)
    pow: struct {
        in: BufferId,
        out: BufferId,
        exponent: f32,
    },

    /// y = rsqrt(x)
    rsqrt: struct {
        in: BufferId,
        out: BufferId,
    },

    /// y = x + param (element-wise add with a parameter tensor)
    add_param: struct {
        in: BufferId,
        out: BufferId,
        param_name: []const u8,
    },

    /// y = param + scalar (element-wise add with a parameter tensor)
    add_param_scalar: struct {
        out: BufferId,
        param_name: []const u8,
        scalar: f32,
    },

    /// y = x * param (element-wise multiply with a parameter tensor)
    mul_param: struct {
        in: BufferId,
        out: BufferId,
        param_name: []const u8,
    },

    /// y = reshape(x, shape) - view operation, no data copy
    reshape: struct {
        in: BufferId,
        out: BufferId,
        shape: []const i32 = &.{}, // Target shape (-1 for infer)
    },

    /// y = transpose(x, dim0, dim1)
    transpose: struct {
        in: BufferId,
        out: BufferId,
        dim0: i8,
        dim1: i8,
    },

    /// y = rope(x) - apply rotary position embedding
    rope: struct {
        in: BufferId,
        out: BufferId,
    },

    /// y = triu(x, diagonal) - upper triangular mask
    triu: struct {
        in: BufferId,
        out: BufferId,
        diagonal: i32 = 0, // Offset from main diagonal
    },

    /// Scaled dot-product attention (PyTorch-compatible 4D layout)
    /// Computes: softmax(Q @ K.T / sqrt(d_k) + mask) @ V
    /// Q/K/V: [batch, heads, seq, head_dim] (PyTorch layout)
    /// Output: [batch, heads, seq, head_dim]
    sdpa: struct {
        q: BufferId,
        k: BufferId,
        v: BufferId,
        out: BufferId,
        is_causal: bool = false,
        scale: ?f32 = null, // If null, uses 1/sqrt(head_dim)
    },

    // =========================================================================
    // Vision pipeline ops (Phase F groundwork)
    // =========================================================================

    /// Vision patch embedding projection from image/patch input.
    patch_embed: struct {
        in: BufferId,
        out: BufferId,
    },

    /// Spatial merge/reduction over vision token grid.
    spatial_merge: struct {
        in: BufferId,
        out: BufferId,
        merge_size: u32,
    },

    /// Extract per-layer deepstack features from vision hidden state.
    deepstack_extract: struct {
        in: BufferId,
        out: BufferId,
        layer_index: u32,
    },

    /// Scatter vision features into text token stream.
    scatter: struct {
        text_in: BufferId,
        vision_in: BufferId,
        out: BufferId,
        image_token_id: u32,
    },

    // Note: model-specific kernels (norm/attn/ffn/mamba) are now emitted as `.kernel`.
};

/// Returns the output buffer produced by the last operation in a LayerOp program.
/// Empty programs default to `.residual`.
pub fn finalOutputBuffer(program: []const LayerOp) BufferId {
    if (program.len == 0) return .residual;
    const last = program[program.len - 1];
    return switch (last) {
        .kernel => |k| k.out,
        .add => .residual,
        .linear => |op| op.out,
        .matmul => |op| op.out,
        .softmax => |op| op.out,
        .silu => |op| op.out,
        .gelu => |op| op.out,
        .mul => |op| op.out,
        .add_tensor => |op| op.out,
        .add_scalar => |op| op.out,
        .mul_scalar => |op| op.out,
        .mean => |op| op.out,
        .pow => |op| op.out,
        .rsqrt => |op| op.out,
        .add_param => |op| op.out,
        .add_param_scalar => |op| op.out,
        .mul_param => |op| op.out,
        .reshape => |op| op.out,
        .transpose => |op| op.out,
        .rope => |op| op.out,
        .triu => |op| op.out,
        .sdpa => |op| op.out,
        .patch_embed => |op| op.out,
        .spatial_merge => |op| op.out,
        .deepstack_extract => |op| op.out,
        .scatter => |op| op.out,
        else => .residual,
    };
}

/// Returns true when the buffer id is one of the canonical runtime core buffers.
pub fn isCoreProgramBuffer(id: BufferId) bool {
    return switch (id) {
        .residual, .norm_out, .branch_out => true,
        else => false,
    };
}

test "finalOutputBuffer defaults to residual for empty program" {
    const testing = @import("std").testing;
    try testing.expectEqual(BufferId.residual, finalOutputBuffer(&.{}));
}

test "finalOutputBuffer resolves last op output for vision scatter" {
    const testing = @import("std").testing;
    const program = [_]LayerOp{
        .{ .patch_embed = .{ .in = .residual, .out = .tmp3 } },
        .{ .scatter = .{ .text_in = .residual, .vision_in = .tmp3, .out = .branch_out, .image_token_id = 42 } },
    };
    try testing.expectEqual(BufferId.branch_out, finalOutputBuffer(&program));
}

test "isCoreProgramBuffer accepts residual/norm_out/branch_out only" {
    const testing = @import("std").testing;
    try testing.expect(isCoreProgramBuffer(.residual));
    try testing.expect(isCoreProgramBuffer(.norm_out));
    try testing.expect(isCoreProgramBuffer(.branch_out));
    try testing.expect(!isCoreProgramBuffer(.tmp3));
}
