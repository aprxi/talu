//! Runtime opcode definitions for compiled model execution plans.
//!
//! This module is intentionally backend-agnostic.

/// Runtime opcode enum.
/// Values are stable for adapter-table indexing.
pub const Opcode = enum(u8) {
    // Macro-ops (v1 primary path)
    rmsnorm = 0,
    multihead_attention = 1,
    swiglu = 2,
    moe = 3,
    mamba_mixer = 4,
    shortconv = 5,
    mla_attention = 6,
    embedding = 7,

    // Structural
    residual_add = 8,

    // Vision pipeline
    vision_patch_embed = 16,
    vision_spatial_merge = 17,
    vision_deepstack_extract = 18,
    vision_scatter = 19,

    // Primitive ops (v1 compatibility path for existing LayerOp programs)
    // Keep 32..63 reserved for future macro-ops.
    mul = 64,
    mean = 65,
    pow = 66,
    rsqrt = 67,
    matmul = 68,
    split = 69,
    transpose = 70,
    reshape = 71,
    softmax = 72,
    silu = 73,
    gelu = 74,
    linear = 75,
    rope = 76,
    triu = 77,
    scaled_dot_product_attention = 78,
    add_tensor = 79,
    add_scalar = 80,
    mul_scalar = 81,
    add_param = 82,
    add_param_scalar = 83,
    mul_param = 84,

    // Reserved 85..127 for future fine-grained primitives/extensions.
    // Reserved 128..255 for backend-local/experimental opcodes
    _,
};

pub fn isVision(opcode: Opcode) bool {
    return switch (opcode) {
        .vision_patch_embed, .vision_spatial_merge, .vision_deepstack_extract, .vision_scatter => true,
        else => false,
    };
}

pub fn isMacro(opcode: Opcode) bool {
    return switch (opcode) {
        .rmsnorm,
        .multihead_attention,
        .swiglu,
        .moe,
        .mamba_mixer,
        .shortconv,
        .mla_attention,
        .embedding,
        .residual_add,
        => true,
        else => false,
    };
}

pub fn isPrimitive(opcode: Opcode) bool {
    return !isMacro(opcode) and !isVision(opcode);
}
