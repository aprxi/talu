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
    mul = 32,
    mean = 33,
    pow = 34,
    rsqrt = 35,
    matmul = 36,
    split = 37,
    transpose = 38,
    reshape = 39,
    softmax = 40,
    silu = 41,
    gelu = 42,
    linear = 43,
    rope = 44,
    triu = 45,
    scaled_dot_product_attention = 46,
    add_tensor = 47,
    add_scalar = 48,
    mul_scalar = 49,
    add_param = 50,
    add_param_scalar = 51,
    mul_param = 52,

    // Reserved 64..127 for future fine-grained primitives
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
