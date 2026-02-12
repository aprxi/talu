//! Tensor Role Mapping
//!
//! Maps SafeTensors tensor names to canonical roles.
//! Single source of truth for weight naming conventions.

const std = @import("std");

/// Canonical tensor roles in a transformer model.
pub const Role = enum {
    // Embeddings
    token_embed,

    // Attention (per-layer)
    attn_q,
    attn_k,
    attn_v,
    attn_o,
    attn_q_norm,
    attn_k_norm,

    // FFN (per-layer)
    ffn_gate,
    ffn_up,
    ffn_down,
    shortconv_in_proj,
    shortconv_out_proj,

    // MoE (per-layer) - Mixture of Experts
    moe_router, // Router/gate weight for expert selection
    moe_expert_gate, // Expert gate projection (SwiGLU)
    moe_expert_up, // Expert up projection (SwiGLU)
    moe_expert_down, // Expert down projection

    // Norms (per-layer)
    attn_norm,
    ffn_norm,
    post_attn_norm, // post_attention_layernorm (4-norm architectures)
    post_ffn_norm, // post_feedforward_layernorm (4-norm architectures)

    // Final
    final_norm,
    lm_head,

    // Unknown - passed through as-is
    unknown,
};

/// Parsed tensor information.
pub const TensorInfo = struct {
    role: Role,
    layer: ?u32, // null for non-layer tensors (embeddings, final norm, lm_head)
    original_name: []const u8, // Original HF name for error messages
};

/// Parse SafeTensors tensor name to canonical role.
/// Handles standard HuggingFace naming conventions.
pub fn parseHfName(name: []const u8) TensorInfo {
    // Non-layer tensors
    if (std.mem.eql(u8, name, "model.embed_tokens.weight")) {
        return .{ .role = .token_embed, .layer = null, .original_name = name };
    }
    if (std.mem.eql(u8, name, "model.norm.weight")) {
        return .{ .role = .final_norm, .layer = null, .original_name = name };
    }
    if (std.mem.eql(u8, name, "model.embedding_norm.weight")) {
        return .{ .role = .final_norm, .layer = null, .original_name = name };
    }
    if (std.mem.eql(u8, name, "lm_head.weight")) {
        return .{ .role = .lm_head, .layer = null, .original_name = name };
    }

    // Layer tensors: model.layers.N.component
    if (std.mem.startsWith(u8, name, "model.layers.")) {
        var parts = std.mem.splitSequence(u8, name, ".");
        _ = parts.next(); // "model"
        _ = parts.next(); // "layers"
        const layer_str = parts.next() orelse return unknownTensorInfo(name);
        const layer_idx = std.fmt.parseInt(u32, layer_str, 10) catch return unknownTensorInfo(name);
        const component = parts.rest();

        const role = parseLayerComponentRole(component);
        return .{ .role = role, .layer = layer_idx, .original_name = name };
    }

    return unknownTensorInfo(name);
}

fn unknownTensorInfo(name: []const u8) TensorInfo {
    return .{ .role = .unknown, .layer = null, .original_name = name };
}

/// Parse the component part of a layer tensor name.
fn parseLayerComponentRole(component: []const u8) Role {
    // Attention projections
    if (std.mem.eql(u8, component, "self_attn.q_proj.weight")) return .attn_q;
    if (std.mem.eql(u8, component, "self_attn.k_proj.weight")) return .attn_k;
    if (std.mem.eql(u8, component, "self_attn.v_proj.weight")) return .attn_v;
    if (std.mem.eql(u8, component, "self_attn.o_proj.weight")) return .attn_o;
    if (std.mem.eql(u8, component, "self_attn.out_proj.weight")) return .attn_o;

    // QK norms (optional)
    if (std.mem.eql(u8, component, "self_attn.q_norm.weight")) return .attn_q_norm;
    if (std.mem.eql(u8, component, "self_attn.k_norm.weight")) return .attn_k_norm;
    if (std.mem.eql(u8, component, "self_attn.q_layernorm.weight")) return .attn_q_norm;
    if (std.mem.eql(u8, component, "self_attn.k_layernorm.weight")) return .attn_k_norm;

    // FFN projections (SwiGLU)
    if (std.mem.eql(u8, component, "mlp.gate_proj.weight")) return .ffn_gate;
    if (std.mem.eql(u8, component, "mlp.up_proj.weight")) return .ffn_up;
    if (std.mem.eql(u8, component, "mlp.down_proj.weight")) return .ffn_down;
    if (std.mem.eql(u8, component, "feed_forward.w1.weight")) return .ffn_gate;
    if (std.mem.eql(u8, component, "feed_forward.w2.weight")) return .ffn_down;
    if (std.mem.eql(u8, component, "feed_forward.w3.weight")) return .ffn_up;

    // Layer norms - standard
    if (std.mem.eql(u8, component, "input_layernorm.weight")) return .attn_norm;
    if (std.mem.eql(u8, component, "post_attention_layernorm.weight")) return .ffn_norm;
    if (std.mem.eql(u8, component, "operator_norm.weight")) return .attn_norm;
    if (std.mem.eql(u8, component, "ffn_norm.weight")) return .ffn_norm;

    // 4-norm architectures
    if (std.mem.eql(u8, component, "pre_feedforward_layernorm.weight")) return .ffn_norm;
    if (std.mem.eql(u8, component, "post_feedforward_layernorm.weight")) return .post_ffn_norm;

    // MoE router (gate)
    if (std.mem.eql(u8, component, "mlp.gate.weight")) return .moe_router;

    // MoE experts (indexed: mlp.experts.N.proj_type.weight)
    if (std.mem.startsWith(u8, component, "mlp.experts.")) {
        if (std.mem.endsWith(u8, component, ".gate_proj.weight")) return .moe_expert_gate;
        if (std.mem.endsWith(u8, component, ".up_proj.weight")) return .moe_expert_up;
        if (std.mem.endsWith(u8, component, ".down_proj.weight")) return .moe_expert_down;
    }

    // ShortConv projections (LFM2/LFM2.5 conv blocks)
    if (std.mem.eql(u8, component, "conv.in_proj.weight")) return .shortconv_in_proj;
    if (std.mem.eql(u8, component, "conv.out_proj.weight")) return .shortconv_out_proj;

    return .unknown;
}

/// Determine if a tensor should be quantized based on its role.
/// Layer norms and small tensors should NOT be quantized.
pub fn shouldQuantize(role: Role) bool {
    return switch (role) {
        // Quantize large weight matrices
        .token_embed, .lm_head => true,
        .attn_q, .attn_k, .attn_v, .attn_o => true,
        .ffn_gate, .ffn_up, .ffn_down => true,
        .shortconv_in_proj, .shortconv_out_proj => true,

        // Quantize MoE expert weights (large matrices)
        .moe_expert_gate, .moe_expert_up, .moe_expert_down => true,

        // Do NOT quantize norms (small, need precision)
        .attn_norm, .ffn_norm, .final_norm => false,
        .post_attn_norm, .post_ffn_norm => false,
        .attn_q_norm, .attn_k_norm => false,

        // Do NOT quantize MoE router (small, need precision)
        .moe_router => false,

        // Unknown - be conservative, don't quantize
        .unknown => false,
    };
}

/// Check if a tensor is the lm_head (output projection).
/// Used for tie_word_embeddings check.
pub fn isLmHead(role: Role) bool {
    return role == .lm_head;
}

// =============================================================================
// Tests
// =============================================================================

test "parseHfName - embeddings" {
    const tensor_info = parseHfName("model.embed_tokens.weight");
    try std.testing.expectEqual(Role.token_embed, tensor_info.role);
    try std.testing.expectEqual(@as(?u32, null), tensor_info.layer);
}

test "parseHfName - layer tensors" {
    {
        const tensor_info = parseHfName("model.layers.0.self_attn.q_proj.weight");
        try std.testing.expectEqual(Role.attn_q, tensor_info.role);
        try std.testing.expectEqual(@as(?u32, 0), tensor_info.layer);
    }
    {
        const tensor_info = parseHfName("model.layers.15.mlp.gate_proj.weight");
        try std.testing.expectEqual(Role.ffn_gate, tensor_info.role);
        try std.testing.expectEqual(@as(?u32, 15), tensor_info.layer);
    }
    {
        const tensor_info = parseHfName("model.layers.7.post_attention_layernorm.weight");
        try std.testing.expectEqual(Role.ffn_norm, tensor_info.role);
        try std.testing.expectEqual(@as(?u32, 7), tensor_info.layer);
    }
}

test "parseHfName - final norm and lm_head" {
    {
        const tensor_info = parseHfName("model.norm.weight");
        try std.testing.expectEqual(Role.final_norm, tensor_info.role);
        try std.testing.expectEqual(@as(?u32, null), tensor_info.layer);
    }
    {
        const tensor_info = parseHfName("lm_head.weight");
        try std.testing.expectEqual(Role.lm_head, tensor_info.role);
        try std.testing.expectEqual(@as(?u32, null), tensor_info.layer);
    }
    {
        const tensor_info = parseHfName("model.embedding_norm.weight");
        try std.testing.expectEqual(Role.final_norm, tensor_info.role);
        try std.testing.expectEqual(@as(?u32, null), tensor_info.layer);
    }
}

test "shouldQuantize" {
    // Large weight matrices - yes
    try std.testing.expect(shouldQuantize(.attn_q));
    try std.testing.expect(shouldQuantize(.ffn_gate));
    try std.testing.expect(shouldQuantize(.token_embed));

    // Norms - no
    try std.testing.expect(!shouldQuantize(.attn_norm));
    try std.testing.expect(!shouldQuantize(.ffn_norm));
    try std.testing.expect(!shouldQuantize(.final_norm));
}

// =============================================================================
// isLmHead Tests
// =============================================================================

test "isLmHead: returns true for lm_head role" {
    try std.testing.expect(isLmHead(.lm_head));
}

test "isLmHead: returns false for all other roles" {
    try std.testing.expect(!isLmHead(.token_embed));
    try std.testing.expect(!isLmHead(.attn_q));
    try std.testing.expect(!isLmHead(.attn_k));
    try std.testing.expect(!isLmHead(.attn_v));
    try std.testing.expect(!isLmHead(.attn_o));
    try std.testing.expect(!isLmHead(.attn_q_norm));
    try std.testing.expect(!isLmHead(.attn_k_norm));
    try std.testing.expect(!isLmHead(.ffn_gate));
    try std.testing.expect(!isLmHead(.ffn_up));
    try std.testing.expect(!isLmHead(.ffn_down));
    try std.testing.expect(!isLmHead(.attn_norm));
    try std.testing.expect(!isLmHead(.ffn_norm));
    try std.testing.expect(!isLmHead(.post_attn_norm));
    try std.testing.expect(!isLmHead(.post_ffn_norm));
    try std.testing.expect(!isLmHead(.final_norm));
    try std.testing.expect(!isLmHead(.unknown));
}

test "isLmHead: usage with parseHfName" {
    const lm_head_info = parseHfName("lm_head.weight");
    try std.testing.expect(isLmHead(lm_head_info.role));

    const embed_info = parseHfName("model.embed_tokens.weight");
    try std.testing.expect(!isLmHead(embed_info.role));

    const attn_info = parseHfName("model.layers.0.self_attn.q_proj.weight");
    try std.testing.expect(!isLmHead(attn_info.role));
}

// =============================================================================
// Additional parseHfName Tests
// =============================================================================

test "parseHfName: pre/post feedforward norms" {
    {
        const info = parseHfName("model.layers.0.pre_feedforward_layernorm.weight");
        try std.testing.expectEqual(Role.ffn_norm, info.role);
        try std.testing.expectEqual(@as(?u32, 0), info.layer);
    }
    {
        const info = parseHfName("model.layers.5.post_feedforward_layernorm.weight");
        try std.testing.expectEqual(Role.post_ffn_norm, info.role);
        try std.testing.expectEqual(@as(?u32, 5), info.layer);
    }
}

test "parseHfName: QK norms" {
    {
        const info = parseHfName("model.layers.10.self_attn.q_norm.weight");
        try std.testing.expectEqual(Role.attn_q_norm, info.role);
        try std.testing.expectEqual(@as(?u32, 10), info.layer);
    }
    {
        const info = parseHfName("model.layers.20.self_attn.k_norm.weight");
        try std.testing.expectEqual(Role.attn_k_norm, info.role);
        try std.testing.expectEqual(@as(?u32, 20), info.layer);
    }
}

test "parseHfName: all attention projections" {
    const layer_idx: u32 = 3;

    {
        const info = parseHfName("model.layers.3.self_attn.q_proj.weight");
        try std.testing.expectEqual(Role.attn_q, info.role);
        try std.testing.expectEqual(@as(?u32, layer_idx), info.layer);
    }
    {
        const info = parseHfName("model.layers.3.self_attn.k_proj.weight");
        try std.testing.expectEqual(Role.attn_k, info.role);
        try std.testing.expectEqual(@as(?u32, layer_idx), info.layer);
    }
    {
        const info = parseHfName("model.layers.3.self_attn.v_proj.weight");
        try std.testing.expectEqual(Role.attn_v, info.role);
        try std.testing.expectEqual(@as(?u32, layer_idx), info.layer);
    }
    {
        const info = parseHfName("model.layers.3.self_attn.o_proj.weight");
        try std.testing.expectEqual(Role.attn_o, info.role);
        try std.testing.expectEqual(@as(?u32, layer_idx), info.layer);
    }
    {
        const info = parseHfName("model.layers.3.self_attn.out_proj.weight");
        try std.testing.expectEqual(Role.attn_o, info.role);
        try std.testing.expectEqual(@as(?u32, layer_idx), info.layer);
    }
}

test "parseHfName: all FFN projections" {
    const layer_idx: u32 = 7;

    {
        const info = parseHfName("model.layers.7.mlp.gate_proj.weight");
        try std.testing.expectEqual(Role.ffn_gate, info.role);
        try std.testing.expectEqual(@as(?u32, layer_idx), info.layer);
    }
    {
        const info = parseHfName("model.layers.7.mlp.up_proj.weight");
        try std.testing.expectEqual(Role.ffn_up, info.role);
        try std.testing.expectEqual(@as(?u32, layer_idx), info.layer);
    }
    {
        const info = parseHfName("model.layers.7.mlp.down_proj.weight");
        try std.testing.expectEqual(Role.ffn_down, info.role);
        try std.testing.expectEqual(@as(?u32, layer_idx), info.layer);
    }
    {
        const info = parseHfName("model.layers.7.feed_forward.w1.weight");
        try std.testing.expectEqual(Role.ffn_gate, info.role);
        try std.testing.expectEqual(@as(?u32, layer_idx), info.layer);
    }
    {
        const info = parseHfName("model.layers.7.feed_forward.w2.weight");
        try std.testing.expectEqual(Role.ffn_down, info.role);
        try std.testing.expectEqual(@as(?u32, layer_idx), info.layer);
    }
    {
        const info = parseHfName("model.layers.7.feed_forward.w3.weight");
        try std.testing.expectEqual(Role.ffn_up, info.role);
        try std.testing.expectEqual(@as(?u32, layer_idx), info.layer);
    }
}

test "parseHfName: LFM2 norms and shortconv projections" {
    {
        const info = parseHfName("model.layers.4.operator_norm.weight");
        try std.testing.expectEqual(Role.attn_norm, info.role);
        try std.testing.expectEqual(@as(?u32, 4), info.layer);
    }
    {
        const info = parseHfName("model.layers.4.ffn_norm.weight");
        try std.testing.expectEqual(Role.ffn_norm, info.role);
        try std.testing.expectEqual(@as(?u32, 4), info.layer);
    }
    {
        const info = parseHfName("model.layers.4.self_attn.q_layernorm.weight");
        try std.testing.expectEqual(Role.attn_q_norm, info.role);
        try std.testing.expectEqual(@as(?u32, 4), info.layer);
    }
    {
        const info = parseHfName("model.layers.4.self_attn.k_layernorm.weight");
        try std.testing.expectEqual(Role.attn_k_norm, info.role);
        try std.testing.expectEqual(@as(?u32, 4), info.layer);
    }
    {
        const info = parseHfName("model.layers.4.conv.in_proj.weight");
        try std.testing.expectEqual(Role.shortconv_in_proj, info.role);
        try std.testing.expectEqual(@as(?u32, 4), info.layer);
    }
    {
        const info = parseHfName("model.layers.4.conv.out_proj.weight");
        try std.testing.expectEqual(Role.shortconv_out_proj, info.role);
        try std.testing.expectEqual(@as(?u32, 4), info.layer);
    }
}

test "parseHfName: unknown tensor names" {
    {
        const info = parseHfName("unknown.tensor.weight");
        try std.testing.expectEqual(Role.unknown, info.role);
        try std.testing.expectEqual(@as(?u32, null), info.layer);
    }
    {
        const info = parseHfName("model.layers.5.something.weight");
        try std.testing.expectEqual(Role.unknown, info.role);
        try std.testing.expectEqual(@as(?u32, 5), info.layer);
    }
    {
        const info = parseHfName("model.custom_component.weight");
        try std.testing.expectEqual(Role.unknown, info.role);
        try std.testing.expectEqual(@as(?u32, null), info.layer);
    }
}

test "parseHfName: edge cases" {
    // Empty string
    {
        const info = parseHfName("");
        try std.testing.expectEqual(Role.unknown, info.role);
    }

    // Invalid layer index
    {
        const info = parseHfName("model.layers.abc.self_attn.q_proj.weight");
        try std.testing.expectEqual(Role.unknown, info.role);
    }

    // Missing components
    {
        const info = parseHfName("model.layers.0");
        try std.testing.expectEqual(Role.unknown, info.role);
    }
}

test "parseHfName: preserves original name" {
    const name = "model.layers.5.self_attn.q_proj.weight";
    const info = parseHfName(name);
    try std.testing.expectEqualStrings(name, info.original_name);
}

// =============================================================================
// shouldQuantize Comprehensive Tests
// =============================================================================

test "shouldQuantize: all weight matrices should be quantized" {
    try std.testing.expect(shouldQuantize(.token_embed));
    try std.testing.expect(shouldQuantize(.lm_head));
    try std.testing.expect(shouldQuantize(.attn_q));
    try std.testing.expect(shouldQuantize(.attn_k));
    try std.testing.expect(shouldQuantize(.attn_v));
    try std.testing.expect(shouldQuantize(.attn_o));
    try std.testing.expect(shouldQuantize(.ffn_gate));
    try std.testing.expect(shouldQuantize(.ffn_up));
    try std.testing.expect(shouldQuantize(.ffn_down));
    try std.testing.expect(shouldQuantize(.shortconv_in_proj));
    try std.testing.expect(shouldQuantize(.shortconv_out_proj));
}

test "shouldQuantize: all norms should NOT be quantized" {
    try std.testing.expect(!shouldQuantize(.attn_norm));
    try std.testing.expect(!shouldQuantize(.ffn_norm));
    try std.testing.expect(!shouldQuantize(.final_norm));
    try std.testing.expect(!shouldQuantize(.post_attn_norm));
    try std.testing.expect(!shouldQuantize(.post_ffn_norm));
    try std.testing.expect(!shouldQuantize(.attn_q_norm));
    try std.testing.expect(!shouldQuantize(.attn_k_norm));
}

test "shouldQuantize: unknown tensors should NOT be quantized" {
    try std.testing.expect(!shouldQuantize(.unknown));
}

test "shouldQuantize: integration with parseHfName" {
    // Should quantize
    {
        const info = parseHfName("model.embed_tokens.weight");
        try std.testing.expect(shouldQuantize(info.role));
    }
    {
        const info = parseHfName("model.layers.0.self_attn.q_proj.weight");
        try std.testing.expect(shouldQuantize(info.role));
    }
    {
        const info = parseHfName("model.layers.5.mlp.gate_proj.weight");
        try std.testing.expect(shouldQuantize(info.role));
    }
    {
        const info = parseHfName("model.layers.5.feed_forward.w1.weight");
        try std.testing.expect(shouldQuantize(info.role));
    }
    {
        const info = parseHfName("model.layers.5.conv.in_proj.weight");
        try std.testing.expect(shouldQuantize(info.role));
    }

    // Should NOT quantize
    {
        const info = parseHfName("model.norm.weight");
        try std.testing.expect(!shouldQuantize(info.role));
    }
    {
        const info = parseHfName("model.layers.3.input_layernorm.weight");
        try std.testing.expect(!shouldQuantize(info.role));
    }
    {
        const info = parseHfName("unknown.tensor.weight");
        try std.testing.expect(!shouldQuantize(info.role));
    }
}

// =============================================================================
// MoE (Mixture of Experts) Tests
// =============================================================================

test "parseHfName: MoE router" {
    const info = parseHfName("model.layers.5.mlp.gate.weight");
    try std.testing.expectEqual(Role.moe_router, info.role);
    try std.testing.expectEqual(@as(?u32, 5), info.layer);
}

test "parseHfName: MoE expert projections" {
    // gate_proj
    {
        const info = parseHfName("model.layers.0.mlp.experts.0.gate_proj.weight");
        try std.testing.expectEqual(Role.moe_expert_gate, info.role);
        try std.testing.expectEqual(@as(?u32, 0), info.layer);
    }
    // up_proj
    {
        const info = parseHfName("model.layers.10.mlp.experts.127.up_proj.weight");
        try std.testing.expectEqual(Role.moe_expert_up, info.role);
        try std.testing.expectEqual(@as(?u32, 10), info.layer);
    }
    // down_proj
    {
        const info = parseHfName("model.layers.47.mlp.experts.63.down_proj.weight");
        try std.testing.expectEqual(Role.moe_expert_down, info.role);
        try std.testing.expectEqual(@as(?u32, 47), info.layer);
    }
}

test "shouldQuantize: MoE expert weights should be quantized" {
    try std.testing.expect(shouldQuantize(.moe_expert_gate));
    try std.testing.expect(shouldQuantize(.moe_expert_up));
    try std.testing.expect(shouldQuantize(.moe_expert_down));
}

test "shouldQuantize: MoE router should NOT be quantized" {
    // Router is small and needs precision for expert selection
    try std.testing.expect(!shouldQuantize(.moe_router));
}

test "shouldQuantize: MoE integration with parseHfName" {
    // Expert weights should be quantized
    {
        const info = parseHfName("model.layers.0.mlp.experts.0.gate_proj.weight");
        try std.testing.expect(shouldQuantize(info.role));
    }
    {
        const info = parseHfName("model.layers.5.mlp.experts.50.down_proj.weight");
        try std.testing.expect(shouldQuantize(info.role));
    }

    // Router should NOT be quantized
    {
        const info = parseHfName("model.layers.3.mlp.gate.weight");
        try std.testing.expect(!shouldQuantize(info.role));
    }
}
