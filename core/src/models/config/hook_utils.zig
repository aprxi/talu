const std = @import("std");
const tensor = @import("../../tensor.zig");

pub fn getObjectIntField(obj: std.json.ObjectMap, key: []const u8) ?i32 {
    const value = obj.get(key) orelse return null;
    return switch (value) {
        .integer => std.math.cast(i32, value.integer),
        else => null,
    };
}

pub fn getObjectFloatField(obj: std.json.ObjectMap, key: []const u8) ?f32 {
    const value = obj.get(key) orelse return null;
    return switch (value) {
        .float => @floatCast(value.float),
        .integer => @floatFromInt(value.integer),
        else => null,
    };
}

pub fn getObjectBoolField(obj: std.json.ObjectMap, key: []const u8) ?bool {
    const value = obj.get(key) orelse return null;
    return switch (value) {
        .bool => value.bool,
        else => null,
    };
}

pub fn getObjectFirstIntField(obj: std.json.ObjectMap, keys: []const []const u8) ?i32 {
    for (keys) |key| {
        if (getObjectIntField(obj, key)) |value| return value;
    }
    return null;
}

pub fn getIntFromConfigOrRoot(config_obj: std.json.ObjectMap, root_obj: std.json.ObjectMap, key: []const u8) ?i32 {
    return getObjectIntField(config_obj, key) orelse getObjectIntField(root_obj, key);
}

pub fn getFloatFromConfigOrRoot(config_obj: std.json.ObjectMap, root_obj: std.json.ObjectMap, key: []const u8) ?f32 {
    return getObjectFloatField(config_obj, key) orelse getObjectFloatField(root_obj, key);
}

pub fn getBoolFromConfigOrRoot(config_obj: std.json.ObjectMap, root_obj: std.json.ObjectMap, key: []const u8) ?bool {
    return getObjectBoolField(config_obj, key) orelse getObjectBoolField(root_obj, key);
}

pub fn applyCommonTextConfig(config_obj: std.json.ObjectMap, root_obj: std.json.ObjectMap, config: *tensor.ModelConfig) void {
    if (getBoolFromConfigOrRoot(config_obj, root_obj, "attention_bias")) |v| config.attention_bias = v;
    if (getBoolFromConfigOrRoot(config_obj, root_obj, "use_qk_norm")) |v| config.use_qk_norm = v;
    if (getFloatFromConfigOrRoot(config_obj, root_obj, "query_pre_attn_scalar")) |v| config.query_pre_attn_scalar = v;
    if (getFloatFromConfigOrRoot(config_obj, root_obj, "rope_local_base_freq")) |v| config.rope_local_theta = v;
    if (getIntFromConfigOrRoot(config_obj, root_obj, "sliding_window")) |v| config.sliding_window = v;
    if (getIntFromConfigOrRoot(config_obj, root_obj, "sliding_window_pattern")) |v| config.sliding_window_pattern = v;
    if (getFloatFromConfigOrRoot(config_obj, root_obj, "embedding_multiplier")) |v| config.embedding_multiplier = v;
    if (getFloatFromConfigOrRoot(config_obj, root_obj, "attention_multiplier")) |v| config.attention_multiplier = v;
    if (getFloatFromConfigOrRoot(config_obj, root_obj, "residual_multiplier")) |v| config.residual_multiplier = v;
    if (getFloatFromConfigOrRoot(config_obj, root_obj, "logits_scaling")) |v| config.logits_scaling = v;
    if (getIntFromConfigOrRoot(config_obj, root_obj, "bos_token_id")) |v| config.bos_token_id = v;
}

pub fn applyCommonTextConfigHook(
    config_obj: std.json.ObjectMap,
    root_obj: std.json.ObjectMap,
    config: *tensor.ModelConfig,
) void {
    applyCommonTextConfig(config_obj, root_obj, config);
}

pub fn applyMambaConfig(config_obj: std.json.ObjectMap, root_obj: std.json.ObjectMap, config: *tensor.ModelConfig) void {
    if (getIntFromConfigOrRoot(config_obj, root_obj, "mamba_d_state")) |v| config.mamba_d_state = v;
    if (getIntFromConfigOrRoot(config_obj, root_obj, "mamba_d_conv")) |v| config.mamba_d_conv = v;
    if (getIntFromConfigOrRoot(config_obj, root_obj, "mamba_n_heads")) |v| config.mamba_n_heads = v;
    if (getIntFromConfigOrRoot(config_obj, root_obj, "mamba_d_head")) |v| config.mamba_d_head = v;
    if (getIntFromConfigOrRoot(config_obj, root_obj, "mamba_n_groups")) |v| config.mamba_n_groups = v;
    if (getIntFromConfigOrRoot(config_obj, root_obj, "mamba_expand")) |v| config.mamba_expand = v;
}

pub fn applyShortConvConfig(config_obj: std.json.ObjectMap, root_obj: std.json.ObjectMap, config: *tensor.ModelConfig) void {
    if (getIntFromConfigOrRoot(config_obj, root_obj, "conv_L_cache")) |v| config.shortconv_d_conv = v;
    if (getIntFromConfigOrRoot(config_obj, root_obj, "conv_dim")) |v| config.shortconv_conv_dim = v;
    if (getIntFromConfigOrRoot(config_obj, root_obj, "conv_dim_out")) |v| config.shortconv_conv_dim_out = v;
    if (getBoolFromConfigOrRoot(config_obj, root_obj, "conv_bias")) |v| config.shortconv_has_bias = v;
}

pub fn applyPhiPartialRotary(config_obj: std.json.ObjectMap, root_obj: std.json.ObjectMap, config: *tensor.ModelConfig) void {
    if (getFloatFromConfigOrRoot(config_obj, root_obj, "partial_rotary_factor")) |partial_rotary_factor| {
        config.rope_dim = @intFromFloat(@as(f32, @floatFromInt(config.head_dim)) * partial_rotary_factor);
    }
}

pub fn applyVisionConfig(root_obj: std.json.ObjectMap, config: *tensor.ModelConfig) void {
    const vision_obj = if (root_obj.get("vision_config")) |vision_cfg|
        switch (vision_cfg) {
            .object => vision_cfg.object,
            else => null,
        }
    else
        null;

    if (vision_obj) |obj| {
        config.vision_hidden_size = getObjectIntField(obj, "hidden_size") orelse config.vision_hidden_size;
        config.vision_depth = getObjectFirstIntField(obj, &.{ "depth", "num_hidden_layers" }) orelse config.vision_depth;
        config.vision_num_heads = getObjectFirstIntField(obj, &.{ "num_heads", "num_attention_heads" }) orelse config.vision_num_heads;
        config.vision_intermediate_size = getObjectIntField(obj, "intermediate_size") orelse config.vision_intermediate_size;
        config.vision_out_hidden_size = getObjectIntField(obj, "out_hidden_size") orelse config.vision_out_hidden_size;
        config.vision_patch_size = getObjectIntField(obj, "patch_size") orelse config.vision_patch_size;
        config.vision_spatial_merge_size = getObjectIntField(obj, "spatial_merge_size") orelse
            getObjectIntField(root_obj, "downsample_factor") orelse
            1;
        config.vision_temporal_patch_size = getObjectIntField(obj, "temporal_patch_size") orelse 1;
        config.vision_num_position_embeddings = getObjectFirstIntField(obj, &.{ "num_position_embeddings", "num_patches" }) orelse config.vision_num_position_embeddings;
        config.vision_max_num_patches = getObjectIntField(root_obj, "max_num_patches") orelse config.vision_num_position_embeddings;

        var probe_layers: [8]u16 = [_]u16{0} ** 8;
        var probe_layer_count: u8 = 0;
        if (obj.get("deepstack_visual_indexes")) |raw_layers| {
            if (raw_layers == .array) {
                for (raw_layers.array.items) |item| {
                    if (probe_layer_count >= probe_layers.len) break;
                    if (item != .integer or item.integer < 0) continue;
                    const casted = std.math.cast(u16, item.integer) orelse continue;
                    probe_layers[probe_layer_count] = casted;
                    probe_layer_count += 1;
                }
            }
        }
        config.vision_probe_layer_count = probe_layer_count;
        config.vision_probe_layers = probe_layers;
    }

    if (getObjectIntField(root_obj, "projector_hidden_size")) |v| config.projector_hidden_size = v;
    if (getObjectFirstIntField(root_obj, &.{ "image_token_id", "image_token_index" })) |v| config.image_token_id = v;
    if (getObjectFirstIntField(root_obj, &.{ "vision_start_token_id", "image_start_token_id" })) |v| config.vision_start_token_id = v;
    if (getObjectFirstIntField(root_obj, &.{ "vision_end_token_id", "image_end_token_id" })) |v| config.vision_end_token_id = v;
}
