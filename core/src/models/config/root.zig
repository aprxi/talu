//! Model configuration parsing for HuggingFace config.json files.
//!
//! Handles parsing of model configuration including architecture detection,
//! RoPE scaling parameters, quantization settings, and tied embeddings.

const std = @import("std");
const json = @import("../../io/json/root.zig");
const tensor = @import("../../tensor.zig");
const model_types = @import("../op_types.zig");
const registry = @import("../registry.zig");

const ModelConfig = tensor.ModelConfig;

// =============================================================================
// Rope Scaling Parsing
// =============================================================================

/// Parse rope_scaling from a JSON object.
pub fn parseRopeScalingFromObject(obj: std.json.ObjectMap) tensor.RopeScaling {
    var rope_type: @TypeOf((tensor.RopeScaling{}).rope_type) = .none;
    if (obj.get("rope_type")) |rtv| {
        if (rtv == .string) {
            if (std.mem.eql(u8, rtv.string, "llama3")) rope_type = .llama3;
            if (std.mem.eql(u8, rtv.string, "linear")) rope_type = .linear;
            if (std.mem.eql(u8, rtv.string, "yarn")) rope_type = .yarn;
        }
    }

    var mrope_section: [3]u32 = .{ 0, 0, 0 };
    if (obj.get("mrope_section")) |section_value| {
        if (section_value == .array and section_value.array.items.len == 3) {
            for (0..3) |idx| {
                const value = section_value.array.items[idx];
                if (value == .integer) {
                    mrope_section[idx] = std.math.cast(u32, value.integer) orelse 0;
                }
            }
        }
    }

    return .{
        .rope_type = rope_type,
        .factor = getFloatField(obj, "factor") orelse 1.0,
        .low_freq_factor = getFloatField(obj, "low_freq_factor") orelse
            getFloatField(obj, "beta_slow") orelse 1.0,
        .high_freq_factor = getFloatField(obj, "high_freq_factor") orelse
            getFloatField(obj, "beta_fast") orelse 4.0,
        .beta_slow = getFloatField(obj, "beta_slow") orelse 1.0,
        .beta_fast = getFloatField(obj, "beta_fast") orelse 32.0,
        .attention_factor = getFloatField(obj, "attention_factor") orelse 0.0,
        .mscale = getFloatField(obj, "mscale") orelse 0.0,
        .mscale_all_dim = getFloatField(obj, "mscale_all_dim") orelse 0.0,
        .truncate = getBoolField(obj, "truncate") orelse true,
        .original_max_position_embeddings = if (obj.get("original_max_position_embeddings")) |v|
            (if (v == .integer) @as(i32, @intCast(v.integer)) else 8192)
        else
            8192,
        .mrope_section = mrope_section,
        .mrope_interleaved = getBoolField(obj, "mrope_interleaved") orelse false,
    };
}

/// Helper to extract a float from a JSON value (handles both float and integer).
fn getFloatField(obj: std.json.ObjectMap, key: []const u8) ?f32 {
    const value = obj.get(key) orelse return null;
    return switch (value) {
        .float => @floatCast(value.float),
        .integer => @floatFromInt(value.integer),
        else => null,
    };
}

fn getBoolField(obj: std.json.ObjectMap, key: []const u8) ?bool {
    const value = obj.get(key) orelse return null;
    return switch (value) {
        .bool => value.bool,
        else => null,
    };
}

/// Quantization config - supports both MLX and MXFP4 formats
const QuantConfig = struct {
    group_size: ?i64 = null,
    bits: ?i64 = null,
    quant_method: ?[]const u8 = null, // "mxfp4", etc.
    mode: ?[]const u8 = null, // Alternative to quant_method used by some models
};

/// Core JSON config struct for shared text-model fields.
/// Model-family-specific fields are parsed via architecture hooks.
const JsonConfig = struct {
    vocab_size: ?i64 = null,
    // Model dimension
    d_model: ?i64 = null,
    hidden_size: ?i64 = null,
    // Layers
    n_layers: ?i64 = null,
    num_layers: ?i64 = null,
    num_hidden_layers: ?i64 = null,
    // Attention heads
    n_heads: ?i64 = null,
    num_heads: ?i64 = null,
    num_attention_heads: ?i64 = null,
    // KV heads
    n_kv_groups: ?i64 = null,
    num_key_value_heads: ?i64 = null,
    // FFN dimension
    d_ff: ?i64 = null,
    intermediate_size: ?i64 = null,
    block_ff_dim: ?i64 = null, // Alternate FFN dim (requires SwiGLU adjustment)
    // SwiGLU adjustment fields (used with block_ff_dim)
    block_use_swiglu: ?bool = null,
    block_multiple_of: ?i64 = null,
    block_ffn_dim_multiplier: ?f64 = null,
    // Max sequence length
    max_seq_len: ?i64 = null,
    context_length: ?i64 = null,
    max_position_embeddings: ?i64 = null,
    // Head dimension
    head_dim: ?i64 = null,
    // RoPE
    rope_base: ?f64 = null,
    rope_theta: ?f64 = null,
    // RoPE scaling (Llama3-style / YaRN-style)
    rope_scaling: ?struct {
        rope_type: ?[]const u8 = null,
        factor: ?f64 = null,
        low_freq_factor: ?f64 = null,
        high_freq_factor: ?f64 = null,
        // YaRN naming (maps to low/high frequency factors)
        beta_slow: ?f64 = null,
        beta_fast: ?f64 = null,
        attention_factor: ?f64 = null,
        mscale: ?f64 = null,
        mscale_all_dim: ?f64 = null,
        truncate: ?bool = null,
        original_max_position_embeddings: ?i64 = null,
        mrope_section: ?[]const i64 = null,
        mrope_interleaved: ?bool = null,
    } = null,
    // RoPE parameters (Mistral3/YARN-style)
    rope_parameters: ?struct {
        rope_theta: ?f64 = null,
        rope_type: ?[]const u8 = null,
        factor: ?f64 = null,
        original_max_position_embeddings: ?i64 = null,
    } = null,
    // Norm epsilon
    norm_eps: ?f64 = null,
    rms_norm_eps: ?f64 = null,
    // Quantization
    quantization: ?struct { group_size: ?i64 = null, bits: ?i64 = null, mode: ?[]const u8 = null } = null,
    quantization_config: ?QuantConfig = null,
    // Tied embeddings (lm_head shares weights with embed_tokens)
    tie_word_embeddings: ?bool = null,
    // MoE (Mixture of Experts) config
    num_local_experts: ?i64 = null,
    num_experts: ?i64 = null,
    num_experts_per_tok: ?i64 = null,
    experts_per_token: ?i64 = null, // Alias used by some models
    moe_intermediate_size: ?i64 = null,
    // Activation
    hidden_activation: ?[]const u8 = null, // "silu" or "gelu_pytorch_tanh"

    /// Get first non-null integer from a list of field names
    fn firstInt(self: @This(), comptime fields: anytype) ?i32 {
        inline for (fields) |f| if (@field(self, f)) |v| return @intCast(v);
        return null;
    }

    /// Get first non-null float from a list of field names
    fn firstFloat(self: @This(), comptime fields: anytype) ?f32 {
        inline for (fields) |f| if (@field(self, f)) |v| return @floatCast(v);
        return null;
    }

    /// Get integer field with default
    fn intOrDefault(self: @This(), comptime field: []const u8, default: i32) i32 {
        return if (@field(self, field)) |v| @intCast(v) else default;
    }

    /// Get float field with default
    fn floatOrDefault(self: @This(), comptime field: []const u8, default: f32) f32 {
        return if (@field(self, field)) |v| @floatCast(v) else default;
    }

    /// Get grouped-affine group size from quantization config, defaulting to 64
    fn gaffineGroupSize(self: @This()) i32 {
        if (self.quantization) |q| return @intCast(q.group_size orelse 64);
        if (self.quantization_config) |q| return @intCast(q.group_size orelse 64);
        return 64;
    }

    /// Get grouped-affine quantization bits from quantization config, defaulting to 4
    fn gaffineBits(self: @This()) i32 {
        if (self.quantization) |q| return @intCast(q.bits orelse 4);
        if (self.quantization_config) |q| return @intCast(q.bits orelse 4);
        return 4;
    }
};

/// Minimal config payload used for architecture detection before full parsing.
pub const BaseConfig = struct {
    model_type: ?[]const u8 = null,
    architecture: ?[]const u8 = null,

    pub fn deinit(self: *BaseConfig, allocator: std.mem.Allocator) void {
        if (self.model_type) |model_type| allocator.free(model_type);
        if (self.architecture) |architecture| allocator.free(architecture);
        self.* = .{};
    }
};

fn mapJsonParseError(err: anyerror) anyerror {
    return switch (err) {
        error.InputTooLarge => error.InvalidJson,
        error.InputTooDeep => error.InvalidJson,
        error.StringTooLong => error.InvalidJson,
        error.InvalidJson => error.InvalidJson,
        error.OutOfMemory => error.OutOfMemory,
        else => err,
    };
}

fn modelTypeFromRootObject(root_obj: std.json.ObjectMap) ?[]const u8 {
    // Prefer root-level model_type. This is the primary architecture identifier
    // for model bundles and conversion.
    if (root_obj.get("model_type")) |v| {
        switch (v) {
            .string => |s| return s,
            else => {},
        }
    }

    // Fall back to text_config.model_type for multimodal wrappers that only
    // define model_type in the text sub-config.
    if (root_obj.get("text_config")) |tc| {
        if (tc == .object) {
            if (tc.object.get("model_type")) |v| {
                switch (v) {
                    .string => |s| return s,
                    else => {},
                }
            }
        }
    }
    return null;
}

fn architectureNameFromRootObject(root_obj: std.json.ObjectMap) ?[]const u8 {
    if (root_obj.get("architectures")) |v| {
        if (v == .array and v.array.items.len > 0) {
            const first = v.array.items[0];
            if (first == .string) return first.string;
        }
    }
    return null;
}

fn resolveConfigParseHook(
    architecture_id: ?[]const u8,
    root_obj: std.json.ObjectMap,
) ?model_types.ConfigParseHook {
    if (architecture_id) |arch_id| {
        if (registry.runtimeArchitectureById(arch_id)) |arch| {
            return arch.parse_config_hook;
        }
    }

    if (modelTypeFromRootObject(root_obj)) |model_type| {
        if (registry.detectByModelType(model_type)) |entry| {
            if (registry.runtimeArchitectureById(entry.id)) |arch| {
                return arch.parse_config_hook;
            }
        }
    }

    return null;
}

/// Read minimal architecture-identification config fields from config.json.
/// Caller owns any returned strings and must call `BaseConfig.deinit`.
pub fn readBaseConfig(allocator: std.mem.Allocator, path: []const u8) !BaseConfig {
    const config_bytes = try std.fs.cwd().readFileAlloc(allocator, path, 256 * 1024);
    defer allocator.free(config_bytes);

    const raw_parsed = json.parseValue(allocator, config_bytes, .{ .max_size_bytes = 256 * 1024 }) catch |err| {
        return mapJsonParseError(err);
    };
    defer raw_parsed.deinit();

    if (raw_parsed.value != .object) return .{};
    const root_obj = raw_parsed.value.object;

    return .{
        .model_type = if (modelTypeFromRootObject(root_obj)) |model_type|
            try allocator.dupe(u8, model_type)
        else
            null,
        .architecture = if (architectureNameFromRootObject(root_obj)) |architecture|
            try allocator.dupe(u8, architecture)
        else
            null,
    };
}

/// Read just the model_type string from config.json.
/// Returns null if model_type is not present.
/// Caller owns the returned string.
pub fn readModelType(allocator: std.mem.Allocator, path: []const u8) !?[]const u8 {
    var base_config = try readBaseConfig(allocator, path);
    errdefer base_config.deinit(allocator);

    const model_type = base_config.model_type;
    base_config.model_type = null;
    base_config.deinit(allocator);
    return model_type;
}

/// Read the first architecture entry from config.json architectures[].
/// Returns null if architectures is absent or empty.
/// Caller owns the returned string.
pub fn readArchitectureName(allocator: std.mem.Allocator, path: []const u8) !?[]const u8 {
    var base_config = try readBaseConfig(allocator, path);
    errdefer base_config.deinit(allocator);

    const architecture = base_config.architecture;
    base_config.architecture = null;
    base_config.deinit(allocator);
    return architecture;
}

/// Parse layer_types from a model's config.json and convert to variant indices.
/// Returns null if layer_types is not present in the config.
///
/// This allows different model sizes of the same architecture to have different
/// layer arrangements. Matching order: exact variant name match first, then
/// alias match. Aliases handle models that use different strings for the same
/// variant (e.g., "conv" in config matching "shortconv" in static model metadata).
pub fn parseLayerTypes(
    allocator: std.mem.Allocator,
    config_path: []const u8,
    variant_names: []const []const u8,
    variant_aliases: ?[]const model_types.VariantAlias,
) !?[]const u8 {
    const config_bytes = std.fs.cwd().readFileAlloc(allocator, config_path, 256 * 1024) catch return null;
    defer allocator.free(config_bytes);

    const parsed = json.parseValue(allocator, config_bytes, .{ .max_size_bytes = 256 * 1024 }) catch return null;
    defer parsed.deinit();

    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return null,
    };

    const layer_types_value = obj.get("layer_types") orelse return null;
    const layer_types_array = switch (layer_types_value) {
        .array => |a| a,
        else => return null,
    };

    const result = try allocator.alloc(u8, layer_types_array.items.len);
    errdefer allocator.free(result);

    for (layer_types_array.items, 0..) |item, i| {
        const layer_type = switch (item) {
            .string => |s| s,
            else => {
                result[i] = 0;
                continue;
            },
        };

        // Try exact variant name match first
        var found = false;
        for (variant_names, 0..) |variant_name, idx| {
            if (std.mem.eql(u8, layer_type, variant_name)) {
                result[i] = @intCast(idx);
                found = true;
                break;
            }
        }

        // Fall back to alias match
        if (!found) {
            if (variant_aliases) |aliases| {
                for (aliases) |alias| {
                    if (std.mem.eql(u8, layer_type, alias.alias)) {
                        result[i] = alias.variant_index;
                        found = true;
                        break;
                    }
                }
            }
        }

        if (!found) {
            // Unknown layer type - default to first variant (index 0)
            result[i] = 0;
        }
    }
    return result;
}

pub fn loadConfig(allocator: std.mem.Allocator, path: []const u8) !ModelConfig {
    return loadConfigForArchitectureWithHook(allocator, path, null, null);
}

pub fn loadConfigForArchitecture(
    allocator: std.mem.Allocator,
    path: []const u8,
    architecture_id: ?[]const u8,
) !ModelConfig {
    return loadConfigForArchitectureWithHook(allocator, path, architecture_id, null);
}

pub fn loadConfigForArchitectureWithHook(
    allocator: std.mem.Allocator,
    path: []const u8,
    architecture_id: ?[]const u8,
    parse_config_hook: ?model_types.ConfigParseHook,
) !ModelConfig {
    const config_bytes = try std.fs.cwd().readFileAlloc(allocator, path, 256 * 1024);
    defer allocator.free(config_bytes);

    // Parse once as generic JSON value, then extract what we need
    const raw_parsed = json.parseValue(allocator, config_bytes, .{ .max_size_bytes = 256 * 1024 }) catch |err| {
        return switch (err) {
            error.InputTooLarge => error.InvalidJson,
            error.InputTooDeep => error.InvalidJson,
            error.StringTooLong => error.InvalidJson,
            error.InvalidJson => error.InvalidJson,
            error.OutOfMemory => error.OutOfMemory,
        };
    };
    defer raw_parsed.deinit();

    const root_obj = if (raw_parsed.value == .object)
        raw_parsed.value.object
    else
        return error.InvalidJson;
    const config_parse_hook = parse_config_hook orelse resolveConfigParseHook(architecture_id, root_obj);

    // Determine which value to parse as JsonConfig:
    // - If multimodal model with text_config, use that subobject
    // - Otherwise use the root object
    const config_value: std.json.Value = if (raw_parsed.value == .object) blk: {
        if (root_obj.get("text_config")) |text_config| {
            if (text_config == .object) break :blk text_config;
        }
        break :blk raw_parsed.value;
    } else raw_parsed.value;
    const config_obj: std.json.ObjectMap = if (config_value == .object) config_value.object else root_obj;

    // Parse the selected value into JsonConfig (no re-parsing of the file)
    const parsed_config = json.parseStructFromValue(allocator, JsonConfig, config_value, config_bytes.len, .{
        .max_size_bytes = 256 * 1024,
        .ignore_unknown_fields = true,
    }) catch |err| {
        return switch (err) {
            error.InputTooLarge => error.InvalidJson,
            error.InputTooDeep => error.InvalidJson,
            error.StringTooLong => error.InvalidJson,
            error.InvalidJson => error.InvalidJson,
            error.OutOfMemory => error.OutOfMemory,
        };
    };
    defer parsed_config.deinit();
    const config_json = parsed_config.value;

    const vocab_size = config_json.firstInt(.{"vocab_size"}) orelse return error.MissingField;
    const d_model = config_json.firstInt(.{ "d_model", "hidden_size" }) orelse return error.MissingField;
    const n_layers = config_json.firstInt(.{ "n_layers", "num_layers", "num_hidden_layers" }) orelse return error.MissingField;
    const n_heads = config_json.firstInt(.{ "n_heads", "num_heads", "num_attention_heads" }) orelse return error.MissingField;
    const max_seq_len = config_json.firstInt(.{ "max_seq_len", "context_length", "max_position_embeddings" }) orelse return error.MissingField;

    // Extract d_ff as a best-effort initial value from config.
    // For MoE models, prefer moe_intermediate_size (expert FFN width) over
    // dense intermediate_size when available.
    const d_ff = try initialDff(config_json);

    if (vocab_size <= 0 or d_model <= 0 or n_layers <= 0 or n_heads <= 0 or d_ff <= 0 or max_seq_len <= 0) {
        return error.InvalidValue;
    }

    // Determine quantization method
    const quant_method_kind: tensor.QuantMethod = blk: {
        // Check quantization_config first (both quant_method and mode fields)
        if (config_json.quantization_config) |quant_config| {
            if (quant_config.quant_method) |method| {
                if (std.mem.eql(u8, method, "mxfp4")) break :blk .mxfp4;
                if (std.mem.eql(u8, method, "talu")) break :blk .native;
            }
            if (quant_config.mode) |mode| {
                if (std.mem.eql(u8, mode, "mxfp4")) break :blk .mxfp4;
            }
        }
        // Check quantization.mode field
        if (config_json.quantization) |quantization| {
            if (quantization.mode) |mode| {
                if (std.mem.eql(u8, mode, "mxfp4")) break :blk .mxfp4;
            }
        }
        // Check if grouped-affine quantization (has group_size or bits)
        if (config_json.quantization != null or (config_json.quantization_config != null and
            (config_json.quantization_config.?.group_size != null or config_json.quantization_config.?.bits != null)))
        {
            break :blk .gaffine;
        }
        break :blk .none;
    };

    // Parse rope_scaling if present.
    // Note: some models use `rope_type="yarn"` with `beta_fast/beta_slow`.
    const rope_scaling_params: tensor.RopeScaling = if (config_json.rope_scaling) |rope_scaling_config| blk: {
        // Determine rope_type enum value
        var rope_type_val: @TypeOf((tensor.RopeScaling{}).rope_type) = .none;
        if (rope_scaling_config.rope_type) |rope_type_str| {
            if (std.mem.eql(u8, rope_type_str, "llama3")) rope_type_val = .llama3;
            if (std.mem.eql(u8, rope_type_str, "linear")) rope_type_val = .linear;
            // YaRN is its own scaling scheme; handled explicitly in RoPE init.
            if (std.mem.eql(u8, rope_type_str, "yarn")) rope_type_val = .yarn;
        }
        const beta_slow = if (rope_scaling_config.beta_slow) |f| @as(f32, @floatCast(f)) else 1.0;
        const beta_fast = if (rope_scaling_config.beta_fast) |f| @as(f32, @floatCast(f)) else 32.0;
        const attention_factor = if (rope_scaling_config.attention_factor) |f| @as(f32, @floatCast(f)) else 0.0;
        const mscale = if (rope_scaling_config.mscale) |f| @as(f32, @floatCast(f)) else 0.0;
        const mscale_all_dim = if (rope_scaling_config.mscale_all_dim) |f| @as(f32, @floatCast(f)) else 0.0;
        const truncate = rope_scaling_config.truncate orelse true;
        var mrope_section: [3]u32 = .{ 0, 0, 0 };
        if (rope_scaling_config.mrope_section) |section| {
            if (section.len == 3) {
                for (0..3) |idx| {
                    mrope_section[idx] = std.math.cast(u32, section[idx]) orelse 0;
                }
            }
        }

        break :blk .{
            .rope_type = rope_type_val,
            .factor = if (rope_scaling_config.factor) |f| @floatCast(f) else 1.0,
            .low_freq_factor = if (rope_scaling_config.low_freq_factor) |f| @floatCast(f) else beta_slow,
            .high_freq_factor = if (rope_scaling_config.high_freq_factor) |f| @floatCast(f) else beta_fast,
            .beta_slow = beta_slow,
            .beta_fast = beta_fast,
            .attention_factor = attention_factor,
            .mscale = mscale,
            .mscale_all_dim = mscale_all_dim,
            .truncate = truncate,
            .original_max_position_embeddings = if (rope_scaling_config.original_max_position_embeddings) |v| @intCast(v) else 8192,
            .mrope_section = mrope_section,
            .mrope_interleaved = rope_scaling_config.mrope_interleaved orelse false,
        };
    } else .{};

    // Detect GELU activation from hidden_activation field
    const use_gelu_activation = if (config_json.hidden_activation) |hidden_activation|
        std.mem.eql(u8, hidden_activation, "gelu_pytorch_tanh") or std.mem.eql(u8, hidden_activation, "gelu")
    else
        false;

    // Model architecture is driven by static metadata from the registry.
    const head_dim_value = config_json.firstInt(.{"head_dim"}) orelse @divTrunc(d_model, n_heads);
    var config = ModelConfig{
        .vocab_size = vocab_size,
        .d_model = d_model,
        .n_layers = n_layers,
        .n_heads = n_heads,
        .n_kv_groups = config_json.firstInt(.{ "n_kv_groups", "num_key_value_heads" }) orelse n_heads,
        .d_ff = d_ff,
        .max_seq_len = max_seq_len,
        .head_dim = head_dim_value,
        .rope_dim = 0,
        .rope_theta = config_json.firstFloat(.{ "rope_base", "rope_theta" }) orelse
            (if (config_json.rope_parameters) |rope_params| if (rope_params.rope_theta) |t| @as(f32, @floatCast(t)) else null else null) orelse 10000.0,
        .norm_eps = config_json.firstFloat(.{ "norm_eps", "rms_norm_eps" }) orelse 1e-5,
        .gaffine_group_size = config_json.gaffineGroupSize(),
        .gaffine_bits = config_json.gaffineBits(),
        .tie_word_embeddings = config_json.tie_word_embeddings orelse true, // Default true for most models
        .num_experts = config_json.firstInt(.{ "num_local_experts", "num_experts" }) orelse 0,
        .experts_per_token = config_json.firstInt(.{ "num_experts_per_tok", "experts_per_token" }) orelse 0,
        .quant_method = quant_method_kind,
        .rope_scaling = rope_scaling_params,
        // Model arch is always .custom - actual architecture comes from models metadata.
        .model_arch = .custom,
        .use_gelu = use_gelu_activation,
        .mamba_d_state = 0,
        .mamba_d_conv = 0,
        .mamba_n_heads = 0,
        .mamba_d_head = 0,
        .mamba_n_groups = 1,
        .mamba_expand = 2,
        .shortconv_d_conv = 0,
        .shortconv_conv_dim = 0,
        .shortconv_conv_dim_out = 0,
        .shortconv_has_bias = false,
    };

    if (config_parse_hook) |hook| {
        hook(config_obj, root_obj, &config);
    }

    config.initFlashAttnCompat();
    return config;
}

fn initialDff(config_json: JsonConfig) !i32 {
    const num_experts = config_json.firstInt(.{ "num_local_experts", "num_experts" }) orelse 0;
    if (num_experts > 0) {
        if (config_json.moe_intermediate_size) |v| return @intCast(v);
    }

    if (config_json.d_ff) |v| return @intCast(v);
    if (config_json.intermediate_size) |v| return @intCast(v);
    if (config_json.block_ff_dim) |v| return @intCast(v);
    return error.MissingField;
}

// =============================================================================
// Directory-based helpers (handle config.json path joining)
// =============================================================================

/// Load model configuration from a model directory.
/// This is a convenience wrapper that joins the directory with "config.json".
pub fn loadConfigFromDir(allocator: std.mem.Allocator, model_dir: []const u8) !ModelConfig {
    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(config_path);
    return loadConfigForArchitecture(allocator, config_path, null);
}

// =============================================================================
// C-compatible model description (for FFI)
// =============================================================================

/// Quantization method enum for C interop.
pub const QuantMethod = enum(c_int) {
    none = 0,
    gaffine = 1,
    mxfp4 = 2,
    native = 3,
};

/// C-compatible model description with null-terminated strings.
/// Used by the C API for model introspection.
pub const ModelDescription = struct {
    // Model dimensions
    vocab_size: i32,
    hidden_size: i32,
    num_layers: i32,
    num_heads: i32,
    num_kv_heads: i32,
    intermediate_size: i32,
    max_seq_len: i32,
    head_dim: i32,
    rope_theta: f32,
    norm_eps: f32,

    // Quantization info
    quant_bits: i32,
    quant_group_size: i32,
    quant_method: QuantMethod,

    // Architecture info (null-terminated, caller must free)
    model_type: ?[:0]u8,
    architecture: ?[:0]u8,

    // Flags
    tie_word_embeddings: bool,
    use_gelu: bool,

    // MoE
    num_experts: i32,
    experts_per_token: i32,

    const Self = @This();

    /// Describe a model from its directory path.
    /// Returns a C-compatible description with allocated strings.
    /// Caller must call deinit() to free the strings.
    pub fn fromDir(allocator: std.mem.Allocator, model_dir: []const u8) !Self {
        const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
        defer allocator.free(config_path);

        const model_type_name = readModelType(allocator, config_path) catch null;
        defer if (model_type_name) |name| allocator.free(name);

        const architecture_name = readArchitectureName(allocator, config_path) catch null;
        defer if (architecture_name) |name| allocator.free(name);

        const config = try loadConfigFromDir(allocator, model_dir);

        // Calculate quant bits from method
        const quant_bits: i32 = switch (config.quant_method) {
            .none => 16,
            .gaffine => config.gaffine_bits,
            .mxfp4 => 4,
            .native => 4,
        };

        // Allocate null-terminated strings for C interop
        var model_type: ?[:0]u8 = null;
        if (model_type_name) |name| {
            const buf = try allocator.allocSentinel(u8, name.len, 0);
            @memcpy(buf, name);
            model_type = buf;
        }

        var architecture: ?[:0]u8 = null;
        errdefer if (model_type) |mt| allocator.free(mt);
        if (architecture_name) |name| {
            const buf = try allocator.allocSentinel(u8, name.len, 0);
            @memcpy(buf, name);
            architecture = buf;
        }

        return Self{
            .vocab_size = config.vocab_size,
            .hidden_size = config.d_model,
            .num_layers = config.n_layers,
            .num_heads = config.n_heads,
            .num_kv_heads = config.n_kv_groups,
            .intermediate_size = config.d_ff,
            .max_seq_len = config.max_seq_len,
            .head_dim = config.head_dim,
            .rope_theta = config.rope_theta,
            .norm_eps = config.norm_eps,
            .quant_bits = quant_bits,
            .quant_group_size = config.gaffine_group_size,
            .quant_method = switch (config.quant_method) {
                .none => .none,
                .gaffine => .gaffine,
                .mxfp4 => .mxfp4,
                .native => .native,
            },
            .model_type = model_type,
            .architecture = architecture,
            .tie_word_embeddings = config.tie_word_embeddings,
            .use_gelu = config.use_gelu,
            .num_experts = config.num_experts,
            .experts_per_token = config.experts_per_token,
        };
    }

    /// Free allocated strings.
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        if (self.model_type) |mt| allocator.free(mt);
        if (self.architecture) |arch| allocator.free(arch);
        self.model_type = null;
        self.architecture = null;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "parseRopeScalingFromObject parses llama3 rope type" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    try obj.put("rope_type", .{ .string = "llama3" });
    try obj.put("factor", .{ .float = 2.0 });
    try obj.put("low_freq_factor", .{ .float = 1.5 });
    try obj.put("high_freq_factor", .{ .float = 3.0 });
    try obj.put("original_max_position_embeddings", .{ .integer = 4096 });

    const result = parseRopeScalingFromObject(obj);

    try std.testing.expectEqual(@as(@TypeOf(result.rope_type), .llama3), result.rope_type);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result.factor, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), result.low_freq_factor, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result.high_freq_factor, 0.001);
    try std.testing.expectEqual(@as(i32, 4096), result.original_max_position_embeddings);
}

test "parseRopeScalingFromObject parses yarn rope type with beta fields" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    try obj.put("rope_type", .{ .string = "yarn" });
    try obj.put("factor", .{ .float = 1.0 });
    try obj.put("beta_slow", .{ .float = 2.0 }); // Alternative name for low_freq_factor
    try obj.put("beta_fast", .{ .float = 8.0 }); // Alternative name for high_freq_factor

    const result = parseRopeScalingFromObject(obj);

    try std.testing.expectEqual(@as(@TypeOf(result.rope_type), .yarn), result.rope_type);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result.low_freq_factor, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), result.high_freq_factor, 0.001);
}

test "parseRopeScalingFromObject uses defaults for missing fields" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    // Empty object - should use all defaults
    const result = parseRopeScalingFromObject(obj);

    try std.testing.expectEqual(@as(@TypeOf(result.rope_type), .none), result.rope_type);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result.factor, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result.low_freq_factor, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result.high_freq_factor, 0.001);
    try std.testing.expectEqual(@as(i32, 8192), result.original_max_position_embeddings);
}

test "parseRopeScalingFromObject parses linear rope type" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    try obj.put("rope_type", .{ .string = "linear" });
    try obj.put("factor", .{ .float = 4.0 });

    const result = parseRopeScalingFromObject(obj);

    try std.testing.expectEqual(@as(@TypeOf(result.rope_type), .linear), result.rope_type);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result.factor, 0.001);
}

test "parseRopeScalingFromObject handles integer factor" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    // Some configs use integer instead of float
    try obj.put("factor", .{ .integer = 2 });

    const result = parseRopeScalingFromObject(obj);

    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result.factor, 0.001);
}

test "getFloatField extracts float from JSON value" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    try obj.put("test_float", .{ .float = 3.14 });

    const result = getFloatField(obj, "test_float");
    try std.testing.expect(result != null);
    try std.testing.expectApproxEqAbs(@as(f32, 3.14), result.?, 0.001);
}

test "getFloatField extracts integer as float" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    try obj.put("test_int", .{ .integer = 42 });

    const result = getFloatField(obj, "test_int");
    try std.testing.expect(result != null);
    try std.testing.expectApproxEqAbs(@as(f32, 42.0), result.?, 0.001);
}

test "getFloatField returns null for missing key" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    const result = getFloatField(obj, "missing");
    try std.testing.expect(result == null);
}

test "getFloatField returns null for wrong type" {
    var obj = std.json.ObjectMap.init(std.testing.allocator);
    defer obj.deinit();

    try obj.put("test_string", .{ .string = "not a number" });

    const result = getFloatField(obj, "test_string");
    try std.testing.expect(result == null);
}

test "JsonConfig.firstInt returns first non-null integer field" {
    const config = JsonConfig{
        .vocab_size = null,
        .d_model = 512,
        .hidden_size = 768,
    };

    const result = config.firstInt(.{ "vocab_size", "d_model", "hidden_size" });
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(i32, 512), result.?);
}

test "JsonConfig.firstInt returns null when all fields are null" {
    const config = JsonConfig{
        .vocab_size = null,
        .d_model = null,
        .hidden_size = null,
    };

    const result = config.firstInt(.{ "vocab_size", "d_model", "hidden_size" });
    try std.testing.expect(result == null);
}

test "JsonConfig.firstFloat returns first non-null float field" {
    const config = JsonConfig{
        .rope_base = null,
        .rope_theta = 10000.0,
        .norm_eps = 1e-5,
    };

    const result = config.firstFloat(.{ "rope_base", "rope_theta", "norm_eps" });
    try std.testing.expect(result != null);
    try std.testing.expectApproxEqAbs(@as(f32, 10000.0), result.?, 0.001);
}

test "JsonConfig.firstFloat returns null when all fields are null" {
    const config = JsonConfig{
        .rope_base = null,
        .rope_theta = null,
        .norm_eps = null,
    };

    const result = config.firstFloat(.{ "rope_base", "rope_theta", "norm_eps" });
    try std.testing.expect(result == null);
}

test "JsonConfig.intOrDefault returns value when present" {
    const config = JsonConfig{
        .n_layers = 12,
    };

    const result = config.intOrDefault("n_layers", 24);
    try std.testing.expectEqual(@as(i32, 12), result);
}

test "JsonConfig.intOrDefault returns default when field is null" {
    const config = JsonConfig{
        .n_layers = null,
    };

    const result = config.intOrDefault("n_layers", 24);
    try std.testing.expectEqual(@as(i32, 24), result);
}

test "JsonConfig.floatOrDefault returns value when present" {
    const config = JsonConfig{
        .norm_eps = 1e-6,
    };

    const result = config.floatOrDefault("norm_eps", 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1e-6), result, 1e-8);
}

test "JsonConfig.floatOrDefault returns default when field is null" {
    const config = JsonConfig{
        .norm_eps = null,
    };

    const result = config.floatOrDefault("norm_eps", 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1e-5), result, 1e-8);
}

test "JsonConfig.gaffineGroupSize returns group size from quantization config" {
    const config = JsonConfig{
        .quantization = .{ .group_size = 128 },
    };

    const result = config.gaffineGroupSize();
    try std.testing.expectEqual(@as(i32, 128), result);
}

test "JsonConfig.gaffineGroupSize returns group size from quantization_config" {
    const config = JsonConfig{
        .quantization_config = .{ .group_size = 256 },
    };

    const result = config.gaffineGroupSize();
    try std.testing.expectEqual(@as(i32, 256), result);
}

test "JsonConfig.gaffineGroupSize returns default 64 when not present" {
    const config = JsonConfig{};

    const result = config.gaffineGroupSize();
    try std.testing.expectEqual(@as(i32, 64), result);
}

test "JsonConfig.gaffineBits returns bits from quantization config" {
    const config = JsonConfig{
        .quantization = .{ .bits = 8 },
    };

    const result = config.gaffineBits();
    try std.testing.expectEqual(@as(i32, 8), result);
}

test "JsonConfig.gaffineBits returns bits from quantization_config" {
    const config = JsonConfig{
        .quantization_config = .{ .bits = 4 },
    };

    const result = config.gaffineBits();
    try std.testing.expectEqual(@as(i32, 4), result);
}

test "JsonConfig.gaffineBits returns default 4 when not present" {
    const config = JsonConfig{};

    const result = config.gaffineBits();
    try std.testing.expectEqual(@as(i32, 4), result);
}

test "initialDff prefers MoE intermediate size when experts are configured" {
    const config = JsonConfig{
        .num_experts = 128,
        .moe_intermediate_size = 768,
        .intermediate_size = 6144,
    };

    const result = try initialDff(config);
    try std.testing.expectEqual(@as(i32, 768), result);
}

test "initialDff falls back to dense intermediate size when MoE is not configured" {
    const config = JsonConfig{
        .intermediate_size = 6144,
    };

    const result = try initialDff(config);
    try std.testing.expectEqual(@as(i32, 6144), result);
}

test "readModelType falls back to root model_type when text_config omits it" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json =
        \\{
        \\  "model_type": "multimodal_vit",
        \\  "text_config": {
        \\    "hidden_size": 2048
        \\  }
        \\}
    ;

    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "config.json");
    defer std.testing.allocator.free(path);

    const model_type = try readModelType(std.testing.allocator, path);
    defer if (model_type) |m| std.testing.allocator.free(m);
    try std.testing.expect(model_type != null);
    try std.testing.expectEqualStrings("multimodal_vit", model_type.?);
}

test "readModelType prefers root model_type when text_config also has model_type" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json =
        \\{
        \\  "model_type": "multimodal_root_arch",
        \\  "text_config": {
        \\    "model_type": "text_sub_arch"
        \\  }
        \\}
    ;

    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "config.json");
    defer std.testing.allocator.free(path);

    const model_type = try readModelType(std.testing.allocator, path);
    defer if (model_type) |m| std.testing.allocator.free(m);
    try std.testing.expect(model_type != null);
    try std.testing.expectEqualStrings("multimodal_root_arch", model_type.?);
}

test "readModelType falls back to text_config model_type when root is missing" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json =
        \\{
        \\  "text_config": {
        \\    "model_type": "llama"
        \\  }
        \\}
    ;

    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "config.json");
    defer std.testing.allocator.free(path);

    const model_type = try readModelType(std.testing.allocator, path);
    defer if (model_type) |m| std.testing.allocator.free(m);
    try std.testing.expect(model_type != null);
    try std.testing.expectEqualStrings("llama", model_type.?);
}

test "readBaseConfig parses model_type and first architecture entry" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json =
        \\{
        \\  "model_type": "llama3",
        \\  "architectures": ["LlamaForCausalLM"]
        \\}
    ;

    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "config.json");
    defer std.testing.allocator.free(path);

    var base_config = try readBaseConfig(std.testing.allocator, path);
    defer base_config.deinit(std.testing.allocator);

    try std.testing.expect(base_config.model_type != null);
    try std.testing.expectEqualStrings("llama3", base_config.model_type.?);
    try std.testing.expect(base_config.architecture != null);
    try std.testing.expectEqualStrings("LlamaForCausalLM", base_config.architecture.?);
}

test "readArchitectureName returns first architectures entry" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json =
        \\{
        \\  "architectures": ["llama3", "ignored"]
        \\}
    ;

    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "config.json");
    defer std.testing.allocator.free(path);

    const architecture_name = try readArchitectureName(std.testing.allocator, path);
    defer if (architecture_name) |name| std.testing.allocator.free(name);
    try std.testing.expect(architecture_name != null);
    try std.testing.expectEqualStrings("llama3", architecture_name.?);
}

test "loadConfigForArchitecture parses Mamba family fields" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json =
        \\{
        \\  "model_type": "qwen3_next",
        \\  "vocab_size": 32000,
        \\  "hidden_size": 1024,
        \\  "num_hidden_layers": 8,
        \\  "num_attention_heads": 16,
        \\  "intermediate_size": 4096,
        \\  "max_position_embeddings": 8192,
        \\  "mamba_d_state": 64,
        \\  "mamba_d_conv": 4,
        \\  "mamba_n_heads": 16,
        \\  "mamba_d_head": 64,
        \\  "mamba_n_groups": 2,
        \\  "mamba_expand": 3
        \\}
    ;

    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "config.json");
    defer std.testing.allocator.free(path);

    const config = try loadConfigForArchitecture(std.testing.allocator, path, "qwen3_next");
    try std.testing.expectEqual(@as(i32, 64), config.mamba_d_state);
    try std.testing.expectEqual(@as(i32, 4), config.mamba_d_conv);
    try std.testing.expectEqual(@as(i32, 16), config.mamba_n_heads);
    try std.testing.expectEqual(@as(i32, 64), config.mamba_d_head);
    try std.testing.expectEqual(@as(i32, 2), config.mamba_n_groups);
    try std.testing.expectEqual(@as(i32, 3), config.mamba_expand);
}

test "loadConfigForArchitecture parses ShortConv family fields" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json =
        \\{
        \\  "model_type": "lfm2_5",
        \\  "vocab_size": 32000,
        \\  "hidden_size": 1024,
        \\  "num_hidden_layers": 8,
        \\  "num_attention_heads": 16,
        \\  "intermediate_size": 4096,
        \\  "max_position_embeddings": 8192,
        \\  "conv_L_cache": 5,
        \\  "conv_dim": 4096,
        \\  "conv_dim_out": 1024,
        \\  "conv_bias": true
        \\}
    ;

    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "config.json");
    defer std.testing.allocator.free(path);

    const config = try loadConfigForArchitecture(std.testing.allocator, path, "lfm2_5");
    try std.testing.expectEqual(@as(i32, 5), config.shortconv_d_conv);
    try std.testing.expectEqual(@as(i32, 4096), config.shortconv_conv_dim);
    try std.testing.expectEqual(@as(i32, 1024), config.shortconv_conv_dim_out);
    try std.testing.expectEqual(true, config.shortconv_has_bias);
}

test "loadConfigForArchitectureWithHook applies explicit hook override" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json =
        \\{
        \\  "model_type": "llama3",
        \\  "vocab_size": 32000,
        \\  "hidden_size": 1024,
        \\  "num_hidden_layers": 8,
        \\  "num_attention_heads": 16,
        \\  "intermediate_size": 4096,
        \\  "max_position_embeddings": 8192
        \\}
    ;

    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "config.json");
    defer std.testing.allocator.free(path);

    const LocalHook = struct {
        fn apply(
            config_obj: std.json.ObjectMap,
            root_obj: std.json.ObjectMap,
            config: *ModelConfig,
        ) void {
            _ = config_obj;
            _ = root_obj;
            config.mamba_d_state = 123;
        }
    };

    const config = try loadConfigForArchitectureWithHook(
        std.testing.allocator,
        path,
        null,
        LocalHook.apply,
    );
    try std.testing.expectEqual(@as(i32, 123), config.mamba_d_state);
}

test "loadConfig fallback parser keeps family fields without architecture id" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json =
        \\{
        \\  "model_type": "granite_hybrid",
        \\  "vocab_size": 32000,
        \\  "hidden_size": 1024,
        \\  "num_hidden_layers": 8,
        \\  "num_attention_heads": 16,
        \\  "intermediate_size": 4096,
        \\  "max_position_embeddings": 8192,
        \\  "mamba_d_state": 72
        \\}
    ;

    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "config.json");
    defer std.testing.allocator.free(path);

    const config = try loadConfig(std.testing.allocator, path);
    try std.testing.expectEqual(@as(i32, 72), config.mamba_d_state);
}

test "loadConfigForArchitecture parses common hook fields across all architectures" {
    const HookCase = struct {
        model_type: []const u8,
        architecture_id: []const u8,
    };
    const cases = [_]HookCase{
        .{ .model_type = "llama2", .architecture_id = "llama2" },
        .{ .model_type = "llama3", .architecture_id = "llama3" },
        .{ .model_type = "gemma3", .architecture_id = "gemma3" },
        .{ .model_type = "granite", .architecture_id = "granite3" },
        .{ .model_type = "granite_hybrid", .architecture_id = "granite_hybrid" },
        .{ .model_type = "lfm2", .architecture_id = "lfm2" },
        .{ .model_type = "lfm2_5", .architecture_id = "lfm2_5" },
        .{ .model_type = "phi4", .architecture_id = "phi" },
        .{ .model_type = "qwen3", .architecture_id = "qwen3" },
        .{ .model_type = "qwen3_moe", .architecture_id = "qwen3_moe" },
        .{ .model_type = "qwen3_next", .architecture_id = "qwen3_next" },
        .{ .model_type = "gpt_oss", .architecture_id = "gpt_oss" },
        .{ .model_type = "bert", .architecture_id = "minilm" },
        .{ .model_type = "ministral3", .architecture_id = "ministral3" },
        .{ .model_type = "youtu_vl", .architecture_id = "youtu_vl" },
    };

    for (cases) |case| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();

        const config_json = try std.fmt.allocPrint(std.testing.allocator,
            \\{{
            \\  "model_type": "{s}",
            \\  "vocab_size": 32000,
            \\  "hidden_size": 1024,
            \\  "num_hidden_layers": 8,
            \\  "num_attention_heads": 16,
            \\  "intermediate_size": 4096,
            \\  "max_position_embeddings": 8192,
            \\  "use_qk_norm": true,
            \\  "attention_bias": true,
            \\  "query_pre_attn_scalar": 128,
            \\  "sliding_window": 512,
            \\  "sliding_window_pattern": 2,
            \\  "rope_local_base_freq": 25000,
            \\  "embedding_multiplier": 1.5,
            \\  "attention_multiplier": 0.5,
            \\  "residual_multiplier": 0.75,
            \\  "logits_scaling": 1.25,
            \\  "bos_token_id": 42
            \\}}
        , .{case.model_type});
        defer std.testing.allocator.free(config_json);

        try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });
        const path = try tmp.dir.realpathAlloc(std.testing.allocator, "config.json");
        defer std.testing.allocator.free(path);

        const config = try loadConfigForArchitecture(std.testing.allocator, path, case.architecture_id);
        try std.testing.expect(config.use_qk_norm);
        try std.testing.expect(config.attention_bias);
        try std.testing.expectApproxEqAbs(@as(f32, 128.0), config.query_pre_attn_scalar, 0.0001);
        try std.testing.expectEqual(@as(i32, 512), config.sliding_window);
        try std.testing.expectEqual(@as(i32, 2), config.sliding_window_pattern);
        try std.testing.expectApproxEqAbs(@as(f32, 25000.0), config.rope_local_theta, 0.0001);
        try std.testing.expectApproxEqAbs(@as(f32, 1.5), config.embedding_multiplier, 0.0001);
        try std.testing.expectApproxEqAbs(@as(f32, 0.5), config.attention_multiplier, 0.0001);
        try std.testing.expectApproxEqAbs(@as(f32, 0.75), config.residual_multiplier, 0.0001);
        try std.testing.expectApproxEqAbs(@as(f32, 1.25), config.logits_scaling, 0.0001);
        try std.testing.expect(config.bos_token_id != null);
        try std.testing.expectEqual(@as(i32, 42), config.bos_token_id.?);
    }
}

test "loadConfigForArchitecture parses Phi partial rotary factor via hook" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json =
        \\{
        \\  "model_type": "phi4",
        \\  "vocab_size": 32000,
        \\  "hidden_size": 1024,
        \\  "num_hidden_layers": 8,
        \\  "num_attention_heads": 16,
        \\  "head_dim": 128,
        \\  "intermediate_size": 4096,
        \\  "max_position_embeddings": 8192,
        \\  "partial_rotary_factor": 0.75
        \\}
    ;

    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "config.json");
    defer std.testing.allocator.free(path);

    const config = try loadConfigForArchitecture(std.testing.allocator, path, "phi");
    try std.testing.expectEqual(@as(i32, 96), config.rope_dim);
}

test "loadConfig parses vision deepstack probe layers from vision_config via youtu hook" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json =
        \\{
        \\  "model_type": "youtu_vl",
        \\  "text_config": {
        \\    "vocab_size": 151936,
        \\    "hidden_size": 2048,
        \\    "num_hidden_layers": 28,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 8,
        \\    "intermediate_size": 6144,
        \\    "max_position_embeddings": 32768,
        \\    "rms_norm_eps": 1e-6,
        \\    "rope_theta": 5000000.0
        \\  },
        \\  "vision_config": {
        \\    "hidden_size": 1024,
        \\    "depth": 24,
        \\    "num_heads": 16,
        \\    "intermediate_size": 4096,
        \\    "out_hidden_size": 2048,
        \\    "patch_size": 16,
        \\    "spatial_merge_size": 2,
        \\    "temporal_patch_size": 2,
        \\    "num_position_embeddings": 2304,
        \\    "deepstack_visual_indexes": [5, 11, 17]
        \\  }
        \\}
    ;

    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "config.json");
    defer std.testing.allocator.free(path);

    const config = try loadConfig(std.testing.allocator, path);
    try std.testing.expectEqual(@as(u8, 3), config.vision_probe_layer_count);
    try std.testing.expectEqual(@as(u16, 5), config.vision_probe_layers[0]);
    try std.testing.expectEqual(@as(u16, 11), config.vision_probe_layers[1]);
    try std.testing.expectEqual(@as(u16, 17), config.vision_probe_layers[2]);
}

test "loadConfig parses generic vision fallback fields and image token aliases via youtu hook" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json =
        \\{
        \\  "model_type": "youtu_vl",
        \\  "image_token_index": 396,
        \\  "downsample_factor": 2,
        \\  "max_num_patches": 1024,
        \\  "projector_hidden_size": 2560,
        \\  "text_config": {
        \\    "vocab_size": 32000,
        \\    "hidden_size": 1024,
        \\    "num_hidden_layers": 16,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 8,
        \\    "intermediate_size": 4096,
        \\    "max_position_embeddings": 32768,
        \\    "rms_norm_eps": 1e-6,
        \\    "rope_theta": 1000000.0
        \\  },
        \\  "vision_config": {
        \\    "hidden_size": 768,
        \\    "num_hidden_layers": 12,
        \\    "num_attention_heads": 12,
        \\    "intermediate_size": 3072,
        \\    "patch_size": 16,
        \\    "num_patches": 256
        \\  }
        \\}
    ;

    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "config.json");
    defer std.testing.allocator.free(path);

    const config = try loadConfig(std.testing.allocator, path);
    try std.testing.expectEqual(@as(i32, 768), config.vision_hidden_size);
    try std.testing.expectEqual(@as(i32, 12), config.vision_depth);
    try std.testing.expectEqual(@as(i32, 12), config.vision_num_heads);
    try std.testing.expectEqual(@as(i32, 3072), config.vision_intermediate_size);
    try std.testing.expectEqual(@as(i32, 2560), config.projector_hidden_size);
    try std.testing.expectEqual(@as(i32, 16), config.vision_patch_size);
    try std.testing.expectEqual(@as(i32, 2), config.vision_spatial_merge_size);
    try std.testing.expectEqual(@as(i32, 1), config.vision_temporal_patch_size);
    try std.testing.expectEqual(@as(i32, 256), config.vision_num_position_embeddings);
    try std.testing.expectEqual(@as(i32, 1024), config.vision_max_num_patches);
    try std.testing.expectEqual(@as(i32, 396), config.image_token_id);
}

test "loadConfig parses qwen3_vl vision fields through qwen3 architecture hook" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json =
        \\{
        \\  "model_type": "qwen3_vl",
        \\  "image_token_id": 151655,
        \\  "vision_start_token_id": 151652,
        \\  "vision_end_token_id": 151653,
        \\  "text_config": {
        \\    "vocab_size": 151936,
        \\    "hidden_size": 2048,
        \\    "num_hidden_layers": 28,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 8,
        \\    "intermediate_size": 6144,
        \\    "max_position_embeddings": 262144,
        \\    "rms_norm_eps": 1e-6,
        \\    "rope_theta": 5000000.0
        \\  },
        \\  "vision_config": {
        \\    "hidden_size": 1024,
        \\    "depth": 24,
        \\    "num_heads": 16,
        \\    "intermediate_size": 4096,
        \\    "out_hidden_size": 2048,
        \\    "patch_size": 16,
        \\    "spatial_merge_size": 2,
        \\    "temporal_patch_size": 2,
        \\    "num_position_embeddings": 2304
        \\  }
        \\}
    ;

    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "config.json");
    defer std.testing.allocator.free(path);

    const config = try loadConfig(std.testing.allocator, path);
    try std.testing.expectEqual(@as(i32, 151655), config.image_token_id);
    try std.testing.expectEqual(@as(i32, 151652), config.vision_start_token_id);
    try std.testing.expectEqual(@as(i32, 151653), config.vision_end_token_id);
    try std.testing.expectEqual(@as(i32, 1024), config.vision_hidden_size);
    try std.testing.expectEqual(@as(i32, 24), config.vision_depth);
    try std.testing.expectEqual(@as(i32, 16), config.vision_num_heads);
    try std.testing.expectEqual(@as(i32, 4096), config.vision_intermediate_size);
    try std.testing.expectEqual(@as(i32, 2048), config.vision_out_hidden_size);
    try std.testing.expectEqual(@as(i32, 16), config.vision_patch_size);
    try std.testing.expectEqual(@as(i32, 2), config.vision_spatial_merge_size);
    try std.testing.expectEqual(@as(i32, 2), config.vision_temporal_patch_size);
    try std.testing.expectEqual(@as(i32, 2304), config.vision_num_position_embeddings);
}

test "loadConfig parses lfm2_vl vision fields through lfm2 architecture hook" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json =
        \\{
        \\  "model_type": "lfm2_vl",
        \\  "image_token_index": 396,
        \\  "downsample_factor": 2,
        \\  "projector_hidden_size": 2560,
        \\  "text_config": {
        \\    "vocab_size": 65536,
        \\    "hidden_size": 1024,
        \\    "num_hidden_layers": 16,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 8,
        \\    "intermediate_size": 6656,
        \\    "max_position_embeddings": 128000,
        \\    "norm_eps": 1e-5,
        \\    "rope_theta": 1000000.0
        \\  },
        \\  "vision_config": {
        \\    "hidden_size": 768,
        \\    "num_hidden_layers": 12,
        \\    "num_attention_heads": 12,
        \\    "intermediate_size": 3072,
        \\    "patch_size": 16,
        \\    "num_patches": 256
        \\  }
        \\}
    ;

    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = config_json });
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "config.json");
    defer std.testing.allocator.free(path);

    const config = try loadConfig(std.testing.allocator, path);
    try std.testing.expectEqual(@as(i32, 396), config.image_token_id);
    try std.testing.expectEqual(@as(i32, 768), config.vision_hidden_size);
    try std.testing.expectEqual(@as(i32, 12), config.vision_depth);
    try std.testing.expectEqual(@as(i32, 12), config.vision_num_heads);
    try std.testing.expectEqual(@as(i32, 3072), config.vision_intermediate_size);
    try std.testing.expectEqual(@as(i32, 2560), config.projector_hidden_size);
    try std.testing.expectEqual(@as(i32, 16), config.vision_patch_size);
    try std.testing.expectEqual(@as(i32, 2), config.vision_spatial_merge_size);
    try std.testing.expectEqual(@as(i32, 1), config.vision_temporal_patch_size);
    try std.testing.expectEqual(@as(i32, 256), config.vision_num_position_embeddings);
}
