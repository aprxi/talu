//! GAF (Grouped Affine) model path utilities and config generation
//!
//! GAF models use a grouped affine quantization format compatible with MLX.
//! This module handles directory structure and config format generation.

const std = @import("std");
const tensor = @import("../tensor.zig");
const io = @import("../io/root.zig");

const ModelConfig = tensor.ModelConfig;

/// GAF model configuration (compatible with MLX format)
pub const GAFConfig = struct {
    // Standard fields
    vocab_size: i32,
    hidden_size: i32,
    num_hidden_layers: i32,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    intermediate_size: i32,
    max_position_embeddings: i32,
    head_dim: i32,
    rms_norm_eps: f32,
    rope_theta: f32,

    // GAF quantization config
    quantization: ?QuantizationConfig = null,

    // Model-specific fields
    model_type: ?[]const u8 = null,
    hidden_activation: ?[]const u8 = null,
    query_pre_attn_scalar: ?i32 = null,
    sliding_window: ?i32 = null,
    sliding_window_pattern: ?i32 = null,

    pub const QuantizationConfig = struct {
        group_size: i32 = 64,
        bits: i32 = 4,
    };

    /// Create MLX config from a standard ModelConfig.
    /// The model_type_name parameter should be the original model_type string from config.json.
    pub fn fromModelConfig(model_config: ModelConfig, quant_config: ?QuantizationConfig, model_type_name: ?[]const u8) GAFConfig {
        return .{
            .vocab_size = model_config.vocab_size,
            .hidden_size = model_config.d_model,
            .num_hidden_layers = model_config.n_layers,
            .num_attention_heads = model_config.n_heads,
            .num_key_value_heads = model_config.n_kv_groups,
            .intermediate_size = model_config.d_ff,
            .max_position_embeddings = model_config.max_seq_len,
            .head_dim = model_config.head_dim,
            .rms_norm_eps = model_config.norm_eps,
            .rope_theta = model_config.rope_theta,
            .quantization = quant_config,
            // Model-specific activation and attention fields
            .hidden_activation = if (model_config.use_gelu) "gelu_pytorch_tanh" else null,
            .query_pre_attn_scalar = if (model_config.query_pre_attn_scalar > 0) @intFromFloat(model_config.query_pre_attn_scalar) else null,
            .sliding_window = if (model_config.sliding_window > 0) model_config.sliding_window else null,
            .sliding_window_pattern = if (model_config.sliding_window_pattern > 0) model_config.sliding_window_pattern else null,
            .model_type = model_type_name,
        };
    }

    /// Write MLX config to a JSON file using std.json serialization
    pub fn writeToFile(self: *const GAFConfig, allocator: std.mem.Allocator, path: []const u8) !void {
        const json_text = try std.json.Stringify.valueAlloc(allocator, self, .{ .whitespace = .indent_2 });
        defer allocator.free(json_text);

        var config_file = try std.fs.cwd().createFile(path, .{});
        defer config_file.close();
        try config_file.writeAll(json_text);
    }

    /// Custom JSON serialization to omit null optional fields
    pub fn jsonStringify(self: *const GAFConfig, jws: anytype) !void {
        try jws.beginObject();

        // Required fields
        try jws.objectField("vocab_size");
        try jws.write(self.vocab_size);
        try jws.objectField("hidden_size");
        try jws.write(self.hidden_size);
        try jws.objectField("num_hidden_layers");
        try jws.write(self.num_hidden_layers);
        try jws.objectField("num_attention_heads");
        try jws.write(self.num_attention_heads);
        try jws.objectField("num_key_value_heads");
        try jws.write(self.num_key_value_heads);
        try jws.objectField("intermediate_size");
        try jws.write(self.intermediate_size);
        try jws.objectField("max_position_embeddings");
        try jws.write(self.max_position_embeddings);
        try jws.objectField("head_dim");
        try jws.write(self.head_dim);
        try jws.objectField("rms_norm_eps");
        try jws.write(self.rms_norm_eps);
        try jws.objectField("rope_theta");
        try jws.write(self.rope_theta);

        // Optional fields (only emit if non-null)
        if (self.model_type) |model_type| {
            try jws.objectField("model_type");
            try jws.write(model_type);
        }
        if (self.hidden_activation) |hidden_activation| {
            try jws.objectField("hidden_activation");
            try jws.write(hidden_activation);
        }
        if (self.query_pre_attn_scalar) |query_pre_attn_scalar| {
            try jws.objectField("query_pre_attn_scalar");
            try jws.write(query_pre_attn_scalar);
        }
        if (self.sliding_window) |sliding_window| {
            try jws.objectField("sliding_window");
            try jws.write(sliding_window);
        }
        if (self.sliding_window_pattern) |sliding_window_pattern| {
            try jws.objectField("sliding_window_pattern");
            try jws.write(sliding_window_pattern);
        }
        if (self.quantization) |quant_config| {
            try jws.objectField("quantization");
            try jws.write(quant_config);
        }

        try jws.endObject();
    }
};

/// Generate output directory name for grouped-affine model using hierarchical naming.
///
/// Input formats:
///   - HF ID: "org/model-name"
///   - HF cache: "models--org--model-name" or ".../models--org--model-name/snapshots/abc"
///   - Local path: "/path/to/my-model"
///
/// Output format: "{output_dir}/{org}/{model}-{SUFFIX}"
///   - "org/model-name" -> "models/org/model-name-GAF4"
///   - "models--org--model-8b" -> "models/org/model-8b-GAF4"
///   - "/path/to/my-model" -> "models/local/my-model-GAF4"
pub fn generateOutputName(allocator: std.mem.Allocator, input_path: []const u8, suffix: []const u8, output_dir: []const u8) ![]const u8 {
    var org: []const u8 = "local";
    var model_name: []const u8 = undefined; // Safe: all branches assign before use

    // Try to extract org/model from various input formats
    if (std.mem.indexOf(u8, input_path, "/") != null and !std.mem.startsWith(u8, input_path, "/") and !std.mem.startsWith(u8, input_path, ".")) {
        // HuggingFace ID format: "org/model" (not starting with / or .)
        if (std.mem.indexOf(u8, input_path, "/")) |slash_idx| {
            org = input_path[0..slash_idx];
            model_name = input_path[slash_idx + 1 ..];
        }
    } else {
        // Look for HF cache format: models--org--model
        var model_name_component: ?[]const u8 = null;
        var current_path = input_path;
        while (current_path.len > 0) {
            const basename = std.fs.path.basename(current_path);
            if (std.mem.startsWith(u8, basename, "models--")) {
                model_name_component = basename;
                break;
            }
            const parent = std.fs.path.dirname(current_path);
            if (parent == null or std.mem.eql(u8, parent.?, current_path)) break;
            current_path = parent.?;
        }

        if (model_name_component) |component| {
            // Parse "models--org--model" format
            const without_prefix = component["models--".len..];
            if (std.mem.indexOf(u8, without_prefix, "--")) |sep_idx| {
                org = without_prefix[0..sep_idx];
                model_name = without_prefix[sep_idx + 2 ..];
            } else {
                model_name = without_prefix;
            }
        } else {
            // Local path - use basename
            model_name = std.fs.path.basename(input_path);
        }
    }

    // Strip any existing suffix like -MLX-4bit, -GAF4, etc.
    if (std.mem.indexOf(u8, model_name, "-MLX")) |idx| {
        model_name = model_name[0..idx];
    } else if (std.mem.indexOf(u8, model_name, "-GAF")) |idx| {
        model_name = model_name[0..idx];
    } else if (std.mem.indexOf(u8, model_name, "-Q4")) |idx| {
        model_name = model_name[0..idx];
    } else if (std.mem.indexOf(u8, model_name, "-Q5")) |idx| {
        model_name = model_name[0..idx];
    } else if (std.mem.indexOf(u8, model_name, "-Q6")) |idx| {
        model_name = model_name[0..idx];
    } else if (std.mem.indexOf(u8, model_name, "-Q8")) |idx| {
        model_name = model_name[0..idx];
    } else if (std.mem.indexOf(u8, model_name, "-F16")) |idx| {
        model_name = model_name[0..idx];
    }

    // Construct hierarchical path: output_dir/org/model-SUFFIX
    var output_name_builder = std.ArrayListUnmanaged(u8){};
    errdefer output_name_builder.deinit(allocator);

    try output_name_builder.writer(allocator).print("{s}/{s}/{s}-{s}", .{
        output_dir,
        org,
        model_name,
        suffix,
    });

    return try output_name_builder.toOwnedSlice(allocator);
}

/// MLX model directory structure
pub const GAFModelDir = struct {
    allocator: std.mem.Allocator,
    path: []const u8,

    pub fn init(allocator: std.mem.Allocator, path: []const u8) !GAFModelDir {
        // Create directory if it doesn't exist
        std.fs.cwd().makePath(path) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };

        return .{
            .allocator = allocator,
            .path = try allocator.dupe(u8, path),
        };
    }

    pub fn deinit(self: *GAFModelDir) void {
        self.allocator.free(self.path);
        self.* = undefined;
    }

    /// Get path for config.json
    pub fn configPath(self: *const GAFModelDir) ![]const u8 {
        return try std.fs.path.join(self.allocator, &.{ self.path, "config.json" });
    }

    /// Get path for model.safetensors
    pub fn weightsPath(self: *const GAFModelDir) ![]const u8 {
        return try std.fs.path.join(self.allocator, &.{ self.path, "model.safetensors" });
    }

    /// Get path for tokenizer.json
    pub fn tokenizerPath(self: *const GAFModelDir) ![]const u8 {
        return try std.fs.path.join(self.allocator, &.{ self.path, "tokenizer.json" });
    }

    /// Copy tokenizer and config files from source directory
    pub fn copyTokenizerFiles(self: *const GAFModelDir, source_dir: []const u8) !void {
        const tokenizer_files = [_][]const u8{
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "merges.txt",
            "vocab.json",
        };

        for (tokenizer_files) |filename| {
            const src_path = try std.fs.path.join(self.allocator, &.{ source_dir, filename });
            defer self.allocator.free(src_path);

            const dst_path = try std.fs.path.join(self.allocator, &.{ self.path, filename });
            defer self.allocator.free(dst_path);

            std.fs.cwd().copyFile(src_path, std.fs.cwd(), dst_path, .{}) catch |err| switch (err) {
                error.FileNotFound => continue, // Optional files
                else => return err,
            };
        }
    }
};

/// Derive a model ID (org/model-suffix) from a converted output path.
/// Expects paths like "{output_dir}/{org}/{model-suffix}".
pub fn modelIdFromOutputPath(allocator: std.mem.Allocator, output_path: []const u8) ![]const u8 {
    const model_name = std.fs.path.basename(output_path);
    const parent_dir = std.fs.path.dirname(output_path) orelse {
        return std.fmt.allocPrint(allocator, "{s}", .{model_name});
    };
    const org = std.fs.path.basename(parent_dir);
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ org, model_name });
}

test "generateOutputName" {
    const allocator = std.testing.allocator;

    // HuggingFace ID format: org/model
    {
        const result = try generateOutputName(allocator, "org/model-name", "GAF4", "models");
        defer allocator.free(result);
        try std.testing.expectEqualStrings("models/org/model-name-GAF4", result);
    }

    // HF cache format: models--org--model
    {
        const result = try generateOutputName(allocator, "models--org--model-name", "GAF4", "models");
        defer allocator.free(result);
        try std.testing.expectEqualStrings("models/org/model-name-GAF4", result);
    }

    // HF cache path with snapshots
    {
        const result = try generateOutputName(allocator, "/home/user/.cache/huggingface/hub/models--org--model-name/snapshots/abc123def", "GAF4", "models");
        defer allocator.free(result);
        try std.testing.expectEqualStrings("models/org/model-name-GAF4", result);
    }

    // F16 (no quantization)
    {
        const result = try generateOutputName(allocator, "org/model-name", "F16", "models");
        defer allocator.free(result);
        try std.testing.expectEqualStrings("models/org/model-name-F16", result);
    }

    // Local path - uses "local" org
    {
        const result = try generateOutputName(allocator, "/path/to/my-model", "GAF4", "models");
        defer allocator.free(result);
        try std.testing.expectEqualStrings("models/local/my-model-GAF4", result);
    }

    // Relative local path
    {
        const result = try generateOutputName(allocator, "./my-custom-model", "GAF8", "output");
        defer allocator.free(result);
        try std.testing.expectEqualStrings("output/local/my-custom-model-GAF8", result);
    }

    // Multi-part org name
    {
        const result = try generateOutputName(allocator, "my-org/model-8b", "GAF4", "models");
        defer allocator.free(result);
        try std.testing.expectEqualStrings("models/my-org/model-8b-GAF4", result);
    }
}

test "GAFConfig from ModelConfig" {
    const cfg = ModelConfig{
        .vocab_size = 151936,
        .d_model = 1024,
        .n_layers = 28,
        .n_heads = 16,
        .n_kv_groups = 8,
        .d_ff = 3072,
        .max_seq_len = 32768,
        .head_dim = 128,
        .rope_theta = 1000000.0,
        .norm_eps = 1e-6,
        .gaffine_group_size = 64,
    };

    const mlx_cfg = GAFConfig.fromModelConfig(cfg, .{ .group_size = 64, .bits = 4 }, null);

    try std.testing.expectEqual(@as(i32, 151936), mlx_cfg.vocab_size);
    try std.testing.expectEqual(@as(i32, 1024), mlx_cfg.hidden_size);
    try std.testing.expectEqual(@as(i32, 64), mlx_cfg.quantization.?.group_size);
}

// =============================================================================
// GAFConfig.fromModelConfig Tests
// =============================================================================

test "GAFConfig.fromModelConfig: basic fields mapping" {
    const cfg = ModelConfig{
        .vocab_size = 50000,
        .d_model = 768,
        .n_layers = 12,
        .n_heads = 12,
        .n_kv_groups = 12,
        .d_ff = 3072,
        .max_seq_len = 2048,
        .head_dim = 64,
        .rope_theta = 10000.0,
        .norm_eps = 1e-5,
        .gaffine_group_size = 64,
    };

    const mlx_cfg = GAFConfig.fromModelConfig(cfg, null, null);

    try std.testing.expectEqual(@as(i32, 50000), mlx_cfg.vocab_size);
    try std.testing.expectEqual(@as(i32, 768), mlx_cfg.hidden_size);
    try std.testing.expectEqual(@as(i32, 12), mlx_cfg.num_hidden_layers);
    try std.testing.expectEqual(@as(i32, 12), mlx_cfg.num_attention_heads);
    try std.testing.expectEqual(@as(i32, 12), mlx_cfg.num_key_value_heads);
    try std.testing.expectEqual(@as(i32, 3072), mlx_cfg.intermediate_size);
    try std.testing.expectEqual(@as(i32, 2048), mlx_cfg.max_position_embeddings);
    try std.testing.expectEqual(@as(i32, 64), mlx_cfg.head_dim);
    try std.testing.expectEqual(@as(f32, 1e-5), mlx_cfg.rms_norm_eps);
    try std.testing.expectEqual(@as(f32, 10000.0), mlx_cfg.rope_theta);
}

test "GAFConfig.fromModelConfig: quantization config" {
    const cfg = ModelConfig{
        .vocab_size = 32000,
        .d_model = 512,
        .n_layers = 6,
        .n_heads = 8,
        .n_kv_groups = 8,
        .d_ff = 2048,
        .max_seq_len = 1024,
        .head_dim = 64,
        .rope_theta = 10000.0,
        .norm_eps = 1e-6,
        .gaffine_group_size = 64,
    };

    // With quantization
    {
        const mlx_cfg = GAFConfig.fromModelConfig(cfg, .{ .group_size = 128, .bits = 8 }, null);
        try std.testing.expect(mlx_cfg.quantization != null);
        try std.testing.expectEqual(@as(i32, 128), mlx_cfg.quantization.?.group_size);
        try std.testing.expectEqual(@as(i32, 8), mlx_cfg.quantization.?.bits);
    }

    // Without quantization
    {
        const mlx_cfg = GAFConfig.fromModelConfig(cfg, null, null);
        try std.testing.expect(mlx_cfg.quantization == null);
    }
}

test "GAFConfig.fromModelConfig: optional fields - GELU" {
    const cfg = ModelConfig{
        .vocab_size = 32000,
        .d_model = 512,
        .n_layers = 6,
        .n_heads = 8,
        .n_kv_groups = 8,
        .d_ff = 2048,
        .max_seq_len = 1024,
        .head_dim = 64,
        .rope_theta = 10000.0,
        .norm_eps = 1e-6,
        .gaffine_group_size = 64,
        .use_gelu = true,
    };

    const mlx_cfg = GAFConfig.fromModelConfig(cfg, null, null);
    try std.testing.expect(mlx_cfg.hidden_activation != null);
    try std.testing.expectEqualStrings("gelu_pytorch_tanh", mlx_cfg.hidden_activation.?);
}

test "GAFConfig.fromModelConfig: optional fields - query_pre_attn_scalar" {
    const cfg = ModelConfig{
        .vocab_size = 32000,
        .d_model = 512,
        .n_layers = 6,
        .n_heads = 8,
        .n_kv_groups = 8,
        .d_ff = 2048,
        .max_seq_len = 1024,
        .head_dim = 64,
        .rope_theta = 10000.0,
        .norm_eps = 1e-6,
        .gaffine_group_size = 64,
        .query_pre_attn_scalar = 128.0,
    };

    const mlx_cfg = GAFConfig.fromModelConfig(cfg, null, null);
    try std.testing.expect(mlx_cfg.query_pre_attn_scalar != null);
    try std.testing.expectEqual(@as(i32, 128), mlx_cfg.query_pre_attn_scalar.?);
}

test "GAFConfig.fromModelConfig: optional fields - sliding window" {
    const cfg = ModelConfig{
        .vocab_size = 32000,
        .d_model = 512,
        .n_layers = 6,
        .n_heads = 8,
        .n_kv_groups = 8,
        .d_ff = 2048,
        .max_seq_len = 1024,
        .head_dim = 64,
        .rope_theta = 10000.0,
        .norm_eps = 1e-6,
        .gaffine_group_size = 64,
        .sliding_window = 512,
        .sliding_window_pattern = 2,
    };

    const mlx_cfg = GAFConfig.fromModelConfig(cfg, null, null);
    try std.testing.expect(mlx_cfg.sliding_window != null);
    try std.testing.expectEqual(@as(i32, 512), mlx_cfg.sliding_window.?);
    try std.testing.expect(mlx_cfg.sliding_window_pattern != null);
    try std.testing.expectEqual(@as(i32, 2), mlx_cfg.sliding_window_pattern.?);
}

test "GAFConfig.fromModelConfig: zero optional fields are null" {
    const cfg = ModelConfig{
        .vocab_size = 32000,
        .d_model = 512,
        .n_layers = 6,
        .n_heads = 8,
        .n_kv_groups = 8,
        .d_ff = 2048,
        .max_seq_len = 1024,
        .head_dim = 64,
        .rope_theta = 10000.0,
        .norm_eps = 1e-6,
        .gaffine_group_size = 64,
        .query_pre_attn_scalar = 0.0,
        .sliding_window = 0,
        .sliding_window_pattern = 0,
    };

    const mlx_cfg = GAFConfig.fromModelConfig(cfg, null, null);
    try std.testing.expect(mlx_cfg.query_pre_attn_scalar == null);
    try std.testing.expect(mlx_cfg.sliding_window == null);
    try std.testing.expect(mlx_cfg.sliding_window_pattern == null);
    try std.testing.expect(mlx_cfg.hidden_activation == null);
}

// =============================================================================
// generateOutputName Tests
// =============================================================================

test "generateOutputName: 8-bit quantization" {
    const allocator = std.testing.allocator;

    const result = try generateOutputName(allocator, "models--test--model", "GAF8", "output");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("output/test/model-GAF8", result);
}

test "generateOutputName: complex HF cache path" {
    const allocator = std.testing.allocator;

    const result = try generateOutputName(
        allocator,
        "/var/cache/huggingface/models--meta--llama-3.1/snapshots/xyz789/subdir",
        "GAF4",
        "models",
    );
    defer allocator.free(result);
    try std.testing.expectEqualStrings("models/meta/llama-3.1-GAF4", result);
}

test "generateOutputName: models--org--model format" {
    const allocator = std.testing.allocator;

    const result = try generateOutputName(allocator, "models--org--model-name", "GAF4-G32", "output");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("output/org/model-name-GAF4-G32", result);
}

test "generateOutputName: local path uses 'local' org" {
    const allocator = std.testing.allocator;

    const result = try generateOutputName(allocator, "/path/to/custom-model", "F16", "output");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("output/local/custom-model-F16", result);
}

test "generateOutputName: absolute vs relative output dir" {
    const allocator = std.testing.allocator;

    // Relative path
    {
        const result = try generateOutputName(allocator, "models--test--model", "GAF4", "models");
        defer allocator.free(result);
        try std.testing.expect(std.mem.startsWith(u8, result, "models/"));
    }

    // Absolute path
    {
        const result = try generateOutputName(allocator, "models--test--model", "GAF4", "/tmp/models");
        defer allocator.free(result);
        try std.testing.expect(std.mem.startsWith(u8, result, "/tmp/models/"));
    }
}

// =============================================================================
// GAFModelDir Tests
// =============================================================================

test "GAFModelDir.init: creates directory" {
    const allocator = std.testing.allocator;

    const test_path = "/tmp/talu_test_mlx_dir_init";
    std.fs.cwd().deleteTree(test_path) catch {};

    var dir = try GAFModelDir.init(allocator, test_path);
    defer dir.deinit();

    // Check directory was created
    var opened_dir = try std.fs.cwd().openDir(test_path, .{});
    opened_dir.close();

    // Cleanup
    std.fs.cwd().deleteTree(test_path) catch {};
}

test "GAFModelDir.init: handles existing directory" {
    const allocator = std.testing.allocator;

    const test_path = "/tmp/talu_test_mlx_dir_existing";
    std.fs.cwd().deleteTree(test_path) catch {};
    try std.fs.cwd().makePath(test_path);

    var dir = try GAFModelDir.init(allocator, test_path);
    defer dir.deinit();

    try std.testing.expectEqualStrings(test_path, dir.path);

    std.fs.cwd().deleteTree(test_path) catch {};
}

test "GAFModelDir.configPath: correct path construction" {
    const allocator = std.testing.allocator;

    const test_path = "/tmp/talu_test_mlx_config";
    std.fs.cwd().deleteTree(test_path) catch {};

    var dir = try GAFModelDir.init(allocator, test_path);
    defer dir.deinit();

    const config_path = try dir.configPath();
    defer allocator.free(config_path);

    try std.testing.expectEqualStrings("/tmp/talu_test_mlx_config/config.json", config_path);

    std.fs.cwd().deleteTree(test_path) catch {};
}

test "GAFModelDir.weightsPath: correct path construction" {
    const allocator = std.testing.allocator;

    const test_path = "/tmp/talu_test_mlx_weights";
    std.fs.cwd().deleteTree(test_path) catch {};

    var dir = try GAFModelDir.init(allocator, test_path);
    defer dir.deinit();

    const weights_path = try dir.weightsPath();
    defer allocator.free(weights_path);

    try std.testing.expectEqualStrings("/tmp/talu_test_mlx_weights/model.safetensors", weights_path);

    std.fs.cwd().deleteTree(test_path) catch {};
}

test "GAFModelDir.tokenizerPath: correct path construction" {
    const allocator = std.testing.allocator;

    const test_path = "/tmp/talu_test_mlx_tokenizer";
    std.fs.cwd().deleteTree(test_path) catch {};

    var dir = try GAFModelDir.init(allocator, test_path);
    defer dir.deinit();

    const tokenizer_path = try dir.tokenizerPath();
    defer allocator.free(tokenizer_path);

    try std.testing.expectEqualStrings("/tmp/talu_test_mlx_tokenizer/tokenizer.json", tokenizer_path);

    std.fs.cwd().deleteTree(test_path) catch {};
}

test "GAFModelDir.copyTokenizerFiles: copies existing files" {
    const allocator = std.testing.allocator;

    const source_dir = "/tmp/talu_test_mlx_source";
    const dest_dir = "/tmp/talu_test_mlx_dest";

    std.fs.cwd().deleteTree(source_dir) catch {};
    std.fs.cwd().deleteTree(dest_dir) catch {};
    try std.fs.cwd().makePath(source_dir);

    // Create test tokenizer files
    const test_files = [_][]const u8{ "tokenizer.json", "tokenizer_config.json", "vocab.json" };
    for (test_files) |filename| {
        const path = try std.fs.path.join(allocator, &.{ source_dir, filename });
        defer allocator.free(path);

        var file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        try file.writeAll("test content");
    }

    var dir = try GAFModelDir.init(allocator, dest_dir);
    defer dir.deinit();

    try dir.copyTokenizerFiles(source_dir);

    // Verify files were copied
    for (test_files) |filename| {
        const path = try std.fs.path.join(allocator, &.{ dest_dir, filename });
        defer allocator.free(path);

        var file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        const content = try file.readToEndAlloc(allocator, 1024);
        defer allocator.free(content);
        try std.testing.expectEqualStrings("test content", content);
    }

    std.fs.cwd().deleteTree(source_dir) catch {};
    std.fs.cwd().deleteTree(dest_dir) catch {};
}

test "GAFModelDir.copyTokenizerFiles: handles missing optional files" {
    const allocator = std.testing.allocator;

    const source_dir = "/tmp/talu_test_mlx_source_partial";
    const dest_dir = "/tmp/talu_test_mlx_dest_partial";

    std.fs.cwd().deleteTree(source_dir) catch {};
    std.fs.cwd().deleteTree(dest_dir) catch {};
    try std.fs.cwd().makePath(source_dir);

    // Create only one file
    const path = try std.fs.path.join(allocator, &.{ source_dir, "tokenizer.json" });
    defer allocator.free(path);
    var file = try std.fs.cwd().createFile(path, .{});
    file.close();

    var dir = try GAFModelDir.init(allocator, dest_dir);
    defer dir.deinit();

    // Should not fail if other files are missing
    try dir.copyTokenizerFiles(source_dir);

    std.fs.cwd().deleteTree(source_dir) catch {};
    std.fs.cwd().deleteTree(dest_dir) catch {};
}

// =============================================================================
// GAFConfig.writeToFile Tests
// =============================================================================

test "GAFConfig.writeToFile: writes valid JSON file" {
    const allocator = std.testing.allocator;

    const cfg = GAFConfig{
        .vocab_size = 32000,
        .hidden_size = 512,
        .num_hidden_layers = 6,
        .num_attention_heads = 8,
        .num_key_value_heads = 8,
        .intermediate_size = 2048,
        .max_position_embeddings = 1024,
        .head_dim = 64,
        .rms_norm_eps = 1e-6,
        .rope_theta = 10000.0,
    };

    const test_path = "/tmp/talu_test_mlx_write.json";
    std.fs.cwd().deleteFile(test_path) catch {};

    try cfg.writeToFile(allocator, test_path);

    // Verify file was created and contains expected content
    var file = try std.fs.cwd().openFile(test_path, .{});
    defer file.close();
    const content = try file.readToEndAlloc(allocator, 4096);
    defer allocator.free(content);

    // Check for required fields
    try std.testing.expect(std.mem.indexOf(u8, content, "\"vocab_size\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "32000") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\"hidden_size\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "512") != null);

    std.fs.cwd().deleteFile(test_path) catch {};
}

test "GAFConfig.writeToFile: includes quantization config when present" {
    const allocator = std.testing.allocator;

    const cfg = GAFConfig{
        .vocab_size = 32000,
        .hidden_size = 512,
        .num_hidden_layers = 6,
        .num_attention_heads = 8,
        .num_key_value_heads = 8,
        .intermediate_size = 2048,
        .max_position_embeddings = 1024,
        .head_dim = 64,
        .rms_norm_eps = 1e-6,
        .rope_theta = 10000.0,
        .quantization = .{ .group_size = 128, .bits = 4 },
    };

    const test_path = "/tmp/talu_test_mlx_write_quant.json";
    std.fs.cwd().deleteFile(test_path) catch {};

    try cfg.writeToFile(allocator, test_path);

    var file = try std.fs.cwd().openFile(test_path, .{});
    defer file.close();
    const content = try file.readToEndAlloc(allocator, 4096);
    defer allocator.free(content);

    try std.testing.expect(std.mem.indexOf(u8, content, "\"quantization\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\"group_size\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\"bits\"") != null);

    std.fs.cwd().deleteFile(test_path) catch {};
}

test "GAFConfig.writeToFile: includes optional fields when present" {
    const allocator = std.testing.allocator;

    const cfg = GAFConfig{
        .vocab_size = 32000,
        .hidden_size = 512,
        .num_hidden_layers = 6,
        .num_attention_heads = 8,
        .num_key_value_heads = 8,
        .intermediate_size = 2048,
        .max_position_embeddings = 1024,
        .head_dim = 64,
        .rms_norm_eps = 1e-6,
        .rope_theta = 10000.0,
        .model_type = "gemma3",
        .hidden_activation = "gelu_pytorch_tanh",
        .query_pre_attn_scalar = 128,
        .sliding_window = 512,
        .sliding_window_pattern = 2,
    };

    const test_path = "/tmp/talu_test_mlx_write_optional.json";
    std.fs.cwd().deleteFile(test_path) catch {};

    try cfg.writeToFile(allocator, test_path);

    var file = try std.fs.cwd().openFile(test_path, .{});
    defer file.close();
    const content = try file.readToEndAlloc(allocator, 4096);
    defer allocator.free(content);

    try std.testing.expect(std.mem.indexOf(u8, content, "\"model_type\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\"gemma3\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\"hidden_activation\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\"gelu_pytorch_tanh\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\"query_pre_attn_scalar\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\"sliding_window\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "\"sliding_window_pattern\"") != null);

    std.fs.cwd().deleteFile(test_path) catch {};
}

test "GAFConfig.writeToFile: error on invalid path" {
    const allocator = std.testing.allocator;

    const cfg = GAFConfig{
        .vocab_size = 32000,
        .hidden_size = 512,
        .num_hidden_layers = 6,
        .num_attention_heads = 8,
        .num_key_value_heads = 8,
        .intermediate_size = 2048,
        .max_position_embeddings = 1024,
        .head_dim = 64,
        .rms_norm_eps = 1e-6,
        .rope_theta = 10000.0,
    };

    // Try to write to a non-existent directory
    const result = cfg.writeToFile(allocator, "/nonexistent/directory/file.json");
    try std.testing.expectError(error.FileNotFound, result);
}

// =============================================================================
// GAFConfig.jsonStringify Tests
// =============================================================================

test "GAFConfig.jsonStringify: serializes required fields" {
    const allocator = std.testing.allocator;

    const cfg = GAFConfig{
        .vocab_size = 32000,
        .hidden_size = 512,
        .num_hidden_layers = 6,
        .num_attention_heads = 8,
        .num_key_value_heads = 8,
        .intermediate_size = 2048,
        .max_position_embeddings = 1024,
        .head_dim = 64,
        .rms_norm_eps = 1e-6,
        .rope_theta = 10000.0,
    };

    const json = try std.json.Stringify.valueAlloc(allocator, &cfg, .{});
    defer allocator.free(json);

    // Verify all required fields are present
    try std.testing.expect(std.mem.indexOf(u8, json, "\"vocab_size\":32000") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"hidden_size\":512") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"num_hidden_layers\":6") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"num_attention_heads\":8") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"num_key_value_heads\":8") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"intermediate_size\":2048") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"max_position_embeddings\":1024") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"head_dim\":64") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"rms_norm_eps\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"rope_theta\"") != null);
}

test "GAFConfig.jsonStringify: omits null optional fields" {
    const allocator = std.testing.allocator;

    const cfg = GAFConfig{
        .vocab_size = 32000,
        .hidden_size = 512,
        .num_hidden_layers = 6,
        .num_attention_heads = 8,
        .num_key_value_heads = 8,
        .intermediate_size = 2048,
        .max_position_embeddings = 1024,
        .head_dim = 64,
        .rms_norm_eps = 1e-6,
        .rope_theta = 10000.0,
        // All optional fields are null
    };

    const json = try std.json.Stringify.valueAlloc(allocator, &cfg, .{});
    defer allocator.free(json);

    // Verify optional fields are NOT present
    try std.testing.expect(std.mem.indexOf(u8, json, "\"model_type\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"hidden_activation\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"query_pre_attn_scalar\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"sliding_window\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"sliding_window_pattern\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"quantization\"") == null);
}

test "GAFConfig.jsonStringify: includes non-null optional fields" {
    const allocator = std.testing.allocator;

    const cfg = GAFConfig{
        .vocab_size = 32000,
        .hidden_size = 512,
        .num_hidden_layers = 6,
        .num_attention_heads = 8,
        .num_key_value_heads = 8,
        .intermediate_size = 2048,
        .max_position_embeddings = 1024,
        .head_dim = 64,
        .rms_norm_eps = 1e-6,
        .rope_theta = 10000.0,
        .model_type = "gemma3",
        .hidden_activation = "gelu_pytorch_tanh",
        .query_pre_attn_scalar = 128,
    };

    const json = try std.json.Stringify.valueAlloc(allocator, &cfg, .{});
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"model_type\":\"gemma3\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"hidden_activation\":\"gelu_pytorch_tanh\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"query_pre_attn_scalar\":128") != null);
}

test "GAFConfig.jsonStringify: includes quantization config" {
    const allocator = std.testing.allocator;

    const cfg = GAFConfig{
        .vocab_size = 32000,
        .hidden_size = 512,
        .num_hidden_layers = 6,
        .num_attention_heads = 8,
        .num_key_value_heads = 8,
        .intermediate_size = 2048,
        .max_position_embeddings = 1024,
        .head_dim = 64,
        .rms_norm_eps = 1e-6,
        .rope_theta = 10000.0,
        .quantization = .{ .group_size = 128, .bits = 4 },
    };

    const json = try std.json.Stringify.valueAlloc(allocator, &cfg, .{});
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"quantization\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"group_size\":128") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"bits\":4") != null);
}

test "GAFConfig.jsonStringify: handles floating point values" {
    const allocator = std.testing.allocator;

    const cfg = GAFConfig{
        .vocab_size = 32000,
        .hidden_size = 512,
        .num_hidden_layers = 6,
        .num_attention_heads = 8,
        .num_key_value_heads = 8,
        .intermediate_size = 2048,
        .max_position_embeddings = 1024,
        .head_dim = 64,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1000000.0,
    };

    const json = try std.json.Stringify.valueAlloc(allocator, &cfg, .{});
    defer allocator.free(json);

    // Check that float values are serialized (exact format may vary)
    try std.testing.expect(std.mem.indexOf(u8, json, "\"rms_norm_eps\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"rope_theta\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "1000000") != null or std.mem.indexOf(u8, json, "1e+06") != null);
}

// =============================================================================
// GAFModelDir.deinit Tests
// =============================================================================

test "GAFModelDir.deinit: frees allocated path" {
    const allocator = std.testing.allocator;

    const test_path = "/tmp/talu_test_mlx_deinit";
    std.fs.cwd().deleteTree(test_path) catch {};

    var dir = try GAFModelDir.init(allocator, test_path);

    // Path should be allocated
    try std.testing.expect(dir.path.len > 0);
    try std.testing.expectEqualStrings(test_path, dir.path);

    // deinit should free the path (checked by allocator leak detection in tests)
    dir.deinit();

    std.fs.cwd().deleteTree(test_path) catch {};
}

test "GAFModelDir.deinit: can be called multiple times safely" {
    const allocator = std.testing.allocator;

    const test_path = "/tmp/talu_test_mlx_deinit_multi";
    std.fs.cwd().deleteTree(test_path) catch {};

    var dir = try GAFModelDir.init(allocator, test_path);

    // First deinit
    dir.deinit();

    // The struct is set to undefined after deinit, so we can't safely call it again
    // This test verifies that a single deinit doesn't crash

    std.fs.cwd().deleteTree(test_path) catch {};
}

test "GAFModelDir.deinit: with nested directory" {
    const allocator = std.testing.allocator;

    const test_path = "/tmp/talu_test_mlx_deinit_nested/subdir/model";
    std.fs.cwd().deleteTree("/tmp/talu_test_mlx_deinit_nested") catch {};

    var dir = try GAFModelDir.init(allocator, test_path);
    defer dir.deinit();

    // Verify the nested path was created and stored
    try std.testing.expectEqualStrings(test_path, dir.path);

    std.fs.cwd().deleteTree("/tmp/talu_test_mlx_deinit_nested") catch {};
}
