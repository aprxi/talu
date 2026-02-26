//! Model Loader
//!
//! Entry point for loading models from disk using static Zig architecture
//! metadata from `core/src/models/*`.

const std = @import("std");
const tensor = @import("../tensor.zig");
const op_types = @import("op_types.zig");
const cfg_loader = @import("config/root.zig");
const weights_impl = @import("load/weights.zig");
const models_registry = @import("registry.zig");
const log = @import("../log.zig");
const progress_mod = @import("../progress.zig");
const validation = @import("load/validation.zig");

// Re-export types
pub const weights = weights_impl;
pub const LoadedModel = weights_impl.LoadedModel;
pub const LoadOptions = weights_impl.LoadOptions;
pub const validateLoadedModel = validation.validate;

// Re-export validation types so check_coverage.sh --integration can verify test coverage
pub const Reporter = validation.Reporter;
pub const ResolvedModelKind = struct {
    descriptor: models_registry.Entry,
    runtime_arch: *const op_types.Architecture,
    parse_config_hook: ?op_types.ConfigParseHook,
};

// =============================================================================
// Model Loading
// =============================================================================

/// Load a model from SafeTensors format.
pub fn loadModel(
    backing_allocator: std.mem.Allocator,
    config_path: []const u8,
    weights_path: []const u8,
    load_options: weights_impl.LoadOptions,
    progress: progress_mod.Context,
) !LoadedModel {
    _ = loadArchitectureDefinitions(backing_allocator);
    return loadSafeTensorsModel(backing_allocator, config_path, weights_path, load_options, progress);
}

/// Static model metadata is compiled into Zig sources, so there is no runtime
/// architecture loading step.
pub fn loadArchitectureDefinitions(allocator: std.mem.Allocator) bool {
    _ = allocator;
    return true;
}

fn loadSafeTensorsModel(
    backing_allocator: std.mem.Allocator,
    config_path: []const u8,
    weights_path: []const u8,
    load_options: weights_impl.LoadOptions,
    progress: progress_mod.Context,
) !LoadedModel {
    // Detect model kind using static model metadata.
    const model_kind = try resolveModelKindForConfig(backing_allocator, config_path);

    // Load model using architecture-driven metadata with MoE support.
    var loaded_model = try weights_impl.loadModelWithArchitecture(
        backing_allocator,
        config_path,
        weights_path,
        model_kind.runtime_arch,
        model_kind.parse_config_hook,
        load_options,
        progress,
    );
    errdefer loaded_model.deinit();

    applyRuntimeArchitectureMetadata(&loaded_model, model_kind.runtime_arch);

    // Validate weight shapes against config after all inference hooks (inferDff, inferMoE,
    // etc.) have corrected config values. This catches config-vs-weight mismatches at load
    // time rather than producing silent corruption during inference.
    try validation.validate(&loaded_model);

    return loaded_model;
}

// =============================================================================
// Model Detection
// =============================================================================

pub fn applyRuntimeArchitectureMetadata(
    loaded_model: *LoadedModel,
    runtime_architecture: *const op_types.Architecture,
) void {
    // All architectures are now .custom - actual behavior comes from runtime_arch
    loaded_model.config.model_arch = .custom;

    // Apply architecture properties from static metadata.
    if (runtime_architecture.has_qk_norm) loaded_model.config.use_qk_norm = true;
    if (runtime_architecture.use_gelu) loaded_model.config.use_gelu = true;

    // Apply norm weight offset (for (1+w) style norms).
    if (runtime_architecture.norm_weight_offset != 0.0) {
        loaded_model.runtime.weight_offset = runtime_architecture.norm_weight_offset;
        loaded_model.runtime.qk_norm_weight_offset = runtime_architecture.norm_weight_offset;
    }

    // Apply embedding multiplier (e.g., sqrt(hidden_size) scaling)
    if (runtime_architecture.embedding_multiplier != 1.0) {
        loaded_model.config.embedding_multiplier = runtime_architecture.embedding_multiplier;
    }

    // Store runtime architecture metadata needed by inference runtime.
    loaded_model.runtime.architecture_id = runtime_architecture.name;
    loaded_model.runtime.has_moe = runtime_architecture.has_moe;
    loaded_model.runtime.has_mamba = runtime_architecture.has_mamba;
    loaded_model.runtime.has_shortconv = runtime_architecture.has_shortconv;
    loaded_model.runtime.has_mla = runtime_architecture.has_mla;
    loaded_model.runtime.explicit_qk_norm_ops = runtime_architecture.explicit_qk_norm_ops;
    log.debug("load", "Set runtime architecture metadata", .{
        .architecture = runtime_architecture.name,
    }, @src());

    // Apply SwiGLU variant (alpha=1.702, clipping, (up+1) formulation).
    if (runtime_architecture.use_swiglu_oss) {
        loaded_model.runtime.use_swiglu_variant = true;
    }
}

pub fn runtimeArchitectureForConfig(
    allocator: std.mem.Allocator,
    config_path: []const u8,
) !*const op_types.Architecture {
    const resolved = try resolveModelKindForConfig(allocator, config_path);
    return resolved.runtime_arch;
}

/// Detect model kind using the static model registry for model_type mapping,
/// then resolve the canonical static architecture payload.
pub fn resolveModelKindForConfig(
    allocator: std.mem.Allocator,
    config_path: []const u8,
) !ResolvedModelKind {
    var base_config = try cfg_loader.readBaseConfig(allocator, config_path);
    defer base_config.deinit(allocator);
    const model_type = base_config.model_type;

    log.trace("load", "Detecting model kind", .{ .model_type = model_type orelse "unknown" }, @src());

    if (model_type) |model_type_str| {
        if (models_registry.detectByModelType(model_type_str)) |entry| {
            const detected_arch = models_registry.runtimeArchitectureById(entry.id) orelse {
                log.err("load", "Missing architecture payload for supported model", .{
                    .model_type = model_type_str,
                    .architecture = entry.id,
                }, @src());
                return error.UnsupportedModel;
            };
            log.trace("load", "Architecture resolved from static model registry", .{
                .model_type = model_type_str,
                .arch_name = entry.id,
            }, @src());

            return .{
                .descriptor = entry,
                .runtime_arch = detected_arch,
                .parse_config_hook = models_registry.configParseHookFor(entry),
            };
        }

        // Model type not found in static model registry - unsupported.
        log.err("load", "Unsupported model type: not in static model registry", .{
            .model_type = model_type_str,
        }, @src());
        return error.UnsupportedModel;
    }

    if (base_config.architecture) |architecture_name| {
        if (models_registry.detectByArchitectureId(architecture_name)) |entry| {
            const detected_arch = models_registry.runtimeArchitectureById(entry.id) orelse {
                log.err("load", "Missing architecture payload for supported architecture id", .{
                    .architecture = architecture_name,
                    .entry = entry.id,
                }, @src());
                return error.UnsupportedModel;
            };
            log.trace("load", "Architecture resolved from config architectures[] fallback", .{
                .architecture = architecture_name,
                .entry = entry.id,
            }, @src());
            return .{
                .descriptor = entry,
                .runtime_arch = detected_arch,
                .parse_config_hook = models_registry.configParseHookFor(entry),
            };
        }
    }

    // No model_type or supported architecture id in config.json.
    log.err("load", "Missing model_type in config.json", .{}, @src());
    return error.UnsupportedModel;
}

test "loadArchitectureDefinitions returns true for static metadata" {
    try std.testing.expect(loadArchitectureDefinitions(std.testing.allocator));
}

test "resolveModelKindForConfig and runtimeArchitectureForConfig detect known model_type" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(.{ .sub_path = "config.json", .data = "{\"model_type\":\"llama3\"}" });
    const config_path = try tmp.dir.realpathAlloc(allocator, "config.json");
    defer allocator.free(config_path);

    const resolved = try resolveModelKindForConfig(allocator, config_path);
    try std.testing.expectEqualStrings("llama3", resolved.descriptor.id);
    try std.testing.expectEqualStrings("llama3", resolved.runtime_arch.name);

    const runtime_arch = try runtimeArchitectureForConfig(allocator, config_path);
    try std.testing.expectEqualStrings("llama3", runtime_arch.name);
}

test "resolveModelKindForConfig falls back to text_config model_type" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(.{
        .sub_path = "config.json",
        .data = "{\"text_config\":{\"model_type\":\"llama3\"}}",
    });
    const config_path = try tmp.dir.realpathAlloc(allocator, "config.json");
    defer allocator.free(config_path);

    const resolved = try resolveModelKindForConfig(allocator, config_path);
    try std.testing.expectEqualStrings("llama3", resolved.descriptor.id);
    try std.testing.expectEqualStrings("llama3", resolved.runtime_arch.name);
}

test "resolveModelKindForConfig falls back to architectures id when model_type missing" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(.{
        .sub_path = "config.json",
        .data = "{\"architectures\":[\"llama3\"]}",
    });
    const config_path = try tmp.dir.realpathAlloc(allocator, "config.json");
    defer allocator.free(config_path);

    const resolved = try resolveModelKindForConfig(allocator, config_path);
    try std.testing.expectEqualStrings("llama3", resolved.descriptor.id);
    try std.testing.expectEqualStrings("llama3", resolved.runtime_arch.name);
}

test "applyRuntimeArchitectureMetadata copies runtime flags from architecture" {
    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = .{
            .vocab_size = 16,
            .d_model = 8,
            .n_layers = 1,
            .n_heads = 1,
            .n_kv_groups = 1,
            .d_ff = 16,
            .max_seq_len = 64,
            .head_dim = 8,
            .rope_theta = 10000.0,
            .norm_eps = 1e-5,
            .gaffine_group_size = 128,
        },
        .token_embeddings = .{
            .dtype = .f32,
            .n_dims = 2,
            .shape = .{ 1, 1, 0, 0, 0, 0, 0, 0 },
            .data_ptr = null,
            .data_size = 0,
            .numel = 1,
            .strides = .{ 1, 1, 0, 0, 0, 0, 0, 0 },
        },
        .blocks = &.{},
        .original_weight_dtype = .f32,
    };
    defer loaded.arena.deinit();

    const arch = models_registry.runtimeArchitectureById("granite_hybrid") orelse return error.TestUnexpectedResult;
    applyRuntimeArchitectureMetadata(&loaded, arch);
    try std.testing.expect(loaded.runtime.architecture_id != null);
    try std.testing.expectEqualStrings("granite_hybrid", loaded.runtime.architecture_id.?);
    try std.testing.expect(loaded.runtime.has_mamba);
}

test "loadModel returns FileNotFound for missing config path" {
    const err_result = loadModel(
        std.testing.allocator,
        "/tmp/talu_missing_model_config.json",
        "/tmp/talu_missing_weights.safetensors",
        .{},
        progress_mod.Context.NONE,
    );
    try std.testing.expectError(error.FileNotFound, err_result);
}
