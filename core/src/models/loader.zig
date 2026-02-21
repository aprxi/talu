//! Model Loader
//!
//! Entry point for loading models from disk using static Zig architecture
//! metadata from `core/src/models/*`.

const std = @import("std");
const json = @import("../io/json/root.zig");
const tensor = @import("../tensor.zig");
const op_types = @import("op_types.zig");
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

    // Check for explicit weight offset in add ops.
    for (runtime_architecture.block_ops) |op| {
        if (op.op_type == .add) {
            var op_scalar_value: f32 = 0.0;
            var op_has_tensor = false;
            for (op.inputs) |inp| {
                switch (inp) {
                    .scalar => |s| op_scalar_value = s,
                    .tensor => op_has_tensor = true,
                }
            }
            if (op_has_tensor and op_scalar_value != 0.0) {
                loaded_model.runtime.weight_offset = op_scalar_value;
                loaded_model.runtime.qk_norm_weight_offset = op_scalar_value;
                break;
            }
        }
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
    const config_bytes = try std.fs.cwd().readFileAlloc(allocator, config_path, 256 * 1024);
    defer allocator.free(config_bytes);

    const parsed_json = json.parseValue(allocator, config_bytes, .{ .max_size_bytes = 256 * 1024 }) catch |err| {
        return switch (err) {
            error.InputTooLarge => error.InvalidJson,
            error.InputTooDeep => error.InvalidJson,
            error.StringTooLong => error.InvalidJson,
            error.InvalidJson => error.InvalidJson,
            error.OutOfMemory => error.OutOfMemory,
        };
    };
    defer parsed_json.deinit();
    if (parsed_json.value != .object) return error.InvalidJson;
    const config_obj = parsed_json.value.object;

    const model_type = if (config_obj.get("model_type")) |v| switch (v) {
        .string => |s| s,
        else => null,
    } else null;

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
            };
        }

        // Model type not found in static model registry - unsupported.
        log.err("load", "Unsupported model type: not in static model registry", .{
            .model_type = model_type_str,
        }, @src());
        return error.UnsupportedModel;
    }

    // No model_type field in config.json - cannot determine architecture
    log.err("load", "Missing model_type in config.json", .{}, @src());
    return error.UnsupportedModel;
}
