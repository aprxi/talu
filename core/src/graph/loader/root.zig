//! Model Loader
//!
//! Entry point for loading models from disk. Loads SafeTensors models
//! and compute graph definitions from JSON.
//!
//! Architecture definitions are embedded in the binary at build time from
//! tools/archs/_graphs/*.json. Set TALU_GRAPHS_PATH to override with
//! filesystem paths (useful for development).
//!
//! Run `make graphs` to regenerate architecture definitions from Python models.

const std = @import("std");
const json = @import("../../io/json/root.zig");
const tensor = @import("../../tensor.zig");
const weights_impl = @import("weights.zig");
const moe = @import("moe.zig");
const graph = @import("../root.zig");
const log = @import("../../log.zig");
const embedded_graphs = @import("embedded_graphs");
const progress_mod = @import("../../capi/progress.zig");
const validation = @import("validation.zig");

// Generic MoE hooks for models that use Mixture of Experts
const moe_hooks = struct {
    pub const inferMoEFromWeights = moe.MoEHooks.inferMoEFromWeights;
};

// Re-export types
pub const weights = weights_impl;
pub const LoadedModel = weights_impl.LoadedModel;
pub const LoadOptions = weights_impl.LoadOptions;
pub const validateLoadedModel = validation.validate;

// Re-export validation types so check_coverage.sh --integration can verify test coverage
pub const Reporter = validation.Reporter;

// =============================================================================
// Model Loading
// =============================================================================

/// Load a model from SafeTensors format.
pub fn loadModel(
    backing_allocator: std.mem.Allocator,
    config_path: []const u8,
    weights_path: []const u8,
    load_options: weights_impl.LoadOptions,
    progress: progress_mod.ProgressContext,
) !LoadedModel {
    // Ensure graph registry is initialized
    graph.init(backing_allocator);

    // Load architecture definitions from _graphs/ directory
    _ = loadArchitectureDefinitions(backing_allocator);

    return loadSafeTensorsModel(backing_allocator, config_path, weights_path, load_options, progress);
}

/// Thread-safe state for architecture loading.
/// Uses three states: not_started (0), initializing (1), ready (2), failed (3).
const InitState = enum(u8) { not_started = 0, initializing = 1, ready = 2, failed = 3 };
var graph_init_state: std.atomic.Value(u8) = .{ .raw = 0 };

/// Load compute graph definitions.
/// Thread-safe: only one thread performs initialization, others spin-wait until ready.
///
/// Loading priority:
/// 1. TALU_GRAPHS_PATH environment variable (filesystem override for development)
/// 2. Embedded graphs compiled into the binary (default for distribution)
pub fn loadArchitectureDefinitions(allocator: std.mem.Allocator) bool {
    // Fast path: already initialized
    const init_state_value = graph_init_state.load(.acquire);
    if (init_state_value == @intFromEnum(InitState.ready)) {
        return true;
    }

    // Try to become the initializing thread
    if (graph_init_state.cmpxchgStrong(
        if (init_state_value == @intFromEnum(InitState.failed)) @intFromEnum(InitState.failed) else @intFromEnum(InitState.not_started),
        @intFromEnum(InitState.initializing),
        .acquire,
        .monotonic,
    )) |_| {
        // Another thread is initializing - spin-wait until ready or failed
        while (true) {
            const observed_state = graph_init_state.load(.acquire);
            if (observed_state == @intFromEnum(InitState.ready)) return true;
            if (observed_state == @intFromEnum(InitState.failed)) return false;
            std.atomic.spinLoopHint();
        }
    }

    // We are the initializing thread - do the actual work
    var any_graphs_loaded = false;

    // Priority 1: Check TALU_GRAPHS_PATH environment variable (filesystem override)
    if (std.posix.getenv("TALU_GRAPHS_PATH")) |graphs_env_path| {
        log.debug("load", "Loading from TALU_GRAPHS_PATH override", .{ .path = graphs_env_path }, @src());
        any_graphs_loaded = loadArchitecturesFromDir(allocator, graphs_env_path) > 0;
    } else {
        // Priority 2: Load from embedded graphs (compiled into binary)
        any_graphs_loaded = loadEmbeddedArchitectures();
    }

    if (any_graphs_loaded) {
        // Publish ready state with release ordering so all writes are visible
        graph_init_state.store(@intFromEnum(InitState.ready), .release);
    } else {
        log.debug("load", "No architectures loaded; registry not ready", .{}, @src());
        graph_init_state.store(@intFromEnum(InitState.failed), .release);
    }
    return any_graphs_loaded;
}

/// Load architectures from embedded graph data (compiled into binary).
fn loadEmbeddedArchitectures() bool {
    if (!embedded_graphs.hasGraphs()) {
        log.debug("load", "No embedded graphs available", .{}, @src());
        return false;
    }

    log.debug("load", "Loading architectures from embedded registry", .{}, @src());

    var loaded_count: usize = 0;
    const keys = embedded_graphs.graphs.keys();
    const values = embedded_graphs.graphs.values();

    for (keys, values) |name, json_content| {
        graph.loadFromJson(json_content) catch |err| {
            log.warn("load", "Failed to load embedded graph", .{
                .name = name,
                .reason = @errorName(err),
            });
            continue;
        };
        loaded_count += 1;
    }

    log.debug("load", "Loaded embedded architectures", .{ .count = loaded_count }, @src());
    return loaded_count > 0;
}

/// Load all .json architecture files from a directory (for TALU_GRAPHS_PATH override)
fn loadArchitecturesFromDir(allocator: std.mem.Allocator, dir_path: []const u8) usize {
    var dir = if (std.fs.path.isAbsolute(dir_path))
        std.fs.openDirAbsolute(dir_path, .{ .iterate = true }) catch {
            log.trace("load", "Graphs dir not found", .{ .path = dir_path }, @src());
            return 0;
        }
    else
        std.fs.cwd().openDir(dir_path, .{ .iterate = true }) catch {
            log.trace("load", "Graphs dir not found", .{ .path = dir_path }, @src());
            return 0;
        };
    defer dir.close();

    log.trace("load", "Scanning for architectures", .{ .path = dir_path }, @src());

    var loaded_count: usize = 0;
    var dir_iter = dir.iterate();
    while (dir_iter.next() catch null) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".json")) continue;

        const graph_path = std.fs.path.join(allocator, &.{ dir_path, entry.name }) catch continue;
        defer allocator.free(graph_path);

        if (graph.loadFromFile(graph_path)) |_| {
            loaded_count += 1;
        } else |_| {
            // Log failures at trace level since they're expected when probing multiple paths
        }
    }
    return loaded_count;
}

fn loadSafeTensorsModel(
    backing_allocator: std.mem.Allocator,
    config_path: []const u8,
    weights_path: []const u8,
    load_options: weights_impl.LoadOptions,
    progress: progress_mod.ProgressContext,
) !LoadedModel {
    // Detect model kind using graph registry (single source of truth)
    const model_kind = try detectModelKind(backing_allocator, config_path);

    // Load model using standard graph-driven loading with MoE support
    var loaded_model = try weights_impl.loadModelWithHooks(
        moe_hooks,
        backing_allocator,
        config_path,
        weights_path,
        model_kind.runtime_arch,
        load_options,
        progress,
    );
    errdefer loaded_model.deinit();

    // All architectures are now .custom - actual behavior comes from runtime_arch
    loaded_model.config.model_arch = .custom;

    // runtime_arch is guaranteed to exist (detectModelKind errors if not found)
    const runtime_architecture = model_kind.runtime_arch orelse unreachable;

    // Apply architecture properties from graph definition
    if (runtime_architecture.has_qk_norm) loaded_model.config.use_qk_norm = true;
    if (runtime_architecture.use_gelu) loaded_model.config.use_gelu = true;

    // Apply norm weight offset from graph (for (1+w) style norms)
    if (runtime_architecture.norm_weight_offset != 0.0) {
        loaded_model.runtime.weight_offset = runtime_architecture.norm_weight_offset;
        loaded_model.runtime.qk_norm_weight_offset = runtime_architecture.norm_weight_offset;
    }

    // Apply embedding multiplier (e.g., sqrt(hidden_size) scaling)
    if (runtime_architecture.embedding_multiplier != 1.0) {
        loaded_model.config.embedding_multiplier = runtime_architecture.embedding_multiplier;
    }

    // Check for explicit weight offset in add ops (fallback for graphs that define it inline)
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

    // Store runtime architecture for graph-driven execution
    loaded_model.runtime_arch = runtime_architecture;
    log.debug("load", "Set runtime_arch on loaded_model", .{
        .has_arch = loaded_model.runtime_arch != null,
    }, @src());

    // Apply SwiGLU variant from graph (alpha=1.702, clipping, (up+1) formulation)
    if (runtime_architecture.use_swiglu_oss) {
        loaded_model.runtime.use_swiglu_variant = true;
    }

    // Validate weight shapes against config after all inference hooks (inferDff, inferMoE,
    // etc.) have corrected config values. This catches config-vs-weight mismatches at load
    // time rather than producing silent corruption during inference.
    try validation.validate(&loaded_model);

    return loaded_model;
}

// =============================================================================
// Model Detection
// =============================================================================

/// Result of model kind detection.
/// Architecture is always .custom - actual behavior comes from runtime_arch.
const ModelKind = struct {
    arch: tensor.ModelArch = .custom,
    runtime_arch: ?*graph.Architecture = null,
};

/// Detect model kind using ONLY the graph registry.
/// The graph registry (populated from _graphs/*.json) is the single source of truth.
/// No hardcoded architecture detection - if no graph exists, the model is unsupported.
fn detectModelKind(allocator: std.mem.Allocator, config_path: []const u8) !ModelKind {
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
        // STRICT LOOKUP: The graph registry is the only source of truth
        if (graph.detectFromModelType(model_type_str)) |detected_arch| {
            log.trace("load", "Architecture found in graph registry", .{
                .model_type = model_type_str,
                .arch_name = detected_arch.name,
            }, @src());

            return .{
                .arch = .custom,
                .runtime_arch = detected_arch,
            };
        }

        // Model type not found in graph registry - this is an error
        log.err("load", "Unsupported model type: no graph definition found", .{
            .model_type = model_type_str,
        }, @src());
        return error.UnsupportedModel;
    }

    // No model_type field in config.json - cannot determine architecture
    log.err("load", "Missing model_type in config.json", .{}, @src());
    return error.UnsupportedModel;
}
