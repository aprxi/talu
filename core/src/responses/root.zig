//! Responses - local LLM response serving.
//!
//! This module owns the local response-serving layer: conversation state,
//! endpoint protocol adapters, batching, tool/schema handling, local model
//! execution, and response commit logic.

const std = @import("std");
const progress_mod = @import("progress_pkg");

pub const local = @import("local.zig");
pub const capi_bridge = @import("capi_bridge.zig");
pub const conversation = @import("conversation/root.zig");
pub const embeddings = @import("embeddings.zig");
pub const spec = @import("spec.zig");
pub const protocol = @import("protocol/root.zig");
pub const tool_schema = @import("tool_schema.zig");
pub const commit = @import("commit.zig");
pub const batch = @import("batch.zig");
const inference_bridge = @import("inference_bridge.zig");

// Model specification exports (from spec.zig)
pub const CanonicalSpec = spec.CanonicalSpec;
pub const InferenceBackend = spec.InferenceBackend;
pub const TaluModelSpec = spec.TaluModelSpec;
pub const TaluCapabilities = spec.TaluCapabilities;
pub const BackendType = spec.BackendType;
pub const BackendUnion = spec.BackendUnion;

// Primary exports
pub const LocalEngine = local.LocalEngine;
pub const GenerateOptions = local.GenerateOptions;
pub const ToolCallRef = local.ToolCallRef;

// Tool schema exports (for tool calling support)
pub const toolsToGrammarSchema = tool_schema.toolsToGrammarSchema;
pub const generateCallId = tool_schema.generateCallId;
pub const parseToolCall = tool_schema.parseToolCall;
pub const parseToolCallsFromText = tool_schema.parseToolCallsFromText;
pub const normalizeToolsJson = tool_schema.normalizeToolsJson;
pub const ParsedToolCall = tool_schema.ParsedToolCall;
pub const ToolSchemaError = tool_schema.ToolSchemaError;

// Inference types (for LocalEngine.run)
pub const InferenceConfig = inference_bridge.types.InferenceConfig;
pub const InferenceState = inference_bridge.types.InferenceState;
pub const FinishReason = inference_bridge.types.FinishReason;

// Scheduler exports (continuous batching via LocalEngine.createScheduler)
pub const Scheduler = local.Scheduler;
pub const SchedulerConfig = local.SchedulerConfig;
pub const SchedulerRequest = local.SchedulerRequest;
pub const SchedulerRequestState = local.SchedulerRequestState;
pub const SchedulerTokenEvent = local.SchedulerTokenEvent;
pub const SamplingStrategy = local.SamplingStrategy;
pub const SamplingConfig = local.SamplingConfig;

// Embedding extraction
pub const PoolingStrategy = embeddings.PoolingStrategy;
pub const ResolutionConfig = local.ResolutionConfig;

// Post-generation commit (shared by all backends)
pub const commitGenerationResult = commit.commitGenerationResult;

// Batch API (continuous batching for response generation)
pub const BatchWrapper = batch.BatchWrapper;
pub const BatchEvent = batch.BatchEvent;
pub const BatchResult = batch.BatchResult;

/// Global engine cache - maps model identifiers to loaded engines.
/// Thread-safe with mutex protection.
var engine_cache: std.StringHashMapUnmanaged(*LocalEngine) = .{};
var engine_cache_mutex: std.Thread.Mutex = .{};

/// Get or create an engine for a model identifier.
/// The engine is cached for reuse across multiple requests.
/// LocalEngine.init handles all path resolution (local paths, cache paths, HF model IDs).
pub fn getOrCreateEngine(allocator: std.mem.Allocator, model_id: []const u8) !*LocalEngine {
    return getOrCreateEngineWithConfig(allocator, model_id, .{});
}

/// Get or create an engine for a model identifier with resolution config.
pub fn getOrCreateEngineWithConfig(
    allocator: std.mem.Allocator,
    model_id: []const u8,
    config: ResolutionConfig,
) !*LocalEngine {
    return getOrCreateEngineWithBackendConfig(allocator, model_id, config, .{});
}

/// Get or create an engine for a model identifier with resolution and backend config.
pub fn getOrCreateEngineWithBackendConfig(
    allocator: std.mem.Allocator,
    model_id: []const u8,
    config: ResolutionConfig,
    backend_init_options: local.BackendInitOptions,
) !*LocalEngine {
    engine_cache_mutex.lock();
    defer engine_cache_mutex.unlock();

    if (engine_cache.get(model_id)) |engine| {
        return engine;
    }

    // Create new engine
    const engine = try allocator.create(LocalEngine);
    errdefer allocator.destroy(engine);

    engine.* = try LocalEngine.initWithSeedAndResolutionConfig(
        allocator,
        model_id,
        42,
        config,
        backend_init_options,
        progress_mod.Context.NONE,
    );

    // Cache with copied key
    const key = try allocator.dupe(u8, model_id);
    try engine_cache.put(allocator, key, engine);

    return engine;
}

/// Close all cached engines and free resources.
pub fn closeAllEngines(allocator: std.mem.Allocator) void {
    engine_cache_mutex.lock();
    defer engine_cache_mutex.unlock();

    var iter = engine_cache.iterator();
    while (iter.next()) |entry| {
        entry.value_ptr.*.deinit();
        allocator.destroy(entry.value_ptr.*);
        allocator.free(entry.key_ptr.*);
    }
    engine_cache.clearRetainingCapacity();
}
