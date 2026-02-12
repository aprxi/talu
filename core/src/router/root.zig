//! Router - Routes inference requests to various destinations.
//!
//! The router is responsible for the first-level split:
//!   - **Repository-based** → LocalEngine → io/repository (loads weights locally)
//!   - **External API** → future backends (remote inference service)
//!
//! ## Model ID Classification
//!
//! The `::` separator identifies the backend (who runs inference).
//! Bare model IDs default to `native::` (talu's built-in inference engine).
//!
//! Native backend (talu's inference engine):
//!   - `org/model-name` - Implicit native, model from HF (cache first, then download)
//!   - `native::org/model-name` - Explicit native backend
//!   - `native::./my-model` - Native backend, local directory
//!   - `/path/to/model`, `./path`, `~/path` - Filesystem paths, native backend
//!
//! External API backends (remote inference via OpenAI-compatible API):
//!   - `vllm::org/model-name` - vLLM server (localhost:8000)
//!   - `ollama::model` - Ollama server (localhost:11434)
//!   - `llamacpp::model` - llama.cpp server (localhost:8080)
//!   - `lmstudio::model` - LM Studio (localhost:1234)
//!   - `localai::model` - LocalAI (localhost:8080)
//!   - `openai::gpt-4o` - OpenAI API (api.openai.com)
//!   - `openrouter::openai/gpt-4o` - OpenRouter (openrouter.ai)
//!
//! ## IMPORTANT: Context Transfer When Switching Models
//!
//! **WARNING**: When implementing external API routing, the router MUST handle
//! context transfer correctly when switching models mid-conversation.
//!
//! The problem: Different models have different tokenizers and chat templates.
//! If a user starts a conversation with one model and switches to another:
//!
//! ```python
//! chat = client.chat()  # Uses local model
//! chat("Hello")
//! chat("Math problem", model="openai::gpt-4o")  # Switch models!
//! ```
//!
//! The conversation history CANNOT be forwarded as raw text or tokens because:
//!   1. Tokenizers are different (each model has different vocabularies)
//!   2. Special tokens are different (`<|im_start|>` vs `<|start_header_id|>`)
//!   3. Chat template formats are different (ChatML vs OpenAI format)
//!
//! **Required implementation**:
//!   - Store conversation history as abstract Message structs (role + content)
//!   - Re-render the chat template for the target model at generation time
//!   - Re-tokenize using the target model's tokenizer
//!   - NEVER forward raw token IDs or pre-formatted prompt strings between models
//!
//! This is already the design in messages/messages.zig (stores role + content),
//! but the routing layer must ensure re-rendering happens on model switch.
//!
//! ## IMPORTANT: Backend-Specific Configuration
//!
//! **WARNING**: When implementing external API backends, each backend requires
//! different configuration options that must be exposed to Python users.
//!
//! Simple model IDs are not enough for production use:
//!
//! ```python
//! # Simple (insufficient for production)
//! Client("openai::gpt-4o")
//!
//! # Production needs (backend-specific options)
//! Client(
//!     targets=[
//!         talu.Target("native::org/model-name", device="cuda:0"),
//!         talu.Target("openai::gpt-4o", api_key="...", timeout=10.0, max_retries=3),
//!         talu.Target("vllm::hosted-model", endpoint="http://..."),
//!         talu.Target("anthropic::claude-3", organization_id="..."),
//!     ]
//! )
//! ```
//!
//! **Required implementation**:
//!   - Define a TargetConfig struct that holds backend-specific options
//!   - Native backend: device selection, memory limits, thread count
//!   - OpenAI/Anthropic: api_key, organization_id, timeout, max_retries, base_url
//!   - vLLM/Ollama: endpoint URL, auth headers, connection pooling
//!   - Pass TargetConfig from Python → C API → Zig router during initialization
//!   - Store configs in Router alongside model IDs
//!
//! The Python ModelTarget class (talu/chat/router.py) already has an `options` dict,
//! but the C API and Zig router need corresponding structs to receive these configs.
//!
//! ## Call Flow
//!
//! ```
//! capi/router.zig
//!     ↓
//! router/root.zig (classifyModel, isExternalApi) ← YOU ARE HERE
//!     │
//!     ├─ External API → error.ExternalApiNotSupported (future backends)
//!     │
//!     └─ Repository-based → router/local.zig (LocalEngine.init)
//!                               ↓
//!                           io/repository/root.zig (resolveModelPath)
//!                               ↓
//!                           io/repository/scheme.zig (parse URI scheme)
//! ```
//!
//! ## Usage
//!
//! ```zig
//! const router = @import("router/root.zig");
//!
//! // Check if model is external API
//! if (router.isExternalApi("openai::gpt-4o")) {
//!     return error.ExternalApiNotSupported;
//! }
//!
//! // native:: is NOT external API (it's talu's built-in engine)
//! if (!router.isExternalApi("native::org/model-name")) {
//!     // This is the native backend
//! }
//!
//! // Local inference
//! var engine = try router.LocalEngine.init(allocator, "org/model-name");
//! defer engine.deinit();
//!
//! const result = try engine.generate(&chat, .{});
//! ```

const std = @import("std");

pub const local = @import("local.zig");
pub const capi_bridge = @import("capi_bridge.zig");
pub const spec = @import("spec.zig");
pub const http_engine = @import("http_engine.zig");
pub const provider = @import("provider.zig");
pub const protocol = @import("protocol/root.zig");
pub const tool_schema = @import("tool_schema.zig");
pub const commit = @import("commit.zig");
pub const iterator = @import("iterator.zig");
const inference_mod = @import("../inference/root.zig");

// Model specification exports (from spec.zig)
pub const CanonicalSpec = spec.CanonicalSpec;
pub const InferenceBackend = spec.InferenceBackend;
pub const TaluModelSpec = spec.TaluModelSpec;
pub const TaluCapabilities = spec.TaluCapabilities;
pub const BackendType = spec.BackendType;
pub const BackendUnion = spec.BackendUnion;
pub const ValidationIssue = spec.ValidationIssue;
pub const ValidationResult = spec.ValidationResult;

// HTTP Engine exports (for remote OpenAI-compatible inference)
pub const HttpEngine = http_engine.HttpEngine;
pub const HttpEngineConfig = http_engine.HttpEngineConfig;
pub const HttpGenerateOptions = http_engine.GenerateOptions;
pub const HttpGenerationResult = http_engine.GenerationResult;
pub const HttpStreamCallback = http_engine.StreamCallback;
pub const HttpModelInfo = http_engine.ModelInfo;
pub const HttpListModelsResult = http_engine.ListModelsResult;

// Provider registry exports (for remote provider defaults)
pub const Provider = provider.Provider;
pub const ProviderParseResult = provider.ParseResult;
pub const PROVIDERS = provider.PROVIDERS;

// Primary exports
pub const LocalEngine = local.LocalEngine;
pub const GenerationResult = local.GenerationResult;
pub const GenerateOptions = local.GenerateOptions;
pub const TokenCallback = local.TokenCallback;
pub const ToolCallRef = local.ToolCallRef;

// Tool schema exports (for tool calling support)
pub const toolsToGrammarSchema = tool_schema.toolsToGrammarSchema;
pub const generateCallId = tool_schema.generateCallId;
pub const parseToolCall = tool_schema.parseToolCall;
pub const ParsedToolCall = tool_schema.ParsedToolCall;
pub const ToolSchemaError = tool_schema.ToolSchemaError;

// Inference types (for LocalEngine.run)
pub const InferenceConfig = inference_mod.session.InferenceConfig;
pub const InferenceState = inference_mod.session.InferenceState;
pub const FinishReason = inference_mod.session.FinishReason;

// Scheduler exports (continuous batching via LocalEngine.createScheduler)
pub const Scheduler = local.Scheduler;
pub const SchedulerConfig = local.SchedulerConfig;
pub const SchedulerRequest = local.SchedulerRequest;
pub const SchedulerRequestState = local.SchedulerRequestState;
pub const SchedulerTokenEvent = local.SchedulerTokenEvent;
pub const SchedulerSubmitOptions = local.SchedulerSubmitOptions;
pub const SamplingStrategy = local.SamplingStrategy;
pub const SamplingConfig = local.SamplingConfig;

// Embedding extraction
pub const PoolingStrategy = local.PoolingStrategy;
pub const ResolutionConfig = local.ResolutionConfig;

// Post-generation commit (shared by all backends)
pub const commitGenerationResult = commit.commitGenerationResult;
pub const CommitParams = commit.CommitParams;
pub const ToolCallInput = commit.ToolCallInput;

// Token iterator for pull-based streaming (no callbacks)
pub const TokenIterator = iterator.TokenIterator;

// =============================================================================
// Model Classification
// =============================================================================

/// Model destination type.
pub const ModelType = enum {
    /// Repository-based: weights loaded locally (even if downloaded from remote storage).
    /// Handled by LocalEngine → io/repository.
    repository,
    /// External API: remote inference service (OpenAI, Anthropic, Bedrock, etc.).
    /// Not yet implemented.
    external_api,
};

/// Classify a model identifier as repository-based or external API.
///
/// Detection logic:
///   - Contains `native::` → native backend (talu's inference engine)
///   - Contains `::` (other) → external API (e.g., `openai::gpt-4o`)
///   - Otherwise → native backend (bare model ID, URI scheme, or path)
pub fn classifyModel(model_id: []const u8) ModelType {
    // Check for :: namespace separator
    if (std.mem.indexOf(u8, model_id, "::")) |pos| {
        // native:: is the native backend, not external API
        if (pos == 6 and std.mem.startsWith(u8, model_id, "native::")) {
            return .repository;
        }
        // All other :: namespaces are external API
        return .external_api;
    }

    // Everything else is native backend (repository-based)
    // (bare model IDs, filesystem paths)
    return .repository;
}

/// Check if a model identifier refers to an external API.
pub fn isExternalApi(model_id: []const u8) bool {
    return classifyModel(model_id) == .external_api;
}

/// Check if a model identifier is repository-based.
pub fn isRepository(model_id: []const u8) bool {
    return classifyModel(model_id) == .repository;
}

/// Strip the "native::" prefix from a model ID if present.
/// Returns the original string if no prefix.
pub fn stripNativePrefix(model_id: []const u8) []const u8 {
    if (std.mem.startsWith(u8, model_id, "native::")) {
        return model_id[8..];
    }
    return model_id;
}

/// Resolve model ID for routing. Returns the model ID to pass to LocalEngine.
///
/// For native backend: returns the model ID (with native:: prefix stripped).
/// For external API backends: returns error.ExternalApiNotSupported.
pub fn resolveForRouting(model_id: []const u8) error{ExternalApiNotSupported}![]const u8 {
    if (isExternalApi(model_id)) {
        return error.ExternalApiNotSupported;
    }
    return stripNativePrefix(model_id);
}

// =============================================================================
// Tests
// =============================================================================

test "classifyModel: external API backends" {
    // Supported providers
    try std.testing.expectEqual(ModelType.external_api, classifyModel("vllm::org/model-name"));
    try std.testing.expectEqual(ModelType.external_api, classifyModel("ollama::model-name"));
    try std.testing.expectEqual(ModelType.external_api, classifyModel("llamacpp::model"));
    try std.testing.expectEqual(ModelType.external_api, classifyModel("lmstudio::model"));
    try std.testing.expectEqual(ModelType.external_api, classifyModel("localai::model"));
    try std.testing.expectEqual(ModelType.external_api, classifyModel("openai::gpt-4o"));
    try std.testing.expectEqual(ModelType.external_api, classifyModel("openrouter::openai/gpt-4o"));
    // Any :: prefix is classified as external_api (even unknown ones)
    try std.testing.expectEqual(ModelType.external_api, classifyModel("unknown::model"));
}

test "classifyModel: native backend models" {
    // Bare model IDs (implicit native::)
    try std.testing.expectEqual(ModelType.repository, classifyModel("org/model-name"));
    try std.testing.expectEqual(ModelType.repository, classifyModel("my-org/model-7b"));

    // Explicit native:: prefix
    try std.testing.expectEqual(ModelType.repository, classifyModel("native::org/model-name"));
    try std.testing.expectEqual(ModelType.repository, classifyModel("native::./my-model"));

    // Filesystem paths
    try std.testing.expectEqual(ModelType.repository, classifyModel("/path/to/model"));
    try std.testing.expectEqual(ModelType.repository, classifyModel("./my-model"));
    try std.testing.expectEqual(ModelType.repository, classifyModel("../other-model"));
    try std.testing.expectEqual(ModelType.repository, classifyModel("~/models/my-model"));
}

test "isExternalApi" {
    try std.testing.expect(isExternalApi("openai::gpt-4o"));
    try std.testing.expect(isExternalApi("vllm::model-name"));
    try std.testing.expect(!isExternalApi("org/model-name"));
    try std.testing.expect(!isExternalApi("native::org/model-name"));
    // Single colon is NOT external API
    try std.testing.expect(!isExternalApi("openai:gpt-4"));
}

test "stripNativePrefix" {
    try std.testing.expectEqualStrings("org/model-name", stripNativePrefix("native::org/model-name"));
    try std.testing.expectEqualStrings("./my-model", stripNativePrefix("native::./my-model"));
    try std.testing.expectEqualStrings("org/model-name", stripNativePrefix("org/model-name"));
}

test "resolveForRouting: native backend models succeed" {
    // Bare model ID
    try std.testing.expectEqualStrings("org/model-name", try resolveForRouting("org/model-name"));

    // With native:: prefix (stripped)
    try std.testing.expectEqualStrings("org/model-name", try resolveForRouting("native::org/model-name"));
    try std.testing.expectEqualStrings("./my-model", try resolveForRouting("native::./my-model"));

    // Filesystem paths
    try std.testing.expectEqualStrings("/path/to/model", try resolveForRouting("/path/to/model"));
    try std.testing.expectEqualStrings("./my-model", try resolveForRouting("./my-model"));
}

test "resolveForRouting: external API backends return error" {
    try std.testing.expectError(error.ExternalApiNotSupported, resolveForRouting("vllm::org/model-name"));
    try std.testing.expectError(error.ExternalApiNotSupported, resolveForRouting("ollama::model-name"));
    try std.testing.expectError(error.ExternalApiNotSupported, resolveForRouting("openai::gpt-4o"));
    try std.testing.expectError(error.ExternalApiNotSupported, resolveForRouting("openrouter::model"));
}

// =============================================================================
// Engine Cache
// =============================================================================

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
    engine_cache_mutex.lock();
    defer engine_cache_mutex.unlock();

    if (engine_cache.get(model_id)) |engine| {
        return engine;
    }

    // Create new engine
    const engine = try allocator.create(LocalEngine);
    errdefer allocator.destroy(engine);

    engine.* = try LocalEngine.initWithResolutionConfig(allocator, model_id, config);

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
