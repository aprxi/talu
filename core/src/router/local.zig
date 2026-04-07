//! Local - Local inference destination for the router.
//!
//! LocalEngine loads a model and handles all local inference. Multiple Chats
//! can share a single engine for efficient multi-user serving.
//!
//! This is the default destination when routing inference requests locally.
//! Other destinations (OpenAI, Anthropic, vLLM, etc.) will be added as
//! separate modules in the router package.
//!
//! Example:
//!     // Load model once (at server startup)
//!     var engine = try LocalEngine.init(allocator, "path/to/model");
//!     defer engine.deinit();
//!
//!     // Create lightweight chats per user
//!     var user1 = Chat.init(allocator);
//!     defer user1.deinit();
//!     try user1.setSystem("You are helpful.");
//!     try user1.append(.user, "Hello!");
//!
//!     // Generate response
//!     const result = try engine.generate(&user1, .{});
//!     defer result.deinit(allocator);

const std = @import("std");
const inference = @import("../inference/root.zig");
const models = @import("../models/root.zig");
const responses_mod = @import("../responses/root.zig");
const Chat = responses_mod.Chat;
const protocol = @import("protocol/root.zig");
const sampler = inference.sampling;
const inference_types = inference.types;
const FinishReason = inference_types.FinishReason;
const backend_root = @import("../inference/backend/root.zig");
const Backend = backend_root.Backend;
const vision_types = inference.vision_types;
const log = @import("../log.zig");
pub const PoolingStrategy = backend_root.PoolingStrategy;
const tokenizer_mod = @import("../tokenizer/root.zig");
const io = @import("../io/root.zig");
const image_mod = @import("../image/root.zig");
const io_json = @import("../io/json/root.zig");
const gen_config_mod = @import("../inference/config/generation.zig");
const preproc_mod = @import("../inference/config/preprocessor.zig");
const validate_mod = @import("../validate/root.zig");
const error_context = @import("../error_context.zig");
const ConstrainedSampler = validate_mod.sampler.ConstrainedSampler;
const GrammarConfig = validate_mod.sampler.GrammarConfig;
const tool_schema_mod = @import("tool_schema.zig");
const reasoning_parser_mod = responses_mod.reasoning_parser;
const commit_mod = @import("commit.zig");
const chat_template = @import("../template/chat_template.zig");
const template_mod = @import("../template/root.zig");
const progress_mod = @import("../capi/progress.zig");
const runtime_contract = inference.runtime_contract;

pub const ResolutionConfig = io.repository.ResolutionConfig;
pub const BackendInitOptions = backend_root.InitOptions;

// Re-export scheduler types for router API
pub const BackendScheduler = inference.scheduler.GenericScheduler(Backend);
pub const Scheduler = BackendScheduler;
pub const SchedulerConfig = inference.SchedulerConfig;
pub const SchedulerRequest = inference.Request;
pub const SchedulerRequestState = inference.RequestState;
pub const SchedulerTokenEvent = inference.TokenEvent;
pub const SchedulerSubmitOptions = BackendScheduler.SubmitOptions;
pub const SamplingStrategy = inference.SamplingStrategy;
pub const SamplingConfig = inference.SamplingConfig;

fn finishReasonToString(reason: FinishReason) [:0]const u8 {
    return switch (reason) {
        .eos_token => "stop",
        .length => "length",
        .stop_sequence => "stop_sequence",
        .tool_calls => "tool_calls",
        .content_filter => "content_filter",
        .cancelled => "cancelled",
    };
}

/// Reference to a tool call in the Conversation.
pub const ToolCallRef = struct {
    /// Index in Conversation.items
    item_index: usize,

    /// Unique identifier for this call (e.g., "call_abc123")
    call_id: []const u8,

    /// Function name
    name: []const u8,

    /// Function arguments JSON
    arguments: []const u8,
};

/// Result from engine.generate().
pub const GenerationResult = struct {
    /// Generated text.
    text: []const u8,

    /// Generated token IDs.
    tokens: []const u32,

    /// Number of tokens in the prompt.
    prompt_tokens: usize,

    /// Number of tokens generated.
    generated_tokens: usize,

    /// Prefill time in nanoseconds.
    prefill_ns: u64,

    /// Decode time in nanoseconds.
    decode_ns: u64,

    /// Why generation stopped.
    finish_reason: FinishReason = .eos_token,

    /// Tool calls requested by the model (if finish_reason == .tool_calls).
    /// Both the tool calls and the text have been committed to the Conversation
    /// via commit.commitGenerationResult() before this result is returned.
    tool_calls: ?[]const ToolCallRef = null,

    /// Free the result's memory.
    pub fn deinit(self: *const GenerationResult, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
        allocator.free(self.tokens);
        if (self.tool_calls) |calls| {
            for (calls) |call| {
                allocator.free(call.call_id);
                allocator.free(call.name);
                allocator.free(call.arguments);
            }
            allocator.free(calls);
        }
    }
};

/// Generation options that override Chat defaults.
pub const GenerateOptions = struct {
    /// Override chat's max_tokens (total hard ceiling: thinking + answer).
    max_tokens: ?usize = null,

    /// Maximum tokens for the answer/completion only (excludes thinking).
    /// When set with max_tokens: answer is capped, thinking gets the rest.
    /// When set without max_tokens: auto-computes max_tokens = thinking_budget + this.
    max_completion_tokens: ?usize = null,

    /// Maximum thinking/reasoning tokens. When set, overrides the budget
    /// derived from reasoning_effort. 0 = no thinking.
    max_reasoning_tokens: ?usize = null,

    /// Override chat's temperature.
    temperature: ?f32 = null,

    /// Override chat's top_k.
    top_k: ?usize = null,

    /// Override chat's top_p.
    top_p: ?f32 = null,

    /// Override chat's min_p.
    min_p: ?f32 = null,

    /// Override chat's repetition_penalty.
    repetition_penalty: ?f32 = null,

    /// Additive presence penalty (0.0 = disabled).
    presence_penalty: ?f32 = null,

    /// Additive frequency penalty (0.0 = disabled).
    frequency_penalty: ?f32 = null,

    /// Optional callback for streaming output. Called after each token is sampled.
    token_callback: ?TokenCallback = null,

    /// User data passed to the token callback.
    callback_data: ?*anyopaque = null,

    /// Stop sequences (already tokenized). Generation stops when any sequence matches.
    /// The stop sequence is NOT included in the output.
    stop_sequences: []const []const u32 = &.{},

    /// Logit bias entries: add bias values to specific token logits before sampling.
    /// Positive values increase probability, negative decrease it.
    /// Use -100 or lower to effectively ban a token.
    logit_bias: ?[]const sampler.LogitBiasEntry = null,

    /// Random seed for reproducibility. 0 = don't reseed (use engine's current state).
    /// Non-zero values reseed the sampler before generation for deterministic output.
    seed: u64 = 0,

    /// Custom chat template to use instead of model's template.
    /// If null, uses the template from tokenizer_config.json or chat_template.jinja.
    template_override: ?[]const u8 = null,

    /// Extra context variables to inject into the template as JSON object.
    /// Must be a JSON object (not array), e.g., {"tools": [...], "date": "2024-01-15"}.
    /// These variables become available in the template alongside messages, bos_token, etc.
    extra_context_json: ?[]const u8 = null,

    /// Reasoning effort level: "none", "low", "medium", "high", "xhigh".
    /// When non-null, the router maps this to template context variables
    /// (e.g. enable_thinking=true for effort != "none").
    reasoning_effort: ?[]const u8 = null,

    /// Tool definitions as JSON array of OpenAI-compatible tool schemas.
    /// When provided, enables grammar-based constrained sampling to enforce
    /// valid tool call JSON, and triggers auto-commit of FunctionCallItem
    /// to the Conversation when a tool call is detected.
    ///
    /// Format: [{"type":"function","function":{"name":"...","parameters":{...}}}]
    tools_json: ?[]const u8 = null,

    /// Tool choice strategy. Controls when tools are used:
    /// - "auto" (default): Model decides whether to call tools
    /// - "required": Model must call at least one tool
    /// - "none": Disable tool calling even if tools are provided
    /// - "<function_name>": Force model to call specific function
    tool_choice: ?[]const u8 = null,

    /// Optional prefill progress callback. Called once per transformer layer
    /// during prefill (not decode). Signature: fn(completed_layers, total_layers, userdata).
    prefill_progress_fn: ?PrefillProgressFn = null,

    /// User data passed to the prefill progress callback.
    prefill_progress_data: ?*anyopaque = null,

    /// Optional stop flag for cancellation. When set to true, generation stops.
    /// This allows external cancellation (e.g., client disconnect, asyncio.CancelledError)
    /// without waiting for the next callback invocation.
    stop_flag: ?*const std.atomic.Value(bool) = null,

    /// Reasoning tag name for post-generation parsing.
    /// When non-null, the parser looks for `<tag>...</tag>` markers and
    /// separates reasoning from response content into distinct items.
    /// Default (null) uses "think" (`<think>...</think>`).
    reasoning_tag: ?[]const u8 = null,

    /// When true, preserve raw model output text (including reasoning tags).
    /// Default false keeps parsing `<think>...</think>` into typed items and
    /// returns only the assistant response text to callers.
    raw_output: bool = false,

    /// When true, behave like a standard completions endpoint: skip
    /// buildEffectiveContext (no enable_thinking injection), skip
    /// commitGenerationResult (no reasoning separation), and set
    /// thinking_budget = 0.  max_tokens is the sole generation cap.
    completions_mode: bool = false,

    /// Output flag: set to true by generateFromPrompt when the rendered prompt
    /// ends with the reasoning tag (e.g. `<think>\n`), indicating that the
    /// model's output starts inside a reasoning block.  Used by the streaming
    /// iterator to set its filter state before the first token callback fires.
    /// Callers that need this information should point it at a bool they own.
    starts_in_reasoning_out: ?*bool = null,

    pub const PrefillProgressFn = *const fn (usize, usize, ?*anyopaque) callconv(.c) void;
};

/// Build effective template context by merging explicit extra_context_json with
/// reasoning-effort-derived variables. Maps reasoning_effort to enable_thinking:
///   - effort "none" → enable_thinking: false
///   - any other effort (or no effort) → enable_thinking: true
/// Caller owns returned memory (if non-null).
fn buildEffectiveContext(allocator: std.mem.Allocator, opts: GenerateOptions) !?[]const u8 {
    // Determine enable_thinking from max_reasoning_tokens and reasoning_effort.
    // max_reasoning_tokens=0 explicitly disables thinking (overrides effort).
    // When max_reasoning_tokens is null, fall back to reasoning_effort
    // (default = true, matching Qwen3.5 documented default behavior).
    const enable_thinking: bool = if (opts.max_reasoning_tokens) |mrt|
        mrt > 0
    else if (opts.reasoning_effort) |effort|
        !std.mem.eql(u8, effort, "none")
    else
        true;

    const val: []const u8 = if (enable_thinking) "true" else "false";

    // Normalize and build tools fragment if present.
    // Flat-format tools ({"type":"function","name":...}) are converted to nested
    // OpenAI format ({"type":"function","function":{"name":...}}) so the template
    // renders the structure the model was trained on.
    const normalized_tools: ?[]const u8 = if (opts.tools_json) |tj|
        try tool_schema_mod.normalizeToolsJson(allocator, tj)
    else
        null;
    defer if (normalized_tools) |nt| allocator.free(nt);
    const tools_fragment: ?[]const u8 = if (normalized_tools) |nt|
        try std.fmt.allocPrint(allocator, ", \"tools\": {s}", .{nt})
    else
        null;
    defer if (tools_fragment) |tf| allocator.free(tf);

    if (opts.extra_context_json) |existing| {
        // Merge: inject enable_thinking (and tools) into existing context JSON.
        // Replace trailing '}' with the additional fields.
        const trimmed = std.mem.trimRight(u8, existing, " \t\n\r");
        if (trimmed.len > 0 and trimmed[trimmed.len - 1] == '}') {
            const inner = std.mem.trimRight(u8, trimmed[0 .. trimmed.len - 1], " \t\n\r");
            const separator: []const u8 = if (inner.len > 1) ", " else " ";
            const result = try std.fmt.allocPrint(allocator, "{s}{s}\"enable_thinking\": {s}{s} }}", .{
                inner, separator, val, tools_fragment orelse "",
            });
            return @as(?[]const u8, result);
        }
        // Can't merge — return as-is (don't override).
        return null;
    }

    // No existing context — build from scratch.
    const result = try std.fmt.allocPrint(allocator, "{{\"enable_thinking\": {s}{s}}}", .{
        val, tools_fragment orelse "",
    });
    return @as(?[]const u8, result);
}

/// Map reasoning effort level to a max thinking token budget.
/// Returns 0 when thinking is disabled ("none").
fn maxThinkingTokensForEffort(effort: ?[]const u8) usize {
    const e = effort orelse return 4096; // default = medium
    if (std.mem.eql(u8, e, "none")) return 0;
    if (std.mem.eql(u8, e, "low")) return 512;
    if (std.mem.eql(u8, e, "medium")) return 4096;
    if (std.mem.eql(u8, e, "high")) return 16384;
    if (std.mem.eql(u8, e, "xhigh")) return 32768;
    return 4096; // unknown → medium
}

/// Callback function type for streaming token output.
pub const TokenCallback = inference_types.TokenCallback;

/// Local inference engine for LLM generation.
///
/// Loads a model and provides generation capabilities. One engine can
/// serve many Chats efficiently. This is the local destination for the
/// router - other destinations (OpenAI, Anthropic, etc.) are separate modules.
pub const LocalEngine = struct {
    allocator: std.mem.Allocator,

    /// Loaded model weights and config.
    loaded: *models.LoadedModel,

    /// Tokenizer for encoding/decoding.
    tok: tokenizer_mod.Tokenizer,

    /// Sampler for token selection.
    samp: sampler.Sampler,

    /// Compute backend.
    backend: Backend,

    /// Generation config from model (EOS tokens, etc).
    gen_config: gen_config_mod.GenerationConfig,

    /// Preprocessor config for vision (pixel limits from preprocessor_config.json).
    preproc_config: preproc_mod.PreprocessorConfig,

    /// Path to the model directory.
    model_path: []const u8,
    /// Cached chat template metadata loaded once at engine init.
    cached_chat_template: ?CachedChatTemplate,
    /// Plan-derived descriptor contract for scheduler state allocation.
    scheduler_state_descriptors: []runtime_contract.StateDescriptor,

    const CachedChatTemplate = struct {
        template_source: []const u8,
        bos_token: []const u8,
        eos_token: []const u8,
        compiled_template: ?template_mod.CompiledTemplate = null,

        fn deinit(self: *CachedChatTemplate, allocator: std.mem.Allocator) void {
            if (self.compiled_template) |*compiled| compiled.deinit();
            allocator.free(self.template_source);
            allocator.free(self.bos_token);
            allocator.free(self.eos_token);
            self.* = undefined;
        }
    };

    fn appendProgramStateDescriptors(
        storage: []runtime_contract.StateDescriptor,
        count: *u8,
        entry: models.registry.Entry,
        program: []const models.layer_ops.LayerOp,
    ) !void {
        for (program) |op| {
            const state_id = switch (op) {
                .kernel => |kernel_op| kernel_op.state_block_id orelse continue,
                else => continue,
            };
            try runtime_contract.appendUniqueStateDescriptor(
                storage,
                count,
                try models.registry.stateDescriptorForId(entry, state_id),
            );
        }
    }

    fn collectSchedulerStateDescriptors(
        allocator: std.mem.Allocator,
        loaded_model: *const models.LoadedModel,
    ) ![]runtime_contract.StateDescriptor {
        const arch_id = loaded_model.runtime.architecture_id orelse return error.UnsupportedModel;
        const entry = models.registry.detectByArchitectureId(arch_id) orelse return error.UnsupportedModel;
        var storage: [runtime_contract.max_state_descriptors]runtime_contract.StateDescriptor = undefined;
        var count: u8 = 0;

        for (loaded_model.blocks) |layer| {
            const program = models.registry.blockProgramFor(entry, layer.block_type) orelse continue;
            try appendProgramStateDescriptors(storage[0..], &count, entry, program);
        }
        if (models.registry.visionProgramByArchitectureId(entry.id)) |program| {
            try appendProgramStateDescriptors(storage[0..], &count, entry, program);
        }
        if (count == 0) return &.{};

        const descriptors = try allocator.alloc(runtime_contract.StateDescriptor, count);
        @memcpy(descriptors, storage[0..count]);
        return descriptors;
    }

    fn loadCachedChatTemplate(allocator: std.mem.Allocator, model_dir: []const u8) !CachedChatTemplate {
        const config_path = try std.fs.path.join(allocator, &.{ model_dir, "tokenizer_config.json" });
        defer allocator.free(config_path);

        const config_bytes = try std.fs.cwd().readFileAlloc(allocator, config_path, 4 * 1024 * 1024);
        defer allocator.free(config_bytes);

        const parsed_config = io_json.parseValue(allocator, config_bytes, .{ .max_size_bytes = 4 * 1024 * 1024 }) catch {
            return error.InvalidJson;
        };
        defer parsed_config.deinit();

        const obj = switch (parsed_config.value) {
            .object => |object| object,
            else => return error.InvalidJson,
        };

        const template_source = if (obj.get("chat_template")) |template_value| switch (template_value) {
            .string => |s| try allocator.dupe(u8, s),
            else => try gen_config_mod.getChatTemplateSource(allocator, model_dir),
        } else try gen_config_mod.getChatTemplateSource(allocator, model_dir);
        errdefer allocator.free(template_source);

        const bos_token = if (obj.get("bos_token")) |v| switch (v) {
            .string => |s| try allocator.dupe(u8, s),
            else => try allocator.dupe(u8, ""),
        } else try allocator.dupe(u8, "");
        errdefer allocator.free(bos_token);

        const eos_token = if (obj.get("eos_token")) |v| switch (v) {
            .string => |s| try allocator.dupe(u8, s),
            else => try allocator.dupe(u8, ""),
        } else try allocator.dupe(u8, "");
        errdefer allocator.free(eos_token);

        // Best-effort compile of the template for faster per-request rendering.
        // If compilation fails (unsupported syntax edge-case), fall back to
        // renderWithContext() using template_source at request time.
        var compiled_template: ?template_mod.CompiledTemplate = template_mod.CompiledTemplate.init(
            allocator,
            template_source,
        ) catch null;
        errdefer if (compiled_template) |*compiled| compiled.deinit();

        return .{
            .template_source = template_source,
            .bos_token = bos_token,
            .eos_token = eos_token,
            .compiled_template = compiled_template,
        };
    }

    pub fn renderPromptWithCachedTemplate(
        self: *LocalEngine,
        messages_json: []const u8,
        add_generation_prompt: bool,
        template_override: ?[]const u8,
        extra_context_json: ?[]const u8,
    ) ![]const u8 {
        if (template_override) |override| {
            if (self.cached_chat_template) |*cached| {
                return chat_template.renderWithContext(
                    self.allocator,
                    override,
                    messages_json,
                    cached.bos_token,
                    cached.eos_token,
                    add_generation_prompt,
                    extra_context_json,
                );
            }
            return gen_config_mod.applyChatTemplateWithOverrides(
                self.allocator,
                self.model_path,
                messages_json,
                add_generation_prompt,
                template_override,
                extra_context_json,
            );
        }

        if (self.cached_chat_template) |*cached| {
            if (cached.compiled_template) |*compiled| {
                return chat_template.renderCompiledWithContext(
                    self.allocator,
                    compiled,
                    messages_json,
                    cached.bos_token,
                    cached.eos_token,
                    add_generation_prompt,
                    extra_context_json,
                );
            }
            return chat_template.renderWithContext(
                self.allocator,
                cached.template_source,
                messages_json,
                cached.bos_token,
                cached.eos_token,
                add_generation_prompt,
                extra_context_json,
            );
        }

        return gen_config_mod.applyChatTemplateWithOverrides(
            self.allocator,
            self.model_path,
            messages_json,
            add_generation_prompt,
            null,
            extra_context_json,
        );
    }

    /// Initialize engine from a model path or model ID.
    ///
    /// The path can be:
    /// - Direct path to model directory (with config.json, model.safetensors, tokenizer.json)
    /// - Cache format path (models--org--name/snapshots/)
    /// - HuggingFace model ID (e.g., "org/model-name") - will be looked up in cache or downloaded
    ///
    /// Ownership: The engine does not retain the input slice; it resolves and
    /// owns its internal model path copy for the engine lifetime.
    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !LocalEngine {
        return initWithSeedAndResolutionConfig(allocator, model_path, 42, .{}, .{}, progress_mod.ProgressContext.NONE);
    }

    /// Initialize engine with a specific random seed.
    pub fn initWithSeed(allocator: std.mem.Allocator, model_path: []const u8, seed: u64) !LocalEngine {
        return initWithSeedAndResolutionConfig(allocator, model_path, seed, .{}, .{}, progress_mod.ProgressContext.NONE);
    }

    /// Initialize engine with resolution configuration.
    pub fn initWithResolutionConfig(
        allocator: std.mem.Allocator,
        model_path: []const u8,
        config: ResolutionConfig,
    ) !LocalEngine {
        return initWithSeedAndResolutionConfig(allocator, model_path, 42, config, .{}, progress_mod.ProgressContext.NONE);
    }

    /// Initialize engine with a specific random seed and resolution configuration.
    pub fn initWithSeedAndResolutionConfig(
        allocator: std.mem.Allocator,
        model_path: []const u8,
        seed: u64,
        config: ResolutionConfig,
        backend_init_options: BackendInitOptions,
        progress: progress_mod.ProgressContext,
    ) !LocalEngine {
        var timing_start_ns: i128 = std.time.nanoTimestamp();

        // Resolve model path using centralized repository logic.
        // Handles: local paths, model IDs (HF Hub), cache paths
        const resolved_model_path = try io.repository.resolveModelPath(allocator, model_path, config);
        errdefer allocator.free(resolved_model_path);

        // Find model files
        var model_bundle = try io.repository.resolve(allocator, resolved_model_path, .{});
        defer model_bundle.deinit();

        // Validate files exist
        std.fs.cwd().access(model_bundle.config_path(), .{}) catch |err| switch (err) {
            error.FileNotFound => return error.FileNotFound,
            else => return err,
        };
        const wp = model_bundle.weights_path() orelse return error.FileNotFound;
        std.fs.cwd().access(wp, .{}) catch |err| switch (err) {
            error.FileNotFound => return error.FileNotFound,
            else => return err,
        };

        // Load generation config
        var generation_config = try gen_config_mod.loadGenerationConfig(allocator, resolved_model_path);
        errdefer generation_config.deinit(allocator);

        // Best-effort cache for chat template rendering metadata.
        // When unavailable, requests fall back to the existing on-demand path.
        var cached_chat_template = loadCachedChatTemplate(allocator, resolved_model_path) catch |err| blk: {
            log.debug("load", "chat template cache unavailable", .{
                .reason = @errorName(err),
            }, @src());
            break :blk null;
        };
        errdefer if (cached_chat_template) |*cached| cached.deinit(allocator);

        // Load preprocessor config (pixel limits for vision smart resize)
        const preproc_config = preproc_mod.loadPreprocessorConfig(allocator, resolved_model_path);

        const model_load_options = backend_root.defaultModelLoadOptions(backend_init_options);

        // Start model loading in background thread
        const ModelLoaderThread = struct {
            alloc: std.mem.Allocator,
            config_path: []const u8,
            weights_path: []const u8,
            load_options: models.LoadOptions,
            prog: progress_mod.ProgressContext,
            loaded_model: ?models.LoadedModel = null,
            err: ?anyerror = null,

            fn loadModel(self: *@This()) void {
                self.loaded_model = models.loadModel(self.alloc, self.config_path, self.weights_path, self.load_options, self.prog) catch |e| {
                    self.err = e;
                    return;
                };
            }
        };

        var loader_thread_state = ModelLoaderThread{
            .alloc = allocator,
            .config_path = model_bundle.config_path(),
            .weights_path = wp,
            .load_options = model_load_options,
            .prog = progress,
        };

        // Try threaded loading
        const loader_thread_handle = std.Thread.spawn(.{}, ModelLoaderThread.loadModel, .{&loader_thread_state}) catch null;

        // Load tokenizer while model loads in background
        var tokenizer_instance = blk: {
            if (model_bundle.tokenizer_json()) |json| {
                break :blk tokenizer_mod.Tokenizer.initFromJson(allocator, json) catch |err| {
                    log.warn("inference", "Tokenizer initialization failed", .{
                        .reason = @errorName(err),
                        .source = "json",
                        .json_bytes = json.len,
                        .model_path = resolved_model_path,
                    });
                    return err;
                };
            }
            const tokenizer_path = model_bundle.tokenizer_path();
            break :blk tokenizer_mod.Tokenizer.initFromPath(allocator, tokenizer_path) catch |err| {
                log.warn("inference", "Tokenizer initialization failed", .{
                    .reason = @errorName(err),
                    .source = "path",
                    .tokenizer_path = tokenizer_path,
                    .model_path = resolved_model_path,
                });
                return err;
            };
        };
        errdefer tokenizer_instance.deinit();

        {
            const now = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(now - timing_start_ns)) / 1_000_000.0;
            log.debug("inference", "Tokenizer loaded", .{ .duration_ms = duration_ms }, @src());
            timing_start_ns = now;
        }

        // Wait for model or load synchronously
        if (loader_thread_handle) |thread| {
            thread.join();
        } else {
            // Thread spawn failed - load synchronously
            loader_thread_state.loaded_model = try models.loadModel(allocator, model_bundle.config_path(), wp, model_load_options, progress);
        }

        if (loader_thread_state.err) |e| return e;

        const loaded_model = try allocator.create(models.LoadedModel);
        errdefer allocator.destroy(loaded_model);
        loaded_model.* = loader_thread_state.loaded_model.?;
        errdefer loaded_model.deinit();

        {
            const now = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(now - timing_start_ns)) / 1_000_000.0;
            log.debug("inference", "Model loaded", .{ .duration_ms = duration_ms }, @src());
            timing_start_ns = now;
        }

        // Create sampler
        var sampler_instance = try sampler.Sampler.init(allocator, seed, @intCast(loaded_model.config.vocab_size));
        errdefer sampler_instance.deinit();

        {
            const now = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(now - timing_start_ns)) / 1_000_000.0;
            log.debug("inference", "Sampler initialized", .{ .duration_ms = duration_ms }, @src());
            timing_start_ns = now;
        }

        // Create backend (progress bar emitted from buildBlocks inside)
        progress.addLine(
            1,
            "Loading",
            2,
            "Initializing backend (large models may take longer)...",
            null,
        );
        var backend_options = backend_init_options;
        backend_options.metal = .{
            .model_path = resolved_model_path,
            .model_id = model_path,
        };
        var compute_backend = try Backend.init(allocator, loaded_model, backend_options, progress);
        errdefer compute_backend.deinit();
        progress.updateLine(1, 1, "Backend initialized, preparing runtime...");
        const scheduler_state_descriptors = try collectSchedulerStateDescriptors(allocator, loaded_model);
        errdefer if (scheduler_state_descriptors.len > 0) allocator.free(scheduler_state_descriptors);

        {
            const now = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(now - timing_start_ns)) / 1_000_000.0;
            log.debug("inference", "Backend initialized", .{ .duration_ms = duration_ms }, @src());
            timing_start_ns = now;
        }

        // Warmup: CPU backend performs a real forward pass during warmup.
        // Metal/CUDA warmup is currently a no-op and does not require state
        // descriptor binding during engine construction.
        const warmup_needs_state_bindings = switch (compute_backend) {
            .cpu => true,
            .metal => false,
            .cuda => false,
        };
        var warmup_bindings = TemporaryStateBindings{};
        defer warmup_bindings.deinit(allocator);
        var warmup_slot_bound = false;
        defer if (warmup_slot_bound) compute_backend.unbindSlotStateBlocks(0);
        if (warmup_needs_state_bindings and scheduler_state_descriptors.len > 0) {
            warmup_bindings = try allocateTemporaryStateBindingsForDescriptors(
                allocator,
                scheduler_state_descriptors,
            );
            if (warmup_bindings.handles.len > 0) {
                compute_backend.bindSlotStateBlocks(0, warmup_bindings.handles) catch |err| {
                    log.warn("inference", "Warmup state bind failed", .{
                        .reason = @errorName(err),
                        .state_blocks = warmup_bindings.handles.len,
                    });
                    return err;
                };
                warmup_slot_bound = true;
            }
        }
        compute_backend.warmup() catch |err| {
            log.warn("inference", "Backend warmup failed", .{
                .reason = @errorName(err),
            });
            return err;
        };
        progress.completeLine(1);

        {
            const now = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(now - timing_start_ns)) / 1_000_000.0;
            log.debug("inference", "Warmup complete", .{ .duration_ms = duration_ms }, @src());
        }

        return LocalEngine{
            .allocator = allocator,
            .loaded = loaded_model,
            .tok = tokenizer_instance,
            .samp = sampler_instance,
            .backend = compute_backend,
            .gen_config = generation_config,
            .preproc_config = preproc_config,
            .model_path = resolved_model_path,
            .cached_chat_template = cached_chat_template,
            .scheduler_state_descriptors = scheduler_state_descriptors,
        };
    }

    /// Free all resources.
    pub fn deinit(self: *LocalEngine) void {
        self.backend.deinit();
        self.samp.deinit();
        self.loaded.deinit();
        self.allocator.destroy(self.loaded);
        self.tok.deinit();
        self.gen_config.deinit(self.allocator);
        self.allocator.free(self.model_path);
        if (self.cached_chat_template) |*cached| cached.deinit(self.allocator);
        if (self.scheduler_state_descriptors.len > 0) self.allocator.free(self.scheduler_state_descriptors);
        self.* = undefined;
    }

    /// Explicit backend/device barrier for xray-style capture finalization.
    pub fn synchronize(self: *LocalEngine) !void {
        try self.backend.synchronize();
    }

    /// Get EOS token IDs.
    pub fn getEosTokens(self: *const LocalEngine) []const u32 {
        return self.gen_config.eos_token_ids;
    }

    /// Build generation parameters JSON for storage.
    ///
    /// Creates a JSON object with the model and sampling parameters used for generation.
    /// Caller owns returned memory.
    fn buildGenerationJson(
        self: *LocalEngine,
        chat: *Chat,
        opts: GenerateOptions,
        max_tokens: usize,
    ) ![]u8 {
        const temperature = opts.temperature orelse chat.temperature;
        const top_k = opts.top_k orelse chat.top_k;
        const top_p = opts.top_p orelse chat.top_p;
        const min_p = opts.min_p orelse chat.min_p;
        const repetition_penalty = opts.repetition_penalty orelse chat.repetition_penalty;

        var json_buf: std.ArrayListUnmanaged(u8) = .{};
        errdefer json_buf.deinit(self.allocator);
        const w = json_buf.writer(self.allocator);

        try w.writeAll("{\"model\":\"");
        try writeJsonStringContent(w, self.model_path);
        try w.print("\",\"temperature\":{d:.6},\"top_p\":{d:.6},\"top_k\":{d},\"min_p\":{d:.6},\"repetition_penalty\":{d:.6},\"max_tokens\":{d}", .{
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            max_tokens,
        });
        if (opts.seed != 0) {
            try w.print(",\"seed\":{d}", .{opts.seed});
        }
        try w.writeByte('}');

        return json_buf.toOwnedSlice(self.allocator);
    }

    /// Write a string value with JSON escaping (without surrounding quotes).
    fn writeJsonStringContent(writer: anytype, s: []const u8) !void {
        for (s) |c| {
            switch (c) {
                '"' => try writer.writeAll("\\\""),
                '\\' => try writer.writeAll("\\\\"),
                '\n' => try writer.writeAll("\\n"),
                '\r' => try writer.writeAll("\\r"),
                '\t' => try writer.writeAll("\\t"),
                else => {
                    if (c < 0x20) {
                        try writer.print("\\u{x:0>4}", .{c});
                    } else {
                        try writer.writeByte(c);
                    }
                },
            }
        }
    }

    /// Get vocabulary size.
    pub fn vocabSize(self: *const LocalEngine) usize {
        return self.backend.vocabSize();
    }

    /// Generate a response for a chat.
    ///
    /// Uses the chat's settings by default, but parameters can be overridden.
    /// The assistant's response is automatically added to the chat history.
    ///
    /// Template customization:
    ///   - opts.template_override: Use a custom template string instead of model's template
    ///   - opts.extra_context_json: Inject additional variables (tools, dates, etc.) into template
    pub fn generate(self: *LocalEngine, chat: *Chat, opts: GenerateOptions) !GenerationResult {
        // Format messages to JSON using protocol layer (single source of truth)
        const messages_json = try protocol.completions.serialize(
            self.allocator,
            chat.conv,
            .{ .image_content_type = .image },
        );
        defer self.allocator.free(messages_json);

        // Build effective template context: merge explicit extra_context_json with
        // reasoning-effort-derived variables (e.g. enable_thinking for Qwen3.5).
        // In completions_mode without tools, disable thinking. With tools,
        // keep thinking enabled — the model needs reasoning to decide tool use.
        var completions_opts = opts;
        if (opts.completions_mode and opts.tools_json == null) {
            completions_opts.max_reasoning_tokens = 0;
        }
        const effective_context = buildEffectiveContext(self.allocator, completions_opts) catch |err| {
            log.err("inference", "Failed to build template context", .{ .err = @errorName(err) }, @src());
            return err;
        };
        defer if (effective_context) |ctx| self.allocator.free(ctx);

        // Apply chat template with optional overrides
        log.debug("inference", "Applying chat template", .{ .messages_len = messages_json.len }, @src());
        const prompt = self.renderPromptWithCachedTemplate(
            messages_json,
            true, // add_generation_prompt
            opts.template_override,
            effective_context,
        ) catch |err| {
            // If no chat template or template render fails, fall back to just using the last user message
            if (err == error.MissingChatTemplate or err == error.FileNotFound) {
                log.warn("inference", "No chat template found, using raw input (equivalent to --no-chat)", .{});
            } else if (err == error.EvalError) {
                log.warn("inference", "Chat template failed to render, using raw input (equivalent to --no-chat)", .{});
            } else {
                return err;
            }
            // Simple fallback: use last user message as prompt
            if (chat.len() > 0) {
                const last_item = chat.get(chat.len() - 1);
                if (last_item) |item| {
                    if (item.asMessage()) |msg| {
                        if (msg.role == .user) {
                            return self.generateFromPrompt(chat, msg.getFirstText(), opts);
                        }
                    }
                }
            }
            return err;
        };
        defer self.allocator.free(prompt);

        log.debug("inference", "Chat template rendered", .{ .prompt_len = prompt.len }, @src());

        return self.generateFromPrompt(chat, prompt, opts);
    }

    /// Internal: generate from a formatted prompt string.
    /// Uses Scheduler for continuous batching when supported by backend.
    fn generateFromPrompt(
        self: *LocalEngine,
        chat: *Chat,
        prompt: []const u8,
        opts: GenerateOptions,
    ) !GenerationResult {
        // Check whether the rendered prompt ends with the reasoning tag
        // (e.g. `<think>\n`).  Publish via output flag so streaming callers
        // (iterator) can set their filter state before the first token fires.
        if (opts.starts_in_reasoning_out) |out| {
            out.* = chat_template.promptEndsWithReasoningTag(prompt, opts.reasoning_tag);
        }

        // Use chat settings with optional overrides.
        // When max_completion_tokens is set without explicit max_tokens,
        // auto-compute: max_tokens = thinking_budget + max_completion_tokens.
        // Token budget hierarchy:
        //   max_tokens          — hard cap on total generated tokens (never exceeded)
        //   max_reasoning_tokens — budget for thinking (within max_tokens)
        //   max_completion_tokens — budget for answer (within max_tokens)
        //
        // When max_tokens is set explicitly, it IS the hard cap. The reasoning
        // and completion budgets operate within it, never additive.
        // When max_tokens is NOT set, auto-compute from budgets.
        const base_max_tokens = if (opts.max_completion_tokens != null and opts.max_tokens == null) blk: {
            const raw_budget = if (opts.max_reasoning_tokens) |mrt|
                mrt
            else
                maxThinkingTokensForEffort(opts.reasoning_effort);
            break :blk raw_budget + opts.max_completion_tokens.?;
        } else opts.max_tokens orelse chat.max_tokens;
        const grammar_slack: usize = 64;

        // If tools_json is provided (and tool_choice != "none"), mark tool generation.
        // No constrained sampler — the model generates freely in its native
        // tool call format (e.g. Qwen3.5 <tool_call> XML). Parsed after generation.
        const use_tools = opts.tools_json != null and
            (opts.tool_choice == null or !std.mem.eql(u8, opts.tool_choice.?, "none"));

        const effective_grammar = chat.grammar_sampler;

        // max_tokens is the hard generation limit — never exceeded.
        const max_tokens = if (effective_grammar != null and base_max_tokens > 0)
            base_max_tokens + grammar_slack
        else
            base_max_tokens;
        const temperature = opts.temperature orelse chat.temperature;
        const top_k = opts.top_k orelse chat.top_k;
        const top_p = opts.top_p orelse chat.top_p;
        const min_p = opts.min_p orelse chat.min_p;
        const repetition_penalty = opts.repetition_penalty orelse chat.repetition_penalty;
        const presence_penalty = opts.presence_penalty orelse 0.0;
        const frequency_penalty = opts.frequency_penalty orelse 0.0;
        // Build sampling config
        var sampling_config = sampler.SamplingConfig{ .strategy = .greedy, .logit_bias = opts.logit_bias, .seed = opts.seed, .presence_penalty = presence_penalty, .frequency_penalty = frequency_penalty };
        if (temperature > 0 and (self.gen_config.do_sample or opts.temperature != null)) {
            sampling_config = .{
                .strategy = .top_k,
                .temperature = temperature,
                .top_k = top_k,
                .top_p = top_p,
                .min_p = min_p,
                .repetition_penalty = repetition_penalty,
                .presence_penalty = presence_penalty,
                .frequency_penalty = frequency_penalty,
                .logit_bias = opts.logit_bias,
                .seed = opts.seed,
            };
        }

        // Get BOS token ID
        const bos_token_id: ?u32 = if (self.gen_config.bos_token_id) |id|
            (if (self.gen_config.add_bos_token) id else null)
        else if (self.loaded.config.bos_token_id) |id|
            (if (self.gen_config.add_bos_token and id >= 0) @intCast(id) else null)
        else
            null;

        // Install prefill progress callback and stop flag (cleared after generation)
        self.backend.setPrefillProgress(opts.prefill_progress_fn, opts.prefill_progress_data);
        defer self.backend.setPrefillProgress(null, null);
        self.backend.setStopFlag(opts.stop_flag);
        defer self.backend.setStopFlag(null);

        log.debug("router", "Generation params", .{
            .max_tokens = max_tokens,
            .use_tools = @as(u8, @intFromBool(use_tools)),
        }, @src());
        return self.generateWithScheduler(
            chat,
            prompt,
            max_tokens,
            sampling_config,
            bos_token_id,
            opts,
            effective_grammar,
            use_tools,
        );
    }

    /// Generate using Scheduler (continuous batching path).
    fn generateWithScheduler(
        self: *LocalEngine,
        chat: *Chat,
        prompt: []const u8,
        max_tokens: usize,
        sampling_config: sampler.SamplingConfig,
        bos_token_id: ?u32,
        opts: GenerateOptions,
        grammar_sampler: ?*ConstrainedSampler,
        is_tool_generation: bool,
    ) !GenerationResult {
        // Every logical generation run must start from a clean backend
        // execution-thread state. Non-stream local generation executes on the
        // caller thread, while streaming uses a dedicated worker thread; both
        // must observe the same invariant instead of inheriting MLX thread-
        // local caches or pooled temporaries from earlier work on that thread.
        self.backend.teardownExecutionThreadState();
        // After the run, clear transient execution-thread resources once
        // scheduler teardown has finished.
        defer self.backend.cleanupExecutionThreadState();

        const SchedulerType = inference.scheduler.GenericScheduler(Backend);

        log.debug("router", "Collecting vision input", .{}, @src());
        var vision_prompt = try self.collectVisionPromptInput(chat);
        defer if (vision_prompt) |*vp| vp.deinit(self.allocator);

        // Tokenize prompt
        log.debug("router", "Tokenizing prompt", .{ .prompt_bytes = prompt.len }, @src());
        const encoded_tokens = try self.tok.encode(prompt);
        defer self.allocator.free(encoded_tokens);

        const vision_boundaries = if (vision_prompt != null) self.resolveVisionBoundaryTokens() else VisionBoundaryTokens{};
        log.debug("router", "Expanding image pad tokens", .{
            .encoded_len = encoded_tokens.len,
            .has_vision = @as(u8, @intFromBool(vision_prompt != null)),
        }, @src());
        const prompt_tokens_no_bos = if (vision_prompt) |*vp|
            try expandImagePadTokens(
                self.allocator,
                encoded_tokens,
                vp.prefill.image_token_id,
                vp.token_counts,
                vision_boundaries,
            )
        else
            try self.allocator.dupe(u32, encoded_tokens);
        defer self.allocator.free(prompt_tokens_no_bos);

        // Prepend BOS token if configured
        var prepend_bos = bos_token_id != null;
        if (prepend_bos and prompt_tokens_no_bos.len > 0 and prompt_tokens_no_bos[0] == bos_token_id.?) {
            prepend_bos = false;
        }

        const prompt_tokens = if (prepend_bos) blk: {
            const tokens = try self.allocator.alloc(u32, prompt_tokens_no_bos.len + 1);
            tokens[0] = bos_token_id.?;
            @memcpy(tokens[1..], prompt_tokens_no_bos);
            break :blk tokens;
        } else try self.allocator.dupe(u32, prompt_tokens_no_bos);
        defer self.allocator.free(prompt_tokens);

        const prompt_len = prompt_tokens.len;

        log.debug("inference", "Tokenization complete", .{ .prompt_tokens = prompt_len }, @src());

        // Create scheduler for this request
        log.debug("router", "Creating scheduler", .{}, @src());
        var scheduler = try SchedulerType.init(self.allocator, &self.backend, .{
            .default_eos_token_ids = self.gen_config.eos_token_ids,
            .default_sampling = sampling_config,
            .tokenizer = &self.tok,
            .state_descriptors = self.scheduler_state_descriptors,
        });
        defer scheduler.deinit();

        // Wrap token callback if provided (Scheduler uses different signature)
        const CallbackWrapper = struct {
            original_callback: ?inference_types.TokenCallback,
            original_data: ?*anyopaque,

            fn wrap(request_id: u64, token: u32, is_final: bool, in_thinking: bool, user_data: ?*anyopaque) void {
                _ = request_id;
                _ = is_final;
                const wrapper: *@This() = @ptrCast(@alignCast(user_data));
                if (wrapper.original_callback) |cb| {
                    cb(token, in_thinking, wrapper.original_data);
                }
            }
        };

        var callback_wrapper = CallbackWrapper{
            .original_callback = opts.token_callback,
            .original_data = opts.callback_data,
        };

        const vision_input_ptr: ?*const anyopaque = if (vision_prompt) |*vp|
            @ptrCast(&vp.prefill)
        else
            null;
        log.debug("inference", "Scheduler request assembled", .{
            .has_vision_input = @as(u8, @intFromBool(vision_input_ptr != null)),
            .image_count = if (vision_prompt) |*vp| vp.prefill.images.len else 0,
        }, @src());

        // Thinking budget: carved from max_tokens, never exceeds it.
        // In completions_mode without tools, thinking is disabled. With tools,
        // the model needs thinking to reason about tool use.
        // When max_completion_tokens is set, use it as the answer reserve.
        // Otherwise fall back to 25% heuristic (floor 256).
        const thinking_budget = if (opts.completions_mode and opts.tools_json == null) @as(usize, 0) else blk: {
            const raw_thinking_budget = if (opts.max_reasoning_tokens) |mrt|
                mrt
            else
                maxThinkingTokensForEffort(opts.reasoning_effort);
            const answer_reserve = if (opts.max_completion_tokens) |mct|
                mct
            else
                @max(@as(usize, 256), max_tokens / 4);
            break :blk if (raw_thinking_budget > 0)
                @min(raw_thinking_budget, max_tokens -| answer_reserve)
            else
                @as(usize, 0);
        };

        // Tokenize thinking end sequence if thinking budget is active.
        const thinking_end_tokens = if (thinking_budget > 0)
            self.tok.encode("</think>\n\n") catch null
        else
            null;
        defer if (thinking_end_tokens) |t| self.allocator.free(t);

        // Generate synchronously
        log.debug("router", "generateSync starting", .{
            .prompt_tokens = prompt_tokens.len,
            .max_tokens = max_tokens,
            .thinking_budget = thinking_budget,
        }, @src());
        var result = scheduler.generateSync(prompt_tokens, max_tokens, .{
            .eos_token_ids = self.gen_config.eos_token_ids,
            .stop_sequences = opts.stop_sequences,
            .callback = if (opts.token_callback != null) CallbackWrapper.wrap else null,
            .callback_data = if (opts.token_callback != null) @ptrCast(&callback_wrapper) else null,
            .sampling = sampling_config,
            .grammar_sampler = grammar_sampler,
            .stop_flag = opts.stop_flag,
            .vision_input = vision_input_ptr,
            .max_thinking_tokens = thinking_budget,
            .thinking_end_tokens = if (thinking_end_tokens) |t| t else &.{},
            .max_completion_tokens = opts.max_completion_tokens orelse 0,
        }) catch |err| {
            const err_ctx = error_context.consumeContext();
            log.warn("inference", "Scheduler generation failed", .{
                .err = @errorName(err),
                .has_vision_input = @as(u8, @intFromBool(vision_input_ptr != null)),
                .context = if (err_ctx) |ctx| ctx else "",
            });
            return err;
        };
        defer result.deinit(self.allocator);

        log.debug("router", "Scheduler token sample", .{
            .count = result.tokens.len,
            .t0 = if (result.tokens.len > 0) result.tokens[0] else @as(u32, std.math.maxInt(u32)),
            .t1 = if (result.tokens.len > 1) result.tokens[1] else @as(u32, std.math.maxInt(u32)),
            .t2 = if (result.tokens.len > 2) result.tokens[2] else @as(u32, std.math.maxInt(u32)),
            .t3 = if (result.tokens.len > 3) result.tokens[3] else @as(u32, std.math.maxInt(u32)),
        }, @src());

        // Strip trailing EOS token if present
        var tokens_to_decode = result.tokens;
        if (tokens_to_decode.len > 0) {
            for (self.gen_config.eos_token_ids) |eos_id| {
                if (tokens_to_decode[tokens_to_decode.len - 1] == eos_id) {
                    tokens_to_decode = tokens_to_decode[0 .. tokens_to_decode.len - 1];
                    break;
                }
            }
        }

        const generated_text = try self.tok.decode(tokens_to_decode);

        // Create owned copy of generated tokens
        const owned_tokens = try self.allocator.dupe(u32, result.tokens);
        errdefer self.allocator.free(owned_tokens);

        // Try to parse tool calls from generated text (supports XML and JSON).
        // If parsing fails, fall through to the normal text path.
        if (is_tool_generation and generated_text.len > 0) tool_call_blk: {
            const parsed_calls = tool_schema_mod.parseToolCallsFromText(self.allocator, generated_text) catch
                break :tool_call_blk;
            defer {
                for (parsed_calls) |*pc| {
                    var call = pc.*;
                    call.deinit(self.allocator);
                }
                self.allocator.free(parsed_calls);
            }
            if (parsed_calls.len == 0) break :tool_call_blk;

            // Generate call IDs and build commit inputs.
            const tc_inputs = try self.allocator.alloc(commit_mod.ToolCallInput, parsed_calls.len);
            defer self.allocator.free(tc_inputs);
            var call_ids = try self.allocator.alloc([]u8, parsed_calls.len);
            defer {
                for (call_ids[0..parsed_calls.len]) |cid| self.allocator.free(cid);
                self.allocator.free(call_ids);
            }

            for (parsed_calls, 0..) |pc, i| {
                call_ids[i] = try tool_schema_mod.generateCallId(self.allocator);
                tc_inputs[i] = .{
                    .id = call_ids[i],
                    .name = pc.name,
                    .arguments = pc.arguments,
                };
            }

            const finish_reason_str = finishReasonToString(.tool_calls);
            try commit_mod.commitGenerationResult(self.allocator, chat, .{
                .text = generated_text,
                .tool_calls = tc_inputs,
                .prompt_tokens = prompt_len,
                .completion_tokens = result.tokens.len,
                .prefill_ns = result.prefill_ns,
                .generation_ns = result.decode_ns,
                .finish_reason = finish_reason_str,
            });

            // Build ToolCallRef array for the caller.
            const tool_calls = try self.allocator.alloc(ToolCallRef, parsed_calls.len);
            errdefer self.allocator.free(tool_calls);

            const item_index = chat.conv.len() - 1;
            for (parsed_calls, 0..) |pc, i| {
                tool_calls[i] = ToolCallRef{
                    .item_index = item_index,
                    .call_id = try self.allocator.dupe(u8, call_ids[i]),
                    .name = try self.allocator.dupe(u8, pc.name),
                    .arguments = try self.allocator.dupe(u8, pc.arguments),
                };
            }

            return GenerationResult{
                .text = generated_text,
                .tokens = owned_tokens,
                .prompt_tokens = prompt_len,
                .generated_tokens = result.tokens.len,
                .prefill_ns = result.prefill_ns,
                .decode_ns = result.decode_ns,
                .finish_reason = .tool_calls,
                .tool_calls = tool_calls,
            };
        }

        // Map scheduler finish reason to session finish reason (text path).
        const finish_reason: FinishReason = switch (result.finish_reason) {
            .in_progress => .eos_token, // Shouldn't happen for sync
            .eos_token => .eos_token,
            .length => .length,
            .stop_sequence => .stop_sequence,
            .cancelled => .cancelled,
        };

        // In completions_mode, skip reasoning separation and conversation commit —
        // return the raw decoded text directly with max_tokens as the sole cap.
        const result_text = if (opts.completions_mode) generated_text else blk: {
            const finish_reason_str = finishReasonToString(finish_reason);

            // Text path: commit reasoning + assistant message via shared path
            const generation_json = try self.buildGenerationJson(chat, opts, max_tokens);
            defer self.allocator.free(generation_json);

            // Detect whether the rendered prompt ends with an opening reasoning
            // tag (e.g. `<think>\n`).  When the template injects this as a
            // generation prefix, the model's output starts inside the reasoning
            // block (no opening `<think>` tag), so the parser must begin in
            // reasoning state to correctly separate reasoning / response.
            const starts_in_reasoning = chat_template.promptEndsWithReasoningTag(prompt, opts.reasoning_tag);

            try commit_mod.commitGenerationResult(self.allocator, chat, .{
                .text = generated_text,
                .prompt_tokens = prompt_len,
                .completion_tokens = result.tokens.len,
                .prefill_ns = result.prefill_ns,
                .generation_ns = result.decode_ns,
                .finish_reason = finish_reason_str,
                .reasoning_tag = opts.reasoning_tag,
                .generation_json = generation_json,
                .starts_in_reasoning = starts_in_reasoning,
            });

            // Return clean text (strip reasoning tags for caller)
            var parser = try reasoning_parser_mod.ReasoningParser.initWithState(self.allocator, opts.reasoning_tag, starts_in_reasoning);
            defer parser.deinit();
            try parser.processChunk(generated_text);
            const parsed = try parser.finalize();

            break :blk if (opts.raw_output) generated_text else if (parsed.reasoning != null) inner: {
                const duped = try self.allocator.dupe(u8, parsed.response orelse "");
                self.allocator.free(generated_text);
                break :inner duped;
            } else generated_text;
        };

        log.debug("router", "Generation complete", .{
            .prompt_tokens = prompt_len,
            .generated_tokens = result.tokens.len,
            .prefill_ns = result.prefill_ns,
            .decode_ns = result.decode_ns,
        }, @src());

        return GenerationResult{
            .text = result_text,
            .tokens = owned_tokens,
            .prompt_tokens = prompt_len,
            .generated_tokens = result.tokens.len,
            .prefill_ns = result.prefill_ns,
            .decode_ns = result.decode_ns,
            .finish_reason = finish_reason,
        };
    }

    const VisionPromptInput = struct {
        prefill: vision_types.PrefillVisionInput,
        token_counts: []usize,

        fn deinit(self: *VisionPromptInput, allocator: std.mem.Allocator) void {
            self.prefill.deinit(allocator);
            if (self.token_counts.len > 0) allocator.free(self.token_counts);
            self.* = undefined;
        }
    };

    const VisionBoundaryTokens = struct {
        start_token_id: ?u32 = null,
        end_token_id: ?u32 = null,
    };

    fn collectVisionPromptInput(self: *LocalEngine, chat: *Chat) !?VisionPromptInput {
        const image_count = countInputImageParts(chat);
        if (image_count == 0) return null;
        log.debug("router", "Vision input detected", .{ .images = image_count }, @src());
        if (self.loaded.config.image_token_id <= 0) return error.UnsupportedContentType;

        const preprocess_opts = try self.buildVisionPreprocessOptions();

        var written: usize = 0;
        const images = try self.allocator.alloc(vision_types.PrefillVisionImage, image_count);
        errdefer {
            for (images[0..written]) |*img| img.deinit(self.allocator);
            self.allocator.free(images);
        }
        const token_counts = try self.allocator.alloc(usize, image_count);
        errdefer self.allocator.free(token_counts);

        for (0..chat.conv.len()) |item_index| {
            const item = chat.conv.getItem(item_index) orelse continue;
            const msg = item.asMessage() orelse continue;

            for (0..msg.partCount()) |part_index| {
                const part = msg.getPart(part_index) orelse continue;
                if (part.getContentType() != .input_image) continue;

                log.trace("router", "Loading image", .{
                    .index = written,
                    .data_len = part.getData().len,
                }, @src());
                const image_bytes = try loadImageBytes(self.allocator, part.getData());
                defer self.allocator.free(image_bytes);

                var decoded = try image_mod.decode(self.allocator, image_bytes, .{
                    .prefer_format = .rgb8,
                    .apply_orientation = true,
                });
                defer decoded.deinit(self.allocator);

                log.trace("router", "Preprocessing image", .{
                    .index = written,
                    .bytes = image_bytes.len,
                }, @src());
                var prep = try image_mod.preprocessImage(self.allocator, decoded, preprocess_opts);
                errdefer prep.deinit(self.allocator);

                images[written] = .{
                    .pixels = prep.pixels,
                    .width = prep.width,
                    .height = prep.height,
                    .grid = prep.grid,
                    .token_count = @intCast(prep.token_count),
                };
                token_counts[written] = @intCast(prep.token_count);
                written += 1;

                prep.pixels = &.{};
                prep.deinit(self.allocator);
            }
        }

        if (written != image_count) return error.InvalidState;

        return VisionPromptInput{
            .prefill = .{
                .images = images,
                .image_token_id = @intCast(self.loaded.config.image_token_id),
            },
            .token_counts = token_counts,
        };
    }

    fn countInputImageParts(chat: *Chat) usize {
        var count: usize = 0;
        for (0..chat.conv.len()) |item_index| {
            const item = chat.conv.getItem(item_index) orelse continue;
            const msg = item.asMessage() orelse continue;
            for (0..msg.partCount()) |part_index| {
                const part = msg.getPart(part_index) orelse continue;
                if (part.getContentType() == .input_image) count += 1;
            }
        }
        return count;
    }

    fn buildVisionPreprocessOptions(self: *const LocalEngine) !image_mod.VisionPreprocessOptions {
        const patch_size = try requirePositiveConfigU32(self.loaded.config.vision_patch_size);
        const temporal_patch_size = try requirePositiveConfigU32(self.loaded.config.vision_temporal_patch_size);
        const spatial_merge_size = try requirePositiveConfigU32(self.loaded.config.vision_spatial_merge_size);
        const resize_factor = try std.math.mul(u32, patch_size, spatial_merge_size);

        // Cross-check patch_size if preprocessor_config provided one.
        if (self.preproc_config.patch_size > 0 and
            self.preproc_config.patch_size != patch_size)
        {
            log.warn("load", "preprocessor_config.json patch_size differs from config.json vision_patch_size; using config.json", .{
                .preprocessor = self.preproc_config.patch_size,
                .config = patch_size,
            });
        }

        const encoder_max = self.backend.visionMaxPixels();

        // Determine pixel limits for smart resize.
        //
        // Priority: preprocessor_config.json values (clamped to encoder limit),
        // then config.json-based defaults.
        var min_pixels: u64 = 0;
        var max_pixels: u64 = 0;
        if (self.preproc_config.max_pixels > 0) {
            // Honour preprocessor_config.json, clamped to backend's vision encoder limit.
            max_pixels = self.preproc_config.max_pixels;
            min_pixels = self.preproc_config.min_pixels;
            if (max_pixels > encoder_max) {
                log.warn("load", "preprocessor_config.json max_pixels exceeds vision encoder limit; clamping", .{
                    .requested = max_pixels,
                    .limit = encoder_max,
                });
                max_pixels = encoder_max;
            }
            // Ensure min does not exceed clamped max.
            if (min_pixels > max_pixels) min_pixels = 0;
        } else if (self.loaded.config.vision_num_position_embeddings > 0) {
            // Fallback when preprocessor_config.json absent or has no pixel limits.
            const n_pos: u64 = std.math.cast(u64, self.loaded.config.vision_num_position_embeddings) orelse 0;
            if (n_pos > 0) {
                const base_pixels = n_pos * @as(u64, patch_size) * @as(u64, patch_size);
                if (temporal_patch_size == 1 and spatial_merge_size == 1) {
                    min_pixels = base_pixels;
                    max_pixels = base_pixels;
                } else {
                    // Dynamic-resolution without preprocessor_config: conservative default.
                    max_pixels = encoder_max;
                }
            }
        }

        return .{
            .normalize = .minus_one_to_one,
            .temporal_frames = temporal_patch_size,
            .patch_size = patch_size,
            .temporal_patch_size = temporal_patch_size,
            .spatial_merge_size = spatial_merge_size,
            .smart_resize = .{
                .factor = resize_factor,
                .min_pixels = min_pixels,
                .max_pixels = max_pixels,
            },
        };
    }

    fn requirePositiveConfigU32(value: i32) !u32 {
        if (value <= 0) return error.UnsupportedContentType;
        return std.math.cast(u32, value) orelse error.UnsupportedContentType;
    }

    /// Load raw image bytes from a URL (file:// or data: scheme).
    fn loadImageBytes(allocator: std.mem.Allocator, image_url: []const u8) ![]u8 {
        if (std.mem.startsWith(u8, image_url, "file://")) {
            return loadImageFromFile(allocator, image_url["file://".len..]);
        }
        return decodeImageDataUrl(allocator, image_url);
    }

    /// Read image bytes directly from a local file path.
    fn loadImageFromFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
        if (path.len == 0) return error.UnsupportedContentType;
        const file = std.fs.openFileAbsolute(path, .{}) catch return error.UnsupportedContentType;
        defer file.close();
        return file.readToEndAlloc(allocator, 100 * 1024 * 1024) catch return error.UnsupportedContentType;
    }

    fn decodeImageDataUrl(allocator: std.mem.Allocator, image_url: []const u8) ![]u8 {
        if (!std.mem.startsWith(u8, image_url, "data:")) return error.UnsupportedContentType;

        const comma_index = std.mem.indexOfScalar(u8, image_url, ',') orelse return error.UnsupportedContentType;
        if (comma_index <= "data:".len or comma_index + 1 >= image_url.len) return error.UnsupportedContentType;

        const metadata = image_url["data:".len..comma_index];
        if (!std.mem.endsWith(u8, metadata, ";base64")) return error.UnsupportedContentType;

        const mime_end = std.mem.indexOfScalar(u8, metadata, ';') orelse metadata.len;
        const mime = metadata[0..mime_end];
        if (!std.mem.startsWith(u8, mime, "image/")) return error.UnsupportedContentType;

        const payload = image_url[comma_index + 1 ..];
        const decoded_capacity = std.base64.standard.Decoder.calcSizeForSlice(payload) catch return error.InvalidArgument;
        const decoded = try allocator.alloc(u8, decoded_capacity);
        errdefer allocator.free(decoded);

        std.base64.standard.Decoder.decode(decoded, payload) catch return error.InvalidArgument;
        return decoded;
    }

    fn expandImagePadTokens(
        allocator: std.mem.Allocator,
        tokens: []const u32,
        image_token_id: u32,
        token_counts: []const usize,
        boundaries: VisionBoundaryTokens,
    ) ![]u32 {
        if (token_counts.len == 0) return allocator.dupe(u32, tokens);

        var placeholders: usize = 0;
        for (tokens) |tok| {
            if (tok == image_token_id) placeholders += 1;
        }
        if (placeholders != token_counts.len) return error.InvalidPromptImageTokens;

        var expanded_len: usize = 0;
        var image_idx_for_len: usize = 0;
        for (tokens, 0..) |tok, token_idx| {
            if (tok != image_token_id) {
                expanded_len += 1;
                continue;
            }

            const count = token_counts[image_idx_for_len];
            if (count == 0) return error.InvalidImageDimensions;
            expanded_len = try std.math.add(usize, expanded_len, count);

            if (boundaries.start_token_id) |start_token_id| {
                if (token_idx == 0 or tokens[token_idx - 1] != start_token_id) {
                    expanded_len = try std.math.add(usize, expanded_len, 1);
                }
            }
            if (boundaries.end_token_id) |end_token_id| {
                if (token_idx + 1 >= tokens.len or tokens[token_idx + 1] != end_token_id) {
                    expanded_len = try std.math.add(usize, expanded_len, 1);
                }
            }
            image_idx_for_len += 1;
        }

        const out = try allocator.alloc(u32, expanded_len);
        errdefer allocator.free(out);

        var image_idx: usize = 0;
        var write_idx: usize = 0;
        for (tokens, 0..) |tok, token_idx| {
            if (tok == image_token_id) {
                const repeat = token_counts[image_idx];
                if (boundaries.start_token_id) |start_token_id| {
                    if (token_idx == 0 or tokens[token_idx - 1] != start_token_id) {
                        out[write_idx] = start_token_id;
                        write_idx += 1;
                    }
                }
                @memset(out[write_idx .. write_idx + repeat], image_token_id);
                write_idx += repeat;
                if (boundaries.end_token_id) |end_token_id| {
                    if (token_idx + 1 >= tokens.len or tokens[token_idx + 1] != end_token_id) {
                        out[write_idx] = end_token_id;
                        write_idx += 1;
                    }
                }
                image_idx += 1;
            } else {
                out[write_idx] = tok;
                write_idx += 1;
            }
        }

        if (write_idx != expanded_len or image_idx != token_counts.len) return error.InvalidPromptImageTokens;
        return out;
    }

    fn resolveVisionBoundaryTokens(self: *const LocalEngine) VisionBoundaryTokens {
        var boundaries = VisionBoundaryTokens{};

        boundaries.start_token_id = if (self.loaded.config.vision_start_token_id > 0)
            std.math.cast(u32, self.loaded.config.vision_start_token_id)
        else
            null;
        if (boundaries.start_token_id == null) {
            boundaries.start_token_id = tokenIdByCandidates(&self.tok, &.{ "<|vision_start|>", "<|image_start|>" });
        }

        boundaries.end_token_id = if (self.loaded.config.vision_end_token_id > 0)
            std.math.cast(u32, self.loaded.config.vision_end_token_id)
        else
            null;
        if (boundaries.end_token_id == null) {
            boundaries.end_token_id = tokenIdByCandidates(&self.tok, &.{ "<|vision_end|>", "<|image_end|>" });
        }

        return boundaries;
    }

    fn tokenIdByCandidates(tok: *const tokenizer_mod.Tokenizer, candidates: []const []const u8) ?u32 {
        for (candidates) |name| {
            const id_i32 = tok.tokenizer_handle.tokenToId(name) orelse continue;
            if (id_i32 < 0) continue;
            return std.math.cast(u32, id_i32) orelse continue;
        }
        return null;
    }

    /// Encode text to token IDs.
    pub fn encode(self: *LocalEngine, text: []const u8) ![]u32 {
        return self.tok.encode(text);
    }

    /// Decode token IDs to text.
    pub fn decode(self: *LocalEngine, tokens: []const u32) ![]u8 {
        return self.tok.decode(tokens);
    }

    /// Get a pointer to the tokenizer (for streaming, etc).
    pub fn tokenizer(self: *LocalEngine) *tokenizer_mod.Tokenizer {
        return &self.tok;
    }

    /// Run inference on a raw prompt string.
    ///
    /// This is the lower-level generation interface for CLI and other use cases
    /// that don't use Chat objects. The result includes all generated tokens
    /// (including the prompt).
    ///
    /// Unlike generate(), this does NOT apply chat templates - the prompt is
    /// used as-is. Use this when you've already formatted the prompt.
    pub fn run(self: *LocalEngine, prompt: []const u8, config: inference_types.InferenceConfig) !inference_types.InferenceState {
        log.debug("inference", "Generation path", .{ .path = "scheduler" }, @src());
        return self.runWithScheduler(prompt, config);
    }

    /// Run inference using Scheduler (continuous batching path).
    fn runWithScheduler(
        self: *LocalEngine,
        prompt: []const u8,
        config: inference_types.InferenceConfig,
    ) !inference_types.InferenceState {
        // Raw inference runs share the same scheduler path and need the same
        // explicit execution-thread cleanup boundary as chat generation.
        defer self.backend.cleanupExecutionThreadState();

        // Tokenize prompt
        const encoded_tokens = try self.tok.encode(prompt);
        defer self.allocator.free(encoded_tokens);

        // Prepend BOS token if configured
        var prepend_bos = config.bos_token_id != null;
        if (prepend_bos and encoded_tokens.len > 0 and encoded_tokens[0] == config.bos_token_id.?) {
            prepend_bos = false;
        }

        const prompt_tokens = if (prepend_bos) blk: {
            const tokens = try self.allocator.alloc(u32, encoded_tokens.len + 1);
            tokens[0] = config.bos_token_id.?;
            @memcpy(tokens[1..], encoded_tokens);
            break :blk tokens;
        } else try self.allocator.dupe(u32, encoded_tokens);
        defer self.allocator.free(prompt_tokens);

        const prompt_len = prompt_tokens.len;

        // Create scheduler for this request
        var scheduler = try BackendScheduler.init(self.allocator, &self.backend, .{
            .default_eos_token_ids = config.eos_token_ids,
            .default_sampling = config.sampling,
            .state_descriptors = self.scheduler_state_descriptors,
        });
        defer scheduler.deinit();

        // Wrap token callback if provided (Scheduler uses different signature)
        const CallbackWrapper = struct {
            original_callback: ?inference_types.TokenCallback,
            original_data: ?*anyopaque,

            fn wrap(request_id: u64, token: u32, is_final: bool, in_thinking: bool, user_data: ?*anyopaque) void {
                _ = request_id;
                _ = is_final;
                const wrapper: *@This() = @ptrCast(@alignCast(user_data));
                if (wrapper.original_callback) |cb| {
                    cb(token, in_thinking, wrapper.original_data);
                }
            }
        };

        var callback_wrapper = CallbackWrapper{
            .original_callback = config.token_callback,
            .original_data = config.callback_data,
        };

        // Generate synchronously
        var result = try scheduler.generateSync(prompt_tokens, config.max_new_tokens, .{
            .eos_token_ids = config.eos_token_ids,
            .stop_sequences = config.stop_sequences,
            .callback = if (config.token_callback != null) CallbackWrapper.wrap else null,
            .callback_data = if (config.token_callback != null) @ptrCast(&callback_wrapper) else null,
            .sampling = config.sampling,
            .stop_flag = config.stop_flag,
            .return_final_logits = true,
        });
        errdefer result.deinit(self.allocator);

        // Capture timing from scheduler result
        const prefill_ns = result.prefill_ns;
        const decode_ns = result.decode_ns;

        // Capture values before deinit
        const generated_len = result.tokens.len;
        const finish_reason: inference_types.FinishReason = switch (result.finish_reason) {
            .in_progress => .eos_token, // Shouldn't happen for sync
            .eos_token => .eos_token,
            .length => .length,
            .stop_sequence => .stop_sequence,
            .cancelled => .cancelled,
        };

        // Build all_tokens = prompt + generated
        const all_tokens = try self.allocator.alloc(u32, prompt_len + generated_len);
        errdefer self.allocator.free(all_tokens);
        @memcpy(all_tokens[0..prompt_len], prompt_tokens);
        @memcpy(all_tokens[prompt_len..], result.tokens);

        if (result.final_logits.len != self.backend.vocabSize()) {
            return error.InvalidStateDescriptorBinding;
        }
        const final_logits = result.final_logits;
        result.final_logits = &.{};

        // Free scheduler result tokens (we've copied them)
        result.deinit(self.allocator);

        return inference_types.InferenceState{
            .tokens = all_tokens,
            .final_logits = final_logits,
            .prompt_len = prompt_len,
            .generated_len = generated_len,
            .prefill_ns = prefill_ns,
            .decode_ns = decode_ns,
            .finish_reason = finish_reason,
        };
    }

    /// Create a scheduler for continuous batching.
    ///
    /// The scheduler allows multiple concurrent generation requests to share
    /// this engine's compute resources efficiently. Requests can join/leave
    /// at token boundaries.
    ///
    /// The returned scheduler borrows the engine's backend - the engine must
    /// outlive the scheduler.
    pub fn createScheduler(self: *LocalEngine, config: SchedulerConfig) !BackendScheduler {
        // Merge config with engine's default EOS tokens
        var merged_config = config;
        if (merged_config.default_eos_token_ids.len == 0) {
            merged_config.default_eos_token_ids = self.gen_config.eos_token_ids;
        }
        if (merged_config.tokenizer == null) {
            merged_config.tokenizer = &self.tok;
        }
        if (merged_config.state_descriptors.len == 0) {
            merged_config.state_descriptors = self.scheduler_state_descriptors;
        }

        return BackendScheduler.init(self.allocator, &self.backend, merged_config);
    }

    const TemporaryStateBindings = struct {
        const StateBlockStorage = struct {
            bytes: []align(64) u8 = &.{},
        };

        handles: []runtime_contract.StateBlockHandle = &.{},
        storage: []StateBlockStorage = &.{},

        fn deinit(self: *TemporaryStateBindings, allocator: std.mem.Allocator) void {
            for (self.storage) |entry| {
                if (entry.bytes.len > 0) allocator.free(entry.bytes);
            }
            if (self.storage.len > 0) allocator.free(self.storage);
            if (self.handles.len > 0) allocator.free(self.handles);
            self.* = .{};
        }
    };

    fn allocateTemporaryStateBytes(
        allocator: std.mem.Allocator,
        align_bytes: u16,
        size_bytes: usize,
    ) ![]align(64) u8 {
        if (align_bytes == 0 or align_bytes > 64) return error.InvalidStateDescriptorBinding;
        return try allocator.alignedAlloc(u8, .@"64", size_bytes);
    }

    fn allocateTemporaryStateBindingsForDescriptors(
        allocator: std.mem.Allocator,
        descriptors: []const runtime_contract.StateDescriptor,
    ) !TemporaryStateBindings {
        if (descriptors.len == 0) return .{};

        var bindings = TemporaryStateBindings{};
        bindings.handles = try allocator.alloc(runtime_contract.StateBlockHandle, descriptors.len);
        errdefer allocator.free(bindings.handles);
        bindings.storage = try allocator.alloc(TemporaryStateBindings.StateBlockStorage, descriptors.len);
        errdefer allocator.free(bindings.storage);
        for (bindings.storage) |*entry| entry.* = .{};

        var initialized: usize = 0;
        errdefer {
            for (bindings.storage[0..initialized]) |entry| {
                allocator.free(entry.bytes);
            }
        }

        for (descriptors, 0..) |descriptor, idx| {
            const requested_size = std.math.cast(usize, descriptor.size_bytes) orelse {
                return error.InvalidStateDescriptorBinding;
            };
            if (requested_size == 0) return error.InvalidStateDescriptorBinding;
            const size_bytes = requested_size;
            const align_bytes: u16 = if (descriptor.align_bytes == 0) 64 else descriptor.align_bytes;

            const bytes = try allocateTemporaryStateBytes(allocator, align_bytes, size_bytes);
            const should_zero = runtime_contract.shouldZeroStateForLifecycleAction(&descriptor, .alloc) catch |err| switch (err) {
                error.InvalidStateLifecycleAction => false,
                else => return err,
            };
            if (should_zero) @memset(bytes, 0);

            bindings.storage[idx] = .{ .bytes = bytes };
            bindings.handles[idx] = .{
                .id = descriptor.id,
                .ptr = bytes.ptr,
                .size = @intCast(bytes.len),
                .align_bytes = align_bytes,
            };
            initialized += 1;
        }

        return bindings;
    }

    /// Extract embeddings from text.
    ///
    /// Runs the full transformer forward pass and returns pooled hidden states
    /// as a dense vector embedding. Unlike generate() which produces logits,
    /// this returns the final layer's hidden state directly.
    ///
    /// Args:
    ///   text: Input text to embed
    ///   pooling: Strategy for reducing sequence to single vector
    ///   normalize: Whether to L2-normalize the output embedding
    ///   embedding_out: Caller-allocated buffer of size embeddingDim()
    pub fn embed(
        self: *LocalEngine,
        text: []const u8,
        pooling: PoolingStrategy,
        normalize: bool,
        embedding_out: []f32,
    ) !void {
        // Tokenize input
        const tokens = try self.tok.encode(text);
        defer self.allocator.free(tokens);

        try self.embedTokens(tokens, pooling, normalize, embedding_out);
    }

    /// Extract embeddings from pre-tokenized input.
    ///
    /// Like embed() but skips tokenization for callers who already have tokens.
    pub fn embedTokens(
        self: *LocalEngine,
        tokens: []const u32,
        pooling: PoolingStrategy,
        normalize: bool,
        embedding_out: []f32,
    ) !void {
        var temp_bindings = try allocateTemporaryStateBindingsForDescriptors(
            self.allocator,
            self.scheduler_state_descriptors,
        );
        defer temp_bindings.deinit(self.allocator);

        if (temp_bindings.handles.len > 0) {
            try self.backend.bindSlotStateBlocks(0, temp_bindings.handles);
            defer self.backend.unbindSlotStateBlocks(0);
        }

        try self.backend.embed(tokens, pooling, normalize, embedding_out);
    }

    /// Returns the embedding dimension (d_model) for this model.
    pub fn embeddingDim(self: *const LocalEngine) usize {
        return self.backend.embeddingDim();
    }

    /// Count tokens for the current chat history, optionally with an additional message.
    ///
    /// This applies the chat template and tokenizes the result, returning the exact
    /// token count that would be used for generation. Useful for context window management.
    ///
    /// Args:
    ///   chat: The chat with conversation history
    ///   additional_message: Optional message to include in the count (not added to chat)
    ///   opts: Options for template rendering (same as GenerateOptions)
    ///
    /// Returns: Token count, or error if template/tokenization fails
    pub fn countTokens(
        self: *LocalEngine,
        chat: *Chat,
        additional_message: ?[]const u8,
        opts: struct {
            template_override: ?[]const u8 = null,
            extra_context_json: ?[]const u8 = null,
        },
    ) !usize {
        // If additional_message provided, temporarily add it to get accurate count
        var temp_added = false;
        if (additional_message) |msg| {
            if (msg.len > 0) {
                try chat.append(.user, msg);
                temp_added = true;
            }
        }
        defer if (temp_added) {
            // Remove the temporarily added message
            _ = chat.pop();
        };

        // Handle empty chat - return 0 (no tokens to count)
        // Many chat templates fail on empty message arrays
        if (chat.len() == 0) {
            return 0;
        }

        // Format messages to JSON using protocol layer (single source of truth)
        const messages_json = try protocol.completions.serialize(
            self.allocator,
            chat.conv,
            .{ .image_content_type = .image },
        );
        defer self.allocator.free(messages_json);

        // Apply chat template with optional overrides
        const prompt = self.renderPromptWithCachedTemplate(
            messages_json,
            true, // add_generation_prompt
            opts.template_override,
            opts.extra_context_json,
        ) catch |err| {
            // If no chat template or template render fails, fall back to counting raw messages
            if (err == error.MissingChatTemplate or err == error.FileNotFound or err == error.EvalError) {
                // Simple fallback: count tokens in raw messages
                var total_tokens: usize = 0;
                for (0..chat.len()) |i| {
                    if (chat.get(i)) |item| {
                        if (item.asMessage()) |msg| {
                            const text = msg.getFirstText();
                            const tokens = try self.tok.encode(text);
                            defer self.allocator.free(tokens);
                            total_tokens += tokens.len;
                        }
                    }
                }
                return total_tokens;
            }
            return err;
        };
        defer self.allocator.free(prompt);

        // Tokenize the rendered prompt
        const tokens = try self.tok.encode(prompt);
        defer self.allocator.free(tokens);

        // Account for BOS token if it would be added
        var count = tokens.len;
        if (self.gen_config.bos_token_id) |bos| {
            if (tokens.len == 0 or tokens[0] != bos) {
                count += 1;
            }
        }

        return count;
    }

    /// Returns the model's maximum context length (from tokenizer_config.json).
    /// Returns null if not specified in the model config.
    pub fn maxContextLength(self: *const LocalEngine) ?u64 {
        return self.gen_config.model_max_length;
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "GenerationResult.deinit frees allocated memory" {
    const allocator = std.testing.allocator;

    // Create owned slices that deinit should free
    const text = try allocator.dupe(u8, "Hello, world!");
    errdefer allocator.free(text);
    const tokens = try allocator.dupe(u32, &[_]u32{ 1, 2, 3, 4, 5 });

    const result = GenerationResult{
        .text = text,
        .tokens = tokens,
        .prompt_tokens = 10,
        .generated_tokens = 5,
        .prefill_ns = 1000,
        .decode_ns = 2000,
    };

    // deinit should free both allocations without leaking
    result.deinit(allocator);
}

test "GenerationResult struct fields" {
    const allocator = std.testing.allocator;

    const text = try allocator.dupe(u8, "test output");
    errdefer allocator.free(text);
    const tokens = try allocator.dupe(u32, &[_]u32{ 100, 200, 300 });

    const result = GenerationResult{
        .text = text,
        .tokens = tokens,
        .prompt_tokens = 42,
        .generated_tokens = 3,
        .prefill_ns = 123456,
        .decode_ns = 789012,
    };
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings("test output", result.text);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 100, 200, 300 }, result.tokens);
    try std.testing.expectEqual(@as(usize, 42), result.prompt_tokens);
    try std.testing.expectEqual(@as(usize, 3), result.generated_tokens);
    try std.testing.expectEqual(@as(u64, 123456), result.prefill_ns);
    try std.testing.expectEqual(@as(u64, 789012), result.decode_ns);
}

test "GenerateOptions struct defaults" {
    const opts = GenerateOptions{};

    try std.testing.expect(opts.max_tokens == null);
    try std.testing.expect(opts.temperature == null);
    try std.testing.expect(opts.top_k == null);
    try std.testing.expect(opts.top_p == null);
    try std.testing.expect(opts.min_p == null);
    try std.testing.expect(opts.repetition_penalty == null);
    try std.testing.expect(opts.token_callback == null);
    try std.testing.expect(opts.callback_data == null);
    try std.testing.expect(!opts.completions_mode);
}

test "GenerateOptions completions_mode flag" {
    const opts = GenerateOptions{ .completions_mode = true };
    try std.testing.expect(opts.completions_mode);
    // Other fields remain at defaults
    try std.testing.expect(opts.max_tokens == null);
    try std.testing.expect(!opts.raw_output);
}

test "GenerateOptions struct overrides" {
    const opts = GenerateOptions{
        .max_tokens = 100,
        .temperature = 0.7,
        .top_k = 40,
        .top_p = 0.9,
        .min_p = 0.05,
        .repetition_penalty = 1.1,
    };

    try std.testing.expectEqual(@as(usize, 100), opts.max_tokens.?);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), opts.temperature.?, 0.001);
    try std.testing.expectEqual(@as(usize, 40), opts.top_k.?);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), opts.top_p.?, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.05), opts.min_p.?, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.1), opts.repetition_penalty.?, 0.001);
}

test "GenerateOptions struct callback" {
    const TestCallback = struct {
        fn callback(token_id: u32, _: bool, user_data: ?*anyopaque) void {
            if (user_data) |ptr| {
                const count: *usize = @ptrCast(@alignCast(ptr));
                count.* += 1;
                _ = token_id;
            }
        }
    };

    var call_count: usize = 0;
    const opts = GenerateOptions{
        .token_callback = TestCallback.callback,
        .callback_data = @ptrCast(&call_count),
    };

    // Verify callback is set and callable
    try std.testing.expect(opts.token_callback != null);
    try std.testing.expect(opts.callback_data != null);

    // Call the callback to verify it works
    opts.token_callback.?(42, false, opts.callback_data);
    try std.testing.expectEqual(@as(usize, 1), call_count);

    opts.token_callback.?(43, false, opts.callback_data);
    try std.testing.expectEqual(@as(usize, 2), call_count);
}

test "GenerationResult struct empty" {
    const allocator = std.testing.allocator;

    const text = try allocator.dupe(u8, "");
    errdefer allocator.free(text);
    const tokens = try allocator.alloc(u32, 0);

    const result = GenerationResult{
        .text = text,
        .tokens = tokens,
        .prompt_tokens = 0,
        .generated_tokens = 0,
        .prefill_ns = 0,
        .decode_ns = 0,
    };
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), result.text.len);
    try std.testing.expectEqual(@as(usize, 0), result.tokens.len);
}

test "allocateTemporaryStateBindingsForDescriptors allocates aligned descriptor blocks" {
    const descriptors = [_]runtime_contract.StateDescriptor{
        .{
            .id = 11,
            .size_bytes = 64,
            .align_bytes = 64,
            .zero_init = true,
            .lifecycle = .slot_persistent,
        },
        .{
            .id = 12,
            .size_bytes = 32,
            .align_bytes = 16,
            .zero_init = false,
            .lifecycle = .request_scoped,
        },
    };

    var bindings = try LocalEngine.allocateTemporaryStateBindingsForDescriptors(
        std.testing.allocator,
        descriptors[0..],
    );
    defer bindings.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), bindings.handles.len);

    for (descriptors, 0..) |descriptor, idx| {
        const handle = bindings.handles[idx];
        try std.testing.expectEqual(descriptor.id, handle.id);
        try std.testing.expectEqual(descriptor.size_bytes, handle.size);
        try std.testing.expectEqual(descriptor.align_bytes, handle.align_bytes);
        try std.testing.expectEqual(@as(usize, 0), @intFromPtr(handle.ptr) % 64);
    }

    for (bindings.storage) |entry| {
        for (entry.bytes) |byte| {
            try std.testing.expectEqual(@as(u8, 0), byte);
        }
    }
}

test "allocateTemporaryStateBindingsForDescriptors rejects zero-sized descriptor" {
    const descriptors = [_]runtime_contract.StateDescriptor{
        .{
            .id = 99,
            .size_bytes = 0,
            .align_bytes = 64,
            .zero_init = true,
            .lifecycle = .slot_persistent,
        },
    };
    try std.testing.expectError(
        error.InvalidStateDescriptorBinding,
        LocalEngine.allocateTemporaryStateBindingsForDescriptors(std.testing.allocator, descriptors[0..]),
    );
}

test "decodeImageDataUrl decodes base64 image payload" {
    const allocator = std.testing.allocator;
    const url = "data:image/png;base64,AQID";

    const decoded = try LocalEngine.decodeImageDataUrl(allocator, url);
    defer allocator.free(decoded);

    try std.testing.expectEqualSlices(u8, &[_]u8{ 1, 2, 3 }, decoded);
}

test "decodeImageDataUrl rejects unsupported scheme" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(
        error.UnsupportedContentType,
        LocalEngine.decodeImageDataUrl(allocator, "https://example.com/image.png"),
    );
}

test "expandImagePadTokens repeats placeholders per token count" {
    const allocator = std.testing.allocator;
    const tokens = [_]u32{ 10, 99, 11, 99, 12 };
    const token_counts = [_]usize{ 3, 1 };

    const expanded = try LocalEngine.expandImagePadTokens(allocator, &tokens, 99, &token_counts, .{});
    defer allocator.free(expanded);

    try std.testing.expectEqualSlices(u32, &[_]u32{ 10, 99, 99, 99, 11, 99, 12 }, expanded);
}

test "expandImagePadTokens inserts boundary tokens around expanded image span" {
    const allocator = std.testing.allocator;
    const tokens = [_]u32{ 10, 99, 11 };
    const token_counts = [_]usize{2};

    const expanded = try LocalEngine.expandImagePadTokens(
        allocator,
        &tokens,
        99,
        &token_counts,
        .{ .start_token_id = 7, .end_token_id = 8 },
    );
    defer allocator.free(expanded);

    try std.testing.expectEqualSlices(u32, &[_]u32{ 10, 7, 99, 99, 8, 11 }, expanded);
}

test "expandImagePadTokens preserves existing boundary tokens" {
    const allocator = std.testing.allocator;
    const tokens = [_]u32{ 10, 7, 99, 8, 11 };
    const token_counts = [_]usize{3};

    const expanded = try LocalEngine.expandImagePadTokens(
        allocator,
        &tokens,
        99,
        &token_counts,
        .{ .start_token_id = 7, .end_token_id = 8 },
    );
    defer allocator.free(expanded);

    try std.testing.expectEqualSlices(u32, &[_]u32{ 10, 7, 99, 99, 99, 8, 11 }, expanded);
}

test "expandImagePadTokens rejects placeholder mismatch" {
    const allocator = std.testing.allocator;
    const tokens = [_]u32{ 10, 99, 11 };
    const token_counts = [_]usize{ 1, 2 };

    try std.testing.expectError(
        error.InvalidPromptImageTokens,
        LocalEngine.expandImagePadTokens(allocator, &tokens, 99, &token_counts, .{}),
    );
}

test "maxThinkingTokensForEffort maps effort levels correctly" {
    try std.testing.expectEqual(@as(usize, 4096), maxThinkingTokensForEffort(null));
    try std.testing.expectEqual(@as(usize, 0), maxThinkingTokensForEffort("none"));
    try std.testing.expectEqual(@as(usize, 512), maxThinkingTokensForEffort("low"));
    try std.testing.expectEqual(@as(usize, 4096), maxThinkingTokensForEffort("medium"));
    try std.testing.expectEqual(@as(usize, 16384), maxThinkingTokensForEffort("high"));
    try std.testing.expectEqual(@as(usize, 32768), maxThinkingTokensForEffort("xhigh"));
    try std.testing.expectEqual(@as(usize, 4096), maxThinkingTokensForEffort("unknown"));
}

test "buildEffectiveContext default enables thinking" {
    const allocator = std.testing.allocator;
    const ctx = try buildEffectiveContext(allocator, GenerateOptions{});
    defer if (ctx) |c| allocator.free(c);
    try std.testing.expect(ctx != null);
    try std.testing.expectEqualStrings("{\"enable_thinking\": true}", ctx.?);
}

test "buildEffectiveContext none disables thinking" {
    const allocator = std.testing.allocator;
    const ctx = try buildEffectiveContext(allocator, GenerateOptions{ .reasoning_effort = "none" });
    defer if (ctx) |c| allocator.free(c);
    try std.testing.expect(ctx != null);
    try std.testing.expectEqualStrings("{\"enable_thinking\": false}", ctx.?);
}

test "buildEffectiveContext medium enables thinking" {
    const allocator = std.testing.allocator;
    const ctx = try buildEffectiveContext(allocator, GenerateOptions{ .reasoning_effort = "medium" });
    defer if (ctx) |c| allocator.free(c);
    try std.testing.expect(ctx != null);
    try std.testing.expectEqualStrings("{\"enable_thinking\": true}", ctx.?);
}

test "buildEffectiveContext max_reasoning_tokens=0 disables thinking" {
    const allocator = std.testing.allocator;
    const ctx = try buildEffectiveContext(allocator, GenerateOptions{ .max_reasoning_tokens = 0 });
    defer if (ctx) |c| allocator.free(c);
    try std.testing.expect(ctx != null);
    try std.testing.expectEqualStrings("{\"enable_thinking\": false}", ctx.?);
}

test "buildEffectiveContext max_reasoning_tokens>0 enables thinking" {
    const allocator = std.testing.allocator;
    const ctx = try buildEffectiveContext(allocator, GenerateOptions{ .max_reasoning_tokens = 128 });
    defer if (ctx) |c| allocator.free(c);
    try std.testing.expect(ctx != null);
    try std.testing.expectEqualStrings("{\"enable_thinking\": true}", ctx.?);
}

test "buildEffectiveContext max_reasoning_tokens=0 overrides effort" {
    const allocator = std.testing.allocator;
    // Even with reasoning_effort="high", max_reasoning_tokens=0 wins.
    const ctx = try buildEffectiveContext(allocator, GenerateOptions{
        .max_reasoning_tokens = 0,
        .reasoning_effort = "high",
    });
    defer if (ctx) |c| allocator.free(c);
    try std.testing.expect(ctx != null);
    try std.testing.expectEqualStrings("{\"enable_thinking\": false}", ctx.?);
}

test "buildEffectiveContext merges with existing extra_context_json" {
    const allocator = std.testing.allocator;
    const ctx = try buildEffectiveContext(allocator, GenerateOptions{
        .extra_context_json = "{\"tools\": []}",
        .reasoning_effort = "high",
    });
    defer if (ctx) |c| allocator.free(c);
    try std.testing.expect(ctx != null);
    try std.testing.expectEqualStrings("{\"tools\": [], \"enable_thinking\": true }", ctx.?);
}

test "buildEffectiveContext normalizes flat tools to nested" {
    const allocator = std.testing.allocator;
    const flat_tools =
        \\[{"type":"function","name":"calc","parameters":{"type":"object"}}]
    ;
    const ctx = try buildEffectiveContext(allocator, GenerateOptions{
        .tools_json = flat_tools,
    });
    defer if (ctx) |c| allocator.free(c);
    try std.testing.expect(ctx != null);
    // The context must contain the nested "function" wrapper.
    const result = ctx.?;
    try std.testing.expect(std.mem.indexOf(u8, result, "\"function\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"calc\"") != null);
}

test "buildEffectiveContext flat tools with thinking disabled" {
    const allocator = std.testing.allocator;
    const flat_tools =
        \\[{"type":"function","name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]
    ;
    const ctx = try buildEffectiveContext(allocator, GenerateOptions{
        .tools_json = flat_tools,
        .max_reasoning_tokens = 0,
    });
    defer if (ctx) |c| allocator.free(c);
    try std.testing.expect(ctx != null);
    const result = ctx.?;
    // Must have enable_thinking: false AND nested tool format.
    try std.testing.expect(std.mem.indexOf(u8, result, "\"enable_thinking\": false") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"function\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"get_weather\"") != null);
}

test "buildEffectiveContext preserves nested tools" {
    const allocator = std.testing.allocator;
    const nested_tools =
        \\[{"type":"function","function":{"name":"calc","parameters":{"type":"object"}}}]
    ;
    const ctx = try buildEffectiveContext(allocator, GenerateOptions{
        .tools_json = nested_tools,
    });
    defer if (ctx) |c| allocator.free(c);
    try std.testing.expect(ctx != null);
    const result = ctx.?;
    try std.testing.expect(std.mem.indexOf(u8, result, "\"function\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "\"calc\"") != null);
}

test "loadCachedChatTemplate reads inline template and special tokens" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const tokenizer_config_json =
        \\{
        \\  "chat_template": "{{ messages[0].content }}",
        \\  "bos_token": "<s>",
        \\  "eos_token": "</s>"
        \\}
    ;
    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = tokenizer_config_json });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    var cached = try LocalEngine.loadCachedChatTemplate(allocator, tmp_path);
    defer cached.deinit(allocator);

    try std.testing.expectEqualStrings("{{ messages[0].content }}", cached.template_source);
    try std.testing.expectEqualStrings("<s>", cached.bos_token);
    try std.testing.expectEqualStrings("</s>", cached.eos_token);
}

test "loadCachedChatTemplate falls back to chat_template.jinja" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const tokenizer_config_json =
        \\{
        \\  "bos_token": "<s>",
        \\  "eos_token": "</s>"
        \\}
    ;
    const template_content = "{% for m in messages %}{{ m.content }}{% endfor %}";
    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = tokenizer_config_json });
    try tmp_dir.dir.writeFile(.{ .sub_path = "chat_template.jinja", .data = template_content });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    var cached = try LocalEngine.loadCachedChatTemplate(allocator, tmp_path);
    defer cached.deinit(allocator);

    try std.testing.expectEqualStrings(template_content, cached.template_source);
    try std.testing.expectEqualStrings("<s>", cached.bos_token);
    try std.testing.expectEqualStrings("</s>", cached.eos_token);
}

test "loadCachedChatTemplate rejects non-object tokenizer config" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    try tmp_dir.dir.writeFile(.{ .sub_path = "tokenizer_config.json", .data = "[]" });

    const tmp_path = try tmp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    try std.testing.expectError(
        error.InvalidJson,
        LocalEngine.loadCachedChatTemplate(allocator, tmp_path),
    );
}
