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
const responses_mod = @import("../responses/root.zig");
const Chat = responses_mod.Chat;
const protocol = @import("protocol/root.zig");
const sampler = inference.sampling;
const inference_types = inference.types;
const FinishReason = inference_types.FinishReason;
const backend_root = @import("../inference/backend/root.zig");
const Backend = backend_root.Backend;
const vision_types = @import("../inference/backend/cpu/vision/types.zig");
const log = @import("../log.zig");
pub const PoolingStrategy = backend_root.PoolingStrategy;
const tokenizer_mod = @import("../tokenizer/root.zig");
const io = @import("../io/root.zig");
const image_mod = @import("../image/root.zig");
const model_loader = inference.model_loader;
const gen_config_mod = @import("../inference/config/generation.zig");
const validate_mod = @import("../validate/root.zig");
const ConstrainedSampler = validate_mod.sampler.ConstrainedSampler;
const GrammarConfig = validate_mod.sampler.GrammarConfig;
const tool_schema_mod = @import("tool_schema.zig");
const reasoning_parser_mod = responses_mod.reasoning_parser;
const commit_mod = @import("commit.zig");
const progress_mod = @import("../capi/progress.zig");

pub const ResolutionConfig = io.repository.ResolutionConfig;

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
    /// Override chat's max_tokens.
    max_tokens: ?usize = null,

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

    pub const PrefillProgressFn = *const fn (usize, usize, ?*anyopaque) callconv(.c) void;
};

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
    loaded: *model_loader.LoadedModel,

    /// Tokenizer for encoding/decoding.
    tok: tokenizer_mod.Tokenizer,

    /// Sampler for token selection.
    samp: sampler.Sampler,

    /// Compute backend.
    backend: Backend,

    /// Generation config from model (EOS tokens, etc).
    gen_config: gen_config_mod.GenerationConfig,

    /// Path to the model directory.
    model_path: []const u8,

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
        return initWithSeedAndResolutionConfig(allocator, model_path, 42, .{}, progress_mod.ProgressContext.NONE);
    }

    /// Initialize engine with a specific random seed.
    pub fn initWithSeed(allocator: std.mem.Allocator, model_path: []const u8, seed: u64) !LocalEngine {
        return initWithSeedAndResolutionConfig(allocator, model_path, seed, .{}, progress_mod.ProgressContext.NONE);
    }

    /// Initialize engine with resolution configuration.
    pub fn initWithResolutionConfig(
        allocator: std.mem.Allocator,
        model_path: []const u8,
        config: ResolutionConfig,
    ) !LocalEngine {
        return initWithSeedAndResolutionConfig(allocator, model_path, 42, config, progress_mod.ProgressContext.NONE);
    }

    /// Initialize engine with a specific random seed and resolution configuration.
    pub fn initWithSeedAndResolutionConfig(
        allocator: std.mem.Allocator,
        model_path: []const u8,
        seed: u64,
        config: ResolutionConfig,
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

        const model_load_options = backend_root.defaultModelLoadOptions();

        // Start model loading in background thread
        const ModelLoaderThread = struct {
            alloc: std.mem.Allocator,
            config_path: []const u8,
            weights_path: []const u8,
            load_options: model_loader.LoadOptions,
            prog: progress_mod.ProgressContext,
            loaded_model: ?model_loader.LoadedModel = null,
            err: ?anyerror = null,

            fn loadModel(self: *@This()) void {
                self.loaded_model = model_loader.loadModel(self.alloc, self.config_path, self.weights_path, self.load_options, self.prog) catch |e| {
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
        var tokenizer_instance = if (model_bundle.tokenizer_json()) |json|
            try tokenizer_mod.Tokenizer.initFromJson(allocator, json)
        else
            try tokenizer_mod.Tokenizer.initFromPath(allocator, model_bundle.tokenizer_path());
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
            loader_thread_state.loaded_model = try model_loader.loadModel(allocator, model_bundle.config_path(), wp, model_load_options, progress);
        }

        if (loader_thread_state.err) |e| return e;

        const loaded_model = try allocator.create(model_loader.LoadedModel);
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
        var compute_backend = try Backend.init(allocator, loaded_model, progress);
        errdefer compute_backend.deinit();

        {
            const now = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(now - timing_start_ns)) / 1_000_000.0;
            log.debug("inference", "Backend initialized", .{ .duration_ms = duration_ms }, @src());
            timing_start_ns = now;
        }

        // Warmup: full single-token forward pass to load all weights into memory
        try compute_backend.warmup();
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
            .model_path = resolved_model_path,
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
        self.* = undefined;
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

        // Apply chat template with optional overrides
        log.debug("inference", "Applying chat template", .{ .messages_len = messages_json.len }, @src());
        const prompt = gen_config_mod.applyChatTemplateWithOverrides(
            self.allocator,
            self.model_path,
            messages_json,
            true, // add_generation_prompt
            opts.template_override,
            opts.extra_context_json,
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
        // Use chat settings with optional overrides
        const base_max_tokens = opts.max_tokens orelse chat.max_tokens;
        const grammar_slack: usize = 64;

        // Determine if we're using tools (create grammar sampler if so)
        var tool_grammar_sampler: ?*ConstrainedSampler = null;
        var tool_grammar_schema: ?[]u8 = null;
        defer {
            if (tool_grammar_sampler) |gs| {
                gs.deinit();
                self.allocator.destroy(gs);
            }
            if (tool_grammar_schema) |schema| {
                self.allocator.free(schema);
            }
        }

        // If tools_json is provided (and tool_choice != "none"), create grammar sampler
        const use_tools = opts.tools_json != null and
            (opts.tool_choice == null or !std.mem.eql(u8, opts.tool_choice.?, "none"));

        if (use_tools) {
            // Convert tools to grammar schema
            tool_grammar_schema = try tool_schema_mod.toolsToGrammarSchema(self.allocator, opts.tools_json.?);

            // Create constrained sampler
            const gs = try self.allocator.create(ConstrainedSampler);
            errdefer self.allocator.destroy(gs);
            gs.* = try ConstrainedSampler.init(
                self.allocator,
                tool_grammar_schema.?,
                GrammarConfig{}, // Default config
                self.gen_config.eos_token_ids,
                null, // prefix_tokens
                null, // prefix_token_ids
            );
            tool_grammar_sampler = gs;
        }

        // Use tool grammar if created, otherwise fall back to chat's grammar
        const effective_grammar = tool_grammar_sampler orelse chat.grammar_sampler;

        const max_tokens = if (effective_grammar != null and base_max_tokens > 0)
            base_max_tokens + grammar_slack
        else
            base_max_tokens;
        const temperature = opts.temperature orelse chat.temperature;
        const top_k = opts.top_k orelse chat.top_k;
        const top_p = opts.top_p orelse chat.top_p;
        const min_p = opts.min_p orelse chat.min_p;
        const repetition_penalty = opts.repetition_penalty orelse chat.repetition_penalty;

        // Build sampling config
        var sampling_config = sampler.SamplingConfig{ .strategy = .greedy, .logit_bias = opts.logit_bias, .seed = opts.seed };
        if (temperature > 0 and (self.gen_config.do_sample or opts.temperature != null)) {
            sampling_config = .{
                .strategy = .top_k,
                .temperature = temperature,
                .top_k = top_k,
                .top_p = top_p,
                .min_p = min_p,
                .repetition_penalty = repetition_penalty,
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

        // Install prefill progress callback (cleared after generation)
        self.backend.setPrefillProgress(opts.prefill_progress_fn, opts.prefill_progress_data);
        defer self.backend.setPrefillProgress(null, null);

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
        });
        defer scheduler.deinit();

        // Wrap token callback if provided (Scheduler uses different signature)
        const CallbackWrapper = struct {
            original_callback: ?inference_types.TokenCallback,
            original_data: ?*anyopaque,

            fn wrap(request_id: u64, token: u32, is_final: bool, user_data: ?*anyopaque) void {
                _ = request_id;
                _ = is_final;
                const wrapper: *@This() = @ptrCast(@alignCast(user_data));
                if (wrapper.original_callback) |cb| {
                    cb(token, wrapper.original_data);
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

        // Generate synchronously
        log.debug("router", "generateSync starting", .{
            .prompt_tokens = prompt_tokens.len,
            .max_tokens = max_tokens,
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
        }) catch |err| {
            log.warn("inference", "Scheduler generation failed", .{
                .err = @errorName(err),
                .has_vision_input = @as(u8, @intFromBool(vision_input_ptr != null)),
            });
            return err;
        };
        defer result.deinit(self.allocator);

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

        // Check if this is a tool call (grammar completed with tool schema)
        const is_tool_call = is_tool_generation and grammar_sampler != null and
            grammar_sampler.?.state == .complete;

        // Map scheduler finish reason to session finish reason
        const finish_reason: FinishReason = if (is_tool_call) .tool_calls else switch (result.finish_reason) {
            .in_progress => .eos_token, // Shouldn't happen for sync
            .eos_token => .eos_token,
            .length => .length,
            .stop_sequence => .stop_sequence,
            .cancelled => .cancelled,
        };
        const finish_reason_str = finishReasonToString(finish_reason);

        if (is_tool_call) {
            // Local-only: parse grammar-constrained JSON into tool call fields
            var parsed_call = tool_schema_mod.parseToolCall(self.allocator, generated_text) catch {
                // If parsing fails, fall through to text commit
                try commit_mod.commitGenerationResult(self.allocator, chat, .{
                    .text = generated_text,
                    .prompt_tokens = prompt_len,
                    .completion_tokens = result.tokens.len,
                    .prefill_ns = result.prefill_ns,
                    .generation_ns = result.decode_ns,
                    .finish_reason = finishReasonToString(.eos_token),
                });
                return GenerationResult{
                    .text = generated_text,
                    .tokens = owned_tokens,
                    .prompt_tokens = prompt_len,
                    .generated_tokens = result.tokens.len,
                    .prefill_ns = result.prefill_ns,
                    .decode_ns = result.decode_ns,
                    .finish_reason = .eos_token,
                };
            };
            defer parsed_call.deinit(self.allocator);

            const call_id = try tool_schema_mod.generateCallId(self.allocator);
            defer self.allocator.free(call_id);

            const tc_input = [_]commit_mod.ToolCallInput{.{
                .id = call_id,
                .name = parsed_call.name,
                .arguments = parsed_call.arguments,
            }};

            try commit_mod.commitGenerationResult(self.allocator, chat, .{
                .text = generated_text,
                .tool_calls = &tc_input,
                .prompt_tokens = prompt_len,
                .completion_tokens = result.tokens.len,
                .prefill_ns = result.prefill_ns,
                .generation_ns = result.decode_ns,
                .finish_reason = finish_reason_str,
            });

            // Build ToolCallRef for the caller
            var tool_calls = try self.allocator.alloc(ToolCallRef, 1);
            errdefer self.allocator.free(tool_calls);

            tool_calls[0] = ToolCallRef{
                .item_index = chat.conv.len() - 1,
                .call_id = try self.allocator.dupe(u8, call_id),
                .name = try self.allocator.dupe(u8, parsed_call.name),
                .arguments = try self.allocator.dupe(u8, parsed_call.arguments),
            };

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

        // Text path: commit reasoning + assistant message via shared path
        const generation_json = try self.buildGenerationJson(chat, opts, max_tokens);
        defer self.allocator.free(generation_json);

        try commit_mod.commitGenerationResult(self.allocator, chat, .{
            .text = generated_text,
            .prompt_tokens = prompt_len,
            .completion_tokens = result.tokens.len,
            .prefill_ns = result.prefill_ns,
            .generation_ns = result.decode_ns,
            .finish_reason = finish_reason_str,
            .reasoning_tag = opts.reasoning_tag,
            .generation_json = generation_json,
        });

        // Return clean text (strip reasoning tags for caller)
        var parser = try reasoning_parser_mod.ReasoningParser.init(self.allocator, opts.reasoning_tag);
        defer parser.deinit();
        try parser.processChunk(generated_text);
        const parsed = try parser.finalize();

        const result_text = if (opts.raw_output) generated_text else if (parsed.reasoning != null) blk: {
            const duped = try self.allocator.dupe(u8, parsed.response orelse "");
            self.allocator.free(generated_text);
            break :blk duped;
        } else generated_text;

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
        const fixed_pixels: u32 = blk: {
            if (self.loaded.config.vision_num_position_embeddings <= 0) break :blk 0;
            if (temporal_patch_size != 1 or spatial_merge_size != 1) break :blk 0;
            const n_pos = std.math.cast(u64, self.loaded.config.vision_num_position_embeddings) orelse break :blk 0;
            const patch_area = try std.math.mul(u64, patch_size, patch_size);
            const pixels_u64 = try std.math.mul(u64, n_pos, patch_area);
            break :blk std.math.cast(u32, pixels_u64) orelse return error.InvalidShape;
        };

        return .{
            .normalize = .minus_one_to_one,
            .temporal_frames = temporal_patch_size,
            .patch_size = patch_size,
            .temporal_patch_size = temporal_patch_size,
            .spatial_merge_size = spatial_merge_size,
            .smart_resize = .{
                .factor = resize_factor,
                .min_pixels = fixed_pixels,
                .max_pixels = fixed_pixels,
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
        });
        defer scheduler.deinit();

        // Wrap token callback if provided (Scheduler uses different signature)
        const CallbackWrapper = struct {
            original_callback: ?inference_types.TokenCallback,
            original_data: ?*anyopaque,

            fn wrap(request_id: u64, token: u32, is_final: bool, user_data: ?*anyopaque) void {
                _ = request_id;
                _ = is_final;
                const wrapper: *@This() = @ptrCast(@alignCast(user_data));
                if (wrapper.original_callback) |cb| {
                    cb(token, wrapper.original_data);
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

        // Preserve run() behavior: return logits for the final sequence position.
        const final_logits = try self.allocator.alloc(f32, self.backend.vocabSize());
        errdefer self.allocator.free(final_logits);
        try self.backend.prefill(all_tokens, final_logits);

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

        return BackendScheduler.init(self.allocator, &self.backend, merged_config);
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

        // Run embedding extraction via backend
        try self.backend.embed(tokens, pooling, normalize, embedding_out);
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
        const prompt = gen_config_mod.applyChatTemplateWithOverrides(
            self.allocator,
            self.model_path,
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
        fn callback(token_id: u32, user_data: ?*anyopaque) void {
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
    opts.token_callback.?(42, opts.callback_data);
    try std.testing.expectEqual(@as(usize, 1), call_count);

    opts.token_callback.?(43, opts.callback_data);
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
