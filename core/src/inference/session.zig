//! Inference Session
//!
//! Runtime state management for LLM inference. Ties together model loading,
//! tokenization, sampling, and backend execution.

const std = @import("std");
const sampler = @import("sampling.zig");
const tokenizer_mod = @import("../tokenizer/root.zig");
const io = @import("../io/root.zig");
const Backend = @import("backend/root.zig").Backend;
const log = @import("../log.zig");
const progress_mod = @import("../capi/progress.zig");

/// Callback function type for streaming token output.
/// Called with each newly generated token ID and optional user data.
pub const TokenCallback = *const fn (token_id: u32, user_data: ?*anyopaque) void;

pub const InferenceConfig = struct {
    max_new_tokens: usize = 32,
    sampling: sampler.SamplingConfig = .{},
    eos_token_ids: []const u32 = &.{},
    /// BOS token to prepend to input (from model config)
    bos_token_id: ?u32 = null,
    /// Optional callback for streaming output. Called after each token is sampled.
    token_callback: ?TokenCallback = null,
    /// User data passed to the token callback
    callback_data: ?*anyopaque = null,
    /// Stop sequences (already tokenized). Generation stops when any sequence matches.
    /// Each inner slice is a tokenized stop sequence.
    stop_sequences: []const []const u32 = &.{},
    /// Optional stop flag for cancellation. When set to true, generation stops.
    /// This allows external cancellation (e.g., client disconnect) without
    /// waiting for the next callback invocation.
    stop_flag: ?*const std.atomic.Value(bool) = null,
};

/// Reason why generation stopped.
pub const FinishReason = enum(u8) {
    /// Generation stopped due to EOS token.
    eos_token = 0,
    /// Maximum token limit reached.
    length = 1,
    /// A stop sequence was matched.
    stop_sequence = 2,
    /// Model requested tool/function calls.
    /// The Core has auto-committed FunctionCallItem(s) to the Conversation.
    tool_calls = 3,
    /// Content was filtered (safety).
    content_filter = 4,
    /// Request was cancelled (e.g., client disconnect, stop flag set).
    cancelled = 5,

    /// Convert to C-compatible integer for C-API.
    pub fn toInt(self: FinishReason) u8 {
        return @intFromEnum(self);
    }
};

pub const InferenceState = struct {
    tokens: []u32,
    final_logits: []f32,
    prompt_len: usize,
    generated_len: usize,
    prefill_ns: u64,
    decode_ns: u64,
    finish_reason: FinishReason = .eos_token,
};

pub const Session = struct {
    allocator: std.mem.Allocator,
    loaded: *io.LoadedModel,
    tok: tokenizer_mod.Tokenizer,
    samp: sampler.Sampler,
    backend: Backend,

    /// Build session from already-loaded components.
    /// This is the single point of truth for session assembly, avoiding duplication
    /// between threaded and synchronous initialization paths.
    /// Includes debug timing/shape output when flags are enabled.
    fn buildFromComponents(
        allocator: std.mem.Allocator,
        loaded_model: *io.LoadedModel,
        tokenizer: tokenizer_mod.Tokenizer,
        rng_seed: u64,
    ) !Session {
        var timing_start_ns: i128 = std.time.nanoTimestamp();

        var sampler_state = try sampler.Sampler.init(allocator, rng_seed, @intCast(loaded_model.config.vocab_size));
        errdefer sampler_state.deinit();
        {
            const now = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(now - timing_start_ns)) / 1_000_000.0;
            log.debug("inference", "Sampler initialized", .{ .duration_ms = duration_ms }, @src());
            timing_start_ns = now;
        }

        var session = Session{
            .allocator = allocator,
            .loaded = loaded_model,
            .tok = tokenizer,
            .samp = sampler_state,
            .backend = undefined,
        };

        var backend_state = try Backend.init(allocator, session.loaded, progress_mod.ProgressContext.NONE);
        errdefer backend_state.deinit();
        session.backend = backend_state;
        {
            const now = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(now - timing_start_ns)) / 1_000_000.0;
            log.debug("inference", "Backend initialized", .{ .duration_ms = duration_ms }, @src());
            timing_start_ns = now;
        }

        try session.backend.warmup();
        {
            const now = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(now - timing_start_ns)) / 1_000_000.0;
            log.debug("inference", "Warmup complete", .{ .duration_ms = duration_ms }, @src());
        }

        return session;
    }

    /// Initialize session with explicit tokenizer path
    pub fn init(
        allocator: std.mem.Allocator,
        config_path: []const u8,
        weights_path: []const u8,
        tokenizer_path: []const u8,
        seed: u64,
    ) !Session {
        return initWithTokenizer(allocator, config_path, weights_path, tokenizer_path, null, seed);
    }

    /// Initialize session with in-memory tokenizer JSON
    pub fn initWithJson(
        allocator: std.mem.Allocator,
        config_path: []const u8,
        weights_path: []const u8,
        tokenizer_json: []const u8,
        seed: u64,
    ) !Session {
        return initWithTokenizer(allocator, config_path, weights_path, "", tokenizer_json, seed);
    }

    /// Internal init with optional tokenizer JSON
    fn initWithTokenizer(
        allocator: std.mem.Allocator,
        config_path: []const u8,
        weights_path: []const u8,
        tokenizer_path: []const u8,
        tokenizer_json: ?[]const u8,
        seed: u64,
    ) !Session {
        var timing_start_ns: i128 = std.time.nanoTimestamp();

        // Fail early with a clear error when required files are missing.
        // This keeps behavior stable across backends and tokenizers.
        std.fs.cwd().access(config_path, .{}) catch |err| switch (err) {
            error.FileNotFound => return error.FileNotFound,
            else => return err,
        };
        std.fs.cwd().access(weights_path, .{}) catch |err| switch (err) {
            error.FileNotFound => return error.FileNotFound,
            else => return err,
        };
        // Skip tokenizer path check if we have JSON in memory
        if (tokenizer_json == null) {
            std.fs.cwd().access(tokenizer_path, .{}) catch |err| switch (err) {
                error.FileNotFound => return error.FileNotFound,
                else => return err,
            };
        }

        // Start model loading in background thread
        const ModelLoaderThread = struct {
            alloc: std.mem.Allocator,
            config_path: []const u8,
            weights_path: []const u8,
            result: ?io.LoadedModel = null,
            err: ?anyerror = null,

            fn loadModel(self: *@This()) void {
                self.result = io.loadModel(self.alloc, self.config_path, self.weights_path, progress_mod.ProgressContext.NONE) catch |e| {
                    self.err = e;
                    return;
                };
            }
        };

        var loader_thread_state = ModelLoaderThread{
            .alloc = allocator,
            .config_path = config_path,
            .weights_path = weights_path,
        };

        const model_loader_thread = std.Thread.spawn(.{}, ModelLoaderThread.loadModel, .{&loader_thread_state}) catch {
            // Thread spawn failed - load synchronously instead
            const loaded_model = try allocator.create(io.LoadedModel);
            errdefer allocator.destroy(loaded_model);
            loaded_model.* = try io.loadModel(allocator, config_path, weights_path, progress_mod.ProgressContext.NONE);
            errdefer loaded_model.deinit();

            var tokenizer_state = if (tokenizer_json) |json|
                try tokenizer_mod.Tokenizer.initFromJson(allocator, json)
            else
                try tokenizer_mod.Tokenizer.initFromPath(allocator, tokenizer_path);
            errdefer tokenizer_state.deinit();

            return buildFromComponents(allocator, loaded_model, tokenizer_state, seed);
        };

        // Load tokenizer while model loads in background (~60ms of parallel work)
        var tokenizer_state = if (tokenizer_json) |json|
            try tokenizer_mod.Tokenizer.initFromJson(allocator, json)
        else
            try tokenizer_mod.Tokenizer.initFromPath(allocator, tokenizer_path);
        errdefer tokenizer_state.deinit();
        {
            const now = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(now - timing_start_ns)) / 1_000_000.0;
            log.debug("inference", "Tokenizer loaded", .{ .duration_ms = duration_ms }, @src());
            timing_start_ns = now;
        }

        // Wait for model to finish loading
        model_loader_thread.join();

        if (loader_thread_state.err) |e| return e;
        const loaded_model = try allocator.create(io.LoadedModel);
        errdefer allocator.destroy(loaded_model);
        loaded_model.* = loader_thread_state.result.?;
        errdefer loaded_model.deinit();

        {
            const now = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(now - timing_start_ns)) / 1_000_000.0;
            log.debug("inference", "Model loaded (parallel)", .{ .duration_ms = duration_ms }, @src());
        }

        // Build session (includes granular timing for sampler/backend/warmup)
        return buildFromComponents(allocator, loaded_model, tokenizer_state, seed);
    }

    pub fn deinit(self: *Session) void {
        self.backend.deinit();
        self.samp.deinit();
        self.loaded.deinit();
        self.allocator.destroy(self.loaded);
        self.tok.deinit();
        self.* = undefined;
    }

    pub fn run(self: *Session, prompt: []const u8, cfg: InferenceConfig) !InferenceState {
        return generate(self.allocator, &self.tok, &self.samp, &self.backend, prompt, cfg);
    }
};

pub fn generate(
    allocator: std.mem.Allocator,
    tokenizer: *tokenizer_mod.Tokenizer,
    sampler_state: *sampler.Sampler,
    backend: *Backend,
    prompt_text: []const u8,
    cfg: InferenceConfig,
) !InferenceState {
    const vocab_size = backend.vocabSize();
    const use_grammar = sampler_state.grammar_sampler != null;

    const prompt_data = try buildPromptTokens(
        allocator,
        tokenizer,
        prompt_text,
        cfg.bos_token_id,
        cfg.max_new_tokens,
    );
    errdefer allocator.free(prompt_data.tokens);

    if (cfg.max_new_tokens == 0) {
        return prefillOnly(allocator, backend, prompt_data.tokens, prompt_data.prompt_len, vocab_size);
    }

    const logits_buffer = try allocator.alloc(f32, vocab_size);
    defer allocator.free(logits_buffer);

    // === PREFILL PHASE ===
    const prefill_ns = try prefillPrompt(backend, prompt_data.tokens[0..prompt_data.prompt_len], logits_buffer);
    log.debug("inference", "Prefill complete", .{ .duration_ms = @as(f64, @floatFromInt(prefill_ns)) / 1_000_000.0 }, @src());

    // Sample first token (not included in prefill timing)
    var final_logits_buffer = try allocator.dupe(f32, logits_buffer);
    errdefer allocator.free(final_logits_buffer);
    const sampled_token = if (use_grammar)
        try sampler_state.sampleConstrained(final_logits_buffer, cfg.sampling, tokenizer)
    else
        try sampler_state.sample(final_logits_buffer, cfg.sampling);
    prompt_data.tokens[prompt_data.prompt_len] = @intCast(sampled_token);
    var total_token_count = prompt_data.prompt_len + 1;

    log.trace("inference", "Prefill sampled", .{ .token = sampled_token, .prompt_len = prompt_data.prompt_len }, @src());

    try acceptGrammarToken(sampler_state, tokenizer, @intCast(sampled_token));
    if (grammarIsComplete(sampler_state)) {
        const result = try allocator.realloc(prompt_data.tokens, total_token_count);
        return InferenceState{
            .tokens = result,
            .final_logits = final_logits_buffer,
            .prompt_len = prompt_data.prompt_len,
            .generated_len = 1,
            .prefill_ns = prefill_ns,
            .decode_ns = 0,
            .finish_reason = .stop_sequence,
        };
    }

    if (cfg.token_callback) |callback| {
        callback(@intCast(sampled_token), cfg.callback_data);
    }

    // Check if first token is EOS
    for (cfg.eos_token_ids) |eos_id| {
        if (sampled_token == eos_id) {
            const result = try allocator.realloc(prompt_data.tokens, total_token_count);
            return InferenceState{
                .tokens = result,
                .final_logits = final_logits_buffer,
                .prompt_len = prompt_data.prompt_len,
                .generated_len = 1,
                .prefill_ns = prefill_ns,
                .decode_ns = 0,
                .finish_reason = if (grammarCompleteOnEos(sampler_state)) .stop_sequence else .eos_token,
            };
        }
    }

    // === DECODE PHASE ===
    var generated_count: usize = 1;
    var finish_reason: FinishReason = .length; // Default to length if we exhaust max_tokens
    var decode_timer = try std.time.Timer.start();

    const remaining_token_budget = cfg.max_new_tokens - 1;
    if (remaining_token_budget > 0) {
        // Note: stop_sequences only supported with non-greedy sampling.
        // The greedy path uses backend.decodeStreaming which doesn't support stop sequences yet.
        const use_greedy = cfg.sampling.strategy == .greedy and cfg.stop_sequences.len == 0 and !use_grammar;

        if (use_greedy) {
            const generated = try backend.decodeStreaming(
                @intCast(sampled_token),
                total_token_count,
                remaining_token_budget,
                cfg.eos_token_ids,
                prompt_data.tokens[total_token_count..],
                cfg.token_callback,
                cfg.callback_data,
            );
            generated_count += generated;
            total_token_count += generated;
            // Infer finish reason: if we generated fewer tokens than requested, we hit EOS
            finish_reason = if (generated < remaining_token_budget) .eos_token else .length;
        } else {
            const decode_result = try decodeWithSampling(
                backend,
                sampler_state,
                logits_buffer,
                tokenizer,
                @intCast(sampled_token),
                total_token_count,
                remaining_token_budget,
                cfg.eos_token_ids,
                prompt_data.tokens[total_token_count..],
                cfg.token_callback,
                cfg.callback_data,
                cfg.sampling,
                cfg.stop_sequences,
            );
            generated_count += decode_result.generated_count;
            total_token_count += decode_result.generated_count;
            finish_reason = decode_result.finish_reason;
        }
    }

    const decode_ns = decode_timer.read();
    log.debug("inference", "Decode complete", .{ .duration_ms = @as(f64, @floatFromInt(decode_ns)) / 1_000_000.0 }, @src());

    if (generated_count > 1) {
        try backend.decode(prompt_data.tokens[total_token_count - 1], total_token_count - 1, logits_buffer);
        allocator.free(final_logits_buffer);
        final_logits_buffer = try allocator.dupe(f32, logits_buffer);
    }

    const result = try allocator.realloc(prompt_data.tokens, total_token_count);
    return InferenceState{
        .tokens = result,
        .final_logits = final_logits_buffer,
        .prompt_len = prompt_data.prompt_len,
        .generated_len = generated_count,
        .prefill_ns = prefill_ns,
        .decode_ns = decode_ns,
        .finish_reason = finish_reason,
    };
}

const DecodeResult = struct {
    generated_count: usize,
    finish_reason: FinishReason,
    /// Number of tokens to trim from output (stop sequence length - 1 for partial matches).
    stop_sequence_trim_len: usize = 0,
};

/// Check if the generated tokens end with any stop sequence.
/// Returns the length of the matching stop sequence, or 0 if no match.
fn checkStopSequence(output_tokens: []const u32, generated_count: usize, stop_sequences: []const []const u32) usize {
    for (stop_sequences) |stop_seq| {
        if (stop_seq.len == 0) continue;
        if (generated_count < stop_seq.len) continue;

        // Check if the last N tokens match this stop sequence
        const start_idx = generated_count - stop_seq.len;
        var matches = true;
        for (stop_seq, 0..) |token, i| {
            if (output_tokens[start_idx + i] != token) {
                matches = false;
                break;
            }
        }
        if (matches) {
            return stop_seq.len;
        }
    }
    return 0;
}

fn decodeWithSampling(
    backend: *Backend,
    sampler_state: *sampler.Sampler,
    logits_buffer: []f32,
    tokenizer: *tokenizer_mod.Tokenizer,
    initial_token: u32,
    initial_position: usize,
    max_tokens: usize,
    eos_token_ids: []const u32,
    output_tokens: []u32,
    callback: ?TokenCallback,
    callback_data: ?*anyopaque,
    sampling_config: sampler.SamplingConfig,
    stop_sequences: []const []const u32,
) !DecodeResult {
    var current_token = initial_token;
    var current_position = initial_position;
    var generated_count: usize = 0;

    while (generated_count < max_tokens) : (generated_count += 1) {
        try backend.decode(current_token, current_position, logits_buffer);
        current_position += 1;

        const next_token = if (sampler_state.grammar_sampler != null)
            try sampler_state.sampleConstrained(logits_buffer, sampling_config, tokenizer)
        else
            try sampler_state.sample(logits_buffer, sampling_config);
        current_token = @intCast(next_token);
        output_tokens[generated_count] = current_token;
        try acceptGrammarToken(sampler_state, tokenizer, current_token);
        if (grammarIsComplete(sampler_state)) {
            generated_count += 1;
            if (callback) |cb| {
                cb(current_token, callback_data);
            }
            return .{
                .generated_count = generated_count,
                .finish_reason = .stop_sequence,
            };
        }

        // Check EOS tokens first (they should be included in output)
        for (eos_token_ids) |eos_id| {
            if (current_token == eos_id) {
                generated_count += 1;
                // Call callback for EOS token
                if (callback) |cb| {
                    cb(current_token, callback_data);
                }
                return .{
                    .generated_count = generated_count,
                    .finish_reason = if (grammarCompleteOnEos(sampler_state)) .stop_sequence else .eos_token,
                };
            }
        }

        // Check stop sequences (they should NOT be included in output)
        // We check BEFORE calling callback so stop sequences aren't streamed
        const stop_len = checkStopSequence(output_tokens, generated_count + 1, stop_sequences);
        if (stop_len > 0) {
            // Don't call callback for stop sequence tokens
            // Return count minus stop sequence length (to trim it from output)
            const trimmed_count = generated_count + 1 - stop_len;
            return .{
                .generated_count = trimmed_count,
                .finish_reason = .stop_sequence,
                .stop_sequence_trim_len = stop_len,
            };
        }

        // Only call callback if not part of potential stop sequence
        if (callback) |cb| {
            cb(current_token, callback_data);
        }
    }

    return .{ .generated_count = generated_count, .finish_reason = .length };
}

fn buildPromptTokens(
    allocator: std.mem.Allocator,
    tokenizer: *tokenizer_mod.Tokenizer,
    prompt_text: []const u8,
    bos_token_id: ?u32,
    max_new_tokens: usize,
) !struct { tokens: []u32, prompt_len: usize } {
    var encode_timer = try std.time.Timer.start();
    const encoded_tokens = try tokenizer.encode(prompt_text);
    const encode_duration_ns = encode_timer.read();
    log.debug("inference", "Prompt encoded", .{
        .duration_ms = @as(f64, @floatFromInt(encode_duration_ns)) / 1_000_000.0,
        .token_count = encoded_tokens.len,
    }, @src());

    // Prepend BOS token if configured,
    // but avoid double-prepending when the prompt already starts with BOS.
    var prepend_bos = bos_token_id != null;
    if (prepend_bos and encoded_tokens.len > 0 and encoded_tokens[0] == bos_token_id.?) {
        prepend_bos = false;
    }
    const bos_token_offset: usize = if (prepend_bos) 1 else 0;
    const prompt_len = encoded_tokens.len + bos_token_offset;
    const max_token_len = prompt_len + max_new_tokens;

    var tokens = try allocator.alloc(u32, max_token_len);
    errdefer allocator.free(tokens);
    if (prepend_bos) {
        tokens[0] = bos_token_id.?;
    }
    @memcpy(tokens[bos_token_offset..prompt_len], encoded_tokens);
    tokenizer.allocator.free(encoded_tokens);

    return .{ .tokens = tokens, .prompt_len = prompt_len };
}

fn prefillPrompt(backend: *Backend, prompt_tokens: []const u32, logits_buffer: []f32) !u64 {
    var prefill_timer = try std.time.Timer.start();
    try backend.prefill(prompt_tokens, logits_buffer);
    return prefill_timer.read();
}

fn prefillOnly(
    allocator: std.mem.Allocator,
    backend: *Backend,
    tokens: []u32,
    prompt_len: usize,
    vocab_size: usize,
) !InferenceState {
    const logits_buffer = try allocator.alloc(f32, vocab_size);
    errdefer allocator.free(logits_buffer);

    const prefill_ns = try prefillPrompt(backend, tokens[0..prompt_len], logits_buffer);
    log.debug("inference", "Prefill only complete", .{ .duration_ms = @as(f64, @floatFromInt(prefill_ns)) / 1_000_000.0 }, @src());

    return InferenceState{
        .tokens = tokens,
        .final_logits = logits_buffer,
        .prompt_len = prompt_len,
        .generated_len = 0,
        .prefill_ns = prefill_ns,
        .decode_ns = 0,
        .finish_reason = .length, // Prefill-only: no generation, stopped by max_tokens=0
    };
}

fn acceptGrammarToken(
    sampler_state: *sampler.Sampler,
    tokenizer: *tokenizer_mod.Tokenizer,
    token_id: u32,
) !void {
    if (sampler_state.grammar_sampler == null) return;
    const token_text = tokenizer.tokenBytes(@intCast(token_id)) orelse return;
    try sampler_state.acceptToken(token_id, token_text);
}

fn grammarIsComplete(sampler_state: *sampler.Sampler) bool {
    const gs = sampler_state.grammar_sampler orelse return false;
    return gs.state == .complete;
}

fn grammarCompleteOnEos(sampler_state: *sampler.Sampler) bool {
    const gs = sampler_state.grammar_sampler orelse return false;
    return gs.state == .complete;
}

test "session init fails cleanly for missing files" {
    const res = Session.init(
        std.testing.allocator,
        "/no/such/config.json",
        "/no/such/model.safetensors",
        "/no/such/tokenizer",
        1,
    );
    try std.testing.expectError(error.FileNotFound, res);
}

test "FinishReason.toInt returns correct enum values" {
    try std.testing.expectEqual(@as(u8, 0), FinishReason.eos_token.toInt());
    try std.testing.expectEqual(@as(u8, 1), FinishReason.length.toInt());
    try std.testing.expectEqual(@as(u8, 2), FinishReason.stop_sequence.toInt());
}
