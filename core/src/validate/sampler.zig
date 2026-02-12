//! Constrained sampler for grammar-based decoding.

const std = @import("std");
const engine_mod = @import("engine.zig");
const mask_mod = @import("mask.zig");
const schema_mod = @import("schema.zig");
const semantic_mod = @import("semantic.zig");
const cache = @import("cache.zig");

const Engine = engine_mod.Engine;
const SemanticValidator = semantic_mod.SemanticValidator;
pub const SemanticViolation = semantic_mod.SemanticViolation;

pub const GrammarConfig = struct {
    allow_thinking: bool = false,
    max_thinking_tokens: usize = 512,
    start_marker: ?[]const u8 = null,
    soft_limit_ratio: f32 = 0.9,
    soft_limit_bias: f32 = -2.0,
    /// Enable anti-repetition bias for array elements.
    /// When consecutive identical array elements are detected, a negative bias
    /// is applied to discourage further repetition and favor array closing.
    /// DISABLED: The bias can interfere with grammar constraints and cause invalid JSON.
    enable_array_repetition_bias: bool = false,
    /// Base bias applied when array element repetition is detected.
    /// Negative values discourage repetition. The actual bias is multiplied
    /// by (consecutive_count - threshold + 1) for escalating penalty.
    array_repetition_bias: f32 = -5.0,
    /// Bias applied to closing bracket when repetition is detected.
    /// Gets stronger with more consecutive repetitions.
    array_close_bias: f32 = 4.0,
    /// Number of consecutive identical elements that triggers the bias.
    /// Set to 1 to trigger on first repeat.
    array_repetition_threshold: usize = 1,
};

const GrammarState = enum {
    waiting_for_trigger,
    panic_forcing_close,
    active,
    complete,
};

const scan_buffer_max = 64;
const think_open_tag = "<think>";
const think_close_tag = "</think>";
const think_tag_max = if (think_open_tag.len > think_close_tag.len) think_open_tag.len else think_close_tag.len;

pub const ConstrainedSampler = struct {
    allocator: std.mem.Allocator,
    engine: Engine,
    state: GrammarState,
    config: GrammarConfig,
    generated_text: std.ArrayList(u8),
    stop_tokens: std.ArrayList(u32),
    waiting_tokens: usize,
    panic_close_sequence: []const u8 = "</think>",
    panic_close_position: usize = 0,
    prefix_advanced: bool = false,
    think_depth: usize = 0,
    json_start_found: bool = false,
    saw_think_close: bool = false,
    scan_buffer: std.ArrayList(u8),
    prefix_token_ids: ?[]u32 = null,
    prefix_pending: bool = false,
    eager_precomputed: bool = false,
    /// Track consecutive identical array elements for repetition detection.
    /// Stores hash of each array element to detect repetition patterns.
    array_element_hashes: std.ArrayList(u64) = .empty,
    /// Current repetition count of identical consecutive elements.
    consecutive_identical_count: usize = 0,
    /// Track if we're currently inside an array (bracket depth).
    array_depth: usize = 0,
    /// Position of the last comma in generated_text (marks element boundaries).
    last_comma_pos: ?usize = null,
    /// Semantic validator for post-generation validation of constraints
    /// that grammar cannot express (required fields, number ranges, etc.).
    semantic_validator: ?SemanticValidator = null,
    /// Flag indicating panic mode was triggered (thinking tokens exhausted).
    /// When true, skip semantic validation since output was truncated.
    panic_mode_triggered: bool = false,

    pub fn init(
        allocator: std.mem.Allocator,
        schema_json: []const u8,
        config: GrammarConfig,
        stop_token_ids: []const u32,
        prefix_tokens: ?[]const []const u8,
        prefix_token_ids: ?[]const u32,
    ) !ConstrainedSampler {
        const grammar_cache = cache.getGlobalCache(allocator);
        const grammar = try grammar_cache.getOrCompile(schema_json, .{});

        var engine = try Engine.init(allocator, grammar);

        // Initialize semantic validator for post-generation validation.
        // This validates constraints grammar cannot express (required fields, etc.).
        var semantic_validator = SemanticValidator.init(allocator, schema_json) catch null;
        errdefer if (semantic_validator) |*sv| sv.deinit();

        var stop_tokens = std.ArrayList(u32).empty;
        for (stop_token_ids) |id| {
            try stop_tokens.append(allocator, id);
        }

        const initial_state: GrammarState = if (config.allow_thinking or
            config.start_marker != null)
            .waiting_for_trigger
        else
            .active;

        var prefix_advanced = false;
        var prefix_ids_owned: ?[]u32 = null;
        var prefix_pending = false;
        if (!config.allow_thinking) {
            if (prefix_tokens) |tokens| {
                prefix_advanced = try fastForwardGrammar(&engine, tokens);
            } else if (prefix_token_ids) |ids| {
                if (ids.len > 0) {
                    prefix_ids_owned = try allocator.dupe(u32, ids);
                    prefix_pending = true;
                }
            }
        }

        return .{
            .allocator = allocator,
            .engine = engine,
            .state = initial_state,
            .config = config,
            .generated_text = .empty,
            .stop_tokens = stop_tokens,
            .waiting_tokens = 0,
            .prefix_advanced = prefix_advanced,
            .think_depth = 0,
            .json_start_found = false,
            .saw_think_close = false,
            .scan_buffer = .empty,
            .prefix_token_ids = prefix_ids_owned,
            .prefix_pending = prefix_pending,
            .eager_precomputed = false,
            .semantic_validator = semantic_validator,
        };
    }

    fn fastForwardGrammar(engine: *Engine, prefix_tokens: []const []const u8) !bool {
        var snapshot = try engine.snapshot();
        var snapshot_owned = true;
        defer if (snapshot_owned) snapshot.deinit();

        for (prefix_tokens) |token_text| {
            if (!try engine.canAccept(token_text)) {
                engine.restore(&snapshot);
                snapshot_owned = false;
                return false;
            }
            try engine.advance(token_text);
        }

        return true;
    }

    fn applyPrefixTokens(self: *ConstrainedSampler, tokenizer: anytype) !bool {
        const ids = self.prefix_token_ids orelse {
            self.prefix_pending = false;
            return false;
        };

        var token_texts = std.ArrayList([]const u8).empty;
        defer token_texts.deinit(self.allocator);

        for (ids) |id| {
            const token_text = tokenizerTokenTextById(tokenizer, id) orelse {
                self.prefix_pending = false;
                self.prefix_advanced = false;
                self.freePrefixTokenIds();
                return false;
            };
            try token_texts.append(self.allocator, token_text);
        }

        const ok = try fastForwardGrammar(&self.engine, token_texts.items);
        self.prefix_pending = false;
        self.prefix_advanced = ok;
        self.freePrefixTokenIds();
        return ok;
    }

    fn freePrefixTokenIds(self: *ConstrainedSampler) void {
        if (self.prefix_token_ids) |ids| {
            if (ids.len > 0) {
                self.allocator.free(ids);
            }
            self.prefix_token_ids = null;
        }
    }

    pub fn deinit(self: *ConstrainedSampler) void {
        self.engine.deinit();
        self.generated_text.deinit(self.allocator);
        self.stop_tokens.deinit(self.allocator);
        self.scan_buffer.deinit(self.allocator);
        self.array_element_hashes.deinit(self.allocator);
        self.freePrefixTokenIds();
        if (self.semantic_validator) |*sv| sv.deinit();
        self.* = undefined;
    }

    fn shouldActivate(self: *ConstrainedSampler) bool {
        if (self.state != .waiting_for_trigger) {
            return self.state == .active;
        }

        const text = self.generated_text.items;

        if (self.config.start_marker) |marker| {
            if (text.len >= marker.len and std.mem.endsWith(u8, text, marker)) {
                self.state = .active;
                return true;
            }
        }

        if (self.config.allow_thinking) {
            if (self.saw_think_close) {
                self.state = .active;
                return true;
            }

            if (self.json_start_found and self.think_depth == 0) {
                self.state = .active;
                return true;
            }
        }

        if (self.waiting_tokens >= self.config.max_thinking_tokens) {
            self.state = .panic_forcing_close;
            self.panic_mode_triggered = true;
            return false;
        }

        return false;
    }

    fn updateScanState(self: *ConstrainedSampler, token_text: []const u8) !void {
        const old_len = self.scan_buffer.items.len;
        try self.scan_buffer.appendSlice(self.allocator, token_text);

        var new_len = self.scan_buffer.items.len;
        var drop: usize = 0;
        if (new_len > scan_buffer_max) {
            drop = new_len - scan_buffer_max;
            std.mem.copyForwards(
                u8,
                self.scan_buffer.items[0 .. new_len - drop],
                self.scan_buffer.items[drop..new_len],
            );
            self.scan_buffer.items.len = new_len - drop;
            new_len = self.scan_buffer.items.len;
        }

        const old_len_post = if (old_len > drop) old_len - drop else 0;
        const overlap = if (old_len_post > think_tag_max - 1) think_tag_max - 1 else old_len_post;
        const start = old_len_post - overlap;
        const scan = self.scan_buffer.items[start..new_len];

        var i: usize = 0;
        while (i < scan.len) {
            const abs_index = start + i;

            if (i + think_open_tag.len <= scan.len and
                std.mem.eql(u8, scan[i..][0..think_open_tag.len], think_open_tag))
            {
                if (abs_index + think_open_tag.len > old_len_post) {
                    self.think_depth += 1;
                }
                i += think_open_tag.len;
                continue;
            }

            if (i + think_close_tag.len <= scan.len and
                std.mem.eql(u8, scan[i..][0..think_close_tag.len], think_close_tag))
            {
                if (abs_index + think_close_tag.len > old_len_post) {
                    if (self.think_depth > 0) {
                        self.think_depth -= 1;
                    }
                    self.saw_think_close = true;
                }
                i += think_close_tag.len;
                continue;
            }

            if (abs_index >= old_len_post and self.think_depth == 0) {
                const c = scan[i];
                if (c == '{' or c == '[') {
                    self.json_start_found = true;
                }
            }

            i += 1;
        }
    }

    fn isJsonStartAtLineBeginning(self: *ConstrainedSampler) bool {
        const text = self.generated_text.items;
        if (text.len == 0) return false;

        const last = text[text.len - 1];
        if (last != '{' and last != '[') return false;

        if (text.len == 1) return true;
        const prev = text[text.len - 2];
        return prev == '\n' or prev == '\r';
    }

    pub fn applyConstraints(
        self: *ConstrainedSampler,
        logits: []f32,
        tokenizer: anytype,
    ) !void {
        // Eagerly precompute token masks on first call
        if (!self.eager_precomputed) {
            try self.engine.eagerPrecompute(tokenizer);
            self.eager_precomputed = true;
        }
        if (self.prefix_pending and !self.prefix_advanced and !self.config.allow_thinking) {
            _ = try self.applyPrefixTokens(tokenizer);
        }
        switch (self.state) {
            .waiting_for_trigger => {
                self.waiting_tokens += 1;

                const soft_limit = @as(usize, @intFromFloat(
                    @as(f32, @floatFromInt(self.config.max_thinking_tokens)) *
                        self.config.soft_limit_ratio,
                ));

                if (self.waiting_tokens > soft_limit and
                    self.waiting_tokens <= self.config.max_thinking_tokens)
                {
                    self.applySoftLimitBias(logits, tokenizer);
                }

                _ = self.shouldActivate();
            },
            .panic_forcing_close => {
                if (self.panic_close_position >= self.panic_close_sequence.len) {
                    self.state = .active;
                    var valid_mask = try self.engine.getValidTokens(tokenizer);
                    defer valid_mask.deinit();
                    mask_mod.applyMask(logits, &valid_mask);
                    return;
                }

                const remaining = self.panic_close_sequence[self.panic_close_position..];
                var best_token: ?u32 = null;
                var best_match_len: usize = 0;

                const vocab_size = tokenizerVocabSize(tokenizer);
                var token_id: usize = 0;
                while (token_id < vocab_size) : (token_id += 1) {
                    const token_text = tokenizerTokenText(tokenizer, token_id) orelse continue;
                    if (token_text.len == 0) continue;

                    const match_len = commonPrefixLen(token_text, remaining);
                    if (match_len > 0 and match_len == token_text.len and match_len > best_match_len) {
                        best_token = @intCast(token_id);
                        best_match_len = match_len;
                    }
                }

                if (best_token) |token| {
                    for (logits) |*logit| {
                        logit.* = -std.math.inf(f32);
                    }
                    if (token < logits.len) {
                        logits[token] = 0;
                    }
                }
            },
            .active => {
                // Jump Forward optimization: if grammar has a deterministic continuation,
                // skip mask computation and force-select the best matching token.
                //
                // NOTE: This only skips mask computation, not model inference.
                // TODO(perf): The real optimization needs to happen at the scheduler/generation lint:ignore no-todo
                // loop level to skip model inference entirely for deterministic tokens.
                // (architectural change beyond current scope)
                // This is a bigger architectural change that would provide significant
                // performance gains for structured output generation.
                if (self.engine.getDeterministicContinuation()) |literal| {
                    if (self.tryJumpForward(logits, tokenizer, literal)) {
                        return;
                    }
                }

                // Full mask computation
                var valid_mask = try self.engine.getValidTokens(tokenizer);
                defer valid_mask.deinit();

                mask_mod.applyMask(logits, &valid_mask);

                // Apply anti-repetition bias for array elements if enabled
                if (self.config.enable_array_repetition_bias and
                    self.consecutive_identical_count >= self.config.array_repetition_threshold and
                    self.array_depth > 0)
                {
                    self.applyArrayRepetitionBias(logits, tokenizer);
                }

                if (self.engine.isComplete()) {
                    self.state = .complete;
                }
            },
            .complete => {
                for (logits) |*logit| {
                    logit.* = -std.math.inf(f32);
                }
                for (self.stop_tokens.items) |stop_id| {
                    if (stop_id < logits.len) {
                        logits[stop_id] = 0;
                    }
                }
            },
        }
    }

    pub fn acceptToken(
        self: *ConstrainedSampler,
        token_id: u32,
        token_text: []const u8,
    ) !void {
        // Track array element boundaries and repetition before appending text
        try self.updateArrayRepetitionState(token_text);

        try self.generated_text.appendSlice(self.allocator, token_text);
        try self.updateScanState(token_text);

        if (self.state == .panic_forcing_close) {
            if (self.panic_close_position < self.panic_close_sequence.len) {
                const remaining = self.panic_close_sequence[self.panic_close_position..];
                if (token_text.len <= remaining.len and
                    std.mem.eql(u8, token_text, remaining[0..token_text.len]))
                {
                    self.panic_close_position += token_text.len;
                    if (self.panic_close_position >= self.panic_close_sequence.len) {
                        self.state = .active;
                    }
                }
            }
            return;
        }

        _ = self.shouldActivate();

        if (self.state == .active) {
            try self.engine.advance(token_text);
            if (self.engine.isComplete()) {
                self.state = .complete;
            }
        }

        _ = token_id;
    }

    /// Track array element boundaries and detect repetition patterns.
    /// This scans the token text for array-related characters and tracks element hashes.
    fn updateArrayRepetitionState(self: *ConstrainedSampler, token_text: []const u8) !void {
        for (token_text) |c| {
            switch (c) {
                '[' => {
                    self.array_depth += 1;
                    // Starting a new array - reset element tracking
                    self.array_element_hashes.clearRetainingCapacity();
                    self.consecutive_identical_count = 0;
                    self.last_comma_pos = null;
                },
                ']' => {
                    if (self.array_depth > 0) {
                        // Closing array - finalize current element if any
                        try self.finalizeArrayElement();
                        self.array_depth -= 1;
                        if (self.array_depth == 0) {
                            self.array_element_hashes.clearRetainingCapacity();
                            self.consecutive_identical_count = 0;
                            self.last_comma_pos = null;
                        }
                    }
                },
                ',' => {
                    if (self.array_depth > 0) {
                        // Element boundary - finalize current element
                        try self.finalizeArrayElement();
                        // Mark start of next element
                        self.last_comma_pos = self.generated_text.items.len + token_text.len;
                    }
                },
                else => {},
            }
        }
    }

    /// Finalize the current array element by computing its hash and checking for repetition.
    fn finalizeArrayElement(self: *ConstrainedSampler) !void {
        // Extract current element from generated_text
        const start = self.last_comma_pos orelse self.findArrayStart();
        if (start == null) return;

        const text = self.generated_text.items;
        if (start.? >= text.len) return;

        const element = text[start.?..];
        // Skip whitespace
        const trimmed = std.mem.trim(u8, element, " \t\n\r");
        if (trimmed.len == 0) return;

        // Compute hash of the element
        const hash = std.hash.Wyhash.hash(0, trimmed);

        // Check if this matches the previous element
        if (self.array_element_hashes.items.len > 0) {
            const last_hash = self.array_element_hashes.items[self.array_element_hashes.items.len - 1];
            if (hash == last_hash) {
                self.consecutive_identical_count += 1;
            } else {
                self.consecutive_identical_count = 1;
            }
        } else {
            self.consecutive_identical_count = 1;
        }

        try self.array_element_hashes.append(self.allocator, hash);
    }

    /// Find the start of the current array (position after '[').
    fn findArrayStart(self: *ConstrainedSampler) ?usize {
        const text = self.generated_text.items;
        // Search backwards for '[' that starts current array
        var depth: usize = 0;
        var i = text.len;
        while (i > 0) {
            i -= 1;
            switch (text[i]) {
                ']' => depth += 1,
                '[' => {
                    if (depth == 0) {
                        return i + 1; // Position after '['
                    }
                    depth -= 1;
                },
                else => {},
            }
        }
        return null;
    }

    /// Apply bias to discourage array element repetition.
    /// Boosts probability of ']' and reduces probability of content tokens.
    /// Bias escalates with repetition count for stronger intervention.
    fn applyArrayRepetitionBias(
        self: *ConstrainedSampler,
        logits: []f32,
        tokenizer: anytype,
    ) void {
        // Calculate escalation factor: bias gets stronger with more repetitions
        const excess_reps = self.consecutive_identical_count -| self.config.array_repetition_threshold;
        const escalation: f32 = @floatFromInt(@min(excess_reps + 1, 5)); // Cap at 5x

        const vocab_size = tokenizerVocabSize(tokenizer);
        var token_id: usize = 0;
        while (token_id < vocab_size) : (token_id += 1) {
            const token_text = tokenizerTokenText(tokenizer, token_id) orelse continue;
            if (token_text.len == 0) continue;

            const first_char = token_text[0];
            if (first_char == ']') {
                // Boost closing bracket probability - escalates with repetition
                logits[token_id] += self.config.array_close_bias * escalation;
            } else if (first_char != ',' and first_char != ' ' and first_char != '\n' and first_char != '\t') {
                // Apply negative bias to content tokens (but not structural ones)
                // Escalating penalty to break stronger repetition patterns
                logits[token_id] += self.config.array_repetition_bias * escalation;
            }
        }
    }

    /// Try to use Jump Forward optimization for a deterministic literal sequence.
    /// Returns true if we successfully forced a token, false to fall back to full mask.
    /// Uses prefix index for O(1) lookup instead of O(vocab_size) scanning.
    fn tryJumpForward(
        _: *ConstrainedSampler,
        logits: []f32,
        tokenizer: anytype,
        literal: []const u8,
    ) bool {
        if (literal.len == 0) return false;

        // Use prefix index for fast lookup - only search tokens starting with first byte
        if (tokenizerHasPrefixIndex(tokenizer)) {
            const first_byte = literal[0];
            const candidates = tokenizerTokensStartingWith(tokenizer, first_byte) orelse return false;

            var best_token: ?u32 = null;
            var best_match_len: usize = 0;

            for (candidates) |token_id| {
                const token_text = tokenizerTokenText(tokenizer, @intCast(token_id)) orelse continue;
                if (token_text.len == 0) continue;

                // Token must be an exact prefix of the literal
                if (token_text.len <= literal.len and
                    std.mem.eql(u8, token_text, literal[0..token_text.len]))
                {
                    if (token_text.len > best_match_len) {
                        best_token = token_id;
                        best_match_len = token_text.len;
                    }
                }
            }

            if (best_token) |token| {
                // Found a matching token - force select it
                for (logits) |*logit| {
                    logit.* = -std.math.inf(f32);
                }
                if (token < logits.len) {
                    logits[token] = 0;
                }
                return true;
            }
        }

        // No prefix index or no match found - fall back to mask computation
        return false;
    }

    fn tokenizerHasPrefixIndex(tokenizer: anytype) bool {
        const T = @TypeOf(tokenizer);
        switch (@typeInfo(T)) {
            .pointer => |info| {
                const Child = info.child;
                return @hasDecl(Child, "getTokensStartingWith");
            },
            else => {},
        }
        return @hasDecl(T, "getTokensStartingWith");
    }

    fn tokenizerTokensStartingWith(tokenizer: anytype, byte: u8) ?[]const u32 {
        const T = @TypeOf(tokenizer);
        switch (@typeInfo(T)) {
            .pointer => |info| {
                const Child = info.child;
                if (@hasDecl(Child, "getTokensStartingWith")) {
                    return tokenizer.getTokensStartingWith(byte);
                }
            },
            else => {},
        }
        if (@hasDecl(T, "getTokensStartingWith")) {
            return tokenizer.getTokensStartingWith(byte);
        }
        return null;
    }

    fn applySoftLimitBias(
        self: *ConstrainedSampler,
        logits: []f32,
        tokenizer: anytype,
    ) void {
        const closing_chars = "</think>{[";

        const vocab_size = tokenizerVocabSize(tokenizer);
        var token_id: usize = 0;
        while (token_id < vocab_size) : (token_id += 1) {
            const token_text = tokenizerTokenText(tokenizer, token_id) orelse continue;
            if (token_text.len == 0) continue;

            var is_closing = false;
            for (closing_chars) |c| {
                if (token_text[0] == c) {
                    is_closing = true;
                    break;
                }
            }

            if (!is_closing) {
                logits[token_id] += self.config.soft_limit_bias;
            }
        }
    }

    /// Validates the final generated output against semantic constraints.
    ///
    /// Call this after generation completes (state == .complete) to check constraints
    /// that grammar cannot express: required fields, number ranges, additionalProperties.
    ///
    /// Returns null if valid, or a SemanticViolation describing the first error found.
    /// The violation includes path, message, and constraint type for error reporting.
    ///
    /// Note: Skips validation if generation was truncated (state != .complete)
    /// or panic mode was triggered (thinking exhausted), since incomplete JSON
    /// cannot satisfy semantic constraints.
    pub fn validateFinal(self: *ConstrainedSampler) ?SemanticViolation {
        // Skip validation for incomplete generation (e.g., max_tokens reached)
        if (self.state != .complete) return null;
        // Skip validation if panic mode was triggered (thinking tokens exhausted)
        // - the model was forced to close early and may not have completed JSON
        if (self.panic_mode_triggered) return null;

        const validator = &(self.semantic_validator orelse return null);
        const json_text = self.extractJsonText() orelse return null;

        return validator.validate(json_text) catch null;
    }

    /// Extracts the JSON portion from generated text, stripping thinking blocks.
    fn extractJsonText(self: *ConstrainedSampler) ?[]const u8 {
        const text = self.generated_text.items;
        if (text.len == 0) return null;

        // Find the JSON content (after </think> if present)
        var start: usize = 0;
        if (std.mem.indexOf(u8, text, think_close_tag)) |close_pos| {
            start = close_pos + think_close_tag.len;
        }

        // Find the start of JSON (first { or [)
        while (start < text.len) {
            if (text[start] == '{' or text[start] == '[') break;
            start += 1;
        }
        if (start >= text.len) return null;

        return text[start..];
    }

    /// Returns true if generation is complete.
    pub fn isComplete(self: *const ConstrainedSampler) bool {
        return self.state == .complete;
    }

    pub fn reset(self: *ConstrainedSampler) void {
        self.engine.reset() catch {};
        self.generated_text.clearRetainingCapacity();
        self.scan_buffer.clearRetainingCapacity();
        self.array_element_hashes.clearRetainingCapacity();
        self.state = if (self.config.allow_thinking or
            self.config.start_marker != null)
            .waiting_for_trigger
        else
            .active;
        self.waiting_tokens = 0;
        self.panic_close_position = 0;
        self.panic_mode_triggered = false;
        self.think_depth = 0;
        self.json_start_found = false;
        self.saw_think_close = false;
        self.prefix_advanced = false;
        self.prefix_pending = self.prefix_token_ids != null;
        self.consecutive_identical_count = 0;
        self.array_depth = 0;
        self.last_comma_pos = null;
    }
};

fn tokenizerTokenTextById(tokenizer: anytype, token_id: u32) ?[]const u8 {
    const id_i32 = std.math.cast(i32, token_id) orelse return null;
    const T = @TypeOf(tokenizer);

    switch (@typeInfo(T)) {
        .pointer => |info| {
            const Child = info.child;
            if (@hasDecl(Child, "tokenBytes")) {
                return tokenizer.tokenBytes(@intCast(token_id));
            }
            if (@hasDecl(Child, "idToToken")) {
                return tokenizer.idToToken(id_i32);
            }
            if (@hasField(Child, "tokenizer_handle")) {
                return tokenizer.tokenizer_handle.idToToken(id_i32);
            }
        },
        else => {},
    }

    if (@hasDecl(T, "tokenBytes")) {
        return tokenizer.tokenBytes(@intCast(token_id));
    }
    if (@hasDecl(T, "idToToken")) {
        return tokenizer.idToToken(id_i32);
    }
    if (@hasField(T, "tokenizer_handle")) {
        return tokenizer.tokenizer_handle.idToToken(id_i32);
    }

    return null;
}

fn tokenizerVocabSize(tokenizer: anytype) usize {
    const T = @TypeOf(tokenizer);
    switch (@typeInfo(T)) {
        .pointer => |info| {
            const Child = info.child;
            if (@hasDecl(Child, "getVocabSize")) {
                return tokenizer.getVocabSize();
            }
            if (@hasField(Child, "tokenizer_handle")) {
                return tokenizer.tokenizer_handle.getVocabSize();
            }
            if (@hasField(Child, "vocab_size")) {
                return tokenizer.vocab_size;
            }
        },
        else => {},
    }

    if (@hasDecl(T, "getVocabSize")) {
        return tokenizer.getVocabSize();
    }
    if (@hasField(T, "tokenizer_handle")) {
        return tokenizer.tokenizer_handle.getVocabSize();
    }
    if (@hasField(T, "vocab_size")) {
        return tokenizer.vocab_size;
    }

    return 0;
}

fn tokenizerTokenText(tokenizer: anytype, token_id: usize) ?[]const u8 {
    const id_i32 = std.math.cast(i32, token_id) orelse return null;
    const T = @TypeOf(tokenizer);

    switch (@typeInfo(T)) {
        .pointer => |info| {
            const Child = info.child;
            if (@hasDecl(Child, "tokenBytes")) {
                return tokenizer.tokenBytes(token_id);
            }
            if (@hasDecl(Child, "idToToken")) {
                return tokenizer.idToToken(id_i32);
            }
            if (@hasField(Child, "tokenizer_handle")) {
                return tokenizer.tokenizer_handle.idToToken(id_i32);
            }
        },
        else => {},
    }

    if (@hasDecl(T, "tokenBytes")) {
        return tokenizer.tokenBytes(token_id);
    }
    if (@hasDecl(T, "idToToken")) {
        return tokenizer.idToToken(id_i32);
    }
    if (@hasField(T, "tokenizer_handle")) {
        return tokenizer.tokenizer_handle.idToToken(id_i32);
    }

    return null;
}

fn commonPrefixLen(a: []const u8, b: []const u8) usize {
    const max_len = @min(a.len, b.len);
    var i: usize = 0;
    while (i < max_len and a[i] == b[i]) : (i += 1) {}
    return i;
}

test "sampler enforces stop tokens on completion" {
    const allocator = std.testing.allocator;
    const schema = "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"}}}";

    const stop_tokens = [_]u32{ 1, 3 };
    var sampler = try ConstrainedSampler.init(allocator, schema, .{}, &stop_tokens, null, null);
    defer sampler.deinit();

    sampler.state = .complete;

    var logits = [_]f32{ 0, 0, 0, 0, 0 };
    const DummyTok = struct {
        pub fn getVocabSize(_: @This()) usize {
            return 5;
        }
        pub fn idToToken(_: @This(), _: i32) ?[]const u8 {
            return null;
        }
    };
    try sampler.applyConstraints(&logits, DummyTok{});

    try std.testing.expectEqual(@as(f32, 0), logits[1]);
    try std.testing.expectEqual(@as(f32, 0), logits[3]);
    try std.testing.expect(std.math.isInf(logits[0]));
    try std.testing.expect(std.math.isInf(logits[2]));
}

test "sampler soft limit bias favors closing tokens" {
    const allocator = std.testing.allocator;
    const schema = "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"}}}";

    const vocab = [_][]const u8{ "<", "a", "{", "x" };
    const MockTokenizer = struct {
        vocab: []const []const u8,
        pub fn getVocabSize(self: @This()) usize {
            return self.vocab.len;
        }
        pub fn idToToken(self: @This(), id: i32) ?[]const u8 {
            if (id < 0) return null;
            const idx: usize = @intCast(id);
            if (idx >= self.vocab.len) return null;
            return self.vocab[idx];
        }
    };

    var sampler = try ConstrainedSampler.init(allocator, schema, .{
        .allow_thinking = true,
        .max_thinking_tokens = 4,
        .soft_limit_ratio = 0.5,
        .soft_limit_bias = -2.0,
    }, &[_]u32{}, null, null);
    defer sampler.deinit();

    sampler.state = .waiting_for_trigger;
    sampler.waiting_tokens = 2;

    var logits = [_]f32{ 0, 0, 0, 0 };
    try sampler.applyConstraints(&logits, MockTokenizer{ .vocab = &vocab });

    try std.testing.expectEqual(@as(f32, 0), logits[0]);
    try std.testing.expectEqual(@as(f32, 0), logits[2]);
    try std.testing.expectEqual(@as(f32, -2.0), logits[1]);
}

test "sampler panic mode forces close sequence" {
    const allocator = std.testing.allocator;
    const schema = "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"}}}";

    const vocab = [_][]const u8{ "END", "a" };
    const MockTokenizer = struct {
        vocab: []const []const u8,
        pub fn getVocabSize(self: @This()) usize {
            return self.vocab.len;
        }
        pub fn idToToken(self: @This(), id: i32) ?[]const u8 {
            if (id < 0) return null;
            const idx: usize = @intCast(id);
            if (idx >= self.vocab.len) return null;
            return self.vocab[idx];
        }
    };

    var sampler = try ConstrainedSampler.init(allocator, schema, .{
        .allow_thinking = true,
    }, &[_]u32{}, null, null);
    defer sampler.deinit();

    sampler.state = .panic_forcing_close;
    sampler.panic_close_sequence = "END";
    sampler.panic_close_position = 0;

    var logits = [_]f32{ 0, 0 };
    try sampler.applyConstraints(&logits, MockTokenizer{ .vocab = &vocab });

    try std.testing.expectEqual(@as(f32, 0), logits[0]);
    try std.testing.expect(std.math.isInf(logits[1]));
}

test "sampler prefill fast-forward advances grammar" {
    const allocator = std.testing.allocator;
    const schema = "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"}}}";

    const prefix = [_][]const u8{ "{", "\"name\"", ":" };
    var sampler = try ConstrainedSampler.init(
        allocator,
        schema,
        .{},
        &[_]u32{},
        @as(?[]const []const u8, prefix[0..]),
        null,
    );
    defer sampler.deinit();

    try std.testing.expect(sampler.prefix_advanced);
    try std.testing.expect(try sampler.engine.canAccept("\""));
}

// =============================================================================
// Array repetition detection tests
// =============================================================================

test "array repetition detection tracks array depth" {
    const allocator = std.testing.allocator;
    const schema = "{\"type\":\"array\",\"items\":{\"type\":\"string\"}}";

    var sampler = try ConstrainedSampler.init(allocator, schema, .{}, &[_]u32{}, null, null);
    defer sampler.deinit();

    // Initially no array depth
    try std.testing.expectEqual(@as(usize, 0), sampler.array_depth);

    // Accept opening bracket
    try sampler.acceptToken(0, "[");
    try std.testing.expectEqual(@as(usize, 1), sampler.array_depth);

    // Accept closing bracket
    try sampler.acceptToken(0, "]");
    try std.testing.expectEqual(@as(usize, 0), sampler.array_depth);
}

test "array repetition detection tracks element hashes" {
    const allocator = std.testing.allocator;
    const schema = "{\"type\":\"array\",\"items\":{\"type\":\"string\"}}";

    var sampler = try ConstrainedSampler.init(allocator, schema, .{}, &[_]u32{}, null, null);
    defer sampler.deinit();

    // Simulate array with identical elements: ["foo", "foo", "foo"]
    try sampler.acceptToken(0, "[");
    try sampler.acceptToken(0, "\"foo\"");
    try sampler.acceptToken(0, ",");

    // After first comma, should have 1 element tracked
    try std.testing.expectEqual(@as(usize, 1), sampler.array_element_hashes.items.len);
    try std.testing.expectEqual(@as(usize, 1), sampler.consecutive_identical_count);

    try sampler.acceptToken(0, "\"foo\"");
    try sampler.acceptToken(0, ",");

    // After second comma with identical element, should detect repetition
    try std.testing.expectEqual(@as(usize, 2), sampler.array_element_hashes.items.len);
    try std.testing.expectEqual(@as(usize, 2), sampler.consecutive_identical_count);

    try sampler.acceptToken(0, "\"foo\"");
    try sampler.acceptToken(0, "]");

    // Closing bracket should finalize element and then reset on array close
    try std.testing.expectEqual(@as(usize, 0), sampler.array_depth);
}

test "array repetition detection different elements reset count" {
    const allocator = std.testing.allocator;
    const schema = "{\"type\":\"array\",\"items\":{\"type\":\"string\"}}";

    var sampler = try ConstrainedSampler.init(allocator, schema, .{}, &[_]u32{}, null, null);
    defer sampler.deinit();

    // Simulate array with different elements: ["foo", "bar", "baz"]
    try sampler.acceptToken(0, "[");
    try sampler.acceptToken(0, "\"foo\"");
    try sampler.acceptToken(0, ",");

    try std.testing.expectEqual(@as(usize, 1), sampler.consecutive_identical_count);

    try sampler.acceptToken(0, "\"bar\"");
    try sampler.acceptToken(0, ",");

    // Different element should reset consecutive count to 1
    try std.testing.expectEqual(@as(usize, 1), sampler.consecutive_identical_count);

    try sampler.acceptToken(0, "\"baz\"");
    try sampler.acceptToken(0, "]");

    try std.testing.expectEqual(@as(usize, 0), sampler.array_depth);
}

test "array repetition bias applied when threshold exceeded" {
    const allocator = std.testing.allocator;
    const schema = "{\"type\":\"array\",\"items\":{\"type\":\"string\"}}";

    const vocab = [_][]const u8{ "]", "\"", "foo", "," };
    const MockTokenizer = struct {
        vocab: []const []const u8,
        pub fn getVocabSize(self: @This()) usize {
            return self.vocab.len;
        }
        pub fn idToToken(self: @This(), id: i32) ?[]const u8 {
            if (id < 0) return null;
            const idx: usize = @intCast(id);
            if (idx >= self.vocab.len) return null;
            return self.vocab[idx];
        }
    };

    var sampler = try ConstrainedSampler.init(allocator, schema, .{
        .enable_array_repetition_bias = true,
        .array_repetition_threshold = 2,
        .array_repetition_bias = -3.0,
        .array_close_bias = 2.0,
    }, &[_]u32{}, null, null);
    defer sampler.deinit();

    // Set up state as if we're in an array with repetition detected
    sampler.state = .active;
    sampler.array_depth = 1;
    sampler.consecutive_identical_count = 3; // Above threshold

    var logits = [_]f32{ 0, 0, 0, 0 };
    sampler.applyArrayRepetitionBias(&logits, MockTokenizer{ .vocab = &vocab });

    // With threshold=2, count=3: excess_reps=1, escalation=2
    // ']' should be boosted (2.0 * 2 = 4.0)
    try std.testing.expectEqual(@as(f32, 4.0), logits[0]);

    // Content tokens should be penalized (-3.0 * 2 = -6.0)
    try std.testing.expectEqual(@as(f32, -6.0), logits[1]); // '"'
    try std.testing.expectEqual(@as(f32, -6.0), logits[2]); // 'foo'

    // ',' should NOT be penalized (structural token)
    try std.testing.expectEqual(@as(f32, 0.0), logits[3]);
}

test "array repetition reset clears state" {
    const allocator = std.testing.allocator;
    const schema = "{\"type\":\"array\",\"items\":{\"type\":\"string\"}}";

    var sampler = try ConstrainedSampler.init(allocator, schema, .{}, &[_]u32{}, null, null);
    defer sampler.deinit();

    // Build up some state
    try sampler.acceptToken(0, "[");
    try sampler.acceptToken(0, "\"foo\"");
    try sampler.acceptToken(0, ",");
    try sampler.acceptToken(0, "\"foo\"");
    try sampler.acceptToken(0, ",");

    try std.testing.expectEqual(@as(usize, 1), sampler.array_depth);
    try std.testing.expect(sampler.consecutive_identical_count >= 2);

    // Reset should clear all state
    sampler.reset();

    try std.testing.expectEqual(@as(usize, 0), sampler.array_depth);
    try std.testing.expectEqual(@as(usize, 0), sampler.consecutive_identical_count);
    try std.testing.expectEqual(@as(usize, 0), sampler.array_element_hashes.items.len);
    try std.testing.expectEqual(@as(?usize, null), sampler.last_comma_pos);
}

// BUG DOCUMENTATION TEST: Array repetition bias is disabled by default because
// it can interfere with grammar constraints. This test documents the issue for
// future implementation attempts.
//
// The problem: When the bias boosts ']' probability, it can make grammar-invalid
// tokens appear valid. For example, while generating a string like "software",
// the bias might make ']' have a higher logit than it should, causing invalid
// JSON like: ["web", "software", "]") where ']' appears in the wrong position.
//
// Requirements for a proper fix:
// 1. Bias must only be applied to tokens that pass the grammar validity check
// 2. The bias should boost ']' only at valid array-close positions (after comma-separated elements)
// 3. The bias should NOT affect generation inside string literals
test "array_repetition_bias_disabled_by_default_due_to_grammar_interference" {
    // Verify the default config has bias disabled
    const default_config = GrammarConfig{};
    try std.testing.expectEqual(false, default_config.enable_array_repetition_bias);
}
