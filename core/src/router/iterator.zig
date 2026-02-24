//! Token Iterator for streaming generation.
//!
//! Provides a pull-based API for streaming token generation. The iterator
//! manages its own generation thread and ring buffer, allowing Python/Rust
//! to poll for tokens without callback lifetime issues.
//!
//! Content classification: each streamed token slot carries `item_type` and
//! `content_type` discriminators from core/src/responses/items.zig.  This
//! enables bindings to emit the correct OpenResponses SSE event per token
//! (e.g. `response.output_text.delta` vs `response.reasoning.delta`).
//!
//! Reasoning tag filtering: when the model emits `<think>…</think>` tags,
//! the iterator parses them inline (same algorithm as ReasoningParser) and
//! classifies each text chunk as reasoning or response.  Tag text itself
//! is never emitted to the ring buffer.

const std = @import("std");
const local_mod = @import("local.zig");
const LocalEngine = local_mod.LocalEngine;
const GenerateOptions = local_mod.GenerateOptions;
const http_engine_mod = @import("http_engine.zig");
const HttpEngine = http_engine_mod.HttpEngine;
const commit_mod = @import("commit.zig");
const tokenizer_mod = @import("../tokenizer/root.zig");
const responses_mod = @import("../responses/root.zig");
const Chat = responses_mod.Chat;
const ItemType = responses_mod.ItemType;
const ContentType = responses_mod.ContentType;
const inference_types = @import("../inference/root.zig").types;
const FinishReason = inference_types.FinishReason;
const gen_config_mod = @import("../inference/config/generation.zig");
const log = @import("../log.zig");

/// Maximum length of a single decoded token in bytes.
const MAX_TOKEN_LEN: usize = 512;

/// Number of token slots in the ring buffer.
/// Larger = more buffering (smoother), smaller = less memory.
const RING_BUFFER_SIZE: usize = 32;

/// Maximum length of a reasoning tag marker (e.g. "<think>" = 7 bytes).
const MAX_TAG_LEN: usize = 64;

/// A slot in the ring buffer holding a decoded token.
const TokenSlot = struct {
    /// Decoded token text (null-terminated for C compatibility).
    text: [MAX_TOKEN_LEN]u8,
    /// Length of valid text (excluding null terminator).
    len: usize,
    /// Token ID from the model vocabulary.
    token_id: u32,
    /// Item type discriminator (responses/items.zig ItemType).
    item_type: u8,
    /// Content type discriminator (responses/items.zig ContentType).
    content_type: u8,
};

/// Filter state for reasoning tag parsing during streaming.
const FilterState = enum(u8) {
    /// In response content (or before any reasoning section).
    normal = 0,
    /// Inside a reasoning section.
    reasoning = 1,
};

/// Backend type for the iterator.
pub const BackendType = enum(u8) {
    local,
    http,
};

/// Token iterator for pull-based streaming generation.
///
/// The iterator owns a background thread that generates tokens into a ring
/// buffer. Callers poll `next()` to retrieve tokens one at a time. The
/// returned pointer is valid until the next `next()` call.
///
/// Thread safety: `next()` and `cancel()` are safe to call from any thread.
/// `deinit()` must only be called once, after generation is complete or cancelled.
pub const TokenIterator = struct {
    allocator: std.mem.Allocator,

    // Ring buffer for tokens
    ring: [RING_BUFFER_SIZE]TokenSlot,
    write_idx: std.atomic.Value(usize),
    read_idx: std.atomic.Value(usize),

    // Synchronization
    mutex: std.Thread.Mutex,
    not_empty: std.Thread.Condition, // Signaled when buffer has data
    not_full: std.Thread.Condition, // Signaled when buffer has space

    // Return buffer - stable storage for the current token
    // Caller's pointer remains valid until next call to next()
    return_buffer: [MAX_TOKEN_LEN]u8,
    return_item_type: u8,
    return_content_type: u8,

    // State
    done: std.atomic.Value(bool), // Generation complete
    cancelled: std.atomic.Value(bool), // User requested cancellation
    worker_exited: std.atomic.Value(bool), // Worker thread has fully exited
    error_code: std.atomic.Value(i32), // 0 = no error
    error_msg: ?[]const u8, // Error message (owned)

    // Stats (populated after generation completes)
    prompt_tokens: std.atomic.Value(usize),
    completion_tokens: std.atomic.Value(usize),
    prefill_ns: std.atomic.Value(u64),
    generation_ns: std.atomic.Value(u64),
    finish_reason: std.atomic.Value(u8), // FinishReason discriminator

    // Reasoning tag filter state (worker thread only — no atomics needed)
    filter_state: FilterState,
    filter_partial_buf: [MAX_TAG_LEN]u8,
    filter_partial_len: usize,
    swallow_next_newline: bool,
    start_marker: []const u8, // e.g. "<think>" (owned)
    end_marker: []const u8, // e.g. "</think>" (owned)
    is_tool_generation: bool,
    raw_output: bool,

    // UTF-8 streaming buffer (worker thread only).
    //
    // Byte-level BPE tokenizers (GPT-2/Qwen/Llama3) can produce incomplete
    // UTF-8 sequences when decoding a single token.  For example, Qwen3
    // token 26525 decodes to bytes [0x20, 0xF0, 0x9F, 0x98] — the leading
    // bytes of " 😊" — while the final byte (0x8A) is a separate token.
    //
    // This buffer holds trailing bytes that don't yet form complete UTF-8
    // codepoints.  Each pushToken call prepends these to the new token's
    // raw bytes, emits only complete codepoints, and retains any new
    // trailing incomplete sequence.
    utf8_pending: [3]u8, // max incomplete = 3 bytes (4-byte seq missing last byte)
    utf8_pending_len: u2, // 0..3
    // Last token that produced visible decoded bytes.
    // Used as decode context so per-call "strip leading space" logic only
    // applies to true stream start, not every token.
    decode_context_token: ?u32,

    // Backend type (local or HTTP)
    backend_type: BackendType,

    // Generation context — local backend (active when backend_type == .local)
    engine: ?*LocalEngine,
    chat: ?*Chat,
    options: ?GenerateOptions,

    // Generation context — HTTP backend (active when backend_type == .http)
    http_engine: ?*HttpEngine,
    http_chat: ?*Chat,
    http_options: ?http_engine_mod.GenerateOptions,

    // Worker thread
    worker_thread: ?std.Thread,

    /// Create a new token iterator.
    ///
    /// Starts a background thread that generates tokens. Call `next()` to
    /// retrieve tokens, `cancel()` to stop early, and `deinit()` to clean up.
    pub fn init(
        allocator: std.mem.Allocator,
        engine: *LocalEngine,
        chat: *Chat,
        options: GenerateOptions,
    ) !*TokenIterator {
        const self = try allocator.create(TokenIterator);
        errdefer allocator.destroy(self);

        // Build reasoning tag markers.
        const tag = options.reasoning_tag orelse "think";
        const start_marker = try std.fmt.allocPrint(allocator, "<{s}>", .{tag});
        errdefer allocator.free(start_marker);
        const end_marker = try std.fmt.allocPrint(allocator, "</{s}>", .{tag});
        errdefer allocator.free(end_marker);

        // Detect tool generation mode.
        const is_tool_gen = options.tools_json != null and
            (options.tool_choice == null or !std.mem.eql(u8, options.tool_choice.?, "none"));

        self.* = TokenIterator{
            .allocator = allocator,
            .ring = undefined,
            .write_idx = std.atomic.Value(usize).init(0),
            .read_idx = std.atomic.Value(usize).init(0),
            .mutex = .{},
            .not_empty = .{},
            .not_full = .{},
            .return_buffer = undefined,
            .return_item_type = @intFromEnum(ItemType.unknown),
            .return_content_type = @intFromEnum(ContentType.unknown),
            .done = std.atomic.Value(bool).init(false),
            .cancelled = std.atomic.Value(bool).init(false),
            .worker_exited = std.atomic.Value(bool).init(false),
            .error_code = std.atomic.Value(i32).init(0),
            .error_msg = null,
            .prompt_tokens = std.atomic.Value(usize).init(0),
            .completion_tokens = std.atomic.Value(usize).init(0),
            .prefill_ns = std.atomic.Value(u64).init(0),
            .generation_ns = std.atomic.Value(u64).init(0),
            .finish_reason = std.atomic.Value(u8).init(@intFromEnum(FinishReason.eos_token)),
            .filter_state = .normal,
            .filter_partial_buf = undefined,
            .filter_partial_len = 0,
            .swallow_next_newline = false,
            .start_marker = start_marker,
            .end_marker = end_marker,
            .is_tool_generation = is_tool_gen,
            .raw_output = options.raw_output,
            .utf8_pending = undefined,
            .utf8_pending_len = 0,
            .decode_context_token = null,
            .backend_type = .local,
            .engine = engine,
            .chat = chat,
            .options = options,
            .http_engine = null,
            .http_chat = null,
            .http_options = null,
            .worker_thread = null,
        };

        // Initialize ring buffer slots
        for (&self.ring) |*slot| {
            slot.len = 0;
            slot.token_id = 0;
            slot.text[0] = 0;
            slot.item_type = @intFromEnum(ItemType.unknown);
            slot.content_type = @intFromEnum(ContentType.unknown);
        }

        // Start worker thread
        self.worker_thread = try std.Thread.spawn(.{}, workerThread, .{self});

        log.debug("router", "TokenIterator created (local)", .{
            .handle = @intFromPtr(self),
            .is_tool_gen = @as(u8, @intFromBool(is_tool_gen)),
        }, @src());

        return self;
    }

    /// Create a token iterator for HTTP (remote) backend streaming.
    ///
    /// Like `init()` but for the HTTP engine. Starts a background thread
    /// that streams tokens from a remote API via SSE.
    pub fn initWithHttpEngine(
        allocator: std.mem.Allocator,
        http_engine: *HttpEngine,
        chat: *Chat,
        options: http_engine_mod.GenerateOptions,
    ) !*TokenIterator {
        const self = try allocator.create(TokenIterator);
        errdefer allocator.destroy(self);

        // Build reasoning tag markers (default "think" for HTTP).
        const tag = "think";
        const start_marker = try std.fmt.allocPrint(allocator, "<{s}>", .{tag});
        errdefer allocator.free(start_marker);
        const end_marker = try std.fmt.allocPrint(allocator, "</{s}>", .{tag});
        errdefer allocator.free(end_marker);

        // Detect tool generation mode.
        const is_tool_gen = options.tools_json != null and
            (options.tool_choice == null or !std.mem.eql(u8, options.tool_choice.?, "none"));

        self.* = TokenIterator{
            .allocator = allocator,
            .ring = undefined,
            .write_idx = std.atomic.Value(usize).init(0),
            .read_idx = std.atomic.Value(usize).init(0),
            .mutex = .{},
            .not_empty = .{},
            .not_full = .{},
            .return_buffer = undefined,
            .return_item_type = @intFromEnum(ItemType.unknown),
            .return_content_type = @intFromEnum(ContentType.unknown),
            .done = std.atomic.Value(bool).init(false),
            .cancelled = std.atomic.Value(bool).init(false),
            .worker_exited = std.atomic.Value(bool).init(false),
            .error_code = std.atomic.Value(i32).init(0),
            .error_msg = null,
            .prompt_tokens = std.atomic.Value(usize).init(0),
            .completion_tokens = std.atomic.Value(usize).init(0),
            .prefill_ns = std.atomic.Value(u64).init(0),
            .generation_ns = std.atomic.Value(u64).init(0),
            .finish_reason = std.atomic.Value(u8).init(@intFromEnum(FinishReason.eos_token)),
            .filter_state = .normal,
            .filter_partial_buf = undefined,
            .filter_partial_len = 0,
            .swallow_next_newline = false,
            .start_marker = start_marker,
            .end_marker = end_marker,
            .is_tool_generation = is_tool_gen,
            .raw_output = options.raw_output,
            .utf8_pending = undefined,
            .utf8_pending_len = 0,
            .decode_context_token = null,
            .backend_type = .http,
            .engine = null,
            .chat = null,
            .options = null,
            .http_engine = http_engine,
            .http_chat = chat,
            .http_options = options,
            .worker_thread = null,
        };

        // Initialize ring buffer slots
        for (&self.ring) |*slot| {
            slot.len = 0;
            slot.token_id = 0;
            slot.text[0] = 0;
            slot.item_type = @intFromEnum(ItemType.unknown);
            slot.content_type = @intFromEnum(ContentType.unknown);
        }

        // Start worker thread
        self.worker_thread = try std.Thread.spawn(.{}, workerThread, .{self});

        log.debug("router", "TokenIterator created (http)", .{
            .handle = @intFromPtr(self),
            .is_tool_gen = @as(u8, @intFromBool(is_tool_gen)),
        }, @src());

        return self;
    }

    /// Get the next token from the iterator.
    ///
    /// Blocks until a token is available or generation completes.
    /// Returns null when generation is complete (EOS, max_tokens, or cancelled).
    /// The returned pointer is valid until the next `next()` call.
    /// After each call, `getItemType()` and `getContentType()` return the
    /// classification of the most recently returned token.
    pub fn next(self: *TokenIterator) ?[*:0]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Wait for data or completion
        while (true) {
            const read = self.read_idx.load(.acquire);
            const write = self.write_idx.load(.acquire);
            const done = self.done.load(.acquire);

            if (read != write) {
                // Data available - read from ring buffer
                const slot_idx = read % RING_BUFFER_SIZE;
                const slot = &self.ring[slot_idx];

                // Copy to return buffer BEFORE advancing read index
                // This ensures caller's pointer remains valid even if producer
                // overwrites the ring slot after we signal space available
                const copy_len = @min(slot.len + 1, MAX_TOKEN_LEN); // +1 for null terminator
                @memcpy(self.return_buffer[0..copy_len], slot.text[0..copy_len]);
                self.return_item_type = slot.item_type;
                self.return_content_type = slot.content_type;

                // NOW advance read index - slot is free for producer
                self.read_idx.store(read + 1, .release);

                // Signal producer that space is available
                self.not_full.signal();

                // Return pointer to stable return buffer, not ring slot
                return @ptrCast(&self.return_buffer);
            }

            if (done) {
                // Buffer empty and generation complete
                return null;
            }

            // Buffer empty but not done - wait for data
            self.not_empty.wait(&self.mutex);
        }
    }

    /// Cancel generation early.
    ///
    /// Signals the worker thread to stop. `next()` will return null after
    /// the currently buffered tokens are consumed.
    pub fn cancel(self: *TokenIterator) void {
        self.cancelled.store(true, .release);
    }

    /// Check if an error occurred during generation.
    pub fn hasError(self: *TokenIterator) bool {
        return self.error_code.load(.acquire) != 0;
    }

    /// Get the error code (0 = no error).
    pub fn getErrorCode(self: *TokenIterator) i32 {
        return self.error_code.load(.acquire);
    }

    /// Get the error message (null if no error).
    pub fn getErrorMsg(self: *TokenIterator) ?[]const u8 {
        return self.error_msg;
    }

    /// Get the item type of the most recently returned token.
    /// Returns ItemType discriminator (u8). 255 = unknown (no token returned yet).
    pub fn getItemType(self: *TokenIterator) u8 {
        return self.return_item_type;
    }

    /// Get the content type of the most recently returned token.
    /// Returns ContentType discriminator (u8). 255 = unknown (no token returned yet).
    pub fn getContentType(self: *TokenIterator) u8 {
        return self.return_content_type;
    }

    /// Get prompt token count (available after generation completes).
    pub fn getPromptTokens(self: *TokenIterator) usize {
        return self.prompt_tokens.load(.acquire);
    }

    /// Get completion token count (available after generation completes).
    pub fn getCompletionTokens(self: *TokenIterator) usize {
        return self.completion_tokens.load(.acquire);
    }

    /// Get prefill time in nanoseconds (available after generation completes).
    pub fn getPrefillNs(self: *TokenIterator) u64 {
        return self.prefill_ns.load(.acquire);
    }

    /// Get generation time in nanoseconds (available after generation completes).
    pub fn getGenerationNs(self: *TokenIterator) u64 {
        return self.generation_ns.load(.acquire);
    }

    /// Get finish reason (available after generation completes).
    /// Returns FinishReason discriminator (u8).
    pub fn getFinishReason(self: *TokenIterator) u8 {
        return self.finish_reason.load(.acquire);
    }

    /// Clean up the iterator.
    ///
    /// Waits for the worker thread to complete (cancels if still running),
    /// then frees all resources. Must only be called once.
    pub fn deinit(self: *TokenIterator) void {
        log.debug("router", "TokenIterator deinit", .{
            .handle = @intFromPtr(self),
            .prompt_tokens = self.prompt_tokens.load(.acquire),
            .completion_tokens = self.completion_tokens.load(.acquire),
        }, @src());
        // Signal cancellation if not already done
        self.cancelled.store(true, .release);

        // Wake up worker if blocked on buffer full
        // Always signal under mutex to avoid races - the worker checks cancelled
        // flag after waking, so spurious signals are harmless
        {
            self.mutex.lock();
            defer self.mutex.unlock();
            self.not_full.signal();
        }

        // Wait for worker thread to fully exit
        if (self.worker_thread) |thread| {
            thread.join();
        }

        // Free error message if any
        if (self.error_msg) |msg| {
            self.allocator.free(msg);
        }

        // Free marker strings
        self.allocator.free(self.start_marker);
        self.allocator.free(self.end_marker);

        // Free self
        self.allocator.destroy(self);
    }

    // =========================================================================
    // Worker Thread
    // =========================================================================

    fn workerThread(self: *TokenIterator) void {
        self.runGeneration() catch |err| {
            // Store error for caller to retrieve (use -1 as generic error code)
            self.error_code.store(-1, .release);
            self.error_msg = (if (err == error.UnsupportedContentType)
                std.fmt.allocPrint(self.allocator, "This model does not support images. Use a vision-language model (e.g. LFM2-VL-450M).", .{})
            else
                std.fmt.allocPrint(self.allocator, "Generation failed: {s}", .{@errorName(err)})) catch null;
        };

        // Signal completion - order matters for correctness:
        // 1. Set done first (consumer sees generation complete)
        // 2. Set worker_exited (deinit knows thread is finishing)
        // 3. Wake blocked consumer under mutex (provides synchronization)
        self.done.store(true, .release);
        self.worker_exited.store(true, .release);

        // Wake up any blocked reader - mutex provides memory synchronization
        self.mutex.lock();
        defer self.mutex.unlock();
        self.not_empty.signal();
    }

    fn runGeneration(self: *TokenIterator) !void {
        switch (self.backend_type) {
            .local => try self.runLocalGeneration(),
            .http => try self.runHttpGeneration(),
        }
    }

    fn runLocalGeneration(self: *TokenIterator) !void {
        // Check external stop flag before starting (if caller already cancelled)
        if (self.options.?.stop_flag) |external_flag| {
            if (external_flag.load(.acquire)) {
                self.cancelled.store(true, .release);
                return;
            }
        }

        // Build options with our internal callback
        var opts = self.options.?;
        opts.token_callback = tokenCallback;
        opts.callback_data = self;

        // Use internal cancellation flag as stop flag for Zig generation
        // We check the external flag in tokenCallback and set cancelled accordingly
        opts.stop_flag = @ptrCast(&self.cancelled);

        const engine = self.engine.?;

        // Run generation (this blocks until complete)
        const result = try engine.generate(self.chat.?, opts);
        // Use engine's allocator for result cleanup since generate() allocated with it
        defer result.deinit(engine.allocator);

        // Store stats for caller to retrieve
        self.prompt_tokens.store(result.prompt_tokens, .release);
        self.completion_tokens.store(result.generated_tokens, .release);
        self.prefill_ns.store(result.prefill_ns, .release);
        self.generation_ns.store(result.decode_ns, .release);
        self.finish_reason.store(@intFromEnum(result.finish_reason), .release);
    }

    fn runHttpGeneration(self: *TokenIterator) !void {
        const engine = self.http_engine.?;
        const chat = self.http_chat.?;
        var opts = self.http_options.?;

        // Phase A: Stream text tokens via callback → filterAndPush → ring buffer.
        opts.stream_callback = httpStreamCallback;
        opts.callback_data = self;

        const start_time = std.time.nanoTimestamp();

        const result = engine.stream(chat, opts) catch |err| {
            self.error_code.store(-1, .release);
            self.error_msg = std.fmt.allocPrint(
                self.allocator,
                "HTTP streaming failed: {s}",
                .{@errorName(err)},
            ) catch null;
            return err;
        };
        defer result.deinit(engine.allocator);

        const end_time = std.time.nanoTimestamp();
        const generation_ns: u64 = @intCast(end_time - start_time);

        // Store stats
        self.prompt_tokens.store(result.prompt_tokens, .release);
        self.completion_tokens.store(result.completion_tokens, .release);
        self.generation_ns.store(generation_ns, .release);
        self.finish_reason.store(@intFromEnum(
            httpFinishReasonToLocal(result.finish_reason),
        ), .release);

        // Phase B: Shared commit (SAME function as local path).
        // Convert HttpEngine.ToolCall → commit.ToolCallInput
        var tc_buf: [32]commit_mod.ToolCallInput = undefined;
        const tc_count = @min(result.tool_calls.len, tc_buf.len);
        for (result.tool_calls[0..tc_count], 0..) |tc, i| {
            tc_buf[i] = .{ .id = tc.id, .name = tc.name, .arguments = tc.arguments };
        }

        const finish_reason_str = httpFinishReasonToString(result.finish_reason);

        commit_mod.commitGenerationResult(self.allocator, chat, .{
            .text = result.text,
            .tool_calls = tc_buf[0..tc_count],
            .prompt_tokens = result.prompt_tokens,
            .completion_tokens = result.completion_tokens,
            .generation_ns = generation_ns,
            .finish_reason = finish_reason_str,
        }) catch |err| {
            self.error_code.store(-1, .release);
            self.error_msg = std.fmt.allocPrint(
                self.allocator,
                "Commit failed: {s}",
                .{@errorName(err)},
            ) catch null;
            return err;
        };
    }

    /// Callback invoked by HttpEngine for each streamed content chunk.
    ///
    /// Routes text through filterAndPush (reasoning tag parsing) or pushSlot
    /// (tool call mode), same as the local backend's tokenCallback→pushToken
    /// but operating on decoded text rather than token IDs.
    fn httpStreamCallback(content: []const u8, user_data: ?*anyopaque) bool {
        const self: *TokenIterator = @ptrCast(@alignCast(user_data orelse return true));

        // Check for cancellation
        if (self.cancelled.load(.acquire)) return false;

        if (content.len == 0) return true;

        if (self.is_tool_generation) {
            self.pushSlot(
                content,
                0,
                @intFromEnum(ItemType.function_call),
                @intFromEnum(ContentType.output_text),
            ) catch {
                self.cancelled.store(true, .release);
                return false;
            };
        } else {
            self.filterAndPush(content, 0) catch {
                self.cancelled.store(true, .release);
                return false;
            };
        }
        return true;
    }

    /// Callback invoked by LocalEngine for each generated token.
    fn tokenCallback(token_id: u32, user_data: ?*anyopaque) void {
        const self: *TokenIterator = @ptrCast(@alignCast(user_data orelse return));

        // Check external stop flag (from Python caller)
        if (self.options) |opts| {
            if (opts.stop_flag) |external_flag| {
                if (external_flag.load(.acquire)) {
                    self.cancelled.store(true, .release);
                    return;
                }
            }
        }

        self.pushToken(token_id) catch {
            // If push fails (cancelled or error), mark as cancelled to stop generation
            self.cancelled.store(true, .release);
        };
    }

    fn pushToken(self: *TokenIterator, token_id: u32) !void {
        // Check for cancellation
        if (self.cancelled.load(.acquire)) {
            return error.Cancelled;
        }

        // Skip EOS/stop tokens — they should not appear in the stream.
        // The post-generation path (local.zig) strips them from the final
        // decoded text; the streaming path must do the same.
        if (gen_config_mod.isEosToken(self.engine.?.gen_config.eos_token_ids, token_id))
            return;

        // Decode token to raw bytes (no UTF-8 sanitization).
        //
        // Important: many tokenizers apply decoder "strip_start"/prefix-space
        // logic at decode-call start. Decoding each token in isolation causes
        // those rules to fire on every token and swallow spaces. We decode
        // with one-token left context and emit only the delta bytes.
        //
        // Byte-level BPE tokens can still produce incomplete UTF-8 sequences
        // when decoded; we assemble complete codepoints via utf8_pending.
        const engine = self.engine.?;
        const raw = try self.decodeRawWithContext(engine, token_id);
        defer engine.allocator.free(raw);

        if (raw.len > 0) {
            self.decode_context_token = token_id;
        }

        // Build combined buffer: pending bytes from previous token + new raw bytes.
        var combined_buf: [3 + MAX_TOKEN_LEN]u8 = undefined;
        const pending_len: usize = self.utf8_pending_len;
        @memcpy(combined_buf[0..pending_len], self.utf8_pending[0..pending_len]);
        const raw_copy_len = @min(raw.len, combined_buf.len - pending_len);
        @memcpy(combined_buf[pending_len..][0..raw_copy_len], raw[0..raw_copy_len]);
        const total_len = pending_len + raw_copy_len;
        const combined = combined_buf[0..total_len];

        // Find the boundary of complete UTF-8 codepoints.
        const valid_end = utf8ValidPrefix(combined);
        const valid = combined[0..valid_end];
        const trailing = combined[valid_end..total_len];

        // Store trailing bytes only if they form a valid incomplete UTF-8
        // sequence (a multi-byte lead followed by valid but insufficient
        // continuation bytes).  Anything else — invalid lead bytes,
        // broken continuations, or oversized leftovers — is irrecoverably
        // invalid and is dropped to prevent infinite accumulation.
        self.utf8_pending_len = 0;
        if (trailing.len > 0 and trailing.len <= 3) {
            const lead = trailing[0];
            const expected_len: usize = if (lead & 0xE0 == 0xC0)
                2
            else if (lead & 0xF0 == 0xE0)
                3
            else if (lead & 0xF8 == 0xF0)
                4
            else
                0;
            if (expected_len > 0 and trailing.len < expected_len) {
                // Verify all continuation bytes present so far are valid.
                var all_valid = true;
                for (trailing[1..]) |cb| {
                    if (cb & 0xC0 != 0x80) {
                        all_valid = false;
                        break;
                    }
                }
                if (all_valid) {
                    self.utf8_pending_len = @intCast(trailing.len);
                    @memcpy(self.utf8_pending[0..trailing.len], trailing);
                }
            }
        }

        if (valid.len == 0) return;

        if (self.is_tool_generation) {
            // Tool mode: all tokens are function call arguments, no tag filtering.
            try self.pushSlot(
                valid,
                token_id,
                @intFromEnum(ItemType.function_call),
                @intFromEnum(ContentType.output_text),
            );
        } else if (self.raw_output) {
            // Raw mode: bypass reasoning tag parsing and emit decoded text verbatim.
            try self.pushSlot(
                valid,
                token_id,
                @intFromEnum(ItemType.message),
                @intFromEnum(ContentType.output_text),
            );
        } else {
            // Normal mode: run reasoning tag filter.
            try self.filterAndPush(valid, token_id);
        }
    }

    fn decodeRawWithContext(self: *TokenIterator, engine: *LocalEngine, token_id: u32) ![]u8 {
        if (self.decode_context_token) |ctx_token| {
            const ctx_raw = engine.tok.decodeRawBytes(
                &[_]u32{ctx_token},
                .{ .skip_special_tokens = true },
            ) catch return engine.tok.decodeRawBytes(
                &[_]u32{token_id},
                .{ .skip_special_tokens = true },
            );
            defer engine.allocator.free(ctx_raw);

            const pair_raw = engine.tok.decodeRawBytes(
                &[_]u32{ ctx_token, token_id },
                .{ .skip_special_tokens = true },
            ) catch return engine.tok.decodeRawBytes(
                &[_]u32{token_id},
                .{ .skip_special_tokens = true },
            );
            errdefer engine.allocator.free(pair_raw);

            const prefix = longestCommonPrefixLen(ctx_raw, pair_raw);
            if (prefix < ctx_raw.len) {
                // Context decode does not prefix pair decode (unexpected).
                // Fall back to single-token decode for correctness.
                engine.allocator.free(pair_raw);
                return engine.tok.decodeRawBytes(
                    &[_]u32{token_id},
                    .{ .skip_special_tokens = true },
                );
            }

            const delta = pair_raw[prefix..];
            if (delta.len == pair_raw.len) {
                // No overlap at all: fallback to single decode.
                engine.allocator.free(pair_raw);
                return engine.tok.decodeRawBytes(
                    &[_]u32{token_id},
                    .{ .skip_special_tokens = true },
                );
            }

            const out = try engine.allocator.alloc(u8, delta.len);
            @memcpy(out, delta);
            engine.allocator.free(pair_raw);
            return out;
        }

        return engine.tok.decodeRawBytes(
            &[_]u32{token_id},
            .{ .skip_special_tokens = true },
        );
    }

    /// Run the reasoning tag filter on decoded text and push typed slot(s).
    ///
    /// Same algorithm as ReasoningParser.processChunk: state-dependent
    /// partial matching against start/end markers.  Text is classified
    /// based on the current filter state.  Tag text is never emitted.
    ///
    /// Tag markers are ASCII-only.  Non-ASCII bytes (>= 0x80) cannot be
    /// part of a tag, so they are forwarded as content without entering
    /// the byte-by-byte matching logic.  This preserves multi-byte UTF-8
    /// sequences: non-ASCII runs are pushed whole rather than split into
    /// individual bytes.
    fn filterAndPush(self: *TokenIterator, decoded: []const u8, token_id: u32) !void {
        var i: usize = 0;
        while (i < decoded.len) {
            const byte = decoded[i];

            // Non-ASCII byte — cannot be part of an ASCII tag marker.
            // Flush any pending tag-match buffer, then forward the entire
            // non-ASCII run (preserving multi-byte UTF-8 codepoints).
            if (byte >= 0x80) {
                try self.flushPartialBuf(token_id);

                // Find the end of the non-ASCII run.
                const run_start = i;
                while (i < decoded.len and decoded[i] >= 0x80) : (i += 1) {}
                const run = decoded[run_start..i];

                // Handle post-tag newline swallowing (non-ASCII is never '\n').
                self.swallow_next_newline = false;

                try self.pushSlotTyped(run, token_id);
                continue;
            }

            // ASCII byte — run the tag-matching state machine.
            if (self.filter_partial_len >= MAX_TAG_LEN) {
                // Overflow safety — flush everything as literal.
                try self.flushPartialBuf(token_id);
            }
            self.filter_partial_buf[self.filter_partial_len] = byte;
            self.filter_partial_len += 1;

            const buf = self.filter_partial_buf[0..self.filter_partial_len];

            // Complete marker check (state-dependent).
            if (self.filter_state == .normal and std.mem.eql(u8, buf, self.start_marker)) {
                self.filter_partial_len = 0;
                self.filter_state = .reasoning;
                self.swallow_next_newline = false;
                i += 1;
                continue;
            }
            if (self.filter_state == .reasoning and std.mem.eql(u8, buf, self.end_marker)) {
                self.filter_partial_len = 0;
                self.filter_state = .normal;
                self.swallow_next_newline = true;
                i += 1;
                continue;
            }

            // Prefix check against the state-relevant marker only.
            const is_prefix = switch (self.filter_state) {
                .normal => std.mem.startsWith(u8, self.start_marker, buf),
                .reasoning => std.mem.startsWith(u8, self.end_marker, buf),
            };

            if (!is_prefix) {
                // Not a tag prefix — flush buffer as content.

                // Handle post-tag newline swallowing.
                if (self.swallow_next_newline) {
                    self.swallow_next_newline = false;
                    // If buf is exactly "\n", consume it silently.
                    if (self.filter_partial_len == 1 and buf[0] == '\n') {
                        self.filter_partial_len = 0;
                        i += 1;
                        continue;
                    }
                }

                try self.flushPartialBuf(token_id);
            }

            i += 1;
        }
    }

    /// Push content from the current filter state (helper to avoid repeating type mapping).
    fn pushSlotTyped(self: *TokenIterator, text: []const u8, token_id: u32) !void {
        const item_type: u8 = switch (self.filter_state) {
            .normal => @intFromEnum(ItemType.message),
            .reasoning => @intFromEnum(ItemType.reasoning),
        };
        const content_type: u8 = switch (self.filter_state) {
            .normal => @intFromEnum(ContentType.output_text),
            .reasoning => @intFromEnum(ContentType.reasoning_text),
        };
        try self.pushSlot(text, token_id, item_type, content_type);
    }

    /// Flush the partial buffer contents as a typed ring slot.
    fn flushPartialBuf(self: *TokenIterator, token_id: u32) !void {
        if (self.filter_partial_len == 0) return;

        const item_type: u8 = switch (self.filter_state) {
            .normal => @intFromEnum(ItemType.message),
            .reasoning => @intFromEnum(ItemType.reasoning),
        };
        const content_type: u8 = switch (self.filter_state) {
            .normal => @intFromEnum(ContentType.output_text),
            .reasoning => @intFromEnum(ContentType.reasoning_text),
        };

        try self.pushSlot(
            self.filter_partial_buf[0..self.filter_partial_len],
            token_id,
            item_type,
            content_type,
        );
        self.filter_partial_len = 0;
    }

    /// Push a text chunk into the ring buffer with type classification.
    fn pushSlot(self: *TokenIterator, text: []const u8, token_id: u32, item_type: u8, content_type: u8) !void {
        if (text.len == 0) return;

        // Truncate if too long
        const len = @min(text.len, MAX_TOKEN_LEN - 1);

        self.mutex.lock();
        defer self.mutex.unlock();

        // Wait for space in ring buffer
        while (true) {
            const write = self.write_idx.load(.acquire);
            const read = self.read_idx.load(.acquire);

            // Check if buffer has space (leave 1 slot empty to distinguish full from empty)
            if ((write + 1) % (RING_BUFFER_SIZE + 1) != read % (RING_BUFFER_SIZE + 1)) {
                break;
            }

            // Buffer full - wait for consumer
            if (self.cancelled.load(.acquire)) {
                return error.Cancelled;
            }
            self.not_full.wait(&self.mutex);
        }

        // Write token to slot
        const write = self.write_idx.load(.acquire);
        const slot_idx = write % RING_BUFFER_SIZE;
        const slot = &self.ring[slot_idx];

        @memcpy(slot.text[0..len], text[0..len]);
        slot.text[len] = 0; // Null terminate
        slot.len = len;
        slot.token_id = token_id;
        slot.item_type = item_type;
        slot.content_type = content_type;

        // Advance write index
        self.write_idx.store(write + 1, .release);

        // Signal consumer that data is available
        self.not_empty.signal();
    }
};

/// Convert HTTP finish reason to local FinishReason enum.
fn httpFinishReasonToLocal(reason: http_engine_mod.FinishReason) FinishReason {
    return switch (reason) {
        .stop => .eos_token,
        .length => .length,
        .tool_calls => .tool_calls,
        .content_filter => .content_filter,
        .unknown => .eos_token,
    };
}

/// Convert HTTP finish reason to string for commit.
fn httpFinishReasonToString(reason: http_engine_mod.FinishReason) [:0]const u8 {
    return switch (reason) {
        .stop => "stop",
        .length => "length",
        .tool_calls => "tool_calls",
        .content_filter => "content_filter",
        .unknown => "stop",
    };
}

/// Return the length of the longest prefix of `bytes` that is valid UTF-8
/// consisting only of complete codepoints (no truncated trailing sequence).
///
/// For example, given [0x20, 0xF0, 0x9F, 0x98] (space + incomplete 4-byte
/// sequence), returns 1 (just the space).  Given [0x20, 0xF0, 0x9F, 0x98,
/// 0x8A], returns 5 (space + complete 😊).
fn utf8ValidPrefix(bytes: []const u8) usize {
    var last_valid: usize = 0;
    var i: usize = 0;
    while (i < bytes.len) {
        const b = bytes[i];
        const seq_len: usize = if (b < 0x80)
            1
        else if (b & 0xE0 == 0xC0)
            2
        else if (b & 0xF0 == 0xE0)
            3
        else if (b & 0xF8 == 0xF0)
            4
        else
            // Invalid lead byte (continuation byte or 0xFE/0xFF).
            // Cannot be part of a valid prefix — stop here.
            break;
        if (i + seq_len > bytes.len) break; // Incomplete sequence at end.
        // Validate continuation bytes.
        var valid = true;
        for (bytes[i + 1 ..][0 .. seq_len - 1]) |cb| {
            if (cb & 0xC0 != 0x80) {
                valid = false;
                break;
            }
        }
        if (!valid) break;
        i += seq_len;
        last_valid = i;
    }
    return last_valid;
}

fn longestCommonPrefixLen(a: []const u8, b: []const u8) usize {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n and a[i] == b[i]) : (i += 1) {}
    return i;
}

// =============================================================================
// Tests
// =============================================================================

test "TokenIterator ring buffer math" {
    // Verify ring buffer index calculations
    const size = RING_BUFFER_SIZE;

    // At start, both indices at 0 - buffer empty
    var write: usize = 0;
    var read: usize = 0;
    try std.testing.expect(write == read); // Empty

    // Write one token
    write = 1;
    try std.testing.expect(write != read); // Has data
    try std.testing.expectEqual(@as(usize, 0), read % size); // Read from slot 0

    // Read one token
    read = 1;
    try std.testing.expect(write == read); // Empty again

    // Fill buffer (leave 1 slot to distinguish full from empty)
    write = size;
    try std.testing.expect((write + 1) % (size + 1) != read % (size + 1)); // Not full yet
    write = size + 1;
    // This would be full condition
}

test "TokenSlot has type fields" {
    var slot: TokenSlot = .{
        .text = undefined,
        .len = 0,
        .token_id = 0,
        .item_type = @intFromEnum(ItemType.reasoning),
        .content_type = @intFromEnum(ContentType.reasoning_text),
    };
    try std.testing.expectEqual(@as(u8, 3), slot.item_type); // reasoning = 3
    try std.testing.expectEqual(@as(u8, 8), slot.content_type); // reasoning_text = 8

    slot.item_type = @intFromEnum(ItemType.message);
    slot.content_type = @intFromEnum(ContentType.output_text);
    try std.testing.expectEqual(@as(u8, 0), slot.item_type); // message = 0
    try std.testing.expectEqual(@as(u8, 5), slot.content_type); // output_text = 5
}

test "FilterState maps to expected types" {
    // Verify our filter state → type mapping logic
    const normal_item: u8 = @intFromEnum(ItemType.message);
    const normal_content: u8 = @intFromEnum(ContentType.output_text);
    const reasoning_item: u8 = @intFromEnum(ItemType.reasoning);
    const reasoning_content: u8 = @intFromEnum(ContentType.reasoning_text);

    try std.testing.expectEqual(@as(u8, 0), normal_item);
    try std.testing.expectEqual(@as(u8, 5), normal_content);
    try std.testing.expectEqual(@as(u8, 3), reasoning_item);
    try std.testing.expectEqual(@as(u8, 8), reasoning_content);
}

test "tool generation type classification" {
    // When is_tool_generation is true, all tokens should get function_call type
    const tool_item: u8 = @intFromEnum(ItemType.function_call);
    const tool_content: u8 = @intFromEnum(ContentType.output_text);

    try std.testing.expectEqual(@as(u8, 1), tool_item);
    try std.testing.expectEqual(@as(u8, 5), tool_content);
}

test "FinishReason discriminator values" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(FinishReason.eos_token));
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(FinishReason.length));
    try std.testing.expectEqual(@as(u8, 2), @intFromEnum(FinishReason.stop_sequence));
    try std.testing.expectEqual(@as(u8, 3), @intFromEnum(FinishReason.tool_calls));
    try std.testing.expectEqual(@as(u8, 4), @intFromEnum(FinishReason.content_filter));
}

test "utf8ValidPrefix pure ASCII" {
    try std.testing.expectEqual(@as(usize, 5), utf8ValidPrefix("hello"));
}

test "longestCommonPrefixLen basic cases" {
    try std.testing.expectEqual(@as(usize, 0), longestCommonPrefixLen("", ""));
    try std.testing.expectEqual(@as(usize, 3), longestCommonPrefixLen("abc", "abc"));
    try std.testing.expectEqual(@as(usize, 2), longestCommonPrefixLen("abX", "abY"));
    try std.testing.expectEqual(@as(usize, 0), longestCommonPrefixLen("x", "abc"));
}

test "utf8ValidPrefix empty" {
    try std.testing.expectEqual(@as(usize, 0), utf8ValidPrefix(""));
}

test "utf8ValidPrefix complete multi-byte" {
    // "é" = C3 A9 (2 bytes), "😊" = F0 9F 98 8A (4 bytes)
    try std.testing.expectEqual(@as(usize, 2), utf8ValidPrefix("\xc3\xa9"));
    try std.testing.expectEqual(@as(usize, 4), utf8ValidPrefix("\xf0\x9f\x98\x8a"));
}

test "utf8ValidPrefix truncated trailing sequence" {
    // Space + first 3 bytes of 4-byte emoji (missing last byte)
    const bytes = [_]u8{ 0x20, 0xF0, 0x9F, 0x98 };
    try std.testing.expectEqual(@as(usize, 1), utf8ValidPrefix(&bytes));
}

test "utf8ValidPrefix lone continuation byte" {
    // 0x8A is a continuation byte without a lead byte
    const bytes = [_]u8{0x8A};
    try std.testing.expectEqual(@as(usize, 0), utf8ValidPrefix(&bytes));
}

test "utf8ValidPrefix mixed valid + incomplete" {
    // "hi" (2 bytes) + incomplete 3-byte sequence (2 of 3)
    const bytes = [_]u8{ 'h', 'i', 0xE2, 0x80 };
    try std.testing.expectEqual(@as(usize, 2), utf8ValidPrefix(&bytes));
}

test "utf8ValidPrefix reassembled emoji" {
    // Full sequence: space + 😊 = [0x20, 0xF0, 0x9F, 0x98, 0x8A]
    const bytes = [_]u8{ 0x20, 0xF0, 0x9F, 0x98, 0x8A };
    try std.testing.expectEqual(@as(usize, 5), utf8ValidPrefix(&bytes));
}

test "utf8ValidPrefix stops at invalid lead byte" {
    // Valid ASCII followed by invalid lead byte (0xAF is a continuation byte).
    // The prefix must NOT include the 0xAF or anything after it.
    const bytes = [_]u8{ 0x48, 0xAF, 0x65 };
    try std.testing.expectEqual(@as(usize, 1), utf8ValidPrefix(&bytes));
}

test "utf8ValidPrefix stops at 0xFF byte" {
    // 0xFF is never valid in UTF-8.
    const bytes = [_]u8{ 0x41, 0xFF, 0x42 };
    try std.testing.expectEqual(@as(usize, 1), utf8ValidPrefix(&bytes));
}

test "utf8ValidPrefix stops at bad continuation" {
    // 2-byte lead 0xC3 followed by non-continuation 0x48 (ASCII 'H').
    const bytes = [_]u8{ 0xC3, 0x48, 0x65 };
    try std.testing.expectEqual(@as(usize, 0), utf8ValidPrefix(&bytes));
}

test "utf8ValidPrefix valid then bad continuation" {
    // Valid "H" then 2-byte lead 0xC3 with bad continuation 0x48.
    const bytes = [_]u8{ 0x48, 0xC3, 0x48 };
    try std.testing.expectEqual(@as(usize, 1), utf8ValidPrefix(&bytes));
}

// =============================================================================
// Filter (filterAndPush) behavioral tests
// =============================================================================

/// A slot captured from the ring buffer for test assertions.
const TestSlot = struct {
    text: []const u8,
    item_type: u8,
    content_type: u8,
};

/// Create a minimal TokenIterator suitable for testing filterAndPush.
/// The engine/chat fields are undefined — only filter + ring buffer state is valid.
fn makeTestIterator(start_marker: []const u8, end_marker: []const u8) TokenIterator {
    var iter: TokenIterator = undefined;

    // Ring buffer + sync
    iter.write_idx = std.atomic.Value(usize).init(0);
    iter.read_idx = std.atomic.Value(usize).init(0);
    iter.mutex = .{};
    iter.not_empty = .{};
    iter.not_full = .{};
    iter.cancelled = std.atomic.Value(bool).init(false);

    // Initialize ring slots
    for (&iter.ring) |*slot| {
        slot.len = 0;
        slot.token_id = 0;
        slot.text[0] = 0;
        slot.item_type = 0;
        slot.content_type = 0;
    }

    // Filter state
    iter.filter_state = .normal;
    iter.filter_partial_buf = undefined;
    iter.filter_partial_len = 0;
    iter.swallow_next_newline = false;
    iter.start_marker = start_marker;
    iter.end_marker = end_marker;
    iter.is_tool_generation = false;

    return iter;
}

/// Read all slots from the ring buffer.
fn drainSlots(iter: *TokenIterator, out: []TestSlot) usize {
    var count: usize = 0;
    const write = iter.write_idx.load(.acquire);
    var read = iter.read_idx.load(.acquire);
    while (read != write and count < out.len) {
        const slot = &iter.ring[read % RING_BUFFER_SIZE];
        out[count] = .{
            .text = slot.text[0..slot.len],
            .item_type = slot.item_type,
            .content_type = slot.content_type,
        };
        count += 1;
        read += 1;
    }
    iter.read_idx.store(read, .release);
    return count;
}

test "filterAndPush: normal text produces message/output_text slots" {
    var iter = makeTestIterator("<think>", "</think>");
    try iter.filterAndPush("Hello world", 1);
    // Flush any partial (last bytes may still be in partial buf)
    try iter.flushPartialBuf(1);

    var slots: [16]TestSlot = undefined;
    const n = drainSlots(&iter, &slots);
    try std.testing.expect(n >= 1);

    // Concatenate all slot text
    var buf: [256]u8 = undefined;
    var pos: usize = 0;
    for (slots[0..n]) |s| {
        @memcpy(buf[pos..][0..s.text.len], s.text);
        pos += s.text.len;
        // All slots should be message/output_text
        try std.testing.expectEqual(@as(u8, @intFromEnum(ItemType.message)), s.item_type);
        try std.testing.expectEqual(@as(u8, @intFromEnum(ContentType.output_text)), s.content_type);
    }
    try std.testing.expectEqualStrings("Hello world", buf[0..pos]);
}

test "filterAndPush: state transitions normal → reasoning → normal" {
    var iter = makeTestIterator("<think>", "</think>");
    try iter.filterAndPush("<think>reasoning</think>response", 1);
    try iter.flushPartialBuf(1);

    var slots: [32]TestSlot = undefined;
    const n = drainSlots(&iter, &slots);

    // Collect text by type
    var reasoning_buf: [256]u8 = undefined; // Filled by @memcpy below before read
    var reasoning_len: usize = 0;
    var response_buf: [256]u8 = undefined; // Filled by @memcpy below before read
    var response_len: usize = 0;
    for (slots[0..n]) |s| {
        if (s.item_type == @intFromEnum(ItemType.reasoning)) {
            @memcpy(reasoning_buf[reasoning_len..][0..s.text.len], s.text);
            reasoning_len += s.text.len;
        } else {
            @memcpy(response_buf[response_len..][0..s.text.len], s.text);
            response_len += s.text.len;
        }
    }
    try std.testing.expectEqualStrings("reasoning", reasoning_buf[0..reasoning_len]);
    try std.testing.expectEqualStrings("response", response_buf[0..response_len]);
}

test "filterAndPush: tag text never emitted" {
    var iter = makeTestIterator("<think>", "</think>");
    try iter.filterAndPush("<think>thought</think>answer", 1);
    try iter.flushPartialBuf(1);

    var slots: [32]TestSlot = undefined;
    const n = drainSlots(&iter, &slots);

    // Concatenate all emitted text
    var buf: [256]u8 = undefined;
    var pos: usize = 0;
    for (slots[0..n]) |s| {
        @memcpy(buf[pos..][0..s.text.len], s.text);
        pos += s.text.len;
    }
    const all_text = buf[0..pos];

    // Neither <think> nor </think> should appear in output
    try std.testing.expect(std.mem.indexOf(u8, all_text, "<think>") == null);
    try std.testing.expect(std.mem.indexOf(u8, all_text, "</think>") == null);
    // But the content should be there
    try std.testing.expect(std.mem.indexOf(u8, all_text, "thought") != null);
    try std.testing.expect(std.mem.indexOf(u8, all_text, "answer") != null);
}

test "filterAndPush: partial tag across multiple calls" {
    var iter = makeTestIterator("<think>", "</think>");

    // Split "<think>" across two calls: "<thi" + "nk>content"
    try iter.filterAndPush("<thi", 1);
    try iter.filterAndPush("nk>content", 2);
    try iter.flushPartialBuf(2);

    var slots: [32]TestSlot = undefined;
    const n = drainSlots(&iter, &slots);

    var buf: [256]u8 = undefined;
    var pos: usize = 0;
    for (slots[0..n]) |s| {
        @memcpy(buf[pos..][0..s.text.len], s.text);
        pos += s.text.len;
        // After <think>, all content should be reasoning
        try std.testing.expectEqual(@as(u8, @intFromEnum(ItemType.reasoning)), s.item_type);
    }
    try std.testing.expectEqualStrings("content", buf[0..pos]);
}

test "filterAndPush: post-tag newline swallowed" {
    // "</think>\nHello" → emits "Hello" not "\nHello"
    var iter = makeTestIterator("<think>", "</think>");
    try iter.filterAndPush("<think>r</think>\nHello", 1);
    try iter.flushPartialBuf(1);

    var slots: [32]TestSlot = undefined;
    const n = drainSlots(&iter, &slots);

    // Collect response text only
    var response_buf: [256]u8 = undefined; // Filled by @memcpy below before read
    var response_len: usize = 0;
    for (slots[0..n]) |s| {
        if (s.item_type == @intFromEnum(ItemType.message)) {
            @memcpy(response_buf[response_len..][0..s.text.len], s.text);
            response_len += s.text.len;
        }
    }
    try std.testing.expectEqualStrings("Hello", response_buf[0..response_len]);
}

test "filterAndPush: only first newline swallowed after tag" {
    // "</think>\n\nHello" → emits "\nHello"
    var iter = makeTestIterator("<think>", "</think>");
    try iter.filterAndPush("<think>r</think>\n\nHello", 1);
    try iter.flushPartialBuf(1);

    var slots: [32]TestSlot = undefined;
    const n = drainSlots(&iter, &slots);

    var response_buf: [256]u8 = undefined; // Filled by @memcpy below before read
    var response_len: usize = 0;
    for (slots[0..n]) |s| {
        if (s.item_type == @intFromEnum(ItemType.message)) {
            @memcpy(response_buf[response_len..][0..s.text.len], s.text);
            response_len += s.text.len;
        }
    }
    try std.testing.expectEqualStrings("\nHello", response_buf[0..response_len]);
}

test "filterAndPush: non-newline after tag preserved" {
    // "</think>Hello" → emits "Hello" (no newline to swallow)
    var iter = makeTestIterator("<think>", "</think>");
    try iter.filterAndPush("<think>r</think>Hello", 1);
    try iter.flushPartialBuf(1);

    var slots: [32]TestSlot = undefined;
    const n = drainSlots(&iter, &slots);

    var response_buf: [256]u8 = undefined; // Filled by @memcpy below before read
    var response_len: usize = 0;
    for (slots[0..n]) |s| {
        if (s.item_type == @intFromEnum(ItemType.message)) {
            @memcpy(response_buf[response_len..][0..s.text.len], s.text);
            response_len += s.text.len;
        }
    }
    try std.testing.expectEqualStrings("Hello", response_buf[0..response_len]);
}
