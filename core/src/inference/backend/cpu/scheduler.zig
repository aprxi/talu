//! Continuous Batching Scheduler
//!
//! This module provides a scheduler for continuous batching, enabling multiple
//! concurrent generation requests to share compute resources efficiently.
//!
//! Key features:
//! - Iteration-level batching: requests can join/leave at token boundaries
//! - Automatic slot management: requests queue when slots are full
//! - Configurable scheduling policy: FIFO (default), priority-based
//! - Non-blocking API: submit requests and poll for completions
//!
//! Usage:
//! ```zig
//! var scheduler = try Scheduler.init(allocator, backend, .{});
//! defer scheduler.deinit();
//!
//! // Submit requests
//! const req1 = try scheduler.submit(&[_]u32{1, 2, 3}, 100, null);
//! const req2 = try scheduler.submit(&[_]u32{4, 5}, 50, null);
//!
//! // Run generation steps
//! while (scheduler.hasActive()) {
//!     const completions = try scheduler.step();
//!     for (completions) |c| {
//!         // Handle completed token or finished request
//!     }
//! }
//! ```

const std = @import("std");
const contract = @import("../contract.zig");
const cpu_engine = @import("engine.zig");
const FusedCpuBackend = cpu_engine.FusedCpuBackend;
const DecodeRequest = contract.DecodeRequest;
const DecodeResult = contract.DecodeResult;
const sampling = @import("sampling.zig");
const log = @import("../../../log.zig");
const validate = @import("../../../validate/root.zig");
const tokenizer_mod = @import("../../../tokenizer/root.zig");

/// Request state in the scheduler.
pub const RequestState = enum {
    /// Waiting for a slot to become available
    queued,
    /// Slot allocated, waiting for prefill
    pending_prefill,
    /// Prefill complete, actively generating
    generating,
    /// Generation complete (hit EOS or max tokens)
    completed,
    /// Request was cancelled
    cancelled,
    /// Request failed with an error
    failed,
};

/// A generation request managed by the scheduler.
pub const Request = struct {
    /// Unique request ID
    id: u64,
    /// Current state
    state: RequestState,
    /// Assigned slot index (valid when state >= pending_prefill)
    slot_index: ?usize,
    /// Input prompt tokens
    prompt_tokens: []const u32,
    /// Maximum tokens to generate
    max_tokens: usize,
    /// Generated tokens so far
    generated_tokens: std.ArrayList(u32),
    /// Current position (prompt_len + generated_count)
    token_position: usize,
    /// EOS token IDs (generation stops if any is produced)
    eos_token_ids: []const u32,
    /// Stop sequences (tokenized). Generation stops when any sequence matches.
    stop_sequences: []const []const u32,
    /// Optional callback for streaming tokens.
    /// Signature: fn(request_id: u64, token: u32, is_final: bool, user_data: ?*anyopaque) void
    callback: ?*const fn (u64, u32, bool, ?*anyopaque) void,
    callback_data: ?*anyopaque,
    /// Sampling configuration
    sampling_config: sampling.SamplingConfig,
    /// Optional grammar sampler for constrained decoding.
    grammar_sampler: ?*validate.sampler.ConstrainedSampler = null,
    /// Optional backend-specific vision prefill payload.
    vision_input: ?*const anyopaque = null,
    /// Error message if state == failed
    error_msg: ?[]const u8,
    /// Priority (higher = more urgent, for priority scheduling)
    priority: i32,
    /// Timestamp when request was submitted (for FIFO ordering)
    submit_time: i64,
    /// Finish reason for completed requests
    finish_reason: FinishReason,

    pub fn deinit(self: *Request, alloc: std.mem.Allocator) void {
        self.generated_tokens.deinit(alloc);
        if (self.error_msg) |msg| {
            alloc.free(msg);
        }
    }
};

/// Reason why generation stopped.
pub const FinishReason = enum(u8) {
    /// Generation in progress (not finished).
    in_progress = 0,
    /// Generation stopped due to EOS token.
    eos_token = 1,
    /// Maximum token limit reached.
    length = 2,
    /// A stop sequence was matched.
    stop_sequence = 3,
    /// Request was cancelled.
    cancelled = 4,
};

/// Token produced by a generation step.
pub const TokenEvent = struct {
    /// Request ID that produced this token
    request_id: u64,
    /// The generated token
    token: u32,
    /// Whether this is the final token (EOS or max reached)
    is_final: bool,
    /// Slot index (for debugging)
    slot_index: usize,
};

/// Scheduler configuration.
pub const SchedulerConfig = struct {
    /// Maximum concurrent requests (limited by backend's max_batch_size)
    max_concurrent: ?usize = null,
    /// Default EOS token IDs if not specified per-request
    default_eos_token_ids: []const u32 = &.{},
    /// Default sampling configuration
    default_sampling: sampling.SamplingConfig = .{},
    /// Enable priority scheduling (otherwise FIFO)
    priority_scheduling: bool = false,
    /// Optional tokenizer for grammar-constrained sampling.
    tokenizer: ?*tokenizer_mod.Tokenizer = null,
};

/// Continuous batching scheduler.
///
/// Manages multiple concurrent generation requests, automatically batching
/// decode steps for efficiency while allowing requests to join/leave at
/// any token boundary.
///
/// The backend type must implement:
/// - `maxBatchSize(*const T) usize` - maximum concurrent slots
/// - `vocabSize(*const T) usize` - vocabulary size for logits
/// - `allocSlot(*T) ?usize` - allocate a slot, returns null if full
/// - `freeSlot(*T, usize) void` - release a slot
/// - `prefillSlot(*T, usize, []const u32, []f32) !void` - prefill
/// - optional: `prefillSlotWithVision(*T, usize, []const u32, ?*const PrefillVisionInput, []f32) !void`
/// - `decodeBatch(*T, []const DecodeRequest, []DecodeResult) !void` - batch decode
/// - optional: `decodeStreaming(*T, u32, usize, usize, []const u32, []u32, ?*const fn(u32, ?*anyopaque) void, ?*anyopaque) !usize`
pub fn GenericScheduler(comptime BackendType: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        backend: *BackendType,
        config: SchedulerConfig,

        /// All tracked requests (active + queued + completed)
        requests: std.AutoHashMap(u64, *Request),

        /// Queue of request IDs waiting for slots (ordered by priority/submit time)
        pending_queue: std.ArrayList(u64),

        /// Active request IDs (have slots, generating)
        active_requests: std.ArrayList(u64),

        /// Completed request IDs (ready for retrieval)
        completed_queue: std.ArrayList(u64),

        /// Next request ID
        next_request_id: u64,

        /// Scratch buffers for batched decode
        decode_requests: []DecodeRequest,
        decode_results: []DecodeResult,

        /// Logits buffer for prefill
        logits_buffer: []f32,

        /// Optimized sampler with SIMD and pre-allocated workspace
        sampler: sampling.Sampler,

        pub fn init(
            allocator: std.mem.Allocator,
            backend: *BackendType,
            config: SchedulerConfig,
        ) !Self {
            const max_batch = backend.maxBatchSize();
            const vocab = backend.vocabSize();
            const requested_max_batch = config.max_concurrent orelse max_batch;
            const effective_batch_size = @min(requested_max_batch, max_batch);

            const decode_request_buffer = try allocator.alloc(DecodeRequest, effective_batch_size);
            errdefer allocator.free(decode_request_buffer);

            const decode_result_buffer = try allocator.alloc(DecodeResult, effective_batch_size);
            errdefer allocator.free(decode_result_buffer);

            // Initialize optimized sampler with pre-allocated workspace
            var sampler_instance = try sampling.Sampler.init(
                allocator,
                @intCast(std.time.milliTimestamp()),
                vocab,
            );
            errdefer sampler_instance.deinit();

            // Allocate logits buffer for prefill
            const logits_scratch = try allocator.alloc(f32, vocab);
            errdefer allocator.free(logits_scratch);

            return Self{
                .allocator = allocator,
                .backend = backend,
                .config = config,
                .requests = std.AutoHashMap(u64, *Request).init(allocator),
                .pending_queue = .{},
                .active_requests = .{},
                .completed_queue = .{},
                .next_request_id = 1,
                .decode_requests = decode_request_buffer,
                .decode_results = decode_result_buffer,
                .logits_buffer = logits_scratch,
                .sampler = sampler_instance,
            };
        }

        pub fn deinit(self: *Self) void {
            // Free all requests
            var request_iter = self.requests.valueIterator();
            while (request_iter.next()) |req_ptr| {
                req_ptr.*.deinit(self.allocator);
                self.allocator.destroy(req_ptr.*);
            }
            self.requests.deinit();

            self.pending_queue.deinit(self.allocator);
            self.active_requests.deinit(self.allocator);
            self.completed_queue.deinit(self.allocator);
            self.allocator.free(self.decode_results);
            self.allocator.free(self.decode_requests);
            self.allocator.free(self.logits_buffer);
            self.sampler.deinit();
        }

        fn sampleToken(
            self: *Self,
            logits: []f32,
            sampling_config: sampling.SamplingConfig,
            grammar_sampler: ?*validate.sampler.ConstrainedSampler,
        ) !u32 {
            self.sampler.grammar_sampler = grammar_sampler;
            defer self.sampler.grammar_sampler = null;

            if (grammar_sampler != null) {
                if (self.config.tokenizer) |tokenizer| {
                    const sampled = try self.sampler.sampleConstrained(logits, sampling_config, tokenizer);
                    const token_id: u32 = @intCast(sampled);
                    if (tokenizer.tokenBytes(@intCast(token_id))) |token_text| {
                        try self.sampler.acceptToken(token_id, token_text);
                    }
                    return token_id;
                }
            }

            const sampled = try self.sampler.sample(logits, sampling_config);
            return @intCast(sampled);
        }

        fn grammarIsComplete(grammar_sampler: ?*validate.sampler.ConstrainedSampler) bool {
            return if (grammar_sampler) |gs| gs.state == .complete else false;
        }

        fn grammarCompleteOnEos(grammar_sampler: ?*validate.sampler.ConstrainedSampler) bool {
            return grammarIsComplete(grammar_sampler);
        }

        /// Submit a new generation request.
        ///
        /// Returns a request ID that can be used to track progress or cancel.
        /// The request will be queued if no slots are available.
        pub fn submit(
            self: *Self,
            prompt_tokens: []const u32,
            max_tokens: usize,
            options: ?SubmitOptions,
        ) !u64 {
            const submit_config = options orelse SubmitOptions{};

            const request_entry = try self.allocator.create(Request);
            errdefer self.allocator.destroy(request_entry);

            request_entry.* = Request{
                .id = self.next_request_id,
                .state = .queued,
                .slot_index = null,
                .prompt_tokens = prompt_tokens,
                .max_tokens = max_tokens,
                .generated_tokens = .{},
                .token_position = 0,
                .eos_token_ids = submit_config.eos_token_ids orelse self.config.default_eos_token_ids,
                .stop_sequences = submit_config.stop_sequences,
                .callback = submit_config.callback,
                .callback_data = submit_config.callback_data,
                .sampling_config = submit_config.sampling orelse self.config.default_sampling,
                .grammar_sampler = submit_config.grammar_sampler,
                .vision_input = submit_config.vision_input,
                .error_msg = null,
                .priority = submit_config.priority,
                .submit_time = std.time.milliTimestamp(),
                .finish_reason = .in_progress,
            };

            try self.requests.put(request_entry.id, request_entry);
            try self.pending_queue.append(self.allocator, request_entry.id);

            self.next_request_id += 1;

            // Try to activate immediately if slots available
            try self.activatePending();

            return request_entry.id;
        }

        /// Options for submit().
        pub const SubmitOptions = struct {
            eos_token_ids: ?[]const u32 = null,
            /// Stop sequences (tokenized). Generation stops when any sequence matches.
            stop_sequences: []const []const u32 = &.{},
            /// Callback invoked for each token. Signature: fn(request_id, token, is_final, user_data)
            callback: ?*const fn (u64, u32, bool, ?*anyopaque) void = null,
            callback_data: ?*anyopaque = null,
            sampling: ?sampling.SamplingConfig = null,
            /// Optional grammar sampler for constrained decoding.
            grammar_sampler: ?*validate.sampler.ConstrainedSampler = null,
            priority: i32 = 0,
            /// Optional stop flag for cancellation. When set to true, generation stops.
            /// This allows external cancellation (e.g., client disconnect) without
            /// waiting for the next callback invocation.
            stop_flag: ?*const std.atomic.Value(bool) = null,
            /// Optional backend-specific vision prefill payload.
            /// For FusedCpuBackend this is `*const PrefillVisionInput`.
            vision_input: ?*const anyopaque = null,
        };

        /// Cancel a request.
        ///
        /// If the request is generating, its slot is freed immediately.
        /// Returns true if the request was found and cancelled.
        pub fn cancel(self: *Self, request_id: u64) bool {
            const request_entry = self.requests.get(request_id) orelse return false;

            if (request_entry.state == .completed or request_entry.state == .cancelled or request_entry.state == .failed) {
                return false; // Already done
            }

            // Free slot if allocated
            if (request_entry.slot_index) |slot_index| {
                self.backend.freeSlot(slot_index);
            }

            // Remove from active list
            for (self.active_requests.items, 0..) |active_id, idx| {
                if (active_id == request_id) {
                    _ = self.active_requests.orderedRemove(idx);
                    break;
                }
            }

            // Remove from pending queue
            for (self.pending_queue.items, 0..) |pending_id, idx| {
                if (pending_id == request_id) {
                    _ = self.pending_queue.orderedRemove(idx);
                    break;
                }
            }

            request_entry.state = .cancelled;
            self.completed_queue.append(self.allocator, request_id) catch {};

            return true;
        }

        /// Get request state.
        pub fn getState(self: *const Self, request_id: u64) ?RequestState {
            const request_entry = self.requests.get(request_id) orelse return null;
            return request_entry.state;
        }

        /// Get generated tokens for a request.
        pub fn getGeneratedTokens(self: *const Self, request_id: u64) ?[]const u32 {
            const request_entry = self.requests.get(request_id) orelse return null;
            return request_entry.generated_tokens.items;
        }

        /// Check if there are any active or pending requests.
        pub fn hasActive(self: *const Self) bool {
            return self.active_requests.items.len > 0 or self.pending_queue.items.len > 0;
        }

        /// Get count of active requests (currently generating).
        pub fn activeCount(self: *const Self) usize {
            return self.active_requests.items.len;
        }

        /// Get count of pending requests (waiting for slots).
        pub fn pendingCount(self: *const Self) usize {
            return self.pending_queue.items.len;
        }

        /// Run one generation step for all active requests.
        ///
        /// This is the main scheduler loop body. It:
        /// 1. Prefills any pending_prefill requests
        /// 2. Batches decode for all generating requests
        /// 3. Samples next tokens
        /// 4. Returns token events for this step
        pub fn step(self: *Self) ![]TokenEvent {
            // First, prefill any requests that need it
            try self.runPrefills();

            // Build batch of decode requests
            var decode_batch_size: usize = 0;
            for (self.active_requests.items) |active_request_id| {
                const request_entry = self.requests.get(active_request_id) orelse continue;
                if (request_entry.state != .generating) continue;
                if (request_entry.slot_index == null) continue;

                // Get the last token (either from prompt or generated)
                const last_token_id = if (request_entry.generated_tokens.items.len > 0)
                    request_entry.generated_tokens.items[request_entry.generated_tokens.items.len - 1]
                else if (request_entry.prompt_tokens.len > 0)
                    request_entry.prompt_tokens[request_entry.prompt_tokens.len - 1]
                else
                    continue;

                self.decode_requests[decode_batch_size] = .{
                    .slot_index = request_entry.slot_index.?,
                    .token = last_token_id,
                };
                decode_batch_size += 1;
            }

            if (decode_batch_size == 0) {
                // Slots may have been freed during prefill - try to activate pending
                try self.activatePending();
                return &.{};
            }

            // Run batched decode
            try self.backend.decodeBatch(
                self.decode_requests[0..decode_batch_size],
                self.decode_results[0..decode_batch_size],
            );

            // Sample next tokens and build events
            var token_events: std.ArrayList(TokenEvent) = .{};
            defer token_events.deinit(self.allocator);

            for (self.decode_results[0..decode_batch_size]) |result| {
                // Find request for this slot
                var matched_request_entry: ?*Request = null;
                for (self.active_requests.items) |active_request_id| {
                    const request_entry = self.requests.get(active_request_id) orelse continue;
                    if (request_entry.slot_index == result.slot_index) {
                        matched_request_entry = request_entry;
                        break;
                    }
                }
                if (matched_request_entry == null) continue;
                const request_entry = matched_request_entry.?;

                // Sample next token using optimized sampler (SIMD, pre-allocated workspace)
                const next_token = self.sampleToken(
                    result.logits,
                    request_entry.sampling_config,
                    request_entry.grammar_sampler,
                ) catch 0;

                // Add to generated tokens
                try request_entry.generated_tokens.append(self.allocator, next_token);
                request_entry.token_position += 1;

                // Check for EOS token
                var finish_reason: FinishReason = .in_progress;
                if (grammarIsComplete(request_entry.grammar_sampler)) {
                    finish_reason = .stop_sequence;
                }
                for (request_entry.eos_token_ids) |eos_id| {
                    if (next_token == eos_id) {
                        finish_reason = if (grammarCompleteOnEos(request_entry.grammar_sampler)) .stop_sequence else .eos_token;
                        break;
                    }
                }

                // Check for stop sequences (only if not already EOS)
                if (finish_reason == .in_progress and request_entry.stop_sequences.len > 0) {
                    const stop_len = checkStopSequence(request_entry.generated_tokens.items, request_entry.stop_sequences);
                    if (stop_len > 0) {
                        finish_reason = .stop_sequence;
                        // Trim the stop sequence from generated tokens
                        request_entry.generated_tokens.shrinkRetainingCapacity(request_entry.generated_tokens.items.len - stop_len);
                    }
                }

                // Check for max tokens
                if (finish_reason == .in_progress and request_entry.generated_tokens.items.len >= request_entry.max_tokens) {
                    finish_reason = .length;
                }

                const is_final_token = finish_reason != .in_progress;

                // Invoke callback if set (don't callback for stop sequence tokens that were trimmed)
                if (request_entry.callback) |cb| {
                    if (finish_reason != .stop_sequence) {
                        cb(request_entry.id, next_token, is_final_token, request_entry.callback_data);
                    }
                }

                // Add event
                try token_events.append(self.allocator, .{
                    .request_id = request_entry.id,
                    .token = next_token,
                    .is_final = is_final_token,
                    .slot_index = result.slot_index,
                });

                // Handle completion
                if (is_final_token) {
                    self.completeRequest(request_entry, finish_reason);
                }
            }

            // Try to activate pending requests (slots may have freed)
            try self.activatePending();

            return try token_events.toOwnedSlice(self.allocator);
        }

        /// Pop completed requests from the queue.
        ///
        /// Returns request IDs that have finished (completed, cancelled, or failed).
        /// Caller should retrieve results and then call `remove()` to free resources.
        pub fn popCompleted(self: *Self) []u64 {
            const completed_ids = self.completed_queue.toOwnedSlice(self.allocator) catch return &.{};
            return completed_ids;
        }

        /// Remove a request and free its resources.
        ///
        /// Should be called after retrieving results from a completed request.
        pub fn remove(self: *Self, request_id: u64) void {
            if (self.requests.fetchRemove(request_id)) |kv| {
                kv.value.deinit(self.allocator);
                self.allocator.destroy(kv.value);
            }
        }

        /// Get the finish reason for a request.
        pub fn getFinishReason(self: *const Self, request_id: u64) ?FinishReason {
            const request_entry = self.requests.get(request_id) orelse return null;
            return request_entry.finish_reason;
        }

        /// Synchronous single-request generation.
        ///
        /// Submits one request and runs to completion, returning generated tokens.
        /// This is a convenience wrapper for single-request use cases.
        ///
        /// Uses a fast path that bypasses the step() overhead when the scheduler
        /// has no other active requests. This avoids per-token ArrayList iterations
        /// and allocation overhead that hurt single-request performance.
        ///
        /// Returns an owned slice of generated tokens (caller must free).
        pub fn generateSync(
            self: *Self,
            prompt_tokens: []const u32,
            max_tokens: usize,
            options: ?SubmitOptions,
        ) !GenerateSyncResult {
            const submit_config = options orelse SubmitOptions{};
            const eos_token_ids = submit_config.eos_token_ids orelse self.config.default_eos_token_ids;
            const stop_sequences = submit_config.stop_sequences;
            const sampling_config = submit_config.sampling orelse self.config.default_sampling;
            const callback = submit_config.callback;
            const callback_data = submit_config.callback_data;
            const grammar_sampler = submit_config.grammar_sampler;
            const stop_flag = submit_config.stop_flag;
            const vision_input = submit_config.vision_input;

            // Fast path: if no other requests are active, bypass the full step() machinery
            // This avoids per-token overhead from ArrayList iterations and allocations
            if (self.active_requests.items.len == 0 and self.pending_queue.items.len == 0) {
                return self.generateSyncFastPath(
                    prompt_tokens,
                    max_tokens,
                    eos_token_ids,
                    stop_sequences,
                    &sampling_config,
                    callback,
                    callback_data,
                    grammar_sampler,
                    vision_input,
                    stop_flag,
                );
            }

            // Slow path: use step() for concurrent requests
            const request_id = try self.submit(prompt_tokens, max_tokens, options);

            // Run until this request completes
            while (self.hasActive()) {
                const events = try self.step();
                defer self.allocator.free(events);

                // Check if our request completed
                for (events) |event| {
                    if (event.request_id == request_id and event.is_final) {
                        break;
                    }
                }

                // Check if request is done
                if (self.getState(request_id)) |state| {
                    if (state == .completed or state == .cancelled or state == .failed) {
                        break;
                    }
                }
            }

            // Get results
            const request_entry = self.requests.get(request_id) orelse return error.RequestNotFound;
            const generated_tokens = try self.allocator.dupe(u32, request_entry.generated_tokens.items);
            const finish_reason = request_entry.finish_reason;

            // Cleanup
            self.remove(request_id);

            // Note: slow path doesn't have accurate timing (would need to track in Request)
            return GenerateSyncResult{
                .tokens = generated_tokens,
                .finish_reason = finish_reason,
                .prefill_ns = 0,
                .decode_ns = 0,
            };
        }

        /// Fast path for single-request generation.
        /// Bypasses all request tracking and step() overhead.
        fn generateSyncFastPath(
            self: *Self,
            prompt_tokens: []const u32,
            max_tokens: usize,
            eos_token_ids: []const u32,
            stop_sequences: []const []const u32,
            sampling_config: *const sampling.SamplingConfig,
            callback: ?*const fn (u64, u32, bool, ?*anyopaque) void,
            callback_data: ?*anyopaque,
            grammar_sampler: ?*validate.sampler.ConstrainedSampler,
            vision_input: ?*const anyopaque,
            stop_flag: ?*const std.atomic.Value(bool),
        ) !GenerateSyncResult {
            // Check stop flag early - if already cancelled, return immediately
            if (stop_flag) |flag| {
                if (flag.load(.acquire)) {
                    return GenerateSyncResult{
                        .tokens = &[_]u32{},
                        .finish_reason = .cancelled,
                        .prefill_ns = 0,
                        .decode_ns = 0,
                    };
                }
            }

            var total_timer = std.time.Timer.start() catch unreachable;

            // Allocate slot directly (bypass request tracking)
            const slot_index = self.backend.allocSlot() orelse return error.NoSlotsAvailable;
            defer self.backend.freeSlot(slot_index);

            var prefill_timer = std.time.Timer.start() catch unreachable;
            // Prefill
            log.debug("inference", "Scheduler fast-path prefill start", .{
                .slot_index = slot_index,
                .prompt_len = prompt_tokens.len,
                .has_vision_input = @as(u8, @intFromBool(vision_input != null)),
            }, @src());
            try self.prefillWithOptionalVision(slot_index, prompt_tokens, vision_input);
            const prefill_ns = prefill_timer.read();
            log.debug("scheduler", "Prefill complete", .{
                .slot_index = slot_index,
                .prompt_len = prompt_tokens.len,
                .duration_ms = @as(f64, @floatFromInt(prefill_ns)) / 1_000_000.0,
                .tok_per_sec = @as(f64, @floatFromInt(prompt_tokens.len)) * 1_000_000_000.0 / @as(f64, @floatFromInt(prefill_ns)),
            }, @src());

            // Reseed sampler if seed is provided (for deterministic generation)
            if (sampling_config.seed != 0) {
                self.sampler.reseed(sampling_config.seed);
            }

            // Sample first token from prefill logits
            var current_token = self.sampleToken(self.logits_buffer, sampling_config.*, grammar_sampler) catch 0;

            // Allocate output buffer (pre-allocate max size to avoid per-token reallocs)
            var generated = try std.ArrayList(u32).initCapacity(self.allocator, max_tokens);
            errdefer generated.deinit(self.allocator);

            try generated.append(self.allocator, current_token);
            if (grammarIsComplete(grammar_sampler)) {
                if (callback) |cb| cb(0, current_token, true, callback_data);
                return GenerateSyncResult{
                    .tokens = try generated.toOwnedSlice(self.allocator),
                    .finish_reason = .stop_sequence,
                    .prefill_ns = prefill_ns,
                    .decode_ns = 0,
                };
            }

            // Check first token for EOS
            for (eos_token_ids) |eos_id| {
                if (current_token == eos_id) {
                    if (callback) |cb| cb(0, current_token, true, callback_data);
                    return GenerateSyncResult{
                        .tokens = try generated.toOwnedSlice(self.allocator),
                        .finish_reason = if (grammarCompleteOnEos(grammar_sampler)) .stop_sequence else .eos_token,
                        .prefill_ns = prefill_ns,
                        .decode_ns = 0,
                    };
                }
            }

            // Check first token for stop sequences
            if (stop_sequences.len > 0) {
                const stop_len = checkStopSequence(generated.items, stop_sequences);
                if (stop_len > 0) {
                    generated.shrinkRetainingCapacity(generated.items.len - stop_len);
                    return GenerateSyncResult{
                        .tokens = try generated.toOwnedSlice(self.allocator),
                        .finish_reason = .stop_sequence,
                        .prefill_ns = prefill_ns,
                        .decode_ns = 0,
                    };
                }
            }

            if (callback) |cb| cb(0, current_token, false, callback_data);

            // Check max_tokens == 1
            if (max_tokens <= 1) {
                return GenerateSyncResult{
                    .tokens = try generated.toOwnedSlice(self.allocator),
                    .finish_reason = .length,
                    .prefill_ns = prefill_ns,
                    .decode_ns = 0,
                };
            }

            // Preserve legacy metal text-only throughput path by using
            // backend streaming decode when safe and available.
            if (vision_input == null and self.shouldUseBackendStreamingDecode(sampling_config, stop_sequences, grammar_sampler)) {
                const remaining_token_budget = max_tokens - generated.items.len;
                const generated_tail = try self.allocator.alloc(u32, remaining_token_budget);
                defer self.allocator.free(generated_tail);

                var decode_timer = std.time.Timer.start() catch unreachable;

                if (stop_flag) |flag| {
                    if (flag.load(.acquire)) {
                        return GenerateSyncResult{
                            .tokens = try generated.toOwnedSlice(self.allocator),
                            .finish_reason = .cancelled,
                            .prefill_ns = prefill_ns,
                            .decode_ns = 0,
                        };
                    }
                }

                const tail_count = if (comptime @hasDecl(BackendType, "decodeStreaming"))
                    try self.backend.decodeStreaming(
                        current_token,
                        prompt_tokens.len + generated.items.len,
                        remaining_token_budget,
                        eos_token_ids,
                        generated_tail,
                        null,
                        null,
                    )
                else
                    unreachable;

                try generated.appendSlice(self.allocator, generated_tail[0..tail_count]);

                if (callback) |cb| {
                    for (generated_tail[0..tail_count], 0..) |token_id, idx| {
                        cb(0, token_id, idx + 1 == tail_count, callback_data);
                    }
                }

                const decode_ns = decode_timer.read();
                const finish_reason: FinishReason = if (tail_count < remaining_token_budget) .eos_token else .length;
                return GenerateSyncResult{
                    .tokens = try generated.toOwnedSlice(self.allocator),
                    .finish_reason = finish_reason,
                    .prefill_ns = prefill_ns,
                    .decode_ns = decode_ns,
                };
            }

            // Decode loop via decodeBatch.
            var decode_timer = std.time.Timer.start() catch unreachable;
            var tokens_generated: usize = 1; // Already have first token from prefill

            while (generated.items.len < max_tokens) {
                // Check stop flag (allows external cancellation)
                if (stop_flag) |flag| {
                    if (flag.load(.acquire)) {
                        const decode_ns = decode_timer.read();
                        log.debug("scheduler", "Decode cancelled (stop flag)", .{
                            .tokens_generated = tokens_generated,
                            .decode_ms = @as(f64, @floatFromInt(decode_ns)) / 1_000_000.0,
                        }, @src());
                        return GenerateSyncResult{
                            .tokens = try generated.toOwnedSlice(self.allocator),
                            .finish_reason = .cancelled,
                            .prefill_ns = prefill_ns,
                            .decode_ns = decode_ns,
                        };
                    }
                }

                // Decode step
                self.decode_requests[0] = .{
                    .slot_index = slot_index,
                    .token = current_token,
                };
                try self.backend.decodeBatch(self.decode_requests[0..1], self.decode_results[0..1]);

                // Sample next token
                const next_token = self.sampleToken(self.decode_results[0].logits, sampling_config.*, grammar_sampler) catch 0;

                try generated.append(self.allocator, next_token);
                current_token = next_token;
                tokens_generated += 1;
                if (grammarIsComplete(grammar_sampler)) {
                    const decode_ns = decode_timer.read();
                    const total_ns = total_timer.read();
                    log.debug("scheduler", "Decode complete (grammar)", .{
                        .tokens_generated = tokens_generated,
                        .decode_ms = @as(f64, @floatFromInt(decode_ns)) / 1_000_000.0,
                        .decode_tok_per_sec = @as(f64, @floatFromInt(tokens_generated)) * 1_000_000_000.0 / @as(f64, @floatFromInt(decode_ns)),
                        .total_ms = @as(f64, @floatFromInt(total_ns)) / 1_000_000.0,
                    }, @src());
                    if (callback) |cb| cb(0, next_token, true, callback_data);
                    return GenerateSyncResult{
                        .tokens = try generated.toOwnedSlice(self.allocator),
                        .finish_reason = .stop_sequence,
                        .prefill_ns = prefill_ns,
                        .decode_ns = decode_ns,
                    };
                }

                // Check for EOS
                for (eos_token_ids) |eos_id| {
                    if (next_token == eos_id) {
                        const decode_ns = decode_timer.read();
                        const total_ns = total_timer.read();
                        log.debug("scheduler", "Decode complete (EOS)", .{
                            .tokens_generated = tokens_generated,
                            .decode_ms = @as(f64, @floatFromInt(decode_ns)) / 1_000_000.0,
                            .decode_tok_per_sec = @as(f64, @floatFromInt(tokens_generated)) * 1_000_000_000.0 / @as(f64, @floatFromInt(decode_ns)),
                            .total_ms = @as(f64, @floatFromInt(total_ns)) / 1_000_000.0,
                        }, @src());
                        if (callback) |cb| cb(0, next_token, true, callback_data);
                        return GenerateSyncResult{
                            .tokens = try generated.toOwnedSlice(self.allocator),
                            .finish_reason = if (grammarCompleteOnEos(grammar_sampler)) .stop_sequence else .eos_token,
                            .prefill_ns = prefill_ns,
                            .decode_ns = decode_ns,
                        };
                    }
                }

                // Check for stop sequences
                if (stop_sequences.len > 0) {
                    const stop_len = checkStopSequence(generated.items, stop_sequences);
                    if (stop_len > 0) {
                        generated.shrinkRetainingCapacity(generated.items.len - stop_len);
                        const decode_ns = decode_timer.read();
                        const total_ns = total_timer.read();
                        log.debug("scheduler", "Decode complete (stop seq)", .{
                            .tokens_generated = tokens_generated,
                            .decode_ms = @as(f64, @floatFromInt(decode_ns)) / 1_000_000.0,
                            .decode_tok_per_sec = @as(f64, @floatFromInt(tokens_generated)) * 1_000_000_000.0 / @as(f64, @floatFromInt(decode_ns)),
                            .total_ms = @as(f64, @floatFromInt(total_ns)) / 1_000_000.0,
                        }, @src());
                        if (callback) |cb| cb(0, next_token, true, callback_data);
                        return GenerateSyncResult{
                            .tokens = try generated.toOwnedSlice(self.allocator),
                            .finish_reason = .stop_sequence,
                            .prefill_ns = prefill_ns,
                            .decode_ns = decode_ns,
                        };
                    }
                }

                // Invoke callback
                if (callback) |cb| {
                    const is_final = generated.items.len >= max_tokens;
                    cb(0, next_token, is_final, callback_data);
                }
            }

            const decode_ns = decode_timer.read();
            log.debug("scheduler", "Decode complete (length)", .{
                .tokens_generated = tokens_generated,
                .decode_ms = @as(f64, @floatFromInt(decode_ns)) / 1_000_000.0,
                .decode_tok_per_sec = @as(f64, @floatFromInt(tokens_generated)) * 1_000_000_000.0 / @as(f64, @floatFromInt(decode_ns)),
            }, @src());

            return GenerateSyncResult{
                .tokens = try generated.toOwnedSlice(self.allocator),
                .finish_reason = .length,
                .prefill_ns = prefill_ns,
                .decode_ns = decode_ns,
            };
        }

        /// Result from generateSync.
        pub const GenerateSyncResult = struct {
            /// Generated tokens (owned, caller must free).
            tokens: []u32,
            /// Reason generation stopped.
            finish_reason: FinishReason,
            /// Prefill time in nanoseconds.
            prefill_ns: u64,
            /// Decode time in nanoseconds.
            decode_ns: u64,

            pub fn deinit(self: *GenerateSyncResult, allocator: std.mem.Allocator) void {
                allocator.free(self.tokens);
                self.* = undefined;
            }
        };

        // =========================================================================
        // Internal helpers
        // =========================================================================

        fn activatePending(self: *Self) !void {
            // Sort pending queue if priority scheduling
            if (self.config.priority_scheduling) {
                std.mem.sort(u64, self.pending_queue.items, self, comparePriority);
            }

            // Try to allocate slots for pending requests
            // Process from front of queue until no more slots available
            while (self.pending_queue.items.len > 0) {
                const pending_request_id = self.pending_queue.items[0];
                const request_entry = self.requests.get(pending_request_id) orelse {
                    _ = self.pending_queue.orderedRemove(0);
                    continue;
                };

                // Try to allocate a slot
                if (self.backend.allocSlot()) |slot_index| {
                    request_entry.slot_index = slot_index;
                    request_entry.state = .pending_prefill;
                    try self.active_requests.append(self.allocator, pending_request_id);
                    _ = self.pending_queue.orderedRemove(0);
                } else {
                    // No more slots available
                    break;
                }
            }
        }

        fn comparePriority(self: *Self, a: u64, b: u64) bool {
            const request_a = self.requests.get(a) orelse return false;
            const request_b = self.requests.get(b) orelse return true;

            // Higher priority first
            if (request_a.priority != request_b.priority) {
                return request_a.priority > request_b.priority;
            }
            // Then earlier submit time
            return request_a.submit_time < request_b.submit_time;
        }

        fn runPrefills(self: *Self) !void {
            for (self.active_requests.items) |active_request_id| {
                const request_entry = self.requests.get(active_request_id) orelse continue;
                if (request_entry.state != .pending_prefill) continue;
                if (request_entry.slot_index == null) continue;

                // Run prefill for this request
                try self.prefillWithOptionalVision(
                    request_entry.slot_index.?,
                    request_entry.prompt_tokens,
                    request_entry.vision_input,
                );

                request_entry.token_position = request_entry.prompt_tokens.len;
                request_entry.state = .generating;

                // Sample first token from prefill logits
                const first_token_id = self.sampleToken(
                    self.logits_buffer,
                    request_entry.sampling_config,
                    request_entry.grammar_sampler,
                ) catch 0;
                try request_entry.generated_tokens.append(self.allocator, first_token_id);

                // Check for EOS token
                var finish_reason: FinishReason = .in_progress;
                if (grammarIsComplete(request_entry.grammar_sampler)) {
                    finish_reason = .stop_sequence;
                }
                for (request_entry.eos_token_ids) |eos_id| {
                    if (first_token_id == eos_id) {
                        finish_reason = if (grammarCompleteOnEos(request_entry.grammar_sampler)) .stop_sequence else .eos_token;
                        break;
                    }
                }

                // Check for stop sequences (first token could match a single-token stop sequence)
                if (finish_reason == .in_progress and request_entry.stop_sequences.len > 0) {
                    const stop_len = checkStopSequence(request_entry.generated_tokens.items, request_entry.stop_sequences);
                    if (stop_len > 0) {
                        finish_reason = .stop_sequence;
                        request_entry.generated_tokens.shrinkRetainingCapacity(request_entry.generated_tokens.items.len - stop_len);
                    }
                }

                // Check for max tokens
                if (finish_reason == .in_progress and request_entry.max_tokens <= 1) {
                    finish_reason = .length;
                }

                const is_final_token = finish_reason != .in_progress;

                if (request_entry.callback) |cb| {
                    if (finish_reason != .stop_sequence) {
                        cb(request_entry.id, first_token_id, is_final_token, request_entry.callback_data);
                    }
                }

                if (is_final_token) {
                    self.completeRequest(request_entry, finish_reason);
                }
            }
        }

        fn completeRequest(self: *Self, request_entry: *Request, reason: FinishReason) void {
            request_entry.state = .completed;
            request_entry.finish_reason = reason;

            // Free slot
            if (request_entry.slot_index) |slot_index| {
                self.backend.freeSlot(slot_index);
                request_entry.slot_index = null;
            }

            // Remove from active list
            for (self.active_requests.items, 0..) |active_id, idx| {
                if (active_id == request_entry.id) {
                    _ = self.active_requests.orderedRemove(idx);
                    break;
                }
            }

            // Add to completed queue
            self.completed_queue.append(self.allocator, request_entry.id) catch {};
        }

        /// Check if the generated tokens end with any stop sequence.
        /// Returns the length of the matching stop sequence, or 0 if no match.
        fn checkStopSequence(generated_tokens: []const u32, stop_sequences: []const []const u32) usize {
            for (stop_sequences) |stop_seq| {
                if (stop_seq.len == 0) continue;
                if (generated_tokens.len < stop_seq.len) continue;

                // Check if the last N tokens match this stop sequence
                const start_idx = generated_tokens.len - stop_seq.len;
                var matches = true;
                for (stop_seq, 0..) |token, i| {
                    if (generated_tokens[start_idx + i] != token) {
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

        fn shouldUseBackendStreamingDecode(
            self: *const Self,
            sampling_config: *const sampling.SamplingConfig,
            stop_sequences: []const []const u32,
            grammar_sampler: ?*validate.sampler.ConstrainedSampler,
        ) bool {
            if (sampling_config.strategy != .greedy) return false;
            if (stop_sequences.len != 0) return false;
            if (grammar_sampler != null) return false;
            if (comptime !@hasDecl(BackendType, "decodeStreaming")) return false;
            if (comptime @hasDecl(BackendType, "supportsSchedulerStreamingFastPath")) {
                return self.backend.supportsSchedulerStreamingFastPath();
            }
            return false;
        }

        fn prefillWithOptionalVision(
            self: *Self,
            slot_index: usize,
            prompt_tokens: []const u32,
            vision_input: ?*const anyopaque,
        ) !void {
            if (vision_input) |opaque_ptr| {
                log.debug("scheduler", "Prefill dispatch (vision)", .{
                    .slot_index = slot_index,
                    .prompt_tokens = prompt_tokens.len,
                }, @src());
                log.debug("inference", "Scheduler prefill dispatch", .{
                    .slot_index = slot_index,
                    .prompt_tokens = prompt_tokens.len,
                    .has_prefill_with_vision = @as(u8, @intFromBool(@hasDecl(BackendType, "prefillSlotWithVision"))),
                    .has_prefill_vision_type = @as(u8, @intFromBool(@hasDecl(BackendType, "PrefillVisionInput"))),
                }, @src());
                if (comptime @hasDecl(BackendType, "prefillSlotWithVision") and @hasDecl(BackendType, "PrefillVisionInput")) {
                    const typed_input: *const BackendType.PrefillVisionInput = @ptrCast(@alignCast(opaque_ptr));
                    try self.backend.prefillSlotWithVision(
                        slot_index,
                        prompt_tokens,
                        typed_input,
                        self.logits_buffer,
                    );
                    log.debug("inference", "Scheduler prefill with vision completed", .{
                        .slot_index = slot_index,
                        .prompt_tokens = prompt_tokens.len,
                    }, @src());
                    return;
                }
                log.warn("inference", "Scheduler backend lacks vision prefill support", .{
                    .slot_index = slot_index,
                });
                return error.UnsupportedContentType;
            }

            return self.backend.prefillSlot(slot_index, prompt_tokens, self.logits_buffer);
        }
    };
}

/// Default Scheduler type using FusedCpuBackend.
pub const Scheduler = GenericScheduler(FusedCpuBackend);

// =============================================================================
// Tests
// =============================================================================

test "init Scheduler types" {
    const testing = std.testing;
    _ = testing;

    // Verify RequestState enum
    try std.testing.expectEqual(RequestState.queued, RequestState.queued);
    try std.testing.expectEqual(RequestState.generating, RequestState.generating);

    // Verify TokenEvent structure
    const event = TokenEvent{
        .request_id = 1,
        .token = 42,
        .is_final = false,
        .slot_index = 0,
    };
    try std.testing.expectEqual(@as(u64, 1), event.request_id);
    try std.testing.expectEqual(@as(u32, 42), event.token);
}

test "init SchedulerConfig defaults" {
    const config = SchedulerConfig{};
    try std.testing.expectEqual(@as(?usize, null), config.max_concurrent);
    try std.testing.expectEqual(false, config.priority_scheduling);
}

// =============================================================================
// RequestState Tests
// =============================================================================

test "RequestState enum values distinct" {
    const states = [_]RequestState{
        .queued,
        .pending_prefill,
        .generating,
        .completed,
        .cancelled,
        .failed,
    };

    // Verify all states are unique
    for (states, 0..) |state1, i| {
        for (states[i + 1 ..]) |state2| {
            try std.testing.expect(state1 != state2);
        }
    }
}

test "RequestState enum terminal states" {
    // Terminal states that cannot transition further
    const terminal_states = [_]RequestState{
        .completed,
        .cancelled,
        .failed,
    };

    for (terminal_states) |state| {
        // Verify these are valid enum values
        try std.testing.expect(@intFromEnum(state) >= 0);
    }

    // Non-terminal states
    const active_states = [_]RequestState{
        .queued,
        .pending_prefill,
        .generating,
    };

    for (active_states) |state| {
        try std.testing.expect(state != .completed);
        try std.testing.expect(state != .cancelled);
        try std.testing.expect(state != .failed);
    }
}

// =============================================================================
// TokenEvent Tests
// =============================================================================

test "TokenEvent struct fields" {
    const event = TokenEvent{
        .request_id = std.math.maxInt(u64),
        .token = std.math.maxInt(u32),
        .is_final = true,
        .slot_index = std.math.maxInt(usize),
    };

    try std.testing.expectEqual(std.math.maxInt(u64), event.request_id);
    try std.testing.expectEqual(std.math.maxInt(u32), event.token);
    try std.testing.expectEqual(true, event.is_final);
    try std.testing.expectEqual(std.math.maxInt(usize), event.slot_index);
}

test "TokenEvent struct is_final flag" {
    const non_final = TokenEvent{
        .request_id = 1,
        .token = 100,
        .is_final = false,
        .slot_index = 0,
    };

    const final = TokenEvent{
        .request_id = 1,
        .token = 0, // EOS token
        .is_final = true,
        .slot_index = 0,
    };

    try std.testing.expectEqual(false, non_final.is_final);
    try std.testing.expectEqual(true, final.is_final);
}

// =============================================================================
// SchedulerConfig Tests
// =============================================================================

test "init SchedulerConfig custom values" {
    const eos_token_ids = [_]u32{ 0, 1, 2 };
    const config = SchedulerConfig{
        .max_concurrent = 16,
        .default_eos_token_ids = &eos_token_ids,
        .default_sampling = .{
            .strategy = .top_k,
            .temperature = 0.8,
            .top_k = 50,
        },
        .priority_scheduling = true,
    };

    try std.testing.expectEqual(@as(?usize, 16), config.max_concurrent);
    try std.testing.expectEqual(true, config.priority_scheduling);
    try std.testing.expectEqual(@as(usize, 3), config.default_eos_token_ids.len);
    try std.testing.expectEqual(sampling.SamplingStrategy.top_k, config.default_sampling.strategy);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), config.default_sampling.temperature, 0.001);
}

test "init SchedulerConfig zero max_concurrent" {
    const config = SchedulerConfig{
        .max_concurrent = null,
    };
    try std.testing.expectEqual(@as(?usize, null), config.max_concurrent);
}

// =============================================================================
// SubmitOptions Tests
// =============================================================================

test "init SubmitOptions defaults" {
    const opts = Scheduler.SubmitOptions{};

    try std.testing.expectEqual(@as(?[]const u32, null), opts.eos_token_ids);
    try std.testing.expect(opts.callback == null);
    try std.testing.expect(opts.callback_data == null);
    try std.testing.expect(opts.sampling == null);
    try std.testing.expectEqual(@as(i32, 0), opts.priority);
}

test "init SubmitOptions custom priority" {
    const opts_high = Scheduler.SubmitOptions{
        .priority = 100,
    };
    const opts_low = Scheduler.SubmitOptions{
        .priority = -50,
    };
    const opts_default = Scheduler.SubmitOptions{};

    try std.testing.expect(opts_high.priority > opts_default.priority);
    try std.testing.expect(opts_low.priority < opts_default.priority);
}

test "init SubmitOptions custom sampling" {
    const opts = Scheduler.SubmitOptions{
        .sampling = .{
            .strategy = .greedy,
            .temperature = 0.0,
        },
    };

    try std.testing.expect(opts.sampling != null);
    try std.testing.expectEqual(sampling.SamplingStrategy.greedy, opts.sampling.?.strategy);
}

// =============================================================================
// Request Tests
// =============================================================================

test "init Request initial state" {
    const alloc = std.testing.allocator;

    var request_entry = Request{
        .id = 42,
        .state = .queued,
        .slot_index = null,
        .prompt_tokens = &[_]u32{ 1, 2, 3 },
        .max_tokens = 100,
        .generated_tokens = .{},
        .token_position = 0,
        .eos_token_ids = &[_]u32{0},
        .stop_sequences = &.{},
        .callback = null,
        .callback_data = null,
        .sampling_config = .{},
        .error_msg = null,
        .priority = 0,
        .submit_time = 0,
        .finish_reason = .in_progress,
    };
    defer request_entry.deinit(alloc);

    try std.testing.expectEqual(@as(u64, 42), request_entry.id);
    try std.testing.expectEqual(RequestState.queued, request_entry.state);
    try std.testing.expectEqual(@as(?usize, null), request_entry.slot_index);
    try std.testing.expectEqual(@as(usize, 3), request_entry.prompt_tokens.len);
    try std.testing.expectEqual(@as(usize, 0), request_entry.generated_tokens.items.len);
}

test "step Request generated tokens append" {
    const alloc = std.testing.allocator;

    var request_entry = Request{
        .id = 1,
        .state = .generating,
        .slot_index = 0,
        .prompt_tokens = &[_]u32{100},
        .max_tokens = 10,
        .generated_tokens = .{},
        .token_position = 1,
        .eos_token_ids = &[_]u32{0},
        .stop_sequences = &.{},
        .callback = null,
        .callback_data = null,
        .sampling_config = .{},
        .error_msg = null,
        .priority = 0,
        .submit_time = 0,
        .finish_reason = .in_progress,
    };
    defer request_entry.deinit(alloc);

    // Simulate generating tokens
    try request_entry.generated_tokens.append(alloc, 200);
    try request_entry.generated_tokens.append(alloc, 201);
    try request_entry.generated_tokens.append(alloc, 202);

    try std.testing.expectEqual(@as(usize, 3), request_entry.generated_tokens.items.len);
    try std.testing.expectEqual(@as(u32, 200), request_entry.generated_tokens.items[0]);
    try std.testing.expectEqual(@as(u32, 201), request_entry.generated_tokens.items[1]);
    try std.testing.expectEqual(@as(u32, 202), request_entry.generated_tokens.items[2]);
}

test "deinit Request error message cleanup" {
    const alloc = std.testing.allocator;

    var request_entry = Request{
        .id = 1,
        .state = .failed,
        .slot_index = null,
        .prompt_tokens = &[_]u32{},
        .max_tokens = 0,
        .generated_tokens = .{},
        .token_position = 0,
        .eos_token_ids = &[_]u32{},
        .stop_sequences = &.{},
        .callback = null,
        .callback_data = null,
        .sampling_config = .{},
        .error_msg = try alloc.dupe(u8, "Test error message"),
        .priority = 0,
        .submit_time = 0,
        .finish_reason = .in_progress,
    };
    defer request_entry.deinit(alloc);

    try std.testing.expect(request_entry.error_msg != null);
    try std.testing.expectEqualStrings("Test error message", request_entry.error_msg.?);
}

// =============================================================================
// Priority Comparison Tests (using standalone function logic)
// =============================================================================

test "step priority higher wins" {
    // Test that higher priority value means higher priority
    const high_priority: i32 = 100;
    const low_priority: i32 = 0;

    // Higher priority should come first (return true means a before b)
    try std.testing.expect(high_priority > low_priority);
}

test "step priority same uses submit time" {
    // When priorities are equal, earlier submit time wins
    const early_time: i64 = 1000;
    const late_time: i64 = 2000;

    // Earlier time should come first
    try std.testing.expect(early_time < late_time);
}

test "step priority negative" {
    // Negative priorities are valid and lower than zero
    const negative: i32 = -10;
    const zero: i32 = 0;
    const positive: i32 = 10;

    try std.testing.expect(negative < zero);
    try std.testing.expect(zero < positive);
}

// =============================================================================
// EOS Token Detection Tests
// =============================================================================

test "step EOS detection single" {
    const eos_token_ids = [_]u32{0};
    const token: u32 = 0;

    var is_eos = false;
    for (eos_token_ids) |eos_id| {
        if (token == eos_id) {
            is_eos = true;
            break;
        }
    }

    try std.testing.expect(is_eos);
}

test "step EOS detection multiple" {
    const eos_token_ids = [_]u32{ 0, 2, 128000, 128001 }; // Common LLM EOS tokens
    const tokens_to_test = [_]u32{ 0, 1, 2, 100, 128000, 128001, 999 };
    const expected_eos = [_]bool{ true, false, true, false, true, true, false };

    for (tokens_to_test, expected_eos) |token, expected| {
        var is_eos = false;
        for (eos_token_ids) |eos_id| {
            if (token == eos_id) {
                is_eos = true;
                break;
            }
        }
        try std.testing.expectEqual(expected, is_eos);
    }
}

test "step EOS detection empty list" {
    const eos_token_ids = [_]u32{};
    const token: u32 = 0;

    var is_eos = false;
    for (eos_token_ids) |eos_id| {
        if (token == eos_id) {
            is_eos = true;
            break;
        }
    }

    try std.testing.expect(!is_eos); // Should never be EOS with empty list
}

// =============================================================================
// Max Tokens Completion Tests
// =============================================================================

test "step max tokens exact" {
    const max_tokens: usize = 5;
    const generated_count: usize = 5;

    const is_at_limit = generated_count >= max_tokens;
    try std.testing.expect(is_at_limit);
}

test "step max tokens under limit" {
    const max_tokens: usize = 100;
    const generated_count: usize = 50;

    const is_at_limit = generated_count >= max_tokens;
    try std.testing.expect(!is_at_limit);
}

test "step max tokens edge case max=1" {
    const max_tokens: usize = 1;
    const is_final = max_tokens <= 1;
    try std.testing.expect(is_final);
}

// =============================================================================
// Greedy Sampling Tests (standalone logic)
// =============================================================================

test "step greedy sampling finds maximum" {
    const logits = [_]f32{ 1.0, 5.0, 2.0, 3.0, 0.5 };

    var max_index: usize = 0;
    var max_logit: f32 = logits[0];
    for (logits[1..], 1..) |logit, logit_idx| {
        if (logit > max_logit) {
            max_logit = logit;
            max_index = logit_idx;
        }
    }

    try std.testing.expectEqual(@as(usize, 1), max_index);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), max_logit, 0.001);
}

test "step greedy sampling negative logits" {
    const logits = [_]f32{ -10.0, -5.0, -1.0, -20.0 };

    var max_index: usize = 0;
    var max_logit: f32 = logits[0];
    for (logits[1..], 1..) |logit, logit_idx| {
        if (logit > max_logit) {
            max_logit = logit;
            max_index = logit_idx;
        }
    }

    try std.testing.expectEqual(@as(usize, 2), max_index); // -1.0 is the max
}

test "step greedy sampling all equal" {
    const logits = [_]f32{ 3.0, 3.0, 3.0, 3.0 };

    var max_index: usize = 0;
    var max_logit: f32 = logits[0];
    for (logits[1..], 1..) |logit, logit_idx| {
        if (logit > max_logit) {
            max_logit = logit;
            max_index = logit_idx;
        }
    }

    try std.testing.expectEqual(@as(usize, 0), max_index); // First occurrence
}

test "step greedy sampling single element" {
    const logits = [_]f32{42.0};

    var max_index: usize = 0;
    var max_logit: f32 = logits[0];
    for (logits[1..], 1..) |logit, logit_idx| {
        if (logit > max_logit) {
            max_logit = logit;
            max_index = logit_idx;
        }
    }

    try std.testing.expectEqual(@as(usize, 0), max_index);
}

// =============================================================================
// Temperature Scaling Tests (standalone logic)
// =============================================================================

test "step temperature scaling 1.0 identity" {
    const logits = [_]f32{ 1.0, 2.0, 3.0 };
    var scaled: [3]f32 = undefined;
    const temp: f32 = 1.0;

    for (logits, 0..) |l, i| {
        scaled[i] = l / temp;
    }

    for (logits, scaled) |orig, s| {
        try std.testing.expectApproxEqAbs(orig, s, 0.001);
    }
}

test "step temperature scaling low sharpens" {
    const logits = [_]f32{ 1.0, 2.0, 3.0 };
    var scaled: [3]f32 = undefined;
    const temp: f32 = 0.5;

    for (logits, 0..) |l, i| {
        scaled[i] = l / temp;
    }

    // With temp < 1, scaled values should be larger (sharper)
    try std.testing.expect(scaled[2] > logits[2]);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), scaled[2], 0.001);
}

test "step temperature scaling high flattens" {
    const logits = [_]f32{ 1.0, 2.0, 3.0 };
    var scaled: [3]f32 = undefined;
    const temp: f32 = 2.0;

    for (logits, 0..) |l, i| {
        scaled[i] = l / temp;
    }

    // With temp > 1, scaled values should be smaller (flatter)
    try std.testing.expect(scaled[2] < logits[2]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), scaled[2], 0.001);
}

// =============================================================================
// Softmax Tests (standalone logic)
// =============================================================================

test "step softmax sums to 1" {
    var logits = [_]f32{ 1.0, 2.0, 3.0 };

    // Find max for numerical stability
    var max_logit: f32 = logits[0];
    for (logits[1..]) |l| {
        if (l > max_logit) max_logit = l;
    }

    // Compute exp and sum
    var sum: f32 = 0.0;
    for (&logits) |*l| {
        l.* = @exp(l.* - max_logit);
        sum += l.*;
    }

    // Normalize
    for (&logits) |*l| {
        l.* /= sum;
    }

    // Verify sum is 1
    var total: f32 = 0.0;
    for (logits) |p| {
        total += p;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 0.001);
}

test "step softmax non-negative" {
    var logits = [_]f32{ -10.0, -5.0, 0.0, 5.0 };

    var max_logit: f32 = logits[0];
    for (logits[1..]) |l| {
        if (l > max_logit) max_logit = l;
    }

    var sum: f32 = 0.0;
    for (&logits) |*l| {
        l.* = @exp(l.* - max_logit);
        sum += l.*;
    }

    for (&logits) |*l| {
        l.* /= sum;
    }

    for (logits) |p| {
        try std.testing.expect(p >= 0.0);
    }
}

test "step softmax larger logits" {
    var logits = [_]f32{ 1.0, 2.0, 3.0 };

    var max_logit: f32 = logits[0];
    for (logits[1..]) |l| {
        if (l > max_logit) max_logit = l;
    }

    var sum: f32 = 0.0;
    for (&logits) |*l| {
        l.* = @exp(l.* - max_logit);
        sum += l.*;
    }

    for (&logits) |*l| {
        l.* /= sum;
    }

    // Verify ordering: prob[0] < prob[1] < prob[2]
    try std.testing.expect(logits[0] < logits[1]);
    try std.testing.expect(logits[1] < logits[2]);
}

// =============================================================================
// Top-K Filtering Tests (standalone logic)
// =============================================================================

test "step top-k filtering keeps top k" {
    const alloc = std.testing.allocator;
    var probs = [_]f32{ 0.1, 0.5, 0.2, 0.15, 0.05 };
    const k: usize = 2;

    // Copy and sort to find threshold
    const sorted = try alloc.alloc(f32, probs.len);
    defer alloc.free(sorted);
    @memcpy(sorted, &probs);
    std.mem.sort(f32, sorted, {}, std.sort.desc(f32));
    const threshold = sorted[k - 1]; // 0.5, 0.2 -> threshold = 0.2

    // Zero out below threshold
    var kept_count: usize = 0;
    for (&probs) |*p| {
        if (p.* < threshold) {
            p.* = 0.0;
        } else {
            kept_count += 1;
        }
    }

    // Should keep exactly k (or more if ties at threshold)
    try std.testing.expect(kept_count >= k);
    try std.testing.expect(probs[0] == 0.0); // 0.1 < 0.2
    try std.testing.expect(probs[1] > 0.0); // 0.5 >= 0.2
    try std.testing.expect(probs[2] > 0.0); // 0.2 >= 0.2
}

test "step top-k filtering k equals vocab" {
    const probs = [_]f32{ 0.25, 0.25, 0.25, 0.25 };

    // With k == len (4), threshold is the minimum value
    // All values should be kept
    var kept_count: usize = 0;
    for (probs) |p| {
        if (p > 0.0) kept_count += 1;
    }

    try std.testing.expectEqual(@as(usize, 4), kept_count);
}

test "step top-k filtering k=1 maximum" {
    const alloc = std.testing.allocator;
    var probs = [_]f32{ 0.1, 0.6, 0.2, 0.1 };
    const k: usize = 1;

    const sorted = try alloc.alloc(f32, probs.len);
    defer alloc.free(sorted);
    @memcpy(sorted, &probs);
    std.mem.sort(f32, sorted, {}, std.sort.desc(f32));
    const threshold = sorted[k - 1]; // 0.6

    for (&probs) |*p| {
        if (p.* < threshold) {
            p.* = 0.0;
        }
    }

    try std.testing.expect(probs[0] == 0.0);
    try std.testing.expect(probs[1] > 0.0); // Only 0.6 survives
    try std.testing.expect(probs[2] == 0.0);
    try std.testing.expect(probs[3] == 0.0);
}

// =============================================================================
// Cumulative Sum Sampling Tests (standalone logic)
// =============================================================================

test "step cumsum deterministic" {
    const probs = [_]f32{ 0.1, 0.3, 0.4, 0.2 };
    const r: f32 = 0.0;

    var cumsum: f32 = 0.0;
    var selected_idx: usize = probs.len - 1;
    for (probs, 0..) |prob, idx| {
        cumsum += prob;
        if (r < cumsum) {
            selected_idx = idx;
            break;
        }
    }

    try std.testing.expectEqual(@as(usize, 0), selected_idx); // First token
}

test "step cumsum probability ranges" {
    const probs = [_]f32{ 0.1, 0.3, 0.4, 0.2 };
    // Cumsum: 0.1, 0.4, 0.8, 1.0
    // r=0.05 -> idx 0
    // r=0.25 -> idx 1
    // r=0.6 -> idx 2
    // r=0.9 -> idx 3

    const test_cases = [_]struct { r: f32, expected_idx: usize }{
        .{ .r = 0.05, .expected_idx = 0 },
        .{ .r = 0.25, .expected_idx = 1 },
        .{ .r = 0.6, .expected_idx = 2 },
        .{ .r = 0.9, .expected_idx = 3 },
    };

    for (test_cases) |tc| {
        var cumsum: f32 = 0.0;
        var selected_idx: usize = probs.len - 1;
        for (probs, 0..) |prob, idx| {
            cumsum += prob;
            if (tc.r < cumsum) {
                selected_idx = idx;
                break;
            }
        }
        try std.testing.expectEqual(tc.expected_idx, selected_idx);
    }
}

test "step cumsum r=0.9999 selects last" {
    const probs = [_]f32{ 0.1, 0.3, 0.4, 0.2 };
    const r: f32 = 0.9999;

    var cumsum: f32 = 0.0;
    var selected_idx: usize = probs.len - 1;
    for (probs, 0..) |prob, idx| {
        cumsum += prob;
        if (r < cumsum) {
            selected_idx = idx;
            break;
        }
    }

    try std.testing.expectEqual(@as(usize, 3), selected_idx);
}

// =============================================================================
// Sampling Config Tests
// =============================================================================

test "init SamplingConfig greedy" {
    const config = sampling.SamplingConfig{
        .strategy = .greedy,
        .temperature = 0.5, // Should be ignored
    };

    // For greedy, we check strategy OR temperature == 0
    const is_greedy = config.strategy == .greedy or config.temperature == 0.0;
    try std.testing.expect(is_greedy);
}

test "init SamplingConfig temperature 0 greedy" {
    const config = sampling.SamplingConfig{
        .strategy = .top_k,
        .temperature = 0.0,
    };

    const is_greedy = config.strategy == .greedy or config.temperature == 0.0;
    try std.testing.expect(is_greedy);
}

test "init SamplingConfig valid top_k" {
    const configs = [_]sampling.SamplingConfig{
        .{ .strategy = .top_k, .top_k = 1 },
        .{ .strategy = .top_k, .top_k = 50 },
        .{ .strategy = .top_k, .top_k = 100 },
    };

    for (configs) |config| {
        try std.testing.expect(config.top_k > 0);
    }
}

// =============================================================================
// State Transition Logic Tests
// =============================================================================

test "step state transitions valid" {
    // Valid state flow: queued -> pending_prefill -> generating -> completed
    const flow = [_]RequestState{
        .queued,
        .pending_prefill,
        .generating,
        .completed,
    };

    for (flow[0 .. flow.len - 1], flow[1..]) |current, next| {
        // Verify transitions are to distinct states
        try std.testing.expect(current != next);
    }
}

test "cancel state transitions any active" {
    const active_states = [_]RequestState{
        .queued,
        .pending_prefill,
        .generating,
    };

    for (active_states) |state| {
        // All active states can transition to cancelled
        try std.testing.expect(state != .cancelled);
        try std.testing.expect(state != .completed);
        try std.testing.expect(state != .failed);
    }
}

// =============================================================================
// Mock Backend for Scheduler Tests
// =============================================================================

/// Minimal mock backend that implements just what the scheduler needs.
const MockBackend = struct {
    allocator: std.mem.Allocator,
    vocab_size: usize,
    max_batch_size: usize,
    slots_used: std.DynamicBitSet,
    prefill_calls: std.ArrayList(PrefillCall),
    decode_calls: std.ArrayList(DecodeCall),
    /// Track logits allocations for cleanup in tests
    allocated_logits: std.ArrayList([]f32),
    /// Token that always wins greedy sampling (default: 100)
    greedy_token: usize = 100,

    const PrefillCall = struct {
        slot_index: usize,
        tokens: []const u32,
    };

    const DecodeCall = struct {
        requests: []const DecodeRequest,
    };

    fn maxBatchSize(self: *const MockBackend) usize {
        return self.max_batch_size;
    }

    fn vocabSize(self: *const MockBackend) usize {
        return self.vocab_size;
    }

    fn init(allocator: std.mem.Allocator, vocab_size: usize, max_batch_size: usize) !MockBackend {
        var slots_used = try std.DynamicBitSet.initEmpty(allocator, max_batch_size);
        errdefer slots_used.deinit();

        return MockBackend{
            .allocator = allocator,
            .vocab_size = vocab_size,
            .max_batch_size = max_batch_size,
            .slots_used = slots_used,
            .prefill_calls = .{},
            .decode_calls = .{},
            .allocated_logits = .{},
            .greedy_token = 100,
        };
    }

    /// Set which token wins greedy sampling.
    fn setGreedyLogits(self: *MockBackend, token: usize) void {
        self.greedy_token = token;
    }

    fn deinit(self: *MockBackend) void {
        self.slots_used.deinit();
        for (self.prefill_calls.items) |call| {
            self.allocator.free(call.tokens);
        }
        self.prefill_calls.deinit(self.allocator);
        for (self.decode_calls.items) |call| {
            self.allocator.free(call.requests);
        }
        self.decode_calls.deinit(self.allocator);
        // Free tracked logits allocations
        for (self.allocated_logits.items) |logits| {
            self.allocator.free(logits);
        }
        self.allocated_logits.deinit(self.allocator);
    }

    fn allocSlot(self: *MockBackend) ?usize {
        var idx: usize = 0;
        while (idx < self.max_batch_size) : (idx += 1) {
            if (!self.slots_used.isSet(idx)) {
                self.slots_used.set(idx);
                return idx;
            }
        }
        return null;
    }

    fn freeSlot(self: *MockBackend, slot_index: usize) void {
        if (slot_index < self.max_batch_size) {
            self.slots_used.unset(slot_index);
        }
    }

    fn prefillSlot(self: *MockBackend, slot_index: usize, tokens: []const u32, logits_out: []f32) !void {
        // Record the call
        const tokens_copy = try self.allocator.dupe(u32, tokens);
        try self.prefill_calls.append(self.allocator, .{
            .slot_index = slot_index,
            .tokens = tokens_copy,
        });

        // Fill logits with dummy data (greedy_token has highest logit)
        for (logits_out, 0..) |*logit, idx| {
            logit.* = if (idx == self.greedy_token) 10.0 else 1.0;
        }
    }

    fn decodeBatch(self: *MockBackend, requests: []const DecodeRequest, results: []DecodeResult) !void {
        // Record the call
        const requests_copy = try self.allocator.dupe(DecodeRequest, requests);
        try self.decode_calls.append(self.allocator, .{
            .requests = requests_copy,
        });

        // Fill results with dummy logits
        for (requests, results) |request_entry, *result| {
            result.slot_index = request_entry.slot_index;

            // Create dummy logits buffer (tracked for cleanup)
            const logits = try self.allocator.alloc(f32, self.vocab_size);
            try self.allocated_logits.append(self.allocator, logits);
            for (logits, 0..) |*logit, logit_idx| {
                // greedy_token has highest logit
                logit.* = if (logit_idx == self.greedy_token) 10.0 else 1.0;
            }
            result.logits = logits;
        }
    }
};

/// MockScheduler for tests - Scheduler backed by MockBackend.
const MockScheduler = GenericScheduler(MockBackend);

/// Mock backend with decodeStreaming support for fast-path tests.
const MockStreamingBackend = struct {
    allocator: std.mem.Allocator,
    vocab_size: usize,
    slot_in_use: bool = false,
    decode_batch_calls: usize = 0,
    decode_streaming_calls: usize = 0,
    allocated_logits: std.ArrayList([]f32),
    greedy_token: usize = 42,

    fn init(allocator: std.mem.Allocator, vocab_size: usize) MockStreamingBackend {
        return .{
            .allocator = allocator,
            .vocab_size = vocab_size,
            .allocated_logits = .{},
        };
    }

    fn deinit(self: *MockStreamingBackend) void {
        for (self.allocated_logits.items) |logits| {
            self.allocator.free(logits);
        }
        self.allocated_logits.deinit(self.allocator);
    }

    fn maxBatchSize(self: *const MockStreamingBackend) usize {
        _ = self;
        return 1;
    }

    fn vocabSize(self: *const MockStreamingBackend) usize {
        return self.vocab_size;
    }

    fn supportsSchedulerStreamingFastPath(self: *const MockStreamingBackend) bool {
        _ = self;
        return true;
    }

    fn allocSlot(self: *MockStreamingBackend) ?usize {
        if (self.slot_in_use) return null;
        self.slot_in_use = true;
        return 0;
    }

    fn freeSlot(self: *MockStreamingBackend, slot_index: usize) void {
        _ = slot_index;
        self.slot_in_use = false;
    }

    fn prefillSlot(self: *MockStreamingBackend, slot_index: usize, tokens: []const u32, logits_out: []f32) !void {
        _ = self;
        _ = slot_index;
        _ = tokens;
        for (logits_out, 0..) |*logit, idx| {
            logit.* = if (idx == 42) 10.0 else 1.0;
        }
    }

    fn decodeBatch(self: *MockStreamingBackend, requests: []const DecodeRequest, results: []DecodeResult) !void {
        self.decode_batch_calls += 1;
        for (requests, results) |request_entry, *result| {
            result.slot_index = request_entry.slot_index;
            const logits = try self.allocator.alloc(f32, self.vocab_size);
            try self.allocated_logits.append(self.allocator, logits);
            for (logits, 0..) |*logit, idx| {
                logit.* = if (idx == self.greedy_token) 10.0 else 1.0;
            }
            result.logits = logits;
        }
    }

    fn decodeStreaming(
        self: *MockStreamingBackend,
        first_token: u32,
        start_position: usize,
        max_tokens: usize,
        eos_token_ids: []const u32,
        output_tokens: []u32,
        callback: ?*const fn (u32, ?*anyopaque) void,
        callback_data: ?*anyopaque,
    ) !usize {
        _ = eos_token_ids;
        _ = callback;
        _ = callback_data;
        self.decode_streaming_calls += 1;
        for (output_tokens[0..max_tokens], 0..) |*out_token, idx| {
            out_token.* = first_token + @as(u32, @intCast(start_position + idx + 1));
        }
        return max_tokens;
    }
};

const MockStreamingScheduler = GenericScheduler(MockStreamingBackend);

// =============================================================================
// Scheduler Integration Tests - init/deinit
// =============================================================================

test "generateSync uses decodeStreaming fast path when backend supports it" {
    const alloc = std.testing.allocator;
    var backend = MockStreamingBackend.init(alloc, 256);
    defer backend.deinit();

    var scheduler = try MockStreamingScheduler.init(alloc, &backend, .{
        .default_sampling = .{
            .strategy = .greedy,
        },
    });
    defer scheduler.deinit();

    var result = try scheduler.generateSync(&[_]u32{ 11, 12 }, 4, .{
        .sampling = .{
            .strategy = .greedy,
        },
        .eos_token_ids = &[_]u32{9999},
    });
    defer result.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 1), backend.decode_streaming_calls);
    try std.testing.expectEqual(@as(usize, 0), backend.decode_batch_calls);
    try std.testing.expectEqual(@as(usize, 4), result.tokens.len);
}

test "generateSync falls back to decodeBatch when stop sequences are configured" {
    const alloc = std.testing.allocator;
    var backend = MockStreamingBackend.init(alloc, 256);
    defer backend.deinit();

    var scheduler = try MockStreamingScheduler.init(alloc, &backend, .{
        .default_sampling = .{
            .strategy = .greedy,
        },
    });
    defer scheduler.deinit();

    const stop_seq = [_]u32{123_456};
    const stop_sequences = [_][]const u32{&stop_seq};

    var result = try scheduler.generateSync(&[_]u32{5}, 3, .{
        .sampling = .{
            .strategy = .greedy,
        },
        .stop_sequences = &stop_sequences,
        .eos_token_ids = &[_]u32{9999},
    });
    defer result.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 0), backend.decode_streaming_calls);
    try std.testing.expect(backend.decode_batch_calls > 0);
    try std.testing.expectEqual(@as(usize, 3), result.tokens.len);
}

test "Scheduler.init - creates with default config" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    try std.testing.expectEqual(@as(usize, 1000), scheduler.backend.vocab_size);
    try std.testing.expectEqual(@as(usize, 4), scheduler.backend.max_batch_size);
    try std.testing.expectEqual(@as(u64, 1), scheduler.next_request_id);
    try std.testing.expectEqual(@as(usize, 0), scheduler.requests.count());
}

test "Scheduler.init - respects max_concurrent config" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 10);
    defer backend.deinit();

    const config = SchedulerConfig{
        .max_concurrent = 5,
    };

    var scheduler = try MockScheduler.init(alloc, &backend, config);
    defer scheduler.deinit();

    // Verify buffers are sized for max_concurrent, not backend max
    try std.testing.expectEqual(@as(usize, 5), scheduler.decode_requests.len);
    try std.testing.expectEqual(@as(usize, 5), scheduler.decode_results.len);
}

test "Scheduler.init - uses backend max when max_concurrent is null" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 8);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    try std.testing.expectEqual(@as(usize, 8), scheduler.decode_requests.len);
}

test "Scheduler.deinit - cleans up all resources" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});

    // Submit a request to create some internal state
    const prompt = [_]u32{ 1, 2, 3 };
    _ = try scheduler.submit(&prompt, 10, null);

    // Deinit should clean everything up without leaks
    scheduler.deinit();

    // Test passes if no memory leaks detected by testing.allocator
}

// =============================================================================
// Scheduler Integration Tests - submit
// =============================================================================

test "Scheduler.submit - creates request with default options" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const request_id = try scheduler.submit(&prompt, 10, null);

    try std.testing.expectEqual(@as(u64, 1), request_id);
    try std.testing.expectEqual(@as(usize, 1), scheduler.requests.count());

    const request_entry = scheduler.requests.get(request_id).?;
    try std.testing.expectEqual(@as(usize, 3), request_entry.prompt_tokens.len);
    try std.testing.expectEqual(@as(usize, 10), request_entry.max_tokens);
}

test "Scheduler.submit - activates immediately when slots available" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const request_id = try scheduler.submit(&prompt, 10, null);

    const request_entry = scheduler.requests.get(request_id).?;

    // Should be activated immediately
    try std.testing.expectEqual(RequestState.pending_prefill, request_entry.state);
    try std.testing.expect(request_entry.slot_index != null);
    try std.testing.expectEqual(@as(usize, 1), scheduler.activeCount());
    try std.testing.expectEqual(@as(usize, 0), scheduler.pendingCount());
}

test "Scheduler.submit - queues when slots full" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 2);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };

    // Fill all slots
    _ = try scheduler.submit(&prompt, 10, null);
    _ = try scheduler.submit(&prompt, 10, null);

    // This should queue
    const queued_id = try scheduler.submit(&prompt, 10, null);
    const request_entry = scheduler.requests.get(queued_id).?;

    try std.testing.expectEqual(RequestState.queued, request_entry.state);
    try std.testing.expectEqual(@as(?usize, null), request_entry.slot_index);
    try std.testing.expectEqual(@as(usize, 2), scheduler.activeCount());
    try std.testing.expectEqual(@as(usize, 1), scheduler.pendingCount());
}

test "Scheduler.submit - increments request ID" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };

    const id1 = try scheduler.submit(&prompt, 10, null);
    const id2 = try scheduler.submit(&prompt, 10, null);
    const id3 = try scheduler.submit(&prompt, 10, null);

    try std.testing.expectEqual(@as(u64, 1), id1);
    try std.testing.expectEqual(@as(u64, 2), id2);
    try std.testing.expectEqual(@as(u64, 3), id3);
}

test "Scheduler.submit - custom options" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const eos_token_ids = [_]u32{ 0, 128001 };

    const opts = Scheduler.SubmitOptions{
        .eos_token_ids = &eos_token_ids,
        .priority = 100,
        .sampling = .{
            .strategy = .top_k,
            .temperature = 0.8,
            .top_k = 50,
        },
    };

    const request_id = try scheduler.submit(&prompt, 10, opts);
    const request_entry = scheduler.requests.get(request_id).?;

    try std.testing.expectEqual(@as(usize, 2), request_entry.eos_token_ids.len);
    try std.testing.expectEqual(@as(i32, 100), request_entry.priority);
    try std.testing.expectEqual(sampling.SamplingStrategy.top_k, request_entry.sampling_config.strategy);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), request_entry.sampling_config.temperature, 0.001);
}

// =============================================================================
// Scheduler Integration Tests - cancel
// =============================================================================

test "Scheduler.cancel - queued request" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 1);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };

    // First request takes the slot
    _ = try scheduler.submit(&prompt, 10, null);

    // Second request queues
    const queued_id = try scheduler.submit(&prompt, 10, null);

    // Cancel the queued request
    const cancelled = scheduler.cancel(queued_id);

    try std.testing.expect(cancelled);

    const request_entry = scheduler.requests.get(queued_id).?;
    try std.testing.expectEqual(RequestState.cancelled, request_entry.state);
    try std.testing.expectEqual(@as(usize, 0), scheduler.pendingCount());
    try std.testing.expectEqual(@as(usize, 1), scheduler.completed_queue.items.len);
}

test "Scheduler.cancel - active request frees slot" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 2);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const request_id = try scheduler.submit(&prompt, 10, null);

    const request_entry = scheduler.requests.get(request_id).?;
    const slot_index = request_entry.slot_index.?;

    // Cancel should free the slot
    const cancelled = scheduler.cancel(request_id);

    try std.testing.expect(cancelled);
    try std.testing.expectEqual(RequestState.cancelled, request_entry.state);
    try std.testing.expectEqual(@as(usize, 0), scheduler.activeCount());

    // Slot should be free (verify by checking backend)
    try std.testing.expect(!backend.slots_used.isSet(slot_index));
}

test "Scheduler.cancel - nonexistent request" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const cancelled = scheduler.cancel(999);

    try std.testing.expect(!cancelled);
}

test "Scheduler.cancel - already completed request" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const request_id = try scheduler.submit(&prompt, 10, null);

    const request_entry = scheduler.requests.get(request_id).?;
    request_entry.state = .completed;

    const cancelled = scheduler.cancel(request_id);

    try std.testing.expect(!cancelled);
    try std.testing.expectEqual(RequestState.completed, request_entry.state);
}

test "Scheduler.cancel - already cancelled request" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const request_id = try scheduler.submit(&prompt, 10, null);

    // Cancel once
    _ = scheduler.cancel(request_id);

    // Try to cancel again
    const cancelled_again = scheduler.cancel(request_id);

    try std.testing.expect(!cancelled_again);
}

// =============================================================================
// Scheduler Integration Tests - getState
// =============================================================================

test "Scheduler.getState - returns correct state" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const request_id = try scheduler.submit(&prompt, 10, null);

    const state = scheduler.getState(request_id);

    try std.testing.expect(state != null);
    try std.testing.expectEqual(RequestState.pending_prefill, state.?);
}

test "Scheduler.getState - nonexistent request" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const state = scheduler.getState(999);

    try std.testing.expect(state == null);
}

test "Scheduler.getState - tracks state changes" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const request_id = try scheduler.submit(&prompt, 10, null);

    // Initial state
    try std.testing.expectEqual(RequestState.pending_prefill, scheduler.getState(request_id).?);

    // Change state manually
    const request_entry = scheduler.requests.get(request_id).?;
    request_entry.state = .generating;

    try std.testing.expectEqual(RequestState.generating, scheduler.getState(request_id).?);
}

// =============================================================================
// Scheduler Integration Tests - getGeneratedTokens
// =============================================================================

test "Scheduler.getGeneratedTokens - empty initially" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const request_id = try scheduler.submit(&prompt, 10, null);

    const tokens = scheduler.getGeneratedTokens(request_id);

    try std.testing.expect(tokens != null);
    try std.testing.expectEqual(@as(usize, 0), tokens.?.len);
}

test "Scheduler.getGeneratedTokens - returns generated tokens" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const request_id = try scheduler.submit(&prompt, 10, null);

    // Simulate generating tokens
    const request_entry = scheduler.requests.get(request_id).?;
    try request_entry.generated_tokens.append(alloc, 100);
    try request_entry.generated_tokens.append(alloc, 101);
    try request_entry.generated_tokens.append(alloc, 102);

    const tokens = scheduler.getGeneratedTokens(request_id);

    try std.testing.expect(tokens != null);
    try std.testing.expectEqual(@as(usize, 3), tokens.?.len);
    try std.testing.expectEqual(@as(u32, 100), tokens.?[0]);
    try std.testing.expectEqual(@as(u32, 101), tokens.?[1]);
    try std.testing.expectEqual(@as(u32, 102), tokens.?[2]);
}

test "Scheduler.getGeneratedTokens - nonexistent request" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const tokens = scheduler.getGeneratedTokens(999);

    try std.testing.expect(tokens == null);
}

// =============================================================================
// Scheduler Integration Tests - hasActive
// =============================================================================

test "Scheduler.hasActive - false when empty" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    try std.testing.expect(!scheduler.hasActive());
}

test "Scheduler.hasActive - true with active request" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    _ = try scheduler.submit(&prompt, 10, null);

    try std.testing.expect(scheduler.hasActive());
}

test "Scheduler.hasActive - true with pending request" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 1);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };

    // First takes slot, second queues
    _ = try scheduler.submit(&prompt, 10, null);
    _ = try scheduler.submit(&prompt, 10, null);

    try std.testing.expect(scheduler.hasActive());
}

test "Scheduler.hasActive - false after cancelling all" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const request_id = try scheduler.submit(&prompt, 10, null);

    _ = scheduler.cancel(request_id);

    try std.testing.expect(!scheduler.hasActive());
}

// =============================================================================
// Scheduler Integration Tests - activeCount/pendingCount
// =============================================================================

test "Scheduler.activeCount - counts active requests" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };

    try std.testing.expectEqual(@as(usize, 0), scheduler.activeCount());

    _ = try scheduler.submit(&prompt, 10, null);
    try std.testing.expectEqual(@as(usize, 1), scheduler.activeCount());

    _ = try scheduler.submit(&prompt, 10, null);
    try std.testing.expectEqual(@as(usize, 2), scheduler.activeCount());
}

test "Scheduler.pendingCount - counts queued requests" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 2);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };

    try std.testing.expectEqual(@as(usize, 0), scheduler.pendingCount());

    // Fill slots
    _ = try scheduler.submit(&prompt, 10, null);
    _ = try scheduler.submit(&prompt, 10, null);
    try std.testing.expectEqual(@as(usize, 0), scheduler.pendingCount());

    // Queue requests
    _ = try scheduler.submit(&prompt, 10, null);
    try std.testing.expectEqual(@as(usize, 1), scheduler.pendingCount());

    _ = try scheduler.submit(&prompt, 10, null);
    try std.testing.expectEqual(@as(usize, 2), scheduler.pendingCount());
}

test "Scheduler.activeCount and pendingCount - independent counts" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 2);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };

    _ = try scheduler.submit(&prompt, 10, null);
    _ = try scheduler.submit(&prompt, 10, null);
    _ = try scheduler.submit(&prompt, 10, null);

    try std.testing.expectEqual(@as(usize, 2), scheduler.activeCount());
    try std.testing.expectEqual(@as(usize, 1), scheduler.pendingCount());
}

// =============================================================================
// Scheduler Integration Tests - step
// =============================================================================

test "Scheduler.step - empty scheduler returns no events" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const events = try scheduler.step();
    defer alloc.free(events);

    try std.testing.expectEqual(@as(usize, 0), events.len);
}

test "Scheduler.step - runs prefill for new requests" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const request_id = try scheduler.submit(&prompt, 10, null);

    // Step should run prefill
    const events = try scheduler.step();
    defer alloc.free(events);

    // Should have called prefillSlot
    try std.testing.expectEqual(@as(usize, 1), backend.prefill_calls.items.len);

    const request_entry = scheduler.requests.get(request_id).?;
    try std.testing.expectEqual(RequestState.generating, request_entry.state);
}

test "Scheduler.step - generates tokens for active requests" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const request_id = try scheduler.submit(&prompt, 10, null);

    // First step: prefill
    {
        const events = try scheduler.step();
        defer alloc.free(events);
    }

    // Second step: decode
    const events = try scheduler.step();
    defer alloc.free(events);
    // Logits are now tracked and freed by MockBackend.deinit()

    try std.testing.expectEqual(@as(usize, 1), events.len);
    try std.testing.expectEqual(request_id, events[0].request_id);

    const request_entry = scheduler.requests.get(request_id).?;
    try std.testing.expect(request_entry.generated_tokens.items.len > 0);
}

test "Scheduler.step - handles EOS token" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const eos_token_ids = [_]u32{100}; // Token 100 is EOS

    const opts = Scheduler.SubmitOptions{
        .eos_token_ids = &eos_token_ids,
    };

    const request_id = try scheduler.submit(&prompt, 10, opts);

    // First step runs prefill, which generates token 100 (EOS)
    const events = try scheduler.step();
    defer alloc.free(events);

    // Request should be completed
    const request_entry = scheduler.requests.get(request_id).?;
    try std.testing.expectEqual(RequestState.completed, request_entry.state);
    try std.testing.expectEqual(@as(usize, 0), scheduler.activeCount());
}

test "Scheduler.step - handles max tokens" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const request_id = try scheduler.submit(&prompt, 1, null); // max_tokens = 1

    // First step runs prefill and generates 1 token (max reached)
    const events = try scheduler.step();
    defer alloc.free(events);

    const request_entry = scheduler.requests.get(request_id).?;
    try std.testing.expectEqual(RequestState.completed, request_entry.state);
    try std.testing.expectEqual(@as(usize, 1), request_entry.generated_tokens.items.len);
}

test "Scheduler.step - batches multiple decode requests" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const eos_token_ids = [_]u32{999}; // Won't hit EOS
    const opts = Scheduler.SubmitOptions{
        .eos_token_ids = &eos_token_ids,
    };

    _ = try scheduler.submit(&prompt, 10, opts);
    _ = try scheduler.submit(&prompt, 10, opts);

    // Run prefills
    {
        const events = try scheduler.step();
        defer alloc.free(events);
    }

    // Run decode step - should batch both
    const events = try scheduler.step();
    defer alloc.free(events);
    // Logits are now tracked and freed by MockBackend.deinit()

    // Should have called decodeBatch with 2 requests
    try std.testing.expect(backend.decode_calls.items.len > 0);
    const last_batch = backend.decode_calls.items[backend.decode_calls.items.len - 1];
    try std.testing.expectEqual(@as(usize, 2), last_batch.requests.len);
}

test "Scheduler.step - activates pending after slot freed" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 1);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const eos_token_ids = [_]u32{100}; // Will hit EOS immediately

    const opts = Scheduler.SubmitOptions{
        .eos_token_ids = &eos_token_ids,
    };

    // First request takes slot and will complete
    _ = try scheduler.submit(&prompt, 10, opts);

    // Second request queues
    const queued_id = try scheduler.submit(&prompt, 10, null);

    try std.testing.expectEqual(@as(usize, 1), scheduler.pendingCount());

    // Step completes first request and should activate queued
    const events = try scheduler.step();
    defer alloc.free(events);

    try std.testing.expectEqual(@as(usize, 0), scheduler.pendingCount());

    const request_entry = scheduler.requests.get(queued_id).?;
    try std.testing.expect(request_entry.state != .queued);
}

// =============================================================================
// Scheduler Integration Tests - popCompleted
// =============================================================================

test "Scheduler.popCompleted - empty when no completions" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const completed_ids = scheduler.popCompleted();
    defer alloc.free(completed_ids);

    try std.testing.expectEqual(@as(usize, 0), completed_ids.len);
}

test "Scheduler.popCompleted - returns completed request IDs" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const eos_token_ids = [_]u32{100};
    const opts = Scheduler.SubmitOptions{
        .eos_token_ids = &eos_token_ids,
    };

    const request_id = try scheduler.submit(&prompt, 10, opts);

    // Complete the request
    _ = try scheduler.step();

    const completed_ids = scheduler.popCompleted();
    defer alloc.free(completed_ids);

    try std.testing.expectEqual(@as(usize, 1), completed_ids.len);
    try std.testing.expectEqual(request_id, completed_ids[0]);
}

test "Scheduler.popCompleted - clears completed queue" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const eos_token_ids = [_]u32{100};
    const opts = Scheduler.SubmitOptions{
        .eos_token_ids = &eos_token_ids,
    };

    _ = try scheduler.submit(&prompt, 10, opts);
    _ = try scheduler.step();

    // First pop
    {
        const completed_ids = scheduler.popCompleted();
        defer alloc.free(completed_ids);
        try std.testing.expectEqual(@as(usize, 1), completed_ids.len);
    }

    // Second pop should be empty
    {
        const completed_ids = scheduler.popCompleted();
        defer alloc.free(completed_ids);
        try std.testing.expectEqual(@as(usize, 0), completed_ids.len);
    }
}

test "Scheduler.popCompleted - includes cancelled requests" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const request_id = try scheduler.submit(&prompt, 10, null);

    _ = scheduler.cancel(request_id);

    const completed_ids = scheduler.popCompleted();
    defer alloc.free(completed_ids);

    try std.testing.expectEqual(@as(usize, 1), completed_ids.len);
    try std.testing.expectEqual(request_id, completed_ids[0]);
}

// =============================================================================
// Scheduler Integration Tests - remove
// =============================================================================

test "Scheduler.remove - frees request resources" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const request_id = try scheduler.submit(&prompt, 10, null);

    try std.testing.expectEqual(@as(usize, 1), scheduler.requests.count());

    scheduler.remove(request_id);

    try std.testing.expectEqual(@as(usize, 0), scheduler.requests.count());
    try std.testing.expect(scheduler.requests.get(request_id) == null);
}

test "Scheduler.remove - nonexistent request is no-op" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    // Should not crash
    scheduler.remove(999);

    try std.testing.expectEqual(@as(usize, 0), scheduler.requests.count());
}

test "Scheduler.remove - can remove after completion" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const eos_token_ids = [_]u32{100};
    const opts = Scheduler.SubmitOptions{
        .eos_token_ids = &eos_token_ids,
    };

    const request_id = try scheduler.submit(&prompt, 10, opts);

    // Complete
    _ = try scheduler.step();

    // Pop completed
    const completed_ids = scheduler.popCompleted();
    defer alloc.free(completed_ids);

    // Remove
    scheduler.remove(request_id);

    try std.testing.expectEqual(@as(usize, 0), scheduler.requests.count());
}

// =============================================================================
// Scheduler Priority Scheduling Tests
// =============================================================================

test "step FIFO order" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 1);
    defer backend.deinit();

    const config = SchedulerConfig{
        .priority_scheduling = false,
    };

    var scheduler = try MockScheduler.init(alloc, &backend, config);
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };

    // All will queue (only 1 slot)
    const id1 = try scheduler.submit(&prompt, 10, .{ .priority = 0 });
    const id2 = try scheduler.submit(&prompt, 10, .{ .priority = 100 }); // Higher priority
    const id3 = try scheduler.submit(&prompt, 10, .{ .priority = 50 });

    // Cancel first to free slot
    _ = scheduler.cancel(id1);

    // Should activate in FIFO order (id2), not priority order
    try scheduler.activatePending();

    const req2 = scheduler.requests.get(id2).?;
    const req3 = scheduler.requests.get(id3).?;

    try std.testing.expect(req2.state != .queued);
    try std.testing.expectEqual(RequestState.queued, req3.state);
}

test "step priority order" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 1);
    defer backend.deinit();

    const config = SchedulerConfig{
        .priority_scheduling = true,
    };

    var scheduler = try MockScheduler.init(alloc, &backend, config);
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };

    // All will queue (only 1 slot)
    const id1 = try scheduler.submit(&prompt, 10, .{ .priority = 0 });
    const id2 = try scheduler.submit(&prompt, 10, .{ .priority = 10 });
    const id3 = try scheduler.submit(&prompt, 10, .{ .priority = 100 }); // Highest

    // Cancel first to free slot
    _ = scheduler.cancel(id1);

    // Should activate highest priority (id3)
    try scheduler.activatePending();

    const req2 = scheduler.requests.get(id2).?;
    const req3 = scheduler.requests.get(id3).?;

    try std.testing.expectEqual(RequestState.queued, req2.state);
    try std.testing.expect(req3.state != .queued);
}

// =============================================================================
// Scheduler Edge Cases
// =============================================================================

test "submit empty prompt" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{};
    const request_id = try scheduler.submit(&prompt, 10, null);

    const request_entry = scheduler.requests.get(request_id).?;
    try std.testing.expectEqual(@as(usize, 0), request_entry.prompt_tokens.len);
}

test "submit max_tokens zero" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const request_id = try scheduler.submit(&prompt, 0, null);

    const request_entry = scheduler.requests.get(request_id).?;
    try std.testing.expectEqual(@as(usize, 0), request_entry.max_tokens);
}

test "step callback invocation" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const CallbackData = struct {
        called: bool = false,
        token: u32 = 0,
        is_final: bool = false,
    };

    var callback_data = CallbackData{};

    const callback = struct {
        fn cb(request_id: u64, token: u32, is_final: bool, user_data: ?*anyopaque) void {
            _ = request_id;
            const data: *CallbackData = @ptrCast(@alignCast(user_data.?));
            data.called = true;
            data.token = token;
            data.is_final = is_final;
        }
    }.cb;

    const prompt = [_]u32{ 1, 2, 3 };
    const eos_token_ids = [_]u32{100};

    const opts = Scheduler.SubmitOptions{
        .callback = callback,
        .callback_data = &callback_data,
        .eos_token_ids = &eos_token_ids,
    };

    _ = try scheduler.submit(&prompt, 10, opts);

    // Step should invoke callback
    _ = try scheduler.step();

    try std.testing.expect(callback_data.called);
    try std.testing.expectEqual(@as(u32, 100), callback_data.token);
    try std.testing.expect(callback_data.is_final);
}

test "generateSync fast path - basic generation" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };

    // generateSync should use fast path when no other requests active
    var result = try scheduler.generateSync(&prompt, 5, null);
    defer result.deinit(alloc);

    // Should generate up to max_tokens
    try std.testing.expectEqual(@as(usize, 5), result.tokens.len);
    try std.testing.expectEqual(FinishReason.length, result.finish_reason);
}

test "generateSync fast path - EOS detection" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    // Set logits so token 100 (EOS) wins every time
    backend.setGreedyLogits(100);

    var scheduler = try MockScheduler.init(alloc, &backend, .{
        .default_eos_token_ids = &[_]u32{100},
    });
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    var result = try scheduler.generateSync(&prompt, 50, null);
    defer result.deinit(alloc);

    // Should stop at EOS
    try std.testing.expectEqual(@as(usize, 1), result.tokens.len);
    try std.testing.expectEqual(@as(u32, 100), result.tokens[0]);
    try std.testing.expectEqual(FinishReason.eos_token, result.finish_reason);
}

test "generateSync fast path - stop sequences" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const stop_seq = [_]u32{ 100, 100, 100 };
    const stop_sequences = [_][]const u32{&stop_seq};

    var result = try scheduler.generateSync(&prompt, 50, .{
        .stop_sequences = &stop_sequences,
    });
    defer result.deinit(alloc);

    // Generation should stop when stop sequence is matched
    // Tokens should not include the stop sequence
    try std.testing.expect(result.tokens.len < 50);
    try std.testing.expectEqual(FinishReason.stop_sequence, result.finish_reason);
}

test "generateSync fast path - callback invocation" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const CallbackData = struct {
        token_count: usize = 0,
        final_token: u32 = 0,
    };

    var callback_data = CallbackData{};

    const callback = struct {
        fn cb(_: u64, token: u32, _: bool, user_data: ?*anyopaque) void {
            const data: *CallbackData = @ptrCast(@alignCast(user_data.?));
            data.token_count += 1;
            data.final_token = token;
        }
    }.cb;

    const prompt = [_]u32{ 1, 2, 3 };
    var result = try scheduler.generateSync(&prompt, 5, .{
        .callback = callback,
        .callback_data = &callback_data,
    });
    defer result.deinit(alloc);

    // Callback should be invoked for each generated token
    try std.testing.expectEqual(@as(usize, 5), callback_data.token_count);
}

test "generateSync slow path - used when other requests active" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    // Set EOS so generation completes
    backend.setGreedyLogits(100);

    var scheduler = try MockScheduler.init(alloc, &backend, .{
        .default_eos_token_ids = &[_]u32{100},
    });
    defer scheduler.deinit();

    const prompt1 = [_]u32{ 1, 2, 3 };
    const prompt2 = [_]u32{ 4, 5, 6 };

    // Submit first request (this should use fast path internally but let's verify slow path too)
    _ = try scheduler.submit(&prompt1, 5, null);

    // Now scheduler has active requests, generateSync should use slow path
    var result = try scheduler.generateSync(&prompt2, 5, null);
    defer result.deinit(alloc);

    // Should still work correctly
    try std.testing.expect(result.tokens.len > 0);
}

test "generateSync fast path - seed produces deterministic output" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    // Configure for sampling mode (not greedy) to test seed effect
    var scheduler = try MockScheduler.init(alloc, &backend, .{
        .default_sampling = .{
            .strategy = .top_k,
            .temperature = 1.0,
            .top_k = 10,
        },
    });
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const test_seed: u64 = 42;

    // First generation with seed
    var result1 = try scheduler.generateSync(&prompt, 5, .{
        .sampling = .{
            .strategy = .top_k,
            .temperature = 1.0,
            .top_k = 10,
            .seed = test_seed,
        },
    });
    defer result1.deinit(alloc);

    // Second generation with same seed - should produce identical output
    var result2 = try scheduler.generateSync(&prompt, 5, .{
        .sampling = .{
            .strategy = .top_k,
            .temperature = 1.0,
            .top_k = 10,
            .seed = test_seed,
        },
    });
    defer result2.deinit(alloc);

    // Same seed must produce same tokens (fundamental determinism invariant)
    try std.testing.expectEqualSlices(u32, result1.tokens, result2.tokens);
}
