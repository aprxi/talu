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
const log = @import("log_pkg");
const validate = @import("validate_pkg");
const runtime_contract = @import("runtime_contract_pkg");
const trace = @import("xray_pkg").trace;
const xray = @import("xray_pkg");

fn generationShouldStop(stop_flag: ?*const std.atomic.Value(bool)) bool {
    if (stop_flag) |flag| {
        if (flag.load(.acquire)) return true;
    }
    return xray.isVerifyStopRequested();
}

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
    /// Signature: fn(request_id: u64, token: u32, is_final: bool, in_thinking: bool, user_data: ?*anyopaque) void
    callback: ?*const fn (u64, u32, bool, bool, ?*anyopaque) void,
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
    /// Time spent in prefill for this request.
    prefill_ns: u64 = 0,
    /// Time spent in decode steps for this request.
    decode_ns: u64 = 0,
    /// Whether to capture final-step logits for this request.
    capture_final_logits: bool = false,
    /// Captured final-step logits (owned when non-empty).
    final_logits: []f32 = &.{},

    // -- Thinking budget enforcement (queued route) --
    /// Maximum thinking tokens before force-injecting end sequence. 0 = no limit.
    max_thinking_tokens: usize = 0,
    /// Token IDs to inject when thinking budget is exceeded (e.g. </think>\n\n).
    thinking_end_tokens: []const u32 = &.{},
    /// Number of thinking tokens generated so far.
    thinking_token_count: usize = 0,
    /// Whether the model is currently in thinking mode.
    in_thinking: bool = false,
    /// Position within thinking_end_tokens being injected (0 = not injecting).
    thinking_inject_pos: usize = 0,
    /// Maximum answer tokens after thinking ends. 0 = no limit.
    max_completion_tokens_limit: usize = 0,
    /// Number of answer tokens generated so far.
    completion_token_count: usize = 0,
    /// Whether we have transitioned to answer generation.
    generating_answer: bool = true,

    pub fn deinit(self: *Request, alloc: std.mem.Allocator) void {
        self.generated_tokens.deinit(alloc);
        if (self.final_logits.len > 0) {
            alloc.free(self.final_logits);
        }
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
    /// Whether the token was generated during thinking/reasoning
    in_thinking: bool,
    /// Slot index (for debugging)
    slot_index: usize,
    /// Wall-clock timestamp when the token event was produced.
    timestamp_ns: i128 = 0,
};

/// Tokenizer view used by grammar-constrained sampling without binding the
/// scheduler to a concrete tokenizer module type.
pub const TokenizerView = struct {
    context: ?*const anyopaque,
    token_bytes_fn: *const fn (?*const anyopaque, usize) ?[]const u8,
    vocab_size_fn: *const fn (?*const anyopaque) usize,

    pub fn tokenBytes(self: *const TokenizerView, token_id: usize) ?[]const u8 {
        return self.token_bytes_fn(self.context, token_id);
    }

    pub fn getVocabSize(self: *const TokenizerView) usize {
        return self.vocab_size_fn(self.context);
    }

    pub fn fromTokenizer(tokenizer_ptr: anytype) TokenizerView {
        const Ptr = @TypeOf(tokenizer_ptr);
        const ptr_info = @typeInfo(Ptr);
        if (ptr_info != .pointer) {
            @compileError("TokenizerView.fromTokenizer expects a pointer");
        }
        const TokenizerType = ptr_info.pointer.child;

        const token_bytes_fn = struct {
            fn call(ctx: ?*const anyopaque, token_id: usize) ?[]const u8 {
                const typed: *const TokenizerType = @ptrCast(@alignCast(ctx orelse return null));
                return typed.tokenBytes(token_id);
            }
        }.call;

        const vocab_size_fn = struct {
            fn call(ctx: ?*const anyopaque) usize {
                const typed: *const TokenizerType = @ptrCast(@alignCast(ctx orelse return 0));
                if (@hasDecl(TokenizerType, "getVocabSize")) {
                    return typed.getVocabSize();
                }
                if (@hasField(TokenizerType, "vocab_size")) {
                    return typed.vocab_size;
                }
                if (@hasField(TokenizerType, "tokenizer_handle")) {
                    return typed.tokenizer_handle.getVocabSize();
                }
                @compileError("Tokenizer type must provide getVocabSize(), vocab_size, or tokenizer_handle.getVocabSize()");
            }
        }.call;

        return .{
            .context = @ptrCast(tokenizer_ptr),
            .token_bytes_fn = token_bytes_fn,
            .vocab_size_fn = vocab_size_fn,
        };
    }
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
    tokenizer: ?TokenizerView = null,
    /// Plan-owned state descriptor contract for scheduler allocation.
    state_descriptors: []const runtime_contract.StateDescriptor = &.{},
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
/// - `bindSlotStateBlocks(*T, usize, []const runtime_contract.StateBlockHandle) !void` - bind opaque slot state
/// - `unbindSlotStateBlocks(*T, usize) void` - unbind opaque slot state
/// - `prefillSlot(*T, usize, []const u32, []f32) !void` - prefill
/// - optional: `prefillSlotWithVision(*T, usize, []const u32, ?*const PrefillVisionInput, []f32) !void`
/// - optional: `prefillBatch(*T, []const contract.PrefillBatchRequest) !void`
/// - optional: `supportsSchedulerBackendDecodeStreamingRoute(*const T) bool`
/// - optional: `decodeStreaming(*T, u32, usize, usize, []const u32, []u32, ?*const fn (u32, ?*anyopaque) void, ?*anyopaque) !usize`
/// - `decodeBatch(*T, []const DecodeRequest, []DecodeResult) !void` - batch decode
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
        /// Direct slot -> request lookup to avoid per-step slot scans.
        slot_request_ids: []?u64,

        /// Completed request IDs (ready for retrieval)
        completed_queue: std.ArrayList(u64),

        /// Next request ID
        next_request_id: u64,

        /// Scratch buffers for batched decode
        decode_requests: []DecodeRequest,
        decode_results: []DecodeResult,
        decode_candidate_logits: []f32,
        decode_candidate_ids: []u32,
        decode_candidate_counts: []usize,
        /// Logits buffer for prefill
        logits_buffer: []f32,
        /// Scratch buffers for optional batched prefill route.
        prefill_requests: []contract.PrefillBatchRequest,
        prefill_request_ids: []u64,
        prefill_logits_buffer: []f32,

        /// Optimized sampler with SIMD and pre-allocated workspace
        sampler: sampling.Sampler,
        /// Scheduler-owned opaque state blocks per active request.
        request_state_blocks: std.AutoHashMap(u64, RequestStateBlocks),
        /// Scheduler-owned slot-persistent state blocks keyed by slot index.
        slot_state_blocks: std.AutoHashMap(usize, RequestStateBlocks),
        /// Immutable descriptor set for this scheduler instance.
        state_descriptors: []runtime_contract.StateDescriptor,
        /// Slot-persistent descriptors (subset of state_descriptors).
        slot_persistent_descs: []runtime_contract.StateDescriptor,
        /// Non-persistent descriptors: request_scoped + step_scoped (subset of state_descriptors).
        non_persistent_descs: []runtime_contract.StateDescriptor,

        /// Persistent buffer for step() token events. Avoids heap alloc/free per step.
        /// Cleared at the start of each step; valid until the next step() call.
        step_events: std.ArrayList(TokenEvent) = .{},

        const StateBlockStorage = struct {
            bytes: []align(64) u8,
        };

        const RequestStateBlocks = struct {
            handles: []runtime_contract.StateBlockHandle = &.{},
            storage: []StateBlockStorage = &.{},
            descriptors: []runtime_contract.StateDescriptor = &.{},

            fn deinit(self: *RequestStateBlocks, allocator: std.mem.Allocator) void {
                for (self.storage) |entry| allocator.free(entry.bytes);
                if (self.storage.len > 0) allocator.free(self.storage);
                if (self.handles.len > 0) allocator.free(self.handles);
                if (self.descriptors.len > 0) allocator.free(self.descriptors);
                self.* = .{};
            }
        };

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

            const decode_candidate_logits = try allocator.alloc(f32, effective_batch_size * 256);
            errdefer allocator.free(decode_candidate_logits);
            const decode_candidate_ids = try allocator.alloc(u32, effective_batch_size * 256);
            errdefer allocator.free(decode_candidate_ids);
            const decode_candidate_counts = try allocator.alloc(usize, effective_batch_size);
            errdefer allocator.free(decode_candidate_counts);

            const prefill_request_buffer = try allocator.alloc(contract.PrefillBatchRequest, effective_batch_size);
            errdefer allocator.free(prefill_request_buffer);
            const prefill_request_ids = try allocator.alloc(u64, effective_batch_size);
            errdefer allocator.free(prefill_request_ids);
            const prefill_logits_buffer = try allocator.alloc(f32, effective_batch_size * vocab);
            errdefer allocator.free(prefill_logits_buffer);

            // Initialize optimized sampler with pre-allocated workspace
            // Use seed from config if specified (non-zero), otherwise use time-based seed
            const sampler_seed = if (config.default_sampling.seed != 0)
                config.default_sampling.seed
            else
                @as(u64, @intCast(std.time.milliTimestamp()));

            var sampler_instance = try sampling.Sampler.init(
                allocator,
                sampler_seed,
                vocab,
            );
            errdefer sampler_instance.deinit();

            // Allocate logits buffer for prefill
            const logits_scratch = try allocator.alloc(f32, vocab);
            errdefer allocator.free(logits_scratch);
            const slot_request_ids = try allocator.alloc(?u64, max_batch);
            errdefer allocator.free(slot_request_ids);
            @memset(slot_request_ids, null);
            const scheduler_state_descriptors = try allocator.alloc(runtime_contract.StateDescriptor, config.state_descriptors.len);
            errdefer allocator.free(scheduler_state_descriptors);
            @memcpy(scheduler_state_descriptors, config.state_descriptors);

            // Split descriptors by lifecycle: slot_persistent vs non-persistent
            var sp_count: usize = 0;
            var np_count: usize = 0;
            for (scheduler_state_descriptors) |desc| {
                if (desc.lifecycle == .slot_persistent) sp_count += 1 else np_count += 1;
            }
            const sp_descs = try allocator.alloc(runtime_contract.StateDescriptor, sp_count);
            errdefer allocator.free(sp_descs);
            const np_descs = try allocator.alloc(runtime_contract.StateDescriptor, np_count);
            errdefer allocator.free(np_descs);
            var sp_idx: usize = 0;
            var np_idx: usize = 0;
            for (scheduler_state_descriptors) |desc| {
                if (desc.lifecycle == .slot_persistent) {
                    sp_descs[sp_idx] = desc;
                    sp_idx += 1;
                } else {
                    np_descs[np_idx] = desc;
                    np_idx += 1;
                }
            }

            return Self{
                .allocator = allocator,
                .backend = backend,
                .config = config,
                .requests = std.AutoHashMap(u64, *Request).init(allocator),
                .pending_queue = .{},
                .active_requests = .{},
                .slot_request_ids = slot_request_ids,
                .completed_queue = .{},
                .next_request_id = 1,
                .decode_requests = decode_request_buffer,
                .decode_results = decode_result_buffer,
                .decode_candidate_logits = decode_candidate_logits,
                .decode_candidate_ids = decode_candidate_ids,
                .decode_candidate_counts = decode_candidate_counts,
                .logits_buffer = logits_scratch,
                .prefill_requests = prefill_request_buffer,
                .prefill_request_ids = prefill_request_ids,
                .prefill_logits_buffer = prefill_logits_buffer,
                .sampler = sampler_instance,
                .request_state_blocks = std.AutoHashMap(u64, RequestStateBlocks).init(allocator),
                .slot_state_blocks = std.AutoHashMap(usize, RequestStateBlocks).init(allocator),
                .state_descriptors = scheduler_state_descriptors,
                .slot_persistent_descs = sp_descs,
                .non_persistent_descs = np_descs,
            };
        }

        pub fn deinit(self: *Self) void {
            var state_iter = self.request_state_blocks.iterator();
            while (state_iter.next()) |entry| {
                const request_id = entry.key_ptr.*;
                if (self.requests.get(request_id)) |request_entry| {
                    if (request_entry.slot_index) |slot_index| {
                        // Scheduler owns the backing storage for both request-
                        // scoped and slot-persistent state blocks. Before that
                        // storage is released, backend slot ownership must end
                        // so no backend can retain detached aliases into freed
                        // state blocks across independent runs.
                        self.backend.freeSlot(slot_index);
                    }
                }
                entry.value_ptr.deinit(self.allocator);
            }
            self.request_state_blocks.deinit();

            var slot_state_iter = self.slot_state_blocks.iterator();
            while (slot_state_iter.next()) |entry| {
                // `freeSlot()` above can leave a backend with a detached
                // binding that still references slot-persistent storage until
                // the next explicit unbind. Scheduler teardown is the lifecycle
                // boundary where that storage is about to be freed, so clear
                // the backend view first.
                self.backend.unbindSlotStateBlocks(entry.key_ptr.*);
                var state_blocks = entry.value_ptr.*;
                self.applyLifecycleActionToRequestStateBlocks(&state_blocks, .evict) catch {};
                state_blocks.deinit(self.allocator);
            }
            self.slot_state_blocks.deinit();

            // Free all requests
            var request_iter = self.requests.valueIterator();
            while (request_iter.next()) |req_ptr| {
                req_ptr.*.deinit(self.allocator);
                self.allocator.destroy(req_ptr.*);
            }
            self.requests.deinit();

            self.pending_queue.deinit(self.allocator);
            self.active_requests.deinit(self.allocator);
            self.allocator.free(self.slot_request_ids);
            self.completed_queue.deinit(self.allocator);
            self.allocator.free(self.state_descriptors);
            self.allocator.free(self.slot_persistent_descs);
            self.allocator.free(self.non_persistent_descs);
            self.allocator.free(self.decode_results);
            self.allocator.free(self.decode_requests);
            self.allocator.free(self.decode_candidate_counts);
            self.allocator.free(self.decode_candidate_ids);
            self.allocator.free(self.decode_candidate_logits);
            self.allocator.free(self.logits_buffer);
            self.allocator.free(self.prefill_logits_buffer);
            self.allocator.free(self.prefill_request_ids);
            self.allocator.free(self.prefill_requests);
            self.step_events.deinit(self.allocator);
            self.sampler.deinit();
        }

        fn sampleToken(
            self: *Self,
            logits: []f32,
            sampling_config: sampling.SamplingConfig,
            grammar_sampler: ?*validate.sampler.ConstrainedSampler,
        ) !u32 {
            // Teacher forcing is used by verification/eval flows to drive an
            // exact target token stream. This must be honored in the main
            // sampler path (not only top-k candidate paths), otherwise
            // queued/full-logit routes cannot run deterministic scoring.
            if (xray.getNextForcedToken()) |forced_token| {
                if (trace.isEnabled()) {
                    trace.emitFinal(
                        .token_select,
                        0,
                        0,
                        @ptrCast(std.mem.asBytes(&forced_token).ptr),
                        .u32,
                        .{ 1, 0, 0, 0 },
                        1,
                        "teacher_forcing",
                    );
                }
                return forced_token;
            }

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

            const sampled = try self.sampler.sampleMut(logits, sampling_config);
            return @intCast(sampled);
        }

        const ParityTopLogit = struct {
            token_id: u32 = 0,
            value: f32 = -std.math.inf(f32),
        };

        const ParityStats = struct {
            top: [3]ParityTopLogit,
            min: f32,
            max: f32,
            finite_count: usize,
            non_finite_count: usize,
            checksum: u64,
        };

        fn parityStats(logits: []const f32) ParityStats {
            var top = [_]ParityTopLogit{ .{}, .{}, .{} };
            var min_value = std.math.inf(f32);
            var max_value = -std.math.inf(f32);
            var finite_count: usize = 0;
            var non_finite_count: usize = 0;
            var checksum: u64 = 0xcbf29ce484222325;

            for (logits, 0..) |value, idx| {
                if (!std.math.isFinite(value)) {
                    non_finite_count += 1;
                    continue;
                }
                finite_count += 1;
                min_value = @min(min_value, value);
                max_value = @max(max_value, value);

                const token_id: u32 = @intCast(idx);
                if (value > top[0].value) {
                    top[2] = top[1];
                    top[1] = top[0];
                    top[0] = .{ .token_id = token_id, .value = value };
                } else if (value > top[1].value) {
                    top[2] = top[1];
                    top[1] = .{ .token_id = token_id, .value = value };
                } else if (value > top[2].value) {
                    top[2] = .{ .token_id = token_id, .value = value };
                }

                const bits: u32 = @bitCast(value);
                checksum ^= (@as(u64, token_id) << 32) ^ bits;
                checksum *%= 0x100000001b3;
            }

            if (finite_count == 0) {
                min_value = std.math.nan(f32);
                max_value = std.math.nan(f32);
            }

            return .{
                .top = top,
                .min = min_value,
                .max = max_value,
                .finite_count = finite_count,
                .non_finite_count = non_finite_count,
                .checksum = checksum,
            };
        }

        fn parityLogits(
            self: *Self,
            phase: []const u8,
            position: usize,
            logits: []const f32,
            selected: u32,
            slot_index: usize,
        ) void {
            _ = self;
            if (@intFromEnum(log.Level.trace) < @intFromEnum(log.getLogLevel())) return;
            const stats = parityStats(logits);
            log.trace(
                "inference",
                "PARITY logits",
                .{
                    .backend = @typeName(BackendType),
                    .phase = phase,
                    .pos = position,
                    .slot = slot_index,
                    .selected = selected,
                    .finite = stats.finite_count,
                    .non_finite = stats.non_finite_count,
                    .min = stats.min,
                    .max = stats.max,
                    .checksum = stats.checksum,
                    .top0_id = stats.top[0].token_id,
                    .top0_val = stats.top[0].value,
                    .top1_id = stats.top[1].token_id,
                    .top1_val = stats.top[1].value,
                    .top2_id = stats.top[2].token_id,
                    .top2_val = stats.top[2].value,
                },
                @src(),
            );
        }

        fn sampleTopKCandidateToken(
            self: *Self,
            candidate_logits: []f32,
            candidate_ids: []u32,
            sampling_config: sampling.SamplingConfig,
        ) !u32 {
            // Teacher forcing tokens are vocabulary IDs, not indices into the
            // backend-provided top-k candidate subset. Handle them directly.
            if (xray.getNextForcedToken()) |forced_token| {
                if (trace.isEnabled()) {
                    trace.emitFinal(
                        .token_select,
                        0,
                        0,
                        @ptrCast(std.mem.asBytes(&forced_token).ptr),
                        .u32,
                        .{ 1, 0, 0, 0 },
                        1,
                        "teacher_forcing_topk",
                    );
                }
                return forced_token;
            }

            if (candidate_logits.len == 0 or candidate_logits.len != candidate_ids.len) {
                return error.InvalidArgument;
            }
            if (self.logits_buffer.len < candidate_logits.len) {
                return error.InvalidArgument;
            }

            sortTopKCandidatesByTokenId(candidate_logits, candidate_ids);

            const working_logits = self.logits_buffer[0..candidate_logits.len];
            @memcpy(working_logits, candidate_logits);

            // Apply sampling penalties to the candidate subset. Penalties only
            // affect tokens present in context — for top-k candidates (k ≤ 256)
            // this is exact for all tokens that could be sampled. The only
            // theoretical gap is promoting a non-top-k token above all candidates,
            // which is vanishingly unlikely with reasonable penalty values.
            applyCandidatePenalties(working_logits, candidate_ids, sampling_config);

            var candidate_sampling = sampling_config;
            candidate_sampling.top_k = @min(candidate_sampling.top_k, candidate_logits.len);

            const sampled_idx = try self.sampler.sample(working_logits, candidate_sampling);
            return candidate_ids[sampled_idx];
        }

        fn sampleTopKCandidateTokenPrePenalized(
            self: *Self,
            candidate_logits: []f32,
            candidate_ids: []u32,
            sampling_config: sampling.SamplingConfig,
        ) !u32 {
            if (candidate_logits.len == 0 or candidate_logits.len != candidate_ids.len) {
                return error.InvalidArgument;
            }
            if (self.logits_buffer.len < candidate_logits.len) {
                return error.InvalidArgument;
            }

            sortTopKCandidatesByTokenId(candidate_logits, candidate_ids);
            const working_logits = self.logits_buffer[0..candidate_logits.len];
            @memcpy(working_logits, candidate_logits);

            var candidate_sampling = sampling_config;
            candidate_sampling.top_k = @min(candidate_sampling.top_k, candidate_logits.len);
            candidate_sampling.repetition_penalty = 1.0;
            candidate_sampling.presence_penalty = 0.0;
            candidate_sampling.frequency_penalty = 0.0;
            candidate_sampling.context_tokens = null;
            candidate_sampling.logit_bias = null;

            const sampled_idx = try self.sampler.sample(working_logits, candidate_sampling);
            return candidate_ids[sampled_idx];
        }

        fn sortTopKCandidatesByTokenId(candidate_logits: []f32, candidate_ids: []u32) void {
            std.debug.assert(candidate_logits.len == candidate_ids.len);
            if (candidate_ids.len <= 1) return;

            var i: usize = 1;
            while (i < candidate_ids.len) : (i += 1) {
                const id = candidate_ids[i];
                const logit = candidate_logits[i];
                var j = i;
                while (j > 0 and candidate_ids[j - 1] > id) : (j -= 1) {
                    candidate_ids[j] = candidate_ids[j - 1];
                    candidate_logits[j] = candidate_logits[j - 1];
                }
                candidate_ids[j] = id;
                candidate_logits[j] = logit;
            }
        }

        fn grammarIsComplete(grammar_sampler: ?*validate.sampler.ConstrainedSampler) bool {
            return if (grammar_sampler) |gs| gs.state == .complete else false;
        }

        fn grammarCompleteOnEos(grammar_sampler: ?*validate.sampler.ConstrainedSampler) bool {
            return grammarIsComplete(grammar_sampler);
        }

        fn captureFinalLogits(self: *Self, enabled: bool, logits: []const f32) ![]f32 {
            if (!enabled) return &.{};
            return try self.allocator.dupe(f32, logits);
        }

        fn negLogProbFromLogits(logits: []const f32, token_id: u32) !f64 {
            const token_index: usize = @intCast(token_id);
            if (token_index >= logits.len) return error.InvalidArgument;

            const target_logit = logits[token_index];
            if (!std.math.isFinite(target_logit)) return error.InvalidArgument;

            var max_logit: f32 = -std.math.inf(f32);
            for (logits) |value| {
                if (std.math.isFinite(value)) {
                    max_logit = @max(max_logit, value);
                }
            }
            if (!std.math.isFinite(max_logit)) return error.InvalidArgument;

            var exp_sum: f64 = 0.0;
            for (logits) |value| {
                if (std.math.isFinite(value)) {
                    exp_sum += std.math.exp(@as(f64, value - max_logit));
                }
            }
            if (!(exp_sum > 0.0) or !std.math.isFinite(exp_sum)) return error.InvalidArgument;

            const log_denom = @as(f64, max_logit) + std.math.log(f64, std.math.e, exp_sum);
            return log_denom - @as(f64, target_logit);
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
                .capture_final_logits = submit_config.return_final_logits,
                .max_thinking_tokens = submit_config.max_thinking_tokens,
                .thinking_end_tokens = submit_config.thinking_end_tokens,
                .in_thinking = submit_config.max_thinking_tokens > 0 and submit_config.thinking_end_tokens.len > 0,
                .generating_answer = !(submit_config.max_thinking_tokens > 0 and submit_config.thinking_end_tokens.len > 0),
                .max_completion_tokens_limit = submit_config.max_completion_tokens,
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
            /// Callback invoked for each token. Signature: fn(request_id, token, is_final, in_thinking, user_data)
            callback: ?*const fn (u64, u32, bool, bool, ?*anyopaque) void = null,
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
            /// Capture final-step logits in generateSync result.
            return_final_logits: bool = false,
            /// Maximum tokens to spend in thinking mode (0 = unlimited).
            /// When exceeded, thinking_end_tokens are force-injected to close the
            /// <think> block, then normal generation continues for the answer.
            max_thinking_tokens: usize = 0,
            /// Token IDs to inject when thinking budget is exceeded (e.g. </think>\n\n).
            /// Only used when max_thinking_tokens > 0.
            thinking_end_tokens: []const u32 = &.{},
            /// Maximum tokens for the answer/completion (0 = unlimited).
            /// Only tokens after thinking ends are counted. Injected end-thinking
            /// tokens (</think>\n\n) are excluded from the count.
            max_completion_tokens: usize = 0,
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
                self.releaseRequestStateBlocks(request_id, slot_index);
                if (slot_index < self.slot_request_ids.len) {
                    self.slot_request_ids[slot_index] = null;
                }
                self.backend.freeSlot(slot_index);
                request_entry.slot_index = null;
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

        fn addDecodeTimeToGeneratingRequests(self: *Self, decode_step_ns: u64) void {
            for (self.active_requests.items) |active_request_id| {
                const request_entry = self.requests.get(active_request_id) orelse continue;
                if (request_entry.state != .generating) continue;
                if (request_entry.slot_index == null) continue;
                request_entry.decode_ns += decode_step_ns;
            }
        }

        /// Returns true when greedy-streaming / argmax routes cannot honour
        /// the requested sampling adjustments (they bypass the sampler entirely).
        /// The top-k candidate route handles these via applyCandidatePenalties.
        fn samplingRequiresFullLogits(sampling_cfg: sampling.SamplingConfig) bool {
            return sampling_cfg.repetition_penalty != 1.0 or
                sampling_cfg.presence_penalty != 0.0 or
                sampling_cfg.frequency_penalty != 0.0 or
                sampling_cfg.logit_bias != null;
        }

        fn canUseDirectGreedyCandidate(sampling_cfg: sampling.SamplingConfig, candidate_count: usize) bool {
            if (candidate_count == 0) return false;
            if (sampling_cfg.strategy != .greedy) return false;
            return !samplingRequiresFullLogits(sampling_cfg);
        }

        /// Apply repetition, presence, frequency penalties and logit bias to
        /// the top-k candidate logits in-place. For each candidate token,
        /// scan context_tokens to compute occurrence count, then apply the
        /// same transformations as sampleMut/applyIndexPenalty/applyAdditivePenalties.
        fn applyCandidatePenalties(
            working_logits: []f32,
            candidate_ids: []const u32,
            cfg: sampling.SamplingConfig,
        ) void {
            // Repetition penalty (multiplicative, per-token in context).
            if (cfg.repetition_penalty != 1.0) {
                if (cfg.context_tokens) |ctx| {
                    for (ctx) |ctx_id| {
                        for (candidate_ids, 0..) |cand_id, idx| {
                            if (cand_id == ctx_id) {
                                const s = working_logits[idx];
                                working_logits[idx] = if (s > 0) s / cfg.repetition_penalty else s * cfg.repetition_penalty;
                            }
                        }
                    }
                }
            }
            // Presence + frequency penalties (additive, once/count per unique token).
            if (cfg.presence_penalty != 0.0 or cfg.frequency_penalty != 0.0) {
                if (cfg.context_tokens) |ctx| {
                    for (candidate_ids, 0..) |cand_id, idx| {
                        var count: f32 = 0;
                        for (ctx) |ctx_id| {
                            if (ctx_id == cand_id) count += 1;
                        }
                        if (count > 0) {
                            working_logits[idx] -= cfg.presence_penalty + cfg.frequency_penalty * count;
                        }
                    }
                }
            }
            // Logit bias (additive, per bias entry).
            if (cfg.logit_bias) |bias_entries| {
                for (bias_entries) |entry| {
                    for (candidate_ids, 0..) |cand_id, idx| {
                        if (cand_id == entry.token_id) {
                            working_logits[idx] += entry.bias;
                        }
                    }
                }
            }
        }

        fn resolveBatchedTopKRoute(self: *Self, decode_batch_size: usize) ?usize {
            const use_batched_topk = comptime blk: {
                if (!@hasDecl(BackendType, "decodeBatchTopKCandidates")) break :blk false;
                if (!@hasDecl(BackendType, "supportsSchedulerBackendTopKDecodeRoute") and
                    !@hasDecl(BackendType, "supportsSchedulerBackendBatchedTopKDecodeRoute"))
                {
                    break :blk false;
                }
                break :blk true;
            };
            if (!use_batched_topk) return null;
            // GPU top-K is faster than full-logits DtoH + CPU sampling for
            // n >= 2 (DtoH scales linearly with rows, CPU sampling doubles).
            // For n=1, DtoH of 1 row (~40µs) + CPU SIMD scan (~50µs) beats
            // the GPU kernel which underutilizes SMs (32 blocks on 128 SMs).
            if (decode_batch_size < 2) return null;

            var common_top_k: usize = 0;
            for (self.decode_requests[0..decode_batch_size]) |req| {
                if (req.slot_index >= self.slot_request_ids.len) return null;
                const request_id = self.slot_request_ids[req.slot_index] orelse return null;
                const request_entry = self.requests.get(request_id) orelse return null;
                if (request_entry.state != .generating) return null;
                if (request_entry.slot_index == null or request_entry.slot_index.? != req.slot_index) return null;
                if (request_entry.grammar_sampler != null) return null;
                if (request_entry.capture_final_logits) return null;
                const route_top_k = switch (request_entry.sampling_config.strategy) {
                    .top_k => blk: {
                        if (request_entry.sampling_config.top_k == 0 or request_entry.sampling_config.top_k > 256) {
                            return null;
                        }
                        break :blk request_entry.sampling_config.top_k;
                    },
                    .greedy => blk: {
                        // Greedy can use the same candidate route with k=1, but
                        // only when no sampling mutations are requested.
                        if (samplingRequiresFullLogits(request_entry.sampling_config)) return null;
                        break :blk @as(usize, 1);
                    },
                    else => return null,
                };
                const supports_topk = if (comptime @hasDecl(BackendType, "supportsSchedulerBackendBatchedTopKDecodeRoute"))
                    self.backend.supportsSchedulerBackendBatchedTopKDecodeRoute(&request_entry.sampling_config)
                else
                    self.backend.supportsSchedulerBackendTopKDecodeRoute(&request_entry.sampling_config);
                if (!supports_topk) return null;
                if (common_top_k == 0) {
                    common_top_k = route_top_k;
                } else if (common_top_k != route_top_k) {
                    return null;
                }
            }
            return if (common_top_k > 0) common_top_k else null;
        }

        fn stepBatchedTopKCandidates(
            self: *Self,
            decode_batch_size: usize,
            top_k: usize,
            completed_before: usize,
            prefill_completed: usize,
        ) ![]const TokenEvent {
            const use_batched_topk = comptime @hasDecl(BackendType, "decodeBatchTopKCandidates");
            if (!use_batched_topk) return error.NotEligible;
            if (decode_batch_size == 0) return &.{};
            if (top_k == 0 or top_k > 256) return error.InvalidArgument;

            // Reset step-scoped state for all generating requests before decode.
            for (self.active_requests.items) |active_id| {
                const req = self.requests.get(active_id) orelse continue;
                if (req.state != .generating) continue;
                try self.resetStepScopedBlocks(active_id);
            }

            const needed_candidates = std.math.mul(usize, decode_batch_size, top_k) catch return error.InvalidArgument;
            if (needed_candidates > self.decode_candidate_logits.len or
                needed_candidates > self.decode_candidate_ids.len or
                decode_batch_size > self.decode_candidate_counts.len)
            {
                return error.InvalidArgument;
            }

            var decode_timer = std.time.Timer.start() catch unreachable;
            try self.backend.decodeBatchTopKCandidates(
                self.decode_requests[0..decode_batch_size],
                top_k,
                self.decode_candidate_logits[0..needed_candidates],
                self.decode_candidate_ids[0..needed_candidates],
                self.decode_candidate_counts[0..decode_batch_size],
            );
            const decode_step_ns = decode_timer.read();
            self.addDecodeTimeToGeneratingRequests(decode_step_ns);

            // Don't clear step_events — prefill first-token events were
            // already appended by runPrefills() and must be preserved.

            for (0..decode_batch_size) |row_idx| {
                const req = self.decode_requests[row_idx];
                if (req.slot_index >= self.slot_request_ids.len) continue;
                const request_id = self.slot_request_ids[req.slot_index] orelse continue;
                const request_entry = self.requests.get(request_id) orelse continue;
                if (request_entry.state != .generating) continue;
                if (request_entry.slot_index == null or request_entry.slot_index.? != req.slot_index) continue;

                const candidate_count = self.decode_candidate_counts[row_idx];
                if (candidate_count == 0 or candidate_count > top_k) continue;
                const row_start = std.math.mul(usize, row_idx, top_k) catch return error.InvalidArgument;
                const row_end = std.math.add(usize, row_start, candidate_count) catch return error.InvalidArgument;
                var sample_cfg = request_entry.sampling_config;
                sample_cfg.context_tokens = request_entry.generated_tokens.items;
                var next_token: u32 = if (canUseDirectGreedyCandidate(sample_cfg, candidate_count))
                    self.decode_candidate_ids[row_start]
                else
                    self.sampleTopKCandidateToken(
                        self.decode_candidate_logits[row_start..row_end],
                        self.decode_candidate_ids[row_start..row_end],
                        sample_cfg,
                    ) catch 0;

                if (request_entry.in_thinking) {
                    if (request_entry.thinking_inject_pos > 0) {
                        next_token = request_entry.thinking_end_tokens[request_entry.thinking_inject_pos];
                        request_entry.thinking_inject_pos += 1;
                        if (request_entry.thinking_inject_pos >= request_entry.thinking_end_tokens.len) {
                            request_entry.in_thinking = false;
                            request_entry.generating_answer = true;
                        }
                    } else if (request_entry.thinking_token_count >= request_entry.max_thinking_tokens) {
                        next_token = request_entry.thinking_end_tokens[0];
                        request_entry.thinking_inject_pos = 1;
                        if (request_entry.thinking_end_tokens.len == 1) {
                            request_entry.in_thinking = false;
                            request_entry.generating_answer = true;
                        }
                    } else {
                        request_entry.thinking_token_count += 1;
                        if (request_entry.thinking_end_tokens.len > 0 and
                            next_token == request_entry.thinking_end_tokens[0])
                        {
                            request_entry.in_thinking = false;
                            request_entry.generating_answer = true;
                        }
                    }
                } else if (!request_entry.generating_answer) {
                    request_entry.generating_answer = true;
                }

                if (trace.isEnabled()) {
                    var selected_token = [_]f32{@floatFromInt(next_token)};
                    trace.emitFinal(
                        .token_select,
                        @intCast(request_entry.generated_tokens.items.len),
                        @intCast(request_entry.token_position),
                        @ptrCast(selected_token[0..].ptr),
                        .f32,
                        .{ 1, 0, 0, 0 },
                        1,
                        "sampleTopKBatch",
                    );
                }

                try request_entry.generated_tokens.append(self.allocator, next_token);
                request_entry.token_position += 1;

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

                if (finish_reason == .in_progress and request_entry.stop_sequences.len > 0) {
                    const stop_len = checkStopSequence(request_entry.generated_tokens.items, request_entry.stop_sequences);
                    if (stop_len > 0) {
                        finish_reason = .stop_sequence;
                        request_entry.generated_tokens.shrinkRetainingCapacity(request_entry.generated_tokens.items.len - stop_len);
                    }
                }

                if (finish_reason == .in_progress and request_entry.generating_answer and
                    request_entry.max_completion_tokens_limit > 0)
                {
                    // Only count tokens that produce visible output.
                    // Special tokens (e.g. <think>) decode to empty bytes
                    // and should not consume the completion budget.
                    const has_visible = if (self.config.tokenizer) |tok|
                        (tok.tokenBytes(next_token) orelse &.{}).len > 0
                    else
                        true;
                    if (has_visible) {
                        request_entry.completion_token_count += 1;
                        if (request_entry.completion_token_count >= request_entry.max_completion_tokens_limit) {
                            finish_reason = .length;
                        }
                    }
                }

                if (finish_reason == .in_progress and request_entry.generated_tokens.items.len >= request_entry.max_tokens) {
                    finish_reason = .length;
                }

                const is_final_token = finish_reason != .in_progress;
                if (request_entry.callback) |cb| {
                    if (finish_reason != .stop_sequence) {
                        cb(request_entry.id, next_token, is_final_token, request_entry.in_thinking, request_entry.callback_data);
                    }
                }

                try self.step_events.append(self.allocator, .{
                    .request_id = request_entry.id,
                    .token = next_token,
                    .is_final = is_final_token,
                    .in_thinking = request_entry.in_thinking,
                    .slot_index = req.slot_index,
                    .timestamp_ns = std.time.nanoTimestamp(),
                });

                if (is_final_token) {
                    self.completeRequest(request_entry, finish_reason);
                }
            }
            if (prefill_completed > 0) {
                try self.appendPrefillCompletedEvents(&self.step_events, completed_before);
            }

            try self.activatePending();
            return self.step_events.items;
        }

        /// Run one generation step for all active requests.
        ///
        /// This is the main scheduler loop body. It:
        /// 1. Prefills any pending_prefill requests
        /// 2. Batches decode for all generating requests
        /// 3. Samples next tokens
        /// 4. Returns token events for this step
        ///
        /// The returned slice is a borrowed view into an internal buffer.
        /// It is valid until the next step() call. Caller must NOT free it.
        pub fn step(self: *Self) ![]const TokenEvent {
            // Snapshot completed queue length so we can detect requests that
            // finish during prefill (e.g. max_tokens ≤ 1 or first token is EOS).
            const completed_before = self.completed_queue.items.len;

            // First, prefill any requests that need it.
            // Collect first-token events for just-prefilled requests so they
            // can be included in the step results (the decode methods clear
            // step_events and would lose them otherwise).
            self.step_events.clearRetainingCapacity();
            try self.runPrefills();

            const prefill_completed = self.completed_queue.items.len - completed_before;

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
                // Emit final events for requests completed during prefill.
                // Without this, batch callers that rely on step() events
                // (rather than generateSync) would never observe completion.
                if (prefill_completed > 0) {
                    return try self.buildPrefillCompletedEvents(completed_before);
                }
                return &.{};
            }

            if (self.resolveBatchedTopKRoute(decode_batch_size)) |top_k| {
                return try self.stepBatchedTopKCandidates(
                    decode_batch_size,
                    top_k,
                    completed_before,
                    prefill_completed,
                );
            }

            // Fast path: when there is exactly one generating request and the
            // backend supports top-K candidate decode, use it instead of the
            // general decodeBatch + full-vocab sampleToken path. This matches
            // the route selection in generateSync() for parity with the non-batch
            // iterator path, avoiding ~13% overhead from full-vocab sampling.
            if (decode_batch_size == 1) {
                const use_topk = comptime blk: {
                    if (!@hasDecl(BackendType, "decodeTopKCandidates")) break :blk false;
                    if (!@hasDecl(BackendType, "supportsSchedulerBackendTopKDecodeRoute")) break :blk false;
                    break :blk true;
                };
                if (use_topk) {
                    // Find the single generating request.
                    const req = self.decode_requests[0];
                    var topk_request: ?*Request = null;
                    for (self.active_requests.items) |aid| {
                        const re = self.requests.get(aid) orelse continue;
                        if (re.slot_index == req.slot_index) {
                            topk_request = re;
                            break;
                        }
                    }
                    if (topk_request) |request_entry| {
                        if (request_entry.grammar_sampler == null and
                            !request_entry.capture_final_logits and
                            request_entry.sampling_config.top_k <= 256 and
                            self.backend.supportsSchedulerBackendTopKDecodeRoute(&request_entry.sampling_config))
                        {
                            return try self.stepTopKCandidate(
                                request_entry,
                                req.slot_index,
                                req.token,
                                completed_before,
                                prefill_completed,
                            );
                        }
                    }
                }
            }

            // Reset step-scoped state for all generating requests before decode
            for (self.active_requests.items) |active_id| {
                const req = self.requests.get(active_id) orelse continue;
                if (req.state != .generating) continue;
                try self.resetStepScopedBlocks(active_id);
            }

            // Run batched decode
            var decode_timer = std.time.Timer.start() catch unreachable;
            try self.backend.decodeBatch(
                self.decode_requests[0..decode_batch_size],
                self.decode_results[0..decode_batch_size],
            );
            const decode_step_ns = decode_timer.read();
            self.addDecodeTimeToGeneratingRequests(decode_step_ns);

            // Sample next tokens and build events.
            // Don't clear step_events — prefill first-token events must be preserved.

            for (self.decode_results[0..decode_batch_size]) |result| {
                if (result.slot_index >= self.slot_request_ids.len) continue;
                const request_id = self.slot_request_ids[result.slot_index] orelse continue;
                const request_entry = self.requests.get(request_id) orelse continue;
                if (request_entry.state != .generating) continue;
                if (request_entry.slot_index == null or request_entry.slot_index.? != result.slot_index) continue;

                if (trace.isEnabled()) {
                    trace.emitFinal(
                        .logits_ready,
                        @intCast(request_entry.generated_tokens.items.len),
                        @intCast(request_entry.token_position),
                        @ptrCast(result.logits.ptr),
                        .f32,
                        .{ @intCast(result.logits.len), 0, 0, 0 },
                        1,
                        "decodeBatch_logits",
                    );
                }

                // Sample next token using optimized sampler (SIMD, pre-allocated workspace)
                var sample_cfg = request_entry.sampling_config;
                sample_cfg.context_tokens = request_entry.generated_tokens.items;
                var next_token = self.sampleToken(
                    result.logits,
                    sample_cfg,
                    request_entry.grammar_sampler,
                ) catch 0;
                self.parityLogits(
                    "decode",
                    request_entry.token_position,
                    result.logits,
                    next_token,
                    result.slot_index,
                );

                // Thinking budget enforcement: when the budget is exhausted,
                // force-inject the end-thinking sequence (e.g. </think>\n\n)
                // then resume normal sampling for the answer.
                if (request_entry.in_thinking) {
                    if (request_entry.thinking_inject_pos > 0) {
                        // Continue injecting end-thinking tokens.
                        next_token = request_entry.thinking_end_tokens[request_entry.thinking_inject_pos];
                        request_entry.thinking_inject_pos += 1;
                        if (request_entry.thinking_inject_pos >= request_entry.thinking_end_tokens.len) {
                            request_entry.in_thinking = false;
                            request_entry.generating_answer = true;
                        }
                    } else if (request_entry.thinking_token_count >= request_entry.max_thinking_tokens) {
                        // Budget exceeded: start injecting end sequence.
                        next_token = request_entry.thinking_end_tokens[0];
                        request_entry.thinking_inject_pos = 1;
                        if (request_entry.thinking_end_tokens.len == 1) {
                            request_entry.in_thinking = false;
                            request_entry.generating_answer = true;
                        }
                    } else {
                        request_entry.thinking_token_count += 1;
                        // Check if model naturally produced the end-thinking token.
                        if (request_entry.thinking_end_tokens.len > 0 and
                            next_token == request_entry.thinking_end_tokens[0])
                        {
                            request_entry.in_thinking = false;
                            request_entry.generating_answer = true;
                        }
                    }
                } else if (!request_entry.generating_answer) {
                    // Transition to answer mode one iteration after thinking ends,
                    // so injected end-thinking tokens are excluded from the count.
                    request_entry.generating_answer = true;
                }

                if (trace.isEnabled()) {
                    var selected_token = [_]f32{@floatFromInt(next_token)};
                    trace.emitFinal(
                        .token_select,
                        @intCast(request_entry.generated_tokens.items.len),
                        @intCast(request_entry.token_position),
                        @ptrCast(selected_token[0..].ptr),
                        .f32,
                        .{ 1, 0, 0, 0 },
                        1,
                        "sampleToken",
                    );
                }

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

                // Completion token limit enforcement (answer tokens only).
                // Special tokens that decode to empty bytes are not counted.
                if (finish_reason == .in_progress and request_entry.generating_answer and
                    request_entry.max_completion_tokens_limit > 0)
                {
                    const has_visible = if (self.config.tokenizer) |tok|
                        (tok.tokenBytes(next_token) orelse &.{}).len > 0
                    else
                        true;
                    if (has_visible) {
                        request_entry.completion_token_count += 1;
                        if (request_entry.completion_token_count >= request_entry.max_completion_tokens_limit) {
                            finish_reason = .length;
                        }
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
                        cb(request_entry.id, next_token, is_final_token, request_entry.in_thinking, request_entry.callback_data);
                    }
                }

                // Add event
                try self.step_events.append(self.allocator, .{
                    .request_id = request_entry.id,
                    .token = next_token,
                    .is_final = is_final_token,
                    .in_thinking = request_entry.in_thinking,
                    .slot_index = result.slot_index,
                    .timestamp_ns = std.time.nanoTimestamp(),
                });

                // Handle completion
                if (is_final_token) {
                    if (request_entry.capture_final_logits and request_entry.final_logits.len == 0) {
                        request_entry.final_logits = try self.captureFinalLogits(true, result.logits);
                    }
                    self.completeRequest(request_entry, finish_reason);
                }
            }

            // Include events for requests completed during prefill (mixed case:
            // some requests finished on first token while others are still generating).
            if (prefill_completed > 0) {
                try self.appendPrefillCompletedEvents(&self.step_events, completed_before);
            }

            // Try to activate pending requests (slots may have freed)
            try self.activatePending();

            return self.step_events.items;
        }

        /// Fast path for step() when there is exactly one generating request
        /// and the sampling config is eligible for the top-K candidate route.
        ///
        /// Uses decodeTopKCandidates + sampleTopKCandidateToken instead of
        /// decodeBatch + sampleToken, avoiding full-vocab softmax/quickselect
        /// overhead (~1.2ms per token on 151K vocab).
        fn stepTopKCandidate(
            self: *Self,
            request_entry: *Request,
            slot_index: usize,
            last_token: u32,
            completed_before: usize,
            prefill_completed: usize,
        ) ![]const TokenEvent {
            const top_k: usize = switch (request_entry.sampling_config.strategy) {
                .greedy => 1,
                .top_k => request_entry.sampling_config.top_k,
                else => return error.NotEligible,
            };

            // Stack-allocated candidate buffers (2KB total for K ≤ 256).
            var candidate_logits_buf: [256]f32 = undefined;
            var candidate_ids_buf: [256]u32 = undefined;

            // Decode: forward pass + bulk logits download + CPU top-K extraction.
            var decode_timer = std.time.Timer.start() catch unreachable;
            const candidate_count = try self.backend.decodeTopKCandidates(
                slot_index,
                last_token,
                top_k,
                candidate_logits_buf[0..top_k],
                candidate_ids_buf[0..top_k],
            );
            request_entry.decode_ns += decode_timer.read();

            // Sample from the K candidates (penalties applied to K entries only).
            var sample_cfg = request_entry.sampling_config;
            sample_cfg.context_tokens = request_entry.generated_tokens.items;
            var next_token: u32 = if (canUseDirectGreedyCandidate(sample_cfg, candidate_count))
                candidate_ids_buf[0]
            else
                try self.sampleTopKCandidateToken(
                    candidate_logits_buf[0..candidate_count],
                    candidate_ids_buf[0..candidate_count],
                    sample_cfg,
                );

            // --- Post-processing (identical to step() Phase 6-8) ---

            // Thinking budget enforcement.
            if (request_entry.in_thinking) {
                if (request_entry.thinking_inject_pos > 0) {
                    next_token = request_entry.thinking_end_tokens[request_entry.thinking_inject_pos];
                    request_entry.thinking_inject_pos += 1;
                    if (request_entry.thinking_inject_pos >= request_entry.thinking_end_tokens.len) {
                        request_entry.in_thinking = false;
                        request_entry.generating_answer = true;
                    }
                } else if (request_entry.thinking_token_count >= request_entry.max_thinking_tokens) {
                    next_token = request_entry.thinking_end_tokens[0];
                    request_entry.thinking_inject_pos = 1;
                    if (request_entry.thinking_end_tokens.len == 1) {
                        request_entry.in_thinking = false;
                        request_entry.generating_answer = true;
                    }
                } else {
                    request_entry.thinking_token_count += 1;
                    if (request_entry.thinking_end_tokens.len > 0 and
                        next_token == request_entry.thinking_end_tokens[0])
                    {
                        request_entry.in_thinking = false;
                        request_entry.generating_answer = true;
                    }
                }
            } else if (!request_entry.generating_answer) {
                request_entry.generating_answer = true;
            }

            // Append token.
            try request_entry.generated_tokens.append(self.allocator, next_token);
            request_entry.token_position += 1;

            // Check EOS.
            var finish_reason: FinishReason = .in_progress;
            for (request_entry.eos_token_ids) |eos_id| {
                if (next_token == eos_id) {
                    finish_reason = .eos_token;
                    break;
                }
            }

            // Check stop sequences.
            if (finish_reason == .in_progress and request_entry.stop_sequences.len > 0) {
                const stop_len = checkStopSequence(request_entry.generated_tokens.items, request_entry.stop_sequences);
                if (stop_len > 0) {
                    finish_reason = .stop_sequence;
                    request_entry.generated_tokens.shrinkRetainingCapacity(request_entry.generated_tokens.items.len - stop_len);
                }
            }

            // Completion token limit enforcement (answer tokens only).
            // Special tokens that decode to empty bytes are not counted.
            if (finish_reason == .in_progress and request_entry.generating_answer and
                request_entry.max_completion_tokens_limit > 0)
            {
                const has_visible = if (self.config.tokenizer) |tok|
                    (tok.tokenBytes(next_token) orelse &.{}).len > 0
                else
                    true;
                if (has_visible) request_entry.completion_token_count += 1;
                if (request_entry.completion_token_count >= request_entry.max_completion_tokens_limit) {
                    finish_reason = .length;
                }
            }

            // Max tokens.
            if (finish_reason == .in_progress and request_entry.generated_tokens.items.len >= request_entry.max_tokens) {
                finish_reason = .length;
            }

            const is_final = finish_reason != .in_progress;

            // Callback.
            if (request_entry.callback) |cb| {
                if (finish_reason != .stop_sequence) {
                    cb(request_entry.id, next_token, is_final, request_entry.in_thinking, request_entry.callback_data);
                }
            }

            // Build event. Don't clear — prefill first-token events must be preserved.
            try self.step_events.append(self.allocator, .{
                .request_id = request_entry.id,
                .token = next_token,
                .is_final = is_final,
                .in_thinking = false,
                .slot_index = slot_index,
                .timestamp_ns = std.time.nanoTimestamp(),
            });

            // Handle completion.
            if (is_final) {
                self.completeRequest(request_entry, finish_reason);
            }

            // Include events for requests completed during prefill.
            if (prefill_completed > 0) {
                try self.appendPrefillCompletedEvents(&self.step_events, completed_before);
            }

            try self.activatePending();
            return self.step_events.items;
        }

        /// Tight decode loop for a single active request using the top-K
        /// candidate route. Eliminates per-step event/dispatch overhead by
        /// running the decode+sample+callback loop without returning.
        ///
        /// The per-token callback is invoked for each generated token with
        /// (request_id, token, is_final, user_data). It must NOT call back
        /// into the scheduler (not reentrant).
        ///
        /// The loop runs until the request completes, pending_flag is set,
        /// or an error occurs. On completion, the request is moved to the
        /// completed queue. Returns error.NotEligible if the request doesn't
        /// qualify (caller should fall back to the step-based loop).
        pub fn runDecodeLoop(
            self: *Self,
            request_id: u64,
            pending_flag: ?*const std.atomic.Value(bool),
            per_token_cb: *const fn (u64, u32, bool, bool, ?*anyopaque) callconv(.c) void,
            cb_data: ?*anyopaque,
        ) !void {
            const re = self.requests.get(request_id) orelse return error.InvalidArgument;
            if (re.state != .generating) return error.NotEligible;
            const slot = re.slot_index orelse return error.NotEligible;
            if (re.grammar_sampler != null) return error.NotEligible;
            if (re.capture_final_logits) return error.NotEligible;
            const top_k: usize = switch (re.sampling_config.strategy) {
                .greedy => 1,
                .top_k => re.sampling_config.top_k,
                else => return error.NotEligible,
            };
            if (top_k == 0 or top_k > 256) return error.NotEligible;

            const use_topk = comptime blk: {
                if (!@hasDecl(BackendType, "decodeTopKCandidates")) break :blk false;
                if (!@hasDecl(BackendType, "supportsSchedulerBackendTopKDecodeRoute")) break :blk false;
                break :blk true;
            };
            if (!use_topk) return error.NotEligible;
            if (!self.backend.supportsSchedulerBackendTopKDecodeRoute(&re.sampling_config))
                return error.NotEligible;

            var candidate_logits: [256]f32 = undefined;
            var candidate_ids: [256]u32 = undefined;

            while (re.generated_tokens.items.len < re.max_tokens) {
                if (pending_flag) |f| {
                    if (f.load(.acquire)) return;
                }

                const last_token = if (re.generated_tokens.items.len > 0)
                    re.generated_tokens.items[re.generated_tokens.items.len - 1]
                else if (re.prompt_tokens.len > 0)
                    re.prompt_tokens[re.prompt_tokens.len - 1]
                else
                    return error.InvalidArgument;

                const count = try self.backend.decodeTopKCandidates(
                    slot,
                    last_token,
                    top_k,
                    candidate_logits[0..top_k],
                    candidate_ids[0..top_k],
                );
                if (count == 0) return error.InvalidArgument;

                // Thinking budget enforcement.
                var next_token: u32 = undefined;
                if (re.in_thinking and re.thinking_inject_pos > 0) {
                    next_token = re.thinking_end_tokens[re.thinking_inject_pos];
                    re.thinking_inject_pos += 1;
                    if (re.thinking_inject_pos >= re.thinking_end_tokens.len) {
                        re.in_thinking = false;
                        re.generating_answer = true;
                    }
                } else if (re.in_thinking and re.thinking_token_count >= re.max_thinking_tokens) {
                    next_token = re.thinking_end_tokens[0];
                    re.thinking_inject_pos = 1;
                    if (re.thinking_end_tokens.len == 1) {
                        re.in_thinking = false;
                        re.generating_answer = true;
                    }
                } else {
                    var sample_cfg = re.sampling_config;
                    sample_cfg.context_tokens = re.generated_tokens.items;
                    next_token = if (canUseDirectGreedyCandidate(sample_cfg, count))
                        candidate_ids[0]
                    else
                        try self.sampleTopKCandidateToken(
                            candidate_logits[0..count],
                            candidate_ids[0..count],
                            sample_cfg,
                        );
                    if (re.in_thinking) {
                        re.thinking_token_count += 1;
                        if (re.thinking_end_tokens.len > 0 and
                            next_token == re.thinking_end_tokens[0])
                        {
                            re.in_thinking = false;
                            re.generating_answer = true;
                        }
                    }
                }

                try re.generated_tokens.append(self.allocator, next_token);
                re.token_position += 1;

                var finish_reason: FinishReason = .in_progress;
                for (re.eos_token_ids) |eos_id| {
                    if (next_token == eos_id) {
                        finish_reason = .eos_token;
                        break;
                    }
                }
                if (finish_reason == .in_progress and re.stop_sequences.len > 0) {
                    const stop_len = checkStopSequence(re.generated_tokens.items, re.stop_sequences);
                    if (stop_len > 0) {
                        finish_reason = .stop_sequence;
                        re.generated_tokens.shrinkRetainingCapacity(re.generated_tokens.items.len - stop_len);
                    }
                }
                if (finish_reason == .in_progress and re.generating_answer and
                    re.max_completion_tokens_limit > 0)
                {
                    const has_visible = if (self.config.tokenizer) |tok|
                        (tok.tokenBytes(next_token) orelse &.{}).len > 0
                    else
                        true;
                    if (has_visible) {
                        re.completion_token_count += 1;
                        if (re.completion_token_count >= re.max_completion_tokens_limit) {
                            finish_reason = .length;
                        }
                    }
                }
                if (!re.in_thinking and !re.generating_answer) {
                    re.generating_answer = true;
                }
                if (finish_reason == .in_progress and re.generated_tokens.items.len >= re.max_tokens) {
                    finish_reason = .length;
                }

                const is_final = finish_reason != .in_progress;
                if (finish_reason != .stop_sequence) {
                    per_token_cb(request_id, next_token, is_final, re.in_thinking, cb_data);
                }

                if (is_final) {
                    self.completeRequest(re, finish_reason);
                    return;
                }
            }

            // Exhausted max_tokens.
            self.completeRequest(re, .length);
            per_token_cb(request_id, 0, true, false, cb_data);
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
            if (self.requests.get(request_id)) |request_entry| {
                self.releaseRequestStateBlocks(request_id, request_entry.slot_index);
            } else {
                self.releaseRequestStateBlocks(request_id, null);
            }
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

        /// Get the prefill time in nanoseconds for a request.
        pub fn getPrefillNs(self: *const Self, request_id: u64) ?u64 {
            const request_entry = self.requests.get(request_id) orelse return null;
            return request_entry.prefill_ns;
        }

        /// Synchronous single-request generation.
        ///
        /// Submits one request and runs to completion, returning generated tokens.
        /// This is a convenience wrapper for single-request use cases.
        ///
        /// Returns an owned slice of generated tokens (caller must free).
        pub fn generateSync(
            self: *Self,
            prompt_tokens: []const u32,
            max_tokens: usize,
            options: ?SubmitOptions,
        ) !GenerateSyncResult {
            const submit_config = options orelse SubmitOptions{};
            const effective_sampling = submit_config.sampling orelse self.config.default_sampling;
            const backend_supports_greedy_streaming = comptime blk: {
                if (!@hasDecl(BackendType, "decodeStreaming")) break :blk false;
                if (!@hasDecl(BackendType, "supportsSchedulerBackendDecodeStreamingRoute")) break :blk false;
                break :blk true;
            };
            const backend_supports_top_k_candidates = comptime blk: {
                if (!@hasDecl(BackendType, "decodeTopKCandidates")) break :blk false;
                if (!@hasDecl(BackendType, "supportsSchedulerBackendTopKDecodeRoute")) break :blk false;
                break :blk true;
            };
            const backend_supports_top_k_streaming = comptime blk: {
                if (!@hasDecl(BackendType, "decodeTopKStreaming")) break :blk false;
                if (!@hasDecl(BackendType, "supportsSchedulerBackendTopKStreamingRoute")) break :blk false;
                break :blk true;
            };
            // Greedy streaming uses argmax without any sampler — penalties cannot
            // be applied. The top-k candidate route handles penalties via
            // applyCandidatePenalties, so it doesn't need this gate.
            const has_sampling_adjustments = samplingRequiresFullLogits(effective_sampling);
            const can_use_greedy_streaming = prompt_tokens.len > 0 and
                self.active_requests.items.len == 0 and
                self.pending_queue.items.len == 0 and
                submit_config.vision_input == null and
                submit_config.stop_sequences.len == 0 and
                submit_config.grammar_sampler == null and
                !submit_config.return_final_logits and
                submit_config.max_thinking_tokens == 0 and // thinking budget needs per-token control
                effective_sampling.strategy == .greedy and
                !has_sampling_adjustments and
                !xray.isTeacherForcingEnabled() and
                backend_supports_greedy_streaming and
                self.backend.supportsSchedulerBackendDecodeStreamingRoute();
            if (can_use_greedy_streaming) {
                log.debug("inference", "Scheduler decode route selected", .{
                    .route = "greedy_streaming",
                    .strategy = @tagName(effective_sampling.strategy),
                    .top_k = effective_sampling.top_k,
                    .temperature = effective_sampling.temperature,
                    .has_callback = @as(u8, @intFromBool(submit_config.callback != null)),
                }, @src());
                return self.generateSyncGreedyStreamingRoute(prompt_tokens, max_tokens, &submit_config, &effective_sampling);
            }
            const can_use_top_k_streaming = prompt_tokens.len > 0 and
                self.active_requests.items.len == 0 and
                self.pending_queue.items.len == 0 and
                submit_config.vision_input == null and
                submit_config.stop_sequences.len == 0 and
                submit_config.grammar_sampler == null and
                !submit_config.return_final_logits and
                submit_config.max_thinking_tokens == 0 and
                effective_sampling.seed == 0 and
                !xray.isTeacherForcingEnabled() and
                backend_supports_top_k_streaming and
                self.backend.supportsSchedulerBackendTopKStreamingRoute(&effective_sampling);
            if (can_use_top_k_streaming) {
                log.debug("inference", "Scheduler decode route selected", .{
                    .route = "topk_streaming",
                    .strategy = @tagName(effective_sampling.strategy),
                    .top_k = effective_sampling.top_k,
                    .temperature = effective_sampling.temperature,
                    .top_p = effective_sampling.top_p,
                    .min_p = effective_sampling.min_p,
                    .seed = effective_sampling.seed,
                    .repetition_penalty = effective_sampling.repetition_penalty,
                    .presence_penalty = effective_sampling.presence_penalty,
                    .frequency_penalty = effective_sampling.frequency_penalty,
                    .has_callback = @as(u8, @intFromBool(submit_config.callback != null)),
                }, @src());
                return self.generateSyncTopKStreamingRoute(prompt_tokens, max_tokens, &submit_config, &effective_sampling);
            }
            const can_use_top_k_candidate_route = prompt_tokens.len > 0 and
                self.active_requests.items.len == 0 and
                self.pending_queue.items.len == 0 and
                submit_config.callback == null and
                submit_config.vision_input == null and
                submit_config.grammar_sampler == null and
                !submit_config.return_final_logits and
                backend_supports_top_k_candidates and
                self.backend.supportsSchedulerBackendTopKDecodeRoute(&effective_sampling);
            if (can_use_top_k_candidate_route) {
                log.debug("inference", "Scheduler decode route selected", .{
                    .route = "topk_candidate",
                    .strategy = @tagName(effective_sampling.strategy),
                    .top_k = effective_sampling.top_k,
                    .temperature = effective_sampling.temperature,
                    .top_p = effective_sampling.top_p,
                    .min_p = effective_sampling.min_p,
                    .seed = effective_sampling.seed,
                    .repetition_penalty = effective_sampling.repetition_penalty,
                    .presence_penalty = effective_sampling.presence_penalty,
                    .frequency_penalty = effective_sampling.frequency_penalty,
                    .has_callback = @as(u8, @intFromBool(submit_config.callback != null)),
                }, @src());
                return self.generateSyncTopKCandidateRoute(prompt_tokens, max_tokens, &submit_config, &effective_sampling);
            }

            log.debug("inference", "Scheduler decode route selected", .{
                .route = "queued",
                .strategy = @tagName(effective_sampling.strategy),
                .top_k = effective_sampling.top_k,
                .temperature = effective_sampling.temperature,
                .has_callback = @as(u8, @intFromBool(submit_config.callback != null)),
            }, @src());

            // Always use the standard queued scheduler path.
            const request_id = try self.submit(prompt_tokens, max_tokens, options);

            // Run until this request completes
            while (self.hasActive()) {
                const events = try self.step();

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
            const prefill_ns = request_entry.prefill_ns;
            const decode_ns = request_entry.decode_ns;
            const final_logits: []f32 = if (request_entry.final_logits.len > 0)
                try self.allocator.dupe(f32, request_entry.final_logits)
            else
                &.{};

            // Cleanup
            self.remove(request_id);

            return GenerateSyncResult{
                .tokens = generated_tokens,
                .final_logits = final_logits,
                .finish_reason = finish_reason,
                .prefill_ns = prefill_ns,
                .decode_ns = decode_ns,
            };
        }

        pub const TeacherForcedNllResult = struct {
            nll_sum: f64,
            scored_tokens: usize,
            prefill_ns: u64,
            decode_ns: u64,
        };

        pub const TeacherForcedCursor = struct {
            request_id: u64,
            slot_index: usize,
            prefill_ns: u64,
            decode_ns: u64,
            started: bool,
        };

        fn reserveInternalRequestId(self: *Self) !u64 {
            var request_id = self.next_request_id;
            self.next_request_id +%= 1;
            if (self.next_request_id == 0) self.next_request_id = 1;

            while (self.requests.contains(request_id) or self.request_state_blocks.contains(request_id)) {
                request_id +%= 1;
                if (request_id == 0) request_id = 1;
                if (request_id == self.next_request_id) return error.OutOfMemory;
            }
            return request_id;
        }

        /// Begin teacher-forced scoring for an already-tokenized prompt.
        ///
        /// The returned cursor owns a backend slot and associated request-scoped
        /// state blocks until `endTeacherForced` is called.
        pub fn beginTeacherForced(
            self: *Self,
            prompt_tokens: []const u32,
        ) !TeacherForcedCursor {
            if (prompt_tokens.len == 0) return error.InvalidArgument;

            const request_id = try self.reserveInternalRequestId();
            const slot_index = self.backend.allocSlot() orelse return error.NoSlotsAvailable;
            errdefer self.backend.freeSlot(slot_index);

            try self.bindAndTrackRequestStateBlocks(request_id, slot_index);
            errdefer self.releaseRequestStateBlocks(request_id, slot_index);

            var prefill_timer = std.time.Timer.start() catch unreachable;
            try self.prefillWithOptionalVision(slot_index, prompt_tokens, null);

            return .{
                .request_id = request_id,
                .slot_index = slot_index,
                .prefill_ns = prefill_timer.read(),
                .decode_ns = 0,
                .started = true,
            };
        }

        /// Return current next-token logits for a started teacher-forced cursor.
        pub fn teacherForcedCurrentLogits(
            self: *Self,
            cursor: *const TeacherForcedCursor,
        ) ![]const f32 {
            if (!cursor.started) return error.InvalidArgument;
            return self.logits_buffer;
        }

        /// Advance a teacher-forced scoring cursor by one known token.
        pub fn advanceTeacherForced(
            self: *Self,
            cursor: *TeacherForcedCursor,
            token: u32,
        ) !void {
            if (!cursor.started) return error.InvalidArgument;

            try self.resetStepScopedBlocks(cursor.request_id);

            var decode_request: [1]DecodeRequest = .{.{
                .slot_index = cursor.slot_index,
                .token = token,
            }};
            var decode_result: [1]DecodeResult = .{.{
                .slot_index = cursor.slot_index,
                .logits = self.logits_buffer,
            }};

            var decode_timer = std.time.Timer.start() catch unreachable;
            try self.backend.decodeBatch(decode_request[0..], decode_result[0..]);
            cursor.decode_ns += decode_timer.read();
        }

        /// End teacher-forced scoring and release all internal resources.
        pub fn endTeacherForced(
            self: *Self,
            cursor: *TeacherForcedCursor,
        ) void {
            if (!cursor.started) return;
            self.releaseRequestStateBlocks(cursor.request_id, cursor.slot_index);
            self.backend.freeSlot(cursor.slot_index);
            cursor.started = false;
        }

        /// Score teacher-forced autoregressive targets in one backend pass.
        ///
        /// This uses the scheduler's canonical slot/state lifecycle (same binding
        /// path as generation) and avoids per-token re-prefill overhead.
        pub fn scoreTeacherForcedNll(
            self: *Self,
            prompt_tokens: []const u32,
            target_tokens: []const u32,
        ) !TeacherForcedNllResult {
            if (prompt_tokens.len == 0 or target_tokens.len == 0) return error.InvalidArgument;

            var cursor = try self.beginTeacherForced(prompt_tokens);
            defer self.endTeacherForced(&cursor);

            var logits = try self.teacherForcedCurrentLogits(&cursor);
            var nll_sum = try negLogProbFromLogits(logits, target_tokens[0]);
            var scored: usize = 1;
            if (target_tokens.len == 1) {
                return .{
                    .nll_sum = nll_sum,
                    .scored_tokens = scored,
                    .prefill_ns = cursor.prefill_ns,
                    .decode_ns = 0,
                };
            }

            for (target_tokens[1..], 0..) |target, idx| {
                try self.advanceTeacherForced(&cursor, target_tokens[idx]);
                logits = try self.teacherForcedCurrentLogits(&cursor);
                nll_sum += try negLogProbFromLogits(logits, target);
                scored += 1;
            }

            return .{
                .nll_sum = nll_sum,
                .scored_tokens = scored,
                .prefill_ns = cursor.prefill_ns,
                .decode_ns = cursor.decode_ns,
            };
        }

        fn generateSyncGreedyStreamingRoute(
            self: *Self,
            prompt_tokens: []const u32,
            max_tokens: usize,
            submit_config: *const SubmitOptions,
            sampling_config: *const sampling.SamplingConfig,
        ) !GenerateSyncResult {
            const StreamCbCtx = struct {
                submit_callback: *const fn (u64, u32, bool, bool, ?*anyopaque) void,
                submit_callback_data: ?*anyopaque,
                eos_token_ids: []const u32,
                max_tail_tokens: usize,
                emitted_tail_tokens: usize = 0,

                fn onToken(token: u32, user_data: ?*anyopaque) void {
                    const ctx: *@This() = @ptrCast(@alignCast(user_data));
                    ctx.emitted_tail_tokens += 1;
                    var is_final = ctx.emitted_tail_tokens >= ctx.max_tail_tokens;
                    if (!is_final) {
                        for (ctx.eos_token_ids) |eos_id| {
                            if (token == eos_id) {
                                is_final = true;
                                break;
                            }
                        }
                    }
                    ctx.submit_callback(0, token, is_final, false, ctx.submit_callback_data);
                }
            };

            if (max_tokens == 0) {
                return .{
                    .tokens = try self.allocator.dupe(u32, &.{}),
                    .finish_reason = .length,
                    .prefill_ns = 0,
                    .decode_ns = 0,
                };
            }

            if (generationShouldStop(submit_config.stop_flag)) {
                return .{
                    .tokens = try self.allocator.dupe(u32, &.{}),
                    .finish_reason = .cancelled,
                    .prefill_ns = 0,
                    .decode_ns = 0,
                };
            }

            const slot_index = self.backend.allocSlot() orelse return error.NoSlotsAvailable;
            defer self.backend.freeSlot(slot_index);

            const sp_blocks = if (self.slot_persistent_descs.len > 0)
                try self.slotStateBlocksForSlot(slot_index)
            else
                null;
            var np_blocks = try self.allocateRequestStateBlocks();
            defer {
                self.applyLifecycleActionToRequestStateBlocks(&np_blocks, .evict) catch {};
                np_blocks.deinit(self.allocator);
            }

            var merged: [runtime_contract.max_state_descriptors]runtime_contract.StateBlockHandle = undefined;
            var merge_count: usize = 0;
            if (sp_blocks) |spb| {
                for (spb.handles) |h| {
                    merged[merge_count] = h;
                    merge_count += 1;
                }
            }
            for (np_blocks.handles) |h| {
                merged[merge_count] = h;
                merge_count += 1;
            }
            if (merge_count > 0) {
                try self.backend.bindSlotStateBlocks(slot_index, merged[0..merge_count]);
            }
            defer self.backend.unbindSlotStateBlocks(slot_index);

            if (sampling_config.seed != 0) {
                self.sampler.reseed(sampling_config.seed);
            }

            var prefill_timer = std.time.Timer.start() catch unreachable;
            try self.prefillWithOptionalVision(slot_index, prompt_tokens, null);
            const prefill_ns = prefill_timer.read();
            if (generationShouldStop(submit_config.stop_flag)) {
                return .{
                    .tokens = try self.allocator.dupe(u32, &.{}),
                    .finish_reason = .cancelled,
                    .prefill_ns = prefill_ns,
                    .decode_ns = 0,
                };
            }

            const eos_token_ids = submit_config.eos_token_ids orelse self.config.default_eos_token_ids;
            var sync_sample_cfg = sampling_config.*;
            sync_sample_cfg.context_tokens = &.{};
            const current_token = self.sampleToken(self.logits_buffer, sync_sample_cfg, null) catch 0;

            var generated = try std.ArrayList(u32).initCapacity(self.allocator, max_tokens);
            errdefer generated.deinit(self.allocator);
            try generated.append(self.allocator, current_token);

            var finished = false;
            for (eos_token_ids) |eos_id| {
                if (current_token == eos_id) {
                    finished = true;
                    break;
                }
            }

            if (submit_config.callback) |cb| {
                const is_final = finished or max_tokens == 1;
                cb(0, current_token, is_final, false, submit_config.callback_data);
            }

            if (finished) {
                return .{
                    .tokens = try generated.toOwnedSlice(self.allocator),
                    .finish_reason = .eos_token,
                    .prefill_ns = prefill_ns,
                    .decode_ns = 0,
                };
            }
            if (max_tokens == 1) {
                return .{
                    .tokens = try generated.toOwnedSlice(self.allocator),
                    .finish_reason = .length,
                    .prefill_ns = prefill_ns,
                    .decode_ns = 0,
                };
            }

            const remaining_token_budget = max_tokens - generated.items.len;
            const generated_tail = try self.allocator.alloc(u32, remaining_token_budget);
            defer self.allocator.free(generated_tail);

            var decode_timer = std.time.Timer.start() catch unreachable;
            if (generationShouldStop(submit_config.stop_flag)) {
                return .{
                    .tokens = try generated.toOwnedSlice(self.allocator),
                    .finish_reason = .cancelled,
                    .prefill_ns = prefill_ns,
                    .decode_ns = 0,
                };
            }

            var stream_cb_ctx: ?StreamCbCtx = null;
            var backend_stream_cb: ?*const fn (u32, ?*anyopaque) void = null;
            var backend_stream_cb_data: ?*anyopaque = null;
            if (submit_config.callback) |cb| {
                stream_cb_ctx = .{
                    .submit_callback = cb,
                    .submit_callback_data = submit_config.callback_data,
                    .eos_token_ids = eos_token_ids,
                    .max_tail_tokens = remaining_token_budget,
                };
                backend_stream_cb = StreamCbCtx.onToken;
                backend_stream_cb_data = @ptrCast(&stream_cb_ctx.?);
            }

            const tail_count = try self.backend.decodeStreaming(
                current_token,
                generated.items.len + 1,
                remaining_token_budget,
                eos_token_ids,
                generated_tail,
                backend_stream_cb,
                backend_stream_cb_data,
            );
            try generated.appendSlice(self.allocator, generated_tail[0..tail_count]);
            const decode_ns = decode_timer.read();
            var finish_reason: FinishReason = if (tail_count < remaining_token_budget) .eos_token else .length;
            if (submit_config.stop_sequences.len > 0) {
                var stop_at: ?usize = null;
                for (submit_config.stop_sequences) |stop_seq| {
                    if (stop_seq.len == 0 or generated.items.len < stop_seq.len) continue;
                    var idx: usize = 0;
                    while (idx + stop_seq.len <= generated.items.len) : (idx += 1) {
                        if (std.mem.eql(u32, generated.items[idx .. idx + stop_seq.len], stop_seq)) {
                            if (stop_at == null or idx < stop_at.?) stop_at = idx;
                            break;
                        }
                    }
                }
                if (stop_at) |idx| {
                    generated.shrinkRetainingCapacity(idx);
                    finish_reason = .stop_sequence;
                }
            }
            return .{
                .tokens = try generated.toOwnedSlice(self.allocator),
                .finish_reason = finish_reason,
                .prefill_ns = prefill_ns,
                .decode_ns = decode_ns,
            };
        }

        fn generateSyncTopKStreamingRoute(
            self: *Self,
            prompt_tokens: []const u32,
            max_tokens: usize,
            submit_config: *const SubmitOptions,
            sampling_config: *const sampling.SamplingConfig,
        ) !GenerateSyncResult {
            const StreamCbCtx = struct {
                submit_callback: *const fn (u64, u32, bool, bool, ?*anyopaque) void,
                submit_callback_data: ?*anyopaque,
                eos_token_ids: []const u32,
                max_tail_tokens: usize,
                emitted_tail_tokens: usize = 0,

                fn onToken(token: u32, user_data: ?*anyopaque) void {
                    const ctx: *@This() = @ptrCast(@alignCast(user_data));
                    ctx.emitted_tail_tokens += 1;
                    var is_final = ctx.emitted_tail_tokens >= ctx.max_tail_tokens;
                    if (!is_final) {
                        for (ctx.eos_token_ids) |eos_id| {
                            if (token == eos_id) {
                                is_final = true;
                                break;
                            }
                        }
                    }
                    ctx.submit_callback(0, token, is_final, false, ctx.submit_callback_data);
                }
            };

            if (max_tokens == 0) {
                return .{
                    .tokens = try self.allocator.dupe(u32, &.{}),
                    .finish_reason = .length,
                    .prefill_ns = 0,
                    .decode_ns = 0,
                };
            }

            if (generationShouldStop(submit_config.stop_flag)) {
                return .{
                    .tokens = try self.allocator.dupe(u32, &.{}),
                    .finish_reason = .cancelled,
                    .prefill_ns = 0,
                    .decode_ns = 0,
                };
            }

            const slot_index = self.backend.allocSlot() orelse return error.NoSlotsAvailable;
            defer self.backend.freeSlot(slot_index);

            const sp_blocks = if (self.slot_persistent_descs.len > 0)
                try self.slotStateBlocksForSlot(slot_index)
            else
                null;
            var np_blocks = try self.allocateRequestStateBlocks();
            defer {
                self.applyLifecycleActionToRequestStateBlocks(&np_blocks, .evict) catch {};
                np_blocks.deinit(self.allocator);
            }

            var merged: [runtime_contract.max_state_descriptors]runtime_contract.StateBlockHandle = undefined;
            var merge_count: usize = 0;
            if (sp_blocks) |spb| {
                for (spb.handles) |h| {
                    merged[merge_count] = h;
                    merge_count += 1;
                }
            }
            for (np_blocks.handles) |h| {
                merged[merge_count] = h;
                merge_count += 1;
            }
            if (merge_count > 0) {
                try self.backend.bindSlotStateBlocks(slot_index, merged[0..merge_count]);
            }
            defer self.backend.unbindSlotStateBlocks(slot_index);

            // Top-k streaming route currently relies on backend RNG state.
            // Keep route-gating strict in generateSync() to avoid changing
            // deterministic seeded behavior.
            if (sampling_config.seed != 0) {
                self.sampler.reseed(sampling_config.seed);
            }

            var prefill_timer = std.time.Timer.start() catch unreachable;
            try self.prefillWithOptionalVision(slot_index, prompt_tokens, null);
            const prefill_ns = prefill_timer.read();
            if (generationShouldStop(submit_config.stop_flag)) {
                return .{
                    .tokens = try self.allocator.dupe(u32, &.{}),
                    .finish_reason = .cancelled,
                    .prefill_ns = prefill_ns,
                    .decode_ns = 0,
                };
            }

            const eos_token_ids = submit_config.eos_token_ids orelse self.config.default_eos_token_ids;
            var topk_sample_cfg = sampling_config.*;
            topk_sample_cfg.context_tokens = &.{};
            const current_token = self.sampleToken(self.logits_buffer, topk_sample_cfg, null) catch 0;

            var generated = try std.ArrayList(u32).initCapacity(self.allocator, max_tokens);
            errdefer generated.deinit(self.allocator);
            try generated.append(self.allocator, current_token);

            var finished = false;
            for (eos_token_ids) |eos_id| {
                if (current_token == eos_id) {
                    finished = true;
                    break;
                }
            }

            if (submit_config.callback) |cb| {
                const is_final = finished or max_tokens == 1;
                cb(0, current_token, is_final, false, submit_config.callback_data);
            }

            if (finished) {
                return .{
                    .tokens = try generated.toOwnedSlice(self.allocator),
                    .finish_reason = .eos_token,
                    .prefill_ns = prefill_ns,
                    .decode_ns = 0,
                };
            }
            if (max_tokens == 1) {
                return .{
                    .tokens = try generated.toOwnedSlice(self.allocator),
                    .finish_reason = .length,
                    .prefill_ns = prefill_ns,
                    .decode_ns = 0,
                };
            }

            const remaining_token_budget = max_tokens - generated.items.len;
            const generated_tail = try self.allocator.alloc(u32, remaining_token_budget);
            defer self.allocator.free(generated_tail);

            var decode_timer = std.time.Timer.start() catch unreachable;
            if (generationShouldStop(submit_config.stop_flag)) {
                return .{
                    .tokens = try generated.toOwnedSlice(self.allocator),
                    .finish_reason = .cancelled,
                    .prefill_ns = prefill_ns,
                    .decode_ns = 0,
                };
            }

            var stream_cb_ctx: ?StreamCbCtx = null;
            var backend_stream_cb: ?*const fn (u32, ?*anyopaque) void = null;
            var backend_stream_cb_data: ?*anyopaque = null;
            if (submit_config.callback) |cb| {
                stream_cb_ctx = .{
                    .submit_callback = cb,
                    .submit_callback_data = submit_config.callback_data,
                    .eos_token_ids = eos_token_ids,
                    .max_tail_tokens = remaining_token_budget,
                };
                backend_stream_cb = StreamCbCtx.onToken;
                backend_stream_cb_data = @ptrCast(&stream_cb_ctx.?);
            }

            const tail_count = try self.backend.decodeTopKStreaming(
                current_token,
                generated.items.len + 1,
                remaining_token_budget,
                eos_token_ids,
                sampling_config,
                generated_tail,
                backend_stream_cb,
                backend_stream_cb_data,
            );
            try generated.appendSlice(self.allocator, generated_tail[0..tail_count]);
            const decode_ns = decode_timer.read();

            const finish_reason: FinishReason = if (tail_count < remaining_token_budget) .eos_token else .length;
            return .{
                .tokens = try generated.toOwnedSlice(self.allocator),
                .finish_reason = finish_reason,
                .prefill_ns = prefill_ns,
                .decode_ns = decode_ns,
            };
        }

        fn generateSyncTopKCandidateRoute(
            self: *Self,
            prompt_tokens: []const u32,
            max_tokens: usize,
            submit_config: *const SubmitOptions,
            sampling_config: *const sampling.SamplingConfig,
        ) !GenerateSyncResult {
            if (max_tokens == 0) {
                return .{
                    .tokens = try self.allocator.dupe(u32, &.{}),
                    .finish_reason = .length,
                    .prefill_ns = 0,
                    .decode_ns = 0,
                };
            }

            if (generationShouldStop(submit_config.stop_flag)) {
                return .{
                    .tokens = try self.allocator.dupe(u32, &.{}),
                    .finish_reason = .cancelled,
                    .prefill_ns = 0,
                    .decode_ns = 0,
                };
            }

            const slot_index = self.backend.allocSlot() orelse return error.NoSlotsAvailable;
            defer self.backend.freeSlot(slot_index);

            const sp_blocks = if (self.slot_persistent_descs.len > 0)
                try self.slotStateBlocksForSlot(slot_index)
            else
                null;
            var np_blocks = try self.allocateRequestStateBlocks();
            defer {
                self.applyLifecycleActionToRequestStateBlocks(&np_blocks, .evict) catch {};
                np_blocks.deinit(self.allocator);
            }

            var merged: [runtime_contract.max_state_descriptors]runtime_contract.StateBlockHandle = undefined;
            var merge_count: usize = 0;
            if (sp_blocks) |spb| {
                for (spb.handles) |h| {
                    merged[merge_count] = h;
                    merge_count += 1;
                }
            }
            for (np_blocks.handles) |h| {
                merged[merge_count] = h;
                merge_count += 1;
            }
            if (merge_count > 0) {
                try self.backend.bindSlotStateBlocks(slot_index, merged[0..merge_count]);
            }
            defer self.backend.unbindSlotStateBlocks(slot_index);

            if (sampling_config.seed != 0) {
                self.sampler.reseed(sampling_config.seed);
            }

            var prefill_timer = std.time.Timer.start() catch unreachable;
            try self.prefillWithOptionalVision(slot_index, prompt_tokens, null);
            const prefill_ns = prefill_timer.read();
            if (generationShouldStop(submit_config.stop_flag)) {
                return .{
                    .tokens = try self.allocator.dupe(u32, &.{}),
                    .finish_reason = .cancelled,
                    .prefill_ns = prefill_ns,
                    .decode_ns = 0,
                };
            }

            const eos_token_ids = submit_config.eos_token_ids orelse self.config.default_eos_token_ids;
            var topk_sample_cfg = sampling_config.*;
            topk_sample_cfg.context_tokens = &.{};
            var current_token = self.sampleToken(self.logits_buffer, topk_sample_cfg, null) catch 0;

            var generated = try std.ArrayList(u32).initCapacity(self.allocator, max_tokens);
            errdefer generated.deinit(self.allocator);
            try generated.append(self.allocator, current_token);

            var finish_reason: FinishReason = .in_progress;
            for (eos_token_ids) |eos_id| {
                if (current_token == eos_id) {
                    finish_reason = .eos_token;
                    break;
                }
            }
            if (finish_reason == .in_progress and submit_config.stop_sequences.len > 0) {
                const stop_len = checkStopSequence(generated.items, submit_config.stop_sequences);
                if (stop_len > 0) {
                    finish_reason = .stop_sequence;
                    generated.shrinkRetainingCapacity(generated.items.len - stop_len);
                }
            }

            if (submit_config.callback) |cb| {
                const is_final = finish_reason != .in_progress or max_tokens == 1;
                if (finish_reason != .stop_sequence) {
                    cb(0, current_token, is_final, submit_config.max_thinking_tokens > 0 and submit_config.thinking_end_tokens.len > 0, submit_config.callback_data);
                }
            }

            if (finish_reason != .in_progress) {
                return .{
                    .tokens = try generated.toOwnedSlice(self.allocator),
                    .finish_reason = finish_reason,
                    .prefill_ns = prefill_ns,
                    .decode_ns = 0,
                };
            }
            if (max_tokens == 1) {
                return .{
                    .tokens = try generated.toOwnedSlice(self.allocator),
                    .finish_reason = .length,
                    .prefill_ns = prefill_ns,
                    .decode_ns = 0,
                };
            }

            const route_top_k: usize = switch (sampling_config.strategy) {
                .greedy => 1,
                .top_k => sampling_config.top_k,
                else => return error.NotEligible,
            };
            const max_candidate_count = @min(route_top_k, self.backend.vocabSize());
            var candidate_logits = try self.allocator.alloc(f32, max_candidate_count);
            defer self.allocator.free(candidate_logits);
            var candidate_ids = try self.allocator.alloc(u32, max_candidate_count);
            defer self.allocator.free(candidate_ids);
            const use_batched_topk_single = comptime blk: {
                if (!@hasDecl(BackendType, "decodeBatchTopKCandidates")) break :blk false;
                if (!@hasDecl(BackendType, "supportsSchedulerBackendBatchedTopKDecodeRoute")) break :blk false;
                break :blk true;
            };
            const use_topk_candidate_sampling_single = comptime blk: {
                if (!@hasDecl(BackendType, "decodeTopKCandidatesWithSampling")) break :blk false;
                if (!@hasDecl(BackendType, "supportsSchedulerBackendTopKCandidateSamplingRoute")) break :blk false;
                break :blk true;
            };
            const can_use_batched_topk_single = use_batched_topk_single and
                self.backend.supportsSchedulerBackendBatchedTopKDecodeRoute(sampling_config);
            const can_use_topk_candidate_sampling_single = use_topk_candidate_sampling_single and
                self.backend.supportsSchedulerBackendTopKCandidateSamplingRoute(sampling_config);
            const prefer_batched_topk_single = blk: {
                if (!comptime @hasDecl(std.process, "getEnvVarOwned")) break :blk true;
                const env_value = std.process.getEnvVarOwned(self.allocator, "TALU_METAL_TOPK_SINGLE_BATCH") catch |err| switch (err) {
                    error.EnvironmentVariableNotFound => break :blk true,
                    else => break :blk true,
                };
                defer self.allocator.free(env_value);
                if (env_value.len == 0) break :blk true;
                break :blk !std.ascii.eqlIgnoreCase(env_value, "0");
            };
            const use_batched_topk_for_single = can_use_batched_topk_single and prefer_batched_topk_single;
            var single_decode_request = [_]contract.DecodeRequest{.{
                .slot_index = slot_index,
                .token = 0,
            }};
            var single_candidate_count = [_]usize{0};

            // Thinking budget tracking: when max_thinking_tokens > 0, generation
            // starts in thinking mode (template prefills <think>\n). Once the budget
            // is exhausted, we force-inject the end sequence (</think>\n\n) then
            // resume normal sampling for the answer.
            const thinking_budget = submit_config.max_thinking_tokens;
            const thinking_end = submit_config.thinking_end_tokens;
            var thinking_tokens: usize = 0;
            var in_thinking: bool = thinking_budget > 0 and thinking_end.len > 0;
            var inject_pos: usize = 0; // position within thinking_end_tokens being injected

            // Completion token tracking: counts answer tokens after thinking ends.
            // Transitions one iteration after in_thinking becomes false so that
            // injected end-thinking tokens (</think>\n\n) are excluded.
            const max_completion_tokens = submit_config.max_completion_tokens;
            var completion_tokens: usize = 0;
            var generating_answer: bool = !in_thinking;

            var decode_timer = std.time.Timer.start() catch unreachable;
            while (generated.items.len < max_tokens) {
                if (generationShouldStop(submit_config.stop_flag)) {
                    return .{
                        .tokens = try generated.toOwnedSlice(self.allocator),
                        .finish_reason = .cancelled,
                        .prefill_ns = prefill_ns,
                        .decode_ns = decode_timer.read(),
                    };
                }

                topk_sample_cfg.context_tokens = generated.items;
                const use_backend_penalty_path = can_use_topk_candidate_sampling_single and
                    samplingRequiresFullLogits(topk_sample_cfg) and
                    topk_sample_cfg.logit_bias == null;
                // Run forward pass (updates KV cache with current_token).
                var candidate_count: usize = 0;
                if (use_backend_penalty_path) {
                    candidate_count = try self.backend.decodeTopKCandidatesWithSampling(
                        slot_index,
                        current_token,
                        &topk_sample_cfg,
                        candidate_logits,
                        candidate_ids,
                    );
                } else if (use_batched_topk_for_single) {
                    single_decode_request[0].token = current_token;
                    single_candidate_count[0] = 0;
                    try self.backend.decodeBatchTopKCandidates(
                        single_decode_request[0..],
                        max_candidate_count,
                        candidate_logits,
                        candidate_ids,
                        single_candidate_count[0..],
                    );
                    candidate_count = single_candidate_count[0];
                } else {
                    candidate_count = try self.backend.decodeTopKCandidates(
                        slot_index,
                        current_token,
                        max_candidate_count,
                        candidate_logits,
                        candidate_ids,
                    );
                }
                if (candidate_count == 0) return error.InvalidArgument;

                if (in_thinking and inject_pos > 0) {
                    // Injecting end-thinking tokens: force the next token instead of sampling.
                    current_token = thinking_end[inject_pos];
                    inject_pos += 1;
                    if (inject_pos >= thinking_end.len) {
                        in_thinking = false;
                    }
                } else if (in_thinking and thinking_tokens >= thinking_budget) {
                    // Budget exceeded: start injecting end sequence.
                    current_token = thinking_end[0];
                    inject_pos = 1;
                    if (thinking_end.len == 1) {
                        in_thinking = false;
                    }
                } else {
                    // Normal sampling.
                    current_token = if (canUseDirectGreedyCandidate(topk_sample_cfg, candidate_count))
                        candidate_ids[0]
                    else if (use_backend_penalty_path)
                        try self.sampleTopKCandidateTokenPrePenalized(
                            candidate_logits[0..candidate_count],
                            candidate_ids[0..candidate_count],
                            topk_sample_cfg,
                        )
                    else
                        try self.sampleTopKCandidateToken(
                            candidate_logits[0..candidate_count],
                            candidate_ids[0..candidate_count],
                            topk_sample_cfg,
                        );
                    if (in_thinking) {
                        thinking_tokens += 1;
                        // Check if model naturally produced the end-thinking token.
                        if (thinking_end.len > 0 and current_token == thinking_end[0]) {
                            in_thinking = false;
                        }
                    }
                }
                try generated.append(self.allocator, current_token);

                // Completion token limit enforcement.
                // Special tokens that decode to empty bytes are not counted.
                if (generating_answer) {
                    const has_visible = if (self.config.tokenizer) |tok|
                        (tok.tokenBytes(@intCast(current_token)) orelse &.{}).len > 0
                    else
                        true;
                    if (has_visible) completion_tokens += 1;
                    if (max_completion_tokens > 0 and completion_tokens >= max_completion_tokens) {
                        finish_reason = .length;
                        if (submit_config.callback) |cb| {
                            cb(0, current_token, true, in_thinking, submit_config.callback_data);
                        }
                        break;
                    }
                }
                if (!in_thinking and !generating_answer) {
                    generating_answer = true;
                }

                finish_reason = .in_progress;
                for (eos_token_ids) |eos_id| {
                    if (current_token == eos_id) {
                        finish_reason = .eos_token;
                        break;
                    }
                }
                if (finish_reason == .in_progress and submit_config.stop_sequences.len > 0) {
                    const stop_len = checkStopSequence(generated.items, submit_config.stop_sequences);
                    if (stop_len > 0) {
                        finish_reason = .stop_sequence;
                        generated.shrinkRetainingCapacity(generated.items.len - stop_len);
                    }
                }
                if (finish_reason == .in_progress and generated.items.len >= max_tokens) {
                    finish_reason = .length;
                }

                if (submit_config.callback) |cb| {
                    if (finish_reason != .stop_sequence) {
                        cb(0, current_token, finish_reason != .in_progress, in_thinking, submit_config.callback_data);
                    }
                }
                if (finish_reason != .in_progress) break;
            }

            return .{
                .tokens = try generated.toOwnedSlice(self.allocator),
                .finish_reason = finish_reason,
                .prefill_ns = prefill_ns,
                .decode_ns = decode_timer.read(),
            };
        }

        /// Result from generateSync.
        pub const GenerateSyncResult = struct {
            /// Generated tokens (owned, caller must free).
            tokens: []u32,
            /// Final-step logits (owned when non-empty).
            final_logits: []f32 = &.{},
            /// Reason generation stopped.
            finish_reason: FinishReason,
            /// Prefill time in nanoseconds.
            prefill_ns: u64,
            /// Decode time in nanoseconds.
            decode_ns: u64,

            pub fn deinit(self: *GenerateSyncResult, allocator: std.mem.Allocator) void {
                allocator.free(self.tokens);
                if (self.final_logits.len > 0) allocator.free(self.final_logits);
                self.* = undefined;
            }
        };

        // =========================================================================
        // Internal helpers
        // =========================================================================

        fn applyLifecycleActionToRequestStateBlocks(
            self: *Self,
            request_state_blocks: *RequestStateBlocks,
            action: runtime_contract.StateLifecycleAction,
        ) !void {
            _ = self;
            if (request_state_blocks.descriptors.len == 0) return;
            if (request_state_blocks.descriptors.len != request_state_blocks.storage.len) {
                return error.InvalidStateDescriptorBinding;
            }
            for (request_state_blocks.descriptors, 0..) |descriptor, idx| {
                const should_zero = runtime_contract.shouldZeroStateForLifecycleAction(&descriptor, action) catch |err| switch (err) {
                    error.InvalidStateLifecycleAction => false,
                    else => return err,
                };
                if (should_zero) {
                    @memset(request_state_blocks.storage[idx].bytes, 0);
                }
            }
        }

        fn resetRequestStateBlocks(self: *Self, request_id: u64) !void {
            const request_state_blocks = self.request_state_blocks.getPtr(request_id) orelse return;
            try self.applyLifecycleActionToRequestStateBlocks(request_state_blocks, .reset);
        }

        /// Reset step-scoped state blocks to zero within a RequestStateBlocks.
        fn applyStepScopedReset(blocks: *RequestStateBlocks) !void {
            for (blocks.descriptors, 0..) |desc, idx| {
                if (desc.lifecycle != .step_scoped) continue;
                const should_zero = runtime_contract.shouldZeroStateForLifecycleAction(&desc, .reset) catch |err| switch (err) {
                    error.InvalidStateLifecycleAction => continue,
                    else => return err,
                };
                if (should_zero) {
                    @memset(blocks.storage[idx].bytes, 0);
                }
            }
        }

        /// Reset step-scoped state blocks to zero for a given request.
        /// Called at each scheduler step boundary before decode.
        fn resetStepScopedBlocks(self: *Self, request_id: u64) !void {
            const blocks = self.request_state_blocks.getPtr(request_id) orelse return;
            try applyStepScopedReset(blocks);
        }

        fn allocateStateBlockStorage(self: *Self, descriptor: runtime_contract.StateDescriptor) !StateBlockStorage {
            if (descriptor.align_bytes == 0 or descriptor.align_bytes > 64) {
                return error.InvalidStateDescriptorBinding;
            }
            const descriptor_size = std.math.cast(usize, descriptor.size_bytes) orelse return error.InvalidStateDescriptorBinding;
            if (descriptor_size == 0) return error.InvalidStateDescriptorBinding;
            const size_bytes: usize = descriptor_size;
            const bytes = try self.allocator.alignedAlloc(u8, .@"64", size_bytes);
            const should_zero_on_alloc = try runtime_contract.shouldZeroStateForLifecycleAction(&descriptor, .alloc);
            if (should_zero_on_alloc) {
                @memset(bytes, 0);
            }
            return .{
                .bytes = bytes,
            };
        }

        fn allocateStateBlocksForDescriptors(self: *Self, descriptors: []const runtime_contract.StateDescriptor) !RequestStateBlocks {
            if (descriptors.len == 0) return .{};

            var request_state_blocks = RequestStateBlocks{};
            request_state_blocks.handles = try self.allocator.alloc(runtime_contract.StateBlockHandle, descriptors.len);
            errdefer self.allocator.free(request_state_blocks.handles);
            request_state_blocks.storage = try self.allocator.alloc(StateBlockStorage, descriptors.len);
            errdefer self.allocator.free(request_state_blocks.storage);
            request_state_blocks.descriptors = try self.allocator.alloc(runtime_contract.StateDescriptor, descriptors.len);
            errdefer self.allocator.free(request_state_blocks.descriptors);
            @memcpy(request_state_blocks.descriptors, descriptors);

            var initialized: usize = 0;
            errdefer {
                for (request_state_blocks.storage[0..initialized]) |entry| {
                    self.allocator.free(entry.bytes);
                }
            }

            for (descriptors, 0..) |descriptor, idx| {
                const storage = try self.allocateStateBlockStorage(descriptor);
                request_state_blocks.storage[idx] = storage;
                request_state_blocks.handles[idx] = .{
                    .id = descriptor.id,
                    .ptr = @ptrCast(storage.bytes.ptr),
                    .size = @intCast(storage.bytes.len),
                    .align_bytes = descriptor.align_bytes,
                };
                initialized += 1;
            }

            return request_state_blocks;
        }

        /// Allocate request-owned (non-persistent) state blocks.
        /// This keeps request-scoped and step-scoped lifecycles explicit in the scheduler contract.
        fn allocateRequestStateBlocks(self: *Self) !RequestStateBlocks {
            return self.allocateStateBlocksForDescriptors(self.non_persistent_descs);
        }

        fn slotStateBlocksForSlot(self: *Self, slot_index: usize) !*RequestStateBlocks {
            if (self.slot_state_blocks.getPtr(slot_index)) |existing| {
                try self.applyLifecycleActionToRequestStateBlocks(existing, .reuse);
                return existing;
            }

            var allocated = try self.allocateStateBlocksForDescriptors(self.slot_persistent_descs);
            errdefer allocated.deinit(self.allocator);
            try self.slot_state_blocks.put(slot_index, allocated);
            return self.slot_state_blocks.getPtr(slot_index) orelse return error.InvalidStateDescriptorBinding;
        }

        fn bindAndTrackRequestStateBlocks(self: *Self, request_id: u64, slot_index: usize) !void {
            var merged: [runtime_contract.max_state_descriptors]runtime_contract.StateBlockHandle = undefined;
            var count: usize = 0;

            // 1. Get slot-persistent blocks if any exist
            if (self.slot_persistent_descs.len > 0) {
                const slot_blocks = try self.slotStateBlocksForSlot(slot_index);
                for (slot_blocks.handles) |h| {
                    merged[count] = h;
                    count += 1;
                }
            }

            // 2. Allocate non-persistent blocks if any exist
            if (self.non_persistent_descs.len > 0) {
                if (self.request_state_blocks.contains(request_id)) {
                    return error.InvalidStateDescriptorBinding;
                }
                var np_blocks = try self.allocateRequestStateBlocks();
                errdefer np_blocks.deinit(self.allocator);
                try self.request_state_blocks.put(request_id, np_blocks);
                for (np_blocks.handles) |h| {
                    merged[count] = h;
                    count += 1;
                }
            }

            // 3. Bind merged handles
            if (count > 0) {
                try self.backend.bindSlotStateBlocks(slot_index, merged[0..count]);
            }
        }

        fn releaseRequestStateBlocks(self: *Self, request_id: u64, slot_index: ?usize) void {
            if (slot_index) |slot| {
                self.backend.unbindSlotStateBlocks(slot);
            }
            // Free only non-persistent blocks; slot-persistent blocks remain in slot_state_blocks
            if (self.request_state_blocks.fetchRemove(request_id)) |entry| {
                var state_blocks = entry.value;
                self.applyLifecycleActionToRequestStateBlocks(&state_blocks, .evict) catch {};
                state_blocks.deinit(self.allocator);
            }
        }

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
                    errdefer self.backend.freeSlot(slot_index);
                    try self.bindAndTrackRequestStateBlocks(pending_request_id, slot_index);
                    errdefer self.releaseRequestStateBlocks(pending_request_id, slot_index);
                    request_entry.slot_index = slot_index;
                    request_entry.state = .pending_prefill;
                    try self.active_requests.append(self.allocator, pending_request_id);
                    if (slot_index < self.slot_request_ids.len) {
                        self.slot_request_ids[slot_index] = pending_request_id;
                    }
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

        fn beginPrefillGeneration(self: *Self, request_entry: *Request) void {
            request_entry.token_position = request_entry.prompt_tokens.len;
            request_entry.state = .generating;

            // Reseed sampler if seed specified (ensures deterministic output with same seed)
            if (request_entry.sampling_config.seed != 0) {
                self.sampler.reseed(request_entry.sampling_config.seed);
            }
        }

        fn handlePrefillToken(
            self: *Self,
            request_entry: *Request,
            first_token_id: u32,
            slot_index: usize,
            prefill_logits: []const f32,
        ) !void {
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

            // Completion token limit enforcement (answer tokens only).
            // Special tokens that decode to empty bytes are not counted.
            if (finish_reason == .in_progress and request_entry.generating_answer and
                request_entry.max_completion_tokens_limit > 0)
            {
                const has_visible = if (self.config.tokenizer) |tok|
                    (tok.tokenBytes(first_token_id) orelse &.{}).len > 0
                else
                    true;
                if (has_visible) {
                    request_entry.completion_token_count += 1;
                    if (request_entry.completion_token_count >= request_entry.max_completion_tokens_limit) {
                        finish_reason = .length;
                    }
                }
            }

            const is_final_token = finish_reason != .in_progress;

            if (request_entry.callback) |cb| {
                if (finish_reason != .stop_sequence) {
                    cb(request_entry.id, first_token_id, is_final_token, request_entry.in_thinking, request_entry.callback_data);
                }
            }

            // Emit step event for the first token so batch consumers
            // (which read step_events rather than callbacks) see it.
            if (finish_reason != .stop_sequence) {
                try self.step_events.append(self.allocator, .{
                    .request_id = request_entry.id,
                    .token = first_token_id,
                    .is_final = is_final_token,
                    .in_thinking = request_entry.in_thinking,
                    .slot_index = slot_index,
                    .timestamp_ns = std.time.nanoTimestamp(),
                });
            }

            if (is_final_token) {
                if (request_entry.capture_final_logits and request_entry.final_logits.len == 0) {
                    request_entry.final_logits = try self.captureFinalLogits(true, prefill_logits);
                }
                self.completeRequest(request_entry, finish_reason);
            }
        }

        fn runSinglePrefill(self: *Self, request_entry: *Request) !void {
            if (request_entry.slot_index == null) return;
            const slot_index = request_entry.slot_index.?;

            // Run prefill for this request
            var prefill_timer = std.time.Timer.start() catch unreachable;
            try self.resetRequestStateBlocks(request_entry.id);
            try self.prefillWithOptionalVision(
                slot_index,
                request_entry.prompt_tokens,
                request_entry.vision_input,
            );
            request_entry.prefill_ns = prefill_timer.read();

            self.beginPrefillGeneration(request_entry);

            if (trace.isEnabled()) {
                trace.emitFinal(
                    .logits_ready,
                    @intCast(request_entry.generated_tokens.items.len),
                    @intCast(request_entry.token_position),
                    @ptrCast(self.logits_buffer.ptr),
                    .f32,
                    .{ @intCast(self.logits_buffer.len), 0, 0, 0 },
                    1,
                    "prefill_logits",
                );
            }

            // Sample first token from prefill logits
            var prefill_sample_cfg = request_entry.sampling_config;
            prefill_sample_cfg.context_tokens = request_entry.generated_tokens.items;
            const first_token_id = self.sampleToken(
                self.logits_buffer,
                prefill_sample_cfg,
                request_entry.grammar_sampler,
            ) catch 0;
            self.parityLogits(
                "prefill",
                request_entry.token_position,
                self.logits_buffer,
                first_token_id,
                slot_index,
            );

            try self.handlePrefillToken(
                request_entry,
                first_token_id,
                slot_index,
                self.logits_buffer,
            );
        }

        fn prefillBatchEligibleRequestCount(self: *Self) usize {
            if (!(comptime @hasDecl(BackendType, "prefillBatch"))) return 0;

            var count: usize = 0;
            for (self.active_requests.items) |active_request_id| {
                const request_entry = self.requests.get(active_request_id) orelse continue;
                if (request_entry.state != .pending_prefill) continue;
                if (request_entry.slot_index == null) continue;
                if (request_entry.vision_input != null) continue;
                if (count >= self.prefill_request_ids.len) break;
                self.prefill_request_ids[count] = active_request_id;
                count += 1;
            }
            return count;
        }

        fn runBatchedPrefills(self: *Self, candidate_count: usize) !void {
            if (!(comptime @hasDecl(BackendType, "prefillBatch"))) return;
            if (candidate_count == 0) return;

            const vocab = self.backend.vocabSize();

            for (0..candidate_count) |idx| {
                const request_id = self.prefill_request_ids[idx];
                const request_entry = self.requests.get(request_id) orelse return error.InvalidState;
                const slot_index = request_entry.slot_index orelse return error.InvalidState;
                try self.resetRequestStateBlocks(request_id);
                const row_start = idx * vocab;
                const row_end = row_start + vocab;
                self.prefill_requests[idx] = .{
                    .slot_index = slot_index,
                    .prompt_tokens = request_entry.prompt_tokens,
                    .logits_out = self.prefill_logits_buffer[row_start..row_end],
                };
            }

            var prefill_timer = std.time.Timer.start() catch unreachable;
            try self.backend.prefillBatch(self.prefill_requests[0..candidate_count]);
            const prefill_ns = prefill_timer.read();

            for (0..candidate_count) |idx| {
                const request_id = self.prefill_request_ids[idx];
                const request_entry = self.requests.get(request_id) orelse continue;
                if (request_entry.state != .pending_prefill) continue;
                const slot_index = request_entry.slot_index orelse return error.InvalidState;

                // Amortize the shared batch wall-time across requests so
                // per-request prefill metrics aggregate correctly.
                request_entry.prefill_ns = amortizeBatchPrefillNs(prefill_ns, idx, candidate_count);
                self.beginPrefillGeneration(request_entry);

                const logits = self.prefill_requests[idx].logits_out;
                if (trace.isEnabled()) {
                    trace.emitFinal(
                        .logits_ready,
                        @intCast(request_entry.generated_tokens.items.len),
                        @intCast(request_entry.token_position),
                        @ptrCast(logits.ptr),
                        .f32,
                        .{ @intCast(logits.len), 0, 0, 0 },
                        1,
                        "prefillBatch_logits",
                    );
                }
                var prefill_sample_cfg = request_entry.sampling_config;
                prefill_sample_cfg.context_tokens = request_entry.generated_tokens.items;
                const first_token_id = self.sampleToken(
                    logits,
                    prefill_sample_cfg,
                    request_entry.grammar_sampler,
                ) catch 0;
                self.parityLogits(
                    "prefill",
                    request_entry.token_position,
                    logits,
                    first_token_id,
                    slot_index,
                );

                try self.handlePrefillToken(
                    request_entry,
                    first_token_id,
                    slot_index,
                    logits,
                );
            }
        }

        fn amortizeBatchPrefillNs(total_ns: u64, request_index: usize, request_count: usize) u64 {
            if (request_count == 0) return total_ns;
            const count_u64: u64 = @intCast(request_count);
            const base = total_ns / count_u64;
            const remainder = total_ns % count_u64;
            const request_index_u64: u64 = @intCast(request_index);
            const bonus: u64 = if (request_index_u64 < remainder) 1 else 0;
            return base + bonus;
        }

        fn runPrefills(self: *Self) !void {
            // Batch prefill requests when backend supports it and there is more
            // than one pending prefill. This removes per-request prefill
            // synchronization barriers and allows requests to enter decode
            // together at the next token boundary.
            const batchable_count = self.prefillBatchEligibleRequestCount();
            if (batchable_count >= 2) {
                try self.runBatchedPrefills(batchable_count);
            }

            for (self.active_requests.items) |active_request_id| {
                const request_entry = self.requests.get(active_request_id) orelse continue;
                if (request_entry.state != .pending_prefill) continue;
                try self.runSinglePrefill(request_entry);
            }
        }

        fn completeRequest(self: *Self, request_entry: *Request, reason: FinishReason) void {
            request_entry.state = .completed;
            request_entry.finish_reason = reason;

            // Free slot
            if (request_entry.slot_index) |slot_index| {
                self.releaseRequestStateBlocks(request_entry.id, slot_index);
                if (slot_index < self.slot_request_ids.len) {
                    self.slot_request_ids[slot_index] = null;
                }
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

        /// Build final TokenEvents for requests that completed during runPrefills.
        ///
        /// When max_tokens ≤ 1 or the first token is EOS, runPrefills completes
        /// the request and moves it to completed_queue without producing events.
        /// Batch callers (as opposed to generateSync) rely on step() events to
        /// observe completion, so we synthesize them here.
        fn buildPrefillCompletedEvents(self: *Self, completed_before: usize) ![]const TokenEvent {
            self.step_events.clearRetainingCapacity();
            try self.appendPrefillCompletedEvents(&self.step_events, completed_before);
            return self.step_events.items;
        }

        fn appendPrefillCompletedEvents(
            self: *Self,
            events: *std.ArrayList(TokenEvent),
            completed_before: usize,
        ) !void {
            for (self.completed_queue.items[completed_before..]) |cid| {
                const req = self.requests.get(cid) orelse continue;
                const last_token = if (req.generated_tokens.items.len > 0)
                    req.generated_tokens.items[req.generated_tokens.items.len - 1]
                else
                    0;
                try events.append(self.allocator, .{
                    .request_id = cid,
                    .token = last_token,
                    .is_final = true,
                    .in_thinking = false,
                    .slot_index = 0,
                    .timestamp_ns = std.time.nanoTimestamp(),
                });
            }
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
        .in_thinking = false,
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
        .in_thinking = true,
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
        .in_thinking = false,
        .slot_index = 0,
    };

    const final = TokenEvent{
        .request_id = 1,
        .token = 0, // EOS token
        .is_final = true,
        .in_thinking = false,
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
    prefill_batch_calls: usize = 0,
    decode_calls: std.ArrayList(DecodeCall),
    /// Track logits allocations for cleanup in tests
    allocated_logits: std.ArrayList([]f32),
    /// Token that always wins greedy sampling (default: 100)
    greedy_token: usize = 100,
    state_descriptors: []const runtime_contract.StateDescriptor = &.{},
    bind_slot_state_blocks_calls: usize = 0,
    unbind_slot_state_blocks_calls: usize = 0,
    last_bound_state_block_count: usize = 0,
    saw_state_blocks_valid: bool = true,
    saw_zero_init_shortconv_state: bool = false,
    first_bound_state_ptr: ?usize = null,
    second_bound_state_ptr: ?usize = null,

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

    fn setStateDescriptors(self: *MockBackend, descriptors: []const runtime_contract.StateDescriptor) void {
        self.state_descriptors = descriptors;
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

    fn stateDescriptors(self: *const MockBackend) []const runtime_contract.StateDescriptor {
        return self.state_descriptors;
    }

    fn bindSlotStateBlocks(
        self: *MockBackend,
        slot_index: usize,
        state_blocks: []const runtime_contract.StateBlockHandle,
    ) !void {
        self.bind_slot_state_blocks_calls += 1;
        self.last_bound_state_block_count = state_blocks.len;
        try runtime_contract.validateStateBlocksForDescriptors(self.stateDescriptors(), state_blocks);
        if (slot_index == 0 and state_blocks.len > 0) {
            const state_ptr_value: usize = @intFromPtr(state_blocks[0].ptr);
            if (self.first_bound_state_ptr == null) {
                self.first_bound_state_ptr = state_ptr_value;
            } else if (self.second_bound_state_ptr == null) {
                self.second_bound_state_ptr = state_ptr_value;
            }
        }
        const descriptors = self.stateDescriptors();
        const shortconv_id = runtime_contract.shortconv_state_id;
        for (descriptors) |descriptor| {
            const incoming = runtime_contract.findStateBlock(state_blocks, descriptor.id) orelse {
                self.saw_state_blocks_valid = false;
                return error.InvalidStateDescriptorBinding;
            };
            if (incoming.align_bytes < descriptor.align_bytes) self.saw_state_blocks_valid = false;
            if (descriptor.size_bytes > 0 and incoming.size < descriptor.size_bytes) self.saw_state_blocks_valid = false;
            if (descriptor.id == shortconv_id and descriptor.zero_init and incoming.size > 0) {
                const first_byte = incoming.ptr[0];
                if (first_byte == 0) self.saw_zero_init_shortconv_state = true;
            }
        }
    }

    fn unbindSlotStateBlocks(self: *MockBackend, slot_index: usize) void {
        _ = slot_index;
        self.unbind_slot_state_blocks_calls += 1;
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

    fn prefillBatch(self: *MockBackend, requests: []const contract.PrefillBatchRequest) !void {
        self.prefill_batch_calls += 1;
        for (requests) |request_entry| {
            try self.prefillSlot(
                request_entry.slot_index,
                request_entry.prompt_tokens,
                request_entry.logits_out,
            );
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

/// Mock backend used by generateSync route selection tests.
const MockStreamingBackend = struct {
    const PrefillVisionInput = struct {
        tag: u8 = 0,
    };

    allocator: std.mem.Allocator,
    vocab_size: usize,
    slot_in_use: bool = false,
    decode_batch_calls: usize = 0,
    decode_streaming_calls: usize = 0,
    decode_top_k_streaming_calls: usize = 0,
    decode_top_k_calls: usize = 0,
    prefill_with_vision_calls: usize = 0,
    allocated_logits: std.ArrayList([]f32),
    greedy_token: usize = 42,
    alternate_top_k_order: bool = false,
    flat_top_k_logits: bool = false,

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

    fn supportsSchedulerBackendDecodeStreamingRoute(self: *const MockStreamingBackend) bool {
        _ = self;
        return true;
    }

    fn supportsSchedulerBackendTopKDecodeRoute(
        self: *const MockStreamingBackend,
        sampling_config: *const sampling.SamplingConfig,
    ) bool {
        _ = self;
        return sampling_config.strategy == .top_k and
            sampling_config.top_k > 0 and
            sampling_config.temperature > 0.0 and
            sampling_config.min_p == 0.0;
    }

    fn supportsSchedulerBackendTopKStreamingRoute(
        self: *const MockStreamingBackend,
        sampling_config: *const sampling.SamplingConfig,
    ) bool {
        _ = self;
        return sampling_config.strategy == .top_k and
            sampling_config.top_k > 0 and
            sampling_config.temperature > 0.0 and
            sampling_config.min_p == 0.0 and
            sampling_config.repetition_penalty == 1.0 and
            sampling_config.presence_penalty == 0.0 and
            sampling_config.frequency_penalty == 0.0 and
            sampling_config.logit_bias == null;
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

    fn stateDescriptors(self: *const MockStreamingBackend) []const runtime_contract.StateDescriptor {
        _ = self;
        return &.{};
    }

    fn bindSlotStateBlocks(
        self: *MockStreamingBackend,
        slot_index: usize,
        state_blocks: []const runtime_contract.StateBlockHandle,
    ) !void {
        _ = slot_index;
        try runtime_contract.validateStateBlocksForDescriptors(self.stateDescriptors(), state_blocks);
    }

    fn unbindSlotStateBlocks(self: *MockStreamingBackend, slot_index: usize) void {
        _ = self;
        _ = slot_index;
    }

    fn prefillSlot(self: *MockStreamingBackend, slot_index: usize, tokens: []const u32, logits_out: []f32) !void {
        _ = self;
        _ = slot_index;
        _ = tokens;
        for (logits_out, 0..) |*logit, idx| {
            logit.* = if (idx == 42) 10.0 else 1.0;
        }
    }

    fn prefillSlotWithVision(
        self: *MockStreamingBackend,
        slot_index: usize,
        tokens: []const u32,
        vision_input: ?*const PrefillVisionInput,
        logits_out: []f32,
    ) !void {
        _ = slot_index;
        _ = tokens;
        if (vision_input == null) return error.InvalidArgument;
        self.prefill_with_vision_calls += 1;
        for (logits_out, 0..) |*logit, idx| {
            logit.* = if (idx == self.greedy_token) 10.0 else 1.0;
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
        self.decode_streaming_calls += 1;
        for (output_tokens[0..max_tokens], 0..) |*out_token, idx| {
            out_token.* = first_token + @as(u32, @intCast(start_position + idx));
            if (callback) |cb| cb(out_token.*, callback_data);
        }
        return max_tokens;
    }

    fn decodeTopKStreaming(
        self: *MockStreamingBackend,
        first_token: u32,
        start_position: usize,
        max_tokens: usize,
        eos_token_ids: []const u32,
        sampling_config: *const sampling.SamplingConfig,
        output_tokens: []u32,
        callback: ?*const fn (u32, ?*anyopaque) void,
        callback_data: ?*anyopaque,
    ) !usize {
        _ = eos_token_ids;
        _ = sampling_config;
        self.decode_top_k_streaming_calls += 1;
        for (output_tokens[0..max_tokens], 0..) |*out_token, idx| {
            out_token.* = first_token + @as(u32, @intCast(start_position + idx));
            if (callback) |cb| cb(out_token.*, callback_data);
        }
        return max_tokens;
    }

    fn decodeTopKCandidates(
        self: *MockStreamingBackend,
        slot_index: usize,
        token: u32,
        top_k: usize,
        candidate_logits_out: []f32,
        candidate_ids_out: []u32,
    ) !usize {
        _ = slot_index;
        _ = token;
        self.decode_top_k_calls += 1;
        const count = @min(top_k, candidate_logits_out.len);
        if (count == 0 or candidate_ids_out.len < count) return error.InvalidArgument;
        if (self.flat_top_k_logits and count >= 2) {
            const base_id: u32 = @intCast(self.greedy_token);
            const flip_order = self.alternate_top_k_order and (self.decode_top_k_calls % 2 == 1);
            candidate_logits_out[0] = 0.0;
            candidate_logits_out[1] = 0.0;
            candidate_ids_out[0] = if (flip_order) base_id + 1 else base_id;
            candidate_ids_out[1] = if (flip_order) base_id else base_id + 1;
            for (2..count) |idx| {
                candidate_logits_out[idx] = -10.0 - @as(f32, @floatFromInt(idx));
                candidate_ids_out[idx] = base_id + @as(u32, @intCast(idx));
            }
            return count;
        }
        for (candidate_logits_out[0..count], 0..) |*logit, idx| {
            logit.* = if (idx == 0) 10.0 else -10.0 - @as(f32, @floatFromInt(idx));
        }
        candidate_ids_out[0] = @intCast(self.greedy_token);
        for (candidate_ids_out[1..count], 1..) |*token_id, idx| {
            token_id.* = @intCast(self.greedy_token + idx);
        }
        return count;
    }
};

const MockStreamingScheduler = GenericScheduler(MockStreamingBackend);

// =============================================================================
// Scheduler Integration Tests - init/deinit
// =============================================================================

test "generateSync uses backend decode-streaming route for greedy sampling" {
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
    try std.testing.expectEqualSlices(u32, &.{ 42, 44, 45, 46 }, result.tokens);
}

test "generateSync callback uses backend decode-streaming route for greedy sampling" {
    const alloc = std.testing.allocator;
    var backend = MockStreamingBackend.init(alloc, 256);
    defer backend.deinit();

    var scheduler = try MockStreamingScheduler.init(alloc, &backend, .{
        .default_sampling = .{
            .strategy = .greedy,
        },
    });
    defer scheduler.deinit();

    const CallbackData = struct {
        token_count: usize = 0,
        final_count: usize = 0,
    };
    var callback_data = CallbackData{};

    const callback = struct {
        fn cb(_: u64, _: u32, is_final: bool, _: bool, user_data: ?*anyopaque) void {
            const data: *CallbackData = @ptrCast(@alignCast(user_data.?));
            data.token_count += 1;
            if (is_final) data.final_count += 1;
        }
    }.cb;

    var result = try scheduler.generateSync(&[_]u32{ 11, 12 }, 4, .{
        .sampling = .{
            .strategy = .greedy,
        },
        .callback = callback,
        .callback_data = &callback_data,
        .eos_token_ids = &[_]u32{9999},
    });
    defer result.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 1), backend.decode_streaming_calls);
    try std.testing.expectEqual(@as(usize, 0), backend.decode_batch_calls);
    try std.testing.expectEqual(@as(usize, 4), callback_data.token_count);
    try std.testing.expectEqual(@as(usize, 1), callback_data.final_count);
}

test "generateSync uses decodeBatch route with vision input" {
    const alloc = std.testing.allocator;
    var backend = MockStreamingBackend.init(alloc, 256);
    defer backend.deinit();

    var scheduler = try MockStreamingScheduler.init(alloc, &backend, .{
        .default_sampling = .{
            .strategy = .greedy,
        },
    });
    defer scheduler.deinit();

    const vision_input = MockStreamingBackend.PrefillVisionInput{ .tag = 1 };
    var result = try scheduler.generateSync(&[_]u32{ 11, 12 }, 4, .{
        .sampling = .{
            .strategy = .greedy,
        },
        .eos_token_ids = &[_]u32{9999},
        .vision_input = @ptrCast(&vision_input),
    });
    defer result.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 1), backend.prefill_with_vision_calls);
    try std.testing.expectEqual(@as(usize, 0), backend.decode_streaming_calls);
    try std.testing.expect(backend.decode_batch_calls > 0);
    try std.testing.expectEqual(@as(usize, 4), result.tokens.len);
}

test "generateSync uses decodeBatch when stop sequences are configured" {
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

test "generateSync uses backend top-k candidate route for top_k sampling" {
    const alloc = std.testing.allocator;
    var backend = MockStreamingBackend.init(alloc, 256);
    defer backend.deinit();

    var scheduler = try MockStreamingScheduler.init(alloc, &backend, .{
        .default_sampling = .{
            .strategy = .top_k,
            .top_k = 4,
            .temperature = 0.7,
            .seed = 7,
        },
    });
    defer scheduler.deinit();

    var result = try scheduler.generateSync(&[_]u32{ 7, 8, 9 }, 4, .{
        .sampling = .{
            .strategy = .top_k,
            .top_k = 4,
            .temperature = 0.7,
            .seed = 7,
        },
        .eos_token_ids = &[_]u32{9999},
    });
    defer result.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 0), backend.decode_streaming_calls);
    try std.testing.expectEqual(@as(usize, 3), backend.decode_top_k_calls);
    try std.testing.expectEqual(@as(usize, 0), backend.decode_batch_calls);
    try std.testing.expectEqual(@as(usize, 4), result.tokens.len);
    try std.testing.expectEqualSlices(u32, &.{ 42, 42, 42, 42 }, result.tokens);
}

test "generateSync uses backend top-k streaming route for top_k sampling without penalties" {
    const alloc = std.testing.allocator;
    var backend = MockStreamingBackend.init(alloc, 256);
    defer backend.deinit();

    var scheduler = try MockStreamingScheduler.init(alloc, &backend, .{
        .default_sampling = .{
            .strategy = .top_k,
            .top_k = 4,
            .temperature = 0.7,
            .seed = 0,
        },
    });
    defer scheduler.deinit();

    var result = try scheduler.generateSync(&[_]u32{ 7, 8, 9 }, 4, .{
        .sampling = .{
            .strategy = .top_k,
            .top_k = 4,
            .temperature = 0.7,
            .seed = 0,
        },
        .eos_token_ids = &[_]u32{9999},
    });
    defer result.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 1), backend.decode_top_k_streaming_calls);
    try std.testing.expectEqual(@as(usize, 0), backend.decode_top_k_calls);
    try std.testing.expectEqual(@as(usize, 0), backend.decode_streaming_calls);
    try std.testing.expectEqual(@as(usize, 0), backend.decode_batch_calls);
    try std.testing.expectEqual(@as(usize, 4), result.tokens.len);
    try std.testing.expectEqualSlices(u32, &.{ 42, 44, 45, 46 }, result.tokens);
}

test "generateSync uses top-k candidate route with additive penalties" {
    const alloc = std.testing.allocator;
    var backend = MockStreamingBackend.init(alloc, 256);
    defer backend.deinit();

    var scheduler = try MockStreamingScheduler.init(alloc, &backend, .{
        .default_sampling = .{
            .strategy = .top_k,
            .top_k = 4,
            .temperature = 0.7,
            .presence_penalty = 0.5,
            .seed = 7,
        },
    });
    defer scheduler.deinit();

    var result = try scheduler.generateSync(&[_]u32{ 7, 8, 9 }, 3, .{
        .sampling = .{
            .strategy = .top_k,
            .top_k = 4,
            .temperature = 0.7,
            .presence_penalty = 0.5,
            .seed = 7,
        },
        .eos_token_ids = &[_]u32{9999},
    });
    defer result.deinit(alloc);

    // Top-k route handles penalties via applyCandidatePenalties.
    try std.testing.expect(backend.decode_top_k_calls > 0);
    try std.testing.expectEqual(@as(usize, 0), backend.decode_streaming_calls);
    try std.testing.expectEqual(@as(usize, 3), result.tokens.len);
}

test "generateSync uses top-k candidate route with logit bias" {
    const alloc = std.testing.allocator;
    var backend = MockStreamingBackend.init(alloc, 256);
    defer backend.deinit();

    const bias_entries = [_]sampling.LogitBiasEntry{
        .{ .token_id = 42, .bias = -3.0 },
    };

    var scheduler = try MockStreamingScheduler.init(alloc, &backend, .{
        .default_sampling = .{
            .strategy = .top_k,
            .top_k = 4,
            .temperature = 0.7,
            .logit_bias = &bias_entries,
            .seed = 7,
        },
    });
    defer scheduler.deinit();

    var result = try scheduler.generateSync(&[_]u32{ 7, 8, 9 }, 3, .{
        .sampling = .{
            .strategy = .top_k,
            .top_k = 4,
            .temperature = 0.7,
            .logit_bias = &bias_entries,
            .seed = 7,
        },
        .eos_token_ids = &[_]u32{9999},
    });
    defer result.deinit(alloc);

    // Top-k route handles logit bias via applyCandidatePenalties.
    try std.testing.expect(backend.decode_top_k_calls > 0);
    try std.testing.expectEqual(@as(usize, 0), backend.decode_streaming_calls);
    try std.testing.expectEqual(@as(usize, 3), result.tokens.len);
}

test "generateSync disables greedy streaming when repetition penalty is set" {
    const alloc = std.testing.allocator;
    var backend = MockStreamingBackend.init(alloc, 256);
    defer backend.deinit();

    var scheduler = try MockStreamingScheduler.init(alloc, &backend, .{
        .default_sampling = .{
            .strategy = .greedy,
            .repetition_penalty = 1.2,
            .seed = 7,
        },
    });
    defer scheduler.deinit();

    var result = try scheduler.generateSync(&[_]u32{ 11, 12 }, 3, .{
        .sampling = .{
            .strategy = .greedy,
            .repetition_penalty = 1.2,
            .seed = 7,
        },
        .eos_token_ids = &[_]u32{9999},
    });
    defer result.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 0), backend.decode_streaming_calls);
    try std.testing.expectEqual(@as(usize, 0), backend.decode_top_k_calls);
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
    _ = try scheduler.step();

    // Should have called prefillSlot
    try std.testing.expectEqual(@as(usize, 1), backend.prefill_calls.items.len);

    const request_entry = scheduler.requests.get(request_id).?;
    try std.testing.expectEqual(RequestState.generating, request_entry.state);
}

test "Scheduler.step - batches prefills when backend supports prefillBatch" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    _ = try scheduler.submit(&prompt, 10, null);
    _ = try scheduler.submit(&prompt, 10, null);

    _ = try scheduler.step();

    try std.testing.expectEqual(@as(usize, 1), backend.prefill_batch_calls);
    try std.testing.expectEqual(@as(usize, 2), backend.prefill_calls.items.len);
}

test "Scheduler.amortizeBatchPrefillNs splits total deterministically" {
    const total_ns: u64 = 10;
    const request_count: usize = 3;

    const part0 = MockScheduler.amortizeBatchPrefillNs(total_ns, 0, request_count);
    const part1 = MockScheduler.amortizeBatchPrefillNs(total_ns, 1, request_count);
    const part2 = MockScheduler.amortizeBatchPrefillNs(total_ns, 2, request_count);

    try std.testing.expectEqual(@as(u64, 4), part0);
    try std.testing.expectEqual(@as(u64, 3), part1);
    try std.testing.expectEqual(@as(u64, 3), part2);
    try std.testing.expectEqual(total_ns, part0 + part1 + part2);
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
    _ = try scheduler.step();

    // Second step: decode
    const events = try scheduler.step();

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
    _ = try scheduler.step();

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
    _ = try scheduler.step();

    const request_entry = scheduler.requests.get(request_id).?;
    try std.testing.expectEqual(RequestState.completed, request_entry.state);
    try std.testing.expectEqual(@as(usize, 1), request_entry.generated_tokens.items.len);
}

test "Scheduler.step - enforces max completion tokens on prefill token" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const eos_token_ids = [_]u32{999}; // Keep EOS out of the way.
    const opts = Scheduler.SubmitOptions{
        .eos_token_ids = &eos_token_ids,
        .max_completion_tokens = 1,
    };
    const request_id = try scheduler.submit(&prompt, 10, opts);

    // First step runs prefill and emits one visible token. The completion
    // budget must be enforced immediately without entering decode.
    _ = try scheduler.step();

    const request_entry = scheduler.requests.get(request_id).?;
    try std.testing.expectEqual(RequestState.completed, request_entry.state);
    try std.testing.expectEqual(@as(usize, 1), request_entry.generated_tokens.items.len);
    try std.testing.expectEqual(@as(usize, 0), backend.decode_calls.items.len);
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
    _ = try scheduler.step();

    // Run decode step - should batch both
    _ = try scheduler.step();

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
    _ = try scheduler.step();

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
        fn cb(request_id: u64, token: u32, is_final: bool, _: bool, user_data: ?*anyopaque) void {
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

test "generateSync queued route - basic generation" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };

    var result = try scheduler.generateSync(&prompt, 5, null);
    defer result.deinit(alloc);

    // Should generate up to max_tokens
    try std.testing.expectEqual(@as(usize, 5), result.tokens.len);
    try std.testing.expectEqual(FinishReason.length, result.finish_reason);
}

test "generateSync queued route captures final logits when requested" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    var result = try scheduler.generateSync(&prompt, 3, .{
        .return_final_logits = true,
    });
    defer result.deinit(alloc);

    try std.testing.expectEqual(@as(usize, backend.vocab_size), result.final_logits.len);
    try std.testing.expect(result.tokens.len > 0);
}

test "Scheduler descriptor-backed state blocks are bound and validated" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 2);
    defer backend.deinit();
    backend.setStateDescriptors(&.{
        runtime_contract.defaultStateDescriptor(.shortconv),
    });

    var scheduler = try MockScheduler.init(alloc, &backend, .{
        .default_eos_token_ids = &[_]u32{100},
        .state_descriptors = backend.state_descriptors,
    });
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    _ = try scheduler.submit(&prompt, 1, null);

    // One step triggers activation, descriptor state allocation, bind, and unbind.
    _ = try scheduler.step();

    try std.testing.expect(backend.bind_slot_state_blocks_calls > 0);
    try std.testing.expect(backend.unbind_slot_state_blocks_calls > 0);
    try std.testing.expectEqual(@as(usize, 1), backend.last_bound_state_block_count);
    try std.testing.expect(backend.saw_state_blocks_valid);
}

test "Scheduler zero-inits shortconv descriptor state on alloc" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 2);
    defer backend.deinit();
    backend.setStateDescriptors(&.{
        runtime_contract.defaultStateDescriptor(.shortconv),
    });

    var scheduler = try MockScheduler.init(alloc, &backend, .{
        .state_descriptors = backend.state_descriptors,
    });
    defer scheduler.deinit();

    const prompt = [_]u32{ 7, 8, 9 };
    _ = try scheduler.submit(&prompt, 1, null);
    _ = try scheduler.step();

    try std.testing.expect(backend.saw_zero_init_shortconv_state);
}

test "Scheduler reuses slot-persistent state blocks across sequential requests on the same slot" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 1);
    defer backend.deinit();
    backend.setStateDescriptors(&.{
        runtime_contract.defaultStateDescriptor(.shortconv),
    });

    var scheduler = try MockScheduler.init(alloc, &backend, .{
        .state_descriptors = backend.state_descriptors,
    });
    defer scheduler.deinit();

    const prompt_a = [_]u32{ 1, 2, 3 };
    _ = try scheduler.submit(&prompt_a, 1, null);
    _ = try scheduler.step();

    const prompt_b = [_]u32{ 4, 5, 6 };
    _ = try scheduler.submit(&prompt_b, 1, null);
    _ = try scheduler.step();

    try std.testing.expectEqual(@as(usize, 1), scheduler.slot_state_blocks.count());
    try std.testing.expectEqual(@as(usize, 0), scheduler.request_state_blocks.count());
    try std.testing.expect(backend.first_bound_state_ptr != null);
    try std.testing.expect(backend.second_bound_state_ptr != null);
    try std.testing.expectEqual(backend.first_bound_state_ptr.?, backend.second_bound_state_ptr.?);
}

test "Scheduler frees request-scoped state blocks at request teardown" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 1);
    defer backend.deinit();
    backend.setStateDescriptors(&.{
        .{
            .id = 99,
            .size_bytes = 64,
            .align_bytes = 64,
            .zero_init = true,
            .lifecycle = .request_scoped,
        },
    });

    var scheduler = try MockScheduler.init(alloc, &backend, .{
        .state_descriptors = backend.state_descriptors,
    });
    defer scheduler.deinit();

    const prompt = [_]u32{ 7, 8, 9 };
    _ = try scheduler.submit(&prompt, 1, null);
    _ = try scheduler.step();

    try std.testing.expectEqual(@as(usize, 0), scheduler.slot_state_blocks.count());
    try std.testing.expectEqual(@as(usize, 0), scheduler.request_state_blocks.count());
}

test "Scheduler rejects descriptor alignment above 64" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 2);
    defer backend.deinit();
    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const invalid_desc = runtime_contract.StateDescriptor{
        .id = @intFromEnum(runtime_contract.StateBlockId.shortconv),
        .size_bytes = 64,
        .align_bytes = 128,
        .zero_init = true,
        .lifecycle = .slot_persistent,
    };
    try std.testing.expectError(error.InvalidStateDescriptorBinding, scheduler.allocateStateBlockStorage(invalid_desc));
}

test "Scheduler rejects descriptor size that cannot fit host usize" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 2);
    defer backend.deinit();
    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const invalid_desc = runtime_contract.StateDescriptor{
        .id = @intFromEnum(runtime_contract.StateBlockId.shortconv),
        .size_bytes = std.math.maxInt(u64),
        .align_bytes = 64,
        .zero_init = true,
        .lifecycle = .slot_persistent,
    };
    _ = scheduler.allocateStateBlockStorage(invalid_desc) catch |err| {
        try std.testing.expect(err == error.InvalidStateDescriptorBinding or err == error.OutOfMemory);
        return;
    };
    return error.TestUnexpectedResult;
}

test "generateSync queued route - EOS detection" {
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

test "generateSync queued route - stop sequences" {
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

test "generateSync queued route - callback invocation" {
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
        fn cb(_: u64, token: u32, _: bool, _: bool, user_data: ?*anyopaque) void {
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

test "generateSync queued route - used when other requests are active" {
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

    // Submit first request so a second request is processed while another is active.
    _ = try scheduler.submit(&prompt1, 5, null);

    // generateSync uses the same queued scheduler route in all cases.
    var result = try scheduler.generateSync(&prompt2, 5, null);
    defer result.deinit(alloc);

    // Should still work correctly
    try std.testing.expect(result.tokens.len > 0);
}

test "generateSync queued route - seed produces deterministic output" {
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

test "generateSync top-k candidate route - candidate order does not affect seeded output" {
    const alloc = std.testing.allocator;
    var backend = MockStreamingBackend.init(alloc, 256);
    backend.alternate_top_k_order = true;
    backend.flat_top_k_logits = true;
    defer backend.deinit();

    var scheduler = try MockStreamingScheduler.init(alloc, &backend, .{
        .default_sampling = .{
            .strategy = .top_k,
            .temperature = 1.0,
            .top_k = 2,
        },
    });
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    const sampling_cfg = sampling.SamplingConfig{
        .strategy = .top_k,
        .temperature = 1.0,
        .top_k = 2,
        .seed = 42,
    };

    var result1 = try scheduler.generateSync(&prompt, 2, .{
        .sampling = sampling_cfg,
        .eos_token_ids = &[_]u32{9999},
    });
    defer result1.deinit(alloc);

    var result2 = try scheduler.generateSync(&prompt, 2, .{
        .sampling = sampling_cfg,
        .eos_token_ids = &[_]u32{9999},
    });
    defer result2.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), result1.tokens.len);
    try std.testing.expectEqual(@as(usize, 2), result2.tokens.len);
    try std.testing.expectEqualSlices(u32, result1.tokens, result2.tokens);
    try std.testing.expectEqual(@as(usize, 2), backend.decode_top_k_calls);
}

test "generateSync top-k candidate route honors teacher forcing vocab ids" {
    const alloc = std.testing.allocator;
    var backend = MockStreamingBackend.init(alloc, 256);
    defer backend.deinit();

    var scheduler = try MockStreamingScheduler.init(alloc, &backend, .{
        .default_sampling = .{
            .strategy = .top_k,
            .temperature = 1.0,
            .top_k = 2,
        },
    });
    defer scheduler.deinit();

    const TeacherState = struct {
        tokens: [2]u32,
        index: usize = 0,
    };
    var teacher_state = TeacherState{
        .tokens = .{ @intCast(backend.greedy_token), 999 },
    };
    const Provider = struct {
        fn getNext(ctx: ?*anyopaque) ?u32 {
            const state: *TeacherState = @ptrCast(@alignCast(ctx.?));
            if (state.index >= state.tokens.len) return null;
            const token = state.tokens[state.index];
            state.index += 1;
            return token;
        }
    };

    xray.enableTeacherForcing(&Provider.getNext, &teacher_state);
    defer xray.disableTeacherForcing();

    var result = try scheduler.generateSync(&[_]u32{ 1, 2, 3 }, 2, .{
        .sampling = .{
            .strategy = .top_k,
            .temperature = 1.0,
            .top_k = 2,
        },
        .eos_token_ids = &[_]u32{9999},
    });
    defer result.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), result.tokens.len);
    try std.testing.expectEqual(@as(u32, @intCast(backend.greedy_token)), result.tokens[0]);
    try std.testing.expectEqual(@as(u32, 999), result.tokens[1]);
}

test "Scheduler mixed lifecycle: slot-persistent stable, request-scoped freed between requests" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 1);
    defer backend.deinit();
    // Mixed descriptor set: 1 slot_persistent + 1 request_scoped
    backend.setStateDescriptors(&.{
        runtime_contract.defaultStateDescriptor(.shortconv),
        .{
            .id = 99,
            .size_bytes = 64,
            .align_bytes = 64,
            .zero_init = true,
            .lifecycle = .request_scoped,
        },
    });

    var scheduler = try MockScheduler.init(alloc, &backend, .{
        .state_descriptors = backend.state_descriptors,
    });
    defer scheduler.deinit();

    // First request
    const prompt_a = [_]u32{ 1, 2, 3 };
    _ = try scheduler.submit(&prompt_a, 1, null);
    _ = try scheduler.step();

    // After first request completes: slot-persistent blocks remain, request-scoped freed
    try std.testing.expectEqual(@as(usize, 1), scheduler.slot_state_blocks.count());
    try std.testing.expectEqual(@as(usize, 0), scheduler.request_state_blocks.count());

    // Second request on same slot
    const prompt_b = [_]u32{ 4, 5, 6 };
    _ = try scheduler.submit(&prompt_b, 1, null);
    _ = try scheduler.step();

    // Slot-persistent ptr must be stable across requests
    try std.testing.expectEqual(@as(usize, 1), scheduler.slot_state_blocks.count());
    try std.testing.expectEqual(@as(usize, 0), scheduler.request_state_blocks.count());
    try std.testing.expect(backend.first_bound_state_ptr != null);
    try std.testing.expect(backend.second_bound_state_ptr != null);
    try std.testing.expectEqual(backend.first_bound_state_ptr.?, backend.second_bound_state_ptr.?);
}

test "scoreTeacherForcedNll scores all target tokens in one pass" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 1);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 11, 12, 13 };
    const targets = [_]u32{ 100, 100, 100, 100 };
    const scored = try scheduler.scoreTeacherForcedNll(prompt[0..], targets[0..]);

    try std.testing.expectEqual(@as(usize, targets.len), scored.scored_tokens);
    try std.testing.expect(std.math.isFinite(scored.nll_sum));
    try std.testing.expect(scored.nll_sum >= 0.0);
    try std.testing.expectEqual(@as(usize, 1), backend.prefill_calls.items.len);
    try std.testing.expectEqual(@as(usize, targets.len - 1), backend.decode_calls.items.len);
}

test "teacher-forced cursor reuses one slot and advances decode state" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 1);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    const prompt = [_]u32{ 21, 22, 23 };
    const targets = [_]u32{ 100, 100, 100 };

    var cursor = try scheduler.beginTeacherForced(prompt[0..]);
    defer scheduler.endTeacherForced(&cursor);

    const logits0 = try scheduler.teacherForcedCurrentLogits(&cursor);
    try std.testing.expect(logits0.len > 0);

    try scheduler.advanceTeacherForced(&cursor, targets[0]);
    const logits1 = try scheduler.teacherForcedCurrentLogits(&cursor);
    try std.testing.expectEqual(logits0.len, logits1.len);

    try scheduler.advanceTeacherForced(&cursor, targets[1]);
    const logits2 = try scheduler.teacherForcedCurrentLogits(&cursor);
    try std.testing.expectEqual(logits1.len, logits2.len);

    try std.testing.expect(cursor.prefill_ns > 0);
    try std.testing.expect(cursor.decode_ns > 0);
    try std.testing.expectEqual(@as(usize, 1), backend.prefill_calls.items.len);
    try std.testing.expectEqual(@as(usize, targets.len - 1), backend.decode_calls.items.len);
}

test "Scheduler step-scoped state blocks are zeroed at each step boundary" {
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 1);
    defer backend.deinit();
    // step_scoped descriptor only
    backend.setStateDescriptors(&.{
        .{
            .id = 50,
            .size_bytes = 64,
            .align_bytes = 64,
            .zero_init = true,
            .lifecycle = .step_scoped,
        },
    });

    var scheduler = try MockScheduler.init(alloc, &backend, .{
        .state_descriptors = backend.state_descriptors,
        .default_eos_token_ids = &[_]u32{},
    });
    defer scheduler.deinit();

    const prompt = [_]u32{ 1, 2, 3 };
    _ = try scheduler.submit(&prompt, 5, null);

    // First step: prefills + generates first token
    _ = try scheduler.step();

    // Verify step_scoped block exists in request_state_blocks
    try std.testing.expectEqual(@as(usize, 1), scheduler.request_state_blocks.count());

    // Write non-zero bytes into the step_scoped block
    var iter = scheduler.request_state_blocks.valueIterator();
    const blocks = iter.next().?;
    try std.testing.expectEqual(@as(usize, 1), blocks.storage.len);
    @memset(blocks.storage[0].bytes, 0xAA);

    // Second step: should zero the step_scoped block before decode
    _ = try scheduler.step();

    // Verify the block was zeroed by resetStepScopedBlocks
    var iter2 = scheduler.request_state_blocks.valueIterator();
    if (iter2.next()) |blocks2| {
        for (blocks2.storage[0].bytes) |byte| {
            try std.testing.expectEqual(@as(u8, 0), byte);
        }
    }
}

test "Scheduler.step - thinking disabled with tool-generation flags" {
    // Regression: tools + max_reasoning_tokens=0 triggered "double free or
    // corruption (out)". This test reproduces the exact submission flags the
    // router sends for that combination: thinking_budget=0,
    // thinking_end_tokens=&.{}, max_completion_tokens set.
    const alloc = std.testing.allocator;
    var backend = try MockBackend.init(alloc, 1000, 4);
    defer backend.deinit();

    var scheduler = try MockScheduler.init(alloc, &backend, .{});
    defer scheduler.deinit();

    // Simulate a prompt with tool definitions (longer than usual).
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const eos_ids = [_]u32{151645}; // Qwen3.5 EOS

    const request_id = try scheduler.submit(&prompt, 64, .{
        .eos_token_ids = &eos_ids,
        .max_thinking_tokens = 0, // thinking disabled
        .thinking_end_tokens = &.{}, // empty (no thinking)
        .max_completion_tokens = 64,
    });

    // Step 1: prefill
    _ = try scheduler.step();

    // Step 2+: decode several tokens
    var total_tokens: usize = 0;
    while (scheduler.hasActive()) {
        const events = try scheduler.step();

        for (events) |ev| {
            total_tokens += 1;
            if (ev.is_final) break;
        }
        if (total_tokens >= 10) break;
    }

    // Request should have generated tokens without crashing.
    const req = scheduler.requests.get(request_id).?;
    try std.testing.expect(req.generated_tokens.items.len > 0);

    // Clean up: remove completed request.
    scheduler.remove(request_id);
}
