//! Backend-neutral scheduler request, event, and route contracts.

const std = @import("std");
const validate = @import("validate_pkg");
const runtime_contract = @import("runtime_contract_pkg");
const sampling = @import("../sampling/contracts.zig");

/// Request state in the scheduler.
pub const RequestState = enum {
    /// Waiting for a slot to become available.
    queued,
    /// Slot allocated, waiting for prefill.
    pending_prefill,
    /// Prefill complete, actively generating.
    generating,
    /// Generation complete (hit EOS or max tokens).
    completed,
    /// Request was cancelled.
    cancelled,
    /// Request failed with an error.
    failed,
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

/// A generation request managed by the scheduler.
pub const Request = struct {
    /// Unique request ID.
    id: u64,
    /// Current state.
    state: RequestState,
    /// Assigned slot index (valid when state >= pending_prefill).
    slot_index: ?usize,
    /// Input prompt tokens.
    prompt_tokens: []const u32,
    /// Maximum tokens to generate.
    max_tokens: usize,
    /// Generated tokens so far.
    generated_tokens: std.ArrayList(u32),
    /// Current position (prompt_len + generated_count).
    token_position: usize,
    /// EOS token IDs (generation stops if any is produced).
    eos_token_ids: []const u32,
    /// Stop sequences (tokenized). Generation stops when any sequence matches.
    stop_sequences: []const []const u32,
    /// Optional callback for streaming tokens.
    /// Signature: fn(request_id: u64, token: u32, is_final: bool, in_thinking: bool, user_data: ?*anyopaque) void
    callback: ?*const fn (u64, u32, bool, bool, ?*anyopaque) void,
    callback_data: ?*anyopaque,
    /// Sampling configuration.
    sampling_config: sampling.SamplingConfig,
    /// Optional grammar sampler for constrained decoding.
    grammar_sampler: ?*validate.sampler.ConstrainedSampler = null,
    /// Optional backend-specific vision prefill payload.
    vision_input: ?*const anyopaque = null,
    /// Error message if state == failed.
    error_msg: ?[]const u8,
    /// Priority (higher = more urgent, for priority scheduling).
    priority: i32,
    /// Timestamp when request was submitted (for FIFO ordering).
    submit_time: i64,
    /// Finish reason for completed requests.
    finish_reason: FinishReason,
    /// Time spent in prefill for this request.
    prefill_ns: u64 = 0,
    /// Time spent in decode steps for this request.
    decode_ns: u64 = 0,
    /// Whether to capture final-step logits for this request.
    capture_final_logits: bool = false,
    /// Captured final-step logits (owned when non-empty).
    final_logits: []f32 = &.{},

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

    pub fn deinit(self: *Request, allocator: std.mem.Allocator) void {
        self.generated_tokens.deinit(allocator);
        if (self.final_logits.len > 0) {
            allocator.free(self.final_logits);
        }
        if (self.error_msg) |msg| {
            allocator.free(msg);
        }
    }
};

/// Token produced by a generation step.
pub const TokenEvent = struct {
    /// Request ID that produced this token.
    request_id: u64,
    /// The generated token.
    token: u32,
    /// Whether this is the final token (EOS or max reached).
    is_final: bool,
    /// Whether the token was generated during thinking/reasoning.
    in_thinking: bool,
    /// Slot index (for debugging).
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
    /// Maximum concurrent requests (limited by backend's max_batch_size).
    max_concurrent: ?usize = null,
    /// Default EOS token IDs if not specified per-request.
    default_eos_token_ids: []const u32 = &.{},
    /// Default sampling configuration.
    default_sampling: sampling.SamplingConfig = .{},
    /// Enable priority scheduling (otherwise FIFO).
    priority_scheduling: bool = false,
    /// Optional tokenizer for grammar-constrained sampling.
    tokenizer: ?TokenizerView = null,
    /// Plan-owned state descriptor contract for scheduler allocation.
    state_descriptors: []const runtime_contract.StateDescriptor = &.{},
};

/// Backend-planned route for single-request `generateSync` decode.
pub const SchedulerSingleDecodeRoute = enum {
    queued,
    greedy_streaming,
    top_k_streaming,
    top_k_candidate,
};

/// Semantic eligibility context for backend single-request decode planning.
pub const SchedulerSingleDecodeRoutePlan = struct {
    sampling_config: *const sampling.SamplingConfig,
    decode_batch_size: usize,
    has_callback: bool,
    capture_final_logits: bool,
    has_grammar_sampler: bool,
    prompt_token_count: usize,
    greedy_streaming_semantic_eligible: bool,
    top_k_streaming_semantic_eligible: bool,
    top_k_candidate_semantic_eligible: bool,
    greedy_streaming_backend_supported: bool,
    top_k_streaming_backend_supported: bool,
    top_k_candidate_backend_supported: bool,
};

/// Context for backend-owned batched top-k route selection in queued decode.
pub const SchedulerBatchedTopKRoutePlan = struct {
    decode_batch_size: usize,
    route_top_k: usize,
    sampling_config: *const sampling.SamplingConfig,
};

test "RequestState enum values distinct" {
    const states = [_]RequestState{
        .queued,
        .pending_prefill,
        .generating,
        .completed,
        .cancelled,
        .failed,
    };
    for (states, 0..) |lhs, lhs_idx| {
        for (states, 0..) |rhs, rhs_idx| {
            if (lhs_idx == rhs_idx) continue;
            try std.testing.expect(lhs != rhs);
        }
    }
}

test "FinishReason numeric contract stays stable" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(FinishReason.in_progress));
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(FinishReason.eos_token));
    try std.testing.expectEqual(@as(u8, 2), @intFromEnum(FinishReason.length));
    try std.testing.expectEqual(@as(u8, 3), @intFromEnum(FinishReason.stop_sequence));
    try std.testing.expectEqual(@as(u8, 4), @intFromEnum(FinishReason.cancelled));
}

test "TokenEvent struct fields preserve values" {
    const event = TokenEvent{
        .request_id = 42,
        .token = 123,
        .is_final = true,
        .in_thinking = false,
        .slot_index = 2,
        .timestamp_ns = 1000,
    };
    try std.testing.expectEqual(@as(u64, 42), event.request_id);
    try std.testing.expectEqual(@as(u32, 123), event.token);
    try std.testing.expect(event.is_final);
    try std.testing.expect(!event.in_thinking);
    try std.testing.expectEqual(@as(usize, 2), event.slot_index);
    try std.testing.expectEqual(@as(i128, 1000), event.timestamp_ns);
}

test "SchedulerConfig defaults keep neutral scheduler contract" {
    const config = SchedulerConfig{};
    try std.testing.expectEqual(@as(?usize, null), config.max_concurrent);
    try std.testing.expectEqual(@as(usize, 0), config.default_eos_token_ids.len);
    try std.testing.expectEqual(sampling.SamplingStrategy.greedy, config.default_sampling.strategy);
    try std.testing.expect(!config.priority_scheduling);
    try std.testing.expect(config.tokenizer == null);
    try std.testing.expectEqual(@as(usize, 0), config.state_descriptors.len);
}

test "TokenizerView.fromTokenizer forwards tokenizer methods" {
    const MockTokenizer = struct {
        vocab_size: usize = 5,

        pub fn tokenBytes(_: *const @This(), token_id: usize) ?[]const u8 {
            return switch (token_id) {
                0 => "zero",
                1 => "one",
                else => null,
            };
        }

        pub fn getVocabSize(self: *const @This()) usize {
            return self.vocab_size;
        }
    };

    const tokenizer = MockTokenizer{};
    const view = TokenizerView.fromTokenizer(&tokenizer);
    try std.testing.expectEqualStrings("one", view.tokenBytes(1).?);
    try std.testing.expect(view.tokenBytes(99) == null);
    try std.testing.expectEqual(@as(usize, 5), view.getVocabSize());
}

test "SchedulerSingleDecodeRoutePlan stores route eligibility" {
    const sampling_config = sampling.SamplingConfig{ .strategy = .top_k, .top_k = 4 };
    const plan = SchedulerSingleDecodeRoutePlan{
        .sampling_config = &sampling_config,
        .decode_batch_size = 1,
        .has_callback = true,
        .capture_final_logits = false,
        .has_grammar_sampler = false,
        .prompt_token_count = 8,
        .greedy_streaming_semantic_eligible = false,
        .top_k_streaming_semantic_eligible = true,
        .top_k_candidate_semantic_eligible = true,
        .greedy_streaming_backend_supported = false,
        .top_k_streaming_backend_supported = true,
        .top_k_candidate_backend_supported = true,
    };
    try std.testing.expectEqual(sampling.SamplingStrategy.top_k, plan.sampling_config.strategy);
    try std.testing.expect(plan.top_k_streaming_semantic_eligible);
    try std.testing.expect(plan.top_k_candidate_backend_supported);
}
