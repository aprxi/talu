//! Scheduler C API — Continuous Batching
//!
//! Exposes the GenericScheduler as an opaque handle for callers that need
//! dynamic request join/leave (server, eval harness). The caller drives
//! the step loop; the scheduler is single-threaded by design.
//!
//! Lifecycle: create → (submit | cancel | step)* → destroy
//! The backend (TaluInferenceBackend) must outlive the scheduler.
//!
//! Thread safety: NOT thread-safe. Caller must serialize all calls.

const std = @import("std");
const local_mod = @import("../router/local.zig");
const spec_mod = @import("../router/spec.zig");
const router_capi = @import("router.zig");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");

const allocator = std.heap.c_allocator;

const TaluInferenceBackend = router_capi.TaluInferenceBackend;
const BackendScheduler = local_mod.BackendScheduler;
const SchedulerConfig = local_mod.SchedulerConfig;
const SamplingConfig = local_mod.SamplingConfig;

// =============================================================================
// Opaque Handle
// =============================================================================

/// Opaque scheduler handle.
pub const TaluScheduler = opaque {};

// =============================================================================
// C-Compatible Types
// =============================================================================

/// Scheduler configuration. Pass null to talu_scheduler_create for defaults.
pub const CSchedulerConfig = extern struct {
    /// Maximum concurrent requests (0 = use backend's max_batch_size).
    max_concurrent: usize = 0,
};

/// Token event returned by talu_scheduler_step.
pub const CTokenEvent = extern struct {
    /// Request ID that produced this token.
    request_id: u64,
    /// The generated token ID.
    token: u32,
    /// 1 if this is the final token for this request, 0 otherwise.
    is_final: u8,
    _pad: [3]u8 = .{ 0, 0, 0 },
    /// Slot index (for diagnostics).
    slot_index: usize,
};

/// Per-request submit options. Pass null for defaults.
pub const CSubmitOptions = extern struct {
    /// EOS token IDs (null = use scheduler default from engine).
    /// Copied internally; caller memory need not remain valid.
    eos_token_ids: ?[*]const u32 = null,
    eos_token_ids_len: usize = 0,
    /// Sampling temperature (-1.0 = use scheduler default).
    /// When > 0 without explicit top_k or top_p, defaults to top-k strategy.
    temperature: f32 = -1.0,
    top_k: usize = 0,
    top_p: f32 = -1.0,
    min_p: f32 = -1.0,
    /// Random seed (0 = use scheduler default).
    seed: u64 = 0,
    /// Priority (higher = more urgent, 0 = default FIFO).
    priority: i32 = 0,
    _pad: [4]u8 = .{ 0, 0, 0, 0 },
};

// =============================================================================
// Internal Handle
// =============================================================================

/// Owned copies of borrowed data for a single request.
const OwnedRequestData = struct {
    prompt_tokens: []u32,
    eos_token_ids: ?[]u32 = null,

    fn deinit(self: *OwnedRequestData) void {
        allocator.free(self.prompt_tokens);
        if (self.eos_token_ids) |eos| allocator.free(eos);
    }
};

const SchedulerWrapper = struct {
    scheduler: BackendScheduler,
    /// Owned copies of borrowed data per request.
    /// The scheduler borrows prompt_tokens and eos_token_ids — we own the backing memory.
    owned_data: std.AutoHashMap(u64, OwnedRequestData),

    fn deinit(self: *SchedulerWrapper) void {
        var iter = self.owned_data.valueIterator();
        while (iter.next()) |data_ptr| {
            data_ptr.deinit();
        }
        self.owned_data.deinit();
        self.scheduler.deinit();
    }

    /// Remove a completed/cancelled request and free its owned data.
    fn removeRequest(self: *SchedulerWrapper, request_id: u64) void {
        self.scheduler.remove(request_id);
        if (self.owned_data.fetchRemove(request_id)) |kv| {
            var data = kv.value;
            data.deinit();
        }
    }

    /// Drain the scheduler's completed_queue to prevent unbounded growth.
    fn drainCompleted(self: *SchedulerWrapper) void {
        const completed = self.scheduler.popCompleted();
        if (completed.len > 0) allocator.free(completed);
    }
};

// =============================================================================
// Lifecycle
// =============================================================================

/// Create a scheduler bound to a local inference backend.
///
/// The backend must remain valid until talu_scheduler_destroy is called.
/// Returns null on error (check talu_last_error).
pub export fn talu_scheduler_create(
    backend: ?*TaluInferenceBackend,
    config: ?*const CSchedulerConfig,
) callconv(.c) ?*TaluScheduler {
    capi_error.clearError();

    const backend_ptr: *spec_mod.InferenceBackend = @ptrCast(@alignCast(backend orelse {
        capi_error.setErrorWithCode(.invalid_argument, "backend is null", .{});
        return null;
    }));

    const engine = backend_ptr.getLocalEngine() orelse {
        capi_error.setErrorWithCode(.invalid_argument, "backend is not a local engine", .{});
        return null;
    };

    var sched_config = SchedulerConfig{};
    if (config) |cfg| {
        if (cfg.max_concurrent > 0) {
            sched_config.max_concurrent = cfg.max_concurrent;
        }
    }

    var scheduler = engine.createScheduler(sched_config) catch |err| {
        capi_error.setError(err, "scheduler creation failed", .{});
        return null;
    };

    const wrapper = allocator.create(SchedulerWrapper) catch {
        scheduler.deinit();
        capi_error.setErrorWithCode(.out_of_memory, "scheduler allocation failed", .{});
        return null;
    };
    wrapper.* = .{
        .scheduler = scheduler,
        .owned_data = std.AutoHashMap(u64, OwnedRequestData).init(allocator),
    };
    return @ptrCast(wrapper);
}

/// Destroy a scheduler and free all resources.
///
/// All pending/active requests are cancelled. Passing null is a safe no-op.
pub export fn talu_scheduler_destroy(handle: ?*TaluScheduler) callconv(.c) void {
    const wrapper: *SchedulerWrapper = @ptrCast(@alignCast(handle orelse return));
    wrapper.deinit();
    allocator.destroy(wrapper);
}

// =============================================================================
// Request Management
// =============================================================================

/// Submit a generation request.
///
/// Both the token buffer and any eos_token_ids in options are copied internally;
/// caller memory need not remain valid after this call.
/// Returns a non-zero request ID on success, 0 on error.
pub export fn talu_scheduler_submit(
    handle: ?*TaluScheduler,
    tokens: ?[*]const u32,
    num_tokens: usize,
    max_tokens: usize,
    options: ?*const CSubmitOptions,
) callconv(.c) u64 {
    capi_error.clearError();

    const wrapper: *SchedulerWrapper = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "scheduler handle is null", .{});
        return 0;
    }));

    const src_tokens = if (tokens) |t| t[0..num_tokens] else {
        capi_error.setErrorWithCode(.invalid_argument, "tokens is null", .{});
        return 0;
    };

    // Copy prompt tokens — the scheduler borrows them.
    const token_copy = allocator.alloc(u32, num_tokens) catch {
        capi_error.setErrorWithCode(.out_of_memory, "token copy failed", .{});
        return 0;
    };
    @memcpy(token_copy, src_tokens);

    // Build submit options, copying any borrowed data from the caller.
    var submit_opts = BackendScheduler.SubmitOptions{};
    var eos_copy: ?[]u32 = null;

    if (options) |opts| {
        // Copy EOS token IDs — the scheduler stores them on the Request.
        // Non-null pointer with len==0 means "no EOS stopping" (explicit empty list).
        if (opts.eos_token_ids) |eos_ptr| {
            if (opts.eos_token_ids_len > 0) {
                const eos = allocator.alloc(u32, opts.eos_token_ids_len) catch {
                    allocator.free(token_copy);
                    capi_error.setErrorWithCode(.out_of_memory, "eos copy failed", .{});
                    return 0;
                };
                @memcpy(eos, eos_ptr[0..opts.eos_token_ids_len]);
                submit_opts.eos_token_ids = eos;
                eos_copy = eos;
            } else {
                submit_opts.eos_token_ids = &.{};
            }
        }
        submit_opts.priority = opts.priority;

        // Overlay sampling fields onto the scheduler's defaults.
        // Only fields explicitly set by the caller are overridden.
        // Strategy resolution chooses the primary route:
        // - explicit top_k => top_k route (top_p/min_p still compose in sampler)
        // - explicit top_p without top_k => top_p route
        // - temperature > 0 without explicit top_k/top_p => top_k route
        var sampling_cfg = wrapper.scheduler.config.default_sampling;
        var has_override = false;
        var explicit_top_k = false;
        var explicit_top_p = false;
        if (opts.temperature >= 0.0) {
            sampling_cfg.temperature = opts.temperature;
            has_override = true;
        }
        if (opts.top_k > 0) {
            sampling_cfg.top_k = opts.top_k;
            explicit_top_k = true;
            has_override = true;
        }
        if (opts.top_p >= 0.0) {
            sampling_cfg.top_p = opts.top_p;
            explicit_top_p = true;
            has_override = true;
        }
        if (opts.min_p >= 0.0) {
            sampling_cfg.min_p = opts.min_p;
            has_override = true;
        }
        if (opts.seed != 0) {
            sampling_cfg.seed = opts.seed;
            has_override = true;
        }
        // Resolve strategy for route selection while preserving all
        // filter fields in sampling_cfg for the sampler stage.
        if (explicit_top_k) {
            sampling_cfg.strategy = .top_k;
        } else if (explicit_top_p) {
            sampling_cfg.strategy = .top_p;
        } else if (opts.temperature > 0.0) {
            sampling_cfg.strategy = .top_k;
        }
        if (has_override) submit_opts.sampling = sampling_cfg;
    }

    const request_id = wrapper.scheduler.submit(token_copy, max_tokens, submit_opts) catch |err| {
        allocator.free(token_copy);
        if (eos_copy) |eos| allocator.free(eos);
        capi_error.setError(err, "submit failed", .{});
        return 0;
    };

    wrapper.owned_data.put(request_id, .{
        .prompt_tokens = token_copy,
        .eos_token_ids = eos_copy,
    }) catch {
        // Can't track ownership. Roll back: cancel the request so the
        // scheduler releases its borrows, then free the copies ourselves.
        _ = wrapper.scheduler.cancel(request_id);
        wrapper.scheduler.remove(request_id);
        wrapper.drainCompleted();
        allocator.free(token_copy);
        if (eos_copy) |eos| allocator.free(eos);
        capi_error.setErrorWithCode(.out_of_memory, "request tracking failed", .{});
        return 0;
    };

    return request_id;
}

/// Cancel a request.
///
/// Frees the slot immediately if the request was generating.
/// Returns 1 if cancelled, 0 if not found or already finished.
pub export fn talu_scheduler_cancel(
    handle: ?*TaluScheduler,
    request_id: u64,
) callconv(.c) u8 {
    capi_error.clearError();
    const wrapper: *SchedulerWrapper = @ptrCast(@alignCast(handle orelse return 0));
    const cancelled = wrapper.scheduler.cancel(request_id);
    if (cancelled) {
        wrapper.removeRequest(request_id);
        wrapper.drainCompleted();
    }
    return @intFromBool(cancelled);
}

// =============================================================================
// Step Loop
// =============================================================================

/// Run one generation step for all active requests.
///
/// Prefills pending requests, runs batched decode, samples tokens, and writes
/// events to the caller's buffer. Completed requests are auto-removed.
///
/// Returns the total number of events produced. If this exceeds max_events,
/// only max_events entries were written to events_out; the caller should
/// size the buffer to at least max_batch_size (default 8) to avoid loss.
pub export fn talu_scheduler_step(
    handle: ?*TaluScheduler,
    events_out: ?*CTokenEvent,
    max_events: usize,
) callconv(.c) usize {
    capi_error.clearError();

    const wrapper: *SchedulerWrapper = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "scheduler handle is null", .{});
        return 0;
    }));

    const events = wrapper.scheduler.step() catch |err| {
        capi_error.setError(err, "step failed", .{});
        return 0;
    };
    defer allocator.free(events);

    // Cleanup runs regardless of whether events are delivered.
    defer {
        for (events) |ev| {
            if (ev.is_final) {
                wrapper.removeRequest(ev.request_id);
            }
        }
        wrapper.drainCompleted();
    }

    // Require a valid output buffer.
    const out_ptr = events_out orelse {
        if (events.len > 0) {
            capi_error.setErrorWithCode(.invalid_argument, "events_out is null", .{});
        }
        return 0;
    };

    const out_buf: [*]CTokenEvent = @ptrCast(out_ptr);
    const copy_count = @min(events.len, max_events);
    for (events[0..copy_count], 0..) |ev, i| {
        out_buf[i] = .{
            .request_id = ev.request_id,
            .token = ev.token,
            .is_final = @intFromBool(ev.is_final),
            .slot_index = ev.slot_index,
        };
    }

    return events.len;
}

// =============================================================================
// Query
// =============================================================================

/// Check if there are any active or pending requests.
///
/// Returns 1 if the scheduler has work to do, 0 otherwise.
pub export fn talu_scheduler_has_active(
    handle: ?*const TaluScheduler,
) callconv(.c) u8 {
    capi_error.clearError();
    const wrapper: *const SchedulerWrapper = @ptrCast(@alignCast(handle orelse return 0));
    return @intFromBool(wrapper.scheduler.hasActive());
}

/// Get the number of currently active (generating) requests.
pub export fn talu_scheduler_active_count(
    handle: ?*const TaluScheduler,
) callconv(.c) usize {
    capi_error.clearError();
    const wrapper: *const SchedulerWrapper = @ptrCast(@alignCast(handle orelse return 0));
    return wrapper.scheduler.activeCount();
}

// =============================================================================
// NLL / PPL Scoring
// =============================================================================

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

fn klDivergenceFromLogits(reference_logits: []const f32, model_logits: []const f32) !f64 {
    var scratch_storage: [256]f64 = undefined;
    if (reference_logits.len <= scratch_storage.len) {
        return klDivergenceFromLogitsWithScratch(reference_logits, model_logits, scratch_storage[0..reference_logits.len]);
    }
    const scratch = try allocator.alloc(f64, reference_logits.len);
    defer allocator.free(scratch);
    return klDivergenceFromLogitsWithScratch(reference_logits, model_logits, scratch);
}

fn klDivergenceFromLogitsWithScratch(
    reference_logits: []const f32,
    model_logits: []const f32,
    ref_exp_scratch: []f64,
) !f64 {
    if (reference_logits.len == 0 or model_logits.len == 0) return error.InvalidArgument;
    if (reference_logits.len != model_logits.len) return error.InvalidArgument;
    if (ref_exp_scratch.len < reference_logits.len) return error.InvalidArgument;

    var max_ref: f32 = -std.math.inf(f32);
    var max_model: f32 = -std.math.inf(f32);
    for (reference_logits, model_logits) |ref_v, model_v| {
        if (!std.math.isFinite(ref_v) or !std.math.isFinite(model_v)) return error.InvalidArgument;
        max_ref = @max(max_ref, ref_v);
        max_model = @max(max_model, model_v);
    }

    var sum_exp_ref: f64 = 0.0;
    var sum_exp_model: f64 = 0.0;
    for (reference_logits, model_logits, 0..) |ref_v, model_v, idx| {
        const ref_exp = std.math.exp(@as(f64, ref_v - max_ref));
        ref_exp_scratch[idx] = ref_exp;
        sum_exp_ref += ref_exp;
        sum_exp_model += std.math.exp(@as(f64, model_v - max_model));
    }
    if (!(sum_exp_ref > 0.0) or !(sum_exp_model > 0.0)) return error.InvalidArgument;

    const ref_log_denom = @as(f64, max_ref) + std.math.log(f64, std.math.e, sum_exp_ref);
    const model_log_denom = @as(f64, max_model) + std.math.log(f64, std.math.e, sum_exp_model);

    var kld: f64 = 0.0;
    const inv_sum_ref = 1.0 / sum_exp_ref;
    for (reference_logits, model_logits, 0..) |ref_v, model_v, idx| {
        const log_p_ref = @as(f64, ref_v) - ref_log_denom;
        const log_p_model = @as(f64, model_v) - model_log_denom;
        const p_ref = ref_exp_scratch[idx] * inv_sum_ref;
        kld += p_ref * (log_p_ref - log_p_model);
    }

    if (!std.math.isFinite(kld) or kld < 0.0) return error.InvalidArgument;
    return kld;
}

fn canUseFastTeacherForcedScoring(context_len: usize, target_len: usize, max_context: usize) bool {
    if (target_len == 0) return false;
    if (max_context == 0) return true;
    if (context_len == 0) return false;
    const needed_history = std.math.add(usize, context_len, target_len - 1) catch return false;
    return needed_history <= max_context;
}

fn runDirectNllFast(
    engine: *local_mod.LocalEngine,
    context_tokens: []const u32,
    target_tokens: []const u32,
    out_nll: *f64,
    out_scored_tokens: *usize,
) !void {
    var scheduler = try engine.createScheduler(SchedulerConfig{});
    defer scheduler.deinit();

    const scored = try scheduler.scoreTeacherForcedNll(context_tokens, target_tokens);
    out_nll.* = scored.nll_sum;
    out_scored_tokens.* = scored.scored_tokens;
}

fn runDirectKldFast(
    reference_engine: *local_mod.LocalEngine,
    model_engine: *local_mod.LocalEngine,
    context_tokens: []const u32,
    target_tokens: []const u32,
    out_kld: *f64,
    out_scored_tokens: *usize,
) !void {
    var reference_scheduler = try reference_engine.createScheduler(SchedulerConfig{});
    defer reference_scheduler.deinit();
    var model_scheduler = try model_engine.createScheduler(SchedulerConfig{});
    defer model_scheduler.deinit();

    const reference_vocab = reference_scheduler.backend.vocabSize();
    const model_vocab = model_scheduler.backend.vocabSize();
    if (reference_vocab != model_vocab) return error.InvalidArgument;
    const ref_exp_scratch = try reference_engine.allocator.alloc(f64, reference_vocab);
    defer reference_engine.allocator.free(ref_exp_scratch);

    var reference_cursor = try reference_scheduler.beginTeacherForced(context_tokens);
    defer reference_scheduler.endTeacherForced(&reference_cursor);
    var model_cursor = try model_scheduler.beginTeacherForced(context_tokens);
    defer model_scheduler.endTeacherForced(&model_cursor);

    var kld_sum: f64 = 0.0;
    for (target_tokens, 0..) |target, idx| {
        const reference_logits = try reference_scheduler.teacherForcedCurrentLogits(&reference_cursor);
        const model_logits = try model_scheduler.teacherForcedCurrentLogits(&model_cursor);
        kld_sum += try klDivergenceFromLogitsWithScratch(reference_logits, model_logits, ref_exp_scratch);

        if (idx + 1 < target_tokens.len) {
            try reference_scheduler.advanceTeacherForced(&reference_cursor, target);
            try model_scheduler.advanceTeacherForced(&model_cursor, target);
        }
    }

    out_kld.* = kld_sum;
    out_scored_tokens.* = target_tokens.len;
}

fn runDirectJointFast(
    reference_engine: *local_mod.LocalEngine,
    model_engine: *local_mod.LocalEngine,
    context_tokens: []const u32,
    target_tokens: []const u32,
    out_reference_nll: *f64,
    out_model_nll: *f64,
    out_kld: *f64,
    out_scored_tokens: *usize,
) !void {
    var reference_scheduler = try reference_engine.createScheduler(SchedulerConfig{});
    defer reference_scheduler.deinit();
    var model_scheduler = try model_engine.createScheduler(SchedulerConfig{});
    defer model_scheduler.deinit();

    const reference_vocab = reference_scheduler.backend.vocabSize();
    const model_vocab = model_scheduler.backend.vocabSize();
    if (reference_vocab != model_vocab) return error.InvalidArgument;
    const ref_exp_scratch = try reference_engine.allocator.alloc(f64, reference_vocab);
    defer reference_engine.allocator.free(ref_exp_scratch);

    var reference_cursor = try reference_scheduler.beginTeacherForced(context_tokens);
    defer reference_scheduler.endTeacherForced(&reference_cursor);
    var model_cursor = try model_scheduler.beginTeacherForced(context_tokens);
    defer model_scheduler.endTeacherForced(&model_cursor);

    var reference_nll_sum: f64 = 0.0;
    var model_nll_sum: f64 = 0.0;
    var kld_sum: f64 = 0.0;
    for (target_tokens, 0..) |target, idx| {
        const reference_logits = try reference_scheduler.teacherForcedCurrentLogits(&reference_cursor);
        const model_logits = try model_scheduler.teacherForcedCurrentLogits(&model_cursor);
        reference_nll_sum += try negLogProbFromLogits(reference_logits, target);
        model_nll_sum += try negLogProbFromLogits(model_logits, target);
        kld_sum += try klDivergenceFromLogitsWithScratch(reference_logits, model_logits, ref_exp_scratch);

        if (idx + 1 < target_tokens.len) {
            try reference_scheduler.advanceTeacherForced(&reference_cursor, target);
            try model_scheduler.advanceTeacherForced(&model_cursor, target);
        }
    }

    out_reference_nll.* = reference_nll_sum;
    out_model_nll.* = model_nll_sum;
    out_kld.* = kld_sum;
    out_scored_tokens.* = target_tokens.len;
}

/// Score an autoregressive token stream and return total negative log-likelihood.
///
/// This computes:
///   nll = -Σ log p(target_t | context + target_<t)
///
/// `context_tokens` are the initial prompt tokens (may be empty).
/// `target_tokens` are the tokens to score.
/// `max_context` limits context length per step (0 = no truncation).
///
/// Returns 0 on success and writes:
///   - `out_nll`: summed NLL across all scored tokens
///   - `out_scored_tokens`: number of scored tokens
pub export fn talu_scheduler_score_tokens_nll(
    backend: ?*TaluInferenceBackend,
    context_tokens: ?[*]const u32,
    context_len: usize,
    target_tokens: ?[*]const u32,
    target_len: usize,
    max_context: usize,
    out_nll: ?*f64,
    out_scored_tokens: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();

    const out_nll_ptr = out_nll orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_nll is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const out_count_ptr = out_scored_tokens orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_scored_tokens is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out_nll_ptr.* = 0.0;
    out_count_ptr.* = 0;

    const backend_ptr: *spec_mod.InferenceBackend = @ptrCast(@alignCast(backend orelse {
        capi_error.setErrorWithCode(.invalid_argument, "backend is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));
    const engine = backend_ptr.getLocalEngine() orelse {
        capi_error.setErrorWithCode(.invalid_argument, "backend is not a local engine", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (target_len == 0) return 0;
    const ctx_tokens = if (context_tokens) |ptr| ptr[0..context_len] else blk: {
        if (context_len != 0) {
            capi_error.setErrorWithCode(.invalid_argument, "context_tokens is null but context_len > 0", .{});
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        }
        break :blk &.{};
    };
    const tgt_tokens = if (target_tokens) |ptr| ptr[0..target_len] else {
        capi_error.setErrorWithCode(.invalid_argument, "target_tokens is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (canUseFastTeacherForcedScoring(ctx_tokens.len, tgt_tokens.len, max_context)) {
        runDirectNllFast(engine, ctx_tokens, tgt_tokens, out_nll_ptr, out_count_ptr) catch |err| {
            std.log.warn("fast NLL scoring unavailable ({s}); falling back to slow path", .{@errorName(err)});
            out_nll_ptr.* = 0.0;
            out_count_ptr.* = 0;
        };
        if (out_count_ptr.* == tgt_tokens.len) return 0;
    }

    var scheduler = engine.createScheduler(SchedulerConfig{}) catch |err| {
        capi_error.setError(err, "failed to create scheduler for scoring", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer scheduler.deinit();

    var token_history = std.ArrayList(u32){};
    defer token_history.deinit(allocator);
    token_history.ensureTotalCapacity(allocator, ctx_tokens.len + tgt_tokens.len) catch |err| {
        capi_error.setError(err, "failed to allocate scoring history", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    token_history.appendSlice(allocator, ctx_tokens) catch |err| {
        capi_error.setError(err, "failed to initialize scoring context", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    var nll_sum: f64 = 0.0;
    var scored: usize = 0;

    for (tgt_tokens) |target| {
        const history_start = if (max_context > 0 and token_history.items.len > max_context)
            token_history.items.len - max_context
        else
            0;
        const prompt = token_history.items[history_start..];
        if (prompt.len == 0) {
            capi_error.setErrorWithCode(.invalid_argument, "scoring prompt is empty; provide non-empty context", .{});
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        }

        var result = scheduler.generateSync(prompt, 1, .{
            .return_final_logits = true,
            .eos_token_ids = &.{},
        }) catch |err| {
            capi_error.setError(err, "scheduler scoring step failed", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
        defer result.deinit(allocator);

        if (result.final_logits.len == 0) {
            capi_error.setErrorWithCode(.internal_error, "scheduler scoring returned no logits", .{});
            return @intFromEnum(error_codes.ErrorCode.internal_error);
        }

        const token_nll = negLogProbFromLogits(result.final_logits, target) catch {
            capi_error.setErrorWithCode(
                .invalid_argument,
                "target token {d} is invalid for vocab size {d}",
                .{ target, result.final_logits.len },
            );
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        };

        nll_sum += token_nll;
        scored += 1;

        token_history.append(allocator, target) catch |err| {
            capi_error.setError(err, "failed to extend scoring context", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
    }

    out_nll_ptr.* = nll_sum;
    out_count_ptr.* = scored;
    return 0;
}

/// Score KL divergence between reference and model next-token distributions.
///
/// This computes:
///   kld = Σ KL(P_ref(·|history_t) || P_model(·|history_t))
///
/// History evolves with `context_tokens` + teacher-forced `target_tokens`.
/// `max_context` limits context length per step (0 = no truncation).
///
/// Returns 0 on success and writes:
///   - `out_kld`: summed KL divergence across all scored tokens
///   - `out_scored_tokens`: number of scored tokens
pub export fn talu_scheduler_score_tokens_kld(
    reference_backend: ?*TaluInferenceBackend,
    model_backend: ?*TaluInferenceBackend,
    context_tokens: ?[*]const u32,
    context_len: usize,
    target_tokens: ?[*]const u32,
    target_len: usize,
    max_context: usize,
    out_kld: ?*f64,
    out_scored_tokens: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();

    const out_kld_ptr = out_kld orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_kld is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const out_count_ptr = out_scored_tokens orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_scored_tokens is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out_kld_ptr.* = 0.0;
    out_count_ptr.* = 0;

    const reference_backend_ptr: *spec_mod.InferenceBackend = @ptrCast(@alignCast(reference_backend orelse {
        capi_error.setErrorWithCode(.invalid_argument, "reference_backend is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));
    const model_backend_ptr: *spec_mod.InferenceBackend = @ptrCast(@alignCast(model_backend orelse {
        capi_error.setErrorWithCode(.invalid_argument, "model_backend is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const reference_engine = reference_backend_ptr.getLocalEngine() orelse {
        capi_error.setErrorWithCode(.invalid_argument, "reference_backend is not a local engine", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const model_engine = model_backend_ptr.getLocalEngine() orelse {
        capi_error.setErrorWithCode(.invalid_argument, "model_backend is not a local engine", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (target_len == 0) return 0;
    const ctx_tokens = if (context_tokens) |ptr| ptr[0..context_len] else blk: {
        if (context_len != 0) {
            capi_error.setErrorWithCode(.invalid_argument, "context_tokens is null but context_len > 0", .{});
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        }
        break :blk &.{};
    };
    const tgt_tokens = if (target_tokens) |ptr| ptr[0..target_len] else {
        capi_error.setErrorWithCode(.invalid_argument, "target_tokens is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (canUseFastTeacherForcedScoring(ctx_tokens.len, tgt_tokens.len, max_context)) {
        runDirectKldFast(reference_engine, model_engine, ctx_tokens, tgt_tokens, out_kld_ptr, out_count_ptr) catch |err| {
            std.log.warn("fast KLD scoring unavailable ({s}); falling back to slow path", .{@errorName(err)});
            out_kld_ptr.* = 0.0;
            out_count_ptr.* = 0;
        };
        if (out_count_ptr.* == tgt_tokens.len) return 0;
    }

    var reference_scheduler = reference_engine.createScheduler(SchedulerConfig{}) catch |err| {
        capi_error.setError(err, "failed to create reference scheduler for KLD scoring", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer reference_scheduler.deinit();

    var model_scheduler = model_engine.createScheduler(SchedulerConfig{}) catch |err| {
        capi_error.setError(err, "failed to create model scheduler for KLD scoring", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer model_scheduler.deinit();

    const reference_vocab = reference_scheduler.backend.vocabSize();
    const model_vocab = model_scheduler.backend.vocabSize();
    if (reference_vocab != model_vocab) {
        capi_error.setErrorWithCode(
            .invalid_argument,
            "reference/model vocab mismatch for KLD scoring ({d} vs {d})",
            .{ reference_vocab, model_vocab },
        );
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }
    const ref_exp_scratch = reference_engine.allocator.alloc(f64, reference_vocab) catch |err| {
        capi_error.setError(err, "failed to allocate KLD scratch buffer", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer reference_engine.allocator.free(ref_exp_scratch);

    var token_history = std.ArrayList(u32){};
    defer token_history.deinit(allocator);
    token_history.ensureTotalCapacity(allocator, ctx_tokens.len + tgt_tokens.len) catch |err| {
        capi_error.setError(err, "failed to allocate KLD scoring history", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    token_history.appendSlice(allocator, ctx_tokens) catch |err| {
        capi_error.setError(err, "failed to initialize KLD scoring context", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    var kld_sum: f64 = 0.0;
    var scored: usize = 0;

    for (tgt_tokens) |target| {
        const history_start = if (max_context > 0 and token_history.items.len > max_context)
            token_history.items.len - max_context
        else
            0;
        const prompt = token_history.items[history_start..];
        if (prompt.len == 0) {
            capi_error.setErrorWithCode(.invalid_argument, "KLD scoring prompt is empty; provide non-empty context", .{});
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        }

        var reference_result = reference_scheduler.generateSync(prompt, 1, .{
            .return_final_logits = true,
            .eos_token_ids = &.{},
        }) catch |err| {
            capi_error.setError(err, "reference scheduler KLD step failed", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
        defer reference_result.deinit(allocator);

        var model_result = model_scheduler.generateSync(prompt, 1, .{
            .return_final_logits = true,
            .eos_token_ids = &.{},
        }) catch |err| {
            capi_error.setError(err, "model scheduler KLD step failed", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
        defer model_result.deinit(allocator);

        if (reference_result.final_logits.len == 0 or model_result.final_logits.len == 0) {
            capi_error.setErrorWithCode(.internal_error, "KLD scoring returned no logits", .{});
            return @intFromEnum(error_codes.ErrorCode.internal_error);
        }
        if (reference_result.final_logits.len != model_result.final_logits.len) {
            capi_error.setErrorWithCode(
                .invalid_argument,
                "reference/model vocab mismatch for KLD scoring ({d} vs {d})",
                .{ reference_result.final_logits.len, model_result.final_logits.len },
            );
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        }

        const token_kld = klDivergenceFromLogitsWithScratch(reference_result.final_logits, model_result.final_logits, ref_exp_scratch) catch {
            capi_error.setErrorWithCode(
                .invalid_argument,
                "invalid logits while computing KLD (vocab={d})",
                .{reference_result.final_logits.len},
            );
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        };

        kld_sum += token_kld;
        scored += 1;

        token_history.append(allocator, target) catch |err| {
            capi_error.setError(err, "failed to extend KLD scoring context", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
    }

    out_kld_ptr.* = kld_sum;
    out_count_ptr.* = scored;
    return 0;
}

/// Joint scoring for model vs reference:
/// - reference NLL
/// - model NLL
/// - KL(reference || model)
///
/// This is mathematically equivalent to running NLL(model), NLL(reference),
/// and KLD separately, but performs a single lockstep teacher-forced pass.
pub export fn talu_scheduler_score_tokens_joint(
    reference_backend: ?*TaluInferenceBackend,
    model_backend: ?*TaluInferenceBackend,
    context_tokens: ?[*]const u32,
    context_len: usize,
    target_tokens: ?[*]const u32,
    target_len: usize,
    max_context: usize,
    out_reference_nll: ?*f64,
    out_model_nll: ?*f64,
    out_kld: ?*f64,
    out_scored_tokens: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();

    const out_reference_nll_ptr = out_reference_nll orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_reference_nll is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const out_model_nll_ptr = out_model_nll orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_model_nll is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const out_kld_ptr = out_kld orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_kld is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const out_count_ptr = out_scored_tokens orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_scored_tokens is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out_reference_nll_ptr.* = 0.0;
    out_model_nll_ptr.* = 0.0;
    out_kld_ptr.* = 0.0;
    out_count_ptr.* = 0;

    const reference_backend_ptr: *spec_mod.InferenceBackend = @ptrCast(@alignCast(reference_backend orelse {
        capi_error.setErrorWithCode(.invalid_argument, "reference_backend is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));
    const model_backend_ptr: *spec_mod.InferenceBackend = @ptrCast(@alignCast(model_backend orelse {
        capi_error.setErrorWithCode(.invalid_argument, "model_backend is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const reference_engine = reference_backend_ptr.getLocalEngine() orelse {
        capi_error.setErrorWithCode(.invalid_argument, "reference_backend is not a local engine", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const model_engine = model_backend_ptr.getLocalEngine() orelse {
        capi_error.setErrorWithCode(.invalid_argument, "model_backend is not a local engine", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (target_len == 0) return 0;
    const ctx_tokens = if (context_tokens) |ptr| ptr[0..context_len] else blk: {
        if (context_len != 0) {
            capi_error.setErrorWithCode(.invalid_argument, "context_tokens is null but context_len > 0", .{});
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        }
        break :blk &.{};
    };
    const tgt_tokens = if (target_tokens) |ptr| ptr[0..target_len] else {
        capi_error.setErrorWithCode(.invalid_argument, "target_tokens is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (canUseFastTeacherForcedScoring(ctx_tokens.len, tgt_tokens.len, max_context)) {
        runDirectJointFast(
            reference_engine,
            model_engine,
            ctx_tokens,
            tgt_tokens,
            out_reference_nll_ptr,
            out_model_nll_ptr,
            out_kld_ptr,
            out_count_ptr,
        ) catch |err| {
            std.log.warn("fast joint scoring unavailable ({s}); falling back to slow path", .{@errorName(err)});
            out_reference_nll_ptr.* = 0.0;
            out_model_nll_ptr.* = 0.0;
            out_kld_ptr.* = 0.0;
            out_count_ptr.* = 0;
        };
        if (out_count_ptr.* == tgt_tokens.len) return 0;
    }

    var reference_scheduler = reference_engine.createScheduler(SchedulerConfig{}) catch |err| {
        capi_error.setError(err, "failed to create reference scheduler for joint scoring", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer reference_scheduler.deinit();

    var model_scheduler = model_engine.createScheduler(SchedulerConfig{}) catch |err| {
        capi_error.setError(err, "failed to create model scheduler for joint scoring", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer model_scheduler.deinit();

    const reference_vocab = reference_scheduler.backend.vocabSize();
    const model_vocab = model_scheduler.backend.vocabSize();
    if (reference_vocab != model_vocab) {
        capi_error.setErrorWithCode(
            .invalid_argument,
            "reference/model vocab mismatch for joint scoring ({d} vs {d})",
            .{ reference_vocab, model_vocab },
        );
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }
    const ref_exp_scratch = reference_engine.allocator.alloc(f64, reference_vocab) catch |err| {
        capi_error.setError(err, "failed to allocate joint scoring scratch buffer", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer reference_engine.allocator.free(ref_exp_scratch);

    var token_history = std.ArrayList(u32){};
    defer token_history.deinit(allocator);
    token_history.ensureTotalCapacity(allocator, ctx_tokens.len + tgt_tokens.len) catch |err| {
        capi_error.setError(err, "failed to allocate joint scoring history", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    token_history.appendSlice(allocator, ctx_tokens) catch |err| {
        capi_error.setError(err, "failed to initialize joint scoring context", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    var reference_nll_sum: f64 = 0.0;
    var model_nll_sum: f64 = 0.0;
    var kld_sum: f64 = 0.0;
    var scored: usize = 0;

    for (tgt_tokens) |target| {
        const history_start = if (max_context > 0 and token_history.items.len > max_context)
            token_history.items.len - max_context
        else
            0;
        const prompt = token_history.items[history_start..];
        if (prompt.len == 0) {
            capi_error.setErrorWithCode(.invalid_argument, "joint scoring prompt is empty; provide non-empty context", .{});
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        }

        var reference_result = reference_scheduler.generateSync(prompt, 1, .{
            .return_final_logits = true,
            .eos_token_ids = &.{},
        }) catch |err| {
            capi_error.setError(err, "reference scheduler joint scoring step failed", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
        defer reference_result.deinit(allocator);

        var model_result = model_scheduler.generateSync(prompt, 1, .{
            .return_final_logits = true,
            .eos_token_ids = &.{},
        }) catch |err| {
            capi_error.setError(err, "model scheduler joint scoring step failed", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
        defer model_result.deinit(allocator);

        if (reference_result.final_logits.len == 0 or model_result.final_logits.len == 0) {
            capi_error.setErrorWithCode(.internal_error, "joint scoring returned no logits", .{});
            return @intFromEnum(error_codes.ErrorCode.internal_error);
        }
        if (reference_result.final_logits.len != model_result.final_logits.len) {
            capi_error.setErrorWithCode(
                .invalid_argument,
                "reference/model vocab mismatch for joint scoring ({d} vs {d})",
                .{ reference_result.final_logits.len, model_result.final_logits.len },
            );
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        }

        const reference_token_nll = negLogProbFromLogits(reference_result.final_logits, target) catch {
            capi_error.setErrorWithCode(
                .invalid_argument,
                "invalid reference target token {d} for vocab size {d}",
                .{ target, reference_result.final_logits.len },
            );
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        };
        const model_token_nll = negLogProbFromLogits(model_result.final_logits, target) catch {
            capi_error.setErrorWithCode(
                .invalid_argument,
                "invalid model target token {d} for vocab size {d}",
                .{ target, model_result.final_logits.len },
            );
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        };
        const token_kld = klDivergenceFromLogitsWithScratch(
            reference_result.final_logits,
            model_result.final_logits,
            ref_exp_scratch,
        ) catch {
            capi_error.setErrorWithCode(
                .invalid_argument,
                "invalid logits while computing joint scoring metrics (vocab={d})",
                .{reference_result.final_logits.len},
            );
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        };

        reference_nll_sum += reference_token_nll;
        model_nll_sum += model_token_nll;
        kld_sum += token_kld;
        scored += 1;

        token_history.append(allocator, target) catch |err| {
            capi_error.setError(err, "failed to extend joint scoring context", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
    }

    out_reference_nll_ptr.* = reference_nll_sum;
    out_model_nll_ptr.* = model_nll_sum;
    out_kld_ptr.* = kld_sum;
    out_count_ptr.* = scored;
    return 0;
}

test "negLogProbFromLogits returns ln2 for uniform binary logits" {
    const logits = [_]f32{ 0.0, 0.0 };
    const nll = try negLogProbFromLogits(&logits, 0);
    try std.testing.expectApproxEqAbs(@as(f64, std.math.log(f64, std.math.e, 2.0)), nll, 1e-12);
}

test "negLogProbFromLogits is smaller for higher target logit" {
    const logits = [_]f32{ 1.5, 0.2, -0.7 };
    const nll_top = try negLogProbFromLogits(&logits, 0);
    const nll_low = try negLogProbFromLogits(&logits, 2);
    try std.testing.expect(nll_top < nll_low);
}

test "klDivergenceFromLogits is near zero for identical logits" {
    const logits = [_]f32{ 1.2, -0.7, 0.3, 2.4 };
    const kld = try klDivergenceFromLogits(&logits, &logits);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), kld, 1e-12);
}

test "klDivergenceFromLogits matches known binary example" {
    const p_logits = [_]f32{ 0.0, 0.0 }; // p = [0.5, 0.5]
    const q_logits = [_]f32{ 0.6931472, 0.0 }; // q ~= [2/3, 1/3]
    const kld = try klDivergenceFromLogits(&p_logits, &q_logits);
    // KL([0.5,0.5] || [2/3,1/3]) = 0.5*ln(9/8)
    const expected = 0.5 * std.math.log(f64, std.math.e, 9.0 / 8.0);
    try std.testing.expectApproxEqRel(expected, kld, 1e-6);
}

test "klDivergenceFromLogitsWithScratch matches baseline" {
    const p_logits = [_]f32{ -0.7, 0.0, 0.3, 1.2, -1.1 };
    const q_logits = [_]f32{ -0.9, 0.1, 0.8, 0.7, -0.3 };
    const baseline = try klDivergenceFromLogits(&p_logits, &q_logits);

    var scratch: [5]f64 = undefined;
    const optimized = try klDivergenceFromLogitsWithScratch(&p_logits, &q_logits, scratch[0..]);
    try std.testing.expectApproxEqAbs(baseline, optimized, 1e-12);
}

test "canUseFastTeacherForcedScoring requires no rolling truncation" {
    try std.testing.expect(canUseFastTeacherForcedScoring(1, 4, 0));
    try std.testing.expect(canUseFastTeacherForcedScoring(3, 2, 4));
    try std.testing.expect(!canUseFastTeacherForcedScoring(3, 3, 4));
    try std.testing.expect(!canUseFastTeacherForcedScoring(0, 5, 4));
}
