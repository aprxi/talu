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
