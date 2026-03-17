//! Batch C API — Responses-Aware Continuous Batching
//!
//! Thin C API glue over router/batch.zig. Wraps the BatchWrapper as an
//! opaque handle (TaluBatch) and exposes submit/step/cancel/result functions.
//!
//! Lifecycle: create → (submit | cancel | step)* → destroy
//! The backend (TaluInferenceBackend) must outlive the batch handle.
//!
//! Thread safety: NOT thread-safe. Caller must serialize all calls.

const std = @import("std");
const batch_mod = @import("../router/batch.zig");
const spec_mod = @import("../router/spec.zig");
const local_mod = @import("../router/local.zig");
const capi_bridge = @import("../router/capi_bridge.zig");
const router_capi = @import("router.zig");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");
const responses_capi = @import("responses.zig");
const responses_mod = @import("../responses/root.zig");

const allocator = std.heap.c_allocator;

const TaluInferenceBackend = router_capi.TaluInferenceBackend;
const ChatHandle = responses_capi.ChatHandle;
const BatchWrapper = batch_mod.BatchWrapper;
const SchedulerConfig = local_mod.SchedulerConfig;

// =============================================================================
// Opaque Handle
// =============================================================================

/// Opaque batch handle for responses-aware continuous batching.
pub const TaluBatch = opaque {};

// =============================================================================
// C-Compatible Types
// =============================================================================

/// Batch configuration. Pass null to talu_batch_create for defaults.
pub const CBatchConfig = extern struct {
    /// Maximum concurrent requests (0 = use backend default).
    max_concurrent: usize = 0,
};

/// Rich event from talu_batch_step.
pub const CBatchEvent = extern struct {
    request_id: u64,
    event_type: u8,
    item_type: u8,
    content_type: u8,
    is_final: u8,
    _pad: [4]u8 = .{ 0, 0, 0, 0 },
    text_ptr: ?[*]const u8 = null,
    text_len: usize = 0,
    token_id: u32,
    _pad2: [4]u8 = .{ 0, 0, 0, 0 },
    tokens_generated: usize = 0,
    timestamp_ns: i64 = 0,
};

/// Completion result from talu_batch_take_result.
pub const CBatchResult = extern struct {
    prompt_tokens: usize = 0,
    completion_tokens: usize = 0,
    prefill_ns: u64 = 0,
    generation_ns: u64 = 0,
    ttft_ns: u64 = 0,
    finish_reason: u8 = 0,
    _pad: [7]u8 = .{0} ** 7,
    text: ?[*:0]u8 = null,
    tool_calls: ?[*]const capi_bridge.CToolCallRef = null,
    tool_call_count: usize = 0,
    error_code: i32 = 0,
    _pad2: [4]u8 = .{0} ** 4,
};

// =============================================================================
// Lifecycle
// =============================================================================

/// Create a batch handle bound to a local inference backend.
///
/// The backend must remain valid until talu_batch_destroy is called.
/// Returns null on error (check talu_last_error).
pub export fn talu_batch_create(
    backend: ?*TaluInferenceBackend,
    config: ?*const CBatchConfig,
) callconv(.c) ?*TaluBatch {
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

    const wrapper = allocator.create(BatchWrapper) catch {
        capi_error.setErrorWithCode(.out_of_memory, "batch allocation failed", .{});
        return null;
    };

    wrapper.* = BatchWrapper.init(engine, sched_config) catch |err| {
        allocator.destroy(wrapper);
        capi_error.setError(err, "batch creation failed", .{});
        return null;
    };

    return @ptrCast(wrapper);
}

/// Destroy a batch handle and free all resources.
///
/// All pending/active requests are cleaned up. Passing null is a safe no-op.
pub export fn talu_batch_destroy(handle: ?*TaluBatch) callconv(.c) void {
    const wrapper: *BatchWrapper = @ptrCast(@alignCast(handle orelse return));
    wrapper.deinit();
    allocator.destroy(wrapper);
}

// =============================================================================
// Request Management
// =============================================================================

/// Submit a generation request.
///
/// Applies the chat template, tokenizes, sets up grammar constraints,
/// and submits to the scheduler. Returns a non-zero request ID on success.
pub export fn talu_batch_submit(
    handle: ?*TaluBatch,
    chat_handle: ?*ChatHandle,
    config: ?*const capi_bridge.CGenerateConfig,
) callconv(.c) u64 {
    capi_error.clearError();

    const wrapper: *BatchWrapper = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "batch handle is null", .{});
        return 0;
    }));

    const chat: *responses_mod.Chat = @ptrCast(@alignCast(chat_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "chat handle is null", .{});
        return 0;
    }));

    return wrapper.submit(chat, config) catch |err| {
        capi_error.setError(err, "batch submit failed", .{});
        return 0;
    };
}

/// Cancel a request.
///
/// Returns 1 if cancelled, 0 if not found or already finished.
pub export fn talu_batch_cancel(
    handle: ?*TaluBatch,
    request_id: u64,
) callconv(.c) u8 {
    capi_error.clearError();
    const wrapper: *BatchWrapper = @ptrCast(@alignCast(handle orelse return 0));
    return @intFromBool(wrapper.cancel(request_id));
}

// =============================================================================
// Step Loop
// =============================================================================

/// Run one generation step for all active requests.
///
/// Produces rich events with decoded text, item/content type metadata,
/// and timing information. Returns the number of events written.
///
/// Text pointers in events are valid until the next talu_batch_step call.
pub export fn talu_batch_step(
    handle: ?*TaluBatch,
    events_out: ?*CBatchEvent,
    max_events: usize,
) callconv(.c) usize {
    capi_error.clearError();

    const wrapper: *BatchWrapper = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "batch handle is null", .{});
        return 0;
    }));

    // Allocate temporary internal event buffer.
    var internal_events = allocator.alloc(batch_mod.BatchEvent, max_events) catch {
        capi_error.setErrorWithCode(.out_of_memory, "event buffer allocation failed", .{});
        return 0;
    };
    defer allocator.free(internal_events);

    const count = wrapper.step(internal_events) catch |err| {
        capi_error.setError(err, "batch step failed", .{});
        return 0;
    };

    // Convert to C events.
    const out_ptr = events_out orelse {
        if (count > 0) {
            capi_error.setErrorWithCode(.invalid_argument, "events_out is null", .{});
        }
        return 0;
    };

    const out_buf: [*]CBatchEvent = @ptrCast(out_ptr);
    const copy_count = @min(count, max_events);

    for (internal_events[0..copy_count], 0..) |ev, i| {
        var c_event = std.mem.zeroes(CBatchEvent);
        c_event.request_id = ev.request_id;
        c_event.event_type = @intFromEnum(ev.event_type);
        c_event.item_type = ev.item_type;
        c_event.content_type = ev.content_type;
        c_event.is_final = @intFromBool(ev.is_final);
        c_event.text_ptr = if (ev.text.len > 0) ev.text.ptr else null;
        c_event.text_len = ev.text.len;
        c_event.token_id = ev.token_id;
        c_event.tokens_generated = ev.tokens_generated;
        c_event.timestamp_ns = @intCast(@min(ev.timestamp_ns, std.math.maxInt(i64)));
        out_buf[i] = c_event;
    }

    return count;
}

// =============================================================================
// Result
// =============================================================================

/// Take the completion result for a finished request.
///
/// Returns null if the request is not complete or was already taken.
/// Caller must free via talu_batch_result_free.
pub export fn talu_batch_take_result(
    handle: ?*TaluBatch,
    request_id: u64,
) callconv(.c) ?*CBatchResult {
    capi_error.clearError();

    const wrapper: *BatchWrapper = @ptrCast(@alignCast(handle orelse return null));

    const internal_result = wrapper.takeResult(request_id) orelse return null;
    defer {
        internal_result.deinit();
        allocator.destroy(internal_result);
    }

    // Convert to C result.
    const c_result = allocator.create(CBatchResult) catch {
        capi_error.setErrorWithCode(.out_of_memory, "result allocation failed", .{});
        return null;
    };
    c_result.* = std.mem.zeroes(CBatchResult);

    c_result.prompt_tokens = internal_result.prompt_tokens;
    c_result.completion_tokens = internal_result.completion_tokens;
    c_result.prefill_ns = internal_result.prefill_ns;
    c_result.generation_ns = internal_result.generation_ns;
    c_result.ttft_ns = internal_result.ttft_ns;
    c_result.finish_reason = @intFromEnum(internal_result.finish_reason);

    // Copy text to null-terminated C string.
    if (internal_result.text) |text| {
        const cstr = allocator.allocSentinel(u8, text.len, 0) catch {
            allocator.destroy(c_result);
            capi_error.setErrorWithCode(.out_of_memory, "result text copy failed", .{});
            return null;
        };
        @memcpy(cstr, text);
        c_result.text = cstr.ptr;
    }

    // Transfer tool calls (already in C format).
    if (internal_result.tool_calls) |calls| {
        // Duplicate the array (internal_result will free its copy).
        const c_calls = allocator.alloc(capi_bridge.CToolCallRef, calls.len) catch {
            if (c_result.text) |t| {
                allocator.free(t[0 .. std.mem.len(t) + 1]);
            }
            allocator.destroy(c_result);
            capi_error.setErrorWithCode(.out_of_memory, "result tool calls copy failed", .{});
            return null;
        };
        for (calls, 0..) |call, i| {
            var c_call = std.mem.zeroes(capi_bridge.CToolCallRef);
            c_call.item_index = call.item_index;
            if (call.call_id) |cid| c_call.call_id = allocator.dupeZ(u8, std.mem.span(cid)) catch null;
            if (call.name) |n| c_call.name = allocator.dupeZ(u8, std.mem.span(n)) catch null;
            if (call.arguments) |a| c_call.arguments = allocator.dupeZ(u8, std.mem.span(a)) catch null;
            c_calls[i] = c_call;
        }
        c_result.tool_calls = c_calls.ptr;
        c_result.tool_call_count = calls.len;
    }

    return c_result;
}

/// Free a batch result returned by talu_batch_take_result.
pub export fn talu_batch_result_free(result: ?*CBatchResult) callconv(.c) void {
    const r = result orelse return;

    if (r.text) |text| {
        allocator.free(text[0 .. std.mem.len(text) + 1]);
    }
    if (r.tool_calls) |calls| {
        for (0..r.tool_call_count) |i| {
            const call = calls[i];
            if (call.call_id) |cid| freeSentinel(cid);
            if (call.name) |n| freeSentinel(n);
            if (call.arguments) |a| freeSentinel(a);
        }
        allocator.free(calls[0..r.tool_call_count]);
    }
    allocator.destroy(r);
}

fn freeSentinel(ptr: [*:0]const u8) void {
    const span = std.mem.span(ptr);
    allocator.free(span.ptr[0 .. span.len + 1]);
}

// =============================================================================
// Query
// =============================================================================

/// Check if there are any active or pending requests.
pub export fn talu_batch_has_active(
    handle: ?*const TaluBatch,
) callconv(.c) u8 {
    capi_error.clearError();
    const wrapper: *const BatchWrapper = @ptrCast(@alignCast(handle orelse return 0));
    return @intFromBool(wrapper.hasActive());
}

/// Get the number of active requests.
pub export fn talu_batch_active_count(
    handle: ?*const TaluBatch,
) callconv(.c) usize {
    capi_error.clearError();
    const wrapper: *const BatchWrapper = @ptrCast(@alignCast(handle orelse return 0));
    return wrapper.activeCount();
}
