//! Shared helpers for CUDA execution routes.
//!
//! Route entrypoints live in sibling files. This module is deliberately limited
//! to helper logic used by more than one route.

const std = @import("std");
const compute = @import("compute_pkg");
const log = @import("log_pkg");

const engine_types = @import("../runtime/root.zig");
const AttentionKernelSet = engine_types.AttentionKernelSet;

pub fn typeHasDecl(comptime T: type, comptime name: []const u8) bool {
    return switch (@typeInfo(T)) {
        .@"struct", .@"enum", .@"union", .@"opaque" => @hasDecl(T, name),
        else => false,
    };
}

pub fn logDecodeInventoryOnce(self: anytype, mode_label: []const u8, token_count: usize, batch_rows: usize) void {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "decode_inventory_logged") and @hasDecl(SelfType, "logDecodeInventorySummaryImpl")) {
        if (!self.decode_inventory_logged) {
            self.logDecodeInventorySummaryImpl(mode_label, token_count, batch_rows);
            self.decode_inventory_logged = true;
        }
    }
}

/// Download and log first N f32 values + L2 norm from a device buffer.
/// Gated by TALU_DUMP_HIDDEN env var. Uses log.warn so it survives ReleaseFast.
pub fn dumpHiddenState(
    self: anytype,
    buf: *const compute.cuda.Buffer,
    global_layer_idx: usize,
    label: []const u8,
    d_model: usize,
    rows: usize,
) void {
    const dump_env = @import("env_pkg").getenv("TALU_DUMP_HIDDEN");
    if (dump_env == null) return;
    _ = rows;

    // Sync the entire CUDA context first.
    self.device.synchronize() catch |err| {
        log.warn("inference", "DUMP_HIDDEN_SKIP", .{ .layer = global_layer_idx, .label = label, .reason = "sync_err", .err = @intFromError(err) });
        return;
    };

    // Download full row to hidden_host using raw cu_memcpy_dtoh.
    const n = @min(d_model, self.runtime_buffers.hidden_host.len);
    if (n == 0) return;
    const download_bytes = n * @sizeOf(f32);
    if (buf.pointer == 0 or buf.size < download_bytes) {
        log.warn("inference", "DUMP_HIDDEN_SKIP", .{
            .layer = global_layer_idx,
            .label = label,
            .reason = "bad_buf",
            .ptr = buf.pointer,
            .buf_size = buf.size,
            .need = download_bytes,
        });
        return;
    }

    self.device.makeCurrent() catch |err| {
        log.warn("inference", "DUMP_HIDDEN_SKIP", .{ .layer = global_layer_idx, .label = label, .reason = "make_current_err", .err = @intFromError(err) });
        return;
    };

    const host_ptr: *anyopaque = @ptrCast(self.runtime_buffers.hidden_host.ptr);
    const rc = self.device.api.cu_memcpy_dtoh(host_ptr, buf.pointer, download_bytes);
    if (rc != 0) {
        log.warn("inference", "DUMP_HIDDEN_SKIP", .{
            .layer = global_layer_idx,
            .label = label,
            .reason = "cu_memcpy_rc",
            .rc = rc,
            .dev_ptr = buf.pointer,
            .buf_size = buf.size,
            .download_bytes = download_bytes,
        });
        return;
    }

    const host = self.runtime_buffers.hidden_host[0..n];
    var sum: f64 = 0.0;
    for (host) |value| sum += @as(f64, value) * @as(f64, value);
    const l2_norm: f32 = @floatCast(@sqrt(sum));

    const n8 = @min(8, n);
    var values: [8]f32 = .{0} ** 8;
    @memcpy(values[0..n8], host[0..n8]);

    log.warn("inference", "DUMP_HIDDEN", .{
        .layer = global_layer_idx,
        .label = label,
        .l2_norm = l2_norm,
        .v0 = values[0],
        .v1 = values[1],
        .v2 = values[2],
        .v3 = values[3],
        .v4 = values[4],
        .v5 = values[5],
        .v6 = values[6],
        .v7 = values[7],
    });
}

pub fn applyHostLogitsPostProcess(
    logits: []f32,
    logits_scaling: f32,
    final_logit_softcapping: f32,
) void {
    if (logits_scaling != 1.0) {
        for (logits) |*value| {
            value.* /= logits_scaling;
        }
    }
    if (final_logit_softcapping > 0.0) {
        for (logits) |*value| {
            value.* = std.math.tanh(value.* / final_logit_softcapping) * final_logit_softcapping;
        }
    }
}

pub fn executeCpuStage0LayerRange(
    stage0: anytype,
    token: u32,
    position: usize,
    slot_index: usize,
    split_layer: usize,
    ensure_kv_capacity: bool,
) !void {
    const Stage0Type = @TypeOf(stage0.*);
    if (comptime !typeHasDecl(Stage0Type, "executeDecodeLayerRange")) {
        return error.InvalidTopologyConfig;
    }
    try stage0.executeDecodeLayerRange(
        token,
        position,
        slot_index,
        null,
        0,
        split_layer,
        false,
        false,
        ensure_kv_capacity,
        false,
    );
}

pub fn localActivationByteCountFor(self: anytype) !usize {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasDecl(SelfType, "localActivationByteCount")) {
        return self.localActivationByteCount();
    }
    return std.math.mul(usize, self.d_model, @sizeOf(f32)) catch error.InvalidArgument;
}

pub fn buildAttentionKernelSet(backend: anytype) !AttentionKernelSet {
    return switch (backend.kv_cache_dtype) {
        .f16 => .{
            .attn_scores_heads_f16_kv_function = backend.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
            .attn_weighted_sum_heads_f16_kv_function = backend.attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
            .attn_fused_heads_f16_kv_function = backend.attn_fused_heads_f16_kv_function,
            .attn_fused_prefill_heads_f16_kv_function = backend.attn_fused_prefill_heads_f16_kv_function,
            .attn_fused_prefill_heads_f16_kv_gqa_function = backend.attn_fused_prefill_heads_f16_kv_gqa_function,
            .softmax_rows_function = backend.softmax_rows_function,
            .causal_attn_softmax_f32_function = backend.causal_attn_softmax_f32_function,
        },
        .i8 => .{
            .attn_scores_heads_i8_kv_function = backend.attn_scores_heads_i8_kv_function,
            .attn_weighted_sum_heads_i8_kv_function = backend.attn_weighted_sum_heads_i8_kv_function,
            .attn_fused_heads_i8_kv_function = backend.attn_fused_heads_i8_kv_function,
            .attn_fused_prefill_heads_i8_kv_function = backend.attn_fused_prefill_heads_i8_kv_function,
            .attn_fused_prefill_heads_i8_kv_gqa_function = backend.attn_fused_prefill_heads_i8_kv_gqa_function,
            .softmax_rows_function = backend.softmax_rows_function,
            .causal_attn_softmax_f32_function = backend.causal_attn_softmax_f32_function,
        },
        .fp8 => .{
            .attn_scores_heads_fp8_kv_function = backend.attn_scores_heads_fp8_kv_function,
            .attn_weighted_sum_heads_fp8_kv_function = backend.attn_weighted_sum_heads_fp8_kv_function,
            .attn_fused_heads_fp8_kv_function = backend.attn_fused_heads_fp8_kv_function,
            .attn_fused_prefill_heads_fp8_kv_function = backend.attn_fused_prefill_heads_fp8_kv_function,
            .attn_fused_prefill_heads_fp8_kv_gqa_function = backend.attn_fused_prefill_heads_fp8_kv_gqa_function,
            .softmax_rows_function = backend.softmax_rows_function,
            .causal_attn_softmax_f32_function = backend.causal_attn_softmax_f32_function,
        },
    };
}
