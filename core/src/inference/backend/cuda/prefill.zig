//! Prefill path entrypoints extracted from CUDA engine.

const std = @import("std");
const log = @import("../../../log.zig");

pub fn prefill(self: anytype, tokens: []const u32, logits_out: []f32) !void {
    if (tokens.len == 0) {
        log.warn("inference", "CUDA prefill invalid args", .{
            .reason = "empty_tokens",
        });
        return error.InvalidArgument;
    }
    if (logits_out.len != self.vocab_size) {
        log.warn("inference", "CUDA prefill invalid args", .{
            .reason = "logits_len_mismatch",
            .logits_len = logits_out.len,
            .vocab_size = self.vocab_size,
        });
        return error.InvalidArgument;
    }
    if (tokens.len > self.max_seq_len) return error.InvalidArgument;
    self.slot_rope_position_delta = 0;
    if (try self.trySequencePrefill(tokens, logits_out, "prefill_seq")) {
        self.slot_position = tokens.len;
        return;
    }
    const prefill_start_ns: i128 = std.time.nanoTimestamp();
    try self.ensureKvCapacity(tokens.len);

    var i: usize = 0;
    while (i < tokens.len) : (i += 1) {
        const download_logits = self.shouldDownloadPrefillLogitsImpl(i, tokens.len);
        try self.computeGpuPrototypeLogitsWithLayerLimit(
            tokens[i],
            i,
            if (download_logits) self.slot_logits else null,
            self.block_runtime.blocks.len,
            download_logits,
            download_logits,
            false,
            null,
        );
    }
    const prefill_elapsed_ns: u64 = @intCast(std.time.nanoTimestamp() - prefill_start_ns);
    self.logPrefillTimingImpl("prefill", tokens.len, prefill_elapsed_ns);
    @memcpy(logits_out, self.slot_logits);
    self.slot_position = tokens.len;
}

pub fn prefillSlot(
    self: anytype,
    slot_index: usize,
    tokens: []const u32,
    logits_out: []f32,
) !void {
    if (tokens.len == 0) {
        log.warn("inference", "CUDA prefillSlot invalid args", .{
            .reason = "empty_tokens",
            .slot_index = slot_index,
        });
        return error.InvalidArgument;
    }
    if (logits_out.len != self.vocab_size) {
        log.warn("inference", "CUDA prefillSlot invalid args", .{
            .reason = "logits_len_mismatch",
            .slot_index = slot_index,
            .logits_len = logits_out.len,
            .vocab_size = self.vocab_size,
        });
        return error.InvalidArgument;
    }
    if (!self.slot_in_use or slot_index != 0) {
        log.warn("inference", "CUDA prefillSlot invalid args", .{
            .reason = "slot_state",
            .slot_index = slot_index,
            .slot_in_use = @as(u8, @intFromBool(self.slot_in_use)),
        });
        return error.InvalidArgument;
    }
    if (tokens.len > self.max_seq_len) return error.InvalidArgument;
    self.slot_rope_position_delta = 0;
    if (try self.trySequencePrefill(tokens, logits_out, "prefill_slot_seq")) {
        self.slot_position = tokens.len;
        return;
    }
    const prefill_start_ns: i128 = std.time.nanoTimestamp();
    try self.ensureKvCapacity(tokens.len);
    var i: usize = 0;
    while (i < tokens.len) : (i += 1) {
        const download_logits = self.shouldDownloadPrefillLogitsImpl(i, tokens.len);
        try self.computeGpuPrototypeLogitsWithLayerLimit(
            tokens[i],
            i,
            if (download_logits) self.slot_logits else null,
            self.block_runtime.blocks.len,
            download_logits,
            download_logits,
            false,
            null,
        );
    }
    const prefill_elapsed_ns: u64 = @intCast(std.time.nanoTimestamp() - prefill_start_ns);
    self.logPrefillTimingImpl("prefill_slot", tokens.len, prefill_elapsed_ns);
    @memcpy(logits_out, self.slot_logits);
    self.slot_position = tokens.len;
}
