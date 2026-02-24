//! Decode path entrypoints extracted from CUDA engine.

const contract = @import("../contract.zig");
const common_mrope = @import("../../vision_mrope.zig");
const log = @import("../../../log.zig");

pub fn decode(self: anytype, token: u32, position: usize, logits_out: []f32) !void {
    if (logits_out.len != self.vocab_size) {
        log.warn("inference", "CUDA decode invalid args", .{
            .reason = "logits_len_mismatch",
            .logits_len = logits_out.len,
            .vocab_size = self.vocab_size,
        });
        return error.InvalidArgument;
    }
    const effective_position = try common_mrope.applyPositionDelta(position, self.slot_rope_position_delta);
    try self.computeGpuPrototypeLogits(token, effective_position, logits_out);
    self.slot_position = position + 1;
}

pub fn decodeBatch(
    self: anytype,
    requests: []const contract.DecodeRequest,
    results: []contract.DecodeResult,
) !void {
    if (results.len < requests.len) {
        log.warn("inference", "CUDA decodeBatch invalid args", .{
            .reason = "results_short",
            .requests = requests.len,
            .results = results.len,
        });
        return error.InvalidArgument;
    }
    if (requests.len == 0) return;
    if (requests.len > 1) {
        log.warn("inference", "CUDA decodeBatch invalid args", .{
            .reason = "batch_gt_one",
            .requests = requests.len,
        });
        return error.InvalidArgument;
    }

    const req = requests[0];
    if (!self.slot_in_use or req.slot_index != 0) {
        log.warn("inference", "CUDA decodeBatch invalid args", .{
            .reason = "slot_state",
            .slot_index = req.slot_index,
            .slot_in_use = @as(u8, @intFromBool(self.slot_in_use)),
        });
        return error.InvalidArgument;
    }

    const effective_position = try common_mrope.applyPositionDelta(self.slot_position, self.slot_rope_position_delta);
    try self.computeGpuPrototypeLogits(req.token, effective_position, self.slot_logits);
    results[0] = .{
        .slot_index = req.slot_index,
        .logits = self.slot_logits,
    };
    self.slot_position += 1;
}

pub fn decodeStreaming(
    self: anytype,
    first_token: u32,
    start_position: usize,
    max_tokens: usize,
    eos_token_ids: []const u32,
    output_tokens: []u32,
    callback: ?*const fn (u32, ?*anyopaque) void,
    callback_data: ?*anyopaque,
) !usize {
    if (max_tokens == 0 or output_tokens.len == 0) return 0;
    if (!self.slot_in_use) {
        self.slot_in_use = true;
        self.slot_position = start_position;
    }

    var current_token = first_token;
    var generated: usize = 0;
    var position = self.slot_position;
    const budget = @min(max_tokens, output_tokens.len);
    while (generated < budget) : (generated += 1) {
        const effective_position = try common_mrope.applyPositionDelta(position, self.slot_rope_position_delta);
        try self.computeGpuPrototypeLogitsWithLayerLimit(
            current_token,
            effective_position,
            null,
            self.block_runtime.blocks.len,
            true,
            false,
            true,
            null,
            null,
            null,
        );
        const next_token = try self.selectNextTokenFromDeviceLogitsImpl();
        output_tokens[generated] = next_token;
        position += 1;
        self.slot_position = position;
        if (callback) |cb| cb(next_token, callback_data);

        for (eos_token_ids) |eos_id| {
            if (next_token == eos_id) {
                return generated + 1;
            }
        }
        current_token = next_token;
    }
    return generated;
}

pub fn allocSlot(self: anytype) ?usize {
    if (self.slot_in_use) return null;
    self.slot_in_use = true;
    self.slot_position = 0;
    self.slot_rope_position_delta = 0;
    return 0;
}

pub fn freeSlot(self: anytype, slot_index: usize) void {
    if (slot_index != 0) return;
    self.slot_in_use = false;
    self.slot_position = 0;
    self.slot_rope_position_delta = 0;
}

pub fn resetSlot(self: anytype, slot_index: usize) void {
    if (slot_index != 0) return;
    self.slot_position = 0;
    self.slot_rope_position_delta = 0;
}

pub fn getPosition(self: anytype, slot_index: usize) usize {
    if (slot_index != 0) return 0;
    return self.slot_position;
}
