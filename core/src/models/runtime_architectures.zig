//! Static runtime architecture registry.
//!
//! The architecture payload for each model lives in its model file.
//! This module provides id/model-type lookup only.

const std = @import("std");
const types = @import("op_types.zig");

const minilm = @import("bert/minilm.zig");
const gemma3 = @import("gemma/gemma3.zig");
const gpt_oss = @import("gpt_oss/gpt_oss.zig");
const granite3 = @import("granite/granite3.zig");
const granite_hybrid = @import("granite/granite_hybrid.zig");
const lfm2 = @import("lfm2/lfm2.zig");
const lfm2_5 = @import("lfm2/lfm2_5.zig");
const llama2 = @import("llama/llama2.zig");
const llama3 = @import("llama/llama3.zig");
const ministral3 = @import("mistral/ministral3.zig");
const phi4 = @import("phi/phi4.zig");
const qwen3 = @import("qwen/qwen3.zig");
const qwen3_moe = @import("qwen/qwen3_moe.zig");
const qwen3_next = @import("qwen/qwen3_next.zig");
const youtu_vl = @import("youtu_vl/youtu_vl.zig");

const registry = [_]struct { id: []const u8, arch: *const types.Architecture }{
    .{ .id = gemma3.id, .arch = &gemma3.arch },
    .{ .id = gpt_oss.id, .arch = &gpt_oss.arch },
    .{ .id = granite3.id, .arch = &granite3.arch },
    .{ .id = granite_hybrid.id, .arch = &granite_hybrid.arch },
    .{ .id = lfm2.id, .arch = &lfm2.arch },
    .{ .id = lfm2_5.id, .arch = &lfm2_5.arch },
    .{ .id = llama2.id, .arch = &llama2.arch },
    .{ .id = llama3.id, .arch = &llama3.arch },
    .{ .id = minilm.id, .arch = &minilm.arch },
    .{ .id = ministral3.id, .arch = &ministral3.arch },
    .{ .id = phi4.id, .arch = &phi4.arch },
    .{ .id = qwen3.id, .arch = &qwen3.arch },
    .{ .id = qwen3_moe.id, .arch = &qwen3_moe.arch },
    .{ .id = qwen3_next.id, .arch = &qwen3_next.arch },
    .{ .id = youtu_vl.id, .arch = &youtu_vl.arch },
};

pub fn byId(id: []const u8) ?*const types.Architecture {
    for (registry) |entry| {
        if (std.mem.eql(u8, id, entry.id)) return entry.arch;
    }
    return null;
}

pub fn detectByModelType(model_type: []const u8) ?*const types.Architecture {
    for (registry) |entry| {
        for (entry.arch.model_types) |candidate| {
            if (std.mem.eql(u8, model_type, candidate)) return entry.arch;
        }
    }
    return null;
}
