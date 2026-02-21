//! Static model registry.
//!
//! Provides compile-time model metadata and block programs.

const std = @import("std");
const common_types = @import("common/types.zig");
const layer_ops = @import("layer_ops.zig");
const op_types = @import("op_types.zig");
const runtime_architectures = @import("runtime_architectures.zig");
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

pub const Entry = common_types.ModelDescriptor;

pub const entries: []const Entry = &.{
    .{
        .id = gemma3.id,
        .family = gemma3.family,
        .version = gemma3.version,
        .model_types = gemma3.model_types,
    },
    .{
        .id = granite3.id,
        .family = granite3.family,
        .version = granite3.version,
        .model_types = granite3.model_types,
    },
    .{
        .id = granite_hybrid.id,
        .family = granite_hybrid.family,
        .version = granite_hybrid.version,
        .model_types = granite_hybrid.model_types,
    },
    .{
        .id = lfm2.id,
        .family = lfm2.family,
        .version = lfm2.version,
        .model_types = lfm2.model_types,
    },
    .{
        .id = lfm2_5.id,
        .family = lfm2_5.family,
        .version = lfm2_5.version,
        .model_types = lfm2_5.model_types,
    },
    .{
        .id = llama2.id,
        .family = llama2.family,
        .version = llama2.version,
        .model_types = llama2.model_types,
    },
    .{
        .id = llama3.id,
        .family = llama3.family,
        .version = llama3.version,
        .model_types = llama3.model_types,
    },
    .{
        .id = ministral3.id,
        .family = ministral3.family,
        .version = ministral3.version,
        .model_types = ministral3.model_types,
    },
    .{
        .id = phi4.id,
        .family = phi4.family,
        .version = phi4.version,
        .model_types = phi4.model_types,
    },
    .{
        .id = qwen3.id,
        .family = qwen3.family,
        .version = qwen3.version,
        .model_types = qwen3.model_types,
    },
    .{
        .id = qwen3_moe.id,
        .family = qwen3_moe.family,
        .version = qwen3_moe.version,
        .model_types = qwen3_moe.model_types,
    },
    .{
        .id = youtu_vl.id,
        .family = youtu_vl.family,
        .version = youtu_vl.version,
        .model_types = youtu_vl.model_types,
    },
    .{
        .id = gpt_oss.id,
        .family = gpt_oss.family,
        .version = gpt_oss.version,
        .model_types = gpt_oss.model_types,
    },
    .{
        .id = minilm.id,
        .family = minilm.family,
        .version = minilm.version,
        .model_types = minilm.model_types,
    },
    .{
        .id = qwen3_next.id,
        .family = qwen3_next.family,
        .version = qwen3_next.version,
        .model_types = qwen3_next.model_types,
    },
};

pub fn detectByModelType(model_type: []const u8) ?Entry {
    for (entries) |entry| {
        for (entry.model_types) |candidate| {
            if (std.mem.eql(u8, model_type, candidate)) return entry;
        }
    }
    return null;
}

pub fn isSupportedModelType(model_type: []const u8) bool {
    return detectByModelType(model_type) != null;
}

pub fn detectByArchitectureId(arch_id: []const u8) ?Entry {
    for (entries) |entry| {
        if (std.mem.eql(u8, entry.id, arch_id)) return entry;
    }
    return null;
}

pub fn blockProgramFor(entry: Entry, block_kind: op_types.BlockKind) ?[]const layer_ops.LayerOp {
    if (std.mem.eql(u8, entry.id, qwen3_next.id)) {
        return switch (block_kind) {
            .attention_mlp => qwen3_next.attention_mlp_program,
            .mamba => qwen3_next.mamba_program,
            .shortconv => null,
        };
    }
    if (std.mem.eql(u8, entry.id, granite_hybrid.id)) {
        return switch (block_kind) {
            .attention_mlp => granite_hybrid.attention_mlp_program,
            .mamba => granite_hybrid.mamba_program,
            .shortconv => null,
        };
    }
    if (std.mem.eql(u8, entry.id, lfm2.id)) {
        return switch (block_kind) {
            .attention_mlp => lfm2.attention_mlp_program,
            .shortconv => lfm2.shortconv_program,
            .mamba => null,
        };
    }
    if (std.mem.eql(u8, entry.id, lfm2_5.id)) {
        return switch (block_kind) {
            .attention_mlp => lfm2_5.attention_mlp_program,
            .shortconv => lfm2_5.shortconv_program,
            .mamba => null,
        };
    }

    if (block_kind != .attention_mlp) return null;
    if (std.mem.eql(u8, entry.id, llama2.id)) return llama2.attention_mlp_program;
    if (std.mem.eql(u8, entry.id, llama3.id)) return llama3.attention_mlp_program;
    if (std.mem.eql(u8, entry.id, gemma3.id)) return gemma3.attention_mlp_program;
    if (std.mem.eql(u8, entry.id, granite3.id)) return granite3.attention_mlp_program;
    if (std.mem.eql(u8, entry.id, ministral3.id)) return ministral3.attention_mlp_program;
    if (std.mem.eql(u8, entry.id, phi4.id)) return phi4.attention_mlp_program;
    if (std.mem.eql(u8, entry.id, qwen3.id)) return qwen3.attention_mlp_program;
    if (std.mem.eql(u8, entry.id, qwen3_moe.id)) return qwen3_moe.attention_mlp_program;
    if (std.mem.eql(u8, entry.id, gpt_oss.id)) return gpt_oss.attention_mlp_program;
    if (std.mem.eql(u8, entry.id, minilm.id)) return minilm.attention_mlp_program;
    if (std.mem.eql(u8, entry.id, youtu_vl.id)) return youtu_vl.attention_mlp_program;
    return null;
}

pub fn loadArchitectureDefinitions(allocator: std.mem.Allocator) bool {
    _ = allocator;
    // Static models are compile-time metadata; no runtime architecture loading needed.
    return true;
}

pub fn runtimeArchitectureById(arch_id: []const u8) ?*const op_types.Architecture {
    return runtime_architectures.byId(arch_id);
}

pub fn runtimeArchitectureByModelType(model_type: []const u8) ?*const op_types.Architecture {
    return runtime_architectures.detectByModelType(model_type);
}

test "registry supports llama and qwen model types" {
    try std.testing.expect(isSupportedModelType("llama3"));
    try std.testing.expect(isSupportedModelType("qwen3"));
    try std.testing.expect(isSupportedModelType("qwen3_moe"));
    try std.testing.expect(isSupportedModelType("gemma3"));
    try std.testing.expect(isSupportedModelType("granite"));
    try std.testing.expect(isSupportedModelType("phi4"));
    try std.testing.expect(isSupportedModelType("lfm2"));
    try std.testing.expect(isSupportedModelType("lfm2_5"));
    try std.testing.expect(isSupportedModelType("granite_hybrid"));
    try std.testing.expect(isSupportedModelType("youtu_vl"));
}

test "registry returns canonical entry for known model type" {
    const entry = detectByModelType("granitehybrid");
    try std.testing.expect(entry != null);
    try std.testing.expectEqualStrings("granite_hybrid", entry.?.id);
}

test "registry rejects unknown model type" {
    try std.testing.expect(!isSupportedModelType("unknown_model_type"));
}

test "registry finds entry by architecture id" {
    const entry = detectByArchitectureId("llama3");
    try std.testing.expect(entry != null);
    try std.testing.expectEqualStrings("llama3", entry.?.id);
}

test "blockProgramFor returns program for known block kinds" {
    const entry = detectByArchitectureId("lfm2") orelse return error.TestUnexpectedResult;
    try std.testing.expect(blockProgramFor(entry, .attention_mlp) != null);
    try std.testing.expect(blockProgramFor(entry, .shortconv) != null);
}

test "loadArchitectureDefinitions returns true for static registry" {
    try std.testing.expect(loadArchitectureDefinitions(std.testing.allocator));
}

test "runtimeArchitectureById and runtimeArchitectureByModelType resolve payloads" {
    const by_id = runtimeArchitectureById("llama3");
    try std.testing.expect(by_id != null);
    try std.testing.expectEqualStrings("llama3", by_id.?.name);

    const by_model_type = runtimeArchitectureByModelType("llama3");
    try std.testing.expect(by_model_type != null);
    try std.testing.expectEqualStrings("llama3", by_model_type.?.name);
}
