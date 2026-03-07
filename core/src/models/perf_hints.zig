//! Model-owned performance hint metadata.
//!
//! This module is the bridge between:
//! 1. inference-facing `xray` output (`core/src/inference/` call sites / points)
//! 2. compute-facing CPU benchmark rows (`core/bench/compute/cpu`)
//!
//! `models/` owns this mapping because it is architecture knowledge. `xray`
//! and `bench` are consumers and MUST NOT hardcode per-architecture mappings.

const std = @import("std");

fn writeJsonString(writer: anytype, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |ch| switch (ch) {
        '"' => try writer.writeAll("\\\""),
        '\\' => try writer.writeAll("\\\\"),
        '\n' => try writer.writeAll("\\n"),
        '\r' => try writer.writeAll("\\r"),
        '\t' => try writer.writeAll("\\t"),
        else => {
            if (ch < 0x20) {
                try writer.print("\\u{X:0>4}", .{@as(u32, ch)});
            } else {
                try writer.writeByte(ch);
            }
        },
    };
    try writer.writeByte('"');
}

pub const PointBenchMap = struct {
    /// Inference-facing xray point name.
    point: []const u8,
    /// Compute-facing CPU bench row name.
    bench_row: []const u8,
};

pub const RoleDims = struct {
    /// CPU bench row name, for example `prefill.attn_q` or `decode.ffn_gate`.
    bench_row: []const u8,
    /// Representative token count for the architecture-level bench preset.
    tokens: usize,
    /// Representative input width for the row.
    hidden: usize,
    /// Representative output width for the row.
    out: usize,
};

pub const PerfHints = struct {
    /// Architecture-level bench id used by `make -C core/bench/compute/cpu model=<id>`.
    bench_model: []const u8,
    /// Direct xray input/prefill point -> bench row mappings.
    prefill_point_mappings: []const PointBenchMap = &.{},
    /// Direct xray output/decode point -> bench row mappings.
    decode_point_mappings: []const PointBenchMap = &.{},
    /// Hidden compute rows that matter for prefill but do not always appear
    /// clearly in xray summaries.
    prefill_hidden_rows: []const []const u8 = &.{},
    /// Hidden compute rows that matter for decode but do not always appear
    /// clearly in xray summaries.
    decode_hidden_rows: []const []const u8 = &.{},
    /// Representative role dimensions for architecture-level bench presets.
    role_dims: []const RoleDims = &.{},
};

/// Shared representative text-model role dimensions for architectures that do
/// not provide their own overrides. These are phase-specific on purpose:
/// prefill and decode stress different compute paths and must not share one
/// ambiguous row name.
///
/// These are not intended to mirror one exact checkpoint; they provide a
/// stable architecture-level proxy so bench remains usable even when a family
/// spans multiple concrete sizes.
pub const default_text_role_dims = [_]RoleDims{
    .{ .bench_row = "prefill.attn_q", .tokens = 14, .hidden = 1024, .out = 1024 },
    .{ .bench_row = "prefill.attn_k", .tokens = 14, .hidden = 1024, .out = 256 },
    .{ .bench_row = "prefill.attn_v", .tokens = 14, .hidden = 1024, .out = 256 },
    .{ .bench_row = "prefill.attn_out", .tokens = 14, .hidden = 1024, .out = 1024 },
    .{ .bench_row = "prefill.ffn_gate", .tokens = 14, .hidden = 1024, .out = 4096 },
    .{ .bench_row = "prefill.ffn_down", .tokens = 14, .hidden = 4096, .out = 1024 },
    .{ .bench_row = "prefill.layer_attn_norm", .tokens = 14, .hidden = 1024, .out = 1024 },
    .{ .bench_row = "prefill.layer_ffn_norm", .tokens = 14, .hidden = 1024, .out = 1024 },
    .{ .bench_row = "prefill.final_norm", .tokens = 14, .hidden = 1024, .out = 1024 },
    .{ .bench_row = "decode.attn_q", .tokens = 1, .hidden = 1024, .out = 1024 },
    .{ .bench_row = "decode.attn_k", .tokens = 1, .hidden = 1024, .out = 256 },
    .{ .bench_row = "decode.attn_v", .tokens = 1, .hidden = 1024, .out = 256 },
    .{ .bench_row = "decode.attn_out", .tokens = 1, .hidden = 1024, .out = 1024 },
    .{ .bench_row = "decode.ffn_gate", .tokens = 1, .hidden = 1024, .out = 4096 },
    .{ .bench_row = "decode.ffn_down", .tokens = 1, .hidden = 4096, .out = 1024 },
    .{ .bench_row = "decode.layer_attn_norm", .tokens = 1, .hidden = 1024, .out = 1024 },
    .{ .bench_row = "decode.layer_ffn_norm", .tokens = 1, .hidden = 1024, .out = 1024 },
    .{ .bench_row = "decode.final_norm", .tokens = 1, .hidden = 1024, .out = 1024 },
};

pub fn pointMappingFor(mappings: []const PointBenchMap, point: []const u8) ?[]const u8 {
    for (mappings) |mapping| {
        if (std.mem.eql(u8, mapping.point, point)) return mapping.bench_row;
    }
    return null;
}

pub fn roleDimsFor(hints: *const PerfHints, bench_row: []const u8) ?RoleDims {
    for (hints.role_dims) |dims| {
        if (std.mem.eql(u8, dims.bench_row, bench_row)) return dims;
    }
    return null;
}

pub fn defaultRoleDimsFor(bench_row: []const u8) ?RoleDims {
    for (default_text_role_dims) |dims| {
        if (std.mem.eql(u8, dims.bench_row, bench_row)) return dims;
    }
    return null;
}

pub fn writeJson(writer: anytype, hints: *const PerfHints) !void {
    try writer.writeAll("{\"bench_model\":");
    try writeJsonString(writer, hints.bench_model);

    try writer.writeAll(",\"prefill_point_mappings\":[");
    for (hints.prefill_point_mappings, 0..) |mapping, idx| {
        if (idx != 0) try writer.writeByte(',');
        try writer.writeAll("{\"point\":");
        try writeJsonString(writer, mapping.point);
        try writer.writeAll(",\"bench_row\":");
        try writeJsonString(writer, mapping.bench_row);
        try writer.writeByte('}');
    }
    try writer.writeByte(']');

    try writer.writeAll(",\"decode_point_mappings\":[");
    for (hints.decode_point_mappings, 0..) |mapping, idx| {
        if (idx != 0) try writer.writeByte(',');
        try writer.writeAll("{\"point\":");
        try writeJsonString(writer, mapping.point);
        try writer.writeAll(",\"bench_row\":");
        try writeJsonString(writer, mapping.bench_row);
        try writer.writeByte('}');
    }
    try writer.writeByte(']');

    try writer.writeAll(",\"prefill_hidden_rows\":[");
    for (hints.prefill_hidden_rows, 0..) |row, idx| {
        if (idx != 0) try writer.writeByte(',');
        try writeJsonString(writer, row);
    }
    try writer.writeByte(']');

    try writer.writeAll(",\"decode_hidden_rows\":[");
    for (hints.decode_hidden_rows, 0..) |row, idx| {
        if (idx != 0) try writer.writeByte(',');
        try writeJsonString(writer, row);
    }
    try writer.writeByte(']');

    try writer.writeAll(",\"role_dims\":[");
    for (hints.role_dims, 0..) |dims, idx| {
        if (idx != 0) try writer.writeByte(',');
        try writer.writeAll("{\"bench_row\":");
        try writeJsonString(writer, dims.bench_row);
        try writer.writeAll(",\"tokens\":");
        try writer.print("{}", .{dims.tokens});
        try writer.writeAll(",\"hidden\":");
        try writer.print("{}", .{dims.hidden});
        try writer.writeAll(",\"out\":");
        try writer.print("{}", .{dims.out});
        try writer.writeByte('}');
    }
    try writer.writeAll("]}");
}

pub const standard_attention_mlp_prefill_point_mappings = [_]PointBenchMap{
    .{ .point = "attn.q", .bench_row = "prefill.attn_q" },
    .{ .point = "attn.k", .bench_row = "prefill.attn_k" },
    .{ .point = "attn.v", .bench_row = "prefill.attn_v" },
    .{ .point = "attn.out", .bench_row = "prefill.attn_out" },
    .{ .point = "ffn.gate", .bench_row = "prefill.ffn_gate" },
    .{ .point = "ffn.down", .bench_row = "prefill.ffn_down" },
    .{ .point = "embed_pos", .bench_row = "rope_f32" },
    .{ .point = "layer_attn_norm", .bench_row = "prefill.layer_attn_norm" },
    .{ .point = "layer_ffn_norm", .bench_row = "prefill.layer_ffn_norm" },
    .{ .point = "final_norm", .bench_row = "prefill.final_norm" },
};

pub const standard_attention_mlp_decode_point_mappings = [_]PointBenchMap{
    .{ .point = "attn.q", .bench_row = "decode.attn_q" },
    .{ .point = "attn.k", .bench_row = "decode.attn_k" },
    .{ .point = "attn.v", .bench_row = "decode.attn_v" },
    .{ .point = "attn.out", .bench_row = "decode.attn_out" },
    .{ .point = "ffn.gate", .bench_row = "decode.ffn_gate" },
    .{ .point = "ffn.down", .bench_row = "decode.ffn_down" },
    .{ .point = "embed_pos", .bench_row = "rope_f32" },
    .{ .point = "layer_attn_norm", .bench_row = "decode.layer_attn_norm" },
    .{ .point = "layer_ffn_norm", .bench_row = "decode.layer_ffn_norm" },
    .{ .point = "final_norm", .bench_row = "decode.final_norm" },
};

pub const attention_norm_prefill_point_mappings = [_]PointBenchMap{
    .{ .point = "attn.q", .bench_row = "prefill.attn_q" },
    .{ .point = "attn.k", .bench_row = "prefill.attn_k" },
    .{ .point = "attn.v", .bench_row = "prefill.attn_v" },
    .{ .point = "attn.out", .bench_row = "prefill.attn_out" },
    .{ .point = "embed_pos", .bench_row = "rope_f32" },
    .{ .point = "layer_attn_norm", .bench_row = "prefill.layer_attn_norm" },
    .{ .point = "layer_ffn_norm", .bench_row = "prefill.layer_ffn_norm" },
    .{ .point = "final_norm", .bench_row = "prefill.final_norm" },
};

pub const attention_norm_decode_point_mappings = [_]PointBenchMap{
    .{ .point = "attn.q", .bench_row = "decode.attn_q" },
    .{ .point = "attn.k", .bench_row = "decode.attn_k" },
    .{ .point = "attn.v", .bench_row = "decode.attn_v" },
    .{ .point = "attn.out", .bench_row = "decode.attn_out" },
    .{ .point = "embed_pos", .bench_row = "rope_f32" },
    .{ .point = "layer_attn_norm", .bench_row = "decode.layer_attn_norm" },
    .{ .point = "layer_ffn_norm", .bench_row = "decode.layer_ffn_norm" },
    .{ .point = "final_norm", .bench_row = "decode.final_norm" },
};

pub const ffn_norm_prefill_point_mappings = [_]PointBenchMap{
    .{ .point = "ffn.gate", .bench_row = "prefill.ffn_gate" },
    .{ .point = "ffn.down", .bench_row = "prefill.ffn_down" },
    .{ .point = "layer_attn_norm", .bench_row = "prefill.layer_attn_norm" },
    .{ .point = "layer_ffn_norm", .bench_row = "prefill.layer_ffn_norm" },
    .{ .point = "final_norm", .bench_row = "prefill.final_norm" },
};

pub const ffn_norm_decode_point_mappings = [_]PointBenchMap{
    .{ .point = "ffn.gate", .bench_row = "decode.ffn_gate" },
    .{ .point = "ffn.down", .bench_row = "decode.ffn_down" },
    .{ .point = "layer_attn_norm", .bench_row = "decode.layer_attn_norm" },
    .{ .point = "layer_ffn_norm", .bench_row = "decode.layer_ffn_norm" },
    .{ .point = "final_norm", .bench_row = "decode.final_norm" },
};

pub const qwen3_5_hidden_rows = [_][]const u8{
    "gdelta_conv_f32",
    "gdelta_qk_norm_f32",
    "gdelta_step_f32",
    "gdelta_norm_f32",
};

pub const shortconv_hidden_rows = [_][]const u8{
    "shortconv_f32",
};

pub const mamba_hidden_rows = [_][]const u8{
    "mamba_scan_f32",
};

pub const qwen3_5_role_dims = [_]RoleDims{
    .{ .bench_row = "prefill.attn_q", .tokens = 14, .hidden = 1024, .out = 2048 },
    .{ .bench_row = "prefill.attn_k", .tokens = 14, .hidden = 1024, .out = 512 },
    .{ .bench_row = "prefill.attn_v", .tokens = 14, .hidden = 1024, .out = 512 },
    .{ .bench_row = "prefill.attn_out", .tokens = 14, .hidden = 1024, .out = 1024 },
    .{ .bench_row = "prefill.ffn_gate", .tokens = 14, .hidden = 1024, .out = 7168 },
    .{ .bench_row = "prefill.ffn_down", .tokens = 14, .hidden = 3584, .out = 1024 },
    .{ .bench_row = "prefill.layer_attn_norm", .tokens = 14, .hidden = 1024, .out = 1024 },
    .{ .bench_row = "prefill.layer_ffn_norm", .tokens = 14, .hidden = 1024, .out = 1024 },
    .{ .bench_row = "prefill.final_norm", .tokens = 14, .hidden = 1024, .out = 1024 },
    .{ .bench_row = "decode.attn_q", .tokens = 1, .hidden = 1024, .out = 2048 },
    .{ .bench_row = "decode.attn_k", .tokens = 1, .hidden = 1024, .out = 512 },
    .{ .bench_row = "decode.attn_v", .tokens = 1, .hidden = 1024, .out = 512 },
    .{ .bench_row = "decode.attn_out", .tokens = 1, .hidden = 1024, .out = 1024 },
    .{ .bench_row = "decode.ffn_gate", .tokens = 1, .hidden = 1024, .out = 7168 },
    .{ .bench_row = "decode.ffn_down", .tokens = 1, .hidden = 3584, .out = 1024 },
    .{ .bench_row = "decode.lm_head_bf16", .tokens = 1, .hidden = 1024, .out = 248320 },
    .{ .bench_row = "decode.lm_head_f16", .tokens = 1, .hidden = 1024, .out = 248320 },
    .{ .bench_row = "decode.lm_head_f32", .tokens = 1, .hidden = 1024, .out = 248320 },
    .{ .bench_row = "decode.lm_head_runtime_f32", .tokens = 1, .hidden = 1024, .out = 248320 },
    .{ .bench_row = "decode.layer_attn_norm", .tokens = 1, .hidden = 1024, .out = 1024 },
    .{ .bench_row = "decode.layer_ffn_norm", .tokens = 1, .hidden = 1024, .out = 1024 },
    .{ .bench_row = "decode.final_norm", .tokens = 1, .hidden = 1024, .out = 1024 },
};

pub const qwen3_5_decode_point_mappings = [_]PointBenchMap{
    .{ .point = "attn.q", .bench_row = "decode.attn_q" },
    .{ .point = "attn.k", .bench_row = "decode.attn_k" },
    .{ .point = "attn.v", .bench_row = "decode.attn_v" },
    .{ .point = "attn.out", .bench_row = "decode.attn_out" },
    .{ .point = "ffn.gate", .bench_row = "decode.ffn_gate" },
    .{ .point = "ffn.down", .bench_row = "decode.ffn_down" },
    .{ .point = "lm_head", .bench_row = "decode.lm_head_runtime_f32" },
    .{ .point = "embed_pos", .bench_row = "rope_f32" },
    .{ .point = "layer_attn_norm", .bench_row = "decode.layer_attn_norm" },
    .{ .point = "layer_ffn_norm", .bench_row = "decode.layer_ffn_norm" },
    .{ .point = "final_norm", .bench_row = "decode.final_norm" },
};

pub fn standardAttentionMlpHints(comptime bench_model: []const u8) PerfHints {
    return .{
        .bench_model = bench_model,
        .prefill_point_mappings = standard_attention_mlp_prefill_point_mappings[0..],
        .decode_point_mappings = standard_attention_mlp_decode_point_mappings[0..],
    };
}

pub fn standardAttentionMlpShortConvHints(comptime bench_model: []const u8) PerfHints {
    return .{
        .bench_model = bench_model,
        .prefill_point_mappings = standard_attention_mlp_prefill_point_mappings[0..],
        .decode_point_mappings = standard_attention_mlp_decode_point_mappings[0..],
        .prefill_hidden_rows = shortconv_hidden_rows[0..],
        .decode_hidden_rows = shortconv_hidden_rows[0..],
    };
}

pub fn standardAttentionMlpMambaHints(comptime bench_model: []const u8) PerfHints {
    return .{
        .bench_model = bench_model,
        .prefill_point_mappings = standard_attention_mlp_prefill_point_mappings[0..],
        .decode_point_mappings = standard_attention_mlp_decode_point_mappings[0..],
        .prefill_hidden_rows = mamba_hidden_rows[0..],
        .decode_hidden_rows = mamba_hidden_rows[0..],
    };
}

pub fn attentionOnlyHints(comptime bench_model: []const u8) PerfHints {
    return .{
        .bench_model = bench_model,
        .prefill_point_mappings = attention_norm_prefill_point_mappings[0..],
        .decode_point_mappings = attention_norm_decode_point_mappings[0..],
    };
}

pub fn attentionOnlyMambaHints(comptime bench_model: []const u8) PerfHints {
    return .{
        .bench_model = bench_model,
        .prefill_point_mappings = attention_norm_prefill_point_mappings[0..],
        .decode_point_mappings = attention_norm_decode_point_mappings[0..],
        .prefill_hidden_rows = mamba_hidden_rows[0..],
        .decode_hidden_rows = mamba_hidden_rows[0..],
    };
}

pub fn ffnNormHints(comptime bench_model: []const u8) PerfHints {
    return .{
        .bench_model = bench_model,
        .prefill_point_mappings = ffn_norm_prefill_point_mappings[0..],
        .decode_point_mappings = ffn_norm_decode_point_mappings[0..],
    };
}

test "pointMappingFor resolves known mapping" {
    const hints = PerfHints{
        .bench_model = "demo",
        .prefill_point_mappings = standard_attention_mlp_prefill_point_mappings[0..],
    };
    try std.testing.expectEqualStrings("prefill.ffn_gate", pointMappingFor(hints.prefill_point_mappings, "ffn.gate").?);
    try std.testing.expect(pointMappingFor(hints.prefill_point_mappings, "not_a_point") == null);
}

test "writeJson includes bench model and hidden rows" {
    const hints = PerfHints{
        .bench_model = "qwen3_5",
        .prefill_point_mappings = standard_attention_mlp_prefill_point_mappings[0..1],
        .decode_point_mappings = standard_attention_mlp_decode_point_mappings[0..1],
        .prefill_hidden_rows = qwen3_5_hidden_rows[0..],
        .decode_hidden_rows = qwen3_5_hidden_rows[0..],
        .role_dims = qwen3_5_role_dims[0..1],
    };

    var out = std.ArrayList(u8).empty;
    defer out.deinit(std.testing.allocator);
    try writeJson(out.writer(std.testing.allocator), &hints);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"bench_model\":\"qwen3_5\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"gdelta_step_f32\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"prefill.attn_q\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"decode.attn_q\"") != null);
}
