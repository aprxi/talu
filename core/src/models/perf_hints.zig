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
    /// CPU bench row name, for example `role.attn_q`.
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
    /// Direct xray point -> bench row mappings.
    point_mappings: []const PointBenchMap = &.{},
    /// Hidden compute rows that matter for the architecture but do not always
    /// appear clearly in xray summaries.
    hidden_rows: []const []const u8 = &.{},
    /// Representative role dimensions for architecture-level bench presets.
    role_dims: []const RoleDims = &.{},
};

/// Shared representative text-model role dimensions for architectures that do
/// not provide their own overrides. These are not intended to mirror one exact
/// checkpoint; they provide a stable architecture-level proxy so bench remains
/// usable even when a family spans multiple concrete sizes.
pub const default_text_role_dims = [_]RoleDims{
    .{ .bench_row = "role.attn_q", .tokens = 14, .hidden = 1024, .out = 1024 },
    .{ .bench_row = "role.attn_k", .tokens = 14, .hidden = 1024, .out = 256 },
    .{ .bench_row = "role.attn_v", .tokens = 14, .hidden = 1024, .out = 256 },
    .{ .bench_row = "role.attn_out", .tokens = 14, .hidden = 1024, .out = 1024 },
    .{ .bench_row = "role.ffn_gate", .tokens = 14, .hidden = 1024, .out = 4096 },
    .{ .bench_row = "role.ffn_down", .tokens = 14, .hidden = 4096, .out = 1024 },
};

pub fn pointMappingFor(hints: *const PerfHints, point: []const u8) ?[]const u8 {
    for (hints.point_mappings) |mapping| {
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

    try writer.writeAll(",\"point_mappings\":[");
    for (hints.point_mappings, 0..) |mapping, idx| {
        if (idx != 0) try writer.writeByte(',');
        try writer.writeAll("{\"point\":");
        try writeJsonString(writer, mapping.point);
        try writer.writeAll(",\"bench_row\":");
        try writeJsonString(writer, mapping.bench_row);
        try writer.writeByte('}');
    }
    try writer.writeByte(']');

    try writer.writeAll(",\"hidden_rows\":[");
    for (hints.hidden_rows, 0..) |row, idx| {
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

pub const standard_attention_mlp_point_mappings = [_]PointBenchMap{
    .{ .point = "attn.q", .bench_row = "role.attn_q" },
    .{ .point = "attn.k", .bench_row = "role.attn_k" },
    .{ .point = "attn.v", .bench_row = "role.attn_v" },
    .{ .point = "attn.out", .bench_row = "role.attn_out" },
    .{ .point = "ffn.gate", .bench_row = "role.ffn_gate" },
    .{ .point = "ffn.down", .bench_row = "role.ffn_down" },
    .{ .point = "embed_pos", .bench_row = "rope_f32" },
    .{ .point = "layer_attn_norm", .bench_row = "rms_f32" },
    .{ .point = "layer_ffn_norm", .bench_row = "rms_f32" },
    .{ .point = "final_norm", .bench_row = "rms_f32" },
};

pub const attention_norm_point_mappings = [_]PointBenchMap{
    .{ .point = "attn.q", .bench_row = "role.attn_q" },
    .{ .point = "attn.k", .bench_row = "role.attn_k" },
    .{ .point = "attn.v", .bench_row = "role.attn_v" },
    .{ .point = "attn.out", .bench_row = "role.attn_out" },
    .{ .point = "embed_pos", .bench_row = "rope_f32" },
    .{ .point = "layer_attn_norm", .bench_row = "rms_f32" },
    .{ .point = "layer_ffn_norm", .bench_row = "rms_f32" },
    .{ .point = "final_norm", .bench_row = "rms_f32" },
};

pub const ffn_norm_point_mappings = [_]PointBenchMap{
    .{ .point = "ffn.gate", .bench_row = "role.ffn_gate" },
    .{ .point = "ffn.down", .bench_row = "role.ffn_down" },
    .{ .point = "layer_attn_norm", .bench_row = "rms_f32" },
    .{ .point = "layer_ffn_norm", .bench_row = "rms_f32" },
    .{ .point = "final_norm", .bench_row = "rms_f32" },
};

pub const qwen3_5_hidden_rows = [_][]const u8{
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
    .{ .bench_row = "role.attn_q", .tokens = 14, .hidden = 1024, .out = 2048 },
    .{ .bench_row = "role.attn_k", .tokens = 14, .hidden = 1024, .out = 512 },
    .{ .bench_row = "role.attn_v", .tokens = 14, .hidden = 1024, .out = 512 },
    .{ .bench_row = "role.attn_out", .tokens = 14, .hidden = 1024, .out = 1024 },
    .{ .bench_row = "role.ffn_gate", .tokens = 14, .hidden = 1024, .out = 7168 },
    .{ .bench_row = "role.ffn_down", .tokens = 14, .hidden = 3584, .out = 1024 },
};

pub fn standardAttentionMlpHints(comptime bench_model: []const u8) PerfHints {
    return .{
        .bench_model = bench_model,
        .point_mappings = standard_attention_mlp_point_mappings[0..],
    };
}

pub fn standardAttentionMlpShortConvHints(comptime bench_model: []const u8) PerfHints {
    return .{
        .bench_model = bench_model,
        .point_mappings = standard_attention_mlp_point_mappings[0..],
        .hidden_rows = shortconv_hidden_rows[0..],
    };
}

pub fn standardAttentionMlpMambaHints(comptime bench_model: []const u8) PerfHints {
    return .{
        .bench_model = bench_model,
        .point_mappings = standard_attention_mlp_point_mappings[0..],
        .hidden_rows = mamba_hidden_rows[0..],
    };
}

pub fn attentionOnlyHints(comptime bench_model: []const u8) PerfHints {
    return .{
        .bench_model = bench_model,
        .point_mappings = attention_norm_point_mappings[0..],
    };
}

pub fn attentionOnlyMambaHints(comptime bench_model: []const u8) PerfHints {
    return .{
        .bench_model = bench_model,
        .point_mappings = attention_norm_point_mappings[0..],
        .hidden_rows = mamba_hidden_rows[0..],
    };
}

pub fn ffnNormHints(comptime bench_model: []const u8) PerfHints {
    return .{
        .bench_model = bench_model,
        .point_mappings = ffn_norm_point_mappings[0..],
    };
}

test "pointMappingFor resolves known mapping" {
    const hints = PerfHints{
        .bench_model = "demo",
        .point_mappings = standard_attention_mlp_point_mappings[0..],
    };
    try std.testing.expectEqualStrings("role.ffn_gate", pointMappingFor(&hints, "ffn.gate").?);
    try std.testing.expect(pointMappingFor(&hints, "not_a_point") == null);
}

test "writeJson includes bench model and hidden rows" {
    const hints = PerfHints{
        .bench_model = "qwen3_5",
        .point_mappings = standard_attention_mlp_point_mappings[0..1],
        .hidden_rows = qwen3_5_hidden_rows[0..],
        .role_dims = qwen3_5_role_dims[0..1],
    };

    var out = std.ArrayList(u8).empty;
    defer out.deinit(std.testing.allocator);
    try writeJson(out.writer(std.testing.allocator), &hints);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"bench_model\":\"qwen3_5\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"gdelta_step_f32\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "\"role.attn_q\"") != null);
}
