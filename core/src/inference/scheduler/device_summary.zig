//! Scheduler execution-target device/layer ownership summary rendering.

const std = @import("std");

const progress_mod = @import("progress_pkg");
const topology = @import("../pipeline/local_pipeline_topology.zig");

pub const DeviceLayerSummary = struct {
    cpu_layers: usize = 0,
    gpu_stage_count: u8 = 0,
    gpu0_ordinal: usize = 0,
    gpu0_layers: usize = 0,
    gpu1_ordinal: usize = 1,
    gpu1_layers: usize = 0,
    metal_layers: usize = 0,

    pub fn totalLayers(self: DeviceLayerSummary) usize {
        return self.cpu_layers + self.gpu0_layers + self.gpu1_layers + self.metal_layers;
    }
};

const DeviceBarSegment = struct {
    layers: usize,
    color: []const u8,
};

fn fillSegmentWidths(segments: []const DeviceBarSegment, widths: []usize, total_layers: usize, bar_width: usize) void {
    @memset(widths, 0);
    if (total_layers == 0 or bar_width == 0) return;

    var remainders = [_]usize{0} ** 4;
    var used: usize = 0;
    for (segments, 0..) |segment, idx| {
        if (segment.layers == 0) continue;
        const scaled = segment.layers * bar_width;
        widths[idx] = scaled / total_layers;
        remainders[idx] = scaled % total_layers;
        used += widths[idx];
    }

    while (used < bar_width) {
        var best_idx: ?usize = null;
        var best_remainder: usize = 0;
        for (segments, 0..) |segment, idx| {
            if (segment.layers == 0) continue;
            if (best_idx == null or remainders[idx] > best_remainder) {
                best_idx = idx;
                best_remainder = remainders[idx];
            }
        }
        const idx = best_idx orelse break;
        widths[idx] += 1;
        remainders[idx] = 0;
        used += 1;
    }

    while (used > bar_width) {
        var largest_idx: ?usize = null;
        var largest_width: usize = 0;
        for (widths[0..segments.len], 0..) |width, idx| {
            if (width > largest_width) {
                largest_idx = idx;
                largest_width = width;
            }
        }
        const idx = largest_idx orelse break;
        if (widths[idx] == 0) break;
        widths[idx] -= 1;
        used -= 1;
    }
}

/// Renders a layer ownership summary across supported device classes.
/// CPU is yellow, CUDA GPU0 is cyan, CUDA GPU1 is green, and Metal is magenta.
/// Example output: `[#######...] 32/32  cpu: 10 · gpu0: 21 · gpu1: 1 · metal: 0`
pub fn render(
    buf: *[512]u8,
    summary: DeviceLayerSummary,
) ?[*:0]const u8 {
    const bar_width: usize = 40;
    const total_layers = summary.totalLayers();
    if (total_layers == 0) return null;

    const yellow = "\x1b[33m";
    const cyan = "\x1b[36m";
    const green = "\x1b[32m";
    const magenta = "\x1b[35m";
    const reset = "\x1b[0m";
    const segments = [_]DeviceBarSegment{
        .{ .layers = summary.cpu_layers, .color = yellow },
        .{ .layers = summary.gpu0_layers, .color = cyan },
        .{ .layers = summary.gpu1_layers, .color = green },
        .{ .layers = summary.metal_layers, .color = magenta },
    };
    var widths = [_]usize{0} ** segments.len;
    fillSegmentWidths(segments[0..], widths[0..], total_layers, bar_width);

    var stream = std.io.fixedBufferStream(buf);
    const w = stream.writer();

    w.writeAll("[") catch return null;
    for (segments, 0..) |segment, idx| {
        const width = widths[idx];
        if (width == 0) continue;
        w.writeAll(segment.color) catch return null;
        for (0..width) |_| w.writeByte('#') catch return null;
        w.writeAll(reset) catch return null;
    }

    w.print("] {d}/{d}  {s}cpu: {d}{s}", .{ total_layers, total_layers, yellow, summary.cpu_layers, reset }) catch return null;
    switch (summary.gpu_stage_count) {
        0 => w.print(" \xc2\xb7 {s}gpu: 0{s}", .{ cyan, reset }) catch return null,
        1 => w.print(" \xc2\xb7 {s}gpu{d}: {d}{s}", .{ cyan, summary.gpu0_ordinal, summary.gpu0_layers, reset }) catch return null,
        else => {
            w.print(" \xc2\xb7 {s}gpu{d}: {d}{s}", .{ cyan, summary.gpu0_ordinal, summary.gpu0_layers, reset }) catch return null;
            w.print(" \xc2\xb7 {s}gpu{d}: {d}{s}", .{ green, summary.gpu1_ordinal, summary.gpu1_layers, reset }) catch return null;
        },
    }
    w.print(" \xc2\xb7 {s}metal: {d}{s}", .{ magenta, summary.metal_layers, reset }) catch return null;

    const pos = stream.pos;
    if (pos >= buf.len) return null;
    buf[pos] = 0;
    return @ptrCast(buf[0..pos :0]);
}

pub fn publish(progress: progress_mod.Context, summary: DeviceLayerSummary) void {
    var bar_buf: [512]u8 = undefined;
    const msg = render(&bar_buf, summary) orelse return;
    progress.addLine(2, "Devices", 0, msg, null);
    progress.completeLine(2);
}

pub fn fromLocalStagePlan(plan: topology.LocalStagePlan) DeviceLayerSummary {
    var summary = DeviceLayerSummary{};
    for (plan.stagesSlice()) |stage| {
        const layer_count = stage.layer_end - stage.layer_start;
        switch (stage.backend_kind) {
            .cpu => summary.cpu_layers += layer_count,
            .metal => summary.metal_layers += layer_count,
            .cuda => {
                if (summary.gpu_stage_count == 0) {
                    summary.gpu0_ordinal = stage.device_ordinal orelse 0;
                    summary.gpu0_layers = layer_count;
                } else if (summary.gpu_stage_count == 1) {
                    summary.gpu1_ordinal = stage.device_ordinal orelse 0;
                    summary.gpu1_layers = layer_count;
                } else {
                    summary.gpu1_layers += layer_count;
                }
                summary.gpu_stage_count += 1;
            },
        }
    }
    return summary;
}

test "render reports all layers on one CUDA GPU" {
    var buf: [512]u8 = undefined;
    const rendered = render(&buf, .{
        .gpu_stage_count = 1,
        .gpu0_ordinal = 0,
        .gpu0_layers = 32,
    }) orelse return error.TestExpectedEqual;
    const text = std.mem.span(rendered);

    try std.testing.expect(std.mem.indexOf(u8, text, "] 32/32") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "cpu: 0") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "gpu0: 32") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "metal: 0") != null);
}

test "render reports CPU prefix and CUDA GPU stages" {
    var buf: [512]u8 = undefined;
    const rendered = render(&buf, .{
        .cpu_layers = 10,
        .gpu_stage_count = 2,
        .gpu0_ordinal = 0,
        .gpu0_layers = 21,
        .gpu1_ordinal = 1,
        .gpu1_layers = 1,
    }) orelse return error.TestExpectedEqual;
    const text = std.mem.span(rendered);

    try std.testing.expect(std.mem.indexOf(u8, text, "] 32/32") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "cpu: 10") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "gpu0: 21") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "gpu1: 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "metal: 0") != null);
}

test "render reports CPU and Metal backends with zero GPU class" {
    var cpu_buf: [512]u8 = undefined;
    const cpu_rendered = render(&cpu_buf, .{
        .cpu_layers = 32,
    }) orelse return error.TestExpectedEqual;
    const cpu_text = std.mem.span(cpu_rendered);
    try std.testing.expect(std.mem.indexOf(u8, cpu_text, "cpu: 32") != null);
    try std.testing.expect(std.mem.indexOf(u8, cpu_text, "gpu: 0") != null);
    try std.testing.expect(std.mem.indexOf(u8, cpu_text, "metal: 0") != null);

    var metal_buf: [512]u8 = undefined;
    const metal_rendered = render(&metal_buf, .{
        .metal_layers = 32,
    }) orelse return error.TestExpectedEqual;
    const metal_text = std.mem.span(metal_rendered);
    try std.testing.expect(std.mem.indexOf(u8, metal_text, "cpu: 0") != null);
    try std.testing.expect(std.mem.indexOf(u8, metal_text, "gpu: 0") != null);
    try std.testing.expect(std.mem.indexOf(u8, metal_text, "metal: 32") != null);
}

test "fromLocalStagePlan counts ordered CPU and CUDA stage layers" {
    const plan = try topology.localStagePlanFromSpecs(&.{
        .{
            .backend_kind = .cpu,
            .layer_start = 0,
            .layer_end = 3,
        },
        .{
            .backend_kind = .cuda,
            .device_ordinal = 1,
            .layer_start = 3,
            .layer_end = 8,
        },
        .{
            .backend_kind = .cuda,
            .device_ordinal = 2,
            .layer_start = 8,
            .layer_end = 10,
        },
    });
    const summary = fromLocalStagePlan(plan);

    try std.testing.expectEqual(@as(usize, 10), summary.totalLayers());
    try std.testing.expectEqual(@as(usize, 3), summary.cpu_layers);
    try std.testing.expectEqual(@as(u8, 2), summary.gpu_stage_count);
    try std.testing.expectEqual(@as(usize, 1), summary.gpu0_ordinal);
    try std.testing.expectEqual(@as(usize, 5), summary.gpu0_layers);
    try std.testing.expectEqual(@as(usize, 2), summary.gpu1_ordinal);
    try std.testing.expectEqual(@as(usize, 2), summary.gpu1_layers);
}
