//! Conformance checks for model-contract consumption in inference backends.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const main = @import("main");

const backend = main.inference.backend;
const models = main.models.dispatcher;
const has_metal = build_options.enable_metal and builtin.os.tag == .macos;
const has_cuda = build_options.enable_cuda and (builtin.os.tag == .linux or builtin.os.tag == .windows);

test "cpu backend consumes models-owned block contracts" {
    const runtime_blocks = models.weights.blocks;

    try std.testing.expect(backend.cpu.executor.weights.BlockType == models.op_types.BlockKind);
    try std.testing.expect(backend.cpu.executor.weights.BlockWeights == runtime_blocks.BlockWeights);
    try std.testing.expect(backend.cpu.executor.weights.BlockMapContext == runtime_blocks.BlockMapContext);
    try std.testing.expect(backend.cpu.executor.weights.WeightMap == runtime_blocks.WeightMap);
}

test "metal backend consumes models-owned block kind contract" {
    if (comptime !has_metal) return;
    try std.testing.expect(backend.metal.executor.weights.BlockType == models.op_types.BlockKind);
    try std.testing.expect(
        backend.metal.executor.weights.WeightHandles.LayerWeights.LayerKind == models.op_types.BlockKind,
    );
}

test "cuda backend consumes models-owned block contracts" {
    if (comptime !has_cuda) return;
    const runtime_blocks = models.weights.blocks;
    try std.testing.expect(backend.cuda.executor.weights.BlockType == models.op_types.BlockKind);
    try std.testing.expect(backend.cuda.executor.weights.BlockWeights == runtime_blocks.BlockWeights);
    try std.testing.expect(backend.cuda.executor.weights.BlockMapContext == runtime_blocks.BlockMapContext);
    try std.testing.expect(backend.cuda.executor.weights.WeightMap == runtime_blocks.WeightMap);
}

fn shouldSkipPath(path: []const u8) bool {
    if (std.mem.endsWith(u8, path, "/vision.zig")) return true;
    if (std.mem.indexOf(u8, path, "/vision/") != null) return true;
    return false;
}

test "non-vision inference backend files do not hardcode model-family string literals" {
    const forbidden_literals = [_][]const u8{
        "\"llama\"",
        "\"llama2\"",
        "\"llama3\"",
        "\"qwen\"",
        "\"granite\"",
        "\"gemma\"",
        "\"mistral\"",
        "\"phi\"",
        "\"gpt_oss\"",
        "\"lfm2\"",
        "\"youtu\"",
    };

    var dir = try std.fs.cwd().openDir("core/src/inference/backend", .{ .iterate = true });
    defer dir.close();

    var walker = try dir.walk(std.testing.allocator);
    defer walker.deinit();

    while (try walker.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.path, ".zig")) continue;
        if (shouldSkipPath(entry.path)) continue;

        const full_path = try std.fmt.allocPrint(std.testing.allocator, "core/src/inference/backend/{s}", .{entry.path});
        defer std.testing.allocator.free(full_path);

        const source = try std.fs.cwd().readFileAlloc(std.testing.allocator, full_path, 2 * 1024 * 1024);
        defer std.testing.allocator.free(source);

        for (forbidden_literals) |needle| {
            if (std.mem.indexOf(u8, source, needle) != null) {
                std.debug.print("forbidden model-family literal {s} in {s}\n", .{ needle, full_path });
                return error.ModelFamilyLiteralLeak;
            }
        }
    }
}
