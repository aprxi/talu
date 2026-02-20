//! Preprocessor Configuration
//!
//! Loads image preprocessing parameters from preprocessor_config.json.
//! This file specifies pixel limits for smart resize — the model author's
//! intended resolution range.  Downstream code clamps these raw values to
//! the vision encoder's hardware limit (see Backend.visionMaxPixels).

const std = @import("std");
const json = @import("../../io/json/root.zig");
const log = @import("../../log.zig");

// =============================================================================
// Preprocessor Config
// =============================================================================

/// Image preprocessing config loaded from preprocessor_config.json.
///
/// Stores the raw values from the config file.  CPU safety clamping is
/// applied downstream in buildVisionPreprocessOptions, not here.
pub const PreprocessorConfig = struct {
    /// Minimum total pixel count for smart resize (0 = no minimum).
    min_pixels: u64 = 0,
    /// Maximum total pixel count for smart resize (0 = not specified).
    max_pixels: u64 = 0,
    /// patch_size from preprocessor_config (0 = not specified).
    /// Used for cross-check against config.json only.
    patch_size: u32 = 0,
};

/// Load preprocessor config from a model directory.
///
/// Soft-fails on missing or invalid files — returns default (zero) config.
/// The caller never needs to handle errors from this function.
pub fn loadPreprocessorConfig(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
) PreprocessorConfig {
    return loadImpl(allocator, model_dir);
}

fn loadImpl(allocator: std.mem.Allocator, model_dir: []const u8) PreprocessorConfig {
    const config_path = std.fs.path.join(allocator, &.{ model_dir, "preprocessor_config.json" }) catch return .{};
    defer allocator.free(config_path);

    const config_bytes = std.fs.cwd().readFileAlloc(allocator, config_path, 256 * 1024) catch |err| {
        if (err == error.FileNotFound or err == error.NotDir) {
            log.info("load", "No preprocessor_config.json found, using defaults", .{});
        } else {
            log.warn("load", "Failed to read preprocessor_config.json", .{ .@"error" = @errorName(err) });
        }
        return .{};
    };
    defer allocator.free(config_bytes);

    const parsed = json.parseValue(allocator, config_bytes, .{ .max_size_bytes = 256 * 1024 }) catch |err| {
        log.warn("load", "Invalid JSON in preprocessor_config.json", .{ .@"error" = @errorName(err) });
        return .{};
    };
    defer parsed.deinit();

    const root = parsed.value.object;

    return .{
        .max_pixels = extractMaxPixels(root),
        .min_pixels = extractMinPixels(root),
        .patch_size = extractPatchSize(root),
    };
}

// =============================================================================
// Field Extraction
// =============================================================================

/// Extract max_pixels with priority:
///   1. root "max_pixels" (Qwen2-VL)
///   2. root "size"."longest_edge" (Qwen3-VL — area)
///   3. root "size"."height" × "size"."width" (LFM2-VL)
///   4. 0 (not specified)
fn extractMaxPixels(root: std.json.ObjectMap) u64 {
    // (1) Direct max_pixels field
    if (getU64(root, "max_pixels")) |v| return v;

    // (2) & (3) From nested "size" object
    if (getSizeObject(root)) |size| {
        // (2) longest_edge (Qwen3-VL — pixel-count area)
        if (getU64(size, "longest_edge")) |v| return v;

        // (3) height × width (LFM2-VL — tile dimensions)
        const h = getU64(size, "height") orelse return 0;
        const w = getU64(size, "width") orelse return 0;
        return h * w;
    }

    return 0;
}

/// Extract min_pixels with priority:
///   1. root "min_pixels" (Qwen2-VL)
///   2. root "size"."shortest_edge" (Qwen3-VL — area)
///   3. 0 (not specified)
fn extractMinPixels(root: std.json.ObjectMap) u64 {
    // (1) Direct min_pixels field
    if (getU64(root, "min_pixels")) |v| return v;

    // (2) From nested "size" object
    if (getSizeObject(root)) |size| {
        if (getU64(size, "shortest_edge")) |v| return v;
    }

    return 0;
}

fn extractPatchSize(root: std.json.ObjectMap) u32 {
    const val = root.get("patch_size") orelse return 0;
    return switch (val) {
        .integer => |i| if (i > 0) @intCast(i) else 0,
        else => 0,
    };
}

// =============================================================================
// JSON Helpers
// =============================================================================

fn getSizeObject(root: std.json.ObjectMap) ?std.json.ObjectMap {
    const val = root.get("size") orelse return null;
    return switch (val) {
        .object => |o| o,
        else => null,
    };
}

/// Get a u64 from a JSON integer field (ignores negative values).
fn getU64(obj: std.json.ObjectMap, key: []const u8) ?u64 {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .integer => |i| if (i >= 0) @intCast(i) else null,
        .float => |f| if (f >= 0 and f <= @as(f64, @floatFromInt(std.math.maxInt(u64)))) @intFromFloat(f) else null,
        else => null,
    };
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

fn writeTmpConfig(tmp_dir: std.testing.TmpDir, content: []const u8) void {
    tmp_dir.dir.writeFile(.{
        .sub_path = "preprocessor_config.json",
        .data = content,
    }) catch unreachable;
}

fn tmpDirPath(tmp_dir: std.testing.TmpDir, buf: *[std.fs.max_path_bytes]u8) []const u8 {
    return tmp_dir.dir.realpath(".", buf) catch unreachable;
}

// Test 1: LFM2-VL schema — size.height × size.width
test "loadPreprocessorConfig: LFM2-VL schema" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    writeTmpConfig(tmp, @embedFile("testdata/preproc_lfm2.json"));
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const result = loadPreprocessorConfig(testing.allocator, tmpDirPath(tmp, &buf));
    try testing.expectEqual(@as(u64, 512 * 512), result.max_pixels);
    try testing.expectEqual(@as(u64, 0), result.min_pixels);
}

// Test 2: Qwen3-VL schema — size.longest_edge / size.shortest_edge
test "loadPreprocessorConfig: Qwen3-VL schema" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    writeTmpConfig(tmp, @embedFile("testdata/preproc_qwen3.json"));
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const result = loadPreprocessorConfig(testing.allocator, tmpDirPath(tmp, &buf));
    try testing.expectEqual(@as(u64, 16777216), result.max_pixels);
    try testing.expectEqual(@as(u64, 65536), result.min_pixels);
}

// Test 3: Qwen2-VL schema — root min_pixels / max_pixels
test "loadPreprocessorConfig: Qwen2-VL schema" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    writeTmpConfig(tmp, @embedFile("testdata/preproc_qwen2.json"));
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const result = loadPreprocessorConfig(testing.allocator, tmpDirPath(tmp, &buf));
    try testing.expectEqual(@as(u64, 12845056), result.max_pixels);
    try testing.expectEqual(@as(u64, 3136), result.min_pixels);
}

// Test 4: longest_edge treated as area directly
test "loadPreprocessorConfig: longest_edge treated as area" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    writeTmpConfig(tmp,
        \\{"size":{"longest_edge":1000000}}
    );
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const result = loadPreprocessorConfig(testing.allocator, tmpDirPath(tmp, &buf));
    try testing.expectEqual(@as(u64, 1000000), result.max_pixels);
}

// Test 5: Missing file returns defaults
test "loadPreprocessorConfig: missing file returns defaults" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const result = loadPreprocessorConfig(testing.allocator, tmpDirPath(tmp, &buf));
    try testing.expectEqual(@as(u64, 0), result.max_pixels);
    try testing.expectEqual(@as(u64, 0), result.min_pixels);
    try testing.expectEqual(@as(u32, 0), result.patch_size);
}

// Test 6: Bad JSON returns defaults
test "loadPreprocessorConfig: bad JSON returns defaults" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    writeTmpConfig(tmp, "{{{");
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const result = loadPreprocessorConfig(testing.allocator, tmpDirPath(tmp, &buf));
    try testing.expectEqual(@as(u64, 0), result.max_pixels);
    try testing.expectEqual(@as(u64, 0), result.min_pixels);
}

// Test 7: Partial fields — only max_pixels
test "loadPreprocessorConfig: partial fields" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    writeTmpConfig(tmp,
        \\{"max_pixels":1000000}
    );
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const result = loadPreprocessorConfig(testing.allocator, tmpDirPath(tmp, &buf));
    try testing.expectEqual(@as(u64, 1000000), result.max_pixels);
    try testing.expectEqual(@as(u64, 0), result.min_pixels);
}

// Test 8: patch_size extraction
test "loadPreprocessorConfig: patch_size extraction" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    writeTmpConfig(tmp,
        \\{"patch_size":14}
    );
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const result = loadPreprocessorConfig(testing.allocator, tmpDirPath(tmp, &buf));
    try testing.expectEqual(@as(u32, 14), result.patch_size);
}
