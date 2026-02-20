//! Lightweight policy linter for core source layering rules.
//!
//! Usage:
//!   zig run core/tests/helpers/lint/root.zig -- core/src

const std = @import("std");

const IMPORT_PREFIX = "@import(";

fn lineNumberForOffset(source: []const u8, offset: usize) usize {
    return 1 + std.mem.count(u8, source[0..offset], "\n");
}

fn isComputePath(path: []const u8) bool {
    return std.mem.startsWith(u8, path, "core/src/compute/");
}

fn isOldTopLevelSimdOrQuantImport(target: []const u8) bool {
    if (std.mem.startsWith(u8, target, "../simd/") and !std.mem.startsWith(u8, target, "../simd/arch/")) return true;
    if (std.mem.startsWith(u8, target, "../../simd/") and !std.mem.startsWith(u8, target, "../../simd/arch/")) return true;
    if (std.mem.startsWith(u8, target, "../../../simd/") and !std.mem.startsWith(u8, target, "../../../simd/arch/")) return true;
    if (std.mem.startsWith(u8, target, "../quant/")) return true;
    if (std.mem.startsWith(u8, target, "../../quant/")) return true;
    if (std.mem.startsWith(u8, target, "../../../quant/")) return true;
    if (std.mem.startsWith(u8, target, "compute/simd/")) return true;
    if (std.mem.startsWith(u8, target, "compute/quant/")) return true;
    if (std.mem.indexOf(u8, target, "/compute/simd/") != null) return true;
    if (std.mem.indexOf(u8, target, "/compute/quant/") != null) return true;
    return false;
}

fn lintSource(file_path: []const u8, source: []const u8, emit: bool) usize {
    var violations: usize = 0;
    var search_from: usize = 0;

    while (std.mem.indexOfPos(u8, source, search_from, IMPORT_PREFIX)) |import_start| {
        var cursor = import_start + IMPORT_PREFIX.len;
        while (cursor < source.len and std.ascii.isWhitespace(source[cursor])) : (cursor += 1) {}
        if (cursor >= source.len or source[cursor] != '"') {
            search_from = import_start + 1;
            continue;
        }

        const target_start = cursor + 1;
        const target_end = std.mem.indexOfScalarPos(u8, source, target_start, '"') orelse break;
        const target = source[target_start..target_end];
        const line = lineNumberForOffset(source, target_start);

        if (isComputePath(file_path) and
            (std.mem.indexOf(u8, target, "inference/") != null or
                std.mem.indexOf(u8, target, "models/") != null))
        {
            violations += 1;
            if (emit) {
                std.debug.print("{s}:{d}: forbidden compute dependency import: \"{s}\"\n", .{ file_path, line, target });
            }
        }

        if (isOldTopLevelSimdOrQuantImport(target)) {
            violations += 1;
            if (emit) {
                std.debug.print("{s}:{d}: forbidden legacy compute import path: \"{s}\"\n", .{ file_path, line, target });
            }
        }

        search_from = target_end + 1;
    }

    return violations;
}

fn lintTree(allocator: std.mem.Allocator, root_path: []const u8) !usize {
    var total_violations: usize = 0;
    var dir = try std.fs.cwd().openDir(root_path, .{ .iterate = true });
    defer dir.close();

    var walker = try dir.walk(allocator);
    defer walker.deinit();

    while (try walker.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.path, ".zig")) continue;

        const full_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ root_path, entry.path });
        defer allocator.free(full_path);

        const source = try std.fs.cwd().readFileAlloc(allocator, full_path, 32 * 1024 * 1024);
        defer allocator.free(source);

        total_violations += lintSource(full_path, source, true);
    }

    return total_violations;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const root_path = if (args.len >= 2) args[1] else "core/src";
    const violations = try lintTree(allocator, root_path);
    if (violations != 0) {
        std.debug.print("lint: found {d} violation(s)\n", .{violations});
        return error.LintFailed;
    }
}

test "lintSource rejects inference import in compute" {
    const src =
        \\const bad = @import("../../inference/root.zig");
    ;
    try std.testing.expectEqual(@as(usize, 1), lintSource("core/src/compute/cpu/foo.zig", src, false));
}

test "lintSource rejects models import in compute" {
    const src =
        \\const bad = @import("../../models/root.zig");
    ;
    try std.testing.expectEqual(@as(usize, 1), lintSource("core/src/compute/cpu/bar.zig", src, false));
}

test "lintSource rejects legacy top-level simd import path" {
    const src =
        \\const simd = @import("../simd/root.zig");
    ;
    try std.testing.expectEqual(@as(usize, 1), lintSource("core/src/compute/cpu/reduction.zig", src, false));
}

test "lintSource allows new cpu simd arch path" {
    const src =
        \\const simd = @import("simd/arch/root.zig");
    ;
    try std.testing.expectEqual(@as(usize, 0), lintSource("core/src/compute/cpu/reduction.zig", src, false));
}
