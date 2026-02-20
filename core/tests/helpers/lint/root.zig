//! Lightweight policy linter for core source layering rules.
//!
//! Usage:
//!   zig run core/tests/helpers/lint/root.zig -- core/src

const std = @import("std");

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

fn extractImportTarget(
    allocator: std.mem.Allocator,
    tree: std.zig.Ast,
    arg_node: std.zig.Ast.Node.Index,
) !?[]u8 {
    return switch (tree.nodeTag(arg_node)) {
        .string_literal => blk: {
            const tok = tree.firstToken(arg_node);
            const raw = tree.tokenSlice(tok);
            break :blk try std.zig.string_literal.parseAlloc(allocator, raw);
        },
        else => null,
    };
}

fn lintSource(allocator: std.mem.Allocator, file_path: []const u8, source: []const u8, emit: bool) !usize {
    const source_z = try allocator.dupeZ(u8, source);
    defer allocator.free(source_z);
    var tree = try std.zig.Ast.parse(allocator, source_z, .zig);
    defer tree.deinit(allocator);

    var violations: usize = 0;

    var builtin_params_buf: [2]std.zig.Ast.Node.Index = undefined;
    for (0..tree.nodes.len) |node_idx_raw| {
        const node: std.zig.Ast.Node.Index = @enumFromInt(node_idx_raw);
        const tag = tree.nodeTag(node);
        switch (tag) {
            .builtin_call,
            .builtin_call_comma,
            .builtin_call_two,
            .builtin_call_two_comma,
            => {},
            else => continue,
        }

        const main_tok = tree.nodeMainToken(node);
        if (!std.mem.eql(u8, tree.tokenSlice(main_tok), "@import")) continue;

        const params = tree.builtinCallParams(&builtin_params_buf, node) orelse continue;
        if (params.len == 0) continue;

        const arg_node = params[0];
        const target_owned = try extractImportTarget(allocator, tree, arg_node);
        defer if (target_owned) |target| allocator.free(target);
        const target = target_owned orelse continue;

        const line = lineNumberForOffset(source, tree.tokenStart(tree.firstToken(arg_node)));
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

        total_violations += try lintSource(allocator, full_path, source, true);
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
    try std.testing.expectEqual(@as(usize, 1), try lintSource(std.testing.allocator, "core/src/compute/cpu/foo.zig", src, false));
}

test "lintSource rejects models import in compute" {
    const src =
        \\const bad = @import("../../models/root.zig");
    ;
    try std.testing.expectEqual(@as(usize, 1), try lintSource(std.testing.allocator, "core/src/compute/cpu/bar.zig", src, false));
}

test "lintSource rejects legacy top-level simd import path" {
    const src =
        \\const simd = @import("../simd/root.zig");
    ;
    try std.testing.expectEqual(@as(usize, 1), try lintSource(std.testing.allocator, "core/src/compute/cpu/reduction.zig", src, false));
}

test "lintSource allows new cpu simd arch path" {
    const src =
        \\const simd = @import("simd/arch/root.zig");
    ;
    try std.testing.expectEqual(@as(usize, 0), try lintSource(std.testing.allocator, "core/src/compute/cpu/reduction.zig", src, false));
}
