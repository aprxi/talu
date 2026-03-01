//! Workspace path sandbox helpers for filesystem operations.
//!
//! All callers pass the canonical workspace root returned by
//! `canonicalizeWorkspace`. Requested paths can be relative or absolute.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const PathError = error{
    InvalidPath,
    PathOutsideWorkspace,
    ParentNotFound,
};

/// Canonicalize and validate a workspace root.
/// Caller owns the returned absolute path.
pub fn canonicalizeWorkspace(allocator: Allocator, workspace_dir: []const u8) ![]u8 {
    if (workspace_dir.len == 0) return PathError.InvalidPath;
    return std.fs.cwd().realpathAlloc(allocator, workspace_dir);
}

/// Resolve an existing path and ensure it remains inside the workspace root.
/// Caller owns the returned absolute canonical path.
pub fn resolveExistingPath(
    allocator: Allocator,
    workspace_dir: []const u8,
    requested_path: []const u8,
) ![]u8 {
    if (requested_path.len == 0) return PathError.InvalidPath;

    const candidate = try buildCandidatePath(allocator, workspace_dir, requested_path);
    defer allocator.free(candidate);

    const canonical = std.fs.cwd().realpathAlloc(allocator, candidate) catch |err| return err;
    errdefer allocator.free(canonical);

    if (!isWithinWorkspace(workspace_dir, canonical)) return PathError.PathOutsideWorkspace;
    return canonical;
}

/// Resolve a writable path and ensure the target is inside workspace root.
///
/// For existing targets this returns the canonical existing path.
/// For new targets this returns `<canonical_parent>/<basename>`.
/// Caller owns the returned absolute path.
pub fn resolveWritablePath(
    allocator: Allocator,
    workspace_dir: []const u8,
    requested_path: []const u8,
) ![]u8 {
    if (requested_path.len == 0) return PathError.InvalidPath;

    const candidate = try buildCandidatePath(allocator, workspace_dir, requested_path);
    defer allocator.free(candidate);

    const existing = std.fs.cwd().realpathAlloc(allocator, candidate) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    if (existing) |canonical_existing| {
        errdefer allocator.free(canonical_existing);
        if (!isWithinWorkspace(workspace_dir, canonical_existing)) return PathError.PathOutsideWorkspace;
        return canonical_existing;
    }

    const parent = std.fs.path.dirname(candidate) orelse return PathError.InvalidPath;
    const basename = std.fs.path.basename(candidate);
    if (basename.len == 0 or std.mem.eql(u8, basename, ".") or std.mem.eql(u8, basename, "..")) {
        return PathError.InvalidPath;
    }

    const canonical_parent = std.fs.cwd().realpathAlloc(allocator, parent) catch |err| switch (err) {
        error.FileNotFound => {
            // Parent does not exist yet. Use lexical normalization so callers can
            // create missing directories (e.g. mkdir=true), while still enforcing
            // workspace sandboxing for `..` traversal attempts.
            const normalized = try std.fs.path.resolve(allocator, &.{candidate});
            errdefer allocator.free(normalized);
            if (!isWithinWorkspace(workspace_dir, normalized)) return PathError.PathOutsideWorkspace;
            return normalized;
        },
        else => return err,
    };
    defer allocator.free(canonical_parent);

    if (!isWithinWorkspace(workspace_dir, canonical_parent)) return PathError.PathOutsideWorkspace;
    return std.fs.path.join(allocator, &.{ canonical_parent, basename });
}

/// Return true if `path` is equal to or a strict descendant of workspace root.
pub fn isWithinWorkspace(workspace_dir: []const u8, path: []const u8) bool {
    if (std.mem.eql(u8, workspace_dir, "/") or std.mem.eql(u8, workspace_dir, "\\")) {
        return std.fs.path.isAbsolute(path);
    }
    if (std.mem.eql(u8, workspace_dir, path)) return true;
    if (!std.mem.startsWith(u8, path, workspace_dir)) return false;
    if (path.len <= workspace_dir.len) return false;

    const separator = path[workspace_dir.len];
    return separator == '/' or separator == '\\';
}

/// Convert an absolute path to workspace-relative display form.
/// Caller owns the returned string.
pub fn toWorkspaceRelative(
    allocator: Allocator,
    workspace_dir: []const u8,
    absolute_path: []const u8,
) ![]u8 {
    if (std.mem.eql(u8, workspace_dir, "/") or std.mem.eql(u8, workspace_dir, "\\")) {
        if (absolute_path.len <= 1) return allocator.dupe(u8, ".");
        return allocator.dupe(u8, absolute_path[1..]);
    }
    if (!isWithinWorkspace(workspace_dir, absolute_path)) {
        return allocator.dupe(u8, absolute_path);
    }
    if (std.mem.eql(u8, workspace_dir, absolute_path)) {
        return allocator.dupe(u8, ".");
    }
    const rel_start = workspace_dir.len + 1;
    if (rel_start >= absolute_path.len) return allocator.dupe(u8, ".");
    return allocator.dupe(u8, absolute_path[rel_start..]);
}

fn buildCandidatePath(
    allocator: Allocator,
    workspace_dir: []const u8,
    requested_path: []const u8,
) ![]u8 {
    if (std.fs.path.isAbsolute(requested_path)) {
        return allocator.dupe(u8, requested_path);
    }
    return std.fs.path.join(allocator, &.{ workspace_dir, requested_path });
}

test "canonicalizeWorkspace rejects empty path" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(PathError.InvalidPath, canonicalizeWorkspace(allocator, ""));
}

test "resolveExistingPath allows in-workspace file and rejects outside" {
    const allocator = std.testing.allocator;

    var workspace_tmp = std.testing.tmpDir(.{});
    defer workspace_tmp.cleanup();
    const workspace = try workspace_tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(workspace);
    try workspace_tmp.dir.writeFile(.{ .sub_path = "inside.txt", .data = "ok" });

    const inside = try resolveExistingPath(allocator, workspace, "inside.txt");
    defer allocator.free(inside);
    try std.testing.expect(isWithinWorkspace(workspace, inside));

    var outside_tmp = std.testing.tmpDir(.{});
    defer outside_tmp.cleanup();
    const outside = try outside_tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(outside);
    const outside_file = try std.fs.path.join(allocator, &.{ outside, "outside.txt" });
    defer allocator.free(outside_file);
    try outside_tmp.dir.writeFile(.{ .sub_path = "outside.txt", .data = "nope" });

    try std.testing.expectError(
        PathError.PathOutsideWorkspace,
        resolveExistingPath(allocator, workspace, outside_file),
    );
}

test "resolveExistingPath returns FileNotFound for missing path" {
    const allocator = std.testing.allocator;

    var workspace_tmp = std.testing.tmpDir(.{});
    defer workspace_tmp.cleanup();
    const workspace = try workspace_tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(workspace);

    try std.testing.expectError(error.FileNotFound, resolveExistingPath(allocator, workspace, "missing.txt"));
}

test "resolveWritablePath resolves new file path inside workspace" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const workspace = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(workspace);
    try tmp.dir.makePath("nested");

    const resolved = try resolveWritablePath(allocator, workspace, "nested/file.txt");
    defer allocator.free(resolved);

    const expected = try std.fs.path.join(allocator, &.{ workspace, "nested", "file.txt" });
    defer allocator.free(expected);
    try std.testing.expectEqualStrings(expected, resolved);
}

test "resolveWritablePath allows missing parent directories within workspace" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const workspace = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(workspace);

    const resolved = try resolveWritablePath(allocator, workspace, "new/deep/file.txt");
    defer allocator.free(resolved);

    const expected = try std.fs.path.join(allocator, &.{ workspace, "new", "deep", "file.txt" });
    defer allocator.free(expected);
    try std.testing.expectEqualStrings(expected, resolved);
}

test "resolveWritablePath rejects lexical escapes for missing parent paths" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const workspace = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(workspace);

    try std.testing.expectError(
        PathError.PathOutsideWorkspace,
        resolveWritablePath(allocator, workspace, "../outside.txt"),
    );
}

test "isWithinWorkspace allows descendants and rejects outside" {
    try std.testing.expect(isWithinWorkspace("/tmp/work", "/tmp/work"));
    try std.testing.expect(isWithinWorkspace("/tmp/work", "/tmp/work/src/a.txt"));
    try std.testing.expect(!isWithinWorkspace("/tmp/work", "/tmp/workspace/file.txt"));
    try std.testing.expect(!isWithinWorkspace("/tmp/work", "/tmp/other/file.txt"));
}

test "toWorkspaceRelative converts descendants and root" {
    const allocator = std.testing.allocator;

    const rel_child = try toWorkspaceRelative(allocator, "/tmp/work", "/tmp/work/src/main.zig");
    defer allocator.free(rel_child);
    try std.testing.expectEqualStrings("src/main.zig", rel_child);

    const rel_root = try toWorkspaceRelative(allocator, "/tmp/work", "/tmp/work");
    defer allocator.free(rel_root);
    try std.testing.expectEqualStrings(".", rel_root);
}
