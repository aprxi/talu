//! Typed filesystem operations for agent runtime features.
//!
//! Inputs are already-resolved workspace-safe paths. Callers own path
//! validation/sandboxing and only use this module for file operations.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const OperationError = error{
    InvalidArgument,
    NonUtf8,
    NoMatches,
    MultipleMatches,
};

pub const ReadResult = struct {
    /// Caller owns this UTF-8 buffer.
    content: []u8,
    /// Total bytes returned in `content`.
    size: u64,
    /// Reserved for future partial-read mode.
    truncated: bool,
};

pub const WriteResult = struct {
    bytes_written: usize,
};

pub const EditResult = struct {
    replacements: usize,
    bytes_written: usize,
};

pub const StatResult = struct {
    exists: bool,
    is_file: bool,
    is_dir: bool,
    is_symlink: bool,
    size: u64,
    mode: u32,
    modified_at: i64,
    created_at: i64,
};

pub const ListEntry = struct {
    /// Entry path relative to the listed root.
    path: []u8,
    name: []u8,
    is_dir: bool,
    is_symlink: bool,
    size: u64,
    modified_at: i64,
};

pub const ListResult = struct {
    entries: []ListEntry,
    truncated: bool,

    pub fn deinit(self: *ListResult, allocator: Allocator) void {
        for (self.entries) |entry| {
            allocator.free(entry.path);
            allocator.free(entry.name);
        }
        allocator.free(self.entries);
        self.* = undefined;
    }
};

/// Read a UTF-8 file up to `max_bytes`.
/// Returns `error.FileTooBig` when the file exceeds this limit.
/// Caller owns `ReadResult.content`.
pub fn readFile(allocator: Allocator, path: []const u8, max_bytes: usize) !ReadResult {
    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, max_bytes);
    errdefer allocator.free(bytes);

    if (!std.unicode.utf8ValidateSlice(bytes)) return OperationError.NonUtf8;

    return .{
        .content = bytes,
        .size = bytes.len,
        .truncated = false,
    };
}

/// Overwrite file content with UTF-8 text.
pub fn writeFile(path: []const u8, content: []const u8) !WriteResult {
    const file = try std.fs.cwd().createFile(path, .{
        .truncate = true,
        .read = false,
        .exclusive = false,
    });
    defer file.close();

    try file.writeAll(content);
    return .{ .bytes_written = content.len };
}

/// Replace `old_text` with `new_text` in a UTF-8 file.
///
/// When `replace_all` is false, exactly one match is required.
pub fn editFile(
    allocator: Allocator,
    path: []const u8,
    old_text: []const u8,
    new_text: []const u8,
    replace_all: bool,
    max_read_bytes: usize,
) !EditResult {
    if (old_text.len == 0) return OperationError.InvalidArgument;

    const original = try std.fs.cwd().readFileAlloc(allocator, path, max_read_bytes);
    defer allocator.free(original);

    if (!std.unicode.utf8ValidateSlice(original)) return OperationError.NonUtf8;

    const replacements = countOccurrences(original, old_text);
    if (replacements == 0) return OperationError.NoMatches;
    if (!replace_all and replacements != 1) return OperationError.MultipleMatches;

    const updated = if (replace_all)
        try replaceAll(allocator, original, old_text, new_text)
    else
        try replaceFirst(allocator, original, old_text, new_text);
    defer allocator.free(updated);

    const file = try std.fs.cwd().createFile(path, .{
        .truncate = true,
        .read = false,
        .exclusive = false,
    });
    defer file.close();
    try file.writeAll(updated);

    return .{
        .replacements = if (replace_all) replacements else @as(usize, 1),
        .bytes_written = updated.len,
    };
}

/// Return metadata for a filesystem path.
///
/// `exists=false` is returned for missing paths.
pub fn stat(path: []const u8) !StatResult {
    const metadata = std.fs.cwd().statFile(path) catch |err| switch (err) {
        error.FileNotFound => {
            return .{
                .exists = false,
                .is_file = false,
                .is_dir = false,
                .is_symlink = false,
                .size = 0,
                .mode = 0,
                .modified_at = 0,
                .created_at = 0,
            };
        },
        else => return err,
    };

    return .{
        .exists = true,
        .is_file = metadata.kind == .file,
        .is_dir = metadata.kind == .directory,
        .is_symlink = metadata.kind == .sym_link,
        .size = metadata.size,
        .mode = @intCast(metadata.mode),
        .modified_at = clampI128ToI64(metadata.mtime),
        .created_at = clampI128ToI64(metadata.ctime),
    };
}

/// List directory entries with optional glob filtering.
///
/// `glob` supports `*` and `?`.
pub fn listDir(
    allocator: Allocator,
    path: []const u8,
    glob: ?[]const u8,
    recursive: bool,
    limit: usize,
) !ListResult {
    var out = std.ArrayList(ListEntry).empty;
    errdefer {
        for (out.items) |entry| {
            allocator.free(entry.path);
            allocator.free(entry.name);
        }
        out.deinit(allocator);
    }

    var truncated = false;
    var dir = try std.fs.cwd().openDir(path, .{ .iterate = true });
    defer dir.close();

    if (recursive) {
        var walker = try dir.walk(allocator);
        defer walker.deinit();

        while (try walker.next()) |entry| {
            if (limit != 0 and out.items.len >= limit) {
                truncated = true;
                break;
            }
            if (glob) |pattern| {
                if (!globMatch(pattern, entry.basename)) continue;
            }

            const st = entry.dir.statFile(entry.basename) catch continue;
            try out.append(allocator, .{
                .path = try allocator.dupe(u8, entry.path),
                .name = try allocator.dupe(u8, entry.basename),
                .is_dir = entry.kind == .directory,
                .is_symlink = entry.kind == .sym_link,
                .size = st.size,
                .modified_at = clampI128ToI64(st.mtime),
            });
        }
    } else {
        var iter = dir.iterate();
        while (try iter.next()) |entry| {
            if (limit != 0 and out.items.len >= limit) {
                truncated = true;
                break;
            }
            if (glob) |pattern| {
                if (!globMatch(pattern, entry.name)) continue;
            }

            const st = dir.statFile(entry.name) catch continue;
            try out.append(allocator, .{
                .path = try allocator.dupe(u8, entry.name),
                .name = try allocator.dupe(u8, entry.name),
                .is_dir = entry.kind == .directory,
                .is_symlink = entry.kind == .sym_link,
                .size = st.size,
                .modified_at = clampI128ToI64(st.mtime),
            });
        }
    }

    return .{
        .entries = try out.toOwnedSlice(allocator),
        .truncated = truncated,
    };
}

/// Remove a file or directory.
pub fn remove(path: []const u8, recursive: bool) !void {
    const metadata = try std.fs.cwd().statFile(path);
    if (metadata.kind == .directory) {
        if (recursive) {
            try std.fs.cwd().deleteTree(path);
        } else {
            try std.fs.cwd().deleteDir(path);
        }
        return;
    }
    try std.fs.cwd().deleteFile(path);
}

/// Create a directory.
pub fn makeDir(path: []const u8, recursive: bool) !void {
    if (recursive) {
        try std.fs.cwd().makePath(path);
    } else {
        try std.fs.cwd().makeDir(path);
    }
}

/// Rename or move a path.
pub fn rename(from: []const u8, to: []const u8) !void {
    try std.fs.cwd().rename(from, to);
}

fn globMatch(pattern: []const u8, text: []const u8) bool {
    var pattern_idx: usize = 0;
    var text_idx: usize = 0;
    var star_idx: ?usize = null;
    var backtrack_text_idx: usize = 0;

    while (text_idx < text.len) {
        if (pattern_idx < pattern.len and (pattern[pattern_idx] == '?' or pattern[pattern_idx] == text[text_idx])) {
            pattern_idx += 1;
            text_idx += 1;
            continue;
        }
        if (pattern_idx < pattern.len and pattern[pattern_idx] == '*') {
            star_idx = pattern_idx;
            pattern_idx += 1;
            backtrack_text_idx = text_idx;
            continue;
        }
        if (star_idx) |star| {
            pattern_idx = star + 1;
            backtrack_text_idx += 1;
            text_idx = backtrack_text_idx;
            continue;
        }
        return false;
    }

    while (pattern_idx < pattern.len and pattern[pattern_idx] == '*') {
        pattern_idx += 1;
    }
    return pattern_idx == pattern.len;
}

fn clampI128ToI64(value: i128) i64 {
    if (value > std.math.maxInt(i64)) return std.math.maxInt(i64);
    if (value < std.math.minInt(i64)) return std.math.minInt(i64);
    return @intCast(value);
}

fn countOccurrences(haystack: []const u8, needle: []const u8) usize {
    if (needle.len == 0) return 0;

    var count: usize = 0;
    var start: usize = 0;
    while (std.mem.indexOfPos(u8, haystack, start, needle)) |idx| {
        count += 1;
        start = idx + needle.len;
    }
    return count;
}

fn replaceFirst(
    allocator: Allocator,
    haystack: []const u8,
    needle: []const u8,
    replacement: []const u8,
) ![]u8 {
    const first_idx = std.mem.indexOf(u8, haystack, needle) orelse return allocator.dupe(u8, haystack);

    const new_len = haystack.len - needle.len + replacement.len;
    const out = try allocator.alloc(u8, new_len);

    var cursor: usize = 0;
    @memcpy(out[cursor..][0..first_idx], haystack[0..first_idx]);
    cursor += first_idx;

    @memcpy(out[cursor..][0..replacement.len], replacement);
    cursor += replacement.len;

    const tail_start = first_idx + needle.len;
    @memcpy(out[cursor..], haystack[tail_start..]);
    return out;
}

fn replaceAll(
    allocator: Allocator,
    haystack: []const u8,
    needle: []const u8,
    replacement: []const u8,
) ![]u8 {
    const occurrences = countOccurrences(haystack, needle);
    if (occurrences == 0) return allocator.dupe(u8, haystack);

    const shrink = occurrences * needle.len;
    const grow = occurrences * replacement.len;
    const new_len = haystack.len - shrink + grow;
    const out = try allocator.alloc(u8, new_len);

    var read_idx: usize = 0;
    var write_idx: usize = 0;
    while (std.mem.indexOfPos(u8, haystack, read_idx, needle)) |idx| {
        const chunk = haystack[read_idx..idx];
        @memcpy(out[write_idx..][0..chunk.len], chunk);
        write_idx += chunk.len;

        @memcpy(out[write_idx..][0..replacement.len], replacement);
        write_idx += replacement.len;

        read_idx = idx + needle.len;
    }

    const tail = haystack[read_idx..];
    @memcpy(out[write_idx..][0..tail.len], tail);
    write_idx += tail.len;

    std.debug.assert(write_idx == new_len);
    return out;
}

test "readFile returns UTF-8 content" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(.{ .sub_path = "note.txt", .data = "hello world" });

    const path = try tmp.dir.realpathAlloc(allocator, "note.txt");
    defer allocator.free(path);

    const result = try readFile(allocator, path, 1024);
    defer allocator.free(result.content);

    try std.testing.expectEqualStrings("hello world", result.content);
    try std.testing.expectEqual(@as(u64, 11), result.size);
    try std.testing.expect(!result.truncated);
}

test "readFile returns NonUtf8 for non-UTF8 bytes" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const binary = [_]u8{ 0xff, 0xfe, 0xfd };
    try tmp.dir.writeFile(.{ .sub_path = "binary.bin", .data = &binary });

    const path = try tmp.dir.realpathAlloc(allocator, "binary.bin");
    defer allocator.free(path);

    try std.testing.expectError(OperationError.NonUtf8, readFile(allocator, path, 1024));
}

test "writeFile writes all bytes" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const workspace = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(workspace);
    const path = try std.fs.path.join(allocator, &.{ workspace, "write.txt" });
    defer allocator.free(path);

    const result = try writeFile(path, "abc123");
    try std.testing.expectEqual(@as(usize, 6), result.bytes_written);

    const written = try tmp.dir.readFileAlloc(allocator, "write.txt", 64);
    defer allocator.free(written);
    try std.testing.expectEqualStrings("abc123", written);
}

test "editFile replaces all matches when requested" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(.{ .sub_path = "edit.txt", .data = "one two one" });
    const path = try tmp.dir.realpathAlloc(allocator, "edit.txt");
    defer allocator.free(path);

    const result = try editFile(allocator, path, "one", "1", true, 1024);
    try std.testing.expectEqual(@as(usize, 2), result.replacements);
    try std.testing.expectEqual(@as(usize, 7), result.bytes_written);

    const edited = try tmp.dir.readFileAlloc(allocator, "edit.txt", 64);
    defer allocator.free(edited);
    try std.testing.expectEqualStrings("1 two 1", edited);
}

test "editFile enforces single-match mode" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(.{ .sub_path = "edit.txt", .data = "dup dup" });
    const path = try tmp.dir.realpathAlloc(allocator, "edit.txt");
    defer allocator.free(path);

    try std.testing.expectError(
        OperationError.MultipleMatches,
        editFile(allocator, path, "dup", "x", false, 1024),
    );
}

test "stat returns missing path when file does not exist" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const workspace = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(workspace);
    const path = try std.fs.path.join(allocator, &.{ workspace, "missing.txt" });
    defer allocator.free(path);

    const result = try stat(path);
    try std.testing.expect(!result.exists);
}

test "stat returns metadata for existing file" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(.{ .sub_path = "present.txt", .data = "abc" });
    const path = try tmp.dir.realpathAlloc(allocator, "present.txt");
    defer allocator.free(path);

    const result = try stat(path);
    try std.testing.expect(result.exists);
    try std.testing.expect(result.is_file);
    try std.testing.expect(!result.is_dir);
    try std.testing.expectEqual(@as(u64, 3), result.size);
}

test "listDir lists files with glob and limit" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(.{ .sub_path = "a.txt", .data = "a" });
    try tmp.dir.writeFile(.{ .sub_path = "b.md", .data = "b" });
    try tmp.dir.writeFile(.{ .sub_path = "c.txt", .data = "c" });
    const root = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root);

    var result = try listDir(allocator, root, "*.txt", false, 1);
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), result.entries.len);
    try std.testing.expect(result.truncated);
    try std.testing.expect(std.mem.endsWith(u8, result.entries[0].name, ".txt"));
}

test "remove deletes files and directories" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(.{ .sub_path = "delete.txt", .data = "x" });
    try tmp.dir.makePath("nested/dir");
    try tmp.dir.writeFile(.{ .sub_path = "nested/dir/file.txt", .data = "x" });
    const root = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root);

    const file_path = try std.fs.path.join(allocator, &.{ root, "delete.txt" });
    defer allocator.free(file_path);
    try remove(file_path, false);
    try std.testing.expectError(error.FileNotFound, std.fs.cwd().statFile(file_path));

    const dir_path = try std.fs.path.join(allocator, &.{ root, "nested" });
    defer allocator.free(dir_path);
    try remove(dir_path, true);
    try std.testing.expectError(error.FileNotFound, std.fs.cwd().statFile(dir_path));
}

test "makeDir and rename create and move paths" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root);

    const src = try std.fs.path.join(allocator, &.{ root, "src" });
    defer allocator.free(src);
    try makeDir(src, false);
    try std.testing.expect((try std.fs.cwd().statFile(src)).kind == .directory);

    const dst = try std.fs.path.join(allocator, &.{ root, "renamed" });
    defer allocator.free(dst);
    try rename(src, dst);
    try std.testing.expect((try std.fs.cwd().statFile(dst)).kind == .directory);
}
