//! Built-in file tools for agent execution.
//!
//! Tools:
//! - `read_file`
//! - `write_file`
//! - `edit_file`
//!
//! All paths are sandboxed to the configured workspace directory.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Tool = @import("../tool.zig").Tool;
const ToolResult = @import("../tool.zig").ToolResult;

const default_read_max_bytes: usize = 256 * 1024;

const ToolError = error{
    InvalidArguments,
    InvalidPath,
    PathOutsideWorkspace,
    ParentNotFound,
};

pub const FileReadTool = struct {
    allocator: Allocator,
    workspace_dir: []const u8,
    max_read_bytes: usize,

    pub fn init(allocator: Allocator, workspace_dir: []const u8, max_read_bytes: usize) !*FileReadTool {
        const self = try allocator.create(FileReadTool);
        errdefer allocator.destroy(self);

        const canonical_workspace = try canonicalizeWorkspace(allocator, workspace_dir);
        errdefer allocator.free(canonical_workspace);

        self.* = .{
            .allocator = allocator,
            .workspace_dir = canonical_workspace,
            .max_read_bytes = if (max_read_bytes == 0) default_read_max_bytes else max_read_bytes,
        };
        return self;
    }

    pub fn deinit(self: *FileReadTool) void {
        self.allocator.free(self.workspace_dir);
        self.allocator.destroy(self);
    }

    pub fn asTool(self: *FileReadTool) Tool {
        return Tool.init(self, &vtable);
    }

    fn name(_: *anyopaque) []const u8 {
        return "read_file";
    }

    fn description(_: *anyopaque) []const u8 {
        return "Read a UTF-8 text file from the workspace sandbox.";
    }

    fn parametersSchema(_: *anyopaque) []const u8 {
        return 
        \\{"type":"object","properties":{"path":{"type":"string","description":"Path to file, relative to workspace or absolute inside workspace."},"max_bytes":{"type":"integer","description":"Optional read limit in bytes."}},"required":["path"],"additionalProperties":false}
        ;
    }

    fn executeFn(ctx: *anyopaque, allocator: Allocator, arguments_json: []const u8) anyerror!ToolResult {
        const self: *FileReadTool = @ptrCast(@alignCast(ctx));
        return self.execute(allocator, arguments_json);
    }

    fn execute(self: *FileReadTool, allocator: Allocator, arguments_json: []const u8) !ToolResult {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, arguments_json, .{}) catch {
            return toolError(allocator, "Invalid JSON arguments.");
        };
        defer parsed.deinit();

        const obj = switch (parsed.value) {
            .object => |o| o,
            else => return toolError(allocator, "Arguments must be a JSON object."),
        };

        const path_value = obj.get("path") orelse return toolError(allocator, "Missing required field: path.");
        const path = switch (path_value) {
            .string => |s| s,
            else => return toolError(allocator, "Field 'path' must be a string."),
        };

        var max_bytes = self.max_read_bytes;
        if (obj.get("max_bytes")) |max_value| {
            switch (max_value) {
                .integer => |n| {
                    if (n <= 0) return toolError(allocator, "Field 'max_bytes' must be > 0.");
                    max_bytes = @intCast(n);
                },
                else => return toolError(allocator, "Field 'max_bytes' must be an integer."),
            }
        }

        const resolved = resolveExistingPath(allocator, self.workspace_dir, path) catch |err| {
            return toolErrorFmt(allocator, "read_file path error: {s}", .{@errorName(err)});
        };
        defer allocator.free(resolved);

        const bytes = std.fs.cwd().readFileAlloc(allocator, resolved, max_bytes) catch |err| switch (err) {
            error.FileTooBig => return toolErrorFmt(allocator, "File exceeds max_bytes ({d}).", .{max_bytes}),
            else => return toolErrorFmt(allocator, "read_file failed: {s}", .{@errorName(err)}),
        };
        defer allocator.free(bytes);

        if (!std.unicode.utf8ValidateSlice(bytes)) {
            return toolError(allocator, "read_file supports UTF-8 text files only.");
        }

        const rel = try toWorkspaceRelative(allocator, self.workspace_dir, resolved);
        defer allocator.free(rel);

        return toolSuccessJson(allocator, .{
            .path = rel,
            .content = bytes,
            .bytes = bytes.len,
        });
    }

    const vtable = Tool.VTable{
        .name = name,
        .description = description,
        .parametersSchema = parametersSchema,
        .execute = executeFn,
        .deinit = deinitOpaque,
    };

    fn deinitOpaque(ctx: *anyopaque) void {
        const self: *FileReadTool = @ptrCast(@alignCast(ctx));
        self.deinit();
    }
};

pub const FileWriteTool = struct {
    allocator: Allocator,
    workspace_dir: []const u8,

    pub fn init(allocator: Allocator, workspace_dir: []const u8) !*FileWriteTool {
        const self = try allocator.create(FileWriteTool);
        errdefer allocator.destroy(self);

        const canonical_workspace = try canonicalizeWorkspace(allocator, workspace_dir);
        errdefer allocator.free(canonical_workspace);

        self.* = .{
            .allocator = allocator,
            .workspace_dir = canonical_workspace,
        };
        return self;
    }

    pub fn deinit(self: *FileWriteTool) void {
        self.allocator.free(self.workspace_dir);
        self.allocator.destroy(self);
    }

    pub fn asTool(self: *FileWriteTool) Tool {
        return Tool.init(self, &vtable);
    }

    fn name(_: *anyopaque) []const u8 {
        return "write_file";
    }

    fn description(_: *anyopaque) []const u8 {
        return "Write UTF-8 text content to a file inside the workspace sandbox.";
    }

    fn parametersSchema(_: *anyopaque) []const u8 {
        return 
        \\{"type":"object","properties":{"path":{"type":"string","description":"Path to file, relative to workspace or absolute inside workspace."},"content":{"type":"string","description":"UTF-8 text content to write."}},"required":["path","content"],"additionalProperties":false}
        ;
    }

    fn executeFn(ctx: *anyopaque, allocator: Allocator, arguments_json: []const u8) anyerror!ToolResult {
        const self: *FileWriteTool = @ptrCast(@alignCast(ctx));
        return self.execute(allocator, arguments_json);
    }

    fn execute(self: *FileWriteTool, allocator: Allocator, arguments_json: []const u8) !ToolResult {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, arguments_json, .{}) catch {
            return toolError(allocator, "Invalid JSON arguments.");
        };
        defer parsed.deinit();

        const obj = switch (parsed.value) {
            .object => |o| o,
            else => return toolError(allocator, "Arguments must be a JSON object."),
        };

        const path_value = obj.get("path") orelse return toolError(allocator, "Missing required field: path.");
        const path = switch (path_value) {
            .string => |s| s,
            else => return toolError(allocator, "Field 'path' must be a string."),
        };

        const content_value = obj.get("content") orelse return toolError(allocator, "Missing required field: content.");
        const content = switch (content_value) {
            .string => |s| s,
            else => return toolError(allocator, "Field 'content' must be a string."),
        };

        const resolved = resolveWritablePath(allocator, self.workspace_dir, path) catch |err| {
            return toolErrorFmt(allocator, "write_file path error: {s}", .{@errorName(err)});
        };
        defer allocator.free(resolved);

        const file = std.fs.cwd().createFile(resolved, .{
            .truncate = true,
            .read = false,
            .exclusive = false,
        }) catch |err| {
            return toolErrorFmt(allocator, "write_file failed: {s}", .{@errorName(err)});
        };
        defer file.close();

        file.writeAll(content) catch |err| {
            return toolErrorFmt(allocator, "write_file failed: {s}", .{@errorName(err)});
        };

        const rel = try toWorkspaceRelative(allocator, self.workspace_dir, resolved);
        defer allocator.free(rel);

        return toolSuccessJson(allocator, .{
            .path = rel,
            .bytes_written = content.len,
        });
    }

    const vtable = Tool.VTable{
        .name = name,
        .description = description,
        .parametersSchema = parametersSchema,
        .execute = executeFn,
        .deinit = deinitOpaque,
    };

    fn deinitOpaque(ctx: *anyopaque) void {
        const self: *FileWriteTool = @ptrCast(@alignCast(ctx));
        self.deinit();
    }
};

pub const FileEditTool = struct {
    allocator: Allocator,
    workspace_dir: []const u8,
    max_read_bytes: usize,

    pub fn init(allocator: Allocator, workspace_dir: []const u8, max_read_bytes: usize) !*FileEditTool {
        const self = try allocator.create(FileEditTool);
        errdefer allocator.destroy(self);

        const canonical_workspace = try canonicalizeWorkspace(allocator, workspace_dir);
        errdefer allocator.free(canonical_workspace);

        self.* = .{
            .allocator = allocator,
            .workspace_dir = canonical_workspace,
            .max_read_bytes = if (max_read_bytes == 0) default_read_max_bytes else max_read_bytes,
        };
        return self;
    }

    pub fn deinit(self: *FileEditTool) void {
        self.allocator.free(self.workspace_dir);
        self.allocator.destroy(self);
    }

    pub fn asTool(self: *FileEditTool) Tool {
        return Tool.init(self, &vtable);
    }

    fn name(_: *anyopaque) []const u8 {
        return "edit_file";
    }

    fn description(_: *anyopaque) []const u8 {
        return "Edit a UTF-8 text file by replacing text (single or all matches) inside the workspace sandbox.";
    }

    fn parametersSchema(_: *anyopaque) []const u8 {
        return 
        \\{"type":"object","properties":{"path":{"type":"string","description":"Path to file, relative to workspace or absolute inside workspace."},"old_text":{"type":"string","description":"Text to replace."},"new_text":{"type":"string","description":"Replacement text."},"replace_all":{"type":"boolean","description":"Replace every match. Default false (requires exactly one match)."}},"required":["path","old_text","new_text"],"additionalProperties":false}
        ;
    }

    fn executeFn(ctx: *anyopaque, allocator: Allocator, arguments_json: []const u8) anyerror!ToolResult {
        const self: *FileEditTool = @ptrCast(@alignCast(ctx));
        return self.execute(allocator, arguments_json);
    }

    fn execute(self: *FileEditTool, allocator: Allocator, arguments_json: []const u8) !ToolResult {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, arguments_json, .{}) catch {
            return toolError(allocator, "Invalid JSON arguments.");
        };
        defer parsed.deinit();

        const obj = switch (parsed.value) {
            .object => |o| o,
            else => return toolError(allocator, "Arguments must be a JSON object."),
        };

        const path = switch (obj.get("path") orelse return toolError(allocator, "Missing required field: path.")) {
            .string => |s| s,
            else => return toolError(allocator, "Field 'path' must be a string."),
        };
        const old_text = switch (obj.get("old_text") orelse return toolError(allocator, "Missing required field: old_text.")) {
            .string => |s| s,
            else => return toolError(allocator, "Field 'old_text' must be a string."),
        };
        const new_text = switch (obj.get("new_text") orelse return toolError(allocator, "Missing required field: new_text.")) {
            .string => |s| s,
            else => return toolError(allocator, "Field 'new_text' must be a string."),
        };
        const replace_all = switch (obj.get("replace_all") orelse std.json.Value{ .bool = false }) {
            .bool => |b| b,
            else => return toolError(allocator, "Field 'replace_all' must be a boolean."),
        };

        if (old_text.len == 0) return toolError(allocator, "Field 'old_text' must be non-empty.");

        const resolved = resolveExistingPath(allocator, self.workspace_dir, path) catch |err| {
            return toolErrorFmt(allocator, "edit_file path error: {s}", .{@errorName(err)});
        };
        defer allocator.free(resolved);

        const original = std.fs.cwd().readFileAlloc(allocator, resolved, self.max_read_bytes) catch |err| switch (err) {
            error.FileTooBig => return toolErrorFmt(allocator, "File exceeds edit limit ({d} bytes).", .{self.max_read_bytes}),
            else => return toolErrorFmt(allocator, "edit_file read failed: {s}", .{@errorName(err)}),
        };
        defer allocator.free(original);

        if (!std.unicode.utf8ValidateSlice(original)) {
            return toolError(allocator, "edit_file supports UTF-8 text files only.");
        }

        const replacements = countOccurrences(original, old_text);
        if (replacements == 0) return toolError(allocator, "edit_file could not find old_text.");
        if (!replace_all and replacements != 1) {
            return toolErrorFmt(allocator, "edit_file expected exactly one match, found {d}. Set replace_all=true to replace all matches.", .{replacements});
        }

        const updated = if (replace_all)
            try replaceAll(allocator, original, old_text, new_text)
        else
            try replaceFirst(allocator, original, old_text, new_text);
        defer allocator.free(updated);

        const file = std.fs.cwd().createFile(resolved, .{
            .truncate = true,
            .read = false,
            .exclusive = false,
        }) catch |err| {
            return toolErrorFmt(allocator, "edit_file write failed: {s}", .{@errorName(err)});
        };
        defer file.close();

        file.writeAll(updated) catch |err| {
            return toolErrorFmt(allocator, "edit_file write failed: {s}", .{@errorName(err)});
        };

        const rel = try toWorkspaceRelative(allocator, self.workspace_dir, resolved);
        defer allocator.free(rel);

        return toolSuccessJson(allocator, .{
            .path = rel,
            .replacements = if (replace_all) replacements else @as(usize, 1),
            .bytes_written = updated.len,
        });
    }

    const vtable = Tool.VTable{
        .name = name,
        .description = description,
        .parametersSchema = parametersSchema,
        .execute = executeFn,
        .deinit = deinitOpaque,
    };

    fn deinitOpaque(ctx: *anyopaque) void {
        const self: *FileEditTool = @ptrCast(@alignCast(ctx));
        self.deinit();
    }
};

fn canonicalizeWorkspace(allocator: Allocator, workspace_dir: []const u8) ![]u8 {
    if (workspace_dir.len == 0) return ToolError.InvalidPath;
    return std.fs.cwd().realpathAlloc(allocator, workspace_dir);
}

fn resolveExistingPath(allocator: Allocator, workspace_dir: []const u8, requested_path: []const u8) ![]u8 {
    if (requested_path.len == 0) return ToolError.InvalidPath;
    const candidate = try buildCandidatePath(allocator, workspace_dir, requested_path);
    defer allocator.free(candidate);

    const canonical = std.fs.cwd().realpathAlloc(allocator, candidate) catch |err| switch (err) {
        error.FileNotFound => return ToolError.InvalidPath,
        else => return err,
    };
    errdefer allocator.free(canonical);

    if (!isWithinWorkspace(workspace_dir, canonical)) return ToolError.PathOutsideWorkspace;
    return canonical;
}

fn resolveWritablePath(allocator: Allocator, workspace_dir: []const u8, requested_path: []const u8) ![]u8 {
    if (requested_path.len == 0) return ToolError.InvalidPath;
    const candidate = try buildCandidatePath(allocator, workspace_dir, requested_path);
    defer allocator.free(candidate);

    const existing = std.fs.cwd().realpathAlloc(allocator, candidate) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    if (existing) |canonical_existing| {
        errdefer allocator.free(canonical_existing);
        if (!isWithinWorkspace(workspace_dir, canonical_existing)) return ToolError.PathOutsideWorkspace;
        return canonical_existing;
    }

    const parent = std.fs.path.dirname(candidate) orelse return ToolError.InvalidPath;
    const basename = std.fs.path.basename(candidate);
    if (basename.len == 0 or std.mem.eql(u8, basename, ".") or std.mem.eql(u8, basename, "..")) {
        return ToolError.InvalidPath;
    }

    const canonical_parent = std.fs.cwd().realpathAlloc(allocator, parent) catch |err| switch (err) {
        error.FileNotFound => return ToolError.ParentNotFound,
        else => return err,
    };
    defer allocator.free(canonical_parent);

    if (!isWithinWorkspace(workspace_dir, canonical_parent)) return ToolError.PathOutsideWorkspace;
    return std.fs.path.join(allocator, &.{ canonical_parent, basename });
}

fn buildCandidatePath(allocator: Allocator, workspace_dir: []const u8, requested_path: []const u8) ![]u8 {
    if (std.fs.path.isAbsolute(requested_path)) {
        return allocator.dupe(u8, requested_path);
    }
    return std.fs.path.join(allocator, &.{ workspace_dir, requested_path });
}

fn isWithinWorkspace(workspace_dir: []const u8, path: []const u8) bool {
    if (std.mem.eql(u8, workspace_dir, "/") or std.mem.eql(u8, workspace_dir, "\\")) {
        return std.fs.path.isAbsolute(path);
    }
    if (std.mem.eql(u8, workspace_dir, path)) return true;
    if (!std.mem.startsWith(u8, path, workspace_dir)) return false;
    if (path.len <= workspace_dir.len) return false;

    const separator = path[workspace_dir.len];
    return separator == '/' or separator == '\\';
}

fn toWorkspaceRelative(allocator: Allocator, workspace_dir: []const u8, absolute_path: []const u8) ![]u8 {
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

fn replaceFirst(allocator: Allocator, haystack: []const u8, needle: []const u8, replacement: []const u8) ![]u8 {
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

fn replaceAll(allocator: Allocator, haystack: []const u8, needle: []const u8, replacement: []const u8) ![]u8 {
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

fn toolSuccessJson(allocator: Allocator, value: anytype) !ToolResult {
    const output = try std.fmt.allocPrint(allocator, "{f}", .{std.json.fmt(value, .{})});
    return .{
        .output = output,
        .is_error = false,
    };
}

fn toolError(allocator: Allocator, message: []const u8) !ToolResult {
    return .{
        .output = try allocator.dupe(u8, message),
        .is_error = true,
    };
}

fn toolErrorFmt(allocator: Allocator, comptime fmt: []const u8, args: anytype) !ToolResult {
    return .{
        .output = try std.fmt.allocPrint(allocator, fmt, args),
        .is_error = true,
    };
}

test "isWithinWorkspace allows descendants and rejects outside" {
    try std.testing.expect(isWithinWorkspace("/tmp/work", "/tmp/work"));
    try std.testing.expect(isWithinWorkspace("/tmp/work", "/tmp/work/src/a.txt"));
    try std.testing.expect(!isWithinWorkspace("/tmp/work", "/tmp/workspace/file.txt"));
    try std.testing.expect(!isWithinWorkspace("/tmp/work", "/tmp/other/file.txt"));
}

test "FileReadTool.execute reads UTF-8 files in workspace" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const workspace = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(workspace);

    try tmp.dir.writeFile(.{ .sub_path = "note.txt", .data = "hello world" });

    var tool_impl = try FileReadTool.init(allocator, workspace, 0);
    defer tool_impl.deinit();

    const args = "{\"path\":\"note.txt\"}";
    var result = try tool_impl.asTool().execute(allocator, args);
    defer result.deinit(allocator);

    try std.testing.expect(!result.is_error);
    try std.testing.expect(std.mem.indexOf(u8, result.output, "\"content\":\"hello world\"") != null);
}

test "FileWriteTool.execute blocks outside-workspace writes" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const workspace = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(workspace);

    var other = std.testing.tmpDir(.{});
    defer other.cleanup();
    const outside = try other.dir.realpathAlloc(allocator, ".");
    defer allocator.free(outside);

    const outside_path = try std.fs.path.join(allocator, &.{ outside, "escape.txt" });
    defer allocator.free(outside_path);

    const args = try std.fmt.allocPrint(allocator, "{{\"path\":\"{s}\",\"content\":\"x\"}}", .{outside_path});
    defer allocator.free(args);

    var tool_impl = try FileWriteTool.init(allocator, workspace);
    defer tool_impl.deinit();

    var result = try tool_impl.asTool().execute(allocator, args);
    defer result.deinit(allocator);

    try std.testing.expect(result.is_error);
    try std.testing.expect(std.mem.indexOf(u8, result.output, "PathOutsideWorkspace") != null);
}

test "FileEditTool.execute replaces all when requested" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const workspace = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(workspace);

    try tmp.dir.writeFile(.{ .sub_path = "edit.txt", .data = "one two one" });

    var tool_impl = try FileEditTool.init(allocator, workspace, 0);
    defer tool_impl.deinit();

    const args =
        \\{"path":"edit.txt","old_text":"one","new_text":"1","replace_all":true}
    ;
    var result = try tool_impl.asTool().execute(allocator, args);
    defer result.deinit(allocator);

    try std.testing.expect(!result.is_error);

    const edited = try tmp.dir.readFileAlloc(allocator, "edit.txt", 64);
    defer allocator.free(edited);
    try std.testing.expectEqualStrings("1 two 1", edited);
}
