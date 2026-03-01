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
const fs = @import("../fs/root.zig");
const Tool = @import("../tool.zig").Tool;
const ToolResult = @import("../tool.zig").ToolResult;

const default_read_max_bytes: usize = 256 * 1024;

pub const FileReadTool = struct {
    allocator: Allocator,
    workspace_dir: []const u8,
    max_read_bytes: usize,

    pub fn init(allocator: Allocator, workspace_dir: []const u8, max_read_bytes: usize) !*FileReadTool {
        const self = try allocator.create(FileReadTool);
        errdefer allocator.destroy(self);

        const canonical_workspace = try fs.path.canonicalizeWorkspace(allocator, workspace_dir);
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

        const resolved = fs.path.resolveExistingPath(allocator, self.workspace_dir, path) catch |err| {
            return toolErrorFmt(allocator, "read_file path error: {s}", .{@errorName(err)});
        };
        defer allocator.free(resolved);

        const read_result = fs.operations.readFile(allocator, resolved, max_bytes) catch |err| switch (err) {
            error.FileTooBig => return toolErrorFmt(allocator, "File exceeds max_bytes ({d}).", .{max_bytes}),
            error.NonUtf8 => return toolError(allocator, "read_file supports UTF-8 text files only."),
            else => return toolErrorFmt(allocator, "read_file failed: {s}", .{@errorName(err)}),
        };
        defer allocator.free(read_result.content);

        const rel = try fs.path.toWorkspaceRelative(allocator, self.workspace_dir, resolved);
        defer allocator.free(rel);

        return toolSuccessJson(allocator, .{
            .path = rel,
            .content = read_result.content,
            .bytes = read_result.size,
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

        const canonical_workspace = try fs.path.canonicalizeWorkspace(allocator, workspace_dir);
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

        const resolved = fs.path.resolveWritablePath(allocator, self.workspace_dir, path) catch |err| {
            return toolErrorFmt(allocator, "write_file path error: {s}", .{@errorName(err)});
        };
        defer allocator.free(resolved);

        const write_result = fs.operations.writeFile(resolved, content) catch |err| {
            return toolErrorFmt(allocator, "write_file failed: {s}", .{@errorName(err)});
        };

        const rel = try fs.path.toWorkspaceRelative(allocator, self.workspace_dir, resolved);
        defer allocator.free(rel);

        return toolSuccessJson(allocator, .{
            .path = rel,
            .bytes_written = write_result.bytes_written,
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

        const canonical_workspace = try fs.path.canonicalizeWorkspace(allocator, workspace_dir);
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

        const resolved = fs.path.resolveExistingPath(allocator, self.workspace_dir, path) catch |err| {
            return toolErrorFmt(allocator, "edit_file path error: {s}", .{@errorName(err)});
        };
        defer allocator.free(resolved);

        const edit_result = fs.operations.editFile(
            allocator,
            resolved,
            old_text,
            new_text,
            replace_all,
            self.max_read_bytes,
        ) catch |err| switch (err) {
            error.FileTooBig => return toolErrorFmt(allocator, "File exceeds edit limit ({d} bytes).", .{self.max_read_bytes}),
            error.NonUtf8 => return toolError(allocator, "edit_file supports UTF-8 text files only."),
            error.NoMatches => return toolError(allocator, "edit_file could not find old_text."),
            error.MultipleMatches => return toolErrorFmt(allocator, "edit_file expected exactly one match, found multiple. Set replace_all=true to replace all matches.", .{}),
            else => return toolErrorFmt(allocator, "edit_file write failed: {s}", .{@errorName(err)}),
        };

        const rel = try fs.path.toWorkspaceRelative(allocator, self.workspace_dir, resolved);
        defer allocator.free(rel);

        return toolSuccessJson(allocator, .{
            .path = rel,
            .replacements = edit_result.replacements,
            .bytes_written = edit_result.bytes_written,
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
    try std.testing.expect(fs.path.isWithinWorkspace("/tmp/work", "/tmp/work"));
    try std.testing.expect(fs.path.isWithinWorkspace("/tmp/work", "/tmp/work/src/a.txt"));
    try std.testing.expect(!fs.path.isWithinWorkspace("/tmp/work", "/tmp/workspace/file.txt"));
    try std.testing.expect(!fs.path.isWithinWorkspace("/tmp/work", "/tmp/other/file.txt"));
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
