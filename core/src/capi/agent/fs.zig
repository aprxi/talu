//! C API for workspace-scoped filesystem operations.
//!
//! This module provides a handle-based API (`talu_fs_*`) that enforces path
//! sandboxing to a canonical workspace root.

const std = @import("std");
const Allocator = std.mem.Allocator;
const agent_fs = @import("../../agent/fs/root.zig");
const policy_api = @import("policy.zig");
const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");

const allocator = std.heap.c_allocator;
// Edit operations need a larger cap than the HTTP default read limit because
// they load full file content for replacement semantics.
const EDIT_MAX_READ_BYTES: usize = 256 * 1024 * 1024;
const ACTION_FS_READ = "tool.fs.read";
const ACTION_FS_WRITE = "tool.fs.write";
const ACTION_FS_DELETE = "tool.fs.delete";

const FsHandle = struct {
    workspace_dir: []u8,
    policy: ?*policy_api.TaluAgentPolicy,
};

const SubtreeOverride = struct {
    prefix: []u8,
    decision: policy_api.FileSubtreeDecision,
};

/// Opaque handle for workspace-scoped filesystem operations.
pub const TaluFs = opaque {};

/// C-ABI filesystem stat payload.
pub const TaluFsStat = extern struct {
    exists: bool = false,
    is_file: bool = false,
    is_dir: bool = false,
    is_symlink: bool = false,
    size: u64 = 0,
    mode: u32 = 0,
    modified_at: i64 = 0,
    created_at: i64 = 0,
    _reserved: [32]u8 = [_]u8{0} ** 32,
};

fn toHandle(handle: ?*TaluFs) !*FsHandle {
    return @ptrCast(@alignCast(handle orelse return error.InvalidHandle));
}

fn requiredArg(ptr: ?[*:0]const u8, comptime name: []const u8) ![]const u8 {
    const slice = std.mem.sliceTo(ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, name ++ " is null", .{});
        return error.InvalidArgument;
    }, 0);
    if (slice.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, name ++ " is empty", .{});
        return error.InvalidArgument;
    }
    return slice;
}

fn bytesArg(ptr: ?[*]const u8, len: usize, comptime name: []const u8) ![]const u8 {
    if (len == 0) return &.{};
    const p = ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, name ++ " is null but len > 0", .{});
        return error.InvalidArgument;
    };
    return p[0..len];
}

fn returnFsError(err: anyerror, comptime context: []const u8) i32 {
    capi_error.setError(err, context, .{});
    return @intFromEnum(error_codes.errorToCode(err));
}

fn toWorkspaceRelativeForPolicy(state: *const FsHandle, resolved_path: []const u8) ![]u8 {
    const relative = try agent_fs.path.toWorkspaceRelative(allocator, state.workspace_dir, resolved_path);
    errdefer allocator.free(relative);
    if (std.mem.eql(u8, relative, ".")) {
        allocator.free(relative);
        return allocator.dupe(u8, "");
    }
    return relative;
}

fn enforceFilePolicyOrDeny(
    state: *const FsHandle,
    action: []const u8,
    relative_path: []const u8,
    is_dir: bool,
) i32 {
    if (policy_api.enforceFilePolicy(state.policy, action, relative_path, is_dir)) {
        return 0;
    }
    capi_error.setErrorWithCode(.io_permission_denied, "agent policy denied filesystem action", .{});
    if (std.mem.eql(u8, action, ACTION_FS_READ)) {
        return @intFromEnum(error_codes.ErrorCode.policy_denied_file_read);
    }
    if (std.mem.eql(u8, action, ACTION_FS_WRITE)) {
        return @intFromEnum(error_codes.ErrorCode.policy_denied_file_write);
    }
    return @intFromEnum(error_codes.ErrorCode.policy_denied_file_delete);
}

fn pathIsInOrUnderSubtree(prefix: []const u8, path: []const u8) bool {
    if (prefix.len == 0) return true;
    if (std.mem.eql(u8, prefix, path)) return true;
    if (!std.mem.startsWith(u8, path, prefix)) return false;
    if (path.len <= prefix.len) return false;

    const sep = path[prefix.len];
    return sep == '/' or sep == '\\';
}

fn findSubtreeOverride(
    overrides: []const SubtreeOverride,
    path: []const u8,
) ?policy_api.FileSubtreeDecision {
    var best: ?policy_api.FileSubtreeDecision = null;
    var best_len: usize = 0;

    for (overrides) |entry| {
        if (!pathIsInOrUnderSubtree(entry.prefix, path)) continue;
        if (best == null or entry.prefix.len > best_len) {
            best = entry.decision;
            best_len = entry.prefix.len;
        }
    }
    return best;
}

/// Create a workspace-scoped filesystem handle.
pub fn talu_fs_create(
    workspace_dir: ?[*:0]const u8,
    policy: ?*policy_api.TaluAgentPolicy,
    out_handle: ?*?*TaluFs,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const workspace = requiredArg(workspace_dir, "workspace_dir") catch {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const state = allocator.create(FsHandle) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate fs handle", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    errdefer allocator.destroy(state);

    state.workspace_dir = agent_fs.path.canonicalizeWorkspace(allocator, workspace) catch |err| {
        return returnFsError(err, "failed to canonicalize workspace");
    };
    state.policy = policy;

    out.* = @ptrCast(state);
    return 0;
}

/// Free a filesystem handle.
pub fn talu_fs_free(handle: ?*TaluFs) callconv(.c) void {
    capi_error.clearError();
    const state: *FsHandle = @ptrCast(@alignCast(handle orelse return));
    allocator.free(state.workspace_dir);
    allocator.destroy(state);
}

/// Read a UTF-8 file from the workspace.
pub fn talu_fs_read(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    max_bytes: usize,
    out_content: ?*?[*]const u8,
    out_content_len: ?*usize,
    out_size: ?*u64,
    out_truncated: ?*bool,
) callconv(.c) i32 {
    capi_error.clearError();
    if (out_content) |ptr| ptr.* = null;
    if (out_content_len) |len| len.* = 0;
    if (out_size) |size| size.* = 0;
    if (out_truncated) |flag| flag.* = false;

    const content_ptr = out_content orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_content is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const content_len_ptr = out_content_len orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_content_len is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const state = toHandle(handle) catch |err| {
        return returnFsError(err, "invalid fs handle");
    };
    const requested_path = requiredArg(path, "path") catch {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const resolved = agent_fs.path.resolveExistingPath(allocator, state.workspace_dir, requested_path) catch |err| {
        return returnFsError(err, "failed to resolve read path");
    };
    defer allocator.free(resolved);

    const relative = toWorkspaceRelativeForPolicy(state, resolved) catch |err| {
        return returnFsError(err, "failed to derive policy path for read");
    };
    defer allocator.free(relative);
    const policy_rc = enforceFilePolicyOrDeny(state, ACTION_FS_READ, relative, false);
    if (policy_rc != 0) return policy_rc;

    const read_result = agent_fs.operations.readFile(allocator, resolved, max_bytes) catch |err| {
        return returnFsError(err, "failed to read file");
    };

    content_ptr.* = if (read_result.content.len == 0) null else read_result.content.ptr;
    content_len_ptr.* = read_result.content.len;
    if (out_size) |size| size.* = read_result.size;
    if (out_truncated) |flag| flag.* = read_result.truncated;
    return 0;
}

/// Write file content in the workspace.
pub fn talu_fs_write(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    content: ?[*]const u8,
    content_len: usize,
    mkdir: bool,
    out_bytes_written: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();
    if (out_bytes_written) |bytes| bytes.* = 0;
    const bytes_written_ptr = out_bytes_written orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_bytes_written is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const state = toHandle(handle) catch |err| {
        return returnFsError(err, "invalid fs handle");
    };
    const requested_path = requiredArg(path, "path") catch {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const content_slice = bytesArg(content, content_len, "content") catch {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const resolved = agent_fs.path.resolveWritablePath(allocator, state.workspace_dir, requested_path) catch |err| {
        return returnFsError(err, "failed to resolve write path");
    };
    defer allocator.free(resolved);

    const relative = toWorkspaceRelativeForPolicy(state, resolved) catch |err| {
        return returnFsError(err, "failed to derive policy path for write");
    };
    defer allocator.free(relative);
    const policy_rc = enforceFilePolicyOrDeny(state, ACTION_FS_WRITE, relative, false);
    if (policy_rc != 0) return policy_rc;

    if (mkdir) {
        const parent = std.fs.path.dirname(resolved) orelse {
            return returnFsError(error.InvalidPath, "failed to resolve write parent");
        };
        std.fs.cwd().makePath(parent) catch |err| {
            return returnFsError(err, "failed to create write parent");
        };
    }

    const result = agent_fs.operations.writeFile(resolved, content_slice) catch |err| {
        return returnFsError(err, "failed to write file");
    };
    bytes_written_ptr.* = result.bytes_written;
    return 0;
}

/// Edit UTF-8 file content by replacing text.
pub fn talu_fs_edit(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    old_text: ?[*]const u8,
    old_len: usize,
    new_text: ?[*]const u8,
    new_len: usize,
    replace_all: bool,
    out_replacements: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();
    if (out_replacements) |count| count.* = 0;
    const replacements_ptr = out_replacements orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_replacements is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const state = toHandle(handle) catch |err| {
        return returnFsError(err, "invalid fs handle");
    };
    const requested_path = requiredArg(path, "path") catch {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const old_slice = bytesArg(old_text, old_len, "old_text") catch {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const new_slice = bytesArg(new_text, new_len, "new_text") catch {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const resolved = agent_fs.path.resolveExistingPath(allocator, state.workspace_dir, requested_path) catch |err| {
        return returnFsError(err, "failed to resolve edit path");
    };
    defer allocator.free(resolved);

    const relative = toWorkspaceRelativeForPolicy(state, resolved) catch |err| {
        return returnFsError(err, "failed to derive policy path for edit");
    };
    defer allocator.free(relative);
    const policy_rc = enforceFilePolicyOrDeny(state, ACTION_FS_WRITE, relative, false);
    if (policy_rc != 0) return policy_rc;

    const result = agent_fs.operations.editFile(
        allocator,
        resolved,
        old_slice,
        new_slice,
        replace_all,
        EDIT_MAX_READ_BYTES,
    ) catch |err| {
        return returnFsError(err, "failed to edit file");
    };
    replacements_ptr.* = result.replacements;
    return 0;
}

/// Stat a path within the workspace.
pub fn talu_fs_stat(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    out_stat: ?*TaluFsStat,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_stat orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_stat is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = std.mem.zeroes(TaluFsStat);

    const state = toHandle(handle) catch |err| {
        return returnFsError(err, "invalid fs handle");
    };
    const requested_path = requiredArg(path, "path") catch {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const resolved = agent_fs.path.resolveWritablePath(allocator, state.workspace_dir, requested_path) catch |err| {
        return returnFsError(err, "failed to resolve stat path");
    };
    defer allocator.free(resolved);

    const result = agent_fs.operations.stat(resolved) catch |err| {
        return returnFsError(err, "failed to stat path");
    };

    const relative = toWorkspaceRelativeForPolicy(state, resolved) catch |err| {
        return returnFsError(err, "failed to derive policy path for stat");
    };
    defer allocator.free(relative);
    const policy_rc = enforceFilePolicyOrDeny(state, ACTION_FS_READ, relative, result.is_dir);
    if (policy_rc != 0) return policy_rc;

    out.* = .{
        .exists = result.exists,
        .is_file = result.is_file,
        .is_dir = result.is_dir,
        .is_symlink = result.is_symlink,
        .size = result.size,
        .mode = result.mode,
        .modified_at = result.modified_at,
        .created_at = result.created_at,
    };
    return 0;
}

/// List a directory within the workspace and return JSON.
pub fn talu_fs_list(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    glob: ?[*:0]const u8,
    recursive: bool,
    limit: usize,
    out_json: ?*?[*]const u8,
    out_json_len: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();
    if (out_json) |ptr| ptr.* = null;
    if (out_json_len) |len| len.* = 0;
    const out_ptr = out_json orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_json is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const out_len_ptr = out_json_len orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_json_len is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const state = toHandle(handle) catch |err| {
        return returnFsError(err, "invalid fs handle");
    };
    const requested_path = requiredArg(path, "path") catch {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const glob_pattern = if (glob) |glob_z| std.mem.sliceTo(glob_z, 0) else null;

    const resolved = agent_fs.path.resolveExistingPath(allocator, state.workspace_dir, requested_path) catch |err| {
        return returnFsError(err, "failed to resolve list path");
    };
    defer allocator.free(resolved);

    const relative_root = toWorkspaceRelativeForPolicy(state, resolved) catch |err| {
        return returnFsError(err, "failed to derive policy path for list root");
    };
    defer allocator.free(relative_root);
    const root_policy_rc = enforceFilePolicyOrDeny(state, ACTION_FS_READ, relative_root, true);
    if (root_policy_rc != 0) return root_policy_rc;

    var result = agent_fs.operations.listDir(allocator, resolved, glob_pattern, recursive, limit) catch |err| {
        return returnFsError(err, "failed to list path");
    };
    defer result.deinit(allocator);

    if (state.policy != null) {
        var filtered = std.ArrayList(agent_fs.operations.ListEntry).empty;
        errdefer {
            for (filtered.items) |entry| {
                allocator.free(entry.path);
                allocator.free(entry.name);
            }
            filtered.deinit(allocator);
        }

        var subtree_overrides = std.ArrayList(SubtreeOverride).empty;
        defer {
            for (subtree_overrides.items) |entry| {
                allocator.free(entry.prefix);
            }
            subtree_overrides.deinit(allocator);
        }

        for (result.entries) |entry| {
            const abs_entry_path = if (entry.path.len == 0)
                allocator.dupe(u8, resolved) catch |err| {
                    return returnFsError(err, "failed to allocate list entry path");
                }
            else
                std.fs.path.join(allocator, &.{ resolved, entry.path }) catch |err| {
                    return returnFsError(err, "failed to build list entry path");
                };
            defer allocator.free(abs_entry_path);

            const relative_entry = toWorkspaceRelativeForPolicy(state, abs_entry_path) catch |err| {
                return returnFsError(err, "failed to derive policy path for list entry");
            };
            defer allocator.free(relative_entry);

            var decision = if (recursive)
                findSubtreeOverride(subtree_overrides.items, relative_entry)
            else
                null;
            if (decision == null) {
                const allowed = policy_api.enforceFilePolicy(state.policy, ACTION_FS_READ, relative_entry, entry.is_dir);
                decision = if (allowed) .allow else .deny;
            }

            if (decision.? == .allow) {
                filtered.append(allocator, entry) catch |err| {
                    return returnFsError(err, "failed to append filtered list entry");
                };
            } else {
                allocator.free(entry.path);
                allocator.free(entry.name);
            }

            // Recursive list responses are depth-first. Cache a definitive
            // subtree decision from parent `/**`-style policy statements to
            // avoid evaluating every child entry individually.
            if (recursive and entry.is_dir and
                findSubtreeOverride(subtree_overrides.items, relative_entry) == null)
            {
                if (policy_api.fileSubtreeDecision(state.policy, ACTION_FS_READ, relative_entry)) |subtree_decision| {
                    const subtree_prefix = allocator.dupe(u8, relative_entry) catch |err| {
                        return returnFsError(err, "failed to allocate subtree override prefix");
                    };
                    errdefer allocator.free(subtree_prefix);

                    subtree_overrides.append(allocator, .{
                        .prefix = subtree_prefix,
                        .decision = subtree_decision,
                    }) catch |err| {
                        return returnFsError(err, "failed to append subtree override");
                    };
                }
            }
        }

        allocator.free(result.entries);
        result.entries = filtered.toOwnedSlice(allocator) catch |err| {
            return returnFsError(err, "failed to materialize filtered list");
        };
    }

    const json = std.fmt.allocPrint(allocator, "{f}", .{std.json.fmt(.{
        .entries = result.entries,
        .truncated = result.truncated,
    }, .{})}) catch |err| {
        return returnFsError(err, "failed to encode list response");
    };
    out_ptr.* = if (json.len == 0) null else json.ptr;
    out_len_ptr.* = json.len;
    return 0;
}

/// Remove file/directory in the workspace.
pub fn talu_fs_remove(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    recursive: bool,
) callconv(.c) i32 {
    capi_error.clearError();
    const state = toHandle(handle) catch |err| {
        return returnFsError(err, "invalid fs handle");
    };
    const requested_path = requiredArg(path, "path") catch {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const resolved = agent_fs.path.resolveExistingPath(allocator, state.workspace_dir, requested_path) catch |err| {
        return returnFsError(err, "failed to resolve remove path");
    };
    defer allocator.free(resolved);

    const metadata = std.fs.cwd().statFile(resolved) catch |err| {
        return returnFsError(err, "failed to stat remove path");
    };
    const relative = toWorkspaceRelativeForPolicy(state, resolved) catch |err| {
        return returnFsError(err, "failed to derive policy path for remove");
    };
    defer allocator.free(relative);
    const policy_rc = enforceFilePolicyOrDeny(state, ACTION_FS_DELETE, relative, metadata.kind == .directory);
    if (policy_rc != 0) return policy_rc;

    agent_fs.operations.remove(resolved, recursive) catch |err| {
        return returnFsError(err, "failed to remove path");
    };
    return 0;
}

/// Create directory in the workspace.
pub fn talu_fs_mkdir(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    recursive: bool,
) callconv(.c) i32 {
    capi_error.clearError();
    const state = toHandle(handle) catch |err| {
        return returnFsError(err, "invalid fs handle");
    };
    const requested_path = requiredArg(path, "path") catch {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const resolved = agent_fs.path.resolveWritablePath(allocator, state.workspace_dir, requested_path) catch |err| {
        return returnFsError(err, "failed to resolve mkdir path");
    };
    defer allocator.free(resolved);

    const relative = toWorkspaceRelativeForPolicy(state, resolved) catch |err| {
        return returnFsError(err, "failed to derive policy path for mkdir");
    };
    defer allocator.free(relative);
    const policy_rc = enforceFilePolicyOrDeny(state, ACTION_FS_WRITE, relative, true);
    if (policy_rc != 0) return policy_rc;

    agent_fs.operations.makeDir(resolved, recursive) catch |err| {
        return returnFsError(err, "failed to create directory");
    };
    return 0;
}

/// Rename or move a path within the workspace.
pub fn talu_fs_rename(
    handle: ?*TaluFs,
    from: ?[*:0]const u8,
    to: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const state = toHandle(handle) catch |err| {
        return returnFsError(err, "invalid fs handle");
    };
    const from_path = requiredArg(from, "from") catch {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const to_path = requiredArg(to, "to") catch {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const resolved_from = agent_fs.path.resolveExistingPath(allocator, state.workspace_dir, from_path) catch |err| {
        return returnFsError(err, "failed to resolve source path");
    };
    defer allocator.free(resolved_from);

    const resolved_to = agent_fs.path.resolveWritablePath(allocator, state.workspace_dir, to_path) catch |err| {
        return returnFsError(err, "failed to resolve destination path");
    };
    defer allocator.free(resolved_to);

    const from_stat = std.fs.cwd().statFile(resolved_from) catch |err| {
        return returnFsError(err, "failed to stat source path");
    };
    const from_relative = toWorkspaceRelativeForPolicy(state, resolved_from) catch |err| {
        return returnFsError(err, "failed to derive policy source path for rename");
    };
    defer allocator.free(from_relative);
    const delete_rc = enforceFilePolicyOrDeny(state, ACTION_FS_DELETE, from_relative, from_stat.kind == .directory);
    if (delete_rc != 0) return delete_rc;

    const to_relative = toWorkspaceRelativeForPolicy(state, resolved_to) catch |err| {
        return returnFsError(err, "failed to derive policy destination path for rename");
    };
    defer allocator.free(to_relative);
    const write_rc = enforceFilePolicyOrDeny(state, ACTION_FS_WRITE, to_relative, from_stat.kind == .directory);
    if (write_rc != 0) return write_rc;

    agent_fs.operations.rename(resolved_from, resolved_to) catch |err| {
        return returnFsError(err, "failed to rename path");
    };
    return 0;
}

/// Free string/buffer returned by `talu_fs_read` and `talu_fs_list`.
pub fn talu_fs_free_string(ptr: ?[*]const u8, len: usize) callconv(.c) void {
    capi_error.clearError();
    if (ptr == null or len == 0) return;
    const mutable: [*]u8 = @constCast(ptr.?);
    allocator.free(mutable[0..len]);
}

test "talu_fs_create and read/write roundtrip" {
    const test_allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const workspace = try tmp.dir.realpathAlloc(test_allocator, ".");
    defer test_allocator.free(workspace);

    const workspace_z = try test_allocator.allocSentinel(u8, workspace.len, 0);
    defer test_allocator.free(workspace_z);
    @memcpy(workspace_z, workspace);

    var fs_handle: ?*TaluFs = null;
    try std.testing.expectEqual(@as(i32, 0), talu_fs_create(workspace_z.ptr, null, &fs_handle));
    defer talu_fs_free(fs_handle);

    const path_text = "hello.txt";
    const path_z = try test_allocator.allocSentinel(u8, path_text.len, 0);
    defer test_allocator.free(path_z);
    @memcpy(path_z, path_text);

    const body = "hello fs";
    var bytes_written: usize = 0;
    try std.testing.expectEqual(
        @as(i32, 0),
        talu_fs_write(fs_handle, path_z.ptr, body.ptr, body.len, false, &bytes_written),
    );
    try std.testing.expectEqual(body.len, bytes_written);

    var out_ptr: ?[*]const u8 = null;
    var out_len: usize = 0;
    var out_size: u64 = 0;
    var out_truncated = false;
    try std.testing.expectEqual(
        @as(i32, 0),
        talu_fs_read(fs_handle, path_z.ptr, 1024, &out_ptr, &out_len, &out_size, &out_truncated),
    );
    defer talu_fs_free_string(out_ptr, out_len);

    try std.testing.expectEqual(body.len, out_len);
    try std.testing.expectEqual(@as(u64, body.len), out_size);
    try std.testing.expect(!out_truncated);
    try std.testing.expectEqualStrings(body, out_ptr.?[0..out_len]);
}

test "talu_fs_read rejects outside-workspace paths" {
    const test_allocator = std.testing.allocator;

    var workspace_tmp = std.testing.tmpDir(.{});
    defer workspace_tmp.cleanup();
    const workspace = try workspace_tmp.dir.realpathAlloc(test_allocator, ".");
    defer test_allocator.free(workspace);

    var outside_tmp = std.testing.tmpDir(.{});
    defer outside_tmp.cleanup();
    try outside_tmp.dir.writeFile(.{ .sub_path = "outside.txt", .data = "nope" });
    const outside_root = try outside_tmp.dir.realpathAlloc(test_allocator, ".");
    defer test_allocator.free(outside_root);
    const outside_file = try std.fs.path.join(test_allocator, &.{ outside_root, "outside.txt" });
    defer test_allocator.free(outside_file);

    const workspace_z = try test_allocator.allocSentinel(u8, workspace.len, 0);
    defer test_allocator.free(workspace_z);
    @memcpy(workspace_z, workspace);
    const outside_z = try test_allocator.allocSentinel(u8, outside_file.len, 0);
    defer test_allocator.free(outside_z);
    @memcpy(outside_z, outside_file);

    var fs_handle: ?*TaluFs = null;
    try std.testing.expectEqual(@as(i32, 0), talu_fs_create(workspace_z.ptr, null, &fs_handle));
    defer talu_fs_free(fs_handle);

    var out_ptr: ?[*]const u8 = null;
    var out_len: usize = 0;
    const rc = talu_fs_read(fs_handle, outside_z.ptr, 1024, &out_ptr, &out_len, null, null);
    try std.testing.expect(rc != 0);
}

test "talu_fs_write enforces agent policy when provided" {
    const test_allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const workspace = try tmp.dir.realpathAlloc(test_allocator, ".");
    defer test_allocator.free(workspace);

    const workspace_z = try test_allocator.allocSentinel(u8, workspace.len, 0);
    defer test_allocator.free(workspace_z);
    @memcpy(workspace_z, workspace);

    var policy_handle: ?*policy_api.TaluAgentPolicy = null;
    const policy_json =
        \\{"default":"deny","statements":[{"effect":"allow","action":"tool.fs.read","resource":"**"}]}
    ;
    try std.testing.expectEqual(
        @as(i32, 0),
        policy_api.talu_agent_policy_create(policy_json.ptr, policy_json.len, &policy_handle),
    );
    defer policy_api.talu_agent_policy_free(policy_handle);

    var fs_handle: ?*TaluFs = null;
    try std.testing.expectEqual(
        @as(i32, 0),
        talu_fs_create(workspace_z.ptr, policy_handle, &fs_handle),
    );
    defer talu_fs_free(fs_handle);

    const path_text = "blocked.txt";
    const path_z = try test_allocator.allocSentinel(u8, path_text.len, 0);
    defer test_allocator.free(path_z);
    @memcpy(path_z, path_text);

    const content = "x";
    var bytes_written: usize = 0;
    const rc = talu_fs_write(fs_handle, path_z.ptr, content.ptr, content.len, false, &bytes_written);
    try std.testing.expectEqual(
        @as(i32, @intFromEnum(error_codes.ErrorCode.policy_denied_file_write)),
        rc,
    );
}

test "talu_fs_list recursive policy filter preserves nested deny semantics" {
    const test_allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makePath("src/private");
    try tmp.dir.writeFile(.{ .sub_path = "src/a.txt", .data = "a" });
    try tmp.dir.writeFile(.{ .sub_path = "src/private/secret.txt", .data = "s" });

    const workspace = try tmp.dir.realpathAlloc(test_allocator, ".");
    defer test_allocator.free(workspace);

    const workspace_z = try test_allocator.allocSentinel(u8, workspace.len, 0);
    defer test_allocator.free(workspace_z);
    @memcpy(workspace_z, workspace);

    var policy_handle: ?*policy_api.TaluAgentPolicy = null;
    const policy_json =
        \\{"default":"deny","statements":[
        \\  {"effect":"allow","action":"tool.fs.read","resource":"src/**"},
        \\  {"effect":"deny","action":"tool.fs.read","resource":"src/private/**"}
        \\]}
    ;
    try std.testing.expectEqual(
        @as(i32, 0),
        policy_api.talu_agent_policy_create(policy_json.ptr, policy_json.len, &policy_handle),
    );
    defer policy_api.talu_agent_policy_free(policy_handle);

    var fs_handle: ?*TaluFs = null;
    try std.testing.expectEqual(@as(i32, 0), talu_fs_create(workspace_z.ptr, policy_handle, &fs_handle));
    defer talu_fs_free(fs_handle);

    const list_path = "src";
    const list_path_z = try test_allocator.allocSentinel(u8, list_path.len, 0);
    defer test_allocator.free(list_path_z);
    @memcpy(list_path_z, list_path);

    var out_json: ?[*]const u8 = null;
    var out_len: usize = 0;
    try std.testing.expectEqual(
        @as(i32, 0),
        talu_fs_list(fs_handle, list_path_z.ptr, null, true, 100, &out_json, &out_len),
    );
    defer talu_fs_free_string(out_json, out_len);

    const EntryView = struct {
        path: []const u8,
        name: []const u8,
        is_dir: bool,
        is_symlink: bool,
        size: u64,
        modified_at: i64,
    };
    const ListResponse = struct {
        entries: []const EntryView,
        truncated: bool,
    };
    const parsed = try std.json.parseFromSlice(ListResponse, test_allocator, out_json.?[0..out_len], .{});
    defer parsed.deinit();

    try std.testing.expect(!parsed.value.truncated);
    try std.testing.expectEqual(@as(usize, 1), parsed.value.entries.len);
    try std.testing.expectEqualStrings("a.txt", parsed.value.entries[0].path);
}

test "fuzz talu_fs_stat path handling" {
    const test_allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const workspace = try tmp.dir.realpathAlloc(test_allocator, ".");
    defer test_allocator.free(workspace);

    const workspace_z = try test_allocator.allocSentinel(u8, workspace.len, 0);
    defer test_allocator.free(workspace_z);
    @memcpy(workspace_z, workspace);

    var fs_handle: ?*TaluFs = null;
    try std.testing.expectEqual(@as(i32, 0), talu_fs_create(workspace_z.ptr, null, &fs_handle));
    defer talu_fs_free(fs_handle);

    const FuzzCtx = struct {
        var handle: ?*TaluFs = null;

        fn testOne(_: @TypeOf(.{}), input: []const u8) !void {
            var path_buf = std.ArrayList(u8).empty;
            defer path_buf.deinit(std.testing.allocator);

            for (input) |byte| {
                const safe = if (byte == 0) @as(u8, '/') else byte;
                try path_buf.append(std.testing.allocator, safe);
                if (path_buf.items.len >= 96) break;
            }
            if (path_buf.items.len == 0) try path_buf.append(std.testing.allocator, '.');

            const path_z = try std.testing.allocator.allocSentinel(u8, path_buf.items.len, 0);
            defer std.testing.allocator.free(path_z);
            @memcpy(path_z, path_buf.items);

            var out_stat = std.mem.zeroes(TaluFsStat);
            _ = talu_fs_stat(handle, path_z.ptr, &out_stat);
        }
    };
    FuzzCtx.handle = fs_handle;
    try std.testing.fuzz(.{}, FuzzCtx.testOne, .{});
}
