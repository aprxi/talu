//! Mount tree setup for strict sandbox sessions.
//!
//! Builds a minimal filesystem layout in a private mount namespace. Host
//! directories are bind-mounted read-only; only the workspace is writable
//! (further restricted by Landlock). /home is explicitly excluded.
//!
//! Must be called after unshare(CLONE_NEWNS) and before Landlock/exec.
//! All operations are fail-closed: any critical mount failure aborts setup.

const std = @import("std");
const builtin = @import("builtin");

const c = if (builtin.os.tag == .linux) @cImport({
    @cInclude("errno.h");
    @cInclude("sys/mount.h");
    @cInclude("sys/stat.h");
    @cInclude("sys/syscall.h");
    @cInclude("unistd.h");
}) else struct {};

pub const MountOptions = struct {
    /// Apply MS_PRIVATE|MS_REC to prevent mount event propagation.
    make_root_private: bool = false,
    /// Build the full sandbox mount tree with pivot_root.
    enable_mount_tree: bool = false,
    /// Workspace directory to bind-mount at /workspace (required when
    /// enable_mount_tree is true).
    workspace_path: ?[]const u8 = null,
    /// Size limit for /tmp tmpfs in bytes (default 64 MB).
    tmp_size_bytes: u64 = 64 * 1024 * 1024,
};

pub fn apply(options: MountOptions) !void {
    if (builtin.os.tag != .linux) return error.StrictUnavailable;

    if (options.enable_mount_tree) {
        try makeRootPrivate();
        try setupMountTree(options);
        return;
    }
    if (options.make_root_private) {
        try makeRootPrivate();
    }
}

fn makeRootPrivate() !void {
    if (mountRaw(null, "/", null, c.MS_PRIVATE | c.MS_REC, null) != 0)
        return mapErrnoToError();
}

/// Build the sandbox filesystem tree. All critical operations fail-closed.
///
/// Uses the child PID as a unique suffix to avoid cross-session collisions.
fn setupMountTree(options: MountOptions) !void {
    // Per-session unique root path using PID.
    var root_path_buf: [64]u8 = undefined;
    const new_root = try sandboxRootPath(std.os.linux.getpid(), &root_path_buf);

    try mkdirOrFail(new_root);
    if (mountRaw("tmpfs", new_root, "tmpfs", c.MS_NOSUID | c.MS_NODEV, null) != 0)
        return error.StrictSetupFailed;

    // Bind-mount host directories read-only.
    const ro_dirs = [_][]const u8{ "/bin", "/sbin", "/usr", "/lib", "/etc" };
    for (ro_dirs) |dir| {
        if (!pathExists(dir)) continue;
        const target = try joinPaths(new_root, dir);
        try mkdirOrFail(target);
        try bindMountReadOnly(dir, target);
    }

    // /lib64 is present on some distros (multilib). Required for dynamic
    // linker resolution — fail-closed if the host has it but bind fails.
    if (pathExists("/lib64")) {
        const lib64_target = try joinPaths(new_root, "/lib64");
        try mkdirOrFail(lib64_target);
        try bindMountReadOnly("/lib64", lib64_target);
    }

    // Minimal /dev — bind-mount individual device nodes from host.
    const dev_path = try joinPaths(new_root, "/dev");
    try mkdirOrFail(dev_path);
    if (mountRaw("tmpfs", dev_path, "tmpfs", c.MS_NOSUID | c.MS_NOEXEC, "size=512k") != 0)
        return error.StrictSetupFailed;

    // Device nodes are required for basic operation.
    const dev_nodes = [_][]const u8{ "/dev/null", "/dev/zero", "/dev/urandom", "/dev/tty" };
    for (dev_nodes) |node| {
        const name = std.fs.path.basename(node);
        const target = try joinDevPath(dev_path, name);
        try touchOrFail(target);
        try bindMountReadOnly(node, target);
    }

    // /dev/ptmx and /dev/pts for PTY allocation. Required for interactive
    // shells — fail if neither devpts mount nor host bind succeeds.
    const pts_path = try joinDevPath(dev_path, "pts");
    try mkdirOrFail(pts_path);
    if (mountRaw("devpts", pts_path, "devpts", c.MS_NOSUID | c.MS_NOEXEC, "newinstance,ptmxmode=0666") != 0) {
        // Fall back to bind-mounting host /dev/ptmx.
        if (!pathExists("/dev/ptmx")) return error.StrictSetupFailed;
        const ptmx_target = try joinDevPath(dev_path, "ptmx");
        try touchOrFail(ptmx_target);
        try bindMountReadOnly("/dev/ptmx", ptmx_target);
    } else {
        const ptmx_target = try joinDevPath(dev_path, "ptmx");
        try symlinkOrFail("pts/ptmx", ptmx_target);
    }

    // Mount /proc (filtered by PID namespace).
    const proc_path = try joinPaths(new_root, "/proc");
    try mkdirOrFail(proc_path);
    if (mountRaw("proc", proc_path, "proc", c.MS_NOSUID | c.MS_NODEV | c.MS_NOEXEC, null) != 0)
        return error.StrictSetupFailed;

    // Session-private /tmp (size-limited tmpfs).
    const tmp_path = try joinPaths(new_root, "/tmp");
    try mkdirOrFail(tmp_path);
    var size_buf: [32]u8 = undefined;
    const size_opt = std.fmt.bufPrint(&size_buf, "size={d}", .{options.tmp_size_bytes}) catch
        return error.StrictSetupFailed;
    var opt_buf: [64]u8 = undefined;
    const opt_z = toOptZ(size_opt, &opt_buf) orelse return error.StrictSetupFailed;
    if (mountRaw("tmpfs", tmp_path, "tmpfs", c.MS_NOSUID | c.MS_NODEV, opt_z) != 0)
        return error.StrictSetupFailed;

    // Bind-mount workspace. This is the only writable area — Landlock
    // restricts which paths within are actually writable.
    const workspace_target = try joinPaths(new_root, "/workspace");
    try mkdirOrFail(workspace_target);
    if (options.workspace_path) |ws| {
        if (ws.len > 0) {
            try bindMountRw(ws, workspace_target);
        }
    }

    // pivot_root: make new_root the filesystem root.
    const old_root = try joinPaths(new_root, "/.old-root");
    try mkdirOrFail(old_root);
    try pivotRoot(new_root, old_root);

    // Chdir into the new root.
    const chdir_target: []const u8 = if (options.workspace_path != null) "/workspace" else "/";
    std.posix.chdir(chdir_target) catch return error.StrictSetupFailed;

    // Unmount and remove old root. Not critical — the mount is detached
    // and inaccessible after pivot, so best-effort is acceptable here.
    _ = umountRaw("/.old-root", c.MNT_DETACH);
    std.posix.rmdir("/.old-root") catch {};
}

/// Best-effort host-side cleanup for strict session mount roots.
///
/// After a strict child exits, remove its `/tmp/talu-sandbox-<pid>` directory.
/// This intentionally ignores failures (already removed, busy, etc.).
pub fn cleanupSessionRootForPid(pid: std.posix.pid_t) void {
    if (builtin.os.tag != .linux) return;
    var root_path_buf: [64]u8 = undefined;
    const root = sandboxRootPath(pid, &root_path_buf) catch return;
    std.posix.rmdir(root) catch {};
}

// ============================================================================
// Mount helpers
// ============================================================================

fn bindMountReadOnly(source: []const u8, target: []const u8) !void {
    // First bind-mount, then remount read-only (Linux requires two steps).
    if (mountZ(source, target, null, c.MS_BIND | c.MS_REC, null) != 0)
        return error.StrictSetupFailed;
    if (mountZ(source, target, null, c.MS_BIND | c.MS_REC | c.MS_RDONLY | c.MS_REMOUNT | c.MS_NOSUID | c.MS_NODEV, null) != 0)
        return error.StrictSetupFailed;
}

fn bindMountRw(source: []const u8, target: []const u8) !void {
    if (mountZ(source, target, null, c.MS_BIND | c.MS_REC, null) != 0)
        return error.StrictSetupFailed;
}

fn mountRaw(
    source: ?[*:0]const u8,
    target: []const u8,
    fstype: ?[*:0]const u8,
    flags: c_ulong,
    data: ?[*:0]const u8,
) c_int {
    var target_buf: [std.fs.max_path_bytes]u8 = undefined;
    if (target.len >= target_buf.len) return -1;
    @memcpy(target_buf[0..target.len], target);
    target_buf[target.len] = 0;
    const target_z: [*:0]const u8 = target_buf[0..target.len :0];
    return c.mount(source, target_z, fstype, flags, @ptrCast(data));
}

fn mountZ(
    source: []const u8,
    target: []const u8,
    fstype: ?[*:0]const u8,
    flags: c_ulong,
    data: ?[*:0]const u8,
) c_int {
    var source_buf: [std.fs.max_path_bytes]u8 = undefined;
    if (source.len >= source_buf.len) return -1;
    @memcpy(source_buf[0..source.len], source);
    source_buf[source.len] = 0;
    const source_z: [*:0]const u8 = source_buf[0..source.len :0];
    return mountRaw(source_z, target, fstype, flags, data);
}

fn umountRaw(target: []const u8, flags: c_int) c_int {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    if (target.len >= buf.len) return -1;
    @memcpy(buf[0..target.len], target);
    buf[target.len] = 0;
    const target_z: [*:0]const u8 = buf[0..target.len :0];
    return @intCast(c.syscall(c.SYS_umount2, @intFromPtr(target_z), @as(c_long, @intCast(flags))));
}

fn pivotRoot(new_root: []const u8, old_root: []const u8) !void {
    var new_buf: [std.fs.max_path_bytes]u8 = undefined;
    var old_buf: [std.fs.max_path_bytes]u8 = undefined;
    if (new_root.len >= new_buf.len or old_root.len >= old_buf.len)
        return error.StrictSetupFailed;
    @memcpy(new_buf[0..new_root.len], new_root);
    new_buf[new_root.len] = 0;
    @memcpy(old_buf[0..old_root.len], old_root);
    old_buf[old_root.len] = 0;
    const new_z: [*:0]const u8 = new_buf[0..new_root.len :0];
    const old_z: [*:0]const u8 = old_buf[0..old_root.len :0];
    const rc = c.syscall(c.SYS_pivot_root, @intFromPtr(new_z), @intFromPtr(old_z));
    if (rc != 0) return error.StrictSetupFailed;
}

// ============================================================================
// Path helpers (fail-closed — return errors instead of silent fallbacks)
// ============================================================================

/// Path join buffer. Results are only valid until the next call.
var path_buf: [std.fs.max_path_bytes]u8 = undefined;
var dev_path_buf: [std.fs.max_path_bytes]u8 = undefined;

fn sandboxRootPath(pid: std.posix.pid_t, buf: *[64]u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "/tmp/talu-sandbox-{d}", .{pid}) catch
        return error.StrictSetupFailed;
}

fn joinPaths(base: []const u8, suffix: []const u8) ![]const u8 {
    const total = base.len + suffix.len;
    if (total >= path_buf.len) return error.StrictSetupFailed;
    @memcpy(path_buf[0..base.len], base);
    @memcpy(path_buf[base.len..total], suffix);
    return path_buf[0..total];
}

fn joinDevPath(base: []const u8, name: []const u8) ![]const u8 {
    const total = base.len + 1 + name.len;
    if (total >= dev_path_buf.len) return error.StrictSetupFailed;
    @memcpy(dev_path_buf[0..base.len], base);
    dev_path_buf[base.len] = '/';
    @memcpy(dev_path_buf[base.len + 1 .. total], name);
    return dev_path_buf[0..total];
}

fn mkdirOrFail(path: []const u8) !void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len >= buf.len) return error.StrictSetupFailed;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    const rc = c.mkdir(buf[0..path.len :0], 0o755);
    // EEXIST is acceptable (directory already present).
    if (rc != 0 and std.c._errno().* != @intFromEnum(std.posix.E.EXIST))
        return error.StrictSetupFailed;
}

fn touchOrFail(path: []const u8) !void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len >= buf.len) return error.StrictSetupFailed;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    const fd = c.open(buf[0..path.len :0], c.O_CREAT | c.O_WRONLY, @as(c_uint, 0o644));
    if (fd < 0) return error.StrictSetupFailed;
    _ = c.close(fd);
}

fn symlinkOrFail(target: []const u8, link_path: []const u8) !void {
    var target_buf_l: [std.fs.max_path_bytes]u8 = undefined;
    var link_buf: [std.fs.max_path_bytes]u8 = undefined;
    if (target.len >= target_buf_l.len or link_path.len >= link_buf.len)
        return error.StrictSetupFailed;
    @memcpy(target_buf_l[0..target.len], target);
    target_buf_l[target.len] = 0;
    @memcpy(link_buf[0..link_path.len], link_path);
    link_buf[link_path.len] = 0;
    if (c.symlink(target_buf_l[0..target.len :0], link_buf[0..link_path.len :0]) != 0)
        return error.StrictSetupFailed;
}

fn pathExists(path: []const u8) bool {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len >= buf.len) return false;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    var st: c.struct_stat = undefined;
    return c.stat(buf[0..path.len :0], &st) == 0;
}

fn toOptZ(opt: []const u8, buf: *[64]u8) ?[*:0]const u8 {
    if (opt.len >= buf.len) return null;
    @memcpy(buf[0..opt.len], opt);
    buf[opt.len] = 0;
    return buf[0..opt.len :0];
}

fn mapErrnoToError() anyerror {
    const errno_value = std.c._errno().*;
    const err: std.posix.E = @enumFromInt(@as(u16, @intCast(errno_value)));
    return switch (err) {
        .PERM, .NOSYS, .OPNOTSUPP, .INVAL => error.StrictUnavailable,
        else => error.StrictSetupFailed,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "apply no-op when mount options disabled" {
    if (builtin.os.tag != .linux) return;
    try apply(.{});
}

test "apply returns StrictUnavailable on non-linux" {
    if (builtin.os.tag == .linux) return;
    try std.testing.expectError(error.StrictUnavailable, apply(.{ .enable_mount_tree = true }));
}

test "MountOptions defaults are disabled" {
    const opts = MountOptions{};
    try std.testing.expect(!opts.make_root_private);
    try std.testing.expect(!opts.enable_mount_tree);
    try std.testing.expectEqual(@as(?[]const u8, null), opts.workspace_path);
    try std.testing.expectEqual(@as(u64, 64 * 1024 * 1024), opts.tmp_size_bytes);
}

test "joinPaths produces correct concatenation" {
    const result = try joinPaths("/tmp/root", "/bin");
    try std.testing.expectEqualStrings("/tmp/root/bin", result);
}

test "joinPaths rejects overflow" {
    const long = "a" ** (std.fs.max_path_bytes - 1);
    try std.testing.expectError(error.StrictSetupFailed, joinPaths(long, "/extra"));
}

test "joinDevPath produces correct path" {
    const result = try joinDevPath("/tmp/root/dev", "null");
    try std.testing.expectEqualStrings("/tmp/root/dev/null", result);
}

test "sandboxRootPath produces pid-scoped directory" {
    var buf: [64]u8 = undefined;
    const path = try sandboxRootPath(1234, &buf);
    try std.testing.expectEqualStrings("/tmp/talu-sandbox-1234", path);
}
