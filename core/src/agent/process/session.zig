//! Persistent non-PTY process session.

const std = @import("std");
const Allocator = std.mem.Allocator;
const signal = @import("../shell/signal.zig");
const sandbox = @import("../sandbox/root.zig");
const helpers = @import("../sandbox/helpers.zig");
const mounts = @import("../sandbox/mounts.zig");

pub const ProcessSession = struct {
    allocator: Allocator,
    child_pid: std.posix.pid_t,
    stdin_file: ?std.fs.File,
    stdout_file: ?std.fs.File,
    stderr_file: ?std.fs.File,
    cwd: []u8,
    created_at: i64,
    last_access: i64,
    exit_code: ?i32 = null,
    closed: bool = false,
    cleanup_mount_root: bool = false,
    signal_process_group: bool = false,

    pub fn open(
        allocator: Allocator,
        command: []const u8,
        cwd: ?[]const u8,
        sandbox_config: SandboxConfig,
    ) !*ProcessSession {
        if (command.len == 0) return error.InvalidArgument;

        const resolved_cwd = try resolveCwd(allocator, cwd);
        errdefer allocator.free(resolved_cwd);

        var stdin_file: std.fs.File = undefined;
        var stdout_file: std.fs.File = undefined;
        var stderr_file: std.fs.File = undefined;
        var child_pid: std.posix.pid_t = 0;
        if (sandbox_config.mode == .host) {
            const command_z = try allocator.dupeZ(u8, command);
            defer allocator.free(command_z);

            const argv = [_][]const u8{ "/bin/sh", "-c", command_z };
            var child = std.process.Child.init(&argv, allocator);
            child.cwd = resolved_cwd;
            child.stdin_behavior = .Pipe;
            child.stdout_behavior = .Pipe;
            child.stderr_behavior = .Pipe;
            try child.spawn();

            stdin_file = child.stdin orelse return error.ExecFailed;
            stdout_file = child.stdout orelse return error.ExecFailed;
            stderr_file = child.stderr orelse return error.ExecFailed;
            child_pid = @intCast(child.id);
        } else {
            const spawned = try spawnStrictProcess(command, resolved_cwd, sandbox_config);
            stdin_file = spawned.stdin_file;
            stdout_file = spawned.stdout_file;
            stderr_file = spawned.stderr_file;
            child_pid = spawned.child_pid;
        }
        errdefer stdin_file.close();
        errdefer stdout_file.close();
        errdefer stderr_file.close();

        try helpers.setNonBlocking(stdout_file.handle);
        try helpers.setNonBlocking(stderr_file.handle);

        const now = std.time.timestamp();
        const self = try allocator.create(ProcessSession);
        self.* = .{
            .allocator = allocator,
            .child_pid = child_pid,
            .stdin_file = stdin_file,
            .stdout_file = stdout_file,
            .stderr_file = stderr_file,
            .cwd = resolved_cwd,
            .created_at = now,
            .last_access = now,
            .exit_code = null,
            .closed = false,
            .cleanup_mount_root = sandbox_config.mode == .strict and
                sandbox_config.backend == .linux_local,
            .signal_process_group = sandbox_config.mode == .strict and
                sandbox_config.backend == .linux_local,
        };
        return self;
    }

    /// Close the session. Idempotent — safe to call multiple times.
    /// Sends SIGTERM, polls for exit up to 100ms, then SIGKILL if needed.
    pub fn close(self: *ProcessSession) void {
        if (self.closed) return;
        self.closed = true;

        if (self.stdin_file) |*f| {
            f.close();
            self.stdin_file = null;
        }
        if (self.stdout_file) |*f| {
            f.close();
            self.stdout_file = null;
        }
        if (self.stderr_file) |*f| {
            f.close();
            self.stderr_file = null;
        }

        if (self.exit_code == null) {
            self.sendManagedSignal(std.posix.SIG.TERM);

            var waited = false;
            for (0..10) |_| {
                const result = std.posix.waitpid(self.child_pid, std.c.W.NOHANG);
                if (result.pid != 0) {
                    self.recordExit(result.status);
                    waited = true;
                    break;
                }
                std.Thread.sleep(10 * std.time.ns_per_ms);
            }

            if (!waited) {
                self.sendManagedSignal(std.posix.SIG.KILL);
                const result = std.posix.waitpid(self.child_pid, 0);
                self.recordExit(result.status);
            }
        }

        if (self.cleanup_mount_root) {
            mounts.cleanupSessionRootForPid(self.child_pid);
        }

        self.allocator.free(self.cwd);
        self.allocator.destroy(self);
    }

    pub fn write(self: *ProcessSession, data: []const u8) !usize {
        if (self.closed) return error.SessionClosed;
        self.last_access = std.time.timestamp();
        if (!self.pollAlive()) return error.ProcessExited;
        if (data.len == 0) return 0;

        const stdin_file = self.stdin_file orelse return error.ProcessExited;
        var written: usize = 0;
        while (written < data.len) {
            const n = std.posix.write(stdin_file.handle, data[written..]) catch |err| switch (err) {
                error.BrokenPipe => {
                    _ = self.pollAlive();
                    return error.ProcessExited;
                },
                else => return err,
            };
            if (n == 0) break;
            written += n;
        }
        return written;
    }

    pub fn read(self: *ProcessSession, buf: []u8) !usize {
        if (self.closed) return error.SessionClosed;
        self.last_access = std.time.timestamp();
        if (buf.len == 0) return 0;

        const stdout_n = try readPipe(self.stdout_file, buf);
        if (stdout_n > 0) return stdout_n;

        const stderr_n = try readPipe(self.stderr_file, buf);
        if (stderr_n > 0) return stderr_n;

        _ = self.pollAlive();
        return 0;
    }

    pub fn sendSignal(self: *ProcessSession, sig: u8) !void {
        if (self.closed) return error.SessionClosed;
        self.last_access = std.time.timestamp();
        if (!self.pollAlive()) return error.ProcessExited;
        if (self.signal_process_group) {
            try signal.sendGroup(self.child_pid, sig);
        } else {
            try signal.send(self.child_pid, sig);
        }
    }

    pub fn isAlive(self: *ProcessSession) !bool {
        if (self.closed) return error.SessionClosed;
        return self.pollAlive();
    }

    pub fn getExitCode(self: *ProcessSession) !?i32 {
        if (self.closed) return error.SessionClosed;
        _ = self.pollAlive();
        return self.exit_code;
    }

    fn pollAlive(self: *ProcessSession) bool {
        if (self.exit_code != null) return false;
        const result = std.posix.waitpid(self.child_pid, std.c.W.NOHANG);
        if (result.pid == 0) return true;
        self.recordExit(result.status);
        return false;
    }

    fn recordExit(self: *ProcessSession, status: u32) void {
        if (std.c.W.IFEXITED(status)) {
            self.exit_code = @as(i32, @intCast(std.c.W.EXITSTATUS(status)));
            return;
        }
        if (std.c.W.IFSIGNALED(status)) {
            self.exit_code = -@as(i32, @intCast(std.c.W.TERMSIG(status)));
            return;
        }
        self.exit_code = -1;
    }

    fn sendManagedSignal(self: *ProcessSession, sig: u8) void {
        if (self.signal_process_group) {
            signal.sendGroup(self.child_pid, sig) catch {};
        } else {
            signal.send(self.child_pid, sig) catch {};
        }
    }
};

pub const SandboxConfig = struct {
    mode: sandbox.RuntimeMode = .host,
    backend: sandbox.Backend = .linux_local,
    exec_profile: ?*const sandbox.profile.ExecProfile = null,

    /// Build the StrictRuntimeConfig with full isolation defaults for
    /// strict mode. `cwd` is the workspace directory to bind-mount.
    pub fn toStrictConfig(self: SandboxConfig, cwd: ?[]const u8) sandbox.StrictRuntimeConfig {
        if (self.mode == .strict) {
            var config = sandbox.StrictRuntimeConfig.defaultStrict(self.backend, cwd);
            config.exec_profile = self.exec_profile;
            return config;
        }
        return .{
            .mode = self.mode,
            .backend = self.backend,
            .exec_profile = self.exec_profile,
        };
    }
};

const SpawnedStrictProcess = struct {
    child_pid: std.posix.pid_t,
    stdin_file: std.fs.File,
    stdout_file: std.fs.File,
    stderr_file: std.fs.File,
};

fn spawnStrictProcess(
    command: []const u8,
    cwd: []const u8,
    sandbox_config: SandboxConfig,
) !SpawnedStrictProcess {
    var bootstrap = try sandbox.launcher.openBootstrapPipe();
    errdefer {
        bootstrap.closeRead();
        bootstrap.closeWrite();
    }

    const stdin_pipe = try helpers.pipeCloexec();
    errdefer {
        std.posix.close(stdin_pipe[0]);
        std.posix.close(stdin_pipe[1]);
    }
    const stdout_pipe = try helpers.pipeCloexec();
    errdefer {
        std.posix.close(stdout_pipe[0]);
        std.posix.close(stdout_pipe[1]);
    }
    const stderr_pipe = try helpers.pipeCloexec();
    errdefer {
        std.posix.close(stderr_pipe[0]);
        std.posix.close(stderr_pipe[1]);
    }

    const command_z = try std.heap.c_allocator.dupeZ(u8, command);
    defer std.heap.c_allocator.free(command_z);

    const pid_raw = std.posix.fork() catch return error.ExecFailed;
    if (pid_raw == 0) {
        bootstrap.closeChildRead();
        std.posix.setpgid(0, 0) catch |err| sandbox.launcher.childReportFailure(bootstrap.write_fd, err);

        std.posix.close(stdin_pipe[1]);
        std.posix.close(stdout_pipe[0]);
        std.posix.close(stderr_pipe[0]);

        std.posix.dup2(stdin_pipe[0], std.posix.STDIN_FILENO) catch |err| {
            sandbox.launcher.childReportFailure(bootstrap.write_fd, err);
        };
        std.posix.dup2(stdout_pipe[1], std.posix.STDOUT_FILENO) catch |err| {
            sandbox.launcher.childReportFailure(bootstrap.write_fd, err);
        };
        std.posix.dup2(stderr_pipe[1], std.posix.STDERR_FILENO) catch |err| {
            sandbox.launcher.childReportFailure(bootstrap.write_fd, err);
        };

        std.posix.close(stdin_pipe[0]);
        std.posix.close(stdout_pipe[1]);
        std.posix.close(stderr_pipe[1]);

        std.posix.chdir(cwd) catch |err| sandbox.launcher.childReportFailure(bootstrap.write_fd, err);
        sandbox.applyInChild(sandbox_config.toStrictConfig(cwd)) catch |err|
            sandbox.launcher.childReportFailure(bootstrap.write_fd, err);

        const shell_path = helpers.preferredStrictShellPath(sandbox_config.exec_profile) orelse "/bin/sh";
        var shell_buf: [std.fs.max_path_bytes]u8 = undefined;
        const shell_z = helpers.toPathZ(shell_path, &shell_buf) catch {
            sandbox.launcher.childReportFailure(bootstrap.write_fd, error.StrictSetupFailed);
        };
        const argv = [_:null]?[*:0]const u8{ shell_z.ptr, "-c", command_z };
        const exec_err = std.posix.execvpeZ(shell_z.ptr, &argv, std.c.environ);
        sandbox.launcher.childReportFailure(bootstrap.write_fd, exec_err);
    }

    bootstrap.closeParentWrite();
    sandbox.launcher.waitForExecBoundary(bootstrap.read_fd) catch |err| {
        bootstrap.closeRead();
        std.posix.close(stdin_pipe[0]);
        std.posix.close(stdin_pipe[1]);
        std.posix.close(stdout_pipe[0]);
        std.posix.close(stdout_pipe[1]);
        std.posix.close(stderr_pipe[0]);
        std.posix.close(stderr_pipe[1]);
        _ = std.posix.waitpid(pid_raw, 0);
        return err;
    };
    bootstrap.closeRead();

    const child_pid: std.posix.pid_t = pid_raw;
    std.posix.close(stdin_pipe[0]);
    std.posix.close(stdout_pipe[1]);
    std.posix.close(stderr_pipe[1]);

    return .{
        .child_pid = child_pid,
        .stdin_file = .{ .handle = stdin_pipe[1] },
        .stdout_file = .{ .handle = stdout_pipe[0] },
        .stderr_file = .{ .handle = stderr_pipe[0] },
    };
}

fn readPipe(file_opt: ?std.fs.File, buf: []u8) !usize {
    const file = file_opt orelse return 0;
    return file.read(buf) catch |err| switch (err) {
        error.WouldBlock => 0,
        error.InputOutput => 0,
        else => err,
    };
}

fn resolveCwd(allocator: Allocator, cwd: ?[]const u8) ![]u8 {
    if (cwd) |value| {
        if (value.len == 0) return error.InvalidPath;
        return std.fs.cwd().realpathAlloc(allocator, value);
    }
    return std.fs.cwd().realpathAlloc(allocator, ".");
}

test "ProcessSession open write read close roundtrip" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cwd = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(cwd);

    var session = try ProcessSession.open(
        allocator,
        "while IFS= read -r line; do echo \"$line\"; [ \"$line\" = \"quit\" ] && break; done",
        cwd,
        .{},
    );
    defer session.close();

    _ = try session.write("hello\n");
    _ = try session.write("quit\n");

    var buffer: [512]u8 = undefined;
    var output = std.ArrayList(u8).empty;
    defer output.deinit(allocator);

    var guard: usize = 0;
    while (try session.isAlive() and guard < 50_000) : (guard += 1) {
        const n = try session.read(&buffer);
        if (n > 0) try output.appendSlice(allocator, buffer[0..n]);
    }
    while (true) {
        const n = try session.read(&buffer);
        if (n == 0) break;
        try output.appendSlice(allocator, buffer[0..n]);
    }

    try std.testing.expect(std.mem.indexOf(u8, output.items, "hello") != null);
    try std.testing.expect((try session.getExitCode()) != null);
}

test "ProcessSession open strict without exec profile fails setup" {
    if (@import("builtin").os.tag != .linux) return;

    const result = ProcessSession.open(
        std.testing.allocator,
        "echo hi",
        null,
        .{
            .mode = .strict,
            .backend = .linux_local,
            .exec_profile = null,
        },
    );
    try std.testing.expectError(error.StrictSetupFailed, result);
}
