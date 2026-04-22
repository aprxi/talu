//! Compile runtime sandbox profile from policy + baseline shell safety rules.

const std = @import("std");
const policy_mod = @import("../policy/root.zig");
const safety = @import("../shell/safety.zig");

const ACTION_FS_READ = "tool.fs.read";
const ACTION_FS_WRITE = "tool.fs.write";
const ACTION_FS_DELETE = "tool.fs.delete";

pub const FsAccessMask = struct {
    read: bool = false,
    write: bool = false,
    delete: bool = false,

    fn merge(self: *FsAccessMask, other: FsAccessMask) void {
        self.read = self.read or other.read;
        self.write = self.write or other.write;
        self.delete = self.delete or other.delete;
    }
};

pub const ExecProfile = struct {
    allocator: std.mem.Allocator,
    names_buf: []u8,
    names: []const []const u8,
    paths_buf: []u8,
    paths: []const []const u8,
    fs_paths_buf: []u8,
    fs_paths: []const []const u8,
    fs_access: []const FsAccessMask,
    enforce_read: bool,
    enforce_write: bool,
    enforce_delete: bool,

    pub fn deinit(self: *ExecProfile) void {
        self.allocator.free(self.fs_access);
        self.allocator.free(self.fs_paths);
        self.allocator.free(self.fs_paths_buf);
        self.allocator.free(self.paths);
        self.allocator.free(self.paths_buf);
        self.allocator.free(self.names);
        self.allocator.free(self.names_buf);
        self.* = undefined;
    }
};

pub const BuildOptions = struct {
    action: []const u8,
    cwd: ?[]const u8 = null,
    include_shell_paths: bool = false,
};

pub fn buildExecProfile(
    allocator: std.mem.Allocator,
    policy: ?*const policy_mod.Policy,
    options: BuildOptions,
) !ExecProfile {
    const names = try collectAllowedNames(allocator, policy, options);
    errdefer {
        allocator.free(names.names);
        allocator.free(names.names_buf);
    }

    const paths = try resolveExecutablePaths(allocator, names.names);
    errdefer {
        allocator.free(paths.paths);
        allocator.free(paths.paths_buf);
    }

    const fs_rules = try compileFsRules(allocator, policy, options.cwd);
    errdefer {
        allocator.free(fs_rules.access);
        allocator.free(fs_rules.paths);
        allocator.free(fs_rules.paths_buf);
    }

    return .{
        .allocator = allocator,
        .names_buf = names.names_buf,
        .names = names.names,
        .paths_buf = paths.paths_buf,
        .paths = paths.paths,
        .fs_paths_buf = fs_rules.paths_buf,
        .fs_paths = fs_rules.paths,
        .fs_access = fs_rules.access,
        .enforce_read = fs_rules.enforce_read,
        .enforce_write = fs_rules.enforce_write,
        .enforce_delete = fs_rules.enforce_delete,
    };
}

const NameList = struct {
    names_buf: []u8,
    names: []const []const u8,
};

fn collectAllowedNames(
    allocator: std.mem.Allocator,
    policy: ?*const policy_mod.Policy,
    options: BuildOptions,
) !NameList {
    var selected = std.ArrayList([]const u8).empty;
    defer selected.deinit(allocator);

    for (safety.ALLOWED_COMMANDS) |cmd| {
        if (commandAllowed(policy, options.action, cmd, options.cwd)) {
            try selected.append(allocator, cmd);
        }
    }

    if (options.include_shell_paths) {
        if (shellBasenameFromEnv()) |name| {
            if (!contains(selected.items, name)) {
                try selected.append(allocator, name);
            }
        }
        const defaults: []const []const u8 = &.{ "sh", "bash", "zsh" };
        for (defaults) |name| {
            if (!contains(selected.items, name)) {
                try selected.append(allocator, name);
            }
        }
    }

    var total_len: usize = 0;
    for (selected.items) |name| total_len += name.len;

    const names_buf = try allocator.alloc(u8, total_len);
    errdefer allocator.free(names_buf);
    const names = try allocator.alloc([]const u8, selected.items.len);
    errdefer allocator.free(names);

    var offset: usize = 0;
    for (selected.items, 0..) |name, i| {
        const start = offset;
        const end = start + name.len;
        @memcpy(names_buf[start..end], name);
        names[i] = names_buf[start..end];
        offset = end;
    }

    return .{ .names_buf = names_buf, .names = names };
}

fn commandAllowed(
    policy: ?*const policy_mod.Policy,
    action: []const u8,
    command_name: []const u8,
    cwd: ?[]const u8,
) bool {
    if (policy == null) return true;
    const result = policy_mod.checkProcessAction(policy.?, action, command_name, cwd);
    return result.allowed;
}

fn shellBasenameFromEnv() ?[]const u8 {
    const shell = std.posix.getenv("SHELL") orelse return null;
    const abs = std.mem.sliceTo(shell, 0);
    if (!std.fs.path.isAbsolute(abs)) return null;
    return std.fs.path.basename(abs);
}

const PathList = struct {
    paths_buf: []u8,
    paths: []const []const u8,
};

fn resolveExecutablePaths(allocator: std.mem.Allocator, names: []const []const u8) !PathList {
    const path_env = std.posix.getenv("PATH") orelse return .{
        .paths_buf = try allocator.alloc(u8, 0),
        .paths = try allocator.alloc([]const u8, 0),
    };
    const path_slice = std.mem.sliceTo(path_env, 0);

    var resolved = std.ArrayList([]const u8).empty;
    defer {
        for (resolved.items) |entry| allocator.free(entry);
        resolved.deinit(allocator);
    }

    var dir_iter = std.mem.tokenizeScalar(u8, path_slice, ':');
    while (dir_iter.next()) |dir| {
        if (dir.len == 0) continue;
        for (names) |name| {
            const candidate = std.fs.path.join(allocator, &.{ dir, name }) catch continue;
            defer allocator.free(candidate);
            try appendResolvedExecutablePath(allocator, &resolved, candidate);
        }
    }

    // Dynamic loader executables required by most dynamically linked tools.
    const runtime_execs: []const []const u8 = &.{
        "/lib64/ld-linux-x86-64.so.2",
        "/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
        "/lib/ld-linux-x86-64.so.2",
        "/lib/ld-musl-x86_64.so.1",
        "/bin/sh",
        "/bin/bash",
        "/bin/zsh",
        "/usr/bin/sh",
        "/usr/bin/bash",
        "/usr/bin/zsh",
    };
    for (runtime_execs) |candidate| {
        try appendResolvedExecutablePath(allocator, &resolved, candidate);
    }

    return finalizeOwnedPathList(allocator, resolved.items);
}

fn appendResolvedExecutablePath(
    allocator: std.mem.Allocator,
    resolved: *std.ArrayList([]const u8),
    candidate: []const u8,
) !void {
    std.posix.access(candidate, std.posix.X_OK) catch return;
    const canonical = std.fs.cwd().realpathAlloc(allocator, candidate) catch return;
    errdefer allocator.free(canonical);
    if (contains(resolved.items, canonical)) {
        allocator.free(canonical);
        return;
    }
    try resolved.append(allocator, canonical);
}

const ActionState = struct {
    has_allow: bool = false,
    has_deny: bool = false,
    enforce: bool = false,
};

const CompiledFsRules = struct {
    paths_buf: []u8,
    paths: []const []const u8,
    access: []const FsAccessMask,
    enforce_read: bool,
    enforce_write: bool,
    enforce_delete: bool,
};

const RuleBuilder = struct {
    allocator: std.mem.Allocator,
    paths: std.ArrayList([]u8) = .empty,
    access: std.ArrayList(FsAccessMask) = .empty,

    fn deinit(self: *RuleBuilder) void {
        for (self.paths.items) |path| self.allocator.free(path);
        self.paths.deinit(self.allocator);
        self.access.deinit(self.allocator);
    }

    fn append(self: *RuleBuilder, path: []u8, mask: FsAccessMask) !void {
        for (self.paths.items, 0..) |existing, i| {
            if (std.mem.eql(u8, existing, path)) {
                self.access.items[i].merge(mask);
                self.allocator.free(path);
                return;
            }
        }
        try self.paths.append(self.allocator, path);
        try self.access.append(self.allocator, mask);
    }
};

fn compileFsRules(
    allocator: std.mem.Allocator,
    policy: ?*const policy_mod.Policy,
    cwd: ?[]const u8,
) !CompiledFsRules {
    if (policy == null) {
        return .{
            .paths_buf = try allocator.alloc(u8, 0),
            .paths = try allocator.alloc([]const u8, 0),
            .access = try allocator.alloc(FsAccessMask, 0),
            .enforce_read = false,
            .enforce_write = false,
            .enforce_delete = false,
        };
    }

    const base_dir = try resolveBaseDir(allocator, cwd);
    defer allocator.free(base_dir);

    var read_state = actionStateFor(policy.?, ACTION_FS_READ) catch return error.StrictSetupFailed;
    var write_state = actionStateFor(policy.?, ACTION_FS_WRITE) catch return error.StrictSetupFailed;
    var delete_state = actionStateFor(policy.?, ACTION_FS_DELETE) catch return error.StrictSetupFailed;

    // Read enforcement without mount isolation is limited: enforce only
    // explicit allow-rules on default-deny policies; fail closed on deny-rules.
    if (read_state.has_deny) return error.StrictSetupFailed;
    if (policy.?.default_effect == .allow) {
        read_state.enforce = false;
    } else {
        read_state.enforce = read_state.has_allow;
    }

    // Write/delete enforcement is currently allowlist-based: only explicit
    // allow statements compile to kernel rules. Deny statements are not yet
    // representable without a full mount/layout isolation layer.
    if (write_state.has_deny) return error.StrictSetupFailed;
    write_state.enforce = write_state.has_allow;
    if (delete_state.has_deny) return error.StrictSetupFailed;
    delete_state.enforce = delete_state.has_allow;

    var builder = RuleBuilder{ .allocator = allocator };
    defer builder.deinit();

    if (read_state.enforce) {
        try appendSystemReadRules(&builder);
        try appendAllowRulesForAction(&builder, policy.?, ACTION_FS_READ, .{ .read = true }, base_dir);
    }
    if (write_state.enforce) {
        try appendAllowRulesForAction(&builder, policy.?, ACTION_FS_WRITE, .{ .write = true }, base_dir);
    }
    if (delete_state.enforce) {
        try appendAllowRulesForAction(&builder, policy.?, ACTION_FS_DELETE, .{ .delete = true }, base_dir);
    }

    var paths_out = try allocator.alloc([]const u8, builder.paths.items.len);
    errdefer allocator.free(paths_out);
    var access_out = try allocator.alloc(FsAccessMask, builder.access.items.len);
    errdefer allocator.free(access_out);

    var total_len: usize = 0;
    for (builder.paths.items) |path| total_len += path.len;
    var paths_buf = try allocator.alloc(u8, total_len);
    errdefer allocator.free(paths_buf);

    var offset: usize = 0;
    for (builder.paths.items, 0..) |path, i| {
        const start = offset;
        const end = start + path.len;
        @memcpy(paths_buf[start..end], path);
        paths_out[i] = paths_buf[start..end];
        access_out[i] = builder.access.items[i];
        offset = end;
    }

    return .{
        .paths_buf = paths_buf,
        .paths = paths_out,
        .access = access_out,
        .enforce_read = read_state.enforce,
        .enforce_write = write_state.enforce,
        .enforce_delete = delete_state.enforce,
    };
}

fn actionStateFor(policy: *const policy_mod.Policy, action: []const u8) !ActionState {
    var state = ActionState{};
    for (policy.statements) |stmt| {
        if (!actionPatternMatches(stmt.action_pattern, action)) continue;
        switch (stmt.effect) {
            .allow => state.has_allow = true,
            .deny => state.has_deny = true,
        }
    }

    if (policy.default_effect == .allow) {
        if (state.has_deny) return error.StrictSetupFailed;
        state.enforce = false;
        return state;
    }

    if (state.has_allow and state.has_deny) return error.StrictSetupFailed;
    state.enforce = true;
    return state;
}

fn appendAllowRulesForAction(
    builder: *RuleBuilder,
    policy: *const policy_mod.Policy,
    action: []const u8,
    mask: FsAccessMask,
    base_dir: []const u8,
) !void {
    for (policy.statements) |stmt| {
        if (stmt.effect != .allow) continue;
        if (!actionPatternMatches(stmt.action_pattern, action)) continue;
        const canonical = try compileResourceRulePath(builder.allocator, base_dir, stmt.resource_pattern);
        try builder.append(canonical, mask);
    }
}

fn appendSystemReadRules(builder: *RuleBuilder) !void {
    const baseline: []const []const u8 = &.{
        "/bin",
        "/usr",
        "/lib",
        "/lib64",
        "/etc",
    };
    for (baseline) |raw| {
        const canonical = std.fs.cwd().realpathAlloc(builder.allocator, raw) catch continue;
        try builder.append(canonical, .{ .read = true });
    }
}

fn resolveBaseDir(allocator: std.mem.Allocator, cwd: ?[]const u8) ![]u8 {
    if (cwd) |value| {
        if (value.len == 0) return error.StrictSetupFailed;
        return std.fs.cwd().realpathAlloc(allocator, value);
    }
    return std.fs.cwd().realpathAlloc(allocator, ".");
}

fn compileResourceRulePath(
    allocator: std.mem.Allocator,
    base_dir: []const u8,
    resource_pattern: ?[]const u8,
) ![]u8 {
    var pattern = resource_pattern orelse "**";
    if (pattern.len == 0) return error.StrictSetupFailed;
    if (pattern[0] == '/') {
        pattern = pattern[1..];
        if (pattern.len == 0) pattern = "**";
    }

    var joined: []u8 = undefined;
    if (std.mem.eql(u8, pattern, "**")) {
        joined = try allocator.dupe(u8, base_dir);
    } else if (std.mem.endsWith(u8, pattern, "/**")) {
        const prefix = pattern[0 .. pattern.len - 3];
        if (containsWildcard(prefix)) return error.StrictSetupFailed;
        if (prefix.len == 0) {
            joined = try allocator.dupe(u8, base_dir);
        } else {
            joined = try std.fs.path.join(allocator, &.{ base_dir, prefix });
        }
    } else {
        if (std.mem.endsWith(u8, pattern, "/")) return error.StrictSetupFailed;
        if (containsWildcard(pattern)) return error.StrictSetupFailed;
        joined = try std.fs.path.join(allocator, &.{ base_dir, pattern });
    }
    defer allocator.free(joined);

    const canonical = std.fs.cwd().realpathAlloc(allocator, joined) catch |err| switch (err) {
        // Some allowlist paths are cwd-dependent and may not exist at startup.
        // Strict runtime startup validation treats this as deferred and retries
        // on request-time profile build with the effective cwd.
        error.FileNotFound, error.NotDir => return error.StrictDeferred,
        else => return error.StrictSetupFailed,
    };
    errdefer allocator.free(canonical);
    if (!isAncestorOrSelf(base_dir, canonical)) return error.StrictSetupFailed;
    return canonical;
}

fn isAncestorOrSelf(ancestor: []const u8, candidate: []const u8) bool {
    if (std.mem.eql(u8, ancestor, candidate)) return true;
    if (!std.mem.startsWith(u8, candidate, ancestor)) return false;
    if (candidate.len <= ancestor.len) return false;
    return candidate[ancestor.len] == std.fs.path.sep;
}

fn actionPatternMatches(pattern: []const u8, action: []const u8) bool {
    return policy_mod.pattern.globMatch(pattern, action);
}

fn containsWildcard(value: []const u8) bool {
    return std.mem.indexOfAny(u8, value, "*?") != null;
}

fn finalizeOwnedPathList(allocator: std.mem.Allocator, items: []const []const u8) !PathList {
    var total_len: usize = 0;
    for (items) |entry| total_len += entry.len;

    const paths_buf = try allocator.alloc(u8, total_len);
    errdefer allocator.free(paths_buf);
    const paths = try allocator.alloc([]const u8, items.len);
    errdefer allocator.free(paths);

    var offset: usize = 0;
    for (items, 0..) |entry, i| {
        const start = offset;
        const end = start + entry.len;
        @memcpy(paths_buf[start..end], entry);
        paths[i] = paths_buf[start..end];
        offset = end;
    }

    return .{ .paths_buf = paths_buf, .paths = paths };
}

fn contains(list: []const []const u8, needle: []const u8) bool {
    for (list) |item| {
        if (std.mem.eql(u8, item, needle)) return true;
    }
    return false;
}

test "buildExecProfile default policy includes whitelisted commands" {
    var profile = try buildExecProfile(std.testing.allocator, null, .{
        .action = "tool.exec",
        .cwd = null,
        .include_shell_paths = true,
    });
    defer profile.deinit();

    try std.testing.expect(profile.names.len > 0);
    try std.testing.expect(contains(profile.names, "ls"));
    try std.testing.expect(!profile.enforce_read);
    try std.testing.expect(!profile.enforce_write);
    try std.testing.expect(!profile.enforce_delete);
}

test "buildExecProfile respects deny policy for command name" {
    const json =
        \\{
        \\  "default":"allow",
        \\  "statements":[
        \\    {"effect":"deny","action":"tool.exec","command":"git *"}
        \\  ]
        \\}
    ;
    var policy = try policy_mod.parsePolicy(std.testing.allocator, json);
    defer policy.deinit();

    var profile = try buildExecProfile(std.testing.allocator, &policy, .{
        .action = "tool.exec",
    });
    defer profile.deinit();

    try std.testing.expect(!contains(profile.names, "git"));
}

test "buildExecProfile compiles fs write allow rules for strict mode" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.makePath("allowed");
    const base = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(base);

    const json =
        \\{
        \\  "default":"deny",
        \\  "statements":[
        \\    {"effect":"allow","action":"tool.exec","command":"python3 *"},
        \\    {"effect":"allow","action":"tool.fs.write","resource":"allowed/**"}
        \\  ]
        \\}
    ;
    var policy = try policy_mod.parsePolicy(std.testing.allocator, json);
    defer policy.deinit();

    var profile = try buildExecProfile(std.testing.allocator, &policy, .{
        .action = "tool.exec",
        .cwd = base,
        .include_shell_paths = true,
    });
    defer profile.deinit();

    try std.testing.expect(profile.enforce_write);
    try std.testing.expect(profile.fs_paths.len > 0);
}

test "buildExecProfile rejects unsupported deny-plus-allow file policy" {
    const json =
        \\{
        \\  "default":"deny",
        \\  "statements":[
        \\    {"effect":"allow","action":"tool.exec","command":"python3 *"},
        \\    {"effect":"allow","action":"tool.fs.write","resource":"src/**"},
        \\    {"effect":"deny","action":"tool.fs.write","resource":"src/private/**"}
        \\  ]
        \\}
    ;
    var policy = try policy_mod.parsePolicy(std.testing.allocator, json);
    defer policy.deinit();

    const result = buildExecProfile(std.testing.allocator, &policy, .{
        .action = "tool.exec",
    });
    try std.testing.expectError(error.StrictSetupFailed, result);
}
