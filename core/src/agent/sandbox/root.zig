//! Runtime sandbox orchestration for agent shell/process execution.
//!
//! Strict mode is fail-closed: if backend capabilities are unavailable,
//! session/process launch returns a typed error.

const builtin = @import("builtin");
pub const profile = @import("profile.zig");
pub const landlock = @import("landlock.zig");
pub const launcher = @import("launcher.zig");
pub const linux_ns = @import("linux_ns.zig");
pub const mounts = @import("mounts.zig");
pub const seccomp = @import("seccomp.zig");
pub const limits = @import("limits.zig");
pub const helpers = @import("helpers.zig");
pub const env = @import("env.zig");

pub const RuntimeMode = enum(u8) {
    host = 0,
    strict = 1,
};

pub const Backend = enum(u8) {
    linux_local = 0,
    oci = 1,
    apple_container = 2,
};

pub const StrictRuntimeConfig = struct {
    mode: RuntimeMode,
    backend: Backend,
    exec_profile: ?*const profile.ExecProfile = null,
    namespace_options: linux_ns.NamespaceOptions = .{},
    mount_options: mounts.MountOptions = .{},
    runtime_limits: limits.RuntimeLimits = .{},
    env_config: env.EnvConfig = .{},
    enable_seccomp: bool = true,

    /// Construct a StrictRuntimeConfig with strict-mode defaults enabled.
    ///
    /// Enables mount/user/ipc namespaces, mount tree, resource limits, env
    /// sanitization, and seccomp. PID namespace is currently disabled by
    /// default until strict launch paths support the required post-unshare
    /// re-fork sequence.
    /// The exec_profile must be provided separately.
    pub fn defaultStrict(backend: Backend, workspace: ?[]const u8) StrictRuntimeConfig {
        return .{
            .mode = .strict,
            .backend = backend,
            .namespace_options = .{
                .enable_mount_ns = true,
                .enable_pid_ns = false,
                .enable_user_ns = true,
                .enable_ipc_ns = true,
            },
            .mount_options = .{
                .make_root_private = true,
                .enable_mount_tree = true,
                .workspace_path = workspace,
            },
            .runtime_limits = limits.RuntimeLimits.defaultStrict(),
            .env_config = .{},
            .enable_seccomp = true,
        };
    }
};

pub fn validate(mode: RuntimeMode, backend: Backend) !void {
    if (mode == .host) return;
    switch (backend) {
        .linux_local => {
            if (builtin.os.tag != .linux) return error.StrictUnavailable;
        },
        .oci, .apple_container => return error.StrictUnavailable,
    }
}

/// Apply strict runtime controls in a just-forked child before exec.
///
/// Call order: namespaces -> mounts -> env -> limits -> landlock -> seccomp.
/// Each step narrows the child's privileges; later steps depend on earlier
/// ones (e.g. mounts require mount namespace, landlock paths must exist in
/// the mount tree).
pub fn applyInChild(config: StrictRuntimeConfig) !void {
    if (config.mode == .host) return;
    switch (config.backend) {
        .linux_local => {
            if (builtin.os.tag != .linux) return error.StrictUnavailable;
            const prof = config.exec_profile orelse return error.StrictSetupFailed;
            try linux_ns.apply(config.namespace_options);
            try mounts.apply(config.mount_options);
            try env.sanitize(config.env_config);
            try limits.apply(config.runtime_limits);
            try landlock.applyExecProfile(prof);
            if (config.enable_seccomp) try seccomp.applyDefaultFilter();
        },
        .oci, .apple_container => return error.StrictUnavailable,
    }
}

test "validate host accepts all backends" {
    try validate(.host, .linux_local);
    try validate(.host, .oci);
    try validate(.host, .apple_container);
}

test "validate strict rejects non-linux-local backends" {
    try std.testing.expectError(error.StrictUnavailable, validate(.strict, .oci));
    try std.testing.expectError(error.StrictUnavailable, validate(.strict, .apple_container));
}

test "defaultStrict enables all isolation layers" {
    const config = StrictRuntimeConfig.defaultStrict(.linux_local, "/tmp/workspace");
    try std.testing.expectEqual(RuntimeMode.strict, config.mode);
    try std.testing.expect(config.namespace_options.enable_mount_ns);
    try std.testing.expect(!config.namespace_options.enable_pid_ns);
    try std.testing.expect(config.namespace_options.enable_user_ns);
    try std.testing.expect(config.namespace_options.enable_ipc_ns);
    try std.testing.expect(config.mount_options.enable_mount_tree);
    try std.testing.expect(config.runtime_limits.max_memory_bytes != null);
    try std.testing.expect(config.enable_seccomp);
}

const std = @import("std");
