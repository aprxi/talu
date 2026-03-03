//! Seccomp filter for strict sandbox launches.
//!
//! The default profile is allow-by-default with explicit ERRNO denials for
//! high-risk syscalls that are not required for normal tool execution.

const std = @import("std");
const builtin = @import("builtin");

const c = if (builtin.os.tag == .linux) @cImport({
    @cInclude("errno.h");
    @cInclude("linux/filter.h");
    @cInclude("linux/seccomp.h");
    @cInclude("sys/prctl.h");
    @cInclude("sys/syscall.h");
}) else struct {
    const struct_sock_filter = extern struct {
        code: u16,
        jt: u8,
        jf: u8,
        k: u32,
    };
    const struct_sock_fprog = extern struct {
        len: u16,
        filter: [*]struct_sock_filter,
    };
    const struct_seccomp_data = extern struct {
        nr: i32,
    };
    const SYS_ptrace: u32 = 0;
    const SYS_mount: u32 = 0;
    const SYS_umount2: u32 = 0;
    const SYS_setns: u32 = 0;
    const SYS_unshare: u32 = 0;
    const SYS_pivot_root: u32 = 0;
    const SYS_chroot: u32 = 0;
    const SYS_kexec_load: u32 = 0;
    const SYS_init_module: u32 = 0;
    const SYS_finit_module: u32 = 0;
    const SYS_delete_module: u32 = 0;
    const SYS_reboot: u32 = 0;
    const BPF_LD: u32 = 0;
    const BPF_W: u32 = 0;
    const BPF_ABS: u32 = 0;
    const BPF_JMP: u32 = 0;
    const BPF_JEQ: u32 = 0;
    const BPF_K: u32 = 0;
    const BPF_RET: u32 = 0;
    const SECCOMP_RET_ERRNO: u32 = 0;
    const SECCOMP_RET_ALLOW: u32 = 0;
    const SECCOMP_MODE_FILTER: u32 = 0;
    const PR_SET_NO_NEW_PRIVS: u32 = 0;
    const PR_SET_SECCOMP: u32 = 0;
    const EPERM: u32 = 1;
    fn prctl(_: anytype, _: anytype, _: anytype, _: anytype, _: anytype) std.c.c_int {
        return 0;
    }
};

const denied_syscalls = if (builtin.os.tag == .linux)
    [_]u32{
        c.SYS_ptrace,
        c.SYS_mount,
        c.SYS_umount2,
        c.SYS_setns,
        c.SYS_unshare,
        c.SYS_pivot_root,
        c.SYS_chroot,
        c.SYS_kexec_load,
        c.SYS_init_module,
        c.SYS_finit_module,
        c.SYS_delete_module,
        c.SYS_reboot,
    }
else
    [_]u32{};

const program_len = 1 + denied_syscalls.len * 2 + 1;

pub fn applyDefaultFilter() !void {
    if (builtin.os.tag != .linux) return error.StrictUnavailable;

    var prog_buf = buildProgram();
    var prog = c.struct_sock_fprog{
        .len = prog_buf.len,
        .filter = &prog_buf[0],
    };

    if (c.prctl(
        c.PR_SET_NO_NEW_PRIVS,
        @as(c_ulong, 1),
        @as(c_ulong, 0),
        @as(c_ulong, 0),
        @as(c_ulong, 0),
    ) != 0) return mapErrnoToError();

    if (c.prctl(
        c.PR_SET_SECCOMP,
        @as(c_ulong, c.SECCOMP_MODE_FILTER),
        @intFromPtr(&prog),
        @as(c_ulong, 0),
        @as(c_ulong, 0),
    ) != 0) return mapErrnoToError();
}

fn buildProgram() [program_len]c.struct_sock_filter {
    var program: [program_len]c.struct_sock_filter = undefined;
    var i: usize = 0;

    // Load syscall number from seccomp_data.nr.
    program[i] = stmt(
        @intCast(c.BPF_LD | c.BPF_W | c.BPF_ABS),
        @intCast(@offsetOf(c.struct_seccomp_data, "nr")),
    );
    i += 1;

    for (denied_syscalls) |nr| {
        // if (nr == denied) -> next insn, else skip deny-return.
        program[i] = jump(@intCast(c.BPF_JMP | c.BPF_JEQ | c.BPF_K), nr, 0, 1);
        i += 1;
        program[i] = stmt(
            @intCast(c.BPF_RET | c.BPF_K),
            @intCast(c.SECCOMP_RET_ERRNO | @as(u32, c.EPERM)),
        );
        i += 1;
    }

    program[i] = stmt(@intCast(c.BPF_RET | c.BPF_K), @intCast(c.SECCOMP_RET_ALLOW));
    i += 1;
    std.debug.assert(i == program_len);
    return program;
}

fn stmt(code: u16, k: u32) c.struct_sock_filter {
    return .{
        .code = code,
        .jt = 0,
        .jf = 0,
        .k = k,
    };
}

fn jump(code: u16, k: u32, jt: u8, jf: u8) c.struct_sock_filter {
    return .{
        .code = code,
        .jt = jt,
        .jf = jf,
        .k = k,
    };
}

fn mapErrnoToError() anyerror {
    const errno_value = std.c._errno().*;
    const err: std.posix.E = @enumFromInt(@as(u16, @intCast(errno_value)));
    return switch (err) {
        .PERM, .NOSYS, .OPNOTSUPP, .INVAL => error.StrictUnavailable,
        else => error.StrictSetupFailed,
    };
}

test "buildProgram ends with allow return" {
    const prog = buildProgram();
    const last = prog[prog.len - 1];
    try std.testing.expectEqual(
        @as(u16, @intCast(c.BPF_RET | c.BPF_K)),
        last.code,
    );
    try std.testing.expectEqual(@as(u32, @intCast(c.SECCOMP_RET_ALLOW)), last.k);
}
