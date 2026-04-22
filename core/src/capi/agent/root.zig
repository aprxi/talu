//! Slim C API exports for workspace tools and runtime policy helpers.
//!
//! This slim repo keeps the low-level agent tool/runtime surface used by the
//! server, while dropping the higher-level agent loop/backend exports.

const fs_api = @import("fs.zig");
const policy_api = @import("policy.zig");
const process_api = @import("process.zig");
const runtime_api = @import("runtime.zig");
const shell_api = @import("shell.zig");

pub const TaluFs = fs_api.TaluFs;
pub const TaluFsStat = fs_api.TaluFsStat;
pub const TaluProcess = process_api.TaluProcess;
pub const TaluShell = shell_api.TaluShell;
pub const TaluAgentPolicy = policy_api.TaluAgentPolicy;
pub const TaluAgentRuntimeMode = runtime_api.TaluAgentRuntimeMode;
pub const TaluSandboxBackend = runtime_api.TaluSandboxBackend;
pub const TaluCapabilityReport = runtime_api.TaluCapabilityReport;

pub export fn talu_agent_policy_create(
    json: ?[*]const u8,
    len: usize,
    out_policy: ?*?*TaluAgentPolicy,
) callconv(.c) i32 {
    return policy_api.talu_agent_policy_create(json, len, out_policy);
}

pub export fn talu_agent_policy_free(policy: ?*TaluAgentPolicy) callconv(.c) void {
    policy_api.talu_agent_policy_free(policy);
}

pub export fn talu_agent_policy_prepare_runtime(
    policy: ?*TaluAgentPolicy,
    cwd: ?[*:0]const u8,
) callconv(.c) i32 {
    return policy_api.talu_agent_policy_prepare_runtime(policy, cwd);
}

pub export fn talu_agent_policy_check_action(
    policy: ?*TaluAgentPolicy,
    action: ?[*:0]const u8,
    command: ?[*:0]const u8,
    cwd: ?[*:0]const u8,
    resource: ?[*:0]const u8,
    timeout_ms: u64,
    out_allowed: ?*bool,
    out_reason: ?*?[*]const u8,
    out_reason_len: ?*usize,
) callconv(.c) i32 {
    return policy_api.talu_agent_policy_check_action(
        policy,
        action,
        command,
        cwd,
        resource,
        timeout_ms,
        out_allowed,
        out_reason,
        out_reason_len,
    );
}

pub export fn talu_agent_policy_check_file(
    policy: ?*TaluAgentPolicy,
    action: ?[*:0]const u8,
    resource: ?[*:0]const u8,
    is_dir: bool,
    out_allowed: ?*bool,
) callconv(.c) i32 {
    return policy_api.talu_agent_policy_check_file(policy, action, resource, is_dir, out_allowed);
}

pub export fn talu_agent_policy_check_process(
    policy: ?*TaluAgentPolicy,
    action: ?[*:0]const u8,
    command: ?[*:0]const u8,
    cwd: ?[*:0]const u8,
    out_allowed: ?*bool,
) callconv(.c) i32 {
    return policy_api.talu_agent_policy_check_process(policy, action, command, cwd, out_allowed);
}

pub export fn talu_agent_policy_check_process_detailed(
    policy: ?*TaluAgentPolicy,
    action: ?[*:0]const u8,
    command: ?[*:0]const u8,
    cwd: ?[*:0]const u8,
    out_allowed: ?*bool,
    out_deny_reason: ?*c_int,
) callconv(.c) i32 {
    return policy_api.talu_agent_policy_check_process_detailed(
        policy,
        action,
        command,
        cwd,
        out_allowed,
        out_deny_reason,
    );
}

pub export fn talu_agent_policy_validate_strict_emulation(
    policy: ?*TaluAgentPolicy,
) callconv(.c) i32 {
    return policy_api.talu_agent_policy_validate_strict_emulation(policy);
}

pub export fn talu_agent_policy_strict_emulation_decisions(
    policy: ?*TaluAgentPolicy,
    cwd: ?[*:0]const u8,
    out_deny_descendant_exec: ?*bool,
    out_deny_write: ?*bool,
    out_allow_python_exec: ?*bool,
) callconv(.c) i32 {
    return policy_api.talu_agent_policy_strict_emulation_decisions(
        policy,
        cwd,
        out_deny_descendant_exec,
        out_deny_write,
        out_allow_python_exec,
    );
}

pub export fn talu_agent_runtime_validate_strict(
    policy: ?*TaluAgentPolicy,
    cwd: ?[*:0]const u8,
    sandbox_backend: c_int,
) callconv(.c) i32 {
    return runtime_api.talu_agent_runtime_validate_strict(policy, cwd, sandbox_backend);
}

pub export fn talu_agent_runtime_validate_strict_ext(
    sandbox_backend: c_int,
    strict_required: bool,
    run_probes: bool,
    cwd: ?[*:0]const u8,
    out_report: ?*runtime_api.TaluCapabilityReport,
) callconv(.c) i32 {
    return runtime_api.talu_agent_runtime_validate_strict_ext(
        sandbox_backend,
        strict_required,
        run_probes,
        cwd,
        out_report,
    );
}

pub export fn talu_fs_create(
    workspace_dir: ?[*:0]const u8,
    policy: ?*TaluAgentPolicy,
    out_handle: ?*?*TaluFs,
) callconv(.c) i32 {
    return fs_api.talu_fs_create(workspace_dir, policy, out_handle);
}

pub export fn talu_fs_free(handle: ?*TaluFs) callconv(.c) void {
    fs_api.talu_fs_free(handle);
}

pub export fn talu_fs_read(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    max_bytes: usize,
    out_content: ?*?[*]const u8,
    out_content_len: ?*usize,
    out_size: ?*u64,
    out_truncated: ?*bool,
) callconv(.c) i32 {
    return fs_api.talu_fs_read(handle, path, max_bytes, out_content, out_content_len, out_size, out_truncated);
}

pub export fn talu_fs_write(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    content: ?[*]const u8,
    content_len: usize,
    mkdir: bool,
    out_bytes_written: ?*usize,
) callconv(.c) i32 {
    return fs_api.talu_fs_write(handle, path, content, content_len, mkdir, out_bytes_written);
}

pub export fn talu_fs_edit(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    old_text: ?[*]const u8,
    old_len: usize,
    new_text: ?[*]const u8,
    new_len: usize,
    replace_all: bool,
    out_replacements: ?*usize,
) callconv(.c) i32 {
    return fs_api.talu_fs_edit(handle, path, old_text, old_len, new_text, new_len, replace_all, out_replacements);
}

pub export fn talu_fs_stat(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    out_stat: ?*TaluFsStat,
) callconv(.c) i32 {
    return fs_api.talu_fs_stat(handle, path, out_stat);
}

pub export fn talu_fs_list(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    glob: ?[*:0]const u8,
    recursive: bool,
    limit: usize,
    out_json: ?*?[*]const u8,
    out_json_len: ?*usize,
) callconv(.c) i32 {
    return fs_api.talu_fs_list(handle, path, glob, recursive, limit, out_json, out_json_len);
}

pub export fn talu_fs_remove(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    recursive: bool,
) callconv(.c) i32 {
    return fs_api.talu_fs_remove(handle, path, recursive);
}

pub export fn talu_fs_mkdir(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    recursive: bool,
) callconv(.c) i32 {
    return fs_api.talu_fs_mkdir(handle, path, recursive);
}

pub export fn talu_fs_rename(
    handle: ?*TaluFs,
    from: ?[*:0]const u8,
    to: ?[*:0]const u8,
) callconv(.c) i32 {
    return fs_api.talu_fs_rename(handle, from, to);
}

pub export fn talu_fs_free_string(ptr: ?[*]const u8, len: usize) callconv(.c) void {
    fs_api.talu_fs_free_string(ptr, len);
}

pub export fn talu_shell_exec(
    command: ?[*:0]const u8,
    cwd: ?[*:0]const u8,
    policy: ?*TaluAgentPolicy,
    runtime_mode: c_int,
    sandbox_backend: c_int,
    out_stdout: ?*?[*]const u8,
    out_stdout_len: ?*usize,
    out_stderr: ?*?[*]const u8,
    out_stderr_len: ?*usize,
    out_exit_code: ?*i32,
) callconv(.c) i32 {
    return shell_api.talu_shell_exec(
        command,
        cwd,
        policy,
        runtime_mode,
        sandbox_backend,
        out_stdout,
        out_stdout_len,
        out_stderr,
        out_stderr_len,
        out_exit_code,
    );
}

pub export fn talu_shell_exec_streaming(
    command: ?[*:0]const u8,
    cwd: ?[*:0]const u8,
    policy: ?*TaluAgentPolicy,
    runtime_mode: c_int,
    sandbox_backend: c_int,
    timeout_ms: u64,
    on_stdout: ?shell_api.StreamCallback,
    on_stdout_ctx: ?*anyopaque,
    on_stderr: ?shell_api.StreamCallback,
    on_stderr_ctx: ?*anyopaque,
    out_exit_code: ?*i32,
) callconv(.c) i32 {
    return shell_api.talu_shell_exec_streaming(
        command,
        cwd,
        policy,
        runtime_mode,
        sandbox_backend,
        timeout_ms,
        on_stdout,
        on_stdout_ctx,
        on_stderr,
        on_stderr_ctx,
        out_exit_code,
    );
}

pub export fn talu_shell_check_command(
    command: ?[*:0]const u8,
    out_allowed: ?*bool,
    out_reason: ?*?[*]const u8,
    out_reason_len: ?*usize,
) callconv(.c) i32 {
    return shell_api.talu_shell_check_command(command, out_allowed, out_reason, out_reason_len);
}

pub export fn talu_shell_default_policy_json(
    out_json: ?*?[*]const u8,
    out_json_len: ?*usize,
) callconv(.c) i32 {
    return shell_api.talu_shell_default_policy_json(out_json, out_json_len);
}

pub export fn talu_shell_normalize_command(
    command: ?[*:0]const u8,
    out_normalized: ?*?[*]const u8,
    out_normalized_len: ?*usize,
) callconv(.c) i32 {
    return shell_api.talu_shell_normalize_command(command, out_normalized, out_normalized_len);
}

pub export fn talu_shell_free_string(ptr: ?[*]const u8, len: usize) callconv(.c) void {
    shell_api.talu_shell_free_string(ptr, len);
}

pub export fn talu_shell_open(
    cols: u16,
    rows: u16,
    cwd: ?[*:0]const u8,
    policy: ?*TaluAgentPolicy,
    runtime_mode: c_int,
    sandbox_backend: c_int,
    out_shell: ?*?*TaluShell,
) callconv(.c) i32 {
    return shell_api.talu_shell_open(cols, rows, cwd, policy, runtime_mode, sandbox_backend, out_shell);
}

pub export fn talu_shell_close(shell_handle: ?*TaluShell) callconv(.c) void {
    shell_api.talu_shell_close(shell_handle);
}

pub export fn talu_shell_write(
    shell_handle: ?*TaluShell,
    data: ?[*]const u8,
    len: usize,
) callconv(.c) i32 {
    return shell_api.talu_shell_write(shell_handle, data, len);
}

pub export fn talu_shell_read(
    shell_handle: ?*TaluShell,
    buf: ?[*]u8,
    buf_len: usize,
    out_read: ?*usize,
) callconv(.c) i32 {
    return shell_api.talu_shell_read(shell_handle, buf, buf_len, out_read);
}

pub export fn talu_shell_resize(
    shell_handle: ?*TaluShell,
    cols: u16,
    rows: u16,
) callconv(.c) i32 {
    return shell_api.talu_shell_resize(shell_handle, cols, rows);
}

pub export fn talu_shell_signal(
    shell_handle: ?*TaluShell,
    sig: u8,
) callconv(.c) i32 {
    return shell_api.talu_shell_signal(shell_handle, sig);
}

pub export fn talu_shell_alive(shell_handle: ?*TaluShell) callconv(.c) bool {
    return shell_api.talu_shell_alive(shell_handle);
}

pub export fn talu_shell_scrollback(
    shell_handle: ?*TaluShell,
    out_data: ?*?[*]const u8,
    out_len: ?*usize,
) callconv(.c) i32 {
    return shell_api.talu_shell_scrollback(shell_handle, out_data, out_len);
}

pub export fn talu_process_open(
    command: ?[*:0]const u8,
    cwd: ?[*:0]const u8,
    policy: ?*TaluAgentPolicy,
    runtime_mode: c_int,
    sandbox_backend: c_int,
    out_process: ?*?*TaluProcess,
) callconv(.c) i32 {
    return process_api.talu_process_open(command, cwd, policy, runtime_mode, sandbox_backend, out_process);
}

pub export fn talu_process_close(process_handle: ?*TaluProcess) callconv(.c) void {
    process_api.talu_process_close(process_handle);
}

pub export fn talu_process_write(
    process_handle: ?*TaluProcess,
    data: ?[*]const u8,
    len: usize,
) callconv(.c) i32 {
    return process_api.talu_process_write(process_handle, data, len);
}

pub export fn talu_process_read(
    process_handle: ?*TaluProcess,
    buf: ?[*]u8,
    buf_len: usize,
    out_read: ?*usize,
) callconv(.c) i32 {
    return process_api.talu_process_read(process_handle, buf, buf_len, out_read);
}

pub export fn talu_process_signal(
    process_handle: ?*TaluProcess,
    sig: u8,
) callconv(.c) i32 {
    return process_api.talu_process_signal(process_handle, sig);
}

pub export fn talu_process_alive(process_handle: ?*TaluProcess) callconv(.c) bool {
    return process_api.talu_process_alive(process_handle);
}

pub export fn talu_process_exit_code(
    process_handle: ?*TaluProcess,
    out_code: ?*i32,
    out_has_code: ?*bool,
) callconv(.c) i32 {
    return process_api.talu_process_exit_code(process_handle, out_code, out_has_code);
}
