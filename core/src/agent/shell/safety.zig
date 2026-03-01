//! Command safety validation for shell execution.
//!
//! Provides a conservative whitelist of allowed commands and validates
//! shell command strings against it. Handles chain operators (`|`, `&&`,
//! `||`, `;`), env-var prefixes, wrapper commands (`sudo`, `env`, etc.),
//! absolute path normalization, and per-command dangerous-argument checks.
//!
//! Callers use `checkCommand()` as the main entry point. The whitelist
//! and deny rules can also be exported as IAM-style policy JSON via
//! `defaultPolicyJson()`.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Result of a safety check on a command string.
pub const CheckResult = struct {
    allowed: bool,
    /// Human-readable reason when `allowed` is false. Slice into static
    /// memory — caller does NOT own it.
    reason: ?[]const u8,
};

/// Read-only / informational commands allowed in shell mode.
///
/// Philosophy: whitelist is conservative. Users always see the command
/// and must confirm, so the whitelist prevents obviously destructive
/// executables while allowing useful inspection commands.
pub const ALLOWED_COMMANDS: []const []const u8 = &.{
    // filesystem browsing
    "ls",
    "exa",
    "eza",
    "tree",
    "find",
    "fd",
    "locate",
    "stat",
    "file",
    "df",
    "lsblk",
    "mount",
    // file reading
    "cat",
    "bat",
    "less",
    "more",
    "head",
    "tail",
    "wc",
    "nl",
    "xxd",
    "hexdump",
    "od",
    // search
    "grep",
    "rg",
    "ag",
    "ack",
    "sed",
    "awk",
    // text processing (read-oriented)
    "sort",
    "uniq",
    "cut",
    "tr",
    "paste",
    "column",
    "jq",
    "yq",
    "xq",
    "diff",
    "comm",
    "tee",
    // process / system info
    "ps",
    "top",
    "htop",
    "uptime",
    "free",
    "vmstat",
    "lscpu",
    "lsusb",
    "lspci",
    "uname",
    "hostname",
    "id",
    "whoami",
    "who",
    "w",
    "last",
    "groups",
    // network inspection
    "ip",
    "ifconfig",
    "ss",
    "netstat",
    "ping",
    "dig",
    "nslookup",
    "host",
    "traceroute",
    "curl",
    "wget",
    // package / language info
    "dpkg",
    "rpm",
    "apt",
    "pip",
    "python",
    "python3",
    "node",
    "npm",
    "cargo",
    "rustc",
    "zig",
    "go",
    "git",
    "gh",
    // misc
    "date",
    "cal",
    "echo",
    "printf",
    "env",
    "printenv",
    "which",
    "whereis",
    "type",
    "man",
    "help",
    "xargs",
    "basename",
    "dirname",
    "realpath",
    "readlink",
};

// =========================================================================
// Public API
// =========================================================================

/// Check whether a shell command string is allowed.
///
/// Splits on chain operators (`|`, `&&`, `||`, `;`), extracts the leading
/// executable from each segment (skipping env assignments and wrappers like
/// `sudo`), and validates against the whitelist. Also applies per-command
/// dangerous-argument rules.
pub fn checkCommand(command: []const u8) CheckResult {
    var iter = ChainIterator.init(command);
    while (iter.next()) |raw_segment| {
        const segment = trim(raw_segment);
        if (segment.len == 0) continue;

        const exe = leadingExecutable(segment);
        if (exe.len == 0) continue;

        if (!isAllowed(exe)) {
            return .{ .allowed = false, .reason = "command not in whitelist" };
        }

        if (checkDangerousArgs(exe, segment)) |reason| {
            return .{ .allowed = false, .reason = reason };
        }
    }
    return .{ .allowed = true, .reason = null };
}

/// Build the default IAM-style policy JSON containing the whitelist
/// (as allow statements) and dangerous-argument deny rules.
///
/// The caller owns the returned slice and must free it with `allocator`.
pub fn defaultPolicyJson(allocator: Allocator) ![]u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);

    const writer = buf.writer(allocator);

    try writer.writeAll("{\"default\":\"deny\",\"statements\":[");

    // Deny dangerous patterns first (explicit deny always wins in IAM)
    try writer.writeAll("{\"effect\":\"deny\",\"action\":\"find * -exec *\"},");
    try writer.writeAll("{\"effect\":\"deny\",\"action\":\"find * -execdir *\"}");

    // Allow each whitelisted command with wildcard args
    for (ALLOWED_COMMANDS) |cmd| {
        try writer.print(",{{\"effect\":\"allow\",\"action\":\"{s}\"}}", .{cmd});
        try writer.print(",{{\"effect\":\"allow\",\"action\":\"{s} *\"}}", .{cmd});
    }

    try writer.writeAll("]}");

    return buf.toOwnedSlice(allocator);
}

/// Normalize a command for policy evaluation: replace a leading absolute
/// path with its basename. `/bin/ls -la` becomes `ls -la`.
pub fn normalizeCommand(allocator: Allocator, command: []const u8) ![]u8 {
    if (command.len == 0 or command[0] != '/') {
        return allocator.dupe(u8, command);
    }

    // Find the first space to separate exe from args.
    if (std.mem.indexOfScalar(u8, command, ' ')) |space| {
        const exe = command[0..space];
        const rest = command[space..];
        const base = fileBasename(exe);
        const result = try allocator.alloc(u8, base.len + rest.len);
        @memcpy(result[0..base.len], base);
        @memcpy(result[base.len..], rest);
        return result;
    }
    return allocator.dupe(u8, fileBasename(command));
}

// =========================================================================
// Internal helpers
// =========================================================================

/// Check whether `exe` is in the whitelist.
fn isAllowed(exe: []const u8) bool {
    for (ALLOWED_COMMANDS) |cmd| {
        if (std.mem.eql(u8, cmd, exe)) return true;
    }
    return false;
}

/// Per-command argument restrictions for whitelisted commands that can
/// delegate execution (e.g. `find -exec`, `xargs rm`).
fn checkDangerousArgs(exe: []const u8, segment: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, exe, "find")) {
        // Scan tokens for -exec / -execdir
        var tok_iter = std.mem.tokenizeScalar(u8, segment, ' ');
        while (tok_iter.next()) |tok| {
            if (std.mem.eql(u8, tok, "-exec") or std.mem.eql(u8, tok, "-execdir")) {
                return "find with -exec/-execdir";
            }
        }
    } else if (std.mem.eql(u8, exe, "xargs")) {
        // xargs <cmd> — check that <cmd> is itself whitelisted.
        // First non-flag token after "xargs" is the delegated command.
        var tok_iter = std.mem.tokenizeScalar(u8, segment, ' ');
        // Skip to "xargs"
        while (tok_iter.next()) |tok| {
            if (std.mem.eql(u8, tok, "xargs")) break;
        }
        // Find delegated command (first non-flag token after xargs)
        while (tok_iter.next()) |tok| {
            if (tok.len > 0 and tok[0] != '-') {
                // This is the delegated command
                if (!isAllowed(tok)) {
                    return "xargs delegates to non-whitelisted command";
                }
                break;
            }
        }
    }
    return null;
}

/// Extract the leading executable from a shell segment, skipping:
/// - leading env assignments (`FOO=bar`)
/// - `sudo`, `env`, `nice`, `nohup`, `time` prefixes
///
/// Absolute paths are resolved to their basename:
/// `/bin/ls` → `ls`, `/usr/bin/git` → `git`.
pub fn leadingExecutable(segment: []const u8) []const u8 {
    var tok_iter = std.mem.tokenizeScalar(u8, segment, ' ');
    while (tok_iter.next()) |tok| {
        // Skip env assignments (KEY=VALUE)
        if (std.mem.indexOfScalar(u8, tok, '=') != null and
            tok[0] != '-' and tok[0] != '/')
        {
            continue;
        }
        // Skip common wrappers (check after basename for absolute paths)
        const name = fileBasename(tok);
        if (std.mem.eql(u8, name, "sudo") or
            std.mem.eql(u8, name, "env") or
            std.mem.eql(u8, name, "nice") or
            std.mem.eql(u8, name, "nohup") or
            std.mem.eql(u8, name, "time"))
        {
            continue;
        }
        return name;
    }
    return "";
}

/// Return the basename of a path: `/usr/bin/git` → `git`, `ls` → `ls`.
fn fileBasename(path: []const u8) []const u8 {
    if (std.mem.lastIndexOfScalar(u8, path, '/')) |i| {
        return path[i + 1 ..];
    }
    return path;
}

/// Trim leading and trailing ASCII whitespace.
fn trim(s: []const u8) []const u8 {
    return std.mem.trim(u8, s, &std.ascii.whitespace);
}

/// Iterator that splits a command string on shell chain operators:
/// `|`, `&&`, `||`, `;`. Does NOT handle subshells / $() — intentionally
/// conservative.
const ChainIterator = struct {
    buf: []const u8,
    pos: usize,

    fn init(buf: []const u8) ChainIterator {
        return .{ .buf = buf, .pos = 0 };
    }

    fn next(self: *ChainIterator) ?[]const u8 {
        if (self.pos >= self.buf.len) return null;

        const start = self.pos;
        while (self.pos < self.buf.len) {
            const b = self.buf[self.pos];
            if (b == '|') {
                const seg = self.buf[start..self.pos];
                if (self.pos + 1 < self.buf.len and self.buf[self.pos + 1] == '|') {
                    self.pos += 2; // ||
                } else {
                    self.pos += 1; // |
                }
                return seg;
            } else if (b == '&' and self.pos + 1 < self.buf.len and self.buf[self.pos + 1] == '&') {
                const seg = self.buf[start..self.pos];
                self.pos += 2;
                return seg;
            } else if (b == ';') {
                const seg = self.buf[start..self.pos];
                self.pos += 1;
                return seg;
            }
            self.pos += 1;
        }
        // Remaining tail
        if (start < self.buf.len) {
            return self.buf[start..];
        }
        return null;
    }
};

// =========================================================================
// Tests
// =========================================================================

test "checkCommand allows simple whitelisted command" {
    const result = checkCommand("ls");
    try std.testing.expect(result.allowed);
    try std.testing.expect(result.reason == null);
}

test "checkCommand allows whitelisted command with args" {
    const result = checkCommand("git status --short");
    try std.testing.expect(result.allowed);
}

test "checkCommand blocks non-whitelisted command" {
    const result = checkCommand("rm -rf /");
    try std.testing.expect(!result.allowed);
    try std.testing.expectEqualStrings("command not in whitelist", result.reason.?);
}

test "checkCommand blocks second command in chain" {
    const result = checkCommand("ls && rm -rf /");
    try std.testing.expect(!result.allowed);
    try std.testing.expectEqualStrings("command not in whitelist", result.reason.?);
}

test "checkCommand blocks piped non-whitelisted command" {
    const result = checkCommand("cat foo | dd if=/dev/zero");
    try std.testing.expect(!result.allowed);
}

test "checkCommand allows piped whitelisted commands" {
    const result = checkCommand("cat foo | grep bar | sort");
    try std.testing.expect(result.allowed);
}

test "checkCommand blocks find -exec" {
    const result = checkCommand("find . -exec rm {} ;");
    try std.testing.expect(!result.allowed);
    try std.testing.expectEqualStrings("find with -exec/-execdir", result.reason.?);
}

test "checkCommand blocks find -execdir" {
    const result = checkCommand("find . -execdir sh -c 'echo {}' ;");
    try std.testing.expect(!result.allowed);
    try std.testing.expectEqualStrings("find with -exec/-execdir", result.reason.?);
}

test "checkCommand allows find without -exec" {
    const result = checkCommand("find . -name '*.zig'");
    try std.testing.expect(result.allowed);
}

test "checkCommand blocks xargs with non-whitelisted delegated command" {
    const result = checkCommand("ls | xargs rm");
    try std.testing.expect(!result.allowed);
    try std.testing.expectEqualStrings("xargs delegates to non-whitelisted command", result.reason.?);
}

test "checkCommand allows xargs with whitelisted delegated command" {
    const result = checkCommand("ls | xargs grep pattern");
    try std.testing.expect(result.allowed);
}

test "checkCommand strips env assignments" {
    const result = checkCommand("FOO=bar ls -la");
    try std.testing.expect(result.allowed);
}

test "checkCommand strips sudo wrapper" {
    const result = checkCommand("sudo git status");
    try std.testing.expect(result.allowed);
}

test "checkCommand strips env wrapper" {
    const result = checkCommand("env git status");
    try std.testing.expect(result.allowed);
}

test "checkCommand strips multiple wrappers" {
    const result = checkCommand("sudo env nice git status");
    try std.testing.expect(result.allowed);
}

test "checkCommand handles absolute path" {
    const result = checkCommand("/usr/bin/git status");
    try std.testing.expect(result.allowed);
}

test "checkCommand handles semicolon chain" {
    const result = checkCommand("echo hello; ls");
    try std.testing.expect(result.allowed);
}

test "checkCommand handles or chain" {
    const result = checkCommand("echo ok || echo fallback");
    try std.testing.expect(result.allowed);
}

test "checkCommand blocks dangerous in or chain" {
    const result = checkCommand("echo ok || rm -rf /");
    try std.testing.expect(!result.allowed);
}

test "checkCommand allows empty command" {
    const result = checkCommand("");
    try std.testing.expect(result.allowed);
}

test "leadingExecutable simple" {
    try std.testing.expectEqualStrings("ls", leadingExecutable("ls -la"));
}

test "leadingExecutable with env var" {
    try std.testing.expectEqualStrings("git", leadingExecutable("FOO=bar git status"));
}

test "leadingExecutable with absolute path" {
    try std.testing.expectEqualStrings("git", leadingExecutable("/usr/bin/git status"));
}

test "leadingExecutable with sudo" {
    try std.testing.expectEqualStrings("git", leadingExecutable("sudo git log"));
}

test "leadingExecutable empty" {
    try std.testing.expectEqualStrings("", leadingExecutable(""));
}

test "normalizeCommand with absolute path" {
    const allocator = std.testing.allocator;
    const result = try normalizeCommand(allocator, "/bin/ls -la");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("ls -la", result);
}

test "normalizeCommand with relative command" {
    const allocator = std.testing.allocator;
    const result = try normalizeCommand(allocator, "git status");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("git status", result);
}

test "normalizeCommand bare absolute path" {
    const allocator = std.testing.allocator;
    const result = try normalizeCommand(allocator, "/usr/bin/git");
    defer allocator.free(result);
    try std.testing.expectEqualStrings("git", result);
}

test "defaultPolicyJson produces valid JSON" {
    const allocator = std.testing.allocator;
    const json = try defaultPolicyJson(allocator);
    defer allocator.free(json);

    // Must start with the expected structure
    try std.testing.expect(std.mem.startsWith(u8, json, "{\"default\":\"deny\",\"statements\":["));
    // Must contain deny rules
    try std.testing.expect(std.mem.indexOf(u8, json, "\"effect\":\"deny\",\"action\":\"find * -exec *\"") != null);
    // Must contain allow rules
    try std.testing.expect(std.mem.indexOf(u8, json, "\"effect\":\"allow\",\"action\":\"ls\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"effect\":\"allow\",\"action\":\"git *\"") != null);
    // Must end with ]}
    try std.testing.expect(std.mem.endsWith(u8, json, "]}"));
}

test "ChainIterator splits pipe" {
    var iter = ChainIterator.init("ls | grep foo");
    try std.testing.expectEqualStrings("ls ", iter.next().?);
    try std.testing.expectEqualStrings(" grep foo", iter.next().?);
    try std.testing.expect(iter.next() == null);
}

test "ChainIterator splits and-and" {
    var iter = ChainIterator.init("ls && echo done");
    try std.testing.expectEqualStrings("ls ", iter.next().?);
    try std.testing.expectEqualStrings(" echo done", iter.next().?);
    try std.testing.expect(iter.next() == null);
}

test "ChainIterator splits or-or" {
    var iter = ChainIterator.init("false || true");
    try std.testing.expectEqualStrings("false ", iter.next().?);
    try std.testing.expectEqualStrings(" true", iter.next().?);
    try std.testing.expect(iter.next() == null);
}

test "ChainIterator splits semicolon" {
    var iter = ChainIterator.init("echo a; echo b");
    try std.testing.expectEqualStrings("echo a", iter.next().?);
    try std.testing.expectEqualStrings(" echo b", iter.next().?);
    try std.testing.expect(iter.next() == null);
}

test "ChainIterator mixed operators" {
    var iter = ChainIterator.init("a | b && c; d || e");
    try std.testing.expectEqualStrings("a ", iter.next().?);
    try std.testing.expectEqualStrings(" b ", iter.next().?);
    try std.testing.expectEqualStrings(" c", iter.next().?);
    try std.testing.expectEqualStrings(" d ", iter.next().?);
    try std.testing.expectEqualStrings(" e", iter.next().?);
    try std.testing.expect(iter.next() == null);
}

test "ChainIterator single command" {
    var iter = ChainIterator.init("ls -la");
    try std.testing.expectEqualStrings("ls -la", iter.next().?);
    try std.testing.expect(iter.next() == null);
}
