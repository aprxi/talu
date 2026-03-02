//! Glob pattern matching for policy action strings.
//!
//! Supports only `*` as a wildcard (matches zero or more characters).
//! No `?`, no character classes. Case-sensitive.
//!
//! Used by the policy evaluator to match tool call actions against
//! statement patterns like `"ls *"`, `"find * -exec *"`, `"git push *"`.

const std = @import("std");

/// Match a glob pattern against a text string.
///
/// The pattern supports `*` as a wildcard that matches zero or more
/// characters. All other characters are matched literally (case-sensitive).
///
/// Uses a greedy backtracking algorithm: each `*` tries to match as
/// few characters as possible first, then backtracks to consume more
/// if the rest of the pattern fails.
pub fn globMatch(pattern: []const u8, text: []const u8) bool {
    var pi: usize = 0;
    var ti: usize = 0;

    // Saved positions for backtracking on `*`.
    var star_pi: ?usize = null;
    var star_ti: usize = 0;

    while (ti < text.len) {
        if (pi < pattern.len and pattern[pi] == '*') {
            // Record star position for backtracking.
            star_pi = pi;
            star_ti = ti;
            pi += 1;
        } else if (pi < pattern.len and pattern[pi] == text[ti]) {
            // Literal match — advance both.
            pi += 1;
            ti += 1;
        } else if (star_pi) |sp| {
            // Mismatch — backtrack: let the last `*` consume one more char.
            pi = sp + 1;
            star_ti += 1;
            ti = star_ti;
        } else {
            return false;
        }
    }

    // Consume trailing `*` in pattern (they match empty).
    while (pi < pattern.len and pattern[pi] == '*') {
        pi += 1;
    }

    return pi == pattern.len;
}

// =============================================================================
// Tests
// =============================================================================

test "globMatch exact match" {
    try std.testing.expect(globMatch("ls", "ls"));
    try std.testing.expect(!globMatch("ls", "lsblk"));
    try std.testing.expect(!globMatch("lsblk", "ls"));
}

test "globMatch trailing wildcard" {
    try std.testing.expect(globMatch("ls *", "ls -la"));
    try std.testing.expect(globMatch("ls *", "ls /tmp"));
    try std.testing.expect(globMatch("ls *", "ls "));
    // "ls *" does not match bare "ls" at the glob level (the space is literal).
    // The policy evaluator handles this via matchesStatement().
    try std.testing.expect(!globMatch("ls *", "ls"));
    try std.testing.expect(!globMatch("ls *", "lsblk"));
}

test "globMatch star matches empty" {
    try std.testing.expect(globMatch("*", ""));
    try std.testing.expect(globMatch("*", "anything"));
    try std.testing.expect(globMatch("ls*", "ls"));
    try std.testing.expect(globMatch("ls*", "lsblk"));
}

test "globMatch subcommand patterns" {
    try std.testing.expect(globMatch("git *", "git show HEAD"));
    try std.testing.expect(globMatch("git *", "git log --oneline"));
    try std.testing.expect(!globMatch("git *", "gitk"));

    try std.testing.expect(globMatch("git push *", "git push origin main"));
    try std.testing.expect(!globMatch("git push *", "git pull origin main"));
}

test "globMatch middle wildcard" {
    // "find * -exec *" matches "find . -exec rm {} ;"
    try std.testing.expect(globMatch("find * -exec *", "find . -exec rm {} ;"));
    try std.testing.expect(globMatch("find * -exec *", "find /tmp -type f -exec cat {} ;"));
    // Does NOT match "find . -name foo" (no -exec)
    try std.testing.expect(!globMatch("find * -exec *", "find . -name foo"));
}

test "globMatch multiple wildcards" {
    try std.testing.expect(globMatch("*foo*", "foobar"));
    try std.testing.expect(globMatch("*foo*", "bazfoo"));
    try std.testing.expect(globMatch("*foo*", "bazfoobar"));
    try std.testing.expect(!globMatch("*foo*", "bar"));
}

test "globMatch case sensitive" {
    try std.testing.expect(!globMatch("LS *", "ls -la"));
    try std.testing.expect(!globMatch("ls *", "LS -LA"));
}

test "globMatch empty pattern and text" {
    try std.testing.expect(globMatch("", ""));
    try std.testing.expect(!globMatch("", "notempty"));
    try std.testing.expect(!globMatch("notempty", ""));
}
