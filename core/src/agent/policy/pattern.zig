//! Glob pattern matching helpers for policy statements.
//!
//! `globMatch` is used for action/command/cwd patterns.
//! `pathMatch` is used for file-resource patterns and supports `**`.

const std = @import("std");

/// Match a glob pattern against a text string.
///
/// Supported wildcards:
/// - `*` matches zero or more characters
/// - `?` matches exactly one character
///
/// Pattern matching is case-sensitive.
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
        } else if (pi < pattern.len and (pattern[pi] == '?' or pattern[pi] == text[ti])) {
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

/// Match a policy resource path pattern against a normalized workspace-relative path.
///
/// Supported wildcards:
/// - `*` matches zero or more non-separator characters
/// - `**` matches zero or more characters including separators
/// - `?` matches one non-separator character
///
/// Pattern details:
/// - leading `/` anchors at workspace root
/// - trailing `/` means directory entry only
pub fn pathMatch(pattern: []const u8, path: []const u8, is_dir: bool) bool {
    if (pattern.len == 0) return false;

    var pat_start: usize = 0;
    var pat_end: usize = pattern.len;

    // Leading "/" anchors to workspace root.
    if (pattern[0] == '/') {
        pat_start = 1;
    }

    // Trailing "/" targets directory entry only.
    if (pat_end > pat_start and pattern[pat_end - 1] == '/') {
        if (!is_dir) return false;
        pat_end -= 1;
    }

    const pat = pattern[pat_start..pat_end];
    if (pat.len == 0) return path.len == 0 or std.mem.eql(u8, path, ".");

    // `dir/**` should include the directory entry itself (`dir`).
    if (pat.len >= 3 and std.mem.endsWith(u8, pat, "/**")) {
        const base = pat[0 .. pat.len - 3];
        if (base.len == 0) return true;
        if (std.mem.eql(u8, path, base)) return true;
    }

    return pathMatchRec(pat, 0, path, 0);
}

fn pathMatchRec(pattern: []const u8, p_idx: usize, text: []const u8, t_idx: usize) bool {
    var pi = p_idx;
    var ti = t_idx;

    while (pi < pattern.len) {
        const pc = pattern[pi];
        if (pc == '*') {
            // `**` matches across separators.
            if (pi + 1 < pattern.len and pattern[pi + 1] == '*') {
                var next = pi + 2;
                // Collapse repeated stars (`***` behaves like `**`).
                while (next < pattern.len and pattern[next] == '*') : (next += 1) {}
                if (next == pattern.len) return true;

                var consume = ti;
                while (consume <= text.len) : (consume += 1) {
                    if (pathMatchRec(pattern, next, text, consume)) return true;
                }
                return false;
            }

            // `*` matches within a single path segment.
            var consume = ti;
            while (consume <= text.len) : (consume += 1) {
                if (consume > ti and isSeparator(text[consume - 1])) break;
                if (pathMatchRec(pattern, pi + 1, text, consume)) return true;
            }
            return false;
        }

        if (ti >= text.len) return false;

        if (pc == '?') {
            if (isSeparator(text[ti])) return false;
            pi += 1;
            ti += 1;
            continue;
        }

        if (isSeparator(pc)) {
            if (!isSeparator(text[ti])) return false;
            pi += 1;
            ti += 1;
            continue;
        }

        if (pc != text[ti]) return false;
        pi += 1;
        ti += 1;
    }

    return ti == text.len;
}

fn isSeparator(ch: u8) bool {
    return ch == '/' or ch == '\\';
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
    // The policy evaluator handles this via action-match fallback.
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

test "pathMatch exact and anchored paths" {
    try std.testing.expect(pathMatch("src/main.zig", "src/main.zig", false));
    try std.testing.expect(pathMatch("/src/main.zig", "src/main.zig", false));
    try std.testing.expect(!pathMatch("/src/main.zig", "nested/src/main.zig", false));
}

test "pathMatch supports * and **" {
    try std.testing.expect(pathMatch("src/*.zig", "src/main.zig", false));
    try std.testing.expect(!pathMatch("src/*.zig", "src/deep/main.zig", false));
    try std.testing.expect(pathMatch("src/**", "src/deep/main.zig", false));
    try std.testing.expect(pathMatch("src/**", "src", true));
    try std.testing.expect(pathMatch("repo/**", "repo", true));
}

test "pathMatch directory intent requires is_dir" {
    try std.testing.expect(pathMatch("src/", "src", true));
    try std.testing.expect(!pathMatch("src/", "src/file.txt", false));
    try std.testing.expect(!pathMatch("src/", "src", false));
}
