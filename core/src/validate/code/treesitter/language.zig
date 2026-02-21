//! Language grammar registry.
//!
//! Maps language identifiers to compiled tree-sitter grammars and their
//! associated highlight query patterns.
//!
//! Thread safety: Immutable after program load. Safe to share across threads.

const std = @import("std");
const c = @import("c.zig").c;

pub const Language = enum {
    python,
    javascript,
    typescript,
    rust,
    go,
    c_lang,
    zig_lang,
    json,
    html,
    css,
    bash,

    /// Returns the compiled TSLanguage pointer for this language.
    pub fn grammar(self: Language) *const c.TSLanguage {
        return switch (self) {
            .python => tree_sitter_python(),
            .javascript => tree_sitter_javascript(),
            .typescript => tree_sitter_typescript(),
            .rust => tree_sitter_rust(),
            .go => tree_sitter_go(),
            .c_lang => tree_sitter_c(),
            .zig_lang => tree_sitter_zig(),
            .json => tree_sitter_json(),
            .html => tree_sitter_html(),
            .css => tree_sitter_css(),
            .bash => tree_sitter_bash(),
        };
    }

    /// Parse a language name string into a Language enum value.
    /// Accepts canonical names and common aliases.
    pub fn fromString(lang_name: []const u8) ?Language {
        const map = std.StaticStringMap(Language).initComptime(.{
            .{ "python", .python },
            .{ "py", .python },
            .{ "javascript", .javascript },
            .{ "js", .javascript },
            .{ "jsx", .javascript },
            .{ "typescript", .typescript },
            .{ "ts", .typescript },
            .{ "tsx", .typescript },
            .{ "rust", .rust },
            .{ "rs", .rust },
            .{ "go", .go },
            .{ "golang", .go },
            .{ "c", .c_lang },
            .{ "zig", .zig_lang },
            .{ "json", .json },
            .{ "html", .html },
            .{ "css", .css },
            .{ "bash", .bash },
            .{ "sh", .bash },
            .{ "shell", .bash },
        });
        return map.get(lang_name);
    }

    /// Returns the canonical name for this language.
    pub fn name(self: Language) []const u8 {
        return switch (self) {
            .python => "python",
            .javascript => "javascript",
            .typescript => "typescript",
            .rust => "rust",
            .go => "go",
            .c_lang => "c",
            .zig_lang => "zig",
            .json => "json",
            .html => "html",
            .css => "css",
            .bash => "bash",
        };
    }

    /// Detect language from a file extension or filename.
    /// Accepts "foo.py", ".py", or just "py".
    pub fn fromFilename(filename: []const u8) ?Language {
        const ext = blk: {
            var i: usize = filename.len;
            while (i > 0) {
                i -= 1;
                if (filename[i] == '.') break :blk filename[i + 1 ..];
            }
            break :blk filename;
        };
        return fromString(ext);
    }

    /// Returns the embedded highlight query patterns (.scm) for this language.
    pub fn highlightQuery(self: Language) []const u8 {
        return switch (self) {
            .python => @embedFile("queries/python.scm"),
            .javascript => @embedFile("queries/javascript.scm"),
            .typescript => @embedFile("queries/typescript.scm"),
            .rust => @embedFile("queries/rust.scm"),
            .go => @embedFile("queries/go.scm"),
            .c_lang => @embedFile("queries/c.scm"),
            .zig_lang => @embedFile("queries/zig.scm"),
            .json => @embedFile("queries/json.scm"),
            .html => @embedFile("queries/html.scm"),
            .css => @embedFile("queries/css.scm"),
            .bash => @embedFile("queries/bash.scm"),
        };
    }
};

// Extern declarations for compiled grammar entry points.
// These symbols are provided by the static library built from each grammar's parser.c.
extern fn tree_sitter_python() callconv(.c) *const c.TSLanguage;
extern fn tree_sitter_javascript() callconv(.c) *const c.TSLanguage;
extern fn tree_sitter_typescript() callconv(.c) *const c.TSLanguage;
extern fn tree_sitter_rust() callconv(.c) *const c.TSLanguage;
extern fn tree_sitter_go() callconv(.c) *const c.TSLanguage;
extern fn tree_sitter_c() callconv(.c) *const c.TSLanguage;
extern fn tree_sitter_zig() callconv(.c) *const c.TSLanguage;
extern fn tree_sitter_json() callconv(.c) *const c.TSLanguage;
extern fn tree_sitter_html() callconv(.c) *const c.TSLanguage;
extern fn tree_sitter_css() callconv(.c) *const c.TSLanguage;
extern fn tree_sitter_bash() callconv(.c) *const c.TSLanguage;

// =============================================================================
// Tests
// =============================================================================

test "Language.fromString canonical names" {
    const cases = .{
        .{ "python", Language.python },
        .{ "javascript", Language.javascript },
        .{ "typescript", Language.typescript },
        .{ "rust", Language.rust },
        .{ "go", Language.go },
        .{ "c", Language.c_lang },
        .{ "zig", Language.zig_lang },
        .{ "json", Language.json },
        .{ "html", Language.html },
        .{ "css", Language.css },
        .{ "bash", Language.bash },
    };
    inline for (cases) |case| {
        try std.testing.expectEqual(case[1], Language.fromString(case[0]).?);
    }
}

test "Language.fromString aliases" {
    try std.testing.expectEqual(Language.python, Language.fromString("py").?);
    try std.testing.expectEqual(Language.javascript, Language.fromString("js").?);
    try std.testing.expectEqual(Language.javascript, Language.fromString("jsx").?);
    try std.testing.expectEqual(Language.typescript, Language.fromString("ts").?);
    try std.testing.expectEqual(Language.rust, Language.fromString("rs").?);
    try std.testing.expectEqual(Language.go, Language.fromString("golang").?);
    try std.testing.expectEqual(Language.bash, Language.fromString("sh").?);
    try std.testing.expectEqual(Language.bash, Language.fromString("shell").?);
}

test "Language.fromFilename detects from full path" {
    try std.testing.expectEqual(Language.python, Language.fromFilename("src/main.py").?);
    try std.testing.expectEqual(Language.javascript, Language.fromFilename("app.js").?);
    try std.testing.expectEqual(Language.javascript, Language.fromFilename("component.jsx").?);
    try std.testing.expectEqual(Language.typescript, Language.fromFilename("index.ts").?);
    try std.testing.expectEqual(Language.rust, Language.fromFilename("lib.rs").?);
    try std.testing.expectEqual(Language.go, Language.fromFilename("main.go").?);
    try std.testing.expectEqual(Language.bash, Language.fromFilename("script.sh").?);
    try std.testing.expectEqual(Language.c_lang, Language.fromFilename("hello.c").?);
    try std.testing.expectEqual(Language.zig_lang, Language.fromFilename("build.zig").?);
    try std.testing.expectEqual(Language.html, Language.fromFilename("index.html").?);
    try std.testing.expectEqual(Language.css, Language.fromFilename("style.css").?);
    try std.testing.expectEqual(Language.json, Language.fromFilename("package.json").?);
}

test "Language.fromFilename returns null for unknown" {
    try std.testing.expect(Language.fromFilename("readme.txt") == null);
    try std.testing.expect(Language.fromFilename("Makefile") == null);
}

test "Language.fromString unknown returns null" {
    try std.testing.expect(Language.fromString("brainfuck") == null);
    try std.testing.expect(Language.fromString("") == null);
}

test "Language.grammar returns non-null for all languages" {
    inline for (std.meta.fields(Language)) |field| {
        const lang: Language = @enumFromInt(field.value);
        const g = lang.grammar();
        try std.testing.expect(@intFromPtr(g) != 0);
    }
}

test "Language.highlightQuery returns non-empty for all languages" {
    inline for (std.meta.fields(Language)) |field| {
        const lang: Language = @enumFromInt(field.value);
        const q = lang.highlightQuery();
        try std.testing.expect(q.len > 0);
    }
}

test "Language.name round-trips through fromString" {
    inline for (std.meta.fields(Language)) |field| {
        const lang: Language = @enumFromInt(field.value);
        const canonical = lang.name();
        const resolved = Language.fromString(canonical);
        try std.testing.expectEqual(lang, resolved.?);
    }
}
