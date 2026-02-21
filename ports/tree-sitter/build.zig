const std = @import("std");

pub const TreeSitter = struct {
    lib: *std.Build.Step.Compile,
    include_dir: std.Build.LazyPath,
};

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) TreeSitter {
    const mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addCMacro("_POSIX_C_SOURCE", "200112L");
    mod.addCMacro("_DEFAULT_SOURCE", "");

    const lib = b.addLibrary(.{
        .name = "tree-sitter",
        .root_module = mod,
        .linkage = .static,
    });
    lib.linkLibC();

    // Include paths for core runtime
    lib.addIncludePath(b.path("deps/tree-sitter/lib/include"));
    lib.addIncludePath(b.path("deps/tree-sitter/lib/src"));

    // Grammar-bundled tree_sitter/parser.h (no longer in core lib/include since v0.25).
    // All grammars ship identical copies; we pick python's as the canonical source.
    lib.addIncludePath(b.path("deps/tree-sitter-python/src"));

    // =========================================================================
    // Core runtime (individual files, excluding lib.c amalgamation)
    // wasm_store.c is included but compiles to no-op stubs without TREE_SITTER_FEATURE_WASM
    // =========================================================================
    lib.addCSourceFiles(.{
        .files = &.{
            "deps/tree-sitter/lib/src/alloc.c",
            "deps/tree-sitter/lib/src/get_changed_ranges.c",
            "deps/tree-sitter/lib/src/language.c",
            "deps/tree-sitter/lib/src/lexer.c",
            "deps/tree-sitter/lib/src/node.c",
            "deps/tree-sitter/lib/src/parser.c",
            "deps/tree-sitter/lib/src/point.c",
            "deps/tree-sitter/lib/src/query.c",
            "deps/tree-sitter/lib/src/stack.c",
            "deps/tree-sitter/lib/src/subtree.c",
            "deps/tree-sitter/lib/src/tree.c",
            "deps/tree-sitter/lib/src/tree_cursor.c",
            "deps/tree-sitter/lib/src/wasm_store.c",
        },
        .flags = &.{"-std=c11"},
    });

    // =========================================================================
    // Language grammars
    //
    // Each grammar ships its own copy of tree_sitter/*.h headers under src/.
    // We add each grammar's src/ as an include path via per-file .flags so
    // grammars don't conflict with each other's bundled headers.
    // =========================================================================

    // Python
    lib.addCSourceFiles(.{
        .files = &.{
            "deps/tree-sitter-python/src/parser.c",
            "deps/tree-sitter-python/src/scanner.c",
        },
        .flags = &.{ "-std=c11", "-Ideps/tree-sitter-python/src" },
    });

    // JavaScript
    lib.addCSourceFiles(.{
        .files = &.{
            "deps/tree-sitter-javascript/src/parser.c",
            "deps/tree-sitter-javascript/src/scanner.c",
        },
        .flags = &.{ "-std=c11", "-Ideps/tree-sitter-javascript/src" },
    });

    // TypeScript (lives under typescript/ subdirectory; scanner includes ../../common/)
    lib.addCSourceFiles(.{
        .files = &.{
            "deps/tree-sitter-typescript/typescript/src/parser.c",
            "deps/tree-sitter-typescript/typescript/src/scanner.c",
        },
        .flags = &.{ "-std=c11", "-Ideps/tree-sitter-typescript/typescript/src" },
    });

    // Rust
    lib.addCSourceFiles(.{
        .files = &.{
            "deps/tree-sitter-rust/src/parser.c",
            "deps/tree-sitter-rust/src/scanner.c",
        },
        .flags = &.{ "-std=c11", "-Ideps/tree-sitter-rust/src" },
    });

    // Go
    lib.addCSourceFiles(.{
        .files = &.{
            "deps/tree-sitter-go/src/parser.c",
        },
        .flags = &.{ "-std=c11", "-Ideps/tree-sitter-go/src" },
    });

    // C
    lib.addCSourceFiles(.{
        .files = &.{
            "deps/tree-sitter-c/src/parser.c",
        },
        .flags = &.{ "-std=c11", "-Ideps/tree-sitter-c/src" },
    });

    // Zig
    lib.addCSourceFiles(.{
        .files = &.{
            "deps/tree-sitter-zig/src/parser.c",
        },
        .flags = &.{ "-std=c11", "-Ideps/tree-sitter-zig/src" },
    });

    // JSON
    lib.addCSourceFiles(.{
        .files = &.{
            "deps/tree-sitter-json/src/parser.c",
        },
        .flags = &.{ "-std=c11", "-Ideps/tree-sitter-json/src" },
    });

    // HTML
    lib.addCSourceFiles(.{
        .files = &.{
            "deps/tree-sitter-html/src/parser.c",
            "deps/tree-sitter-html/src/scanner.c",
        },
        .flags = &.{ "-std=c11", "-Ideps/tree-sitter-html/src" },
    });

    // CSS
    lib.addCSourceFiles(.{
        .files = &.{
            "deps/tree-sitter-css/src/parser.c",
            "deps/tree-sitter-css/src/scanner.c",
        },
        .flags = &.{ "-std=c11", "-Ideps/tree-sitter-css/src" },
    });

    // Bash
    lib.addCSourceFiles(.{
        .files = &.{
            "deps/tree-sitter-bash/src/parser.c",
            "deps/tree-sitter-bash/src/scanner.c",
        },
        .flags = &.{ "-std=c11", "-Ideps/tree-sitter-bash/src" },
    });

    return .{ .lib = lib, .include_dir = b.path("deps/tree-sitter/lib/include") };
}
