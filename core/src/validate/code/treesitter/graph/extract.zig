//! Language-dispatch entry points for call graph extraction.
//!
//! Provides top-level functions that parse source code, detect the language,
//! and delegate to language-specific extractors.
//!
//! Thread safety: All functions are safe to call concurrently.

const std = @import("std");
const Language = @import("../language.zig").Language;
const parser_mod = @import("../parser.zig");
const types = @import("types.zig");
const python = @import("python.zig");
const rust_lang = @import("rust_lang.zig");
const javascript = @import("javascript.zig");
const CallSiteDetail = types.CallSiteDetail;
const ExtractionResult = types.ExtractionResult;
const ImportEntry = types.ImportEntry;

/// Extract callable definitions and import aliases from source code.
///
/// Parses the source, dispatches to the language-specific extractor.
/// Caller owns all returned data (allocated with `allocator`).
pub fn extractCallablesAndAliases(
    allocator: std.mem.Allocator,
    source: []const u8,
    language: Language,
    file_path: []const u8,
    project_root: []const u8,
) !ExtractionResult {
    var p = try parser_mod.Parser.init(language);
    defer p.deinit();

    var tree = try p.parse(source, null);
    defer tree.deinit();

    return switch (language) {
        .python => python.extractCallables(allocator, tree.rootNode(), source, file_path, project_root),
        .rust => rust_lang.extractCallables(allocator, tree.rootNode(), source, file_path, project_root),
        .javascript, .typescript => javascript.extractCallables(allocator, tree.rootNode(), source, file_path, project_root, language),
        else => .{ .callables = &.{}, .aliases = &.{} },
    };
}

/// Extract call sites from source code with import-aware resolution.
///
/// Parses the source, builds per-file import context, then extracts call
/// sites with candidate resolution paths informed by imports.
/// Caller owns all returned data.
pub fn extractCallSites(
    allocator: std.mem.Allocator,
    source: []const u8,
    language: Language,
    definer_fqn: []const u8,
    file_path: []const u8,
    project_root: []const u8,
) ![]CallSiteDetail {
    var p = try parser_mod.Parser.init(language);
    defer p.deinit();

    var tree = try p.parse(source, null);
    defer tree.deinit();

    const root = tree.rootNode();

    // Build import context from the full file AST
    const imports: []const ImportEntry = switch (language) {
        .python => try python.buildImportContext(allocator, root, source, file_path, project_root),
        .rust => try rust_lang.buildImportContext(allocator, root, source, file_path, project_root),
        .javascript, .typescript => try javascript.buildImportContext(allocator, root, source),
        else => &.{},
    };
    defer {
        for (imports) |entry| {
            allocator.free(entry.local_name);
            allocator.free(entry.canonical_path);
        }
        allocator.free(imports);
    }

    return switch (language) {
        .python => python.extractCallSites(allocator, root, source, definer_fqn, imports),
        .rust => rust_lang.extractCallSites(allocator, root, source, definer_fqn, imports),
        .javascript, .typescript => javascript.extractCallSites(allocator, root, source, definer_fqn, imports),
        else => &.{},
    };
}

// =============================================================================
// Tests
// =============================================================================

test "extractCallablesAndAliases works for Python" {
    const source = "def foo(): pass\ndef bar(): pass\n";
    const result = try extractCallablesAndAliases(std.testing.allocator, source, .python, "test.py", "");
    defer {
        for (result.callables) |c| {
            std.testing.allocator.free(c.fqn);
            std.testing.allocator.free(c.parameters);
        }
        std.testing.allocator.free(result.callables);
        std.testing.allocator.free(result.aliases);
    }

    try std.testing.expectEqual(@as(usize, 2), result.callables.len);
}

test "extractCallablesAndAliases returns empty for unsupported language" {
    const source = "{\"key\": 42}";
    const result = try extractCallablesAndAliases(std.testing.allocator, source, .json, "test.json", "");
    // Empty slices from &.{} are not allocated, so no free needed
    try std.testing.expectEqual(@as(usize, 0), result.callables.len);
    try std.testing.expectEqual(@as(usize, 0), result.aliases.len);
}

test "extractCallSites works for Python" {
    const source = "foo(1)\nbar()\n";
    const calls = try extractCallSites(std.testing.allocator, source, .python, "::test::main", "test.py", "");
    defer {
        for (calls) |cs| {
            for (cs.potential_resolved_paths) |rp| {
                std.testing.allocator.free(rp);
            }
            std.testing.allocator.free(cs.potential_resolved_paths);
            std.testing.allocator.free(cs.arguments);
        }
        std.testing.allocator.free(calls);
    }

    try std.testing.expectEqual(@as(usize, 2), calls.len);
}

test "extractCallSites resolves via import context" {
    const source = "import numpy as np\nnp.array([1,2])\n";
    const calls = try extractCallSites(std.testing.allocator, source, .python, "::test::main", "test.py", "");
    defer {
        for (calls) |cs| {
            for (cs.potential_resolved_paths) |rp| {
                std.testing.allocator.free(rp);
            }
            std.testing.allocator.free(cs.potential_resolved_paths);
            std.testing.allocator.free(cs.arguments);
        }
        std.testing.allocator.free(calls);
    }

    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("np.array", calls[0].raw_target_name);
    // Should have both naive and resolved paths
    try std.testing.expectEqual(@as(usize, 2), calls[0].potential_resolved_paths.len);
    try std.testing.expectEqualStrings("::np::array", calls[0].potential_resolved_paths[0]);
    try std.testing.expectEqualStrings("::numpy::array", calls[0].potential_resolved_paths[1]);
}
