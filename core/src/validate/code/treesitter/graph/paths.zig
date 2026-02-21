//! FQN building and module path conversion.
//!
//! Utilities for converting file paths to module paths and building canonical
//! fully-qualified names (FQNs) for callables.
//!
//! FQN format: "::module::submodule::ClassName::method_name"
//! - Always starts with "::"
//! - Uses "::" as separator
//!
//! Thread safety: All functions are pure. Safe to call concurrently.

const std = @import("std");

/// Build a canonical FQN by joining namespace components with "::".
///
/// E.g., `buildFqn(alloc, "::mymod", "MyClass", "method")` -> "::mymod::MyClass::method"
pub fn buildFqn(
    allocator: std.mem.Allocator,
    parts: []const []const u8,
) ![]const u8 {
    // Calculate total length
    var total_len: usize = 0;
    for (parts) |part| {
        if (total_len > 0) total_len += 2; // "::"
        total_len += part.len;
    }

    var result = try allocator.alloc(u8, total_len);
    var pos: usize = 0;
    for (parts, 0..) |part, i| {
        if (i > 0) {
            result[pos] = ':';
            result[pos + 1] = ':';
            pos += 2;
        }
        @memcpy(result[pos .. pos + part.len], part);
        pos += part.len;
    }
    return result;
}

/// Convert a file path to a module path prefix.
///
/// Strips project root, removes file extension, replaces '/' with '::',
/// prefixes with "::".
///
/// E.g., `fileToModulePath(alloc, "src/utils/helper.py", "")` -> "::src::utils::helper"
pub fn fileToModulePath(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    project_root: []const u8,
) ![]const u8 {
    // Strip project root prefix
    var path = file_path;
    if (project_root.len > 0 and std.mem.startsWith(u8, path, project_root)) {
        path = path[project_root.len..];
        // Skip leading separator
        if (path.len > 0 and (path[0] == '/' or path[0] == '\\')) {
            path = path[1..];
        }
    }

    // Strip common source directory prefixes
    if (std.mem.startsWith(u8, path, "src/") or std.mem.startsWith(u8, path, "src\\")) {
        path = path[4..];
    } else if (std.mem.startsWith(u8, path, "lib/") or std.mem.startsWith(u8, path, "lib\\")) {
        path = path[4..];
    }

    // Detect Rust files before stripping extension
    const is_rs = std.mem.endsWith(u8, path, ".rs");

    // Strip file extension
    if (std.mem.lastIndexOfScalar(u8, path, '.')) |dot| {
        path = path[0..dot];
    }

    // Strip __init__ for Python packages
    if (std.mem.endsWith(u8, path, "/__init__") or std.mem.endsWith(u8, path, "\\__init__")) {
        path = path[0 .. path.len - 9]; // len("/__init__") == 9
    }

    // Strip Rust special filenames (mod.rs, lib.rs, main.rs represent parent directory)
    if (is_rs) {
        if (std.mem.endsWith(u8, path, "/mod") or std.mem.endsWith(u8, path, "\\mod")) {
            path = path[0 .. path.len - 4];
        } else if (std.mem.endsWith(u8, path, "/lib") or std.mem.endsWith(u8, path, "\\lib")) {
            path = path[0 .. path.len - 4];
        } else if (std.mem.endsWith(u8, path, "/main") or std.mem.endsWith(u8, path, "\\main")) {
            path = path[0 .. path.len - 5];
        } else if (std.mem.eql(u8, path, "mod") or std.mem.eql(u8, path, "lib") or std.mem.eql(u8, path, "main")) {
            path = path[0..0];
        }
    }

    // Calculate output size: "::" prefix + replace / with ::
    var count: usize = 2; // leading "::"
    for (path) |ch| {
        if (ch == '/' or ch == '\\') {
            count += 2; // "::"
        } else {
            count += 1;
        }
    }

    var result = try allocator.alloc(u8, count);
    result[0] = ':';
    result[1] = ':';
    var pos: usize = 2;
    for (path) |ch| {
        if (ch == '/' or ch == '\\') {
            result[pos] = ':';
            result[pos + 1] = ':';
            pos += 2;
        } else {
            result[pos] = ch;
            pos += 1;
        }
    }

    return result;
}

/// Convert a Python dotted module path to canonical FQN format.
///
/// E.g., "os.path.join" -> "::os::path::join"
pub fn dottedToCanonical(
    allocator: std.mem.Allocator,
    dotted: []const u8,
) ![]const u8 {
    // Count dots to determine length
    var dot_count: usize = 0;
    for (dotted) |ch| {
        if (ch == '.') dot_count += 1;
    }

    // "::" prefix + replace each "." with "::" (adds 1 char per dot)
    const total = 2 + dotted.len + dot_count;
    var result = try allocator.alloc(u8, total);
    result[0] = ':';
    result[1] = ':';
    var pos: usize = 2;
    for (dotted) |ch| {
        if (ch == '.') {
            result[pos] = ':';
            result[pos + 1] = ':';
            pos += 2;
        } else {
            result[pos] = ch;
            pos += 1;
        }
    }

    return result;
}

/// Convert a Rust ::-separated path to canonical FQN format.
///
/// Rust paths already use "::" as separators. This ensures the leading "::"
/// prefix. E.g., "std::collections::HashMap" -> "::std::collections::HashMap".
pub fn colonSeparatedToCanonical(
    allocator: std.mem.Allocator,
    path: []const u8,
) ![]const u8 {
    if (std.mem.startsWith(u8, path, "::")) {
        return allocator.dupe(u8, path);
    }
    const result = try allocator.alloc(u8, 2 + path.len);
    result[0] = ':';
    result[1] = ':';
    @memcpy(result[2..], path);
    return result;
}

/// Convert a JS module path to canonical FQN format.
///
/// Strips leading "./", converts "/" to "::", adds "::" prefix.
/// E.g., "./utils/helper" -> "::utils::helper", "lodash" -> "::lodash".
pub fn jsModuleToCanonical(
    allocator: std.mem.Allocator,
    js_path: []const u8,
) ![]const u8 {
    var path = js_path;
    // Strip leading ./
    if (std.mem.startsWith(u8, path, "./")) {
        path = path[2..];
    }

    // Strip trailing /index (index files represent the directory)
    if (std.mem.endsWith(u8, path, "/index")) {
        path = path[0 .. path.len - 6];
    }

    // Count '/' to calculate output size
    var slash_count: usize = 0;
    for (path) |ch| {
        if (ch == '/') slash_count += 1;
    }

    // "::" prefix + each "/" becomes "::" (1 extra char per slash)
    const total = 2 + path.len + slash_count;
    var result = try allocator.alloc(u8, total);
    result[0] = ':';
    result[1] = ':';
    var pos: usize = 2;
    for (path) |ch| {
        if (ch == '/') {
            result[pos] = ':';
            result[pos + 1] = ':';
            pos += 2;
        } else {
            result[pos] = ch;
            pos += 1;
        }
    }

    return result;
}

/// Extract the module path from a fully-qualified name by stripping the last segment.
///
/// Returns a view into the input (no allocation).
/// E.g., "::mymod::Class::method" -> "::mymod::Class"
///       "::mymod::func" -> "::mymod"
///       "::func" -> "::"
pub fn extractModuleFromFqn(fqn: []const u8) []const u8 {
    // Need at least "::X::Y" (6 chars) to have a strippable segment
    if (fqn.len < 4) return "::";

    // Search backwards for "::" (but not the leading "::")
    var i: usize = fqn.len - 1;
    while (i >= 2) : (i -= 1) {
        if (fqn[i] == ':' and fqn[i - 1] == ':') {
            return fqn[0 .. i - 1];
        }
    }
    return "::";
}

// =============================================================================
// Tests
// =============================================================================

test "buildFqn joins parts with ::" {
    const fqn = try buildFqn(std.testing.allocator, &.{ "::mymod", "MyClass", "method" });
    defer std.testing.allocator.free(fqn);
    try std.testing.expectEqualStrings("::mymod::MyClass::method", fqn);
}

test "buildFqn single part" {
    const fqn = try buildFqn(std.testing.allocator, &.{"::mymod"});
    defer std.testing.allocator.free(fqn);
    try std.testing.expectEqualStrings("::mymod", fqn);
}

test "fileToModulePath strips extension and src prefix" {
    const mod = try fileToModulePath(std.testing.allocator, "src/utils/helper.py", "");
    defer std.testing.allocator.free(mod);
    try std.testing.expectEqualStrings("::utils::helper", mod);
}

test "fileToModulePath strips project root and src prefix" {
    const mod = try fileToModulePath(std.testing.allocator, "project/src/utils.py", "project");
    defer std.testing.allocator.free(mod);
    try std.testing.expectEqualStrings("::utils", mod);
}

test "fileToModulePath strips lib prefix" {
    const mod = try fileToModulePath(std.testing.allocator, "lib/core/engine.py", "");
    defer std.testing.allocator.free(mod);
    try std.testing.expectEqualStrings("::core::engine", mod);
}

test "fileToModulePath strips Rust mod.rs" {
    const mod = try fileToModulePath(std.testing.allocator, "src/utils/mod.rs", "");
    defer std.testing.allocator.free(mod);
    try std.testing.expectEqualStrings("::utils", mod);
}

test "fileToModulePath strips Rust lib.rs at root" {
    const mod = try fileToModulePath(std.testing.allocator, "src/lib.rs", "");
    defer std.testing.allocator.free(mod);
    try std.testing.expectEqualStrings("::", mod);
}

test "fileToModulePath strips Rust main.rs at root" {
    const mod = try fileToModulePath(std.testing.allocator, "src/main.rs", "");
    defer std.testing.allocator.free(mod);
    try std.testing.expectEqualStrings("::", mod);
}

test "fileToModulePath handles __init__" {
    const mod = try fileToModulePath(std.testing.allocator, "mypackage/__init__.py", "");
    defer std.testing.allocator.free(mod);
    try std.testing.expectEqualStrings("::mypackage", mod);
}

test "dottedToCanonical converts Python module path" {
    const fqn = try dottedToCanonical(std.testing.allocator, "os.path.join");
    defer std.testing.allocator.free(fqn);
    try std.testing.expectEqualStrings("::os::path::join", fqn);
}

test "dottedToCanonical handles single name" {
    const fqn = try dottedToCanonical(std.testing.allocator, "print");
    defer std.testing.allocator.free(fqn);
    try std.testing.expectEqualStrings("::print", fqn);
}

test "colonSeparatedToCanonical adds prefix" {
    const fqn = try colonSeparatedToCanonical(std.testing.allocator, "std::collections::HashMap");
    defer std.testing.allocator.free(fqn);
    try std.testing.expectEqualStrings("::std::collections::HashMap", fqn);
}

test "colonSeparatedToCanonical preserves existing prefix" {
    const fqn = try colonSeparatedToCanonical(std.testing.allocator, "::std::io::Read");
    defer std.testing.allocator.free(fqn);
    try std.testing.expectEqualStrings("::std::io::Read", fqn);
}

test "jsModuleToCanonical strips ./ and converts slashes" {
    const fqn = try jsModuleToCanonical(std.testing.allocator, "./utils/helper");
    defer std.testing.allocator.free(fqn);
    try std.testing.expectEqualStrings("::utils::helper", fqn);
}

test "jsModuleToCanonical handles bare module name" {
    const fqn = try jsModuleToCanonical(std.testing.allocator, "lodash");
    defer std.testing.allocator.free(fqn);
    try std.testing.expectEqualStrings("::lodash", fqn);
}

test "jsModuleToCanonical strips trailing index" {
    const fqn = try jsModuleToCanonical(std.testing.allocator, "./utils/index");
    defer std.testing.allocator.free(fqn);
    try std.testing.expectEqualStrings("::utils", fqn);
}

test "jsModuleToCanonical does not strip index in middle" {
    const fqn = try jsModuleToCanonical(std.testing.allocator, "./index/utils");
    defer std.testing.allocator.free(fqn);
    try std.testing.expectEqualStrings("::index::utils", fqn);
}

test "extractModuleFromFqn strips last segment" {
    try std.testing.expectEqualStrings("::mymod::MyClass", extractModuleFromFqn("::mymod::MyClass::method"));
}

test "extractModuleFromFqn single segment" {
    try std.testing.expectEqualStrings("::", extractModuleFromFqn("::func"));
}

test "extractModuleFromFqn two segments" {
    try std.testing.expectEqualStrings("::mymod", extractModuleFromFqn("::mymod::func"));
}

test "extractModuleFromFqn root" {
    try std.testing.expectEqualStrings("::", extractModuleFromFqn("::"));
}
