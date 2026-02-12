//! Plugin directory scanner.
//!
//! Scans a plugins directory for UI plugins. Each entry is either a single
//! .js/.ts file or a directory containing an entry point and optional manifest.
//!
//! NOT thread-safe. Caller must synchronize access.

const std = @import("std");
const log = @import("../../log.zig");

pub const ScanError = error{
    OutOfMemory,
    NoHomeDir,
    AccessDenied,
    Unexpected,
};

/// Discovered plugin information.
/// All string fields are owned by the allocator that created the list.
pub const PluginInfo = struct {
    /// Plugin identifier (derived from dirname or filename without extension).
    plugin_id: []const u8,
    /// Absolute path to the plugin directory (or parent dir for single-file plugins).
    plugin_dir: []const u8,
    /// Relative entry path within plugin_dir (e.g., "index.ts" or the filename itself).
    entry_path: []const u8,
    /// Raw JSON string of talu.json manifest, or inferred defaults if no manifest.
    manifest_json: []const u8,
    /// True if this is a directory plugin, false if single-file.
    is_directory: bool,
};

const max_manifest_size: usize = 64 * 1024;

/// Get the default plugins directory path.
/// Resolution: $TALU_PLUGINS_DIR > ~/.talu/plugins/
/// Caller owns returned memory.
pub fn getPluginsDir(allocator: std.mem.Allocator) ScanError![]const u8 {
    if (std.posix.getenv("TALU_PLUGINS_DIR")) |dir| {
        return allocator.dupe(u8, dir) catch return ScanError.OutOfMemory;
    }
    const home = std.posix.getenv("HOME") orelse return ScanError.NoHomeDir;
    return std.fs.path.join(allocator, &.{ home, ".talu", "plugins" }) catch return ScanError.OutOfMemory;
}

/// Resolve the plugins directory and scan it.
/// If `plugins_dir` is null, uses the default directory.
/// Caller owns returned memory and must call freePluginList to release it.
pub fn resolveAndScan(alloc: std.mem.Allocator, plugins_dir: ?[]const u8) ScanError![]PluginInfo {
    if (plugins_dir) |dir| return scanPlugins(alloc, dir);
    const default_dir = try getPluginsDir(alloc);
    defer alloc.free(default_dir);
    return scanPlugins(alloc, default_dir);
}

/// Scan the plugins directory and return discovered plugins.
/// Caller owns returned memory and must call freePluginList to release it.
pub fn scanPlugins(allocator: std.mem.Allocator, plugins_dir: []const u8) ScanError![]PluginInfo {
    var dir = std.fs.cwd().openDir(plugins_dir, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return allocator.alloc(PluginInfo, 0) catch return ScanError.OutOfMemory,
        error.AccessDenied => return ScanError.AccessDenied,
        else => return allocator.alloc(PluginInfo, 0) catch return ScanError.OutOfMemory,
    };
    defer dir.close();

    var results = std.ArrayListUnmanaged(PluginInfo){};
    errdefer freePluginList(allocator, results.toOwnedSlice(allocator) catch &.{});

    var iter = dir.iterate();
    while (iter.next() catch return ScanError.Unexpected) |entry| {
        // Skip dotfiles
        if (entry.name.len > 0 and entry.name[0] == '.') continue;

        const plugin = switch (entry.kind) {
            .directory => scanDirectoryPlugin(allocator, plugins_dir, entry.name) catch |err| {
                logScanSkip(entry.name, err);
                continue;
            },
            .file => scanFilePlugin(allocator, plugins_dir, entry.name) catch |err| {
                logScanSkip(entry.name, err);
                continue;
            },
            else => continue,
        };

        const p = plugin orelse continue;
        results.append(allocator, p) catch return ScanError.OutOfMemory;
    }

    return results.toOwnedSlice(allocator) catch return ScanError.OutOfMemory;
}

/// Free a plugin list returned by scanPlugins.
pub fn freePluginList(allocator: std.mem.Allocator, plugins: []PluginInfo) void {
    for (plugins) |p| {
        allocator.free(p.plugin_id);
        allocator.free(p.plugin_dir);
        allocator.free(p.entry_path);
        allocator.free(p.manifest_json);
    }
    allocator.free(plugins);
}

// =============================================================================
// Internal helpers
// =============================================================================

const ScanItemError = error{
    OutOfMemory,
    NotAPlugin,
    ReadFailed,
};

/// Scan a directory entry as a directory plugin.
fn scanDirectoryPlugin(
    allocator: std.mem.Allocator,
    plugins_dir: []const u8,
    dir_name: []const u8,
) ScanItemError!?PluginInfo {
    const plugin_path = std.fs.path.join(allocator, &.{ plugins_dir, dir_name }) catch
        return ScanItemError.OutOfMemory;
    errdefer allocator.free(plugin_path);

    var sub_dir = std.fs.cwd().openDir(plugin_path, .{}) catch
        return ScanItemError.NotAPlugin;
    defer sub_dir.close();

    // Try reading talu.json manifest
    var manifest_json: ?[]const u8 = null;
    var entry_from_manifest: ?[]const u8 = null;

    if (sub_dir.readFileAlloc(allocator, "talu.json", max_manifest_size)) |content| {
        manifest_json = content;
        // Extract "entry" field to find entry point
        entry_from_manifest = extractEntryField(allocator, content);
    } else |_| {
        // No manifest — will infer defaults
    }

    errdefer if (manifest_json) |m| allocator.free(m);
    errdefer if (entry_from_manifest) |e| allocator.free(e);

    // Determine entry point: manifest "entry" field, then index.ts, then index.js
    const entry_path = if (entry_from_manifest) |e|
        e
    else if (fileExists(sub_dir, "index.ts"))
        allocator.dupe(u8, "index.ts") catch return ScanItemError.OutOfMemory
    else if (fileExists(sub_dir, "index.js"))
        allocator.dupe(u8, "index.js") catch return ScanItemError.OutOfMemory
    else {
        // No entry point found — skip
        if (manifest_json) |m| allocator.free(m);
        allocator.free(plugin_path);
        return null;
    };
    errdefer allocator.free(entry_path);

    // Rewrite .ts entry to .js for the URL (server transpiles on the fly)
    const url_entry = rewriteTsToJs(allocator, entry_path) catch return ScanItemError.OutOfMemory;
    errdefer allocator.free(url_entry);

    // If no manifest, generate default
    const final_manifest = manifest_json orelse
        buildDefaultManifest(allocator, dir_name) catch return ScanItemError.OutOfMemory;

    const plugin_id = allocator.dupe(u8, dir_name) catch return ScanItemError.OutOfMemory;

    // Free the original entry_path since we use url_entry
    allocator.free(entry_path);

    return PluginInfo{
        .plugin_id = plugin_id,
        .plugin_dir = plugin_path,
        .entry_path = url_entry,
        .manifest_json = final_manifest,
        .is_directory = true,
    };
}

/// Scan a file entry as a single-file plugin.
fn scanFilePlugin(
    allocator: std.mem.Allocator,
    plugins_dir: []const u8,
    file_name: []const u8,
) ScanItemError!?PluginInfo {
    // Only .ts and .js files are plugins
    const ext = std.fs.path.extension(file_name);
    if (!std.mem.eql(u8, ext, ".ts") and !std.mem.eql(u8, ext, ".js")) {
        return null;
    }

    const stem = file_name[0 .. file_name.len - ext.len];
    if (stem.len == 0) return null;

    const plugin_id = allocator.dupe(u8, stem) catch return ScanItemError.OutOfMemory;
    errdefer allocator.free(plugin_id);

    const plugin_dir = allocator.dupe(u8, plugins_dir) catch return ScanItemError.OutOfMemory;
    errdefer allocator.free(plugin_dir);

    // Rewrite .ts to .js for the URL
    const url_entry = rewriteTsToJs(allocator, file_name) catch return ScanItemError.OutOfMemory;
    errdefer allocator.free(url_entry);

    const manifest = buildDefaultManifest(allocator, stem) catch return ScanItemError.OutOfMemory;

    return PluginInfo{
        .plugin_id = plugin_id,
        .plugin_dir = plugin_dir,
        .entry_path = url_entry,
        .manifest_json = manifest,
        .is_directory = false,
    };
}

/// Check if a file exists in a directory.
fn fileExists(dir: std.fs.Dir, name: []const u8) bool {
    dir.access(name, .{}) catch return false;
    return true;
}

/// Build a default manifest JSON for plugins without talu.json.
fn buildDefaultManifest(allocator: std.mem.Allocator, plugin_id: []const u8) ![]const u8 {
    return std.fmt.allocPrint(
        allocator,
        "{{\"id\":\"{s}\",\"activationEvents\":[\"*\"]}}",
        .{plugin_id},
    );
}

/// Extract the "entry" field from a manifest JSON string.
/// Returns null if not found or not a string. Caller owns returned memory.
fn extractEntryField(allocator: std.mem.Allocator, json_bytes: []const u8) ?[]const u8 {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{}) catch return null;
    defer parsed.deinit();

    const entry_val = switch (parsed.value) {
        .object => |obj| obj.get("entry") orelse return null,
        else => return null,
    };

    const entry_str = switch (entry_val) {
        .string => |s| s,
        else => return null,
    };

    return allocator.dupe(u8, entry_str) catch null;
}

/// Rewrite .ts extension to .js for browser-facing URLs.
fn rewriteTsToJs(allocator: std.mem.Allocator, path: []const u8) ![]const u8 {
    if (std.mem.endsWith(u8, path, ".ts")) {
        const stem = path[0 .. path.len - 3];
        return std.fmt.allocPrint(allocator, "{s}.js", .{stem});
    }
    return allocator.dupe(u8, path);
}

fn logScanSkip(name: []const u8, err: anyerror) void {
    log.debug("plugins", "Skipped plugin entry", .{
        .name = name,
        .reason = @errorName(err),
    }, @src());
}

// =============================================================================
// Tests
// =============================================================================

test "scanPlugins empty directory returns empty" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const dir_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(dir_path);

    const plugins = try scanPlugins(std.testing.allocator, dir_path);
    defer freePluginList(std.testing.allocator, plugins);

    try std.testing.expectEqual(@as(usize, 0), plugins.len);
}

test "scanPlugins directory plugin with talu.json" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    // Create plugin directory with manifest and entry point
    tmp.dir.makeDir("my-plugin") catch unreachable;
    var plugin_dir = tmp.dir.openDir("my-plugin", .{}) catch unreachable;
    defer plugin_dir.close();

    plugin_dir.writeFile(.{
        .sub_path = "talu.json",
        .data = "{\"id\":\"my-plugin\",\"name\":\"My Plugin\",\"version\":\"1.0.0\",\"entry\":\"main.ts\"}",
    }) catch unreachable;
    plugin_dir.writeFile(.{ .sub_path = "main.ts", .data = "export default function() {}" }) catch unreachable;

    const dir_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(dir_path);

    const plugins = try scanPlugins(std.testing.allocator, dir_path);
    defer freePluginList(std.testing.allocator, plugins);

    try std.testing.expectEqual(@as(usize, 1), plugins.len);
    try std.testing.expectEqualStrings("my-plugin", plugins[0].plugin_id);
    try std.testing.expectEqualStrings("main.js", plugins[0].entry_path);
    try std.testing.expect(plugins[0].is_directory);
    // Manifest should be the raw talu.json content
    try std.testing.expect(std.mem.indexOf(u8, plugins[0].manifest_json, "My Plugin") != null);
}

test "scanPlugins directory plugin without manifest infers defaults" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    tmp.dir.makeDir("auto-plugin") catch unreachable;
    var plugin_dir = tmp.dir.openDir("auto-plugin", .{}) catch unreachable;
    defer plugin_dir.close();

    plugin_dir.writeFile(.{ .sub_path = "index.ts", .data = "export default function() {}" }) catch unreachable;

    const dir_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(dir_path);

    const plugins = try scanPlugins(std.testing.allocator, dir_path);
    defer freePluginList(std.testing.allocator, plugins);

    try std.testing.expectEqual(@as(usize, 1), plugins.len);
    try std.testing.expectEqualStrings("auto-plugin", plugins[0].plugin_id);
    try std.testing.expectEqualStrings("index.js", plugins[0].entry_path);
    try std.testing.expect(plugins[0].is_directory);
    // Should have inferred manifest
    try std.testing.expect(std.mem.indexOf(u8, plugins[0].manifest_json, "auto-plugin") != null);
    try std.testing.expect(std.mem.indexOf(u8, plugins[0].manifest_json, "activationEvents") != null);
}

test "scanPlugins single-file .ts plugin" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    tmp.dir.writeFile(.{ .sub_path = "safety-gate.ts", .data = "export default function() {}" }) catch unreachable;

    const dir_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(dir_path);

    const plugins = try scanPlugins(std.testing.allocator, dir_path);
    defer freePluginList(std.testing.allocator, plugins);

    try std.testing.expectEqual(@as(usize, 1), plugins.len);
    try std.testing.expectEqualStrings("safety-gate", plugins[0].plugin_id);
    try std.testing.expectEqualStrings("safety-gate.js", plugins[0].entry_path);
    try std.testing.expect(!plugins[0].is_directory);
}

test "scanPlugins non-plugin files are skipped" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    tmp.dir.writeFile(.{ .sub_path = "readme.md", .data = "# Hello" }) catch unreachable;
    tmp.dir.writeFile(.{ .sub_path = "notes.txt", .data = "some notes" }) catch unreachable;
    tmp.dir.writeFile(.{ .sub_path = ".hidden.ts", .data = "hidden" }) catch unreachable;

    const dir_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(dir_path);

    const plugins = try scanPlugins(std.testing.allocator, dir_path);
    defer freePluginList(std.testing.allocator, plugins);

    try std.testing.expectEqual(@as(usize, 0), plugins.len);
}

test "scanPlugins nonexistent directory returns empty" {
    const plugins = try scanPlugins(std.testing.allocator, "/tmp/talu-test-nonexistent-dir-12345");
    defer freePluginList(std.testing.allocator, plugins);

    try std.testing.expectEqual(@as(usize, 0), plugins.len);
}

test "scanPlugins directory without entry point is skipped" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    // Directory with no index.ts, index.js, or talu.json entry
    tmp.dir.makeDir("empty-plugin") catch unreachable;
    var plugin_dir = tmp.dir.openDir("empty-plugin", .{}) catch unreachable;
    defer plugin_dir.close();
    plugin_dir.writeFile(.{ .sub_path = "readme.md", .data = "no entry" }) catch unreachable;

    const dir_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(dir_path);

    const plugins = try scanPlugins(std.testing.allocator, dir_path);
    defer freePluginList(std.testing.allocator, plugins);

    try std.testing.expectEqual(@as(usize, 0), plugins.len);
}

test "rewriteTsToJs converts .ts to .js" {
    const result = try rewriteTsToJs(std.testing.allocator, "main.ts");
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("main.js", result);
}

test "rewriteTsToJs preserves .js" {
    const result = try rewriteTsToJs(std.testing.allocator, "main.js");
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("main.js", result);
}

test "extractEntryField returns entry from manifest" {
    const result = extractEntryField(
        std.testing.allocator,
        "{\"id\":\"test\",\"entry\":\"src/main.ts\"}",
    );
    defer if (result) |r| std.testing.allocator.free(r);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("src/main.ts", result.?);
}

test "extractEntryField returns null for missing entry" {
    const result = extractEntryField(
        std.testing.allocator,
        "{\"id\":\"test\"}",
    );
    try std.testing.expect(result == null);
}

test "getPluginsDir returns HOME-based path when env unset" {
    // With TALU_PLUGINS_DIR unset, should fall back to $HOME/.talu/plugins
    const home = std.posix.getenv("HOME") orelse return error.SkipZigTest;

    const result = try getPluginsDir(std.testing.allocator);
    defer std.testing.allocator.free(result);

    const expected = try std.fs.path.join(std.testing.allocator, &.{ home, ".talu", "plugins" });
    defer std.testing.allocator.free(expected);

    try std.testing.expectEqualStrings(expected, result);
}

test "freePluginList frees all allocations" {
    // Allocate a list with one entry; testing allocator detects leaks.
    const plugins = try std.testing.allocator.alloc(PluginInfo, 1);
    errdefer std.testing.allocator.free(plugins);

    const id = try std.testing.allocator.dupe(u8, "test-id");
    errdefer std.testing.allocator.free(id);
    const dir = try std.testing.allocator.dupe(u8, "/tmp/test");
    errdefer std.testing.allocator.free(dir);
    const entry = try std.testing.allocator.dupe(u8, "index.js");
    errdefer std.testing.allocator.free(entry);
    const manifest = try std.testing.allocator.dupe(u8, "{\"id\":\"test-id\"}");

    plugins[0] = .{
        .plugin_id = id,
        .plugin_dir = dir,
        .entry_path = entry,
        .manifest_json = manifest,
        .is_directory = true,
    };
    freePluginList(std.testing.allocator, plugins);
}

test "freePluginList handles empty list" {
    const plugins = try std.testing.allocator.alloc(PluginInfo, 0);
    freePluginList(std.testing.allocator, plugins);
}

test "resolveAndScan with explicit dir scans that dir" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    tmp.dir.writeFile(.{ .sub_path = "hello.js", .data = "export default function() {}" }) catch unreachable;

    const dir_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(dir_path);

    const plugins = try resolveAndScan(std.testing.allocator, dir_path);
    defer freePluginList(std.testing.allocator, plugins);

    try std.testing.expectEqual(@as(usize, 1), plugins.len);
    try std.testing.expectEqualStrings("hello", plugins[0].plugin_id);
}

test "resolveAndScan with null dir resolves default and scans" {
    // With HOME set, resolveAndScan(null) resolves ~/.talu/plugins/ which
    // likely doesn't exist, so scanPlugins returns empty.
    if (std.posix.getenv("HOME") == null) return error.SkipZigTest;

    const plugins = try resolveAndScan(std.testing.allocator, null);
    defer freePluginList(std.testing.allocator, plugins);

    // Default plugins dir is unlikely to exist in test env → empty result.
    try std.testing.expectEqual(@as(usize, 0), plugins.len);
}
