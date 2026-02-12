//! C-API Plugins Module - UI plugin discovery.
//!
//! Thin C-ABI glue over io.plugins.scanner. Scans ~/.talu/plugins/ and
//! returns an arena-backed list of discovered plugins.

const std = @import("std");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");
const scanner = @import("../io/plugins/scanner.zig");

const allocator = std.heap.c_allocator;

// =============================================================================
// C-ABI Plugin Types
// =============================================================================

/// C-ABI plugin information record.
pub const CPluginInfo = extern struct {
    /// Plugin identifier (null-terminated).
    plugin_id: ?[*:0]const u8,
    /// Absolute path to the plugin directory (null-terminated).
    plugin_dir: ?[*:0]const u8,
    /// Relative entry path within plugin_dir (null-terminated).
    entry_path: ?[*:0]const u8,
    /// Raw JSON manifest string (null-terminated).
    manifest_json: ?[*:0]const u8,
    /// True if directory plugin, false if single-file.
    is_directory: bool,
    /// Reserved for future expansion.
    _reserved: [7]u8 = [_]u8{0} ** 7,
};

/// Plugin list container for C API.
pub const CPluginList = extern struct {
    /// Array of plugin info records.
    items: ?[*]CPluginInfo,
    /// Number of plugins in the array.
    count: usize,
    /// Internal: backing arena for string data.
    _arena: ?*anyopaque,
};

// =============================================================================
// List Builder
// =============================================================================

/// Build a CPluginList from scanned PluginInfo records.
fn buildPluginList(plugins: []const scanner.PluginInfo) !*CPluginList {
    const list = allocator.create(CPluginList) catch return error.OutOfMemory;
    errdefer allocator.destroy(list);

    const arena_ptr = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
    errdefer allocator.destroy(arena_ptr);
    arena_ptr.* = std.heap.ArenaAllocator.init(allocator);
    errdefer arena_ptr.deinit();

    const arena = arena_ptr.allocator();

    const items_buf = arena.alloc(CPluginInfo, plugins.len) catch return error.OutOfMemory;

    for (plugins, 0..) |plugin, i| {
        items_buf[i] = std.mem.zeroes(CPluginInfo);
        items_buf[i].plugin_id = (arena.dupeZ(u8, plugin.plugin_id) catch return error.OutOfMemory).ptr;
        items_buf[i].plugin_dir = (arena.dupeZ(u8, plugin.plugin_dir) catch return error.OutOfMemory).ptr;
        items_buf[i].entry_path = (arena.dupeZ(u8, plugin.entry_path) catch return error.OutOfMemory).ptr;
        items_buf[i].manifest_json = (arena.dupeZ(u8, plugin.manifest_json) catch return error.OutOfMemory).ptr;
        items_buf[i].is_directory = plugin.is_directory;
    }

    list.* = .{
        .items = if (plugins.len > 0) items_buf.ptr else null,
        .count = plugins.len,
        ._arena = @ptrCast(arena_ptr),
    };

    return list;
}

// =============================================================================
// Exported C API Functions
// =============================================================================

/// Scan a plugins directory and return discovered plugins.
///
/// Parameters:
///   - plugins_dir: Path to plugins directory (null-terminated).
///                  Pass null to use the default (~/.talu/plugins/).
///   - out_list: Output parameter to receive plugin list handle.
///
/// Returns: 0 on success, negative error code on failure.
/// On success, caller must free the handle with talu_plugins_list_free().
pub export fn talu_plugins_scan(
    plugins_dir: ?[*:0]const u8,
    out_list: ?*?*CPluginList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_list orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_list is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const dir_slice = if (plugins_dir) |p| @as(?[]const u8, std.mem.span(p)) else null;
    const plugins = scanner.resolveAndScan(allocator, dir_slice) catch |err| {
        const code: error_codes.ErrorCode = switch (err) {
            scanner.ScanError.NoHomeDir => .invalid_argument,
            scanner.ScanError.OutOfMemory => .out_of_memory,
            else => .io_read_failed,
        };
        capi_error.setErrorWithCode(code, "Plugin scan failed: {s}", .{@errorName(err)});
        return @intFromEnum(code);
    };
    defer scanner.freePluginList(allocator, plugins);

    const list = buildPluginList(plugins) catch {
        capi_error.setErrorWithCode(.out_of_memory, "Failed to build plugin list", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };

    out.* = list;
    return 0;
}

/// Get the number of plugins in a list.
///
/// Parameters:
///   - list: Plugin list handle (may be null).
///
/// Returns: Number of plugins, or 0 if list is null.
pub export fn talu_plugins_list_count(list: ?*const CPluginList) callconv(.c) u32 {
    const l = list orelse return 0;
    return @intCast(l.count);
}

/// Get a plugin info record by index.
///
/// Parameters:
///   - list: Plugin list handle.
///   - index: Zero-based index into the list.
///
/// Returns: Pointer to plugin info, or null if out of bounds or list is null.
pub export fn talu_plugins_list_get(list: ?*const CPluginList, index: u32) callconv(.c) ?*const CPluginInfo {
    const l = list orelse return null;
    if (index >= l.count) return null;
    const items = l.items orelse return null;
    return &items[index];
}

/// Free a plugin list returned by talu_plugins_scan.
///
/// Parameters:
///   - list: Plugin list handle to free (may be null).
pub export fn talu_plugins_list_free(list: ?*CPluginList) callconv(.c) void {
    const l = list orelse return;

    if (l._arena) |arena_ptr| {
        const arena: *std.heap.ArenaAllocator = @ptrCast(@alignCast(arena_ptr));
        arena.deinit();
        allocator.destroy(arena);
    }

    allocator.destroy(l);
}
