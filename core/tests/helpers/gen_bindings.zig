//! Python Binding Generator
//!
//! Scans core/src/capi/*.zig files and generates Python ctypes bindings.
//! This ensures the Python bindings are always in sync with the Zig C API.
//!
//! Usage:
//!   zig build gen-bindings
//!
//! Output: bindings/python/talu/_native.py
//!
//! Features:
//!   - Generates ctypes.Structure classes from Zig extern struct definitions
//!   - Auto-detects struct return types and sets correct restype
//!   - Type mapping (Zig -> ctypes)

const std = @import("std");

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

/// Information about an extern struct
const StructInfo = struct {
    name: []const u8,
    fields: []FieldInfo,
    source_file: []const u8,

    const FieldInfo = struct {
        name: []const u8,
        zig_type: []const u8,
        array_size: ?usize = null,
    };
};

/// Information about a C API function
const FunctionSignature = struct {
    name: []const u8,
    params: []ParamInfo,
    return_type: []const u8,
    source_file: []const u8,
    line: usize,

    const ParamInfo = struct {
        name: []const u8,
        zig_type: []const u8,
    };

    /// Check if this function matches the out-pointer pattern:
    /// - First parameter is *?*T (pointer-to-optional-pointer)
    /// - Return type is i32 (error code)
    /// - Name starts with "talu_"
    fn isOutPointerPattern(self: FunctionSignature) bool {
        if (!std.mem.startsWith(u8, self.name, "talu_")) return false;
        if (!eql(self.return_type, "i32")) return false;
        if (self.params.len == 0) return false;

        // Check first param is *?*T (pointer-to-optional-pointer)
        const first_type = self.params[0].zig_type;
        return std.mem.startsWith(u8, first_type, "*?*");
    }

    /// Get a Python-friendly wrapper name (strip talu_ prefix)
    fn getWrapperName(self: FunctionSignature) []const u8 {
        if (std.mem.startsWith(u8, self.name, "talu_")) {
            return self.name["talu_".len..];
        }
        return self.name;
    }
};

/// Information about a callback function type
const CallbackInfo = struct {
    name: []const u8,
    params: []ParamInfo,
    return_type: []const u8,
    source_file: []const u8,

    const ParamInfo = struct {
        name: []const u8,
        zig_type: []const u8,
    };
};

/// Map Zig C-ABI types to Python ctypes
/// Returns null if the type is an extern struct (needs special handling)
fn zigToCtype(zig_type: []const u8, known_structs: *std.StringHashMap(StructInfo)) ?[]const u8 {
    // Basic integer types
    if (eql(zig_type, "i32") or eql(zig_type, "c_int")) return "c_int32";
    if (eql(zig_type, "i64")) return "c_int64";
    if (eql(zig_type, "u8")) return "c_uint8";
    if (eql(zig_type, "u16")) return "c_uint16";
    if (eql(zig_type, "u32")) return "c_uint32";
    if (eql(zig_type, "u64")) return "c_uint64";
    if (eql(zig_type, "usize")) return "c_size_t";
    if (eql(zig_type, "isize")) return "c_ssize_t";

    // Known enum types (map to their underlying integer type)
    // All C-API enums are enum(u8) unless noted otherwise
    if (eql(zig_type, "CPoolingStrategy")) return "c_uint8";
    if (eql(zig_type, "CItemType")) return "c_uint8";
    if (eql(zig_type, "CMessageRole")) return "c_uint8";
    if (eql(zig_type, "CStorageEventType")) return "c_uint8";
    if (eql(zig_type, "ProgressAction")) return "c_uint8"; // enum(u8)
    if (eql(zig_type, "QuantMethodEnum")) return "c_int32"; // enum(i32)

    // Floating point
    if (eql(zig_type, "f32")) return "c_float";
    if (eql(zig_type, "f64")) return "c_double";

    // Boolean
    if (eql(zig_type, "bool")) return "c_bool";

    // Void
    if (eql(zig_type, "void")) return "None";

    // Check if it's a known struct type (before pointer handling)
    if (known_structs.contains(zig_type)) {
        return null; // Signal that this is a struct type
    }

    // Generic pointers -> c_void_p
    if (std.mem.startsWith(u8, zig_type, "?*") or
        std.mem.startsWith(u8, zig_type, "*") or
        std.mem.indexOf(u8, zig_type, "anyopaque") != null)
    {
        return "c_void_p";
    }

    // POINTER types (arrays, slices)
    if (std.mem.startsWith(u8, zig_type, "[*]") or
        std.mem.startsWith(u8, zig_type, "?[*]"))
    {
        // Skip sentinel-terminated pointers [*:0] which are handled separately
        if (std.mem.indexOf(u8, zig_type, ":0]") != null) {
            // Check for array of strings: ?[*][*:0]u8 -> POINTER(c_char_p)
            if (std.mem.indexOf(u8, zig_type, "[*][*:0]u8") != null or
                std.mem.indexOf(u8, zig_type, "?[*][*:0]u8") != null)
            {
                return "POINTER(c_char_p)";
            }
            // Fall through to sentinel handling below
        } else {
            // Double pointer check (non-sentinel)
            var ptr_count: usize = 0;
            var pos: usize = 0;
            while (std.mem.indexOfPos(u8, zig_type, pos, "[*]")) |idx| {
                ptr_count += 1;
                pos = idx + 3;
            }
            if (ptr_count > 1) {
                return "c_void_p";
            }
            if (std.mem.indexOf(u8, zig_type, "f32") != null) return "POINTER(c_float)";
            if (std.mem.indexOf(u8, zig_type, "u32") != null) return "POINTER(c_uint32)";
            if (std.mem.indexOf(u8, zig_type, "usize") != null) return "POINTER(c_size_t)";
            if (std.mem.indexOf(u8, zig_type, "u8") != null) return "c_void_p"; // For function params; struct fields handled separately
            // Check if it's a pointer to a known struct type
            // Format: ?[*]StructName or [*]StructName
            const start_idx = if (std.mem.startsWith(u8, zig_type, "?[*]")) @as(usize, 4) else @as(usize, 3);
            const elem_type = zig_type[start_idx..];
            if (known_structs.contains(elem_type)) {
                return null; // Signal this is POINTER(StructName), handled by caller
            }
            return "c_void_p";
        }
    }

    // Sentinel-terminated string pointers
    if (std.mem.startsWith(u8, zig_type, "[*:0]") or
        std.mem.startsWith(u8, zig_type, "?[*:0]"))
    {
        if (std.mem.indexOf(u8, zig_type, "u8") != null) {
            const is_const = std.mem.indexOf(u8, zig_type, "const") != null;
            if (!is_const) {
                return "c_void_p";
            }
            return "c_char_p";
        }
    }

    // Default: treat as opaque pointer
    return "c_void_p";
}

/// Map Zig type to Python ctypes for struct fields
fn zigToPythonFieldType(zig_type: []const u8, known_structs: *std.StringHashMap(StructInfo)) []const u8 {
    // Check for fixed-size array first: [N]type
    if (std.mem.startsWith(u8, zig_type, "[") and !std.mem.startsWith(u8, zig_type, "[*")) {
        if (std.mem.indexOf(u8, zig_type, "]")) |close_bracket| {
            const elem_type = std.mem.trim(u8, zig_type[close_bracket + 1 ..], " ");
            return zigToPythonFieldType(elem_type, known_structs);
        }
    }

    // Check for pointer-to-struct: ?[*]StructName or [*]StructName (not sentinel-terminated)
    // Also handles ?[*]const StructName
    if ((std.mem.startsWith(u8, zig_type, "?[*]") or std.mem.startsWith(u8, zig_type, "[*]")) and
        std.mem.indexOf(u8, zig_type, ":0]") == null)
    {
        const start_idx = if (std.mem.startsWith(u8, zig_type, "?[*]")) @as(usize, 4) else @as(usize, 3);
        var elem_type = zig_type[start_idx..];
        // Strip 'const ' prefix if present
        if (std.mem.startsWith(u8, elem_type, "const ")) {
            elem_type = elem_type[6..];
        }
        if (known_structs.contains(elem_type)) {
            // Return the struct name - caller will wrap with POINTER()
            return elem_type;
        }
        // For struct fields, [*]u8 or [*]const u8 should be POINTER(c_uint8) to allow slicing
        if (std.mem.eql(u8, elem_type, "u8")) {
            return "POINTER(c_uint8)";
        }
    }

    if (zigToCtype(zig_type, known_structs)) |ctype| {
        return ctype;
    }

    return zig_type;
}

/// Parse array size from type like [3]u8
fn parseArraySize(zig_type: []const u8) ?usize {
    if (!std.mem.startsWith(u8, zig_type, "[") or std.mem.startsWith(u8, zig_type, "[*")) {
        return null;
    }
    if (std.mem.indexOf(u8, zig_type, "]")) |close_bracket| {
        const size_str = zig_type[1..close_bracket];
        return std.fmt.parseInt(usize, size_str, 10) catch null;
    }
    return null;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const project_root = if (args.len > 1) args[1] else ".";
    const output_path = if (args.len > 2) args[2] else "bindings/python/talu/_native.py";

    // Use ArrayListUnmanaged for all collections
    var functions = std.ArrayListUnmanaged(FunctionSignature){};
    defer functions.deinit(allocator);

    var structs = std.StringHashMap(StructInfo).init(allocator);
    defer structs.deinit();

    var aliases = std.StringHashMap([]const u8).init(allocator);
    defer aliases.deinit();

    var callbacks = std.ArrayListUnmanaged(CallbackInfo){};
    defer callbacks.deinit(allocator);

    // Directories to scan
    const scan_dirs = [_][]const u8{
        "core/src/capi",
        "core/src/router",
    };

    for (scan_dirs) |rel_dir| {
        const dir_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ project_root, rel_dir });
        defer allocator.free(dir_path);

        var dir = std.fs.cwd().openDir(dir_path, .{ .iterate = true }) catch |err| {
            if (err == error.FileNotFound) continue;
            std.debug.print("Error: Could not open '{s}': {}\n", .{ dir_path, err });
            return err;
        };
        defer dir.close();

        var walker = try dir.walk(allocator);
        defer walker.deinit();

        while (try walker.next()) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.path, ".zig")) continue;
            if (std.mem.endsWith(u8, entry.path, "_test.zig")) continue;

            const file = try dir.openFile(entry.path, .{});
            defer file.close();

            const source = try file.readToEndAlloc(allocator, 10 * 1024 * 1024);
            defer allocator.free(source);

            const full_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ rel_dir, entry.path });
            defer allocator.free(full_path);

            try extractStructs(allocator, source, full_path, &structs);
            try extractAliases(allocator, source, &aliases);
            try extractCallbacks(allocator, source, full_path, &callbacks);

            if (std.mem.startsWith(u8, rel_dir, "core/src/capi")) {
                try extractFunctions(allocator, source, entry.path, &functions);
            }
        }
    }

    // Sort functions
    std.mem.sort(FunctionSignature, functions.items, {}, struct {
        fn lessThan(_: void, a: FunctionSignature, b: FunctionSignature) bool {
            return std.mem.lessThan(u8, a.name, b.name);
        }
    }.lessThan);

    // Generate output
    const output_full_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ project_root, output_path });
    defer allocator.free(output_full_path);

    // Create parent directories if they don't exist
    if (std.fs.path.dirname(output_full_path)) |parent_dir| {
        std.fs.cwd().makePath(parent_dir) catch |err| {
            if (err != error.PathAlreadyExists) {
                std.debug.print("Error: Could not create directory '{s}': {}\n", .{ parent_dir, err });
                return err;
            }
        };
    }

    const output_file = std.fs.cwd().createFile(output_full_path, .{}) catch |err| {
        std.debug.print("Error: Could not create '{s}': {}\n", .{ output_full_path, err });
        return err;
    };
    defer output_file.close();

    var output_buffer = std.ArrayListUnmanaged(u8){};
    defer output_buffer.deinit(allocator);

    try generatePythonBindings(allocator, output_buffer.writer(allocator), functions.items, &structs, &aliases, callbacks.items);
    try output_file.writeAll(output_buffer.items);

    std.debug.print("Generated {s} with {} functions\n", .{ output_full_path, functions.items.len });
}

fn extractStructs(
    allocator: std.mem.Allocator,
    source: []const u8,
    file_path: []const u8,
    structs: *std.StringHashMap(StructInfo),
) !void {
    var lines = std.mem.splitScalar(u8, source, '\n');
    var in_struct = false;
    var struct_name: []const u8 = "";
    var fields = std.ArrayListUnmanaged(StructInfo.FieldInfo){};
    var brace_depth: i32 = 0;

    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t");

        if (!in_struct) {
            if (std.mem.indexOf(u8, trimmed, "= extern struct")) |_| {
                if (std.mem.indexOf(u8, trimmed, "pub const ")) |const_start| {
                    const name_start = const_start + "pub const ".len;
                    if (std.mem.indexOfPos(u8, trimmed, name_start, " =")) |name_end| {
                        struct_name = trimmed[name_start..name_end];
                        in_struct = true;
                        brace_depth = 1;
                        fields = std.ArrayListUnmanaged(StructInfo.FieldInfo){};
                    }
                }
            }
        } else {
            for (trimmed) |c| {
                if (c == '{') brace_depth += 1;
                if (c == '}') brace_depth -= 1;
            }

            if (brace_depth == 0) {
                const name_copy = try allocator.dupe(u8, struct_name);
                const file_copy = try allocator.dupe(u8, file_path);
                const fields_owned = try allocator.dupe(StructInfo.FieldInfo, fields.items);

                try structs.put(name_copy, .{
                    .name = name_copy,
                    .fields = fields_owned,
                    .source_file = file_copy,
                });
                fields.deinit(allocator);
                in_struct = false;
                continue;
            }

            if (std.mem.indexOf(u8, trimmed, ":")) |colon_pos| {
                if (std.mem.startsWith(u8, trimmed, "pub ") or
                    std.mem.startsWith(u8, trimmed, "fn ") or
                    std.mem.startsWith(u8, trimmed, "comptime ") or
                    std.mem.startsWith(u8, trimmed, "//"))
                {
                    continue;
                }

                const field_name = std.mem.trim(u8, trimmed[0..colon_pos], " \t");

                var type_end = colon_pos + 1;
                while (type_end < trimmed.len) : (type_end += 1) {
                    if (trimmed[type_end] == ',' or trimmed[type_end] == '=') break;
                }
                const field_type = std.mem.trim(u8, trimmed[colon_pos + 1 .. type_end], " \t");

                if (field_type.len > 0) {
                    try fields.append(allocator, .{
                        .name = try allocator.dupe(u8, field_name),
                        .zig_type = try allocator.dupe(u8, field_type),
                        .array_size = parseArraySize(field_type),
                    });
                }
            }
        }
    }
}

fn extractAliases(
    allocator: std.mem.Allocator,
    source: []const u8,
    aliases: *std.StringHashMap([]const u8),
) !void {
    var lines = std.mem.splitScalar(u8, source, '\n');

    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t");

        if (std.mem.startsWith(u8, trimmed, "pub const ")) {
            if (std.mem.indexOf(u8, trimmed, "extern struct") != null) continue;
            if (std.mem.indexOf(u8, trimmed, "enum(") != null) continue;

            const after_const = trimmed["pub const ".len..];
            if (std.mem.indexOf(u8, after_const, " = ")) |eq_pos| {
                const name = after_const[0..eq_pos];
                var target_end = eq_pos + " = ".len;

                while (target_end < after_const.len and after_const[target_end] != ';') {
                    target_end += 1;
                }

                var target = std.mem.trim(u8, after_const[eq_pos + " = ".len .. target_end], " \t");

                if (std.mem.lastIndexOf(u8, target, ".")) |dot_pos| {
                    target = target[dot_pos + 1 ..];
                }

                if (target.len > 0 and std.ascii.isUpper(target[0])) {
                    const name_copy = try allocator.dupe(u8, name);
                    const target_copy = try allocator.dupe(u8, target);
                    try aliases.put(name_copy, target_copy);
                }
            }
        }
    }
}

/// Extract callback function type definitions.
/// Pattern: `pub const CXxxCallback = *const fn (...) callconv(.c) ReturnType;`
fn extractCallbacks(
    allocator: std.mem.Allocator,
    source: []const u8,
    file_path: []const u8,
    callbacks: *std.ArrayListUnmanaged(CallbackInfo),
) !void {
    var lines = std.mem.splitScalar(u8, source, '\n');
    var callback_buffer = std.ArrayListUnmanaged(u8){};
    defer callback_buffer.deinit(allocator);
    var in_callback = false;
    var callback_name: []const u8 = "";

    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t");

        if (!in_callback) {
            // Look for: pub const CXxxCallback = *const fn (
            if (std.mem.startsWith(u8, trimmed, "pub const C") and
                std.mem.indexOf(u8, trimmed, "Callback") != null and
                std.mem.indexOf(u8, trimmed, "*const fn") != null)
            {
                // Extract name
                const after_const = trimmed["pub const ".len..];
                if (std.mem.indexOf(u8, after_const, " =")) |eq_pos| {
                    callback_name = after_const[0..eq_pos];

                    // Check if it ends with "Callback"
                    if (std.mem.endsWith(u8, callback_name, "Callback")) {
                        callback_buffer.clearRetainingCapacity();
                        try callback_buffer.appendSlice(allocator, trimmed);

                        // Check if definition is complete on this line
                        if (std.mem.indexOf(u8, trimmed, ";") != null) {
                            if (try parseCallback(allocator, callback_buffer.items, file_path, callback_name)) |cb| {
                                try callbacks.append(allocator, cb);
                            }
                        } else {
                            in_callback = true;
                        }
                    }
                }
            }
        } else {
            // Continue accumulating callback definition
            try callback_buffer.appendSlice(allocator, " ");
            try callback_buffer.appendSlice(allocator, trimmed);

            if (std.mem.indexOf(u8, trimmed, ";") != null) {
                in_callback = false;
                if (try parseCallback(allocator, callback_buffer.items, file_path, callback_name)) |cb| {
                    try callbacks.append(allocator, cb);
                }
            }
        }
    }
}

/// Parse a callback type definition
fn parseCallback(
    allocator: std.mem.Allocator,
    line: []const u8,
    file_path: []const u8,
    name: []const u8,
) !?CallbackInfo {
    // Find "*const fn ("
    const fn_start = std.mem.indexOf(u8, line, "*const fn (") orelse return null;
    const paren_start = fn_start + "*const fn ".len;

    const paren_end = findMatchingParen(line, paren_start) orelse return null;
    const params_str = line[paren_start + 1 .. paren_end];

    // Parse return type after "callconv(.c)"
    var return_type: []const u8 = "void";
    if (std.mem.indexOf(u8, line[paren_end..], "callconv(.c)")) |cc_offset| {
        const after_cc = paren_end + cc_offset + "callconv(.c)".len;
        var rt_start = after_cc;
        while (rt_start < line.len and (line[rt_start] == ' ' or line[rt_start] == '\t')) {
            rt_start += 1;
        }
        var rt_end = rt_start;
        while (rt_end < line.len and line[rt_end] != ';' and line[rt_end] != '\n') {
            rt_end += 1;
        }
        if (rt_end > rt_start) {
            return_type = std.mem.trim(u8, line[rt_start..rt_end], " \t");
        }
    }

    // Parse parameters
    var params = std.ArrayListUnmanaged(CallbackInfo.ParamInfo){};

    if (params_str.len > 0) {
        var param_iter = std.mem.splitScalar(u8, params_str, ',');
        while (param_iter.next()) |param| {
            const trimmed = std.mem.trim(u8, param, " \t");
            if (trimmed.len == 0) continue;

            // Find colon, handling brackets
            var colon: ?usize = null;
            var bracket_depth: i32 = 0;
            for (trimmed, 0..) |c, i| {
                if (c == '[') bracket_depth += 1;
                if (c == ']') bracket_depth -= 1;
                if (c == ':' and bracket_depth == 0) {
                    colon = i;
                    break;
                }
            }
            if (colon) |colon_pos| {
                const param_name = std.mem.trim(u8, trimmed[0..colon_pos], " \t");
                const param_type = std.mem.trim(u8, trimmed[colon_pos + 1 ..], " \t");
                try params.append(allocator, .{
                    .name = try allocator.dupe(u8, param_name),
                    .zig_type = try allocator.dupe(u8, param_type),
                });
            }
        }
    }

    return CallbackInfo{
        .name = try allocator.dupe(u8, name),
        .params = try allocator.dupe(CallbackInfo.ParamInfo, params.items),
        .return_type = try allocator.dupe(u8, return_type),
        .source_file = try allocator.dupe(u8, file_path),
    };
}

fn extractFunctions(
    allocator: std.mem.Allocator,
    source: []const u8,
    file_path: []const u8,
    functions: *std.ArrayListUnmanaged(FunctionSignature),
) !void {
    var in_function = false;
    var func_buffer = std.ArrayListUnmanaged(u8){};
    defer func_buffer.deinit(allocator);
    var func_start_line: usize = 0;

    var lines = std.mem.splitScalar(u8, source, '\n');
    var line_num: usize = 0;

    while (lines.next()) |line| {
        line_num += 1;

        if (!in_function) {
            if (std.mem.indexOf(u8, line, "pub export fn talu_")) |_| {
                in_function = true;
                func_start_line = line_num;
                func_buffer.clearRetainingCapacity();
                try func_buffer.appendSlice(allocator, line);

                if (std.mem.indexOf(u8, line, "{")) |_| {
                    in_function = false;
                    if (try parseFunction(allocator, func_buffer.items, file_path, func_start_line)) |func| {
                        try functions.append(allocator, func);
                    }
                }
            }
        } else {
            try func_buffer.appendSlice(allocator, " ");
            try func_buffer.appendSlice(allocator, std.mem.trim(u8, line, " \t"));

            if (std.mem.indexOf(u8, line, "{")) |_| {
                in_function = false;
                if (try parseFunction(allocator, func_buffer.items, file_path, func_start_line)) |func| {
                    try functions.append(allocator, func);
                }
            }
        }
    }
}

fn findMatchingParen(text: []const u8, start: usize) ?usize {
    var depth: i32 = 0;
    var i = start;
    while (i < text.len) : (i += 1) {
        if (text[i] == '(') {
            depth += 1;
        } else if (text[i] == ')') {
            depth -= 1;
            if (depth == 0) {
                return i;
            }
        }
    }
    return null;
}

fn parseFunction(
    allocator: std.mem.Allocator,
    line: []const u8,
    file_path: []const u8,
    line_num: usize,
) !?FunctionSignature {
    const fn_start = std.mem.indexOf(u8, line, "pub export fn ") orelse return null;
    const name_start = fn_start + "pub export fn ".len;

    const paren_start = std.mem.indexOfPos(u8, line, name_start, "(") orelse return null;
    const func_name = line[name_start..paren_start];

    const paren_end = findMatchingParen(line, paren_start) orelse return null;
    const params_str = line[paren_start + 1 .. paren_end];

    var return_type: []const u8 = "void";
    if (std.mem.indexOf(u8, line[paren_end..], "callconv(.c)")) |cc_offset| {
        const after_cc = paren_end + cc_offset + "callconv(.c)".len;
        var rt_start = after_cc;
        while (rt_start < line.len and (line[rt_start] == ' ' or line[rt_start] == '\t')) {
            rt_start += 1;
        }
        var rt_end = rt_start;
        while (rt_end < line.len and line[rt_end] != '{' and line[rt_end] != '\n') {
            rt_end += 1;
        }
        if (rt_end > rt_start) {
            return_type = std.mem.trim(u8, line[rt_start..rt_end], " \t");
        }
    }

    var params = std.ArrayListUnmanaged(FunctionSignature.ParamInfo){};

    if (params_str.len > 0) {
        var param_iter = std.mem.splitScalar(u8, params_str, ',');
        while (param_iter.next()) |param| {
            const trimmed = std.mem.trim(u8, param, " \t");
            if (trimmed.len == 0) continue;

            var colon: ?usize = null;
            var bracket_depth: i32 = 0;
            for (trimmed, 0..) |c, i| {
                if (c == '[') bracket_depth += 1;
                if (c == ']') bracket_depth -= 1;
                if (c == ':' and bracket_depth == 0) {
                    colon = i;
                    break;
                }
            }
            if (colon) |colon_pos| {
                const param_name = std.mem.trim(u8, trimmed[0..colon_pos], " \t");
                const param_type = std.mem.trim(u8, trimmed[colon_pos + 1 ..], " \t");
                try params.append(allocator, .{
                    .name = try allocator.dupe(u8, param_name),
                    .zig_type = try allocator.dupe(u8, param_type),
                });
            }
        }
    }

    return FunctionSignature{
        .name = try allocator.dupe(u8, func_name),
        .params = try allocator.dupe(FunctionSignature.ParamInfo, params.items),
        .return_type = try allocator.dupe(u8, return_type),
        .source_file = try allocator.dupe(u8, file_path),
        .line = line_num,
    };
}

fn isStructReturnType(
    return_type: []const u8,
    structs: *std.StringHashMap(StructInfo),
    aliases: *std.StringHashMap([]const u8),
) bool {
    if (structs.contains(return_type)) return true;
    if (aliases.get(return_type)) |target| {
        if (structs.contains(target)) return true;
    }
    return false;
}

fn resolveStructName(
    type_name: []const u8,
    structs: *std.StringHashMap(StructInfo),
    aliases: *std.StringHashMap([]const u8),
) []const u8 {
    if (structs.contains(type_name)) return type_name;
    if (aliases.get(type_name)) |target| {
        if (structs.contains(target)) return target;
    }
    return type_name;
}

fn generatePythonBindings(
    allocator: std.mem.Allocator,
    writer: anytype,
    functions: []const FunctionSignature,
    structs: *std.StringHashMap(StructInfo),
    aliases: *std.StringHashMap([]const u8),
    callbacks: []const CallbackInfo,
) !void {
    // Header
    try writer.writeAll(
        \\"""
        \\Auto-generated Python ctypes bindings for Talu C API.
        \\
        \\DO NOT EDIT - Generated by: zig build gen-bindings
        \\Source: core/src/capi/*.zig
        \\
        \\This module provides:
        \\  - ctypes.Structure classes for all extern structs used in the C API
        \\  - Type-safe function signatures (argtypes/restype) for all exported functions
        \\
        \\Usage:
        \\    from talu._native import setup_signatures
        \\    setup_signatures(lib)
        \\
        \\For structs with helper methods, extend the generated base class:
        \\    from talu._native import EncodeResult as _EncodeResultBase
        \\
        \\    class EncodeResult(_EncodeResultBase):
        \\        def to_list(self) -> list[int]:
        \\            return [self.tokens[i] for i in range(self.num_tokens)]
        \\"""
        \\
        \\from __future__ import annotations
        \\
        \\import ctypes
        \\from ctypes import (
        \\    CFUNCTYPE,
        \\    POINTER,
        \\    Structure,
        \\    c_bool,
        \\    c_char_p,
        \\    c_float,
        \\    c_int32,
        \\    c_int64,
        \\    c_size_t,
        \\    c_uint8,
        \\    c_uint16,
        \\    c_uint32,
        \\    c_uint64,
        \\    c_void_p,
        \\)
        \\
        \\
    );

    // Topologically sort structs by dependencies
    var sorted_structs = std.ArrayListUnmanaged([]const u8){};
    defer sorted_structs.deinit(allocator);
    var visited = std.StringHashMap(void).init(allocator);
    defer visited.deinit();

    // Helper to get struct dependencies
    const getStructDeps = struct {
        fn get(s: *std.StringHashMap(StructInfo), name: []const u8) []const []const u8 {
            if (s.get(name)) |info| {
                var deps: []const []const u8 = &[_][]const u8{};
                for (info.fields) |field| {
                    // Check if field type is a struct (not a pointer to struct)
                    const ft = field.zig_type;
                    if (!std.mem.startsWith(u8, ft, "?") and
                        !std.mem.startsWith(u8, ft, "*") and
                        !std.mem.startsWith(u8, ft, "["))
                    {
                        if (s.contains(ft)) {
                            deps = &[_][]const u8{ft};
                        }
                    }
                }
                return deps;
            }
            return &[_][]const u8{};
        }
    }.get;
    _ = getStructDeps;

    // Simple topological sort - structs with no struct dependencies first
    var struct_it = structs.keyIterator();
    while (struct_it.next()) |key| {
        try sorted_structs.append(allocator, key.*);
    }

    // Sort: structs with dependencies on other structs come after their dependencies
    // This includes both embedded structs AND pointer-to-struct fields (for POINTER(StructName))
    // Simple approach: multiple passes until stable
    var changed = true;
    while (changed) {
        changed = false;
        var i: usize = 0;
        while (i < sorted_structs.items.len) {
            const name = sorted_structs.items[i];
            if (structs.get(name)) |info| {
                for (info.fields) |field| {
                    const ft = field.zig_type;
                    var dep_type: ?[]const u8 = null;

                    // Check for pointer-to-struct: ?[*]StructName or [*]StructName (not sentinel-terminated)
                    // Also handles ?[*]const StructName
                    if ((std.mem.startsWith(u8, ft, "?[*]") or std.mem.startsWith(u8, ft, "[*]")) and
                        std.mem.indexOf(u8, ft, ":0]") == null)
                    {
                        const start_idx = if (std.mem.startsWith(u8, ft, "?[*]")) @as(usize, 4) else @as(usize, 3);
                        var elem_type = ft[start_idx..];
                        // Strip "const " prefix if present
                        if (std.mem.startsWith(u8, elem_type, "const ")) {
                            elem_type = elem_type["const ".len..];
                        }
                        if (structs.contains(elem_type)) {
                            dep_type = elem_type;
                        }
                    }
                    // Check for embedded struct (not a pointer)
                    else if (!std.mem.startsWith(u8, ft, "?") and
                        !std.mem.startsWith(u8, ft, "*") and
                        !std.mem.startsWith(u8, ft, "["))
                    {
                        if (structs.contains(ft)) {
                            dep_type = ft;
                        }
                    }

                    if (dep_type) |dt| {
                        // Check if this dependency comes after us
                        for (sorted_structs.items[i + 1 ..], i + 1..) |other, j| {
                            if (std.mem.eql(u8, dt, other)) {
                                // Dependency comes after us - swap
                                sorted_structs.items[i] = sorted_structs.items[j];
                                sorted_structs.items[j] = name;
                                changed = true;
                                break;
                            }
                        }
                        if (changed) break;
                    }
                }
            }
            if (changed) break;
            i += 1;
        }
    }

    // Generate struct definitions (inline _fields_)
    try writer.writeAll("# =============================================================================\n");
    try writer.writeAll("# Structure definitions (topologically sorted)\n");
    try writer.writeAll("# =============================================================================\n\n");

    for (sorted_structs.items) |name| {
        const info = structs.get(name).?;

        try writer.print("# Source: {s}\n", .{info.source_file});
        try writer.print("class {s}(Structure):\n", .{name});
        try writer.writeAll("    _fields_ = [\n");

        for (info.fields) |field| {
            const py_type = zigToPythonFieldType(field.zig_type, structs);

            // Check if this is a pointer-to-struct field
            const is_ptr_to_struct = (std.mem.startsWith(u8, field.zig_type, "?[*]") or
                std.mem.startsWith(u8, field.zig_type, "[*]")) and
                std.mem.indexOf(u8, field.zig_type, ":0]") == null and
                structs.contains(py_type);

            if (field.array_size) |size| {
                try writer.print("        (\"{s}\", {s} * {d}),\n", .{ field.name, py_type, size });
            } else if (is_ptr_to_struct) {
                try writer.print("        (\"{s}\", POINTER({s})),\n", .{ field.name, py_type });
            } else {
                try writer.print("        (\"{s}\", {s}),\n", .{ field.name, py_type });
            }
        }

        try writer.writeAll("    ]\n\n");
    }

    // Type aliases
    var alias_it = aliases.iterator();
    var has_aliases = false;
    while (alias_it.next()) |entry| {
        if (structs.contains(entry.value_ptr.*)) {
            if (!has_aliases) {
                try writer.writeAll("\n# =============================================================================\n");
                try writer.writeAll("# Type aliases\n");
                try writer.writeAll("# =============================================================================\n\n");
                has_aliases = true;
            }
            try writer.print("{s} = {s}\n", .{ entry.key_ptr.*, entry.value_ptr.* });
        }
    }

    // Callback type definitions
    if (callbacks.len > 0) {
        try writer.writeAll("\n\n# =============================================================================\n");
        try writer.writeAll("# Callback types (CFUNCTYPE definitions)\n");
        try writer.writeAll("# =============================================================================\n");
        try writer.writeAll("#\n");
        try writer.writeAll("# Use these types to create callbacks that can be passed to C functions.\n");
        try writer.writeAll("# Example:\n");
        try writer.writeAll("#     @CProgressCallback\n");
        try writer.writeAll("#     def my_callback(update_ptr, user_data):\n");
        try writer.writeAll("#         update = update_ptr.contents\n");
        try writer.writeAll("#         print(f\"Progress: {update.current}/{update.total}\")\n");
        try writer.writeAll("#\n\n");

        for (callbacks) |cb| {
            try writer.print("# Source: {s}\n", .{cb.source_file});
            try writer.print("{s} = CFUNCTYPE(\n", .{cb.name});

            // Return type
            const return_ctype = zigToCtype(cb.return_type, structs) orelse "c_void_p";
            try writer.print("    {s},  # return type\n", .{return_ctype});

            // Parameters
            for (cb.params, 0..) |param, i| {
                // For pointer to struct, use POINTER(StructName)
                var param_ctype: []const u8 = "c_void_p";
                if (std.mem.startsWith(u8, param.zig_type, "*const ")) {
                    const inner = param.zig_type["*const ".len..];
                    if (structs.contains(inner)) {
                        // Will be handled as POINTER(struct)
                        try writer.print("    POINTER({s}),  # {s}\n", .{ inner, param.name });
                        continue;
                    }
                }
                param_ctype = zigToCtype(param.zig_type, structs) orelse "c_void_p";
                if (i < cb.params.len - 1) {
                    try writer.print("    {s},  # {s}\n", .{ param_ctype, param.name });
                } else {
                    try writer.print("    {s},  # {s}\n", .{ param_ctype, param.name });
                }
            }

            try writer.writeAll(")\n\n");
        }
    }

    // Function signatures
    try writer.writeAll(
        \\
        \\
        \\# =============================================================================
        \\# Function signatures
        \\# =============================================================================
        \\
        \\
        \\def setup_signatures(lib: ctypes.CDLL) -> None:
        \\    """Configure argtypes and restype for all C API functions.
        \\
        \\    Call this once after loading the library to ensure type safety.
        \\    Missing argtypes causes pointer corruption on 64-bit systems.
        \\
        \\    Struct return types are automatically detected and configured.
        \\    """
    );

    var current_file: []const u8 = "";
    for (functions) |func| {
        if (!std.mem.eql(u8, func.source_file, current_file)) {
            current_file = func.source_file;
            try writer.print("\n    # === {s} ===\n", .{current_file});
        }

        try writer.print("    lib.{s}.argtypes = [", .{func.name});

        for (func.params, 0..) |param, i| {
            if (i > 0) try writer.writeAll(", ");
            if (zigToCtype(param.zig_type, structs)) |ctype| {
                try writer.writeAll(ctype);
            } else {
                try writer.writeAll(param.zig_type);
            }
        }

        try writer.writeAll("]\n");

        if (isStructReturnType(func.return_type, structs, aliases)) {
            const resolved = resolveStructName(func.return_type, structs, aliases);
            try writer.print("    lib.{s}.restype = {s}\n", .{ func.name, resolved });
        } else if (zigToCtype(func.return_type, structs)) |restype| {
            try writer.print("    lib.{s}.restype = {s}\n", .{ func.name, restype });
        } else {
            try writer.print("    lib.{s}.restype = c_void_p\n", .{func.name});
        }
    }

    // Generate wrapper functions for out-pointer pattern
    try generateWrapperFunctions(allocator, writer, functions, structs);

    // Footer
    var struct_count: usize = 0;
    var it2 = structs.iterator();
    while (it2.next()) |_| struct_count += 1;

    // Count wrappers
    var wrapper_count: usize = 0;
    for (functions) |func| {
        if (func.isOutPointerPattern()) wrapper_count += 1;
    }

    var alias_names = std.ArrayListUnmanaged([]const u8){};
    defer alias_names.deinit(allocator);
    var alias_it2 = aliases.iterator();
    while (alias_it2.next()) |entry| {
        try alias_names.append(allocator, entry.key_ptr.*);
    }

    try writer.print(
        \\
        \\
        \\# Total: {} functions, {} structs, {} aliases, {} callbacks, {} wrappers
        \\
    , .{ functions.len, struct_count, alias_names.items.len, callbacks.len, wrapper_count });

    // __all__
    try writer.writeAll("\n__all__ = [\n    'setup_signatures',\n");
    for (sorted_structs.items) |name| {
        try writer.print("    '{s}',\n", .{name});
    }

    std.mem.sort([]const u8, alias_names.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lessThan);

    for (alias_names.items) |name| {
        if (aliases.get(name)) |target| {
            if (structs.contains(target)) {
                try writer.print("    '{s}',\n", .{name});
            }
        }
    }

    // Add callback types to __all__
    for (callbacks) |cb| {
        try writer.print("    '{s}',\n", .{cb.name});
    }

    // Add wrapper function names to __all__
    for (functions) |func| {
        if (func.isOutPointerPattern()) {
            try writer.print("    '{s}',\n", .{func.getWrapperName()});
        }
    }

    try writer.writeAll("]\n");
}

/// Generate Python wrapper functions for out-pointer pattern functions.
///
/// For each function matching the pattern:
///   talu_foo(out_ptr: *?*T, a: A, b: B, ...) -> i32
///
/// Generates a Python wrapper:
///   def foo(a: A, b: B, ...) -> int:
///       _out = c_void_p()
///       _err = _lib.talu_foo(byref(_out), a, b, ...)
///       if _err != 0:
///           raise _get_error(_err)
///       return _out.value
fn generateWrapperFunctions(
    allocator: std.mem.Allocator,
    writer: anytype,
    functions: []const FunctionSignature,
    structs: *std.StringHashMap(StructInfo),
) !void {
    // Count wrappers first
    var wrapper_count: usize = 0;
    for (functions) |func| {
        if (func.isOutPointerPattern()) wrapper_count += 1;
    }

    if (wrapper_count == 0) return;

    // Header for wrapper section
    try writer.writeAll(
        \\
        \\
        \\# =============================================================================
        \\# Wrapper functions (auto-generated from out-pointer pattern)
        \\# =============================================================================
        \\#
        \\# These wrappers handle the common C API pattern where the first parameter
        \\# is an output pointer and the return value is an error code.
        \\#
        \\# Usage:
        \\#     ptr = rms_norm(x_ptr, weight_ptr, eps)  # Raises on error
        \\#     # vs raw:
        \\#     out = c_void_p()
        \\#     err = lib.talu_rms_norm(byref(out), x_ptr, weight_ptr, eps)
        \\#     if err != 0: raise ...
        \\#
        \\
        \\# Reference to library - set by setup_wrappers()
        \\_lib: ctypes.CDLL | None = None
        \\
        \\
        \\def _get_lib() -> ctypes.CDLL:
        \\    """Get library reference, raising if not initialized."""
        \\    if _lib is None:
        \\        raise RuntimeError("Library not initialized. Call setup_wrappers() first.")
        \\    return _lib
        \\
        \\
        \\def _get_error(code: int) -> Exception:
        \\    """Get the last error as an exception."""
        \\    lib = _get_lib()
        \\    msg = lib.talu_last_error()
        \\    if msg:
        \\        return RuntimeError(msg.decode("utf-8"))
        \\    return RuntimeError(f"Unknown error (code {code})")
        \\
        \\
        \\def setup_wrappers(lib: ctypes.CDLL) -> None:
        \\    """Initialize wrapper functions with library reference.
        \\
        \\    Call this after setup_signatures() to enable wrapper functions.
        \\    """
        \\    global _lib
        \\    _lib = lib
        \\
        \\
    );

    // Generate each wrapper
    for (functions) |func| {
        if (!func.isOutPointerPattern()) continue;

        const wrapper_name = func.getWrapperName();

        // Generate function signature
        try writer.print("def {s}(", .{wrapper_name});

        // Parameters (skip first out param)
        // No default values - all params are explicit. Users pass None for optional.
        var first_param = true;
        for (func.params[1..]) |param| {
            if (!first_param) try writer.writeAll(", ");
            first_param = false;

            try writer.print("{s}", .{param.name});
            const py_type = zigParamToPythonType(param.zig_type, structs);
            try writer.print(": {s}", .{py_type});
        }

        try writer.writeAll(") -> int:\n");

        // Docstring (imperative mood per D401)
        try writer.print("    \"\"\"Call {s} and return output pointer.\"\"\"\n", .{func.name});

        // Function body
        try writer.writeAll("    _out = c_void_p()\n");
        try writer.print("    _err = _get_lib().{s}(ctypes.byref(_out)", .{func.name});

        // Pass remaining params with type conversions
        for (func.params[1..]) |param| {
            try writer.writeAll(", ");
            try writeParamConversion(writer, param, structs);
        }

        try writer.writeAll(")\n");
        try writer.writeAll("    if _err != 0:\n");
        try writer.writeAll("        raise _get_error(_err)\n");
        try writer.writeAll("    assert _out.value is not None  # Guaranteed by successful error check\n");
        try writer.writeAll("    return _out.value\n\n\n");
    }
    _ = allocator;
}

/// Check if a parameter type is an optional pointer
fn isOptionalPointer(zig_type: []const u8) bool {
    return std.mem.startsWith(u8, zig_type, "?*") or
        std.mem.startsWith(u8, zig_type, "?[*]");
}

/// Check if a parameter name suggests epsilon
fn isEpsParam(name: []const u8) bool {
    _ = name;
    return false; // Currently unused but kept for future use
}

/// Map Zig parameter type to Python type hint
fn zigParamToPythonType(zig_type: []const u8, structs: *std.StringHashMap(StructInfo)) []const u8 {
    // Optional pointers -> int | None
    if (isOptionalPointer(zig_type)) {
        return "int | None";
    }

    // Pointers -> int
    if (std.mem.startsWith(u8, zig_type, "*") or
        std.mem.startsWith(u8, zig_type, "[*]"))
    {
        return "int";
    }

    // Integer types
    if (eql(zig_type, "i32") or eql(zig_type, "c_int")) return "int";
    if (eql(zig_type, "i64")) return "int";
    if (eql(zig_type, "u32") or eql(zig_type, "u64")) return "int";
    if (eql(zig_type, "usize") or eql(zig_type, "isize")) return "int";

    // Float types
    if (eql(zig_type, "f32") or eql(zig_type, "f64")) return "float";

    // Bool
    if (eql(zig_type, "bool")) return "bool";

    _ = structs;
    return "int"; // Default to int for unknown types
}

/// Write parameter conversion for wrapper function call
fn writeParamConversion(
    writer: anytype,
    param: FunctionSignature.ParamInfo,
    structs: *std.StringHashMap(StructInfo),
) !void {
    const zig_type = param.zig_type;

    // Float types need explicit conversion
    if (eql(zig_type, "f32")) {
        try writer.print("c_float({s})", .{param.name});
        return;
    }
    if (eql(zig_type, "f64")) {
        try writer.print("c_double({s})", .{param.name});
        return;
    }

    // Just pass through other types
    try writer.writeAll(param.name);
    _ = structs;
}
