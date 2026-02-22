//! Rust Binding Generator
//!
//! Scans core/src/capi/*.zig files and generates Rust FFI bindings.
//! This ensures the Rust bindings are always in sync with the Zig C API.
//!
//! Usage:
//!   zig build gen-bindings-rust
//!
//! Output: bindings/rust/talu-sys/src/lib.rs
//!
//! Features:
//!   - Generates #[repr(C)] structs from Zig extern struct definitions
//!   - Generates #[repr(T)] enums from Zig enum definitions
//!   - Generates extern "C" function declarations
//!   - Includes Default and From trait implementations

const std = @import("std");

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

/// Information about an extern struct
const StructInfo = struct {
    name: []const u8,
    fields: []FieldInfo,
    source_file: []const u8,
    is_union: bool = false,

    const FieldInfo = struct {
        name: []const u8,
        zig_type: []const u8,
        array_size: ?usize = null,
    };
};

/// Information about an enum
const EnumInfo = struct {
    name: []const u8,
    repr_type: []const u8, // "u8", "i32", "c_int"
    variants: []VariantInfo,
    source_file: []const u8,

    const VariantInfo = struct {
        name: []const u8,
        value: ?i64 = null, // explicit value if provided
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
};

// Known opaque handle types (manually generated in output)
const OPAQUE_HANDLES = [_][]const u8{
    "ResponsesHandle",
    "StringList",
    "CachedModelList",
    "TreeSitterParserHandle",
    "TreeSitterTreeHandle",
    "TreeSitterQueryHandle",
};

fn isOpaqueHandle(name: []const u8) bool {
    for (OPAQUE_HANDLES) |handle| {
        if (eql(name, handle)) return true;
    }
    return false;
}

/// Map Zig C-ABI types to Rust types
fn zigToRustType(zig_type: []const u8, known_structs: *std.StringHashMap(StructInfo), known_enums: *std.StringHashMap(EnumInfo)) []const u8 {
    // Basic integer types
    if (eql(zig_type, "i32") or eql(zig_type, "c_int")) return "c_int";
    if (eql(zig_type, "i64")) return "i64";
    if (eql(zig_type, "u8")) return "u8";
    if (eql(zig_type, "u16")) return "u16";
    if (eql(zig_type, "u32")) return "u32";
    if (eql(zig_type, "u64")) return "u64";
    if (eql(zig_type, "usize")) return "usize";
    if (eql(zig_type, "isize")) return "isize";

    // Floating point
    if (eql(zig_type, "f32")) return "f32";
    if (eql(zig_type, "f64")) return "f64";

    // Boolean
    if (eql(zig_type, "bool")) return "bool";

    // Void
    if (eql(zig_type, "void")) return "()";

    // Check if it's a known enum type
    if (known_enums.contains(zig_type)) {
        return zig_type;
    }

    // Check if it's a known struct type
    if (known_structs.contains(zig_type)) {
        return zig_type;
    }

    // Check for type aliases (e.g., RemoteModelListResult -> CRemoteModelListResult)
    if (eql(zig_type, "RemoteModelListResult")) return "CRemoteModelListResult";
    if (eql(zig_type, "RemoteModelInfo")) return "CRemoteModelInfo";
    // Router types are aliases to C types in capi_bridge.zig
    if (eql(zig_type, "RouterGenerateResult")) return "CGenerateResult";
    if (eql(zig_type, "RouterGenerateConfig")) return "CGenerateConfig";

    // Check for optional or regular pointer to opaque handle: ?*HandleName or *HandleName
    if (std.mem.startsWith(u8, zig_type, "?*") or std.mem.startsWith(u8, zig_type, "*")) {
        const start_idx: usize = if (std.mem.startsWith(u8, zig_type, "?*")) 2 else 1;
        // Handle "const " prefix in pointee
        var pointee = zig_type[start_idx..];
        const is_const = std.mem.startsWith(u8, pointee, "const ");
        if (is_const) {
            pointee = pointee[6..]; // Skip "const "
        }
        if (isOpaqueHandle(pointee)) {
            // Return a formatted string like "*mut ResponsesHandle"
            // Since we can't allocate here, we use a static buffer trick or return a known constant
            // For now, handle the common cases
            if (eql(pointee, "ResponsesHandle")) return "*mut ResponsesHandle";
            if (eql(pointee, "StringList")) return "*mut StringList";
            if (eql(pointee, "CachedModelList")) return "*mut CachedModelList";
        }
        // Check for type aliases (Router* -> C*)
        if (eql(pointee, "RouterGenerateConfig")) {
            return if (is_const) "*const CGenerateConfig" else "*mut CGenerateConfig";
        }
        if (eql(pointee, "RouterGenerateResult")) {
            return if (is_const) "*const CGenerateResult" else "*mut CGenerateResult";
        }
        // Check if it points to a known struct - need to return with pointer wrapper
        // We can't allocate, so use static strings for common types
        if (known_structs.contains(pointee)) {
            // Common output parameter types
            if (eql(pointee, "CItem")) return "*mut CItem";
            if (eql(pointee, "CMessageItem")) return "*mut CMessageItem";
            if (eql(pointee, "CFunctionCallItem")) return "*mut CFunctionCallItem";
            if (eql(pointee, "CFunctionCallOutputItem")) return "*mut CFunctionCallOutputItem";
            if (eql(pointee, "CReasoningItem")) return "*mut CReasoningItem";
            if (eql(pointee, "CItemReferenceItem")) return "*mut CItemReferenceItem";
            if (eql(pointee, "CContentPart")) return "*mut CContentPart";
            if (eql(pointee, "CGenerateResult")) return "*mut CGenerateResult";
            if (eql(pointee, "CGenerateConfig")) return if (is_const) "*const CGenerateConfig" else "*mut CGenerateConfig";
            if (eql(pointee, "GenerateContentPart")) return if (is_const) "*const GenerateContentPart" else "*mut GenerateContentPart";
            if (eql(pointee, "CProviderInfo")) return "*mut CProviderInfo";
            if (eql(pointee, "TaluModelSpec")) return "*mut TaluModelSpec";
            if (eql(pointee, "TaluCapabilities")) return "*mut TaluCapabilities";
            if (eql(pointee, "ModelInfo")) return "*mut ModelInfo";
            if (eql(pointee, "EncodeResult")) return "*mut EncodeResult";
            if (eql(pointee, "TokenizeResult")) return "*mut TokenizeResult";
            if (eql(pointee, "EosTokenResult")) return "*mut EosTokenResult";
            if (eql(pointee, "GenerationConfigInfo")) return "*mut GenerationConfigInfo";
            if (eql(pointee, "ConvertResult")) return "*mut ConvertResult";
            if (eql(pointee, "DownloadOptions")) return "*mut DownloadOptions";
            if (eql(pointee, "ChatCreateOptions")) return "*mut ChatCreateOptions";
            if (eql(pointee, "CRemoteModelListResult")) return "*mut CRemoteModelListResult";
            if (eql(pointee, "ConvertOptions")) return if (is_const) "*const ConvertOptions" else "*mut ConvertOptions";
            if (eql(pointee, "CSessionList")) return if (is_const) "*const CSessionList" else "*mut CSessionList";
            if (eql(pointee, "CSessionRecord")) return if (is_const) "*const CSessionRecord" else "*mut CSessionRecord";
            if (eql(pointee, "CapturedTensorInfo")) return if (is_const) "*const CapturedTensorInfo" else "*mut CapturedTensorInfo";
            if (eql(pointee, "CStringList")) return if (is_const) "*const CStringList" else "*mut CStringList";
            if (eql(pointee, "CTagList")) return if (is_const) "*const CTagList" else "*mut CTagList";
            if (eql(pointee, "CTagRecord")) return if (is_const) "*const CTagRecord" else "*mut CTagRecord";
            // File/image C API structs
            if (eql(pointee, "TaluImage")) return if (is_const) "*const TaluImage" else "*mut TaluImage";
            if (eql(pointee, "TaluImageDecodeOptions")) return if (is_const) "*const TaluImageDecodeOptions" else "*mut TaluImageDecodeOptions";
            if (eql(pointee, "TaluImageConvertOptions")) return if (is_const) "*const TaluImageConvertOptions" else "*mut TaluImageConvertOptions";
            if (eql(pointee, "TaluImageResizeOptions")) return if (is_const) "*const TaluImageResizeOptions" else "*mut TaluImageResizeOptions";
            if (eql(pointee, "TaluImageEncodeOptions")) return if (is_const) "*const TaluImageEncodeOptions" else "*mut TaluImageEncodeOptions";
            if (eql(pointee, "TaluModelInputSpec")) return if (is_const) "*const TaluModelInputSpec" else "*mut TaluModelInputSpec";
            if (eql(pointee, "TaluModelBuffer")) return if (is_const) "*const TaluModelBuffer" else "*mut TaluModelBuffer";
            if (eql(pointee, "TaluFileInfo")) return if (is_const) "*const TaluFileInfo" else "*mut TaluFileInfo";
            if (eql(pointee, "TaluImageInfo")) return if (is_const) "*const TaluImageInfo" else "*mut TaluImageInfo";
            if (eql(pointee, "TaluFileTransformOptions")) return if (is_const) "*const TaluFileTransformOptions" else "*mut TaluFileTransformOptions";
            // Document types
            if (eql(pointee, "CDocumentRecord")) return if (is_const) "*const CDocumentRecord" else "*mut CDocumentRecord";
            if (eql(pointee, "CDocumentSummary")) return if (is_const) "*const CDocumentSummary" else "*mut CDocumentSummary";
            if (eql(pointee, "CDocumentList")) return if (is_const) "*const CDocumentList" else "*mut CDocumentList";
            if (eql(pointee, "CSearchResult")) return if (is_const) "*const CSearchResult" else "*mut CSearchResult";
            if (eql(pointee, "CSearchResultList")) return if (is_const) "*const CSearchResultList" else "*mut CSearchResultList";
            if (eql(pointee, "CChangeRecord")) return if (is_const) "*const CChangeRecord" else "*mut CChangeRecord";
            if (eql(pointee, "CChangeList")) return if (is_const) "*const CChangeList" else "*mut CChangeList";
            if (eql(pointee, "CDeltaChain")) return if (is_const) "*const CDeltaChain" else "*mut CDeltaChain";
            if (eql(pointee, "CCompactionStats")) return if (is_const) "*const CCompactionStats" else "*mut CCompactionStats";
            // Plugin types
            if (eql(pointee, "CPluginInfo")) return if (is_const) "*const CPluginInfo" else "*mut CPluginInfo";
            if (eql(pointee, "CPluginList")) return if (is_const) "*const CPluginList" else "*mut CPluginList";
            // Fallback: return without wrapper (will cause compile error if used)
            return pointee;
        }
    }

    // Check for pointer-to-optional-pointer pattern: *?*Type (output param pattern)
    if (std.mem.startsWith(u8, zig_type, "*?*")) {
        // This is a double-pointer pattern used for output params
        return "*mut *mut c_void";
    }

    // Generic pointers -> *mut c_void or *const c_void
    if (std.mem.startsWith(u8, zig_type, "?*") or
        std.mem.startsWith(u8, zig_type, "*") or
        std.mem.indexOf(u8, zig_type, "anyopaque") != null)
    {
        return "*mut c_void";
    }

    // POINTER types (arrays, slices)
    if (std.mem.startsWith(u8, zig_type, "[*]") or
        std.mem.startsWith(u8, zig_type, "?[*]"))
    {
        // Check if the OUTER pointer is sentinel-terminated (e.g. [*:0]u8, ?[*:0]u8).
        // Must not match inner sentinels in types like ?[*]const ?[*:0]const u8.
        const is_outer_sentinel = std.mem.startsWith(u8, zig_type, "[*:0]") or
            std.mem.startsWith(u8, zig_type, "?[*:0]");
        if (is_outer_sentinel) {
            // String pointer
            if (std.mem.indexOf(u8, zig_type, "u8") != null) {
                return "*const c_char";
            }
        }
        // Array of strings: outer [*] pointing to inner string pointers [*:0]u8.
        // Handles patterns like ?[*]const ?[*:0]const u8, [*][*:0]u8, etc.
        if (!is_outer_sentinel and std.mem.indexOf(u8, zig_type, "[*:0]") != null) {
            if (std.mem.indexOf(u8, zig_type, "u8") != null) {
                return "*const *const c_char";
            }
        }

        // Check element type
        if (std.mem.indexOf(u8, zig_type, "f32") != null) return "*const f32";
        if (std.mem.indexOf(u8, zig_type, "u32") != null) return "*const u32";
        if (std.mem.indexOf(u8, zig_type, "usize") != null) return "*const usize";
        if (std.mem.indexOf(u8, zig_type, "u8") != null) return "*const u8";

        // Check if pointer to known struct
        const start_idx = if (std.mem.startsWith(u8, zig_type, "?[*]")) @as(usize, 4) else @as(usize, 3);
        var elem_type = zig_type[start_idx..];
        // Strip 'const ' prefix if present
        const is_const = std.mem.startsWith(u8, elem_type, "const ");
        if (is_const) {
            elem_type = elem_type[6..];
        }
        if (known_structs.contains(elem_type)) {
            // Return pointer to struct with correct constness
            if (eql(elem_type, "GenerateContentPart")) {
                return if (is_const) "*const GenerateContentPart" else "*mut GenerateContentPart";
            }
            if (eql(elem_type, "CLogitBiasEntry")) {
                return if (is_const) "*const CLogitBiasEntry" else "*mut CLogitBiasEntry";
            }
            if (eql(elem_type, "CToolCallRef")) {
                return if (is_const) "*const CToolCallRef" else "*mut CToolCallRef";
            }
            if (eql(elem_type, "OverrideRule")) {
                return if (is_const) "*const OverrideRule" else "*mut OverrideRule";
            }
            if (eql(elem_type, "CStorageRecord")) {
                return if (is_const) "*const CStorageRecord" else "*mut CStorageRecord";
            }
            // Default fallback — log a warning if we hit this; likely a new struct
            // that needs an entry above.
            return elem_type;
        }

        return "*mut c_void";
    }

    // Sentinel-terminated string pointers
    if (std.mem.startsWith(u8, zig_type, "[*:0]") or
        std.mem.startsWith(u8, zig_type, "?[*:0]"))
    {
        if (std.mem.indexOf(u8, zig_type, "u8") != null) {
            return "*const c_char";
        }
    }

    // Default: treat as opaque pointer
    return "*mut c_void";
}

/// Map Zig type to Rust type for struct fields
fn zigToRustFieldType(zig_type: []const u8, known_structs: *std.StringHashMap(StructInfo), known_enums: *std.StringHashMap(EnumInfo)) []const u8 {
    // Check for fixed-size array first: [N]type
    // Note: For arrays, we return the element type; caller handles array syntax
    if (std.mem.startsWith(u8, zig_type, "[") and !std.mem.startsWith(u8, zig_type, "[*")) {
        if (std.mem.indexOf(u8, zig_type, "]")) |close_bracket| {
            const elem_type = std.mem.trim(u8, zig_type[close_bracket + 1 ..], " ");
            // Get the element type recursively
            const rust_elem = zigToRustFieldType(elem_type, known_structs, known_enums);
            // For basic types, return them directly
            if (std.mem.eql(u8, rust_elem, "u8") or
                std.mem.eql(u8, rust_elem, "u16") or
                std.mem.eql(u8, rust_elem, "u32") or
                std.mem.eql(u8, rust_elem, "u64") or
                std.mem.eql(u8, rust_elem, "i32") or
                std.mem.eql(u8, rust_elem, "i64") or
                std.mem.eql(u8, rust_elem, "f32") or
                std.mem.eql(u8, rust_elem, "f64") or
                std.mem.eql(u8, rust_elem, "bool"))
            {
                return rust_elem;
            }
            // For other types, return element type (caller adds array syntax)
            return rust_elem;
        }
    }

    // Check for pointer-to-struct: ?[*]StructName or [*]StructName (not sentinel-terminated)
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
            return elem_type; // Caller will wrap with *const/*mut
        }
        // [*]u8 -> *const u8
        if (std.mem.eql(u8, elem_type, "u8")) {
            return "*const u8";
        }
    }

    return zigToRustType(zig_type, known_structs, known_enums);
}

/// Known compile-time constants used in array sizes.
/// Maps constant names to their values.
const KNOWN_CONSTANTS = std.StaticStringMap(usize).initComptime(.{
    .{ "MAX_OVERRIDES", 32 },
});

/// Parse array size from type like [3]u8 or [MAX_OVERRIDES]OverrideRule
fn parseArraySize(zig_type: []const u8) ?usize {
    if (!std.mem.startsWith(u8, zig_type, "[") or std.mem.startsWith(u8, zig_type, "[*")) {
        return null;
    }
    if (std.mem.indexOf(u8, zig_type, "]")) |close_bracket| {
        const size_str = zig_type[1..close_bracket];
        // First try to parse as integer
        if (std.fmt.parseInt(usize, size_str, 10)) |size| {
            return size;
        } else |_| {
            // Try to resolve as known constant
            return KNOWN_CONSTANTS.get(size_str);
        }
    }
    return null;
}

/// Get Rust repr type from Zig enum repr
fn zigEnumReprToRust(zig_repr: []const u8) []const u8 {
    if (eql(zig_repr, "u8")) return "u8";
    if (eql(zig_repr, "i32") or eql(zig_repr, "c_int")) return "i32";
    if (eql(zig_repr, "u32")) return "u32";
    return "i32"; // default
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const project_root = if (args.len > 1) args[1] else ".";
    const output_path = if (args.len > 2) args[2] else "bindings/rust/talu-sys/src/lib.rs";

    var functions = std.ArrayListUnmanaged(FunctionSignature){};
    defer functions.deinit(allocator);

    var structs = std.StringHashMap(StructInfo).init(allocator);
    defer structs.deinit();

    var enums = std.StringHashMap(EnumInfo).init(allocator);
    defer enums.deinit();

    // Directories to scan
    // Scan capi, router, and converter directories.
    // The converter module has complex enums (with methods) but the parser handles them.
    const scan_dirs = [_][]const u8{
        "core/src/capi",
        "core/src/router",
        "core/src/converter",
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

            // Parse extern structs
            try parseExternStructs(allocator, source, full_path, &structs);

            // Parse enums
            try parseEnums(allocator, source, full_path, &enums);

            // Resolve re-exports (pub const X = imported.X;) by reading imported files
            try resolveReExports(allocator, source, full_path, project_root, &structs, &enums);

            // Parse exported functions
            try parseExportedFunctions(allocator, source, full_path, &functions);
        }
    }

    // Generate Rust output to buffer first, then write
    var output_buffer = std.ArrayListUnmanaged(u8){};
    defer output_buffer.deinit(allocator);
    const writer = output_buffer.writer(allocator);

    // Write header
    try writer.writeAll(
        \\// =============================================================================
        \\// AUTO-GENERATED FILE - DO NOT EDIT MANUALLY
        \\//
        \\// Generated by: zig build gen-bindings-rust
        \\// Source:        core/src/capi/*.zig
        \\// Generator:     core/tests/helpers/gen_bindings_rust.zig
        \\//
        \\// Any manual edits WILL BE LOST on the next regeneration.
        \\// To add safe wrappers, edit the generator template instead.
        \\// =============================================================================
        \\
        \\//! Low-level FFI bindings for libtalu.
        \\//!
        \\//! This crate provides raw, unsafe bindings to the C API of libtalu.
        \\//! For a safe, idiomatic Rust API, use the `talu` crate instead.
        \\
        \\#![allow(dead_code)]
        \\#![allow(non_camel_case_types)]
        \\#![allow(clippy::missing_safety_doc)]
        \\
        \\use std::os::raw::{c_char, c_int, c_void};
        \\
        \\
    );

    // Write enums first (they may be used in structs)
    try writer.writeAll(
        \\// =============================================================================
        \\// Enums
        \\// =============================================================================
        \\
        \\
    );

    var enum_it = enums.iterator();
    while (enum_it.next()) |entry| {
        const info = entry.value_ptr.*;
        const rust_repr = zigEnumReprToRust(info.repr_type);

        try writer.print("/// Source: {s}\n", .{info.source_file});
        try writer.print("#[repr({s})]\n", .{rust_repr});
        try writer.writeAll("#[derive(Copy, Clone, Debug, PartialEq, Eq)]\n");
        try writer.print("pub enum {s} {{\n", .{info.name});

        for (info.variants) |variant| {
            if (variant.value) |v| {
                try writer.print("    {s} = {d},\n", .{ variant.name, v });
            } else {
                try writer.print("    {s},\n", .{variant.name});
            }
        }

        try writer.writeAll("}\n\n");

        // Generate From<repr_type> implementation
        try writer.print("impl From<{s}> for {s} {{\n", .{ rust_repr, info.name });
        try writer.print("    fn from(value: {s}) -> Self {{\n", .{rust_repr});
        try writer.writeAll("        match value {\n");

        for (info.variants) |variant| {
            if (variant.value) |v| {
                try writer.print("            {d} => {s}::{s},\n", .{ v, info.name, variant.name });
            }
        }

        // Default case
        if (info.variants.len > 0) {
            try writer.print("            _ => {s}::{s},\n", .{ info.name, info.variants[info.variants.len - 1].name });
        }

        try writer.writeAll("        }\n");
        try writer.writeAll("    }\n");
        try writer.writeAll("}\n\n");
    }

    // Topologically sort structs
    var sorted_structs = std.ArrayListUnmanaged([]const u8){};
    defer sorted_structs.deinit(allocator);

    var struct_it = structs.keyIterator();
    while (struct_it.next()) |key| {
        try sorted_structs.append(allocator, key.*);
    }

    // Sort: structs with dependencies come after their dependencies
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

                    // Check for pointer-to-struct
                    if ((std.mem.startsWith(u8, ft, "?[*]") or std.mem.startsWith(u8, ft, "[*]")) and
                        std.mem.indexOf(u8, ft, ":0]") == null)
                    {
                        const start_idx = if (std.mem.startsWith(u8, ft, "?[*]")) @as(usize, 4) else @as(usize, 3);
                        var elem_type = ft[start_idx..];
                        if (std.mem.startsWith(u8, elem_type, "const ")) {
                            elem_type = elem_type[6..];
                        }
                        if (structs.contains(elem_type)) {
                            dep_type = elem_type;
                        }
                    }
                    // Check for embedded struct
                    else if (!std.mem.startsWith(u8, ft, "?") and
                        !std.mem.startsWith(u8, ft, "*") and
                        !std.mem.startsWith(u8, ft, "["))
                    {
                        if (structs.contains(ft)) {
                            dep_type = ft;
                        }
                    }

                    if (dep_type) |dep| {
                        // Find position of dependency
                        for (sorted_structs.items, 0..) |s, j| {
                            if (std.mem.eql(u8, s, dep) and j > i) {
                                // Dependency comes after - swap
                                sorted_structs.items[i] = dep;
                                sorted_structs.items[j] = name;
                                changed = true;
                                break;
                            }
                        }
                    }
                }
            }
            i += 1;
        }
    }

    // Write structs
    try writer.writeAll(
        \\// =============================================================================
        \\// Structures
        \\// =============================================================================
        \\
        \\
    );

    for (sorted_structs.items) |name| {
        const info = structs.get(name).?;

        try writer.print("/// Source: {s}\n", .{info.source_file});
        try writer.writeAll("#[repr(C)]\n");
        try writer.writeAll("#[derive(Copy, Clone)]\n");
        const keyword = if (info.is_union) "union" else "struct";
        try writer.print("pub {s} {s} {{\n", .{ keyword, name });

        for (info.fields) |field| {
            const rust_type = zigToRustFieldType(field.zig_type, &structs, &enums);

            // Check if this is a pointer-to-struct field
            const is_ptr_to_struct = (std.mem.startsWith(u8, field.zig_type, "?[*]") or
                std.mem.startsWith(u8, field.zig_type, "[*]")) and
                std.mem.indexOf(u8, field.zig_type, ":0]") == null and
                structs.contains(rust_type);

            // Handle reserved field name
            var field_name = field.name;
            if (std.mem.eql(u8, field_name, "ref")) {
                field_name = "ref_";
            } else if (std.mem.eql(u8, field_name, "type")) {
                field_name = "type_";
            }

            if (field.array_size) |size| {
                try writer.print("    pub {s}: [{s}; {d}],\n", .{ field_name, rust_type, size });
            } else if (is_ptr_to_struct) {
                // Determine const vs mut based on original type
                const ptr_kind = if (std.mem.indexOf(u8, field.zig_type, "const ") != null)
                    "*const"
                else
                    "*mut";
                try writer.print("    pub {s}: {s} {s},\n", .{ field_name, ptr_kind, rust_type });
            } else {
                try writer.print("    pub {s}: {s},\n", .{ field_name, rust_type });
            }
        }

        try writer.writeAll("}\n\n");

        // Generate Default implementation (skip for unions - they need manual impl)
        if (info.is_union) continue;

        try writer.print("impl Default for {s} {{\n", .{name});
        try writer.writeAll("    fn default() -> Self {\n");
        try writer.writeAll("        Self {\n");

        for (info.fields) |field| {
            const rust_type = zigToRustFieldType(field.zig_type, &structs, &enums);

            // Check if this is a pointer-to-struct field
            const is_ptr_to_struct = (std.mem.startsWith(u8, field.zig_type, "?[*]") or
                std.mem.startsWith(u8, field.zig_type, "[*]")) and
                std.mem.indexOf(u8, field.zig_type, ":0]") == null and
                structs.contains(rust_type);

            // Handle reserved field name
            var field_name = field.name;
            if (std.mem.eql(u8, field_name, "ref")) {
                field_name = "ref_";
            } else if (std.mem.eql(u8, field_name, "type")) {
                field_name = "type_";
            }

            // Generate default value based on type
            if (field.array_size) |size| {
                // For struct arrays, use Default::default() for each element
                if (structs.contains(rust_type)) {
                    try writer.print("            {s}: [{s}::default(); {d}],\n", .{ field_name, rust_type, size });
                } else {
                    try writer.print("            {s}: [0; {d}],\n", .{ field_name, size });
                }
            } else if (is_ptr_to_struct) {
                // Determine const vs mut from original type
                if (std.mem.indexOf(u8, field.zig_type, "const ") != null) {
                    try writer.print("            {s}: std::ptr::null(),\n", .{field_name});
                } else {
                    try writer.print("            {s}: std::ptr::null_mut(),\n", .{field_name});
                }
            } else if (std.mem.eql(u8, rust_type, "*const c_void") or
                std.mem.eql(u8, rust_type, "*const c_char") or
                std.mem.eql(u8, rust_type, "*const *const c_char") or
                std.mem.eql(u8, rust_type, "*const u8") or
                std.mem.eql(u8, rust_type, "*const f32") or
                std.mem.eql(u8, rust_type, "*const u32") or
                std.mem.eql(u8, rust_type, "*const usize"))
            {
                try writer.print("            {s}: std::ptr::null(),\n", .{field_name});
            } else if (std.mem.eql(u8, rust_type, "*mut c_void") or
                std.mem.eql(u8, rust_type, "*mut u8"))
            {
                try writer.print("            {s}: std::ptr::null_mut(),\n", .{field_name});
            } else if (std.mem.eql(u8, rust_type, "bool")) {
                try writer.print("            {s}: false,\n", .{field_name});
            } else if (std.mem.eql(u8, rust_type, "f32") or std.mem.eql(u8, rust_type, "f64")) {
                try writer.print("            {s}: 0.0,\n", .{field_name});
            } else if (enums.contains(rust_type)) {
                // For enums, use From<0> to get the default variant
                try writer.print("            {s}: {s}::from(0),\n", .{ field_name, rust_type });
            } else if (structs.contains(rust_type)) {
                // Check if it's a union - unions need unsafe zeroed init
                const embedded_struct = structs.get(rust_type).?;
                if (embedded_struct.is_union) {
                    try writer.print("            {s}: unsafe {{ std::mem::zeroed() }},\n", .{field_name});
                } else {
                    // Embedded struct, use Default::default()
                    try writer.print("            {s}: {s}::default(),\n", .{ field_name, rust_type });
                }
            } else if (std.mem.startsWith(u8, rust_type, "*const ")) {
                try writer.print("            {s}: std::ptr::null(),\n", .{field_name});
            } else if (std.mem.startsWith(u8, rust_type, "*mut ")) {
                try writer.print("            {s}: std::ptr::null_mut(),\n", .{field_name});
            } else {
                // Numeric types default to 0
                try writer.print("            {s}: 0,\n", .{field_name});
            }
        }

        try writer.writeAll("        }\n");
        try writer.writeAll("    }\n");
        try writer.writeAll("}\n\n");
    }

    // Write opaque handle type
    try writer.writeAll(
        \\// =============================================================================
        \\// Opaque Handles
        \\// =============================================================================
        \\
        \\/// Opaque handle for Responses/Conversation.
        \\#[repr(C)]
        \\#[derive(Copy, Clone)]
        \\pub struct ResponsesHandle {
        \\    _private: [u8; 0],
        \\}
        \\
        \\/// Opaque handle for string list.
        \\#[repr(C)]
        \\#[derive(Copy, Clone)]
        \\pub struct StringList {
        \\    _private: [u8; 0],
        \\}
        \\
        \\/// Opaque handle for cached model list.
        \\#[repr(C)]
        \\#[derive(Copy, Clone)]
        \\pub struct CachedModelList {
        \\    _private: [u8; 0],
        \\}
        \\
        \\/// Opaque handle for tree-sitter parser.
        \\/// Thread safety: NOT thread-safe. Create one per thread.
        \\#[repr(C)]
        \\#[derive(Copy, Clone)]
        \\pub struct TreeSitterParserHandle {
        \\    _private: [u8; 0],
        \\}
        \\
        \\/// Opaque handle for parsed syntax tree.
        \\/// Thread safety: Immutable after creation. Safe to share read-only.
        \\#[repr(C)]
        \\#[derive(Copy, Clone)]
        \\pub struct TreeSitterTreeHandle {
        \\    _private: [u8; 0],
        \\}
        \\
        \\/// Opaque handle for compiled query pattern.
        \\/// Thread safety: Immutable after creation. Safe to share read-only.
        \\#[repr(C)]
        \\#[derive(Copy, Clone)]
        \\pub struct TreeSitterQueryHandle {
        \\    _private: [u8; 0],
        \\}
        \\
        \\
    );

    // Write callback types
    try writer.writeAll(
        \\// =============================================================================
        \\// Callback Types
        \\// =============================================================================
        \\
        \\pub type RouterTokenCallback =
        \\    Option<unsafe extern "C" fn(*const c_char, u32, *mut c_void) -> bool>;
        \\
        \\/// Unified progress callback - receives structured progress updates from core.
        \\/// This is the new callback type used by all long-running operations.
        \\pub type CProgressCallback =
        \\    Option<unsafe extern "C" fn(*const ProgressUpdate, *mut c_void)>;
        \\
        \\/// Legacy progress callback (deprecated - use CProgressCallback).
        \\pub type ProgressCallback =
        \\    Option<unsafe extern "C" fn(u64, u64, *mut c_void)>;
        \\
        \\/// Legacy file start callback (deprecated - use CProgressCallback).
        \\pub type FileStartCallback =
        \\    Option<unsafe extern "C" fn(*const c_char, *mut c_void)>;
        \\
        \\pub type StorageCallback =
        \\    Option<unsafe extern "C" fn(*const CStorageEvent, *mut c_void) -> i32>;
        \\
        \\
    );

    // Write type aliases for backwards compatibility
    try writer.writeAll(
        \\// =============================================================================
        \\// Type Aliases (Backwards Compatibility)
        \\// =============================================================================
        \\//
        \\// These aliases allow wrapper code to use cleaner names without the C prefix.
        \\// New code should prefer the C-prefixed names to match the Zig C API.
        \\
        \\pub type MessageRole = CMessageRole;
        \\pub type ItemType = CItemType;
        \\pub type StorageEventType = CStorageEventType;
        \\pub type PoolingStrategy = CPoolingStrategy;
        \\
        \\/// Alias for CContentPart (legacy name used in some wrapper code)
        \\pub type CResponsesContentPart = CContentPart;
        \\
        \\// =============================================================================
        \\// Semantic Enums (Not in Zig C API, but documented in field comments)
        \\// =============================================================================
        \\//
        \\// These enums represent values stored in u8 fields with documented meanings.
        \\// They are manually defined here for type safety in Rust wrapper code.
        \\
        \\/// Content type discriminator.
        \\/// Matches values documented in CContentPart.content_type.
        \\#[repr(u8)]
        \\#[derive(Copy, Clone, Debug, PartialEq, Eq)]
        \\pub enum ContentType {
        \\    InputText = 0,
        \\    InputImage = 1,
        \\    InputAudio = 2,
        \\    InputFile = 3,
        \\    InputVideo = 4,
        \\    OutputText = 5,
        \\    Refusal = 6,
        \\    Text = 7,
        \\    ReasoningText = 8,
        \\    SummaryText = 9,
        \\    Unknown = 255,
        \\}
        \\
        \\impl From<u8> for ContentType {
        \\    fn from(value: u8) -> Self {
        \\        match value {
        \\            0 => ContentType::InputText,
        \\            1 => ContentType::InputImage,
        \\            2 => ContentType::InputAudio,
        \\            3 => ContentType::InputFile,
        \\            4 => ContentType::InputVideo,
        \\            5 => ContentType::OutputText,
        \\            6 => ContentType::Refusal,
        \\            7 => ContentType::Text,
        \\            8 => ContentType::ReasoningText,
        \\            9 => ContentType::SummaryText,
        \\            _ => ContentType::Unknown,
        \\        }
        \\    }
        \\}
        \\
        \\/// Image detail level.
        \\/// Matches values documented in CContentPart.image_detail.
        \\#[repr(u8)]
        \\#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
        \\pub enum ImageDetail {
        \\    #[default]
        \\    Auto = 0,
        \\    Low = 1,
        \\    High = 2,
        \\}
        \\
        \\impl From<u8> for ImageDetail {
        \\    fn from(value: u8) -> Self {
        \\        match value {
        \\            0 => ImageDetail::Auto,
        \\            1 => ImageDetail::Low,
        \\            2 => ImageDetail::High,
        \\            _ => ImageDetail::Auto,
        \\        }
        \\    }
        \\}
        \\
        \\/// Item status.
        \\/// Matches values documented in CItem.status and CStorageRecord.status.
        \\#[repr(u8)]
        \\#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
        \\pub enum ItemStatus {
        \\    #[default]
        \\    InProgress = 0,
        \\    Waiting = 1,
        \\    Completed = 2,
        \\    Incomplete = 3,
        \\    Failed = 4,
        \\}
        \\
        \\impl From<u8> for ItemStatus {
        \\    fn from(value: u8) -> Self {
        \\        match value {
        \\            0 => ItemStatus::InProgress,
        \\            1 => ItemStatus::Waiting,
        \\            2 => ItemStatus::Completed,
        \\            3 => ItemStatus::Incomplete,
        \\            4 => ItemStatus::Failed,
        \\            _ => ItemStatus::InProgress,
        \\        }
        \\    }
        \\}
        \\
        \\
    );

    // Write extern functions
    try writer.writeAll(
        \\// =============================================================================
        \\// C API Functions
        \\// =============================================================================
        \\
        \\extern "C" {
        \\
    );

    // Sort functions by name for consistent output
    std.mem.sort(FunctionSignature, functions.items, {}, struct {
        fn lessThan(_: void, a: FunctionSignature, b: FunctionSignature) bool {
            return std.mem.order(u8, a.name, b.name) == .lt;
        }
    }.lessThan);

    for (functions.items) |func| {
        // Write function signature
        try writer.print("    // {s}\n", .{func.source_file});
        try writer.print("    pub fn {s}(", .{func.name});

        for (func.params, 0..) |param, idx| {
            if (idx > 0) try writer.writeAll(", ");

            // Handle reserved names
            var param_name = param.name;
            if (std.mem.eql(u8, param_name, "type")) {
                param_name = "type_";
            } else if (std.mem.eql(u8, param_name, "ref")) {
                param_name = "ref_";
            } else if (std.mem.eql(u8, param_name, "self")) {
                param_name = "self_";
            }

            const rust_type = zigToRustType(param.zig_type, &structs, &enums);
            try writer.print("{s}: {s}", .{ param_name, rust_type });
        }

        try writer.writeAll(")");

        // Return type
        const ret_type = zigToRustType(func.return_type, &structs, &enums);
        if (!std.mem.eql(u8, ret_type, "()")) {
            try writer.print(" -> {s}", .{ret_type});
        }

        try writer.writeAll(";\n");
    }

    try writer.writeAll("}\n");

    try writer.writeAll(
        \\
        \\// =============================================================================
        \\// Safe Helpers
        \\// =============================================================================
        \\
        \\
        \\/// Safe wrapper for xray capture creation.
        \\pub fn xray_capture_create(points_mask: u32, mode: u8, sample_count: u32) -> *mut c_void {
        \\    unsafe { talu_xray_capture_create(points_mask, mode, sample_count) }
        \\}
        \\
        \\/// Safe wrapper for xray capture creation (all points).
        \\pub fn xray_capture_create_all(mode: u8, sample_count: u32) -> *mut c_void {
        \\    unsafe { talu_xray_capture_create_all(mode, sample_count) }
        \\}
        \\
        \\/// Safe wrapper to enable xray capture.
        \\pub fn xray_capture_enable(handle: *mut c_void) {
        \\    unsafe { talu_xray_capture_enable(handle) }
        \\}
        \\
        \\/// Safe wrapper to disable xray capture.
        \\pub fn xray_capture_disable() {
        \\    unsafe { talu_xray_capture_disable() }
        \\}
        \\
        \\/// Safe wrapper to destroy xray capture.
        \\pub fn xray_capture_destroy(handle: *mut c_void) {
        \\    unsafe { talu_xray_capture_destroy(handle) }
        \\}
        \\
        \\/// Safe wrapper to get xray capture count.
        \\pub fn xray_capture_count(handle: *mut c_void) -> usize {
        \\    unsafe { talu_xray_capture_count(handle) }
        \\}
        \\
        \\/// Safe wrapper to get a captured tensor info entry.
        \\pub fn xray_get(handle: *mut c_void, index: usize, info: &mut CapturedTensorInfo) -> bool {
        \\    unsafe { talu_xray_get(handle, index, info as *mut CapturedTensorInfo) }
        \\}
        \\
        \\/// Safe wrapper to get xray point name.
        \\pub fn xray_point_name(point: u8) -> &'static str {
        \\    let ptr = unsafe { talu_xray_point_name(point) };
        \\    if ptr.is_null() {
        \\        "unknown"
        \\    } else {
        \\        unsafe { std::ffi::CStr::from_ptr(ptr) }.to_str().unwrap_or("unknown")
        \\    }
        \\}
        \\
    );

    // Write output to file
    const output_file = try std.fs.cwd().createFile(output_path, .{});
    defer output_file.close();
    try output_file.writeAll(output_buffer.items);

    std.debug.print("Generated {s} with {d} functions, {d} structs, {d} enums\n", .{
        output_path,
        functions.items.len,
        sorted_structs.items.len,
        enums.count(),
    });
}

// =============================================================================
// Parsing Functions
// =============================================================================

/// Resolve re-exports: when a capi file has `pub const X = imported.X;`,
/// find and parse the type from the imported source file.
fn resolveReExports(
    allocator: std.mem.Allocator,
    source: []const u8,
    source_file: []const u8,
    project_root: []const u8,
    structs: *std.StringHashMap(StructInfo),
    enums: *std.StringHashMap(EnumInfo),
) !void {
    // Build import map: alias name → relative import path
    var import_map = std.StringHashMap([]const u8).init(allocator);
    defer import_map.deinit();

    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");

        // Match: const alias = @import("path");
        if (!std.mem.startsWith(u8, trimmed, "const ")) continue;
        const import_marker = " = @import(\"";
        const import_pos = std.mem.indexOf(u8, trimmed, import_marker) orelse continue;
        const alias = trimmed["const ".len..import_pos];
        const path_start = import_pos + import_marker.len;
        const path_end = std.mem.indexOf(u8, trimmed[path_start..], "\")") orelse continue;
        const rel_path = trimmed[path_start .. path_start + path_end];

        try import_map.put(alias, rel_path);
    }

    if (import_map.count() == 0) return;

    // Look for pub const Name = alias.Name; patterns
    lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (!std.mem.startsWith(u8, trimmed, "pub const ")) continue;

        const name_start = "pub const ".len;
        const eq_pos = std.mem.indexOf(u8, trimmed[name_start..], " = ") orelse continue;
        const type_name = trimmed[name_start .. name_start + eq_pos];

        // Already discovered this type? Skip.
        if (structs.contains(type_name)) continue;
        if (enums.contains(type_name)) continue;

        const rhs_start = name_start + eq_pos + " = ".len;
        const rhs = trimmed[rhs_start..];
        if (!std.mem.endsWith(u8, rhs, ";")) continue;
        const rhs_body = rhs[0 .. rhs.len - 1];

        // Must be alias.TypeName pattern
        const dot_pos = std.mem.indexOf(u8, rhs_body, ".") orelse continue;
        const alias = rhs_body[0..dot_pos];

        const rel_import_path = import_map.get(alias) orelse continue;

        // Resolve path relative to the current source file's directory
        const dir_end = std.mem.lastIndexOfScalar(u8, source_file, '/') orelse continue;
        const source_dir = source_file[0..dir_end];

        const abs_path = try std.fmt.allocPrint(allocator, "{s}/{s}/{s}", .{
            project_root, source_dir, rel_import_path,
        });
        defer allocator.free(abs_path);

        const file = std.fs.cwd().openFile(abs_path, .{}) catch continue;
        defer file.close();

        const imported_source = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch continue;
        defer allocator.free(imported_source);

        // Parse types from the imported file, attributed to the capi source
        try parseExternStructs(allocator, imported_source, source_file, structs);
        try parseEnums(allocator, imported_source, source_file, enums);
    }
}

fn parseExternStructs(
    allocator: std.mem.Allocator,
    source: []const u8,
    source_file: []const u8,
    structs: *std.StringHashMap(StructInfo),
) !void {
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");

        // Look for: pub const StructName = extern struct { OR extern union {
        const is_struct = std.mem.startsWith(u8, trimmed, "pub const ") and
            std.mem.indexOf(u8, trimmed, " = extern struct {") != null;
        const is_union = std.mem.startsWith(u8, trimmed, "pub const ") and
            std.mem.indexOf(u8, trimmed, " = extern union {") != null;

        if (is_struct or is_union) {
            const name_start = "pub const ".len;
            const name_end = if (is_struct)
                std.mem.indexOf(u8, trimmed, " = extern struct {") orelse continue
            else
                std.mem.indexOf(u8, trimmed, " = extern union {") orelse continue;
            const struct_name = trimmed[name_start..name_end];

            // Parse fields
            var fields = std.ArrayListUnmanaged(StructInfo.FieldInfo){};

            while (lines.next()) |field_line| {
                const field_trimmed = std.mem.trim(u8, field_line, " \t\r");

                // End of struct
                if (std.mem.startsWith(u8, field_trimmed, "};")) break;
                if (std.mem.startsWith(u8, field_trimmed, "pub const ")) break;
                if (std.mem.startsWith(u8, field_trimmed, "pub fn ")) break;

                // Skip comments and empty lines
                if (field_trimmed.len == 0) continue;
                if (std.mem.startsWith(u8, field_trimmed, "//")) continue;

                // Parse field: name: type,
                if (std.mem.indexOf(u8, field_trimmed, ": ")) |colon_pos| {
                    const field_name = field_trimmed[0..colon_pos];

                    // Skip fields starting with _ (padding/reserved)
                    // Actually, include them for ABI compatibility
                    // But skip comptime fields
                    if (std.mem.indexOf(u8, field_name, "comptime") != null) continue;

                    // Check for default value first (" ="), then comma
                    const type_end = std.mem.indexOf(u8, field_trimmed[colon_pos + 2 ..], " =") orelse
                        std.mem.indexOf(u8, field_trimmed[colon_pos + 2 ..], ",") orelse
                        (field_trimmed.len - colon_pos - 2);

                    var field_type = field_trimmed[colon_pos + 2 .. colon_pos + 2 + type_end];
                    field_type = std.mem.trim(u8, field_type, " \t,");

                    // Parse array size
                    const array_size = parseArraySize(field_type);

                    const field_name_copy = try allocator.dupe(u8, field_name);
                    const field_type_copy = try allocator.dupe(u8, field_type);

                    try fields.append(allocator, .{
                        .name = field_name_copy,
                        .zig_type = field_type_copy,
                        .array_size = array_size,
                    });
                }
            }

            const name_copy = try allocator.dupe(u8, struct_name);
            const source_copy = try allocator.dupe(u8, source_file);
            const fields_slice = try allocator.dupe(StructInfo.FieldInfo, fields.items);

            try structs.put(name_copy, .{
                .name = name_copy,
                .fields = fields_slice,
                .source_file = source_copy,
                .is_union = is_union,
            });
        }
    }
}

fn parseEnums(
    allocator: std.mem.Allocator,
    source: []const u8,
    source_file: []const u8,
    enums_map: *std.StringHashMap(EnumInfo),
) !void {
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");

        // Look for: pub const EnumName = enum(type) {
        if (std.mem.startsWith(u8, trimmed, "pub const ") and
            std.mem.indexOf(u8, trimmed, " = enum(") != null)
        {
            const name_start = "pub const ".len;
            const name_end = std.mem.indexOf(u8, trimmed, " = enum(") orelse continue;
            const enum_name = trimmed[name_start..name_end];

            // Parse repr type
            const repr_start = std.mem.indexOf(u8, trimmed, "enum(").? + 5;
            const repr_end = std.mem.indexOf(u8, trimmed[repr_start..], ")") orelse continue;
            const repr_type = trimmed[repr_start .. repr_start + repr_end];

            // Parse variants
            var variants = std.ArrayListUnmanaged(EnumInfo.VariantInfo){};

            // Track brace depth - any code inside { } is not a variant
            // Start at 0, the enum body itself doesn't count (we're already past the opening brace)
            var brace_depth: usize = 0;

            while (lines.next()) |variant_line| {
                const variant_trimmed = std.mem.trim(u8, variant_line, " \t\r");

                // End of enum (only when brace_depth is 0)
                if (std.mem.startsWith(u8, variant_trimmed, "};") and brace_depth == 0) break;

                // Count braces on this line
                for (variant_trimmed) |c| {
                    if (c == '{') brace_depth += 1;
                    if (c == '}') {
                        if (brace_depth > 0) brace_depth -= 1;
                    }
                }

                // If we're inside a code block (brace_depth > 0), skip this line
                if (brace_depth > 0) continue;

                // Skip comments and empty lines
                if (variant_trimmed.len == 0) continue;
                if (std.mem.startsWith(u8, variant_trimmed, "//")) continue;
                if (std.mem.startsWith(u8, variant_trimmed, "///")) continue;

                // Skip function declarations (they'll have their body handled by brace tracking)
                if (std.mem.startsWith(u8, variant_trimmed, "pub fn ")) continue;
                if (std.mem.startsWith(u8, variant_trimmed, "fn ")) continue;

                // Skip underscore placeholder
                if (std.mem.startsWith(u8, variant_trimmed, "_")) continue;

                // Skip const declarations and comptime code
                if (std.mem.startsWith(u8, variant_trimmed, "pub const ")) continue;
                if (std.mem.startsWith(u8, variant_trimmed, "const ")) continue;
                if (std.mem.startsWith(u8, variant_trimmed, "comptime ")) continue;
                if (std.mem.startsWith(u8, variant_trimmed, "inline ")) continue;
                if (std.mem.startsWith(u8, variant_trimmed, "return ")) continue;
                if (std.mem.startsWith(u8, variant_trimmed, "var ")) continue;

                // Skip lines that contain array indexing (e.g., buf[pos] = c)
                if (std.mem.indexOf(u8, variant_trimmed, "[") != null and
                    std.mem.indexOf(u8, variant_trimmed, "]") != null)
                {
                    continue;
                }

                // Variant names start with letter (upper or lower case) or dot
                // Skip lines that start with other characters (e.g., `}`, `{`, punctuation)
                if (variant_trimmed.len > 0 and !std.ascii.isAlphabetic(variant_trimmed[0]) and variant_trimmed[0] != '.') continue;

                // Parse variant: name = value, or just name,
                var variant_name: []const u8 = undefined;
                var variant_value: ?i64 = null;

                if (std.mem.indexOf(u8, variant_trimmed, " = ")) |eq_pos| {
                    variant_name = variant_trimmed[0..eq_pos];
                    const value_start = eq_pos + 3;
                    const value_end = std.mem.indexOf(u8, variant_trimmed[value_start..], ",") orelse
                        (variant_trimmed.len - value_start);
                    const value_str = variant_trimmed[value_start .. value_start + value_end];
                    variant_value = std.fmt.parseInt(i64, value_str, 10) catch null;
                } else if (std.mem.indexOf(u8, variant_trimmed, ",")) |comma_pos| {
                    variant_name = variant_trimmed[0..comma_pos];
                } else {
                    continue;
                }

                // Convert to PascalCase for Rust
                const pascal_name = try toPascalCase(allocator, variant_name);

                try variants.append(allocator, .{
                    .name = pascal_name,
                    .value = variant_value,
                });
            }

            const name_copy = try allocator.dupe(u8, enum_name);
            const repr_copy = try allocator.dupe(u8, repr_type);
            const source_copy = try allocator.dupe(u8, source_file);
            const variants_slice = try allocator.dupe(EnumInfo.VariantInfo, variants.items);

            try enums_map.put(name_copy, .{
                .name = name_copy,
                .repr_type = repr_copy,
                .variants = variants_slice,
                .source_file = source_copy,
            });
        }
    }
}

fn parseExportedFunctions(
    allocator: std.mem.Allocator,
    source: []const u8,
    source_file: []const u8,
    functions: *std.ArrayListUnmanaged(FunctionSignature),
) !void {
    var line_num: usize = 0;
    var lines = std.mem.splitScalar(u8, source, '\n');

    while (lines.next()) |line| {
        line_num += 1;
        const trimmed = std.mem.trim(u8, line, " \t\r");

        // Look for: pub export fn talu_function_name(
        // This may be a multi-line function signature
        if (!std.mem.startsWith(u8, trimmed, "pub export fn talu_") and
            !std.mem.startsWith(u8, trimmed, "pub fn talu_"))
        {
            continue;
        }

        // Find function name
        const fn_start = std.mem.indexOf(u8, trimmed, "fn ").? + 3;
        const fn_end = std.mem.indexOf(u8, trimmed[fn_start..], "(").? + fn_start;
        const fn_name = trimmed[fn_start..fn_end];

        // Accumulate the full function signature (may span multiple lines)
        var full_sig = std.ArrayListUnmanaged(u8){};
        defer full_sig.deinit(allocator);
        try full_sig.appendSlice(allocator, trimmed);

        // Keep reading lines until we find ") callconv(.c)"
        while (std.mem.indexOf(u8, full_sig.items, ") callconv(.c)") == null) {
            const next_line = lines.next() orelse break;
            line_num += 1;
            const next_trimmed = std.mem.trim(u8, next_line, " \t\r");
            try full_sig.append(allocator, ' ');
            try full_sig.appendSlice(allocator, next_trimmed);
        }

        const sig = full_sig.items;

        // Must have callconv(.c) to be a C API function
        if (std.mem.indexOf(u8, sig, "callconv(.c)") == null) continue;

        // Parse parameters from the full signature
        const params_start = std.mem.indexOf(u8, sig, "(").? + 1;
        const params_end = std.mem.lastIndexOf(u8, sig, ") callconv(.c)") orelse continue;
        const params_str = sig[params_start..params_end];

        var params = std.ArrayListUnmanaged(FunctionSignature.ParamInfo){};

        if (params_str.len > 0) {
            var param_it = std.mem.splitScalar(u8, params_str, ',');
            while (param_it.next()) |param| {
                const param_trimmed = std.mem.trim(u8, param, " \t\r\n");
                if (param_trimmed.len == 0) continue;

                // Parse param: name: type
                if (std.mem.indexOf(u8, param_trimmed, ": ")) |colon| {
                    const param_name = std.mem.trim(u8, param_trimmed[0..colon], " ");
                    const param_type = std.mem.trim(u8, param_trimmed[colon + 2 ..], " ");

                    const name_copy = try allocator.dupe(u8, param_name);
                    const type_copy = try allocator.dupe(u8, param_type);

                    try params.append(allocator, .{
                        .name = name_copy,
                        .zig_type = type_copy,
                    });
                }
            }
        }

        // Parse return type
        var return_type: []const u8 = "void";
        if (std.mem.indexOf(u8, sig, ") callconv(.c) ")) |ret_start| {
            const type_start = ret_start + ") callconv(.c) ".len;
            const type_end = std.mem.indexOf(u8, sig[type_start..], " {") orelse
                std.mem.indexOf(u8, sig[type_start..], "{") orelse
                (sig.len - type_start);
            return_type = std.mem.trim(u8, sig[type_start .. type_start + type_end], " ");
        }

        const name_copy = try allocator.dupe(u8, fn_name);
        const source_copy = try allocator.dupe(u8, source_file);
        const ret_copy = try allocator.dupe(u8, return_type);
        const params_slice = try allocator.dupe(FunctionSignature.ParamInfo, params.items);

        try functions.append(allocator, .{
            .name = name_copy,
            .params = params_slice,
            .return_type = ret_copy,
            .source_file = source_copy,
            .line = line_num,
        });
    }
}

/// Convert snake_case to PascalCase
fn toPascalCase(allocator: std.mem.Allocator, name: []const u8) ![]const u8 {
    var result = std.ArrayListUnmanaged(u8){};
    var capitalize_next = true;

    for (name) |c| {
        if (c == '_') {
            capitalize_next = true;
        } else if (capitalize_next) {
            try result.append(allocator, std.ascii.toUpper(c));
            capitalize_next = false;
        } else {
            try result.append(allocator, c);
        }
    }

    return result.items;
}
