//! Python-specific call graph extraction.
//!
//! Extracts callable definitions (functions, methods, classes), call sites,
//! and import aliases from Python source code ASTs.
//!
//! Thread safety: All functions are pure. Safe to call concurrently
//! (each call creates its own parser/tree state).

const std = @import("std");
const Node = @import("../node.zig").Node;
const Language = @import("../language.zig").Language;
const types = @import("types.zig");
const paths = @import("paths.zig");
const CallableDefinitionInfo = types.CallableDefinitionInfo;
const CallSiteDetail = types.CallSiteDetail;
const AliasInfo = types.AliasInfo;
const ParamDetail = types.ParamDetail;
const Argument = types.Argument;
const Span = types.Span;
const Visibility = types.Visibility;
const ArgumentSource = types.ArgumentSource;
const ExtractionResult = types.ExtractionResult;
const ExtractError = types.ExtractError;
const ImportEntry = types.ImportEntry;

/// Extract callable definitions and import aliases from a Python AST.
pub fn extractCallables(
    allocator: std.mem.Allocator,
    root: Node,
    source: []const u8,
    file_path: []const u8,
    project_root: []const u8,
) ExtractError!ExtractionResult {
    var callables = std.ArrayList(CallableDefinitionInfo).empty;
    errdefer callables.deinit(allocator);
    var aliases = std.ArrayList(AliasInfo).empty;
    errdefer aliases.deinit(allocator);

    const module_path = try paths.fileToModulePath(allocator, file_path, project_root);

    // Namespace stack for tracking class nesting
    var ns_stack = std.ArrayList([]const u8).empty;
    defer ns_stack.deinit(allocator);
    try ns_stack.append(allocator, module_path);

    try walkCallables(allocator, root, source, file_path, ns_stack.items, &callables, &aliases);

    return ExtractionResult{
        .callables = try callables.toOwnedSlice(allocator),
        .aliases = try aliases.toOwnedSlice(allocator),
    };
}

fn walkCallables(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    file_path: []const u8,
    namespace: []const []const u8,
    callables: *std.ArrayList(CallableDefinitionInfo),
    aliases: *std.ArrayList(AliasInfo),
) ExtractError!void {
    const kind = node.kind();

    if (std.mem.eql(u8, kind, "function_definition")) {
        try extractFunction(allocator, node, source, file_path, namespace, callables);
        return; // Don't recurse into function body for nested function defs
    }

    if (std.mem.eql(u8, kind, "class_definition")) {
        try extractClass(allocator, node, source, file_path, namespace, callables, aliases);
        return;
    }

    if (std.mem.eql(u8, kind, "import_statement") or
        std.mem.eql(u8, kind, "import_from_statement"))
    {
        try extractImport(allocator, node, source, namespace, aliases);
    }

    // Recurse into children
    var i: u32 = 0;
    const count = node.childCount();
    while (i < count) : (i += 1) {
        if (node.child(i)) |ch| {
            try walkCallables(allocator, ch, source, file_path, namespace, callables, aliases);
        }
    }
}

fn extractFunction(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    file_path: []const u8,
    namespace: []const []const u8,
    callables: *std.ArrayList(CallableDefinitionInfo),
) ExtractError!void {
    const name_node = node.childByFieldName("name") orelse return;
    const func_name = name_node.text(source);
    if (func_name.len == 0) return;

    // Build FQN from namespace + function name
    var fqn_parts = std.ArrayList([]const u8).empty;
    defer fqn_parts.deinit(allocator);
    for (namespace) |ns| {
        try fqn_parts.append(allocator, ns);
    }
    try fqn_parts.append(allocator, func_name);
    const fqn = try paths.buildFqn(allocator, fqn_parts.items);

    // Extract parameters
    const params = try extractParams(allocator, node, source);

    // Extract return type
    const return_type = if (node.childByFieldName("return_type")) |rt|
        rt.text(source)
    else
        null;

    // Determine visibility from naming convention
    const visibility: Visibility = if (func_name.len >= 2 and func_name[0] == '_' and func_name[1] == '_')
        .private // dunder
    else if (func_name.len >= 1 and func_name[0] == '_')
        .private
    else
        .public;

    // Body span
    const body_node = node.childByFieldName("body");
    const body_span = if (body_node) |b|
        Span{ .start = b.startByte(), .end = b.endByte() }
    else
        Span{ .start = node.startByte(), .end = node.endByte() };

    try callables.append(allocator, .{
        .fqn = fqn,
        .name_span = .{ .start = name_node.startByte(), .end = name_node.endByte() },
        .body_span = body_span,
        .signature_span = .{ .start = node.startByte(), .end = if (body_node) |b| b.startByte() else node.endByte() },
        .language = .python,
        .file_path = file_path,
        .parameters = params,
        .return_type = return_type,
        .visibility = visibility,
    });
}

fn extractClass(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    file_path: []const u8,
    namespace: []const []const u8,
    callables: *std.ArrayList(CallableDefinitionInfo),
    aliases: *std.ArrayList(AliasInfo),
) ExtractError!void {
    const name_node = node.childByFieldName("name") orelse return;
    const class_name = name_node.text(source);
    if (class_name.len == 0) return;

    // Build new namespace with class name
    var new_ns = std.ArrayList([]const u8).empty;
    defer new_ns.deinit(allocator);
    for (namespace) |ns| {
        try new_ns.append(allocator, ns);
    }
    try new_ns.append(allocator, class_name);

    // Recurse into class body with updated namespace
    if (node.childByFieldName("body")) |body| {
        var i: u32 = 0;
        const count = body.childCount();
        while (i < count) : (i += 1) {
            if (body.child(i)) |ch| {
                try walkCallables(allocator, ch, source, file_path, new_ns.items, callables, aliases);
            }
        }
    }
}

fn extractParams(
    allocator: std.mem.Allocator,
    func_node: Node,
    source: []const u8,
) ExtractError![]const ParamDetail {
    const params_node = func_node.childByFieldName("parameters") orelse return &.{};

    var params = std.ArrayList(ParamDetail).empty;
    errdefer params.deinit(allocator);

    var i: u32 = 0;
    const count = params_node.childCount();
    while (i < count) : (i += 1) {
        const ch = params_node.child(i) orelse continue;
        const ch_kind = ch.kind();

        if (std.mem.eql(u8, ch_kind, "identifier")) {
            const param_name = ch.text(source);
            if (param_name.len > 0 and !std.mem.eql(u8, param_name, "self") and !std.mem.eql(u8, param_name, "cls")) {
                try params.append(allocator, .{ .name = param_name, .type_annotation = null });
            }
        } else if (std.mem.eql(u8, ch_kind, "typed_parameter")) {
            // typed_parameter has name and type children
            const param_name_node = ch.child(0);
            const type_node = ch.childByFieldName("type");
            if (param_name_node) |pn| {
                const param_name = pn.text(source);
                if (param_name.len > 0 and !std.mem.eql(u8, param_name, "self") and !std.mem.eql(u8, param_name, "cls")) {
                    try params.append(allocator, .{
                        .name = param_name,
                        .type_annotation = if (type_node) |t| t.text(source) else null,
                    });
                }
            }
        } else if (std.mem.eql(u8, ch_kind, "default_parameter") or
            std.mem.eql(u8, ch_kind, "typed_default_parameter"))
        {
            if (ch.childByFieldName("name")) |pn| {
                const param_name = pn.text(source);
                if (param_name.len > 0 and !std.mem.eql(u8, param_name, "self") and !std.mem.eql(u8, param_name, "cls")) {
                    const type_node = ch.childByFieldName("type");
                    try params.append(allocator, .{
                        .name = param_name,
                        .type_annotation = if (type_node) |t| t.text(source) else null,
                    });
                }
            }
        }
    }

    return params.toOwnedSlice(allocator);
}

fn extractImport(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    namespace: []const []const u8,
    aliases: *std.ArrayList(AliasInfo),
) ExtractError!void {
    const kind = node.kind();
    const module_path = if (namespace.len > 0) namespace[0] else "";

    if (std.mem.eql(u8, kind, "import_statement")) {
        // `import foo` or `import foo as bar`
        var i: u32 = 0;
        const count = node.childCount();
        while (i < count) : (i += 1) {
            const ch = node.child(i) orelse continue;
            const ch_kind = ch.kind();

            if (std.mem.eql(u8, ch_kind, "aliased_import")) {
                // `import foo as bar` -> alias_fqn for "bar", target = "::foo"
                const name_node = ch.childByFieldName("name") orelse continue;
                const alias_node = ch.childByFieldName("alias") orelse continue;
                const original = name_node.text(source);
                const alias_name = alias_node.text(source);
                if (original.len == 0 or alias_name.len == 0) continue;

                const target = try paths.dottedToCanonical(allocator, original);
                const alias_fqn = try paths.buildFqn(allocator, &.{ module_path, alias_name });
                try aliases.append(allocator, .{
                    .alias_fqn = alias_fqn,
                    .target_path_guess = target,
                    .defining_module = module_path,
                    .is_public = true,
                });
            } else if (std.mem.eql(u8, ch_kind, "dotted_name")) {
                // `import foo` -> alias_fqn = target = "::foo"
                const import_text = ch.text(source);
                if (import_text.len > 0) {
                    const target = try paths.dottedToCanonical(allocator, import_text);
                    try aliases.append(allocator, .{
                        .alias_fqn = target,
                        .target_path_guess = target,
                        .defining_module = module_path,
                        .is_public = true,
                    });
                }
            }
        }
    } else if (std.mem.eql(u8, kind, "import_from_statement")) {
        // `from foo import bar` or `from foo import bar as baz`
        const module_name_node = node.childByFieldName("module_name");
        const module_name = if (module_name_node) |m| m.text(source) else "";

        var i: u32 = 0;
        const count = node.childCount();
        while (i < count) : (i += 1) {
            const ch = node.child(i) orelse continue;
            const ch_kind = ch.kind();

            if (std.mem.eql(u8, ch_kind, "aliased_import")) {
                // `from foo import bar as baz` -> local="baz", target="::foo::bar"
                const name_node = ch.childByFieldName("name") orelse continue;
                const alias_node = ch.childByFieldName("alias") orelse continue;
                const original = name_node.text(source);
                const alias_name = alias_node.text(source);
                if (original.len == 0 or alias_name.len == 0) continue;

                const target = try buildFromTarget(allocator, module_name, original);
                const alias_fqn = try paths.buildFqn(allocator, &.{ module_path, alias_name });
                try aliases.append(allocator, .{
                    .alias_fqn = alias_fqn,
                    .target_path_guess = target,
                    .defining_module = module_path,
                    .is_public = true,
                });
            } else if (std.mem.eql(u8, ch_kind, "dotted_name")) {
                // Skip the module_name node itself
                if (module_name_node) |mn| {
                    if (ch.startByte() == mn.startByte()) continue;
                }

                const import_name = ch.text(source);
                if (import_name.len > 0) {
                    const target = try buildFromTarget(allocator, module_name, import_name);
                    const alias_fqn = try paths.buildFqn(allocator, &.{ module_path, import_name });
                    try aliases.append(allocator, .{
                        .alias_fqn = alias_fqn,
                        .target_path_guess = target,
                        .defining_module = module_path,
                        .is_public = true,
                    });
                }
            }
        }
    }
}

/// Build canonical target path for `from module import name`.
fn buildFromTarget(
    allocator: std.mem.Allocator,
    module_name: []const u8,
    import_name: []const u8,
) ExtractError![]const u8 {
    if (module_name.len > 0) {
        const dotted = try allocator.alloc(u8, module_name.len + 1 + import_name.len);
        @memcpy(dotted[0..module_name.len], module_name);
        dotted[module_name.len] = '.';
        @memcpy(dotted[module_name.len + 1 ..], import_name);
        const target = try paths.dottedToCanonical(allocator, dotted);
        allocator.free(dotted);
        return target;
    }
    return paths.dottedToCanonical(allocator, import_name);
}

/// Build import context mapping local names to canonical paths.
///
/// Walks the AST root for import/import_from statements and produces
/// ImportEntry values for call resolution. Relative imports (leading dots)
/// are resolved using the file's position within the project.
pub fn buildImportContext(
    allocator: std.mem.Allocator,
    root: Node,
    source: []const u8,
    file_path: []const u8,
    project_root: []const u8,
) ExtractError![]const ImportEntry {
    var entries = std.ArrayList(ImportEntry).empty;
    errdefer entries.deinit(allocator);

    try walkImports(allocator, root, source, file_path, project_root, &entries);

    return entries.toOwnedSlice(allocator);
}

fn walkImports(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    file_path: []const u8,
    project_root: []const u8,
    entries: *std.ArrayList(ImportEntry),
) ExtractError!void {
    const kind = node.kind();

    if (std.mem.eql(u8, kind, "import_statement")) {
        try collectImportEntries(allocator, node, source, entries);
    } else if (std.mem.eql(u8, kind, "import_from_statement")) {
        try collectFromImportEntries(allocator, node, source, file_path, project_root, entries);
    }

    // Recurse (but not into function/class bodies)
    if (!std.mem.eql(u8, kind, "function_definition") and
        !std.mem.eql(u8, kind, "class_definition"))
    {
        var i: u32 = 0;
        const count = node.childCount();
        while (i < count) : (i += 1) {
            if (node.child(i)) |ch| {
                try walkImports(allocator, ch, source, file_path, project_root, entries);
            }
        }
    }
}

fn collectImportEntries(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    entries: *std.ArrayList(ImportEntry),
) ExtractError!void {
    var i: u32 = 0;
    const count = node.childCount();
    while (i < count) : (i += 1) {
        const ch = node.child(i) orelse continue;
        const ch_kind = ch.kind();

        if (std.mem.eql(u8, ch_kind, "aliased_import")) {
            // `import foo as bar` -> local="bar", target="::foo"
            const name_node = ch.childByFieldName("name") orelse continue;
            const alias_node = ch.childByFieldName("alias") orelse continue;
            const original = name_node.text(source);
            const alias_name = alias_node.text(source);
            if (original.len == 0 or alias_name.len == 0) continue;

            try entries.append(allocator, .{
                .local_name = try allocator.dupe(u8, alias_name),
                .canonical_path = try paths.dottedToCanonical(allocator, original),
            });
        } else if (std.mem.eql(u8, ch_kind, "dotted_name")) {
            // `import foo.bar` -> local="bar" (last segment), target="::foo::bar"
            const import_text = ch.text(source);
            if (import_text.len == 0) continue;

            // Local name is the last segment after the last dot
            const local = if (std.mem.lastIndexOfScalar(u8, import_text, '.')) |dot|
                import_text[dot + 1 ..]
            else
                import_text;

            try entries.append(allocator, .{
                .local_name = try allocator.dupe(u8, local),
                .canonical_path = try paths.dottedToCanonical(allocator, import_text),
            });
        }
    }
}

fn collectFromImportEntries(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    file_path: []const u8,
    project_root: []const u8,
    entries: *std.ArrayList(ImportEntry),
) ExtractError!void {
    const module_name_node = node.childByFieldName("module_name");
    const module_name = if (module_name_node) |m| m.text(source) else "";
    const is_relative = module_name.len > 0 and module_name[0] == '.';

    var i: u32 = 0;
    const count = node.childCount();
    while (i < count) : (i += 1) {
        const ch = node.child(i) orelse continue;
        const ch_kind = ch.kind();

        if (std.mem.eql(u8, ch_kind, "aliased_import")) {
            // `from foo import bar as baz` -> local="baz", target="::foo::bar"
            const name_node = ch.childByFieldName("name") orelse continue;
            const alias_node = ch.childByFieldName("alias") orelse continue;
            const original = name_node.text(source);
            const alias_name = alias_node.text(source);
            if (original.len == 0 or alias_name.len == 0) continue;

            const target = if (is_relative)
                try resolveRelativeImport(allocator, module_name, original, file_path, project_root)
            else
                try buildFromTarget(allocator, module_name, original);

            try entries.append(allocator, .{
                .local_name = try allocator.dupe(u8, alias_name),
                .canonical_path = target,
            });
        } else if (std.mem.eql(u8, ch_kind, "dotted_name")) {
            // Skip the module_name node itself
            if (module_name_node) |mn| {
                if (ch.startByte() == mn.startByte()) continue;
            }

            const import_name = ch.text(source);
            if (import_name.len > 0) {
                const target = if (is_relative)
                    try resolveRelativeImport(allocator, module_name, import_name, file_path, project_root)
                else
                    try buildFromTarget(allocator, module_name, import_name);

                try entries.append(allocator, .{
                    .local_name = try allocator.dupe(u8, import_name),
                    .canonical_path = target,
                });
            }
        }
    }
}

/// Resolve a Python relative import to a canonical path.
///
/// `from ..utils import foo` in file `pkg/sub/module.py`:
/// - module_name = "..utils" (2 dots + "utils")
/// - item_name = "foo"
/// - current module = "::pkg::sub::module"
/// - go up 2 -> "::pkg", append "utils" -> "::pkg::utils::foo"
fn resolveRelativeImport(
    allocator: std.mem.Allocator,
    module_name: []const u8,
    item_name: []const u8,
    file_path: []const u8,
    project_root: []const u8,
) ExtractError![]const u8 {
    // Count leading dots
    var dot_count: usize = 0;
    while (dot_count < module_name.len and module_name[dot_count] == '.') {
        dot_count += 1;
    }
    const remaining = module_name[dot_count..];

    // Get current module path from file location
    const current_module = try paths.fileToModulePath(allocator, file_path, project_root);
    defer allocator.free(current_module);

    // Parse segments from current_module ("::pkg::sub" -> ["pkg", "sub"])
    var seg_list = std.ArrayList([]const u8).empty;
    defer seg_list.deinit(allocator);
    {
        var rest = current_module;
        if (std.mem.startsWith(u8, rest, "::")) rest = rest[2..];
        var it = std.mem.splitSequence(u8, rest, "::");
        while (it.next()) |s| {
            if (s.len > 0) try seg_list.append(allocator, s);
        }
    }

    // Go up dot_count levels
    const keep = if (dot_count <= seg_list.items.len)
        seg_list.items.len - dot_count
    else
        0;

    // Build target parts: kept parent segments + remaining dotted path + item
    var result_parts = std.ArrayList([]const u8).empty;
    defer result_parts.deinit(allocator);
    for (seg_list.items[0..keep]) |s| try result_parts.append(allocator, s);
    if (remaining.len > 0) {
        var it = std.mem.splitScalar(u8, remaining, '.');
        while (it.next()) |s| {
            if (s.len > 0) try result_parts.append(allocator, s);
        }
    }
    try result_parts.append(allocator, item_name);

    // Build "::seg1::seg2::..." canonical path
    if (result_parts.items.len == 0) {
        return allocator.dupe(u8, "::");
    }

    var len: usize = 2; // leading "::"
    for (result_parts.items, 0..) |p, idx| {
        if (idx > 0) len += 2; // "::" separator
        len += p.len;
    }

    var result = try allocator.alloc(u8, len);
    result[0] = ':';
    result[1] = ':';
    var pos: usize = 2;
    for (result_parts.items, 0..) |p, idx| {
        if (idx > 0) {
            result[pos] = ':';
            result[pos + 1] = ':';
            pos += 2;
        }
        @memcpy(result[pos .. pos + p.len], p);
        pos += p.len;
    }
    return result;
}

/// Check if a name is a Python builtin function/type.
fn isPythonBuiltin(name: []const u8) bool {
    const map = std.StaticStringMap(void).initComptime(.{
        .{ "abs", {} },           .{ "all", {} },        .{ "any", {} },        .{ "ascii", {} },
        .{ "bin", {} },           .{ "bool", {} },       .{ "callable", {} },   .{ "chr", {} },
        .{ "classmethod", {} },   .{ "compile", {} },    .{ "complex", {} },    .{ "delattr", {} },
        .{ "dict", {} },          .{ "dir", {} },        .{ "divmod", {} },     .{ "enumerate", {} },
        .{ "eval", {} },          .{ "exec", {} },       .{ "filter", {} },     .{ "float", {} },
        .{ "format", {} },        .{ "frozenset", {} },  .{ "getattr", {} },    .{ "globals", {} },
        .{ "hasattr", {} },       .{ "hash", {} },       .{ "help", {} },       .{ "hex", {} },
        .{ "id", {} },            .{ "input", {} },      .{ "int", {} },        .{ "isinstance", {} },
        .{ "issubclass", {} },    .{ "iter", {} },       .{ "len", {} },        .{ "list", {} },
        .{ "locals", {} },        .{ "map", {} },        .{ "max", {} },        .{ "memoryview", {} },
        .{ "min", {} },           .{ "next", {} },       .{ "object", {} },     .{ "oct", {} },
        .{ "open", {} },          .{ "ord", {} },        .{ "pow", {} },        .{ "print", {} },
        .{ "property", {} },      .{ "range", {} },      .{ "repr", {} },       .{ "reversed", {} },
        .{ "round", {} },         .{ "set", {} },        .{ "setattr", {} },    .{ "slice", {} },
        .{ "sorted", {} },        .{ "staticmethod", {} }, .{ "str", {} },      .{ "sum", {} },
        .{ "super", {} },         .{ "tuple", {} },      .{ "type", {} },       .{ "vars", {} },
        .{ "zip", {} },           .{ "__import__", {} },
    });
    return map.has(name);
}

/// Generate candidate resolution paths for a Python call target.
///
/// Strategies:
/// 1. Always includes naive dottedToCanonical as fallback
/// 2. For dotted names (e.g., "np.array"): looks up prefix in imports
/// 3. For simple names (e.g., "join"): looks up directly in imports
/// 4. For simple names: tries current_module::name
pub fn generateResolutionPaths(
    allocator: std.mem.Allocator,
    raw_target: []const u8,
    imports: []const ImportEntry,
    definer_callable_fqn: []const u8,
) ExtractError![]const []const u8 {
    var resolved = std.ArrayList([]const u8).empty;
    errdefer resolved.deinit(allocator);

    // Always include naive canonical form
    const naive = try paths.dottedToCanonical(allocator, raw_target);
    try resolved.append(allocator, naive);

    if (std.mem.indexOfScalar(u8, raw_target, '.')) |dot_pos| {
        // Dotted name: look up prefix in imports
        const prefix = raw_target[0..dot_pos];
        const rest = raw_target[dot_pos + 1 ..];
        for (imports) |entry| {
            if (std.mem.eql(u8, entry.local_name, prefix)) {
                // Append rest to resolved prefix: "::numpy" + "::" + "array"
                const rest_canonical = try paths.dottedToCanonical(allocator, rest);
                defer allocator.free(rest_canonical);
                // rest_canonical starts with "::", so combine: prefix_path + rest_canonical
                const combined = try allocator.alloc(u8, entry.canonical_path.len + rest_canonical.len);
                @memcpy(combined[0..entry.canonical_path.len], entry.canonical_path);
                @memcpy(combined[entry.canonical_path.len..], rest_canonical);
                try resolved.append(allocator, combined);
                break;
            }
        }
    } else {
        // Simple name: direct lookup in imports
        for (imports) |entry| {
            if (std.mem.eql(u8, entry.local_name, raw_target)) {
                try resolved.append(allocator, try allocator.dupe(u8, entry.canonical_path));
                break;
            }
        }

        // Current-module-relative candidate
        if (definer_callable_fqn.len > 2) {
            const current_module = paths.extractModuleFromFqn(definer_callable_fqn);
            if (!std.mem.eql(u8, current_module, "::")) {
                const candidate = try allocator.alloc(u8, current_module.len + 2 + raw_target.len);
                @memcpy(candidate[0..current_module.len], current_module);
                candidate[current_module.len] = ':';
                candidate[current_module.len + 1] = ':';
                @memcpy(candidate[current_module.len + 2 ..], raw_target);
                try resolved.append(allocator, candidate);
            }
        }

        // Python builtins candidate
        if (isPythonBuiltin(raw_target)) {
            const prefix = "::builtins::";
            const bp = try allocator.alloc(u8, prefix.len + raw_target.len);
            @memcpy(bp[0..prefix.len], prefix);
            @memcpy(bp[prefix.len..], raw_target);
            try resolved.append(allocator, bp);
        }
    }

    return resolved.toOwnedSlice(allocator);
}

/// Extract call sites from a node subtree.
pub fn extractCallSites(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    definer_fqn: []const u8,
    imports: []const ImportEntry,
) ExtractError![]CallSiteDetail {
    var calls = std.ArrayList(CallSiteDetail).empty;
    errdefer calls.deinit(allocator);

    try walkCallSites(allocator, node, source, definer_fqn, imports, &calls);

    return calls.toOwnedSlice(allocator);
}

fn walkCallSites(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    definer_fqn: []const u8,
    imports: []const ImportEntry,
    calls: *std.ArrayList(CallSiteDetail),
) ExtractError!void {
    const kind = node.kind();

    if (std.mem.eql(u8, kind, "call")) {
        try extractCallSite(allocator, node, source, definer_fqn, imports, calls);
    }

    // Recurse (but skip nested function/class definitions)
    if (!std.mem.eql(u8, kind, "function_definition") and
        !std.mem.eql(u8, kind, "class_definition"))
    {
        var i: u32 = 0;
        const count = node.childCount();
        while (i < count) : (i += 1) {
            if (node.child(i)) |ch| {
                try walkCallSites(allocator, ch, source, definer_fqn, imports, calls);
            }
        }
    }
}

fn extractCallSite(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    definer_fqn: []const u8,
    imports: []const ImportEntry,
    calls: *std.ArrayList(CallSiteDetail),
) ExtractError!void {
    const func_node = node.childByFieldName("function") orelse return;
    const target_name = func_node.text(source);
    if (target_name.len == 0) return;

    // Extract arguments
    const args = try extractArguments(allocator, node, source);

    // Check for result assignment: `x = foo()`
    const result_var = getResultVariable(node, source);

    // Generate import-aware resolution paths
    const resolved_paths = try generateResolutionPaths(allocator, target_name, imports, definer_fqn);

    try calls.append(allocator, .{
        .raw_target_name = target_name,
        .potential_resolved_paths = resolved_paths,
        .call_expr_span = .{ .start = node.startByte(), .end = node.endByte() },
        .target_name_span = .{ .start = func_node.startByte(), .end = func_node.endByte() },
        .definer_callable_fqn = definer_fqn,
        .arguments = args,
        .result_usage_variable = result_var,
    });
}

fn extractArguments(
    allocator: std.mem.Allocator,
    call_node: Node,
    source: []const u8,
) ExtractError![]const Argument {
    const args_node = call_node.childByFieldName("arguments") orelse return &.{};

    var args = std.ArrayList(Argument).empty;
    errdefer args.deinit(allocator);

    var i: u32 = 0;
    const count = args_node.childCount();
    while (i < count) : (i += 1) {
        const ch = args_node.child(i) orelse continue;
        const ch_kind = ch.kind();

        // Skip punctuation
        if (std.mem.eql(u8, ch_kind, "(") or
            std.mem.eql(u8, ch_kind, ")") or
            std.mem.eql(u8, ch_kind, ","))
        {
            continue;
        }

        const text = ch.text(source);
        if (text.len == 0) continue;

        const arg_source: ArgumentSource = if (std.mem.eql(u8, ch_kind, "string") or
            std.mem.eql(u8, ch_kind, "integer") or
            std.mem.eql(u8, ch_kind, "float") or
            std.mem.eql(u8, ch_kind, "true") or
            std.mem.eql(u8, ch_kind, "false") or
            std.mem.eql(u8, ch_kind, "none"))
            .literal
        else if (std.mem.eql(u8, ch_kind, "identifier"))
            .variable
        else if (std.mem.eql(u8, ch_kind, "call"))
            .function_call
        else
            .expression;

        try args.append(allocator, .{ .source = arg_source, .text = text });
    }

    return args.toOwnedSlice(allocator);
}

fn getResultVariable(call_node: Node, source: []const u8) ?[]const u8 {
    // Check parent: assignment -> left side is the result variable
    const parent_node = call_node.parent() orelse return null;
    const parent_kind = parent_node.kind();

    if (std.mem.eql(u8, parent_kind, "assignment")) {
        if (parent_node.childByFieldName("left")) |left| {
            const left_kind = left.kind();
            if (std.mem.eql(u8, left_kind, "identifier")) {
                return left.text(source);
            }
        }
    }

    return null;
}

// =============================================================================
// Tests
// =============================================================================

const parser_mod = @import("../parser.zig");

test "extractCallables finds Python function" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    const source = "def hello(name: str) -> int:\n    return 42\n";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const result = try extractCallables(std.testing.allocator, tree.rootNode(), source, "test.py", "");
    defer {
        for (result.callables) |c| {
            std.testing.allocator.free(c.fqn);
            std.testing.allocator.free(c.parameters);
        }
        std.testing.allocator.free(result.callables);
        for (result.aliases) |a| {
            std.testing.allocator.free(a.alias_fqn);
        }
        std.testing.allocator.free(result.aliases);
    }

    try std.testing.expectEqual(@as(usize, 1), result.callables.len);
    const callable = result.callables[0];
    try std.testing.expect(std.mem.endsWith(u8, callable.fqn, "::hello"));
    try std.testing.expectEqual(Visibility.public, callable.visibility);
    try std.testing.expectEqual(@as(usize, 1), callable.parameters.len);
    try std.testing.expectEqualStrings("name", callable.parameters[0].name);
}

test "extractCallables finds class methods" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    const source = "class Foo:\n    def bar(self):\n        pass\n";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const result = try extractCallables(std.testing.allocator, tree.rootNode(), source, "test.py", "");
    defer {
        for (result.callables) |c| {
            std.testing.allocator.free(c.fqn);
            std.testing.allocator.free(c.parameters);
        }
        std.testing.allocator.free(result.callables);
        std.testing.allocator.free(result.aliases);
    }

    try std.testing.expectEqual(@as(usize, 1), result.callables.len);
    try std.testing.expect(std.mem.endsWith(u8, result.callables[0].fqn, "::Foo::bar"));
}

test "extractCallables detects private functions" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    const source = "def _private(): pass\ndef __dunder(): pass\ndef public(): pass\n";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const result = try extractCallables(std.testing.allocator, tree.rootNode(), source, "test.py", "");
    defer {
        for (result.callables) |c| {
            std.testing.allocator.free(c.fqn);
            std.testing.allocator.free(c.parameters);
        }
        std.testing.allocator.free(result.callables);
        std.testing.allocator.free(result.aliases);
    }

    try std.testing.expectEqual(@as(usize, 3), result.callables.len);
    try std.testing.expectEqual(Visibility.private, result.callables[0].visibility);
    try std.testing.expectEqual(Visibility.private, result.callables[1].visibility);
    try std.testing.expectEqual(Visibility.public, result.callables[2].visibility);
}

test "extractCallSites finds function calls" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    const source = "result = foo(42, x)\nbar()\n";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const calls = try extractCallSites(std.testing.allocator, tree.rootNode(), source, "::test::main", &.{});
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
    try std.testing.expectEqualStrings("foo", calls[0].raw_target_name);
    try std.testing.expectEqualStrings("bar", calls[1].raw_target_name);
    // First call has result_usage_variable
    try std.testing.expectEqualStrings("result", calls[0].result_usage_variable.?);
    try std.testing.expect(calls[1].result_usage_variable == null);
}

test "extractCallSites detects argument types" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    const source = "foo(42, x, bar())\n";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const calls = try extractCallSites(std.testing.allocator, tree.rootNode(), source, "::test", &.{});
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

    // We should have calls: the outer foo() and the inner bar()
    try std.testing.expect(calls.len >= 1);
    const foo_call = calls[0];
    try std.testing.expectEqualStrings("foo", foo_call.raw_target_name);
    try std.testing.expect(foo_call.arguments.len >= 2);
    try std.testing.expectEqual(ArgumentSource.literal, foo_call.arguments[0].source);
    try std.testing.expectEqual(ArgumentSource.variable, foo_call.arguments[1].source);
}

test "buildImportContext handles import and aliased import" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    const source = "import os\nimport numpy as np\nfrom os.path import join\nfrom foo import bar as baz\n";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const ctx = try buildImportContext(std.testing.allocator, tree.rootNode(), source, "test.py", "");
    defer {
        for (ctx) |entry| {
            std.testing.allocator.free(entry.local_name);
            std.testing.allocator.free(entry.canonical_path);
        }
        std.testing.allocator.free(ctx);
    }

    try std.testing.expectEqual(@as(usize, 4), ctx.len);
    // import os -> local="os", target="::os"
    try std.testing.expectEqualStrings("os", ctx[0].local_name);
    try std.testing.expectEqualStrings("::os", ctx[0].canonical_path);
    // import numpy as np -> local="np", target="::numpy"
    try std.testing.expectEqualStrings("np", ctx[1].local_name);
    try std.testing.expectEqualStrings("::numpy", ctx[1].canonical_path);
    // from os.path import join -> local="join", target="::os::path::join"
    try std.testing.expectEqualStrings("join", ctx[2].local_name);
    try std.testing.expectEqualStrings("::os::path::join", ctx[2].canonical_path);
    // from foo import bar as baz -> local="baz", target="::foo::bar"
    try std.testing.expectEqualStrings("baz", ctx[3].local_name);
    try std.testing.expectEqualStrings("::foo::bar", ctx[3].canonical_path);
}

test "buildImportContext resolves relative import with single dot" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    const source = "from .utils import helper\n";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const ctx = try buildImportContext(
        std.testing.allocator,
        tree.rootNode(),
        source,
        "pkg/sub/module.py",
        "",
    );
    defer {
        for (ctx) |entry| {
            std.testing.allocator.free(entry.local_name);
            std.testing.allocator.free(entry.canonical_path);
        }
        std.testing.allocator.free(ctx);
    }

    try std.testing.expectEqual(@as(usize, 1), ctx.len);
    try std.testing.expectEqualStrings("helper", ctx[0].local_name);
    // "pkg/sub/module.py" -> module "::pkg::sub::module"
    // from .utils -> go up 1 from module -> "::pkg::sub", append "utils"
    // -> "::pkg::sub::utils::helper"
    try std.testing.expectEqualStrings("::pkg::sub::utils::helper", ctx[0].canonical_path);
}

test "buildImportContext resolves relative import with double dot" {
    var p = try parser_mod.Parser.init(.python);
    defer p.deinit();
    const source = "from ..utils import foo\n";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const ctx = try buildImportContext(
        std.testing.allocator,
        tree.rootNode(),
        source,
        "pkg/sub/deep/module.py",
        "",
    );
    defer {
        for (ctx) |entry| {
            std.testing.allocator.free(entry.local_name);
            std.testing.allocator.free(entry.canonical_path);
        }
        std.testing.allocator.free(ctx);
    }

    try std.testing.expectEqual(@as(usize, 1), ctx.len);
    try std.testing.expectEqualStrings("foo", ctx[0].local_name);
    // "pkg/sub/deep/module.py" -> "::pkg::sub::deep::module"
    // ".." goes up 2 -> "::pkg::sub", append "utils" -> "::pkg::sub::utils::foo"
    try std.testing.expectEqualStrings("::pkg::sub::utils::foo", ctx[0].canonical_path);
}

test "generateResolutionPaths resolves dotted name via imports" {
    const imports = &[_]ImportEntry{
        .{ .local_name = "np", .canonical_path = "::numpy" },
    };
    const resolved = try generateResolutionPaths(std.testing.allocator, "np.array", imports, "");
    defer {
        for (resolved) |p| std.testing.allocator.free(p);
        std.testing.allocator.free(resolved);
    }

    try std.testing.expectEqual(@as(usize, 2), resolved.len);
    try std.testing.expectEqualStrings("::np::array", resolved[0]); // naive
    try std.testing.expectEqualStrings("::numpy::array", resolved[1]); // resolved
}

test "generateResolutionPaths resolves simple name via imports" {
    const imports = &[_]ImportEntry{
        .{ .local_name = "join", .canonical_path = "::os::path::join" },
    };
    const resolved = try generateResolutionPaths(std.testing.allocator, "join", imports, "");
    defer {
        for (resolved) |p| std.testing.allocator.free(p);
        std.testing.allocator.free(resolved);
    }

    try std.testing.expectEqual(@as(usize, 2), resolved.len);
    try std.testing.expectEqualStrings("::join", resolved[0]); // naive
    try std.testing.expectEqualStrings("::os::path::join", resolved[1]); // resolved
}

test "generateResolutionPaths returns only naive when no import match" {
    const resolved = try generateResolutionPaths(std.testing.allocator, "unknown_func", &.{}, "");
    defer {
        for (resolved) |p| std.testing.allocator.free(p);
        std.testing.allocator.free(resolved);
    }

    try std.testing.expectEqual(@as(usize, 1), resolved.len);
    try std.testing.expectEqualStrings("::unknown_func", resolved[0]);
}

test "generateResolutionPaths adds current-module candidate for simple name" {
    const resolved = try generateResolutionPaths(
        std.testing.allocator,
        "helper",
        &.{},
        "::mymod::submod::caller",
    );
    defer {
        for (resolved) |p| std.testing.allocator.free(p);
        std.testing.allocator.free(resolved);
    }

    try std.testing.expectEqual(@as(usize, 2), resolved.len);
    try std.testing.expectEqualStrings("::helper", resolved[0]);
    try std.testing.expectEqualStrings("::mymod::submod::helper", resolved[1]);
}

test "isPythonBuiltin recognizes common builtins" {
    try std.testing.expect(isPythonBuiltin("print"));
    try std.testing.expect(isPythonBuiltin("len"));
    try std.testing.expect(isPythonBuiltin("range"));
    try std.testing.expect(isPythonBuiltin("isinstance"));
    try std.testing.expect(!isPythonBuiltin("my_function"));
    try std.testing.expect(!isPythonBuiltin("numpy"));
}

test "generateResolutionPaths adds builtins path for Python builtins" {
    const resolved = try generateResolutionPaths(
        std.testing.allocator,
        "print",
        &.{},
        "::test::func",
    );
    defer {
        for (resolved) |p| std.testing.allocator.free(p);
        std.testing.allocator.free(resolved);
    }

    // naive + module-relative + builtins
    try std.testing.expectEqual(@as(usize, 3), resolved.len);
    try std.testing.expectEqualStrings("::print", resolved[0]);
    try std.testing.expectEqualStrings("::test::print", resolved[1]);
    try std.testing.expectEqualStrings("::builtins::print", resolved[2]);
}
