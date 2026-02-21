//! Rust-specific call graph extraction.
//!
//! Extracts callable definitions (functions, methods), call sites, and
//! import aliases from Rust source code ASTs.
//!
//! Thread safety: All functions are pure. Safe to call concurrently.

const std = @import("std");
const Node = @import("../node.zig").Node;
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

    var ns_stack = std.ArrayList([]const u8).empty;
    defer ns_stack.deinit(allocator);
    try ns_stack.append(allocator, module_path);

    try walkNode(allocator, root, source, file_path, ns_stack.items, &callables, &aliases);

    return ExtractionResult{
        .callables = try callables.toOwnedSlice(allocator),
        .aliases = try aliases.toOwnedSlice(allocator),
    };
}

fn walkNode(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    file_path: []const u8,
    namespace: []const []const u8,
    callables: *std.ArrayList(CallableDefinitionInfo),
    aliases: *std.ArrayList(AliasInfo),
) ExtractError!void {
    const kind = node.kind();

    if (std.mem.eql(u8, kind, "function_item")) {
        try extractFunction(allocator, node, source, file_path, namespace, callables);
        return;
    }

    if (std.mem.eql(u8, kind, "impl_item")) {
        try extractImpl(allocator, node, source, file_path, namespace, callables, aliases);
        return;
    }

    if (std.mem.eql(u8, kind, "mod_item")) {
        try extractMod(allocator, node, source, file_path, namespace, callables, aliases);
        return;
    }

    if (std.mem.eql(u8, kind, "use_declaration")) {
        try extractUse(allocator, node, source, namespace, aliases);
    }

    var i: u32 = 0;
    const count = node.childCount();
    while (i < count) : (i += 1) {
        if (node.child(i)) |ch| {
            try walkNode(allocator, ch, source, file_path, namespace, callables, aliases);
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

    var fqn_parts = std.ArrayList([]const u8).empty;
    defer fqn_parts.deinit(allocator);
    for (namespace) |ns| try fqn_parts.append(allocator, ns);
    try fqn_parts.append(allocator, func_name);
    const fqn = try paths.buildFqn(allocator, fqn_parts.items);

    const params = try extractParams(allocator, node, source);
    const return_type = if (node.childByFieldName("return_type")) |rt| rt.text(source) else null;
    const visibility = detectVisibility(node);

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
        .language = .rust,
        .file_path = file_path,
        .parameters = params,
        .return_type = return_type,
        .visibility = visibility,
    });
}

fn extractImpl(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    file_path: []const u8,
    namespace: []const []const u8,
    callables: *std.ArrayList(CallableDefinitionInfo),
    aliases: *std.ArrayList(AliasInfo),
) ExtractError!void {
    // impl TypeName { ... } — use type name as namespace
    const type_node = node.childByFieldName("type") orelse return;
    const type_name = type_node.text(source);
    if (type_name.len == 0) return;

    var new_ns = std.ArrayList([]const u8).empty;
    defer new_ns.deinit(allocator);
    for (namespace) |ns| try new_ns.append(allocator, ns);
    try new_ns.append(allocator, type_name);

    if (node.childByFieldName("body")) |body| {
        var i: u32 = 0;
        const count = body.childCount();
        while (i < count) : (i += 1) {
            if (body.child(i)) |ch| {
                try walkNode(allocator, ch, source, file_path, new_ns.items, callables, aliases);
            }
        }
    }
}

fn extractMod(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    file_path: []const u8,
    namespace: []const []const u8,
    callables: *std.ArrayList(CallableDefinitionInfo),
    aliases: *std.ArrayList(AliasInfo),
) ExtractError!void {
    const name_node = node.childByFieldName("name") orelse return;
    const mod_name = name_node.text(source);
    if (mod_name.len == 0) return;

    var new_ns = std.ArrayList([]const u8).empty;
    defer new_ns.deinit(allocator);
    for (namespace) |ns| try new_ns.append(allocator, ns);
    try new_ns.append(allocator, mod_name);

    if (node.childByFieldName("body")) |body| {
        var i: u32 = 0;
        const count = body.childCount();
        while (i < count) : (i += 1) {
            if (body.child(i)) |ch| {
                try walkNode(allocator, ch, source, file_path, new_ns.items, callables, aliases);
            }
        }
    }
}

fn detectVisibility(node: Node) Visibility {
    // Check for `pub` visibility modifier
    var i: u32 = 0;
    const count = node.childCount();
    while (i < count) : (i += 1) {
        if (node.child(i)) |ch| {
            if (std.mem.eql(u8, ch.kind(), "visibility_modifier")) {
                return .public;
            }
        }
    }
    return .private;
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

        if (std.mem.eql(u8, ch_kind, "parameter")) {
            const pattern_node = ch.childByFieldName("pattern");
            const type_node = ch.childByFieldName("type");
            if (pattern_node) |pn| {
                const param_name = pn.text(source);
                // Skip self/&self/&mut self
                if (param_name.len > 0 and !std.mem.eql(u8, param_name, "self") and
                    !std.mem.eql(u8, param_name, "&self") and
                    !std.mem.eql(u8, param_name, "&mut self"))
                {
                    try params.append(allocator, .{
                        .name = param_name,
                        .type_annotation = if (type_node) |t| t.text(source) else null,
                    });
                }
            }
        } else if (std.mem.eql(u8, ch_kind, "self_parameter")) {
            // Skip self parameter
            continue;
        }
    }

    return params.toOwnedSlice(allocator);
}

fn extractUse(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    namespace: []const []const u8,
    aliases: *std.ArrayList(AliasInfo),
) ExtractError!void {
    const module_path = if (namespace.len > 0) namespace[0] else "";
    const vis = detectVisibility(node);

    var i: u32 = 0;
    const count = node.childCount();
    while (i < count) : (i += 1) {
        const ch = node.child(i) orelse continue;
        const ch_kind = ch.kind();

        if (std.mem.eql(u8, ch_kind, "use") or
            std.mem.eql(u8, ch_kind, "visibility_modifier") or
            std.mem.eql(u8, ch_kind, ";"))
        {
            continue;
        }

        if (std.mem.eql(u8, ch_kind, "use_as_clause")) {
            // `use foo::bar as baz`
            const path_node = ch.childByFieldName("path") orelse continue;
            const alias_node = ch.childByFieldName("alias") orelse continue;
            const path_text = path_node.text(source);
            const alias_text = alias_node.text(source);
            if (path_text.len == 0 or alias_text.len == 0) continue;

            const target = try paths.colonSeparatedToCanonical(allocator, path_text);
            const alias_fqn = try paths.buildFqn(allocator, &.{ module_path, alias_text });
            try aliases.append(allocator, .{
                .alias_fqn = alias_fqn,
                .target_path_guess = target,
                .defining_module = module_path,
                .is_public = vis == .public,
            });
        } else if (std.mem.eql(u8, ch_kind, "use_wildcard")) {
            // `use std::io::*` — skip glob imports
            continue;
        } else {
            const use_text = ch.text(source);
            if (use_text.len > 0) {
                const target = try paths.colonSeparatedToCanonical(allocator, use_text);
                try aliases.append(allocator, .{
                    .alias_fqn = target,
                    .target_path_guess = target,
                    .defining_module = module_path,
                    .is_public = vis == .public,
                });
            }
        }
    }
}

/// Build import context from Rust use declarations.
///
/// Resolves crate::/super::/self:: prefixes in use statements using the
/// file's position within the project.
pub fn buildImportContext(
    allocator: std.mem.Allocator,
    root: Node,
    source: []const u8,
    file_path: []const u8,
    project_root: []const u8,
) ExtractError![]const ImportEntry {
    var entries = std.ArrayList(ImportEntry).empty;
    errdefer entries.deinit(allocator);

    try walkUseDecls(allocator, root, source, file_path, project_root, &entries);

    return entries.toOwnedSlice(allocator);
}

fn walkUseDecls(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    file_path: []const u8,
    project_root: []const u8,
    entries: *std.ArrayList(ImportEntry),
) ExtractError!void {
    const kind = node.kind();

    if (std.mem.eql(u8, kind, "use_declaration")) {
        try collectUseEntries(allocator, node, source, file_path, project_root, entries);
    }

    // Don't recurse into function bodies
    if (!std.mem.eql(u8, kind, "function_item")) {
        var i: u32 = 0;
        const count = node.childCount();
        while (i < count) : (i += 1) {
            if (node.child(i)) |ch| {
                try walkUseDecls(allocator, ch, source, file_path, project_root, entries);
            }
        }
    }
}

/// Resolve a use path that may have crate::/super::/self:: prefix.
fn resolveUsePath(
    allocator: std.mem.Allocator,
    use_text: []const u8,
    file_path: []const u8,
    project_root: []const u8,
) ExtractError![]const u8 {
    if (std.mem.startsWith(u8, use_text, "crate::")) {
        const rest = use_text["crate::".len..];
        return paths.colonSeparatedToCanonical(allocator, rest);
    } else if (std.mem.startsWith(u8, use_text, "super::")) {
        const current_module = try paths.fileToModulePath(allocator, file_path, project_root);
        defer allocator.free(current_module);
        var super_count: usize = 0;
        var rest = use_text;
        while (std.mem.startsWith(u8, rest, "super::")) {
            super_count += 1;
            rest = rest["super::".len..];
        }
        const parent = try resolveSuper(allocator, current_module, super_count);
        defer allocator.free(parent);
        if (rest.len > 0) {
            const rest_canonical = try paths.colonSeparatedToCanonical(allocator, rest);
            defer allocator.free(rest_canonical);
            const combined = try allocator.alloc(u8, parent.len + rest_canonical.len);
            @memcpy(combined[0..parent.len], parent);
            @memcpy(combined[parent.len..], rest_canonical);
            return combined;
        }
        return allocator.dupe(u8, parent);
    } else if (std.mem.startsWith(u8, use_text, "self::")) {
        const rest = use_text["self::".len..];
        const current_module = try paths.fileToModulePath(allocator, file_path, project_root);
        defer allocator.free(current_module);
        const rest_canonical = try paths.colonSeparatedToCanonical(allocator, rest);
        defer allocator.free(rest_canonical);
        const combined = try allocator.alloc(u8, current_module.len + rest_canonical.len);
        @memcpy(combined[0..current_module.len], current_module);
        @memcpy(combined[current_module.len..], rest_canonical);
        return combined;
    } else {
        return paths.colonSeparatedToCanonical(allocator, use_text);
    }
}

fn collectUseEntries(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    file_path: []const u8,
    project_root: []const u8,
    entries: *std.ArrayList(ImportEntry),
) ExtractError!void {
    var i: u32 = 0;
    const count = node.childCount();
    while (i < count) : (i += 1) {
        const ch = node.child(i) orelse continue;
        const ch_kind = ch.kind();

        if (std.mem.eql(u8, ch_kind, "use") or
            std.mem.eql(u8, ch_kind, "visibility_modifier") or
            std.mem.eql(u8, ch_kind, ";") or
            std.mem.eql(u8, ch_kind, "use_wildcard"))
        {
            continue;
        }

        if (std.mem.eql(u8, ch_kind, "use_as_clause")) {
            // `use std::io::Read as IoRead` -> local="IoRead", target="::std::io::Read"
            const path_node = ch.childByFieldName("path") orelse continue;
            const alias_node = ch.childByFieldName("alias") orelse continue;
            const path_text = path_node.text(source);
            const alias_text = alias_node.text(source);
            if (path_text.len == 0 or alias_text.len == 0) continue;

            try entries.append(allocator, .{
                .local_name = try allocator.dupe(u8, alias_text),
                .canonical_path = try resolveUsePath(allocator, path_text, file_path, project_root),
            });
        } else {
            // `use std::collections::HashMap` -> local="HashMap", target="::std::collections::HashMap"
            const use_text = ch.text(source);
            if (use_text.len == 0) continue;

            // Local name is the last segment after `::`
            const local = if (std.mem.lastIndexOf(u8, use_text, "::")) |pos|
                use_text[pos + 2 ..]
            else
                use_text;

            try entries.append(allocator, .{
                .local_name = try allocator.dupe(u8, local),
                .canonical_path = try resolveUsePath(allocator, use_text, file_path, project_root),
            });
        }
    }
}

/// Go up `levels` from a canonical module path.
///
/// "::pkg::sub::mod" with levels=2 -> "::pkg"
/// "::pkg" with levels=1 -> "::"
fn resolveSuper(
    allocator: std.mem.Allocator,
    module_path: []const u8,
    levels: usize,
) ExtractError![]const u8 {
    var rest = module_path;
    if (std.mem.startsWith(u8, rest, "::")) rest = rest[2..];

    var segs = std.ArrayList([]const u8).empty;
    defer segs.deinit(allocator);
    var it = std.mem.splitSequence(u8, rest, "::");
    while (it.next()) |s| {
        if (s.len > 0) try segs.append(allocator, s);
    }

    const keep = if (levels <= segs.items.len) segs.items.len - levels else 0;

    if (keep == 0) {
        return allocator.dupe(u8, "::");
    }

    var len: usize = 2; // leading "::"
    for (segs.items[0..keep], 0..) |s, idx| {
        if (idx > 0) len += 2;
        len += s.len;
    }
    var result = try allocator.alloc(u8, len);
    result[0] = ':';
    result[1] = ':';
    var pos: usize = 2;
    for (segs.items[0..keep], 0..) |s, idx| {
        if (idx > 0) {
            result[pos] = ':';
            result[pos + 1] = ':';
            pos += 2;
        }
        @memcpy(result[pos .. pos + s.len], s);
        pos += s.len;
    }
    return result;
}

/// Generate candidate resolution paths for a Rust call target.
///
/// Strategies:
/// 1. Handles crate::/super::/self:: prefixes with module-relative resolution
/// 2. Always includes canonical form as fallback for normal paths
/// 3. For paths with "::" (e.g., "HashMap::new"): looks up first segment in imports
/// 4. For simple names: looks up directly in imports, then tries current module
pub fn generateResolutionPaths(
    allocator: std.mem.Allocator,
    raw_target: []const u8,
    imports: []const ImportEntry,
    definer_callable_fqn: []const u8,
) ExtractError![]const []const u8 {
    var resolved = std.ArrayList([]const u8).empty;
    errdefer resolved.deinit(allocator);

    // Handle crate:: prefix — strip "crate::" and treat as absolute from crate root
    if (std.mem.startsWith(u8, raw_target, "crate::")) {
        const rest = raw_target["crate::".len..];
        const abs_path = try paths.colonSeparatedToCanonical(allocator, rest);
        try resolved.append(allocator, abs_path);
        return resolved.toOwnedSlice(allocator);
    }

    // Handle super:: prefix — go up N levels from current module
    if (std.mem.startsWith(u8, raw_target, "super::")) {
        const current_module = paths.extractModuleFromFqn(definer_callable_fqn);
        var super_count: usize = 0;
        var rest = raw_target;
        while (std.mem.startsWith(u8, rest, "super::")) {
            super_count += 1;
            rest = rest["super::".len..];
        }
        const parent = try resolveSuper(allocator, current_module, super_count);
        defer allocator.free(parent);
        if (rest.len > 0) {
            const rest_canonical = try paths.colonSeparatedToCanonical(allocator, rest);
            defer allocator.free(rest_canonical);
            // parent ends with valid path, rest_canonical starts with "::"
            const combined = try allocator.alloc(u8, parent.len + rest_canonical.len);
            @memcpy(combined[0..parent.len], parent);
            @memcpy(combined[parent.len..], rest_canonical);
            try resolved.append(allocator, combined);
        } else {
            try resolved.append(allocator, try allocator.dupe(u8, parent));
        }
        return resolved.toOwnedSlice(allocator);
    }

    // Handle self:: prefix — prepend current module path
    if (std.mem.startsWith(u8, raw_target, "self::")) {
        const rest = raw_target["self::".len..];
        const current_module = paths.extractModuleFromFqn(definer_callable_fqn);
        if (rest.len > 0) {
            const rest_canonical = try paths.colonSeparatedToCanonical(allocator, rest);
            defer allocator.free(rest_canonical);
            const combined = try allocator.alloc(u8, current_module.len + rest_canonical.len);
            @memcpy(combined[0..current_module.len], current_module);
            @memcpy(combined[current_module.len..], rest_canonical);
            try resolved.append(allocator, combined);
        } else {
            try resolved.append(allocator, try allocator.dupe(u8, current_module));
        }
        return resolved.toOwnedSlice(allocator);
    }

    // Always include naive canonical form
    const naive = try paths.colonSeparatedToCanonical(allocator, raw_target);
    try resolved.append(allocator, naive);

    if (std.mem.indexOf(u8, raw_target, "::")) |sep_pos| {
        // Path with :: — look up first segment in imports
        const prefix = raw_target[0..sep_pos];
        const rest = raw_target[sep_pos..]; // includes leading "::"
        for (imports) |entry| {
            if (std.mem.eql(u8, entry.local_name, prefix)) {
                // Combine resolved prefix + rest
                const combined = try allocator.alloc(u8, entry.canonical_path.len + rest.len);
                @memcpy(combined[0..entry.canonical_path.len], entry.canonical_path);
                @memcpy(combined[entry.canonical_path.len..], rest);
                try resolved.append(allocator, combined);
                break;
            }
        }
    } else {
        // Simple name: direct lookup
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

    if (std.mem.eql(u8, kind, "call_expression")) {
        try extractCallSite(allocator, node, source, definer_fqn, imports, calls);
    } else if (std.mem.eql(u8, kind, "macro_invocation")) {
        try extractMacroCall(allocator, node, source, definer_fqn, imports, calls);
    }

    if (!std.mem.eql(u8, kind, "function_item")) {
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

    const args = try extractArguments(allocator, node, source);
    const result_var = getResultVariable(node, source);
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

fn extractMacroCall(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    definer_fqn: []const u8,
    imports: []const ImportEntry,
    calls: *std.ArrayList(CallSiteDetail),
) ExtractError!void {
    const macro_node = node.child(0) orelse return;
    const macro_name = macro_node.text(source);
    if (macro_name.len == 0) return;

    const resolved_paths = try generateResolutionPaths(allocator, macro_name, imports, definer_fqn);

    try calls.append(allocator, .{
        .raw_target_name = macro_name,
        .potential_resolved_paths = resolved_paths,
        .call_expr_span = .{ .start = node.startByte(), .end = node.endByte() },
        .target_name_span = .{ .start = macro_node.startByte(), .end = macro_node.endByte() },
        .definer_callable_fqn = definer_fqn,
        .arguments = &.{},
        .result_usage_variable = null,
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

        if (std.mem.eql(u8, ch_kind, "(") or
            std.mem.eql(u8, ch_kind, ")") or
            std.mem.eql(u8, ch_kind, ","))
        {
            continue;
        }

        const text = ch.text(source);
        if (text.len == 0) continue;

        const arg_source: ArgumentSource = if (std.mem.eql(u8, ch_kind, "string_literal") or
            std.mem.eql(u8, ch_kind, "integer_literal") or
            std.mem.eql(u8, ch_kind, "float_literal") or
            std.mem.eql(u8, ch_kind, "boolean_literal") or
            std.mem.eql(u8, ch_kind, "char_literal"))
            .literal
        else if (std.mem.eql(u8, ch_kind, "identifier"))
            .variable
        else if (std.mem.eql(u8, ch_kind, "call_expression"))
            .function_call
        else
            .expression;

        try args.append(allocator, .{ .source = arg_source, .text = text });
    }

    return args.toOwnedSlice(allocator);
}

fn getResultVariable(call_node: Node, source: []const u8) ?[]const u8 {
    const parent_node = call_node.parent() orelse return null;
    const parent_kind = parent_node.kind();

    if (std.mem.eql(u8, parent_kind, "let_declaration")) {
        if (parent_node.childByFieldName("pattern")) |pat| {
            if (std.mem.eql(u8, pat.kind(), "identifier")) {
                return pat.text(source);
            }
        }
    }

    return null;
}

// =============================================================================
// Tests
// =============================================================================

const parser_mod = @import("../parser.zig");

fn freeExtractionResult(allocator: std.mem.Allocator, result: ExtractionResult) void {
    for (result.callables) |c| {
        allocator.free(c.fqn);
        allocator.free(c.parameters);
    }
    allocator.free(result.callables);
    for (result.aliases) |a| {
        allocator.free(a.alias_fqn);
        allocator.free(a.target_path_guess);
    }
    allocator.free(result.aliases);
}

test "extractCallables finds Rust function" {
    var p = try parser_mod.Parser.init(.rust);
    defer p.deinit();
    const source = "pub fn hello(name: &str) -> i32 { 42 }";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const result = try extractCallables(std.testing.allocator, tree.rootNode(), source, "test.rs", "");
    defer freeExtractionResult(std.testing.allocator, result);

    try std.testing.expectEqual(@as(usize, 1), result.callables.len);
    try std.testing.expect(std.mem.endsWith(u8, result.callables[0].fqn, "::hello"));
    try std.testing.expectEqual(Visibility.public, result.callables[0].visibility);
}

test "extractCallables finds impl methods" {
    var p = try parser_mod.Parser.init(.rust);
    defer p.deinit();
    const source = "impl Foo { fn bar(&self) { } }";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const result = try extractCallables(std.testing.allocator, tree.rootNode(), source, "test.rs", "");
    defer freeExtractionResult(std.testing.allocator, result);

    try std.testing.expectEqual(@as(usize, 1), result.callables.len);
    try std.testing.expect(std.mem.endsWith(u8, result.callables[0].fqn, "::Foo::bar"));
    try std.testing.expectEqual(Visibility.private, result.callables[0].visibility);
}

test "buildImportContext handles use declarations" {
    var p = try parser_mod.Parser.init(.rust);
    defer p.deinit();
    const source = "use std::collections::HashMap;\n";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const ctx = try buildImportContext(std.testing.allocator, tree.rootNode(), source, "src/main.rs", "");
    defer {
        for (ctx) |entry| {
            std.testing.allocator.free(entry.local_name);
            std.testing.allocator.free(entry.canonical_path);
        }
        std.testing.allocator.free(ctx);
    }

    try std.testing.expectEqual(@as(usize, 1), ctx.len);
    try std.testing.expectEqualStrings("HashMap", ctx[0].local_name);
    try std.testing.expectEqualStrings("::std::collections::HashMap", ctx[0].canonical_path);
}

test "buildImportContext resolves crate:: in use statement" {
    var p = try parser_mod.Parser.init(.rust);
    defer p.deinit();
    const source = "use crate::utils::Helper;\n";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const ctx = try buildImportContext(std.testing.allocator, tree.rootNode(), source, "src/main.rs", "");
    defer {
        for (ctx) |entry| {
            std.testing.allocator.free(entry.local_name);
            std.testing.allocator.free(entry.canonical_path);
        }
        std.testing.allocator.free(ctx);
    }

    try std.testing.expectEqual(@as(usize, 1), ctx.len);
    try std.testing.expectEqualStrings("Helper", ctx[0].local_name);
    try std.testing.expectEqualStrings("::utils::Helper", ctx[0].canonical_path);
}

test "generateResolutionPaths resolves crate:: prefix" {
    const resolved = try generateResolutionPaths(
        std.testing.allocator,
        "crate::utils::helper",
        &.{},
        "::mymod::func",
    );
    defer {
        for (resolved) |rp| std.testing.allocator.free(rp);
        std.testing.allocator.free(resolved);
    }

    try std.testing.expectEqual(@as(usize, 1), resolved.len);
    try std.testing.expectEqualStrings("::utils::helper", resolved[0]);
}

test "generateResolutionPaths resolves super:: prefix" {
    const resolved = try generateResolutionPaths(
        std.testing.allocator,
        "super::sibling::func",
        &.{},
        "::pkg::sub::deep::caller",
    );
    defer {
        for (resolved) |rp| std.testing.allocator.free(rp);
        std.testing.allocator.free(resolved);
    }

    try std.testing.expectEqual(@as(usize, 1), resolved.len);
    try std.testing.expectEqualStrings("::pkg::sub::sibling::func", resolved[0]);
}

test "generateResolutionPaths resolves self:: prefix" {
    const resolved = try generateResolutionPaths(
        std.testing.allocator,
        "self::helper",
        &.{},
        "::mymod::sub::func",
    );
    defer {
        for (resolved) |rp| std.testing.allocator.free(rp);
        std.testing.allocator.free(resolved);
    }

    try std.testing.expectEqual(@as(usize, 1), resolved.len);
    try std.testing.expectEqualStrings("::mymod::sub::helper", resolved[0]);
}

test "generateResolutionPaths resolves qualified Rust call" {
    const imports = &[_]ImportEntry{
        .{ .local_name = "HashMap", .canonical_path = "::std::collections::HashMap" },
    };
    const resolved = try generateResolutionPaths(std.testing.allocator, "HashMap::new", imports, "");
    defer {
        for (resolved) |p| std.testing.allocator.free(p);
        std.testing.allocator.free(resolved);
    }

    try std.testing.expectEqual(@as(usize, 2), resolved.len);
    try std.testing.expectEqualStrings("::HashMap::new", resolved[0]); // naive
    try std.testing.expectEqualStrings("::std::collections::HashMap::new", resolved[1]); // resolved
}

test "generateResolutionPaths resolves simple Rust name" {
    const imports = &[_]ImportEntry{
        .{ .local_name = "helper", .canonical_path = "::crate::utils::helper" },
    };
    const resolved = try generateResolutionPaths(std.testing.allocator, "helper", imports, "");
    defer {
        for (resolved) |p| std.testing.allocator.free(p);
        std.testing.allocator.free(resolved);
    }

    try std.testing.expectEqual(@as(usize, 2), resolved.len);
    try std.testing.expectEqualStrings("::helper", resolved[0]);
    try std.testing.expectEqualStrings("::crate::utils::helper", resolved[1]);
}

test "generateResolutionPaths adds current-module candidate for simple Rust name" {
    const resolved = try generateResolutionPaths(
        std.testing.allocator,
        "do_work",
        &.{},
        "::mymod::sub::caller",
    );
    defer {
        for (resolved) |p| std.testing.allocator.free(p);
        std.testing.allocator.free(resolved);
    }

    try std.testing.expectEqual(@as(usize, 2), resolved.len);
    try std.testing.expectEqualStrings("::do_work", resolved[0]);
    try std.testing.expectEqualStrings("::mymod::sub::do_work", resolved[1]);
}
