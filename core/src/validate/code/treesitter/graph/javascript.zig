//! JavaScript/TypeScript-specific call graph extraction.
//!
//! Extracts callable definitions (functions, methods, arrow functions), call
//! sites, and import aliases from JavaScript/TypeScript source code ASTs.
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
const Language = @import("../language.zig").Language;
const ExtractionResult = types.ExtractionResult;
const ExtractError = types.ExtractError;
const ImportEntry = types.ImportEntry;

pub fn extractCallables(
    allocator: std.mem.Allocator,
    root: Node,
    source: []const u8,
    file_path: []const u8,
    project_root: []const u8,
    language: Language,
) ExtractError!ExtractionResult {
    var callables = std.ArrayList(CallableDefinitionInfo).empty;
    errdefer callables.deinit(allocator);
    var aliases = std.ArrayList(AliasInfo).empty;
    errdefer aliases.deinit(allocator);

    const module_path = try paths.fileToModulePath(allocator, file_path, project_root);

    var ns_stack = std.ArrayList([]const u8).empty;
    defer ns_stack.deinit(allocator);
    try ns_stack.append(allocator, module_path);

    try walkNode(allocator, root, source, file_path, ns_stack.items, language, &callables, &aliases);

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
    language: Language,
    callables: *std.ArrayList(CallableDefinitionInfo),
    aliases: *std.ArrayList(AliasInfo),
) ExtractError!void {
    const kind = node.kind();

    if (std.mem.eql(u8, kind, "function_declaration")) {
        try extractFunction(allocator, node, source, file_path, namespace, language, callables);
        return;
    }

    if (std.mem.eql(u8, kind, "method_definition")) {
        try extractMethod(allocator, node, source, file_path, namespace, language, callables);
        return;
    }

    if (std.mem.eql(u8, kind, "class_declaration")) {
        try extractClass(allocator, node, source, file_path, namespace, language, callables, aliases);
        return;
    }

    if (std.mem.eql(u8, kind, "lexical_declaration") or std.mem.eql(u8, kind, "variable_declaration")) {
        try extractArrowFunctions(allocator, node, source, file_path, namespace, language, callables);
    }

    if (std.mem.eql(u8, kind, "import_statement")) {
        try extractImport(allocator, node, source, namespace, aliases);
    }

    var i: u32 = 0;
    const count = node.childCount();
    while (i < count) : (i += 1) {
        if (node.child(i)) |ch| {
            try walkNode(allocator, ch, source, file_path, namespace, language, callables, aliases);
        }
    }
}

fn extractFunction(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    file_path: []const u8,
    namespace: []const []const u8,
    language: Language,
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

    const body_node = node.childByFieldName("body");
    const body_span = if (body_node) |b|
        Span{ .start = b.startByte(), .end = b.endByte() }
    else
        Span{ .start = node.startByte(), .end = node.endByte() };

    // JS functions with `export` are public
    const visibility = detectExportVisibility(node);

    try callables.append(allocator, .{
        .fqn = fqn,
        .name_span = .{ .start = name_node.startByte(), .end = name_node.endByte() },
        .body_span = body_span,
        .signature_span = .{ .start = node.startByte(), .end = if (body_node) |b| b.startByte() else node.endByte() },
        .language = language,
        .file_path = file_path,
        .parameters = params,
        .return_type = null,
        .visibility = visibility,
    });
}

fn extractMethod(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    file_path: []const u8,
    namespace: []const []const u8,
    language: Language,
    callables: *std.ArrayList(CallableDefinitionInfo),
) ExtractError!void {
    const name_node = node.childByFieldName("name") orelse return;
    const method_name = name_node.text(source);
    if (method_name.len == 0) return;

    var fqn_parts = std.ArrayList([]const u8).empty;
    defer fqn_parts.deinit(allocator);
    for (namespace) |ns| try fqn_parts.append(allocator, ns);
    try fqn_parts.append(allocator, method_name);
    const fqn = try paths.buildFqn(allocator, fqn_parts.items);

    const params = try extractParams(allocator, node, source);

    const body_node = node.childByFieldName("body");
    const body_span = if (body_node) |b|
        Span{ .start = b.startByte(), .end = b.endByte() }
    else
        Span{ .start = node.startByte(), .end = node.endByte() };

    // Methods starting with _ are private
    const visibility: Visibility = if (method_name.len >= 1 and method_name[0] == '_')
        .private
    else
        .public;

    try callables.append(allocator, .{
        .fqn = fqn,
        .name_span = .{ .start = name_node.startByte(), .end = name_node.endByte() },
        .body_span = body_span,
        .signature_span = .{ .start = node.startByte(), .end = if (body_node) |b| b.startByte() else node.endByte() },
        .language = language,
        .file_path = file_path,
        .parameters = params,
        .return_type = null,
        .visibility = visibility,
    });
}

fn extractClass(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    file_path: []const u8,
    namespace: []const []const u8,
    language: Language,
    callables: *std.ArrayList(CallableDefinitionInfo),
    aliases: *std.ArrayList(AliasInfo),
) ExtractError!void {
    const name_node = node.childByFieldName("name") orelse return;
    const class_name = name_node.text(source);
    if (class_name.len == 0) return;

    var new_ns = std.ArrayList([]const u8).empty;
    defer new_ns.deinit(allocator);
    for (namespace) |ns| try new_ns.append(allocator, ns);
    try new_ns.append(allocator, class_name);

    if (node.childByFieldName("body")) |body| {
        var i: u32 = 0;
        const count = body.childCount();
        while (i < count) : (i += 1) {
            if (body.child(i)) |ch| {
                try walkNode(allocator, ch, source, file_path, new_ns.items, language, callables, aliases);
            }
        }
    }
}

fn extractArrowFunctions(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    file_path: []const u8,
    namespace: []const []const u8,
    language: Language,
    callables: *std.ArrayList(CallableDefinitionInfo),
) ExtractError!void {
    // Matches arrow function declarations: `foo = (x) => { ... }` or `foo = () => expr`
    var i: u32 = 0;
    const count = node.childCount();
    while (i < count) : (i += 1) {
        const ch = node.child(i) orelse continue;
        if (!std.mem.eql(u8, ch.kind(), "variable_declarator")) continue;

        const name_node = ch.childByFieldName("name") orelse continue;
        const value_node = ch.childByFieldName("value") orelse continue;

        if (!std.mem.eql(u8, value_node.kind(), "arrow_function")) continue;

        const func_name = name_node.text(source);
        if (func_name.len == 0) continue;

        var fqn_parts = std.ArrayList([]const u8).empty;
        defer fqn_parts.deinit(allocator);
        for (namespace) |ns| try fqn_parts.append(allocator, ns);
        try fqn_parts.append(allocator, func_name);
        const fqn = try paths.buildFqn(allocator, fqn_parts.items);

        const params = try extractParams(allocator, value_node, source);
        const body_node = value_node.childByFieldName("body");
        const body_span = if (body_node) |b|
            Span{ .start = b.startByte(), .end = b.endByte() }
        else
            Span{ .start = value_node.startByte(), .end = value_node.endByte() };

        const visibility = detectExportVisibility(node);

        try callables.append(allocator, .{
            .fqn = fqn,
            .name_span = .{ .start = name_node.startByte(), .end = name_node.endByte() },
            .body_span = body_span,
            .signature_span = .{ .start = value_node.startByte(), .end = if (body_node) |b| b.startByte() else value_node.endByte() },
            .language = language,
            .file_path = file_path,
            .parameters = params,
            .return_type = null,
            .visibility = visibility,
        });
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

        if (std.mem.eql(u8, ch_kind, "(") or
            std.mem.eql(u8, ch_kind, ")") or
            std.mem.eql(u8, ch_kind, ","))
        {
            continue;
        }

        if (std.mem.eql(u8, ch_kind, "identifier")) {
            try params.append(allocator, .{ .name = ch.text(source), .type_annotation = null });
        } else if (std.mem.eql(u8, ch_kind, "required_parameter") or
            std.mem.eql(u8, ch_kind, "optional_parameter"))
        {
            // TypeScript typed parameters
            const pn = ch.childByFieldName("pattern") orelse ch.child(0);
            if (pn) |p| {
                const type_node = ch.childByFieldName("type");
                try params.append(allocator, .{
                    .name = p.text(source),
                    .type_annotation = if (type_node) |t| t.text(source) else null,
                });
            }
        } else if (std.mem.eql(u8, ch_kind, "assignment_pattern")) {
            // Default parameters: `x = 5`
            if (ch.childByFieldName("left")) |left| {
                try params.append(allocator, .{ .name = left.text(source), .type_annotation = null });
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
    const module_path = if (namespace.len > 0) namespace[0] else "";

    const source_node = node.childByFieldName("source") orelse return;
    const import_path = source_node.text(source);
    if (import_path.len < 2) return;

    // Strip quotes
    const path_stripped = import_path[1 .. import_path.len - 1];
    const module_canonical = try paths.jsModuleToCanonical(allocator, path_stripped);

    // Walk children to find import clause with named/default/namespace imports
    var i: u32 = 0;
    const count = node.childCount();
    var found_specific = false;
    while (i < count) : (i += 1) {
        const ch = node.child(i) orelse continue;
        const ch_kind = ch.kind();

        if (std.mem.eql(u8, ch_kind, "import_clause")) {
            // Walk import_clause children
            var j: u32 = 0;
            const cl_count = ch.childCount();
            while (j < cl_count) : (j += 1) {
                const cl_ch = ch.child(j) orelse continue;
                const cl_kind = cl_ch.kind();

                if (std.mem.eql(u8, cl_kind, "named_imports")) {
                    // import { foo, bar } from 'module'
                    try extractNamedImports(allocator, cl_ch, source, module_path, module_canonical, aliases);
                    found_specific = true;
                } else if (std.mem.eql(u8, cl_kind, "identifier")) {
                    // import React from 'react' (default import)
                    const name = cl_ch.text(source);
                    if (name.len > 0) {
                        const alias_fqn = try paths.buildFqn(allocator, &.{ module_path, name });
                        try aliases.append(allocator, .{
                            .alias_fqn = alias_fqn,
                            .target_path_guess = try allocator.dupe(u8, module_canonical),
                            .defining_module = module_path,
                            .is_public = true,
                        });
                        found_specific = true;
                    }
                } else if (std.mem.eql(u8, cl_kind, "namespace_import")) {
                    // import * as utils from './utils'
                    if (cl_ch.childByFieldName("name") orelse cl_ch.child(cl_ch.childCount() -| 1)) |name_node| {
                        if (std.mem.eql(u8, name_node.kind(), "identifier")) {
                            const name = name_node.text(source);
                            if (name.len > 0) {
                                const alias_fqn = try paths.buildFqn(allocator, &.{ module_path, name });
                                try aliases.append(allocator, .{
                                    .alias_fqn = alias_fqn,
                                    .target_path_guess = try allocator.dupe(u8, module_canonical),
                                    .defining_module = module_path,
                                    .is_public = true,
                                });
                                found_specific = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // Fallback: if no specific imports found, create one alias for the module
    if (!found_specific) {
        try aliases.append(allocator, .{
            .alias_fqn = try allocator.dupe(u8, module_canonical),
            .target_path_guess = try allocator.dupe(u8, module_canonical),
            .defining_module = module_path,
            .is_public = true,
        });
    }

    allocator.free(module_canonical);
}

fn extractNamedImports(
    allocator: std.mem.Allocator,
    named_imports_node: Node,
    source: []const u8,
    module_path: []const u8,
    module_canonical: []const u8,
    aliases: *std.ArrayList(AliasInfo),
) ExtractError!void {
    var i: u32 = 0;
    const count = named_imports_node.childCount();
    while (i < count) : (i += 1) {
        const ch = named_imports_node.child(i) orelse continue;
        const ch_kind = ch.kind();

        if (std.mem.eql(u8, ch_kind, "import_specifier")) {
            // import { foo } or import { foo as bar }
            const name_node = ch.childByFieldName("name") orelse continue;
            const alias_node = ch.childByFieldName("alias");
            const original_name = name_node.text(source);
            if (original_name.len == 0) continue;

            const local_name = if (alias_node) |a| a.text(source) else original_name;
            if (local_name.len == 0) continue;

            // target = module_canonical + "::" + original_name
            const target = try allocator.alloc(u8, module_canonical.len + 2 + original_name.len);
            errdefer allocator.free(target);
            @memcpy(target[0..module_canonical.len], module_canonical);
            target[module_canonical.len] = ':';
            target[module_canonical.len + 1] = ':';
            @memcpy(target[module_canonical.len + 2 ..], original_name);

            const alias_fqn = try paths.buildFqn(allocator, &.{ module_path, local_name });
            errdefer allocator.free(alias_fqn);
            try aliases.append(allocator, .{
                .alias_fqn = alias_fqn,
                .target_path_guess = target,
                .defining_module = module_path,
                .is_public = true,
            });
        }
    }
}

/// Build import context from JS/TS import statements.
pub fn buildImportContext(
    allocator: std.mem.Allocator,
    root: Node,
    source: []const u8,
) ExtractError![]const ImportEntry {
    var entries = std.ArrayList(ImportEntry).empty;
    errdefer entries.deinit(allocator);

    try walkJsImports(allocator, root, source, &entries);

    return entries.toOwnedSlice(allocator);
}

fn walkJsImports(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    entries: *std.ArrayList(ImportEntry),
) ExtractError!void {
    const kind = node.kind();

    if (std.mem.eql(u8, kind, "import_statement")) {
        try collectJsImportEntries(allocator, node, source, entries);
    }

    if (!std.mem.eql(u8, kind, "function_declaration") and
        !std.mem.eql(u8, kind, "arrow_function"))
    {
        var i: u32 = 0;
        const count = node.childCount();
        while (i < count) : (i += 1) {
            if (node.child(i)) |ch| {
                try walkJsImports(allocator, ch, source, entries);
            }
        }
    }
}

fn collectJsImportEntries(
    allocator: std.mem.Allocator,
    node: Node,
    source: []const u8,
    entries: *std.ArrayList(ImportEntry),
) ExtractError!void {
    const source_node = node.childByFieldName("source") orelse return;
    const import_path = source_node.text(source);
    if (import_path.len < 2) return;

    const path_stripped = import_path[1 .. import_path.len - 1];
    const module_canonical = try paths.jsModuleToCanonical(allocator, path_stripped);
    defer allocator.free(module_canonical);

    var i: u32 = 0;
    const count = node.childCount();
    while (i < count) : (i += 1) {
        const ch = node.child(i) orelse continue;
        if (!std.mem.eql(u8, ch.kind(), "import_clause")) continue;

        var j: u32 = 0;
        const cl_count = ch.childCount();
        while (j < cl_count) : (j += 1) {
            const cl_ch = ch.child(j) orelse continue;
            const cl_kind = cl_ch.kind();

            if (std.mem.eql(u8, cl_kind, "named_imports")) {
                // { foo, bar as baz }
                var k: u32 = 0;
                const ni_count = cl_ch.childCount();
                while (k < ni_count) : (k += 1) {
                    const spec = cl_ch.child(k) orelse continue;
                    if (!std.mem.eql(u8, spec.kind(), "import_specifier")) continue;

                    const name_node = spec.childByFieldName("name") orelse continue;
                    const alias_node = spec.childByFieldName("alias");
                    const original = name_node.text(source);
                    if (original.len == 0) continue;

                    const local = if (alias_node) |a| a.text(source) else original;
                    if (local.len == 0) continue;

                    // target = module_canonical + "::" + original
                    const target = try allocator.alloc(u8, module_canonical.len + 2 + original.len);
                    @memcpy(target[0..module_canonical.len], module_canonical);
                    target[module_canonical.len] = ':';
                    target[module_canonical.len + 1] = ':';
                    @memcpy(target[module_canonical.len + 2 ..], original);

                    try entries.append(allocator, .{
                        .local_name = try allocator.dupe(u8, local),
                        .canonical_path = target,
                    });
                }
            } else if (std.mem.eql(u8, cl_kind, "identifier")) {
                // default import: import React from 'react'
                const name = cl_ch.text(source);
                if (name.len > 0) {
                    try entries.append(allocator, .{
                        .local_name = try allocator.dupe(u8, name),
                        .canonical_path = try allocator.dupe(u8, module_canonical),
                    });
                }
            } else if (std.mem.eql(u8, cl_kind, "namespace_import")) {
                // import * as utils from './utils'
                if (cl_ch.childByFieldName("name") orelse cl_ch.child(cl_ch.childCount() -| 1)) |name_node| {
                    if (std.mem.eql(u8, name_node.kind(), "identifier")) {
                        const name = name_node.text(source);
                        if (name.len > 0) {
                            try entries.append(allocator, .{
                                .local_name = try allocator.dupe(u8, name),
                                .canonical_path = try allocator.dupe(u8, module_canonical),
                            });
                        }
                    }
                }
            }
        }
    }
}

/// Generate candidate resolution paths for a JS/TS call target.
///
/// Same strategy as Python: dot-separated names use import context lookup.
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
        const prefix = raw_target[0..dot_pos];
        const rest = raw_target[dot_pos + 1 ..];
        for (imports) |entry| {
            if (std.mem.eql(u8, entry.local_name, prefix)) {
                const rest_canonical = try paths.dottedToCanonical(allocator, rest);
                defer allocator.free(rest_canonical);
                const combined = try allocator.alloc(u8, entry.canonical_path.len + rest_canonical.len);
                @memcpy(combined[0..entry.canonical_path.len], entry.canonical_path);
                @memcpy(combined[entry.canonical_path.len..], rest_canonical);
                try resolved.append(allocator, combined);
                break;
            }
        }
    } else {
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

fn detectExportVisibility(node: Node) Visibility {
    // Check if parent is export_statement
    if (node.parent()) |p| {
        if (std.mem.eql(u8, p.kind(), "export_statement")) {
            return .public;
        }
    }
    return .unknown;
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
    }

    if (!std.mem.eql(u8, kind, "function_declaration") and
        !std.mem.eql(u8, kind, "arrow_function"))
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

        const arg_source: ArgumentSource = if (std.mem.eql(u8, ch_kind, "string") or
            std.mem.eql(u8, ch_kind, "number") or
            std.mem.eql(u8, ch_kind, "true") or
            std.mem.eql(u8, ch_kind, "false") or
            std.mem.eql(u8, ch_kind, "null") or
            std.mem.eql(u8, ch_kind, "undefined"))
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
    // Matches variable binding: `x = foo()` or `y = bar()`
    const parent_node = call_node.parent() orelse return null;
    if (!std.mem.eql(u8, parent_node.kind(), "variable_declarator")) return null;

    if (parent_node.childByFieldName("name")) |name| {
        if (std.mem.eql(u8, name.kind(), "identifier")) {
            return name.text(source);
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

test "extractCallables finds JS function declaration" {
    var p = try parser_mod.Parser.init(.javascript);
    defer p.deinit();
    const source = "function hello(name) { return name; }";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const result = try extractCallables(std.testing.allocator, tree.rootNode(), source, "test.js", "", .javascript);
    defer freeExtractionResult(std.testing.allocator, result);

    try std.testing.expectEqual(@as(usize, 1), result.callables.len);
    try std.testing.expect(std.mem.endsWith(u8, result.callables[0].fqn, "::hello"));
    try std.testing.expectEqual(@as(usize, 1), result.callables[0].parameters.len);
    try std.testing.expectEqualStrings("name", result.callables[0].parameters[0].name);
}

test "extractCallables finds arrow functions" {
    var p = try parser_mod.Parser.init(.javascript);
    defer p.deinit();
    const source = "const greet = (name) => { return name; }";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const result = try extractCallables(std.testing.allocator, tree.rootNode(), source, "test.js", "", .javascript);
    defer freeExtractionResult(std.testing.allocator, result);

    try std.testing.expectEqual(@as(usize, 1), result.callables.len);
    try std.testing.expect(std.mem.endsWith(u8, result.callables[0].fqn, "::greet"));
}

test "buildImportContext handles named and default imports" {
    var p = try parser_mod.Parser.init(.javascript);
    defer p.deinit();
    const source = "import React from 'react';\nimport { readFile } from 'fs';\n";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const ctx = try buildImportContext(std.testing.allocator, tree.rootNode(), source);
    defer {
        for (ctx) |entry| {
            std.testing.allocator.free(entry.local_name);
            std.testing.allocator.free(entry.canonical_path);
        }
        std.testing.allocator.free(ctx);
    }

    try std.testing.expectEqual(@as(usize, 2), ctx.len);
    // import React from 'react' -> local="React", target="::react"
    try std.testing.expectEqualStrings("React", ctx[0].local_name);
    try std.testing.expectEqualStrings("::react", ctx[0].canonical_path);
    // import { readFile } from 'fs' -> local="readFile", target="::fs::readFile"
    try std.testing.expectEqualStrings("readFile", ctx[1].local_name);
    try std.testing.expectEqualStrings("::fs::readFile", ctx[1].canonical_path);
}

test "generateResolutionPaths resolves JS dotted call" {
    const imports = &[_]ImportEntry{
        .{ .local_name = "React", .canonical_path = "::react" },
    };
    const resolved = try generateResolutionPaths(std.testing.allocator, "React.createElement", imports, "");
    defer {
        for (resolved) |p| std.testing.allocator.free(p);
        std.testing.allocator.free(resolved);
    }

    try std.testing.expectEqual(@as(usize, 2), resolved.len);
    try std.testing.expectEqualStrings("::React::createElement", resolved[0]);
    try std.testing.expectEqualStrings("::react::createElement", resolved[1]);
}

fn freeCallSites(allocator: std.mem.Allocator, calls: []CallSiteDetail) void {
    for (calls) |cs| {
        for (cs.potential_resolved_paths) |rp| allocator.free(rp);
        allocator.free(cs.potential_resolved_paths);
        allocator.free(cs.arguments);
    }
    allocator.free(calls);
}

test "extractCallSites finds simple function call" {
    var p = try parser_mod.Parser.init(.javascript);
    defer p.deinit();
    const source = "foo(1, 2)";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const calls = try extractCallSites(std.testing.allocator, tree.rootNode(), source, "::test::main", &.{});
    defer freeCallSites(std.testing.allocator, calls);

    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("foo", calls[0].raw_target_name);
    try std.testing.expectEqualStrings("::test::main", calls[0].definer_callable_fqn);
    try std.testing.expectEqual(@as(usize, 2), calls[0].arguments.len);
}

test "extractCallSites finds method call with dotted target" {
    var p = try parser_mod.Parser.init(.javascript);
    defer p.deinit();
    const source = "obj.method()";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const calls = try extractCallSites(std.testing.allocator, tree.rootNode(), source, "::mod::fn", &.{});
    defer freeCallSites(std.testing.allocator, calls);

    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("obj.method", calls[0].raw_target_name);
}

test "extractCallSites classifies argument types" {
    var p = try parser_mod.Parser.init(.javascript);
    defer p.deinit();
    const source = "foo(42, \"hello\", x, bar())";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const calls = try extractCallSites(std.testing.allocator, tree.rootNode(), source, "::m::f", &.{});
    defer freeCallSites(std.testing.allocator, calls);

    // foo(...) is the outer call; bar() is nested inside it
    // Find the outer call (the one targeting "foo")
    var foo_call: ?CallSiteDetail = null;
    for (calls) |cs| {
        if (std.mem.eql(u8, cs.raw_target_name, "foo")) {
            foo_call = cs;
            break;
        }
    }
    try std.testing.expect(foo_call != null);
    const args = foo_call.?.arguments;
    try std.testing.expectEqual(@as(usize, 4), args.len);
    try std.testing.expectEqual(ArgumentSource.literal, args[0].source);
    try std.testing.expectEqualStrings("42", args[0].text);
    try std.testing.expectEqual(ArgumentSource.literal, args[1].source);
    try std.testing.expectEqual(ArgumentSource.variable, args[2].source);
    try std.testing.expectEqual(ArgumentSource.function_call, args[3].source);
}

test "extractCallSites captures result variable" {
    var p = try parser_mod.Parser.init(.javascript);
    defer p.deinit();
    const source = "const result = getValue()";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const calls = try extractCallSites(std.testing.allocator, tree.rootNode(), source, "::m::f", &.{});
    defer freeCallSites(std.testing.allocator, calls);

    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("getValue", calls[0].raw_target_name);
    try std.testing.expect(calls[0].result_usage_variable != null);
    try std.testing.expectEqualStrings("result", calls[0].result_usage_variable.?);
}

test "extractCallSites resolves dotted call via imports" {
    var p = try parser_mod.Parser.init(.javascript);
    defer p.deinit();
    const source = "React.createElement('div')";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const imports = &[_]ImportEntry{
        .{ .local_name = "React", .canonical_path = "::react" },
    };
    const calls = try extractCallSites(std.testing.allocator, tree.rootNode(), source, "::app::render", imports);
    defer freeCallSites(std.testing.allocator, calls);

    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("React.createElement", calls[0].raw_target_name);
    // Should have naive + resolved paths
    try std.testing.expect(calls[0].potential_resolved_paths.len >= 2);
    try std.testing.expectEqualStrings("::React::createElement", calls[0].potential_resolved_paths[0]);
    try std.testing.expectEqualStrings("::react::createElement", calls[0].potential_resolved_paths[1]);
}

test "extractCallSites returns empty for source with no calls" {
    var p = try parser_mod.Parser.init(.javascript);
    defer p.deinit();
    const source = "const x = 42";
    var tree = try p.parse(source, null);
    defer tree.deinit();

    const calls = try extractCallSites(std.testing.allocator, tree.rootNode(), source, "::m::f", &.{});
    defer freeCallSites(std.testing.allocator, calls);

    try std.testing.expectEqual(@as(usize, 0), calls.len);
}

test "generateResolutionPaths adds current-module candidate for simple JS name" {
    const resolved = try generateResolutionPaths(
        std.testing.allocator,
        "doWork",
        &.{},
        "::app::components::render",
    );
    defer {
        for (resolved) |p| std.testing.allocator.free(p);
        std.testing.allocator.free(resolved);
    }

    try std.testing.expectEqual(@as(usize, 2), resolved.len);
    try std.testing.expectEqualStrings("::doWork", resolved[0]);
    try std.testing.expectEqualStrings("::app::components::doWork", resolved[1]);
}
