//! Python Function Documentation Parser
//!
//! Extracts function metadata (name, parameters, types, docstrings) from Python
//! source code strings. This enables templates to describe Python functions
//! without Python-side introspection.
//!
//! ## Example
//!
//! ```zig
//! const source =
//!     \\def search(query: str, limit: int = 10) -> list:
//!     \\    """Search the web for information.
//!     \\
//!     \\    Args:
//!     \\        query: The search query
//!     \\        limit: Maximum results to return
//!     \\    """
//!     \\    pass
//! ;
//!
//! const functions = try parseFunctions(allocator, source);
//! // Returns array with function metadata including name, description, parameters
//! ```
//!
//! ## Supported Features
//!
//! - Function signatures: `def name(params) -> return_type:`
//! - Type annotations: `str`, `int`, `float`, `bool`, `list`, `dict`, `Optional[T]`
//! - Default values: captured as source text
//! - Google-style docstrings: extracts description and Args section
//! - Async functions: `async def` parsed, marked with `async: true`

const std = @import("std");
const eval = @import("eval.zig");
const TemplateInput = eval.TemplateInput;

/// A parsed parameter from a Python function signature
pub const Parameter = struct {
    name: []const u8,
    type_name: []const u8, // "string", "integer", "number", "boolean", "array", "object", "any"
    required: bool,
    default: ?[]const u8,
    description: ?[]const u8,
};

/// A parsed Python function
pub const FunctionDoc = struct {
    name: []const u8,
    description: ?[]const u8,
    return_type: ?[]const u8,
    parameters: []const Parameter,
    is_async: bool,

    /// Free all memory allocated for this function doc.
    pub fn deinit(self: *const FunctionDoc, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        if (self.description) |desc| allocator.free(desc);
        for (self.parameters) |param| {
            allocator.free(param.name);
            if (param.default) |def| allocator.free(def);
            if (param.description) |desc| allocator.free(desc);
        }
        allocator.free(self.parameters);
    }
};

/// Free all memory allocated by parseFunctions.
pub fn freeFunctions(allocator: std.mem.Allocator, functions: []const FunctionDoc) void {
    for (functions) |*func| {
        func.deinit(allocator);
    }
    allocator.free(functions);
}

/// Parse Python source code and extract function documentation.
/// Returns an array of FunctionDoc structs.
pub fn parseFunctions(allocator: std.mem.Allocator, source: []const u8) ![]const FunctionDoc {
    var functions = std.ArrayListUnmanaged(FunctionDoc){};
    errdefer functions.deinit(allocator);

    var pos: usize = 0;
    while (pos < source.len) {
        // Skip whitespace and find next potential function
        pos = skipWhitespaceAndComments(source, pos);
        if (pos >= source.len) break;

        // Check for async def or def
        var is_async = false;
        if (startsWith(source[pos..], "async ")) {
            is_async = true;
            pos += 6;
            pos = skipWhitespace(source, pos);
        }

        if (startsWith(source[pos..], "def ")) {
            pos += 4;
            if (try parseFunction(allocator, source, &pos, is_async)) |func| {
                try functions.append(allocator, func);
            }
        } else {
            // Skip to next line
            pos = skipToNextLine(source, pos);
        }
    }

    return functions.toOwnedSlice(allocator);
}

/// Parse a single function starting after "def "
fn parseFunction(allocator: std.mem.Allocator, source: []const u8, pos: *usize, is_async: bool) !?FunctionDoc {
    // Parse function name
    const name_start = pos.*;
    while (pos.* < source.len and (std.ascii.isAlphanumeric(source[pos.*]) or source[pos.*] == '_')) {
        pos.* += 1;
    }
    if (pos.* == name_start) return null;
    const name = source[name_start..pos.*];

    // Skip to opening paren
    pos.* = skipWhitespace(source, pos.*);
    if (pos.* >= source.len or source[pos.*] != '(') return null;
    pos.* += 1;

    // Parse parameters
    var params = std.ArrayListUnmanaged(Parameter){};
    errdefer params.deinit(allocator);

    while (pos.* < source.len and source[pos.*] != ')') {
        pos.* = skipWhitespace(source, pos.*);
        if (pos.* >= source.len) break;
        if (source[pos.*] == ')') break;

        // Skip 'self' parameter
        if (startsWith(source[pos.*..], "self")) {
            const after_self = pos.* + 4;
            if (after_self >= source.len or source[after_self] == ',' or source[after_self] == ')' or source[after_self] == ':') {
                pos.* = after_self;
                if (pos.* < source.len and source[pos.*] == ',') pos.* += 1;
                continue;
            }
        }

        // Skip *args, **kwargs
        if (source[pos.*] == '*') {
            // Skip to comma or closing paren
            while (pos.* < source.len and source[pos.*] != ',' and source[pos.*] != ')') {
                pos.* += 1;
            }
            if (pos.* < source.len and source[pos.*] == ',') pos.* += 1;
            continue;
        }

        if (try parseParameter(allocator, source, pos)) |param| {
            try params.append(allocator, param);
        }

        // Skip comma
        pos.* = skipWhitespace(source, pos.*);
        if (pos.* < source.len and source[pos.*] == ',') {
            pos.* += 1;
        }
    }

    // Skip closing paren
    if (pos.* < source.len and source[pos.*] == ')') pos.* += 1;

    // Parse return type
    pos.* = skipWhitespace(source, pos.*);
    var return_type: ?[]const u8 = null;
    if (pos.* + 1 < source.len and source[pos.*] == '-' and source[pos.* + 1] == '>') {
        pos.* += 2;
        pos.* = skipWhitespace(source, pos.*);
        const rt_start = pos.*;
        while (pos.* < source.len and source[pos.*] != ':' and source[pos.*] != '\n') {
            pos.* += 1;
        }
        const rt_raw = std.mem.trim(u8, source[rt_start..pos.*], " \t");
        return_type = mapPythonType(allocator, rt_raw);
    }

    // Skip to colon
    while (pos.* < source.len and source[pos.*] != ':') {
        pos.* += 1;
    }
    if (pos.* < source.len) pos.* += 1; // Skip colon

    // Parse docstring
    pos.* = skipWhitespace(source, pos.*);
    var description: ?[]const u8 = null;
    var param_descriptions = std.StringHashMap([]const u8).init(allocator);
    defer param_descriptions.deinit();

    var docstring_keys_to_free: ?std.StringHashMap([]const u8) = null;
    if (parseDocstring(allocator, source, pos)) |docstring_result| {
        description = docstring_result.description;
        var it = docstring_result.param_descriptions.iterator();
        while (it.next()) |entry| {
            try param_descriptions.put(entry.key_ptr.*, entry.value_ptr.*);
        }
        // Save the map so we can free the keys after using param_descriptions
        docstring_keys_to_free = docstring_result.param_descriptions;
    }

    // Update parameter descriptions from docstring
    var final_params = try allocator.alloc(Parameter, params.items.len);
    for (params.items, 0..) |param, i| {
        final_params[i] = Parameter{
            .name = param.name,
            .type_name = param.type_name,
            .required = param.required,
            .default = param.default,
            .description = param_descriptions.get(param.name) orelse param.description,
        };
    }

    // Now that we're done using param_descriptions, free the docstring keys
    if (docstring_keys_to_free) |*keys_map| {
        var key_iter = keys_map.keyIterator();
        while (key_iter.next()) |key| {
            allocator.free(key.*);
        }
        keys_map.deinit();
    }

    // Free the temporary params ArrayList storage (items were moved to final_params)
    params.deinit(allocator);

    return FunctionDoc{
        .name = try allocator.dupe(u8, name),
        .description = description,
        .return_type = return_type,
        .parameters = final_params,
        .is_async = is_async,
    };
}

/// Parse a single parameter from the signature
fn parseParameter(allocator: std.mem.Allocator, source: []const u8, pos: *usize) !?Parameter {
    pos.* = skipWhitespace(source, pos.*);
    if (pos.* >= source.len) return null;

    // Parse parameter name
    const name_start = pos.*;
    while (pos.* < source.len and (std.ascii.isAlphanumeric(source[pos.*]) or source[pos.*] == '_')) {
        pos.* += 1;
    }
    if (pos.* == name_start) return null;
    const name = try allocator.dupe(u8, source[name_start..pos.*]);

    pos.* = skipWhitespace(source, pos.*);

    // Parse type annotation
    var type_name: []const u8 = "any";
    var is_optional = false;
    if (pos.* < source.len and source[pos.*] == ':') {
        pos.* += 1;
        pos.* = skipWhitespace(source, pos.*);

        const type_start = pos.*;
        var bracket_depth: usize = 0;
        while (pos.* < source.len) {
            const c = source[pos.*];
            if (c == '[') {
                bracket_depth += 1;
            } else if (c == ']') {
                if (bracket_depth > 0) bracket_depth -= 1;
            } else if (bracket_depth == 0 and (c == ',' or c == ')' or c == '=')) {
                break;
            }
            pos.* += 1;
        }
        const type_raw = std.mem.trim(u8, source[type_start..pos.*], " \t");

        // Check for Optional[T]
        if (startsWith(type_raw, "Optional[")) {
            is_optional = true;
            // Extract inner type
            const inner_start: usize = 9; // len("Optional[")
            const inner_end = if (std.mem.lastIndexOf(u8, type_raw, "]")) |idx| idx else type_raw.len;
            const inner_type = type_raw[inner_start..inner_end];
            type_name = mapPythonType(allocator, inner_type);
        } else {
            type_name = mapPythonType(allocator, type_raw);
        }
    }

    pos.* = skipWhitespace(source, pos.*);

    // Parse default value
    var default: ?[]const u8 = null;
    var required = true;
    if (pos.* < source.len and source[pos.*] == '=') {
        pos.* += 1;
        pos.* = skipWhitespace(source, pos.*);
        required = false;

        const default_start = pos.*;
        var paren_depth: usize = 0;
        var bracket_depth: usize = 0;
        var in_string = false;
        var string_char: u8 = 0;

        while (pos.* < source.len) {
            const c = source[pos.*];

            if (in_string) {
                if (c == string_char and (pos.* == 0 or source[pos.* - 1] != '\\')) {
                    in_string = false;
                }
            } else {
                if (c == '"' or c == '\'') {
                    in_string = true;
                    string_char = c;
                } else if (c == '(' or c == '[' or c == '{') {
                    paren_depth += 1;
                    if (c == '[') bracket_depth += 1;
                } else if (c == ')' or c == ']' or c == '}') {
                    if (paren_depth > 0) paren_depth -= 1;
                    if (c == ']' and bracket_depth > 0) bracket_depth -= 1;
                    if (paren_depth == 0 and c == ')') break;
                } else if (paren_depth == 0 and c == ',') {
                    break;
                }
            }
            pos.* += 1;
        }

        const default_raw = std.mem.trim(u8, source[default_start..pos.*], " \t");
        if (default_raw.len > 0) {
            default = try allocator.dupe(u8, default_raw);
        }
    }

    // Optional types without default are still not required
    if (is_optional) required = false;

    return Parameter{
        .name = name,
        .type_name = type_name,
        .required = required,
        .default = default,
        .description = null,
    };
}

const DocstringResult = struct {
    description: ?[]const u8,
    param_descriptions: std.StringHashMap([]const u8),
};

/// Parse a docstring and extract description and parameter descriptions
fn parseDocstring(allocator: std.mem.Allocator, source: []const u8, pos: *usize) ?DocstringResult {
    // Skip whitespace including newlines to find docstring
    while (pos.* < source.len and (source[pos.*] == ' ' or source[pos.*] == '\t' or source[pos.*] == '\n' or source[pos.*] == '\r')) {
        pos.* += 1;
    }
    if (pos.* >= source.len) return null;

    // Check for triple quotes
    const quote_char: u8 = if (source[pos.*] == '"') '"' else if (source[pos.*] == '\'') '\'' else return null;

    if (pos.* + 2 >= source.len) return null;
    if (source[pos.*] != quote_char or source[pos.* + 1] != quote_char or source[pos.* + 2] != quote_char) {
        return null;
    }
    pos.* += 3;

    // Find closing triple quotes
    const docstring_start = pos.*;
    var docstring_end: usize = pos.*;
    while (pos.* + 2 < source.len) {
        if (source[pos.*] == quote_char and source[pos.* + 1] == quote_char and source[pos.* + 2] == quote_char) {
            docstring_end = pos.*;
            pos.* += 3;
            break;
        }
        pos.* += 1;
    }

    if (docstring_end <= docstring_start) return null;

    const docstring = source[docstring_start..docstring_end];

    // Parse docstring content
    var description: ?[]const u8 = null;
    var param_descriptions = std.StringHashMap([]const u8).init(allocator);

    // Find first paragraph (description)
    var desc_end: usize = 0;
    var i: usize = 0;
    while (i < docstring.len) {
        if (docstring[i] == '\n') {
            // Check for blank line or Args section
            const next_line_start = i + 1;
            if (next_line_start < docstring.len) {
                const remaining = docstring[next_line_start..];
                const trimmed = std.mem.trimLeft(u8, remaining, " \t");
                if (trimmed.len == 0 or trimmed[0] == '\n') {
                    desc_end = i;
                    break;
                }
                if (startsWith(trimmed, "Args:") or startsWith(trimmed, "Arguments:") or
                    startsWith(trimmed, "Parameters:") or startsWith(trimmed, "Returns:") or
                    startsWith(trimmed, "Raises:") or startsWith(trimmed, "Example:") or
                    startsWith(trimmed, "Examples:") or startsWith(trimmed, "Note:") or
                    startsWith(trimmed, "Notes:"))
                {
                    desc_end = i;
                    break;
                }
            }
        }
        i += 1;
    }

    if (desc_end == 0) {
        // Single line docstring or no sections found
        desc_end = docstring.len;
    }

    const desc_raw = std.mem.trim(u8, docstring[0..desc_end], " \t\n");
    if (desc_raw.len > 0) {
        // Clean up multi-line description (join lines)
        description = cleanDescription(allocator, desc_raw) catch null;
    }

    // Find and parse Args section
    if (std.mem.indexOf(u8, docstring, "Args:")) |args_start| {
        parseArgsSection(allocator, docstring[args_start + 5 ..], &param_descriptions) catch {};
    } else if (std.mem.indexOf(u8, docstring, "Arguments:")) |args_start| {
        parseArgsSection(allocator, docstring[args_start + 10 ..], &param_descriptions) catch {};
    } else if (std.mem.indexOf(u8, docstring, "Parameters:")) |args_start| {
        parseArgsSection(allocator, docstring[args_start + 11 ..], &param_descriptions) catch {};
    }

    return DocstringResult{
        .description = description,
        .param_descriptions = param_descriptions,
    };
}

/// Clean up a multi-line description
fn cleanDescription(allocator: std.mem.Allocator, desc: []const u8) ![]const u8 {
    var result = std.ArrayListUnmanaged(u8){};
    errdefer result.deinit(allocator);

    var lines = std.mem.splitScalar(u8, desc, '\n');
    var first = true;
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t");
        if (trimmed.len == 0) continue;
        if (!first) {
            try result.append(allocator, ' ');
        }
        try result.appendSlice(allocator, trimmed);
        first = false;
    }

    return result.toOwnedSlice(allocator);
}

/// Parse the Args section of a Google-style docstring
fn parseArgsSection(allocator: std.mem.Allocator, section: []const u8, param_descriptions: *std.StringHashMap([]const u8)) !void {
    var lines = std.mem.splitScalar(u8, section, '\n');
    var current_param: ?[]const u8 = null;
    var current_desc = std.ArrayListUnmanaged(u8){};
    defer current_desc.deinit(allocator);

    while (lines.next()) |line| {
        const trimmed = std.mem.trimLeft(u8, line, " \t");

        // Check for end of Args section
        if (trimmed.len > 0 and !std.ascii.isWhitespace(line[0]) and line[0] != ' ' and line[0] != '\t') {
            // New section started
            if (startsWith(trimmed, "Returns:") or startsWith(trimmed, "Raises:") or
                startsWith(trimmed, "Example:") or startsWith(trimmed, "Note:"))
            {
                break;
            }
        }

        // Empty line might end current param
        if (trimmed.len == 0) {
            if (current_param) |param| {
                const desc = try current_desc.toOwnedSlice(allocator);
                try param_descriptions.put(param, desc);
                current_param = null;
                current_desc = std.ArrayListUnmanaged(u8){};
            }
            continue;
        }

        // Check if this is a new parameter (name: description)
        if (std.mem.indexOf(u8, trimmed, ":")) |colon_pos| {
            const before_colon = trimmed[0..colon_pos];
            // Check if it's a valid param name (no spaces, alphanumeric + underscore)
            var is_param_name = before_colon.len > 0;
            for (before_colon) |c| {
                if (!std.ascii.isAlphanumeric(c) and c != '_') {
                    is_param_name = false;
                    break;
                }
            }

            if (is_param_name) {
                // Save previous param
                if (current_param) |param| {
                    const desc = try current_desc.toOwnedSlice(allocator);
                    try param_descriptions.put(param, desc);
                    current_desc = std.ArrayListUnmanaged(u8){};
                }

                current_param = try allocator.dupe(u8, before_colon);
                const desc_part = std.mem.trim(u8, trimmed[colon_pos + 1 ..], " \t");
                if (desc_part.len > 0) {
                    try current_desc.appendSlice(allocator, desc_part);
                }
                continue;
            }
        }

        // Continuation of current parameter description
        if (current_param != null) {
            if (current_desc.items.len > 0) {
                try current_desc.append(allocator, ' ');
            }
            try current_desc.appendSlice(allocator, trimmed);
        }
    }

    // Save final param
    if (current_param) |param| {
        const desc = try current_desc.toOwnedSlice(allocator);
        try param_descriptions.put(param, desc);
    }
}

/// Map Python type annotations to simple type names
fn mapPythonType(allocator: std.mem.Allocator, python_type: []const u8) []const u8 {
    _ = allocator;
    const trimmed = std.mem.trim(u8, python_type, " \t");

    // Basic types
    if (std.mem.eql(u8, trimmed, "str")) return "string";
    if (std.mem.eql(u8, trimmed, "int")) return "integer";
    if (std.mem.eql(u8, trimmed, "float")) return "number";
    if (std.mem.eql(u8, trimmed, "bool")) return "boolean";
    if (std.mem.eql(u8, trimmed, "list")) return "array";
    if (std.mem.eql(u8, trimmed, "dict")) return "object";
    if (std.mem.eql(u8, trimmed, "None")) return "null";
    if (std.mem.eql(u8, trimmed, "Any")) return "any";

    // Generic types
    if (startsWith(trimmed, "List[") or startsWith(trimmed, "list[")) return "array";
    if (startsWith(trimmed, "Dict[") or startsWith(trimmed, "dict[")) return "object";
    if (startsWith(trimmed, "Tuple[") or startsWith(trimmed, "tuple[")) return "array";
    if (startsWith(trimmed, "Set[") or startsWith(trimmed, "set[")) return "array";
    if (startsWith(trimmed, "Sequence[")) return "array";
    if (startsWith(trimmed, "Mapping[")) return "object";
    if (startsWith(trimmed, "Iterable[")) return "array";

    // Union types - just return "any" for now
    if (startsWith(trimmed, "Union[")) return "any";

    // If no match, return as-is (custom type)
    return trimmed;
}

/// Free a TemplateInput returned by toTemplateInput (recursively frees arrays and maps)
pub fn freeTemplateInput(allocator: std.mem.Allocator, input: TemplateInput) void {
    switch (input) {
        .array => |arr| {
            for (arr) |item| {
                freeTemplateInput(allocator, item);
            }
            allocator.free(arr);
        },
        .map => |m| {
            var map_copy = m;
            var iter = map_copy.valueIterator();
            while (iter.next()) |value| {
                freeTemplateInput(allocator, value.*);
            }
            map_copy.deinit(allocator);
        },
        else => {},
    }
}

/// Convert FunctionDoc array to TemplateInput array for use in templates
pub fn toTemplateInput(allocator: std.mem.Allocator, functions: []const FunctionDoc) !TemplateInput {
    var result = std.ArrayListUnmanaged(TemplateInput){};
    errdefer result.deinit(allocator);

    for (functions) |func| {
        var func_map = std.StringHashMapUnmanaged(TemplateInput){};

        try func_map.put(allocator, "name", .{ .string = func.name });
        try func_map.put(allocator, "description", if (func.description) |d| .{ .string = d } else .none);
        try func_map.put(allocator, "return_type", if (func.return_type) |rt| .{ .string = rt } else .none);
        try func_map.put(allocator, "async", .{ .boolean = func.is_async });

        // Convert parameters
        var params = std.ArrayListUnmanaged(TemplateInput){};
        for (func.parameters) |param| {
            var param_map = std.StringHashMapUnmanaged(TemplateInput){};
            try param_map.put(allocator, "name", .{ .string = param.name });
            try param_map.put(allocator, "type", .{ .string = param.type_name });
            try param_map.put(allocator, "required", .{ .boolean = param.required });
            try param_map.put(allocator, "default", if (param.default) |d| .{ .string = d } else .none);
            try param_map.put(allocator, "description", if (param.description) |d| .{ .string = d } else .none);
            try params.append(allocator, .{ .map = param_map });
        }
        try func_map.put(allocator, "parameters", .{ .array = try params.toOwnedSlice(allocator) });

        try result.append(allocator, .{ .map = func_map });
    }

    return .{ .array = try result.toOwnedSlice(allocator) };
}

// ============================================================================
// Helper functions
// ============================================================================

fn startsWith(haystack: []const u8, needle: []const u8) bool {
    if (haystack.len < needle.len) return false;
    return std.mem.eql(u8, haystack[0..needle.len], needle);
}

fn skipWhitespace(source: []const u8, start: usize) usize {
    var pos = start;
    while (pos < source.len and (source[pos] == ' ' or source[pos] == '\t')) {
        pos += 1;
    }
    return pos;
}

fn skipWhitespaceAndComments(source: []const u8, start: usize) usize {
    var pos = start;
    while (pos < source.len) {
        // Skip whitespace
        while (pos < source.len and (source[pos] == ' ' or source[pos] == '\t' or source[pos] == '\n' or source[pos] == '\r')) {
            pos += 1;
        }
        if (pos >= source.len) break;

        // Skip comments
        if (source[pos] == '#') {
            pos = skipToNextLine(source, pos);
            continue;
        }

        // Skip decorators
        if (source[pos] == '@') {
            pos = skipToNextLine(source, pos);
            continue;
        }

        break;
    }
    return pos;
}

fn skipToNextLine(source: []const u8, start: usize) usize {
    var pos = start;
    while (pos < source.len and source[pos] != '\n') {
        pos += 1;
    }
    if (pos < source.len) pos += 1; // Skip newline
    return pos;
}

// ============================================================================
// Tests
// ============================================================================

test "parseFunctions simple function" {
    const source =
        \\def hello(name: str) -> str:
        \\    """Say hello to someone."""
        \\    return f"Hello, {name}!"
    ;

    const functions = try parseFunctions(std.testing.allocator, source);
    defer freeFunctions(std.testing.allocator, functions);

    try std.testing.expectEqual(@as(usize, 1), functions.len);
    try std.testing.expectEqualStrings("hello", functions[0].name);
    try std.testing.expectEqualStrings("Say hello to someone.", functions[0].description.?);
    try std.testing.expectEqualStrings("string", functions[0].return_type.?);
    try std.testing.expectEqual(@as(usize, 1), functions[0].parameters.len);
    try std.testing.expectEqualStrings("name", functions[0].parameters[0].name);
    try std.testing.expectEqualStrings("string", functions[0].parameters[0].type_name);
    try std.testing.expect(functions[0].parameters[0].required);
}

test "parseFunctions function with default parameter" {
    const source =
        \\def search(query: str, limit: int = 10) -> list:
        \\    """Search for something."""
        \\    pass
    ;

    const functions = try parseFunctions(std.testing.allocator, source);
    defer freeFunctions(std.testing.allocator, functions);

    try std.testing.expectEqual(@as(usize, 1), functions.len);
    try std.testing.expectEqual(@as(usize, 2), functions[0].parameters.len);

    try std.testing.expectEqualStrings("query", functions[0].parameters[0].name);
    try std.testing.expect(functions[0].parameters[0].required);

    try std.testing.expectEqualStrings("limit", functions[0].parameters[1].name);
    try std.testing.expect(!functions[0].parameters[1].required);
    try std.testing.expectEqualStrings("10", functions[0].parameters[1].default.?);
}

test "parseFunctions function with google-style docstring" {
    const source =
        \\def search(query: str, limit: int = 10) -> list:
        \\    """Search the web for information.
        \\
        \\    Args:
        \\        query: The search query
        \\        limit: Maximum results to return
        \\    """
        \\    pass
    ;

    const functions = try parseFunctions(std.testing.allocator, source);
    defer freeFunctions(std.testing.allocator, functions);

    try std.testing.expectEqual(@as(usize, 1), functions.len);
    try std.testing.expectEqualStrings("Search the web for information.", functions[0].description.?);
    try std.testing.expectEqualStrings("The search query", functions[0].parameters[0].description.?);
    try std.testing.expectEqualStrings("Maximum results to return", functions[0].parameters[1].description.?);
}

test "parseFunctions multiple functions" {
    const source =
        \\def add(a: int, b: int) -> int:
        \\    """Add two numbers."""
        \\    return a + b
        \\
        \\def multiply(a: int, b: int) -> int:
        \\    """Multiply two numbers."""
        \\    return a * b
    ;

    const functions = try parseFunctions(std.testing.allocator, source);
    defer freeFunctions(std.testing.allocator, functions);

    try std.testing.expectEqual(@as(usize, 2), functions.len);
    try std.testing.expectEqualStrings("add", functions[0].name);
    try std.testing.expectEqualStrings("multiply", functions[1].name);
}

test "parseFunctions async function" {
    const source =
        \\async def fetch(url: str) -> str:
        \\    """Fetch a URL."""
        \\    pass
    ;

    const functions = try parseFunctions(std.testing.allocator, source);
    defer freeFunctions(std.testing.allocator, functions);

    try std.testing.expectEqual(@as(usize, 1), functions.len);
    try std.testing.expect(functions[0].is_async);
}

test "parseFunctions function with Optional type" {
    const source =
        \\def find(name: str, default: Optional[str] = None) -> str:
        \\    """Find something."""
        \\    pass
    ;

    const functions = try parseFunctions(std.testing.allocator, source);
    defer freeFunctions(std.testing.allocator, functions);

    try std.testing.expectEqual(@as(usize, 1), functions.len);
    try std.testing.expectEqual(@as(usize, 2), functions[0].parameters.len);
    try std.testing.expectEqualStrings("string", functions[0].parameters[1].type_name);
    try std.testing.expect(!functions[0].parameters[1].required);
}

test "parseFunctions function without type hints" {
    const source =
        \\def process(data):
        \\    """Process some data."""
        \\    pass
    ;

    const functions = try parseFunctions(std.testing.allocator, source);
    defer freeFunctions(std.testing.allocator, functions);

    try std.testing.expectEqual(@as(usize, 1), functions.len);
    try std.testing.expectEqual(@as(usize, 1), functions[0].parameters.len);
    try std.testing.expectEqualStrings("any", functions[0].parameters[0].type_name);
}

test "parseFunctions function with decorated" {
    const source =
        \\@decorator
        \\def decorated_func(x: int) -> int:
        \\    """A decorated function."""
        \\    return x
    ;

    const functions = try parseFunctions(std.testing.allocator, source);
    defer freeFunctions(std.testing.allocator, functions);

    try std.testing.expectEqual(@as(usize, 1), functions.len);
    try std.testing.expectEqualStrings("decorated_func", functions[0].name);
}

test "toTemplateInput type mapping" {
    const allocator = std.testing.allocator;
    try std.testing.expectEqualStrings("string", mapPythonType(allocator, "str"));
    try std.testing.expectEqualStrings("integer", mapPythonType(allocator, "int"));
    try std.testing.expectEqualStrings("number", mapPythonType(allocator, "float"));
    try std.testing.expectEqualStrings("boolean", mapPythonType(allocator, "bool"));
    try std.testing.expectEqualStrings("array", mapPythonType(allocator, "list"));
    try std.testing.expectEqualStrings("object", mapPythonType(allocator, "dict"));
    try std.testing.expectEqualStrings("array", mapPythonType(allocator, "List[str]"));
    try std.testing.expectEqualStrings("object", mapPythonType(allocator, "Dict[str, int]"));
}

test "toTemplateInput" {
    const source =
        \\def greet(name: str) -> str:
        \\    """Greet someone."""
        \\    pass
    ;

    const functions = try parseFunctions(std.testing.allocator, source);
    defer freeFunctions(std.testing.allocator, functions);

    const input = try toTemplateInput(std.testing.allocator, functions);
    defer freeTemplateInput(std.testing.allocator, input);

    try std.testing.expect(input == .array);
    try std.testing.expectEqual(@as(usize, 1), input.array.len);

    const func = input.array[0];
    try std.testing.expect(func == .map);

    const name = func.map.get("name").?;
    try std.testing.expectEqualStrings("greet", name.string);
}

test "deinit frees function doc" {
    const allocator = std.testing.allocator;

    const source =
        \\def test_func(param1: str, param2: int = 5) -> str:
        \\    """Test function with parameters.
        \\
        \\    Args:
        \\        param1: First parameter
        \\        param2: Second parameter
        \\    """
        \\    pass
    ;

    const functions = try parseFunctions(allocator, source);
    defer {
        for (functions) |*func| {
            func.deinit(allocator);
        }
        allocator.free(functions);
    }

    try std.testing.expectEqual(@as(usize, 1), functions.len);
    try std.testing.expectEqualStrings("test_func", functions[0].name);
}

test "freeFunctions frees function array" {
    const allocator = std.testing.allocator;

    const source =
        \\def func1(x: int) -> int:
        \\    """First function."""
        \\    return x
        \\
        \\def func2(y: str) -> str:
        \\    """Second function."""
        \\    return y
    ;

    const functions = try parseFunctions(allocator, source);
    defer freeFunctions(allocator, functions);

    try std.testing.expectEqual(@as(usize, 2), functions.len);
    try std.testing.expectEqualStrings("func1", functions[0].name);
    try std.testing.expectEqualStrings("func2", functions[1].name);
}

test "freeTemplateInput frees template input" {
    const allocator = std.testing.allocator;

    const source =
        \\def process(data: str, limit: int = 10) -> list:
        \\    """Process data with limit."""
        \\    pass
    ;

    const functions = try parseFunctions(allocator, source);
    defer freeFunctions(allocator, functions);

    const input = try toTemplateInput(allocator, functions);
    defer freeTemplateInput(allocator, input);

    try std.testing.expect(input == .array);
    try std.testing.expectEqual(@as(usize, 1), input.array.len);
}
