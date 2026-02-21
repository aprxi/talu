//! Data structures for call graph analysis.
//!
//! Represent callable definitions, call sites, and import aliases extracted
//! from source code ASTs. All string data is either borrowed from the source
//! text or allocated by the caller's allocator.
//!
//! Thread safety: All types are plain data. Safe to share read-only.

const std = @import("std");
const Language = @import("../language.zig").Language;

pub const Visibility = enum {
    public,
    private,
    package,
    unknown,

    pub fn name(self: Visibility) []const u8 {
        return switch (self) {
            .public => "public",
            .private => "private",
            .package => "package",
            .unknown => "unknown",
        };
    }
};

pub const Span = struct {
    start: u32,
    end: u32,
};

pub const ParamDetail = struct {
    name: []const u8,
    type_annotation: ?[]const u8,
};

pub const ArgumentSource = enum {
    literal,
    variable,
    parameter_of_caller,
    function_call,
    expression,

    pub fn name(self: ArgumentSource) []const u8 {
        return switch (self) {
            .literal => "literal",
            .variable => "variable",
            .parameter_of_caller => "parameter_of_caller",
            .function_call => "function_call",
            .expression => "expression",
        };
    }
};

pub const Argument = struct {
    source: ArgumentSource,
    text: []const u8,
};

pub const CallableDefinitionInfo = struct {
    /// Canonical fully-qualified name (e.g., "::module::Class::method").
    fqn: []const u8,
    name_span: Span,
    body_span: Span,
    signature_span: Span,
    language: Language,
    file_path: []const u8,
    parameters: []const ParamDetail,
    return_type: ?[]const u8,
    visibility: Visibility,
};

pub const CallSiteDetail = struct {
    /// The raw target name as written in source (e.g., "os.path.join").
    raw_target_name: []const u8,
    /// Potential canonical FQNs this call might resolve to.
    potential_resolved_paths: []const []const u8,
    call_expr_span: Span,
    target_name_span: Span,
    /// FQN of the callable containing this call site.
    definer_callable_fqn: []const u8,
    arguments: []const Argument,
    result_usage_variable: ?[]const u8,
};

pub const AliasInfo = struct {
    alias_fqn: []const u8,
    target_path_guess: []const u8,
    defining_module: []const u8,
    is_public: bool,
};

/// Result of callable/alias extraction.
pub const ExtractionResult = struct {
    callables: []CallableDefinitionInfo,
    aliases: []AliasInfo,
};

/// Per-file import mapping: local name used in code -> canonical target path.
///
/// E.g., `import numpy as np` produces `{ .local_name = "np", .canonical_path = "::numpy" }`.
pub const ImportEntry = struct {
    local_name: []const u8,
    canonical_path: []const u8,
};

/// Common error set for graph extraction operations.
pub const ExtractError = std.mem.Allocator.Error;

// =============================================================================
// Tests
// =============================================================================

test "Visibility.name returns correct strings" {
    try std.testing.expectEqualStrings("public", Visibility.public.name());
    try std.testing.expectEqualStrings("private", Visibility.private.name());
    try std.testing.expectEqualStrings("unknown", Visibility.unknown.name());
}

test "ArgumentSource.name returns correct strings" {
    try std.testing.expectEqualStrings("literal", ArgumentSource.literal.name());
    try std.testing.expectEqualStrings("variable", ArgumentSource.variable.name());
    try std.testing.expectEqualStrings("function_call", ArgumentSource.function_call.name());
}
