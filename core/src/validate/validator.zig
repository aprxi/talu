//! High-level schema validator.
//!
//! Wraps Grammar + Engine for common validation workflows:
//! - One-shot validation of complete JSON
//! - Streaming validation byte-by-byte
//! - Position tracking for error reporting

const std = @import("std");
const schema_mod = @import("schema.zig");
const engine_mod = @import("engine.zig");
const ast_mod = @import("ast.zig");
const cache = @import("cache.zig");

const Grammar = ast_mod.Grammar;
const Engine = engine_mod.Engine;

/// High-level validator that owns both compiled grammar and engine state.
/// Uses global grammar cache for stable pointer references.
pub const Validator = struct {
    engine: Engine,
    allocator: std.mem.Allocator,
    bytes_consumed: usize,

    /// Create a validator from a JSON schema string.
    /// Grammar is cached globally for pointer stability.
    pub fn init(allocator: std.mem.Allocator, schema_json: []const u8) !Validator {
        // Use global grammar cache for stable pointer (same as ConstrainedSampler)
        const grammar_cache = cache.getGlobalCache(allocator);
        const grammar = grammar_cache.getOrCompile(schema_json, .{}) catch |err| {
            switch (err) {
                error.InvalidSchema, error.UnsupportedFeature, error.RecursionDepthExceeded => return error.InvalidSchema,
                error.OutOfMemory => return error.OutOfMemory,
            }
        };

        // Initialize the engine from the cached grammar
        const engine = try Engine.init(allocator, grammar);

        return .{
            .engine = engine,
            .allocator = allocator,
            .bytes_consumed = 0,
        };
    }

    pub fn deinit(self: *Validator) void {
        self.engine.deinit();
        // Grammar is owned by the global cache, don't deinit it here
        self.* = undefined;
    }

    /// Reset to initial state for reuse.
    pub fn reset(self: *Validator) !void {
        try self.engine.reset();
        self.bytes_consumed = 0;
    }

    /// Check if the validator has reached a complete/accepting state.
    pub fn isComplete(self: *const Validator) bool {
        return self.engine.isComplete();
    }

    /// Get current byte position in the stream.
    pub fn getPosition(self: *const Validator) usize {
        return self.bytes_consumed;
    }

    /// Get number of active states (complexity indicator).
    pub fn getStateCount(self: *const Validator) usize {
        return self.engine.states.stacks.items.len;
    }

    /// Get valid next bytes from current state.
    pub fn getValidFirstBytes(self: *const Validator, valid: *[256]bool) void {
        for (valid) |*v| v.* = false;
        self.engine.getValidFirstBytes(valid);
    }

    /// Count valid next bytes from current state.
    pub fn countValidBytes(self: *const Validator) usize {
        var valid: [256]bool = [_]bool{false} ** 256;
        self.engine.getValidFirstBytes(&valid);

        var count: usize = 0;
        for (valid) |v| {
            if (v) count += 1;
        }
        return count;
    }

    /// Check if byte sequence can be accepted from current state (read-only).
    pub fn canAccept(self: *Validator, data: []const u8) !bool {
        return self.engine.canAccept(data);
    }

    /// Advance state by one byte.
    /// Returns true if valid and advanced, false if invalid.
    pub fn advanceByte(self: *Validator, byte: u8) !bool {
        var valid: [256]bool = [_]bool{false} ** 256;
        self.engine.getValidFirstBytes(&valid);

        if (!valid[byte]) {
            return false;
        }

        try self.engine.advance(&[_]u8{byte});
        self.bytes_consumed += 1;
        return true;
    }

    /// Advance state by byte sequence.
    /// Returns number of bytes successfully consumed.
    pub fn advance(self: *Validator, data: []const u8) !usize {
        // Fast path: check if entire sequence can be accepted
        if (try self.engine.canAccept(data)) {
            try self.engine.advance(data);
            self.bytes_consumed += data.len;
            return data.len;
        }

        // Slow path: byte-by-byte
        var consumed: usize = 0;
        for (data) |byte| {
            var valid: [256]bool = [_]bool{false} ** 256;
            self.engine.getValidFirstBytes(&valid);

            if (!valid[byte]) {
                break;
            }

            try self.engine.advance(&[_]u8{byte});
            consumed += 1;
        }

        self.bytes_consumed += consumed;
        return consumed;
    }

    /// Validate complete input: resets, advances, checks completion.
    /// Returns true if valid and complete.
    pub fn validate(self: *Validator, data: []const u8) !bool {
        try self.reset();

        if (!(try self.engine.canAccept(data))) {
            return false;
        }

        try self.engine.advance(data);
        self.bytes_consumed = data.len;

        return self.engine.isComplete();
    }

    /// Get deterministic continuation if grammar requires specific bytes next.
    pub fn getDeterministicContinuation(self: *const Validator) ?[]const u8 {
        return self.engine.getDeterministicContinuation();
    }
};

test "Validator.init compiles object schema" {
    const allocator = std.testing.allocator;

    // Use an object schema (matches schema.zig test patterns)
    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    );
    defer validator.deinit();

    try std.testing.expectEqual(@as(usize, 0), validator.bytes_consumed);
}

test "Validator.init returns error for invalid schema" {
    const allocator = std.testing.allocator;

    // Invalid schema (missing type)
    const result = Validator.init(allocator, "not valid json");
    try std.testing.expectError(error.InvalidSchema, result);
}

test "Validator.deinit cleans up resources" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"x": {"type": "integer"}}}
    );
    validator.deinit();
    // No leak = test passes (std.testing.allocator checks)
}

test "Validator.reset clears bytes_consumed" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"x": {"type": "integer"}}}
    );
    defer validator.deinit();

    // Manually set bytes_consumed to simulate advance
    validator.bytes_consumed = 10;
    try validator.reset();

    try std.testing.expectEqual(@as(usize, 0), validator.bytes_consumed);
}

test "Validator.getPosition returns bytes_consumed" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"x": {"type": "integer"}}}
    );
    defer validator.deinit();

    validator.bytes_consumed = 42;
    try std.testing.expectEqual(@as(usize, 42), validator.getPosition());
}

test "Validator.getStateCount returns engine state count" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"x": {"type": "integer"}}}
    );
    defer validator.deinit();

    // After init, should have at least one state
    try std.testing.expect(validator.getStateCount() > 0);
}

test "Validator.countValidBytes returns count from engine" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"x": {"type": "integer"}}}
    );
    defer validator.deinit();

    // Should have at least '{' as valid first byte for object
    const count = validator.countValidBytes();
    try std.testing.expect(count > 0);
}

test "Validator.getValidFirstBytes populates array" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"x": {"type": "integer"}}}
    );
    defer validator.deinit();

    var valid: [256]bool = undefined;
    validator.getValidFirstBytes(&valid);

    // '{' should be valid for object start
    try std.testing.expectEqual(true, valid['{']);
}

test "Validator.advanceByte rejects invalid byte" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"x": {"type": "integer"}}}
    );
    defer validator.deinit();

    // 'x' is not a valid first byte (expecting '{')
    const advanced = try validator.advanceByte('x');
    try std.testing.expectEqual(false, advanced);
    try std.testing.expectEqual(@as(usize, 0), validator.getPosition());
}

test "Validator.advanceByte accepts valid byte and updates position" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"x": {"type": "integer"}}}
    );
    defer validator.deinit();

    // '{' should be valid for object start
    const advanced = try validator.advanceByte('{');
    try std.testing.expectEqual(true, advanced);
    try std.testing.expectEqual(@as(usize, 1), validator.getPosition());
}

test "Validator.canAccept does not modify state" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"x": {"type": "integer"}}}
    );
    defer validator.deinit();

    const pos_before = validator.getPosition();
    _ = try validator.canAccept("{");
    const pos_after = validator.getPosition();

    try std.testing.expectEqual(pos_before, pos_after);
}
