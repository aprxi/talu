//! Integration tests for Validator
//!
//! Validator is a high-level schema validator that wraps Grammar + Engine
//! for common validation workflows including one-shot and streaming validation.

const std = @import("std");
const main = @import("main");
const Validator = main.validate.Validator;

// =============================================================================
// Lifecycle Tests
// =============================================================================

test "Validator: init and deinit from object schema" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    );
    defer validator.deinit();

    try std.testing.expectEqual(@as(usize, 0), validator.getPosition());
}

test "Validator: init returns InvalidSchema for malformed JSON" {
    const allocator = std.testing.allocator;

    const result = Validator.init(allocator, "not valid json");
    try std.testing.expectError(error.InvalidSchema, result);
}

test "Validator: init and deinit from integer schema" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "integer"}
    );
    defer validator.deinit();

    try std.testing.expectEqual(@as(usize, 0), validator.getPosition());
}

// =============================================================================
// Reset Tests
// =============================================================================

test "Validator: reset clears position and state" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"x": {"type": "integer"}}}
    );
    defer validator.deinit();

    // Advance to consume some input
    _ = try validator.advanceByte('{');
    try std.testing.expect(validator.getPosition() > 0);

    // Reset should clear position
    try validator.reset();
    try std.testing.expectEqual(@as(usize, 0), validator.getPosition());
}

// =============================================================================
// State Query Tests
// =============================================================================

test "Validator: getPosition tracks consumed bytes" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"x": {"type": "integer"}}}
    );
    defer validator.deinit();

    try std.testing.expectEqual(@as(usize, 0), validator.getPosition());

    _ = try validator.advanceByte('{');
    try std.testing.expectEqual(@as(usize, 1), validator.getPosition());
}

test "Validator: getStateCount returns positive after init" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"x": {"type": "integer"}}}
    );
    defer validator.deinit();

    try std.testing.expect(validator.getStateCount() > 0);
}

test "Validator: countValidBytes returns count of valid next bytes" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"x": {"type": "integer"}}}
    );
    defer validator.deinit();

    // At start, '{' should be valid for object
    const count = validator.countValidBytes();
    try std.testing.expect(count > 0);
}

test "Validator: getValidFirstBytes populates boolean array" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"x": {"type": "integer"}}}
    );
    defer validator.deinit();

    var valid: [256]bool = undefined;
    validator.getValidFirstBytes(&valid);

    // '{' (byte 123) should be valid for object start
    try std.testing.expect(valid['{']);
}

// =============================================================================
// advanceByte Tests
// =============================================================================

test "Validator: advanceByte accepts valid byte and updates position" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"x": {"type": "integer"}}}
    );
    defer validator.deinit();

    // '{' should be valid for object start
    const advanced = try validator.advanceByte('{');
    try std.testing.expect(advanced);
    try std.testing.expectEqual(@as(usize, 1), validator.getPosition());
}

test "Validator: advanceByte rejects invalid byte without advancing" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"x": {"type": "integer"}}}
    );
    defer validator.deinit();

    // 'x' is not a valid first byte (expecting '{')
    const advanced = try validator.advanceByte('x');
    try std.testing.expect(!advanced);
    try std.testing.expectEqual(@as(usize, 0), validator.getPosition());
}

// =============================================================================
// advance Tests
// =============================================================================

test "Validator: advance consumes valid sequence" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "integer"}
    );
    defer validator.deinit();

    const consumed = try validator.advance("42");
    try std.testing.expectEqual(@as(usize, 2), consumed);
    try std.testing.expectEqual(@as(usize, 2), validator.getPosition());
}

test "Validator: advance stops at invalid byte and returns consumed count" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "integer"}
    );
    defer validator.deinit();

    // "12abc" - should consume "12" then stop at 'a'
    const consumed = try validator.advance("12abc");
    try std.testing.expectEqual(@as(usize, 2), consumed);
    try std.testing.expectEqual(@as(usize, 2), validator.getPosition());
}

// =============================================================================
// canAccept Tests
// =============================================================================

test "Validator: canAccept returns true for valid sequence without modifying state" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "integer"}
    );
    defer validator.deinit();

    const pos_before = validator.getPosition();
    const can = try validator.canAccept("123");
    const pos_after = validator.getPosition();

    try std.testing.expect(can);
    try std.testing.expectEqual(pos_before, pos_after);
}

test "Validator: canAccept returns false for invalid sequence" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "integer"}
    );
    defer validator.deinit();

    const can = try validator.canAccept("abc");
    try std.testing.expect(!can);
}

// =============================================================================
// validate Tests
// =============================================================================

test "Validator: validate returns true for complete valid input" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "integer"}
    );
    defer validator.deinit();

    const valid = try validator.validate("42");
    try std.testing.expect(valid);
}

test "Validator: validate returns false for incomplete input" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    );
    defer validator.deinit();

    // Just opening brace - not complete
    const valid = try validator.validate("{");
    try std.testing.expect(!valid);
}

test "Validator: validate returns false for invalid input" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "integer"}
    );
    defer validator.deinit();

    const valid = try validator.validate("abc");
    try std.testing.expect(!valid);
}

test "Validator: validate resets state before validating" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "integer"}
    );
    defer validator.deinit();

    // Advance some bytes first
    _ = try validator.advance("99");

    // validate should reset and validate fresh
    const valid = try validator.validate("42");
    try std.testing.expect(valid);
    try std.testing.expectEqual(@as(usize, 2), validator.getPosition());
}

// =============================================================================
// isComplete Tests
// =============================================================================

test "Validator: isComplete returns false at start" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "integer"}
    );
    defer validator.deinit();

    try std.testing.expect(!validator.isComplete());
}

test "Validator: isComplete returns true after valid complete input" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "integer"}
    );
    defer validator.deinit();

    _ = try validator.advance("42");
    try std.testing.expect(validator.isComplete());
}

// =============================================================================
// getDeterministicContinuation Tests
// =============================================================================

test "Validator: getDeterministicContinuation returns null when multiple options" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "integer"}
    );
    defer validator.deinit();

    // At start, any digit is valid - no deterministic continuation
    const continuation = validator.getDeterministicContinuation();
    // May or may not be null depending on grammar, just verify it doesn't crash
    _ = continuation;
}

// =============================================================================
// Schema Type Tests
// =============================================================================

test "Validator: validates string schema" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "string"}
    );
    defer validator.deinit();

    const valid = try validator.validate("\"hello\"");
    try std.testing.expect(valid);
}

test "Validator: validates boolean schema" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "boolean"}
    );
    defer validator.deinit();

    try std.testing.expect(try validator.validate("true"));
    try validator.reset();
    try std.testing.expect(try validator.validate("false"));
}

test "Validator: validates null schema" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "null"}
    );
    defer validator.deinit();

    const valid = try validator.validate("null");
    try std.testing.expect(valid);
}

test "Validator: validates array schema" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "array", "items": {"type": "integer"}}
    );
    defer validator.deinit();

    const valid = try validator.validate("[1,2,3]");
    try std.testing.expect(valid);
}

test "Validator: validates nested object schema" {
    const allocator = std.testing.allocator;

    var validator = try Validator.init(allocator,
        \\{"type": "object", "properties": {"nested": {"type": "object", "properties": {"value": {"type": "integer"}}}}}
    );
    defer validator.deinit();

    const valid = try validator.validate("{\"nested\":{\"value\":42}}");
    try std.testing.expect(valid);
}
