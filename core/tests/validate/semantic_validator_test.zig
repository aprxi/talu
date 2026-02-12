//! Integration tests for SemanticValidator
//!
//! SemanticValidator validates JSON against semantic constraints that grammar
//! cannot express: number ranges, additionalProperties, and type mismatches.

const std = @import("std");
const main = @import("main");
const SemanticValidator = main.validate.SemanticValidator;
const SemanticViolation = main.validate.SemanticViolation;

// =============================================================================
// Lifecycle Tests
// =============================================================================

test "SemanticValidator: init and deinit from valid schema" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{"type": "object", "properties": {"age": {"type": "integer", "minimum": 0}}}
    );
    defer validator.deinit();
}

test "SemanticValidator: init returns error for invalid JSON" {
    const allocator = std.testing.allocator;

    const result = SemanticValidator.init(allocator, "not valid json");
    try std.testing.expectError(error.InvalidSchema, result);
}

test "SemanticValidator: deinit is idempotent-safe" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{"type": "integer"}
    );
    validator.deinit();
    // After deinit, fields are undefined - no second deinit attempted
}

// =============================================================================
// Number Validation Tests
// =============================================================================

test "SemanticValidator: accepts number within range" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{"type": "number", "minimum": 0, "maximum": 100}
    );
    defer validator.deinit();

    const result = try validator.validate("50");
    try std.testing.expectEqual(@as(?SemanticViolation, null), result);
}

test "SemanticValidator: rejects number below minimum" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{"type": "number", "minimum": 0}
    );
    defer validator.deinit();

    const result = try validator.validate("-5");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_minimum, result.?.constraint_type);
}

test "SemanticValidator: rejects number above maximum" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{"type": "number", "maximum": 100}
    );
    defer validator.deinit();

    const result = try validator.validate("150");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_maximum, result.?.constraint_type);
}

test "SemanticValidator: rejects number at exclusiveMinimum boundary" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{"type": "number", "exclusiveMinimum": 0}
    );
    defer validator.deinit();

    const result = try validator.validate("0");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_exclusive_minimum, result.?.constraint_type);
}

test "SemanticValidator: accepts number above exclusiveMinimum" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{"type": "number", "exclusiveMinimum": 0}
    );
    defer validator.deinit();

    const result = try validator.validate("0.001");
    try std.testing.expectEqual(@as(?SemanticViolation, null), result);
}

test "SemanticValidator: rejects number at exclusiveMaximum boundary" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{"type": "number", "exclusiveMaximum": 100}
    );
    defer validator.deinit();

    const result = try validator.validate("100");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_exclusive_maximum, result.?.constraint_type);
}

test "SemanticValidator: validates integer type with range" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{"type": "integer", "minimum": 1, "maximum": 10}
    );
    defer validator.deinit();

    const valid_result = try validator.validate("5");
    try std.testing.expectEqual(@as(?SemanticViolation, null), valid_result);

    const invalid_result = try validator.validate("15");
    try std.testing.expect(invalid_result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_maximum, invalid_result.?.constraint_type);
}

// =============================================================================
// Object Validation Tests
// =============================================================================

test "SemanticValidator: rejects additional properties when false" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{
        \\  "type": "object",
        \\  "properties": {"name": {"type": "string"}},
        \\  "additionalProperties": false
        \\}
    );
    defer validator.deinit();

    const result = try validator.validate(
        \\{"name": "Alice", "extra": "value"}
    );
    try std.testing.expect(result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.additional_properties, result.?.constraint_type);
}

test "SemanticValidator: accepts object with only allowed properties" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{
        \\  "type": "object",
        \\  "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        \\  "additionalProperties": false
        \\}
    );
    defer validator.deinit();

    const result = try validator.validate(
        \\{"name": "Bob", "age": 30}
    );
    try std.testing.expectEqual(@as(?SemanticViolation, null), result);
}

test "SemanticValidator: allows additional properties by default" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{
        \\  "type": "object",
        \\  "properties": {"name": {"type": "string"}}
        \\}
    );
    defer validator.deinit();

    const result = try validator.validate(
        \\{"name": "Charlie", "extra": "allowed"}
    );
    try std.testing.expectEqual(@as(?SemanticViolation, null), result);
}

test "SemanticValidator: validates additionalProperties with schema" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{
        \\  "type": "object",
        \\  "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1}
        \\}
    );
    defer validator.deinit();

    const valid_result = try validator.validate(
        \\{"score": 0.8, "confidence": 0.9}
    );
    try std.testing.expectEqual(@as(?SemanticViolation, null), valid_result);

    const invalid_result = try validator.validate(
        \\{"score": 1.5}
    );
    try std.testing.expect(invalid_result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_maximum, invalid_result.?.constraint_type);
}

test "SemanticValidator: rejects missing required property" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{
        \\  "type": "object",
        \\  "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
        \\  "required": ["id", "name"]
        \\}
    );
    defer validator.deinit();

    const result = try validator.validate(
        \\{"id": 1}
    );
    try std.testing.expect(result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.required_property, result.?.constraint_type);
}

// =============================================================================
// Nested Object Tests
// =============================================================================

test "SemanticValidator: validates nested object properties" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{
        \\  "type": "object",
        \\  "properties": {
        \\    "person": {
        \\      "type": "object",
        \\      "properties": {
        \\        "age": {"type": "integer", "minimum": 0, "maximum": 120}
        \\      }
        \\    }
        \\  }
        \\}
    );
    defer validator.deinit();

    const valid_result = try validator.validate(
        \\{"person": {"age": 25}}
    );
    try std.testing.expectEqual(@as(?SemanticViolation, null), valid_result);

    const invalid_result = try validator.validate(
        \\{"person": {"age": 200}}
    );
    try std.testing.expect(invalid_result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_maximum, invalid_result.?.constraint_type);
}

// =============================================================================
// Array Validation Tests
// =============================================================================

test "SemanticValidator: validates array items" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{
        \\  "type": "array",
        \\  "items": {"type": "number", "minimum": 0, "maximum": 10}
        \\}
    );
    defer validator.deinit();

    const valid_result = try validator.validate("[1, 5, 10]");
    try std.testing.expectEqual(@as(?SemanticViolation, null), valid_result);

    const invalid_result = try validator.validate("[1, 5, 15]");
    try std.testing.expect(invalid_result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_maximum, invalid_result.?.constraint_type);
}

test "SemanticValidator: validates nested arrays" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{
        \\  "type": "array",
        \\  "items": {
        \\    "type": "array",
        \\    "items": {"type": "integer", "minimum": 0}
        \\  }
        \\}
    );
    defer validator.deinit();

    const valid_result = try validator.validate("[[1, 2], [3, 4]]");
    try std.testing.expectEqual(@as(?SemanticViolation, null), valid_result);

    const invalid_result = try validator.validate("[[1, -2], [3, 4]]");
    try std.testing.expect(invalid_result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_minimum, invalid_result.?.constraint_type);
}

// =============================================================================
// Type Mismatch Tests
// =============================================================================

test "SemanticValidator: detects type mismatch for number" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{"type": "number"}
    );
    defer validator.deinit();

    const result = try validator.validate("\"not a number\"");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.type_mismatch, result.?.constraint_type);
}

test "SemanticValidator: detects type mismatch for string" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{"type": "string"}
    );
    defer validator.deinit();

    const result = try validator.validate("123");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.type_mismatch, result.?.constraint_type);
}

test "SemanticValidator: detects type mismatch for object" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{"type": "object"}
    );
    defer validator.deinit();

    const result = try validator.validate("[1, 2, 3]");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.type_mismatch, result.?.constraint_type);
}

// =============================================================================
// $ref Resolution Tests
// =============================================================================

test "SemanticValidator: resolves $ref to $defs" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{
        \\  "type": "object",
        \\  "properties": {
        \\    "score": {"$ref": "#/$defs/Score"}
        \\  },
        \\  "$defs": {
        \\    "Score": {"type": "number", "minimum": 0, "maximum": 100}
        \\  }
        \\}
    );
    defer validator.deinit();

    const valid_result = try validator.validate(
        \\{"score": 85}
    );
    try std.testing.expectEqual(@as(?SemanticViolation, null), valid_result);

    const invalid_result = try validator.validate(
        \\{"score": 150}
    );
    try std.testing.expect(invalid_result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_maximum, invalid_result.?.constraint_type);
}

// =============================================================================
// anyOf/oneOf Tests
// =============================================================================

test "SemanticValidator: validates anyOf - passes if one matches" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{
        \\  "anyOf": [
        \\    {"type": "integer", "minimum": 0},
        \\    {"type": "string"}
        \\  ]
        \\}
    );
    defer validator.deinit();

    // Integer in range should pass
    const int_result = try validator.validate("5");
    try std.testing.expectEqual(@as(?SemanticViolation, null), int_result);

    // String should pass
    const str_result = try validator.validate("\"hello\"");
    try std.testing.expectEqual(@as(?SemanticViolation, null), str_result);
}

// =============================================================================
// Error Message Tests
// =============================================================================

test "SemanticValidator: violation includes path" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{
        \\  "type": "object",
        \\  "properties": {
        \\    "data": {
        \\      "type": "object",
        \\      "properties": {
        \\        "value": {"type": "number", "maximum": 10}
        \\      }
        \\    }
        \\  }
        \\}
    );
    defer validator.deinit();

    const result = try validator.validate(
        \\{"data": {"value": 100}}
    );
    try std.testing.expect(result != null);
    try std.testing.expect(std.mem.indexOf(u8, result.?.path, "data") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.?.path, "value") != null);
}

test "SemanticValidator: validate returns error for invalid JSON input" {
    const allocator = std.testing.allocator;

    var validator = try SemanticValidator.init(allocator,
        \\{"type": "integer"}
    );
    defer validator.deinit();

    const result = validator.validate("not valid json");
    try std.testing.expectError(error.InvalidJson, result);
}
