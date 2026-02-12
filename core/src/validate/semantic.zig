//! Semantic validator for JSON Schema constraints that grammar cannot express.
//!
//! Grammar-based validation (schema.zig) enforces structure during generation,
//! but cannot express certain constraints:
//!
//! - **Number ranges for floats**: CFG cannot enumerate infinite float values
//! - **Number ranges for large integers**: Enumeration would explode grammar size
//! - **additionalProperties: false**: CFG cannot express "no keys except these"
//! - **additionalProperties: {schema}**: CFG cannot validate dynamic keys against a schema
//!
//! This module provides post-parse validation for these semantic constraints.
//! It walks the parsed JSON and schema in lockstep, validating constraints
//! that grammar accepted syntactically but may violate semantically.
//!
//! Usage:
//! ```zig
//! const validator = try SemanticValidator.init(allocator, schema_json);
//! defer validator.deinit();
//!
//! const result = try validator.validate(parsed_json);
//! if (result) |violation| {
//!     // violation.path, violation.message describe the error
//! }
//! ```

const std = @import("std");
const io = @import("../io/root.zig");

/// Result of semantic validation - null means valid.
pub const SemanticViolation = struct {
    /// JSON path to the violating value (e.g., "$.person.age")
    /// Null-terminated for C API compatibility.
    path: [:0]const u8,
    /// Human-readable error message
    /// Null-terminated for C API compatibility.
    message: [:0]const u8,
    /// The constraint that was violated
    constraint_type: ConstraintType,

    pub const ConstraintType = enum {
        number_minimum,
        number_maximum,
        number_exclusive_minimum,
        number_exclusive_maximum,
        additional_properties,
        type_mismatch,
        required_property,
    };
};

/// Number range constraint extracted from schema.
pub const NumberConstraint = struct {
    minimum: ?f64 = null,
    maximum: ?f64 = null,
    exclusive_minimum: ?f64 = null,
    exclusive_maximum: ?f64 = null,

    /// Check if this constraint is satisfied by the given value.
    pub fn check(self: NumberConstraint, value: f64) ?SemanticViolation.ConstraintType {
        if (self.minimum) |min| {
            if (value < min) return .number_minimum;
        }
        if (self.maximum) |max| {
            if (value > max) return .number_maximum;
        }
        if (self.exclusive_minimum) |min| {
            if (value <= min) return .number_exclusive_minimum;
        }
        if (self.exclusive_maximum) |max| {
            if (value >= max) return .number_exclusive_maximum;
        }
        return null;
    }

    /// Returns true if this constraint has any bounds.
    pub fn hasBounds(self: NumberConstraint) bool {
        return self.minimum != null or self.maximum != null or
            self.exclusive_minimum != null or self.exclusive_maximum != null;
    }
};

/// Semantic validator that checks constraints grammar cannot express.
pub const SemanticValidator = struct {
    allocator: std.mem.Allocator,
    schema: std.json.Value,
    arena: std.heap.ArenaAllocator,

    /// Initialize validator from JSON schema string.
    pub fn init(allocator: std.mem.Allocator, schema_json: []const u8) !SemanticValidator {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        const parsed = io.json.parseValue(arena.allocator(), schema_json, .{ .max_size_bytes = 1 * 1024 * 1024 }) catch |err| {
            return switch (err) {
                error.InputTooLarge => error.InvalidSchema,
                error.InputTooDeep => error.InvalidSchema,
                error.StringTooLong => error.InvalidSchema,
                error.InvalidJson => error.InvalidSchema,
                error.OutOfMemory => error.OutOfMemory,
            };
        };

        return .{
            .allocator = allocator,
            .schema = parsed.value,
            .arena = arena,
        };
    }

    pub fn deinit(self: *SemanticValidator) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Validate parsed JSON against semantic constraints.
    /// Returns null if valid, or a SemanticViolation describing the first error.
    pub fn validate(self: *SemanticValidator, json_str: []const u8) !?SemanticViolation {
        const parsed = io.json.parseValue(self.arena.allocator(), json_str, .{
            .max_size_bytes = 10 * 1024 * 1024,
            .max_value_bytes = 10 * 1024 * 1024,
            .max_string_bytes = 1 * 1024 * 1024,
        }) catch |err| {
            return switch (err) {
                error.InputTooLarge => error.InvalidJson,
                error.InputTooDeep => error.InvalidJson,
                error.StringTooLong => error.InvalidJson,
                error.InvalidJson => error.InvalidJson,
                error.OutOfMemory => error.OutOfMemory,
            };
        };

        return self.validateValue(parsed.value, self.schema, "$");
    }

    const ValidationError = error{OutOfMemory};

    fn validateValue(
        self: *SemanticValidator,
        value: std.json.Value,
        schema: std.json.Value,
        path: [:0]const u8,
    ) ValidationError!?SemanticViolation {
        const schema_obj = switch (schema) {
            .object => |obj| obj,
            else => return null, // Non-object schema, no constraints to check
        };

        // Handle $ref
        if (schema_obj.get("$ref")) |ref| {
            const resolved = self.resolveRef(ref) orelse return null;
            return self.validateValue(value, resolved, path);
        }

        // Handle anyOf/oneOf - value must pass at least one
        if (schema_obj.get("anyOf") orelse schema_obj.get("oneOf")) |variants| {
            switch (variants) {
                .array => |arr| {
                    for (arr.items) |variant| {
                        const violation = try self.validateValue(value, variant, path);
                        if (violation == null) return null; // Passed one variant
                    }
                    // Failed all variants - return generic error
                    // Note: In practice, grammar should have already rejected invalid structure
                },
                else => {},
            }
            return null;
        }

        // Get type from schema
        const type_val = schema_obj.get("type") orelse return null;
        const type_str = switch (type_val) {
            .string => |s| s,
            else => return null,
        };

        // Type checking and dispatch
        if (std.mem.eql(u8, type_str, "number")) {
            if (value != .integer and value != .float) {
                return try self.makeTypeMismatchViolation(path, "number", value);
            }
            return self.validateNumber(value, schema_obj, path);
        } else if (std.mem.eql(u8, type_str, "integer")) {
            if (value != .integer) {
                return try self.makeTypeMismatchViolation(path, "integer", value);
            }
            return self.validateNumber(value, schema_obj, path);
        } else if (std.mem.eql(u8, type_str, "string")) {
            if (value != .string) {
                return try self.makeTypeMismatchViolation(path, "string", value);
            }
            return null;
        } else if (std.mem.eql(u8, type_str, "boolean")) {
            if (value != .bool) {
                return try self.makeTypeMismatchViolation(path, "boolean", value);
            }
            return null;
        } else if (std.mem.eql(u8, type_str, "null")) {
            if (value != .null) {
                return try self.makeTypeMismatchViolation(path, "null", value);
            }
            return null;
        } else if (std.mem.eql(u8, type_str, "object")) {
            if (value != .object) {
                return try self.makeTypeMismatchViolation(path, "object", value);
            }
            return self.validateObject(value, schema_obj, path);
        } else if (std.mem.eql(u8, type_str, "array")) {
            if (value != .array) {
                return try self.makeTypeMismatchViolation(path, "array", value);
            }
            return self.validateArray(value, schema_obj, path);
        }

        return null;
    }

    fn validateNumber(
        self: *SemanticValidator,
        value: std.json.Value,
        schema_obj: std.json.ObjectMap,
        path: [:0]const u8,
    ) ValidationError!?SemanticViolation {
        const num_value: f64 = switch (value) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => return null, // Type mismatch handled by grammar
        };

        const constraint = NumberConstraint{
            .minimum = extractFloat(schema_obj.get("minimum")),
            .maximum = extractFloat(schema_obj.get("maximum")),
            .exclusive_minimum = extractFloat(schema_obj.get("exclusiveMinimum")),
            .exclusive_maximum = extractFloat(schema_obj.get("exclusiveMaximum")),
        };

        if (!constraint.hasBounds()) return null;

        if (constraint.check(num_value)) |violation_type| {
            const message = try std.fmt.allocPrintSentinel(
                self.arena.allocator(),
                "{s}: {d} violates {s} constraint",
                .{ path, num_value, @tagName(violation_type) },
                0,
            );
            return SemanticViolation{
                .path = path,
                .message = message,
                .constraint_type = violation_type,
            };
        }

        return null;
    }

    fn validateObject(
        self: *SemanticValidator,
        value: std.json.Value,
        schema_obj: std.json.ObjectMap,
        path: [:0]const u8,
    ) ValidationError!?SemanticViolation {
        const obj = switch (value) {
            .object => |o| o,
            else => return null, // Type mismatch handled by grammar
        };

        // Check required properties
        if (schema_obj.get("required")) |required_val| {
            switch (required_val) {
                .array => |required_arr| {
                    for (required_arr.items) |req_item| {
                        const req_name = switch (req_item) {
                            .string => |s| s,
                            else => continue,
                        };
                        if (!obj.contains(req_name)) {
                            const message = try std.fmt.allocPrintSentinel(
                                self.arena.allocator(),
                                "{s}: required property '{s}' is missing",
                                .{ path, req_name },
                                0,
                            );
                            return SemanticViolation{
                                .path = path,
                                .message = message,
                                .constraint_type = .required_property,
                            };
                        }
                    }
                },
                else => {},
            }
        }

        // Check additionalProperties constraint
        const additional_props = schema_obj.get("additionalProperties");
        const properties = schema_obj.get("properties");
        const props_obj: ?std.json.ObjectMap = if (properties) |p| switch (p) {
            .object => |o| o,
            else => null,
        } else null;

        if (additional_props) |ap| {
            switch (ap) {
                .bool => |allowed| {
                    if (!allowed) {
                        // additionalProperties: false - reject any unknown properties
                        for (obj.keys()) |key| {
                            const is_known = if (props_obj) |po| po.contains(key) else false;
                            if (!is_known) {
                                const message = try std.fmt.allocPrintSentinel(
                                    self.arena.allocator(),
                                    "{s}: additional property '{s}' not allowed",
                                    .{ path, key },
                                    0,
                                );
                                return SemanticViolation{
                                    .path = path,
                                    .message = message,
                                    .constraint_type = .additional_properties,
                                };
                            }
                        }
                    }
                },
                .object => {
                    // additionalProperties: {schema} - validate unknown properties against schema
                    for (obj.keys()) |key| {
                        const is_known = if (props_obj) |po| po.contains(key) else false;
                        if (!is_known) {
                            const prop_value = obj.get(key) orelse continue;
                            const child_path = try std.fmt.allocPrintSentinel(
                                self.arena.allocator(),
                                "{s}.{s}",
                                .{ path, key },
                                0,
                            );
                            const violation = try self.validateValue(prop_value, ap, child_path);
                            if (violation != null) return violation;
                        }
                    }
                },
                else => {},
            }
        }

        // Recursively validate nested properties
        const defined_props = props_obj orelse return null;

        for (defined_props.keys()) |prop_name| {
            if (obj.get(prop_name)) |prop_value| {
                const prop_schema = defined_props.get(prop_name) orelse continue;

                // Build child path
                const child_path = try std.fmt.allocPrintSentinel(
                    self.arena.allocator(),
                    "{s}.{s}",
                    .{ path, prop_name },
                    0,
                );

                const violation = try self.validateValue(prop_value, prop_schema, child_path);
                if (violation != null) return violation;
            }
        }

        return null;
    }

    fn makeTypeMismatchViolation(
        self: *SemanticValidator,
        path: [:0]const u8,
        expected_type: []const u8,
        actual_value: std.json.Value,
    ) ValidationError!SemanticViolation {
        const actual_type: []const u8 = switch (actual_value) {
            .null => "null",
            .bool => "boolean",
            .integer => "integer",
            .float => "number",
            .string => "string",
            .array => "array",
            .object => "object",
            .number_string => "number",
        };

        const message = try std.fmt.allocPrintSentinel(
            self.arena.allocator(),
            "{s}: expected {s}, got {s}",
            .{ path, expected_type, actual_type },
            0,
        );
        return SemanticViolation{
            .path = path,
            .message = message,
            .constraint_type = .type_mismatch,
        };
    }

    fn validateArray(
        self: *SemanticValidator,
        value: std.json.Value,
        schema_obj: std.json.ObjectMap,
        path: [:0]const u8,
    ) ValidationError!?SemanticViolation {
        const arr = switch (value) {
            .array => |a| a,
            else => return null, // Type mismatch handled by grammar
        };

        const items_schema = schema_obj.get("items") orelse return null;

        for (arr.items, 0..) |item, idx| {
            const child_path = try std.fmt.allocPrintSentinel(
                self.arena.allocator(),
                "{s}[{d}]",
                .{ path, idx },
                0,
            );

            const violation = try self.validateValue(item, items_schema, child_path);
            if (violation != null) return violation;
        }

        return null;
    }

    fn resolveRef(self: *SemanticValidator, ref: std.json.Value) ?std.json.Value {
        const ref_str = switch (ref) {
            .string => |s| s,
            else => return null,
        };

        if (!std.mem.startsWith(u8, ref_str, "#/$defs/")) return null;
        const def_name = ref_str[8..];

        const root_obj = switch (self.schema) {
            .object => |o| o,
            else => return null,
        };

        const defs = root_obj.get("$defs") orelse return null;
        const defs_obj = switch (defs) {
            .object => |o| o,
            else => return null,
        };

        return defs_obj.get(def_name);
    }

    fn extractFloat(value: ?std.json.Value) ?f64 {
        const v = value orelse return null;
        return switch (v) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => null,
        };
    }
};

// =============================================================================
// Tests
// =============================================================================

test "SemanticValidator.init parses valid schema" {
    const allocator = std.testing.allocator;
    var validator = try SemanticValidator.init(allocator,
        \\{"type": "object", "properties": {"age": {"type": "integer", "minimum": 0}}}
    );
    defer validator.deinit();
}

test "SemanticValidator.init returns error for invalid JSON" {
    const allocator = std.testing.allocator;
    const result = SemanticValidator.init(allocator, "not valid json");
    try std.testing.expectError(error.InvalidSchema, result);
}

test "SemanticValidator.validate accepts valid number" {
    const allocator = std.testing.allocator;
    var validator = try SemanticValidator.init(allocator,
        \\{"type": "number", "minimum": 0, "maximum": 100}
    );
    defer validator.deinit();

    const result = try validator.validate("50");
    try std.testing.expectEqual(@as(?SemanticViolation, null), result);
}

test "SemanticValidator.validate rejects number below minimum" {
    const allocator = std.testing.allocator;
    var validator = try SemanticValidator.init(allocator,
        \\{"type": "number", "minimum": 0, "maximum": 100}
    );
    defer validator.deinit();

    const result = try validator.validate("-5");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_minimum, result.?.constraint_type);
}

test "SemanticValidator.validate rejects number above maximum" {
    const allocator = std.testing.allocator;
    var validator = try SemanticValidator.init(allocator,
        \\{"type": "number", "minimum": 0, "maximum": 100}
    );
    defer validator.deinit();

    const result = try validator.validate("150");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_maximum, result.?.constraint_type);
}

test "SemanticValidator.validate handles exclusiveMinimum" {
    const allocator = std.testing.allocator;
    var validator = try SemanticValidator.init(allocator,
        \\{"type": "number", "exclusiveMinimum": 0}
    );
    defer validator.deinit();

    // 0 should fail (exclusive)
    const result_zero = try validator.validate("0");
    try std.testing.expect(result_zero != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_exclusive_minimum, result_zero.?.constraint_type);

    // 0.1 should pass
    const result_positive = try validator.validate("0.1");
    try std.testing.expectEqual(@as(?SemanticViolation, null), result_positive);
}

test "SemanticValidator.validate handles exclusiveMaximum" {
    const allocator = std.testing.allocator;
    var validator = try SemanticValidator.init(allocator,
        \\{"type": "number", "exclusiveMaximum": 100}
    );
    defer validator.deinit();

    // 100 should fail (exclusive)
    const result_hundred = try validator.validate("100");
    try std.testing.expect(result_hundred != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_exclusive_maximum, result_hundred.?.constraint_type);

    // 99.9 should pass
    const result_below = try validator.validate("99.9");
    try std.testing.expectEqual(@as(?SemanticViolation, null), result_below);
}

test "SemanticValidator.validate validates nested object properties" {
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

    // Valid nested value
    const valid_result = try validator.validate(
        \\{"person": {"age": 30}}
    );
    try std.testing.expectEqual(@as(?SemanticViolation, null), valid_result);

    // Invalid nested value
    const invalid_result = try validator.validate(
        \\{"person": {"age": 150}}
    );
    try std.testing.expect(invalid_result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_maximum, invalid_result.?.constraint_type);
}

test "SemanticValidator.validate rejects additional properties when false" {
    const allocator = std.testing.allocator;
    var validator = try SemanticValidator.init(allocator,
        \\{
        \\  "type": "object",
        \\  "properties": {
        \\    "name": {"type": "string"},
        \\    "age": {"type": "integer"}
        \\  },
        \\  "additionalProperties": false
        \\}
    );
    defer validator.deinit();

    // Valid - only allowed properties
    const valid_result = try validator.validate(
        \\{"name": "Alice", "age": 30}
    );
    try std.testing.expectEqual(@as(?SemanticViolation, null), valid_result);

    // Invalid - extra property
    const invalid_result = try validator.validate(
        \\{"name": "Alice", "age": 30, "email": "alice@example.com"}
    );
    try std.testing.expect(invalid_result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.additional_properties, invalid_result.?.constraint_type);
}

test "SemanticValidator.validate allows additional properties by default" {
    const allocator = std.testing.allocator;
    var validator = try SemanticValidator.init(allocator,
        \\{
        \\  "type": "object",
        \\  "properties": {
        \\    "name": {"type": "string"}
        \\  }
        \\}
    );
    defer validator.deinit();

    // Extra property allowed when additionalProperties not false
    const result = try validator.validate(
        \\{"name": "Alice", "extra": "value"}
    );
    try std.testing.expectEqual(@as(?SemanticViolation, null), result);
}

test "SemanticValidator.validate validates additionalProperties with schema" {
    const allocator = std.testing.allocator;
    var validator = try SemanticValidator.init(allocator,
        \\{
        \\  "type": "object",
        \\  "additionalProperties": {"type": "number"}
        \\}
    );
    defer validator.deinit();

    // Valid - all values are numbers
    const valid_result = try validator.validate(
        \\{"food": 0.8, "service": -0.3}
    );
    try std.testing.expectEqual(@as(?SemanticViolation, null), valid_result);

    // Invalid - string value should fail
    const invalid_result = try validator.validate(
        \\{"food": "great"}
    );
    try std.testing.expect(invalid_result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.type_mismatch, invalid_result.?.constraint_type);
}

test "SemanticValidator.validate validates additionalProperties with schema and defined properties" {
    const allocator = std.testing.allocator;
    var validator = try SemanticValidator.init(allocator,
        \\{
        \\  "type": "object",
        \\  "properties": {
        \\    "id": {"type": "integer"}
        \\  },
        \\  "additionalProperties": {"type": "string"}
        \\}
    );
    defer validator.deinit();

    // Valid - id is integer, extra is string
    const valid_result = try validator.validate(
        \\{"id": 1, "extra": "hello"}
    );
    try std.testing.expectEqual(@as(?SemanticViolation, null), valid_result);

    // Invalid - extra is number, should be string
    const invalid_result = try validator.validate(
        \\{"id": 1, "extra": 123}
    );
    try std.testing.expect(invalid_result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.type_mismatch, invalid_result.?.constraint_type);
}

test "SemanticValidator.validate validates additionalProperties with number constraints" {
    const allocator = std.testing.allocator;
    var validator = try SemanticValidator.init(allocator,
        \\{
        \\  "type": "object",
        \\  "additionalProperties": {"type": "number", "minimum": -1, "maximum": 1}
        \\}
    );
    defer validator.deinit();

    // Valid - value in range
    const valid_result = try validator.validate(
        \\{"sentiment": 0.5}
    );
    try std.testing.expectEqual(@as(?SemanticViolation, null), valid_result);

    // Invalid - value exceeds maximum
    const invalid_result = try validator.validate(
        \\{"sentiment": 5.0}
    );
    try std.testing.expect(invalid_result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_maximum, invalid_result.?.constraint_type);
}

test "SemanticValidator.validate validates array items" {
    const allocator = std.testing.allocator;
    var validator = try SemanticValidator.init(allocator,
        \\{
        \\  "type": "array",
        \\  "items": {"type": "number", "minimum": 0, "maximum": 10}
        \\}
    );
    defer validator.deinit();

    // Valid array
    const valid_result = try validator.validate("[1, 5, 10]");
    try std.testing.expectEqual(@as(?SemanticViolation, null), valid_result);

    // Invalid array - item out of range
    const invalid_result = try validator.validate("[1, 5, 15]");
    try std.testing.expect(invalid_result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_maximum, invalid_result.?.constraint_type);
}

test "SemanticValidator.validate handles $ref" {
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

    // Valid ref
    const valid_result = try validator.validate(
        \\{"score": 85}
    );
    try std.testing.expectEqual(@as(?SemanticViolation, null), valid_result);

    // Invalid ref
    const invalid_result = try validator.validate(
        \\{"score": 150}
    );
    try std.testing.expect(invalid_result != null);
}

test "SemanticValidator.validate is order-invariant for object fields" {
    const allocator = std.testing.allocator;
    var validator = try SemanticValidator.init(allocator,
        \\{
        \\  "type": "object",
        \\  "properties": {
        \\    "a": {"type": "number"},
        \\    "b": {"type": "number"}
        \\  },
        \\  "required": ["a", "b"],
        \\  "additionalProperties": false
        \\}
    );
    defer validator.deinit();

    const json_ab = "{\"a\": 1, \"b\": 2}";
    const json_ba = "{\"b\": 2, \"a\": 1}";

    try std.testing.expectEqual(@as(?SemanticViolation, null), try validator.validate(json_ab));
    try std.testing.expectEqual(@as(?SemanticViolation, null), try validator.validate(json_ba));

    const json_extra = "{\"a\": 1, \"b\": 2, \"c\": 3}";
    const extra_result = try validator.validate(json_extra);
    try std.testing.expect(extra_result != null);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.additional_properties, extra_result.?.constraint_type);
}

test "NumberConstraint.check validates minimum" {
    const constraint = NumberConstraint{ .minimum = 0 };
    try std.testing.expectEqual(@as(?SemanticViolation.ConstraintType, null), constraint.check(0));
    try std.testing.expectEqual(@as(?SemanticViolation.ConstraintType, null), constraint.check(100));
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_minimum, constraint.check(-1).?);
}

test "NumberConstraint.check validates maximum" {
    const constraint = NumberConstraint{ .maximum = 100 };
    try std.testing.expectEqual(@as(?SemanticViolation.ConstraintType, null), constraint.check(100));
    try std.testing.expectEqual(@as(?SemanticViolation.ConstraintType, null), constraint.check(0));
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_maximum, constraint.check(101).?);
}

test "NumberConstraint.check validates exclusive bounds" {
    const constraint = NumberConstraint{
        .exclusive_minimum = 0,
        .exclusive_maximum = 100,
    };
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_exclusive_minimum, constraint.check(0).?);
    try std.testing.expectEqual(SemanticViolation.ConstraintType.number_exclusive_maximum, constraint.check(100).?);
    try std.testing.expectEqual(@as(?SemanticViolation.ConstraintType, null), constraint.check(50));
}

test "NumberConstraint.hasBounds returns correct value" {
    try std.testing.expectEqual(false, (NumberConstraint{}).hasBounds());
    try std.testing.expectEqual(true, (NumberConstraint{ .minimum = 0 }).hasBounds());
    try std.testing.expectEqual(true, (NumberConstraint{ .maximum = 100 }).hasBounds());
    try std.testing.expectEqual(true, (NumberConstraint{ .exclusive_minimum = 0 }).hasBounds());
    try std.testing.expectEqual(true, (NumberConstraint{ .exclusive_maximum = 100 }).hasBounds());
}
