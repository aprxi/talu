//! JSON Schema parser and grammar compiler.

const std = @import("std");
const io = @import("../io/root.zig");
const ast = @import("ast.zig");
const generic = @import("generic.zig");
const regex = @import("regex.zig");

const Grammar = ast.Grammar;
const RuleId = ast.RuleId;

pub const SchemaError = error{
    InvalidSchema,
    UnsupportedFeature,
    RecursionDepthExceeded,
    OutOfMemory,
};

pub const CompilerConfig = struct {
    max_exact_span: i64 = 1000,
    max_exact_value: i64 = 100_000,
    max_depth: usize = 32,
};

pub fn compile(
    allocator: std.mem.Allocator,
    schema_json: []const u8,
    config: CompilerConfig,
) SchemaError!Grammar {
    var grammar = Grammar.init(allocator);
    errdefer grammar.deinit();

    const parsed = io.json.parseValue(allocator, schema_json, .{ .max_size_bytes = 1 * 1024 * 1024 }) catch |err| {
        return switch (err) {
            error.InputTooLarge => SchemaError.InvalidSchema,
            error.InputTooDeep => SchemaError.InvalidSchema,
            error.StringTooLong => SchemaError.InvalidSchema,
            error.InvalidJson => SchemaError.InvalidSchema,
            error.OutOfMemory => SchemaError.OutOfMemory,
        };
    };
    defer parsed.deinit();

    grammar.ws_rule = try ast.JsonRules.whitespace(&grammar);

    var compiler = Compiler.init(&grammar, config);
    defer compiler.deinit();

    if (parsed.value == .object) {
        const root_obj = parsed.value.object;
        if (root_obj.get("$defs")) |defs| {
            try compiler.compileDefs(defs);
        }
    }

    grammar.root_rule = try compiler.compileValue(parsed.value);

    return grammar;
}

const Compiler = struct {
    grammar: *Grammar,
    config: CompilerConfig,
    depth: usize,
    def_rules: std.StringHashMap(RuleId),
    compiling_defs: std.StringHashMap(void),
    generic_value_rule: ?RuleId,

    fn init(grammar: *Grammar, config: CompilerConfig) Compiler {
        return .{
            .grammar = grammar,
            .config = config,
            .depth = 0,
            .def_rules = std.StringHashMap(RuleId).init(grammar.allocator()),
            .compiling_defs = std.StringHashMap(void).init(grammar.allocator()),
            .generic_value_rule = null,
        };
    }

    /// Get or create a generic JSON value rule that accepts any valid JSON.
    /// Used when additionalProperties is not false - semantic validation handles constraints.
    fn getGenericValueRule(self: *Compiler) SchemaError!RuleId {
        if (self.generic_value_rule) |rule| return rule;

        // Create a full generic JSON grammar
        const rule = generic.compileGeneric(self.grammar) catch return SchemaError.OutOfMemory;
        self.generic_value_rule = rule;
        return rule;
    }

    /// Compile a generic JSON object grammar where values can be any valid JSON.
    /// Used when additionalProperties is not false - accepts any property names.
    fn compileGenericObject(self: *Compiler) SchemaError!RuleId {
        const ws = self.grammar.ws_rule;
        const value_rule = try self.getGenericValueRule();

        const comma = self.grammar.addRule(.{ .char = ',' }) catch return SchemaError.OutOfMemory;
        const colon = self.grammar.addRule(.{ .char = ':' }) catch return SchemaError.OutOfMemory;
        const open = self.grammar.addRule(.{ .char = '{' }) catch return SchemaError.OutOfMemory;
        const close = self.grammar.addRule(.{ .char = '}' }) catch return SchemaError.OutOfMemory;

        const key = ast.JsonRules.string(self.grammar) catch return SchemaError.OutOfMemory;

        // Build: { ws ( key ws : ws value ( ws , ws key ws : ws value )* )? ws }
        const pair_ids = [_]RuleId{ key, ws, colon, ws, value_rule };
        const pair = self.grammar.addRule(.{
            .sequence = ast.dupeRuleIds(self.grammar.allocator(), &pair_ids) catch return SchemaError.OutOfMemory,
        }) catch return SchemaError.OutOfMemory;

        const tail_ids = [_]RuleId{ ws, comma, ws, pair };
        const tail = self.grammar.addRule(.{
            .sequence = ast.dupeRuleIds(self.grammar.allocator(), &tail_ids) catch return SchemaError.OutOfMemory,
        }) catch return SchemaError.OutOfMemory;
        const tail_star = self.grammar.addRule(.{ .star = tail }) catch return SchemaError.OutOfMemory;

        const pairs_ids = [_]RuleId{ pair, tail_star };
        const pairs = self.grammar.addRule(.{
            .sequence = ast.dupeRuleIds(self.grammar.allocator(), &pairs_ids) catch return SchemaError.OutOfMemory,
        }) catch return SchemaError.OutOfMemory;
        const opt_pairs = self.grammar.addRule(.{ .optional = pairs }) catch return SchemaError.OutOfMemory;

        const object_ids = [_]RuleId{ open, ws, opt_pairs, ws, close };
        return self.grammar.addRule(.{
            .sequence = ast.dupeRuleIds(self.grammar.allocator(), &object_ids) catch return SchemaError.OutOfMemory,
        }) catch return SchemaError.OutOfMemory;
    }

    fn deinit(self: *Compiler) void {
        self.def_rules.deinit();
        self.compiling_defs.deinit();
    }

    fn compileDefs(self: *Compiler, defs: std.json.Value) SchemaError!void {
        const defs_obj = switch (defs) {
            .object => |obj| obj,
            else => return SchemaError.InvalidSchema,
        };

        var iter = defs_obj.iterator();
        while (iter.next()) |entry| {
            const def_name = entry.key_ptr.*;
            const placeholder = try self.grammar.addRule(.{ .end = {} });
            try self.def_rules.put(def_name, placeholder);
        }

        iter = defs_obj.iterator();
        while (iter.next()) |entry| {
            const def_name = entry.key_ptr.*;
            const def_schema = entry.value_ptr.*;

            try self.compiling_defs.put(def_name, {});

            const rule_id = try self.compileValue(def_schema);
            if (self.def_rules.get(def_name)) |placeholder| {
                self.grammar.rules.items[placeholder] = self.grammar.rules.items[rule_id];
            }

            _ = self.compiling_defs.remove(def_name);
        }
    }

    fn compileValue(self: *Compiler, value: std.json.Value) SchemaError!RuleId {
        if (self.depth > self.config.max_depth) {
            return SchemaError.RecursionDepthExceeded;
        }
        self.depth += 1;
        defer self.depth -= 1;

        return switch (value) {
            .object => |obj| self.compileObject(obj),
            else => SchemaError.InvalidSchema,
        };
    }

    fn compileObject(self: *Compiler, obj: std.json.ObjectMap) SchemaError!RuleId {
        if (obj.get("$ref")) |ref| {
            return self.compileRef(ref);
        }

        if (obj.get("const")) |const_val| {
            if (try self.compileConstLiteral(obj, const_val)) |literal_rule| {
                return literal_rule;
            }
            return self.compileConst(const_val);
        }

        if (obj.get("type")) |type_val| {
            return self.compileType(type_val, obj);
        }

        if (obj.get("enum")) |enum_val| {
            return self.compileEnum(enum_val);
        }

        if (obj.get("anyOf")) |any_of| {
            return self.compileAnyOf(any_of);
        }
        if (obj.get("oneOf")) |one_of| {
            return self.compileAnyOf(one_of);
        }

        return SchemaError.InvalidSchema;
    }

    fn compileConst(self: *Compiler, value: std.json.Value) SchemaError!RuleId {
        const literal = switch (value) {
            .string => |s| std.fmt.allocPrint(self.grammar.allocator(), "\"{s}\"", .{s}) catch
                return SchemaError.OutOfMemory,
            .integer => |i| std.fmt.allocPrint(self.grammar.allocator(), "{d}", .{i}) catch
                return SchemaError.OutOfMemory,
            .float => |f| blk: {
                if (f == 0.0 and std.math.signbit(f)) {
                    break :blk self.grammar.allocator().dupe(u8, "-0.0") catch
                        return SchemaError.OutOfMemory;
                }
                break :blk std.fmt.allocPrint(self.grammar.allocator(), "{}", .{f}) catch
                    return SchemaError.OutOfMemory;
            },
            .bool => |b| std.fmt.allocPrint(self.grammar.allocator(), "{s}", .{if (b) "true" else "false"}) catch
                return SchemaError.OutOfMemory,
            .null => try self.grammar.allocator().dupe(u8, "null"),
            else => return SchemaError.UnsupportedFeature,
        };

        return self.grammar.addRule(.{ .literal = literal });
    }

    fn compileConstLiteral(
        self: *Compiler,
        obj: std.json.ObjectMap,
        value: std.json.Value,
    ) SchemaError!?RuleId {
        if (value != .integer and value != .float) return null;
        const literal_value = obj.get("x-talu-const-literal") orelse return null;
        const literal = switch (literal_value) {
            .string => |s| s,
            else => return null,
        };
        const owned = self.grammar.allocator().dupe(u8, literal) catch
            return SchemaError.OutOfMemory;
        return try self.grammar.addRule(.{ .literal = owned });
    }

    fn compileRef(self: *Compiler, ref: std.json.Value) SchemaError!RuleId {
        const ref_str = switch (ref) {
            .string => |s| s,
            else => return SchemaError.InvalidSchema,
        };

        if (std.mem.startsWith(u8, ref_str, "#/$defs/")) {
            const def_name = ref_str[8..];

            if (self.def_rules.get(def_name)) |rule_id| {
                if (self.compiling_defs.contains(def_name)) {
                    return self.grammar.addRule(.{ .reference = rule_id });
                }
                return rule_id;
            }
        }

        return SchemaError.InvalidSchema;
    }

    fn compileType(
        self: *Compiler,
        type_val: std.json.Value,
        obj: std.json.ObjectMap,
    ) SchemaError!RuleId {
        const type_str = switch (type_val) {
            .string => |s| s,
            else => return SchemaError.InvalidSchema,
        };

        if (std.mem.eql(u8, type_str, "string")) {
            return self.compileString(obj);
        } else if (std.mem.eql(u8, type_str, "number")) {
            return self.compileNumber(obj, false);
        } else if (std.mem.eql(u8, type_str, "integer")) {
            return self.compileNumber(obj, true);
        } else if (std.mem.eql(u8, type_str, "boolean")) {
            return ast.JsonRules.boolean(self.grammar);
        } else if (std.mem.eql(u8, type_str, "null")) {
            return ast.JsonRules.null_(self.grammar);
        } else if (std.mem.eql(u8, type_str, "array")) {
            return self.compileArray(obj);
        } else if (std.mem.eql(u8, type_str, "object")) {
            return self.compileObjectType(obj);
        } else if (std.mem.eql(u8, type_str, "json_object")) {
            return generic.compileGeneric(self.grammar);
        }

        return SchemaError.UnsupportedFeature;
    }

    fn compileString(self: *Compiler, obj: std.json.ObjectMap) SchemaError!RuleId {
        if (obj.get("enum")) |enum_val| {
            return self.compileEnum(enum_val);
        }

        if (obj.get("format")) |format_val| {
            const format_str = switch (format_val) {
                .string => |s| s,
                else => return SchemaError.InvalidSchema,
            };
            if (regex.Format.fromString(format_str)) |format| {
                return regex.compileFormat(self.grammar, format);
            }
        }

        if (obj.get("pattern")) |_| {
            return SchemaError.UnsupportedFeature;
        }

        // Check for length constraints (minLength/maxLength)
        const min_length = if (obj.get("minLength")) |v| switch (v) {
            .integer => |i| if (i >= 0) @as(usize, @intCast(i)) else null,
            else => null,
        } else null;

        const max_length = if (obj.get("maxLength")) |v| switch (v) {
            .integer => |i| if (i >= 0) @as(usize, @intCast(i)) else null,
            else => null,
        } else null;

        // If we have length constraints, use bounded string
        if (min_length != null or max_length != null) {
            const min = min_length orelse 0;
            // Enforce reasonable max to prevent grammar explosion
            // If maxLength > 1000, fall back to unbounded (grammar would be too large)
            const max: ?usize = if (max_length) |m| (if (m <= self.config.max_exact_span) m else null) else null;
            return ast.JsonRules.stringBounded(self.grammar, min, max);
        }

        return ast.JsonRules.string(self.grammar);
    }

    fn compileNumber(
        self: *Compiler,
        obj: std.json.ObjectMap,
        is_integer: bool,
    ) SchemaError!RuleId {
        // Check for enum constraint (e.g., Pydantic integer Literal/Enum)
        if (obj.get("enum")) |enum_val| {
            return self.compileEnum(enum_val);
        }

        const min = if (obj.get("minimum")) |v| switch (v) {
            .integer => |i| i,
            .float => |f| @as(i64, @intFromFloat(f)),
            else => null,
        } else null;

        const max = if (obj.get("maximum")) |v| switch (v) {
            .integer => |i| i,
            .float => |f| @as(i64, @intFromFloat(f)),
            else => null,
        } else null;

        if (is_integer and min != null and max != null) {
            const span = max.? - min.?;
            if (span >= 0 and span <= self.config.max_exact_span and
                @abs(min.?) <= self.config.max_exact_value and
                @abs(max.?) <= self.config.max_exact_value)
            {
                return self.compileIntegerRange(min.?, max.?);
            }
        }

        // Use strict integer grammar (no decimal/exponent) for integer type
        if (is_integer) {
            return ast.JsonRules.integer(self.grammar);
        }

        return ast.JsonRules.number(self.grammar);
    }

    fn compileIntegerRange(self: *Compiler, min: i64, max: i64) SchemaError!RuleId {
        var alternatives = std.ArrayList(RuleId).empty;
        defer alternatives.deinit(self.grammar.allocator());

        var value = min;
        while (value <= max) : (value += 1) {
            const literal = std.fmt.allocPrint(self.grammar.allocator(), "{d}", .{value}) catch
                return SchemaError.OutOfMemory;
            const rule = try self.grammar.addRule(.{ .literal = literal });
            try alternatives.append(self.grammar.allocator(), rule);
        }

        return self.grammar.addRule(.{
            .alternatives = try alternatives.toOwnedSlice(self.grammar.allocator()),
        });
    }

    fn compileEnum(self: *Compiler, enum_val: std.json.Value) SchemaError!RuleId {
        const arr = switch (enum_val) {
            .array => |a| a,
            else => return SchemaError.InvalidSchema,
        };

        var alternatives = std.ArrayList(RuleId).empty;
        defer alternatives.deinit(self.grammar.allocator());

        for (arr.items) |item| {
            const literal = switch (item) {
                .string => |s| std.fmt.allocPrint(self.grammar.allocator(), "\"{s}\"", .{s}) catch
                    return SchemaError.OutOfMemory,
                .integer => |i| std.fmt.allocPrint(self.grammar.allocator(), "{d}", .{i}) catch
                    return SchemaError.OutOfMemory,
                .float => |f| std.fmt.allocPrint(self.grammar.allocator(), "{}", .{f}) catch
                    return SchemaError.OutOfMemory,
                .bool => |b| std.fmt.allocPrint(self.grammar.allocator(), "{s}", .{if (b) "true" else "false"}) catch
                    return SchemaError.OutOfMemory,
                .null => std.fmt.allocPrint(self.grammar.allocator(), "null", .{}) catch
                    return SchemaError.OutOfMemory,
                else => continue,
            };

            const rule = try self.grammar.addRule(.{ .literal = literal });
            try alternatives.append(self.grammar.allocator(), rule);
        }

        return self.grammar.addRule(.{
            .alternatives = try alternatives.toOwnedSlice(self.grammar.allocator()),
        });
    }

    fn compileAnyOf(self: *Compiler, any_of: std.json.Value) SchemaError!RuleId {
        const arr = switch (any_of) {
            .array => |a| a,
            else => return SchemaError.InvalidSchema,
        };
        var alternatives = std.ArrayList(RuleId).empty;
        defer alternatives.deinit(self.grammar.allocator());
        for (arr.items) |item| {
            const rule_id = try self.compileValue(item);
            try alternatives.append(self.grammar.allocator(), rule_id);
        }
        return self.grammar.addRule(.{
            .alternatives = try alternatives.toOwnedSlice(self.grammar.allocator()),
        });
    }

    fn compileArray(self: *Compiler, obj: std.json.ObjectMap) SchemaError!RuleId {
        const items = obj.get("items") orelse return SchemaError.InvalidSchema;
        const item_rule = try self.compileValue(items);

        const ws = self.grammar.ws_rule;
        const comma = try self.grammar.addRule(.{ .char = ',' });
        const open = try self.grammar.addRule(.{ .char = '[' });
        const close = try self.grammar.addRule(.{ .char = ']' });

        // Check for item count constraints (minItems/maxItems)
        const min_items = if (obj.get("minItems")) |v| switch (v) {
            .integer => |i| if (i >= 0) @as(usize, @intCast(i)) else null,
            else => null,
        } else null;

        const max_items = if (obj.get("maxItems")) |v| switch (v) {
            .integer => |i| if (i >= 0) @as(usize, @intCast(i)) else null,
            else => null,
        } else null;

        const min = min_items orelse 0;
        // Enforce reasonable max to prevent grammar explosion (use same limit as string)
        const max: ?usize = if (max_items) |m| (if (m <= self.config.max_exact_span) m else null) else null;

        // comma_item = ws , ws item
        const comma_item_ids = [_]RuleId{ ws, comma, ws, item_rule };
        const comma_item = try self.grammar.addRule(.{
            .sequence = try ast.dupeRuleIds(self.grammar.allocator(), &comma_item_ids),
        });

        // Build array grammar based on constraints:
        // [ ws content ws ] where content depends on min/max
        // Note: Using arena allocator, no need to deinit - arena owns all memory
        var seq = std.ArrayList(RuleId).empty;

        try seq.append(self.grammar.allocator(), open);
        try seq.append(self.grammar.allocator(), ws);

        if (min == 0 and max == null) {
            // Unbounded: [ ws (item (, item)*)? ws ]
            const more_items = try self.grammar.addRule(.{ .star = comma_item });
            const items_list_ids = [_]RuleId{ item_rule, more_items };
            const items_list = try self.grammar.addRule(.{
                .sequence = try ast.dupeRuleIds(self.grammar.allocator(), &items_list_ids),
            });
            const opt_items = try self.grammar.addRule(.{ .optional = items_list });
            try seq.append(self.grammar.allocator(), opt_items);
        } else if (min == 0 and max != null) {
            // maxItems only: [ ws (item (, item){0,max-1})? ws ]
            if (max.? == 0) {
                // Empty array only: [ ws ]
            } else {
                // First item + up to (max-1) optional comma_items
                var content = std.ArrayList(RuleId).empty;

                try content.append(self.grammar.allocator(), item_rule);
                for (0..(max.? - 1)) |_| {
                    const opt_comma_item = try self.grammar.addRule(.{ .optional = comma_item });
                    try content.append(self.grammar.allocator(), opt_comma_item);
                }

                const content_seq = try self.grammar.addRule(.{
                    .sequence = try content.toOwnedSlice(self.grammar.allocator()),
                });
                const opt_content = try self.grammar.addRule(.{ .optional = content_seq });
                try seq.append(self.grammar.allocator(), opt_content);
            }
        } else if (min > 0 and max == null) {
            // minItems only: [ ws item (, item){min-1,} ws ]
            // First item required
            try seq.append(self.grammar.allocator(), item_rule);
            // min-1 more required items
            for (0..(min - 1)) |_| {
                try seq.append(self.grammar.allocator(), comma_item);
            }
            // Unbounded additional items
            const more_items = try self.grammar.addRule(.{ .star = comma_item });
            try seq.append(self.grammar.allocator(), more_items);
        } else {
            // Both minItems and maxItems: [ ws item (, item){min-1} (, item){0,max-min}? ws ]
            // First item required (if min > 0)
            try seq.append(self.grammar.allocator(), item_rule);
            // min-1 more required items
            for (0..(min - 1)) |_| {
                try seq.append(self.grammar.allocator(), comma_item);
            }
            // (max - min) optional additional items
            for (0..(max.? - min)) |_| {
                const opt_comma_item = try self.grammar.addRule(.{ .optional = comma_item });
                try seq.append(self.grammar.allocator(), opt_comma_item);
            }
        }

        try seq.append(self.grammar.allocator(), ws);
        try seq.append(self.grammar.allocator(), close);

        return self.grammar.addRule(.{
            .sequence = try seq.toOwnedSlice(self.grammar.allocator()),
        });
    }

    fn compileObjectType(self: *Compiler, obj: std.json.ObjectMap) SchemaError!RuleId {
        // Check if additionalProperties is explicitly false
        const additional_props = obj.get("additionalProperties");
        const is_closed = if (additional_props) |ap| switch (ap) {
            .bool => |b| !b,
            else => false,
        } else false;

        // If additionalProperties is NOT false (i.e., additional properties allowed),
        // use generic JSON object that accepts any property names with any JSON values.
        // Semantic validation will handle type checking, required, and additionalProperties constraints.
        if (!is_closed) {
            return self.compileGenericObject();
        }

        // additionalProperties: false - use strict grammar with only defined properties
        const properties = obj.get("properties") orelse
            return SchemaError.InvalidSchema;

        const props_obj = switch (properties) {
            .object => |o| o,
            else => return SchemaError.InvalidSchema,
        };

        // Get required properties list
        var required_set = std.StringHashMap(void).init(self.grammar.allocator());
        defer required_set.deinit();

        if (obj.get("required")) |req| {
            const req_arr = switch (req) {
                .array => |a| a,
                else => return SchemaError.InvalidSchema,
            };
            for (req_arr.items) |item| {
                const name = switch (item) {
                    .string => |s| s,
                    else => continue,
                };
                try required_set.put(name, {});
            }
        }

        const ws = self.grammar.ws_rule;
        const open = try self.grammar.addRule(.{ .char = '{' });
        const close = try self.grammar.addRule(.{ .char = '}' });
        const comma = try self.grammar.addRule(.{ .char = ',' });

        // Collect required and optional properties separately
        var required_props = std.ArrayList(struct { name: []const u8, rule: RuleId }).empty;
        defer required_props.deinit(self.grammar.allocator());

        var optional_props = std.ArrayList(RuleId).empty;
        defer optional_props.deinit(self.grammar.allocator());

        var iter = props_obj.iterator();
        while (iter.next()) |entry| {
            const prop_name = entry.key_ptr.*;
            const prop_schema = entry.value_ptr.*;

            const value_rule = try self.compileValue(prop_schema);

            const quoted = std.fmt.allocPrint(self.grammar.allocator(), "\"{s}\"", .{prop_name}) catch
                return SchemaError.OutOfMemory;
            const key_rule = try self.grammar.addRule(.{ .literal = quoted });
            const colon = try self.grammar.addRule(.{ .char = ':' });

            const prop_rule = try self.grammar.addRule(.{
                .sequence = try ast.dupeRuleIds(
                    self.grammar.allocator(),
                    &[_]RuleId{ key_rule, ws, colon, ws, value_rule },
                ),
            });

            if (required_set.contains(prop_name)) {
                try required_props.append(self.grammar.allocator(), .{ .name = prop_name, .rule = prop_rule });
            } else {
                try optional_props.append(self.grammar.allocator(), prop_rule);
            }
        }

        // Build the object grammar:
        // Case 1: No required, no optional -> {}
        // Case 2: Has required, no optional -> { req1, req2, ... }
        // Case 3: No required, has optional -> { } | { opt1 (, opt2)* }
        // Case 4: Has required, has optional -> { req1, req2, ... (, opt1)* }
        var full_seq = std.ArrayList(RuleId).empty;
        defer full_seq.deinit(self.grammar.allocator());

        try full_seq.append(self.grammar.allocator(), open);
        try full_seq.append(self.grammar.allocator(), ws);

        // Add required properties in sequence
        for (required_props.items, 0..) |prop, i| {
            if (i > 0) {
                try full_seq.append(self.grammar.allocator(), comma);
                try full_seq.append(self.grammar.allocator(), ws);
            }
            try full_seq.append(self.grammar.allocator(), prop.rule);
            try full_seq.append(self.grammar.allocator(), ws);
        }

        // Handle optional properties
        if (optional_props.items.len > 0) {
            if (required_props.items.len > 0) {
                // Has required: optional props are (, ws prop)*
                for (optional_props.items) |opt_prop| {
                    const comma_prop = try self.grammar.addRule(.{
                        .sequence = try ast.dupeRuleIds(
                            self.grammar.allocator(),
                            &[_]RuleId{ comma, ws, opt_prop, ws },
                        ),
                    });
                    const opt_comma_prop = try self.grammar.addRule(.{ .optional = comma_prop });
                    try full_seq.append(self.grammar.allocator(), opt_comma_prop);
                }
            } else {
                // No required: first optional has no comma, rest have comma
                // Build: (opt1 ws (, ws opt2 ws)* )?
                var opt_content = std.ArrayList(RuleId).empty;
                defer opt_content.deinit(self.grammar.allocator());

                // First optional (no leading comma)
                try opt_content.append(self.grammar.allocator(), optional_props.items[0]);
                try opt_content.append(self.grammar.allocator(), ws);

                // Remaining optionals with leading comma
                for (optional_props.items[1..]) |opt_prop| {
                    const comma_prop = try self.grammar.addRule(.{
                        .sequence = try ast.dupeRuleIds(
                            self.grammar.allocator(),
                            &[_]RuleId{ comma, ws, opt_prop, ws },
                        ),
                    });
                    const opt_comma_prop = try self.grammar.addRule(.{ .optional = comma_prop });
                    try opt_content.append(self.grammar.allocator(), opt_comma_prop);
                }

                const opt_seq = try self.grammar.addRule(.{
                    .sequence = try opt_content.toOwnedSlice(self.grammar.allocator()),
                });
                const opt_all = try self.grammar.addRule(.{ .optional = opt_seq });
                try full_seq.append(self.grammar.allocator(), opt_all);
            }
        }

        try full_seq.append(self.grammar.allocator(), close);

        return self.grammar.addRule(.{
            .sequence = try full_seq.toOwnedSlice(self.grammar.allocator()),
        });
    }
};

// Tests

test "compile simple object schema" {
    const allocator = std.testing.allocator;

    const schema =
        \\{
        \\  "type": "object",
        \\  "properties": {
        \\    "name": {"type": "string"},
        \\    "age": {"type": "integer"}
        \\  },
        \\  "required": ["name"]
        \\}
    ;

    var grammar = try compile(allocator, schema, .{});
    defer grammar.deinit();

    try std.testing.expect(grammar.root_rule != ast.INVALID_RULE);
}

test "compile recursive schema (Comment with replies)" {
    const allocator = std.testing.allocator;

    const schema =
        \\{
        \\  "type": "object",
        \\  "properties": {
        \\    "text": {"type": "string"},
        \\    "replies": {
        \\      "type": "array",
        \\      "items": {"$ref": "#/$defs/Comment"}
        \\    }
        \\  },
        \\  "required": ["text"],
        \\  "$defs": {
        \\    "Comment": {
        \\      "type": "object",
        \\      "properties": {
        \\        "text": {"type": "string"},
        \\        "replies": {
        \\          "type": "array",
        \\          "items": {"$ref": "#/$defs/Comment"}
        \\        }
        \\      },
        \\      "required": ["text"]
        \\    }
        \\  }
        \\}
    ;

    var grammar = try compile(allocator, schema, .{});
    defer grammar.deinit();

    try std.testing.expect(grammar.root_rule != ast.INVALID_RULE);

    var has_reference = false;
    for (grammar.rules.items) |rule| {
        if (rule == .reference) {
            has_reference = true;
            break;
        }
    }
    try std.testing.expect(has_reference);
}

test "compile linked list (Node with next)" {
    const allocator = std.testing.allocator;

    const schema =
        \\{
        \\  "$defs": {
        \\    "Node": {
        \\      "type": "object",
        \\      "properties": {
        \\        "value": {"type": "integer"},
        \\        "next": {
        \\          "anyOf": [
        \\            {"$ref": "#/$defs/Node"},
        \\            {"type": "null"}
        \\          ]
        \\        }
        \\      },
        \\      "required": ["value"]
        \\    }
        \\  },
        \\  "$ref": "#/$defs/Node"
        \\}
    ;

    var grammar = try compile(allocator, schema, .{});
    defer grammar.deinit();

    try std.testing.expect(grammar.root_rule != ast.INVALID_RULE);
}

test "compile string with minLength" {
    const allocator = std.testing.allocator;

    const schema =
        \\{
        \\  "type": "object",
        \\  "properties": {
        \\    "name": {"type": "string", "minLength": 1}
        \\  },
        \\  "required": ["name"]
        \\}
    ;

    var grammar = try compile(allocator, schema, .{});
    defer grammar.deinit();

    try std.testing.expect(grammar.root_rule != ast.INVALID_RULE);
}

test "compile string with maxLength" {
    const allocator = std.testing.allocator;

    const schema =
        \\{
        \\  "type": "object",
        \\  "properties": {
        \\    "name": {"type": "string", "maxLength": 50}
        \\  },
        \\  "required": ["name"]
        \\}
    ;

    var grammar = try compile(allocator, schema, .{});
    defer grammar.deinit();

    try std.testing.expect(grammar.root_rule != ast.INVALID_RULE);
}

test "compile string with minLength and maxLength" {
    const allocator = std.testing.allocator;

    const schema =
        \\{
        \\  "type": "object",
        \\  "properties": {
        \\    "name": {"type": "string", "minLength": 1, "maxLength": 50}
        \\  },
        \\  "required": ["name"]
        \\}
    ;

    var grammar = try compile(allocator, schema, .{});
    defer grammar.deinit();

    try std.testing.expect(grammar.root_rule != ast.INVALID_RULE);
}

test "compile array with minItems" {
    const allocator = std.testing.allocator;

    const schema =
        \\{
        \\  "type": "object",
        \\  "properties": {
        \\    "items": {"type": "array", "items": {"type": "string"}, "minItems": 1}
        \\  },
        \\  "required": ["items"]
        \\}
    ;

    var grammar = try compile(allocator, schema, .{});
    defer grammar.deinit();

    try std.testing.expect(grammar.root_rule != ast.INVALID_RULE);
}

test "compile array with maxItems" {
    const allocator = std.testing.allocator;

    const schema =
        \\{
        \\  "type": "object",
        \\  "properties": {
        \\    "items": {"type": "array", "items": {"type": "string"}, "maxItems": 10}
        \\  },
        \\  "required": ["items"]
        \\}
    ;

    var grammar = try compile(allocator, schema, .{});
    defer grammar.deinit();

    try std.testing.expect(grammar.root_rule != ast.INVALID_RULE);
}

test "compile array with minItems and maxItems" {
    const allocator = std.testing.allocator;

    const schema =
        \\{
        \\  "type": "object",
        \\  "properties": {
        \\    "items": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 10}
        \\  },
        \\  "required": ["items"]
        \\}
    ;

    var grammar = try compile(allocator, schema, .{});
    defer grammar.deinit();

    try std.testing.expect(grammar.root_rule != ast.INVALID_RULE);
}

test "compile unbounded array (no minItems/maxItems)" {
    const allocator = std.testing.allocator;

    // This is the common case - arrays without length constraints
    // Must produce the same grammar as before the minItems/maxItems feature
    const schema =
        \\{
        \\  "type": "object",
        \\  "properties": {
        \\    "tags": {"type": "array", "items": {"type": "string"}}
        \\  },
        \\  "required": ["tags"]
        \\}
    ;

    var grammar = try compile(allocator, schema, .{});
    defer grammar.deinit();

    try std.testing.expect(grammar.root_rule != ast.INVALID_RULE);

    // The grammar should have a star rule for unbounded repetition
    var has_star = false;
    for (grammar.rules.items) |rule| {
        if (rule == .star) {
            has_star = true;
            break;
        }
    }
    try std.testing.expect(has_star);
}

test "compile unbounded string (no minLength/maxLength)" {
    const allocator = std.testing.allocator;

    // This is the common case - strings without length constraints
    const schema =
        \\{
        \\  "type": "object",
        \\  "properties": {
        \\    "name": {"type": "string"}
        \\  },
        \\  "required": ["name"]
        \\}
    ;

    var grammar = try compile(allocator, schema, .{});
    defer grammar.deinit();

    try std.testing.expect(grammar.root_rule != ast.INVALID_RULE);

    // The grammar should have a star rule for unbounded string chars
    var has_star = false;
    for (grammar.rules.items) |rule| {
        if (rule == .star) {
            has_star = true;
            break;
        }
    }
    try std.testing.expect(has_star);
}
