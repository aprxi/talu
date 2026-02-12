import { describe, test, expect } from "bun:test";
import { validateArgs } from "../../../src/kernel/core/schema-validator.ts";

describe("validateArgs", () => {
  // --- Basic type checking ---

  test("valid string → OK", () => {
    const result = validateArgs({ type: "string" }, "hello");
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  test("wrong type → error", () => {
    const result = validateArgs({ type: "string" }, 42);
    expect(result.valid).toBe(false);
    expect(result.errors[0]).toContain("string");
  });

  test("number type accepts numbers", () => {
    expect(validateArgs({ type: "number" }, 3.14).valid).toBe(true);
  });

  test("number type rejects NaN", () => {
    expect(validateArgs({ type: "number" }, NaN).valid).toBe(false);
  });

  test("integer type accepts integers", () => {
    expect(validateArgs({ type: "integer" }, 42).valid).toBe(true);
  });

  test("integer type rejects floats", () => {
    const result = validateArgs({ type: "integer" }, 3.5);
    expect(result.valid).toBe(false);
    expect(result.errors[0]).toContain("integer");
  });

  test("boolean type", () => {
    expect(validateArgs({ type: "boolean" }, true).valid).toBe(true);
    expect(validateArgs({ type: "boolean" }, "true").valid).toBe(false);
  });

  test("null type", () => {
    expect(validateArgs({ type: "null" }, null).valid).toBe(true);
    expect(validateArgs({ type: "null" }, undefined).valid).toBe(false);
  });

  test("array type", () => {
    expect(validateArgs({ type: "array" }, [1, 2]).valid).toBe(true);
    expect(validateArgs({ type: "array" }, "not array").valid).toBe(false);
  });

  test("object type rejects arrays and null", () => {
    expect(validateArgs({ type: "object" }, {}).valid).toBe(true);
    expect(validateArgs({ type: "object" }, []).valid).toBe(false);
    expect(validateArgs({ type: "object" }, null).valid).toBe(false);
  });

  test("unknown schema type → permissive (OK)", () => {
    expect(validateArgs({ type: "custom" }, "anything").valid).toBe(true);
  });

  // --- Enum ---

  test("enum match → OK", () => {
    const result = validateArgs({ type: "string", enum: ["a", "b", "c"] }, "b");
    expect(result.valid).toBe(true);
  });

  test("enum mismatch → error", () => {
    const result = validateArgs({ type: "string", enum: ["a", "b", "c"] }, "d");
    expect(result.valid).toBe(false);
    expect(result.errors[0]).toContain("enum");
  });

  // --- Required fields ---

  test("required field missing → error", () => {
    const result = validateArgs(
      { type: "object", properties: { name: { type: "string" } }, required: ["name"] },
      {},
    );
    expect(result.valid).toBe(false);
    expect(result.errors[0]).toContain("required");
  });

  test("required field present → OK", () => {
    const result = validateArgs(
      { type: "object", properties: { name: { type: "string" } }, required: ["name"] },
      { name: "Alice" },
    );
    expect(result.valid).toBe(true);
  });

  // --- Nested objects ---

  test("nested object validation", () => {
    const schema = {
      type: "object",
      properties: {
        config: {
          type: "object",
          properties: {
            enabled: { type: "boolean" },
          },
        },
      },
    };

    expect(validateArgs(schema, { config: { enabled: true } }).valid).toBe(true);
    expect(validateArgs(schema, { config: { enabled: "yes" } }).valid).toBe(false);
  });

  // --- Array items ---

  test("array items validation", () => {
    const schema = { type: "array", items: { type: "number" } };

    expect(validateArgs(schema, [1, 2, 3]).valid).toBe(true);

    const result = validateArgs(schema, [1, "two", 3]);
    expect(result.valid).toBe(false);
    expect(result.errors[0]).toContain("[1]");
  });

  // --- Combined ---

  test("multiple errors accumulated", () => {
    const schema = {
      type: "object",
      properties: {
        name: { type: "string" },
        age: { type: "integer" },
      },
      required: ["name", "age"],
    };

    const result = validateArgs(schema, {});
    expect(result.valid).toBe(false);
    expect(result.errors.length).toBeGreaterThanOrEqual(2);
  });

  test("optional fields not validated when missing", () => {
    const schema = {
      type: "object",
      properties: {
        name: { type: "string" },
        optional: { type: "number" },
      },
      required: ["name"],
    };

    const result = validateArgs(schema, { name: "Alice" });
    expect(result.valid).toBe(true);
  });
});
