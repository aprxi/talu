/**
 * Lightweight JSON Schema validator for tool argument validation.
 *
 * Validates against the JsonSchema subset used by ToolDefinition.parameters:
 * type, properties, required, items, enum.
 */

import type { JsonSchema } from "../types.ts";

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

export function validateArgs(schema: JsonSchema, value: unknown): ValidationResult {
  const errors: string[] = [];
  validate(schema, value, "", errors);
  return { valid: errors.length === 0, errors };
}

function validate(schema: JsonSchema, value: unknown, path: string, errors: string[]): void {
  const label = path || "root";

  if (!checkType(schema.type, value)) {
    errors.push(`${label}: expected type "${schema.type}", got ${typeLabel(value)}`);
    return;
  }

  if (schema.enum && !schema.enum.some((v) => v === value)) {
    errors.push(`${label}: value not in enum`);
  }

  if (schema.type === "object" && schema.properties && typeof value === "object" && value !== null) {
    const obj = value as Record<string, unknown>;

    if (schema.required) {
      for (const key of schema.required) {
        if (!(key in obj)) {
          errors.push(`${path ? path + "." : ""}${key}: required property missing`);
        }
      }
    }

    for (const [key, propSchema] of Object.entries(schema.properties)) {
      if (key in obj) {
        validate(propSchema, obj[key], path ? `${path}.${key}` : key, errors);
      }
    }
  }

  if (schema.type === "array" && schema.items && Array.isArray(value)) {
    for (let i = 0; i < value.length; i++) {
      validate(schema.items, value[i], `${path}[${i}]`, errors);
    }
  }
}

function checkType(type: string, value: unknown): boolean {
  switch (type) {
    case "string": return typeof value === "string";
    case "number": return typeof value === "number" && !Number.isNaN(value);
    case "integer": return typeof value === "number" && Number.isInteger(value);
    case "boolean": return typeof value === "boolean";
    case "object": return typeof value === "object" && value !== null && !Array.isArray(value);
    case "array": return Array.isArray(value);
    case "null": return value === null;
    default: return true;
  }
}

function typeLabel(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  return typeof value;
}
