import { describe, test, expect } from "bun:test";
import {
  validateManifest,
  sanitizeManifestString,
  KERNEL_API_VERSION,
  KNOWN_PERMISSIONS,
} from "../../../src/kernel/core/manifest-validator.ts";
import type { PluginManifest } from "../../../src/kernel/types.ts";

/** Minimal valid builtin manifest. */
function builtinManifest(overrides: Partial<PluginManifest> = {}): PluginManifest {
  return {
    id: "talu.test",
    name: "Test",
    version: "1.0.0",
    builtin: true,
    ...overrides,
  } as PluginManifest;
}

/** Minimal valid third-party manifest. */
function thirdPartyManifest(overrides: Partial<PluginManifest> = {}): PluginManifest {
  return {
    id: "com.example.test",
    name: "Test",
    version: "1.0.0",
    apiVersion: KERNEL_API_VERSION,
    ...overrides,
  } as PluginManifest;
}

describe("validateManifest", () => {
  // --- Valid manifests ---

  test("valid builtin manifest passes", () => {
    const result = validateManifest(builtinManifest());
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  test("valid third-party manifest passes", () => {
    const result = validateManifest(thirdPartyManifest());
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  // --- Required fields ---

  test("missing id → error", () => {
    const result = validateManifest(builtinManifest({ id: "" }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("id"))).toBe(true);
  });

  test("missing name → error", () => {
    const result = validateManifest(builtinManifest({ name: "" }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("name"))).toBe(true);
  });

  test("missing version → error", () => {
    const result = validateManifest(builtinManifest({ version: "" }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("version"))).toBe(true);
  });

  // --- ID format ---

  test("uppercase in id → error", () => {
    const result = validateManifest(builtinManifest({ id: "Talu.Test" }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("reverse-domain"))).toBe(true);
  });

  test("spaces in id → error", () => {
    const result = validateManifest(builtinManifest({ id: "talu test" }));
    expect(result.valid).toBe(false);
  });

  test("leading digit in id → error", () => {
    const result = validateManifest(builtinManifest({ id: "1plugin" }));
    expect(result.valid).toBe(false);
  });

  test("single-segment lowercase id → valid", () => {
    const result = validateManifest(builtinManifest({ id: "myplugin" }));
    expect(result.valid).toBe(true);
  });

  test("multi-segment reverse-domain id → valid", () => {
    const result = validateManifest(builtinManifest({ id: "com.example.plugin" }));
    expect(result.valid).toBe(true);
  });

  // --- Version format ---

  test("non-semver version → error", () => {
    const result = validateManifest(builtinManifest({ version: "v1.0" }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("semver"))).toBe(true);
  });

  test("valid semver version → OK", () => {
    const result = validateManifest(builtinManifest({ version: "0.1.0" }));
    expect(result.valid).toBe(true);
  });

  // --- apiVersion enforcement ---

  test("missing apiVersion for non-builtin → error", () => {
    const result = validateManifest(thirdPartyManifest({ apiVersion: undefined }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("apiVersion"))).toBe(true);
  });

  test("wrong apiVersion → error", () => {
    const result = validateManifest(thirdPartyManifest({ apiVersion: "999" }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("not supported"))).toBe(true);
  });

  test("builtin without apiVersion → OK", () => {
    const result = validateManifest(builtinManifest({ apiVersion: undefined }));
    expect(result.valid).toBe(true);
  });

  // --- Permissions ---

  test("unknown permission → error", () => {
    const result = validateManifest(builtinManifest({ permissions: ["network", "teleport"] }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("teleport"))).toBe(true);
  });

  test("all known permissions → OK", () => {
    const result = validateManifest(builtinManifest({ permissions: [...KNOWN_PERMISSIONS] }));
    expect(result.valid).toBe(true);
  });

  // --- String length / control characters ---

  test("name exceeds max length → error", () => {
    const result = validateManifest(builtinManifest({ name: "x".repeat(100) }));
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("exceeds"))).toBe(true);
  });

  test("control characters in name → warning", () => {
    const result = validateManifest(builtinManifest({ name: "Test\x01" }));
    expect(result.valid).toBe(true);
    expect(result.warnings.some((w) => w.includes("control characters"))).toBe(true);
  });

  // --- Mode key ---

  test("invalid mode key format → error", () => {
    const result = validateManifest(
      builtinManifest({ contributes: { mode: { key: "My Mode!", label: "m" } } }),
    );
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("mode.key"))).toBe(true);
  });

  test("valid mode key → OK", () => {
    const result = validateManifest(
      builtinManifest({ contributes: { mode: { key: "my-mode", label: "m" } } }),
    );
    expect(result.valid).toBe(true);
  });

  // --- Contributes validation ---

  test("empty view id → error", () => {
    const result = validateManifest(
      builtinManifest({ contributes: { views: [{ id: "", slot: "sidebar", label: "V" }] } }),
    );
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("views"))).toBe(true);
  });

  test("empty command id → error", () => {
    const result = validateManifest(
      builtinManifest({ contributes: { commands: [{ id: "", label: "C" }] } }),
    );
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("commands"))).toBe(true);
  });

  test("empty tool id → error", () => {
    const result = validateManifest(
      builtinManifest({
        contributes: { tools: [{ id: "", description: "A tool" }] },
      }),
    );
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("tools"))).toBe(true);
  });

  test("empty statusBarItems id → error", () => {
    const result = validateManifest(
      builtinManifest({ contributes: { statusBarItems: [{ id: "", label: "S" }] } }),
    );
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("statusBarItems"))).toBe(true);
  });

  test("empty menus id → error", () => {
    const result = validateManifest(
      builtinManifest({ contributes: { menus: [{ id: "", slot: "toolbar", label: "M", command: "cmd" }] } }),
    );
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("menus") && e.includes("id"))).toBe(true);
  });

  test("empty menus slot → error", () => {
    const result = validateManifest(
      builtinManifest({ contributes: { menus: [{ id: "m", slot: "", label: "M", command: "cmd" }] } }),
    );
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("menus") && e.includes("slot"))).toBe(true);
  });

  test("empty menus command → error", () => {
    const result = validateManifest(
      builtinManifest({ contributes: { menus: [{ id: "m", slot: "toolbar", label: "M", command: "" }] } }),
    );
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("menus") && e.includes("command"))).toBe(true);
  });

  test("valid menus contribution → OK", () => {
    const result = validateManifest(
      builtinManifest({
        contributes: { menus: [{ id: "action", slot: "chat:toolbar", label: "Action", command: "p.run" }] },
      }),
    );
    expect(result.valid).toBe(true);
  });

  test("valid contributes → OK", () => {
    const result = validateManifest(
      builtinManifest({
        contributes: {
          views: [{ id: "main", slot: "sidebar", label: "Main" }],
          commands: [{ id: "run", label: "Run" }],
          tools: [{ id: "calc", description: "Calculator" }],
          statusBarItems: [{ id: "status", label: "Status" }],
          menus: [{ id: "action", slot: "toolbar", label: "Action", command: "run" }],
        },
      }),
    );
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });
});

describe("sanitizeManifestString", () => {
  test("strips control characters", () => {
    expect(sanitizeManifestString("Hello\x01World", 64)).toBe("HelloWorld");
  });

  test("truncates with ellipsis when over max length", () => {
    const result = sanitizeManifestString("a".repeat(20), 10);
    expect(result.length).toBe(10);
    expect(result.endsWith("\u2026")).toBe(true);
  });

  test("preserves newlines when allowed", () => {
    const result = sanitizeManifestString("line1\nline2", 64, true);
    expect(result).toBe("line1\nline2");
  });

  test("strips newlines when not allowed", () => {
    const result = sanitizeManifestString("line1\nline2", 64, false);
    expect(result).toBe("line1line2");
  });

  test("returns unchanged string under limit", () => {
    expect(sanitizeManifestString("hello", 64)).toBe("hello");
  });
});
