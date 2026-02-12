import { describe, test, expect } from "bun:test";
import { namespacedId, validateLocalId } from "../../../src/kernel/core/id-namespace.ts";

describe("namespacedId", () => {
  test("prefixes local ID with plugin ID", () => {
    expect(namespacedId("my-plugin", "myCommand")).toBe("my-plugin.myCommand");
  });

  test("does not double-prefix if already prefixed", () => {
    expect(namespacedId("my-plugin", "my-plugin.myCommand")).toBe("my-plugin.myCommand");
  });

  test("prefixes even if ID contains a dot from another namespace", () => {
    expect(namespacedId("plugin-a", "plugin-b.thing")).toBe("plugin-a.plugin-b.thing");
  });

  test("handles empty local ID", () => {
    expect(namespacedId("p", "")).toBe("p.");
  });
});

describe("validateLocalId", () => {
  test("allows simple local IDs for third-party plugins", () => {
    expect(() => validateLocalId("csv-viewer", "render", false)).not.toThrow();
  });

  test("allows hyphenated local IDs for third-party plugins", () => {
    expect(() => validateLocalId("csv-viewer", "render-table", false)).not.toThrow();
  });

  test("throws if third-party plugin uses dots in local ID", () => {
    expect(() => validateLocalId("csv-viewer", "render.table", false)).toThrow(
      /local name.*no dots/i,
    );
  });

  test("allows dots for built-in plugins", () => {
    expect(() => validateLocalId("talu.chat", "render.table", true)).not.toThrow();
  });

  test("throws if third-party plugin's fully-qualified ID would claim talu.* namespace", () => {
    // pluginId = "talu", localId = "steal" → fqId = "talu.steal" → reserved
    expect(() => validateLocalId("talu", "steal", false)).toThrow(/reserved/i);
  });

  test("allows built-in plugins in talu.* namespace", () => {
    expect(() => validateLocalId("talu.chat", "selectChat", true)).not.toThrow();
  });
});
