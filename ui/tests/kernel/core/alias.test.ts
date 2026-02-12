import { describe, test, expect, spyOn } from "bun:test";
import { registerAliases, resolveAlias } from "../../../src/kernel/core/alias.ts";

describe("resolveAlias", () => {
  test("unregistered ID returns itself", () => {
    expect(resolveAlias("no.such.alias")).toBe("no.such.alias");
  });

  test("registered alias resolves to target", () => {
    registerAliases({ "old.service.id": "new.service.id" });
    expect(resolveAlias("old.service.id")).toBe("new.service.id");
  });

  test("target ID (not an alias) resolves to itself", () => {
    registerAliases({ "alias.x": "target.x" });
    expect(resolveAlias("target.x")).toBe("target.x");
  });
});

describe("registerAliases", () => {
  test("undefined input is a no-op", () => {
    expect(() => registerAliases(undefined)).not.toThrow();
  });

  test("empty object is a no-op", () => {
    expect(() => registerAliases({})).not.toThrow();
  });

  test("duplicate alias warns and keeps first registration", () => {
    const spy = spyOn(console, "warn").mockImplementation(() => {});
    registerAliases({ "dup.alias": "first.target" });
    registerAliases({ "dup.alias": "second.target" });
    expect(resolveAlias("dup.alias")).toBe("first.target");
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  test("multiple aliases registered in one call", () => {
    registerAliases({
      "batch.a": "target.a",
      "batch.b": "target.b",
    });
    expect(resolveAlias("batch.a")).toBe("target.a");
    expect(resolveAlias("batch.b")).toBe("target.b");
  });
});
