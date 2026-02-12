import { describe, test, expect } from "bun:test";
import {
  partitionByActivation,
  parseActivationEvent,
  topologicalSort,
  type PluginDescriptor,
} from "../../../src/kernel/core/activation.ts";
import type { PluginManifest } from "../../../src/kernel/types.ts";

/** Create a minimal PluginDescriptor for testing. */
function desc(id: string, overrides: Partial<PluginManifest> = {}): PluginDescriptor {
  return {
    manifest: {
      id,
      name: id,
      version: "1.0.0",
      builtin: true,
      ...overrides,
    } as PluginManifest,
    entryUrl: `/${id}.js`,
  };
}

describe("partitionByActivation", () => {
  test("no activationEvents → eager", () => {
    const { eager, lazy } = partitionByActivation([desc("a")]);
    expect(eager).toHaveLength(1);
    expect(lazy).toHaveLength(0);
  });

  test("empty activationEvents → eager", () => {
    const { eager } = partitionByActivation([desc("a", { activationEvents: [] })]);
    expect(eager).toHaveLength(1);
  });

  test('["*"] → eager', () => {
    const { eager } = partitionByActivation([desc("a", { activationEvents: ["*"] })]);
    expect(eager).toHaveLength(1);
  });

  test("specific activationEvents → lazy", () => {
    const { eager, lazy } = partitionByActivation([
      desc("a", { activationEvents: ["onView:sidebar"] }),
    ]);
    expect(eager).toHaveLength(0);
    expect(lazy).toHaveLength(1);
  });

  test("lazy dependency of eager → promoted to eager", () => {
    const { eager, lazy } = partitionByActivation([
      desc("a", { requires: [{ id: "b" }] }),                    // eager, requires b
      desc("b", { activationEvents: ["onView:sidebar"] }),        // lazy → promoted
    ]);
    expect(eager).toHaveLength(2);
    expect(lazy).toHaveLength(0);
  });

  test("transitive dependency promotion", () => {
    const { eager, lazy } = partitionByActivation([
      desc("a", { requires: [{ id: "b" }] }),                    // eager
      desc("b", { activationEvents: ["onView:x"], requires: [{ id: "c" }] }), // lazy → promoted
      desc("c", { activationEvents: ["onCommand:y"] }),           // lazy → promoted transitively
    ]);
    expect(eager).toHaveLength(3);
    expect(lazy).toHaveLength(0);
  });

  test("mixed eager and lazy partition", () => {
    const { eager, lazy } = partitionByActivation([
      desc("a"),                                                  // eager
      desc("b", { activationEvents: ["onView:x"] }),              // lazy
    ]);
    expect(eager).toHaveLength(1);
    expect(lazy).toHaveLength(1);
    expect(eager[0].manifest.id).toBe("a");
    expect(lazy[0].manifest.id).toBe("b");
  });
});

describe("parseActivationEvent", () => {
  test("onView:foo → { type: 'onView', arg: 'foo' }", () => {
    expect(parseActivationEvent("onView:foo")).toEqual({ type: "onView", arg: "foo" });
  });

  test("onCommand:bar → { type: 'onCommand', arg: 'bar' }", () => {
    expect(parseActivationEvent("onCommand:bar")).toEqual({ type: "onCommand", arg: "bar" });
  });

  test("onLanguage:js → { type: 'onLanguage', arg: 'js' }", () => {
    expect(parseActivationEvent("onLanguage:js")).toEqual({ type: "onLanguage", arg: "js" });
  });

  test("unrecognized pattern → null", () => {
    expect(parseActivationEvent("invalid")).toBeNull();
  });

  test("'*' → null (not parsed as a specific event)", () => {
    expect(parseActivationEvent("*")).toBeNull();
  });
});

describe("topologicalSort", () => {
  test("respects requires ordering", () => {
    const sorted = topologicalSort([
      desc("b", { requires: [{ id: "a" }] }),
      desc("a"),
    ]);
    const ids = sorted.map((p) => p.manifest.id);
    expect(ids.indexOf("a")).toBeLessThan(ids.indexOf("b"));
  });

  test("independent plugins maintain relative order", () => {
    const sorted = topologicalSort([desc("a"), desc("b"), desc("c")]);
    expect(sorted.map((p) => p.manifest.id)).toEqual(["a", "b", "c"]);
  });

  test("detects cycles → throws", () => {
    expect(() =>
      topologicalSort([
        desc("a", { requires: [{ id: "b" }] }),
        desc("b", { requires: [{ id: "a" }] }),
      ]),
    ).toThrow(/cycle/i);
  });

  test("three-node cycle → throws", () => {
    expect(() =>
      topologicalSort([
        desc("a", { requires: [{ id: "b" }] }),
        desc("b", { requires: [{ id: "c" }] }),
        desc("c", { requires: [{ id: "a" }] }),
      ]),
    ).toThrow(/cycle/i);
  });

  test("unknown dependency silently skipped", () => {
    const sorted = topologicalSort([desc("a", { requires: [{ id: "missing" }] })]);
    expect(sorted).toHaveLength(1);
    expect(sorted[0].manifest.id).toBe("a");
  });
});
