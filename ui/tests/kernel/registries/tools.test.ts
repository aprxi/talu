import { describe, test, expect, spyOn } from "bun:test";
import { ToolRegistryImpl } from "../../../src/kernel/registries/tools.ts";
import { HookPipelineImpl } from "../../../src/kernel/registries/hooks.ts";
import type { ToolDefinition, ToolResult } from "../../../src/kernel/types.ts";

function createRegistry(): ToolRegistryImpl {
  return new ToolRegistryImpl(new HookPipelineImpl());
}

function validResult(): ToolResult {
  return { content: [{ id: "1", type: "text", text: "ok" }] };
}

function simpleTool(overrides: Partial<ToolDefinition> = {}): ToolDefinition {
  return {
    description: "test tool",
    parameters: { type: "object", properties: { input: { type: "string" } } },
    execute: async () => validResult(),
    ...overrides,
  };
}

const signal = new AbortController().signal;

describe("ToolRegistryImpl registration", () => {
  test("register and get a tool", () => {
    const registry = createRegistry();
    const def = simpleTool();
    registry.registerScoped("plugin.a", "plugin.a.calc", def);
    expect(registry.get("plugin.a.calc")).toBe(def);
  });

  test("get returns undefined for unregistered tool", () => {
    const registry = createRegistry();
    expect(registry.get("no.such.tool")).toBeUndefined();
  });

  test("duplicate registration warns and keeps first", () => {
    const spy = spyOn(console, "warn").mockImplementation(() => {});
    const registry = createRegistry();
    const first = simpleTool();
    const second = simpleTool();
    registry.registerScoped("p", "p.tool", first);
    registry.registerScoped("p", "p.tool", second);
    expect(registry.get("p.tool")).toBe(first);
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  test("dispose removes the tool", () => {
    const registry = createRegistry();
    const d = registry.registerScoped("p", "p.tool", simpleTool());
    d.dispose();
    expect(registry.get("p.tool")).toBeUndefined();
  });

  test("dispose is safe if tool was already replaced", () => {
    const registry = createRegistry();
    const def1 = simpleTool();
    const d1 = registry.registerScoped("p", "p.tool", def1);
    d1.dispose();
    const def2 = simpleTool();
    registry.registerScoped("p", "p.tool", def2);
    // Disposing d1 again should not remove def2.
    d1.dispose();
    expect(registry.get("p.tool")).toBe(def2);
  });
});

describe("ToolRegistryImpl execute", () => {
  test("executes tool and returns result", async () => {
    const registry = createRegistry();
    registry.registerScoped("p", "p.tool", simpleTool());
    const result = await registry.execute("p.tool", { input: "hello" }, signal);
    expect(result.content).toHaveLength(1);
  });

  test("throws for unknown tool", async () => {
    const registry = createRegistry();
    await expect(registry.execute("missing", {}, signal)).rejects.toThrow(/not found/i);
  });

  test("validates args before execution", async () => {
    const registry = createRegistry();
    registry.registerScoped("p", "p.tool", simpleTool({
      parameters: {
        type: "object",
        properties: { count: { type: "integer" } },
        required: ["count"],
      },
    }));
    await expect(
      registry.execute("p.tool", {}, signal),
    ).rejects.toThrow(/validation failed/i);
  });

  test("validates tool result structure", async () => {
    const registry = createRegistry();
    registry.registerScoped("p", "p.bad", simpleTool({
      execute: async () => ({ bad: true }) as unknown as ToolResult,
    }));
    await expect(
      registry.execute("p.bad", {}, signal),
    ).rejects.toThrow(/content/i);
  });

  test("validates result content items have id and type", async () => {
    const registry = createRegistry();
    registry.registerScoped("p", "p.bad2", simpleTool({
      execute: async () => ({ content: [{ text: "no id/type" }] }) as unknown as ToolResult,
    }));
    await expect(
      registry.execute("p.bad2", {}, signal),
    ).rejects.toThrow(/missing required fields/i);
  });

  test("tool execution error is caught and re-thrown with context", async () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    const registry = createRegistry();
    registry.registerScoped("p", "p.boom", simpleTool({
      execute: async () => { throw new Error("tool exploded"); },
    }));
    await expect(
      registry.execute("p.boom", {}, signal),
    ).rejects.toThrow(/execution failed.*tool exploded/i);
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });
});
