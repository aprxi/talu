import { describe, test, expect, spyOn, beforeEach, afterEach } from "bun:test";
import { CommandRegistryImpl } from "../../../src/kernel/registries/commands.ts";
import { ContextKeyService } from "../../../src/kernel/registries/context-keys.ts";

let contextKeys: ContextKeyService;
let registry: CommandRegistryImpl;

beforeEach(() => {
  contextKeys = new ContextKeyService();
  registry = new CommandRegistryImpl(contextKeys);
});

afterEach(() => {
  registry.dispose();
  contextKeys.dispose();
});

describe("CommandRegistryImpl", () => {
  test("register and execute a command", () => {
    let called = false;
    registry.registerScoped("plugin.a", "plugin.a.run", () => { called = true; });
    const result = registry.execute("plugin.a.run");
    expect(result).toBe(true);
    expect(called).toBe(true);
  });

  test("execute returns false for unknown command", () => {
    expect(registry.execute("no.such.cmd")).toBe(false);
  });

  test("dispose removes the command", () => {
    const d = registry.registerScoped("p", "p.cmd", () => {});
    d.dispose();
    expect(registry.execute("p.cmd")).toBe(false);
  });

  test("getAll returns all registered commands", () => {
    registry.registerScoped("p", "p.a", () => {});
    registry.registerScoped("p", "p.b", () => {});
    const all = registry.getAll();
    expect(all).toHaveLength(2);
    expect(all.map((c) => c.id).sort()).toEqual(["p.a", "p.b"]);
  });

  test("handler error is caught and returns false", () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    registry.registerScoped("p", "p.boom", () => { throw new Error("boom"); });
    const result = registry.execute("p.boom");
    expect(result).toBe(false);
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  test("when-clause: focusedView == match allows execution", () => {
    // With no focused view, focusedView is "" — so matching against '' works.
    let called = false;
    registry.registerScoped("p", "p.cmd", () => { called = true; }, {
      when: "focusedView == ''",
    });
    expect(registry.execute("p.cmd")).toBe(true);
    expect(called).toBe(true);
  });

  test("when-clause: focusedView == mismatch blocks execution", () => {
    let called = false;
    registry.registerScoped("p", "p.cmd", () => { called = true; }, {
      when: "focusedView == 'editor'",
    });
    expect(registry.execute("p.cmd")).toBe(false);
    expect(called).toBe(false);
  });

  test("when-clause: focusedView != match allows execution", () => {
    let called = false;
    registry.registerScoped("p", "p.cmd", () => { called = true; }, {
      when: "focusedView != 'editor'",
    });
    expect(registry.execute("p.cmd")).toBe(true);
    expect(called).toBe(true);
  });

  test("no when-clause always matches", () => {
    let called = false;
    registry.registerScoped("p", "p.cmd", () => { called = true; });
    expect(registry.execute("p.cmd")).toBe(true);
    expect(called).toBe(true);
  });

  test("register with label is retrievable via getAll", () => {
    registry.registerScoped("p", "p.cmd", () => {}, { label: "My Command" });
    const all = registry.getAll();
    expect(all[0].label).toBe("My Command");
  });

  test("dispose clears all commands", () => {
    registry.registerScoped("p", "p.a", () => {});
    registry.registerScoped("p", "p.b", () => {});
    registry.dispose();
    expect(registry.getAll()).toHaveLength(0);
  });

  test("when-clause: bare key truthy when context key is set", () => {
    let called = false;
    contextKeys.set("isEditing", true);
    registry.registerScoped("p", "p.cmd", () => { called = true; }, {
      when: "isEditing",
    });
    expect(registry.execute("p.cmd")).toBe(true);
    expect(called).toBe(true);
  });

  test("when-clause: !key blocks when key is truthy", () => {
    let called = false;
    contextKeys.set("isEditing", true);
    registry.registerScoped("p", "p.cmd", () => { called = true; }, {
      when: "!isEditing",
    });
    expect(registry.execute("p.cmd")).toBe(false);
    expect(called).toBe(false);
  });

  test("when-clause: arbitrary key == 'value' via context keys", () => {
    let called = false;
    contextKeys.set("activeMode", "dark");
    registry.registerScoped("p", "p.cmd", () => { called = true; }, {
      when: "activeMode == 'dark'",
    });
    expect(registry.execute("p.cmd")).toBe(true);
    expect(called).toBe(true);
  });
});
