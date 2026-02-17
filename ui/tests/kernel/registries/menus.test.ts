import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { MenuRegistry } from "../../../src/kernel/registries/menus.ts";
import { ContextKeyService } from "../../../src/kernel/registries/context-keys.ts";
import { CommandRegistryImpl } from "../../../src/kernel/registries/commands.ts";
import type { PluginManifest } from "../../../src/kernel/types.ts";

let contextKeys: ContextKeyService;
let commandRegistry: CommandRegistryImpl;
let registry: MenuRegistry;

beforeEach(() => {
  document.body.innerHTML = "";
  contextKeys = new ContextKeyService();
  commandRegistry = new CommandRegistryImpl(contextKeys);
  registry = new MenuRegistry(contextKeys, commandRegistry);
});

afterEach(() => {
  registry.dispose();
  commandRegistry.dispose();
  contextKeys.dispose();
});

function makeManifest(menus: PluginManifest["contributes"] extends infer C ? NonNullable<C>["menus"] : never): PluginManifest {
  return {
    id: "test.plugin",
    name: "Test Plugin",
    version: "1.0.0",
    contributes: { menus },
  } as PluginManifest;
}

// ── registerFromManifest ─────────────────────────────────────────────────────

describe("MenuRegistry — registerFromManifest", () => {
  test("registers items from manifest", () => {
    registry.registerFromManifest("test.plugin", makeManifest([
      { id: "action", slot: "toolbar", label: "Action", command: "test.plugin.run" },
    ]));
    const items = registry.getItems("toolbar");
    expect(items).toHaveLength(1);
    expect(items[0].id).toBe("test.plugin.action");
    expect(items[0].label).toBe("Action");
  });

  test("no-op for empty manifest contributes", () => {
    const d = registry.registerFromManifest("test.plugin", {
      id: "test.plugin",
      name: "Test",
      version: "1.0.0",
    } as PluginManifest);
    expect(() => d.dispose()).not.toThrow();
  });

  test("no-op for empty menus array", () => {
    const d = registry.registerFromManifest("test.plugin", makeManifest([]));
    expect(() => d.dispose()).not.toThrow();
    expect(registry.getItems("anything")).toHaveLength(0);
  });

  test("duplicate FQ IDs are skipped", () => {
    registry.registerFromManifest("test.plugin", makeManifest([
      { id: "action", slot: "toolbar", label: "First", command: "cmd" },
    ]));
    registry.registerFromManifest("test.plugin", makeManifest([
      { id: "action", slot: "toolbar", label: "Duplicate", command: "cmd" },
    ]));
    const items = registry.getItems("toolbar");
    expect(items).toHaveLength(1);
    expect(items[0].label).toBe("First");
  });

  test("dispose removes registered items", () => {
    const d = registry.registerFromManifest("test.plugin", makeManifest([
      { id: "action", slot: "toolbar", label: "Action", command: "cmd" },
    ]));
    expect(registry.getItems("toolbar")).toHaveLength(1);
    d.dispose();
    expect(registry.getItems("toolbar")).toHaveLength(0);
  });

  test("registers multiple items to different slots", () => {
    registry.registerFromManifest("test.plugin", makeManifest([
      { id: "a", slot: "toolbar", label: "A", command: "cmd.a" },
      { id: "b", slot: "sidebar", label: "B", command: "cmd.b" },
    ]));
    expect(registry.getItems("toolbar")).toHaveLength(1);
    expect(registry.getItems("sidebar")).toHaveLength(1);
  });
});

// ── registerItem ─────────────────────────────────────────────────────────────

describe("MenuRegistry — registerItem", () => {
  test("registers a single item imperatively", () => {
    registry.registerItem("test.plugin", {
      id: "btn", slot: "toolbar", label: "Button", command: "test.plugin.click",
    });
    const items = registry.getItems("toolbar");
    expect(items).toHaveLength(1);
    expect(items[0].id).toBe("test.plugin.btn");
  });

  test("dispose removes the item", () => {
    const d = registry.registerItem("test.plugin", {
      id: "btn", slot: "toolbar", label: "Button", command: "cmd",
    });
    expect(registry.getItems("toolbar")).toHaveLength(1);
    d.dispose();
    expect(registry.getItems("toolbar")).toHaveLength(0);
  });

  test("duplicate FQ ID returns no-op disposable", () => {
    registry.registerItem("test.plugin", {
      id: "btn", slot: "toolbar", label: "First", command: "cmd",
    });
    const d = registry.registerItem("test.plugin", {
      id: "btn", slot: "toolbar", label: "Duplicate", command: "cmd",
    });
    expect(registry.getItems("toolbar")).toHaveLength(1);
    expect(registry.getItems("toolbar")[0].label).toBe("First");
    d.dispose(); // Should not remove the original.
    expect(registry.getItems("toolbar")).toHaveLength(1);
  });
});

// ── getItems ─────────────────────────────────────────────────────────────────

describe("MenuRegistry — getItems", () => {
  test("returns empty array for unknown slot", () => {
    expect(registry.getItems("nonexistent")).toHaveLength(0);
  });

  test("sorts by priority descending (higher first)", () => {
    registry.registerItem("a", { id: "low", slot: "toolbar", label: "Low", command: "cmd", priority: 1 });
    registry.registerItem("b", { id: "high", slot: "toolbar", label: "High", command: "cmd", priority: 10 });
    registry.registerItem("c", { id: "mid", slot: "toolbar", label: "Mid", command: "cmd", priority: 5 });
    const items = registry.getItems("toolbar");
    expect(items.map((i) => i.label)).toEqual(["High", "Mid", "Low"]);
  });

  test("default priority is 0", () => {
    registry.registerItem("a", { id: "explicit", slot: "toolbar", label: "Explicit", command: "cmd", priority: 1 });
    registry.registerItem("b", { id: "default", slot: "toolbar", label: "Default", command: "cmd" });
    const items = registry.getItems("toolbar");
    expect(items[0].label).toBe("Explicit");
    expect(items[1].priority).toBe(0);
  });

  test("filters out items whose when-clause does not match", () => {
    registry.registerItem("a", { id: "always", slot: "toolbar", label: "Always", command: "cmd" });
    registry.registerItem("b", {
      id: "gated", slot: "toolbar", label: "Gated", command: "cmd", when: "isEditing",
    });
    // isEditing is not set → only "Always" visible.
    expect(registry.getItems("toolbar")).toHaveLength(1);
    expect(registry.getItems("toolbar")[0].label).toBe("Always");
    // Set context key → both visible.
    contextKeys.set("isEditing", true);
    expect(registry.getItems("toolbar")).toHaveLength(2);
  });

  test("items without when-clause are always visible", () => {
    registry.registerItem("a", { id: "btn", slot: "toolbar", label: "Btn", command: "cmd" });
    expect(registry.getItems("toolbar")).toHaveLength(1);
  });
});

// ── renderSlot ───────────────────────────────────────────────────────────────

describe("MenuRegistry — renderSlot", () => {
  test("creates buttons in container", () => {
    commandRegistry.registerScoped("a", "a.run", () => {});
    registry.registerItem("a", { id: "run", slot: "toolbar", label: "Run", command: "a.run" });
    const container = document.createElement("div");
    registry.renderSlot("toolbar", container);
    const btns = container.querySelectorAll(".menu-slot-btn");
    expect(btns).toHaveLength(1);
    expect(btns[0].getAttribute("title")).toBe("Run");
  });

  test("button click executes the command", () => {
    let executed = false;
    commandRegistry.registerScoped("a", "a.run", () => { executed = true; });
    registry.registerItem("a", { id: "run", slot: "toolbar", label: "Run", command: "a.run" });
    const container = document.createElement("div");
    registry.renderSlot("toolbar", container);
    const btn = container.querySelector(".menu-slot-btn") as HTMLElement;
    btn.click();
    expect(executed).toBe(true);
  });

  test("renders icon as innerHTML when provided", () => {
    commandRegistry.registerScoped("a", "a.run", () => {});
    registry.registerItem("a", {
      id: "run", slot: "toolbar", label: "Run", command: "a.run",
      icon: "<svg>icon</svg>",
    });
    const container = document.createElement("div");
    registry.renderSlot("toolbar", container);
    const btn = container.querySelector(".menu-slot-btn")!;
    expect(btn.innerHTML).toContain("<svg>");
  });

  test("uses label as text when no icon", () => {
    commandRegistry.registerScoped("a", "a.run", () => {});
    registry.registerItem("a", { id: "run", slot: "toolbar", label: "Run", command: "a.run" });
    const container = document.createElement("div");
    registry.renderSlot("toolbar", container);
    const btn = container.querySelector(".menu-slot-btn")!;
    expect(btn.textContent).toBe("Run");
  });

  test("re-renders when items are added after initial render", () => {
    const container = document.createElement("div");
    registry.renderSlot("toolbar", container);
    expect(container.querySelectorAll(".menu-slot-btn")).toHaveLength(0);
    // Add an item after rendering.
    commandRegistry.registerScoped("a", "a.run", () => {});
    registry.registerItem("a", { id: "run", slot: "toolbar", label: "Run", command: "a.run" });
    // Should have re-rendered.
    expect(container.querySelectorAll(".menu-slot-btn")).toHaveLength(1);
  });

  test("re-renders when items are removed", () => {
    commandRegistry.registerScoped("a", "a.run", () => {});
    const d = registry.registerItem("a", { id: "run", slot: "toolbar", label: "Run", command: "a.run" });
    const container = document.createElement("div");
    registry.renderSlot("toolbar", container);
    expect(container.querySelectorAll(".menu-slot-btn")).toHaveLength(1);
    d.dispose();
    expect(container.querySelectorAll(".menu-slot-btn")).toHaveLength(0);
  });

  test("dispose stops re-rendering and clears container", () => {
    const container = document.createElement("div");
    const d = registry.renderSlot("toolbar", container);
    commandRegistry.registerScoped("a", "a.run", () => {});
    registry.registerItem("a", { id: "run", slot: "toolbar", label: "Run", command: "a.run" });
    expect(container.querySelectorAll(".menu-slot-btn")).toHaveLength(1);
    d.dispose();
    expect(container.innerHTML).toBe("");
    // Adding more items should not re-render.
    registry.registerItem("b", { id: "run2", slot: "toolbar", label: "Run2", command: "a.run" });
    expect(container.innerHTML).toBe("");
  });

  test("empty slot renders nothing", () => {
    const container = document.createElement("div");
    registry.renderSlot("empty-slot", container);
    expect(container.children).toHaveLength(0);
  });

  test("sets data-menu-item-id on buttons", () => {
    commandRegistry.registerScoped("a", "a.run", () => {});
    registry.registerItem("a", { id: "run", slot: "toolbar", label: "Run", command: "a.run" });
    const container = document.createElement("div");
    registry.renderSlot("toolbar", container);
    const btn = container.querySelector(".menu-slot-btn") as HTMLElement;
    expect(btn.dataset["menuItemId"]).toBe("a.run");
  });
});

// ── dispose ──────────────────────────────────────────────────────────────────

describe("MenuRegistry — dispose", () => {
  test("clears all items and listeners", () => {
    registry.registerItem("a", { id: "run", slot: "toolbar", label: "Run", command: "cmd" });
    registry.dispose();
    expect(registry.getItems("toolbar")).toHaveLength(0);
  });

  test("dispose is idempotent", () => {
    registry.dispose();
    registry.dispose();
    expect(registry.getItems("toolbar")).toHaveLength(0);
  });
});
