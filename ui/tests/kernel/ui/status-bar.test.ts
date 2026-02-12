import { describe, test, expect, beforeEach } from "bun:test";
import { StatusBarManager } from "../../../src/kernel/ui/status-bar.ts";
import type { PluginManifest } from "../../../src/kernel/types.ts";

function makeManifest(items: { id: string; label: string; alignment?: "left" | "right"; priority?: number }[]): PluginManifest {
  return {
    id: "test.plugin",
    name: "Test Plugin",
    version: "1.0.0",
    contributes: { statusBarItems: items },
  } as PluginManifest;
}

describe("StatusBarManager", () => {
  beforeEach(() => {
    document.body.innerHTML = `<div id="status-bar"></div>`;
  });

  test("creates left and right groups", () => {
    new StatusBarManager();
    const bar = document.getElementById("status-bar")!;
    expect(bar.querySelector(".status-bar-left")).not.toBeNull();
    expect(bar.querySelector(".status-bar-right")).not.toBeNull();
  });

  test("registerFromManifest adds items to DOM", () => {
    const mgr = new StatusBarManager();
    mgr.registerFromManifest("test.plugin", makeManifest([
      { id: "item1", label: "Ready" },
    ]));

    const bar = document.getElementById("status-bar")!;
    const item = bar.querySelector(".status-bar-item");
    expect(item).not.toBeNull();
    expect(item!.textContent).toBe("Ready");
  });

  test("items default to left alignment", () => {
    const mgr = new StatusBarManager();
    mgr.registerFromManifest("test.plugin", makeManifest([
      { id: "item1", label: "Left" },
    ]));

    const left = document.querySelector(".status-bar-left")!;
    expect(left.querySelector(".status-bar-item")).not.toBeNull();
  });

  test("right-aligned items go to right group", () => {
    const mgr = new StatusBarManager();
    mgr.registerFromManifest("test.plugin", makeManifest([
      { id: "item1", label: "Right", alignment: "right" },
    ]));

    const right = document.querySelector(".status-bar-right")!;
    expect(right.querySelector(".status-bar-item")).not.toBeNull();
  });

  test("higher priority items appear first", () => {
    const mgr = new StatusBarManager();
    mgr.registerFromManifest("a", makeManifest([
      { id: "low", label: "Low", priority: 1 },
    ]));
    mgr.registerFromManifest("b", makeManifest([
      { id: "high", label: "High", priority: 10 },
    ]));

    const left = document.querySelector(".status-bar-left")!;
    const items = left.querySelectorAll(".status-bar-item");
    expect(items[0]!.textContent).toBe("High");
    expect(items[1]!.textContent).toBe("Low");
  });

  test("updateLabel changes item text", () => {
    const mgr = new StatusBarManager();
    mgr.registerFromManifest("test.plugin", makeManifest([
      { id: "item1", label: "Initial" },
    ]));

    mgr.updateLabel("test.plugin.item1", "Updated");
    const item = document.querySelector(".status-bar-item")!;
    expect(item.textContent).toBe("Updated");
  });

  test("updateLabel no-op for unknown item", () => {
    const mgr = new StatusBarManager();
    expect(() => mgr.updateLabel("nonexistent", "value")).not.toThrow();
  });

  test("dispose removes items from DOM", () => {
    const mgr = new StatusBarManager();
    const d = mgr.registerFromManifest("test.plugin", makeManifest([
      { id: "item1", label: "Temp" },
    ]));

    expect(document.querySelector(".status-bar-item")).not.toBeNull();
    d.dispose();
    expect(document.querySelector(".status-bar-item")).toBeNull();
  });

  test("no-op for empty manifest contributes", () => {
    const mgr = new StatusBarManager();
    const d = mgr.registerFromManifest("test.plugin", {
      id: "test.plugin",
      name: "Test",
      version: "1.0.0",
    } as PluginManifest);
    expect(() => d.dispose()).not.toThrow();
  });

  test("duplicate item IDs are skipped", () => {
    const mgr = new StatusBarManager();
    mgr.registerFromManifest("test.plugin", makeManifest([
      { id: "item1", label: "First" },
    ]));
    mgr.registerFromManifest("test.plugin", makeManifest([
      { id: "item1", label: "Duplicate" },
    ]));

    const items = document.querySelectorAll(".status-bar-item");
    expect(items.length).toBe(1);
    expect(items[0]!.textContent).toBe("First");
  });

  test("no-op when status-bar element missing", () => {
    document.body.innerHTML = "";
    const mgr = new StatusBarManager();
    expect(() => mgr.registerFromManifest("test", makeManifest([
      { id: "x", label: "X" },
    ]))).not.toThrow();
  });
});
