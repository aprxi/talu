import { describe, test, expect, beforeEach, spyOn } from "bun:test";
import { ViewManager } from "../../../src/kernel/ui/view-manager.ts";

describe("ViewManager", () => {
  beforeEach(() => {
    document.body.innerHTML = `
      <div class="activity-bar">
        <button class="activity-btn" data-mode="chat">Chat</button>
      </div>
      <div class="app-content"></div>
    `;
  });

  test("registerView creates mode-view element", () => {
    const mgr = new ViewManager();
    mgr.registerView("ext.plugin", "main", "Ext Plugin", false, () => {});
    const view = document.getElementById("view-ext.plugin.main");
    expect(view).not.toBeNull();
    expect(view!.classList.contains("mode-view")).toBe(true);
  });

  test("registerView creates activity bar button", () => {
    const mgr = new ViewManager();
    mgr.registerView("ext.plugin", "main", "Ext Plugin", false, () => {});
    const btn = document.querySelector<HTMLElement>('.activity-btn[data-mode="ext.plugin.main"]');
    expect(btn).not.toBeNull();
    expect(btn!.title).toBe("Ext Plugin");
  });

  test("third-party views get ext badge", () => {
    const mgr = new ViewManager();
    mgr.registerView("ext.plugin", "main", "Ext Plugin", false, () => {});
    const badge = document.querySelector(".activity-btn-badge");
    expect(badge).not.toBeNull();
    expect(badge!.textContent).toBe("ext");
  });

  test("builtin views do not get ext badge", () => {
    const mgr = new ViewManager();
    mgr.registerView("talu.builtin", "main", "Builtin", true, () => {});
    const badge = document.querySelector(".activity-btn-badge");
    expect(badge).toBeNull();
  });

  test("factory receives shadow root", () => {
    const mgr = new ViewManager();
    let received: ShadowRoot | null = null;
    mgr.registerView("ext.plugin", "main", "Plugin", false, (sr) => {
      received = sr;
    });
    expect(received).not.toBeNull();
    expect(received).toBeInstanceOf(ShadowRoot);
  });

  test("factory error is caught", () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    const mgr = new ViewManager();
    const d = mgr.registerView("ext.plugin", "main", "Plugin", false, () => {
      throw new Error("factory boom");
    });
    expect(d).toBeDefined();
    spy.mockRestore();
  });

  test("dispose removes view and button from DOM", () => {
    const mgr = new ViewManager();
    const d = mgr.registerView("ext.plugin", "main", "Plugin", false, () => {});
    expect(document.getElementById("view-ext.plugin.main")).not.toBeNull();
    d.dispose();
    expect(document.getElementById("view-ext.plugin.main")).toBeNull();
    expect(document.querySelector('.activity-btn[data-mode="ext.plugin.main"]')).toBeNull();
  });

  test("duplicate view registration is rejected", () => {
    const spy = spyOn(console, "warn").mockImplementation(() => {});
    const mgr = new ViewManager();
    mgr.registerView("ext.plugin", "main", "Plugin", false, () => {});
    const d2 = mgr.registerView("ext.plugin", "main", "Plugin", false, () => {});
    // Second registration should be no-op.
    expect(() => d2.dispose()).not.toThrow();
    spy.mockRestore();
  });

  test("getRegisteredViewIds returns all view IDs", () => {
    const mgr = new ViewManager();
    mgr.registerView("a", "v1", "A", false, () => {});
    mgr.registerView("b", "v2", "B", false, () => {});
    const ids = mgr.getRegisteredViewIds();
    expect(ids.sort()).toEqual(["a.v1", "b.v2"]);
  });

  test("has returns true for registered view", () => {
    const mgr = new ViewManager();
    mgr.registerView("ext.plugin", "main", "Plugin", false, () => {});
    expect(mgr.has("ext.plugin.main")).toBe(true);
    expect(mgr.has("nonexistent")).toBe(false);
  });

  test("no-op when content area missing", () => {
    document.body.innerHTML = "";
    const mgr = new ViewManager();
    const d = mgr.registerView("ext.plugin", "main", "Plugin", false, () => {});
    expect(() => d.dispose()).not.toThrow();
  });
});
