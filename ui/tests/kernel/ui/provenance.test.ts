import { describe, test, expect, beforeEach } from "bun:test";
import {
  initProvenance,
  setProvenanceAction,
  updateProvenance,
} from "../../../src/kernel/ui/provenance.ts";

describe("provenance indicator", () => {
  beforeEach(() => {
    document.body.innerHTML = `<div class="topbar"></div>`;
    // Re-init to bind to the fresh DOM.
    initProvenance();
  });

  test("initProvenance creates indicator in topbar", () => {
    const indicator = document.getElementById("view-provenance");
    expect(indicator).not.toBeNull();
    expect(indicator!.tagName).toBe("SPAN");
    expect(indicator!.title).toBe("Click for plugin info");
  });

  test("initProvenance uses existing element if present", () => {
    document.body.innerHTML = `
      <div class="topbar">
        <span id="view-provenance">existing</span>
      </div>
    `;
    initProvenance();
    const indicators = document.querySelectorAll("#view-provenance");
    expect(indicators.length).toBe(1);
  });

  test("updateProvenance shows plugin name for builtin", () => {
    updateProvenance("Chat", "talu.chat", true);
    const indicator = document.getElementById("view-provenance")!;
    expect(indicator.textContent).toBe("Chat");
    // No ext badge for builtin.
    expect(indicator.querySelectorAll("span").length).toBe(1);
  });

  test("updateProvenance shows name + id + ext badge for third-party", () => {
    updateProvenance("My Plugin", "ext.my-plugin", false);
    const indicator = document.getElementById("view-provenance")!;
    expect(indicator.textContent).toContain("My Plugin");
    expect(indicator.textContent).toContain("(ext.my-plugin)");
    expect(indicator.textContent).toContain("ext");
    // Three spans: name, id, badge.
    expect(indicator.querySelectorAll("span").length).toBe(3);
  });

  test("updateProvenance clears previous content", () => {
    updateProvenance("First", "a", true);
    updateProvenance("Second", "b", true);
    const indicator = document.getElementById("view-provenance")!;
    expect(indicator.textContent).toBe("Second");
  });

  test("click invokes command palette opener", () => {
    let called = false;
    setProvenanceAction(() => { called = true; });
    const indicator = document.getElementById("view-provenance")!;
    indicator.click();
    expect(called).toBe(true);
  });

  test("click with no opener does nothing", () => {
    const indicator = document.getElementById("view-provenance")!;
    expect(() => indicator.click()).not.toThrow();
  });
});
