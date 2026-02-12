import { describe, test, expect, beforeEach } from "bun:test";
import { PopoverManager } from "../../../src/kernel/ui/popover.ts";

describe("PopoverManager", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
  });

  test("showPopover appends wrapper to document body", () => {
    const mgr = new PopoverManager();
    const anchor = document.createElement("button");
    document.body.appendChild(anchor);

    mgr.showPopover("test.plugin", {
      anchor,
      content: document.createElement("div"),
    });

    const wrapper = document.querySelector("[data-popover-owner='test.plugin']");
    expect(wrapper).not.toBeNull();
  });

  test("showPopover returns disposable", () => {
    const mgr = new PopoverManager();
    const anchor = document.createElement("button");
    document.body.appendChild(anchor);

    const d = mgr.showPopover("test.plugin", {
      anchor,
      content: document.createElement("div"),
    });
    expect(typeof d.dispose).toBe("function");
  });

  test("dispose removes wrapper from DOM", () => {
    const mgr = new PopoverManager();
    const anchor = document.createElement("button");
    document.body.appendChild(anchor);

    const d = mgr.showPopover("test.plugin", {
      anchor,
      content: document.createElement("div"),
    });
    d.dispose();

    expect(document.querySelector("[data-popover-owner='test.plugin']")).toBeNull();
  });

  test("second popover for same plugin dismisses first", () => {
    const mgr = new PopoverManager();
    const anchor = document.createElement("button");
    document.body.appendChild(anchor);

    mgr.showPopover("test.plugin", {
      anchor,
      content: document.createElement("div"),
    });
    mgr.showPopover("test.plugin", {
      anchor,
      content: document.createElement("div"),
    });

    const wrappers = document.querySelectorAll("[data-popover-owner='test.plugin']");
    expect(wrappers.length).toBe(1);
  });

  test("different plugins can have concurrent popovers", () => {
    const mgr = new PopoverManager();
    const anchor = document.createElement("button");
    document.body.appendChild(anchor);

    mgr.showPopover("plugin.a", { anchor, content: document.createElement("div") });
    mgr.showPopover("plugin.b", { anchor, content: document.createElement("div") });

    expect(document.querySelector("[data-popover-owner='plugin.a']")).not.toBeNull();
    expect(document.querySelector("[data-popover-owner='plugin.b']")).not.toBeNull();
  });

  test("escape key dismisses popover", () => {
    const mgr = new PopoverManager();
    const anchor = document.createElement("button");
    document.body.appendChild(anchor);

    mgr.showPopover("test.plugin", {
      anchor,
      content: document.createElement("div"),
    });

    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(document.querySelector("[data-popover-owner='test.plugin']")).toBeNull();
  });

  test("popover has shadow DOM for content isolation", () => {
    const mgr = new PopoverManager();
    const anchor = document.createElement("button");
    document.body.appendChild(anchor);

    const content = document.createElement("div");
    content.textContent = "Popover content";

    mgr.showPopover("test.plugin", { anchor, content });

    const wrapper = document.querySelector("[data-popover-owner='test.plugin']")!;
    const shadowHost = wrapper.firstElementChild!;
    expect(shadowHost.shadowRoot).not.toBeNull();
  });
});
