import { describe, test, expect, beforeEach } from "bun:test";
import {
  setupAccessibility,
  setupPluginViewAccessibility,
  updateActiveTab,
} from "../../../src/kernel/ui/accessibility.ts";

describe("setupAccessibility", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
  });

  test("sets activity bar ARIA roles", () => {
    document.body.innerHTML = `
      <div id="activity-bar">
        <button data-mode="chat">Chat</button>
        <button data-mode="settings">Settings</button>
      </div>
      <div id="status-bar"></div>
    `;
    setupAccessibility();

    const bar = document.getElementById("activity-bar")!;
    expect(bar.getAttribute("role")).toBe("tablist");
    expect(bar.getAttribute("aria-label")).toBe("Plugin views");

    const items = bar.querySelectorAll("[data-mode]");
    expect(items[0]!.getAttribute("role")).toBe("tab");
    expect(items[0]!.getAttribute("aria-selected")).toBe("false");
    expect(items[0]!.getAttribute("aria-label")).toBe("chat");
    expect(items[1]!.getAttribute("aria-label")).toBe("settings");
  });

  test("sets status bar ARIA roles", () => {
    document.body.innerHTML = `<div id="status-bar"></div>`;
    setupAccessibility();

    const bar = document.getElementById("status-bar")!;
    expect(bar.getAttribute("role")).toBe("status");
    expect(bar.getAttribute("aria-live")).toBe("polite");
  });

  test("handles hyphenated mode names", () => {
    document.body.innerHTML = `
      <div id="activity-bar">
        <button data-mode="my-custom-view">X</button>
      </div>
    `;
    setupAccessibility();

    const btn = document.querySelector("[data-mode]")!;
    expect(btn.getAttribute("aria-label")).toBe("my custom view");
  });

  test("no-op when elements missing", () => {
    document.body.innerHTML = "";
    expect(() => setupAccessibility()).not.toThrow();
  });
});

describe("setupPluginViewAccessibility", () => {
  test("sets tabpanel role and aria-label", () => {
    const host = document.createElement("div");
    setupPluginViewAccessibility(host, "My Plugin");
    expect(host.getAttribute("role")).toBe("tabpanel");
    expect(host.getAttribute("aria-label")).toBe("My Plugin");
  });

  test("sets aria-labelledby when tabId provided", () => {
    const host = document.createElement("div");
    setupPluginViewAccessibility(host, "Plugin", "tab-chat");
    expect(host.getAttribute("aria-labelledby")).toBe("tab-chat");
  });

  test("omits aria-labelledby when no tabId", () => {
    const host = document.createElement("div");
    setupPluginViewAccessibility(host, "Plugin");
    expect(host.hasAttribute("aria-labelledby")).toBe(false);
  });
});

describe("updateActiveTab", () => {
  beforeEach(() => {
    document.body.innerHTML = `
      <div id="activity-bar">
        <button data-mode="chat">Chat</button>
        <button data-mode="settings">Settings</button>
      </div>
    `;
  });

  test("sets aria-selected on active tab", () => {
    updateActiveTab("chat");
    const items = document.querySelectorAll("[data-mode]");
    expect(items[0]!.getAttribute("aria-selected")).toBe("true");
    expect(items[1]!.getAttribute("aria-selected")).toBe("false");
  });

  test("switches active tab", () => {
    updateActiveTab("chat");
    updateActiveTab("settings");
    const items = document.querySelectorAll("[data-mode]");
    expect(items[0]!.getAttribute("aria-selected")).toBe("false");
    expect(items[1]!.getAttribute("aria-selected")).toBe("true");
  });

  test("no-op when activity bar missing", () => {
    document.body.innerHTML = "";
    expect(() => updateActiveTab("chat")).not.toThrow();
  });
});
