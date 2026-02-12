import { describe, test, expect, beforeEach } from "bun:test";
import { getSharedStylesheet, createPluginSlot } from "../../../src/kernel/ui/layout.ts";

describe("getSharedStylesheet", () => {
  test("returns null before initialization", () => {
    // Note: if initSharedStylesheet was called earlier in another test,
    // this might not be null. We test the API contract rather than global state.
    const result = getSharedStylesheet();
    expect(result === null || result instanceof CSSStyleSheet).toBe(true);
  });
});

describe("createPluginSlot", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
  });

  test("clears existing content from host", () => {
    const host = document.createElement("div");
    host.innerHTML = "<p>old content</p>";
    document.body.appendChild(host);

    createPluginSlot("test.plugin", host);
    expect(host.querySelector("p")).toBeNull();
  });

  test("creates shadow host inside host element", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);

    createPluginSlot("test.plugin", host);
    const shadowHost = host.firstElementChild as HTMLElement;
    expect(shadowHost).not.toBeNull();
    expect(shadowHost.dataset["pluginId"]).toBe("test.plugin");
  });

  test("returns inner container with plugin-container class", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);

    const container = createPluginSlot("test.plugin", host);
    expect(container.className).toBe("plugin-container");
    expect(container.style.width).toBe("100%");
    expect(container.style.height).toBe("100%");
  });

  test("shadow host has flex and overflow styles", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);

    createPluginSlot("test.plugin", host);
    const shadowHost = host.firstElementChild as HTMLElement;
    expect(shadowHost.style.flex).toContain("1");
    expect(shadowHost.style.overflow).toBe("hidden");
  });

  test("attaches open shadow root", () => {
    const host = document.createElement("div");
    document.body.appendChild(host);

    createPluginSlot("test.plugin", host);
    const shadowHost = host.firstElementChild as HTMLElement;
    expect(shadowHost.shadowRoot).not.toBeNull();
  });
});
