import { describe, test, expect } from "bun:test";
import {
  MAX_TIMERS,
  MAX_OBSERVERS,
  MAX_STORAGE_KEYS,
  MAX_STORAGE_BYTES,
  MAX_DOCUMENT_BYTES,
  MAX_TOOL_RESULT_BYTES,
  MAX_RENDERER_DOM_NODES,
  truncateToolResult,
  checkRendererNodeCap,
} from "../../../src/kernel/security/resource-caps.ts";

describe("resource cap constants", () => {
  test("cap values are positive numbers", () => {
    for (const cap of [MAX_TIMERS, MAX_OBSERVERS, MAX_STORAGE_KEYS, MAX_STORAGE_BYTES, MAX_DOCUMENT_BYTES, MAX_TOOL_RESULT_BYTES, MAX_RENDERER_DOM_NODES]) {
      expect(cap).toBeGreaterThan(0);
    }
  });

  test("storage document limit is less than total storage limit", () => {
    expect(MAX_DOCUMENT_BYTES).toBeLessThan(MAX_STORAGE_BYTES);
  });
});

describe("truncateToolResult", () => {
  test("returns original string when under limit", () => {
    const text = "short result";
    expect(truncateToolResult(text)).toBe(text);
  });

  test("returns original string at exact limit", () => {
    const text = "x".repeat(MAX_TOOL_RESULT_BYTES);
    expect(truncateToolResult(text)).toBe(text);
  });

  test("truncates string over limit with message", () => {
    const text = "x".repeat(MAX_TOOL_RESULT_BYTES + 100);
    const result = truncateToolResult(text);
    expect(result.length).toBeLessThan(text.length);
    expect(result).toContain("truncated");
    expect(result).toContain(String(text.length));
  });
});

describe("checkRendererNodeCap", () => {
  test("returns 0 when within limit", () => {
    const container = document.createElement("div");
    container.attachShadow({ mode: "open" });
    container.shadowRoot!.innerHTML = "<p>hello</p><span>world</span>";
    expect(checkRendererNodeCap(container.shadowRoot!)).toBe(0);
  });

  test("returns node count when exceeding limit", () => {
    const container = document.createElement("div");
    container.attachShadow({ mode: "open" });
    const root = container.shadowRoot!;
    // Create nodes beyond the limit.
    const wrapper = document.createElement("div");
    for (let i = 0; i < MAX_RENDERER_DOM_NODES + 1; i++) {
      wrapper.appendChild(document.createElement("span"));
    }
    root.appendChild(wrapper);
    const count = checkRendererNodeCap(root);
    expect(count).toBeGreaterThan(MAX_RENDERER_DOM_NODES);
  });
});
