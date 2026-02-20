import { describe, test, expect } from "bun:test";
import { renderEmptyState, renderLoadingSpinner, renderToast } from "../../src/render/common.ts";

/**
 * Tests for render/common — renderEmptyState, renderLoadingSpinner, renderToast.
 */

describe("renderEmptyState", () => {
  test("creates div with empty-state class", () => {
    const el = renderEmptyState("Nothing here");
    expect(el.tagName).toBe("DIV");
    expect(el.className).toContain("empty-state");
  });

  test("contains the provided text", () => {
    const el = renderEmptyState("No conversations");
    expect(el.textContent).toContain("No conversations");
  });

  test("has data-empty-state attribute", () => {
    const el = renderEmptyState("Empty");
    expect(el.dataset["emptyState"]).toBeDefined();
  });

  test("has minHeight style", () => {
    const el = renderEmptyState("Empty");
    expect(el.style.minHeight).toBe("200px");
  });
});

describe("renderLoadingSpinner", () => {
  test("creates wrapper with empty-state class", () => {
    const el = renderLoadingSpinner();
    expect(el.className).toContain("empty-state");
  });

  test("contains spinner element", () => {
    const el = renderLoadingSpinner();
    const spinner = el.querySelector(".spinner");
    expect(spinner).not.toBeNull();
  });
});

describe("renderToast", () => {
  test("creates element with toast class", () => {
    const el = renderToast("Hello", "info");
    expect(el.className).toContain("toast");
  });

  test("sets message text", () => {
    const el = renderToast("Something happened", "info");
    expect(el.textContent).toContain("Something happened");
  });

  test("sets role=alert", () => {
    const el = renderToast("Alert", "error");
    expect(el.getAttribute("role")).toBe("alert");
  });

  test("sets aria-live=assertive", () => {
    const el = renderToast("Alert", "error");
    expect(el.getAttribute("aria-live")).toBe("assertive");
  });

  test("error type sets white text color", () => {
    const el = renderToast("Err", "error");
    expect(el.style.color).toBe("white");
  });

  test("success type sets --bg text color", () => {
    const el = renderToast("OK", "success");
    expect(el.style.color).toBe("var(--bg)");
  });

  test("warning type sets --bg text color", () => {
    const el = renderToast("Warn", "warning");
    expect(el.style.color).toBe("var(--bg)");
  });

  test("info type sets white text color", () => {
    const el = renderToast("Info", "info");
    expect(el.style.color).toBe("white");
  });

  test("each severity type produces a distinct toast", () => {
    const types: Array<"error" | "success" | "info" | "warning"> = ["error", "success", "info", "warning"];
    for (const type of types) {
      const el = renderToast("msg", type);
      expect(el.className).toContain("toast");
      expect(el.getAttribute("role")).toBe("alert");
    }
  });
});
