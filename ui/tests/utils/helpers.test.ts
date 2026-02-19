import { describe, test, expect } from "bun:test";
import { escapeHtml } from "../../src/utils/helpers.ts";

/**
 * Tests for escapeHtml — XSS defense for HTML special characters.
 */

describe("escapeHtml", () => {
  test("escapes ampersand", () => {
    expect(escapeHtml("a&b")).toBe("a&amp;b");
  });

  test("escapes less-than", () => {
    expect(escapeHtml("a<b")).toBe("a&lt;b");
  });

  test("escapes greater-than", () => {
    expect(escapeHtml("a>b")).toBe("a&gt;b");
  });

  test("escapes double quote", () => {
    expect(escapeHtml('a"b')).toBe("a&quot;b");
  });

  test("escapes all special characters together", () => {
    expect(escapeHtml('<script>"alert&</script>')).toBe(
      "&lt;script&gt;&quot;alert&amp;&lt;/script&gt;",
    );
  });

  test("returns empty string unchanged", () => {
    expect(escapeHtml("")).toBe("");
  });

  test("returns plain text unchanged", () => {
    expect(escapeHtml("hello world")).toBe("hello world");
  });

  test("handles multiple occurrences of same character", () => {
    expect(escapeHtml("<<>>")).toBe("&lt;&lt;&gt;&gt;");
  });
});
