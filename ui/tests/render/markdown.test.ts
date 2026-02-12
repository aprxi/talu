import { describe, test, expect } from "bun:test";
import { sanitizedMarkdown } from "../../src/render/markdown.ts";

/**
 * Tests for sanitizedMarkdown — security-critical since it processes
 * untrusted LLM output that could contain XSS vectors.
 */

// ── Basic markdown rendering ────────────────────────────────────────────────

describe("sanitizedMarkdown — basic rendering", () => {
  test("renders plain text as paragraph", () => {
    const html = sanitizedMarkdown("Hello world");
    expect(html).toContain("Hello world");
    expect(html).toContain("<p>");
  });

  test("renders bold text", () => {
    const html = sanitizedMarkdown("**bold**");
    expect(html).toContain("<strong>bold</strong>");
  });

  test("renders italic text", () => {
    const html = sanitizedMarkdown("*italic*");
    expect(html).toContain("<em>italic</em>");
  });

  test("renders links", () => {
    const html = sanitizedMarkdown("[example](https://example.com)");
    expect(html).toContain('href="https://example.com"');
    expect(html).toContain("example");
  });

  test("renders unordered lists", () => {
    const html = sanitizedMarkdown("- item 1\n- item 2");
    expect(html).toContain("<li>");
    expect(html).toContain("item 1");
    expect(html).toContain("item 2");
  });

  test("renders headings", () => {
    const html = sanitizedMarkdown("# Heading 1\n## Heading 2");
    expect(html).toContain("<h1>");
    expect(html).toContain("Heading 1");
    expect(html).toContain("<h2>");
  });

  test("empty string returns empty string", () => {
    const html = sanitizedMarkdown("");
    expect(html).toBe("");
  });
});

// ── Code block structure ────────────────────────────────────────────────────

describe("sanitizedMarkdown — code blocks", () => {
  test("wraps code in .code-block div", () => {
    const html = sanitizedMarkdown("```js\nconsole.log('hi');\n```");
    expect(html).toContain('class="code-block"');
  });

  test("includes data-code attribute with raw code", () => {
    const html = sanitizedMarkdown("```\nraw code\n```");
    expect(html).toContain("data-code=");
    expect(html).toContain("raw code");
  });

  test("includes copy button", () => {
    const html = sanitizedMarkdown("```python\nprint('hello')\n```");
    expect(html).toContain('class="code-copy"');
    expect(html).toContain("Copy code");
  });

  test("applies language class", () => {
    const html = sanitizedMarkdown("```typescript\nconst x = 1;\n```");
    expect(html).toContain('class="language-typescript"');
  });

  test("shows language label in header", () => {
    const html = sanitizedMarkdown("```python\npass\n```");
    expect(html).toContain('class="code-lang"');
    expect(html).toContain("python");
  });

  test("inline code is not wrapped in code-block", () => {
    const html = sanitizedMarkdown("Use `inline` code");
    expect(html).toContain("<code>inline</code>");
    expect(html).not.toContain("code-block");
  });

  test("escapes quotes and angle brackets in data-code attribute", () => {
    const code = 'if (a < b && c > d) { x = "y"; }';
    const html = sanitizedMarkdown("```\n" + code + "\n```");
    // data-code uses escapeHtml — must not contain raw " or < or > inside the attribute.
    expect(html).toContain("data-code=");
    expect(html).not.toContain('data-code="if (a < b');
    // The escaped form should contain &lt; &gt; &quot;
    expect(html).toContain("&lt;");
    expect(html).toContain("&gt;");
    expect(html).toContain("&quot;");
  });

  test("code block without language uses 'code' as default label", () => {
    const html = sanitizedMarkdown("```\nhello\n```");
    expect(html).toContain('class="language-code"');
  });
});

// ── XSS sanitization ────────────────────────────────────────────────────────

describe("sanitizedMarkdown — XSS prevention", () => {
  test("strips <script> tags", () => {
    const html = sanitizedMarkdown("<script>alert('xss')</script>");
    expect(html).not.toContain("<script>");
    expect(html).not.toContain("alert");
  });

  test("strips onerror attributes", () => {
    const html = sanitizedMarkdown('<img src=x onerror="alert(1)">');
    expect(html).not.toContain("onerror");
    expect(html).not.toContain("alert");
  });

  test("strips onload attributes", () => {
    const html = sanitizedMarkdown('<body onload="alert(1)">');
    expect(html).not.toContain("onload");
  });

  test("strips javascript: URLs in links", () => {
    const html = sanitizedMarkdown('[click](javascript:alert(1))');
    expect(html).not.toContain("javascript:");
  });

  test("strips event handler attributes", () => {
    const html = sanitizedMarkdown('<div onclick="alert(1)">click me</div>');
    expect(html).not.toContain("onclick");
  });

  test("strips iframe tags", () => {
    const html = sanitizedMarkdown('<iframe src="https://evil.com"></iframe>');
    expect(html).not.toContain("<iframe");
  });

  test("preserves safe HTML content after stripping", () => {
    const html = sanitizedMarkdown("Safe text <script>bad</script> more text");
    expect(html).toContain("Safe text");
    expect(html).toContain("more text");
    expect(html).not.toContain("<script>");
  });
});
