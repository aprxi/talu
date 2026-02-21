import { describe, test, expect, beforeEach, afterEach, spyOn } from "bun:test";
import { highlightCodeBlocks } from "../../src/render/highlight.ts";
import { flushAsync } from "../helpers/mocks.ts";

/**
 * Tests for runtime syntax highlighting via the tree-sitter highlight API.
 *
 * Strategy: mock globalThis.fetch to return canned token arrays, call
 * highlightCodeBlocks(), verify the resulting DOM contains the expected
 * <span class="syntax-*"> elements.
 *
 * highlightCodeBlocks() is fire-and-forget (returns void, launches async
 * work internally).  flushAsync() drains the microtask queue so all
 * mocked fetch → json → applyTokens chains complete before assertions.
 */

let fetchSpy: ReturnType<typeof spyOn>;
let container: HTMLElement;

function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    statusText: status === 200 ? "OK" : "Error",
    headers: { "Content-Type": "application/json" },
  });
}

beforeEach(() => {
  container = document.createElement("div");
  fetchSpy = spyOn(globalThis, "fetch");
});

afterEach(() => {
  fetchSpy.mockRestore();
});

// -- Helpers -----------------------------------------------------------------

/** Create a <pre><code class="language-{lang}">source</code></pre> inside container. */
function addCodeBlock(lang: string, source: string): HTMLElement {
  const pre = document.createElement("pre");
  const code = document.createElement("code");
  code.className = `language-${lang}`;
  code.textContent = source;
  pre.appendChild(code);
  container.appendChild(pre);
  return code;
}

// ── Token application ───────────────────────────────────────────────────────

describe("highlightCodeBlocks — token application", () => {
  test("applies syntax-* spans from highlight tokens", async () => {
    const code = addCodeBlock("python", "def foo():");
    fetchSpy.mockResolvedValueOnce(jsonResponse([
      { s: 0, e: 3, t: "syntax-keyword" },
      { s: 4, e: 7, t: "syntax-function" },
      { s: 7, e: 8, t: "syntax-punctuation" },
      { s: 8, e: 9, t: "syntax-punctuation" },
      { s: 9, e: 10, t: "syntax-punctuation" },
    ]));

    highlightCodeBlocks(container);
    await flushAsync();

    expect(code.querySelector(".syntax-keyword")!.textContent).toBe("def");
    expect(code.querySelector(".syntax-function")!.textContent).toBe("foo");
    expect(code.dataset["highlighted"]).toBe("true");
  });

  test("preserves plain text in gaps between tokens", async () => {
    // Source: "a + b" — tokens only cover the operator.
    // "a " before and " b" after must appear as plain unspanned text.
    const code = addCodeBlock("python", "a + b");
    fetchSpy.mockResolvedValueOnce(jsonResponse([
      { s: 2, e: 3, t: "syntax-operator" },
    ]));

    highlightCodeBlocks(container);
    await flushAsync();

    const html = code.innerHTML;
    // Leading gap: "a " (bytes 0-2)
    expect(html).toMatch(/^a /);
    // Token span
    expect(html).toContain('<span class="syntax-operator">+</span>');
    // Trailing gap: " b" (bytes 3-5)
    expect(html).toMatch(/ b$/);
    // Only 1 span total
    expect(code.querySelectorAll("span").length).toBe(1);
  });

  test("renders trailing text after last token", async () => {
    // Source: "x = 1\n" — tokens cover "=" and "1" but not the trailing newline.
    const code = addCodeBlock("python", "x = 1\n");
    fetchSpy.mockResolvedValueOnce(jsonResponse([
      { s: 2, e: 3, t: "syntax-operator" },
      { s: 4, e: 5, t: "syntax-number" },
    ]));

    highlightCodeBlocks(container);
    await flushAsync();

    // textContent should preserve the full source including trailing newline
    expect(code.textContent).toBe("x = 1\n");
    expect(code.querySelectorAll("span").length).toBe(2);
  });

  test("syntax-plain tokens are not wrapped in spans", async () => {
    const code = addCodeBlock("python", "x");
    fetchSpy.mockResolvedValueOnce(jsonResponse([
      { s: 0, e: 1, t: "syntax-plain" },
    ]));

    highlightCodeBlocks(container);
    await flushAsync();

    expect(code.innerHTML).toBe("x");
    expect(code.querySelectorAll("span").length).toBe(0);
  });

  test("empty token array leaves code unchanged", async () => {
    const code = addCodeBlock("python", "pass");
    fetchSpy.mockResolvedValueOnce(jsonResponse([]));

    highlightCodeBlocks(container);
    await flushAsync();

    expect(code.textContent).toBe("pass");
    expect(code.querySelectorAll("span").length).toBe(0);
    // Still marked as highlighted (API succeeded, just no tokens).
    expect(code.dataset["highlighted"]).toBe("true");
  });

  test("escapes HTML entities in token text", async () => {
    const code = addCodeBlock("python", 'x = "<br>"');
    fetchSpy.mockResolvedValueOnce(jsonResponse([
      { s: 4, e: 10, t: "syntax-string" },
    ]));

    highlightCodeBlocks(container);
    await flushAsync();

    const span = code.querySelector(".syntax-string")!;
    // textContent gives decoded text; innerHTML should have escaped entities
    expect(span.textContent).toBe('"<br>"');
    expect(span.innerHTML).toContain("&lt;br&gt;");
  });

  test("escapes HTML entities in gap text too", async () => {
    // The gap text (not covered by any token) must also be escaped.
    const code = addCodeBlock("html", "<div>hi</div>");
    fetchSpy.mockResolvedValueOnce(jsonResponse([]));

    highlightCodeBlocks(container);
    await flushAsync();

    // Empty tokens → no innerHTML change (applyTokens returns early).
    // textContent should still be the original source.
    expect(code.textContent).toBe("<div>hi</div>");
  });
});

// ── Element selection ───────────────────────────────────────────────────────

describe("highlightCodeBlocks — element selection", () => {
  test("skips code elements without language-* class", async () => {
    const pre = document.createElement("pre");
    const code = document.createElement("code");
    code.textContent = "no language";
    pre.appendChild(code);
    container.appendChild(pre);

    highlightCodeBlocks(container);
    await flushAsync();

    expect(fetchSpy).not.toHaveBeenCalled();
  });

  test("skips already-highlighted elements", async () => {
    const code = addCodeBlock("python", "pass");
    code.dataset["highlighted"] = "true";

    highlightCodeBlocks(container);
    await flushAsync();

    expect(fetchSpy).not.toHaveBeenCalled();
  });

  test("skips elements with pending highlight", async () => {
    const code = addCodeBlock("python", "pass");
    code.dataset["highlighted"] = "pending";

    highlightCodeBlocks(container);
    await flushAsync();

    expect(fetchSpy).not.toHaveBeenCalled();
  });

  test("skips empty code elements without calling fetch", async () => {
    // Empty source should be detected before any API call.
    const pre = document.createElement("pre");
    const code = document.createElement("code");
    code.className = "language-python";
    code.textContent = "";
    pre.appendChild(code);
    container.appendChild(pre);

    highlightCodeBlocks(container);
    await flushAsync();

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(code.dataset["highlighted"]).toBeUndefined();
  });

  test("highlights multiple code blocks in parallel", async () => {
    addCodeBlock("python", "x = 1");
    addCodeBlock("javascript", "let y = 2");

    // Must create a fresh Response per call — Response.json() consumes the body stream.
    fetchSpy.mockImplementation(() =>
      Promise.resolve(jsonResponse([{ s: 0, e: 1, t: "syntax-keyword" }])),
    );

    highlightCodeBlocks(container);
    await flushAsync();

    expect(fetchSpy).toHaveBeenCalledTimes(2);
    // Both blocks should be marked highlighted.
    const codes = container.querySelectorAll("code");
    expect(codes[0]!.dataset["highlighted"]).toBe("true");
    expect(codes[1]!.dataset["highlighted"]).toBe("true");
  });

  test("sends POST with correct URL, method, headers, and body", async () => {
    addCodeBlock("rust", "fn main() {}");
    fetchSpy.mockResolvedValueOnce(jsonResponse([]));

    highlightCodeBlocks(container);
    await flushAsync();

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url, init] = fetchSpy.mock.calls[0]!;
    expect(url).toBe("/v1/code/highlight");
    expect(init.method).toBe("POST");
    expect(init.headers).toEqual({ "Content-Type": "application/json" });
    const body = JSON.parse(init.body as string);
    expect(body.source).toBe("fn main() {}");
    expect(body.language).toBe("rust");
  });
});

// ── Graceful degradation ────────────────────────────────────────────────────

describe("highlightCodeBlocks — error handling", () => {
  test("leaves plain text on HTTP error and clears pending state", async () => {
    const code = addCodeBlock("python", "pass");
    fetchSpy.mockResolvedValueOnce(jsonResponse({ error: "bad" }, 400));

    highlightCodeBlocks(container);
    await flushAsync();

    expect(code.textContent).toBe("pass");
    expect(code.querySelectorAll("span").length).toBe(0);
    // data-highlighted must be removed so a retry can succeed later.
    expect(code.dataset["highlighted"]).toBeUndefined();
  });

  test("leaves plain text on network error and clears pending state", async () => {
    const code = addCodeBlock("python", "pass");
    fetchSpy.mockRejectedValueOnce(new Error("network down"));

    highlightCodeBlocks(container);
    await flushAsync();

    expect(code.textContent).toBe("pass");
    expect(code.querySelectorAll("span").length).toBe(0);
    expect(code.dataset["highlighted"]).toBeUndefined();
  });

  test("does not double-highlight on concurrent calls", async () => {
    const code = addCodeBlock("python", "x = 1");
    fetchSpy.mockImplementation(() =>
      Promise.resolve(jsonResponse([{ s: 0, e: 1, t: "syntax-keyword" }])),
    );

    // highlightElement() sets data-highlighted="pending" synchronously
    // (before its first await), so the second highlightCodeBlocks() call
    // sees the attribute and the :not([data-highlighted]) selector excludes it.
    highlightCodeBlocks(container);
    // Verify pending state is set synchronously before any await resolves.
    expect(code.dataset["highlighted"]).toBe("pending");
    highlightCodeBlocks(container);
    await flushAsync();

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(code.dataset["highlighted"]).toBe("true");
  });
});

// ── Copy button safety ──────────────────────────────────────────────────────

describe("highlightCodeBlocks — copy button safety", () => {
  test("textContent still returns plain text after highlighting", async () => {
    const code = addCodeBlock("python", "def foo(): pass");
    fetchSpy.mockResolvedValueOnce(jsonResponse([
      { s: 0, e: 3, t: "syntax-keyword" },
      { s: 4, e: 7, t: "syntax-function" },
      { s: 7, e: 8, t: "syntax-punctuation" },
      { s: 8, e: 9, t: "syntax-punctuation" },
      { s: 9, e: 10, t: "syntax-punctuation" },
      { s: 11, e: 15, t: "syntax-keyword" },
    ]));

    highlightCodeBlocks(container);
    await flushAsync();

    // innerHTML has spans, but textContent strips them — copy button reads textContent.
    expect(code.innerHTML).toContain("<span");
    expect(code.textContent).toBe("def foo(): pass");
  });
});
