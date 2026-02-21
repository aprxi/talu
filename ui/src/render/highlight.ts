/**
 * Runtime syntax highlighting via the tree-sitter highlight API.
 *
 * Progressive enhancement: code renders as plain text immediately, then
 * `highlightCodeBlocks()` asynchronously fetches tokens and applies
 * `<span class="syntax-*">` overlays.  Failures are silently ignored
 * (plain text is always the fallback).
 */

import { escapeHtml } from "../utils/helpers.ts";

interface HighlightToken {
  /** Start byte offset. */
  s: number;
  /** End byte offset. */
  e: number;
  /** CSS class (e.g. "syntax-keyword"). */
  t: string;
}

/**
 * Apply syntax highlighting to all unhighlighted `<code>` elements within root.
 *
 * Finds elements with a `language-*` class that haven't been highlighted yet,
 * calls `POST /v1/code/highlight` for each, and replaces the element's innerHTML
 * with token spans.  Fires all requests in parallel.
 */
export function highlightCodeBlocks(root: HTMLElement): void {
  const codeEls = root.querySelectorAll<HTMLElement>(
    'code[class*="language-"]:not([data-highlighted])',
  );
  if (codeEls.length === 0) return;

  for (const codeEl of codeEls) {
    const lang = extractLanguage(codeEl);
    if (!lang) continue;
    highlightElement(codeEl, lang);
  }
}

const LANG_RE = /language-(\S+)/;

function extractLanguage(codeEl: HTMLElement): string | null {
  const match = codeEl.className.match(LANG_RE);
  return match ? match[1]! : null;
}

async function highlightElement(codeEl: HTMLElement, language: string): Promise<void> {
  const source = codeEl.textContent ?? "";
  if (!source) return;

  // Mark immediately so concurrent calls don't double-highlight.
  codeEl.dataset["highlighted"] = "pending";

  try {
    const resp = await fetch("/v1/code/highlight", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, language }),
    });

    if (!resp.ok) {
      delete codeEl.dataset["highlighted"];
      return;
    }

    const tokens: HighlightToken[] = await resp.json();
    applyTokens(codeEl, source, tokens);
    codeEl.dataset["highlighted"] = "true";
  } catch {
    delete codeEl.dataset["highlighted"];
  }
}

/**
 * Build highlighted innerHTML from token spans.
 *
 * Tokens are byte-offset ranges with a CSS class.  Gaps between tokens
 * (and after the last token) are emitted as plain escaped text.
 */
function applyTokens(codeEl: HTMLElement, source: string, tokens: HighlightToken[]): void {
  if (tokens.length === 0) return;

  const parts: string[] = [];
  let cursor = 0;

  for (const token of tokens) {
    // Gap before this token — plain text.
    if (token.s > cursor) {
      parts.push(escapeHtml(source.slice(cursor, token.s)));
    }

    const text = escapeHtml(source.slice(token.s, token.e));

    if (token.t && token.t !== "syntax-plain") {
      parts.push(`<span class="${token.t}">${text}</span>`);
    } else {
      parts.push(text);
    }

    cursor = token.e;
  }

  // Remaining text after last token.
  if (cursor < source.length) {
    parts.push(escapeHtml(source.slice(cursor)));
  }

  codeEl.innerHTML = parts.join("");
}
