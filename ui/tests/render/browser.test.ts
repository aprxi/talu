import { describe, test, expect } from "bun:test";
import { renderBrowserCard } from "../../src/render/browser.ts";
import type { Conversation, MessageItem, InputTextPart } from "../../src/types.ts";

/**
 * Tests for browser card rendering — extractPreview, renderHighlightedText,
 * findSnippetInItems, and the full renderBrowserCard composition.
 */

function makeConvo(overrides: Partial<Conversation> = {}): Conversation {
  return {
    id: "conv-1",
    object: "conversation",
    title: "Test Chat",
    created_at: 1700000000,
    updated_at: 1700000000,
    model: "gpt-4",
    items: [],
    metadata: {},
    ...overrides,
  } as Conversation;
}

function userMsg(text: string): MessageItem {
  return {
    type: "message",
    role: "user",
    content: [{ type: "input_text", text } as InputTextPart],
  } as MessageItem;
}

function assistantMsg(text: string): MessageItem {
  return {
    type: "message",
    role: "assistant",
    content: [{ type: "output_text", text } as any],
  } as MessageItem;
}

// ── Basic rendering ──────────────────────────────────────────────────────────

describe("renderBrowserCard — basic", () => {
  test("renders card with title", () => {
    const card = renderBrowserCard(makeConvo({ title: "My Chat" }), false);
    expect(card.querySelector(".browser-card-title")!.textContent).toBe("My Chat");
  });

  test("untitled conversation shows 'Untitled'", () => {
    const card = renderBrowserCard(makeConvo({ title: "" }), false);
    expect(card.querySelector(".browser-card-title")!.textContent).toBe("Untitled");
  });

  test("selected card has selected class", () => {
    const card = renderBrowserCard(makeConvo(), true);
    expect(card.classList.contains("selected")).toBe(true);
  });

  test("checkbox reflects selection state", () => {
    const card = renderBrowserCard(makeConvo(), true);
    const cb = card.querySelector<HTMLInputElement>(".browser-checkbox")!;
    expect(cb.checked).toBe(true);
  });

  test("model badge rendered when model present", () => {
    const card = renderBrowserCard(makeConvo({ model: "llama-3" }), false);
    const badge = card.querySelector(".browser-card-badge");
    expect(badge!.textContent).toBe("llama-3");
  });

  test("data-id attribute set", () => {
    const card = renderBrowserCard(makeConvo({ id: "xyz" }), false);
    expect(card.dataset["id"]).toBe("xyz");
  });
});

// ── Preview text ─────────────────────────────────────────────────────────────

describe("renderBrowserCard — preview text", () => {
  test("extracts preview from user message", () => {
    const items = [userMsg("Hello world, this is a test")];
    const card = renderBrowserCard(makeConvo({ items }), false);
    const preview = card.querySelector(".browser-card-preview");
    expect(preview!.textContent).toContain("Hello world");
  });

  test("truncates long preview at ~120 chars", () => {
    const longText = "a".repeat(200);
    const items = [userMsg(longText)];
    const card = renderBrowserCard(makeConvo({ items }), false);
    const preview = card.querySelector(".browser-card-preview")!;
    expect(preview.textContent!.length).toBeLessThan(130);
    expect(preview.textContent!.endsWith("\u2026")).toBe(true);
  });

  test("no preview when no items", () => {
    const card = renderBrowserCard(makeConvo({ items: [] }), false);
    const preview = card.querySelector(".browser-card-preview");
    expect(preview).toBeNull();
  });
});

// ── Search highlighting ──────────────────────────────────────────────────────

describe("renderBrowserCard — search highlighting", () => {
  test("title wraps query match in <mark>", () => {
    const card = renderBrowserCard(makeConvo({ title: "Rust Guide" }), false, "rust");
    const titleSpan = card.querySelector(".browser-card-title")!;
    const mark = titleSpan.querySelector("mark");
    expect(mark).not.toBeNull();
    expect(mark!.textContent!.toLowerCase()).toBe("rust");
  });

  test("preview highlights query in snippet", () => {
    const items = [userMsg("Tell me about Rust programming")];
    const card = renderBrowserCard(makeConvo({ items }), false, "Rust");
    const preview = card.querySelector(".browser-card-preview")!;
    const mark = preview.querySelector("mark");
    expect(mark).not.toBeNull();
    expect(mark!.textContent).toBe("Rust");
  });

  test("search_snippet preferred over item extraction", () => {
    const conv = makeConvo({
      items: [userMsg("some text")],
      search_snippet: "this is the server snippet with Rust",
    } as any);
    const card = renderBrowserCard(conv, false, "Rust");
    const preview = card.querySelector(".browser-card-preview")!;
    expect(preview.textContent).toContain("server snippet");
  });
});

// ── Tags ─────────────────────────────────────────────────────────────────────

describe("renderBrowserCard — tags", () => {
  test("renders tag chips", () => {
    const conv = makeConvo({ tags: [{ name: "rust" }, { name: "wasm" }] } as any);
    const card = renderBrowserCard(conv, false);
    const chips = card.querySelectorAll(".tag-chip");
    expect(chips.length).toBe(2);
  });

  test("active tag filter gets active class", () => {
    const conv = makeConvo({ tags: [{ name: "rust" }, { name: "python" }] } as any);
    const card = renderBrowserCard(conv, false, undefined, ["rust"]);
    const chips = card.querySelectorAll(".tag-chip");
    const activeChip = Array.from(chips).find((c) => c.textContent === "rust")!;
    expect(activeChip.classList.contains("active")).toBe(true);
  });

  test("non-active tags dimmed when filter active", () => {
    const conv = makeConvo({ tags: [{ name: "rust" }, { name: "python" }] } as any);
    const card = renderBrowserCard(conv, false, undefined, ["rust"]);
    const chips = card.querySelectorAll(".tag-chip");
    const dimmedChip = Array.from(chips).find((c) => c.textContent === "python")!;
    expect(dimmedChip.classList.contains("dimmed")).toBe(true);
  });
});

// ── Message count ────────────────────────────────────────────────────────────

describe("renderBrowserCard — meta row", () => {
  test("shows message count for multi-message conversations", () => {
    const items = [userMsg("hi"), assistantMsg("hello"), userMsg("bye")];
    const card = renderBrowserCard(makeConvo({ items }), false);
    const meta = card.querySelector(".browser-card-meta")!;
    expect(meta.textContent).toContain("3 messages");
  });

  test("singular message label", () => {
    const items = [userMsg("hi")];
    const card = renderBrowserCard(makeConvo({ items }), false);
    const meta = card.querySelector(".browser-card-meta")!;
    expect(meta.textContent).toContain("1 message");
    expect(meta.textContent).not.toContain("1 messages");
  });
});

// ── Forked / Archived status ─────────────────────────────────────────────────

describe("renderBrowserCard — status", () => {
  test("forked conversation has forked class", () => {
    const card = renderBrowserCard(makeConvo({ parent_session_id: "parent-1" } as any), false);
    expect(card.classList.contains("forked")).toBe(true);
  });

  test("archived badge shown when showStatusBadge is true", () => {
    const conv = makeConvo({ marker: "archived" });
    const card = renderBrowserCard(conv, false, undefined, [], true);
    const badge = card.querySelector(".browser-card-badge.archived");
    expect(badge).not.toBeNull();
    expect(badge!.textContent).toBe("Archived");
  });

  test("archived badge hidden when showStatusBadge is false", () => {
    const conv = makeConvo({ marker: "archived" });
    const card = renderBrowserCard(conv, false, undefined, [], false);
    const badge = card.querySelector(".browser-card-badge.archived");
    expect(badge).toBeNull();
  });
});
