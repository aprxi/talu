import { describe, test, expect } from "bun:test";
import { renderSidebarItem, renderSectionLabel } from "../../src/render/sidebar.ts";
import type { Conversation } from "../../src/types.ts";

/**
 * Tests for sidebar item rendering — title, active state, icons (pin/fork),
 * model badge, tags (max 2 + overflow), pin button, and section labels.
 */

function makeSession(overrides: Partial<Conversation> = {}): Conversation {
  return {
    id: "sess-1",
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

// ── Basic rendering ──────────────────────────────────────────────────────────

describe("renderSidebarItem — basic", () => {
  test("renders title text", () => {
    const el = renderSidebarItem(makeSession({ title: "My Chat" }), false);
    expect(el.querySelector(".sidebar-item-title")!.textContent).toBe("My Chat");
  });

  test("untitled conversation shows 'Untitled'", () => {
    const el = renderSidebarItem(makeSession({ title: "" }), false);
    expect(el.querySelector(".sidebar-item-title")!.textContent).toBe("Untitled");
  });

  test("data-id attribute set", () => {
    const el = renderSidebarItem(makeSession({ id: "abc-123" }), false);
    expect(el.dataset["id"]).toBe("abc-123");
  });

  test("active session has active class", () => {
    const el = renderSidebarItem(makeSession(), true);
    expect(el.classList.contains("active")).toBe(true);
  });

  test("inactive session does not have active class", () => {
    const el = renderSidebarItem(makeSession(), false);
    expect(el.classList.contains("active")).toBe(false);
  });
});

// ── Model and time ──────────────────────────────────────────────────────────

describe("renderSidebarItem — meta", () => {
  test("shows model in meta row", () => {
    const el = renderSidebarItem(makeSession({ model: "llama-3" }), false);
    const meta = el.querySelector(".sidebar-item-meta")!;
    expect(meta.textContent).toContain("llama-3");
  });

  test("meta row has relative time", () => {
    const el = renderSidebarItem(makeSession(), false);
    const meta = el.querySelector(".sidebar-item-meta")!;
    // relativeTime returns something like "2y" — just check it exists
    expect(meta.children.length).toBeGreaterThanOrEqual(1);
  });
});

// ── Pin icon and button ─────────────────────────────────────────────────────

describe("renderSidebarItem — pinned", () => {
  test("pinned session shows pin icon in title row", () => {
    const el = renderSidebarItem(makeSession({ marker: "pinned" }), false);
    expect(el.querySelector(".icon-pin")).not.toBeNull();
  });

  test("unpinned session has no pin icon in title row", () => {
    const el = renderSidebarItem(makeSession(), false);
    expect(el.querySelector(".sidebar-item-title-row .icon-pin")).toBeNull();
  });

  test("pin button has data-pin attribute", () => {
    const el = renderSidebarItem(makeSession({ id: "xyz" }), false);
    const pinBtn = el.querySelector<HTMLButtonElement>(".pin-btn")!;
    expect(pinBtn.dataset["pin"]).toBe("xyz");
  });

  test("pinned session pin button has pinned class", () => {
    const el = renderSidebarItem(makeSession({ marker: "pinned" }), false);
    const pinBtn = el.querySelector<HTMLButtonElement>(".pin-btn")!;
    expect(pinBtn.classList.contains("pinned")).toBe(true);
  });

  test("unpinned session pin button lacks pinned class", () => {
    const el = renderSidebarItem(makeSession(), false);
    const pinBtn = el.querySelector<HTMLButtonElement>(".pin-btn")!;
    expect(pinBtn.classList.contains("pinned")).toBe(false);
  });

  test("pinned pin button title is 'Unpin'", () => {
    const el = renderSidebarItem(makeSession({ marker: "pinned" }), false);
    const pinBtn = el.querySelector<HTMLButtonElement>(".pin-btn")!;
    expect(pinBtn.title).toBe("Unpin");
  });

  test("unpinned pin button title is 'Pin'", () => {
    const el = renderSidebarItem(makeSession(), false);
    const pinBtn = el.querySelector<HTMLButtonElement>(".pin-btn")!;
    expect(pinBtn.title).toBe("Pin");
  });
});

// ── Forked ──────────────────────────────────────────────────────────────────

describe("renderSidebarItem — forked", () => {
  test("forked session has forked class", () => {
    const el = renderSidebarItem(makeSession({ parent_session_id: "parent-1" } as any), false);
    expect(el.classList.contains("forked")).toBe(true);
  });

  test("forked session shows fork icon", () => {
    const el = renderSidebarItem(makeSession({ parent_session_id: "parent-1" } as any), false);
    expect(el.querySelector(".icon-fork")).not.toBeNull();
  });

  test("non-forked session has no forked class", () => {
    const el = renderSidebarItem(makeSession(), false);
    expect(el.classList.contains("forked")).toBe(false);
  });
});

// ── Tags ────────────────────────────────────────────────────────────────────

describe("renderSidebarItem — tags", () => {
  test("renders up to 2 tags", () => {
    const session = makeSession({ tags: [{ name: "rust" }, { name: "wasm" }] } as any);
    const el = renderSidebarItem(session, false);
    const tagEls = el.querySelectorAll(".sidebar-item-tag");
    expect(tagEls.length).toBe(2);
    expect(tagEls[0]!.textContent).toBe("rust");
    expect(tagEls[1]!.textContent).toBe("wasm");
  });

  test("shows +N overflow badge for more than 2 tags", () => {
    const session = makeSession({
      tags: [{ name: "rust" }, { name: "wasm" }, { name: "zig" }, { name: "go" }],
    } as any);
    const el = renderSidebarItem(session, false);
    const tagEls = el.querySelectorAll(".sidebar-item-tag");
    expect(tagEls.length).toBe(3); // 2 visible + 1 overflow badge
    expect(tagEls[2]!.textContent).toBe("+2");
  });

  test("no tags row when no tags", () => {
    const el = renderSidebarItem(makeSession(), false);
    expect(el.querySelector(".sidebar-item-tags")).toBeNull();
  });
});

// ── renderSectionLabel ──────────────────────────────────────────────────────

describe("renderSectionLabel", () => {
  test("renders label with correct text", () => {
    const el = renderSectionLabel("Pinned");
    expect(el.textContent).toBe("Pinned");
    expect(el.classList.contains("sidebar-section-label")).toBe(true);
  });
});
