import { describe, test, expect, beforeEach } from "bun:test";
import {
  el,
  isPinned,
  isArchived,
  getTags,
  relativeTime,
  formatDate,
  populateModelSelect,
  initThinkingState,
  isThinkingExpanded,
  setThinkingExpanded,
} from "../../src/render/helpers.ts";
import type { Conversation, ModelEntry } from "../../src/types.ts";

/** Minimal conversation stub. */
function makeConvo(overrides: Partial<Conversation> = {}): Conversation {
  return {
    id: "c1",
    object: "conversation",
    created_at: 1700000000,
    updated_at: 1700000000,
    model: "gpt-4",
    title: "Test",
    items: [],
    ...overrides,
  } as Conversation;
}

// ── el (DOM factory) ────────────────────────────────────────────────────────

describe("el", () => {
  test("creates element with tag", () => {
    const node = el("div");
    expect(node.tagName).toBe("DIV");
  });

  test("sets className", () => {
    const node = el("span", "my-class");
    expect(node.className).toBe("my-class");
  });

  test("sets textContent", () => {
    const node = el("p", undefined, "hello");
    expect(node.textContent).toBe("hello");
  });

  test("returns correct element type", () => {
    const input = el("input");
    expect(input.tagName).toBe("INPUT");
  });

  test("empty string className and text are treated as falsy (not set)", () => {
    const node = el("div", "", "");
    // Source: `if (className)` / `if (text)` — empty strings are falsy.
    expect(node.className).toBe("");
    expect(node.textContent).toBe("");
  });

  test("className and text both set together", () => {
    const node = el("div", "cls", "txt");
    expect(node.className).toBe("cls");
    expect(node.textContent).toBe("txt");
  });
});

// ── isPinned / isArchived ───────────────────────────────────────────────────

describe("isPinned", () => {
  test("returns true for pinned conversation", () => {
    expect(isPinned(makeConvo({ marker: "pinned" }))).toBe(true);
  });

  test("returns false for non-pinned conversation", () => {
    expect(isPinned(makeConvo({ marker: undefined }))).toBe(false);
  });

  test("returns false for archived conversation", () => {
    expect(isPinned(makeConvo({ marker: "archived" }))).toBe(false);
  });
});

describe("isArchived", () => {
  test("returns true for archived conversation", () => {
    expect(isArchived(makeConvo({ marker: "archived" }))).toBe(true);
  });

  test("returns false for non-archived conversation", () => {
    expect(isArchived(makeConvo({ marker: undefined }))).toBe(false);
  });

  test("returns false for pinned conversation", () => {
    expect(isArchived(makeConvo({ marker: "pinned" }))).toBe(false);
  });
});

// ── getTags ─────────────────────────────────────────────────────────────────

describe("getTags", () => {
  test("returns tag names from conversation.tags", () => {
    const c = makeConvo({ tags: [{ id: "1", name: "a" }, { id: "2", name: "b" }] } as any);
    expect(getTags(c)).toEqual(["a", "b"]);
  });

  test("returns empty array when tags is undefined", () => {
    const c = makeConvo({ tags: undefined } as any);
    expect(getTags(c)).toEqual([]);
  });

  test("returns empty array when tags is not an array", () => {
    const c = makeConvo({ tags: "not-array" } as any);
    expect(getTags(c)).toEqual([]);
  });

  test("filters out entries without a name", () => {
    const c = makeConvo({ tags: [{ id: "1", name: "valid" }, null, { id: "3", name: "also-valid" }] } as any);
    expect(getTags(c)).toEqual(["valid", "also-valid"]);
  });

  test("returns empty array when tags is null", () => {
    const c = makeConvo({ tags: null } as any);
    expect(getTags(c)).toEqual([]);
  });

  test("returns empty array for empty tags array", () => {
    const c = makeConvo({ tags: [] } as any);
    expect(getTags(c)).toEqual([]);
  });

  test("handles bare string entries for backward compatibility", () => {
    const c = makeConvo({ tags: ["a", "b"] } as any);
    expect(getTags(c)).toEqual(["a", "b"]);
  });
});

// ── relativeTime ────────────────────────────────────────────────────────────

describe("relativeTime", () => {
  test("recent time (< 60s) → 'just now'", () => {
    const epoch = Date.now() / 1000; // seconds
    expect(relativeTime(epoch)).toBe("just now");
  });

  test("minutes ago", () => {
    const epoch = (Date.now() - 5 * 60 * 1000) / 1000;
    expect(relativeTime(epoch)).toBe("5m ago");
  });

  test("hours ago", () => {
    const epoch = (Date.now() - 3 * 3600 * 1000) / 1000;
    expect(relativeTime(epoch)).toBe("3h ago");
  });

  test("days ago", () => {
    const epoch = (Date.now() - 2 * 86400 * 1000) / 1000;
    expect(relativeTime(epoch)).toBe("2d ago");
  });

  test("over a week → formatted date", () => {
    const epoch = (Date.now() - 14 * 86400 * 1000) / 1000;
    const result = relativeTime(epoch);
    // Should be a formatted date string, not relative.
    expect(result).not.toContain("ago");
    expect(result).not.toBe("just now");
    expect(result.length).toBeGreaterThan(0);
  });

  test("handles millisecond timestamps (> 1e12)", () => {
    const ms = Date.now() - 120 * 1000; // 2 minutes ago, in ms
    expect(relativeTime(ms)).toBe("2m ago");
  });

  test("handles second timestamps (< 1e12)", () => {
    const sec = (Date.now() - 120 * 1000) / 1000;
    expect(relativeTime(sec)).toBe("2m ago");
  });

  test("exactly 60s → '1m ago' (boundary)", () => {
    const epoch = (Date.now() - 60 * 1000) / 1000;
    expect(relativeTime(epoch)).toBe("1m ago");
  });

  test("exactly 3600s → '1h ago' (boundary)", () => {
    const epoch = (Date.now() - 3600 * 1000) / 1000;
    expect(relativeTime(epoch)).toBe("1h ago");
  });

  test("exactly 86400s → '1d ago' (boundary)", () => {
    const epoch = (Date.now() - 86400 * 1000) / 1000;
    expect(relativeTime(epoch)).toBe("1d ago");
  });

  test("exactly 604800s (7d) → formatted date (boundary)", () => {
    const epoch = (Date.now() - 604800 * 1000) / 1000;
    const result = relativeTime(epoch);
    expect(result).not.toContain("ago");
    expect(result).not.toBe("just now");
  });

  test("future timestamps clamped to 'just now'", () => {
    // Source: Math.max(0, ...) clamps negative deltas to 0.
    const futureEpoch = (Date.now() + 60000) / 1000;
    expect(relativeTime(futureEpoch)).toBe("just now");
  });
});

// ── formatDate ──────────────────────────────────────────────────────────────

describe("formatDate", () => {
  test("returns non-empty string", () => {
    const result = formatDate(Date.now() / 1000);
    expect(typeof result).toBe("string");
    expect(result.length).toBeGreaterThan(0);
  });

  test("handles millisecond timestamps", () => {
    const result = formatDate(Date.now());
    expect(result.length).toBeGreaterThan(0);
  });
});

// ── populateModelSelect ─────────────────────────────────────────────────────

describe("populateModelSelect", () => {
  let sel: HTMLSelectElement;

  beforeEach(() => {
    sel = document.createElement("select");
  });

  test("populates with model entries", () => {
    const models: ModelEntry[] = [
      { id: "gpt-4", source: "managed", defaults: {} as any, overrides: {} as any },
      { id: "gpt-3.5", source: "hub", defaults: {} as any, overrides: {} as any },
    ];
    populateModelSelect(sel, models, "gpt-4");
    expect(sel.options.length).toBe(2);
    expect(sel.options[0]!.value).toBe("gpt-4");
    expect(sel.options[0]!.textContent).toBe("gpt-4");
    expect(sel.options[1]!.value).toBe("gpt-3.5");
    expect(sel.options[1]!.textContent).toBe("gpt-3.5");
    expect(sel.value).toBe("gpt-4");
  });

  test("selects first model when selected not found", () => {
    const models: ModelEntry[] = [
      { id: "gpt-4", source: "managed", defaults: {} as any, overrides: {} as any },
    ];
    populateModelSelect(sel, models, "nonexistent");
    expect(sel.value).toBe("gpt-4");
  });

  test("shows 'No models available' for empty list", () => {
    populateModelSelect(sel, [], "");
    expect(sel.options.length).toBe(1);
    expect(sel.options[0]!.textContent).toBe("No models available");
    expect(sel.options[0]!.value).toBe("");
  });

  test("clears previous options on re-populate", () => {
    const opt = document.createElement("option");
    opt.value = "old";
    sel.appendChild(opt);
    const models: ModelEntry[] = [
      { id: "new-model", source: "managed", defaults: {} as any, overrides: {} as any },
    ];
    populateModelSelect(sel, models, "new-model");
    expect(sel.options.length).toBe(1);
    expect(sel.options[0]!.value).toBe("new-model");
  });

  test("groups remote models by provider prefix", () => {
    const models: ModelEntry[] = [
      { id: "local-model", source: "managed", defaults: {} as any, overrides: {} as any },
      { id: "openai::gpt-4o", source: "managed", defaults: {} as any, overrides: {} as any },
      { id: "openai::gpt-3.5", source: "managed", defaults: {} as any, overrides: {} as any },
    ];
    populateModelSelect(sel, models, "openai::gpt-4o");
    // Should have 2 optgroups: Local and Openai.
    const groups = sel.querySelectorAll("optgroup");
    expect(groups.length).toBe(2);
    expect(groups[0]!.label).toBe("Local");
    expect(groups[1]!.label).toBe("Openai");
    // Labels should strip the provider prefix.
    expect(sel.options.length).toBe(3);
    expect(sel.options[1]!.textContent).toBe("gpt-4o");
    expect(sel.value).toBe("openai::gpt-4o");
  });

  test("skips optgroup wrapper for single group", () => {
    const models: ModelEntry[] = [
      { id: "model-a", source: "managed", defaults: {} as any, overrides: {} as any },
      { id: "model-b", source: "managed", defaults: {} as any, overrides: {} as any },
    ];
    populateModelSelect(sel, models, "model-a");
    const groups = sel.querySelectorAll("optgroup");
    expect(groups.length).toBe(0);
    expect(sel.options.length).toBe(2);
  });

  test("preserves full ID as option value for remote models", () => {
    const models: ModelEntry[] = [
      { id: "openai::gpt-4o", source: "managed", defaults: {} as any, overrides: {} as any },
    ];
    populateModelSelect(sel, models, "openai::gpt-4o");
    expect(sel.options[0]!.value).toBe("openai::gpt-4o");
    expect(sel.options[0]!.textContent).toBe("gpt-4o");
  });
});

// ── Thinking state ──────────────────────────────────────────────────────────

describe("thinking state", () => {
  test("default is collapsed (false)", () => {
    initThinkingState(false, () => {});
    expect(isThinkingExpanded()).toBe(false);
  });

  test("initThinkingState sets initial value", () => {
    initThinkingState(true, () => {});
    expect(isThinkingExpanded()).toBe(true);
    // Reset.
    initThinkingState(false, () => {});
  });

  test("setThinkingExpanded updates value", () => {
    initThinkingState(false, () => {});
    setThinkingExpanded(true);
    expect(isThinkingExpanded()).toBe(true);
    // Reset.
    initThinkingState(false, () => {});
  });

  test("setThinkingExpanded calls write-through function", () => {
    let written: boolean | undefined;
    initThinkingState(false, (v) => { written = v; });
    setThinkingExpanded(true);
    expect(written).toBe(true);
    // Reset.
    initThinkingState(false, () => {});
  });
});
