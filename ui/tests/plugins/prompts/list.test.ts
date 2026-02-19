import { describe, test, expect, beforeEach } from "bun:test";
import { renderList } from "../../../src/plugins/prompts/list.ts";
import { promptsState } from "../../../src/plugins/prompts/state.ts";
import { initPromptsDom, getPromptsDom } from "../../../src/plugins/prompts/dom.ts";
import { initPromptsDeps } from "../../../src/plugins/prompts/deps.ts";
import { createDomRoot, PROMPTS_DOM_IDS } from "../../helpers/dom.ts";
import { mockTimers, mockNotifications } from "../../helpers/mocks.ts";

/**
 * Tests for prompts list — sorting, default prompt placement, separator.
 */

const PROMPTS_DOM_TAGS: Record<string, string> = {
  "pp-name": "input",
  "pp-content": "textarea",
  "pp-save-btn": "button",
  "pp-delete-btn": "button",
  "pp-copy-btn": "button",
  "pp-new-btn": "button",
};

beforeEach(() => {
  const notifs = mockNotifications();
  initPromptsDom(createDomRoot(PROMPTS_DOM_IDS, undefined, PROMPTS_DOM_TAGS));

  promptsState.prompts = [];
  promptsState.selectedId = null;
  promptsState.defaultId = null;

  initPromptsDeps({
    api: {} as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    storage: { get: () => null, set: () => {} } as any,
    clipboard: { writeText: async () => {} } as any,
    timers: mockTimers(),
    notifications: notifs.mock,
    log: { info: () => {}, warn: () => {}, error: () => {} } as any,
  });
});

// ── Empty state ─────────────────────────────────────────────────────────────

describe("renderList — empty", () => {
  test("shows empty hint when no prompts", () => {
    renderList();
    const list = getPromptsDom().listEl;
    const hint = list.querySelector(".prompts-empty-hint");
    expect(hint).not.toBeNull();
    expect(hint!.textContent).toContain("No prompts");
  });
});

// ── Sorting ─────────────────────────────────────────────────────────────────

describe("renderList — sorting", () => {
  test("sorts alphabetically by name", () => {
    promptsState.prompts = [
      { id: "c", name: "Charlie", content: "", createdAt: 0, updatedAt: 0 },
      { id: "a", name: "Alpha", content: "", createdAt: 0, updatedAt: 0 },
      { id: "b", name: "Bravo", content: "", createdAt: 0, updatedAt: 0 },
    ];
    renderList();
    const items = getPromptsDom().listEl.querySelectorAll(".prompt-item");
    expect(items[0]!.querySelector(".prompt-item-name")!.textContent).toBe("Alpha");
    expect(items[1]!.querySelector(".prompt-item-name")!.textContent).toBe("Bravo");
    expect(items[2]!.querySelector(".prompt-item-name")!.textContent).toBe("Charlie");
  });

  test("default prompt is always first", () => {
    promptsState.prompts = [
      { id: "z", name: "Zulu", content: "", createdAt: 0, updatedAt: 0 },
      { id: "a", name: "Alpha", content: "", createdAt: 0, updatedAt: 0 },
    ];
    promptsState.defaultId = "z";
    renderList();
    const items = getPromptsDom().listEl.querySelectorAll(".prompt-item");
    expect(items[0]!.querySelector(".prompt-item-name")!.textContent).toBe("Zulu");
    expect(items[1]!.querySelector(".prompt-item-name")!.textContent).toBe("Alpha");
  });
});

// ── Separator ───────────────────────────────────────────────────────────────

describe("renderList — separator", () => {
  test("inserts separator between default and non-default", () => {
    promptsState.prompts = [
      { id: "def", name: "Default", content: "", createdAt: 0, updatedAt: 0 },
      { id: "other", name: "Other", content: "", createdAt: 0, updatedAt: 0 },
    ];
    promptsState.defaultId = "def";
    renderList();
    const sep = getPromptsDom().listEl.querySelector(".prompts-list-separator");
    expect(sep).not.toBeNull();
  });

  test("no separator when no default", () => {
    promptsState.prompts = [
      { id: "a", name: "Alpha", content: "", createdAt: 0, updatedAt: 0 },
      { id: "b", name: "Bravo", content: "", createdAt: 0, updatedAt: 0 },
    ];
    renderList();
    const sep = getPromptsDom().listEl.querySelector(".prompts-list-separator");
    expect(sep).toBeNull();
  });

  test("no separator when only default exists", () => {
    promptsState.prompts = [
      { id: "def", name: "Default", content: "", createdAt: 0, updatedAt: 0 },
    ];
    promptsState.defaultId = "def";
    renderList();
    const sep = getPromptsDom().listEl.querySelector(".prompts-list-separator");
    expect(sep).toBeNull();
  });
});

// ── Selection state ─────────────────────────────────────────────────────────

describe("renderList — selection", () => {
  test("selected prompt has active class", () => {
    promptsState.prompts = [
      { id: "p1", name: "One", content: "", createdAt: 0, updatedAt: 0 },
      { id: "p2", name: "Two", content: "", createdAt: 0, updatedAt: 0 },
    ];
    promptsState.selectedId = "p2";
    renderList();
    const items = getPromptsDom().listEl.querySelectorAll(".prompt-item");
    expect(items[0]!.classList.contains("active")).toBe(false);
    expect(items[1]!.classList.contains("active")).toBe(true);
  });
});

// ── Default button ──────────────────────────────────────────────────────────

describe("renderList — default button", () => {
  test("default prompt has active default button", () => {
    promptsState.prompts = [
      { id: "p1", name: "One", content: "", createdAt: 0, updatedAt: 0 },
    ];
    promptsState.defaultId = "p1";
    renderList();
    const btn = getPromptsDom().listEl.querySelector(".prompt-default-btn");
    expect(btn).not.toBeNull();
    expect(btn!.classList.contains("active")).toBe(true);
  });

  test("non-default prompt has inactive default button", () => {
    promptsState.prompts = [
      { id: "p1", name: "One", content: "", createdAt: 0, updatedAt: 0 },
    ];
    promptsState.defaultId = null;
    renderList();
    const btn = getPromptsDom().listEl.querySelector(".prompt-default-btn");
    expect(btn).not.toBeNull();
    expect(btn!.classList.contains("active")).toBe(false);
  });

  test("default button has data-prompt-id and data-action", () => {
    promptsState.prompts = [
      { id: "p1", name: "One", content: "", createdAt: 0, updatedAt: 0 },
    ];
    renderList();
    const btn = getPromptsDom().listEl.querySelector<HTMLElement>(".prompt-default-btn")!;
    expect(btn.dataset["promptId"]).toBe("p1");
    expect(btn.dataset["action"]).toBe("toggle-default");
  });
});
