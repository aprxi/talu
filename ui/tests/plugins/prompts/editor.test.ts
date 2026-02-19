import { describe, test, expect, beforeEach } from "bun:test";
import {
  updateSaveButton,
  selectPrompt,
  createNew,
  showEditor,
  showEmpty,
} from "../../../src/plugins/prompts/editor.ts";
import { promptsState } from "../../../src/plugins/prompts/state.ts";
import { initPromptsDom, getPromptsDom } from "../../../src/plugins/prompts/dom.ts";
import { initPromptsDeps } from "../../../src/plugins/prompts/deps.ts";
import { createDomRoot, PROMPTS_DOM_IDS } from "../../helpers/dom.ts";
import { mockTimers, mockNotifications } from "../../helpers/mocks.ts";

/**
 * Tests for prompts editor — dirty detection, selection, creation, visibility.
 */

const PROMPTS_DOM_TAGS: Record<string, string> = {
  "pp-name": "input",
  "pp-content": "textarea",
  "pp-save-btn": "button",
  "pp-delete-btn": "button",
  "pp-copy-btn": "button",
  "pp-new-btn": "button",
};

let notifs: ReturnType<typeof mockNotifications>;

beforeEach(() => {
  notifs = mockNotifications();

  initPromptsDom(createDomRoot(PROMPTS_DOM_IDS, undefined, PROMPTS_DOM_TAGS));

  promptsState.prompts = [
    { id: "p1", name: "Prompt One", content: "Content one", createdAt: 0, updatedAt: 0 },
    { id: "p2", name: "Prompt Two", content: "Content two", createdAt: 0, updatedAt: 0 },
  ];
  promptsState.selectedId = null;
  promptsState.defaultId = null;
  promptsState.originalName = "";
  promptsState.originalContent = "";

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

// ── updateSaveButton ────────────────────────────────────────────────────────

describe("updateSaveButton", () => {
  test("save disabled when editing existing and no changes", () => {
    promptsState.selectedId = "p1";
    promptsState.originalName = "Prompt One";
    promptsState.originalContent = "Content one";
    const dom = getPromptsDom();
    dom.nameInput.value = "Prompt One";
    dom.contentInput.value = "Content one";
    updateSaveButton();
    expect(dom.saveBtn.disabled).toBe(true);
  });

  test("save enabled when name changes", () => {
    promptsState.selectedId = "p1";
    promptsState.originalName = "Prompt One";
    promptsState.originalContent = "Content one";
    const dom = getPromptsDom();
    dom.nameInput.value = "New Name";
    dom.contentInput.value = "Content one";
    updateSaveButton();
    expect(dom.saveBtn.disabled).toBe(false);
  });

  test("save enabled when content changes", () => {
    promptsState.selectedId = "p1";
    promptsState.originalName = "Prompt One";
    promptsState.originalContent = "Content one";
    const dom = getPromptsDom();
    dom.nameInput.value = "Prompt One";
    dom.contentInput.value = "Modified content";
    updateSaveButton();
    expect(dom.saveBtn.disabled).toBe(false);
  });

  test("save disabled for new prompt when both fields empty", () => {
    promptsState.selectedId = null;
    promptsState.originalName = "";
    promptsState.originalContent = "";
    const dom = getPromptsDom();
    dom.nameInput.value = "";
    dom.contentInput.value = "";
    updateSaveButton();
    expect(dom.saveBtn.disabled).toBe(true);
  });

  test("save enabled for new prompt when name has content", () => {
    promptsState.selectedId = null;
    promptsState.originalName = "";
    promptsState.originalContent = "";
    const dom = getPromptsDom();
    dom.nameInput.value = "New Prompt";
    dom.contentInput.value = "";
    updateSaveButton();
    expect(dom.saveBtn.disabled).toBe(false);
  });

  test("save disabled for new prompt with only whitespace", () => {
    promptsState.selectedId = null;
    const dom = getPromptsDom();
    dom.nameInput.value = "   ";
    dom.contentInput.value = "   ";
    updateSaveButton();
    // hasContent checks trim() !== "" — both are whitespace only
    // But name !== originalName (""), so hasChanges is true for existing.
    // For new prompt (selectedId=null), it checks hasContent which is false.
    expect(dom.saveBtn.disabled).toBe(true);
  });
});

// ── selectPrompt ────────────────────────────────────────────────────────────

describe("selectPrompt", () => {
  test("populates name and content inputs", () => {
    selectPrompt("p1");
    const dom = getPromptsDom();
    expect(dom.nameInput.value).toBe("Prompt One");
    expect(dom.contentInput.value).toBe("Content one");
  });

  test("sets selectedId", () => {
    selectPrompt("p2");
    expect(promptsState.selectedId).toBe("p2");
  });

  test("stores original values for dirty detection", () => {
    selectPrompt("p1");
    expect(promptsState.originalName).toBe("Prompt One");
    expect(promptsState.originalContent).toBe("Content one");
  });

  test("shows delete button", () => {
    const dom = getPromptsDom();
    dom.deleteBtn.classList.add("hidden");
    selectPrompt("p1");
    expect(dom.deleteBtn.classList.contains("hidden")).toBe(false);
  });

  test("shows editor panel", () => {
    selectPrompt("p1");
    const dom = getPromptsDom();
    expect(dom.editorEl.classList.contains("hidden")).toBe(false);
  });

  test("no-op for nonexistent prompt", () => {
    promptsState.selectedId = "p1";
    selectPrompt("nonexistent");
    expect(promptsState.selectedId).toBe("p1"); // unchanged
  });
});

// ── createNew ───────────────────────────────────────────────────────────────

describe("createNew", () => {
  test("clears selectedId", () => {
    promptsState.selectedId = "p1";
    createNew();
    expect(promptsState.selectedId).toBeNull();
  });

  test("clears name and content inputs", () => {
    const dom = getPromptsDom();
    dom.nameInput.value = "old";
    dom.contentInput.value = "old content";
    createNew();
    expect(dom.nameInput.value).toBe("");
    expect(dom.contentInput.value).toBe("");
  });

  test("hides delete button", () => {
    createNew();
    expect(getPromptsDom().deleteBtn.classList.contains("hidden")).toBe(true);
  });

  test("shows editor panel", () => {
    createNew();
    expect(getPromptsDom().editorEl.classList.contains("hidden")).toBe(false);
  });

  test("resets original values", () => {
    promptsState.originalName = "stale";
    promptsState.originalContent = "stale";
    createNew();
    expect(promptsState.originalName).toBe("");
    expect(promptsState.originalContent).toBe("");
  });
});

// ── showEditor / showEmpty ──────────────────────────────────────────────────

describe("showEditor / showEmpty", () => {
  test("showEditor shows editor, hides empty", () => {
    const dom = getPromptsDom();
    dom.editorEl.classList.add("hidden");
    dom.emptyEl.classList.remove("hidden");
    showEditor();
    expect(dom.editorEl.classList.contains("hidden")).toBe(false);
    expect(dom.emptyEl.classList.contains("hidden")).toBe(true);
  });

  test("showEmpty shows empty, hides editor", () => {
    const dom = getPromptsDom();
    dom.editorEl.classList.remove("hidden");
    dom.emptyEl.classList.add("hidden");
    showEmpty();
    expect(dom.emptyEl.classList.contains("hidden")).toBe(false);
    expect(dom.editorEl.classList.contains("hidden")).toBe(true);
  });
});
