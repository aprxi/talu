import { describe, test, expect, beforeEach, spyOn } from "bun:test";
import {
  saveCurrentPrompt,
  handleDelete,
  resetDeleteBtn,
  toggleDefault,
  copyPrompt,
} from "../../../src/plugins/prompts/crud.ts";
import { promptsState, DEFAULT_PROMPT_KEY } from "../../../src/plugins/prompts/state.ts";
import { initPromptsDeps } from "../../../src/plugins/prompts/deps.ts";
import { initPromptsDom, getPromptsDom } from "../../../src/plugins/prompts/dom.ts";
import { createDomRoot, PROMPTS_DOM_IDS } from "../../helpers/dom.ts";
import { mockNotifications } from "../../helpers/mocks.ts";
import type { Disposable } from "../../../src/kernel/types.ts";

/**
 * Tests for prompts CRUD — save, delete (2-click confirm + timeout),
 * toggle default, copy to clipboard.
 *
 * Strategy: mock API, storage, clipboard, timers, and notifications via
 * initPromptsDeps. DOM is a minimal root with expected element IDs.
 */

// -- Mock state --------------------------------------------------------------

let apiCalls: { method: string; args: unknown[] }[];
let storageCalls: { method: string; key: string }[];
let clipboardText: string | null;
let timerCallbacks: (() => void)[];
let emittedEvents: unknown[];
let notif: ReturnType<typeof mockNotifications>;

beforeEach(() => {
  apiCalls = [];
  storageCalls = [];
  clipboardText = null;
  timerCallbacks = [];
  emittedEvents = [];

  // Reset state.
  promptsState.prompts = [];
  promptsState.builtinId = null;
  promptsState.selectedId = null;
  promptsState.defaultId = null;
  promptsState.originalName = "";
  promptsState.originalContent = "";
  promptsState.deleteConfirmHandle = null;

  // DOM.
  initPromptsDom(createDomRoot(PROMPTS_DOM_IDS));

  // Notifications.
  notif = mockNotifications();

  // Deps with controllable timer (does NOT auto-fire).
  initPromptsDeps({
    api: {
      createDocument: async (doc: any) => {
        apiCalls.push({ method: "createDocument", args: [doc] });
        return { ok: true, data: { id: "new-1", created_at: 1000, updated_at: 1000 } };
      },
      updateDocument: async (id: string, patch: any) => {
        apiCalls.push({ method: "updateDocument", args: [id, patch] });
        return { ok: true };
      },
      deleteDocument: async (id: string) => {
        apiCalls.push({ method: "deleteDocument", args: [id] });
        return { ok: true };
      },
      patchSettings: async (patch: any) => {
        apiCalls.push({ method: "patchSettings", args: [patch] });
        return { ok: true, data: {} };
      },
    } as any,
    events: {
      emit: (event: string, data: unknown) => { emittedEvents.push({ event, data }); },
      on: () => ({ dispose() {} }),
    } as any,
    storage: {
      get: async () => null,
      set: async (key: string) => { storageCalls.push({ method: "set", key }); },
      delete: async (key: string) => { storageCalls.push({ method: "delete", key }); },
      keys: async () => [],
      clear: async () => {},
      onDidChange: () => ({ dispose() {} }),
    } as any,
    clipboard: {
      writeText: async (text: string) => { clipboardText = text; },
    } as any,
    timers: {
      setTimeout(callback: () => void, _ms: number): Disposable {
        timerCallbacks.push(callback);
        return { dispose() {} };
      },
      setInterval() { return { dispose() {} }; },
      requestAnimationFrame(fn: () => void) { fn(); return { dispose() {} }; },
    } as any,
    notifications: notif.mock,
    log: { info: () => {}, warn: () => {}, error: () => {}, debug: () => {} } as any,
  });
});

// -- Helpers -----------------------------------------------------------------

function setEditorValues(name: string, content: string): void {
  const dom = getPromptsDom();
  dom.nameInput.value = name;
  dom.contentInput.value = content;
}

function addPrompt(id: string, name: string, content = ""): void {
  promptsState.prompts.push({
    id,
    name,
    content,
    createdAt: 1000,
    updatedAt: 1000,
  });
}

// ── saveCurrentPrompt ──────────────────────────────────────────────────────

describe("saveCurrentPrompt", () => {
  test("creates new prompt via API when no selectedId", async () => {
    setEditorValues("My Prompt", "Be helpful");
    await saveCurrentPrompt();

    expect(apiCalls[0]!.method).toBe("createDocument");
    const doc = apiCalls[0]!.args[0] as Record<string, unknown>;
    expect(doc.type).toBe("prompt");
    expect(doc.title).toBe("My Prompt");
    expect((doc.content as any).system).toBe("Be helpful");
  });

  test("adds created prompt to state array", async () => {
    setEditorValues("New", "content");
    await saveCurrentPrompt();

    expect(promptsState.prompts.length).toBe(1);
    expect(promptsState.prompts[0]!.id).toBe("new-1");
    expect(promptsState.prompts[0]!.name).toBe("New");
    expect(promptsState.selectedId).toBe("new-1");
  });

  test("shows success notification on create", async () => {
    setEditorValues("Test", "body");
    await saveCurrentPrompt();
    expect(notif.messages.some((m) => m.type === "success" && m.msg.includes("created"))).toBe(true);
  });

  test("updates existing prompt via API when selectedId is set", async () => {
    addPrompt("p1", "Old Name", "Old Content");
    promptsState.selectedId = "p1";
    setEditorValues("New Name", "New Content");

    await saveCurrentPrompt();

    expect(apiCalls[0]!.method).toBe("updateDocument");
    expect(apiCalls[0]!.args[0]).toBe("p1");
    const patch = apiCalls[0]!.args[1] as Record<string, unknown>;
    expect(patch.title).toBe("New Name");
    expect((patch.content as any).system).toBe("New Content");
  });

  test("updates state array on successful update", async () => {
    addPrompt("p1", "Old", "old");
    promptsState.selectedId = "p1";
    setEditorValues("Updated", "updated");

    await saveCurrentPrompt();

    expect(promptsState.prompts[0]!.name).toBe("Updated");
    expect(promptsState.prompts[0]!.content).toBe("updated");
  });

  test("updates originalName/originalContent for dirty tracking", async () => {
    setEditorValues("Name", "Content");
    await saveCurrentPrompt();
    expect(promptsState.originalName).toBe("Name");
    expect(promptsState.originalContent).toBe("Content");
  });

  test("shows error and returns early when name is empty", async () => {
    setEditorValues("", "content");
    await saveCurrentPrompt();
    expect(apiCalls.length).toBe(0);
    expect(notif.messages.some((m) => m.type === "error" && m.msg.includes("name"))).toBe(true);
  });

  test("trims whitespace-only name as empty", async () => {
    setEditorValues("   ", "content");
    await saveCurrentPrompt();
    expect(apiCalls.length).toBe(0);
  });

  test("shows error on API failure (create)", async () => {
    // Override API to fail.
    initPromptsDeps({
      api: { createDocument: async () => ({ ok: false, error: "Server error" }) } as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      storage: { get: async () => null, set: async () => {}, delete: async () => {}, keys: async () => [], clear: async () => {}, onDidChange: () => ({ dispose() {} }) } as any,
      clipboard: { writeText: async () => {} } as any,
      timers: { setTimeout: () => ({ dispose() {} }), setInterval: () => ({ dispose() {} }), requestAnimationFrame: (fn: () => void) => { fn(); return { dispose() {} }; } } as any,
      notifications: notif.mock,
      log: { info: () => {}, warn: () => {}, error: () => {}, debug: () => {} } as any,
    });
    setEditorValues("Test", "body");
    await saveCurrentPrompt();
    expect(notif.messages.some((m) => m.type === "error" && m.msg.includes("Server error"))).toBe(true);
  });

  test("emits prompts.changed event after save", async () => {
    setEditorValues("Emit Test", "body");
    await saveCurrentPrompt();
    expect(emittedEvents.some((e: any) => e.event === "prompts.changed")).toBe(true);
  });

  test("no-op when selectedId is the built-in prompt", async () => {
    addPrompt("builtin-1", "Default", "You are a helpful assistant.");
    promptsState.builtinId = "builtin-1";
    promptsState.selectedId = "builtin-1";
    setEditorValues("Hacked Name", "Hacked Content");

    await saveCurrentPrompt();

    expect(apiCalls.length).toBe(0);
    expect(promptsState.prompts[0]!.name).toBe("Default");
  });
});

// ── handleDelete (2-click confirmation) ────────────────────────────────────

describe("handleDelete", () => {
  test("first click enters confirming state", () => {
    addPrompt("p1", "Test");
    promptsState.selectedId = "p1";
    handleDelete();

    const dom = getPromptsDom();
    expect(dom.deleteBtn.classList.contains("confirming")).toBe(true);
    expect(dom.deleteBtn.textContent).toBe("Delete?");
  });

  test("first click does not call API", () => {
    addPrompt("p1", "Test");
    promptsState.selectedId = "p1";
    handleDelete();
    expect(apiCalls.length).toBe(0);
  });

  test("first click sets timeout to auto-reset", () => {
    addPrompt("p1", "Test");
    promptsState.selectedId = "p1";
    handleDelete();
    expect(timerCallbacks.length).toBe(1);
  });

  test("second click within timeout deletes via API", async () => {
    addPrompt("p1", "Test");
    promptsState.selectedId = "p1";

    handleDelete(); // First click → confirming
    handleDelete(); // Second click → doDelete()

    // Give the async doDelete() a tick to complete.
    await new Promise((r) => setTimeout(r, 10));

    expect(apiCalls.length).toBe(1);
    expect(apiCalls[0]!.method).toBe("deleteDocument");
    expect(apiCalls[0]!.args[0]).toBe("p1");
  });

  test("delete removes prompt from state", async () => {
    addPrompt("p1", "Test");
    promptsState.selectedId = "p1";

    handleDelete();
    handleDelete();
    await new Promise((r) => setTimeout(r, 10));

    expect(promptsState.prompts.length).toBe(0);
    expect(promptsState.selectedId).toBeNull();
  });

  test("deleting default prompt reverts to built-in", async () => {
    addPrompt("p1", "Default");
    addPrompt("p2", "Other");
    promptsState.selectedId = "p1";
    promptsState.defaultId = "p1";

    handleDelete();
    handleDelete();
    await new Promise((r) => setTimeout(r, 10));

    expect(promptsState.defaultId).toBeNull();
    expect(apiCalls.some((c) => c.method === "patchSettings" && (c.args[0] as any).default_prompt_id === null)).toBe(true);
  });

  test("deleting only default prompt sets defaultId to null", async () => {
    addPrompt("p1", "Default");
    promptsState.selectedId = "p1";
    promptsState.defaultId = "p1";

    handleDelete();
    handleDelete();
    await new Promise((r) => setTimeout(r, 10));

    expect(promptsState.defaultId).toBeNull();
  });

  test("timeout fires resetDeleteBtn to revert UI", () => {
    addPrompt("p1", "Test");
    promptsState.selectedId = "p1";
    handleDelete(); // First click

    const dom = getPromptsDom();
    expect(dom.deleteBtn.classList.contains("confirming")).toBe(true);

    // Manually fire the timeout callback.
    timerCallbacks[0]!();

    expect(dom.deleteBtn.classList.contains("confirming")).toBe(false);
    expect(dom.deleteBtn.title).toBe("Delete prompt");
  });

  test("no-op when no selectedId", () => {
    promptsState.selectedId = null;
    handleDelete();
    expect(apiCalls.length).toBe(0);
    expect(timerCallbacks.length).toBe(0);
  });

  test("no-op when selectedId is the built-in prompt", () => {
    promptsState.builtinId = "builtin-1";
    promptsState.selectedId = "builtin-1";
    handleDelete();
    expect(apiCalls.length).toBe(0);
    expect(timerCallbacks.length).toBe(0);
  });
});

// ── resetDeleteBtn ─────────────────────────────────────────────────────────

describe("resetDeleteBtn", () => {
  test("removes confirming classes and restores icon", () => {
    const dom = getPromptsDom();
    dom.deleteBtn.classList.add("confirming", "text-danger", "bg-danger/10");
    dom.deleteBtn.textContent = "Delete?";

    resetDeleteBtn();

    expect(dom.deleteBtn.classList.contains("confirming")).toBe(false);
    expect(dom.deleteBtn.classList.contains("text-danger")).toBe(false);
    expect(dom.deleteBtn.classList.contains("btn-icon")).toBe(true);
    expect(dom.deleteBtn.title).toBe("Delete prompt");
  });

  test("clears deleteConfirmHandle", () => {
    promptsState.deleteConfirmHandle = { dispose() {} };
    resetDeleteBtn();
    expect(promptsState.deleteConfirmHandle).toBeNull();
  });
});

// ── toggleDefault ──────────────────────────────────────────────────────────

describe("toggleDefault", () => {
  test("sets defaultId when different from current", () => {
    addPrompt("p1", "Test");
    toggleDefault("p1");
    expect(promptsState.defaultId).toBe("p1");
  });

  test("persists via patchSettings API", () => {
    addPrompt("p1", "Test");
    toggleDefault("p1");
    expect(apiCalls.some((c) => c.method === "patchSettings" && (c.args[0] as any).default_prompt_id === "p1")).toBe(true);
  });

  test("no-op when toggling same id", () => {
    promptsState.defaultId = "p1";
    toggleDefault("p1");
    expect(promptsState.defaultId).toBe("p1");
    expect(apiCalls.length).toBe(0);
  });

  test("null reverts to built-in default", () => {
    promptsState.defaultId = "p1";
    toggleDefault(null);
    expect(promptsState.defaultId).toBeNull();
    expect(apiCalls.some((c) => c.method === "patchSettings" && (c.args[0] as any).default_prompt_id === null)).toBe(true);
  });

  test("null is no-op when already on built-in default", () => {
    promptsState.defaultId = null;
    toggleDefault(null);
    expect(promptsState.defaultId).toBeNull();
    expect(apiCalls.length).toBe(0);
  });

  test("switches default to a different prompt", () => {
    addPrompt("p1", "First");
    addPrompt("p2", "Second");
    promptsState.defaultId = "p1";
    toggleDefault("p2");
    expect(promptsState.defaultId).toBe("p2");
    expect(apiCalls.some((c) => c.method === "patchSettings" && (c.args[0] as any).default_prompt_id === "p2")).toBe(true);
  });

  test("emits prompts.changed event", () => {
    addPrompt("p1", "Test");
    toggleDefault("p1");
    expect(emittedEvents.some((e: any) => e.event === "prompts.changed")).toBe(true);
  });
});

// ── copyPrompt ─────────────────────────────────────────────────────────────

describe("copyPrompt", () => {
  test("copies content to clipboard", async () => {
    const dom = getPromptsDom();
    dom.contentInput.value = "System prompt content";
    copyPrompt();
    // Let the promise resolve.
    await new Promise((r) => setTimeout(r, 10));
    expect(clipboardText).toBe("System prompt content");
  });

  test("shows success notification", async () => {
    const dom = getPromptsDom();
    dom.contentInput.value = "content";
    copyPrompt();
    await new Promise((r) => setTimeout(r, 10));
    expect(notif.messages.some((m) => m.type === "success" && m.msg.includes("clipboard"))).toBe(true);
  });

  test("no-op when content is empty", () => {
    const dom = getPromptsDom();
    dom.contentInput.value = "";
    copyPrompt();
    expect(clipboardText).toBeNull();
  });
});
