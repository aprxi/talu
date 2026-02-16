import { describe, test, expect, beforeEach } from "bun:test";
import {
  handleToggleThinking,
  handleChatCopy,
  handleChatExport,
  handleChatFork,
  handleChatArchive,
  handleChatUnarchive,
  handleChatDelete,
} from "../../../src/plugins/chat/actions.ts";
import { chatState } from "../../../src/plugins/chat/state.ts";
import { initChatDeps } from "../../../src/plugins/chat/deps.ts";
import { initChatDom, getChatDom } from "../../../src/plugins/chat/dom.ts";
import { initThinkingState } from "../../../src/render/helpers.ts";
import { createDomRoot, CHAT_DOM_IDS } from "../../helpers/dom.ts";
import { mockNotifications, mockControllableTimers } from "../../helpers/mocks.ts";

/**
 * Tests for chat actions — toggle thinking, copy, export, fork,
 * archive/unarchive, and 2-click delete confirmation.
 *
 * Strategy: mock API (records calls), clipboard, download, notifications,
 * and timers. DOM is a minimal root with expected element IDs. The sidebar
 * sentinel is placed inside the sidebar list for renderSidebar() compat.
 */

// -- Mock state --------------------------------------------------------------

let apiCalls: { method: string; args: unknown[] }[];
let clipboardText: string | null;
let downloadCalls: { blob: Blob; filename: string }[];
let ct: ReturnType<typeof mockControllableTimers>;
let notif: ReturnType<typeof mockNotifications>;

let forkResult: any;
let patchResult: any;
let deleteResult: any;

beforeEach(() => {
  apiCalls = [];
  clipboardText = null;
  downloadCalls = [];
  ct = mockControllableTimers();
  notif = mockNotifications();

  forkResult = { ok: true, data: { id: "forked-1" } };
  patchResult = { ok: true };
  deleteResult = { ok: true };

  // Reset state.
  chatState.sessions = [];
  chatState.activeSessionId = null;
  chatState.activeChat = null;
  chatState.lastResponseId = null;
  chatState.isGenerating = false;
  chatState.streamAbort = null;
  chatState.pagination = { cursor: null, hasMore: true, isLoading: false };

  // DOM — sentinel must be inside sidebarList for renderSidebar().
  const root = createDomRoot(CHAT_DOM_IDS);
  const list = root.querySelector("#sidebar-list")!;
  const sentinel = root.querySelector("#loader-sentinel")!;
  list.appendChild(sentinel);
  initChatDom(root);

  // Thinking state.
  initThinkingState(false, () => {});

  // Deps with controllable timer.
  initChatDeps({
    api: {
      forkConversation: async (id: string, body: any) => {
        apiCalls.push({ method: "forkConversation", args: [id, body] });
        return forkResult;
      },
      patchConversation: async (id: string, patch: any) => {
        apiCalls.push({ method: "patchConversation", args: [id, patch] });
        return patchResult;
      },
      deleteConversation: async (id: string) => {
        apiCalls.push({ method: "deleteConversation", args: [id] });
        return deleteResult;
      },
      listConversations: async () => ({
        ok: true,
        data: { data: [], cursor: null, has_more: false },
      }),
      getConversation: async (id: string) => ({
        ok: true,
        data: {
          id, title: "Loaded", items: [], object: "conversation",
          created_at: 1000, updated_at: 1000, model: "gpt-4",
        },
      }),
    } as any,
    notifications: notif.mock as any,
    services: { get: () => undefined } as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    layout: { setTitle: () => {} } as any,
    clipboard: {
      writeText: async (text: string) => { clipboardText = text; },
    } as any,
    download: {
      save: (blob: Blob, filename: string) => { downloadCalls.push({ blob, filename }); },
    } as any,
    upload: {} as any,
    hooks: {
      on: () => ({ dispose() {} }),
      run: async <T>(_name: string, value: T) => value,
    } as any,
    timers: ct.timers,
    observe: {
      intersection: () => ({ dispose() {} }),
      mutation: () => ({ dispose() {} }),
      resize: () => ({ dispose() {} }),
    } as any,
    format: {
      date: () => "", dateTime: () => "", relativeTime: () => "",
      duration: () => "", number: () => "",
    } as any,
  });
});

// -- Helpers -----------------------------------------------------------------

function makeConvo(id: string, title = "Test", items: any[] = []): any {
  return {
    id, title, items, marker: "",
    object: "conversation", created_at: 1000, updated_at: 1000, model: "gpt-4",
  };
}

function makeMessage(role: string, text: string, textType = "input_text"): any {
  return { type: "message", role, content: [{ type: textType, text }] };
}

// ── handleToggleThinking ──────────────────────────────────────────────────────

describe("handleToggleThinking", () => {
  test("expand: adds active class, sets collapse title", () => {
    const btn = document.createElement("button");
    handleToggleThinking(btn);
    expect(btn.classList.contains("active")).toBe(true);
    expect(btn.title).toBe("Collapse thoughts");
  });

  test("collapse: removes active class, sets expand title", () => {
    const btn = document.createElement("button");
    handleToggleThinking(btn); // expand
    handleToggleThinking(btn); // collapse
    expect(btn.classList.contains("active")).toBe(false);
    expect(btn.title).toBe("Expand thoughts");
  });

  test("round-trips expand → collapse → expand", () => {
    const btn = document.createElement("button");
    handleToggleThinking(btn);
    expect(btn.title).toBe("Collapse thoughts");
    handleToggleThinking(btn);
    expect(btn.title).toBe("Expand thoughts");
    handleToggleThinking(btn);
    expect(btn.title).toBe("Collapse thoughts");
  });
});

// ── handleChatCopy ────────────────────────────────────────────────────────────

describe("handleChatCopy", () => {
  test("copies markdown with title and messages", async () => {
    chatState.activeChat = makeConvo("c1", "My Chat", [
      makeMessage("user", "Hello"),
      makeMessage("assistant", "Hi there", "output_text"),
    ]);
    await handleChatCopy();

    expect(clipboardText).toContain("# My Chat");
    expect(clipboardText).toContain("**User:** Hello");
    expect(clipboardText).toContain("**Assistant:** Hi there");
  });

  test("shows success notification", async () => {
    chatState.activeChat = makeConvo("c1", "Chat", [makeMessage("user", "Hi")]);
    await handleChatCopy();
    expect(notif.messages.some((m) => m.type === "info" && m.msg.includes("clipboard"))).toBe(true);
  });

  test("no-op when no active chat", async () => {
    chatState.activeChat = null;
    await handleChatCopy();
    expect(clipboardText).toBeNull();
  });

  test("omits title when chat has no title", async () => {
    chatState.activeChat = makeConvo("c1", "", [makeMessage("user", "Hi")]);
    await handleChatCopy();
    expect(clipboardText).not.toContain("# ");
  });

  test("shows error on clipboard failure", async () => {
    chatState.activeChat = makeConvo("c1", "Chat", [makeMessage("user", "Hi")]);
    // Override clipboard to throw.
    initChatDeps({
      api: {} as any,
      notifications: notif.mock as any,
      services: { get: () => undefined } as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      layout: { setTitle: () => {} } as any,
      clipboard: { writeText: async () => { throw new Error("denied"); } } as any,
      download: {} as any,
      upload: {} as any,
      hooks: {
        on: () => ({ dispose() {} }),
        run: async <T>(_name: string, value: T) => value,
      } as any,
      timers: { setTimeout: () => ({ dispose() {} }), setInterval: () => ({ dispose() {} }), requestAnimationFrame: (fn: () => void) => { fn(); return { dispose() {} }; } } as any,
      observe: { intersection: () => ({ dispose() {} }), mutation: () => ({ dispose() {} }), resize: () => ({ dispose() {} }) } as any,
      format: { date: () => "", dateTime: () => "", relativeTime: () => "", duration: () => "", number: () => "" } as any,
    });
    await handleChatCopy();
    expect(notif.messages.some((m) => m.type === "error" && m.msg.includes("copy"))).toBe(true);
  });
});

// ── handleChatExport ──────────────────────────────────────────────────────────

describe("handleChatExport", () => {
  test("triggers download with JSON blob", () => {
    chatState.activeChat = makeConvo("c1", "Test Chat");
    handleChatExport();

    expect(downloadCalls.length).toBe(1);
    expect(downloadCalls[0]!.filename).toBe("Test Chat.json");
    expect(downloadCalls[0]!.blob.type).toBe("application/json");
  });

  test("uses fallback filename when title is empty", () => {
    chatState.activeChat = makeConvo("c1", "");
    handleChatExport();
    expect(downloadCalls[0]!.filename).toBe("conversation.json");
  });

  test("no-op when no active chat", () => {
    chatState.activeChat = null;
    handleChatExport();
    expect(downloadCalls.length).toBe(0);
  });

  test("shows notification on export", () => {
    chatState.activeChat = makeConvo("c1", "Chat");
    handleChatExport();
    expect(notif.messages.some((m) => m.type === "info" && m.msg.includes("Exported"))).toBe(true);
  });
});

// ── handleChatFork ────────────────────────────────────────────────────────────

describe("handleChatFork", () => {
  test("calls fork API with target_item_id", async () => {
    chatState.activeSessionId = "c1";
    chatState.activeChat = makeConvo("c1", "Chat", [
      makeMessage("user", "Hi"),
      makeMessage("assistant", "Hello"),
    ]);
    await handleChatFork();

    expect(apiCalls[0]!.method).toBe("forkConversation");
    expect(apiCalls[0]!.args[0]).toBe("c1");
    expect((apiCalls[0]!.args[1] as any).target_item_id).toBe(1); // lastIndex
  });

  test("no-op when no active session", async () => {
    chatState.activeSessionId = null;
    await handleChatFork();
    expect(apiCalls.length).toBe(0);
  });

  test("no-op when no items", async () => {
    chatState.activeSessionId = "c1";
    chatState.activeChat = null;
    await handleChatFork();
    expect(apiCalls.length).toBe(0);
  });

  test("shows error on API failure", async () => {
    chatState.activeSessionId = "c1";
    chatState.activeChat = makeConvo("c1", "Chat", [makeMessage("user", "Hi")]);
    forkResult = { ok: false, error: "Server error" };

    await handleChatFork();
    expect(notif.messages.some((m) => m.type === "error" && m.msg.includes("Server error"))).toBe(true);
  });

  test("shows success notification on fork", async () => {
    chatState.activeSessionId = "c1";
    chatState.activeChat = makeConvo("c1", "Chat", [makeMessage("user", "Hi")]);
    await handleChatFork();
    expect(notif.messages.some((m) => m.type === "info" && m.msg.includes("Forked"))).toBe(true);
  });
});

// ── handleChatArchive ─────────────────────────────────────────────────────────

describe("handleChatArchive", () => {
  test("calls patchConversation with archived marker", async () => {
    chatState.activeSessionId = "c1";
    chatState.activeChat = makeConvo("c1");
    await handleChatArchive();

    expect(apiCalls[0]!.method).toBe("patchConversation");
    expect(apiCalls[0]!.args[0]).toBe("c1");
    expect((apiCalls[0]!.args[1] as any).marker).toBe("archived");
  });

  test("clears active state after archive", async () => {
    chatState.activeSessionId = "c1";
    chatState.activeChat = makeConvo("c1");
    chatState.lastResponseId = "resp-1";
    await handleChatArchive();

    expect(chatState.activeSessionId).toBeNull();
    expect(chatState.activeChat).toBeNull();
    expect(chatState.lastResponseId).toBeNull();
  });

  test("shows success notification", async () => {
    chatState.activeSessionId = "c1";
    chatState.activeChat = makeConvo("c1");
    await handleChatArchive();
    expect(notif.messages.some((m) => m.type === "info" && m.msg.includes("Archived"))).toBe(true);
  });

  test("no-op when no active session", async () => {
    chatState.activeSessionId = null;
    await handleChatArchive();
    expect(apiCalls.length).toBe(0);
  });

  test("shows error on API failure", async () => {
    chatState.activeSessionId = "c1";
    chatState.activeChat = makeConvo("c1");
    patchResult = { ok: false, error: "Forbidden" };
    await handleChatArchive();
    expect(notif.messages.some((m) => m.type === "error" && m.msg.includes("Forbidden"))).toBe(true);
    // State should NOT be cleared on failure.
    expect(chatState.activeSessionId).toBe("c1");
  });
});

// ── handleChatUnarchive ───────────────────────────────────────────────────────

describe("handleChatUnarchive", () => {
  test("calls patchConversation with empty marker", async () => {
    chatState.activeSessionId = "c1";
    chatState.activeChat = makeConvo("c1");
    chatState.activeChat.marker = "archived";
    await handleChatUnarchive();

    expect(apiCalls[0]!.method).toBe("patchConversation");
    expect((apiCalls[0]!.args[1] as any).marker).toBe("");
  });

  test("shows success notification", async () => {
    chatState.activeSessionId = "c1";
    chatState.activeChat = makeConvo("c1");
    await handleChatUnarchive();
    expect(notif.messages.some((m) => m.type === "info" && m.msg.includes("Unarchived"))).toBe(true);
  });

  test("no-op when no active session", async () => {
    chatState.activeSessionId = null;
    await handleChatUnarchive();
    expect(apiCalls.length).toBe(0);
  });

  test("shows error on API failure", async () => {
    chatState.activeSessionId = "c1";
    chatState.activeChat = makeConvo("c1");
    patchResult = { ok: false, error: "Not found" };
    await handleChatUnarchive();
    expect(notif.messages.some((m) => m.type === "error" && m.msg.includes("Not found"))).toBe(true);
  });
});

// ── handleChatDelete (2-click confirmation) ──────────────────────────────────

describe("handleChatDelete", () => {
  test("first click enters confirmation state", () => {
    chatState.activeSessionId = "c1";
    const btn = document.createElement("button");
    handleChatDelete(btn);

    expect(btn.dataset["confirm"]).toBe("true");
    expect(btn.title).toBe("Click again to confirm");
    expect(btn.classList.contains("text-danger")).toBe(true);
    expect(btn.classList.contains("bg-danger/10")).toBe(true);
  });

  test("first click does not call API", () => {
    chatState.activeSessionId = "c1";
    const btn = document.createElement("button");
    handleChatDelete(btn);
    expect(apiCalls.length).toBe(0);
  });

  test("first click sets 3s auto-reset timer", () => {
    chatState.activeSessionId = "c1";
    const btn = document.createElement("button");
    handleChatDelete(btn);
    expect(ct.pending.length).toBe(1);
    expect(ct.pending[0]!.ms).toBe(3000);
  });

  test("timer auto-resets confirmation state", () => {
    chatState.activeSessionId = "c1";
    const btn = document.createElement("button");
    handleChatDelete(btn);

    // Fire the auto-reset timer.
    ct.pending[0]!.fn();

    expect(btn.dataset["confirm"]).toBe("");
    expect(btn.title).toBe("Delete conversation");
    expect(btn.classList.contains("text-danger")).toBe(false);
    expect(btn.classList.contains("bg-danger/10")).toBe(false);
  });

  test("second click calls delete API", async () => {
    chatState.activeSessionId = "c1";
    chatState.activeChat = makeConvo("c1");
    const btn = document.createElement("button");

    handleChatDelete(btn); // First click → confirmation
    handleChatDelete(btn); // Second click → delete
    await new Promise((r) => setTimeout(r, 10));

    expect(apiCalls.length).toBe(1);
    expect(apiCalls[0]!.method).toBe("deleteConversation");
    expect(apiCalls[0]!.args[0]).toBe("c1");
  });

  test("delete clears active state on success", async () => {
    chatState.activeSessionId = "c1";
    chatState.activeChat = makeConvo("c1");
    chatState.lastResponseId = "resp-1";
    const btn = document.createElement("button");

    handleChatDelete(btn);
    handleChatDelete(btn);
    await new Promise((r) => setTimeout(r, 10));

    expect(chatState.activeSessionId).toBeNull();
    expect(chatState.activeChat).toBeNull();
    expect(chatState.lastResponseId).toBeNull();
  });

  test("delete shows success notification", async () => {
    chatState.activeSessionId = "c1";
    chatState.activeChat = makeConvo("c1");
    const btn = document.createElement("button");

    handleChatDelete(btn);
    handleChatDelete(btn);
    await new Promise((r) => setTimeout(r, 10));

    expect(notif.messages.some((m) => m.type === "info" && m.msg.includes("Deleted"))).toBe(true);
  });

  test("delete shows error on API failure", async () => {
    chatState.activeSessionId = "c1";
    chatState.activeChat = makeConvo("c1");
    deleteResult = { ok: false, error: "Forbidden" };
    const btn = document.createElement("button");

    handleChatDelete(btn);
    handleChatDelete(btn);
    await new Promise((r) => setTimeout(r, 10));

    expect(notif.messages.some((m) => m.type === "error" && m.msg.includes("Forbidden"))).toBe(true);
  });

  test("no-op when no active session", () => {
    chatState.activeSessionId = null;
    const btn = document.createElement("button");
    handleChatDelete(btn);
    expect(btn.dataset["confirm"]).toBeUndefined();
    expect(apiCalls.length).toBe(0);
  });
});
