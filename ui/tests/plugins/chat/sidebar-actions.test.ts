import { describe, test, expect, beforeEach } from "bun:test";
import { handleTogglePin, handleTitleRename } from "../../../src/plugins/chat/sidebar-actions.ts";
import { chatState } from "../../../src/plugins/chat/state.ts";
import { initChatDom, getChatDom } from "../../../src/plugins/chat/dom.ts";
import { initChatDeps } from "../../../src/plugins/chat/deps.ts";
import { createDomRoot, CHAT_DOM_IDS, CHAT_DOM_TAGS } from "../../helpers/dom.ts";
import { mockTimers, mockNotifications } from "../../helpers/mocks.ts";
import type { Conversation } from "../../../src/types.ts";

/**
 * Tests for sidebar-actions — optimistic pin/title updates with rollback on failure.
 */

let notifs: ReturnType<typeof mockNotifications>;
let apiCalls: { method: string; args: unknown[] }[];
let apiShouldFail: boolean;

function makeConvo(id: string, overrides: Partial<Conversation> = {}): Conversation {
  return {
    id,
    object: "conversation",
    title: `Convo ${id}`,
    created_at: 1700000000,
    updated_at: 1700000000,
    model: "gpt-4",
    items: [],
    metadata: {},
    marker: "",
    ...overrides,
  } as Conversation;
}

beforeEach(() => {
  notifs = mockNotifications();
  apiCalls = [];
  apiShouldFail = false;

  const domRoot = createDomRoot(CHAT_DOM_IDS, undefined, CHAT_DOM_TAGS);
  // renderSidebar() needs sentinel as child of sidebar-list
  const sidebarList = domRoot.querySelector("#sidebar-list")!;
  const sentinel = domRoot.querySelector("#loader-sentinel")!;
  sidebarList.appendChild(sentinel);
  initChatDom(domRoot);

  chatState.activeChat = makeConvo("sess-1");
  chatState.activeSessionId = "sess-1";
  chatState.sessions = [chatState.activeChat];
  chatState.pagination = { offset: 0, hasMore: false, isLoading: false };

  initChatDeps({
    api: {
      patchConversation: async (id: string, patch: any) => {
        apiCalls.push({ method: "patchConversation", args: [id, patch] });
        if (apiShouldFail) return { ok: false, error: "API error" };
        return { ok: true, data: {} };
      },
    } as any,
    notifications: notifs.mock,
    timers: mockTimers(),
    services: {} as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    hooks: { run: async (_name: string, payload: any) => payload } as any,
    clipboard: { writeText: async () => {} } as any,
    download: {} as any,
    observe: { onResize: () => ({ dispose() {} }) } as any,
    format: { dateTime: () => "" } as any,
    upload: { upload: async () => ({}) } as any,
    layout: { setTitle: () => {} } as any,
    menus: { registerItem: () => ({ dispose() {} }), renderSlot: () => ({ dispose() {} }) } as any,
  });
});

// ── handleTogglePin ─────────────────────────────────────────────────────────

describe("handleTogglePin", () => {
  test("pins an unpinned session", async () => {
    await handleTogglePin("sess-1");
    expect(chatState.sessions[0]!.marker).toBe("pinned");
    expect(apiCalls.length).toBe(1);
    expect(apiCalls[0]!.args).toEqual(["sess-1", { marker: "pinned" }]);
  });

  test("unpins a pinned session", async () => {
    chatState.sessions[0]!.marker = "pinned";
    chatState.activeChat!.marker = "pinned";
    await handleTogglePin("sess-1");
    expect(chatState.sessions[0]!.marker).toBe("");
    expect(apiCalls[0]!.args).toEqual(["sess-1", { marker: "" }]);
  });

  test("updates activeChat marker when it matches", async () => {
    await handleTogglePin("sess-1");
    expect(chatState.activeChat!.marker).toBe("pinned");
  });

  test("reverts on API failure", async () => {
    apiShouldFail = true;
    await handleTogglePin("sess-1");
    // Should revert to unpinned
    expect(chatState.sessions[0]!.marker).toBe("");
    expect(chatState.activeChat!.marker).toBe("");
    expect(notifs.messages.some((m) => m.type === "error")).toBe(true);
  });

  test("reverts pinned → unpinned on API failure", async () => {
    chatState.sessions[0]!.marker = "pinned";
    chatState.activeChat!.marker = "pinned";
    apiShouldFail = true;
    await handleTogglePin("sess-1");
    // Should revert to pinned
    expect(chatState.sessions[0]!.marker).toBe("pinned");
  });

  test("no-op when session not found", async () => {
    await handleTogglePin("nonexistent");
    expect(apiCalls.length).toBe(0);
  });
});

// ── handleTitleRename ───────────────────────────────────────────────────────

describe("handleTitleRename", () => {
  test("updates title on success", async () => {
    const titleEl = document.createElement("div");
    titleEl.textContent = "New Title";
    await handleTitleRename(titleEl, "sess-1");
    expect(chatState.sessions[0]!.title).toBe("New Title");
    expect(apiCalls[0]!.args).toEqual(["sess-1", { title: "New Title" }]);
  });

  test("updates activeChat title when it matches", async () => {
    const titleEl = document.createElement("div");
    titleEl.textContent = "Updated";
    await handleTitleRename(titleEl, "sess-1");
    expect(chatState.activeChat!.title).toBe("Updated");
  });

  test("no-op when title unchanged", async () => {
    chatState.sessions[0]!.title = "Same";
    const titleEl = document.createElement("div");
    titleEl.textContent = "Same";
    await handleTitleRename(titleEl, "sess-1");
    expect(apiCalls.length).toBe(0);
  });

  test("reverts title on API failure", async () => {
    apiShouldFail = true;
    const titleEl = document.createElement("div");
    titleEl.textContent = "New Title";
    await handleTitleRename(titleEl, "sess-1");
    // Should revert to original
    expect(chatState.sessions[0]!.title).toBe("Convo sess-1");
    expect(titleEl.textContent).toBe("Convo sess-1");
    expect(notifs.messages.some((m) => m.type === "error")).toBe(true);
  });

  test("reverts activeChat title on API failure", async () => {
    apiShouldFail = true;
    const titleEl = document.createElement("div");
    titleEl.textContent = "Failed Title";
    await handleTitleRename(titleEl, "sess-1");
    expect(chatState.activeChat!.title).toBe("Convo sess-1");
  });

  test("no-op when session not found", async () => {
    const titleEl = document.createElement("div");
    titleEl.textContent = "Whatever";
    await handleTitleRename(titleEl, "nonexistent");
    expect(apiCalls.length).toBe(0);
  });

  test("sends 'Untitled' to API when text is empty", async () => {
    const titleEl = document.createElement("div");
    titleEl.textContent = "";
    await handleTitleRename(titleEl, "sess-1");
    expect(apiCalls[0]!.args).toEqual(["sess-1", { title: "Untitled" }]);
    // Local state stores null for empty title
    expect(chatState.sessions[0]!.title).toBeNull();
  });
});
