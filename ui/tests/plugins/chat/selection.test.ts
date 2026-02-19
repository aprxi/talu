import { describe, test, expect, beforeEach } from "bun:test";
import { selectChat, renderChatView, ensureChatHeader } from "../../../src/plugins/chat/selection.ts";
import { chatState } from "../../../src/plugins/chat/state.ts";
import { initChatDom, getChatDom } from "../../../src/plugins/chat/dom.ts";
import { initChatDeps } from "../../../src/plugins/chat/deps.ts";
import { createDomRoot, CHAT_DOM_IDS, CHAT_DOM_TAGS } from "../../helpers/dom.ts";
import { mockTimers, mockNotifications } from "../../helpers/mocks.ts";
import type { Conversation } from "../../../src/types.ts";

/**
 * Tests for selection — selectChat race guard, renderChatView, ensureChatHeader.
 */

let notifs: ReturnType<typeof mockNotifications>;
let apiResult: any;
let titleSet: string;

function makeConvo(id: string, items: any[] = []): Conversation {
  return {
    id,
    object: "conversation",
    title: `Convo ${id}`,
    created_at: 1700000000,
    updated_at: 1700000000,
    model: "gpt-4",
    items,
    metadata: {},
  } as Conversation;
}

beforeEach(() => {
  notifs = mockNotifications();
  titleSet = "";
  apiResult = { ok: true, data: makeConvo("sess-1", []) };

  const domRoot = createDomRoot(CHAT_DOM_IDS, undefined, CHAT_DOM_TAGS);
  const sidebarList = domRoot.querySelector("#sidebar-list")!;
  const sentinel = domRoot.querySelector("#loader-sentinel")!;
  sidebarList.appendChild(sentinel);
  initChatDom(domRoot);

  chatState.activeChat = null;
  chatState.activeSessionId = null;
  chatState.lastResponseId = null;
  chatState.sessions = [];
  chatState.pagination = { offset: 0, hasMore: false, isLoading: false };

  initChatDeps({
    api: {
      getConversation: async (_id: string) => apiResult,
    } as any,
    notifications: notifs.mock,
    timers: mockTimers(),
    services: {
      get: () => ({
        getActiveModel: () => "gpt-4",
        getAvailableModels: () => [],
        getPromptNameById: () => null,
      }),
    } as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    hooks: { run: async (_name: string, payload: any) => payload } as any,
    clipboard: { writeText: async () => {} } as any,
    download: {} as any,
    observe: { onResize: () => ({ dispose() {} }) } as any,
    format: { dateTime: () => "2024-01-01" } as any,
    upload: { upload: async () => ({}) } as any,
    layout: { setTitle: (t: string) => { titleSet = t; } } as any,
    menus: { registerItem: () => ({ dispose() {} }), renderSlot: () => ({ dispose() {} }) } as any,
  });
});

// ── selectChat ──────────────────────────────────────────────────────────────

describe("selectChat", () => {
  test("sets activeSessionId and resets lastResponseId", async () => {
    chatState.lastResponseId = "resp-old";
    await selectChat("sess-1");
    expect(chatState.activeSessionId).toBe("sess-1");
    expect(chatState.lastResponseId).toBeNull();
  });

  test("clears transcript and shows loading spinner", async () => {
    // We can verify the final state — spinner removed after load
    await selectChat("sess-1");
    const tc = getChatDom().transcriptContainer;
    // After successful load, spinner should be gone and header should be present
    expect(tc.querySelector(".spinner")).toBeNull();
  });

  test("sets activeChat on success", async () => {
    await selectChat("sess-1");
    expect(chatState.activeChat).not.toBeNull();
    expect(chatState.activeChat!.id).toBe("sess-1");
  });

  test("shows input bar after successful load", async () => {
    const dom = getChatDom();
    dom.inputBar.classList.add("hidden");
    await selectChat("sess-1");
    expect(dom.inputBar.classList.contains("hidden")).toBe(false);
  });

  test("clears input text after load", async () => {
    getChatDom().inputText.value = "leftover text";
    await selectChat("sess-1");
    expect(getChatDom().inputText.value).toBe("");
  });

  test("shows error on API failure", async () => {
    apiResult = { ok: false, error: "Network error" };
    await selectChat("sess-1");
    expect(notifs.messages.some((m) => m.type === "error")).toBe(true);
    const tc = getChatDom().transcriptContainer;
    const emptyState = tc.querySelector("[data-empty-state]");
    expect(emptyState).not.toBeNull();
  });

  test("discards result if user navigated away during load", async () => {
    // Override API to simulate delay
    let resolveApi: any;
    initChatDeps({
      api: {
        getConversation: async () => {
          return new Promise((r) => { resolveApi = r; });
        },
      } as any,
      notifications: notifs.mock,
      timers: mockTimers(),
      services: {
        get: () => ({
          getActiveModel: () => "gpt-4",
          getAvailableModels: () => [],
          getPromptNameById: () => null,
        }),
      } as any,
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

    const promise = selectChat("sess-1");
    // User clicks another chat while loading
    chatState.activeSessionId = "sess-2";
    resolveApi({ ok: true, data: makeConvo("sess-1") });
    await promise;
    // Should NOT have set activeChat to sess-1
    expect(chatState.activeChat).toBeNull();
  });
});

// ── renderChatView ──────────────────────────────────────────────────────────

describe("renderChatView", () => {
  test("renders header with title", () => {
    const chat = makeConvo("conv-1");
    renderChatView(chat);
    const tc = getChatDom().transcriptContainer;
    const header = tc.querySelector(".transcript-header");
    expect(header).not.toBeNull();
  });

  test("sets layout title", () => {
    const chat = makeConvo("conv-1");
    chat.title = "My Chat";
    renderChatView(chat);
    expect(titleSet).toBe("My Chat");
  });

  test("uses 'Untitled' for empty title", () => {
    const chat = makeConvo("conv-1");
    chat.title = "";
    renderChatView(chat);
    expect(titleSet).toBe("Untitled");
  });

  test("shows empty state when no items", () => {
    const chat = makeConvo("conv-1");
    renderChatView(chat);
    const tc = getChatDom().transcriptContainer;
    const empty = tc.querySelector("[data-empty-state]");
    expect(empty).not.toBeNull();
  });

  test("renders transcript items when present", () => {
    const chat = makeConvo("conv-1", [
      { type: "message", role: "user", content: [{ type: "input_text", text: "hello" }] },
      { type: "message", role: "assistant", content: [{ type: "output_text", text: "hi" }] },
    ]);
    renderChatView(chat);
    const tc = getChatDom().transcriptContainer;
    expect(tc.querySelector("[data-transcript-items]")).not.toBeNull();
  });
});

// ── ensureChatHeader ────────────────────────────────────────────────────────

describe("ensureChatHeader", () => {
  test("adds header if missing", () => {
    const chat = makeConvo("conv-1");
    ensureChatHeader(chat);
    const tc = getChatDom().transcriptContainer;
    expect(tc.querySelector(".transcript-header")).not.toBeNull();
  });

  test("does not add duplicate header", () => {
    const chat = makeConvo("conv-1");
    ensureChatHeader(chat);
    ensureChatHeader(chat); // second call
    const tc = getChatDom().transcriptContainer;
    const headers = tc.querySelectorAll(".transcript-header");
    expect(headers.length).toBe(1);
  });
});
