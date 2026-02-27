import { describe, test, expect, beforeEach } from "bun:test";
import { setupSidebarEvents } from "../../../src/plugins/chat/sidebar-events.ts";
import { chatState } from "../../../src/plugins/chat/state.ts";
import { initChatDom, getChatDom } from "../../../src/plugins/chat/dom.ts";
import { initChatDeps } from "../../../src/plugins/chat/deps.ts";
import { createDomRoot, CHAT_DOM_IDS, CHAT_DOM_TAGS } from "../../helpers/dom.ts";
import { mockTimers, mockNotifications } from "../../helpers/mocks.ts";
import type { Conversation } from "../../../src/types.ts";

/**
 * Tests for sidebar-events — click delegation for pin, chat selection,
 * and new conversation button.
 */

let notifs: ReturnType<typeof mockNotifications>;
let apiCalls: { method: string; args: unknown[] }[];

function makeConvo(id: string, marker = ""): Conversation {
  return {
    id,
    object: "conversation",
    title: `Convo ${id}`,
    created_at: 1700000000,
    updated_at: 1700000000,
    model: "gpt-4",
    items: [],
    metadata: {},
    marker,
  } as Conversation;
}

beforeEach(() => {
  notifs = mockNotifications();
  apiCalls = [];

  const domRoot = createDomRoot(CHAT_DOM_IDS, undefined, CHAT_DOM_TAGS);
  const sidebarList = domRoot.querySelector("#sidebar-list")!;
  const sentinel = domRoot.querySelector("#loader-sentinel")!;
  sidebarList.appendChild(sentinel);
  initChatDom(domRoot);

  chatState.activeChat = null;
  chatState.activeSessionId = null;
  chatState.lastResponseId = null;
  chatState.sessions = [makeConvo("chat-1"), makeConvo("chat-2")];
  chatState.sidebarSearchQuery = "";
  chatState.pagination = { offset: 0, hasMore: false, isLoading: false };

  initChatDeps({
    api: {
      patchConversation: async (id: string, patch: any) => {
        apiCalls.push({ method: "patchConversation", args: [id, patch] });
        return { ok: true, data: {} };
      },
      getConversation: async (id: string) => {
        apiCalls.push({ method: "getConversation", args: [id] });
        return { ok: true, data: makeConvo(id) };
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

  setupSidebarEvents();
});

// ── Pin button delegation ───────────────────────────────────────────────────

describe("setupSidebarEvents — pin", () => {
  test("clicking pin button calls handleTogglePin", async () => {
    const dom = getChatDom();
    const pinBtn = document.createElement("button");
    pinBtn.dataset["pin"] = "chat-1";
    dom.sidebarList.appendChild(pinBtn);

    pinBtn.click();
    await new Promise((r) => setTimeout(r, 0));

    expect(apiCalls.some((c) => c.method === "patchConversation")).toBe(true);
  });

  test("pin button click does not trigger chat selection", async () => {
    const dom = getChatDom();
    // Create a sidebar item with a pin button inside it
    const item = document.createElement("div");
    item.className = "sidebar-item";
    item.dataset["id"] = "chat-1";
    const pinBtn = document.createElement("button");
    pinBtn.dataset["pin"] = "chat-1";
    item.appendChild(pinBtn);
    dom.sidebarList.appendChild(item);

    pinBtn.click();
    await new Promise((r) => setTimeout(r, 0));

    // Should have called patchConversation (pin) but NOT getConversation (selection)
    expect(apiCalls.some((c) => c.method === "patchConversation")).toBe(true);
    expect(apiCalls.some((c) => c.method === "getConversation")).toBe(false);
  });
});

// ── Chat selection delegation ───────────────────────────────────────────────

describe("setupSidebarEvents — chat selection", () => {
  test("clicking sidebar-item calls selectChat", async () => {
    const dom = getChatDom();
    const item = document.createElement("div");
    item.className = "sidebar-item";
    item.dataset["id"] = "chat-2";
    dom.sidebarList.appendChild(item);

    item.click();
    await new Promise((r) => setTimeout(r, 0));

    expect(chatState.activeSessionId).toBe("chat-2");
    expect(apiCalls.some((c) => c.method === "getConversation")).toBe(true);
  });
});

// ── New conversation button ─────────────────────────────────────────────────

describe("setupSidebarEvents — new conversation", () => {
  test("clicking new project button exists", () => {
    // The new-conversation button was replaced by sidebar-new-project-btn.
    // The new project button is wired by setupSidebarEvents.
    const dom = getChatDom();
    expect(dom.sidebarNewProject).not.toBeNull();
  });
});

// ── Sidebar search ──────────────────────────────────────────────────────────

describe("setupSidebarEvents — sidebar search", () => {
  test("typing in search input updates state", () => {
    const dom = getChatDom();
    dom.sidebarSearch.value = "hello";
    dom.sidebarSearch.dispatchEvent(new Event("input"));

    expect(chatState.sidebarSearchQuery).toBe("hello");
  });

  test("typing shows clear button", () => {
    const dom = getChatDom();
    dom.sidebarSearch.value = "test";
    dom.sidebarSearch.dispatchEvent(new Event("input"));

    expect(dom.sidebarSearchClear.classList.contains("hidden")).toBe(false);
  });

  test("clearing input hides clear button", () => {
    const dom = getChatDom();
    dom.sidebarSearch.value = "test";
    dom.sidebarSearch.dispatchEvent(new Event("input"));
    dom.sidebarSearch.value = "";
    dom.sidebarSearch.dispatchEvent(new Event("input"));

    expect(dom.sidebarSearchClear.classList.contains("hidden")).toBe(true);
  });

  test("clicking clear button resets search", () => {
    const dom = getChatDom();
    dom.sidebarSearch.value = "test";
    dom.sidebarSearch.dispatchEvent(new Event("input"));

    dom.sidebarSearchClear.click();

    expect(dom.sidebarSearch.value).toBe("");
    expect(chatState.sidebarSearchQuery).toBe("");
    expect(dom.sidebarSearchClear.classList.contains("hidden")).toBe(true);
  });
});
