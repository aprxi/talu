import { describe, test, expect, beforeEach } from "bun:test";
import {
  showWelcome,
  hideWelcome,
  showInputBar,
  hideInputBar,
  setInputEnabled,
} from "../../../src/plugins/chat/welcome.ts";
import { chatState } from "../../../src/plugins/chat/state.ts";
import { initChatDom, getChatDom } from "../../../src/plugins/chat/dom.ts";
import { initChatDeps } from "../../../src/plugins/chat/deps.ts";
import { createDomRoot, CHAT_DOM_IDS, CHAT_DOM_TAGS } from "../../helpers/dom.ts";
import { mockTimers, mockNotifications } from "../../helpers/mocks.ts";
import { SEND_ICON, STOP_ICON } from "../../../src/icons.ts";

/**
 * Tests for welcome module — view toggling and input state management.
 */

let notifs: ReturnType<typeof mockNotifications>;

beforeEach(() => {
  notifs = mockNotifications();

  const domRoot = createDomRoot(CHAT_DOM_IDS, undefined, CHAT_DOM_TAGS);
  // renderSidebar() needs sentinel as child of sidebar-list
  const sidebarList = domRoot.querySelector("#sidebar-list")!;
  const sentinel = domRoot.querySelector("#loader-sentinel")!;
  sidebarList.appendChild(sentinel);
  initChatDom(domRoot);

  chatState.activeChat = null;
  chatState.activeSessionId = null;
  chatState.sessions = [];
  chatState.pagination = { offset: 0, hasMore: false, isLoading: false };

  initChatDeps({
    api: {} as any,
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

// ── View toggling ───────────────────────────────────────────────────────────

describe("showWelcome / hideWelcome", () => {
  test("showWelcome removes hidden from welcome-state", () => {
    const dom = getChatDom();
    dom.welcomeState.classList.add("hidden");
    showWelcome();
    expect(dom.welcomeState.classList.contains("hidden")).toBe(false);
  });

  test("showWelcome hides input bar", () => {
    const dom = getChatDom();
    dom.inputBar.classList.remove("hidden");
    showWelcome();
    expect(dom.inputBar.classList.contains("hidden")).toBe(true);
  });

  test("hideWelcome adds hidden to welcome-state", () => {
    const dom = getChatDom();
    dom.welcomeState.classList.remove("hidden");
    hideWelcome();
    expect(dom.welcomeState.classList.contains("hidden")).toBe(true);
  });
});

describe("showInputBar / hideInputBar", () => {
  test("showInputBar removes hidden from input bar", () => {
    const dom = getChatDom();
    dom.inputBar.classList.add("hidden");
    showInputBar();
    expect(dom.inputBar.classList.contains("hidden")).toBe(false);
  });

  test("showInputBar hides welcome state", () => {
    const dom = getChatDom();
    dom.welcomeState.classList.remove("hidden");
    showInputBar();
    expect(dom.welcomeState.classList.contains("hidden")).toBe(true);
  });

  test("hideInputBar adds hidden to input bar", () => {
    const dom = getChatDom();
    dom.inputBar.classList.remove("hidden");
    hideInputBar();
    expect(dom.inputBar.classList.contains("hidden")).toBe(true);
  });
});

// ── setInputEnabled ─────────────────────────────────────────────────────────

describe("setInputEnabled", () => {
  test("enabled=true: inputs become enabled", () => {
    // Start disabled
    setInputEnabled(false);
    setInputEnabled(true);
    const dom = getChatDom();
    expect(dom.inputText.disabled).toBe(false);
    expect(dom.welcomeInput.disabled).toBe(false);
    expect(dom.welcomeSend.disabled).toBe(false);
    expect(dom.welcomeAttach.disabled).toBe(false);
    expect(dom.inputAttach.disabled).toBe(false);
    expect(dom.fileInput.disabled).toBe(false);
  });

  test("enabled=false: inputs become disabled", () => {
    setInputEnabled(false);
    const dom = getChatDom();
    expect(dom.inputText.disabled).toBe(true);
    expect(dom.welcomeInput.disabled).toBe(true);
    expect(dom.welcomeSend.disabled).toBe(true);
    expect(dom.welcomeAttach.disabled).toBe(true);
    expect(dom.inputAttach.disabled).toBe(true);
    expect(dom.fileInput.disabled).toBe(true);
  });

  test("enabled=true: send button shows send icon and primary style", () => {
    setInputEnabled(true);
    const dom = getChatDom();
    // HappyDOM serializes self-closing SVG tags as open/close pairs, so use toContain
    expect(dom.inputSend.innerHTML).toContain("viewBox");
    expect(dom.inputSend.innerHTML).toContain("polygon");
    expect(dom.inputSend.disabled).toBe(false);
    expect(dom.inputSend.classList.contains("bg-primary")).toBe(true);
    expect(dom.inputSend.classList.contains("bg-danger")).toBe(false);
  });

  test("enabled=false: send button shows stop icon and danger style", () => {
    setInputEnabled(false);
    const dom = getChatDom();
    expect(dom.inputSend.innerHTML).toContain("viewBox");
    expect(dom.inputSend.innerHTML).toContain("rect");
    expect(dom.inputSend.disabled).toBe(false); // stop button is still clickable
    expect(dom.inputSend.classList.contains("bg-danger")).toBe(true);
    expect(dom.inputSend.classList.contains("bg-primary")).toBe(false);
  });

  test("toggling back to enabled restores primary classes", () => {
    setInputEnabled(false);
    setInputEnabled(true);
    const dom = getChatDom();
    expect(dom.inputSend.classList.contains("bg-primary")).toBe(true);
    expect(dom.inputSend.classList.contains("hover:bg-accent")).toBe(true);
    expect(dom.inputSend.classList.contains("bg-danger")).toBe(false);
    expect(dom.inputSend.classList.contains("hover:bg-danger/80")).toBe(false);
  });
});
