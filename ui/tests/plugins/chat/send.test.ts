import { describe, test, expect, beforeEach, spyOn } from "bun:test";
import { setupInputEvents, cancelGeneration, streamResponse } from "../../../src/plugins/chat/send.ts";
import { chatState } from "../../../src/plugins/chat/state.ts";
import { initChatDeps } from "../../../src/plugins/chat/deps.ts";
import { initChatDom, getChatDom } from "../../../src/plugins/chat/dom.ts";
import { createDomRoot, CHAT_DOM_IDS } from "../../helpers/dom.ts";
import { mockTimers, mockNotifications, flushAsync } from "../../helpers/mocks.ts";

/**
 * Tests for chat input handling — keyboard events, auto-resize, send/cancel
 * button behavior, and the streamResponse pipeline.
 *
 * Strategy: create DOM with proper textarea/button elements, wire events,
 * dispatch keyboard/click events and verify state mutations and API calls.
 */

// -- Tag overrides for elements that need textarea/button/select/input --------

const CHAT_TAGS: Record<string, string> = {
  "welcome-input": "textarea",
  "welcome-send": "button",
  "welcome-attach": "button",
  "welcome-model": "select",
  "welcome-prompt": "select",
  "input-text": "textarea",
  "input-send": "button",
  "input-attach": "button",
  "chat-file-input": "input",
  "close-right-panel": "button",
  "panel-model": "select",
  "panel-temperature": "input",
  "panel-top-p": "input",
  "panel-top-k": "input",
  "panel-min-p": "input",
  "panel-max-output-tokens": "input",
  "panel-repetition-penalty": "input",
  "panel-seed": "input",
  "new-conversation": "button",
};

// -- Mock state --------------------------------------------------------------

let apiCalls: { method: string; args: unknown[] }[];
let notif: ReturnType<typeof mockNotifications>;
let createResponseResult: Response | null;

beforeEach(() => {
  apiCalls = [];
  notif = mockNotifications();
  createResponseResult = null;

  // Reset state.
  chatState.sessions = [];
  chatState.activeSessionId = null;
  chatState.activeChat = null;
  chatState.lastResponseId = null;
  chatState.attachments = [];
  chatState.isUploadingAttachments = false;
  chatState.isGenerating = false;
  chatState.streamAbort = null;
  chatState.pagination = { cursor: null, hasMore: true, isLoading: false };

  // DOM with proper element types.
  const root = createDomRoot(CHAT_DOM_IDS, undefined, CHAT_TAGS);
  const list = root.querySelector("#sidebar-list")!;
  const sentinel = root.querySelector("#loader-sentinel")!;
  list.appendChild(sentinel);
  initChatDom(root);

  // Deps.
  initChatDeps({
    api: {
      createResponse: async (body: any, signal?: AbortSignal) => {
        apiCalls.push({ method: "createResponse", args: [body, signal] });
        if (createResponseResult) return createResponseResult;
        return new Response(JSON.stringify({ error: { message: "mock" } }), { status: 500 });
      },
      listConversations: async (_cursor?: any, _limit?: number) => {
        apiCalls.push({ method: "listConversations", args: [_cursor, _limit] });
        return { ok: true, data: { data: [{ id: "conv-1" }], cursor: null, has_more: false } };
      },
      getConversation: async (id: string) => {
        apiCalls.push({ method: "getConversation", args: [id] });
        return {
          ok: true,
          data: {
            id, title: "Test", items: [], object: "conversation",
            created_at: 1000, updated_at: 1000, model: "gpt-4",
          },
        };
      },
    } as any,
    notifications: notif.mock as any,
    services: {
      get: (name: string) => {
        if (name === "talu.models") return { getActiveModel: () => "gpt-4" };
        if (name === "talu.prompts") return { getSelectedPromptId: () => null };
        return undefined;
      },
    } as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    layout: { setTitle: () => {} } as any,
    clipboard: { writeText: async () => {} } as any,
    download: { save: () => {} } as any,
    upload: {} as any,
    hooks: {
      on: () => ({ dispose() {} }),
      run: async <T>(_name: string, value: T) => value,
    } as any,
    timers: mockTimers(),
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

// ── setupInputEvents — keyboard handling ─────────────────────────────────────

describe("setupInputEvents — keyboard handling", () => {
  test("Enter key on inputText triggers send", () => {
    const dom = getChatDom();
    (dom.inputText as HTMLTextAreaElement).value = "Hello";
    chatState.activeSessionId = "c1";

    setupInputEvents();

    let prevented = false;
    const event = new KeyboardEvent("keydown", { key: "Enter", bubbles: true });
    Object.defineProperty(event, "preventDefault", { value: () => { prevented = true; } });
    dom.inputText.dispatchEvent(event);

    expect(prevented).toBe(true);
  });

  test("Shift+Enter on inputText does NOT send", () => {
    const dom = getChatDom();
    (dom.inputText as HTMLTextAreaElement).value = "Hello";

    setupInputEvents();

    let prevented = false;
    const event = new KeyboardEvent("keydown", { key: "Enter", shiftKey: true, bubbles: true });
    Object.defineProperty(event, "preventDefault", { value: () => { prevented = true; } });
    dom.inputText.dispatchEvent(event);

    expect(prevented).toBe(false);
  });

  test("Enter key on welcomeInput triggers welcome send", () => {
    const dom = getChatDom();
    (dom.welcomeInput as HTMLTextAreaElement).value = "Hello";

    setupInputEvents();

    let prevented = false;
    const event = new KeyboardEvent("keydown", { key: "Enter", bubbles: true });
    Object.defineProperty(event, "preventDefault", { value: () => { prevented = true; } });
    dom.welcomeInput.dispatchEvent(event);

    expect(prevented).toBe(true);
  });

  test("Shift+Enter on welcomeInput does NOT send", () => {
    const dom = getChatDom();
    (dom.welcomeInput as HTMLTextAreaElement).value = "Hello";

    setupInputEvents();

    let prevented = false;
    const event = new KeyboardEvent("keydown", { key: "Enter", shiftKey: true, bubbles: true });
    Object.defineProperty(event, "preventDefault", { value: () => { prevented = true; } });
    dom.welcomeInput.dispatchEvent(event);

    expect(prevented).toBe(false);
  });
});

// ── setupInputEvents — auto-resize ───────────────────────────────────────────

describe("setupInputEvents — auto-resize", () => {
  test("input event resets and adjusts textarea height", () => {
    const dom = getChatDom();
    setupInputEvents();

    // Simulate scrollHeight being set by content.
    Object.defineProperty(dom.inputText, "scrollHeight", { value: 80, configurable: true });

    dom.inputText.dispatchEvent(new Event("input", { bubbles: true }));

    // Height should be set based on scrollHeight (capped at 200px).
    expect(dom.inputText.style.height).toBe("80px");
  });

  test("auto-resize caps at 200px", () => {
    const dom = getChatDom();
    setupInputEvents();

    Object.defineProperty(dom.inputText, "scrollHeight", { value: 300, configurable: true });

    dom.inputText.dispatchEvent(new Event("input", { bubbles: true }));

    expect(dom.inputText.style.height).toBe("200px");
  });

  test("welcomeInput auto-resizes on input", () => {
    const dom = getChatDom();
    setupInputEvents();

    Object.defineProperty(dom.welcomeInput, "scrollHeight", { value: 100, configurable: true });

    dom.welcomeInput.dispatchEvent(new Event("input", { bubbles: true }));

    expect(dom.welcomeInput.style.height).toBe("100px");
  });
});

// ── cancelGeneration ─────────────────────────────────────────────────────────

describe("cancelGeneration", () => {
  test("aborts active stream controller", () => {
    const controller = new AbortController();
    chatState.streamAbort = controller;

    cancelGeneration();

    expect(controller.signal.aborted).toBe(true);
    expect(chatState.streamAbort).toBeNull();
  });

  test("no-op when no active stream", () => {
    chatState.streamAbort = null;
    cancelGeneration(); // Should not throw.
    expect(chatState.streamAbort).toBeNull();
  });
});

// ── setupInputEvents — send button behavior ──────────────────────────────────

describe("setupInputEvents — send button", () => {
  test("click on inputSend when generating calls cancel", () => {
    chatState.isGenerating = true;
    const controller = new AbortController();
    chatState.streamAbort = controller;

    setupInputEvents();
    getChatDom().inputSend.dispatchEvent(new Event("click"));

    expect(controller.signal.aborted).toBe(true);
  });

  test("click on inputSend when not generating with empty input is no-op", async () => {
    chatState.isGenerating = false;
    const dom = getChatDom();
    (dom.inputText as HTMLTextAreaElement).value = "";

    setupInputEvents();
    dom.inputSend.dispatchEvent(new Event("click"));
    await flushAsync();

    expect(apiCalls.length).toBe(0);
  });
});

// ── streamResponse ───────────────────────────────────────────────────────────

describe("streamResponse", () => {
  test("sets isGenerating to true during stream", async () => {
    let wasGenerating = false;
    createResponseResult = new Response(JSON.stringify({ error: { message: "test" } }), { status: 500 });

    initChatDeps({
      api: {
        createResponse: async () => {
          wasGenerating = chatState.isGenerating;
          return new Response(JSON.stringify({ error: { message: "err" } }), { status: 500 });
        },
        listConversations: async () => ({ ok: true, data: { data: [], cursor: null, has_more: false } }),
        getConversation: async () => ({ ok: false, error: "not found" }),
      } as any,
      notifications: notif.mock as any,
      services: { get: () => ({ getActiveModel: () => "gpt-4" }) } as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      layout: { setTitle: () => {} } as any,
      clipboard: { writeText: async () => {} } as any,
      download: { save: () => {} } as any,
      upload: {} as any,
      hooks: { on: () => ({ dispose() {} }), run: async <T>(_: string, v: T) => v } as any,
      timers: mockTimers(),
      observe: { intersection: () => ({ dispose() {} }), mutation: () => ({ dispose() {} }), resize: () => ({ dispose() {} }) } as any,
      format: { date: () => "", dateTime: () => "", relativeTime: () => "", duration: () => "", number: () => "" } as any,
    });

    await streamResponse({ text: "Hello" });

    expect(wasGenerating).toBe(true);
    expect(chatState.isGenerating).toBe(false);
  });

  test("resets isGenerating to false after error", async () => {
    await streamResponse({ text: "Hello" });
    expect(chatState.isGenerating).toBe(false);
  });

  test("clears streamAbort after completion", async () => {
    await streamResponse({ text: "Hello" });
    expect(chatState.streamAbort).toBeNull();
  });

  test("passes streamAbort signal to createResponse", async () => {
    let capturedSignal: AbortSignal | undefined;
    initChatDeps({
      api: {
        createResponse: async (_body: any, signal?: AbortSignal) => {
          capturedSignal = signal;
          // Verify streamAbort is set during the call.
          expect(chatState.streamAbort).not.toBeNull();
          expect(chatState.streamAbort!.signal).toBe(signal);
          return new Response(JSON.stringify({ error: { message: "err" } }), { status: 500 });
        },
        listConversations: async () => ({ ok: true, data: { data: [], cursor: null, has_more: false } }),
        getConversation: async () => ({ ok: false, error: "not found" }),
      } as any,
      notifications: notif.mock as any,
      services: { get: () => ({ getActiveModel: () => "gpt-4" }) } as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      layout: { setTitle: () => {} } as any,
      clipboard: { writeText: async () => {} } as any,
      download: { save: () => {} } as any,
      upload: {} as any,
      hooks: { on: () => ({ dispose() {} }), run: async <T>(_: string, v: T) => v } as any,
      timers: mockTimers(),
      observe: { intersection: () => ({ dispose() {} }), mutation: () => ({ dispose() {} }), resize: () => ({ dispose() {} }) } as any,
      format: { date: () => "", dateTime: () => "", relativeTime: () => "", duration: () => "", number: () => "" } as any,
    });

    await streamResponse({ text: "Hello" });

    // Signal was actually passed.
    expect(capturedSignal).toBeInstanceOf(AbortSignal);
    // After completion, streamAbort is cleaned up.
    expect(chatState.streamAbort).toBeNull();
  });

  test("shows error notification on API 500", async () => {
    await streamResponse({ text: "Hello" });

    expect(notif.messages.some((m) => m.type === "error")).toBe(true);
  });

  test("appends user message to transcript", async () => {
    await streamResponse({ text: "Hello" });

    const dom = getChatDom();
    const userMsg = dom.transcriptContainer.querySelector(".user-msg");
    expect(userMsg).not.toBeNull();
    expect(userMsg!.textContent).toContain("Hello");
  });

  test("appends assistant placeholder", async () => {
    await streamResponse({ text: "Hello" });

    const dom = getChatDom();
    const assistantMsg = dom.transcriptContainer.querySelector(".assistant-msg");
    expect(assistantMsg).not.toBeNull();
  });

  test("discovers session when discoverSession is true", async () => {
    chatState.activeSessionId = null;
    // Provide a successful SSE stream so streamResponse doesn't return early.
    const sseBody = "event: response.completed\ndata: {}\n\n";
    const now = Math.floor(Date.now() / 1000);
    initChatDeps({
      api: {
        createResponse: async () => new Response(sseBody, { status: 200 }),
        listConversations: async (_cursor?: any, _limit?: number) => {
          apiCalls.push({ method: "listConversations", args: [_cursor, _limit] });
          return {
            ok: true,
            data: {
              data: [{ id: "conv-discovered", title: "D", items: [], object: "conversation", created_at: now, updated_at: now, model: "gpt-4", marker: "" }],
              cursor: null, has_more: false,
            },
          };
        },
        getConversation: async (id: string) => {
          apiCalls.push({ method: "getConversation", args: [id] });
          return { ok: true, data: { id, title: "Test", items: [], object: "conversation", created_at: now, updated_at: now, model: "gpt-4", marker: "" } };
        },
      } as any,
      notifications: notif.mock as any,
      services: { get: () => ({ getActiveModel: () => "gpt-4" }) } as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      layout: { setTitle: () => {} } as any,
      clipboard: { writeText: async () => {} } as any,
      download: { save: () => {} } as any,
      upload: {} as any,
      hooks: { on: () => ({ dispose() {} }), run: async <T>(_: string, v: T) => v } as any,
      timers: mockTimers(),
      observe: { intersection: () => ({ dispose() {} }), mutation: () => ({ dispose() {} }), resize: () => ({ dispose() {} }) } as any,
      format: { date: () => "", dateTime: () => "", relativeTime: () => "", duration: () => "", number: () => "" } as any,
    });

    await streamResponse({ text: "Hello", discoverSession: true });

    // discoverSessionId calls listConversations(null, 1); refreshSidebar calls with (cursor, 100).
    expect(apiCalls.some((c) => c.method === "listConversations" && c.args[1] === 1)).toBe(true);
  });

  test("skips session discovery when session already known", async () => {
    chatState.activeSessionId = "existing-session";
    // Provide a successful SSE stream.
    const sseBody = "event: response.completed\ndata: {}\n\n";
    const now = Math.floor(Date.now() / 1000);
    initChatDeps({
      api: {
        createResponse: async () => new Response(sseBody, { status: 200 }),
        listConversations: async (_cursor?: any, _limit?: number) => {
          apiCalls.push({ method: "listConversations", args: [_cursor, _limit] });
          return {
            ok: true,
            data: {
              data: [{ id: "existing-session", title: "S", items: [], object: "conversation", created_at: now, updated_at: now, model: "gpt-4", marker: "" }],
              cursor: null, has_more: false,
            },
          };
        },
        getConversation: async (id: string) => ({
          ok: true, data: { id, title: "Test", items: [], object: "conversation", created_at: now, updated_at: now, model: "gpt-4", marker: "" },
        }),
      } as any,
      notifications: notif.mock as any,
      services: { get: () => ({ getActiveModel: () => "gpt-4" }) } as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      layout: { setTitle: () => {} } as any,
      clipboard: { writeText: async () => {} } as any,
      download: { save: () => {} } as any,
      upload: {} as any,
      hooks: { on: () => ({ dispose() {} }), run: async <T>(_: string, v: T) => v } as any,
      timers: mockTimers(),
      observe: { intersection: () => ({ dispose() {} }), mutation: () => ({ dispose() {} }), resize: () => ({ dispose() {} }) } as any,
      format: { date: () => "", dateTime: () => "", relativeTime: () => "", duration: () => "", number: () => "" } as any,
    });

    await streamResponse({ text: "Hello", discoverSession: true });

    // discoverSessionId calls listConversations(null, 1); should NOT be called
    // when activeSessionId is already set.
    expect(apiCalls.some((c) => c.method === "listConversations" && c.args[1] === 1)).toBe(false);
  });
});
