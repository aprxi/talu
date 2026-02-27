import { describe, test, expect, beforeEach, spyOn } from "bun:test";
import { setupInputEvents, cancelGeneration, streamResponse } from "../../../src/plugins/chat/send.ts";
import { chatState } from "../../../src/plugins/chat/state.ts";
import { initChatDeps } from "../../../src/plugins/chat/deps.ts";
import { initChatDom, getChatDom } from "../../../src/plugins/chat/dom.ts";
import { createDomRoot, CHAT_DOM_IDS, CHAT_DOM_TAGS } from "../../helpers/dom.ts";
import { mockTimers, mockNotifications, flushAsync } from "../../helpers/mocks.ts";

/**
 * Tests for chat input handling — keyboard events, auto-resize, send/cancel
 * button behavior, and the streamResponse pipeline.
 *
 * Strategy: create DOM with proper textarea/button elements, wire events,
 * dispatch keyboard/click events and verify state mutations and API calls.
 */

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
  chatState.pagination = { offset: 0, hasMore: true, isLoading: false };

  // DOM with proper element types.
  const root = createDomRoot(CHAT_DOM_IDS, undefined, CHAT_DOM_TAGS);
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
      listConversations: async (params?: any) => {
        apiCalls.push({ method: "listConversations", args: [params] });
        return { ok: true, data: { data: [{ id: "conv-1" }], has_more: false } };
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
        if (name === "talu.prompts") return { getSelectedPromptId: () => null, getDefaultPromptId: () => null };
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
    menus: {
      registerItem: () => ({ dispose() {} }),
      renderSlot: () => ({ dispose() {} }),
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
        listConversations: async () => ({ ok: true, data: { data: [], has_more: false } }),
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
      menus: {
        registerItem: () => ({ dispose() {} }),
        renderSlot: () => ({ dispose() {} }),
      } as any,
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
        listConversations: async () => ({ ok: true, data: { data: [], has_more: false } }),
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
      menus: {
        registerItem: () => ({ dispose() {} }),
        renderSlot: () => ({ dispose() {} }),
      } as any,
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
        listConversations: async (params?: any) => {
          apiCalls.push({ method: "listConversations", args: [params] });
          return {
            ok: true,
            data: {
              data: [{ id: "conv-discovered", title: "D", items: [], object: "conversation", created_at: now, updated_at: now, model: "gpt-4", marker: "" }],
              has_more: false,
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
      menus: {
        registerItem: () => ({ dispose() {} }),
        renderSlot: () => ({ dispose() {} }),
      } as any,
    });

    await streamResponse({ text: "Hello", discoverSession: true });

    // discoverSessionId calls listConversations({ limit: 1 }).
    expect(apiCalls.some((c) => c.method === "listConversations" && c.args[0]?.limit === 1)).toBe(true);
  });

  test("skips session discovery when session already known", async () => {
    chatState.activeSessionId = "existing-session";
    // Provide a successful SSE stream.
    const sseBody = "event: response.completed\ndata: {}\n\n";
    const now = Math.floor(Date.now() / 1000);
    initChatDeps({
      api: {
        createResponse: async () => new Response(sseBody, { status: 200 }),
        listConversations: async (params?: any) => {
          apiCalls.push({ method: "listConversations", args: [params] });
          return {
            ok: true,
            data: {
              data: [{ id: "existing-session", title: "S", items: [], object: "conversation", created_at: now, updated_at: now, model: "gpt-4", marker: "" }],
              has_more: false,
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
      menus: {
        registerItem: () => ({ dispose() {} }),
        renderSlot: () => ({ dispose() {} }),
      } as any,
    });

    await streamResponse({ text: "Hello", discoverSession: true });

    // discoverSessionId calls listConversations(null, 1); should NOT be called
    // when activeSessionId is already set.
    expect(apiCalls.some((c) => c.method === "listConversations" && c.args[0]?.limit === 1)).toBe(false);
  });
});

// ── chat.send.before hook ─────────────────────────────────────────────────────

describe("chat.send.before hook", () => {
  function makeDepsWithHook(hookFn: (name: string, value: any) => any, overrides?: { createResponse?: any }) {
    const now = Math.floor(Date.now() / 1000);
    initChatDeps({
      api: {
        createResponse: overrides?.createResponse ?? (async (body: any) => {
          apiCalls.push({ method: "createResponse", args: [body] });
          return new Response(JSON.stringify({ error: { message: "mock" } }), { status: 500 });
        }),
        listConversations: async () => ({ ok: true, data: { data: [], has_more: false } }),
        getConversation: async (id: string) => ({
          ok: true, data: { id, title: "Test", items: [], object: "conversation", created_at: now, updated_at: now, model: "gpt-4" },
        }),
      } as any,
      notifications: notif.mock as any,
      services: { get: () => ({ getActiveModel: () => "gpt-4" }) } as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      layout: { setTitle: () => {} } as any,
      clipboard: { writeText: async () => {} } as any,
      download: { save: () => {} } as any,
      upload: {} as any,
      hooks: {
        on: () => ({ dispose() {} }),
        run: hookFn,
      } as any,
      timers: mockTimers(),
      observe: { intersection: () => ({ dispose() {} }), mutation: () => ({ dispose() {} }), resize: () => ({ dispose() {} }) } as any,
      format: { date: () => "", dateTime: () => "", relativeTime: () => "", duration: () => "", number: () => "" } as any,
      menus: {
        registerItem: () => ({ dispose() {} }),
        renderSlot: () => ({ dispose() {} }),
      } as any,
    });
  }

  test("modifies request body before API call", async () => {
    let capturedBody: any;
    makeDepsWithHook(
      async (name: string, value: any) => {
        if (name === "chat.send.before") {
          return { ...value, model: "overridden-model" };
        }
        return value;
      },
      {
        createResponse: async (body: any) => {
          capturedBody = body;
          return new Response(JSON.stringify({ error: { message: "err" } }), { status: 500 });
        },
      },
    );

    await streamResponse({ text: "Hello", input: "Hello" });

    expect(capturedBody.model).toBe("overridden-model");
    expect(capturedBody.input).toBe("Hello");
  });

  test("blocks request when hook returns $block sentinel", async () => {
    makeDepsWithHook(async (name: string, _value: any) => {
      if (name === "chat.send.before") {
        return { $block: true, reason: "Content policy violation" };
      }
      return _value;
    });

    await streamResponse({ text: "Hello" });

    // API should not have been called.
    expect(apiCalls.filter((c) => c.method === "createResponse").length).toBe(0);
    // Warning notification should be shown.
    expect(notif.messages.some((m) => m.type === "warning" && m.msg === "Content policy violation")).toBe(true);
    // State should be reset.
    expect(chatState.isGenerating).toBe(false);
    expect(chatState.streamAbort).toBeNull();
  });

  test("passes through unchanged when no hook handlers registered", async () => {
    let capturedBody: any;
    makeDepsWithHook(
      async (_name: string, value: any) => value, // pass-through
      {
        createResponse: async (body: any) => {
          capturedBody = body;
          return new Response(JSON.stringify({ error: { message: "err" } }), { status: 500 });
        },
      },
    );

    await streamResponse({ text: "Test message", input: "Test message" });

    expect(capturedBody.input).toBe("Test message");
    expect(capturedBody.model).toBe("gpt-4");
  });
});

// ── chat.receive.after hook ───────────────────────────────────────────────────

describe("chat.receive.after hook", () => {
  test("transforms conversation data after response", async () => {
    const now = Math.floor(Date.now() / 1000);
    const sseBody = "event: response.completed\ndata: {}\n\n";
    let hookCalledWith: any = null;

    chatState.activeSessionId = "sess-1";

    initChatDeps({
      api: {
        createResponse: async () => new Response(sseBody, { status: 200 }),
        listConversations: async () => ({ ok: true, data: { data: [], has_more: false } }),
        getConversation: async (id: string) => ({
          ok: true, data: { id, title: "Original Title", items: [], object: "conversation", created_at: now, updated_at: now, model: "gpt-4" },
        }),
      } as any,
      notifications: notif.mock as any,
      services: { get: () => ({ getActiveModel: () => "gpt-4" }) } as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      layout: { setTitle: () => {} } as any,
      clipboard: { writeText: async () => {} } as any,
      download: { save: () => {} } as any,
      upload: {} as any,
      hooks: {
        on: () => ({ dispose() {} }),
        run: async (name: string, value: any) => {
          if (name === "chat.receive.after") {
            hookCalledWith = value;
            return { ...value, title: "Modified Title" };
          }
          return value;
        },
      } as any,
      timers: mockTimers(),
      observe: { intersection: () => ({ dispose() {} }), mutation: () => ({ dispose() {} }), resize: () => ({ dispose() {} }) } as any,
      format: { date: () => "", dateTime: () => "", relativeTime: () => "", duration: () => "", number: () => "" } as any,
      menus: {
        registerItem: () => ({ dispose() {} }),
        renderSlot: () => ({ dispose() {} }),
      } as any,
    });

    await streamResponse({ text: "Hello" });

    // Hook was called with the original conversation data.
    expect(hookCalledWith).not.toBeNull();
    expect(hookCalledWith.title).toBe("Original Title");
    // The modified title should be stored in state.
    expect(chatState.activeChat!.title).toBe("Modified Title");
  });

  test("receives conversation via afterResponse callback", async () => {
    const now = Math.floor(Date.now() / 1000);
    const sseBody = "event: response.completed\ndata: {}\n\n";
    let callbackConversation: any = null;

    chatState.activeSessionId = "sess-2";

    initChatDeps({
      api: {
        createResponse: async () => new Response(sseBody, { status: 200 }),
        listConversations: async () => ({ ok: true, data: { data: [], has_more: false } }),
        getConversation: async (id: string) => ({
          ok: true, data: { id, title: "Chat", items: [], object: "conversation", created_at: now, updated_at: now, model: "gpt-4" },
        }),
      } as any,
      notifications: notif.mock as any,
      services: { get: () => ({ getActiveModel: () => "gpt-4" }) } as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      layout: { setTitle: () => {} } as any,
      clipboard: { writeText: async () => {} } as any,
      download: { save: () => {} } as any,
      upload: {} as any,
      hooks: {
        on: () => ({ dispose() {} }),
        run: async (name: string, value: any) => {
          if (name === "chat.receive.after") {
            return { ...value, title: "Hook-Modified" };
          }
          return value;
        },
      } as any,
      timers: mockTimers(),
      observe: { intersection: () => ({ dispose() {} }), mutation: () => ({ dispose() {} }), resize: () => ({ dispose() {} }) } as any,
      format: { date: () => "", dateTime: () => "", relativeTime: () => "", duration: () => "", number: () => "" } as any,
      menus: {
        registerItem: () => ({ dispose() {} }),
        renderSlot: () => ({ dispose() {} }),
      } as any,
    });

    await streamResponse({
      text: "Hello",
      afterResponse: (conv) => { callbackConversation = conv; },
    });

    // afterResponse callback should receive the hook-modified conversation.
    expect(callbackConversation).not.toBeNull();
    expect(callbackConversation.title).toBe("Hook-Modified");
  });

  test("ignores $block on receive — response already happened", async () => {
    const now = Math.floor(Date.now() / 1000);
    const sseBody = "event: response.completed\ndata: {}\n\n";

    chatState.activeSessionId = "sess-3";

    initChatDeps({
      api: {
        createResponse: async () => new Response(sseBody, { status: 200 }),
        listConversations: async () => ({ ok: true, data: { data: [], has_more: false } }),
        getConversation: async (id: string) => ({
          ok: true, data: { id, title: "Original", items: [], object: "conversation", created_at: now, updated_at: now, model: "gpt-4" },
        }),
      } as any,
      notifications: notif.mock as any,
      services: { get: () => ({ getActiveModel: () => "gpt-4" }) } as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      layout: { setTitle: () => {} } as any,
      clipboard: { writeText: async () => {} } as any,
      download: { save: () => {} } as any,
      upload: {} as any,
      hooks: {
        on: () => ({ dispose() {} }),
        run: async (name: string, value: any) => {
          if (name === "chat.receive.after") {
            return { $block: true, reason: "Cannot block a response" };
          }
          return value;
        },
      } as any,
      timers: mockTimers(),
      observe: { intersection: () => ({ dispose() {} }), mutation: () => ({ dispose() {} }), resize: () => ({ dispose() {} }) } as any,
      format: { date: () => "", dateTime: () => "", relativeTime: () => "", duration: () => "", number: () => "" } as any,
      menus: {
        registerItem: () => ({ dispose() {} }),
        renderSlot: () => ({ dispose() {} }),
      } as any,
    });

    await streamResponse({ text: "Hello" });

    // Should fall back to original data, not crash.
    expect(chatState.activeChat).not.toBeNull();
    expect(chatState.activeChat!.title).toBe("Original");
  });
});

// ── Optimistic sidebar add (regression guard) ────────────────────────────────
//
// These tests guard the contract: "a new chat item appears in the sidebar the
// moment the user presses send, regardless of strict_responses mode."  This
// behavior has regressed multiple times — these are explicit regression guards.

describe("optimistic sidebar add", () => {
  test("new chat adds placeholder sidebar item immediately on send", async () => {
    chatState.activeSessionId = null;
    chatState.sessions = [];

    // Default mock returns 500, which causes early return — placeholder persists.
    await streamResponse({ text: "Hello", input: "Hello" });

    expect(chatState.sessions.length).toBeGreaterThanOrEqual(1);
    expect(chatState.sessions[0]!.id).toMatch(/^__pending_/);
    expect(chatState.activeSessionId).toMatch(/^__pending_/);
  });

  test("placeholder has correct title and project_id", async () => {
    chatState.activeSessionId = null;
    chatState.sessions = [];
    chatState.pendingProjectId = "proj-1";

    await streamResponse({ text: "Hello world from test", input: "Hello world from test" });

    const placeholder = chatState.sessions.find((s: any) => s.id.startsWith("__pending_"));
    expect(placeholder).toBeDefined();
    expect(placeholder!.title).toBe("Hello world from test");
    expect(placeholder!.project_id).toBe("proj-1");
  });

  test("discoverSessionId replaces placeholder with real session ID", async () => {
    chatState.activeSessionId = null;
    chatState.sessions = [];

    const sseBody = "event: response.completed\ndata: {}\n\n";
    const now = Math.floor(Date.now() / 1000);

    initChatDeps({
      api: {
        createResponse: async () => new Response(sseBody, { status: 200 }),
        listConversations: async (params?: any) => ({
          ok: true,
          data: {
            data: [{ id: "real-session-123", title: "T", object: "conversation", created_at: now, updated_at: now, model: "gpt-4", marker: "" }],
            has_more: false,
          },
        }),
        getConversation: async (id: string) => ({
          ok: true,
          data: { id, title: "Test", items: [], object: "conversation", created_at: now, updated_at: now, model: "gpt-4" },
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
      menus: {
        registerItem: () => ({ dispose() {} }),
        renderSlot: () => ({ dispose() {} }),
      } as any,
    });

    await streamResponse({ text: "Hello", input: "Hello", discoverSession: true });

    expect(chatState.activeSessionId).toBe("real-session-123");
    expect(chatState.sessions.some((s: any) => s.id.startsWith("__pending_"))).toBe(false);
  });

  test("onSessionDiscovered removes placeholder and adds real session", async () => {
    chatState.activeSessionId = null;
    chatState.sessions = [];

    const sseBody =
      "event: response.created\n" +
      'data: {"response":{"id":"resp-1","metadata":{"session_id":"real-session-456"}}}\n\n' +
      "event: response.completed\n" +
      'data: {"response":{"id":"resp-1"}}\n\n';
    const now = Math.floor(Date.now() / 1000);

    initChatDeps({
      api: {
        createResponse: async () => new Response(sseBody, { status: 200 }),
        listConversations: async () => ({
          ok: true,
          data: {
            data: [{ id: "real-session-456", title: "T", object: "conversation", created_at: now, updated_at: now, model: "gpt-4", marker: "" }],
            has_more: false,
          },
        }),
        getConversation: async (id: string) => ({
          ok: true,
          data: { id, title: "Test", items: [], object: "conversation", created_at: now, updated_at: now, model: "gpt-4" },
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
      menus: {
        registerItem: () => ({ dispose() {} }),
        renderSlot: () => ({ dispose() {} }),
      } as any,
    });

    await streamResponse({ text: "Hello", input: "Hello" });

    expect(chatState.sessions.some((s: any) => s.id.startsWith("__pending_"))).toBe(false);
    expect(chatState.sessions.some((s: any) => s.id === "real-session-456")).toBe(true);
    expect(chatState.activeSessionId).toBe("real-session-456");
  });

  test("existing session does not create placeholder", async () => {
    chatState.activeSessionId = "existing-session";
    chatState.sessions = [];

    await streamResponse({ text: "Hello", input: "Hello" });

    expect(chatState.sessions.every((s: any) => !s.id.startsWith("__pending_"))).toBe(true);
  });
});
