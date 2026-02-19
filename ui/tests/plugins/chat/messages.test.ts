import { describe, test, expect, beforeEach } from "bun:test";
import {
  scrollToBottomIfNear,
  appendUserMessage,
  appendAssistantPlaceholder,
  appendStoppedIndicator,
  addAssistantActionButtons,
} from "../../../src/plugins/chat/messages.ts";
import { initChatDom, getChatDom } from "../../../src/plugins/chat/dom.ts";
import { initChatDeps } from "../../../src/plugins/chat/deps.ts";
import { createDomRoot, CHAT_DOM_IDS, CHAT_DOM_TAGS } from "../../helpers/dom.ts";
import { mockTimers, mockNotifications } from "../../helpers/mocks.ts";
import type { Conversation } from "../../../src/types.ts";

let notifs: ReturnType<typeof mockNotifications>;

beforeEach(() => {
  notifs = mockNotifications();
  initChatDom(createDomRoot(CHAT_DOM_IDS, undefined, CHAT_DOM_TAGS));
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

// ── scrollToBottomIfNear ────────────────────────────────────────────────────

describe("scrollToBottomIfNear", () => {
  test("scrolls when distance from bottom < 80", () => {
    const tc = getChatDom().transcriptContainer;
    // Simulate: scrollHeight=1000, clientHeight=500, scrollTop=430
    // distance = 1000 - 430 - 500 = 70 < 80 → should scroll
    Object.defineProperty(tc, "scrollHeight", { value: 1000, configurable: true });
    Object.defineProperty(tc, "clientHeight", { value: 500, configurable: true });
    tc.scrollTop = 430;
    scrollToBottomIfNear();
    expect(tc.scrollTop).toBe(1000);
  });

  test("does not scroll when distance from bottom >= 80", () => {
    const tc = getChatDom().transcriptContainer;
    Object.defineProperty(tc, "scrollHeight", { value: 1000, configurable: true });
    Object.defineProperty(tc, "clientHeight", { value: 500, configurable: true });
    tc.scrollTop = 400; // distance = 1000 - 400 - 500 = 100 >= 80
    scrollToBottomIfNear();
    expect(tc.scrollTop).toBe(400);
  });

  test("scrolls when exactly at bottom", () => {
    const tc = getChatDom().transcriptContainer;
    Object.defineProperty(tc, "scrollHeight", { value: 1000, configurable: true });
    Object.defineProperty(tc, "clientHeight", { value: 500, configurable: true });
    tc.scrollTop = 500; // distance = 0
    scrollToBottomIfNear();
    expect(tc.scrollTop).toBe(1000);
  });
});

// ── appendUserMessage ───────────────────────────────────────────────────────

describe("appendUserMessage", () => {
  test("creates user-msg wrapper inside transcript-items", () => {
    appendUserMessage("hello");
    const tc = getChatDom().transcriptContainer;
    const wrapper = tc.querySelector(".user-msg");
    expect(wrapper).not.toBeNull();
  });

  test("creates transcript-items container if missing", () => {
    appendUserMessage("test");
    const tc = getChatDom().transcriptContainer;
    expect(tc.querySelector("[data-transcript-items]")).not.toBeNull();
  });

  test("renders message text in bubble", () => {
    appendUserMessage("hello world");
    const tc = getChatDom().transcriptContainer;
    const bubble = tc.querySelector(".user-bubble");
    expect(bubble).not.toBeNull();
    expect(bubble!.textContent).toContain("hello world");
  });

  test("removes empty state placeholder if present", () => {
    const tc = getChatDom().transcriptContainer;
    const placeholder = document.createElement("div");
    placeholder.dataset["emptyState"] = "";
    tc.appendChild(placeholder);

    appendUserMessage("test");
    expect(tc.querySelector("[data-empty-state]")).toBeNull();
  });

  test("appends inline image from structured input", () => {
    const input = [
      {
        content: [
          { type: "input_image", image_url: "http://example.com/test.jpg" },
        ],
      },
    ];
    appendUserMessage("with image", input as any);
    const tc = getChatDom().transcriptContainer;
    const msgImage = tc.querySelector(".msg-image");
    expect(msgImage).not.toBeNull();
    const img = msgImage!.querySelector("img");
    expect(img).not.toBeNull();
  });

  test("appends file pill from structured input", () => {
    const input = [
      {
        content: [
          { type: "input_file", filename: "doc.pdf" },
        ],
      },
    ];
    appendUserMessage("with file", input as any);
    const tc = getChatDom().transcriptContainer;
    const pill = tc.querySelector(".msg-file-pill");
    expect(pill).not.toBeNull();
    expect(pill!.textContent).toBe("doc.pdf");
  });

  test("includes user action buttons", () => {
    appendUserMessage("test");
    const tc = getChatDom().transcriptContainer;
    const wrapper = tc.querySelector(".user-msg")!;
    const actions = wrapper.querySelector(".user-actions, .user-action-btn");
    // createUserActionButtons creates the action button container
    expect(wrapper.children.length).toBeGreaterThanOrEqual(2);
  });
});

// ── appendAssistantPlaceholder ──────────────────────────────────────────────

describe("appendAssistantPlaceholder", () => {
  test("creates assistant-msg wrapper", () => {
    const { wrapper } = appendAssistantPlaceholder();
    expect(wrapper.className).toBe("assistant-msg");
  });

  test("returns body with assistant-body class", () => {
    const { body } = appendAssistantPlaceholder();
    expect(body.className).toBe("assistant-body");
  });

  test("returns textEl with markdown-body class", () => {
    const { textEl } = appendAssistantPlaceholder();
    expect(textEl.className).toBe("markdown-body");
  });

  test("creates transcript-items container if missing", () => {
    appendAssistantPlaceholder();
    const tc = getChatDom().transcriptContainer;
    expect(tc.querySelector("[data-transcript-items]")).not.toBeNull();
  });

  test("appends to existing transcript-items", () => {
    appendUserMessage("user first");
    appendAssistantPlaceholder();
    const items = getChatDom().transcriptContainer.querySelector("[data-transcript-items]")!;
    expect(items.children.length).toBe(2);
  });
});

// ── appendStoppedIndicator ──────────────────────────────────────────────────

describe("appendStoppedIndicator", () => {
  test("appends stopped-indicator to parent", () => {
    const parent = document.createElement("div");
    const textEl = document.createElement("div");
    parent.appendChild(textEl);

    appendStoppedIndicator(textEl);
    const indicator = parent.querySelector(".stopped-indicator");
    expect(indicator).not.toBeNull();
    expect(indicator!.textContent).toBe("Stopped");
  });
});

// ── addAssistantActionButtons ───────────────────────────────────────────────

describe("addAssistantActionButtons", () => {
  test("appends action buttons to wrapper", () => {
    const wrapper = document.createElement("div");
    const chat: Conversation = {
      id: "conv-1",
      items: [
        { type: "message", role: "assistant", content: [{ type: "output_text", text: "reply" }] },
      ],
    } as Conversation;

    addAssistantActionButtons(wrapper, chat);
    // Should have appended an element (the action button container)
    expect(wrapper.children.length).toBeGreaterThanOrEqual(1);
  });

  test("finds generation settings from last assistant message", () => {
    const wrapper = document.createElement("div");
    const chat: Conversation = {
      id: "conv-1",
      items: [
        { type: "message", role: "user", content: [] },
        {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "answer" }],
          generation: { model: "gpt-4", temperature: 0.7 },
        },
      ],
    } as Conversation;

    addAssistantActionButtons(wrapper, chat);
    // Should not throw and should append buttons
    expect(wrapper.children.length).toBeGreaterThanOrEqual(1);
  });
});
