import { describe, test, expect, beforeEach } from "bun:test";
import { setupTranscriptEvents } from "../../../src/plugins/chat/transcript-events.ts";
import { chatState } from "../../../src/plugins/chat/state.ts";
import { initChatDom, getChatDom } from "../../../src/plugins/chat/dom.ts";
import { initChatDeps } from "../../../src/plugins/chat/deps.ts";
import { createDomRoot, CHAT_DOM_IDS, CHAT_DOM_TAGS } from "../../helpers/dom.ts";
import { mockTimers, mockNotifications } from "../../helpers/mocks.ts";
import type { Conversation, OutputTextPart } from "../../../src/types.ts";

/**
 * Tests for transcript-events — click delegation on the transcript container.
 * Covers: image lightbox, user/assistant action dispatch, tag buttons,
 * and chat-level action buttons.
 */

let notifs: ReturnType<typeof mockNotifications>;
let clipboardText: string | null;
let domRoot: HTMLElement;

function makeConvo(items: any[] = []): Conversation {
  return {
    id: "conv-1",
    object: "conversation",
    title: "Test",
    created_at: 1700000000,
    updated_at: 1700000000,
    model: "gpt-4",
    items,
    metadata: {},
    tags: [],
  } as Conversation;
}

beforeEach(() => {
  notifs = mockNotifications();
  clipboardText = null;

  domRoot = createDomRoot(CHAT_DOM_IDS, undefined, CHAT_DOM_TAGS);
  initChatDom(domRoot);

  chatState.activeChat = makeConvo();
  chatState.activeSessionId = "sess-1";
  chatState.sessions = [];

  initChatDeps({
    api: {
      addConversationTags: async () => ({ ok: true, data: { tags: [] } }),
      removeConversationTags: async () => ({ ok: true, data: { tags: [] } }),
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
    clipboard: {
      writeText: async (text: string) => { clipboardText = text; },
    } as any,
    download: {} as any,
    observe: { onResize: () => ({ dispose() {} }) } as any,
    format: { dateTime: () => "2024-01-01" } as any,
    upload: { upload: async () => ({}) } as any,
    menus: { registerItem: () => ({ dispose() {} }), renderSlot: () => ({ dispose() {} }) } as any,
  });

  setupTranscriptEvents();
});

/** Build a transcript items container with child elements. */
function buildTranscript(children: HTMLElement[]): void {
  const tc = getChatDom().transcriptContainer;
  const items = document.createElement("div");
  items.dataset["transcriptItems"] = "";
  for (const child of children) items.appendChild(child);
  tc.appendChild(items);
}

// ── Image lightbox ──────────────────────────────────────────────────────────

describe("setupTranscriptEvents — image lightbox", () => {
  function makeImageInTranscript(): HTMLImageElement {
    const tc = getChatDom().transcriptContainer;
    const msgImage = document.createElement("div");
    msgImage.className = "msg-image";
    const img = document.createElement("img");
    img.src = "http://example.com/photo.jpg";
    msgImage.appendChild(img);
    tc.appendChild(msgImage);
    return img;
  }

  test("clicking inline image creates lightbox overlay", () => {
    const img = makeImageInTranscript();
    img.click();
    const overlay = document.querySelector(".image-lightbox");
    expect(overlay).not.toBeNull();
  });

  test("lightbox contains an img with same src", () => {
    const img = makeImageInTranscript();
    img.click();
    const lightboxImg = document.querySelector<HTMLImageElement>(".image-lightbox img");
    expect(lightboxImg).not.toBeNull();
    expect(lightboxImg!.src).toBe("http://example.com/photo.jpg");
  });

  test("lightbox is appended to document.body", () => {
    const img = makeImageInTranscript();
    img.click();
    const overlay = document.body.querySelector(".image-lightbox");
    expect(overlay).not.toBeNull();
  });

  test("clicking lightbox overlay dismisses it", () => {
    const img = makeImageInTranscript();
    img.click();
    const overlay = document.querySelector<HTMLElement>(".image-lightbox")!;
    overlay.click();
    expect(document.querySelector(".image-lightbox")).toBeNull();
  });

  test("Escape key dismisses lightbox", () => {
    const img = makeImageInTranscript();
    img.click();
    expect(document.querySelector(".image-lightbox")).not.toBeNull();
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(document.querySelector(".image-lightbox")).toBeNull();
  });

  test("clicking non-image element does not create lightbox", () => {
    const tc = getChatDom().transcriptContainer;
    const div = document.createElement("div");
    div.className = "msg-image";
    const span = document.createElement("span");
    div.appendChild(span);
    tc.appendChild(div);
    span.click();
    expect(document.querySelector(".image-lightbox")).toBeNull();
  });
});

// ── User action buttons ─────────────────────────────────────────────────────

describe("setupTranscriptEvents — user actions", () => {
  function setupUserMsg(text: string): { btn: HTMLButtonElement; msgEl: HTMLElement } {
    chatState.activeChat = makeConvo([
      { type: "message", role: "user", content: [{ type: "input_text", text }] },
    ]);

    const msgEl = document.createElement("div");
    msgEl.className = "user-msg";
    const bubble = document.createElement("div");
    bubble.textContent = text;
    msgEl.appendChild(bubble);

    const btn = document.createElement("button");
    btn.className = "user-action-btn";
    msgEl.appendChild(btn);

    buildTranscript([msgEl]);
    return { btn, msgEl };
  }

  test("copy action calls handleCopyUserMessage", async () => {
    const { btn } = setupUserMsg("hello world");
    btn.dataset["action"] = "copy";
    btn.click();
    await new Promise((r) => setTimeout(r, 0));
    expect(clipboardText).toBe("hello world");
  });

  test("unknown action does not throw", () => {
    const { btn } = setupUserMsg("hello");
    btn.dataset["action"] = "unknown";
    expect(() => btn.click()).not.toThrow();
  });
});

// ── Assistant action buttons ────────────────────────────────────────────────

describe("setupTranscriptEvents — assistant actions", () => {
  function setupAssistantMsg(text: string): { btn: HTMLButtonElement } {
    chatState.activeChat = makeConvo([
      { type: "message", role: "assistant", content: [{ type: "output_text", text } as OutputTextPart] },
    ]);

    const msgEl = document.createElement("div");
    msgEl.className = "assistant-msg";
    const bubble = document.createElement("div");
    bubble.textContent = text;
    msgEl.appendChild(bubble);

    const btn = document.createElement("button");
    btn.className = "assistant-action-btn";
    msgEl.appendChild(btn);

    buildTranscript([msgEl]);
    return { btn };
  }

  test("copy action calls handleCopyAssistantMessage", async () => {
    const { btn } = setupAssistantMsg("assistant reply");
    btn.dataset["action"] = "copy";
    btn.click();
    await new Promise((r) => setTimeout(r, 0));
    expect(clipboardText).toBe("assistant reply");
  });

  test("show-message-details opens read-only panel", () => {
    const { btn } = setupAssistantMsg("reply");
    btn.dataset["action"] = "show-message-details";
    btn.dataset["generation"] = JSON.stringify({ model: "gpt-4", temperature: 0.5 });
    btn.dataset["usage"] = JSON.stringify({ output_tokens: 100 });
    btn.click();
    const dom = getChatDom();
    expect(dom.rightPanel.classList.contains("read-only")).toBe(true);
  });
});

// ── Tag buttons ─────────────────────────────────────────────────────────────

describe("setupTranscriptEvents — tag buttons", () => {
  test("add-tag-btn click triggers tag prompt", () => {
    const tc = getChatDom().transcriptContainer;
    // Add a tags container for handleAddTagPrompt to find
    const tagsContainer = document.createElement("div");
    tagsContainer.className = "transcript-tags";
    tc.appendChild(tagsContainer);

    const addBtn = document.createElement("button");
    addBtn.className = "add-tag-btn";
    tagsContainer.appendChild(addBtn);
    addBtn.click();

    // handleAddTagPrompt creates an input wrapper
    expect(tagsContainer.querySelector(".tag-input-wrapper")).not.toBeNull();
  });
});

// ── Chat-level action buttons ───────────────────────────────────────────────

describe("setupTranscriptEvents — chat actions", () => {
  test("toggle-thinking toggles button active class", () => {
    const tc = getChatDom().transcriptContainer;
    const btn = document.createElement("button");
    btn.className = "chat-action";
    btn.dataset["action"] = "toggle-thinking";
    tc.appendChild(btn);

    btn.click();
    expect(btn.classList.contains("active")).toBe(true);
    btn.click();
    expect(btn.classList.contains("active")).toBe(false);
  });

  test("toggle-tuning opens right panel", () => {
    const dom = getChatDom();
    dom.rightPanel.classList.add("hidden");

    const btn = document.createElement("button");
    btn.className = "chat-action";
    btn.dataset["action"] = "toggle-tuning";
    dom.transcriptContainer.appendChild(btn);

    btn.click();
    expect(dom.rightPanel.classList.contains("hidden")).toBe(false);
  });
});
