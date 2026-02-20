import { describe, test, expect, beforeEach } from "bun:test";
import { chatState } from "../../../src/plugins/chat/state.ts";
import { initChatDom, getChatDom } from "../../../src/plugins/chat/dom.ts";
import { initChatDeps } from "../../../src/plugins/chat/deps.ts";
import { createDomRoot, CHAT_DOM_IDS, CHAT_DOM_TAGS } from "../../helpers/dom.ts";
import { mockTimers, mockNotifications } from "../../helpers/mocks.ts";
import type {
  Conversation,
  MessageItem,
  InputTextPart,
  InputImagePart,
  InputContentItem,
  OutputTextPart,
} from "../../../src/types.ts";

/**
 * Tests for message action helpers — copy and edit user/assistant messages.
 *
 * handleCopyUserMessage, handleCopyAssistantMessage, and handleEditUserMessage
 * depend on DOM and clipboard. We test the pure helper `replaceInputText`
 * indirectly and the copy flow via mock clipboard.
 */

let notifs: ReturnType<typeof mockNotifications>;
let clipboardText: string | null;

function userMsg(text: string, ...extraParts: any[]): MessageItem {
  return {
    type: "message",
    role: "user",
    content: [{ type: "input_text", text } as InputTextPart, ...extraParts],
  } as MessageItem;
}

function assistantMsg(text: string): MessageItem {
  return {
    type: "message",
    role: "assistant",
    content: [{ type: "output_text", text } as OutputTextPart],
  } as MessageItem;
}

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
  } as Conversation;
}

beforeEach(() => {
  notifs = mockNotifications();
  clipboardText = null;

  chatState.activeChat = null;
  chatState.activeSessionId = null;
  chatState.sessions = [];

  initChatDom(createDomRoot(CHAT_DOM_IDS, undefined, CHAT_DOM_TAGS));

  initChatDeps({
    api: {} as any,
    notifications: notifs.mock,
    timers: mockTimers(),
    services: {} as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    hooks: { run: async (_name: string, payload: any) => payload } as any,
    clipboard: {
      writeText: async (text: string) => { clipboardText = text; },
    } as any,
    download: {} as any,
    observe: { onResize: () => ({ dispose() {} }) } as any,
    format: { dateTime: () => "" } as any,
    upload: { upload: async () => ({}) } as any,
    menus: { registerItem: () => ({ dispose() {} }), renderSlot: () => ({ dispose() {} }) } as any,
  });
});

// We need to import after deps are set up (they're module-level singletons)
// But since bun:test imports eagerly, we import at top and rely on beforeEach
// to reset state.

describe("handleCopyUserMessage", () => {
  // Import the function dynamically to test it
  test("copies user message text to clipboard", async () => {
    const { handleCopyUserMessage } = await import("../../../src/plugins/chat/message-actions.ts");
    chatState.activeChat = makeConvo([
      userMsg("first message"),
      assistantMsg("response"),
      userMsg("second message"),
    ]);
    chatState.activeSessionId = "sess-1";
    handleCopyUserMessage(1);
    // Allow async clipboard promise to resolve
    await new Promise((r) => setTimeout(r, 0));
    expect(clipboardText).toBe("second message");
  });
});

describe("handleCopyAssistantMessage", () => {
  test("copies assistant message text to clipboard", async () => {
    const { handleCopyAssistantMessage } = await import("../../../src/plugins/chat/message-actions.ts");
    chatState.activeChat = makeConvo([
      userMsg("hello"),
      assistantMsg("I am an assistant"),
      userMsg("bye"),
      assistantMsg("Goodbye"),
    ]);
    chatState.activeSessionId = "sess-1";
    handleCopyAssistantMessage(0);
    await new Promise((r) => setTimeout(r, 0));
    expect(clipboardText).toBe("I am an assistant");
  });

  test("copies second assistant message by index", async () => {
    const { handleCopyAssistantMessage } = await import("../../../src/plugins/chat/message-actions.ts");
    chatState.activeChat = makeConvo([
      userMsg("hello"),
      assistantMsg("first"),
      userMsg("again"),
      assistantMsg("second"),
    ]);
    chatState.activeSessionId = "sess-1";
    handleCopyAssistantMessage(1);
    await new Promise((r) => setTimeout(r, 0));
    expect(clipboardText).toBe("second");
  });

  test("no-op when no active chat", async () => {
    const { handleCopyAssistantMessage } = await import("../../../src/plugins/chat/message-actions.ts");
    chatState.activeChat = null;
    handleCopyAssistantMessage(0);
    await new Promise((r) => setTimeout(r, 0));
    expect(clipboardText).toBeNull();
  });
});

// ── handleEditUserMessage ───────────────────────────────────────────────────

/**
 * Build a minimal transcript DOM structure that handleEditUserMessage expects:
 *   #transcript > [data-transcript-items] > .user-msg > div (bubble)
 */
function buildTranscriptDom(messages: { role: string; text: string }[]): void {
  const tc = getChatDom().transcriptContainer;
  tc.innerHTML = "";
  const items = document.createElement("div");
  items.dataset["transcriptItems"] = "";

  for (const msg of messages) {
    const msgEl = document.createElement("div");
    msgEl.className = msg.role === "user" ? "user-msg" : "assistant-msg";
    const bubble = document.createElement("div");
    const textDiv = document.createElement("div");
    textDiv.textContent = msg.text;
    bubble.appendChild(textDiv);
    msgEl.appendChild(bubble);
    items.appendChild(msgEl);
  }

  tc.appendChild(items);
}

describe("handleEditUserMessage", () => {
  test("no-op when no active chat", async () => {
    const { handleEditUserMessage } = await import("../../../src/plugins/chat/message-actions.ts");
    chatState.activeChat = null;
    handleEditUserMessage(0);
    // Should not create any textarea
    const tc = getChatDom().transcriptContainer;
    expect(tc.querySelector(".edit-textarea")).toBeNull();
  });

  test("creates textarea editor in user message bubble", async () => {
    const { handleEditUserMessage } = await import("../../../src/plugins/chat/message-actions.ts");
    chatState.activeChat = makeConvo([userMsg("hello world")]);
    chatState.activeSessionId = "sess-1";
    buildTranscriptDom([{ role: "user", text: "hello world" }]);

    handleEditUserMessage(0);

    const tc = getChatDom().transcriptContainer;
    const textarea = tc.querySelector<HTMLTextAreaElement>(".edit-textarea")!;
    expect(textarea).not.toBeNull();
    expect(textarea.value).toBe("hello world");
  });

  test("textarea rows match line count (min 2)", async () => {
    const { handleEditUserMessage } = await import("../../../src/plugins/chat/message-actions.ts");
    chatState.activeChat = makeConvo([userMsg("line1\nline2\nline3\nline4")]);
    chatState.activeSessionId = "sess-1";
    buildTranscriptDom([{ role: "user", text: "line1\nline2\nline3\nline4" }]);

    handleEditUserMessage(0);

    const textarea = getChatDom().transcriptContainer.querySelector<HTMLTextAreaElement>(".edit-textarea")!;
    expect(String(textarea.rows)).toBe("4");
  });

  test("single-line message gets minimum 2 rows", async () => {
    const { handleEditUserMessage } = await import("../../../src/plugins/chat/message-actions.ts");
    chatState.activeChat = makeConvo([userMsg("short")]);
    chatState.activeSessionId = "sess-1";
    buildTranscriptDom([{ role: "user", text: "short" }]);

    handleEditUserMessage(0);

    const textarea = getChatDom().transcriptContainer.querySelector<HTMLTextAreaElement>(".edit-textarea")!;
    expect(String(textarea.rows)).toBe("2");
  });

  test("renders Save & Submit and Cancel buttons", async () => {
    const { handleEditUserMessage } = await import("../../../src/plugins/chat/message-actions.ts");
    chatState.activeChat = makeConvo([userMsg("hello")]);
    chatState.activeSessionId = "sess-1";
    buildTranscriptDom([{ role: "user", text: "hello" }]);

    handleEditUserMessage(0);

    const tc = getChatDom().transcriptContainer;
    const actions = tc.querySelector(".edit-actions")!;
    expect(actions).not.toBeNull();
    const buttons = actions.querySelectorAll("button");
    expect(buttons[0]!.textContent).toBe("Save & Submit");
    expect(buttons[1]!.textContent).toBe("Cancel");
  });

  test("Cancel restores original content", async () => {
    const { handleEditUserMessage } = await import("../../../src/plugins/chat/message-actions.ts");
    chatState.activeChat = makeConvo([userMsg("original text")]);
    chatState.activeSessionId = "sess-1";
    buildTranscriptDom([{ role: "user", text: "original text" }]);

    handleEditUserMessage(0);

    const tc = getChatDom().transcriptContainer;
    const cancelBtn = tc.querySelector<HTMLButtonElement>(".btn-ghost")!;
    cancelBtn.click();

    // Textarea should be gone, original content restored
    expect(tc.querySelector(".edit-textarea")).toBeNull();
    const bubble = tc.querySelector(".user-msg div")!;
    expect(bubble.textContent).toContain("original text");
  });

  test("Escape key cancels editing", async () => {
    const { handleEditUserMessage } = await import("../../../src/plugins/chat/message-actions.ts");
    chatState.activeChat = makeConvo([userMsg("hello")]);
    chatState.activeSessionId = "sess-1";
    buildTranscriptDom([{ role: "user", text: "hello" }]);

    handleEditUserMessage(0);

    const tc = getChatDom().transcriptContainer;
    const textarea = tc.querySelector<HTMLTextAreaElement>(".edit-textarea")!;
    textarea.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));

    expect(tc.querySelector(".edit-textarea")).toBeNull();
  });

  test("no-op when already editing (prevents double textarea)", async () => {
    const { handleEditUserMessage } = await import("../../../src/plugins/chat/message-actions.ts");
    chatState.activeChat = makeConvo([userMsg("hello")]);
    chatState.activeSessionId = "sess-1";
    buildTranscriptDom([{ role: "user", text: "hello" }]);

    handleEditUserMessage(0);
    handleEditUserMessage(0); // second call

    const textareas = getChatDom().transcriptContainer.querySelectorAll(".edit-textarea");
    expect(textareas.length).toBe(1);
  });

  test("preserves image elements during edit", async () => {
    const { handleEditUserMessage } = await import("../../../src/plugins/chat/message-actions.ts");
    chatState.activeChat = makeConvo([userMsg("hello", { type: "input_image", image_url: "test.jpg" })]);
    chatState.activeSessionId = "sess-1";

    // Build DOM with an image inside the bubble
    const tc = getChatDom().transcriptContainer;
    tc.innerHTML = "";
    const items = document.createElement("div");
    items.dataset["transcriptItems"] = "";
    const msgEl = document.createElement("div");
    msgEl.className = "user-msg";
    const bubble = document.createElement("div");
    const textDiv = document.createElement("div");
    textDiv.textContent = "hello";
    const imgDiv = document.createElement("div");
    imgDiv.className = "msg-image";
    bubble.appendChild(textDiv);
    bubble.appendChild(imgDiv);
    msgEl.appendChild(bubble);
    items.appendChild(msgEl);
    tc.appendChild(items);

    handleEditUserMessage(0);

    // Image should still be present
    expect(bubble.querySelector(".msg-image")).not.toBeNull();
    // But text div should be replaced by textarea
    expect(bubble.querySelector(".edit-textarea")).not.toBeNull();
  });

  test("edits correct message by index", async () => {
    const { handleEditUserMessage } = await import("../../../src/plugins/chat/message-actions.ts");
    chatState.activeChat = makeConvo([
      userMsg("first"),
      assistantMsg("response"),
      userMsg("second"),
    ]);
    chatState.activeSessionId = "sess-1";
    buildTranscriptDom([
      { role: "user", text: "first" },
      { role: "assistant", text: "response" },
      { role: "user", text: "second" },
    ]);

    handleEditUserMessage(1); // second user message

    const tc = getChatDom().transcriptContainer;
    const textarea = tc.querySelector<HTMLTextAreaElement>(".edit-textarea")!;
    expect(textarea.value).toBe("second");
  });
});
