import { describe, test, expect, beforeEach } from "bun:test";
import {
  renderTranscript,
  renderOutputText,
  createUserActionButtons,
  createAssistantActionButtons,
  renderTranscriptHeader,
  initCodeBlockCopyHandler,
} from "../../src/render/transcript.ts";
import { initThinkingState, setThinkingExpanded } from "../../src/render/helpers.ts";
import type { Item, MessageItem, ReasoningItem, FunctionCallItem, FunctionCallOutputItem, Conversation, CodeBlock } from "../../src/types.ts";
import { mockTimers } from "../helpers/mocks.ts";

/**
 * Structural tests for transcript rendering — verifies DOM structure,
 * classes, data attributes, and element composition for all item types.
 */

beforeEach(() => {
  initThinkingState(false, () => {});
  initCodeBlockCopyHandler(document, { writeText: async () => {} } as any, mockTimers());
});

// -- Helpers -----------------------------------------------------------------

function makeUserMessage(text: string): MessageItem {
  return {
    type: "message",
    role: "user",
    content: [{ type: "input_text", text }],
  } as MessageItem;
}

function makeAssistantMessage(text: string, generation?: any): MessageItem {
  return {
    type: "message",
    role: "assistant",
    content: [{ type: "output_text", text }],
    generation,
  } as MessageItem;
}

function makeReasoning(text: string, finishReason?: string): ReasoningItem {
  return {
    type: "reasoning",
    content: [{ type: "reasoning", text }],
    finish_reason: finishReason,
  } as ReasoningItem;
}

function makeFunctionCall(name: string, args: string): FunctionCallItem {
  return { type: "function_call", name, arguments: args, call_id: "call-1" } as FunctionCallItem;
}

function makeFunctionOutput(output: string): FunctionCallOutputItem {
  return { type: "function_call_output", output, call_id: "call-1" } as FunctionCallOutputItem;
}

function makeConvo(id: string, overrides: Partial<Conversation> = {}): Conversation {
  return {
    id,
    object: "conversation",
    title: "Test Chat",
    created_at: 1700000000,
    updated_at: 1700000000,
    model: "gpt-4",
    items: [],
    marker: "",
    ...overrides,
  } as Conversation;
}

// ── createUserActionButtons ──────────────────────────────────────────────────

describe("createUserActionButtons", () => {
  test("creates container with user-msg-actions class", () => {
    const el = createUserActionButtons();
    expect(el.className).toBe("user-msg-actions");
  });

  test("contains rerun, edit, and copy buttons", () => {
    const el = createUserActionButtons();
    const buttons = el.querySelectorAll("[data-action]");
    const actions = Array.from(buttons).map((b) => (b as HTMLElement).dataset["action"]);
    expect(actions).toContain("rerun");
    expect(actions).toContain("edit");
    expect(actions).toContain("copy");
  });

  test("all buttons have user-action-btn class", () => {
    const el = createUserActionButtons();
    const buttons = el.querySelectorAll(".user-action-btn");
    expect(buttons.length).toBe(3);
  });
});

// ── createAssistantActionButtons ──────────────────────────────────────────────

describe("createAssistantActionButtons", () => {
  test("creates container with assistant-msg-actions class", () => {
    const el = createAssistantActionButtons();
    expect(el.className).toBe("assistant-msg-actions");
  });

  test("always includes copy button", () => {
    const el = createAssistantActionButtons();
    const copyBtn = el.querySelector("[data-action='copy']");
    expect(copyBtn).not.toBeNull();
  });

  test("includes stats button when generation data provided", () => {
    const el = createAssistantActionButtons({
      generation: { model: "gpt-4", temperature: 0.7 } as any,
    });
    const statsBtn = el.querySelector("[data-action='show-message-details']");
    expect(statsBtn).not.toBeNull();
    expect((statsBtn as HTMLElement).dataset["generation"]).toBeDefined();
  });

  test("includes stats button when usage data provided", () => {
    const el = createAssistantActionButtons({
      usage: { input_tokens: 10, output_tokens: 20, total_tokens: 30 } as any,
    });
    const statsBtn = el.querySelector("[data-action='show-message-details']");
    expect(statsBtn).not.toBeNull();
    expect((statsBtn as HTMLElement).dataset["usage"]).toBeDefined();
  });

  test("no stats button when no generation or usage", () => {
    const el = createAssistantActionButtons();
    const statsBtn = el.querySelector("[data-action='show-message-details']");
    expect(statsBtn).toBeNull();
  });
});

// ── renderOutputText ─────────────────────────────────────────────────────────

describe("renderOutputText", () => {
  test("renders markdown when no code blocks provided", () => {
    const el = renderOutputText("Hello **world**");
    expect(el.className).toBe("markdown-body");
    expect(el.innerHTML).toContain("<strong>");
  });

  test("renders code blocks from metadata", () => {
    const text = "Before\n```python\nprint('hi')\n```\nAfter";
    const codeBlocks: CodeBlock[] = [{
      fence_start: 7,
      fence_end: text.length - 6,
      language_start: 10,
      language_end: 16,
      content_start: 17,
      content_end: 28,
      complete: true,
    }];
    const el = renderOutputText(text, codeBlocks);
    const block = el.querySelector(".code-block");
    expect(block).not.toBeNull();
  });

  test("code block has language label", () => {
    const text = "```javascript\nconsole.log('test')\n```";
    const codeBlocks: CodeBlock[] = [{
      fence_start: 0,
      fence_end: text.length,
      language_start: 3,
      language_end: 13,
      content_start: 14,
      content_end: 33,
      complete: true,
    }];
    const el = renderOutputText(text, codeBlocks);
    const lang = el.querySelector(".code-lang");
    expect(lang).not.toBeNull();
    expect(lang!.textContent).toBe("javascript");
  });

  test("incomplete code block shows indicator", () => {
    const text = "```python\nprint('hi')";
    const codeBlocks: CodeBlock[] = [{
      fence_start: 0,
      fence_end: text.length,
      language_start: 3,
      language_end: 9,
      content_start: 10,
      content_end: text.length,
      complete: false,
    }];
    const el = renderOutputText(text, codeBlocks);
    const incomplete = el.querySelector(".code-incomplete");
    expect(incomplete).not.toBeNull();
  });

  test("code block has copy button", () => {
    const text = "```python\nprint('hi')\n```";
    const codeBlocks: CodeBlock[] = [{
      fence_start: 0,
      fence_end: text.length,
      language_start: 3,
      language_end: 9,
      content_start: 10,
      content_end: 21,
      complete: true,
    }];
    const el = renderOutputText(text, codeBlocks);
    const copyBtn = el.querySelector(".code-copy");
    expect(copyBtn).not.toBeNull();
  });

  test("renders prose between code blocks", () => {
    const text = "Before code\n```py\nx=1\n```\nAfter code";
    const codeBlocks: CodeBlock[] = [{
      fence_start: 12,
      fence_end: 25,
      language_start: 15,
      language_end: 17,
      content_start: 18,
      content_end: 21,
      complete: true,
    }];
    const el = renderOutputText(text, codeBlocks);
    // Should have prose + code block + prose
    expect(el.children.length).toBeGreaterThanOrEqual(2);
  });

  test("code element has language-* class for highlighting", () => {
    const text = "```python\nprint('hi')\n```";
    const codeBlocks: CodeBlock[] = [{
      fence_start: 0,
      fence_end: text.length,
      language_start: 3,
      language_end: 9,
      content_start: 10,
      content_end: 21,
      complete: true,
    }];
    const el = renderOutputText(text, codeBlocks);
    const code = el.querySelector("code");
    expect(code).not.toBeNull();
    expect(code!.className).toBe("language-python");
  });
});

// ── renderTranscript ─────────────────────────────────────────────────────────

describe("renderTranscript", () => {
  test("wraps items in transcript-messages container", () => {
    const items: Item[] = [makeUserMessage("Hi")];
    const el = renderTranscript(items);
    expect(el.className).toBe("transcript-messages");
    expect(el.dataset["transcriptItems"]).toBeDefined();
  });

  test("renders user message as user-msg wrapper with user-bubble", () => {
    const items: Item[] = [makeUserMessage("Hello")];
    const el = renderTranscript(items);
    const userMsg = el.querySelector(".user-msg");
    expect(userMsg).not.toBeNull();
    const bubble = userMsg!.querySelector(".user-bubble");
    expect(bubble).not.toBeNull();
    expect(bubble!.textContent).toContain("Hello");
  });

  test("user message has action buttons", () => {
    const items: Item[] = [makeUserMessage("Hello")];
    const el = renderTranscript(items);
    const actions = el.querySelector(".user-msg-actions");
    expect(actions).not.toBeNull();
  });

  test("renders assistant message as assistant-msg with assistant-body", () => {
    const items: Item[] = [makeAssistantMessage("Response here")];
    const el = renderTranscript(items);
    const assistantMsg = el.querySelector(".assistant-msg");
    expect(assistantMsg).not.toBeNull();
    const body = assistantMsg!.querySelector(".assistant-body");
    expect(body).not.toBeNull();
  });

  test("assistant message has action buttons", () => {
    const items: Item[] = [makeAssistantMessage("Response")];
    const el = renderTranscript(items);
    const actions = el.querySelector(".assistant-msg-actions");
    expect(actions).not.toBeNull();
  });

  test("renders reasoning as details.reasoning-block", () => {
    const items: Item[] = [makeReasoning("Thinking about it...")];
    const el = renderTranscript(items);
    const details = el.querySelector("details.reasoning-block");
    expect(details).not.toBeNull();
    const summary = details!.querySelector("summary");
    expect(summary!.textContent).toBe("Thought process");
  });

  test("reasoning is collapsed by default", () => {
    initThinkingState(false, () => {});
    const items: Item[] = [makeReasoning("Thinking...")];
    const el = renderTranscript(items);
    const details = el.querySelector("details.reasoning-block") as HTMLDetailsElement;
    expect(details.open).toBe(false);
  });

  test("reasoning is expanded when thinking state is expanded", () => {
    setThinkingExpanded(true);
    const items: Item[] = [makeReasoning("Thinking...")];
    const el = renderTranscript(items);
    const details = el.querySelector("details.reasoning-block") as HTMLDetailsElement;
    expect(details.open).toBe(true);
    setThinkingExpanded(false);
  });

  test("renders function call as details.function-block", () => {
    const items: Item[] = [makeFunctionCall("get_weather", '{"city": "NYC"}')];
    const el = renderTranscript(items);
    const details = el.querySelector("details.function-block");
    expect(details).not.toBeNull();
    const summary = details!.querySelector("summary");
    expect(summary!.textContent).toContain("get_weather");
  });

  test("function call formats JSON arguments", () => {
    const items: Item[] = [makeFunctionCall("test_fn", '{"key":"value"}')];
    const el = renderTranscript(items);
    const pre = el.querySelector(".tool-pre");
    expect(pre).not.toBeNull();
    expect(pre!.textContent).toContain('"key"');
  });

  test("function call handles invalid JSON gracefully", () => {
    const items: Item[] = [makeFunctionCall("test_fn", "not-json")];
    const el = renderTranscript(items);
    const pre = el.querySelector(".tool-pre");
    expect(pre!.textContent).toBe("not-json");
  });

  test("renders function output as details.function-block with Tool Output", () => {
    const items: Item[] = [makeFunctionOutput("Result: 42")];
    const el = renderTranscript(items);
    const blocks = el.querySelectorAll("details.function-block");
    expect(blocks.length).toBe(1);
    const summary = blocks[0]!.querySelector("summary");
    expect(summary!.textContent).toBe("Tool Output");
  });

  test("function output shows output text in pre", () => {
    const items: Item[] = [makeFunctionOutput("Result: 42")];
    const el = renderTranscript(items);
    const pre = el.querySelector(".tool-pre");
    expect(pre!.textContent).toBe("Result: 42");
  });

  test("renders unknown item type as fallback card", () => {
    const items: Item[] = [{ type: "unknown_type" } as any];
    const el = renderTranscript(items);
    const card = el.querySelector(".card");
    expect(card).not.toBeNull();
    expect(card!.textContent).toContain("Unknown item type");
  });

  test("shows Stopped indicator for cancelled generation", () => {
    const items: Item[] = [{
      ...makeAssistantMessage("Partial response"),
      finish_reason: "cancelled",
    } as any];
    const el = renderTranscript(items);
    expect(el.textContent).toContain("Stopped");
  });

  test("normalizes system+empty-user into user message", () => {
    const items: Item[] = [
      { type: "message", role: "system", content: [{ type: "input_text", text: "System prompt" }] } as any,
      { type: "message", role: "user", content: [{ type: "input_text", text: "" }] } as any,
      makeAssistantMessage("Response"),
    ];
    const el = renderTranscript(items);
    // System text should appear as user message
    const userMsgs = el.querySelectorAll(".user-msg");
    expect(userMsgs.length).toBe(1);
    expect(userMsgs[0]!.textContent).toContain("System prompt");
  });

  test("promotes standalone system message before assistant to user", () => {
    const items: Item[] = [
      { type: "message", role: "system", content: [{ type: "input_text", text: "Hidden" }] } as any,
      makeAssistantMessage("Response"),
    ];
    const el = renderTranscript(items);
    const userMsgs = el.querySelectorAll(".user-msg");
    expect(userMsgs.length).toBe(1);
    expect(userMsgs[0]!.textContent).toContain("Hidden");
  });

  test("renders multiple items in order", () => {
    const items: Item[] = [
      makeUserMessage("Question"),
      makeReasoning("Thinking..."),
      makeFunctionCall("search", "{}"),
      makeFunctionOutput("Found 3 results"),
      makeAssistantMessage("Answer"),
    ];
    const el = renderTranscript(items);
    expect(el.children.length).toBe(5);
    expect(el.children[0]!.classList.contains("user-msg")).toBe(true);
    expect((el.children[1]! as HTMLElement).tagName.toLowerCase()).toBe("details");
  });
});

// ── renderTranscriptHeader ───────────────────────────────────────────────────

describe("renderTranscriptHeader", () => {
  test("creates header with transcript-header class", () => {
    const el = renderTranscriptHeader(makeConvo("c1"));
    expect(el.className).toBe("transcript-header");
  });

  test("sets data-chatId on header", () => {
    const el = renderTranscriptHeader(makeConvo("c1"));
    expect(el.dataset["chatId"]).toBe("c1");
  });

  test("renders date", () => {
    const el = renderTranscriptHeader(makeConvo("c1"));
    const date = el.querySelector(".transcript-date");
    expect(date).not.toBeNull();
  });

  test("renders tags from conversation.tags", () => {
    const chat = makeConvo("c1", { tags: [{ id: "1", name: "rust" }, { id: "2", name: "wasm" }] } as any);
    const el = renderTranscriptHeader(chat);
    const tags = el.querySelectorAll(".tag-pill");
    expect(tags.length).toBe(2);
  });

  test("tag pills have remove buttons", () => {
    const chat = makeConvo("c1", { tags: [{ id: "1", name: "test" }] } as any);
    const el = renderTranscriptHeader(chat);
    const removeBtn = el.querySelector(".tag-remove");
    expect(removeBtn).not.toBeNull();
    expect((removeBtn as HTMLElement).dataset["tag"]).toBe("test");
  });

  test("shows add-tag button when under 5 tags", () => {
    const chat = makeConvo("c1", { tags: [{ id: "1", name: "a" }, { id: "2", name: "b" }] } as any);
    const el = renderTranscriptHeader(chat);
    const addBtn = el.querySelector("[data-action='add-tag']");
    expect(addBtn).not.toBeNull();
  });

  test("hides add-tag button at 5 tags", () => {
    const chat = makeConvo("c1", { tags: [{ id: "1", name: "a" }, { id: "2", name: "b" }, { id: "3", name: "c" }, { id: "4", name: "d" }, { id: "5", name: "e" }] } as any);
    const el = renderTranscriptHeader(chat);
    const addBtn = el.querySelector("[data-action='add-tag']");
    expect(addBtn).toBeNull();
  });

  test("contains all action buttons", () => {
    const el = renderTranscriptHeader(makeConvo("c1"));
    const actions = el.querySelectorAll("[data-action]");
    const actionTypes = Array.from(actions).map((a) => (a as HTMLElement).dataset["action"]);
    expect(actionTypes).toContain("toggle-thinking");
    expect(actionTypes).toContain("copy");
    expect(actionTypes).toContain("export");
    expect(actionTypes).toContain("fork");
    expect(actionTypes).toContain("toggle-tuning");
  });

  test("shows archive button for non-archived chat", () => {
    const el = renderTranscriptHeader(makeConvo("c1", { marker: "" }));
    const archiveBtn = el.querySelector("[data-action='archive']");
    expect(archiveBtn).not.toBeNull();
  });

  test("shows unarchive button for archived chat", () => {
    const el = renderTranscriptHeader(makeConvo("c1", { marker: "archived" }));
    const unarchiveBtn = el.querySelector("[data-action='unarchive']");
    expect(unarchiveBtn).not.toBeNull();
  });

  test("thinking toggle starts with correct state", () => {
    initThinkingState(false, () => {});
    const el = renderTranscriptHeader(makeConvo("c1"));
    const btn = el.querySelector("#chat-thinking-toggle");
    expect(btn!.classList.contains("active")).toBe(false);
  });

  test("thinking toggle active when expanded", () => {
    setThinkingExpanded(true);
    const el = renderTranscriptHeader(makeConvo("c1"));
    const btn = el.querySelector("#chat-thinking-toggle");
    expect(btn!.classList.contains("active")).toBe(true);
    setThinkingExpanded(false);
  });
});
