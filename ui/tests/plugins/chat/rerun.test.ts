import { describe, test, expect, beforeEach } from "bun:test";
import { findLastUserMessage, getUserMessageInfo } from "../../../src/plugins/chat/rerun.ts";
import { chatState } from "../../../src/plugins/chat/state.ts";
import type { Item, MessageItem, ContentPart, InputTextPart } from "../../../src/types.ts";

/**
 * Tests for the pure/near-pure message extraction functions in rerun.ts.
 *
 * findLastUserMessage: scans items backwards for last user message text.
 * getUserMessageInfo: maps a display-order index to message text + fork point.
 */

/** Create a user message item with input_text content. */
function userMsg(text: string, ...extraParts: ContentPart[]): MessageItem {
  const parts: ContentPart[] = [{ type: "input_text", text } as InputTextPart, ...extraParts];
  return { type: "message", role: "user", content: parts };
}

/** Create an assistant message item with output_text content. */
function assistantMsg(text: string): MessageItem {
  return { type: "message", role: "assistant", content: [{ type: "output_text", text }] } as MessageItem;
}

/** Create a system message item. */
function systemMsg(text: string): MessageItem {
  return { type: "message", role: "system", content: [{ type: "input_text", text } as InputTextPart] };
}

/** Create a developer message item (alternative system-message workaround). */
function developerMsg(text: string): MessageItem {
  return { type: "message", role: "developer", content: [{ type: "input_text", text } as InputTextPart] } as MessageItem;
}

/** Create an empty user message (for the system-message workaround). */
function emptyUserMsg(): MessageItem {
  return { type: "message", role: "user", content: [{ type: "input_text", text: "" } as InputTextPart] };
}

beforeEach(() => {
  chatState.activeChat = null;
  chatState.activeSessionId = null;
});

// ── findLastUserMessage ─────────────────────────────────────────────────────

describe("findLastUserMessage", () => {
  test("returns null for empty items", () => {
    expect(findLastUserMessage([])).toBeNull();
  });

  test("finds last user message text", () => {
    const items: Item[] = [
      userMsg("first"),
      assistantMsg("response"),
      userMsg("second"),
      assistantMsg("response 2"),
    ];
    const result = findLastUserMessage(items);
    expect(result).not.toBeNull();
    expect(result!.text).toBe("second");
    expect(result!.itemIndex).toBe(2);
  });

  test("returns null when no user messages", () => {
    const items: Item[] = [assistantMsg("hello"), assistantMsg("world")];
    expect(findLastUserMessage(items)).toBeNull();
  });

  test("skips assistant and non-message items", () => {
    const items: Item[] = [
      userMsg("only user"),
      assistantMsg("response"),
      { type: "function_call", id: "fc1", call_id: "c1", name: "fn", arguments: "{}" } as Item,
    ];
    const result = findLastUserMessage(items);
    expect(result!.text).toBe("only user");
    expect(result!.itemIndex).toBe(0);
  });

  test("joins multiple input_text parts with newlines", () => {
    const item: MessageItem = {
      type: "message",
      role: "user",
      content: [
        { type: "input_text", text: "line 1" } as InputTextPart,
        { type: "input_text", text: "line 2" } as InputTextPart,
      ],
    };
    const result = findLastUserMessage([item]);
    expect(result!.text).toBe("line 1\nline 2");
  });

  test("skips user messages with only whitespace", () => {
    const items: Item[] = [
      userMsg("real content"),
      assistantMsg("reply"),
      userMsg("   "), // whitespace-only
    ];
    const result = findLastUserMessage(items);
    expect(result!.text).toBe("real content");
    expect(result!.itemIndex).toBe(0);
  });

  test("system message workaround — empty user msg preceded by system msg", () => {
    const items: Item[] = [
      systemMsg("actual user input"),
      emptyUserMsg(),
      assistantMsg("response"),
    ];
    const result = findLastUserMessage(items);
    expect(result).not.toBeNull();
    expect(result!.text).toBe("actual user input");
    // itemIndex points to the user message position, not the system message.
    expect(result!.itemIndex).toBe(1);
  });

  test("prefers normal user message over system workaround", () => {
    const items: Item[] = [
      systemMsg("sys input"),
      emptyUserMsg(),
      assistantMsg("reply"),
      userMsg("normal input"),
      assistantMsg("reply 2"),
    ];
    const result = findLastUserMessage(items);
    expect(result!.text).toBe("normal input");
  });
});

// ── getUserMessageInfo ──────────────────────────────────────────────────────

describe("getUserMessageInfo", () => {
  test("returns null when no active chat", () => {
    chatState.activeChat = null;
    expect(getUserMessageInfo(0)).toBeNull();
  });

  test("returns null for out-of-range index", () => {
    chatState.activeChat = {
      id: "c1",
      items: [userMsg("hello"), assistantMsg("reply")],
    } as any;
    expect(getUserMessageInfo(5)).toBeNull();
    expect(getUserMessageInfo(-1)).toBeNull();
  });

  test("returns first user message at index 0", () => {
    chatState.activeChat = {
      id: "c1",
      items: [userMsg("first"), assistantMsg("reply"), userMsg("second")],
    } as any;
    const result = getUserMessageInfo(0);
    expect(result).not.toBeNull();
    expect(result!.text).toBe("first");
    expect(result!.forkBeforeIndex).toBe(-1);
  });

  test("returns second user message at index 1", () => {
    chatState.activeChat = {
      id: "c1",
      items: [
        userMsg("first"),
        assistantMsg("reply 1"),
        userMsg("second"),
        assistantMsg("reply 2"),
      ],
    } as any;
    const result = getUserMessageInfo(1);
    expect(result!.text).toBe("second");
    expect(result!.forkBeforeIndex).toBe(1);
  });

  test("returns null for empty text at index", () => {
    chatState.activeChat = {
      id: "c1",
      items: [userMsg("   ")],
    } as any;
    expect(getUserMessageInfo(0)).toBeNull();
  });

  test("handles system message workaround in getUserMessageInfo", () => {
    chatState.activeChat = {
      id: "c1",
      items: [
        systemMsg("actual input"),
        emptyUserMsg(), // paired with system msg above
        assistantMsg("reply"),
      ],
    } as any;
    const result = getUserMessageInfo(0);
    expect(result).not.toBeNull();
    expect(result!.text).toBe("actual input");
    // forkBeforeIndex is itemIndex-1; system workaround has itemIndex=0, so -1.
    expect(result!.forkBeforeIndex).toBe(-1);
  });

  test("handles developer message workaround (same as system)", () => {
    chatState.activeChat = {
      id: "c1",
      items: [
        developerMsg("dev input"),
        emptyUserMsg(),
        assistantMsg("reply"),
      ],
    } as any;
    const result = getUserMessageInfo(0);
    expect(result).not.toBeNull();
    expect(result!.text).toBe("dev input");
    expect(result!.forkBeforeIndex).toBe(-1);
  });

  test("mixed normal and system-workaround messages index correctly", () => {
    chatState.activeChat = {
      id: "c1",
      items: [
        systemMsg("sys input"),  // 0: system workaround → userMsgPositions[0]
        emptyUserMsg(),          // 1: empty user (skipped by i++)
        assistantMsg("reply"),   // 2
        userMsg("normal input"), // 3: normal user → userMsgPositions[1]
        assistantMsg("reply 2"), // 4
      ],
    } as any;

    const first = getUserMessageInfo(0);
    expect(first!.text).toBe("sys input");

    const second = getUserMessageInfo(1);
    expect(second!.text).toBe("normal input");
    expect(second!.forkBeforeIndex).toBe(2);
  });
});
