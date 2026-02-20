import { describe, test, expect, beforeEach } from "bun:test";
import { addTagToChat, removeTagFromChat, updateSessionInList, handleAddTagPrompt, updateHeaderTags } from "../../../src/plugins/chat/tags.ts";
import { chatState } from "../../../src/plugins/chat/state.ts";
import { initChatDeps } from "../../../src/plugins/chat/deps.ts";
import { initChatDom, getChatDom } from "../../../src/plugins/chat/dom.ts";
import { createDomRoot, CHAT_DOM_IDS } from "../../helpers/dom.ts";
import { mockTimers, mockNotifications } from "../../helpers/mocks.ts";
import type { Conversation } from "../../../src/types.ts";

/**
 * Tests for chat tag management — validation, API sync, and state updates.
 */

let notifs: ReturnType<typeof mockNotifications>;
let apiCalls: { method: string; args: unknown[] }[];

function makeConvo(id: string, tags: string[] = []): Conversation {
  return {
    id,
    object: "conversation",
    title: `Convo ${id}`,
    created_at: 1700000000,
    updated_at: 1700000000,
    model: "gpt-4",
    items: [],
    metadata: {},
    tags: tags.map((t) => ({ name: t })),
  } as Conversation;
}

beforeEach(() => {
  notifs = mockNotifications();
  apiCalls = [];

  chatState.activeSessionId = "sess-1";
  chatState.activeChat = makeConvo("sess-1");
  chatState.sessions = [chatState.activeChat];

  const domRoot = createDomRoot(CHAT_DOM_IDS);
  // renderSidebar() inserts before the sentinel, so it must be a child of sidebar-list.
  const sidebarList = domRoot.querySelector("#sidebar-list")!;
  const sentinel = domRoot.querySelector("#loader-sentinel")!;
  sidebarList.appendChild(sentinel);
  initChatDom(domRoot);

  initChatDeps({
    api: {
      addConversationTags: async (_id: string, tags: string[]) => {
        apiCalls.push({ method: "addConversationTags", args: [_id, tags] });
        const existing = chatState.activeChat?.tags ?? [];
        const merged = [...existing, ...tags.map((t) => ({ name: t }))];
        return { ok: true, data: { tags: merged } };
      },
      removeConversationTags: async (_id: string, tags: string[]) => {
        apiCalls.push({ method: "removeConversationTags", args: [_id, tags] });
        const existing = chatState.activeChat?.tags ?? [];
        const remaining = existing.filter((t: any) => !tags.includes(t.name));
        return { ok: true, data: { tags: remaining } };
      },
    } as any,
    notifications: notifs.mock,
    timers: mockTimers(),
    services: {} as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    hooks: { run: async (name: string, payload: any) => payload } as any,
    clipboard: { writeText: async () => {} } as any,
    download: {} as any,
    observe: { onResize: () => ({ dispose() {} }) } as any,
    format: { dateTime: (ms: number) => new Date(ms).toISOString() } as any,
    upload: { upload: async () => ({}) } as any,
    menus: { registerItem: () => ({ dispose() {} }), renderSlot: () => ({ dispose() {} }) } as any,
  });
});

// ── Validation ───────────────────────────────────────────────────────────────

describe("addTagToChat — validation", () => {
  test("rejects empty string", async () => {
    await addTagToChat("");
    expect(apiCalls.length).toBe(0);
    expect(notifs.messages[0]?.msg).toContain("1-50 characters");
  });

  test("rejects whitespace-only", async () => {
    await addTagToChat("   ");
    expect(apiCalls.length).toBe(0);
  });

  test("rejects tag over 50 characters", async () => {
    await addTagToChat("a".repeat(51));
    expect(apiCalls.length).toBe(0);
    expect(notifs.messages[0]?.msg).toContain("1-50 characters");
  });

  test("rejects tags with special characters", async () => {
    await addTagToChat("hello world");
    expect(apiCalls.length).toBe(0);
    expect(notifs.messages[0]?.msg).toContain("letters, numbers, hyphens");
  });

  test("rejects tags with dots", async () => {
    await addTagToChat("v1.0");
    expect(apiCalls.length).toBe(0);
  });

  test("accepts valid tag with hyphens and underscores", async () => {
    await addTagToChat("my-tag_2");
    expect(apiCalls.length).toBe(1);
  });

  test("normalizes to lowercase", async () => {
    await addTagToChat("MyTag");
    expect(apiCalls[0]?.args).toEqual(["sess-1", ["mytag"]]);
  });

  test("trims whitespace", async () => {
    await addTagToChat("  hello  ");
    expect(apiCalls[0]?.args).toEqual(["sess-1", ["hello"]]);
  });

  test("rejects duplicate tag", async () => {
    chatState.activeChat = makeConvo("sess-1", ["existing"]);
    await addTagToChat("existing");
    expect(apiCalls.length).toBe(0);
    expect(notifs.messages[0]?.msg).toContain("already exists");
  });

  test("rejects when at 20 tag limit", async () => {
    const tags = Array.from({ length: 20 }, (_, i) => `tag${i}`);
    chatState.activeChat = makeConvo("sess-1", tags);
    await addTagToChat("tag20");
    expect(apiCalls.length).toBe(0);
    expect(notifs.messages[0]?.msg).toContain("Maximum 20");
  });

  test("no-op when no active session", async () => {
    chatState.activeSessionId = null;
    chatState.activeChat = null;
    await addTagToChat("test");
    expect(apiCalls.length).toBe(0);
  });
});

// ── Successful add ───────────────────────────────────────────────────────────

describe("addTagToChat — success", () => {
  test("calls API and updates local state", async () => {
    await addTagToChat("newtag");
    expect(apiCalls.length).toBe(1);
    expect(notifs.messages.some((m) => m.msg === "Tag added")).toBe(true);
    expect(chatState.activeChat!.tags).toEqual(
      expect.arrayContaining([expect.objectContaining({ name: "newtag" })]),
    );
  });
});

// ── Remove ───────────────────────────────────────────────────────────────────

describe("removeTagFromChat", () => {
  test("calls API and updates local state", async () => {
    chatState.activeChat = makeConvo("sess-1", ["rust", "wasm"]);
    await removeTagFromChat("rust");
    expect(apiCalls.length).toBe(1);
    expect(apiCalls[0]?.args).toEqual(["sess-1", ["rust"]]);
  });

  test("no-op when no active session", async () => {
    chatState.activeSessionId = null;
    chatState.activeChat = null;
    await removeTagFromChat("test");
    expect(apiCalls.length).toBe(0);
  });
});

// ── updateSessionInList ──────────────────────────────────────────────────────

describe("updateSessionInList", () => {
  test("updates matching session", () => {
    const updated = makeConvo("sess-1", ["new"]);
    updated.title = "Updated Title";
    chatState.sessions = [makeConvo("sess-1")];
    updateSessionInList(updated);
    expect(chatState.sessions[0]!.title).toBe("Updated Title");
  });

  test("no-op when session not found", () => {
    chatState.sessions = [makeConvo("other")];
    updateSessionInList(makeConvo("missing"));
    expect(chatState.sessions.length).toBe(1);
    expect(chatState.sessions[0]!.id).toBe("other");
  });
});

// ── updateHeaderTags ────────────────────────────────────────────────────────

/** Add a .transcript-tags container inside the transcript element. */
function addTagsContainer(): HTMLElement {
  const tc = getChatDom().transcriptContainer;
  let container = tc.querySelector<HTMLElement>(".transcript-tags");
  if (!container) {
    container = document.createElement("div");
    container.className = "transcript-tags";
    tc.appendChild(container);
  }
  return container;
}

describe("updateHeaderTags", () => {
  test("renders tag pills for each tag", () => {
    addTagsContainer();
    const chat = makeConvo("sess-1", ["rust", "wasm"]);
    updateHeaderTags(chat);
    const pills = getChatDom().transcriptContainer.querySelectorAll(".tag-pill");
    expect(pills.length).toBe(2);
  });

  test("tag pill has data-tag attribute", () => {
    addTagsContainer();
    const chat = makeConvo("sess-1", ["rust"]);
    updateHeaderTags(chat);
    const pill = getChatDom().transcriptContainer.querySelector<HTMLElement>(".tag-pill")!;
    expect(pill.dataset["tag"]).toBe("rust");
  });

  test("tag pill has remove button", () => {
    addTagsContainer();
    const chat = makeConvo("sess-1", ["rust"]);
    updateHeaderTags(chat);
    const removeBtn = getChatDom().transcriptContainer.querySelector(".tag-remove")!;
    expect(removeBtn).not.toBeNull();
    expect((removeBtn as HTMLElement).dataset["tag"]).toBe("rust");
  });

  test("shows add button when fewer than 5 tags", () => {
    addTagsContainer();
    const chat = makeConvo("sess-1", ["a", "b"]);
    updateHeaderTags(chat);
    const addBtn = getChatDom().transcriptContainer.querySelector(".add-tag-btn");
    expect(addBtn).not.toBeNull();
  });

  test("hides add button at 5 tags", () => {
    addTagsContainer();
    const chat = makeConvo("sess-1", ["a", "b", "c", "d", "e"]);
    updateHeaderTags(chat);
    const addBtn = getChatDom().transcriptContainer.querySelector(".add-tag-btn");
    expect(addBtn).toBeNull();
  });

  test("clears previous tags on re-render", () => {
    addTagsContainer();
    updateHeaderTags(makeConvo("sess-1", ["old"]));
    updateHeaderTags(makeConvo("sess-1", ["new"]));
    const pills = getChatDom().transcriptContainer.querySelectorAll(".tag-pill");
    expect(pills.length).toBe(1);
    expect((pills[0]! as HTMLElement).dataset["tag"]).toBe("new");
  });

  test("no-op when .transcript-tags not in DOM", () => {
    // Don't add a tags container
    const chat = makeConvo("sess-1", ["rust"]);
    updateHeaderTags(chat); // should not throw
  });
});

// ── handleAddTagPrompt ──────────────────────────────────────────────────────

describe("handleAddTagPrompt", () => {
  test("no-op when no active session", () => {
    chatState.activeSessionId = null;
    chatState.activeChat = null;
    handleAddTagPrompt();
    expect(notifs.messages.some((m) => m.msg === "No active conversation")).toBe(true);
  });

  test("creates inline input wrapper", () => {
    const container = addTagsContainer();
    handleAddTagPrompt();
    expect(container.querySelector(".tag-input-wrapper")).not.toBeNull();
    expect(container.querySelector<HTMLInputElement>(".tag-input")).not.toBeNull();
  });

  test("input has maxLength 50", () => {
    addTagsContainer();
    handleAddTagPrompt();
    const input = getChatDom().transcriptContainer.querySelector<HTMLInputElement>(".tag-input")!;
    expect(input.maxLength).toBe(50);
  });

  test("hides add button when input is shown", () => {
    const container = addTagsContainer();
    // Add an add button first
    const addBtn = document.createElement("button");
    addBtn.className = "add-tag-btn";
    container.appendChild(addBtn);

    handleAddTagPrompt();
    expect(addBtn.classList.contains("hidden")).toBe(true);
  });

  test("no-op if input already exists (prevents double input)", () => {
    addTagsContainer();
    handleAddTagPrompt();
    handleAddTagPrompt(); // second call
    const wrappers = getChatDom().transcriptContainer.querySelectorAll(".tag-input-wrapper");
    expect(wrappers.length).toBe(1);
  });

  test("cancel button removes wrapper", () => {
    const container = addTagsContainer();
    handleAddTagPrompt();
    const cancelBtn = container.querySelector<HTMLButtonElement>(".tag-input-cancel")!;
    cancelBtn.click();
    expect(container.querySelector(".tag-input-wrapper")).toBeNull();
  });

  test("Escape key removes wrapper", () => {
    const container = addTagsContainer();
    handleAddTagPrompt();
    const input = container.querySelector<HTMLInputElement>(".tag-input")!;
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    expect(container.querySelector(".tag-input-wrapper")).toBeNull();
  });

  test("cancel restores add button visibility", () => {
    const container = addTagsContainer();
    const addBtn = document.createElement("button");
    addBtn.className = "add-tag-btn";
    container.appendChild(addBtn);

    handleAddTagPrompt();
    expect(addBtn.classList.contains("hidden")).toBe(true);

    const cancelBtn = container.querySelector<HTMLButtonElement>(".tag-input-cancel")!;
    cancelBtn.click();
    expect(addBtn.classList.contains("hidden")).toBe(false);
  });

  test("Enter key with text calls addTagToChat and removes wrapper", async () => {
    const container = addTagsContainer();
    handleAddTagPrompt();
    const input = container.querySelector<HTMLInputElement>(".tag-input")!;
    input.value = "newtag";
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));

    // Wrapper should be removed after submit
    expect(container.querySelector(".tag-input-wrapper")).toBeNull();
    // API call should have been made
    expect(apiCalls.length).toBe(1);
    expect(apiCalls[0]!.args).toEqual(["sess-1", ["newtag"]]);
  });

  test("Save button with text calls addTagToChat", async () => {
    const container = addTagsContainer();
    handleAddTagPrompt();
    const input = container.querySelector<HTMLInputElement>(".tag-input")!;
    input.value = "saved-tag";
    const saveBtn = container.querySelector<HTMLButtonElement>(".tag-input-save")!;
    saveBtn.click();

    expect(container.querySelector(".tag-input-wrapper")).toBeNull();
    expect(apiCalls.length).toBe(1);
  });

  test("Enter with empty value just cleans up", () => {
    const container = addTagsContainer();
    handleAddTagPrompt();
    const input = container.querySelector<HTMLInputElement>(".tag-input")!;
    input.value = "";
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));

    expect(container.querySelector(".tag-input-wrapper")).toBeNull();
    expect(apiCalls.length).toBe(0);
  });
});
