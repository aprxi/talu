import { describe, test, expect, beforeEach } from "bun:test";
import {
  loadSessions,
  renderSidebar,
  refreshSidebar,
  setupInfiniteScroll,
} from "../../../src/plugins/chat/sidebar-list.ts";
import { chatState } from "../../../src/plugins/chat/state.ts";
import { initChatDeps } from "../../../src/plugins/chat/deps.ts";
import { initChatDom, getChatDom } from "../../../src/plugins/chat/dom.ts";
import { createDomRoot, CHAT_DOM_IDS } from "../../helpers/dom.ts";
import { mockNotifications } from "../../helpers/mocks.ts";
import type { Disposable } from "../../../src/kernel/types.ts";

/**
 * Tests for chat sidebar list — pagination (append semantics), rendering
 * (Pinned/Recent split, archived filtering, sentinel management),
 * refresh, and IntersectionObserver-driven infinite scroll.
 *
 * Strategy: mock API with controllable pagination results. The sentinel
 * element is placed inside the sidebar list for renderSidebar() compat.
 */

// -- Mock state --------------------------------------------------------------

let apiCalls: { method: string; args: unknown[] }[];
let notif: ReturnType<typeof mockNotifications>;
let listResult: any;
let observerCallback: ((entries: any[]) => void) | null;

beforeEach(() => {
  apiCalls = [];
  notif = mockNotifications();
  observerCallback = null;

  listResult = {
    ok: true,
    data: { data: [], cursor: null, has_more: false },
  };

  // Reset state.
  chatState.sessions = [];
  chatState.activeSessionId = null;
  chatState.activeChat = null;
  chatState.lastResponseId = null;
  chatState.isGenerating = false;
  chatState.streamAbort = null;
  chatState.pagination = { cursor: null, hasMore: true, isLoading: false };

  // DOM — sentinel must be inside sidebarList.
  const root = createDomRoot(CHAT_DOM_IDS);
  const list = root.querySelector("#sidebar-list")!;
  const sentinel = root.querySelector("#loader-sentinel")!;
  list.appendChild(sentinel);
  initChatDom(root);

  // Deps.
  initChatDeps({
    api: {
      listConversations: async (cursor?: string, limit?: number) => {
        apiCalls.push({ method: "listConversations", args: [cursor, limit] });
        return listResult;
      },
    } as any,
    notifications: notif.mock as any,
    services: { get: () => undefined } as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    layout: { setTitle: () => {} } as any,
    clipboard: { writeText: async () => {} } as any,
    download: { save: () => {} } as any,
    upload: {} as any,
    hooks: {
      on: () => ({ dispose() {} }),
      run: async <T>(_name: string, value: T) => value,
    } as any,
    timers: {
      setTimeout(fn: () => void) { fn(); return { dispose() {} }; },
      setInterval() { return { dispose() {} }; },
      requestAnimationFrame(fn: () => void) { fn(); return { dispose() {} }; },
    } as any,
    observe: {
      intersection: (_target: any, callback: any) => {
        observerCallback = callback;
        return { dispose() {} };
      },
      mutation: () => ({ dispose() {} }),
      resize: () => ({ dispose() {} }),
    } as any,
    format: {
      date: () => "", dateTime: () => "", relativeTime: () => "1h ago",
      duration: () => "", number: () => "",
    } as any,
    menus: {
      registerItem: () => ({ dispose() {} }),
      renderSlot: () => ({ dispose() {} }),
    } as any,
  });
});

// -- Helpers -----------------------------------------------------------------

function makeConvo(id: string, title = "Chat", marker = ""): any {
  return {
    id, title, marker,
    object: "conversation", created_at: 1700000000, updated_at: 1700000000,
    model: "gpt-4", items: [], metadata: {},
  };
}

function sidebarItems(): HTMLElement[] {
  return [...getChatDom().sidebarList.querySelectorAll<HTMLElement>(".sidebar-item")];
}

function sectionLabels(): string[] {
  return [...getChatDom().sidebarList.querySelectorAll<HTMLElement>(".sidebar-section-label")]
    .map((el) => el.textContent ?? "");
}

// ── loadSessions ──────────────────────────────────────────────────────────────

describe("loadSessions", () => {
  test("calls API with cursor and limit", async () => {
    chatState.pagination.cursor = "abc";
    await loadSessions();

    expect(apiCalls.length).toBe(1);
    expect(apiCalls[0]!.args[0]).toBe("abc");
    expect(apiCalls[0]!.args[1]).toBe(100);
  });

  test("appends results to sessions (not replaces)", async () => {
    chatState.sessions = [makeConvo("existing")];
    listResult = {
      ok: true,
      data: { data: [makeConvo("new-1"), makeConvo("new-2")], cursor: "c2", has_more: true },
    };

    await loadSessions();

    expect(chatState.sessions.length).toBe(3);
    expect(chatState.sessions[0]!.id).toBe("existing");
    expect(chatState.sessions[1]!.id).toBe("new-1");
    expect(chatState.sessions[2]!.id).toBe("new-2");
  });

  test("updates pagination cursor and hasMore", async () => {
    listResult = {
      ok: true,
      data: { data: [makeConvo("c1")], cursor: "next-page", has_more: true },
    };

    await loadSessions();

    expect(chatState.pagination.cursor).toBe("next-page");
    expect(chatState.pagination.hasMore).toBe(true);
  });

  test("sets hasMore=false when no more pages", async () => {
    listResult = {
      ok: true,
      data: { data: [makeConvo("c1")], cursor: null, has_more: false },
    };

    await loadSessions();
    expect(chatState.pagination.hasMore).toBe(false);
  });

  test("no-op when already loading (concurrent guard)", async () => {
    chatState.pagination.isLoading = true;
    await loadSessions();
    expect(apiCalls.length).toBe(0);
  });

  test("no-op when hasMore is false", async () => {
    chatState.pagination.hasMore = false;
    await loadSessions();
    expect(apiCalls.length).toBe(0);
  });

  test("sets isLoading during fetch", async () => {
    let capturedIsLoading = false;
    initChatDeps({
      api: {
        listConversations: async () => {
          capturedIsLoading = chatState.pagination.isLoading;
          return listResult;
        },
      } as any,
      notifications: notif.mock as any,
      services: { get: () => undefined } as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      layout: { setTitle: () => {} } as any,
      clipboard: { writeText: async () => {} } as any,
      download: { save: () => {} } as any,
      upload: {} as any,
      hooks: {
        on: () => ({ dispose() {} }),
        run: async <T>(_name: string, value: T) => value,
      } as any,
      timers: { setTimeout(fn: () => void) { fn(); return { dispose() {} }; }, setInterval() { return { dispose() {} }; }, requestAnimationFrame(fn: () => void) { fn(); return { dispose() {} }; } } as any,
      observe: { intersection: () => ({ dispose() {} }), mutation: () => ({ dispose() {} }), resize: () => ({ dispose() {} }) } as any,
      format: { date: () => "", dateTime: () => "", relativeTime: () => "", duration: () => "", number: () => "" } as any,
      menus: {
        registerItem: () => ({ dispose() {} }),
        renderSlot: () => ({ dispose() {} }),
      } as any,
    });

    await loadSessions();
    expect(capturedIsLoading).toBe(true);
    expect(chatState.pagination.isLoading).toBe(false); // cleared after
  });

  test("shows error on API failure", async () => {
    listResult = { ok: false, error: "Server error" };
    await loadSessions();
    expect(notif.messages.some((m) => m.type === "error" && m.msg.includes("Server error"))).toBe(true);
  });

  test("multiple pages accumulate via successive calls", async () => {
    // Page 1.
    listResult = {
      ok: true,
      data: { data: [makeConvo("p1-a"), makeConvo("p1-b")], cursor: "c2", has_more: true },
    };
    await loadSessions();

    // Page 2.
    listResult = {
      ok: true,
      data: { data: [makeConvo("p2-a")], cursor: null, has_more: false },
    };
    await loadSessions();

    expect(chatState.sessions.length).toBe(3);
    expect(chatState.sessions.map((s: any) => s.id)).toEqual(["p1-a", "p1-b", "p2-a"]);
    expect(chatState.pagination.hasMore).toBe(false);
  });
});

// ── renderSidebar ─────────────────────────────────────────────────────────────

describe("renderSidebar", () => {
  test("shows empty state when no sessions", () => {
    chatState.sessions = [];
    renderSidebar();
    const empty = getChatDom().sidebarList.querySelector(".empty-state");
    expect(empty).not.toBeNull();
    expect(sidebarItems().length).toBe(0);
  });

  test("renders non-archived sessions as sidebar items", () => {
    chatState.sessions = [makeConvo("c1", "Chat 1"), makeConvo("c2", "Chat 2")];
    renderSidebar();
    expect(sidebarItems().length).toBe(2);
  });

  test("filters out archived sessions", () => {
    chatState.sessions = [
      makeConvo("c1", "Active"),
      makeConvo("c2", "Archived", "archived"),
      makeConvo("c3", "Also Active"),
    ];
    renderSidebar();
    expect(sidebarItems().length).toBe(2);
    expect(sidebarItems().map((el) => el.dataset["id"])).toEqual(["c1", "c3"]);
  });

  test("splits into Pinned and Recent sections", () => {
    chatState.sessions = [
      makeConvo("c1", "Pinned Chat", "pinned"),
      makeConvo("c2", "Regular Chat"),
    ];
    renderSidebar();

    const labels = sectionLabels();
    expect(labels).toContain("Pinned");
    expect(labels).toContain("Recent");
    expect(sidebarItems().length).toBe(2);
  });

  test("shows only Pinned label when all are pinned (no Recent label)", () => {
    chatState.sessions = [
      makeConvo("c1", "Pin 1", "pinned"),
      makeConvo("c2", "Pin 2", "pinned"),
    ];
    renderSidebar();

    const labels = sectionLabels();
    expect(labels).toContain("Pinned");
    expect(labels).not.toContain("Recent");
  });

  test("no section labels when only unpinned sessions", () => {
    chatState.sessions = [makeConvo("c1", "Chat 1"), makeConvo("c2", "Chat 2")];
    renderSidebar();

    const labels = sectionLabels();
    expect(labels).not.toContain("Pinned");
    expect(labels).not.toContain("Recent");
  });

  test("hides sentinel when no more pages", () => {
    chatState.pagination.hasMore = false;
    chatState.sessions = [makeConvo("c1")];
    renderSidebar();
    expect(getChatDom().sidebarSentinel.style.display).toBe("none");
  });

  test("shows sentinel when more pages available", () => {
    chatState.pagination.hasMore = true;
    chatState.sessions = [makeConvo("c1")];
    renderSidebar();
    expect(getChatDom().sidebarSentinel.style.display).toBe("flex");
  });

  test("marks active session item", () => {
    chatState.sessions = [makeConvo("c1"), makeConvo("c2")];
    chatState.activeSessionId = "c2";
    renderSidebar();

    const items = sidebarItems();
    const activeItem = items.find((el) => el.dataset["id"] === "c2");
    expect(activeItem?.classList.contains("active")).toBe(true);
  });

  test("clears previous items on re-render", () => {
    chatState.sessions = [makeConvo("c1"), makeConvo("c2")];
    renderSidebar();
    expect(sidebarItems().length).toBe(2);

    chatState.sessions = [makeConvo("c3")];
    renderSidebar();
    expect(sidebarItems().length).toBe(1);
    expect(sidebarItems()[0]!.dataset["id"]).toBe("c3");
  });
});

// ── refreshSidebar ────────────────────────────────────────────────────────────

describe("refreshSidebar", () => {
  test("resets pagination state before reload", async () => {
    chatState.sessions = [makeConvo("old")];
    chatState.pagination.cursor = "old-cursor";
    chatState.pagination.hasMore = false;

    listResult = {
      ok: true,
      data: { data: [makeConvo("fresh")], cursor: null, has_more: false },
    };

    await refreshSidebar();

    // Old sessions replaced by fresh ones.
    expect(chatState.sessions.length).toBe(1);
    expect(chatState.sessions[0]!.id).toBe("fresh");
  });

  test("resets cursor and hasMore before load", async () => {
    chatState.pagination.offset = 50;
    chatState.pagination.hasMore = false;

    await refreshSidebar();

    // refreshSidebar fetches from offset 0 to pick up new conversations.
    expect(apiCalls[0]!.args[0]).toEqual({ offset: 0, limit: 100 });
    expect(chatState.pagination.hasMore).toBe(false);
  });
});

// ── setupInfiniteScroll ──────────────────────────────────────────────────────

describe("setupInfiniteScroll", () => {
  test("registers intersection observer on sentinel", () => {
    setupInfiniteScroll();
    expect(observerCallback).not.toBeNull();
  });

  test("triggers loadSessions when sentinel is intersecting", async () => {
    listResult = {
      ok: true,
      data: { data: [makeConvo("scroll-1")], cursor: null, has_more: false },
    };
    setupInfiniteScroll();

    observerCallback!([{ isIntersecting: true }]);
    // loadSessions is async — give it a tick.
    await new Promise((r) => setTimeout(r, 10));

    expect(apiCalls.length).toBe(1);
    expect(apiCalls[0]!.method).toBe("listConversations");
  });

  test("ignores non-intersecting entries", async () => {
    setupInfiniteScroll();
    observerCallback!([{ isIntersecting: false }]);
    await new Promise((r) => setTimeout(r, 10));
    expect(apiCalls.length).toBe(0);
  });
});
