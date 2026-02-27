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

/**
 * Tests for chat sidebar list — pagination (append semantics), rendering
 * (project group headers, archived filtering, sentinel management),
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
    data: { data: [], has_more: false },
  };

  // Reset state.
  chatState.sessions = [];
  chatState.activeSessionId = null;
  chatState.activeChat = null;
  chatState.lastResponseId = null;
  chatState.isGenerating = false;
  chatState.streamAbort = null;
  chatState.backgroundStreamSessions = new Set();
  chatState.sidebarSearchQuery = "";
  chatState.collapsedGroups = new Set();
  chatState.expandedGroups = new Set();
  chatState.sidebarSort = "recent";
  chatState.pagination = { offset: 0, hasMore: true, isLoading: false };

  // DOM — sentinel must be inside sidebarList.
  const root = createDomRoot(CHAT_DOM_IDS);
  const list = root.querySelector("#sidebar-list")!;
  const sentinel = root.querySelector("#loader-sentinel")!;
  list.appendChild(sentinel);
  initChatDom(root);

  // Deps.
  initChatDeps({
    api: {
      listConversations: async (params?: any) => {
        apiCalls.push({ method: "listConversations", args: [params] });
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

function makeConvo(id: string, title = "Chat", marker = "", projectId?: string): any {
  return {
    id, title, marker,
    object: "conversation", created_at: 1700000000, updated_at: 1700000000,
    model: "gpt-4", items: [], metadata: {},
    ...(projectId ? { project_id: projectId } : {}),
  };
}

function sidebarItems(): HTMLElement[] {
  return [...getChatDom().sidebarList.querySelectorAll<HTMLElement>(".sidebar-item")];
}

function groupLabels(): string[] {
  return [...getChatDom().sidebarList.querySelectorAll<HTMLElement>(".sidebar-group-label .sidebar-group-name")]
    .map((el) => el.textContent ?? "");
}

// ── loadSessions ──────────────────────────────────────────────────────────────

describe("loadSessions", () => {
  test("calls API with offset and limit", async () => {
    chatState.pagination.offset = 10;
    await loadSessions();

    expect(apiCalls.length).toBe(1);
    expect(apiCalls[0]!.args[0]).toEqual({ offset: 10, limit: 100 });
  });

  test("appends results to sessions (not replaces)", async () => {
    chatState.sessions = [makeConvo("existing")];
    listResult = {
      ok: true,
      data: { data: [makeConvo("new-1"), makeConvo("new-2")], has_more: true },
    };

    await loadSessions();

    expect(chatState.sessions.length).toBe(3);
    expect(chatState.sessions[0]!.id).toBe("existing");
    expect(chatState.sessions[1]!.id).toBe("new-1");
    expect(chatState.sessions[2]!.id).toBe("new-2");
  });

  test("updates pagination offset and hasMore", async () => {
    listResult = {
      ok: true,
      data: { data: [makeConvo("c1")], has_more: true },
    };

    await loadSessions();

    expect(chatState.pagination.offset).toBe(1);
    expect(chatState.pagination.hasMore).toBe(true);
  });

  test("sets hasMore=false when no more pages", async () => {
    listResult = {
      ok: true,
      data: { data: [makeConvo("c1")], has_more: false },
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
      data: { data: [makeConvo("p1-a"), makeConvo("p1-b")], has_more: true },
    };
    await loadSessions();

    // Page 2.
    listResult = {
      ok: true,
      data: { data: [makeConvo("p2-a")], has_more: false },
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

  test("renders pinned items before unpinned within a group", () => {
    chatState.sessions = [
      makeConvo("c1", "Regular Chat"),
      makeConvo("c2", "Pinned Chat", "pinned"),
    ];
    renderSidebar();

    const items = sidebarItems();
    expect(items.length).toBe(2);
    // Pinned item should appear first.
    expect(items[0]!.dataset["id"]).toBe("c2");
    expect(items[1]!.dataset["id"]).toBe("c1");
  });

  test("renders project group headers when multiple projects", () => {
    chatState.sessions = [
      makeConvo("c1", "Chat A", "", "ProjectX"),
      makeConvo("c2", "Chat B", "", "ProjectY"),
    ];
    renderSidebar();

    const labels = groupLabels();
    expect(labels).toContain("ProjectX");
    expect(labels).toContain("ProjectY");
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

  test("marks active generating session with generating class", () => {
    chatState.sessions = [makeConvo("c1"), makeConvo("c2")];
    chatState.activeSessionId = "c1";
    chatState.isGenerating = true;
    renderSidebar();

    const items = sidebarItems();
    const c1 = items.find((el) => el.dataset["id"] === "c1");
    const c2 = items.find((el) => el.dataset["id"] === "c2");
    expect(c1?.classList.contains("generating")).toBe(true);
    expect(c2?.classList.contains("generating")).toBe(false);
  });

  test("marks background streaming session with generating class", () => {
    chatState.sessions = [makeConvo("c1"), makeConvo("c2")];
    chatState.backgroundStreamSessions.add("c2");
    renderSidebar();

    const items = sidebarItems();
    const c1 = items.find((el) => el.dataset["id"] === "c1");
    const c2 = items.find((el) => el.dataset["id"] === "c2");
    expect(c1?.classList.contains("generating")).toBe(false);
    expect(c2?.classList.contains("generating")).toBe(true);
  });

  test("generating item has pulsing dot element", () => {
    chatState.sessions = [makeConvo("c1")];
    chatState.backgroundStreamSessions.add("c1");
    renderSidebar();

    const item = sidebarItems()[0]!;
    const dot = item.querySelector(".sidebar-generating-dot");
    expect(dot).not.toBeNull();
  });

  test("non-generating item has no pulsing dot", () => {
    chatState.sessions = [makeConvo("c1")];
    renderSidebar();

    const item = sidebarItems()[0]!;
    const dot = item.querySelector(".sidebar-generating-dot");
    expect(dot).toBeNull();
  });

  test("filters sessions by search query (case insensitive)", () => {
    chatState.sessions = [
      makeConvo("c1", "Alpha chat"),
      makeConvo("c2", "Beta chat"),
      makeConvo("c3", "Alpha two"),
    ];
    chatState.sidebarSearchQuery = "alpha";
    renderSidebar();

    expect(sidebarItems().length).toBe(2);
    expect(sidebarItems().map((el) => el.dataset["id"])).toEqual(["c1", "c3"]);
  });

  test("shows empty state when search has no matches", () => {
    chatState.sessions = [makeConvo("c1", "Hello")];
    chatState.sidebarSearchQuery = "zzz";
    renderSidebar();

    expect(sidebarItems().length).toBe(0);
    const empty = getChatDom().sidebarList.querySelector(".empty-state");
    expect(empty).not.toBeNull();
  });

  test("shows all sessions when search query is empty", () => {
    chatState.sessions = [makeConvo("c1", "A"), makeConvo("c2", "B")];
    chatState.sidebarSearchQuery = "";
    renderSidebar();
    expect(sidebarItems().length).toBe(2);
  });
});

// ── refreshSidebar ────────────────────────────────────────────────────────────

describe("refreshSidebar", () => {
  test("replaces sessions with fresh data", async () => {
    chatState.sessions = [makeConvo("old")];
    chatState.pagination.offset = 50;
    chatState.pagination.hasMore = false;

    listResult = {
      ok: true,
      data: { data: [makeConvo("fresh")], has_more: false },
    };

    await refreshSidebar();

    // Old sessions replaced by fresh ones.
    expect(chatState.sessions.length).toBe(1);
    expect(chatState.sessions[0]!.id).toBe("fresh");
  });

  test("fetches from offset 0", async () => {
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
      data: { data: [makeConvo("scroll-1")], has_more: false },
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
