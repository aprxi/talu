import { describe, test, expect, beforeEach } from "bun:test";
import { loadBrowserConversations, loadAvailableTags } from "../../../src/plugins/browser/data.ts";
import { bState, search } from "../../../src/plugins/browser/state.ts";
import { initBrowserDom } from "../../../src/plugins/browser/dom.ts";
import { initBrowserDeps } from "../../../src/plugins/browser/deps.ts";
import { createDomRoot, BROWSER_DOM_IDS, BROWSER_DOM_EXTRAS, BROWSER_DOM_TAGS } from "../../helpers/dom.ts";
import { mockTimers } from "../../helpers/mocks.ts";
import type { Conversation } from "../../../src/types.ts";

/**
 * Tests for browser data loading — loadBrowserConversations (pagination,
 * search, tags, generation guards) and loadAvailableTags.
 */

let apiCalls: { method: string; args: unknown[] }[];
let listResult: { data: Conversation[]; total: number; has_more: boolean };
let searchResult: { aggregations: { tags: { name: string; count: number }[] } };

function makeConvo(id: string, marker = ""): Conversation {
  return {
    id,
    object: "conversation",
    title: `Convo ${id}`,
    created_at: 1700000000,
    updated_at: 1700000000,
    model: "gpt-4",
    items: [],
    metadata: {},
    marker,
  } as Conversation;
}

beforeEach(() => {
  apiCalls = [];
  listResult = { data: [], total: 0, has_more: false };
  searchResult = { aggregations: { tags: [] } };

  bState.selectedIds.clear();
  bState.conversations = [];
  bState.tab = "all";
  bState.isLoading = false;
  bState.loadGeneration = 0;
  bState.pagination = { currentPage: 1, pageSize: 50, totalItems: 0 };

  search.query = "";
  search.tagFilters = [];
  search.availableTags = [];

  initBrowserDom(createDomRoot(BROWSER_DOM_IDS, BROWSER_DOM_EXTRAS, BROWSER_DOM_TAGS));

  initBrowserDeps({
    api: {
      listConversations: async (opts: any) => {
        apiCalls.push({ method: "listConversations", args: [opts] });
        return { ok: true, data: listResult };
      },
      search: async (req: any) => {
        apiCalls.push({ method: "search", args: [req] });
        return { ok: true, data: searchResult };
      },
    } as any,
    notify: { info: () => {}, error: () => {}, warn: () => {} } as any,
    dialogs: {} as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    chatService: { refreshSidebar: async () => {}, selectChat: async () => {}, getSessions: () => [] } as any,
    download: {} as any,
    timers: mockTimers(),
    menus: { registerItem: () => ({ dispose() {} }), renderSlot: () => ({ dispose() {} }) } as any,
  });
});

// ── loadBrowserConversations ────────────────────────────────────────────────

describe("loadBrowserConversations", () => {
  test("calls API with default offset and limit", async () => {
    await loadBrowserConversations();
    expect(apiCalls.length).toBe(1);
    const opts = apiCalls[0]!.args[0] as any;
    expect(opts.offset).toBe(0);
    expect(opts.limit).toBe(50);
  });

  test("sets page before loading", async () => {
    await loadBrowserConversations(3);
    const opts = apiCalls[0]!.args[0] as any;
    expect(opts.offset).toBe(100); // (3-1)*50
    expect(bState.pagination.currentPage).toBe(3);
  });

  test("passes marker for archived tab", async () => {
    bState.tab = "archived";
    await loadBrowserConversations();
    const opts = apiCalls[0]!.args[0] as any;
    expect(opts.marker).toBe("archived");
  });

  test("no marker for all tab", async () => {
    bState.tab = "all";
    await loadBrowserConversations();
    const opts = apiCalls[0]!.args[0] as any;
    expect(opts.marker).toBeUndefined();
  });

  test("passes search query when set", async () => {
    search.query = "  rust  ";
    await loadBrowserConversations();
    const opts = apiCalls[0]!.args[0] as any;
    expect(opts.search).toBe("rust");
  });

  test("omits search when query is whitespace", async () => {
    search.query = "   ";
    await loadBrowserConversations();
    const opts = apiCalls[0]!.args[0] as any;
    expect(opts.search).toBeUndefined();
  });

  test("passes tag filters joined by space", async () => {
    search.tagFilters = ["rust", "wasm"];
    await loadBrowserConversations();
    const opts = apiCalls[0]!.args[0] as any;
    expect(opts.tags_any).toBe("rust wasm");
  });

  test("omits tags_any when no filters", async () => {
    search.tagFilters = [];
    await loadBrowserConversations();
    const opts = apiCalls[0]!.args[0] as any;
    expect(opts.tags_any).toBeUndefined();
  });

  test("populates conversations and totalItems on success", async () => {
    const convos = [makeConvo("c1"), makeConvo("c2")];
    listResult = { data: convos, total: 75, has_more: true };
    await loadBrowserConversations();
    expect(bState.conversations).toEqual(convos);
    expect(bState.pagination.totalItems).toBe(75);
  });

  test("resets isLoading after completion", async () => {
    await loadBrowserConversations();
    expect(bState.isLoading).toBe(false);
  });

  test("skips when already loading", async () => {
    bState.isLoading = true;
    await loadBrowserConversations();
    expect(apiCalls.length).toBe(0);
  });

  test("generation guard: stale response ignored", async () => {
    // Start a load, then increment generation to simulate a superseding load.
    const convos = [makeConvo("c1")];
    let resolveApi!: (v: any) => void;
    initBrowserDeps({
      api: {
        listConversations: async () => {
          return new Promise((r) => { resolveApi = r; });
        },
      } as any,
      notify: { info: () => {}, error: () => {}, warn: () => {} } as any,
      dialogs: {} as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      chatService: { refreshSidebar: async () => {}, selectChat: async () => {}, getSessions: () => [] } as any,
      download: {} as any,
      timers: mockTimers(),
      menus: { registerItem: () => ({ dispose() {} }), renderSlot: () => ({ dispose() {} }) } as any,
    });

    const promise = loadBrowserConversations();
    // Simulate a second load starting (increments generation).
    bState.loadGeneration++;
    resolveApi({ ok: true, data: { data: convos, total: 1, has_more: false } });
    await promise;
    // The stale result should be discarded.
    expect(bState.conversations).toEqual([]);
  });
});

// ── loadAvailableTags ───────────────────────────────────────────────────────

describe("loadAvailableTags", () => {
  test("calls search API with tags aggregation", async () => {
    await loadAvailableTags();
    expect(apiCalls.length).toBe(1);
    const req = apiCalls[0]!.args[0] as any;
    expect(req.scope).toBe("conversations");
    expect(req.aggregations).toEqual(["tags"]);
    expect(req.limit).toBe(1);
  });

  test("populates availableTags on success", async () => {
    searchResult = { aggregations: { tags: [{ name: "rust", count: 5 }, { name: "wasm", count: 3 }] } };
    await loadAvailableTags();
    expect(search.availableTags).toEqual([
      { name: "rust", count: 5 },
      { name: "wasm", count: 3 },
    ]);
  });

  test("does not update tags on failed response", async () => {
    search.availableTags = [{ name: "existing", count: 1 }];
    initBrowserDeps({
      api: {
        search: async () => ({ ok: false, error: "Server error" }),
      } as any,
      notify: { info: () => {}, error: () => {}, warn: () => {} } as any,
      dialogs: {} as any,
      events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
      chatService: { refreshSidebar: async () => {}, selectChat: async () => {}, getSessions: () => [] } as any,
      download: {} as any,
      timers: mockTimers(),
      menus: { registerItem: () => ({ dispose() {} }), renderSlot: () => ({ dispose() {} }) } as any,
    });
    await loadAvailableTags();
    // State unchanged.
    expect(search.availableTags).toEqual([{ name: "existing", count: 1 }]);
  });
});
