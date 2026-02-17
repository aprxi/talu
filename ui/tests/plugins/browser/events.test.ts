import { describe, test, expect, beforeEach } from "bun:test";
import { wireEvents } from "../../../src/plugins/browser/events.ts";
import { bState, search } from "../../../src/plugins/browser/state.ts";
import { initBrowserDom, getBrowserDom } from "../../../src/plugins/browser/dom.ts";
import { initBrowserDeps } from "../../../src/plugins/browser/deps.ts";
import { createDomRoot, BROWSER_DOM_IDS, BROWSER_DOM_EXTRAS } from "../../helpers/dom.ts";
import { mockControllableTimers } from "../../helpers/mocks.ts";

/**
 * Tests for browser event wiring — search input debouncing, tab switching,
 * select-all toggle, cancel, and clear button.
 *
 * Strategy: wire events with a controllable timer mock, then dispatch DOM
 * events and verify state mutations. API is mocked to return empty results.
 */

// -- Mock state --------------------------------------------------------------

let ct: ReturnType<typeof mockControllableTimers>;
let apiCalls: { method: string; args: unknown[] }[];

beforeEach(() => {
  ct = mockControllableTimers();
  apiCalls = [];

  // Reset state.
  search.query = "";
  search.tagFilters = [];
  search.results = [];
  search.cursor = null;
  search.hasMore = true;
  search.isLoading = false;
  search.availableTags = [];
  bState.selectedIds.clear();
  bState.conversations = [];
  bState.tab = "all";

  // DOM.
  initBrowserDom(createDomRoot(BROWSER_DOM_IDS, BROWSER_DOM_EXTRAS));

  // Deps with controllable timer.
  initBrowserDeps({
    api: {
      search: async (req: any) => {
        apiCalls.push({ method: "search", args: [req] });
        return { ok: true, data: { data: [], cursor: null, has_more: false } };
      },
      getConversation: async () => ({ ok: false, error: "mock" }),
    } as any,
    notify: { info: () => {}, error: () => {}, warn: () => {} } as any,
    dialogs: {} as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    chatService: {
      getSessions: () => [],
      refreshSidebar: async () => {},
      selectChat: async () => {},
    } as any,
    download: {} as any,
    timers: ct.timers,
    menus: {
      registerItem: () => ({ dispose() {} }),
      renderSlot: () => ({ dispose() {} }),
    } as any,
  });
});

// -- Helpers -----------------------------------------------------------------

function makeConvo(id: string, marker = ""): any {
  return {
    id, title: `Convo ${id}`, marker,
    object: "conversation", created_at: 1700000000, updated_at: 1700000000,
    model: "gpt-4", items: [], metadata: {},
  };
}

// ── Tab switching ─────────────────────────────────────────────────────────────

describe("Tab switching", () => {
  test("clicking archived tab switches to archived", () => {
    wireEvents();
    getBrowserDom().tabArchived.dispatchEvent(new Event("click"));
    expect(bState.tab).toBe("archived");
  });

  test("clicking all tab switches back to all", () => {
    bState.tab = "archived";
    wireEvents();
    getBrowserDom().tabAll.dispatchEvent(new Event("click"));
    expect(bState.tab).toBe("all");
  });

  test("tab switch clears selections", () => {
    bState.selectedIds.add("c1");
    bState.selectedIds.add("c2");
    wireEvents();
    getBrowserDom().tabArchived.dispatchEvent(new Event("click"));
    expect(bState.selectedIds.size).toBe(0);
  });

  test("no-op when clicking same tab", () => {
    bState.tab = "all";
    wireEvents();
    // Add selection to verify it's NOT cleared.
    bState.selectedIds.add("c1");
    getBrowserDom().tabAll.dispatchEvent(new Event("click"));
    expect(bState.selectedIds.size).toBe(1); // unchanged
  });

  test("tabs disabled when tag filter is active", () => {
    search.tagFilters = ["rust"];
    wireEvents();
    getBrowserDom().tabArchived.dispatchEvent(new Event("click"));
    expect(bState.tab).toBe("all"); // unchanged
  });
});

// ── Search debouncing ─────────────────────────────────────────────────────────

describe("Search debouncing", () => {
  test("input schedules 300ms debounce", () => {
    wireEvents();
    const dom = getBrowserDom();
    (dom.searchInput as HTMLInputElement).value = "test";
    dom.searchInput.dispatchEvent(new Event("input"));

    expect(ct.pending.length).toBe(1);
    expect(ct.pending[0]!.ms).toBe(300);
  });

  test("rapid typing cancels previous timer", () => {
    wireEvents();
    const dom = getBrowserDom();
    (dom.searchInput as HTMLInputElement).value = "t";
    dom.searchInput.dispatchEvent(new Event("input"));
    (dom.searchInput as HTMLInputElement).value = "te";
    dom.searchInput.dispatchEvent(new Event("input"));
    (dom.searchInput as HTMLInputElement).value = "tes";
    dom.searchInput.dispatchEvent(new Event("input"));

    expect(ct.pending[0]!.disposed).toBe(true);
    expect(ct.pending[1]!.disposed).toBe(true);
    expect(ct.pending[2]!.disposed).toBe(false);
  });

  test("debounce callback resets search state", async () => {
    search.results = [makeConvo("old")];
    search.cursor = "old-cursor";
    search.hasMore = false;

    wireEvents();
    const dom = getBrowserDom();
    (dom.searchInput as HTMLInputElement).value = "new query";
    dom.searchInput.dispatchEvent(new Event("input"));

    // Fire the debounced callback.
    ct.pending[0]!.fn();
    await new Promise((r) => setTimeout(r, 10));

    expect(search.query).toBe("new query");
    expect(search.results).toEqual([]);
    expect(search.cursor).toBeNull();
    // Note: hasMore is transiently set true by the callback, then
    // immediately overwritten by loadBrowserConversations() from the API response.
  });

  test("debounce no-op when query unchanged", async () => {
    search.query = "same";

    wireEvents();
    const dom = getBrowserDom();
    (dom.searchInput as HTMLInputElement).value = "same";
    dom.searchInput.dispatchEvent(new Event("input"));

    ct.pending[0]!.fn();
    await new Promise((r) => setTimeout(r, 10));

    // No API call because query didn't change.
    expect(apiCalls.length).toBe(0);
  });

  test("clear button shows when input has text", () => {
    wireEvents();
    const dom = getBrowserDom();
    (dom.searchInput as HTMLInputElement).value = "text";
    dom.searchInput.dispatchEvent(new Event("input"));

    expect(dom.clearBtn.classList.contains("hidden")).toBe(false);
  });

  test("clear button hides when input is empty", () => {
    wireEvents();
    const dom = getBrowserDom();
    dom.clearBtn.classList.remove("hidden"); // start visible
    (dom.searchInput as HTMLInputElement).value = "";
    dom.searchInput.dispatchEvent(new Event("input"));

    expect(dom.clearBtn.classList.contains("hidden")).toBe(true);
  });

  test("clear button resets input and triggers search", () => {
    wireEvents();
    const dom = getBrowserDom();
    (dom.searchInput as HTMLInputElement).value = "query";

    dom.clearBtn.dispatchEvent(new Event("click"));

    expect((dom.searchInput as HTMLInputElement).value).toBe("");
    // Should have scheduled a debounce timer via the input event.
    expect(ct.pending.length).toBeGreaterThan(0);
  });
});

// ── Select All / Cancel ─────────────────────────────────────────────────────

describe("Select All / Cancel", () => {
  test("select all adds all visible conversations", () => {
    bState.conversations = [makeConvo("c1"), makeConvo("c2"), makeConvo("c3")];
    wireEvents();
    getBrowserDom().selectAllBtn.dispatchEvent(new Event("click"));
    expect(bState.selectedIds.size).toBe(3);
  });

  test("select all deselects when all already selected", () => {
    bState.conversations = [makeConvo("c1"), makeConvo("c2")];
    bState.selectedIds.add("c1");
    bState.selectedIds.add("c2");
    wireEvents();
    getBrowserDom().selectAllBtn.dispatchEvent(new Event("click"));
    expect(bState.selectedIds.size).toBe(0);
  });

  test("cancel clears all selections", () => {
    bState.selectedIds.add("c1");
    bState.selectedIds.add("c2");
    wireEvents();
    getBrowserDom().cancelBtn.dispatchEvent(new Event("click"));
    expect(bState.selectedIds.size).toBe(0);
  });

  test("select all on archived tab only selects archived", () => {
    bState.tab = "archived";
    bState.conversations = [
      makeConvo("c1"),                   // not archived
      makeConvo("c2", "archived"),       // archived
      makeConvo("c3", "archived"),       // archived
    ];
    wireEvents();
    getBrowserDom().selectAllBtn.dispatchEvent(new Event("click"));
    expect(bState.selectedIds.size).toBe(2);
    expect(bState.selectedIds.has("c2")).toBe(true);
    expect(bState.selectedIds.has("c3")).toBe(true);
  });
});
