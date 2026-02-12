import { describe, test, expect, beforeEach } from "bun:test";
import { filterByTag, removeTagFilter, clearTagFilter } from "../../../src/plugins/browser/tags.ts";
import { search, bState } from "../../../src/plugins/browser/state.ts";
import { initBrowserDom } from "../../../src/plugins/browser/dom.ts";
import { initBrowserDeps } from "../../../src/plugins/browser/deps.ts";
import type { Conversation } from "../../../src/types.ts";
import { createDomRoot, BROWSER_DOM_IDS, BROWSER_DOM_EXTRAS } from "../../helpers/dom.ts";
import { mockTimers } from "../../helpers/mocks.ts";

/**
 * Tests for browser tag filtering — state mutations on search/bState.
 *
 * Strategy: initialize browser deps and DOM with minimal mocks, then test
 * that filterByTag/removeTagFilter/clearTagFilter correctly mutate the
 * search and bState objects. Render functions execute against mock DOM
 * (harmless). API calls are mocked to return empty results.
 */

// -- Setup -------------------------------------------------------------------

function makeConvo(id: string, tags: string[] = []): Conversation {
  return {
    id,
    object: "conversation",
    created_at: 1700000000,
    updated_at: 1700000000,
    model: "gpt-4",
    title: `Convo ${id}`,
    items: [],
    metadata: { tags },
  } as Conversation;
}

beforeEach(() => {
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

  initBrowserDom(createDomRoot(BROWSER_DOM_IDS, BROWSER_DOM_EXTRAS));

  // Initialize deps with mock API (search returns empty).
  initBrowserDeps({
    api: {
      search: async () => ({ ok: true, data: { data: [], cursor: null, has_more: false } }),
      getConversation: async () => ({ ok: false, error: "mock" }),
    } as any,
    notify: { info: () => {}, error: () => {}, warn: () => {} } as any,
    dialogs: {} as any,
    events: {} as any,
    chatService: {
      getSessions: () => [],
      refreshSidebar: async () => {},
    } as any,
    download: {} as any,
    timers: mockTimers(),
  });
});

// ── filterByTag ────────────────────────────────────────────────────────────

describe("filterByTag", () => {
  test("adds tag to filters when not present", () => {
    filterByTag("rust");
    expect(search.tagFilters).toEqual(["rust"]);
  });

  test("removes tag from filters when already present (toggle)", () => {
    search.tagFilters = ["rust", "python"];
    filterByTag("rust");
    expect(search.tagFilters).toEqual(["python"]);
  });

  test("resets search pagination state", () => {
    search.results = [makeConvo("c1")];
    search.cursor = "abc";
    search.hasMore = false;
    filterByTag("new");
    expect(search.results).toEqual([]);
    expect(search.cursor).toBeNull();
    expect(search.hasMore).toBe(true);
    // Note: isLoading is transiently set true by the async loadBrowserConversations().
  });

  test("clears selectedIds", () => {
    bState.selectedIds.add("c1");
    bState.selectedIds.add("c2");
    filterByTag("tag");
    expect(bState.selectedIds.size).toBe(0);
  });

  test("toggle on → off → on round-trips", () => {
    filterByTag("ts");
    expect(search.tagFilters).toContain("ts");
    filterByTag("ts");
    expect(search.tagFilters).not.toContain("ts");
    filterByTag("ts");
    expect(search.tagFilters).toContain("ts");
  });

  test("multiple tags accumulate", () => {
    filterByTag("a");
    filterByTag("b");
    filterByTag("c");
    expect(search.tagFilters).toEqual(["a", "b", "c"]);
  });
});

// ── removeTagFilter ────────────────────────────────────────────────────────

describe("removeTagFilter", () => {
  test("removes specific tag from filters", () => {
    search.tagFilters = ["a", "b", "c"];
    removeTagFilter("b");
    expect(search.tagFilters).toEqual(["a", "c"]);
  });

  test("no-op when tag not in filters", () => {
    search.tagFilters = ["a"];
    removeTagFilter("z");
    expect(search.tagFilters).toEqual(["a"]);
  });

  test("resets search pagination state", () => {
    search.tagFilters = ["a", "b"];
    search.results = [makeConvo("c1")];
    search.cursor = "xyz";
    removeTagFilter("a");
    expect(search.results).toEqual([]);
    expect(search.cursor).toBeNull();
    expect(search.hasMore).toBe(true);
  });

  test("removing last tag leaves empty filters", () => {
    search.tagFilters = ["only"];
    removeTagFilter("only");
    expect(search.tagFilters).toEqual([]);
  });
});

// ── clearTagFilter ─────────────────────────────────────────────────────────

describe("clearTagFilter", () => {
  test("clears all tag filters", () => {
    search.tagFilters = ["a", "b", "c"];
    clearTagFilter();
    expect(search.tagFilters).toEqual([]);
  });

  test("resets search state", () => {
    search.tagFilters = ["a"];
    search.results = [makeConvo("c1")];
    search.cursor = "cur";
    search.hasMore = false;
    search.isLoading = true;
    clearTagFilter();
    expect(search.results).toEqual([]);
    expect(search.cursor).toBeNull();
    expect(search.hasMore).toBe(true);
    expect(search.isLoading).toBe(false);
  });

  test("no-op when already empty", () => {
    clearTagFilter();
    expect(search.tagFilters).toEqual([]);
  });
});
