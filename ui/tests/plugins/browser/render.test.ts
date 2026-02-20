import { describe, test, expect, beforeEach } from "bun:test";
import {
  syncBrowserTabs,
  renderBrowserTags,
  updateBrowserToolbar,
  renderBrowserCards,
} from "../../../src/plugins/browser/render.ts";
import { bState, search } from "../../../src/plugins/browser/state.ts";
import { initBrowserDom, getBrowserDom } from "../../../src/plugins/browser/dom.ts";
import { initBrowserDeps } from "../../../src/plugins/browser/deps.ts";
import {
  createDomRoot,
  BROWSER_DOM_IDS,
  BROWSER_DOM_TAGS,
  BROWSER_DOM_EXTRAS,
} from "../../helpers/dom.ts";
import { mockTimers, mockNotifications } from "../../helpers/mocks.ts";
import type { Conversation } from "../../../src/types.ts";

/**
 * Tests for browser/render — tab sync, tag rendering, toolbar state.
 */

let notifs: ReturnType<typeof mockNotifications>;

function makeConvo(id: string): Conversation {
  return {
    id,
    object: "conversation",
    title: `Convo ${id}`,
    created_at: 1700000000,
    updated_at: 1700000000,
    model: "gpt-4",
    items: [],
    metadata: {},
  } as Conversation;
}

beforeEach(() => {
  notifs = mockNotifications();
  initBrowserDom(createDomRoot(BROWSER_DOM_IDS, BROWSER_DOM_EXTRAS, BROWSER_DOM_TAGS));

  bState.conversations = [];
  bState.selectedIds = new Set();
  bState.tab = "all";
  bState.isLoading = false;
  bState.loadGeneration = 0;
  bState.pagination = { currentPage: 1, pageSize: 50, totalItems: 0 };

  search.query = "";
  search.tagFilters = [];
  search.availableTags = [];

  initBrowserDeps({
    api: {} as any,
    notify: notifs.mock,
    dialogs: { confirm: async () => true } as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    chatService: {} as any,
    download: {} as any,
    timers: mockTimers(),
    menus: { registerItem: () => ({ dispose() {} }), renderSlot: () => ({ dispose() {} }) } as any,
  });
});

// ── syncBrowserTabs ─────────────────────────────────────────────────────────

describe("syncBrowserTabs", () => {
  test("all tab active when tab is 'all' and no tag filter", () => {
    bState.tab = "all";
    syncBrowserTabs();
    const dom = getBrowserDom();
    expect(dom.tabAll.className).toContain("active");
    expect(dom.tabArchived.className).not.toContain("active");
  });

  test("archived tab active when tab is 'archived'", () => {
    bState.tab = "archived";
    syncBrowserTabs();
    const dom = getBrowserDom();
    expect(dom.tabArchived.className).toContain("active");
  });

  test("both tabs get dimmed class when tag filter active", () => {
    search.tagFilters = ["rust"];
    syncBrowserTabs();
    const dom = getBrowserDom();
    expect(dom.tabAll.className).toContain("dimmed");
    expect(dom.tabArchived.className).toContain("dimmed");
  });

  test("archive button visible, restore hidden for 'all' tab", () => {
    bState.tab = "all";
    syncBrowserTabs();
    const dom = getBrowserDom();
    expect(dom.archiveBtn.classList.contains("hidden")).toBe(false);
    expect(dom.restoreBtn.classList.contains("hidden")).toBe(true);
  });

  test("restore button visible, archive hidden for 'archived' tab", () => {
    bState.tab = "archived";
    syncBrowserTabs();
    const dom = getBrowserDom();
    expect(dom.archiveBtn.classList.contains("hidden")).toBe(true);
    expect(dom.restoreBtn.classList.contains("hidden")).toBe(false);
  });

  test("both archive/restore visible when tag filter active", () => {
    search.tagFilters = ["rust"];
    syncBrowserTabs();
    const dom = getBrowserDom();
    expect(dom.archiveBtn.classList.contains("hidden")).toBe(false);
    expect(dom.restoreBtn.classList.contains("hidden")).toBe(false);
  });
});

// ── renderBrowserTags ───────────────────────────────────────────────────────

describe("renderBrowserTags", () => {
  test("hides tags section when no tags", () => {
    search.availableTags = [];
    renderBrowserTags();
    expect(getBrowserDom().tagsSection.classList.contains("hidden")).toBe(true);
  });

  test("shows tags section when tags exist", () => {
    search.availableTags = [{ name: "rust", count: 5 }];
    renderBrowserTags();
    expect(getBrowserDom().tagsSection.classList.contains("hidden")).toBe(false);
  });

  test("renders tag buttons with name and count", () => {
    search.availableTags = [
      { name: "rust", count: 5 },
      { name: "wasm", count: 3 },
    ];
    renderBrowserTags();
    const buttons = getBrowserDom().tagsEl.querySelectorAll("button");
    expect(buttons.length).toBe(2);

    const firstBtn = buttons[0]!;
    expect(firstBtn.textContent).toContain("rust");
    expect(firstBtn.textContent).toContain("5");

    const secondBtn = buttons[1]!;
    expect(secondBtn.textContent).toContain("wasm");
    expect(secondBtn.textContent).toContain("3");
  });

  test("active tag filter gets active class", () => {
    search.availableTags = [{ name: "rust", count: 5 }];
    search.tagFilters = ["rust"];
    renderBrowserTags();
    const btn = getBrowserDom().tagsEl.querySelector("button")!;
    expect(btn.classList.contains("active")).toBe(true);
  });

  test("inactive tag does not have active class", () => {
    search.availableTags = [{ name: "rust", count: 5 }];
    search.tagFilters = [];
    renderBrowserTags();
    const btn = getBrowserDom().tagsEl.querySelector("button")!;
    expect(btn.classList.contains("active")).toBe(false);
  });

  test("tag button has data-tag attribute", () => {
    search.availableTags = [{ name: "rust", count: 5 }];
    renderBrowserTags();
    const btn = getBrowserDom().tagsEl.querySelector<HTMLElement>("button")!;
    expect(btn.dataset["tag"]).toBe("rust");
  });

  test("active tag count has accent class", () => {
    search.availableTags = [{ name: "rust", count: 5 }];
    search.tagFilters = ["rust"];
    renderBrowserTags();
    const countSpan = getBrowserDom().tagsEl.querySelector(".text-xs")!;
    expect(countSpan.className).toContain("text-accent");
  });

  test("inactive tag count has subtle class", () => {
    search.availableTags = [{ name: "rust", count: 5 }];
    search.tagFilters = [];
    renderBrowserTags();
    const countSpan = getBrowserDom().tagsEl.querySelector(".text-xs")!;
    expect(countSpan.className).toContain("text-text-subtle");
  });
});

// ── updateBrowserToolbar ────────────────────────────────────────────────────

describe("updateBrowserToolbar", () => {
  test("disables action buttons when no selection", () => {
    bState.selectedIds = new Set();
    updateBrowserToolbar();
    const dom = getBrowserDom();
    expect(dom.deleteBtn.disabled).toBe(true);
    expect(dom.exportBtn.disabled).toBe(true);
    expect(dom.archiveBtn.disabled).toBe(true);
    expect(dom.restoreBtn.disabled).toBe(true);
  });

  test("enables action buttons when selection exists", () => {
    bState.selectedIds = new Set(["conv-1"]);
    updateBrowserToolbar();
    const dom = getBrowserDom();
    expect(dom.deleteBtn.disabled).toBe(false);
    expect(dom.exportBtn.disabled).toBe(false);
  });

  test("shows bulk actions bar when selection exists", () => {
    bState.selectedIds = new Set(["conv-1"]);
    updateBrowserToolbar();
    expect(getBrowserDom().bulkActions.classList.contains("active")).toBe(true);
  });

  test("hides cancel button when no selection", () => {
    bState.selectedIds = new Set();
    updateBrowserToolbar();
    expect(getBrowserDom().cancelBtn.classList.contains("hidden")).toBe(true);
  });

  test("shows cancel button when selection exists", () => {
    bState.selectedIds = new Set(["conv-1"]);
    updateBrowserToolbar();
    expect(getBrowserDom().cancelBtn.classList.contains("hidden")).toBe(false);
  });

  test("select all button shows 'Deselect All' when all selected", () => {
    bState.conversations = [makeConvo("c1"), makeConvo("c2")];
    bState.selectedIds = new Set(["c1", "c2"]);
    updateBrowserToolbar();
    expect(getBrowserDom().selectAllBtn.textContent).toBe("Deselect All");
  });

  test("select all button shows 'Select All' when not all selected", () => {
    bState.conversations = [makeConvo("c1"), makeConvo("c2")];
    bState.selectedIds = new Set(["c1"]);
    updateBrowserToolbar();
    expect(getBrowserDom().selectAllBtn.textContent).toBe("Select All");
  });

  test("tabs get opacity class when tag filter is active", () => {
    search.tagFilters = ["rust"];
    updateBrowserToolbar();
    const dom = getBrowserDom();
    expect(dom.tabAll.classList.contains("opacity-40")).toBe(true);
    expect(dom.tabArchived.classList.contains("opacity-40")).toBe(true);
  });

  test("tabs normal when no tag filter", () => {
    search.tagFilters = [];
    updateBrowserToolbar();
    const dom = getBrowserDom();
    expect(dom.tabAll.classList.contains("opacity-40")).toBe(false);
  });
});

// ── renderBrowserCards ──────────────────────────────────────────────────────

describe("renderBrowserCards", () => {
  test("shows empty state when no conversations", () => {
    bState.conversations = [];
    renderBrowserCards();
    const empty = getBrowserDom().cardsEl.querySelector("[data-empty-state]");
    expect(empty).not.toBeNull();
  });

  test("shows 'No archived' message in archived tab", () => {
    bState.conversations = [];
    bState.tab = "archived";
    renderBrowserCards();
    const empty = getBrowserDom().cardsEl.querySelector("[data-empty-state]")!;
    expect(empty.textContent).toContain("archived");
  });

  test("renders cards for conversations", () => {
    bState.conversations = [makeConvo("c1"), makeConvo("c2")];
    renderBrowserCards();
    const cards = getBrowserDom().cardsEl.querySelectorAll(".browser-card");
    expect(cards.length).toBe(2);
  });

  test("shows search indicator when query is present", () => {
    bState.conversations = [makeConvo("c1")];
    search.query = "test";
    renderBrowserCards();
    const indicator = getBrowserDom().cardsEl.querySelector(".search-indicator");
    expect(indicator).not.toBeNull();
    expect(indicator!.textContent).toContain("test");
  });
});
