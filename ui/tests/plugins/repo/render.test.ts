import { describe, test, expect, beforeEach } from "bun:test";
import {
  renderModelsTable,
  renderDiscoverResults,
  renderStats,
  renderSortIndicators,
  updateRepoToolbar,
  syncRepoTabs,
} from "../../../src/plugins/repo/render.ts";
import { repoState } from "../../../src/plugins/repo/state.ts";
import { initRepoDom, getRepoDom } from "../../../src/plugins/repo/dom.ts";
import { initRepoDeps } from "../../../src/plugins/repo/deps.ts";
import { createDomRoot, REPO_DOM_IDS, REPO_DOM_TAGS } from "../../helpers/dom.ts";
import { mockTimers } from "../../helpers/mocks.ts";

/**
 * Tests for repo plugin rendering — models table, discover results list,
 * stats display, toolbar state, tab sync, and size filtering.
 */

beforeEach(() => {
  // Reset state.
  repoState.tab = "local";
  repoState.models = [];
  repoState.totalSizeBytes = 0;
  repoState.searchResults = [];
  repoState.searchQuery = "";
  repoState.isLoading = false;
  repoState.activeDownloads.clear();
  repoState.selectedIds.clear();
  repoState.sortBy = "name";
  repoState.sortDir = "asc";
  repoState.searchGeneration = 0;
  repoState.discoverSort = "trending";
  repoState.discoverSize = "8";
  repoState.discoverTask = "text-generation";
  repoState.discoverLibrary = "safetensors";

  // DOM.
  initRepoDom(createDomRoot(REPO_DOM_IDS, undefined, REPO_DOM_TAGS));

  // Deps.
  initRepoDeps({
    api: {} as any,
    notifications: { info: () => {}, error: () => {}, warn: () => {}, success: () => {} } as any,
    dialogs: {} as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    timers: mockTimers(),
    format: {
      date: () => "",
      dateTime: () => "Jan 1, 2025",
      relativeTime: () => "",
      duration: () => "",
      number: () => "",
    } as any,
    status: { setBusy: () => {}, setReady: () => {} } as any,
  });
});

// -- Helpers -----------------------------------------------------------------

function makeModel(id: string, opts?: Partial<{
  size_bytes: number; pinned: boolean; mtime: number;
  architecture: string; quant_scheme: string; source: string;
}>): any {
  return {
    id,
    path: `/models/${id}`,
    source: opts?.source ?? "hub",
    size_bytes: opts?.size_bytes ?? 1024,
    mtime: opts?.mtime ?? 1700000000,
    architecture: opts?.architecture ?? "llama",
    quant_scheme: opts?.quant_scheme ?? "Q4_K_M",
    pinned: opts?.pinned ?? false,
  };
}

function makeSearchResult(id: string, params = 7_000_000_000): any {
  return {
    model_id: id,
    downloads: 50000,
    likes: 1200,
    last_modified: "2025-01-15T00:00:00Z",
    params_total: params,
  };
}

// ── renderModelsTable ───────────────────────────────────────────────────────

describe("renderModelsTable", () => {
  test("renders correct number of rows", () => {
    repoState.models = [makeModel("m1"), makeModel("m2"), makeModel("m3")];
    renderModelsTable();

    const dom = getRepoDom();
    const rows = dom.tbody.querySelectorAll(".files-row");
    expect(rows.length).toBe(3);
  });

  test("renders empty state when no models", () => {
    repoState.models = [];
    renderModelsTable();

    const dom = getRepoDom();
    expect(dom.tbody.innerHTML).toContain("No cached models");
  });

  test("renders pinned empty state on pinned tab", () => {
    repoState.tab = "pinned";
    repoState.models = [];
    renderModelsTable();

    const dom = getRepoDom();
    expect(dom.tbody.innerHTML).toContain("No pinned models");
  });

  test("renders search empty state when query has no matches", () => {
    repoState.models = [];
    repoState.searchQuery = "nonexistent";
    renderModelsTable();

    const dom = getRepoDom();
    expect(dom.tbody.innerHTML).toContain("nonexistent");
  });

  test("row contains model ID in name cell", () => {
    repoState.models = [makeModel("org/my-model")];
    renderModelsTable();

    const dom = getRepoDom();
    const nameSpan = dom.tbody.querySelector(".files-name-text");
    expect(nameSpan).not.toBeNull();
    expect(nameSpan!.textContent).toBe("org/my-model");
  });

  test("row has data-id attribute", () => {
    repoState.models = [makeModel("m1")];
    renderModelsTable();

    const dom = getRepoDom();
    const row = dom.tbody.querySelector("tr[data-id]");
    expect(row).not.toBeNull();
    expect(row!.getAttribute("data-id")).toBe("m1");
  });

  test("selected row has files-row-selected class", () => {
    repoState.models = [makeModel("m1")];
    repoState.selectedIds.add("m1");
    renderModelsTable();

    const dom = getRepoDom();
    const row = dom.tbody.querySelector(".files-row");
    expect(row!.classList.contains("files-row-selected")).toBe(true);
  });

  test("pinned model shows pinned class on pin button", () => {
    repoState.models = [makeModel("m1", { pinned: true })];
    renderModelsTable();

    const dom = getRepoDom();
    const pinBtn = dom.tbody.querySelector(".repo-pin-btn");
    expect(pinBtn).not.toBeNull();
    expect(pinBtn!.classList.contains("pinned")).toBe(true);
  });

  test("unpinned model does not have pinned class", () => {
    repoState.models = [makeModel("m1", { pinned: false })];
    renderModelsTable();

    const dom = getRepoDom();
    const pinBtn = dom.tbody.querySelector(".repo-pin-btn");
    expect(pinBtn!.classList.contains("pinned")).toBe(false);
  });

  test("renders source badge when source is present", () => {
    repoState.models = [makeModel("m1", { source: "hub" })];
    renderModelsTable();

    const dom = getRepoDom();
    const badge = dom.tbody.querySelector(".repo-source-badge");
    expect(badge).not.toBeNull();
    expect(badge!.textContent).toBe("hub");
  });

  test("renders quantization badge when quant_scheme is present", () => {
    repoState.models = [makeModel("m1", { quant_scheme: "Q4_K_M" })];
    renderModelsTable();

    const dom = getRepoDom();
    const badge = dom.tbody.querySelector(".repo-quant-badge");
    expect(badge).not.toBeNull();
    expect(badge!.textContent).toBe("Q4_K_M");
  });

  test("client-side search filters models by ID", () => {
    repoState.models = [
      makeModel("org/llama-7b"),
      makeModel("org/mistral-7b", { architecture: "mistral" }),
      makeModel("org/llama-13b"),
    ];
    repoState.searchQuery = "llama";
    renderModelsTable();

    const dom = getRepoDom();
    const rows = dom.tbody.querySelectorAll(".files-row");
    expect(rows.length).toBe(2);
  });

  test("client-side search filters by architecture", () => {
    repoState.models = [
      makeModel("m1", { architecture: "llama" }),
      makeModel("m2", { architecture: "mistral" }),
    ];
    repoState.searchQuery = "mistral";
    renderModelsTable();

    const dom = getRepoDom();
    const rows = dom.tbody.querySelectorAll(".files-row");
    expect(rows.length).toBe(1);
  });

  test("pinned tab shows only pinned models", () => {
    repoState.tab = "pinned";
    repoState.models = [
      makeModel("m1", { pinned: true }),
      makeModel("m2", { pinned: false }),
      makeModel("m3", { pinned: true }),
    ];
    renderModelsTable();

    const dom = getRepoDom();
    const rows = dom.tbody.querySelectorAll(".files-row");
    expect(rows.length).toBe(2);
  });

  test("sorts by name ascending", () => {
    repoState.sortBy = "name";
    repoState.sortDir = "asc";
    repoState.models = [
      makeModel("z-model"),
      makeModel("a-model"),
      makeModel("m-model"),
    ];
    renderModelsTable();

    const dom = getRepoDom();
    const rows = dom.tbody.querySelectorAll("tr[data-id]");
    expect(rows[0]!.getAttribute("data-id")).toBe("a-model");
    expect(rows[2]!.getAttribute("data-id")).toBe("z-model");
  });

  test("sorts by size descending", () => {
    repoState.sortBy = "size";
    repoState.sortDir = "desc";
    repoState.models = [
      makeModel("small", { size_bytes: 100 }),
      makeModel("large", { size_bytes: 10000 }),
      makeModel("medium", { size_bytes: 5000 }),
    ];
    renderModelsTable();

    const dom = getRepoDom();
    const rows = dom.tbody.querySelectorAll("tr[data-id]");
    expect(rows[0]!.getAttribute("data-id")).toBe("large");
    expect(rows[2]!.getAttribute("data-id")).toBe("small");
  });

  test("updates count display", () => {
    repoState.models = [makeModel("m1"), makeModel("m2")];
    renderModelsTable();

    const dom = getRepoDom();
    expect(dom.count.textContent).toBe("2 models");
  });

  test("singular count for 1 model", () => {
    repoState.models = [makeModel("m1")];
    renderModelsTable();

    const dom = getRepoDom();
    expect(dom.count.textContent).toBe("1 model");
  });

  test("sorts by date ascending", () => {
    repoState.sortBy = "date";
    repoState.sortDir = "asc";
    repoState.models = [
      makeModel("newest", { mtime: 1700000300 }),
      makeModel("oldest", { mtime: 1700000100 }),
      makeModel("middle", { mtime: 1700000200 }),
    ];
    renderModelsTable();

    const dom = getRepoDom();
    const rows = dom.tbody.querySelectorAll("tr[data-id]");
    expect(rows[0]!.getAttribute("data-id")).toBe("oldest");
    expect(rows[1]!.getAttribute("data-id")).toBe("middle");
    expect(rows[2]!.getAttribute("data-id")).toBe("newest");
  });

  test("model without source renders no source badge", () => {
    repoState.models = [makeModel("m1", { source: "" })];
    renderModelsTable();

    const dom = getRepoDom();
    const badge = dom.tbody.querySelector(".repo-source-badge");
    expect(badge).toBeNull();
  });

  test("model without quant_scheme renders dash", () => {
    repoState.models = [makeModel("m1", { quant_scheme: "" })];
    renderModelsTable();

    const dom = getRepoDom();
    // Find the quant cell (4th td in the row after check, name, arch).
    const cells = dom.tbody.querySelectorAll(".files-row td");
    // Quant is 4th cell (index 3). With no quant_scheme, it should show "—".
    const quantCell = cells[3] as HTMLElement;
    expect(quantCell.textContent).toBe("—");
  });

  test("model with mtime=0 renders dash for date", () => {
    repoState.models = [makeModel("m1", { mtime: 0 })];
    renderModelsTable();

    const dom = getRepoDom();
    const dateCell = dom.tbody.querySelector(".files-cell-date") as HTMLElement;
    expect(dateCell.textContent).toBe("—");
  });
});

// ── renderDiscoverResults ───────────────────────────────────────────────────

describe("renderDiscoverResults", () => {
  test("renders results as list items", () => {
    repoState.searchResults = [
      makeSearchResult("org/model-a"),
      makeSearchResult("org/model-b"),
    ];
    renderDiscoverResults();

    const dom = getRepoDom();
    const items = dom.discoverResults.querySelectorAll(".repo-discover-item");
    expect(items.length).toBe(2);
  });

  test("renders model ID in title", () => {
    repoState.searchResults = [makeSearchResult("org/llama-7b")];
    renderDiscoverResults();

    const dom = getRepoDom();
    const title = dom.discoverResults.querySelector(".repo-discover-item-title");
    expect(title!.textContent).toBe("org/llama-7b");
  });

  test("renders download button with data-action", () => {
    repoState.searchResults = [makeSearchResult("org/model")];
    renderDiscoverResults();

    const dom = getRepoDom();
    const btn = dom.discoverResults.querySelector("[data-action='download']");
    expect(btn).not.toBeNull();
  });

  test("renders progress bar instead of button when downloading", () => {
    repoState.searchResults = [makeSearchResult("org/model")];
    repoState.activeDownloads.set("org/model", {
      modelId: "org/model", current: 50, total: 100, label: "50%", status: "downloading",
    });
    renderDiscoverResults();

    const dom = getRepoDom();
    const progress = dom.discoverResults.querySelector(".repo-progress-bar");
    expect(progress).not.toBeNull();
    expect((progress as HTMLElement).style.width).toBe("50%");
    // No download button when downloading.
    const btn = dom.discoverResults.querySelector("[data-action='download']");
    expect(btn).toBeNull();
  });

  test("renders metadata (downloads, likes, params, date)", () => {
    repoState.searchResults = [makeSearchResult("org/model", 7_000_000_000)];
    renderDiscoverResults();

    const dom = getRepoDom();
    const meta = dom.discoverResults.querySelector(".repo-discover-item-meta");
    expect(meta).not.toBeNull();
    const html = meta!.innerHTML;
    expect(html).toContain("50.0K"); // downloads
    expect(html).toContain("1.2K"); // likes
    expect(html).toContain("7.0B"); // params
    expect(html).toContain("2025-01-15"); // date
  });

  test("shows spinner when loading", () => {
    repoState.isLoading = true;
    renderDiscoverResults();

    const dom = getRepoDom();
    const spinner = dom.discoverResults.querySelector(".spinner");
    expect(spinner).not.toBeNull();
  });

  test("shows empty state when no query", () => {
    repoState.searchResults = [];
    repoState.searchQuery = "";
    renderDiscoverResults();

    const dom = getRepoDom();
    expect(dom.discoverResults.innerHTML).toContain("Search HuggingFace");
  });

  test("shows empty state when query has no results", () => {
    repoState.searchResults = [];
    repoState.searchQuery = "nonexistent";
    renderDiscoverResults();

    const dom = getRepoDom();
    expect(dom.discoverResults.innerHTML).toContain("nonexistent");
  });

  test("updates count display", () => {
    repoState.searchResults = [
      makeSearchResult("a"),
      makeSearchResult("b"),
      makeSearchResult("c"),
    ];
    renderDiscoverResults();

    const dom = getRepoDom();
    expect(dom.count.textContent).toBe("3 results");
  });

  test("result with params_total=0 does not render params span", () => {
    repoState.discoverSize = "any"; // Don't filter out.
    repoState.searchResults = [makeSearchResult("org/unknown-params", 0)];
    renderDiscoverResults();

    const dom = getRepoDom();
    const meta = dom.discoverResults.querySelector(".repo-discover-item-meta")!;
    expect(meta.innerHTML).not.toContain("params");
  });

  // -- Size filtering (client-side) --

  test("size filter ≤8B excludes models over 8.5B params", () => {
    repoState.discoverSize = "8";
    repoState.searchResults = [
      makeSearchResult("small", 3_000_000_000),
      makeSearchResult("medium", 7_000_000_000),
      makeSearchResult("large", 13_000_000_000),
      makeSearchResult("huge", 70_000_000_000),
    ];
    renderDiscoverResults();

    const dom = getRepoDom();
    const items = dom.discoverResults.querySelectorAll(".repo-discover-item");
    expect(items.length).toBe(2);
  });

  test("size filter 'any' shows all models", () => {
    repoState.discoverSize = "any";
    repoState.searchResults = [
      makeSearchResult("small", 1_000_000_000),
      makeSearchResult("huge", 200_000_000_000),
    ];
    renderDiscoverResults();

    const dom = getRepoDom();
    const items = dom.discoverResults.querySelectorAll(".repo-discover-item");
    expect(items.length).toBe(2);
  });

  test("size filter excludes models with params_total=0 (unknown)", () => {
    repoState.discoverSize = "8";
    repoState.searchResults = [
      makeSearchResult("known", 3_000_000_000),
      makeSearchResult("unknown", 0),
    ];
    renderDiscoverResults();

    const dom = getRepoDom();
    const items = dom.discoverResults.querySelectorAll(".repo-discover-item");
    expect(items.length).toBe(1);
  });

  test("size filter ≤1B uses correct threshold", () => {
    repoState.discoverSize = "1";
    repoState.searchResults = [
      makeSearchResult("tiny", 500_000_000),
      makeSearchResult("small", 1_400_000_000),
      makeSearchResult("too-big", 1_600_000_000),
    ];
    renderDiscoverResults();

    const dom = getRepoDom();
    const items = dom.discoverResults.querySelectorAll(".repo-discover-item");
    expect(items.length).toBe(2); // 500M and 1.4B are under 1.5B threshold
  });

  test("empty results after size filter shows empty state", () => {
    repoState.discoverSize = "1";
    repoState.searchQuery = "llama";
    repoState.searchResults = [
      makeSearchResult("big", 70_000_000_000),
    ];
    renderDiscoverResults();

    const dom = getRepoDom();
    const items = dom.discoverResults.querySelectorAll(".repo-discover-item");
    expect(items.length).toBe(0);
    expect(dom.discoverResults.innerHTML).toContain("llama");
  });
});

// ── renderSortIndicators ────────────────────────────────────────────────────

describe("renderSortIndicators", () => {
  test("adds sorted class and arrow to active column", () => {
    repoState.sortBy = "size";
    repoState.sortDir = "asc";
    const dom = getRepoDom();

    // Add sortable th elements to thead.
    const thName = document.createElement("th");
    thName.dataset["sort"] = "name";
    const thSize = document.createElement("th");
    thSize.dataset["sort"] = "size";
    dom.thead.appendChild(thName);
    dom.thead.appendChild(thSize);

    renderSortIndicators();

    expect(thSize.classList.contains("files-th-sorted")).toBe(true);
    expect(thName.classList.contains("files-th-sorted")).toBe(false);

    const arrow = thSize.querySelector(".sort-arrow");
    expect(arrow).not.toBeNull();
    expect(arrow!.textContent).toContain("\u25B2"); // Up arrow for asc.
  });

  test("shows down arrow for descending sort", () => {
    repoState.sortBy = "name";
    repoState.sortDir = "desc";
    const dom = getRepoDom();

    const th = document.createElement("th");
    th.dataset["sort"] = "name";
    dom.thead.appendChild(th);

    renderSortIndicators();

    const arrow = th.querySelector(".sort-arrow");
    expect(arrow!.textContent).toContain("\u25BC"); // Down arrow for desc.
  });

  test("removes old arrow when re-rendered", () => {
    repoState.sortBy = "name";
    repoState.sortDir = "asc";
    const dom = getRepoDom();

    const th = document.createElement("th");
    th.dataset["sort"] = "name";
    dom.thead.appendChild(th);

    renderSortIndicators();
    // Switch to different column.
    repoState.sortBy = "size";
    renderSortIndicators();

    expect(th.querySelector(".sort-arrow")).toBeNull();
    expect(th.classList.contains("files-th-sorted")).toBe(false);
  });
});

// ── renderStats ─────────────────────────────────────────────────────────────

describe("renderStats", () => {
  test("shows model count and total size", () => {
    repoState.models = [makeModel("m1"), makeModel("m2")];
    repoState.totalSizeBytes = 1_073_741_824; // 1 GB
    renderStats();

    const dom = getRepoDom();
    const text = dom.stats.textContent!;
    expect(text).toContain("2 models");
    expect(text).toContain("1.0 GB");
  });

  test("singular for 1 model", () => {
    repoState.models = [makeModel("m1")];
    repoState.totalSizeBytes = 512;
    renderStats();

    const dom = getRepoDom();
    expect(dom.stats.textContent).toContain("1 model");
  });
});

// ── updateRepoToolbar ───────────────────────────────────────────────────────

describe("updateRepoToolbar", () => {
  test("buttons disabled when no selection", () => {
    updateRepoToolbar();
    const dom = getRepoDom();
    expect(dom.pinAllBtn.disabled).toBe(true);
    expect(dom.deleteBtn.disabled).toBe(true);
  });

  test("buttons enabled when models are selected", () => {
    repoState.selectedIds.add("m1");
    updateRepoToolbar();
    const dom = getRepoDom();
    expect(dom.pinAllBtn.disabled).toBe(false);
    expect(dom.deleteBtn.disabled).toBe(false);
  });

  test("bulk actions active class when selection exists", () => {
    repoState.selectedIds.add("m1");
    updateRepoToolbar();
    const dom = getRepoDom();
    expect(dom.bulkActions.classList.contains("active")).toBe(true);
  });

  test("cancel hidden when no selection", () => {
    updateRepoToolbar();
    const dom = getRepoDom();
    expect(dom.cancelBtn.classList.contains("hidden")).toBe(true);
  });

  test("cancel visible when selection exists", () => {
    repoState.selectedIds.add("m1");
    updateRepoToolbar();
    const dom = getRepoDom();
    expect(dom.cancelBtn.classList.contains("hidden")).toBe(false);
  });
});

// ── syncRepoTabs ────────────────────────────────────────────────────────────

describe("syncRepoTabs", () => {
  test("local tab active on local tab", () => {
    repoState.tab = "local";
    syncRepoTabs();
    const dom = getRepoDom();
    expect(dom.tabLocal.classList.contains("active")).toBe(true);
    expect(dom.tabPinned.classList.contains("active")).toBe(false);
    expect(dom.tabDiscover.classList.contains("active")).toBe(false);
  });

  test("discover tab shows discover container and hides table", () => {
    repoState.tab = "discover";
    syncRepoTabs();
    const dom = getRepoDom();
    expect(dom.tableContainer.classList.contains("hidden")).toBe(true);
    expect(dom.discoverContainer.classList.contains("hidden")).toBe(false);
  });

  test("local tab shows table and hides discover", () => {
    repoState.tab = "local";
    syncRepoTabs();
    const dom = getRepoDom();
    expect(dom.tableContainer.classList.contains("hidden")).toBe(false);
    expect(dom.discoverContainer.classList.contains("hidden")).toBe(true);
  });

  test("bulk actions hidden in discover mode", () => {
    repoState.tab = "discover";
    syncRepoTabs();
    const dom = getRepoDom();
    expect(dom.selectAllBtn.classList.contains("hidden")).toBe(true);
    expect(dom.bulkActions.classList.contains("hidden")).toBe(true);
  });

  test("bulk actions visible in local mode", () => {
    repoState.tab = "local";
    syncRepoTabs();
    const dom = getRepoDom();
    expect(dom.selectAllBtn.classList.contains("hidden")).toBe(false);
  });
});
