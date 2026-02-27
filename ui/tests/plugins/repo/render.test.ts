import { describe, test, expect, beforeEach } from "bun:test";
import {
  renderModelsTable,
  renderDiscoverResults,
  renderDownloads,
  updateDownloadProgress,
  renderStats,
  renderSortIndicators,
  updateRepoToolbar,
  syncRepoTabs,
  syncSourceToggle,
} from "../../../src/plugins/repo/render.ts";
import { repoState } from "../../../src/plugins/repo/state.ts";
import { initRepoDom, getRepoDom } from "../../../src/plugins/repo/dom.ts";
import { initRepoDeps } from "../../../src/plugins/repo/deps.ts";
import { createDomRoot, REPO_DOM_IDS, REPO_DOM_TAGS } from "../../helpers/dom.ts";
import { mockTimers } from "../../helpers/mocks.ts";

/**
 * Tests for repo plugin rendering — models table, discover results list,
 * stats display, toolbar state, tab sync, source toggle, and size filtering.
 */

beforeEach(() => {
  // Reset state.
  repoState.tab = "local";
  repoState.localSourceFilter = "all";
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
  test("renders all models with source filter 'all'", () => {
    repoState.localSourceFilter = "all";
    repoState.models = [
      makeModel("m1", { source: "hub" }),
      makeModel("m2", { source: "hub" }),
      makeModel("m3", { source: "managed" }),
    ];
    renderModelsTable();

    const dom = getRepoDom();
    const rows = dom.localTbody.querySelectorAll(".files-row");
    expect(rows.length).toBe(3);
  });

  test("renders only hub models with source filter 'hub'", () => {
    repoState.localSourceFilter = "hub";
    repoState.models = [
      makeModel("m1", { source: "hub" }),
      makeModel("m2", { source: "hub" }),
      makeModel("m3", { source: "managed" }),
    ];
    renderModelsTable();

    const dom = getRepoDom();
    const rows = dom.localTbody.querySelectorAll(".files-row");
    expect(rows.length).toBe(2);
  });

  test("renders only managed models with source filter 'managed'", () => {
    repoState.localSourceFilter = "managed";
    repoState.models = [
      makeModel("m1", { source: "hub" }),
      makeModel("m2", { source: "managed" }),
      makeModel("m3", { source: "managed" }),
    ];
    renderModelsTable();

    const dom = getRepoDom();
    const rows = dom.localTbody.querySelectorAll(".files-row");
    expect(rows.length).toBe(2);
  });

  test("renders empty state when no models", () => {
    repoState.models = [];
    renderModelsTable();

    const dom = getRepoDom();
    expect(dom.localTbody.innerHTML).toContain("No local models");
  });

  test("renders search empty state when query has no matches", () => {
    repoState.models = [makeModel("m1", { source: "hub" })];
    repoState.searchQuery = "nonexistent";
    renderModelsTable();

    const dom = getRepoDom();
    expect(dom.localTbody.innerHTML).toContain("nonexistent");
  });

  test("row contains model ID in name cell", () => {
    repoState.models = [makeModel("org/my-model", { source: "hub" })];
    renderModelsTable();

    const dom = getRepoDom();
    const nameSpan = dom.localTbody.querySelector(".files-name-text");
    expect(nameSpan).not.toBeNull();
    expect(nameSpan!.textContent).toBe("org/my-model");
  });

  test("row has data-id attribute", () => {
    repoState.models = [makeModel("m1", { source: "hub" })];
    renderModelsTable();

    const dom = getRepoDom();
    const row = dom.localTbody.querySelector("tr[data-id]");
    expect(row).not.toBeNull();
    expect(row!.getAttribute("data-id")).toBe("m1");
  });

  test("selected row has files-row-selected class", () => {
    repoState.models = [makeModel("m1", { source: "hub" })];
    repoState.selectedIds.add("m1");
    renderModelsTable();

    const dom = getRepoDom();
    const row = dom.localTbody.querySelector(".files-row");
    expect(row!.classList.contains("files-row-selected")).toBe(true);
  });

  test("pinned model shows pinned class on pin button", () => {
    repoState.models = [makeModel("m1", { pinned: true, source: "hub" })];
    renderModelsTable();

    const dom = getRepoDom();
    const pinBtn = dom.localTbody.querySelector(".repo-pin-btn");
    expect(pinBtn).not.toBeNull();
    expect(pinBtn!.classList.contains("pinned")).toBe(true);
  });

  test("unpinned model does not have pinned class", () => {
    repoState.models = [makeModel("m1", { pinned: false, source: "hub" })];
    renderModelsTable();

    const dom = getRepoDom();
    const pinBtn = dom.localTbody.querySelector(".repo-pin-btn");
    expect(pinBtn!.classList.contains("pinned")).toBe(false);
  });

  test("renders source badge when source is present", () => {
    repoState.models = [makeModel("m1", { source: "hub" })];
    renderModelsTable();

    const dom = getRepoDom();
    const badge = dom.localTbody.querySelector(".repo-source-badge");
    expect(badge).not.toBeNull();
    expect(badge!.textContent).toBe("hub");
  });

  test("renders quantization badge when quant_scheme is present", () => {
    repoState.models = [makeModel("m1", { quant_scheme: "Q4_K_M", source: "hub" })];
    renderModelsTable();

    const dom = getRepoDom();
    const badge = dom.localTbody.querySelector(".repo-quant-badge");
    expect(badge).not.toBeNull();
    expect(badge!.textContent).toBe("Q4_K_M");
  });

  test("client-side search filters models by ID", () => {
    repoState.models = [
      makeModel("org/llama-7b", { source: "hub" }),
      makeModel("org/mistral-7b", { architecture: "mistral", source: "hub" }),
      makeModel("org/llama-13b", { source: "hub" }),
    ];
    repoState.searchQuery = "llama";
    renderModelsTable();

    const dom = getRepoDom();
    const rows = dom.localTbody.querySelectorAll(".files-row");
    expect(rows.length).toBe(2);
  });

  test("client-side search filters by architecture", () => {
    repoState.models = [
      makeModel("m1", { architecture: "llama", source: "hub" }),
      makeModel("m2", { architecture: "mistral", source: "hub" }),
    ];
    repoState.searchQuery = "mistral";
    renderModelsTable();

    const dom = getRepoDom();
    const rows = dom.localTbody.querySelectorAll(".files-row");
    expect(rows.length).toBe(1);
  });

  test("sorts by name ascending", () => {
    repoState.sortBy = "name";
    repoState.sortDir = "asc";
    repoState.models = [
      makeModel("z-model", { source: "hub" }),
      makeModel("a-model", { source: "hub" }),
      makeModel("m-model", { source: "hub" }),
    ];
    renderModelsTable();

    const dom = getRepoDom();
    const rows = dom.localTbody.querySelectorAll("tr[data-id]");
    expect(rows[0]!.getAttribute("data-id")).toBe("a-model");
    expect(rows[2]!.getAttribute("data-id")).toBe("z-model");
  });

  test("sorts by size descending", () => {
    repoState.sortBy = "size";
    repoState.sortDir = "desc";
    repoState.models = [
      makeModel("small", { size_bytes: 100, source: "hub" }),
      makeModel("large", { size_bytes: 10000, source: "hub" }),
      makeModel("medium", { size_bytes: 5000, source: "hub" }),
    ];
    renderModelsTable();

    const dom = getRepoDom();
    const rows = dom.localTbody.querySelectorAll("tr[data-id]");
    expect(rows[0]!.getAttribute("data-id")).toBe("large");
    expect(rows[2]!.getAttribute("data-id")).toBe("small");
  });

  test("updates count display", () => {
    repoState.models = [makeModel("m1", { source: "hub" }), makeModel("m2", { source: "hub" })];
    renderModelsTable();

    const dom = getRepoDom();
    expect(dom.count.textContent).toBe("2 models");
  });

  test("singular count for 1 model", () => {
    repoState.models = [makeModel("m1", { source: "hub" })];
    renderModelsTable();

    const dom = getRepoDom();
    expect(dom.count.textContent).toBe("1 model");
  });

  test("sorts by date ascending", () => {
    repoState.sortBy = "date";
    repoState.sortDir = "asc";
    repoState.models = [
      makeModel("newest", { mtime: 1700000300, source: "hub" }),
      makeModel("oldest", { mtime: 1700000100, source: "hub" }),
      makeModel("middle", { mtime: 1700000200, source: "hub" }),
    ];
    renderModelsTable();

    const dom = getRepoDom();
    const rows = dom.localTbody.querySelectorAll("tr[data-id]");
    expect(rows[0]!.getAttribute("data-id")).toBe("oldest");
    expect(rows[1]!.getAttribute("data-id")).toBe("middle");
    expect(rows[2]!.getAttribute("data-id")).toBe("newest");
  });

  test("model without quant_scheme renders dash", () => {
    repoState.models = [makeModel("m1", { quant_scheme: "", source: "hub" })];
    renderModelsTable();

    const dom = getRepoDom();
    const cells = dom.localTbody.querySelectorAll(".files-row td");
    const quantCell = cells[3] as HTMLElement;
    expect(quantCell.textContent).toBe("—");
  });

  test("model with mtime=0 renders dash for date", () => {
    repoState.models = [makeModel("m1", { mtime: 0, source: "hub" })];
    renderModelsTable();

    const dom = getRepoDom();
    const dateCell = dom.localTbody.querySelector(".files-cell-date") as HTMLElement;
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

  test("renders static Downloading label instead of button when downloading", () => {
    repoState.searchResults = [makeSearchResult("org/model")];
    repoState.activeDownloads.set("org/model", {
      modelId: "org/model", current: 50, total: 100, label: "50%", status: "downloading",
      abort: new AbortController(),
    });
    renderDiscoverResults();

    const dom = getRepoDom();
    const label = dom.discoverResults.querySelector(".repo-downloading-label");
    expect(label).not.toBeNull();
    expect(label!.textContent).toContain("Downloading");
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
    expect(html).toContain("50.0K");
    expect(html).toContain("1.2K");
    expect(html).toContain("7.0B");
    expect(html).toContain("2025-01-15");
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

  test("result with params_total=0 does not render params span", () => {
    repoState.discoverSize = "any";
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

  test("size filter includes models with no size info at all", () => {
    repoState.discoverSize = "1";
    repoState.searchResults = [
      makeSearchResult("known-small", 500_000_000),
      makeSearchResult("no-size-info", 0),
    ];
    renderDiscoverResults();

    const dom = getRepoDom();
    const items = dom.discoverResults.querySelectorAll(".repo-discover-item");
    expect(items.length).toBe(2);
  });

  test("size filter estimates params from model name when params_total=0", () => {
    repoState.discoverSize = "1";
    repoState.searchResults = [
      makeSearchResult("Qwen/Qwen3-0.6B", 0),
      makeSearchResult("Qwen/Qwen3-4B", 0),
    ];
    renderDiscoverResults();

    const dom = getRepoDom();
    const items = dom.discoverResults.querySelectorAll(".repo-discover-item");
    expect(items.length).toBe(1);
  });

  test("size filter uses params_total over name estimate when available", () => {
    repoState.discoverSize = "1";
    repoState.searchResults = [
      makeSearchResult("some-model-8B", 500_000_000),
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
    expect(items.length).toBe(2);
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

  test("shows Delete button for cached model instead of Download", () => {
    repoState.models = [makeModel("org/cached-model")];
    repoState.searchResults = [
      makeSearchResult("org/cached-model"),
      makeSearchResult("org/not-cached"),
    ];
    renderDiscoverResults();

    const dom = getRepoDom();
    const items = dom.discoverResults.querySelectorAll(".repo-discover-item");
    expect(items.length).toBe(2);

    const cachedBtn = items[0].querySelector("[data-action]") as HTMLElement;
    expect(cachedBtn.dataset["action"]).toBe("delete");
    expect(cachedBtn.textContent).toContain("Delete");

    const newBtn = items[1].querySelector("[data-action]") as HTMLElement;
    expect(newBtn.dataset["action"]).toBe("download");
    expect(newBtn.textContent).toContain("Download");
  });

  test("downloading model shows label even if cached", () => {
    repoState.models = [makeModel("org/model")];
    repoState.activeDownloads.set("org/model", {
      modelId: "org/model", current: 0, total: 100,
      label: "Starting...", status: "downloading",
      abort: new AbortController(),
    });
    repoState.searchResults = [makeSearchResult("org/model")];
    renderDiscoverResults();

    const dom = getRepoDom();
    const item = dom.discoverResults.querySelector(".repo-discover-item")!;
    expect(item.querySelector(".repo-downloading-label")).toBeTruthy();
    expect(item.querySelector("[data-action='delete']")).toBeNull();
  });
});

// ── renderDownloads ─────────────────────────────────────────────────────────

describe("renderDownloads", () => {
  test("hides strip when no active downloads", () => {
    repoState.activeDownloads.clear();
    renderDownloads();

    const dom = getRepoDom();
    expect(dom.downloads.classList.contains("hidden")).toBe(true);
    expect(dom.downloads.innerHTML).toBe("");
  });

  test("shows strip with download row", () => {
    repoState.activeDownloads.set("org/model", {
      modelId: "org/model", current: 500_000_000, total: 2_000_000_000,
      label: "Downloading", status: "downloading", abort: new AbortController(),
    });
    renderDownloads();

    const dom = getRepoDom();
    expect(dom.downloads.classList.contains("hidden")).toBe(false);
    const row = dom.downloads.querySelector(".repo-dl-row");
    expect(row).not.toBeNull();
  });

  test("row contains model ID, progress bar, byte counter, and cancel button", () => {
    repoState.activeDownloads.set("org/model-7b", {
      modelId: "org/model-7b", current: 1_073_741_824, total: 4_294_967_296,
      label: "dl", status: "downloading", abort: new AbortController(),
    });
    renderDownloads();

    const dom = getRepoDom();
    const row = dom.downloads.querySelector(".repo-dl-row")!;

    expect(row.querySelector(".repo-dl-name")!.textContent).toBe("org/model-7b");
    const bar = row.querySelector<HTMLElement>(".repo-dl-bar")!;
    expect(bar.style.width).toBe("25%");
    const bytes = row.querySelector(".repo-dl-bytes")!;
    expect(bytes.textContent).toContain("1.0 GB");
    expect(bytes.textContent).toContain("4.0 GB");
    const cancel = row.querySelector("[data-action='cancel']");
    expect(cancel).not.toBeNull();
    expect((cancel as HTMLElement).dataset["downloadId"]).toBe("org/model-7b");
  });

  test("shows multiple rows for concurrent downloads", () => {
    repoState.activeDownloads.set("model-a", {
      modelId: "model-a", current: 0, total: 100, label: "", status: "downloading",
      abort: new AbortController(),
    });
    repoState.activeDownloads.set("model-b", {
      modelId: "model-b", current: 0, total: 200, label: "", status: "downloading",
      abort: new AbortController(),
    });
    renderDownloads();

    const dom = getRepoDom();
    const rows = dom.downloads.querySelectorAll(".repo-dl-row");
    expect(rows.length).toBe(2);
  });

  test("shows label when total is 0 and current is 0", () => {
    repoState.activeDownloads.set("org/model", {
      modelId: "org/model", current: 0, total: 0, label: "Starting...",
      status: "downloading", abort: new AbortController(),
    });
    renderDownloads();

    const dom = getRepoDom();
    const bytes = dom.downloads.querySelector(".repo-dl-bytes")!;
    expect(bytes.textContent).toBe("Starting...");
  });
});

// ── updateDownloadProgress ──────────────────────────────────────────────────

describe("updateDownloadProgress", () => {
  test("updates bar width and byte text without DOM rebuild", () => {
    repoState.activeDownloads.set("org/model", {
      modelId: "org/model", current: 100, total: 1000, label: "",
      status: "downloading", abort: new AbortController(),
    });
    renderDownloads();

    const dom = getRepoDom();
    const row = dom.downloads.querySelector(".repo-dl-row")!;
    const bar = row.querySelector<HTMLElement>(".repo-dl-bar")!;
    const bytes = row.querySelector<HTMLElement>(".repo-dl-bytes")!;

    expect(bar.style.width).toBe("10%");

    const dl = repoState.activeDownloads.get("org/model")!;
    dl.current = 500;
    updateDownloadProgress();

    expect(row.querySelector<HTMLElement>(".repo-dl-bar")).toBe(bar);
    expect(bar.style.width).toBe("50%");
    expect(bytes.textContent).toContain("500");
  });
});

// ── renderSortIndicators ────────────────────────────────────────────────────

describe("renderSortIndicators", () => {
  test("adds sorted class and arrow to active column", () => {
    repoState.sortBy = "size";
    repoState.sortDir = "asc";
    const dom = getRepoDom();

    const thName = document.createElement("th");
    thName.dataset["sort"] = "name";
    const thSize = document.createElement("th");
    thSize.dataset["sort"] = "size";
    dom.localThead.appendChild(thName);
    dom.localThead.appendChild(thSize);

    renderSortIndicators();

    expect(thSize.classList.contains("files-th-sorted")).toBe(true);
    expect(thName.classList.contains("files-th-sorted")).toBe(false);

    const arrow = thSize.querySelector(".sort-arrow");
    expect(arrow).not.toBeNull();
    expect(arrow!.textContent).toContain("\u25B2");
  });

  test("shows down arrow for descending sort", () => {
    repoState.sortBy = "name";
    repoState.sortDir = "desc";
    const dom = getRepoDom();

    const th = document.createElement("th");
    th.dataset["sort"] = "name";
    dom.localThead.appendChild(th);

    renderSortIndicators();

    const arrow = th.querySelector(".sort-arrow");
    expect(arrow!.textContent).toContain("\u25BC");
  });

  test("removes old arrow when re-rendered", () => {
    repoState.sortBy = "name";
    repoState.sortDir = "asc";
    const dom = getRepoDom();

    const th = document.createElement("th");
    th.dataset["sort"] = "name";
    dom.localThead.appendChild(th);

    renderSortIndicators();
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
    repoState.totalSizeBytes = 1_073_741_824;
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
  test("discover tab shows discover view and toolbar", () => {
    repoState.tab = "discover";
    syncRepoTabs();
    const dom = getRepoDom();
    expect(dom.discoverView.classList.contains("hidden")).toBe(false);
    expect(dom.discoverToolbar.classList.contains("hidden")).toBe(false);
    expect(dom.localView.classList.contains("hidden")).toBe(true);
    expect(dom.localToolbar.classList.contains("hidden")).toBe(true);
  });

  test("local tab shows local view and toolbar", () => {
    repoState.tab = "local";
    syncRepoTabs();
    const dom = getRepoDom();
    expect(dom.discoverView.classList.contains("hidden")).toBe(true);
    expect(dom.discoverToolbar.classList.contains("hidden")).toBe(true);
    expect(dom.localView.classList.contains("hidden")).toBe(false);
    expect(dom.localToolbar.classList.contains("hidden")).toBe(false);
  });
});

// ── syncSourceToggle ────────────────────────────────────────────────────────

describe("syncSourceToggle", () => {
  test("all button active when filter is all", () => {
    repoState.localSourceFilter = "all";
    syncSourceToggle();
    const dom = getRepoDom();
    expect(dom.sourceAll.classList.contains("active")).toBe(true);
    expect(dom.sourceHub.classList.contains("active")).toBe(false);
    expect(dom.sourceManaged.classList.contains("active")).toBe(false);
  });

  test("hub button active when filter is hub", () => {
    repoState.localSourceFilter = "hub";
    syncSourceToggle();
    const dom = getRepoDom();
    expect(dom.sourceAll.classList.contains("active")).toBe(false);
    expect(dom.sourceHub.classList.contains("active")).toBe(true);
    expect(dom.sourceManaged.classList.contains("active")).toBe(false);
  });

  test("managed button active when filter is managed", () => {
    repoState.localSourceFilter = "managed";
    syncSourceToggle();
    const dom = getRepoDom();
    expect(dom.sourceAll.classList.contains("active")).toBe(false);
    expect(dom.sourceHub.classList.contains("active")).toBe(false);
    expect(dom.sourceManaged.classList.contains("active")).toBe(true);
  });
});
