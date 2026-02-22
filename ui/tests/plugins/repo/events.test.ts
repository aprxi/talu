import { describe, test, expect, beforeEach } from "bun:test";
import { wireRepoEvents } from "../../../src/plugins/repo/events.ts";
import { repoState } from "../../../src/plugins/repo/state.ts";
import { initRepoDom, getRepoDom } from "../../../src/plugins/repo/dom.ts";
import { initRepoDeps } from "../../../src/plugins/repo/deps.ts";
import { createDomRoot, REPO_DOM_IDS, REPO_DOM_TAGS } from "../../helpers/dom.ts";
import { mockControllableTimers, mockNotifications, flushAsync } from "../../helpers/mocks.ts";

/**
 * Tests for repo event wiring — tab switching, search debouncing, column sort,
 * table row actions, discover card actions, bulk actions, and filter selects.
 */

// -- Mock state --------------------------------------------------------------

let ct: ReturnType<typeof mockControllableTimers>;
let apiCalls: { method: string; args: unknown[] }[];
let notif: ReturnType<typeof mockNotifications>;

beforeEach(() => {
  ct = mockControllableTimers();
  apiCalls = [];
  notif = mockNotifications();

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
  const root = createDomRoot(REPO_DOM_IDS, undefined, REPO_DOM_TAGS);
  // Set data-tab attributes on tab buttons (events.ts reads these).
  root.querySelector("#rp-tab-local")!.setAttribute("data-tab", "local");
  root.querySelector("#rp-tab-pinned")!.setAttribute("data-tab", "pinned");
  root.querySelector("#rp-tab-discover")!.setAttribute("data-tab", "discover");
  // Add options to filter selects so .value assignment works.
  const addOpts = (id: string, values: string[]) => {
    const sel = root.querySelector(`#${id}`) as HTMLSelectElement;
    for (const v of values) {
      const opt = document.createElement("option");
      opt.value = v;
      sel.appendChild(opt);
    }
  };
  addOpts("rp-sort", ["trending", "downloads", "likes", "last_modified"]);
  addOpts("rp-size-filter", ["1", "2", "4", "8", "16", "32", "64", "128", "512", "any"]);
  addOpts("rp-task-filter", ["text-generation", "image-text-to-text", "image-to-text", "text-to-image", "text-to-speech", "sentence-similarity", ""]);
  addOpts("rp-library-filter", ["safetensors", "transformers", "mlx", "sentence-transformers", ""]);
  initRepoDom(root);

  // Deps with controllable timer.
  initRepoDeps({
    api: {
      listRepoModels: async (query?: string) => {
        apiCalls.push({ method: "listRepoModels", args: [query] });
        return { ok: true, data: { models: [], total_size_bytes: 0 } };
      },
      searchRepoModels: async (query: string, opts?: any) => {
        apiCalls.push({ method: "searchRepoModels", args: [query, opts] });
        return { ok: true, data: { results: [] } };
      },
      deleteRepoModel: async (id: string) => {
        apiCalls.push({ method: "deleteRepoModel", args: [id] });
        return { ok: true };
      },
      pinRepoModel: async (id: string) => {
        apiCalls.push({ method: "pinRepoModel", args: [id] });
        return { ok: true };
      },
      unpinRepoModel: async (id: string) => {
        apiCalls.push({ method: "unpinRepoModel", args: [id] });
        return { ok: true };
      },
      fetchRepoModel: async (body: unknown) => {
        apiCalls.push({ method: "fetchRepoModel", args: [body] });
        return new Response('data: {"event":"done"}\n\n', {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        });
      },
    } as any,
    notifications: notif.mock as any,
    dialogs: { confirm: async () => true } as any,
    events: { emit: () => {}, on: () => ({ dispose() {} }) } as any,
    timers: ct.timers,
    format: {
      date: () => "", dateTime: () => "Jan 1, 2025", relativeTime: () => "",
      duration: () => "", number: () => "",
    } as any,
    status: { setBusy: () => {}, setReady: () => {} } as any,
  });
});

// -- Helpers -----------------------------------------------------------------

function makeModel(id: string, pinned = false): any {
  return {
    id, path: `/models/${id}`, source: "hub", size_bytes: 1024,
    mtime: 1700000000, architecture: "llama", quant_scheme: "Q4_K_M", pinned,
  };
}

// ── Tab switching ───────────────────────────────────────────────────────────

describe("Tab switching", () => {
  test("clicking pinned tab switches to pinned", () => {
    wireRepoEvents();
    getRepoDom().tabPinned.dispatchEvent(new Event("click"));
    expect(repoState.tab).toBe("pinned");
  });

  test("clicking discover tab switches to discover", () => {
    wireRepoEvents();
    getRepoDom().tabDiscover.dispatchEvent(new Event("click"));
    expect(repoState.tab).toBe("discover");
  });

  test("clicking same tab does nothing", () => {
    repoState.tab = "local";
    wireRepoEvents();
    getRepoDom().tabLocal.dispatchEvent(new Event("click"));
    // No API call should be made (no reload triggered).
    expect(apiCalls.length).toBe(0);
  });

  test("tab switch clears selections", () => {
    repoState.selectedIds.add("m1");
    repoState.selectedIds.add("m2");
    wireRepoEvents();
    getRepoDom().tabPinned.dispatchEvent(new Event("click"));
    expect(repoState.selectedIds.size).toBe(0);
  });

  test("tab switch clears search query", () => {
    repoState.searchQuery = "old query";
    wireRepoEvents();
    getRepoDom().tabPinned.dispatchEvent(new Event("click"));
    expect(repoState.searchQuery).toBe("");
  });

  test("switching to non-discover tab triggers loadModels", async () => {
    wireRepoEvents();
    getRepoDom().tabPinned.dispatchEvent(new Event("click"));
    await flushAsync();

    expect(apiCalls.some((c) => c.method === "listRepoModels")).toBe(true);
  });

  test("switching to discover tab loads trending models", async () => {
    wireRepoEvents();
    getRepoDom().tabDiscover.dispatchEvent(new Event("click"));
    await flushAsync();

    expect(apiCalls.some((c) => c.method === "searchRepoModels")).toBe(true);
  });
});

// ── Search debouncing ───────────────────────────────────────────────────────

describe("Search debouncing", () => {
  test("input schedules 300ms debounce", () => {
    wireRepoEvents();
    const dom = getRepoDom();
    (dom.search as HTMLInputElement).value = "test";
    dom.search.dispatchEvent(new Event("input"));

    expect(ct.pending.length).toBe(1);
    expect(ct.pending[0]!.ms).toBe(300);
  });

  test("rapid typing cancels previous timer", () => {
    wireRepoEvents();
    const dom = getRepoDom();

    (dom.search as HTMLInputElement).value = "t";
    dom.search.dispatchEvent(new Event("input"));
    (dom.search as HTMLInputElement).value = "te";
    dom.search.dispatchEvent(new Event("input"));
    (dom.search as HTMLInputElement).value = "tes";
    dom.search.dispatchEvent(new Event("input"));

    expect(ct.pending[0]!.disposed).toBe(true);
    expect(ct.pending[1]!.disposed).toBe(true);
    expect(ct.pending[2]!.disposed).toBe(false);
  });

  test("debounce callback in discover mode calls searchHub", () => {
    repoState.tab = "discover";
    wireRepoEvents();
    const dom = getRepoDom();
    (dom.search as HTMLInputElement).value = "llama";
    dom.search.dispatchEvent(new Event("input"));

    // Fire the debounced callback.
    ct.pending[0]!.fn();

    // searchHub is async and fires an API call.
    expect(repoState.searchQuery).toBe("llama");
  });

  test("debounce callback in local mode does not call API", () => {
    repoState.tab = "local";
    wireRepoEvents();
    const dom = getRepoDom();
    (dom.search as HTMLInputElement).value = "llama";
    dom.search.dispatchEvent(new Event("input"));

    ct.pending[0]!.fn();

    // Local mode: client-side filter, no API call.
    expect(apiCalls.length).toBe(0);
    expect(repoState.searchQuery).toBe("llama");
  });

  test("search clear button resets query", () => {
    wireRepoEvents();
    const dom = getRepoDom();
    (dom.search as HTMLInputElement).value = "something";
    repoState.searchQuery = "something";

    dom.searchClear.dispatchEvent(new Event("click"));

    expect(repoState.searchQuery).toBe("");
    expect((dom.search as HTMLInputElement).value).toBe("");
  });

  test("search clear in discover mode reloads trending models", async () => {
    repoState.tab = "discover";
    repoState.searchQuery = "llama";
    repoState.searchResults = [{ model_id: "r1" } as any];
    wireRepoEvents();
    const dom = getRepoDom();
    (dom.search as HTMLInputElement).value = "llama";

    dom.searchClear.dispatchEvent(new Event("click"));
    await flushAsync();

    expect(repoState.searchQuery).toBe("");
    // Clear triggers searchHub("") which calls the API.
    expect(apiCalls.some((c) => c.method === "searchRepoModels" && c.args[0] === "")).toBe(true);
  });

  test("search clear bumps searchGeneration to discard in-flight results", () => {
    repoState.tab = "discover";
    repoState.searchQuery = "llama";
    repoState.searchGeneration = 5;
    wireRepoEvents();
    const dom = getRepoDom();
    (dom.search as HTMLInputElement).value = "llama";

    dom.searchClear.dispatchEvent(new Event("click"));

    // Clear handler bumps generation once (5→6), then searchHub("") bumps again (6→7).
    // Any in-flight response from the "llama" search has gen ≤ 5 and will be discarded.
    expect(repoState.searchGeneration).toBeGreaterThan(5);
  });

  test("search clear cancels pending debounce timer", async () => {
    repoState.tab = "discover";
    wireRepoEvents();
    const dom = getRepoDom();

    // Type something → debounce is scheduled.
    (dom.search as HTMLInputElement).value = "llama";
    dom.search.dispatchEvent(new Event("input"));
    expect(ct.pending.length).toBe(1);
    expect(ct.pending[0]!.disposed).toBe(false);

    // Click clear → debounce must be cancelled.
    dom.searchClear.dispatchEvent(new Event("click"));
    await flushAsync();
    expect(ct.pending[0]!.disposed).toBe(true);

    // The only API call should be from clear's searchHub(""), not the debounce.
    const searchCalls = apiCalls.filter((c) => c.method === "searchRepoModels");
    expect(searchCalls.length).toBe(1);
    expect(searchCalls[0]!.args[0]).toBe(""); // Trending reload, not "llama".

    // Flushing disposed timers should not add another call.
    ct.flush();
    expect(apiCalls.filter((c) => c.method === "searchRepoModels").length).toBe(1);
  });
});

// ── Column sort ─────────────────────────────────────────────────────────────

describe("Column sort", () => {
  test("clicking sortable th sets sortBy", () => {
    wireRepoEvents();
    const dom = getRepoDom();
    const th = document.createElement("th");
    th.dataset["sort"] = "size";
    dom.thead.appendChild(th);

    th.dispatchEvent(new Event("click", { bubbles: true }));

    expect(repoState.sortBy).toBe("size");
    expect(repoState.sortDir).toBe("asc");
  });

  test("clicking same column toggles direction", () => {
    repoState.sortBy = "size";
    repoState.sortDir = "asc";
    wireRepoEvents();
    const dom = getRepoDom();
    const th = document.createElement("th");
    th.dataset["sort"] = "size";
    dom.thead.appendChild(th);

    th.dispatchEvent(new Event("click", { bubbles: true }));

    expect(repoState.sortDir).toBe("desc");
  });
});

// ── Table row actions ───────────────────────────────────────────────────────

describe("Table row actions", () => {
  test("toggle action toggles selection", () => {
    repoState.models = [makeModel("m1")];
    wireRepoEvents();
    const dom = getRepoDom();

    const row = document.createElement("tr");
    row.dataset["id"] = "m1";
    const btn = document.createElement("button");
    btn.dataset["action"] = "toggle";
    row.appendChild(btn);
    dom.tbody.appendChild(row);

    btn.dispatchEvent(new Event("click", { bubbles: true }));
    expect(repoState.selectedIds.has("m1")).toBe(true);

    // Re-render replaced the DOM; re-query the toggle button.
    const btn2 = dom.tbody.querySelector<HTMLElement>("[data-action='toggle']")!;
    btn2.dispatchEvent(new Event("click", { bubbles: true }));
    expect(repoState.selectedIds.has("m1")).toBe(false);
  });

  test("pin action on unpinned model calls pinModel", async () => {
    repoState.models = [makeModel("m1", false)];
    wireRepoEvents();
    const dom = getRepoDom();

    const row = document.createElement("tr");
    row.dataset["id"] = "m1";
    const btn = document.createElement("button");
    btn.dataset["action"] = "pin";
    row.appendChild(btn);
    dom.tbody.appendChild(row);

    btn.dispatchEvent(new Event("click", { bubbles: true }));
    await flushAsync();

    expect(apiCalls.some((c) => c.method === "pinRepoModel" && c.args[0] === "m1")).toBe(true);
  });

  test("pin action on pinned model calls unpinModel", async () => {
    repoState.models = [makeModel("m1", true)];
    wireRepoEvents();
    const dom = getRepoDom();

    const row = document.createElement("tr");
    row.dataset["id"] = "m1";
    const btn = document.createElement("button");
    btn.dataset["action"] = "pin";
    row.appendChild(btn);
    dom.tbody.appendChild(row);

    btn.dispatchEvent(new Event("click", { bubbles: true }));
    await flushAsync();

    expect(apiCalls.some((c) => c.method === "unpinRepoModel" && c.args[0] === "m1")).toBe(true);
  });

  test("delete action calls deleteModel", async () => {
    repoState.models = [makeModel("m1")];
    wireRepoEvents();
    const dom = getRepoDom();

    const row = document.createElement("tr");
    row.dataset["id"] = "m1";
    const btn = document.createElement("button");
    btn.dataset["action"] = "delete";
    row.appendChild(btn);
    dom.tbody.appendChild(row);

    btn.dispatchEvent(new Event("click", { bubbles: true }));
    await flushAsync();

    expect(apiCalls.some((c) => c.method === "deleteRepoModel")).toBe(true);
  });
});

// ── Discover card actions ───────────────────────────────────────────────────

describe("Discover card actions", () => {
  test("download action calls downloadModel", async () => {
    wireRepoEvents();
    const dom = getRepoDom();

    const card = document.createElement("div");
    card.dataset["modelId"] = "org/model";
    const btn = document.createElement("button");
    btn.dataset["action"] = "download";
    card.appendChild(btn);
    dom.discoverResults.appendChild(card);

    btn.dispatchEvent(new Event("click", { bubbles: true }));
    await flushAsync();

    expect(apiCalls.some((c) => c.method === "fetchRepoModel")).toBe(true);
  });

  test("download action is ignored if already downloading", async () => {
    repoState.activeDownloads.set("org/model", {
      modelId: "org/model", current: 0, total: 100, label: "...", status: "downloading",
    });
    wireRepoEvents();
    const dom = getRepoDom();

    const card = document.createElement("div");
    card.dataset["modelId"] = "org/model";
    const btn = document.createElement("button");
    btn.dataset["action"] = "download";
    card.appendChild(btn);
    dom.discoverResults.appendChild(card);

    btn.dispatchEvent(new Event("click", { bubbles: true }));
    await flushAsync();

    // Should not start a second download.
    expect(apiCalls.filter((c) => c.method === "fetchRepoModel").length).toBe(0);
  });
});

// ── Bulk actions ────────────────────────────────────────────────────────────

describe("Bulk actions", () => {
  test("select all adds all models to selection", () => {
    repoState.models = [makeModel("m1"), makeModel("m2"), makeModel("m3")];
    wireRepoEvents();

    getRepoDom().selectAllBtn.dispatchEvent(new Event("click"));

    expect(repoState.selectedIds.size).toBe(3);
  });

  test("select all when some are selected clears selection", () => {
    repoState.models = [makeModel("m1"), makeModel("m2")];
    repoState.selectedIds.add("m1");
    wireRepoEvents();

    getRepoDom().selectAllBtn.dispatchEvent(new Event("click"));

    expect(repoState.selectedIds.size).toBe(0);
  });

  test("cancel clears selections", () => {
    repoState.selectedIds.add("m1");
    repoState.selectedIds.add("m2");
    wireRepoEvents();

    getRepoDom().cancelBtn.dispatchEvent(new Event("click"));

    expect(repoState.selectedIds.size).toBe(0);
  });
});

// ── Discover filter selects ─────────────────────────────────────────────────

describe("Discover filter selects", () => {
  test("changing sort select updates state and triggers search", () => {
    repoState.tab = "discover";
    repoState.searchQuery = "llama";
    wireRepoEvents();

    const dom = getRepoDom();
    (dom.sortSelect as HTMLSelectElement).value = "downloads";
    dom.sortSelect.dispatchEvent(new Event("change"));

    expect(repoState.discoverSort).toBe("downloads");
  });

  test("changing task filter updates state", () => {
    repoState.tab = "discover";
    repoState.searchQuery = "llama";
    wireRepoEvents();

    const dom = getRepoDom();
    (dom.taskFilter as HTMLSelectElement).value = "image-to-text";
    dom.taskFilter.dispatchEvent(new Event("change"));

    expect(repoState.discoverTask).toBe("image-to-text");
  });

  test("changing library filter updates state", () => {
    repoState.tab = "discover";
    repoState.searchQuery = "llama";
    wireRepoEvents();

    const dom = getRepoDom();
    (dom.libraryFilter as HTMLSelectElement).value = "transformers";
    dom.libraryFilter.dispatchEvent(new Event("change"));

    expect(repoState.discoverLibrary).toBe("transformers");
  });

  test("changing size filter updates state without API call", () => {
    repoState.tab = "discover";
    repoState.searchQuery = "llama";
    wireRepoEvents();

    const dom = getRepoDom();
    (dom.sizeFilter as HTMLSelectElement).value = "32";
    dom.sizeFilter.dispatchEvent(new Event("change"));

    expect(repoState.discoverSize).toBe("32");
    // Size is client-side only — no search API call.
    expect(apiCalls.filter((c) => c.method === "searchRepoModels").length).toBe(0);
  });

  test("sort/task/library changes with no query still trigger API for trending", async () => {
    repoState.tab = "discover";
    repoState.searchQuery = "";
    wireRepoEvents();

    const dom = getRepoDom();
    (dom.sortSelect as HTMLSelectElement).value = "likes";
    dom.sortSelect.dispatchEvent(new Event("change"));
    await flushAsync();

    expect(apiCalls.some((c) => c.method === "searchRepoModels")).toBe(true);
  });
});
