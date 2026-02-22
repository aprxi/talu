/** Event wiring for the repo plugin. */

import type { Disposable } from "../../kernel/types.ts";
import { timers } from "./deps.ts";
import { getRepoDom } from "./dom.ts";
import { repoState } from "./state.ts";
import {
  loadModels,
  searchHub,
  deleteModel,
  pinModel,
  unpinModel,
  deleteSelectedModels,
  pinSelectedModels,
  downloadModel,
} from "./data.ts";
import {
  renderModelsTable,
  renderDiscoverResults,
  syncRepoTabs,
  updateRepoToolbar,
} from "./render.ts";

export function wireRepoEvents(): void {
  const dom = getRepoDom();

  // -----------------------------------------------------------------------
  // Tabs
  // -----------------------------------------------------------------------

  for (const btn of [dom.tabLocal, dom.tabPinned, dom.tabDiscover]) {
    btn.addEventListener("click", () => {
      const tab = btn.dataset["tab"] as typeof repoState.tab;
      if (tab === repoState.tab) return;
      repoState.tab = tab;
      repoState.selectedIds.clear();
      repoState.searchQuery = "";
      dom.search.value = "";
      dom.searchClear.classList.add("hidden");
      syncRepoTabs();
      updateRepoToolbar();
      if (tab === "discover") {
        renderDiscoverResults();
      } else {
        loadModels();
      }
    });
  }

  // -----------------------------------------------------------------------
  // Search (debounced)
  // -----------------------------------------------------------------------

  let searchDebounce: Disposable | null = null;

  dom.search.addEventListener("input", () => {
    const query = dom.search.value.trim();
    repoState.searchQuery = query;
    dom.searchClear.classList.toggle("hidden", query.length === 0);

    searchDebounce?.dispose();
    searchDebounce = timers.setTimeout(() => {
      searchDebounce = null;
      if (repoState.tab === "discover") {
        searchHub(query);
      } else {
        renderModelsTable();
      }
    }, 300);
  });

  dom.searchClear.addEventListener("click", () => {
    dom.search.value = "";
    repoState.searchQuery = "";
    dom.searchClear.classList.add("hidden");
    if (repoState.tab === "discover") {
      repoState.searchResults = [];
      renderDiscoverResults();
    } else {
      renderModelsTable();
    }
  });

  // -----------------------------------------------------------------------
  // Discover filters
  // -----------------------------------------------------------------------

  // Sort/Task/Library → server-side re-fetch.
  for (const sel of [dom.sortSelect, dom.taskFilter, dom.libraryFilter]) {
    sel.addEventListener("change", () => {
      repoState.discoverSort = dom.sortSelect.value as typeof repoState.discoverSort;
      repoState.discoverTask = dom.taskFilter.value as typeof repoState.discoverTask;
      repoState.discoverLibrary = dom.libraryFilter.value as typeof repoState.discoverLibrary;
      if (repoState.tab === "discover" && repoState.searchQuery) {
        searchHub(repoState.searchQuery);
      }
    });
  }

  // Size → client-side filter only (no re-fetch).
  dom.sizeFilter.addEventListener("change", () => {
    repoState.discoverSize = dom.sizeFilter.value as typeof repoState.discoverSize;
    if (repoState.tab === "discover") {
      renderDiscoverResults();
    }
  });

  // -----------------------------------------------------------------------
  // Column sort (delegated on thead)
  // -----------------------------------------------------------------------

  dom.thead.addEventListener("click", (e) => {
    const th = (e.target as HTMLElement).closest<HTMLElement>("[data-sort]");
    if (!th) return;
    const col = th.dataset["sort"] as typeof repoState.sortBy;
    if (col === repoState.sortBy) {
      repoState.sortDir = repoState.sortDir === "asc" ? "desc" : "asc";
    } else {
      repoState.sortBy = col;
      repoState.sortDir = "asc";
    }
    renderModelsTable();
  });

  // -----------------------------------------------------------------------
  // Table row actions (delegated on tbody)
  // -----------------------------------------------------------------------

  dom.tbody.addEventListener("click", (e) => {
    const target = e.target as HTMLElement;
    const row = target.closest<HTMLElement>("tr[data-id]");
    if (!row) return;
    const modelId = row.dataset["id"]!;

    const action = target.closest<HTMLElement>("[data-action]")?.dataset["action"];

    if (action === "toggle") {
      if (repoState.selectedIds.has(modelId)) {
        repoState.selectedIds.delete(modelId);
      } else {
        repoState.selectedIds.add(modelId);
      }
      renderModelsTable();
      updateRepoToolbar();
      return;
    }

    if (action === "pin") {
      const model = repoState.models.find((m) => m.id === modelId);
      if (model?.pinned) {
        unpinModel(modelId);
      } else {
        pinModel(modelId);
      }
      return;
    }

    if (action === "delete") {
      deleteModel(modelId);
      return;
    }
  });

  // -----------------------------------------------------------------------
  // Discover card actions (delegated)
  // -----------------------------------------------------------------------

  dom.discoverResults.addEventListener("click", (e) => {
    const target = e.target as HTMLElement;
    const action = target.closest<HTMLElement>("[data-action]")?.dataset["action"];
    if (!action) return;

    const card = target.closest<HTMLElement>("[data-model-id]");
    if (!card) return;
    const modelId = card.dataset["modelId"]!;

    if (action === "download") {
      // Prevent double-click.
      if (repoState.activeDownloads.has(modelId)) return;
      downloadModel(modelId);
    }
  });

  // -----------------------------------------------------------------------
  // Bulk actions
  // -----------------------------------------------------------------------

  dom.selectAllBtn.addEventListener("click", () => {
    if (repoState.selectedIds.size > 0) {
      repoState.selectedIds.clear();
    } else {
      for (const m of repoState.models) {
        repoState.selectedIds.add(m.id);
      }
    }
    renderModelsTable();
    updateRepoToolbar();
  });

  dom.pinAllBtn.addEventListener("click", () => {
    pinSelectedModels();
  });

  dom.deleteBtn.addEventListener("click", () => {
    deleteSelectedModels();
  });

  dom.cancelBtn.addEventListener("click", () => {
    repoState.selectedIds.clear();
    renderModelsTable();
    updateRepoToolbar();
  });
}
