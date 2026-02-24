/**
 * Browser plugin event wiring — search, tabs, cards, bulk actions.
 */

import type { Disposable } from "../../kernel/types.ts";
import { pluginEvents, pluginTimers, chatService } from "./deps.ts";
import { bState, search } from "./state.ts";
import { getBrowserDom } from "./dom.ts";
import { syncBrowserTabs, renderBrowserCards, updateBrowserToolbar } from "./render.ts";
import { loadBrowserConversations } from "./data.ts";
import { handleBrowserDelete, handleBrowserExport, handleBrowserArchive, handleBrowserBulkRestore, handleCardRestore, showBrowserProjectContextMenu } from "./actions.ts";
import { filterByTag, removeTagFilter } from "./tags.ts";

export function wireEvents(): void {
  const dom = getBrowserDom();

  // Tab switching (disabled when tag filter active).
  dom.tabAll.addEventListener("click", () => {
    if (search.tagFilters.length > 0) return;
    if (bState.tab !== "all") {
      bState.tab = "all";
      bState.pagination.currentPage = 1;
      bState.selectedIds.clear();
      syncBrowserTabs();
      updateBrowserToolbar();
      loadBrowserConversations(1);
    }
  });
  dom.tabArchived.addEventListener("click", () => {
    if (search.tagFilters.length > 0) return;
    if (bState.tab !== "archived") {
      bState.tab = "archived";
      bState.pagination.currentPage = 1;
      bState.selectedIds.clear();
      syncBrowserTabs();
      updateBrowserToolbar();
      loadBrowserConversations(1);
    }
  });

  // Select All / Deselect All.
  dom.selectAllBtn.addEventListener("click", () => {
    const allSelected =
      bState.selectedIds.size === bState.conversations.length && bState.conversations.length > 0;
    if (allSelected) {
      bState.selectedIds.clear();
    } else {
      for (const c of bState.conversations) bState.selectedIds.add(c.id);
    }
    updateBrowserToolbar();
    renderBrowserCards();
  });

  // Bulk action buttons.
  dom.deleteBtn.addEventListener("click", () => handleBrowserDelete());
  dom.archiveBtn.addEventListener("click", () => handleBrowserArchive());
  dom.restoreBtn.addEventListener("click", () => handleBrowserBulkRestore());
  dom.exportBtn.addEventListener("click", () => handleBrowserExport());
  dom.cancelBtn.addEventListener("click", () => {
    bState.selectedIds.clear();
    updateBrowserToolbar();
    renderBrowserCards();
  });

  // Card clicks (delegated).
  dom.cardsEl.addEventListener("click", (e) => {
    const target = e.target as HTMLElement;

    const tagChip = target.closest<HTMLElement>(".tag-chip");
    if (tagChip?.dataset["tag"]) {
      e.stopPropagation();
      filterByTag(tagChip.dataset["tag"]);
      return;
    }

    const card = target.closest<HTMLElement>(".browser-card");
    if (!card?.dataset["id"]) return;
    const id = card.dataset["id"];

    const actionBtn = target.closest<HTMLElement>(".browser-card-action");
    if (actionBtn) {
      const action = actionBtn.dataset["action"];
      if (action === "restore") {
        handleCardRestore(id);
      } else {
        pluginEvents.emit("sessions.selected", { sessionId: id });
        chatService.selectChat(id);
      }
      return;
    }

    // Toggle selection.
    if (bState.selectedIds.has(id)) {
      bState.selectedIds.delete(id);
    } else {
      bState.selectedIds.add(id);
    }
    updateBrowserToolbar();
    renderBrowserCards();
  });

  // Tag sidebar clicks (delegated).
  dom.tagsEl.addEventListener("click", (e) => {
    if (!(e.target instanceof Element)) return;
    const btn = e.target.closest<HTMLElement>("[data-tag]");
    if (!btn) return;
    const tag = btn.dataset["tag"]!;
    const isActive = btn.classList.contains("active");
    if (isActive) {
      removeTagFilter(tag);
    } else {
      filterByTag(tag);
    }
  });

  // Search with debouncing.
  let searchDebounce: Disposable | null = null;

  dom.searchInput.addEventListener("input", () => {
    searchDebounce?.dispose();
    searchDebounce = pluginTimers.setTimeout(async () => {
      const query = dom.searchInput.value.trim();
      if (query === search.query) return;

      search.query = query;
      bState.pagination.currentPage = 1;
      await loadBrowserConversations(1);
    }, 300);
    const hasText = dom.searchInput.value.trim().length > 0;
    dom.clearBtn.classList.toggle("hidden", !hasText);
  });

  dom.clearBtn.addEventListener("click", () => {
    dom.searchInput.value = "";
    dom.clearBtn.classList.add("hidden");
    dom.searchInput.dispatchEvent(new Event("input"));
  });

  // Right-click on cards to assign project.
  dom.cardsEl.addEventListener("contextmenu", (e) => {
    const card = (e.target as HTMLElement).closest<HTMLElement>(".browser-card");
    if (!card?.dataset["id"]) return;
    e.preventDefault();
    showBrowserProjectContextMenu(card, card.dataset["id"]);
  });
}
