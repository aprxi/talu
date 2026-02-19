/**
 * Browser plugin rendering — tabs, cards, tags, and toolbar.
 */

import { renderBrowserCard } from "../../render/browser.ts";
import { renderEmptyState } from "../../render/common.ts";
import { isArchived } from "../../render/helpers.ts";
import { computePagination, renderPagination } from "../../render/pagination.ts";
import { TAG_ICON as ICON_TAG } from "../../icons.ts";
import { bState, search } from "./state.ts";
import { getBrowserDom } from "./dom.ts";
import { loadBrowserConversations } from "./data.ts";

export function syncBrowserTabs(): void {
  const dom = getBrowserDom();
  const baseClass = "browser-tab";
  const activeClass = "active";
  const activeDimClass = "active dimmed";
  const hasTagFilter = search.tagFilters.length > 0;

  if (hasTagFilter) {
    dom.tabAll.className = `${baseClass} ${activeDimClass}`;
    dom.tabArchived.className = `${baseClass} ${activeDimClass}`;
    dom.archiveBtn.classList.remove("hidden");
    dom.restoreBtn.classList.remove("hidden");
  } else {
    dom.tabAll.className = `${baseClass} ${bState.tab === "all" ? activeClass : ""}`;
    dom.tabArchived.className = `${baseClass} ${bState.tab === "archived" ? activeClass : ""}`;
    if (bState.tab === "all") {
      dom.archiveBtn.classList.remove("hidden");
      dom.restoreBtn.classList.add("hidden");
    } else {
      dom.archiveBtn.classList.add("hidden");
      dom.restoreBtn.classList.remove("hidden");
    }
  }
}

export function renderBrowserCards(): void {
  const dom = getBrowserDom();
  dom.cardsEl.innerHTML = "";
  const isSearching = search.query.trim().length > 0 || search.tagFilters.length > 0;
  const hasTagFilter = search.tagFilters.length > 0;

  const baseConversations = isSearching
    ? search.results
    : bState.conversations;

  // In non-search mode, the backend already filters by marker — no client
  // filtering needed. For tag-filter mode, sort archived to the end.
  const conversations: typeof baseConversations = hasTagFilter
    ? [...baseConversations].sort((a, b) => {
        const aArchived = isArchived(a) ? 1 : 0;
        const bArchived = isArchived(b) ? 1 : 0;
        return aArchived - bArchived;
      })
    : baseConversations;

  if (conversations.length === 0 && !search.isLoading) {
    const emptyMsg = isSearching
      ? "No matching conversations"
      : bState.tab === "archived"
        ? "No archived conversations"
        : "No conversations";
    dom.cardsEl.appendChild(renderEmptyState(emptyMsg));
    return;
  }

  if (search.query.trim().length > 0) {
    const indicator = document.createElement("div");
    indicator.className = "search-indicator";
    const strong = document.createElement("strong");
    strong.textContent = search.query;
    indicator.append("Showing results for \u201c", strong, "\u201d");
    dom.cardsEl.appendChild(indicator);
  }

  for (const chat of conversations) {
    dom.cardsEl.appendChild(
      renderBrowserCard(
        chat,
        bState.selectedIds.has(chat.id),
        search.query.trim().length > 0 ? search.query : undefined,
        search.tagFilters,
        hasTagFilter,
      ),
    );
  }

  if (isSearching && search.hasMore) {
    const loadMore = document.createElement("div");
    loadMore.className = "load-more-container";
    const btn = document.createElement("button");
    btn.id = "browser-load-more";
    btn.className = "btn btn-ghost";
    btn.textContent = "Load more";
    loadMore.appendChild(btn);
    dom.cardsEl.appendChild(loadMore);
  }
}

export function renderBrowserTags(): void {
  const dom = getBrowserDom();
  dom.tagsEl.innerHTML = "";

  if (search.availableTags.length === 0) {
    dom.tagsSection.classList.add("hidden");
    return;
  }

  dom.tagsSection.classList.remove("hidden");

  for (const tag of search.availableTags) {
    const isActive = search.tagFilters.includes(tag.name);
    const item = document.createElement("button");
    item.className = isActive ? "browser-tab active" : "browser-tab";
    item.dataset["tag"] = tag.name;

    const icon = document.createElement("span");
    icon.className = "shrink-0";
    icon.innerHTML = ICON_TAG;
    item.appendChild(icon);

    const nameSpan = document.createElement("span");
    nameSpan.className = "flex-1 truncate";
    nameSpan.textContent = tag.name;
    item.appendChild(nameSpan);

    const countSpan = document.createElement("span");
    countSpan.className = isActive
      ? "text-xs text-accent/70"
      : "text-xs text-text-subtle";
    countSpan.textContent = String(tag.count);
    item.appendChild(countSpan);

    dom.tagsEl.appendChild(item);
  }
}

export function updateBrowserToolbar(): void {
  const dom = getBrowserDom();
  const hasSelection = bState.selectedIds.size > 0;
  const hasTagFilter = search.tagFilters.length > 0;

  dom.deleteBtn.disabled = !hasSelection;
  dom.exportBtn.disabled = !hasSelection;
  dom.archiveBtn.disabled = !hasSelection;
  dom.restoreBtn.disabled = !hasSelection;

  dom.bulkActions.classList.toggle("active", hasSelection);
  dom.cancelBtn.classList.toggle("hidden", !hasSelection);

  const isSearching = search.query.trim().length > 0 || hasTagFilter;
  const visible = isSearching ? search.results : bState.conversations;
  const allSelected =
    bState.selectedIds.size === visible.length && visible.length > 0;
  dom.selectAllBtn.textContent = allSelected ? "Deselect All" : "Select All";

  dom.tabAll.classList.toggle("opacity-40", hasTagFilter);
  dom.tabAll.classList.toggle("pointer-events-none", hasTagFilter);
  dom.tabArchived.classList.toggle("opacity-40", hasTagFilter);
  dom.tabArchived.classList.toggle("pointer-events-none", hasTagFilter);
}

export function renderBrowserPagination(): void {
  const dom = getBrowserDom();
  dom.paginationEl.innerHTML = "";

  const { totalItems, pageSize, currentPage } = bState.pagination;
  if (totalItems <= pageSize) return;

  const state = computePagination(totalItems, pageSize, currentPage);
  dom.paginationEl.appendChild(
    renderPagination(state, (page) => loadBrowserConversations(page)),
  );
}
