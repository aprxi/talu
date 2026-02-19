/**
 * Browser plugin tag filtering — add/remove/clear tag filters and toolbar indicator.
 */

import {
  CLOSE_ICON as ICON_CLEAR,
  FILTER_ICON as ICON_FILTER,
} from "../../icons.ts";
import { escapeHtml } from "../../utils/helpers.ts";
import { search, bState } from "./state.ts";
import { getBrowserDom } from "./dom.ts";
import { renderBrowserTags, renderBrowserCards, syncBrowserTabs, updateBrowserToolbar } from "./render.ts";
import { loadBrowserConversations } from "./data.ts";

export function filterByTag(tag: string): void {
  const idx = search.tagFilters.indexOf(tag);
  if (idx >= 0) {
    search.tagFilters.splice(idx, 1);
  } else {
    search.tagFilters.push(tag);
  }

  renderBrowserTags();
  bState.selectedIds.clear();
  bState.pagination.currentPage = 1;

  updateBrowserTagFilter();
  syncBrowserTabs();
  updateBrowserToolbar();
  loadBrowserConversations(1);
}

export function removeTagFilter(tag: string): void {
  const idx = search.tagFilters.indexOf(tag);
  if (idx >= 0) {
    search.tagFilters.splice(idx, 1);
  }

  bState.pagination.currentPage = 1;
  renderBrowserTags();
  updateBrowserTagFilter();
  syncBrowserTabs();
  updateBrowserToolbar();
  loadBrowserConversations(1);
}

export function clearTagFilter(): void {
  search.tagFilters = [];
  bState.pagination.currentPage = 1;

  updateBrowserTagFilter();
  renderBrowserTags();
  syncBrowserTabs();
  updateBrowserToolbar();
  loadBrowserConversations(1);
}

export function updateBrowserTagFilter(): void {
  const dom = getBrowserDom();
  let indicator = dom.toolbarEl.querySelector("#bp-tag-filter") as HTMLElement | null;

  if (search.tagFilters.length > 0) {
    if (!indicator) {
      indicator = document.createElement("div");
      indicator.id = "bp-tag-filter";
      dom.toolbarEl.appendChild(indicator);
    }
    indicator.className = "filter-indicator";
    indicator.innerHTML = `
      ${ICON_FILTER}
      <span class="filter-label">Filtering by:</span>
      <div class="filter-chips" id="bp-tag-filter-chips"></div>
      <button id="bp-clear-tag-filter" class="filter-clear-btn" title="Clear all filters">
        ${ICON_CLEAR}
        Clear all
      </button>
    `;

    const chipsContainer = indicator.querySelector("#bp-tag-filter-chips");
    if (chipsContainer) {
      for (const tag of search.tagFilters) {
        const chip = document.createElement("span");
        chip.className = "filter-chip";
        chip.innerHTML = `
          ${escapeHtml(tag)}
          <button class="filter-chip-remove" data-tag="${escapeHtml(tag)}" title="Remove this filter">
            ${ICON_CLEAR}
          </button>
        `;
        chipsContainer.appendChild(chip);
      }
    }

    const clearAllBtn = indicator.querySelector("#bp-clear-tag-filter");
    if (clearAllBtn) {
      clearAllBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        clearTagFilter();
      });
    }

    indicator.querySelectorAll(".filter-chip-remove").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const t = (btn as HTMLElement).dataset["tag"];
        if (t) removeTagFilter(t);
      });
    });
  } else if (indicator) {
    indicator.remove();
  }
}
