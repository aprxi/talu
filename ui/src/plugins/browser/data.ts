/**
 * Browser plugin data loading — conversations and tags from the API.
 */

import { renderLoadingSpinner } from "../../render/common.ts";
import { api } from "./deps.ts";
import { bState, search } from "./state.ts";
import { getBrowserDom } from "./dom.ts";
import { renderBrowserCards, renderBrowserTags, renderBrowserPagination, updateBrowserProjectSelector } from "./render.ts";

export async function loadBrowserConversations(page?: number): Promise<void> {
  if (bState.isLoading) return;
  bState.isLoading = true;

  const dom = getBrowserDom();
  const gen = ++bState.loadGeneration;

  if (page !== undefined) bState.pagination.currentPage = page;

  dom.cardsEl.innerHTML = "";
  dom.cardsEl.appendChild(renderLoadingSpinner());

  const offset = (bState.pagination.currentPage - 1) * bState.pagination.pageSize;
  const marker = bState.tab === "archived" ? "archived" : "";
  const searchText = search.query.trim() || undefined;
  const tagsAny = search.tagFilters.length > 0
    ? search.tagFilters.join(" ")
    : undefined;

  const result = await api.listConversations({
    offset,
    limit: bState.pagination.pageSize,
    marker: marker || undefined,
    search: searchText,
    tags_any: tagsAny,
    project_id: search.projectFilter || undefined,
  });

  bState.isLoading = false;
  if (gen !== bState.loadGeneration) return; // superseded

  if (result.ok && result.data) {
    bState.conversations = result.data.data;
    bState.pagination.totalItems = result.data.total;
  }
  renderBrowserCards();
  renderBrowserPagination();
}

export async function loadAvailableTags(): Promise<void> {
  const result = await api.search({
    scope: "sessions",
    aggregations: ["tags", "projects"],
    limit: 1,
  });

  if (result.ok && result.data?.aggregations) {
    if (result.data.aggregations.tags) {
      search.availableTags = result.data.aggregations.tags;
      renderBrowserTags();
    }
    if (result.data.aggregations.projects) {
      search.availableProjects = result.data.aggregations.projects;
      updateBrowserProjectSelector();
    }
  }
}
