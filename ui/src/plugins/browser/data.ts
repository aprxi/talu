/**
 * Browser plugin data loading — conversations and tags from the API.
 */

import { renderLoadingSpinner } from "../../render/common.ts";
import type { Conversation, SearchRequest } from "../../types.ts";
import { api } from "./deps.ts";
import { bState, search } from "./state.ts";
import { getBrowserDom } from "./dom.ts";
import { renderBrowserCards, renderBrowserTags, renderBrowserPagination } from "./render.ts";

/**
 * Fetch full conversation details in batches to avoid saturating the
 * browser's per-origin connection pool (typically 6 connections).
 *
 * Only used for search results (which return slim data without tags/metadata).
 */
async function fetchConversationsBatched<T extends { id: string; metadata?: Record<string, unknown> }>(
  items: T[],
  enrichFn?: (full: Conversation, slim: T) => void,
): Promise<Conversation[]> {
  const BATCH_SIZE = 4;
  const results: Conversation[] = [];
  for (let i = 0; i < items.length; i += BATCH_SIZE) {
    const batch = items.slice(i, i + BATCH_SIZE);
    const resolved = await Promise.all(
      batch.map(async (s) => {
        const r = await api.getConversation(s.id);
        if (r.ok && r.data) {
          enrichFn?.(r.data, s);
          return r.data;
        }
        return s as unknown as Conversation;
      }),
    );
    results.push(...resolved);
  }
  return results;
}

export async function loadBrowserConversations(page?: number): Promise<void> {
  const dom = getBrowserDom();
  const isSearching = search.query.trim().length > 0 || search.tagFilters.length > 0;

  if (isSearching) {
    if (search.isLoading || !search.hasMore) return;
    search.isLoading = true;

    if (search.results.length === 0) {
      dom.cardsEl.innerHTML = "";
      dom.cardsEl.appendChild(renderLoadingSpinner());
    }

    const searchReq: SearchRequest = {
      scope: "conversations",
      limit: 20,
    };
    if (search.cursor) searchReq.cursor = search.cursor;
    if (search.query.trim()) searchReq.text = search.query.trim();
    if (search.tagFilters.length > 0) {
      searchReq.filters = { tags_any: search.tagFilters };
    }

    const gen = ++bState.loadGeneration;
    const result = await api.search(searchReq);
    search.isLoading = false;
    if (gen !== bState.loadGeneration) return; // superseded

    if (result.ok && result.data) {
      const slim = result.data.data;
      const full = await fetchConversationsBatched(slim, (conv, s) => {
        conv.search_snippet = (s as { search_snippet?: string }).search_snippet;
        if (!conv.metadata || Object.keys(conv.metadata).length === 0) {
          conv.metadata = s.metadata;
        }
      });
      if (gen !== bState.loadGeneration) return; // superseded during batching
      search.results.push(...full);
      search.cursor = result.data.cursor ?? null;
      search.hasMore = result.data.has_more;
    }
    renderBrowserCards();
  } else {
    if (bState.isLoading) return;
    bState.isLoading = true;
    const gen = ++bState.loadGeneration;

    if (page !== undefined) bState.pagination.currentPage = page;

    dom.cardsEl.innerHTML = "";
    dom.cardsEl.appendChild(renderLoadingSpinner());

    const offset = (bState.pagination.currentPage - 1) * bState.pagination.pageSize;
    const marker = bState.tab === "archived" ? "archived" : "";
    const result = await api.listConversations({
      offset,
      limit: bState.pagination.pageSize,
      marker,
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
}

export async function loadAvailableTags(): Promise<void> {
  const result = await api.search({
    scope: "conversations",
    aggregations: ["tags"],
    limit: 1,
  });

  if (result.ok && result.data?.aggregations?.tags) {
    search.availableTags = result.data.aggregations.tags;
    renderBrowserTags();
  }
}
