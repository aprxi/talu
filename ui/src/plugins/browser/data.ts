/**
 * Browser plugin data loading — conversations and tags from the API.
 */

import { renderLoadingSpinner } from "../../render/common.ts";
import type { SearchRequest } from "../../types.ts";
import { api, chatService } from "./deps.ts";
import { bState, search } from "./state.ts";
import { getBrowserDom } from "./dom.ts";
import { renderBrowserCards, renderBrowserTags } from "./render.ts";

export async function loadBrowserConversations(): Promise<void> {
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

    const result = await api.search(searchReq);
    search.isLoading = false;

    if (result.ok && result.data) {
      const slim = result.data.data;
      const full = await Promise.all(
        slim.map(async (s) => {
          const r = await api.getConversation(s.id);
          if (r.ok && r.data) {
            r.data.search_snippet = s.search_snippet;
            if (!r.data.metadata || Object.keys(r.data.metadata).length === 0) {
              r.data.metadata = s.metadata;
            }
            return r.data;
          }
          return s;
        }),
      );
      search.results.push(...full);
      search.cursor = result.data.cursor ?? null;
      search.hasMore = result.data.has_more;
    }
    renderBrowserCards();
  } else {
    dom.cardsEl.innerHTML = "";
    dom.cardsEl.appendChild(renderLoadingSpinner());
    const sessions = chatService.getSessions();
    if (sessions.length === 0) {
      await chatService.refreshSidebar();
    }
    const currentSessions = chatService.getSessions();
    const fullConversations = await Promise.all(
      currentSessions.map(async (s) => {
        const result = await api.getConversation(s.id);
        if (result.ok && result.data) {
          if (!result.data.metadata || Object.keys(result.data.metadata).length === 0) {
            result.data.metadata = s.metadata;
          }
          return result.data;
        }
        return s;
      }),
    );
    bState.conversations = fullConversations;
    renderBrowserCards();
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
