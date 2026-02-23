import { getChatDom } from "./dom.ts";
import { chatState } from "./state.ts";
import { api, notifications, observe } from "./deps.ts";
import { renderSidebarItem, renderSectionLabel } from "../../render/sidebar.ts";
import { renderEmptyState } from "../../render/common.ts";
import { isPinned, isArchived } from "../../render/helpers.ts";

export async function loadSessions(): Promise<void> {
  if (chatState.pagination.isLoading || !chatState.pagination.hasMore) return;
  chatState.pagination.isLoading = true;

  const result = await api.listConversations({ offset: chatState.pagination.offset, limit: 100 });
  chatState.pagination.isLoading = false;

  if (!result.ok || !result.data) {
    notifications.error(result.error ?? "Failed to load conversations");
    return;
  }

  const list = result.data;
  chatState.sessions.push(...list.data);
  chatState.pagination.offset += list.data.length;
  chatState.pagination.hasMore = list.has_more;

  renderSidebar();
}

export function renderSidebar(): void {
  const dom = getChatDom();

  // Clear existing items (keep sentinel)
  while (dom.sidebarList.firstChild && dom.sidebarList.firstChild !== dom.sidebarSentinel) {
    dom.sidebarList.removeChild(dom.sidebarList.firstChild);
  }

  // Show/hide sentinel
  dom.sidebarSentinel.style.display = chatState.pagination.hasMore ? "flex" : "none";

  if (chatState.sessions.length === 0) {
    dom.sidebarList.insertBefore(
      renderEmptyState("No conversations"),
      dom.sidebarSentinel,
    );
    return;
  }

  // Filter out archived, then apply search, then split into pinned / unpinned
  const visible = chatState.sessions.filter((s) => !isArchived(s));
  const query = chatState.sidebarSearchQuery.toLowerCase();
  const filtered = query
    ? visible.filter((s) => (s.title ?? "").toLowerCase().includes(query))
    : visible;

  if (query && filtered.length === 0) {
    dom.sidebarList.insertBefore(
      renderEmptyState("No matches"),
      dom.sidebarSentinel,
    );
    return;
  }

  const pinned = filtered.filter(isPinned);
  const unpinned = filtered.filter((s) => !isPinned(s));

  const isSessionGenerating = (id: string) =>
    (id === chatState.activeSessionId && chatState.isGenerating) ||
    chatState.backgroundStreamSessions.has(id);

  if (pinned.length > 0) {
    dom.sidebarList.insertBefore(renderSectionLabel("Pinned"), dom.sidebarSentinel);
    for (const session of pinned) {
      dom.sidebarList.insertBefore(
        renderSidebarItem(session, session.id === chatState.activeSessionId, isSessionGenerating(session.id)),
        dom.sidebarSentinel,
      );
    }
  }

  if (unpinned.length > 0 && pinned.length > 0) {
    dom.sidebarList.insertBefore(renderSectionLabel("Recent"), dom.sidebarSentinel);
  }

  for (const session of unpinned) {
    dom.sidebarList.insertBefore(
      renderSidebarItem(session, session.id === chatState.activeSessionId, isSessionGenerating(session.id)),
      dom.sidebarSentinel,
    );
  }
}

export async function refreshSidebar(): Promise<void> {
  // Fetch fresh data before replacing — avoids a flash of empty sidebar
  // while the API call is in flight.
  chatState.pagination.isLoading = true;
  const result = await api.listConversations({ offset: 0, limit: 100 });
  chatState.pagination.isLoading = false;

  if (!result.ok || !result.data) {
    notifications.error(result.error ?? "Failed to load conversations");
    return;
  }

  chatState.sessions = result.data.data;
  chatState.pagination.offset = result.data.data.length;
  chatState.pagination.hasMore = result.data.has_more;
  renderSidebar();
}

export function setupInfiniteScroll(): void {
  const dom = getChatDom();
  observe.intersection(
    dom.sidebarSentinel,
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          loadSessions();
        }
      }
    },
    { root: dom.sidebarList.parentElement, threshold: 0.1 },
  );
}
