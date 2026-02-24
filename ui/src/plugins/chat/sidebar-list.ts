import { getChatDom } from "./dom.ts";
import { chatState } from "./state.ts";
import { api, notifications, observe } from "./deps.ts";
import { renderSidebarItem, renderSectionLabel } from "../../render/sidebar.ts";
import { renderEmptyState } from "../../render/common.ts";
import { el, isPinned, isArchived } from "../../render/helpers.ts";
import { CHEVRON_DOWN_ICON, CHEVRON_RIGHT_ICON } from "../../icons.ts";
import type { Conversation } from "../../types.ts";

const COLLAPSED_LIMIT = 3;

export async function loadSessions(): Promise<void> {
  if (chatState.pagination.isLoading || !chatState.pagination.hasMore) return;
  chatState.pagination.isLoading = true;

  const result = await api.listConversations({
    offset: chatState.pagination.offset,
    limit: 100,
  });
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

  // Filter out archived, then apply search
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

  const isGen = (id: string) =>
    (id === chatState.activeSessionId && chatState.isGenerating) ||
    chatState.backgroundStreamSessions.has(id);

  renderGroupedList(dom, filtered, isGen);
}

// ---------------------------------------------------------------------------
// Grouped list — always group by project with expand/collapse per group.
// ---------------------------------------------------------------------------

function projectKey(s: Conversation): string {
  return s.project_id ?? "__default__";
}

function persistCollapsed(): void {
  const arr = [...chatState.collapsedGroups];
  if (arr.length > 0) {
    localStorage.setItem("talu-collapsed-groups", JSON.stringify(arr));
  } else {
    localStorage.removeItem("talu-collapsed-groups");
  }
}

function renderGroupedList(
  dom: ReturnType<typeof getChatDom>,
  filtered: Conversation[],
  isGen: (id: string) => boolean,
): void {
  if (filtered.length === 0) {
    dom.sidebarList.insertBefore(renderEmptyState("No conversations"), dom.sidebarSentinel);
    return;
  }

  // Group by project, discovering groups from data.
  const groups = new Map<string, Conversation[]>();
  for (const s of filtered) {
    const key = projectKey(s);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(s);
  }

  const multiGroup = groups.size > 1;

  for (const [pValue, sessions] of groups) {
    if (sessions.length === 0) continue;

    const pinned = sessions.filter(isPinned);
    const unpinned = sessions.filter((s) => !isPinned(s));
    const isOpen = !chatState.collapsedGroups.has(pValue);
    const isFullyExpanded = chatState.expandedGroups.has(pValue);
    const canShowMore = multiGroup && unpinned.length > COLLAPSED_LIMIT;

    // Group header: [collapse toggle] name ... [+N / less toggle]
    const displayName = pValue === "__default__" ? "Default" : pValue;
    const label = el("div", "sidebar-group-label");

    const toggleCollapse = () => {
      if (isOpen) {
        chatState.collapsedGroups.add(pValue);
        chatState.expandedGroups.delete(pValue);
      } else {
        chatState.collapsedGroups.delete(pValue);
      }
      persistCollapsed();
      renderSidebar();
    };

    // Collapse/expand chevron on the left.
    if (multiGroup) {
      const collapseBtn = el("button", "sidebar-group-collapse");
      collapseBtn.innerHTML = isOpen ? CHEVRON_DOWN_ICON : CHEVRON_RIGHT_ICON;
      collapseBtn.title = isOpen ? "Collapse" : "Expand";
      label.appendChild(collapseBtn);
    }

    label.appendChild(el("span", "sidebar-group-name", displayName));

    // Clicking anywhere on the header row toggles collapse.
    if (multiGroup) {
      label.style.cursor = "pointer";
      label.addEventListener("click", toggleCollapse);
    }

    // Show-more toggle on the right (only when open and has enough items).
    if (isOpen && canShowMore) {
      const hiddenCount = isFullyExpanded ? 0 : unpinned.length - COLLAPSED_LIMIT;
      const moreBtn = el("button", "sidebar-group-toggle", isFullyExpanded ? "less" : `+${hiddenCount}`);
      moreBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (isFullyExpanded) {
          chatState.expandedGroups.delete(pValue);
        } else {
          chatState.expandedGroups.add(pValue);
        }
        renderSidebar();
      });
      label.appendChild(moreBtn);
    }

    dom.sidebarList.insertBefore(label, dom.sidebarSentinel);

    // If collapsed, skip rendering items.
    if (!isOpen) continue;

    // Pinned items always shown.
    for (const session of pinned) {
      dom.sidebarList.insertBefore(
        renderSidebarItem(session, session.id === chatState.activeSessionId, isGen(session.id)),
        dom.sidebarSentinel,
      );
    }

    // Unpinned: limited when multi-group and not fully expanded.
    const showAll = !multiGroup || isFullyExpanded;
    const visibleUnpinned = showAll ? unpinned : unpinned.slice(0, COLLAPSED_LIMIT);

    for (const session of visibleUnpinned) {
      dom.sidebarList.insertBefore(
        renderSidebarItem(session, session.id === chatState.activeSessionId, isGen(session.id)),
        dom.sidebarSentinel,
      );
    }

    // Inline "Show N more" / "Show less" line below items.
    if (canShowMore) {
      const hiddenCount = unpinned.length - COLLAPSED_LIMIT;
      const showMoreLine = el("button", "sidebar-show-more",
        isFullyExpanded ? "Show less" : `Show ${hiddenCount} more`);
      showMoreLine.addEventListener("click", () => {
        if (isFullyExpanded) {
          chatState.expandedGroups.delete(pValue);
        } else {
          chatState.expandedGroups.add(pValue);
        }
        renderSidebar();
      });
      dom.sidebarList.insertBefore(showMoreLine, dom.sidebarSentinel);
    }
  }
}

export async function refreshSidebar(): Promise<void> {
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
