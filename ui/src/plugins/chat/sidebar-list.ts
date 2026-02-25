import { getChatDom } from "./dom.ts";
import { chatState, getActiveProjectId } from "./state.ts";
import { api, notifications, observe, getModelsService } from "./deps.ts";
import { setSetting, deleteSetting } from "../../kernel/system/kv-settings.ts";
import { renderSidebarItem, renderSectionLabel } from "../../render/sidebar.ts";
import { renderEmptyState } from "../../render/common.ts";
import { el, isPinned, isArchived } from "../../render/helpers.ts";
import { getCachedProjects, createApiProject, loadApiProjects } from "../../render/project-combo.ts";
import { CHEVRON_DOWN_ICON, CHEVRON_RIGHT_ICON } from "../../icons.ts";
import type { Conversation } from "../../types.ts";

/** Callback for "New Chat" in a project group. Set by sidebar-events to avoid circular imports. */
let onNewChat: ((projectId: string | null) => void) | null = null;
export function setNewChatHandler(handler: (projectId: string | null) => void): void {
  onNewChat = handler;
}

/** Callback for right-click on project group headers. Set by sidebar-events. */
let onGroupContextMenu: ((anchor: HTMLElement, projectName: string, nameSpan: HTMLElement) => void) | null = null;
export function setGroupContextMenuHandler(handler: (anchor: HTMLElement, projectName: string, nameSpan: HTMLElement) => void): void {
  onGroupContextMenu = handler;
}

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
  const json = arr.length > 0 ? JSON.stringify(arr) : null;
  // KV primary (fire-and-forget).
  if (json) {
    void setSetting("talu-collapsed-groups", json);
  } else {
    void deleteSetting("talu-collapsed-groups");
  }
  // localStorage sync cache.
  try {
    if (json) {
      localStorage.setItem("talu-collapsed-groups", json);
    } else {
      localStorage.removeItem("talu-collapsed-groups");
    }
  } catch { /* storage full or unavailable */ }
}

function renderGroupedList(
  dom: ReturnType<typeof getChatDom>,
  filtered: Conversation[],
  isGen: (id: string) => boolean,
): void {
  // Group by project, discovering groups from data.
  const groupMap = new Map<string, Conversation[]>();
  for (const s of filtered) {
    const key = projectKey(s);
    if (!groupMap.has(key)) groupMap.set(key, []);
    groupMap.get(key)!.push(s);
  }

  // Merge empty projects from API cache so they appear as groups.
  for (const project of getCachedProjects()) {
    if (!groupMap.has(project.name)) {
      groupMap.set(project.name, []);
    }
  }

  if (groupMap.size === 0) {
    dom.sidebarList.insertBefore(renderEmptyState("No conversations"), dom.sidebarSentinel);
    return;
  }

  // Stable sort: "__default__" first, then alphabetical.
  const sortedKeys = [...groupMap.keys()].sort((a, b) => {
    if (a === "__default__") return -1;
    if (b === "__default__") return 1;
    return a.localeCompare(b);
  });

  const multiGroup = sortedKeys.length > 1;

  for (const pValue of sortedKeys) {
    const sessions = groupMap.get(pValue)!;

    const pinned = sessions.filter(isPinned);
    const unpinned = sessions.filter((s) => !isPinned(s));
    const isOpen = !chatState.collapsedGroups.has(pValue);
    const isFullyExpanded = chatState.expandedGroups.has(pValue);
    const canShowMore = multiGroup && unpinned.length > COLLAPSED_LIMIT;

    // Group header: [collapse toggle] name ... [+N / less toggle]
    const displayName = pValue === "__default__" ? "Default" : pValue;
    const activeProject = getActiveProjectId();
    const activeGroupKey = activeProject ?? "__default__";
    const label = el("div", "sidebar-group-label");
    if (pValue === activeGroupKey) label.classList.add("active");

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

    const nameSpan = el("span", "sidebar-group-name", displayName);
    label.appendChild(nameSpan);

    // Conversation count badge.
    label.appendChild(el("span", "sidebar-group-count", String(sessions.length)));

    // Clicking anywhere on the header row toggles collapse.
    if (multiGroup) {
      label.style.cursor = "pointer";
      label.addEventListener("click", toggleCollapse);
    }

    // Right-click on group header → context menu (skip for Default).
    if (onGroupContextMenu && pValue !== "__default__") {
      label.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        e.stopPropagation();
        onGroupContextMenu!(label, pValue, nameSpan);
      });
    }

    const controls = el("span", "sidebar-group-controls");

    // Show-more/less toggle (only when open and has enough items).
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
      controls.appendChild(moreBtn);
    }

    // "+" new chat button.
    if (onNewChat) {
      const projectForNew = pValue === "__default__" ? null : pValue;
      const addBtn = el("button", "sidebar-group-add");
      addBtn.textContent = "+";
      addBtn.title = "New chat in this project";
      addBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        onNewChat!(projectForNew);
      });
      controls.appendChild(addBtn);
    }

    label.appendChild(controls);

    dom.sidebarList.insertBefore(label, dom.sidebarSentinel);

    // If collapsed, skip rendering items.
    if (!isOpen) continue;

    // Draft item for this group (if any).
    const draftGroupKey = chatState.draftSession?.projectId ?? "__default__";
    const hasDraft = chatState.draftSession && draftGroupKey === pValue;
    const insertDraft = () => {
      const now = Math.floor(Date.now() / 1000);
      const draftItem = renderSidebarItem({
        id: "__draft__",
        object: "session",
        created_at: now,
        updated_at: now,
        model: getModelsService()?.getActiveModel() ?? "",
        title: "New Chat",
        marker: chatState.draftSession!.pinned ? "pinned" : "",
        group_id: null,
        parent_session_id: null,
        source_doc_id: null,
        project_id: chatState.draftSession!.projectId,
        metadata: {},
      }, true, false);
      draftItem.classList.add("draft");
      dom.sidebarList.insertBefore(draftItem, dom.sidebarSentinel);
    };

    // Pinned items + pinned draft first.
    if (hasDraft && chatState.draftSession!.pinned) insertDraft();
    for (const session of pinned) {
      dom.sidebarList.insertBefore(
        renderSidebarItem(session, session.id === chatState.activeSessionId, isGen(session.id)),
        dom.sidebarSentinel,
      );
    }

    // Unpinned draft, then unpinned items.
    if (hasDraft && !chatState.draftSession!.pinned) insertDraft();

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

  // Reload both sessions and project cache in parallel so empty projects persist.
  const [result] = await Promise.all([
    api.listConversations({ offset: 0, limit: 100 }),
    loadApiProjects(),
  ]);
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

/** Show an inline input at the top of the sidebar list for creating a new project. */
export function showNewProjectInput(): void {
  const dom = getChatDom();
  const existing = dom.sidebarList.querySelector(".sidebar-new-project-input");
  if (existing) { (existing as HTMLInputElement).focus(); return; }

  const input = el("input", "sidebar-new-project-input");
  input.type = "text";
  input.placeholder = "Project name\u2026";
  dom.sidebarList.insertBefore(input, dom.sidebarList.firstChild);
  input.focus();

  let submitted = false;
  const submit = () => {
    if (submitted) return;
    submitted = true;
    const name = input.value.trim();
    input.remove();
    if (name) {
      void createApiProject(name).then(() => renderSidebar());
    }
  };

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); submit(); }
    else if (e.key === "Escape") { e.preventDefault(); input.remove(); }
  });
  input.addEventListener("blur", () => submit());
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
