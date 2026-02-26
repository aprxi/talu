import { getChatDom } from "./dom.ts";
import { chatState } from "./state.ts";
import { handleTogglePin, showProjectContextMenu, showGroupContextMenu } from "./sidebar-actions.ts";
import { selectChat } from "./selection.ts";
import { startNewConversation } from "./welcome.ts";
import { renderSidebar, persistCollapsed, getRenderedGroupKeys, setProjectNavigateHandler, setGroupContextMenuHandler, showNewProjectInput } from "./sidebar-list.ts";
import { SORT_RECENT_ICON, SORT_CREATED_ICON } from "../../icons.ts";

export function setupSidebarEvents(): void {
  setProjectNavigateHandler((projectId, firstChatId) => {
    if (firstChatId) {
      selectChat(firstChatId);
    } else {
      startNewConversation(projectId);
    }
  });
  setGroupContextMenuHandler((anchor, name, nameSpan) => showGroupContextMenu(anchor, name, nameSpan));
  const dom = getChatDom();

  dom.sidebarNewProject.addEventListener("click", () => showNewProjectInput());

  // Collapse-all / expand-all toggle.
  dom.sidebarCollapseAll.addEventListener("click", () => {
    const keys = getRenderedGroupKeys();
    const allCollapsed = keys.length > 0 && keys.every(k => chatState.collapsedGroups.has(k));
    if (allCollapsed) {
      chatState.collapsedGroups.clear();
    } else {
      for (const k of keys) chatState.collapsedGroups.add(k);
      chatState.expandedGroups.clear();
    }
    persistCollapsed();
    renderSidebar();
  });

  // Sort toggle: recent activity ↔ project creation time.
  dom.sidebarSort.addEventListener("click", () => {
    if (chatState.sidebarSort === "recent") {
      chatState.sidebarSort = "created";
      dom.sidebarSort.innerHTML = SORT_CREATED_ICON;
      dom.sidebarSort.title = "Sorted by creation time";
    } else {
      chatState.sidebarSort = "recent";
      dom.sidebarSort.innerHTML = SORT_RECENT_ICON;
      dom.sidebarSort.title = "Sorted by recent activity";
    }
    renderSidebar();
  });

  dom.sidebarList.addEventListener("click", (e) => {
    const target = e.target as HTMLElement;

    // Pin/unpin button
    const pinBtn = target.closest<HTMLElement>("[data-pin]");
    if (pinBtn?.dataset["pin"]) {
      handleTogglePin(pinBtn.dataset["pin"]);
      return;
    }

    // Click on sidebar item → open chat
    const item = target.closest<HTMLElement>(".sidebar-item");
    if (item?.dataset["id"]) {
      selectChat(item.dataset["id"]);
    }
  });

  // Sidebar search
  dom.sidebarSearch.addEventListener("input", () => {
    chatState.sidebarSearchQuery = dom.sidebarSearch.value;
    dom.sidebarSearchClear.classList.toggle("hidden", !dom.sidebarSearch.value);
    renderSidebar();
  });

  dom.sidebarSearchClear.addEventListener("click", () => {
    dom.sidebarSearch.value = "";
    chatState.sidebarSearchQuery = "";
    dom.sidebarSearchClear.classList.add("hidden");
    renderSidebar();
  });

  dom.sidebarList.addEventListener("contextmenu", (e) => {
    const item = (e.target as HTMLElement).closest<HTMLElement>(".sidebar-item");
    if (!item?.dataset["id"]) return;
    e.preventDefault();
    showProjectContextMenu(item, item.dataset["id"]);
  });
}
