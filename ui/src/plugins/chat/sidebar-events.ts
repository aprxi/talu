import { getChatDom } from "./dom.ts";
import { chatState } from "./state.ts";
import { handleTogglePin, showProjectContextMenu, showGroupContextMenu } from "./sidebar-actions.ts";
import { selectChat } from "./selection.ts";
import { startNewConversation } from "./welcome.ts";
import { renderSidebar, setNewChatHandler, setGroupContextMenuHandler, showNewProjectInput } from "./sidebar-list.ts";

export function setupSidebarEvents(): void {
  setNewChatHandler((projectId) => startNewConversation(projectId));
  setGroupContextMenuHandler((anchor, name, nameSpan) => showGroupContextMenu(anchor, name, nameSpan));
  const dom = getChatDom();

  dom.sidebarNewProject.addEventListener("click", () => showNewProjectInput());

  dom.sidebarList.addEventListener("click", (e) => {
    const target = e.target as HTMLElement;

    // Pin/unpin button
    const pinBtn = target.closest<HTMLElement>("[data-pin]");
    if (pinBtn?.dataset["pin"]) {
      handleTogglePin(pinBtn.dataset["pin"]);
      return;
    }

    // Click on sidebar item → open chat (or restore draft)
    const item = target.closest<HTMLElement>(".sidebar-item");
    if (item?.dataset["id"]) {
      if (item.dataset["id"] === "__draft__") {
        startNewConversation(chatState.draftSession?.projectId);
      } else {
        selectChat(item.dataset["id"]);
      }
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
