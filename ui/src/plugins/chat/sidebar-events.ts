import { getChatDom } from "./dom.ts";
import { handleTogglePin } from "./sidebar-actions.ts";
import { selectChat } from "./selection.ts";
import { startNewConversation } from "./welcome.ts";

export function setupSidebarEvents(): void {
  const dom = getChatDom();

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

  dom.newConversationBtn.addEventListener("click", () => {
    startNewConversation();
  });
}
