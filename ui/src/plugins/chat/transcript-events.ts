import { getChatDom } from "./dom.ts";
import { chatState } from "./state.ts";
import { handleCopyUserMessage, handleCopyAssistantMessage, handleEditUserMessage } from "./message-actions.ts";
import { handleRerunFromMessage } from "./rerun.ts";
import { handleAddTagPrompt, removeTagFromChat } from "./tags.ts";
import {
  handleToggleThinking,
  handleChatCopy,
  handleChatExport,
  handleChatFork,
  handleChatArchive,
  handleChatUnarchive,
  handleChatDelete,
} from "./actions.ts";
import { handleToggleTuning } from "./panel-readonly.ts";
import { showReadOnlyParams } from "./panel-readonly.ts";

export function setupTranscriptEvents(): void {
  const tc = getChatDom().transcriptContainer;

  tc.addEventListener("click", (e) => {
    const target = e.target as HTMLElement;

    // User message action buttons
    const userActionBtn = target.closest<HTMLElement>(".user-action-btn[data-action]");
    if (userActionBtn) {
      e.stopPropagation();
      const action = userActionBtn.dataset["action"];
      const userMsgs = tc.querySelectorAll("[data-transcript-items] .user-msg");
      const userMsg = userActionBtn.closest(".user-msg");
      const msgIndex = Array.from(userMsgs).indexOf(userMsg!);

      if (msgIndex >= 0) {
        switch (action) {
          case "copy":
            handleCopyUserMessage(msgIndex);
            break;
          case "edit":
            handleEditUserMessage(msgIndex);
            break;
          case "rerun":
            handleRerunFromMessage(msgIndex);
            break;
        }
      }
      return;
    }

    // Assistant message action buttons (copy)
    const assistantActionBtn = target.closest<HTMLElement>(".assistant-action-btn[data-action]");
    if (assistantActionBtn) {
      e.stopPropagation();
      const action = assistantActionBtn.dataset["action"];
      const assistantMsgs = tc.querySelectorAll("[data-transcript-items] .assistant-msg");
      const assistantMsg = assistantActionBtn.closest(".assistant-msg");
      const msgIndex = Array.from(assistantMsgs).indexOf(assistantMsg!);

      if (msgIndex >= 0 && action === "copy") {
        handleCopyAssistantMessage(msgIndex);
      } else if (action === "show-message-details") {
        // Always show read-only+stats (X button closes)
        const generationData = assistantActionBtn.dataset["generation"];
        const usageData = assistantActionBtn.dataset["usage"];
        showReadOnlyParams(
          generationData ? JSON.parse(generationData) : null,
          usageData ? JSON.parse(usageData) : null,
        );
      }
      return;
    }

    // Add tag button in chat header
    const addTagBtn = target.closest<HTMLElement>(".add-tag-btn");
    if (addTagBtn) {
      handleAddTagPrompt();
      return;
    }

    // Remove tag button in chat header
    const removeTagBtn = target.closest<HTMLElement>(".tag-remove");
    if (removeTagBtn?.dataset["tag"]) {
      removeTagFromChat(removeTagBtn.dataset["tag"]);
      return;
    }

    // Chat action buttons (copy, export, fork, archive, delete, toggle-thinking)
    const actionBtn = target.closest<HTMLButtonElement>(".chat-action[data-action]");
    if (actionBtn) {
      const action = actionBtn.dataset["action"];
      switch (action) {
        case "toggle-thinking":
          handleToggleThinking(actionBtn);
          break;
        case "toggle-tuning":
          handleToggleTuning(actionBtn);
          break;
        case "copy":
          if (chatState.activeSessionId) handleChatCopy();
          break;
        case "export":
          if (chatState.activeSessionId) handleChatExport();
          break;
        case "fork":
          if (chatState.activeSessionId) handleChatFork();
          break;
        case "archive":
          if (chatState.activeSessionId) handleChatArchive();
          break;
        case "unarchive":
          if (chatState.activeSessionId) handleChatUnarchive();
          break;
        case "delete":
          if (chatState.activeSessionId) handleChatDelete(actionBtn);
          break;
      }
    }
  });
}
