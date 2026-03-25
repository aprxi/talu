import { getChatDom } from "./dom.ts";
import { chatState } from "./state.ts";
import { SEND_ICON, STOP_ICON } from "../../icons.ts";
import { hideChatPanel } from "./panel-readonly.ts";
import { updatePanelChatInfo } from "./panel-params.ts";
import { layout } from "./deps.ts";
import { renderSidebar } from "./sidebar-list.ts";
import { clearAttachments } from "./attachments.ts";
import { navigate } from "../../kernel/system/router.ts";

export function showWelcome(): void {
  const dom = getChatDom();
  dom.welcomeState.classList.remove("hidden");
  hideInputBar();
  hideChatPanel();
  updatePanelChatInfo(null);
  layout.setTitle("");
}

export function hideWelcome(): void {
  getChatDom().welcomeState.classList.add("hidden");
}

export function showInputBar(): void {
  const dom = getChatDom();
  dom.inputBar.classList.remove("hidden");
  hideWelcome();
  dom.inputText.focus();
}

export function hideInputBar(): void {
  getChatDom().inputBar.classList.add("hidden");
}

export function startNewConversation(projectId?: string | null): void {
  // If the current view has an active stream, save the transcript DOM so the
  // stream can continue writing to detached elements while backgrounded.
  const currentId = chatState.activeSessionId;
  if (currentId && (chatState.isGenerating || chatState.backgroundStreamSessions.has(currentId))) {
    chatState.backgroundStreamSessions.add(currentId);
    const dom = getChatDom();
    const fragment = document.createDocumentFragment();
    while (dom.transcriptContainer.firstChild) {
      fragment.appendChild(dom.transcriptContainer.firstChild);
    }
    chatState.backgroundStreamDom.set(currentId, fragment);
  }
  chatState.activeViewId++;
  chatState.isGenerating = false;
  chatState.streamAbort = null;

  const dom = getChatDom();
  setInputEnabled(true);
  chatState.activeSessionId = null;
  chatState.activeChat = null;
  chatState.lastResponseId = null;
  chatState.pendingProjectId = projectId ?? null;

  // Update URL hash to chat welcome.
  navigate({ mode: "chat", sub: null, resource: null });

  // Ensure the project group is open so the user can see chats.
  const groupKey = projectId ?? "__default__";
  chatState.collapsedGroups.delete(groupKey);
  renderSidebar();

  dom.transcriptContainer.innerHTML = "";
  dom.welcomeProject.textContent = projectId && projectId !== "__default__" ? projectId : "";
  showWelcome();
  clearAttachments();
  dom.welcomeInput.value = "";
  dom.welcomeInput.style.height = "auto";
  // Reset inline advanced options.
  dom.welcomeAdvanced.classList.add("hidden");
  dom.welcomeGeneration.classList.remove("active");
  dom.welcomeTemperature.value = "";
  dom.welcomeTopP.value = "";
  dom.welcomeTopK.value = "";
  dom.welcomeMaxTokens.value = "";
  dom.welcomeContextLength.value = "";
  dom.welcomeInput.focus();
}

export function setInputEnabled(enabled: boolean): void {
  const dom = getChatDom();
  // Keep text inputs enabled so the user can type ahead during generation.
  dom.welcomeSend.disabled = !enabled;
  dom.welcomeAttach.disabled = !enabled;
  dom.inputAttach.disabled = !enabled;
  dom.fileInput.disabled = !enabled;

  if (enabled) {
    dom.inputSend.innerHTML = SEND_ICON;
    dom.inputSend.disabled = false;
    dom.inputSend.classList.remove("bg-danger", "hover:bg-danger/80");
    dom.inputSend.classList.add("bg-primary", "hover:bg-accent");
    dom.generationBar.classList.add("hidden");
  } else {
    dom.inputSend.innerHTML = STOP_ICON;
    dom.inputSend.disabled = false;
    dom.inputSend.classList.remove("bg-primary", "hover:bg-accent");
    dom.inputSend.classList.add("bg-danger", "hover:bg-danger/80");
    dom.generationBar.classList.remove("hidden");
  }
}
