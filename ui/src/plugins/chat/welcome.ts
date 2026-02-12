import { getChatDom } from "./dom.ts";
import { chatState } from "./state.ts";
import { SEND_ICON, STOP_ICON } from "../../icons.ts";
import { hideRightPanel } from "./panel-readonly.ts";
import { updatePanelChatInfo } from "./panel-params.ts";
import { layout } from "./deps.ts";
import { renderSidebar } from "./sidebar-list.ts";

export function showWelcome(): void {
  const dom = getChatDom();
  dom.welcomeState.classList.remove("hidden");
  hideInputBar();
  hideRightPanel();
  updatePanelChatInfo(null);
  layout.setTitle("");
}

export function hideWelcome(): void {
  getChatDom().welcomeState.classList.add("hidden");
}

export function showInputBar(): void {
  getChatDom().inputBar.classList.remove("hidden");
  hideWelcome();
}

export function hideInputBar(): void {
  getChatDom().inputBar.classList.add("hidden");
}

export function startNewConversation(): void {
  const dom = getChatDom();
  chatState.activeSessionId = null;
  chatState.activeChat = null;
  chatState.lastResponseId = null;
  renderSidebar();

  dom.transcriptContainer.innerHTML = "";
  showWelcome();
  dom.welcomeInput.value = "";
  dom.welcomeInput.style.height = "auto";
  dom.welcomeInput.focus();
}

export function setInputEnabled(enabled: boolean): void {
  const dom = getChatDom();
  dom.inputText.disabled = !enabled;
  dom.welcomeInput.disabled = !enabled;
  dom.welcomeSend.disabled = !enabled;

  if (enabled) {
    dom.inputSend.innerHTML = SEND_ICON;
    dom.inputSend.disabled = false;
    dom.inputSend.classList.remove("bg-danger", "hover:bg-danger/80");
    dom.inputSend.classList.add("bg-primary", "hover:bg-accent");
  } else {
    dom.inputSend.innerHTML = STOP_ICON;
    dom.inputSend.disabled = false;
    dom.inputSend.classList.remove("bg-primary", "hover:bg-accent");
    dom.inputSend.classList.add("bg-danger", "hover:bg-danger/80");
  }
}
