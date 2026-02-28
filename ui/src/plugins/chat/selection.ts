import { getChatDom } from "./dom.ts";
import { chatState } from "./state.ts";
import { api, notifications, getModelsService, menus } from "./deps.ts";
import { renderTranscriptHeader, renderTranscript } from "../../render/transcript.ts";
import { renderEmptyState, renderLoadingSpinner } from "../../render/common.ts";
import { showInputBar, hideWelcome, hideInputBar, setInputEnabled } from "./welcome.ts";
import { layout } from "./deps.ts";
import { updatePanelChatInfo } from "./panel-params.ts";
import { hideChatPanel } from "./panel-readonly.ts";
import { handleTitleRename } from "./sidebar-actions.ts";
import { renderSidebar } from "./sidebar-list.ts";
import { appendGeneratingIndicator } from "./messages.ts";
import type { Conversation } from "../../types.ts";

export async function selectChat(id: string): Promise<void> {
  // If the current view has an active stream (foreground or background), save
  // the transcript DOM so the stream can continue writing to detached elements.
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
  chatState.activeSessionId = id;
  chatState.lastResponseId = null;
  setInputEnabled(true);
  renderSidebar();

  hideWelcome();
  hideChatPanel();

  // If navigating to a background-streaming chat with saved DOM, restore it
  // so the user sees live-streaming text instead of a stale API snapshot.
  const savedDom = chatState.backgroundStreamDom.get(id);
  if (chatState.backgroundStreamSessions.has(id) && savedDom) {
    dom.transcriptContainer.innerHTML = "";
    dom.transcriptContainer.appendChild(savedDom);
    chatState.backgroundStreamDom.delete(id);
    appendGeneratingIndicator(dom.transcriptContainer);
    hideInputBar();
    dom.inputText.value = "";
    dom.inputText.style.height = "auto";
    return;
  }

  dom.transcriptContainer.innerHTML = "";
  dom.transcriptContainer.appendChild(renderLoadingSpinner());
  hideInputBar();

  const result = await api.getConversation(id);
  if (!result.ok || !result.data) {
    dom.transcriptContainer.innerHTML = "";
    dom.transcriptContainer.appendChild(
      renderEmptyState(result.error ?? "Failed to load conversation"),
    );
    notifications.error(result.error ?? "Failed to load conversation");
    return;
  }

  // If user navigated away while loading, discard
  if (chatState.activeSessionId !== id) return;

  chatState.activeChat = result.data;
  renderChatView(result.data);

  if (chatState.backgroundStreamSessions.has(id)) {
    appendGeneratingIndicator(dom.transcriptContainer);
    hideInputBar();
  } else {
    showInputBar();
  }

  dom.inputText.value = "";
  dom.inputText.style.height = "auto";
}

export function renderChatView(chat: Conversation): void {
  const dom = getChatDom();
  dom.transcriptContainer.innerHTML = "";

  const displayModel = getModelsService()?.getActiveModel() ?? "";
  const header = renderTranscriptHeader(chat, { displayModel });
  dom.transcriptContainer.appendChild(header);

  setupHeaderEvents(header, chat.id);
  activateMenuSlot(header);

  layout.setTitle(chat.title || "Untitled");

  if (chat.items && chat.items.length > 0) {
    dom.transcriptContainer.appendChild(renderTranscript(chat.items));
  } else {
    dom.transcriptContainer.appendChild(renderEmptyState("No messages in this conversation"));
  }

  updatePanelChatInfo(chat);
}

/** Set up event listeners on the chat header. */
function setupHeaderEvents(header: HTMLElement, chatId: string): void {
  const titleEl = header.querySelector<HTMLElement>(".transcript-title");
  if (titleEl) {
    titleEl.addEventListener("blur", () => handleTitleRename(titleEl, chatId));
    titleEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        titleEl.blur();
      }
    });
  }
}

/** Ensure chat header exists, add it if missing (for new chats after completion). */
export function ensureChatHeader(chat: Conversation): void {
  const tc = getChatDom().transcriptContainer;
  const existingHeader = tc.querySelector(".transcript-header");
  if (existingHeader) return;

  const displayModel = getModelsService()?.getActiveModel() ?? "";
  const header = renderTranscriptHeader(chat, { displayModel });
  tc.insertBefore(header, tc.firstChild);
  setupHeaderEvents(header, chat.id);
  activateMenuSlot(header);
}

/** Activate contributed menu items in the transcript header's slot anchor. */
function activateMenuSlot(header: HTMLElement): void {
  const slot = header.querySelector<HTMLElement>('[data-slot="chat:transcript-actions"]');
  if (slot) menus.renderSlot("chat:transcript-actions", slot);
}
