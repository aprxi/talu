import { chatState } from "./state.ts";
import { api, notifications } from "./deps.ts";
import { renderSidebar } from "./sidebar-list.ts";
import { isPinned } from "../../render/helpers.ts";

export async function handleTogglePin(chatId: string): Promise<void> {
  const session = chatState.sessions.find((s) => s.id === chatId);
  if (!session) return;

  const wasPinned = isPinned(session);
  const newMarker = wasPinned ? "" : "pinned";

  // Optimistic update
  session.marker = newMarker;
  if (chatState.activeChat?.id === chatId) {
    chatState.activeChat.marker = newMarker;
  }
  renderSidebar();

  const result = await api.patchConversation(chatId, { marker: newMarker });
  if (!result.ok) {
    // Revert
    session.marker = wasPinned ? "pinned" : "";
    if (chatState.activeChat?.id === chatId) {
      chatState.activeChat.marker = session.marker;
    }
    renderSidebar();
    notifications.error(result.error ?? "Failed to update pin");
  }
}

export async function handleTitleRename(
  titleEl: HTMLElement,
  chatId: string,
): Promise<void> {
  const newTitle = (titleEl.textContent ?? "").trim();
  const session = chatState.sessions.find((s) => s.id === chatId);
  if (!session) return;

  const oldTitle = session.title ?? "Untitled";
  if (newTitle === oldTitle) return;

  // Optimistic update
  session.title = newTitle || null;
  if (chatState.activeChat?.id === chatId) {
    chatState.activeChat.title = newTitle || null;
  }
  renderSidebar();

  const result = await api.patchConversation(chatId, {
    title: newTitle || "Untitled",
  });

  if (!result.ok) {
    // Revert
    session.title = oldTitle;
    if (chatState.activeChat?.id === chatId) {
      chatState.activeChat.title = oldTitle;
    }
    titleEl.textContent = oldTitle;
    renderSidebar();
    notifications.error(result.error ?? "Failed to rename");
  }
}
