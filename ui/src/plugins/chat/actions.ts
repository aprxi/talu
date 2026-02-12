import { getChatDom } from "./dom.ts";
import { chatState } from "./state.ts";
import { api, notifications, clipboard, download, timers } from "./deps.ts";
import { isThinkingExpanded, setThinkingExpanded } from "../../render/helpers.ts";
import { refreshSidebar } from "./sidebar-list.ts";
import { showWelcome, hideInputBar } from "./welcome.ts";
import { selectChat, renderChatView } from "./selection.ts";
import type { InputTextPart, OutputTextPart } from "../../types.ts";

/** Toggle thinking/reasoning blocks expanded/collapsed. */
export function handleToggleThinking(btn: HTMLButtonElement): void {
  const next = !isThinkingExpanded();
  setThinkingExpanded(next);

  if (next) {
    btn.classList.add("active");
    btn.title = "Collapse thoughts";
  } else {
    btn.classList.remove("active");
    btn.title = "Expand thoughts";
  }

  for (const d of getChatDom().transcriptContainer.querySelectorAll<HTMLDetailsElement>("details:has(.reasoning-summary)")) {
    d.open = next;
  }
}

/** Copy the conversation as markdown to clipboard. */
export async function handleChatCopy(): Promise<void> {
  if (!chatState.activeChat?.items) return;

  const lines: string[] = [];
  if (chatState.activeChat.title) {
    lines.push(`# ${chatState.activeChat.title}`, "");
  }

  for (const item of chatState.activeChat.items) {
    if (item.type === "message") {
      const role = item.role === "user" ? "User" : item.role === "assistant" ? "Assistant" : item.role;
      const text = item.content
        .filter((p): p is InputTextPart | OutputTextPart =>
          p.type === "input_text" || p.type === "output_text")
        .map((p) => p.text)
        .join("\n");
      if (text.trim()) {
        lines.push(`**${role}:** ${text}`, "");
      }
    }
  }

  try {
    await clipboard.writeText(lines.join("\n"));
    notifications.info("Copied to clipboard");
  } catch {
    notifications.error("Failed to copy");
  }
}

/** Export the conversation as JSON file download. */
export function handleChatExport(): void {
  if (!chatState.activeChat) return;

  const data = JSON.stringify(chatState.activeChat, null, 2);
  const blob = new Blob([data], { type: "application/json" });
  const filename = `${chatState.activeChat.title || "conversation"}.json`;
  download.save(blob, filename);
  notifications.info("Exported conversation");
}

/** Fork the conversation (create a branch from current state). */
export async function handleChatFork(): Promise<void> {
  if (!chatState.activeSessionId || !chatState.activeChat?.items) return;

  const lastIndex = chatState.activeChat.items.length - 1;
  const result = await api.forkConversation(chatState.activeSessionId, {
    target_item_id: lastIndex,
  });

  if (!result.ok || !result.data) {
    notifications.error(result.error ?? "Failed to fork conversation");
    return;
  }

  notifications.info("Forked conversation");
  await selectChat(result.data.id);
  await refreshSidebar();
}

/** Archive the conversation (set marker to "archived"). */
export async function handleChatArchive(): Promise<void> {
  if (!chatState.activeSessionId) return;

  const result = await api.patchConversation(chatState.activeSessionId, { marker: "archived" });
  if (!result.ok) {
    notifications.error(result.error ?? "Failed to archive");
    return;
  }

  notifications.info("Archived conversation");

  if (chatState.activeChat) {
    chatState.activeChat.marker = "archived";
    renderChatView(chatState.activeChat);
  }

  chatState.activeSessionId = null;
  chatState.activeChat = null;
  chatState.lastResponseId = null;
  getChatDom().transcriptContainer.innerHTML = "";
  showWelcome();
  hideInputBar();

  await refreshSidebar();
}

/** Unarchive the conversation (clear marker). */
export async function handleChatUnarchive(): Promise<void> {
  if (!chatState.activeSessionId) return;

  const result = await api.patchConversation(chatState.activeSessionId, { marker: "" });
  if (!result.ok) {
    notifications.error(result.error ?? "Failed to unarchive");
    return;
  }

  notifications.info("Unarchived conversation");

  if (chatState.activeChat) {
    chatState.activeChat.marker = "";
    renderChatView(chatState.activeChat);
  }

  await refreshSidebar();
}

/** Delete the conversation permanently with confirmation. */
export function handleChatDelete(btn: HTMLButtonElement): void {
  if (!chatState.activeSessionId) return;

  // Two-click confirmation
  if (btn.dataset["confirm"] !== "true") {
    btn.dataset["confirm"] = "true";
    btn.title = "Click again to confirm";
    btn.classList.add("text-danger", "bg-danger/10");
    timers.setTimeout(() => {
      btn.dataset["confirm"] = "";
      btn.title = "Delete conversation";
      btn.classList.remove("text-danger", "bg-danger/10");
    }, 3000);
    return;
  }

  const idToDelete = chatState.activeSessionId;
  api.deleteConversation(idToDelete).then(async (result) => {
    if (!result.ok) {
      notifications.error(result.error ?? "Failed to delete");
      return;
    }

    notifications.info("Deleted conversation");

    chatState.activeSessionId = null;
    chatState.activeChat = null;
    chatState.lastResponseId = null;

    getChatDom().transcriptContainer.innerHTML = "";
    showWelcome();
    hideInputBar();

    await refreshSidebar();
  });
}
