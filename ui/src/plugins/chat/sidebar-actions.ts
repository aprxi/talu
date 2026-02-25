import { chatState } from "./state.ts";
import { api, notifications, layout } from "./deps.ts";
import { renderSidebar, refreshSidebar } from "./sidebar-list.ts";
import { isPinned, el } from "../../render/helpers.ts";
import { renderProjectPicker } from "../../render/project-picker.ts";
import { getCachedProjects, deleteApiProject } from "../../render/project-combo.ts";

/** Derive project list from loaded sessions (for the right-click project picker). */
function getKnownProjects(): { value: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const s of chatState.sessions) {
    const key = s.project_id ?? "__default__";
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([value, count]) => ({ value, count }))
    .sort((a, b) => b.count - a.count);
}

export async function handleTogglePin(chatId: string): Promise<void> {
  // Draft pin is local-only (no API call).
  if (chatId === "__draft__" && chatState.draftSession) {
    chatState.draftSession.pinned = !chatState.draftSession.pinned;
    renderSidebar();
    return;
  }

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

export function showProjectContextMenu(anchor: HTMLElement, chatId: string): void {
  const session = chatState.sessions.find((s) => s.id === chatId);
  if (!session) return;

  const currentProjectId = session.project_id ?? null;
  const popoverDisposable = layout.showPopover({
    anchor,
    content: renderProjectPicker({
      currentProjectId,
      projects: getKnownProjects(),
      onSelect: (projectId) => {
        popoverDisposable.dispose();
        void handleSetProject(chatId, projectId);
      },
    }),
    placement: "right",
  });
}

// ---------------------------------------------------------------------------
// Project group management (right-click on sidebar group headers)
// ---------------------------------------------------------------------------

/** Show a right-click context menu on a project group header. */
export function showGroupContextMenu(anchor: HTMLElement, projectName: string, nameSpan: HTMLElement): void {
  const menu = el("div", "context-menu");

  const renameBtn = el("button", "context-menu-item", "Rename");
  const deleteBtn = el("button", "context-menu-item destructive", "Delete");
  menu.appendChild(renameBtn);
  menu.appendChild(deleteBtn);

  const popover = layout.showPopover({ anchor, content: menu, placement: "right" });

  renameBtn.addEventListener("click", () => {
    popover.dispose();
    beginInlineRename(nameSpan, projectName);
  });

  deleteBtn.addEventListener("click", () => {
    popover.dispose();
    void handleGroupDelete(projectName);
  });
}

/** Make the group name span editable inline (same pattern as chat title rename). */
function beginInlineRename(nameSpan: HTMLElement, oldName: string): void {
  nameSpan.contentEditable = "plaintext-only";
  nameSpan.focus();

  // Select all text.
  const range = document.createRange();
  range.selectNodeContents(nameSpan);
  const sel = window.getSelection();
  sel?.removeAllRanges();
  sel?.addRange(range);

  let committed = false;
  const commit = () => {
    if (committed) return;
    committed = true;
    nameSpan.removeEventListener("keydown", onKey);
    nameSpan.contentEditable = "false";
    const newName = (nameSpan.textContent ?? "").trim();
    if (!newName || newName === oldName) {
      nameSpan.textContent = oldName;
      return;
    }
    void submitGroupRename(oldName, newName, nameSpan);
  };

  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Enter" || e.key === "Tab") { e.preventDefault(); nameSpan.blur(); }
    else if (e.key === "Escape") { e.preventDefault(); nameSpan.textContent = oldName; nameSpan.blur(); }
  };

  nameSpan.addEventListener("keydown", onKey);
  nameSpan.addEventListener("blur", commit, { once: true });
}

async function submitGroupRename(oldName: string, newName: string, nameSpan: HTMLElement): Promise<void> {
  const project = getCachedProjects().find((p) => p.name === oldName);
  if (!project) {
    notifications.error("Project not found");
    nameSpan.textContent = oldName;
    return;
  }

  const result = await api.updateProject(project.id, { name: newName });
  if (!result.ok) {
    notifications.error(result.error ?? "Failed to rename project");
    nameSpan.textContent = oldName;
    return;
  }

  project.name = newName;
  await refreshSidebar();
}

async function handleGroupDelete(name: string): Promise<void> {
  if (!confirm(`Delete "${name}"? Conversations will move to Default.`)) return;

  try {
    await deleteApiProject(name);
  } catch {
    notifications.error("Failed to delete project");
    return;
  }

  await refreshSidebar();
}

export async function handleSetProject(chatId: string, projectId: string | null): Promise<void> {
  const session = chatState.sessions.find((s) => s.id === chatId);
  if (!session) return;

  const oldProjectId = session.project_id ?? null;

  // Optimistic update
  session.project_id = projectId;
  if (chatState.activeChat?.id === chatId) {
    chatState.activeChat.project_id = projectId;
  }
  renderSidebar();

  const result = await api.patchConversation(chatId, { project_id: projectId });
  if (!result.ok) {
    // Revert
    session.project_id = oldProjectId;
    if (chatState.activeChat?.id === chatId) {
      chatState.activeChat.project_id = oldProjectId;
    }
    renderSidebar();
    notifications.error(result.error ?? "Failed to update project");
    return;
  }

  renderSidebar();
}
