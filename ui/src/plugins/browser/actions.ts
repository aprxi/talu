/**
 * Browser plugin actions — delete, export, archive, restore.
 */

import { renderProjectPicker } from "../../render/project-picker.ts";
import { api, notify, dialogs, chatService, pluginDownload, layout } from "./deps.ts";
import { bState, search } from "./state.ts";
import { getBrowserDom } from "./dom.ts";
import { updateBrowserToolbar } from "./render.ts";
import { loadBrowserConversations, loadAvailableTags } from "./data.ts";

export async function handleBrowserDelete(): Promise<void> {
  if (bState.selectedIds.size === 0) return;

  const count = bState.selectedIds.size;
  const confirmed = await dialogs.confirm({
    title: "Delete conversations",
    message: `Permanently delete ${count} conversation(s)? This cannot be undone.`,
    destructive: true,
  });
  if (!confirmed) return;

  const dom = getBrowserDom();
  const idsToDelete = [...bState.selectedIds];
  dom.deleteBtn.disabled = true;
  dom.deleteBtn.textContent = "Deleting...";

  const result = await api.batchConversations({
    action: "delete",
    ids: idsToDelete,
  });

  dom.deleteBtn.textContent = "Delete";
  dom.deleteBtn.disabled = false;

  if (result.ok) {
    notify.info(`Deleted ${idsToDelete.length} conversation(s)`);
  } else {
    notify.error(result.error ?? "Failed to delete conversations");
  }

  bState.selectedIds.clear();
  updateBrowserToolbar();
  await loadBrowserConversations();
  await chatService.refreshSidebar();
}

export function handleBrowserExport(): void {
  if (bState.selectedIds.size === 0) return;

  const selected = bState.conversations.filter((c) => bState.selectedIds.has(c.id));
  const data = JSON.stringify(selected, null, 2);
  const blob = new Blob([data], { type: "application/json" });
  pluginDownload.save(blob, `talu-export-${selected.length}-conversations.json`);
  notify.info(`Exported ${selected.length} conversation(s)`);
}

export async function handleCardRestore(id: string): Promise<void> {
  const result = await api.patchConversation(id, { marker: "" });
  if (!result.ok) {
    notify.error(result.error ?? "Failed to restore");
    return;
  }

  notify.info("Restored conversation");

  await loadBrowserConversations();
  await chatService.selectChat(id);
  await chatService.refreshSidebar();
}

export async function handleBrowserArchive(): Promise<void> {
  if (bState.selectedIds.size === 0) return;

  const dom = getBrowserDom();
  dom.archiveBtn.disabled = true;
  dom.archiveBtn.textContent = "Archiving...";

  const idsToArchive = [...bState.selectedIds];
  const result = await api.batchConversations({
    action: "archive",
    ids: idsToArchive,
  });

  dom.archiveBtn.textContent = "Archive";
  dom.archiveBtn.disabled = false;

  if (result.ok) {
    notify.info(`Archived ${idsToArchive.length} conversation(s)`);
  } else {
    notify.error(result.error ?? "Failed to archive conversations");
  }

  bState.selectedIds.clear();
  updateBrowserToolbar();
  await loadBrowserConversations();
  await chatService.refreshSidebar();
}

export async function handleBrowserBulkRestore(): Promise<void> {
  if (bState.selectedIds.size === 0) return;

  const dom = getBrowserDom();
  dom.restoreBtn.disabled = true;
  dom.restoreBtn.textContent = "Restoring...";

  const idsToRestore = [...bState.selectedIds];
  const result = await api.batchConversations({
    action: "unarchive",
    ids: idsToRestore,
  });

  dom.restoreBtn.textContent = "Restore";
  dom.restoreBtn.disabled = false;

  if (result.ok) {
    notify.info(`Restored ${idsToRestore.length} conversation(s)`);
  } else {
    notify.error(result.error ?? "Failed to restore conversations");
  }

  bState.selectedIds.clear();
  updateBrowserToolbar();
  await loadBrowserConversations();
  await chatService.refreshSidebar();
}

export function showBrowserProjectContextMenu(anchor: HTMLElement, chatId: string): void {
  const session = bState.conversations.find((c) => c.id === chatId);
  if (!session) return;

  const currentProjectId = session.project_id ?? null;
  const popoverDisposable = layout.showPopover({
    anchor,
    content: renderProjectPicker({
      currentProjectId,
      projects: search.availableProjects,
      onSelect: (projectId) => {
        popoverDisposable.dispose();
        void handleSetBrowserProject(chatId, projectId);
      },
    }),
    placement: "right",
  });
}

export async function handleSetBrowserProject(chatId: string, projectId: string | null): Promise<void> {
  const result = await api.patchConversation(chatId, { project_id: projectId });
  if (!result.ok) {
    notify.error(result.error ?? "Failed to update project");
    return;
  }

  // Refresh data.
  await loadBrowserConversations();
  await loadAvailableTags();
}
