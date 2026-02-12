/**
 * Browser plugin actions — delete, export, archive, restore.
 */

import { api, notify, dialogs, chatService, pluginDownload } from "./deps.ts";
import { bState, search } from "./state.ts";
import { getBrowserDom } from "./dom.ts";
import { updateBrowserToolbar, renderBrowserCards } from "./render.ts";

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

  if (result.ok) {
    const deleted = new Set(idsToDelete);
    bState.conversations = bState.conversations.filter((c) => !deleted.has(c.id));
    search.results = search.results.filter((c) => !deleted.has(c.id));
    notify.info(`Deleted ${idsToDelete.length} conversation(s)`);
  } else {
    notify.error(result.error ?? "Failed to delete conversations");
  }

  dom.deleteBtn.textContent = "Delete";
  dom.deleteBtn.disabled = false;

  bState.selectedIds.clear();
  updateBrowserToolbar();
  renderBrowserCards();
  await chatService.refreshSidebar();
}

export function handleBrowserExport(): void {
  if (bState.selectedIds.size === 0) return;

  const source = search.query.trim()
    ? search.results
    : bState.conversations;
  const selected = source.filter((c) => bState.selectedIds.has(c.id));
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

  const conv = bState.conversations.find((c) => c.id === id);
  if (conv) conv.marker = "";
  const searchConv = search.results.find((c) => c.id === id);
  if (searchConv) searchConv.marker = "";

  notify.info("Restored conversation");

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

  if (result.ok) {
    for (const id of idsToArchive) {
      const conv = bState.conversations.find((c) => c.id === id);
      if (conv) conv.marker = "archived";
      const searchConv = search.results.find((c) => c.id === id);
      if (searchConv) searchConv.marker = "archived";
    }
    notify.info(`Archived ${idsToArchive.length} conversation(s)`);
  } else {
    notify.error(result.error ?? "Failed to archive conversations");
  }

  dom.archiveBtn.textContent = "Archive";
  dom.archiveBtn.disabled = false;

  bState.selectedIds.clear();
  updateBrowserToolbar();
  renderBrowserCards();
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

  if (result.ok) {
    for (const id of idsToRestore) {
      const conv = bState.conversations.find((c) => c.id === id);
      if (conv) conv.marker = "";
      const searchConv = search.results.find((c) => c.id === id);
      if (searchConv) searchConv.marker = "";
    }
    notify.info(`Restored ${idsToRestore.length} conversation(s)`);
  } else {
    notify.error(result.error ?? "Failed to restore conversations");
  }

  dom.restoreBtn.textContent = "Restore";
  dom.restoreBtn.disabled = false;

  bState.selectedIds.clear();
  updateBrowserToolbar();
  renderBrowserCards();
  await chatService.refreshSidebar();
}
