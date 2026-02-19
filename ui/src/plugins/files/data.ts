/**
 * Files plugin data loading — list, search, rename, delete, upload, bulk actions.
 */

import type { FileObject } from "../../types.ts";
import { renderLoadingSpinner } from "../../render/common.ts";
import { api, notify, dialogs, upload } from "./deps.ts";
import { fState } from "./state.ts";
import { getFilesDom } from "./dom.ts";
import { renderFilesTable, renderStats, renderPreview, renderFilesPagination, updateFilesToolbar } from "./render.ts";

/** Load a page of files from the server. */
export async function loadFiles(page: number = 1): Promise<void> {
  fState.pagination.currentPage = page;
  const dom = getFilesDom();

  fState.isLoading = true;
  dom.tbody.innerHTML = "";
  dom.tableContainer.prepend(renderLoadingSpinner());

  const offset = (page - 1) * fState.pagination.pageSize;
  const marker = fState.tab === "archived" ? "archived" : "active";
  const result = await api.listFiles({
    limit: fState.pagination.pageSize,
    marker,
    offset,
    sort: fState.sortBy,
    order: fState.sortDir,
    search: fState.searchQuery || undefined,
  });

  fState.isLoading = false;
  const spinner = dom.tableContainer.querySelector(".empty-state");
  spinner?.remove();

  if (result.ok && result.data) {
    fState.files = result.data.data;
    fState.pagination.totalItems = result.data.total;
  } else {
    notify.error(result.ok ? "No data returned" : result.error);
  }

  renderFilesTable();
  renderFilesPagination();
  renderStats();
  renderPreview();
}

/** Reset to page 1 and reload from scratch. */
export async function refreshFiles(): Promise<void> {
  fState.files = [];
  fState.pagination.currentPage = 1;
  await loadFiles();
}

export async function renameFile(id: string, newName: string): Promise<void> {
  const trimmed = newName.trim();
  if (!trimmed) return;

  const result = await api.updateFile(id, { filename: trimmed });
  if (result.ok && result.data) {
    const idx = fState.files.findIndex((f) => f.id === id);
    if (idx >= 0) fState.files[idx] = result.data;
    notify.success("File renamed");
  } else {
    notify.error(result.ok ? "Rename failed" : result.error);
  }

  fState.editingFileId = null;
  renderFilesTable();
  renderPreview();
}

export async function deleteFile(id: string): Promise<void> {
  const file = fState.files.find((f) => f.id === id);
  const name = file?.filename ?? "this file";
  const ok = await dialogs.confirm({
    title: "Delete file",
    message: `Delete "${name}"? This cannot be undone.`,
    destructive: true,
  });
  if (!ok) return;

  const result = await api.deleteFile(id);
  if (result.ok) {
    fState.files = fState.files.filter((f) => f.id !== id);
    if (fState.selectedFileId === id) fState.selectedFileId = null;
    fState.selectedIds.delete(id);
    notify.success("File deleted");
  } else {
    notify.error(result.error);
  }

  renderFilesTable();
  renderStats();
  renderPreview();
}

export async function uploadFiles(files: globalThis.FileList): Promise<void> {
  let count = 0;
  for (const file of Array.from(files)) {
    try {
      await upload.upload(file, "assistants");
      count++;
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e));
    }
  }
  if (count > 0) {
    notify.success(`Uploaded ${count} file${count > 1 ? "s" : ""}`);
    await refreshFiles();
  }
}

// -- Bulk actions -------------------------------------------------------------

export async function archiveFiles(): Promise<void> {
  const ids = [...fState.selectedIds];
  if (ids.length === 0) return;

  const result = await api.batchFiles({ action: "archive", ids });
  fState.selectedIds.clear();
  fState.selectedFileId = null;

  if (result.ok) {
    notify.success(`Archived ${ids.length} file${ids.length > 1 ? "s" : ""}`);
  } else {
    notify.error(result.error ?? "Archive failed");
  }

  await refreshFiles();
}

export async function restoreFiles(): Promise<void> {
  const ids = [...fState.selectedIds];
  if (ids.length === 0) return;

  const result = await api.batchFiles({ action: "unarchive", ids });
  fState.selectedIds.clear();
  fState.selectedFileId = null;

  if (result.ok) {
    notify.success(`Restored ${ids.length} file${ids.length > 1 ? "s" : ""}`);
  } else {
    notify.error(result.error ?? "Restore failed");
  }

  await refreshFiles();
}

export async function deleteFiles(): Promise<void> {
  const ids = [...fState.selectedIds];
  if (ids.length === 0) return;

  const ok = await dialogs.confirm({
    title: "Delete files",
    message: `Delete ${ids.length} file${ids.length > 1 ? "s" : ""}? This cannot be undone.`,
    destructive: true,
  });
  if (!ok) return;

  const result = await api.batchFiles({ action: "delete", ids });
  fState.selectedIds.clear();
  fState.selectedFileId = null;

  if (result.ok) {
    notify.success(`Deleted ${ids.length} file${ids.length > 1 ? "s" : ""}`);
  } else {
    notify.error(result.error ?? "Delete failed");
  }

  await refreshFiles();
}
