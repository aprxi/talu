/**
 * Files plugin data loading — list, search, rename, delete, upload, bulk actions.
 */

import type { FileObject } from "../../types.ts";
import { renderLoadingSpinner } from "../../render/common.ts";
import { api, notify, dialogs, upload } from "./deps.ts";
import { fState } from "./state.ts";
import { getFilesDom } from "./dom.ts";
import { renderFilesTable, renderStats, renderPreview, updateFilesToolbar } from "./render.ts";

export async function loadFiles(): Promise<void> {
  const dom = getFilesDom();
  fState.isLoading = true;
  dom.tbody.innerHTML = "";
  dom.tableContainer.prepend(renderLoadingSpinner());

  const marker = fState.tab === "archived" ? "archived" : "active";
  const result = await api.listFiles(100, marker);
  fState.isLoading = false;

  // Remove spinner.
  const spinner = dom.tableContainer.querySelector(".empty-state");
  spinner?.remove();

  if (result.ok && result.data) {
    fState.files = result.data.data;
  } else {
    notify.error(result.ok ? "No data returned" : result.error);
    fState.files = [];
  }

  renderFilesTable();
  renderStats();
  renderPreview();
}

/** Return files filtered by the current search query and sorted. */
export function getFilteredFiles(): FileObject[] {
  const q = fState.searchQuery.toLowerCase().trim();
  const filtered = q
    ? fState.files.filter((f) => f.filename.toLowerCase().includes(q))
    : [...fState.files];

  const dir = fState.sortDir === "asc" ? 1 : -1;
  filtered.sort((a, b) => {
    switch (fState.sortBy) {
      case "name":
        return dir * a.filename.localeCompare(b.filename);
      case "kind": {
        const ak = a.kind ?? "";
        const bk = b.kind ?? "";
        return dir * ak.localeCompare(bk);
      }
      case "size":
        return dir * (a.bytes - b.bytes);
      case "date":
        return dir * (a.created_at - b.created_at);
      default:
        return 0;
    }
  });

  return filtered;
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
    await loadFiles();
  }
}

// -- Bulk actions -------------------------------------------------------------

export async function archiveFiles(): Promise<void> {
  const ids = [...fState.selectedIds];
  if (ids.length === 0) return;

  let count = 0;
  for (const id of ids) {
    const result = await api.updateFile(id, { marker: "archived" });
    if (result.ok) count++;
  }

  fState.selectedIds.clear();
  fState.selectedFileId = null;
  notify.success(`Archived ${count} file${count > 1 ? "s" : ""}`);
  await loadFiles();
}

export async function restoreFiles(): Promise<void> {
  const ids = [...fState.selectedIds];
  if (ids.length === 0) return;

  let count = 0;
  for (const id of ids) {
    const result = await api.updateFile(id, { marker: "active" });
    if (result.ok) count++;
  }

  fState.selectedIds.clear();
  fState.selectedFileId = null;
  notify.success(`Restored ${count} file${count > 1 ? "s" : ""}`);
  await loadFiles();
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

  let count = 0;
  for (const id of ids) {
    const result = await api.deleteFile(id);
    if (result.ok) count++;
  }

  fState.selectedIds.clear();
  fState.selectedFileId = null;
  notify.success(`Deleted ${count} file${count > 1 ? "s" : ""}`);
  await loadFiles();
}
