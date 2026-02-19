/**
 * Files plugin event wiring — search, row clicks, rename, drag-and-drop,
 * upload, selection, tabs, and bulk actions.
 */

import type { Disposable } from "../../kernel/types.ts";
import { timers } from "./deps.ts";
import { fState, type SortColumn } from "./state.ts";
import { getFilesDom } from "./dom.ts";
import { renderFilesTable, renderPreview, syncFilesTabs, updateFilesToolbar } from "./render.ts";
import { loadFiles, refreshFiles, renameFile, deleteFile, uploadFiles, archiveFiles, restoreFiles, deleteFiles } from "./data.ts";

export function wireFileEvents(): void {
  const dom = getFilesDom();

  // -- Tabs -----------------------------------------------------------------

  dom.tabAll.addEventListener("click", () => {
    if (fState.tab === "all") return;
    fState.tab = "all";
    fState.selectedIds.clear();
    fState.selectedFileId = null;
    syncFilesTabs();
    refreshFiles();
  });

  dom.tabArchived.addEventListener("click", () => {
    if (fState.tab === "archived") return;
    fState.tab = "archived";
    fState.selectedIds.clear();
    fState.selectedFileId = null;
    syncFilesTabs();
    refreshFiles();
  });

  // -- Bulk actions ---------------------------------------------------------

  dom.selectAllBtn.addEventListener("click", () => {
    const allSelected = fState.selectedIds.size === fState.files.length && fState.files.length > 0;
    if (allSelected) {
      fState.selectedIds.clear();
    } else {
      for (const f of fState.files) fState.selectedIds.add(f.id);
    }
    renderFilesTable();
  });

  dom.archiveBtn.addEventListener("click", () => archiveFiles());
  dom.restoreBtn.addEventListener("click", () => restoreFiles());
  dom.deleteBtn.addEventListener("click", () => deleteFiles());
  dom.cancelBtn.addEventListener("click", () => {
    fState.selectedIds.clear();
    renderFilesTable();
  });

  // -- Search (debounced) ---------------------------------------------------

  let searchDebounce: Disposable | null = null;

  dom.searchInput.addEventListener("input", () => {
    searchDebounce?.dispose();
    searchDebounce = timers.setTimeout(() => {
      const query = dom.searchInput.value.trim();
      if (query === fState.searchQuery) return;
      fState.searchQuery = query;
      fState.selectedFileId = null;
      loadFiles(1);
    }, 200);
    const hasText = dom.searchInput.value.trim().length > 0;
    dom.searchClear.classList.toggle("hidden", !hasText);
  });

  dom.searchClear.addEventListener("click", () => {
    dom.searchInput.value = "";
    dom.searchClear.classList.add("hidden");
    fState.searchQuery = "";
    fState.selectedFileId = null;
    loadFiles(1);
  });

  // -- Column sort (delegated on thead) ------------------------------------

  dom.thead.addEventListener("click", (e) => {
    const th = (e.target as HTMLElement).closest<HTMLElement>("[data-sort]");
    if (!th) return;
    const col = th.dataset["sort"] as SortColumn;
    if (fState.sortBy === col) {
      fState.sortDir = fState.sortDir === "asc" ? "desc" : "asc";
    } else {
      fState.sortBy = col;
      fState.sortDir = "asc";
    }
    loadFiles(1);
  });

  // -- Table row clicks (delegated) ----------------------------------------

  dom.tbody.addEventListener("click", (e) => {
    const target = e.target as HTMLElement;

    // Checkbox toggle.
    const toggleBtn = target.closest<HTMLElement>("[data-action='toggle']");
    if (toggleBtn?.dataset["id"]) {
      e.stopPropagation();
      const id = toggleBtn.dataset["id"];
      if (fState.selectedIds.has(id)) {
        fState.selectedIds.delete(id);
      } else {
        fState.selectedIds.add(id);
      }
      renderFilesTable();
      return;
    }

    // Delete action.
    const delBtn = target.closest<HTMLElement>("[data-action='delete']");
    if (delBtn?.dataset["id"]) {
      e.stopPropagation();
      deleteFile(delBtn.dataset["id"]);
      return;
    }

    // Download link — let native <a> handle it.
    if (target.closest("[data-action='download']")) return;

    // Row click → preview (not select).
    const row = target.closest<HTMLElement>(".files-row");
    if (!row?.dataset["id"]) return;
    fState.selectedFileId = row.dataset["id"];
    renderFilesTable();
    renderPreview();
  });

  // Double-click on name → enter edit mode.
  dom.tbody.addEventListener("dblclick", (e) => {
    const target = e.target as HTMLElement;
    const nameCell = target.closest<HTMLElement>(".files-cell-name");
    if (!nameCell) return;
    const row = nameCell.closest<HTMLElement>(".files-row");
    if (!row?.dataset["id"]) return;

    fState.editingFileId = row.dataset["id"];
    renderFilesTable();

    // Auto-focus the input.
    const input = dom.tbody.querySelector<HTMLInputElement>(".files-name-input");
    if (input) {
      input.focus();
      // Select filename without extension.
      const dot = input.value.lastIndexOf(".");
      input.setSelectionRange(0, dot > 0 ? dot : input.value.length);
    }
  });

  // Rename input handlers (delegated via keydown/blur on tbody).
  dom.tbody.addEventListener("keydown", (e) => {
    const target = e.target as HTMLElement;
    if (!target.classList.contains("files-name-input")) return;
    const input = target as HTMLInputElement;
    const id = input.dataset["id"];
    if (!id) return;

    if (e.key === "Enter") {
      e.preventDefault();
      renameFile(id, input.value);
    } else if (e.key === "Escape") {
      e.preventDefault();
      fState.editingFileId = null;
      renderFilesTable();
    }
  });

  dom.tbody.addEventListener("focusout", (e) => {
    const target = e.target as HTMLElement;
    if (!target.classList.contains("files-name-input")) return;
    const input = target as HTMLInputElement;
    const id = input.dataset["id"];
    if (!id || fState.editingFileId !== id) return;
    renameFile(id, input.value);
  });

  // -- Preview panel actions (delegated) -----------------------------------

  dom.previewContent.addEventListener("click", (e) => {
    const target = e.target as HTMLElement;

    // Image lightbox: click on preview image to expand
    if (target.tagName === "IMG" && target.classList.contains("files-preview-img")) {
      e.stopPropagation();
      const src = (target as HTMLImageElement).src;
      const overlay = document.createElement("div");
      overlay.className = "image-lightbox";
      const img = document.createElement("img");
      img.src = src;
      overlay.appendChild(img);

      const dismiss = () => overlay.remove();
      overlay.addEventListener("click", dismiss);
      const onKey = (ev: KeyboardEvent) => {
        if (ev.key === "Escape") {
          dismiss();
          document.removeEventListener("keydown", onKey);
        }
      };
      document.addEventListener("keydown", onKey);

      document.body.appendChild(overlay);
      return;
    }

    const actionBtn = target.closest<HTMLElement>("[data-action]");
    if (!actionBtn) return;
    const action = actionBtn.dataset["action"];
    const id = actionBtn.dataset["id"];
    if (!id) return;

    if (action === "delete") {
      deleteFile(id);
    } else if (action === "rename") {
      fState.editingFileId = id;
      renderFilesTable();
      const input = dom.tbody.querySelector<HTMLInputElement>(".files-name-input");
      if (input) {
        input.focus();
        const dot = input.value.lastIndexOf(".");
        input.setSelectionRange(0, dot > 0 ? dot : input.value.length);
      }
    }
  });

  // -- Upload button -------------------------------------------------------

  dom.uploadBtn.addEventListener("click", () => {
    dom.fileInput.click();
  });

  dom.fileInput.addEventListener("change", () => {
    if (dom.fileInput.files && dom.fileInput.files.length > 0) {
      uploadFiles(dom.fileInput.files);
      dom.fileInput.value = "";
    }
  });

  // -- Drag-and-drop -------------------------------------------------------

  let dragCounter = 0;

  dom.mainDrop.addEventListener("dragenter", (e) => {
    e.preventDefault();
    dragCounter++;
    dom.dropOverlay.classList.remove("hidden");
  });

  dom.mainDrop.addEventListener("dragover", (e) => {
    e.preventDefault();
  });

  dom.mainDrop.addEventListener("dragleave", () => {
    dragCounter--;
    if (dragCounter <= 0) {
      dragCounter = 0;
      dom.dropOverlay.classList.add("hidden");
    }
  });

  dom.mainDrop.addEventListener("drop", (e) => {
    e.preventDefault();
    dragCounter = 0;
    dom.dropOverlay.classList.add("hidden");
    if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
      uploadFiles(e.dataTransfer.files);
    }
  });

}
