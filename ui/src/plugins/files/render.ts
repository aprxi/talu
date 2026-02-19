/**
 * Files plugin rendering — table, stats, preview panel, toolbar.
 */

import { renderEmptyState } from "../../render/common.ts";
import {
  DELETE_ICON as ICON_DELETE,
  EXPORT_ICON as ICON_DOWNLOAD,
  EDIT_ICON as ICON_RENAME,
  CHECK_CIRCLE_ICON as ICON_CHECKED,
  CIRCLE_ICON as ICON_UNCHECKED,
} from "../../icons.ts";
import { el } from "../../render/helpers.ts";
import { computePagination, renderPagination } from "../../render/pagination.ts";
import { format } from "./deps.ts";
import { fState, type SortColumn } from "./state.ts";
import { getFilesDom } from "./dom.ts";
import { loadFiles } from "./data.ts";

// -- Sort indicators ----------------------------------------------------------

const SORT_ASC = " \u25b2";   // ▲
const SORT_DESC = " \u25bc";  // ▼

const SORT_LABELS: Record<SortColumn, string> = {
  name: "Name",
  kind: "Kind",
  size: "Size",
  date: "Date",
};

function updateSortIndicators(thead: HTMLTableSectionElement): void {
  const headers = thead.querySelectorAll<HTMLElement>("[data-sort]");
  for (const th of headers) {
    const col = th.dataset["sort"] as SortColumn;
    const label = SORT_LABELS[col] ?? col;
    if (col === fState.sortBy) {
      th.textContent = label + (fState.sortDir === "asc" ? SORT_ASC : SORT_DESC);
      th.classList.add("files-th-sorted");
    } else {
      th.textContent = label;
      th.classList.remove("files-th-sorted");
    }
  }
}

// -- Helpers ------------------------------------------------------------------

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function kindBadgeClass(kind?: string, mimeType?: string): string {
  if (kind === "image" || mimeType?.startsWith("image/")) return "files-kind-badge files-kind-image";
  if (kind === "text" || mimeType?.startsWith("text/")) return "files-kind-badge files-kind-text";
  return "files-kind-badge files-kind-binary";
}

function kindLabel(kind?: string, mimeType?: string): string {
  if (kind) return kind;
  if (mimeType?.startsWith("image/")) return "image";
  if (mimeType?.startsWith("text/")) return "text";
  if (mimeType?.startsWith("application/pdf")) return "document";
  return "binary";
}

// -- Tabs ---------------------------------------------------------------------

export function syncFilesTabs(): void {
  const dom = getFilesDom();
  dom.tabAll.className = `browser-tab${fState.tab === "all" ? " active" : ""}`;
  dom.tabArchived.className = `browser-tab${fState.tab === "archived" ? " active" : ""}`;

  if (fState.tab === "all") {
    dom.archiveBtn.classList.remove("hidden");
    dom.restoreBtn.classList.add("hidden");
  } else {
    dom.archiveBtn.classList.add("hidden");
    dom.restoreBtn.classList.remove("hidden");
  }
}

// -- Toolbar ------------------------------------------------------------------

export function updateFilesToolbar(): void {
  const dom = getFilesDom();
  const hasSelection = fState.selectedIds.size > 0;

  dom.deleteBtn.disabled = !hasSelection;
  dom.archiveBtn.disabled = !hasSelection;
  dom.restoreBtn.disabled = !hasSelection;

  dom.bulkActions.classList.toggle("active", hasSelection);
  dom.cancelBtn.classList.toggle("hidden", !hasSelection);

  const allSelected = fState.selectedIds.size === fState.files.length && fState.files.length > 0;
  dom.selectAllBtn.textContent = allSelected ? "Deselect All" : "Select All";
}

// -- Table rendering ----------------------------------------------------------

export function renderFilesTable(): void {
  const dom = getFilesDom();
  const files = fState.files;
  const total = fState.pagination.totalItems;

  dom.tbody.innerHTML = "";
  dom.countEl.textContent = `${total} file${total !== 1 ? "s" : ""}`;

  // Update sort indicators on column headers.
  updateSortIndicators(dom.thead);

  if (total === 0 && !fState.isLoading) {
    const msg = fState.searchQuery
      ? "No matching files"
      : fState.tab === "archived"
        ? "No archived files"
        : "No files uploaded";
    const empty = renderEmptyState(msg);
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 6;
    td.className = "files-cell";
    td.appendChild(empty);
    tr.appendChild(td);
    dom.tbody.appendChild(tr);
    updateFilesToolbar();
    return;
  }

  for (const file of files) {
    const isSelected = fState.selectedIds.has(file.id);
    const isPreviewed = fState.selectedFileId === file.id;
    const isEditing = fState.editingFileId === file.id;

    const rowClass = [
      "files-row",
      isSelected ? "files-row-selected" : "",
      isPreviewed ? "files-row-previewed" : "",
    ].filter(Boolean).join(" ");

    const row = el("tr", rowClass);
    row.dataset["id"] = file.id;

    // Checkbox cell.
    const checkCell = el("td", "files-cell files-cell-check");
    const checkBtn = el("button", "files-check-btn");
    checkBtn.innerHTML = isSelected ? ICON_CHECKED : ICON_UNCHECKED;
    checkBtn.dataset["action"] = "toggle";
    checkBtn.dataset["id"] = file.id;
    checkCell.appendChild(checkBtn);
    row.appendChild(checkCell);

    // Name cell.
    const nameCell = el("td", "files-cell files-cell-name");
    if (isEditing) {
      const input = document.createElement("input");
      input.type = "text";
      input.className = "files-name-input";
      input.value = file.filename;
      input.dataset["id"] = file.id;
      nameCell.appendChild(input);
    } else {
      const nameSpan = el("span", "files-name-text", file.filename);
      nameCell.appendChild(nameSpan);
    }
    row.appendChild(nameCell);

    // Kind cell.
    const kindCell = el("td", "files-cell");
    const badge = el("span", kindBadgeClass(file.kind, file.mime_type), kindLabel(file.kind, file.mime_type));
    kindCell.appendChild(badge);
    row.appendChild(kindCell);

    // Size cell.
    row.appendChild(el("td", "files-cell files-cell-mono", formatSize(file.bytes)));

    // Date cell.
    const dateStr = format.dateTime(file.created_at * 1000, "short");
    row.appendChild(el("td", "files-cell files-cell-date", dateStr));

    // Actions cell.
    const actionsCell = el("td", "files-cell files-row-actions");

    const downloadBtn = document.createElement("a");
    downloadBtn.className = "btn btn-ghost btn-sm files-action-btn";
    downloadBtn.href = `/v1/files/${encodeURIComponent(file.id)}/content`;
    downloadBtn.download = file.filename;
    downloadBtn.title = "Download";
    downloadBtn.innerHTML = ICON_DOWNLOAD;
    downloadBtn.dataset["action"] = "download";
    actionsCell.appendChild(downloadBtn);

    const deleteBtn = el("button", "btn btn-ghost btn-sm files-action-btn");
    deleteBtn.title = "Delete";
    deleteBtn.innerHTML = ICON_DELETE;
    deleteBtn.dataset["action"] = "delete";
    deleteBtn.dataset["id"] = file.id;
    actionsCell.appendChild(deleteBtn);

    row.appendChild(actionsCell);
    dom.tbody.appendChild(row);
  }

  renderFilesPagination();
  updateFilesToolbar();
}

// -- Pagination ---------------------------------------------------------------

export function renderFilesPagination(): void {
  const dom = getFilesDom();
  dom.paginationContainer.innerHTML = "";

  const { totalItems, pageSize, currentPage } = fState.pagination;
  if (totalItems <= pageSize) return;

  const state = computePagination(totalItems, pageSize, currentPage);
  fState.pagination.currentPage = state.currentPage;

  const paginationEl = renderPagination(state, (page) => {
    loadFiles(page);
  });
  dom.paginationContainer.appendChild(paginationEl);
}

// -- Stats rendering ----------------------------------------------------------

export function renderStats(): void {
  const dom = getFilesDom();
  const pageBytes = fState.files.reduce((sum, f) => sum + f.bytes, 0);
  const total = fState.pagination.totalItems;
  dom.statsEl.textContent = `${total} files \u00b7 ${formatSize(pageBytes)}`;
}

// -- Preview panel rendering --------------------------------------------------

export function renderPreview(): void {
  const dom = getFilesDom();
  const file = fState.selectedFileId
    ? fState.files.find((f) => f.id === fState.selectedFileId)
    : null;

  if (!file) {
    dom.previewPanel.classList.add("hidden");
    dom.previewContent.innerHTML = "";
    return;
  }

  dom.previewPanel.classList.remove("hidden");
  dom.previewContent.innerHTML = "";

  const isImage = file.kind === "image" || file.mime_type?.startsWith("image/");
  const isText = file.kind === "text" || file.mime_type?.startsWith("text/");

  // Preview area.
  if (isImage) {
    const img = document.createElement("img");
    img.className = "files-preview-img";
    img.src = `/v1/files/${encodeURIComponent(file.id)}/content`;
    img.alt = file.filename;
    if (file.image) {
      img.style.aspectRatio = `${file.image.width} / ${file.image.height}`;
    }
    dom.previewContent.appendChild(img);
  } else if (isText) {
    const pre = el("pre", "files-preview-code", "Loading...");
    dom.previewContent.appendChild(pre);
    fetch(`/v1/files/${encodeURIComponent(file.id)}/content`, {
      headers: { Range: "bytes=0-1023" },
    })
      .then((r) => r.text())
      .then((text) => { pre.textContent = text; })
      .catch(() => { pre.textContent = "(failed to load preview)"; });
  } else {
    dom.previewContent.appendChild(el("div", "files-preview-empty", "No preview available"));
  }

  // Metadata.
  const meta = el("dl", "files-preview-meta");

  const addField = (label: string, value: string) => {
    meta.appendChild(el("dt", "", label));
    meta.appendChild(el("dd", "", value));
  };

  addField("Filename", file.filename);
  addField("Size", formatSize(file.bytes));
  addField("Kind", kindLabel(file.kind, file.mime_type));
  if (file.mime_type) addField("MIME", file.mime_type);
  if (file.image) {
    addField("Dimensions", `${file.image.width}\u00d7${file.image.height}`);
    addField("Format", file.image.format);
  }
  addField("Uploaded", format.dateTime(file.created_at * 1000));

  dom.previewContent.appendChild(meta);

  // Actions.
  const actions = el("div", "files-preview-actions");

  const dlLink = document.createElement("a");
  dlLink.className = "btn btn-sm";
  dlLink.href = `/v1/files/${encodeURIComponent(file.id)}/content`;
  dlLink.download = file.filename;
  dlLink.innerHTML = `${ICON_DOWNLOAD} Download`;
  actions.appendChild(dlLink);

  const renameBtn = el("button", "btn btn-ghost btn-sm");
  renameBtn.innerHTML = `${ICON_RENAME} Rename`;
  renameBtn.dataset["action"] = "rename";
  renameBtn.dataset["id"] = file.id;
  actions.appendChild(renameBtn);

  const delBtn = el("button", "btn btn-ghost btn-sm btn-danger");
  delBtn.innerHTML = `${ICON_DELETE} Delete`;
  delBtn.dataset["action"] = "delete";
  delBtn.dataset["id"] = file.id;
  actions.appendChild(delBtn);

  dom.previewContent.appendChild(actions);
}
