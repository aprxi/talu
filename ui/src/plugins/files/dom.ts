/**
 * Files plugin DOM element cache — lazy querySelector pattern.
 */

export interface FilesDom {
  uploadBtn: HTMLButtonElement;
  fileInput: HTMLInputElement;
  searchInput: HTMLInputElement;
  searchClear: HTMLButtonElement;
  statsEl: HTMLElement;
  countEl: HTMLElement;
  thead: HTMLTableSectionElement;
  tbody: HTMLTableSectionElement;
  tableContainer: HTMLElement;
  mainDrop: HTMLElement;
  dropOverlay: HTMLElement;
  previewPanel: HTMLElement;
  previewContent: HTMLElement;
  tabAll: HTMLButtonElement;
  tabArchived: HTMLButtonElement;
  selectAllBtn: HTMLButtonElement;
  archiveBtn: HTMLButtonElement;
  restoreBtn: HTMLButtonElement;
  deleteBtn: HTMLButtonElement;
  cancelBtn: HTMLButtonElement;
  bulkActions: HTMLElement;
}

let root: HTMLElement;
let cached: FilesDom | null = null;

export function initFilesDom(container: HTMLElement): void {
  root = container;
  cached = null;
}

export function getFilesDom(): FilesDom {
  if (cached) return cached;
  const q = (sel: string) => root.querySelector(sel)! as HTMLElement;
  cached = {
    uploadBtn: q("#fp-upload") as HTMLButtonElement,
    fileInput: q("#fp-file-input") as HTMLInputElement,
    searchInput: q("#fp-search") as HTMLInputElement,
    searchClear: q("#fp-search-clear") as HTMLButtonElement,
    statsEl: q("#fp-stats"),
    countEl: q("#fp-count"),
    thead: q("#fp-thead") as HTMLTableSectionElement,
    tbody: q("#fp-tbody") as HTMLTableSectionElement,
    tableContainer: q("#fp-table-container"),
    mainDrop: q(".files-main-drop"),
    dropOverlay: q("#fp-drop-overlay"),
    previewPanel: q("#fp-preview"),
    previewContent: q("#fp-preview-content"),
    tabAll: q("#fp-tab-all") as HTMLButtonElement,
    tabArchived: q("#fp-tab-archived") as HTMLButtonElement,
    selectAllBtn: q("#fp-select-all") as HTMLButtonElement,
    archiveBtn: q("#fp-archive") as HTMLButtonElement,
    restoreBtn: q("#fp-restore") as HTMLButtonElement,
    deleteBtn: q("#fp-delete") as HTMLButtonElement,
    cancelBtn: q("#fp-cancel") as HTMLButtonElement,
    bulkActions: q("#fp-bulk-actions"),
  };
  return cached;
}
