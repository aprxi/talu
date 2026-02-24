/**
 * Browser plugin DOM element cache — lazy querySelector pattern.
 */

export interface BrowserDom {
  cardsEl: HTMLElement;
  searchInput: HTMLInputElement;
  clearBtn: HTMLButtonElement;
  tabAll: HTMLButtonElement;
  tabArchived: HTMLButtonElement;
  tagsEl: HTMLElement;
  tagsSection: HTMLElement;
  projectCombo: HTMLElement;
  projectsSection: HTMLElement;
  selectAllBtn: HTMLButtonElement;
  deleteBtn: HTMLButtonElement;
  exportBtn: HTMLButtonElement;
  archiveBtn: HTMLButtonElement;
  restoreBtn: HTMLButtonElement;
  cancelBtn: HTMLButtonElement;
  bulkActions: HTMLElement;
  toolbarEl: HTMLElement;
  paginationEl: HTMLElement;
}

let root: HTMLElement;
let cached: BrowserDom | null = null;

export function initBrowserDom(container: HTMLElement): void {
  root = container;
  cached = null;
}

export function getBrowserDom(): BrowserDom {
  if (cached) return cached;
  const q = (sel: string) => root.querySelector(sel)! as HTMLElement;
  cached = {
    cardsEl: q("#bp-cards"),
    searchInput: q("#bp-search") as HTMLInputElement,
    clearBtn: q("#bp-search-clear") as HTMLButtonElement,
    tabAll: q("#bp-tab-all") as HTMLButtonElement,
    tabArchived: q("#bp-tab-archived") as HTMLButtonElement,
    tagsEl: q("#bp-tags"),
    tagsSection: q("#bp-tags-section"),
    projectCombo: q("#bp-project-combo") as HTMLElement,
    projectsSection: q("#bp-projects-section"),
    selectAllBtn: q("#bp-select-all") as HTMLButtonElement,
    deleteBtn: q("#bp-delete") as HTMLButtonElement,
    exportBtn: q("#bp-export") as HTMLButtonElement,
    archiveBtn: q("#bp-archive") as HTMLButtonElement,
    restoreBtn: q("#bp-restore") as HTMLButtonElement,
    cancelBtn: q("#bp-cancel") as HTMLButtonElement,
    bulkActions: q("#bp-bulk-actions"),
    toolbarEl: q("#bp-toolbar"),
    paginationEl: q("#bp-pagination"),
  };
  return cached;
}
