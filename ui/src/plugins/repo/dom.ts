/** Lazy DOM cache for the repo plugin. */

export interface RepoDom {
  search: HTMLInputElement;
  searchClear: HTMLButtonElement;
  tabLocal: HTMLButtonElement;
  tabPinned: HTMLButtonElement;
  tabDiscover: HTMLButtonElement;
  stats: HTMLElement;
  thead: HTMLTableSectionElement;
  tbody: HTMLTableSectionElement;
  tableContainer: HTMLElement;
  discoverContainer: HTMLElement;
  discoverResults: HTMLElement;
  count: HTMLElement;
  selectAllBtn: HTMLButtonElement;
  pinAllBtn: HTMLButtonElement;
  deleteBtn: HTMLButtonElement;
  cancelBtn: HTMLButtonElement;
  bulkActions: HTMLElement;
  sortSelect: HTMLSelectElement;
  sizeFilter: HTMLSelectElement;
  taskFilter: HTMLSelectElement;
  libraryFilter: HTMLSelectElement;
}

let root: HTMLElement;
let cached: RepoDom | null = null;

export function initRepoDom(container: HTMLElement): void {
  root = container;
  cached = null;
}

export function getRepoDom(): RepoDom {
  if (cached) return cached;
  cached = {
    search: root.querySelector<HTMLInputElement>("#rp-search")!,
    searchClear: root.querySelector<HTMLButtonElement>("#rp-search-clear")!,
    tabLocal: root.querySelector<HTMLButtonElement>("#rp-tab-local")!,
    tabPinned: root.querySelector<HTMLButtonElement>("#rp-tab-pinned")!,
    tabDiscover: root.querySelector<HTMLButtonElement>("#rp-tab-discover")!,
    stats: root.querySelector<HTMLElement>("#rp-stats")!,
    thead: root.querySelector<HTMLTableSectionElement>("#rp-thead")!,
    tbody: root.querySelector<HTMLTableSectionElement>("#rp-tbody")!,
    tableContainer: root.querySelector<HTMLElement>("#rp-table-container")!,
    discoverContainer: root.querySelector<HTMLElement>("#rp-discover-container")!,
    discoverResults: root.querySelector<HTMLElement>("#rp-discover-results")!,
    count: root.querySelector<HTMLElement>("#rp-count")!,
    selectAllBtn: root.querySelector<HTMLButtonElement>("#rp-select-all")!,
    pinAllBtn: root.querySelector<HTMLButtonElement>("#rp-pin-all")!,
    deleteBtn: root.querySelector<HTMLButtonElement>("#rp-delete")!,
    cancelBtn: root.querySelector<HTMLButtonElement>("#rp-cancel")!,
    bulkActions: root.querySelector<HTMLElement>("#rp-bulk-actions")!,
    sortSelect: root.querySelector<HTMLSelectElement>("#rp-sort")!,
    sizeFilter: root.querySelector<HTMLSelectElement>("#rp-size-filter")!,
    taskFilter: root.querySelector<HTMLSelectElement>("#rp-task-filter")!,
    libraryFilter: root.querySelector<HTMLSelectElement>("#rp-library-filter")!,
  };
  return cached;
}
