/** Lazy DOM cache for the repo plugin. */

export interface RepoDom {
  search: HTMLInputElement;
  searchClear: HTMLButtonElement;
  sourceAll: HTMLButtonElement;
  sourceHub: HTMLButtonElement;
  sourceManaged: HTMLButtonElement;
  stats: HTMLElement;
  discoverView: HTMLElement;
  discoverToolbar: HTMLElement;
  discoverContainer: HTMLElement;
  discoverResults: HTMLElement;
  localView: HTMLElement;
  localToolbar: HTMLElement;
  localThead: HTMLTableSectionElement;
  localTbody: HTMLTableSectionElement;
  localTableContainer: HTMLElement;
  downloads: HTMLElement;
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
  providersView: HTMLElement;
  providersList: HTMLElement;
  addProviderSelect: HTMLSelectElement;
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
    sourceAll: root.querySelector<HTMLButtonElement>("#rp-source-all")!,
    sourceHub: root.querySelector<HTMLButtonElement>("#rp-source-hub")!,
    sourceManaged: root.querySelector<HTMLButtonElement>("#rp-source-managed")!,
    stats: root.querySelector<HTMLElement>("#rp-stats")!,
    discoverView: root.querySelector<HTMLElement>("#rp-discover-view")!,
    discoverToolbar: root.querySelector<HTMLElement>("#rp-discover-toolbar")!,
    discoverContainer: root.querySelector<HTMLElement>("#rp-discover-container")!,
    discoverResults: root.querySelector<HTMLElement>("#rp-discover-results")!,
    localView: root.querySelector<HTMLElement>("#rp-local-view")!,
    localToolbar: root.querySelector<HTMLElement>("#rp-local-toolbar")!,
    localThead: root.querySelector<HTMLTableSectionElement>("#rp-local-thead")!,
    localTbody: root.querySelector<HTMLTableSectionElement>("#rp-local-tbody")!,
    localTableContainer: root.querySelector<HTMLElement>("#rp-local-table-container")!,
    downloads: root.querySelector<HTMLElement>("#rp-downloads")!,
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
    providersView: root.querySelector<HTMLElement>("#rp-providers-view")!,
    providersList: root.querySelector<HTMLElement>("#rp-providers-list")!,
    addProviderSelect: root.querySelector<HTMLSelectElement>("#rp-add-provider")!,
  };
  return cached;
}
