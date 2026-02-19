/**
 * Files plugin state — file list, selection, and search.
 */

import type { FileObject } from "../../types.ts";

export type SortColumn = "name" | "kind" | "size" | "date";
export type SortDir = "asc" | "desc";

export const fState = {
  files: [] as FileObject[],
  isLoading: false,
  searchQuery: "",
  selectedFileId: null as string | null,
  editingFileId: null as string | null,
  selectedIds: new Set<string>(),
  tab: "all" as "all" | "archived",
  sortBy: "name" as SortColumn,
  sortDir: "asc" as SortDir,
  pagination: {
    currentPage: 1,
    pageSize: 50,
    totalItems: 0,
  },
};
