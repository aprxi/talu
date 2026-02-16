/**
 * Files plugin state — file list, selection, and search.
 */

import type { FileObject } from "../../types.ts";

export const fState = {
  files: [] as FileObject[],
  isLoading: false,
  searchQuery: "",
  selectedFileId: null as string | null,
  editingFileId: null as string | null,
  selectedIds: new Set<string>(),
  tab: "all" as "all" | "archived",
};
