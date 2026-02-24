/**
 * Browser plugin state — conversation list, selection, search, and tag filters.
 */

import type { Conversation } from "../../types.ts";

export const bState = {
  selectedIds: new Set<string>(),
  conversations: [] as Conversation[],
  tab: "all" as "all" | "archived",
  /** Guard against overlapping non-search loads. */
  isLoading: false,
  /** Incremented on each load to detect stale completions. */
  loadGeneration: 0,
  pagination: {
    currentPage: 1,
    pageSize: 50,
    totalItems: 0,
  },
};

export const search = {
  query: "",
  tagFilters: [] as string[],
  availableTags: [] as { name: string; count: number }[],
  /** Available projects from search aggregation (complete, not limited by pagination). */
  availableProjects: [] as { value: string; count: number }[],
  /** Project filter — only show sessions in this project (null = all). */
  projectFilter: null as string | null,
};
