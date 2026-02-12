/**
 * Browser plugin state — conversation list, selection, search, and tag filters.
 */

import type { Conversation } from "../../types.ts";

export const bState = {
  selectedIds: new Set<string>(),
  conversations: [] as Conversation[],
  tab: "all" as "all" | "archived",
};

export const search = {
  query: "",
  tagFilters: [] as string[],
  results: [] as Conversation[],
  cursor: null as string | null,
  hasMore: true,
  isLoading: false,
  availableTags: [] as { name: string; count: number }[],
};
