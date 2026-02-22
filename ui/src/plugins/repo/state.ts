/** Plugin-local state for the repository manager. */

export type RepoTab = "local" | "pinned" | "discover";
export type SortColumn = "name" | "size" | "date";
export type SortDir = "asc" | "desc";

// Discover filter types (matching CLI hf/filters.rs).
export type DiscoverSort = "trending" | "downloads" | "likes" | "last_modified";
export type SizeFilter = "1" | "2" | "4" | "8" | "16" | "32" | "64" | "128" | "512" | "any";
export type TaskFilter = "text-generation" | "image-text-to-text" | "image-to-text" | "text-to-image" | "text-to-speech" | "sentence-similarity" | "";
export type LibraryFilter = "safetensors" | "transformers" | "mlx" | "sentence-transformers" | "";

/** Client-side size filter thresholds (same as CLI filters.rs max_params). */
export const SIZE_MAX_PARAMS: Record<string, number | null> = {
  "1": 1_500_000_000,
  "2": 2_500_000_000,
  "4": 4_500_000_000,
  "8": 8_500_000_000,
  "16": 16_500_000_000,
  "32": 32_500_000_000,
  "64": 65_000_000_000,
  "128": 130_000_000_000,
  "512": 520_000_000_000,
  "any": null,
};

export interface CachedModel {
  id: string;
  path: string;
  source: string;
  size_bytes: number;
  mtime: number;
  architecture?: string;
  quant_scheme?: string;
  pinned: boolean;
}

export interface SearchResult {
  model_id: string;
  downloads: number;
  likes: number;
  last_modified: string;
  params_total: number;
}

export interface PinnedModel {
  model_uri: string;
  pinned_at_ms: number;
  size_bytes?: number;
}

export interface DownloadProgress {
  modelId: string;
  current: number;
  total: number;
  label: string;
  status: "downloading" | "done" | "error";
  abort: AbortController;
}

export const repoState = {
  tab: "local" as RepoTab,
  models: [] as CachedModel[],
  totalSizeBytes: 0,
  searchResults: [] as SearchResult[],
  searchQuery: "",
  isLoading: false,
  pins: [] as PinnedModel[],
  activeDownloads: new Map<string, DownloadProgress>(),
  selectedIds: new Set<string>(),
  sortBy: "name" as SortColumn,
  sortDir: "asc" as SortDir,
  searchGeneration: 0,
  discoverSort: "trending" as DiscoverSort,
  discoverSize: "8" as SizeFilter,
  discoverTask: "text-generation" as TaskFilter,
  discoverLibrary: "safetensors" as LibraryFilter,
};
