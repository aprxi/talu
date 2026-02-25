/** DOM rendering for the repo plugin. */

import { PIN_ICON, DELETE_ICON, CHECK_CIRCLE_ICON, CIRCLE_ICON, EXPORT_ICON, CLOSE_ICON } from "../../icons.ts";
import { renderEmptyState } from "../../render/common.ts";
import { el } from "../../render/helpers.ts";
import { format } from "./deps.ts";
import { getRepoDom } from "./dom.ts";
import { repoState, SIZE_MAX_PARAMS } from "./state.ts";
import type { CachedModel, SearchResult } from "./state.ts";

const ICON_CHECKED = CHECK_CIRCLE_ICON;
const ICON_UNCHECKED = CIRCLE_ICON;

/** Regex to extract parameter size from model IDs like "Qwen3-0.6B", "Llama-3-8B". */
const PARAMS_RE = /[\-_](\d+(?:\.\d+)?)[Bb]\b/;

/** Estimate params from model ID when metadata is missing (params_total=0). */
function estimateParams(modelId: string): number {
  const m = PARAMS_RE.exec(modelId);
  if (!m) return 0;
  return parseFloat(m[1]!) * 1_000_000_000;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const val = bytes / Math.pow(1024, i);
  return `${val < 10 ? val.toFixed(1) : Math.round(val)} ${units[i]}`;
}

function formatNumber(n: number): string {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function sortModels(models: CachedModel[]): CachedModel[] {
  const dir = repoState.sortDir === "asc" ? 1 : -1;
  return [...models].sort((a, b) => {
    switch (repoState.sortBy) {
      case "name": return dir * a.id.localeCompare(b.id);
      case "size": return dir * (a.size_bytes - b.size_bytes);
      case "date": return dir * (a.mtime - b.mtime);
      default: return 0;
    }
  });
}

/** Get the source filter value(s) for the local tab. */
function sourceFilter(): string | null {
  const f = repoState.localSourceFilter;
  return f === "all" ? null : f;
}

// ---------------------------------------------------------------------------
// Table (renders into the active tab's tbody)
// ---------------------------------------------------------------------------

export function renderModelsTable(): void {
  const dom = getRepoDom();
  const tbody = dom.localTbody;
  tbody.innerHTML = "";

  const source = sourceFilter();
  let models = source
    ? repoState.models.filter((m) => m.source === source)
    : repoState.models;

  // Client-side search filter.
  if (repoState.searchQuery) {
    const q = repoState.searchQuery.toLowerCase();
    models = models.filter(
      (m) => m.id.toLowerCase().includes(q) || (m.architecture ?? "").toLowerCase().includes(q),
    );
  }

  models = sortModels(models);

  dom.count.textContent = `${models.length} model${models.length !== 1 ? "s" : ""}`;

  if (models.length === 0) {
    const msg = repoState.searchQuery
      ? `No models matching "${repoState.searchQuery}".`
      : "No local models. Use Discover to find and download models.";
    tbody.appendChild(renderEmptyState(msg));
    return;
  }

  renderSortIndicators();

  for (const model of models) {
    const selected = repoState.selectedIds.has(model.id);
    const row = el("tr", `files-row${selected ? " files-row-selected" : ""}`);
    row.dataset["id"] = model.id;

    // Checkbox
    const checkTd = el("td", "files-cell files-cell-check");
    const checkBtn = el("button", "files-check-btn");
    checkBtn.innerHTML = selected ? ICON_CHECKED : ICON_UNCHECKED;
    checkBtn.dataset["action"] = "toggle";
    checkTd.appendChild(checkBtn);
    row.appendChild(checkTd);

    // Model ID
    const nameTd = el("td", "files-cell files-cell-name");
    const nameSpan = el("span", "files-name-text", model.id);
    nameTd.appendChild(nameSpan);
    if (model.source) {
      const badge = el("span", `repo-source-badge repo-source-${model.source}`, model.source);
      nameTd.appendChild(badge);
    }
    row.appendChild(nameTd);

    // Architecture
    const archTd = el("td", "files-cell", model.architecture ?? "—");
    row.appendChild(archTd);

    // Quantization
    const quantTd = el("td", "files-cell");
    if (model.quant_scheme) {
      const qBadge = el("span", "repo-quant-badge", model.quant_scheme);
      quantTd.appendChild(qBadge);
    } else {
      quantTd.textContent = "—";
    }
    row.appendChild(quantTd);

    // Size
    const sizeTd = el("td", "files-cell files-cell-mono", formatBytes(model.size_bytes));
    row.appendChild(sizeTd);

    // Modified
    const dateTd = el("td", "files-cell files-cell-date");
    dateTd.textContent = model.mtime > 0 ? format.dateTime(model.mtime * 1000, "short") : "—";
    row.appendChild(dateTd);

    // Pin toggle
    const pinTd = el("td", "files-cell");
    const pinBtn = el("button", `repo-pin-btn${model.pinned ? " pinned" : ""}`);
    pinBtn.innerHTML = PIN_ICON;
    pinBtn.title = model.pinned ? "Unpin" : "Pin";
    pinBtn.dataset["action"] = "pin";
    pinTd.appendChild(pinBtn);
    row.appendChild(pinTd);

    // Delete
    const delTd = el("td", "files-cell files-row-actions");
    const delBtn = el("button", "files-action-btn");
    delBtn.innerHTML = DELETE_ICON;
    delBtn.title = "Delete";
    delBtn.dataset["action"] = "delete";
    delTd.appendChild(delBtn);
    row.appendChild(delTd);

    tbody.appendChild(row);
  }
}

// ---------------------------------------------------------------------------
// Discover (Hub search results)
// ---------------------------------------------------------------------------

export function renderDiscoverResults(): void {
  const dom = getRepoDom();
  dom.discoverResults.innerHTML = "";

  if (repoState.isLoading) {
    const spinner = el("div", "empty-state");
    spinner.innerHTML = '<div class="spinner"></div>';
    dom.discoverResults.appendChild(spinner);
    return;
  }

  // Apply client-side size filter.
  const maxParams = SIZE_MAX_PARAMS[repoState.discoverSize] ?? null;
  let results = repoState.searchResults;
  if (maxParams !== null) {
    results = results.filter((r) => {
      const params = r.params_total || estimateParams(r.model_id);
      return params === 0 || params <= maxParams;
    });
  }

  if (results.length === 0) {
    if (repoState.searchQuery) {
      dom.discoverResults.appendChild(
        renderEmptyState(`No results for "${repoState.searchQuery}".`),
      );
    } else {
      dom.discoverResults.appendChild(
        renderEmptyState("Search HuggingFace Hub for models to download."),
      );
    }
    return;
  }

  for (const result of results) {
    const item = el("div", "repo-discover-item");
    item.dataset["modelId"] = result.model_id;

    // Top row: model id + download button / progress
    const top = el("div", "repo-discover-item-top");
    const title = el("div", "repo-discover-item-title", result.model_id);
    top.appendChild(title);

    if (repoState.activeDownloads.has(result.model_id)) {
      const label = el("span", "repo-downloading-label", "Downloading\u2026");
      top.appendChild(label);
    } else if (repoState.models.some((m) => m.id === result.model_id)) {
      const delBtn = el("button", "btn btn-ghost btn-sm repo-delete-btn");
      delBtn.innerHTML = `${DELETE_ICON} Delete`;
      delBtn.dataset["action"] = "delete";
      top.appendChild(delBtn);
    } else {
      const dlBtn = el("button", "btn btn-ghost btn-sm repo-download-btn");
      dlBtn.innerHTML = `${EXPORT_ICON} Download`;
      dlBtn.dataset["action"] = "download";
      top.appendChild(dlBtn);
    }

    item.appendChild(top);

    // Meta row
    const meta = el("div", "repo-discover-item-meta");
    meta.innerHTML = [
      `<span title="Downloads">&darr; ${formatNumber(result.downloads)}</span>`,
      `<span title="Likes">&hearts; ${formatNumber(result.likes)}</span>`,
      result.params_total > 0 ? `<span title="Parameters">${formatNumber(result.params_total)} params</span>` : "",
      `<span title="Last modified">${result.last_modified.split("T")[0]}</span>`,
    ]
      .filter(Boolean)
      .join('<span class="browser-card-meta-separator">&middot;</span>');
    item.appendChild(meta);

    dom.discoverResults.appendChild(item);
  }
}

// ---------------------------------------------------------------------------
// Downloads strip (persistent across tabs)
// ---------------------------------------------------------------------------

export function renderDownloads(): void {
  const dom = getRepoDom();
  const { activeDownloads } = repoState;

  if (activeDownloads.size === 0) {
    dom.downloads.classList.add("hidden");
    dom.downloads.innerHTML = "";
    return;
  }

  dom.downloads.classList.remove("hidden");
  dom.downloads.innerHTML = "";

  for (const dl of activeDownloads.values()) {
    const row = el("div", "repo-dl-row");
    row.dataset["downloadId"] = dl.modelId;

    const name = el("span", "repo-dl-name", dl.modelId);
    row.appendChild(name);

    const progress = el("div", "repo-dl-progress");
    const bar = el("div", "repo-dl-bar");
    const pct = dl.total > 0 ? Math.round((dl.current / dl.total) * 100) : 0;
    bar.style.width = `${pct}%`;
    progress.appendChild(bar);
    row.appendChild(progress);

    const bytes = el("span", "repo-dl-bytes");
    bytes.textContent = dl.total > 0
      ? `${formatBytes(dl.current)} / ${formatBytes(dl.total)}`
      : dl.current > 0 ? formatBytes(dl.current) : dl.label;
    row.appendChild(bytes);

    const cancelBtn = el("button", "repo-dl-cancel");
    cancelBtn.innerHTML = CLOSE_ICON;
    cancelBtn.title = "Cancel download";
    cancelBtn.dataset["action"] = "cancel";
    cancelBtn.dataset["downloadId"] = dl.modelId;
    row.appendChild(cancelBtn);

    dom.downloads.appendChild(row);
  }
}

/** Targeted update of progress bar and byte counter — no DOM creation. */
export function updateDownloadProgress(): void {
  const dom = getRepoDom();
  for (const dl of repoState.activeDownloads.values()) {
    const row = dom.downloads.querySelector<HTMLElement>(`[data-download-id="${dl.modelId}"]`);
    if (!row) continue;

    const bar = row.querySelector<HTMLElement>(".repo-dl-bar");
    if (bar) {
      const pct = dl.total > 0 ? Math.round((dl.current / dl.total) * 100) : 0;
      bar.style.width = `${pct}%`;
    }

    const bytes = row.querySelector<HTMLElement>(".repo-dl-bytes");
    if (bytes) {
      bytes.textContent = dl.total > 0
        ? `${formatBytes(dl.current)} / ${formatBytes(dl.total)}`
        : dl.current > 0 ? formatBytes(dl.current) : dl.label;
    }
  }
}

// ---------------------------------------------------------------------------
// Sidebar stats
// ---------------------------------------------------------------------------

export function renderStats(): void {
  const dom = getRepoDom();
  const n = repoState.models.length;
  dom.stats.textContent = `${n} model${n !== 1 ? "s" : ""} \u00B7 ${formatBytes(repoState.totalSizeBytes)}`;
}

// ---------------------------------------------------------------------------
// Sort indicators
// ---------------------------------------------------------------------------

export function renderSortIndicators(): void {
  const dom = getRepoDom();
  const thead = dom.localThead;
  for (const th of thead.querySelectorAll<HTMLElement>("[data-sort]")) {
    const col = th.dataset["sort"];
    th.classList.toggle("files-th-sorted", col === repoState.sortBy);
    // Remove old arrow.
    const arrow = th.querySelector(".sort-arrow");
    if (arrow) arrow.remove();
    if (col === repoState.sortBy) {
      const span = document.createElement("span");
      span.className = "sort-arrow";
      span.textContent = repoState.sortDir === "asc" ? " \u25B2" : " \u25BC";
      th.appendChild(span);
    }
  }
}

// ---------------------------------------------------------------------------
// Toolbar
// ---------------------------------------------------------------------------

export function updateRepoToolbar(): void {
  const dom = getRepoDom();
  const count = repoState.selectedIds.size;
  dom.pinAllBtn.disabled = count === 0;
  dom.deleteBtn.disabled = count === 0;
  dom.bulkActions.classList.toggle("active", count > 0);
  dom.cancelBtn.classList.toggle("hidden", count === 0);
}

// ---------------------------------------------------------------------------
// Tab sync
// ---------------------------------------------------------------------------

export function syncRepoTabs(): void {
  const dom = getRepoDom();
  const { tab } = repoState;

  dom.discoverView.classList.toggle("hidden", tab !== "discover");
  dom.discoverToolbar.classList.toggle("hidden", tab !== "discover");
  dom.localView.classList.toggle("hidden", tab !== "local");
  dom.localToolbar.classList.toggle("hidden", tab !== "local");
}

export function syncSourceToggle(): void {
  const dom = getRepoDom();
  const { localSourceFilter } = repoState;

  dom.sourceAll.classList.toggle("active", localSourceFilter === "all");
  dom.sourceHub.classList.toggle("active", localSourceFilter === "hub");
  dom.sourceManaged.classList.toggle("active", localSourceFilter === "managed");
}
