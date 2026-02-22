/** Data loading and mutation functions for the repo plugin. */

import { api, events, notifications, dialogs, status } from "./deps.ts";
import { repoState } from "./state.ts";
import {
  renderModelsTable,
  renderDiscoverResults,
  renderStats,
  updateRepoToolbar,
} from "./render.ts";

// ---------------------------------------------------------------------------
// List cached models
// ---------------------------------------------------------------------------

export async function loadModels(): Promise<void> {
  repoState.isLoading = true;

  const qs: string[] = [];
  if (repoState.tab === "pinned") qs.push("pinned=true");
  const query = qs.length ? `?${qs.join("&")}` : "";

  const res = await api.listRepoModels(query);
  repoState.isLoading = false;

  if (res.ok && res.data) {
    repoState.models = res.data.models ?? [];
    repoState.totalSizeBytes = res.data.total_size_bytes ?? 0;
  } else {
    repoState.models = [];
    repoState.totalSizeBytes = 0;
  }

  renderModelsTable();
  renderStats();
  updateRepoToolbar();
}

// ---------------------------------------------------------------------------
// Hub search
// ---------------------------------------------------------------------------

export async function searchHub(query: string): Promise<void> {
  if (!query.trim()) {
    repoState.searchResults = [];
    renderDiscoverResults();
    return;
  }

  const gen = ++repoState.searchGeneration;
  repoState.isLoading = true;
  renderDiscoverResults(); // Show spinner.

  const res = await api.searchRepoModels(query, {
    sort: repoState.discoverSort,
    filter: repoState.discoverTask || undefined,
    library: repoState.discoverLibrary || undefined,
    limit: 50,
  });

  if (gen !== repoState.searchGeneration) return; // Superseded by newer search.
  repoState.isLoading = false;

  if (res.ok && res.data) {
    repoState.searchResults = res.data.results ?? [];
  } else {
    repoState.searchResults = [];
    if (res.error) notifications.error(`Search failed: ${res.error}`);
  }

  renderDiscoverResults();
}

// ---------------------------------------------------------------------------
// Download with SSE streaming
// ---------------------------------------------------------------------------

export async function downloadModel(modelId: string): Promise<void> {
  repoState.activeDownloads.set(modelId, {
    modelId,
    current: 0,
    total: 0,
    label: "Starting...",
    status: "downloading",
  });
  status.setBusy(`Downloading ${modelId}...`);
  renderDiscoverResults();

  try {
    const resp = await api.fetchRepoModel({ model_id: modelId });
    if (!resp.ok || !resp.body) {
      throw new Error(`HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (value) buffer += decoder.decode(value, { stream: true });

      // Parse SSE events from buffer.
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          const data = JSON.parse(line.slice(6));
          handleDownloadEvent(modelId, data);
        } catch {
          // Ignore malformed JSON.
        }
      }

      if (done) break;
    }

    // Ensure completion state.
    const dl = repoState.activeDownloads.get(modelId);
    if (dl && dl.status === "downloading") {
      dl.status = "done";
    }
  } catch (err) {
    const dl = repoState.activeDownloads.get(modelId);
    if (dl) dl.status = "error";
    notifications.error(`Download failed: ${err}`);
  }

  repoState.activeDownloads.delete(modelId);
  if (repoState.activeDownloads.size === 0) status.setReady();
  renderDiscoverResults();
  await loadModels();
  events.emit("repo.models.changed", {});
}

function handleDownloadEvent(modelId: string, data: Record<string, unknown>): void {
  const dl = repoState.activeDownloads.get(modelId);
  if (!dl) return;

  if (data.event === "done") {
    dl.status = "done";
    notifications.success(`Downloaded ${modelId}`);
    return;
  }

  if (data.event === "error") {
    dl.status = "error";
    notifications.error(`Download error: ${data.message ?? "unknown"}`);
    return;
  }

  // Progress event from the fetch streaming endpoint.
  if (typeof data.current === "number") dl.current = data.current as number;
  if (typeof data.total === "number") dl.total = data.total as number;
  if (typeof data.label === "string") dl.label = data.label as string;

  renderDiscoverResults();
}

// ---------------------------------------------------------------------------
// Delete
// ---------------------------------------------------------------------------

export async function deleteModel(modelId: string): Promise<void> {
  const confirmed = await dialogs.confirm(`Delete ${modelId} from local cache?`);
  if (!confirmed) return;

  const res = await api.deleteRepoModel(modelId);
  if (res.ok) {
    repoState.models = repoState.models.filter((m) => m.id !== modelId);
    repoState.selectedIds.delete(modelId);
    renderModelsTable();
    renderStats();
    updateRepoToolbar();
    notifications.success(`Deleted ${modelId}`);
    events.emit("repo.models.changed", {});
  } else {
    notifications.error(`Delete failed: ${res.error ?? "unknown"}`);
  }
}

// ---------------------------------------------------------------------------
// Pin / Unpin
// ---------------------------------------------------------------------------

export async function pinModel(modelId: string): Promise<void> {
  const res = await api.pinRepoModel(modelId);
  if (res.ok) {
    const model = repoState.models.find((m) => m.id === modelId);
    if (model) model.pinned = true;
    renderModelsTable();
  } else {
    notifications.error(`Pin failed: ${res.error ?? "unknown"}`);
  }
}

export async function unpinModel(modelId: string): Promise<void> {
  const res = await api.unpinRepoModel(modelId);
  if (res.ok) {
    const model = repoState.models.find((m) => m.id === modelId);
    if (model) model.pinned = false;
    renderModelsTable();
  } else {
    notifications.error(`Unpin failed: ${res.error ?? "unknown"}`);
  }
}

// ---------------------------------------------------------------------------
// Bulk operations
// ---------------------------------------------------------------------------

export async function deleteSelectedModels(): Promise<void> {
  const ids = [...repoState.selectedIds];
  if (ids.length === 0) return;

  const confirmed = await dialogs.confirm(
    `Delete ${ids.length} model${ids.length > 1 ? "s" : ""} from local cache?`,
  );
  if (!confirmed) return;

  const results = await Promise.all(ids.map((id) => api.deleteRepoModel(id)));
  const deleted = results.filter((r) => r.ok).length;

  repoState.selectedIds.clear();
  notifications.success(`Deleted ${deleted} model${deleted !== 1 ? "s" : ""}`);
  events.emit("repo.models.changed", {});
  await loadModels();
}

export async function pinSelectedModels(): Promise<void> {
  const ids = [...repoState.selectedIds];
  if (ids.length === 0) return;

  await Promise.all(ids.map(async (id) => {
    const res = await api.pinRepoModel(id);
    if (res.ok) {
      const model = repoState.models.find((m) => m.id === id);
      if (model) model.pinned = true;
    }
  }));

  repoState.selectedIds.clear();
  renderModelsTable();
  updateRepoToolbar();
}
