/**
 * Files plugin — file manager with table view, preview panel,
 * drag-and-drop upload, search, and rename.
 */

import type { PluginDefinition, PluginContext } from "../../kernel/types.ts";
import { createApiClient } from "../../api.ts";
import { initFilesDeps } from "./deps.ts";
import { fState } from "./state.ts";
import { initFilesDom } from "./dom.ts";
import { buildFilesDOM } from "./build-dom.ts";
import { renderFilesTable, renderStats, renderPreview, syncFilesTabs } from "./render.ts";
import { loadFiles } from "./data.ts";
import { wireFileEvents } from "./events.ts";

function initFilesView(): void {
  // If already loading, let the in-flight request finish; just re-render.
  if (fState.isLoading) return;

  fState.files = [];
  fState.searchQuery = "";
  fState.selectedFileId = null;
  fState.editingFileId = null;
  fState.selectedIds.clear();
  fState.tab = "all";
  syncFilesTabs();
  renderFilesTable();
  renderStats();
  renderPreview();
  loadFiles();
}

export const filesPlugin: PluginDefinition = {
  manifest: {
    id: "talu.files",
    name: "File Manager",
    version: "0.1.0",
    builtin: true,
    contributes: { mode: { key: "files", label: "Files" } },
  },

  register(_ctx: PluginContext) {
    // No services to provide.
  },

  async run(ctx: PluginContext, _signal: AbortSignal) {
    const api = createApiClient((url, init) => ctx.network.fetch(url, init));

    initFilesDeps({
      api,
      notify: ctx.notifications,
      dialogs: ctx.dialogs,
      events: ctx.events,
      upload: ctx.upload,
      download: ctx.download,
      timers: ctx.timers,
      format: ctx.format,
    });

    buildFilesDOM(ctx.container);
    initFilesDom(ctx.container);
    wireFileEvents();

    ctx.events.on<{ to: string }>("mode.changed", ({ to }) => {
      if (to === "files") {
        initFilesView();
      }
    });

    ctx.log.info("Files plugin ready.");
  },
};
