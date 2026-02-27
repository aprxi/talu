/** talu.repo — Model repository management plugin. */

import type { PluginDefinition, PluginContext } from "../../kernel/types.ts";
import { createApiClient } from "../../api.ts";
import { initRepoDeps } from "./deps.ts";
import { buildRepoDOM } from "./build-dom.ts";
import { initRepoDom } from "./dom.ts";
import { wireRepoEvents } from "./events.ts";
import { repoState } from "./state.ts";
import { loadModels, searchHub } from "./data.ts";
import { loadProviders } from "./providers-data.ts";
import { wireProviderEvents } from "./providers-render.ts";
import {
  renderModelsTable,
  renderStats,
  syncRepoTabs,
  updateRepoToolbar,
} from "./render.ts";
import { getRepoDom } from "./dom.ts";

export const repoPlugin: PluginDefinition = {
  manifest: {
    id: "talu.repo",
    name: "Models",
    version: "0.1.0",
    builtin: true,
    contributes: {
      mode: { key: "models", label: "Models" },
    },
  },

  register(_ctx: PluginContext): void {
    // No services to provide in V1.
  },

  async run(ctx: PluginContext): Promise<void> {
    const api = createApiClient((url, init) => ctx.network.fetch(url, init));

    initRepoDeps({
      api,
      events: ctx.events,
      notifications: ctx.notifications,
      dialogs: ctx.dialogs,
      timers: ctx.timers,
      format: ctx.format,
      status: ctx.status,
    });

    buildRepoDOM(ctx.container);
    initRepoDom(ctx.container);
    wireRepoEvents();
    wireProviderEvents(getRepoDom().providersList);

    // Refresh models when the Models tab is activated.
    ctx.events.on<{ to: string }>("mode.changed", ({ to }) => {
      if (to === "models") {
        initRepoView();
      }
    });

    ctx.events.on<{ tab: string }>("subnav.tab", ({ tab }) => {
      if (tab !== "discover" && tab !== "local" && tab !== "providers") return;
      if (tab === repoState.tab) return;
      const dom = getRepoDom();
      repoState.tab = tab as typeof repoState.tab;
      repoState.selectedIds.clear();
      repoState.searchQuery = "";
      dom.search.value = "";
      dom.searchClear.classList.add("hidden");
      syncRepoTabs();
      updateRepoToolbar();
      if (tab === "discover") {
        searchHub(repoState.searchQuery);
      } else if (tab === "providers") {
        loadProviders();
      } else {
        renderModelsTable();
      }
    });

    ctx.log.info("Repo plugin ready.");
  },
};

/** Reset state and reload data when the mode becomes active. */
function initRepoView(): void {
  repoState.selectedIds.clear();
  syncRepoTabs();
  updateRepoToolbar();
  loadModels();
}
