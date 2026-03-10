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
import { loadChatModels } from "./chat-models-data.ts";
import { renderChatModels, wireChatModelEvents } from "./chat-models-render.ts";
import {
  renderModelsTable,
  renderStats,
  syncRepoTabs,
  updateRepoToolbar,
} from "./render.ts";
import { getRepoDom } from "./dom.ts";
import { renderHosts, wireHostEvents } from "./hosts-render.ts";
import { openTerminalForHost, wireTerminalEvents, disposeTerminals } from "./hosts-terminal.ts";
import { navigate, onRouteChange } from "../../kernel/system/router.ts";

export const repoPlugin: PluginDefinition = {
  manifest: {
    id: "talu.repo",
    name: "Router",
    version: "0.1.0",
    builtin: true,
    permissions: ["exec"],
    requiresCapabilities: ["exec"],
    contributes: {
      mode: { key: "routing", label: "Router" },
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

    const dom = getRepoDom();
    wireProviderEvents(dom.providersList);
    wireChatModelEvents(dom.chatModelsList);
    wireHostEvents(dom.hostsList, (hostId) => openTerminalForHost(hostId, ctx));
    wireTerminalEvents();

    // Load chat models from KV (needed before providers render for "Added" state).
    await loadChatModels();

    // Refresh when the Routing mode is activated; clean up terminal when leaving.
    ctx.events.on<{ to: string; from: string }>("mode.changed", ({ to, from }) => {
      if (to === "routing") {
        initRepoView();
      } else if (from === "routing" && repoState.subPage === "terminal") {
        disposeTerminals();
        repoState.subPage = null;
        repoState.activeTerminalHostId = null;
      }
    });

    // Back button: exit manage-local sub-page → return to providers view.
    dom.manageBackBtn.addEventListener("click", () => {
      repoState.subPage = null;
      repoState.selectedIds.clear();
      repoState.searchQuery = "";
      dom.search.value = "";
      dom.searchClear.classList.add("hidden");
      syncRepoTabs();
      loadProviders();
      renderChatModels();
      navigate({ mode: "routing", sub: null, resource: null });
    });

    // Manage-local tab switching (Local ↔ Discover).
    for (const btn of [dom.manageLocalTabBtn, dom.manageDiscoverTabBtn]) {
      btn.addEventListener("click", () => {
        const tab = btn.dataset["manageTab"] as "discover" | "local";
        if (tab === repoState.manageLocalTab) return;
        repoState.manageLocalTab = tab;
        repoState.selectedIds.clear();
        repoState.searchQuery = "";
        dom.search.value = "";
        dom.searchClear.classList.add("hidden");
        syncRepoTabs();
        updateRepoToolbar();
        if (tab === "discover") {
          searchHub(repoState.searchQuery);
        } else {
          renderModelsTable();
        }
      });
    }

    // Route-driven sub-page handling (Back/Forward + deep links).
    ctx.subscriptions.add(onRouteChange((route) => {
      if (route.mode !== "routing") return;
      if (route.sub === "manage-local" && repoState.subPage !== "manage-local") {
        repoState.subPage = "manage-local";
        syncRepoTabs();
      } else if (route.sub === "terminal" && route.resource) {
        if (repoState.activeTerminalHostId !== route.resource) {
          openTerminalForHost(route.resource, ctx);
        }
      } else if (!route.sub && repoState.subPage !== null) {
        disposeTerminals();
        repoState.subPage = null;
        repoState.activeTerminalHostId = null;
        syncRepoTabs();
        loadProviders();
        renderChatModels();
        renderHosts();
      }
    }));

    ctx.log.info("Repo plugin ready.");
  },
};

/** Reset state and show providers view when the mode becomes active. */
function initRepoView(): void {
  disposeTerminals();
  repoState.selectedIds.clear();
  repoState.subPage = null;
  repoState.activeTerminalHostId = null;
  repoState.tab = "providers";
  syncRepoTabs();
  updateRepoToolbar();
  loadProviders();
  renderChatModels();
  renderHosts();
}
