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
import { loadChatModels, syncPinnedToChatModels } from "./chat-models-data.ts";
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
    wireHostEvents(dom.hostsList, (hostId) => openTerminalForHost(hostId, ctx), () => {
      // Test bench: spawn a process via WebSocket, display output in a <pre>.
      const output = document.createElement("pre");
      output.style.cssText = "margin:0.5rem 0;padding:0.5rem;background:var(--bg-secondary);border-radius:4px;font-size:12px;max-height:200px;overflow:auto;";
      output.textContent = "[spawning process...]\n";
      dom.hostsList.appendChild(output);

      void (async () => {
        try {
          const proc = await ctx.agent.process.open("echo hello-from-process-ws");
          proc.onEvent((event) => {
            if (event.type === "data") {
              output.textContent += event.data ?? "";
            } else if (event.type === "exit") {
              output.textContent += `\n[exit code=${event.code ?? "unknown"}]\n`;
              setTimeout(() => { void proc.close().catch(() => {}); output.remove(); }, 5000);
            } else if (event.type === "error") {
              output.textContent += `\n[error: ${event.message}]\n`;
            }
          });
        } catch (err) {
          output.textContent += `[error: ${err instanceof Error ? err.message : String(err)}]\n`;
          setTimeout(() => output.remove(), 5000);
        }
      })();
    });
    wireTerminalEvents();

    // Load models first so inferFamilyKey works when emitChanged fires.
    await loadModels();
    await loadChatModels();
    await syncPinnedToChatModels();

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
  loadModels().then(() => syncPinnedToChatModels());
  renderChatModels();
  renderHosts();
}
