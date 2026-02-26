/**
 * Browser plugin — conversations browser with search, tag filtering,
 * tab switching (active/archived), batch actions (delete, archive, restore, export).
 *
 * Owns its own search/filter state. Communicates with the chat plugin
 * via the "talu.chat" service for selectChat/refreshSidebar operations.
 */

import type { PluginDefinition, PluginContext } from "../../kernel/types.ts";
import { createApiClient } from "../../api.ts";
import type { SearchRequest, ChatService } from "../../types.ts";
import { initBrowserDeps } from "./deps.ts";
import { bState, search } from "./state.ts";
import { initBrowserDom, getBrowserDom } from "./dom.ts";
import { buildBrowserDOM } from "./build-dom.ts";
import { syncBrowserTabs, updateBrowserToolbar } from "./render.ts";
import { loadBrowserConversations, loadAvailableTags } from "./data.ts";
import { wireEvents } from "./events.ts";

function initConversationsBrowser(): void {
  // Cancel any in-flight load so it won't overwrite state.
  bState.loadGeneration++;
  bState.isLoading = false;

  const dom = getBrowserDom();
  bState.selectedIds.clear();
  dom.searchInput.value = "";
  search.query = "";
  search.tagFilters = [];
  search.projectFilter = null;
  bState.tab = "all";
  bState.pagination.currentPage = 1;
  bState.pagination.totalItems = 0;
  syncBrowserTabs();
  dom.clearBtn.classList.add("hidden");
  updateBrowserToolbar();

  loadAvailableTags();
  loadBrowserConversations();
}

export const browserPlugin: PluginDefinition = {
  manifest: {
    id: "talu.browser",
    name: "Conversations Browser",
    version: "0.1.0",
    builtin: true,
    contributes: { mode: { key: "conversations", label: "Conversations" } },
  },

  register(_ctx: PluginContext) {
    // No services to provide in MVP.
  },

  async run(ctx: PluginContext, _signal: AbortSignal) {
    const api = createApiClient((url, init) => ctx.network.fetch(url, init));
    const chatService = ctx.services.get<ChatService>("talu.chat")!;

    initBrowserDeps({
      api,
      notify: ctx.notifications,
      dialogs: ctx.dialogs,
      events: ctx.events,
      chatService,
      download: ctx.download,
      timers: ctx.timers,
      menus: ctx.menus,
      layout: ctx.layout,
      mode: ctx.mode,
    });

    ctx.services.provide("talu.sessions", {
      search: (req: SearchRequest) => api.search(req),
      getAvailableTags: () => search.availableTags,
      getConversations: () => bState.conversations,
    });

    buildBrowserDOM(ctx.container);
    initBrowserDom(ctx.container);
    wireEvents();

    // Activate contributed menu slot in the toolbar.
    const toolbarSlot = ctx.container.querySelector<HTMLElement>('[data-slot="browser:toolbar"]');
    if (toolbarSlot) {
      ctx.subscriptions.add(ctx.menus.renderSlot("browser:toolbar", toolbarSlot));
    }

    ctx.events.on<{ to: string }>("mode.changed", ({ to }) => {
      if (to === "conversations") {
        initConversationsBrowser();
      }
    });

    ctx.log.info("Browser plugin ready.");
  },
};
