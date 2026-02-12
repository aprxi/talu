/**
 * Prompts plugin — CRUD prompt management via Documents API.
 *
 * Renders into Shadow DOM. Emits "prompts.changed" events so the chat
 * plugin can sync the welcome prompt selector. Provides "talu.prompts"
 * service for getSelectedPromptId/getPromptNameById.
 */

import type { PluginDefinition, PluginContext } from "../../kernel/types.ts";
import { createApiClient } from "../../api.ts";
import type { PromptsService } from "../../types.ts";
import { initPromptsDeps, storage, log } from "./deps.ts";
import { promptsState, DEFAULT_PROMPT_KEY, type SavedPrompt } from "./state.ts";
import { initPromptsDom } from "./dom.ts";
import { buildPromptsDOM } from "./build-dom.ts";
import { renderList } from "./list.ts";
import { selectPrompt, showEmpty, wireEvents, emitPromptsChanged } from "./editor.ts";

/** Auto-set defaults and select initial prompt on first load. */
function initPage(): void {
  // Auto-set first prompt as default if exactly one and no default set.
  if (promptsState.prompts.length === 1 && !promptsState.defaultId && promptsState.prompts[0]) {
    promptsState.defaultId = promptsState.prompts[0].id;
    storage.set(DEFAULT_PROMPT_KEY, promptsState.defaultId).catch(() => log.warn("Failed to save default prompt"));
    emitPromptsChanged();
  }

  // Auto-select default prompt.
  if (promptsState.defaultId && promptsState.prompts.some((p) => p.id === promptsState.defaultId)) {
    selectPrompt(promptsState.defaultId);
  } else {
    renderList();
    showEmpty();
  }
}

export const promptsPlugin: PluginDefinition = {
  manifest: {
    id: "talu.prompts",
    name: "Prompts",
    version: "0.1.0",
    builtin: true,
    contributes: { mode: { key: "prompts", label: "Prompts" } },
  },

  register(ctx: PluginContext): void {
    const service: PromptsService = {
      getSelectedPromptId: () => promptsState.selectedId,
      getPromptNameById: (id) => promptsState.prompts.find((p) => p.id === id)?.name ?? null,
      getAll: () => promptsState.prompts,
    };
    ctx.subscriptions.add(ctx.services.provide("talu.prompts", service));
    ctx.log.info("Registered prompts service.");
  },

  async run(ctx: PluginContext): Promise<void> {
    const api = createApiClient((url, init) => ctx.network.fetch(url, init));
    initPromptsDeps({
      api,
      events: ctx.events,
      storage: ctx.storage,
      clipboard: ctx.clipboard,
      timers: ctx.timers,
      notifications: ctx.notifications,
      log: ctx.log,
    });

    buildPromptsDOM(ctx.container);
    initPromptsDom(ctx.container);
    wireEvents();

    // Load prompts from API.
    try {
      const result = await api.listDocuments("prompt");
      if (result.ok && result.data) {
        const fullPrompts: SavedPrompt[] = [];
        for (const summary of result.data.data) {
          const full = await api.getDocument(summary.id);
          if (full.ok && full.data) {
            fullPrompts.push({
              id: full.data.id,
              name: full.data.title,
              content: full.data.content?.system ?? "",
              createdAt: full.data.created_at,
              updatedAt: full.data.updated_at,
            });
          }
        }
        promptsState.prompts = fullPrompts;
      }
    } catch (e) {
      ctx.log.error("Failed to load prompts", e);
      promptsState.prompts = [];
    }

    promptsState.defaultId = await ctx.storage.get<string>(DEFAULT_PROMPT_KEY);
    emitPromptsChanged();
    initPage();

    ctx.log.info("Prompts loaded.", { count: promptsState.prompts.length });
  },
};
