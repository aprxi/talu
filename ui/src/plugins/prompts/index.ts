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
import { initPromptsDeps, api as depsApi, log } from "./deps.ts";
import {
  promptsState,
  DEFAULT_PROMPT_KEY,
  BUILTIN_PROMPT_NAME,
  BUILTIN_PROMPT_CONTENT,
  type SavedPrompt,
} from "./state.ts";
import { renderList } from "./list.ts";
import { selectPrompt, showEmpty, emitPromptsChanged } from "./editor.ts";

/** Auto-select the effective default prompt. Exported for use by hosting plugin. */
export function initPage(): void {
  const effectiveId = promptsState.defaultId ?? promptsState.builtinId;
  if (effectiveId && promptsState.prompts.some((p) => p.id === effectiveId)) {
    selectPrompt(effectiveId);
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
  },

  register(ctx: PluginContext): void {
    const service: PromptsService = {
      getSelectedPromptId: () => promptsState.selectedId,
      getDefaultPromptId: () => promptsState.defaultId ?? promptsState.builtinId,
      getPromptNameById: (id) => promptsState.prompts.find((p) => p.id === id)?.name ?? null,
      getPromptContentById: (id) =>
        promptsState.prompts.find((p) => p.id === id)?.content ?? null,
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
            // Identify the built-in prompt by its tags.
            if (full.data.tags_text?.includes("builtin")) {
              promptsState.builtinId = full.data.id;
            }
          }
        }
        promptsState.prompts = fullPrompts;
      }
    } catch (e) {
      ctx.log.error("Failed to load prompts", e);
      promptsState.prompts = [];
    }

    // Read custom default prompt from server settings.
    const settingsResult = await api.getSettings();
    if (settingsResult.ok && settingsResult.data) {
      promptsState.defaultId = settingsResult.data.default_prompt_id ?? null;
    }

    // One-time migration from localStorage to server.
    const localDefault = await ctx.storage.get<string>(DEFAULT_PROMPT_KEY);
    if (localDefault && !promptsState.defaultId) {
      promptsState.defaultId = localDefault;
      api.patchSettings({ default_prompt_id: localDefault }).catch(() =>
        ctx.log.warn("Failed to migrate default prompt to server"),
      );
    }
    if (localDefault) {
      ctx.storage.delete(DEFAULT_PROMPT_KEY).catch(() =>
        ctx.log.warn("Failed to clear migrated default prompt from localStorage"),
      );
    }

    // Clear stale default if it points to a prompt that no longer exists.
    if (promptsState.defaultId && !promptsState.prompts.some((p) => p.id === promptsState.defaultId)) {
      promptsState.defaultId = null;
    }

    // Create built-in default prompt document if it doesn't exist.
    if (!promptsState.builtinId) {
      try {
        const result = await api.createDocument({
          type: "prompt",
          title: BUILTIN_PROMPT_NAME,
          content: { system: BUILTIN_PROMPT_CONTENT },
          tags_text: "builtin",
        });
        if (result.ok && result.data) {
          promptsState.prompts.push({
            id: result.data.id,
            name: BUILTIN_PROMPT_NAME,
            content: BUILTIN_PROMPT_CONTENT,
            createdAt: result.data.created_at,
            updatedAt: result.data.updated_at,
          });
          promptsState.builtinId = result.data.id;
        }
      } catch (e) {
        ctx.log.error("Failed to create built-in default prompt", e);
      }
    }

    emitPromptsChanged();

    ctx.log.info("Prompts loaded.", { count: promptsState.prompts.length });
  },
};
