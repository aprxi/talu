/**
 * Settings plugin — global model and sampling configuration.
 *
 * Renders into a Shadow DOM container created by the kernel.
 * Provides "talu.models" service for model selection across plugins.
 */

import type { PluginDefinition, PluginContext, Disposable } from "../../kernel/types.ts";
import { createApiClient } from "../../api.ts";
import type { ModelsService, PromptsService } from "../../types.ts";
import { initSettingsDeps } from "./deps.ts";
import { settingsState, emitModelChanged } from "./state.ts";
import { initSettingsDom } from "./dom.ts";
import { buildSettingsDOM } from "./build-dom.ts";
import { populateForm, populateLocalModelSelect, updateSystemPromptDisplay, showModelParams, handleModelChange } from "./form.ts";
import { wireEvents } from "./events.ts";

export const settingsPlugin: PluginDefinition = {
  manifest: {
    id: "talu.settings",
    name: "Settings",
    version: "0.1.0",
    builtin: true,
    contributes: { mode: { key: "settings", label: "Settings" } },
  },

  register(ctx: PluginContext): void {
    const service: ModelsService = {
      getActiveModel: () => settingsState.activeModel,
      getAvailableModels: () => settingsState.availableModels,
      setActiveModel(id: string) {
        handleModelChange(id);
      },
      onChange(handler: () => void): Disposable {
        settingsState.changeHandlers.add(handler);
        return { dispose: () => settingsState.changeHandlers.delete(handler) };
      },
    };
    ctx.subscriptions.add(ctx.services.provide("talu.models", service));
    ctx.log.info("Registered models service.");
  },

  async run(ctx: PluginContext): Promise<void> {
    const api = createApiClient((url, init) => ctx.network.fetch(url, init));
    initSettingsDeps({ api, events: ctx.events, timers: ctx.timers, mode: ctx.mode });

    buildSettingsDOM(ctx.container);
    initSettingsDom(ctx.container);
    wireEvents();

    // Fetch settings from API.
    const result = await api.getSettings();
    if (!result.ok || !result.data) {
      ctx.notifications.error(result.error ?? "Failed to load settings");
      return;
    }

    settingsState.availableModels = result.data.available_models ?? [];
    settingsState.systemPromptEnabled = result.data.system_prompt_enabled;
    if (result.data.model) {
      settingsState.activeModel = result.data.model;
    } else if (settingsState.availableModels.length > 0 && settingsState.availableModels[0]) {
      settingsState.activeModel = settingsState.availableModels[0].id;
    }

    populateForm(result.data);
    populateLocalModelSelect();
    showModelParams(settingsState.activeModel);
    emitModelChanged();

    // Show current default prompt name from prompts service (if already loaded).
    // The prompts.changed event will update the display with the correct
    // isBuiltinDefault flag once the prompts plugin finishes loading.
    const promptsSvc = ctx.services.get<PromptsService>("talu.prompts");
    if (promptsSvc) {
      updateSystemPromptDisplay(promptsSvc.getAll(), promptsSvc.getDefaultPromptId(), false);
    }

    // Update display when prompts change (add/delete/rename/default toggle).
    ctx.events.on<{ prompts: { id: string; name: string }[]; defaultId: string | null; isBuiltinDefault: boolean }>(
      "prompts.changed",
      ({ prompts, defaultId, isBuiltinDefault }) => {
        updateSystemPromptDisplay(prompts, defaultId, isBuiltinDefault);
      },
    );

    // Re-fetch available models when the repo plugin downloads or deletes models.
    // Only update if chat models list is empty (user hasn't curated a list yet).
    ctx.events.on("repo.models.changed", async () => {
      if (settingsState.chatModelsActive) return;
      const res = await api.getSettings();
      if (res.ok && res.data) {
        settingsState.availableModels = res.data.available_models ?? [];
        populateLocalModelSelect();
        emitModelChanged();
      }
    });

    // Chat models (user-curated list from providers tab) replaces available models.
    ctx.events.on<{ models: string[] }>("repo.chatModels.changed", ({ models }) => {
      if (models.length === 0) {
        settingsState.chatModelsActive = false;
        return;
      }
      settingsState.chatModelsActive = true;
      settingsState.availableModels = models.map((id) => ({
        id,
        source: (id.includes("::") ? "hub" : "managed") as "hub" | "managed",
        defaults: { temperature: 1.0, top_k: 50, top_p: 1.0, do_sample: true },
        overrides: {},
      }));
      // If the active model is not in the new list, switch to the first entry.
      if (!models.includes(settingsState.activeModel) && models.length > 0) {
        settingsState.activeModel = models[0]!;
      }
      emitModelChanged();
    });

    ctx.log.info("Settings loaded.", { model: settingsState.activeModel, models: settingsState.availableModels.length });
  },
};
