/**
 * Settings plugin — global model and sampling configuration.
 *
 * Renders into a Shadow DOM container created by the kernel.
 * Provides "talu.models" service for model selection across plugins.
 */

import type { PluginDefinition, PluginContext, Disposable } from "../../kernel/types.ts";
import { createApiClient } from "../../api.ts";
import type { ModelsService } from "../../types.ts";
import { initSettingsDeps } from "./deps.ts";
import { settingsState, emitModelChanged } from "./state.ts";
import { initSettingsDom } from "./dom.ts";
import { buildSettingsDOM } from "./build-dom.ts";
import { populateForm, populateLocalModelSelect, showModelParams, handleModelChange } from "./form.ts";
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
    initSettingsDeps({ api, events: ctx.events, timers: ctx.timers });

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
    if (result.data.model) {
      settingsState.activeModel = result.data.model;
    } else if (settingsState.availableModels.length > 0 && settingsState.availableModels[0]) {
      settingsState.activeModel = settingsState.availableModels[0].id;
    }

    populateForm(result.data);
    populateLocalModelSelect();
    showModelParams(settingsState.activeModel);
    emitModelChanged();

    // Re-fetch available models when the repo plugin downloads or deletes models.
    ctx.events.on("repo.models.changed", async () => {
      const res = await api.getSettings();
      if (res.ok && res.data) {
        settingsState.availableModels = res.data.available_models ?? [];
        populateLocalModelSelect();
        emitModelChanged();
      }
    });

    ctx.log.info("Settings loaded.", { model: settingsState.activeModel, models: settingsState.availableModels.length });
  },
};
