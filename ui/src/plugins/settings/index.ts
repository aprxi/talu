/**
 * Settings plugin — global model, sampling configuration, and theme management.
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
import { loadCustomThemes, populateThemeSelects } from "./theme-editor.ts";
import { navigate, onRouteChange } from "../../kernel/system/router.ts";
import type { ModelEntry } from "../../types.ts";

/** Collapse quant variants into one entry per family.
 *  Managed models with TQ suffixes (e.g. Qwen/Qwen3-0.6B-TQ4) share
 *  the same parent (Qwen/Qwen3-0.6B). Only the first variant per family
 *  appears in the selector, with the family name as display_name. */
function deduplicateByFamily(models: ModelEntry[]): ModelEntry[] {
  const families = new Map<string, { entry: ModelEntry; variants: { id: string; label: string }[] }>();
  for (const m of models) {
    let familyKey = m.id;
    if (m.source === "managed") {
      const stripped = m.id.replace(/-TQ\d+(?:_\d+|-G\d+)?$/, "");
      if (stripped !== m.id) familyKey = stripped;
    }
    const label = familyKey !== m.id
      ? m.id.slice(familyKey.length + 1)
      : (m.id.split("/").pop() ?? m.id);
    if (!families.has(familyKey)) {
      families.set(familyKey, {
        entry: { ...m, display_name: familyKey !== m.id ? familyKey : undefined },
        variants: [],
      });
    }
    families.get(familyKey)!.variants.push({ id: m.id, label });
  }
  return [...families.values()].map(({ entry, variants }) => ({
    ...entry,
    variants,
  }));
}

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
    initSettingsDeps({
      api,
      events: ctx.events,
      timers: ctx.timers,
      mode: ctx.mode,
      layout: ctx.layout,
      theme: ctx.theme,
      storage: ctx.storage,
      download: ctx.download,
      notifications: ctx.notifications,
      dialogs: ctx.dialogs,
    });

    buildSettingsDOM(ctx.container);
    initSettingsDom(ctx.container);
    wireEvents();

    // Tab switching: show/hide settings tab pages via subnav buttons.
    const SETTINGS_TABS = ["model", "appearance", "generation", "bench"] as const;
    type SettingsTab = (typeof SETTINGS_TABS)[number];

    const isSettingsTab = (tab: string): tab is SettingsTab =>
      (SETTINGS_TABS as readonly string[]).includes(tab);

    const syncSettingsSubnav = (tab: SettingsTab): void => {
      const buttons = document.querySelectorAll<HTMLElement>(
        '.subnav-group[data-subnav-group="settings"] .subnav-btn[data-nav-tab]',
      );
      buttons.forEach((button) => {
        button.classList.toggle("active", button.getAttribute("data-nav-tab") === tab);
      });
    };

    const showSettingsTab = (tab: SettingsTab): void => {
      for (const t of SETTINGS_TABS) {
        const el = ctx.container.querySelector<HTMLElement>(`[data-settings-tab="${t}"]`);
        if (el) el.style.display = t === tab ? "" : "none";
      }
      const settingsContent = ctx.container.querySelector<HTMLElement>("#sp-settings-content");
      settingsContent?.classList.toggle("is-bench-active", tab === "bench");
      syncSettingsSubnav(tab);
    };

    ctx.events.on<{ tab: string }>("subnav.tab", ({ tab }) => {
      if (!isSettingsTab(tab)) return;
      showSettingsTab(tab);
      navigate({ mode: "settings", sub: tab === "model" ? null : tab, resource: null }, { replace: true });
    });

    // Route-driven tab switching (Back/Forward + deep links).
    ctx.subscriptions.add(onRouteChange((route) => {
      if (route.mode !== "settings") return;
      const tab: SettingsTab = isSettingsTab(route.sub ?? "")
        ? (route.sub as SettingsTab)
        : "model";
      showSettingsTab(tab);
    }));

    // Load persisted custom themes and register with kernel.
    await loadCustomThemes();
    populateThemeSelects();

    // Fetch settings from API.
    const result = await api.getSettings();
    if (!result.ok || !result.data) {
      ctx.notifications.error(result.error ?? "Failed to load settings");
      return;
    }

    settingsState.availableModels = deduplicateByFamily(result.data.available_models ?? []);
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
        settingsState.availableModels = deduplicateByFamily(res.data.available_models ?? []);
        populateLocalModelSelect();
        emitModelChanged();
      }
    });

    // Chat models (user-curated list from providers tab) replaces available models.
    // Uses family dedup so the model selector shows one entry per family.
    ctx.events.on<{ models: string[]; families?: { familyId: string; defaultVariant: string; variants?: { id: string; label: string; size_bytes?: number }[] }[] }>(
      "repo.chatModels.changed",
      ({ models, families }) => {
        if (models.length === 0) {
          settingsState.chatModelsActive = false;
          return;
        }
        settingsState.chatModelsActive = true;

        // One entry per family (display name = family, value = default variant).
        const localEntries = (families ?? []).map((f) => ({
          id: f.defaultVariant,
          display_name: f.familyId !== f.defaultVariant ? f.familyId : undefined,
          source: "managed" as const,
          defaults: { temperature: 1.0, top_k: 50, top_p: 1.0, do_sample: true },
          overrides: {},
          variants: f.variants,
        }));
        // Remote models as individual entries.
        const remoteEntries = models
          .filter((id) => id.includes("::"))
          .map((id) => ({
            id,
            source: "hub" as const,
            defaults: { temperature: 1.0, top_k: 50, top_p: 1.0, do_sample: true },
            overrides: {},
          }));
        settingsState.availableModels = [...localEntries, ...remoteEntries];

        if (!settingsState.availableModels.some((m) => m.id === settingsState.activeModel) && settingsState.availableModels.length > 0) {
          settingsState.activeModel = settingsState.availableModels[0]!.id;
        }
        emitModelChanged();
      },
    );

    // Select a specific model variant (emitted by repo plugin variant pills).
    // The variant ID may not be in availableModels (which has one entry per family),
    // so we set it directly — the server handles any valid model ID.
    // Don't emit model.changed: variant switches within a family don't need to
    // re-populate model selects, and doing so can cascade a change event that
    // resets the active model back to the family default.
    ctx.events.on<{ modelId: string }>("repo.selectModel", ({ modelId }) => {
      settingsState.activeModel = modelId;
    });

    ctx.log.info("Settings loaded.", { model: settingsState.activeModel, models: settingsState.availableModels.length });
  },
};
