/**
 * Settings plugin local state — shared across all settings submodules.
 */

import type { ModelEntry } from "../../types.ts";
import type { Disposable } from "../../kernel/types.ts";
import { events } from "./deps.ts";

export interface CustomTheme {
  id: string;
  name: string;
  category: "dark" | "light";
  tokens: Record<string, string>;
}

export interface SettingsState {
  activeModel: string;
  availableModels: ModelEntry[];
  changeHandlers: Set<() => void>;
  systemPromptEnabled: boolean;
  chatModelsActive: boolean;
  customThemes: CustomTheme[];
  /** Disposables for registered custom themes (keyed by theme ID). */
  themeDisposables: Map<string, Disposable>;
}

export const settingsState: SettingsState = {
  activeModel: "",
  availableModels: [],
  changeHandlers: new Set(),
  systemPromptEnabled: true,
  chatModelsActive: false,
  customThemes: [],
  themeDisposables: new Map(),
};

/** Notify internal change listeners (e.g. chat plugin's model service subscription). */
export function notifyChange(): void {
  for (const handler of settingsState.changeHandlers) {
    try {
      handler();
    } catch {
      // Swallow — handlers should not break the plugin.
    }
  }
}

/** Emit model change event so other plugins (chat) can sync their selectors. */
export function emitModelChanged(): void {
  events.emit("model.changed", {
    modelId: settingsState.activeModel,
    availableModels: settingsState.availableModels,
  });
}
