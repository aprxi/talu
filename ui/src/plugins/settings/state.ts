/**
 * Settings plugin local state — shared across all settings submodules.
 */

import type { ModelEntry } from "../../types.ts";
import { events } from "./deps.ts";

export interface SettingsState {
  activeModel: string;
  availableModels: ModelEntry[];
  changeHandlers: Set<() => void>;
  systemPromptEnabled: boolean;
}

export const settingsState: SettingsState = {
  activeModel: "",
  availableModels: [],
  changeHandlers: new Set(),
  systemPromptEnabled: true,
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
