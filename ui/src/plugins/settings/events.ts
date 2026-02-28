/**
 * Settings event wiring — debounced auto-save for all form inputs.
 */

import type { Disposable } from "../../kernel/types.ts";
import { timers, mode, theme } from "./deps.ts";
import { getSettingsDom } from "./dom.ts";
import { saveTopLevelSettings, saveModelOverrides, handleResetModelOverrides, handleModelChange } from "./form.ts";
import {
  populateThemeSelects,
  handleDarkSlotChange,
  handleLightSlotChange,
  createCustomTheme,
  handleImport,
} from "./theme-editor.ts";

export function wireEvents(): void {
  let debounceHandle: Disposable | null = null;

  function scheduleSettingsSave(delay: number): void {
    debounceHandle?.dispose();
    debounceHandle = timers.setTimeout(() => {
      debounceHandle = null;
      saveTopLevelSettings();
    }, delay);
  }

  function scheduleOverridesSave(delay: number): void {
    debounceHandle?.dispose();
    debounceHandle = timers.setTimeout(() => {
      debounceHandle = null;
      saveModelOverrides();
    }, delay);
  }

  const dom = getSettingsDom();

  // Model.
  dom.model.addEventListener("change", () => handleModelChange(dom.model.value));
  dom.systemPromptEnabled.addEventListener("change", () => saveTopLevelSettings());
  dom.openPrompts.addEventListener("click", () => mode.switch("conversations"));
  dom.maxOutputTokens.addEventListener("input", () => scheduleSettingsSave(400));
  dom.contextLength.addEventListener("input", () => scheduleSettingsSave(400));
  dom.autoTitle.addEventListener("change", () => saveTopLevelSettings());
  dom.temperature.addEventListener("input", () => scheduleOverridesSave(400));
  dom.topP.addEventListener("input", () => scheduleOverridesSave(400));
  dom.topK.addEventListener("input", () => scheduleOverridesSave(400));
  dom.resetModel.addEventListener("click", () => handleResetModelOverrides());

  // Theme mode slots.
  dom.themeDarkSelect.addEventListener("change", () => handleDarkSlotChange(dom.themeDarkSelect.value));
  dom.themeLightSelect.addEventListener("change", () => handleLightSlotChange(dom.themeLightSelect.value));
  dom.themeNewBtn.addEventListener("click", () => void createCustomTheme());
  dom.themeImportBtn.addEventListener("click", () => handleImport());

  // Sync dropdowns when theme changes externally (e.g. sun/moon toggle).
  theme.onChange(() => populateThemeSelects());
}
