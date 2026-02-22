/**
 * Settings event wiring — debounced auto-save for all form inputs.
 */

import type { Disposable } from "../../kernel/types.ts";
import { timers } from "./deps.ts";
import { getSettingsDom } from "./dom.ts";
import { saveTopLevelSettings, saveModelOverrides, handleResetModelOverrides, handleModelChange } from "./form.ts";

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
  dom.model.addEventListener("change", () => handleModelChange(dom.model.value));
  dom.systemPrompt.addEventListener("input", () => scheduleSettingsSave(600));
  dom.maxOutputTokens.addEventListener("input", () => scheduleSettingsSave(400));
  dom.contextLength.addEventListener("input", () => scheduleSettingsSave(400));
  dom.autoTitle.addEventListener("change", () => saveTopLevelSettings());
  dom.temperature.addEventListener("input", () => scheduleOverridesSave(400));
  dom.topP.addEventListener("input", () => scheduleOverridesSave(400));
  dom.topK.addEventListener("input", () => scheduleOverridesSave(400));
  dom.resetModel.addEventListener("click", () => handleResetModelOverrides());
}
