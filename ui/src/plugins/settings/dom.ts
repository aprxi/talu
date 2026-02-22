/**
 * Settings plugin DOM cache — queries elements within the plugin's
 * container (set via initSettingsDom).
 */

export interface SettingsDom {
  model: HTMLSelectElement;
  systemPrompt: HTMLTextAreaElement;
  maxOutputTokens: HTMLInputElement;
  contextLength: HTMLInputElement;
  autoTitle: HTMLInputElement;
  temperature: HTMLInputElement;
  topP: HTMLInputElement;
  topK: HTMLInputElement;
  temperatureDefault: HTMLElement;
  topPDefault: HTMLElement;
  topKDefault: HTMLElement;
  modelLabel: HTMLElement;
  resetModel: HTMLButtonElement;
  status: HTMLElement;
}

let root: HTMLElement;
let cached: SettingsDom | null = null;

/** Set the root container for DOM queries. Must be called after buildSettingsDOM(). */
export function initSettingsDom(container: HTMLElement): void {
  root = container;
  cached = null;
}

export function getSettingsDom(): SettingsDom {
  if (cached) return cached;

  const q = <T extends HTMLElement>(id: string) => root.querySelector<T>(`#${id}`)!;

  cached = {
    model: q<HTMLSelectElement>("sp-model"),
    systemPrompt: q<HTMLTextAreaElement>("sp-system-prompt"),
    maxOutputTokens: q<HTMLInputElement>("sp-max-output-tokens"),
    contextLength: q<HTMLInputElement>("sp-context-length"),
    autoTitle: q<HTMLInputElement>("sp-auto-title"),
    temperature: q<HTMLInputElement>("sp-temperature"),
    topP: q<HTMLInputElement>("sp-top-p"),
    topK: q<HTMLInputElement>("sp-top-k"),
    temperatureDefault: q("sp-temperature-default"),
    topPDefault: q("sp-top-p-default"),
    topKDefault: q("sp-top-k-default"),
    modelLabel: q("sp-model-label"),
    resetModel: q<HTMLButtonElement>("sp-reset-model"),
    status: q("sp-status"),
  };

  return cached;
}
