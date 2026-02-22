/**
 * Settings form — population, save operations, model handling.
 */

import type { Settings, SettingsPatch, ApiResult } from "../../types.ts";
import { populateModelSelect } from "../../render/helpers.ts";
import { api, timers } from "./deps.ts";
import { settingsState, notifyChange, emitModelChanged } from "./state.ts";
import { getSettingsDom } from "./dom.ts";

export function populateForm(s: Settings): void {
  const dom = getSettingsDom();
  dom.systemPrompt.value = s.system_prompt ?? "";
  dom.maxOutputTokens.value = s.max_output_tokens != null ? String(s.max_output_tokens) : "";
  dom.contextLength.value = s.context_length != null ? String(s.context_length) : "";
  dom.autoTitle.checked = s.auto_title;
}

export function populateLocalModelSelect(): void {
  const dom = getSettingsDom();
  populateModelSelect(dom.model, settingsState.availableModels, settingsState.activeModel);
}

export function showModelParams(modelId: string): void {
  const dom = getSettingsDom();
  dom.modelLabel.textContent = modelId ? `(${modelId})` : "";
  const entry = settingsState.availableModels.find((m) => m.id === modelId);
  if (!entry) return;
  const d = entry.defaults;
  const o = entry.overrides;

  dom.temperature.placeholder = String(d.temperature);
  dom.temperature.value = o.temperature != null ? String(o.temperature) : "";
  dom.temperatureDefault.textContent = `default: ${d.temperature}`;

  dom.topP.placeholder = String(d.top_p);
  dom.topP.value = o.top_p != null ? String(o.top_p) : "";
  dom.topPDefault.textContent = `default: ${d.top_p}`;

  dom.topK.placeholder = String(d.top_k);
  dom.topK.value = o.top_k != null ? String(o.top_k) : "";
  dom.topKDefault.textContent = `default: ${d.top_k}`;
}

function showSaveResult(result: ApiResult<Settings>): void {
  const dom = getSettingsDom();
  if (!result.ok) {
    dom.status.textContent = result.error ?? "Failed to save";
    dom.status.className = "text-xs text-danger";
    return;
  }
  if (result.data?.model) {
    settingsState.activeModel = result.data.model;
  }
  dom.status.textContent = "Saved";
  dom.status.className = "text-xs text-success";
  timers.setTimeout(() => {
    dom.status.textContent = "";
  }, 1500);
}

export async function saveTopLevelSettings(): Promise<void> {
  const dom = getSettingsDom();
  dom.status.textContent = "Saving...";
  dom.status.className = "text-xs text-text-subtle";

  const maxTok = dom.maxOutputTokens.value.trim();
  const ctxLen = dom.contextLength.value.trim();

  const patch: SettingsPatch = {
    system_prompt: dom.systemPrompt.value.trim() || null,
    max_output_tokens: maxTok ? parseInt(maxTok, 10) : null,
    context_length: ctxLen ? parseInt(ctxLen, 10) : null,
    auto_title: dom.autoTitle.checked,
  };

  const result = await api.patchSettings(patch);
  showSaveResult(result);
}

export async function saveModelOverrides(): Promise<void> {
  const dom = getSettingsDom();
  dom.status.textContent = "Saving...";
  dom.status.className = "text-xs text-text-subtle";

  const patch: SettingsPatch = {
    model_overrides: {
      temperature: dom.temperature.value.trim() ? parseFloat(dom.temperature.value) : null,
      top_p: dom.topP.value.trim() ? parseFloat(dom.topP.value) : null,
      top_k: dom.topK.value.trim() ? parseInt(dom.topK.value, 10) : null,
    },
  };

  const result = await api.patchSettings(patch);
  if (result.ok && result.data) {
    settingsState.availableModels = result.data.available_models ?? [];
    emitModelChanged();
  }
  showSaveResult(result);
}

export async function handleResetModelOverrides(): Promise<void> {
  if (!settingsState.activeModel) return;
  const dom = getSettingsDom();
  dom.status.textContent = "Resetting...";
  dom.status.className = "text-xs text-text-subtle";

  const result = await api.resetModelOverrides(settingsState.activeModel);
  if (result.ok && result.data) {
    settingsState.availableModels = result.data.available_models ?? [];
    showModelParams(settingsState.activeModel);
    emitModelChanged();
  }
  showSaveResult(result);
}

export function handleModelChange(newModel: string): void {
  if (!newModel) return;
  settingsState.activeModel = newModel;
  showModelParams(newModel);
  emitModelChanged();
  notifyChange();
  api.patchSettings({ model: newModel });
}
