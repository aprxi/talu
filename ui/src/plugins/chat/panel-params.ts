import { getChatDom } from "./dom.ts";
import { api, notifications, getModelsService, format, timers } from "./deps.ts";
import { isPanelReadOnly, restoreEditableParams, hideRightPanel } from "./panel-readonly.ts";
import type { Conversation, CreateResponseRequest, SettingsPatch } from "../../types.ts";
import type { Disposable } from "../../kernel/types.ts";

/** Get current sampling parameters from the right panel inputs. */
export function getSamplingParams(): Partial<CreateResponseRequest> {
  const dom = getChatDom();
  const temp = dom.panelTemperature.value.trim();
  const topP = dom.panelTopP.value.trim();
  const topK = dom.panelTopK.value.trim();
  const minP = dom.panelMinP.value.trim();
  const maxTok = dom.panelMaxOutputTokens.value.trim();
  const repPen = dom.panelRepetitionPenalty.value.trim();
  const seed = dom.panelSeed.value.trim();

  const params: Partial<CreateResponseRequest> = {};
  if (temp) params.temperature = parseFloat(temp);
  if (topP) params.top_p = parseFloat(topP);
  if (maxTok) params.max_output_tokens = parseInt(maxTok, 10);
  return params;
}

/** Save sampling overrides from the right panel inputs. */
export async function savePanelOverrides(): Promise<void> {
  const dom = getChatDom();
  const temp = dom.panelTemperature.value.trim();
  const topP = dom.panelTopP.value.trim();
  const topK = dom.panelTopK.value.trim();
  const minP = dom.panelMinP.value.trim();
  const maxTok = dom.panelMaxOutputTokens.value.trim();
  const repPen = dom.panelRepetitionPenalty.value.trim();
  const seed = dom.panelSeed.value.trim();

  const patch: SettingsPatch = {
    model_overrides: {
      temperature: temp ? parseFloat(temp) : null,
      top_p: topP ? parseFloat(topP) : null,
      top_k: topK ? parseInt(topK, 10) : null,
      min_p: minP ? parseFloat(minP) : null,
      max_output_tokens: maxTok ? parseInt(maxTok, 10) : null,
      repetition_penalty: repPen ? parseFloat(repPen) : null,
      seed: seed ? parseInt(seed, 10) : null,
    },
  };

  const result = await api.patchSettings(patch);
  if (!result.ok) {
    notifications.error(result.error ?? "Failed to save overrides");
  }
}

/** Sync right panel sampling params for a given model. */
export function syncRightPanelParams(modelId: string): void {
  const dom = getChatDom();
  const models = getModelsService()?.getAvailableModels() ?? [];
  const entry = models.find((m) => m.id === modelId);
  if (!entry) return;

  const d = entry.defaults;
  const o = entry.overrides;

  dom.panelTemperature.placeholder = String(d.temperature);
  dom.panelTemperature.value = o.temperature != null ? String(o.temperature) : "";
  dom.panelTemperatureDefault.textContent = `Default: ${d.temperature}`;

  dom.panelTopP.placeholder = String(d.top_p);
  dom.panelTopP.value = o.top_p != null ? String(o.top_p) : "";
  dom.panelTopPDefault.textContent = `Default: ${d.top_p}`;

  dom.panelTopK.placeholder = String(d.top_k);
  dom.panelTopK.value = o.top_k != null ? String(o.top_k) : "";
  dom.panelTopKDefault.textContent = `Default: ${d.top_k}`;
}

/** Update right panel chat info section. */
export function updatePanelChatInfo(chat: Conversation | null): void {
  const dom = getChatDom();
  if (!chat) return;

  dom.panelInfoCreated.textContent = format.dateTime(chat.created_at);

  if (chat.parent_session_id) {
    dom.panelInfoForkedRow.classList.remove("hidden");
    dom.panelInfoForked.textContent = chat.parent_session_id.slice(0, 8) + "...";
  } else {
    dom.panelInfoForkedRow.classList.add("hidden");
  }
}

/** Wire up right panel event handlers (close button, model change, sampling saves). */
export function setupPanelEvents(): void {
  const dom = getChatDom();

  // Close button
  dom.closeRightPanelBtn.addEventListener("click", () => {
    if (isPanelReadOnly()) {
      restoreEditableParams();
    }
    hideRightPanel();
  });

  // Model selectors — sync each other and notify settings service
  dom.welcomeModel.addEventListener("change", () => {
    const modelId = dom.welcomeModel.value;
    dom.panelModel.value = modelId;
    getModelsService()?.setActiveModel(modelId);
    syncRightPanelParams(modelId);
  });

  dom.panelModel.addEventListener("change", () => {
    const modelId = dom.panelModel.value;
    dom.welcomeModel.value = modelId;
    getModelsService()?.setActiveModel(modelId);
    syncRightPanelParams(modelId);
  });

  // Sampling inputs — debounced save
  let saveDebounce: Disposable | null = null;
  const scheduleSave = () => {
    saveDebounce?.dispose();
    saveDebounce = timers.setTimeout(() => savePanelOverrides(), 400);
  };
  dom.panelTemperature.addEventListener("input", scheduleSave);
  dom.panelTopP.addEventListener("input", scheduleSave);
  dom.panelTopK.addEventListener("input", scheduleSave);
  dom.panelMinP.addEventListener("input", scheduleSave);
  dom.panelMaxOutputTokens.addEventListener("input", scheduleSave);
  dom.panelRepetitionPenalty.addEventListener("input", scheduleSave);
  dom.panelSeed.addEventListener("input", scheduleSave);
}
