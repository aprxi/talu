import { getChatDom } from "./dom.ts";
import { getChatPanelDom } from "./chat-panel-dom.ts";
import { api, notifications, getModelsService, format, timers } from "./deps.ts";
import type { Conversation, CreateResponseRequest, SettingsPatch } from "../../types.ts";
import type { Disposable } from "../../kernel/types.ts";

/** Get current sampling parameters from the panel inputs. */
export function getSamplingParams(): Partial<CreateResponseRequest> {
  const pd = getChatPanelDom();
  const temp = pd.panelTemperature.value.trim();
  const topP = pd.panelTopP.value.trim();
  const topK = pd.panelTopK.value.trim();
  const minP = pd.panelMinP.value.trim();
  const maxTok = pd.panelMaxOutputTokens.value.trim();
  const repPen = pd.panelRepetitionPenalty.value.trim();
  const seed = pd.panelSeed.value.trim();

  const params: Partial<CreateResponseRequest> = {};
  if (temp) params.temperature = parseFloat(temp);
  if (topP) params.top_p = parseFloat(topP);
  if (maxTok) params.max_output_tokens = parseInt(maxTok, 10);
  return params;
}

/** Save sampling overrides from the panel inputs. */
export async function savePanelOverrides(): Promise<void> {
  const pd = getChatPanelDom();
  const temp = pd.panelTemperature.value.trim();
  const topP = pd.panelTopP.value.trim();
  const topK = pd.panelTopK.value.trim();
  const minP = pd.panelMinP.value.trim();
  const maxTok = pd.panelMaxOutputTokens.value.trim();
  const repPen = pd.panelRepetitionPenalty.value.trim();
  const seed = pd.panelSeed.value.trim();

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

/** Sync panel sampling params for a given model. */
export function syncRightPanelParams(modelId: string): void {
  const pd = getChatPanelDom();
  const models = getModelsService()?.getAvailableModels() ?? [];
  const entry = models.find((m) => m.id === modelId);
  if (!entry) return;

  const d = entry.defaults;
  const o = entry.overrides;

  pd.panelTemperature.placeholder = String(d.temperature);
  pd.panelTemperature.value = o.temperature != null ? String(o.temperature) : "";
  pd.panelTemperatureDefault.textContent = `Default: ${d.temperature}`;

  pd.panelTopP.placeholder = String(d.top_p);
  pd.panelTopP.value = o.top_p != null ? String(o.top_p) : "";
  pd.panelTopPDefault.textContent = `Default: ${d.top_p}`;

  pd.panelTopK.placeholder = String(d.top_k);
  pd.panelTopK.value = o.top_k != null ? String(o.top_k) : "";
  pd.panelTopKDefault.textContent = `Default: ${d.top_k}`;
}

/** Update panel chat info section. */
export function updatePanelChatInfo(chat: Conversation | null): void {
  const pd = getChatPanelDom();
  if (!chat) return;

  pd.panelInfoCreated.textContent = format.dateTime(chat.created_at);

  if (chat.parent_session_id) {
    pd.panelInfoForkedRow.classList.remove("hidden");
    pd.panelInfoForked.textContent = chat.parent_session_id.slice(0, 8) + "...";
  } else {
    pd.panelInfoForkedRow.classList.add("hidden");
  }
}

/** Wire up panel event handlers (model change, sampling saves). */
export function setupPanelEvents(): void {
  const dom = getChatDom();
  const pd = getChatPanelDom();

  // Model selectors — sync each other and notify settings service
  dom.welcomeModel.addEventListener("change", () => {
    const modelId = dom.welcomeModel.value;
    pd.panelModel.value = modelId;
    getModelsService()?.setActiveModel(modelId);
    syncRightPanelParams(modelId);
  });

  pd.panelModel.addEventListener("change", () => {
    const modelId = pd.panelModel.value;
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
  pd.panelTemperature.addEventListener("input", scheduleSave);
  pd.panelTopP.addEventListener("input", scheduleSave);
  pd.panelTopK.addEventListener("input", scheduleSave);
  pd.panelMinP.addEventListener("input", scheduleSave);
  pd.panelMaxOutputTokens.addEventListener("input", scheduleSave);
  pd.panelRepetitionPenalty.addEventListener("input", scheduleSave);
  pd.panelSeed.addEventListener("input", scheduleSave);
}
