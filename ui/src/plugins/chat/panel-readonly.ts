import { getChatDom } from "./dom.ts";
import { chatState } from "./state.ts";
import { getModelsService, getPromptsService, format } from "./deps.ts";
import { syncRightPanelParams } from "./panel-params.ts";
import { escapeHtml } from "../../utils/helpers.ts";
import type { GenerationSettings, UsageStats } from "../../types.ts";

/** Track if panel is in read-only mode (showing past message params). */
let panelReadOnlyMode = false;

export function isPanelReadOnly(): boolean {
  return panelReadOnlyMode;
}

/** Show generation params in the right panel (read-only mode for past messages). */
export function showReadOnlyParams(
  gen: GenerationSettings | null,
  usage: UsageStats | null,
): void {
  const dom = getChatDom();
  if (!gen) return;

  // Open right panel if hidden
  if (dom.rightPanel.classList.contains("hidden")) {
    dom.rightPanel.classList.remove("hidden");
    dom.rightPanel.classList.add("flex");
    const tuningBtn = dom.transcriptContainer.querySelector<HTMLButtonElement>('[data-action="toggle-tuning"]');
    if (tuningBtn) {
      tuningBtn.classList.add("active");
    }
  }

  panelReadOnlyMode = true;

  // Set values from the past message's generation params
  if (gen.model) {
    const modelOption = Array.from(dom.panelModel.options).find(opt => opt.value === gen.model);
    if (modelOption) {
      dom.panelModel.value = gen.model;
    }
  }
  dom.panelTemperature.value = gen.temperature?.toString() ?? "";
  dom.panelTopP.value = gen.top_p?.toString() ?? "";
  dom.panelTopK.value = gen.top_k?.toString() ?? "";
  dom.panelMinP.value = gen.min_p?.toString() ?? "";
  dom.panelMaxOutputTokens.value = gen.max_output_tokens?.toString() ?? "";
  dom.panelRepetitionPenalty.value = gen.repetition_penalty?.toString() ?? "";
  dom.panelSeed.value = gen.seed?.toString() ?? "";

  // Disable all inputs
  dom.panelModel.disabled = true;
  dom.panelTemperature.disabled = true;
  dom.panelTopP.disabled = true;
  dom.panelTopK.disabled = true;
  dom.panelMinP.disabled = true;
  dom.panelMaxOutputTokens.disabled = true;
  dom.panelRepetitionPenalty.disabled = true;
  dom.panelSeed.disabled = true;

  // Add visual indicator
  dom.rightPanel.classList.add("read-only");

  // Update the Info section with stats
  if (usage) {
    const panelChatInfo = dom.panelChatInfo;
    const rows: string[] = [];
    rows.push(`<div class="info-row"><span class="info-label">Output tokens</span><span class="info-value">${usage.output_tokens}</span></div>`);
    if (usage.input_tokens) {
      rows.push(`<div class="info-row"><span class="info-label">Input tokens</span><span class="info-value">${usage.input_tokens}</span></div>`);
    }
    if (usage.tokens_per_second) {
      rows.push(`<div class="info-row"><span class="info-label">Speed</span><span class="info-value">${usage.tokens_per_second} tok/s</span></div>`);
    }
    if (usage.duration_ms) {
      rows.push(`<div class="info-row"><span class="info-label">Duration</span><span class="info-value">${(usage.duration_ms / 1000).toFixed(2)}s</span></div>`);
    }
    panelChatInfo.innerHTML = rows.join("\n");
  }
}

/** Restore panel to editable mode with current settings. */
export function restoreEditableParams(): void {
  const dom = getChatDom();
  if (!panelReadOnlyMode) return;

  panelReadOnlyMode = false;

  // Re-enable all inputs
  dom.panelModel.disabled = false;
  dom.panelTemperature.disabled = false;
  dom.panelTopP.disabled = false;
  dom.panelTopK.disabled = false;
  dom.panelMinP.disabled = false;
  dom.panelMaxOutputTokens.disabled = false;
  dom.panelRepetitionPenalty.disabled = false;
  dom.panelSeed.disabled = false;

  // Remove visual indicator
  dom.rightPanel.classList.remove("read-only");

  // Restore the Info section to show chat info
  if (chatState.activeChat) {
    const createdStr = format.dateTime(chatState.activeChat.created_at);

    let infoHtml = `<div class="info-row"><span class="info-label">Created</span><span id="panel-info-created" class="info-value">${createdStr}</span></div>`;

    if (chatState.activeChat.parent_session_id) {
      const shortId = chatState.activeChat.parent_session_id.slice(0, 8);
      infoHtml += `<div id="panel-info-forked-row" class="info-row"><span class="info-label">Forked from</span><span id="panel-info-forked" class="info-value mono">${shortId}</span></div>`;
    }

    // Show source prompt if this conversation was created from one
    if (chatState.activeChat.source_doc_id) {
      const promptName = getPromptsService()?.getPromptNameById(chatState.activeChat.source_doc_id) ?? null;
      if (promptName) {
        infoHtml += `<div class="info-row"><span class="info-label">Prompt</span><span class="info-value">${escapeHtml(promptName)}</span></div>`;
      } else {
        const shortId = chatState.activeChat.source_doc_id.slice(0, 8);
        infoHtml += `<div class="info-row"><span class="info-label">Prompt</span><span class="info-value mono">${shortId}...</span></div>`;
      }
    }

    dom.panelChatInfo.innerHTML = infoHtml;
  }

  // Restore right panel to current settings values
  const modelId = getModelsService()?.getActiveModel() ?? "";
  syncRightPanelParams(modelId);
}

export function hideRightPanel(): void {
  const dom = getChatDom();
  dom.rightPanel.classList.add("hidden");
  dom.rightPanel.classList.remove("flex");
  // Reset the tuning button appearance if visible
  const tuningBtn = dom.transcriptContainer.querySelector<HTMLButtonElement>('[data-action="toggle-tuning"]');
  if (tuningBtn) {
    tuningBtn.classList.remove("active");
  }
}

/** Open the right panel in editable mode. Only X closes it. */
export function handleToggleTuning(btn: HTMLButtonElement): void {
  const dom = getChatDom();
  // Always restore editable mode
  if (panelReadOnlyMode) {
    restoreEditableParams();
  }

  // Always open (never close via this button)
  dom.rightPanel.classList.remove("hidden");
  dom.rightPanel.classList.add("flex");
  btn.classList.add("active");
}
