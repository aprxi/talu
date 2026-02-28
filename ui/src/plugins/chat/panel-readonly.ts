import { getChatDom } from "./dom.ts";
import { getChatPanelDom } from "./chat-panel-dom.ts";
import { chatState } from "./state.ts";
import { getModelsService, getPromptsService, format, layout } from "./deps.ts";
import { syncRightPanelParams } from "./panel-params.ts";
import { escapeHtml } from "../../utils/helpers.ts";
import type { GenerationSettings, UsageStats } from "../../types.ts";

/** Track if panel is in read-only mode (showing past message params). */
let panelReadOnlyMode = false;

export function isPanelReadOnly(): boolean {
  return panelReadOnlyMode;
}

/** Show generation params in the panel (read-only mode for past messages). */
export function showReadOnlyParams(
  gen: GenerationSettings | null,
  usage: UsageStats | null,
): void {
  const pd = getChatPanelDom();
  if (!gen) return;

  panelReadOnlyMode = true;

  // Set values from the past message's generation params
  if (gen.model) {
    const modelOption = Array.from(pd.panelModel.options).find(opt => opt.value === gen.model);
    if (modelOption) {
      pd.panelModel.value = gen.model;
    }
  }
  pd.panelTemperature.value = gen.temperature?.toString() ?? "";
  pd.panelTopP.value = gen.top_p?.toString() ?? "";
  pd.panelTopK.value = gen.top_k?.toString() ?? "";
  pd.panelMinP.value = gen.min_p?.toString() ?? "";
  pd.panelMaxOutputTokens.value = gen.max_output_tokens?.toString() ?? "";
  pd.panelRepetitionPenalty.value = gen.repetition_penalty?.toString() ?? "";
  pd.panelSeed.value = gen.seed?.toString() ?? "";

  // Disable all inputs
  pd.panelModel.disabled = true;
  pd.panelTemperature.disabled = true;
  pd.panelTopP.disabled = true;
  pd.panelTopK.disabled = true;
  pd.panelMinP.disabled = true;
  pd.panelMaxOutputTokens.disabled = true;
  pd.panelRepetitionPenalty.disabled = true;
  pd.panelSeed.disabled = true;

  // Add visual indicator
  pd.root.classList.add("read-only");

  // Update the Info section with stats
  if (usage) {
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
    pd.panelChatInfo.innerHTML = rows.join("\n");
  }

  // Open the panel with the chat content
  showChatPanel();

  // Activate tuning button
  const tuningBtn = getChatDom().transcriptContainer.querySelector<HTMLButtonElement>('[data-action="toggle-tuning"]');
  if (tuningBtn) {
    tuningBtn.classList.add("active");
  }
}

/** Restore panel to editable mode with current settings. */
export function restoreEditableParams(): void {
  const pd = getChatPanelDom();
  if (!panelReadOnlyMode) return;

  panelReadOnlyMode = false;

  // Re-enable all inputs
  pd.panelModel.disabled = false;
  pd.panelTemperature.disabled = false;
  pd.panelTopP.disabled = false;
  pd.panelTopK.disabled = false;
  pd.panelMinP.disabled = false;
  pd.panelMaxOutputTokens.disabled = false;
  pd.panelRepetitionPenalty.disabled = false;
  pd.panelSeed.disabled = false;

  // Remove visual indicator
  pd.root.classList.remove("read-only");

  // Restore the Info section to show chat info
  if (chatState.activeChat) {
    const createdStr = format.dateTime(chatState.activeChat.created_at);

    let infoHtml = `<div class="info-row"><span class="info-label">Created</span><span class="info-value">${createdStr}</span></div>`;

    if (chatState.activeChat.parent_session_id) {
      const shortId = chatState.activeChat.parent_session_id.slice(0, 8);
      infoHtml += `<div class="info-row"><span class="info-label">Forked from</span><span class="info-value mono">${shortId}</span></div>`;
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

    pd.panelChatInfo.innerHTML = infoHtml;
  }

  // Restore right panel to current settings values
  const modelId = getModelsService()?.getActiveModel() ?? "";
  syncRightPanelParams(modelId);
}

export function hideChatPanel(): void {
  layout.hidePanel("chat");
  // Reset the tuning button appearance if visible
  const tuningBtn = getChatDom().transcriptContainer.querySelector<HTMLButtonElement>('[data-action="toggle-tuning"]');
  if (tuningBtn) {
    tuningBtn.classList.remove("active");
  }
}

/** Open the chat panel via the unified app panel. */
function showChatPanel(): void {
  const pd = getChatPanelDom();
  layout.showPanel({
    title: "Chat",
    content: pd.root,
    owner: "chat",
    onHide: () => {
      if (panelReadOnlyMode) {
        restoreEditableParams();
      }
      // Reset the tuning button appearance
      const tuningBtn = getChatDom().transcriptContainer.querySelector<HTMLButtonElement>('[data-action="toggle-tuning"]');
      if (tuningBtn) {
        tuningBtn.classList.remove("active");
      }
    },
  });
}

/** Open the panel in editable mode. Only X closes it. */
export function handleToggleTuning(btn: HTMLButtonElement): void {
  // Always restore editable mode
  if (panelReadOnlyMode) {
    restoreEditableParams();
  }

  // Show the chat panel
  showChatPanel();
  btn.classList.add("active");
}
