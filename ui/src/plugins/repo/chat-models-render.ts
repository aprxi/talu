/** Render the Chat Models section in the providers tab. */

import { CLOSE_ICON } from "../../icons.ts";
import { el } from "../../render/helpers.ts";
import { renderEmptyState } from "../../render/common.ts";
import { getRepoDom } from "./dom.ts";
import { repoState } from "./state.ts";
import { removeChatModel, moveChatModel } from "./chat-models-data.ts";

export function renderChatModels(): void {
  const list = getRepoDom().chatModelsList;
  list.innerHTML = "";

  if (repoState.chatModels.length === 0) {
    list.appendChild(renderEmptyState("No models selected. Browse a provider below."));
    return;
  }

  for (const modelId of repoState.chatModels) {
    list.appendChild(buildChatModelItem(modelId));
  }
}

function buildChatModelItem(modelId: string): HTMLElement {
  const row = el("div", "repo-chat-model-item");
  row.dataset["modelId"] = modelId;

  // Move buttons
  const moveUp = el("button", "btn btn-ghost repo-chat-model-move", "\u25B2");
  moveUp.title = "Move up";
  moveUp.dataset["action"] = "cm-move-up";
  row.appendChild(moveUp);

  const moveDown = el("button", "btn btn-ghost repo-chat-model-move", "\u25BC");
  moveDown.title = "Move down";
  moveDown.dataset["action"] = "cm-move-down";
  row.appendChild(moveDown);

  // Model name (strip provider:: prefix for display)
  const sep = modelId.indexOf("::");
  const displayName = sep >= 0 ? modelId.substring(sep + 2) : modelId;
  const providerName = sep >= 0 ? modelId.substring(0, sep) : "local";

  const nameEl = el("span", "repo-chat-model-name", displayName);
  nameEl.title = modelId;
  row.appendChild(nameEl);

  // Provider badge
  const badge = el("span", "repo-chat-model-provider", providerName);
  row.appendChild(badge);

  // Remove button
  const removeBtn = el("button", "btn btn-ghost repo-chat-model-remove");
  removeBtn.innerHTML = CLOSE_ICON;
  removeBtn.title = "Remove";
  removeBtn.dataset["action"] = "cm-remove";
  row.appendChild(removeBtn);

  return row;
}

export function wireChatModelEvents(container: HTMLElement): void {
  container.addEventListener("click", (e) => {
    const target = e.target as HTMLElement;
    const actionEl = target.closest<HTMLElement>("[data-action]");
    if (!actionEl) return;
    const action = actionEl.dataset["action"];

    if (action === "cm-move-up" || action === "cm-move-down" || action === "cm-remove") {
      const row = target.closest<HTMLElement>("[data-model-id]");
      if (!row) return;
      const modelId = row.dataset["modelId"]!;

      if (action === "cm-remove") {
        removeChatModel(modelId);
      } else {
        moveChatModel(modelId, action === "cm-move-up" ? "up" : "down");
      }
    }
  });
}
