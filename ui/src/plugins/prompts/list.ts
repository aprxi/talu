/**
 * Prompts list rendering — renders sorted prompt list in the sidebar.
 */

import { CHECK_CIRCLE_ICON as ICON_CHECK_CIRCLE, CIRCLE_ICON as ICON_CIRCLE } from "../../icons.ts";
import { getPromptsDom } from "./dom.ts";
import { promptsState } from "./state.ts";

export function renderList(): void {
  const { listEl } = getPromptsDom();
  listEl.innerHTML = "";

  if (promptsState.prompts.length === 0) {
    const empty = document.createElement("div");
    empty.className = "prompts-empty-hint";
    empty.textContent = "No prompts yet";
    listEl.appendChild(empty);
    return;
  }

  const sorted = [...promptsState.prompts].sort((a, b) => {
    if (a.id === promptsState.defaultId) return -1;
    if (b.id === promptsState.defaultId) return 1;
    return a.name.localeCompare(b.name);
  });

  let addedSeparator = false;
  for (let i = 0; i < sorted.length; i++) {
    const p = sorted[i];
    if (!p) continue;
    const isDefault = p.id === promptsState.defaultId;

    if (!addedSeparator && promptsState.defaultId && !isDefault && i > 0) {
      const sep = document.createElement("div");
      sep.className = "prompts-list-separator";
      listEl.appendChild(sep);
      addedSeparator = true;
    }

    const item = document.createElement("button");
    item.className = promptsState.selectedId === p.id ? "prompt-item active" : "prompt-item";
    item.dataset.promptId = p.id;

    const nameSpan = document.createElement("span");
    nameSpan.className = "prompt-item-name truncate";
    nameSpan.textContent = p.name;
    item.appendChild(nameSpan);

    const defaultBtn = document.createElement("button");
    defaultBtn.className = isDefault ? "prompt-default-btn active" : "prompt-default-btn";
    defaultBtn.title = isDefault ? "Remove as default" : "Set as default";
    defaultBtn.innerHTML = isDefault ? ICON_CHECK_CIRCLE : ICON_CIRCLE;
    defaultBtn.dataset.action = "toggle-default";
    defaultBtn.dataset.promptId = p.id;
    item.appendChild(defaultBtn);

    listEl.appendChild(item);
  }
}
