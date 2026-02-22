/**
 * Prompts list rendering — renders sorted prompt list in the sidebar.
 */

import { CHECK_CIRCLE_ICON as ICON_CHECK_CIRCLE, CIRCLE_ICON as ICON_CIRCLE } from "../../icons.ts";
import { getPromptsDom } from "./dom.ts";
import { promptsState } from "./state.ts";

export function renderList(): void {
  const { listEl } = getPromptsDom();
  listEl.innerHTML = "";

  // Always render built-in prompt first (dimmed, non-removable).
  const builtin = promptsState.prompts.find((p) => p.id === promptsState.builtinId);
  if (builtin) {
    const isActiveDefault = promptsState.defaultId === null;
    const isSelected = promptsState.selectedId === builtin.id;

    const item = document.createElement("button");
    item.className = `prompt-item builtin${isSelected ? " active" : ""}`;
    item.dataset.promptId = builtin.id;

    const nameSpan = document.createElement("span");
    nameSpan.className = "prompt-item-name truncate";
    nameSpan.textContent = "Default";
    item.appendChild(nameSpan);

    // Clickable default button (dimmed styling handled by CSS).
    const defaultBtn = document.createElement("button");
    defaultBtn.className = isActiveDefault ? "prompt-default-btn builtin active" : "prompt-default-btn builtin";
    defaultBtn.title = isActiveDefault ? "Default prompt" : "Revert to default";
    defaultBtn.innerHTML = isActiveDefault ? ICON_CHECK_CIRCLE : ICON_CIRCLE;
    defaultBtn.dataset.action = "toggle-default";
    defaultBtn.dataset.promptId = builtin.id;
    item.appendChild(defaultBtn);

    listEl.appendChild(item);
  }

  // User-created prompts (excluding built-in), sorted alphabetically.
  const userPrompts = promptsState.prompts
    .filter((p) => p.id !== promptsState.builtinId)
    .sort((a, b) => a.name.localeCompare(b.name));

  for (const p of userPrompts) {
    const isDefault = p.id === promptsState.defaultId;
    const isSelected = promptsState.selectedId === p.id;

    const item = document.createElement("button");
    item.className = isSelected ? "prompt-item active" : "prompt-item";
    item.dataset.promptId = p.id;

    const nameSpan = document.createElement("span");
    nameSpan.className = "prompt-item-name truncate";
    nameSpan.textContent = p.name;
    item.appendChild(nameSpan);

    const defaultBtn = document.createElement("button");
    defaultBtn.className = isDefault ? "prompt-default-btn active" : "prompt-default-btn";
    defaultBtn.title = isDefault ? "Active default" : "Set as default";
    defaultBtn.innerHTML = isDefault ? ICON_CHECK_CIRCLE : ICON_CIRCLE;
    defaultBtn.dataset.action = "toggle-default";
    defaultBtn.dataset.promptId = p.id;
    item.appendChild(defaultBtn);

    listEl.appendChild(item);
  }

  if (userPrompts.length === 0 && !builtin) {
    const empty = document.createElement("div");
    empty.className = "prompts-empty-hint";
    empty.textContent = "No prompts yet";
    listEl.appendChild(empty);
  }
}
