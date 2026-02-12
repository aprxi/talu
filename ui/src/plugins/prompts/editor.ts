/**
 * Prompts editor — selection, visibility, input state, event wiring.
 */

import { events } from "./deps.ts";
import { promptsState } from "./state.ts";
import { getPromptsDom } from "./dom.ts";
import { renderList } from "./list.ts";
import { saveCurrentPrompt, handleDelete, copyPrompt, toggleDefault } from "./crud.ts";

/** Broadcast current prompts state to other plugins (e.g. chat prompt selector). */
export function emitPromptsChanged(): void {
  events.emit("prompts.changed", {
    prompts: promptsState.prompts.map((p) => ({ id: p.id, name: p.name })),
    defaultId: promptsState.defaultId,
  });
}

export function selectPrompt(id: string): void {
  const p = promptsState.prompts.find((pr) => pr.id === id);
  if (!p) return;

  const dom = getPromptsDom();
  promptsState.selectedId = id;
  dom.nameInput.value = p.name;
  dom.contentInput.value = p.content;
  promptsState.originalName = p.name;
  promptsState.originalContent = p.content;
  dom.deleteBtn.classList.remove("hidden");
  updateSaveButton();
  renderList();
  showEditor();
  emitPromptsChanged();
}

export function createNew(): void {
  const dom = getPromptsDom();
  promptsState.selectedId = null;
  dom.nameInput.value = "";
  dom.contentInput.value = "";
  promptsState.originalName = "";
  promptsState.originalContent = "";
  dom.deleteBtn.classList.add("hidden");
  updateSaveButton();
  renderList();
  showEditor();
  dom.nameInput.focus();
  emitPromptsChanged();
}

export function showEditor(): void {
  const dom = getPromptsDom();
  dom.emptyEl.classList.add("hidden");
  dom.editorEl.classList.remove("hidden");
}

export function showEmpty(): void {
  const dom = getPromptsDom();
  dom.editorEl.classList.add("hidden");
  dom.emptyEl.classList.remove("hidden");
}

export function updateSaveButton(): void {
  const dom = getPromptsDom();
  const name = dom.nameInput.value;
  const content = dom.contentInput.value;
  const hasChanges = name !== promptsState.originalName || content !== promptsState.originalContent;
  const hasContent = name.trim() !== "" || content.trim() !== "";
  dom.saveBtn.disabled = !(promptsState.selectedId ? hasChanges : hasContent);
}

/** Wire all event handlers — buttons + delegated list clicks. */
export function wireEvents(): void {
  const dom = getPromptsDom();

  dom.newBtn.addEventListener("click", createNew);
  dom.saveBtn.addEventListener("click", saveCurrentPrompt);
  dom.deleteBtn.addEventListener("click", handleDelete);
  dom.copyBtn.addEventListener("click", copyPrompt);
  dom.nameInput.addEventListener("input", updateSaveButton);
  dom.contentInput.addEventListener("input", updateSaveButton);

  // Delegated list click handler.
  dom.listEl.addEventListener("click", (e) => {
    const target = e.target as HTMLElement;

    // Toggle-default button click.
    const defaultBtn = target.closest<HTMLElement>("[data-action='toggle-default']");
    if (defaultBtn) {
      e.stopPropagation();
      const id = defaultBtn.dataset.promptId!;
      const isDefault = id === promptsState.defaultId;
      toggleDefault(isDefault ? null : id);
      if (!isDefault) selectPrompt(id);
      return;
    }

    // Prompt item click.
    const item = target.closest<HTMLElement>("[data-prompt-id]");
    if (item && item.dataset.promptId) {
      selectPrompt(item.dataset.promptId);
    }
  });
}
