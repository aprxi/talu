/**
 * Prompts CRUD operations — save, delete, toggle default, copy.
 */

import { DELETE_ICON as ICON_DELETE } from "../../icons.ts";
import { api, clipboard, timers, notifications, log } from "./deps.ts";
import { promptsState } from "./state.ts";
import { getPromptsDom } from "./dom.ts";
import { renderList } from "./list.ts";
import { showEmpty, updateSaveButton, emitPromptsChanged } from "./editor.ts";

export async function saveCurrentPrompt(): Promise<void> {
  const dom = getPromptsDom();
  const name = dom.nameInput.value.trim();
  const content = dom.contentInput.value.trim();

  if (!name) {
    notifications.error("Please enter a prompt name");
    return;
  }

  // Prevent saving over the built-in prompt.
  if (promptsState.builtinId && promptsState.selectedId === promptsState.builtinId) return;

  if (promptsState.selectedId) {
    const result = await api.updateDocument(promptsState.selectedId, {
      title: name,
      content: { system: content },
    });
    if (result.ok) {
      const idx = promptsState.prompts.findIndex((p) => p.id === promptsState.selectedId);
      if (idx >= 0 && promptsState.prompts[idx]) {
        promptsState.prompts[idx].name = name;
        promptsState.prompts[idx].content = content;
        promptsState.prompts[idx].updatedAt = Date.now();
      }
      notifications.success("Prompt saved");
    } else {
      notifications.error(result.error ?? "Failed to save prompt");
      return;
    }
  } else {
    const result = await api.createDocument({
      type: "prompt",
      title: name,
      content: { system: content },
    });
    if (result.ok && result.data) {
      const newPrompt = {
        id: result.data.id,
        name,
        content,
        createdAt: result.data.created_at,
        updatedAt: result.data.updated_at,
      };
      promptsState.prompts.push(newPrompt);
      promptsState.selectedId = newPrompt.id;
      dom.deleteBtn.classList.remove("hidden");
      notifications.success("Prompt created");
    } else {
      notifications.error(result.error ?? "Failed to create prompt");
      return;
    }
  }

  promptsState.originalName = name;
  promptsState.originalContent = content;
  updateSaveButton();
  renderList();
  emitPromptsChanged();
}

export function handleDelete(): void {
  if (!promptsState.selectedId) return;
  if (promptsState.selectedId === promptsState.builtinId) return;
  const dom = getPromptsDom();

  if (!dom.deleteBtn.classList.contains("confirming")) {
    dom.deleteBtn.classList.add("confirming", "text-danger", "bg-danger/10");
    dom.deleteBtn.classList.remove("btn-icon");
    dom.deleteBtn.textContent = "Delete?";
    dom.deleteBtn.title = "Click again to confirm delete";
    promptsState.deleteConfirmHandle = timers.setTimeout(() => resetDeleteBtn(), 3000);
    return;
  }

  promptsState.deleteConfirmHandle?.dispose();
  promptsState.deleteConfirmHandle = null;
  doDelete();
}

export function resetDeleteBtn(): void {
  const dom = getPromptsDom();
  dom.deleteBtn.classList.remove("confirming", "text-danger", "bg-danger/10");
  dom.deleteBtn.classList.add("btn-icon");
  dom.deleteBtn.innerHTML = ICON_DELETE;
  dom.deleteBtn.title = "Delete prompt";
  promptsState.deleteConfirmHandle = null;
}

async function doDelete(): Promise<void> {
  if (!promptsState.selectedId) return;
  if (promptsState.selectedId === promptsState.builtinId) return;
  resetDeleteBtn();

  const result = await api.deleteDocument(promptsState.selectedId);
  if (!result.ok) {
    notifications.error(result.error ?? "Failed to delete prompt");
    return;
  }

  const idx = promptsState.prompts.findIndex((p) => p.id === promptsState.selectedId);
  if (idx >= 0) promptsState.prompts.splice(idx, 1);

  // If deleting the custom default, revert to built-in.
  if (promptsState.defaultId === promptsState.selectedId) {
    promptsState.defaultId = null;
    api.patchSettings({ default_prompt_id: null }).catch(() =>
      log.warn("Failed to clear default prompt"),
    );
  }

  promptsState.selectedId = null;
  renderList();
  showEmpty();
  emitPromptsChanged();
  notifications.success("Prompt deleted");
}

export function toggleDefault(promptId: string | null): void {
  if (promptId === null) {
    // Unsetting custom default → revert to built-in.
    if (promptsState.defaultId === null) return; // Already on built-in
    promptsState.defaultId = null;
  } else {
    // Setting a custom prompt as default.
    if (promptsState.defaultId === promptId) return; // Already set
    promptsState.defaultId = promptId;
  }

  api.patchSettings({ default_prompt_id: promptsState.defaultId }).catch(() =>
    log.warn("Failed to save default prompt"),
  );
  renderList();
  emitPromptsChanged();
}

export function copyPrompt(): void {
  const dom = getPromptsDom();
  const content = dom.contentInput.value;
  if (!content) return;
  clipboard
    .writeText(content)
    .then(() => notifications.success("Copied to clipboard"))
    .catch(() => notifications.error("Failed to copy"));
}
