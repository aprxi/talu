/**
 * Prompts plugin DOM cache — queries elements within the plugin's
 * container (set via initPromptsDom).
 */

export interface PromptsDom {
  listEl: HTMLElement;
  editorEl: HTMLElement;
  emptyEl: HTMLElement;
  nameInput: HTMLInputElement;
  contentInput: HTMLTextAreaElement;
  saveBtn: HTMLButtonElement;
  deleteBtn: HTMLButtonElement;
  copyBtn: HTMLButtonElement;
  newBtn: HTMLButtonElement;
}

let root: HTMLElement;
let cached: PromptsDom | null = null;

/** Set the root container for DOM queries. Must be called after buildPromptsDOM(). */
export function initPromptsDom(container: HTMLElement): void {
  root = container;
  cached = null;
}

export function getPromptsDom(): PromptsDom {
  if (cached) return cached;

  const q = <T extends HTMLElement>(id: string) => root.querySelector<T>(`#${id}`)!;

  cached = {
    listEl: q("pp-list"),
    editorEl: q("pp-editor"),
    emptyEl: q("pp-empty"),
    nameInput: q<HTMLInputElement>("pp-name"),
    contentInput: q<HTMLTextAreaElement>("pp-content"),
    saveBtn: q<HTMLButtonElement>("pp-save-btn"),
    deleteBtn: q<HTMLButtonElement>("pp-delete-btn"),
    copyBtn: q<HTMLButtonElement>("pp-copy-btn"),
    newBtn: q<HTMLButtonElement>("pp-new-btn"),
  };

  return cached;
}
