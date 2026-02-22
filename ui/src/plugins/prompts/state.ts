/**
 * Prompts plugin local state — centralized state object
 * shared across all prompts submodules.
 */

import type { Disposable } from "../../kernel/types.ts";

export interface SavedPrompt {
  id: string;
  name: string;
  content: string;
  createdAt: number;
  updatedAt: number;
}

export const DEFAULT_PROMPT_KEY = "defaultPrompt";

/** Built-in default prompt name and content (hardcoded fallback). */
export const BUILTIN_PROMPT_NAME = "Default";
export const BUILTIN_PROMPT_CONTENT = "You are a helpful assistant.";

export interface PromptsState {
  prompts: SavedPrompt[];
  /** Document ID of the built-in default prompt (always present after init). */
  builtinId: string | null;
  selectedId: string | null;
  /** Custom default override. null = built-in default is active. */
  defaultId: string | null;
  originalName: string;
  originalContent: string;
  deleteConfirmHandle: Disposable | null;
}

export const promptsState: PromptsState = {
  prompts: [],
  builtinId: null,
  selectedId: null,
  defaultId: null,
  originalName: "",
  originalContent: "",
  deleteConfirmHandle: null,
};
