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

export interface PromptsState {
  prompts: SavedPrompt[];
  selectedId: string | null;
  defaultId: string | null;
  originalName: string;
  originalContent: string;
  deleteConfirmHandle: Disposable | null;
}

export const promptsState: PromptsState = {
  prompts: [],
  selectedId: null,
  defaultId: null,
  originalName: "",
  originalContent: "",
  deleteConfirmHandle: null,
};
