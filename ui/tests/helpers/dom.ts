/**
 * Shared DOM setup utilities for plugin tests.
 *
 * Each plugin's DOM cache (getChatDom, getBrowserDom, getPromptsDom) queries
 * elements by ID from a root container. These helpers create minimal root
 * elements with the expected IDs so tests can run without full HTML templates.
 */

/** Create a root element with child elements for each ID (default: div). */
export function createDomRoot(
  elementIds: string[],
  extras?: { tag?: string; className?: string }[],
  tagOverrides?: Record<string, string>,
): HTMLElement {
  const root = document.createElement("div");
  for (const id of elementIds) {
    const tag = tagOverrides?.[id] ?? "div";
    const el = document.createElement(tag);
    el.id = id;
    root.appendChild(el);
  }
  for (const { tag = "div", className = "" } of extras ?? []) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    root.appendChild(el);
  }
  return root;
}

/** Element IDs expected by getChatDom(). */
export const CHAT_DOM_IDS = [
  "sidebar-list", "loader-sentinel", "new-conversation", "transcript",
  "welcome-state", "welcome-input", "welcome-send", "welcome-attach", "welcome-library", "welcome-attachment-list", "welcome-model", "welcome-prompt",
  "input-bar", "input-text", "input-send", "input-attach", "input-library", "input-attachment-list", "chat-file-input",
  "right-panel", "close-right-panel", "panel-model",
  "panel-temperature", "panel-top-p", "panel-top-k", "panel-min-p",
  "panel-max-output-tokens", "panel-repetition-penalty", "panel-seed",
  "panel-temperature-default", "panel-top-p-default", "panel-top-k-default",
  "panel-chat-info", "panel-info-created", "panel-info-forked-row", "panel-info-forked",
];

/** Tag overrides for chat DOM elements that need specific types. */
export const CHAT_DOM_TAGS: Record<string, string> = {
  "welcome-input": "textarea",
  "welcome-send": "button",
  "welcome-attach": "button",
  "welcome-library": "button",
  "welcome-model": "select",
  "welcome-prompt": "select",
  "new-conversation": "button",
  "input-text": "textarea",
  "input-send": "button",
  "input-attach": "button",
  "input-library": "button",
  "chat-file-input": "input",
  "close-right-panel": "button",
  "panel-model": "select",
  "panel-temperature": "input",
  "panel-top-p": "input",
  "panel-top-k": "input",
  "panel-min-p": "input",
  "panel-max-output-tokens": "input",
  "panel-repetition-penalty": "input",
  "panel-seed": "input",
};

/** Element IDs expected by getBrowserDom(). */
export const BROWSER_DOM_IDS = [
  "bp-cards", "bp-search", "bp-tab-all", "bp-tab-archived",
  "bp-tags", "bp-tags-section", "bp-select-all", "bp-delete",
  "bp-export", "bp-archive", "bp-restore", "bp-cancel",
  "bp-bulk-actions", "bp-toolbar", "bp-pagination",
];

/** Tag overrides for browser DOM elements that need specific types. */
export const BROWSER_DOM_TAGS: Record<string, string> = {
  "bp-search": "input",
  "bp-tab-all": "button",
  "bp-tab-archived": "button",
  "bp-select-all": "button",
  "bp-delete": "button",
  "bp-export": "button",
  "bp-archive": "button",
  "bp-restore": "button",
  "bp-cancel": "button",
};

/** Extra class-based elements for getBrowserDom(). */
export const BROWSER_DOM_EXTRAS: { tag: string; className: string }[] = [
  { tag: "button", className: "browser-clear-btn" },
];

/** Element IDs expected by getPromptsDom(). */
export const PROMPTS_DOM_IDS = [
  "pp-list", "pp-editor", "pp-empty", "pp-name",
  "pp-content", "pp-save-btn", "pp-delete-btn", "pp-copy-btn", "pp-new-btn",
];

/** Element IDs expected by getSettingsDom(). */
export const SETTINGS_DOM_IDS = [
  "sp-model", "sp-system-prompt-enabled", "sp-system-prompt-name", "sp-open-prompts",
  "sp-max-output-tokens", "sp-context-length",
  "sp-auto-title",
  "sp-temperature", "sp-top-p", "sp-top-k",
  "sp-temperature-default", "sp-top-p-default", "sp-top-k-default",
  "sp-model-label", "sp-reset-model", "sp-status",
];

/** Element IDs expected by getFilesDom(). */
export const FILES_DOM_IDS = [
  "fp-upload", "fp-file-input", "fp-search", "fp-search-clear",
  "fp-stats", "fp-count", "fp-tbody", "fp-table-container",
  "fp-drop-overlay", "fp-preview", "fp-preview-content",
  "fp-tab-all", "fp-tab-archived", "fp-select-all",
  "fp-archive", "fp-restore", "fp-delete", "fp-cancel",
  "fp-bulk-actions", "fp-toolbar",
];

/** Extra class-based elements for getFilesDom(). */
export const FILES_DOM_EXTRAS: { tag: string; className: string }[] = [
  { tag: "div", className: "files-main-drop" },
];

/** Tag overrides for files DOM elements that need specific types. */
export const FILES_DOM_TAGS: Record<string, string> = {
  "fp-file-input": "input",
  "fp-search": "input",
  "fp-upload": "button",
  "fp-tab-all": "button",
  "fp-tab-archived": "button",
  "fp-select-all": "button",
  "fp-archive": "button",
  "fp-restore": "button",
  "fp-delete": "button",
  "fp-cancel": "button",
  "fp-search-clear": "button",
  "fp-tbody": "tbody",
};

/** Tag overrides for settings DOM elements that need specific types. */
export const SETTINGS_DOM_TAGS: Record<string, string> = {
  "sp-model": "select",
  "sp-system-prompt-enabled": "input",
  "sp-open-prompts": "button",
  "sp-max-output-tokens": "input",
  "sp-context-length": "input",
  "sp-auto-title": "input",
  "sp-temperature": "input",
  "sp-top-p": "input",
  "sp-top-k": "input",
  "sp-reset-model": "button",
};

/** Element IDs expected by getRepoDom(). */
export const REPO_DOM_IDS = [
  "rp-search", "rp-search-clear",
  "rp-tab-local", "rp-tab-pinned", "rp-tab-discover",
  "rp-stats", "rp-thead", "rp-tbody",
  "rp-table-container", "rp-discover-container", "rp-discover-results", "rp-downloads",
  "rp-count", "rp-select-all", "rp-pin-all", "rp-delete", "rp-cancel",
  "rp-bulk-actions",
  "rp-sort", "rp-size-filter", "rp-task-filter", "rp-library-filter",
];

/** Tag overrides for repo DOM elements that need specific types. */
export const REPO_DOM_TAGS: Record<string, string> = {
  "rp-search": "input",
  "rp-search-clear": "button",
  "rp-tab-local": "button",
  "rp-tab-pinned": "button",
  "rp-tab-discover": "button",
  "rp-thead": "thead",
  "rp-tbody": "tbody",
  "rp-select-all": "button",
  "rp-pin-all": "button",
  "rp-delete": "button",
  "rp-cancel": "button",
  "rp-sort": "select",
  "rp-size-filter": "select",
  "rp-task-filter": "select",
  "rp-library-filter": "select",
};
