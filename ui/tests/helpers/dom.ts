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
  "welcome-state", "welcome-input", "welcome-send", "welcome-model", "welcome-prompt",
  "input-bar", "input-text", "input-send",
  "right-panel", "close-right-panel", "panel-model",
  "panel-temperature", "panel-top-p", "panel-top-k", "panel-min-p",
  "panel-max-output-tokens", "panel-repetition-penalty", "panel-seed",
  "panel-temperature-default", "panel-top-p-default", "panel-top-k-default",
  "panel-chat-info", "panel-info-created", "panel-info-forked-row", "panel-info-forked",
];

/** Element IDs expected by getBrowserDom(). */
export const BROWSER_DOM_IDS = [
  "bp-cards", "bp-search", "bp-tab-all", "bp-tab-archived",
  "bp-tags", "bp-tags-section", "bp-select-all", "bp-delete",
  "bp-export", "bp-archive", "bp-restore", "bp-cancel",
  "bp-bulk-actions", "bp-toolbar",
];

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
  "sp-model", "sp-system-prompt", "sp-max-output-tokens", "sp-context-length",
  "sp-temperature", "sp-top-p", "sp-top-k",
  "sp-temperature-default", "sp-top-p-default", "sp-top-k-default",
  "sp-model-label", "sp-reset-model", "sp-status",
];

/** Tag overrides for settings DOM elements that need specific types. */
export const SETTINGS_DOM_TAGS: Record<string, string> = {
  "sp-model": "select",
  "sp-system-prompt": "textarea",
  "sp-max-output-tokens": "input",
  "sp-context-length": "input",
  "sp-temperature": "input",
  "sp-top-p": "input",
  "sp-top-k": "input",
  "sp-reset-model": "button",
};
