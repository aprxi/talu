/**
 * Resource Caps — per-plugin limits on timers, observers, storage, etc.
 *
 * Enforced at the individual module level (ManagedTimers, ManagedObservers,
 * StorageFacade). This module provides shared constants and a centralized
 * renderer DOM node cap check.
 */

/** Maximum active timers per plugin (enforced in timers.ts). */
export const MAX_TIMERS = 100;

/** Maximum active observers per plugin (enforced in observers.ts). */
export const MAX_OBSERVERS = 50;

/** Maximum storage keys per plugin. */
export const MAX_STORAGE_KEYS = 1000;

/** Maximum total storage bytes per plugin (50 MB). */
export const MAX_STORAGE_BYTES = 50 * 1024 * 1024;

/** Maximum bytes per storage document (5 MB). */
export const MAX_DOCUMENT_BYTES = 5 * 1024 * 1024;

/** Maximum tool result size for renderer display (1 MB). */
export const MAX_TOOL_RESULT_BYTES = 1024 * 1024;

/** Maximum DOM nodes per renderer instance before collapsed summary. */
export const MAX_RENDERER_DOM_NODES = 10_000;

/**
 * Check if a shadow root exceeds the DOM node cap.
 * Returns the node count if exceeded, or 0 if within limit.
 */
export function checkRendererNodeCap(shadowRoot: ShadowRoot): number {
  const count = shadowRoot.querySelectorAll("*").length;
  return count > MAX_RENDERER_DOM_NODES ? count : 0;
}

/**
 * Truncate a tool result string to the maximum display size.
 * Returns the original string if within limit.
 */
export function truncateToolResult(text: string): string {
  if (text.length <= MAX_TOOL_RESULT_BYTES) return text;
  return text.slice(0, MAX_TOOL_RESULT_BYTES) + `\n(truncated: ${text.length} bytes total)`;
}
