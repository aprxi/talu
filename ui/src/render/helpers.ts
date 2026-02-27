import type { Conversation, ModelEntry } from "../types.ts";
import { FormatAccessImpl } from "../kernel/system/format.ts";

const fmt = new FormatAccessImpl();

// -- Thinking preference ------------------------------------------------------
// Sync in-memory cache, initialized by the chat plugin via initThinkingState().
// Write-through function persists changes to ctx.storage (fire-and-forget).

let thinkingExpanded = false;
let thinkingWriteFn: ((expanded: boolean) => void) | null = null;

/**
 * Initialize the thinking state. Called once by the chat plugin during run().
 * @param initial - The initial expanded state (from ctx.storage).
 * @param writeFn - Callback to persist changes (writes to ctx.storage).
 */
export function initThinkingState(initial: boolean, writeFn: (expanded: boolean) => void): void {
  thinkingExpanded = initial;
  thinkingWriteFn = writeFn;
}

/** Whether reasoning blocks should be expanded. Defaults to false (collapsed). */
export function isThinkingExpanded(): boolean {
  return thinkingExpanded;
}

export function setThinkingExpanded(expanded: boolean): void {
  thinkingExpanded = expanded;
  thinkingWriteFn?.(expanded);
}

// -- DOM helpers --------------------------------------------------------------

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

// -- Conversation helpers -----------------------------------------------------

export function isPinned(chat: Conversation): boolean {
  return chat.marker === "pinned";
}

export function isArchived(chat: Conversation): boolean {
  return chat.marker === "archived";
}

/** Extract tag names from conversation. Reads from the relational `tags` field
 *  (source of truth, synced with the search index). Returns empty array if none. */
export function getTags(chat: Conversation): string[] {
  const tags = chat.tags;
  if (!Array.isArray(tags)) return [];
  return tags.map((t) => (typeof t === "string" ? t : t?.name)).filter((n): n is string => typeof n === "string");
}

// Re-export icons used by sidebar/browser render modules.
export { PIN_ICON as PIN_SVG, FORK_SMALL_ICON as FORK_SVG } from "../icons.ts";

// -- Time formatting ----------------------------------------------------------

export function relativeTime(epoch: number): string {
  const ms = epoch > 1e12 ? epoch : epoch * 1000;
  const delta = Math.max(0, (Date.now() - ms) / 1000);
  if (delta < 60) return "just now";
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  if (delta < 604800) return `${Math.floor(delta / 86400)}d ago`;
  return fmt.date(ms);
}

/** Format epoch as a localized date-time string (e.g. "Jan 31, 2026, 03:45 PM"). */
export function formatDate(epoch: number): string {
  const ms = epoch > 1e12 ? epoch : epoch * 1000;
  return fmt.dateTime(ms);
}

// -- Model select helpers -----------------------------------------------------

/** Populate a <select> element with model entries, grouped by provider.
 *  Remote models use `provider::model_id` syntax — these are grouped under
 *  an `<optgroup>` labeled with the provider name. Local models (no `::`)
 *  appear under a "Local" group. */
export function populateModelSelect(sel: HTMLSelectElement, models: ModelEntry[], selected: string): void {
  sel.innerHTML = "";
  if (models.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No models available";
    sel.appendChild(opt);
    return;
  }

  // Group models by provider prefix.
  const groups = new Map<string, { value: string; label: string }[]>();
  for (const m of models) {
    const sep = m.id.indexOf("::");
    const group = sep >= 0 ? m.id.substring(0, sep) : "Local";
    const label = sep >= 0 ? m.id.substring(sep + 2) : m.id;
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group)!.push({ value: m.id, label });
  }

  // If only one group, skip the optgroup wrapper for cleaner UI.
  if (groups.size === 1) {
    for (const entries of groups.values()) {
      for (const entry of entries) {
        const opt = document.createElement("option");
        opt.value = entry.value;
        opt.textContent = entry.label;
        sel.appendChild(opt);
      }
    }
  } else {
    for (const [groupName, entries] of groups) {
      const optgroup = document.createElement("optgroup");
      optgroup.label = groupName.charAt(0).toUpperCase() + groupName.slice(1);
      for (const entry of entries) {
        const opt = document.createElement("option");
        opt.value = entry.value;
        opt.textContent = entry.label;
        optgroup.appendChild(opt);
      }
      sel.appendChild(optgroup);
    }
  }

  if (selected && models.some((m) => m.id === selected)) {
    sel.value = selected;
  } else if (models.length > 0 && models[0]) {
    sel.value = models[0].id;
  }
}
