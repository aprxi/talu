/**
 * View Manager — dynamic view registration for the activity bar + content area.
 *
 * Plugins call ctx.layout.registerView(slot, factory) to add a view.
 * Built-in views are declared via manifest contributes.mode and rendered
 * in static HTML slots. Third-party views get dynamic activity bar
 * buttons with "ext" badges.
 */

import type { Disposable } from "../types.ts";
import { createPluginSlot } from "./layout.ts";

interface RegisteredView {
  slot: string;
  pluginId: string;
  label: string;
  priority: number;
  element: HTMLElement;
  activityBtn: HTMLElement;
  shadowRoot: ShadowRoot;
  factory: (shadowRoot: ShadowRoot) => void;
  mounted: boolean;
}

export class ViewManager {
  private views = new Map<string, RegisteredView>();
  private contentArea: HTMLElement | null = null;
  private activityBar: HTMLElement | null = null;

  constructor() {
    this.contentArea = document.querySelector(".app-content");
    this.activityBar = document.querySelector(".activity-bar");
  }

  /**
   * Register a view into a named slot.
   * Creates a new mode-view, activity bar button, and calls the factory with a shadow root.
   */
  registerView(pluginId: string, slot: string, label: string, builtin: boolean, factory: (shadowRoot: ShadowRoot) => void, priority = 0): Disposable {
    const viewId = `${pluginId}.${slot}`;
    if (this.views.has(viewId)) {
      console.warn(`[kernel] View "${viewId}" already registered.`);
      return { dispose() {} };
    }
    if (!this.contentArea || !this.activityBar) {
      return { dispose() {} };
    }

    // Create the mode-view container.
    const modeEl = document.createElement("div");
    modeEl.id = `view-${viewId}`;
    modeEl.className = "mode-view hidden";
    this.contentArea.appendChild(modeEl);

    // Create a shadow DOM container inside the mode-view.
    const pluginContainer = createPluginSlot(pluginId, modeEl);
    const shadowRoot = pluginContainer.getRootNode() as ShadowRoot;

    // Create activity bar button.
    const btn = document.createElement("button");
    btn.className = "activity-btn";
    btn.dataset["mode"] = viewId;
    btn.title = label;
    // Default icon for third-party views.
    btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg>`;

    // "ext" badge for third-party plugins.
    if (!builtin) {
      const badge = document.createElement("span");
      badge.className = "activity-btn-badge";
      badge.textContent = "ext";
      btn.appendChild(badge);
      btn.style.position = "relative";
    }

    // Insert into activity bar ordered by priority among dynamic buttons.
    // Built-in buttons are static HTML; dynamic buttons go before the last built-in button.
    this.insertButtonByPriority(btn, priority);

    const view: RegisteredView = {
      slot,
      pluginId,
      label,
      priority,
      element: modeEl,
      activityBtn: btn,
      shadowRoot,
      factory,
      mounted: false,
    };
    this.views.set(viewId, view);

    // Mount the view immediately.
    try {
      factory(shadowRoot);
      view.mounted = true;
    } catch (err) {
      console.error(`[kernel] View "${viewId}" factory threw:`, err);
    }

    return {
      dispose: () => {
        modeEl.remove();
        btn.remove();
        this.views.delete(viewId);
      },
    };
  }

  /**
   * Insert a dynamic activity bar button ordered by priority (higher first).
   * Dynamic buttons are placed before the last built-in button.
   */
  private insertButtonByPriority(btn: HTMLElement, priority: number): void {
    if (!this.activityBar) return;

    // Find insertion point: among existing dynamic buttons, insert before the first
    // with lower priority. Dynamic buttons have a viewId-style data-mode (contains '.').
    const dynamicBtns = this.activityBar.querySelectorAll<HTMLElement>(".activity-btn[data-mode]");
    let insertBefore: HTMLElement | null = null;

    for (const existing of dynamicBtns) {
      const mode = existing.dataset["mode"] ?? "";
      // Skip static built-in buttons (no '.' in data-mode).
      if (!mode.includes(".")) continue;
      const existingView = this.views.get(mode);
      if (existingView && existingView.priority < priority) {
        insertBefore = existing;
        break;
      }
    }

    if (insertBefore) {
      this.activityBar.insertBefore(btn, insertBefore);
    } else {
      // Append after all dynamic buttons but before the last built-in button.
      const builtinBtns = [...this.activityBar.querySelectorAll<HTMLElement>(".activity-btn[data-mode]")]
        .filter(b => !(b.dataset["mode"] ?? "").includes("."));
      const lastBuiltin = builtinBtns.at(-1) ?? null;
      if (lastBuiltin) {
        this.activityBar.insertBefore(btn, lastBuiltin);
      } else {
        this.activityBar.appendChild(btn);
      }
    }
  }

  /** Get all registered view IDs (for mode switching). */
  getRegisteredViewIds(): string[] {
    return [...this.views.keys()];
  }

  /** Check if a view ID is registered. */
  has(viewId: string): boolean {
    return this.views.has(viewId);
  }
}
