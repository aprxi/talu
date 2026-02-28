/**
 * Shadow DOM slot creation for plugin view containers + app panel manager.
 *
 * CSS custom properties (--text, --bg, --accent, etc.) inherit through shadow
 * boundaries automatically. Class-based selectors are shared via a constructed
 * stylesheet applied through adoptedStyleSheets (synchronous, no FOUC).
 */

import type { Disposable } from "../types.ts";

let sharedSheet: CSSStyleSheet | null = null;

/** Access the pre-loaded shared stylesheet (for popover/renderer shadow roots). */
export function getSharedStylesheet(): CSSStyleSheet | null {
  return sharedSheet;
}

/**
 * Pre-load the shared stylesheet text and create a constructed CSSStyleSheet.
 * Must be called once during bootKernel() before any plugins register.
 */
export async function initSharedStylesheet(): Promise<void> {
  try {
    const resp = await fetch("/assets/style.css");
    if (!resp.ok) {
      console.warn("[kernel] Failed to load shared stylesheet — shadow roots will lack styles.");
      return;
    }
    const text = await resp.text();
    sharedSheet = new CSSStyleSheet();
    sharedSheet.replaceSync(text);
  } catch {
    console.warn("[kernel] Failed to load shared stylesheet — shadow roots will lack styles.");
  }
}

/**
 * Create a Shadow DOM container inside `hostElement` for a plugin to render into.
 * Clears existing content, attaches shadow root, injects stylesheet, returns container.
 */
export function createPluginSlot(pluginId: string, hostElement: HTMLElement): HTMLElement {
  // Clear static HTML (e.g., the settings form from index.html).
  hostElement.innerHTML = "";

  // Create a child div that owns the shadow root.
  const shadowHost = document.createElement("div");
  shadowHost.dataset["pluginId"] = pluginId;
  shadowHost.style.flex = "1";
  shadowHost.style.minWidth = "0";
  shadowHost.style.minHeight = "0";
  shadowHost.style.overflow = "hidden";
  hostElement.appendChild(shadowHost);

  const shadowRoot = shadowHost.attachShadow({ mode: "open" });

  // Apply the shared stylesheet via adoptedStyleSheets (synchronous, no network request).
  if (sharedSheet) {
    shadowRoot.adoptedStyleSheets = [sharedSheet];
  }

  // Inner container that the plugin renders into — fills the shadow host.
  const container = document.createElement("div");
  container.className = "plugin-container";
  container.style.width = "100%";
  container.style.height = "100%";
  shadowRoot.appendChild(container);

  return container;
}

// ── App Panel Manager ───────────────────────────────────────────────────────

export class AppPanelManager {
  private panelEl: HTMLElement;
  private titleEl: HTMLElement;
  private contentEl: HTMLElement;
  private closeBtn: HTMLElement;
  private escHandler: ((e: KeyboardEvent) => void) | null = null;
  private onHideFn: (() => void) | null = null;
  private currentOwner: string | null = null;

  constructor() {
    this.panelEl = document.getElementById("app-panel")!;
    this.titleEl = document.getElementById("app-panel-title")!;
    this.contentEl = document.getElementById("app-panel-content")!;
    this.closeBtn = document.getElementById("app-panel-close")!;

    this.closeBtn.addEventListener("click", () => this.hide());
  }

  show(options: { title: string; content: HTMLElement; owner?: string; onHide?: () => void }): Disposable {
    // Fire previous consumer's onHide (being displaced by new content).
    if (this.onHideFn) {
      this.onHideFn();
      this.onHideFn = null;
    }

    this.currentOwner = options.owner ?? null;
    this.onHideFn = options.onHide ?? null;
    this.titleEl.textContent = options.title;
    this.contentEl.innerHTML = "";
    this.contentEl.appendChild(options.content);
    this.panelEl.classList.remove("hidden");

    // Escape key dismisses the panel.
    if (!this.escHandler) {
      this.escHandler = (e: KeyboardEvent) => {
        if (e.key === "Escape") this.hide();
      };
      document.addEventListener("keydown", this.escHandler);
    }

    return { dispose: () => this.hide() };
  }

  /**
   * Hide the panel. If `owner` is specified, only hides when the current
   * panel belongs to that owner — otherwise it's a no-op (won't close
   * another consumer's content).
   */
  hide(owner?: string): void {
    if (owner && this.currentOwner !== owner) return;

    this.panelEl.classList.add("hidden");
    this.contentEl.innerHTML = "";
    this.titleEl.textContent = "";
    this.currentOwner = null;
    if (this.escHandler) {
      document.removeEventListener("keydown", this.escHandler);
      this.escHandler = null;
    }
    if (this.onHideFn) {
      const fn = this.onHideFn;
      this.onHideFn = null;
      fn();
    }
  }
}
