/**
 * Mode Manager — kernel-owned mode switching for the activity bar.
 *
 * Owns the active mode state, activity bar button handling, provenance
 * updates, and persistence via localStorage.
 * Emits "mode.changed" on the kernel EventBus for plugin coordination.
 */

import type { Disposable } from "../types.ts";
import type { EventBusImpl } from "../system/event-bus.ts";
import { updateProvenance } from "./provenance.ts";

const STORAGE_KEY = "talu-last-active-mode";
const CHAT_GROUP_MODES = new Set(["chat", "conversations", "prompts"]);

export class ModeManager {
  private active: string;
  private eventBus: EventBusImpl;
  private modes = new Map<string, { label: string; pluginId: string }>();

  constructor(eventBus: EventBusImpl) {
    this.eventBus = eventBus;
    // Derive the initial mode from whichever activity-bar button is already active in the DOM.
    const activeBtn = document.querySelector(".activity-btn.active");
    this.active = activeBtn?.getAttribute("data-mode") ?? "";
  }

  /** Register a mode contributed by a plugin manifest. */
  registerMode(key: string, label: string, pluginId: string): void {
    this.modes.set(key, { label, pluginId });
  }

  getActiveMode(): string {
    return this.active;
  }

  switchMode(mode: string): void {
    if (this.active === mode) return;
    const from = this.active;
    this.active = mode;

    const appLayout = document.querySelector(".app-layout");
    if (appLayout) {
      appLayout.setAttribute("data-mode", mode);
    }

    // Hide all mode views, then show the target.
    for (const el of document.querySelectorAll<HTMLElement>(".mode-view")) {
      el.classList.add("hidden");
    }
    const target = document.getElementById(`${mode}-mode`);
    if (target) {
      target.classList.remove("hidden");
    }

    // Update activity bar button states.
    // Keep the chat button active for all chat-group sub-modes.
    for (const btn of document.querySelectorAll<HTMLElement>(".activity-btn")) {
      const btnMode = btn.getAttribute("data-mode");
      if (btnMode === mode || (btnMode === "chat" && CHAT_GROUP_MODES.has(mode))) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    }

    // Update subnav button active states (only mode-switching buttons;
    // tab-switching buttons are managed by their own click handlers).
    for (const btn of document.querySelectorAll<HTMLElement>(".subnav-btn[data-nav-mode]")) {
      if (btn.getAttribute("data-nav-mode") === mode) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    }

    // Update provenance indicator.
    const info = this.modes.get(mode);
    if (info) {
      updateProvenance(info.label, info.pluginId, true);
    }

    // Persist to localStorage.
    try {
      localStorage.setItem(STORAGE_KEY, mode);
    } catch {
      // Storage full or disabled — degrade silently.
    }

    // Notify plugins.
    this.eventBus.emit("mode.changed", { from, to: mode });
  }

  /** Restore the last active mode from localStorage (synchronous). */
  restoreLastMode(): void {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved && this.modes.has(saved) && saved !== this.active) {
        this.switchMode(saved);
      }
    } catch {
      // Storage disabled — stay on default mode (chat).
    }
  }

  /** Install click handlers on activity bar buttons. Returns a Disposable for cleanup. */
  installActivityBarListeners(): Disposable {
    const handler = (e: Event) => {
      if (!(e.target instanceof Element)) return;
      const btn = e.target.closest<HTMLElement>(".activity-btn");
      if (!btn) return;
      const mode = btn.getAttribute("data-mode");
      if (!mode) return;

      this.switchMode(mode);
    };

    const bar = document.getElementById("activity-bar");
    if (bar) {
      bar.addEventListener("click", handler);

      // Logo click → new chat (delegates to the chat activity button).
      const logo = bar.querySelector<HTMLElement>(".topnav-logo");
      const logoHandler = () => {
        document.querySelector<HTMLElement>('.activity-btn[data-mode="chat"]')?.click();
      };
      if (logo) logo.addEventListener("click", logoHandler);

      return {
        dispose: () => {
          bar.removeEventListener("click", handler);
          if (logo) logo.removeEventListener("click", logoHandler);
        },
      };
    }
    return { dispose() {} };
  }

  /** Install click handlers on the subnav sidebar buttons. Returns a Disposable for cleanup. */
  installChatGroupNavListeners(): Disposable {
    const handler = (e: Event) => {
      if (!(e.target instanceof Element)) return;
      const btn = e.target.closest<HTMLElement>(".subnav-btn");
      if (!btn) return;

      // Mode-switching buttons (chat group).
      const mode = btn.getAttribute("data-nav-mode");
      if (mode) {
        // For chat mode: delegate to the activity-bar chat button so the
        // chat plugin's "new conversation" handler also fires.
        if (mode === "chat") {
          document.querySelector<HTMLElement>('.activity-btn[data-mode="chat"]')?.click();
          return;
        }
        this.switchMode(mode);
        return;
      }

      // Tab-switching buttons (files, models).
      const tab = btn.getAttribute("data-nav-tab");
      if (tab) {
        const group = btn.closest<HTMLElement>(".subnav-group");
        // Sync active state within this group.
        if (group) {
          for (const b of group.querySelectorAll<HTMLElement>(".subnav-btn")) {
            b.classList.toggle("active", b === btn);
          }
        }
        this.eventBus.emit("subnav.tab", { tab });
      }
    };

    const nav = document.getElementById("subnav");
    if (nav) {
      nav.addEventListener("click", handler);
      return {
        dispose: () => nav.removeEventListener("click", handler),
      };
    }
    return { dispose() {} };
  }
}
