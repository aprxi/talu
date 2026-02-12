/**
 * Command Registry — register commands with keybindings and when-clause matching.
 *
 * Commands are dispatched based on focus context. The `when` clause evaluates
 * simple equality tests (key == 'value' | key != 'value') against the current
 * focus state (focusedView).
 */

import type { Disposable, CommandRegistry } from "../types.ts";
import { getFocusedViewId } from "../ui/focus.ts";
import { resolveAlias } from "../core/alias.ts";

interface CommandEntry {
  id: string;
  pluginId: string;
  handler: () => void;
  keybinding?: string;
  when?: string;
  label?: string;
}

export class CommandRegistryImpl implements CommandRegistry {
  private commands = new Map<string, CommandEntry>();
  private keyHandler: ((e: KeyboardEvent) => void) | null = null;

  constructor() {
    this.installKeyHandler();
  }

  /** Register with full metadata (used by kernel internals). */
  registerScoped(
    pluginId: string,
    fqId: string,
    handler: () => void,
    options?: { keybinding?: string; when?: string; label?: string },
  ): Disposable {
    this.commands.set(fqId, {
      id: fqId,
      pluginId,
      handler,
      keybinding: options?.keybinding,
      when: options?.when,
      label: options?.label,
    });

    return {
      dispose: () => {
        this.commands.delete(fqId);
      },
    };
  }

  register(id: string, handler: () => void, options?: { keybinding?: string; when?: string }): Disposable {
    return this.registerScoped("unknown", id, handler, options);
  }

  /** Get all registered commands (for command palette). */
  getAll(): CommandEntry[] {
    return [...this.commands.values()];
  }

  /** Execute a command by ID. */
  execute(commandId: string): boolean {
    const cmd = this.commands.get(resolveAlias(commandId));
    if (!cmd) return false;

    if (!this.matchesWhen(cmd.when)) return false;

    try {
      cmd.handler();
      return true;
    } catch (err) {
      console.error(`[kernel] Command "${commandId}" threw:`, err);
      return false;
    }
  }

  private installKeyHandler(): void {
    this.keyHandler = (e: KeyboardEvent) => {
      // Suppress keybindings when focused on editable elements.
      const active = document.activeElement;
      if (active instanceof HTMLInputElement ||
          active instanceof HTMLTextAreaElement ||
          (active instanceof HTMLElement && active.isContentEditable)) {
        return;
      }

      const pressed = this.normalizeKeybinding(e);
      if (!pressed) return;

      for (const cmd of this.commands.values()) {
        if (!cmd.keybinding) continue;
        if (this.normalizeKeybindingString(cmd.keybinding) !== pressed) continue;
        if (!this.matchesWhen(cmd.when)) continue;

        e.preventDefault();
        e.stopPropagation();
        try {
          cmd.handler();
        } catch (err) {
          console.error(`[kernel] Command "${cmd.id}" threw:`, err);
        }
        return;
      }
    };

    document.addEventListener("keydown", this.keyHandler);
  }

  private matchesWhen(when?: string): boolean {
    if (!when) return true;

    const focusedView = getFocusedViewId() ?? "";

    // Parse simple expressions: key == 'value' or key != 'value'
    const eqMatch = when.match(/^(\w+)\s*==\s*'([^']*)'$/);
    if (eqMatch) {
      const [, key, value] = eqMatch;
      if (key === "focusedView") return focusedView === value;
      return false;
    }

    const neqMatch = when.match(/^(\w+)\s*!=\s*'([^']*)'$/);
    if (neqMatch) {
      const [, key, value] = neqMatch;
      if (key === "focusedView") return focusedView !== value;
      return false;
    }

    return true;
  }

  private normalizeKeybinding(e: KeyboardEvent): string | null {
    const parts: string[] = [];
    if (e.ctrlKey || e.metaKey) parts.push("ctrl");
    if (e.altKey) parts.push("alt");
    if (e.shiftKey) parts.push("shift");

    const key = e.key.toLowerCase();
    if (key === "control" || key === "alt" || key === "shift" || key === "meta") return null;
    parts.push(key);

    return parts.join("+");
  }

  private normalizeKeybindingString(keybinding: string): string {
    return keybinding
      .toLowerCase()
      .replace(/cmd/g, "ctrl")
      .replace(/\s/g, "")
      .split("+")
      .sort((a, b) => {
        const order = ["ctrl", "alt", "shift"];
        const ai = order.indexOf(a);
        const bi = order.indexOf(b);
        if (ai >= 0 && bi >= 0) return ai - bi;
        if (ai >= 0) return -1;
        if (bi >= 0) return 1;
        return a.localeCompare(b);
      })
      .join("+");
  }

  dispose(): void {
    if (this.keyHandler) {
      document.removeEventListener("keydown", this.keyHandler);
      this.keyHandler = null;
    }
    this.commands.clear();
  }
}
