/**
 * Command Palette — Kernel-owned Ctrl+P fuzzy-search overlay.
 *
 * Lists all registered commands with keybindings and plugin attribution.
 * when-clause filtering based on current focus context.
 * Keyboard-only: arrow keys navigate, Enter executes, Escape dismisses.
 */

import type { Disposable, ContextValue } from "../types.ts";
import type { CommandRegistryImpl } from "../registries/commands.ts";
import type { ContextKeyService } from "../registries/context-keys.ts";

let paletteOverlay: HTMLElement | null = null;
let isOpen = false;

export interface CommandPaletteHandle extends Disposable {
  open(): void;
}

/**
 * Install the command palette. Registers the Ctrl+P global keybinding.
 * Returns a handle with dispose() and open() for programmatic triggering.
 */
export function installCommandPalette(
  commandRegistry: CommandRegistryImpl,
  contextKeys: ContextKeyService,
): CommandPaletteHandle {
  const onKey = (e: KeyboardEvent) => {
    // Ctrl+P or Cmd+P — don't trigger in editable elements.
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "p") {
      const active = document.activeElement;
      if (active instanceof HTMLInputElement ||
          active instanceof HTMLTextAreaElement ||
          (active instanceof HTMLElement && active.isContentEditable)) {
        return;
      }
      e.preventDefault();
      if (isOpen) {
        closePalette();
      } else {
        openPalette(commandRegistry, contextKeys);
      }
    }
  };

  document.addEventListener("keydown", onKey);

  return {
    open() {
      if (!isOpen) openPalette(commandRegistry, contextKeys);
    },
    dispose() {
      document.removeEventListener("keydown", onKey);
      closePalette();
    },
  };
}

function openPalette(commandRegistry: CommandRegistryImpl, contextKeys: ContextKeyService): void {
  if (isOpen) return;
  isOpen = true;

  paletteOverlay = document.createElement("div");
  paletteOverlay.id = "command-palette-overlay";
  paletteOverlay.style.cssText =
    "position:fixed;inset:0;z-index:2500;display:flex;align-items:flex-start;" +
    "justify-content:center;padding-top:20vh;background:rgba(0,0,0,0.3);";

  const container = document.createElement("div");
  container.style.cssText =
    "background:var(--bg-secondary,#1e1e2e);border:1px solid var(--border,#45475a);" +
    "border-radius:8px;width:480px;max-height:400px;display:flex;flex-direction:column;" +
    "font-family:var(--font-family,system-ui);color:var(--text,#cdd6f4);" +
    "box-shadow:0 8px 32px rgba(0,0,0,0.4);";

  // Search input.
  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = "Type a command...";
  input.style.cssText =
    "width:100%;padding:0.75rem 1rem;border:none;border-bottom:1px solid var(--border,#45475a);" +
    "background:transparent;color:var(--text,#cdd6f4);font-size:0.875rem;outline:none;";
  container.appendChild(input);

  // Command list.
  const list = document.createElement("div");
  list.style.cssText = "overflow-y:auto;flex:1;";
  container.appendChild(list);

  paletteOverlay.appendChild(container);
  document.body.appendChild(paletteOverlay);

  // Snapshot context at open time so palette filtering is stable
  // (the palette input will steal focus, changing focusedView).
  const snapshot = new Map<string, ContextValue>([
    ["focusedView", contextKeys.get("focusedView") ?? ""],
  ]);
  let selectedIndex = 0;

  function getFilteredCommands(): { id: string; label?: string; keybinding?: string; pluginId: string }[] {
    const query = input.value.toLowerCase();
    return commandRegistry.getAll()
      .filter((cmd) => {
        if (!contextKeys.evaluate(cmd.when, snapshot)) return false;
        if (!query) return true;
        const text = (cmd.label ?? cmd.id).toLowerCase();
        return text.includes(query);
      })
      .map((cmd) => ({
        id: cmd.id,
        label: cmd.label,
        keybinding: cmd.keybinding,
        pluginId: cmd.pluginId,
      }));
  }

  function renderList(): void {
    const commands = getFilteredCommands();
    list.innerHTML = "";
    selectedIndex = Math.min(selectedIndex, Math.max(0, commands.length - 1));

    for (let i = 0; i < commands.length; i++) {
      const cmd = commands[i]!;
      const row = document.createElement("div");
      row.style.cssText =
        `padding:0.5rem 1rem;cursor:pointer;display:flex;justify-content:space-between;` +
        `align-items:center;font-size:0.8125rem;` +
        `background:${i === selectedIndex ? "var(--bg-hover,#313244)" : "transparent"};`;

      const label = document.createElement("span");
      label.textContent = cmd.label ?? cmd.id;
      row.appendChild(label);

      const right = document.createElement("span");
      right.style.cssText = "display:flex;gap:0.5rem;align-items:center;";

      if (cmd.keybinding) {
        const kbd = document.createElement("kbd");
        kbd.textContent = cmd.keybinding;
        kbd.style.cssText =
          "font-size:0.6875rem;padding:0.125rem 0.375rem;border-radius:3px;" +
          "border:1px solid var(--border,#45475a);color:var(--text-muted,#a6adc8);";
        right.appendChild(kbd);
      }

      const plugin = document.createElement("span");
      plugin.textContent = cmd.pluginId.replace("talu.", "");
      plugin.style.cssText = "font-size:0.6875rem;color:var(--text-subtle,#6c7086);";
      right.appendChild(plugin);

      if (!cmd.pluginId.startsWith("talu.")) {
        const badge = document.createElement("span");
        badge.textContent = "ext";
        badge.style.cssText =
          "font-size:0.5625rem;font-weight:600;padding:0.0625rem 0.25rem;" +
          "background:var(--accent,#6366f1);color:var(--bg,#09090b);border-radius:2px;";
        right.appendChild(badge);
      }

      row.appendChild(right);

      row.addEventListener("click", () => {
        closePalette();
        commandRegistry.execute(cmd.id);
      });

      list.appendChild(row);
    }
  }

  input.addEventListener("input", () => {
    selectedIndex = 0;
    renderList();
  });

  input.addEventListener("keydown", (e) => {
    const commands = getFilteredCommands();
    if (e.key === "ArrowDown") {
      e.preventDefault();
      selectedIndex = Math.min(selectedIndex + 1, commands.length - 1);
      renderList();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      selectedIndex = Math.max(selectedIndex - 1, 0);
      renderList();
    } else if (e.key === "Enter") {
      e.preventDefault();
      const cmd = commands[selectedIndex];
      if (cmd) {
        closePalette();
        commandRegistry.execute(cmd.id);
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      closePalette();
    }
  });

  // Close on overlay click.
  paletteOverlay.addEventListener("click", (e) => {
    if (e.target === paletteOverlay) closePalette();
  });

  renderList();
  input.focus();
}

function closePalette(): void {
  if (!isOpen) return;
  isOpen = false;
  paletteOverlay?.remove();
  paletteOverlay = null;
}
