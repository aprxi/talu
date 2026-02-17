/**
 * Menu Registry — manifest-driven menu contributions to named slots.
 *
 * Plugins declare `contributes.menus` to add buttons/actions to slots in other
 * plugins' UI. Items are filtered by when-clause, sorted by priority, and
 * rendered as buttons that execute registered commands on click.
 *
 * Follows the StatusBarManager pattern: FQ-ID namespacing, priority ordering,
 * Disposable lifecycle.
 */

import type { PluginManifest, Disposable } from "../types.ts";
import type { ContextKeyService } from "./context-keys.ts";
import type { CommandRegistryImpl } from "./commands.ts";

export interface MenuItemDeclaration {
  id: string;
  slot: string;
  label: string;
  icon?: string;
  command: string;
  when?: string;
  priority?: number;
}

interface MenuItem {
  id: string;
  pluginId: string;
  slot: string;
  label: string;
  icon?: string;
  command: string;
  when?: string;
  priority: number;
}

export class MenuRegistry {
  private items = new Map<string, MenuItem>();
  private slotIndex = new Map<string, Set<string>>();
  private slotListeners = new Map<string, Set<() => void>>();
  private contextKeys: ContextKeyService;
  private commandRegistry: CommandRegistryImpl;

  constructor(contextKeys: ContextKeyService, commandRegistry: CommandRegistryImpl) {
    this.contextKeys = contextKeys;
    this.commandRegistry = commandRegistry;
  }

  /** Register menu items declared in a plugin manifest. */
  registerFromManifest(pluginId: string, manifest: PluginManifest): Disposable {
    const menus = manifest.contributes?.menus;
    if (!menus || menus.length === 0) return { dispose() {} };

    const ids: string[] = [];
    for (const decl of menus) {
      const fqId = `${pluginId}.${decl.id}`;
      if (this.items.has(fqId)) continue;

      const item: MenuItem = {
        id: fqId,
        pluginId,
        slot: decl.slot,
        label: decl.label,
        icon: decl.icon,
        command: decl.command,
        when: decl.when,
        priority: decl.priority ?? 0,
      };

      this.items.set(fqId, item);
      ids.push(fqId);
      this.addToSlotIndex(decl.slot, fqId);
      this.notifySlot(decl.slot);
    }

    return {
      dispose: () => {
        for (const id of ids) {
          const item = this.items.get(id);
          if (item) {
            this.removeFromSlotIndex(item.slot, id);
            this.items.delete(id);
            this.notifySlot(item.slot);
          }
        }
      },
    };
  }

  /** Register a single menu item imperatively. */
  registerItem(pluginId: string, decl: MenuItemDeclaration): Disposable {
    const fqId = `${pluginId}.${decl.id}`;
    if (this.items.has(fqId)) return { dispose() {} };

    const item: MenuItem = {
      id: fqId,
      pluginId,
      slot: decl.slot,
      label: decl.label,
      icon: decl.icon,
      command: decl.command,
      when: decl.when,
      priority: decl.priority ?? 0,
    };

    this.items.set(fqId, item);
    this.addToSlotIndex(decl.slot, fqId);
    this.notifySlot(decl.slot);

    return {
      dispose: () => {
        if (this.items.has(fqId)) {
          this.removeFromSlotIndex(decl.slot, fqId);
          this.items.delete(fqId);
          this.notifySlot(decl.slot);
        }
      },
    };
  }

  /** Get visible items for a slot, filtered by when-clause, sorted by priority (higher first). */
  getItems(slotId: string): MenuItem[] {
    const ids = this.slotIndex.get(slotId);
    if (!ids || ids.size === 0) return [];

    const result: MenuItem[] = [];
    for (const id of ids) {
      const item = this.items.get(id);
      if (!item) continue;
      if (!this.contextKeys.evaluate(item.when)) continue;
      result.push(item);
    }

    result.sort((a, b) => b.priority - a.priority);
    return result;
  }

  /**
   * Render contributed menu items into a container element.
   * Creates buttons for each visible item and wires click → command execution.
   * Re-renders automatically when items for the slot change.
   */
  renderSlot(slotId: string, container: HTMLElement): Disposable {
    let disposed = false;

    const render = () => {
      if (disposed) return;
      container.innerHTML = "";
      const items = this.getItems(slotId);
      for (const item of items) {
        const btn = document.createElement("button");
        btn.className = "menu-slot-btn";
        btn.title = item.label;
        btn.dataset["menuItemId"] = item.id;
        if (item.icon) {
          btn.innerHTML = item.icon;
        } else {
          btn.textContent = item.label;
        }
        btn.addEventListener("click", () => {
          this.commandRegistry.execute(item.command);
        });
        container.appendChild(btn);
      }
    };

    render();

    const listeners = this.getOrCreateSlotListeners(slotId);
    listeners.add(render);

    return {
      dispose: () => {
        disposed = true;
        listeners.delete(render);
        container.innerHTML = "";
      },
    };
  }

  dispose(): void {
    this.items.clear();
    this.slotIndex.clear();
    this.slotListeners.clear();
  }

  // --- Private helpers ---

  private addToSlotIndex(slot: string, fqId: string): void {
    let ids = this.slotIndex.get(slot);
    if (!ids) {
      ids = new Set();
      this.slotIndex.set(slot, ids);
    }
    ids.add(fqId);
  }

  private removeFromSlotIndex(slot: string, fqId: string): void {
    const ids = this.slotIndex.get(slot);
    if (ids) {
      ids.delete(fqId);
      if (ids.size === 0) this.slotIndex.delete(slot);
    }
  }

  private notifySlot(slot: string): void {
    const listeners = this.slotListeners.get(slot);
    if (!listeners) return;
    for (const listener of listeners) {
      try {
        listener();
      } catch (err) {
        console.error(`[kernel] Menu slot "${slot}" listener threw:`, err);
      }
    }
  }

  private getOrCreateSlotListeners(slot: string): Set<() => void> {
    let listeners = this.slotListeners.get(slot);
    if (!listeners) {
      listeners = new Set();
      this.slotListeners.set(slot, listeners);
    }
    return listeners;
  }
}
