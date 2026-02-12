/**
 * Status Bar — renders manifest-declared statusBarItems and plugin status indicators.
 *
 * Items are positioned left/right with priority ordering.
 * Updates are driven by PluginStatusImpl setBusy/setReady.
 */

import type { PluginManifest, Disposable } from "../types.ts";

interface StatusBarItem {
  id: string;
  pluginId: string;
  label: string;
  alignment: "left" | "right";
  priority: number;
  element: HTMLElement;
}

export class StatusBarManager {
  private items = new Map<string, StatusBarItem>();
  private barEl: HTMLElement | null = null;
  private leftGroup: HTMLElement | null = null;
  private rightGroup: HTMLElement | null = null;

  constructor() {
    this.barEl = document.getElementById("status-bar");
    if (this.barEl) {
      this.leftGroup = document.createElement("div");
      this.leftGroup.className = "status-bar-left";
      this.rightGroup = document.createElement("div");
      this.rightGroup.className = "status-bar-right";
      this.barEl.appendChild(this.leftGroup);
      this.barEl.appendChild(this.rightGroup);
    }
  }

  /** Register status bar items declared in a plugin manifest. */
  registerFromManifest(pluginId: string, manifest: PluginManifest): Disposable {
    const items = manifest.contributes?.statusBarItems;
    if (!items || items.length === 0) return { dispose() {} };

    const ids: string[] = [];
    for (const decl of items) {
      const fqId = `${pluginId}.${decl.id}`;
      if (this.items.has(fqId)) continue;

      const el = document.createElement("span");
      el.className = "status-bar-item";
      el.textContent = decl.label;
      el.dataset["pluginId"] = pluginId;
      el.dataset["statusItemId"] = fqId;

      const item: StatusBarItem = {
        id: fqId,
        pluginId,
        label: decl.label,
        alignment: decl.alignment ?? "left",
        priority: decl.priority ?? 0,
        element: el,
      };

      this.items.set(fqId, item);
      ids.push(fqId);
      this.insertItem(item);
    }

    return {
      dispose: () => {
        for (const id of ids) {
          const item = this.items.get(id);
          if (item) {
            item.element.remove();
            this.items.delete(id);
          }
        }
      },
    };
  }

  /** Update a status bar item's label. */
  updateLabel(itemId: string, label: string): void {
    const item = this.items.get(itemId);
    if (item) {
      item.label = label;
      item.element.textContent = label;
    }
  }

  private insertItem(item: StatusBarItem): void {
    const group = item.alignment === "right" ? this.rightGroup : this.leftGroup;
    if (!group) return;

    // Insert in priority order (higher priority = earlier position).
    const existing = [...group.children] as HTMLElement[];
    let inserted = false;
    for (const child of existing) {
      const childId = child.dataset["statusItemId"];
      const childItem = childId ? this.items.get(childId) : null;
      if (childItem && childItem.priority < item.priority) {
        group.insertBefore(item.element, child);
        inserted = true;
        break;
      }
    }
    if (!inserted) {
      group.appendChild(item.element);
    }
  }
}
