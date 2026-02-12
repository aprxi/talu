/**
 * Configuration Access — read plugin settings from manifest JSON Schema.
 *
 * Configuration values are loaded once during activation and cached.
 * Changes trigger onChange callbacks (debounced).
 */

import type { Disposable, ConfigurationAccess } from "../types.ts";

export class ConfigurationAccessImpl implements ConfigurationAccess {
  private config: unknown = {};
  private changeCallbacks = new Set<(config: unknown) => void>();
  private debounceTimer: ReturnType<typeof setTimeout> | null = null;

  /** Set the initial configuration (called by kernel during activation). */
  setConfig(config: unknown): void {
    this.config = config;
    this.notifyChange();
  }

  get<T = unknown>(): T {
    return this.config as T;
  }

  onChange<T = unknown>(callback: (config: T) => void): Disposable {
    const cb = callback as (config: unknown) => void;
    this.changeCallbacks.add(cb);
    return {
      dispose: () => {
        this.changeCallbacks.delete(cb);
      },
    };
  }

  /** Update config and notify listeners (debounced). */
  update(config: unknown): void {
    this.config = config;
    if (this.debounceTimer) clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => {
      this.debounceTimer = null;
      this.notifyChange();
    }, 100);
  }

  private notifyChange(): void {
    for (const cb of this.changeCallbacks) {
      try {
        cb(this.config);
      } catch (err) {
        console.error("[kernel] Configuration onChange callback threw:", err);
      }
    }
  }

  dispose(): void {
    if (this.debounceTimer) clearTimeout(this.debounceTimer);
    this.changeCallbacks.clear();
  }
}
