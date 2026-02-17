/**
 * Context Key Service — reactive key-value store for when-clause evaluation.
 *
 * Kernel singleton. Keys are strings, values are strings or booleans.
 * Supports per-key change subscriptions and when-clause evaluation:
 *   key == 'value', key != 'value', bare key (truthy), !key (falsy).
 */

import type { Disposable, ContextValue } from "../types.ts";

type ChangeCallback = (key: string, value: ContextValue | undefined) => void;

export class ContextKeyService implements Disposable {
  private store = new Map<string, ContextValue>();
  private listeners = new Map<string, Set<ChangeCallback>>();

  set(key: string, value: ContextValue): Disposable {
    const prev = this.store.get(key);
    this.store.set(key, value);
    if (prev !== value) {
      this.notify(key, value);
    }
    return {
      dispose: () => {
        // Only delete if the value was not overwritten by a later set().
        if (this.store.get(key) === value) {
          this.store.delete(key);
          this.notify(key, undefined);
        }
      },
    };
  }

  get(key: string): ContextValue | undefined {
    return this.store.get(key);
  }

  has(key: string): boolean {
    return this.store.has(key);
  }

  delete(key: string): void {
    if (this.store.has(key)) {
      this.store.delete(key);
      this.notify(key, undefined);
    }
  }

  onChange(key: string, callback: ChangeCallback): Disposable {
    let set = this.listeners.get(key);
    if (!set) {
      set = new Set();
      this.listeners.set(key, set);
    }
    set.add(callback);
    return {
      dispose: () => {
        set!.delete(callback);
        if (set!.size === 0) this.listeners.delete(key);
      },
    };
  }

  /**
   * Evaluate a when-clause expression against the current context.
   * Optional overrides map takes precedence over stored values (for snapshots).
   */
  evaluate(when?: string, overrides?: ReadonlyMap<string, ContextValue>): boolean {
    if (!when) return true;

    const resolve = (key: string): ContextValue => {
      if (overrides?.has(key)) return overrides.get(key)!;
      return this.store.get(key) ?? "";
    };

    // key == 'value'
    const eqMatch = when.match(/^(\w+)\s*==\s*'([^']*)'$/);
    if (eqMatch) {
      return String(resolve(eqMatch[1]!)) === eqMatch[2]!;
    }

    // key != 'value'
    const neqMatch = when.match(/^(\w+)\s*!=\s*'([^']*)'$/);
    if (neqMatch) {
      return String(resolve(neqMatch[1]!)) !== neqMatch[2]!;
    }

    // !key (falsy)
    const negMatch = when.match(/^!(\w+)$/);
    if (negMatch) {
      return !this.isTruthy(resolve(negMatch[1]!));
    }

    // bare key (truthy)
    const bareMatch = when.match(/^(\w+)$/);
    if (bareMatch) {
      return this.isTruthy(resolve(bareMatch[1]!));
    }

    // Unknown expression — permissive (matches existing behavior).
    return true;
  }

  dispose(): void {
    this.store.clear();
    this.listeners.clear();
  }

  private isTruthy(value: ContextValue): boolean {
    if (value === false || value === "") return false;
    return true;
  }

  private notify(key: string, value: ContextValue | undefined): void {
    const set = this.listeners.get(key);
    if (!set) return;
    for (const cb of set) {
      try {
        cb(key, value);
      } catch (err) {
        console.error(`[kernel] Context key change listener for "${key}" threw:`, err);
      }
    }
  }
}
