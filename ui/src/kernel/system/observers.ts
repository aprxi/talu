/**
 * Managed Observers — wraps MutationObserver, ResizeObserver, IntersectionObserver
 * with Disposable tracking, error boundaries, and auto-disconnect.
 *
 * Resource cap: 50 active observers per plugin instance.
 */

import type { Disposable, ManagedObservers } from "../types.ts";

const MAX_OBSERVERS = 50;

export class ManagedObserversImpl implements ManagedObservers, Disposable {
  private active = new Set<Disposable>();
  private pluginId: string;

  constructor(pluginId: string) {
    this.pluginId = pluginId;
  }

  private checkCap(): void {
    if (this.active.size >= MAX_OBSERVERS) {
      throw new Error(`Observer limit exceeded (${MAX_OBSERVERS}) for plugin "${this.pluginId}".`);
    }
  }

  mutation(
    target: Node,
    callback: (records: MutationRecord[]) => void,
    options?: MutationObserverInit,
  ): Disposable {
    this.checkCap();
    const observer = new MutationObserver((records) => {
      try {
        callback(records);
      } catch (err) {
        console.error(`[kernel] MutationObserver callback for "${this.pluginId}" threw:`, err);
      }
    });
    observer.observe(target, options);

    const disposable: Disposable = {
      dispose: () => {
        observer.disconnect();
        this.active.delete(disposable);
      },
    };
    this.active.add(disposable);
    return disposable;
  }

  resize(
    target: Element,
    callback: (entries: ResizeObserverEntry[]) => void,
    options?: ResizeObserverOptions,
  ): Disposable {
    this.checkCap();
    const observer = new ResizeObserver((entries) => {
      try {
        callback(entries);
      } catch (err) {
        console.error(`[kernel] ResizeObserver callback for "${this.pluginId}" threw:`, err);
      }
    });
    observer.observe(target, options);

    const disposable: Disposable = {
      dispose: () => {
        observer.disconnect();
        this.active.delete(disposable);
      },
    };
    this.active.add(disposable);
    return disposable;
  }

  intersection(
    target: Element,
    callback: (entries: IntersectionObserverEntry[]) => void,
    options?: IntersectionObserverInit,
  ): Disposable {
    this.checkCap();
    const observer = new IntersectionObserver((entries) => {
      try {
        callback(entries);
      } catch (err) {
        console.error(`[kernel] IntersectionObserver callback for "${this.pluginId}" threw:`, err);
      }
    }, options);
    observer.observe(target);

    const disposable: Disposable = {
      dispose: () => {
        observer.disconnect();
        this.active.delete(disposable);
      },
    };
    this.active.add(disposable);
    return disposable;
  }

  dispose(): void {
    for (const d of this.active) {
      d.dispose();
    }
    this.active.clear();
  }
}
