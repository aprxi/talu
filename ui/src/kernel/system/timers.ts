/**
 * Managed Timers — wraps setTimeout/setInterval/rAF with error boundaries
 * and auto-disposal on plugin deactivation.
 *
 * Resource cap: 100 active timers per plugin instance.
 */

import type { Disposable, ManagedTimers } from "../types.ts";

const MAX_TIMERS = 100;

export class ManagedTimersImpl implements ManagedTimers, Disposable {
  private active = new Set<Disposable>();
  private pluginId: string;

  constructor(pluginId: string) {
    this.pluginId = pluginId;
  }

  private checkCap(): void {
    if (this.active.size >= MAX_TIMERS) {
      throw new Error(`Timer limit exceeded (${MAX_TIMERS}) for plugin "${this.pluginId}".`);
    }
  }

  private wrapCallback(callback: () => void): () => void {
    return () => {
      try {
        callback();
      } catch (err) {
        console.error(`[kernel] Timer callback for "${this.pluginId}" threw:`, err);
      }
    };
  }

  setTimeout(callback: () => void, ms: number): Disposable {
    this.checkCap();
    const wrapped = this.wrapCallback(callback);
    const id = window.setTimeout(() => {
      this.active.delete(disposable);
      wrapped();
    }, ms);
    const disposable: Disposable = {
      dispose: () => {
        window.clearTimeout(id);
        this.active.delete(disposable);
      },
    };
    this.active.add(disposable);
    return disposable;
  }

  setInterval(callback: () => void, ms: number): Disposable {
    this.checkCap();
    const wrapped = this.wrapCallback(callback);
    const id = window.setInterval(wrapped, ms);
    const disposable: Disposable = {
      dispose: () => {
        window.clearInterval(id);
        this.active.delete(disposable);
      },
    };
    this.active.add(disposable);
    return disposable;
  }

  requestAnimationFrame(callback: FrameRequestCallback): Disposable {
    this.checkCap();
    const id = window.requestAnimationFrame((time) => {
      this.active.delete(disposable);
      try {
        callback(time);
      } catch (err) {
        console.error(`[kernel] rAF callback for "${this.pluginId}" threw:`, err);
      }
    });
    const disposable: Disposable = {
      dispose: () => {
        window.cancelAnimationFrame(id);
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
