import type { Disposable, EventBus } from "../types.ts";

type Handler = (data: unknown) => void;

export class EventBusImpl implements EventBus {
  private listeners = new Map<string, Set<Handler>>();

  on<T = unknown>(event: string, handler: (data: T) => void): Disposable {
    let set = this.listeners.get(event);
    if (!set) {
      set = new Set();
      this.listeners.set(event, set);
    }
    const h = handler as Handler;
    set.add(h);
    return {
      dispose: () => {
        set!.delete(h);
        if (set!.size === 0) this.listeners.delete(event);
      },
    };
  }

  once<T = unknown>(event: string, handler: (data: T) => void): Disposable {
    const sub = this.on<T>(event, (data) => {
      sub.dispose();
      handler(data);
    });
    return sub;
  }

  emit<T = unknown>(event: string, data: T): void {
    const set = this.listeners.get(event);
    if (!set) return;
    const frozen = Object.freeze(data);
    for (const handler of set) {
      try {
        handler(frozen);
      } catch (err) {
        console.error(`[kernel] EventBus handler for "${event}" threw:`, err);
      }
    }
  }
}
