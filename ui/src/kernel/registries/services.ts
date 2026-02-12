import type { Disposable, ServiceAccess } from "../types.ts";
import type { EventBusImpl } from "../system/event-bus.ts";
import { resolveAlias } from "../core/alias.ts";

type ChangeCallback = (service: unknown | undefined) => void;

export class ServiceRegistry implements ServiceAccess {
  private registry = new Map<string, unknown>();
  private changeListeners = new Map<string, Set<ChangeCallback>>();

  constructor(private eventBus?: EventBusImpl) {}

  get<T = unknown>(id: string): T | undefined {
    return this.registry.get(resolveAlias(id)) as T | undefined;
  }

  provide<T = unknown>(id: string, instance: T): Disposable {
    if (this.registry.has(id)) {
      console.warn(`[kernel] Service "${id}" already registered — ignoring duplicate.`);
      return { dispose() {} };
    }
    this.registry.set(id, instance);
    this.notifyChange(id, instance);
    this.eventBus?.emit("system.service.registered", { serviceId: id });
    return {
      dispose: () => {
        if (this.registry.get(id) === instance) {
          this.registry.delete(id);
          this.notifyChange(id, undefined);
          this.eventBus?.emit("system.service.unregistered", { serviceId: id });
        }
      },
    };
  }

  onDidChange(serviceId: string, callback: ChangeCallback): Disposable {
    const resolved = resolveAlias(serviceId);
    let set = this.changeListeners.get(resolved);
    if (!set) {
      set = new Set();
      this.changeListeners.set(resolved, set);
    }
    set.add(callback);
    return {
      dispose: () => {
        set!.delete(callback);
        if (set!.size === 0) this.changeListeners.delete(resolved);
      },
    };
  }

  private notifyChange(serviceId: string, service: unknown | undefined): void {
    const listeners = this.changeListeners.get(serviceId);
    if (!listeners) return;
    for (const cb of listeners) {
      try {
        cb(service);
      } catch (err) {
        console.error(`[kernel] Service change listener for "${serviceId}" threw:`, err);
      }
    }
  }
}
