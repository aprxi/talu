/**
 * Global Event Wrappers — ctx.events.onWindow/onDocument.
 *
 * Wraps addEventListener on window/document with Disposable tracking
 * and error boundaries. Auto-removed on plugin deactivation.
 */

import type { Disposable } from "../types.ts";

export class GlobalEventManager implements Disposable {
  private active = new Set<Disposable>();
  private pluginId: string;

  constructor(pluginId: string) {
    this.pluginId = pluginId;
  }

  onWindow(
    eventName: string,
    handler: EventListener,
    options?: AddEventListenerOptions,
  ): Disposable {
    const wrapped: EventListener = (event) => {
      try {
        handler(event);
      } catch (err) {
        console.error(`[kernel] Window event "${eventName}" handler for "${this.pluginId}" threw:`, err);
      }
    };
    window.addEventListener(eventName, wrapped, options);

    const disposable: Disposable = {
      dispose: () => {
        window.removeEventListener(eventName, wrapped, options);
        this.active.delete(disposable);
      },
    };
    this.active.add(disposable);
    return disposable;
  }

  onDocument(
    eventName: string,
    handler: EventListener,
    options?: AddEventListenerOptions,
  ): Disposable {
    const wrapped: EventListener = (event) => {
      try {
        handler(event);
      } catch (err) {
        console.error(`[kernel] Document event "${eventName}" handler for "${this.pluginId}" threw:`, err);
      }
    };
    document.addEventListener(eventName, wrapped, options);

    const disposable: Disposable = {
      dispose: () => {
        document.removeEventListener(eventName, wrapped, options);
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
