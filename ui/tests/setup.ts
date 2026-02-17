/**
 * Global test setup — initializes a DOM environment via HappyDOM
 * and polyfills browser APIs missing in the simulated environment.
 *
 * Loaded via `bun test --preload ./tests/setup.ts`.
 */
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { afterEach, mock } from "bun:test";

// Initialize HappyDOM (provides window, document, HTMLElement, ShadowRoot, etc.)
GlobalRegistrator.register();

// Polyfill ResizeObserver (not provided by HappyDOM)
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
} as unknown as typeof ResizeObserver;

// Polyfill IntersectionObserver (not provided by HappyDOM)
global.IntersectionObserver = class IntersectionObserver {
  readonly root: Element | Document | null = null;
  readonly rootMargin: string = "";
  readonly thresholds: ReadonlyArray<number> = [];
  observe() {}
  unobserve() {}
  disconnect() {}
  takeRecords(): IntersectionObserverEntry[] {
    return [];
  }
} as unknown as typeof IntersectionObserver;

// -- Event listener isolation ------------------------------------------------
//
// HappyDOM silently swallows errors thrown inside event listeners.  This guard
// surfaces those errors as test failures, and removes leaked document/window
// listeners between tests so one file's handlers can't fire during another's.

type Entry = {
  type: string;
  wrapped: EventListener;
  options?: boolean | AddEventListenerOptions;
};

const listenerErrors: Error[] = [];

function guardTarget(target: EventTarget) {
  const origAdd = target.addEventListener.bind(target);
  const origRemove = target.removeEventListener.bind(target);

  // Map original listener → our wrapper so removeEventListener still works.
  const wrapperMap = new Map<EventListenerOrEventListenerObject, EventListener>();
  const active: Entry[] = [];

  target.addEventListener = function (
    type: string,
    listener: EventListenerOrEventListenerObject | null,
    options?: boolean | AddEventListenerOptions,
  ) {
    if (!listener) return;
    const wrapped: EventListener = (e: Event) => {
      try {
        if (typeof listener === "function") listener.call(target, e);
        else listener.handleEvent(e);
      } catch (err) {
        listenerErrors.push(err instanceof Error ? err : new Error(String(err)));
        throw err; // re-throw so HappyDOM behaviour is unchanged
      }
    };
    wrapperMap.set(listener, wrapped);
    active.push({ type, wrapped, options });
    origAdd(type, wrapped, options);
  };

  target.removeEventListener = function (
    type: string,
    listener: EventListenerOrEventListenerObject | null,
    options?: boolean | EventListenerOptions,
  ) {
    if (!listener) return;
    const wrapped = wrapperMap.get(listener);
    if (wrapped) {
      wrapperMap.delete(listener);
      const idx = active.findIndex((a) => a.wrapped === wrapped);
      if (idx >= 0) active.splice(idx, 1);
      origRemove(type, wrapped, options);
    } else {
      origRemove(type, listener as EventListener, options);
    }
  };

  return {
    /** Remove all listeners that were not cleaned up by the test. */
    cleanup() {
      for (const { type, wrapped, options } of active) {
        origRemove(type, wrapped, options);
      }
      active.length = 0;
      wrapperMap.clear();
    },
  };
}

const docGuard = guardTarget(document);
const winGuard = guardTarget(window);

// Reset DOM, mocks, and event listeners after every test.
afterEach(() => {
  document.body.innerHTML = "";
  document.head.innerHTML = "";
  mock.restore();

  // Remove leaked document/window event listeners.
  docGuard.cleanup();
  winGuard.cleanup();

  // Fail the test if any event listener threw an error that HappyDOM swallowed.
  if (listenerErrors.length > 0) {
    const msgs = listenerErrors.map((e) => e.message).join("; ");
    listenerErrors.length = 0;
    throw new Error(`Event listener error(s) during test: ${msgs}`);
  }
});
