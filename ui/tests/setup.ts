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

// Reset DOM and mocks after every test to prevent cross-test pollution.
afterEach(() => {
  document.body.innerHTML = "";
  document.head.innerHTML = "";
  mock.restore();
});
