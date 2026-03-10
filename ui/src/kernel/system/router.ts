/**
 * Global Hash Router — URL-driven navigation for the entire app.
 *
 * Format: #/<mode>[/<sub>[/<resource>]]
 *
 * Examples:
 *   #/chat                 → chat welcome
 *   #/chat/sess_abc        → specific conversation
 *   #/routing/terminal/x   → terminal for host "x"
 *   #/files/archived       → files archived tab
 *   #/settings/appearance  → settings appearance tab
 *
 * Empty hash or #/ defaults to { mode: "chat", sub: null, resource: null }.
 */

import type { Disposable, HashRouter } from "../types.ts";

// ---------------------------------------------------------------------------
// Route types
// ---------------------------------------------------------------------------

export interface ParsedRoute {
  mode: string;
  sub: string | null;
  resource: string | null;
}

// ---------------------------------------------------------------------------
// Parsing & serialization
// ---------------------------------------------------------------------------

export function parseHash(hash: string): ParsedRoute {
  const raw = hash.replace(/^#\/?/, "");
  if (!raw) return { mode: "chat", sub: null, resource: null };

  const segments = raw.split("/").map(decodeURIComponent);
  return {
    mode: segments[0] || "chat",
    sub: segments[1] || null,
    resource: segments[2] || null,
  };
}

export function serializeRoute(route: ParsedRoute): string {
  let hash = `#/${encodeURIComponent(route.mode)}`;
  if (route.sub) {
    hash += `/${encodeURIComponent(route.sub)}`;
    if (route.resource) {
      hash += `/${encodeURIComponent(route.resource)}`;
    }
  }
  return hash;
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

type RouteChangeCallback = (route: ParsedRoute, previous: ParsedRoute) => void;

let current: ParsedRoute = parseHash(window.location.hash);
const routeListeners = new Set<RouteChangeCallback>();
let initialized = false;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function getCurrentRoute(): ParsedRoute {
  return current;
}

/** Navigate to a route. Uses pushState by default; set replace:true for replaceState. */
export function navigate(route: ParsedRoute, opts?: { replace?: boolean }): void {
  const previous = current;
  const hash = serializeRoute(route);

  // No-op if hash hasn't changed.
  if (hash === serializeRoute(previous)) return;

  current = route;

  if (opts?.replace) {
    window.history.replaceState(null, "", hash);
  } else {
    window.history.pushState(null, "", hash);
  }

  notifyListeners(route, previous);
}

/** Subscribe to route changes (from navigate() calls and browser Back/Forward). */
export function onRouteChange(callback: RouteChangeCallback): Disposable {
  routeListeners.add(callback);
  return {
    dispose() {
      routeListeners.delete(callback);
    },
  };
}

/** Fire the current route to all listeners. Call once after boot to handle deep links. */
export function replayCurrentRoute(): void {
  notifyListeners(current, { mode: "", sub: null, resource: null });
}

/** Install the hashchange listener for browser Back/Forward. Call once at boot. */
export function initRouter(): Disposable {
  if (initialized) return { dispose() {} };
  initialized = true;

  const handler = () => {
    const previous = current;
    current = parseHash(window.location.hash);

    // Only notify if route actually changed (avoids duplicate with navigate()).
    if (serializeRoute(current) !== serializeRoute(previous)) {
      notifyListeners(current, previous);
    }
  };

  window.addEventListener("hashchange", handler);
  return {
    dispose() {
      window.removeEventListener("hashchange", handler);
      initialized = false;
    },
  };
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

function notifyListeners(route: ParsedRoute, previous: ParsedRoute): void {
  for (const cb of routeListeners) {
    try {
      cb(route, previous);
    } catch (err) {
      console.error("[kernel] Route change callback threw:", err);
    }
  }
}

// ---------------------------------------------------------------------------
// Legacy HashRouter stub (unused by plugins, kept for ctx.router type compat)
// ---------------------------------------------------------------------------

export class HashRouterImpl implements HashRouter {
  getHash(): string {
    return "";
  }

  setHash(_path: string, _options?: { history?: "replace" | "push" }): void {
    // no-op — use navigate() from the global router instead.
  }

  onHashChange(_callback: (hash: string) => void): Disposable {
    return { dispose() {} };
  }
}
