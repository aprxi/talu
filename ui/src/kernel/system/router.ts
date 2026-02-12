/**
 * Hash Router — composite hash for per-plugin tab-specific state.
 *
 * Format: #/<pluginId>:<segment>;<pluginId>:<segment>
 * Each plugin sees only its own segment. The Kernel manages the composite.
 */

import type { Disposable, HashRouter } from "../types.ts";
import { resolveAlias } from "../core/alias.ts";

type HashChangeCallback = (hash: string) => void;

/** Parse composite hash into per-plugin segments. */
function parseCompositeHash(hash: string): Map<string, string> {
  const result = new Map<string, string>();
  // Strip leading #/ or #
  const raw = hash.replace(/^#\/?/, "");
  if (!raw) return result;

  for (const part of raw.split(";")) {
    const colonIdx = part.indexOf(":");
    if (colonIdx < 1) continue;
    const pluginId = resolveAlias(decodeURIComponent(part.slice(0, colonIdx)));
    const segment = decodeURIComponent(part.slice(colonIdx + 1));
    result.set(pluginId, segment);
  }
  return result;
}

/** Serialize per-plugin segments into composite hash. */
function serializeCompositeHash(segments: Map<string, string>): string {
  const parts: string[] = [];
  for (const [pluginId, segment] of segments) {
    if (segment) {
      parts.push(`${encodeURIComponent(pluginId)}:${encodeURIComponent(segment)}`);
    }
  }
  return parts.length > 0 ? `#/${parts.join(";")}` : "";
}

/** Global state shared across all HashRouterImpl instances. */
let segments = parseCompositeHash(window.location.hash);
const listeners = new Map<string, Set<HashChangeCallback>>();
let windowListenerInstalled = false;

function installWindowListener(): void {
  if (windowListenerInstalled) return;
  windowListenerInstalled = true;
  window.addEventListener("hashchange", () => {
    const prev = segments;
    segments = parseCompositeHash(window.location.hash);

    // Notify plugins whose segment changed.
    for (const [pluginId, cbs] of listeners) {
      const oldSeg = prev.get(pluginId) ?? "";
      const newSeg = segments.get(pluginId) ?? "";
      if (oldSeg !== newSeg) {
        for (const cb of cbs) {
          try {
            cb(newSeg);
          } catch (err) {
            console.error(`[kernel] HashRouter callback for "${pluginId}" threw:`, err);
          }
        }
      }
    }
  });
}

export class HashRouterImpl implements HashRouter {
  private pluginId: string;

  constructor(pluginId: string) {
    this.pluginId = pluginId;
    installWindowListener();
  }

  getHash(): string {
    return segments.get(this.pluginId) ?? "";
  }

  setHash(path: string, options?: { history?: "replace" | "push" }): void {
    segments.set(this.pluginId, path);
    const hash = serializeCompositeHash(segments);
    const method = options?.history ?? "replace";

    if (method === "push") {
      window.history.pushState(null, "", hash || window.location.pathname);
    } else {
      window.history.replaceState(null, "", hash || window.location.pathname);
    }

    // Notify own listeners.
    const cbs = listeners.get(this.pluginId);
    if (cbs) {
      for (const cb of cbs) {
        try {
          cb(path);
        } catch (err) {
          console.error(`[kernel] HashRouter callback for "${this.pluginId}" threw:`, err);
        }
      }
    }
  }

  onHashChange(callback: HashChangeCallback): Disposable {
    let set = listeners.get(this.pluginId);
    if (!set) {
      set = new Set();
      listeners.set(this.pluginId, set);
    }
    set.add(callback);
    return {
      dispose: () => {
        set!.delete(callback);
        if (set!.size === 0) listeners.delete(this.pluginId);
      },
    };
  }
}
