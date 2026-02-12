/**
 * Activation Event Evaluator — partition plugins into eager/lazy,
 * parse activation event strings.
 *
 * Plugins with no activationEvents or ["*"] are eager (loaded immediately).
 * Others are lazy (deferred until their activation event fires).
 * Eager dependency closure: if A requires B and B is lazy, promote B to eager.
 */

import type { PluginManifest } from "../types.ts";

export interface PluginDescriptor {
  manifest: PluginManifest;
  entryUrl: string;
  token?: string;
}

export interface ActivationPartition {
  eager: PluginDescriptor[];
  lazy: PluginDescriptor[];
}

/**
 * Partition plugins into eager (load immediately) and lazy (defer until event).
 * Applies eager dependency closure: if A requires B and B is lazy, B is promoted.
 */
export function partitionByActivation(plugins: PluginDescriptor[]): ActivationPartition {
  const classification = new Map<string, "eager" | "lazy">();

  // First pass: classify each plugin.
  for (const p of plugins) {
    const events = p.manifest.activationEvents;
    if (!events || events.length === 0 || events.includes("*")) {
      classification.set(p.manifest.id, "eager");
    } else {
      classification.set(p.manifest.id, "lazy");
    }
  }

  // Second pass: promote lazy deps to eager (transitive closure).
  let changed = true;
  while (changed) {
    changed = false;
    for (const p of plugins) {
      if (classification.get(p.manifest.id) !== "eager") continue;
      for (const dep of p.manifest.requires ?? []) {
        if (classification.get(dep.id) === "lazy") {
          classification.set(dep.id, "eager");
          changed = true;
        }
      }
    }
  }

  // Third pass: partition.
  const eager: PluginDescriptor[] = [];
  const lazy: PluginDescriptor[] = [];
  for (const p of plugins) {
    if (classification.get(p.manifest.id) === "eager") {
      eager.push(p);
    } else {
      lazy.push(p);
    }
  }

  return { eager, lazy };
}

/**
 * Parse an activation event string into its type and argument.
 * Returns null for unrecognized patterns.
 */
export function parseActivationEvent(event: string): { type: string; arg: string } | null {
  const viewMatch = event.match(/^onView:(.+)$/);
  if (viewMatch && viewMatch[1]) return { type: "onView", arg: viewMatch[1] };

  const cmdMatch = event.match(/^onCommand:(.+)$/);
  if (cmdMatch && cmdMatch[1]) return { type: "onCommand", arg: cmdMatch[1] };

  const langMatch = event.match(/^onLanguage:(.+)$/);
  if (langMatch && langMatch[1]) return { type: "onLanguage", arg: langMatch[1] };

  return null;
}

/**
 * Topological sort of plugins by `requires` dependencies.
 * Visit-based DFS with cycle detection. Unknown deps silently skipped.
 * Throws on dependency cycles.
 */
export function topologicalSort(plugins: PluginDescriptor[]): PluginDescriptor[] {
  const byId = new Map(plugins.map((p) => [p.manifest.id, p]));
  const visited = new Set<string>();
  const visiting = new Set<string>(); // Gray nodes — currently in DFS stack.
  const sorted: PluginDescriptor[] = [];

  function visit(id: string, path: string[]): void {
    if (visited.has(id)) return;

    if (visiting.has(id)) {
      const cycle = [...path.slice(path.indexOf(id)), id].join(" -> ");
      throw new Error(`[kernel] Dependency cycle detected: ${cycle}`);
    }

    visiting.add(id);
    const plugin = byId.get(id);
    if (!plugin) {
      visiting.delete(id);
      return;
    }

    for (const dep of plugin.manifest.requires ?? []) {
      visit(dep.id, [...path, id]);
    }

    visiting.delete(id);
    visited.add(id);
    sorted.push(plugin);
  }

  for (const p of plugins) visit(p.manifest.id, []);
  return sorted;
}
