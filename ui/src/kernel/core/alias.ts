/**
 * Alias Resolution — redirect table for renamed plugin IDs.
 *
 * Manifest `aliases` field maps old fully-qualified IDs to new ones.
 * Applied at all registry lookup sites (services, commands, tools, router).
 */

/** Global alias table, populated during plugin registration. */
const aliasTable = new Map<string, string>();

/** Register aliases from a plugin manifest. */
export function registerAliases(aliases: Record<string, string> | undefined): void {
  if (!aliases) return;
  for (const [from, to] of Object.entries(aliases)) {
    if (aliasTable.has(from)) {
      console.warn(`[kernel] Alias "${from}" already defined — ignoring duplicate.`);
      continue;
    }
    aliasTable.set(from, to);
  }
}

/** Resolve an ID through the alias table. Returns the canonical ID. */
export function resolveAlias(id: string): string {
  const resolved = aliasTable.get(id);
  if (resolved !== undefined) {
    return resolved;
  }
  return id;
}
