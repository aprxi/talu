/**
 * Capability Requirements — validate requiresCapabilities manifest field.
 *
 * The Kernel checks declared capabilities against its known set before
 * loading a plugin. Unknown capabilities → plugin not loaded.
 */

/** V1 known capability IDs matching PluginContext sub-APIs. */
const KNOWN_CAPABILITIES = new Set([
  "hooks",
  "tools",
  "popover",
  "storage",
  "router",
  "dialogs",
  "timers",
  "observers",
  "commands",
  "renderers",
  "themes",
]);

export interface CapabilityCheckResult {
  satisfied: boolean;
  unsatisfied: string[];
}

/**
 * Check if all required capabilities are supported.
 * Returns unsatisfied capabilities (if any).
 */
export function checkCapabilities(required: string[] | undefined): CapabilityCheckResult {
  if (!required || required.length === 0) {
    return { satisfied: true, unsatisfied: [] };
  }

  const unsatisfied = required.filter((cap) => !KNOWN_CAPABILITIES.has(cap));
  return {
    satisfied: unsatisfied.length === 0,
    unsatisfied,
  };
}

/** Get the list of all V1 known capabilities. */
export function getKnownCapabilities(): string[] {
  return [...KNOWN_CAPABILITIES];
}
