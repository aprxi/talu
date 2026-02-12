/**
 * ID namespacing — auto-prefix all registered identifiers with pluginId.
 *
 * Prevents collisions between plugins that independently choose the same
 * local name. The `talu.*` namespace is reserved for built-in plugins.
 */

/** Auto-prefix unless the ID already starts with `${pluginId}.`. */
export function namespacedId(pluginId: string, localId: string): string {
  if (localId.startsWith(`${pluginId}.`)) return localId;
  return `${pluginId}.${localId}`;
}

/**
 * Validate a local ID for runtime registration.
 * Third-party plugins cannot use dots in IDs (prevents impersonation).
 * No plugin can use the `talu.*` namespace unless built-in.
 */
export function validateLocalId(pluginId: string, localId: string, isBuiltin: boolean): void {
  if (!isBuiltin && localId.includes(".")) {
    throw new Error(
      `Registration ID must be a local name (no dots). Got: '${localId}'. ` +
      `The Kernel will prefix it with your plugin ID.`,
    );
  }

  // Check that the resulting fully-qualified ID doesn't claim talu.* for non-built-ins.
  const fqId = namespacedId(pluginId, localId);
  if (!isBuiltin && fqId.startsWith("talu.")) {
    throw new Error(`Namespace 'talu.*' is reserved for built-in plugins.`);
  }
}
