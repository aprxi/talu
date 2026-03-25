/**
 * Keybinding Persistence — stores user keybinding overrides via KV backend.
 *
 * Overrides are keyed by command ID. When a command is registered, the
 * kernel checks for a user override before applying the manifest default.
 * Reads KV first, falls back to localStorage. Writes to both KV (primary)
 * and localStorage (sync cache).
 */

import { preferences } from "../system/preferences.ts";

/** In-memory cache, loaded once on boot. */
let overrides: Record<string, string> = {};

/** Load user keybinding overrides from unified preferences. */
export async function loadKeybindingOverrides(): Promise<void> {
  try {
    const stored = preferences.get<Record<string, string>>("kernel", "keybindings");
    if (stored && typeof stored === "object") {
      overrides = stored;
    }
  } catch {
    overrides = {};
  }
}

/** Get the effective keybinding for a command: user override or default. */
export function resolveKeybinding(commandId: string, defaultKeybinding?: string): string | undefined {
  return overrides[commandId] ?? defaultKeybinding;
}

/** Set a user keybinding override for a command. */
export function setKeybindingOverride(commandId: string, keybinding: string): void {
  overrides[commandId] = keybinding;
  persistOverrides();
}

/** Remove a user keybinding override, reverting to the default. */
export function removeKeybindingOverride(commandId: string): void {
  delete overrides[commandId];
  persistOverrides();
}

/** Get all user overrides (for UI display). */
export function getKeybindingOverrides(): Readonly<Record<string, string>> {
  return overrides;
}

function persistOverrides(): void {
  preferences.set("kernel", "keybindings", overrides);
  // localStorage sync cache.
  try {
    const json = Object.keys(overrides).length === 0 ? null : JSON.stringify(overrides);
    if (json) {
      localStorage.setItem("talu.keybindings", json);
    } else {
      localStorage.removeItem("talu.keybindings");
    }
  } catch { /* storage full or unavailable */ }
}
