/**
 * Keybinding Persistence — stores user keybinding overrides via KV backend.
 *
 * Overrides are keyed by command ID. When a command is registered, the
 * kernel checks for a user override before applying the manifest default.
 * Reads KV first, falls back to localStorage. Writes to both KV (primary)
 * and localStorage (sync cache).
 */

import { getSetting, setSetting, deleteSetting } from "../system/kv-settings.ts";

const STORAGE_KEY = "talu.keybindings";

/** In-memory cache, loaded once on boot. */
let overrides: Record<string, string> = {};

/** Load user keybinding overrides from KV (falls back to localStorage). */
export async function loadKeybindingOverrides(): Promise<void> {
  try {
    const raw = await getSetting(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (typeof parsed === "object" && parsed !== null) {
        overrides = parsed as Record<string, string>;
      }
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
  const json = Object.keys(overrides).length === 0 ? null : JSON.stringify(overrides);
  // KV primary (fire-and-forget).
  if (json) {
    void setSetting(STORAGE_KEY, json);
  } else {
    void deleteSetting(STORAGE_KEY);
  }
  // localStorage sync cache.
  try {
    if (json) {
      localStorage.setItem(STORAGE_KEY, json);
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  } catch { /* storage full or unavailable */ }
}
