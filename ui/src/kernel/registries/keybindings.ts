/**
 * Keybinding Persistence — stores user keybinding overrides in localStorage.
 *
 * Overrides are keyed by command ID. When a command is registered, the
 * kernel checks for a user override before applying the manifest default.
 */

const STORAGE_KEY = "talu.keybindings";

/** In-memory cache, loaded once on boot. */
let overrides: Record<string, string> = {};

/** Load user keybinding overrides from localStorage. */
export function loadKeybindingOverrides(): void {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
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
  try {
    if (Object.keys(overrides).length === 0) {
      localStorage.removeItem(STORAGE_KEY);
    } else {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides));
    }
  } catch {
    // Storage full or unavailable — silent.
  }
}
