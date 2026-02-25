/**
 * Kernel-level KV settings helper.
 *
 * Wraps the backend KV namespace "ui" for storing kernel settings
 * (keybindings, last active mode, collapsed groups, integrity hashes).
 * Falls back to localStorage when the API is not yet initialized.
 */

import type { ApiClient } from "../../api.ts";

const NAMESPACE = "ui";

type KvApi = Pick<ApiClient, "kvGet" | "kvPut" | "kvDelete" | "kvList">;

let apiRef: KvApi | null = null;

/** Initialize with a kernel-level API client. Call once during boot. */
export function initKvSettings(api: KvApi): void {
  apiRef = api;
}

/** Read a setting from the KV backend, falling back to localStorage. */
export async function getSetting(key: string): Promise<string | null> {
  if (apiRef) {
    try {
      const result = await apiRef.kvGet(NAMESPACE, key);
      if (result.ok && result.data?.value != null) return result.data.value;
    } catch { /* fall through to localStorage */ }
  }
  try { return localStorage.getItem(key); } catch { return null; }
}

/** Write a setting to the KV backend. */
export async function setSetting(key: string, value: string): Promise<void> {
  if (apiRef) {
    await apiRef.kvPut(NAMESPACE, key, value);
    return;
  }
  try { localStorage.setItem(key, value); } catch { /* ignore */ }
}

/** Delete a setting from the KV backend. */
export async function deleteSetting(key: string): Promise<void> {
  if (apiRef) {
    await apiRef.kvDelete(NAMESPACE, key);
    return;
  }
  try { localStorage.removeItem(key); } catch { /* ignore */ }
}

// ---------------------------------------------------------------------------
// One-time migration
// ---------------------------------------------------------------------------

const MIGRATION_FLAG = "talu-kv-migrated";

/**
 * Migrate known localStorage keys to the KV backend.
 *
 * Idempotent — guarded by a localStorage flag. Each key is best-effort;
 * individual failures do not block migration of remaining keys.
 */
export async function migrateLocalStorageToKv(): Promise<void> {
  if (!apiRef) return;
  try { if (localStorage.getItem(MIGRATION_FLAG)) return; } catch { return; }

  const keysToMigrate = [
    "talu-last-active-mode",
    "talu.keybindings",
    "talu:integrityHashes",
    "talu-collapsed-groups",
  ];

  for (const key of keysToMigrate) {
    try {
      const value = localStorage.getItem(key);
      if (value !== null) {
        await apiRef.kvPut(NAMESPACE, key, value);
        localStorage.removeItem(key);
      }
    } catch { /* best-effort */ }
  }

  // Migrate built-in plugin localStorage keys: talu-storage:{pluginId}:{key}
  const pluginPrefix = "talu-storage:";
  const pluginKeys: string[] = [];
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k?.startsWith(pluginPrefix)) pluginKeys.push(k);
    }
  } catch { /* ignore */ }

  for (const fullKey of pluginKeys) {
    try {
      const value = localStorage.getItem(fullKey);
      if (value === null) continue;
      const rest = fullKey.slice(pluginPrefix.length);
      const sepIdx = rest.indexOf(":");
      if (sepIdx < 0) continue;
      const pluginId = rest.slice(0, sepIdx);
      const key = rest.slice(sepIdx + 1);
      await apiRef.kvPut(`plugin:${pluginId}`, key, value);
      localStorage.removeItem(fullKey);
    } catch { /* best-effort */ }
  }

  try { localStorage.setItem(MIGRATION_FLAG, "1"); } catch { /* ignore */ }
}
