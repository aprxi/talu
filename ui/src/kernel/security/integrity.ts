/**
 * Integrity Verification — compare server-reported content hash
 * against the fetched module source.
 *
 * Uses SHA-256 via SubtleCrypto. If the manifest declares an `integrity`
 * field (format: "sha256-<base64>"), we fetch the module source, hash it,
 * and compare. On mismatch, the plugin is refused.
 *
 * Last-known-good hashes are stored in the KV backend (namespace "ui").
 */

import { getSetting, setSetting } from "../system/kv-settings.ts";

const HASH_CACHE_KEY = "talu:integrityHashes";

/** Verify a module URL against a declared integrity hash. Returns true if valid. */
export async function verifyIntegrity(url: string, declaredIntegrity: string): Promise<boolean> {
  const match = declaredIntegrity.match(/^sha256-(.+)$/);
  if (!match || !match[1]) {
    console.warn(`[kernel] Unsupported integrity format: "${declaredIntegrity}"`);
    return false;
  }
  const expected = match[1];

  try {
    const resp = await fetch(url);
    if (!resp.ok) return false;
    const buf = await resp.arrayBuffer();
    const hashBuf = await crypto.subtle.digest("SHA-256", buf);
    const actual = btoa(String.fromCharCode(...new Uint8Array(hashBuf)));

    if (actual !== expected) {
      console.error(
        `[kernel] Integrity mismatch for "${url}": expected ${expected}, got ${actual}`,
      );

      // Check against last-known-good hash.
      const lastGood = await getLastKnownGoodHash(url);
      if (lastGood && lastGood !== expected) {
        console.warn(`[kernel] Last-known-good hash also differs — possible update or tampering.`);
      }
      return false;
    }

    // Store as last-known-good.
    await setLastKnownGoodHash(url, actual);
    return true;
  } catch (err) {
    console.error(`[kernel] Integrity check failed for "${url}":`, err);
    return false;
  }
}

async function getLastKnownGoodHash(url: string): Promise<string | null> {
  try {
    const stored = await getSetting(HASH_CACHE_KEY);
    if (!stored) return null;
    const map = JSON.parse(stored) as Record<string, string>;
    return map[url] ?? null;
  } catch {
    return null;
  }
}

async function setLastKnownGoodHash(url: string, hash: string): Promise<void> {
  try {
    const stored = await getSetting(HASH_CACHE_KEY);
    const map = stored ? (JSON.parse(stored) as Record<string, string>) : {};
    map[url] = hash;
    await setSetting(HASH_CACHE_KEY, JSON.stringify(map));
  } catch { /* KV unavailable */ }
}
