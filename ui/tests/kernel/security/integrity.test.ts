import { describe, test, expect, beforeEach, afterEach, spyOn } from "bun:test";
import { verifyIntegrity } from "../../../src/kernel/security/integrity.ts";
import { getSetting, initKvSettings } from "../../../src/kernel/system/kv-settings.ts";

const HASH_CACHE_KEY = "talu:integrityHashes";

/** Compute SHA-256 base64 hash for a string. */
async function computeHash(content: string): Promise<string> {
  const buf = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(content),
  );
  return btoa(String.fromCharCode(...new Uint8Array(buf)));
}

describe("verifyIntegrity", () => {
  let originalFetch: typeof window.fetch;

  beforeEach(() => {
    originalFetch = window.fetch;
    localStorage.removeItem(HASH_CACHE_KEY);
    // Reset KV settings to localStorage fallback (another test may have initialized an API).
    (initKvSettings as (api: any) => void)(null);
  });

  afterEach(() => {
    window.fetch = originalFetch;
  });

  test("returns true for matching hash", async () => {
    const content = "console.log('hello');";
    const hash = await computeHash(content);

    window.fetch = async () => new Response(content);

    const result = await verifyIntegrity("/plugin.js", `sha256-${hash}`);
    expect(result).toBe(true);
  });

  test("returns false for mismatching hash", async () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    window.fetch = async () => new Response("actual content");

    const result = await verifyIntegrity("/plugin.js", "sha256-WRONGHASH");
    expect(result).toBe(false);
    spy.mockRestore();
  });

  test("returns false for unsupported hash format", async () => {
    const spy = spyOn(console, "warn").mockImplementation(() => {});
    const result = await verifyIntegrity("/plugin.js", "md5-abc123");
    expect(result).toBe(false);
    spy.mockRestore();
  });

  test("returns false when fetch fails", async () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    window.fetch = async () => { throw new Error("network down"); };

    const result = await verifyIntegrity("/plugin.js", "sha256-abc");
    expect(result).toBe(false);
    spy.mockRestore();
  });

  test("returns false for non-ok response", async () => {
    window.fetch = async () => new Response("not found", { status: 404 });

    const result = await verifyIntegrity("/plugin.js", "sha256-abc");
    expect(result).toBe(false);
  });

  test("stores last-known-good hash on success", async () => {
    const content = "export default {}";
    const hash = await computeHash(content);

    window.fetch = async () => new Response(content);

    const result = await verifyIntegrity("/plugin.js", `sha256-${hash}`);
    expect(result).toBe(true);

    // Read through the KV settings layer (same backend integrity.ts uses).
    const raw = await getSetting(HASH_CACHE_KEY);
    const stored = JSON.parse(raw ?? "{}");
    expect(stored["/plugin.js"]).toBe(hash);
  });

  test("does not store hash on mismatch", async () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    window.fetch = async () => new Response("different content");

    await verifyIntegrity("/plugin.js", "sha256-WRONGHASH");

    const stored = await getSetting(HASH_CACHE_KEY);
    expect(stored).toBeNull();
    spy.mockRestore();
  });

  test("detects when last-known-good also differs", async () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    const warnSpy = spyOn(console, "warn").mockImplementation(() => {});

    // Seed a different last-known-good hash.
    localStorage.setItem(HASH_CACHE_KEY, JSON.stringify({ "/plugin.js": "oldHash" }));

    window.fetch = async () => new Response("tampered content");

    await verifyIntegrity("/plugin.js", "sha256-declaredHash");

    // Should warn about last-known-good also differing.
    expect(warnSpy).toHaveBeenCalled();
    spy.mockRestore();
    warnSpy.mockRestore();
  });
});
