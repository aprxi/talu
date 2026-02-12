import { describe, test, expect } from "bun:test";
import { ImportTimeoutError, importWithWatchdog } from "../../../src/kernel/core/import-watchdog.ts";

describe("ImportTimeoutError", () => {
  test("has correct name", () => {
    const err = new ImportTimeoutError("http://example.com/plugin.js");
    expect(err.name).toBe("ImportTimeoutError");
  });

  test("preserves the URL", () => {
    const err = new ImportTimeoutError("http://example.com/plugin.js");
    expect(err.url).toBe("http://example.com/plugin.js");
  });

  test("message includes the URL", () => {
    const err = new ImportTimeoutError("http://example.com/plugin.js");
    expect(err.message).toContain("http://example.com/plugin.js");
  });

  test("is instanceof Error", () => {
    const err = new ImportTimeoutError("test");
    expect(err instanceof Error).toBe(true);
  });
});

describe("importWithWatchdog", () => {
  test("rejects with ImportTimeoutError on timeout", async () => {
    try {
      // Import a non-existent module with a very short timeout.
      await importWithWatchdog("data:text/javascript,await new Promise(()=>{})", 50);
      // Should not reach here.
      expect(true).toBe(false);
    } catch (err) {
      expect(err instanceof ImportTimeoutError).toBe(true);
    }
  });

  test("resolves for fast-loading modules", async () => {
    const mod = await importWithWatchdog<{ default?: unknown }>(
      "data:text/javascript,export default 42",
      5000,
    );
    expect(mod.default).toBe(42);
  });
});
