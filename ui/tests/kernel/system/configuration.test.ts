import { describe, test, expect, spyOn } from "bun:test";
import { ConfigurationAccessImpl } from "../../../src/kernel/system/configuration.ts";

/**
 * Configuration debounce tests require real time for the 100ms debounce timer
 * to fire. Bun 1.2.8 lacks advanceTimersByTime() so we use deadlock-guard
 * waits with generous margins (5x the debounce period).
 *
 * Synchronous contracts are tested without timing dependency.
 */

// ── Synchronous contracts ───────────────────────────────────────────────────

describe("ConfigurationAccessImpl — synchronous", () => {
  test("get returns empty object by default", () => {
    const config = new ConfigurationAccessImpl();
    expect(config.get()).toEqual({});
  });

  test("setConfig updates the config", () => {
    const config = new ConfigurationAccessImpl();
    config.setConfig({ theme: "dark" });
    expect(config.get<{ theme: string }>().theme).toBe("dark");
  });

  test("setConfig triggers onChange callbacks synchronously", () => {
    const config = new ConfigurationAccessImpl();
    let received: unknown;
    config.onChange((c) => { received = c; });
    config.setConfig({ value: 42 });
    expect(received).toEqual({ value: 42 });
    config.dispose();
  });

  test("update does NOT trigger onChange synchronously (debounced)", () => {
    const config = new ConfigurationAccessImpl();
    let callCount = 0;
    config.onChange(() => { callCount++; });
    config.update({ v: 1 });
    // Must not have fired yet — debounce defers the notification.
    expect(callCount).toBe(0);
    config.dispose();
  });

  test("onChange dispose stops notifications", () => {
    const config = new ConfigurationAccessImpl();
    let callCount = 0;
    const d = config.onChange(() => { callCount++; });
    config.setConfig({ a: 1 });
    expect(callCount).toBe(1);
    d.dispose();
    config.setConfig({ a: 2 });
    expect(callCount).toBe(1);
    config.dispose();
  });

  test("onChange callback error does not affect other listeners", () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    const config = new ConfigurationAccessImpl();
    let secondCalled = false;
    config.onChange(() => { throw new Error("listener boom"); });
    config.onChange(() => { secondCalled = true; });
    config.setConfig({});
    expect(secondCalled).toBe(true);
    spy.mockRestore();
    config.dispose();
  });
});

// ── Debounce behavior (deadlock-guard waits) ────────────────────────────────

describe("ConfigurationAccessImpl — debounce", () => {
  test("update fires onChange after debounce period", async () => {
    const config = new ConfigurationAccessImpl();
    let received: unknown = null;
    config.onChange((c) => { received = c; });
    config.update({ debounced: true });
    expect(received).toBeNull(); // Synchronous: not yet fired.
    // Deadlock guard: 300ms for a 100ms debounce (3x margin).
    await new Promise((r) => setTimeout(r, 300));
    expect(received).toEqual({ debounced: true });
    config.dispose();
  });

  test("rapid updates coalesce — only last value fires", async () => {
    const config = new ConfigurationAccessImpl();
    const values: unknown[] = [];
    config.onChange((c) => { values.push(c); });
    config.update({ v: 1 });
    config.update({ v: 2 });
    config.update({ v: 3 });
    // Deadlock guard: 300ms for a 100ms debounce (3x margin).
    await new Promise((r) => setTimeout(r, 300));
    expect(values.length).toBe(1);
    expect(values[0]).toEqual({ v: 3 });
    config.dispose();
  });

  test("dispose cancels pending debounce", async () => {
    const config = new ConfigurationAccessImpl();
    let callCount = 0;
    config.onChange(() => { callCount++; });
    config.update({ a: 1 });
    config.dispose();
    // Deadlock guard: 500ms after dispose — must NOT fire.
    await new Promise((r) => setTimeout(r, 300));
    expect(callCount).toBe(0);
  });
});
