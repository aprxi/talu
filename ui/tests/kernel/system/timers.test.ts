import { describe, test, expect, spyOn, beforeEach, afterEach } from "bun:test";
import { ManagedTimersImpl } from "../../../src/kernel/system/timers.ts";

/**
 * Timer tests inherently require real time to pass for setTimeout/setInterval
 * callbacks to fire. Bun 1.2.8 has useFakeTimers() but lacks
 * advanceTimersByTime(), so deterministic timer advancement is not possible.
 *
 * Strategy: synchronous tests for cap/dispose/error-boundary logic (no timing).
 * Async tests use generous deadlock-guard waits (≥10x the timer interval) and
 * verify the callback contract (fired / not fired) rather than exact timing.
 */

let timers: ManagedTimersImpl;

beforeEach(() => {
  timers = new ManagedTimersImpl("test.plugin");
});

afterEach(() => {
  timers.dispose();
});

// ── Synchronous contract tests (no timing dependency) ───────────────────────

describe("ManagedTimersImpl — synchronous contracts", () => {
  test("setTimeout returns a disposable", () => {
    const d = timers.setTimeout(() => {}, 100_000);
    expect(typeof d.dispose).toBe("function");
  });

  test("setInterval returns a disposable", () => {
    const d = timers.setInterval(() => {}, 100_000);
    expect(typeof d.dispose).toBe("function");
  });

  test("resource cap throws at limit (100)", () => {
    for (let i = 0; i < 100; i++) {
      timers.setInterval(() => {}, 100_000);
    }
    expect(() => timers.setTimeout(() => {}, 1)).toThrow(/limit exceeded/i);
  });

  test("disposed timers free cap space", () => {
    const disposables = [];
    for (let i = 0; i < 100; i++) {
      disposables.push(timers.setInterval(() => {}, 100_000));
    }
    expect(() => timers.setTimeout(() => {}, 1)).toThrow(/limit exceeded/i);
    disposables[0]!.dispose();
    expect(() => timers.setTimeout(() => {}, 100_000)).not.toThrow();
  });

  test("dispose clears all timers from cap tracking", () => {
    for (let i = 0; i < 100; i++) {
      timers.setInterval(() => {}, 100_000);
    }
    timers.dispose();
    // After dispose, can allocate again.
    expect(() => timers.setTimeout(() => {}, 100_000)).not.toThrow();
  });
});

// ── Async behavior tests (deadlock-guard waits) ─────────────────────────────

describe("ManagedTimersImpl — async behavior", () => {
  test("setTimeout fires callback", async () => {
    let called = false;
    timers.setTimeout(() => { called = true; }, 1);
    // Deadlock guard: 200ms for a 1ms timer (200x margin).
    await new Promise((r) => setTimeout(r, 200));
    expect(called).toBe(true);
  });

  test("setTimeout dispose cancels the timer", async () => {
    let called = false;
    const d = timers.setTimeout(() => { called = true; }, 1);
    d.dispose();
    await new Promise((r) => setTimeout(r, 200));
    expect(called).toBe(false);
  });

  test("setInterval fires callback more than once", async () => {
    let count = 0;
    timers.setInterval(() => { count++; }, 1);
    // Deadlock guard: 200ms for 1ms intervals → should fire many times.
    await new Promise((r) => setTimeout(r, 200));
    expect(count).toBeGreaterThanOrEqual(2);
  });

  test("setInterval dispose stops the interval", async () => {
    let count = 0;
    const d = timers.setInterval(() => { count++; }, 1);
    await new Promise((r) => setTimeout(r, 200));
    d.dispose();
    const countAfterDispose = count;
    expect(countAfterDispose).toBeGreaterThanOrEqual(1);
    // Wait again — count must not increase.
    await new Promise((r) => setTimeout(r, 200));
    expect(count).toBe(countAfterDispose);
  });

  test("dispose prevents pending setTimeout from firing", async () => {
    let called1 = false;
    let called2 = false;
    timers.setTimeout(() => { called1 = true; }, 1);
    timers.setInterval(() => { called2 = true; }, 1);
    timers.dispose();
    await new Promise((r) => setTimeout(r, 200));
    expect(called1).toBe(false);
    expect(called2).toBe(false);
  });

  test("callback error is caught and logged", async () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    timers.setTimeout(() => { throw new Error("boom"); }, 1);
    await new Promise((r) => setTimeout(r, 200));
    expect(spy).toHaveBeenCalled();
    expect(spy.mock.calls[0]![0]).toContain("test.plugin");
    spy.mockRestore();
  });
});
