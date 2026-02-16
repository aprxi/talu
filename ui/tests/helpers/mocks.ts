/**
 * Shared mock factories for plugin integration tests.
 *
 * Provides sensible defaults — tests override specific methods as needed.
 */

import type { ManagedTimers, Notifications, Disposable } from "../../src/kernel/types.ts";

/** Timers that execute callbacks synchronously (no real delay). */
export function mockTimers(): ManagedTimers {
  return {
    setTimeout(fn: () => void) { fn(); return { dispose() {} } as Disposable; },
    setInterval() { return { dispose() {} } as Disposable; },
    clearTimeout() {},
    clearInterval() {},
    requestAnimationFrame(fn: () => void) { fn(); return 0 as any; },
    cancelAnimationFrame() {},
    cap: 100,
  } as ManagedTimers;
}

/**
 * Controllable timer mock for debounce/timeout verification.
 *
 * Captures scheduled callbacks in `.pending` instead of firing them.
 * Use `.flush()` to fire all pending, or `.advance(ms)` to fire only
 * those with delay ≤ ms.
 */
export function mockControllableTimers() {
  const pending: { fn: () => void; ms: number; disposed: boolean }[] = [];
  return {
    timers: {
      setTimeout(fn: () => void, ms: number): Disposable {
        const entry = { fn, ms, disposed: false };
        pending.push(entry);
        return { dispose() { entry.disposed = true; } };
      },
      setInterval() { return { dispose() {} } as Disposable; },
      clearTimeout() {},
      clearInterval() {},
      requestAnimationFrame(fn: () => void) { fn(); return 0 as any; },
      cancelAnimationFrame() {},
      cap: 100,
    } as ManagedTimers,
    pending,
    /** Fire all pending non-disposed callbacks and clear the queue. */
    flush() {
      for (const e of pending) if (!e.disposed) e.fn();
      pending.length = 0;
    },
    /** Fire only callbacks with delay ≤ ms, then mark them disposed. */
    advance(ms: number) {
      for (const e of [...pending]) {
        if (!e.disposed && e.ms <= ms) {
          e.fn();
          e.disposed = true;
        }
      }
    },
  };
}

/** Notifications that record messages for assertion. */
export function mockNotifications(): {
  mock: Notifications & { success(msg: string): void };
  messages: { type: string; msg: string }[];
} {
  const messages: { type: string; msg: string }[] = [];
  return {
    messages,
    mock: {
      info: (msg: string) => messages.push({ type: "info", msg }),
      error: (msg: string) => messages.push({ type: "error", msg }),
      warn: (msg: string) => messages.push({ type: "warn", msg }),
      success: (msg: string) => messages.push({ type: "success", msg }),
    } as Notifications & { success(msg: string): void },
  };
}
