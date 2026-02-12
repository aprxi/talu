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
