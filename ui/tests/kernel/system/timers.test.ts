import { describe, test, expect, spyOn, beforeEach, afterEach } from "bun:test";
import { ManagedTimersImpl } from "../../../src/kernel/system/timers.ts";

/**
 * Timer tests use local stubs for window timers so callback firing/cancellation
 * is driven directly by the test instead of wall-clock waits.
 */

let timers: ManagedTimersImpl;
let restoreWindowTimers: (() => void) | null = null;

interface FakeTimerTask {
  id: number;
  callback: () => void;
  cleared: boolean;
}

function installFakeWindowTimers(): {
  timeouts: FakeTimerTask[];
  intervals: FakeTimerTask[];
  fireTimeout(task: FakeTimerTask): void;
  fireInterval(task: FakeTimerTask, count?: number): void;
} {
  const timeouts: FakeTimerTask[] = [];
  const intervals: FakeTimerTask[] = [];
  let nextId = 1;

  const timeoutSpy = spyOn(window, "setTimeout").mockImplementation((callback: TimerHandler) => {
    const task = { id: nextId++, callback: callback as () => void, cleared: false };
    timeouts.push(task);
    return task.id as any;
  });
  const clearTimeoutSpy = spyOn(window, "clearTimeout").mockImplementation((id: number) => {
    const task = timeouts.find((entry) => entry.id === id);
    if (task) task.cleared = true;
  });
  const intervalSpy = spyOn(window, "setInterval").mockImplementation((callback: TimerHandler) => {
    const task = { id: nextId++, callback: callback as () => void, cleared: false };
    intervals.push(task);
    return task.id as any;
  });
  const clearIntervalSpy = spyOn(window, "clearInterval").mockImplementation((id: number) => {
    const task = intervals.find((entry) => entry.id === id);
    if (task) task.cleared = true;
  });

  restoreWindowTimers = () => {
    clearIntervalSpy.mockRestore();
    intervalSpy.mockRestore();
    clearTimeoutSpy.mockRestore();
    timeoutSpy.mockRestore();
    restoreWindowTimers = null;
  };

  return {
    timeouts,
    intervals,
    fireTimeout(task) {
      if (task.cleared) return;
      task.cleared = true;
      task.callback();
    },
    fireInterval(task, count = 1) {
      for (let i = 0; i < count; i++) {
        if (task.cleared) return;
        task.callback();
      }
    },
  };
}

beforeEach(() => {
  timers = new ManagedTimersImpl("test.plugin");
});

afterEach(() => {
  timers.dispose();
  restoreWindowTimers?.();
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

// ── Async behavior tests (deterministic timer control) ──────────────────────

describe("ManagedTimersImpl — async behavior", () => {
  test("setTimeout fires callback", () => {
    const fake = installFakeWindowTimers();
    let called = false;
    timers.setTimeout(() => { called = true; }, 1);
    fake.fireTimeout(fake.timeouts[0]!);
    expect(called).toBe(true);
  });

  test("setTimeout dispose cancels the timer", () => {
    const fake = installFakeWindowTimers();
    let called = false;
    const d = timers.setTimeout(() => { called = true; }, 1);
    d.dispose();
    fake.fireTimeout(fake.timeouts[0]!);
    expect(called).toBe(false);
  });

  test("setInterval fires callback more than once", () => {
    const fake = installFakeWindowTimers();
    let count = 0;
    timers.setInterval(() => { count++; }, 1);
    fake.fireInterval(fake.intervals[0]!, 2);
    expect(count).toBeGreaterThanOrEqual(2);
  });

  test("setInterval dispose stops the interval", () => {
    const fake = installFakeWindowTimers();
    let count = 0;
    const d = timers.setInterval(() => { count++; }, 1);
    fake.fireInterval(fake.intervals[0]!, 2);
    d.dispose();
    const countAfterDispose = count;
    expect(countAfterDispose).toBeGreaterThanOrEqual(1);
    fake.fireInterval(fake.intervals[0]!, 2);
    expect(count).toBe(countAfterDispose);
  });

  test("dispose prevents pending setTimeout from firing", () => {
    const fake = installFakeWindowTimers();
    let called1 = false;
    let called2 = false;
    timers.setTimeout(() => { called1 = true; }, 1);
    timers.setInterval(() => { called2 = true; }, 1);
    timers.dispose();
    fake.fireTimeout(fake.timeouts[0]!);
    fake.fireInterval(fake.intervals[0]!, 2);
    expect(called1).toBe(false);
    expect(called2).toBe(false);
  });

  test("callback error is caught and logged", () => {
    const fake = installFakeWindowTimers();
    const spy = spyOn(console, "error").mockImplementation(() => {});
    timers.setTimeout(() => { throw new Error("boom"); }, 1);
    fake.fireTimeout(fake.timeouts[0]!);
    expect(spy).toHaveBeenCalled();
    expect(spy.mock.calls[0]![0]).toContain("test.plugin");
    spy.mockRestore();
  });
});
