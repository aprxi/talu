import { describe, test, expect, spyOn } from "bun:test";
import { ConfigurationAccessImpl } from "../../../src/kernel/system/configuration.ts";

/**
 * Configuration debounce tests stub global timers so the queued notification is
 * triggered directly by the test without real elapsed time.
 */

interface FakeTimeoutTask {
  id: number;
  callback: () => void;
  cleared: boolean;
}

function installFakeTimeouts(): {
  tasks: FakeTimeoutTask[];
  fire(task: FakeTimeoutTask): void;
  restore(): void;
} {
  const tasks: FakeTimeoutTask[] = [];
  let nextId = 1;

  const timeoutSpy = spyOn(globalThis, "setTimeout").mockImplementation((callback: TimerHandler) => {
    const task = { id: nextId++, callback: callback as () => void, cleared: false };
    tasks.push(task);
    return task.id as any;
  });
  const clearTimeoutSpy = spyOn(globalThis, "clearTimeout").mockImplementation((id: number) => {
    const task = tasks.find((entry) => entry.id === id);
    if (task) task.cleared = true;
  });

  return {
    tasks,
    fire(task) {
      if (task.cleared) return;
      task.cleared = true;
      task.callback();
    },
    restore() {
      clearTimeoutSpy.mockRestore();
      timeoutSpy.mockRestore();
    },
  };
}

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

// ── Debounce behavior (deterministic timer control) ─────────────────────────

describe("ConfigurationAccessImpl — debounce", () => {
  test("update fires onChange after debounce period", () => {
    const fake = installFakeTimeouts();
    const config = new ConfigurationAccessImpl();
    let received: unknown = null;
    config.onChange((c) => { received = c; });
    config.update({ debounced: true });
    expect(received).toBeNull(); // Synchronous: not yet fired.
    fake.fire(fake.tasks[0]!);
    expect(received).toEqual({ debounced: true });
    config.dispose();
    fake.restore();
  });

  test("rapid updates coalesce — only last value fires", () => {
    const fake = installFakeTimeouts();
    const config = new ConfigurationAccessImpl();
    const values: unknown[] = [];
    config.onChange((c) => { values.push(c); });
    config.update({ v: 1 });
    config.update({ v: 2 });
    config.update({ v: 3 });
    expect(fake.tasks[0]!.cleared).toBe(true);
    expect(fake.tasks[1]!.cleared).toBe(true);
    fake.fire(fake.tasks[2]!);
    expect(values.length).toBe(1);
    expect(values[0]).toEqual({ v: 3 });
    config.dispose();
    fake.restore();
  });

  test("dispose cancels pending debounce", () => {
    const fake = installFakeTimeouts();
    const config = new ConfigurationAccessImpl();
    let callCount = 0;
    config.onChange(() => { callCount++; });
    config.update({ a: 1 });
    config.dispose();
    fake.fire(fake.tasks[0]!);
    expect(callCount).toBe(0);
    fake.restore();
  });
});
