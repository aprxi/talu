import { describe, test, expect, spyOn } from "bun:test";
import { DisposableStore } from "../../../src/kernel/core/disposable.ts";

describe("DisposableStore", () => {
  test("tracks and disposes resources", () => {
    const store = new DisposableStore();
    let disposed = false;

    store.track({ dispose: () => { disposed = true; } });
    expect(disposed).toBe(false);

    store.dispose();
    expect(disposed).toBe(true);
  });

  test("track returns the same disposable", () => {
    const store = new DisposableStore();
    const resource = { dispose() {} };
    const returned = store.track(resource);
    expect(returned).toBe(resource);
  });

  test("disposes all tracked resources", () => {
    const store = new DisposableStore();
    const calls: number[] = [];

    store.track({ dispose: () => calls.push(1) });
    store.track({ dispose: () => calls.push(2) });
    store.track({ dispose: () => calls.push(3) });

    store.dispose();
    expect(calls).toEqual([1, 2, 3]);
  });

  test("dispose is idempotent", () => {
    const store = new DisposableStore();
    let count = 0;

    store.track({ dispose: () => { count++; } });

    store.dispose();
    store.dispose();
    expect(count).toBe(1);
  });

  test("isDisposed reflects state", () => {
    const store = new DisposableStore();
    expect(store.isDisposed).toBe(false);

    store.dispose();
    expect(store.isDisposed).toBe(true);
  });

  test("handles errors during disposal gracefully", () => {
    const store = new DisposableStore();
    const spy = spyOn(console, "error").mockImplementation(() => {});

    let secondDisposed = false;
    store.track({ dispose: () => { throw new Error("boom"); } });
    store.track({ dispose: () => { secondDisposed = true; } });

    // Should not throw, and should still dispose the second resource.
    expect(() => store.dispose()).not.toThrow();
    expect(secondDisposed).toBe(true);
    expect(spy).toHaveBeenCalled();

    spy.mockRestore();
  });

  test("immediately disposes items tracked after store is disposed", () => {
    const store = new DisposableStore();
    store.dispose();

    let disposed = false;
    store.track({ dispose: () => { disposed = true; } });
    expect(disposed).toBe(true);
  });

  test("swallows errors from late-tracked items", () => {
    const store = new DisposableStore();
    store.dispose();

    // Tracking an item that throws on dispose should not propagate.
    expect(() => {
      store.track({ dispose: () => { throw new Error("late"); } });
    }).not.toThrow();
  });
});
