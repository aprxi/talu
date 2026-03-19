import { describe, test, expect, spyOn } from "bun:test";
import { ManagedObserversImpl } from "../../../src/kernel/system/observers.ts";

interface FakeMutationObserverInstance {
  callback: MutationCallback;
  target: Node | null;
  options: MutationObserverInit | undefined;
  disconnected: boolean;
}

function installFakeMutationObserver(): {
  instances: FakeMutationObserverInstance[];
  restore(): void;
} {
  const instances: FakeMutationObserverInstance[] = [];
  const OriginalMutationObserver = globalThis.MutationObserver;

  class FakeMutationObserver {
    private instance: FakeMutationObserverInstance;

    constructor(callback: MutationCallback) {
      this.instance = {
        callback,
        target: null,
        options: undefined,
        disconnected: false,
      };
      instances.push(this.instance);
    }

    observe(target: Node, options?: MutationObserverInit): void {
      this.instance.target = target;
      this.instance.options = options;
    }

    disconnect(): void {
      this.instance.disconnected = true;
    }

    takeRecords(): MutationRecord[] {
      return [];
    }
  }

  (globalThis as { MutationObserver: typeof MutationObserver }).MutationObserver = FakeMutationObserver as any;

  return {
    instances,
    restore() {
      (globalThis as { MutationObserver: typeof MutationObserver }).MutationObserver = OriginalMutationObserver;
    },
  };
}

describe("ManagedObserversImpl", () => {
  test("resize returns disposable and tracks observer", () => {
    const observers = new ManagedObserversImpl("test.plugin");
    const el = document.createElement("div");
    const d = observers.resize(el, () => {});
    expect(typeof d.dispose).toBe("function");
    // The observer is tracked — creating 49 more should succeed, 50th should not.
    for (let i = 0; i < 49; i++) observers.resize(el, () => {});
    expect(() => observers.resize(el, () => {})).toThrow(/limit exceeded/i);
    d.dispose(); // Freeing one slot.
    expect(() => observers.resize(el, () => {})).not.toThrow(); // Can add again.
    observers.dispose();
  });

  test("intersection returns disposable and tracks observer", () => {
    const observers = new ManagedObserversImpl("test.plugin");
    const el = document.createElement("div");
    const d = observers.intersection(el, () => {});
    expect(typeof d.dispose).toBe("function");
    d.dispose();
    observers.dispose();
  });

  test("mutation returns disposable and observes target", () => {
    const observers = new ManagedObserversImpl("test.plugin");
    const el = document.createElement("div");
    document.body.appendChild(el);
    const d = observers.mutation(el, () => {}, { childList: true });
    expect(typeof d.dispose).toBe("function");
    d.dispose();
    observers.dispose();
  });

  test("callback error is caught and logged", () => {
    const fake = installFakeMutationObserver();
    const spy = spyOn(console, "error").mockImplementation(() => {});
    const observers = new ManagedObserversImpl("test.plugin");
    const el = document.createElement("div");
    document.body.appendChild(el);
    // MutationObserver with a throwing callback.
    observers.mutation(el, () => { throw new Error("observer boom"); }, { childList: true });
    fake.instances[0]!.callback([] as MutationRecord[], {} as MutationObserver);
    expect(spy).toHaveBeenCalled();
    expect(spy.mock.calls.some((c) => String(c[0]).includes("test.plugin"))).toBe(true);
    spy.mockRestore();
    observers.dispose();
    fake.restore();
  });

  test("dispose clears all active observers", () => {
    const observers = new ManagedObserversImpl("test.plugin");
    const el = document.createElement("div");
    observers.resize(el, () => {});
    observers.intersection(el, () => {});
    observers.dispose();
    // After dispose, cap is reset — can add 50 more.
    for (let i = 0; i < 50; i++) observers.resize(el, () => {});
    // Should hit cap again at 50.
    expect(() => observers.resize(el, () => {})).toThrow(/limit exceeded/i);
    observers.dispose();
  });

  test("resource cap throws at limit", () => {
    const observers = new ManagedObserversImpl("test.plugin");
    const el = document.createElement("div");
    for (let i = 0; i < 50; i++) {
      observers.resize(el, () => {});
    }
    expect(() => observers.resize(el, () => {})).toThrow(/limit exceeded/i);
    observers.dispose();
  });

  test("disposed observers free cap space", () => {
    const observers = new ManagedObserversImpl("test.plugin");
    const el = document.createElement("div");
    const disposables = [];
    for (let i = 0; i < 50; i++) {
      disposables.push(observers.resize(el, () => {}));
    }
    expect(() => observers.resize(el, () => {})).toThrow(/limit exceeded/i);
    disposables[0].dispose();
    expect(() => observers.resize(el, () => {})).not.toThrow();
    observers.dispose();
  });
});
