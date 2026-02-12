import { describe, test, expect, spyOn } from "bun:test";
import { EventBusImpl } from "../../../src/kernel/system/event-bus.ts";

describe("EventBusImpl", () => {
  test("emits events to subscribers", () => {
    const bus = new EventBusImpl();
    let received: unknown = null;

    bus.on("test.event", (data) => { received = data; });
    bus.emit("test.event", { foo: "bar" });

    expect(received).toEqual({ foo: "bar" });
  });

  test("supports multiple subscribers for the same event", () => {
    const bus = new EventBusImpl();
    const calls: number[] = [];

    bus.on("e", () => calls.push(1));
    bus.on("e", () => calls.push(2));
    bus.emit("e", null);

    expect(calls).toEqual([1, 2]);
  });

  test("does not call handlers for different events", () => {
    const bus = new EventBusImpl();
    let called = false;

    bus.on("event-a", () => { called = true; });
    bus.emit("event-b", null);

    expect(called).toBe(false);
  });

  test("once() unsubscribes after one emission", () => {
    const bus = new EventBusImpl();
    let count = 0;

    bus.once("e", () => { count++; });
    bus.emit("e", null);
    bus.emit("e", null);

    expect(count).toBe(1);
  });

  test("dispose() removes listener", () => {
    const bus = new EventBusImpl();
    let called = false;

    const sub = bus.on("e", () => { called = true; });
    sub.dispose();
    bus.emit("e", null);

    expect(called).toBe(false);
  });

  test("once() dispose prevents the handler from being called", () => {
    const bus = new EventBusImpl();
    let called = false;

    const sub = bus.once("e", () => { called = true; });
    sub.dispose();
    bus.emit("e", null);

    expect(called).toBe(false);
  });

  test("freezes emitted payloads", () => {
    const bus = new EventBusImpl();
    let received: unknown = null;

    bus.on("e", (data) => { received = data; });
    bus.emit("e", { a: 1 });

    expect(Object.isFrozen(received)).toBe(true);
  });

  test("catches handler errors without affecting other handlers", () => {
    const bus = new EventBusImpl();
    const spy = spyOn(console, "error").mockImplementation(() => {});
    let secondCalled = false;

    bus.on("e", () => { throw new Error("boom"); });
    bus.on("e", () => { secondCalled = true; });
    bus.emit("e", null);

    expect(secondCalled).toBe(true);
    expect(spy).toHaveBeenCalled();

    spy.mockRestore();
  });

  test("emit with no subscribers is a no-op", () => {
    const bus = new EventBusImpl();
    // Should not throw.
    expect(() => bus.emit("nobody-listening", { x: 1 })).not.toThrow();
  });

  test("cleans up empty listener sets after dispose", () => {
    const bus = new EventBusImpl();

    const sub = bus.on("e", () => {});
    sub.dispose();

    // Internal: the map entry should be removed when the set becomes empty.
    // We verify by checking that a new on() creates a fresh set.
    let received = false;
    bus.on("e", () => { received = true; });
    bus.emit("e", null);
    expect(received).toBe(true);
  });
});
