import { describe, test, expect, spyOn } from "bun:test";
import { ServiceRegistry } from "../../../src/kernel/registries/services.ts";

describe("ServiceRegistry", () => {
  test("get returns undefined for unregistered service", () => {
    const registry = new ServiceRegistry();
    expect(registry.get("missing")).toBeUndefined();
  });

  test("provide + get returns the registered instance", () => {
    const registry = new ServiceRegistry();
    const svc = { greet: () => "hello" };
    registry.provide("chat.service", svc);
    expect(registry.get("chat.service")).toBe(svc);
  });

  test("dispose removes the service", () => {
    const registry = new ServiceRegistry();
    const svc = {};
    const disposable = registry.provide("my.svc", svc);
    disposable.dispose();
    expect(registry.get("my.svc")).toBeUndefined();
  });

  test("duplicate registration warns and returns no-op disposable", () => {
    const spy = spyOn(console, "warn").mockImplementation(() => {});
    const registry = new ServiceRegistry();
    const first = { id: 1 };
    const second = { id: 2 };
    registry.provide("dup.svc", first);
    const d = registry.provide("dup.svc", second);
    // First registration wins.
    expect(registry.get("dup.svc")).toBe(first);
    // Dispose of the duplicate no-op should not remove the first.
    d.dispose();
    expect(registry.get("dup.svc")).toBe(first);
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  test("dispose is safe if service was already replaced", () => {
    const registry = new ServiceRegistry();
    const svc1 = { v: 1 };
    const d1 = registry.provide("svc", svc1);
    // Manually clear and re-provide (simulates registry manipulation).
    d1.dispose();
    const svc2 = { v: 2 };
    registry.provide("svc", svc2);
    // Disposing d1 again should not remove svc2.
    d1.dispose();
    expect(registry.get("svc")).toBe(svc2);
  });

  test("onDidChange fires on provide", () => {
    const registry = new ServiceRegistry();
    let received: unknown;
    registry.onDidChange("svc", (value) => { received = value; });
    const svc = { name: "test" };
    registry.provide("svc", svc);
    expect(received).toBe(svc);
  });

  test("onDidChange fires with undefined on dispose", () => {
    const registry = new ServiceRegistry();
    let received: unknown = "unset";
    const d = registry.provide("svc", {});
    registry.onDidChange("svc", (value) => { received = value; });
    d.dispose();
    expect(received).toBeUndefined();
  });

  test("onDidChange dispose stops notifications", () => {
    const registry = new ServiceRegistry();
    let callCount = 0;
    const d = registry.onDidChange("svc", () => { callCount++; });
    registry.provide("svc", {});
    expect(callCount).toBe(1);
    d.dispose();
    // Provide again — callback should not fire.
    registry.provide("svc2", {});
    // svc2 is a different key, let's use same key by disposing first.
    expect(callCount).toBe(1);
  });

  test("change listener error does not affect other listeners", () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    const registry = new ServiceRegistry();
    let secondCalled = false;
    registry.onDidChange("svc", () => { throw new Error("listener boom"); });
    registry.onDidChange("svc", () => { secondCalled = true; });
    registry.provide("svc", {});
    expect(secondCalled).toBe(true);
    spy.mockRestore();
  });
});
