import { describe, test, expect, spyOn, beforeEach } from "bun:test";
import { ContextKeyService } from "../../../src/kernel/registries/context-keys.ts";

let svc: ContextKeyService;

beforeEach(() => {
  svc = new ContextKeyService();
});

// ── Store CRUD ────────────────────────────────────────────────────────────────

describe("ContextKeyService — store", () => {
  test("set + get round-trips a string value", () => {
    svc.set("mode", "dark");
    expect(svc.get("mode")).toBe("dark");
  });

  test("set + get round-trips a boolean value", () => {
    svc.set("isReady", true);
    expect(svc.get("isReady")).toBe(true);
  });

  test("get returns undefined for unset key", () => {
    expect(svc.get("missing")).toBeUndefined();
  });

  test("has returns true for set key, false for unset", () => {
    svc.set("k", "v");
    expect(svc.has("k")).toBe(true);
    expect(svc.has("other")).toBe(false);
  });

  test("delete removes a key", () => {
    svc.set("k", "v");
    svc.delete("k");
    expect(svc.get("k")).toBeUndefined();
    expect(svc.has("k")).toBe(false);
  });

  test("delete on non-existent key does not throw", () => {
    svc.delete("missing"); // no-op
  });

  test("set returns Disposable that deletes the key", () => {
    const d = svc.set("k", "v");
    expect(svc.get("k")).toBe("v");
    d.dispose();
    expect(svc.get("k")).toBeUndefined();
  });

  test("set Disposable does not delete if key was overwritten", () => {
    const d = svc.set("k", "old");
    svc.set("k", "new");
    d.dispose();
    // Key was overwritten — dispose should not delete.
    expect(svc.get("k")).toBe("new");
  });
});

// ── onChange ───────────────────────────────────────────────────────────────────

describe("ContextKeyService — onChange", () => {
  test("callback fires on set", () => {
    let received: unknown;
    svc.onChange("k", (_key, value) => { received = value; });
    svc.set("k", "hello");
    expect(received).toBe("hello");
  });

  test("callback fires on delete", () => {
    svc.set("k", "v");
    let received: unknown = "not-called";
    svc.onChange("k", (_key, value) => { received = value; });
    svc.delete("k");
    expect(received).toBeUndefined();
  });

  test("callback does not fire if value unchanged", () => {
    svc.set("k", "v");
    let callCount = 0;
    svc.onChange("k", () => { callCount++; });
    svc.set("k", "v"); // same value
    expect(callCount).toBe(0);
  });

  test("Disposable stops notifications", () => {
    let callCount = 0;
    const d = svc.onChange("k", () => { callCount++; });
    svc.set("k", "a");
    expect(callCount).toBe(1);
    d.dispose();
    svc.set("k", "b");
    expect(callCount).toBe(1);
  });

  test("multiple listeners on same key all fire", () => {
    let count1 = 0;
    let count2 = 0;
    svc.onChange("k", () => { count1++; });
    svc.onChange("k", () => { count2++; });
    svc.set("k", "v");
    expect(count1).toBe(1);
    expect(count2).toBe(1);
  });

  test("listener error is caught and logged", () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    svc.onChange("k", () => { throw new Error("boom"); });
    svc.set("k", "v");
    expect(spy).toHaveBeenCalled();
    const found = spy.mock.calls.some(
      (args) => typeof args[0] === "string" && args[0].includes("Context key"),
    );
    expect(found).toBe(true);
    spy.mockRestore();
  });
});

// ── evaluate ──────────────────────────────────────────────────────────────────

describe("ContextKeyService — evaluate", () => {
  test("undefined returns true", () => {
    expect(svc.evaluate(undefined)).toBe(true);
  });

  test("empty string returns true", () => {
    expect(svc.evaluate("")).toBe(true);
  });

  // -- Equality --

  test("key == 'value' matches", () => {
    svc.set("focusedView", "chat");
    expect(svc.evaluate("focusedView == 'chat'")).toBe(true);
  });

  test("key == 'value' does not match", () => {
    svc.set("focusedView", "files");
    expect(svc.evaluate("focusedView == 'chat'")).toBe(false);
  });

  test("key == '' with unset key defaults to empty string", () => {
    expect(svc.evaluate("focusedView == ''")).toBe(true);
  });

  // -- Inequality --

  test("key != 'value' matches", () => {
    svc.set("focusedView", "files");
    expect(svc.evaluate("focusedView != 'chat'")).toBe(true);
  });

  test("key != 'value' does not match", () => {
    svc.set("focusedView", "chat");
    expect(svc.evaluate("focusedView != 'chat'")).toBe(false);
  });

  // -- Bare key (truthy) --

  test("bare key truthy when set to non-empty string", () => {
    svc.set("mode", "dark");
    expect(svc.evaluate("mode")).toBe(true);
  });

  test("bare key truthy when set to true", () => {
    svc.set("isReady", true);
    expect(svc.evaluate("isReady")).toBe(true);
  });

  test("bare key falsy when unset", () => {
    expect(svc.evaluate("isReady")).toBe(false);
  });

  test("bare key falsy when set to false", () => {
    svc.set("isReady", false);
    expect(svc.evaluate("isReady")).toBe(false);
  });

  test("bare key falsy when set to empty string", () => {
    svc.set("mode", "");
    expect(svc.evaluate("mode")).toBe(false);
  });

  // -- Negated key (!key) --

  test("!key truthy when unset", () => {
    expect(svc.evaluate("!isReady")).toBe(true);
  });

  test("!key falsy when set to truthy value", () => {
    svc.set("isReady", true);
    expect(svc.evaluate("!isReady")).toBe(false);
  });

  test("!key truthy when set to false", () => {
    svc.set("isReady", false);
    expect(svc.evaluate("!isReady")).toBe(true);
  });

  // -- Unknown expression --

  test("unknown expression returns true (permissive)", () => {
    expect(svc.evaluate("something && other")).toBe(true);
  });

  // -- Overrides --

  test("overrides map takes precedence over store", () => {
    svc.set("focusedView", "files");
    const overrides = new Map([["focusedView", "chat" as const]]);
    expect(svc.evaluate("focusedView == 'chat'", overrides)).toBe(true);
    expect(svc.evaluate("focusedView == 'files'", overrides)).toBe(false);
  });

  test("overrides falls through to store for other keys", () => {
    svc.set("mode", "dark");
    const overrides = new Map([["focusedView", "chat" as const]]);
    expect(svc.evaluate("mode == 'dark'", overrides)).toBe(true);
  });

  test("overrides work with bare key evaluation", () => {
    const overrides = new Map([["isActive", true as const]]);
    expect(svc.evaluate("isActive", overrides)).toBe(true);
    expect(svc.evaluate("!isActive", overrides)).toBe(false);
  });
});

// ── dispose ───────────────────────────────────────────────────────────────────

describe("ContextKeyService — dispose", () => {
  test("dispose clears store and listeners", () => {
    svc.set("k", "v");
    svc.onChange("k", () => {});
    svc.dispose();
    expect(svc.get("k")).toBeUndefined();
    expect(svc.has("k")).toBe(false);
  });
});
