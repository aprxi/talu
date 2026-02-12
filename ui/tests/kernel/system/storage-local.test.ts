import { describe, test, expect, spyOn, beforeEach, afterEach } from "bun:test";
import { LocalStorageFacade } from "../../../src/kernel/system/storage.ts";

/**
 * Tests for LocalStorageFacade — localStorage-backed storage for built-in plugins.
 * Verifies key prefixing, JSON round-trip, keys enumeration, clear, change
 * notifications, and dispose lifecycle.
 */

let storage: LocalStorageFacade;

beforeEach(() => {
  localStorage.clear();
  storage = new LocalStorageFacade("builtin.test");
});

afterEach(() => {
  storage.dispose();
  localStorage.clear();
});

// ── Key prefixing ──────────────────────────────────────────────────────────

describe("LocalStorageFacade — key prefixing", () => {
  test("set writes with namespaced key", async () => {
    await storage.set("mykey", "val");
    expect(localStorage.getItem("talu-storage:builtin.test:mykey")).toBe('"val"');
  });

  test("plugins have isolated namespaces", async () => {
    const other = new LocalStorageFacade("builtin.other");
    await storage.set("k", "a");
    await other.set("k", "b");
    expect(await storage.get("k")).toBe("a");
    expect(await other.get("k")).toBe("b");
    other.dispose();
  });
});

// ── JSON round-trip ────────────────────────────────────────────────────────

describe("LocalStorageFacade — get/set round-trip", () => {
  test("string value round-trips", async () => {
    await storage.set("s", "hello");
    expect(await storage.get("s")).toBe("hello");
  });

  test("object value round-trips", async () => {
    await storage.set("obj", { a: 1, b: [2, 3] });
    expect(await storage.get("obj")).toEqual({ a: 1, b: [2, 3] });
  });

  test("number value round-trips", async () => {
    await storage.set("n", 42);
    expect(await storage.get("n")).toBe(42);
  });

  test("boolean value round-trips", async () => {
    await storage.set("b", true);
    expect(await storage.get("b")).toBe(true);
  });

  test("null value round-trips", async () => {
    await storage.set("nil", null);
    expect(await storage.get("nil")).toBeNull();
  });

  test("returns null for missing key", async () => {
    expect(await storage.get("missing")).toBeNull();
  });

  test("returns null when localStorage contains invalid JSON", async () => {
    localStorage.setItem("talu-storage:builtin.test:bad", "not-json{");
    expect(await storage.get("bad")).toBeNull();
  });
});

// ── DELETE ──────────────────────────────────────────────────────────────────

describe("LocalStorageFacade — delete", () => {
  test("removes key from localStorage", async () => {
    await storage.set("k", "v");
    await storage.delete("k");
    expect(localStorage.getItem("talu-storage:builtin.test:k")).toBeNull();
    expect(await storage.get("k")).toBeNull();
  });

  test("delete on missing key is a no-op", async () => {
    await storage.delete("nonexistent");
    // Should not throw.
  });
});

// ── KEYS ───────────────────────────────────────────────────────────────────

describe("LocalStorageFacade — keys", () => {
  test("returns all keys for this plugin", async () => {
    await storage.set("a", 1);
    await storage.set("b", 2);
    await storage.set("c", 3);
    const keys = await storage.keys();
    expect(keys.sort()).toEqual(["a", "b", "c"]);
  });

  test("does not include keys from other plugins", async () => {
    const other = new LocalStorageFacade("builtin.other");
    await other.set("foreign", "val");
    await storage.set("mine", "val");
    const keys = await storage.keys();
    expect(keys).toEqual(["mine"]);
    other.dispose();
  });

  test("returns empty array when no keys stored", async () => {
    const keys = await storage.keys();
    expect(keys).toEqual([]);
  });
});

// ── CLEAR ──────────────────────────────────────────────────────────────────

describe("LocalStorageFacade — clear", () => {
  test("removes all keys for this plugin", async () => {
    await storage.set("a", 1);
    await storage.set("b", 2);
    await storage.clear();
    expect(await storage.keys()).toEqual([]);
    expect(await storage.get("a")).toBeNull();
  });

  test("does not remove keys from other plugins", async () => {
    const other = new LocalStorageFacade("builtin.other");
    await other.set("keep", "val");
    await storage.set("remove", "val");
    await storage.clear();
    expect(await other.get("keep")).toBe("val");
    other.dispose();
  });
});

// ── Change notifications ────────────────────────────────────────────────────

describe("LocalStorageFacade — change notifications", () => {
  test("set notifies onDidChange with key", async () => {
    let received: string | null | undefined;
    storage.onDidChange((key) => { received = key; });
    await storage.set("k", "v");
    expect(received).toBe("k");
  });

  test("delete notifies onDidChange with key", async () => {
    await storage.set("k", "v");
    let received: string | null | undefined;
    storage.onDidChange((key) => { received = key; });
    await storage.delete("k");
    expect(received).toBe("k");
  });

  test("clear notifies onDidChange with null", async () => {
    let received: string | null | undefined = "unset";
    storage.onDidChange((key) => { received = key; });
    await storage.clear();
    expect(received).toBeNull();
  });

  test("onDidChange dispose stops notifications", async () => {
    let count = 0;
    const d = storage.onDidChange(() => { count++; });
    await storage.set("a", 1);
    expect(count).toBe(1);
    d.dispose();
    await storage.set("b", 2);
    expect(count).toBe(1);
  });

  test("callback error does not break storage", async () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    storage.onDidChange(() => { throw new Error("boom"); });
    await storage.set("a", 1);
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });
});

// ── Lifecycle ──────────────────────────────────────────────────────────────

describe("LocalStorageFacade — lifecycle", () => {
  test("dispose clears callbacks", async () => {
    let count = 0;
    storage.onDidChange(() => { count++; });
    storage.dispose();
    // Re-create to set a value (original storage is disposed).
    const fresh = new LocalStorageFacade("builtin.test");
    await fresh.set("x", 1);
    fresh.dispose();
    expect(count).toBe(0);
  });
});
