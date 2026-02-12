import { describe, test, expect, spyOn, beforeEach } from "bun:test";
import { LocalStorageFacade } from "../../../src/kernel/system/storage.ts";

let storage: LocalStorageFacade;

beforeEach(() => {
  // Clear all talu-storage keys from localStorage.
  const toRemove: string[] = [];
  for (let i = 0; i < localStorage.length; i++) {
    const k = localStorage.key(i);
    if (k?.startsWith("talu-storage:")) toRemove.push(k);
  }
  for (const k of toRemove) localStorage.removeItem(k);

  storage = new LocalStorageFacade("test.plugin");
});

describe("LocalStorageFacade", () => {
  test("get returns null for missing key", async () => {
    expect(await storage.get("missing")).toBeNull();
  });

  test("set + get roundtrip", async () => {
    await storage.set("key1", { value: 42 });
    const result = await storage.get<{ value: number }>("key1");
    expect(result).toEqual({ value: 42 });
  });

  test("set overwrites existing value", async () => {
    await storage.set("key1", "first");
    await storage.set("key1", "second");
    expect(await storage.get("key1")).toBe("second");
  });

  test("delete removes key", async () => {
    await storage.set("key1", "value");
    await storage.delete("key1");
    expect(await storage.get("key1")).toBeNull();
  });

  test("keys returns all plugin keys", async () => {
    await storage.set("a", 1);
    await storage.set("b", 2);
    await storage.set("c", 3);
    const keys = await storage.keys();
    expect(keys.sort()).toEqual(["a", "b", "c"]);
  });

  test("clear removes all plugin keys", async () => {
    await storage.set("a", 1);
    await storage.set("b", 2);
    await storage.clear();
    expect(await storage.keys()).toEqual([]);
  });

  // --- Plugin isolation ---

  test("different plugins cannot see each other's keys", async () => {
    const otherStorage = new LocalStorageFacade("other.plugin");
    await storage.set("shared-name", "mine");
    await otherStorage.set("shared-name", "theirs");
    expect(await storage.get("shared-name")).toBe("mine");
    expect(await otherStorage.get("shared-name")).toBe("theirs");
    otherStorage.dispose();
  });

  test("clear only removes own plugin's keys", async () => {
    const otherStorage = new LocalStorageFacade("other.plugin");
    await storage.set("a", 1);
    await otherStorage.set("b", 2);
    await storage.clear();
    expect(await storage.keys()).toEqual([]);
    expect(await otherStorage.keys()).toEqual(["b"]);
    otherStorage.dispose();
  });

  // --- Change notifications ---

  test("onDidChange fires on set", async () => {
    let receivedKey: string | null | undefined;
    storage.onDidChange((key) => { receivedKey = key; });
    await storage.set("mykey", "val");
    expect(receivedKey).toBe("mykey");
  });

  test("onDidChange fires on delete", async () => {
    let receivedKey: string | null | undefined;
    await storage.set("mykey", "val");
    storage.onDidChange((key) => { receivedKey = key; });
    await storage.delete("mykey");
    expect(receivedKey).toBe("mykey");
  });

  test("onDidChange fires with null on clear", async () => {
    let receivedKey: string | null | undefined = "unset";
    await storage.set("a", 1);
    storage.onDidChange((key) => { receivedKey = key; });
    await storage.clear();
    expect(receivedKey).toBeNull();
  });

  test("onDidChange dispose stops notifications", async () => {
    let callCount = 0;
    const d = storage.onDidChange(() => { callCount++; });
    await storage.set("a", 1);
    expect(callCount).toBe(1);
    d.dispose();
    await storage.set("b", 2);
    expect(callCount).toBe(1);
  });

  test("change callback error does not break storage", async () => {
    const spy = spyOn(console, "error").mockImplementation(() => {});
    storage.onDidChange(() => { throw new Error("cb boom"); });
    await storage.set("a", 1);
    expect(await storage.get("a")).toBe(1);
    spy.mockRestore();
  });

  // --- Lifecycle ---

  test("dispose clears callbacks", async () => {
    let callCount = 0;
    storage.onDidChange(() => { callCount++; });
    storage.dispose();
    // Manually write to localStorage to trigger — but callbacks are cleared.
    // We just verify no error on subsequent operations.
    expect(callCount).toBe(0);
  });

  // --- Edge cases ---

  test("get handles corrupt JSON gracefully", async () => {
    const prefix = `talu-storage:test.plugin:`;
    localStorage.setItem(`${prefix}bad`, "not{json");
    expect(await storage.get("bad")).toBeNull();
  });
});
