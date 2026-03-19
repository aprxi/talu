import { describe, test, expect, beforeEach, afterEach, spyOn } from "bun:test";

type KvSettingsModule = typeof import("../../../src/kernel/system/kv-settings.ts");

async function loadKvSettings(): Promise<KvSettingsModule> {
  return import(`../../../src/kernel/system/kv-settings.ts?test=${Math.random()}`);
}

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  localStorage.clear();
});

describe("kv-settings", () => {
  test("getSetting falls back to localStorage when API is not initialized", async () => {
    const kv = await loadKvSettings();
    localStorage.setItem("theme", "dark");

    expect(await kv.getSetting("theme")).toBe("dark");
  });

  test("getSetting reads from the KV API namespace when initialized", async () => {
    const kv = await loadKvSettings();
    const apiCalls: unknown[][] = [];
    kv.initKvSettings({
      kvGet: async (...args: unknown[]) => {
        apiCalls.push(args);
        return { ok: true, data: { value: "from-kv" } };
      },
      kvPut: async () => ({ ok: true }),
      kvDelete: async () => ({ ok: true }),
      kvList: async () => ({ ok: true, data: { items: [] } }),
    });

    expect(await kv.getSetting("theme")).toBe("from-kv");
    expect(apiCalls).toEqual([["ui", "theme"]]);
  });

  test("getSetting falls back to localStorage when KV read throws", async () => {
    const kv = await loadKvSettings();
    localStorage.setItem("theme", "local");
    kv.initKvSettings({
      kvGet: async () => {
        throw new Error("backend offline");
      },
      kvPut: async () => ({ ok: true }),
      kvDelete: async () => ({ ok: true }),
      kvList: async () => ({ ok: true, data: { items: [] } }),
    });

    expect(await kv.getSetting("theme")).toBe("local");
  });

  test("setSetting and deleteSetting use localStorage when API is not initialized", async () => {
    const kv = await loadKvSettings();

    await kv.setSetting("sidebar", "collapsed");
    expect(localStorage.getItem("sidebar")).toBe("collapsed");

    await kv.deleteSetting("sidebar");
    expect(localStorage.getItem("sidebar")).toBeNull();
  });

  test("migrateLocalStorageToKv migrates known UI keys and plugin keys", async () => {
    const kv = await loadKvSettings();
    const puts: { namespace: string; key: string; value: string }[] = [];
    kv.initKvSettings({
      kvGet: async () => ({ ok: true, data: { value: null } }),
      kvPut: async (namespace: string, key: string, value: string) => {
        puts.push({ namespace, key, value });
        return { ok: true };
      },
      kvDelete: async () => ({ ok: true }),
      kvList: async () => ({ ok: true, data: { items: [] } }),
    });

    localStorage.setItem("talu-last-active-mode", "chat");
    localStorage.setItem("talu.keybindings", '{"chat.send":"Enter"}');
    localStorage.setItem("talu-storage:talu.chat:draft", "hello");
    localStorage.setItem("talu-storage:talu.repo:selected", "model-a");

    await kv.migrateLocalStorageToKv();

    expect(puts).toEqual([
      { namespace: "ui", key: "talu-last-active-mode", value: "chat" },
      { namespace: "ui", key: "talu.keybindings", value: '{"chat.send":"Enter"}' },
      { namespace: "plugin:talu.chat", key: "draft", value: "hello" },
      { namespace: "plugin:talu.repo", key: "selected", value: "model-a" },
    ]);
    expect(localStorage.getItem("talu-last-active-mode")).toBeNull();
    expect(localStorage.getItem("talu.keybindings")).toBeNull();
    expect(localStorage.getItem("talu-storage:talu.chat:draft")).toBeNull();
    expect(localStorage.getItem("talu-storage:talu.repo:selected")).toBeNull();
    expect(localStorage.getItem("talu-kv-migrated")).toBe("1");
  });

  test("migrateLocalStorageToKv is idempotent once the migration flag is set", async () => {
    const kv = await loadKvSettings();
    const kvPut = spyOn(
      {
        async put(_namespace: string, _key: string, _value: string) {
          return { ok: true };
        },
      },
      "put",
    );
    kv.initKvSettings({
      kvGet: async () => ({ ok: true, data: { value: null } }),
      kvPut: kvPut,
      kvDelete: async () => ({ ok: true }),
      kvList: async () => ({ ok: true, data: { items: [] } }),
    });
    localStorage.setItem("talu-kv-migrated", "1");
    localStorage.setItem("talu-last-active-mode", "chat");

    await kv.migrateLocalStorageToKv();

    expect(kvPut).not.toHaveBeenCalled();
    expect(localStorage.getItem("talu-last-active-mode")).toBe("chat");
  });

  test("migrateLocalStorageToKv continues after a per-key KV failure", async () => {
    const kv = await loadKvSettings();
    const puts: { namespace: string; key: string; value: string }[] = [];
    kv.initKvSettings({
      kvGet: async () => ({ ok: true, data: { value: null } }),
      kvPut: async (namespace: string, key: string, value: string) => {
        if (key === "talu-last-active-mode") throw new Error("write failed");
        puts.push({ namespace, key, value });
        return { ok: true };
      },
      kvDelete: async () => ({ ok: true }),
      kvList: async () => ({ ok: true, data: { items: [] } }),
    });
    localStorage.setItem("talu-last-active-mode", "chat");
    localStorage.setItem("talu.keybindings", '{"chat.send":"Enter"}');

    await kv.migrateLocalStorageToKv();

    expect(puts).toEqual([
      { namespace: "ui", key: "talu.keybindings", value: '{"chat.send":"Enter"}' },
    ]);
    expect(localStorage.getItem("talu-last-active-mode")).toBe("chat");
    expect(localStorage.getItem("talu.keybindings")).toBeNull();
    expect(localStorage.getItem("talu-kv-migrated")).toBe("1");
  });
});
