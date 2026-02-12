import { describe, test, expect, beforeEach } from "bun:test";
import {
  loadKeybindingOverrides,
  resolveKeybinding,
  setKeybindingOverride,
  removeKeybindingOverride,
  getKeybindingOverrides,
} from "../../../src/kernel/registries/keybindings.ts";

const STORAGE_KEY = "talu.keybindings";

beforeEach(() => {
  // loadKeybindingOverrides only overwrites the in-memory cache when
  // localStorage contains a parseable value, so seed an empty object first.
  localStorage.setItem(STORAGE_KEY, "{}");
  loadKeybindingOverrides();
  localStorage.removeItem(STORAGE_KEY);
});

describe("resolveKeybinding", () => {
  test("returns default when no override exists", () => {
    expect(resolveKeybinding("my.cmd", "ctrl+s")).toBe("ctrl+s");
  });

  test("returns undefined when no override and no default", () => {
    expect(resolveKeybinding("my.cmd")).toBeUndefined();
  });

  test("override takes precedence over default", () => {
    setKeybindingOverride("my.cmd", "ctrl+shift+s");
    expect(resolveKeybinding("my.cmd", "ctrl+s")).toBe("ctrl+shift+s");
  });
});

describe("setKeybindingOverride", () => {
  test("persists override to localStorage", () => {
    setKeybindingOverride("cmd.a", "ctrl+a");
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY)!);
    expect(stored["cmd.a"]).toBe("ctrl+a");
  });

  test("multiple overrides coexist", () => {
    setKeybindingOverride("cmd.a", "ctrl+a");
    setKeybindingOverride("cmd.b", "ctrl+b");
    expect(resolveKeybinding("cmd.a")).toBe("ctrl+a");
    expect(resolveKeybinding("cmd.b")).toBe("ctrl+b");
  });
});

describe("removeKeybindingOverride", () => {
  test("removes override and reverts to default", () => {
    setKeybindingOverride("cmd.a", "ctrl+a");
    removeKeybindingOverride("cmd.a");
    expect(resolveKeybinding("cmd.a", "default")).toBe("default");
  });

  test("clears localStorage when last override is removed", () => {
    setKeybindingOverride("cmd.a", "ctrl+a");
    removeKeybindingOverride("cmd.a");
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });
});

describe("getKeybindingOverrides", () => {
  test("returns empty object when no overrides", () => {
    expect(Object.keys(getKeybindingOverrides())).toHaveLength(0);
  });

  test("returns all current overrides", () => {
    setKeybindingOverride("cmd.a", "ctrl+a");
    setKeybindingOverride("cmd.b", "ctrl+b");
    const overrides = getKeybindingOverrides();
    expect(overrides["cmd.a"]).toBe("ctrl+a");
    expect(overrides["cmd.b"]).toBe("ctrl+b");
  });
});

describe("loadKeybindingOverrides", () => {
  test("loads overrides from localStorage", () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ "cmd.x": "alt+x" }));
    loadKeybindingOverrides();
    expect(resolveKeybinding("cmd.x")).toBe("alt+x");
  });

  test("handles corrupt localStorage gracefully", () => {
    localStorage.setItem(STORAGE_KEY, "not-json{{{");
    expect(() => loadKeybindingOverrides()).not.toThrow();
    expect(Object.keys(getKeybindingOverrides())).toHaveLength(0);
  });

  test("handles missing localStorage gracefully", () => {
    expect(() => loadKeybindingOverrides()).not.toThrow();
  });
});
